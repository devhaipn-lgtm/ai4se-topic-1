import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from issue_classifier import ALLOWED_REPOSITORIES
from issue_classifier import cli
from issue_classifier.model_selection import candidate_specs
from issue_classifier.workflow import train_chosen_models


REPOSITORIES = sorted(ALLOWED_REPOSITORIES)
LABELS = ("bug", "feature", "question")


class SpyStore:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.frames = {"train": train, "test": test}
        self.calls: list[str] = []

    def load(self, split: str) -> pd.DataFrame:
        self.calls.append(split)
        return self.frames[split].copy(deep=True)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = []
    test = []
    for repo_index, repository in enumerate(REPOSITORIES):
        for label in LABELS:
            for number in range(3):
                train.append({"repo": repository, "created_at": str(repo_index), "label": label, "title": f"{repository} {label} {number}", "body": f"body {label} {number}"})
                test.append({"repo": repository, "created_at": str(repo_index), "label": label, "title": f"test {repository} {label} {number}", "body": f"test body {label} {number}"})
    return pd.DataFrame(train), pd.DataFrame(test)


def _selection(path: Path) -> None:
    path.write_text(json.dumps({"selected_candidate": "dummy_most_frequent"}), encoding="utf-8")


def test_parser_help_and_command_choices() -> None:
    parser = cli.build_parser()
    for command, model in (
        ("validate-data", None),
        ("cross-validate", "all"),
        ("train", "selected"),
        ("evaluate", "selected"),
    ):
        command_args = [] if model is None else ["--model", model]
        if command == "validate-data":
            command_args = ["--backend", "file"]
        assert parser.parse_args([command] + command_args)
    with pytest.raises(SystemExit):
        parser.parse_args(["validate-data", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        parser.parse_args(["cross-validate", "--model", "test"])


def test_validate_file_and_memory_outputs_agree(capsys) -> None:
    base = {"data_dir": Path("official_nlbse24/data")}
    assert cli._validate_data(Namespace(backend="file", **base)) == 0
    file_payload = json.loads(capsys.readouterr().out)
    assert cli._validate_data(Namespace(backend="memory", **base)) == 0
    memory_payload = json.loads(capsys.readouterr().out)
    file_payload.pop("backend")
    memory_payload.pop("backend")
    assert file_payload == memory_payload


def test_cross_validate_dispatch_is_train_only(monkeypatch, tmp_path, capsys) -> None:
    train, test = _frames()
    store = SpyStore(train, test)
    monkeypatch.setattr(cli, "_store", lambda backend, data_dir, required_splits: store)
    monkeypatch.setattr(cli, "run_cross_validation", lambda store, output: {"selected_candidate": "dummy"})
    args = Namespace(backend="file", data_dir=tmp_path, output_dir=tmp_path, model="all")

    assert cli._cross_validate(args) == 0
    assert store.calls == []
    assert json.loads(capsys.readouterr().out)["selected_candidate"] == "dummy"


def test_train_dispatch_loads_train_only_and_writes_manifest(monkeypatch, tmp_path, capsys) -> None:
    train, test = _frames()
    store = SpyStore(train, test)
    selection = tmp_path / "selection.json"
    _selection(selection)
    monkeypatch.setattr(cli, "_store", lambda backend, data_dir, required_splits: store)
    args = Namespace(
        backend="memory",
        data_dir=tmp_path,
        model="selected",
        selection=selection,
        model_dir=tmp_path / "models",
        training_manifest=tmp_path / "training_manifest.json",
    )

    assert cli._train(args) == 0
    assert store.calls == ["train"]
    assert (tmp_path / "training_manifest.json").is_file()
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 5
    assert json.loads(capsys.readouterr().out)["selected_candidate"] == "dummy_most_frequent"


def test_evaluate_validates_manifest_then_loads_test_without_fitting(monkeypatch, tmp_path) -> None:
    train, test = _frames()
    selection = tmp_path / "selection.json"
    _selection(selection)
    train_store = SpyStore(train, test)
    trained = train_chosen_models(
        train_store, selection, tmp_path / "models", tmp_path / "training_manifest.json", "memory"
    )
    test_store = SpyStore(train, test)
    monkeypatch.setattr(cli, "_store", lambda backend, data_dir, required_splits: test_store)
    args = Namespace(
        backend="memory",
        data_dir=tmp_path,
        model="selected",
        selection=selection,
        model_dir=Path(trained["model_root"]),
        training_manifest=Path(trained["manifest_path"]),
        output_dir=tmp_path / "results",
    )

    assert cli._evaluate(args) == 0
    assert test_store.calls == ["test"]


def test_subprocess_help_from_workspace_root() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-m", "issue_classifier", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0
    assert "validate-data" in completed.stdout


def test_notebooks_are_thin_valid_json() -> None:
    for notebook_name in ("01_eda.ipynb", "02_chosen_model.ipynb", "04_results.ipynb"):
        notebook = json.loads((Path("notebooks") / notebook_name).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "sys.path.insert" in source
        assert "TfidfVectorizer" not in source
        assert "LinearSVC" not in source
        assert "precision_recall_fscore_support" not in source
        assert all(not cell.get("outputs") for cell in notebook["cells"])
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(isinstance(cell_id, str) and cell_id for cell_id in ids)
        assert len(ids) == len(set(ids))
    eda = (Path("notebooks") / "01_eda.ipynb").read_text(encoding="utf-8")
    assert 'CsvIssueStore(PROJECT_ROOT / \\"official_nlbse24\\" / \\"data\\")' in eda
    assert 'CsvIssueStore(Path.cwd() / \\"data\\")' not in eda


def test_memory_store_construction_reads_only_requested_splits(monkeypatch, tmp_path) -> None:
    train, test = _frames()
    loaded: list[str] = []

    class RecordingCsvStore:
        def __init__(self, data_dir):
            self.data_dir = data_dir

        def load(self, split):
            loaded.append(split)
            return train if split == "train" else test

    class RecordingMemoryStore:
        def __init__(self, *, train=None, test=None):
            self.train = train
            self.test = test

    monkeypatch.setattr(cli, "CsvIssueStore", RecordingCsvStore)
    monkeypatch.setattr(cli, "InMemoryIssueStore", RecordingMemoryStore)

    train_store = cli._store("memory", tmp_path, ("train",))
    assert loaded == ["train"]
    assert train_store.train is train
    assert train_store.test is None

    loaded.clear()
    test_store = cli._store("memory", tmp_path, ("test",))
    assert loaded == ["test"]
    assert test_store.train is None
    assert test_store.test is test


def test_train_handler_validates_selection_before_memory_store(monkeypatch, tmp_path, capsys) -> None:
    events: list[str] = []
    selection = tmp_path / "selection.json"
    _selection(selection)
    spec = candidate_specs()[0]
    monkeypatch.setattr(cli, "validate_selection_artifact", lambda path: (events.append("selection_validated"), (spec, {"selected_candidate": spec.name}))[1])
    monkeypatch.setattr(cli, "_store", lambda *args: (events.append("store_constructed"), object())[1])
    monkeypatch.setattr(cli, "train_chosen_models", lambda *args, **kwargs: (events.append("train_called"), {"manifest": {"selected_candidate": {"name": spec.name}}, "model_root": "models", "manifest_path": "manifest.json"})[1])
    args = Namespace(backend="memory", data_dir=tmp_path, model="selected", selection=selection, model_dir=tmp_path / "models", training_manifest=tmp_path / "manifest.json")

    assert cli._train(args) == 0
    assert events == ["selection_validated", "store_constructed", "train_called"]
    capsys.readouterr()


def test_evaluate_handler_verifies_before_memory_store(monkeypatch, tmp_path, capsys) -> None:
    events: list[str] = []
    selection = tmp_path / "selection.json"
    _selection(selection)
    spec = candidate_specs()[0]
    bundle = {"selected_spec": spec, "manifest": {}, "manifest_path": tmp_path / "manifest.json", "model_root": tmp_path / "models"}
    monkeypatch.setattr(cli, "verify_training_artifacts", lambda *args: (events.extend(["manifest_verified", "model_hashes_verified"]), bundle)[1])
    monkeypatch.setattr(cli, "_store", lambda *args: (events.append("store_constructed"), object())[1])
    monkeypatch.setattr(cli, "evaluate_chosen_models", lambda *args, **kwargs: (events.append("test_read"), {"manifest": {"selected_candidate": {"name": spec.name}}, "metrics": {}})[1])
    args = Namespace(backend="memory", data_dir=tmp_path, model="selected", selection=selection, model_dir=tmp_path / "models", training_manifest=tmp_path / "manifest.json", output_dir=tmp_path / "results")

    assert cli._evaluate(args) == 0
    assert events == ["manifest_verified", "model_hashes_verified", "store_constructed", "test_read"]
    capsys.readouterr()
