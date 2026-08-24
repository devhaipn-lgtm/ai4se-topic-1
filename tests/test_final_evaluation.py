import json
from pathlib import Path

import pandas as pd
import pytest

from issue_classifier import (
    ALLOWED_REPOSITORIES,
    CsvIssueStore,
    InMemoryIssueStore,
    run_final_chosen_evaluation,
)
from issue_classifier.final_evaluation import (
    _metrics,
    _publish_complete_directories,
)


REPOSITORIES = sorted(ALLOWED_REPOSITORIES)
LABELS = ("bug", "feature", "question")


class SpyStore:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.frames = {"train": train, "test": test}
        self.calls = []

    def load(self, split: str) -> pd.DataFrame:
        self.calls.append(split)
        return self.frames[split].copy(deep=True)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []
    for repo_index, repository in enumerate(REPOSITORIES):
        for label_index, label in enumerate(LABELS):
            for number in range(3):
                title = f"{repository} {label} {number}"
                train_rows.append({
                    "repo": repository,
                    "created_at": f"2024-01-{repo_index + 1:02d}",
                    "label": label,
                    "title": title,
                    "body": f"body {label} {number}",
                })
                test_rows.append({
                    "repo": repository,
                    "created_at": f"2024-02-{repo_index + 1:02d}",
                    "label": label,
                    "title": f"new {repository} {label} {number}",
                    "body": f"new body {label} {number}",
                })
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def _selection(path: Path) -> None:
    path.write_text(json.dumps({"selected_candidate": "dummy_most_frequent"}), encoding="utf-8")


def test_metrics_use_fixed_labels_and_hand_calculated_values() -> None:
    result = _metrics(
        ["bug", "bug", "feature", "question"],
        ["bug", "feature", "feature", "question"],
    )

    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert list(result["per_class"]) == list(LABELS)
    assert result["per_class"]["bug"]["support"] == 2
    assert result["macro"]["f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert result["micro"]["f1"] == pytest.approx(3 / 4)


def test_final_run_is_train_then_test_once_and_writes_artifacts(tmp_path) -> None:
    train, test = _frames()
    selection = tmp_path / "cross_validation.json"
    _selection(selection)
    store = SpyStore(train, test)

    result = run_final_chosen_evaluation(
        store,
        selection_path=selection,
        output_root=tmp_path / "results",
        model_root=tmp_path / "models",
        backend="memory",
    )

    assert store.calls == ["train", "test"]
    assert len(result["model_files"]) == 5
    assert len(result["predictions"]["primary"]) == len(test)
    assert result["metrics"]["policies"]["primary"]["repositories"][REPOSITORIES[0]]["confusion_matrix"]
    assert set(result["metrics"]["policies"]["primary"]["global"]["per_class"]) == set(LABELS)
    assert result["manifest"]["runtime_seconds"] >= 0
    assert result["manifest"]["package_versions"]["matplotlib"]
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 5
    assert json.loads((tmp_path / "results/metrics.json").read_text()) == result["metrics"]
    assert pd.read_csv(tmp_path / "results/predictions.csv").shape[0] == len(test) * 2
    assert len(list((tmp_path / "results/figures").glob("*.png"))) == 10


def test_repeated_run_replaces_stale_models(tmp_path) -> None:
    train, test = _frames()
    selection = tmp_path / "cross_validation.json"
    _selection(selection)
    output = tmp_path / "results"
    models = tmp_path / "models"
    run_final_chosen_evaluation(SpyStore(train, test), selection, output, models, "memory")
    (models / "stale.joblib").write_bytes(b"stale")
    run_final_chosen_evaluation(SpyStore(train, test), selection, output, models, "memory")
    assert not (models / "stale.joblib").exists()
    assert (output / "manifest.json").is_file()


def test_file_and_memory_backends_have_same_predictions_and_metrics(tmp_path) -> None:
    file_store = CsvIssueStore(Path("official_nlbse24/data"))
    train = file_store.load("train")
    test = file_store.load("test")
    memory_store = InMemoryIssueStore(train, test)
    selection = tmp_path / "cross_validation.json"
    _selection(selection)
    first = run_final_chosen_evaluation(
        CsvIssueStore(Path("official_nlbse24/data")), selection, tmp_path / "file", tmp_path / "file-models", "file"
    )
    second = run_final_chosen_evaluation(
        memory_store, selection, tmp_path / "memory", tmp_path / "memory-models", "memory"
    )
    assert first["metrics"] == second["metrics"]
    assert first["predictions"] == second["predictions"]


def test_metrics_csv_contains_all_global_entries_for_both_policies(tmp_path) -> None:
    train, test = _frames()
    selection = tmp_path / "cross_validation.json"
    _selection(selection)
    run_final_chosen_evaluation(
        SpyStore(train, test), selection, tmp_path / "results", tmp_path / "models", "memory"
    )

    metrics = pd.read_csv(tmp_path / "results/metrics.csv")
    global_rows = metrics[metrics["repository"] == "__global__"]
    assert set(global_rows["policy"]) == {"primary", "leakage_sensitive"}
    assert set(global_rows["metric"]) == {"class", "macro", "micro", "confusion"}
    for policy in ("primary", "leakage_sensitive"):
        policy_rows = global_rows[global_rows["policy"] == policy]
        assert len(policy_rows[policy_rows["metric"] == "class"]) == 3
        assert len(policy_rows[policy_rows["metric"] == "confusion"]) == 9


def test_publish_rollback_restores_both_previous_directories(tmp_path, monkeypatch) -> None:
    output = tmp_path / "results"
    models = tmp_path / "models"
    staged_results = tmp_path / "staged-results"
    staged_models = tmp_path / "staged-models"
    output.mkdir()
    models.mkdir()
    staged_results.mkdir()
    staged_models.mkdir()
    (output / "manifest.json").write_text("previous-result", encoding="utf-8")
    (models / "marker.joblib").write_text("previous-model", encoding="utf-8")
    (staged_results / "manifest.json").write_text("new-result", encoding="utf-8")
    (staged_models / "new.joblib").write_text("new-model", encoding="utf-8")

    original_rename = Path.rename

    def fail_second_rename(self, target):
        if self == staged_models:
            raise OSError("simulated second publish failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_second_rename)
    with pytest.raises(OSError, match="simulated second publish failure"):
        _publish_complete_directories(staged_results, output, staged_models, models)

    assert (output / "manifest.json").read_text(encoding="utf-8") == "previous-result"
    assert (models / "marker.joblib").read_text(encoding="utf-8") == "previous-model"
    assert not (output / "new.joblib").exists()
    assert not (models / "new.joblib").exists()


def test_missing_or_malformed_selection_fails_before_store_access(tmp_path) -> None:
    train, test = _frames()
    store = SpyStore(train, test)
    with pytest.raises(FileNotFoundError, match="Missing selection artifact"):
        run_final_chosen_evaluation(store, tmp_path / "missing.json", tmp_path / "out")
    assert store.calls == []
    malformed = tmp_path / "bad.json"
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed selection artifact"):
        run_final_chosen_evaluation(store, malformed, tmp_path / "out")
    assert store.calls == []


def test_committed_official_artifact_has_both_prediction_policies() -> None:
    predictions_path = Path("results/chosen-model-seed42/predictions.csv")
    metrics_path = Path("results/chosen-model-seed42/metrics.json")
    if not predictions_path.is_file() or not metrics_path.is_file():
        pytest.fail("Committed official final artifacts are missing")
    predictions = pd.read_csv(predictions_path)
    assert predictions.groupby("policy").size().to_dict() == {
        "leakage_sensitive": 1497,
        "primary": 1500,
    }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["policies"]["primary"]["global"]["macro_f1"] == pytest.approx(
        sum(
            value["macro"]["f1"]
            for value in metrics["policies"]["primary"]["repositories"].values()
        )
        / 5
    )
