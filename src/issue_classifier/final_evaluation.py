"""Final train-once evaluation for the chosen model."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .dataset_inspection import inspect_dataset
from .constants import ALLOWED_REPOSITORIES, REQUIRED_COLUMNS
from .model_selection import CLASS_LABELS, CandidateSpec, candidate_specs
from .stores import IssueStore

SEED = 42
RUN_ID = "chosen-model-seed42"


def _load_selected_spec(selection_path: Path) -> tuple[CandidateSpec, dict[str, Any]]:
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing selection artifact: {selection_path}")
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed selection artifact: {selection_path}") from exc
    selected_name = payload.get("selected_candidate") if isinstance(payload, dict) else None
    if not isinstance(selected_name, str):
        raise ValueError(
            f"Malformed selection artifact: {selection_path}; missing selected_candidate"
        )
    specs = {spec.name: spec for spec in candidate_specs()}
    if selected_name not in specs:
        raise ValueError(f"Selection artifact names unknown candidate: {selected_name!r}")
    return specs[selected_name], payload


def _json_safe(value: object) -> object:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        item = value.item()
        if item is not value:
            return _json_safe(item)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_frame_hash(frame: pd.DataFrame) -> str:
    records = []
    for row in frame.loc[:, REQUIRED_COLUMNS].itertuples(index=False, name=None):
        records.append([_json_safe(value) for value in row])
    payload = json.dumps(
        {"columns": list(REQUIRED_COLUMNS), "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlap_details(train: pd.DataFrame, test: pd.DataFrame) -> tuple[set[str], list[dict[str, Any]]]:
    inspection = inspect_dataset(train, test)
    records = inspection["exact_train_test_overlap"]["records"]
    hashes = {record["text_hash"] for record in records}
    return hashes, records


def _metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_LABELS), zero_division=0
    )
    class_metrics = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(CLASS_LABELS)
    }
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_LABELS), average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_LABELS), average="micro", zero_division=0
    )
    return {
        "row_count": len(y_true),
        "per_class": class_metrics,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f),
            "support": len(y_true),
        },
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f),
            "support": len(y_true),
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(CLASS_LABELS)
        ).astype(int).tolist(),
    }


def _policy_metrics(
    predictions: list[dict[str, Any]],
    excluded_count: int,
) -> dict[str, Any]:
    repositories = sorted(ALLOWED_REPOSITORIES)
    per_repository: dict[str, Any] = {}
    all_true: list[str] = []
    all_pred: list[str] = []
    for repository in repositories:
        rows = [row for row in predictions if row["repo"] == repository]
        y_true = [row["true_label"] for row in rows]
        y_pred = [row["predicted_label"] for row in rows]
        per_repository[repository] = _metrics(y_true, y_pred)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    aggregate = _metrics(all_true, all_pred)
    repo_macro_f1 = [per_repository[repo]["macro"]["f1"] for repo in repositories]
    global_per_class = {
        label: {
            "precision": float(sum(per_repository[repo]["per_class"][label]["precision"] for repo in repositories) / 5),
            "recall": float(sum(per_repository[repo]["per_class"][label]["recall"] for repo in repositories) / 5),
            "f1": float(sum(per_repository[repo]["per_class"][label]["f1"] for repo in repositories) / 5),
            "support": sum(per_repository[repo]["per_class"][label]["support"] for repo in repositories),
        }
        for label in CLASS_LABELS
    }
    global_metrics = {
        "per_class": global_per_class,
        "macro": {
            "precision": float(sum(per_repository[repo]["macro"]["precision"] for repo in repositories) / 5),
            "recall": float(sum(per_repository[repo]["macro"]["recall"] for repo in repositories) / 5),
            "f1": float(sum(repo_macro_f1) / 5),
            "support": sum(per_repository[repo]["macro"]["support"] for repo in repositories),
        },
        "micro": aggregate["micro"],
        "macro_f1": float(sum(repo_macro_f1) / 5),
        "aggregate_confusion_matrix": aggregate["confusion_matrix"],
    }
    return {
        "excluded_count": excluded_count,
        "included_count": len(predictions),
        "repositories": per_repository,
        "global": global_metrics,
        "aggregate_confusion_matrix": aggregate["confusion_matrix"],
    }


def _metric_rows(policy: str, policy_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repository, metrics in policy_result["repositories"].items():
        for label, values in metrics["per_class"].items():
            rows.append({"policy": policy, "repository": repository, "metric": "class", "label": label, "precision": values["precision"], "recall": values["recall"], "f1": values["f1"], "support": values["support"], "true_label": "", "predicted_label": "", "value": ""})
        for average in ("macro", "micro"):
            values = metrics[average]
            rows.append({"policy": policy, "repository": repository, "metric": average, "label": "", "precision": values["precision"], "recall": values["recall"], "f1": values["f1"], "support": values["support"], "true_label": "", "predicted_label": "", "value": ""})
        for row_index, true_label in enumerate(CLASS_LABELS):
            for column_index, predicted_label in enumerate(CLASS_LABELS):
                rows.append({"policy": policy, "repository": repository, "metric": "confusion", "label": "", "precision": "", "recall": "", "f1": "", "support": "", "true_label": true_label, "predicted_label": predicted_label, "value": metrics["confusion_matrix"][row_index][column_index]})
    global_values = policy_result["global"]
    for label, values in global_values["per_class"].items():
        rows.append({"policy": policy, "repository": "__global__", "metric": "class", "label": label, "precision": values["precision"], "recall": values["recall"], "f1": values["f1"], "support": values["support"], "true_label": "", "predicted_label": "", "value": ""})
    for average in ("macro", "micro"):
        values = global_values[average]
        rows.append({"policy": policy, "repository": "__global__", "metric": average, "label": "", "precision": values["precision"], "recall": values["recall"], "f1": values["f1"], "support": values["support"], "true_label": "", "predicted_label": "", "value": ""})
    for row_index, true_label in enumerate(CLASS_LABELS):
        for column_index, predicted_label in enumerate(CLASS_LABELS):
            rows.append({"policy": policy, "repository": "__global__", "metric": "confusion", "label": "", "precision": "", "recall": "", "f1": "", "support": "", "true_label": true_label, "predicted_label": predicted_label, "value": global_values["aggregate_confusion_matrix"][row_index][column_index]})
    return rows


def _write_png(path: Path, matrix: list[list[int]]) -> None:
    """Write a labeled confusion-matrix heatmap using the declared final extra."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Confusion-matrix figures require matplotlib; install with "
            'python -m pip install -e ".[final]"'
        ) from exc
    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(3), CLASS_LABELS, rotation=45, ha="right")
    axis.set_yticks(range(3), CLASS_LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion matrix")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["policy", "repository", "metric", "label", "precision", "recall", "f1", "support", "true_label", "predicted_label", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_final_chosen_evaluation(
    store: IssueStore,
    selection_path: str | Path = "results/cross_validation.json",
    output_root: str | Path = "results/chosen-model-seed42",
    model_root: str | Path = "artifacts/chosen-model-seed42/models",
    backend: str = "file",
) -> dict[str, Any]:
    """Train the selected candidate once per repository and write final artifacts."""
    from .workflow import run_final_chosen_evaluation as run_workflow

    return run_workflow(
        store,
        selection_path=selection_path,
        output_root=output_root,
        model_root=model_root,
        backend=backend,
    )

def _publish_complete_directories(
    staged_results: Path,
    output: Path,
    staged_models: Path,
    models: Path,
) -> None:
    """Replace result and model directories together, leaving no stale model files."""
    backups: list[tuple[Path, Path]] = []
    try:
        for existing in (output, models):
            if existing.exists():
                parent = existing.parent
                empty = Path(tempfile.mkdtemp(prefix=f".{existing.name}-backup-", dir=str(parent)))
                empty.rmdir()
                shutil.move(str(existing), str(empty))
                backups.append((existing, empty))
        staged_results.rename(output)
        staged_models.rename(models)
    except Exception:
        if output.exists() and not any(original == output for original, _ in backups):
            shutil.rmtree(output, ignore_errors=True)
        if models.exists() and not any(original == models for original, _ in backups):
            shutil.rmtree(models, ignore_errors=True)
        for original, backup in reversed(backups):
            if original.exists():
                shutil.rmtree(original, ignore_errors=True)
            if backup.exists():
                shutil.move(str(backup), str(original))
        raise
    else:
        for _, backup in backups:
            shutil.rmtree(backup, ignore_errors=True)
