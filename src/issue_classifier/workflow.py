"""Separate train-only and test-only workflows for the chosen model."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_inspection import raw_text_hash
from .constants import ALLOWED_REPOSITORIES
from .final_evaluation import (
    RUN_ID,
    SEED,
    _canonical_frame_hash,
    _file_hash,
    _json_safe,
    _load_selected_spec,
    _metric_rows,
    _package_versions,
    _policy_metrics,
    _write_csv,
    _write_png,
)
from .model_selection import CLASS_LABELS, CandidateSpec
from .preprocessing import build_model_text
from .stores import IssueStore


def _manifest_path(model_root: Path, training_manifest_path: str | Path | None) -> Path:
    return Path(training_manifest_path) if training_manifest_path else model_root.parent / "training_manifest.json"


def _raw_training_pairs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "text_hash": raw_text_hash(title, body),
            "repo": str(repo),
            "label": str(label),
        }
        for repo, label, title, body in zip(
            frame["repo"], frame["label"], frame["title"], frame["body"]
        )
    ]


def _publish_training_bundle(
    staged_models: Path,
    model_root: Path,
    staged_manifest: Path,
    manifest_path: Path,
) -> None:
    backups: list[tuple[Path, Path]] = []
    try:
        for existing in (model_root, manifest_path):
            if existing.exists():
                empty = Path(tempfile.mkdtemp(prefix=f".{existing.name}-backup-", dir=str(existing.parent)))
                empty.rmdir()
                shutil.move(str(existing), str(empty))
                backups.append((existing, empty))
        staged_models.rename(model_root)
        staged_manifest.rename(manifest_path)
    except Exception:
        if model_root.exists() and not any(original == model_root for original, _ in backups):
            shutil.rmtree(model_root, ignore_errors=True)
        if manifest_path.exists() and not any(original == manifest_path for original, _ in backups):
            manifest_path.unlink(missing_ok=True)
        for original, backup in reversed(backups):
            if original.exists():
                if original.is_dir():
                    shutil.rmtree(original, ignore_errors=True)
                else:
                    original.unlink(missing_ok=True)
            if backup.exists():
                shutil.move(str(backup), str(original))
        raise
    else:
        for _, backup in backups:
            shutil.rmtree(backup, ignore_errors=True)


def train_chosen_models(
    store: IssueStore,
    selection_path: str | Path = "results/cross_validation.json",
    model_root: str | Path = "artifacts/chosen-model-seed42/models",
    training_manifest_path: str | Path | None = None,
    backend: str = "file",
    prepared: tuple[CandidateSpec, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit exactly the selected five repository models from train data only."""
    selected_spec, selection_payload = prepared or validate_selection_artifact(selection_path)
    train = store.load("train")
    repositories = sorted(ALLOWED_REPOSITORIES)
    missing = sorted(set(repositories) - set(train["repo"]))
    if missing:
        raise ValueError(f"Missing repository rows: {', '.join(missing)}")
    train_text = build_model_text(train)
    model_root_path = Path(model_root)
    manifest_path = _manifest_path(model_root_path, training_manifest_path)
    model_root_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staged_models = Path(tempfile.mkdtemp(prefix=f".{RUN_ID}-train-models-", dir=str(model_root_path.parent)))
    manifest_fd, manifest_name = tempfile.mkstemp(
        prefix=f".{RUN_ID}-training-", suffix=".json", dir=str(manifest_path.parent)
    )
    os.close(manifest_fd)
    staged_manifest = Path(manifest_name)
    model_files: list[dict[str, str]] = []
    started = time.perf_counter()
    try:
        import joblib

        for repository in repositories:
            mask = train["repo"] == repository
            estimator = selected_spec.factory()
            estimator.fit(train_text.loc[mask], train.loc[mask, "label"])
            filename = repository.replace("/", "_") + ".joblib"
            path = staged_models / filename
            joblib.dump(estimator, path)
            model_files.append({"repository": repository, "filename": filename, "sha256": _file_hash(path)})
        manifest = {
            "run_id": RUN_ID,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.perf_counter() - started),
            "seed": SEED,
            "selected_candidate": selected_spec.metadata(),
            "backend": backend,
            "train_content_sha256": _canonical_frame_hash(train),
            "train_rows": len(train),
            "labels": list(CLASS_LABELS),
            "repositories": repositories,
            "package_versions": _package_versions(),
            "selected_cv_result": {
                "path": str(selection_path),
                "sha256": _file_hash(Path(selection_path)),
                "selected_candidate": selection_payload["selected_candidate"],
            },
            "model_files": model_files,
            "train_raw_pairs": _raw_training_pairs(train),
        }
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish_training_bundle(staged_models, model_root_path, staged_manifest, manifest_path)
        return {"manifest": manifest, "manifest_path": str(manifest_path), "model_root": str(model_root_path)}
    finally:
        shutil.rmtree(staged_models, ignore_errors=True)
        staged_manifest.unlink(missing_ok=True)


def _load_training_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed training manifest: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model_files"), list):
        raise ValueError(f"Malformed training manifest: {path}; missing model_files")
    return payload


def validate_selection_artifact(selection_path: str | Path) -> tuple[CandidateSpec, dict[str, Any]]:
    """Validate the committed selection before any command-specific store exists."""
    return _load_selected_spec(Path(selection_path))


def _verify_model_entries(
    selected_spec: CandidateSpec,
    manifest: dict[str, Any],
    model_root: Path,
) -> None:
    selected = manifest.get("selected_candidate", {})
    if selected.get("name") != selected_spec.name:
        raise ValueError(
            "Training manifest selected candidate does not match selection artifact: "
            f"{selected.get('name')!r} != {selected_spec.name!r}"
        )
    expected = {repo.replace("/", "_") + ".joblib": repo for repo in sorted(ALLOWED_REPOSITORIES)}
    entries = manifest["model_files"]
    if {entry.get("filename") for entry in entries} != set(expected) or len(entries) != 5:
        raise ValueError("Training manifest must list exactly five repository models")
    for entry in entries:
        filename = entry.get("filename")
        repository = entry.get("repository")
        if filename not in expected or repository != expected[filename]:
            raise ValueError(f"Training manifest has invalid model entry: {entry!r}")
        path = model_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained model: {path}")
        if _file_hash(path) != entry.get("sha256"):
            raise ValueError(f"Trained model hash mismatch: {path}")


def verify_training_artifacts(
    selection_path: str | Path,
    model_root: str | Path,
    training_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify selection, manifest, filenames, and hashes without loading a data split."""
    selected_spec, selection_payload = validate_selection_artifact(selection_path)
    model_root_path = Path(model_root)
    manifest_path = _manifest_path(model_root_path, training_manifest_path)
    manifest = _load_training_manifest(manifest_path)
    _verify_model_entries(selected_spec, manifest, model_root_path)
    return {
        "selected_spec": selected_spec,
        "selection_payload": selection_payload,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "model_root": model_root_path,
    }


def _load_verified_models(bundle: dict[str, Any]) -> dict[str, Any]:
    """Load models after the caller has verified their hashes."""
    import joblib

    return {
        entry["repository"]: joblib.load(bundle["model_root"] / entry["filename"])
        for entry in bundle["manifest"]["model_files"]
    }


def _overlap_from_training_manifest(
    train_pairs: list[dict[str, Any]], test: pd.DataFrame
) -> tuple[dict[Any, str], list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in train_pairs:
        lookup[pair["text_hash"]].append(pair)
    by_index: dict[Any, str] = {}
    records: list[dict[str, Any]] = []
    for index, title, body, repo, label in zip(
        test.index, test["title"], test["body"], test["repo"], test["label"]
    ):
        text_hash = raw_text_hash(title, body)
        matches = lookup.get(text_hash, [])
        if not matches:
            continue
        by_index[index] = text_hash
        train_repos = [entry["repo"] for entry in matches]
        train_labels = [entry["label"] for entry in matches]
        records.append(
            {
                "text_hash": text_hash,
                "train_repo": train_repos[0] if len(set(train_repos)) == 1 else train_repos,
                "train_label": train_labels[0] if len(set(train_labels)) == 1 else train_labels,
                "test_repo": repo,
                "test_label": label,
                "title": title,
                "conflicting_label": bool(any(item != label for item in train_labels)),
            }
        )
    return by_index, records


def _publish_results(staged_results: Path, output: Path) -> None:
    backup: Path | None = None
    try:
        if output.exists():
            empty = Path(tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=str(output.parent)))
            empty.rmdir()
            shutil.move(str(output), str(empty))
            backup = empty
        staged_results.rename(output)
    except Exception:
        if output.exists() and backup is None:
            shutil.rmtree(output, ignore_errors=True)
        if backup is not None:
            if output.exists():
                shutil.rmtree(output, ignore_errors=True)
            shutil.move(str(backup), str(output))
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def evaluate_chosen_models(
    store: IssueStore,
    selection_path: str | Path = "results/cross_validation.json",
    model_root: str | Path = "artifacts/chosen-model-seed42/models",
    training_manifest_path: str | Path | None = None,
    output_root: str | Path = "results/chosen-model-seed42",
    backend: str = "file",
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load verified train artifacts, then load test once and write evaluation results."""
    started = time.perf_counter()
    bundle = prepared or verify_training_artifacts(
        selection_path, model_root, training_manifest_path
    )
    selected_spec = bundle["selected_spec"]
    manifest_path = bundle["manifest_path"]
    training_manifest = bundle["manifest"]
    models = _load_verified_models(bundle)
    test = store.load("test")
    test_text = build_model_text(test)
    overlap_by_index, overlap_records = _overlap_from_training_manifest(
        training_manifest.get("train_raw_pairs", []), test
    )
    if training_manifest.get("train_rows") == 1500 and len(test) == 1500:
        conflicts = sum(record["conflicting_label"] for record in overlap_records)
        if len(overlap_records) != 3 or conflicts != 1:
            raise ValueError(
                f"Unexpected official overlap count: {len(overlap_records)} records, {conflicts} conflicts"
            )
    predictions_by_policy = {"primary": [], "leakage_sensitive": []}
    for repository in sorted(ALLOWED_REPOSITORIES):
        mask = test["repo"] == repository
        if not bool(mask.any()):
            raise ValueError(f"Missing repository rows: {repository}")
        rows = test.loc[mask]
        predictions = models[repository].predict(test_text.loc[mask])
        for index, true_label, prediction in zip(rows.index, rows["label"], predictions):
            text_hash = overlap_by_index.get(index, raw_text_hash(test.loc[index, "title"], test.loc[index, "body"]))
            row = {
                "row_index": _json_safe(index),
                "repo": repository,
                "created_at": _json_safe(test.loc[index, "created_at"]),
                "true_label": str(true_label),
                "predicted_label": str(prediction),
                "text_hash": text_hash,
                "is_overlap": index in overlap_by_index,
            }
            predictions_by_policy["primary"].append({"policy": "primary", **row})
            if index not in overlap_by_index:
                predictions_by_policy["leakage_sensitive"].append({"policy": "leakage_sensitive", **row})
    policy_results = {
        policy: _policy_metrics(rows, 0 if policy == "primary" else len(predictions_by_policy["primary"]) - len(rows))
        for policy, rows in predictions_by_policy.items()
    }
    metrics = {
        "run_id": RUN_ID,
        "seed": SEED,
        "labels": list(CLASS_LABELS),
        "repositories": sorted(ALLOWED_REPOSITORIES),
        "selected_candidate": selected_spec.metadata(),
        "policies": policy_results,
    }
    output = Path(output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{RUN_ID}-results-", dir=str(output.parent)))
    figures = staged / "figures"
    figures.mkdir()
    try:
        (staged / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metric_rows: list[dict[str, Any]] = []
        for policy, result in policy_results.items():
            metric_rows.extend(_metric_rows(policy, result))
        _write_csv(staged / "metrics.csv", metric_rows)
        with (staged / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv

            fields = ["policy", "row_index", "repo", "created_at", "true_label", "predicted_label", "text_hash", "is_overlap"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for policy in ("primary", "leakage_sensitive"):
                writer.writerows(predictions_by_policy[policy])
        for policy, result in policy_results.items():
            for repository, repo_result in result["repositories"].items():
                _write_png(figures / f"{repository.replace('/', '_')}-{policy}-confusion-matrix.png", repo_result["confusion_matrix"])
        manifest = {
            "run_id": RUN_ID,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.perf_counter() - started),
            "seed": SEED,
            "selected_candidate": selected_spec.metadata(),
            "backend": backend,
            "train_content_sha256": training_manifest["train_content_sha256"],
            "test_content_sha256": _canonical_frame_hash(test),
            "row_counts": {"train": training_manifest["train_rows"], "test": len(test)},
            "labels": list(CLASS_LABELS),
            "repositories": sorted(ALLOWED_REPOSITORIES),
            "package_versions": _package_versions(),
            "selected_cv_result": training_manifest["selected_cv_result"],
            "training_manifest": {"path": str(manifest_path), "sha256": _file_hash(manifest_path)},
            "model_files": training_manifest["model_files"],
            "leakage_policy": {
                "definition": "raw title/body pair after scalar-null body conversion only",
                "excluded_count": len(overlap_by_index),
                "excluded_hashes": sorted(set(overlap_by_index.values())),
                "overlap_records": overlap_records,
            },
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish_results(staged, output)
        return {"metrics": metrics, "manifest": manifest, "predictions": predictions_by_policy, "model_files": training_manifest["model_files"]}
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def run_final_chosen_evaluation(
    store: IssueStore,
    selection_path: str | Path = "results/cross_validation.json",
    output_root: str | Path = "results/chosen-model-seed42",
    model_root: str | Path = "artifacts/chosen-model-seed42/models",
    backend: str = "file",
) -> dict[str, Any]:
    """Compose train-only fitting and verified test-only evaluation."""
    training = train_chosen_models(
        store,
        selection_path=selection_path,
        model_root=model_root,
        backend=backend,
    )
    prepared = verify_training_artifacts(
        selection_path, model_root, training["manifest_path"]
    )
    return evaluate_chosen_models(
        store,
        selection_path=selection_path,
        model_root=model_root,
        training_manifest_path=training["manifest_path"],
        output_root=output_root,
        backend=backend,
        prepared=prepared,
    )
