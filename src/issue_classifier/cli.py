"""Argparse command-line interface for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .model_selection import run_cross_validation
from .stores import CsvIssueStore, InMemoryIssueStore
from .workflow import (
    evaluate_chosen_models,
    train_chosen_models,
    validate_selection_artifact,
    verify_training_artifacts,
)


def _store(backend: str, data_dir: Path, required_splits: tuple[str, ...] = ("train", "test")):
    csv_store = CsvIssueStore(data_dir)
    if backend == "file":
        return csv_store
    frames = {
        split: csv_store.load(split)
        for split in required_splits
    }
    return InMemoryIssueStore(train=frames.get("train"), test=frames.get("test"))


def _frame_summary(frame) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "repositories": sorted(str(value) for value in frame["repo"].unique()),
        "labels": sorted(str(value) for value in frame["label"].unique()),
        "missing_bodies": int(frame["body"].isna().sum()),
    }


def _validate_data(args: argparse.Namespace) -> int:
    store = _store(args.backend, args.data_dir, ("train", "test"))
    train = store.load("train")
    test = store.load("test")
    payload = {
        "backend": args.backend,
        "train": _frame_summary(train),
        "test": _frame_summary(test),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _cross_validate(args: argparse.Namespace) -> int:
    store = _store(args.backend, args.data_dir, ("train",))
    result = run_cross_validation(store, args.output_dir)
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "json_path": str(args.output_dir / "cross_validation.json"),
                "csv_path": str(args.output_dir / "cross_validation.csv"),
            },
            sort_keys=True,
        )
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    prepared = validate_selection_artifact(args.selection)
    result = train_chosen_models(
        _store(args.backend, args.data_dir, ("train",)),
        selection_path=args.selection,
        model_root=args.model_dir,
        training_manifest_path=args.training_manifest,
        backend=args.backend,
        prepared=prepared,
    )
    print(
        json.dumps(
            {
                "selected_candidate": result["manifest"]["selected_candidate"]["name"],
                "model_dir": result["model_root"],
                "training_manifest": result["manifest_path"],
            },
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    prepared = verify_training_artifacts(
        args.selection, args.model_dir, args.training_manifest
    )
    result = evaluate_chosen_models(
        _store(args.backend, args.data_dir, ("test",)),
        selection_path=args.selection,
        model_root=args.model_dir,
        training_manifest_path=args.training_manifest,
        output_root=args.output_dir,
        backend=args.backend,
        prepared=prepared,
    )
    print(
        json.dumps(
            {
                "selected_candidate": result["manifest"]["selected_candidate"]["name"],
                "metrics": str(args.output_dir / "metrics.json"),
                "manifest": str(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m issue_classifier",
        description="Train-only issue classification workflows and final evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate both official data splits.")
    validate.add_argument("--backend", choices=("file", "memory"), required=True)
    validate.add_argument("--data-dir", type=Path, default=Path("official_nlbse24/data"))
    validate.set_defaults(handler=_validate_data)

    cross = subparsers.add_parser("cross-validate", help="Run train-only candidate cross-validation.")
    cross.add_argument("--model", choices=("all",), required=True)
    cross.add_argument("--backend", choices=("file", "memory"), default="file")
    cross.add_argument("--data-dir", type=Path, default=Path("official_nlbse24/data"))
    cross.add_argument("--output-dir", type=Path, default=Path("results"))
    cross.set_defaults(handler=_cross_validate)

    common = {
        "selection": ("--selection", {"type": Path, "default": Path("results/cross_validation.json")}),
        "data_dir": ("--data-dir", {"type": Path, "default": Path("official_nlbse24/data")}),
        "backend": ("--backend", {"choices": ("file", "memory"), "default": "file"}),
        "model_dir": ("--model-dir", {"type": Path, "default": Path("artifacts/chosen-model-seed42/models")}),
        "training_manifest": ("--training-manifest", {"type": Path, "default": Path("artifacts/chosen-model-seed42/training_manifest.json")}),
    }
    train = subparsers.add_parser("train", help="Fit the chosen model using training data.")
    train.add_argument("--model", choices=("selected",), required=True)
    for _, (flag, kwargs) in common.items():
        train.add_argument(flag, **kwargs)
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate the trained chosen model.")
    evaluate.add_argument("--model", choices=("selected",), required=True)
    for _, (flag, kwargs) in common.items():
        evaluate.add_argument(flag, **kwargs)
    evaluate.add_argument("--output-dir", type=Path, default=Path("results/chosen-model-seed42"))
    evaluate.set_defaults(handler=_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:  # CLI boundary: preserve a concise nonzero failure.
        print(f"error: {exc}", file=sys.stderr)
        return 2
