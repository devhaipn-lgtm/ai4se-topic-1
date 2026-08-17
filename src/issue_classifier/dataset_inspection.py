"""Dataset quality checks and exact train/test overlap inspection."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time
from collections import defaultdict
from typing import Any, Iterable

import pandas as pd

from .preprocessing import build_model_text
from .validation import validate_issue_frame


def _raw_body(value: object) -> str:
    return "" if pd.isna(value) else value


def raw_text_hash(title: str, body: object) -> str:
    """Hash the exact title/body pair after only scalar-null body conversion."""

    payload = json.dumps(
        [title, _raw_body(body)], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return [(title, _raw_body(body)) for title, body in zip(frame["title"], frame["body"])]


def _json_value(value: object) -> object:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        item = value.item()
        if item is not value:
            return _json_value(item)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _length_statistics(values: Iterable[str]) -> dict[str, int | float]:
    lengths = [len(value) for value in values]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    series = pd.Series(lengths, dtype="int64")
    return {
        "count": int(series.count()),
        "min": int(series.min()),
        "max": int(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def _duplicate_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[object]] = defaultdict(list)
    for index, pair in zip(frame.index, _raw_pairs(frame)):
        groups[pair].append(_json_value(index))

    duplicates = []
    for (title, body), rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        if len(rows) > 1:
            duplicates.append(
                {
                    "text_hash": raw_text_hash(title, body),
                    "title": title,
                    "count": len(rows),
                    "rows": rows,
                }
            )
    return duplicates


def _single_or_list(values: Iterable[object]) -> object:
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else unique


def _inspect_split(frame: pd.DataFrame) -> dict[str, Any]:
    validate_issue_frame(frame)
    model_text = build_model_text(frame)
    raw_pairs = _raw_pairs(frame)
    repo_label_counts = frame.groupby(["repo", "label"]).size().unstack(fill_value=0)
    return {
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "repo_counts": {str(key): int(value) for key, value in frame["repo"].value_counts().items()},
        "label_counts": {str(key): int(value) for key, value in frame["label"].value_counts().items()},
        "repo_label_counts": {
            str(repo): {str(label): int(count) for label, count in row.items()}
            for repo, row in repo_label_counts.sort_index().iterrows()
        },
        "missing_values": {
            str(column): int(count) for column, count in frame.isna().sum().items()
        },
        "text_length_statistics": {
            "title": _length_statistics([pair[0] for pair in raw_pairs]),
            "body": _length_statistics([pair[1] for pair in raw_pairs]),
            "model_text": _length_statistics(model_text.tolist()),
        },
        "exact_duplicate_groups": _duplicate_groups(frame),
    }


def _cross_split_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    train_pairs = _raw_pairs(train)
    test_pairs = _raw_pairs(test)
    train_matches: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, pair in zip(train.index, train_pairs):
        train_matches[pair].append(index)

    records = []
    for test_index, pair in zip(test.index, test_pairs):
        if pair not in train_matches:
            continue
        matching_train = train.loc[train.index.isin(train_matches[pair])]
        train_labels = matching_train["label"].tolist()
        train_repos = matching_train["repo"].tolist()
        test_repo = test.loc[test_index, "repo"]
        test_label = test.loc[test_index, "label"]
        records.append(
            {
                "text_hash": raw_text_hash(*pair),
                "train_repo": _single_or_list(train_repos),
                "train_label": _single_or_list(train_labels),
                "test_repo": test_repo,
                "test_label": test_label,
                "title": pair[0],
                "conflicting_label": bool(any(label != test_label for label in train_labels)),
            }
        )
    return {
        "count": len(records),
        "conflicting_label_count": sum(record["conflicting_label"] for record in records),
        "records": records,
    }


def inspect_dataset(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Summarize data quality and overlaps in both official splits."""

    return {
        "train": _inspect_split(train),
        "test": _inspect_split(test),
        "exact_train_test_overlap": _cross_split_overlap(train, test),
    }
