"""Schema, domain, and official-dataset invariant validation."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .constants import (
    ALLOWED_LABELS,
    ALLOWED_REPOSITORIES,
    OFFICIAL_MISSING_BODIES,
    OFFICIAL_ROWS_PER_REPOSITORY_LABEL,
    OFFICIAL_ROWS_PER_SPLIT,
    REQUIRED_COLUMNS,
    Split,
)


def validate_split(split: str) -> Split:
    """Return a valid split or fail before any backend access occurs."""

    if split not in ("train", "test"):
        raise ValueError(f"Invalid split {split!r}; expected 'train' or 'test'")
    return split


def _format_values(values: Iterable[object]) -> str:
    return ", ".join(repr(value) for value in values)


def _invalid_values(series: pd.Series, allowed: frozenset[str]) -> list[object]:
    invalid = series[~series.isin(allowed)].drop_duplicates().tolist()
    return sorted(invalid, key=repr)


def _is_scalar_null(value: object) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def _invalid_text_rows(
    series: pd.Series, *, allow_scalar_null: bool
) -> list[object]:
    invalid_rows: list[object] = []
    for index, value in series.items():
        valid_string = isinstance(value, str)
        valid_null = allow_scalar_null and _is_scalar_null(value)
        if not valid_string and not valid_null:
            invalid_rows.append(index)
    return invalid_rows


def validate_issue_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate common issue-frame schema and domain rules without mutation."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas.DataFrame")

    duplicate_columns = list(dict.fromkeys(frame.columns[frame.columns.duplicated()].tolist()))
    if duplicate_columns:
        raise ValueError(f"Duplicate column names: {_format_values(duplicate_columns)}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {_format_values(missing_columns)}")

    missing_title_rows = [
        index for index, value in frame["title"].items() if _is_scalar_null(value)
    ]
    if missing_title_rows:
        raise ValueError(f"Missing titles: {len(missing_title_rows)} rows")

    invalid_title_rows = _invalid_text_rows(frame["title"], allow_scalar_null=False)
    if invalid_title_rows:
        raise ValueError(
            "Invalid title values: expected non-null strings; "
            f"invalid rows {invalid_title_rows}"
        )

    invalid_body_rows = _invalid_text_rows(frame["body"], allow_scalar_null=True)
    if invalid_body_rows:
        raise ValueError(
            "Invalid body values: expected strings or scalar nulls; "
            f"invalid rows {invalid_body_rows}"
        )

    invalid_labels = _invalid_values(frame["label"], ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(f"Invalid labels: {_format_values(invalid_labels)}")

    invalid_repositories = _invalid_values(frame["repo"], ALLOWED_REPOSITORIES)
    if invalid_repositories:
        raise ValueError(f"Invalid repositories: {_format_values(invalid_repositories)}")

    return frame


def validate_official_dataset_invariants(frame: pd.DataFrame, split: str) -> None:
    """Validate the known row-count, balance, and null-count invariants."""

    valid_split = validate_split(split)
    validate_issue_frame(frame)

    failures: list[str] = []
    if len(frame) != OFFICIAL_ROWS_PER_SPLIT:
        failures.append(
            f"Expected {OFFICIAL_ROWS_PER_SPLIT} rows for {valid_split}; found {len(frame)}"
        )

    pair_counts = frame.groupby(["repo", "label"], dropna=False).size()
    bad_pairs = pair_counts[pair_counts != OFFICIAL_ROWS_PER_REPOSITORY_LABEL]
    if not bad_pairs.empty:
        details = ", ".join(
            f"{repo}/{label}={count}" for (repo, label), count in bad_pairs.items()
        )
        failures.append(
            "Expected 100 rows per repository/label pair; violations: " + details
        )

    missing_bodies = int(frame["body"].isna().sum())
    expected_missing_bodies = OFFICIAL_MISSING_BODIES[valid_split]
    if missing_bodies != expected_missing_bodies:
        failures.append(
            f"Expected {expected_missing_bodies} missing bodies in {valid_split}; "
            f"found {missing_bodies}"
        )

    if failures:
        raise ValueError("Official dataset invariant violations: " + "; ".join(failures))
