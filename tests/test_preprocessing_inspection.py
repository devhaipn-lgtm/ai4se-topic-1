import json
from pathlib import Path

import pandas as pd
import pytest

from issue_classifier import (
    inspect_dataset,
    build_model_text,
    normalize_whitespace,
    raw_text_hash,
)


DATA_DIR = Path(__file__).parents[1] / "official_nlbse24/data"


@pytest.fixture
def official_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "issues_train.csv"),
        pd.read_csv(DATA_DIR / "issues_test.csv"),
    )


def test_model_text_is_exact_and_preserves_content(official_frames) -> None:
    train, _ = official_frames
    sample = train.iloc[[0]].copy()
    sample.loc[sample.index[0], "title"] = "  Fix `x_id`?  "
    sample.loc[sample.index[0], "body"] = "code:\n```py\nurl=https://x.test/a?b=1\n```"

    result = build_model_text(sample)

    assert result.iloc[0] == "[TITLE] Fix `x_id`? [BODY] code: ```py url=https://x.test/a?b=1 ```"


def test_whitespace_rule_and_null_body(official_frames) -> None:
    train, _ = official_frames
    sample = train.iloc[[0]].copy()
    sample.loc[sample.index[0], "title"] = "A\tB"
    sample.loc[sample.index[0], "body"] = None

    assert normalize_whitespace("  A\n\tB  ") == "A B"
    assert build_model_text(sample).iloc[0] == "[TITLE] A B [BODY]"


def test_preprocessing_does_not_mutate_input(official_frames) -> None:
    train, _ = official_frames
    before = train.copy(deep=True)

    build_model_text(train)

    pd.testing.assert_frame_equal(train, before)


def test_inspection_is_json_serializable_and_reports_official_overlap(official_frames) -> None:
    train, test = official_frames
    result = inspect_dataset(train, test)

    json.dumps(result)
    overlap = result["exact_train_test_overlap"]
    assert overlap["count"] == 3
    assert overlap["conflicting_label_count"] == 1
    assert all(
        {
            "text_hash",
            "train_repo",
            "train_label",
            "test_repo",
            "test_label",
            "title",
            "conflicting_label",
        }
        <= set(record)
        for record in overlap["records"]
    )


def test_inspection_reports_duplicates_and_missing_values(official_frames) -> None:
    train, test = official_frames
    result = inspect_dataset(train, test)

    assert result["train"]["missing_values"]["body"] == 0
    assert result["test"]["missing_values"]["body"] == 2
    assert len(result["train"]["exact_duplicate_groups"]) == 1
    assert result["train"]["exact_duplicate_groups"][0]["count"] == 2
    assert result["test"]["exact_duplicate_groups"] == []


def test_inspection_datetime_index_is_json_serializable_and_deterministic(official_frames) -> None:
    train, test = official_frames
    train.index = pd.date_range("2024-01-01", periods=len(train), freq="min")
    test.index = pd.date_range("2024-02-01", periods=len(test), freq="min")

    first = inspect_dataset(train, test)
    second = inspect_dataset(train, test)

    assert json.dumps(first)
    assert first == second
    duplicate_rows = first["train"]["exact_duplicate_groups"][0]["rows"]
    assert all(isinstance(row, str) for row in duplicate_rows)


def test_raw_hash_is_deterministic_and_exact() -> None:
    first = raw_text_hash("Title", "body\n")
    second = raw_text_hash("Title", "body\n")

    assert first == second
    assert first != raw_text_hash("Title", "body ")
