from pathlib import Path

import pandas as pd
import pytest

from issue_classifier import (
    CsvIssueStore,
    InMemoryIssueStore,
    validate_issue_frame,
    validate_official_dataset_invariants,
)


DATA_DIR = Path(__file__).parents[1] / "official_nlbse24/data"


@pytest.fixture
def source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "issues_train.csv"),
        pd.read_csv(DATA_DIR / "issues_test.csv"),
    )


def test_csv_and_memory_stores_match_on_official_data(source_frames) -> None:
    train, test = source_frames
    csv_store = CsvIssueStore(DATA_DIR)
    memory_store = InMemoryIssueStore(train, test)

    pd.testing.assert_frame_equal(csv_store.load("train"), memory_store.load("train"))
    pd.testing.assert_frame_equal(csv_store.load("test"), memory_store.load("test"))


@pytest.mark.parametrize("store_kind", ["csv", "memory"])
def test_stores_return_defensive_copies(source_frames, store_kind) -> None:
    train, test = source_frames
    store = CsvIssueStore(DATA_DIR) if store_kind == "csv" else InMemoryIssueStore(train, test)

    first = store.load("train")
    first.loc[first.index[0], "label"] = "changed"
    second = store.load("train")

    assert second.loc[second.index[0], "label"] != "changed"


def test_in_memory_store_copies_constructor_inputs(source_frames) -> None:
    train, test = source_frames
    store = InMemoryIssueStore(train, test)
    train.loc[train.index[0], "label"] = "changed"

    assert store.load("train").loc[0, "label"] != "changed"


def test_in_memory_store_can_hold_only_official_train(source_frames) -> None:
    train, _ = source_frames
    store = InMemoryIssueStore(train=train, test=None)

    loaded = store.load("train")
    loaded.loc[loaded.index[0], "label"] = "changed"

    assert store.load("train").loc[0, "label"] != "changed"
    with pytest.raises(ValueError, match="Split 'test' is unavailable in this in-memory store"):
        store.load("test")


def test_in_memory_store_can_hold_only_official_test(source_frames) -> None:
    _, test = source_frames
    store = InMemoryIssueStore(train=None, test=test)

    loaded = store.load("test")
    loaded.loc[loaded.index[0], "label"] = "changed"

    assert store.load("test").loc[0, "label"] != "changed"
    with pytest.raises(ValueError, match="Split 'train' is unavailable in this in-memory store"):
        store.load("train")


def test_in_memory_store_requires_a_split(source_frames) -> None:
    with pytest.raises(ValueError, match="requires at least one available split"):
        InMemoryIssueStore(train=None, test=None)


def test_invalid_split_is_rejected_before_csv_access(tmp_path, source_frames) -> None:
    with pytest.raises(ValueError, match="Invalid split"):
        CsvIssueStore(tmp_path / "does-not-exist").load("validation")

    train, test = source_frames
    with pytest.raises(ValueError, match="Invalid split"):
        InMemoryIssueStore(train, test).load("validation")


def test_missing_columns_are_named(source_frames) -> None:
    train, _ = source_frames
    with pytest.raises(ValueError, match="Missing required columns: 'body'"):
        validate_issue_frame(train.drop(columns=["body"]))


def test_invalid_label_is_named(source_frames) -> None:
    train, _ = source_frames
    train.loc[0, "label"] = "other"
    with pytest.raises(ValueError, match="Invalid labels: 'other'"):
        validate_issue_frame(train)


def test_invalid_repository_is_named(source_frames) -> None:
    train, _ = source_frames
    train.loc[0, "repo"] = "unknown/project"
    with pytest.raises(ValueError, match="Invalid repositories: 'unknown/project'"):
        validate_issue_frame(train)


def test_missing_titles_are_rejected_and_body_nulls_are_valid(source_frames) -> None:
    train, _ = source_frames
    train.loc[0, "body"] = None
    validate_issue_frame(train)
    train.loc[0, "title"] = None

    with pytest.raises(ValueError, match="Missing titles: 1 rows"):
        validate_issue_frame(train)


def test_nested_mutable_body_values_are_rejected(source_frames) -> None:
    train, _ = source_frames
    train["body"] = train["body"].astype(object)
    train.at[0, "body"] = [{"nested": "value"}]

    with pytest.raises(ValueError, match="Invalid body values"):
        validate_issue_frame(train)


def test_duplicate_required_columns_are_rejected(source_frames) -> None:
    train, _ = source_frames
    train = pd.concat([train, train[["body"]]], axis=1)
    train.columns = ["repo", "created_at", "label", "title", "body", "body"]

    with pytest.raises(ValueError, match="Duplicate column names: 'body'"):
        validate_issue_frame(train)


def test_official_dataset_invariants(source_frames) -> None:
    train, test = source_frames
    validate_official_dataset_invariants(train, "train")
    validate_official_dataset_invariants(test, "test")


def test_official_row_count_and_balance_failures_are_named(source_frames) -> None:
    train, _ = source_frames
    shortened = train.iloc[:-1].copy()

    with pytest.raises(ValueError, match="Expected 1500 rows"):
        validate_official_dataset_invariants(shortened, "train")

    shortened.loc[0, "label"] = "feature"
    with pytest.raises(ValueError, match="Expected 100 rows per repository/label pair"):
        validate_official_dataset_invariants(shortened, "train")


def test_official_body_null_count_failure_is_named(source_frames) -> None:
    train, _ = source_frames
    train.loc[0, "body"] = None

    with pytest.raises(ValueError, match="Expected 0 missing bodies in train; found 1"):
        validate_official_dataset_invariants(train, "train")


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("row_count", "Expected 1500 rows"),
        ("balance", "Expected 100 rows per repository/label pair"),
        ("body_nulls", "Expected 2 missing bodies in test; found 1"),
    ],
)
@pytest.mark.parametrize("store_kind", ["csv", "memory"])
def test_both_official_stores_reject_corrupted_data(
    source_frames, tmp_path, corruption, message, store_kind
) -> None:
    train, test = source_frames
    if corruption == "row_count":
        train = train.iloc[:-1].copy()
        split = "train"
    elif corruption == "balance":
        train = train.copy()
        train.loc[0, "label"] = "feature"
        split = "train"
    else:
        test = test.copy()
        first_missing_body = test.index[test["body"].isna()][0]
        test.loc[first_missing_body, "body"] = "filled"
        split = "test"

    if store_kind == "csv":
        train.to_csv(tmp_path / "issues_train.csv", index=False)
        test.to_csv(tmp_path / "issues_test.csv", index=False)
        with pytest.raises(ValueError, match=message):
            CsvIssueStore(tmp_path).load(split)
    else:
        with pytest.raises(ValueError, match=message):
            InMemoryIssueStore(train, test)
