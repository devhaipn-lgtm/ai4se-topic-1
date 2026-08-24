import json

import pandas as pd
import pytest

from issue_classifier import (
    ALLOWED_REPOSITORIES,
    CandidateSpec,
    candidate_specs,
    run_cross_validation,
    select_candidate,
    summarize_candidates,
)


REPOSITORIES = sorted(ALLOWED_REPOSITORIES)
LABELS = ("bug", "feature", "question")


class TrainOnlyStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[str] = []

    def load(self, split: str) -> pd.DataFrame:
        self.calls.append(split)
        if split != "train":
            raise AssertionError("test split was accessed")
        return self.frame.copy(deep=True)


@pytest.fixture
def synthetic_training_frame() -> pd.DataFrame:
    rows = []
    for repository in REPOSITORIES:
        for label in LABELS:
            for number in range(10):
                rows.append(
                    {
                        "repo": repository,
                        "created_at": f"2024-01-{number + 1:02d}",
                        "label": label,
                        "title": f"{repository} {label} title {number}",
                        "body": f"{label} body token {number} for {repository}",
                    }
                )
    return pd.DataFrame(rows)


def test_candidate_grid_has_six_explicit_candidates() -> None:
    specs = candidate_specs()

    assert len(specs) == 6
    assert [spec.name for spec in specs] == [
        "dummy_most_frequent",
        "word_tfidf_logreg_c1",
        "word_tfidf_logreg_c4",
        "word_char_tfidf_svc_c0_5",
        "word_char_tfidf_svc_c1",
        "word_char_tfidf_svc_c2",
    ]
    assert all(spec.factory() is not None for spec in specs)


def test_cross_validation_is_train_only_and_has_expected_shape(
    synthetic_training_frame, tmp_path
) -> None:
    store = TrainOnlyStore(synthetic_training_frame)

    result = run_cross_validation(store, tmp_path)

    assert store.calls == ["train"]
    assert len(result["fold_scores"]) == 6 * 5 * 5
    assert {record["candidate"] for record in result["fold_scores"]} == {
        spec.name for spec in candidate_specs()
    }
    for summary in result["summaries"]:
        assert summary["fold_count"] == 25
        assert set(summary["per_repository"]) == set(REPOSITORIES)
        assert all(
            len(values["fold_scores"]) == 5
            for values in summary["per_repository"].values()
        )


def test_cross_validation_is_deterministic_and_writes_schemas(
    synthetic_training_frame, tmp_path
) -> None:
    first = run_cross_validation(TrainOnlyStore(synthetic_training_frame), tmp_path / "first")
    second = run_cross_validation(TrainOnlyStore(synthetic_training_frame), tmp_path / "second")

    assert first == second
    json_result = json.loads((tmp_path / "first/cross_validation.json").read_text())
    csv_result = pd.read_csv(tmp_path / "first/cross_validation.csv")
    assert json_result == first
    assert len(csv_result) == 150
    assert set(csv_result.columns) == {
        "candidate",
        "repository",
        "fold",
        "macro_f1",
        "train_rows",
        "validation_rows",
    }


def test_aggregation_uses_repository_means_and_population_std() -> None:
    spec = CandidateSpec("candidate", 1, {}, lambda: None)
    records = [
        {
            "candidate": "candidate",
            "repository": repository,
            "fold": fold,
            "macro_f1": 0.1 * (index + 1),
        }
        for index, repository in enumerate(REPOSITORIES)
        for fold in range(1, 6)
    ]

    summary = summarize_candidates(records, [spec], REPOSITORIES)[0]

    expected_repo_means = [0.1 * (index + 1) for index in range(5)]
    assert summary["global_mean"] == pytest.approx(sum(expected_repo_means) / 5)
    assert summary["global_std"] == pytest.approx(pd.Series([record["macro_f1"] for record in records]).std(ddof=0))


def test_tie_breaking_is_mean_std_rank_then_name() -> None:
    summaries = [
        {"candidate": "z", "global_mean": 1.0, "global_std": 0.2, "complexity_rank": 0},
        {"candidate": "b", "global_mean": 1.0, "global_std": 0.1, "complexity_rank": 9},
        {"candidate": "a", "global_mean": 1.0, "global_std": 0.1, "complexity_rank": 1},
        {"candidate": "c", "global_mean": 1.0, "global_std": 0.1, "complexity_rank": 1},
    ]

    assert select_candidate(summaries) == "a"
