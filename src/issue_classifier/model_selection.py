"""Train-only candidate selection with deterministic cross-validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .constants import ALLOWED_REPOSITORIES
from .preprocessing import build_model_text
from .stores import IssueStore

SEED = 42
FOLD_COUNT = 5
CLASS_LABELS = ("bug", "feature", "question")


@dataclass(frozen=True)
class CandidateSpec:
    """Serializable candidate metadata plus its estimator factory."""

    name: str
    complexity_rank: int
    parameters: dict[str, Any]
    factory: Callable[[], Pipeline]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "complexity_rank": self.complexity_rank,
            "parameters": self.parameters,
        }


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)


def _word_logistic_regression(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", _word_vectorizer()),
            (
                "classifier",
                LogisticRegression(C=c_value, max_iter=2000, random_state=SEED),
            ),
        ]
    )


def _word_char_linear_svc(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", _word_vectorizer()),
            ("char_wb", _char_vectorizer()),
        ]
    )
    return Pipeline(
        [
            ("tfidf", features),
            ("classifier", LinearSVC(C=c_value, random_state=SEED, max_iter=5000)),
        ]
    )


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the six candidates in their explicit deterministic rank order."""

    return (
        CandidateSpec(
            "dummy_most_frequent",
            0,
            {"estimator": "DummyClassifier", "strategy": "most_frequent"},
            lambda: DummyClassifier(strategy="most_frequent", random_state=SEED),
        ),
        CandidateSpec(
            "word_tfidf_logreg_c1",
            1,
            {
                "estimator": "LogisticRegression",
                "features": "word_tfidf",
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "C": 1,
                "max_iter": 2000,
                "random_state": SEED,
            },
            lambda: _word_logistic_regression(1),
        ),
        CandidateSpec(
            "word_tfidf_logreg_c4",
            2,
            {
                "estimator": "LogisticRegression",
                "features": "word_tfidf",
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "C": 4,
                "max_iter": 2000,
                "random_state": SEED,
            },
            lambda: _word_logistic_regression(4),
        ),
        CandidateSpec(
            "word_char_tfidf_svc_c0_5",
            3,
            {
                "estimator": "LinearSVC",
                "features": "word_plus_char_wb_tfidf",
                "word_ngram_range": [1, 2],
                "char_analyzer": "char_wb",
                "char_ngram_range": [3, 5],
                "sublinear_tf": True,
                "C": 0.5,
                "max_iter": 5000,
                "random_state": SEED,
            },
            lambda: _word_char_linear_svc(0.5),
        ),
        CandidateSpec(
            "word_char_tfidf_svc_c1",
            4,
            {
                "estimator": "LinearSVC",
                "features": "word_plus_char_wb_tfidf",
                "word_ngram_range": [1, 2],
                "char_analyzer": "char_wb",
                "char_ngram_range": [3, 5],
                "sublinear_tf": True,
                "C": 1,
                "max_iter": 5000,
                "random_state": SEED,
            },
            lambda: _word_char_linear_svc(1),
        ),
        CandidateSpec(
            "word_char_tfidf_svc_c2",
            5,
            {
                "estimator": "LinearSVC",
                "features": "word_plus_char_wb_tfidf",
                "word_ngram_range": [1, 2],
                "char_analyzer": "char_wb",
                "char_ngram_range": [3, 5],
                "sublinear_tf": True,
                "C": 2,
                "max_iter": 5000,
                "random_state": SEED,
            },
            lambda: _word_char_linear_svc(2),
        ),
    )


def _per_repository_summary(
    fold_records: list[dict[str, Any]], repositories: Iterable[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for repository in repositories:
        scores = [
            record["macro_f1"]
            for record in fold_records
            if record["repository"] == repository
        ]
        result[repository] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=0)),
            "fold_scores": [float(score) for score in scores],
        }
    return result


def summarize_candidates(
    fold_records: list[dict[str, Any]],
    specs: Iterable[CandidateSpec],
    repositories: Iterable[str],
) -> list[dict[str, Any]]:
    """Aggregate fold scores with explicit population-standard-deviation rules."""

    repository_list = list(repositories)
    summaries = []
    for spec in specs:
        candidate_records = [
            record for record in fold_records if record["candidate"] == spec.name
        ]
        per_repository = _per_repository_summary(candidate_records, repository_list)
        all_scores = [record["macro_f1"] for record in candidate_records]
        global_mean = float(
            np.mean([per_repository[repo]["mean"] for repo in repository_list])
        )
        summaries.append(
            {
                "candidate": spec.name,
                "complexity_rank": spec.complexity_rank,
                "parameters": spec.parameters,
                "fold_count": len(all_scores),
                "per_repository": per_repository,
                "global_mean": global_mean,
                "global_std": float(np.std(all_scores, ddof=0)),
            }
        )
    return summaries


def select_candidate(summaries: Iterable[dict[str, Any]]) -> str:
    """Select by mean, population std, complexity rank, then candidate name."""

    ordered = sorted(
        summaries,
        key=lambda summary: (
            -summary["global_mean"],
            summary["global_std"],
            summary["complexity_rank"],
            summary["candidate"],
        ),
    )
    if not ordered:
        raise ValueError("Cannot select a candidate from empty summaries")
    return ordered[0]["candidate"]


def _write_results(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "cross_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "repository", "fold", "macro_f1", "train_rows", "validation_rows"],
        )
        writer.writeheader()
        writer.writerows(result["fold_scores"])


def run_cross_validation(
    store: IssueStore,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Evaluate all candidates using only one ``load('train')`` call."""

    train = store.load("train")
    model_text = build_model_text(train)
    repositories = [repo for repo in ALLOWED_REPOSITORIES if repo in set(train["repo"])]
    repositories.sort()
    if repositories != sorted(ALLOWED_REPOSITORIES):
        raise ValueError("Training data must contain all five official repositories")

    specs = candidate_specs()
    fold_records: list[dict[str, Any]] = []
    for repository in repositories:
        mask = train["repo"] == repository
        repository_text = model_text.loc[mask]
        repository_labels = train.loc[mask, "label"]
        splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
        folds = list(splitter.split(repository_text, repository_labels))
        for spec in specs:
            for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
                estimator = spec.factory()
                estimator.fit(
                    repository_text.iloc[train_indices],
                    repository_labels.iloc[train_indices],
                )
                predictions = estimator.predict(repository_text.iloc[validation_indices])
                score = f1_score(
                    repository_labels.iloc[validation_indices],
                    predictions,
                    labels=CLASS_LABELS,
                    average="macro",
                    zero_division=0,
                )
                fold_records.append(
                    {
                        "candidate": spec.name,
                        "repository": repository,
                        "fold": fold_number,
                        "macro_f1": float(score),
                        "train_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                    }
                )

    summaries = summarize_candidates(fold_records, specs, repositories)
    result = {
        "protocol": "chosen_model_selection_train_only",
        "seed": SEED,
        "fold_count": FOLD_COUNT,
        "labels": list(CLASS_LABELS),
        "repositories": repositories,
        "candidate_parameters": {spec.name: spec.metadata() for spec in specs},
        "fold_scores": fold_records,
        "summaries": summaries,
        "selected_candidate": select_candidate(summaries),
    }
    _write_results(result, Path(output_dir))
    return result
