"""Shared data-store constants and types."""

from __future__ import annotations

from typing import Final, Literal

Split = Literal["train", "test"]

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "created_at",
    "label",
    "title",
    "body",
)
ALLOWED_LABELS: Final[frozenset[str]] = frozenset({"bug", "feature", "question"})
ALLOWED_REPOSITORIES: Final[frozenset[str]] = frozenset(
    {
        "facebook/react",
        "tensorflow/tensorflow",
        "microsoft/vscode",
        "bitcoin/bitcoin",
        "opencv/opencv",
    }
)
OFFICIAL_ROWS_PER_SPLIT: Final[int] = 1500
OFFICIAL_ROWS_PER_REPOSITORY_LABEL: Final[int] = 100
OFFICIAL_MISSING_BODIES: Final[dict[Split, int]] = {"train": 0, "test": 2}
