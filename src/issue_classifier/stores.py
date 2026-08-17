"""CSV-backed and in-memory issue stores."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from .constants import Split
from .validation import validate_official_dataset_invariants, validate_split


class IssueStore(Protocol):
    """Interface shared by persistent and in-memory issue stores."""

    def load(self, split: Split) -> pd.DataFrame:
        """Load a validated defensive copy of one dataset split."""


class CsvIssueStore:
    """Load the official CSV splits from a data directory."""

    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)

    def load(self, split: Split) -> pd.DataFrame:
        valid_split = validate_split(split)
        path = self._data_dir / f"issues_{valid_split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {valid_split} data file: {path}")
        frame = pd.read_csv(path)
        validate_official_dataset_invariants(frame, valid_split)
        return frame.copy(deep=True)


class InMemoryIssueStore:
    """Store validated train and test frames behind the IssueStore interface."""

    def __init__(
        self,
        train: pd.DataFrame | None = None,
        test: pd.DataFrame | None = None,
    ):
        if train is None and test is None:
            raise ValueError("InMemoryIssueStore requires at least one available split")
        self._frames: dict[str, pd.DataFrame] = {}
        if train is not None:
            validate_official_dataset_invariants(train, "train")
            self._frames["train"] = train.copy(deep=True)
        if test is not None:
            validate_official_dataset_invariants(test, "test")
            self._frames["test"] = test.copy(deep=True)

    def load(self, split: Split) -> pd.DataFrame:
        valid_split = validate_split(split)
        if valid_split not in self._frames:
            raise ValueError(f"Split {valid_split!r} is unavailable in this in-memory store")
        return self._frames[valid_split].copy(deep=True)
