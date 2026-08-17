"""Core package for the AI4SE issue report classification project."""

from .constants import (
    ALLOWED_LABELS,
    ALLOWED_REPOSITORIES,
    REQUIRED_COLUMNS,
    Split,
)
from .dataset_inspection import inspect_dataset, raw_text_hash
from .model_selection import (
    CandidateSpec,
    candidate_specs,
    run_cross_validation,
    select_candidate,
    summarize_candidates,
)
from .preprocessing import build_model_text, normalize_whitespace
from .stores import CsvIssueStore, InMemoryIssueStore, IssueStore
from .validation import validate_issue_frame, validate_official_dataset_invariants
from .final_evaluation import run_final_chosen_evaluation
from .workflow import (
    evaluate_chosen_models,
    train_chosen_models,
    validate_selection_artifact,
    verify_training_artifacts,
)
__version__ = "0.1.0"

__all__ = [
    "ALLOWED_LABELS",
    "ALLOWED_REPOSITORIES",
    "inspect_dataset",
    "build_model_text",
    "CandidateSpec",
    "candidate_specs",
    "CsvIssueStore",
    "InMemoryIssueStore",
    "IssueStore",
    "REQUIRED_COLUMNS",
    "Split",
    "normalize_whitespace",
    "raw_text_hash",
    "run_cross_validation",
    "select_candidate",
    "summarize_candidates",
    "validate_issue_frame",
    "validate_official_dataset_invariants",
    "run_final_chosen_evaluation",
    "train_chosen_models",
    "evaluate_chosen_models",
    "validate_selection_artifact",
    "verify_training_artifacts",
]
