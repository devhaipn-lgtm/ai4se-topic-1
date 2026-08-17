"""Pure, deterministic text construction for issue reports."""

from __future__ import annotations

import re

import pandas as pd

from .validation import validate_issue_frame

_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Replace every whitespace run with one ASCII space and trim the ends."""

    return _WHITESPACE.sub(" ", text).strip()


def build_model_text(frame: pd.DataFrame) -> pd.Series:
    """Build `[TITLE] <title> [BODY] <body>` text without mutating ``frame``.

    Scalar-null bodies become empty strings before the deterministic whitespace
    rule is applied. Punctuation, case, URLs, identifiers, Markdown, and code
    characters are otherwise retained.
    """

    validate_issue_frame(frame)
    texts = []
    for title, body in zip(frame["title"], frame["body"]):
        body_text = "" if pd.isna(body) else body
        texts.append(normalize_whitespace(f"[TITLE] {title} [BODY] {body_text}"))
    return pd.Series(texts, index=frame.index, name="text")
