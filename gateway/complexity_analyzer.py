"""Lightweight text complexity scoring for gateway model routing."""

from __future__ import annotations

import re
from typing import Mapping


_CODE_RE = re.compile(r"\b(function|class|def|let|const|var|=>)\b", re.IGNORECASE)
_LOGIC_RE = re.compile(
    r"\b(if|else|how|why|explain|design|architecture|实现|架构)\b",
    re.IGNORECASE,
)


def _runtime_feedback_penalty(feedback: Mapping[str, object]) -> float:
    decayed_penalty = feedback.get("decayed_penalty")
    if decayed_penalty is not None:
        try:
            return max(0.0, min(float(decayed_penalty), 1.0))
        except (TypeError, ValueError):
            pass

    penalty = 0.0
    if bool(feedback.get("used_tools_previously")):
        penalty += 0.2
    retry_count = int(feedback.get("retry_count", 0) or 0)
    if retry_count > 0:
        penalty += min(retry_count, 5) * 0.1
    if bool(feedback.get("previous_failure")):
        penalty += 0.2
    return min(penalty, 1.0)


def analyze_complexity(input_text: str, runtime_feedback: Mapping[str, object] | None = None) -> float:
    """Return a normalized complexity score in the range [0.0, 1.0]."""
    text = str(input_text or "")
    length_score = min(len(text) / 500.0, 1.0)
    base_score = length_score

    if _CODE_RE.search(text):
        base_score += 0.3
    if _LOGIC_RE.search(text):
        base_score += 0.2

    feedback = dict(runtime_feedback or {})
    base_score += _runtime_feedback_penalty(feedback)

    return min(base_score, 1.0)


__all__ = ["analyze_complexity"]
