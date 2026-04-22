"""Lightweight semantic similarity helpers for gateway fast-path caching."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping


_TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_CANONICAL_TOKENS = {
    "hi": "greeting",
    "hello": "greeting",
    "hey": "greeting",
    "yo": "greeting",
    "thanks": "gratitude",
    "thank": "gratitude",
    "thx": "gratitude",
    "bye": "farewell",
    "goodbye": "farewell",
}


def vectorize_text(text: str) -> dict[str, float]:
    """Build a sparse semantic vector from normalized tokens and char trigrams."""
    raw = str(text or "").strip().lower()
    if not raw:
        return {}

    vector: Counter[str] = Counter()
    tokens = _TOKEN_RE.findall(raw)
    normalized_tokens = [_CANONICAL_TOKENS.get(tok, tok) for tok in tokens]
    for token in normalized_tokens:
        vector[f"tok:{token}"] += 1.0

    joined = " ".join(normalized_tokens) or raw
    padded = f"  {joined}  "
    for idx in range(max(len(padded) - 2, 0)):
        trigram = padded[idx:idx + 3]
        vector[f"tri:{trigram}"] += 0.2

    return dict(vector)


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return cosine similarity for two sparse vectors."""
    if not left or not right:
        return 0.0

    dot = 0.0
    for key, value in left.items():
        dot += float(value) * float(right.get(key, 0.0))

    left_norm = math.sqrt(sum(float(value) * float(value) for value in left.values()))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


__all__ = ["cosine_similarity", "vectorize_text"]


def sim(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for l_value, r_value in zip(left, right):
        dot += float(l_value) * float(r_value)
        left_norm += float(l_value) * float(l_value)
        right_norm += float(r_value) * float(r_value)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def semantic_lookup(
    query_vec: list[float],
    cache: list[dict[str, object]],
    threshold: float = 0.9,
) -> str | None:
    for item in cache:
        embedding = item.get("embedding")
        response = item.get("response")
        if isinstance(embedding, list) and isinstance(response, str):
            if sim(query_vec, embedding) > threshold:
                return response
    return None


__all__ = ["cosine_similarity", "semantic_lookup", "sim", "vectorize_text"]
