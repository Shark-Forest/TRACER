"""Action-value helpers for proposal-review controllers."""

from __future__ import annotations

import math
from typing import Any


def coerce_number(value: Any) -> float | None:
    """Return a float for numeric answers, or None for invalid answers."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).replace(",", "").replace("$", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def answers_match(prediction: Any, ground_truth: Any, tolerance: float = 1e-3) -> bool:
    """Compare numeric answers with a small tolerance."""

    pred = coerce_number(prediction)
    gold = coerce_number(ground_truth)
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tolerance


def pending_answer_skip_value(pending_answer: Any, ground_truth: Any) -> float:
    """Value of skipping reviewer speech based on the current pending answer."""

    if coerce_number(pending_answer) is None:
        return 0.0
    return 1.0 if answers_match(pending_answer, ground_truth) else -1.0


def signed_answer_value(answer: Any, ground_truth: Any) -> float:
    """Signed value for an answer-producing action."""

    if coerce_number(answer) is None:
        return 0.0
    return 1.0 if answers_match(answer, ground_truth) else -1.0
