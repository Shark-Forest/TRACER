"""Action-value helpers for proposal-review controllers."""

from __future__ import annotations

import math
import re
from typing import Any


FRAC_PATTERN = re.compile(r"\\(?:d|t)?frac\s*{([^{}]+)}\s*{([^{}]+)}")


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


def normalize_answer_text(value: Any) -> str | None:
    """Normalize non-numeric answers for exact string comparison."""

    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "").replace("\\,", "").replace("\\!", "")
    while True:
        updated = FRAC_PATTERN.sub(r"\1/\2", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"^(?:final\s+answer|answer)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.strip().strip(".")
    text = re.sub(r"\s+", "", text)
    return text.lower() if text else None


def is_valid_answer(value: Any) -> bool:
    """Return whether an answer is parseable as numeric or normalized text."""

    return coerce_number(value) is not None or normalize_answer_text(value) is not None


def answers_match(prediction: Any, ground_truth: Any, tolerance: float = 1e-3) -> bool:
    """Compare numeric answers first, then normalized exact text answers."""

    pred = coerce_number(prediction)
    gold = coerce_number(ground_truth)
    if pred is not None and gold is not None:
        return abs(pred - gold) <= tolerance
    pred_text = normalize_answer_text(prediction)
    gold_text = normalize_answer_text(ground_truth)
    return pred_text is not None and pred_text == gold_text


def pending_answer_skip_value(pending_answer: Any, ground_truth: Any) -> float:
    """Value of skipping reviewer speech based on the current pending answer."""

    if not is_valid_answer(pending_answer):
        return -1.0
    return 1.0 if answers_match(pending_answer, ground_truth) else -1.0


def signed_answer_value(answer: Any, ground_truth: Any) -> float:
    """Signed value for an answer-producing action."""

    if not is_valid_answer(answer):
        return -1.0
    return 1.0 if answers_match(answer, ground_truth) else -1.0
