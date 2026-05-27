"""Metric helpers for training and evaluation outputs."""

from __future__ import annotations

from .values import answers_match


def accuracy(records: list[dict[str, object]]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for record in records if bool(record.get("correct")))
    return correct / len(records)


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "num_samples": len(records),
        "accuracy": accuracy(records),
        "records": records,
    }
