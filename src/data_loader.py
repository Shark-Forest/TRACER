"""Dataset loading and answer extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from .values import coerce_number


NUMBER_PATTERN = re.compile(r"(?<![\w/.-])-?\$?\d[\d,]*\.?\d*")


@dataclass(frozen=True)
class Sample:
    """One question-answer item used by the trainer."""

    question: str
    answer: object
    source: str = "unknown"


BUILTIN_GSM8K = [
    Sample("If a box has 40 apples and Mary adds 2, how many apples are there?", 42, "builtin_gsm8k"),
    Sample("Tom has 5 marbles and buys 7 more. How many marbles does he have?", 12, "builtin_gsm8k"),
    Sample("A train has 9 cars and leaves 4 behind. How many cars remain?", 5, "builtin_gsm8k"),
    Sample("There are 3 bags with 6 coins each. How many coins are there?", 18, "builtin_gsm8k"),
]


def extract_final_answer(text: str | None) -> float | None:
    """Extract the last numeric answer from a model response."""

    if not text:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        if "final answer" in line.lower() or "answer" in line.lower():
            numbers = NUMBER_PATTERN.findall(line)
            if numbers:
                return coerce_number(numbers[-1])
    numbers = NUMBER_PATTERN.findall(str(text))
    return coerce_number(numbers[-1]) if numbers else None


def _limit(samples: Iterable[Sample], limit: int | None) -> list[Sample]:
    values = list(samples)
    if limit is None:
        return values
    return values[: max(0, int(limit))]


def _load_jsonl(path: Path, limit: int | None) -> list[Sample]:
    samples: list[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = row.get("question") or row.get("prompt")
        answer = row.get("answer") or row.get("ground_truth")
        if question is None or answer is None:
            continue
        samples.append(Sample(str(question), answer, str(path)))
    return _limit(samples, limit)


def _load_gsm8k_from_datasets(split: str, limit: int | None) -> list[Sample]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is not installed") from exc
    dataset = load_dataset("gsm8k", "main", split=split)
    samples = []
    for row in dataset:
        answer_text = str(row.get("answer", ""))
        answer = extract_final_answer(answer_text)
        samples.append(Sample(str(row["question"]), answer, "gsm8k"))
    return _limit(samples, limit)


def load_dataset(name: str, split: str, limit: int | None = None) -> list[Sample]:
    """Load a dataset by name or JSONL path.

    `gsm8k` uses Hugging Face datasets when available and otherwise falls back
    to a tiny built-in sample set so smoke tests remain model-free.
    """

    value = str(name or "gsm8k")
    path = Path(value)
    if path.exists():
        return _load_jsonl(path, limit)
    if value.lower() != "gsm8k":
        raise ValueError(f"Unsupported dataset '{name}'. Use 'gsm8k' or a JSONL path.")
    try:
        return _load_gsm8k_from_datasets("train" if split == "train" else "test", limit)
    except Exception:
        return _limit(BUILTIN_GSM8K, limit)
