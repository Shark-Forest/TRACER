"""Dataset loading and answer extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

from .values import coerce_number, normalize_answer_text


NUMBER_PATTERN = re.compile(r"(?<![\w/.-])-?\$?\d[\d,]*\.?\d*")
BOXED_PATTERN = re.compile(r"\\boxed\s*{")
CHOICE_PATTERN = re.compile(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", re.I)
CHOICE_LETTERS = "ABCD"


@dataclass(frozen=True)
class Sample:
    """One question-answer item used by the trainer."""

    question: str
    answer: object
    source: str = "unknown"
    dataset_name: str = "gsm8k"


@dataclass(frozen=True)
class DatasetSpec:
    """External dataset reference from Hugging Face or ModelScope."""

    provider: str
    name: str
    config: str | None
    raw: str


EXTERNAL_DATASET_PREFIXES = {
    "hf:": "hf",
    "huggingface:": "hf",
    "modelscope:": "modelscope",
    "ms:": "modelscope",
}


BUILTIN_GSM8K = [
    Sample("If a box has 40 apples and Mary adds 2, how many apples are there?", 42, "builtin_gsm8k", "gsm8k"),
    Sample("Tom has 5 marbles and buys 7 more. How many marbles does he have?", 12, "builtin_gsm8k", "gsm8k"),
    Sample("A train has 9 cars and leaves 4 behind. How many cars remain?", 5, "builtin_gsm8k", "gsm8k"),
    Sample("There are 3 bags with 6 coins each. How many coins are there?", 18, "builtin_gsm8k", "gsm8k"),
]

BUILTIN_MATH500 = [
    Sample("Compute 1 + 1.", 2, "builtin_math500", "math500"),
    Sample("Simplify \\frac{1}{2} + \\frac{1}{2}.", 1, "builtin_math500", "math500"),
]

BUILTIN_GPQA_DIAMOND = [
    Sample(
        "Which option is the only prime number?\nChoices:\nA. 21\nB. 27\nC. 29\nD. 35",
        "C",
        "builtin_gpqa_diamond",
        "gpqa-diamond",
    ),
    Sample(
        "Which option is a noble gas?\nChoices:\nA. Oxygen\nB. Helium\nC. Nitrogen\nD. Chlorine",
        "B",
        "builtin_gpqa_diamond",
        "gpqa-diamond",
    ),
]


def parse_external_dataset_spec(name: str | None) -> DatasetSpec | None:
    """Parse `hf:repo::config` or `modelscope:repo::subset` dataset specs."""

    if not name:
        return None
    raw = str(name).strip()
    lower = raw.lower()
    provider = None
    body = raw
    for prefix, canonical in EXTERNAL_DATASET_PREFIXES.items():
        if lower.startswith(prefix):
            provider = canonical
            body = raw[len(prefix) :]
            break
    if provider is None and "/" in raw and not raw.startswith((".", "/")):
        provider = "hf"
    if provider is None:
        return None
    dataset_name, config = (body.split("::", 1) + [None])[:2] if "::" in body else (body, None)
    dataset_name = dataset_name.strip()
    config = config.strip() if isinstance(config, str) and config.strip() else None
    if not dataset_name:
        raise ValueError(f"Invalid dataset spec '{name}'.")
    return DatasetSpec(provider=provider, name=dataset_name, config=config, raw=raw)


def _has_external_prefix(value: str) -> bool:
    lower = value.lower()
    return any(lower.startswith(prefix) for prefix in EXTERNAL_DATASET_PREFIXES)


def normalize_dataset_name(name: str | None) -> str:
    """Normalize built-in names while preserving external/custom dataset specs."""

    if not name:
        return "gsm8k"
    raw = str(name).strip()
    if _has_external_prefix(raw):
        external = parse_external_dataset_spec(raw)
        assert external is not None
        return external.raw
    compact = re.sub(r"[^a-z0-9]", "", raw.lower())
    if "gsm8k" in compact:
        return "gsm8k"
    if "math500" in compact:
        return "math500"
    if "gpqadiamond" in compact:
        return "gpqa-diamond"
    external = parse_external_dataset_spec(raw)
    if external is not None:
        return external.raw
    if raw.lower().startswith("custom:"):
        return raw
    return f"custom:{raw}"


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _extract_numeric_answer(text: str | None) -> float | None:
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


def _extract_braced_argument(text: str, open_brace_index: int) -> str | None:
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index + 1 : index]
    return None


def _last_boxed_content(text: str) -> str | None:
    boxed = None
    for match in BOXED_PATTERN.finditer(text):
        content = _extract_braced_argument(text, match.end() - 1)
        if content is not None:
            boxed = content
    return boxed


def _strip_outer_braces(text: str) -> str:
    value = text.strip()
    while value.startswith("{") and value.endswith("}"):
        inner = _extract_braced_argument(value, 0)
        if inner is None or len(inner) != len(value) - 2:
            break
        value = inner.strip()
    return value


def _normalize_math_answer(value: object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    boxed = _last_boxed_content(text)
    if boxed is not None:
        text = boxed
    text = re.sub(r"^(?:final\s+answer|answer)\s*[:\-]\s*", "", text, flags=re.I)
    text = _strip_outer_braces(text.strip().strip("."))
    normalized = normalize_answer_text(text)
    return normalized if normalized else None


def _return_math_answer(value: object) -> object | None:
    normalized = _normalize_math_answer(value)
    if normalized is None:
        return None
    number = coerce_number(normalized)
    return number if number is not None else normalized


def _extract_math500_answer(text: str | None) -> object | None:
    if not text:
        return None
    raw = str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if "final answer" in lower or "answer" in lower or "\\boxed" in line:
            answer = _return_math_answer(line)
            if answer is not None:
                return answer
    boxed = _last_boxed_content(raw)
    if boxed is not None:
        answer = _return_math_answer(boxed)
        if answer is not None:
            return answer
    return _extract_numeric_answer(raw)


def _extract_gpqa_answer(text: str | None) -> str | None:
    if not text:
        return None
    raw = str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    final_segments = [
        line for line in reversed(lines) if "final answer" in line.lower() or "answer" in line.lower()
    ]
    for segment in final_segments + [raw]:
        match = re.search(
            r"(?:final\s+answer|answer)\s*[:\-]?\s*(?:[([]\s*)?([A-D])(?:\s*[)\]])?",
            segment,
            re.I,
        )
        if match:
            return match.group(1).upper()
    for line in reversed(lines):
        matches = list(CHOICE_PATTERN.finditer(line))
        if matches:
            return matches[-1].group(1).upper()
    return None


def extract_answer(text: str | None, dataset_name: str | None = "gsm8k") -> object | None:
    """Extract a model answer using the dataset-specific convention."""

    dataset = normalize_dataset_name(dataset_name)
    if dataset == "math500":
        return _extract_math500_answer(text)
    if dataset == "gpqa-diamond":
        return _extract_gpqa_answer(text)
    return _extract_numeric_answer(text)


def extract_final_answer(text: str | None) -> float | None:
    """Extract the GSM8K-style final numeric answer from a model response."""

    answer = extract_answer(text, "gsm8k")
    return coerce_number(answer)


def _limit(samples: Iterable[Sample], limit: int | None) -> list[Sample]:
    values = list(samples)
    if limit is None:
        return values
    return values[: max(0, int(limit))]


def _generic_sample_from_row(row: Mapping[str, Any], source: str, dataset_name: str) -> Sample | None:
    question = _first_present(row, "question", "Question", "problem", "prompt", "input", "query")
    answer = _first_present(row, "answer", "Answer", "ground_truth", "target", "label", "output")
    if question is None or answer is None:
        return None
    return Sample(str(question), answer, source, dataset_name)


def _load_jsonl(path: Path, limit: int | None) -> list[Sample]:
    samples: list[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dataset_name = normalize_dataset_name(
            _first_present(row, "dataset_name", "dataset") or "gsm8k"
        )
        sample = _generic_sample_from_row(row, str(path), dataset_name)
        if sample is not None:
            samples.append(sample)
    return _limit(samples, limit)


def _load_gsm8k_from_datasets(split: str, limit: int | None) -> list[Sample]:
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is not installed") from exc
    dataset = hf_load_dataset("gsm8k", "main", split=split)
    samples = []
    for row in dataset:
        answer_text = str(row.get("answer", ""))
        answer = extract_answer(answer_text, "gsm8k")
        samples.append(Sample(str(row["question"]), answer, "gsm8k", "gsm8k"))
    return _limit(samples, limit)


def _load_math500_from_datasets(split: str, limit: int | None) -> list[Sample]:
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is not installed") from exc
    split_options = [split]
    if split != "test":
        split_options.append("test")
    last_error: Exception | None = None
    for hf_split in split_options:
        try:
            dataset = hf_load_dataset("HuggingFaceH4/MATH-500", split=hf_split)
            break
        except Exception as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    samples = []
    for row in dataset:
        question = _first_present(row, "problem", "question", "prompt")
        answer_raw = _first_present(row, "answer", "ground_truth", "solution")
        if question is None or answer_raw is None:
            continue
        answer = _return_math_answer(answer_raw)
        samples.append(Sample(str(question), answer if answer is not None else answer_raw, "math500", "math500"))
    return _limit(samples, limit)


def _format_gpqa_sample(row: Mapping[str, Any]) -> Sample | None:
    question = _first_present(row, "Question", "question", "prompt")
    correct = _first_present(row, "Correct Answer", "correct_answer", "answer", "ground_truth")
    if question is None or correct is None:
        return None
    incorrects = [
        _first_present(row, "Incorrect Answer 1", "incorrect_answer_1"),
        _first_present(row, "Incorrect Answer 2", "incorrect_answer_2"),
        _first_present(row, "Incorrect Answer 3", "incorrect_answer_3"),
    ]
    if all(choice is not None for choice in incorrects):
        choices = [(str(correct), True)] + [(str(choice), False) for choice in incorrects]
        seed = int(hashlib.sha256(str(question).encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        rng.shuffle(choices)
        lines = [str(question), "Choices:"]
        answer_letter = "A"
        for index, (choice, is_correct) in enumerate(choices):
            letter = CHOICE_LETTERS[index]
            lines.append(f"{letter}. {choice}")
            if is_correct:
                answer_letter = letter
        return Sample("\n".join(lines), answer_letter, "gpqa-diamond", "gpqa-diamond")
    answer = extract_answer(str(correct), "gpqa-diamond") or correct
    return Sample(str(question), answer, "gpqa-diamond", "gpqa-diamond")


def _load_gpqa_diamond_from_datasets(split: str, limit: int | None) -> list[Sample]:
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is not installed") from exc
    split_options = [split]
    if split != "train":
        split_options.append("train")
    last_error: Exception | None = None
    for hf_split in split_options:
        try:
            dataset = hf_load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=hf_split)
            break
        except Exception as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    samples = []
    for row in dataset:
        sample = _format_gpqa_sample(row)
        if sample is not None:
            samples.append(sample)
    return _limit(samples, limit)


def _load_external_huggingface(spec: DatasetSpec, split: str, limit: int | None) -> list[Sample]:
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is not installed") from exc
    if spec.config:
        dataset = hf_load_dataset(spec.name, spec.config, split=split)
    else:
        dataset = hf_load_dataset(spec.name, split=split)
    samples = [
        sample
        for row in dataset
        if (sample := _generic_sample_from_row(row, spec.raw, spec.raw)) is not None
    ]
    return _limit(samples, limit)


def _load_external_modelscope(spec: DatasetSpec, split: str, limit: int | None) -> list[Sample]:
    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("modelscope is not installed") from exc
    dataset = MsDataset.load(spec.name, subset_name=spec.config, split=split)
    samples = [
        sample
        for row in dataset
        if (sample := _generic_sample_from_row(row, spec.raw, spec.raw)) is not None
    ]
    return _limit(samples, limit)


def _load_external_dataset(spec: DatasetSpec, split: str, limit: int | None) -> list[Sample]:
    if spec.provider == "hf":
        return _load_external_huggingface(spec, split, limit)
    if spec.provider == "modelscope":
        return _load_external_modelscope(spec, split, limit)
    raise ValueError(f"Unsupported dataset provider '{spec.provider}'.")


def load_dataset(name: str, split: str, limit: int | None = None) -> list[Sample]:
    """Load a built-in, external Hub/ModelScope, or JSONL dataset."""

    value = str(name or "gsm8k")
    path = Path(value)
    if path.exists():
        return _load_jsonl(path, limit)

    dataset_name = normalize_dataset_name(value)
    external = parse_external_dataset_spec(dataset_name)
    if external is not None:
        return _load_external_dataset(external, "train" if split == "train" else "test", limit)

    hf_split = "train" if split == "train" else "test"
    if dataset_name == "gsm8k":
        try:
            return _load_gsm8k_from_datasets(hf_split, limit)
        except Exception:
            return _limit(BUILTIN_GSM8K, limit)
    if dataset_name == "math500":
        try:
            return _load_math500_from_datasets(hf_split, limit)
        except Exception:
            return _limit(BUILTIN_MATH500, limit)
    if dataset_name == "gpqa-diamond":
        try:
            return _load_gpqa_diamond_from_datasets(hf_split, limit)
        except Exception:
            return _limit(BUILTIN_GPQA_DIAMOND, limit)
    raise ValueError(
        f"Unsupported dataset '{name}'. Use a built-in name, JSONL path, hf:repo::config, or modelscope:repo::subset."
    )
