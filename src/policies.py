"""Policy abstractions for proposer/reviewer agents and RL updaters."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import random
from typing import Protocol


class TextPolicy(Protocol):
    """Minimal interface required by the dialogue runtime."""

    role: str

    def generate(self, context: str) -> str:
        """Generate one text response."""

    def sample_candidates(self, context: str, count: int) -> list[str]:
        """Generate candidate responses used by an RL updater."""


@dataclass
class MockPolicy:
    """Small deterministic policy used by tests and README smoke runs."""

    role: str
    answers: list[int] = field(default_factory=lambda: [42, 12, 5, 18])
    cursor: int = 0

    def generate(self, context: str) -> str:
        if self.role == "reviewer":
            return "RIGHT\nThe pending answer is acceptable."
        answer = self.answers[self.cursor % len(self.answers)]
        self.cursor += 1
        return f"Reasoning: compute the requested quantity.\nFinal answer: {answer}"

    def sample_candidates(self, context: str, count: int) -> list[str]:
        return [self.generate(context) for _ in range(max(1, int(count)))]


class TransformersPolicy:
    """Lazy Transformers-backed text policy.

    The class is intentionally thin. It provides public-repo readability and a
    real generation path while keeping heavy model setup outside smoke tests.
    """

    def __init__(self, role: str, model_name: str, max_new_tokens: int, temperature: float):
        self.role = role
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, context: str) -> str:
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        prompt = build_prompt(self.role, context)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-6),
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def sample_candidates(self, context: str, count: int) -> list[str]:
        return [self.generate(context) for _ in range(max(1, int(count)))]


class PolicyUpdater(Protocol):
    """Interface for GSPO or custom policy-learning algorithms."""

    def update(self, policy: TextPolicy, context: str, candidates: list[str], rewards: list[float]) -> None:
        """Update a policy from candidates and rewards."""


class NoOpUpdater:
    """Updater used for mock runs and as a safe custom baseline."""

    def update(self, policy: TextPolicy, context: str, candidates: list[str], rewards: list[float]) -> None:
        return None


class GSPOUpdater(NoOpUpdater):
    """GSPO-compatible hook.

    Full GSPO fine-tuning can be plugged in here. The cleaned repository keeps
    the interface explicit while allowing mock and no-op runs to execute without
    downloading large models.
    """


def load_custom_updater(import_path: str) -> PolicyUpdater:
    module_name, object_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    updater = getattr(module, object_name)
    return updater() if isinstance(updater, type) else updater


def build_updater(algorithm: str, custom_path: str | None = None) -> PolicyUpdater:
    if algorithm == "custom":
        if not custom_path:
            raise ValueError("custom_updater must be set when rl_algorithm='custom'")
        return load_custom_updater(custom_path)
    if algorithm == "gspo":
        return GSPOUpdater()
    if algorithm in {"none", "noop"}:
        return NoOpUpdater()
    raise ValueError(f"Unsupported rl_algorithm: {algorithm}")


def resolve_model_name(model_key: str) -> str:
    aliases = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    return aliases.get(model_key, model_key)


def build_policy(role: str, backend: str, model_key: str, max_new_tokens: int, temperature: float) -> TextPolicy:
    if backend == "mock":
        return MockPolicy(role=role)
    if backend == "transformers":
        return TransformersPolicy(
            role=role,
            model_name=resolve_model_name(model_key),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported policy_backend: {backend}")


def build_prompt(role: str, context: str) -> str:
    if role == "reviewer":
        return (
            "Check whether the current pending answer is correct.\n"
            "First line: RIGHT or WRONG.\n"
            "Second line: one short reason.\n\n"
            f"{context}\n\nReview:\n"
        )
    return (
        "Solve the original problem directly.\n"
        "End with exactly one final line: Final answer: <answer>.\n\n"
        f"{context}\n\nSolution:\n"
    )
