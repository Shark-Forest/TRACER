"""Policy abstractions for proposer/reviewer agents and RL updaters."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
from typing import Any, Protocol

from .customization import load_callable
from .data_loader import normalize_dataset_name


_GSPO_LOG_RATIO_CLAMP = 2.0


def _modelscope_model_cache_dir() -> str | None:
    return (
        os.environ.get("TRACER_MODELSCOPE_MODEL_CACHE_DIR")
        or os.environ.get("TRACER_MODELSCOPE_CACHE_DIR")
        or os.environ.get("MODELSCOPE_CACHE")
    )


def _download_modelscope_snapshot(model_id: str, cache_dir: str | None = None) -> str:
    from modelscope import snapshot_download  # type: ignore

    return str(snapshot_download(model_id=model_id, cache_dir=cache_dir))


def resolve_model_source(model_name: str) -> str:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return str(local_path)
    try:
        return _download_modelscope_snapshot(
            model_name,
            cache_dir=_modelscope_model_cache_dir(),
        )
    except Exception:
        return model_name


def _env_value(primary: str, fallback: str, default: str) -> str:
    return os.environ.get(primary, os.environ.get(fallback, default))


def _env_float(primary: str, fallback: str, default: float) -> float:
    return float(_env_value(primary, fallback, str(default)) or default)


def _env_int(primary: str, fallback: str, default: int) -> int:
    return int(_env_value(primary, fallback, str(default)) or default)


def _env_bool(primary: str, fallback: str, default: bool) -> bool:
    raw = _env_value(primary, fallback, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_finetune_mode(value: str | None) -> str:
    mode = str(value or "lora").strip().lower()
    aliases = {
        "lora": "lora",
        "last_n_layers": "last_n_layers",
        "last_layers": "last_n_layers",
        "partial": "last_n_layers",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported GSPO finetune mode: {value}")
    return aliases[mode]


def _parse_target_module_names(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _infer_default_lora_target_modules(model_type: str, available_suffixes: set[str]) -> list[str]:
    if model_type == "phi3":
        ordered = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif model_type.startswith("qwen2"):
        ordered = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        ordered = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
            "c_attn",
            "c_proj",
            "c_fc",
        ]
    resolved = [name for name in ordered if name in available_suffixes]
    if not resolved:
        raise ValueError(
            f"Could not infer LoRA target modules for model_type={model_type or '<unknown>'}. "
            "Set TRACER_GSPO_LORA_TARGET_MODULES."
        )
    return resolved


class TextPolicy(Protocol):
    """Minimal interface required by the dialogue runtime."""

    role: str
    agent_index: int

    def generate(self, context: str, dataset_name: str = "gsm8k") -> str:
        """Generate one text response."""

    def sample_candidates(self, context: str, count: int, dataset_name: str = "gsm8k") -> list[str]:
        """Generate candidate responses used by an RL updater."""


@dataclass
class MockPolicy:
    """Small deterministic policy used by tests and README smoke runs."""

    role: str
    agent_index: int = 0
    answers: list[int] = field(default_factory=lambda: [42, 12, 5, 18])
    choice_answers: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    cursor: int = 0
    last_update: dict[str, Any] | None = None

    def generate(self, context: str, dataset_name: str = "gsm8k") -> str:
        dataset = normalize_dataset_name(dataset_name)
        if self.role == "reviewer":
            return "RIGHT\nThe pending answer is acceptable."
        if dataset == "gpqa-diamond":
            answer = self.choice_answers[self.cursor % len(self.choice_answers)]
            self.cursor += 1
            return f"Reasoning: eliminate inconsistent choices.\nFinal answer: {answer}"
        answer = self.answers[self.cursor % len(self.answers)]
        self.cursor += 1
        if dataset == "math500":
            return f"Reasoning: compute the requested quantity.\nFinal answer: \\boxed{{{answer}}}"
        return f"Reasoning: compute the requested quantity.\nFinal answer: {answer}"

    def sample_candidates(self, context: str, count: int, dataset_name: str = "gsm8k") -> list[str]:
        return [self.generate(context, dataset_name=dataset_name) for _ in range(max(1, int(count)))]


class TransformersPolicy:
    """Transformers-backed text policy with a GSPO update path.

    The GSPO implementation follows the MAS/final update structure: sample a
    group of candidates, normalize rewards into group-relative advantages,
    compute sequence-level importance ratios, and optimize a clipped objective.
    """

    def __init__(
        self,
        role: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        custom_prompt_function: str | None = None,
        agent_index: int = 0,
    ):
        self.role = role
        self.agent_index = int(agent_index)
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.custom_prompt_function = custom_prompt_function
        self.last_update: dict[str, Any] | None = None
        self._tokenizer = None
        self._model = None
        self._optimizer = None
        self._trainable_params = None
        self._gspo_configured = False
        self._lora_target_modules: list[str] = []
        self._base_model_type = ""

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def load_from_source(model_source: str):
            tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            return tokenizer, model

        model_source = resolve_model_source(self.model_name)
        try:
            self._tokenizer, self._model = load_from_source(model_source)
        except Exception:
            if model_source == self.model_name:
                raise
            self._tokenizer, self._model = load_from_source(self.model_name)
        self._base_model_type = str(getattr(getattr(self._model, "config", None), "model_type", "") or "").lower()
        self._model.eval()

    def _model_device(self):
        assert self._model is not None
        device = getattr(self._model, "device", None)
        if device is not None:
            return device
        return next(self._model.parameters()).device

    def _prompt(self, context: str, dataset_name: str) -> str:
        return build_prompt(
            self.role,
            context,
            dataset_name=dataset_name,
            custom_prompt_function=self.custom_prompt_function,
            agent_index=self.agent_index,
        )

    def generate(self, context: str, dataset_name: str = "gsm8k") -> str:
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        prompt = self._prompt(context, dataset_name)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model_device())
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-6),
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def sample_candidates(self, context: str, count: int, dataset_name: str = "gsm8k") -> list[str]:
        return [self.generate(context, dataset_name=dataset_name) for _ in range(max(1, int(count)))]

    def _resolve_lora_target_modules(self) -> list[str]:
        assert self._model is not None
        override = _parse_target_module_names(
            os.environ.get("TRACER_GSPO_LORA_TARGET_MODULES")
            or os.environ.get("MAS_GSPO_LORA_TARGET_MODULES")
        )
        available_suffixes = {name.rsplit(".", 1)[-1] for name, _ in self._model.named_modules() if name}
        if override:
            missing = [name for name in override if name not in available_suffixes]
            if missing:
                raise ValueError("LoRA target modules not found: " + ", ".join(missing))
            return override
        return _infer_default_lora_target_modules(self._base_model_type, available_suffixes)

    def _configure_last_layer_training(self) -> list[Any]:
        assert self._model is not None
        trainable = []
        last_n = _env_int("TRACER_GSPO_TRAIN_LAST_N_LAYERS", "MAS_GSPO_TRAIN_LAST_N_LAYERS", 2)
        num_hidden_layers = int(getattr(self._model.config, "num_hidden_layers", 0) or 0)
        last_layer_start = max(0, num_hidden_layers - last_n)
        for name, param in self._model.named_parameters():
            should_train = False
            if name.startswith("lm_head"):
                should_train = True
            elif name.endswith("norm.weight") or name.endswith("norm.bias"):
                should_train = True
            elif ".layers." in name:
                parts = name.split(".layers.", 1)[1].split(".", 1)
                if parts and parts[0].isdigit():
                    should_train = int(parts[0]) >= last_layer_start
            elif ".h." in name:
                parts = name.split(".h.", 1)[1].split(".", 1)
                if parts and parts[0].isdigit():
                    should_train = int(parts[0]) >= last_layer_start
            param.requires_grad_(should_train)
            if should_train:
                trainable.append(param)
        if not trainable:
            raise RuntimeError("No trainable parameters found for GSPO last_n_layers mode.")
        return trainable

    def _ensure_gspo_training(self) -> None:
        if self._gspo_configured:
            return
        self._ensure_model()
        assert self._model is not None
        import torch

        mode = _normalize_finetune_mode(
            os.environ.get("TRACER_GSPO_FINETUNE_MODE")
            or os.environ.get("MAS_GSPO_FINETUNE_MODE")
            or "lora"
        )
        if mode == "lora":
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("GSPO LoRA mode requires peft to be installed.") from exc
            self._lora_target_modules = self._resolve_lora_target_modules()
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=_env_int("TRACER_GSPO_LORA_R", "MAS_GSPO_LORA_R", 8),
                lora_alpha=_env_int("TRACER_GSPO_LORA_ALPHA", "MAS_GSPO_LORA_ALPHA", 16),
                lora_dropout=_env_float("TRACER_GSPO_LORA_DROPOUT", "MAS_GSPO_LORA_DROPOUT", 0.0),
                bias=_env_value("TRACER_GSPO_LORA_BIAS", "MAS_GSPO_LORA_BIAS", "none"),
                target_modules=self._lora_target_modules,
            )
            self._model = get_peft_model(self._model, lora_config)
            trainable = [param for param in self._model.parameters() if param.requires_grad]
        else:
            trainable = self._configure_last_layer_training()
        if not trainable:
            raise RuntimeError("No trainable parameters found for GSPO update.")
        self._trainable_params = trainable
        self._optimizer = torch.optim.Adam(
            self._trainable_params,
            lr=_env_float("TRACER_GSPO_LR", "MAS_GSPO_LR", 5e-6),
        )
        self._model.eval()
        self._gspo_configured = True

    def _build_gspo_batch(self, context: str, candidates: list[str], dataset_name: str):
        self._ensure_model()
        assert self._tokenizer is not None
        import torch

        prompt = self._prompt(context, dataset_name)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        prompt_len = int(prompt_ids.shape[1])
        encoded = []
        eos_id = self._tokenizer.eos_token_id
        for candidate in candidates:
            answer_ids = self._tokenizer(str(candidate), return_tensors="pt", add_special_tokens=False)["input_ids"]
            if int(answer_ids.shape[1]) == 0:
                filler = eos_id if eos_id is not None else self._tokenizer.pad_token_id
                if filler is None:
                    filler = 0
                answer_ids = torch.tensor([[int(filler)]], dtype=prompt_ids.dtype)
            input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            attention_mask = torch.ones_like(input_ids)
            encoded.append((input_ids[0], attention_mask[0], labels[0]))

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0
        max_len = max(int(item[0].shape[0]) for item in encoded)
        input_batch = torch.full((len(encoded), max_len), int(pad_id), dtype=encoded[0][0].dtype)
        mask_batch = torch.zeros((len(encoded), max_len), dtype=encoded[0][1].dtype)
        label_batch = torch.full((len(encoded), max_len), -100, dtype=encoded[0][2].dtype)
        for row, (input_ids, attention_mask, labels) in enumerate(encoded):
            seq_len = int(input_ids.shape[0])
            input_batch[row, :seq_len] = input_ids
            mask_batch[row, :seq_len] = attention_mask
            label_batch[row, :seq_len] = labels
        return input_batch, mask_batch, label_batch

    def _sequence_log_prob(self, input_ids, attention_mask, labels):
        assert self._model is not None
        import torch

        device = self._model_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        valid_mask = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * valid_mask
        seq_log_prob = token_log_probs.sum(dim=-1)
        seq_len = valid_mask.sum(dim=-1).clamp(min=1)
        return seq_log_prob, seq_len

    def gspo_update(
        self,
        context: str,
        candidates: list[str],
        rewards: list[float],
        dataset_name: str = "gsm8k",
    ) -> dict[str, Any]:
        if len(candidates) != len(rewards):
            raise ValueError("GSPO candidates and rewards must have the same length.")
        if not candidates:
            self.last_update = {"skipped_update": True, "skip_reason": "empty_candidates"}
            return self.last_update

        self._ensure_gspo_training()
        assert self._model is not None and self._optimizer is not None and self._trainable_params is not None
        import torch

        self._model.eval()
        rewards_tensor = torch.tensor([float(value) for value in rewards], dtype=torch.float32, device=self._model_device())
        rewards_tensor = torch.nan_to_num(rewards_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        avg_reward = float(rewards_tensor.mean().detach().cpu().item())
        best_idx = int(torch.argmax(rewards_tensor).detach().cpu().item())
        reward_std = float(rewards_tensor.std(unbiased=False).detach().cpu().item())
        min_std = _env_float("TRACER_GSPO_MIN_REWARD_STD", "MAS_GSPO_MIN_REWARD_STD", 0.05)
        if _env_bool("TRACER_GSPO_SKIP_ZERO_SIGNAL_UPDATE", "MAS_GSPO_SKIP_ZERO_SIGNAL_UPDATE", True) and reward_std < min_std:
            self.last_update = {
                "avg_reward": avg_reward,
                "best_reward": float(rewards_tensor[best_idx].detach().cpu().item()),
                "best_text": candidates[best_idx],
                "candidates": list(candidates),
                "rewards": [float(value) for value in rewards],
                "skipped_update": True,
                "skip_reason": "zero_signal",
            }
            return self.last_update

        advantage = (rewards_tensor - rewards_tensor.mean()) / rewards_tensor.std(unbiased=False).clamp(min=1e-8)
        input_ids, attention_mask, labels = self._build_gspo_batch(context, candidates, dataset_name)
        with torch.no_grad():
            old_log_prob, old_seq_len = self._sequence_log_prob(input_ids, attention_mask, labels)
            old_log_prob = old_log_prob.detach()
            old_seq_len = old_seq_len.detach()

        self._optimizer.zero_grad(set_to_none=True)
        current_log_prob, _ = self._sequence_log_prob(input_ids, attention_mask, labels)
        log_ratio = (current_log_prob - old_log_prob.to(current_log_prob.device)) / old_seq_len.to(current_log_prob.device).clamp(min=1)
        log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=_GSPO_LOG_RATIO_CLAMP, neginf=-_GSPO_LOG_RATIO_CLAMP)
        log_ratio = log_ratio.clamp(min=-_GSPO_LOG_RATIO_CLAMP, max=_GSPO_LOG_RATIO_CLAMP)
        ratio = torch.exp(log_ratio)
        clip_eps = _env_float("TRACER_GSPO_CLIP_EPS", "MAS_GSPO_CLIP_EPS", 0.1)
        clipped_ratio = ratio.clamp(min=1.0 - clip_eps, max=1.0 + clip_eps)
        advantage = advantage.to(ratio.device)
        objective = torch.min(ratio * advantage, clipped_ratio * advantage)
        loss = -objective.mean()
        if not torch.isfinite(loss):
            self._optimizer.zero_grad(set_to_none=True)
            self.last_update = {
                "avg_reward": avg_reward,
                "best_reward": float(rewards_tensor[best_idx].detach().cpu().item()),
                "best_text": candidates[best_idx],
                "candidates": list(candidates),
                "rewards": [float(value) for value in rewards],
                "skipped_update": True,
                "skip_reason": "nonfinite_policy_loss",
            }
            return self.last_update
        loss.backward()
        max_grad_norm = _env_float("TRACER_GSPO_MAX_GRAD_NORM", "MAS_GSPO_MAX_GRAD_NORM", 0.5)
        grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_params, max_grad_norm)
        if not torch.isfinite(grad_norm):
            self._optimizer.zero_grad(set_to_none=True)
            self.last_update = {
                "avg_reward": avg_reward,
                "best_reward": float(rewards_tensor[best_idx].detach().cpu().item()),
                "best_text": candidates[best_idx],
                "candidates": list(candidates),
                "rewards": [float(value) for value in rewards],
                "skipped_update": True,
                "skip_reason": "nonfinite_grad_norm",
            }
            return self.last_update
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._model.eval()
        self.last_update = {
            "avg_reward": avg_reward,
            "best_reward": float(rewards_tensor[best_idx].detach().cpu().item()),
            "best_text": candidates[best_idx],
            "candidates": list(candidates),
            "rewards": [float(value) for value in rewards],
            "policy_loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item()) if hasattr(grad_norm, "detach") else float(grad_norm),
            "skipped_update": False,
        }
        return self.last_update


class PolicyUpdater(Protocol):
    """Interface for GSPO or custom policy-learning algorithms."""

    def update(
        self,
        policy: TextPolicy,
        context: str,
        candidates: list[str],
        rewards: list[float],
        dataset_name: str = "gsm8k",
    ) -> Any:
        """Update a policy from candidates and rewards."""


class NoOpUpdater:
    """Explicit updater that intentionally performs no policy learning."""

    def __init__(self):
        self.last_update: dict[str, Any] | None = None

    def update(
        self,
        policy: TextPolicy,
        context: str,
        candidates: list[str],
        rewards: list[float],
        dataset_name: str = "gsm8k",
    ) -> dict[str, Any]:
        self.last_update = {
            "skipped_update": True,
            "skip_reason": "noop_updater",
            "num_candidates": len(candidates),
        }
        return self.last_update


class GSPOUpdater:
    """Default inner-agent GSPO updater.

    Policies that implement `gspo_update` receive a real GSPO parameter update.
    Mock policies are kept as smoke-test policies and record an explicit skip.
    """

    def __init__(self):
        self.last_update: dict[str, Any] | None = None

    def update(
        self,
        policy: TextPolicy,
        context: str,
        candidates: list[str],
        rewards: list[float],
        dataset_name: str = "gsm8k",
    ) -> dict[str, Any]:
        update_fn = getattr(policy, "gspo_update", None)
        if callable(update_fn):
            self.last_update = update_fn(
                context=context,
                candidates=candidates,
                rewards=rewards,
                dataset_name=dataset_name,
            )
            return self.last_update
        self.last_update = {
            "skipped_update": True,
            "skip_reason": "policy_without_gspo_update",
            "policy_type": type(policy).__name__,
            "num_candidates": len(candidates),
        }
        if hasattr(policy, "last_update"):
            setattr(policy, "last_update", self.last_update)
        return self.last_update


def load_custom_updater(import_path: str) -> PolicyUpdater:
    module_name, object_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    updater = getattr(module, object_name)
    return updater() if isinstance(updater, type) else updater


def build_updater(algorithm: str, custom_path: str | None = None) -> PolicyUpdater:
    if custom_path:
        return load_custom_updater(custom_path)
    if algorithm == "custom":
        raise ValueError("custom_updater must be set when rl_algorithm='custom'")
    if algorithm == "gspo":
        return GSPOUpdater()
    if algorithm in {"none", "noop"}:
        return NoOpUpdater()
    raise ValueError(f"Unsupported rl_algorithm: {algorithm}. Set custom_updater for custom algorithms.")


def resolve_model_name(model_key: str) -> str:
    aliases = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "phi3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    }
    return aliases.get(model_key, model_key)


def build_policy(
    role: str,
    backend: str,
    model_key: str,
    max_new_tokens: int,
    temperature: float,
    custom_prompt_function: str | None = None,
    agent_index: int = 0,
) -> TextPolicy:
    if backend == "mock":
        return MockPolicy(role=role, agent_index=agent_index)
    if backend == "transformers":
        return TransformersPolicy(
            role=role,
            model_name=resolve_model_name(model_key),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            custom_prompt_function=custom_prompt_function,
            agent_index=agent_index,
        )
    raise ValueError(f"Unsupported policy_backend: {backend}")


def _reviewer_prompt(context: str, dataset_name: str) -> str:
    if dataset_name == "math500":
        instruction = (
            "Review the pending MATH-500 solution. Check the derivation and whether the "
            "boxed final answer solves the problem.\n"
            "First line: RIGHT or WRONG.\n"
            "Second line: one short reason about the boxed final answer."
        )
    elif dataset_name == "gpqa-diamond":
        instruction = (
            "Review the pending GPQA-Diamond solution. Check whether the chosen option letter "
            "is supported by the question and choices.\n"
            "First line: RIGHT or WRONG.\n"
            "Second line: one short reason about the chosen option letter."
        )
    else:
        instruction = (
            "Check whether the current pending answer is correct.\n"
            "First line: RIGHT or WRONG.\n"
            "Second line: one short reason."
        )
    return f"{instruction}\n\n{context}\n\nReview:\n"


def _proposer_prompt(context: str, dataset_name: str) -> str:
    if dataset_name == "math500":
        instruction = (
            "Solve the MATH-500 problem directly. Use concise chain-of-thought: write the "
            "key equations, simplify, and avoid filler.\n"
            "End with exactly one final line: Final answer: \\boxed{<answer>}."
        )
    elif dataset_name == "gpqa-diamond":
        instruction = (
            "Answer the GPQA-Diamond multiple-choice problem. Use concise chain-of-thought: "
            "eliminate impossible choices, then pick the best option.\n"
            "End with exactly one final line: Final answer: <A, B, C, or D>."
        )
    else:
        instruction = (
            "Solve the original problem directly. Use concise chain-of-thought.\n"
            "End with exactly one final line: Final answer: <answer>."
        )
    return f"{instruction}\n\n{context}\n\nSolution:\n"


def build_prompt(
    role: str,
    context: str,
    dataset_name: str = "gsm8k",
    custom_prompt_function: str | None = None,
    agent_index: int | None = None,
) -> str:
    dataset = normalize_dataset_name(dataset_name)
    if custom_prompt_function:
        prompt_builder = load_callable(custom_prompt_function)
        try:
            return str(
                prompt_builder(
                    role=role,
                    context=context,
                    dataset_name=dataset,
                    agent_index=agent_index,
                )
            )
        except TypeError:
            return str(prompt_builder(role=role, context=context, dataset_name=dataset))
    if role == "reviewer":
        return _reviewer_prompt(context, dataset)
    return _proposer_prompt(context, dataset)
