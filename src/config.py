"""Configuration loading for the standalone proposal-review experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationConfig:
    """Generation settings shared by proposer and reviewer policies."""

    max_new_tokens: int = 256
    temperature: float = 0.3
    num_candidates: int = 4
    num_greedy_candidates: int = 0


@dataclass
class EvaluationConfig:
    """Evaluation settings for trained controller policies."""

    controller_modes: list[str] = field(
        default_factory=lambda: ["stochastic", "greedy"]
    )


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    num_agents: int = 2
    train_rounds: int = 5
    eval_rounds: int = 5
    train_dataset: str = "gsm8k"
    test_dataset: str = "gsm8k"
    train_limit: int | None = None
    eval_limit: int | None = None
    agent_model: str = "qwen2.5-7b"
    policy_backend: str = "transformers"
    rl_algorithm: str = "gspo"
    custom_updater: str | None = None
    agent_rl_algorithms: dict[str, str] = field(default_factory=dict)
    agent_updaters: dict[str, str] = field(default_factory=dict)
    controller_update: str = "regret_matching"
    middle_value_mode: str = "pending_answer_signed"
    custom_value_function: str | None = None
    custom_reward_function: str | None = None
    custom_prompt_function: str | None = None
    custom_answer_extractor: str | None = None
    agent_schedule: str = "round_robin"
    seed: int = 42
    output_dir: str = "runs/default"
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text in {"null", "None", ""}:
        return None
    if text == "{}":
        return {}
    if text == "[]":
        return []
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text.strip("\"'")


def _minimal_yaml_load(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  - ") and current_map is not None and current_key:
            current_map.setdefault(current_key, []).append(_coerce_scalar(raw_line[4:]))
            continue
        if raw_line.startswith("  ") and current_map is not None:
            key, value = raw_line.strip().split(":", 1)
            value = value.strip()
            current_map[key] = [] if value == "" else _coerce_scalar(value)
            current_key = key
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            root[key] = {}
            current_map = root[key]
            current_key = None
        else:
            root[key] = _coerce_scalar(value)
            current_map = None
            current_key = None
    return root


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return _minimal_yaml_load(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(values: dict[str, Any]) -> ExperimentConfig:
    values = dict(values)
    generation_values = values.pop("generation", {}) or {}
    evaluation_values = values.pop("evaluation", {}) or {}
    allowed = set(ExperimentConfig.__dataclass_fields__)
    filtered = {key: value for key, value in values.items() if key in allowed}
    return ExperimentConfig(
        **filtered,
        generation=GenerationConfig(**generation_values),
        evaluation=EvaluationConfig(**evaluation_values),
    )


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    values: dict[str, Any] = {}
    if path is not None:
        values = _load_mapping(Path(path))
    if overrides:
        values = _merge_dict(values, dict(overrides))
    return _build_config(values)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "num_agents": config.num_agents,
        "train_rounds": config.train_rounds,
        "eval_rounds": config.eval_rounds,
        "train_dataset": config.train_dataset,
        "test_dataset": config.test_dataset,
        "train_limit": config.train_limit,
        "eval_limit": config.eval_limit,
        "agent_model": config.agent_model,
        "policy_backend": config.policy_backend,
        "rl_algorithm": config.rl_algorithm,
        "custom_updater": config.custom_updater,
        "agent_rl_algorithms": dict(config.agent_rl_algorithms),
        "agent_updaters": dict(config.agent_updaters),
        "controller_update": config.controller_update,
        "middle_value_mode": config.middle_value_mode,
        "custom_value_function": config.custom_value_function,
        "custom_reward_function": config.custom_reward_function,
        "custom_prompt_function": config.custom_prompt_function,
        "custom_answer_extractor": config.custom_answer_extractor,
        "agent_schedule": config.agent_schedule,
        "seed": config.seed,
        "output_dir": config.output_dir,
        "generation": vars(config.generation),
        "evaluation": vars(config.evaluation),
    }
