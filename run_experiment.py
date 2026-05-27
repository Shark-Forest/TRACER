"""Command-line entry point for the cleaned proposal-review experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_config
from src.trainer import run_experiment


def _non_null_overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "num_agents": args.num_agents,
        "train_rounds": args.train_rounds,
        "eval_rounds": args.eval_rounds,
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "agent_model": args.agent_model,
        "policy_backend": args.policy_backend,
        "rl_algorithm": args.rl_algorithm,
        "custom_updater": args.custom_updater,
        "controller_update": args.controller_update,
        "middle_value_mode": args.middle_value_mode,
        "custom_value_function": args.custom_value_function,
        "custom_reward_function": args.custom_reward_function,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone proposal-review experiment."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--train-rounds", type=int)
    parser.add_argument("--eval-rounds", type=int)
    parser.add_argument("--train-dataset")
    parser.add_argument("--test-dataset")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--agent-model")
    parser.add_argument("--policy-backend", choices=["mock", "transformers"])
    parser.add_argument("--rl-algorithm", choices=["gspo", "custom", "none", "noop"])
    parser.add_argument("--custom-updater")
    parser.add_argument("--custom-value-function")
    parser.add_argument("--custom-reward-function")
    parser.add_argument("--controller-update", choices=["regret_matching"])
    parser.add_argument("--middle-value-mode", choices=["pending_answer_signed"])
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=_non_null_overrides(args))
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
