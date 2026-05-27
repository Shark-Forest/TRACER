"""Training and evaluation loops for the standalone proposal-review repository."""

from __future__ import annotations

import json
from pathlib import Path
import random

from .config import ExperimentConfig, config_to_dict
from .data_loader import load_dataset
from .metrics import summarize_records
from .runtime import Runtime, run_dialogue


def _state_key_to_string(key: object) -> str:
    if isinstance(key, tuple):
        return " | ".join(str(part) for part in key)
    return str(key)


def _stringify_state_map(state_map: dict[object, object]) -> dict[str, object]:
    return {_state_key_to_string(key): value for key, value in state_map.items()}


def _run_split(
    runtime: Runtime,
    samples,
    *,
    num_rounds: int,
    update_controller: bool,
    use_average: bool,
    greedy: bool,
) -> dict[str, object]:
    records = [
        run_dialogue(
            runtime,
            sample,
            num_rounds,
            update_controller=update_controller,
            use_average=use_average,
            greedy=greedy,
        )
        for sample in samples
    ]
    return summarize_records(records)


def controller_snapshot(runtime: Runtime) -> list[dict[str, object]]:
    snapshot = []
    for index, controller in enumerate(runtime.controllers):
        snapshot.append(
            {
                "agent": index,
                "proposal_states": len(controller.proposal.regret_sum),
                "reviewer_states": len(controller.reviewer.regret_sum),
                "proposal_regret": _stringify_state_map(controller.proposal.regret_sum),
                "reviewer_regret": _stringify_state_map(controller.reviewer.regret_sum),
            }
        )
    return snapshot


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Train and evaluate the configured experiment."""

    random.seed(config.seed)
    runtime = Runtime.build(config)
    train_samples = load_dataset(config.train_dataset, "train", config.train_limit)
    eval_samples = load_dataset(config.test_dataset, "test", config.eval_limit)
    train_summary = _run_split(
        runtime,
        train_samples,
        num_rounds=config.train_rounds,
        update_controller=True,
        use_average=False,
        greedy=False,
    )
    eval_summary = _run_split(
        runtime,
        eval_samples,
        num_rounds=config.eval_rounds,
        update_controller=False,
        use_average=True,
        greedy=True,
    )
    result = {
        "config": config_to_dict(config),
        "train": {key: value for key, value in train_summary.items() if key != "records"},
        "eval": {key: value for key, value in eval_summary.items() if key != "records"},
        "controller_snapshot": controller_snapshot(runtime),
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "eval_samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in eval_summary["records"]),
        encoding="utf-8",
    )
    return result
