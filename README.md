# Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning

Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative
Multi-LLM Reasoning is a cleaned, standalone implementation of a
proposal-review multi-agent reinforcement-learning dialogue system.

The repository is designed to be readable first. It keeps the core idea from
the original experiment code but removes paper-specific clutter, historical
ablation scripts, and mixed-language comments.

## Method Overview

Each task is solved through a multi-round dialogue. Odd rounds are proposal
rounds and even rounds are review rounds.

In a proposal round, the active agent's proposer either creates a pending
answer or refreshes an existing pending answer.

In a review round, the active agent's reviewer may either skip or speak. The
reviewer only generates a judgment when its controller chooses `speak`.

## Per-Agent Controllers

If the run uses `N` agent bundles, the runtime creates `N` agent-level
controllers. There is no single shared global reviewer controller.

Each agent-level controller owns two regret-matching policies:

- A proposal-stage policy for `keep` versus `refresh`.
- A reviewer-stage policy for `skip` versus `speak`.

This means every agent can learn its own control behavior. For example, agent 0
can learn that its reviewer should usually speak in a state where agent 1's
reviewer should usually skip.

## Reviewer Skip Value

The reviewer controller compares two values:

- `value(skip)`: value of preserving the current pending answer without another
  reviewer call.
- `value(speak)`: value of asking the reviewer to judge the pending answer.

The default skip value is:

```text
if pending answer is correct: value(skip) = +1
if pending answer is wrong:   value(skip) = -1
if pending answer is invalid: value(skip) = 0
```

This rule is implemented in `src/values.py` as
`pending_answer_skip_value`.

## Repository Layout

```text
Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning/
  README.md
  requirements.txt
  run_experiment.py
  configs/default.yaml
  src/
    config.py          # YAML and CLI configuration
    cfr_core.py        # Simple regret matching
    controllers.py     # Per-agent controller sets
    customization.py   # Custom value and reward hooks
    data_loader.py     # GSM8K/JSONL loading and answer extraction
    policies.py        # Mock, Transformers, GSPO/custom updater interfaces
    runtime.py         # Proposal-review state machine
    trainer.py         # Train/evaluate loops and output writing
    metrics.py         # Accuracy summaries
  tests/
    test_config.py
    test_controller_values.py
    test_customization.py
    test_regret_matching.py
    test_runtime_smoke.py
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model-free smoke experiment:

```bash
python run_experiment.py --policy-backend mock --train-limit 2 --eval-limit 2
```

Run tests:

```bash
pytest tests -q
```

## Configuration

Defaults live in `configs/default.yaml`.

Important options:

| Option | Default | Meaning |
| --- | --- | --- |
| `num_agents` | `2` | Number of agent bundles and controller sets. |
| `train_rounds` | `5` | Number of training dialogue rounds. |
| `eval_rounds` | `5` | Number of evaluation dialogue rounds. |
| `train_dataset` | `gsm8k` | Training dataset name or JSONL path. |
| `test_dataset` | `gsm8k` | Test dataset name or JSONL path. |
| `agent_model` | `qwen2.5-7b` | Model key or Hugging Face model path. |
| `policy_backend` | `mock` | `mock` for smoke tests, `transformers` for real generation. |
| `rl_algorithm` | `gspo` | Policy updater key: `gspo`, `custom`, or `none`. |
| `custom_updater` | `null` | Import path for a custom inner policy updater. |
| `controller_update` | `regret_matching` | Controller update rule. |
| `middle_value_mode` | `pending_answer_signed` | Built-in value rule for controller updates. |
| `custom_value_function` | `null` | Import path for custom middle-action values. |
| `custom_reward_function` | `null` | Import path for custom inner rewards. |

Any top-level option can be overridden from the CLI:

```bash
python run_experiment.py \
  --num-agents 4 \
  --train-rounds 7 \
  --eval-rounds 7 \
  --agent-model Qwen/Qwen2.5-7B-Instruct \
  --policy-backend transformers
```

## Dataset Format

`gsm8k` is supported by default. If the `datasets` package is available, the
loader tries Hugging Face GSM8K. If not, it falls back to a tiny built-in sample
set so the repository can still run tests without downloads.

You can also pass a JSONL file path as `train_dataset` or `test_dataset`. Each
line should contain:

```json
{"question": "What is 40 + 2?", "answer": 42}
```

## Custom Reinforcement-Learning Updaters

Agent policy learning is intentionally pluggable. A custom updater is any
object with this method:

```python
class MyUpdater:
    def update(self, policy, context: str, candidates: list[str], rewards: list[float]) -> None:
        ...
```

Save it in an importable module and run:

```bash
python run_experiment.py \
  --rl-algorithm custom \
  --custom-updater my_package.my_updater:MyUpdater
```

The trainer calls the updater after candidate generation and reward
calculation. This is where GSPO, PPO, DPO-style updates, or a custom rule can be
implemented without changing the controller code.

## Custom Middle-Action Values

You can override controller action values with `custom_value_function`.

The function receives an action name, the default value, and contextual keyword
arguments:

```python
def my_value(action: str, default: float, **context) -> float:
    if action == "reviewer_skip":
        pending_answer = context["pending_answer"]
        ground_truth = context["ground_truth"]
        ...
    return default
```

Supported action names are:

- `proposal_keep`
- `proposal_refresh`
- `reviewer_skip`
- `reviewer_speak`

Run with:

```bash
python run_experiment.py \
  --custom-value-function my_package.values:my_value
```

## Custom Inner Rewards

You can override proposer/reviewer candidate rewards with
`custom_reward_function`.

The function receives the role, generated text, the default reward, and
contextual keyword arguments:

```python
def my_reward(role: str, text: str, default: float, **context) -> float:
    if role == "proposer":
        parsed_answer = context["parsed_answer"]
        ...
    if role == "reviewer":
        pending_answer = context["pending_answer"]
        ...
    return default
```

Run with:

```bash
python run_experiment.py \
  --custom-reward-function my_package.rewards:my_reward
```

## Output Files

Each run writes to `output_dir`:

- `summary.json`: configuration, train/eval metrics, and controller snapshots.
- `eval_samples.jsonl`: per-sample evaluation traces.

## Notes on Real Model Runs

The default public configuration uses `policy_backend: mock` so the repository
is easy to test. To run Qwen2.5-7B, set:

```bash
python run_experiment.py --policy-backend transformers --agent-model Qwen/Qwen2.5-7B-Instruct
```

Real model-backed training needs sufficient GPU memory and the optional
Transformers/PyTorch dependencies listed in `requirements.txt`.
