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
reviewer should usually skip. The reviewer controller action space remains
binary: `skip` or `speak`.

## Reviewer Skip Value

The reviewer controller compares two values:

- `value(skip)`: value of preserving the current pending answer without another
  reviewer call.
- `value(speak)`: value of asking the reviewer to judge the pending answer.

The default skip value is:

```text
if pending answer is correct: value(skip) = +1
if pending answer is wrong:   value(skip) = -1
if pending answer is invalid: value(skip) = -1
```

This rule is implemented in `src/values.py` as
`pending_answer_skip_value`. An invalid pending answer means the proposer has
produced pending text, but the active dataset parser cannot extract a valid
answer. It is still a pending answer for reviewer-controller training, so
`reviewer_skip` receives `-1` rather than being treated as `no_pending`.

## Proposer Inner Rewards

When the proposal-stage controller chooses `refresh`, the proposer samples
candidate answers for the inner updater. Each candidate is parsed with the
active dataset's answer extractor, then scored with `signed_answer_value`:

```text
if parsed answer is correct: value(answer) = +1
if parsed answer is wrong:   value(answer) = -1
if answer is invalid:        value(answer) = -1
```

This default reward is passed to the configured updater, including the GSPO
hook. A `custom_reward_function` can override it.

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
    data_loader.py     # GSM8K/MATH500/GPQA/JSONL loading and answer extraction
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
| `train_dataset` | `gsm8k` | Training dataset name, HF/ModelScope spec, or JSONL path. |
| `test_dataset` | `gsm8k` | Test dataset name, HF/ModelScope spec, or JSONL path. |
| `agent_model` | `qwen2.5-7b` | Model key or Hugging Face model path. |
| `policy_backend` | `mock` | `mock` for smoke tests, `transformers` for real generation. |
| `rl_algorithm` | `gspo` | Inner-agent updater label. Built-ins: `gspo`, `none`; any custom label is allowed with `custom_updater`. |
| `custom_updater` | `null` | Import path for a custom inner policy updater. |
| `agent_rl_algorithms` | `{}` | Optional per-agent updater labels keyed by agent index. |
| `agent_updaters` | `{}` | Optional per-agent updater import paths keyed by agent index. |
| `controller_update` | `regret_matching` | Controller update rule. |
| `middle_value_mode` | `pending_answer_signed` | Built-in value rule for controller updates. |
| `custom_value_function` | `null` | Import path for custom middle-action values. |
| `custom_reward_function` | `null` | Import path for custom inner rewards. |
| `custom_prompt_function` | `null` | Import path for a custom proposer/reviewer prompt builder. |
| `custom_answer_extractor` | `null` | Import path for a custom generated-answer extractor. |

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

Built-in dataset names are `gsm8k`, `math500`, and `gpqa-diamond`. The loader
normalizes common aliases such as `MATH-500`, `gpqa_diamond`, and
`Idavidrein/gpqa:gpqa_diamond`.

Each dataset uses its own proposer/reviewer prompts and answer parser:

- `gsm8k`: concise chain-of-thought; final line `Final answer: <answer>`; numeric extraction.
- `math500`: concise chain-of-thought; final line `Final answer: \boxed{<answer>}`; boxed/LaTeX fraction extraction.
- `gpqa-diamond`: concise chain-of-thought; final line `Final answer: <A, B, C, or D>`; multiple-choice letter extraction.

Train and test datasets can be selected independently in one command:

```bash
python run_experiment.py \
  --train-dataset math500 \
  --test-dataset gpqa-diamond \
  --policy-backend transformers
```

If the `datasets` package or remote dataset access is unavailable, named
datasets fall back to tiny built-in sample sets so smoke tests still run.

External Hugging Face and ModelScope datasets can be selected directly:

```bash
python run_experiment.py \
  --train-dataset hf:org/custom-dataset::config-name \
  --test-dataset modelscope:owner/custom-dataset::subset-name
```

The `::config-name` or `::subset-name` suffix is optional. Without an explicit
prefix, strings shaped like `org/dataset` are treated as Hugging Face datasets.
Generic external rows are mapped with common fields: `question`, `problem`,
`prompt`, `input`, or `query` for the question, and `answer`, `ground_truth`,
`target`, `label`, or `output` for the gold answer.

You can also pass a JSONL file path as `train_dataset` or `test_dataset`. Each
line should contain `question` or `prompt`, plus `answer` or `ground_truth`.
Optionally include `dataset_name` to select the parser for that row:

```json
{"dataset_name": "math500", "question": "What is 40 + 2?", "answer": "42"}
```

## Custom Prompts and Answer Extraction

For datasets whose format does not match the built-in parsers, provide import
paths for custom hooks:

```python
def my_prompt(role: str, context: str, dataset_name: str, agent_index: int | None = None) -> str:
    return f"Agent {agent_index}: solve with my format.\n\n{context}\n"


def my_extract(text: str, dataset_name: str, agent_index: int | None = None):
    return text.split("FINAL=", 1)[1].strip() if "FINAL=" in text else None
```

Run with:

```bash
python run_experiment.py \
  --num-agents 4 \
  --train-dataset hf:org/custom-dataset \
  --test-dataset hf:org/custom-dataset \
  --custom-prompt-function my_package.hooks:my_prompt \
  --custom-answer-extractor my_package.hooks:my_extract
```

Both hooks receive `agent_index` and `dataset_name`, so one hook can route to
different prompt or extraction logic for different agents and datasets. Older
hooks that only accept `role/context/dataset_name` or `text/dataset_name` still
work.

## Custom Reinforcement-Learning Updaters

Agent policy learning is intentionally pluggable. A custom updater is any
object with this method:

```python
class MyUpdater:
    def update(self, policy, context: str, candidates: list[str], rewards: list[float]) -> None:
        ...
```

The default inner-agent updater is `gspo`. To use a different algorithm for
all agents, set any descriptive `rl_algorithm` label and provide
`custom_updater`:

```bash
python run_experiment.py \
  --rl-algorithm ppo \
  --custom-updater my_package.my_updater:PPOUpdater
```

For different inner algorithms per agent, use config-file maps keyed by agent
index:

```yaml
num_agents: 4
agent_rl_algorithms:
  "2": ppo
  "3": dpo
agent_updaters:
  "2": my_package.my_updater:PPOUpdater
  "3": my_package.my_updater:DPOUpdater
```

If an agent declares its own `agent_rl_algorithms` entry, that algorithm is used directly; it only uses an import path when `agent_updaters` provides one. Agents without an entry inherit the global `rl_algorithm` and `custom_updater`.

The trainer calls the selected updater after candidate generation and reward
calculation. This is where PPO, DPO-style updates, or a custom rule can be
implemented without changing the controller code.

## Custom Middle-Action Values

You can override controller action values with `custom_value_function`.

The function receives an action name, the default value, and contextual keyword
arguments including `agent_index` and `dataset_name`:

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
contextual keyword arguments including `agent_index` and `dataset_name`:

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
