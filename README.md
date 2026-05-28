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
| `agent_model` | `qwen2.5-7b` | Model key, ModelScope/Hugging Face model id, or local model path. |
| `policy_backend` | `transformers` | `transformers` for real generation, `mock` for smoke tests. |
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

## Model Loading

For `policy_backend: transformers`, TRACER loads models in this order:

1. Existing local path, when `agent_model` points to a directory on disk.
2. ModelScope snapshot via `modelscope.snapshot_download(...)`.
3. Hugging Face `from_pretrained(...)` fallback with the original model id.

The built-in alias `qwen2.5-7b` resolves to `Qwen/Qwen2.5-7B-Instruct`, then
follows the same ModelScope-first path. Install `modelscope` for live ModelScope
downloads:

```bash
pip install modelscope
```

Model cache selection uses the first configured value from this list:

```bash
export TRACER_MODELSCOPE_MODEL_CACHE_DIR=/path/to/model-cache
export TRACER_MODELSCOPE_CACHE_DIR=/path/to/shared-modelscope-cache
export MODELSCOPE_CACHE=/path/to/modelscope-cache
```

If ModelScope is not installed, the download fails, or the downloaded snapshot
cannot be loaded by Transformers, TRACER falls back to Hugging Face with the
original `agent_model` id.

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

Built-in datasets are ModelScope-first. The loading order is:

1. `MsDataset.load(...)` from ModelScope.
2. Local ModelScope arrow cache, if ModelScope is unavailable.
3. Hugging Face fallback for compatibility.
4. Tiny built-in samples for smoke tests only.

Built-in ModelScope defaults:

- `gsm8k`: `modelscope/gsm8k`, subset `main`, train/test splits.
- `math500`: `AI-ModelScope/MATH-500`, test split; fallback `modelscope/R1-Distill-Math-Test`.
- `gpqa-diamond`: `AI-ModelScope/GPQA`, subset `gpqa_diamond`; fallback `modelscope/R1-Distill-Math-Test`.

You can point TRACER at explicit local arrow caches:

```bash
export TRACER_GSM8K_TRAIN_ARROW=/path/to/gsm8k-train.arrow
export TRACER_GSM8K_TEST_ARROW=/path/to/gsm8k-test.arrow
export TRACER_GSM8K_CACHE_DIR=/path/to/modelscope___gsm8k

export TRACER_MATH500_TEST_ARROW=/path/to/math-500-test.arrow
export TRACER_MATH500_ARROW=/path/to/math-500-test.arrow
export TRACER_MATH500_CACHE_DIR=/path/to/AI-ModelScope___math-500

export TRACER_GPQA_DIAMOND_TRAIN_ARROW=/path/to/gpqa-train.arrow
export TRACER_GPQA_DIAMOND_ARROW=/path/to/gpqa-train.arrow
export TRACER_GPQA_DIAMOND_CACHE_DIR=/path/to/AI-ModelScope___gpqa
```

For live ModelScope downloads, `TRACER_MODELSCOPE_CACHE_DIR` controls the cache
root. Dataset-specific overrides are also supported:

```bash
export TRACER_MATH500_DATASET=AI-ModelScope/MATH-500
export TRACER_MATH500_SPLIT=test
export TRACER_GPQA_DATASET=AI-ModelScope/GPQA
export TRACER_GPQA_SUBSET=gpqa_diamond
export TRACER_GPQA_SPLIT=train
```

If remote dataset packages or network access are unavailable, named datasets
fall back to tiny built-in sample sets so smoke tests still run.

### New ModelScope Datasets

New external datasets default to ModelScope. Use the ModelScope dataset id,
with or without an owner namespace, and append `::subset-name` only when the
dataset has a subset/config:

```bash
python run_experiment.py \
  --train-dataset owner/custom-dataset::subset-name \
  --test-dataset owner/custom-eval-dataset::subset-name \
  --policy-backend transformers
```

The explicit `modelscope:` or `ms:` prefix is optional for ModelScope datasets:

```bash
python run_experiment.py \
  --train-dataset modelscope:owner/custom-dataset::subset-name \
  --test-dataset ms:owner/custom-eval-dataset::subset-name
```

Use Hugging Face only when explicitly requested with `hf:` or `huggingface:`:

```bash
python run_experiment.py \
  --train-dataset hf:org/custom-dataset::config-name \
  --test-dataset huggingface:org/custom-eval-dataset::config-name
```

The `::config-name` or `::subset-name` suffix is optional. Generic external
rows are mapped with common fields: `question`, `problem`, `prompt`, `input`,
or `query` for the question, and `answer`, `ground_truth`, `target`, `label`,
or `output` for the gold answer. Rows that wrap the original item under
`prompt.raw_input` are also supported.

For a new dataset whose answer format differs from GSM8K/MATH-500/GPQA-Diamond,
provide both hooks so prompting and scoring agree:

```bash
python run_experiment.py \
  --train-dataset owner/new-dataset::subset-name \
  --test-dataset owner/new-eval::subset-name \
  --custom-prompt-function my_hooks:prompt \
  --custom-answer-extractor my_hooks:extract
```

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
  --train-dataset owner/custom-dataset \
  --test-dataset owner/custom-dataset \
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

The default inner-agent updater is `gspo`. For `transformers` policies this is
a real GSPO update aligned with `MAS/final/src/gspo_verl.py`: group-relative
advantages, sequence-level importance ratios, clipped surrogate loss, gradient
clipping, and an optimizer step. LoRA is the default finetuning mode; configure
it with `TRACER_GSPO_*` environment variables, or the matching `MAS_GSPO_*`
variables used by `MAS/final`. The `mock` backend records an explicit skipped
update and is only for smoke tests.

To use a different algorithm for all agents, set any descriptive
`rl_algorithm` label and provide `custom_updater`:

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

If an agent declares its own `agent_rl_algorithms` entry, that algorithm is used
directly; it only uses an import path when `agent_updaters` provides one.
Agents without an entry inherit the global `rl_algorithm` and `custom_updater`.

The trainer calls the selected updater after candidate generation and reward
calculation. The built-in regret matcher for the middle controller is already
implemented in `src/cfr_core.py`; custom middle values still come from
`custom_value_function`.

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

The default configuration uses `policy_backend: transformers` and the default
`gspo` updater. For a model-free smoke run, pass `--policy-backend mock`.

```bash
python run_experiment.py --policy-backend mock --train-limit 2 --eval-limit 2
```

Real model-backed training needs sufficient GPU memory and the optional
Transformers/PyTorch dependencies listed in `requirements.txt`.
