"""Runtime state and action execution for proposal-review dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import ExperimentConfig
from .controllers import (
    ControllerBank,
    PROPOSAL_REFRESH,
    REVIEWER_SPEAK,
)
from .customization import Customization
from .data_loader import Sample, extract_final_answer
from .policies import PolicyUpdater, TextPolicy, build_policy, build_updater
from .values import answers_match, pending_answer_skip_value, signed_answer_value


@dataclass
class DialogueState:
    """Mutable state for one sample's multi-round dialogue."""

    question: str
    ground_truth: object
    history: list[dict[str, object]] = field(default_factory=list)
    pending_text: str | None = None
    pending_answer: object | None = None
    pending_vote_score: int = 0
    final_answer: object | None = None

    def context(self) -> str:
        parts = [f"QUESTION:\n{self.question}"]
        if self.pending_text:
            parts.append(f"CURRENT PENDING SOLUTION:\n{self.pending_text}")
        if self.history:
            rendered = []
            for turn in self.history[-4:]:
                rendered.append(
                    f"[Round {turn['round']} | agent{turn['agent']} | {turn['kind']}]\n{turn['text']}"
                )
            parts.append("RECENT HISTORY:\n" + "\n\n".join(rendered))
        return "\n\n".join(parts)

    def append(self, round_index: int, agent_index: int, kind: str, text: str) -> None:
        self.history.append(
            {
                "round": round_index,
                "agent": agent_index,
                "kind": kind,
                "text": text,
            }
        )


@dataclass
class AgentBundle:
    """One agent bundle: proposer, reviewer, and its own controller."""

    proposer: TextPolicy
    reviewer: TextPolicy


@dataclass
class Runtime:
    """Full runtime object shared between training and evaluation."""

    config: ExperimentConfig
    agents: list[AgentBundle]
    controllers: ControllerBank
    updater: PolicyUpdater
    customization: Customization

    @classmethod
    def build(cls, config: ExperimentConfig) -> "Runtime":
        agents = []
        for _ in range(config.num_agents):
            agents.append(
                AgentBundle(
                    proposer=build_policy(
                        "proposer",
                        config.policy_backend,
                        config.agent_model,
                        config.generation.max_new_tokens,
                        config.generation.temperature,
                    ),
                    reviewer=build_policy(
                        "reviewer",
                        config.policy_backend,
                        config.agent_model,
                        config.generation.max_new_tokens,
                        config.generation.temperature,
                    ),
                )
            )
        return cls(
            config=config,
            agents=agents,
            controllers=ControllerBank(config.num_agents),
            updater=build_updater(config.rl_algorithm, config.custom_updater),
            customization=Customization(
                value_function=config.custom_value_function,
                reward_function=config.custom_reward_function,
            ),
        )

    def active_agent_index(self, round_index: int) -> int:
        if self.config.agent_schedule != "round_robin":
            raise ValueError(f"Unsupported agent_schedule: {self.config.agent_schedule}")
        return (round_index - 1) % len(self.agents)


def is_proposal_round(round_index: int) -> bool:
    return int(round_index) % 2 == 1


def controller_state(state: DialogueState, round_index: int, total_rounds: int) -> tuple[str, str, str]:
    if state.pending_answer is None:
        score_bucket = "no_pending"
    elif state.pending_vote_score > 0:
        score_bucket = f"vote_plus_{state.pending_vote_score}"
    elif state.pending_vote_score < 0:
        score_bucket = f"vote_minus_{abs(state.pending_vote_score)}"
    else:
        score_bucket = "vote_0"
    remaining_proposal_rounds = sum(
        1 for current in range(round_index, total_rounds + 1) if is_proposal_round(current)
    )
    time_bucket = "last" if remaining_proposal_rounds <= 1 else "not_last"
    return ("controller", score_bucket, time_bucket)


def parse_review_verdict(text: str) -> str | None:
    first_line = next((line.strip().lower() for line in text.splitlines() if line.strip()), "")
    if first_line.startswith("right") or first_line.startswith("correct"):
        return "right"
    if first_line.startswith("wrong") or first_line.startswith("incorrect"):
        return "wrong"
    return None


def default_reviewer_reward(review_text: str, pending_answer: object, ground_truth: object) -> float:
    verdict = parse_review_verdict(review_text)
    if verdict is None or pending_answer is None:
        return 0.0
    target = "right" if answers_match(pending_answer, ground_truth) else "wrong"
    return 1.0 if verdict == target else -1.0


def reviewer_reward(runtime: Runtime, review_text: str, state: DialogueState) -> float:
    default = default_reviewer_reward(review_text, state.pending_answer, state.ground_truth)
    return runtime.customization.reward(
        "reviewer",
        review_text,
        default=default,
        pending_answer=state.pending_answer,
        ground_truth=state.ground_truth,
        state=state,
    )


def proposer_reward(runtime: Runtime, proposal_text: str, state: DialogueState) -> float:
    answer = extract_final_answer(proposal_text)
    default = signed_answer_value(answer, state.ground_truth)
    return runtime.customization.reward(
        "proposer",
        proposal_text,
        default=default,
        parsed_answer=answer,
        ground_truth=state.ground_truth,
        state=state,
    )


def action_value(runtime: Runtime, action: str, default: float, state: DialogueState, **extra) -> float:
    return runtime.customization.value(
        action,
        default=default,
        pending_answer=state.pending_answer,
        ground_truth=state.ground_truth,
        pending_vote_score=state.pending_vote_score,
        state=state,
        **extra,
    )


def apply_review(state: DialogueState, review_text: str) -> None:
    verdict = parse_review_verdict(review_text)
    if verdict == "right":
        state.pending_vote_score += 1
    elif verdict == "wrong":
        state.pending_vote_score -= 1


def update_policy_from_candidates(
    runtime: Runtime,
    policy: TextPolicy,
    context: str,
    reward_fn: Callable[[str], float],
) -> None:
    candidates = policy.sample_candidates(context, runtime.config.generation.num_candidates)
    rewards = [float(reward_fn(candidate)) for candidate in candidates]
    runtime.updater.update(policy, context, candidates, rewards)


def execute_proposal_round(
    runtime: Runtime,
    state: DialogueState,
    round_index: int,
    total_rounds: int,
    *,
    update_controller: bool,
    use_average: bool,
    greedy: bool,
) -> dict[str, object]:
    agent_index = runtime.active_agent_index(round_index)
    agent = runtime.agents[agent_index]
    controller = runtime.controllers.for_agent(agent_index)
    c_state = controller_state(state, round_index, total_rounds)
    if state.pending_answer is None:
        action = PROPOSAL_REFRESH
    else:
        action = controller.proposal_action(c_state, use_average=use_average, greedy=greedy)

    old_pending = state.pending_answer
    context = state.context()
    if action == PROPOSAL_REFRESH:
        if update_controller:
            update_policy_from_candidates(
                runtime,
                agent.proposer,
                context,
                lambda text: proposer_reward(runtime, text, state),
            )
        text = agent.proposer.generate(context)
        answer = extract_final_answer(text)
        state.pending_text = text
        state.pending_answer = answer
        state.final_answer = answer
        state.pending_vote_score = 0
        state.append(round_index, agent_index, "proposal", text)
    else:
        state.final_answer = state.pending_answer

    keep_default = signed_answer_value(old_pending if old_pending is not None else state.pending_answer, state.ground_truth)
    refresh_default = signed_answer_value(state.pending_answer, state.ground_truth)
    keep_value = action_value(runtime, "proposal_keep", keep_default, state, previous_pending_answer=old_pending)
    refresh_value = action_value(runtime, "proposal_refresh", refresh_default, state, previous_pending_answer=old_pending)
    if update_controller and old_pending is not None:
        controller.update_proposal(c_state, keep_value=keep_value, refresh_value=refresh_value)

    return {
        "agent": agent_index,
        "stage": "proposal",
        "action": "refresh" if action == PROPOSAL_REFRESH else "keep",
        "final_answer": state.final_answer,
    }


def estimate_speak_value(runtime: Runtime, agent: AgentBundle, state: DialogueState) -> float:
    review_text = agent.reviewer.generate(state.context())
    return reviewer_reward(runtime, review_text, state)


def execute_review_round(
    runtime: Runtime,
    state: DialogueState,
    round_index: int,
    total_rounds: int,
    *,
    update_controller: bool,
    use_average: bool,
    greedy: bool,
) -> dict[str, object]:
    agent_index = runtime.active_agent_index(round_index)
    agent = runtime.agents[agent_index]
    controller = runtime.controllers.for_agent(agent_index)
    c_state = controller_state(state, round_index, total_rounds)

    if state.pending_answer is None:
        return {
            "agent": agent_index,
            "stage": "review",
            "action": "skip",
            "final_answer": state.final_answer,
        }

    action = controller.reviewer_action(c_state, use_average=use_average, greedy=greedy)
    if action == REVIEWER_SPEAK:
        context = state.context()
        if update_controller:
            update_policy_from_candidates(
                runtime,
                agent.reviewer,
                context,
                lambda text: reviewer_reward(runtime, text, state),
            )
        review_text = agent.reviewer.generate(context)
        speak_default = reviewer_reward(runtime, review_text, state)
        apply_review(state, review_text)
        state.append(round_index, agent_index, "review", review_text)
    else:
        speak_default = estimate_speak_value(runtime, agent, state) if update_controller else 0.0

    skip_default = pending_answer_skip_value(state.pending_answer, state.ground_truth)
    skip_value = action_value(runtime, "reviewer_skip", skip_default, state)
    speak_value = action_value(runtime, "reviewer_speak", speak_default, state)

    if update_controller:
        controller.reviewer.update(
            c_state,
            action_values=[skip_value, speak_value],
            allowed_actions=[0, 1],
        )
    state.final_answer = state.pending_answer
    return {
        "agent": agent_index,
        "stage": "review",
        "action": "speak" if action == REVIEWER_SPEAK else "skip",
        "skip_value": skip_value,
        "speak_value": speak_value,
        "final_answer": state.final_answer,
    }


def run_dialogue(
    runtime: Runtime,
    sample: Sample,
    num_rounds: int,
    *,
    update_controller: bool,
    use_average: bool = False,
    greedy: bool = False,
) -> dict[str, object]:
    state = DialogueState(question=sample.question, ground_truth=sample.answer)
    rounds = []
    for round_index in range(1, int(num_rounds) + 1):
        if is_proposal_round(round_index):
            record = execute_proposal_round(
                runtime,
                state,
                round_index,
                int(num_rounds),
                update_controller=update_controller,
                use_average=use_average,
                greedy=greedy,
            )
        else:
            record = execute_review_round(
                runtime,
                state,
                round_index,
                int(num_rounds),
                update_controller=update_controller,
                use_average=use_average,
                greedy=greedy,
            )
        rounds.append(record)
    return {
        "question": sample.question,
        "ground_truth": sample.answer,
        "final_answer": state.final_answer,
        "correct": answers_match(state.final_answer, sample.answer),
        "rounds": rounds,
    }
