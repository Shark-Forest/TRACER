"""Per-agent proposal and reviewer controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cfr_core import RegretMatcher, StateKey
from .values import pending_answer_skip_value


PROPOSAL_KEEP = 0
PROPOSAL_REFRESH = 1
REVIEWER_SKIP = 0
REVIEWER_SPEAK = 1


@dataclass
class AgentController:
    """Controller set owned by one agent bundle."""

    proposal: RegretMatcher
    reviewer: RegretMatcher

    @classmethod
    def create(cls) -> "AgentController":
        return cls(
            proposal=RegretMatcher(num_actions=2, default_action=PROPOSAL_REFRESH),
            reviewer=RegretMatcher(num_actions=2, default_action=REVIEWER_SPEAK),
        )

    def proposal_action(
        self,
        state: StateKey,
        *,
        use_average: bool = False,
        greedy: bool = False,
    ) -> int:
        return self.proposal.choose(
            state,
            [PROPOSAL_KEEP, PROPOSAL_REFRESH],
            use_average=use_average,
            greedy=greedy,
        )

    def reviewer_action(
        self,
        state: StateKey,
        *,
        use_average: bool = False,
        greedy: bool = False,
    ) -> int:
        return self.reviewer.choose(
            state,
            [REVIEWER_SKIP, REVIEWER_SPEAK],
            use_average=use_average,
            greedy=greedy,
        )

    def update_reviewer(
        self,
        state: StateKey,
        *,
        pending_answer: object,
        ground_truth: object,
        speak_value: float,
        selected_action: int,
    ) -> list[float]:
        """Update reviewer-stage regrets for skip versus speak."""

        action_values = [
            pending_answer_skip_value(pending_answer, ground_truth),
            float(speak_value),
        ]
        return self.reviewer.update(
            state,
            action_values=action_values,
            allowed_actions=[REVIEWER_SKIP, REVIEWER_SPEAK],
        )

    def update_proposal(
        self,
        state: StateKey,
        *,
        keep_value: float,
        refresh_value: float,
    ) -> list[float]:
        return self.proposal.update(
            state,
            action_values=[float(keep_value), float(refresh_value)],
            allowed_actions=[PROPOSAL_KEEP, PROPOSAL_REFRESH],
        )


class ControllerBank:
    """Container that creates exactly one controller set per agent bundle."""

    def __init__(self, num_agents: int):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.agent_controllers = [AgentController.create() for _ in range(num_agents)]

    def __len__(self) -> int:
        return len(self.agent_controllers)

    def for_agent(self, agent_index: int) -> AgentController:
        return self.agent_controllers[int(agent_index)]

    def __iter__(self) -> Iterable[AgentController]:
        return iter(self.agent_controllers)
