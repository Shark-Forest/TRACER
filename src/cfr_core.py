"""Small regret-matching controller used by proposal and reviewer policies."""

from __future__ import annotations

import random
from typing import Hashable, Iterable


StateKey = Hashable | tuple[Hashable, ...]


class RegretMatcher:
    """State-conditioned regret matching with average-strategy tracking."""

    def __init__(self, num_actions: int, default_action: int = 0):
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.num_actions = int(num_actions)
        self.default_action = int(default_action)
        self.regret_sum: dict[tuple[Hashable, ...], list[float]] = {}
        self.strategy_sum: dict[tuple[Hashable, ...], list[float]] = {}
        self.iterations: dict[tuple[Hashable, ...], int] = {}

    def _state(self, state: StateKey | None) -> tuple[Hashable, ...]:
        if state is None:
            return ("__default__",)
        if isinstance(state, tuple):
            return state
        return (state,)

    def _ensure(self, state: StateKey | None) -> tuple[Hashable, ...]:
        key = self._state(state)
        if key not in self.regret_sum:
            self.regret_sum[key] = [0.0 for _ in range(self.num_actions)]
            self.strategy_sum[key] = [0.0 for _ in range(self.num_actions)]
            self.iterations[key] = 0
        return key

    def _allowed(self, allowed_actions: Iterable[int] | None) -> list[int]:
        if allowed_actions is None:
            return list(range(self.num_actions))
        allowed = sorted({int(action) for action in allowed_actions})
        return [action for action in allowed if 0 <= action < self.num_actions]

    def strategy(
        self,
        state: StateKey | None = None,
        allowed_actions: Iterable[int] | None = None,
    ) -> list[float]:
        """Return the current regret-matching strategy for a state."""

        key = self._ensure(state)
        allowed = self._allowed(allowed_actions)
        strategy = [0.0 for _ in range(self.num_actions)]
        positive = [max(self.regret_sum[key][action], 0.0) for action in allowed]
        total = sum(positive)
        if total > 0:
            for action, value in zip(allowed, positive):
                strategy[action] = value / total
            return strategy
        if allowed:
            probability = 1.0 / len(allowed)
            for action in allowed:
                strategy[action] = probability
            return strategy
        strategy[self.default_action] = 1.0
        return strategy

    def average_strategy(
        self,
        state: StateKey | None = None,
        allowed_actions: Iterable[int] | None = None,
    ) -> list[float]:
        """Return the accumulated average strategy for evaluation."""

        key = self._ensure(state)
        allowed = self._allowed(allowed_actions)
        total = sum(self.strategy_sum[key][action] for action in allowed)
        if total <= 0:
            return self.strategy(state, allowed)
        strategy = [0.0 for _ in range(self.num_actions)]
        for action in allowed:
            strategy[action] = self.strategy_sum[key][action] / total
        return strategy

    def choose(
        self,
        state: StateKey | None = None,
        allowed_actions: Iterable[int] | None = None,
        *,
        use_average: bool = False,
        greedy: bool = False,
    ) -> int:
        """Choose an action from the current or average strategy."""

        strategy = (
            self.average_strategy(state, allowed_actions)
            if use_average else self.strategy(state, allowed_actions)
        )
        allowed = self._allowed(allowed_actions)
        if greedy:
            return max(allowed, key=lambda action: strategy[action])
        threshold = random.random()
        cumulative = 0.0
        for action in allowed:
            cumulative += strategy[action]
            if threshold <= cumulative:
                return action
        return allowed[-1] if allowed else self.default_action

    def update(
        self,
        state: StateKey | None,
        action_values: Iterable[float],
        allowed_actions: Iterable[int] | None = None,
        strategy_sum_override: Iterable[float] | None = None,
    ) -> list[float]:
        """Apply one expected-baseline regret update."""

        key = self._ensure(state)
        allowed = self._allowed(allowed_actions)
        values = [float(value) for value in action_values]
        if len(values) != self.num_actions:
            raise ValueError("action_values length must match num_actions")
        current = self.strategy(state, allowed)
        baseline = sum(current[action] * values[action] for action in allowed)
        regrets = [0.0 for _ in range(self.num_actions)]
        for action in allowed:
            regrets[action] = values[action] - baseline
            self.regret_sum[key][action] += regrets[action]
        if strategy_sum_override is None:
            strategy_to_add = current
        else:
            strategy_to_add = [float(value) for value in strategy_sum_override]
            if len(strategy_to_add) != self.num_actions:
                raise ValueError("strategy_sum_override length must match num_actions")
        for action in allowed:
            self.strategy_sum[key][action] += max(strategy_to_add[action], 0.0)
        self.iterations[key] += 1
        return regrets
