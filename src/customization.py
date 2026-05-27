"""Customization hooks for middle-action values and inner rewards."""

from __future__ import annotations

import importlib
from typing import Callable


ValueFunction = Callable[..., float]
RewardFunction = Callable[..., float]


def load_callable(import_path: str | None):
    """Load `module.submodule:object` and return the referenced object."""

    if not import_path:
        return None
    module_name, object_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, object_name)
    return target() if isinstance(target, type) else target


class Customization:
    """Optional hooks used by the trainer.

    `value_function` customizes controller action values. It receives the
    action name, a default value, and any contextual keyword arguments.

    `reward_function` customizes inner policy rewards. It receives the policy
    role, generated text, and contextual keyword arguments.
    """

    def __init__(
        self,
        value_function: ValueFunction | str | None = None,
        reward_function: RewardFunction | str | None = None,
    ):
        self.value_function = (
            load_callable(value_function)
            if isinstance(value_function, str)
            else value_function
        )
        self.reward_function = (
            load_callable(reward_function)
            if isinstance(reward_function, str)
            else reward_function
        )

    def value(self, action: str, default: float, **context) -> float:
        if self.value_function is None:
            return float(default)
        return float(self.value_function(action=action, default=float(default), **context))

    def reward(self, role: str, text: str, default: float = 0.0, **context) -> float:
        if self.reward_function is None:
            return float(default)
        return float(self.reward_function(role=role, text=text, default=float(default), **context))
