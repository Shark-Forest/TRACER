from src.customization import Customization


def custom_value(action, default, **kwargs):
    if action == "reviewer_skip":
        return 7.0
    return default


def custom_reward(role, text, **kwargs):
    if role == "proposer":
        return 0.25
    if role == "reviewer":
        return -0.5
    return 0.0


def test_custom_value_function_can_override_middle_action_value():
    customization = Customization(value_function=custom_value)

    assert customization.value("reviewer_skip", default=-1.0) == 7.0
    assert customization.value("proposal_keep", default=1.0) == 1.0


def test_custom_reward_function_can_override_inner_rewards():
    customization = Customization(reward_function=custom_reward)

    assert customization.reward("proposer", "Final answer: 42") == 0.25
    assert customization.reward("reviewer", "WRONG") == -0.5
