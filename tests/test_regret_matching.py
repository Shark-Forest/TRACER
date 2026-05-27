from src.cfr_core import RegretMatcher


def test_regret_matching_prefers_action_with_positive_regret():
    matcher = RegretMatcher(num_actions=2)

    matcher.update("state", action_values=[-1.0, 1.0], allowed_actions=[0, 1])
    strategy = matcher.strategy("state", allowed_actions=[0, 1])

    assert strategy[1] > strategy[0]


def test_average_strategy_accumulates_current_strategy():
    matcher = RegretMatcher(num_actions=2)

    matcher.update("state", action_values=[1.0, -1.0], allowed_actions=[0, 1])
    matcher.update("state", action_values=[1.0, -1.0], allowed_actions=[0, 1])

    average = matcher.average_strategy("state", allowed_actions=[0, 1])
    assert average[0] > average[1]
