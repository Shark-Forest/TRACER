from src.controllers import ControllerBank
from src.values import pending_answer_skip_value


def test_pending_answer_skip_value_is_positive_for_correct_answer():
    assert pending_answer_skip_value(42, 42) == 1.0


def test_pending_answer_skip_value_is_negative_for_wrong_answer():
    assert pending_answer_skip_value(41, 42) == -1.0


def test_pending_answer_skip_value_is_zero_for_invalid_answer():
    assert pending_answer_skip_value(None, 42) == 0.0


def test_controller_bank_creates_one_agent_controller_per_agent():
    bank = ControllerBank(num_agents=3)

    assert len(bank.agent_controllers) == 3
    assert bank.for_agent(0) is not bank.for_agent(1)
    assert bank.for_agent(1) is not bank.for_agent(2)


def test_reviewer_controller_is_independent_from_proposal_controller():
    bank = ControllerBank(num_agents=1)
    controller = bank.for_agent(0)

    assert controller.proposal is not controller.reviewer
    assert controller.reviewer.num_actions == 2
