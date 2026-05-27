from pathlib import Path

from src.config import ExperimentConfig, load_config


def test_default_config_matches_public_defaults():
    config = ExperimentConfig()

    assert config.num_agents == 2
    assert config.train_rounds == 5
    assert config.eval_rounds == 5
    assert config.agent_model == "qwen2.5-7b"
    assert config.rl_algorithm == "gspo"
    assert config.controller_update == "regret_matching"
    assert config.train_dataset == "gsm8k"
    assert config.test_dataset == "gsm8k"


def test_yaml_config_and_overrides_are_merged(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "num_agents: 4\n"
        "train_rounds: 7\n"
        "agent_model: tiny-test-model\n",
        encoding="utf-8",
    )

    config = load_config(path, overrides={"num_agents": 3})

    assert config.num_agents == 3
    assert config.train_rounds == 7
    assert config.agent_model == "tiny-test-model"
