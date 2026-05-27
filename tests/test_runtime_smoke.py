from src.config import ExperimentConfig
from src.trainer import run_experiment


def test_mock_runtime_runs_with_two_agent_specific_controllers(tmp_path):
    config = ExperimentConfig(
        output_dir=str(tmp_path / "run"),
        train_limit=2,
        eval_limit=2,
        train_rounds=3,
        eval_rounds=3,
        policy_backend="mock",
        num_agents=2,
    )

    result = run_experiment(config)

    assert result["config"]["num_agents"] == 2
    assert len(result["controller_snapshot"]) == 2
    assert result["train"]["num_samples"] == 2
    assert result["eval"]["num_samples"] == 2
    assert "accuracy" in result["eval"]
