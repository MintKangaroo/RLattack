from pathlib import Path

import pytest

from rlattack.training import (
    DQNTrainingConfig,
    PPOTrainingConfig,
    training_dependencies_available,
)


def test_dqn_config_is_reproducible_and_validates_budgets() -> None:
    config = DQNTrainingConfig(total_timesteps=10, seed=7, output_dir=Path("artifacts/test"))

    assert config.total_timesteps == 10
    assert config.seed == 7
    with pytest.raises(ValueError, match="total_timesteps"):
        DQNTrainingConfig(total_timesteps=0)
    with pytest.raises(ValueError, match="checkpoint_frequency"):
        DQNTrainingConfig(checkpoint_frequency=0)
    with pytest.raises(ValueError, match="evaluation_frequency"):
        DQNTrainingConfig(evaluation_frequency=0)


def test_training_dependency_probe_returns_boolean() -> None:
    assert isinstance(training_dependencies_available(), bool)


def test_ppo_config_shares_benchmark_contract() -> None:
    config = PPOTrainingConfig(total_timesteps=20, rollout_steps=8, batch_size=4)

    assert config.rollout_steps == 8
    assert config.batch_size == 4
    with pytest.raises(ValueError, match="rollout_steps"):
        PPOTrainingConfig(rollout_steps=0)
    with pytest.raises(ValueError, match="batch_size"):
        PPOTrainingConfig(batch_size=0)


def test_ppo_learning_knobs_are_validated() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        PPOTrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="entropy_coefficient"):
        PPOTrainingConfig(entropy_coefficient=-0.1)
