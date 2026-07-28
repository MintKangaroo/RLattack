from pathlib import Path

import pytest

from rlattack.training import DQNTrainingConfig, training_dependencies_available


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
