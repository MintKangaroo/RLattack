"""Optional CPU smoke training, skipped unless the training extras are installed."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlattack.env import AttackPathEnv
from rlattack.generator import generate_scenario
from rlattack.policies import load_policy
from rlattack.training import (
    DQNTrainingConfig,
    PPOTrainingConfig,
    train_dqn,
    train_ppo,
    training_dependencies_available,
)

pytestmark = pytest.mark.skipif(
    not training_dependencies_available(),
    reason="requires the optional '.[training]' dependencies",
)


def env_factory() -> AttackPathEnv:
    return AttackPathEnv(generate_scenario("small", "easy", 0), step_budget=16)


@pytest.mark.training
def test_dqn_smoke_training_writes_a_loadable_checkpoint(tmp_path: Path) -> None:
    train_dqn(
        env_factory,
        DQNTrainingConfig(
            total_timesteps=64,
            checkpoint_frequency=64,
            evaluation_frequency=64,
            output_dir=tmp_path / "dqn",
            tensorboard_log=tmp_path / "runs",
        ),
    )
    agent = load_policy(tmp_path / "dqn" / "final", "dqn")
    env = env_factory()
    observation, info = env.reset(seed=0)

    assert 0 <= int(agent.predict(observation, info)) < len(info["action_mask"])


@pytest.mark.training
def test_ppo_smoke_training_writes_a_loadable_checkpoint(tmp_path: Path) -> None:
    train_ppo(
        env_factory,
        PPOTrainingConfig(
            total_timesteps=64,
            rollout_steps=32,
            batch_size=16,
            checkpoint_frequency=64,
            evaluation_frequency=64,
            output_dir=tmp_path / "ppo",
            tensorboard_log=tmp_path / "runs",
        ),
    )

    assert (tmp_path / "ppo" / "final.zip").exists()
