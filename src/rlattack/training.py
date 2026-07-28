"""Optional Stable-Baselines3 training pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class DQNTrainingConfig:
    """Reproducible settings for one DQN experiment."""

    total_timesteps: int = 10_000
    seed: int = 0
    checkpoint_frequency: int = 1_000
    evaluation_frequency: int = 1_000
    output_dir: Path = Path("artifacts/dqn")
    tensorboard_log: Path = Path("runs/dqn")

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.checkpoint_frequency < 1:
            raise ValueError("checkpoint_frequency must be positive")
        if self.evaluation_frequency < 1:
            raise ValueError("evaluation_frequency must be positive")


def training_dependencies_available() -> bool:
    """Return whether Stable-Baselines3 and PyTorch are installed."""

    return find_spec("stable_baselines3") is not None and find_spec("torch") is not None


def train_dqn(  # pragma: no cover - exercised by the optional long-running training job
    env_factory: Callable[[], gym.Env[Any, Any]],
    config: DQNTrainingConfig | None = None,
) -> Any:
    """Train and checkpoint a DQN using vectorized simulator environments.

    This function is intentionally not invoked by CI. It creates only local simulator
    environments supplied by ``env_factory`` and writes local model artifacts.
    """

    selected = config or DQNTrainingConfig()
    try:
        from stable_baselines3 import DQN  # type: ignore
        from stable_baselines3.common.callbacks import (  # type: ignore
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "DQN training requires the optional '.[training]' dependencies"
        ) from error

    selected.output_dir.mkdir(parents=True, exist_ok=True)
    selected.tensorboard_log.mkdir(parents=True, exist_ok=True)
    train_env = DummyVecEnv([env_factory])
    eval_env = DummyVecEnv([env_factory])
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=selected.checkpoint_frequency,
                save_path=str(selected.output_dir),
                name_prefix="rlattack-dqn",
            ),
            EvalCallback(
                eval_env,
                eval_freq=selected.evaluation_frequency,
                best_model_save_path=str(selected.output_dir / "best"),
                log_path=str(selected.output_dir / "evaluation"),
                deterministic=True,
            ),
        ]
    )
    model = DQN(
        "MultiInputPolicy",
        train_env,
        seed=selected.seed,
        tensorboard_log=str(selected.tensorboard_log),
        verbose=1,
    )
    model.learn(
        total_timesteps=selected.total_timesteps,
        callback=callbacks,
        tb_log_name="rlattack-dqn",
    )
    model.save(str(selected.output_dir / "final"))
    train_env.close()
    eval_env.close()
    return model
