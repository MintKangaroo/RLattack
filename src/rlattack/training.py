"""Optional Stable-Baselines3 training pipelines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


@dataclass(frozen=True)
class PPOTrainingConfig(DQNTrainingConfig):
    """Reproducible settings for one PPO experiment."""

    rollout_steps: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must not be negative")


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
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise RuntimeError(
            "DQN training requires the optional '.[training]' dependencies"
        ) from error

    selected.output_dir.mkdir(parents=True, exist_ok=True)
    selected.tensorboard_log.mkdir(parents=True, exist_ok=True)
    train_env = DummyVecEnv([lambda: Monitor(env_factory())])
    eval_env = DummyVecEnv([lambda: Monitor(env_factory())])
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


def train_ppo(  # pragma: no cover - exercised by the optional long-running training job
    env_factory: Callable[[], gym.Env[Any, Any]],
    config: PPOTrainingConfig | None = None,
) -> Any:
    """Train PPO with the same vectorized environment and evaluation contract as DQN."""

    selected = config or PPOTrainingConfig()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise RuntimeError(
            "PPO training requires the optional '.[training]' dependencies"
        ) from error

    selected.output_dir.mkdir(parents=True, exist_ok=True)
    selected.tensorboard_log.mkdir(parents=True, exist_ok=True)
    train_env = DummyVecEnv([lambda: Monitor(env_factory())])
    eval_env = DummyVecEnv([lambda: Monitor(env_factory())])
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=selected.checkpoint_frequency,
                save_path=str(selected.output_dir),
                name_prefix="rlattack-ppo",
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
    model = PPO(
        "MultiInputPolicy",
        train_env,
        seed=selected.seed,
        n_steps=selected.rollout_steps,
        batch_size=selected.batch_size,
        tensorboard_log=str(selected.tensorboard_log),
        verbose=1,
    )
    model.learn(
        total_timesteps=selected.total_timesteps,
        callback=callbacks,
        tb_log_name="rlattack-ppo",
    )
    model.save(str(selected.output_dir / "final"))
    train_env.close()
    eval_env.close()
    return model


def train_curriculum(  # pragma: no cover - exercised by the optional training job
    env_factories: Sequence[Callable[[], gym.Env[Any, Any]]],
    stage_timesteps: Sequence[int],
    config: PPOTrainingConfig | None = None,
    algorithm: str = "ppo",
) -> Any:
    """Train one policy across a sequence of scenario stages.

    The model is carried between stages with ``set_env`` and
    ``reset_num_timesteps=False``, so the run is a curriculum rather than a set of
    independent trainings. Every stage must expose the same observation and action
    space; use ``ObservationConfig.for_curriculum()`` to guarantee that.

    ``maskable-ppo`` is usually the right choice here. Only a small fraction of the
    targeted action space is valid in any state, so an unmasked learner spends its
    exploration budget on invalid actions and converges on stopping immediately.
    """

    if len(env_factories) != len(stage_timesteps):
        raise ValueError("each stage needs its own timestep budget")
    if not env_factories:
        raise ValueError("at least one curriculum stage is required")
    if algorithm not in ("dqn", "ppo", "maskable-ppo"):
        raise ValueError("algorithm must be 'dqn', 'ppo', or 'maskable-ppo'")

    selected = config or PPOTrainingConfig()
    try:
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise RuntimeError(
            "Curriculum training requires the optional '.[training]' dependencies"
        ) from error
    if algorithm == "maskable-ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as error:
            raise RuntimeError(
                "Masked training requires the optional '.[training]' dependencies"
            ) from error

    selected.output_dir.mkdir(parents=True, exist_ok=True)
    selected.tensorboard_log.mkdir(parents=True, exist_ok=True)
    model: Any = None
    for index, (factory, timesteps) in enumerate(zip(env_factories, stage_timesteps, strict=True)):

        def monitored(build: Callable[[], gym.Env[Any, Any]] = factory) -> Any:
            return Monitor(build())

        env = DummyVecEnv([monitored])
        if model is None:
            builder = {"dqn": DQN, "ppo": PPO}.get(algorithm)
            if builder is None:
                builder = MaskablePPO
            model = builder(
                "MultiInputPolicy",
                env,
                seed=selected.seed,
                tensorboard_log=str(selected.tensorboard_log),
                verbose=1,
            )
        else:
            model.set_env(env)
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"rlattack-{algorithm}-curriculum",
        )
        model.save(str(selected.output_dir / f"stage-{index:02d}"))
        env.close()
    model.save(str(selected.output_dir / "final"))
    return model
