"""Scenario curricula and transfer evaluation across scenario classes.

A curriculum is only meaningful when one policy can be applied to every stage, which
is what :meth:`rlattack.env.ObservationConfig.for_curriculum` guarantees: identical
observation and action spaces for every generated size and difficulty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rlattack.agents import Agent
from rlattack.defender import DefenderConfig
from rlattack.env import AttackPathEnv, DynamicsConfig, Observation, ObservationConfig
from rlattack.evaluation import BenchmarkMetrics, evaluate_agent
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario
from rlattack.reward import RewardStrategy, build_reward_config


@dataclass(frozen=True)
class CurriculumStage:
    """One scenario class, and how long to train on it."""

    size: ScenarioSize
    difficulty: Difficulty
    timesteps: int = 20_000

    def __post_init__(self) -> None:
        if self.size not in ("small", "medium", "large"):
            raise ValueError("size must be small, medium, or large")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium, or hard")
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")

    @property
    def label(self) -> str:
        """Return a stable ``size/difficulty`` label for tables and exports."""

        return f"{self.size}/{self.difficulty}"

    def step_budget(self, base: int) -> int:
        """Scale a base step budget to this stage's scenario size.

        A budget that fits ``small`` starves ``large``: without scaling, a transfer
        table reports budget exhaustion as if it were a generalization failure.
        """

        return max(1, round(base * _BUDGET_SCALE[self.size]))


_BUDGET_SCALE: dict[str, float] = {"small": 1.0, "medium": 1.6, "large": 3.0}

DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("small", "easy", 20_000),
    CurriculumStage("small", "hard", 20_000),
    CurriculumStage("medium", "medium", 30_000),
    CurriculumStage("medium", "hard", 30_000),
)

HELD_OUT_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage("large", "medium"),
    CurriculumStage("large", "hard"),
)

SIZES: tuple[ScenarioSize, ...] = ("small", "medium", "large")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

ALL_STAGES: tuple[CurriculumStage, ...] = tuple(
    CurriculumStage(size, difficulty) for size in SIZES for difficulty in DIFFICULTIES
)


def stage_env_factory(
    stage: CurriculumStage,
    *,
    step_budget: int = 64,
    reward_strategy: RewardStrategy = "risk-aware",
    dynamics: DynamicsConfig | None = None,
    defender: DefenderConfig | None = None,
    observation_config: ObservationConfig | None = None,
) -> Callable[[int], AttackPathEnv]:
    """Build a seeded environment factory for one curriculum stage.

    The observation interface defaults to the fixed curriculum capacities so that the
    same policy fits every stage.
    """

    reward_config = build_reward_config(reward_strategy)
    interface = observation_config or ObservationConfig.for_curriculum()
    scaled_budget = stage.step_budget(step_budget)

    def build(seed: int) -> AttackPathEnv:
        return AttackPathEnv(
            generate_scenario(stage.size, stage.difficulty, seed),
            step_budget=scaled_budget,
            reward_config=reward_config,
            dynamics=dynamics,
            observation_config=interface,
            defender=defender,
        )

    return build


class StageEnv(gym.Env[Observation, np.int64]):
    """Draw a fresh scenario from one stage on every reset.

    Stable-Baselines3 builds its environment once, so training directly on an
    :class:`~rlattack.env.AttackPathEnv` would repeat a single graph for the whole
    stage and teach the policy that graph rather than the class. This wrapper holds one
    environment per training seed and rotates between them, which is only sound because
    the curriculum capacities give every scenario identical spaces.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: CurriculumStage,
        seeds: Sequence[int],
        env_factory: Callable[[int], AttackPathEnv],
    ) -> None:
        if not seeds:
            raise ValueError("a stage needs at least one training seed")
        self.stage = stage
        self.seeds = tuple(seeds)
        self._envs = [env_factory(seed) for seed in self.seeds]
        self._current = self._envs[0]
        self.observation_space = self._current.observation_space
        self.action_space = self._current.action_space

    @property
    def current(self) -> AttackPathEnv:
        """Return the environment the last ``reset`` selected."""

        return self._current

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self._envs)))
        self._current = self._envs[index]
        return self._current.reset(seed=int(self.np_random.integers(2**31)), options=options)

    def step(self, action: np.int64) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        return self._current.step(action)


def evaluate_transfer(
    agent_factory: Callable[[int], Agent],
    seeds: tuple[int, ...],
    stages: Sequence[CurriculumStage] = ALL_STAGES,
    *,
    step_budget: int = 64,
    reward_strategy: RewardStrategy = "risk-aware",
    dynamics: DynamicsConfig | None = None,
    defender: DefenderConfig | None = None,
) -> dict[str, BenchmarkMetrics]:
    """Evaluate one agent on every stage, holding the interface and seeds fixed.

    Every stage uses the same seed list, so results across scenario classes are paired
    and can be compared with :mod:`rlattack.stats`.
    """

    if not stages:
        raise ValueError("at least one curriculum stage is required")
    return {
        stage.label: evaluate_agent(
            stage.label,
            agent_factory,
            stage_env_factory(
                stage,
                step_budget=step_budget,
                reward_strategy=reward_strategy,
                dynamics=dynamics,
                defender=defender,
            ),
            seeds,
        )
        for stage in stages
    }
