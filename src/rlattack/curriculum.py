"""Scenario curricula and transfer evaluation across scenario classes.

A curriculum is only meaningful when one policy can be applied to every stage, which
is what :meth:`rlattack.env.ObservationConfig.for_curriculum` guarantees: identical
observation and action spaces for every generated size and difficulty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np

from rlattack.agents import Agent
from rlattack.defender import ContextualDefender, DefenderConfig
from rlattack.env import AttackPathEnv, DynamicsConfig, Observation, ObservationConfig
from rlattack.evaluation import BenchmarkMetrics, evaluate_agent
from rlattack.families import FAMILIES, build_scenario
from rlattack.game import episode_defender_reward
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario
from rlattack.reward import RewardStrategy, build_reward_config


@dataclass(frozen=True)
class CurriculumStage:
    """One scenario class, and how long to train on it."""

    size: ScenarioSize
    difficulty: Difficulty
    timesteps: int = 20_000
    family: str | None = None
    hosts: int = 8

    def __post_init__(self) -> None:
        if self.size not in ("small", "medium", "large"):
            raise ValueError("size must be small, medium, or large")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium, or hard")
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if self.family is not None and self.family not in FAMILIES:
            raise ValueError(f"unknown topology family: {self.family}")
        if self.hosts < 2:
            raise ValueError("hosts must be at least 2")

    @property
    def label(self) -> str:
        """Return a stable label for tables and exports."""

        if self.family is not None:
            return f"{self.family}/{self.hosts}"
        return f"{self.size}/{self.difficulty}"

    def step_budget(self, base: int) -> int:
        """Scale a base step budget to this stage's scenario size.

        A budget that fits ``small`` starves ``large``: without scaling, a transfer
        table reports budget exhaustion as if it were a generalization failure. A
        family stage scales on its host count instead, which is the only size it has.
        """

        if self.family is not None:
            return max(1, round(base * self.hosts / _FAMILY_REFERENCE_HOSTS))
        return max(1, round(base * _BUDGET_SCALE[self.size]))


_BUDGET_SCALE: dict[str, float] = {"small": 1.0, "medium": 1.6, "large": 3.0}
_FAMILY_REFERENCE_HOSTS = 6

DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("small", "easy", 20_000),
    CurriculumStage("small", "hard", 20_000),
    CurriculumStage("medium", "medium", 30_000),
    CurriculumStage("medium", "hard", 30_000),
)


def family_curriculum(
    family: str, host_counts: Sequence[int] = (5, 6, 8, 10), timesteps: int = 20_000
) -> tuple[CurriculumStage, ...]:
    """Return a curriculum over one topology family, growing the host count.

    The generator emits one shape, so a policy trained on its curriculum has never seen
    an alternative route to anything. Training on a family whose instances have more
    than one node-disjoint route is what gives a learner the chance to discover that
    routing around a watcher is a strategy at all.
    """

    if not host_counts:
        raise ValueError("a family curriculum needs at least one host count")
    return tuple(
        CurriculumStage("medium", "hard", timesteps=timesteps, family=family, hosts=hosts)
        for hosts in host_counts
    )


def scale_curriculum(
    stages: Sequence[CurriculumStage], total_timesteps: int
) -> tuple[CurriculumStage, ...]:
    """Rescale a curriculum to a total budget, keeping each stage's share.

    Training budget is the first thing to vary when a published policy is a floor
    rather than a ceiling, so it is a parameter rather than a constant.
    """

    if not stages:
        raise ValueError("at least one curriculum stage is required")
    if total_timesteps < len(stages):
        raise ValueError("total_timesteps must leave at least one step per stage")
    budget = sum(stage.timesteps for stage in stages)
    # Rescaling changes only the budget. Rebuilding the stage from a subset of its
    # fields is how a family curriculum silently reverts to generated scenarios.
    return tuple(
        replace(
            stage,
            timesteps=max(1, round(total_timesteps * stage.timesteps / budget)),
        )
        for stage in stages
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
            build_scenario(stage.family, stage.hosts, seed)
            if stage.family is not None
            else generate_scenario(stage.size, stage.difficulty, seed),
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
        previous: Sequence[Callable[[int], AttackPathEnv]] = (),
        defender_policy: ContextualDefender | None = None,
    ) -> None:
        if not seeds:
            raise ValueError("a stage needs at least one training seed")
        self.stage = stage
        self.seeds = tuple(seeds)
        # Sampling earlier stages alongside the current one keeps them alive. Training
        # each stage in isolation makes the policy forget the previous ones outright,
        # and the curriculum's rolling reward hides it because Stable-Baselines3 carries
        # its episode buffer across stages.
        factories = [env_factory, *previous]
        self._envs = [factory(seed) for factory in factories for seed in self.seeds]
        self.defender_policy = defender_policy
        if defender_policy is not None:
            for env in self._envs:
                env.defender_policy = defender_policy
                env.defender = defender_policy.config
        self._pending_outcome: dict[str, Any] | None = None
        self._current = self._envs[0]
        self.observation_space = self._current.observation_space
        self.action_space = self._current.action_space

    @property
    def pool_size(self) -> int:
        """Return how many scenarios this stage draws from, across all mixed classes."""

        return len(self._envs)

    @property
    def current(self) -> AttackPathEnv:
        """Return the environment the last ``reset`` selected."""

        return self._current

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._close_defender_episode()
        index = int(self.np_random.integers(len(self._envs)))
        self._current = self._envs[index]
        return self._current.reset(seed=int(self.np_random.integers(2**31)), options=options)

    def step(self, action: np.int64) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._current.step(action)
        if self.defender_policy is not None and (terminated or truncated):
            self._pending_outcome = info
        return observation, reward, terminated, truncated, info

    def _close_defender_episode(self) -> None:
        """Score the finished episode for the defender and start its next one.

        Stable-Baselines3 owns the episode loop, so the adapting defender is driven from
        the environment's own reset rather than from a game loop outside it.
        """

        policy = self.defender_policy
        if policy is None:
            return
        if self._pending_outcome is not None:
            policy.finish_episode(episode_defender_reward(self._pending_outcome))
            self._pending_outcome = None
        policy.start_episode()

    def action_masks(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Delegate the action mask so maskable algorithms see the live scenario."""

        masks: np.ndarray[Any, np.dtype[np.bool_]] = self._current.action_masks()
        return masks


def _bind_stage(
    agent_factory: Callable[[CurriculumStage, int], Agent], stage: CurriculumStage
) -> Callable[[int], Agent]:
    """Adapt a stage-aware factory to the seed-only factory the evaluator expects."""

    def build(seed: int) -> Agent:
        return agent_factory(stage, seed)

    return build


def evaluate_transfer(
    agent_factory: Callable[[CurriculumStage, int], Agent],
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

    ``agent_factory`` receives the stage as well as the seed. A graph-aware baseline
    must be built from the scenario it will actually act in; building it from one fixed
    scenario class silently mis-indexes it on every other stage.
    """

    if not stages:
        raise ValueError("at least one curriculum stage is required")
    return {
        stage.label: evaluate_agent(
            stage.label,
            _bind_stage(agent_factory, stage),
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
