"""Evaluate one agent across the experimental condition grid.

A policy reported only under the control condition says nothing about how it holds up
when the world pushes back. Every condition here reuses the same scenario class and the
same seed list, so the episodes stay paired and :mod:`rlattack.stats` applies directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rlattack.agents import Agent
from rlattack.env import AttackPathEnv, DynamicsConfig
from rlattack.evaluation import BenchmarkMetrics, evaluate_agent
from rlattack.experiment import DefenderMode, DiscoveryMode, ExperimentConfig, benchmark_seeds
from rlattack.generator import generate_scenario
from rlattack.reward import build_reward_config


@dataclass(frozen=True)
class Condition:
    """One point of the defender x discovery grid."""

    defender: DefenderMode = "passive"
    discovery: DiscoveryMode = "exact"

    def __post_init__(self) -> None:
        if self.defender not in ("passive", "adaptive"):
            raise ValueError("defender must be passive or adaptive")
        if self.discovery not in ("exact", "noisy"):
            raise ValueError("discovery must be exact or noisy")

    @property
    def label(self) -> str:
        """Return a stable ``defender/discovery`` label for tables and exports."""

        return f"{self.defender}/{self.discovery}"

    @property
    def is_control(self) -> bool:
        """Whether this is the untreated control condition."""

        return self.defender == "passive" and self.discovery == "exact"

    def apply(self, config: ExperimentConfig) -> ExperimentConfig:
        """Return ``config`` with this condition's treatments applied."""

        return ExperimentConfig(
            size=config.size,
            difficulty=config.difficulty,
            seed=config.seed,
            agent=config.agent,
            reward_strategy=config.reward_strategy,
            step_budget=config.step_budget,
            benchmark_episodes=config.benchmark_episodes,
            stochastic=config.stochastic,
            observation=config.observation,
            defender=self.defender,
            discovery=self.discovery,
        )


CONDITION_GRID: tuple[Condition, ...] = (
    Condition("passive", "exact"),
    Condition("adaptive", "exact"),
    Condition("passive", "noisy"),
    Condition("adaptive", "noisy"),
)

CONTROL_LABEL = CONDITION_GRID[0].label


def run_condition_sweep(
    config: ExperimentConfig,
    agent_factory: Callable[[int], Agent],
    conditions: Sequence[Condition] = CONDITION_GRID,
) -> dict[str, BenchmarkMetrics]:
    """Evaluate one agent under every condition, on identical scenarios and seeds."""

    if not conditions:
        raise ValueError("at least one condition is required")
    seeds = benchmark_seeds(config)
    reward_config = build_reward_config(config.reward_strategy)
    observation_config = config.observation_config()

    def env_factory(condition: Condition) -> Callable[[int], AttackPathEnv]:
        applied = condition.apply(config)
        dynamics: DynamicsConfig = applied.dynamics()
        defender = applied.defender_config()

        def build(seed: int) -> AttackPathEnv:
            return AttackPathEnv(
                generate_scenario(config.size, config.difficulty, seed),
                step_budget=config.step_budget,
                reward_config=reward_config,
                dynamics=dynamics,
                observation_config=observation_config,
                defender=defender,
            )

        return build

    return {
        condition.label: evaluate_agent(
            condition.label, agent_factory, env_factory(condition), seeds
        )
        for condition in conditions
    }
