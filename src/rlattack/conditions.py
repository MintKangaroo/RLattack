"""Evaluate one agent across the experimental condition grid.

A policy reported only under the control condition says nothing about how it holds up
when the world pushes back. Every condition here reuses the same scenario class and the
same seed list, so the episodes stay paired and :mod:`rlattack.stats` applies directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from rlattack.agents import Agent
from rlattack.env import AttackPathEnv, DynamicsConfig
from rlattack.evaluation import BenchmarkMetrics, evaluate_agent
from rlattack.experiment import DefenderMode, DiscoveryMode, ExperimentConfig, benchmark_seeds
from rlattack.generator import generate_scenario
from rlattack.reward import build_reward_config
from rlattack.scenario import Scenario


@dataclass(frozen=True)
class Condition:
    """One point of the defender x discovery grid."""

    defender: DefenderMode = "passive"
    discovery: DiscoveryMode = "exact"

    def __post_init__(self) -> None:
        if self.defender not in ("passive", "adaptive", "targeted"):
            raise ValueError("defender must be passive, adaptive, or targeted")
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
        """Return ``config`` with this condition's treatments applied.

        Only the two treated axes change. Rebuilding the config field by field drops
        whatever the grid does not know about - ``detection_threshold`` among them -
        and silently evaluates a condition that was never configured.
        """

        return replace(config, defender=self.defender, discovery=self.discovery)


CONDITION_GRID: tuple[Condition, ...] = (
    Condition("passive", "exact"),
    Condition("adaptive", "exact"),
    Condition("passive", "noisy"),
    Condition("adaptive", "noisy"),
)

CONTROL_LABEL = CONDITION_GRID[0].label

ATTENTION_GRID: tuple[Condition, ...] = (
    Condition("passive", "exact"),
    Condition("adaptive", "exact"),
    Condition("targeted", "exact"),
    Condition("targeted", "noisy"),
)
"""The grid for the targeted-attention study.

Kept separate from :data:`CONDITION_GRID` so that the published v0.9 condition tables
stay comparable rather than silently gaining two columns.
"""


def run_condition_sweep(
    config: ExperimentConfig,
    agent_factory: Callable[[int], Agent],
    conditions: Sequence[Condition] = CONDITION_GRID,
    scenario_builder: Callable[[int], Scenario] | None = None,
) -> dict[str, BenchmarkMetrics]:
    """Evaluate one agent under every condition, on identical scenarios and seeds.

    ``scenario_builder`` replaces the generator, so a policy can be swept on the
    topology family it was trained on.
    """

    if not conditions:
        raise ValueError("at least one condition is required")
    seeds = benchmark_seeds(config)
    reward_config = build_reward_config(config.reward_strategy)
    # One observation space for the whole sweep, so a single policy fits every cell.
    # The monitoring channel reads all zeros wherever the defender is not targeted.
    observation_config = replace(
        config.observation_config(),
        expose_monitoring=any(condition.defender == "targeted" for condition in conditions),
    )
    build_scenario = scenario_builder or (
        lambda seed: generate_scenario(config.size, config.difficulty, seed)
    )

    def env_factory(condition: Condition) -> Callable[[int], AttackPathEnv]:
        applied = condition.apply(config)
        dynamics: DynamicsConfig = applied.dynamics()
        defender = applied.defender_config()

        def build(seed: int) -> AttackPathEnv:
            return AttackPathEnv(
                build_scenario(seed),
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
