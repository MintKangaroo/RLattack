"""Equilibrium analysis over the attacker x defender policy grid.

One-sided sweeps answer "which defender is best against *this* attacker" and "which
attacker beats *that* defender". Neither answers what both sides should play when each
knows the other is choosing too. This module builds the payoff matrix over the policy
grid and solves it.

The outcome payoffs are complementary - an objective captured is a win for one side and
a loss for the other - so the grid is treated as a zero-sum matrix game. The defender's
response cost breaks that exactly, which is why the reported value is the attacker's
mean episode reward and the solution is described as an equilibrium of the outcome
game rather than of the full cost-adjusted one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from rlattack.agents import Agent
from rlattack.defender import DEFAULT_ARMS, DefenderArm
from rlattack.env import AttackPathEnv
from rlattack.evaluation import run_episode_outcome
from rlattack.experiment import AgentName, ExperimentConfig, benchmark_seeds, create_agent
from rlattack.game import ATTACKER_ARMS, attacker_reward
from rlattack.generator import generate_scenario
from rlattack.reward import build_reward_config
from rlattack.scenario import Scenario

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class Equilibrium:
    """A solved matrix game over the attacker x defender policy grid."""

    attacker_labels: tuple[str, ...]
    defender_labels: tuple[str, ...]
    payoffs: tuple[tuple[float, ...], ...]
    attacker_mixture: tuple[float, ...]
    defender_mixture: tuple[float, ...]
    value: float
    iterations: int

    @property
    def attacker_support(self) -> dict[str, float]:
        """Return the attacker strategies the equilibrium actually plays."""

        return {
            label: weight
            for label, weight in zip(self.attacker_labels, self.attacker_mixture, strict=True)
            if weight > 0.01
        }

    @property
    def defender_support(self) -> dict[str, float]:
        """Return the defender strategies the equilibrium actually plays."""

        return {
            label: weight
            for label, weight in zip(self.defender_labels, self.defender_mixture, strict=True)
            if weight > 0.01
        }


def solve_zero_sum(payoffs: FloatArray, iterations: int = 20_000) -> Equilibrium:
    """Solve a zero-sum matrix game by fictitious play.

    Fictitious play converges to a Nash equilibrium for zero-sum games and needs no
    linear-programming dependency. ``payoffs[i][j]`` is the row player's payoff when the
    row plays ``i`` and the column plays ``j``; the row player maximizes.
    """

    if payoffs.ndim != 2 or payoffs.size == 0:
        raise ValueError("payoffs must be a non-empty matrix")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rows: int = int(payoffs.shape[0])
    columns: int = int(payoffs.shape[1])
    row_counts: FloatArray = np.zeros(rows)
    column_counts: FloatArray = np.zeros(columns)
    row_beliefs: FloatArray = np.zeros(columns)
    column_beliefs: FloatArray = np.zeros(rows)
    for _ in range(iterations):
        row_choice = int(np.argmax(payoffs @ _normalize(row_beliefs, columns)))
        column_choice = int(np.argmin(_normalize(column_beliefs, rows) @ payoffs))
        row_counts[row_choice] += 1
        column_counts[column_choice] += 1
        row_beliefs[column_choice] += 1
        column_beliefs[row_choice] += 1
    row_mixture: FloatArray = row_counts / row_counts.sum()
    column_mixture: FloatArray = column_counts / column_counts.sum()
    return Equilibrium(
        attacker_labels=(),
        defender_labels=(),
        payoffs=_as_rows(payoffs),
        attacker_mixture=tuple(float(weight) for weight in row_mixture),
        defender_mixture=tuple(float(weight) for weight in column_mixture),
        value=float(row_mixture @ payoffs @ column_mixture),
        iterations=iterations,
    )


def _as_rows(payoffs: FloatArray) -> tuple[tuple[float, ...], ...]:
    """Return the matrix as plain nested tuples for a serializable record."""

    rows = cast(list[list[float]], payoffs.tolist())
    return tuple(tuple(float(value) for value in row) for row in rows)


def _normalize(counts: FloatArray, size: int) -> FloatArray:
    total = counts.sum()
    if total == 0:
        return np.full(size, 1.0 / size)
    normalized: FloatArray = counts / total
    return normalized


def build_payoffs(
    config: ExperimentConfig,
    attacker_arms: Sequence[AgentName] = ATTACKER_ARMS,
    defender_arms: Sequence[DefenderArm] = DEFAULT_ARMS,
    scenario_builder: Callable[[int], Scenario] | None = None,
) -> FloatArray:
    """Evaluate every attacker x defender pair on the same seeds.

    The entry is the attacker's mean episode reward, so the row player maximizes.

    ``scenario_builder`` replaces the generator, so the grid can be solved on a
    held-out topology family. Structure decides whether the grid has anything to
    trade off: routing around a defender needs somewhere else to route, and only the
    mesh family offers more than one node-disjoint route to the objective.
    """

    if not attacker_arms or not defender_arms:
        raise ValueError("both sides need at least one policy")
    build = scenario_builder or (
        lambda seed: generate_scenario(config.size, config.difficulty, seed)
    )
    seeds = benchmark_seeds(config)
    reward_config = build_reward_config(config.reward_strategy)
    # Every cell of the grid uses the same observation space, including the monitoring
    # channel, so that the attacker arms are compared on one interface. The channel is
    # all zeros against a defender that watches uniformly, so exposing it everywhere
    # costs the columns that do not use it nothing.
    observation_config = replace(config.observation_config(), expose_monitoring=True)
    dynamics = config.dynamics()

    def agent_for(name: AgentName, seed: int) -> Agent:
        return create_agent(name, build(seed), seed=seed)

    matrix: FloatArray = np.zeros((len(attacker_arms), len(defender_arms)))
    for row, attacker in enumerate(attacker_arms):
        for column, defender in enumerate(defender_arms):
            rewards = []
            for seed in seeds:
                env = AttackPathEnv(
                    build(seed),
                    step_budget=config.step_budget,
                    reward_config=reward_config,
                    dynamics=dynamics,
                    observation_config=observation_config,
                    defender=defender.config,
                )
                rewards.append(
                    attacker_reward(run_episode_outcome(agent_for(attacker, seed), env, seed))
                )
            matrix[row, column] = float(np.mean(rewards))
    return matrix


def solve_grid(
    config: ExperimentConfig,
    attacker_arms: Sequence[AgentName] = ATTACKER_ARMS,
    defender_arms: Sequence[DefenderArm] = DEFAULT_ARMS,
    *,
    iterations: int = 20_000,
    payoff_builder: Callable[..., FloatArray] = build_payoffs,
    scenario_builder: Callable[[int], Scenario] | None = None,
) -> Equilibrium:
    """Build the payoff grid and solve it, keeping the policy labels attached."""

    matrix = payoff_builder(config, attacker_arms, defender_arms, scenario_builder)
    solved = solve_zero_sum(matrix, iterations=iterations)
    return Equilibrium(
        attacker_labels=tuple(str(name) for name in attacker_arms),
        defender_labels=tuple(arm.label for arm in defender_arms),
        payoffs=solved.payoffs,
        attacker_mixture=solved.attacker_mixture,
        defender_mixture=solved.defender_mixture,
        value=solved.value,
        iterations=solved.iterations,
    )
