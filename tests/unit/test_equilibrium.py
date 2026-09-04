from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from rlattack.defender import DEFAULT_ARMS, DefenderArm
from rlattack.equilibrium import Equilibrium, build_payoffs, solve_grid, solve_zero_sum
from rlattack.experiment import AgentName, ExperimentConfig
from rlattack.scenario import Scenario


def test_matching_pennies_mixes_evenly() -> None:
    solved = solve_zero_sum(np.array([[1.0, -1.0], [-1.0, 1.0]]), iterations=4000)

    assert solved.attacker_mixture[0] == pytest.approx(0.5, abs=0.05)
    assert solved.defender_mixture[0] == pytest.approx(0.5, abs=0.05)
    assert solved.value == pytest.approx(0.0, abs=0.02)


def test_a_dominant_row_yields_a_pure_strategy() -> None:
    solved = solve_zero_sum(np.array([[1.0, 0.8], [0.2, 0.1]]), iterations=2000)

    assert solved.attacker_mixture == (1.0, 0.0)
    assert solved.value == pytest.approx(0.8)


def test_solver_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="non-empty matrix"):
        solve_zero_sum(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="non-empty matrix"):
        solve_zero_sum(np.zeros(3))
    with pytest.raises(ValueError, match="iterations"):
        solve_zero_sum(np.ones((2, 2)), iterations=0)


def test_support_hides_negligible_weights() -> None:
    solved = Equilibrium(
        attacker_labels=("a", "b"),
        defender_labels=("x", "y"),
        payoffs=((1.0, 0.0), (0.0, 1.0)),
        attacker_mixture=(0.999, 0.001),
        defender_mixture=(0.5, 0.5),
        value=0.5,
        iterations=10,
    )

    assert solved.attacker_support == {"a": 0.999}
    assert set(solved.defender_support) == {"x", "y"}


def test_the_payoff_grid_covers_every_pair() -> None:
    config = ExperimentConfig(
        size="small", difficulty="easy", seed=1, benchmark_episodes=2, stochastic=False
    )

    matrix = build_payoffs(config, ("greedy", "shortest-path"), DEFAULT_ARMS[:2])

    assert matrix.shape == (2, 2)
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))

    with pytest.raises(ValueError, match="at least one policy"):
        build_payoffs(config, (), DEFAULT_ARMS)


def test_solving_the_grid_keeps_the_labels() -> None:
    calls: list[tuple[int, int]] = []

    def fake_payoffs(
        config: ExperimentConfig,
        attackers: Sequence[AgentName],
        defenders: Sequence[DefenderArm],
        scenario_builder: Callable[[int], Scenario] | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        calls.append((len(attackers), len(defenders)))
        return np.array([[0.9, 0.4], [0.2, 0.1]])

    solved = solve_grid(
        ExperimentConfig(),
        ("greedy", "random"),
        DEFAULT_ARMS[:2],
        iterations=500,
        payoff_builder=fake_payoffs,
    )

    assert calls == [(2, 2)]
    assert solved.attacker_labels == ("greedy", "random")
    assert solved.defender_labels == (DEFAULT_ARMS[0].label, DEFAULT_ARMS[1].label)
    assert solved.attacker_support == {"greedy": 1.0}
