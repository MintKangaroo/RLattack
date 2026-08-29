import pytest

from rlattack.agents import Agent, GreedyAgent
from rlattack.conditions import (
    CONDITION_GRID,
    CONTROL_LABEL,
    Condition,
    run_condition_sweep,
)
from rlattack.experiment import ExperimentConfig


def greedy_factory(seed: int) -> Agent:
    del seed
    return GreedyAgent()


def test_condition_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="defender"):
        Condition(defender="chaotic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="discovery"):
        Condition(discovery="blind")  # type: ignore[arg-type]


def test_the_grid_starts_from_an_untreated_control() -> None:
    assert CONDITION_GRID[0].is_control is True
    assert CONDITION_GRID[0].label == CONTROL_LABEL == "passive/exact"
    assert [condition.label for condition in CONDITION_GRID[1:]] == [
        "adaptive/exact",
        "passive/noisy",
        "adaptive/noisy",
    ]
    assert all(condition.is_control is False for condition in CONDITION_GRID[1:])


def test_a_condition_only_overrides_its_own_treatments() -> None:
    config = ExperimentConfig(size="large", seed=9, reward_strategy="sparse", stochastic=False)

    applied = Condition("adaptive", "noisy").apply(config)

    assert applied.defender == "adaptive"
    assert applied.discovery == "noisy"
    assert applied.size == "large"
    assert applied.seed == 9
    assert applied.reward_strategy == "sparse"
    assert applied.stochastic is False


def test_the_sweep_evaluates_every_condition_on_shared_seeds() -> None:
    config = ExperimentConfig(
        size="small", difficulty="easy", seed=4, benchmark_episodes=3, stochastic=False
    )

    results = run_condition_sweep(config, greedy_factory)

    assert list(results) == [condition.label for condition in CONDITION_GRID]
    assert all(metric.episodes == 3 for metric in results.values())
    seeds = {tuple(outcome.seed for outcome in metric.outcomes) for metric in results.values()}
    assert len(seeds) == 1, "conditions must be paired on one seed list"


def test_the_treatments_actually_change_the_episodes() -> None:
    config = ExperimentConfig(
        size="medium", difficulty="hard", seed=1, benchmark_episodes=6, step_budget=80
    )

    results = run_condition_sweep(config, greedy_factory)

    assert results["passive/noisy"].mean_reward != results[CONTROL_LABEL].mean_reward


def test_an_empty_condition_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one condition"):
        run_condition_sweep(ExperimentConfig(), greedy_factory, ())
