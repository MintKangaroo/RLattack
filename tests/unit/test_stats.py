import pytest

from rlattack.evaluation import BenchmarkMetrics, EpisodeOutcome
from rlattack.stats import (
    bootstrap_difference_interval,
    compare,
    compare_benchmarks,
    paired_permutation_test,
)


def metrics(name: str, rewards: tuple[float, ...]) -> BenchmarkMetrics:
    outcomes = tuple(
        EpisodeOutcome(
            seed=index,
            success=reward > 0,
            detected=False,
            steps=10 + index,
            reward=reward,
            detection_risk=0.1,
            path_cost=1.0,
        )
        for index, reward in enumerate(rewards)
    )
    return BenchmarkMetrics(
        agent_name=name,
        episodes=len(outcomes),
        successes=sum(outcome.success for outcome in outcomes),
        success_rate=1.0,
        detection_rate=0.0,
        mean_steps=0.0,
        mean_reward=0.0,
        mean_detection_risk=0.0,
        mean_path_cost=0.0,
        std_steps=0.0,
        std_reward=0.0,
        reward_ci_low=0.0,
        reward_ci_high=0.0,
        outcomes=outcomes,
    )


LOW = (1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.1)
HIGH = (5.0, 5.6, 4.4, 5.3, 4.9, 4.5, 5.7, 5.2)


def test_a_clear_separation_is_significant() -> None:
    p_value = paired_permutation_test(LOW, HIGH, iterations=2000, seed=1)

    assert p_value < 0.05


def test_identical_samples_cannot_be_significant() -> None:
    assert paired_permutation_test(LOW, LOW, iterations=10) == 1.0


def test_bootstrap_interval_brackets_the_difference() -> None:
    low, high = bootstrap_difference_interval(LOW, HIGH, iterations=2000, seed=1)

    assert low < 4.0 < high
    assert high - low > 0.0


def test_paired_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="same number of episodes"):
        paired_permutation_test(LOW, HIGH[:2])
    with pytest.raises(ValueError, match="at least one episode"):
        paired_permutation_test((), ())
    with pytest.raises(ValueError, match="iterations"):
        paired_permutation_test(LOW, HIGH, iterations=0)
    with pytest.raises(ValueError, match="iterations"):
        bootstrap_difference_interval(LOW, HIGH, iterations=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_difference_interval(LOW, HIGH, confidence=1.5)


def test_comparison_records_the_reference_and_direction() -> None:
    result = compare("low", LOW, "high", HIGH, iterations=500, seed=2)

    assert result.reference == "low"
    assert result.candidate == "high"
    assert result.episodes == len(LOW)
    assert result.mean_difference > 0
    assert result.significant is True


def test_benchmark_comparison_covers_every_supported_metric() -> None:
    records = {"low": metrics("low", LOW), "high": metrics("high", HIGH)}

    for metric in ("reward", "steps", "success"):
        comparisons = compare_benchmarks(records, "low", metric=metric, iterations=200, seed=3)

        assert [item.candidate for item in comparisons] == ["high"]
        assert comparisons[0].metric == metric

    with pytest.raises(ValueError, match="unknown reference"):
        compare_benchmarks(records, "missing")
    with pytest.raises(ValueError, match="metric must be"):
        compare_benchmarks(records, "low", metric="risk")
