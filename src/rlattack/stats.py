"""Significance testing for benchmark comparisons.

Benchmarks run every agent on the same seed list, so the episodes are *paired*: seed
``k`` gives every agent the same scenario and the same environment stream. The tests
here exploit that pairing, and they are resampling-based so they carry no normality
assumption and need no SciPy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlattack.evaluation import BenchmarkMetrics

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

DEFAULT_ITERATIONS = 10_000


@dataclass(frozen=True)
class Comparison:
    """One paired comparison of a candidate against a reference."""

    reference: str
    candidate: str
    metric: str
    episodes: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        """Whether the paired difference is significant at ``alpha``."""

        return self.p_value < self.alpha


def _paired(reference: Sequence[float], candidate: Sequence[float]) -> FloatArray:
    if len(reference) != len(candidate):
        raise ValueError("paired tests require the same number of episodes per agent")
    if not reference:
        raise ValueError("paired tests require at least one episode")
    paired: FloatArray = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return paired


def paired_permutation_test(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> float:
    """Return the two-sided p-value of a paired sign-flip permutation test.

    Under the null hypothesis the two agents are exchangeable on each seed, so the sign
    of every paired difference is equally likely to be positive or negative.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    differences = _paired(reference, candidate)
    observed = abs(float(np.mean(differences)))
    if not np.any(differences):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(iterations, differences.size))
    resampled = np.abs((signs * differences).mean(axis=1))
    extreme = int(np.count_nonzero(resampled >= observed))
    return (extreme + 1) / (iterations + 1)


def bootstrap_difference_interval(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the paired mean difference."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    differences = _paired(reference, candidate)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(iterations, differences.size))
    means = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(low), float(high)


def compare(
    reference_name: str,
    reference: Sequence[float],
    candidate_name: str,
    candidate: Sequence[float],
    *,
    metric: str = "reward",
    alpha: float = 0.05,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> Comparison:
    """Compare one candidate against a reference on paired per-episode samples."""

    differences = _paired(reference, candidate)
    ci_low, ci_high = bootstrap_difference_interval(
        reference, candidate, iterations=iterations, seed=seed
    )
    return Comparison(
        reference=reference_name,
        candidate=candidate_name,
        metric=metric,
        episodes=differences.size,
        mean_difference=float(np.mean(differences)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=paired_permutation_test(reference, candidate, iterations=iterations, seed=seed),
        alpha=alpha,
    )


def _samples(metrics: BenchmarkMetrics, metric: str) -> tuple[float, ...]:
    if metric == "reward":
        return tuple(outcome.reward for outcome in metrics.outcomes)
    if metric == "steps":
        return tuple(float(outcome.steps) for outcome in metrics.outcomes)
    if metric == "success":
        return tuple(float(outcome.success) for outcome in metrics.outcomes)
    raise ValueError("metric must be reward, steps, or success")


def compare_benchmarks(
    metrics: Mapping[str, BenchmarkMetrics],
    reference_name: str,
    *,
    metric: str = "reward",
    alpha: float = 0.05,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> list[Comparison]:
    """Compare every benchmarked agent against ``reference_name`` on paired episodes."""

    if reference_name not in metrics:
        raise ValueError(f"unknown reference agent: {reference_name}")
    reference = _samples(metrics[reference_name], metric)
    return [
        compare(
            reference_name,
            reference,
            name,
            _samples(record, metric),
            metric=metric,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
        )
        for name, record in metrics.items()
        if name != reference_name
    ]
