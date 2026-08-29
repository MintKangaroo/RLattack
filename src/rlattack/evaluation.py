"""Reproducible benchmark metrics for simulator agents."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from rlattack.agents import Agent, reset_agent
from rlattack.env import AttackPathEnv

_CONFIDENCE_Z = 1.96


@dataclass(frozen=True)
class EpisodeOutcome:
    """Per-episode summary retained so aggregate statistics stay auditable."""

    seed: int
    success: bool
    detected: bool
    steps: int
    reward: float
    detection_risk: float
    path_cost: float
    defender_actions: int = 0
    defender_false_positives: int = 0


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Aggregated metrics for one Agent over a fixed set of episodes.

    Means alone cannot support a research claim, so the record also carries the
    dispersion and a 95% normal-approximation interval for the episode reward.
    """

    agent_name: str
    episodes: int
    successes: int
    success_rate: float
    detection_rate: float
    mean_steps: float
    mean_reward: float
    mean_detection_risk: float
    mean_path_cost: float
    std_steps: float
    std_reward: float
    reward_ci_low: float
    reward_ci_high: float
    outcomes: tuple[EpisodeOutcome, ...] = ()


def _confidence_interval(mean: float, std: float, count: int) -> tuple[float, float]:
    if count < 2:
        return (mean, mean)
    margin = _CONFIDENCE_Z * std / math.sqrt(count)
    return (mean - margin, mean + margin)


def run_episode_outcome(
    agent: Agent, env: AttackPathEnv, seed: int, *, reset_policy: bool = True
) -> EpisodeOutcome:
    """Run one seeded episode and return its terminal statistics."""

    if reset_policy:
        reset_agent(agent, seed=seed)
    observation, raw_info = env.reset(seed=seed)
    info: dict[str, object] = raw_info
    total_reward = 0.0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = agent.predict(observation, info)
        observation, reward, terminated, truncated, raw_info = env.step(action)
        info = raw_info
        total_reward += reward
    return EpisodeOutcome(
        seed=seed,
        success=bool(info["objective_captured"]),
        detected=bool(info["detected"]),
        steps=cast(int, info["steps"]),
        reward=total_reward,
        detection_risk=cast(float, info["detection_risk"]),
        path_cost=cast(float, info["path_cost"]),
        defender_actions=cast(int, info["defender_actions"]),
        defender_false_positives=cast(int, info["defender_false_positives"]),
    )


def evaluate_agent(
    agent_name: str,
    agent_factory: Callable[[int], Agent],
    env_factory: Callable[[int], AttackPathEnv],
    seeds: tuple[int, ...],
) -> BenchmarkMetrics:
    """Evaluate one Agent across independently seeded scenarios.

    ``agent_factory`` and ``env_factory`` both receive the seed so that generalization
    benchmarks can vary the scenario instead of replaying one fixed graph.
    """

    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    outcomes = tuple(
        run_episode_outcome(agent_factory(seed), env_factory(seed), seed) for seed in seeds
    )
    rewards = [outcome.reward for outcome in outcomes]
    steps = [outcome.steps for outcome in outcomes]
    successes = sum(outcome.success for outcome in outcomes)
    detections = sum(outcome.detected for outcome in outcomes)
    episode_count = len(outcomes)
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards, ddof=1)) if episode_count > 1 else 0.0
    std_steps = float(np.std(steps, ddof=1)) if episode_count > 1 else 0.0
    ci_low, ci_high = _confidence_interval(mean_reward, std_reward, episode_count)
    return BenchmarkMetrics(
        agent_name=agent_name,
        episodes=episode_count,
        successes=successes,
        success_rate=successes / episode_count,
        detection_rate=detections / episode_count,
        mean_steps=float(np.mean(steps)),
        mean_reward=mean_reward,
        mean_detection_risk=float(np.mean([outcome.detection_risk for outcome in outcomes])),
        mean_path_cost=float(np.mean([outcome.path_cost for outcome in outcomes])),
        std_steps=std_steps,
        std_reward=std_reward,
        reward_ci_low=ci_low,
        reward_ci_high=ci_high,
        outcomes=outcomes,
    )


def evaluate_agents(
    agent_factories: Mapping[str, Callable[[int], Agent]],
    env_factory: Callable[[int], AttackPathEnv],
    seeds: tuple[int, ...],
) -> dict[str, BenchmarkMetrics]:
    """Evaluate multiple Agents with identical seeds and environment factory."""

    return {
        name: evaluate_agent(name, factory, env_factory, seeds)
        for name, factory in agent_factories.items()
    }
