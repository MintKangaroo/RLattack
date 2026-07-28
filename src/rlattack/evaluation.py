"""Reproducible benchmark metrics for simulator agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from rlattack.agents import Agent
from rlattack.env import ACTION_NAMES, AttackPathEnv


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Aggregated metrics for one Agent over a fixed set of Episodes."""

    agent_name: str
    episodes: int
    successes: int
    success_rate: float
    mean_steps: float
    mean_reward: float
    mean_detection_risk: float
    mean_path_cost: float


def evaluate_agent(
    agent_name: str,
    agent: Agent,
    env_factory: Callable[[int], AttackPathEnv],
    seeds: tuple[int, ...],
) -> BenchmarkMetrics:
    """Evaluate one Agent on deterministic environments created from ``seeds``."""

    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    successes = 0
    rewards: list[float] = []
    steps: list[int] = []
    risks: list[float] = []
    path_costs: list[float] = []
    for seed in seeds:
        env = env_factory(seed)
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
        if terminated and info.get("action_name") == ACTION_NAMES[7]:
            successes += 1
        rewards.append(total_reward)
        episode_steps = cast(int, info["steps"])
        episode_risk = cast(float, info["detection_risk"])
        steps.append(episode_steps)
        risks.append(episode_risk)
        path_costs.append(float(episode_steps))
    episode_count = len(seeds)
    return BenchmarkMetrics(
        agent_name=agent_name,
        episodes=episode_count,
        successes=successes,
        success_rate=successes / episode_count,
        mean_steps=float(np.mean(steps)),
        mean_reward=float(np.mean(rewards)),
        mean_detection_risk=float(np.mean(risks)),
        mean_path_cost=float(np.mean(path_costs)),
    )


def evaluate_agents(
    agents: Mapping[str, Agent],
    env_factory: Callable[[int], AttackPathEnv],
    seeds: tuple[int, ...],
) -> dict[str, BenchmarkMetrics]:
    """Evaluate multiple Agents with identical seeds and environment factory."""

    return {name: evaluate_agent(name, agent, env_factory, seeds) for name, agent in agents.items()}
