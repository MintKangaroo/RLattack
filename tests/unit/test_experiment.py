from typing import cast

import pytest

from rlattack.agents import GreedyAgent, RandomAgent, RuleBasedAgent, ShortestPathOracle
from rlattack.env import DynamicsConfig
from rlattack.experiment import (
    AgentName,
    ExperimentConfig,
    build_dashboard_data,
    create_agent,
    run_episode,
)
from rlattack.generator import generate_scenario


def test_experiment_config_validates_public_inputs() -> None:
    assert ExperimentConfig().seed == 42

    invalid_values = (
        {"size": "tiny"},
        {"difficulty": "extreme"},
        {"agent": "unknown"},
        {"reward_strategy": "unknown"},
        {"step_budget": 0},
        {"benchmark_episodes": 0},
        {"step_budget": 10_000},
        {"benchmark_episodes": 10_000},
    )
    for values in invalid_values:
        with pytest.raises(ValueError):
            ExperimentConfig(**values)


def test_agent_factory_covers_all_baselines() -> None:
    scenario = generate_scenario("small", "easy", seed=3)

    assert isinstance(create_agent("random", scenario, seed=3), RandomAgent)
    assert isinstance(create_agent("greedy", scenario, seed=3), GreedyAgent)
    assert isinstance(create_agent("rule-based", scenario, seed=3), RuleBasedAgent)
    assert isinstance(create_agent("shortest-path", scenario, seed=3), ShortestPathOracle)
    with pytest.raises(ValueError, match="unsupported"):
        create_agent(cast(AgentName, "unknown"), scenario, seed=3)


def test_episode_runner_returns_trace_and_real_graph_cost() -> None:
    scenario = generate_scenario("small", "easy", seed=7)
    result = run_episode(
        scenario,
        agent_name="greedy",
        seed=7,
        step_budget=64,
        dynamics=DynamicsConfig.deterministic(),
    )

    assert result.success is True
    assert result.terminated is True
    assert result.truncated is False
    assert result.steps == len(result.trace)
    assert result.path_cost > 0
    assert scenario.objectives[0].id in result.visited_nodes
    assert result.trace[-1].action == "collect_simulated_objective"

    truncated = run_episode(
        scenario,
        agent_name="greedy",
        seed=7,
        step_budget=1,
        dynamics=DynamicsConfig.deterministic(),
    )
    assert truncated.success is False
    assert truncated.truncated is True


def test_dashboard_data_is_reproducible_and_complete() -> None:
    config = ExperimentConfig(
        size="small",
        difficulty="hard",
        seed=9,
        agent="rule-based",
        reward_strategy="cost-aware",
        step_budget=64,
        benchmark_episodes=2,
        stochastic=False,
    )

    first = build_dashboard_data(config)
    second = build_dashboard_data(config)

    assert first == second
    assert first["schema_version"] == "2.0"
    assert first["episode"]["success"] is True
    assert len(first["benchmarks"]) == 4
    rule_metrics = next(
        metric for metric in first["benchmarks"] if metric["agent_name"] == "rule-based"
    )
    assert rule_metrics["success_rate"] == 1.0
    assert first["scenario"]["oracle_route"][0] == "host-00"
    assert first["safety"]["network_access"] is False
    assert first["benchmark_protocol"]["mode"] == "per-seed-scenario"
    assert first["benchmark_protocol"]["seeds"] == [9, 10]
    assert "std_reward" in rule_metrics
    assert "outcomes" not in rule_metrics
    assert first["episode"]["detected"] is False
    assert first["episode"]["trace"][0]["target_id"] is not None


def test_dashboard_data_uses_default_config() -> None:
    data = build_dashboard_data()

    assert data["config"]["size"] == "medium"
    assert data["reward"]["strategy"] == "risk-aware"
