import pytest

from rlattack.agents import Agent, GreedyAgent, RandomAgent, ShortestPathOracle
from rlattack.env import AttackPathEnv, DynamicsConfig
from rlattack.evaluation import evaluate_agent, evaluate_agents
from rlattack.generator import generate_scenario


def env_factory(seed: int) -> AttackPathEnv:
    return AttackPathEnv(
        generate_scenario("small", "easy", seed),
        step_budget=60,
        dynamics=DynamicsConfig.deterministic(),
    )


def greedy_factory(seed: int) -> Agent:
    del seed
    return GreedyAgent()


def test_evaluation_metrics_are_reproducible() -> None:
    seeds = (1, 2, 3, 4)

    first = evaluate_agent("greedy", greedy_factory, env_factory, seeds)
    second = evaluate_agent("greedy", greedy_factory, env_factory, seeds)

    assert first == second
    assert first.episodes == 4
    assert first.success_rate == 1.0
    assert first.detection_rate == 0.0
    assert first.mean_steps > 0
    assert first.std_steps >= 0.0
    assert first.reward_ci_low <= first.mean_reward <= first.reward_ci_high
    assert tuple(outcome.seed for outcome in first.outcomes) == seeds


def test_benchmark_seeds_generate_independent_scenarios() -> None:
    def varied_factory(seed: int) -> AttackPathEnv:
        return AttackPathEnv(generate_scenario("medium", "hard", seed), step_budget=90)

    assert varied_factory(1).scenario.id != varied_factory(2).scenario.id

    metrics = evaluate_agent("greedy", greedy_factory, varied_factory, (1, 2, 3, 4, 5, 6))
    rewards = {round(outcome.reward, 6) for outcome in metrics.outcomes}

    assert len(rewards) > 1, "per-seed scenarios must not replay one fixed episode"
    assert metrics.std_reward > 0.0


def test_single_episode_interval_collapses_to_the_mean() -> None:
    metrics = evaluate_agent("greedy", greedy_factory, env_factory, (5,))

    assert metrics.std_reward == 0.0
    assert metrics.reward_ci_low == metrics.mean_reward == metrics.reward_ci_high


def test_multiple_agents_share_benchmark_seeds() -> None:
    def oracle_factory(seed: int) -> Agent:
        return ShortestPathOracle(generate_scenario("small", "easy", seed))

    results = evaluate_agents(
        {
            "greedy": greedy_factory,
            "random": lambda seed: RandomAgent(seed=seed),
            "shortest-path": oracle_factory,
        },
        env_factory,
        (3, 4),
    )

    assert set(results) == {"greedy", "random", "shortest-path"}
    assert all(metric.episodes == 2 for metric in results.values())


def test_empty_evaluation_seed_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_agent("greedy", greedy_factory, env_factory, ())
