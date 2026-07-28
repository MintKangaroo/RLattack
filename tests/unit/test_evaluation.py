import pytest

from rlattack.agents import GreedyAgent, RandomAgent
from rlattack.env import AttackPathEnv
from rlattack.evaluation import evaluate_agent, evaluate_agents
from rlattack.generator import generate_scenario


def test_evaluation_metrics_are_reproducible() -> None:
    scenario = generate_scenario("small", "easy", seed=4)

    def factory(seed: int) -> AttackPathEnv:
        del seed
        return AttackPathEnv(scenario, step_budget=50)

    seeds = (1, 2)

    first = evaluate_agent("greedy", GreedyAgent(), factory, seeds)
    second = evaluate_agent("greedy", GreedyAgent(), factory, seeds)

    assert first == second
    assert first.episodes == 2
    assert first.success_rate == 1.0
    assert first.mean_steps > 0
    assert first.mean_path_cost == 2.0


def test_multiple_agents_share_benchmark_seeds() -> None:
    scenario = generate_scenario("small", "easy", seed=8)
    results = evaluate_agents(
        {"greedy": GreedyAgent(), "random": RandomAgent(seed=5)},
        lambda seed: AttackPathEnv(scenario, step_budget=30),
        (3,),
    )

    assert set(results) == {"greedy", "random"}
    assert all(metric.episodes == 1 for metric in results.values())


def test_empty_evaluation_seed_set_is_rejected() -> None:
    scenario = generate_scenario("small", "easy", seed=0)

    with pytest.raises(ValueError, match="at least one"):
        evaluate_agent(
            "greedy",
            GreedyAgent(),
            lambda seed: AttackPathEnv(scenario),
            (),
        )
