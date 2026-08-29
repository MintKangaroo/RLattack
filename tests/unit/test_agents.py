import numpy as np
import pytest

from rlattack.agents import (
    GreedyAgent,
    RandomAgent,
    RuleBasedAgent,
    ShortestPathOracle,
    action_name,
)
from rlattack.env import ACTION_NAMES, Action, AttackPathEnv, DynamicsConfig
from rlattack.generator import generate_scenario
from rlattack.scenario import Host, NetworkEdge, Objective, Scenario


def test_oracle_reaches_the_objective_more_efficiently_than_greedy() -> None:
    scenario = generate_scenario("medium", "hard", seed=42)
    steps = {}
    for name, agent in (("greedy", GreedyAgent()), ("oracle", ShortestPathOracle(scenario))):
        env = AttackPathEnv(scenario, step_budget=120, dynamics=DynamicsConfig.deterministic())
        observation, info = env.reset(seed=0)
        terminated = truncated = False
        while not terminated and not truncated:
            observation, _, terminated, truncated, info = env.step(agent.predict(observation, info))
        assert info["objective_captured"] is True
        steps[name] = info["steps"]

    assert steps["oracle"] < steps["greedy"]


def test_baselines_choose_valid_actions() -> None:
    scenario = generate_scenario("small", "easy", seed=3)
    env = AttackPathEnv(scenario)
    observation, info = env.reset(seed=3)
    agents = (
        RandomAgent(seed=1),
        GreedyAgent(),
        RuleBasedAgent(),
        ShortestPathOracle(scenario),
    )

    for agent in agents:
        action = agent.predict(observation, info)
        assert info["action_mask"][int(action)] == 1


def test_random_agent_is_reproducible() -> None:
    scenario = generate_scenario("small", "medium", seed=2)
    env = AttackPathEnv(scenario)
    observation, info = env.reset()
    first = RandomAgent(seed=9)
    second = RandomAgent(seed=9)

    first_actions = [first.predict(observation, info) for _ in range(5)]
    second_actions = [second.predict(observation, info) for _ in range(5)]

    assert first_actions == second_actions


def stop_only_info(stride: int = 1) -> dict[str, object]:
    mask = np.zeros(len(Action) * stride, dtype=np.int8)
    mask[int(Action.STOP) * stride] = 1
    return {"action_mask": mask, "target_count": stride}


def test_rule_agent_falls_back_to_the_only_valid_action() -> None:
    agent = RuleBasedAgent()
    observation = {"state": np.zeros(1, dtype=np.int8)}

    assert agent.predict(observation, stop_only_info()) == int(Action.STOP)


def test_action_name_validates_catalogue_bounds() -> None:
    assert action_name(Action.SCAN_SERVICE) == "scan_service"
    assert action_name(len(ACTION_NAMES) - 1) == "stop"
    with pytest.raises(ValueError, match="outside"):
        action_name(len(ACTION_NAMES))


def test_empty_action_mask_is_rejected() -> None:
    empty: dict[str, object] = {
        "action_mask": np.zeros(len(Action), dtype=np.int8),
        "target_count": 1,
    }
    with pytest.raises(ValueError, match="no valid actions"):
        RandomAgent().predict({}, empty)


def test_action_mask_shape_and_target_count_are_validated() -> None:
    with pytest.raises(ValueError, match="one entry"):
        RandomAgent().predict({}, {"action_mask": np.zeros(1, dtype=np.int8), "target_count": 1})
    with pytest.raises(ValueError, match="target_count"):
        RandomAgent().predict({}, {"action_mask": np.zeros(len(Action), dtype=np.int8)})

    full: dict[str, object] = {
        "action_mask": np.ones(len(Action), dtype=np.int8),
        "target_count": 1,
    }
    with pytest.raises(RuntimeError, match="priority list"):
        GreedyAgent(priority=()).predict({}, full)


def test_random_agent_prefers_progress_over_stopping() -> None:
    stride = 1
    mask = np.zeros(len(Action) * stride, dtype=np.int8)
    mask[int(Action.SCAN_SERVICE)] = 1
    mask[int(Action.STOP)] = 1
    info: dict[str, object] = {"action_mask": mask, "target_count": stride}

    actions = {int(RandomAgent(seed=seed).predict({}, info)) for seed in range(20)}

    assert actions == {int(Action.SCAN_SERVICE)}


def test_oracle_falls_back_to_greedy_when_no_route_action_is_available() -> None:
    scenario = generate_scenario("small", "easy", seed=3)
    oracle = ShortestPathOracle(scenario)
    observation = {
        "discovered_hosts": np.ones(len(scenario.hosts), dtype=np.int8),
        "reachable_hosts": np.ones(len(scenario.hosts), dtype=np.int8),
    }

    assert oracle.predict(observation, stop_only_info()) == int(Action.STOP)


def test_oracle_validates_graph_and_supports_implicit_entry() -> None:
    with pytest.raises(ValueError, match="at least one host"):
        ShortestPathOracle(Scenario(id="empty", name="empty"))
    with pytest.raises(ValueError, match="at least one objective"):
        ShortestPathOracle(Scenario(id="no-goal", name="no goal", hosts=(Host(id="entry"),)))
    disconnected = Scenario(
        id="disconnected",
        name="disconnected",
        hosts=(Host(id="entry"), Host(id="goal")),
        entry_host_ids=("entry",),
        objectives=(Objective(id="objective", name="goal", host_id="goal"),),
    )
    with pytest.raises(ValueError, match="path"):
        ShortestPathOracle(disconnected)
    implicit_entry = Scenario(
        id="implicit-entry",
        name="implicit entry",
        hosts=(Host(id="entry"), Host(id="goal")),
        objectives=(Objective(id="objective", name="goal", host_id="goal"),),
        network_edges=(NetworkEdge(source_host_id="entry", target_host_id="goal"),),
    )

    assert ShortestPathOracle(implicit_entry).route == ("entry", "goal")


def test_rule_agent_requires_rules_that_cover_the_action_space() -> None:
    with pytest.raises(RuntimeError, match="rule list"):
        RuleBasedAgent(rules=()).predict({}, stop_only_info())
