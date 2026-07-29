import numpy as np
import pytest

from rlattack.agents import (
    GreedyAgent,
    RandomAgent,
    RuleBasedAgent,
    ShortestPathOracle,
    action_name,
)
from rlattack.env import ACTION_NAMES, Action, AttackPathEnv
from rlattack.generator import generate_scenario
from rlattack.scenario import Host, NetworkEdge, Objective, Scenario


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


def test_rule_agent_stops_when_no_configured_rule_is_valid() -> None:
    agent = RuleBasedAgent(rules=(Action.COLLECT_SIMULATED_OBJECTIVE,))
    observation = {"state": np.zeros(1, dtype=np.int8)}
    mask = np.zeros(len(Action), dtype=np.int8)
    mask[Action.STOP] = 1
    info: dict[str, object] = {"action_mask": mask}

    assert agent.predict(observation, info) == Action.STOP


def test_action_name_validates_catalogue_bounds() -> None:
    assert action_name(Action.SCAN_SERVICE) == "scan_service"
    assert action_name(len(ACTION_NAMES) - 1) == "stop"
    with pytest.raises(ValueError, match="outside"):
        action_name(len(ACTION_NAMES))


def test_empty_action_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="no valid actions"):
        RandomAgent().predict({}, {"action_mask": np.zeros(len(Action), dtype=np.int8)})


def test_action_mask_shape_and_greedy_priority_failures_are_rejected() -> None:
    with pytest.raises(ValueError, match="one entry"):
        RandomAgent().predict({}, {"action_mask": np.zeros(1, dtype=np.int8)})

    with pytest.raises(RuntimeError, match="priority list"):
        GreedyAgent(priority=()).predict({}, {"action_mask": np.ones(len(Action), dtype=np.int8)})


def test_oracle_falls_back_to_greedy_when_all_hosts_are_discovered() -> None:
    scenario = generate_scenario("small", "easy", seed=3)
    oracle = ShortestPathOracle(scenario)
    observation = {
        "discovered_hosts": np.ones(len(scenario.hosts), dtype=np.int8),
    }
    mask = np.zeros(len(Action), dtype=np.int8)
    mask[Action.STOP] = 1

    assert oracle.predict(observation, {"action_mask": mask}) == Action.STOP


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
