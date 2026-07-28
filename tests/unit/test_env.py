import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rlattack.env import ACTION_NAMES, Action, AttackPathEnv
from rlattack.scenario import (
    AccessEdge,
    Credential,
    Host,
    NetworkEdge,
    Objective,
    Privilege,
    PrivilegeEdge,
    Scenario,
    Service,
    Vulnerability,
)


def make_scenario() -> Scenario:
    return Scenario(
        id="env-demo",
        name="Environment demo",
        entry_host_ids=("web",),
        hosts=(Host(id="web"), Host(id="db")),
        services=(Service(id="http", host_id="web", name="http", port=80),),
        vulnerabilities=(
            Vulnerability(id="weak-http", service_id="http", name="simulated weakness"),
        ),
        credentials=(
            Credential(id="web-user", host_id="web", username="agent", privilege_id="user"),
        ),
        privileges=(
            Privilege(id="user", name="user", level=10),
            Privilege(id="admin", name="admin", level=90),
        ),
        objectives=(
            Objective(
                id="collect",
                name="collect objective",
                host_id="db",
                required_privilege_id="admin",
            ),
        ),
        network_edges=(NetworkEdge(source_host_id="web", target_host_id="db"),),
        access_edges=(AccessEdge(vulnerability_id="weak-http", credential_id="web-user"),),
        privilege_edges=(PrivilegeEdge(source_privilege_id="user", target_privilege_id="admin"),),
    )


def test_gymnasium_checker_passes() -> None:
    check_env(AttackPathEnv(make_scenario()))


def test_reset_with_same_seed_reproduces_observation() -> None:
    env = AttackPathEnv(make_scenario())

    first, first_info = env.reset(seed=7)
    second, second_info = env.reset(seed=7)

    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert np.array_equal(first_info["action_mask"], second_info["action_mask"])


def test_action_mask_and_simulated_success_path() -> None:
    env = AttackPathEnv(make_scenario(), step_budget=20)
    observation, info = env.reset(seed=1)

    assert observation["discovered_hosts"].tolist() == [1, 0]
    assert info["action_mask"].tolist() == [1, 1, 0, 0, 0, 0, 1, 0, 1]
    assert ACTION_NAMES[Action.SCAN_SERVICE] == "scan_service"

    for action in (
        Action.SCAN_SERVICE,
        Action.ENUMERATE_SERVICE,
        Action.VALIDATE_VULNERABILITY,
        Action.ATTEMPT_SIMULATED_ACCESS,
        Action.ESCALATE_SIMULATED_PRIVILEGE,
        Action.PIVOT_SIMULATED_NETWORK,
        Action.COLLECT_SIMULATED_OBJECTIVE,
    ):
        observation, reward, terminated, truncated, info = env.step(np.int64(action))

    assert terminated is True
    assert truncated is False
    assert reward > 0
    assert observation["discovered_hosts"].tolist() == [1, 1]
    assert info["action_name"] == "collect_simulated_objective"


def test_invalid_action_is_penalized_and_budget_truncates() -> None:
    env = AttackPathEnv(make_scenario(), step_budget=1)
    env.reset(seed=1)

    _, reward, terminated, truncated, info = env.step(np.int64(Action.COLLECT_SIMULATED_OBJECTIVE))

    assert reward < 0
    assert terminated is False
    assert truncated is True
    assert info["steps"] == 1


def test_invalid_action_value_and_constructor_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="step_budget"):
        AttackPathEnv(make_scenario(), step_budget=0)
    with pytest.raises(ValueError, match="at least one host"):
        AttackPathEnv(Scenario(id="empty", name="empty"))

    env = AttackPathEnv(make_scenario())
    env.reset()
    with pytest.raises(ValueError, match="action must be an integer"):
        env.step(np.int64(99))


def test_duplicate_actions_are_penalized() -> None:
    env = AttackPathEnv(make_scenario())
    env.reset()
    actions = (
        Action.SCAN_SERVICE,
        Action.SCAN_SERVICE,
        Action.ENUMERATE_SERVICE,
        Action.ENUMERATE_SERVICE,
        Action.VALIDATE_VULNERABILITY,
        Action.VALIDATE_VULNERABILITY,
        Action.ATTEMPT_SIMULATED_ACCESS,
        Action.ATTEMPT_SIMULATED_ACCESS,
        Action.ESCALATE_SIMULATED_PRIVILEGE,
        Action.ESCALATE_SIMULATED_PRIVILEGE,
        Action.PIVOT_SIMULATED_NETWORK,
        Action.PIVOT_SIMULATED_NETWORK,
    )
    rewards = [env.step(np.int64(action))[1] for action in actions]

    assert all(reward < 0 for reward in rewards[1::2])


def test_step_after_termination_requires_reset() -> None:
    env = AttackPathEnv(make_scenario())
    env.reset()
    env.step(np.int64(Action.STOP))

    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.int64(Action.STOP))
