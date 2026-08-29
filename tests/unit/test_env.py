from typing import cast

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rlattack.defender import ContextualDefender, DefenderConfig
from rlattack.env import (
    ACTION_NAMES,
    Action,
    AttackPathEnv,
    DynamicsConfig,
    ObservationConfig,
)
from rlattack.generator import generate_scenario
from rlattack.scenario import (
    AccessEdge,
    Credential,
    Host,
    NetworkEdge,
    Objective,
    Privilege,
    PrivilegeEdge,
    Scenario,
    SecurityControl,
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
        security_controls=(
            SecurityControl(id="edr", name="edr", detection_probability=0.4, host_ids=("web",)),
        ),
        network_edges=(NetworkEdge(source_host_id="web", target_host_id="db"),),
        access_edges=(AccessEdge(vulnerability_id="weak-http", credential_id="web-user"),),
        privilege_edges=(PrivilegeEdge(source_privilege_id="user", target_privilege_id="admin"),),
    )


def deterministic_env(**kwargs: object) -> AttackPathEnv:
    return AttackPathEnv(
        make_scenario(),
        dynamics=DynamicsConfig.deterministic(),
        **kwargs,  # type: ignore[arg-type]
    )


SUCCESS_PATH: tuple[tuple[Action, int], ...] = (
    (Action.SCAN_SERVICE, 0),
    (Action.ENUMERATE_SERVICE, 0),
    (Action.VALIDATE_VULNERABILITY, 0),
    (Action.ATTEMPT_SIMULATED_ACCESS, 0),
    (Action.ESCALATE_SIMULATED_PRIVILEGE, 0),
    (Action.DISCOVER_HOST, 1),
    (Action.PIVOT_SIMULATED_NETWORK, 1),
    (Action.COLLECT_SIMULATED_OBJECTIVE, 0),
)


def test_gymnasium_checker_passes() -> None:
    check_env(AttackPathEnv(make_scenario()), skip_render_check=True)


def test_reset_with_same_seed_reproduces_observation() -> None:
    env = AttackPathEnv(make_scenario())

    first, first_info = env.reset(seed=7)
    second, second_info = env.reset(seed=7)

    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert np.array_equal(first_info["action_mask"], second_info["action_mask"])
    assert first_info["target_count"] == 2
    assert first_info["objective_captured"] is False


def test_action_mask_targets_individual_graph_elements() -> None:
    env = deterministic_env(step_budget=20)
    observation, info = env.reset(seed=1)
    mask = info["action_mask"].reshape(len(Action), env.target_count)

    assert observation["discovered_hosts"].tolist() == [1, 0]
    assert mask[Action.DISCOVER_HOST].tolist() == [0, 1]
    assert mask[Action.SCAN_SERVICE].tolist() == [1, 0]
    assert mask[Action.PIVOT_SIMULATED_NETWORK].tolist() == [0, 0]
    assert mask[Action.STOP].tolist() == [1, 0]
    assert ACTION_NAMES[Action.SCAN_SERVICE] == "scan_service"


def test_simulated_success_path_reaches_the_objective() -> None:
    env = deterministic_env(step_budget=20)
    observation, info = env.reset(seed=1)

    for action_type, target in SUCCESS_PATH:
        observation, reward, terminated, truncated, info = env.step(
            env.encode_action(action_type, target)
        )

    assert terminated is True
    assert truncated is False
    assert reward > 0
    assert info["objective_captured"] is True
    assert info["detected"] is False
    assert info["action_name"] == "collect_simulated_objective"
    assert info["target_id"] == "collect"
    assert info["affected_nodes"] == ("db", "collect")
    assert info["outcome"] == "objective"
    assert info["path_cost"] == 1.0
    assert observation["reachable_hosts"].tolist() == [1, 1]


def test_pivot_requires_a_foothold_on_the_source_host() -> None:
    env = deterministic_env(step_budget=20)
    env.reset(seed=1)
    env.step(env.encode_action(Action.DISCOVER_HOST, 1))

    _, reward, _, _, info = env.step(env.encode_action(Action.PIVOT_SIMULATED_NETWORK, 1))

    assert info["valid_action"] is False
    assert reward < 0


def test_target_ids_are_reported_for_every_action_kind() -> None:
    env = deterministic_env()
    env.reset(seed=1)
    observed = {}
    for action_type, target in SUCCESS_PATH:
        _, _, _, _, info = env.step(env.encode_action(action_type, target))
        observed[info["action_name"]] = info["target_id"]

    assert observed["attempt_simulated_access"] == "weak-http->web-user"
    assert observed["escalate_simulated_privilege"] == "user->admin"
    assert observed["scan_service"] == "http"
    assert observed["discover_host"] == "db"


def test_stop_and_out_of_range_targets_have_no_target_id() -> None:
    env = deterministic_env()
    env.reset(seed=1)

    _, _, terminated, _, info = env.step(env.encode_action(Action.STOP, 0))

    assert terminated is True
    assert info["target_id"] is None
    assert info["outcome"] == "stopped"

    env.reset(seed=1)
    _, _, _, _, info = env.step(env.encode_action(Action.COLLECT_SIMULATED_OBJECTIVE, 1))

    assert info["target_id"] is None
    assert info["valid_action"] is False


def test_failed_attempts_raise_detection_risk() -> None:
    env = AttackPathEnv(
        make_scenario(),
        dynamics=DynamicsConfig(base_success_probability=0.01, failed_attempt_risk=0.2),
    )
    env.reset(seed=1)
    env.step(env.encode_action(Action.SCAN_SERVICE, 0))
    env.step(env.encode_action(Action.ENUMERATE_SERVICE, 0))

    _, reward, _, _, info = env.step(env.encode_action(Action.VALIDATE_VULNERABILITY, 0))

    assert info["outcome"] == "failed"
    assert info["valid_action"] is True
    assert reward < 0
    assert info["detection_risk"] > 0.2


def test_crossing_the_detection_threshold_ends_the_episode() -> None:
    env = AttackPathEnv(
        make_scenario(),
        dynamics=DynamicsConfig(stochastic=False, pivot_risk=1.0, detection_threshold=0.9),
    )
    env.reset(seed=1)
    for action_type, target in SUCCESS_PATH[:6]:
        env.step(env.encode_action(action_type, target))

    _, reward, terminated, _, info = env.step(env.encode_action(Action.PIVOT_SIMULATED_NETWORK, 1))

    assert terminated is True
    assert info["detected"] is True
    assert info["objective_captured"] is False
    assert info["outcome"] == "detected"
    assert reward < 0


def test_invalid_action_is_penalized_and_budget_truncates() -> None:
    env = deterministic_env(step_budget=1)
    env.reset(seed=1)

    _, reward, terminated, truncated, info = env.step(
        env.encode_action(Action.COLLECT_SIMULATED_OBJECTIVE, 0)
    )

    assert reward < 0
    assert terminated is False
    assert truncated is True
    assert info["steps"] == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.encode_action(Action.STOP, 0))


def test_invalid_action_value_and_constructor_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="step_budget"):
        AttackPathEnv(make_scenario(), step_budget=0)
    with pytest.raises(ValueError, match="at least one host"):
        AttackPathEnv(Scenario(id="empty", name="empty"))

    env = AttackPathEnv(make_scenario())
    env.reset()
    with pytest.raises(ValueError, match="action must be an integer"):
        env.step(np.int64(999))
    with pytest.raises(ValueError, match="action_type"):
        env.encode_action(len(Action), 0)
    with pytest.raises(ValueError, match="target"):
        env.encode_action(Action.STOP, env.target_count)
    with pytest.raises(ValueError, match="outside"):
        env.decode_action(len(Action) * env.target_count)


def test_dynamics_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="base_success_probability"):
        DynamicsConfig(base_success_probability=0.0)
    with pytest.raises(ValueError, match="minimum_success_probability"):
        DynamicsConfig(minimum_success_probability=1.5)
    with pytest.raises(ValueError, match="detection_threshold"):
        DynamicsConfig(detection_threshold=0.0)
    with pytest.raises(ValueError, match="failed_attempt_risk"):
        DynamicsConfig(failed_attempt_risk=-0.1)
    with pytest.raises(ValueError, match="pivot_risk"):
        DynamicsConfig(pivot_risk=1.5)
    with pytest.raises(ValueError, match="risk_reference_hosts"):
        DynamicsConfig(risk_reference_hosts=0)


def test_duplicate_actions_are_penalized() -> None:
    env = deterministic_env()
    env.reset()
    rewards = []
    for action_type, target in SUCCESS_PATH[:-1]:
        rewards.append(env.step(env.encode_action(action_type, target))[1])
        rewards.append(env.step(env.encode_action(action_type, target))[1])

    assert all(reward < 0 for reward in rewards[1::2])


def test_hosts_without_controls_use_the_default_detection_increment() -> None:
    scenario = make_scenario().model_copy(update={"security_controls": ()})
    env = AttackPathEnv(scenario, dynamics=DynamicsConfig.deterministic())
    env.reset(seed=1)
    env.step(env.encode_action(Action.SCAN_SERVICE, 0))

    _, _, _, _, info = env.step(env.encode_action(Action.ENUMERATE_SERVICE, 0))

    assert info["detection_risk"] == pytest.approx(0.05)


def test_step_after_termination_requires_reset() -> None:
    env = deterministic_env()
    env.reset()
    env.step(env.encode_action(Action.STOP, 0))

    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.encode_action(Action.STOP, 0))


def test_hosts_without_modeled_credentials_can_be_pivoted_from() -> None:
    scenario = make_scenario().model_copy(update={"credentials": (), "access_edges": ()})
    env = AttackPathEnv(scenario, dynamics=DynamicsConfig.deterministic())
    env.reset(seed=1)
    env.step(env.encode_action(Action.DISCOVER_HOST, 1))

    _, reward, _, _, info = env.step(env.encode_action(Action.PIVOT_SIMULATED_NETWORK, 1))

    assert info["valid_action"] is True
    assert info["outcome"] == "success"
    assert reward > 0


def test_observation_capacities_are_validated() -> None:
    with pytest.raises(ValueError, match="host_capacity"):
        ObservationConfig(host_capacity=0)
    with pytest.raises(ValueError, match="alert_levels"):
        ObservationConfig(alert_levels=1)
    with pytest.raises(ValueError, match="smaller than the scenario"):
        AttackPathEnv(make_scenario(), observation_config=ObservationConfig(host_capacity=1))


def test_fixed_capacities_hide_the_network_size() -> None:
    env = AttackPathEnv(make_scenario(), observation_config=ObservationConfig.for_curriculum())
    observation, info = env.reset(seed=1)

    assert observation["discovered_hosts"].shape == (16,)
    assert observation["known_services"].shape == (32,)
    assert info["target_count"] == 32
    assert int(observation["discovered_hosts"].sum()) == 1


def test_the_agent_observes_a_quantized_alert_level_by_default() -> None:
    env = AttackPathEnv(
        make_scenario(),
        dynamics=DynamicsConfig(stochastic=False, pivot_risk=0.5, detection_threshold=0.9),
    )
    observation, info = env.reset(seed=1)

    assert "detection_risk" not in observation
    assert observation["alert_level"].tolist() == [1, 0, 0]
    assert info["alert_level"] == 0

    for action_type, target in SUCCESS_PATH[:7]:
        observation, _, _, _, info = env.step(env.encode_action(action_type, target))

    assert info["detection_risk"] > 0.5
    assert observation["alert_level"].tolist() == [0, 1, 0]
    assert info["alert_level"] == 1


def test_exact_risk_can_be_exposed_for_analysis_runs() -> None:
    env = AttackPathEnv(
        make_scenario(),
        observation_config=ObservationConfig(expose_exact_risk=True),
        dynamics=DynamicsConfig.deterministic(),
    )
    observation, _ = env.reset(seed=1)

    assert observation["detection_risk"].tolist() == [0.0]
    assert "alert_level" in observation


def test_the_defender_is_passive_by_default() -> None:
    env = deterministic_env()
    env.reset(seed=1)

    _, _, _, _, info = env.step(env.encode_action(Action.SCAN_SERVICE, 0))

    assert info["defender_action"] == "none"
    assert info["defender_actions"] == 0


def test_an_adaptive_defender_hardens_monitoring_and_revokes_credentials() -> None:
    env = AttackPathEnv(
        make_scenario(),
        step_budget=40,
        dynamics=DynamicsConfig.deterministic(),
        defender=DefenderConfig(
            enabled=True,
            alert_threshold=0.0,
            response_cooldown=1,
            revocation_probability=1.0,
            hardening_step=1.0,
            response_latency=0,
            observation_noise=0.0,
        ),
    )
    env.reset(seed=1)
    for action_type, target in SUCCESS_PATH[:4]:
        _, _, _, _, info = env.step(env.encode_action(action_type, target))

    assert info["revoked_credentials"] >= 1
    assert info["defender_action"] == "revoke_credential"

    observation = env._observation()

    assert int(observation["acquired_credentials"].sum()) == 0
    assert int(observation["acquired_privileges"].sum()) == 0


def test_hardening_raises_the_detection_cost_of_reached_hosts() -> None:
    def enumeration_risk(defender: DefenderConfig) -> float:
        env = AttackPathEnv(
            make_scenario(),
            dynamics=DynamicsConfig.deterministic(),
            defender=defender,
        )
        env.reset(seed=1)
        env.step(env.encode_action(Action.SCAN_SERVICE, 0))
        _, _, _, _, info = env.step(env.encode_action(Action.ENUMERATE_SERVICE, 0))
        return float(info["detection_risk"])

    hardened = DefenderConfig(
        enabled=True,
        alert_threshold=0.0,
        response_cooldown=1,
        revocation_probability=0.0,
        hardening_step=1.0,
        response_latency=0,
        observation_noise=0.0,
    )

    assert enumeration_risk(hardened) > enumeration_risk(DefenderConfig())


def test_revocation_keeps_a_privilege_another_credential_still_grants() -> None:
    scenario = make_scenario()
    extra = scenario.credentials[0].model_copy(update={"id": "web-user-2"})
    scenario = scenario.model_copy(update={"credentials": (*scenario.credentials, extra)})
    env = AttackPathEnv(scenario, dynamics=DynamicsConfig.deterministic())
    env.reset(seed=1)
    for action_type, target in SUCCESS_PATH[:4]:
        env.step(env.encode_action(action_type, target))
    env._acquired_credentials[1] = 1

    env._revoke_credential(0)

    assert env._acquired_credentials.tolist()[:2] == [0, 1]
    assert env._acquired_privileges[env._privilege_index["user"]] == 1


def test_every_objective_must_be_collected_before_the_episode_ends() -> None:
    scenario = make_scenario()
    second = scenario.objectives[0].model_copy(
        update={"id": "collect-staging", "host_id": "web", "required_privilege_id": "user"}
    )
    scenario = scenario.model_copy(update={"objectives": (*scenario.objectives, second)})
    env = AttackPathEnv(scenario, step_budget=40, dynamics=DynamicsConfig.deterministic())
    env.reset(seed=1)
    for action_type, target in SUCCESS_PATH[:-1]:
        env.step(env.encode_action(action_type, target))

    _, _, terminated, _, info = env.step(env.encode_action(Action.COLLECT_SIMULATED_OBJECTIVE, 0))

    assert terminated is False
    assert info["objective_captured"] is False
    assert info["collected_objectives"] == 1

    _, _, terminated, _, info = env.step(env.encode_action(Action.COLLECT_SIMULATED_OBJECTIVE, 1))

    assert terminated is True
    assert info["objective_captured"] is True
    assert info["collected_objectives"] == 2


def test_a_collected_objective_cannot_be_collected_twice() -> None:
    env = deterministic_env(step_budget=40)
    env.reset(seed=1)
    for action_type, target in SUCCESS_PATH[:-1]:
        env.step(env.encode_action(action_type, target))
    env.step(env.encode_action(Action.COLLECT_SIMULATED_OBJECTIVE, 0))
    env.reset(seed=1)
    for action_type, target in SUCCESS_PATH[:-1]:
        env.step(env.encode_action(action_type, target))
    mask = env.action_mask().reshape(len(Action), env.target_count)

    assert mask[Action.COLLECT_SIMULATED_OBJECTIVE, 0] == 1

    env.step(env.encode_action(Action.COLLECT_SIMULATED_OBJECTIVE, 0))

    assert (
        env.action_mask().reshape(len(Action), env.target_count)[
            Action.COLLECT_SIMULATED_OBJECTIVE, 0
        ]
        == 0
    )


def test_detection_risk_is_normalized_by_network_size() -> None:
    """A larger network must not be unwinnable purely because it takes more steps."""

    def enumeration_risk(host_count: int, normalize: bool) -> float:
        scenario = generate_scenario("small" if host_count == 3 else "large", "easy", 0)
        env = AttackPathEnv(
            scenario,
            dynamics=DynamicsConfig(stochastic=False, normalize_risk_by_size=normalize),
        )
        env.reset(seed=0)
        env.step(env.encode_action(Action.SCAN_SERVICE, 0))
        _, _, _, _, info = env.step(env.encode_action(Action.ENUMERATE_SERVICE, 0))
        return float(info["detection_risk"])

    assert enumeration_risk(12, normalize=True) < enumeration_risk(12, normalize=False)
    assert enumeration_risk(3, normalize=True) == enumeration_risk(3, normalize=False)


def latent_defender(**overrides: object) -> DefenderConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "alert_threshold": 0.0,
        "response_cooldown": 1,
        "revocation_probability": 0.0,
        "hardening_step": 1.0,
        "response_latency": 3,
        "observation_noise": 0.0,
    }
    settings.update(overrides)
    return DefenderConfig(**settings)  # type: ignore[arg-type]


def test_a_defender_response_lands_only_after_its_latency() -> None:
    env = AttackPathEnv(
        make_scenario(),
        step_budget=40,
        dynamics=DynamicsConfig.deterministic(),
        defender=latent_defender(),
    )
    env.reset(seed=1)

    _, _, _, _, info = env.step(env.encode_action(Action.SCAN_SERVICE, 0))

    assert info["defender_pending"] is True
    assert info["defender_actions"] == 0

    for _ in range(3):
        _, _, _, _, info = env.step(env.encode_action(Action.ENUMERATE_SERVICE, 0))

    assert info["defender_pending"] is False
    assert info["defender_actions"] == 1
    assert info["defender_action"] == "harden_monitoring"


def test_defender_observation_noise_produces_false_positives() -> None:
    env = AttackPathEnv(
        make_scenario(),
        step_budget=40,
        dynamics=DynamicsConfig.deterministic(),
        defender=latent_defender(alert_threshold=0.9, observation_noise=5.0, response_latency=0),
    )
    env.reset(seed=1)
    for _ in range(6):
        _, _, _, _, info = env.step(env.encode_action(Action.SCAN_SERVICE, 0))

    assert info["detection_risk"] < 0.9
    assert info["defender_false_positives"] >= 1


def test_noisy_discovery_hides_the_topology_from_the_action_mask() -> None:
    base = make_scenario()
    scenario = base.model_copy(update={"hosts": (*base.hosts, Host(id="isolated"))})
    exact = AttackPathEnv(scenario, dynamics=DynamicsConfig.deterministic())
    noisy = AttackPathEnv(scenario, dynamics=DynamicsConfig(stochastic=False, noisy_discovery=True))
    exact.reset(seed=1)
    noisy.reset(seed=1)

    exact_mask = exact.action_mask().reshape(len(Action), exact.target_count)
    noisy_mask = noisy.action_mask().reshape(len(Action), noisy.target_count)

    assert exact_mask[Action.DISCOVER_HOST].tolist()[:3] == [0, 1, 0]
    assert noisy_mask[Action.DISCOVER_HOST].tolist()[:3] == [0, 1, 1]

    _, reward, _, _, info = noisy.step(noisy.encode_action(Action.DISCOVER_HOST, 2))

    assert info["outcome"] == "failed"
    assert reward < 0
    assert (
        noisy.action_mask().reshape(len(Action), noisy.target_count)[Action.DISCOVER_HOST, 2] == 0
    )


def test_probing_reopens_once_every_candidate_has_been_missed() -> None:
    env = AttackPathEnv(
        make_scenario(),
        step_budget=40,
        dynamics=DynamicsConfig(
            stochastic=True,
            noisy_discovery=True,
            discovery_probability=0.01,
            base_success_probability=1.0,
        ),
    )
    env.reset(seed=1)

    _, _, _, _, info = env.step(env.encode_action(Action.DISCOVER_HOST, 1))

    assert info["outcome"] == "failed"
    assert env.action_mask().reshape(len(Action), env.target_count)[Action.DISCOVER_HOST, 1] == 1, (
        "the only candidate must be offered again instead of deadlocking"
    )


def test_discovery_probability_is_validated() -> None:
    with pytest.raises(ValueError, match="discovery_probability"):
        DynamicsConfig(discovery_probability=0.0)


def test_boolean_action_masks_are_exposed_for_maskable_learners() -> None:
    env = deterministic_env()
    _, info = env.reset(seed=1)

    masks = env.action_masks()

    assert masks.dtype == np.bool_
    assert masks.tolist() == np.asarray(info["action_mask"]).astype(bool).tolist()


def test_enabling_the_defender_does_not_shift_the_attacker_stream() -> None:
    """Passive and adaptive must be paired: same seed, same attacker draws.

    The defender drawing from the shared stream made an inert defender change the
    attacker's outcomes, quietly confounding every control/treatment comparison.
    """

    def outcomes(defender: DefenderConfig) -> tuple[str, ...]:
        env = AttackPathEnv(
            generate_scenario("medium", "hard", 7), step_budget=60, defender=defender
        )
        env.reset(seed=3)
        trace = []
        terminated = truncated = False
        while not terminated and not truncated:
            mask = env.action_mask()
            action = np.int64(int(np.flatnonzero(mask)[0]))
            _, _, terminated, truncated, info = env.step(action)
            trace.append(str(info["outcome"]))
        return tuple(trace)

    # Responds on every step, but its responses change nothing: any divergence can
    # only come from the defender consuming the attacker's random stream.
    inert = DefenderConfig(
        enabled=True,
        alert_threshold=0.0,
        response_cooldown=1,
        response_latency=0,
        hardening_step=0.0,
        revocation_probability=0.0,
        observation_noise=0.5,
    )

    assert outcomes(DefenderConfig()) == outcomes(inert)


def test_the_agent_remembers_which_hosts_it_probed_and_missed() -> None:
    """Probe state must be in the observation, not only in the action mask.

    A maskable learner uses the mask to filter its action distribution, not as a network
    input, so without this channel a policy cannot tell an exhausted sweep from an
    untouched one.
    """

    base = make_scenario()
    scenario = base.model_copy(update={"hosts": (*base.hosts, Host(id="isolated"))})
    env = AttackPathEnv(scenario, dynamics=DynamicsConfig(stochastic=False, noisy_discovery=True))
    observation, _ = env.reset(seed=1)

    assert observation["probed_hosts"].tolist() == [0, 0, 0]

    observation, _, _, _, _ = env.step(env.encode_action(Action.DISCOVER_HOST, 2))

    assert observation["probed_hosts"].tolist() == [0, 0, 1]

    observation, _, _, _, _ = env.step(env.encode_action(Action.DISCOVER_HOST, 1))

    assert observation["probed_hosts"].tolist() == [0, 0, 0], (
        "a successful discovery re-opens the sweep, and the memory must follow"
    )


def test_probe_memory_is_present_under_exact_discovery_too() -> None:
    """The channel is unconditional so one policy fits both discovery conditions."""

    env = deterministic_env()
    observation, _ = env.reset(seed=1)

    assert observation["probed_hosts"].tolist() == [0, 0]
    assert env.observation_space.contains(observation)


def test_a_response_budget_caps_what_the_defender_can_spend() -> None:
    def responses(budget: int | None) -> tuple[int, int]:
        env = AttackPathEnv(
            generate_scenario("medium", "hard", 3),
            step_budget=60,
            defender=DefenderConfig(
                enabled=True,
                alert_threshold=0.0,
                response_cooldown=1,
                response_latency=0,
                observation_noise=0.0,
                response_budget=budget,
            ),
        )
        env.reset(seed=3)
        info: dict[str, object] = {}
        terminated = truncated = False
        while not terminated and not truncated:
            mask = env.action_mask()
            action = np.int64(int(np.flatnonzero(mask)[0]))
            _, _, terminated, truncated, info = env.step(action)
        return int(cast(int, info["defender_actions"])), int(
            cast(int, info["defender_over_budget"])
        )

    unlimited, unlimited_dropped = responses(None)
    capped, dropped = responses(3)

    assert unlimited > 3
    assert unlimited_dropped == 0
    assert capped == 3
    assert dropped > 0, "decisions past the budget must be recorded, not silently lost"


def test_budget_remaining_is_reported_and_validated() -> None:
    env = AttackPathEnv(make_scenario(), defender=DefenderConfig(response_budget=4))
    _, info = env.reset(seed=1)

    assert info["defender_budget_remaining"] == 4

    unlimited = AttackPathEnv(make_scenario())
    _, info = unlimited.reset(seed=1)

    assert info["defender_budget_remaining"] is None

    with pytest.raises(ValueError, match="response_budget"):
        DefenderConfig(response_budget=0)


def test_budget_pressure_rises_as_the_defender_spends() -> None:
    """The learned defender reads this band, so it must track real spending."""

    env = AttackPathEnv(
        generate_scenario("medium", "hard", 3),
        step_budget=60,
        dynamics=DynamicsConfig.deterministic(),
        defender=DefenderConfig(
            enabled=True,
            alert_threshold=0.0,
            response_cooldown=1,
            response_latency=0,
            observation_noise=0.0,
            response_budget=2,
        ),
    )
    env.reset(seed=3)

    assert env._budget_pressure() == 0

    pressures = []
    terminated = truncated = False
    while not terminated and not truncated:
        mask = env.action_mask()
        _, _, terminated, truncated, _ = env.step(np.int64(int(np.flatnonzero(mask)[0])))
        pressures.append(env._budget_pressure())

    assert max(pressures) == 2, "a spent budget must read as maximum pressure"
    assert env._budget_remaining() == 0


def test_the_contextual_defender_reads_budget_pressure() -> None:
    env = AttackPathEnv(
        make_scenario(),
        dynamics=DynamicsConfig.deterministic(),
        defender_policy=ContextualDefender(
            config=DefenderConfig(enabled=True, alert_threshold=0.0, response_budget=2),
            exploration=0.0,
        ),
    )
    env.reset(seed=1)
    env.step(env.encode_action(Action.SCAN_SERVICE, 0))

    assert env.defender_policy is not None
    assert env.defender.response_budget == 2
