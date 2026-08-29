import pytest

from rlattack.env import Action, AttackPathEnv
from rlattack.explain import EpisodeTrace, explain_action
from rlattack.generator import generate_scenario


def test_action_explanation_contains_observable_reasons() -> None:
    scenario = generate_scenario("small", "easy", seed=1)
    env = AttackPathEnv(scenario)
    observation, info = env.reset(seed=1)

    action = env.encode_action(Action.SCAN_SERVICE, 0)
    explanation = explain_action(
        observation, int(action), 0.5, info, target_id="host-00-ssh", q_value=1.2
    )

    assert explanation.action == "scan_service"
    assert explanation.target_id == "host-00-ssh"
    assert explanation.outcome == "unknown"
    assert explanation.valid is True
    assert explanation.observation_summary["discovered_hosts"] == 1
    assert explanation.q_value == 1.2


def test_episode_trace_and_graph_overlay_are_serializable() -> None:
    scenario = generate_scenario("small", "easy", seed=1)
    env = AttackPathEnv(scenario)
    observation, info = env.reset()
    trace = EpisodeTrace()
    trace.append(
        explain_action(
            observation,
            int(env.encode_action(Action.SCAN_SERVICE, 0)),
            0.5,
            info,
            affected_nodes=("host-00",),
            outcome="success",
        )
    )
    overlay = trace.graph_overlay(scenario)

    assert trace.actions == ("scan_service",)
    assert trace.cumulative_reward == 0.5
    assert any(node["kind"] == "host" for node in overlay)
    assert next(node for node in overlay if node["id"] == "host-00")["visited"] is True


def test_invalid_action_and_invalid_probability_are_rejected() -> None:
    scenario = generate_scenario("small", "easy", seed=1)
    env = AttackPathEnv(scenario)
    observation, info = env.reset()

    with pytest.raises(ValueError, match="outside"):
        explain_action(observation, len(Action) * env.target_count, 0.0, info)
    with pytest.raises(ValueError, match="target_count"):
        explain_action(observation, 0, 0.0, {"action_mask": info["action_mask"]})
    with pytest.raises(ValueError):
        explain_action(
            observation,
            int(env.encode_action(Action.STOP, 0)),
            0.0,
            info,
            action_probability=2.0,
        )


def test_explanation_marks_actions_invalid_when_the_mask_shape_is_unknown() -> None:
    scenario = generate_scenario("small", "easy", seed=1)
    env = AttackPathEnv(scenario)
    observation, info = env.reset()

    explanation = explain_action(
        observation,
        0,
        0.0,
        {"action_mask": info["action_mask"][:3], "target_count": env.target_count},
    )

    assert explanation.valid is False
