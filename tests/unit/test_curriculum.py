from typing import cast

import numpy as np
import pytest
from gymnasium import spaces

from rlattack.agents import Agent, GreedyAgent, ShortestPathOracle
from rlattack.curriculum import (
    ALL_STAGES,
    DEFAULT_CURRICULUM,
    HELD_OUT_STAGES,
    CurriculumStage,
    StageEnv,
    evaluate_transfer,
    family_curriculum,
    scale_curriculum,
    stage_env_factory,
)
from rlattack.defender import ContextualDefender
from rlattack.env import DynamicsConfig, ObservationConfig
from rlattack.families import build_scenario
from rlattack.generator import generate_scenario


def greedy_factory(stage: CurriculumStage, seed: int) -> Agent:
    del stage, seed
    return GreedyAgent()


def test_stage_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="size"):
        CurriculumStage("tiny", "easy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="difficulty"):
        CurriculumStage("small", "extreme")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timesteps"):
        CurriculumStage("small", "easy", timesteps=0)


def test_stage_labels_and_budgets_scale_with_scenario_size() -> None:
    assert CurriculumStage("small", "hard").label == "small/hard"
    assert CurriculumStage("small", "hard").step_budget(64) == 64
    assert CurriculumStage("large", "hard").step_budget(64) > 64
    assert CurriculumStage("large", "hard").step_budget(0) == 1


def test_the_default_curriculum_holds_out_the_largest_class() -> None:
    trained = {stage.label for stage in DEFAULT_CURRICULUM}
    held_out = {stage.label for stage in HELD_OUT_STAGES}

    assert trained.isdisjoint(held_out)
    assert len(ALL_STAGES) == 9


def test_stage_environments_share_one_interface() -> None:
    first = stage_env_factory(ALL_STAGES[0])(1)
    last = stage_env_factory(ALL_STAGES[-1])(1)

    assert first.observation_space == last.observation_space
    assert first.action_space == last.action_space


def test_stage_env_resamples_the_scenario_between_episodes() -> None:
    stage = CurriculumStage("medium", "hard")
    env = StageEnv(stage, (1, 2, 3, 4), stage_env_factory(stage))
    scenarios = set()
    for seed in range(12):
        env.reset(seed=seed)
        scenarios.add(env.current.scenario.id)

    assert len(scenarios) > 1
    assert env.observation_space.contains(env.reset(seed=0)[0])

    observation, _, _, _, info = env.step(np.int64(env.current.encode_action(8, 0)))

    assert info["action_name"] == "stop"
    channels = cast(spaces.Dict, env.current.observation_space)

    assert set(observation) == set(channels.spaces)

    with pytest.raises(ValueError, match="at least one training seed"):
        StageEnv(stage, (), stage_env_factory(stage))


def test_transfer_evaluation_covers_every_requested_stage() -> None:
    seen: list[str] = []

    def oracle_factory(stage: CurriculumStage, seed: int) -> Agent:
        seen.append(stage.label)
        return ShortestPathOracle(generate_scenario(stage.size, stage.difficulty, seed))

    stages = (CurriculumStage("small", "easy"), CurriculumStage("small", "medium"))
    results = evaluate_transfer(
        oracle_factory,
        (1, 2),
        stages,
        dynamics=DynamicsConfig.deterministic(),
    )

    assert set(results) == {"small/easy", "small/medium"}
    assert all(metric.episodes == 2 for metric in results.values())
    assert set(seen) == {"small/easy", "small/medium"}, (
        "a graph-aware baseline must be built from the stage it acts in"
    )

    with pytest.raises(ValueError, match="at least one curriculum stage"):
        evaluate_transfer(greedy_factory, (1,), ())


def test_transfer_accepts_an_explicit_observation_interface() -> None:
    factory = stage_env_factory(
        CurriculumStage("small", "easy"),
        observation_config=ObservationConfig(),
    )

    channels = cast(spaces.Dict, factory(1).observation_space)

    assert cast(spaces.MultiBinary, channels.spaces["discovered_hosts"]).n == 3


def test_stage_env_delegates_the_action_mask_to_the_live_scenario() -> None:
    stage = CurriculumStage("small", "easy")
    env = StageEnv(stage, (1, 2), stage_env_factory(stage))
    env.reset(seed=0)

    masks = env.action_masks()

    assert masks.dtype == np.bool_
    assert masks.tolist() == env.current.action_masks().tolist()


def test_curriculum_budgets_scale_while_keeping_each_stage_share() -> None:
    scaled = scale_curriculum(DEFAULT_CURRICULUM, 400_000)

    assert sum(stage.timesteps for stage in scaled) == 400_000
    assert [stage.label for stage in scaled] == [stage.label for stage in DEFAULT_CURRICULUM]
    original = DEFAULT_CURRICULUM[0].timesteps / sum(s.timesteps for s in DEFAULT_CURRICULUM)
    assert scaled[0].timesteps / 400_000 == pytest.approx(original)

    tiny = scale_curriculum(DEFAULT_CURRICULUM, len(DEFAULT_CURRICULUM))

    assert all(stage.timesteps == 1 for stage in tiny)


def test_curriculum_scaling_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one curriculum stage"):
        scale_curriculum((), 1000)
    with pytest.raises(ValueError, match="at least one step per stage"):
        scale_curriculum(DEFAULT_CURRICULUM, 2)


def test_a_stage_can_sample_earlier_stages_alongside_its_own() -> None:
    """Training a stage in isolation makes the policy forget the earlier ones."""

    stage = CurriculumStage("medium", "hard")
    earlier = CurriculumStage("small", "easy")
    env = StageEnv(
        stage,
        (1, 2),
        stage_env_factory(stage),
        [stage_env_factory(earlier)],
    )
    sizes = set()
    for seed in range(20):
        env.reset(seed=seed)
        sizes.add(len(env.current.scenario.hosts))

    assert len(sizes) > 1, "a mixed stage must draw from more than its own class"
    assert env.pool_size == 4
    assert env.observation_space == stage_env_factory(earlier)(1).observation_space


def test_the_defender_learns_across_training_episodes() -> None:
    """Stable-Baselines3 owns the episode loop, so the defender is driven from reset."""

    stage = CurriculumStage("small", "easy")
    policy = ContextualDefender()
    policy.reset(seed=0)
    env = StageEnv(stage, (1, 2), stage_env_factory(stage), (), policy)

    assert env.current.defender_policy is policy
    assert env.current.defender is policy.config

    for episode in range(6):
        env.reset(seed=episode)
        terminated = truncated = False
        while not terminated and not truncated:
            action = np.int64(int(np.flatnonzero(env.action_masks())[0]))
            _, _, terminated, truncated, _ = env.step(action)
    env.reset(seed=99)

    assert policy.table, "the defender must have scored the finished episodes"


def test_a_stage_without_a_defender_policy_is_untouched() -> None:
    stage = CurriculumStage("small", "easy")
    env = StageEnv(stage, (1,), stage_env_factory(stage))

    assert env.defender_policy is None
    assert env.current.defender_policy is None

    env.reset(seed=0)
    action = np.int64(int(np.flatnonzero(env.action_masks())[0]))
    _, _, _, _, info = env.step(action)

    assert info["defender_action"] == "none"


def test_a_family_stage_describes_its_topology_and_scale() -> None:
    stage = CurriculumStage("medium", "hard", family="mesh", hosts=10)

    assert stage.label == "mesh/10"
    # A family stage has no size to scale on, so it scales on its host count.
    assert stage.step_budget(60) == 100
    assert CurriculumStage("medium", "hard").label == "medium/hard"


def test_a_family_stage_is_validated() -> None:
    with pytest.raises(ValueError, match="unknown topology family"):
        CurriculumStage("medium", "hard", family="hypercube")
    with pytest.raises(ValueError, match="hosts"):
        CurriculumStage("medium", "hard", family="mesh", hosts=1)


def test_a_family_curriculum_grows_the_host_count() -> None:
    stages = family_curriculum("mesh")

    assert [stage.label for stage in stages] == ["mesh/5", "mesh/6", "mesh/8", "mesh/10"]

    with pytest.raises(ValueError, match="at least one host count"):
        family_curriculum("mesh", ())


def test_rescaling_keeps_a_family_curriculum_on_its_family() -> None:
    """Rebuilding a stage from a subset of its fields silently reverts the topology."""

    rescaled = scale_curriculum(family_curriculum("mesh"), 1_000)

    assert [stage.label for stage in rescaled] == ["mesh/5", "mesh/6", "mesh/8", "mesh/10"]
    assert sum(stage.timesteps for stage in rescaled) == 1_000


def test_a_family_stage_builds_its_own_topology() -> None:
    stage = CurriculumStage("medium", "hard", family="mesh", hosts=8)
    env = stage_env_factory(stage)(3)

    assert len(env.scenario.hosts) == 8
    assert env.scenario.id == build_scenario("mesh", 8, 3).id
