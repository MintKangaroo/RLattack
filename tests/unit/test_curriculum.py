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
    scale_curriculum,
    stage_env_factory,
)
from rlattack.env import DynamicsConfig, ObservationConfig
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
