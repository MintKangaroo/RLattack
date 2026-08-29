"""Cross-module guarantees: reproducibility, solvability, and the CLI round trip."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import pytest

from rlattack import cli
from rlattack.agents import ShortestPathOracle
from rlattack.env import AttackPathEnv, DynamicsConfig, ObservationConfig
from rlattack.experiment import ExperimentConfig, build_dashboard_data, run_episode
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario

SIZES: tuple[ScenarioSize, ...] = ("small", "medium", "large")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")


def trajectory_digest(size: ScenarioSize, difficulty: Difficulty, seed: int) -> str:
    """Hash one full episode so any dynamics change shows up as a regression."""

    episode = run_episode(
        generate_scenario(size, difficulty, seed),
        agent_name="shortest-path",
        seed=seed,
        step_budget=128,
    )
    payload = [
        [step.step, step.action, step.target_id, step.outcome, round(step.reward, 6)]
        for step in episode.trace
    ]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("size", "difficulty"), list(product(SIZES, DIFFICULTIES)))
def test_seeded_episodes_are_bit_for_bit_reproducible(
    size: ScenarioSize, difficulty: Difficulty
) -> None:
    assert trajectory_digest(size, difficulty, 5) == trajectory_digest(size, difficulty, 5)


@pytest.mark.parametrize(("size", "difficulty"), list(product(SIZES, DIFFICULTIES)))
def test_every_generated_scenario_is_solvable(size: ScenarioSize, difficulty: Difficulty) -> None:
    """The oracle must reach the objective on every generated graph.

    A generator that can emit unsolvable graphs would silently cap every reported
    success rate, so this is checked across the whole size/difficulty grid.
    """

    for seed in range(6):
        scenario = generate_scenario(size, difficulty, seed)
        env = AttackPathEnv(scenario, step_budget=256, dynamics=DynamicsConfig.deterministic())
        agent = ShortestPathOracle(scenario)
        observation, info = env.reset(seed=seed)
        terminated = truncated = False
        while not terminated and not truncated:
            observation, _, terminated, truncated, info = env.step(agent.predict(observation, info))

        assert info["objective_captured"] is True, f"{size}/{difficulty}/{seed} is unsolvable"


def test_scenario_references_stay_valid_across_the_generator_grid() -> None:
    for size, difficulty, seed in product(SIZES, DIFFICULTIES, range(4)):
        scenario = generate_scenario(size, difficulty, seed)
        host_ids = {host.id for host in scenario.hosts}
        service_ids = {service.id for service in scenario.services}

        assert scenario.entry_host_ids[0] in host_ids
        assert {service.host_id for service in scenario.services} <= host_ids
        assert {record.service_id for record in scenario.vulnerabilities} <= service_ids
        assert all(objective.host_id in host_ids for objective in scenario.objectives)


def test_cli_demo_round_trip_embeds_the_experiment_in_the_report(tmp_path: Path) -> None:
    report = tmp_path / "report.html"
    result = tmp_path / "result.json"

    assert (
        cli.main(
            [
                "demo",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--seed",
                "3",
                "--episodes",
                "2",
                "--deterministic",
                "--report",
                str(report),
                "--json",
                str(result),
            ]
        )
        == 0
    )
    html = report.read_text(encoding="utf-8")
    data = json.loads(result.read_text(encoding="utf-8"))

    assert "__RLATTACK_DATA__" in html
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert data["scenario"]["id"] == "generated-small-easy-3"
    assert data["benchmark_protocol"]["seeds"] == [3, 4]


def test_dashboard_view_model_matches_the_configured_experiment() -> None:
    config = ExperimentConfig(size="small", difficulty="easy", seed=2, benchmark_episodes=2)
    data = build_dashboard_data(config)

    assert data["config"]["seed"] == 2
    assert len(data["benchmarks"]) == 5
    assert data["episode"]["steps"] == len(data["episode"]["trace"])


def test_curriculum_observations_are_identical_across_scenario_classes() -> None:
    """One policy must fit every scenario class for transfer to be measurable."""

    spaces = set()
    action_counts = set()
    for size, difficulty in product(SIZES, DIFFICULTIES):
        env = AttackPathEnv(
            generate_scenario(size, difficulty, 1),
            observation_config=ObservationConfig.for_curriculum(),
        )
        observation, info = env.reset(seed=1)
        spaces.add(tuple(sorted((key, value.shape) for key, value in observation.items())))
        action_counts.add(info["target_count"])

    assert len(spaces) == 1
    assert len(action_counts) == 1


def test_curriculum_mode_does_not_change_experiment_outcomes() -> None:
    """Padding must be inert: only the interface widths change, never the simulation."""

    scenario = generate_scenario("small", "easy", 3)
    scenario_sized = run_episode(scenario, agent_name="shortest-path", seed=3)
    padded = run_episode(
        scenario,
        agent_name="shortest-path",
        seed=3,
        observation_config=ObservationConfig.for_curriculum(),
    )

    assert padded.steps == scenario_sized.steps
    assert padded.cumulative_reward == scenario_sized.cumulative_reward
    assert padded.trace == scenario_sized.trace
