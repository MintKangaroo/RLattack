from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest

from rlattack import cli
from rlattack.agents import GreedyAgent
from rlattack.curriculum import DEFAULT_CURRICULUM, StageEnv
from rlattack.env import Action, AttackPathEnv
from rlattack.training import PPOTrainingConfig


def test_cli_prints_help_without_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "simulation-only" in capsys.readouterr().out


def test_cli_exports_scenario(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "scenario.json"

    assert (
        cli.main(
            [
                "scenario",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--seed",
                "5",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "generated-small-easy-5"
    assert "Scenario exported" in capsys.readouterr().out


def test_cli_demo_writes_report_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
                "4",
                "--episodes",
                "1",
                "--deterministic",
                "--report",
                str(report),
                "--json",
                str(result),
            ]
        )
        == 0
    )
    assert report.exists()
    assert json.loads(result.read_text(encoding="utf-8"))["episode"]["success"] is True
    assert "outcome  : success" in capsys.readouterr().out


def test_cli_starts_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        cli,
        "run_dashboard",
        lambda *, host, port: calls.append((host, port)),
    )

    assert cli.main(["dashboard", "--host", "localhost", "--port", "8124"]) == 0
    assert calls == [("localhost", 8124)]


def test_cli_benchmark_exports_episode_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "benchmark.jsonl"

    assert (
        cli.main(
            [
                "benchmark",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--seed",
                "11",
                "--episodes",
                "3",
                "--deterministic",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 12
    assert {row["agent"] for row in rows} == {"random", "greedy", "rule-based", "shortest-path"}
    assert "generalization benchmark" in capsys.readouterr().out


def test_cli_benchmark_includes_a_trained_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubAgent:
        def predict(self, observation: object, info: dict[str, object]) -> np.int64:
            return np.int64(int(Action.STOP) * int(cast(int, info["target_count"])))

    monkeypatch.setattr(cli, "load_policy", lambda path, algorithm: StubAgent())
    output = tmp_path / "policy.csv"

    assert (
        cli.main(
            [
                "benchmark",
                "--size",
                "small",
                "--episodes",
                "2",
                "--format",
                "csv",
                "--output",
                str(output),
                "--policy",
                str(tmp_path / "model.zip"),
                "--policy-algorithm",
                "ppo",
            ]
        )
        == 0
    )

    assert "ppo," in output.read_text(encoding="utf-8")


def test_cli_train_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: False)

    assert cli.main(["train"]) == 1
    assert "optional dependencies" in capsys.readouterr().out


def test_cli_train_invokes_the_requested_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: True)
    calls: list[str] = []

    def record(name: str) -> object:
        def run(env_factory: object, config: object) -> None:
            calls.append(name)
            factory = cast(Callable[[], object], env_factory)
            assert isinstance(factory(), AttackPathEnv)

        return run

    monkeypatch.setattr(cli, "train_dqn", record("dqn"))
    monkeypatch.setattr(cli, "train_ppo", record("ppo"))

    for algorithm in ("dqn", "ppo"):
        assert (
            cli.main(
                [
                    "train",
                    "--algorithm",
                    algorithm,
                    "--size",
                    "small",
                    "--timesteps",
                    "8",
                    "--output-dir",
                    str(tmp_path / algorithm),
                ]
            )
            == 0
        )

    assert calls == ["dqn", "ppo"]
    assert "Trained ppo policy" in capsys.readouterr().out


def test_cli_demo_reports_a_detected_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "demo",
                "--size",
                "medium",
                "--difficulty",
                "hard",
                "--seed",
                "42",
                "--agent",
                "rule-based",
                "--episodes",
                "1",
                "--report",
                str(tmp_path / "report.html"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "outcome  : detected" in output
    assert "dynamics : stochastic" in output


def test_cli_demo_reports_a_budget_limited_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "demo",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--step-budget",
                "3",
                "--episodes",
                "1",
                "--deterministic",
                "--report",
                str(tmp_path / "report.html"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "outcome  : incomplete" in output
    assert "dynamics : deterministic" in output


def test_cli_benchmark_prints_paired_significance_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "benchmark",
                "--size",
                "small",
                "--difficulty",
                "hard",
                "--episodes",
                "6",
                "--resamples",
                "200",
                "--compare-to",
                "greedy",
                "--metric",
                "steps",
                "--output",
                str(tmp_path / "b.jsonl"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "paired vs greedy on steps" in output
    assert "shortest-path" in output


def test_cli_skips_significance_tests_for_an_unknown_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "benchmark",
                "--size",
                "small",
                "--episodes",
                "2",
                "--compare-to",
                "nobody",
                "--output",
                str(tmp_path / "b.jsonl"),
            ]
        )
        == 0
    )

    assert "unknown reference 'nobody'" in capsys.readouterr().out


def test_cli_ablation_compares_reward_strategies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "ablation.csv"

    assert (
        cli.main(
            [
                "ablation",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--episodes",
                "4",
                "--strategies",
                "sparse",
                "shaped",
                "--resamples",
                "200",
                "--format",
                "csv",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out

    assert "RLAttack reward ablation" in printed
    assert "paired vs shaped on reward" in printed
    assert "baseline heuristics ignore the reward signal" in printed
    assert "sparse," in output.read_text(encoding="utf-8")


def test_cli_transfer_evaluates_every_scenario_class(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "transfer.jsonl"

    assert (
        cli.main(
            [
                "transfer",
                "--agent",
                "greedy",
                "--episodes",
                "2",
                "--deterministic",
                "--resamples",
                "100",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert "RLAttack transfer evaluation" in printed
    assert "large/hard" in printed
    assert len({row["agent"] for row in rows}) == 9


def test_cli_transfer_can_evaluate_a_trained_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class StubAgent:
        def predict(self, observation: object, info: dict[str, object]) -> np.int64:
            return np.int64(int(Action.STOP) * int(cast(int, info["target_count"])))

    monkeypatch.setattr(cli, "load_policy", lambda path, algorithm: StubAgent())

    assert (
        cli.main(
            [
                "transfer",
                "--episodes",
                "2",
                "--resamples",
                "100",
                "--policy",
                str(tmp_path / "model.zip"),
                "--output",
                str(tmp_path / "transfer.csv"),
                "--format",
                "csv",
            ]
        )
        == 0
    )

    assert "policy    : maskable-ppo" in capsys.readouterr().out


def test_cli_curriculum_training_carries_one_policy_across_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: True)
    captured: dict[str, object] = {}

    def record(
        builders: list[object], timesteps: list[int], config: object, algorithm: str
    ) -> None:
        captured["stages"] = len(builders)
        captured["timesteps"] = timesteps
        captured["algorithm"] = algorithm
        first = cast(Callable[[], object], builders[0])()
        assert isinstance(first, StageEnv)

    monkeypatch.setattr(cli, "train_curriculum", record)

    assert (
        cli.main(
            [
                "train",
                "--algorithm",
                "ppo",
                "--curriculum",
                "--output-dir",
                str(tmp_path / "curriculum"),
            ]
        )
        == 0
    )

    assert captured["stages"] == len(DEFAULT_CURRICULUM)
    assert captured["algorithm"] == "ppo"
    assert "curriculum" in capsys.readouterr().out


def test_cli_transfer_writes_a_self_contained_report(tmp_path: Path) -> None:
    report = tmp_path / "transfer.html"

    assert (
        cli.main(
            [
                "transfer",
                "--agent",
                "greedy",
                "--episodes",
                "2",
                "--deterministic",
                "--resamples",
                "100",
                "--output",
                str(tmp_path / "t.jsonl"),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    html = report.read_text(encoding="utf-8")

    assert "__RLATTACK_TRANSFER__" in html
    assert '"reference":"small/easy"' in html
    assert '"significant"' in html


def test_cli_transfer_report_omits_tests_for_an_unknown_reference(tmp_path: Path) -> None:
    report = tmp_path / "transfer.html"

    assert (
        cli.main(
            [
                "transfer",
                "--agent",
                "greedy",
                "--episodes",
                "2",
                "--deterministic",
                "--compare-to",
                "nobody",
                "--output",
                str(tmp_path / "t.jsonl"),
                "--report",
                str(report),
            ]
        )
        == 0
    )

    assert '"comparisons":[]' in report.read_text(encoding="utf-8")


def test_cli_masked_training_requires_the_curriculum(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: True)

    assert cli.main(["train", "--algorithm", "maskable-ppo", "--size", "small"]) == 1
    assert "--curriculum" in capsys.readouterr().out


def test_cli_conditions_sweeps_the_treatment_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "conditions.jsonl"

    assert (
        cli.main(
            [
                "conditions",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--agent",
                "greedy",
                "--episodes",
                "4",
                "--resamples",
                "100",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert "RLAttack condition sweep" in printed
    assert "adaptive/noisy" in printed
    assert "paired vs passive/exact" in printed
    assert {row["agent"] for row in rows} == {
        "passive/exact",
        "adaptive/exact",
        "passive/noisy",
        "adaptive/noisy",
    }


def test_cli_conditions_accepts_a_trained_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class StubAgent:
        def predict(self, observation: object, info: dict[str, object]) -> np.int64:
            return np.int64(int(Action.STOP) * int(cast(int, info["target_count"])))

    monkeypatch.setattr(cli, "load_policy", lambda path, algorithm: StubAgent())

    assert (
        cli.main(
            [
                "conditions",
                "--size",
                "small",
                "--episodes",
                "2",
                "--resamples",
                "100",
                "--policy",
                str(tmp_path / "model.zip"),
                "--output",
                str(tmp_path / "c.jsonl"),
            ]
        )
        == 0
    )

    assert "policy    : maskable-ppo" in capsys.readouterr().out


def test_cli_game_reports_the_defenders_learned_preference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "game.jsonl"

    assert (
        cli.main(
            [
                "game",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--agent",
                "greedy",
                "--rounds",
                "10",
                "--defender-policy",
                "bandit",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert "attacker vs adaptive defender" in printed
    assert "settled on" in printed
    assert len(rows) == 10
    assert rows[0]["episode"] == 0


def test_cli_sweep_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: False)

    assert cli.main(["sweep"]) == 1
    assert "optional dependencies" in capsys.readouterr().out


def test_cli_sweep_trains_and_benchmarks_each_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: True)
    trained: list[float] = []

    def fake_train(
        builders: list[object], timesteps: list[int], config: object, algorithm: str
    ) -> None:
        trained.append(cast(PPOTrainingConfig, config).learning_rate)
        assert algorithm == "maskable-ppo"

    monkeypatch.setattr(cli, "train_curriculum", fake_train)
    monkeypatch.setattr(cli, "load_policy", lambda path, algorithm: GreedyAgent())
    output = tmp_path / "sweep.jsonl"

    assert (
        cli.main(
            [
                "sweep",
                "--trials",
                "baseline",
                "fast-lr",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--episodes",
                "3",
                "--curriculum-timesteps",
                "8",
                "--resamples",
                "100",
                "--output-dir",
                str(tmp_path / "runs"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert trained == [3e-4, 1e-3]
    assert "hyperparameter sweep" in printed
    assert "paired vs baseline" in printed
    assert {row["agent"] for row in rows} == {"baseline", "fast-lr"}


def test_cli_game_supports_the_contextual_defender(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "game",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--agent",
                "greedy",
                "--rounds",
                "8",
                "--output",
                str(tmp_path / "game.jsonl"),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out

    assert "defender policy  : contextual" in printed
    assert "settled on" not in printed, "a contextual defender has no single arm"


def test_cli_imports_a_published_attack_graph(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    graph: nx.DiGraph[Any] = nx.DiGraph()
    graph.add_edges_from([("web", "app"), ("app", "db")])
    source = tmp_path / "topology.graphml"
    nx.write_graphml(graph, source)
    output = tmp_path / "scenario.json"

    assert cli.main(["import", "--input", str(source), "--output", str(output)]) == 0
    imported = json.loads(output.read_text(encoding="utf-8"))
    printed = capsys.readouterr().out

    assert imported["id"] == "imported-topology"
    assert len(imported["hosts"]) == 3
    assert "app" not in output.read_text(encoding="utf-8").replace("host-", "")
    assert "RLAttack scenario import" in printed


def test_cli_can_import_topology_only(tmp_path: Path) -> None:
    graph: nx.DiGraph[Any] = nx.DiGraph()
    graph.add_edge("a", "b")
    source = tmp_path / "t.graphml"
    nx.write_graphml(graph, source)
    output = tmp_path / "scenario.json"

    assert (
        cli.main(
            [
                "import",
                "--input",
                str(source),
                "--output",
                str(output),
                "--topology-only",
                "--scenario-id",
                "custom-id",
            ]
        )
        == 0
    )
    imported = json.loads(output.read_text(encoding="utf-8"))

    assert imported["id"] == "custom-id"
    assert imported["services"] == []


def test_cli_game_can_let_the_attacker_learn_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "game",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--attacker",
                "bandit",
                "--rounds",
                "8",
                "--output",
                str(tmp_path / "game.jsonl"),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out

    assert "attacker  : bandit over baselines" in printed
    assert "attacker shortest-path" in printed


def test_cli_curriculum_stages_mix_earlier_stages_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "training_dependencies_available", lambda: True)
    stage_sizes: list[int] = []

    def record(
        builders: list[object], timesteps: list[int], config: object, algorithm: str
    ) -> None:
        for builder in builders:
            env = cast(Callable[[], StageEnv], builder)()
            stage_sizes.append(env.pool_size)

    monkeypatch.setattr(cli, "train_curriculum", record)

    assert (
        cli.main(
            [
                "train",
                "--curriculum",
                "--curriculum-timesteps",
                "8",
                "--output-dir",
                str(tmp_path / "mixed"),
            ]
        )
        == 0
    )
    mixed = stage_sizes[:]
    stage_sizes.clear()

    assert (
        cli.main(
            [
                "train",
                "--curriculum",
                "--forget-previous-stages",
                "--curriculum-timesteps",
                "8",
                "--output-dir",
                str(tmp_path / "isolated"),
            ]
        )
        == 0
    )

    assert stage_sizes == [stage_sizes[0]] * len(stage_sizes), (
        "isolated stages each draw from one class"
    )
    assert mixed == [stage_sizes[0] * (index + 1) for index in range(len(mixed))], (
        "a mixed stage must also draw from every earlier stage"
    )


def test_cli_training_conditions_reach_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flags must reach the env, not just the log line.

    A refactor once left --discovery parsed and printed but never passed through, so a
    run advertised as noisy trained under exact adjacency and its results were wrong.
    """

    monkeypatch.setattr(cli, "training_dependencies_available", lambda: True)
    seen: dict[str, object] = {}

    def record(
        builders: list[object], timesteps: list[int], config: object, algorithm: str
    ) -> None:
        env = cast(Callable[[], StageEnv], builders[1])().current
        seen["noisy"] = env.dynamics.noisy_discovery
        seen["defender"] = env.defender.enabled
        seen["step_cost"] = env.reward_config.step_cost

    monkeypatch.setattr(cli, "train_curriculum", record)

    assert (
        cli.main(
            [
                "train",
                "--curriculum",
                "--discovery",
                "noisy",
                "--defender",
                "adaptive",
                "--reward",
                "cost-aware",
                "--curriculum-timesteps",
                "8",
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )

    assert seen == {"noisy": True, "defender": True, "step_cost": -0.2}


def test_cli_equilibrium_solves_the_policy_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "equilibrium.json"

    assert (
        cli.main(
            [
                "equilibrium",
                "--size",
                "small",
                "--difficulty",
                "easy",
                "--episodes",
                "2",
                "--deterministic",
                "--iterations",
                "500",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    solved = json.loads(output.read_text(encoding="utf-8"))

    assert "attacker x defender equilibrium" in printed
    assert "value     :" in printed
    assert len(solved["payoffs"]) == 4
    assert len(solved["payoffs"][0]) == 5
    assert sum(solved["attacker_mixture"]) == pytest.approx(1.0)
