import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from rlattack import cli
from rlattack.env import Action, AttackPathEnv


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
