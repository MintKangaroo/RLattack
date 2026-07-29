import json
from pathlib import Path

import pytest

from rlattack import cli


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
