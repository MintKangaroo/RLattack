import csv
import json
from pathlib import Path

import pytest

from rlattack.evaluation import BenchmarkMetrics, EpisodeOutcome
from rlattack.export import episode_rows, summary_rows, write_results


def make_metrics(agent: str = "greedy") -> BenchmarkMetrics:
    outcome = EpisodeOutcome(
        seed=1,
        success=True,
        detected=False,
        steps=12,
        reward=3.5,
        detection_risk=0.2,
        path_cost=2.0,
    )
    return BenchmarkMetrics(
        agent_name=agent,
        episodes=1,
        successes=1,
        success_rate=1.0,
        detection_rate=0.0,
        mean_steps=12.0,
        mean_reward=3.5,
        mean_detection_risk=0.2,
        mean_path_cost=2.0,
        std_steps=0.0,
        std_reward=0.0,
        reward_ci_low=3.5,
        reward_ci_high=3.5,
        outcomes=(outcome,),
    )


def test_rows_expose_episodes_and_summaries() -> None:
    metrics = {"greedy": make_metrics()}

    assert episode_rows(metrics)[0]["agent"] == "greedy"
    assert summary_rows(metrics)[0]["success_rate"] == 1.0
    assert "outcomes" not in summary_rows(metrics)[0]


def test_write_jsonl_and_csv(tmp_path: Path) -> None:
    metrics = {"greedy": make_metrics(), "random": make_metrics("random")}

    jsonl = write_results(metrics, tmp_path / "nested" / "out.jsonl", "jsonl")
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]

    assert [row["agent"] for row in rows] == ["greedy", "random"]

    csv_path = write_results(metrics, tmp_path / "out.csv", "csv")
    with csv_path.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    assert records[0]["success"] == "True"
    assert records[1]["agent"] == "random"


def test_invalid_format_and_empty_results_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="export_format"):
        write_results({"greedy": make_metrics()}, tmp_path / "out.txt", "txt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no benchmark episodes"):
        write_results({}, tmp_path / "out.jsonl")
