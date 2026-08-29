"""Batch export of benchmark results for offline statistical analysis."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from rlattack.evaluation import BenchmarkMetrics

ExportFormat = Literal["jsonl", "csv"]

EPISODE_FIELDS: tuple[str, ...] = (
    "agent",
    "seed",
    "success",
    "detected",
    "steps",
    "reward",
    "detection_risk",
    "path_cost",
)


def episode_rows(metrics: Mapping[str, BenchmarkMetrics]) -> list[dict[str, Any]]:
    """Flatten every benchmarked episode into one analysis-ready row per episode."""

    return [
        {"agent": name, **asdict(outcome)}
        for name, metric in metrics.items()
        for outcome in metric.outcomes
    ]


def summary_rows(metrics: Mapping[str, BenchmarkMetrics]) -> list[dict[str, Any]]:
    """Return one aggregate row per agent, without the per-episode detail."""

    return [
        {key: value for key, value in asdict(metric).items() if key != "outcomes"}
        for metric in metrics.values()
    ]


def write_results(
    metrics: Mapping[str, BenchmarkMetrics],
    path: Path,
    export_format: ExportFormat = "jsonl",
) -> Path:
    """Write per-episode benchmark records to ``path`` and return the resolved path."""

    if export_format not in ("jsonl", "csv"):
        raise ValueError("export_format must be 'jsonl' or 'csv'")
    rows = episode_rows(metrics)
    if not rows:
        raise ValueError("no benchmark episodes to export")
    path.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "jsonl":
        path.write_text(
            "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
            encoding="utf-8",
        )
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EPISODE_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
    return path.resolve()
