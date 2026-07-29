"""Command-line interface for reproducible RLAttack experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from rlattack import __version__
from rlattack.dashboard import run_dashboard
from rlattack.experiment import AgentName, ExperimentConfig, build_dashboard_data
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario
from rlattack.report import write_dashboard_report
from rlattack.reward import RewardStrategy


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agent",
        choices=("random", "greedy", "rule-based", "shortest-path"),
        default="greedy",
    )
    parser.add_argument(
        "--reward",
        choices=("sparse", "shaped", "risk-aware", "cost-aware"),
        default="risk-aware",
    )
    parser.add_argument("--step-budget", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="rlattack",
        description="Deterministic, simulation-only attack-path research.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    demo = commands.add_parser("demo", help="run an experiment and create a visual report")
    _add_experiment_arguments(demo)
    demo.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/rlattack-report.html"),
        help="self-contained HTML report path",
    )
    demo.add_argument("--json", type=Path, dest="json_path", help="optional raw result path")

    scenario = commands.add_parser("scenario", help="export one generated scenario as JSON")
    scenario.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    scenario.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="hard")
    scenario.add_argument("--seed", type=int, default=42)
    scenario.add_argument("--output", type=Path, default=Path("artifacts/scenario.json"))

    dashboard = commands.add_parser("dashboard", help="start the interactive local dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8000)
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        size=cast(ScenarioSize, args.size),
        difficulty=cast(Difficulty, args.difficulty),
        seed=args.seed,
        agent=cast(AgentName, args.agent),
        reward_strategy=cast(RewardStrategy, args.reward),
        step_budget=args.step_budget,
        benchmark_episodes=args.episodes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return its process status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "dashboard":
        print(f"RLAttack dashboard: http://{args.host}:{args.port}")
        run_dashboard(host=args.host, port=args.port)
        return 0
    if args.command == "scenario":
        generated = generate_scenario(
            cast(ScenarioSize, args.size),
            cast(Difficulty, args.difficulty),
            args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated.model_dump_json(indent=2), encoding="utf-8")
        print(f"Scenario exported: {args.output.resolve()}")
        return 0

    config = _config_from_args(args)
    data = build_dashboard_data(config)
    report_path = write_dashboard_report(data, args.report)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    episode = data["episode"]
    print("RLAttack deterministic experiment")
    print(f"  scenario : {data['scenario']['id']}")
    print(f"  policy   : {episode['agent_label']}")
    print(f"  outcome  : {'success' if episode['success'] else 'incomplete'}")
    print(f"  steps    : {episode['steps']} / {config.step_budget}")
    print(f"  reward   : {episode['cumulative_reward']:.2f}")
    print(f"  report   : {report_path}")
    return 0
