"""Command-line interface for reproducible RLAttack experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from rlattack import __version__
from rlattack.agents import Agent
from rlattack.dashboard import run_dashboard
from rlattack.env import AttackPathEnv, ObservationConfig
from rlattack.experiment import (
    AgentName,
    DefenderMode,
    ExperimentConfig,
    ObservationMode,
    build_dashboard_data,
    run_benchmarks,
)
from rlattack.export import write_results
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario
from rlattack.policies import Algorithm, load_policy
from rlattack.report import write_dashboard_report
from rlattack.reward import RewardStrategy
from rlattack.training import (
    DQNTrainingConfig,
    PPOTrainingConfig,
    train_dqn,
    train_ppo,
    training_dependencies_available,
)


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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="disable transition uncertainty so every valid action succeeds",
    )
    parser.add_argument(
        "--observation",
        choices=("scenario", "curriculum"),
        default="scenario",
        help="scenario-sized observations, or fixed capacities that transfer across sizes",
    )
    parser.add_argument(
        "--defender",
        choices=("passive", "adaptive"),
        default="passive",
        help="passive is the control condition; adaptive responds to the attacker",
    )


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

    benchmark = commands.add_parser(
        "benchmark", help="run a multi-seed generalization benchmark and export the episodes"
    )
    _add_experiment_arguments(benchmark)
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark.jsonl"))
    benchmark.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    benchmark.add_argument(
        "--policy",
        type=Path,
        help="optional local Stable-Baselines3 checkpoint to benchmark alongside the baselines",
    )
    benchmark.add_argument("--policy-algorithm", choices=("dqn", "ppo"), default="dqn")

    train = commands.add_parser(
        "train", help="train an optional Stable-Baselines3 policy on generated scenarios"
    )
    train.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    train.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    train.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="hard")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--timesteps", type=int, default=10_000)
    train.add_argument("--step-budget", type=int, default=64)
    train.add_argument(
        "--observation",
        choices=("scenario", "curriculum"),
        default="curriculum",
        help="fixed capacities let one policy transfer across scenario sizes",
    )
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/policies"))
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
        stochastic=not args.deterministic,
        observation=cast(ObservationMode, args.observation),
        defender=cast(DefenderMode, args.defender),
    )


def _run_benchmark(args: argparse.Namespace) -> int:
    """Run the multi-seed benchmark and write per-episode records."""

    config = _config_from_args(args)
    extra: dict[str, Callable[[int], Agent]] = {}
    if args.policy is not None:
        policy = load_policy(args.policy, cast(Algorithm, args.policy_algorithm))
        extra[args.policy_algorithm] = lambda seed: policy
    metrics = run_benchmarks(config, extra)
    output = write_results(metrics, args.output, args.format)
    print("RLAttack generalization benchmark")
    print(f"  scenarios : {config.size}/{config.difficulty} x {config.benchmark_episodes} seeds")
    print(f"  dynamics  : {'stochastic' if config.stochastic else 'deterministic'}")
    print(f"  defender  : {config.defender}")
    for name, metric in metrics.items():
        print(
            f"  {name:<14} success={metric.success_rate:5.1%} "
            f"detected={metric.detection_rate:5.1%} "
            f"steps={metric.mean_steps:6.2f}±{metric.std_steps:5.2f} "
            f"reward={metric.mean_reward:7.2f} "
            f"[{metric.reward_ci_low:7.2f}, {metric.reward_ci_high:7.2f}]"
        )
    print(f"  export    : {output}")
    return 0


def _run_training(args: argparse.Namespace) -> int:
    """Train one optional Stable-Baselines3 policy on generated scenarios."""

    if not training_dependencies_available():
        print("Training requires the optional dependencies: pip install -e '.[training]'")
        return 1
    scenario = generate_scenario(
        cast(ScenarioSize, args.size),
        cast(Difficulty, args.difficulty),
        args.seed,
    )

    observation_config = (
        ObservationConfig.for_curriculum()
        if args.observation == "curriculum"
        else ObservationConfig()
    )

    def env_factory() -> AttackPathEnv:
        return AttackPathEnv(
            scenario,
            step_budget=args.step_budget,
            observation_config=observation_config,
        )

    if args.algorithm == "dqn":
        train_dqn(
            env_factory,
            DQNTrainingConfig(
                total_timesteps=args.timesteps,
                seed=args.seed,
                output_dir=args.output_dir,
            ),
        )
    else:
        train_ppo(
            env_factory,
            PPOTrainingConfig(
                total_timesteps=args.timesteps,
                seed=args.seed,
                output_dir=args.output_dir,
            ),
        )
    print(f"Trained {args.algorithm} policy saved under {args.output_dir.resolve()}")
    return 0


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

    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "train":
        return _run_training(args)

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
    if episode["success"]:
        outcome = "success"
    elif episode["detected"]:
        outcome = "detected"
    else:
        outcome = "incomplete"
    print("RLAttack reproducible experiment")
    print(f"  scenario : {data['scenario']['id']}")
    print(f"  policy   : {episode['agent_label']}")
    print(f"  dynamics : {'stochastic' if config.stochastic else 'deterministic'}")
    print(f"  outcome  : {outcome}")
    print(f"  steps    : {episode['steps']} / {config.step_budget}")
    print(f"  reward   : {episode['cumulative_reward']:.2f}")
    print(f"  report   : {report_path}")
    return 0
