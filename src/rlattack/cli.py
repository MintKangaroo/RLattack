"""Command-line interface for reproducible RLAttack experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from rlattack import __version__
from rlattack.agents import Agent
from rlattack.conditions import CONTROL_LABEL, run_condition_sweep
from rlattack.curriculum import (
    DEFAULT_CURRICULUM,
    CurriculumStage,
    StageEnv,
    evaluate_transfer,
    scale_curriculum,
    stage_env_factory,
)
from rlattack.dashboard import run_dashboard
from rlattack.defender import BanditDefender, ContextualDefender, DefenderConfig
from rlattack.env import AttackPathEnv, DynamicsConfig, ObservationConfig
from rlattack.equilibrium import solve_grid
from rlattack.evaluation import BenchmarkMetrics
from rlattack.experiment import (
    REWARD_STRATEGIES,
    AgentName,
    DefenderMode,
    DiscoveryMode,
    ExperimentConfig,
    ObservationMode,
    benchmark_seeds,
    build_dashboard_data,
    create_agent,
    run_benchmarks,
    run_reward_ablation,
)
from rlattack.export import write_results
from rlattack.game import BanditAttacker, play
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario
from rlattack.importers import import_scenario_file
from rlattack.policies import Algorithm, load_policy
from rlattack.report import write_dashboard_report, write_transfer_report
from rlattack.reward import RewardStrategy
from rlattack.stats import compare_benchmarks
from rlattack.sweep import trials_by_label
from rlattack.training import (
    DQNTrainingConfig,
    PPOTrainingConfig,
    train_curriculum,
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
        choices=("sparse", "shaped", "risk-aware", "cost-aware", "pivot-focused"),
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
    parser.add_argument(
        "--discovery",
        choices=("exact", "noisy"),
        default="exact",
        help="exact adjacency, or a noisy scan that does not reveal the topology",
    )


TRAINING_SEEDS: tuple[int, ...] = tuple(range(32))


def _add_significance_arguments(parser: argparse.ArgumentParser, *, default_reference: str) -> None:
    parser.add_argument(
        "--compare-to",
        default=default_reference,
        help="reference for the paired significance test",
    )
    parser.add_argument(
        "--metric",
        choices=("reward", "steps", "success"),
        default="reward",
        help="per-episode metric the significance test compares",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--resamples",
        type=int,
        default=2_000,
        help="permutation and bootstrap iterations",
    )


def _print_comparisons(metrics: dict[str, BenchmarkMetrics], args: argparse.Namespace) -> None:
    """Print paired significance tests against the chosen reference."""

    if args.compare_to not in metrics:
        print(f"  (skipping significance tests: unknown reference '{args.compare_to}')")
        return
    print(f"  paired vs {args.compare_to} on {args.metric} (alpha={args.alpha})")
    for comparison in compare_benchmarks(
        metrics,
        args.compare_to,
        metric=args.metric,
        alpha=args.alpha,
        iterations=args.resamples,
    ):
        marker = "*" if comparison.significant else " "
        print(
            f"  {marker} {comparison.candidate:<14} "
            f"diff={comparison.mean_difference:+8.3f} "
            f"CI=[{comparison.ci_low:+8.3f}, {comparison.ci_high:+8.3f}] "
            f"p={comparison.p_value:.4f}"
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

    importer = commands.add_parser(
        "import",
        help="convert a published attack graph into a sanitized scenario",
    )
    importer.add_argument("--input", type=Path, required=True, help="GraphML, GML, or JSON")
    importer.add_argument("--output", type=Path, default=Path("artifacts/imported.json"))
    importer.add_argument("--scenario-id", help="scenario ID (default: derived from the filename)")
    importer.add_argument(
        "--topology-only",
        action="store_true",
        help="import only hosts and reachability, without a playable exploitation layer",
    )

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
    benchmark.add_argument(
        "--policy-algorithm", choices=("dqn", "ppo", "maskable-ppo"), default="maskable-ppo"
    )
    _add_significance_arguments(benchmark, default_reference="greedy")

    ablation = commands.add_parser(
        "ablation", help="compare reward strategies for one agent on identical seeds"
    )
    _add_experiment_arguments(ablation)
    ablation.add_argument(
        "--strategies",
        nargs="+",
        choices=("sparse", "shaped", "risk-aware", "cost-aware", "pivot-focused"),
        default=list(REWARD_STRATEGIES),
    )
    ablation.add_argument("--output", type=Path, default=Path("artifacts/ablation.jsonl"))
    ablation.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    _add_significance_arguments(ablation, default_reference="shaped")

    transfer = commands.add_parser(
        "transfer",
        help="evaluate one policy on every scenario class to measure generalization",
    )
    _add_experiment_arguments(transfer)
    transfer.add_argument(
        "--policy",
        type=Path,
        help="optional local Stable-Baselines3 checkpoint; defaults to the --agent baseline",
    )
    transfer.add_argument(
        "--policy-algorithm", choices=("dqn", "ppo", "maskable-ppo"), default="maskable-ppo"
    )
    transfer.add_argument("--output", type=Path, default=Path("artifacts/transfer.jsonl"))
    transfer.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    transfer.add_argument(
        "--report",
        type=Path,
        help="optional self-contained HTML transfer table",
    )
    _add_significance_arguments(transfer, default_reference="small/easy")

    conditions = commands.add_parser(
        "conditions",
        help="evaluate one policy across the defender x discovery condition grid",
    )
    _add_experiment_arguments(conditions)
    conditions.add_argument(
        "--policy",
        type=Path,
        help="optional local Stable-Baselines3 checkpoint; defaults to the --agent baseline",
    )
    conditions.add_argument(
        "--policy-algorithm", choices=("dqn", "ppo", "maskable-ppo"), default="maskable-ppo"
    )
    conditions.add_argument("--output", type=Path, default=Path("artifacts/conditions.jsonl"))
    conditions.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    _add_significance_arguments(conditions, default_reference=CONTROL_LABEL)

    game = commands.add_parser(
        "game",
        help="play a fixed attacker against a defender that adapts between episodes",
    )
    _add_experiment_arguments(game)
    game.add_argument(
        "--policy",
        type=Path,
        help="optional local Stable-Baselines3 checkpoint; defaults to the --agent baseline",
    )
    game.add_argument(
        "--policy-algorithm", choices=("dqn", "ppo", "maskable-ppo"), default="maskable-ppo"
    )
    game.add_argument("--rounds", type=int, default=200, help="episodes the defender learns over")
    game.add_argument(
        "--attacker",
        choices=("fixed", "bandit"),
        default="fixed",
        help="hold the attacker fixed, or let it learn which policy beats this defender",
    )
    game.add_argument(
        "--defender-policy",
        choices=("bandit", "contextual"),
        default="contextual",
        help="one configuration per episode, or a policy conditioned on the episode so far",
    )
    game.add_argument("--exploration", type=float, default=0.15)
    game.add_argument("--output", type=Path, default=Path("artifacts/game.jsonl"))

    equilibrium = commands.add_parser(
        "equilibrium",
        help="solve the attacker x defender policy grid as a matrix game",
    )
    _add_experiment_arguments(equilibrium)
    equilibrium.add_argument("--iterations", type=int, default=20_000)
    equilibrium.add_argument("--output", type=Path, default=Path("artifacts/equilibrium.json"))

    sweep = commands.add_parser(
        "sweep",
        help="train several hyperparameter trials and benchmark each resulting policy",
    )
    _add_experiment_arguments(sweep)
    sweep.add_argument("--trials", nargs="+", help="trial labels to run (default: all)")
    sweep.add_argument("--curriculum-timesteps", type=int, default=40_000)
    sweep.add_argument("--output-dir", type=Path, default=Path("artifacts/sweep"))
    sweep.add_argument("--output", type=Path, default=Path("artifacts/sweep.jsonl"))
    sweep.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    _add_significance_arguments(sweep, default_reference="baseline")

    train = commands.add_parser(
        "train", help="train an optional Stable-Baselines3 policy on generated scenarios"
    )
    train.add_argument(
        "--algorithm",
        choices=("dqn", "ppo", "maskable-ppo"),
        default="maskable-ppo",
        help="maskable-ppo respects the environment's action mask during training",
    )
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
    train.add_argument(
        "--discovery",
        choices=("exact", "noisy"),
        default="exact",
        help="train under exact adjacency, or under the noisy scan model",
    )
    train.add_argument(
        "--defender",
        choices=("passive", "adaptive"),
        default="passive",
        help="train against a passive or an adaptive defender",
    )
    train.add_argument(
        "--reward",
        choices=("sparse", "shaped", "risk-aware", "cost-aware", "pivot-focused"),
        default="risk-aware",
        help="reward strategy to train against",
    )
    train.add_argument(
        "--adversarial",
        action="store_true",
        help="let a contextual defender learn alongside the attacker during training",
    )
    train.add_argument(
        "--forget-previous-stages",
        action="store_true",
        help="train each stage in isolation instead of sampling earlier stages too",
    )
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/policies"))
    train.add_argument(
        "--curriculum",
        action="store_true",
        help="train one policy across the staged scenario curriculum",
    )
    train.add_argument(
        "--curriculum-timesteps",
        type=int,
        help="total curriculum budget, split across stages in their default proportions",
    )
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
        discovery=cast(DiscoveryMode, args.discovery),
    )


def _run_benchmark(args: argparse.Namespace) -> int:
    """Run the multi-seed benchmark and write per-episode records."""

    config = _config_from_args(args)
    extra: dict[str, Callable[[int], Agent]] = {}
    if args.policy is not None:
        policy = load_policy(args.policy, cast(Algorithm, args.policy_algorithm))
        extra[args.policy_algorithm] = _constant_agent(policy)
    metrics = run_benchmarks(config, extra)
    output = write_results(metrics, args.output, args.format)
    print("RLAttack generalization benchmark")
    print(f"  scenarios : {config.size}/{config.difficulty} x {config.benchmark_episodes} seeds")
    print(f"  dynamics  : {'stochastic' if config.stochastic else 'deterministic'}")
    print(f"  defender  : {config.defender}")
    print(f"  discovery : {config.discovery}")
    for name, metric in metrics.items():
        print(
            f"  {name:<14} success={metric.success_rate:5.1%} "
            f"detected={metric.detection_rate:5.1%} "
            f"steps={metric.mean_steps:6.2f}±{metric.std_steps:5.2f} "
            f"reward={metric.mean_reward:7.2f} "
            f"[{metric.reward_ci_low:7.2f}, {metric.reward_ci_high:7.2f}]"
        )
    _print_comparisons(metrics, args)
    print(f"  export    : {output}")
    return 0


def _run_ablation(args: argparse.Namespace) -> int:
    """Compare reward strategies for one agent on identical seeds."""

    config = _config_from_args(args)
    strategies = [cast(RewardStrategy, strategy) for strategy in args.strategies]
    metrics = run_reward_ablation(config, strategies)
    output = write_results(metrics, args.output, args.format)
    print("RLAttack reward ablation")
    print(f"  agent     : {config.agent}")
    print(f"  scenarios : {config.size}/{config.difficulty} x {config.benchmark_episodes} seeds")
    print(f"  defender  : {config.defender}")
    for name, metric in metrics.items():
        print(
            f"  {name:<14} success={metric.success_rate:5.1%} "
            f"detected={metric.detection_rate:5.1%} "
            f"steps={metric.mean_steps:6.2f}±{metric.std_steps:5.2f} "
            f"reward={metric.mean_reward:7.2f}"
        )
    print(
        "  note      : baseline heuristics ignore the reward signal, so a behavioural"
        " difference here requires a trained policy"
    )
    _print_comparisons(metrics, args)
    print(f"  export    : {output}")
    return 0


def _transfer_view_model(
    metrics: dict[str, BenchmarkMetrics],
    config: ExperimentConfig,
    policy_label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build the view model behind the self-contained transfer report."""

    comparisons = (
        [
            asdict(item) | {"significant": item.significant}
            for item in compare_benchmarks(
                metrics,
                args.compare_to,
                metric=args.metric,
                alpha=args.alpha,
                iterations=args.resamples,
            )
        ]
        if args.compare_to in metrics
        else []
    )
    return {
        "policy": policy_label,
        "reference": args.compare_to,
        "seeds": list(benchmark_seeds(config)),
        "stages": [
            {key: value for key, value in asdict(metric).items() if key != "outcomes"}
            for metric in metrics.values()
        ],
        "comparisons": comparisons,
        "conditions": [
            ["Dynamics", "stochastic" if config.stochastic else "deterministic"],
            ["Defender", config.defender],
            ["Discovery", config.discovery],
            ["Reward", config.reward_strategy],
            ["Metric", args.metric],
        ],
        "note": (
            "Every class is evaluated on the same seed list, so the episodes are paired and "
            "the difference column is a paired sign-flip permutation test against the "
            f"'{args.compare_to}' class. Step budgets scale with scenario size."
        ),
    }


def _stage_env_builder(
    stage: CurriculumStage,
    step_budget: int,
    observation_config: ObservationConfig,
    dynamics: DynamicsConfig | None = None,
    defender: DefenderConfig | None = None,
    reward_strategy: RewardStrategy = "risk-aware",
    previous: Sequence[CurriculumStage] = (),
    defender_policy: ContextualDefender | None = None,
) -> Callable[[], StageEnv]:
    """Return a zero-argument environment builder for one curriculum stage.

    Training conditions must be passed through here: a policy trained under the default
    dynamics has no reason to handle any other condition, which is exactly how the v0.6
    policies ended up scoring 0% under noisy discovery.
    """

    factory = stage_env_factory(
        stage,
        step_budget=step_budget,
        reward_strategy=reward_strategy,
        observation_config=observation_config,
        dynamics=dynamics,
        defender=defender,
    )

    earlier = [
        stage_env_factory(
            previous_stage,
            step_budget=step_budget,
            reward_strategy=reward_strategy,
            observation_config=observation_config,
            dynamics=dynamics,
            defender=defender,
        )
        for previous_stage in previous
    ]

    def build() -> StageEnv:
        return StageEnv(stage, TRAINING_SEEDS, factory, earlier, defender_policy)

    return build


def _run_transfer(args: argparse.Namespace) -> int:
    """Evaluate one policy on every scenario class with a shared seed list."""

    config = _config_from_args(args)
    if args.policy is not None:
        policy = load_policy(args.policy, cast(Algorithm, args.policy_algorithm))
        label = args.policy_algorithm

        def agent_factory(stage: CurriculumStage, seed: int) -> Agent:
            del stage, seed
            return policy
    else:
        label = config.agent

        def agent_factory(stage: CurriculumStage, seed: int) -> Agent:
            # The baseline must see the scenario it will act in, not the configured one.
            return create_agent(
                config.agent,
                generate_scenario(stage.size, stage.difficulty, seed),
                seed=seed,
            )

    metrics = evaluate_transfer(
        agent_factory,
        benchmark_seeds(config),
        step_budget=config.step_budget,
        reward_strategy=config.reward_strategy,
        dynamics=config.dynamics(),
        defender=config.defender_config(),
    )
    output = write_results(metrics, args.output, args.format)
    print("RLAttack transfer evaluation")
    print(f"  policy    : {label}")
    print("  observati.: curriculum (transfer requires one shared interface)")
    print(f"  seeds     : {config.benchmark_episodes} shared across every scenario class")
    print(f"  defender  : {config.defender}")
    for name, metric in metrics.items():
        print(
            f"  {name:<14} success={metric.success_rate:5.1%} "
            f"detected={metric.detection_rate:5.1%} "
            f"steps={metric.mean_steps:6.2f}±{metric.std_steps:5.2f} "
            f"reward={metric.mean_reward:7.2f}"
        )
    _print_comparisons(metrics, args)
    print(f"  export    : {output}")
    if args.report is not None:
        report = write_transfer_report(
            _transfer_view_model(metrics, config, label, args), args.report
        )
        print(f"  report    : {report}")
    return 0


def _run_conditions(args: argparse.Namespace) -> int:
    """Evaluate one policy across the defender x discovery grid."""

    config = _config_from_args(args)
    agent_factory, label = _agent_factory_from_args(args, config)
    metrics = run_condition_sweep(config, agent_factory)
    output = write_results(metrics, args.output, args.format)
    print("RLAttack condition sweep")
    print(f"  policy    : {label}")
    print(f"  scenarios : {config.size}/{config.difficulty} x {config.benchmark_episodes} seeds")
    for name, metric in metrics.items():
        print(
            f"  {name:<17} success={metric.success_rate:5.1%} "
            f"detected={metric.detection_rate:5.1%} "
            f"steps={metric.mean_steps:6.2f}±{metric.std_steps:5.2f} "
            f"reward={metric.mean_reward:7.2f}"
        )
    _print_comparisons(metrics, args)
    print(f"  export    : {output}")
    return 0


def _constant_agent(agent: Agent) -> Callable[[int], Agent]:
    """Wrap one already-built agent as the seed-indexed factory evaluators expect."""

    def build(seed: int) -> Agent:
        del seed
        return agent

    return build


def _agent_factory_from_args(
    args: argparse.Namespace, config: ExperimentConfig
) -> tuple[Callable[[int], Agent], str]:
    """Build the attacker factory shared by the evaluation commands."""

    if args.policy is not None:
        policy = load_policy(args.policy, cast(Algorithm, args.policy_algorithm))
        return _constant_agent(policy), str(args.policy_algorithm)

    def from_baseline(seed: int) -> Agent:
        return create_agent(
            config.agent,
            generate_scenario(config.size, config.difficulty, seed),
            seed=seed,
        )

    return from_baseline, config.agent


def _run_game(args: argparse.Namespace) -> int:
    """Play a fixed attacker against a defender that adapts between episodes."""

    config = _config_from_args(args)
    agent_factory, label = _agent_factory_from_args(args, config)
    opponent: BanditDefender | ContextualDefender = (
        ContextualDefender(exploration=args.exploration)
        if args.defender_policy == "contextual"
        else BanditDefender(exploration=args.exploration)
    )
    attacker = BanditAttacker(exploration=args.exploration) if args.attacker == "bandit" else None
    result = play(
        config,
        agent_factory,
        opponent,
        attacker=attacker,
        episodes=args.rounds,
        seed=config.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            f"{json.dumps({'episode': index, **asdict(outcome)}, ensure_ascii=False)}\n"
            for index, outcome in enumerate(result.outcomes)
        ),
        encoding="utf-8",
    )
    print("RLAttack attacker vs adaptive defender")
    print(f"  attacker  : {'bandit over baselines' if attacker is not None else label}")
    print(f"  scenarios : {config.size}/{config.difficulty} x {result.episodes} rounds")
    print(f"  attacker success : {result.attacker_success_rate:5.1%}")
    print(f"  detected         : {result.detection_rate:5.1%}")
    print(f"  defender reward  : {result.mean_defender_reward:.3f}")
    print(f"  defender policy  : {args.defender_policy}")
    if result.pulls:
        print(f"  settled on       : {result.preferred_arm}")
        for arm, pulls in result.pulls.items():
            print(f"  {arm:<16} pulls={pulls:4}  value={result.values[arm]:.3f}")
    for policy, pulls in result.attacker_pulls.items():
        print(
            f"  attacker {policy:<14} pulls={pulls:4}  value={result.attacker_values[policy]:.3f}"
        )
    print(f"  export    : {args.output.resolve()}")
    return 0


def _run_equilibrium(args: argparse.Namespace) -> int:
    """Solve the attacker x defender policy grid and report the mixtures."""

    config = _config_from_args(args)
    solved = solve_grid(config, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(solved), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("RLAttack attacker x defender equilibrium")
    print(f"  scenarios : {config.size}/{config.difficulty} x {config.benchmark_episodes} seeds")
    header = "  ".join(f"{label[:9]:>9}" for label in solved.defender_labels)
    corner = "attacker / defender"
    print(f"  {corner:<22} {header}")
    for label, row in zip(solved.attacker_labels, solved.payoffs, strict=True):
        print(f"  {label:<22} " + "  ".join(f"{value:9.3f}" for value in row))
    print(f"  attacker  : {_format_mixture(solved.attacker_support)}")
    print(f"  defender  : {_format_mixture(solved.defender_support)}")
    print(f"  value     : {solved.value:.4f} (attacker mean episode reward)")
    if len(solved.attacker_support) == 1 and len(solved.defender_support) == 1:
        print("  note      : a pure equilibrium - this policy grid has a dominant strategy")
    print(f"  export    : {args.output.resolve()}")
    return 0


def _format_mixture(mixture: dict[str, float]) -> str:
    return ", ".join(f"{label} {weight:.0%}" for label, weight in mixture.items())


def _run_sweep(args: argparse.Namespace) -> int:
    """Train each hyperparameter trial and benchmark the resulting policies."""

    if not training_dependencies_available():
        print("Sweeping requires the optional dependencies: pip install -e '.[training]'")
        return 1
    config = _config_from_args(args)
    trials = trials_by_label(tuple(args.trials) if args.trials else None)
    stages = scale_curriculum(DEFAULT_CURRICULUM, args.curriculum_timesteps)
    observation_config = ObservationConfig.for_curriculum()
    metrics: dict[str, BenchmarkMetrics] = {}
    for trial in trials:
        training_config = trial.config(seed=config.seed, output_dir=args.output_dir)
        print(f"training trial '{trial.label}' ({args.curriculum_timesteps} timesteps)")
        train_curriculum(
            [_stage_env_builder(stage, config.step_budget, observation_config) for stage in stages],
            [stage.timesteps for stage in stages],
            training_config,
            algorithm="maskable-ppo",
        )
        policy = load_policy(training_config.output_dir / "final", "maskable-ppo")
        factory = _constant_agent(policy)
        metrics[trial.label] = run_benchmarks(config, {trial.label: factory})[trial.label]
    output = write_results(metrics, args.output, args.format)
    print("RLAttack hyperparameter sweep")
    print(f"  scenarios : {config.size}/{config.difficulty} x {config.benchmark_episodes} seeds")
    for name, metric in metrics.items():
        print(
            f"  {name:<14} success={metric.success_rate:5.1%} "
            f"detected={metric.detection_rate:5.1%} "
            f"steps={metric.mean_steps:6.2f}±{metric.std_steps:5.2f} "
            f"reward={metric.mean_reward:7.2f}"
        )
    _print_comparisons(metrics, args)
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

    if args.curriculum:
        stages = (
            scale_curriculum(DEFAULT_CURRICULUM, args.curriculum_timesteps)
            if args.curriculum_timesteps
            else DEFAULT_CURRICULUM
        )
        dynamics = DynamicsConfig(noisy_discovery=args.discovery == "noisy")
        opponent = ContextualDefender() if args.adversarial else None
        defender = DefenderConfig.adaptive() if args.defender == "adaptive" else DefenderConfig()
        train_curriculum(
            [
                _stage_env_builder(
                    stage,
                    args.step_budget,
                    observation_config,
                    dynamics,
                    defender,
                    cast(RewardStrategy, args.reward),
                    () if args.forget_previous_stages else stages[:index],
                    opponent,
                )
                for index, stage in enumerate(stages)
            ],
            [stage.timesteps for stage in stages],
            PPOTrainingConfig(seed=args.seed, output_dir=args.output_dir),
            algorithm=args.algorithm,
        )
        labels = ", ".join(stage.label for stage in stages)
        print(
            f"Trained {args.algorithm} curriculum ({labels}) under "
            f"{'adversarial' if args.adversarial else args.defender}/{args.discovery} "
            f"into {args.output_dir.resolve()}"
        )
        return 0
    if args.algorithm == "maskable-ppo":
        print("Masked training requires --curriculum; rerun with --curriculum")
        return 1
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
    if args.command == "import":
        imported = import_scenario_file(
            args.input,
            scenario_id=args.scenario_id,
            synthesize_layers=not args.topology_only,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(imported.model_dump_json(indent=2), encoding="utf-8")
        print("RLAttack scenario import")
        print(f"  source    : {args.input}")
        print(f"  hosts     : {len(imported.hosts)}  edges: {len(imported.network_edges)}")
        print(f"  entry     : {imported.entry_host_ids[0]}")
        print(f"  scenario  : {args.output.resolve()}")
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
    if args.command == "ablation":
        return _run_ablation(args)
    if args.command == "transfer":
        return _run_transfer(args)
    if args.command == "conditions":
        return _run_conditions(args)
    if args.command == "game":
        return _run_game(args)
    if args.command == "sweep":
        return _run_sweep(args)
    if args.command == "equilibrium":
        return _run_equilibrium(args)
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
