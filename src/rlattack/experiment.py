"""Shared experiment runner used by the CLI, reports, and local dashboard."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np

from rlattack.agents import (
    Agent,
    GreedyAgent,
    RandomAgent,
    RuleBasedAgent,
    ShortestPathOracle,
    reset_agent,
)
from rlattack.defender import DefenderConfig
from rlattack.env import AttackPathEnv, DynamicsConfig, ObservationConfig
from rlattack.evaluation import BenchmarkMetrics, evaluate_agent
from rlattack.explain import EpisodeTrace, explain_action
from rlattack.generator import Difficulty, ScenarioSize, generate_scenario
from rlattack.reward import RewardStrategy, build_reward_config
from rlattack.scenario import Scenario

AgentName = Literal[
    "random",
    "greedy",
    "rule-based",
    "shortest-path",
    "shortest-path-broad",
    "shortest-path-evasive",
]
ObservationMode = Literal["scenario", "curriculum"]
DefenderMode = Literal["passive", "adaptive", "targeted"]
DiscoveryMode = Literal["exact", "noisy"]

REWARD_STRATEGIES: tuple[RewardStrategy, ...] = (
    "sparse",
    "shaped",
    "risk-aware",
    "cost-aware",
    "pivot-focused",
)

MAX_STEP_BUDGET = 512
MAX_BENCHMARK_EPISODES = 256

AGENT_LABELS: dict[AgentName, str] = {
    "random": "Random",
    "greedy": "Greedy",
    "rule-based": "Rule-based",
    "shortest-path": "Graph oracle",
    "shortest-path-broad": "Graph oracle (redundant)",
    "shortest-path-evasive": "Graph oracle (evasive)",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable inputs for a reproducible simulator experiment."""

    size: ScenarioSize = "medium"
    difficulty: Difficulty = "hard"
    seed: int = 42
    agent: AgentName = "greedy"
    reward_strategy: RewardStrategy = "risk-aware"
    step_budget: int = 64
    benchmark_episodes: int = 8
    stochastic: bool = True
    observation: ObservationMode = "scenario"
    defender: DefenderMode = "passive"
    discovery: DiscoveryMode = "exact"
    detection_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.size not in {"small", "medium", "large"}:
            raise ValueError("size must be small, medium, or large")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        if self.agent not in AGENT_LABELS:
            raise ValueError("unsupported baseline agent")
        if self.reward_strategy not in set(REWARD_STRATEGIES):
            raise ValueError("unsupported reward strategy")
        if self.step_budget < 1:
            raise ValueError("step_budget must be positive")
        if self.benchmark_episodes < 1:
            raise ValueError("benchmark_episodes must be positive")
        if self.step_budget > MAX_STEP_BUDGET:
            raise ValueError(f"step_budget must be at most {MAX_STEP_BUDGET}")
        if self.benchmark_episodes > MAX_BENCHMARK_EPISODES:
            raise ValueError(f"benchmark_episodes must be at most {MAX_BENCHMARK_EPISODES}")
        if self.observation not in ("scenario", "curriculum"):
            raise ValueError("observation must be scenario or curriculum")
        if self.defender not in ("passive", "adaptive", "targeted"):
            raise ValueError("defender must be passive, adaptive, or targeted")
        if self.discovery not in ("exact", "noisy"):
            raise ValueError("discovery must be exact or noisy")
        if not 0.0 < self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")

    def dynamics(self) -> DynamicsConfig:
        """Return the transition-uncertainty configuration for this experiment.

        ``detection_threshold`` is a condition rather than a constant because at the
        default of 0.9 detection almost never fires against a competent attacker - the
        graph oracle is caught in roughly one episode in twenty - so nothing that
        re-prices risk can change an outcome, and a grid built on risk trade-offs has
        no trade-off to find.
        """

        noisy = self.discovery == "noisy"
        return DynamicsConfig(
            stochastic=self.stochastic,
            noisy_discovery=noisy,
            detection_threshold=self.detection_threshold,
        )

    def defender_config(self) -> DefenderConfig:
        """Return the defender condition: passive control, or one of two treatments.

        ``adaptive`` spreads monitoring uniformly; ``targeted`` concentrates it on a
        few hosts, which is what lets an attacker route around the defender instead of
        only doing less.
        """

        if self.defender == "targeted":
            return DefenderConfig.targeted()
        if self.defender == "adaptive":
            return DefenderConfig.adaptive()
        return DefenderConfig()

    def observation_config(self) -> ObservationConfig:
        """Return the observation interface: scenario-sized, or fixed for transfer.

        A targeted defender is reported to the agent, because monitoring it cannot see
        is not something it can route around.
        """

        monitoring = self.defender == "targeted"
        if self.observation == "curriculum":
            return ObservationConfig.for_curriculum(expose_monitoring=monitoring)
        return ObservationConfig(expose_monitoring=monitoring)


@dataclass(frozen=True)
class EpisodeStep:
    """One decision and its observable outcome."""

    step: int
    action: str
    target_id: str | None
    outcome: str
    valid: bool
    reward: float
    cumulative_reward: float
    detection_risk: float
    affected_nodes: tuple[str, ...]
    state: dict[str, int | float]


@dataclass(frozen=True)
class EpisodeResult:
    """Complete deterministic episode result."""

    agent: AgentName
    success: bool
    detected: bool
    defender_actions: int
    defender_false_positives: int
    revoked_credentials: int
    terminated: bool
    truncated: bool
    steps: int
    cumulative_reward: float
    detection_risk: float
    path_cost: float
    visited_nodes: tuple[str, ...]
    trace: tuple[EpisodeStep, ...]


def create_agent(name: AgentName, scenario: Scenario, *, seed: int) -> Agent:
    """Create one baseline policy with a stable public name."""

    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent()
    if name == "rule-based":
        return RuleBasedAgent()
    if name == "shortest-path":
        return ShortestPathOracle(scenario)
    if name == "shortest-path-broad":
        return ShortestPathOracle(scenario, redundant=True)
    if name == "shortest-path-evasive":
        return ShortestPathOracle(scenario, evasive=True)
    raise ValueError(f"unsupported baseline agent: {name}")


def run_episode(
    scenario: Scenario,
    *,
    agent_name: AgentName = "greedy",
    seed: int = 0,
    step_budget: int = 64,
    reward_strategy: RewardStrategy = "shaped",
    dynamics: DynamicsConfig | None = None,
    observation_config: ObservationConfig | None = None,
    defender: DefenderConfig | None = None,
) -> EpisodeResult:
    """Run one baseline episode and retain an explainable trajectory."""

    reward_config = build_reward_config(reward_strategy)
    env = AttackPathEnv(
        scenario,
        step_budget=step_budget,
        reward_config=reward_config,
        dynamics=dynamics,
        observation_config=observation_config,
        defender=defender,
    )
    agent = create_agent(agent_name, scenario, seed=seed)
    reset_agent(agent, seed=seed)
    observation, info = env.reset(seed=seed)
    trace = EpisodeTrace()
    records: list[EpisodeStep] = []
    cumulative_reward = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.predict(observation, info)
        previous_observation = observation
        previous_info = info
        observation, reward, terminated, truncated, info = env.step(action)
        affected_nodes = cast(tuple[str, ...], info["affected_nodes"])
        explanation = explain_action(
            previous_observation,
            int(action),
            reward,
            previous_info,
            affected_nodes=affected_nodes,
            target_id=cast("str | None", info["target_id"]),
            outcome=cast(str, info["outcome"]),
        )
        trace.append(explanation)
        cumulative_reward += reward
        records.append(
            EpisodeStep(
                step=cast(int, info["steps"]),
                action=explanation.action,
                target_id=explanation.target_id,
                outcome=explanation.outcome,
                valid=cast(bool, info["valid_action"]),
                reward=reward,
                cumulative_reward=cumulative_reward,
                detection_risk=cast(float, info["detection_risk"]),
                affected_nodes=affected_nodes,
                state={
                    "discovered_hosts": int(np.sum(observation["discovered_hosts"])),
                    "known_services": int(np.sum(observation["known_services"])),
                    "validated_vulnerabilities": int(
                        np.sum(observation["validated_vulnerabilities"])
                    ),
                    "acquired_privileges": int(np.sum(observation["acquired_privileges"])),
                    "steps_remaining": float(cast(int, info["steps_remaining"])),
                },
            )
        )

    overlay = trace.graph_overlay(scenario)
    return EpisodeResult(
        agent=agent_name,
        success=bool(info["objective_captured"]),
        detected=bool(info["detected"]),
        defender_actions=cast(int, info["defender_actions"]),
        defender_false_positives=cast(int, info["defender_false_positives"]),
        revoked_credentials=cast(int, info["revoked_credentials"]),
        terminated=terminated,
        truncated=truncated,
        steps=cast(int, info["steps"]),
        cumulative_reward=cumulative_reward,
        detection_risk=cast(float, info["detection_risk"]),
        path_cost=cast(float, info["path_cost"]),
        visited_nodes=tuple(
            cast(str, node["id"]) for node in overlay if cast(bool, node["visited"])
        ),
        trace=tuple(records),
    )


def benchmark_seeds(config: ExperimentConfig) -> tuple[int, ...]:
    """Return the evaluation seeds implied by one experiment configuration."""

    return tuple(config.seed + offset for offset in range(config.benchmark_episodes))


def run_benchmarks(
    config: ExperimentConfig,
    extra_agents: Mapping[str, Callable[[int], Agent]] | None = None,
) -> dict[str, BenchmarkMetrics]:
    """Benchmark every baseline over independently generated scenarios.

    Each seed regenerates the scenario, so the result is a generalization benchmark
    rather than a repeated replay of one fixed graph.
    """

    reward_config = build_reward_config(config.reward_strategy)
    dynamics = config.dynamics()
    observation_config = config.observation_config()
    defender = config.defender_config()

    def scenario_for(seed: int) -> Scenario:
        return generate_scenario(config.size, config.difficulty, seed)

    def env_factory(seed: int) -> AttackPathEnv:
        return AttackPathEnv(
            scenario_for(seed),
            step_budget=config.step_budget,
            reward_config=reward_config,
            dynamics=dynamics,
            observation_config=observation_config,
            defender=defender,
        )

    def agent_factory(name: AgentName) -> Callable[[int], Agent]:
        def build(seed: int) -> Agent:
            return create_agent(name, scenario_for(seed), seed=seed)

        return build

    seeds = benchmark_seeds(config)
    factories: dict[str, Callable[[int], Agent]] = {
        agent_name: agent_factory(agent_name) for agent_name in AGENT_LABELS
    }
    factories.update(extra_agents or {})
    return {
        name: evaluate_agent(name, factory, env_factory, seeds)
        for name, factory in factories.items()
    }


def run_reward_ablation(
    config: ExperimentConfig,
    strategies: Sequence[RewardStrategy] = REWARD_STRATEGIES,
) -> dict[str, BenchmarkMetrics]:
    """Evaluate one agent under several reward strategies on identical seeds.

    Everything except the reward is held fixed - same scenarios, same seeds, same
    dynamics and defender - so the episodes stay paired and the strategies can be
    compared with a paired significance test.
    """

    if not strategies:
        raise ValueError("at least one reward strategy is required")
    seeds = benchmark_seeds(config)
    dynamics = config.dynamics()
    observation_config = config.observation_config()
    defender = config.defender_config()

    def scenario_for(seed: int) -> Scenario:
        return generate_scenario(config.size, config.difficulty, seed)

    def env_factory(strategy: RewardStrategy) -> Callable[[int], AttackPathEnv]:
        def build(seed: int) -> AttackPathEnv:
            return AttackPathEnv(
                scenario_for(seed),
                step_budget=config.step_budget,
                reward_config=build_reward_config(strategy),
                dynamics=dynamics,
                observation_config=observation_config,
                defender=defender,
            )

        return build

    def agent_factory(seed: int) -> Agent:
        return create_agent(config.agent, scenario_for(seed), seed=seed)

    return {
        strategy: evaluate_agent(strategy, agent_factory, env_factory(strategy), seeds)
        for strategy in strategies
    }


def monitored_hosts(scenario: Scenario, config: ExperimentConfig) -> tuple[str, ...]:
    """Return the hosts a targeted defender starts the episode watching.

    Empty for every other defender: uniform monitoring has no hosts to name, and
    reporting all of them would read as attention where there is none.
    """

    return AttackPathEnv(
        scenario,
        step_budget=config.step_budget,
        dynamics=config.dynamics(),
        observation_config=config.observation_config(),
        defender=config.defender_config(),
    ).monitored_hosts()


def build_dashboard_data(config: ExperimentConfig | None = None) -> dict[str, Any]:
    """Build the deterministic view model consumed by HTML and JSON clients."""

    selected = config or ExperimentConfig()
    scenario = generate_scenario(selected.size, selected.difficulty, selected.seed)
    episode = run_episode(
        scenario,
        agent_name=selected.agent,
        seed=selected.seed,
        step_budget=selected.step_budget,
        reward_strategy=selected.reward_strategy,
        dynamics=selected.dynamics(),
        observation_config=selected.observation_config(),
        defender=selected.defender_config(),
    )
    benchmarks = run_benchmarks(selected)
    metrics = [
        {
            **{key: value for key, value in asdict(metric).items() if key != "outcomes"},
            "label": AGENT_LABELS[cast(AgentName, name)],
        }
        for name, metric in benchmarks.items()
    ]

    route = ShortestPathOracle(scenario).route
    visited = set(episode.visited_nodes)
    entry_hosts = set(scenario.entry_host_ids)
    objective_hosts = {objective.host_id for objective in scenario.objectives}
    monitored = monitored_hosts(scenario, selected)
    host_nodes = [
        {
            "id": host.id,
            "label": host.id.replace("host-", "NODE "),
            "os": host.operating_system or "unknown",
            "zone": host.zone,
            "services": sum(service.host_id == host.id for service in scenario.services),
            "detection": round(
                sum(
                    control.detection_probability
                    for control in scenario.security_controls
                    if host.id in control.host_ids
                ),
                3,
            ),
            "visited": host.id in visited,
            "entry": host.id in entry_hosts,
            "objective": host.id in objective_hosts,
            "monitored": host.id in monitored,
        }
        for host in scenario.hosts
    ]
    route_edges = set(zip(route[:-1], route[1:], strict=True))
    edges = [
        {
            "source": edge.source_host_id,
            "target": edge.target_host_id,
            "cost": edge.cost,
            "route": (edge.source_host_id, edge.target_host_id) in route_edges,
        }
        for edge in scenario.network_edges
    ]
    reward_values = asdict(build_reward_config(selected.reward_strategy))
    return {
        "schema_version": "2.0",
        "config": asdict(selected),
        "scenario": {
            "id": scenario.id,
            "name": scenario.name,
            "hosts": len(scenario.hosts),
            "services": len(scenario.services),
            "vulnerabilities": len(scenario.vulnerabilities),
            "edges": len(scenario.network_edges),
            "nodes": host_nodes,
            "network_edges": edges,
            "oracle_route": route,
        },
        "episode": {
            **asdict(episode),
            "agent_label": AGENT_LABELS[episode.agent],
        },
        "benchmarks": metrics,
        "benchmark_protocol": {
            "mode": "per-seed-scenario",
            "seeds": list(benchmark_seeds(selected)),
            "note": (
                "Each benchmark episode regenerates the scenario from its own seed, so the "
                "metrics measure generalization across graphs of the configured class."
            ),
        },
        "reward": {
            "strategy": selected.reward_strategy,
            "values": reward_values,
        },
        "safety": {
            "mode": "simulation-only",
            "network_access": False,
            "subprocess_execution": False,
            "live_credentials": False,
        },
    }
