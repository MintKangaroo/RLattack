"""Structural scenario families the generator cannot produce.

The generator emits one shape: a chain of hosts with difficulty-dependent shortcuts.
A policy evaluated only on that shape may have learned the shape rather than the task.
These families are built from NetworkX topology generators and imported through the same
sanitized path as an external attack graph, so they are a genuine held-out test set
rather than another draw from the training distribution.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import networkx as nx

from rlattack.agents import Agent
from rlattack.defender import DefenderConfig
from rlattack.env import AttackPathEnv, DynamicsConfig, ObservationConfig
from rlattack.evaluation import BenchmarkMetrics, evaluate_agent
from rlattack.importers import scenario_from_graph
from rlattack.reward import RewardStrategy, build_reward_config
from rlattack.scenario import Scenario

TopologyBuilder = Callable[[int, int], "nx.DiGraph[object]"]


def _chain_with_shortcuts(hosts: int, seed: int) -> nx.DiGraph[object]:
    """The generator's own shape, included as the in-distribution reference."""

    graph: nx.DiGraph[object] = nx.DiGraph()
    graph.add_edges_from((f"h{index}", f"h{index + 1}") for index in range(hosts - 1))
    rng = random.Random(seed)  # noqa: S311 - topology shape, not cryptography
    for index in range(hosts - 2):
        if rng.random() < 0.5:
            graph.add_edge(f"h{index}", f"h{index + 2}")
    return graph


def _star(hosts: int, seed: int) -> nx.DiGraph[object]:
    """One hub reaching every leaf: breadth with no depth."""

    del seed
    return nx.DiGraph(("h0", f"h{index}") for index in range(1, hosts))


def _balanced_tree(hosts: int, seed: int) -> nx.DiGraph[object]:
    """A branching hierarchy: several distinct routes of equal depth."""

    del seed
    depth = max(1, (hosts - 1).bit_length() - 1)
    tree = nx.balanced_tree(2, depth, create_using=nx.DiGraph)
    return nx.DiGraph((f"h{source}", f"h{target}") for source, target in tree.edges())


def _mesh(hosts: int, seed: int) -> nx.DiGraph[object]:
    """A densely connected segment: many equivalent paths."""

    rng = random.Random(seed)  # noqa: S311 - topology shape, not cryptography
    graph: nx.DiGraph[object] = nx.DiGraph()
    graph.add_edges_from((f"h{index}", f"h{index + 1}") for index in range(hosts - 1))
    for source in range(hosts):
        for target in range(source + 1, hosts):
            if rng.random() < 0.45:
                graph.add_edge(f"h{source}", f"h{target}")
    return graph


def _ring(hosts: int, seed: int) -> nx.DiGraph[object]:
    """A cycle: no node is a natural entry and the shortest path wraps."""

    del seed
    return nx.DiGraph((f"h{index}", f"h{(index + 1) % hosts}") for index in range(hosts))


FAMILIES: dict[str, TopologyBuilder] = {
    "chain": _chain_with_shortcuts,
    "star": _star,
    "tree": _balanced_tree,
    "mesh": _mesh,
    "ring": _ring,
}

HELD_OUT_FAMILIES: tuple[str, ...] = ("star", "tree", "mesh", "ring")


def build_scenario(family: str, hosts: int, seed: int) -> Scenario:
    """Build one scenario of ``family`` through the sanitized import path."""

    if family not in FAMILIES:
        raise ValueError(f"unknown scenario family: {family}")
    if hosts < 3:
        raise ValueError("a family scenario needs at least three hosts")
    graph = FAMILIES[family](hosts, seed)
    return scenario_from_graph(
        graph, scenario_id=f"family-{family}-{hosts}-{seed}", name=f"{family} topology"
    )


def evaluate_families(
    agent_factory: Callable[[str, int], Agent],
    seeds: tuple[int, ...],
    families: Sequence[str] = tuple(FAMILIES),
    *,
    hosts: int = 8,
    step_budget: int = 96,
    reward_strategy: RewardStrategy = "risk-aware",
    dynamics: DynamicsConfig | None = None,
    defender: DefenderConfig | None = None,
) -> dict[str, BenchmarkMetrics]:
    """Evaluate one agent on every family, on identical seeds and one interface.

    ``agent_factory`` receives the family as well as the seed, because a graph-aware
    baseline has to be built from the scenario it will act in.
    """

    if not families:
        raise ValueError("at least one scenario family is required")
    reward_config = build_reward_config(reward_strategy)
    interface = ObservationConfig.for_curriculum()

    def env_factory(family: str) -> Callable[[int], AttackPathEnv]:
        def build(seed: int) -> AttackPathEnv:
            return AttackPathEnv(
                build_scenario(family, hosts, seed),
                step_budget=step_budget,
                reward_config=reward_config,
                dynamics=dynamics,
                observation_config=interface,
                defender=defender,
            )

        return build

    def bind(family: str) -> Callable[[int], Agent]:
        def make(seed: int) -> Agent:
            return agent_factory(family, seed)

        return make

    return {
        family: evaluate_agent(family, bind(family), env_factory(family), seeds)
        for family in families
    }


@dataclass(frozen=True)
class FamilyShape:
    """Structural summary of one family, for reporting what was actually tested."""

    family: str
    hosts: int
    edges: int
    mean_out_degree: float
    diameter: int
    route_diversity: int = 1

    @classmethod
    def measure(cls, family: str, hosts: int, seed: int) -> FamilyShape:
        """Measure the shape of one generated family instance."""

        graph = FAMILIES[family](hosts, seed)
        undirected = graph.to_undirected()
        components = nx.connected_components(undirected)
        largest = max(components, key=len)
        return cls(
            family=family,
            hosts=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            mean_out_degree=graph.number_of_edges() / max(1, graph.number_of_nodes()),
            diameter=nx.diameter(undirected.subgraph(largest)),
            route_diversity=_route_diversity(graph),
        )


def _route_diversity(graph: nx.DiGraph[object]) -> int:
    """Return how many node-disjoint routes lead from the entry to the deepest host.

    This is the structural variable that decides whether targeted monitoring can be
    routed around at all. With one route there is no alternative to take, so a
    defender's attention cannot be evaded however it is shaped - which is why the
    policy grid mixes on ``mesh`` and stays pure on every other family.
    """

    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return 0
    entry = nodes[0]
    lengths = nx.single_source_shortest_path_length(graph, entry)
    reachable = {node: length for node, length in lengths.items() if node != entry}
    if not reachable:
        return 0
    deepest = max(reachable, key=lambda node: (reachable[node], str(node)))
    return len(list(nx.node_disjoint_paths(graph, entry, deepest)))
