import json
from pathlib import Path

import networkx as nx
import pytest

from rlattack.agents import ShortestPathOracle
from rlattack.env import AttackPathEnv, DynamicsConfig
from rlattack.importers import import_scenario_file, load_graph, scenario_from_graph


def topology() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [("web", "app-1"), ("web", "app-2"), ("app-1", "db"), ("app-2", "db"), ("db", "backup")]
    )
    return graph


def write_graphml(tmp_path: Path) -> Path:
    path = tmp_path / "topology.graphml"
    nx.write_graphml(topology(), path)
    return path


def test_every_supported_format_loads(tmp_path: Path) -> None:
    graphml = write_graphml(tmp_path)
    gml = tmp_path / "topology.gml"
    nx.write_gml(topology(), gml)
    node_link = tmp_path / "topology.json"
    node_link.write_text(json.dumps(nx.node_link_data(topology(), edges="edges")), encoding="utf-8")

    for path in (graphml, gml, node_link):
        assert load_graph(path).number_of_edges() == 5

    with pytest.raises(ValueError, match="unsupported attack graph format"):
        load_graph(tmp_path / "topology.txt")


def test_imported_identifiers_are_replaced(tmp_path: Path) -> None:
    scenario = import_scenario_file(write_graphml(tmp_path))
    identifiers = {host.id for host in scenario.hosts}

    assert identifiers == {f"host-{index:02d}" for index in range(5)}
    assert scenario.model_dump_json().find("app-1") == -1
    assert all(host.hostname is None for host in scenario.hosts)


def test_the_imported_structure_is_preserved(tmp_path: Path) -> None:
    scenario = import_scenario_file(write_graphml(tmp_path))
    source = topology()

    assert len(scenario.hosts) == source.number_of_nodes()
    assert len(scenario.network_edges) == source.number_of_edges()
    degrees = sorted(
        len([edge for edge in scenario.network_edges if edge.source_host_id == host.id])
        for host in scenario.hosts
    )
    assert degrees == sorted(dict(source.out_degree()).values())


def test_an_imported_topology_is_playable(tmp_path: Path) -> None:
    scenario = import_scenario_file(write_graphml(tmp_path))
    env = AttackPathEnv(scenario, step_budget=150, dynamics=DynamicsConfig.deterministic())
    agent = ShortestPathOracle(scenario)
    observation, info = env.reset(seed=0)
    terminated = truncated = False
    while not terminated and not truncated:
        observation, _, terminated, truncated, info = env.step(agent.predict(observation, info))

    assert info["objective_captured"] is True


def test_topology_only_imports_skip_the_exploitation_layer(tmp_path: Path) -> None:
    scenario = import_scenario_file(write_graphml(tmp_path), synthesize_layers=False)

    assert scenario.services == ()
    assert scenario.objectives == ()
    assert len(scenario.network_edges) == 5


def test_edge_costs_are_carried_over(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=2.5)
    graph.add_edge("b", "c", cost=4.0)

    scenario = scenario_from_graph(graph)
    costs = sorted(edge.cost for edge in scenario.network_edges)

    assert costs == [2.5, 4.0]


def test_unusable_graphs_are_rejected(tmp_path: Path) -> None:
    empty = nx.DiGraph()
    empty.add_node("v", kind="vulnerability")
    with pytest.raises(ValueError, match="no host nodes"):
        scenario_from_graph(empty)

    isolated = nx.DiGraph()
    isolated.add_node("a")
    with pytest.raises(ValueError, match="no reachability edges"):
        scenario_from_graph(isolated)


def test_live_identifiers_are_rejected(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_node("a", hostname="prod-db")
    graph.add_edge("a", "b")

    with pytest.raises(ValueError, match="unsanitized"):
        scenario_from_graph(graph)


def test_a_cyclic_graph_still_picks_an_entry_and_objective() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

    scenario = scenario_from_graph(graph)

    assert scenario.entry_host_ids[0] in {host.id for host in scenario.hosts}
    assert scenario.objectives[0].host_id != scenario.entry_host_ids[0]
