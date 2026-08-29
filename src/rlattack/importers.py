"""Import externally published attack graphs as sanitized scenarios.

Results measured only on this repository's generator describe that generator. This
module reads a topology from a file - GraphML, GML, or NetworkX node-link JSON, the
formats attack-graph tooling commonly emits - and converts it into a validated
:class:`~rlattack.scenario.Scenario`.

Imported identifiers are never kept. Every node is renamed to an anonymous ID, and the
payload must pass the same sanitization as the ThreatGraph adapter, so an import cannot
smuggle a hostname, an address, or exploit material into a scenario file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from rlattack.adapter import assert_sanitized
from rlattack.scenario import (
    AccessEdge,
    Credential,
    Host,
    NetworkEdge,
    Objective,
    Privilege,
    PrivilegeEdge,
    Scenario,
    SecurityControl,
    Service,
    Vulnerability,
)

HOST_KINDS = {"host", "node", "machine", "asset", ""}


def load_graph(path: Path) -> nx.DiGraph[Any]:
    """Read a directed graph from GraphML, GML, or NetworkX node-link JSON."""

    suffix = path.suffix.lower()
    if suffix == ".graphml":
        return nx.DiGraph(nx.read_graphml(path))
    if suffix == ".gml":
        return nx.DiGraph(nx.read_gml(path))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert_sanitized(payload)
        return nx.DiGraph(nx.node_link_graph(payload, edges="edges"))
    raise ValueError(f"unsupported attack graph format: {path.suffix or '(none)'}")


def _host_nodes(graph: nx.DiGraph[Any]) -> list[Any]:
    hosts = [
        node
        for node, data in graph.nodes(data=True)
        if str(data.get("kind", data.get("type", ""))).lower() in HOST_KINDS
    ]
    if not hosts:
        raise ValueError("the imported graph contains no host nodes")
    return hosts


def scenario_from_graph(
    graph: nx.DiGraph[Any],
    *,
    scenario_id: str = "imported",
    name: str = "Imported attack graph",
    synthesize_layers: bool = True,
) -> Scenario:
    """Convert an imported topology into a validated, anonymized scenario.

    Published attack graphs usually publish reachability between hosts and nothing
    else. ``synthesize_layers`` attaches the deterministic service, vulnerability,
    credential, and objective layer this simulator needs to make such a topology
    playable; the imported *structure* is preserved either way. Turn it off to import
    the topology alone.
    """

    assert_sanitized({str(node): dict(data) for node, data in graph.nodes(data=True)})
    nodes = _host_nodes(graph)
    anonymous = {node: f"host-{index:02d}" for index, node in enumerate(sorted(nodes, key=str))}
    hosts = tuple(
        Host(id=host_id, operating_system=None, zone="imported") for host_id in anonymous.values()
    )
    network_edges = tuple(
        NetworkEdge(
            source_host_id=anonymous[source],
            target_host_id=anonymous[target],
            cost=float(data.get("cost", data.get("weight", 1.0))),
        )
        for source, target, data in graph.edges(data=True)
        if source in anonymous and target in anonymous
    )
    if not network_edges:
        raise ValueError("the imported graph contains no reachability edges between hosts")

    entry = _entry_host(graph, anonymous)
    if not synthesize_layers:
        return Scenario(
            id=scenario_id,
            name=name,
            entry_host_ids=(entry,),
            hosts=hosts,
            network_edges=network_edges,
        )

    host_ids = tuple(anonymous.values())
    services = tuple(
        Service(id=f"{host_id}-svc", host_id=host_id, name="service", port=22, protocol="tcp")
        for host_id in host_ids
    )
    vulnerabilities = tuple(
        Vulnerability(
            id=f"vuln-{service.id}",
            service_id=service.id,
            name="imported-weakness",
            exploitability=0.6,
        )
        for service in services
    )
    credentials = tuple(
        Credential(
            id=f"cred-{host_id}", host_id=host_id, username="simulated", privilege_id="priv-user"
        )
        for host_id in host_ids
    )
    return Scenario(
        id=scenario_id,
        name=name,
        entry_host_ids=(entry,),
        hosts=hosts,
        services=services,
        vulnerabilities=vulnerabilities,
        credentials=credentials,
        privileges=(
            Privilege(id="priv-user", name="simulated user", level=10),
            Privilege(id="priv-admin", name="simulated administrator", level=90),
        ),
        objectives=(
            Objective(
                id="objective-imported",
                name="collect simulated objective",
                host_id=_objective_host(graph, anonymous, entry),
                required_privilege_id="priv-admin",
            ),
        ),
        security_controls=tuple(
            SecurityControl(
                id=f"control-{host_id}",
                name="imported detection control",
                detection_probability=0.15,
                host_ids=(host_id,),
            )
            for host_id in host_ids
        ),
        network_edges=network_edges,
        access_edges=tuple(
            AccessEdge(vulnerability_id=f"vuln-{host_id}-svc", credential_id=f"cred-{host_id}")
            for host_id in host_ids
        ),
        privilege_edges=(
            PrivilegeEdge(source_privilege_id="priv-user", target_privilege_id="priv-admin"),
        ),
    )


def _entry_host(graph: nx.DiGraph[Any], anonymous: dict[Any, str]) -> str:
    """Pick the entry host: a source node if the graph has one, else the first host."""

    sources = [node for node in anonymous if graph.in_degree(node) == 0]
    return anonymous[sources[0] if sources else next(iter(anonymous))]


def _objective_host(graph: nx.DiGraph[Any], anonymous: dict[Any, str], entry: str) -> str:
    """Place the objective on the reachable host furthest from the entry."""

    origin = next(node for node, host_id in anonymous.items() if host_id == entry)
    distances = nx.single_source_shortest_path_length(graph, origin)
    reachable = {node: depth for node, depth in distances.items() if node in anonymous}
    furthest = max(reachable, key=lambda node: (reachable[node], str(node)))
    return anonymous[furthest]


def import_scenario_file(
    path: Path,
    *,
    scenario_id: str | None = None,
    synthesize_layers: bool = True,
) -> Scenario:
    """Load and convert one attack-graph file into a sanitized scenario."""

    graph = load_graph(path)
    return scenario_from_graph(
        graph,
        scenario_id=scenario_id or f"imported-{path.stem}",
        name=f"Imported {path.stem}",
        synthesize_layers=synthesize_layers,
    )
