"""Sanitized, file-oriented ThreatGraph Scenario adapter.

This module intentionally accepts data rather than a live ThreatGraph client. It strips
identifiers and rejects fields that could carry actionable target or exploit material.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, cast

from rlattack.scenario import Host, NetworkEdge, Scenario, Service, Vulnerability

_FORBIDDEN_KEYS = {
    "ip",
    "ip_address",
    "domain",
    "hostname",
    "url",
    "command",
    "payload",
    "exploit",
    "secret",
    "password",
    "token",
    "private_key",
}
_IP_OR_URL = re.compile(r"(?:https?://|\b(?:\d{1,3}\.){3}\d{1,3}\b)", re.IGNORECASE)


def _assert_sanitized(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"unsanitized ThreatGraph field at {path}.{key}")
            _assert_sanitized(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_sanitized(child, f"{path}[{index}]")
    elif isinstance(value, str) and _IP_OR_URL.search(value):
        raise ValueError(f"live target identifier at {path}")


def import_sanitized_threatgraph(payload: Mapping[str, Any]) -> Scenario:
    """Convert an anonymized ThreatGraph export into a validated Scenario."""

    _assert_sanitized(payload)
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("ThreatGraph export must contain nodes and edges lists")
    records: list[tuple[str, str, Mapping[str, Any]]] = []
    raw_ids: set[str] = set()
    for node in raw_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("ThreatGraph nodes must be objects")
        original_id = str(node.get("id", ""))
        kind = str(node.get("kind", "")).lower()
        attributes = node.get("attributes", {})
        if not original_id or not isinstance(attributes, Mapping):
            raise ValueError("ThreatGraph nodes require an id and attributes object")
        if original_id in raw_ids:
            raise ValueError("ThreatGraph node ids must be unique")
        raw_ids.add(original_id)
        records.append((original_id, kind, attributes))

    node_map: dict[str, str] = {}
    hosts: list[Host] = []
    services: list[Service] = []
    vulnerabilities: list[Vulnerability] = []
    for original_id, kind, attributes in records:
        if kind == "host":
            sanitized_id = f"host-{len(hosts):02d}"
            hosts.append(
                Host(id=sanitized_id, operating_system=str(attributes.get("os", "unknown")))
            )
            node_map[original_id] = sanitized_id

    for original_id, kind, attributes in records:
        if kind == "service":
            host_ref = str(attributes.get("host_ref", ""))
            sanitized_id = f"service-{len(services):02d}"
            host_id = node_map.get(host_ref)
            if host_id is None:
                raise ValueError("service references an unknown host")
            services.append(
                Service(
                    id=sanitized_id,
                    host_id=host_id,
                    name=str(attributes.get("name", "service")),
                    port=int(attributes.get("port", 1)),
                )
            )
            node_map[original_id] = sanitized_id

    for original_id, kind, attributes in records:
        if kind == "vulnerability":
            service_ref = str(attributes.get("service_ref", ""))
            sanitized_id = f"vulnerability-{len(vulnerabilities):02d}"
            service_id = node_map.get(service_ref)
            if service_id is None:
                raise ValueError("vulnerability references an unknown service")
            vulnerabilities.append(
                Vulnerability(
                    id=sanitized_id,
                    service_id=service_id,
                    name=str(attributes.get("name", "simulated vulnerability")),
                    severity=cast(
                        Literal["low", "medium", "high", "critical"],
                        str(attributes.get("severity", "medium")),
                    ),
                )
            )
            node_map[original_id] = sanitized_id

    raw_entry_refs = payload.get("entry_refs")
    entry_host_ids: tuple[str, ...]
    if raw_entry_refs is None:
        entry_host_ids = (hosts[0].id,) if hosts else ()
    elif not isinstance(raw_entry_refs, list):
        raise ValueError("ThreatGraph entry_refs must be a list")
    else:
        entry_host_ids = tuple(node_map.get(str(reference), "") for reference in raw_entry_refs)
        host_ids = {host.id for host in hosts}
        if any(entry not in host_ids for entry in entry_host_ids):
            raise ValueError("entry_refs references an unknown host")

    network_edges: list[NetworkEdge] = []
    host_ids = {host.id for host in hosts}
    for edge in raw_edges:
        if not isinstance(edge, Mapping):
            raise ValueError("ThreatGraph edges must be objects")
        source = node_map.get(str(edge.get("source", "")))
        target = node_map.get(str(edge.get("target", "")))
        if source in host_ids and target in host_ids:
            network_edges.append(
                NetworkEdge(
                    source_host_id=str(source),
                    target_host_id=str(target),
                    cost=float(edge.get("cost", 1.0)),
                    initially_reachable=bool(edge.get("initially_reachable", False)),
                )
            )
    return Scenario(
        id="sanitized-scenario",
        name="Sanitized ThreatGraph Scenario",
        entry_host_ids=entry_host_ids,
        hosts=tuple(hosts),
        services=tuple(services),
        vulnerabilities=tuple(vulnerabilities),
        network_edges=tuple(network_edges),
    )


def export_sanitized_scenario(scenario: Scenario) -> dict[str, Any]:
    """Export re-anonymized Scenario fields suitable for a safe file fixture."""

    host_map = {host.id: f"host-{index:02d}" for index, host in enumerate(scenario.hosts)}
    service_map = {
        service.id: f"service-{index:02d}" for index, service in enumerate(scenario.services)
    }
    vulnerability_map = {
        vulnerability.id: f"vulnerability-{index:02d}"
        for index, vulnerability in enumerate(scenario.vulnerabilities)
    }
    exported: dict[str, Any] = {
        "id": "sanitized-scenario",
        "kind": "sanitized_rlattack_scenario",
        "entry_refs": [host_map[host_id] for host_id in scenario.entry_host_ids],
        "nodes": [
            *(
                {
                    "id": host_map[host.id],
                    "kind": "host",
                    "attributes": {"os": host.operating_system or "unknown"},
                }
                for host in scenario.hosts
            ),
            *(
                {
                    "id": service_map[service.id],
                    "kind": "service",
                    "attributes": {
                        "host_ref": host_map[service.host_id],
                        "name": service.name,
                        "port": service.port,
                    },
                }
                for service in scenario.services
            ),
            *(
                {
                    "id": vulnerability_map[vulnerability.id],
                    "kind": "vulnerability",
                    "attributes": {
                        "service_ref": service_map[vulnerability.service_id],
                        "name": vulnerability.name,
                        "severity": vulnerability.severity,
                    },
                }
                for vulnerability in scenario.vulnerabilities
            ),
        ],
        "edges": [
            {
                "source": host_map[edge.source_host_id],
                "target": host_map[edge.target_host_id],
                "kind": "network",
                "cost": edge.cost,
                "initially_reachable": edge.initially_reachable,
            }
            for edge in scenario.network_edges
        ],
    }
    _assert_sanitized(exported)
    return exported
