"""Sanitized, file-oriented ThreatGraph Scenario adapter.

This module intentionally accepts data rather than a live ThreatGraph client. It strips
identifiers and rejects fields that could carry actionable target or exploit material.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

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
    node_map: dict[str, str] = {}
    hosts: list[Host] = []
    services: list[Service] = []
    vulnerabilities: list[Vulnerability] = []
    for node in raw_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("ThreatGraph nodes must be objects")
        original_id = str(node.get("id", ""))
        kind = str(node.get("kind", "")).lower()
        attributes = node.get("attributes", {})
        if not original_id or not isinstance(attributes, Mapping):
            raise ValueError("ThreatGraph nodes require an id and attributes object")
        if kind == "host":
            sanitized_id = f"host-{len(hosts):02d}"
            hosts.append(
                Host(id=sanitized_id, operating_system=str(attributes.get("os", "unknown")))
            )
        elif kind == "service":
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
        elif kind == "vulnerability":
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
                )
            )
        else:
            continue
        node_map[original_id] = sanitized_id
    network_edges: list[NetworkEdge] = []
    for edge in raw_edges:
        if not isinstance(edge, Mapping):
            raise ValueError("ThreatGraph edges must be objects")
        source = node_map.get(str(edge.get("source", "")))
        target = node_map.get(str(edge.get("target", "")))
        if source in {host.id for host in hosts} and target in {host.id for host in hosts}:
            network_edges.append(NetworkEdge(source_host_id=source, target_host_id=target))
    return Scenario(
        id=f"sanitized-{payload.get('id', 'scenario')}",
        name="Sanitized ThreatGraph Scenario",
        entry_host_ids=(hosts[0].id,) if hosts else (),
        hosts=tuple(hosts),
        services=tuple(services),
        vulnerabilities=tuple(vulnerabilities),
        network_edges=tuple(network_edges),
    )


def export_sanitized_scenario(scenario: Scenario) -> dict[str, Any]:
    """Export only anonymous Scenario fields suitable for a file fixture."""

    return {
        "id": scenario.id,
        "kind": "sanitized_rlattack_scenario",
        "nodes": [
            {
                "id": host.id,
                "kind": "host",
                "attributes": {"os": host.operating_system or "unknown"},
            }
            for host in scenario.hosts
        ],
        "edges": [
            {"source": edge.source_host_id, "target": edge.target_host_id, "kind": "network"}
            for edge in scenario.network_edges
        ],
    }
