"""Validated, simulation-only attack graph scenario models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class ScenarioModel(BaseModel):
    """Common strict configuration for all scenario records."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Host(ScenarioModel):
    """A simulated host in the scenario graph."""

    id: Identifier
    hostname: str | None = None
    operating_system: str | None = None
    zone: str = "default"


class Service(ScenarioModel):
    """A simulated service exposed by a host."""

    id: Identifier
    host_id: Identifier
    name: str
    port: int = Field(ge=1, le=65535)
    protocol: str = "tcp"
    version: str | None = None


class Vulnerability(ScenarioModel):
    """A modeled vulnerability; it never represents an executable exploit."""

    id: Identifier
    service_id: Identifier
    name: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    exploitability: Probability = 0.0


class Credential(ScenarioModel):
    """A simulated credential state attached to a host and privilege."""

    id: Identifier
    host_id: Identifier
    username: str
    privilege_id: Identifier


class Privilege(ScenarioModel):
    """A simulated privilege level."""

    id: Identifier
    name: str
    level: int = Field(ge=0, le=100)


class Objective(ScenarioModel):
    """A goal that an environment episode may collect."""

    id: Identifier
    name: str
    host_id: Identifier
    required_privilege_id: Identifier | None = None


class SecurityControl(ScenarioModel):
    """A simulation-only control that contributes to detection risk."""

    id: Identifier
    name: str
    detection_probability: Probability = 0.0
    host_ids: tuple[Identifier, ...] = ()


class NetworkEdge(ScenarioModel):
    """A directed reachability edge between simulated hosts."""

    source_host_id: Identifier
    target_host_id: Identifier
    cost: float = Field(default=1.0, ge=0.0)
    initially_reachable: bool = False


class AccessEdge(ScenarioModel):
    """A modeled transition from a vulnerability to a credential state."""

    vulnerability_id: Identifier
    credential_id: Identifier


class PrivilegeEdge(ScenarioModel):
    """A modeled privilege transition."""

    source_privilege_id: Identifier
    target_privilege_id: Identifier
    vulnerability_id: Identifier | None = None


class Scenario(ScenarioModel):
    """Complete validated graph data for one deterministic simulated scenario."""

    id: Identifier
    name: str
    version: str = "1"
    entry_host_ids: tuple[Identifier, ...] = ()
    hosts: tuple[Host, ...] = ()
    services: tuple[Service, ...] = ()
    vulnerabilities: tuple[Vulnerability, ...] = ()
    credentials: tuple[Credential, ...] = ()
    privileges: tuple[Privilege, ...] = ()
    objectives: tuple[Objective, ...] = ()
    security_controls: tuple[SecurityControl, ...] = ()
    network_edges: tuple[NetworkEdge, ...] = ()
    access_edges: tuple[AccessEdge, ...] = ()
    privilege_edges: tuple[PrivilegeEdge, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Scenario:
        """Reject duplicate IDs and references to records outside this scenario."""

        collections: tuple[tuple[str, tuple[ScenarioModel, ...]], ...] = (
            ("host", self.hosts),
            ("service", self.services),
            ("vulnerability", self.vulnerabilities),
            ("credential", self.credentials),
            ("privilege", self.privileges),
            ("objective", self.objectives),
            ("security control", self.security_controls),
        )
        ids: dict[str, set[str]] = {}
        for kind, records in collections:
            record_ids = [record.id for record in records if hasattr(record, "id")]
            if len(record_ids) != len(set(record_ids)):
                raise ValueError(f"duplicate {kind} id")
            ids[kind] = set(record_ids)

        self._require_refs(
            "entry_host_ids",
            ((host_id, "host") for host_id in self.entry_host_ids),
            ids,
        )
        self._require_refs(
            "service.host_id",
            ((record.host_id, "host") for record in self.services),
            ids,
        )
        self._require_refs(
            "vulnerability.service_id",
            ((record.service_id, "service") for record in self.vulnerabilities),
            ids,
        )
        self._require_refs(
            "credential.host_id",
            ((record.host_id, "host") for record in self.credentials),
            ids,
        )
        self._require_refs(
            "credential.privilege_id",
            ((record.privilege_id, "privilege") for record in self.credentials),
            ids,
        )
        self._require_refs(
            "objective.host_id",
            ((record.host_id, "host") for record in self.objectives),
            ids,
        )
        self._require_refs(
            "objective.required_privilege_id",
            (
                (record.required_privilege_id, "privilege")
                for record in self.objectives
                if record.required_privilege_id is not None
            ),
            ids,
        )
        self._require_refs(
            "security_control.host_ids",
            ((host_id, "host") for record in self.security_controls for host_id in record.host_ids),
            ids,
        )
        self._require_refs(
            "network_edge.host_id",
            (
                (host_id, "host")
                for edge in self.network_edges
                for host_id in (edge.source_host_id, edge.target_host_id)
            ),
            ids,
        )
        self._require_refs(
            "access_edge.vulnerability_id",
            ((edge.vulnerability_id, "vulnerability") for edge in self.access_edges),
            ids,
        )
        self._require_refs(
            "access_edge.credential_id",
            ((edge.credential_id, "credential") for edge in self.access_edges),
            ids,
        )
        self._require_refs(
            "privilege_edge.privilege_id",
            (
                (privilege_id, "privilege")
                for edge in self.privilege_edges
                for privilege_id in (edge.source_privilege_id, edge.target_privilege_id)
            ),
            ids,
        )
        self._require_refs(
            "privilege_edge.vulnerability_id",
            (
                (edge.vulnerability_id, "vulnerability")
                for edge in self.privilege_edges
                if edge.vulnerability_id is not None
            ),
            ids,
        )
        return self

    @staticmethod
    def _require_refs(
        field_name: str,
        references: Iterable[tuple[str, str]],
        ids: dict[str, set[str]],
    ) -> None:
        for reference, kind in references:
            if reference not in ids[kind]:
                raise ValueError(f"{field_name} references unknown {kind} '{reference}'")

    def to_networkx(self) -> nx.MultiDiGraph[str, dict[str, object], dict[str, object]]:
        """Build a NetworkX graph without performing any external I/O."""

        graph: nx.MultiDiGraph[str, dict[str, object], dict[str, object]] = nx.MultiDiGraph(
            id=self.id,
            name=self.name,
            version=self.version,
        )
        for host in self.hosts:
            graph.add_node(host.id, kind="host", data=host.model_dump())
        for service in self.services:
            graph.add_node(service.id, kind="service", data=service.model_dump())
            graph.add_edge(service.host_id, service.id, kind="hosts")
        for vulnerability in self.vulnerabilities:
            graph.add_node(vulnerability.id, kind="vulnerability", data=vulnerability.model_dump())
            graph.add_edge(vulnerability.service_id, vulnerability.id, kind="affects")
        for credential in self.credentials:
            graph.add_node(credential.id, kind="credential", data=credential.model_dump())
            graph.add_edge(credential.host_id, credential.id, kind="available_on")
        for privilege in self.privileges:
            graph.add_node(privilege.id, kind="privilege", data=privilege.model_dump())
        for objective in self.objectives:
            graph.add_node(objective.id, kind="objective", data=objective.model_dump())
            graph.add_edge(objective.host_id, objective.id, kind="contains")
        for control in self.security_controls:
            graph.add_node(control.id, kind="security_control", data=control.model_dump())
            for host_id in control.host_ids:
                graph.add_edge(control.id, host_id, kind="protects")
        for network_edge in self.network_edges:
            graph.add_edge(
                network_edge.source_host_id,
                network_edge.target_host_id,
                kind="network",
                cost=network_edge.cost,
                initially_reachable=network_edge.initially_reachable,
            )
        for access_edge in self.access_edges:
            graph.add_edge(
                access_edge.vulnerability_id,
                access_edge.credential_id,
                kind="access",
            )
        for privilege_edge in self.privilege_edges:
            graph.add_edge(
                privilege_edge.source_privilege_id,
                privilege_edge.target_privilege_id,
                kind="privilege",
                vulnerability_id=privilege_edge.vulnerability_id,
            )
        return graph
