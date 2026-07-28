"""Deterministic generators for synthetic RLAttack scenarios."""

from __future__ import annotations

import random
from typing import Literal

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

ScenarioSize = Literal["small", "medium", "large"]
Difficulty = Literal["easy", "medium", "hard"]

_HOST_COUNTS: dict[ScenarioSize, int] = {"small": 3, "medium": 6, "large": 12}
_DIFFICULTY_BRANCHING: dict[Difficulty, float] = {"easy": 0.0, "medium": 0.35, "hard": 0.7}


def generate_scenario(
    size: ScenarioSize = "small",
    difficulty: Difficulty = "medium",
    seed: int = 0,
) -> Scenario:
    """Generate a deterministic, sanitized scenario from a size, difficulty, and seed.

    The generator uses a private ``random.Random`` instance and never reads external data,
    opens a network connection, or creates executable exploit material.
    """

    rng = random.Random(seed)  # noqa: S311 - deterministic simulation, not cryptography
    host_count = _HOST_COUNTS[size]
    host_ids = tuple(f"host-{index:02d}" for index in range(host_count))
    hosts = tuple(
        Host(
            id=host_id,
            hostname=f"{host_id}.sim",
            operating_system=("linux" if index % 2 == 0 else "windows"),
            zone=("entry" if index == 0 else "internal"),
        )
        for index, host_id in enumerate(host_ids)
    )
    services = tuple(
        service
        for host_index, host_id in enumerate(host_ids)
        for service in (
            Service(
                id=f"{host_id}-ssh",
                host_id=host_id,
                name="ssh",
                port=22,
                protocol="tcp",
                version="simulated",
            ),
            Service(
                id=f"{host_id}-web",
                host_id=host_id,
                name="http",
                port=80,
                protocol="tcp",
                version="simulated",
            )
            if host_index % 2 == 0
            else Service(
                id=f"{host_id}-db",
                host_id=host_id,
                name="database",
                port=5432,
                protocol="tcp",
                version="simulated",
            ),
        )
    )
    vulnerabilities = tuple(
        Vulnerability(
            id=f"vuln-{service.id}",
            service_id=service.id,
            name=f"simulated-{service.name}-weakness",
            severity=("high" if service.name == "database" else "medium"),
            exploitability=round(rng.uniform(0.35, 0.9), 3),
        )
        for service in services
    )
    privileges = (
        Privilege(id="priv-user", name="simulated user", level=10),
        Privilege(id="priv-admin", name="simulated administrator", level=90),
    )
    credentials = (
        Credential(
            id="cred-entry-user",
            host_id=host_ids[0],
            username="simulated-agent",
            privilege_id="priv-user",
        ),
    )
    objectives = (
        Objective(
            id="objective-final-host",
            name="collect simulated objective",
            host_id=host_ids[-1],
            required_privilege_id="priv-admin",
        ),
    )
    network_edges = _network_edges(host_ids, difficulty, rng)
    security_controls = tuple(
        SecurityControl(
            id=f"control-{host_id}",
            name="simulated detection control",
            detection_probability=round(
                0.05 + rng.random() * (0.15 if difficulty == "easy" else 0.35),
                3,
            ),
            host_ids=(host_id,),
        )
        for host_id in host_ids
    )
    return Scenario(
        id=f"generated-{size}-{difficulty}-{seed}",
        name=f"Generated {size} {difficulty} scenario",
        version="1",
        entry_host_ids=(host_ids[0],),
        hosts=hosts,
        services=services,
        vulnerabilities=vulnerabilities,
        credentials=credentials,
        privileges=privileges,
        objectives=objectives,
        security_controls=security_controls,
        network_edges=network_edges,
        access_edges=(
            AccessEdge(vulnerability_id=f"vuln-{services[0].id}", credential_id="cred-entry-user"),
        ),
        privilege_edges=(
            PrivilegeEdge(source_privilege_id="priv-user", target_privilege_id="priv-admin"),
        ),
    )


def _network_edges(
    host_ids: tuple[str, ...], difficulty: Difficulty, rng: random.Random
) -> tuple[NetworkEdge, ...]:
    """Create a connected chain plus deterministic difficulty-dependent shortcuts."""

    edges = [
        NetworkEdge(
            source_host_id=source,
            target_host_id=target,
            cost=1.0,
            initially_reachable=index == 0,
        )
        for index, (source, target) in enumerate(zip(host_ids[:-1], host_ids[1:], strict=True))
    ]
    branching = _DIFFICULTY_BRANCHING[difficulty]
    for source_index, source in enumerate(host_ids[:-2]):
        target = host_ids[source_index + 2]
        if rng.random() < branching:
            edges.append(
                NetworkEdge(
                    source_host_id=source,
                    target_host_id=target,
                    cost=round(1.5 + rng.random(), 2),
                )
            )
    return tuple(edges)
