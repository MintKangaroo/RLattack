import pytest
from pydantic import ValidationError

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


def make_scenario() -> Scenario:
    return Scenario(
        id="demo",
        name="Deterministic demo",
        hosts=(Host(id="web", hostname="web.local"), Host(id="db")),
        services=(Service(id="web-http", host_id="web", name="http", port=80),),
        vulnerabilities=(
            Vulnerability(
                id="web-vuln",
                service_id="web-http",
                name="simulated web weakness",
                exploitability=0.5,
            ),
        ),
        privileges=(Privilege(id="user", name="user", level=10),),
        credentials=(
            Credential(id="web-user", host_id="web", username="researcher", privilege_id="user"),
        ),
        objectives=(Objective(id="objective", name="collect token", host_id="db"),),
        security_controls=(SecurityControl(id="edr", name="simulated EDR", host_ids=("web",)),),
        network_edges=(NetworkEdge(source_host_id="web", target_host_id="db"),),
        access_edges=(AccessEdge(vulnerability_id="web-vuln", credential_id="web-user"),),
        privilege_edges=(PrivilegeEdge(source_privilege_id="user", target_privilege_id="user"),),
    )


def test_scenario_validates_references_and_builds_graph() -> None:
    scenario = make_scenario()

    graph = scenario.to_networkx()

    assert graph.nodes["web"]["kind"] == "host"
    assert graph.nodes["web-http"]["kind"] == "service"
    assert graph.has_edge("web", "db")
    assert graph.get_edge_data("web-vuln", "web-user")[0]["kind"] == "access"


def test_unknown_reference_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown host 'missing'"):
        Scenario(
            id="invalid",
            name="Invalid",
            hosts=(Host(id="web"),),
            services=(Service(id="http", host_id="missing", name="http", port=80),),
        )


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate host id"):
        Scenario(id="invalid", name="Invalid", hosts=(Host(id="web"), Host(id="web")))


def test_entity_constraints_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Service(id="http", host_id="web", name="http", port=70000)

    with pytest.raises(ValidationError):
        Host(id="bad id")


def test_scenario_records_are_immutable() -> None:
    host = Host(id="web")

    with pytest.raises(ValidationError):
        host.id = "changed"
