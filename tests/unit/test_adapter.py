import json

import pytest

from rlattack.adapter import export_sanitized_scenario, import_sanitized_threatgraph
from rlattack.generator import generate_scenario
from rlattack.scenario import Host, Scenario


def test_sanitized_import_anonymizes_nodes_and_preserves_safe_graph() -> None:
    payload = {
        "id": "fixture-1",
        "nodes": [
            {
                "id": "raw-vulnerability",
                "kind": "vulnerability",
                "attributes": {"service_ref": "raw-service", "name": "weakness"},
            },
            {
                "id": "raw-service",
                "kind": "service",
                "attributes": {"host_ref": "raw-a", "name": "http", "port": 80},
            },
            {"id": "raw-a", "kind": "host", "attributes": {"os": "linux"}},
            {"id": "ignored", "kind": "event", "attributes": {}},
        ],
        "edges": [{"source": "raw-a", "target": "raw-a", "kind": "network"}],
    }

    scenario = import_sanitized_threatgraph(payload)

    assert scenario.hosts[0].id == "host-00"
    assert scenario.services[0].host_id == "host-00"
    assert scenario.vulnerabilities[0].service_id == "service-00"
    assert len(scenario.network_edges) == 1
    assert "raw-a" not in scenario.model_dump_json()


def test_export_contains_only_sanitized_fields() -> None:
    scenario = generate_scenario("small", "easy", seed=1)
    exported = export_sanitized_scenario(scenario)
    restored = import_sanitized_threatgraph(exported)
    encoded = json.dumps(exported)

    assert exported["kind"] == "sanitized_rlattack_scenario"
    assert all("ip" not in str(node).lower() for node in exported["nodes"])
    assert scenario.id not in encoded
    assert scenario.services[0].id not in encoded
    assert len(restored.hosts) == len(scenario.hosts)
    assert len(restored.services) == len(scenario.services)
    assert len(restored.vulnerabilities) == len(scenario.vulnerabilities)
    assert restored.entry_host_ids == scenario.entry_host_ids
    assert restored.network_edges[0].cost == scenario.network_edges[0].cost


@pytest.mark.parametrize("field", ("ip", "domain", "command", "payload", "secret"))
def test_unsafe_fields_are_rejected(field: str) -> None:
    payload = {"nodes": [], "edges": [], field: "unsafe"}

    with pytest.raises(ValueError, match="unsanitized"):
        import_sanitized_threatgraph(payload)


def test_live_target_identifier_is_rejected_and_invalid_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="live target"):
        import_sanitized_threatgraph({"nodes": [], "edges": [], "note": "10.0.0.1"})
    with pytest.raises(ValueError, match="lists"):
        import_sanitized_threatgraph({"nodes": {}, "edges": []})


def test_malformed_nodes_edges_and_references_are_rejected() -> None:
    with pytest.raises(ValueError, match="nodes must be objects"):
        import_sanitized_threatgraph({"nodes": ["bad"], "edges": []})
    with pytest.raises(ValueError, match="require an id"):
        import_sanitized_threatgraph({"nodes": [{"kind": "host"}], "edges": []})
    with pytest.raises(ValueError, match="unknown host"):
        import_sanitized_threatgraph(
            {
                "nodes": [
                    {"id": "service", "kind": "service", "attributes": {"host_ref": "missing"}}
                ],
                "edges": [],
            }
        )
    with pytest.raises(ValueError, match="unknown service"):
        import_sanitized_threatgraph(
            {
                "nodes": [
                    {
                        "id": "vulnerability",
                        "kind": "vulnerability",
                        "attributes": {"service_ref": "missing"},
                    }
                ],
                "edges": [],
            }
        )
    with pytest.raises(ValueError, match="edges must be objects"):
        import_sanitized_threatgraph({"nodes": [], "edges": ["bad"]})


def test_adapter_rejects_ambiguous_refs_and_unsafe_export_values() -> None:
    duplicate = {
        "nodes": [
            {"id": "same", "kind": "host", "attributes": {}},
            {"id": "same", "kind": "host", "attributes": {}},
        ],
        "edges": [],
    }
    with pytest.raises(ValueError, match="unique"):
        import_sanitized_threatgraph(duplicate)
    with pytest.raises(ValueError, match="entry_refs must be a list"):
        import_sanitized_threatgraph({"nodes": [], "edges": [], "entry_refs": "host"})
    with pytest.raises(ValueError, match="unknown host"):
        import_sanitized_threatgraph({"nodes": [], "edges": [], "entry_refs": ["missing"]})
    unsafe = Scenario(
        id="unsafe-export",
        name="unsafe",
        hosts=(Host(id="source", operating_system="https://live.example"),),
    )
    with pytest.raises(ValueError, match="live target"):
        export_sanitized_scenario(unsafe)
