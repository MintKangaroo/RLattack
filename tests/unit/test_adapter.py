import pytest

from rlattack.adapter import export_sanitized_scenario, import_sanitized_threatgraph
from rlattack.generator import generate_scenario


def test_sanitized_import_anonymizes_nodes_and_preserves_safe_graph() -> None:
    payload = {
        "id": "fixture-1",
        "nodes": [
            {"id": "raw-a", "kind": "host", "attributes": {"os": "linux"}},
            {
                "id": "raw-service",
                "kind": "service",
                "attributes": {"host_ref": "raw-a", "name": "http", "port": 80},
            },
            {
                "id": "raw-vulnerability",
                "kind": "vulnerability",
                "attributes": {"service_ref": "raw-service", "name": "weakness"},
            },
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
    exported = export_sanitized_scenario(generate_scenario("small", "easy", seed=1))

    assert exported["kind"] == "sanitized_rlattack_scenario"
    assert all("ip" not in str(node).lower() for node in exported["nodes"])


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
