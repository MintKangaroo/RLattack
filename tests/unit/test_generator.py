import networkx as nx
import pytest

from rlattack.generator import generate_scenario


def test_generator_is_reproducible_for_same_inputs() -> None:
    first = generate_scenario("medium", "hard", seed=42)
    second = generate_scenario("medium", "hard", seed=42)

    assert first.model_dump() == second.model_dump()


def test_generator_sizes_are_deterministic_and_connected() -> None:
    expected_hosts = {"small": 3, "medium": 6, "large": 12}

    for size, host_count in expected_hosts.items():
        scenario = generate_scenario(size, "easy", seed=1)  # type: ignore[arg-type]
        graph = scenario.to_networkx()
        host_graph = nx.DiGraph(
            (source, target)
            for source, target, data in graph.edges(data=True)
            if data["kind"] == "network"
        )

        assert len(scenario.hosts) == host_count
        assert scenario.entry_host_ids == ("host-00",)
        assert nx.is_directed_acyclic_graph(host_graph)
        assert nx.has_path(host_graph, "host-00", f"host-{host_count - 1:02d}")


def test_seed_changes_stochastic_scenario_attributes() -> None:
    first = generate_scenario("large", "hard", seed=10)
    second = generate_scenario("large", "hard", seed=11)

    assert first.id != second.id
    assert first.model_dump() != second.model_dump()


def test_difficulty_controls_shortcut_count() -> None:
    easy = generate_scenario("large", "easy", seed=4)
    hard = generate_scenario("large", "hard", seed=4)

    assert len(hard.network_edges) >= len(easy.network_edges)


def test_invalid_generator_arguments_are_rejected() -> None:
    with pytest.raises((KeyError, ValueError)):
        generate_scenario("tiny", "easy", seed=0)  # type: ignore[arg-type]
    with pytest.raises((KeyError, ValueError)):
        generate_scenario("small", "extreme", seed=0)  # type: ignore[arg-type]
