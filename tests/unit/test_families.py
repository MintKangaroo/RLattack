from __future__ import annotations

import pytest

from rlattack.agents import Agent, GreedyAgent, ShortestPathOracle
from rlattack.env import DynamicsConfig
from rlattack.families import (
    FAMILIES,
    HELD_OUT_FAMILIES,
    FamilyShape,
    build_scenario,
    evaluate_families,
)


def greedy_factory(family: str, seed: int) -> Agent:
    del family, seed
    return GreedyAgent()


def test_the_generators_own_shape_is_the_only_in_distribution_family() -> None:
    assert set(FAMILIES) - set(HELD_OUT_FAMILIES) == {"chain"}


def test_every_family_builds_a_valid_anonymized_scenario() -> None:
    for family in FAMILIES:
        scenario = build_scenario(family, 8, seed=1)

        assert scenario.id.startswith(f"family-{family}")
        assert all(host.id.startswith("host-") for host in scenario.hosts)
        assert scenario.objectives[0].host_id != scenario.entry_host_ids[0]
        assert scenario.network_edges


def test_family_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown scenario family"):
        build_scenario("hypercube", 8, 1)
    with pytest.raises(ValueError, match="at least three hosts"):
        build_scenario("star", 2, 1)


def test_the_families_are_structurally_distinct() -> None:
    shapes = {family: FamilyShape.measure(family, 8, 1) for family in FAMILIES}

    assert shapes["star"].diameter < shapes["chain"].diameter
    assert shapes["mesh"].edges > shapes["star"].edges
    assert shapes["mesh"].mean_out_degree > shapes["chain"].mean_out_degree
    assert len({shape.edges for shape in shapes.values()}) > 1


def test_every_family_is_solvable() -> None:
    def oracle_factory(family: str, seed: int) -> Agent:
        return ShortestPathOracle(build_scenario(family, 8, seed))

    results = evaluate_families(
        oracle_factory, (0, 1, 2, 3), dynamics=DynamicsConfig.deterministic()
    )

    assert set(results) == set(FAMILIES)
    for family, metric in results.items():
        assert metric.success_rate == 1.0, f"{family} must be solvable"


def test_family_evaluation_shares_one_seed_list() -> None:
    results = evaluate_families(
        greedy_factory, (5, 6), ("star", "ring"), dynamics=DynamicsConfig.deterministic()
    )
    seeds = {tuple(outcome.seed for outcome in metric.outcomes) for metric in results.values()}

    assert list(results) == ["star", "ring"]
    assert len(seeds) == 1

    with pytest.raises(ValueError, match="at least one scenario family"):
        evaluate_families(greedy_factory, (1,), ())
