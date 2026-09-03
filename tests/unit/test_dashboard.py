from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlattack.dashboard import create_app, run_dashboard


def test_dashboard_routes_render_and_run_experiments() -> None:
    client = TestClient(create_app())

    home = client.get("/")
    assert home.status_code == 200
    assert "시뮬레이션 관측소" in home.text
    assert "window.__RLATTACK_API__=true" in home.text

    health = client.get("/health")
    assert health.json() == {"status": "ok", "mode": "simulation-only"}

    experiment = client.get(
        "/api/experiment",
        params={
            "size": "small",
            "difficulty": "easy",
            "seed": 2,
            "agent": "shortest-path",
            "reward_strategy": "shaped",
            "step_budget": 32,
            "benchmark_episodes": 1,
        },
    )
    assert experiment.status_code == 200
    assert experiment.json()["episode"]["success"] is True


def test_dashboard_rejects_invalid_experiment_and_bind_options() -> None:
    client = TestClient(create_app())

    assert client.get("/api/experiment", params={"size": "tiny"}).status_code == 422
    with pytest.raises(ValueError, match="loopback"):
        run_dashboard(host="0.0.0.0")  # noqa: S104 - verifies unsafe bind rejection
    with pytest.raises(ValueError, match="port"):
        run_dashboard(port=0)


def test_dashboard_runner_passes_loopback_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(app: Any, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    run_dashboard(host="localhost", port=8123)

    assert calls[0]["host"] == "localhost"
    assert calls[0]["port"] == 8123


def test_dashboard_exposes_the_targeted_defender_conditions() -> None:
    """A condition the dashboard cannot select is a condition it cannot show."""

    client = TestClient(create_app())

    targeted = client.get(
        "/api/experiment",
        params={
            "size": "small",
            "difficulty": "easy",
            "agent": "greedy",
            "benchmark_episodes": 1,
            "defender": "targeted",
            "detection_threshold": 0.4,
        },
    )

    assert targeted.status_code == 200
    data = targeted.json()

    assert data["config"]["detection_threshold"] == 0.4
    assert data["config"]["defender"] == "targeted"
    watched = [node["id"] for node in data["scenario"]["nodes"] if node["monitored"]]
    assert watched, "a targeted defender must report which hosts it is watching"

    # Uniform monitoring has no hosts to name, so none are marked.
    passive = client.get(
        "/api/experiment",
        params={
            "size": "small",
            "difficulty": "easy",
            "agent": "greedy",
            "benchmark_episodes": 1,
        },
    )

    assert not any(node["monitored"] for node in passive.json()["scenario"]["nodes"])

    rejected = client.get("/api/experiment", params={"detection_threshold": 0.0})

    assert rejected.status_code == 422


def test_dashboard_controls_cover_the_v1_conditions() -> None:
    home = TestClient(create_app()).get("/").text

    assert '<option value="targeted">' in home
    assert 'id="threshold"' in home
    assert "detection_threshold:$('threshold').value" in home


def test_dashboard_selects_an_attack_target_inside_the_synthetic_graph() -> None:
    """The target names an objective in the graph, never an external address."""

    client = TestClient(create_app())

    full = client.get(
        "/api/experiment",
        params={"size": "medium", "difficulty": "hard", "agent": "greedy", "benchmark_episodes": 1},
    ).json()
    available = full["target"]["available"]

    assert full["target"]["selected"] == ""
    assert not any(node["target"] for node in full["scenario"]["nodes"])
    assert len(available) > 1, "a hard medium scenario has more than one objective"

    chosen = available[-1]
    focused = client.get(
        "/api/experiment",
        params={
            "size": "medium",
            "difficulty": "hard",
            "agent": "greedy",
            "benchmark_episodes": 1,
            "target": chosen["id"],
        },
    ).json()

    marked = [node["id"] for node in focused["scenario"]["nodes"] if node["target"]]
    objectives = [node["id"] for node in focused["scenario"]["nodes"] if node["objective"]]

    assert focused["target"]["selected"] == chosen["id"]
    assert marked == [chosen["host"]]
    # The win condition narrowed to the one target, so only it counts as an objective.
    assert objectives == [chosen["host"]]

    rejected = client.get("/api/experiment", params={"target": "objective-not-real"})

    assert rejected.status_code == 422


def test_dashboard_controls_include_the_attack_target_selector() -> None:
    home = TestClient(create_app()).get("/").text

    assert 'id="target"' in home
    assert "공격 대상" in home
    assert "target:$('target').value" in home
