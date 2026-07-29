from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlattack.dashboard import create_app, run_dashboard


def test_dashboard_routes_render_and_run_experiments() -> None:
    client = TestClient(create_app())

    home = client.get("/")
    assert home.status_code == 200
    assert "Simulation Observatory" in home.text
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
