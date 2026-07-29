"""Local-only FastAPI dashboard for simulated RLAttack experiments."""

from __future__ import annotations

from typing import Any, cast

from rlattack import __version__
from rlattack.experiment import AgentName, ExperimentConfig, build_dashboard_data
from rlattack.generator import Difficulty, ScenarioSize
from rlattack.report import render_dashboard
from rlattack.reward import RewardStrategy

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def create_app() -> Any:
    """Create the optional FastAPI application without burdening core imports."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "The dashboard requires optional dependencies: pip install -e '.[dashboard]'"
        ) from error

    app = FastAPI(
        title="RLAttack Simulation Observatory",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        description="Local, simulation-only attack-path experiment dashboard.",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_dashboard(build_dashboard_data(), api_enabled=True)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "simulation-only"}

    @app.get("/api/experiment")
    def experiment(
        size: str = "medium",
        difficulty: str = "hard",
        seed: int = 42,
        agent: str = "greedy",
        reward_strategy: str = "risk-aware",
        step_budget: int = 64,
        benchmark_episodes: int = 8,
    ) -> dict[str, object]:
        try:
            config = ExperimentConfig(
                size=cast(ScenarioSize, size),
                difficulty=cast(Difficulty, difficulty),
                seed=seed,
                agent=cast(AgentName, agent),
                reward_strategy=cast(RewardStrategy, reward_strategy),
                step_budget=step_budget,
                benchmark_episodes=benchmark_episodes,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return build_dashboard_data(config)

    return app


def run_dashboard(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the dashboard on a loopback address only."""

    if host not in _LOOPBACK_HOSTS:
        raise ValueError("dashboard host must be a loopback address")
    if port < 1 or port > 65535:
        raise ValueError("dashboard port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "The dashboard requires optional dependencies: pip install -e '.[dashboard]'"
        ) from error
    uvicorn.run(create_app(), host=host, port=port, log_level="info")  # pragma: no cover
