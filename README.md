# RLAttack

RLAttack is a research platform for learning attack-path exploration policies inside
deterministic, simulated network and vulnerability graphs.

> Status: phase 1 foundation only. Scenario models, Gymnasium environments, agents, and
> training pipelines are intentionally deferred to later milestones.

## Safety boundary

RLAttack does not scan, exploit, authenticate to, or open shells on external systems.
Actions such as `scan_service` and `attempt_simulated_access` are domain labels for local
state transitions, not wrappers around offensive tools. The project will support only:

- deterministic in-process simulators;
- explicitly local cyber-range adapters with no public-network target path; and
- sanitized graph import/export that contains no live target or exploit material.

Nmap, exploit frameworks, remote shells, arbitrary command execution, and public target
adapters are outside the project scope.

## Foundation

- Typed Python package using the `src/` layout
- Core dependency boundaries for Gymnasium, NetworkX, NumPy, and Pydantic
- Optional training dependencies for Stable-Baselines3, PyTorch, and TensorBoard
- Optional FastAPI dashboard dependency group
- Reproducible lint, format, type-check, and test commands
- CI quality gate with read-only repository permissions

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install training or dashboard dependencies only when working on those milestones:

```bash
python -m pip install -e ".[dev,training]"
python -m pip install -e ".[dev,dashboard]"
```

## Running

There is no simulator or training entry point in this foundation release. Run all quality
checks with:

```bash
make check
```

## Architecture

The initial package uses a `src/` layout and keeps tests outside production code. Planned
scenario, simulator, agent, training, evaluation, and explainability layers must remain
independently testable. See [docs/architecture.md](docs/architecture.md).

## Limitations

- No scenario schema or Gymnasium environment exists yet.
- No baseline or learned agent exists yet.
- No training, evaluation, explainability, or integration code exists yet.
- The declared optional dependencies are not exercised until their respective milestones.

## Security policy

The platform is simulation-only. Real-system discovery, credential use, persistence,
defensive-control bypass, malware delivery, destruction, and exfiltration are out of
scope. See [SECURITY.md](SECURITY.md) and [docs/threat-model.md](docs/threat-model.md).

## Roadmap

The next milestone defines the simulated graph scenario schema. See
[docs/roadmap.md](docs/roadmap.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project is licensed under the MIT License.
