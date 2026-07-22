# RLAttack

RLAttack is a research platform for learning attack-path exploration policies inside
deterministic, simulated network and vulnerability graphs.

The platform frames attack planning as a reproducible Markov decision process (MDP): an
agent observes a partial cyber-range state, chooses one simulated action, receives a
configurable reward, and continues until the objective is collected, the agent stops, or
the episode budget is exhausted. The same scenario and seed always produce the same
trajectory, which makes algorithm comparisons and ablation studies auditable.

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

## Research questions

The benchmark is intended to answer questions such as:

- How do DQN and PPO compare when exploration, cost, and detection risk compete?
- When do learned policies outperform random, greedy, rule-based, and shortest-path
  baselines?
- Do policies generalize from small training graphs to larger or unseen scenarios?
- Can an action be explained using the observed host, service, vulnerability, privilege,
  reachability, risk, and budget signals that led to its selection?

## Environment contract (planned)

The Gymnasium environment will expose a fixed action catalogue. Every action is a local
state transition in the simulator:

| Action | Simulated effect |
| --- | --- |
| `discover_host` | Reveal a reachable host in the scenario graph |
| `scan_service` | Reveal a host's simulated service and port information |
| `enumerate_service` | Add simulated service details to the observation |
| `validate_vulnerability` | Confirm whether a modeled vulnerability is usable |
| `attempt_simulated_access` | Attempt a modeled access edge and credential transition |
| `escalate_simulated_privilege` | Apply a modeled privilege transition |
| `pivot_simulated_network` | Move reachability across a modeled network edge |
| `collect_simulated_objective` | Complete the objective when its preconditions hold |
| `stop` | End the episode voluntarily |

Observations will represent discovered hosts, known ports and services, vulnerability and
credential state, privilege state, reachability, detection risk, and remaining step budget.
Invalid or out-of-scope actions will be masked and rejected by the environment.

Rewards will be selected through configuration. Candidate strategies include sparse,
shaped, risk-aware, and cost-aware rewards. Positive terms can represent discovery,
validated vulnerabilities, access, privilege escalation, and objective completion; negative
terms can represent duplicate scans, failed actions, noisy actions, detection events, and
step cost. Each run records the selected reward strategy and its values.

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

The `dev` extra is sufficient for the phase 1 quality gate. The `training` extra is
intentionally separate because PyTorch and Stable-Baselines3 are large and are not needed
for scenario or environment development.

## Running

There is no simulator or training entry point in this foundation release. Run all quality
checks with:

```bash
make check
```

When later milestones add entry points, long training runs will remain separate from CI.
Expected experiment artifacts include a seed, scenario identifier, algorithm, reward
strategy, environment version, metrics, checkpoints, and TensorBoard or MLflow metadata.

## Architecture

The initial package uses a `src/` layout and keeps tests outside production code. Planned
scenario, simulator, agent, training, evaluation, and explainability layers must remain
independently testable. See [docs/architecture.md](docs/architecture.md).

The intended flow is:

```text
scenario schema/generator -> deterministic Gymnasium environment -> agent
                                      |                         |
                                      +-> metrics/evaluation <--+
                                      +-> explanations/graph overlays
```

NetworkX models graph relationships, Pydantic validates scenario and experiment data,
Gymnasium defines the environment API, and Stable-Baselines3/PyTorch are used only by the
optional training pipelines. No package boundary is allowed to introduce a live network
client or command runner.

## Reproducible experiments

Use a fixed seed for scenario generation, environment reset, and agent evaluation. A
benchmark report should identify the scenario size (`small`, `medium`, or `large`),
difficulty, seed, algorithm, reward strategy, training budget, and evaluation seeds. Core
metrics are success rate, mean steps, cumulative reward, detection risk, path cost, and
performance on unseen scenarios. CI will run only a short smoke experiment; longer runs
belong in an explicitly invoked research command.

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

The implementation is intentionally staged so every milestone has a focused commit:

1. Project initialization and quality gates (current)
2. Graph scenario schema
3. Gymnasium attack-path environment
4. Deterministic scenario generator
5. Baseline agents
6. DQN training pipeline
7. PPO benchmark pipeline
8. Configurable reward experiments
9. Reproducible evaluation
10. Policy explainability
11. Sanitized ThreatGraph adapter
12. Experimental methodology documentation

See the detailed [roadmap](docs/roadmap.md) for the planned commit sequence.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project is licensed under the MIT License.
