# Changelog

All notable changes to RLAttack are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-29

### Changed — research validity

- **Targeted action space.** Actions are now `action_type * target_count + target_index`
  instead of nine parameterless verbs. Previously the environment resolved *which* host,
  service, or vulnerability to act on by scanning its own record order, so path selection
  was hard-coded rather than learned and every policy collapsed onto the same trajectory.
- **Generalization benchmark.** Each benchmark seed now regenerates the scenario. The
  previous runner passed a fixed scenario and ignored the seed, so every "episode" for a
  deterministic policy was a replay of one graph and the reported means had zero variance.
- **Reproducible stochastic dynamics.** Exploitation attempts can fail with a probability
  derived from `Vulnerability.exploitability`, failed attempts raise detection risk, and
  crossing `DynamicsConfig.detection_threshold` ends the episode as a failure. Every draw
  comes from the seeded `np_random` stream, so a seed still fixes the whole trajectory.
  `DynamicsConfig.deterministic()` restores the old always-succeeds behaviour.
- **Pivoting is a real decision.** `discover_host` now only reveals a host; `pivot` makes
  it reachable and requires a credential foothold on the source host. Scanning requires
  reachability, not mere discovery.
- **Episode success is read from state.** `info["objective_captured"]` replaces the
  previous inference from the last action's name.
- Generated scenarios model one credential and access edge per host, so lateral movement
  requires exploiting each host on the route.
- The graph oracle exploits only the credential path of each route host instead of every
  service, which is what now separates it from the greedy baselines.
- `RandomAgent` no longer samples `stop` while progress actions remain available.

### Added

- `rlattack benchmark` — multi-seed generalization benchmark with JSONL/CSV export of
  every episode (`--policy` benchmarks a trained checkpoint alongside the baselines).
- `rlattack train` — CPU training entry point for the optional DQN/PPO pipelines.
- `rlattack.policies.SB3PolicyAgent` / `load_policy` so a trained Stable-Baselines3
  checkpoint can be evaluated against the baselines. Trained policies are reachable only
  from the CLI; the dashboard still accepts no file paths.
- `BenchmarkMetrics` now carries `std_steps`, `std_reward`, a 95% reward confidence
  interval, a detection rate, and the per-episode outcomes.
- Integration tests for seeded-trajectory reproducibility, generator solvability across
  the whole size/difficulty grid, and the CLI → HTML report round trip.
- Optional CPU smoke-training tests plus a scheduled `training-smoke` workflow.
- Explanations and the report trace now show the action's target and outcome.

### Fixed

- Scenarios with no credentials, services, or vulnerabilities no longer crash the
  observation space (`MultiBinary(0)`).
- The dashboard API bounds `step_budget` and `benchmark_episodes`, so a loopback request
  can no longer schedule unbounded computation.
- Removed the unreachable duplicate-action fallbacks in the environment, which existed
  only to be excluded from coverage.

### Infrastructure

- CI runs the full gate on Python 3.10–3.13 instead of 3.12 alone.
- Added `CHANGELOG.md` and `CITATION.cff`.

## [0.2.0] - 2026-07-30

- Simulation observatory release: scenario schema, Gymnasium environment, baselines,
  optional DQN/PPO pipelines, reward strategies, explainability, sanitized ThreatGraph
  adapter, CLI, self-contained HTML report, and loopback-only dashboard.
