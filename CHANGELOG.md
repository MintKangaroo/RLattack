# Changelog

All notable changes to RLAttack are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-08-29

### Added

- **Action masking in training** (`maskable-ppo`, via `sb3-contrib`). The environment
  already published an action mask; training now uses it. Without it only 1-2% of the
  targeted action space is valid in any state (4 of 288 at reset on a small scenario),
  and a 100k-step unmasked PPO curriculum converged on a degenerate policy - three
  actions then `stop`, 0% success and an identical 4-step episode in all nine scenario
  classes. `action_masks()` exposes the mask as booleans and `StageEnv` delegates it.
  Forwarding the mask at evaluation is part of a maskable policy's interface, not a
  correction; unmasked `dqn`/`ppo` checkpoints are still evaluated without it.
- **Defender response latency and noisy telemetry.** The defender previously read the
  attacker's exact risk and acted the instant it crossed the threshold. It now reads a
  noisy estimate (`observation_noise`) and its decision lands `response_latency` steps
  later, so evasion is a timing problem. False positives are counted in `info`.
- **Noisy neighbour discovery** (`DynamicsConfig.noisy_discovery`). With exact adjacency
  the action mask *was* the topology - it offered `discover_host` for precisely the
  adjacent hosts. Under noisy discovery any undiscovered host can be probed, only
  adjacent ones can succeed, and probing re-opens once every candidate has been missed
  (without which a chain topology deadlocks).
- **Condition strip and defender tile** in the dashboard report, plus defender and
  discovery controls, and reward confidence intervals in the baseline rows. A report
  now states the conditions it ran under.
- **`rlattack transfer --report`** writes a self-contained transfer table with paired
  significance tests against a reference class.
- Published curriculum policies and their transfer table (see README).

### Fixed

- **Transfer baselines were built from the wrong scenario.** `evaluate_transfer` took a
  seed-only agent factory and the CLI built the baseline from the *configured* size and
  difficulty, so a graph-aware baseline like `ShortestPathOracle` was constructed from
  one scenario class and then run against the other eight, with a route and indices
  belonging to a different graph. It still acted, via its greedy fallback, so the table
  looked plausible while eight of nine rows measured a mis-indexed oracle. Corrected,
  the oracle scores 83-100% across all nine classes rather than 18-38% on the large
  ones, so the "generalization gradient" reported for 0.4.0 was mostly this bug. The
  factory now receives the stage it will act in.

### Known gaps

- The dashboard screenshots still show the v0.2 UI; re-capturing them needs a browser.
- Learned policies are evaluated under the control condition only.

## [0.4.0] - 2026-08-29

### Added

- **Partial observability** (`ObservationConfig`). The agent observes a quantized
  `alert_level` one-hot instead of the defender's exact suspicion score; the exact risk
  stays in `info` for reporting, and `expose_exact_risk` puts it back for analysis runs.
- **Fixed interface capacities.** `ObservationConfig.for_curriculum()` pads every
  channel to a width that covers all generated classes, so the observation and action
  spaces are identical for `small` and `large`. Previously the vector length told the
  agent the network size before it discovered anything, and a policy trained on one
  size could not be applied to another. Padding is inert - an integration test asserts
  the trace, steps, and reward are unchanged.
- **An adaptive defender** (`rlattack.defender`). Once accumulated risk crosses its
  alert threshold, it hardens monitoring on every host the attacker has reached and can
  revoke an acquired credential, forcing the attacker to re-earn that foothold.
  `decide_response` is a pure function of a small `DefenderState`. Off by default: a
  passive run is the control condition, `--defender adaptive` the treatment.
- **Multi-objective episodes.** The environment tracks `collected_objectives` and ends
  only once every objective is held; `hard` scenarios gain a second objective on a
  mid-route host. The graph oracle plans a route chaining shortest paths through every
  objective host, shallowest first.
- **Paired significance testing** (`rlattack.stats`). Benchmarks pair episodes by seed,
  so a sign-flip permutation test and a percentile bootstrap of the paired difference
  apply directly - no normality assumption and no SciPy. Reported by
  `rlattack benchmark --compare-to`, `rlattack ablation`, and `rlattack transfer`.
- **Reward ablation** (`rlattack ablation`) holding scenario, seeds, dynamics, and
  defender fixed so only the reward varies.
- **Curriculum and transfer** (`rlattack.curriculum`). `rlattack transfer` evaluates one
  policy over all nine size/difficulty classes on a shared seed list;
  `rlattack train --curriculum` carries one policy across stages. `StageEnv` draws a
  fresh scenario on every reset, so a stage teaches its class rather than one graph.

### Changed

- `steps_remaining` in the observation became `budget_fraction` in [0, 1]. As a Box
  bounded by the step budget it broke the shared interface, because the budget scales
  with scenario size. The absolute remaining steps moved to `info`.
- Detection risk is normalized by network size (`normalize_risk_by_size`, on by
  default). As an absolute budget of noisy actions it made `large` scenarios unwinnable
  purely because they take more steps. Re-measured in 0.5.0 after the transfer-baseline
  fix: with absolute risk the graph oracle scores 0% on `large/easy` and 0-50% across
  the large classes; normalized it scores 88-100%. (The "19-38%" figure reported when
  0.4.0 was written was measured with the mis-indexed transfer baseline fixed in
  0.5.0.)
- Curriculum stages scale the step budget with scenario size, so a transfer table does
  not report budget exhaustion as a generalization failure.

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
