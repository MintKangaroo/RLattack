# Changelog

All notable changes to RLAttack are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Topology-family training and sweeps.** `rlattack train --family <name>` trains the
  curriculum on a topology family (growing the host count across stages) instead of the
  generator's single chain shape, and `rlattack conditions --family --attention-grid`
  sweeps a policy across the passive/adaptive/targeted grid on that family. All training
  conditions now flow from one `ExperimentConfig`, and `--detection-threshold`,
  `--defender targeted`, and the monitoring channel reach the training environment - a
  test asserts this on the built environment, not the banner.
- **Attack-target selection in the dashboard.** A control narrows the episode's win
  condition to one objective inside the synthetic graph, marked on the topology; there is
  no field for an external address, by design.

### Reported

- **Adversarial training does not help against the targeted defender - it hurts**
  (item 58, re-running item 52 in the live condition item 55 built). On mesh with the
  threshold at 0.4, a MaskablePPO policy trained against a *fixed* targeted defender
  scores 77.1% on targeted/exact; trained against a *re-aiming* one it scores 46.4%
  (-30.7 pp, p=0.0001, consistent across three training seeds). The fixed policy learns
  the defender's standing attention posture and routes through the blind hosts - 11.4% of
  its risky actions land on a watched host against a 23.0% chance rate, matching the
  hand-written evasive oracle and giving a **first partial answer to item 59: a learner
  does find the evasion strategy** when the posture is stable. The adversarial policy,
  facing no stable posture, does less instead (27.3% on watched, above chance). Item 52's
  negative result reproduces with a live defender axis, with a mechanism.

## [1.0.0] - 2026-09-02

### Added

- **Targeted defender attention.** `DefenderConfig.attention_hosts` concentrates
  monitoring on a few hosts instead of spreading it uniformly, and detection risk is now
  attributed to the host an action lands on. The blind multiplier is *derived* from the
  focus so the mean over hosts is exactly 1, which keeps the arms differing in the shape
  of the monitoring budget rather than its total - without conservation, a defender
  watching more hosts is just a stronger defender. For a targeted defender, hardening
  re-aims the watchers onto the ground the attacker already holds rather than raising
  sensitivity everywhere.
- **`shortest-path-evasive` attacker** and the `monitored_hosts` observation channel
  (`ObservationConfig.expose_monitoring`), which reports watched hosts among those the
  agent has discovered. The oracle re-plans its route to prefer unwatched hops, pricing
  a watched hop at `evasion_penalty` rather than forbidding it.
- Defender arms `attention-narrow` and `attention-broad`, and the `targeted` defender
  mode for `--defender`.
- **`--detection-threshold`** as a first-class experimental condition, and `--family` /
  `--family-hosts` on `rlattack equilibrium` so the policy grid can be solved on a
  held-out topology.
- `rlattack families` reports **`routes=`**, the number of node-disjoint routes from the
  entry to the deepest host - the structural variable that decides whether monitoring
  can be routed around at all.

### Reported

- **The policy grid mixes for the first time**, on `mesh` with a binding detection
  threshold: attacker `shortest-path` 82% / `shortest-path-evasive` 18% against defender
  `attention-narrow` 54% / `attention-broad` 46%, value 0.811 over 64 seeds. The
  defender's entire support is the two attention arms; all seven uniform arms carry
  weight zero. Evasion gains **+14.1 pp** against the narrow defender (95% CI
  [+7.8, +21.1], p=0.0001, 128 paired seeds) and nothing against the broad one
  (-3.1 pp, p=0.395). Against a uniform defender the gain is exactly zero on every
  seed, which is the control.
- **Two conditions are both required, and each alone leaves the grid pure.** Route
  diversity: `mesh` averages 2.12 node-disjoint routes and is the only family that
  mixes; chain, star, tree, and ring average ~1.00 and all stay pure. A binding
  threshold: at the 0.9 default the oracle is detected in 2 of 32 episodes, so nothing
  that re-prices risk can change an outcome. The window is narrow - at 0.25 the grid
  collapses back to pure.
- This resolves the root cause behind both of v0.9's negative results (items 52 and 53),
  which were blocked on a defender axis that could not be traded off against. **Structure,
  not policy richness, decided them.**

### Changed

- `ExperimentConfig.dynamics()` now honours `detection_threshold`; the default is
  unchanged at 0.9, so every previously published number stands.
- `ShortestPathOracle` plans segments under monitoring weights but orders objectives by
  unweighted depth. Re-ordering by weighted distance can put the deep objective first on
  a directed graph and leave no path back to the shallow one.

### Known limitations

- The evasive attacker is a hand-written oracle reading the monitoring channel, not a
  learned policy: it shows the strategy pays, not that a learner finds it.
- The monitoring channel reports watched hosts exactly for every discovered host, so the
  evasion numbers are an optimistic bound on what an attacker who has to fingerprint
  monitoring could achieve.

## [0.9.0] - 2026-08-30

### Added

- **`rlattack families`** evaluates on star, tree, mesh, and ring topologies the
  generator cannot produce, imported through the same sanitized path as an external
  attack graph. Structure dominates the result: under noisy discovery the graph oracle
  scores 100% on star and mesh, 56.2% on the generator's own chain, and **12.5% on a
  ring** - worse on a held-out shape than on the one it was designed for. A single
  success rate on the generator's shape says little about a policy.
- **`pivot-focused` reward strategy**, moving reward mass from discovery (1.0 to 0.2) to
  pivoting (1.0 to 2.5). Measured, it is **worse**: 0% success under noisy discovery
  against 6.2% for `risk-aware`. Pivoting requires an already-discovered host, so
  pricing discovery below its expected value makes the prerequisite unprofitable and the
  reward it leads to unreachable. Kept as the recorded counter-example rather than a
  recommended setting.
- Two specializing defender arms (`harden-only`, `revoke-only`) and a broad attacker
  (`shortest-path-broad`) that takes the whole network rather than only its route.
- **`scripts/ramguard.sh`** runs a long training job inside a transient cgroup with a
  hard memory cap and a pre-flight free-memory check, so a runaway run is killed on
  its own instead of pushing a shared machine into swap.

### Reported

- **Training against a learning defender did not transfer better.** Two MaskablePPO
  policies differing only in `--adversarial`, evaluated on the same condition grid with
  32 seeds paired: no condition favours the adversarially trained one significantly, and
  both exact conditions trend against it (passive/exact -2.50 reward, p = 0.069; 84.4%
  success against 96.9%). Its apparent edge under noisy discovery is 2 episodes out of
  32 against 0 (p = 0.71) bought at a 15.6% detection rate against 0%. The reason shows
  inside each policy's own grid: adaptive against passive is insignificant for both
  (p = 0.515 and p = 0.228), so the defender axis is nearly inert and adversarial
  training hardens against pressure that is not there. Same root cause as the missing
  mixed equilibrium below; both wait on targeted defender attention.
- **Enriching the policy grid did not produce a mixed equilibrium.** It stays pure -
  shortest-path against `fast`, value 0.79 - because redundancy is not a trade-off here
  but strictly worse: against `revoke-only` the broad oracle scores 43.8% where the
  focused one scores 75.0%, the extra exploits costing more in detection than the spare
  credentials are worth. Detection risk is a single scalar penalizing all activity
  uniformly, so doing less is always better and no attacker strategy trades off against
  another. A mixed equilibrium needs targeted defender attention.

## [0.8.0] - 2026-08-29

### Added

- **`probed_hosts` observation channel.** Which hosts the agent had probed and missed
  lived only in the action mask, and a maskable learner uses the mask to filter its
  action distribution rather than as a network input - so the policy could not tell an
  exhausted sweep from an untouched one. Retrained at the same budget, success under
  noisy discovery moves from 0% to 6.2%. The graph oracle reaches 68.8% there, so the
  gap is narrowed, not closed. **This changes the observation space**; policies trained
  before this release cannot be loaded against it.
- **`rlattack equilibrium`** solves the attacker x defender policy grid as a matrix game
  by fictitious play, checked against matching pennies and a dominant-row game. On
  medium/hard the equilibrium is pure - shortest-path against `fast`, value 0.75 -
  because shortest-path dominates every other attacker row, which the command reports
  rather than dressing up as strategy.
- **`DefenderConfig.response_budget`** caps how many responses one episode can absorb,
  with the overflow counted rather than silently dropped, and the learned defender reads
  a `budget_pressure` band so it can ration. A budget of 8 beats no budget at all
  (85.8% +/- 2.6% attacker success against 89.8% +/- 4.6%): the constraint forces the
  defender to spend where it matters.
- **`rlattack train --adversarial`** attaches a learning defender to the training
  environments, driven from `StageEnv.reset` since Stable-Baselines3 owns the episode
  loop, so the attacker trains against an opponent that adapts.
- **`docs/importing.md`**, a mapping guide for the import contract with worked
  conversions for MulVAL-style graphs, CyberBattleSim topologies, and edge lists.

### Fixed

- **The sanitizer missed fully qualified names in ordinary fields.** It checked
  forbidden keys and strings containing a URL or dotted-quad address, so
  `fact="hostname(webserver.corp.example)"` passed. Names of three or more dotted labels
  are now rejected, while the simulator's own two-label hostnames stay acceptable.

## [0.7.0] - 2026-08-29

### Added

- **`rlattack train --discovery/--defender`** so a policy can be trained under the
  conditions it will be evaluated in. The v0.6 policies scored 0% under noisy discovery
  purely because they had only ever trained under exact adjacency.
- **`BanditAttacker`**, so both sides of `rlattack game` adapt between episodes rather
  than one learner facing a stationary opponent. Against the contextual defender it
  rediscovers shortest-path unaided (232 of 256 pulls), at a lower overall success rate
  than a fixed oracle because it pays to find out which baseline that is.
- **`ContextualDefender`**, a defender policy conditioned on the episode so far - alert
  band, whether there is anything to revoke, and how far into the budget the episode is
  - learned by Monte-Carlo over the episode's terminal reward. Over five seeds of 256
  rounds it holds the graph oracle to 87.7% +/- 3.9% against the bandit's
  92.7% +/- 1.0%, at four times the defender reward.
- **`rlattack import`** converts published attack graphs (GraphML, GML, NetworkX
  node-link JSON) into sanitized scenarios. Node identifiers are replaced with anonymous
  IDs and the payload must pass the ThreatGraph sanitizer, so an import cannot carry a
  hostname or address into a scenario file. `--topology-only` keeps just the structure;
  by default a deterministic exploitation layer is synthesized so an imported topology
  is playable.
- `rlattack.bandit.EpsilonGreedy`, shared by the adaptive attacker and defender.

### Fixed

- **Training conditions never reached the environment.** `--discovery`, `--defender`,
  and `--reward` were parsed and printed but not passed through, so a run advertised as
  noisy trained under exact adjacency. A test now asserts the conditions arrive at the
  environment rather than in the log line.

### Changed

- Curriculum stages sample every earlier stage by default
  (`--forget-previous-stages` restores isolation), the standard guard against
  catastrophic forgetting - Stable-Baselines3 carries its episode buffer across stages,
  so a curriculum's rolling reward would not reveal the loss.
- **Responding now costs the defender.** It was free, which made "respond on every step"
  trivially optimal and the learning problem vacuous. `defender_reward` charges per
  response and charges more for false positives, which the environment already counted.
  `EpisodeOutcome` carries both counters so the cost is computed from measured
  behaviour, and they are included in exports.
- `adapter._assert_sanitized` is now the public `adapter.assert_sanitized`.

## [0.6.0] - 2026-08-29

### Added

- **`rlattack conditions`** evaluates one agent across the defender x discovery grid on
  a shared seed list. Published policies had only ever been reported under the control
  condition. The result matters: the curriculum policy is unaffected by the adaptive
  defender (p = 0.29) but scores **0%** under noisy discovery at both training budgets,
  because it trained under exact adjacency and never learned to probe. The graph oracle
  degrades gracefully over the same grid (96.9% -> 68.8% -> 46.9%) because its route
  knowledge is privileged and no observation model can take it away.
- **`rlattack game`** plays a fixed attacker against `BanditDefender`, which picks a
  response policy per episode and updates it from the outcome - a deliberately simple
  two-player setup. Against the graph oracle over 200 rounds it concentrates on one arm
  and holds the attacker to 92.0%, against 96.5% for the weakest fixed arm; it does not
  beat the best fixed arm chosen with hindsight (91.5%), which is the exploration it
  pays to find the arm unaided.
- **`rlattack sweep`** trains a grid of hyperparameter trials and benchmarks each
  resulting policy against a baseline trial. `PPOTrainingConfig` gains `learning_rate`
  and `entropy_coefficient`, and `--curriculum-timesteps` sets the budget.
- A **400k-timestep** curriculum policy alongside the 100k one. It widens the margin
  over the graph oracle from +1.04 (p = 0.030) to +2.71 (p = 0.0005) and takes
  detection from 3.1% to 0%, which is why the 100k figures are a floor.

### Changed

- Screenshots re-captured against the current UI; they had shown the v0.2 layout.

### Fixed

- **Two responsive defects the screenshots exposed.** The benchmark rows wrapped into
  three cramped lines on a phone once reward confidence intervals were added; they now
  stack, with the interval hidden below 720px. A `white-space:nowrap` tag in a
  non-wrapping panel head pushed the page 41px past a 320px viewport. Verified at
  320/375/414/768/1024/1440/1920 with zero horizontal overflow, and pinned by a unit
  test since CI has no browser.

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
- Published curriculum policies and their results in `docs/results.md`: a MaskablePPO
  curriculum matches the graph oracle's success and detection rate on `medium/hard`
  (96.9% / 3.1%) and earns significantly more reward (+1.04, 95% CI [+0.11, +1.83],
  p = 0.030) by taking quieter rather than shorter paths, and holds 87-100% on the
  classes it trained on while trailing the oracle by 17-25 points on the held-out
  `large` classes.

### Fixed

- **The defender shared the attacker's random stream.** Enabling a defender shifted
  the attacker's draws, so a passive/adaptive comparison on one seed compared two
  different episodes and was not paired at all - in 1 of 12 episodes an entirely inert
  defender changed the outcome. The defender now draws from its own seeded stream, and
  a regression test asserts an inert defender leaves the attacker's episode identical.
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
