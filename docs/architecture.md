# Architecture

## Current state

The foundation consists of a typed Python package under `src/rlattack`, unit tests, and
a CI quality gate. It contains no simulator, agent, or network-capable code.

## Planned package boundaries

- `scenario`: validated graph entities and serialization
- `env`: deterministic Gymnasium state transitions and action masks
- `rewards`: configurable sparse, shaped, risk-aware, and cost-aware rewards
- `baselines`: random, greedy, rule-based, and shortest-path agents
- `training`: DQN/PPO training, checkpoints, and experiment tracking
- `evaluation`: reproducible metrics and unseen-scenario benchmarks
- `explainability`: decisions, paths, values, contributions, and graph overlays
- `adapters`: sanitized file-based imports and explicitly local cyber ranges

Scenario and transition logic form the dependency core. Training and presentation layers
depend inward on that core; the core does not depend on Stable-Baselines3, TensorBoard, or
FastAPI.

## Execution boundary

Simulator actions mutate only an in-memory episode state. Their security terminology does
not imply operating-system or network operations. Adapters must expose the same transition
interface and may not invoke scanners, exploit frameworks, remote shells, or public targets.

## Quality attributes

Runs are seedable, configurations and scenario identifiers are recorded, observations are
machine-checkable, invalid actions are masked and rejected, and CI executes only short
deterministic tests. Long training runs remain separate from CI.
