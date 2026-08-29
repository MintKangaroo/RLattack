# Published results

All numbers here come from `rlattack` commands in this repository. Every run is seeded,
so they can be reproduced exactly with the commands shown.

Model checkpoints are not committed (`artifacts/` is ignored). Regenerate them with the
training command below.

## Setup

```bash
python -m pip install -e ".[dev,training]"
rlattack train --algorithm maskable-ppo --curriculum --seed 42 \
  --output-dir artifacts/policies/mppo
```

The curriculum is `small/easy → small/hard → medium/medium → medium/hard`, 100k
timesteps total, carried as one policy with `set_env` and `reset_num_timesteps=False`.

## Why action masking is not optional

Only 1-2% of the targeted action space is valid in any given state - 4 of 288 at reset
on a small scenario. An unmasked learner spends its exploration budget on invalid
actions.

A 100k-step **unmasked** PPO curriculum, evaluated across all nine scenario classes:

| Scenario class | Success | Steps | Reward |
| --- | --- | --- | --- |
| every class | 0.0% | 4.00 ± 0.00 | 1.88 |

The policy converged on three actions followed by `stop`, producing an identical episode
in every class. That is the reason `maskable-ppo` is the default training algorithm.

## Baseline transfer table

Graph oracle, 24 shared seeds, control condition (passive defender, exact discovery,
`risk-aware` reward). Step budgets scale with scenario size, so `steps` is comparable
within a class but not across them.

| Scenario class | Success | Detected | Steps | Reward |
| --- | --- | --- | --- | --- |
| small/easy | 100.0% | 0.0% | 17.1 ± 3.0 | 17.42 |
| small/medium | 100.0% | 0.0% | 14.7 ± 4.0 | 16.60 |
| small/hard | 100.0% | 0.0% | 18.4 ± 3.3 | 26.07 |
| medium/easy | 87.5% | 12.5% | 38.5 ± 3.4 | 11.17 |
| medium/medium | 91.7% | 8.3% | 30.2 ± 6.1 | 11.62 |
| medium/hard | 91.7% | 8.3% | 28.3 ± 4.9 | 22.56 |
| large/easy | 83.3% | 16.7% | 82.2 ± 4.7 | 9.22 |
| large/medium | 91.7% | 8.3% | 62.6 ± 7.3 | 13.53 |
| large/hard | 100.0% | 0.0% | 55.5 ± 6.8 | 29.15 |

```bash
rlattack transfer --agent shortest-path --episodes 24 --resamples 1000 \
  --output artifacts/oracle-transfer.jsonl
```

`easy` classes are not the easiest: without shortcut edges the route is longer, so the
attacker spends more actions and accumulates more detection risk.

### Why detection risk is normalized by network size

With `normalize_risk_by_size=False` the threshold is an absolute budget of noisy
actions, and the same correctly-built oracle scores:

| Condition | small | medium | large |
| --- | --- | --- | --- |
| normalized (default) | 100% | 92-100% | 88-100% |
| absolute | 100% | 92-100% | 0-50% |

Large scenarios become unreachable purely because their routes are longer, which a
transfer table would report as a generalization failure.


## Reproducing

```bash
rlattack transfer --policy artifacts/policies/mppo/final.zip \
  --policy-algorithm maskable-ppo --observation curriculum \
  --episodes 24 --resamples 1000 \
  --output artifacts/transfer.jsonl --report artifacts/transfer.html
```

## Limitations

- 100k timesteps is a floor, not a ceiling; no hyperparameter search was run.
- Published policies are evaluated under the **control** condition (passive defender,
  exact discovery). Evaluating them under the adaptive defender and noisy discovery
  conditions is roadmap item 40.
- The step budget scales with scenario size, so `steps` is comparable within a class but
  not across classes.
