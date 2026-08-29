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

TRANSFER_PLACEHOLDER

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
