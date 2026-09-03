# Published results

All numbers here come from `rlattack` commands in this repository. Every run is seeded,
so they can be reproduced exactly with the commands shown.

Model checkpoints are not committed (`artifacts/` is ignored). Regenerate them with the
training command below.

## How to read these tables

Each section states its own seeds, condition, and scenario class, and they are not the
same across sections. Three rules:

- **Compare within a table, not across them.** Step budgets differ by section (64 for
  the head-to-head, 96 for the families, 102 where a budget was matched to training),
  and `steps` is meaningless across different budgets.
- **A success rate is per condition and per scenario class.** The same policy scores
  96.9% on `medium/hard` under exact discovery and 0% under noisy discovery, and the
  graph oracle scores 100% on a star and 12.5% on a ring. Neither number generalizes.
- **Significance is paired by seed.** Every agent in a table saw the same seed list, so
  the reported differences are paired; a difference against a number from another
  section is not.

## Setup

```bash
python -m pip install -e ".[dev,training]"
rlattack train --algorithm maskable-ppo --curriculum --seed 42 \
  --curriculum-timesteps 400000 --output-dir artifacts/policies/mppo-400k

# the same budget under the noisy-discovery condition
rlattack train --algorithm maskable-ppo --curriculum --seed 42 \
  --curriculum-timesteps 400000 --discovery noisy \
  --output-dir artifacts/policies/mppo-noisy
```

The curriculum is `small/easy → small/hard → medium/medium → medium/hard`, carried as
one policy with `set_env` and `reset_num_timesteps=False`. Two budgets are reported
below, 100k and 400k, because the shorter one is a floor rather than a result.

## Why action masking is not optional

Only 1-2% of the targeted action space is valid in any given state - 4 of 288 at reset
on a small scenario. An unmasked learner spends its exploration budget on invalid
actions.

100k-step **unmasked** curriculum runs, evaluated across all nine scenario classes,
24 seeds each:

| Algorithm | Success | Steps | Reward |
| --- | --- | --- | --- |
| PPO (unmasked), every class | 0.0% | 4.00 ± 0.00 | 1.88 |
| DQN (unmasked), every class | 0.0% | 4.00 ± 0.00 | 1.88 |

Both converged on three actions followed by `stop`, producing an identical episode in
every class - the zero standard deviation is the tell. That is the reason
`maskable-ppo` is the default training algorithm.

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


## Trained policy vs the baselines

`medium/hard`, 32 shared seeds, control condition, paired against the graph oracle.

| Policy | Success | Detected | Steps | Reward | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Random | 50.0% | 50.0% | 50.31 ± 6.50 | -1.72 | [-6.45, 3.02] |
| Greedy | 37.5% | 62.5% | 49.66 ± 6.86 | -6.82 | [-11.52, -2.13] |
| Rule-based | 9.4% | 90.6% | 51.50 ± 6.60 | -17.66 | [-21.09, -14.24] |
| Graph oracle | 96.9% | 3.1% | 27.34 ± 4.27 | 24.11 | [20.27, 27.96] |
| MaskablePPO (100k) | 96.9% | 3.1% | 31.72 ± 3.27 | 25.15 | [20.89, 29.42] |
| **MaskablePPO (400k)** | **96.9%** | **0.0%** | 30.31 ± 4.73 | **26.82** | [23.96, 29.69] |

Paired sign-flip permutation test against the graph oracle, on episode reward:

| Policy | Mean difference | Bootstrap CI | p |
| --- | --- | --- | --- |
| Random | -25.83 | [-29.58, -21.90] | 0.0005 |
| Greedy | -30.94 | [-34.48, -26.98] | 0.0005 |
| Rule-based | -41.78 | [-45.81, -36.54] | 0.0005 |
| MaskablePPO (100k) | +1.04 | [+0.11, +1.83] | 0.0300 |
| MaskablePPO (400k) | **+2.71** | [+1.46, +4.59] | **0.0005** |

The learned policy matches the oracle's success rate and earns significantly more
reward, while taking *more* steps. It is not finding shorter paths - it is finding
quieter ones, which the `risk-aware` reward pays for. At 400k it is never detected at
all. The oracle remains the shortest-path upper bound.

Quadrupling the budget roughly triples the margin over the oracle and takes detection
from 3.1% to 0%, which is the concrete reason the 100k figures should be read as a
floor.

## Trained policy transfer

MaskablePPO (400k), 24 shared seeds, control condition. The curriculum trained on
`small/easy → small/hard → medium/medium → medium/hard`; the three `large` rows are
**held out** - the policy never saw that class.

| Scenario class | Success | Detected | Steps | Reward | Oracle success |
| --- | --- | --- | --- | --- | --- |
| small/easy | 95.8% | 0.0% | 17.96 ± 2.84 | 17.60 | 100.0% |
| small/medium | 95.8% | 0.0% | 16.08 ± 3.59 | 17.05 | 100.0% |
| small/hard | 95.8% | 0.0% | 15.79 ± 3.53 | 26.11 | 100.0% |
| medium/easy | 87.5% | 8.3% | 38.83 ± 6.81 | 11.39 | 87.5% |
| medium/medium | 91.7% | 4.2% | 32.79 ± 7.54 | 13.08 | 91.7% |
| medium/hard | 95.8% | 0.0% | 30.12 ± 5.09 | 26.42 | 91.7% |
| large/easy *(held out)* | 66.7% | 33.3% | 99.88 ± 6.55 | -9.49 | 83.3% |
| large/medium *(held out)* | 70.8% | 29.2% | 88.96 ± 6.79 | -6.18 | 91.7% |
| large/hard *(held out)* | 83.3% | 16.7% | 82.54 ± 6.01 | 11.28 | 100.0% |

The policy matches the oracle on the classes it trained on and degrades on the held-out
`large` ones, where it trails by 17-21 points. That gap is the honest measure of how far
the curriculum generalizes; it is not evidence that the class is unwinnable, since the
oracle scores 83-100% there.

## Training under noisy discovery

Roadmap item 42 asked whether the collapse under noisy discovery is fixed by training
under it. **It is not.** A 400k MaskablePPO curriculum trained with
`--discovery noisy`, evaluated on `medium/hard` over 32 seeds at a matched step budget:

| Policy | Condition | Success | Detected | Steps | Reward |
| --- | --- | --- | --- | --- | --- |
| Graph oracle | passive/noisy | **68.8%** | 31.2% | 33.5 ± 7.4 | 7.37 |
| MaskablePPO (exact-trained) | passive/noisy | 0.0% | 3.1% | 9.6 ± 2.2 | -4.18 |
| MaskablePPO (noisy-trained) | passive/noisy | 0.0% | 0.0% | 19.0 ± 4.0 | **7.45** |
| MaskablePPO (noisy-trained) | passive/exact | 0.0% | 0.0% | 21.1 ± 2.6 | 14.65 |

Training under the condition triples the reward there (-4.18 to 7.45) and removes
detection entirely, but success stays at zero. A trace explains why: the policy learns
the local exploitation chain correctly - scan, enumerate, validate, retry a failed
access, escalate - then probes for neighbours, misses seven times, and **stops**. It
learned to bank the shaped reward and quit rather than persist through probe failures.

The condition is not unwinnable: the oracle scores 68.8% on it. This is an exploration
gap, not an environment defect, and 400k steps did not close it - the training reward
plateaued around 12-15 rather than climbing as the exact-condition run did.

A policy trained under noisy discovery also stops succeeding under *exact* adjacency
(0.0%), so this is not a strictly better policy; it is one adapted to a harder
observation model at the cost of the easier one.

### Giving the agent its own probe memory

The diagnosis above was an exploration gap. Part of it turned out to be an
*observability* gap: which hosts had been probed and missed lived only in the action
mask, and a maskable learner uses the mask to filter its action distribution rather than
as a network input. The policy could not distinguish an exhausted sweep from an
untouched one, which makes stopping early a reasonable thing to do.

Adding a `probed_hosts` observation channel and retraining at the same 400k budget:

| Policy | Condition | Success | Detected | Reward |
| --- | --- | --- | --- | --- |
| Graph oracle | passive/noisy | **68.8%** | 31.2% | 7.37 |
| MaskablePPO, noisy-trained | passive/noisy | 0.0% | 0.0% | 7.45 |
| MaskablePPO, noisy-trained + probe memory | passive/noisy | **6.2%** | 0.0% | 8.51 |

The channel moves success off zero and raises reward, so it was a real defect worth
fixing - but 6.2% against the oracle's 68.8% means **the gap is narrowed, not closed**.
The observability fix was necessary and is not sufficient.

### Reshaping the reward toward pivoting made it worse

The next hypothesis was that the reward itself misprices the work: `new_host` pays 1.0
per discovered host, so probing is worth doing on its own, while pivoting - which
actually advances the attack - pays the same. `pivot-focused` moves the mass across
(discovery 0.2, pivot 2.5, failed probe -0.3). Retrained at 400k under noisy discovery:

| Reward | Success | Steps | Behaviour |
| --- | --- | --- | --- |
| `risk-aware` + probe memory | **6.2%** | 18.3 | probes, then stops |
| `pivot-focused` + probe memory | **0.0%** | 18.1 | probes three times, then stops at step 12 |

It is worse, and the trace says why. **Pivoting requires an already-discovered host.**
A probe lands on an adjacent host perhaps a quarter of the time, so 0.2 per success
against -0.3 per miss is a net loss - the prerequisite became unprofitable, and the
reward it leads to unreachable. Rewarding the goal of a two-step sequence while
penalizing its precondition removes the sequence.

Two hypotheses tested, one small gain and one regression, and the gap to the oracle's
68.8% is still open. What the pair rules out is that this is a simple mispricing: a
shaped reward has to make the *prerequisite* pay for itself, or training has to be long
enough for the value function to carry the pivot reward back to the probe that earns it.
400k steps does not.

## Held-out structural families

The generator emits one shape: a chain with difficulty-dependent shortcuts. `rlattack
families` evaluates on topologies it cannot produce, imported through the same sanitized
path as an external attack graph. 8 hosts, 16 shared seeds, noisy discovery.

| Family | Edges | Diameter | Graph oracle | Greedy |
| --- | --- | --- | --- | --- |
| chain *(in distribution)* | 10 | 5 | 56.2% | 37.5% |
| star *(held out)* | 7 | 2 | **100.0%** | 75.0% |
| tree *(held out)* | 6 | 4 | 93.8% | 68.8% |
| mesh *(held out)* | 18 | 2 | **100.0%** | 81.2% |
| ring *(held out)* | 8 | 4 | **12.5%** | 12.5% |

Structure dominates the result. A star puts every host one hop from the hub, so probing
almost always lands; a ring has no natural entry and its shortest path wraps, so the
oracle's route knowledge buys it nothing and it scores 12.5% - worse than it does on the
shape it was designed for. Every difference against the chain reference is significant.

Two consequences for reading any other number in this document: a single success rate on
the generator's own shape says little about a policy's competence, and the gap between
the oracle and a learned policy should be read per family rather than in aggregate.

Every family is solvable under deterministic dynamics, which a test asserts, so a low
score is the agent's and not the topology's.

## Targeted attention, and the first mixed equilibrium

Up to v0.9 the defender's monitoring was a single scalar: every action raised the same
detection risk wherever it happened. That is why the policy grid had a dominant strategy
(item 53) and why adversarial training had no pressure to learn against (item 52) - if
risk depends only on *what* you do and never on *where*, then doing less is always
better and no two attacker strategies trade off.

v1.0 replaces the scalar with a **conserved attention budget**. A targeted defender
watches `attention_hosts` hosts closely and is correspondingly blind on the rest, with
the blind multiplier *derived* from the focus so that the mean over hosts is exactly 1.
Conservation is the point: without it a defender watching more hosts is simply a
stronger defender, and a grid over allocations would measure how much monitoring there
is rather than where it points. The attacker sees which of its *discovered* hosts are
watched, and `shortest-path-evasive` re-plans its route to prefer unwatched hops.

### The grid mixes on mesh

`--family mesh --family-hosts 8 --detection-threshold 0.4`, 64 shared seeds, full
6 x 9 grid. Attacker payoff is mean episode reward; the row player maximizes.

| Attacker | attention-narrow | attention-broad |
| --- | --- | --- |
| shortest-path | 0.797 | **0.828** |
| shortest-path-evasive | **0.875** | 0.734 |

Neither side has a dominant strategy: evasion beats the narrow defender and loses to
the broad one, so the equilibrium is **mixed for the first time in this project** -
attacker `shortest-path` 82% / `shortest-path-evasive` 18%, defender `attention-narrow`
54% / `attention-broad` 46%, value 0.811.

The defender's entire support is the two attention arms. All seven uniform arms -
`patient`, `balanced`, `trigger-happy`, `revoke-heavy`, `fast`, `harden-only`,
`revoke-only` - carry weight zero, which is direct evidence that the mixing comes from
targeted attention and not from enriching the grid with more arms.

Evasion's advantage against the narrow defender is real, not sampling noise
(mesh, 128 seeds, paired):

| Defender | Direct | Evasive | Gain | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| attention-narrow | 73.4% | 87.5% | **+14.1 pp** | [+7.8, +21.1] | 0.0001 |
| attention-broad | 84.4% | 81.2% | -3.1 pp | [-8.6, +2.3] | 0.395 |
| balanced *(uniform)* | - | - | **+0.000** | - | - |

The uniform row is the control and it is exactly zero on every seed: with no monitoring
to observe, the evasive agent is the plain oracle. That is what rules out the gain being
an artefact of the rerouting code rather than of routing around a watcher.

### Two conditions are required, and each alone is not enough

Solving the same grid on every family at the same threshold:

| Family | Node-disjoint routes (mean of 16 seeds) | Equilibrium |
| --- | --- | --- |
| chain | 1.06 | pure (`shortest-path` / `revoke-only`) |
| star | 1.00 | pure (`shortest-path` / `attention-narrow`) |
| tree | 1.00 | pure (`shortest-path` / `attention-broad`) |
| ring | 1.00 | pure (`random` / `trigger-happy`, value 0.000) |
| **mesh** | **2.12** | **mixed** |

**Route diversity** is the discriminating variable, and `rlattack families` now reports
it per instance (`routes=`). Mesh is the only family that reliably offers more than one
node-disjoint route to the deepest host, and it is the only one that mixes. This is not a tuning accident: with
a single route there is nowhere else to go, so a watcher cannot be evaded however the
attention budget is shaped. It also explains why the generator's own shape never mixed -
it averages 1.50 disjoint routes, and in 13 of 16 seeds the host the defender watches is
the entry host or an objective host, both of which every route must contain.

**A binding detection threshold** is the second requirement. At the published default of
0.9 the graph oracle is detected in 2 of 32 episodes and accumulates a mean risk of 0.48,
so detection is simply not the constraint that decides an episode and nothing that
re-prices risk can change an outcome. `--detection-threshold` is therefore a first-class
experimental condition rather than a constant. The window is narrow: at 0.25 the grid
collapses back to pure, because evasion starts helping against both defenders at once.

Read together with the held-out family results above, both of v0.9's negative results
have the same explanation - **structure, not policy richness, decided them**.

### Reproducing

```bash
rlattack equilibrium --family mesh --family-hosts 8 \
  --detection-threshold 0.4 --episodes 64
rlattack equilibrium --family chain --detection-threshold 0.4 --episodes 64
rlattack families --agent shortest-path --episodes 8   # reports routes= per family
```

### What this does not show

The monitoring channel is generous - it reports watched hosts exactly for every
discovered host, so the evasion numbers are an optimistic bound on what an attacker who
has to fingerprint monitoring could achieve (roadmap item 60). The hand-written evasive
oracle shows the strategy pays; whether a *learner* finds it is the next section.

## Adversarial training in the live condition (item 58)

Item 52 asked whether training against a defender that learns alongside the attacker
produces a more robust policy, and answered no - but with the caveat that the defender
axis was inert, so there was no pressure to be robust to. Item 55 built an axis that is
not inert. This re-runs the question in that condition: MaskablePPO on a mesh curriculum
(5, 6, 8, 10 hosts, 300k steps) under a targeted defender with the detection threshold at
0.4, trained either against the **fixed** targeted defender or against a **contextual**
one that re-aims each episode (`--adversarial`). Three training seeds per arm, each swept
over the attention grid on 64 held-out mesh scenarios.

| Condition | Fixed | Adversarial | Difference | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| passive/exact | 47.4% | 55.2% | +7.8 pp | [-0.5, +16.1] | 0.058 |
| adaptive/exact | 43.2% | 51.0% | +7.8 pp | [+0.0, +15.6] | 0.051 |
| **targeted/exact** | **77.1%** | **46.4%** | **-30.7 pp** | [-40.1, -21.4] | **0.0001** |
| targeted/noisy | 6.8% | 0.0% | -6.8 pp | [-10.4, -3.6] | 0.0008 |

The answer is item 52's, sharpened: **adversarial training does not help - against the
targeted defender it significantly hurts**, and the −30.7 pp gap holds in all three
training seeds individually (fixed 76.6 / 75.0 / 79.7 vs adversarial 40.6 / 51.6 / 46.9).
The defender axis is no longer the excuse; the axis is live and the adversarial policy is
worse on it.

### Why - the fixed policy exploits a posture the adversarial one is denied

The striking number is that the **fixed** policy scores *higher* against the targeted
defender than against no defender at all (77.1% vs 47.4%). Attention is conserved, so the
six hosts the defender is not watching are *cheaper* than they are under uniform
monitoring, and a policy that has learned the defender's standing watchlist routes its
risky actions through those blind hosts. Measured, it does exactly that: 11.4% of its
risk-bearing actions land on a watched host, against a 23.0% chance rate and matching the
hand-written evasive oracle's ~10% - so **a learner does find the evasion strategy**
(a first, partial answer to item 59), when there is a stable posture to find.

The adversarial policy trains against a defender that re-aims every episode, so there is
no stable blindness to learn. It puts 27.3% of its risky actions on watched hosts - above
chance - and instead of routing around, it falls back to doing less. That buys robustness
to re-aiming that costs more than the exploit it forfeits. The value of a *learning*
defender, then, is not that it responds harder but that it **denies the attacker a fixed
target to route around** - which is the one thing the whole targeted-attention mechanism
depends on.

### Reproducing

```bash
# one training seed per arm; repeat with --seed for the dispersion above
rlattack train --algorithm maskable-ppo --curriculum --family mesh \
  --defender targeted --detection-threshold 0.4 --curriculum-timesteps 300000 \
  --output-dir artifacts/policies/mesh-fixed
rlattack train --algorithm maskable-ppo --curriculum --family mesh \
  --defender targeted --detection-threshold 0.4 --adversarial \
  --curriculum-timesteps 300000 --output-dir artifacts/policies/mesh-adversarial
rlattack conditions --policy artifacts/policies/mesh-fixed/final.zip \
  --observation curriculum --family mesh --attention-grid \
  --detection-threshold 0.4 --episodes 64
```

## Robustness to the experimental conditions

`medium/hard`, 32 shared seeds, paired against the control condition.

| Condition | Oracle success | MaskablePPO (100k) | MaskablePPO (400k) |
| --- | --- | --- | --- |
| passive / exact *(control)* | 96.9% | 96.9% | 96.9% |
| adaptive / exact | 96.9% | 93.8% (p = 0.23) | 96.9% (p = 0.29) |
| passive / noisy | 68.8% | **0.0%** | **0.0%** |
| adaptive / noisy | 46.9% | **0.0%** | **0.0%** |

The adaptive defender is not what beats the learned policy - neither budget loses
significantly to it. Noisy discovery is. Both policies trained under exact adjacency,
where the action mask itself reveals which hosts are reachable, so they never learned to
probe; when the observation model shifts they cannot act at all. The 400k policy at
least recognizes the situation and stops after 9.5 steps instead of sweeping itself into
detection.

The graph oracle degrades gracefully over the same grid because it carries privileged
route knowledge that no observation model can take away. That is a property of the
baseline, not an achievement of it.

This is the sharpest limitation in this document: **the published policies are only
valid under the conditions they trained on.**

## Training against a learning defender did not help

> Superseded by [item 58](#adversarial-training-in-the-live-condition-item-58): this
> section is the original v0.8 result under a defender axis that turned out to be inert.
> Item 58 re-runs it in the live condition and finds adversarial training does not just
> fail to help - it hurts. This is kept as the record of how the question first read.

Roadmap item 52 asked whether a policy trained against a defender that learns
alongside it transfers better than one trained against a fixed condition. Two
MaskablePPO policies, identical except for that one flag:

- `mppo-adv` — 400k steps, staged curriculum, `--adversarial` (a `ContextualDefender`
  learns during training), exact discovery.
- `mppo-control` — the same 400k staged curriculum against the passive defender.

Both were then evaluated on the same condition grid: `medium/hard`, 32 shared seeds,
`--step-budget 102`, curriculum observations. Rewards are paired by seed; positive
`diff` favours the adversarially trained policy.

| Condition | adv reward | control reward | diff | 95% CI | p | success adv / control |
| --- | --- | --- | --- | --- | --- | --- |
| passive / exact | 24.32 | 26.82 | −2.50 | [−5.26, −0.21] | 0.069 | 84.4% / 96.9% |
| adaptive / exact | 24.20 | 26.07 | −1.87 | [−4.77, +0.66] | 0.188 | 84.4% / 93.8% |
| passive / noisy | −11.39 | −11.90 | +0.50 | [−2.02, +3.03] | 0.709 | 6.2% / 0.0% |
| adaptive / noisy | −11.61 | −11.90 | +0.28 | [−2.22, +2.78] | 0.831 | 6.2% / 0.0% |

**The answer is no.** No condition favours adversarial training significantly. The two
exact conditions trend *against* it, and the apparent edge under noisy discovery is
2 episodes out of 32 versus 0 — well inside noise (p ≈ 0.71) — bought at a 15.6%
detection rate where the control policy is detected 0% of the time.

The reason is visible inside each policy's own grid: the defender axis is nearly inert.
Comparing `adaptive/exact` against `passive/exact` *within* a policy is insignificant
for both (adv p = 0.515, control p = 0.228). Training against a learning defender cannot
build robustness to pressure that is not there, so the only thing the adversarial run
bought was a training distribution slightly further from the evaluation one.

This is the same root cause as item 53's missing mixed equilibrium: detection risk is a
single scalar that penalizes all activity uniformly, so the defender has no lever that
some attacker strategies feel more than others. Item 55 — targeted per-host defender
attention — is the prerequisite for both, and neither question is worth re-running
before it lands.

## Reproducing

```bash
# transfer table
rlattack transfer --policy artifacts/policies/mppo-400k/final.zip \
  --policy-algorithm maskable-ppo --episodes 24 --resamples 1000 \
  --output artifacts/transfer.jsonl --report artifacts/transfer.html

# robustness to the experimental conditions
rlattack conditions --size medium --difficulty hard --episodes 32 \
  --observation curriculum --resamples 1000 \
  --policy artifacts/policies/mppo-400k/final.zip --output artifacts/conditions.jsonl

# head-to-head against the baselines
rlattack benchmark --size medium --difficulty hard --episodes 32 \
  --observation curriculum --resamples 2000 --compare-to shortest-path \
  --policy artifacts/policies/mppo-400k/final.zip --policy-algorithm maskable-ppo \
  --output artifacts/benchmark.jsonl
```

## Transfer report

`rlattack transfer --report` writes a self-contained table:

![Transfer report](assets/transfer.png)

## Limitations

- 400k timesteps is still small. The 100k-to-400k jump improved every headline number,
  so the curve had not flattened when these runs stopped.
- Published policies score 0% under noisy discovery, and training under that condition
  does not fix it (see above). The oracle reaches 68.8% there, so the gap is
  exploration, not the environment.
- No hyperparameter search was run for the published policies; `rlattack sweep` exists
  for it, but the reported runs use the defaults.
- The step budget scales with scenario size, so `steps` is comparable within a class but
  not across classes.
