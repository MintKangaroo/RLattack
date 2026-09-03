# Roadmap

1. Initialize the Python research environment and quality gates. (completed)
2. Define the simulated network graph scenario schema. (completed)
3. Implement the Gymnasium attack-path environment. (completed)
4. Add deterministic small, medium, and large scenario generators. (completed)
5. Add random, greedy, shortest-path, and rule-based baseline agents. (completed)
6. Add the DQN training pipeline. (completed)
7. Add the PPO benchmark pipeline. (completed)
8. Add configurable reward strategies and experiment records. (completed)
9. Add reproducible evaluation and generalization benchmarks. (completed)
10. Add policy and graph explainability outputs. (completed)
11. Add a sanitized ThreatGraph scenario adapter. (completed)
12. Document the experimental methodology, limitations, ethics, and safety scope. (completed)
13. Add a shared explainable experiment runner and weighted path metrics. (completed)
14. Add a CLI and portable self-contained HTML reports. (completed)
15. Add a loopback-only interactive dashboard and experiment API. (completed)
16. Verify desktop/mobile rendering and publish real dashboard screenshots. (completed)
17. Make sanitized graph export fully anonymous and structure-preserving. (completed)
18. Add dependency auditing and automated dependency update monitoring. (completed)
19. Verify monitored DQN and PPO CPU training and final checkpoints. (completed)
20. Promote the validated v0.2 observatory to the GitHub default branch with release documentation. (completed)

## v0.3 — research validity

21. Replace the parameterless action catalogue with a targeted action space so that path
    selection is a policy decision rather than an environment implementation detail.
    (completed)
22. Regenerate the scenario for every benchmark seed so the benchmark measures
    generalization instead of replaying one fixed graph. (completed)
23. Add reproducible stochastic exploitation outcomes and detection-threshold
    termination so risk-aware rewards affect the dynamics. (completed)
24. Separate discovery from pivoting and require a credential foothold to move
    laterally. (completed)
25. Expose trained Stable-Baselines3 checkpoints through the shared Agent protocol and
    benchmark them from the CLI. (completed)
26. Report dispersion, confidence intervals, and detection rates, and export every
    episode as JSONL/CSV. (completed)
27. Add reproducibility, solvability, and CLI round-trip integration tests, and remove
    the unreachable environment fallbacks. (completed)
28. Run the quality gate on Python 3.10-3.13 and add a scheduled CPU training smoke
    workflow. (completed)

## v0.4 - research protocol

29. Restrict the observation to what an attacker could know: a quantized alert level
    instead of the exact detection risk, and fixed capacities so the vector length no
    longer reveals the network size. (completed)
30. Add an adaptive defender that hardens monitoring and revokes credentials in
    response to the attacker's accumulated risk. (completed)
31. Support multi-objective episodes, and report reward ablations and benchmark
    comparisons with paired significance tests. (completed)
32. Add a scenario curriculum and a transfer evaluation across every size and
    difficulty class. (completed)

## v0.5 - closing the loop

33. Train curriculum policies and report their transfer table against the baselines
    with significance tests. (completed)
34. Surface the defender and transfer views in the reports. (completed; the dashboard
    screenshots still need a browser and are not re-captured)
35. Model defender response latency and noisy telemetry, so evasion is a timing problem
    rather than a threshold problem. (completed)
36. Reveal neighbours through a noisy scan model rather than an exact adjacency check.
    (completed)
37. Train against the environment's action mask, without which only 1-2% of the action
    space is valid and learning collapses onto stopping immediately. (completed)

## v0.6 - closing the measurement loop

38. Make the training budget and hyperparameters explicit and searchable, and publish a
    longer run. (completed)
39. Re-capture the dashboard and transfer screenshots against the current UI.
    (completed)
40. Evaluate learned policies across the condition grid, not only the control.
    (completed)
41. A defender that adapts its own policy, turning the setup into a two-player game.
    (completed)

## v0.7 - two learners and external graphs

42. Train under the conditions a policy will be evaluated in, and publish a policy
    trained under noisy discovery. (completed; the policy improves reward and avoids
    detection under that condition but still scores 0% success - see item 50)
43. Let the attacker adapt between episodes, so the game has two learners. (completed)
44. Condition the defender's policy on the episode so far instead of committing to one
    configuration per episode, and charge for responding. (completed)
45. Import published attack graphs as sanitized scenarios. (completed)

## v0.8 - both sides learn

46. A published-dataset mapping guide for the import contract. (completed)
47. Train the attacker against a defender that learns alongside it. (completed)
48. Report equilibria over the attacker x defender policy grid. (completed)
49. Cost-model the defender's responses against an operational budget. (completed)
50. Close the exploration gap under noisy discovery. (**partial**: the agent could not
    observe which hosts it had probed, and giving it that memory moved success from 0%
    to 6.2% - against the oracle's 68.8%, so the gap is narrowed, not closed)

## v0.9 - held-out structure and strategic play

51. Finish item 50 - close the remaining exploration gap under noisy discovery.
    (**open**: two hypotheses tested. Probe memory gained 6.2 points; reshaping the
    reward toward pivoting lost them again, because pivoting needs an already-discovered
    host and pricing discovery below its expected value makes the prerequisite
    unprofitable. Next: price discovery against the probe success rate, or train long
    enough for the value function to carry the pivot reward back to the probe.)
52. Report whether a policy trained against a learning defender transfers better than
    one trained against a fixed condition. (**answered negatively**: no condition on the
    grid favours the adversarially trained policy significantly, and the two exact
    conditions trend against it. The defender axis is nearly inert - adaptive vs passive
    is insignificant *within* each policy - so adversarial training builds robustness to
    pressure that is not there. Same root cause as item 53; see items 55 and 58)
53. Enrich the policy grid until the equilibrium is mixed. (**answered negatively**:
    enriching it did not produce mixing, and the reason is that detection risk is a
    single scalar penalizing all activity uniformly, so no attacker strategy trades off
    against another - see item 55, which reproduces the mixing this item could not by
    replacing the single scalar with per-host attention)
54. A held-out scenario family the generator cannot produce. (completed: star, tree,
    mesh, and ring topologies, which separate agents sharply - the graph oracle scores
    100% on star and 12.5% on ring under noisy discovery)

## v1.0 - targeted attention and the first mixed equilibrium

55. Give the defender targeted attention - per-host monitoring an attacker can route
    around - so that evading one defender exposes you to another. (**completed, and it
    unblocks items 52 and 53**: with a conserved attention budget, an evasive attacker
    arm, and detection made binding, the attacker x defender grid mixes for the first
    time - attacker `shortest-path` 82% / `shortest-path-evasive` 18% against defender
    `attention-narrow` 54% / `attention-broad` 46%. Two conditions are *both* required,
    and each alone leaves the grid pure: **route diversity**, because routing around a
    watcher needs somewhere else to route - `mesh` is the only family with more than
    one node-disjoint route and the only one that mixes - and a **binding detection
    threshold**, because at the 0.9 default the oracle is detected in ~5% of episodes
    and nothing that re-prices risk can change an outcome.)

58. Re-run item 52 (adversarial training) in the condition that item 55 found: mesh
    structure with a binding threshold. (**answered, and it sharpens item 52**:
    adversarial training does not help - against the targeted defender it *hurts*,
    77.1% -> 46.4% on targeted/exact, -30.7 pp, p=0.0001, consistent across three
    training seeds. The defender axis is no longer inert, so the excuse from item 52 is
    gone; the reason is that the fixed-trained policy learns the defender's standing
    attention posture and routes through the blind hosts - scoring *higher* against the
    targeted defender than against none - while the adversarial policy trains against a
    re-aiming defender with no stable posture to learn and falls back to doing less. A
    learning defender's value is that it denies the attacker a fixed target.)
59. Train a policy that learns to route around monitoring, rather than reading the
    monitoring channel through a hand-written oracle. (**answered partially by item 58**:
    the fixed-trained MaskablePPO puts only 11.4% of its risky actions on watched hosts,
    against a 23.0% chance rate and matching the hand-written oracle's ~10% - so a learner
    does find the evasion strategy when the defender's posture is stable. Open: whether it
    can be learned against a re-aiming defender, where item 58 found it is not.)

## Next

56. Report the family results for trained policies, not only baselines, and check
    whether curriculum training on chains transfers to star, tree, mesh, and ring.
57. Vary host count within a family, so structure and scale are separable rather than
    confounded in one number.
60. Give the attacker a noisier read on monitoring. The current channel reports watched
    hosts exactly for every discovered host, which is an optimistic bound on what an
    attacker can fingerprint.
61. Learn evasion against a re-aiming defender, or show it cannot be learned - the open
    half of item 59. Item 58 found a fixed posture is learnable and a re-aiming one is
    not for this policy; whether a policy with memory of the defender's moves can close
    that gap is the question.

Each milestone is delivered as a small logical change only after lint, formatting,
typing, and tests pass.
