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

## Next

38. Longer training budgets and a hyperparameter sweep; the published policies are
    100k-step curriculum runs, which is a floor rather than a ceiling.
39. Re-capture the dashboard and transfer screenshots in an environment with a browser.
40. Evaluate learned policies under the adaptive defender and noisy discovery
    conditions, not only the control condition.
41. A defender that adapts its own policy, turning the setup into a two-player game.

Each milestone is delivered as a small logical change only after lint, formatting,
typing, and tests pass.
