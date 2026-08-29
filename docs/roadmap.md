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

## Next

29. Partial observability: hide unvisited graph structure behind an observation model
    instead of exposing the full binary state.
30. An active defender model whose responses depend on the attacker's trajectory.
31. Multi-objective scenarios and reward ablations reported with significance tests.
32. A curriculum over scenario size and difficulty, with transfer measured on unseen
    classes.

Each milestone is delivered as a small logical change only after lint, formatting,
typing, and tests pass.
