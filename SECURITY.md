# Security Policy

## Supported versions

The project is pre-alpha. Security fixes are applied to the latest `main` branch.

## Reporting a vulnerability

Do not open a public issue containing exploit details, secrets, personal data, or live
target information. Use the repository host's private security-advisory feature. If that
is unavailable, contact the maintainers privately through a verified project profile.
Expect acknowledgement within five business days.

## Simulation-only boundary

RLAttack is designed for deterministic simulated graphs and explicitly local cyber-range
adapters. It is not a scanner or exploitation system. Network discovery, public target
access, real credential use, remote shells, persistence, evasion, malware deployment, data
destruction, and data exfiltration are prohibited and outside the architecture.

Contributions must not add wrappers for Nmap, exploit frameworks, shell commands, or other
real-system attack mechanisms. Security action names must resolve only to validated simulator
state transitions. Local cyber-range adapters must be explicit, isolated, and incapable of
addressing public targets.

The optional dashboard is limited to `127.0.0.1`, `localhost`, or `::1` and accepts only
synthetic scenario and experiment parameters. Do not remove this loopback restriction or add
target addresses, commands, credentials, payloads, or arbitrary file paths to its API.

Never submit real credentials, tokens, cookies, private keys, personal information, or
live target data in issues, logs, fixtures, scenarios, or commits. ThreatGraph fixtures must
be anonymized and stripped of actionable risk material.
