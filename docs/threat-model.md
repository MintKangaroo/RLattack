# Threat Model

## Safety objective

RLAttack studies decision-making over synthetic security graphs without creating a route to
operate on real systems. Security-flavored actions are simulator events only.

## Prohibited capabilities

- network discovery or scanning;
- exploit-framework or payload execution;
- real credential authentication or collection;
- local or remote shell execution;
- persistence, evasion, destructive behavior, or data exfiltration; and
- import of live target identifiers or actionable exploit material.

## Planned trust boundaries

Scenario files, experiment configuration, sanitized ThreatGraph exports, saved policies, and
optional local cyber-range adapters are untrusted inputs. They must be validated before use.
Dashboard views are observability surfaces and synthetic-experiment controls rather than
control planes for real infrastructure. The server rejects non-loopback bind addresses.

## Foundation controls

- no client for public or target networks;
- loopback-only dashboard bind validation;
- no subprocess or shell integration;
- secrets and generated artifacts excluded from version control;
- read-only GitHub Actions permissions; and
- static analysis includes common Python security rules.

Later milestones must preserve these constraints and add tests proving that simulator actions
cannot address external systems.
