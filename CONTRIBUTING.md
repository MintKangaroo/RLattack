# Contributing

Use Python 3.10 or later and create changes on a focused branch. Keep each pull request
small, typed, tested, and documented.

Before committing, run:

```bash
ruff check .
ruff format --check .
mypy .
pytest
```

Use Conventional Commits, such as `feat(env): add deterministic reset` or
`test(env): reject an invalid action`. Never commit `.env`, credentials, tokens, live
target data, or unsanitized security data. Changes must preserve the simulation-only
boundary described in [SECURITY.md](SECURITY.md).
