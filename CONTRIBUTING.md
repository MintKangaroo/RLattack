# Contributing

Use Python 3.10 or later and create changes on a focused branch. Install the complete
development environment with `python -m pip install -e ".[dev,dashboard]"`. Keep each pull
request small, typed, tested, and documented.

## Branch policy

- `main`: 항상 실행 가능한 안정 버전
- `develop`: 다음 릴리스 통합 브랜치
- `feat/<기능명>`: 기능별 작업 브랜치
- `fix/<문제명>`: 버그 수정 브랜치
- `docs/<문서명>`: 문서 변경 브랜치

기능과 수정 작업은 `develop`에서 분기하고, 검토가 끝난 Pull Request를 통해 `develop`에
통합합니다. 릴리스 준비가 완료된 변경만 `main`으로 병합합니다. `main`과 `develop`에
직접 push하지 않습니다.

Before committing, run:

```bash
make check
```

Use Conventional Commits, such as `feat(env): add deterministic reset` or
`test(env): reject an invalid action`. Never commit `.env`, credentials, tokens, live
target data, or unsanitized security data. Changes must preserve the simulation-only
boundary described in [SECURITY.md](SECURITY.md).
