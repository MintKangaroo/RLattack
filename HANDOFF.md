# RLAttack 인수인계

## 현재 상태

- Package version: `0.2.0`
- 기준 브랜치: `develop`
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 65 tests, package coverage 100%

## 구현 완료

1. Pydantic/NetworkX scenario schema와 reference validation
2. Deterministic Gymnasium environment와 action mask
3. `small`/`medium`/`large`, `easy`/`medium`/`hard` scenario generator
4. Random, Greedy, Rule-based, Graph Oracle baseline
5. Optional Stable-Baselines3 DQN/PPO training pipeline
6. Sparse, Shaped, Risk-aware, Cost-aware reward
7. Shared-seed evaluation과 weighted graph path cost
8. Action explanation, affected-node trace, visited graph overlay
9. Sanitized ThreatGraph file adapter
10. Shared experiment runner와 JSON view model
11. `rlattack demo`, `scenario`, `dashboard` CLI
12. Self-contained HTML report와 loopback-only FastAPI dashboard
13. Desktop/mobile screenshot와 browser interaction verification
14. Architecture, API, methodology, threat model, README 정비

## 주요 명령

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,dashboard]"

make check
rlattack demo
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 구현 메모

- `rlattack.experiment`가 CLI, HTML report, dashboard의 공통 실행 엔진입니다.
- `evaluate_agent`는 각 episode 전에 agent를 reset해 policy state 누출을 방지합니다.
- Environment `info`는 `affected_nodes`, `valid_action`, `path_cost`, `detection_risk`를
  제공합니다.
- Dashboard는 외부 asset/CDN이 없고 `127.0.0.1`, `localhost`, `::1` 외 bind를 거부합니다.
- `docs/assets/dashboard.png`와 `dashboard-mobile.png`는 seed 42 실제 실행 화면입니다.

## 유지보수 시 주의사항

- `main` 또는 `develop`에 직접 push하지 않습니다.
- 기능 변경은 `feat/*`, 수정은 `fix/*`, 문서는 `docs/*`에서 시작합니다.
- 장기 training은 CI에서 실행하지 않습니다.
- Scanner, exploit framework, shell, 실제 credential, public target 연결을 추가하지 않습니다.
- Dashboard input에 target address, command, payload, 임의 file path를 추가하지 않습니다.
- UI metric은 별도 mock data가 아니라 반드시 `rlattack.experiment` 결과를 사용합니다.
