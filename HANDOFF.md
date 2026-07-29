# RLAttack 인수인계

> 다음 세션 시작점: 2026-07-30 기준 v0.2.0 완성본

## 현재 상태

- Package version: `0.2.0`
- 기준 브랜치: `main`
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 최신 main 병합 커밋: `1031c8b` (`release: promote RLAttack v0.2 simulation observatory`)
- 최근 릴리스 PR: [#3](https://github.com/MintKangaroo/RLattack/pull/3)
- 현재 상태: GitHub Actions `quality` 통과, 작업 트리 clean
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 67 tests, package coverage 100%

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
15. ThreatGraph export ID 재익명화와 구조 보존 round-trip
16. pip-audit CI gate와 Python/GitHub Actions Dependabot
17. DQN/PPO CPU smoke training, Monitor evaluation, final checkpoint 검증

## 다음 세션 시작 명령

```bash
git clone https://github.com/MintKangaroo/RLattack.git
cd RLattack
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,dashboard]"
make check
make audit
```

## 주요 실행 명령

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,dashboard]"

make check
make audit
rlattack demo
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 현재 검증 결과

- `67 passed`, package coverage `100%` (916 statements)
- Ruff check/format 및 strict mypy 통과
- project-scoped `pip-audit`: known vulnerability 없음
- clean virtualenv `pip check`: broken requirement 없음
- DQN/PPO CPU smoke training과 final checkpoint 생성 확인
- desktop/mobile dashboard browser rendering 및 API validation 확인

## 다음 확장 후보

기능 누락으로 인한 blocker는 없습니다. 다음 작업은 v0.3 연구 확장으로 별도 계획을 세워 진행합니다.

1. 더 큰 synthetic graph와 다중 seed benchmark report
2. 실험 결과 artifact schema 및 CSV/JSONL batch export
3. dashboard benchmark history 비교와 결과 다운로드 UX
4. 실제 연구 질문에 맞춘 ablation protocol과 통계 요약

확장 시에도 synthetic graph, in-process transition, loopback-only dashboard라는 안전 경계를 유지합니다.

## 구현 메모

- `rlattack.experiment`가 CLI, HTML report, dashboard의 공통 실행 엔진입니다.
- `evaluate_agent`는 각 episode 전에 agent를 reset해 policy state 누출을 방지합니다.
- Environment `info`는 `affected_nodes`, `valid_action`, `path_cost`, `detection_risk`를
  제공합니다.
- Episode는 `terminated`와 budget `truncated` 모두에서 reset 전 추가 step을 거부합니다.
- DQN/PPO train·evaluation environment는 Stable-Baselines3 `Monitor`로 감쌉니다.
- Dashboard는 외부 asset/CDN이 없고 `127.0.0.1`, `localhost`, `::1` 외 bind를 거부합니다.
- `docs/assets/dashboard.png`와 `dashboard-mobile.png`는 seed 42 실제 실행 화면입니다.

## 유지보수 시 주의사항

- `main` 또는 `develop`에 직접 push하지 않습니다.
- 기능 변경은 `feat/*`, 수정은 `fix/*`, 문서는 `docs/*`에서 시작합니다.
- 장기 training은 CI에서 실행하지 않습니다.
- Scanner, exploit framework, shell, 실제 credential, public target 연결을 추가하지 않습니다.
- Dashboard input에 target address, command, payload, 임의 file path를 추가하지 않습니다.
- UI metric은 별도 mock data가 아니라 반드시 `rlattack.experiment` 결과를 사용합니다.
