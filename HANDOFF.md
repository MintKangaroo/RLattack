# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-29 기준 v0.3.0 작업본

## 현재 상태

- Package version: `0.3.0`
- 작업 브랜치: `feat/v030-targeted-actions` (기준 브랜치는 `main`)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 118 tests, package coverage 100%

## v0.3에서 바뀐 것 (연구 타당성)

v0.2는 엔지니어링 품질은 갖췄지만 RL 문제로서 degenerate했습니다. v0.3은 그 부분을
고쳤습니다. 자세한 내용은 `CHANGELOG.md` 참고.

1. **Targeted action space.** Action이 `action_type * target_count + target_index`가
   되었습니다. 이전에는 9개 parameterless action만 있었고 환경이 record 순서로 대상을
   골랐기 때문에, 경로 선택이 정책이 아니라 구현 세부사항이었습니다.
2. **Generalization benchmark.** Benchmark seed마다 scenario를 새로 생성합니다.
   이전 runner는 seed를 무시하고 고정 scenario를 재사용해 결정론적 policy의 분산이
   항상 0이었습니다(= 표본 크기 1을 8번 보고).
3. **재현 가능한 확률적 dynamics.** Exploitation은 `exploitability` 기반 확률로
   성공하고, 실패는 risk를 올리며, `detection_threshold` 초과는 episode 실패입니다.
   난수는 전부 seed된 `np_random`에서 나옵니다.
4. **Discover / pivot 분리.** Pivot은 source host의 credential foothold를 요구합니다.
5. **학습 policy 평가 경로.** `rlattack.policies.SB3PolicyAgent` +
   `rlattack benchmark --policy`로 checkpoint를 baseline과 같은 protocol에서 비교합니다.
6. **통계 보고.** std, 95% 신뢰구간, detection rate, episode 단위 JSONL/CSV export.
7. **테스트.** 재현성·solvability·CLI round-trip integration test 추가, env의 도달
   불가능한 fallback 제거, MultiBinary(0) 크래시 수정, dashboard 입력 상한 추가.
8. **인프라.** CI Python 3.10–3.13 matrix, 주간 training smoke workflow,
   `CHANGELOG.md`, `CITATION.cff`.

### 결과 (medium/hard, risk-aware, 16 seeds)

| Policy | success | detected | mean steps |
| --- | --- | --- | --- |
| Random | 0.44 | 0.56 | 48.6 ± 7.7 |
| Greedy | 0.25 | 0.75 | 48.4 ± 7.0 |
| Rule-based | 0.00 | 1.00 | 49.8 ± 5.8 |
| Graph oracle | 0.81 | 0.19 | 36.7 ± 4.6 |

Oracle과 heuristic baseline 사이에 실제 격차가 생겼습니다. 학습 policy가 채울 여지가
있는 상태입니다.

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
rlattack demo
rlattack benchmark --size medium --difficulty hard --episodes 64 \
  --output artifacts/benchmark.jsonl
rlattack train --algorithm ppo --timesteps 200000      # .[training] 필요
rlattack benchmark --policy artifacts/policies/final.zip --policy-algorithm ppo
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v0.4)

`docs/roadmap.md`의 29–32번 항목입니다.

1. 부분 관측(POMDP): 방문하지 않은 graph 구조를 관측 모델 뒤로 숨기기
2. 공격자 궤적에 반응하는 능동적 defender model
3. 다목적 시나리오와 유의성 검정을 포함한 reward ablation
4. Scenario 크기·난이도 curriculum과 unseen class 전이 측정

확장 시에도 synthetic graph, in-process transition, loopback-only dashboard라는 안전
경계를 유지합니다.

## 구현 메모

- `rlattack.experiment`가 CLI, HTML report, dashboard의 공통 실행 엔진입니다.
- `run_benchmarks`가 generalization benchmark 진입점이고, `evaluate_agent`는
  agent와 env를 모두 seed 기반 factory로 받습니다.
- `ShortestPathOracle`은 route host의 **credential 경로만** 공략합니다. 모든 service를
  enumerate하면 detection threshold에 걸리며, 이것이 oracle과 greedy를 가르는 요인입니다.
- `SB3PolicyAgent`는 invalid action을 교정하지 않습니다(있는 그대로 평가).
- Environment `info`는 `target_id`, `outcome`, `objective_captured`, `detected`,
  `target_count`를 포함합니다.
- Dashboard는 외부 asset/CDN이 없고 `127.0.0.1`, `localhost`, `::1` 외 bind를 거부하며
  file path 입력을 받지 않습니다.
- `docs/assets/dashboard.png`와 `dashboard-mobile.png`는 v0.2 실행 화면이라
  v0.3 UI(target/outcome 열, ±SD, detection rate) 반영을 위해 재촬영이 필요합니다.

## 유지보수 시 주의사항

- `main` 또는 `develop`에 직접 push하지 않습니다.
- 기능 변경은 `feat/*`, 수정은 `fix/*`, 문서는 `docs/*`에서 시작합니다.
- 장기 training은 CI의 `quality` job에서 실행하지 않습니다.
- Scanner, exploit framework, shell, 실제 credential, public target 연결을 추가하지 않습니다.
- Dashboard input에 target address, command, payload, 임의 file path를 추가하지 않습니다.
- UI metric은 별도 mock data가 아니라 반드시 `rlattack.experiment` 결과를 사용합니다.
- 평균만 보고하지 않습니다. 표본 크기와 분산을 함께 보고합니다.
