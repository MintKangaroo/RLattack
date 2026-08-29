# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-29 기준 v0.4.0 작업본

## 현재 상태

- Package version: `0.4.0`
- 작업 브랜치: `feat/v040-partial-observability` (v0.3 PR #8 위에 쌓여 있습니다)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 159 tests, package coverage 100%

## v0.3에서 바뀐 것 (연구 타당성)

v0.2는 엔지니어링 품질은 갖췄지만 RL 문제로서 degenerate했습니다. v0.3이 그 부분을
고쳤습니다 (PR #8).

1. Targeted action space (`action_type * target_count + target_index`)
2. Seed마다 scenario를 새로 만드는 generalization benchmark
3. 재현 가능한 확률적 dynamics와 detection threshold 종료
4. Discover / pivot 분리 (pivot은 credential foothold를 요구)
5. 학습 policy 평가 경로 (`SB3PolicyAgent`, `benchmark --policy`)
6. std, 95% CI, detection rate, JSONL/CSV export

## v0.4에서 바뀐 것 (연구 프로토콜)

1. **부분 관측** — agent는 정확한 detection risk 대신 양자화된 `alert_level`만 봅니다.
   고정 capacity padding으로 vector 길이가 네트워크 크기를 누설하지 않습니다.
2. **능동적 defender** — 누적 risk가 임계값을 넘으면 도달한 host의 탐지 민감도를 올리고
   credential을 회수합니다. 기본 off(대조군), `--defender adaptive`가 처치군입니다.
3. **다목적 episode** — 모든 objective를 수집해야 종료됩니다. Oracle은 모든 objective
   host를 지나는 경로를 계획합니다.
4. **Paired 유의성 검정** — `rlattack.stats`의 sign-flip permutation test와 percentile
   bootstrap. `benchmark --compare-to`, `ablation`, `transfer`에서 보고합니다.
5. **Curriculum과 전이 평가** — `rlattack transfer`가 9개 class를 공유 seed로 평가하고,
   `train --curriculum`이 하나의 policy를 stage에 걸쳐 학습합니다.

### 작업 중 발견해 고친 캘리브레이션 문제

- `steps_remaining`이 step budget으로 bound된 Box여서 stage 간 관측 공간이 달라졌습니다.
  → `budget_fraction` (0~1)로 정규화하고 절대값은 `info`로 옮겼습니다.
- Detection risk가 절대적인 "시끄러운 행동 예산"이라 `large` scenario는 경로가 길다는
  이유만으로 도달 불가능했습니다(oracle 0/16). → host 수로 정규화(`normalize_risk_by_size`),
  이후 19~38%. 남은 격차는 실제 일반화 gradient입니다.
- Curriculum stage는 scenario 크기에 비례해 step budget을 확장합니다.

### 전이 평가 결과 (graph oracle, 16 seeds, passive)

| Class | success | detected | mean steps |
| --- | --- | --- | --- |
| small/easy | 93.8% | 6.2% | 22.7 ± 4.0 |
| small/medium | 100.0% | 0.0% | 20.1 ± 6.1 |
| small/hard | 93.8% | 6.2% | 25.4 ± 3.6 |
| medium/easy | 68.8% | 31.2% | 46.6 ± 6.1 |
| medium/medium | 75.0% | 25.0% | 39.1 ± 10.0 |
| medium/hard | 93.8% | 6.2% | 28.4 ± 4.2 |
| large/easy | 37.5% | 62.5% | 104.6 ± 15.7 |
| large/medium | 18.8% | 81.2% | 95.6 ± 15.3 |
| large/hard | 18.8% | 81.2% | 97.4 ± 15.0 |

`easy`가 `hard`보다 어려운 구간이 있는 것은 shortcut edge가 없어 경로가 길기 때문입니다.

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
rlattack ablation --agent greedy --episodes 32 --compare-to shaped
rlattack transfer --episodes 32 --output artifacts/transfer.jsonl
rlattack train --algorithm ppo --curriculum          # .[training] 필요
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v0.5)

`docs/roadmap.md`의 33–36번 항목입니다.

1. DQN/PPO curriculum policy를 실제로 학습해 전이 표를 baseline과 유의성 검정으로 비교
2. Dashboard 스크린샷 재촬영과 defender/transfer 뷰 노출
3. Defender 반응 지연과 오탐 모델링 (임계값 문제 → 타이밍 문제)
4. 인접 노드를 정확한 adjacency가 아니라 noisy scan model로 드러내기

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
- `run_reward_ablation`은 reward만 바꾸고 나머지를 고정합니다. Heuristic baseline은
  reward signal을 쓰지 않으므로 행동 차이는 학습된 policy에서만 나타납니다.
- `StageEnv`는 reset마다 stage에서 새 scenario를 뽑습니다. SB3가 environment를 한 번만
  만들기 때문에, 그러지 않으면 stage가 class가 아니라 graph 하나를 가르칩니다.
- `docs/assets/dashboard.png`와 `dashboard-mobile.png`는 v0.2 실행 화면이라
  v0.3/v0.4 UI 반영을 위해 재촬영이 필요합니다.

## 유지보수 시 주의사항

- `main` 또는 `develop`에 직접 push하지 않습니다.
- 기능 변경은 `feat/*`, 수정은 `fix/*`, 문서는 `docs/*`에서 시작합니다.
- 장기 training은 CI의 `quality` job에서 실행하지 않습니다.
- Scanner, exploit framework, shell, 실제 credential, public target 연결을 추가하지 않습니다.
- Dashboard input에 target address, command, payload, 임의 file path를 추가하지 않습니다.
- UI metric은 별도 mock data가 아니라 반드시 `rlattack.experiment` 결과를 사용합니다.
- 평균만 보고하지 않습니다. 표본 크기와 분산을 함께 보고합니다.
