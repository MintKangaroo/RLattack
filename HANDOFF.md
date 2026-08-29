# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-29 기준 v0.5.0 작업본

## 현재 상태

- Package version: `0.5.0`
- 작업 브랜치: `feat/v050-defender-latency` (PR #8 → #9 → 이 브랜치 순으로 쌓여 있습니다)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 175 tests, package coverage 100%

## 이전 릴리스 요약

- **v0.3 (PR #8)** — RL 문제를 non-degenerate하게: targeted action space, seed별
  scenario를 만드는 generalization benchmark, 재현 가능한 확률적 dynamics,
  discover/pivot 분리, 학습 policy 평가 경로, 분산·신뢰구간 보고.
- **v0.4 (PR #9)** — 실험 프로토콜: 부분 관측(양자화 alert level, 크기를 숨기는 고정
  capacity), 능동적 defender, 다목적 episode, paired 유의성 검정, curriculum과 전이 평가.
  ⚠️ v0.4에서 보고한 전이 표는 아래 5번 버그의 영향을 받았습니다. v0.5의 수치를 쓰세요.

## v0.5에서 바뀐 것

1. **Action masking 학습** (`maskable-ppo`, `sb3-contrib`). 이것이 v0.5의 핵심입니다.
   각 상태에서 유효한 action은 전체의 1~2%(small scenario reset 시 288개 중 4개)뿐이라,
   masking 없이 학습한 PPO 100k step curriculum은 **퇴화 정책**으로 수렴했습니다 —
   3개 action 후 `stop`, 9개 scenario class 전부에서 동일한 4-step episode, 성공률 0%.
   Environment가 이미 mask를 노출하고 있었으므로, 학습이 그것을 쓰도록 바꿨습니다.
2. **Defender 반응 지연과 노이즈** — 이전에는 공격자의 정확한 risk를 읽고 임계값을 넘는
   순간 반응했습니다. 이제 노이즈 섞인 추정치를 읽고 `response_latency` step 뒤에 효과가
   발생합니다. 오탐 횟수는 `info`에 집계합니다.
3. **Noisy discovery** — exact adjacency에서는 action mask 자체가 토폴로지였습니다
   (인접한 host에만 `discover_host`를 제공). 이제 모든 미발견 host를 probe할 수 있고
   인접한 것만 확률적으로 성공합니다.
4. **리포트 뷰** — condition strip(dynamics/defender/discovery/observation),
   defender 타일, `transfer --report`로 만드는 self-contained 전이 표.
5. **전이 baseline 버그 수정** — `evaluate_transfer`가 seed만 받는 factory를 썼고 CLI가
   **설정된** size/difficulty로 baseline을 만들었습니다. 즉 `ShortestPathOracle`이 한
   scenario class로 만들어져 나머지 8개 class에서 실행됐고, route와 index가 다른 graph의
   것이었습니다. Greedy fallback 덕에 동작은 해서 표가 그럴듯해 보였을 뿐입니다.
   수정 후 oracle은 9개 class 전부에서 83~100%입니다(이전 large 18~38%).
   **v0.4에서 "실제 일반화 격차"라고 한 것은 대부분 이 버그였습니다.**

### 전이 표 (graph oracle, 24 seeds, passive defender, 수정 후)

| Class | success | detected | mean steps |
| --- | --- | --- | --- |
| small/easy | 100.0% | 0.0% | 17.1 ± 3.0 |
| small/medium | 100.0% | 0.0% | 14.7 ± 4.0 |
| small/hard | 100.0% | 0.0% | 18.4 ± 3.3 |
| medium/easy | 87.5% | 12.5% | 38.5 ± 3.4 |
| medium/medium | 91.7% | 8.3% | 30.2 ± 6.1 |
| medium/hard | 91.7% | 8.3% | 28.3 ± 4.9 |
| large/easy | 83.3% | 16.7% | 82.2 ± 4.7 |
| large/medium | 91.7% | 8.3% | 62.6 ± 7.3 |
| large/hard | 100.0% | 0.0% | 55.5 ± 6.8 |

Risk 정규화는 이 버그와 무관하게 여전히 필요합니다. 정규화를 끄면 수정된 oracle도
`large/easy`에서 0%, large 전체 0~50%입니다.

### 학습된 정책 재현

```bash
python -m pip install -e ".[dev,training]"
rlattack train --algorithm maskable-ppo --curriculum --seed 42 \
  --output-dir artifacts/policies/mppo
rlattack transfer --policy artifacts/policies/mppo/final.zip \
  --policy-algorithm maskable-ppo --observation curriculum \
  --episodes 24 --output artifacts/transfer.jsonl --report artifacts/transfer.html
```

Checkpoint는 저장소에 커밋하지 않습니다(`artifacts/`는 gitignore). 위 명령으로 재생성합니다.
100k step curriculum은 하한선이지 상한선이 아닙니다.

### 학습 결과 요약 (`docs/results.md`에 전체 표)

- `medium/hard` 32 seeds: MaskablePPO가 oracle과 동일한 96.9% 성공·3.1% 탐지,
  보상은 유의하게 높음(+1.04, p=0.030). 경로가 더 **짧아서**가 아니라 더 **조용해서**입니다
  (31.7 vs 27.3 steps).
- 전이: 학습한 class에서 87~100%, held-out `large`에서 58~83%로 oracle(83~100%) 대비
  17~25점 뒤처집니다.
- 마스킹 없는 PPO와 DQN은 **둘 다** 9개 class 전부에서 4-step·성공률 0%로 퇴화했습니다.

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
rlattack transfer --episodes 32 --report artifacts/transfer.html
rlattack train --algorithm maskable-ppo --curriculum      # .[training] 필요
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v0.6)

`docs/roadmap.md`의 38–41번 항목입니다.

1. 더 긴 학습 예산과 hyperparameter sweep (현재 공개 정책은 100k step)
2. 브라우저가 있는 환경에서 dashboard/transfer 스크린샷 재촬영
3. 학습된 정책을 adaptive defender·noisy discovery 조건에서도 평가
4. 자신의 정책을 학습하는 defender (2인 게임으로 확장)

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
- 학습에는 반드시 `maskable-ppo`를 씁니다. Masking 없는 `dqn`/`ppo`는 1~2%만 유효한
  action 공간에서 즉시 `stop`하는 정책으로 수렴합니다. 이는 v0.5에서 실제로 관측한
  결과이지 추측이 아닙니다.
- `SB3PolicyAgent`는 invalid action을 교정하지 않습니다. 단, `maskable-ppo`는 mask를
  입력으로 받아 학습되었으므로 평가 시에도 mask를 전달합니다(교정이 아니라 인터페이스).
- `docs/assets/dashboard.png`와 `dashboard-mobile.png`는 v0.2 실행 화면입니다.
  이 환경에는 브라우저가 없어 재촬영하지 못했습니다.

## 유지보수 시 주의사항

- `main` 또는 `develop`에 직접 push하지 않습니다.
- 기능 변경은 `feat/*`, 수정은 `fix/*`, 문서는 `docs/*`에서 시작합니다.
- 장기 training은 CI의 `quality` job에서 실행하지 않습니다.
- Scanner, exploit framework, shell, 실제 credential, public target 연결을 추가하지 않습니다.
- Dashboard input에 target address, command, payload, 임의 file path를 추가하지 않습니다.
- UI metric은 별도 mock data가 아니라 반드시 `rlattack.experiment` 결과를 사용합니다.
- 평균만 보고하지 않습니다. 표본 크기와 분산을 함께 보고합니다.
