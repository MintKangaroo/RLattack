# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-29 기준 v0.6.0 작업본

## 현재 상태

- Package version: `0.6.0`
- 작업 브랜치: `feat/v060-sweep-and-conditions` (PR #8 → #9 → #10 → 이 브랜치 순으로 스택)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 204 tests, package coverage 100%

## 이전 릴리스 요약

- **v0.3 (PR #8)** — RL 문제를 non-degenerate하게: targeted action space, seed별
  scenario를 만드는 generalization benchmark, 재현 가능한 확률적 dynamics.
- **v0.4 (PR #9)** — 실험 프로토콜: 부분 관측, 능동적 defender, 다목적 episode,
  paired 유의성 검정, curriculum과 전이 평가. ⚠️ 이 PR의 전이 표는 v0.5에서 고친
  baseline 버그의 영향을 받았습니다.
- **v0.5 (PR #10)** — action masking 학습(이것 없이는 정책이 즉시 `stop`으로 퇴화),
  defender 반응 지연·노이즈, noisy discovery, 학습 정책 결과 공개.

## v0.6에서 바뀐 것

1. **조건 격자 평가** (`rlattack conditions`) — 공개 정책을 대조 조건에서만 보고하고
   있었습니다. 결과가 중요합니다: curriculum 정책은 adaptive defender에는 영향을 받지
   않지만(p=0.29) **noisy discovery에서 성공률 0%**입니다. Exact adjacency로 학습해
   probe하는 법을 배우지 못했기 때문입니다. Oracle은 같은 격자에서 우아하게
   저하됩니다(96.9 → 68.8 → 46.9%) — 다만 이는 특권적 경로 지식 덕분이지 성취가 아닙니다.
2. **학습하는 defender** (`rlattack game`) — `BanditDefender`가 에피소드마다 대응 정책을
   고르고 결과로 갱신합니다. Oracle 상대 200 rounds에서 한 arm으로 수렴해 공격자를
   92.0%로 억제(최악 고정 arm 96.5%). 사후에 고른 최선 고정 arm(91.5%)은 못 이기며,
   그 차이가 탐색 비용입니다.
3. **Sweep 인프라** (`rlattack sweep`) — learning rate·rollout·entropy를 명시적 knob로
   만들고, trial마다 학습·벤치마크 후 baseline 대비 유의성 검정을 붙입니다.
4. **400k 정책 공개** — oracle 대비 마진이 +1.04(p=0.030) → **+2.71(p=0.0005)**,
   탐지율 3.1% → **0.0%**. 100k가 하한선이었음을 보여줍니다.
5. **스크린샷 재촬영** — 이 환경에 chromium을 `--with-deps` 없이 설치할 수 있었습니다.
   재촬영 과정에서 반응형 결함 2건을 발견해 고쳤습니다(모바일 benchmark 행 3줄 깨짐,
   320px에서 41px 가로 넘침). 320~1920px 전 구간 가로 넘침 0을 확인했고, CI에는
   브라우저가 없으므로 CSS 규칙을 단위 테스트로 고정했습니다.

### 스크린샷 재생성

```bash
python -m pip install playwright && python -m playwright install chromium
rlattack demo --size medium --difficulty hard --seed 42 --agent shortest-path \
  --defender adaptive --episodes 8 --report /tmp/dash.html
# 이후 playwright로 docs/assets/*.png 촬영 (HANDOFF 이력의 스크립트 참조)
```

`--with-deps`는 sudo가 필요해 실패합니다. 그냥 `install chromium`만 쓰면 됩니다.

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
rlattack benchmark --size medium --difficulty hard --episodes 64
rlattack ablation --agent greedy --episodes 32 --compare-to shaped
rlattack transfer --episodes 32 --report artifacts/transfer.html
rlattack conditions --episodes 32 --policy artifacts/policies/mppo-400k/final.zip
rlattack game --agent shortest-path --rounds 200
rlattack sweep --curriculum-timesteps 40000        # .[training] 필요
rlattack train --algorithm maskable-ppo --curriculum --curriculum-timesteps 400000
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v0.7)

`docs/roadmap.md`의 42–45번 항목입니다.

1. **noisy discovery 조건에서 학습** — 현재 최대 약점입니다. 정책이 그 조건에서 0%입니다.
2. 공격자도 에피소드 간 적응시켜 학습자 2명의 게임으로
3. Bandit defender를 에피소드 진행 상황에 조건화된 정책으로 교체
4. 공개된 attack-graph 데이터셋 importer (현재 결과는 이 generator 구조에 한정)

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
- 공개 정책은 **학습한 조건에서만 유효합니다**. `docs/results.md`의 조건 격자 표를
  보지 않고 성능을 인용하지 마세요.
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
