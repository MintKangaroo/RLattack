# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-29 기준 v0.7.0 작업본

## 현재 상태

- Package version: `0.7.0`
- 작업 브랜치: `feat/v070-noisy-training` (PR #8 → #9 → #10 → #11 → 이 브랜치 순으로 스택)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 233 tests, package coverage 100%

## 이전 릴리스 요약

- **v0.3 (PR #8)** — targeted action space, generalization benchmark, 확률적 dynamics.
- **v0.4 (PR #9)** — 부분 관측, 능동적 defender, 다목적 episode, paired 유의성 검정,
  curriculum·전이 평가. ⚠️ 전이 표는 v0.5에서 고친 baseline 버그의 영향을 받았습니다.
- **v0.5 (PR #10)** — action masking 학습, defender 지연·노이즈, noisy discovery,
  학습 정책 결과 공개.
- **v0.6 (PR #11)** — 조건 격자 평가, 학습하는 defender, sweep 인프라, 400k 정책,
  스크린샷 재촬영.

## v0.7에서 바뀐 것

1. **학습 조건 배선 버그 수정 (중요)** — `--discovery`/`--defender`/`--reward`가
   파싱·출력만 되고 **환경까지 전달되지 않았습니다.** ruff 재포맷으로 빌더 호출이 한 줄이
   되면서 이후 문자열 치환이 조용히 no-op이 됐습니다. "noisy로 학습"이라던 실행은 실제로는
   exact로 학습된 것이었고, 그로부터 끌어낸 결론(망각 포함)은 전부 폐기했습니다.
   이제 조건이 로그가 아니라 **환경에 도달하는지** 테스트가 검증합니다.
2. **noisy discovery 학습 결과 (로드맵 42)** — 조건에서 학습하면 그 조건의 보상은
   -4.18 → 7.45로 오르고 탐지는 0%가 되지만 **성공률은 여전히 0%**입니다. Trace를 보면
   entry host의 exploitation chain은 정확히 학습했고, 이웃 probe에 7번 실패한 뒤
   **멈춥니다**. Oracle이 같은 조건에서 68.8%이므로 환경 결함이 아니라 탐색 문제입니다.
   400k에서 보상이 12~15로 정체했습니다. 후속은 로드맵 50번.
3. **맥락 조건부 defender (44)** — alert band·회수 가능 credential·진행 단계에 조건화된
   정책을 MC로 학습. 5 seed × 256 rounds에서 oracle을 87.7%±3.9%로 억제
   (bandit 92.7%±1.0%), defender 보상 4배. 이를 의미 있게 만들려면 **대응 비용**이
   필요했습니다 — 무료일 때는 "매 step 대응"이 자명한 최적해였습니다.
4. **학습하는 공격자 (43)** — 양쪽이 에피소드 간 적응합니다. 공격자가 shortest-path를
   스스로 찾아냅니다(256 중 232). epsilon-greedy 코어는 `rlattack.bandit`으로 공유.
5. **외부 attack graph import (45)** — GraphML/GML/node-link JSON → 익명화 scenario.
   구조는 보존하고 ID는 전부 치환하며, hostname·주소가 있으면 거부합니다.
6. **Curriculum 누적 혼합** — 각 stage가 이전 stage도 샘플링합니다
   (`--forget-previous-stages`로 격리). 관측된 회귀를 고친 것이 아니라 표준적 예방입니다.

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
rlattack conditions --episodes 32 --policy artifacts/policies/mppo-400k/final.zip
rlattack transfer --episodes 32 --report artifacts/transfer.html
rlattack game --attacker bandit --rounds 200
rlattack import --input topology.graphml --output artifacts/imported.json
rlattack sweep --curriculum-timesteps 40000                  # .[training] 필요
rlattack train --algorithm maskable-ppo --curriculum \
  --curriculum-timesteps 400000 --discovery noisy
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v0.8)

`docs/roadmap.md`의 46–50번입니다. 우선순위는 **50번(noisy discovery 탐색 격차)**입니다 —
현재 학습 정책의 가장 큰 실패 지점이고 원인이 특정돼 있습니다.

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
- **학습 조건 플래그를 추가하면 반드시 환경까지 도달하는지 테스트하세요.** v0.7에서
  로그에는 찍히지만 환경에는 전달되지 않는 버그로 400k 실행 하나를 통째로 버렸습니다.
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
