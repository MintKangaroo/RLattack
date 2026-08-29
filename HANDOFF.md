# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-29 기준 v0.8.0 작업본

## 현재 상태

- Package version: `0.8.0`
- 작업 브랜치: `feat/v080-probe-memory` (PR #8–#12는 머지 완료, main이 v0.7.0)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 254 tests, package coverage 100%

## 이전 릴리스 요약 (PR #8–#12, 모두 머지됨)

- **v0.3** targeted action space, generalization benchmark, 확률적 dynamics
- **v0.4** 부분 관측, 능동적 defender, 다목적 episode, paired 유의성 검정, 전이 평가
- **v0.5** action masking 학습, defender 지연·노이즈, noisy discovery, 정책 결과 공개
- **v0.6** 조건 격자 평가, 학습하는 defender, sweep, 400k 정책, 스크린샷 재촬영
- **v0.7** 두 학습자 게임, 외부 graph import, noisy 학습의 정직한 음성 결과

## v0.8에서 바뀐 것

1. **`probed_hosts` 관측 채널 (로드맵 50, 부분 해결)** — 어느 host를 probe해서 실패했는지가
   action mask에만 있었고, maskable learner는 mask를 정책 입력이 아니라 분포 필터로만
   씁니다. 즉 정책이 "다 훑어봤다"와 "아직 안 훑었다"를 구분할 수 없었습니다.
   채널 추가 후 같은 400k 예산으로 재학습하니 noisy discovery 성공률이
   **0% → 6.2%**로 올랐습니다. Oracle은 68.8%이므로 **격차는 좁혔을 뿐 닫지 못했습니다.**
   관측 문제는 필요조건이었지 충분조건이 아니었고, 남은 거리는 여전히 탐색 문제입니다.
   ⚠️ **관측 공간이 바뀌었으므로 이전 정책 체크포인트는 로드할 수 없습니다.**
2. **균형 분석 (48)** — `rlattack equilibrium`이 정책 격자를 fictitious play로 풉니다.
   medium/hard에서 균형은 순수 전략(shortest-path vs `fast`, 값 0.75)입니다.
   shortest-path가 다른 모든 공격자 행을 지배하기 때문이고, 명령이 그 사실을 명시합니다.
3. **Defender 대응 예산 (49)** — 에피소드당 대응 횟수 상한. 초과 결정은 버려지되 집계됩니다.
   학습 defender는 `budget_pressure`를 읽고 배급합니다. **예산 8이 무제한보다 낫습니다**
   (공격자 85.8%±2.6% vs 89.8%±4.6%) — 제약이 낭비를 막습니다.
4. **Adversarial 학습 (47)** — `--adversarial`로 학습 중 defender가 함께 학습합니다.
   SB3가 에피소드 루프를 소유하므로 `StageEnv.reset`에서 구동합니다.
5. **Import 매핑 가이드 (46)** — `docs/importing.md`. 작성·검증 중 sanitizer 구멍을
   발견해 고쳤습니다(일반 필드 안의 FQDN이 통과했습니다).

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
rlattack conditions --episodes 32 --policy artifacts/policies/<name>/final.zip
rlattack transfer --episodes 32 --report artifacts/transfer.html
rlattack equilibrium --episodes 16
rlattack game --attacker bandit --rounds 200
rlattack import --input topology.graphml --output artifacts/imported.json
rlattack train --algorithm maskable-ppo --curriculum \
  --curriculum-timesteps 400000 --discovery noisy      # .[training] 필요
rlattack train --curriculum --adversarial
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v0.9)

`docs/roadmap.md`의 51–54번입니다. 우선순위는 **51번(로드맵 50 마무리)** —
남은 격차는 관측이 아니라 탐색이며, intrinsic novelty bonus나 pivot 성공에 대한 보상,
또는 400k를 크게 넘는 예산이 후보입니다.

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
- 관측 공간이 v0.8에서 바뀌었습니다. v0.7 이전 체크포인트는 로드되지 않습니다.
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
