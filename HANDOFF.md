# RLAttack 인수인계

> 다음 세션 시작점: 2026-08-30 기준 v0.9.0 작업본

## 현재 상태

- Package version: `0.9.0`
- 작업 브랜치: `feat/v090-exploration` (PR #8–#13 머지 완료, main이 v0.8.0)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 264 tests, package coverage 100%

## 이전 릴리스 요약 (PR #8–#13, 모두 머지됨)

- **v0.3** targeted action space, generalization benchmark, 확률적 dynamics
- **v0.4** 부분 관측, 능동적 defender, 다목적 episode, paired 유의성 검정, 전이 평가
- **v0.5** action masking 학습, defender 지연·노이즈, noisy discovery
- **v0.6** 조건 격자 평가, 학습하는 defender, sweep, 400k 정책
- **v0.7** 두 학습자 게임, 외부 graph import, noisy 학습의 음성 결과
- **v0.8** probe memory(0% → 6.2%), 균형 분석, 대응 예산, adversarial 학습

## v0.9에서 바뀐 것

1. **Held-out 구조 가족 (로드맵 54)** — `rlattack families`가 생성기가 만들 수 없는
   star/tree/mesh/ring 토폴로지에서 평가합니다(import 경로를 그대로 씁니다).
   **구조가 결과를 지배합니다** — noisy discovery, 8 hosts, 16 seeds에서 graph oracle이
   star 100%, mesh 100%, chain 56.2%, **ring 12.5%**입니다. Ring은 자연스러운 진입점이
   없고 최단 경로가 감기므로 oracle의 경로 지식이 값을 못 합니다.
   ⚠️ **생성기 shape 하나의 성공률로 정책 역량을 말하면 안 됩니다.**
2. **격자 확장은 혼합 균형을 만들지 못했습니다 (53, 음성 결과)** — 특화된 defender arm
   2개(`harden-only`, `revoke-only`)와 넓게 가는 공격자(`shortest-path-broad`)를
   추가했지만 균형은 여전히 순수합니다. 이유가 명확해졌습니다: **redundancy가 여기서는
   trade-off가 아니라 그냥 손해**입니다(revoke-only 상대 43.8% vs 75.0%). 여분 credential을
   얻는 추가 exploit의 노이즈가 이득을 넘습니다. 일반화하면 **detection risk가 모든 활동을
   균일하게 벌하는 단일 스칼라라서, 더 많이 하는 전략은 언제나 더 나쁩니다.**
   혼합 균형에는 defender의 주의가 **표적화**돼야 합니다(호스트별 감시를 우회 가능하게).
   → 로드맵 55번.
3. **`pivot-focused` 보상 전략 (51, 가설 기각)** — discover 0.2 / pivot 2.5로 보상 질량을
   옮겼더니 **더 나빠졌습니다**(6.2% → 0%). Pivot은 이미 발견한 호스트를 전제로 하는데,
   probe가 인접 호스트에 맞을 확률이 1/4 정도라 `0.2 × 0.25 = 0.05` 기대이득 대비
   실패비용 −0.3은 순손실입니다. **전제를 벌하면서 목표에만 보상을 주면 시퀀스에 도달할 수
   없습니다.** 3번 probe하고 12 step에 멈춥니다.
   → 로드맵 51은 여전히 열려 있습니다. 다음 후보: discovery 보상을 probe 성공률에 맞춰
   가격 매기기, 또는 value function이 pivot 보상을 probe까지 역전파할 만큼 긴 학습.

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
rlattack families --agent shortest-path --discovery noisy --episodes 16
rlattack transfer --episodes 32 --report artifacts/transfer.html
rlattack equilibrium --episodes 16
rlattack game --attacker bandit --rounds 200
rlattack import --input topology.graphml --output artifacts/imported.json
rlattack train --algorithm maskable-ppo --curriculum \
  --curriculum-timesteps 400000 --discovery noisy --reward pivot-focused
rlattack dashboard
```

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v1.0)

`docs/roadmap.md`의 55–57번입니다. 우선순위는 **55번(표적화된 defender 주의)** —
이것 없이는 정책 격자가 혼합 균형을 가질 수 없습니다.

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
