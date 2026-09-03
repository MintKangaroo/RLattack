# RLAttack 인수인계

> 다음 세션 시작점: 2026-09-02 기준 v1.0.0 작업본

## 현재 상태

- Package version: `1.0.0`
- 작업 브랜치: `feat/v100-targeted-attention` (main이 v0.9.0)
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 실행 경계: synthetic graph와 in-process state transition만 허용
- 품질 기준: Ruff, strict mypy, 284 tests, package coverage 100%

## 이전 릴리스 요약 (PR #8–#13, 모두 머지됨)

- **v0.3** targeted action space, generalization benchmark, 확률적 dynamics
- **v0.4** 부분 관측, 능동적 defender, 다목적 episode, paired 유의성 검정, 전이 평가
- **v0.5** action masking 학습, defender 지연·노이즈, noisy discovery
- **v0.6** 조건 격자 평가, 학습하는 defender, sweep, 400k 정책
- **v0.7** 두 학습자 게임, 외부 graph import, noisy 학습의 음성 결과
- **v0.8** probe memory(0% → 6.2%), 균형 분석, 대응 예산, adversarial 학습

## v1.0에서 바뀐 것 — 로드맵 55번 완료, 52·53번의 원인 규명

1. **표적화된 defender 주의 (55)** — 탐지 risk가 이제 **행동이 일어난 호스트에 귀속**됩니다.
   `attention_hosts`를 켜면 defender가 소수 호스트를 집중 감시하고 나머지에는 그만큼
   눈이 멉니다. **blind 배수는 focus에서 유도되어 호스트 평균이 정확히 1로 보존됩니다.**
   보존이 핵심입니다 — 보존하지 않으면 "더 많이 감시하는 defender"는 그냥 "더 강한
   defender"라서, 격자가 감시의 *배치*가 아니라 *총량*을 측정하게 됩니다.
   표적 defender에게 `harden`은 감시자를 공격자가 이미 점유한 지반으로 **재조준**하는
   행위로 해석됩니다.
2. **혼합 균형이 처음으로 나왔습니다** — `mesh` + 구속되는 탐지 임계값에서
   공격자 `shortest-path` 82% / `shortest-path-evasive` 18%,
   defender `attention-narrow` 54% / `attention-broad` 46%, value 0.811 (64 seeds).
   **defender 지지집합이 attention arm 2개뿐이고 균일 arm 7개는 전부 가중치 0**입니다.
   혼합이 arm을 늘려서가 아니라 표적 주의에서 나왔다는 직접 증거입니다.
   회피 이득: narrow 상대 **+14.1pp** (95% CI [+7.8, +21.1], p=0.0001, 128 paired seeds),
   broad 상대 −3.1pp (p=0.395). 균일 defender 상대로는 **모든 seed에서 정확히 0** (대조군).
3. **두 조건이 *모두* 필요하고, 하나만으로는 순수 균형으로 남습니다.**
   - **경로 다양성**: `mesh`만 node-disjoint 경로가 평균 2.12개고 유일하게 혼합됩니다.
     chain/star/tree/ring은 ~1.00이고 전부 순수입니다. 경로가 하나면 우회할 곳이
     없으므로 주의를 어떻게 배치해도 회피가 불가능합니다.
   - **구속되는 탐지 임계값**: 기본값 0.9에서는 oracle이 32 episode 중 2회만 탐지됩니다.
     탐지가 결과를 결정하지 않으므로 risk를 재가격해도 아무것도 안 바뀝니다.
     창은 좁습니다 — 0.25로 내리면 다시 순수로 붕괴합니다(회피가 양쪽 모두에 이득이 됨).
   ⚠️ **v0.9의 두 음성 결과(52, 53)는 정책이 빈약해서가 아니라 구조 때문이었습니다.**
4. **`--detection-threshold`가 일급 실험 조건이 되었습니다.** 기본값은 0.9 그대로라
   기존에 발표한 수치는 전부 유효합니다.
5. `rlattack families`가 **`routes=`**(진입점→최심 호스트의 node-disjoint 경로 수)를
   보고합니다. 혼합 여부를 결정하는 구조 변수입니다.

### v1.0의 한계 (반드시 함께 인용할 것)

- `shortest-path-evasive`는 monitoring 채널을 읽는 **손으로 쓴 oracle**이지 학습된
  정책이 아닙니다. 전략이 이득이라는 것만 보이고, 학습자가 그걸 찾아낸다는 건 아닙니다.
- monitoring 채널은 발견한 호스트의 감시 여부를 **정확히** 알려줍니다. 실제 공격자의
  fingerprinting은 이보다 노이즈가 크므로 회피 수치는 **낙관적 상한**입니다.

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
rlattack equilibrium --family mesh --family-hosts 8 \
  --detection-threshold 0.4 --episodes 64   # 혼합 균형이 나오는 조건
rlattack benchmark --defender targeted --episodes 32
rlattack game --attacker bandit --rounds 200
rlattack import --input topology.graphml --output artifacts/imported.json
rlattack train --algorithm maskable-ppo --curriculum \
  --curriculum-timesteps 400000 --discovery noisy --reward pivot-focused
rlattack dashboard
```

### 장시간 training은 램가드 아래에서 실행합니다

이 머신은 다른 프로젝트와 메모리를 공유합니다. 상한 없이 돌리면 실패하는 대신
스왑으로 밀려 머신 전체가 느려지므로, training은 transient cgroup으로 감쌉니다.

```bash
scripts/ramguard.sh -m 2G -- rlattack train --algorithm maskable-ppo \
  --curriculum --curriculum-timesteps 400000 --output-dir artifacts/policies/<name>
```

`-m`은 잡의 상한(초과 시 커널이 **해당 잡만** 종료, rc=137), `-r`은 시작 전
필요한 시스템 여유 메모리입니다(부족하면 시작 자체를 거부, rc=1). 기본값은
`RLATTACK_MEM_MAX=2G`, `RLATTACK_MEM_FREE=1G`이며 환경변수로 바꿀 수 있습니다.
curriculum training 1회의 실측 RSS는 약 400–600MiB라 2G면 충분합니다.

⚠️ **램가드는 이 잡 하나만 지킵니다. 머신 전체는 지켜주지 않습니다.**
v1.0 작업 중 학습을 백그라운드로 돌리면서 평가 스윕을 **동시에** 실행해 머신을
두 번 죽였습니다. 이 머신은 9.7GB인데 다른 프로젝트가 4GB 넘게 상주할 때가 있어
(예: `tools/measure_ic.py` 4.2GB) 여유가 수백 MB까지 떨어집니다. 규칙:

1. **학습 중에는 평가·벤치마크를 돌리지 않습니다.** 한 번에 하나씩입니다.
2. 학습은 **반드시 `-r`을 주어** 여유가 없으면 시작 자체가 거부되게 합니다
   (`-r 1500M` 권장). 상한 없이 시작하면 스왑으로 밀립니다.
3. 여러 seed를 돌릴 때 detached 체인(`nohup ... for seed in ...`)으로 걸어두지
   마세요. 중간에 죽으면 부분 체크포인트만 남고 어디까지 됐는지 알기 어렵습니다.
   한 번 돌리고 끝난 걸 확인한 뒤 다음을 돌립니다.
4. 시작 전과 도중에 `free -m`의 **available**을 봅니다. 1GB 아래면 돌리지 않습니다.

Dashboard: <http://127.0.0.1:8000>

## 다음 확장 후보 (v1.1)

`docs/roadmap.md`의 56–60번입니다. 우선순위는 **58번** — 52번(adversarial 학습)을
55번이 찾아낸 조건(mesh + 구속되는 임계값)에서 다시 돌리는 것입니다. 52번의 음성 결과는
"defender 축이 무기력하다"로 설명됐는데, 그 축은 이제 무기력하지 않습니다(attention arm
2개가 혼합 균형의 defender 지지집합 전부입니다). 그 다음이 **59번**(oracle이 아니라
학습된 정책이 우회를 찾아내는지)입니다.

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
- `expose_monitoring`은 기본 **꺼짐**입니다. 켜면 관측 공간이 넓어져 끄고 학습한
  체크포인트를 못 읽습니다. `--defender targeted`일 때만 자동으로 켜집니다.
- **주의(attention) 배분은 반드시 보존되어야 합니다.** `attention_split`이 focus에서
  blind를 유도해 호스트 평균을 1로 맞춥니다. blind를 자유 파라미터로 두면 arm 간
  비교가 배치가 아니라 총량 비교가 되어 교란됩니다 — 실제로 첫 시도에서 이 버그로
  broad arm의 평균 배수가 1.4가 되어 결과가 오염됐습니다.
- `ShortestPathOracle`은 구간 경로에만 monitoring 가중치를 씁니다. **objective 방문
  순서는 무가중 깊이로 정합니다.** 가중 거리로 순서를 정하면 방향 그래프에서 깊은
  objective가 먼저 오고 얕은 쪽으로 돌아갈 경로가 없어 `NetworkXNoPath`가 납니다.
- 탐지 기본 임계값 0.9에서는 탐지가 사실상 발동하지 않습니다(oracle 2/32).
  risk·defender 관련 실험을 새로 설계할 때는 **먼저 탐지가 구속하는지 측정하세요.**
  구속하지 않으면 어떤 정교한 defender 기제도 결과를 못 바꿉니다.
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
- 로컬 장기 training은 `scripts/ramguard.sh`로 메모리 상한을 걸고 실행합니다.
- Scanner, exploit framework, shell, 실제 credential, public target 연결을 추가하지 않습니다.
- Dashboard input에 target address, command, payload, 임의 file path를 추가하지 않습니다.
- UI metric은 별도 mock data가 아니라 반드시 `rlattack.experiment` 결과를 사용합니다.
- 평균만 보고하지 않습니다. 표본 크기와 분산을 함께 보고합니다.
