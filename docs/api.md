# Dashboard and API

RLAttack `v0.7.0`은 simulation 결과를 탐색하는 local FastAPI dashboard를 제공합니다.
이 API가 실행하는 것은 검증된 `AttackPathEnv`뿐이며 scanner, shell, 외부 target client는
포함하지 않습니다.

## 실행

```bash
python -m pip install -e ".[dashboard]"
rlattack dashboard
```

기본 주소는 <http://127.0.0.1:8000>입니다. 안전 경계를 유지하기 위해 host는
`127.0.0.1`, `localhost`, `::1`만 허용합니다.

## Endpoint

### `GET /`

Interactive Simulation Observatory HTML을 반환합니다. 모든 CSS와 JavaScript는 문서에
포함되어 있고 외부 CDN을 호출하지 않습니다.

### `GET /health`

```json
{
  "status": "ok",
  "mode": "simulation-only"
}
```

### `GET /api/experiment`

새 deterministic episode와 baseline benchmark를 실행합니다.

| Query | Type | Default | 허용 값 |
| --- | --- | --- | --- |
| `size` | string | `medium` | `small`, `medium`, `large` |
| `difficulty` | string | `hard` | `easy`, `medium`, `hard` |
| `seed` | integer | `42` | 모든 integer |
| `agent` | string | `greedy` | `random`, `greedy`, `rule-based`, `shortest-path` |
| `reward_strategy` | string | `risk-aware` | `sparse`, `shaped`, `risk-aware`, `cost-aware` |
| `step_budget` | integer | `64` | 1–512 |
| `benchmark_episodes` | integer | `8` | 1–256 |
| `stochastic` | boolean | `true` | `true`, `false` |
| `defender` | string | `passive` | `passive`, `adaptive` |

예:

```bash
curl "http://127.0.0.1:8000/api/experiment?size=small&difficulty=easy&seed=7&agent=shortest-path&reward_strategy=shaped&step_budget=32&benchmark_episodes=4"
```

Response에는 다음 top-level field가 있습니다.

- `config`: 결과를 재현하는 전체 입력
- `scenario`: host node, network edge, oracle route와 graph 통계
- `episode`: outcome, reward, risk, path cost, affected node, decision trace
- `benchmarks`: 네 baseline의 success rate, detection rate, 평균 metric, 표준편차,
  95% reward 신뢰구간
- `benchmark_protocol`: benchmark 방식(`per-seed-scenario`)과 사용한 seed 목록
- `reward`: 선택 strategy의 명시적 reward parameter
- `safety`: simulation-only capability 선언

잘못된 enum이나 범위를 벗어난 budget은 HTTP `422`를 반환합니다. Dashboard API는 file
path를 받지 않으므로 학습된 checkpoint 평가는 CLI에서만 가능합니다.

## Batch benchmark

통계 분석용 raw episode record가 필요하면 CLI를 사용합니다.

```bash
rlattack benchmark --size medium --difficulty hard --episodes 64 \
  --output artifacts/benchmark.jsonl

rlattack benchmark --episodes 64 --format csv --output artifacts/benchmark.csv \
  --policy artifacts/policies/final.zip --policy-algorithm dqn
```

각 record는 `agent`, `seed`, `success`, `detected`, `steps`, `reward`,
`detection_risk`, `path_cost`를 포함합니다.

## Reward ablation

```bash
rlattack ablation --agent greedy --episodes 32 --compare-to shaped
```

Scenario, seed, dynamics, defender를 고정하고 reward strategy만 바꿔 paired 유의성 검정을
보고합니다. Heuristic baseline은 reward signal을 사용하지 않으므로, 행동 차이는 학습된
policy에서만 나타납니다.

## Transfer evaluation

```bash
rlattack transfer --episodes 32 --policy artifacts/policies/final.zip \
  --policy-algorithm ppo --output artifacts/transfer.jsonl
```

9개 (size × difficulty) class 전체를 공유 seed로 평가하고, 기준 class 대비 paired 검정을
출력합니다.

## Condition sweep

```bash
rlattack conditions --size medium --difficulty hard --episodes 32 \
  --policy artifacts/policies/mppo-400k/final.zip --output artifacts/conditions.jsonl
```

`defender × discovery` 격자 전체를 같은 seed로 평가하고 대조 조건 대비 paired 검정을
출력합니다. **학습된 policy는 반드시 이 표와 함께 인용해야 합니다** — 학습한 조건 밖에서는
성능이 보장되지 않습니다.

## Two-player game

```bash
rlattack game --agent shortest-path --rounds 200            # 학습 defender
rlattack game --attacker bandit --rounds 200                # 양쪽 모두 학습
rlattack game --defender-policy bandit --rounds 200         # 에피소드당 고정 설정 defender
```

`contextual`(기본) defender는 alert band·회수 가능 credential 유무·episode 진행 단계에
조건화된 정책을 학습합니다. `bandit`은 에피소드마다 설정 하나를 고릅니다.

## Importing an external attack graph

```bash
rlattack import --input topology.graphml --output artifacts/imported.json
rlattack import --input topology.json --topology-only
```

GraphML, GML, NetworkX node-link JSON을 받습니다. 모든 node ID는 익명 ID로 치환되고,
hostname·주소·exploit 자료가 포함되면 import를 거부합니다. 기본적으로 재생 가능한
service/vulnerability/credential/objective 층을 합성하며, `--topology-only`로 끌 수 있습니다.

## Training

```bash
python -m pip install -e ".[training]"
rlattack train --algorithm maskable-ppo --curriculum --curriculum-timesteps 400000
rlattack train --algorithm maskable-ppo --curriculum --discovery noisy
rlattack train --algorithm maskable-ppo --curriculum --defender adaptive
```

`--discovery`와 `--defender`로 **평가할 조건에서 학습**할 수 있습니다. v0.6 정책이
noisy discovery에서 0%였던 이유가 exact adjacency로만 학습했기 때문입니다.

`--curriculum`은 하나의 policy를 여러 stage에 걸쳐 이어서 학습합니다. 기본 algorithm은
`maskable-ppo`입니다. 각 상태에서 유효한 action이 전체의 1~2%뿐이라, masking 없이는
탐색 예산이 invalid action에 소모되어 정책이 즉시 `stop`하도록 수렴합니다.

## Portable report

HTTP server 없이 같은 view model을 보는 report도 만들 수 있습니다.

```bash
rlattack demo \
  --report artifacts/experiment.html \
  --json artifacts/experiment.json
```

HTML은 self-contained이고 실험 결과를 내부에 포함합니다. Offline report에서는 기존 결과를
탐색하고 JSON으로 저장할 수 있으며, 새 실험 실행은 local dashboard에서만 가능합니다.
