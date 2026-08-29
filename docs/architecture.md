# Architecture

## 개요

RLAttack은 simulation domain, experiment application, presentation의 세 계층으로 나뉩니다.
의존 방향은 presentation에서 domain을 향하며, core environment는 FastAPI,
Stable-Baselines3, TensorBoard를 알지 못합니다.

```text
Presentation
  CLI ───────── HTML report ───────── loopback dashboard/API
   │                 │                         │
   └─────────────────┴─────────────────────────┘
                         │
Application              ▼
  experiment runner ─ evaluation ─ explainability
          │                 │              │
          └─────────────────┴──────────────┘
                         │
Domain                   ▼
  agents ─ reward ─ defender ─ Gymnasium environment ─ scenario/generator
                                                ▲
                                                │
                                      sanitized file adapter
```

## Domain

### Scenario

`rlattack.scenario`의 frozen Pydantic model이 Host, Service, Vulnerability, Credential,
Privilege, Objective, SecurityControl과 edge를 검증합니다. 중복 ID와 끊어진 reference는
환경 생성 전에 거부됩니다. `Scenario.to_networkx()`는 같은 데이터를 `MultiDiGraph`로
표현합니다.

### Generator

`rlattack.generator`는 전용 `random.Random(seed)`를 사용합니다. Global RNG, file,
network 상태를 읽지 않으므로 size·difficulty·seed가 같으면 같은 model dump를 만듭니다.

### Environment

`AttackPathEnv`는 Gymnasium `reset`/`step` 계약을 따르는 in-memory MDP입니다.

- **Targeted action space**: action은 `action_type * target_count + target_index`로
  encode됩니다. Policy가 "무엇을 할지"와 "graph의 어느 요소에 할지"를 모두 선택합니다.
  Discrete space이므로 DQN과 PPO 모두 그대로 사용할 수 있습니다.
- Observable-state-only action mask (flat, `9 * target_count`)
- **Reproducible stochastic dynamics**: `DynamicsConfig`가 exploit 성공 확률,
  실패 시 risk 증가, detection threshold를 정의합니다. 모든 난수는 seed된
  `np_random`에서 나오므로 seed가 같으면 trajectory도 같습니다.
  `DynamicsConfig.deterministic()`은 모든 valid action이 성공하는 회귀 테스트용 모드입니다.
- **Detection termination**: `detection_risk`가 threshold를 넘으면 episode가 실패로
  종료됩니다. Risk가 dynamics에 실제로 영향을 주므로 risk-aware reward ablation이
  의미를 갖습니다.
- Discovery와 pivot의 분리: `discover_host`는 host를 알게 할 뿐이고,
  `pivot_simulated_network`가 reachability를 부여하며 source host의 credential
  foothold를 요구합니다. Service scan은 reachability를 전제로 합니다.
- **Partial observability**: `ObservationConfig`가 agent가 볼 수 있는 것을 정의합니다.
  기본적으로 정확한 detection risk 대신 양자화된 `alert_level`만 노출하며, 고정 capacity로
  모든 channel을 padding해 vector 길이가 네트워크 크기를 누설하지 않게 합니다.
  같은 capacity가 scenario class 간 policy 전이를 가능하게 합니다.
- **Adaptive defender**: `DefenderConfig`가 켜지면 누적 risk가 임계값을 넘을 때 도달한
  host의 탐지 민감도를 높이고 credential을 회수합니다. 기본값은 off(대조군)입니다.
- **Multi-objective**: 모든 objective를 수집해야 episode가 종료됩니다.
- `terminated`와 budget `truncated` 구분, `info["objective_captured"]`로 성공을 판정
- Action이 영향을 준 sanitized node ID와 `target_id`
- 실제로 진행한 network edge의 누적 weighted path cost

OS command, subprocess, socket, external SDK를 import하지 않습니다.

## Application

`rlattack.experiment`가 CLI, report, dashboard에서 공유하는 유일한 실행 경로입니다.
`run_episode`는 environment와 agent를 실행해 `EpisodeResult`를 만들고,
`build_dashboard_data`는 같은 view model을 CLI, report, dashboard에 공급합니다.
따라서 화면 수치와 Python benchmark가 갈라지지 않습니다.

`run_benchmarks`는 **generalization benchmark**입니다. Benchmark seed마다 scenario를
새로 생성하므로, 하나의 고정 graph를 반복 재생하지 않습니다. 이전 버전은 seed를 무시하고
고정 scenario를 재사용해 결정론적 policy의 분산이 항상 0이었습니다.

`rlattack.evaluation`은 episode마다 agent를 factory로 새로 만들고 state를 reset하며,
평균과 함께 표준편차, 95% 신뢰구간, detection rate, episode별 outcome을 보고합니다.
평균만으로는 연구 주장을 뒷받침할 수 없기 때문입니다.

`rlattack.export`는 episode 단위 결과를 JSONL/CSV로 내보내 외부 통계 분석에 사용합니다.

`rlattack.stats`는 seed로 pairing된 episode에 대해 sign-flip permutation test와 percentile
bootstrap을 수행합니다. 정규성 가정과 SciPy 의존이 없습니다.

`rlattack.curriculum`은 scenario stage와 전이 평가를 제공합니다. `StageEnv`는 reset마다
stage에서 새 scenario를 뽑습니다. Stable-Baselines3는 environment를 한 번만 만들기 때문에,
그렇게 하지 않으면 stage가 class가 아니라 하나의 graph를 가르치게 됩니다.

`rlattack.explain`은 action 당시의 observation과 실제 `affected_nodes`를 기록합니다.
Graph overlay의 visited state도 이 node ID를 사용합니다.

## Presentation

- `rlattack.cli`: scenario export, experiment report, benchmark/ablation/transfer
  export, optional training, dashboard 실행
- `rlattack.report`: 외부 asset이 없는 self-contained HTML
- `rlattack.dashboard`: FastAPI HTML, health, experiment endpoint

Dashboard bind는 loopback allowlist로 제한합니다. Browser와 server 사이에서 전달되는 값은
scenario size, difficulty, policy, reward, seed, budget, dynamics flag뿐이며 target address,
command field, file path가 없습니다. `step_budget`과 `benchmark_episodes`에는 상한이 있어
loopback 요청이 무제한 계산을 예약할 수 없습니다.

학습된 policy checkpoint는 file path 입력이므로 dashboard가 아니라 CLI
(`rlattack benchmark --policy`)에서만 사용할 수 있습니다.

## Optional training

`rlattack.training`은 Stable-Baselines3가 설치된 경우에만 DQN/PPO를 import합니다.
두 pipeline은 vectorized train/evaluation environment, checkpoint, TensorBoard log 계약을
공유합니다. Core import에는 PyTorch가 필요하지 않습니다.

`rlattack.policies`의 `SB3PolicyAgent`는 학습된 checkpoint를 baseline과 동일한 `Agent`
protocol로 감쌉니다. Wrapper는 invalid action을 교정하지 **않습니다**. Policy가 학습한
action space 그대로 평가해야 실제 성능을 보고할 수 있기 때문입니다.

예외는 `maskable-ppo`입니다. 이 정책은 action mask를 **입력으로 받아 학습**되므로 mask가
인터페이스의 일부이고, 평가 시에도 동일하게 전달합니다. 사후 교정이 아닙니다.
Environment는 `action_masks()`로 boolean mask를 노출합니다.

## Data flow

1. `ExperimentConfig`가 user input을 검증합니다.
2. Generator가 deterministic `Scenario`를 만듭니다.
3. Environment가 observation과 action mask를 제공합니다.
4. Agent가 valid action 하나를 선택합니다.
5. Runner가 reward, risk, path cost, affected node를 trace에 추가합니다.
6. Evaluator가 seed마다 새 scenario를 만들어 baseline metric과 분산을 집계합니다.
7. CLI는 JSON/HTML로, dashboard는 same-origin API로 같은 view model을 제공합니다.

## Trust boundary

Scenario file, experiment input, policy checkpoint, sanitized export는 untrusted input입니다.
Pydantic validation과 adapter sanitization을 통과해야 합니다. Dashboard는 observability와
synthetic experiment control만 제공하며 실제 인프라의 control plane이 아닙니다.

Sanitized export는 source Scenario의 ID를 그대로 쓰지 않고 Host, Service, Vulnerability ID를
새 anonymous ID로 치환합니다. Import는 node 순서와 무관하게 reference를 복원하고, entry
host와 weighted network edge를 보존합니다.
