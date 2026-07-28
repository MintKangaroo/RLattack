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
  agents ─ reward ─ Gymnasium environment ─ scenario/generator
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

- Fixed discrete action catalogue
- Observable-state-only action mask
- `terminated`와 budget `truncated` 구분
- Reward contribution과 detection risk
- Action이 영향을 준 sanitized node ID
- 실제로 진행한 network edge의 누적 weighted path cost

OS command, subprocess, socket, external SDK를 import하지 않습니다.

## Application

`rlattack.experiment`가 CLI, report, dashboard에서 공유하는 유일한 실행 경로입니다.
`run_episode`는 environment와 agent를 실행해 `EpisodeResult`를 만들고,
`build_dashboard_data`는 같은 scenario에서 네 baseline을 같은 seed 묶음으로 평가합니다.
따라서 화면 수치와 Python benchmark가 갈라지지 않습니다.

`rlattack.evaluation`은 episode마다 agent state를 reset합니다. Rule-based cursor처럼
episode 간 상태가 누출되어 결과를 왜곡하는 일을 방지합니다.

`rlattack.explain`은 action 당시의 observation과 실제 `affected_nodes`를 기록합니다.
Graph overlay의 visited state도 이 node ID를 사용합니다.

## Presentation

- `rlattack.cli`: scenario export, experiment report, dashboard 실행
- `rlattack.report`: 외부 asset이 없는 self-contained HTML
- `rlattack.dashboard`: FastAPI HTML, health, experiment endpoint

Dashboard bind는 loopback allowlist로 제한합니다. Browser와 server 사이에서 전달되는 값은
scenario size, difficulty, policy, reward, seed, budget뿐이며 target address나 command field가
없습니다.

## Optional training

`rlattack.training`은 Stable-Baselines3가 설치된 경우에만 DQN/PPO를 import합니다.
두 pipeline은 vectorized train/evaluation environment, checkpoint, TensorBoard log 계약을
공유합니다. Core import에는 PyTorch가 필요하지 않습니다.

## Data flow

1. `ExperimentConfig`가 user input을 검증합니다.
2. Generator가 deterministic `Scenario`를 만듭니다.
3. Environment가 observation과 action mask를 제공합니다.
4. Agent가 valid action 하나를 선택합니다.
5. Runner가 reward, risk, path cost, affected node를 trace에 추가합니다.
6. Evaluator가 shared seeds로 baseline metric을 집계합니다.
7. CLI는 JSON/HTML로, dashboard는 same-origin API로 같은 view model을 제공합니다.

## Trust boundary

Scenario file, experiment input, policy checkpoint, sanitized export는 untrusted input입니다.
Pydantic validation과 adapter sanitization을 통과해야 합니다. Dashboard는 observability와
synthetic experiment control만 제공하며 실제 인프라의 control plane이 아닙니다.
