# RLAttack

RLAttack은 결정론적 시뮬레이션 네트워크와 취약점 그래프 안에서 강화학습 Agent가
공격 경로 탐색 전략을 학습하는 연구 플랫폼입니다.

공격 계획을 재현 가능한 MDP(Markov Decision Process)로 모델링합니다. Agent는 부분적인
사이버 레인지 상태를 관찰하고, 하나의 시뮬레이션 Action을 선택하며, 설정 가능한 Reward를
받습니다. 목표를 수집하거나 Agent가 중단하거나 Step 예산이 소진되면 Episode가 종료됩니다.
동일한 Scenario와 Seed는 항상 동일한 Trajectory를 생성하므로 알고리즘 비교와 Ablation
실험을 검증할 수 있습니다.

> 현재 상태: 4단계 결정론적 Scenario Generator까지 구현되어 있습니다. Baseline Agent와
> 학습 파이프라인은 이후 단계에서 구현합니다.

## 안전 범위

RLAttack은 외부 시스템을 스캔하거나 공격하거나 인증하거나 Shell을 실행하지 않습니다.
`scan_service`, `attempt_simulated_access`와 같은 이름은 공격 도구를 호출하는 기능이
아니라 시뮬레이터 내부 상태 전이를 뜻합니다.

허용되는 실행 환경은 다음과 같습니다.

- 결정론적 In-process 시뮬레이터
- 공용 네트워크에 연결할 수 없는 명시적 로컬 사이버 레인지 Adapter
- 실제 대상과 Exploit 자료가 제거된 Sanitized Graph Import/Export

Nmap, Exploit Framework, 원격 Shell, 임의 명령 실행, 공용 대상 Adapter는 프로젝트 범위에
포함되지 않습니다.

## 연구 질문

- 탐색·비용·탐지 위험이 동시에 존재할 때 DQN과 PPO는 어떻게 비교되는가?
- 학습된 Policy는 Random, Greedy, Rule-based, Shortest-path Baseline보다 언제 우수한가?
- 작은 학습 Graph에서 학습한 Policy가 더 큰 Graph 또는 보지 못한 Scenario로 일반화되는가?
- 선택된 Action을 Host, Service, Vulnerability, Privilege, Reachability, Risk, Budget 신호로
  설명할 수 있는가?

## 환경 계약 (예정)

Gymnasium 환경은 고정된 Action 목록을 제공하며, 모든 Action은 시뮬레이터 내부의 상태 전이만
수행합니다.

| Action | 시뮬레이션 효과 |
| --- | --- |
| `discover_host` | Scenario Graph에서 도달 가능한 Host를 발견 |
| `scan_service` | Host의 시뮬레이션 Port와 Service 정보를 확인 |
| `enumerate_service` | Service 상세 정보를 Observation에 추가 |
| `validate_vulnerability` | 모델링된 Vulnerability의 유효성을 검증 |
| `attempt_simulated_access` | 모델링된 Access Edge와 Credential 전이를 시도 |
| `escalate_simulated_privilege` | 모델링된 Privilege 전이를 적용 |
| `pivot_simulated_network` | 모델링된 Network Edge를 따라 Reachability를 변경 |
| `collect_simulated_objective` | 사전 조건 충족 시 Objective를 완료 |
| `stop` | Episode를 자발적으로 종료 |

Observation에는 발견된 Host, 알려진 Port와 Service, Vulnerability 및 Credential 상태,
Privilege, Reachability, Detection Risk, 남은 Step Budget이 포함될 예정입니다. 범위를 벗어난
Action은 Masking하고 환경에서 거부합니다.

Reward는 설정 파일로 선택합니다. Sparse, Shaped, Risk-aware, Cost-aware 전략을 지원할
예정이며, 발견·취약점 검증·접근·권한 상승·목표 달성에는 Positive Reward를, 중복 Scan·실패
Action·노이즈 Action·탐지 이벤트·Step 비용에는 Negative Reward를 줄 수 있습니다. 모든 실험은
선택한 Reward 전략과 값을 기록합니다.

## 프로젝트 기반

- `src/` Layout 기반 Typed Python Package
- Gymnasium, NetworkX, NumPy, Pydantic 의존성 경계
- Stable-Baselines3, PyTorch, TensorBoard 선택 학습 의존성
- 선택적 FastAPI Dashboard 의존성 그룹
- Lint, Format, Type Check, Test를 재현하는 품질 명령
- Read-only 권한으로 동작하는 CI 품질 게이트

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

해당 단계에서 필요한 기능을 개발할 때만 추가 의존성을 설치합니다.

```bash
python -m pip install -e ".[dev,training]"
python -m pip install -e ".[dev,dashboard]"
```

`dev` Extra만으로 1단계 품질 검사를 수행할 수 있습니다. PyTorch와 Stable-Baselines3는
용량이 크고 Scenario·Environment 개발에는 필요하지 않으므로 `training` Extra를 분리했습니다.

## 실행 및 검증

현재 단계에는 Simulator나 Training Entry Point가 없습니다. 전체 품질 검사는 다음 명령으로
실행합니다.

```bash
make check
```

향후 장기 학습은 CI와 분리하여 명시적인 Research 명령으로 실행합니다. 실험 산출물에는
Seed, Scenario ID, Algorithm, Reward Strategy, Environment Version, Metric, Checkpoint,
TensorBoard 또는 MLflow Metadata를 포함합니다.

## 아키텍처

초기 Package는 `src/` Layout을 사용하고 Test는 Production Code와 분리합니다. Scenario,
Simulator, Agent, Training, Evaluation, Explainability 계층은 서로 독립적으로 테스트할 수
있어야 합니다. 자세한 내용은 [아키텍처 문서](docs/architecture.md)를 참고하세요.

```text
Scenario Schema/Generator -> 결정론적 Gymnasium Environment -> Agent
                                      |                         |
                                      +-> Metric/Evaluation <---+
                                      +-> Explanation/Graph Overlay
```

NetworkX는 Graph 관계를 모델링하고, Pydantic은 Scenario와 Experiment 데이터를 검증하며,
Gymnasium은 Environment API를 정의합니다. Stable-Baselines3와 PyTorch는 선택적 Training
Pipeline에서만 사용합니다. 어떤 Package 경계도 실제 Network Client나 Command Runner를
추가해서는 안 됩니다.

## 재현 가능한 실험

Scenario Generator, Environment Reset, Agent Evaluation에 고정 Seed를 사용합니다. Benchmark
보고서에는 Scenario 크기(`small`, `medium`, `large`), 난이도, Seed, Algorithm, Reward
Strategy, Training Budget, Evaluation Seed를 기록합니다.

핵심 Metric은 다음과 같습니다.

- Success Rate
- 평균 Step 수
- Cumulative Reward
- Detection Risk
- Path Cost
- Unseen Scenario 성능

CI에서는 짧은 Smoke Experiment만 실행하고, 장기 학습은 별도 Research 실행으로 분리합니다.

## 현재 한계

- Scenario Schema와 NetworkX Graph 변환이 구현되어 있습니다.
- Gymnasium Environment가 구현되어 있으며 실제 통신 없이 메모리 상태만 전이합니다.
- `small`, `medium`, `large` 크기와 `easy`, `medium`, `hard` 난이도의 Scenario를 고정
  Seed로 생성할 수 있습니다.
- Baseline Agent와 학습 Agent가 아직 없습니다.
- Training, Evaluation, Explainability, Integration Code가 아직 없습니다.
- 선언된 선택 의존성은 해당 단계가 시작될 때까지 CI에서 사용하지 않습니다.

## 보안 정책

이 플랫폼은 Simulation-only입니다. 실제 시스템 Discovery, Credential 사용, Persistence,
방어 제어 우회, Malware 배포, 파괴, Exfiltration은 범위 밖이며 금지됩니다.

[SECURITY.md](SECURITY.md)와 [Threat Model](docs/threat-model.md)을 함께 확인하세요.

## 로드맵

각 단계는 독립적으로 검토 가능한 작은 Commit으로 구현합니다.

1. 프로젝트 초기화 및 품질 게이트 (현재)
2. Graph Scenario Schema
3. Gymnasium 공격 경로 Environment
4. 결정론적 Scenario Generator
5. Baseline Agent
6. DQN Training Pipeline
7. PPO Benchmark Pipeline
8. 설정 가능한 Reward 실험
9. 재현 가능한 Evaluation
10. Policy Explainability
11. Sanitized ThreatGraph Adapter
12. 실험 방법론 문서

자세한 Commit 계획은 [로드맵](docs/roadmap.md)을 참고하세요.

## 기여

[CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요. 이 프로젝트는 MIT License를 따릅니다.
