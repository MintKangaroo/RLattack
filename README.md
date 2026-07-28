# RLAttack

RLAttack은 실제 외부 시스템을 공격하지 않고, **결정론적 시뮬레이션 네트워크와 취약점
그래프** 안에서 강화학습 Agent의 공격 경로 탐색 전략을 연구하는 플랫폼입니다.

공격 계획을 재현 가능한 MDP(Markov Decision Process)로 모델링합니다. Agent는 현재 Episode
상태를 관찰하고 하나의 시뮬레이션 Action을 선택한 뒤 Reward를 받습니다. 목표 달성, 자발적
중단, Step Budget 소진 중 하나가 발생하면 Episode가 끝납니다. Scenario와 Seed가 같으면
동일한 Trajectory가 생성되므로 DQN·PPO·Baseline의 비교 실험을 재현할 수 있습니다.

> 현재 상태: 6단계 DQN Training Pipeline을 구현 중입니다. Scenario Schema, Gymnasium
> Environment, Scenario Generator, Baseline Agent는 구현되어 있습니다.

![RLAttack 시뮬레이션 연구 플랫폼 개요](docs/assets/rlattack-overview.svg)

*위 그림은 현재 구조를 설명하는 아키텍처 개요입니다. 실제 외부 시스템을 대상으로 하는
화면이나 공격 도구의 Screenshot은 프로젝트 범위에 포함되지 않습니다.*

## 핵심 원칙

| 원칙 | 내용 |
| --- | --- |
| 시뮬레이션 전용 | 모든 Action은 In-process 상태 전이이며 외부 통신을 수행하지 않음 |
| 결정론 | 고정 Seed와 Scenario로 동일한 Trajectory 재현 |
| 비교 가능성 | 동일 Environment·Metric으로 Baseline, DQN, PPO 비교 |
| 설명 가능성 | 선택 Action, 방문 경로, Reward, Risk를 기록할 수 있는 구조 |
| 안전한 확장 | Sanitized Scenario와 로컬 Cyber Range Adapter만 허용 |

## 안전 범위

RLAttack은 다음 기능을 구현하거나 호출하지 않습니다.

- Nmap 또는 네트워크 Scanner
- Exploit Framework, Payload, Malware 실행
- 실제 Credential 인증·수집
- 로컬·원격 Shell 및 임의 Subprocess 실행
- Persistence, Evasion, 방어 제어 우회
- 공용 네트워크 또는 실시간 대상 연결

`scan_service`, `attempt_simulated_access`, `pivot_simulated_network` 등의 이름은 연구
문제를 설명하기 위한 Domain Action입니다. 이 Action은 Scenario Graph와 Episode 상태만
변경합니다.

## 현재 구현

### 1. Graph Scenario Schema

Pydantic 모델과 NetworkX 변환을 제공합니다.

- `Host`, `Service`, `Vulnerability`
- `Credential`, `Privilege`, `Objective`
- `SecurityControl`
- `NetworkEdge`, `AccessEdge`, `PrivilegeEdge`
- ID 중복 및 참조 무결성 검증
- Port, Severity, Probability, Cost 범위 검증

### 2. Gymnasium Environment

`AttackPathEnv`는 메모리 기반 상태만 사용합니다.

- Observation: Host, Service, Vulnerability, Credential, Privilege, Reachability
- Detection Risk 및 남은 Step Budget
- Action Mask와 잘못된 Action 거부
- `terminated`와 `truncated` 구분
- `reset(seed=...)` 기반 재현성
- Gymnasium Environment Checker 테스트

### 3. 결정론적 Scenario Generator

```python
from rlattack.generator import generate_scenario

scenario = generate_scenario(size="medium", difficulty="hard", seed=42)
```

지원 조합:

- 크기: `small`, `medium`, `large`
- 난이도: `easy`, `medium`, `hard`
- 고정 Seed에 따른 동일 Graph와 Risk 값
- 난이도에 따른 Network Shortcut과 Security Control 변화

### 4. Baseline Agent

- `RandomAgent`: 유효한 Action Mask에서 균등 선택
- `GreedyAgent`: 목표·권한 상승·접근·검증을 우선 선택
- `RuleBasedAgent`: 정찰부터 목표 수집까지 명시적 순서 실행
- `ShortestPathOracle`: 시뮬레이션 Graph의 Entry-Objective 경로를 참고하는 상한선

## Environment Action

| Action | 시뮬레이션 효과 |
| --- | --- |
| `discover_host` | 도달 가능한 Host를 발견 |
| `scan_service` | Host의 Port와 Service 정보를 확인 |
| `enumerate_service` | Service 상세 정보를 Observation에 추가 |
| `validate_vulnerability` | 모델링된 Vulnerability를 검증 |
| `attempt_simulated_access` | Access Edge와 Credential 전이를 시도 |
| `escalate_simulated_privilege` | Privilege 전이를 적용 |
| `pivot_simulated_network` | Network Edge를 따라 Reachability를 변경 |
| `collect_simulated_objective` | 사전 조건을 만족하면 Objective 완료 |
| `stop` | Episode를 자발적으로 종료 |

## Reward 설계 방향

Reward 값은 코드에 고정하지 않고 실험 설정으로 관리할 예정입니다.

| 구분 | 예시 |
| --- | --- |
| Positive | Host 발견, Vulnerability 검증, Access, Privilege 상승, Objective 달성 |
| Negative | 중복 Scan, 실패 Action, 노이즈 Action, Detection Risk, Step Cost |
| 전략 | Sparse, Shaped, Risk-aware, Cost-aware |

각 실험은 Reward 전략, 파라미터, Scenario ID, Seed를 함께 기록합니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

학습 또는 Dashboard 개발 시에만 선택 Extra를 설치합니다.

```bash
# Stable-Baselines3, PyTorch, TensorBoard
python -m pip install -e ".[dev,training]"

# 선택적 로컬 Dashboard
python -m pip install -e ".[dev,dashboard]"
```

`training` Extra는 용량이 큰 PyTorch를 포함하므로 기본 개발 설치와 분리되어 있습니다.

## 검증

```bash
make check
```

현재 품질 게이트는 다음을 수행합니다.

- Ruff lint 및 Format 검사
- strict mypy Type Check
- pytest
- `rlattack` 패키지 커버리지 100%
- Gymnasium Environment Checker

CI에서는 짧은 Smoke Test만 실행하고 장기 Training은 별도 Research 명령으로 실행합니다.

## 아키텍처

```text
Scenario Schema / Generator
            |
            v
결정론적 Gymnasium Environment <---- Baseline / DQN / PPO Agent
            |
            +---- Evaluation / Metrics
            +---- Explainability / Graph Overlay
            +---- Checkpoint / TensorBoard / MLflow
```

주요 Package 역할:

- `rlattack.scenario`: 검증된 Graph Domain 모델
- `rlattack.generator`: 고정 Seed Scenario 생성
- `rlattack.env`: Gymnasium 상태 전이와 Action Mask
- `rlattack.agents`: 비교용 Baseline Policy
- `rlattack.training`: 선택적 DQN/PPO 학습 Pipeline

Training·Dashboard 계층은 Core Domain에 의존하지만, Core는 Stable-Baselines3·FastAPI·외부
SDK에 의존하지 않습니다.

## 재현 가능한 Benchmark

Benchmark 보고서에는 다음 정보를 기록합니다.

- Scenario 크기와 난이도
- Scenario ID와 Generator Seed
- Algorithm 및 Reward Strategy
- Training Budget과 Evaluation Seed
- Environment 버전과 의존성 버전

핵심 Metric은 Success Rate, 평균 Step, Cumulative Reward, Detection Risk, Path Cost,
Unseen Scenario 성능입니다.

## 브랜치 운영

- `main`: 항상 실행 가능한 안정 버전
- `develop`: 다음 Release 통합
- `feat/<기능명>`: 기능별 작업
- `fix/<문제명>`: Bug Fix
- `docs/<문서명>`: 문서 변경

기능은 `develop`에서 분기하고 기능 브랜치에서 검증한 뒤 Pull Request로 `develop`에
통합합니다. `main`과 `develop`에 직접 Push하지 않습니다.

## 로드맵

1. 프로젝트 초기화 및 품질 게이트 — 완료
2. Graph Scenario Schema — 완료
3. Gymnasium 공격 경로 Environment — 완료
4. 결정론적 Scenario Generator — 완료
5. Baseline Agent — 완료
6. DQN Training Pipeline — 진행 중
7. PPO Benchmark Pipeline
8. 설정 가능한 Reward 실험
9. 재현 가능한 Evaluation
10. Policy Explainability 및 Graph Overlay
11. Sanitized ThreatGraph Scenario Adapter
12. 실험 방법론·한계·윤리·안전 문서

세부 계획은 [docs/roadmap.md](docs/roadmap.md), 다음 세션 인수인계는
[HANDOFF.md](HANDOFF.md)를 참고하세요.

## 기여 및 보안

기여 방법은 [CONTRIBUTING.md](CONTRIBUTING.md), 보안 범위는 [SECURITY.md](SECURITY.md),
Threat Model은 [docs/threat-model.md](docs/threat-model.md)을 참고하세요.

이 프로젝트는 MIT License를 따릅니다.
