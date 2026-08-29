# RLAttack 실험 방법론

## 1. 목적과 범위

RLAttack의 목적은 실제 인프라에 대한 공격 자동화가 아니라, 합성 Network/Vulnerability
Graph에서 공격 경로 탐색 정책을 비교하는 것입니다. 모든 실험은 In-process Simulator에서
실행하며 Nmap, Exploit Framework, Remote Shell, 외부 Network Client를 사용하지 않습니다.

## 2. 문제 정의

각 Episode는 다음 MDP로 정의합니다.

\[
M = (S, A, P, R, \gamma)
\]

- `S`: 발견 Host, 알려진 Service, 검증 Vulnerability, Credential·Privilege 상태,
  Reachability, Detection Risk, 남은 Step Budget
- `A`: (Action Type × Target) 쌍. Action Type은 `discover_host`, `scan_service`,
  `enumerate_service`, `validate_vulnerability`, `attempt_simulated_access`,
  `escalate_simulated_privilege`, `pivot_simulated_network`,
  `collect_simulated_objective`, `stop`이고, Target은 Graph의 Host, Service,
  Vulnerability, Access Edge, Privilege Edge, Objective Index입니다.
  Flat Encoding은 `action_type * target_count + target_index`입니다.
- `P`: Scenario Graph와 현재 Episode 상태에 따른 상태 전이. Exploitation Action
  (`validate_vulnerability`, `attempt_simulated_access`,
  `escalate_simulated_privilege`)은 `Vulnerability.exploitability`에서 유도한 확률로
  성공하며, 실패는 Detection Risk를 증가시킵니다.
- `R`: Reward Strategy가 제공하는 탐색·비용·위험 기반 보상
- `γ`: 학습 알고리즘이 사용하는 Discount Factor

### 2.1 Action Parameterization

Action에 Target을 포함하는 것은 본 환경의 핵심 설계 결정입니다. Action Type만 선택하고
Environment가 대상 Node를 내부 순서로 고르면, 경로 선택이 정책이 아니라 구현 세부사항이
되어 모든 정책이 동일한 Trajectory로 수렴합니다. `v0.3.0` 이전 버전이 이 문제를 가지고
있었으며, 그 결과 Baseline 간 비교가 성립하지 않았습니다.

### 2.2 Reproducibility

모든 난수는 Seed된 `np_random` Stream에서 추출합니다. 따라서 전이가 확률적이어도
동일한 Scenario와 Seed는 동일한 Trajectory를 생성합니다. `DynamicsConfig.deterministic()`
은 모든 Valid Action이 성공하는 회귀 검증 모드입니다.

## 3. Scenario와 Environment

Scenario는 Pydantic으로 검증하고 NetworkX Graph로 변환합니다. Generator는 `small`,
`medium`, `large` 크기와 `easy`, `medium`, `hard` 난이도를 제공합니다. 외부 자료를
사용하는 경우에는 `adapter.py`의 Sanitized Import만 허용합니다.

Environment는 Gymnasium `reset`/`step` 계약을 따릅니다. Action Mask는 현재 상태에서 가능한
(Action Type, Target) 조합을 표시하고, Mask에 없는 Action은 상태를 변경하지 않고 Negative
Reward를 반환합니다. 목표 달성, `stop`, Detection Threshold 초과는 `terminated`,
Budget 소진은 `truncated`로 기록합니다. Episode 성공은 마지막 Action 이름이 아니라
`info["objective_captured"]` 상태로 판정합니다.

Generator가 만드는 모든 (Size, Difficulty, Seed) 조합은 Oracle이 Objective에 도달할 수
있어야 하며, 이는 Integration Test로 검증합니다. 도달 불가능한 Graph가 섞이면 보고되는
Success Rate 상한이 조용히 낮아지기 때문입니다.

## 4. Reward 실험

동일 Scenario·Seed·Training Budget에서 다음 전략을 비교합니다.

| 전략 | 목적 |
| --- | --- |
| Sparse | Objective 달성 중심의 최소 보상 |
| Shaped | 발견·검증·접근·권한 상승의 중간 보상 |
| Risk-aware | Detection Risk 증가를 크게 감점 |
| Cost-aware | Step과 중복 Action 비용을 크게 감점 |

Reward 설정과 함께 Scenario ID, Seed, Environment 버전, Algorithm을 저장합니다.

## 5. Agent 비교 프로토콜

비교 대상은 Random, Greedy, Rule-based, Shortest-path Oracle, DQN, PPO입니다.

1. Generator Seed를 고정하여 Training Scenario를 생성합니다.
2. 각 Algorithm에 동일한 Observation Space, Action Space, Step Budget을 적용합니다.
3. Evaluation Seed 목록을 공유합니다. **각 Seed는 Scenario를 새로 생성합니다.**
   고정된 하나의 Graph를 반복 실행하면 결정론적 정책의 분산이 0이 되어 평균이
   표본 크기 1의 결과가 됩니다.
4. 장기 학습은 CI와 분리하고 Checkpoint와 TensorBoard 기록을 보존합니다.
5. Unseen Scenario 크기·난이도 조합으로 일반화 성능을 측정합니다.
6. 학습된 Policy는 `rlattack.policies.SB3PolicyAgent`로 감싸 Baseline과 동일한
   Protocol에서 평가합니다. Wrapper는 Invalid Action을 교정하지 않습니다.

### 5.1 보고 지표

각 Agent에 대해 다음을 함께 보고합니다.

| 지표 | 설명 |
| --- | --- |
| `success_rate` | Objective 획득 Episode 비율 |
| `detection_rate` | Detection Threshold 초과로 종료된 Episode 비율 |
| `mean_steps` / `std_steps` | Episode 길이의 평균과 표준편차 |
| `mean_reward` / `std_reward` | 누적 보상의 평균과 표준편차 |
| `reward_ci_low` / `reward_ci_high` | 95% 정규근사 신뢰구간 |
| `mean_path_cost` | 실제 이동한 Network Edge Weight 합의 평균 |

평균만 보고하는 것은 표본 크기와 분산을 감추므로 허용하지 않습니다. Episode 단위 원본
Record는 `rlattack benchmark --output`으로 JSONL/CSV로 내보내 외부에서 재분석합니다.

## 6. Metric

- `success_rate`: Objective를 수집한 Episode 비율
- `mean_steps`: Episode당 평균 Step
- `mean_reward`: 누적 Reward 평균
- `mean_detection_risk`: Episode 종료 시 Detection Risk 평균
- `mean_path_cost`: Episode에서 새 Host로 진행할 때 사용한 Network Edge 가중치의 합
- `unseen_scenario_performance`: 학습에 사용하지 않은 Scenario 결과

현재 `rlattack.evaluation`은 동일 Seed 묶음에 대한 기본 Metric 집계를 제공합니다.

## 7. 설명 가능성

각 Decision은 Action 이름, 유효성, 관찰된 Host·Service·Privilege 수, Risk, 남은 Budget,
Reward 기여를 기록할 수 있습니다. `EpisodeTrace`는 순서가 있는 Decision 목록과 누적
Reward를 저장하고, Graph Overlay용 익명 Node Metadata를 생성합니다. DQN의 Q-value나
PPO의 Action Probability가 존재할 때 Optional 필드로 추가합니다.

## 8. 재현성과 산출물

실험 산출물에는 최소한 다음을 포함합니다.

- Git Commit 또는 Environment 버전
- Scenario 크기·난이도·Seed
- Algorithm·Reward Strategy·Hyperparameter
- Training/Evaluation Seed
- Checkpoint, Metric, 설명 Trace

CI는 짧은 Smoke Test만 실행합니다. PyTorch·Stable-Baselines3가 필요한 장기 학습은
`.[training]` Extra를 설치한 별도 실행 환경에서 수행합니다.

## 9. 한계

- 합성 Graph의 구조와 난이도가 실제 조직 Network의 다양성을 대표하지 않습니다.
- Path Cost는 Scenario의 합성 Edge 가중치이며 현실의 운영 비용을 의미하지 않습니다.
- 학습 성능은 Seed, Reward Scale, Scenario Generator에 민감할 수 있습니다.
- Explainability는 정책의 관찰 근거를 보여주며 인과적 설명이나 보안 보장을 의미하지 않습니다.
- ThreatGraph Adapter는 실행 가능한 Exploit이나 실제 대상 식별자를 지원하지 않습니다.

## 10. 윤리와 안전

이 저장소의 모든 Security Action은 시뮬레이션 상태 전이로 제한됩니다. 실제 대상에 대한
Discovery, Credential 사용, Exploit, Persistence, Evasion, Malware, 파괴, Exfiltration을
추가하는 변경은 허용하지 않습니다. 실험 데이터는 익명화하고 실제 Credential, Token,
IP, Domain, 개인 정보, 운영 환경 자료를 Commit하지 않습니다.
