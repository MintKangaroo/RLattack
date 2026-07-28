# RLAttack 인수인계

## 현재 상태

- 기준 브랜치: `develop`
- 현재 작업 브랜치: `feat/dqn-training`
- 원격 저장소: <https://github.com/MintKangaroo/RLattack>
- 브랜치 정책: `main`은 안정 버전, `develop`은 릴리스 통합, 기능은 `feat/*`에서 작업
- 실제 네트워크 통신·Subprocess·공격 도구 호출은 구현하지 않음

## 완료된 단계

1. 프로젝트 초기화 및 품질 게이트
2. Pydantic/NetworkX Graph Scenario Schema
3. 결정론적 Gymnasium 공격 경로 Environment
4. Seed 기반 `small`/`medium`/`large` Scenario Generator
5. Baseline Agent (completed)
6. DQN Training Pipeline (completed; optional dependencies)
7. PPO Benchmark Pipeline (completed; optional dependencies)
8. Configurable Reward Strategies (completed)
9. Reproducible Evaluation Benchmarks (completed)
10. Policy Explainability and Graph Overlay (completed)
11. Sanitized ThreatGraph Scenario Adapter (completed)

## 현재 변경사항

`src/rlattack/training.py`에 `DQNTrainingConfig`, `PPOTrainingConfig`, Optional Dependency
Probe, Stable-Baselines3 기반 `train_dqn`/`train_ppo`를 제공합니다. 두 알고리즘은 동일한
Vectorized Environment, Checkpoint, Evaluation Callback, TensorBoard 기록 계약을 공유합니다.
PyTorch와 Stable-Baselines3는 CI에서 설치하지 않으며, 장기 학습은 별도 명령으로 실행합니다.

`src/rlattack/agents.py`에 다음 Baseline을 제공합니다.

- `RandomAgent`: Action Mask 내 균등 무작위 선택
- `GreedyAgent`: 목표·권한·접근·검증 우선순위 선택
- `RuleBasedAgent`: 정찰부터 목표 수집까지 명시적 규칙 실행
- `ShortestPathOracle`: 시뮬레이션 Graph의 Entry-Objective 최단 경로를 참고하는 상한선

테스트 명령은 다음과 같습니다.

```bash
source .venv/bin/activate
make check
```

현재 기준 테스트는 26건이며 커버리지 100%를 목표로 합니다.

## 다음 작업

1. PPO 변경을 `feat/ppo-training`에 커밋하고 원격에 push
2. 기능 브랜치를 `develop`에 통합
3. `feat/reward-strategies` 생성
4. Sparse/Shaped/Risk-aware/Cost-aware Reward 전략 구현
5. ThreatGraph Adapter 구현 완료. 이후 실험 방법론·한계·윤리 문서 순서로 진행

## 주의사항

- `main` 또는 `develop`에 직접 push하지 않습니다.
- 기능 완료 시 `feat/*`에서 검증 후 Push하고 Pull Request로 `develop`에 통합합니다.
- 장기 학습은 CI에서 실행하지 않고 별도 Research 명령으로 분리합니다.
- Scenario의 보안 관련 Action은 모두 메모리 상태 전이이며 외부 대상과 연결하면 안 됩니다.
