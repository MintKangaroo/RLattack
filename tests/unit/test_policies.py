import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rlattack.policies import SB3PolicyAgent, load_policy


class StubModel:
    def __init__(self, action: int = 3) -> None:
        self.action = action
        self.calls: list[bool] = []

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, None]:
        del observation
        self.calls.append(deterministic)
        return np.array([self.action]), None


class StubAlgorithm:
    loaded: list[str] = []

    @classmethod
    def load(cls, path: str) -> StubModel:
        cls.loaded.append(path)
        return StubModel()


class StubMaskableModel(StubModel):
    def __init__(self, action: int = 7) -> None:
        super().__init__(action)
        self.masks: list[Any] = []

    def predict(
        self, observation: Any, deterministic: bool = True, action_masks: Any = None
    ) -> tuple[Any, None]:
        self.masks.append(action_masks)
        return super().predict(observation, deterministic)


def stub_sb3_module() -> types.ModuleType:
    module = types.ModuleType("stable_baselines3")
    module.DQN = StubAlgorithm  # type: ignore[attr-defined]
    module.PPO = StubAlgorithm  # type: ignore[attr-defined]
    return module


class StubMaskableAlgorithm:
    loaded: list[str] = []

    @classmethod
    def load(cls, path: str) -> StubMaskableModel:
        cls.loaded.append(path)
        return StubMaskableModel()


def stub_contrib_module() -> types.ModuleType:
    module = types.ModuleType("sb3_contrib")
    module.MaskablePPO = StubMaskableAlgorithm  # type: ignore[attr-defined]
    return module


def test_policy_agent_returns_the_raw_model_action() -> None:
    model = StubModel(action=5)
    agent = SB3PolicyAgent(model=model, name="dqn")
    agent.reset(seed=1)

    action = agent.predict({"discovered_hosts": np.zeros(1, dtype=np.int8)}, {"target_count": 2})

    assert action == np.int64(5)
    assert model.calls == [True]


def test_load_policy_uses_the_requested_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    StubAlgorithm.loaded.clear()
    monkeypatch.setitem(sys.modules, "stable_baselines3", stub_sb3_module())

    agent = load_policy(Path("artifacts/final.zip"), "ppo", deterministic=False)

    assert agent.name == "ppo"
    assert agent.deterministic is False
    assert StubAlgorithm.loaded == ["artifacts/final.zip"]


def test_load_policy_rejects_unknown_algorithms() -> None:
    with pytest.raises(ValueError, match="maskable-ppo"):
        load_policy("model.zip", "a2c")  # type: ignore[arg-type]


def test_load_policy_requires_the_optional_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)

    with pytest.raises(RuntimeError, match="training"):
        load_policy("model.zip", "dqn")


def test_a_masked_policy_receives_the_action_mask() -> None:
    model = StubMaskableModel(action=7)
    agent = SB3PolicyAgent(model=model, name="maskable-ppo", masked=True)
    mask = np.array([0, 1, 1, 0], dtype=np.int8)

    action = agent.predict({}, {"action_mask": mask, "target_count": 2})

    assert action == np.int64(7)
    assert model.masks[0].tolist() == [False, True, True, False]


def test_load_policy_supports_maskable_ppo(monkeypatch: pytest.MonkeyPatch) -> None:
    StubMaskableAlgorithm.loaded.clear()
    monkeypatch.setitem(sys.modules, "sb3_contrib", stub_contrib_module())

    agent = load_policy(Path("artifacts/final.zip"), "maskable-ppo")

    assert agent.masked is True
    assert agent.name == "maskable-ppo"
    assert StubMaskableAlgorithm.loaded == ["artifacts/final.zip"]


def test_maskable_policies_require_the_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sb3_contrib", None)

    with pytest.raises(RuntimeError, match="Masked policies"):
        load_policy("model.zip", "maskable-ppo")
