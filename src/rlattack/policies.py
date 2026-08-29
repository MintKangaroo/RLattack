"""Adapters that expose trained Stable-Baselines3 policies as RLAttack agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Algorithm = Literal["dqn", "ppo", "maskable-ppo"]

Observation = dict[str, np.ndarray[Any, Any]]


@dataclass
class SB3PolicyAgent:
    """Wrap a Stable-Baselines3 model in the RLAttack :class:`~rlattack.agents.Agent` API.

    The wrapper deliberately does not repair invalid actions. A learned policy is scored
    on the same masked action space it was trained against, so masking its output here
    would report an ability the policy does not have.

    ``masked`` policies are a different case: they were *trained* with the environment's
    action mask as an input, so the mask is part of their interface and is forwarded at
    evaluation time rather than being a correction.
    """

    model: Any
    deterministic: bool = True
    name: str = "sb3"
    masked: bool = False

    def reset(self, *, seed: int | None = None) -> None:
        """Reset episode-local state (the wrapped policy is itself stateless)."""

        del seed

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        if self.masked:
            mask = np.asarray(info["action_mask"]).astype(bool)
            action, _ = self.model.predict(
                observation, deterministic=self.deterministic, action_masks=mask
            )
        else:
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.int64(np.asarray(action).reshape(-1)[0])


def load_policy(
    path: Path | str,
    algorithm: Algorithm = "dqn",
    *,
    deterministic: bool = True,
) -> SB3PolicyAgent:
    """Load a local checkpoint written by :mod:`rlattack.training`."""

    if algorithm not in ("dqn", "ppo", "maskable-ppo"):
        raise ValueError("algorithm must be 'dqn', 'ppo', or 'maskable-ppo'")
    checkpoint = Path(path)
    if algorithm == "maskable-ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as error:
            raise RuntimeError(
                "Masked policies require the optional '.[training]' dependencies"
            ) from error
        return SB3PolicyAgent(
            model=MaskablePPO.load(str(checkpoint)),
            deterministic=deterministic,
            name=algorithm,
            masked=True,
        )
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as error:
        raise RuntimeError(
            "Loading a trained policy requires the optional '.[training]' dependencies"
        ) from error
    loader = DQN if algorithm == "dqn" else PPO
    model = loader.load(str(checkpoint))
    return SB3PolicyAgent(model=model, deterministic=deterministic, name=algorithm)
