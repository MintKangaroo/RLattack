"""A small epsilon-greedy learner shared by the adaptive attacker and defender."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpsilonGreedy:
    """Epsilon-greedy selection over labelled arms with running-mean estimates.

    Both sides of the game see one scalar reward per episode, so there is nothing for a
    heavier learner to exploit; keeping the mechanism in one place keeps their reported
    behaviour comparable.
    """

    labels: Sequence[str]
    exploration: float = 0.15
    _rng: np.random.Generator = field(init=False, repr=False)
    _pulls: list[int] = field(init=False, default_factory=list, repr=False)
    _values: list[float] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("at least one arm is required")
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        self.reset()

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the estimates and restart the selection stream."""

        self._rng = np.random.default_rng(seed)
        self._pulls = [0] * len(self.labels)
        self._values = [0.0] * len(self.labels)

    @property
    def pulls(self) -> dict[str, int]:
        """Return how many episodes each arm was selected for."""

        return dict(zip(self.labels, self._pulls, strict=True))

    @property
    def values(self) -> dict[str, float]:
        """Return the mean reward estimated for each arm."""

        return dict(zip(self.labels, self._values, strict=True))

    @property
    def preferred(self) -> str:
        """Return the label with the highest estimate."""

        return self.labels[self._values.index(max(self._values))]

    def select(self) -> int:
        """Choose an arm, trying every arm once before preferring any of them."""

        untried = [index for index, count in enumerate(self._pulls) if count == 0]
        if untried:
            return untried[0]
        if self._rng.random() < self.exploration:
            return int(self._rng.integers(len(self.labels)))
        return self._values.index(max(self._values))

    def update(self, index: int, reward: float) -> None:
        """Fold one reward into the running mean of ``index``."""

        if not 0 <= index < len(self.labels):
            raise ValueError("arm index is outside the arm set")
        self._pulls[index] += 1
        self._values[index] += (reward - self._values[index]) / self._pulls[index]
