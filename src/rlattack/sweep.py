"""Hyperparameter trials for curriculum training.

A published policy trained on one configuration is a floor, not a ceiling. These trials
make the obvious knobs explicit so a run can say which settings it searched instead of
leaving them implicit in the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlattack.training import PPOTrainingConfig


@dataclass(frozen=True)
class SweepTrial:
    """One named training configuration."""

    label: str
    learning_rate: float = 3e-4
    rollout_steps: int = 256
    batch_size: int = 64
    entropy_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a sweep trial needs a label")

    def config(self, *, seed: int, output_dir: Path) -> PPOTrainingConfig:
        """Return the training configuration for this trial."""

        return PPOTrainingConfig(
            seed=seed,
            output_dir=output_dir / self.label,
            tensorboard_log=output_dir / self.label / "tb",
            rollout_steps=self.rollout_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            entropy_coefficient=self.entropy_coefficient,
        )


DEFAULT_TRIALS: tuple[SweepTrial, ...] = (
    SweepTrial("baseline"),
    SweepTrial("fast-lr", learning_rate=1e-3),
    SweepTrial("slow-lr", learning_rate=1e-4),
    SweepTrial("explorer", entropy_coefficient=0.01),
    SweepTrial("long-rollout", rollout_steps=1024, batch_size=128),
)


def trials_by_label(labels: tuple[str, ...] | None = None) -> tuple[SweepTrial, ...]:
    """Select trials by label, defaulting to the whole grid."""

    if labels is None:
        return DEFAULT_TRIALS
    known = {trial.label: trial for trial in DEFAULT_TRIALS}
    missing = [label for label in labels if label not in known]
    if missing:
        raise ValueError(f"unknown sweep trial(s): {', '.join(missing)}")
    return tuple(known[label] for label in labels)
