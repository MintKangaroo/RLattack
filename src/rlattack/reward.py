"""Configurable reward strategies for controlled RLAttack experiments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rlattack.env import RewardConfig

RewardStrategy = Literal["sparse", "shaped", "risk-aware", "cost-aware"]


class RewardExperiment(BaseModel):
    """Serializable record of the reward policy used by an experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: RewardStrategy
    scenario_id: str = Field(min_length=1)
    seed: int
    values: RewardConfig


def build_reward_config(strategy: RewardStrategy) -> RewardConfig:
    """Return an explicit reward configuration for one controlled ablation."""

    if strategy == "sparse":
        return RewardConfig(
            new_host=0.0,
            new_service=0.0,
            validated_vulnerability=0.0,
            access=0.0,
            privilege_escalation=0.0,
            objective=1.0,
            duplicate_or_invalid=-0.1,
            detection_risk=0.0,
            step_cost=0.0,
        )
    if strategy == "risk-aware":
        return RewardConfig(detection_risk=-2.0, step_cost=-0.03)
    if strategy == "cost-aware":
        return RewardConfig(step_cost=-0.2, duplicate_or_invalid=-1.5)
    if strategy == "shaped":
        return RewardConfig()
    raise ValueError(f"unsupported reward strategy: {strategy}")


def make_experiment_record(
    strategy: RewardStrategy,
    scenario_id: str,
    seed: int,
) -> RewardExperiment:
    """Create a record that can be serialized with ``model_dump_json``."""

    return RewardExperiment(
        strategy=strategy,
        scenario_id=scenario_id,
        seed=seed,
        values=build_reward_config(strategy),
    )
