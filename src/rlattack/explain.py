"""Structured explanations for simulator decisions and episode traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rlattack.env import ACTION_NAMES, Action
from rlattack.scenario import Scenario


class ActionExplanation(BaseModel):
    """Human- and machine-readable explanation for one selected Action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    valid: bool
    reasons: tuple[str, ...]
    observation_summary: dict[str, int | float]
    reward_contribution: float
    action_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    q_value: float | None = None


def explain_action(
    observation: dict[str, np.ndarray[Any, Any]],
    action: int,
    reward: float,
    info: dict[str, Any],
    *,
    action_probability: float | None = None,
    q_value: float | None = None,
) -> ActionExplanation:
    """Explain an action from observable state only; no hidden target data is used."""

    if action < 0 or action >= len(ACTION_NAMES):
        raise ValueError("action is outside the RLAttack action catalogue")
    mask = np.asarray(info.get("action_mask"), dtype=np.int8)
    valid = bool(mask[action]) if mask.shape == (len(Action),) else False
    discovered = int(np.sum(observation["discovered_hosts"]))
    services = int(np.sum(observation["known_services"]))
    privileges = int(np.sum(observation["acquired_privileges"]))
    risk = float(observation["detection_risk"][0])
    budget = float(observation["steps_remaining"][0])
    reasons = (
        f"discovered_hosts={discovered}",
        f"known_services={services}",
        f"acquired_privileges={privileges}",
        f"detection_risk={risk:.3f}",
        f"steps_remaining={budget:.0f}",
    )
    return ActionExplanation(
        action=ACTION_NAMES[action],
        valid=valid,
        reasons=reasons,
        observation_summary={
            "discovered_hosts": discovered,
            "known_services": services,
            "acquired_privileges": privileges,
            "detection_risk": risk,
            "steps_remaining": budget,
        },
        reward_contribution=reward,
        action_probability=action_probability,
        q_value=q_value,
    )


@dataclass
class EpisodeTrace:
    """Append-only trace suitable for reports and graph overlays."""

    records: list[ActionExplanation] = field(default_factory=list)

    def append(self, record: ActionExplanation) -> None:
        self.records.append(record)

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(record.action for record in self.records)

    @property
    def cumulative_reward(self) -> float:
        return sum(record.reward_contribution for record in self.records)

    def graph_overlay(self, scenario: Scenario) -> list[dict[str, object]]:
        """Return sanitized node metadata for a visualization layer."""

        graph = scenario.to_networkx()
        return [
            {
                "id": str(node),
                "kind": str(data.get("kind", "unknown")),
                "visited": node in self.actions,
            }
            for node, data in graph.nodes(data=True)
        ]
