"""Safe baseline policies for benchmarking the simulated attack environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx
import numpy as np

from rlattack.env import ACTION_NAMES, Action
from rlattack.scenario import Scenario

Observation = dict[str, np.ndarray[Any, Any]]


class Agent(Protocol):
    """Protocol shared by baseline and learned policies."""

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        """Select one valid action for the current state."""


def _valid_actions(info: dict[str, object]) -> np.ndarray[Any, Any]:
    mask = np.asarray(info.get("action_mask"), dtype=np.int8)
    if mask.shape != (len(Action),):
        raise ValueError("info must contain an action_mask with one entry per action")
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        raise ValueError("action_mask contains no valid actions")
    return valid


@dataclass
class RandomAgent:
    """Uniformly sample one action from the environment's valid-action mask."""

    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        del observation
        valid = _valid_actions(info)
        return np.int64(self._rng.choice(valid))


@dataclass
class GreedyAgent:
    """Choose the highest-priority currently valid progress action."""

    priority: tuple[Action, ...] = (
        Action.COLLECT_SIMULATED_OBJECTIVE,
        Action.ESCALATE_SIMULATED_PRIVILEGE,
        Action.ATTEMPT_SIMULATED_ACCESS,
        Action.VALIDATE_VULNERABILITY,
        Action.ENUMERATE_SERVICE,
        Action.SCAN_SERVICE,
        Action.PIVOT_SIMULATED_NETWORK,
        Action.DISCOVER_HOST,
        Action.STOP,
    )

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        del observation
        valid = set(int(action) for action in _valid_actions(info))
        for action in self.priority:
            if int(action) in valid:
                return np.int64(action)
        raise RuntimeError("priority list does not cover the action space")


@dataclass
class RuleBasedAgent:
    """Follow an explicit reconnaissance-to-objective action sequence."""

    rules: tuple[Action, ...] = (
        Action.DISCOVER_HOST,
        Action.SCAN_SERVICE,
        Action.ENUMERATE_SERVICE,
        Action.VALIDATE_VULNERABILITY,
        Action.ATTEMPT_SIMULATED_ACCESS,
        Action.ESCALATE_SIMULATED_PRIVILEGE,
        Action.PIVOT_SIMULATED_NETWORK,
        Action.COLLECT_SIMULATED_OBJECTIVE,
        Action.STOP,
    )

    def __post_init__(self) -> None:
        self._cursor = 0

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        del observation
        valid = set(int(action) for action in _valid_actions(info))
        for index in range(self._cursor, len(self.rules)):
            action = self.rules[index]
            if int(action) in valid:
                self._cursor = index + 1
                return np.int64(action)
        return np.int64(Action.STOP)


@dataclass
class ShortestPathOracle:
    """Use the graph's shortest entry-to-objective path to guide pivot actions.

    The oracle sees the static simulated graph and is intended as an upper-bound baseline,
    never as an adapter to a real target.
    """

    scenario: Scenario

    def __post_init__(self) -> None:
        graph = self.scenario.to_networkx()
        hosts = {host.id for host in self.scenario.hosts}
        self._path_hosts = tuple(
            node
            for node in nx.shortest_path(
                graph.subgraph(hosts),
                self.scenario.entry_host_ids[0],
                self.scenario.objectives[0].host_id,
            )
        )

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        valid = set(int(action) for action in _valid_actions(info))
        discovered = observation["discovered_hosts"]
        if not np.all(discovered):
            return np.int64(
                Action.PIVOT_SIMULATED_NETWORK
                if int(Action.PIVOT_SIMULATED_NETWORK) in valid
                else Action.DISCOVER_HOST
            )
        return GreedyAgent().predict(observation, info)


def action_name(action: int) -> str:
    """Return a stable human-readable name for a baseline decision."""

    if action < 0 or action >= len(ACTION_NAMES):
        raise ValueError("action is outside the RLAttack action catalogue")
    return ACTION_NAMES[action]
