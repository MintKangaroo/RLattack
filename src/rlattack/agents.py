"""Safe baseline policies for benchmarking the simulated attack environment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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


def reset_agent(agent: Agent, *, seed: int | None = None) -> None:
    """Reset episode-local policy state when an Agent exposes that optional hook."""

    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(seed=seed)


def target_count(info: dict[str, object]) -> int:
    """Read the per-action target stride the environment published in ``info``."""

    count = info.get("target_count")
    if not isinstance(count, int) or count < 1:
        raise ValueError("info must contain a positive target_count")
    return count


def valid_actions(info: dict[str, object]) -> np.ndarray[Any, Any]:
    """Return the flat indices of every action currently allowed by the mask."""

    mask = np.asarray(info.get("action_mask"), dtype=np.int8)
    stride = target_count(info)
    if mask.shape != (len(Action) * stride,):
        raise ValueError("info must contain an action_mask with one entry per action")
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        raise ValueError("action_mask contains no valid actions")
    return valid


def _first_valid(
    info: dict[str, object],
    action_type: Action,
    targets: Iterable[int] | None = None,
) -> np.int64 | None:
    """Return the lowest-index valid action of ``action_type``, optionally target-filtered."""

    mask = np.asarray(info["action_mask"], dtype=np.int8)
    stride = target_count(info)
    offset = int(action_type) * stride
    candidates = range(stride) if targets is None else targets
    for target in candidates:
        if 0 <= target < stride and mask[offset + target]:
            return np.int64(offset + target)
    return None


def _first_valid_by_priority(
    info: dict[str, object], priority: Sequence[Action]
) -> np.int64 | None:
    for action_type in priority:
        action = _first_valid(info, action_type)
        if action is not None:
            return action
    return None


@dataclass
class RandomAgent:
    """Uniformly sample one valid action, preferring progress over stopping.

    ``stop`` is excluded while any other action is available so that the baseline
    measures undirected exploration instead of an immediate exit.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, *, seed: int | None = None) -> None:
        """Start a reproducible random stream for a new episode."""

        episode_seed = self.seed if seed is None else seed
        self._rng = np.random.default_rng(episode_seed)

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        del observation
        valid = valid_actions(info)
        stop_offset = int(Action.STOP) * target_count(info)
        progress = valid[valid < stop_offset]
        if len(progress):
            valid = progress
        return np.int64(self._rng.choice(valid))


@dataclass
class GreedyAgent:
    """Choose the lowest-index target of the highest-priority valid action type."""

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

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the stateless policy (provided for the shared Agent protocol)."""

        del seed

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        del observation
        valid_actions(info)
        action = _first_valid_by_priority(info, self.priority)
        if action is None:
            raise RuntimeError("priority list does not cover the action space")
        return action


@dataclass
class RuleBasedAgent:
    """Follow an explicit reconnaissance-to-objective action ordering."""

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

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the stateless rule policy for a new episode."""

        del seed

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        del observation
        valid_actions(info)
        action = _first_valid_by_priority(info, self.rules)
        if action is None:
            raise RuntimeError("rule list does not cover the action space")
        return action


@dataclass
class ShortestPathOracle:
    """Advance along the graph's shortest entry-to-objective host path.

    The oracle sees the static simulated graph and is an upper-bound baseline; it never
    adapts to a real target.
    """

    scenario: Scenario
    route: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.scenario.hosts:
            raise ValueError("graph oracle requires at least one host")
        if not self.scenario.objectives:
            raise ValueError("graph oracle requires at least one objective")
        graph = self.scenario.to_networkx()
        hosts = {host.id for host in self.scenario.hosts}
        entry_host = (
            self.scenario.entry_host_ids[0]
            if self.scenario.entry_host_ids
            else self.scenario.hosts[0].id
        )
        self.route = self._plan_route(graph.subgraph(hosts), entry_host)
        self._host_index = {host.id: index for index, host in enumerate(self.scenario.hosts)}
        self._route_indices = tuple(self._host_index[host_id] for host_id in self.route)
        self._required_actions = {
            host.id: self._required_for_host(host.id) for host in self.scenario.hosts
        }

    def _plan_route(self, host_graph: Any, entry_host: str) -> tuple[str, ...]:
        """Chain shortest paths through every objective host, shallowest first.

        Routing only to the deepest objective is not enough: network shortcuts can skip
        past a shallower objective host entirely, leaving the oracle to backtrack.
        """

        try:
            depths = {
                objective.host_id: len(nx.shortest_path(host_graph, entry_host, objective.host_id))
                for objective in self.scenario.objectives
            }
        except nx.NetworkXNoPath as error:
            raise ValueError("graph oracle requires a path to the objective") from error
        route = [entry_host]
        current = entry_host
        for host_id in sorted(depths, key=lambda host: (depths[host], host)):
            segment = nx.shortest_path(host_graph, current, host_id)
            route.extend(segment[1:])
            current = host_id
        return tuple(route)

    def _credential_host(self, credential_id: str) -> str:
        return next(
            credential.host_id
            for credential in self.scenario.credentials
            if credential.id == credential_id
        )

    def _required_for_host(self, host_id: str) -> tuple[tuple[Action, tuple[int, ...]], ...]:
        """Return only the steps needed to earn a foothold on ``host_id``.

        Exploiting every service on a host is what makes the naive baselines trip the
        detection threshold, so the oracle touches just the credential path.
        """

        access = tuple(
            index
            for index, edge in enumerate(self.scenario.access_edges)
            if self._credential_host(edge.credential_id) == host_id
        )
        vulnerability_ids = {self.scenario.access_edges[index].vulnerability_id for index in access}
        vulnerabilities = tuple(
            index
            for index, record in enumerate(self.scenario.vulnerabilities)
            if record.id in vulnerability_ids
        )
        service_ids = {self.scenario.vulnerabilities[index].service_id for index in vulnerabilities}
        services = tuple(
            index for index, record in enumerate(self.scenario.services) if record.id in service_ids
        )
        return (
            (Action.ATTEMPT_SIMULATED_ACCESS, access),
            (Action.VALIDATE_VULNERABILITY, vulnerabilities),
            (Action.ENUMERATE_SERVICE, services),
            (Action.SCAN_SERVICE, services),
        )

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the stateless oracle for a new episode."""

        del seed

    def predict(self, observation: Observation, info: dict[str, object]) -> np.int64:
        valid_actions(info)
        collect = _first_valid(info, Action.COLLECT_SIMULATED_OBJECTIVE)
        if collect is not None:
            return collect
        escalate = _first_valid(info, Action.ESCALATE_SIMULATED_PRIVILEGE)
        if escalate is not None:
            return escalate
        reachable = observation["reachable_hosts"]
        for host_id, host_index in zip(self.route, self._route_indices, strict=True):
            if reachable[host_index]:
                action = self._advance_on_host(info, host_id)
                if action is not None:
                    return action
                continue
            action = _first_valid(info, Action.PIVOT_SIMULATED_NETWORK, (host_index,))
            if action is None:
                action = _first_valid(info, Action.DISCOVER_HOST, (host_index,))
            if action is not None:
                return action
        return GreedyAgent().predict(observation, info)

    def _advance_on_host(self, info: dict[str, object], host_id: str) -> np.int64 | None:
        """Exploit the current route host far enough to unlock the next pivot."""

        for action_type, targets in self._required_actions[host_id]:
            action = _first_valid(info, action_type, targets)
            if action is not None:
                return action
        return None


def action_name(action_type: int) -> str:
    """Return a stable human-readable name for one baseline action type."""

    if action_type < 0 or action_type >= len(ACTION_NAMES):
        raise ValueError("action is outside the RLAttack action catalogue")
    return ACTION_NAMES[action_type]
