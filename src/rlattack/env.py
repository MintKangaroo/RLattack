"""Gymnasium environment for safe, deterministic attack-path simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlattack.scenario import Scenario, Service, Vulnerability

Observation = dict[str, np.ndarray[Any, Any]]


class Action(IntEnum):
    """Stable action IDs exposed by :class:`AttackPathEnv`."""

    DISCOVER_HOST = 0
    SCAN_SERVICE = 1
    ENUMERATE_SERVICE = 2
    VALIDATE_VULNERABILITY = 3
    ATTEMPT_SIMULATED_ACCESS = 4
    ESCALATE_SIMULATED_PRIVILEGE = 5
    PIVOT_SIMULATED_NETWORK = 6
    COLLECT_SIMULATED_OBJECTIVE = 7
    STOP = 8


ACTION_NAMES: tuple[str, ...] = (
    "discover_host",
    "scan_service",
    "enumerate_service",
    "validate_vulnerability",
    "attempt_simulated_access",
    "escalate_simulated_privilege",
    "pivot_simulated_network",
    "collect_simulated_objective",
    "stop",
)


@dataclass(frozen=True)
class RewardConfig:
    """Configurable baseline reward values for the simulator."""

    new_host: float = 1.0
    new_service: float = 0.5
    validated_vulnerability: float = 1.0
    access: float = 2.0
    privilege_escalation: float = 3.0
    objective: float = 10.0
    duplicate_or_invalid: float = -1.0
    detection_risk: float = -0.5
    step_cost: float = -0.01


class AttackPathEnv(gym.Env[Observation, np.int64]):
    """Deterministic in-memory attack graph environment.

    The environment never opens sockets, executes subprocesses, or invokes offensive
    tooling. Each action only updates arrays derived from a validated :class:`Scenario`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, scenario: Scenario, step_budget: int = 50, reward_config: RewardConfig | None = None
    ) -> None:
        if step_budget < 1:
            raise ValueError("step_budget must be positive")
        if not scenario.hosts:
            raise ValueError("scenario must contain at least one host")
        self.scenario = scenario
        self.step_budget = step_budget
        self.reward_config = reward_config or RewardConfig()
        self._host_index = {record.id: index for index, record in enumerate(scenario.hosts)}
        self._service_index = {record.id: index for index, record in enumerate(scenario.services)}
        self._vulnerability_index = {
            record.id: index for index, record in enumerate(scenario.vulnerabilities)
        }
        self._credential_index = {
            record.id: index for index, record in enumerate(scenario.credentials)
        }
        self._privilege_index = {
            record.id: index for index, record in enumerate(scenario.privileges)
        }
        self._entry_hosts = set(scenario.entry_host_ids or (scenario.hosts[0].id,))
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "discovered_hosts": spaces.MultiBinary(len(scenario.hosts)),
                "known_services": spaces.MultiBinary(len(scenario.services)),
                "validated_vulnerabilities": spaces.MultiBinary(len(scenario.vulnerabilities)),
                "acquired_credentials": spaces.MultiBinary(len(scenario.credentials)),
                "acquired_privileges": spaces.MultiBinary(len(scenario.privileges)),
                "reachable_hosts": spaces.MultiBinary(len(scenario.hosts)),
                "detection_risk": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "steps_remaining": spaces.Box(0.0, step_budget, shape=(1,), dtype=np.float32),
            }
        )
        self._reset_state()

    def _reset_state(self) -> None:
        host_count = len(self.scenario.hosts)
        self._discovered_hosts = np.zeros(host_count, dtype=np.int8)
        self._known_services = np.zeros(len(self.scenario.services), dtype=np.int8)
        self._enumerated_services = np.zeros(len(self.scenario.services), dtype=np.int8)
        self._validated_vulnerabilities = np.zeros(
            len(self.scenario.vulnerabilities), dtype=np.int8
        )
        self._acquired_credentials = np.zeros(len(self.scenario.credentials), dtype=np.int8)
        self._acquired_privileges = np.zeros(len(self.scenario.privileges), dtype=np.int8)
        self._reachable_hosts = np.zeros(host_count, dtype=np.int8)
        for host_id in self._entry_hosts:
            index = self._host_index[host_id]
            self._discovered_hosts[index] = 1
            self._reachable_hosts[index] = 1
        self._detection_risk = 0.0
        self._path_cost = 0.0
        self._affected_nodes: tuple[str, ...] = ()
        self._steps = 0
        self._terminated = False
        self._finished = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._reset_state()
        return self._observation(), {
            "action_mask": self.action_mask(),
            "detection_risk": self._detection_risk,
            "path_cost": self._path_cost,
            "steps": self._steps,
        }

    def step(self, action: np.int64) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._finished:
            raise RuntimeError("step() called after episode completion; call reset() first")
        action_id = self._validate_action(action)
        valid = bool(self.action_mask()[action_id])
        self._affected_nodes = ()
        reward = self.reward_config.step_cost
        if not valid:
            reward += self.reward_config.duplicate_or_invalid
        elif action_id == Action.DISCOVER_HOST:
            reward += self._discover_host()
        elif action_id == Action.SCAN_SERVICE:
            reward += self._scan_service()
        elif action_id == Action.ENUMERATE_SERVICE:
            reward += self._enumerate_service()
        elif action_id == Action.VALIDATE_VULNERABILITY:
            reward += self._validate_vulnerability()
        elif action_id == Action.ATTEMPT_SIMULATED_ACCESS:
            reward += self._attempt_access()
        elif action_id == Action.ESCALATE_SIMULATED_PRIVILEGE:
            reward += self._escalate_privilege()
        elif action_id == Action.PIVOT_SIMULATED_NETWORK:
            reward += self._pivot_network()
        elif action_id == Action.COLLECT_SIMULATED_OBJECTIVE:
            reward += self._collect_objective()
            self._terminated = True
        elif action_id == Action.STOP:
            self._terminated = True
        self._steps += 1
        risk_penalty = self._detection_risk * self.reward_config.detection_risk
        reward += risk_penalty
        truncated = self._steps >= self.step_budget and not self._terminated
        self._finished = self._terminated or truncated
        info: dict[str, Any] = {
            "action_mask": self.action_mask(),
            "action_name": ACTION_NAMES[action_id],
            "affected_nodes": self._affected_nodes,
            "detection_risk": self._detection_risk,
            "path_cost": self._path_cost,
            "steps": self._steps,
            "valid_action": valid,
        }
        return self._observation(), float(reward), self._terminated, truncated, info

    def action_mask(self) -> np.ndarray[Any, Any]:
        """Return a boolean mask of actions currently valid in the state."""

        return np.array(
            [
                self._can_discover(),
                self._can_scan(),
                self._can_enumerate(),
                self._can_validate(),
                self._can_access(),
                self._can_escalate(),
                self._can_pivot(),
                self._can_collect(),
                True,
            ],
            dtype=np.int8,
        )

    def _validate_action(self, action: np.int64) -> int:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {len(Action) - 1}]")
        return int(action)

    def _observation(self) -> Observation:
        return {
            "discovered_hosts": self._discovered_hosts.copy(),
            "known_services": self._known_services.copy(),
            "validated_vulnerabilities": self._validated_vulnerabilities.copy(),
            "acquired_credentials": self._acquired_credentials.copy(),
            "acquired_privileges": self._acquired_privileges.copy(),
            "reachable_hosts": self._reachable_hosts.copy(),
            "detection_risk": np.array([self._detection_risk], dtype=np.float32),
            "steps_remaining": np.array([self.step_budget - self._steps], dtype=np.float32),
        }

    def _host_has_discovered_service(self, service_id: str) -> bool:
        return bool(self._discovered_hosts[self._host_index[self._service(service_id).host_id]])

    def _service(self, service_id: str) -> Service:
        return self.scenario.services[self._service_index[service_id]]

    def _can_discover(self) -> bool:
        return any(
            not self._discovered_hosts[self._host_index[edge.target_host_id]]
            for edge in self.scenario.network_edges
            if self._discovered_hosts[self._host_index[edge.source_host_id]]
        )

    def _can_scan(self) -> bool:
        return any(
            not self._known_services[index] and self._host_has_discovered_service(service.id)
            for index, service in enumerate(self.scenario.services)
        )

    def _can_enumerate(self) -> bool:
        return bool(np.any(self._known_services & (1 - self._enumerated_services)))

    def _can_validate(self) -> bool:
        return any(
            not self._validated_vulnerabilities[index]
            and self._enumerated_services[
                self._service_index[self._vulnerability(index).service_id]
            ]
            for index in range(len(self.scenario.vulnerabilities))
        )

    def _can_access(self) -> bool:
        return any(
            self._validated_vulnerabilities[self._vulnerability_index[edge.vulnerability_id]]
            and not self._acquired_credentials[self._credential_index[edge.credential_id]]
            for edge in self.scenario.access_edges
        )

    def _can_escalate(self) -> bool:
        return any(
            self._acquired_privileges[self._privilege_index[edge.source_privilege_id]]
            and not self._acquired_privileges[self._privilege_index[edge.target_privilege_id]]
            for edge in self.scenario.privilege_edges
        )

    def _can_pivot(self) -> bool:
        return self._can_discover()

    def _can_collect(self) -> bool:
        return any(
            self._discovered_hosts[self._host_index[objective.host_id]]
            and (
                objective.required_privilege_id is None
                or self._acquired_privileges[self._privilege_index[objective.required_privilege_id]]
            )
            for objective in self.scenario.objectives
        )

    def _discover_host(self) -> float:
        for edge in self.scenario.network_edges:
            target = self._host_index[edge.target_host_id]
            if (
                self._discovered_hosts[self._host_index[edge.source_host_id]]
                and not self._discovered_hosts[target]
            ):
                self._discovered_hosts[target] = 1
                self._reachable_hosts[target] = 1
                self._path_cost += edge.cost
                self._affected_nodes = (edge.source_host_id, edge.target_host_id)
                return self.reward_config.new_host
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _scan_service(self) -> float:
        for index, service in enumerate(self.scenario.services):
            if not self._known_services[index] and self._host_has_discovered_service(service.id):
                self._known_services[index] = 1
                self._affected_nodes = (service.host_id, service.id)
                return self.reward_config.new_service
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _enumerate_service(self) -> float:
        candidates = np.flatnonzero(self._known_services & (1 - self._enumerated_services))
        if len(candidates):
            service = self.scenario.services[int(candidates[0])]
            self._enumerated_services[candidates[0]] = 1
            self._detection_risk = min(
                1.0,
                self._detection_risk + self._host_detection_increment(service.host_id),
            )
            self._affected_nodes = (service.host_id, service.id)
            return 0.0
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _validate_vulnerability(self) -> float:
        for index, vulnerability in enumerate(self.scenario.vulnerabilities):
            service_index = self._service_index[vulnerability.service_id]
            if (
                not self._validated_vulnerabilities[index]
                and self._enumerated_services[service_index]
            ):
                self._validated_vulnerabilities[index] = 1
                self._affected_nodes = (vulnerability.service_id, vulnerability.id)
                return self.reward_config.validated_vulnerability
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _attempt_access(self) -> float:
        for edge in self.scenario.access_edges:
            vulnerability = self._vulnerability_index[edge.vulnerability_id]
            credential = self._credential_index[edge.credential_id]
            if (
                self._validated_vulnerabilities[vulnerability]
                and not self._acquired_credentials[credential]
            ):
                self._acquired_credentials[credential] = 1
                self._acquired_privileges[
                    self._privilege_index[self.scenario.credentials[credential].privilege_id]
                ] = 1
                credential_record = self.scenario.credentials[credential]
                self._affected_nodes = (
                    edge.vulnerability_id,
                    edge.credential_id,
                    credential_record.host_id,
                    credential_record.privilege_id,
                )
                return self.reward_config.access
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _escalate_privilege(self) -> float:
        for edge in self.scenario.privilege_edges:
            source = self._privilege_index[edge.source_privilege_id]
            target = self._privilege_index[edge.target_privilege_id]
            if self._acquired_privileges[source] and not self._acquired_privileges[target]:
                self._acquired_privileges[target] = 1
                self._affected_nodes = (edge.source_privilege_id, edge.target_privilege_id)
                return self.reward_config.privilege_escalation
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _pivot_network(self) -> float:
        return self._discover_host()

    def _collect_objective(self) -> float:
        if self._can_collect():
            objective = next(
                objective
                for objective in self.scenario.objectives
                if self._discovered_hosts[self._host_index[objective.host_id]]
                and (
                    objective.required_privilege_id is None
                    or self._acquired_privileges[
                        self._privilege_index[objective.required_privilege_id]
                    ]
                )
            )
            self._affected_nodes = (objective.host_id, objective.id)
            return self.reward_config.objective
        return self.reward_config.duplicate_or_invalid  # pragma: no cover

    def _vulnerability(self, index: int) -> Vulnerability:
        return self.scenario.vulnerabilities[index]

    def _host_detection_increment(self, host_id: str) -> float:
        probabilities = [
            control.detection_probability
            for control in self.scenario.security_controls
            if host_id in control.host_ids
        ]
        if not probabilities:
            return 0.05
        return max(0.02, (sum(probabilities) / len(probabilities)) * 0.15)
