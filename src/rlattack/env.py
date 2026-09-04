"""Gymnasium environment for safe, reproducible attack-path simulation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlattack.defender import (
    ContextualDefender,
    DefenderConfig,
    DefenderContext,
    DefenderResponse,
    DefenderState,
    decide_response,
)
from rlattack.scenario import NetworkEdge, Scenario

Observation = dict[str, np.ndarray[Any, Any]]


class Action(IntEnum):
    """Stable action-type IDs exposed by :class:`AttackPathEnv`.

    A concrete environment action pairs one of these types with a target index; see
    :meth:`AttackPathEnv.encode_action`.
    """

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

TARGET_KINDS: tuple[str, ...] = (
    "host",
    "service",
    "service",
    "vulnerability",
    "access_edge",
    "privilege_edge",
    "host",
    "objective",
    "none",
)


_DEFENDER_STREAM = 0x5EED


def _capacity(records: tuple[Any, ...], configured: int | None) -> int:
    """Return the observation width for one record kind.

    Widths are always positive because Gymnasium rejects zero-length binary spaces, and
    a configured capacity must be able to hold the scenario it is used with.
    """

    if configured is None:
        return max(1, len(records))
    if configured < len(records):
        raise ValueError(
            f"observation capacity {configured} is smaller than the scenario's {len(records)}"
        )
    return configured


@dataclass(frozen=True)
class RewardConfig:
    """Configurable baseline reward values for the simulator."""

    new_host: float = 1.0
    new_service: float = 0.5
    validated_vulnerability: float = 1.0
    access: float = 2.0
    privilege_escalation: float = 3.0
    pivot: float = 1.0
    objective: float = 10.0
    duplicate_or_invalid: float = -1.0
    failed_attempt: float = -0.2
    detected: float = -5.0
    detection_risk: float = -0.5
    step_cost: float = -0.01


@dataclass(frozen=True)
class DynamicsConfig:
    """Transition uncertainty and detection rules for one experiment.

    The environment stays reproducible: every random draw comes from the seeded
    :attr:`gymnasium.Env.np_random` stream, so a seed fixes the whole trajectory.

    Detection risk is normalized by network size by default. Without it the threshold is
    an absolute budget of noisy actions, so a larger network is unwinnable purely
    because reaching its objective takes more steps - which shows up in a transfer
    table as a generalization failure that is really a calibration artifact.

    The agent observes ``probed_hosts``: which hosts it has probed and missed since its
    last new vantage point. That is its own memory of its own actions, not privileged
    knowledge, and without it a policy cannot tell "I have probed everything once" from
    "I have not probed at all" - the distinction lives only in the action mask, which a
    maskable learner uses to filter its action distribution rather than as an input.

    ``noisy_discovery`` turns neighbour discovery into a scan. With exact adjacency the
    action mask itself reveals the topology: it offers ``discover_host`` for precisely
    the hosts that are genuinely adjacent, so the agent reads the graph instead of
    probing it. When enabled, every undiscovered host can be probed, only genuinely
    adjacent ones can succeed, and they succeed with ``discovery_probability``.
    """

    stochastic: bool = True
    base_success_probability: float = 0.75
    minimum_success_probability: float = 0.35
    detection_threshold: float = 0.9
    failed_attempt_risk: float = 0.05
    pivot_risk: float = 0.03
    normalize_risk_by_size: bool = True
    risk_reference_hosts: int = 6
    noisy_discovery: bool = False
    discovery_probability: float = 0.7

    def __post_init__(self) -> None:
        for name in ("base_success_probability", "minimum_success_probability"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.0 < self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")
        for name in ("failed_attempt_risk", "pivot_risk"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.risk_reference_hosts < 1:
            raise ValueError("risk_reference_hosts must be positive")
        if not 0.0 < self.discovery_probability <= 1.0:
            raise ValueError("discovery_probability must be in (0, 1]")

    @classmethod
    def deterministic(cls) -> DynamicsConfig:
        """Return dynamics where every valid action succeeds, for regression tests."""

        return cls(stochastic=False)


@dataclass(frozen=True)
class ObservationConfig:
    """What the agent may observe, and the fixed widths it observes it through.

    Two separate concerns are deliberately handled together here, because both are
    about the *interface* a policy sees rather than about the simulated world:

    * **Capacities.** Without them the observation and action widths equal the record
      counts of one scenario, which both leaks the network size to the agent and makes
      a policy trained on ``small`` structurally incompatible with ``large``. Fixed
      capacities pad every channel, so undiscovered records and unused slots are
      indistinguishable and one policy transfers across scenario classes.
    * **Detection visibility.** An attacker does not read the defender's exact
      suspicion score. By default the agent sees a quantized ``alert_level`` one-hot;
      the exact risk stays in ``info`` for reporting and analysis only.

    ``expose_monitoring`` adds a per-host bit for the hosts a targeted defender is
    watching, and only for hosts the agent has already discovered. Targeted monitoring
    that cannot be observed is not something to route around, only extra variance, so
    an attention experiment needs this on. It is off by default because it widens the
    observation space, and a checkpoint trained without it cannot be loaded with it.
    The modelled capability is fingerprinting a scanned host's monitoring posture; a
    real attacker's read on that is noisier than this, so treat the resulting evasion
    numbers as an optimistic bound.

    ``expose_watch_history`` adds a per-host recency channel: how long ago (as a
    fraction of the step budget, discovered hosts only) attention last landed there.
    ``monitored_hosts`` is a snapshot of *right now*; it cannot tell a policy whether a
    re-aiming defender's current posture is stable or the result of the attacker's own
    last move, which is the distinction a memoryless policy needs to exploit a
    contextual (``--adversarial``) defender rather than only a fixed one (item 61). Off
    by default for the same reason as ``expose_monitoring`` - it widens the observation
    space, and a checkpoint trained without it cannot be loaded with it.
    """

    host_capacity: int | None = None
    service_capacity: int | None = None
    vulnerability_capacity: int | None = None
    credential_capacity: int | None = None
    privilege_capacity: int | None = None
    access_capacity: int | None = None
    privilege_edge_capacity: int | None = None
    objective_capacity: int | None = None
    expose_exact_risk: bool = False
    expose_monitoring: bool = False
    expose_watch_history: bool = False
    alert_levels: int = 3

    def __post_init__(self) -> None:
        for name in (
            "host_capacity",
            "service_capacity",
            "vulnerability_capacity",
            "credential_capacity",
            "privilege_capacity",
            "access_capacity",
            "privilege_edge_capacity",
            "objective_capacity",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when set")
        if self.alert_levels < 2:
            raise ValueError("alert_levels must be at least 2")

    @classmethod
    def for_curriculum(
        cls,
        *,
        expose_exact_risk: bool = False,
        expose_monitoring: bool = False,
        expose_watch_history: bool = False,
    ) -> ObservationConfig:
        """Return capacities large enough for every generated scenario class.

        A single policy can then be trained on ``small`` scenarios and evaluated on
        ``large`` ones without changing the observation or action space.
        """

        return cls(
            host_capacity=16,
            service_capacity=32,
            vulnerability_capacity=32,
            credential_capacity=16,
            privilege_capacity=8,
            access_capacity=16,
            privilege_edge_capacity=8,
            objective_capacity=8,
            expose_exact_risk=expose_exact_risk,
            expose_monitoring=expose_monitoring,
            expose_watch_history=expose_watch_history,
        )


class AttackPathEnv(gym.Env[Observation, np.int64]):
    """Reproducible in-memory attack graph environment.

    The environment never opens sockets, executes subprocesses, or invokes offensive
    tooling. Each action only updates arrays derived from a validated :class:`Scenario`.

    Actions are ``action_type * target_count + target_index`` so that a policy chooses
    *what* to do and *which* graph element to do it to.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario,
        step_budget: int = 50,
        reward_config: RewardConfig | None = None,
        dynamics: DynamicsConfig | None = None,
        observation_config: ObservationConfig | None = None,
        defender: DefenderConfig | None = None,
        defender_policy: ContextualDefender | None = None,
    ) -> None:
        if step_budget < 1:
            raise ValueError("step_budget must be positive")
        if not scenario.hosts:
            raise ValueError("scenario must contain at least one host")
        self.scenario = scenario
        self.step_budget = step_budget
        self.reward_config = reward_config or RewardConfig()
        self.dynamics = dynamics or DynamicsConfig()
        self.observation_config = observation_config or ObservationConfig()
        self.defender_policy = defender_policy
        self.defender = defender or (
            defender_policy.config if defender_policy is not None else DefenderConfig()
        )
        self._risk_scale = (
            min(1.0, self.dynamics.risk_reference_hosts / len(scenario.hosts))
            if self.dynamics.normalize_risk_by_size
            else 1.0
        )
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
        self._host_credentials: dict[str, tuple[int, ...]] = {
            host.id: tuple(
                index
                for index, credential in enumerate(scenario.credentials)
                if credential.host_id == host.id
            )
            for host in scenario.hosts
        }
        self._entry_hosts = set(scenario.entry_host_ids or (scenario.hosts[0].id,))
        config = self.observation_config
        self._host_width = _capacity(scenario.hosts, config.host_capacity)
        self._service_width = _capacity(scenario.services, config.service_capacity)
        self._vulnerability_width = _capacity(
            scenario.vulnerabilities, config.vulnerability_capacity
        )
        self._credential_width = _capacity(scenario.credentials, config.credential_capacity)
        self._privilege_width = _capacity(scenario.privileges, config.privilege_capacity)
        self._access_width = _capacity(scenario.access_edges, config.access_capacity)
        self._privilege_edge_width = _capacity(
            scenario.privilege_edges, config.privilege_edge_capacity
        )
        self._objective_width = _capacity(scenario.objectives, config.objective_capacity)
        self.target_count = max(
            self._host_width,
            self._service_width,
            self._vulnerability_width,
            self._access_width,
            self._privilege_edge_width,
            self._objective_width,
        )
        host_ids = tuple(record.id for record in scenario.hosts)
        service_ids = tuple(record.id for record in scenario.services)
        self._target_ids: tuple[tuple[str, ...], ...] = (
            host_ids,
            service_ids,
            service_ids,
            tuple(record.id for record in scenario.vulnerabilities),
            tuple(
                f"{edge.vulnerability_id}->{edge.credential_id}" for edge in scenario.access_edges
            ),
            tuple(
                f"{edge.source_privilege_id}->{edge.target_privilege_id}"
                for edge in scenario.privilege_edges
            ),
            host_ids,
            tuple(record.id for record in scenario.objectives),
            (),
        )
        self.action_space = spaces.Discrete(len(Action) * self.target_count)
        channels: dict[str, spaces.Space[Any]] = {
            "discovered_hosts": spaces.MultiBinary(self._host_width),
            "reachable_hosts": spaces.MultiBinary(self._host_width),
            "known_services": spaces.MultiBinary(self._service_width),
            "enumerated_services": spaces.MultiBinary(self._service_width),
            "validated_vulnerabilities": spaces.MultiBinary(self._vulnerability_width),
            "acquired_credentials": spaces.MultiBinary(self._credential_width),
            "acquired_privileges": spaces.MultiBinary(self._privilege_width),
            "collected_objectives": spaces.MultiBinary(self._objective_width),
            "probed_hosts": spaces.MultiBinary(self._host_width),
            "alert_level": spaces.MultiBinary(config.alert_levels),
            "budget_fraction": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
        if config.expose_exact_risk:
            channels["detection_risk"] = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        if config.expose_monitoring:
            channels["monitored_hosts"] = spaces.MultiBinary(self._host_width)
        if config.expose_watch_history:
            channels["watch_recency"] = spaces.Box(
                0.0, 1.0, shape=(self._host_width,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(channels)
        self._reset_state()

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        self._discovered_hosts = np.zeros(self._host_width, dtype=np.int8)
        self._reachable_hosts = np.zeros(self._host_width, dtype=np.int8)
        self._known_services = np.zeros(self._service_width, dtype=np.int8)
        self._enumerated_services = np.zeros(self._service_width, dtype=np.int8)
        self._validated_vulnerabilities = np.zeros(self._vulnerability_width, dtype=np.int8)
        self._acquired_credentials = np.zeros(self._credential_width, dtype=np.int8)
        self._acquired_privileges = np.zeros(self._privilege_width, dtype=np.int8)
        self._collected_objectives = np.zeros(self._objective_width, dtype=np.int8)
        self._failed_discovery = np.zeros(self._host_width, dtype=np.int8)
        for host_id in self._entry_hosts:
            index = self._host_index[host_id]
            self._discovered_hosts[index] = 1
            self._reachable_hosts[index] = 1
        self._detection_risk = 0.0
        self._path_cost = 0.0
        self._affected_nodes: tuple[str, ...] = ()
        self._outcome = "reset"
        self._steps = 0
        self._terminated = False
        self._finished = False
        self._detected = False
        self._objective_captured = False
        self._host_hardening = np.zeros(self._host_width, dtype=np.float32)
        self._host_attention = np.ones(self._host_width, dtype=np.float32)
        # A sentinel far outside the step budget, not -1: "never watched" must clip to
        # the same 1.0 (maximally stale) as "watched a very long time ago", not be
        # mistaken for a small step count.
        self._host_last_watched_step = np.full(self._host_width, -10_000, dtype=np.int32)
        self._aim_attention(self._default_watchlist())
        self._last_response_step = 0
        self._defender_actions = 0
        self._revoked_credentials = 0
        self._defender_false_positives = 0
        self._defender_over_budget = 0
        self._last_defender_response = DefenderResponse()
        self._pending_response: tuple[DefenderResponse, int, bool] | None = None
        self._defender_rng = np.random.default_rng(_DEFENDER_STREAM)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._reset_state()
        # The defender draws from its own stream so that enabling it does not shift the
        # attacker's draws. Otherwise a passive/adaptive comparison on the same seed is
        # not actually paired: the control and the treatment see different episodes.
        self._defender_rng = np.random.default_rng(
            None if seed is None else seed ^ _DEFENDER_STREAM
        )
        return self._observation(), {
            "action_mask": self.action_mask(),
            "detection_risk": self._detection_risk,
            "alert_level": self.alert_level(),
            "path_cost": self._path_cost,
            "steps": self._steps,
            "steps_remaining": self.step_budget,
            "target_count": self.target_count,
            "defender_action": "none",
            "defender_actions": 0,
            "defender_false_positives": 0,
            "defender_over_budget": 0,
            "defender_budget_remaining": self._budget_remaining(),
            "defender_pending": False,
            "revoked_credentials": 0,
            "objective_captured": False,
            "collected_objectives": 0,
            "detected": False,
        }

    # ----------------------------------------------------------------- action

    def encode_action(self, action_type: int, target: int = 0) -> np.int64:
        """Encode an ``(action_type, target)`` pair into one environment action."""

        if not 0 <= action_type < len(Action):
            raise ValueError("action_type is outside the RLAttack action catalogue")
        if not 0 <= target < self.target_count:
            raise ValueError("target is outside the scenario target range")
        return np.int64(action_type * self.target_count + target)

    def decode_action(self, action: int | np.int64) -> tuple[int, int]:
        """Split one environment action back into ``(action_type, target)``."""

        action_type, target = divmod(int(action), self.target_count)
        if action_type >= len(Action):
            raise ValueError("action is outside the RLAttack action catalogue")
        return action_type, target

    def action_mask(self) -> np.ndarray[Any, Any]:
        """Return a flat boolean mask of the actions currently valid in the state."""

        mask = np.zeros((len(Action), self.target_count), dtype=np.int8)
        for index in self._discoverable_hosts():
            mask[Action.DISCOVER_HOST, index] = 1
        for index in self._pivotable_hosts():
            mask[Action.PIVOT_SIMULATED_NETWORK, index] = 1
        for index in self._scannable_services():
            mask[Action.SCAN_SERVICE, index] = 1
        for index in self._enumerable_services():
            mask[Action.ENUMERATE_SERVICE, index] = 1
        for index in self._validatable_vulnerabilities():
            mask[Action.VALIDATE_VULNERABILITY, index] = 1
        for index in self._accessible_edges():
            mask[Action.ATTEMPT_SIMULATED_ACCESS, index] = 1
        for index in self._escalatable_edges():
            mask[Action.ESCALATE_SIMULATED_PRIVILEGE, index] = 1
        for index in self._collectable_objectives():
            mask[Action.COLLECT_SIMULATED_OBJECTIVE, index] = 1
        mask[Action.STOP, 0] = 1
        return mask.reshape(-1)

    def action_masks(self) -> np.ndarray[Any, Any]:
        """Return the mask as booleans, the shape ``sb3-contrib`` expects."""

        return self.action_mask().astype(bool)

    def step(self, action: np.int64) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._finished:
            raise RuntimeError("step() called after episode completion; call reset() first")
        action_type, target = self._validate_action(action)
        mask = self.action_mask()
        valid = bool(mask[action_type * self.target_count + target])
        self._affected_nodes = ()
        self._outcome = "invalid"
        reward = self.reward_config.step_cost
        if not valid:
            reward += self.reward_config.duplicate_or_invalid
        else:
            reward += self._apply(action_type, target)
        self._steps += 1
        self._refresh_exhausted_probes()
        self._apply_defender()
        reward += self._detection_risk * self.reward_config.detection_risk
        if self._detection_risk >= self.dynamics.detection_threshold and not self._terminated:
            self._detected = True
            self._terminated = True
            self._outcome = "detected"
            reward += self.reward_config.detected
        truncated = self._steps >= self.step_budget and not self._terminated
        self._finished = self._terminated or truncated
        info: dict[str, Any] = {
            "action_mask": self.action_mask(),
            "action_name": ACTION_NAMES[action_type],
            "action_type": action_type,
            "target": target,
            "target_id": self._target_id(action_type, target),
            "affected_nodes": self._affected_nodes,
            "outcome": self._outcome,
            "detection_risk": self._detection_risk,
            "alert_level": self.alert_level(),
            "defender_action": self._last_defender_response.name,
            "defender_actions": self._defender_actions,
            "defender_false_positives": self._defender_false_positives,
            "defender_over_budget": self._defender_over_budget,
            "defender_budget_remaining": self._budget_remaining(),
            "defender_pending": self._pending_response is not None,
            "revoked_credentials": self._revoked_credentials,
            "detected": self._detected,
            "objective_captured": self._objective_captured,
            "collected_objectives": int(self._collected_objectives.sum()),
            "path_cost": self._path_cost,
            "steps": self._steps,
            "steps_remaining": self.step_budget - self._steps,
            "target_count": self.target_count,
            "valid_action": valid,
        }
        return self._observation(), float(reward), self._terminated, truncated, info

    def _validate_action(self, action: np.int64) -> tuple[int, int]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"action must be an integer in [0, {len(Action) * self.target_count - 1}]"
            )
        return self.decode_action(action)

    def _apply(self, action_type: int, target: int) -> float:
        if action_type == Action.DISCOVER_HOST:
            return self._discover_host(target)
        if action_type == Action.SCAN_SERVICE:
            return self._scan_service(target)
        if action_type == Action.ENUMERATE_SERVICE:
            return self._enumerate_service(target)
        if action_type == Action.VALIDATE_VULNERABILITY:
            return self._validate_vulnerability(target)
        if action_type == Action.ATTEMPT_SIMULATED_ACCESS:
            return self._attempt_access(target)
        if action_type == Action.ESCALATE_SIMULATED_PRIVILEGE:
            return self._escalate_privilege(target)
        if action_type == Action.PIVOT_SIMULATED_NETWORK:
            return self._pivot_network(target)
        if action_type == Action.COLLECT_SIMULATED_OBJECTIVE:
            return self._collect_objective(target)
        self._terminated = True
        self._outcome = "stopped"
        return 0.0

    # ------------------------------------------------------------ validity

    def _discoverable_hosts(self) -> Iterator[int]:
        if self.dynamics.noisy_discovery:
            # Probing is available for any host we have not discovered or already
            # probed from the current vantage points; adjacency is not revealed here.
            for index in range(len(self.scenario.hosts)):
                if not self._discovered_hosts[index] and not self._failed_discovery[index]:
                    yield index
            return
        for index in range(len(self.scenario.hosts)):
            if not self._discovered_hosts[index] and self._reachable_edges(index):
                yield index

    def _pivotable_hosts(self) -> Iterator[int]:
        for index in range(len(self.scenario.hosts)):
            if (
                self._discovered_hosts[index]
                and not self._reachable_hosts[index]
                and self._pivot_edges(index)
            ):
                yield index

    def _scannable_services(self) -> Iterator[int]:
        for index, service in enumerate(self.scenario.services):
            if (
                not self._known_services[index]
                and self._reachable_hosts[self._host_index[service.host_id]]
            ):
                yield index

    def _enumerable_services(self) -> Iterator[int]:
        for index in range(len(self.scenario.services)):
            if self._known_services[index] and not self._enumerated_services[index]:
                yield index

    def _validatable_vulnerabilities(self) -> Iterator[int]:
        for index, vulnerability in enumerate(self.scenario.vulnerabilities):
            service_index = self._service_index[vulnerability.service_id]
            if (
                not self._validated_vulnerabilities[index]
                and self._enumerated_services[service_index]
            ):
                yield index

    def _accessible_edges(self) -> Iterator[int]:
        for index, edge in enumerate(self.scenario.access_edges):
            if (
                self._validated_vulnerabilities[self._vulnerability_index[edge.vulnerability_id]]
                and not self._acquired_credentials[self._credential_index[edge.credential_id]]
            ):
                yield index

    def _escalatable_edges(self) -> Iterator[int]:
        for index, edge in enumerate(self.scenario.privilege_edges):
            if (
                self._acquired_privileges[self._privilege_index[edge.source_privilege_id]]
                and not self._acquired_privileges[self._privilege_index[edge.target_privilege_id]]
            ):
                yield index

    def _collectable_objectives(self) -> Iterator[int]:
        for index, objective in enumerate(self.scenario.objectives):
            if self._collected_objectives[index]:
                continue
            if self._reachable_hosts[self._host_index[objective.host_id]] and (
                objective.required_privilege_id is None
                or self._acquired_privileges[self._privilege_index[objective.required_privilege_id]]
            ):
                yield index

    def _reachable_edges(self, host_index: int) -> list[NetworkEdge]:
        """Return every edge into ``host_index`` whose source host is reachable."""

        host_id = self.scenario.hosts[host_index].id
        return [
            edge
            for edge in self.scenario.network_edges
            if edge.target_host_id == host_id
            and self._reachable_hosts[self._host_index[edge.source_host_id]]
        ]

    def _pivot_edges(self, host_index: int) -> list[NetworkEdge]:
        """Return edges into ``host_index`` from a reachable host we hold a foothold on."""

        return [
            edge
            for edge in self._reachable_edges(host_index)
            if self._has_foothold(edge.source_host_id)
        ]

    def _has_foothold(self, host_id: str) -> bool:
        """A host grants a pivot when it models no credential or one we already hold."""

        credentials = self._host_credentials[host_id]
        if not credentials:
            return True
        return any(bool(self._acquired_credentials[index]) for index in credentials)

    # ------------------------------------------------------------- handlers

    def _discover_host(self, target: int) -> float:
        edges = self._reachable_edges(target)
        if self.dynamics.noisy_discovery and (
            not edges or not self._succeeds(self.dynamics.discovery_probability)
        ):
            self._failed_discovery[target] = 1
            host_id = self.scenario.hosts[target].id
            return self._failed_attempt((host_id,), host_id)
        edge = min(edges, key=lambda edge: (edge.cost, edge.source_host_id))
        self._discovered_hosts[target] = 1
        self._path_cost += edge.cost
        self._affected_nodes = (edge.source_host_id, edge.target_host_id)
        self._outcome = "success"
        return self.reward_config.new_host

    def _pivot_network(self, target: int) -> float:
        edge = self._pivot_edges(target)[0]
        self._reachable_hosts[target] = 1
        # A new vantage point makes previously fruitless probes worth repeating.
        self._failed_discovery[:] = 0
        self._raise_risk(self.dynamics.pivot_risk, edge.target_host_id)
        self._affected_nodes = (edge.source_host_id, edge.target_host_id)
        self._outcome = "success"
        return self.reward_config.pivot

    def _scan_service(self, target: int) -> float:
        service = self.scenario.services[target]
        self._known_services[target] = 1
        self._affected_nodes = (service.host_id, service.id)
        self._outcome = "success"
        return self.reward_config.new_service

    def _enumerate_service(self, target: int) -> float:
        service = self.scenario.services[target]
        self._enumerated_services[target] = 1
        self._raise_risk(self._host_detection_increment(service.host_id), service.host_id)
        self._affected_nodes = (service.host_id, service.id)
        self._outcome = "success"
        return 0.0

    def _validate_vulnerability(self, target: int) -> float:
        vulnerability = self.scenario.vulnerabilities[target]
        nodes = (vulnerability.service_id, vulnerability.id)
        if not self._succeeds(vulnerability.exploitability):
            service = self.scenario.services[self._service_index[vulnerability.service_id]]
            return self._failed_attempt(nodes, service.host_id)
        self._validated_vulnerabilities[target] = 1
        self._affected_nodes = nodes
        self._outcome = "success"
        return self.reward_config.validated_vulnerability

    def _attempt_access(self, target: int) -> float:
        edge = self.scenario.access_edges[target]
        credential_index = self._credential_index[edge.credential_id]
        credential = self.scenario.credentials[credential_index]
        nodes = (edge.vulnerability_id, edge.credential_id, credential.host_id)
        exploitability = self.scenario.vulnerabilities[
            self._vulnerability_index[edge.vulnerability_id]
        ].exploitability
        if not self._succeeds(exploitability):
            return self._failed_attempt(nodes, credential.host_id)
        self._acquired_credentials[credential_index] = 1
        self._acquired_privileges[self._privilege_index[credential.privilege_id]] = 1
        self._affected_nodes = (*nodes, credential.privilege_id)
        self._outcome = "success"
        return self.reward_config.access

    def _escalate_privilege(self, target: int) -> float:
        edge = self.scenario.privilege_edges[target]
        nodes = (edge.source_privilege_id, edge.target_privilege_id)
        exploitability = (
            self.scenario.vulnerabilities[
                self._vulnerability_index[edge.vulnerability_id]
            ].exploitability
            if edge.vulnerability_id is not None
            else None
        )
        if not self._succeeds(exploitability):
            return self._failed_attempt(nodes)
        self._acquired_privileges[self._privilege_index[edge.target_privilege_id]] = 1
        self._affected_nodes = nodes
        self._outcome = "success"
        return self.reward_config.privilege_escalation

    def _collect_objective(self, target: int) -> float:
        """Collect one objective; the episode ends only once every objective is held."""

        objective = self.scenario.objectives[target]
        self._collected_objectives[target] = 1
        self._affected_nodes = (objective.host_id, objective.id)
        self._outcome = "objective"
        if int(self._collected_objectives.sum()) == len(self.scenario.objectives):
            self._objective_captured = True
            self._terminated = True
        return self.reward_config.objective

    def _failed_attempt(self, nodes: tuple[str, ...], host_id: str | None = None) -> float:
        self._raise_risk(self.dynamics.failed_attempt_risk, host_id)
        self._affected_nodes = nodes
        self._outcome = "failed"
        return self.reward_config.failed_attempt

    # -------------------------------------------------------------- helpers

    def _apply_defender(self) -> None:
        """Advance the defender: land a scheduled response, or schedule a new one.

        A decision does not take effect immediately. It is queued for
        ``response_latency`` steps, so an attacker that finishes quickly can outrun a
        response the defender has already decided on.
        """

        self._last_defender_response = DefenderResponse()
        if self._land_pending_response():
            return
        if self._pending_response is not None:
            return
        observed_risk = self._observed_risk()
        credentials = tuple(int(index) for index in np.flatnonzero(self._acquired_credentials))
        if self.defender_policy is not None:
            response = self._policy_response(self.defender_policy, observed_risk, credentials)
        else:
            response = decide_response(
                self.defender,
                DefenderState(
                    observed_risk=observed_risk,
                    steps_since_response=self._steps - self._last_response_step,
                    acquired_credentials=credentials,
                ),
                self._defender_rng,
            )
        if response.name == "none":
            return
        false_positive = self._detection_risk < self.defender.alert_threshold
        self._pending_response = (
            response,
            self._steps + self.defender.response_latency,
            false_positive,
        )
        # A zero-latency defender responds within the same step it decided.
        self._land_pending_response()

    def _refresh_exhausted_probes(self) -> None:
        """Re-open probing once every undiscovered host has been probed and missed.

        Without this a chain topology deadlocks: the single adjacent host fails its
        scan, every other probe fails because it is not adjacent, and the agent has no
        discovery action left to reach a new vantage point.
        """

        if not self.dynamics.noisy_discovery:
            return
        undiscovered = [
            index for index in range(len(self.scenario.hosts)) if not self._discovered_hosts[index]
        ]
        if undiscovered and all(self._failed_discovery[index] for index in undiscovered):
            self._failed_discovery[:] = 0

    def _policy_response(
        self, policy: ContextualDefender, observed_risk: float, credentials: tuple[int, ...]
    ) -> DefenderResponse:
        """Ask the learned defender policy what to do in the current context."""

        bands = self.observation_config.alert_levels
        band = min(bands - 1, int(observed_risk / self.dynamics.detection_threshold * bands))
        context = DefenderContext(
            alert_band=max(0, band),
            has_credentials=bool(credentials),
            phase=min(2, int(self._steps / self.step_budget * 3)),
            budget_pressure=self._budget_pressure(),
        )
        action = policy.action_for(context)
        if action == "harden":
            return DefenderResponse(harden=True)
        if action == "revoke" and credentials:
            return DefenderResponse(
                harden=True,
                revoke_credential=int(self._defender_rng.choice(np.asarray(credentials))),
            )
        return DefenderResponse()

    def _land_pending_response(self) -> bool:
        """Apply a queued defender response once its latency has elapsed."""

        if self._pending_response is None:
            return False
        response, due_step, false_positive = self._pending_response
        if self._steps < due_step:
            return False
        self._pending_response = None
        budget = self.defender.response_budget
        if budget is not None and self._defender_actions >= budget:
            # The response was decided but the team has no capacity left to run it.
            self._defender_over_budget += 1
            return True
        self._last_defender_response = response
        self._last_response_step = self._steps
        self._defender_actions += 1
        if false_positive:
            self._defender_false_positives += 1
        if response.harden:
            self._host_hardening[np.flatnonzero(self._reachable_hosts)] += (
                self.defender.hardening_step
            )
            # For a targeted defender, hardening *is* re-aiming: it moves the watchers
            # onto the ground the attacker has taken instead of raising sensitivity
            # everywhere at once. Attention stays conserved, so the frontier the
            # attacker has not reached yet gets cheaper as its rear gets watched.
            self._aim_attention(self._alerted_watchlist())
        if response.revoke_credential is not None:
            self._revoke_credential(response.revoke_credential)
        return True

    def _budget_remaining(self) -> int | None:
        """Return the responses the defender can still afford this episode."""

        budget = self.defender.response_budget
        if budget is None:
            return None
        return max(0, budget - self._defender_actions)

    def _budget_pressure(self) -> int:
        """Quantize how much of the response budget is spent, for the defender policy."""

        budget = self.defender.response_budget
        if budget is None:
            return 0
        return min(2, int(self._defender_actions / budget * 3))

    def _default_watchlist(self) -> tuple[str, ...]:
        """Return the hosts a defender watches before it has seen anything.

        Ranked by what a defender can actually justify watching without knowing the
        attacker: exposure (how many services the host runs), then centrality (how many
        network edges it carries), with a crown-jewel bonus for hosts holding an
        objective. Ranking by objectives alone puts the watchers where nothing happens
        - in these scenarios the objective host sees one pivot while the entry and route
        hosts absorb every enumeration - which makes concentrated monitoring strictly
        worse than uniform rather than a different shape of the same budget.

        Deterministic on purpose: a standing monitoring posture is a deployment
        decision, not a draw, and keeping it reproducible keeps a paired comparison
        paired.
        """

        exposure: dict[str, int] = {host.id: 0 for host in self.scenario.hosts}
        for service in self.scenario.services:
            exposure[service.host_id] = exposure.get(service.host_id, 0) + 1
        degree: dict[str, int] = {host.id: 0 for host in self.scenario.hosts}
        for edge in self.scenario.network_edges:
            degree[edge.source_host_id] = degree.get(edge.source_host_id, 0) + 1
            degree[edge.target_host_id] = degree.get(edge.target_host_id, 0) + 1
        crown = {objective.host_id for objective in self.scenario.objectives}

        def rank(host_id: str) -> tuple[int, int, int, str]:
            return (
                -exposure.get(host_id, 0),
                -degree.get(host_id, 0),
                0 if host_id in crown else 1,
                host_id,
            )

        return tuple(sorted((host.id for host in self.scenario.hosts), key=rank))

    def _aim_attention(self, watchlist: Sequence[str]) -> None:
        """Point the defender's monitoring at the first hosts of ``watchlist``.

        Attention is conserved rather than added: the watched hosts are raised and every
        other host drops by exactly as much in the mean, so a defender that looks harder
        at one place is looking less hard everywhere else.
        """

        if not self.defender.targeted_attention:
            return
        focus, blind = self.defender.attention_split(len(self.scenario.hosts))
        watched = list(dict.fromkeys(watchlist))[: self.defender.attention_hosts]
        self._host_attention[:] = blind
        for host_id in watched:
            index = self._host_index[host_id]
            self._host_attention[index] = focus
            self._host_last_watched_step[index] = self._steps

    def _alerted_watchlist(self) -> tuple[str, ...]:
        """Return where a defender re-aims once the attacker has shown itself.

        Ground the attacker already holds comes first, because that is the evidence the
        defender actually has; the standing posture fills the remaining slots.
        """

        reached = tuple(
            self.scenario.hosts[index].id
            for index in np.flatnonzero(self._reachable_hosts[: len(self.scenario.hosts)])
        )
        return (*reached, *self._default_watchlist())

    def monitored_hosts(self) -> tuple[str, ...]:
        """Return the hosts the defender is currently watching most closely."""

        if not self.defender.targeted_attention:
            return ()
        return tuple(
            host.id
            for index, host in enumerate(self.scenario.hosts)
            if self._host_attention[index] > 1.0
        )

    def _observed_risk(self) -> float:
        """Return the defender's noisy estimate of the attacker's detection risk."""

        if self.defender.observation_noise <= 0.0:
            return self._detection_risk
        noise = float(self._defender_rng.normal(0.0, self.defender.observation_noise))
        return min(1.0, max(0.0, self._detection_risk + noise))

    def _revoke_credential(self, index: int) -> None:
        """Drop one simulated credential, and the privilege it alone granted."""

        self._acquired_credentials[index] = 0
        self._revoked_credentials += 1
        privilege_id = self.scenario.credentials[index].privilege_id
        still_granted = any(
            self._acquired_credentials[other] and credential.privilege_id == privilege_id
            for other, credential in enumerate(self.scenario.credentials)
        )
        if not still_granted:
            self._acquired_privileges[self._privilege_index[privilege_id]] = 0

    def _raise_risk(self, increment: float, host_id: str | None = None) -> None:
        """Accumulate detection risk for one action, priced by where it happened.

        Attribution is what makes targeted monitoring routable. With a uniform
        defender every action costs the same wherever it lands, so the only way to
        lower risk is to act less; once attention is concentrated, acting on a blind
        host is cheaper than acting on a watched one and *where* becomes a decision.
        """

        scaled = increment * self._risk_scale * self._attention_multiplier(host_id)
        self._detection_risk = min(1.0, self._detection_risk + scaled)

    def _attention_multiplier(self, host_id: str | None) -> float:
        """Return how closely the defender is watching one host.

        An action that cannot be attributed to a host - privilege escalation walks the
        privilege graph, not the network - is priced at the defender's mean attention,
        so it is neither a free move nor a targeted one.
        """

        if not self.defender.targeted_attention:
            return 1.0
        if host_id is None:
            return float(np.mean(self._host_attention[: len(self.scenario.hosts)]))
        return float(self._host_attention[self._host_index[host_id]])

    def _success_probability(self, exploitability: float | None) -> float:
        if not self.dynamics.stochastic:
            return 1.0
        if exploitability is None or exploitability <= 0.0:
            return self.dynamics.base_success_probability
        return max(self.dynamics.minimum_success_probability, exploitability)

    def _succeeds(self, exploitability: float | None) -> bool:
        probability = self._success_probability(exploitability)
        if probability >= 1.0:
            return True
        return bool(self.np_random.random() < probability)

    def _target_id(self, action_type: int, target: int) -> str | None:
        """Return the scenario ID addressed by one action, or ``None`` when unused."""

        catalogue = self._target_ids[action_type]
        if target >= len(catalogue):
            return None
        return catalogue[target]

    def alert_level(self) -> int:
        """Quantize the true detection risk into the band the agent can observe."""

        levels = self.observation_config.alert_levels
        threshold = self.dynamics.detection_threshold
        band = int(self._detection_risk / threshold * levels)
        return min(levels - 1, max(0, band))

    def _observation(self) -> Observation:
        alert = np.zeros(self.observation_config.alert_levels, dtype=np.int8)
        alert[self.alert_level()] = 1
        observation: Observation = {
            "discovered_hosts": self._discovered_hosts.copy(),
            "reachable_hosts": self._reachable_hosts.copy(),
            "known_services": self._known_services.copy(),
            "enumerated_services": self._enumerated_services.copy(),
            "validated_vulnerabilities": self._validated_vulnerabilities.copy(),
            "acquired_credentials": self._acquired_credentials.copy(),
            "acquired_privileges": self._acquired_privileges.copy(),
            "collected_objectives": self._collected_objectives.copy(),
            "probed_hosts": self._failed_discovery.copy(),
            "alert_level": alert,
            "budget_fraction": np.array(
                [(self.step_budget - self._steps) / self.step_budget], dtype=np.float32
            ),
        }
        if self.observation_config.expose_exact_risk:
            observation["detection_risk"] = np.array([self._detection_risk], dtype=np.float32)
        if self.observation_config.expose_monitoring:
            observation["monitored_hosts"] = self._monitoring_observation()
        if self.observation_config.expose_watch_history:
            observation["watch_recency"] = self._watch_recency_observation()
        return observation

    def _monitoring_observation(self) -> np.ndarray[Any, np.dtype[np.int8]]:
        """Return the watched hosts the agent has discovered, as a padded bit vector."""

        watched = np.zeros(self._host_width, dtype=np.int8)
        if not self.defender.targeted_attention:
            return watched
        real = len(self.scenario.hosts)
        watched[:real] = (self._host_attention[:real] > 1.0).astype(np.int8)
        return watched * self._discovered_hosts

    def _watch_recency_observation(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Return, per discovered host, how long ago attention last landed there.

        0.0 means watched right now; 1.0 means never watched (or long enough ago that
        it no longer matters) - the same value, deliberately, since a memoryless
        ``monitored_hosts`` snapshot cannot tell those apart but a re-aiming defender's
        *pattern* is exactly what this channel is for (item 61). Undiscovered hosts are
        also pinned to 1.0 so this cannot leak the topology ahead of discovery.
        """

        recency = np.ones(self._host_width, dtype=np.float32)
        if not self.defender.targeted_attention:
            return recency
        real = len(self.scenario.hosts)
        since = self._steps - self._host_last_watched_step[:real]
        steps_since = np.clip(since, 0, self.step_budget)
        recency[:real] = steps_since.astype(np.float32) / self.step_budget
        recency[:real] = np.where(self._discovered_hosts[:real] > 0, recency[:real], 1.0)
        return recency

    def _host_detection_increment(self, host_id: str) -> float:
        probabilities = [
            control.detection_probability
            for control in self.scenario.security_controls
            if host_id in control.host_ids
        ]
        base = (
            0.05
            if not probabilities
            else max(0.02, (sum(probabilities) / len(probabilities)) * 0.15)
        )
        hardening = float(self._host_hardening[self._host_index[host_id]])
        return base * (1.0 + hardening)
