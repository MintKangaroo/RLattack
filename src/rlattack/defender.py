"""A simulated defender that responds to the attacker's observed trajectory.

The defender is a modeled adversary inside the simulation. It changes nothing outside
the environment's arrays: it raises the detection sensitivity of hosts the attacker has
reached, and it can revoke a simulated credential so the attacker has to earn that
foothold again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DefenderConfig:
    """Response policy for the simulated defender.

    The defender is off by default so that a run without one stays the control
    condition of an experiment; enable it explicitly as the treatment.

    Two properties make evasion a timing problem rather than a threshold problem:

    * ``response_latency`` - a decision takes effect this many steps later, so an
      attacker that finishes quickly can outrun a response that has already been
      triggered.
    * ``observation_noise`` - the defender reads a noisy estimate of the attacker's
      detection risk, which produces both false positives below the threshold and
      missed responses above it. A defender with perfect risk telemetry is a stronger
      adversary than any real one.
    """

    enabled: bool = False
    alert_threshold: float = 0.2
    hardening_step: float = 0.6
    revocation_probability: float = 0.35
    response_cooldown: int = 6
    response_latency: int = 3
    observation_noise: float = 0.06

    def __post_init__(self) -> None:
        if not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError("alert_threshold must be in [0, 1]")
        if self.hardening_step < 0.0:
            raise ValueError("hardening_step must not be negative")
        if not 0.0 <= self.revocation_probability <= 1.0:
            raise ValueError("revocation_probability must be in [0, 1]")
        if self.response_cooldown < 1:
            raise ValueError("response_cooldown must be positive")
        if self.response_latency < 0:
            raise ValueError("response_latency must not be negative")
        if self.observation_noise < 0.0:
            raise ValueError("observation_noise must not be negative")

    @classmethod
    def adaptive(cls) -> DefenderConfig:
        """Return the default enabled defender."""

        return cls(enabled=True)


@dataclass(frozen=True)
class DefenderState:
    """The observable slice of the episode the defender reacts to.

    ``observed_risk`` is the defender's *estimate*, not the attacker's true detection
    risk; the environment adds the configured observation noise before deciding.
    """

    observed_risk: float
    steps_since_response: int
    acquired_credentials: tuple[int, ...]


@dataclass(frozen=True)
class DefenderResponse:
    """One defender decision. ``none`` when the defender stays passive this step."""

    harden: bool = False
    revoke_credential: int | None = None

    @property
    def name(self) -> str:
        """Return a stable label for traces and reports."""

        if self.revoke_credential is not None:
            return "revoke_credential"
        if self.harden:
            return "harden_monitoring"
        return "none"


def decide_response(
    config: DefenderConfig,
    state: DefenderState,
    rng: np.random.Generator,
) -> DefenderResponse:
    """Decide how the defender reacts to the attacker's current trajectory.

    The decision depends only on what a defender could plausibly see - a noisy estimate
    of accumulated detection risk - never on the attacker's hidden intent or the
    objective it chose.
    """

    if not config.enabled:
        return DefenderResponse()
    if state.observed_risk < config.alert_threshold:
        return DefenderResponse()
    if state.steps_since_response < config.response_cooldown:
        return DefenderResponse()
    if state.acquired_credentials and rng.random() < config.revocation_probability:
        index = int(rng.choice(np.asarray(state.acquired_credentials)))
        return DefenderResponse(harden=True, revoke_credential=index)
    return DefenderResponse(harden=True)
