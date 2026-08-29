"""A simulated defender that responds to the attacker's observed trajectory.

The defender is a modeled adversary inside the simulation. It changes nothing outside
the environment's arrays: it raises the detection sensitivity of hosts the attacker has
reached, and it can revoke a simulated credential so the attacker has to earn that
foothold again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rlattack.bandit import EpsilonGreedy


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

    ``response_budget`` caps how many responses one episode can absorb. A flat
    per-response penalty prices responding but still lets a defender respond without
    limit; a budget models the operational load a team can actually carry, and forces
    the defender to choose *when* to spend rather than only whether to act at all.
    """

    enabled: bool = False
    alert_threshold: float = 0.2
    hardening_step: float = 0.6
    revocation_probability: float = 0.35
    response_cooldown: int = 6
    response_latency: int = 3
    observation_noise: float = 0.06
    response_budget: int | None = None

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
        if self.response_budget is not None and self.response_budget < 1:
            raise ValueError("response_budget must be positive when set")

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


@dataclass(frozen=True)
class DefenderArm:
    """One named defender configuration the adaptive defender can choose."""

    label: str
    config: DefenderConfig

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a defender arm needs a label")
        if not self.config.enabled:
            raise ValueError("a defender arm must be enabled")


DEFAULT_ARMS: tuple[DefenderArm, ...] = (
    DefenderArm("patient", DefenderConfig(enabled=True, alert_threshold=0.35)),
    DefenderArm("balanced", DefenderConfig(enabled=True, alert_threshold=0.2)),
    DefenderArm(
        "trigger-happy",
        DefenderConfig(enabled=True, alert_threshold=0.08, response_cooldown=3),
    ),
    DefenderArm(
        "revoke-heavy",
        DefenderConfig(enabled=True, alert_threshold=0.2, revocation_probability=0.9),
    ),
    DefenderArm(
        "fast",
        DefenderConfig(enabled=True, alert_threshold=0.2, response_latency=0),
    ),
    # The two arms below specialize, so that attackers with different weaknesses are
    # punished by different defenders. Without that, one attacker dominates the grid and
    # the equilibrium is trivially pure.
    DefenderArm(
        "harden-only",
        DefenderConfig(
            enabled=True,
            alert_threshold=0.12,
            hardening_step=1.4,
            revocation_probability=0.0,
            response_cooldown=2,
        ),
    ),
    DefenderArm(
        "revoke-only",
        DefenderConfig(
            enabled=True,
            alert_threshold=0.2,
            hardening_step=0.0,
            revocation_probability=1.0,
            response_cooldown=2,
        ),
    ),
)


@dataclass
class BanditDefender:
    """A defender that learns which fixed response policy works against this attacker.

    A static defender is an opponent an attacker only has to beat once. This one picks
    an arm per episode and updates its estimate from the outcome, which makes the setup
    a two-player game. See :class:`ContextualDefender` for one that conditions on the
    episode instead of committing to a configuration for all of it.
    """

    arms: tuple[DefenderArm, ...] = DEFAULT_ARMS
    exploration: float = 0.15

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one defender arm is required")
        self._learner = EpsilonGreedy(
            [arm.label for arm in self.arms], exploration=self.exploration
        )

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the learned estimates and restart the selection stream."""

        self._learner.reset(seed=seed)

    @property
    def pulls(self) -> dict[str, int]:
        """Return how many episodes each arm was selected for."""

        return self._learner.pulls

    @property
    def values(self) -> dict[str, float]:
        """Return the mean defender reward estimated for each arm."""

        return self._learner.values

    def select(self) -> int:
        """Choose an arm for the next episode."""

        return self._learner.select()

    def update(self, index: int, reward: float) -> None:
        """Fold one episode's defender reward into that arm's running mean."""

        self._learner.update(index, reward)


DEFENDER_ACTIONS: tuple[str, ...] = ("none", "harden", "revoke")

ContextKey = tuple[int, bool, int, int]


@dataclass(frozen=True)
class DefenderContext:
    """What the defender can see about the episode so far.

    Deliberately coarse and observable: an alert band rather than the exact risk,
    whether there is anything to revoke, how far into the step budget the episode is,
    and how much of the response budget is already spent.
    """

    alert_band: int
    has_credentials: bool
    phase: int
    budget_pressure: int = 0

    @property
    def key(self) -> tuple[int, bool, int, int]:
        """Return the table key for this context."""

        return (self.alert_band, self.has_credentials, self.phase, self.budget_pressure)


@dataclass
class ContextualDefender:
    """A defender whose response depends on the episode so far, and which learns it.

    :class:`BanditDefender` commits to one configuration for a whole episode, so it can
    only learn *which fixed policy* is best. This one chooses per decision point from
    an observable context and learns a table of context to action, so it can hold fire
    early and respond once the episode looks dangerous.

    Credit assignment is Monte-Carlo: every (context, action) pair visited in an episode
    is updated toward that episode's terminal defender reward. With one scalar per
    episode there is nothing finer to assign.
    """

    config: DefenderConfig = field(default_factory=DefenderConfig.adaptive)
    exploration: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if not self.config.enabled:
            raise ValueError("a contextual defender must be enabled")
        self.reset()

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the learned table and restart the selection stream."""

        self._rng = np.random.default_rng(seed)
        self._counts: dict[tuple[ContextKey, int], int] = {}
        self._values: dict[tuple[ContextKey, int], float] = {}
        self._episode: list[tuple[ContextKey, int]] = []

    def start_episode(self) -> None:
        """Forget the previous episode's visited pairs."""

        self._episode = []

    @property
    def table(self) -> dict[tuple[ContextKey, int], float]:
        """Return the learned value of each (context, action) pair."""

        return dict(self._values)

    def action_for(self, context: DefenderContext) -> str:
        """Choose a response for this context and record the visit."""

        untried = [
            candidate
            for candidate in range(len(DEFENDER_ACTIONS))
            if (context.key, candidate) not in self._counts
        ]
        if untried:
            # Every action in a context is tried once before any of them is preferred;
            # otherwise a zero-initialised table with no exploration never moves off the
            # first action and learns nothing.
            index = untried[0]
        elif self._rng.random() < self.exploration:
            index = int(self._rng.integers(len(DEFENDER_ACTIONS)))
        else:
            scored = [
                self._values.get((context.key, candidate), 0.0)
                for candidate in range(len(DEFENDER_ACTIONS))
            ]
            index = scored.index(max(scored))
        self._episode.append((context.key, index))
        return DEFENDER_ACTIONS[index]

    def finish_episode(self, reward: float) -> None:
        """Fold the episode's terminal reward into every pair it visited."""

        for pair in self._episode:
            self._counts[pair] = self._counts.get(pair, 0) + 1
            previous = self._values.get(pair, 0.0)
            self._values[pair] = previous + (reward - previous) / self._counts[pair]
        self._episode = []
