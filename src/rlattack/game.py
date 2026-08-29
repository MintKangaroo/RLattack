"""A two-player loop: a fixed attacker against a defender that adapts between episodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from rlattack.agents import Agent
from rlattack.bandit import EpsilonGreedy
from rlattack.defender import BanditDefender, ContextualDefender, DefenderConfig
from rlattack.env import AttackPathEnv
from rlattack.evaluation import EpisodeOutcome, run_episode_outcome
from rlattack.experiment import AgentName, ExperimentConfig, create_agent
from rlattack.generator import generate_scenario
from rlattack.reward import build_reward_config

ATTACKER_ARMS: tuple[AgentName, ...] = (
    "random",
    "greedy",
    "rule-based",
    "shortest-path",
    "shortest-path-broad",
)


@dataclass
class BanditAttacker:
    """An attacker that learns which baseline works against the current defender.

    With a fixed attacker only the defender adapts, so the run measures one side
    learning against a stationary opponent. Letting the attacker pick a policy per
    episode and learn from the outcome makes both sides non-stationary for each other.
    """

    arms: tuple[AgentName, ...] = ATTACKER_ARMS
    exploration: float = 0.15

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one attacker arm is required")
        self._learner = EpsilonGreedy(list(self.arms), exploration=self.exploration)

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the learned estimates and restart the selection stream."""

        self._learner.reset(seed=seed)

    @property
    def pulls(self) -> dict[str, int]:
        """Return how many episodes each policy was selected for."""

        return self._learner.pulls

    @property
    def values(self) -> dict[str, float]:
        """Return the mean attacker reward estimated for each policy."""

        return self._learner.values

    def select(self) -> int:
        """Choose a policy for the next episode."""

        return self._learner.select()

    def update(self, index: int, reward: float) -> None:
        """Fold one episode's attacker reward into that policy's running mean."""

        self._learner.update(index, reward)


def attacker_reward(outcome: EpisodeOutcome) -> float:
    """Score one episode from the attacker's side: the mirror of the defender's."""

    if outcome.success:
        return 1.0
    return 0.0 if outcome.detected else 0.5


@dataclass(frozen=True)
class GameResult:
    """Outcome of one attacker-versus-adaptive-defender run."""

    episodes: int
    attacker_success_rate: float
    detection_rate: float
    mean_defender_reward: float
    pulls: dict[str, int] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    attacker_pulls: dict[str, int] = field(default_factory=dict)
    attacker_values: dict[str, float] = field(default_factory=dict)
    outcomes: tuple[EpisodeOutcome, ...] = ()

    @property
    def preferred_arm(self) -> str:
        """Return the arm the defender settled on, if it chose between arms."""

        if not self.values:
            return "contextual"
        return max(self.values, key=lambda label: self.values[label])


RESPONSE_COST = 0.01
FALSE_POSITIVE_COST = 0.05


def defender_reward(
    outcome: EpisodeOutcome,
    *,
    response_cost: float = RESPONSE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    """Score one episode from the defender's side.

    Stopping the attacker is what the defender is for, so a captured objective is a
    loss and a detection is a win. Responding is not free: without a cost, a defender
    that responds on every step would be trivially optimal, which no real defender is.
    False positives - responses fired while true risk was below the alert threshold -
    cost more than justified ones.

    Nothing here reads the attacker's hidden state.
    """

    if response_cost < 0.0 or false_positive_cost < 0.0:
        raise ValueError("response costs must not be negative")
    outcome_reward = 0.0 if outcome.success else (1.0 if outcome.detected else 0.5)
    justified = max(0, outcome.defender_actions - outcome.defender_false_positives)
    penalty = justified * response_cost + outcome.defender_false_positives * false_positive_cost
    return outcome_reward - penalty


def episode_defender_reward(info: dict[str, Any]) -> float:
    """Score a finished episode for the defender straight from the environment's info.

    The adversarial training loop has no :class:`EpisodeOutcome` to hand - Stable-
    Baselines3 owns the episode - so the same scoring is available from ``info``.
    """

    return defender_reward(
        EpisodeOutcome(
            seed=0,
            success=bool(info["objective_captured"]),
            detected=bool(info["detected"]),
            steps=int(info["steps"]),
            reward=0.0,
            detection_risk=float(info["detection_risk"]),
            path_cost=float(info["path_cost"]),
            defender_actions=int(info["defender_actions"]),
            defender_false_positives=int(info["defender_false_positives"]),
        )
    )


def play(
    config: ExperimentConfig,
    agent_factory: Callable[[int], Agent],
    defender: BanditDefender | ContextualDefender | None = None,
    *,
    attacker: BanditAttacker | None = None,
    episodes: int = 200,
    seed: int = 0,
) -> GameResult:
    """Run ``episodes`` of a fixed attacker against a defender that learns between them.

    The attacker policy is held fixed within a run; the defender picks a response
    policy per episode and updates it from the outcome.
    """

    if episodes < 1:
        raise ValueError("episodes must be positive")
    opponent = defender or BanditDefender()
    opponent.reset(seed=seed)
    if attacker is not None:
        attacker.reset(seed=seed + 1)
    reward_config = build_reward_config(config.reward_strategy)
    observation_config = config.observation_config()
    dynamics = config.dynamics()

    contextual = isinstance(opponent, ContextualDefender)
    outcomes: list[EpisodeOutcome] = []
    rewards: list[float] = []
    for episode in range(episodes):
        episode_seed = config.seed + episode
        index = 0
        arm: DefenderConfig | None = None
        policy: ContextualDefender | None = None
        if isinstance(opponent, ContextualDefender):
            opponent.start_episode()
            policy = opponent
        else:
            index = opponent.select()
            arm = opponent.arms[index].config
        env = AttackPathEnv(
            generate_scenario(config.size, config.difficulty, episode_seed),
            step_budget=config.step_budget,
            reward_config=reward_config,
            dynamics=dynamics,
            observation_config=observation_config,
            defender=arm,
            defender_policy=policy,
        )
        attacker_arm = attacker.select() if attacker is not None else 0
        agent = (
            create_agent(
                attacker.arms[attacker_arm],
                generate_scenario(config.size, config.difficulty, episode_seed),
                seed=episode_seed,
            )
            if attacker is not None
            else agent_factory(episode_seed)
        )
        outcome = run_episode_outcome(agent, env, episode_seed)
        if attacker is not None:
            attacker.update(attacker_arm, attacker_reward(outcome))
        reward = defender_reward(outcome)
        if isinstance(opponent, ContextualDefender):
            opponent.finish_episode(reward)
        else:
            opponent.update(index, reward)
        outcomes.append(outcome)
        rewards.append(reward)

    if contextual:
        pulls: dict[str, int] = {}
        values: dict[str, float] = {}
    else:
        bandit = cast(BanditDefender, opponent)
        pulls, values = bandit.pulls, bandit.values
    return GameResult(
        episodes=episodes,
        attacker_success_rate=sum(outcome.success for outcome in outcomes) / episodes,
        detection_rate=sum(outcome.detected for outcome in outcomes) / episodes,
        mean_defender_reward=sum(rewards) / episodes,
        pulls=pulls,
        values=values,
        attacker_pulls=attacker.pulls if attacker is not None else {},
        attacker_values=attacker.values if attacker is not None else {},
        outcomes=tuple(outcomes),
    )
