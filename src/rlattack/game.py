"""A two-player loop: a fixed attacker against a defender that adapts between episodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rlattack.agents import Agent
from rlattack.defender import BanditDefender, DefenderConfig
from rlattack.env import AttackPathEnv
from rlattack.evaluation import EpisodeOutcome, run_episode_outcome
from rlattack.experiment import ExperimentConfig
from rlattack.generator import generate_scenario
from rlattack.reward import build_reward_config


@dataclass(frozen=True)
class GameResult:
    """Outcome of one attacker-versus-adaptive-defender run."""

    episodes: int
    attacker_success_rate: float
    detection_rate: float
    mean_defender_reward: float
    pulls: dict[str, int] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    outcomes: tuple[EpisodeOutcome, ...] = ()

    @property
    def preferred_arm(self) -> str:
        """Return the arm the defender settled on."""

        return max(self.values, key=lambda label: self.values[label])


def defender_reward(outcome: EpisodeOutcome) -> float:
    """Score one episode from the defender's side.

    Stopping the attacker is what the defender is for, so a captured objective is a
    loss and a detection is a win. Nothing here reads the attacker's hidden state.
    """

    if outcome.success:
        return 0.0
    return 1.0 if outcome.detected else 0.5


def play(
    config: ExperimentConfig,
    agent_factory: Callable[[int], Agent],
    defender: BanditDefender | None = None,
    *,
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
    reward_config = build_reward_config(config.reward_strategy)
    observation_config = config.observation_config()
    dynamics = config.dynamics()

    outcomes: list[EpisodeOutcome] = []
    rewards: list[float] = []
    for episode in range(episodes):
        episode_seed = config.seed + episode
        index = opponent.select()
        arm: DefenderConfig = opponent.arms[index].config
        env = AttackPathEnv(
            generate_scenario(config.size, config.difficulty, episode_seed),
            step_budget=config.step_budget,
            reward_config=reward_config,
            dynamics=dynamics,
            observation_config=observation_config,
            defender=arm,
        )
        outcome = run_episode_outcome(agent_factory(episode_seed), env, episode_seed)
        reward = defender_reward(outcome)
        opponent.update(index, reward)
        outcomes.append(outcome)
        rewards.append(reward)

    return GameResult(
        episodes=episodes,
        attacker_success_rate=sum(outcome.success for outcome in outcomes) / episodes,
        detection_rate=sum(outcome.detected for outcome in outcomes) / episodes,
        mean_defender_reward=sum(rewards) / episodes,
        pulls=opponent.pulls,
        values=opponent.values,
        outcomes=tuple(outcomes),
    )
