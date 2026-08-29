import pytest

from rlattack.agents import Agent, GreedyAgent, ShortestPathOracle
from rlattack.defender import DEFAULT_ARMS, BanditDefender, DefenderArm, DefenderConfig
from rlattack.evaluation import EpisodeOutcome
from rlattack.experiment import ExperimentConfig
from rlattack.game import GameResult, defender_reward, play
from rlattack.generator import generate_scenario


def outcome(*, success: bool, detected: bool) -> EpisodeOutcome:
    return EpisodeOutcome(
        seed=0,
        success=success,
        detected=detected,
        steps=10,
        reward=0.0,
        detection_risk=0.0,
        path_cost=0.0,
    )


def greedy_factory(seed: int) -> Agent:
    del seed
    return GreedyAgent()


def test_defender_reward_scores_from_the_defenders_side() -> None:
    assert defender_reward(outcome(success=True, detected=False)) == 0.0
    assert defender_reward(outcome(success=False, detected=True)) == 1.0
    assert defender_reward(outcome(success=False, detected=False)) == 0.5


def test_defender_arms_are_validated() -> None:
    with pytest.raises(ValueError, match="label"):
        DefenderArm("", DefenderConfig(enabled=True))
    with pytest.raises(ValueError, match="must be enabled"):
        DefenderArm("off", DefenderConfig())


def test_bandit_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one defender arm"):
        BanditDefender(arms=())
    with pytest.raises(ValueError, match="exploration"):
        BanditDefender(exploration=1.5)
    with pytest.raises(ValueError, match="arm index"):
        BanditDefender().update(99, 1.0)


def test_the_bandit_tries_every_arm_before_exploiting() -> None:
    defender = BanditDefender(exploration=0.0)
    defender.reset(seed=0)

    for expected in range(len(DEFAULT_ARMS)):
        index = defender.select()
        assert index == expected
        defender.update(index, 0.0)

    assert sum(defender.pulls.values()) == len(DEFAULT_ARMS)


def test_the_bandit_converges_on_the_arm_that_scores() -> None:
    defender = BanditDefender(exploration=0.0)
    defender.reset(seed=1)
    for _ in range(40):
        index = defender.select()
        defender.update(index, 1.0 if index == 2 else 0.0)

    assert defender.pulls[DEFAULT_ARMS[2].label] > 30
    assert defender.values[DEFAULT_ARMS[2].label] == pytest.approx(1.0)


def test_exploration_still_samples_other_arms() -> None:
    defender = BanditDefender(exploration=1.0)
    defender.reset(seed=2)
    for _ in range(60):
        index = defender.select()
        defender.update(index, 1.0 if index == 0 else 0.0)

    assert sum(1 for count in defender.pulls.values() if count > 0) > 1


def test_a_game_run_is_reproducible_and_records_every_round() -> None:
    config = ExperimentConfig(
        size="small", difficulty="easy", seed=3, step_budget=60, stochastic=False
    )

    first = play(config, greedy_factory, episodes=12, seed=5)
    second = play(config, greedy_factory, episodes=12, seed=5)

    assert first == second
    assert isinstance(first, GameResult)
    assert first.episodes == 12
    assert len(first.outcomes) == 12
    assert sum(first.pulls.values()) == 12
    assert first.preferred_arm in first.values
    assert 0.0 <= first.attacker_success_rate <= 1.0


def test_the_defender_prefers_an_arm_that_stops_the_attacker() -> None:
    config = ExperimentConfig(size="medium", difficulty="hard", seed=0, step_budget=80)

    def oracle_factory(seed: int) -> Agent:
        return ShortestPathOracle(generate_scenario(config.size, config.difficulty, seed))

    result = play(config, oracle_factory, BanditDefender(exploration=0.1), episodes=60, seed=1)
    settled = result.pulls[result.preferred_arm]

    assert settled > 60 // len(DEFAULT_ARMS), "the defender must exploit, not just explore"


def test_a_game_needs_at_least_one_round() -> None:
    with pytest.raises(ValueError, match="episodes must be positive"):
        play(ExperimentConfig(), greedy_factory, episodes=0)
