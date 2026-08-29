import pytest

from rlattack.agents import Agent, GreedyAgent, ShortestPathOracle
from rlattack.defender import (
    DEFAULT_ARMS,
    DEFENDER_ACTIONS,
    BanditDefender,
    ContextualDefender,
    DefenderArm,
    DefenderConfig,
    DefenderContext,
)
from rlattack.evaluation import EpisodeOutcome
from rlattack.experiment import ExperimentConfig
from rlattack.game import (
    BanditAttacker,
    GameResult,
    attacker_reward,
    defender_reward,
    episode_defender_reward,
    play,
)
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


def test_responding_is_not_free() -> None:
    """Without a response cost, a defender that always responds is trivially optimal."""

    quiet = EpisodeOutcome(
        seed=0,
        success=False,
        detected=True,
        steps=10,
        reward=0.0,
        detection_risk=0.0,
        path_cost=0.0,
        defender_actions=1,
        defender_false_positives=0,
    )
    noisy = EpisodeOutcome(
        seed=0,
        success=False,
        detected=True,
        steps=10,
        reward=0.0,
        detection_risk=0.0,
        path_cost=0.0,
        defender_actions=20,
        defender_false_positives=15,
    )

    assert defender_reward(quiet) > defender_reward(noisy)
    assert defender_reward(quiet, response_cost=0.0, false_positive_cost=0.0) == 1.0
    with pytest.raises(ValueError, match="must not be negative"):
        defender_reward(quiet, response_cost=-1.0)


def test_a_false_positive_costs_more_than_a_justified_response() -> None:
    def scored(actions: int, false_positives: int) -> float:
        return defender_reward(
            EpisodeOutcome(
                seed=0,
                success=False,
                detected=False,
                steps=10,
                reward=0.0,
                detection_risk=0.0,
                path_cost=0.0,
                defender_actions=actions,
                defender_false_positives=false_positives,
            )
        )

    assert scored(4, 0) > scored(4, 4)


def test_the_contextual_defender_learns_a_table_and_is_reproducible() -> None:
    config = ExperimentConfig(size="small", difficulty="easy", seed=2, step_budget=60)

    first = play(config, greedy_factory, ContextualDefender(), episodes=10, seed=3)
    second = play(config, greedy_factory, ContextualDefender(), episodes=10, seed=3)

    assert first == second
    assert first.pulls == {}
    assert first.preferred_arm == "contextual"


def test_the_contextual_defender_is_validated_and_records_visits() -> None:
    with pytest.raises(ValueError, match="exploration"):
        ContextualDefender(exploration=2.0)
    with pytest.raises(ValueError, match="must be enabled"):
        ContextualDefender(config=DefenderConfig())

    defender = ContextualDefender(exploration=0.0)
    defender.reset(seed=0)
    defender.start_episode()
    context = DefenderContext(alert_band=1, has_credentials=True, phase=0)

    assert defender.action_for(context) in DEFENDER_ACTIONS

    defender.finish_episode(1.0)
    table = defender.table

    assert len(table) == 1
    assert next(iter(table.values())) == 1.0
    assert context.key == (1, True, 0, 0)


def test_the_contextual_defender_exploits_what_it_learned() -> None:
    defender = ContextualDefender(exploration=0.0)
    defender.reset(seed=0)
    context = DefenderContext(alert_band=2, has_credentials=True, phase=2)
    for _ in range(6):
        defender.start_episode()
        chosen = defender.action_for(context)
        defender.finish_episode(1.0 if chosen == "revoke" else 0.0)

    defender.start_episode()

    assert defender.action_for(context) == "revoke"


def test_attacker_reward_mirrors_the_defenders() -> None:
    assert attacker_reward(outcome(success=True, detected=False)) == 1.0
    assert attacker_reward(outcome(success=False, detected=True)) == 0.0
    assert attacker_reward(outcome(success=False, detected=False)) == 0.5


def test_the_attacker_bandit_is_validated() -> None:
    with pytest.raises(ValueError, match="at least one attacker arm"):
        BanditAttacker(arms=())
    with pytest.raises(ValueError, match="exploration"):
        BanditAttacker(exploration=-1.0)


def test_a_learning_attacker_rediscovers_the_strongest_baseline() -> None:
    config = ExperimentConfig(size="medium", difficulty="hard", seed=0, step_budget=80)
    attacker = BanditAttacker()

    result = play(
        config,
        greedy_factory,
        ContextualDefender(),
        attacker=attacker,
        episodes=120,
        seed=1,
    )

    assert sum(result.attacker_pulls.values()) == 120
    assert result.attacker_pulls["shortest-path"] > 60
    assert max(result.attacker_values, key=lambda arm: result.attacker_values[arm]) == (
        "shortest-path"
    )


def test_a_fixed_attacker_reports_no_learned_preference() -> None:
    config = ExperimentConfig(size="small", difficulty="easy", seed=1, step_budget=60)

    result = play(config, greedy_factory, episodes=6, seed=2)

    assert result.attacker_pulls == {}
    assert result.attacker_values == {}


def test_budget_pressure_is_part_of_the_defender_context() -> None:
    relaxed = DefenderContext(alert_band=1, has_credentials=True, phase=1, budget_pressure=0)
    strained = DefenderContext(alert_band=1, has_credentials=True, phase=1, budget_pressure=2)

    assert relaxed.key != strained.key

    defender = ContextualDefender(exploration=0.0)
    defender.reset(seed=0)
    defender.start_episode()
    defender.action_for(relaxed)
    defender.action_for(strained)
    defender.finish_episode(1.0)

    assert len(defender.table) == 2, "budget pressure must select a distinct policy entry"


def test_an_episode_can_be_scored_straight_from_info() -> None:
    info = {
        "objective_captured": False,
        "detected": True,
        "steps": 12,
        "detection_risk": 0.9,
        "path_cost": 3.0,
        "defender_actions": 2,
        "defender_false_positives": 1,
    }

    scored = episode_defender_reward(info)
    equivalent = defender_reward(
        EpisodeOutcome(
            seed=0,
            success=False,
            detected=True,
            steps=12,
            reward=0.0,
            detection_risk=0.9,
            path_cost=3.0,
            defender_actions=2,
            defender_false_positives=1,
        )
    )

    assert scored == equivalent
