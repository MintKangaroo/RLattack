import pytest

from rlattack.bandit import EpsilonGreedy


def test_arm_set_and_exploration_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        EpsilonGreedy([])
    with pytest.raises(ValueError, match="exploration"):
        EpsilonGreedy(["a"], exploration=1.5)
    with pytest.raises(ValueError, match="arm index"):
        EpsilonGreedy(["a"]).update(3, 1.0)


def test_every_arm_is_tried_before_any_is_preferred() -> None:
    learner = EpsilonGreedy(["a", "b", "c"], exploration=0.0)
    learner.reset(seed=0)

    for expected in range(3):
        index = learner.select()
        assert index == expected
        learner.update(index, 0.0)

    assert sum(learner.pulls.values()) == 3


def test_estimates_are_running_means_and_drive_selection() -> None:
    learner = EpsilonGreedy(["a", "b"], exploration=0.0)
    learner.reset(seed=1)
    learner.update(0, 0.0)
    learner.update(1, 1.0)
    learner.update(1, 0.5)

    assert learner.values == {"a": 0.0, "b": 0.75}
    assert learner.preferred == "b"
    assert learner.select() == 1


def test_exploration_still_reaches_other_arms() -> None:
    learner = EpsilonGreedy(["a", "b", "c"], exploration=1.0)
    learner.reset(seed=3)
    for index in range(3):
        learner.update(index, 1.0 if index == 0 else 0.0)
    chosen = {learner.select() for _ in range(40)}

    assert len(chosen) > 1
