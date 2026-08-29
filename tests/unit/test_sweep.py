from pathlib import Path

import pytest

from rlattack.sweep import DEFAULT_TRIALS, SweepTrial, trials_by_label


def test_a_trial_needs_a_label() -> None:
    with pytest.raises(ValueError, match="label"):
        SweepTrial("")


def test_a_trial_builds_its_own_training_configuration(tmp_path: Path) -> None:
    trial = SweepTrial("explorer", learning_rate=1e-3, entropy_coefficient=0.02)

    config = trial.config(seed=7, output_dir=tmp_path)

    assert config.seed == 7
    assert config.learning_rate == 1e-3
    assert config.entropy_coefficient == 0.02
    assert config.output_dir == tmp_path / "explorer"
    assert config.tensorboard_log == tmp_path / "explorer" / "tb"


def test_the_grid_varies_one_knob_at_a_time() -> None:
    labels = [trial.label for trial in DEFAULT_TRIALS]

    assert labels[0] == "baseline"
    assert len(set(labels)) == len(labels)
    baseline = DEFAULT_TRIALS[0]
    for trial in DEFAULT_TRIALS[1:]:
        changed = sum(
            getattr(trial, field) != getattr(baseline, field)
            for field in ("learning_rate", "rollout_steps", "batch_size", "entropy_coefficient")
        )
        assert changed >= 1


def test_trials_can_be_selected_by_label() -> None:
    assert trials_by_label() == DEFAULT_TRIALS
    assert [trial.label for trial in trials_by_label(("slow-lr", "baseline"))] == [
        "slow-lr",
        "baseline",
    ]
    with pytest.raises(ValueError, match="unknown sweep trial"):
        trials_by_label(("nonexistent",))
