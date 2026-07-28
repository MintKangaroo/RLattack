import pytest
from pydantic import ValidationError

from rlattack.reward import build_reward_config, make_experiment_record


@pytest.mark.parametrize("strategy", ("sparse", "shaped", "risk-aware", "cost-aware"))
def test_reward_strategies_are_explicit(strategy: str) -> None:
    config = build_reward_config(strategy)  # type: ignore[arg-type]

    assert config.objective > 0


def test_reward_strategy_values_capture_ablation_differences() -> None:
    sparse = build_reward_config("sparse")
    shaped = build_reward_config("shaped")
    risk_aware = build_reward_config("risk-aware")
    cost_aware = build_reward_config("cost-aware")

    assert sparse.new_host == 0
    assert shaped.new_host > sparse.new_host
    assert risk_aware.detection_risk < shaped.detection_risk
    assert cost_aware.step_cost < shaped.step_cost


def test_experiment_record_is_serializable_and_immutable() -> None:
    record = make_experiment_record("risk-aware", "scenario-1", seed=3)

    assert record.model_dump()["strategy"] == "risk-aware"
    assert '"scenario_id":"scenario-1"' in record.model_dump_json()
    with pytest.raises(ValidationError):
        record.seed = 4


def test_invalid_reward_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_reward_config("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        make_experiment_record("shaped", "", seed=0)
