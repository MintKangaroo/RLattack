import numpy as np
import pytest

from rlattack.defender import (
    DefenderConfig,
    DefenderResponse,
    DefenderState,
    decide_response,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def alerted(credentials: tuple[int, ...] = (0, 1)) -> DefenderState:
    return DefenderState(
        observed_risk=0.9,
        steps_since_response=99,
        acquired_credentials=credentials,
    )


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="response_latency"):
        DefenderConfig(response_latency=-1)
    with pytest.raises(ValueError, match="observation_noise"):
        DefenderConfig(observation_noise=-0.1)
    with pytest.raises(ValueError, match="alert_threshold"):
        DefenderConfig(alert_threshold=1.5)
    with pytest.raises(ValueError, match="hardening_step"):
        DefenderConfig(hardening_step=-1.0)
    with pytest.raises(ValueError, match="revocation_probability"):
        DefenderConfig(revocation_probability=2.0)
    with pytest.raises(ValueError, match="response_cooldown"):
        DefenderConfig(response_cooldown=0)


def test_a_disabled_defender_is_the_control_condition() -> None:
    response = decide_response(DefenderConfig(), alerted(), rng())

    assert response == DefenderResponse()
    assert response.name == "none"


def test_the_defender_waits_below_the_alert_threshold() -> None:
    state = DefenderState(
        observed_risk=0.01,
        steps_since_response=99,
        acquired_credentials=(0,),
    )

    assert decide_response(DefenderConfig.adaptive(), state, rng()).name == "none"


def test_the_defender_respects_its_cooldown() -> None:
    state = DefenderState(
        observed_risk=0.9,
        steps_since_response=1,
        acquired_credentials=(0,),
    )

    assert decide_response(DefenderConfig.adaptive(), state, rng()).name == "none"


def test_the_defender_hardens_when_it_cannot_revoke() -> None:
    config = DefenderConfig(enabled=True, revocation_probability=0.0)

    response = decide_response(config, alerted(), rng())

    assert response.name == "harden_monitoring"
    assert response.revoke_credential is None

    empty = decide_response(DefenderConfig.adaptive(), alerted(credentials=()), rng())

    assert empty.name == "harden_monitoring"


def test_the_defender_revokes_an_acquired_credential() -> None:
    config = DefenderConfig(enabled=True, revocation_probability=1.0)

    response = decide_response(config, alerted(credentials=(2,)), rng())

    assert response.name == "revoke_credential"
    assert response.revoke_credential == 2
    assert response.harden is True
