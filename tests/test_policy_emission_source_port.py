from __future__ import annotations

from robot_motion.deploy import (
    PassivePolicyEmissionObserverPort,
    PolicyEmissionSourcePort,
    require_passive_policy_emission_observer,
    require_policy_emission_source,
)
from control_station.sim_hil_profile.policy_source import (
    SimHILPolicySourcePort,
    require_sim_hil_policy_source,
)


class _PolicySourceFixture:
    ready = True
    last_error = None
    maximum_hold_sec = 0.2

    def start(self) -> None:
        return None

    def reset_for_activation(self) -> None:
        return None

    def step(self, *, now=None, step_index=None):
        del now, step_index
        return None

    def status(self, *, now=None):
        del now
        return {"ready": True}

    def close(self) -> None:
        return None


def test_sim_policy_source_name_is_the_shared_backend_neutral_port() -> None:
    source = _PolicySourceFixture()

    assert SimHILPolicySourcePort is PolicyEmissionSourcePort
    assert require_policy_emission_source(source) is source
    assert require_sim_hil_policy_source(source) is source


def test_policy_source_rejects_non_finite_or_non_positive_hold() -> None:
    source = _PolicySourceFixture()
    for maximum_hold in (0.0, -0.1, float("inf"), float("nan")):
        source.maximum_hold_sec = maximum_hold
        try:
            require_policy_emission_source(source)
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("invalid maximum_hold_sec was accepted")


def test_passive_observer_requires_explicit_bounded_nonblocking_contract() -> None:
    class _Observer:
        nonblocking_bounded = True

        def put_nowait(self, emission) -> None:
            del emission

    observer = _Observer()
    assert isinstance(observer, PassivePolicyEmissionObserverPort)
    assert require_passive_policy_emission_observer(observer) is observer

    observer.nonblocking_bounded = False
    try:
        require_passive_policy_emission_observer(observer)
    except ValueError as error:
        assert "nonblocking_bounded=True" in str(error)
    else:
        raise AssertionError("blocking observer declaration was accepted")
