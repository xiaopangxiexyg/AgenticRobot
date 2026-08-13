from __future__ import annotations

import inspect

import pytest

from teleop_runtime.deploy import run_policy


class _BlockingObserverDeclaration:
    nonblocking_bounded = False

    def put_nowait(self, emission) -> None:
        del emission


def test_real_deploy_policy_emission_observer_is_default_off_and_passive() -> None:
    with pytest.raises(ValueError, match="nonblocking_bounded=True"):
        run_policy.main([], policy_emission_observer=_BlockingObserverDeclaration())

    source = inspect.getsource(run_policy.main)
    assert source.count("policy_emission_observer.put_nowait(") == 1
    assert "and action_step.emission is not None" in source
    assert "passive_policy_emission_observer_error" in source
    assert "policy_emission_observer:" in source
