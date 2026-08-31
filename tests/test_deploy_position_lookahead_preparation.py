from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_motion.deploy.actions import FilteredNormalizedAction, NormalizedAction
from robot_motion.deploy.canonical_action import CanonicalDeployActionResult
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import ACTION_DIM
from teleop_runtime.deploy import run_policy


def _action(value: float) -> NormalizedAction:
    return NormalizedAction(
        vector=np.full((ACTION_DIM,), value, dtype=np.float32),
    )


def _canonical(
    value: float, *, faults: tuple[str, ...] = ()
) -> CanonicalDeployActionResult:
    return CanonicalDeployActionResult(
        filtered=FilteredNormalizedAction(action=_action(value)),
        faults=faults,
    )


def _observation() -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=12.5,
        step_index=7,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
    )


class _Pipeline:
    def __init__(self, outputs: tuple[CanonicalDeployActionResult, ...]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[tuple[Any, DeploymentObservation, str]] = []

    def prepare(
        self,
        raw_action: Any,
        observation: DeploymentObservation,
        *,
        policy_name: str,
    ) -> CanonicalDeployActionResult:
        self.calls.append((raw_action, observation, policy_name))
        return next(self._outputs)


def test_position_lookahead_preparation_uses_each_safe_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_actions = (_action(10.0), _action(11.0), _action(12.0))
    prepared_actions = (_canonical(1.0), _canonical(2.0), _canonical(3.0))
    pipeline = _Pipeline(prepared_actions)
    phase_timing: dict[str, Any] = {}
    clock = iter((20.0, 20.001, 21.0, 21.002, 22.0, 22.003))
    monkeypatch.setattr(run_policy.time, "monotonic", clock.__next__)

    result = run_policy._prepare_position_lookahead_chain(
        raw_actions,
        _observation(),
        _canonical(0.0),
        pipeline=pipeline,
        policy_name="causal-policy",
        phase_timing=phase_timing,
    )

    assert tuple(item[0] for item in result) == prepared_actions
    assert [call[0] for call in pipeline.calls] == list(raw_actions)
    assert [call[2] for call in pipeline.calls] == ["causal-policy"] * 3
    for offset, (_, successor_observation, _) in enumerate(pipeline.calls):
        np.testing.assert_array_equal(
            successor_observation.last_action,
            np.full((ACTION_DIM,), float(offset), dtype=np.float32),
        )
        assert result[offset][1] is successor_observation
    assert phase_timing == pytest.approx(
        {
            "lookahead_prepare_1_ms": 1.0,
            "lookahead_prepare_2_ms": 2.0,
            "lookahead_prepare_3_ms": 3.0,
        }
    )


@pytest.mark.parametrize("fault_at", (0, 1, 2))
def test_position_lookahead_preparation_stops_at_first_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault_at: int,
) -> None:
    raw_actions = (_action(10.0), _action(11.0), _action(12.0))
    outputs = tuple(
        _canonical(float(index + 1), faults=("blocked",) if index == fault_at else ())
        for index in range(3)
    )
    pipeline = _Pipeline(outputs)
    phase_timing: dict[str, Any] = {}
    clock = iter(float(value) for pair in range(3) for value in (pair, pair + 0.001))
    monkeypatch.setattr(run_policy.time, "monotonic", clock.__next__)

    result = run_policy._prepare_position_lookahead_chain(
        raw_actions,
        _observation(),
        _canonical(0.0),
        pipeline=pipeline,
        policy_name="fault-policy",
        phase_timing=phase_timing,
    )

    assert len(result) == fault_at
    assert len(pipeline.calls) == fault_at + 1
    assert set(phase_timing) == {
        f"lookahead_prepare_{index}_ms" for index in range(1, fault_at + 2)
    }
