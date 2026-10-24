"""Exact causal planner continuation; no simulator, hardware or dispatch."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import fields

import numpy as np
import pytest

from robot_motion.position_control.trajectory import (
    AccelerationLimitedJointTrajectoryPlanner,
    JointTrajectoryLimits,
)
from robot_motion.position_control.trajectory_checkpoint import (
    capture_acceleration_planner,
    restore_acceleration_planner,
)

pytestmark = pytest.mark.unit


def _configuration():
    return {
        "limits": JointTrajectoryLimits([1.0, 1.5], [2.0, 3.0], [0.02, 0.03]),
        "velocity_ramp_sec": 0.1,
        "target_velocity_filter_tau_sec": 0.03,
        "reference_period_sec": 1 / 60,
        "monotonic_clock": lambda: 100.0,
    }


def _planner():
    planner = AccelerationLimitedJointTrajectoryPlanner(**_configuration())
    planner.reset([0.0, 0.0], monotonic_time=100.0)
    for i in range(1, 9):
        now = 100.0 + i / 120
        if i in (1, 3, 5):
            planner.accept_target(
                [0.1 + i / 30, -0.2 - i / 30],
                sequence=i,
                published_monotonic_time=now,
                monotonic_time=now,
            )
        proposal = planner.propose(monotonic_time=now)
        planner.commit(proposal, accepted_monotonic_time=now)
    return planner


def _same_proposal(first, second):
    for field in fields(first):
        a, b = getattr(first, field.name), getattr(second, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b, field.name


def test_complete_planner_history_survives_json_and_identical_future_inputs():
    source = _planner()
    state = capture_acceleration_planner(source)
    assert any(state["vectors"]["_qd"])
    assert any(state["vectors"]["_target_qd_reference"])
    candidate = restore_acceleration_planner(
        json.loads(json.dumps(state)), **_configuration()
    )
    assert capture_acceleration_planner(candidate) == state
    assert candidate is not source
    assert not np.shares_memory(candidate._q, source._q)
    for i in range(9, 31):
        now = 100.0 + i / 120
        for planner in (source, candidate):
            if i == 12:
                planner.accept_target(
                    [-0.15, 0.25],
                    sequence=i,
                    published_monotonic_time=now,
                    monotonic_time=now,
                    mode="tracking",
                    reason="reversal",
                )
            if i == 22:
                planner.quick_stop(sequence=i, monotonic_time=now, reason="hold")
        first = source.propose(monotonic_time=now)
        second = candidate.propose(monotonic_time=now)
        _same_proposal(first, second)
        source.commit(first, accepted_monotonic_time=now)
        candidate.commit(second, accepted_monotonic_time=now)
        assert capture_acceleration_planner(source) == capture_acceleration_planner(
            candidate
        )


def test_two_planner_branches_and_saved_document_are_independent():
    source = _planner()
    state = capture_acceleration_planner(source)
    original = copy.deepcopy(state)
    a = restore_acceleration_planner(state, **_configuration())
    b = restore_acceleration_planner(state, **_configuration())
    a.quick_stop(sequence=9, monotonic_time=101.0, reason="branch-a")
    state["vectors"]["_q"][0] += 1
    assert (
        capture_acceleration_planner(source)
        == capture_acceleration_planner(b)
        == original
    )
    assert capture_acceleration_planner(a) != original


def test_planner_continues_in_fresh_interpreter_without_replaying_targets(tmp_path):
    source = _planner()
    path = tmp_path / "planner.json"
    path.write_text(json.dumps(capture_acceleration_planner(source)))
    now = 100.1
    proposal = source.propose(monotonic_time=now)
    source.commit(proposal, accepted_monotonic_time=now)
    code = """
import json, sys
from tests.test_robot_motion_trajectory_checkpoint import _configuration
from robot_motion.position_control.trajectory_checkpoint import (
    capture_acceleration_planner, restore_acceleration_planner,
)
with open(sys.argv[1]) as stream:
    source = restore_acceleration_planner(json.load(stream), **_configuration())
proposal = source.propose(monotonic_time=100.1)
source.commit(proposal, accepted_monotonic_time=100.1)
print(json.dumps(capture_acceleration_planner(source)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(result.stdout) == capture_acceleration_planner(source)


def test_pending_command_transaction_and_unknown_owner_state_are_rejected():
    planner = _planner()
    proposal = planner.propose(monotonic_time=100.1)
    with pytest.raises(ValueError, match="completed command"):
        capture_acceleration_planner(planner)
    planner.reject(proposal)
    planner._new_causal_field = 1
    with pytest.raises(ValueError, match="layout differs"):
        capture_acceleration_planner(planner)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(extra=0),
        lambda s: s["scalars"].update(_generation=True),
        lambda s: s["scalars"].update(_speed_scale=float("nan")),
        lambda s: s["scalars"].update(_speed_scale=None),
        lambda s: s["scalars"].update(_target_velocity_filter_alpha=2.0),
        lambda s: s["scalars"].update(_target_published_monotonic=None),
        lambda s: s["scalars"].update(_last_safe_target_published_monotonic=None),
        lambda s: s["vectors"].update(_last_safe_target_q=None),
        lambda s: s["scalars"].pop("_last_safe_target_published_monotonic"),
        lambda s: s["vectors"].update(_q=[1.0]),
        lambda s: s["vectors"].update(_braking_guard_direction=[1.0, 0.0]),
        lambda s: s["vectors"].update(_q=[True, 1.0]),
        lambda s: s["optional"].update(_last_anchor_resynced=1),
        lambda s: s["configuration"].update(velocity_ramp_sec=0.2),
        lambda s: s["configuration"]["max_velocity_rad_s"].__setitem__(0, 100.0),
    ],
)
def test_invalid_checkpoint_does_not_change_live_source(mutation):
    source = _planner()
    original = capture_acceleration_planner(source)
    damaged = copy.deepcopy(original)
    mutation(damaged)
    with pytest.raises(ValueError):
        restore_acceleration_planner(damaged, **_configuration())
    assert capture_acceleration_planner(source) == original


def test_uninitialized_or_subclassed_planner_is_not_admitted():
    with pytest.raises(RuntimeError, match="reset"):
        capture_acceleration_planner(
            AccelerationLimitedJointTrajectoryPlanner(**_configuration())
        )
    with pytest.raises(ValueError, match="exact acceleration"):
        capture_acceleration_planner(object())


@pytest.mark.parametrize("boundary", ["reset", "pause", "quick_stop"])
def test_initialized_idle_and_hold_boundaries_remain_restorable(boundary):
    source = _planner()
    if boundary == "reset":
        source.reset([0.1, 0.2], monotonic_time=101.0)
    elif boundary == "pause":
        source.pause_reference(monotonic_time=101.0, reason="waiting")
    else:
        source.quick_stop(sequence=9, monotonic_time=101.0, reason="hold")
    state = capture_acceleration_planner(source)
    restored = restore_acceleration_planner(
        json.loads(json.dumps(state)), **_configuration()
    )
    assert capture_acceleration_planner(restored) == state


def test_reset_checkpoint_cannot_retain_a_phantom_publication_watermark():
    source = _planner()
    source.reset([0.0, 0.0], monotonic_time=101.0)
    state = capture_acceleration_planner(source)
    state["scalars"]["_target_published_monotonic"] = 1000.0
    with pytest.raises(ValueError, match="target identity"):
        restore_acceleration_planner(state, **_configuration())
