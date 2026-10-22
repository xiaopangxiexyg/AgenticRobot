"""Measured release windows; no simulator, controller or task-score authority."""

from dataclasses import replace
import math

import pytest

from robot_motion.object_release import (
    ObjectReleaseMeasurement,
    ObjectReleasePolicy,
    ObjectReleaseWindow,
)

pytestmark = pytest.mark.unit


def _policy(**changes):
    return ObjectReleasePolicy(duration_s=.6, min_measurements=60, max_sample_gap_s=.05,
        position_tolerance_m=.002, rotation_tolerance_rad=.05, **changes)


def _cut(stamp, *, x=0., angle=0., **changes):
    return ObjectReleaseMeasurement(stamp, (x, 0., .025, 0., 0., math.sin(angle/2), math.cos(angle/2)),
        support_contact=changes.get("support_contact", True),
        gripper_contact=changes.get("gripper_contact", False),
        other_contact=changes.get("other_contact", False))


def _stationary(window, *, start=0., x=0.):
    for index in range(61):
        window.observe(_cut(start + index * .01, x=x))
    return window.snapshot(now=start + .6)


def test_full_measured_supported_release_window_is_required_and_reads_do_not_count():
    window = ObjectReleaseWindow(_policy())
    assert window.snapshot(now=0.)["status"] == "not_observed"
    first = _cut(0.)
    for _ in range(100):
        window.observe(first)
        assert not window.snapshot(now=0.)["stable_supported_release"]
    assert window.snapshot(now=0.)["measurement_count"] == 1
    evidence = _stationary(window)
    assert evidence["stable_supported_release"]
    assert evidence["observed_duration_s"] == pytest.approx(.6)
    assert evidence["measurement_count"] == 61
    assert evidence["max_position_drift_m"] == 0.
    assert evidence["max_observed_sample_gap_s"] == pytest.approx(.01)
    assert evidence["window_reason"] == "complete"
    assert not window.snapshot(now=.651)["stable_supported_release"]
    assert window.snapshot(now=.651)["status"] == "stale"


@pytest.mark.parametrize("contact,status", [
    ({"gripper_contact": True}, "not_released"),
    ({"support_contact": False}, "not_supported"),
    ({"other_contact": True}, "unexpected_contact"),
    ({"support_contact": None}, "contact_evidence_unavailable"),
    ({"gripper_contact": None}, "contact_evidence_unavailable"),
    ({"other_contact": None}, "contact_evidence_unavailable"),
])
def test_contact_loss_and_unknowns_revoke_the_window_without_motion_fallback(contact, status):
    window = ObjectReleaseWindow(_policy())
    assert _stationary(window)["stable_supported_release"]
    window.observe(_cut(.61, **contact))
    snapshot = window.snapshot(now=.61)
    assert snapshot["status"] == status and snapshot["measurement_count"] == 0
    window.observe(_cut(.62))
    assert window.snapshot(now=.62)["status"] == "insufficient_window"


def test_known_contact_failure_does_not_become_unknown():
    window = ObjectReleaseWindow(_policy())
    window.observe(_cut(0., support_contact=None, gripper_contact=True))
    assert window.snapshot(now=0.)["status"] == "not_released"


def test_sampling_gap_cannot_bridge_two_windows_and_sparse_samples_are_not_a_hold():
    window = ObjectReleaseWindow(_policy())
    assert _stationary(window)["stable_supported_release"]
    window.observe(_cut(.7))
    snapshot = window.snapshot(now=.7)
    assert snapshot["max_observed_sample_gap_s"] is None
    assert snapshot["measurement_count"] == 1 and snapshot["window_reason"] == "observation_gap"
    assert not snapshot["stable_supported_release"]
    sparse = ObjectReleaseWindow(_policy())
    for index in range(21):
        sparse.observe(_cut(index * .04))
    snapshot = sparse.snapshot(now=.8)
    assert snapshot["observed_duration_s"] >= .6
    assert snapshot["status"] == "insufficient_window"


@pytest.mark.parametrize("mode", ["translation", "rotation", "out_and_back"])
def test_window_drift_cannot_be_hidden_by_small_adjacent_steps(mode):
    window = ObjectReleaseWindow(_policy())
    for index in range(61):
        x = .0001 * index if mode == "translation" else (.003 if index == 30 else 0.)
        angle = .001 * index if mode == "rotation" else 0.
        if mode == "rotation":
            x = 0.
        window.observe(_cut(index * .01, x=x, angle=angle))
    snapshot = window.snapshot(now=.6)
    assert snapshot["status"] == "moving" and not snapshot["stable_supported_release"]
    assert (snapshot["max_position_drift_m"] > .002
            or snapshot["max_rotation_drift_rad"] > .05)


def test_new_motion_revokes_stability_and_later_measured_rest_can_qualify():
    window = ObjectReleaseWindow(_policy())
    assert _stationary(window)["stable_supported_release"]
    window.observe(_cut(.61, x=.004))
    assert window.snapshot(now=.61)["status"] == "moving"
    # A genuinely new full stationary window is allowed; no object/hand anchor exists here.
    assert _stationary(window, start=.62, x=.004)["stable_supported_release"]


def test_quaternion_sign_is_not_motion_but_conflicting_same_cut_is_rejected():
    window = ObjectReleaseWindow(_policy())
    for index in range(61):
        sample = _cut(index * .01)
        if index % 2:
            sample = replace(sample, pose_xyz_quat_xyzw=(*sample.pose_xyz_quat_xyzw[:3], 0., 0., 0., -1.))
        window.observe(sample)
    assert window.snapshot(now=.6)["stable_supported_release"]
    with pytest.raises(ValueError, match="Conflicting"):
        window.observe(_cut(.6, x=.001))
    with pytest.raises(ValueError, match="backwards"):
        window.observe(_cut(.59))
    with pytest.raises(ValueError, match="future"):
        window.snapshot(now=.59)


def test_high_rate_history_is_bounded_without_inventing_missing_duration():
    window = ObjectReleaseWindow(_policy())
    for index in range(5000):
        window.observe(_cut(index * .00001))
    snapshot = window.snapshot(now=4999 * .00001)
    assert snapshot["measurement_count"] == 4096
    assert snapshot["observed_duration_s"] < .6
    assert not snapshot["stable_supported_release"]


@pytest.mark.parametrize("change", [
    {"duration_s": True}, {"duration_s": 0}, {"max_sample_gap_s": float("inf")},
    {"max_sample_gap_s": .4}, {"min_measurements": 2}, {"min_measurements": True},
    {"min_measurements": 4097}, {"rotation_tolerance_rad": 4.},
    {"position_tolerance_m": float("nan")},
])
def test_invalid_policy_is_not_a_runtime_default(change):
    with pytest.raises(ValueError):
        replace(_policy(), **change)


@pytest.mark.parametrize("change", [
    {"measured_at": True}, {"measured_at": -1.}, {"measured_at": float("nan")},
    {"support_contact": 1}, {"gripper_contact": "false"},
    {"pose_xyz_quat_xyzw": (0, 0, 0, 0, 0, 0, 2)},
    {"pose_xyz_quat_xyzw": (0, 0, 0, 0, 0, 0, float("inf"))},
    {"pose_xyz_quat_xyzw": (True, 0, 0, 0, 0, 0, 1)},
])
def test_invalid_measurements_cannot_establish_release(change):
    with pytest.raises(ValueError):
        replace(_cut(0.), **change)


def test_derived_position_overflow_is_explicit():
    window = ObjectReleaseWindow(_policy())
    window.observe(_cut(0., x=-1e308))
    window.observe(_cut(.01, x=1e308))
    with pytest.raises(ValueError, match="overflowed"):
        window.snapshot(now=.01)
