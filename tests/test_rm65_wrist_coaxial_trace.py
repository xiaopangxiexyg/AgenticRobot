from __future__ import annotations

import numpy as np
import pytest

from teleop_runtime.debug.analyze_rm65_wrist_coaxial_trace import (
    analyze_qpos_series,
)
from robot_motion.rm65.wrist_coaxial import (
    canonicalize_rm65_wrist_coaxial_step,
    stabilize_rm65_wrist_q5_step,
)


def test_wrist_coaxial_shadow_preserves_observable_and_reduces_null_motion() -> None:
    qpos = np.zeros((5, 6), dtype=float)
    qpos[:, 3] = np.deg2rad([0.0, 10.0, 20.0, 30.0, 40.0])
    qpos[:, 5] = np.deg2rad([0.0, -8.0, -16.0, -24.0, -32.0])

    result = analyze_qpos_series(qpos)

    assert result["projected_samples"] == 4
    assert result["observable_q4_plus_q6_max_error_deg"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert (
        result["projected"]["near_soft_max_abs_null_step_deg"]
        < result["raw"]["near_soft_max_abs_null_step_deg"]
    )


def test_wrist_coaxial_shadow_rejects_malformed_series() -> None:
    with pytest.raises(ValueError, match="shape"):
        analyze_qpos_series(np.zeros((4, 5), dtype=float))


def test_wrist_coaxial_small_null_input_stays_more_responsive_than_branch_jump() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(3.25)

    small = reference.copy()
    small[3] += np.deg2rad(1.0)
    small[5] -= np.deg2rad(1.0)
    small_out, small_meta = canonicalize_rm65_wrist_coaxial_step(
        reference,
        small,
        soft_threshold_deg=4.0,
        hard_threshold_deg=3.0,
        null_motion_soft_limit_deg=5.0,
    )

    large = reference.copy()
    large[3] += np.deg2rad(7.0)
    large[5] -= np.deg2rad(7.0)
    large_out, large_meta = canonicalize_rm65_wrist_coaxial_step(
        reference,
        large,
        soft_threshold_deg=4.0,
        hard_threshold_deg=3.0,
        null_motion_soft_limit_deg=5.0,
    )

    assert small_meta["effective_null_scale"] > large_meta["effective_null_scale"]
    assert small_meta["effective_null_scale"] > 0.5
    assert large_meta["effective_null_scale"] == pytest.approx(
        large_meta["null_scale"]
    )
    np.testing.assert_allclose(
        small_out[3] + small_out[5],
        small[3] + small[5],
    )
    np.testing.assert_allclose(
        large_out[3] + large_out[5],
        large[3] + large[5],
    )


def test_wrist_q5_escape_prevents_smoothing_back_into_sdk_singularity() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(-2.2)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-2.9)

    escaped, metadata = stabilize_rm65_wrist_q5_step(
        reference,
        candidate,
        escape_threshold_deg=4.0,
        max_step_rad=np.deg2rad(4.0),
    )

    assert metadata["applied"] is True
    assert np.rad2deg(escaped[4]) == pytest.approx(-4.0)
    # The projection is a q5-only task relaxation; wrist-null coordinates are
    # not touched by this helper.
    np.testing.assert_allclose(escaped[[3, 5]], candidate[[3, 5]])


def test_wrist_q5_escape_keeps_previous_side_when_smoothing_crosses_zero() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(-2.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(1.0)

    escaped, metadata = stabilize_rm65_wrist_q5_step(
        reference,
        candidate,
        escape_threshold_deg=4.0,
        max_step_rad=np.deg2rad(4.0),
    )

    assert metadata["applied"] is True
    assert np.rad2deg(escaped[4]) == pytest.approx(-4.0)


def test_wrist_q5_escape_is_noop_when_both_endpoints_are_safe() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(-8.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-7.0)

    escaped, metadata = stabilize_rm65_wrist_q5_step(
        reference,
        candidate,
        escape_threshold_deg=4.0,
        max_step_rad=np.deg2rad(4.0),
    )

    assert metadata["applied"] is False
    np.testing.assert_allclose(escaped, candidate)


def test_wrist_q5_escape_does_not_pull_far_safe_sign_change_into_gate() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(20.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-20.0)

    escaped, metadata = stabilize_rm65_wrist_q5_step(
        reference,
        candidate,
        escape_threshold_deg=3.25,
        max_step_rad=np.deg2rad(4.0),
    )

    assert metadata["crossing"] is True
    assert metadata["crossing_near_band"] is False
    assert metadata["applied"] is False
    np.testing.assert_allclose(escaped, candidate)


def test_wrist_q5_escape_hysteresis_holds_inward_boundary_candidate() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(-5.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-3.7)

    escaped, metadata = stabilize_rm65_wrist_q5_step(
        reference,
        candidate,
        escape_threshold_deg=3.25,
        max_step_rad=np.deg2rad(4.0),
        hysteresis_release_threshold_deg=6.0,
    )

    assert metadata["hysteresis_hold"] is True
    assert metadata["reason"] == "hysteresis_hold_previous_side"
    assert np.rad2deg(escaped[4]) == pytest.approx(-5.0)


def test_wrist_q5_escape_hysteresis_releases_outside_band() -> None:
    reference = np.zeros(6, dtype=float)
    reference[4] = np.deg2rad(-5.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-6.5)

    escaped, metadata = stabilize_rm65_wrist_q5_step(
        reference,
        candidate,
        escape_threshold_deg=3.25,
        max_step_rad=np.deg2rad(4.0),
        hysteresis_release_threshold_deg=6.0,
    )

    assert metadata["hysteresis_hold"] is False
    assert metadata["reason"] == "outside_escape_band"
    np.testing.assert_allclose(escaped[4], candidate[4])
