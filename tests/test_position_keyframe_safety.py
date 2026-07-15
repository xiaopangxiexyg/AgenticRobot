from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from robot_motion.position_control.safety import (
    PositionKeyframeSafetyConfig,
    limit_joint_keyframes,
    limit_pose_keyframes,
)


def test_pose_keyframe_guard_passes_safe_action_without_hidden_smoothing() -> None:
    reference = np.asarray([0.20, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0])
    target = np.asarray([0.22, 0.01, 0.31, 0.0, 0.0, 0.0, 1.0])

    result = limit_pose_keyframes(
        {"right": target},
        reference_by_side={"right": reference},
    )

    np.testing.assert_allclose(result.values_by_side["right"], target)
    assert result.modified_by_side == {"right": False}
    assert result.reasons_by_side == {"right": ()}


def test_pose_keyframe_guard_applies_only_explicit_workspace_and_step_limits() -> None:
    reference = np.asarray([0.20, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0])
    target = np.concatenate(
        (
            np.asarray([1.0, 0.0, 0.30]),
            R.from_euler("z", 90.0, degrees=True).as_quat(),
        )
    )
    config = PositionKeyframeSafetyConfig(
        workspace_min=(-0.5, -0.5, 0.0),
        workspace_max=(0.5, 0.5, 0.8),
        max_translation_step_m=0.1,
        max_rotation_step_deg=10.0,
    )

    result = limit_pose_keyframes(
        {"right": target},
        reference_by_side={"right": reference},
        config=config,
    )

    safe = result.values_by_side["right"]
    assert np.linalg.norm(safe[:3] - reference[:3]) == pytest.approx(0.1)
    rotation_step = (
        R.from_quat(safe[3:7]) * R.from_quat(reference[3:7]).inv()
    ).magnitude()
    assert np.rad2deg(rotation_step) == pytest.approx(10.0)
    assert result.reasons_by_side["right"] == (
        "workspace_soft_limit",
        "translation_step_limit",
        "rotation_step_limit",
    )


def test_joint_keyframe_guard_clips_absolute_and_delta_limits() -> None:
    config = PositionKeyframeSafetyConfig(
        max_joint_step_rad=0.2,
        joint_lower_rad=(-1.0,) * 6,
        joint_upper_rad=(1.0,) * 6,
    )
    result = limit_joint_keyframes(
        {"left": np.asarray([2.0, -2.0, 0.5, 0.0, 0.0, 0.0])},
        reference_by_side={"left": np.zeros(6)},
        config=config,
    )

    np.testing.assert_allclose(
        result.values_by_side["left"],
        [0.2, -0.2, 0.2, 0.0, 0.0, 0.0],
    )
    assert result.reasons_by_side["left"] == (
        "absolute_joint_limit",
        "joint_step_limit",
    )


def test_soft_workspace_is_l2_bounded_after_translation_step_limit() -> None:
    config = PositionKeyframeSafetyConfig(
        workspace_min=(-0.5, -0.5, 0.0),
        workspace_max=(0.5, 0.5, 0.8),
        workspace_soft_margin_m=0.05,
        max_translation_step_m=0.1,
    )
    reference = np.asarray([0.5, 0.5, 0.8, 0.0, 0.0, 0.0, 1.0])
    # The raw corner is 0.173m outside the hard box.  Independent axis
    # expansion would incorrectly accept a 0.086m soft-space distance.
    target = np.asarray([0.6, 0.6, 0.9, 0.0, 0.0, 0.0, 1.0])

    result = limit_pose_keyframes(
        {"right": target},
        reference_by_side={"right": reference},
        config=config,
    )

    safe = result.values_by_side["right"]
    hard_projection = np.clip(safe[:3], config.workspace_min, config.workspace_max)
    assert np.linalg.norm(safe[:3] - hard_projection) <= 0.05 + 1e-12
    assert np.linalg.norm(safe[:3] - reference[:3]) <= 0.1 + 1e-12
    assert "workspace_soft_limit" in result.reasons_by_side["right"]


def test_joint_keyframe_faults_when_hard_limits_and_step_interval_do_not_intersect() -> None:
    config = PositionKeyframeSafetyConfig(
        max_joint_step_rad=0.1,
        joint_lower_rad=(-1.0,) * 6,
        joint_upper_rad=(1.0,) * 6,
    )
    result = limit_joint_keyframes(
        {"left": np.zeros(6)},
        reference_by_side={"left": np.full(6, 1.15)},
        config=config,
    )

    assert result.valid is False
    assert result.faults_by_side["left"] == ("joint_constraint_infeasible",)
    # No possibly unsafe output is consumed by callers once invalid; the
    # returned diagnostic value still remains finite for reporting.
    assert np.all(np.isfinite(result.values_by_side["left"]))


def test_keyframe_guard_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        limit_pose_keyframes({"right": [0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 1.0]})
