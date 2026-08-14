from __future__ import annotations

import numpy as np

import robot_motion.pose_smoothing as robot_pose_smoothing
import teleop_runtime.configs.pose_smoothing as teleop_pose_smoothing
from teleop_runtime.configs.pose_smoothing import (
    BimanualPoseSmoother,
    PoseSmoother,
    PoseSmoothingConfig,
    load_default_pose_smoothing_config,
    load_impedance_pose_smoothing_config,
)


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(x)
    return pose


def test_teleop_pose_smoothing_reexports_robot_motion_classes() -> None:
    assert PoseSmoothingConfig is robot_pose_smoothing.PoseSmoothingConfig
    assert PoseSmoother is robot_pose_smoothing.PoseSmoother
    assert BimanualPoseSmoother is robot_pose_smoothing.BimanualPoseSmoother
    assert teleop_pose_smoothing.load_pose_smoothing_config is (
        robot_pose_smoothing.load_pose_smoothing_config
    )


def test_default_pose_smoothing_config_loads_from_configs_dir() -> None:
    config = load_default_pose_smoothing_config()

    assert config.enabled is True
    assert config.source_path is not None
    assert config.source_path.endswith("teleop_runtime/configs/pose_smoothing.json")
    assert config.deadband_translation_m > 0.0
    assert config == robot_pose_smoothing.RM65_POSITION_TARGET_POSE_SMOOTHING_V1


def test_impedance_pose_smoothing_is_lower_latency_and_position_isolated() -> None:
    default = load_default_pose_smoothing_config()
    impedance = load_impedance_pose_smoothing_config()

    assert impedance.enabled is True
    assert impedance.source_path is not None
    assert impedance.source_path.endswith("teleop_runtime/configs/impedance_pose_smoothing.json")
    assert impedance.time_constant_sec < default.time_constant_sec
    assert impedance.deadband_translation_m > default.deadband_translation_m
    assert impedance.deadband_rotation_deg > default.deadband_rotation_deg


def test_pose_smoother_holds_deadband_then_interpolates() -> None:
    smoother = PoseSmoother(
        PoseSmoothingConfig(
            time_constant_sec=1.0,
            nominal_dt_sec=0.1,
            max_dt_sec=0.1,
            deadband_translation_m=0.01,
            deadband_rotation_deg=1.0,
            snap_translation_m=0.0,
            snap_rotation_deg=0.0,
        )
    )

    first = smoother.smooth(_pose(), timestamp=0.0)
    held = smoother.smooth(_pose(0.005), timestamp=0.1)
    smoothed = smoother.smooth(_pose(0.1), timestamp=0.2)

    np.testing.assert_allclose(first.pose, _pose())
    assert first.metadata["status"] == "initialized"
    np.testing.assert_allclose(held.pose, _pose())
    assert held.metadata["status"] == "deadband_hold"
    assert smoothed.metadata["status"] == "smoothed"
    assert 0.0 < smoothed.pose[0, 3] < 0.1


def test_bimanual_pose_smoother_resets_inactive_side() -> None:
    smoother = BimanualPoseSmoother(
        PoseSmoothingConfig(
            time_constant_sec=1.0,
            nominal_dt_sec=0.1,
            max_dt_sec=0.1,
            deadband_translation_m=0.0,
            deadband_rotation_deg=0.0,
            snap_translation_m=0.0,
            snap_rotation_deg=0.0,
        )
    )

    smoother.smooth_pair(left_target=_pose(), right_target=_pose(), timestamp=0.0)
    left_active, metadata = smoother.smooth_pair(
        left_target=_pose(0.1),
        right_target=_pose(0.1),
        left_active=True,
        right_active=False,
        timestamp=0.1,
    )
    left_reactivated, reactivated_metadata = smoother.smooth_pair(
        left_target=_pose(0.2),
        right_target=_pose(0.2),
        left_active=True,
        right_active=True,
        timestamp=0.2,
    )

    assert metadata["right"]["status"] == "inactive_passthrough"
    np.testing.assert_allclose(left_active["right"], _pose(0.1))
    assert reactivated_metadata["right"]["status"] == "initialized"
    np.testing.assert_allclose(left_reactivated["right"], _pose(0.2))
