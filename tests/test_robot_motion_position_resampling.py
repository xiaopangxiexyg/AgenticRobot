from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.position_control import (
    CausalMotionResampler,
    TimestampedMotionTarget,
    build_uniform_time_grid,
    resample_motion_targets,
)


def _qpos(value: float) -> np.ndarray:
    return np.full(6, value, dtype=float)


def _z_pose(angle_deg: float, xyz: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    return pose


def test_offline_resampling_uses_jittered_timestamps_and_appends_exact_endpoint() -> None:
    start = 10.0
    targets = [
        TimestampedMotionTarget(
            timestamp=start,
            source_index=0,
            qpos_by_side={"right": _qpos(0.0)},
        ),
        TimestampedMotionTarget(
            timestamp=start + 0.071,
            source_index=1,
            qpos_by_side={"right": _qpos(0.71)},
        ),
        TimestampedMotionTarget(
            timestamp=start + 0.139,
            source_index=2,
            qpos_by_side={"right": _qpos(1.39)},
        ),
    ]

    references = resample_motion_targets(targets, reference_hz=45.0)

    assert len(references) == 8
    assert references[0].timestamp == pytest.approx(start)
    assert references[1].timestamp == pytest.approx(start + (1.0 / 45.0))
    assert references[-2].timestamp == pytest.approx(start + (6.0 / 45.0))
    assert references[-1].timestamp == pytest.approx(start + 0.139)
    assert references[-1].timestamp - references[-2].timestamp < 1.0 / 45.0
    assert references[1].interpolation_alpha == pytest.approx((1.0 / 45.0) / 0.071)
    np.testing.assert_allclose(references[1].qpos_by_side["right"], _qpos(10.0 / 45.0))
    np.testing.assert_allclose(references[-1].qpos_by_side["right"], _qpos(1.39))
    assert references[1].metadata["resampling_mode"] == "offline_interpolation"
    assert references[1].metadata["causal"] is False
    assert references[1].metadata["lookahead_used"] is True


def test_pose_resampling_uses_linear_xyz_and_shortest_quaternion_arc() -> None:
    start = _z_pose(170.0, (0.0, 0.0, 0.0))
    end = _z_pose(-170.0, (2.0, 4.0, 6.0))
    references = resample_motion_targets(
        [
            TimestampedMotionTarget(
                timestamp=0.0,
                source_index=0,
                pose_by_side={"right": start},
            ),
            TimestampedMotionTarget(
                timestamp=2.0,
                source_index=1,
                pose_by_side={"right": end},
            ),
        ],
        reference_hz=1.0,
    )

    midpoint = references[1].pose_by_side["right"]
    np.testing.assert_allclose(midpoint[:3, 3], [1.0, 2.0, 3.0])
    start_to_mid_deg = np.rad2deg(
        Rotation.from_matrix(midpoint[:3, :3] @ start[:3, :3].T).magnitude()
    )
    mid_to_end_deg = np.rad2deg(
        Rotation.from_matrix(end[:3, :3] @ midpoint[:3, :3].T).magnitude()
    )
    assert start_to_mid_deg == pytest.approx(10.0)
    assert mid_to_end_deg == pytest.approx(10.0)
    assert references[1].metadata["pose_interpolation"] == (
        "xyz_linear+quaternion_shortest_path_slerp"
    )


def test_causal_resampling_interpolates_only_when_both_brackets_were_available() -> None:
    resampler = CausalMotionResampler()
    resampler.push(
        TimestampedMotionTarget(
            timestamp=0.0,
            source_index=0,
            qpos_by_side={"left": _qpos(0.0), "right": _qpos(0.0)},
        )
    )
    # Reversed mapping insertion order is semantically identical and must be accepted.
    resampler.push(
        TimestampedMotionTarget(
            timestamp=1.0,
            source_index=1,
            qpos_by_side={"right": _qpos(1.0), "left": _qpos(1.0)},
        )
    )
    resampler.push(
        TimestampedMotionTarget(
            timestamp=2.0,
            source_index=2,
            qpos_by_side={"left": _qpos(2.0), "right": _qpos(2.0)},
        )
    )

    delayed = resampler.sample_at(0.5, emitted_timestamp=1.0)
    np.testing.assert_allclose(delayed.qpos_by_side["right"], _qpos(0.5))
    assert delayed.left_source_index == 0
    assert delayed.right_source_index == 1
    assert delayed.metadata["resampling_mode"] == "causal_delayed_interpolation"
    assert delayed.metadata["causal"] is True
    assert delayed.metadata["lookahead_used"] is True
    assert delayed.metadata["lookahead_available_at_emit"] is True
    assert delayed.metadata["output_lag_sec"] == pytest.approx(0.5)

    zero_lag = resampler.sample_at(1.5, emitted_timestamp=1.5)
    np.testing.assert_allclose(zero_lag.qpos_by_side["right"], _qpos(1.0))
    assert zero_lag.left_source_index == 1
    assert zero_lag.right_source_index == 1
    assert zero_lag.metadata["resampling_mode"] == "causal_hold"
    assert zero_lag.metadata["lookahead_used"] is False
    assert zero_lag.metadata["fallback_reason"] == "right_bracket_unavailable"

    with pytest.raises(ValueError, match="cannot be later"):
        resampler.sample_at(2.1, emitted_timestamp=2.0)


def test_uniform_grid_endpoint_rule_and_motion_target_immutability() -> None:
    unaligned = build_uniform_time_grid(0.0, 0.1, hz=45.0, include_final=True)
    assert unaligned[-2] == pytest.approx(4.0 / 45.0)
    assert unaligned[-1] == pytest.approx(0.1)

    without_appended_endpoint = build_uniform_time_grid(
        0.0,
        0.1,
        hz=45.0,
        include_final=False,
    )
    assert without_appended_endpoint[-1] == pytest.approx(4.0 / 45.0)

    aligned = build_uniform_time_grid(
        0.0,
        2.0 / 45.0,
        hz=45.0,
        include_final=False,
    )
    assert aligned[-1] == pytest.approx(2.0 / 45.0)

    source = _qpos(3.0)
    target = TimestampedMotionTarget(
        timestamp=0.0,
        source_index=0,
        qpos_by_side={"right": source},
    )
    source[:] = 9.0
    np.testing.assert_allclose(target.qpos_by_side["right"], _qpos(3.0))
    with pytest.raises(ValueError):
        target.qpos_by_side["right"][0] = 4.0

