from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control import (
    OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
    JointHorizonLimits,
    OfflineWaypointTrajectoryInfeasibleError,
    build_offline_quintic_waypoint_trajectory,
)
from robot_motion.replay.control_schedule import (
    REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
    ReplayControlAction,
    iter_position_qpos_control_ticks,
)


def _limits(
    dimension: int,
    *,
    velocity: float = 2.0,
    acceleration: float = 100.0,
    jerk: float = 10_000.0,
) -> JointHorizonLimits:
    return JointHorizonLimits(
        min_position_rad=np.full(dimension, -1.0),
        max_position_rad=np.full(dimension, 1.0),
        max_velocity_rad_s=np.full(dimension, velocity),
        max_acceleration_rad_s2=np.full(dimension, acceleration),
        max_jerk_rad_s3=np.full(dimension, jerk),
    )


def _steady_waypoints() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(4, dtype=float) / 15.0
    q = np.zeros((4, 6), dtype=float)
    q[:, 0] = (0.0, 0.01, 0.02, 0.03)
    return timestamps, q


def test_continuous_waypoints_do_not_stop_at_internal_15hz_knots() -> None:
    timestamps, q = _steady_waypoints()
    trajectory = build_offline_quintic_waypoint_trajectory(
        timestamps,
        q,
        command_hz=120.0,
        limits=_limits(6),
    )

    # The source samples are exact waypoints, but sustained motion crosses the
    # two internal 15 Hz boundaries with a non-zero velocity.
    np.testing.assert_allclose(trajectory.q_rad[8], q[1], atol=1e-12)
    np.testing.assert_allclose(trajectory.q_rad[16], q[2], atol=1e-12)
    assert trajectory.qd_rad_s[8, 0] == pytest.approx(0.15)
    assert trajectory.qd_rad_s[16, 0] == pytest.approx(0.15)
    assert trajectory.knot_qdd_rad_s2[1, 0] == pytest.approx(0.0)
    assert trajectory.knot_qdd_rad_s2[2, 0] == pytest.approx(0.0)
    assert trajectory.metadata["continuous_hard_limit_certificate"]["passed"]


def test_explicit_stops_preserve_joint_space_polyline_without_corner_cutting() -> None:
    q = np.asarray(((0.0, 0.0), (0.1, 0.1), (0.2, 0.05)))
    trajectory = build_offline_quintic_waypoint_trajectory(
        (0.0, 1.0, 2.0),
        q,
        command_hz=120.0,
        limits=_limits(2),
        stop_at_waypoints=True,
    )
    np.testing.assert_allclose(trajectory.knot_qd_rad_s, 0, atol=1e-12)
    np.testing.assert_allclose(trajectory.knot_qdd_rad_s2, 0, atol=1e-12)
    # Each half lies on its original line, not a blended PCHIP corner.
    np.testing.assert_allclose(
        trajectory.q_rad[:121, 0], trajectory.q_rad[:121, 1], atol=1e-12
    )
    np.testing.assert_allclose(
        trajectory.q_rad[120:, 1], 0.15 - 0.5 * trajectory.q_rad[120:, 0], atol=1e-12
    )
    assert trajectory.metadata["continuous_hard_limit_certificate"]["passed"]
    assert trajectory.metadata["internal_knot_zero_velocity_required"]


def test_explicit_stops_reject_nonzero_endpoint_derivatives() -> None:
    with pytest.raises(ValueError, match="zero endpoint derivatives"):
        build_offline_quintic_waypoint_trajectory(
            (0, 1),
            ((0.0,), (0.1,)),
            command_hz=120,
            limits=_limits(1),
            stop_at_waypoints=True,
            initial_velocity_rad_s=(0.1,),
        )


def test_continuous_replay_mode_emits_one_certified_command_clock() -> None:
    timestamps, q = _steady_waypoints()
    actions = [
        ReplayControlAction(
            action_index=index,
            frame_index=100 + index,
            relative_time_sec=float(timestamp),
            qpos_by_side={"right": q[index]},
        )
        for index, timestamp in enumerate(timestamps)
    ]

    ticks = iter_position_qpos_control_ticks(
        actions,
        reference_hz=60.0,
        control_hz=120.0,
        include_final_settle=False,
        execution_mode=REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
        continuous_limits=_limits(6),
    )

    assert ticks[8].reference_index == 1
    assert ticks[8].action_index == 1
    np.testing.assert_allclose(ticks[8].qpos_by_side["right"], q[1], atol=1e-12)
    assert ticks[8].metadata["continuous_qd_rad_s"][0] == pytest.approx(0.15)
    assert ticks[8].metadata["continuous_internal_knot_zero_velocity_required"] is False
    assert ticks[8].metadata["command_resampling_mode"] == (
        REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC
    )


def test_continuous_waypoint_builder_fails_closed_on_between_tick_limit() -> None:
    timestamps, q = _steady_waypoints()

    with pytest.raises(
        OfflineWaypointTrajectoryInfeasibleError,
        match="velocity:joint_1",
    ):
        build_offline_quintic_waypoint_trajectory(
            timestamps,
            q,
            command_hz=120.0,
            limits=_limits(6, velocity=0.05),
        )


def test_safe_retime_preserves_waypoints_and_certifies_dynamic_limits() -> None:
    timestamps, q = _steady_waypoints()
    limits = _limits(6, velocity=0.10, acceleration=5.0, jerk=100.0)

    trajectory = build_offline_quintic_waypoint_trajectory(
        timestamps,
        q,
        command_hz=120.0,
        limits=limits,
        timing_mode=OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
        max_time_scale=4.0,
    )

    assert trajectory.metadata["retimed"] is True
    assert 1.0 < trajectory.metadata["time_scale"] <= 4.0
    np.testing.assert_allclose(
        trajectory.waypoint_timestamps,
        timestamps * trajectory.metadata["time_scale"],
    )
    np.testing.assert_allclose(trajectory.waypoint_q_rad, q)
    assert trajectory.metadata["continuous_hard_limit_certificate"]["passed"]


def test_continuous_replay_safe_retime_uses_execution_timestamps() -> None:
    timestamps, q = _steady_waypoints()
    actions = [
        ReplayControlAction(
            action_index=index,
            frame_index=index,
            relative_time_sec=float(timestamp),
            qpos_by_side={"right": q[index]},
        )
        for index, timestamp in enumerate(timestamps)
    ]

    ticks = iter_position_qpos_control_ticks(
        actions,
        reference_hz=60.0,
        control_hz=120.0,
        include_final_settle=False,
        execution_mode=REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
        continuous_limits=_limits(6, velocity=0.10, acceleration=5.0, jerk=100.0),
        continuous_timing_mode=OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
        continuous_max_time_scale=4.0,
    )

    assert ticks[-1].metadata["continuous_retimed"] is True
    assert ticks[-1].metadata["continuous_time_scale"] > 1.0
    assert ticks[-1].relative_time_sec > timestamps[-1]
    np.testing.assert_allclose(ticks[-1].qpos_by_side["right"], q[-1])


def test_shape_preserving_waypoints_do_not_overshoot_monotone_positions() -> None:
    timestamps, q = _steady_waypoints()
    trajectory = build_offline_quintic_waypoint_trajectory(
        timestamps,
        q,
        command_hz=240.0,
        limits=_limits(6),
    )

    delta = np.diff(trajectory.q_rad[:, 0])
    assert np.min(delta) >= -1e-12
    assert np.min(trajectory.q_rad[:, 0]) >= -1e-12
    assert np.max(trajectory.q_rad[:, 0]) <= 0.03 + 1e-12
