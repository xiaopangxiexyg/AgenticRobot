from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_URDF_RELATIVE_PATH,
    B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    B601AssistCommissioningState,
    B601CartesianAssistConfig,
    B601CartesianAssistSession,
    B601CartesianBreakawayConfig,
    B601HapticWrenchChannel,
    B601HapticWrenchSample,
    B601PinocchioModel,
    B601PoseVirtualSpring,
    B601SpatialWrench,
    B601TranslationalVirtualSpring,
    B601VirtualWall,
    map_world_aligned_ee_wrench_to_joint_torque,
    transform_spatial_twist,
    transform_spatial_wrench,
)
from scripts.hardware import rebot_b601_rs_low_gain_probe as low_gain_probe

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _feedback_channel(**overrides):
    config = {
        "source_frame_id": "follower_tip", "target_T_source": np.eye(4), "gain": 1.0,
        "max_abs_force_n": (2.0,) * 3, "max_abs_torque_nm": (0.2,) * 3,
        "max_force_rate_n_s": (1.0,) * 3, "max_torque_rate_nm_s": (0.1,) * 3,
        "filter_tau_sec": 0.0, "maximum_source_age_sec": 0.1,
    }
    config.update(overrides)
    return B601HapticWrenchChannel(**config)


def _feedback_sample(sequence=0, time=0.0, **overrides):
    fields = {
        "sequence": sequence, "source_timestamp_sec": time, "receive_timestamp_sec": time,
        "wrench": B601SpatialWrench((1.0, 0.0, 0.0), (0.0,) * 3, "follower_tip"),
    }
    fields.update(overrides)
    return B601HapticWrenchSample(**fields)


def test_haptic_channel_ramps_from_zero_and_expires_original_source_age() -> None:
    channel = _feedback_channel()
    first = channel.update(_feedback_sample(), now_sec=0.0)
    np.testing.assert_array_equal(first.wrench.as_vector(), np.zeros(6))
    held = channel.update(None, now_sec=0.05)
    assert held.source_sequence == 0
    assert held.source_age_sec == 0.05
    assert held.source_timestamp_sec == held.receive_timestamp_sec == 0.0
    assert held.wrench.force_n == pytest.approx((0.05, 0.0, 0.0))
    expired = channel.update(None, now_sec=0.100001)
    assert expired.rejection_reason == "expired_source"
    np.testing.assert_array_equal(expired.wrench.as_vector(), np.zeros(6))


def test_cartesian_assist_reuses_jacobian_mapping_and_independent_overlay_slew(
    b601_model: B601PinocchioModel,
) -> None:
    qpos = (0.0,) * 6
    anchor_pose = np.asarray(b601_model.forward_kinematics(qpos), dtype=float)
    anchor = anchor_pose[:3, 3]
    session = B601CartesianAssistSession(
        B601CartesianAssistConfig(
            axis="x",
            offset_m=0.01,
            stiffness_n_per_m=50.0,
            damping_n_s_per_m=0.0,
            max_abs_force_n=0.5,
            max_abs_joint_torque_nm=0.15,
            max_joint_torque_rate_nm_s=0.10,
            authority_rise_per_sec=1.0,
        )
    )
    session.anchor_position_m = tuple(float(value) for value in anchor)
    session.anchor_rotation = tuple(
        tuple(float(value) for value in row)
        for row in anchor_pose[:3, :3]
    )
    session.target_position_m = (
        float(anchor[0] + 0.01),
        float(anchor[1]),
        float(anchor[2]),
    )
    session.active_at_sec = 1.0
    session.state = B601AssistCommissioningState.ACTIVE

    overlay, audit = low_gain_probe.compute_cartesian_assist_overlay(
        b601_model,
        session,
        qpos_rad=qpos,
        qvel_rad_s=(0.0,) * 6,
        timestamp_sec=2.0,
        previous_overlay_nm=(0.0,) * 6,
        dt_sec=0.05,
        maximum_lateral_drift_m=0.005,
    )

    assert audit["commanded_force_n"] == pytest.approx((0.5, 0.0, 0.0))
    assert any(abs(value) > 0.0 for value in audit["raw_joint_torque_overlay_nm"])
    assert max(abs(value) for value in overlay) <= 0.005 + 1e-12
    assert audit["joint_torque_slew_limited"] is True
    assert audit["joint_torque_slew_mode"] == ("rise",) * 6
    assert audit["joint_torque_selected_rate_nm_s"] == pytest.approx(
        (0.10,) * 6
    )


def test_cartesian_breakaway_composes_before_existing_overlay_limits(
    b601_model: B601PinocchioModel,
) -> None:
    qpos = (0.0,) * 6
    anchor_pose = np.asarray(b601_model.forward_kinematics(qpos), dtype=float)
    anchor = anchor_pose[:3, 3]
    session = B601CartesianAssistSession(
        B601CartesianAssistConfig(
            axis="x",
            offset_m=0.01,
            stiffness_n_per_m=50.0,
            damping_n_s_per_m=0.0,
            max_abs_force_n=0.5,
            max_abs_joint_torque_nm=0.5,
            max_joint_torque_rate_nm_s=100.0,
            authority_rise_per_sec=10.0,
            breakaway=B601CartesianBreakawayConfig(
                positive_torque_nm=(0.0, 0.10, 0.10, 0.0, 0.0, 0.0),
                negative_torque_nm=(0.0, 0.10, 0.10, 0.0, 0.0, 0.0),
                intent_deadband_nm=0.0,
                intent_ramp_nm=1e-6,
            ),
        )
    )
    session.anchor_position_m = tuple(float(value) for value in anchor)
    session.anchor_rotation = tuple(
        tuple(float(value) for value in row)
        for row in anchor_pose[:3, :3]
    )
    session.target_position_m = (
        float(anchor[0] + 0.01),
        float(anchor[1]),
        float(anchor[2]),
    )
    session.active_at_sec = 1.0
    session.state = B601AssistCommissioningState.ACTIVE

    overlay, audit = low_gain_probe.compute_cartesian_assist_overlay(
        b601_model,
        session,
        qpos_rad=qpos,
        qvel_rad_s=(0.0,) * 6,
        timestamp_sec=2.0,
        previous_overlay_nm=(0.0,) * 6,
        dt_sec=0.05,
        maximum_lateral_drift_m=0.005,
    )

    task = audit["raw_task_joint_torque_overlay_nm"]
    compensation = audit["breakaway"]["torque_nm"]
    composed = audit["raw_joint_torque_overlay_nm"]
    assert audit["breakaway"]["enabled"] is True
    assert any(audit["breakaway"]["active"])
    assert composed == pytest.approx(
        tuple(
            task_value + compensation_value
            for task_value, compensation_value in zip(
                task, compensation, strict=True
            )
        )
    )
    assert overlay == pytest.approx(composed)


@pytest.mark.parametrize(
    "bad,reason",
    [
        (_feedback_sample(1, 0.05, valid=False), "invalid_sample"),
        (
            _feedback_sample(
                1, 0.05, wrench=B601SpatialWrench((1.0,) * 3, (0.0,) * 3, "wrong")
            ),
            "unexpected_frame",
        ),
        (_feedback_sample(1, 0.2), "future_timestamp"),
        (_feedback_sample(0, 0.05), "stale_sequence"),
        (_feedback_sample(1, 0.0), "stale_source_timestamp"),
    ],
)
def test_haptic_channel_bad_input_immediately_cancels_and_requires_new_sample(
    bad, reason
) -> None:
    channel = _feedback_channel(filter_tau_sec=0.02)
    channel.update(_feedback_sample(), now_sec=0.0)
    assert channel.update(None, now_sec=0.04).wrench.force_n[0] > 0.0
    rejected = channel.update(bad, now_sec=0.06)
    assert rejected.rejection_reason == reason
    assert rejected.source_sequence is None
    np.testing.assert_array_equal(rejected.wrench.as_vector(), np.zeros(6))
    np.testing.assert_array_equal(
        channel.update(None, now_sec=0.07).wrench.as_vector(), np.zeros(6)
    )
    restarted = channel.update(_feedback_sample(1, 0.08), now_sec=0.08)
    assert 0.0 < restarted.wrench.force_n[0] <= 0.01 + 1e-12


def test_haptic_channel_delayed_sample_does_not_refresh_acquisition_time() -> None:
    channel = _feedback_channel()
    sample = _feedback_sample(receive_timestamp_sec=0.09)
    output = channel.update(sample, now_sec=0.09)
    assert output.source_age_sec == 0.09
    assert channel.update(None, now_sec=0.11).rejection_reason == "expired_source"
    assert (
        _feedback_channel().update(sample, now_sec=0.11).rejection_reason
        == "expired_source"
    )


def test_haptic_channel_rejects_receive_regression_and_mapped_overflow() -> None:
    channel = _feedback_channel()
    channel.update(_feedback_sample(receive_timestamp_sec=0.08), now_sec=0.08)
    rejected = channel.update(_feedback_sample(1, 0.02), now_sec=0.09)
    assert rejected.rejection_reason == "stale_receive_timestamp"
    np.testing.assert_array_equal(rejected.wrench.as_vector(), np.zeros(6))
    overflow = _feedback_channel(gain=1e308)
    overflow.update(_feedback_sample(wrench=B601SpatialWrench((1e-308, 0.0, 0.0), (0.0,) * 3, "follower_tip")), now_sec=0.0)
    assert overflow.update(None, now_sec=0.04).wrench.force_n[0] > 0.0
    rejected = overflow.update(_feedback_sample(1, 0.05, wrench=B601SpatialWrench((2.0, 0.0, 0.0), (0.0,) * 3, "follower_tip")), now_sec=0.05)
    assert rejected.rejection_reason == "non_finite_mapped_wrench"
    np.testing.assert_array_equal(rejected.wrench.as_vector(), np.zeros(6))


def test_haptic_channel_mapping_uses_rotation_and_origin_before_clipping() -> None:
    transform = np.eye(4)
    transform[:3, :3] = _rotation_z(np.pi / 2.0)
    transform[:3, 3] = (1.0, 0.0, 0.0)
    channel = _feedback_channel(
        target_T_source=transform,
        gain=10.0,
        max_force_rate_n_s=(1000.0,) * 3,
        max_torque_rate_nm_s=(1000.0,) * 3,
    )
    transform[:] = 0.0  # The caller cannot mutate the admitted transform.
    channel.update(_feedback_sample(), now_sec=0.0)
    output = channel.update(None, now_sec=0.05)
    assert output.amplitude_limited is True
    assert output.wrench.force_n == pytest.approx((0.0, 2.0, 0.0), abs=1e-12)
    assert output.wrench.torque_nm == pytest.approx((0.0, 0.0, 0.2), abs=1e-12)
    assert output.wrench.frame_id == B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME


def test_haptic_channel_low_pass_and_slew_use_consumer_dt_not_source_interval() -> None:
    channel = _feedback_channel(filter_tau_sec=0.02, max_force_rate_n_s=(100.0,) * 3)
    channel.update(_feedback_sample(), now_sec=0.0)
    assert channel.update(None, now_sec=0.01).wrench.force_n[0] == pytest.approx(
        1.0 - np.exp(-0.5)
    )
    assert channel.update(None, now_sec=0.02).wrench.force_n[0] == pytest.approx(
        1.0 - np.exp(-1.0)
    )
    with pytest.raises(ValueError, match="strictly advance"):
        channel.update(None, now_sec=0.02)


@pytest.mark.parametrize(
    "overrides",
    [
        {"gain": -1.0},
        {"gain": float("nan")},
        {"filter_tau_sec": -0.1},
        {"maximum_source_age_sec": 0.0},
        {"max_abs_force_n": (float("inf"),) * 3},
        {"max_force_rate_n_s": (-1.0,) * 3},
        {"target_T_source": np.zeros((4, 4))},
    ],
)
def test_haptic_channel_rejects_invalid_configuration(overrides) -> None:
    with pytest.raises(ValueError):
        _feedback_channel(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequence": -1},
        {"sequence": True},
        {"source_timestamp_sec": float("nan")},
        {"source_timestamp_sec": 0.2, "receive_timestamp_sec": 0.1},
        {"valid": "true"},
    ],
)
def test_haptic_sample_rejects_invalid_metadata(overrides) -> None:
    with pytest.raises((ValueError, TypeError)):
        _feedback_sample(**overrides)


@pytest.fixture(scope="module")
def b601_model() -> B601PinocchioModel:
    pytest.importorskip("pinocchio")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    return B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def test_feedback_motion_guard_first_sample_has_no_synthetic_acceleration() -> None:
    qacc, rejection = low_gain_probe.evaluate_feedback_motion_guard(
        sequence=7,
        feedback_timestamp_sec=12.05,
        dt_sec=0.05,
        raw_qpos_rad=(0.0,) * 6,
        normalized_qpos_rad=(0.0,) * 6,
        previous_qvel_rad_s=None,
        current_qvel_rad_s=(0.01,) * 6,
        velocity_abort_rad_s=0.12,
        acceleration_abort_rad_s2=4.0,
    )

    assert qacc is None
    assert rejection is None


def test_feedback_motion_guard_rejects_acceleration_with_complete_evidence() -> None:
    raw_q = (0.0, -0.0002, 0.1, 0.2, 0.3, 0.4)
    normalized_q = (0.0, 0.0, 0.1, 0.2, 0.3, 0.4)
    previous_qvel = (0.0,) * 6
    current_qvel = (0.0, 0.11, 0.0, 0.0, 0.0, 0.0)

    qacc, rejection = low_gain_probe.evaluate_feedback_motion_guard(
        sequence=19,
        feedback_timestamp_sec=42.0,
        dt_sec=0.05,
        raw_qpos_rad=raw_q,
        normalized_qpos_rad=normalized_q,
        previous_qvel_rad_s=previous_qvel,
        current_qvel_rad_s=current_qvel,
        velocity_abort_rad_s=0.12,
        acceleration_abort_rad_s2=2.0,
    )

    assert qacc == pytest.approx((0.0, 2.2, 0.0, 0.0, 0.0, 0.0))
    assert rejection is not None
    assert rejection["guard"] == "acceleration"
    assert rejection["offending_joint"] == "joint2"
    assert rejection["sequence"] == 19
    assert rejection["raw_qpos_rad"] == raw_q
    assert rejection["normalized_qpos_rad"] == normalized_q
    assert rejection["previous_qvel_rad_s"] == previous_qvel
    assert rejection["current_qvel_rad_s"] == current_qvel
    assert rejection["qacc_fd_rad_s2"] == pytest.approx(qacc)
    assert rejection["command_dispatched"] is False
    assert rejection["eligible_for_last_safe"] is False


def test_feedback_motion_guard_velocity_rejection_keeps_computed_qacc() -> None:
    qacc, rejection = low_gain_probe.evaluate_feedback_motion_guard(
        sequence=20,
        feedback_timestamp_sec=42.05,
        dt_sec=0.05,
        raw_qpos_rad=(0.0,) * 6,
        normalized_qpos_rad=(0.0,) * 6,
        previous_qvel_rad_s=(0.0,) * 6,
        current_qvel_rad_s=(0.0, 0.0, 0.0, 0.13, 0.0, 0.0),
        velocity_abort_rad_s=0.12,
        acceleration_abort_rad_s2=4.0,
    )

    assert qacc == pytest.approx((0.0, 0.0, 0.0, 2.6, 0.0, 0.0))
    assert rejection is not None
    assert rejection["guard"] == "velocity"
    assert rejection["offending_joint"] == "joint4"
    assert rejection["qacc_fd_rad_s2"] == pytest.approx(qacc)


@pytest.mark.parametrize("dt_sec", [0.0, -0.01, np.nan])
def test_feedback_motion_guard_rejects_invalid_feedback_dt(dt_sec: float) -> None:
    with pytest.raises(ValueError, match="feedback_dt_sec"):
        low_gain_probe.evaluate_feedback_motion_guard(
            sequence=1,
            feedback_timestamp_sec=1.0,
            dt_sec=dt_sec,
            raw_qpos_rad=(0.0,) * 6,
            normalized_qpos_rad=(0.0,) * 6,
            previous_qvel_rad_s=(0.0,) * 6,
            current_qvel_rad_s=(0.0,) * 6,
            velocity_abort_rad_s=0.12,
            acceleration_abort_rad_s2=4.0,
        )


def test_wrench_transform_rotates_force_and_shifts_torque_origin() -> None:
    target_T_source = np.eye(4)
    target_T_source[:3, :3] = _rotation_z(np.pi / 2.0)
    target_T_source[:3, 3] = (1.0, 0.0, 0.0)
    source = B601SpatialWrench(
        force_n=(1.0, 0.0, 0.0),
        torque_nm=(0.0, 0.0, 2.0),
        frame_id="source",
    )

    target = transform_spatial_wrench(
        source,
        target_frame_id="target",
        target_T_source=target_T_source,
    )

    np.testing.assert_allclose(target.force_n, (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(target.torque_nm, (0.0, 0.0, 3.0), atol=1e-12)
    assert target.frame_id == "target"


def test_wrench_and_twist_transforms_preserve_power_and_round_trip() -> None:
    target_T_source = np.eye(4)
    target_T_source[:3, :3] = _rotation_z(0.63)
    target_T_source[:3, 3] = (0.2, -0.1, 0.35)
    source_wrench = B601SpatialWrench(
        force_n=(2.0, -3.0, 1.5),
        torque_nm=(0.4, 0.2, -0.8),
        frame_id="source",
    )
    source_twist = np.asarray((0.4, -0.2, 0.1, 0.3, -0.5, 0.2))

    target_wrench = transform_spatial_wrench(
        source_wrench,
        target_frame_id="target",
        target_T_source=target_T_source,
    )
    target_twist = transform_spatial_twist(
        source_twist,
        target_T_source=target_T_source,
    )
    source_power = float(source_wrench.as_vector() @ source_twist)
    target_power = float(target_wrench.as_vector() @ target_twist)

    recovered = transform_spatial_wrench(
        target_wrench,
        target_frame_id="source",
        target_T_source=np.linalg.inv(target_T_source),
    )
    assert target_power == pytest.approx(source_power, abs=1e-12)
    np.testing.assert_allclose(recovered.as_vector(), source_wrench.as_vector())


@pytest.mark.parametrize(
    "transform, message",
    [
        (np.eye(3), "4-by-4"),
        (np.diag([1.0, 1.0, 1.0, 2.0]), "homogeneous"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "right handed"),
        (
            np.asarray(
                [
                    [1.0, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "orthonormal",
        ),
    ],
)
def test_spatial_transform_rejects_non_se3_inputs(
    transform: np.ndarray,
    message: str,
) -> None:
    wrench = B601SpatialWrench((0.0,) * 3, (0.0,) * 3, "source")
    with pytest.raises(ValueError, match=message):
        transform_spatial_wrench(
            wrench,
            target_frame_id="target",
            target_T_source=transform,
        )


def test_jacobian_transpose_mapping_preserves_instantaneous_power(
    b601_model: B601PinocchioModel,
) -> None:
    qpos = (0.2, 1.0, 1.1, -0.2, 0.15, 0.3)
    qvel = (0.1, -0.2, 0.05, 0.3, -0.1, 0.2)
    wrench = B601SpatialWrench(
        force_n=(1.2, -2.5, 0.4),
        torque_nm=(0.2, 0.1, -0.3),
        frame_id=B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    )

    result = map_world_aligned_ee_wrench_to_joint_torque(
        b601_model,
        qpos,
        wrench,
        qvel_rad_s=qvel,
    )
    expected = b601_model.arm_jacobian(qpos).T @ wrench.as_vector()

    np.testing.assert_allclose(result.joint_torque_nm, expected, atol=1e-12)
    assert result.cartesian_power_w == pytest.approx(result.joint_power_w, abs=1e-12)


def test_jacobian_transpose_mapping_rejects_wrong_frame(
    b601_model: B601PinocchioModel,
) -> None:
    with pytest.raises(ValueError, match="wrench frame"):
        map_world_aligned_ee_wrench_to_joint_torque(
            b601_model,
            (0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            B601SpatialWrench((1.0, 0.0, 0.0), (0.0,) * 3, "tool_local"),
        )


def test_virtual_spring_has_restoring_sign_damping_and_component_limits() -> None:
    spring = B601TranslationalVirtualSpring(
        anchor_position_m=(0.3, 0.0, 0.4),
        stiffness_n_per_m=(100.0, 50.0, 20.0),
        damping_n_s_per_m=(2.0, 4.0, 6.0),
        max_abs_force_n=(3.0, 2.0, 1.0),
    )

    result = spring.evaluate(
        position_m=(0.35, -0.02, 0.41),
        linear_velocity_m_s=(0.2, -0.1, 0.5),
    )

    np.testing.assert_allclose(result.displacement_m, (0.05, -0.02, 0.01))
    np.testing.assert_allclose(result.raw_force_n, (-5.4, 1.4, -3.2))
    np.testing.assert_allclose(result.wrench.force_n, (-3.0, 1.4, -1.0))
    np.testing.assert_allclose(result.wrench.torque_nm, 0.0)
    assert result.force_limited is True


def test_pose_virtual_spring_is_zero_at_anchor() -> None:
    anchor = np.eye(4)
    anchor[:3, 3] = (0.3, -0.1, 0.4)
    spring = B601PoseVirtualSpring(
        anchor_pose=anchor,
        translational_stiffness_n_per_m=(20.0,) * 3,
        translational_damping_n_s_per_m=(1.0,) * 3,
        rotational_stiffness_nm_per_rad=(0.5,) * 3,
        rotational_damping_nm_s_per_rad=(0.05,) * 3,
        max_abs_force_n=(2.0,) * 3,
        max_abs_torque_nm=(0.25,) * 3,
    )

    result = spring.evaluate(anchor, (0.0,) * 6)

    np.testing.assert_allclose(result.displacement_m, 0.0)
    np.testing.assert_allclose(result.rotation_error_rad, 0.0)
    np.testing.assert_allclose(result.wrench.as_vector(), 0.0)
    assert result.force_limited is False
    assert result.torque_limited is False


def test_pose_virtual_spring_restores_rotation_and_bounds_force_and_torque() -> None:
    anchor = np.eye(4)
    current = np.eye(4)
    current[:3, :3] = _rotation_z(0.2)
    current[:3, 3] = (0.2, -0.1, 0.0)
    spring = B601PoseVirtualSpring(
        anchor_pose=anchor,
        translational_stiffness_n_per_m=(20.0, 10.0, 5.0),
        translational_damping_n_s_per_m=(1.0, 2.0, 3.0),
        rotational_stiffness_nm_per_rad=(1.0, 2.0, 3.0),
        rotational_damping_nm_s_per_rad=(0.1, 0.2, 0.3),
        max_abs_force_n=(2.0, 2.0, 2.0),
        max_abs_torque_nm=(0.1, 0.1, 0.1),
    )

    result = spring.evaluate(
        current,
        (0.5, -0.5, 0.0, 0.0, 0.0, 0.5),
    )

    np.testing.assert_allclose(result.displacement_m, (0.2, -0.1, 0.0))
    np.testing.assert_allclose(result.rotation_error_rad, (0.0, 0.0, 0.2))
    np.testing.assert_allclose(result.raw_force_n, (-4.5, 2.0, 0.0))
    np.testing.assert_allclose(result.wrench.force_n, (-2.0, 2.0, 0.0))
    np.testing.assert_allclose(result.raw_torque_nm, (0.0, 0.0, -0.75))
    np.testing.assert_allclose(result.wrench.torque_nm, (0.0, 0.0, -0.1))
    assert result.force_limited is True
    assert result.torque_limited is True


def test_pose_virtual_spring_rejects_invalid_anchor_and_sample_pose() -> None:
    invalid = np.eye(4)
    invalid[3, 3] = 2.0
    with pytest.raises(ValueError, match="homogeneous"):
        B601PoseVirtualSpring(
            anchor_pose=invalid,
            translational_stiffness_n_per_m=(1.0,) * 3,
            translational_damping_n_s_per_m=(1.0,) * 3,
            rotational_stiffness_nm_per_rad=(1.0,) * 3,
            rotational_damping_nm_s_per_rad=(1.0,) * 3,
            max_abs_force_n=(1.0,) * 3,
            max_abs_torque_nm=(1.0,) * 3,
        )
    spring = B601PoseVirtualSpring(
        anchor_pose=np.eye(4),
        translational_stiffness_n_per_m=(1.0,) * 3,
        translational_damping_n_s_per_m=(1.0,) * 3,
        rotational_stiffness_nm_per_rad=(1.0,) * 3,
        rotational_damping_nm_s_per_rad=(1.0,) * 3,
        max_abs_force_n=(1.0,) * 3,
        max_abs_torque_nm=(1.0,) * 3,
    )
    with pytest.raises(ValueError, match="orthonormal"):
        sample = np.eye(4)
        sample[0, 0] = 2.0
        spring.evaluate(sample, (0.0,) * 6)


def test_virtual_wall_is_unilateral_bounded_and_never_attractive() -> None:
    wall = B601VirtualWall(
        normal=(2.0, 0.0, 0.0),
        offset_m=0.35,
        stiffness_n_per_m=200.0,
        damping_n_s_per_m=10.0,
        max_force_n=4.0,
    )

    outside = wall.evaluate((0.34, 0.0, 0.0), (1.0, 0.0, 0.0))
    entering = wall.evaluate((0.36, 0.0, 0.0), (0.2, 0.0, 0.0))
    exiting = wall.evaluate((0.36, 0.0, 0.0), (-1.0, 0.0, 0.0))
    limited = wall.evaluate((0.40, 0.0, 0.0), (0.0, 0.0, 0.0))

    np.testing.assert_allclose(outside.wrench.force_n, 0.0)
    np.testing.assert_allclose(entering.wrench.force_n, (-4.0, 0.0, 0.0))
    np.testing.assert_allclose(exiting.wrench.force_n, 0.0)
    np.testing.assert_allclose(limited.wrench.force_n, (-4.0, 0.0, 0.0))
    assert entering.penetration_m == pytest.approx(0.01)
    assert entering.force_limited is True
    assert exiting.raw_restoring_force_n == pytest.approx(0.0)
    assert limited.force_limited is True


def test_virtual_fixture_configuration_and_samples_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        B601TranslationalVirtualSpring(
            anchor_position_m=(0.0,) * 3,
            stiffness_n_per_m=(1.0,) * 3,
            damping_n_s_per_m=(1.0,) * 3,
            max_abs_force_n=(1.0, 0.0, 1.0),
        )
    with pytest.raises(ValueError, match="non-zero"):
        B601VirtualWall(
            normal=(0.0,) * 3,
            offset_m=0.0,
            stiffness_n_per_m=1.0,
            damping_n_s_per_m=0.0,
            max_force_n=1.0,
        )
    spring = B601TranslationalVirtualSpring(
        anchor_position_m=(0.0,) * 3,
        stiffness_n_per_m=(1.0,) * 3,
        damping_n_s_per_m=(1.0,) * 3,
        max_abs_force_n=(1.0,) * 3,
    )
    with pytest.raises(ValueError, match="finite"):
        spring.evaluate((np.nan, 0.0, 0.0), (0.0,) * 3)
