from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_station.composition.rebot_b601_rm65_profile import (
    load_b601_rm65_frame_mapping_profile,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.rebot_b601_rs import (
    B601IKResult,
    B601SharedControlConfig,
    B601SharedControlMode,
    B601SharedControlRuntime,
    RM65ToB601PolicyAssistBridge,
)
from robot_motion.rebot_b601_rs.rm65_retarget import (
    RM65FreeDriveSample,
    RM65ToB601RetargetConfig,
    RM65ToB601Retargeter,
)


def _pose(*, translation=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = translation
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def _sample(
    sequence: int,
    *,
    translation=(0.0, 0.0, 0.0),
    rotation=None,
    source_timestamp=10.0,
    receive_monotonic=20.0,
) -> RM65FreeDriveSample:
    return RM65FreeDriveSample(
        sequence=sequence,
        source_timestamp_sec=source_timestamp,
        receive_timestamp_sec=10.0,
        receive_timestamp_monotonic_sec=receive_monotonic,
        qpos_rad=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        pose_source_reference_source_controlled=_pose(
            translation=translation,
            rotation=rotation,
        ),
        source_id="rm65-test",
        source_clock_domain_id="local_wall_time",
        source_reference_frame_id="right_arm_base",
        source_controlled_frame_id="tool_tcp",
    )


def _graph() -> CoordinateFrameGraph:
    frames = tuple(
        CoordinateFrameSpec(frame_id=name)
        for name in ("base_link", "gripper_end", "right_arm_base", "tool_tcp")
    )
    reference = _pose(
        translation=(0.2, -0.1, 0.3),
        rotation=Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
    )
    controlled = _pose(
        rotation=Rotation.from_euler("x", 90.0, degrees=True).as_matrix(),
    )
    return CoordinateFrameGraph(
        graph_id="rm65-b601-test",
        frames=frames,
        transforms=(
            RigidFrameTransformSpec(
                transform_id="reference",
                source_frame_id="base_link",
                target_frame_id="right_arm_base",
                matrix_target_source=reference,
            ),
            RigidFrameTransformSpec(
                transform_id="controlled",
                source_frame_id="tool_tcp",
                target_frame_id="gripper_end",
                matrix_target_source=controlled,
            ),
        ),
    )


class _Model:
    def forward_kinematics(self, q_arm):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(q_arm, dtype=float)[:3]
        return pose


class _IK:
    def __init__(self, *, success: bool = True):
        self.success = success
        self.calls: list[tuple[np.ndarray, tuple[float, ...]]] = []

    def solve(self, target_pose, *, seed_qpos):
        target = np.asarray(target_pose, dtype=float).copy()
        seed = tuple(float(item) for item in seed_qpos)
        self.calls.append((target, seed))
        return B601IKResult(
            success=self.success,
            status="converged" if self.success else "max_iterations",
            iterations=1,
            qpos=seed,
            position_error_m=0.0 if self.success else 1.0,
            orientation_error_rad=0.0 if self.success else 1.0,
            seed_distance_rad=0.0,
            limit_projection_applied=False,
            sigma_min=1.0,
            condition_number=1.0,
        )


def _retargeter(ik: _IK) -> RM65ToB601Retargeter:
    config = RM65ToB601RetargetConfig(
        frame_graph=_graph(),
        source_reference_frame_id="right_arm_base",
        source_controlled_frame_id="tool_tcp",
        target_reference_frame_id="base_link",
        target_controlled_frame_id="gripper_end",
        translation_gain_target_reference_xyz=(0.5, 0.5, 0.5),
        rotation_gain=1.0,
    )
    return RM65ToB601Retargeter(
        config,
        _Model(),
        ik,
        wall_clock=lambda: 10.1,
        monotonic_clock=lambda: 20.1,
    )


def _shared_control_runtime() -> B601SharedControlRuntime:
    return B601SharedControlRuntime(
        B601SharedControlConfig(
            position_kp_nm_per_rad=(2.0,) * 6,
            velocity_kd_nm_s_per_rad=(0.0,) * 6,
            max_target_error_rad=(0.1,) * 6,
            max_abs_assist_torque_nm=(0.2,) * 6,
            max_assist_torque_rate_nm_s=(1.0,) * 6,
            authority_rise_per_sec=10.0,
            policy_stale_timeout_sec=0.25,
        )
    )


def test_reverse_mapping_reuses_graph_inverse_and_seeded_ik() -> None:
    ik = _IK()
    retargeter = _retargeter(ik)
    anchor_sample = _sample(1)
    target_qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    retargeter.capture_anchor(anchor_sample, target_qpos_rad=target_qpos)

    current = _sample(2, translation=(0.1, 0.0, 0.0))
    result = retargeter.retarget(current, seed_qpos_rad=target_qpos)

    assert result.accepted
    assert result.qpos_rad == target_qpos
    assert len(ik.calls) == 1
    # The inverse graph maps +X in the RM65 reference frame to -Y in the
    # B601 reference frame; the explicit 0.5 gain is applied after mapping.
    np.testing.assert_allclose(
        ik.calls[0][0][:3, 3],
        (0.1, 0.15, 0.3),
        atol=1e-12,
    )


def test_anchor_hold_is_exact_and_rejection_has_no_previous_qpos_fallback() -> None:
    ik = _IK()
    retargeter = _retargeter(ik)
    target_qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    retargeter.capture_anchor(_sample(1), target_qpos_rad=target_qpos)

    hold = retargeter.retarget(_sample(2), seed_qpos_rad=target_qpos)
    stale = retargeter.retarget(
        _sample(3, source_timestamp=9.0, receive_monotonic=19.0),
        seed_qpos_rad=target_qpos,
    )

    assert hold.accepted
    np.testing.assert_allclose(ik.calls[0][0], _Model().forward_kinematics(target_qpos))
    assert not stale.accepted
    assert stale.qpos_rad is None
    assert "stale" in (stale.rejection_reason or "")


def test_ik_failure_is_rejected_without_command_target() -> None:
    ik = _IK(success=False)
    retargeter = _retargeter(ik)
    target_qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    retargeter.capture_anchor(_sample(1), target_qpos_rad=target_qpos)

    result = retargeter.retarget(
        _sample(2, translation=(0.1, 0.0, 0.0)),
        seed_qpos_rad=target_qpos,
    )

    assert not result.accepted
    assert result.qpos_rad is None
    assert result.target_pose is None
    assert result.ik_result is not None
    assert result.rejection_reason == "ik_rejected:max_iterations"
    assert result.diagnostics["target_seed_translation_delta_m"] == pytest.approx(0.05)
    assert result.diagnostics["target_seed_orientation_delta_rad"] == pytest.approx(0.0)
    assert result.diagnostics["ik_result_qpos_rad"] == target_qpos
    assert result.diagnostics["target_pose_candidate"] is not None


def test_repeated_sequence_is_rejected_without_replaying_target() -> None:
    ik = _IK()
    retargeter = _retargeter(ik)
    target_qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    retargeter.capture_anchor(_sample(1), target_qpos_rad=target_qpos)
    assert retargeter.retarget(_sample(2), seed_qpos_rad=target_qpos).accepted

    repeated = retargeter.retarget(_sample(2), seed_qpos_rad=target_qpos)

    assert not repeated.accepted
    assert repeated.qpos_rad is None
    assert len(ik.calls) == 1


def test_existing_b601_rm65_profile_is_reused_in_reverse_direction() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "docs/hardware/profiles/b601_rm65_mapping_150_v2.json"
    )
    profile = load_b601_rm65_frame_mapping_profile(profile_path)
    config = RM65ToB601RetargetConfig(
        frame_graph=profile.build_frame_graph(),
        source_reference_frame_id=profile.target_reference_frame_id,
        source_controlled_frame_id=profile.target_controlled_frame_id,
        target_reference_frame_id=profile.source_reference_frame_id,
        target_controlled_frame_id=profile.source_controlled_frame_id,
        # Reverse gain is explicit; the forward 150% setting is not silently
        # reused as a reverse command scale.
        translation_gain_target_reference_xyz=(1.0, 1.0, 1.0),
    )
    ik = _IK()
    retargeter = RM65ToB601Retargeter(
        config,
        _Model(),
        ik,
        wall_clock=lambda: 10.1,
        monotonic_clock=lambda: 20.1,
    )

    retargeter.capture_anchor(_sample(1), target_qpos_rad=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    assert retargeter.anchor is not None
    assert retargeter.anchor.frame_graph_hash == profile.build_frame_graph().semantic_content_hash


def test_reverse_retarget_result_publishes_shared_control_intent() -> None:
    retargeter = _retargeter(_IK())
    runtime = _shared_control_runtime()
    bridge = RM65ToB601PolicyAssistBridge(retargeter, runtime)
    qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    bridge.capture_anchor(_sample(1), measured_b601_qpos_rad=qpos)
    result = bridge.process_sample(
        _sample(2, translation=(0.1, 0.0, 0.0)),
        measured_b601_qpos_rad=qpos,
    )

    assert result.accepted is True
    assert result.policy_command is not None
    assert result.policy_command.sequence == 2
    assert result.policy_command.source_timestamp_sec == 20.0
    assert result.policy_command.receive_timestamp_sec == 20.1
    assert result.policy_command.qpos_target_rad == qpos
    frame = runtime.step(
        control_sequence=1,
        control_timestamp_sec=20.11,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=(0.0,) * 6,
        dt_sec=0.01,
    )
    assert frame.decision.state.mode is B601SharedControlMode.POLICY_ASSIST


def test_retarget_rejection_revokes_previous_policy_immediately() -> None:
    retargeter = _retargeter(_IK())
    runtime = _shared_control_runtime()
    bridge = RM65ToB601PolicyAssistBridge(retargeter, runtime)
    qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    bridge.capture_anchor(_sample(1), measured_b601_qpos_rad=qpos)
    assert bridge.process_sample(
        _sample(2),
        measured_b601_qpos_rad=qpos,
    ).accepted
    rejected = bridge.process_sample(
        _sample(3, source_timestamp=9.0, receive_monotonic=19.0),
        measured_b601_qpos_rad=qpos,
    )

    assert rejected.accepted is False
    assert rejected.revoked_policy_sequence == 2
    assert "stale" in rejected.reason
    frame = runtime.step(
        control_sequence=1,
        control_timestamp_sec=20.11,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=(0.0,) * 6,
        dt_sec=0.01,
    )
    assert frame.decision.state.mode is B601SharedControlMode.PASSIVE
    assert frame.decision.reason == "no_policy_command"
    assert frame.reference.joint_torque_overlay_nm == (0.0,) * 6


def test_policy_clutch_clears_anchor_and_requires_explicit_recenter() -> None:
    retargeter = _retargeter(_IK())
    runtime = _shared_control_runtime()
    bridge = RM65ToB601PolicyAssistBridge(retargeter, runtime)
    qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    bridge.capture_anchor(_sample(1), measured_b601_qpos_rad=qpos)
    assert bridge.process_sample(
        _sample(2),
        measured_b601_qpos_rad=qpos,
    ).accepted
    bridge.set_policy_clutch(True)
    bridge.set_policy_clutch(False)

    rejected = bridge.process_sample(
        _sample(3),
        measured_b601_qpos_rad=qpos,
    )
    assert rejected.accepted is False
    assert rejected.reason == "retarget_rejected:anchor_not_captured"
    assert runtime.step(
        control_sequence=1,
        control_timestamp_sec=20.11,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=(0.0,) * 6,
        dt_sec=0.01,
    ).reference.joint_torque_overlay_nm == (0.0,) * 6
