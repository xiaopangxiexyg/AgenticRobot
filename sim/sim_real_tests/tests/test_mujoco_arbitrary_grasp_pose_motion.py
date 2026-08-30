from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.rm65.grasp_reachability import (
    GraspReachabilityReceipt,
    GraspReachabilityStatus,
)
from sim.sim_real_tests.mujoco import arbitrary_grasp_pose_motion as motion
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    GraspMotionTarget,
    ToolTcpFrameBinding,
    build_smooth_se3_trajectory,
    execute_grasp_motion,
    pose_from_xyz_quat_xyzw,
    pose_from_xyz_rpy_deg,
    quintic_progress,
    required_motion_duration_sec,
)
from sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion import (
    _validate_args,
    build_parser,
)
from sim.sim_real_tests.rm65 import (
    DEFAULT_REFERENCE_FRAME_ID,
    DEFAULT_REFERENCE_FRAME_SPEC_HASH,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_pose_builders_share_explicit_base_from_dmtac_tcp_semantics() -> None:
    xyz = (0.42, -0.18, 0.51)
    from_rpy = pose_from_xyz_rpy_deg(xyz, (20.0, -35.0, 80.0))
    quaternion = Rotation.from_euler(
        "xyz", (20.0, -35.0, 80.0), degrees=True
    ).as_quat()
    from_quaternion = pose_from_xyz_quat_xyzw(xyz, quaternion)

    np.testing.assert_allclose(from_rpy, from_quaternion, atol=1e-12)
    np.testing.assert_allclose(from_rpy[:3, 3], xyz, atol=0.0)
    with pytest.raises(ValueError, match="nonzero norm"):
        pose_from_xyz_quat_xyzw(xyz, (0.0, 0.0, 0.0, 0.0))


def test_quintic_se3_reference_has_exact_endpoints_and_respects_speed_limits() -> None:
    start = np.eye(4, dtype=float)
    target = pose_from_xyz_rpy_deg((0.36, -0.21, 0.17), (65.0, -30.0, 95.0))
    linear_limit = 0.11
    angular_limit_deg = 38.0
    duration, metadata = required_motion_duration_sec(
        start,
        target,
        linear_speed_m_s=linear_limit,
        angular_speed_deg_s=angular_limit_deg,
        minimum_duration_sec=0.5,
    )
    trajectory = build_smooth_se3_trajectory(
        start,
        target,
        reference_hz=60.0,
        duration_sec=duration,
        terminal_hold_sec=0.0,
    )

    np.testing.assert_allclose(trajectory[0], start, atol=1e-12)
    np.testing.assert_allclose(trajectory[-1], target, atol=1e-12)
    assert quintic_progress(0.0) == 0.0
    assert quintic_progress(1.0) == 1.0
    dt = duration / (len(trajectory) - 1)
    translation_speed = [
        np.linalg.norm(rhs[:3, 3] - lhs[:3, 3]) / dt
        for lhs, rhs in pairwise(trajectory)
    ]
    angular_speed_deg = [
        np.rad2deg(
            Rotation.from_matrix(lhs[:3, :3].T @ rhs[:3, :3]).magnitude()
        )
        / dt
        for lhs, rhs in pairwise(trajectory)
    ]
    assert max(translation_speed) <= linear_limit * 1.0001
    assert max(angular_speed_deg) <= angular_limit_deg * 1.0001
    assert metadata["translation_distance_m"] > 0.4
    assert metadata["rotation_distance_deg"] > 90.0
    # Discrete endpoint speeds converge to the analytical zero derivative.
    assert translation_speed[0] < 2e-4
    assert translation_speed[-1] < 2e-4
    assert angular_speed_deg[0] < 0.02
    assert angular_speed_deg[-1] < 0.02


def test_tool_tcp_binding_converts_actual_dmtac_target_to_solver_frame() -> None:
    single_from_dmtac = np.eye(4, dtype=float)
    single_from_dmtac[:3, 3] = (0.0, 0.0, 0.131)
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=single_from_dmtac,
        position_offset_m=0.131,
        orientation_offset_deg=0.0,
    )
    base_from_single = pose_from_xyz_rpy_deg(
        (0.32, -0.08, 0.56), (15.0, 30.0, -25.0)
    )
    base_from_dmtac = base_from_single @ single_from_dmtac

    recovered = binding.solver_target_from_dmtac_target(base_from_dmtac)

    np.testing.assert_allclose(recovered, base_from_single, atol=1e-12)
    assert binding.to_payload()["position_offset_m"] == pytest.approx(0.131)


def test_final_settled_frame_does_not_erase_intermediate_ik_failure() -> None:
    assert motion.classify_motion_status(
        final_endpoint_status="settled",
        endpoint_status_counts={"tracking": 20, "ik_target_unmet": 1, "settled": 5},
    ) == motion.MOTION_STATUS_IK_TARGET_UNMET
    assert motion.classify_motion_status(
        final_endpoint_status="settled",
        endpoint_status_counts={"tracking": 20, "settled": 5},
    ) == motion.MOTION_STATUS_COMPLETED


class _FakeScene:
    def get_arm_qpos(self, _side: str) -> np.ndarray:
        return np.zeros(6, dtype=float)


class _FakeRetargeter:
    def tool_tcp_world_targets_for_qpos_pair(
        self, *, left_qpos: np.ndarray, right_qpos: np.ndarray
    ) -> dict[str, np.ndarray]:
        del left_qpos, right_qpos
        return {"left": np.eye(4), "right": np.eye(4)}


class _FakeControl:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.retargeter = _FakeRetargeter()
        self.world_to_base_by_side = {"left": np.eye(4), "right": np.eye(4)}
        self.world_from_desk = np.eye(4)
        self.contract = SimpleNamespace(
            timing=SimpleNamespace(ik_hz=60.0, command_hz=120.0, physics_hz=1000.0)
        )
        self.dispatch_count = 0

    def solve_and_dispatch(self, *args, **kwargs):
        del args, kwargs
        self.dispatch_count += 1
        raise AssertionError("rejected endpoint must not dispatch")


def _blocked_receipt() -> GraspReachabilityReceipt:
    return GraspReachabilityReceipt(
        status=GraspReachabilityStatus.COLLISION_BLOCKED,
        reason_code="all_accepted_ik_candidates_collision_blocked",
        proof_scope="endpoint_collision_candidates",
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        target_pose=tuple(tuple(row) for row in np.eye(4)),
    )


def test_rejected_endpoint_emits_no_motion_or_dispatch(monkeypatch) -> None:
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=np.eye(4),
        position_offset_m=0.0,
        orientation_offset_deg=0.0,
    )
    monkeypatch.setattr(
        motion,
        "evaluate_motion_target_endpoint",
        lambda _control, _target: (_blocked_receipt(), binding, None),
    )
    control = _FakeControl()
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test",
    )

    playback = execute_grasp_motion(control, target)

    assert playback.motion_status == motion.MOTION_STATUS_ENDPOINT_REJECTED
    assert playback.frame_count == 0
    assert playback.summary["motion_emitted"] is False
    assert control.dispatch_count == 0


def test_cli_requires_position_and_orientation_as_one_explicit_pose() -> None:
    parser = build_parser()

    _validate_args(parser.parse_args([]))
    _validate_args(
        parser.parse_args(
            [
                "--target-xyz-m",
                "0.3",
                "0.1",
                "0.5",
                "--target-rpy-deg",
                "0",
                "90",
                "0",
            ]
        )
    )
    with pytest.raises(ValueError, match="explicit target requires"):
        _validate_args(
            parser.parse_args(["--target-xyz-m", "0.3", "0.1", "0.5"])
        )
    with pytest.raises(ValueError, match="explicit target requires"):
        _validate_args(
            parser.parse_args(["--target-rpy-deg", "0", "90", "0"])
        )
