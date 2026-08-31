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
from sim.sim_real_tests.mujoco import (
    continuous_random_grasp_pose_motion as random_motion,
)
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    MOTION_STATUS_ENDPOINT_REJECTED,
    MOTION_STATUS_PATH_COLLISION_BLOCKED,
    GraspMotionPlayback,
    GraspMotionTarget,
    ToolTcpFrameBinding,
    build_smooth_se3_trajectory,
    execute_grasp_motion,
    pose_from_xyz_quat_xyzw,
    pose_from_xyz_rpy_deg,
    quintic_progress,
    required_motion_duration_sec,
)
from sim.sim_real_tests.mujoco.continuous_random_grasp_pose_motion import (
    ContinuousRandomGraspPoseConfig,
    RawCartesianGraspTargetStream,
    classify_random_grasp_attempt,
    sample_random_joint_fk_target,
)
from sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion import (
    _uses_continuous_random_mode,
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


def test_raw_cartesian_target_stream_is_deterministic_and_never_prefilters() -> None:
    config = ContinuousRandomGraspPoseConfig(
        seed=17,
        position_min_m=(-0.4, -0.3, 0.1),
        position_max_m=(0.2, 0.5, 0.9),
    )
    lhs = RawCartesianGraspTargetStream(side="left", config=config)
    rhs = RawCartesianGraspTargetStream(side="left", config=config)

    lhs_first = lhs.next_target()
    lhs_second = lhs.next_target()
    rhs_first = rhs.next_target()

    np.testing.assert_array_equal(
        lhs_first.target_pose_base_from_dmtac_tcp,
        rhs_first.target_pose_base_from_dmtac_tcp,
    )
    assert not np.array_equal(
        lhs_first.target_pose_base_from_dmtac_tcp,
        lhs_second.target_pose_base_from_dmtac_tcp,
    )
    xyz = lhs_first.target_pose_base_from_dmtac_tcp[:3, 3]
    assert np.all(xyz >= np.asarray(config.position_min_m))
    assert np.all(xyz < np.asarray(config.position_max_m))
    rotation = lhs_first.target_pose_base_from_dmtac_tcp[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert lhs_first.known_qpos_rad is None
    assert lhs_first.metadata["sample_index"] == 0
    assert lhs_first.metadata["rejection_sampled"] is False
    assert len(str(lhs_first.metadata["target_pose_content_hash"])) == 64


def test_joint_fk_target_is_one_draw_without_collision_prefilter(monkeypatch) -> None:
    monkeypatch.setattr(
        random_motion,
        "build_rm65_single_arm_ik_solver",
        lambda _config: SimpleNamespace(
            q_min=np.full(6, -2.0),
            q_max=np.full(6, 2.0),
        ),
    )
    target = sample_random_joint_fk_target(
        _FakeControl(),
        side="left",
        seed=23,
        sample_index=5,
        maximum_joint_offset_deg=30.0,
    )

    assert target.known_qpos_rad is not None
    assert target.metadata["sample_index"] == 5
    assert target.metadata["sample_kind"] == "random_joint_fk_candidate"
    assert target.metadata["rejection_sampled"] is False
    assert target.metadata["selected_max_joint_motion_deg"] <= 30.0


def _random_attempt_playback(
    *,
    reachability_status: GraspReachabilityStatus,
    motion_status: str = MOTION_STATUS_ENDPOINT_REJECTED,
    summary: dict[str, object] | None = None,
    frame_count: int = 0,
) -> GraspMotionPlayback:
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test_random_cartesian",
        metadata={
            "sample_index": 4,
            "target_pose_content_hash": "a" * 64,
        },
    )
    receipt = GraspReachabilityReceipt(
        status=reachability_status,
        reason_code=f"test_{reachability_status.value}",
        proof_scope="test",
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        target_pose=tuple(tuple(row) for row in np.eye(4)),
    )
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=np.eye(4),
        position_offset_m=0.0,
        orientation_offset_deg=0.0,
    )
    frames = tuple(
        {"left": np.zeros(6), "right": np.zeros(6)} for _ in range(frame_count)
    )
    return GraspMotionPlayback(
        motion_status=motion_status,
        side="left",
        target=target,
        endpoint_receipt=receipt,
        frame_binding=binding,
        target_pose_world_by_side={"left": np.eye(4), "right": np.eye(4)},
        reference_pose_world=(),
        actual_pose_world=(),
        actual_qpos_by_frame=frames,
        summary=summary or {"motion_emitted": False},
    )


def test_random_attempt_classification_never_calls_ik_exhaustion_unreachable() -> None:
    proven = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.KINEMATICALLY_UNREACHABLE
        )
    )
    exhausted = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.IK_SEARCH_EXHAUSTED
        )
    )
    collision = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.COLLISION_BLOCKED
        )
    )

    assert proven.proven_unreachable is True
    assert proven.terminal_classification == "proven_kinematically_unreachable"
    assert exhausted.proven_unreachable is False
    assert exhausted.terminal_classification == (
        "ik_search_exhausted_not_proven_unreachable"
    )
    assert collision.terminal_classification == "endpoint_collision_blocked"
    assert all(not outcome.motion_emitted for outcome in (proven, exhausted, collision))


def test_random_attempt_path_collision_is_distinct_and_emits_zero_motion() -> None:
    outcome = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.REACHABLE,
            motion_status=MOTION_STATUS_PATH_COLLISION_BLOCKED,
            summary={
                "motion_emitted": False,
                "path_preflight": {"reason_code": "blocked_at_progress_0_4"},
            },
        )
    )

    assert outcome.reachability_status == "reachable"
    assert outcome.terminal_classification == "direct_path_collision_blocked"
    assert outcome.reason_code == "blocked_at_progress_0_4"
    assert outcome.motion_emitted is False


def test_continuous_cli_is_default_and_headless_requires_a_finite_count() -> None:
    parser = build_parser()
    continuous = parser.parse_args([])
    assert _uses_continuous_random_mode(continuous) is True
    _validate_args(continuous)

    single = parser.parse_args(["--single-reachable-demo"])
    assert _uses_continuous_random_mode(single) is False
    _validate_args(single)

    with pytest.raises(ValueError, match="headless continuous mode requires"):
        _validate_args(parser.parse_args(["--headless"]))
    finite_headless = parser.parse_args(
        ["--headless", "--random-sample-count", "3"]
    )
    _validate_args(finite_headless)
