"""Plan and execute one arbitrary RM65 grasp TCP pose in MuJoCo.

The public target is ``arm_base_from_dmtac_tool_tcp`` for one selected side.
Endpoint admission is typed and bounded.  A reachable endpoint is converted to
the canonical desk frame, interpolated with a zero-endpoint-derivative quintic
SE(3) reference, and sent through the existing 60 Hz collision-aware IK,
120 Hz joint-command planner, physical actuators, and MuJoCo physics.

This simulation test module owns no real-device authority and never dispatches
to hardware.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.position_control import ResampledMotionReference
from robot_motion.rm65.grasp_reachability import (
    CollisionCandidateAssessment,
    CollisionCandidateStatus,
    ConservativeWorkspaceEnvelope,
    GraspReachabilityReceipt,
    GraspReachabilityStatus,
    IKCandidate,
    evaluate_grasp_pose_reachability,
)
from robot_motion.rm65.ik import RM65SingleArmIKConfig, build_rm65_single_arm_ik_solver
from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl
from sim.sim_real_tests.rm65.random_grasp_pose_reachability import (
    DEFAULT_REFERENCE_FRAME_ID,
    DEFAULT_REFERENCE_FRAME_SPEC_HASH,
    RM65RandomGraspPoseReachabilityConfig,
    annotate_ik_search_exhaustion,
    build_rm65_multistart_candidate_provider,
)

SIDES = ("left", "right")
MOTION_STATUS_COMPLETED = "completed"
MOTION_STATUS_ENDPOINT_REJECTED = "endpoint_rejected"
MOTION_STATUS_PATH_COLLISION_BLOCKED = "path_collision_blocked"
MOTION_STATUS_IK_TARGET_UNMET = "ik_target_unmet"
MOTION_STATUS_TRACKING_INCOMPLETE = "tracking_incomplete"
DEFAULT_IK_TARGET_UNMET_RETRY_LIMIT = 2


def _pose(value: Any, *, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must have homogeneous final row [0, 0, 0, 1]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must be proper")
    return pose.copy()


def pose_from_xyz_quat_xyzw(
    xyz_m: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=float).reshape(-1)
    quaternion = np.asarray(quaternion_xyzw, dtype=float).reshape(-1)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("xyz_m must contain three finite values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion_xyzw must have nonzero norm")
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    pose[:3, 3] = xyz
    return pose


def pose_from_xyz_rpy_deg(
    xyz_m: Sequence[float], rpy_deg: Sequence[float]
) -> np.ndarray:
    rpy = np.asarray(rpy_deg, dtype=float).reshape(-1)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("rpy_deg must contain three finite values")
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    pose[:3, 3] = np.asarray(xyz_m, dtype=float).reshape(3)
    return _pose(pose, name="target_pose")


def quintic_progress(unit_time: float | np.ndarray) -> float | np.ndarray:
    """Minimum-jerk scalar progress with zero velocity/acceleration at ends."""

    value = np.clip(np.asarray(unit_time, dtype=float), 0.0, 1.0)
    result = value**3 * (10.0 - 15.0 * value + 6.0 * value**2)
    return float(result) if result.ndim == 0 else result


def required_motion_duration_sec(
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    linear_speed_m_s: float,
    angular_speed_deg_s: float,
    minimum_duration_sec: float,
) -> tuple[float, dict[str, float]]:
    start = _pose(start_pose, name="start_pose")
    target = _pose(target_pose, name="target_pose")
    linear_speed = float(linear_speed_m_s)
    angular_speed = math.radians(float(angular_speed_deg_s))
    minimum = float(minimum_duration_sec)
    if not math.isfinite(linear_speed) or linear_speed <= 0.0:
        raise ValueError("linear_speed_m_s must be finite and positive")
    if not math.isfinite(angular_speed) or angular_speed <= 0.0:
        raise ValueError("angular_speed_deg_s must be finite and positive")
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_duration_sec must be finite and positive")
    translation_distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    rotation_angle = float(
        Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).magnitude()
    )
    # max d/du of 10u^3 - 15u^4 + 6u^5 is 1.875 at u=0.5.
    peak_progress_derivative = 1.875
    translation_duration = peak_progress_derivative * translation_distance / linear_speed
    rotation_duration = peak_progress_derivative * rotation_angle / angular_speed
    duration = max(minimum, translation_duration, rotation_duration)
    return duration, {
        "translation_distance_m": translation_distance,
        "rotation_distance_deg": math.degrees(rotation_angle),
        "linear_speed_limit_m_s": linear_speed,
        "angular_speed_limit_deg_s": math.degrees(angular_speed),
        "minimum_duration_sec": minimum,
        "duration_sec": duration,
        "peak_progress_derivative": peak_progress_derivative,
    }


def build_smooth_se3_trajectory(
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    reference_hz: float,
    duration_sec: float,
    terminal_hold_sec: float,
) -> tuple[np.ndarray, ...]:
    """Return geodesic-orientation, quintic-progress 4x4 reference poses."""

    start = _pose(start_pose, name="start_pose")
    target = _pose(target_pose, name="target_pose")
    rate = float(reference_hz)
    duration = float(duration_sec)
    hold = float(terminal_hold_sec)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("reference_hz must be finite and positive")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    if not math.isfinite(hold) or hold < 0.0:
        raise ValueError("terminal_hold_sec must be finite and non-negative")
    motion_frames = math.ceil(duration * rate) + 1
    unit_times = np.linspace(0.0, 1.0, motion_frames, dtype=float)
    relative_rotvec = Rotation.from_matrix(
        start[:3, :3].T @ target[:3, :3]
    ).as_rotvec()
    poses: list[np.ndarray] = []
    for progress in np.asarray(quintic_progress(unit_times), dtype=float):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = start[:3, 3] + progress * (
            target[:3, 3] - start[:3, 3]
        )
        pose[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(
            progress * relative_rotvec
        ).as_matrix()
        poses.append(pose)
    hold_frames = math.ceil(hold * rate)
    poses.extend(target.copy() for _ in range(hold_frames))
    return tuple(poses)


@dataclass(frozen=True)
class ToolTcpFrameBinding:
    side: str
    single_tool_from_dmtac_tool: np.ndarray
    position_offset_m: float
    orientation_offset_deg: float

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        transform = _pose(
            self.single_tool_from_dmtac_tool,
            name="single_tool_from_dmtac_tool",
        )
        transform.setflags(write=False)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "single_tool_from_dmtac_tool", transform)
        object.__setattr__(self, "position_offset_m", float(self.position_offset_m))
        object.__setattr__(
            self, "orientation_offset_deg", float(self.orientation_offset_deg)
        )

    def solver_target_from_dmtac_target(self, target_pose_base: np.ndarray) -> np.ndarray:
        return _pose(target_pose_base, name="target_pose_base") @ np.linalg.inv(
            self.single_tool_from_dmtac_tool
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "side": self.side,
            "semantics": "base_from_dmtac_tool = base_from_single_tool @ single_tool_from_dmtac_tool",
            "single_tool_from_dmtac_tool": self.single_tool_from_dmtac_tool.tolist(),
            "position_offset_m": self.position_offset_m,
            "orientation_offset_deg": self.orientation_offset_deg,
        }


@dataclass(frozen=True)
class GraspMotionTarget:
    side: str
    target_pose_base_from_dmtac_tcp: np.ndarray
    source: str
    known_qpos_rad: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        pose = _pose(
            self.target_pose_base_from_dmtac_tcp,
            name="target_pose_base_from_dmtac_tcp",
        )
        pose.setflags(write=False)
        qpos = self.known_qpos_rad
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=float).reshape(-1).copy()
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                raise ValueError("known_qpos_rad must contain six finite values")
            qpos.setflags(write=False)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "target_pose_base_from_dmtac_tcp", pose)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "known_qpos_rad", qpos)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class GraspMotionPlayback:
    motion_status: str
    side: str
    target: GraspMotionTarget
    endpoint_receipt: GraspReachabilityReceipt
    frame_binding: ToolTcpFrameBinding
    target_pose_world_by_side: Mapping[str, np.ndarray]
    reference_pose_world: tuple[np.ndarray, ...]
    actual_pose_world: tuple[np.ndarray, ...]
    actual_qpos_by_frame: tuple[Mapping[str, np.ndarray], ...]
    summary: Mapping[str, object]

    @property
    def frame_count(self) -> int:
        return len(self.actual_qpos_by_frame)

    def to_payload(self) -> dict[str, object]:
        return {
            "motion_status": self.motion_status,
            "side": self.side,
            "frame_count": self.frame_count,
            "target_source": self.target.source,
            "target_metadata": _plain(self.target.metadata),
            "target_known_qpos_rad": (
                None
                if self.target.known_qpos_rad is None
                else self.target.known_qpos_rad.tolist()
            ),
            "target_pose_base_from_dmtac_tcp": (
                self.target.target_pose_base_from_dmtac_tcp.tolist()
            ),
            "endpoint_reachability": self.endpoint_receipt.to_dict(),
            "frame_binding": self.frame_binding.to_payload(),
            "summary": _plain(self.summary),
        }


@dataclass(frozen=True)
class JointFkReferencePlan:
    status: str
    pose_desk: tuple[np.ndarray, ...]
    qpos_rad: tuple[np.ndarray, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in {"safe", "collision_blocked"}:
            raise ValueError("joint-FK plan status must be safe or collision_blocked")
        if len(self.pose_desk) != len(self.qpos_rad):
            raise ValueError("joint-FK plan poses and qpos must have equal lengths")
        if self.status == "safe" and len(self.pose_desk) < 2:
            raise ValueError("safe joint-FK plan must contain at least two frames")
        if self.status == "collision_blocked" and self.pose_desk:
            raise ValueError("collision-blocked joint-FK plan must emit no frames")
        poses: list[np.ndarray] = []
        qpos_values: list[np.ndarray] = []
        for index, value in enumerate(self.pose_desk):
            pose = _pose(value, name=f"pose_desk[{index}]")
            pose.setflags(write=False)
            poses.append(pose)
        for index, value in enumerate(self.qpos_rad):
            qpos = np.asarray(value, dtype=float).reshape(-1).copy()
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                raise ValueError(f"qpos_rad[{index}] must contain six finite values")
            qpos.setflags(write=False)
            qpos_values.append(qpos)
        object.__setattr__(self, "pose_desk", tuple(poses))
        object.__setattr__(self, "qpos_rad", tuple(qpos_values))
        object.__setattr__(self, "metadata", dict(self.metadata))


def derive_tool_tcp_frame_binding(
    control: DeskTcpPositionControl,
    *,
    side: str,
    single_arm_solver: Any,
) -> ToolTcpFrameBinding:
    normalized_side = str(side).strip().lower()
    if normalized_side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    qpos = {
        item: control.scene.get_arm_qpos(item).copy() for item in SIDES
    }
    single_base_from_tool = np.asarray(
        single_arm_solver.compute_pose(qpos[normalized_side]), dtype=float
    ).reshape(4, 4)
    world_from_dmtac = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=qpos["left"], right_qpos=qpos["right"]
    )[normalized_side]
    base_from_dmtac = control.world_to_base_by_side[normalized_side] @ world_from_dmtac
    binding = np.linalg.inv(single_base_from_tool) @ base_from_dmtac
    return ToolTcpFrameBinding(
        side=normalized_side,
        single_tool_from_dmtac_tool=binding,
        position_offset_m=float(np.linalg.norm(binding[:3, 3])),
        orientation_offset_deg=float(
            math.degrees(Rotation.from_matrix(binding[:3, :3]).magnitude())
        ),
    )


def _collision_checker(
    control: DeskTcpPositionControl,
    *,
    side: str,
) -> Any:
    projector = control.scene.collision_projector()
    base_vector = control.scene.kinematic_controlled_qpos()

    def check(candidate: IKCandidate, _target_pose: np.ndarray) -> CollisionCandidateAssessment:
        if candidate.qpos_rad is None:
            return CollisionCandidateAssessment(
                status=CollisionCandidateStatus.ERROR,
                reason_code="candidate_qpos_missing",
            )
        if projector is None:
            return CollisionCandidateAssessment(
                status=CollisionCandidateStatus.SEARCH_EXHAUSTED,
                reason_code="mujoco_collision_projector_unavailable",
            )
        target = control.scene.kinematic_target_vector(
            base_vector,
            {side: np.asarray(candidate.qpos_rad, dtype=float)},
        )
        result = projector.check_qpos(control.scene.data, target)
        payload = result.to_dict()
        return CollisionCandidateAssessment(
            status=(
                CollisionCandidateStatus.SAFE
                if result.safe
                else CollisionCandidateStatus.BLOCKED
            ),
            reason_code=(
                "mujoco_endpoint_collision_safe"
                if result.safe
                else "mujoco_endpoint_collision_blocked"
            ),
            minimum_distance_m=(
                None
                if payload.get("minimum_distance_m") is None
                else float(payload["minimum_distance_m"])
            ),
            required_distance_m=float(projector.policy.minimum_clearance_m),
            obstacle_ids=tuple(
                str(value)
                for value in (payload.get("body1"), payload.get("body2"))
                if value
            ),
            metadata=payload,
        )

    return check


def evaluate_motion_target_endpoint(
    control: DeskTcpPositionControl,
    target: GraspMotionTarget,
    *,
    config: RM65RandomGraspPoseReachabilityConfig | None = None,
) -> tuple[GraspReachabilityReceipt, ToolTcpFrameBinding, Any]:
    cfg = config or RM65RandomGraspPoseReachabilityConfig()
    solver = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(controlled_frame="tool_tcp")
    )
    binding = derive_tool_tcp_frame_binding(
        control, side=target.side, single_arm_solver=solver
    )
    solver_target = binding.solver_target_from_dmtac_target(
        target.target_pose_base_from_dmtac_tcp
    )
    current = control.scene.get_arm_qpos(target.side).copy()
    seed_rng = np.random.Generator(np.random.PCG64(int(cfg.seed) ^ 0x47524153))
    seeds: list[np.ndarray] = []
    if target.known_qpos_rad is not None:
        seeds.append(np.asarray(target.known_qpos_rad, dtype=float).copy())
    seeds.extend((current, np.clip(np.zeros(6), solver.q_min, solver.q_max)))
    while len(seeds) < int(cfg.max_ik_candidates):
        seeds.append(seed_rng.uniform(solver.q_min, solver.q_max))
    provider = build_rm65_multistart_candidate_provider(
        solver,
        qpos_seeds=seeds[: int(cfg.max_ik_candidates)],
        max_iterations=int(cfg.ik_max_iterations),
        max_joint_step_rad=float(cfg.ik_max_joint_step_rad),
        damping=float(cfg.ik_damping),
        near_miss_refinement_enabled=cfg.near_miss_refinement_enabled,
        near_miss_position_threshold_m=float(cfg.near_miss_position_threshold_m),
        near_miss_orientation_threshold_rad=math.radians(
            float(cfg.near_miss_orientation_threshold_deg)
        ),
        near_miss_max_nfev=int(cfg.near_miss_max_nfev),
        acceptance_position_m=float(cfg.maximum_position_residual_m),
        acceptance_orientation_rad=math.radians(
            float(cfg.maximum_orientation_residual_deg)
        ),
    )
    receipt = evaluate_grasp_pose_reachability(
        solver_target,
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        candidate_provider=provider,
        max_candidates=int(cfg.max_ik_candidates),
        maximum_position_residual_m=float(cfg.maximum_position_residual_m),
        maximum_orientation_residual_rad=math.radians(
            float(cfg.maximum_orientation_residual_deg)
        ),
        workspace_envelope=ConservativeWorkspaceEnvelope(
            reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
            reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
            origin_xyz_m=cfg.conservative_workspace_origin_m,
            maximum_radius_m=float(cfg.conservative_workspace_maximum_radius_m),
        ),
        collision_checker=_collision_checker(control, side=target.side),
    )
    receipt = annotate_ik_search_exhaustion(
        receipt,
        solver=solver,
        target_pose=solver_target,
        maximum_position_residual_m=float(cfg.maximum_position_residual_m),
        near_miss_position_threshold_m=float(cfg.near_miss_position_threshold_m),
        near_miss_orientation_threshold_rad=math.radians(
            float(cfg.near_miss_orientation_threshold_deg)
        ),
        position_diagnostic_max_nfev=int(cfg.exhaustion_position_diagnostic_max_nfev),
        joint_limit_margin_rad=float(cfg.joint_limit_margin_rad),
        enabled=cfg.exhaustion_position_diagnostic_enabled,
    )
    return receipt, binding, solver


def _direct_joint_path_collision_assessment(
    control: DeskTcpPositionControl,
    *,
    side: str,
    start_qpos_rad: np.ndarray,
    target_qpos_rad: np.ndarray,
) -> dict[str, object]:
    normalized_side = str(side).strip().lower()
    if normalized_side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    start = np.asarray(start_qpos_rad, dtype=float).reshape(-1)
    target = np.asarray(target_qpos_rad, dtype=float).reshape(-1)
    if (
        start.shape != (6,)
        or target.shape != (6,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("direct path endpoints must be finite 6-vectors")
    projector = control.scene.collision_projector()
    if projector is None:
        raise RuntimeError("direct joint path requires MuJoCo collision safety")
    base_vector = control.scene.kinematic_controlled_qpos()
    maximum_joint_motion_deg = float(
        np.max(np.abs(np.rad2deg(target - start)))
    )
    sample_count = max(
        241,
        math.ceil(maximum_joint_motion_deg / 0.1) + 1,
    )
    sample_count = min(sample_count, 5001)
    minimum_clearance_m = math.inf
    for index, progress in enumerate(
        np.linspace(0.0, 1.0, sample_count, dtype=float)
    ):
        qpos = start + float(progress) * (target - start)
        target_vector = control.scene.kinematic_target_vector(
            base_vector, {normalized_side: qpos}
        )
        collision = projector.check_qpos(control.scene.data, target_vector)
        payload = collision.to_dict()
        distance = payload.get("minimum_distance_m")
        if distance is not None:
            minimum_clearance_m = min(minimum_clearance_m, float(distance))
        if not collision.safe:
            return {
                "safe": False,
                "reason_code": "mujoco_direct_joint_path_collision_blocked",
                "blocked_sample_index": index,
                "blocked_progress": float(progress),
                "collision_sample_count": sample_count,
                "minimum_path_clearance_m": (
                    None
                    if not math.isfinite(minimum_clearance_m)
                    else minimum_clearance_m
                ),
                "blocked_collision": payload,
            }
    return {
        "safe": True,
        "reason_code": "mujoco_direct_joint_path_collision_safe",
        "collision_sample_count": sample_count,
        "minimum_path_clearance_m": (
            None
            if not math.isfinite(minimum_clearance_m)
            else minimum_clearance_m
        ),
    }


def select_random_reachable_demo_target(
    control: DeskTcpPositionControl,
    *,
    side: str,
    seed: int,
    maximum_trials: int = 256,
    maximum_joint_offset_deg: float = 55.0,
    minimum_joint_motion_deg: float = 25.0,
) -> GraspMotionTarget:
    """Sample a nontrivial, endpoint-collision-safe qpos and expose its TCP pose."""

    normalized_side = str(side).strip().lower()
    if normalized_side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if int(maximum_trials) < 1:
        raise ValueError("maximum_trials must be positive")
    maximum_offset = math.radians(float(maximum_joint_offset_deg))
    minimum_motion = math.radians(float(minimum_joint_motion_deg))
    if not math.isfinite(maximum_offset) or maximum_offset <= 0.0:
        raise ValueError("maximum_joint_offset_deg must be finite and positive")
    if not math.isfinite(minimum_motion) or minimum_motion <= 0.0:
        raise ValueError("minimum_joint_motion_deg must be finite and positive")
    solver = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(controlled_frame="tool_tcp")
    )
    current = {
        item: control.scene.get_arm_qpos(item).copy() for item in SIDES
    }
    projector = control.scene.collision_projector()
    if projector is None:
        raise RuntimeError("demo target selection requires MuJoCo collision safety")
    base_vector = control.scene.kinematic_controlled_qpos()
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for trial in range(int(maximum_trials)):
        offset = rng.uniform(-maximum_offset, maximum_offset, size=6)
        candidate = np.clip(
            current[normalized_side] + offset,
            np.asarray(solver.q_min, dtype=float),
            np.asarray(solver.q_max, dtype=float),
        )
        motion = float(np.max(np.abs(candidate - current[normalized_side])))
        if motion < minimum_motion:
            continue
        target_vector = control.scene.kinematic_target_vector(
            base_vector, {normalized_side: candidate}
        )
        collision = projector.check_qpos(control.scene.data, target_vector)
        if not collision.safe:
            continue
        path_collision = _direct_joint_path_collision_assessment(
            control,
            side=normalized_side,
            start_qpos_rad=current[normalized_side],
            target_qpos_rad=candidate,
        )
        if not bool(path_collision["safe"]):
            continue
        qpos_pair = {item: values.copy() for item, values in current.items()}
        qpos_pair[normalized_side] = candidate.copy()
        world_from_dmtac = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos_pair["left"], right_qpos=qpos_pair["right"]
        )[normalized_side]
        base_from_dmtac = (
            control.world_to_base_by_side[normalized_side] @ world_from_dmtac
        )
        return GraspMotionTarget(
            side=normalized_side,
            target_pose_base_from_dmtac_tcp=base_from_dmtac,
            source="collision_safe_random_reachable_qpos_fk",
            known_qpos_rad=candidate,
            metadata={
                "seed": int(seed),
                "trial_index": trial,
                "maximum_joint_offset_deg": float(maximum_joint_offset_deg),
                "minimum_joint_motion_deg": float(minimum_joint_motion_deg),
                "selected_max_joint_motion_deg": math.degrees(motion),
                "endpoint_collision": collision.to_dict(),
                "direct_path_collision": path_collision,
            },
        )
    raise RuntimeError(
        "no collision-safe nontrivial demo target found within "
        f"{int(maximum_trials)} trials"
    )


def _selected_candidate_qpos(receipt: GraspReachabilityReceipt) -> np.ndarray:
    selected_index = receipt.selected_candidate_index
    if selected_index is None:
        raise RuntimeError("reachable endpoint receipt lacks a selected IK candidate")
    for candidate in receipt.candidates:
        if int(candidate.get("candidate_index", -1)) != int(selected_index):
            continue
        qpos = np.asarray(candidate.get("qpos_rad"), dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise RuntimeError("selected endpoint IK candidate has invalid qpos")
        return qpos.copy()
    raise RuntimeError("selected endpoint IK candidate is absent from the receipt")


def _pose_step_statistics(
    poses: Sequence[np.ndarray],
    *,
    step_size: float,
) -> tuple[float, float]:
    if len(poses) < 2:
        return 0.0, 0.0
    denominator = float(step_size)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("step_size must be finite and positive")
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for lhs, rhs in pairwise(poses):
        maximum_translation = max(
            maximum_translation,
            float(np.linalg.norm(rhs[:3, 3] - lhs[:3, 3])) / denominator,
        )
        maximum_rotation = max(
            maximum_rotation,
            float(
                Rotation.from_matrix(lhs[:3, :3].T @ rhs[:3, :3]).magnitude()
            )
            / denominator,
        )
    return maximum_translation, maximum_rotation


def classify_motion_status(
    *,
    final_endpoint_status: str,
    endpoint_status_counts: Mapping[str, int],
    unrecovered_ik_target_unmet_count: int | None = None,
) -> str:
    """Classify the path while preserving only unresolved execution failures.

    Older callers can omit ``unrecovered_ik_target_unmet_count`` and retain the
    conservative historical rule that any observed IK miss poisons the run.
    The execution loop supplies the explicit unresolved count so a bounded,
    same-reference retry that genuinely recovers remains visible in diagnostics
    without being misreported as a terminal failure.
    """

    if unrecovered_ik_target_unmet_count is None:
        unresolved = int(endpoint_status_counts.get("ik_target_unmet", 0))
    else:
        unresolved = int(unrecovered_ik_target_unmet_count)
        if unresolved < 0:
            raise ValueError("unrecovered_ik_target_unmet_count must be non-negative")
    if unresolved > 0:
        return MOTION_STATUS_IK_TARGET_UNMET
    if str(final_endpoint_status) == "settled":
        return MOTION_STATUS_COMPLETED
    if str(final_endpoint_status) == "ik_target_unmet":
        return MOTION_STATUS_IK_TARGET_UNMET
    return MOTION_STATUS_TRACKING_INCOMPLETE


def _dispatch_frozen_reference_with_bounded_ik_recovery(
    control: DeskTcpPositionControl,
    reference: ResampledMotionReference,
    *,
    inactive_side: str,
    retry_limit: int,
    settle_once: Callable[[], int],
) -> dict[str, object]:
    """Dispatch one frozen reference and retry only an explicit IK target miss.

    Every attempt uses the exact same ``ResampledMotionReference`` object and
    the existing collision-aware ``solve_and_dispatch`` owner.  The helper does
    not rewrite targets, inject qpos, or relax lower safety.  Exhaustion is a
    terminal, explicit result for the caller rather than a swallowed transient.
    """

    limit = int(retry_limit)
    if limit < 0:
        raise ValueError("retry_limit must be non-negative")
    observations: list[dict[str, object]] = []
    result_statuses: list[str] = []
    physics_ticks_by_attempt: list[int] = []
    total_physics_ticks = 0
    initial_unmet = False
    final_result: Any = None

    for attempt_index in range(limit + 1):
        final_result = control.solve_and_dispatch(
            reference,
            gripper_close_by_side={"left": 0.0, "right": 0.0},
            hold_sides=(inactive_side,),
        )
        result_statuses.append(str(final_result.status))
        physics_ticks = int(settle_once())
        physics_ticks_by_attempt.append(physics_ticks)
        total_physics_ticks += physics_ticks
        observation = dict(control.endpoint_tracking_observation())
        observations.append(observation)
        unmet = str(observation.get("status", "unknown")) == "ik_target_unmet"
        if attempt_index == 0:
            initial_unmet = unmet
        if not unmet:
            break

    assert final_result is not None
    final_status = str(observations[-1].get("status", "unknown"))
    unrecovered = final_status == "ik_target_unmet"
    recovered = initial_unmet and not unrecovered
    retry_count = len(observations) - 1
    return {
        "result": final_result,
        "result_statuses": tuple(result_statuses),
        "observations": tuple(observations),
        "initial_ik_target_unmet": initial_unmet,
        "status": (
            "unrecovered"
            if unrecovered
            else "transient_recovered"
            if recovered
            else "not_needed"
        ),
        "recovered": recovered,
        "unrecovered": unrecovered,
        "retry_count": retry_count,
        "attempt_count": len(observations),
        "physics_ticks": total_physics_ticks,
        "retry_physics_ticks": sum(physics_ticks_by_attempt[1:]),
    }


def build_collision_checked_joint_fk_reference_plan(
    control: DeskTcpPositionControl,
    *,
    side: str,
    start_qpos_rad: np.ndarray,
    target_qpos_rad: np.ndarray,
    linear_speed_m_s: float,
    angular_speed_deg_s: float,
    minimum_duration_sec: float,
) -> JointFkReferencePlan:
    """Build a reachable-by-construction direct path for the simulation viewer.

    The path is a quintic scalar interpolation in joint space.  Full-scene
    collision admission samples the geometric path independently of time.
    Every emitted Cartesian reference is then obtained by FK from that admitted
    qpos path, avoiding the invalid assumption that a straight SE(3) segment
    between two reachable poses is itself continuously reachable.
    """

    normalized_side = str(side).strip().lower()
    if normalized_side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    start = np.asarray(start_qpos_rad, dtype=float).reshape(-1)
    target = np.asarray(target_qpos_rad, dtype=float).reshape(-1)
    if (
        start.shape != (6,)
        or target.shape != (6,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("start_qpos_rad and target_qpos_rad must be finite 6-vectors")
    linear_speed = float(linear_speed_m_s)
    angular_speed = math.radians(float(angular_speed_deg_s))
    minimum_duration = float(minimum_duration_sec)
    if not math.isfinite(linear_speed) or linear_speed <= 0.0:
        raise ValueError("linear_speed_m_s must be finite and positive")
    if not math.isfinite(angular_speed) or angular_speed <= 0.0:
        raise ValueError("angular_speed_deg_s must be finite and positive")
    if not math.isfinite(minimum_duration) or minimum_duration <= 0.0:
        raise ValueError("minimum_duration_sec must be finite and positive")

    initial_by_side = {
        item: control.scene.get_arm_qpos(item).copy() for item in SIDES
    }
    delta = target - start
    maximum_joint_motion_deg = float(np.max(np.abs(np.rad2deg(delta))))
    path_collision = _direct_joint_path_collision_assessment(
        control,
        side=normalized_side,
        start_qpos_rad=start,
        target_qpos_rad=target,
    )
    if not bool(path_collision["safe"]):
        return JointFkReferencePlan(
            status="collision_blocked",
            pose_desk=(),
            qpos_rad=(),
            metadata={
                "planner": "direct_joint_quintic_fk",
                **path_collision,
                "motion_emitted": False,
            },
        )

    inactive_side = "right" if normalized_side == "left" else "left"

    def world_pose(qpos: np.ndarray) -> np.ndarray:
        qpos_by_side = {
            item: values.copy() for item, values in initial_by_side.items()
        }
        qpos_by_side[normalized_side] = np.asarray(qpos, dtype=float).copy()
        return control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos_by_side["left"],
            right_qpos=qpos_by_side["right"],
        )[normalized_side]

    dense_count = 1001
    dense_u = np.linspace(0.0, 1.0, dense_count, dtype=float)
    dense_progress = np.asarray(quintic_progress(dense_u), dtype=float)
    dense_qpos = tuple(start + value * delta for value in dense_progress)
    dense_pose_world = tuple(world_pose(qpos) for qpos in dense_qpos)
    translation_per_unit, rotation_per_unit = _pose_step_statistics(
        dense_pose_world,
        step_size=1.0 / float(dense_count - 1),
    )
    joint_deg = np.rad2deg(np.stack(dense_qpos, axis=0))
    joint_deg_per_unit = np.max(
        np.abs(np.diff(joint_deg, axis=0)) * float(dense_count - 1),
        axis=0,
    )
    profiles = load_joint_velocity_limit_profiles()
    profile_name = str(control.config.joint_velocity_limit_profile)
    if profile_name not in profiles:
        raise RuntimeError(f"unknown joint velocity profile: {profile_name}")
    joint_velocity_limits = np.asarray(
        profiles[profile_name].max_velocity_deg_s, dtype=float
    )
    duration = max(
        minimum_duration,
        1.005 * translation_per_unit / linear_speed,
        1.005 * rotation_per_unit / angular_speed,
        1.005 * float(np.max(joint_deg_per_unit / joint_velocity_limits)),
    )
    reference_hz = float(control.contract.timing.ik_hz)
    frame_count = math.ceil(duration * reference_hz) + 1
    effective_duration = float(frame_count - 1) / reference_hz
    final_u = np.linspace(0.0, 1.0, frame_count, dtype=float)
    final_progress = np.asarray(quintic_progress(final_u), dtype=float)
    qpos_path = tuple(start + value * delta for value in final_progress)
    pose_world = tuple(world_pose(qpos) for qpos in qpos_path)
    pose_desk = tuple(
        np.linalg.inv(control.world_from_desk) @ pose for pose in pose_world
    )
    maximum_linear_speed, maximum_angular_speed = _pose_step_statistics(
        pose_world,
        step_size=1.0 / reference_hz,
    )
    maximum_joint_speed = np.max(
        np.abs(np.diff(np.rad2deg(np.stack(qpos_path, axis=0)), axis=0))
        * reference_hz,
        axis=0,
    )
    if maximum_linear_speed > linear_speed * 1.0001:
        raise RuntimeError("joint-FK plan exceeded requested linear TCP speed")
    if maximum_angular_speed > angular_speed * 1.0001:
        raise RuntimeError("joint-FK plan exceeded requested angular TCP speed")
    if np.any(maximum_joint_speed > joint_velocity_limits * 1.0001):
        raise RuntimeError("joint-FK plan exceeded the configured joint velocity profile")
    return JointFkReferencePlan(
        status="safe",
        pose_desk=pose_desk,
        qpos_rad=qpos_path,
        metadata={
            "planner": "direct_joint_quintic_fk",
            "path_reference_semantics": "dmtac_tcp_fk_of_admitted_joint_path",
            "inactive_side": inactive_side,
            "collision_sample_count": path_collision["collision_sample_count"],
            "minimum_path_clearance_m": path_collision[
                "minimum_path_clearance_m"
            ],
            "maximum_joint_motion_deg": maximum_joint_motion_deg,
            "reference_frame_count": frame_count,
            "duration_sec": effective_duration,
            "requested_linear_speed_limit_m_s": linear_speed,
            "requested_angular_speed_limit_deg_s": math.degrees(angular_speed),
            "maximum_reference_linear_speed_m_s": maximum_linear_speed,
            "maximum_reference_angular_speed_deg_s": math.degrees(
                maximum_angular_speed
            ),
            "joint_velocity_profile": profile_name,
            "maximum_reference_joint_speed_deg_s": maximum_joint_speed.tolist(),
            "joint_velocity_limits_deg_s": joint_velocity_limits.tolist(),
        },
    )


def execute_grasp_motion(
    control: DeskTcpPositionControl,
    target: GraspMotionTarget,
    *,
    linear_speed_m_s: float = 0.12,
    angular_speed_deg_s: float = 45.0,
    minimum_duration_sec: float = 2.0,
    terminal_hold_sec: float = 1.5,
    ik_target_unmet_retry_limit: int = DEFAULT_IK_TARGET_UNMET_RETRY_LIMIT,
) -> GraspMotionPlayback:
    retry_limit = int(ik_target_unmet_retry_limit)
    if retry_limit < 0:
        raise ValueError("ik_target_unmet_retry_limit must be non-negative")
    receipt, binding, _solver = evaluate_motion_target_endpoint(control, target)
    initial_qpos = {
        side: control.scene.get_arm_qpos(side).copy() for side in SIDES
    }
    initial_world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"], right_qpos=initial_qpos["right"]
    )
    world_from_base = np.linalg.inv(control.world_to_base_by_side[target.side])
    target_world_active = (
        world_from_base @ target.target_pose_base_from_dmtac_tcp
    )
    target_world_by_side = {
        side: initial_world[side].copy() for side in SIDES
    }
    target_world_by_side[target.side] = target_world_active.copy()
    empty_summary = {
        "reference_rate_hz": float(control.contract.timing.ik_hz),
        "command_rate_hz": float(control.contract.timing.command_hz),
        "physics_rate_hz": float(control.contract.timing.physics_hz),
        "motion_emitted": False,
    }
    if receipt.status is not GraspReachabilityStatus.REACHABLE:
        return GraspMotionPlayback(
            motion_status=MOTION_STATUS_ENDPOINT_REJECTED,
            side=target.side,
            target=target,
            endpoint_receipt=receipt,
            frame_binding=binding,
            target_pose_world_by_side=target_world_by_side,
            reference_pose_world=(),
            actual_pose_world=(),
            actual_qpos_by_frame=(),
            summary=empty_summary,
        )

    target_pose_desk = np.linalg.inv(control.world_from_desk) @ target_world_active
    selected_qpos = _selected_candidate_qpos(receipt)
    joint_fk_plan = build_collision_checked_joint_fk_reference_plan(
        control,
        side=target.side,
        start_qpos_rad=initial_qpos[target.side],
        target_qpos_rad=selected_qpos,
        linear_speed_m_s=linear_speed_m_s,
        angular_speed_deg_s=angular_speed_deg_s,
        minimum_duration_sec=minimum_duration_sec,
    )
    if joint_fk_plan.status == "collision_blocked":
        return GraspMotionPlayback(
            motion_status=MOTION_STATUS_PATH_COLLISION_BLOCKED,
            side=target.side,
            target=target,
            endpoint_receipt=receipt,
            frame_binding=binding,
            target_pose_world_by_side=target_world_by_side,
            reference_pose_world=(),
            actual_pose_world=(),
            actual_qpos_by_frame=(),
            summary={
                **empty_summary,
                "path_preflight": dict(joint_fk_plan.metadata),
            },
        )
    correction_duration, correction_metadata = required_motion_duration_sec(
        joint_fk_plan.pose_desk[-1],
        target_pose_desk,
        linear_speed_m_s=linear_speed_m_s,
        angular_speed_deg_s=angular_speed_deg_s,
        minimum_duration_sec=0.25,
    )
    terminal_correction = build_smooth_se3_trajectory(
        joint_fk_plan.pose_desk[-1],
        target_pose_desk,
        reference_hz=float(control.contract.timing.ik_hz),
        duration_sec=correction_duration,
        terminal_hold_sec=terminal_hold_sec,
    )
    active_poses_desk = joint_fk_plan.pose_desk + terminal_correction[1:]
    inactive_side = "right" if target.side == "left" else "left"
    inactive_pose_desk = np.linalg.inv(control.world_from_desk) @ initial_world[inactive_side]
    physics_ticks_exact = (
        float(control.contract.timing.physics_hz)
        / float(control.contract.timing.ik_hz)
    )
    physics_tick_accumulator = 0.0
    # Preserve the exact pre-dispatch state as frame zero.  The GUI therefore
    # replays the complete physical movement instead of beginning after the
    # first 16/17 MuJoCo ticks have already elapsed.
    actual_qpos_by_frame: list[dict[str, np.ndarray]] = [
        {side: values.copy() for side, values in initial_qpos.items()}
    ]
    actual_pose_world: list[np.ndarray] = [initial_world[target.side].copy()]
    reference_pose_world: list[np.ndarray] = [initial_world[target.side].copy()]
    ik_status_counts: dict[str, int] = {}
    endpoint_status_counts: dict[str, int] = {}
    recovery_records: list[dict[str, object]] = []
    recovery_retry_count = 0
    recovery_physics_ticks = 0
    execution_physics_ticks = 0
    recovered_reference_count = 0
    unrecovered_reference_count = 0
    dispatched_reference_count = 0
    maximum_tracking_error_m = 0.0

    def settle_reference_interval() -> int:
        nonlocal physics_tick_accumulator
        physics_tick_accumulator += physics_ticks_exact
        physics_ticks = math.floor(physics_tick_accumulator + 1e-12)
        physics_tick_accumulator -= physics_ticks
        for _ in range(physics_ticks):
            control.step_physics()
        return physics_ticks

    for index, active_pose_desk in enumerate(active_poses_desk):
        timestamp = float(index) / float(control.contract.timing.ik_hz)
        poses = {
            target.side: active_pose_desk,
            inactive_side: inactive_pose_desk,
        }
        reference = ResampledMotionReference(
            sequence=index,
            timestamp=timestamp,
            left_source_index=index,
            right_source_index=index,
            left_source_timestamp=timestamp,
            right_source_timestamp=timestamp,
            interpolation_alpha=0.0,
            pose_by_side={side: poses[side] for side in SIDES},
            metadata={
                "source": "arbitrary_grasp_pose_motion",
                "target_side": target.side,
                "trajectory": "collision_checked_joint_quintic_fk_then_local_tcp_correction",
            },
        )
        recovery = _dispatch_frozen_reference_with_bounded_ik_recovery(
            control,
            reference,
            inactive_side=inactive_side,
            retry_limit=retry_limit,
            settle_once=settle_reference_interval,
        )
        for result_status in recovery["result_statuses"]:
            status = str(result_status)
            ik_status_counts[status] = int(ik_status_counts.get(status, 0)) + 1
        observations = tuple(recovery["observations"])
        for observation in observations:
            endpoint_status = str(observation.get("status", "unknown"))
            endpoint_status_counts[endpoint_status] = int(
                endpoint_status_counts.get(endpoint_status, 0)
            ) + 1
        retry_count = int(recovery["retry_count"])
        dispatched_reference_count += 1
        recovery_retry_count += retry_count
        recovery_physics_ticks += int(recovery["retry_physics_ticks"])
        execution_physics_ticks += int(recovery["physics_ticks"])
        if bool(recovery["recovered"]):
            recovered_reference_count += 1
        if bool(recovery["unrecovered"]):
            unrecovered_reference_count += 1
        if bool(recovery["initial_ik_target_unmet"]):
            recovery_records.append(
                {
                    "reference_index": index,
                    "reference_sequence": int(reference.sequence),
                    "status": str(recovery["status"]),
                    "retry_count": retry_count,
                    "attempt_count": int(recovery["attempt_count"]),
                    "result_statuses": [
                        str(value) for value in recovery["result_statuses"]
                    ],
                    "endpoint_statuses": [
                        str(value.get("status", "unknown")) for value in observations
                    ],
                }
            )
        qpos = {
            side: control.scene.get_arm_qpos(side).copy() for side in SIDES
        }
        actual = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"], right_qpos=qpos["right"]
        )[target.side]
        reference_world = control.world_from_desk @ active_pose_desk
        maximum_tracking_error_m = max(
            maximum_tracking_error_m,
            float(np.linalg.norm(actual[:3, 3] - reference_world[:3, 3])),
        )
        actual_qpos_by_frame.append(qpos)
        actual_pose_world.append(actual.copy())
        reference_pose_world.append(reference_world.copy())
        if bool(recovery["unrecovered"]):
            break

    endpoint = control.endpoint_tracking_observation()
    final_status = str(endpoint.get("status", "unknown"))
    # A final settled observation can clear only a miss that the bounded
    # same-reference loop explicitly recovered.  An exhausted reference remains
    # an execution failure even if a later observation happens to look settled.
    motion_status = classify_motion_status(
        final_endpoint_status=final_status,
        endpoint_status_counts=endpoint_status_counts,
        unrecovered_ik_target_unmet_count=unrecovered_reference_count,
    )
    maximum_joint_motion_deg = max(
        float(
            np.max(
                np.abs(
                    np.rad2deg(frame[target.side] - initial_qpos[target.side])
                )
            )
        )
        for frame in actual_qpos_by_frame
    )
    summary = {
        **empty_summary,
        "motion_emitted": True,
        "trajectory": {
            "planner": "collision_checked_joint_quintic_fk_then_local_tcp_correction",
            "joint_fk_path": dict(joint_fk_plan.metadata),
            "terminal_tcp_correction": correction_metadata,
            "total_reference_duration_sec": (
                float(max(0, dispatched_reference_count - 1))
                / float(control.contract.timing.ik_hz)
            ),
            "planned_reference_duration_sec": (
                float(len(active_poses_desk) - 1)
                / float(control.contract.timing.ik_hz)
            ),
            "execution_physics_duration_sec": (
                float(execution_physics_ticks)
                / float(control.contract.timing.physics_hz)
            ),
        },
        "terminal_hold_sec": float(terminal_hold_sec),
        "reference_frame_count": dispatched_reference_count,
        "planned_reference_frame_count": len(active_poses_desk),
        "dispatched_reference_count": dispatched_reference_count,
        "ik_status_counts": ik_status_counts,
        "endpoint_status_counts": endpoint_status_counts,
        "ik_target_unmet_recovery": {
            "semantics": "bounded_same_frozen_reference_retry_v1",
            "retry_limit_per_reference": retry_limit,
            "retry_count": recovery_retry_count,
            "retry_physics_ticks": recovery_physics_ticks,
            "transient_recovered_reference_count": recovered_reference_count,
            "unrecovered_reference_count": unrecovered_reference_count,
            "records": recovery_records,
        },
        "final_endpoint_tracking": endpoint,
        "maximum_actual_joint_motion_deg": maximum_joint_motion_deg,
        "maximum_translation_tracking_error_m": maximum_tracking_error_m,
        "target_pose_world": target_world_active.tolist(),
    }
    return GraspMotionPlayback(
        motion_status=motion_status,
        side=target.side,
        target=target,
        endpoint_receipt=receipt,
        frame_binding=binding,
        target_pose_world_by_side=target_world_by_side,
        reference_pose_world=tuple(reference_pose_world),
        actual_pose_world=tuple(actual_pose_world),
        actual_qpos_by_frame=tuple(actual_qpos_by_frame),
        summary=summary,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "MOTION_STATUS_COMPLETED",
    "MOTION_STATUS_ENDPOINT_REJECTED",
    "MOTION_STATUS_IK_TARGET_UNMET",
    "MOTION_STATUS_PATH_COLLISION_BLOCKED",
    "MOTION_STATUS_TRACKING_INCOMPLETE",
    "GraspMotionPlayback",
    "GraspMotionTarget",
    "JointFkReferencePlan",
    "ToolTcpFrameBinding",
    "build_collision_checked_joint_fk_reference_plan",
    "build_smooth_se3_trajectory",
    "classify_motion_status",
    "derive_tool_tcp_frame_binding",
    "evaluate_motion_target_endpoint",
    "execute_grasp_motion",
    "pose_from_xyz_quat_xyzw",
    "pose_from_xyz_rpy_deg",
    "quintic_progress",
    "required_motion_duration_sec",
    "select_random_reachable_demo_target",
]
