"""Execute one registered joint trajectory through the MuJoCo qpos owner.

The executor consumes only immutable shadow contracts.  It collision-checks
every registered point with the measured gripper geometry, publishes the exact
dual-arm qpos references on their relative-time lattice, waits for measured
terminal convergence, and then confirms a lower-owned terminal HOLD.  No
result from this module grants real-hardware motion authority.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.path_planning import JointPathPlannerConfig, JointPathStatus
from robot_motion.rm65 import load_rm65_tool_profile, tool_tcp_pose_flange_m_rad
from robot_motion.rm65.grasp_reachability import GraspReachabilityStatus
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
)

_SIDES = ("left", "right")
_EXECUTION_EVIDENCE_HASH_DOMAIN = (
    "agenticrobot.shadow_execution.mujoco_exact_qpos_evidence.v1"
)
_COMPATIBILITY_REASON_REF_PREFIX = "mujoco-shadow-compatibility-reason:"


@dataclass(frozen=True)
class RealTcpObstacleTrajectoryRequest:
    """One DMTac TCP goal and immutable cube scene for simulation-first planning."""

    side: str
    target_pose_base_from_dmtac_tcp: np.ndarray
    obstacle_scene: ObstacleSceneSnapshot
    source: str = "real_tcp_obstacle_acceptance"
    known_qpos_rad: np.ndarray | None = None
    linear_speed_m_s: float = 0.12
    angular_speed_deg_s: float = 45.0
    minimum_duration_sec: float = 2.0

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in _SIDES:
            raise ValueError("side must be left or right.")
        pose = np.asarray(self.target_pose_base_from_dmtac_tcp, dtype=float).copy()
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("target_pose_base_from_dmtac_tcp must be a finite 4x4 pose.")
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("target pose must have a homogeneous final row.")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise ValueError("target pose rotation must be proper and orthonormal.")
        if not isinstance(self.obstacle_scene, ObstacleSceneSnapshot):
            raise TypeError("obstacle_scene must be ObstacleSceneSnapshot.")
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must be non-empty.")
        known = self.known_qpos_rad
        if known is not None:
            known = np.asarray(known, dtype=float).reshape(-1).copy()
            if known.shape != (6,) or not np.all(np.isfinite(known)):
                raise ValueError("known_qpos_rad must contain six finite values.")
            known.setflags(write=False)
        for name in (
            "linear_speed_m_s",
            "angular_speed_deg_s",
            "minimum_duration_sec",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        pose.setflags(write=False)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "target_pose_base_from_dmtac_tcp", pose)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "known_qpos_rad", known)


class MujocoShadowTrajectoryPreparationError(RuntimeError):
    """Typed endpoint/path rejection from concrete MuJoCo preparation."""

    def __init__(
        self,
        reason_code: str,
        *,
        additional_reason_codes: tuple[str, ...] = (),
        details: Mapping[str, object] | None = None,
    ) -> None:
        reasons = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (reason_code, *additional_reason_codes)
                if str(value).strip()
            )
        )
        if not reasons:
            raise ValueError("trajectory preparation error requires a reason code.")
        self.reason_code = reasons[0]
        self.reason_codes = reasons
        self.details = dict(details or {})
        super().__init__(self.reason_code)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _refs(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    refs = tuple(sorted({str(value).strip() for value in values}))
    if not refs or any(not value for value in refs):
        raise ValueError(f"{name} requires non-empty evidence refs.")
    return refs


def build_obstacle_bound_shadow_control(
    snapshot: AuthoritativeRobotStateSnapshot,
    obstacle_scene: ObstacleSceneSnapshot,
) -> object:
    """Cold-start MuJoCo from the real cut and bind the exact immutable scene."""

    from .arbitrary_grasp_pose_motion import (
        build_obstacle_collision_oracle,
        build_shadow_initialized_control,
    )

    control = build_shadow_initialized_control(snapshot)
    try:
        control.shadow_obstacle_oracle = build_obstacle_collision_oracle(
            control,
            obstacle_scene,
        )
        return control
    except BaseException:
        close = getattr(control, "close", None)
        if callable(close):
            close()
        raise


def _path_preparation_reason(status: JointPathStatus) -> str:
    return {
        JointPathStatus.START_BLOCKED: "path_start_collision_blocked",
        JointPathStatus.TARGET_BLOCKED: "path_target_collision_blocked",
        JointPathStatus.PATH_SEARCH_EXHAUSTED: "path_search_exhausted",
        JointPathStatus.COLLISION_CHECK_UNAVAILABLE: (
            "path_collision_check_unavailable"
        ),
        JointPathStatus.INVALID_REQUEST: "path_planner_invalid_request",
    }.get(status, "path_planner_error")


def _selected_reachable_candidate_qpos(endpoint: object) -> np.ndarray:
    selected_index = endpoint.selected_candidate_index
    if selected_index is None:
        raise MujocoShadowTrajectoryPreparationError(
            "reachable_endpoint_candidate_missing"
        )
    for candidate in endpoint.candidates:
        if int(candidate.get("candidate_index", -1)) != int(selected_index):
            continue
        qpos = np.asarray(candidate.get("qpos_rad"), dtype=float).reshape(-1)
        if qpos.shape == (6,) and np.all(np.isfinite(qpos)):
            return qpos.copy()
        break
    raise MujocoShadowTrajectoryPreparationError(
        "reachable_endpoint_candidate_invalid"
    )


def build_registered_tcp_obstacle_trajectory(
    snapshot: AuthoritativeRobotStateSnapshot,
    request: RealTcpObstacleTrajectoryRequest,
    *,
    reachability_config: object | None = None,
    planner_config: JointPathPlannerConfig | None = None,
    control_factory: (
        Callable[[AuthoritativeRobotStateSnapshot, ObstacleSceneSnapshot], object]
        | None
    ) = None,
    wall_clock: Callable[[], float] = time.time,
) -> RegisteredShadowJointTrajectory:
    """Plan a real TCP request with Sim endpoint and obstacle semantics."""

    from .arbitrary_grasp_pose_motion import (
        GraspMotionTarget,
        _MujocoFclObstacleOracle,
        build_collision_checked_multiwaypoint_joint_fk_reference_plan,
        evaluate_motion_target_endpoint,
        plan_obstacle_aware_joint_path,
        revalidate_obstacle_path,
    )
    from ..rm65.random_grasp_pose_reachability import (
        RM65RandomGraspPoseReachabilityConfig,
    )

    if not isinstance(snapshot, AuthoritativeRobotStateSnapshot):
        raise TypeError("snapshot must be AuthoritativeRobotStateSnapshot.")
    if not isinstance(request, RealTcpObstacleTrajectoryRequest):
        raise TypeError("request must be RealTcpObstacleTrajectoryRequest.")
    if not callable(wall_clock):
        raise TypeError("wall_clock must be callable.")
    path_config = planner_config or JointPathPlannerConfig()
    if not isinstance(path_config, JointPathPlannerConfig):
        raise TypeError("planner_config must be JointPathPlannerConfig.")
    endpoint_config = reachability_config or RM65RandomGraspPoseReachabilityConfig()
    if not isinstance(endpoint_config, RM65RandomGraspPoseReachabilityConfig):
        raise TypeError(
            "reachability_config must be RM65RandomGraspPoseReachabilityConfig."
        )
    factory = control_factory or build_obstacle_bound_shadow_control
    if not callable(factory):
        raise TypeError("control_factory must be callable.")

    target = GraspMotionTarget(
        side=request.side,
        target_pose_base_from_dmtac_tcp=request.target_pose_base_from_dmtac_tcp,
        source=request.source,
        known_qpos_rad=request.known_qpos_rad,
        metadata={
            "obstacle_scene_hash": request.obstacle_scene.semantic_content_hash,
            "planning_origin": "authoritative_real_state_cut",
        },
    )
    target_hash = canonical_content_hash(
        {
            "side": request.side,
            "target_pose_base_from_dmtac_tcp": (
                request.target_pose_base_from_dmtac_tcp
            ),
            "source": request.source,
            "known_qpos_rad": request.known_qpos_rad,
        },
        domain="agenticrobot.real_shadow_acceptance.tcp_target.v1",
    )
    planner_config_hash = canonical_content_hash(
        {
            "endpoint_reachability": asdict(endpoint_config),
            "joint_path": asdict(path_config),
            "linear_speed_m_s": request.linear_speed_m_s,
            "angular_speed_deg_s": request.angular_speed_deg_s,
            "minimum_duration_sec": request.minimum_duration_sec,
            "path_strategy": "direct_then_deterministic_birrt",
            "time_parameterization": "per_segment_quintic_stop_at_waypoint",
        },
        domain="agenticrobot.real_shadow_acceptance.planner_config.v1",
    )

    control = None
    try:
        try:
            control = factory(snapshot, request.obstacle_scene)
            fcl_oracle = control.shadow_obstacle_oracle
            path_oracle = _MujocoFclObstacleOracle(control, fcl_oracle)
        except Exception as exc:
            raise MujocoShadowTrajectoryPreparationError(
                "planning_simulation_initialization_failed",
                additional_reason_codes=(type(exc).__name__,),
                details={"exception": str(exc)},
            ) from exc

        try:
            endpoint, _binding, solver = evaluate_motion_target_endpoint(
                control,
                target,
                config=endpoint_config,
                obstacle_oracle=fcl_oracle,
            )
        except Exception as exc:
            raise MujocoShadowTrajectoryPreparationError(
                "endpoint_evaluation_failed",
                additional_reason_codes=(type(exc).__name__,),
                details={"exception": str(exc)},
            ) from exc
        if endpoint.status is not GraspReachabilityStatus.REACHABLE:
            raise MujocoShadowTrajectoryPreparationError(
                endpoint.status.value,
                additional_reason_codes=(endpoint.reason_code,),
                details={"endpoint_receipt": endpoint.to_dict()},
            )

        selected_qpos = _selected_reachable_candidate_qpos(endpoint)
        path = plan_obstacle_aware_joint_path(
            control,
            side=request.side,
            target_qpos_rad=selected_qpos,
            single_arm_solver=solver,
            obstacle_oracle=path_oracle,
            planner_config=path_config,
        )
        if path.status is not JointPathStatus.PATH_FOUND:
            raise MujocoShadowTrajectoryPreparationError(
                _path_preparation_reason(path.status),
                additional_reason_codes=(path.reason or path.status.value,),
                details={"path_receipt": path.to_dict()},
            )
        final_check = revalidate_obstacle_path(path.waypoints, oracle=path_oracle)
        if not bool(final_check.get("safe", False)):
            raise MujocoShadowTrajectoryPreparationError(
                str(final_check.get("reason_code", "final_path_revalidation_failed")),
                details={"final_revalidation": final_check},
            )

        reference = build_collision_checked_multiwaypoint_joint_fk_reference_plan(
            control,
            side=request.side,
            waypoints_qpos_rad=tuple(
                waypoint[request.side] for waypoint in path.waypoints
            ),
            linear_speed_m_s=request.linear_speed_m_s,
            angular_speed_deg_s=request.angular_speed_deg_s,
            minimum_duration_sec=request.minimum_duration_sec,
        )
        if reference.status != "safe":
            raise MujocoShadowTrajectoryPreparationError(
                "path_collision_blocked",
                additional_reason_codes=(
                    str(
                        reference.metadata.get(
                            "reason_code",
                            "mujoco_time_parameterized_path_collision_blocked",
                        )
                    ),
                ),
                details={"reference_plan": dict(reference.metadata)},
            )
        reference_hz = float(control.contract.timing.ik_hz)
        if not np.isfinite(reference_hz) or reference_hz <= 0.0:
            raise MujocoShadowTrajectoryPreparationError(
                "invalid_reference_rate",
                details={"reference_hz": reference_hz},
            )
        inactive_side = "right" if request.side == "left" else "left"
        points = tuple(
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=float(index) / reference_hz,
                qpos_by_side={
                    request.side: qpos,
                    inactive_side: snapshot.qpos_by_side[inactive_side],
                },
            )
            for index, qpos in enumerate(reference.qpos_rad)
        )
        if not points or any(
            not np.array_equal(points[0].qpos_by_side[side], snapshot.qpos_by_side[side])
            for side in _SIDES
        ):
            raise MujocoShadowTrajectoryPreparationError(
                "registered_trajectory_start_mismatch"
            )
        created_at = max(float(wall_clock()), snapshot.available_at)
        trajectory_seed_hash = canonical_content_hash(
            {
                "snapshot_hash": snapshot.semantic_content_hash,
                "scene_hash": request.obstacle_scene.semantic_content_hash,
                "target_hash": target_hash,
                "planner_config_hash": planner_config_hash,
                "path_receipt_hash": path.semantic_content_hash,
            },
            domain="agenticrobot.real_shadow_acceptance.trajectory_id.v1",
        )
        return RegisteredShadowJointTrajectory(
            trajectory_id="real-tcp-obstacle:" + trajectory_seed_hash[:24],
            created_at=created_at,
            initial_snapshot=snapshot,
            scene_hash=request.obstacle_scene.semantic_content_hash,
            target_hash=target_hash,
            planner_config_hash=planner_config_hash,
            moving_sides=(request.side,),
            points=points,
        )
    finally:
        if control is not None:
            close = getattr(control, "close", None)
            if callable(close):
                close()


def _side_text_mapping(values: Mapping[str, str], *, name: str):
    result = {side: str(values[side]).strip() for side in _SIDES}
    if any(not value for value in result.values()) or set(values) != set(_SIDES):
        raise ValueError(f"{name} must map exactly left and right to non-empty values.")
    return MappingProxyType(result)


def _side_sequence_mapping(
    values: Mapping[str, Sequence[object]],
    *,
    name: str,
    length: int,
    numeric: bool,
):
    if set(values) != set(_SIDES):
        raise ValueError(f"{name} must map exactly left and right.")
    result = {}
    for side in _SIDES:
        if numeric:
            sequence = tuple(
                _finite(value, name=f"{name}.{side}") for value in values[side]
            )
        else:
            sequence = tuple(str(value).strip() for value in values[side])
        if len(sequence) != length or any(value == "" for value in sequence):
            raise ValueError(f"{name}.{side} must contain {length} values.")
        result[side] = sequence
    return MappingProxyType(result)


@dataclass(frozen=True)
class MujocoShadowCompatibilityExpectation:
    """Small real-owner contract that MuJoCo observes once before execution.

    These are the values that materially affect joint-trajectory replay. This
    deliberately is not a URDF/mesh/artifact hash attestation and it contains
    no caller-supplied ``verified`` flag.
    """

    station_revision: str
    reference_frame_id: str
    desk_from_arm_base_by_side: Mapping[str, Sequence[float]]
    joint_names_by_side: Mapping[str, Sequence[str]]
    tool_profile_id_by_side: Mapping[str, str]
    tool_tcp_pose_flange_m_rad_by_side: Mapping[str, Sequence[float]]
    gripper_profile_name: str
    gripper_side_by_name: Mapping[str, str]
    scene_revision: str
    simulation_scene_profile_name: str
    collision_profile_name: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "station_revision",
            "reference_frame_id",
            "gripper_profile_name",
            "scene_revision",
            "simulation_scene_profile_name",
            "collision_profile_name",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "desk_from_arm_base_by_side",
            _side_sequence_mapping(
                self.desk_from_arm_base_by_side,
                name="desk_from_arm_base_by_side",
                length=16,
                numeric=True,
            ),
        )
        object.__setattr__(
            self,
            "joint_names_by_side",
            _side_sequence_mapping(
                self.joint_names_by_side,
                name="joint_names_by_side",
                length=6,
                numeric=False,
            ),
        )
        object.__setattr__(
            self,
            "tool_profile_id_by_side",
            _side_text_mapping(
                self.tool_profile_id_by_side,
                name="tool_profile_id_by_side",
            ),
        )
        object.__setattr__(
            self,
            "tool_tcp_pose_flange_m_rad_by_side",
            _side_sequence_mapping(
                self.tool_tcp_pose_flange_m_rad_by_side,
                name="tool_tcp_pose_flange_m_rad_by_side",
                length=6,
                numeric=True,
            ),
        )
        side_by_name = {
            str(name).strip(): str(side).strip()
            for name, side in self.gripper_side_by_name.items()
        }
        if (
            len(side_by_name) != len(_SIDES)
            or any(not name for name in side_by_name)
            or set(side_by_name.values()) != set(_SIDES)
        ):
            raise ValueError(
                "gripper_side_by_name must map non-empty grippers to left and right."
            )
        object.__setattr__(
            self,
            "gripper_side_by_name",
            MappingProxyType(side_by_name),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _refs(self.evidence_refs, name="compatibility expectation"),
        )
        if any(
            ref.startswith(_COMPATIBILITY_REASON_REF_PREFIX)
            for ref in self.evidence_refs
        ):
            raise ValueError(
                "compatibility expectation evidence cannot use the executor-owned "
                "reason namespace."
            )


# Compatibility name for the application orchestration surface. The fields
# now describe an expectation, not a caller-asserted model identity.
MujocoShadowModelBinding = MujocoShadowCompatibilityExpectation


def _tool_profile_id(profile: Mapping[str, object], *, side: str) -> str:
    name = str(profile.get("name", profile.get("tool_name", ""))).strip()
    arm = str(profile.get("arm", "")).strip()
    if not name or arm != side:
        raise ValueError(f"{side} tool profile must identify its name and arm.")
    return f"{name}:{arm}"


def _observe_mujoco_shadow_compatibility(control) -> dict[str, object]:
    station = control.station_pose_contract
    source = station.source
    scene = control.scene
    rig_config = control.rig.config
    arm_profile = rig_config.actuation_profile
    gripper_profile = rig_config.gripper_actuation_profile
    retargeter = control.retargeter

    station_revision = str(station.source_id).strip()
    reference_frame_id = str(station.canonical_absolute_frame).strip()
    if not station_revision or not reference_frame_id:
        raise ValueError("station revision and reference frame must be non-empty.")
    joint_names = dict(
        _side_sequence_mapping(
            {
                side: scene.arm_specs[side].joint_names for side in _SIDES
            },
            name="observed_joint_names_by_side",
            length=6,
            numeric=False,
        )
    )
    actuation_joint_names = dict(
        _side_sequence_mapping(
            {
                side: arm_profile.joint_names_by_side[side] for side in _SIDES
            },
            name="observed_actuation_joint_names_by_side",
            length=6,
            numeric=False,
        )
    )
    scene_profiles = (
        str(control.contract.scene.profile_name).strip(),
        str(scene.scene_spec.profile_name).strip(),
        str(scene.conversion.scene_profile_name).strip(),
    )
    if any(not value for value in scene_profiles):
        raise ValueError("MuJoCo scene profiles must be non-empty.")
    tool_profiles = {
        side: load_rm65_tool_profile(
            getattr(control.config, f"{side}_tool_profile"), side=side
        )
        for side in _SIDES
    }
    tool_profile_tcp = dict(
        _side_sequence_mapping(
            {
                side: tool_tcp_pose_flange_m_rad(tool_profiles[side])
                for side in _SIDES
            },
            name="observed_tool_profile_tcp_pose_flange_m_rad_by_side",
            length=6,
            numeric=True,
        )
    )
    obstacle_oracle = control.shadow_obstacle_oracle
    shadow_scene_revision = str(obstacle_oracle.scene_hash).strip()
    if not shadow_scene_revision:
        raise ValueError("shadow obstacle oracle scene revision must be non-empty.")
    if not bool(obstacle_oracle.collision_model_is_independent):
        raise ValueError("shadow obstacle oracle must own an independent collision model.")
    return {
        "station_revision": station_revision,
        "reference_frame_id": reference_frame_id,
        "desk_from_arm_base_by_side": dict(
            _side_sequence_mapping(
                {
                    side: np.asarray(
                        source.effective_raw_to_canonical_desk_transform(side),
                        dtype=float,
                    ).reshape(-1)
                    for side in _SIDES
                },
                name="observed_desk_from_arm_base_by_side",
                length=16,
                numeric=True,
            )
        ),
        "joint_names_by_side": joint_names,
        "actuation_joint_names_by_side": actuation_joint_names,
        "tool_profile_id_by_side": {
            side: _tool_profile_id(tool_profiles[side], side=side)
            for side in _SIDES
        },
        "tool_profile_tcp_pose_flange_m_rad_by_side": tool_profile_tcp,
        "tool_tcp_pose_flange_m_rad_by_side": dict(
            _side_sequence_mapping(
                {
                    side: np.asarray(
                        getattr(
                            retargeter.config,
                            f"{side}_tool_tcp_pose_flange_m_rad",
                        ),
                        dtype=float,
                    ).reshape(-1)
                    for side in _SIDES
                },
                name="observed_tool_tcp_pose_flange_m_rad_by_side",
                length=6,
                numeric=True,
            )
        ),
        "gripper_profile_name": str(gripper_profile.name),
        "gripper_side_by_name": dict(gripper_profile.side_by_gripper),
        "scene_profile_names": scene_profiles,
        "shadow_scene_revision": shadow_scene_revision,
        "collision_profile_name": str(retargeter.profile.name),
        "collision_projector_available": scene.collision_projector() is not None,
    }


def verify_mujoco_shadow_compatibility(
    control,
    trajectory: RegisteredShadowJointTrajectory,
    expectation: MujocoShadowCompatibilityExpectation,
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Observe and compare the small replay-critical contract exactly once."""

    reasons: list[str] = []
    try:
        observed = _observe_mujoco_shadow_compatibility(control)
    except Exception as exc:  # noqa: BLE001 - fail closed with a typed reason.
        observed = {
            "observation_error_type": type(exc).__name__,
            "observation_error": str(exc),
        }
        reasons.append("compatibility_observation_failed")
    else:
        if observed["station_revision"] != expectation.station_revision:
            reasons.append("station_revision_mismatch")
        if observed["reference_frame_id"] != expectation.reference_frame_id:
            reasons.append("reference_frame_mismatch")
        if any(
            not np.allclose(
                observed["desk_from_arm_base_by_side"][side],
                expectation.desk_from_arm_base_by_side[side],
                atol=1e-9,
                rtol=0.0,
            )
            for side in _SIDES
        ):
            reasons.append("reference_frame_spec_mismatch")
        if (
            observed["joint_names_by_side"]
            != dict(expectation.joint_names_by_side)
            or observed["actuation_joint_names_by_side"]
            != observed["joint_names_by_side"]
        ):
            reasons.append("joint_layout_mismatch")
        if observed["tool_profile_id_by_side"] != dict(
            expectation.tool_profile_id_by_side
        ):
            reasons.append("tool_profile_mismatch")
        if any(
            not np.allclose(
                observed["tool_tcp_pose_flange_m_rad_by_side"][side],
                expectation.tool_tcp_pose_flange_m_rad_by_side[side],
                atol=1e-9,
                rtol=0.0,
            )
            for side in _SIDES
        ):
            reasons.append("tool_tcp_mismatch")
        if any(
            not np.allclose(
                observed["tool_profile_tcp_pose_flange_m_rad_by_side"][side],
                observed["tool_tcp_pose_flange_m_rad_by_side"][side],
                atol=1e-9,
                rtol=0.0,
            )
            for side in _SIDES
        ):
            reasons.append("tool_profile_runtime_tcp_mismatch")
        if observed["gripper_profile_name"] != expectation.gripper_profile_name:
            reasons.append("gripper_profile_mismatch")
        if observed["gripper_side_by_name"] != dict(
            expectation.gripper_side_by_name
        ):
            reasons.append("gripper_assignment_mismatch")
        scene_profiles = observed["scene_profile_names"]
        if (
            len(set(scene_profiles)) != 1
            or scene_profiles[0] != expectation.simulation_scene_profile_name
        ):
            reasons.append("simulation_scene_profile_mismatch")
        if observed["collision_profile_name"] != expectation.collision_profile_name:
            reasons.append("collision_profile_mismatch")
        if not observed["collision_projector_available"]:
            reasons.append("collision_projector_missing")
        if observed["shadow_scene_revision"] != expectation.scene_revision:
            reasons.append("simulation_scene_revision_mismatch")
        if trajectory.scene_hash != expectation.scene_revision:
            reasons.append("trajectory_scene_revision_mismatch")

    unique_reasons = tuple(dict.fromkeys(reasons))
    evidence_payload = {
        "compatible": not unique_reasons,
        "reason_codes": unique_reasons,
        "observed": observed,
    }
    reason_label = "compatible" if not unique_reasons else "+".join(unique_reasons)
    summary_ref = (
        f"mujoco-shadow-compatibility:{reason_label}:"
        f"{canonical_content_hash(evidence_payload, domain='agenticrobot.shadow_execution.mujoco_compatibility.v1')}"
    )
    reason_refs = tuple(
        f"{_COMPATIBILITY_REASON_REF_PREFIX}{reason}" for reason in unique_reasons
    )
    return not unique_reasons, unique_reasons, (*reason_refs, summary_ref)


@dataclass(frozen=True)
class MujocoExactQposExecutionConfig:
    """Offline execution limits; none are accepted from an Agent request."""

    start_tolerance_rad: float = 1e-9
    tracking_tolerance_rad: float = 2e-3
    gripper_hold_tolerance: float = 2e-3
    final_settle_timeout_sec: float = 4.0
    terminal_hold_timeout_sec: float = 2.0
    terminal_hold_dwell_sec: float = 0.08
    terminal_velocity_tolerance_rad_s: float = 2e-3
    collision_sample_max_joint_step_rad: float = math.radians(0.1)

    def __post_init__(self) -> None:
        for name in (
            "start_tolerance_rad",
            "tracking_tolerance_rad",
            "gripper_hold_tolerance",
            "final_settle_timeout_sec",
            "terminal_hold_timeout_sec",
            "terminal_hold_dwell_sec",
            "terminal_velocity_tolerance_rad_s",
            "collision_sample_max_joint_step_rad",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.tracking_tolerance_rad <= 0.0:
            raise ValueError("tracking_tolerance_rad must be positive.")
        if self.final_settle_timeout_sec <= 0.0:
            raise ValueError("final_settle_timeout_sec must be positive.")
        if self.terminal_hold_timeout_sec <= 0.0:
            raise ValueError("terminal_hold_timeout_sec must be positive.")
        if self.collision_sample_max_joint_step_rad <= 0.0:
            raise ValueError("collision_sample_max_joint_step_rad must be positive.")


def _gripper_close_by_name(control, trajectory: RegisteredShadowJointTrajectory):
    profile = getattr(control.rig.config, "gripper_actuation_profile", None)
    if profile is None:
        raise RuntimeError("exact shadow execution requires a gripper model.")
    result = {
        name: float(trajectory.initial_snapshot.gripper_close_by_side[side])
        for name, side in profile.side_by_gripper.items()
    }
    if set(profile.side_by_gripper.values()) != set(_SIDES):
        raise RuntimeError("gripper model must map exactly left and right.")
    return result


def _gripper_close_by_side(control) -> dict[str, float]:
    profile = getattr(control.rig.config, "gripper_actuation_profile", None)
    if profile is None:
        raise RuntimeError("exact shadow execution requires a gripper model.")
    return {
        side: float(control.scene.get_gripper_close(name))
        for name, side in profile.side_by_gripper.items()
    }


def _maximum_qpos_error(
    observed: Mapping[str, Sequence[float]],
    expected: Mapping[str, Sequence[float]],
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(observed[side], dtype=float)
                    - np.asarray(expected[side], dtype=float)
                )
            )
        )
        for side in _SIDES
    )


def _maximum_gripper_error(control, trajectory) -> float:
    observed = _gripper_close_by_side(control)
    return max(
        abs(
            observed[side]
            - float(trajectory.initial_snapshot.gripper_close_by_side[side])
        )
        for side in _SIDES
    )


def _point_command(
    point: RegisteredJointTrajectoryPoint,
    trajectory: RegisteredShadowJointTrajectory,
    *,
    timestamp: float,
) -> TeleopCommand:
    command = TeleopCommand(timestamp=float(timestamp))
    for side in _SIDES:
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=point.qpos_by_side[side],
                metadata={
                    "source": "registered_shadow_exact_qpos_v1",
                    "trajectory_hash": trajectory.semantic_content_hash,
                    "point_hash": point.semantic_content_hash,
                    "time_from_start_sec": point.time_from_start_sec,
                },
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="gripper_scalar",
                value=np.asarray(
                    [trajectory.initial_snapshot.gripper_close_by_side[side]],
                    dtype=float,
                ),
                metadata={"source": "registered_shadow_hold_measured_v1"},
            )
        )
    return command


def _preflight_points(control, trajectory, gripper_close_by_name, config):
    projector = control.scene.collision_projector()
    if projector is None:
        raise RuntimeError("exact shadow execution requires MuJoCo collision safety.")
    obstacle_oracle = getattr(control, "shadow_obstacle_oracle", None)
    if obstacle_oracle is None:
        raise RuntimeError("exact shadow execution requires a bound obstacle oracle.")
    if str(obstacle_oracle.scene_hash).strip() != trajectory.scene_hash:
        raise RuntimeError("obstacle oracle scene revision changed after compatibility.")
    base = control.scene.kinematic_controlled_qpos()
    minimum_clearance: float | None = None
    def check(qpos_by_side, *, point_index, segment_index, segment_fraction):
        nonlocal minimum_clearance
        target = control.scene.kinematic_target_vector(
            base,
            qpos_by_side,
            gripper_close_by_name=gripper_close_by_name,
        )
        collision = projector.check_qpos(control.scene.data, target)
        payload = collision.to_dict()
        obstacle = obstacle_oracle.check(qpos_by_side)
        obstacle_payload = obstacle.to_dict()
        for raw_clearance in (
            payload.get("minimum_distance_m"),
            obstacle_payload.get("minimum_distance_m"),
        ):
            if raw_clearance is not None:
                clearance = float(raw_clearance)
                minimum_clearance = (
                    clearance
                    if minimum_clearance is None
                    else min(minimum_clearance, clearance)
                )
        collision_safe = bool(collision.safe)
        obstacle_safe = bool(
            obstacle.collision_check_available and obstacle.safe
        )
        if not collision_safe or not obstacle_safe:
            if point_index == 0:
                status = ShadowSimulationStatus.PATH_START_COLLISION_BLOCKED
            elif point_index == len(trajectory.points) - 1:
                status = ShadowSimulationStatus.PATH_TARGET_COLLISION_BLOCKED
            else:
                status = ShadowSimulationStatus.PATH_COLLISION_BLOCKED
            return (
                status,
                {
                    "point_index": point_index,
                    "segment_index": segment_index,
                    "segment_fraction": segment_fraction,
                },
                minimum_clearance,
                {
                    "mujoco": payload,
                    "obstacle_oracle": obstacle_payload,
                },
            )
        return None

    start = check(
        trajectory.points[0].qpos_by_side,
        point_index=0,
        segment_index=None,
        segment_fraction=0.0,
    )
    if start is not None:
        return start
    for segment_index, (left, right) in enumerate(pairwise(trajectory.points)):
        maximum_motion = max(
            float(
                np.max(
                    np.abs(right.qpos_by_side[side] - left.qpos_by_side[side])
                )
            )
            for side in _SIDES
        )
        sample_count = max(
            1,
            math.ceil(maximum_motion / config.collision_sample_max_joint_step_rad),
        )
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            qpos = {
                side: left.qpos_by_side[side]
                + fraction
                * (right.qpos_by_side[side] - left.qpos_by_side[side])
                for side in _SIDES
            }
            is_registered_endpoint = sample_index == sample_count
            point_index = segment_index + 1 if is_registered_endpoint else None
            result = check(
                qpos,
                point_index=point_index,
                segment_index=segment_index,
                segment_fraction=fraction,
            )
            if result is not None:
                # Only the exact final registered point is a target collision;
                # every between-point sample remains a path collision.
                if not is_registered_endpoint and result[0] is not ShadowSimulationStatus.PATH_COLLISION_BLOCKED:
                    result = (
                        ShadowSimulationStatus.PATH_COLLISION_BLOCKED,
                        result[1],
                        result[2],
                        result[3],
                    )
                return result
    return None, None, minimum_clearance, None


def _evidence_ref(kind: str, payload: Mapping[str, object]) -> str:
    digest = canonical_content_hash(
        dict(payload),
        domain=_EXECUTION_EVIDENCE_HASH_DOMAIN,
    )
    return f"mujoco-shadow-{kind}:{digest}"


def _terminal_hold(control, trajectory, config):
    control.rig.hold_motion("registered shadow trajectory terminal HOLD")
    timing = control.rig.config.timing
    dt = float(timing.physics_timestep_sec)
    max_ticks = max(1, math.ceil(config.terminal_hold_timeout_sec / dt))
    dwell_ticks = max(1, math.ceil(config.terminal_hold_dwell_sec / dt))
    consecutive = 0
    previous = control.rig.measured_qpos_by_side()
    last_observation: dict[str, object] = {}
    for tick in range(max_ticks):
        control.step_physics()
        measured = control.rig.measured_qpos_by_side()
        accepted = control.rig.accepted_qcmd_by_side()
        velocity = max(
            float(
                np.max(
                    np.abs(
                        (np.asarray(measured[side]) - np.asarray(previous[side])) / dt
                    )
                )
            )
            for side in _SIDES
        )
        tracking_error = _maximum_qpos_error(measured, accepted)
        gripper_error = _maximum_gripper_error(control, trajectory)
        ready = (
            velocity <= config.terminal_velocity_tolerance_rad_s
            and tracking_error <= config.tracking_tolerance_rad
            and gripper_error <= config.gripper_hold_tolerance
        )
        consecutive = consecutive + 1 if ready else 0
        last_observation = {
            "tick": tick + 1,
            "maximum_measured_velocity_rad_s": velocity,
            "maximum_accepted_tracking_error_rad": tracking_error,
            "maximum_gripper_hold_error": gripper_error,
            "required_dwell_ticks": dwell_ticks,
            "observed_dwell_ticks": consecutive,
        }
        previous = measured
        if consecutive >= dwell_ticks:
            return True, last_observation
    return False, last_observation


def execute_registered_shadow_trajectory(
    *,
    control,
    trajectory: RegisteredShadowJointTrajectory,
    model_binding: MujocoShadowModelBinding,
    config: MujocoExactQposExecutionConfig | None = None,
) -> ShadowSimulationReceipt:
    """Run exact qpos references in MuJoCo and return typed terminal evidence."""

    if not isinstance(trajectory, RegisteredShadowJointTrajectory):
        raise TypeError("trajectory must be RegisteredShadowJointTrajectory.")
    if not isinstance(model_binding, MujocoShadowCompatibilityExpectation):
        raise TypeError(
            "model_binding must be MujocoShadowCompatibilityExpectation."
        )
    cfg = config or MujocoExactQposExecutionConfig()
    if not isinstance(cfg, MujocoExactQposExecutionConfig):
        raise TypeError("config must be MujocoExactQposExecutionConfig.")
    for name in ("scene", "rig", "step_physics"):
        if not hasattr(control, name):
            raise TypeError(f"control must expose {name}.")

    evidence = list(model_binding.evidence_refs)
    compatibility_matches, _compatibility_reasons, compatibility_evidence = (
        verify_mujoco_shadow_compatibility(control, trajectory, model_binding)
    )
    evidence.extend(compatibility_evidence)
    if not compatibility_matches:
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.append(_evidence_ref("terminal-hold", hold_observation))
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH,
            executed_point_count=0,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=False,
            evidence_refs=tuple(evidence),
        )

    measured_start = control.rig.measured_qpos_by_side()
    accepted_start = control.rig.accepted_qcmd_by_side()
    start_arm_error = max(
        _maximum_qpos_error(measured_start, trajectory.initial_snapshot.qpos_by_side),
        _maximum_qpos_error(accepted_start, trajectory.initial_snapshot.qpos_by_side),
    )
    start_gripper_error = _maximum_gripper_error(control, trajectory)
    if (
        start_arm_error > cfg.start_tolerance_rad
        or start_gripper_error > cfg.gripper_hold_tolerance
    ):
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.extend(
            (
                _evidence_ref(
                    "start-mismatch",
                    {
                        "maximum_arm_error_rad": start_arm_error,
                        "maximum_gripper_error": start_gripper_error,
                    },
                ),
                _evidence_ref("terminal-hold", hold_observation),
            )
        )
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=ShadowSimulationStatus.TRAJECTORY_START_MISMATCH,
            executed_point_count=0,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=tuple(evidence),
        )

    gripper_close_by_name = _gripper_close_by_name(control, trajectory)
    blocked, blocked_location, minimum_clearance, collision = _preflight_points(
        control,
        trajectory,
        gripper_close_by_name,
        cfg,
    )
    if blocked is not None:
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.extend(
            (
                _evidence_ref(
                    "collision-blocked",
                    {"location": blocked_location, "collision": collision},
                ),
                _evidence_ref("terminal-hold", hold_observation),
            )
        )
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=blocked,
            executed_point_count=0,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=tuple(evidence),
            minimum_clearance_m=minimum_clearance,
        )

    timing = control.rig.config.timing
    dt = float(timing.physics_timestep_sec)
    started_at = float(control.scene.data.time)
    executed_count = 0
    for index, point in enumerate(trajectory.points):
        due = started_at + point.time_from_start_sec
        while float(control.scene.data.time) + 1e-12 < due:
            control.step_physics()
            if _maximum_gripper_error(control, trajectory) > cfg.gripper_hold_tolerance:
                held, hold_observation = _terminal_hold(control, trajectory, cfg)
                evidence.extend(
                    (
                        _evidence_ref("gripper-drift", {"point_index": index}),
                        _evidence_ref("terminal-hold", hold_observation),
                    )
                )
                return ShadowSimulationReceipt(
                    trajectory_hash=trajectory.semantic_content_hash,
                    initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
                    identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
                    scene_hash=trajectory.scene_hash,
                    status=ShadowSimulationStatus.GRIPPER_HOLD_DRIFT,
                    executed_point_count=executed_count,
                    terminal_hold_confirmed=held,
                    exact_planned_qpos_execution_confirmed=False,
                    model_identity_confirmed=True,
                    evidence_refs=tuple(evidence),
                    minimum_clearance_m=minimum_clearance,
                )
        receipts = control.rig.send_commands(
            _point_command(
                point,
                trajectory,
                timestamp=float(control.scene.data.time),
            )
        )
        required = {"left_arm", "right_arm", "left_hand", "right_hand"}
        if set(receipts) != required or any(
            not bool(receipts[name].accepted) for name in required
        ):
            raise RuntimeError("MuJoCo exact-qpos dispatch was not atomically accepted.")
        executed_count += 1
        evidence.append(
            _evidence_ref(
                "point",
                {
                    "point_index": index,
                    "point_hash": point.semantic_content_hash,
                    "published_sim_time": float(control.scene.data.time) - started_at,
                },
            )
        )
        # Consume the mailbox before another registered point can replace it.
        control.step_physics()

    final_qpos = trajectory.points[-1].qpos_by_side
    max_ticks = max(1, math.ceil(cfg.final_settle_timeout_sec / dt))
    final_observation: dict[str, object] = {}
    final_settled = False
    for tick in range(max_ticks):
        measured = control.rig.measured_qpos_by_side()
        accepted = control.rig.accepted_qcmd_by_side()
        measured_error = _maximum_qpos_error(measured, final_qpos)
        accepted_error = _maximum_qpos_error(accepted, final_qpos)
        gripper_error = _maximum_gripper_error(control, trajectory)
        final_observation = {
            "tick": tick,
            "maximum_measured_error_rad": measured_error,
            "maximum_accepted_error_rad": accepted_error,
            "maximum_gripper_hold_error": gripper_error,
        }
        if gripper_error > cfg.gripper_hold_tolerance:
            break
        if (
            measured_error <= cfg.tracking_tolerance_rad
            and accepted_error <= cfg.tracking_tolerance_rad
        ):
            final_settled = True
            break
        control.step_physics()

    if not final_settled:
        status = (
            ShadowSimulationStatus.GRIPPER_HOLD_DRIFT
            if final_observation.get("maximum_gripper_hold_error", 0.0)
            > cfg.gripper_hold_tolerance
            else ShadowSimulationStatus.POINT_TRACKING_TIMEOUT
        )
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.extend(
            (
                _evidence_ref("final-unsettled", final_observation),
                _evidence_ref("terminal-hold", hold_observation),
            )
        )
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=status,
            executed_point_count=executed_count,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=tuple(evidence),
            minimum_clearance_m=minimum_clearance,
        )

    held, hold_observation = _terminal_hold(control, trajectory, cfg)
    evidence.extend(
        (
            _evidence_ref("final-settled", final_observation),
            _evidence_ref("terminal-hold", hold_observation),
        )
    )
    status = (
        ShadowSimulationStatus.COMPLETED
        if held
        else ShadowSimulationStatus.TERMINAL_HOLD_FAILED
    )
    return ShadowSimulationReceipt(
        trajectory_hash=trajectory.semantic_content_hash,
        initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
        identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=status,
        executed_point_count=executed_count,
        terminal_hold_confirmed=held,
        exact_planned_qpos_execution_confirmed=True,
        model_identity_confirmed=True,
        evidence_refs=tuple(evidence),
        minimum_clearance_m=minimum_clearance,
    )


__all__ = [
    "MujocoExactQposExecutionConfig",
    "MujocoShadowCompatibilityExpectation",
    "MujocoShadowModelBinding",
    "MujocoShadowTrajectoryPreparationError",
    "RealTcpObstacleTrajectoryRequest",
    "build_obstacle_bound_shadow_control",
    "build_registered_tcp_obstacle_trajectory",
    "execute_registered_shadow_trajectory",
    "verify_mujoco_shadow_compatibility",
]
