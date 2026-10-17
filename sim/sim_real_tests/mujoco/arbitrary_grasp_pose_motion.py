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

import ast
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathPlanResult,
    JointPathStatus,
    plan_collision_aware_joint_path,
)
from robot_motion.position_control import ResampledMotionReference
from robot_motion.rm65 import DEFAULT_GRASP_PATH_COLLISION_SCENE
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
from robot_motion.rm65.obstacle_collision import (
    RM65_D405_ATTACHED_COLLISION_PROXIES,
    RM65ObstacleCollisionOracle,
    build_rm65_profile_obstacle_collision_oracle,
)
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredShadowJointTrajectory,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
)
from sim.mujoco.collision_safety import (
    CollisionCheckResult,
    KinematicCollisionProjector,
)
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_SIM_HOME_QPOS_BY_SIDE,
    DeskTcpPositionControl,
    DeskTcpPositionControlConfig,
)
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
MOTION_STATUS_PATH_START_BLOCKED = "path_start_collision_blocked"
MOTION_STATUS_PATH_TARGET_BLOCKED = "path_target_collision_blocked"
MOTION_STATUS_PATH_SEARCH_EXHAUSTED = "path_search_exhausted"
MOTION_STATUS_PATH_PLANNER_ERROR = "path_planner_error"
MOTION_STATUS_IK_TARGET_UNMET = "ik_target_unmet"
MOTION_STATUS_TRACKING_INCOMPLETE = "tracking_incomplete"
DEFAULT_IK_TARGET_UNMET_RETRY_LIMIT = 2
OBSTACLE_SCENE_FRAME_ID = "rm65_scene_world"
OBSTACLE_SCENE_FRAME_SPEC_HASH = "mujoco-desk-collision-world-v1"
FINAL_OBSTACLE_EDGE_RESOLUTION_RAD = float(np.deg2rad(0.1))
ADAPTIVE_OBSTACLE_REFINEMENT_DISTANCE_M = 0.03
ADAPTIVE_SELF_COLLISION_REFINEMENT_DISTANCE_M = 0.03
ZERO_DISTANCE_MESH_QUERY_PAIR_CLASSES = frozenset({"1:2", "1:4", "2:2", "4:4"})
class ShadowMujocoInitializationError(RuntimeError):
    """Typed fail-closed rejection while cold-initializing the shadow scene."""

    def __init__(
        self,
        reason_code: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.reason_code = str(reason_code)
        self.details = dict(details or {})
        super().__init__(self.reason_code)


def _collision_details_from_build_error(message: str) -> dict[str, object]:
    """Recover the structured MuJoCo check from the legacy build exception."""

    prefix = "simulation home qpos is collision-unsafe:"
    if not message.startswith(prefix):
        return {}
    try:
        value = ast.literal_eval(message[len(prefix) :].strip())
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(value, Mapping):
        return {}
    return {"collision_check": dict(value)}


def _is_zero_distance_robot_self_query_artifact(
    details: Mapping[str, object],
) -> bool:
    """Recognize the narrow MuJoCo mesh-query failure seen at build time.

    ``mj_geomDistance`` can return an exact zero for a separated robot-self
    mesh pair.  The build API only exposes the serialized distance result, so
    require all of the stable fields that identify that case before retrying
    with a known-safe compile seed.  Real environment/gripper collisions and
    malformed diagnostics remain fail-closed.
    """

    check = details.get("collision_check")
    if not isinstance(check, Mapping):
        return False
    try:
        minimum_distance = float(check.get("minimum_distance_m"))
    except (TypeError, ValueError):
        return False
    return bool(
        minimum_distance == 0.0
        and str(check.get("pair_class", "")) == "2:2"
        and str(check.get("collision_category", "")) == "robot_self"
    )


def _pair_has_contact(data: Any, collision: CollisionCheckResult) -> bool:
    """Return whether MuJoCo reports an actual contact for this geom pair."""

    if collision.geom1_id is None or collision.geom2_id is None:
        return False
    expected = tuple(sorted((int(collision.geom1_id), int(collision.geom2_id))))
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        observed = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        if observed == expected:
            return True
    return False


def _recover_zero_distance_mesh_query(
    collision: CollisionCheckResult,
    *,
    pair_has_contact: bool,
    witness_distance_m: float,
    minimum_clearance_m: float,
) -> CollisionCheckResult:
    """Recover a zero-distance mesh query only with a separated full witness.

    A contact, a non-zero penetration result, an unknown pair, or a witness
    shorter than the configured clearance all remain blocked.  This keeps the
    fallback specific to the observed discontinuous ``0.0`` query artifact.
    """

    if (
        collision.safe
        or collision.minimum_distance_m != 0.0
        or collision.pair_class not in ZERO_DISTANCE_MESH_QUERY_PAIR_CLASSES
        or pair_has_contact
        or not math.isfinite(float(witness_distance_m))
        or float(witness_distance_m) < float(minimum_clearance_m)
    ):
        return collision
    return replace(
        collision,
        safe=True,
        minimum_distance_m=float(witness_distance_m),
        minimum_distance_is_lower_bound=False,
        distance_query_raw_distance_m=None,
    )


def _recover_projector_zero_distance_result(
    projector: Any,
    source_data: Any,
    target_vector: np.ndarray,
    collision: CollisionCheckResult,
    *,
    minimum_clearance_m: float | None = None,
) -> CollisionCheckResult:
    """Repair one separated mesh false-zero using a full MuJoCo witness."""

    bounded_disagreement = (
        collision.safe
        and collision.minimum_distance_is_lower_bound
        and collision.distance_query_disagreement_count > 0
        and collision.distance_query_raw_distance_m == 0.0
    )
    if (
        (collision.safe and not bounded_disagreement)
        or (collision.minimum_distance_m != 0.0 and not bounded_disagreement)
        or collision.pair_class not in ZERO_DISTANCE_MESH_QUERY_PAIR_CLASSES
        or collision.geom1_id is None
        or collision.geom2_id is None
    ):
        return collision

    # ``check_qpos`` deliberately uses ``mj_kinematics`` for speed and hence
    # does not populate MuJoCo's contact list.  Recompute a private full-data
    # snapshot only for the suspicious exact-zero result, then use
    # the geom-distance witness to distinguish a separated mesh from contact.
    mujoco = projector.mujoco
    scratch = mujoco.MjData(projector.model)
    scratch.qpos[:] = source_data.qpos
    scratch.qpos[projector.controlled_qpos_indices] = np.asarray(
        target_vector, dtype=float
    )
    mujoco.mj_forward(projector.model, scratch)
    if _pair_has_contact(scratch, collision):
        return collision
    fromto = np.full(6, np.nan, dtype=np.float64)
    signed_distance = mujoco.mj_geomDistance(
        projector.model,
        scratch,
        int(collision.geom1_id),
        int(collision.geom2_id),
        float(projector.policy.distance_query_limit_m),
        fromto,
    )
    # The canonical oracle may already have proved separation with disjoint
    # AABBs. Recover measured clearance only for the original exact-zero
    # artifact; a negative or otherwise inconsistent query keeps its bound.
    if (not math.isfinite(float(signed_distance)) or signed_distance < 0.0
            or signed_distance >= projector.policy.distance_query_limit_m):
        return collision
    witness_distance = float(np.linalg.norm(fromto[:3] - fromto[3:]))
    if bounded_disagreement and witness_distance < collision.minimum_distance_m:
        return collision
    recovered = _recover_zero_distance_mesh_query(
        replace(collision, safe=False, minimum_distance_m=0.0)
        if bounded_disagreement else collision,
        pair_has_contact=False,
        witness_distance_m=witness_distance,
        minimum_clearance_m=float(
            projector.policy.minimum_clearance_m
            if minimum_clearance_m is None
            else minimum_clearance_m
        ),
    )
    return collision if bounded_disagreement and not recovered.safe else recovered


class _ArtifactAwareKinematicCollisionProjector(KinematicCollisionProjector):
    """Use the proven full-forward witness in every projector entry point.

    Exact-shadow preflight and the physical command projector both call the
    projector directly.  Keeping the recovery in ``_check_candidate`` makes
    those paths agree with grasp planning instead of fixing only one caller.
    """

    def _check_candidate(
        self,
        source_data: Any,
        controlled_qpos: np.ndarray,
        *,
        minimum_clearance_m: float | None = None,
    ) -> CollisionCheckResult:
        collision = super()._check_candidate(
            source_data,
            controlled_qpos,
            minimum_clearance_m=minimum_clearance_m,
            distance_query_limit_m=max(
                0.0,
                self.policy.minimum_clearance_m
                if minimum_clearance_m is None else minimum_clearance_m,
            ),
        )
        admission = collision
        if admission.safe:
            # Prefer measured clearance when the broader diagnostic agrees
            # with admission. This profile owns full-forward zero recovery.
            collision = self.oracle.check(
                self.scratch,
                minimum_clearance_m=minimum_clearance_m,
                distance_query_limit_m=max(
                    self.policy.distance_query_limit_m,
                    float(minimum_clearance_m or 0.0),
                ),
            )
        measured = _recover_projector_zero_distance_result(
            self,
            source_data,
            controlled_qpos,
            collision,
            minimum_clearance_m=minimum_clearance_m,
        )
        # Admission remains the threshold query, not the optional broad
        # measurement: separated convex meshes can report negative distances
        # only at the larger radius. If that measurement is inconsistent,
        # retain the conservative admitted lower bound, never its invalid
        # distance. An unsafe admission still follows the existing narrow
        # full-forward zero-witness recovery and is never broadly overridden.
        if admission.safe and (
            not measured.safe
            or (measured.minimum_distance_is_lower_bound
                and measured.distance_query_disagreement_count > 0)
        ):
            return admission
        return measured


def _install_artifact_aware_collision_projector(
    control: DeskTcpPositionControl,
) -> Any:
    """Install one projector shared by planning, exact replay, and actuation."""

    projector = control.scene.collision_projector()
    if projector is None:
        raise ShadowMujocoInitializationError("mujoco_collision_projector_unavailable")
    if isinstance(projector, _ArtifactAwareKinematicCollisionProjector):
        return projector
    recovered = _ArtifactAwareKinematicCollisionProjector(
        projector.mujoco,
        projector.model,
        projector.controlled_qpos_indices,
        projector.policy,
    )
    # MujocoScene owns this lazy projector.  Replacing its cached instance is
    # deliberately local to the real-shadow control; the general simulation
    # scene and deployed hardware runtime are not changed.
    control.scene._collision_projector = recovered
    return recovered


def _check_mujoco_qpos(
    control: DeskTcpPositionControl,
    projector: Any,
    target_vector: np.ndarray,
) -> CollisionCheckResult:
    """Check qpos through the shadow control's artifact-aware projector."""

    return projector.check_qpos(control.scene.data, target_vector)


def shadow_control_initialization_observation(
    control: DeskTcpPositionControl,
    snapshot: AuthoritativeRobotStateSnapshot,
) -> dict[str, object]:
    """Observe whether a cold MuJoCo control chain matches one real state cut."""

    measured = control.rig.measured_qpos_by_side()
    accepted = control.rig.accepted_qcmd_by_side()
    qpos_error = {
        side: float(np.max(np.abs(measured[side] - snapshot.qpos_by_side[side])))
        for side in SIDES
    }
    accepted_error = {
        side: float(np.max(np.abs(accepted[side] - snapshot.qpos_by_side[side])))
        for side in SIDES
    }
    profile = control.rig.config.gripper_actuation_profile
    if profile is None:
        raise ShadowMujocoInitializationError("gripper_model_unavailable")
    gripper_by_side = {
        side: float(control.scene.get_gripper_close(name))
        for name, side in profile.side_by_gripper.items()
    }
    if set(gripper_by_side) != set(SIDES):
        raise ShadowMujocoInitializationError(
            "gripper_side_mapping_invalid",
            details={"mapped_sides": tuple(sorted(gripper_by_side))},
        )
    gripper_error = {
        side: abs(gripper_by_side[side] - snapshot.gripper_close_by_side[side])
        for side in SIDES
    }
    projector = control.scene.collision_projector()
    if projector is None:
        raise ShadowMujocoInitializationError("mujoco_collision_projector_unavailable")
    collision = _check_mujoco_qpos(
        control,
        projector,
        control.scene.kinematic_controlled_qpos(),
    )
    maximum_qvel = float(np.max(np.abs(np.asarray(control.scene.data.qvel))))
    return {
        "snapshot_hash": snapshot.semantic_content_hash,
        "identity_hash": snapshot.identity.semantic_content_hash,
        "qpos_max_abs_error_rad_by_side": qpos_error,
        "accepted_qcmd_max_abs_error_rad_by_side": accepted_error,
        "gripper_close_abs_error_by_side": gripper_error,
        "maximum_qvel_rad_s": maximum_qvel,
        "collision_safe": bool(collision.safe),
        "collision_check": collision.to_dict(),
        "exact_state_match": bool(
            max(*qpos_error.values(), *accepted_error.values()) <= 1e-12
            and max(gripper_error.values()) <= 1e-12
            and maximum_qvel <= 1e-12
        ),
    }


def build_shadow_initialized_control(
    snapshot: AuthoritativeRobotStateSnapshot,
    *,
    config: DeskTcpPositionControlConfig | None = None,
) -> DeskTcpPositionControl:
    """Cold-start MuJoCo from an authoritative real arm/gripper state.

    The routine intentionally requires the deterministic non-realtime owner.
    It first seeds both arms through the public control configuration, stops
    the newly created rig, installs the measured gripper geometry, zeroes all
    velocities, rechecks collision, and restarts every planner/executor from
    that same state.  Any mismatch closes the scene and raises a typed error.
    """

    if not isinstance(snapshot, AuthoritativeRobotStateSnapshot):
        raise TypeError("snapshot must be AuthoritativeRobotStateSnapshot.")
    base = config or DeskTcpPositionControlConfig()
    if base.realtime_worker:
        raise ShadowMujocoInitializationError(
            "realtime_shadow_initialization_forbidden"
        )
    cfg = replace(
        base,
        left_home_qpos=tuple(float(value) for value in snapshot.qpos_by_side["left"]),
        right_home_qpos=tuple(float(value) for value in snapshot.qpos_by_side["right"]),
        left_initial_gripper_close=float(snapshot.gripper_close_by_side["left"]),
        right_initial_gripper_close=float(snapshot.gripper_close_by_side["right"]),
    )
    try:
        control = DeskTcpPositionControl.build(cfg)
    except RuntimeError as exc:
        if "simulation home qpos is collision-unsafe" in str(exc):
            details = _collision_details_from_build_error(str(exc))
            if _is_zero_distance_robot_self_query_artifact(details):
                # ``DeskTcpPositionControl.build`` checks the configured home
                # qpos before returning a control object.  A known MuJoCo
                # exact-zero robot-self query artifact can prevent that
                # object from being returned even though the actual contact
                # pipeline is empty.  Compile once from the canonical safe
                # simulation homes, then install and revalidate the measured
                # state below.  No real or simulated motion is issued here.
                compile_cfg = replace(
                    base,
                    left_home_qpos=tuple(
                        float(value) for value in DEFAULT_SIM_HOME_QPOS_BY_SIDE["left"]
                    ),
                    right_home_qpos=tuple(
                        float(value) for value in DEFAULT_SIM_HOME_QPOS_BY_SIDE["right"]
                    ),
                    left_initial_gripper_close=None,
                    right_initial_gripper_close=None,
                )
                try:
                    control = DeskTcpPositionControl.build(compile_cfg)
                except Exception as retry_exc:
                    raise ShadowMujocoInitializationError(
                        "mujoco_shadow_compile_failed",
                        details={
                            "backend_error": str(exc),
                            **details,
                            "compile_seed": "canonical_safe_simulation_home",
                            "compile_retry_error": str(retry_exc),
                        },
                    ) from retry_exc
                # Keep the caller-visible configuration tied to the measured
                # state.  The temporary compile seed must not leak into later
                # receipts or diagnostics.
                control.config = cfg
            else:
                # Keep the backend diagnostic verbatim for real/unknown
                # collisions and expose the measured-state collision class.
                raise ShadowMujocoInitializationError(
                    "measured_initial_state_collision_blocked",
                    details={
                        "backend_error": str(exc),
                        **details,
                    },
                ) from exc
        else:
            raise
    try:
        # No realtime worker exists, so this is an atomic cold-reset boundary.
        # Restarting the rig rebases the arm planners and gripper executors to
        # the exact state installed below instead of retaining default targets.
        projector = _install_artifact_aware_collision_projector(control)
        control.rig.close()
        control.scene.set_qpos_by_side(snapshot.qpos_by_side)
        profile = control.rig.config.gripper_actuation_profile
        if profile is None:
            raise ShadowMujocoInitializationError("gripper_model_unavailable")
        for name, side in profile.side_by_gripper.items():
            control.scene.set_gripper_close(
                name,
                float(snapshot.gripper_close_by_side[side]),
            )
        control.scene.data.qvel[:] = 0.0
        control.scene.forward()
        # The temporary safe compile seed may have initialized the retargeter
        # at a different qpos.  Rebase its internal solver and all continuity
        # state to the authoritative measured arm state before any target is
        # admitted.
        control.retargeter.set_current_qpos_pair(
            left_qpos=snapshot.qpos_by_side["left"],
            right_qpos=snapshot.qpos_by_side["right"],
            reset_tracking=True,
        )
        collision = _check_mujoco_qpos(
            control,
            projector,
            control.scene.kinematic_controlled_qpos(),
        )
        if not collision.safe:
            raise ShadowMujocoInitializationError(
                "measured_initial_state_collision_blocked",
                details={"collision_check": collision.to_dict()},
            )
        control.rig.start()
        observation = shadow_control_initialization_observation(control, snapshot)
        if not observation["collision_safe"]:
            raise ShadowMujocoInitializationError(
                "measured_initial_state_collision_blocked",
                details=observation,
            )
        if not observation["exact_state_match"]:
            raise ShadowMujocoInitializationError(
                "shadow_initial_state_mismatch",
                details=observation,
            )
        return control
    except BaseException:
        control.close()
        raise


class _MujocoFclObstacleOracle:
    """Require both the immutable FCL scene and live MuJoCo scene to be safe."""

    def __init__(
        self,
        control: DeskTcpPositionControl,
        fcl_oracle: RM65ObstacleCollisionOracle,
    ) -> None:
        projector = control.scene.collision_projector()
        if projector is None:
            raise RuntimeError(
                "combined path planning requires MuJoCo collision safety"
            )
        self.scene_hash = fcl_oracle.scene_hash
        self._control = control
        self._fcl_oracle = fcl_oracle
        self._projector = projector
        self._base_vector = control.scene.kinematic_controlled_qpos()
        self._check_cache: dict[
            tuple[tuple[str, tuple[int, ...], bytes], ...], JointConfigurationCheck
        ] = {}

    def check(
        self, qpos_by_component: Mapping[str, np.ndarray]
    ) -> JointConfigurationCheck:
        key = tuple(
            (
                str(component),
                tuple(np.asarray(value).shape),
                np.asarray(value, dtype=np.float64).tobytes(),
            )
            for component, value in sorted(qpos_by_component.items())
        )
        if key not in self._check_cache:
            self._check_cache[key] = self._check_uncached(qpos_by_component)
        return self._check_cache[key]

    def _check_uncached(
        self, qpos_by_component: Mapping[str, np.ndarray]
    ) -> JointConfigurationCheck:
        fcl = self._fcl_oracle.check(qpos_by_component)
        if not fcl.collision_check_available or not fcl.safe:
            return fcl
        try:
            target_vector = self._control.scene.kinematic_target_vector(
                self._base_vector,
                {
                    side: np.asarray(qpos_by_component[side], dtype=float)
                    for side in SIDES
                },
            )
            mujoco = _check_mujoco_qpos(
                self._control,
                self._projector,
                target_vector,
            )
            payload = mujoco.to_dict()
        except Exception as exc:  # noqa: BLE001 - backend failures fail closed
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="mujoco_collision_query_failed",
                metadata={
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "fcl_check": fcl.to_dict(),
                },
            )
        mujoco_distance = payload.get("minimum_distance_m")
        distances = [
            float(value)
            for value in (fcl.minimum_distance_m, mujoco_distance)
            if value is not None
        ]
        minimum_distance = min(distances) if distances else None
        closest_fcl_names = tuple(
            str(value) for value in fcl.metadata.get("closest_geometry_names", ())
        )
        fixed_scene_is_nearest = any(
            "table" in name or "camera_pole" in name for name in closest_fcl_names
        )
        adaptive_edge_distance = (
            fcl.minimum_distance_m
            if fixed_scene_is_nearest
            else (
                minimum_distance
                if minimum_distance is not None
                and minimum_distance <= ADAPTIVE_SELF_COLLISION_REFINEMENT_DISTANCE_M
                else None
            )
        )
        return JointConfigurationCheck(
            safe=bool(mujoco.safe),
            collision_check_available=True,
            reason=(
                "combined_fcl_mujoco_collision_safe"
                if mujoco.safe
                else "mujoco_collision_blocked"
            ),
            minimum_distance_m=minimum_distance,
            metadata={
                "collision_sources": ("pinocchio_fcl", "mujoco"),
                "adaptive_edge_distance_m": adaptive_edge_distance,
                "fcl_check": fcl.to_dict(),
                "mujoco_check": payload,
            },
        )


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
    translation_duration = (
        peak_progress_derivative * translation_distance / linear_speed
    )
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
    relative_rotvec = Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).as_rotvec()
    poses: list[np.ndarray] = []
    for progress in np.asarray(quintic_progress(unit_times), dtype=float):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = start[:3, 3] + progress * (target[:3, 3] - start[:3, 3])
        pose[:3, :3] = (
            start[:3, :3] @ Rotation.from_rotvec(progress * relative_rotvec).as_matrix()
        )
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

    def solver_target_from_dmtac_target(
        self, target_pose_base: np.ndarray
    ) -> np.ndarray:
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


def build_shadow_simulation_receipt_from_playback(
    *,
    playback: GraspMotionPlayback,
    trajectory: RegisteredShadowJointTrajectory,
    evidence_refs: Sequence[str],
) -> ShadowSimulationReceipt:
    """Translate current MuJoCo playback into fail-closed replay evidence.

    The existing arbitrary-grasp runner re-solves Cartesian IK while following
    FK-derived waypoints.  Therefore even a visually completed playback is not
    evidence that the exact registered qpos trajectory was executed.  Such a
    run is reported as ``execution_uncertified`` until a lower-owned exact-qpos
    simulation executor and terminal HOLD receipt are present.
    """

    if not isinstance(playback, GraspMotionPlayback):
        raise TypeError("playback must be GraspMotionPlayback.")
    if not isinstance(trajectory, RegisteredShadowJointTrajectory):
        raise TypeError("trajectory must be RegisteredShadowJointTrajectory.")
    status_by_motion = {
        MOTION_STATUS_ENDPOINT_REJECTED: ShadowSimulationStatus.ENDPOINT_REJECTED,
        MOTION_STATUS_PATH_START_BLOCKED: (
            ShadowSimulationStatus.PATH_START_COLLISION_BLOCKED
        ),
        MOTION_STATUS_PATH_TARGET_BLOCKED: (
            ShadowSimulationStatus.PATH_TARGET_COLLISION_BLOCKED
        ),
        MOTION_STATUS_PATH_COLLISION_BLOCKED: (
            ShadowSimulationStatus.PATH_COLLISION_BLOCKED
        ),
        MOTION_STATUS_PATH_SEARCH_EXHAUSTED: (
            ShadowSimulationStatus.PATH_SEARCH_EXHAUSTED
        ),
        MOTION_STATUS_PATH_PLANNER_ERROR: ShadowSimulationStatus.PATH_PLANNER_ERROR,
        MOTION_STATUS_IK_TARGET_UNMET: ShadowSimulationStatus.IK_TARGET_UNMET,
        MOTION_STATUS_TRACKING_INCOMPLETE: ShadowSimulationStatus.TRACKING_INCOMPLETE,
    }
    # GraspMotionPlayback has neither a typed exact-qpos lower receipt nor a
    # typed terminal HOLD receipt.  Free-form summary flags cannot upgrade it.
    exact_confirmed = False
    terminal_hold_confirmed = False
    model_identity_confirmed = False
    if playback.motion_status == MOTION_STATUS_COMPLETED:
        status = (
            ShadowSimulationStatus.COMPLETED
            if exact_confirmed and terminal_hold_confirmed
            else ShadowSimulationStatus.EXECUTION_UNCERTIFIED
        )
    else:
        try:
            status = status_by_motion[playback.motion_status]
        except KeyError as exc:
            raise ValueError(
                f"unsupported playback motion_status: {playback.motion_status!r}"
            ) from exc
    clearance = None
    path_preflight = dict(playback.summary.get("path_preflight", {}))
    raw_clearance = path_preflight.get("minimum_distance_m")
    if raw_clearance is not None:
        clearance = float(raw_clearance)
    return ShadowSimulationReceipt(
        trajectory_hash=trajectory.semantic_content_hash,
        initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
        identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=status,
        executed_point_count=(
            len(trajectory.points)
            if exact_confirmed and playback.frame_count == len(trajectory.points)
            else 0
        ),
        terminal_hold_confirmed=terminal_hold_confirmed,
        exact_planned_qpos_execution_confirmed=exact_confirmed,
        model_identity_confirmed=model_identity_confirmed,
        evidence_refs=tuple(evidence_refs),
        minimum_clearance_m=clearance,
    )


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
    qpos = {item: control.scene.get_arm_qpos(item).copy() for item in SIDES}
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
    obstacle_oracle: RM65ObstacleCollisionOracle | None = None,
) -> Any:
    projector = control.scene.collision_projector()
    base_vector = control.scene.kinematic_controlled_qpos()
    current_qpos = {item: control.scene.get_arm_qpos(item).copy() for item in SIDES}

    def check(
        candidate: IKCandidate, _target_pose: np.ndarray
    ) -> CollisionCandidateAssessment:
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
        result = _check_mujoco_qpos(control, projector, target)
        payload = result.to_dict()
        obstacle_payload: dict[str, object] | None = None
        obstacle_distance: float | None = None
        obstacle_required_distance = 0.0
        if not result.safe:
            return CollisionCandidateAssessment(
                status=CollisionCandidateStatus.BLOCKED,
                reason_code="mujoco_endpoint_collision_blocked",
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
                metadata={"collision_source": "mujoco", **payload},
            )
        if obstacle_oracle is not None:
            qpos_by_side = {
                item: values.copy() for item, values in current_qpos.items()
            }
            qpos_by_side[side] = np.asarray(candidate.qpos_rad, dtype=float).copy()
            obstacle = obstacle_oracle.check(qpos_by_side)
            obstacle_payload = obstacle.to_dict()
            obstacle_distance = obstacle.minimum_distance_m
            obstacle_required_distance = float(
                obstacle.metadata.get("minimum_clearance_m", 0.0)
            )
            if not obstacle.collision_check_available:
                return CollisionCandidateAssessment(
                    status=CollisionCandidateStatus.ERROR,
                    reason_code="obstacle_endpoint_collision_check_unavailable",
                    metadata={
                        "collision_source": "pinocchio_fcl_obstacle_scene",
                        **obstacle_payload,
                    },
                )
            if not obstacle.safe:
                obstacle_id = obstacle.metadata.get("closest_obstacle_id")
                return CollisionCandidateAssessment(
                    status=CollisionCandidateStatus.BLOCKED,
                    reason_code="obstacle_endpoint_collision_blocked",
                    minimum_distance_m=obstacle.minimum_distance_m,
                    required_distance_m=float(
                        obstacle.metadata.get("minimum_clearance_m", 0.0)
                    ),
                    obstacle_ids=(str(obstacle_id),) if obstacle_id else (),
                    metadata={
                        "collision_source": "pinocchio_fcl_obstacle_scene",
                        **obstacle_payload,
                    },
                )
        return CollisionCandidateAssessment(
            status=CollisionCandidateStatus.SAFE,
            reason_code=(
                "mujoco_and_obstacle_endpoint_collision_safe"
                if obstacle_oracle is not None
                else "mujoco_endpoint_collision_safe"
            ),
            minimum_distance_m=(
                min(
                    (
                        float(value)
                        for value in (
                            payload.get("minimum_distance_m"),
                            obstacle_distance,
                        )
                        if value is not None
                    ),
                    default=None,
                )
            ),
            required_distance_m=max(
                float(projector.policy.minimum_clearance_m),
                obstacle_required_distance,
            ),
            obstacle_ids=tuple(
                str(value)
                for value in (payload.get("body1"), payload.get("body2"))
                if value
            ),
            metadata={
                "collision_source": (
                    "mujoco_and_pinocchio_fcl_obstacle_scene"
                    if obstacle_payload is not None
                    else "mujoco"
                ),
                "mujoco_check": payload,
                "obstacle_check": obstacle_payload,
            },
        )

    return check


def build_obstacle_collision_oracle(
    control: DeskTcpPositionControl,
    scene: ObstacleSceneSnapshot,
) -> RM65ObstacleCollisionOracle:
    """Bind one immutable world-frame cube scene to the active dual-arm model."""

    if scene.reference_frame_id != OBSTACLE_SCENE_FRAME_ID:
        raise ValueError(
            f"obstacle scene reference_frame_id must be {OBSTACLE_SCENE_FRAME_ID!r}"
        )
    if scene.reference_frame_spec_hash != OBSTACLE_SCENE_FRAME_SPEC_HASH:
        raise ValueError(
            "obstacle scene reference_frame_spec_hash must be "
            f"{OBSTACLE_SCENE_FRAME_SPEC_HASH!r}"
        )
    solver = control.retargeter.ik_solver
    return RM65ObstacleCollisionOracle.from_solver(
        solver,
        scene=scene,
        solution_layout=control.retargeter.profile.solution_layout,
        expected_reference_frame_id=OBSTACLE_SCENE_FRAME_ID,
        expected_reference_frame_spec_hash=OBSTACLE_SCENE_FRAME_SPEC_HASH,
        attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
    )


def build_path_obstacle_collision_oracle(
    scene: ObstacleSceneSnapshot,
    *,
    use_profile_proxy_geometry: bool = True,
    stop_at_first_clearance_violation: bool = False,
) -> RM65ObstacleCollisionOracle:
    """Bind the conservative strict proxy/table profile used for path search."""

    return build_rm65_profile_obstacle_collision_oracle(
        scene_profile=DEFAULT_GRASP_PATH_COLLISION_SCENE,
        scene=scene,
        expected_reference_frame_id=OBSTACLE_SCENE_FRAME_ID,
        expected_reference_frame_spec_hash=OBSTACLE_SCENE_FRAME_SPEC_HASH,
        attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
        use_profile_proxy_geometry=use_profile_proxy_geometry,
        stop_at_first_clearance_violation=stop_at_first_clearance_violation,
    )


def evaluate_motion_target_endpoint(
    control: DeskTcpPositionControl,
    target: GraspMotionTarget,
    *,
    config: RM65RandomGraspPoseReachabilityConfig | None = None,
    obstacle_oracle: RM65ObstacleCollisionOracle | None = None,
    collision_checker: Callable[[IKCandidate, np.ndarray], CollisionCandidateAssessment]
    | None = None,
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
    checker = collision_checker or _collision_checker(
        control,
        side=target.side,
        obstacle_oracle=obstacle_oracle,
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
        collision_checker=checker,
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
    maximum_joint_motion_deg = float(np.max(np.abs(np.rad2deg(target - start))))
    sample_count = max(
        241,
        math.ceil(maximum_joint_motion_deg / 0.1) + 1,
    )
    sample_count = min(sample_count, 5001)
    minimum_clearance_m = math.inf
    for index, progress in enumerate(np.linspace(0.0, 1.0, sample_count, dtype=float)):
        qpos = start + float(progress) * (target - start)
        target_vector = control.scene.kinematic_target_vector(
            base_vector, {normalized_side: qpos}
        )
        collision = _check_mujoco_qpos(control, projector, target_vector)
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
            None if not math.isfinite(minimum_clearance_m) else minimum_clearance_m
        ),
    }


def _direct_joint_path_oracle_assessment(
    control: Any,
    *,
    side: str,
    start_qpos_rad: np.ndarray,
    target_qpos_rad: np.ndarray,
    oracle: Any,
) -> dict[str, object]:
    """Densely check one joint segment with its bound URDF/FCL oracle."""

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
    qpos_by_side = {
        item: control.scene.get_arm_qpos(item).copy() for item in SIDES
    }
    maximum_joint_motion = float(np.max(np.abs(target - start)))
    edge_samples = max(
        1,
        math.ceil(maximum_joint_motion / FINAL_OBSTACLE_EDGE_RESOLUTION_RAD),
    )
    sample_count = edge_samples + 1
    minimum_clearance_m = math.inf
    for index in range(sample_count):
        progress = float(index) / float(edge_samples)
        candidate = {item: values.copy() for item, values in qpos_by_side.items()}
        candidate[normalized_side] = start + float(progress) * (target - start)
        assessment = oracle.check(candidate)
        if assessment.minimum_distance_m is not None:
            minimum_clearance_m = min(
                minimum_clearance_m, float(assessment.minimum_distance_m)
            )
        if not assessment.collision_check_available or not assessment.safe:
            return {
                "safe": False,
                "reason_code": (
                    "urdf_path_collision_blocked"
                    if assessment.collision_check_available
                    else "urdf_path_collision_check_unavailable"
                ),
                "blocked_sample_index": index,
                "blocked_progress": float(progress),
                "collision_sample_count": sample_count,
                "minimum_path_clearance_m": (
                    None
                    if not math.isfinite(minimum_clearance_m)
                    else minimum_clearance_m
                ),
                "blocked_collision": assessment.to_dict(),
            }
    return {
        "safe": True,
        "reason_code": "urdf_direct_joint_path_collision_safe",
        "collision_sample_count": sample_count,
        "minimum_path_clearance_m": (
            None if not math.isfinite(minimum_clearance_m) else minimum_clearance_m
        ),
    }


def plan_obstacle_aware_joint_path(
    control: DeskTcpPositionControl,
    *,
    side: str,
    target_qpos_rad: np.ndarray,
    single_arm_solver: Any,
    obstacle_oracle: Any,
    planner_config: JointPathPlannerConfig | None = None,
) -> JointPathPlanResult:
    """Run direct-first bounded BiRRT while freezing the inactive arm."""

    normalized_side = str(side).strip().lower()
    if normalized_side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    start = {item: control.scene.get_arm_qpos(item).copy() for item in SIDES}
    target = {item: values.copy() for item, values in start.items()}
    target[normalized_side] = np.asarray(target_qpos_rad, dtype=float).reshape(6)
    cfg = planner_config or JointPathPlannerConfig()
    physical_lower = np.asarray(single_arm_solver.q_min, dtype=float).reshape(6)
    physical_upper = np.asarray(single_arm_solver.q_max, dtype=float).reshape(6)
    # A full-limit search can find enormous, numerically awkward detours even
    # when a small local avoidance move exists.  Keep the search around the
    # current/goal corridor while intersecting it with physical joint limits.
    lower = np.maximum(
        physical_lower,
        np.minimum(start[normalized_side], target[normalized_side])
        - cfg.sample_padding_rad,
    )
    upper = np.minimum(
        physical_upper,
        np.maximum(start[normalized_side], target[normalized_side])
        + cfg.sample_padding_rad,
    )
    return plan_collision_aware_joint_path(
        start_qpos_by_component=start,
        target_qpos_by_component=target,
        movable_components=(normalized_side,),
        oracle=obstacle_oracle,
        config=cfg,
        joint_lower_by_component={normalized_side: lower},
        joint_upper_by_component={normalized_side: upper},
        scene_hash=obstacle_oracle.scene_hash,
    )


def revalidate_obstacle_path(
    waypoints: Sequence[Mapping[str, np.ndarray]],
    *,
    oracle: Any,
    edge_resolution_rad: float = FINAL_OBSTACLE_EDGE_RESOLUTION_RAD,
) -> dict[str, object]:
    """Densely recheck the final shortened path against the bound FCL scene."""

    resolution = float(edge_resolution_rad)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("edge_resolution_rad must be finite and positive")
    if not waypoints:
        raise ValueError("obstacle path revalidation requires waypoints")
    sample_count = 0
    minimum_distance_m = math.inf
    for segment_index, (left, right) in enumerate(pairwise(waypoints)):
        if set(left) != set(right):
            raise ValueError("path waypoint component names must match")
        maximum_delta = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(right[item], dtype=float)
                        - np.asarray(left[item], dtype=float)
                    )
                )
            )
            for item in left
        )
        edge_samples = max(1, math.ceil(maximum_delta / resolution))
        for edge_sample_index in range(edge_samples + 1):
            alpha = float(edge_sample_index) / float(edge_samples)
            candidate = {
                item: np.asarray(left[item], dtype=float)
                + alpha
                * (
                    np.asarray(right[item], dtype=float)
                    - np.asarray(left[item], dtype=float)
                )
                for item in left
            }
            assessment = oracle.check(candidate)
            sample_count += 1
            if assessment.minimum_distance_m is not None:
                minimum_distance_m = min(
                    minimum_distance_m, float(assessment.minimum_distance_m)
                )
            if not assessment.collision_check_available:
                return {
                    "safe": False,
                    "collision_check_available": False,
                    "reason_code": "final_obstacle_revalidation_unavailable",
                    "segment_index": segment_index,
                    "edge_sample_index": edge_sample_index,
                    "sample_count": sample_count,
                    "check": assessment.to_dict(),
                }
            if not assessment.safe:
                return {
                    "safe": False,
                    "collision_check_available": True,
                    "reason_code": "final_obstacle_revalidation_blocked",
                    "segment_index": segment_index,
                    "edge_sample_index": edge_sample_index,
                    "sample_count": sample_count,
                    "check": assessment.to_dict(),
                }
    if len(waypoints) == 1:
        assessment = oracle.check(waypoints[0])
        sample_count = 1
        if assessment.minimum_distance_m is not None:
            minimum_distance_m = float(assessment.minimum_distance_m)
        if not assessment.collision_check_available or not assessment.safe:
            return {
                "safe": False,
                "collision_check_available": assessment.collision_check_available,
                "reason_code": (
                    "final_obstacle_revalidation_blocked"
                    if assessment.collision_check_available
                    else "final_obstacle_revalidation_unavailable"
                ),
                "sample_count": sample_count,
                "check": assessment.to_dict(),
            }
    return {
        "safe": True,
        "collision_check_available": True,
        "reason_code": "final_obstacle_revalidation_safe",
        "edge_resolution_deg": math.degrees(resolution),
        "sample_count": sample_count,
        "minimum_path_distance_m": (
            None if not math.isfinite(minimum_distance_m) else minimum_distance_m
        ),
    }


def plan_revalidated_obstacle_aware_joint_path(
    control: DeskTcpPositionControl,
    *,
    side: str,
    target_qpos_rad: np.ndarray,
    single_arm_solver: Any,
    obstacle_oracle: Any,
    final_obstacle_oracle: Any | None = None,
    planner_config: JointPathPlannerConfig | None = None,
) -> tuple[
    JointPathPlanResult, dict[str, object] | None, tuple[dict[str, object], ...]
]:
    """Plan once with adaptive edge checks, then densely revalidate the result."""

    path_config = planner_config or JointPathPlannerConfig(
        adaptive_edge_resolution_rad=FINAL_OBSTACLE_EDGE_RESOLUTION_RAD,
        adaptive_refinement_distance_m=ADAPTIVE_OBSTACLE_REFINEMENT_DISTANCE_M,
    )
    path = plan_obstacle_aware_joint_path(
        control,
        side=side,
        target_qpos_rad=target_qpos_rad,
        single_arm_solver=single_arm_solver,
        obstacle_oracle=obstacle_oracle,
        planner_config=path_config,
    )
    attempts: list[dict[str, object]] = [
        {
            "phase": "planning",
            "edge_resolution_deg": math.degrees(path_config.edge_resolution_rad),
            "adaptive_edge_resolution_deg": (
                None
                if path_config.adaptive_edge_resolution_rad is None
                else math.degrees(path_config.adaptive_edge_resolution_rad)
            ),
            "path_status": path.status.value,
            "path_reason": path.reason,
        }
    ]
    if path.status is not JointPathStatus.PATH_FOUND:
        return path, None, tuple(attempts)

    final_check = revalidate_obstacle_path(
        path.waypoints,
        oracle=(
            obstacle_oracle
            if final_obstacle_oracle is None
            else final_obstacle_oracle
        ),
    )
    attempts[-1]["final_revalidation"] = final_check
    return path, final_check, tuple(attempts)


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
    current = {item: control.scene.get_arm_qpos(item).copy() for item in SIDES}
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
        collision = _check_mujoco_qpos(control, projector, target_vector)
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
            float(Rotation.from_matrix(lhs[:3, :3].T @ rhs[:3, :3]).magnitude())
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
    collision_oracle: Any | None = None,
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

    initial_by_side = {item: control.scene.get_arm_qpos(item).copy() for item in SIDES}
    delta = target - start
    maximum_joint_motion_deg = float(np.max(np.abs(np.rad2deg(delta))))
    path_collision = (
        _direct_joint_path_collision_assessment(
            control,
            side=normalized_side,
            start_qpos_rad=start,
            target_qpos_rad=target,
        )
        if collision_oracle is None
        else _direct_joint_path_oracle_assessment(
            control,
            side=normalized_side,
            start_qpos_rad=start,
            target_qpos_rad=target,
            oracle=collision_oracle,
        )
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
        qpos_by_side = {item: values.copy() for item, values in initial_by_side.items()}
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
        np.abs(np.diff(np.rad2deg(np.stack(qpos_path, axis=0)), axis=0)) * reference_hz,
        axis=0,
    )
    if maximum_linear_speed > linear_speed * 1.0001:
        raise RuntimeError("joint-FK plan exceeded requested linear TCP speed")
    if maximum_angular_speed > angular_speed * 1.0001:
        raise RuntimeError("joint-FK plan exceeded requested angular TCP speed")
    if np.any(maximum_joint_speed > joint_velocity_limits * 1.0001):
        raise RuntimeError(
            "joint-FK plan exceeded the configured joint velocity profile"
        )
    return JointFkReferencePlan(
        status="safe",
        pose_desk=pose_desk,
        qpos_rad=qpos_path,
        metadata={
            "planner": "direct_joint_quintic_fk",
            "path_reference_semantics": "dmtac_tcp_fk_of_admitted_joint_path",
            "inactive_side": inactive_side,
            "collision_sample_count": path_collision["collision_sample_count"],
            "minimum_path_clearance_m": path_collision["minimum_path_clearance_m"],
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


def build_collision_checked_multiwaypoint_joint_fk_reference_plan(
    control: DeskTcpPositionControl,
    *,
    side: str,
    waypoints_qpos_rad: Sequence[np.ndarray],
    linear_speed_m_s: float,
    angular_speed_deg_s: float,
    minimum_duration_sec: float,
    collision_oracle: Any | None = None,
) -> JointFkReferencePlan:
    """Time-parameterize a validated waypoint path as C2 stop-at-waypoint segments."""

    waypoints = tuple(
        np.asarray(value, dtype=float).reshape(6).copy() for value in waypoints_qpos_rad
    )
    if not waypoints:
        raise ValueError("multiwaypoint plan requires at least one waypoint")
    if any(not np.all(np.isfinite(value)) for value in waypoints):
        raise ValueError("multiwaypoint plan requires finite waypoints")
    if len(waypoints) == 1:
        waypoints = (waypoints[0], waypoints[0].copy())
    segment_count = len(waypoints) - 1
    segment_minimum_duration = max(
        0.25, float(minimum_duration_sec) / float(segment_count)
    )
    poses: list[np.ndarray] = []
    qpos_values: list[np.ndarray] = []
    segment_metadata: list[dict[str, object]] = []
    for segment_index, (start, target) in enumerate(pairwise(waypoints)):
        segment = build_collision_checked_joint_fk_reference_plan(
            control,
            side=side,
            start_qpos_rad=start,
            target_qpos_rad=target,
            linear_speed_m_s=linear_speed_m_s,
            angular_speed_deg_s=angular_speed_deg_s,
            minimum_duration_sec=segment_minimum_duration,
            collision_oracle=collision_oracle,
        )
        segment_metadata.append(
            {"segment_index": segment_index, **dict(segment.metadata)}
        )
        if segment.status != "safe":
            return JointFkReferencePlan(
                status="collision_blocked",
                pose_desk=(),
                qpos_rad=(),
                metadata={
                    "planner": "multiwaypoint_joint_quintic_fk",
                    "reason_code": (
                        "mujoco_waypoint_segment_collision_blocked"
                        if collision_oracle is None
                        else "urdf_waypoint_segment_collision_blocked"
                    ),
                    "blocked_segment_index": segment_index,
                    "waypoint_count": len(waypoints),
                    "segment_count": segment_count,
                    "segments": segment_metadata,
                    "motion_emitted": False,
                },
            )
        offset = 0 if segment_index == 0 else 1
        poses.extend(segment.pose_desk[offset:])
        qpos_values.extend(segment.qpos_rad[offset:])
    return JointFkReferencePlan(
        status="safe",
        pose_desk=tuple(poses),
        qpos_rad=tuple(qpos_values),
        metadata={
            "planner": "multiwaypoint_joint_quintic_fk",
            "path_reference_semantics": "dmtac_tcp_fk_of_admitted_joint_path",
            "time_parameterization": "per_segment_quintic_stop_at_waypoint",
            "continuity": "C2_at_waypoints_with_zero_velocity_and_acceleration",
            "jerk_bounded": False,
            "waypoint_count": len(waypoints),
            "segment_count": segment_count,
            "reference_frame_count": len(poses),
            "segments": segment_metadata,
        },
    )


def _planner_error_receipt(
    target: GraspMotionTarget,
    *,
    reason_code: str,
    metadata: Mapping[str, object],
) -> GraspReachabilityReceipt:
    return GraspReachabilityReceipt(
        status=GraspReachabilityStatus.PLANNER_ERROR,
        reason_code=str(reason_code),
        proof_scope="obstacle_scene_planner_initialization",
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        target_pose=tuple(
            tuple(float(value) for value in row)
            for row in target.target_pose_base_from_dmtac_tcp
        ),
        metadata=dict(metadata),
    )


def _path_failure_motion_status(status: JointPathStatus) -> str:
    if status is JointPathStatus.START_BLOCKED:
        return MOTION_STATUS_PATH_START_BLOCKED
    if status is JointPathStatus.TARGET_BLOCKED:
        return MOTION_STATUS_PATH_TARGET_BLOCKED
    if status is JointPathStatus.PATH_SEARCH_EXHAUSTED:
        return MOTION_STATUS_PATH_SEARCH_EXHAUSTED
    return MOTION_STATUS_PATH_PLANNER_ERROR


def execute_grasp_motion(
    control: DeskTcpPositionControl,
    target: GraspMotionTarget,
    *,
    linear_speed_m_s: float = 0.12,
    angular_speed_deg_s: float = 45.0,
    minimum_duration_sec: float = 2.0,
    terminal_hold_sec: float = 1.5,
    ik_target_unmet_retry_limit: int = DEFAULT_IK_TARGET_UNMET_RETRY_LIMIT,
    obstacle_scene: ObstacleSceneSnapshot | None = None,
    obstacle_planner_config: JointPathPlannerConfig | None = None,
) -> GraspMotionPlayback:
    retry_limit = int(ik_target_unmet_retry_limit)
    if retry_limit < 0:
        raise ValueError("ik_target_unmet_retry_limit must be non-negative")
    initial_qpos = {side: control.scene.get_arm_qpos(side).copy() for side in SIDES}
    initial_world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"], right_qpos=initial_qpos["right"]
    )
    world_from_base = np.linalg.inv(control.world_to_base_by_side[target.side])
    target_world_active = world_from_base @ target.target_pose_base_from_dmtac_tcp
    target_world_by_side = {side: initial_world[side].copy() for side in SIDES}
    target_world_by_side[target.side] = target_world_active.copy()
    obstacle_oracle: RM65ObstacleCollisionOracle | None = None
    path_oracle: Any | None = None
    path_revalidation_oracle: Any | None = None
    obstacle_initialization_error: dict[str, object] | None = None
    if obstacle_scene is not None:
        try:
            obstacle_oracle = build_obstacle_collision_oracle(control, obstacle_scene)
            path_oracle = _MujocoFclObstacleOracle(
                control, build_path_obstacle_collision_oracle(obstacle_scene)
            )
            path_revalidation_oracle = _MujocoFclObstacleOracle(
                control,
                build_path_obstacle_collision_oracle(
                    obstacle_scene,
                    use_profile_proxy_geometry=False,
                    stop_at_first_clearance_violation=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - planner initialization fails closed
            obstacle_initialization_error = {
                "reason_code": "obstacle_collision_oracle_initialization_failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
    if obstacle_initialization_error is None:
        if obstacle_oracle is None:
            receipt, binding, endpoint_solver = evaluate_motion_target_endpoint(
                control, target
            )
        else:
            receipt, binding, endpoint_solver = evaluate_motion_target_endpoint(
                control,
                target,
                obstacle_oracle=obstacle_oracle,
            )
    else:
        endpoint_solver = build_rm65_single_arm_ik_solver(
            RM65SingleArmIKConfig(controlled_frame="tool_tcp")
        )
        binding = derive_tool_tcp_frame_binding(
            control, side=target.side, single_arm_solver=endpoint_solver
        )
        receipt = _planner_error_receipt(
            target,
            reason_code="obstacle_collision_oracle_initialization_failed",
            metadata=obstacle_initialization_error,
        )
    empty_summary = {
        "reference_rate_hz": float(control.contract.timing.ik_hz),
        "command_rate_hz": float(control.contract.timing.command_hz),
        "physics_rate_hz": float(control.contract.timing.physics_hz),
        "motion_emitted": False,
        "obstacle_scene": (
            None
            if obstacle_scene is None
            else {
                **obstacle_scene.to_dict(),
                "semantic_content_hash": obstacle_scene.semantic_content_hash,
            }
        ),
    }
    if receipt.status is not GraspReachabilityStatus.REACHABLE:
        return GraspMotionPlayback(
            motion_status=(
                MOTION_STATUS_PATH_PLANNER_ERROR
                if obstacle_initialization_error is not None
                else MOTION_STATUS_ENDPOINT_REJECTED
            ),
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
    joint_path_search: dict[str, object] | None = None
    final_obstacle_revalidation: dict[str, object] | None = None
    if obstacle_oracle is None:
        joint_fk_plan = build_collision_checked_joint_fk_reference_plan(
            control,
            side=target.side,
            start_qpos_rad=initial_qpos[target.side],
            target_qpos_rad=selected_qpos,
            linear_speed_m_s=linear_speed_m_s,
            angular_speed_deg_s=angular_speed_deg_s,
            minimum_duration_sec=minimum_duration_sec,
        )
    else:
        (
            joint_path_result,
            final_obstacle_revalidation,
            planning_attempts,
        ) = plan_revalidated_obstacle_aware_joint_path(
            control,
            side=target.side,
            target_qpos_rad=selected_qpos,
            single_arm_solver=endpoint_solver,
            obstacle_oracle=path_oracle,
            final_obstacle_oracle=path_revalidation_oracle,
            planner_config=obstacle_planner_config,
        )
        joint_path_search = {
            **joint_path_result.to_dict(),
            "planning_attempts": planning_attempts,
        }
        if joint_path_result.status is not JointPathStatus.PATH_FOUND:
            return GraspMotionPlayback(
                motion_status=_path_failure_motion_status(joint_path_result.status),
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
                    "path_preflight": joint_path_search,
                },
            )
        if final_obstacle_revalidation is None:
            raise RuntimeError("found obstacle path is missing final revalidation")
        if not bool(final_obstacle_revalidation["safe"]):
            return GraspMotionPlayback(
                motion_status=(
                    MOTION_STATUS_PATH_COLLISION_BLOCKED
                    if bool(
                        final_obstacle_revalidation.get(
                            "collision_check_available", False
                        )
                    )
                    else MOTION_STATUS_PATH_PLANNER_ERROR
                ),
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
                    "path_preflight": {
                        "joint_path_search": joint_path_search,
                        "final_obstacle_revalidation": (final_obstacle_revalidation),
                    },
                },
            )
        joint_fk_plan = build_collision_checked_multiwaypoint_joint_fk_reference_plan(
            control,
            side=target.side,
            waypoints_qpos_rad=tuple(
                waypoint[target.side] for waypoint in joint_path_result.waypoints
            ),
            linear_speed_m_s=linear_speed_m_s,
            angular_speed_deg_s=angular_speed_deg_s,
            minimum_duration_sec=minimum_duration_sec,
            collision_oracle=path_revalidation_oracle,
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
                "path_preflight": (
                    dict(joint_fk_plan.metadata)
                    if joint_path_search is None
                    else {
                        "joint_path_search": joint_path_search,
                        "final_obstacle_revalidation": (final_obstacle_revalidation),
                        "mujoco_segment_preflight": dict(joint_fk_plan.metadata),
                    }
                ),
            },
        )
    if obstacle_oracle is None:
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
    else:
        hold_frame_count = math.ceil(
            float(terminal_hold_sec) * float(control.contract.timing.ik_hz)
        )
        active_poses_desk = joint_fk_plan.pose_desk + tuple(
            joint_fk_plan.pose_desk[-1].copy() for _ in range(hold_frame_count)
        )
        correction_metadata = {
            "enabled": False,
            "reason": "obstacle_mode_preserves_only_fcl_revalidated_fk_references",
            "terminal_hold_frame_count": hold_frame_count,
        }
    inactive_side = "right" if target.side == "left" else "left"
    inactive_pose_desk = (
        np.linalg.inv(control.world_from_desk) @ initial_world[inactive_side]
    )
    physics_ticks_exact = float(control.contract.timing.physics_hz) / float(
        control.contract.timing.ik_hz
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
            endpoint_status_counts[endpoint_status] = (
                int(endpoint_status_counts.get(endpoint_status, 0)) + 1
            )
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
        qpos = {side: control.scene.get_arm_qpos(side).copy() for side in SIDES}
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
            np.max(np.abs(np.rad2deg(frame[target.side] - initial_qpos[target.side])))
        )
        for frame in actual_qpos_by_frame
    )
    summary = {
        **empty_summary,
        "motion_emitted": True,
        "trajectory": {
            "planner": (
                "collision_checked_joint_quintic_fk_then_local_tcp_correction"
                if obstacle_oracle is None
                else "deterministic_birrt_then_multiwaypoint_joint_quintic_fk"
            ),
            "joint_path_search": joint_path_search,
            "final_obstacle_revalidation": final_obstacle_revalidation,
            "joint_fk_path": dict(joint_fk_plan.metadata),
            "terminal_tcp_correction": correction_metadata,
            "execution_contract": {
                "simulation_only": True,
                "planner_execution_certified": False,
                "real_motion_authorized": False,
                "reference_mode": "validated_q_waypoint_fk_then_cartesian_re_ik",
                "exact_planned_qpos_dispatch": False,
                "runtime_cube_constraints_in_lower_ik": False,
                "limitation": (
                    "static cubes constrain endpoint selection and offline path "
                    "planning, but the lower Cartesian IK does not yet include "
                    "those added cube geometries"
                ),
            },
            "total_reference_duration_sec": (
                float(max(0, dispatched_reference_count - 1))
                / float(control.contract.timing.ik_hz)
            ),
            "planned_reference_duration_sec": (
                float(len(active_poses_desk) - 1) / float(control.contract.timing.ik_hz)
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
    "MOTION_STATUS_PATH_PLANNER_ERROR",
    "MOTION_STATUS_PATH_SEARCH_EXHAUSTED",
    "MOTION_STATUS_PATH_START_BLOCKED",
    "MOTION_STATUS_PATH_TARGET_BLOCKED",
    "MOTION_STATUS_TRACKING_INCOMPLETE",
    "OBSTACLE_SCENE_FRAME_ID",
    "OBSTACLE_SCENE_FRAME_SPEC_HASH",
    "GraspMotionPlayback",
    "GraspMotionTarget",
    "JointFkReferencePlan",
    "ShadowMujocoInitializationError",
    "ToolTcpFrameBinding",
    "build_collision_checked_joint_fk_reference_plan",
    "build_collision_checked_multiwaypoint_joint_fk_reference_plan",
    "build_obstacle_collision_oracle",
    "build_path_obstacle_collision_oracle",
    "build_shadow_initialized_control",
    "build_shadow_simulation_receipt_from_playback",
    "build_smooth_se3_trajectory",
    "classify_motion_status",
    "derive_tool_tcp_frame_binding",
    "evaluate_motion_target_endpoint",
    "execute_grasp_motion",
    "plan_obstacle_aware_joint_path",
    "pose_from_xyz_quat_xyzw",
    "pose_from_xyz_rpy_deg",
    "quintic_progress",
    "required_motion_duration_sec",
    "revalidate_obstacle_path",
    "select_random_reachable_demo_target",
    "shadow_control_initialization_observation",
]
