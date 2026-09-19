from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.pose import pose6_to_matrix, pose_delta_metrics
from robot_motion.position_control.interpolation import DEFAULT_POSITION_REFERENCE_HZ
from robot_motion.safety_constraints import constrain_joint_vector
from robot_motion.transforms import invert_pose, validate_pose_matrix

from .constants import (
    ARM_DOF,
    ARM_ENDPOINT_FRAME,
    DEFAULT_COLLISION_SCENE,
    LEFT,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RIGHT,
    SIDES,
    TOOL_TCP_FRAME,
)
from .defaults import RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
from .ik import (
    DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN,
    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
    DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
    DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN,
    DEFAULT_DUAL_ARM_POSTURE_COST,
    RM65DualArmCollisionIKConfig,
    build_rm65_dual_arm_collision_ik_solver,
    canonical_rm65_controlled_frame,
)
from .qpos_layout import as_rm65_qpos, normalize_side, validate_qpos_by_side
from .pose_lift_profile import (
    RM65_POSE_LIFT_V1_BASELINE_NAME,
    RM65PoseLiftProfile,
    resolve_rm65_pose_lift_profile,
)
from .wrist_coaxial import canonicalize_rm65_wrist_coaxial_step


DEFAULT_POSITION_COLLISION_REFERENCE_HZ = DEFAULT_POSITION_REFERENCE_HZ
DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S = (
    0.35,
    0.35,
    0.35,
    0.50,
    0.50,
    0.50,
)
DEFAULT_RM65_JACOBIAN_CHARACTERISTIC_LENGTH_M = 0.5
_DEFAULT_MOUNT_JOINT_NAMES = {LEFT: "left_arm", RIGHT: "right_arm"}


class _PlanningFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _finite_positive(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _six_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (ARM_DOF,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain {ARM_DOF} positive finite values.")
    return tuple(float(value) for value in array)


def _pose_matrix(value: Any, *, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception as exc:  # noqa: BLE001 - annotate external pose failures.
        raise ValueError(f"{name} must be a numeric 4x4 pose matrix.") from exc
    valid, reason = validate_pose_matrix(pose)
    if not valid:
        raise ValueError(f"{name} is invalid: {reason}")
    return pose.astype(float, copy=True)


def _se3_exp_local(twist_linear_angular: Sequence[float]) -> np.ndarray:
    """Return a local SE(3) increment in Pinocchio linear/angular order."""

    twist = np.asarray(twist_linear_angular, dtype=float).reshape(6)
    if not np.all(np.isfinite(twist)):
        raise ValueError("local SE(3) increment must be finite.")
    linear = twist[:3]
    angular = twist[3:]
    theta = float(np.linalg.norm(angular))
    skew = np.array(
        [
            [0.0, -angular[2], angular[1]],
            [angular[2], 0.0, -angular[0]],
            [-angular[1], angular[0], 0.0],
        ],
        dtype=float,
    )
    if theta <= 1e-9:
        left_jacobian = (
            np.eye(3, dtype=float)
            + 0.5 * skew
            + (1.0 / 6.0) * (skew @ skew)
        )
    else:
        theta2 = theta * theta
        left_jacobian = (
            np.eye(3, dtype=float)
            + ((1.0 - math.cos(theta)) / theta2) * skew
            + ((theta - math.sin(theta)) / (theta2 * theta))
            * (skew @ skew)
        )
    result = np.eye(4, dtype=float)
    result[:3, :3] = R.from_rotvec(angular).as_matrix()
    result[:3, 3] = left_jacobian @ linear
    return result


def _normalize_pose_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
    require_both: bool,
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping.")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, value in values.items():
        side = normalize_side(str(raw_side))
        if side in normalized:
            raise ValueError(f"{name} contains duplicate side {side!r}.")
        normalized[side] = _pose_matrix(value, name=f"{name}.{side}")
    if require_both and set(normalized) != set(SIDES):
        missing = sorted(set(SIDES) - set(normalized))
        raise ValueError(f"{name} must contain both arms; missing={missing}.")
    return normalized


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _freeze_array_mapping(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for side, value in values.items():
        array = np.asarray(value, dtype=float).copy()
        array.setflags(write=False)
        frozen[str(side)] = array
    return MappingProxyType(frozen)


def _rm65_jacobian_metrics(
    jacobian: np.ndarray,
    *,
    characteristic_length_m: float,
) -> dict[str, Any]:
    """Return dimensionless RM65 task-Jacobian diagnostics.

    Pinocchio exposes motion rows as linear xyz followed by angular xyz.  The
    translational block is divided by one declared characteristic length so
    that its units are comparable with the angular block.  These diagnostics
    are intentionally observational: no universal singularity threshold is
    assigned here because thresholds depend on this exact scaling and model.
    """

    length_m = _finite_positive(
        characteristic_length_m,
        name="characteristic_length_m",
    )
    raw = np.asarray(jacobian, dtype=float)
    if raw.shape != (6, ARM_DOF) or not np.all(np.isfinite(raw)):
        raise ValueError(
            f"arm Jacobian must be finite with shape (6, {ARM_DOF}); "
            f"got {raw.shape}."
        )
    normalized = raw.copy()
    normalized[:3, :] /= length_m
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    sigma_min = float(np.min(singular_values))
    sigma_max = float(np.max(singular_values))
    return {
        "jacobian_row_order": "linear_xyz_then_angular_xyz",
        "jacobian_scaling": "linear_rows_divided_by_characteristic_length",
        "characteristic_length_m": float(length_m),
        "singular_values": singular_values.astype(float).tolist(),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": (
            float("inf") if sigma_min <= 1e-12 else float(sigma_max / sigma_min)
        ),
    }


@dataclass(frozen=True)
class RM65PositionCollisionPlannerConfig:
    """Configuration for one coupled high-rate pose-to-qpos planning stage.

    This configuration intentionally has no pose or qpos temporal-smoothing
    option.  Pose interpolation belongs before this planner, while joint
    interpolation from an accepted reference to high-rate qcmd belongs after
    it.  Joint velocity/step clipping here is a safety bound, not a filter.
    """

    reference_hz: float = DEFAULT_POSITION_COLLISION_REFERENCE_HZ
    scene_profile: str = DEFAULT_COLLISION_SCENE
    solver: str | None = None
    posture_cost: float | None = None
    collision_min_distance_m: float = DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M
    collision_distance_epsilon_m: float = DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M
    collision_barrier_gain: float = DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN
    collision_safe_displacement_gain: float = DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN
    collision_constraints_enabled: bool = True
    # Real measured-qpos admission may refresh the solver's external FK cache.
    # Keep the default false for simulation/replay, whose established planner
    # path owns and recomputes its candidate state explicitly.
    refresh_external_qpos_cache: bool = False
    max_joint_velocity_rad_s: tuple[float, ...] | Sequence[float] = (
        DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S
    )
    max_joint_acceleration_rad_s2: tuple[float, ...] | Sequence[float] = (
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
    )
    max_joint_step_rad: tuple[float, ...] | Sequence[float] | None = None
    joint_lower_rad: tuple[float, ...] | Sequence[float] = RM65_ABSOLUTE_Q_MIN_RAD
    joint_upper_rad: tuple[float, ...] | Sequence[float] = RM65_ABSOLUTE_Q_MAX_RAD
    collision_backoff_scales: tuple[float, ...] | Sequence[float] = (0.5, 0.25, 0.1)
    solve_time_budget_ms: float | None = None
    damping: float = 0.01
    solve_max_iterations: int = 1
    solve_min_iterations: int = 1
    solve_adaptive_iterations_enabled: bool = False
    solve_iteration_reserve_ms: float = 5.0
    solve_translation_tolerance_m: float = 2e-3
    solve_orientation_tolerance_deg: float = 0.5
    pose_lift_profile: str | RM65PoseLiftProfile = RM65_POSE_LIFT_V1_BASELINE_NAME
    debug_profile: bool = False
    ik_tuning: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.collision_constraints_enabled, bool):
            raise TypeError("collision_constraints_enabled must be a bool.")
        if not isinstance(self.refresh_external_qpos_cache, bool):
            raise TypeError("refresh_external_qpos_cache must be a bool.")
        reference_hz = _finite_positive(self.reference_hz, name="reference_hz")
        scene_profile = str(self.scene_profile or DEFAULT_COLLISION_SCENE).strip()
        if not scene_profile:
            raise ValueError("scene_profile must be non-empty.")
        pose_lift_profile = resolve_rm65_pose_lift_profile(self.pose_lift_profile)
        posture_cost = float(
            DEFAULT_DUAL_ARM_POSTURE_COST
            if self.posture_cost is None and pose_lift_profile.ik_posture_cost is None
            else (
                pose_lift_profile.ik_posture_cost
                if self.posture_cost is None
                else self.posture_cost
            )
        )
        if not math.isfinite(posture_cost) or posture_cost < 0.0:
            raise ValueError("posture_cost must be finite and >= 0.")
        min_distance = float(self.collision_min_distance_m)
        epsilon = float(self.collision_distance_epsilon_m)
        if not math.isfinite(min_distance) or min_distance < 0.0:
            raise ValueError("collision_min_distance_m must be finite and >= 0.")
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("collision_distance_epsilon_m must be finite and >= 0.")
        if epsilon > min_distance:
            raise ValueError(
                "collision_distance_epsilon_m must not exceed "
                "collision_min_distance_m."
            )

        lower = np.asarray(self.joint_lower_rad, dtype=float).reshape(-1)
        upper = np.asarray(self.joint_upper_rad, dtype=float).reshape(-1)
        if (
            lower.shape != (ARM_DOF,)
            or upper.shape != (ARM_DOF,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("joint_lower_rad/joint_upper_rad must define 6 finite increasing bounds.")

        scales: list[float] = []
        for raw_scale in self.collision_backoff_scales:
            scale = float(raw_scale)
            if not math.isfinite(scale) or not 0.0 < scale < 1.0:
                raise ValueError("collision_backoff_scales must contain finite values in (0, 1).")
            if not any(abs(scale - existing) <= 1e-12 for existing in scales):
                scales.append(scale)
        scales.sort(reverse=True)
        if not scales:
            raise ValueError("collision_backoff_scales must not be empty.")

        budget = self.solve_time_budget_ms
        if budget is None:
            budget = 1000.0 / reference_hz
        budget = float(budget)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("solve_time_budget_ms must be finite and >= 0.")
        damping = _finite_positive(self.damping, name="damping")
        if (
            isinstance(self.solve_max_iterations, bool)
            or int(self.solve_max_iterations) != self.solve_max_iterations
            or int(self.solve_max_iterations) <= 0
        ):
            raise ValueError("solve_max_iterations must be a positive integer.")
        if (
            isinstance(self.solve_min_iterations, bool)
            or int(self.solve_min_iterations) != self.solve_min_iterations
            or int(self.solve_min_iterations) <= 0
        ):
            raise ValueError("solve_min_iterations must be a positive integer.")
        if int(self.solve_min_iterations) > int(self.solve_max_iterations):
            raise ValueError("solve_min_iterations must be <= solve_max_iterations.")
        reserve_ms = float(self.solve_iteration_reserve_ms)
        if not math.isfinite(reserve_ms) or reserve_ms < 0.0:
            raise ValueError("solve_iteration_reserve_ms must be finite and >= 0.")
        translation_tolerance_m = float(self.solve_translation_tolerance_m)
        orientation_tolerance_deg = float(self.solve_orientation_tolerance_deg)
        if not math.isfinite(translation_tolerance_m) or translation_tolerance_m < 0.0:
            raise ValueError("solve_translation_tolerance_m must be finite and >= 0.")
        if not math.isfinite(orientation_tolerance_deg) or orientation_tolerance_deg < 0.0:
            raise ValueError("solve_orientation_tolerance_deg must be finite and >= 0.")

        object.__setattr__(self, "reference_hz", reference_hz)
        object.__setattr__(self, "scene_profile", scene_profile)
        object.__setattr__(self, "posture_cost", posture_cost)
        object.__setattr__(self, "collision_min_distance_m", min_distance)
        object.__setattr__(self, "collision_distance_epsilon_m", epsilon)
        object.__setattr__(self, "collision_barrier_gain", float(self.collision_barrier_gain))
        object.__setattr__(
            self,
            "collision_safe_displacement_gain",
            float(self.collision_safe_displacement_gain),
        )
        object.__setattr__(
            self,
            "max_joint_velocity_rad_s",
            _six_positive(self.max_joint_velocity_rad_s, name="max_joint_velocity_rad_s"),
        )
        object.__setattr__(
            self,
            "max_joint_acceleration_rad_s2",
            _six_positive(
                self.max_joint_acceleration_rad_s2,
                name="max_joint_acceleration_rad_s2",
            ),
        )
        object.__setattr__(
            self,
            "max_joint_step_rad",
            None
            if self.max_joint_step_rad is None
            else _six_positive(self.max_joint_step_rad, name="max_joint_step_rad"),
        )
        object.__setattr__(self, "joint_lower_rad", tuple(float(value) for value in lower))
        object.__setattr__(self, "joint_upper_rad", tuple(float(value) for value in upper))
        object.__setattr__(self, "collision_backoff_scales", tuple(scales))
        object.__setattr__(self, "solve_time_budget_ms", budget)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "solve_max_iterations", int(self.solve_max_iterations))
        object.__setattr__(self, "solve_min_iterations", int(self.solve_min_iterations))
        object.__setattr__(
            self,
            "solve_adaptive_iterations_enabled",
            bool(self.solve_adaptive_iterations_enabled),
        )
        object.__setattr__(self, "solve_iteration_reserve_ms", reserve_ms)
        object.__setattr__(
            self,
            "solve_translation_tolerance_m",
            translation_tolerance_m,
        )
        object.__setattr__(
            self,
            "solve_orientation_tolerance_deg",
            orientation_tolerance_deg,
        )
        object.__setattr__(
            self,
            "pose_lift_profile",
            pose_lift_profile,
        )
        object.__setattr__(self, "debug_profile", bool(self.debug_profile))

    @property
    def reference_period_sec(self) -> float:
        return 1.0 / float(self.reference_hz)

    @property
    def effective_max_joint_step_rad(self) -> np.ndarray:
        velocity_step = np.asarray(self.max_joint_velocity_rad_s, dtype=float) * self.reference_period_sec
        if self.max_joint_step_rad is None:
            return velocity_step
        return np.minimum(velocity_step, np.asarray(self.max_joint_step_rad, dtype=float))

    @property
    def effective_solve_min_iterations(self) -> int:
        return (
            int(self.pose_lift_profile.qp_substeps_per_reference)
            if self.pose_lift_profile.qp_substeps_per_reference > 0
            else int(self.solve_min_iterations)
        )

    @property
    def effective_solve_max_iterations(self) -> int:
        return (
            int(self.pose_lift_profile.qp_substeps_per_reference)
            if self.pose_lift_profile.qp_substeps_per_reference > 0
            else int(self.solve_max_iterations)
        )


@dataclass(frozen=True)
class RM65PositionCollisionPlanResult:
    sequence: int
    source_timestamp: float | None
    source: str
    status: Literal["tracking", "limited", "collision_backoff", "hold_last_safe"]
    valid: bool
    hold: bool
    active_sides: tuple[str, ...]
    q_ik_raw_by_side: Mapping[str, np.ndarray]
    q_ref_by_side: Mapping[str, np.ndarray]
    target_endpoint_pose_world_by_side: Mapping[str, np.ndarray]
    realized_endpoint_pose_world_by_side: Mapping[str, np.ndarray]
    modification_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or int(self.sequence) != self.sequence or int(self.sequence) < 0:
            raise ValueError("sequence must be a non-negative integer.")
        source_timestamp = self.source_timestamp
        if source_timestamp is not None and not math.isfinite(float(source_timestamp)):
            raise ValueError("source_timestamp must be finite when provided.")
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must be non-empty.")
        sides = tuple(normalize_side(side) for side in self.active_sides)
        if not sides or len(set(sides)) != len(sides):
            raise ValueError("active_sides must contain unique RM65 sides.")
        q_ref = {
            side: as_rm65_qpos(value, label=f"q_ref_by_side.{side}")
            for side, value in self.q_ref_by_side.items()
        }
        if set(q_ref) != set(SIDES):
            raise ValueError("q_ref_by_side must contain a complete dual-arm hold/reference.")

        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(
            self,
            "source_timestamp",
            None if source_timestamp is None else float(source_timestamp),
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "hold", bool(self.hold))
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "q_ik_raw_by_side", _freeze_array_mapping(self.q_ik_raw_by_side))
        object.__setattr__(self, "q_ref_by_side", _freeze_array_mapping(q_ref))
        object.__setattr__(
            self,
            "target_endpoint_pose_world_by_side",
            _freeze_array_mapping(self.target_endpoint_pose_world_by_side),
        )
        object.__setattr__(
            self,
            "realized_endpoint_pose_world_by_side",
            _freeze_array_mapping(self.realized_endpoint_pose_world_by_side),
        )
        object.__setattr__(
            self,
            "modification_reasons",
            tuple(dict.fromkeys(str(reason) for reason in self.modification_reasons if str(reason))),
        )
        object.__setattr__(self, "metadata", _freeze_value(dict(self.metadata)))


@dataclass(frozen=True)
class RM65QposCollisionValidationResult:
    """Read-only endpoint validation for an explicit qpos reference.

    This check does not mutate the planner's last-safe state.  It is intended
    for low-level replay modes that already own their qpos/reference timeline.
    """

    safe: bool
    reason: str | None
    active_sides: tuple[str, ...]
    qpos_by_side: Mapping[str, np.ndarray]
    collision_distance_m: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        qpos = {
            side: as_rm65_qpos(value, label=f"qpos_by_side.{side}")
            for side, value in self.qpos_by_side.items()
        }
        if set(qpos) != set(SIDES):
            raise ValueError("qpos collision validation must contain both arms.")
        object.__setattr__(self, "safe", bool(self.safe))
        object.__setattr__(self, "reason", None if self.reason is None else str(self.reason))
        object.__setattr__(
            self,
            "active_sides",
            tuple(normalize_side(side) for side in self.active_sides),
        )
        object.__setattr__(self, "qpos_by_side", _freeze_array_mapping(qpos))
        object.__setattr__(
            self,
            "collision_distance_m",
            None
            if self.collision_distance_m is None
            else float(self.collision_distance_m),
        )
        object.__setattr__(self, "metadata", _freeze_value(dict(self.metadata)))


class RM65PositionPlannerProtocol(Protocol):
    """Device-neutral boundary consumed by real, replay, and sim runtimes.

    Implementations own coupled pose-to-qref planning state, hard joint
    constraints, endpoint collision certification, and last-safe recovery.
    They must not own transport threads, vendor SDK calls, or device lifecycle.
    """

    @property
    def collision_pair_count(self) -> int:
        ...

    def reset(self, current_qpos_by_side: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def plan_pose_reference(
        self,
        target_pose_by_side: Mapping[str, Any],
        *,
        target_reference_frame: Literal["arm_base", "world", "scene_world"] = "arm_base",
        target_controlled_frame: str = TOOL_TCP_FRAME,
        source_timestamp: float | None = None,
        source: str = "position_reference",
        metadata: Mapping[str, Any] | None = None,
    ) -> RM65PositionCollisionPlanResult:
        ...

    def admit_qpos_reference(
        self,
        qpos_by_side: Mapping[str, Any],
        *,
        source: str = "qpos_reference",
    ) -> RM65QposCollisionValidationResult:
        ...

    def validate_qpos_reference(
        self,
        qpos_by_side: Mapping[str, Any],
        *,
        source: str = "qpos_reference",
    ) -> RM65QposCollisionValidationResult:
        ...


def _parse_numeric_tuple(text: str | None, *, count: int, label: str) -> tuple[float, ...]:
    raw = "" if text is None else str(text).strip()
    values = (0.0,) * count if not raw else tuple(float(value) for value in raw.split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain {count} finite values.")
    return values


def _urdf_joint_origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = _parse_numeric_tuple(
        None if origin is None else origin.get("xyz"),
        count=3,
        label=f"joint {joint.get('name')!r} origin xyz",
    )
    rpy = _parse_numeric_tuple(
        None if origin is None else origin.get("rpy"),
        count=3,
        label=f"joint {joint.get('name')!r} origin rpy",
    )
    return pose6_to_matrix((*xyz, *rpy))


def _load_urdf_kinematic_graph(
    scene_profile_or_urdf: Any,
) -> tuple[
    dict[str, ET.Element],
    frozenset[str],
    dict[str, tuple[str, np.ndarray, str, str]],
]:
    urdf_value = getattr(scene_profile_or_urdf, "urdf_path", scene_profile_or_urdf)
    urdf_path = Path(urdf_value).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"RM65 collision scene URDF does not exist: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    links = frozenset(
        str(link.get("name") or "").strip()
        for link in root.findall("link")
        if str(link.get("name") or "").strip()
    )
    joints_by_name: dict[str, ET.Element] = {}
    parent_joint_by_child: dict[str, tuple[str, np.ndarray, str, str]] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name") or "").strip()
        if name:
            if name in joints_by_name:
                raise ValueError(f"URDF contains duplicate joint name {name!r}.")
            joints_by_name[name] = joint
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = str(parent.get("link") or "").strip()
        child_link = str(child.get("link") or "").strip()
        if not parent_link or not child_link:
            continue
        if child_link in parent_joint_by_child:
            raise ValueError(f"URDF link {child_link!r} has multiple parent joints.")
        joint_type = str(joint.get("type") or "").strip()
        parent_joint_by_child[child_link] = (
            parent_link,
            (
                _urdf_joint_origin(joint)
                if joint_type == "fixed"
                else np.eye(4, dtype=float)
            ),
            name,
            joint_type,
        )
    return joints_by_name, links, parent_joint_by_child


def _world_from_fixed_link(
    link_name: str,
    *,
    links: frozenset[str],
    parent_joint_by_child: Mapping[str, tuple[str, np.ndarray, str, str]],
    cache: dict[str, np.ndarray],
    stack: tuple[str, ...] = (),
) -> np.ndarray:
    if link_name not in links:
        raise ValueError(f"Scene URDF is missing link {link_name!r}.")
    if link_name in cache:
        return cache[link_name].copy()
    if link_name in stack:
        raise ValueError(f"Cycle detected in fixed URDF chain: {(*stack, link_name)}")
    edge = parent_joint_by_child.get(link_name)
    if edge is None:
        raise ValueError(
            f"Cannot resolve world transform for link {link_name!r}; "
            "its parent chain is missing."
        )
    parent_link, parent_from_child, joint_name, joint_type = edge
    if joint_type != "fixed":
        raise ValueError(
            f"Cannot resolve world transform for link {link_name!r}; "
            f"parent joint {joint_name!r} is not fixed."
        )
    pose = _world_from_fixed_link(
        parent_link,
        links=links,
        parent_joint_by_child=parent_joint_by_child,
        cache=cache,
        stack=(*stack, link_name),
    ) @ parent_from_child
    pose = _pose_matrix(pose, name=f"world_from_link[{link_name}]")
    cache[link_name] = pose.copy()
    return pose


def load_world_from_fixed_link_transform(
    scene_profile_or_urdf: Any,
    *,
    link_name: str,
) -> np.ndarray:
    """Resolve ``T_world_link`` through one all-fixed scene-URDF chain."""

    normalized_link = str(link_name).strip()
    if not normalized_link:
        raise ValueError("link_name must be non-empty.")
    _joints, links, parent_joint_by_child = _load_urdf_kinematic_graph(
        scene_profile_or_urdf
    )
    return _world_from_fixed_link(
        normalized_link,
        links=links,
        parent_joint_by_child=parent_joint_by_child,
        cache={"world": np.eye(4, dtype=float)},
    )


def load_world_from_arm_base_transforms(
    scene_profile_or_urdf: Any,
    *,
    mount_joint_names: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Parse exact world-to-arm-base transforms from the scene URDF.

    The legacy scene profile exposes orientation-only ``left_pose_transform``
    and ``right_pose_transform`` matrices.  They omit the physical base
    translation and are therefore deliberately not consulted here.
    """

    joints_by_name, links, parent_joint_by_child = _load_urdf_kinematic_graph(
        scene_profile_or_urdf
    )

    requested = dict(_DEFAULT_MOUNT_JOINT_NAMES if mount_joint_names is None else mount_joint_names)
    normalized_mounts: dict[str, str] = {}
    for raw_side, joint_name in requested.items():
        normalized_mounts[normalize_side(str(raw_side))] = str(joint_name).strip()
    if set(normalized_mounts) != set(SIDES):
        raise ValueError("mount_joint_names must contain left and right mount joints.")

    cache: dict[str, np.ndarray] = {"world": np.eye(4, dtype=float)}

    transforms: dict[str, np.ndarray] = {}
    for side in SIDES:
        mount_name = normalized_mounts[side]
        mount = joints_by_name.get(mount_name)
        if mount is None:
            raise ValueError(f"Scene URDF is missing {side} mount joint {mount_name!r}.")
        if str(mount.get("type") or "").strip() != "fixed":
            raise ValueError(f"Scene mount joint {mount_name!r} must be fixed.")
        child = mount.find("child")
        child_link = "" if child is None else str(child.get("link") or "").strip()
        if not child_link:
            raise ValueError(f"Scene mount joint {mount_name!r} has no child base link.")
        transforms[side] = _world_from_fixed_link(
            child_link,
            links=links,
            parent_joint_by_child=parent_joint_by_child,
            cache=cache,
        )
    return transforms


class RM65PositionCollisionPlanner:
    """Coupled, sim/real-neutral RM65 pose-to-qpos planner.

    Inputs are already-resampled high-rate reference poses.  The
    planner uses one full dual-arm collision model, limits the resulting arm
    qpos, and rechecks collision distance on the final limited endpoint.  It
    does not generate the downstream 120 Hz qcmd and performs no temporal
    smoothing.
    """

    def __init__(
        self,
        config: RM65PositionCollisionPlannerConfig | None = None,
        *,
        endpoint_to_tool_by_side: Mapping[str, Any],
        world_from_arm_base_by_side: Mapping[str, Any] | None = None,
        bundle: Any | None = None,
    ) -> None:
        self.config = config if config is not None else RM65PositionCollisionPlannerConfig()
        if bundle is None:
            bundle = build_rm65_dual_arm_collision_ik_solver(
                RM65DualArmCollisionIKConfig(
                    scene_profile=self.config.scene_profile,
                    solver=self.config.solver,
                    solver_frequency=float(self.config.reference_hz),
                    damping=float(self.config.damping),
                    posture_cost=float(self.config.posture_cost),
                    collision_min_distance_m=float(self.config.collision_min_distance_m),
                    collision_distance_epsilon_m=float(
                        self.config.collision_distance_epsilon_m
                    ),
                    collision_barrier_gain=float(self.config.collision_barrier_gain),
                    collision_safe_displacement_gain=float(
                        self.config.collision_safe_displacement_gain
                    ),
                    collision_constraints_enabled=(
                        self.config.collision_constraints_enabled
                    ),
                    debug_profile=bool(self.config.debug_profile),
                    ik_tuning=self.config.ik_tuning,
                )
            )
        self.bundle = bundle
        self.solver = getattr(bundle, "solver")
        self.scene_profile = getattr(bundle, "scene_profile")
        self.endpoint_to_tool_by_side = _freeze_array_mapping(
            _normalize_pose_mapping(
                endpoint_to_tool_by_side,
                name="endpoint_to_tool_by_side",
                require_both=True,
            )
        )
        if world_from_arm_base_by_side is None:
            self.world_from_arm_base_by_side = _freeze_array_mapping(
                load_world_from_arm_base_transforms(self.scene_profile)
            )
            self.world_from_arm_base_source = "scene_urdf_fixed_joint_chain"
        else:
            self.world_from_arm_base_by_side = _freeze_array_mapping(
                _normalize_pose_mapping(
                    world_from_arm_base_by_side,
                    name="world_from_arm_base_by_side",
                    require_both=True,
                )
            )
            self.world_from_arm_base_source = "explicit"

        current_full_q = self._solver_current_q()
        self._arm_slices = self._resolve_arm_slices(current_full_q.size)
        self._last_admitted_velocity_full = np.zeros_like(current_full_q)
        self._configure_qp_motion_constraints(current_full_q)
        self._joint_lower_by_side, self._joint_upper_by_side = self._effective_joint_limits(
            current_full_q
        )
        self._collision_pair_count = self._resolve_collision_pair_count()
        self._lock = threading.RLock()
        self._last_safe_full_q: np.ndarray | None = None
        self._last_safe_collision_distance_m: float | None = None
        self._next_sequence = 0
        self._last_pose_lift_metadata: dict[str, Any] = {
            "profile": self.config.pose_lift_profile.to_metadata(),
            "applied": False,
            "reason": "not_planned",
        }
        self._last_nonlinear_candidate_metadata: dict[str, Any] = {
            "enabled": bool(
                self.config.pose_lift_profile.qp_nonlinear_best_so_far
            ),
            "reason": "not_planned",
            "candidates": [],
        }
        self._directional_task_guard_active_by_side = {
            side: False for side in SIDES
        }
        self._last_directional_task_guard_metadata: dict[str, Any] = {
            "enabled": bool(
                self.config.pose_lift_profile.directional_task_guard_max_utilization
                > 0.0
            ),
            "reason": "not_planned",
            "sides": {},
        }

    @property
    def collision_pair_count(self) -> int:
        return int(self._collision_pair_count)

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._last_safe_full_q is not None

    @property
    def last_safe_qpos_by_side(self) -> Mapping[str, np.ndarray] | None:
        with self._lock:
            if self._last_safe_full_q is None:
                return None
            return _freeze_array_mapping(self._split_arm_qpos(self._last_safe_full_q))

    def analyze_configuration(
        self,
        qpos_by_side: Mapping[str, Any],
        *,
        characteristic_length_m: float = DEFAULT_RM65_JACOBIAN_CHARACTERISTIC_LENGTH_M,
    ) -> Mapping[str, Any]:
        """Observe Jacobian and RM65 structural-singularity metrics for qpos.

        This method performs no planning, admission, reference advancement, or
        dispatch.  It exists so offline parity tools do not reach through the
        planner into private solver state.  The solver's Pinocchio data cache is
        evaluated while holding the planner lock; authoritative qpos and
        last-safe state are not changed.
        """

        length_m = _finite_positive(
            characteristic_length_m,
            name="characteristic_length_m",
        )
        qpos = validate_qpos_by_side(qpos_by_side)
        if set(qpos) != set(SIDES):
            missing = sorted(set(SIDES) - set(qpos))
            raise ValueError(
                "configuration diagnostics require qpos for both arms; "
                f"missing={missing}."
            )
        with self._lock:
            full_q = self._embed_arm_qpos(self._solver_current_q(), qpos)
            sides = self._configuration_singularity_metrics(
                full_q,
                characteristic_length_m=length_m,
            )
            return _freeze_value(
                {
                    "schema": "robot_motion.rm65.configuration_diagnostics/v1",
                    "observational_only": True,
                    "threshold_classification": "not_assigned",
                    "sides": sides,
                }
            )

    def observe_endpoint_poses_world(
        self,
        qpos_by_side: Mapping[str, Any],
    ) -> Mapping[str, np.ndarray]:
        """Return detached endpoint FK for a complete dual-arm configuration.

        This observer is intentionally read-only: it does not reset the
        planner, advance the Pink configuration, alter the admitted-motion
        history, or certify the configuration for execution.  Offline
        path-lift diagnostics use it to construct targets around recorded
        accepted-qref seeds without reaching through the planner into its
        private solver.
        """

        qpos = validate_qpos_by_side(qpos_by_side)
        if set(qpos) != set(SIDES):
            missing = sorted(set(SIDES) - set(qpos))
            raise ValueError(
                "endpoint FK observation requires qpos for both arms; "
                f"missing={missing}."
            )
        with self._lock:
            full_q = self._embed_arm_qpos(self._solver_current_q(), qpos)
            return _freeze_array_mapping(self._realized_endpoint_poses(full_q))

    def analyze_task_conditioning(
        self,
        qpos_by_side: Mapping[str, Any],
        target_endpoint_pose_world_by_side: Mapping[str, Any],
        *,
        dt_sec: float | None = None,
    ) -> Mapping[str, Any]:
        """Observe target-conditioned Pink task authority for one reference.

        Unlike :meth:`analyze_configuration`, this metric depends on both the
        candidate configuration and the exact endpoint target.  It calls the
        same Pink ``FrameTask`` SE(3)-Log error and Jacobian implementation as
        the planner, but uses an isolated temporary configuration and never
        advances solver, motion-limit, collision, or admitted-reference state.
        """

        period = (
            float(self.config.reference_period_sec)
            if dt_sec is None
            else _finite_positive(dt_sec, name="dt_sec")
        )
        qpos = validate_qpos_by_side(qpos_by_side)
        if set(qpos) != set(SIDES):
            missing = sorted(set(SIDES) - set(qpos))
            raise ValueError(
                "task-conditioning diagnostics require qpos for both arms; "
                f"missing={missing}."
            )
        targets = _normalize_pose_mapping(
            target_endpoint_pose_world_by_side,
            name="target_endpoint_pose_world_by_side",
            require_both=True,
        )
        analyze = getattr(self.solver, "compute_frame_task_conditioning", None)
        if not callable(analyze):
            return _freeze_value(
                {
                    "schema": "robot_motion.rm65.task_conditioning/v1",
                    "available": False,
                    "observational_only": True,
                    "reason": "solver_frame_task_conditioning_unavailable",
                    "sides": {},
                }
            )
        with self._lock:
            current_full_q = self._solver_current_q()
            full_q = self._embed_arm_qpos(current_full_q, qpos)
            full_velocity_limit = self._full_arm_tangent_limit(
                self.config.max_joint_velocity_rad_s,
                full_dof=full_q.size,
            )
            joint_indices = [
                np.arange(full_q.size, dtype=int)[self._arm_slices[side]]
                for side in SIDES
            ]
            solver_targets = [
                self._pose_to_xyzquat(targets[side]) for side in SIDES
            ]
            metrics = analyze(
                solver_targets,
                q=full_q,
                joint_indices_by_target=joint_indices,
                joint_velocity_limit_rad_s=full_velocity_limit,
                dt_sec=period,
            )
            if len(metrics) != len(SIDES):
                raise RuntimeError(
                    "solver task-conditioning result does not match arm count."
                )
            return _freeze_value(
                {
                    "schema": "robot_motion.rm65.task_conditioning/v1",
                    "available": True,
                    "observational_only": True,
                    "control_authority": "none",
                    "threshold_classification": "not_assigned",
                    "target_contract": "exact_solver_endpoint_pose_world",
                    "sides": {
                        side: dict(side_metrics)
                        for side, side_metrics in zip(SIDES, metrics)
                    },
                }
            )

    def analyze_constrained_task_feasibility(
        self,
        qpos_by_side: Mapping[str, Any],
        target_endpoint_pose_world_by_side: Mapping[str, Any],
        *,
        dt_sec: float | None = None,
    ) -> Mapping[str, Any]:
        """Run detached feasibility QPs for one exact endpoint reference."""

        period = (
            float(self.config.reference_period_sec)
            if dt_sec is None
            else _finite_positive(dt_sec, name="dt_sec")
        )
        qpos = validate_qpos_by_side(qpos_by_side)
        if set(qpos) != set(SIDES):
            missing = sorted(set(SIDES) - set(qpos))
            raise ValueError(
                "constrained-feasibility diagnostics require both arms; "
                f"missing={missing}."
            )
        targets = _normalize_pose_mapping(
            target_endpoint_pose_world_by_side,
            name="target_endpoint_pose_world_by_side",
            require_both=True,
        )
        analyze = getattr(self.solver, "compute_constrained_task_feasibility", None)
        if not callable(analyze):
            return _freeze_value(
                {
                    "schema": (
                        "robot_motion.rm65.constrained_task_feasibility/v1"
                    ),
                    "available": False,
                    "observational_only": True,
                    "reason": "solver_constrained_task_feasibility_unavailable",
                }
            )
        with self._lock:
            full_q = self._embed_arm_qpos(self._solver_current_q(), qpos)
            result = analyze(
                [self._pose_to_xyzquat(targets[side]) for side in SIDES],
                q=full_q,
                dt_sec=period,
            )
            return _freeze_value(dict(result))

    def resolve_target_endpoint_pose_world(
        self,
        target_pose_by_side: Mapping[str, Any],
        *,
        target_reference_frame: Literal["arm_base", "world", "scene_world"] = (
            "arm_base"
        ),
        target_controlled_frame: str = TOOL_TCP_FRAME,
    ) -> Mapping[str, np.ndarray]:
        """Resolve planner input poses without planning or advancing state."""

        targets = _normalize_pose_mapping(
            target_pose_by_side,
            name="target_pose_by_side",
            require_both=True,
        )
        reference_frame = str(target_reference_frame).strip().lower()
        if reference_frame == "scene_world":
            reference_frame = "world"
        if reference_frame not in {"arm_base", "world"}:
            raise ValueError(
                f"unsupported target reference frame {target_reference_frame!r}."
            )
        controlled_frame = canonical_rm65_controlled_frame(
            target_controlled_frame
        )
        if controlled_frame not in {TOOL_TCP_FRAME, ARM_ENDPOINT_FRAME}:
            raise ValueError(
                f"unsupported target controlled frame {target_controlled_frame!r}."
            )
        endpoint_from_controlled_by_side = {
            side: (
                self.endpoint_to_tool_by_side[side]
                if controlled_frame == TOOL_TCP_FRAME
                else np.eye(4, dtype=float)
            )
            for side in SIDES
        }
        return _freeze_array_mapping(
            {
                side: _pose_matrix(
                    (
                        self.world_from_arm_base_by_side[side] @ targets[side]
                        if reference_frame == "arm_base"
                        else targets[side]
                    )
                    @ invert_pose(endpoint_from_controlled_by_side[side]),
                    name=f"target_endpoint_pose_world_by_side.{side}",
                )
                for side in SIDES
            }
        )

    def _directionally_shape_solver_targets(
        self,
        reference_full_q: np.ndarray,
        target_endpoint_world: Mapping[str, np.ndarray],
        *,
        active_sides: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """Project only over-demanded Pink singular directions.

        This path is gated by an explicit offline-only Pose Lift profile.
        Original target poses remain authoritative for error reporting; the
        returned poses are solver-local shaped targets for one control cut.
        """

        profile = self.config.pose_lift_profile
        max_utilization = float(
            profile.directional_task_guard_max_utilization
        )
        shaped = {
            side: np.asarray(target_endpoint_world[side], dtype=float).copy()
            for side in SIDES
        }
        if max_utilization <= 0.0:
            self._last_directional_task_guard_metadata = {
                "enabled": False,
                "reason": "profile_disabled",
                "sides": {},
            }
            return shaped
        analyze = getattr(self.solver, "compute_frame_task_conditioning", None)
        if not callable(analyze):
            raise _PlanningFailure(
                "directional_task_guard_unavailable",
                "directional task guard requires Pink FrameTask diagnostics.",
            )
        full_q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        full_velocity_limit = self._full_arm_tangent_limit(
            self.config.max_joint_velocity_rad_s,
            full_dof=full_q.size,
        )
        joint_indices = [
            np.arange(full_q.size, dtype=int)[self._arm_slices[side]]
            for side in SIDES
        ]
        metrics = analyze(
            [self._pose_to_xyzquat(shaped[side]) for side in SIDES],
            q=full_q,
            joint_indices_by_target=joint_indices,
            joint_velocity_limit_rad_s=full_velocity_limit,
            dt_sec=float(self.config.reference_period_sec),
        )
        if len(metrics) != len(SIDES):
            raise _PlanningFailure(
                "directional_task_guard_invalid_metrics",
                "directional task metrics do not match the dual-arm layout.",
            )
        active_set = {normalize_side(str(side)) for side in active_sides}
        qpos_by_side = self._split_arm_qpos(full_q)
        metadata_by_side: dict[str, Any] = {}
        shaped_count = 0
        for side, side_metrics in zip(SIDES, metrics):
            qpos = qpos_by_side[side]
            structural_indices = (2, 4)
            selected_joint = min(
                structural_indices,
                key=lambda index: abs(float(qpos[index])),
            )
            structural_abs_deg = float(
                abs(math.degrees(float(qpos[selected_joint])))
            )
            max_required = float(
                side_metrics["max_required_velocity_utilization"]
            )
            was_active = bool(
                self._directional_task_guard_active_by_side.get(side, False)
            )
            if side not in active_set:
                guard_active = False
                transition = "passive_side"
            elif was_active:
                release = bool(
                    max_required
                    <= float(
                        profile.directional_task_guard_release_utilization
                    )
                    or structural_abs_deg
                    >= float(
                        profile.directional_task_guard_release_band_deg
                    )
                )
                guard_active = not release
                transition = "released" if release else "protected"
            else:
                guard_active = bool(
                    structural_abs_deg
                    <= float(
                        profile.directional_task_guard_structural_band_deg
                    )
                    and max_required
                    >= float(
                        profile.directional_task_guard_enter_utilization
                    )
                )
                transition = "entered" if guard_active else "clear"
            self._directional_task_guard_active_by_side[side] = guard_active
            side_metadata: dict[str, Any] = {
                "active": guard_active,
                "transition": transition,
                "max_required_velocity_utilization": max_required,
                "selected_structural_joint": f"q{selected_joint + 1}",
                "selected_structural_abs_deg": structural_abs_deg,
                "task_sigma_min": float(
                    side_metrics["sigma_task_velocity_min"]
                ),
                "direction_scales": [1.0] * 6,
                "away_direction_modes": [],
            }
            if not guard_active:
                metadata_by_side[side] = side_metadata
                continue

            task_error = np.asarray(
                side_metrics["task_error_local"],
                dtype=float,
            ).reshape(6)
            task_cost = np.asarray(
                side_metrics["task_cost"],
                dtype=float,
            ).reshape(6)
            task_direction_rows = np.asarray(
                side_metrics["task_direction_rows"],
                dtype=float,
            ).reshape(6, 6)
            joint_direction_rows = np.asarray(
                side_metrics["joint_utilization_direction_rows"],
                dtype=float,
            ).reshape(6, 6)
            singular_values = np.asarray(
                side_metrics["singular_values_task_velocity"],
                dtype=float,
            ).reshape(6)
            directional_demand = np.asarray(
                side_metrics["directional_demand"],
                dtype=float,
            ).reshape(6)
            required = np.asarray(
                side_metrics["directional_required_velocity_utilization"],
                dtype=float,
            ).reshape(6)
            velocity_limit = np.asarray(
                side_metrics["joint_velocity_limit_rad_s"],
                dtype=float,
            ).reshape(6)
            scales = np.ones(6, dtype=float)
            away_modes: list[int] = []
            selected_sign = (
                1.0 if float(qpos[selected_joint]) >= 0.0 else -1.0
            )
            for mode_index in range(6):
                sigma = float(singular_values[mode_index])
                demand = float(directional_demand[mode_index])
                modal_joint_velocity = np.zeros(6, dtype=float)
                if sigma > 1e-12:
                    modal_joint_velocity = (
                        velocity_limit
                        * joint_direction_rows[mode_index]
                        * (demand / sigma)
                    )
                leaving = bool(
                    selected_sign
                    * float(modal_joint_velocity[selected_joint])
                    > 1e-9
                )
                if leaving:
                    away_modes.append(mode_index)
                cap = max_utilization * (
                    float(profile.directional_task_guard_leaving_multiplier)
                    if leaving
                    else 1.0
                )
                mode_required = float(required[mode_index])
                if not math.isfinite(mode_required):
                    scales[mode_index] = float(
                        profile.directional_task_guard_min_scale
                    )
                elif mode_required > cap:
                    scales[mode_index] = max(
                        float(profile.directional_task_guard_min_scale),
                        cap / max(mode_required, 1e-12),
                    )
            weighted_error = task_cost * task_error
            modal_error = task_direction_rows @ weighted_error
            safe_weighted_error = (
                task_direction_rows.T @ (scales * modal_error)
            )
            safe_error = safe_weighted_error / task_cost
            current_pose = self._compute_frame_pose(
                str(getattr(self.scene_profile, f"{side}_frame_name")),
                full_q,
            )
            shaped[side] = _pose_matrix(
                current_pose @ _se3_exp_local(safe_error),
                name=f"directionally_shaped_target.{side}",
            )
            shaped_count += int(np.any(scales < 1.0 - 1e-12))
            side_metadata.update(
                {
                    "direction_scales": scales.astype(float).tolist(),
                    "away_direction_modes": away_modes,
                    "raw_task_error_local": task_error.astype(float).tolist(),
                    "shaped_task_error_local": safe_error.astype(float).tolist(),
                    "minimum_direction_scale": float(np.min(scales)),
                }
            )
            metadata_by_side[side] = side_metadata
        self._last_directional_task_guard_metadata = {
            "enabled": True,
            "reason": (
                "directionally_shaped"
                if shaped_count > 0
                else "no_direction_required_shaping"
            ),
            "shaped_side_count": shaped_count,
            "max_utilization": max_utilization,
            "enter_utilization": float(
                profile.directional_task_guard_enter_utilization
            ),
            "release_utilization": float(
                profile.directional_task_guard_release_utilization
            ),
            "sides": metadata_by_side,
        }
        return shaped

    def _configuration_singularity_metrics(
        self,
        full_q: np.ndarray,
        *,
        characteristic_length_m: float,
    ) -> dict[str, dict[str, Any]]:
        q = np.asarray(full_q, dtype=float).reshape(-1)
        if q.shape != self._solver_current_q().shape or not np.all(np.isfinite(q)):
            raise ValueError("diagnostic full qpos must match the finite solver layout.")
        sides: dict[str, dict[str, Any]] = {}
        arm_qpos = self._split_arm_qpos(q)
        for side in SIDES:
            frame_name = str(getattr(self.scene_profile, f"{side}_frame_name"))
            jacobian_full = np.asarray(
                self.solver.compute_frame_jacobian(frame_name, q),
                dtype=float,
            )
            metrics = _rm65_jacobian_metrics(
                jacobian_full[:, self._arm_slices[side]],
                characteristic_length_m=characteristic_length_m,
            )
            arm_q = arm_qpos[side]
            abs_q3_deg = abs(float(math.degrees(arm_q[2])))
            abs_q5_deg = abs(float(math.degrees(arm_q[4])))
            metrics.update(
                {
                    "abs_q3_deg": abs_q3_deg,
                    "abs_q5_deg": abs_q5_deg,
                    "min_abs_q3_or_q5_deg": min(abs_q3_deg, abs_q5_deg),
                }
            )
            sides[side] = metrics
        return sides

    def _evaluate_nonlinear_iteration_candidate(
        self,
        *,
        iteration: int,
        candidate_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        pose_error_by_side: Mapping[str, Mapping[str, float]],
    ) -> dict[str, Any]:
        """Evaluate one SQP iterate using final nonlinear admission geometry.

        This observer is enabled only by an explicit offline Pose Lift profile.
        It does not advance the admitted reference or solver history.
        """

        candidate = np.asarray(candidate_full_q, dtype=float).reshape(
            reference_full_q.shape
        )
        result: dict[str, Any] = {
            "iteration": int(iteration),
            "eligible": False,
            "reason": "pending",
        }
        try:
            self._require_arm_qpos_within_limits(
                candidate,
                name=f"nonlinear iteration {iteration} qpos",
            )
            candidate_by_side = self._split_arm_qpos(candidate)
            reference_by_side = self._split_arm_qpos(reference_full_q)
            max_step = self.config.effective_max_joint_step_rad
            for side in SIDES:
                constrained = constrain_joint_vector(
                    candidate_by_side[side],
                    reference=reference_by_side[side],
                    lower=self._joint_lower_by_side[side],
                    upper=self._joint_upper_by_side[side],
                    max_step_rad=max_step,
                )
                if constrained.infeasible or constrained.value is None:
                    result["reason"] = f"joint_certificate_infeasible:{side}"
                    return result
                if not np.allclose(
                    np.asarray(constrained.value, dtype=float),
                    candidate_by_side[side],
                    atol=1e-9,
                    rtol=0.0,
                ):
                    result["reason"] = f"joint_certificate_modified:{side}"
                    return result

            distance, distance_ms, distance_details = self._collision_distance(
                candidate
            )
            collision_safe, collision_reason = self._collision_safe(distance)
            if not collision_safe:
                result.update(
                    {
                        "reason": str(collision_reason or "collision_unsafe"),
                        "collision_distance_m": distance,
                        "collision_check_ms": float(distance_ms),
                        **distance_details,
                    }
                )
                return result

            singularity = self._configuration_singularity_metrics(
                candidate,
                characteristic_length_m=(
                    DEFAULT_RM65_JACOBIAN_CHARACTERISTIC_LENGTH_M
                ),
            )
            min_sigma = min(
                float(singularity[side]["sigma_min"]) for side in SIDES
            )
            translation_scale = max(
                float(self.config.solve_translation_tolerance_m),
                1e-12,
            )
            rotation_scale = max(
                float(self.config.solve_orientation_tolerance_deg),
                1e-12,
            )
            pose_score = max(
                max(
                    float(pose_error_by_side[side]["translation_m"])
                    / translation_scale,
                    float(pose_error_by_side[side]["rotation_deg"])
                    / rotation_scale,
                )
                for side in SIDES
            )
            result.update(
                {
                    "eligible": True,
                    "reason": "exact_nonlinear_certificate_passed",
                    "min_sigma": float(min_sigma),
                    "pose_score": float(pose_score),
                    "collision_distance_m": distance,
                    "collision_check_ms": float(distance_ms),
                    "pose_error": {
                        side: {
                            "translation_m": float(
                                pose_error_by_side[side]["translation_m"]
                            ),
                            "rotation_deg": float(
                                pose_error_by_side[side]["rotation_deg"]
                            ),
                        }
                        for side in SIDES
                    },
                    "_candidate_full_q": candidate.copy(),
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001 - candidate is rejected, not fatal.
            result["reason"] = (
                f"candidate_evaluation_error:{type(exc).__name__}:{exc}"
            )
            return result

    def _nonlinear_iteration_candidate_is_better(
        self,
        candidate: Mapping[str, Any],
        best: Mapping[str, Any],
    ) -> tuple[bool, str]:
        profile = self.config.pose_lift_profile
        candidate_sigma = float(candidate["min_sigma"])
        best_sigma = float(best["min_sigma"])
        candidate_pose = float(candidate["pose_score"])
        best_pose = float(best["pose_score"])
        sigma_improvement = float(
            profile.qp_nonlinear_best_sigma_improvement
        )
        pose_slack = float(profile.qp_nonlinear_best_pose_score_slack)
        guarded = min(candidate_sigma, best_sigma) < float(
            profile.qp_nonlinear_best_guard_sigma
        )
        if guarded:
            if (
                candidate_sigma >= best_sigma + sigma_improvement
                and candidate_pose <= best_pose + pose_slack
            ):
                return True, "guarded_sigma_improved_within_pose_slack"
            if (
                candidate_sigma >= best_sigma - 1e-9
                and candidate_pose < best_pose - 1e-9
            ):
                return True, "guarded_pose_improved_without_sigma_regression"
            return False, "guarded_candidate_not_lexicographically_better"
        if candidate_pose < best_pose - 1e-9:
            return True, "clear_pose_improved"
        if (
            candidate_pose <= best_pose + 1e-9
            and candidate_sigma > best_sigma + sigma_improvement
        ):
            return True, "clear_equal_pose_sigma_improved"
        return False, "clear_candidate_not_lexicographically_better"

    def reset(self, current_qpos_by_side: Mapping[str, Any]) -> Mapping[str, Any]:
        """Initialize from the complete measured/authoritative dual-arm qpos."""

        with self._lock:
            # A failed reinitialization must not leave a stale prior epoch
            # available as an apparently valid hold for the new physical state.
            self._last_safe_full_q = None
            self._last_safe_collision_distance_m = None
            self._next_sequence = 0
            qpos = validate_qpos_by_side(current_qpos_by_side)
            if set(qpos) != set(SIDES):
                missing = sorted(set(SIDES) - set(qpos))
                raise ValueError(
                    "collision planner reset requires current qpos for both arms; "
                    f"missing passive/active sides={missing}."
                )
            reference = self._solver_current_q()
            full_q = self._embed_arm_qpos(reference, qpos)
            self._require_arm_qpos_within_limits(full_q, name="initial qpos")
            self._set_external_qpos(full_q)
            self._last_admitted_velocity_full = np.zeros_like(full_q)
            reset_motion_state = getattr(
                self.solver,
                "reset_admitted_reference_motion_state",
                None,
            )
            if callable(reset_motion_state):
                reset_motion_state()
            self._set_qp_motion_reference(
                full_q,
                self._last_admitted_velocity_full,
            )
            distance, distance_ms, _distance_details = self._collision_distance(
                full_q
            )
            safe, reason = self._collision_safe(distance)
            if not safe:
                raise ValueError(
                    "collision planner initial qpos is not certifiably safe: "
                    f"reason={reason}, distance_m={distance}."
                )
            # Validate that both solver frames are usable before declaring the
            # state a recoverable last-safe hold.
            self._realized_endpoint_poses(full_q)
            self._last_safe_full_q = full_q.copy()
            self._last_safe_collision_distance_m = distance
            self._next_sequence = 0
            self._last_pose_lift_metadata = {
                "profile": self.config.pose_lift_profile.to_metadata(),
                "applied": False,
                "reason": "reset",
            }
            self._last_nonlinear_candidate_metadata = {
                "enabled": bool(
                    self.config.pose_lift_profile.qp_nonlinear_best_so_far
                ),
                "reason": "reset",
                "candidates": [],
            }
            self._directional_task_guard_active_by_side = {
                side: False for side in SIDES
            }
            self._last_directional_task_guard_metadata = {
                "enabled": bool(
                    self.config.pose_lift_profile.directional_task_guard_max_utilization
                    > 0.0
                ),
                "reason": "reset",
                "sides": {},
            }
            return MappingProxyType(
                {
                    "initialized": True,
                    "qpos_sides": tuple(SIDES),
                    "collision_pair_count": int(self._collision_pair_count),
                    "collision_distance_m": distance,
                    "collision_check_ms": float(distance_ms),
                    "world_from_arm_base_source": self.world_from_arm_base_source,
                    "reference_hz": float(self.config.reference_hz),
                    "temporal_smoothing_applied": False,
                }
            )

    def plan(
        self,
        target_pose_by_side: Mapping[str, Any],
        *,
        target_reference_frame: Literal["arm_base", "world", "scene_world"] = "arm_base",
        target_controlled_frame: str = TOOL_TCP_FRAME,
        source_timestamp: float | None = None,
        source: str = "position_reference",
        metadata: Mapping[str, Any] | None = None,
    ) -> RM65PositionCollisionPlanResult:
        """Plan one coupled qpos endpoint from one high-rate pose reference.

        A one-sided target is valid only after ``reset`` supplied both arms.
        The untargeted arm is frozen at the last-safe qpos and remains present
        in collision geometry; it is never replaced with a guessed default.
        """

        with self._lock:
            if self._last_safe_full_q is None:
                raise RuntimeError("collision planner must be reset with both arms before plan().")
            sequence = self._next_sequence
            self._next_sequence += 1
            frame_started = time.perf_counter()
            source_metadata = dict(metadata or {})
            target_endpoint_world: dict[str, np.ndarray] = {}
            raw_full_q: np.ndarray | None = None
            collision_attempts: list[dict[str, Any]] = []
            solve_ms = 0.0
            solve_iterations = 0
            solve_stop_reason: str | None = None
            solve_pose_error: dict[str, dict[str, float]] = {}
            self._last_pose_lift_metadata = {
                "profile": self.config.pose_lift_profile.to_metadata(),
                "applied": False,
                "reason": "pending_solver_output",
            }
            self._last_nonlinear_candidate_metadata = {
                "enabled": bool(
                    self.config.pose_lift_profile.qp_nonlinear_best_so_far
                ),
                "reason": "pending_solver_output",
                "candidates": [],
            }
            self._last_directional_task_guard_metadata = {
                "enabled": bool(
                    self.config.pose_lift_profile.directional_task_guard_max_utilization
                    > 0.0
                ),
                "reason": "pending_target_resolution",
                "sides": {},
            }
            try:
                if source_timestamp is not None and not math.isfinite(float(source_timestamp)):
                    raise _PlanningFailure(
                        "invalid_source_timestamp",
                        "source_timestamp must be finite when provided.",
                    )
                source_text = str(source).strip()
                if not source_text:
                    raise _PlanningFailure("invalid_source", "source must be non-empty.")
                targets = _normalize_pose_mapping(
                    target_pose_by_side,
                    name="target_pose_by_side",
                    require_both=False,
                )
                active_sides = tuple(side for side in SIDES if side in targets)
                reference_frame = str(target_reference_frame).strip().lower()
                if reference_frame == "scene_world":
                    reference_frame = "world"
                if reference_frame not in {"arm_base", "world"}:
                    raise _PlanningFailure(
                        "unsupported_target_reference_frame",
                        f"unsupported target reference frame {target_reference_frame!r}.",
                    )
                controlled_frame = canonical_rm65_controlled_frame(target_controlled_frame)
                if controlled_frame not in {TOOL_TCP_FRAME, ARM_ENDPOINT_FRAME}:
                    raise _PlanningFailure(
                        "unsupported_target_controlled_frame",
                        f"unsupported target controlled frame {target_controlled_frame!r}.",
                    )

                reference_full_q = self._last_safe_full_q.copy()
                self._require_arm_qpos_within_limits(
                    reference_full_q,
                    name="last-safe collision-planner qpos",
                )
                self.solver.set_configuration_q(reference_full_q.copy())
                self._set_qp_motion_reference(
                    reference_full_q,
                    self._last_admitted_velocity_full
                )
                frame_name_by_side = {
                    LEFT: str(self.scene_profile.left_frame_name),
                    RIGHT: str(self.scene_profile.right_frame_name),
                }
                for side in SIDES:
                    if side not in targets:
                        target_endpoint_world[side] = self._compute_frame_pose(
                            frame_name_by_side[side], reference_full_q
                        )
                        continue
                    controlled_pose = targets[side]
                    world_from_controlled = (
                        self.world_from_arm_base_by_side[side] @ controlled_pose
                        if reference_frame == "arm_base"
                        else controlled_pose.copy()
                    )
                    endpoint_from_controlled = (
                        self.endpoint_to_tool_by_side[side]
                        if controlled_frame == TOOL_TCP_FRAME
                        else np.eye(4, dtype=float)
                    )
                    target_endpoint_world[side] = _pose_matrix(
                        world_from_controlled @ invert_pose(endpoint_from_controlled),
                        name=f"target_endpoint_pose_world_by_side.{side}",
                    )

                solver_target_endpoint_world = (
                    self._directionally_shape_solver_targets(
                        reference_full_q,
                        target_endpoint_world,
                        active_sides=active_sides,
                    )
                )
                solver_targets = [
                    self._pose_to_xyzquat(
                        solver_target_endpoint_world[LEFT]
                    ),
                    self._pose_to_xyzquat(
                        solver_target_endpoint_world[RIGHT]
                    ),
                ]
                previous_solve_ms: float | None = None
                iteration_budget_ms = (
                    max(
                        0.0,
                        float(self.config.solve_time_budget_ms or 0.0)
                        - float(self.config.solve_iteration_reserve_ms),
                    )
                    if self.config.solve_adaptive_iterations_enabled
                    and float(self.config.solve_time_budget_ms or 0.0) > 0.0
                    else None
                )
                solve_min_iterations = self.config.effective_solve_min_iterations
                solve_max_iterations = self.config.effective_solve_max_iterations
                nonlinear_candidates: list[dict[str, Any]] = []
                for iteration in range(1, solve_max_iterations + 1):
                    if (
                        iteration > solve_min_iterations
                        and iteration_budget_ms is not None
                    ):
                        elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
                        estimated_next_ms = max(0.05, float(previous_solve_ms or 0.0))
                        if elapsed_ms + estimated_next_ms > iteration_budget_ms:
                            solve_stop_reason = "iteration_budget_estimate"
                            break
                    iteration_started = time.perf_counter()
                    substep_reference_q = self._solver_current_q()
                    raw_full_q = np.asarray(
                        self.solver.solve(targets=solver_targets),
                        dtype=float,
                    ).reshape(-1)
                    if (
                        self.config.pose_lift_profile.qp_acceleration_limit
                        and self.config.pose_lift_profile.qp_substeps_per_reference > 0
                        and raw_full_q.shape == substep_reference_q.shape
                        and np.all(np.isfinite(raw_full_q))
                    ):
                        substep_dt = (
                            self.config.reference_period_sec
                            / float(solve_max_iterations)
                        )
                        self._set_qp_motion_reference(
                            raw_full_q,
                            (raw_full_q - substep_reference_q) / substep_dt
                        )
                    iteration_ms = (time.perf_counter() - iteration_started) * 1000.0
                    solve_ms += iteration_ms
                    previous_solve_ms = max(0.05, iteration_ms)
                    solve_iterations = iteration
                    if (
                        raw_full_q.shape != reference_full_q.shape
                        or not np.all(np.isfinite(raw_full_q))
                    ):
                        break
                    solve_pose_error = {}
                    within_tolerance = True
                    for side in SIDES:
                        realized_pose = self._compute_frame_pose(
                            frame_name_by_side[side],
                            raw_full_q,
                        )
                        translation_m, rotation_deg = pose_delta_metrics(
                            realized_pose,
                            target_endpoint_world[side],
                        )
                        solve_pose_error[side] = {
                            "translation_m": float(translation_m),
                            "rotation_deg": float(rotation_deg),
                        }
                        within_tolerance = bool(
                            within_tolerance
                            and translation_m
                            <= float(self.config.solve_translation_tolerance_m)
                            and rotation_deg
                            <= float(self.config.solve_orientation_tolerance_deg)
                        )
                    if (
                        self.config.pose_lift_profile.qp_nonlinear_best_so_far
                        and iteration
                        >= int(
                            self.config.pose_lift_profile.qp_nonlinear_best_min_iteration
                        )
                    ):
                        nonlinear_candidates.append(
                            self._evaluate_nonlinear_iteration_candidate(
                                iteration=iteration,
                                candidate_full_q=raw_full_q,
                                reference_full_q=reference_full_q,
                                pose_error_by_side=solve_pose_error,
                            )
                        )
                    if (
                        iteration >= solve_min_iterations
                        and within_tolerance
                    ):
                        solve_stop_reason = "pose_tolerance"
                        break
                if solve_stop_reason is None:
                    solve_stop_reason = "max_iterations"
                if (
                    raw_full_q is None
                    or raw_full_q.shape != reference_full_q.shape
                    or not np.all(np.isfinite(raw_full_q))
                ):
                    raise _PlanningFailure(
                        "invalid_ik_output",
                        "coupled collision IK returned a non-finite or mismatched configuration.",
                    )

                if self.config.pose_lift_profile.qp_nonlinear_best_so_far:
                    eligible_candidates = [
                        candidate
                        for candidate in nonlinear_candidates
                        if bool(candidate.get("eligible", False))
                    ]
                    selection_trace: list[dict[str, Any]] = []
                    selected: dict[str, Any] | None = None
                    if eligible_candidates:
                        selected = eligible_candidates[0]
                        selection_trace.append(
                            {
                                "iteration": int(selected["iteration"]),
                                "selected": True,
                                "reason": "first_eligible_candidate",
                            }
                        )
                        for candidate in eligible_candidates[1:]:
                            better, reason = (
                                self._nonlinear_iteration_candidate_is_better(
                                    candidate,
                                    selected,
                                )
                            )
                            selection_trace.append(
                                {
                                    "iteration": int(candidate["iteration"]),
                                    "selected": bool(better),
                                    "reason": reason,
                                }
                            )
                            if better:
                                selected = candidate
                        raw_full_q = np.asarray(
                            selected["_candidate_full_q"],
                            dtype=float,
                        ).copy()
                        solve_pose_error = {
                            side: {
                                "translation_m": float(
                                    selected["pose_error"][side]["translation_m"]
                                ),
                                "rotation_deg": float(
                                    selected["pose_error"][side]["rotation_deg"]
                                ),
                            }
                            for side in SIDES
                        }
                        self.solver.set_configuration_q(raw_full_q.copy())
                        solve_stop_reason = (
                            f"{solve_stop_reason}:best_so_far_iteration_"
                            f"{int(selected['iteration'])}"
                        )
                    self._last_nonlinear_candidate_metadata = {
                        "enabled": True,
                        "reason": (
                            "exact_nonlinear_best_candidate_selected"
                            if selected is not None
                            else "no_eligible_nonlinear_candidate"
                        ),
                        "selected_iteration": (
                            None
                            if selected is None
                            else int(selected["iteration"])
                        ),
                        "selection_trace": selection_trace,
                        "candidates": [
                            {
                                key: copy.deepcopy(value)
                                for key, value in candidate.items()
                                if not str(key).startswith("_")
                            }
                            for candidate in nonlinear_candidates
                        ],
                    }

                solver_raw_full_q = raw_full_q.copy()
                raw_full_q, self._last_pose_lift_metadata = (
                    self._canonicalize_pose_lift_candidate(
                        reference_full_q=reference_full_q,
                        solver_candidate_full_q=solver_raw_full_q,
                        target_endpoint_world=target_endpoint_world,
                        frame_name_by_side=frame_name_by_side,
                        active_sides=active_sides,
                    )
                )

                raw_arm_qpos = self._split_arm_qpos(raw_full_q)
                reference_arm_qpos = self._split_arm_qpos(reference_full_q)
                limited_arm_qpos: dict[str, np.ndarray] = {}
                reasons: list[str] = []
                max_step = self.config.effective_max_joint_step_rad
                for side in SIDES:
                    if side not in active_sides:
                        limited_arm_qpos[side] = reference_arm_qpos[side].copy()
                        if not np.allclose(
                            raw_arm_qpos[side],
                            reference_arm_qpos[side],
                            atol=1e-12,
                            rtol=0.0,
                        ):
                            reasons.append(f"passive_side_held:{side}")
                        continue
                    constrained = constrain_joint_vector(
                        raw_arm_qpos[side],
                        reference=reference_arm_qpos[side],
                        lower=self._joint_lower_by_side[side],
                        upper=self._joint_upper_by_side[side],
                        max_step_rad=max_step,
                    )
                    if constrained.infeasible or constrained.value is None:
                        raise _PlanningFailure(
                            "joint_constraint_infeasible",
                            f"{side} hard joint bounds and reference-step interval do not intersect; "
                            f"indices={list(constrained.infeasible_indices)}.",
                        )
                    if constrained.absolute_limited:
                        reasons.append(f"joint_position_limited:{side}")
                    if constrained.step_limited:
                        reasons.append(f"joint_step_limited:{side}")
                    limited_arm_qpos[side] = np.asarray(constrained.value, dtype=float)

                limited_full_q = self._embed_arm_qpos(reference_full_q, limited_arm_qpos)
                self._require_arm_qpos_within_limits(limited_full_q, name="limited IK qpos")

                distance, distance_ms, distance_details = (
                    self._collision_distance(limited_full_q)
                )
                collision_attempts.append(
                    {
                        "phase": "post_joint_limits",
                        "scale": 1.0,
                        "distance_m": distance,
                        "elapsed_ms": distance_ms,
                        **distance_details,
                    }
                )
                safe, collision_reason = self._collision_safe(distance)
                accepted_full_q = limited_full_q
                accepted_distance = distance
                accepted_scale = 1.0
                if not safe:
                    if collision_reason == "collision_distance_unavailable":
                        raise _PlanningFailure(
                            collision_reason,
                            "collision distance is unavailable for a scene with configured collision pairs.",
                        )
                    accepted_full_q = reference_full_q.copy()
                    accepted_distance = self._last_safe_collision_distance_m
                    accepted_scale = 0.0
                    for scale in self.config.collision_backoff_scales:
                        candidate_full_q = reference_full_q + scale * (
                            limited_full_q - reference_full_q
                        )
                        self._require_arm_qpos_within_limits(
                            candidate_full_q,
                            name=f"collision backoff qpos scale={scale:g}",
                        )
                        (
                            candidate_distance,
                            candidate_ms,
                            candidate_details,
                        ) = self._collision_distance(candidate_full_q)
                        collision_attempts.append(
                            {
                                "phase": "post_joint_limits_collision_backoff",
                                "scale": float(scale),
                                "distance_m": candidate_distance,
                                "elapsed_ms": candidate_ms,
                                **candidate_details,
                            }
                        )
                        candidate_safe, candidate_reason = self._collision_safe(
                            candidate_distance
                        )
                        if candidate_reason == "collision_distance_unavailable":
                            raise _PlanningFailure(
                                candidate_reason,
                                "collision distance became unavailable during collision backoff.",
                            )
                        if candidate_safe:
                            accepted_full_q = candidate_full_q
                            accepted_distance = candidate_distance
                            accepted_scale = float(scale)
                            reasons.append("collision_backoff")
                            break
                    if accepted_scale <= 0.0:
                        raise _PlanningFailure(
                            "collision_margin_low",
                            "no collision-safe bounded qpos step was found.",
                        )

                self.solver.set_configuration_q(accepted_full_q.copy())
                self._commit_qp_admitted_velocity(
                    reference_full_q=reference_full_q,
                    admitted_full_q=accepted_full_q,
                )
                realized_endpoint_world = self._realized_endpoint_poses(accepted_full_q)
                q_ref_by_side = self._split_arm_qpos(accepted_full_q)
                q_ik_raw_by_side = self._split_arm_qpos(solver_raw_full_q)
                self._last_safe_full_q = accepted_full_q.copy()
                self._last_safe_collision_distance_m = accepted_distance
                elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
                status: Literal["tracking", "limited", "collision_backoff"]
                if accepted_scale < 1.0:
                    status = "collision_backoff"
                elif reasons:
                    status = "limited"
                else:
                    status = "tracking"
                return RM65PositionCollisionPlanResult(
                    sequence=sequence,
                    source_timestamp=source_timestamp,
                    source=source_text,
                    status=status,
                    valid=True,
                    hold=False,
                    active_sides=active_sides,
                    q_ik_raw_by_side=q_ik_raw_by_side,
                    q_ref_by_side=q_ref_by_side,
                    target_endpoint_pose_world_by_side=target_endpoint_world,
                    realized_endpoint_pose_world_by_side=realized_endpoint_world,
                    modification_reasons=tuple(reasons),
                    metadata=self._result_metadata(
                        source_metadata=source_metadata,
                        reference_frame=reference_frame,
                        controlled_frame=controlled_frame,
                        passive_sides=tuple(side for side in SIDES if side not in active_sides),
                        solve_ms=solve_ms,
                        elapsed_ms=elapsed_ms,
                        collision_attempts=collision_attempts,
                        collision_distance_m=accepted_distance,
                        collision_backoff_scale=accepted_scale,
                        failure=None,
                        solve_iterations=solve_iterations,
                        solve_stop_reason=solve_stop_reason,
                        solve_pose_error=solve_pose_error,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - all planning failures fail closed to hold.
                code = exc.code if isinstance(exc, _PlanningFailure) else f"planner_exception:{type(exc).__name__}"
                return self._hold_result(
                    sequence=sequence,
                    source_timestamp=source_timestamp,
                    source=str(source).strip() or "position_reference",
                    active_sides=tuple(
                        side
                        for side in SIDES
                        if isinstance(target_pose_by_side, Mapping) and side in target_pose_by_side
                    )
                    or SIDES,
                    target_endpoint_world=target_endpoint_world,
                    raw_full_q=raw_full_q,
                    failure_code=code,
                    error=exc,
                    frame_started=frame_started,
                    solve_ms=solve_ms,
                    collision_attempts=collision_attempts,
                    source_metadata=source_metadata,
                    solve_iterations=solve_iterations,
                    solve_stop_reason=solve_stop_reason,
                    solve_pose_error=solve_pose_error,
                )

    def plan_pose_reference(
        self,
        target_pose_by_side: Mapping[str, Any],
        *,
        target_reference_frame: Literal["arm_base", "world", "scene_world"] = "arm_base",
        target_controlled_frame: str = TOOL_TCP_FRAME,
        source_timestamp: float | None = None,
        source: str = "position_reference",
        metadata: Mapping[str, Any] | None = None,
    ) -> RM65PositionCollisionPlanResult:
        """Named integration entry point for a resampled high-rate pose reference."""

        return self.plan(
            target_pose_by_side,
            target_reference_frame=target_reference_frame,
            target_controlled_frame=target_controlled_frame,
            source_timestamp=source_timestamp,
            source=source,
            metadata=metadata,
        )

    def validate_qpos_reference(
        self,
        qpos_by_side: Mapping[str, Any],
        *,
        source: str = "qpos_reference",
    ) -> RM65QposCollisionValidationResult:
        """Fail-closed endpoint check for a caller-owned qpos reference.

        Missing sides are frozen at the initialized last-safe state.  This is
        an endpoint check only: it neither certifies the path from the previous
        qpos nor commits the candidate as the planner's new last-safe state.
        """

        with self._lock:
            if self._last_safe_full_q is None:
                raise RuntimeError(
                    "collision planner must be reset with both arms before qpos validation."
                )
            requested = validate_qpos_by_side(qpos_by_side)
            active_sides = tuple(side for side in SIDES if side in requested)
            reference_qpos = self._split_arm_qpos(self._last_safe_full_q)
            complete_qpos = {
                side: (
                    requested[side].copy()
                    if side in requested
                    else reference_qpos[side].copy()
                )
                for side in SIDES
            }
            full_q = self._embed_arm_qpos(self._last_safe_full_q, complete_qpos)
            reason: str | None = None
            try:
                self._require_arm_qpos_within_limits(full_q, name="qpos reference")
            except _PlanningFailure as exc:
                reason = exc.code
            distance: float | None = None
            distance_ms = 0.0
            distance_details: dict[str, object] = {}
            if reason is None:
                try:
                    distance, distance_ms, distance_details = (
                        self._collision_distance(full_q)
                    )
                    collision_safe, collision_reason = self._collision_safe(distance)
                    if not collision_safe:
                        reason = collision_reason
                except Exception as exc:  # noqa: BLE001 - validation is fail-closed.
                    reason = f"collision_check_exception:{type(exc).__name__}"
            safe = reason is None
            return RM65QposCollisionValidationResult(
                safe=safe,
                reason=reason,
                active_sides=active_sides,
                qpos_by_side=complete_qpos,
                collision_distance_m=distance,
                metadata={
                    "planner": "rm65_coupled_position_collision",
                    "validation_stage": "explicit_qpos_endpoint",
                    "source": str(source),
                    "collision_pair_count": int(self._collision_pair_count),
                    "collision_check_ms": float(distance_ms),
                    **distance_details,
                    "collision_min_distance_m": float(
                        self.config.collision_min_distance_m
                    ),
                    "endpoint_collision_check_required": bool(
                        self._collision_pair_count > 0
                    ),
                    "endpoint_collision_checked": bool(
                        self._collision_pair_count > 0 and distance is not None
                    ),
                    "endpoint_collision_passed": bool(safe),
                    "segment_collision_checked": False,
                    "segment_collision_certified": False,
                    "collision_certification_status": (
                        "qpos_endpoint_only_segment_not_certified"
                    ),
                    "passive_sides": [
                        side for side in SIDES if side not in active_sides
                    ],
                    "passive_side_policy": "freeze_at_last_safe_qpos",
                    "mutates_last_safe_state": False,
                    "temporal_smoothing_applied": False,
                },
            )

    def admit_qpos_reference(
        self,
        qpos_by_side: Mapping[str, Any],
        *,
        source: str = "qpos_reference",
    ) -> RM65QposCollisionValidationResult:
        """Validate and commit an exact caller-owned qpos endpoint.

        The candidate is never modified.  A safe endpoint becomes the next
        last-safe anchor; an unsafe endpoint leaves planner state unchanged.
        """

        with self._lock:
            result = self.validate_qpos_reference(qpos_by_side, source=source)
            if not result.safe:
                return result
            assert self._last_safe_full_q is not None
            committed = {
                side: np.asarray(result.qpos_by_side[side], dtype=float).reshape(ARM_DOF)
                for side in SIDES
            }
            reference_full_q = self._last_safe_full_q.copy()
            full_q = self._embed_arm_qpos(reference_full_q, committed)
            self._require_arm_qpos_within_limits(full_q, name="admitted qpos reference")
            self._set_external_qpos(full_q)
            self._commit_qp_admitted_velocity(
                reference_full_q=reference_full_q,
                admitted_full_q=full_q,
            )
            self._last_safe_full_q = full_q.copy()
            self._last_safe_collision_distance_m = result.collision_distance_m
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "validation_stage": "explicit_qpos_endpoint_admission",
                    "mutates_last_safe_state": True,
                    "qpos_target_modified": False,
                }
            )
            return RM65QposCollisionValidationResult(
                safe=True,
                reason=None,
                active_sides=result.active_sides,
                qpos_by_side=result.qpos_by_side,
                collision_distance_m=result.collision_distance_m,
                metadata=metadata,
            )

    def _canonicalize_pose_lift_candidate(
        self,
        *,
        reference_full_q: np.ndarray,
        solver_candidate_full_q: np.ndarray,
        target_endpoint_world: Mapping[str, np.ndarray],
        frame_name_by_side: Mapping[str, str],
        active_sides: tuple[str, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Choose only pose-certified equivalent representations near anchor."""

        profile = self.config.pose_lift_profile
        metadata: dict[str, Any] = {
            "profile": profile.to_metadata(),
            "anchor": "last_formally_admitted_qref",
            "applied": False,
            "reason": "baseline_passthrough",
            "stages": [],
        }
        if not (
            profile.equivalent_joint_unwrap
            or profile.wrist_coaxial_canonicalization
        ):
            return solver_candidate_full_q.copy(), metadata

        reference_by_side = self._split_arm_qpos(reference_full_q)
        solver_by_side = self._split_arm_qpos(solver_candidate_full_q)
        unwrapped_by_side: dict[str, np.ndarray] = {}
        unwrap_applied = False
        for side in SIDES:
            candidate = solver_by_side[side].copy()
            if side in active_sides and profile.equivalent_joint_unwrap:
                candidate = self._unwrap_qpos_near_reference(
                    reference_by_side[side],
                    candidate,
                    lower=self._joint_lower_by_side[side],
                    upper=self._joint_upper_by_side[side],
                )
                unwrap_applied = bool(
                    unwrap_applied
                    or not np.allclose(
                        candidate,
                        solver_by_side[side],
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
            unwrapped_by_side[side] = candidate
        unwrapped_full_q = self._embed_arm_qpos(reference_full_q, unwrapped_by_side)
        metadata["stages"].append(
            {"name": "legal_joint_unwrap", "applied": unwrap_applied}
        )

        if profile.wrist_bounded_local_search:
            return self._bounded_pose_lift_candidate(
                reference_full_q=reference_full_q,
                solver_candidate_full_q=solver_candidate_full_q,
                solver_by_side=solver_by_side,
                unwrapped_by_side=unwrapped_by_side,
                target_endpoint_world=target_endpoint_world,
                frame_name_by_side=frame_name_by_side,
                active_sides=active_sides,
                metadata=metadata,
            )

        candidate_by_side = {
            side: value.copy() for side, value in unwrapped_by_side.items()
        }
        wrist_metadata: dict[str, Any] = {}
        if profile.wrist_coaxial_canonicalization:
            for side in active_sides:
                candidate_by_side[side], wrist_metadata[side] = (
                    canonicalize_rm65_wrist_coaxial_step(
                        reference_by_side[side],
                        candidate_by_side[side],
                        soft_threshold_deg=profile.wrist_soft_threshold_deg,
                        hard_threshold_deg=profile.wrist_hard_threshold_deg,
                        null_motion_soft_limit_deg=(
                            profile.wrist_null_motion_soft_limit_deg
                        ),
                    )
                )
        wrist_full_q = self._embed_arm_qpos(reference_full_q, candidate_by_side)
        wrist_applied = any(
            bool(value.get("applied", False)) for value in wrist_metadata.values()
        )
        metadata["stages"].append(
            {
                "name": "wrist_coaxial_equivalent_normalization",
                "applied": wrist_applied,
                "sides": wrist_metadata,
            }
        )

        for candidate_name, candidate_full_q in (
            ("wrist_canonicalized", wrist_full_q),
            ("legal_joint_unwrap", unwrapped_full_q),
        ):
            certification = self._pose_lift_candidate_certification(
                candidate_full_q,
                target_endpoint_world=target_endpoint_world,
                frame_name_by_side=frame_name_by_side,
                active_sides=active_sides,
            )
            metadata.setdefault("certification_attempts", []).append(
                {"candidate": candidate_name, **certification}
            )
            if certification["accepted"]:
                applied = not np.allclose(
                    candidate_full_q,
                    solver_candidate_full_q,
                    atol=1e-12,
                    rtol=0.0,
                )
                metadata.update(
                    {
                        "applied": bool(applied),
                        "reason": (
                            f"accepted_{candidate_name}"
                            if applied
                            else "candidate_already_canonical"
                        ),
                    }
                )
                return candidate_full_q.copy(), metadata

        metadata["reason"] = "normalization_rejected_pose_or_limits"
        return solver_candidate_full_q.copy(), metadata

    def _bounded_pose_lift_candidate(
        self,
        *,
        reference_full_q: np.ndarray,
        solver_candidate_full_q: np.ndarray,
        solver_by_side: Mapping[str, np.ndarray],
        unwrapped_by_side: Mapping[str, np.ndarray],
        target_endpoint_world: Mapping[str, np.ndarray],
        frame_name_by_side: Mapping[str, str],
        active_sides: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select the strongest certified wrist-continuity correction per arm.

        The raw coupled IK solution is the pose baseline.  This is important:
        an IK solve may already be slightly outside its absolute residual
        tolerance, and an exactly equivalent representation must not be
        rejected merely because it inherited that pre-existing residual.
        Every accepted correction is instead bounded relative to the raw IK
        pose and may worsen its target residual only by the profile slack.
        """

        profile = self.config.pose_lift_profile
        reference_by_side = self._split_arm_qpos(reference_full_q)
        selected_by_side = {
            side: np.asarray(solver_by_side[side], dtype=float).copy()
            for side in SIDES
        }
        side_results: dict[str, Any] = {}

        for side in active_sides:
            reference_side = np.asarray(reference_by_side[side], dtype=float)
            side_result: dict[str, Any] = {
                "anchor": "last_formally_admitted_qref",
                "selected_scale": 0.0,
                "unwrap_applied": False,
                "canonicalizer": {},
                "attempts": [],
            }

            # Legal 2pi representatives are checked independently so a
            # problematic arm cannot force the other arm back to the raw
            # branch.
            proposed = {
                name: value.copy() for name, value in selected_by_side.items()
            }
            proposed[side] = np.asarray(unwrapped_by_side[side], dtype=float).copy()
            proposed_full_q = self._embed_arm_qpos(reference_full_q, proposed)
            unwrap_certification = self._pose_lift_equivalence_certification(
                proposed_full_q,
                baseline_full_q=solver_candidate_full_q,
                target_endpoint_world=target_endpoint_world,
                frame_name_by_side=frame_name_by_side,
                active_sides=(side,),
            )
            side_result["attempts"].append(
                {
                    "candidate": "legal_joint_unwrap",
                    "scale": 0.0,
                    **unwrap_certification,
                }
            )
            if unwrap_certification["accepted"]:
                selected_by_side[side] = proposed[side]
                side_result["unwrap_applied"] = bool(
                    not np.allclose(
                        proposed[side],
                        solver_by_side[side],
                        atol=1e-12,
                        rtol=0.0,
                    )
                )

            wrist_target, wrist_metadata = canonicalize_rm65_wrist_coaxial_step(
                reference_side,
                selected_by_side[side],
                soft_threshold_deg=profile.wrist_soft_threshold_deg,
                hard_threshold_deg=profile.wrist_hard_threshold_deg,
                null_motion_soft_limit_deg=profile.wrist_null_motion_soft_limit_deg,
            )
            side_result["canonicalizer"] = wrist_metadata
            correction = wrist_target - selected_by_side[side]
            if bool(wrist_metadata.get("applied", False)):
                for scale in profile.wrist_local_search_scales:
                    proposed = {
                        name: value.copy()
                        for name, value in selected_by_side.items()
                    }
                    proposed[side] = selected_by_side[side] + float(scale) * correction
                    proposed_full_q = self._embed_arm_qpos(reference_full_q, proposed)
                    certification = self._pose_lift_equivalence_certification(
                        proposed_full_q,
                        baseline_full_q=solver_candidate_full_q,
                        target_endpoint_world=target_endpoint_world,
                        frame_name_by_side=frame_name_by_side,
                        active_sides=(side,),
                    )
                    side_result["attempts"].append(
                        {
                            "candidate": "wrist_continuity_correction",
                            "scale": float(scale),
                            **certification,
                        }
                    )
                    if certification["accepted"]:
                        selected_by_side[side] = proposed[side]
                        side_result["selected_scale"] = float(scale)
                        break
            side_results[side] = side_result

        selected_full_q = self._embed_arm_qpos(reference_full_q, selected_by_side)
        final_certification = self._pose_lift_equivalence_certification(
            selected_full_q,
            baseline_full_q=solver_candidate_full_q,
            target_endpoint_world=target_endpoint_world,
            frame_name_by_side=frame_name_by_side,
            active_sides=active_sides,
        )
        metadata["stages"].append(
            {
                "name": "bounded_per_side_wrist_continuity_search",
                "applied": any(
                    bool(result.get("unwrap_applied", False))
                    or float(result.get("selected_scale", 0.0)) > 0.0
                    for result in side_results.values()
                ),
                "sides": side_results,
                "final_certification": final_certification,
            }
        )
        if not final_certification["accepted"]:
            metadata["reason"] = "bounded_search_final_certification_failed"
            return solver_candidate_full_q.copy(), metadata

        applied = not np.allclose(
            selected_full_q,
            solver_candidate_full_q,
            atol=1e-12,
            rtol=0.0,
        )
        selected_scales = {
            side: float(result.get("selected_scale", 0.0))
            for side, result in side_results.items()
        }
        metadata.update(
            {
                "applied": bool(applied),
                "reason": (
                    "accepted_bounded_wrist_continuity"
                    if applied
                    else "bounded_candidate_already_canonical"
                ),
                "selected_scales": selected_scales,
            }
        )
        return selected_full_q.copy(), metadata

    @staticmethod
    def _unwrap_qpos_near_reference(
        reference: np.ndarray,
        candidate: np.ndarray,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        output = np.asarray(candidate, dtype=float).reshape(ARM_DOF).copy()
        reference = np.asarray(reference, dtype=float).reshape(ARM_DOF)
        lower = np.asarray(lower, dtype=float).reshape(ARM_DOF)
        upper = np.asarray(upper, dtype=float).reshape(ARM_DOF)
        period = 2.0 * math.pi
        for index in range(ARM_DOF):
            center_k = int(round((reference[index] - output[index]) / period))
            equivalents = [
                output[index] + period * k
                for k in range(center_k - 2, center_k + 3)
                if lower[index] - 1e-12
                <= output[index] + period * k
                <= upper[index] + 1e-12
            ]
            if equivalents:
                output[index] = min(
                    equivalents,
                    key=lambda value: abs(float(value - reference[index])),
                )
        return output

    def _pose_lift_candidate_certification(
        self,
        candidate_full_q: np.ndarray,
        *,
        target_endpoint_world: Mapping[str, np.ndarray],
        frame_name_by_side: Mapping[str, str],
        active_sides: tuple[str, ...],
    ) -> dict[str, Any]:
        try:
            self._require_arm_qpos_within_limits(
                candidate_full_q,
                name="pose-lift normalized candidate",
            )
        except _PlanningFailure as exc:
            return {"accepted": False, "reason": exc.code, "pose_error": {}}
        errors: dict[str, dict[str, float]] = {}
        accepted = True
        for side in active_sides:
            realized = self._compute_frame_pose(
                frame_name_by_side[side],
                candidate_full_q,
            )
            translation_m, rotation_deg = pose_delta_metrics(
                realized,
                target_endpoint_world[side],
            )
            errors[side] = {
                "translation_m": float(translation_m),
                "rotation_deg": float(rotation_deg),
            }
            accepted = bool(
                accepted
                and translation_m <= self.config.solve_translation_tolerance_m
                and rotation_deg <= self.config.solve_orientation_tolerance_deg
            )
        return {
            "accepted": bool(accepted),
            "reason": "pose_tolerance" if accepted else "pose_tolerance_exceeded",
            "pose_error": errors,
        }

    def _pose_lift_equivalence_certification(
        self,
        candidate_full_q: np.ndarray,
        *,
        baseline_full_q: np.ndarray,
        target_endpoint_world: Mapping[str, np.ndarray],
        frame_name_by_side: Mapping[str, str],
        active_sides: tuple[str, ...],
    ) -> dict[str, Any]:
        """Certify a V3 candidate against raw IK, not an absolute residual."""

        profile = self.config.pose_lift_profile
        try:
            self._require_arm_qpos_within_limits(
                candidate_full_q,
                name="pose-lift bounded candidate",
            )
        except _PlanningFailure as exc:
            return {"accepted": False, "reason": exc.code, "sides": {}}

        sides: dict[str, dict[str, float | bool]] = {}
        accepted = True
        for side in active_sides:
            baseline_pose = self._compute_frame_pose(
                frame_name_by_side[side], baseline_full_q
            )
            candidate_pose = self._compute_frame_pose(
                frame_name_by_side[side], candidate_full_q
            )
            target_pose = target_endpoint_world[side]
            equivalence_translation_m, equivalence_rotation_deg = pose_delta_metrics(
                candidate_pose,
                baseline_pose,
            )
            baseline_translation_m, baseline_rotation_deg = pose_delta_metrics(
                baseline_pose,
                target_pose,
            )
            candidate_translation_m, candidate_rotation_deg = pose_delta_metrics(
                candidate_pose,
                target_pose,
            )
            equivalence_ok = bool(
                equivalence_translation_m
                <= profile.equivalence_translation_tolerance_m + 1e-12
                and equivalence_rotation_deg
                <= profile.equivalence_rotation_tolerance_deg + 1e-12
            )
            target_not_worse = bool(
                candidate_translation_m
                <= baseline_translation_m + profile.target_translation_slack_m + 1e-12
                and candidate_rotation_deg
                <= baseline_rotation_deg + profile.target_rotation_slack_deg + 1e-12
            )
            side_ok = bool(equivalence_ok and target_not_worse)
            accepted = bool(accepted and side_ok)
            sides[side] = {
                "accepted": side_ok,
                "equivalence_translation_m": float(equivalence_translation_m),
                "equivalence_rotation_deg": float(equivalence_rotation_deg),
                "baseline_target_translation_m": float(baseline_translation_m),
                "baseline_target_rotation_deg": float(baseline_rotation_deg),
                "candidate_target_translation_m": float(candidate_translation_m),
                "candidate_target_rotation_deg": float(candidate_rotation_deg),
                "equivalence_ok": equivalence_ok,
                "target_not_worse": target_not_worse,
            }
        return {
            "accepted": bool(accepted),
            "reason": (
                "equivalent_and_target_not_worse"
                if accepted
                else "equivalence_or_target_slack_exceeded"
            ),
            "sides": sides,
        }

    def _hold_result(
        self,
        *,
        sequence: int,
        source_timestamp: float | None,
        source: str,
        active_sides: tuple[str, ...],
        target_endpoint_world: Mapping[str, np.ndarray],
        raw_full_q: np.ndarray | None,
        failure_code: str,
        error: BaseException,
        frame_started: float,
        solve_ms: float,
        collision_attempts: list[dict[str, Any]],
        source_metadata: Mapping[str, Any],
        solve_iterations: int,
        solve_stop_reason: str | None,
        solve_pose_error: Mapping[str, Mapping[str, float]],
    ) -> RM65PositionCollisionPlanResult:
        assert self._last_safe_full_q is not None
        hold_q = self._last_safe_full_q.copy()
        self.solver.set_configuration_q(hold_q.copy())
        self._commit_qp_admitted_velocity(
            reference_full_q=hold_q,
            admitted_full_q=hold_q,
        )
        try:
            realized = self._realized_endpoint_poses(hold_q)
        except Exception:
            realized = {}
        q_ik_raw = (
            {}
            if raw_full_q is None
            or np.asarray(raw_full_q).reshape(-1).shape != hold_q.shape
            or not np.all(np.isfinite(raw_full_q))
            else self._split_arm_qpos(raw_full_q)
        )
        elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
        return RM65PositionCollisionPlanResult(
            sequence=sequence,
            source_timestamp=(
                source_timestamp
                if source_timestamp is None or math.isfinite(float(source_timestamp))
                else None
            ),
            source=source,
            status="hold_last_safe",
            valid=True,
            hold=True,
            active_sides=tuple(side for side in SIDES if side in set(active_sides)) or SIDES,
            q_ik_raw_by_side=q_ik_raw,
            q_ref_by_side=self._split_arm_qpos(hold_q),
            target_endpoint_pose_world_by_side=target_endpoint_world,
            realized_endpoint_pose_world_by_side=realized,
            modification_reasons=(failure_code, "hold_last_safe"),
            metadata=self._result_metadata(
                source_metadata=source_metadata,
                reference_frame=None,
                controlled_frame=None,
                passive_sides=tuple(side for side in SIDES if side not in set(active_sides)),
                solve_ms=solve_ms,
                elapsed_ms=elapsed_ms,
                collision_attempts=collision_attempts,
                collision_distance_m=self._last_safe_collision_distance_m,
                collision_backoff_scale=0.0,
                failure={
                    "code": failure_code,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                solve_iterations=solve_iterations,
                solve_stop_reason=solve_stop_reason,
                solve_pose_error=solve_pose_error,
            ),
        )

    def _result_metadata(
        self,
        *,
        source_metadata: Mapping[str, Any],
        reference_frame: str | None,
        controlled_frame: str | None,
        passive_sides: tuple[str, ...],
        solve_ms: float,
        elapsed_ms: float,
        collision_attempts: Sequence[Mapping[str, Any]],
        collision_distance_m: float | None,
        collision_backoff_scale: float,
        failure: Mapping[str, Any] | None,
        solve_iterations: int,
        solve_stop_reason: str | None,
        solve_pose_error: Mapping[str, Mapping[str, float]],
    ) -> dict[str, Any]:
        budget_ms = float(self.config.solve_time_budget_ms or 0.0)
        output_endpoint_safe = bool(
            failure is None
            or self._collision_pair_count == 0
            or self._last_safe_collision_distance_m is not None
        )
        candidate_collision_checked = bool(
            self._collision_pair_count > 0
            and collision_attempts
            and collision_attempts[-1].get("distance_m") is not None
        )
        return {
            "planner": "rm65_coupled_position_collision",
            "planner_stage": "reference_pose_to_collision_aware_qpos",
            "reference_hz": float(self.config.reference_hz),
            "reference_period_sec": float(self.config.reference_period_sec),
            "scene_profile": str(self.scene_profile.name),
            "collision_pair_count": int(self._collision_pair_count),
            "collision_min_distance_m": float(self.config.collision_min_distance_m),
            "collision_distance_epsilon_m": float(
                self.config.collision_distance_epsilon_m
            ),
            "collision_distance_m": collision_distance_m,
            "collision_check_attempts": [dict(attempt) for attempt in collision_attempts],
            "collision_check_scope": "reference_limited_qpos_endpoint_only",
            "endpoint_collision_check_required": bool(self._collision_pair_count > 0),
            "endpoint_collision_checked": bool(
                self._collision_pair_count > 0 and collision_distance_m is not None
            ),
            "endpoint_collision_passed": output_endpoint_safe,
            "candidate_endpoint_collision_checked": candidate_collision_checked,
            "candidate_endpoint_collision_passed": bool(failure is None),
            "hold_reference_endpoint_collision_passed": bool(
                failure is not None
                and (
                    self._collision_pair_count == 0
                    or self._last_safe_collision_distance_m is not None
                )
            ),
            "segment_collision_checked": False,
            "segment_collision_certified": False,
            "collision_certification_status": "reference_endpoint_only_segment_not_certified",
            "collision_backoff_scale": float(collision_backoff_scale),
            "joint_limit_order": "hard_joint_bounds_intersect_per_reference_step_then_collision_recheck",
            "joint_lower_rad_by_side": {
                side: self._joint_lower_by_side[side].tolist() for side in SIDES
            },
            "joint_upper_rad_by_side": {
                side: self._joint_upper_by_side[side].tolist() for side in SIDES
            },
            "max_joint_step_rad": self.config.effective_max_joint_step_rad.tolist(),
            "qp_motion_constraints": {
                "substeps_per_reference": int(
                    self.config.pose_lift_profile.qp_substeps_per_reference
                ),
                "substep_dt_sec": (
                    None
                    if self.config.pose_lift_profile.qp_substeps_per_reference <= 0
                    else self.config.reference_period_sec
                    / float(
                        self.config.pose_lift_profile.qp_substeps_per_reference
                    )
                ),
                "velocity_limit_enabled": bool(
                    self.config.pose_lift_profile.qp_velocity_limit
                ),
                "acceleration_limit_enabled": bool(
                    self.config.pose_lift_profile.qp_acceleration_limit
                ),
                "sigma_escape_enabled": bool(
                    self.config.pose_lift_profile.qp_sigma_escape_cost > 0.0
                ),
                "sigma_escape_cost": float(
                    self.config.pose_lift_profile.qp_sigma_escape_cost
                ),
                "sigma_escape_guard": float(
                    self.config.pose_lift_profile.qp_sigma_escape_guard
                ),
                "sigma_escape_rate_per_sec": float(
                    self.config.pose_lift_profile.qp_sigma_escape_rate_per_sec
                ),
                "sigma_escape_authority": (
                    "soft_objective_only_hard_motion_and_collision_constraints_win"
                ),
                "structural_branch_guard_enabled": bool(
                    self.config.pose_lift_profile.qp_structural_branch_margin_deg
                    > 0.0
                ),
                "structural_branch_guard_margin_deg": float(
                    self.config.pose_lift_profile.qp_structural_branch_margin_deg
                ),
                "structural_branch_guard_sigma": float(
                    self.config.pose_lift_profile.qp_structural_branch_guard_sigma
                ),
                "structural_branch_release_sigma": float(
                    self.config.pose_lift_profile.qp_structural_branch_release_sigma
                ),
                "structural_branch_escape_cost": float(
                    self.config.pose_lift_profile.qp_structural_branch_escape_cost
                ),
                "structural_branch_escape_target_deg": float(
                    self.config.pose_lift_profile.qp_structural_branch_escape_target_deg
                ),
                "structural_branch_escape_guard_sigma": float(
                    self.config.pose_lift_profile.qp_structural_branch_escape_guard_sigma
                ),
                "structural_branch_constraint_mode": (
                    self.config.pose_lift_profile.qp_structural_branch_constraint_mode
                ),
                "structural_branch_barrier_gain": float(
                    self.config.pose_lift_profile.qp_structural_branch_barrier_gain
                ),
                "structural_branch_barrier_activation_sigma": float(
                    self.config.pose_lift_profile.qp_structural_branch_barrier_activation_sigma
                ),
                "structural_branch_prediction_horizon_sec": float(
                    self.config.pose_lift_profile.qp_structural_branch_prediction_horizon_sec
                ),
                "structural_branch_guard_authority": (
                    "offline_profile_only_one_sided_q3_q5_zero_lock"
                ),
                "structural_branch_guard_runtime": copy.deepcopy(
                    getattr(
                        getattr(
                            self.solver,
                            "structural_branch_guard_limit",
                            None,
                        ),
                        "metadata",
                        {},
                    )
                ),
                "directional_task_guard_enabled": bool(
                    self.config.pose_lift_profile.directional_task_guard_max_utilization
                    > 0.0
                ),
                "directional_task_guard": copy.deepcopy(
                    self._last_directional_task_guard_metadata
                ),
                "nonlinear_best_so_far_enabled": bool(
                    self.config.pose_lift_profile.qp_nonlinear_best_so_far
                ),
                "nonlinear_best_so_far": copy.deepcopy(
                    self._last_nonlinear_candidate_metadata
                ),
                "velocity_limit_rad_s": list(
                    self.config.max_joint_velocity_rad_s
                ),
                "acceleration_limit_rad_s2": list(
                    self.config.max_joint_acceleration_rad_s2
                ),
                "previous_velocity_source": "last_formally_admitted_qref",
                "last_admitted_qdot_rad_s_by_side": {
                    side: self._last_admitted_velocity_full[arm_slice]
                    .astype(float)
                    .tolist()
                    for side, arm_slice in self._arm_slices.items()
                },
            },
            "target_reference_frame": reference_frame,
            "target_controlled_frame": controlled_frame,
            "world_from_arm_base_source": self.world_from_arm_base_source,
            "passive_sides": list(passive_sides),
            "passive_side_policy": "freeze_at_last_safe_qpos_in_full_collision_model",
            "solve_ms": float(solve_ms),
            "ik_solve_iterations": int(solve_iterations),
            "ik_solve_iteration_min": int(
                self.config.effective_solve_min_iterations
            ),
            "ik_solve_iteration_max": int(
                self.config.effective_solve_max_iterations
            ),
            "ik_solve_iteration_requested_min": int(
                self.config.solve_min_iterations
            ),
            "ik_solve_iteration_requested_max": int(
                self.config.solve_max_iterations
            ),
            "ik_solve_iteration_mode": (
                "fixed_differential_substeps"
                if self.config.pose_lift_profile.qp_substeps_per_reference > 0
                else (
                    "adaptive_budget"
                    if self.config.solve_adaptive_iterations_enabled
                    else "fixed_max"
                )
            ),
            "ik_solve_iteration_stop_reason": solve_stop_reason,
            "ik_solve_pose_error": {
                str(side): {
                    str(name): float(value) for name, value in dict(metrics).items()
                }
                for side, metrics in dict(solve_pose_error).items()
            },
            "ik_damping": float(self.config.damping),
            "ik_solve_iteration_reserve_ms": float(
                self.config.solve_iteration_reserve_ms
            ),
            "frame_total_ms": float(elapsed_ms),
            "solve_time_budget_ms": budget_ms,
            "solve_time_budget_exceeded": bool(budget_ms > 0.0 and elapsed_ms > budget_ms),
            "pose_interpolation_stage": "upstream_reference_generator",
            "qpos_interpolation_stage": "downstream_reference_to_command_generator",
            "temporal_smoothing_applied": False,
            "target_pose_smoothing": "none",
            "output_qpos_smoothing": "none",
            "pose_lift": copy.deepcopy(self._last_pose_lift_metadata),
            "failure": None if failure is None else dict(failure),
            "source_metadata": dict(source_metadata),
        }

    def _solver_current_q(self) -> np.ndarray:
        current = np.asarray(getattr(self.solver, "current_q"), dtype=float).reshape(-1)
        if current.size == 0 or not np.all(np.isfinite(current)):
            raise ValueError("dual-arm collision solver current_q must be finite and non-empty.")
        return current.copy()

    def _set_external_qpos(self, full_q: np.ndarray) -> None:
        """Set a caller-owned qpos, optionally refreshing the Pink FK cache."""

        if self.config.refresh_external_qpos_cache:
            rebase_configuration_q = getattr(
                self.solver,
                "rebase_configuration_q",
                self.solver.set_configuration_q,
            )
            rebase_configuration_q(np.asarray(full_q, dtype=float).copy())
            return
        self.solver.set_configuration_q(np.asarray(full_q, dtype=float).copy())

    def _full_arm_tangent_limit(
        self,
        arm_limit: Sequence[float],
        *,
        full_dof: int,
    ) -> np.ndarray:
        values = np.asarray(arm_limit, dtype=float).reshape(ARM_DOF)
        full = np.full(int(full_dof), np.inf, dtype=float)
        for arm_slice in self._arm_slices.values():
            full[arm_slice] = values
        return full

    def _configure_qp_motion_constraints(self, current_full_q: np.ndarray) -> None:
        profile = self.config.pose_lift_profile
        full_dof = int(np.asarray(current_full_q).size)
        if profile.qp_substeps_per_reference > 0:
            set_integration_dt = getattr(self.solver, "set_integration_dt", None)
            if not callable(set_integration_dt):
                raise ValueError(
                    f"pose-lift profile {profile.name!r} requires an IK solver "
                    "with set_integration_dt()."
                )
            set_integration_dt(
                self.config.reference_period_sec
                / float(profile.qp_substeps_per_reference)
            )
        if profile.qp_velocity_limit or profile.qp_acceleration_limit:
            configure_motion_limit = getattr(
                self.solver,
                "configure_admitted_reference_motion_limit",
                None,
            )
            if not callable(configure_motion_limit):
                raise ValueError(
                    f"pose-lift profile {profile.name!r} requires an IK solver "
                    "with configure_admitted_reference_motion_limit()."
                )
            configure_motion_limit(
                self._full_arm_tangent_limit(
                    self.config.max_joint_velocity_rad_s,
                    full_dof=full_dof,
                ),
                (
                    self._full_arm_tangent_limit(
                        self.config.max_joint_acceleration_rad_s2,
                        full_dof=full_dof,
                    )
                    if profile.qp_acceleration_limit
                    else None
                ),
            )
            self._set_qp_motion_reference(
                current_full_q,
                self._last_admitted_velocity_full,
            )
        if profile.qp_sigma_escape_cost > 0.0:
            configure_escape = getattr(
                self.solver,
                "configure_normalized_sigma_escape_task",
                None,
            )
            if not callable(configure_escape):
                raise ValueError(
                    f"pose-lift profile {profile.name!r} requires an IK solver "
                    "with configure_normalized_sigma_escape_task()."
                )
            configure_escape(
                cost=profile.qp_sigma_escape_cost,
                guard_sigma=profile.qp_sigma_escape_guard,
                rate_per_sec=profile.qp_sigma_escape_rate_per_sec,
            )
            self._set_qp_motion_reference(
                current_full_q,
                self._last_admitted_velocity_full,
            )
        if profile.qp_structural_branch_margin_deg > 0.0:
            configure_branch_guard = getattr(
                self.solver,
                "configure_structural_branch_guard",
                None,
            )
            if not callable(configure_branch_guard):
                raise ValueError(
                    f"pose-lift profile {profile.name!r} requires an IK solver "
                    "with configure_structural_branch_guard()."
                )
            branch_kwargs = {
                "guard_sigma": profile.qp_structural_branch_guard_sigma,
                "margin_deg": profile.qp_structural_branch_margin_deg,
                "release_sigma": profile.qp_structural_branch_release_sigma,
            }
            if (
                profile.qp_structural_branch_constraint_mode != "hard_margin"
                or profile.qp_structural_branch_barrier_gain > 0.0
                or profile.qp_structural_branch_barrier_activation_sigma > 0.0
                or profile.qp_structural_branch_prediction_horizon_sec > 0.0
            ):
                branch_kwargs.update(
                    {
                        "constraint_mode": (
                            profile.qp_structural_branch_constraint_mode
                        ),
                        "barrier_gain": (
                            profile.qp_structural_branch_barrier_gain
                        ),
                        "barrier_activation_sigma": (
                            profile.qp_structural_branch_barrier_activation_sigma
                        ),
                        "prediction_horizon_sec": (
                            profile.qp_structural_branch_prediction_horizon_sec
                        ),
                    }
                )
            if profile.qp_structural_branch_escape_cost > 0.0:
                branch_kwargs.update(
                    {
                        "escape_cost": profile.qp_structural_branch_escape_cost,
                        "escape_target_deg": (
                            profile.qp_structural_branch_escape_target_deg
                        ),
                        "escape_guard_sigma": (
                            profile.qp_structural_branch_escape_guard_sigma
                        ),
                    }
                )
            configure_branch_guard(**branch_kwargs)
            self._set_qp_motion_reference(
                current_full_q,
                self._last_admitted_velocity_full,
            )

    def _set_qp_motion_reference(
        self,
        anchor_full_q: np.ndarray,
        velocity_full: np.ndarray,
    ) -> None:
        profile = self.config.pose_lift_profile
        if not (
            profile.qp_velocity_limit
            or profile.qp_acceleration_limit
            or profile.qp_sigma_escape_cost > 0.0
            or profile.qp_structural_branch_margin_deg > 0.0
        ):
            return
        setter = getattr(self.solver, "set_admitted_reference_motion", None)
        if not callable(setter):
            raise RuntimeError(
                "QP motion-limited solver lost set_admitted_reference_motion()."
            )
        anchor = np.asarray(anchor_full_q, dtype=float).reshape(
            self._last_admitted_velocity_full.shape
        )
        velocity = np.asarray(velocity_full, dtype=float).reshape(
            self._last_admitted_velocity_full.shape
        )
        if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(velocity)):
            raise ValueError("admitted qref motion anchor must be finite.")
        setter(anchor, velocity, self.config.reference_period_sec)

    def _commit_qp_admitted_velocity(
        self,
        *,
        reference_full_q: np.ndarray,
        admitted_full_q: np.ndarray,
    ) -> None:
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        admitted = np.asarray(admitted_full_q, dtype=float).reshape(reference.shape)
        velocity = (admitted - reference) / float(self.config.reference_period_sec)
        self._last_admitted_velocity_full = velocity.copy()
        self._set_qp_motion_reference(admitted, velocity)

    def _resolve_arm_slices(self, full_dof: int) -> dict[str, slice]:
        layout = str(getattr(self.scene_profile, "solution_layout", ""))
        if layout == "dual_arm_hand" and full_dof >= 24:
            return {LEFT: slice(0, 6), RIGHT: slice(18, 24)}
        if full_dof == 14:
            return {LEFT: slice(1, 7), RIGHT: slice(8, 14)}
        if full_dof == 12:
            return {LEFT: slice(0, 6), RIGHT: slice(6, 12)}
        raise ValueError(f"unsupported dual-arm collision solver q shape ({full_dof},).")

    def _split_arm_qpos(self, full_q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(full_q, dtype=float).reshape(-1)
        return {
            side: q[arm_slice].astype(float, copy=True)
            for side, arm_slice in self._arm_slices.items()
        }

    def _embed_arm_qpos(
        self,
        reference_full_q: np.ndarray,
        qpos_by_side: Mapping[str, Any],
    ) -> np.ndarray:
        full_q = np.asarray(reference_full_q, dtype=float).reshape(-1).copy()
        for side in SIDES:
            full_q[self._arm_slices[side]] = as_rm65_qpos(
                qpos_by_side[side], label=f"{side}.qpos"
            )
        return full_q

    def _effective_joint_limits(
        self,
        full_q: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        configured_lower = np.asarray(self.config.joint_lower_rad, dtype=float)
        configured_upper = np.asarray(self.config.joint_upper_rad, dtype=float)
        model = getattr(getattr(self.solver, "robot", None), "model", None)
        if model is None:
            model_lower_by_side = {side: configured_lower.copy() for side in SIDES}
            model_upper_by_side = {side: configured_upper.copy() for side in SIDES}
        else:
            model_lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
            model_upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)
            if model_lower.shape != full_q.shape or model_upper.shape != full_q.shape:
                raise ValueError("collision solver model joint-limit shape does not match current_q.")
            model_lower_by_side = self._split_arm_qpos(model_lower)
            model_upper_by_side = self._split_arm_qpos(model_upper)
        lower = {
            side: np.maximum(configured_lower, model_lower_by_side[side]) for side in SIDES
        }
        upper = {
            side: np.minimum(configured_upper, model_upper_by_side[side]) for side in SIDES
        }
        for side in SIDES:
            if (
                not np.all(np.isfinite(lower[side]))
                or not np.all(np.isfinite(upper[side]))
                or np.any(lower[side] >= upper[side])
            ):
                raise ValueError(f"{side} effective joint limits are invalid or empty.")
        return lower, upper

    def _require_arm_qpos_within_limits(self, full_q: np.ndarray, *, name: str) -> None:
        qpos = self._split_arm_qpos(full_q)
        for side in SIDES:
            if np.any(qpos[side] < self._joint_lower_by_side[side] - 1e-10) or np.any(
                qpos[side] > self._joint_upper_by_side[side] + 1e-10
            ):
                raise _PlanningFailure(
                    "joint_position_limit",
                    f"{name} violates {side} effective absolute joint limits.",
                )

    def _resolve_collision_pair_count(self) -> int:
        profile_pairs = tuple(getattr(self.scene_profile, "collision_pairs", ()) or ())
        if profile_pairs:
            return len(profile_pairs)
        collision_model = getattr(getattr(self.solver, "robot", None), "collision_model", None)
        return len(getattr(collision_model, "collisionPairs", ()) or ())

    def _collision_distance(
        self,
        full_q: np.ndarray,
    ) -> tuple[float | None, float, dict[str, object]]:
        started = time.perf_counter()
        details_method = getattr(
            self.solver,
            "compute_min_collision_distance_details",
            None,
        )
        details: dict[str, object] = {}
        if callable(details_method):
            raw_details = dict(
                details_method(np.asarray(full_q, dtype=float).copy()) or {}
            )
            value = raw_details.pop("minimum_distance_m", None)
            details = {
                str(name): item
                for name, item in raw_details.items()
                if item is None or isinstance(item, (bool, int, float, str))
            }
        else:
            method = getattr(self.solver, "compute_min_collision_distance", None)
            value = (
                None
                if not callable(method)
                else method(np.asarray(full_q, dtype=float).copy())
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if value is None:
            return None, elapsed_ms, details
        distance = float(value)
        if not math.isfinite(distance):
            return None, elapsed_ms, details
        return distance, elapsed_ms, details

    def _collision_safe(self, distance_m: float | None) -> tuple[bool, str | None]:
        if self._collision_pair_count <= 0:
            return True, None
        if distance_m is None:
            return False, "collision_distance_unavailable"
        threshold = float(self.config.collision_min_distance_m) - float(
            self.config.collision_distance_epsilon_m
        )
        if float(distance_m) < threshold:
            return False, "collision_margin_low"
        return True, None

    def _compute_frame_pose(self, frame_name: str, full_q: np.ndarray) -> np.ndarray:
        method = getattr(self.solver, "compute_frame_pose")
        try:
            pose = method(str(frame_name), np.asarray(full_q, dtype=float).copy())
        except TypeError:
            self.solver.set_configuration_q(np.asarray(full_q, dtype=float).copy())
            pose = method(str(frame_name))
        return _pose_matrix(pose, name=f"solver frame {frame_name!r}")

    def _realized_endpoint_poses(self, full_q: np.ndarray) -> dict[str, np.ndarray]:
        return {
            LEFT: self._compute_frame_pose(self.scene_profile.left_frame_name, full_q),
            RIGHT: self._compute_frame_pose(self.scene_profile.right_frame_name, full_q),
        }

    @staticmethod
    def _pose_to_xyzquat(pose: np.ndarray) -> np.ndarray:
        matrix = _pose_matrix(pose, name="solver target pose")
        quat = R.from_matrix(matrix[:3, :3]).as_quat()
        return np.concatenate((matrix[:3, 3], quat)).astype(float, copy=False)


__all__ = [
    "DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S",
    "DEFAULT_POSITION_COLLISION_REFERENCE_HZ",
    "RM65QposCollisionValidationResult",
    "RM65PositionCollisionPlanResult",
    "RM65PositionCollisionPlanner",
    "RM65PositionCollisionPlannerConfig",
    "RM65PositionPlannerProtocol",
    "load_world_from_fixed_link_transform",
    "load_world_from_arm_base_transforms",
]
