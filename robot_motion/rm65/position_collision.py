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


DEFAULT_POSITION_COLLISION_REFERENCE_HZ = DEFAULT_POSITION_REFERENCE_HZ
DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S = (
    0.35,
    0.35,
    0.35,
    0.50,
    0.50,
    0.50,
)
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
    posture_cost: float = DEFAULT_DUAL_ARM_POSTURE_COST
    collision_min_distance_m: float = DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M
    collision_distance_epsilon_m: float = DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M
    collision_barrier_gain: float = DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN
    collision_safe_displacement_gain: float = DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN
    max_joint_velocity_rad_s: tuple[float, ...] | Sequence[float] = (
        DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S
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
    debug_profile: bool = False
    ik_tuning: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        reference_hz = _finite_positive(self.reference_hz, name="reference_hz")
        scene_profile = str(self.scene_profile or DEFAULT_COLLISION_SCENE).strip()
        if not scene_profile:
            raise ValueError("scene_profile must be non-empty.")
        posture_cost = float(self.posture_cost)
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

    urdf_value = getattr(scene_profile_or_urdf, "urdf_path", scene_profile_or_urdf)
    urdf_path = Path(urdf_value).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"RM65 collision scene URDF does not exist: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    joints_by_name: dict[str, ET.Element] = {}
    fixed_joint_by_child: dict[str, tuple[str, np.ndarray, str]] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name") or "").strip()
        if name:
            joints_by_name[name] = joint
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = str(parent.get("link") or "").strip()
        child_link = str(child.get("link") or "").strip()
        if not parent_link or not child_link:
            continue
        if child_link in fixed_joint_by_child:
            raise ValueError(f"URDF link {child_link!r} has multiple parent joints.")
        if str(joint.get("type") or "").strip() == "fixed":
            fixed_joint_by_child[child_link] = (parent_link, _urdf_joint_origin(joint), name)

    requested = dict(_DEFAULT_MOUNT_JOINT_NAMES if mount_joint_names is None else mount_joint_names)
    normalized_mounts: dict[str, str] = {}
    for raw_side, joint_name in requested.items():
        normalized_mounts[normalize_side(str(raw_side))] = str(joint_name).strip()
    if set(normalized_mounts) != set(SIDES):
        raise ValueError("mount_joint_names must contain left and right mount joints.")

    cache: dict[str, np.ndarray] = {"world": np.eye(4, dtype=float)}

    def world_from_link(link_name: str, stack: tuple[str, ...] = ()) -> np.ndarray:
        if link_name in cache:
            return cache[link_name].copy()
        if link_name in stack:
            raise ValueError(f"Cycle detected in fixed URDF chain: {(*stack, link_name)}")
        edge = fixed_joint_by_child.get(link_name)
        if edge is None:
            raise ValueError(
                f"Cannot resolve world transform for link {link_name!r}; "
                "its parent chain is missing or contains a non-fixed joint."
            )
        parent_link, parent_from_child, _joint_name = edge
        pose = world_from_link(parent_link, (*stack, link_name)) @ parent_from_child
        pose = _pose_matrix(pose, name=f"world_from_link[{link_name}]")
        cache[link_name] = pose.copy()
        return pose

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
        transforms[side] = world_from_link(child_link)
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
        self._joint_lower_by_side, self._joint_upper_by_side = self._effective_joint_limits(
            current_full_q
        )
        self._collision_pair_count = self._resolve_collision_pair_count()
        self._lock = threading.RLock()
        self._last_safe_full_q: np.ndarray | None = None
        self._last_safe_collision_distance_m: float | None = None
        self._next_sequence = 0

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
            self.solver.set_configuration_q(full_q.copy())
            distance, distance_ms = self._collision_distance(full_q)
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

                solver_targets = [
                    self._pose_to_xyzquat(target_endpoint_world[LEFT]),
                    self._pose_to_xyzquat(target_endpoint_world[RIGHT]),
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
                for iteration in range(1, int(self.config.solve_max_iterations) + 1):
                    if (
                        iteration > int(self.config.solve_min_iterations)
                        and iteration_budget_ms is not None
                    ):
                        elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
                        estimated_next_ms = max(0.05, float(previous_solve_ms or 0.0))
                        if elapsed_ms + estimated_next_ms > iteration_budget_ms:
                            solve_stop_reason = "iteration_budget_estimate"
                            break
                    iteration_started = time.perf_counter()
                    raw_full_q = np.asarray(
                        self.solver.solve(targets=solver_targets),
                        dtype=float,
                    ).reshape(-1)
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
                        iteration >= int(self.config.solve_min_iterations)
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

                distance, distance_ms = self._collision_distance(limited_full_q)
                collision_attempts.append(
                    {
                        "phase": "post_joint_limits",
                        "scale": 1.0,
                        "distance_m": distance,
                        "elapsed_ms": distance_ms,
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
                        candidate_distance, candidate_ms = self._collision_distance(
                            candidate_full_q
                        )
                        collision_attempts.append(
                            {
                                "phase": "post_joint_limits_collision_backoff",
                                "scale": float(scale),
                                "distance_m": candidate_distance,
                                "elapsed_ms": candidate_ms,
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
                realized_endpoint_world = self._realized_endpoint_poses(accepted_full_q)
                q_ref_by_side = self._split_arm_qpos(accepted_full_q)
                q_ik_raw_by_side = self._split_arm_qpos(raw_full_q)
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
            if reason is None:
                try:
                    distance, distance_ms = self._collision_distance(full_q)
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
            "target_reference_frame": reference_frame,
            "target_controlled_frame": controlled_frame,
            "world_from_arm_base_source": self.world_from_arm_base_source,
            "passive_sides": list(passive_sides),
            "passive_side_policy": "freeze_at_last_safe_qpos_in_full_collision_model",
            "solve_ms": float(solve_ms),
            "ik_solve_iterations": int(solve_iterations),
            "ik_solve_iteration_min": int(self.config.solve_min_iterations),
            "ik_solve_iteration_max": int(self.config.solve_max_iterations),
            "ik_solve_iteration_mode": (
                "adaptive_budget"
                if self.config.solve_adaptive_iterations_enabled
                else "fixed_max"
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
            "failure": None if failure is None else dict(failure),
            "source_metadata": dict(source_metadata),
        }

    def _solver_current_q(self) -> np.ndarray:
        current = np.asarray(getattr(self.solver, "current_q"), dtype=float).reshape(-1)
        if current.size == 0 or not np.all(np.isfinite(current)):
            raise ValueError("dual-arm collision solver current_q must be finite and non-empty.")
        return current.copy()

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

    def _collision_distance(self, full_q: np.ndarray) -> tuple[float | None, float]:
        started = time.perf_counter()
        method = getattr(self.solver, "compute_min_collision_distance", None)
        value = None if not callable(method) else method(np.asarray(full_q, dtype=float).copy())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if value is None:
            return None, elapsed_ms
        distance = float(value)
        if not math.isfinite(distance):
            return None, elapsed_ms
        return distance, elapsed_ms

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
    "load_world_from_arm_base_transforms",
]
