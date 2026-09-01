"""Pure contracts for simulation-first, fail-closed real trajectory replay.

These values carry state, identity, trajectory, and simulation evidence.  They
never own a backend, dispatch permit, execution segment, or hardware authority.
An admissible replay receipt is therefore a preflight result only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

SIDES = ("left", "right")
IDENTITY_HASH_DOMAIN = "agenticrobot.shadow_execution.identity.v1"
STATE_HASH_DOMAIN = "agenticrobot.shadow_execution.authoritative_state.v1"
POINT_HASH_DOMAIN = "agenticrobot.shadow_execution.joint_point.v1"
TRAJECTORY_HASH_DOMAIN = "agenticrobot.shadow_execution.registered_trajectory.v1"
SIM_RECEIPT_HASH_DOMAIN = "agenticrobot.shadow_execution.sim_receipt.v1"
ADMISSION_HASH_DOMAIN = "agenticrobot.shadow_execution.replay_admission.v1"


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _qpos_by_side(
    values: Mapping[str, Sequence[float] | np.ndarray],
    *,
    name: str,
) -> Mapping[str, np.ndarray]:
    if set(values) != set(SIDES):
        raise ValueError(f"{name} must contain exactly {SIDES}.")
    normalized: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = np.asarray(values[side], dtype=float).reshape(-1).copy()
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name}[{side!r}] must contain six finite values.")
        value.setflags(write=False)
        normalized[side] = value
    return MappingProxyType(normalized)


def _scalar_by_side(
    values: Mapping[str, object],
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> Mapping[str, float]:
    if set(values) != set(SIDES):
        raise ValueError(f"{name} must contain exactly {SIDES}.")
    normalized: dict[str, float] = {}
    for side in SIDES:
        value = _finite(values[side], name=f"{name}[{side!r}]")
        if lower is not None and value < lower:
            raise ValueError(f"{name}[{side!r}] must be >= {lower}.")
        if upper is not None and value > upper:
            raise ValueError(f"{name}[{side!r}] must be <= {upper}.")
        normalized[side] = value
    return MappingProxyType(normalized)


def _refs(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    refs = tuple(sorted({_required(value, name=name) for value in values}))
    if not refs:
        raise ValueError(f"{name} requires at least one value.")
    return refs


def _qpos_payload(values: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {side: values[side].tolist() for side in SIDES}


def _side_scalar_payload(values: Mapping[str, float]) -> dict[str, float]:
    return {side: float(values[side]) for side in SIDES}


@dataclass(frozen=True)
class ShadowExecutionIdentity:
    """Exact model/frame/calibration identity shared by state and replay."""

    station_revision: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_profile_hash_by_side: Mapping[str, str]
    robot_model_hash: str
    collision_model_hash: str
    joint_layout_hash: str
    gripper_model_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        scalar_fields = (
            "station_revision",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "robot_model_hash",
            "collision_model_hash",
            "joint_layout_hash",
            "gripper_model_hash",
        )
        normalized = {
            name: _required(getattr(self, name), name=name) for name in scalar_fields
        }
        raw_tools = dict(self.tool_profile_hash_by_side)
        if set(raw_tools) != set(SIDES):
            raise ValueError(f"tool_profile_hash_by_side must contain exactly {SIDES}.")
        tools = MappingProxyType(
            {
                side: _required(raw_tools[side], name=f"tool_profile_hash_by_side[{side}]")
                for side in SIDES
            }
        )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tool_profile_hash_by_side", tools)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(self.to_payload(include_hash=False), domain=IDENTITY_HASH_DOMAIN),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "station_revision": self.station_revision,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "frame_graph_hash": self.frame_graph_hash,
            "tool_profile_hash_by_side": dict(self.tool_profile_hash_by_side),
            "robot_model_hash": self.robot_model_hash,
            "collision_model_hash": self.collision_model_hash,
            "joint_layout_hash": self.joint_layout_hash,
            "gripper_model_hash": self.gripper_model_hash,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class AuthoritativeRobotStateSnapshot:
    """One causally complete dual-arm and dual-gripper measured state cut."""

    qpos_by_side: Mapping[str, Sequence[float] | np.ndarray]
    gripper_close_by_side: Mapping[str, float]
    arm_measured_at: float
    gripper_measured_at_by_side: Mapping[str, float]
    available_at: float
    arm_evidence_refs: tuple[str, ...]
    gripper_evidence_refs: tuple[str, ...]
    identity: ShadowExecutionIdentity
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ShadowExecutionIdentity):
            raise TypeError("identity must be ShadowExecutionIdentity.")
        qpos = _qpos_by_side(self.qpos_by_side, name="qpos_by_side")
        grippers = _scalar_by_side(
            self.gripper_close_by_side,
            name="gripper_close_by_side",
            lower=0.0,
            upper=1.0,
        )
        arm_time = _finite(self.arm_measured_at, name="arm_measured_at")
        gripper_times = _scalar_by_side(
            self.gripper_measured_at_by_side,
            name="gripper_measured_at_by_side",
        )
        available = _finite(self.available_at, name="available_at")
        if available < arm_time or any(available < value for value in gripper_times.values()):
            raise ValueError("available_at cannot precede a component measurement.")
        arm_refs = _refs(self.arm_evidence_refs, name="arm_evidence_ref")
        gripper_refs = _refs(self.gripper_evidence_refs, name="gripper_evidence_ref")
        object.__setattr__(self, "qpos_by_side", qpos)
        object.__setattr__(self, "gripper_close_by_side", grippers)
        object.__setattr__(self, "arm_measured_at", arm_time)
        object.__setattr__(self, "gripper_measured_at_by_side", gripper_times)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "arm_evidence_refs", arm_refs)
        object.__setattr__(self, "gripper_evidence_refs", gripper_refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(self.to_payload(include_hash=False), domain=STATE_HASH_DOMAIN),
        )

    @property
    def oldest_measurement_at(self) -> float:
        return min(self.arm_measured_at, *self.gripper_measured_at_by_side.values())

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "qpos_by_side": _qpos_payload(self.qpos_by_side),
            "gripper_close_by_side": _side_scalar_payload(self.gripper_close_by_side),
            "arm_measured_at": self.arm_measured_at,
            "gripper_measured_at_by_side": _side_scalar_payload(
                self.gripper_measured_at_by_side
            ),
            "available_at": self.available_at,
            "arm_evidence_refs": self.arm_evidence_refs,
            "gripper_evidence_refs": self.gripper_evidence_refs,
            "identity_hash": self.identity.semantic_content_hash,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class RegisteredJointTrajectoryPoint:
    """One exact dual-arm point on a monotonic relative-time lattice."""

    time_from_start_sec: float
    qpos_by_side: Mapping[str, Sequence[float] | np.ndarray]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        timestamp = _finite(self.time_from_start_sec, name="time_from_start_sec")
        if timestamp < 0.0:
            raise ValueError("time_from_start_sec must be non-negative.")
        qpos = _qpos_by_side(self.qpos_by_side, name="qpos_by_side")
        object.__setattr__(self, "time_from_start_sec", timestamp)
        object.__setattr__(self, "qpos_by_side", qpos)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "time_from_start_sec": timestamp,
                    "qpos_by_side": _qpos_payload(qpos),
                },
                domain=POINT_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class RegisteredShadowJointTrajectory:
    """Exact simulation-tested joint trajectory; never a dispatch authority."""

    trajectory_id: str
    created_at: float
    initial_snapshot: AuthoritativeRobotStateSnapshot
    scene_hash: str
    target_hash: str
    planner_config_hash: str
    moving_sides: tuple[str, ...]
    points: tuple[RegisteredJointTrajectoryPoint, ...]
    gripper_policy: str = "hold_measured"
    non_authoritative: bool = True
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        trajectory_id = _required(self.trajectory_id, name="trajectory_id")
        created_at = _finite(self.created_at, name="created_at")
        if not isinstance(self.initial_snapshot, AuthoritativeRobotStateSnapshot):
            raise TypeError("initial_snapshot must be AuthoritativeRobotStateSnapshot.")
        if created_at < self.initial_snapshot.available_at:
            raise ValueError("created_at cannot precede the initial snapshot.")
        scene_hash = _required(self.scene_hash, name="scene_hash")
        target_hash = _required(self.target_hash, name="target_hash")
        planner_hash = _required(self.planner_config_hash, name="planner_config_hash")
        moving = tuple(_required(side, name="moving_side").lower() for side in self.moving_sides)
        if not moving or len(moving) != len(set(moving)) or any(side not in SIDES for side in moving):
            raise ValueError("moving_sides must be a non-empty unique subset of left/right.")
        points = tuple(self.points)
        if len(points) < 2 or any(
            not isinstance(point, RegisteredJointTrajectoryPoint) for point in points
        ):
            raise ValueError("points must contain at least two registered points.")
        if points[0].time_from_start_sec != 0.0:
            raise ValueError("the first trajectory point must start at t=0.")
        previous = -1.0
        for point in points:
            if point.time_from_start_sec <= previous:
                raise ValueError("trajectory point times must be strictly increasing.")
            previous = point.time_from_start_sec
        for side in SIDES:
            if not np.array_equal(
                points[0].qpos_by_side[side], self.initial_snapshot.qpos_by_side[side]
            ):
                raise ValueError(f"first point does not match initial {side} qpos.")
            if side not in moving and any(
                not np.array_equal(point.qpos_by_side[side], points[0].qpos_by_side[side])
                for point in points[1:]
            ):
                raise ValueError(f"inactive {side} arm must remain constant.")
        gripper_policy = _required(self.gripper_policy, name="gripper_policy")
        if gripper_policy != "hold_measured":
            raise ValueError("only hold_measured gripper policy is supported.")
        if self.non_authoritative is not True:
            raise ValueError("registered shadow trajectories must be non-authoritative.")
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "scene_hash", scene_hash)
        object.__setattr__(self, "target_hash", target_hash)
        object.__setattr__(self, "planner_config_hash", planner_hash)
        object.__setattr__(self, "moving_sides", moving)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "gripper_policy", gripper_policy)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "trajectory_id": trajectory_id,
                    "created_at": created_at,
                    "initial_snapshot_hash": self.initial_snapshot.semantic_content_hash,
                    "identity_hash": self.initial_snapshot.identity.semantic_content_hash,
                    "scene_hash": scene_hash,
                    "target_hash": target_hash,
                    "planner_config_hash": planner_hash,
                    "moving_sides": moving,
                    "point_hashes": tuple(point.semantic_content_hash for point in points),
                    "gripper_policy": gripper_policy,
                    "non_authoritative": True,
                },
                domain=TRAJECTORY_HASH_DOMAIN,
            ),
        )


class ShadowSimulationStatus(str, Enum):
    COMPLETED = "completed"
    EXECUTION_UNCERTIFIED = "execution_uncertified"
    MODEL_IDENTITY_MISMATCH = "model_identity_mismatch"
    TRAJECTORY_START_MISMATCH = "trajectory_start_mismatch"
    ENDPOINT_REJECTED = "endpoint_rejected"
    PATH_START_COLLISION_BLOCKED = "path_start_collision_blocked"
    PATH_TARGET_COLLISION_BLOCKED = "path_target_collision_blocked"
    PATH_COLLISION_BLOCKED = "path_collision_blocked"
    PATH_SEARCH_EXHAUSTED = "path_search_exhausted"
    PATH_PLANNER_ERROR = "path_planner_error"
    POINT_TRACKING_TIMEOUT = "point_tracking_timeout"
    GRIPPER_HOLD_DRIFT = "gripper_hold_drift"
    TERMINAL_HOLD_FAILED = "terminal_hold_failed"
    IK_TARGET_UNMET = "ik_target_unmet"
    TRACKING_INCOMPLETE = "tracking_incomplete"


@dataclass(frozen=True)
class ShadowSimulationReceipt:
    """Terminal simulation evidence bound to one exact registered trajectory."""

    trajectory_hash: str
    initial_snapshot_hash: str
    identity_hash: str
    scene_hash: str
    status: ShadowSimulationStatus
    executed_point_count: int
    terminal_hold_confirmed: bool
    exact_planned_qpos_execution_confirmed: bool
    model_identity_confirmed: bool
    evidence_refs: tuple[str, ...]
    minimum_clearance_m: float | None = None
    non_authoritative: bool = True
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        trajectory_hash = _required(self.trajectory_hash, name="trajectory_hash")
        snapshot_hash = _required(self.initial_snapshot_hash, name="initial_snapshot_hash")
        identity_hash = _required(self.identity_hash, name="identity_hash")
        scene_hash = _required(self.scene_hash, name="scene_hash")
        if not isinstance(self.status, ShadowSimulationStatus):
            raise TypeError("status must be ShadowSimulationStatus.")
        if (
            isinstance(self.executed_point_count, bool)
            or not isinstance(self.executed_point_count, int)
            or self.executed_point_count < 0
        ):
            raise ValueError("executed_point_count must be a non-negative integer.")
        if not isinstance(self.terminal_hold_confirmed, bool):
            raise TypeError("terminal_hold_confirmed must be boolean.")
        if not isinstance(self.exact_planned_qpos_execution_confirmed, bool):
            raise TypeError("exact_planned_qpos_execution_confirmed must be boolean.")
        if not isinstance(self.model_identity_confirmed, bool):
            raise TypeError("model_identity_confirmed must be boolean.")
        if self.status is ShadowSimulationStatus.COMPLETED and (
            not self.terminal_hold_confirmed
            or not self.exact_planned_qpos_execution_confirmed
            or not self.model_identity_confirmed
        ):
            raise ValueError(
                "completed simulation requires exact qpos execution, verified model "
                "identity, and terminal HOLD."
            )
        clearance = self.minimum_clearance_m
        if clearance is not None:
            clearance = _finite(clearance, name="minimum_clearance_m")
        refs = _refs(self.evidence_refs, name="simulation_evidence_ref")
        if self.non_authoritative is not True:
            raise ValueError("simulation receipts must be non-authoritative.")
        object.__setattr__(self, "trajectory_hash", trajectory_hash)
        object.__setattr__(self, "initial_snapshot_hash", snapshot_hash)
        object.__setattr__(self, "identity_hash", identity_hash)
        object.__setattr__(self, "scene_hash", scene_hash)
        object.__setattr__(self, "executed_point_count", int(self.executed_point_count))
        object.__setattr__(self, "minimum_clearance_m", clearance)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "trajectory_hash": trajectory_hash,
                    "initial_snapshot_hash": snapshot_hash,
                    "identity_hash": identity_hash,
                    "scene_hash": scene_hash,
                    "status": self.status.value,
                    "executed_point_count": int(self.executed_point_count),
                    "terminal_hold_confirmed": self.terminal_hold_confirmed,
                    "exact_planned_qpos_execution_confirmed": (
                        self.exact_planned_qpos_execution_confirmed
                    ),
                    "model_identity_confirmed": self.model_identity_confirmed,
                    "minimum_clearance_m": clearance,
                    "evidence_refs": refs,
                    "non_authoritative": True,
                },
                domain=SIM_RECEIPT_HASH_DOMAIN,
            ),
        )


class ShadowReplayAdmissionStatus(str, Enum):
    ADMISSIBLE_PREFLIGHT_ONLY = "admissible_preflight_only"
    TRAJECTORY_BINDING_MISMATCH = "trajectory_binding_mismatch"
    SIM_EXECUTION_UNCONFIRMED = "sim_execution_unconfirmed"
    SIM_TERMINAL_HOLD_UNCONFIRMED = "sim_terminal_hold_unconfirmed"
    IDENTITY_MISMATCH = "identity_mismatch"
    SCENE_MISMATCH = "scene_mismatch"
    SNAPSHOT_STALE = "snapshot_stale"
    ARM_START_STATE_DRIFT = "arm_start_state_drift"
    GRIPPER_START_STATE_DRIFT = "gripper_start_state_drift"


@dataclass(frozen=True)
class ShadowReplayAdmissionReceipt:
    """Non-authoritative preflight decision; never a real motion permit."""

    status: ShadowReplayAdmissionStatus
    reason_codes: tuple[str, ...]
    current_snapshot_hash: str
    trajectory_hash: str
    simulation_receipt_hash: str
    evaluated_at: float
    maximum_arm_drift_rad: float
    maximum_gripper_drift: float
    preflight_only: bool = True
    real_motion_authorized: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ShadowReplayAdmissionStatus):
            raise TypeError("status must be ShadowReplayAdmissionStatus.")
        reasons = tuple(_required(value, name="reason_code") for value in self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must be unique.")
        if self.status is ShadowReplayAdmissionStatus.ADMISSIBLE_PREFLIGHT_ONLY:
            if reasons:
                raise ValueError("admissible preflight cannot contain rejection reasons.")
        elif not reasons or reasons[0] != self.status.value:
            raise ValueError("the first reason code must match rejection status.")
        if self.preflight_only is not True or self.real_motion_authorized is not False:
            raise ValueError("replay admission is preflight-only and cannot authorize motion.")
        current_hash = _required(self.current_snapshot_hash, name="current_snapshot_hash")
        trajectory_hash = _required(self.trajectory_hash, name="trajectory_hash")
        sim_hash = _required(self.simulation_receipt_hash, name="simulation_receipt_hash")
        evaluated_at = _finite(self.evaluated_at, name="evaluated_at")
        arm_drift = _finite(self.maximum_arm_drift_rad, name="maximum_arm_drift_rad")
        gripper_drift = _finite(
            self.maximum_gripper_drift, name="maximum_gripper_drift"
        )
        if arm_drift < 0.0 or gripper_drift < 0.0:
            raise ValueError("reported drift cannot be negative.")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "current_snapshot_hash", current_hash)
        object.__setattr__(self, "trajectory_hash", trajectory_hash)
        object.__setattr__(self, "simulation_receipt_hash", sim_hash)
        object.__setattr__(self, "evaluated_at", evaluated_at)
        object.__setattr__(self, "maximum_arm_drift_rad", arm_drift)
        object.__setattr__(self, "maximum_gripper_drift", gripper_drift)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "status": self.status.value,
                    "reason_codes": reasons,
                    "current_snapshot_hash": current_hash,
                    "trajectory_hash": trajectory_hash,
                    "simulation_receipt_hash": sim_hash,
                    "evaluated_at": evaluated_at,
                    "maximum_arm_drift_rad": arm_drift,
                    "maximum_gripper_drift": gripper_drift,
                    "preflight_only": True,
                    "real_motion_authorized": False,
                },
                domain=ADMISSION_HASH_DOMAIN,
            ),
        )

    @property
    def admissible(self) -> bool:
        return self.status is ShadowReplayAdmissionStatus.ADMISSIBLE_PREFLIGHT_ONLY


def evaluate_shadow_replay_admission(
    *,
    current_snapshot: AuthoritativeRobotStateSnapshot,
    trajectory: RegisteredShadowJointTrajectory,
    simulation_receipt: ShadowSimulationReceipt,
    current_scene_hash: str,
    evaluated_at: float,
    maximum_snapshot_age_sec: float,
    maximum_arm_start_drift_rad: float,
    maximum_gripper_start_drift: float,
) -> ShadowReplayAdmissionReceipt:
    """Evaluate replay preconditions without dispatching or granting authority."""

    if not isinstance(current_snapshot, AuthoritativeRobotStateSnapshot):
        raise TypeError("current_snapshot must be AuthoritativeRobotStateSnapshot.")
    if not isinstance(trajectory, RegisteredShadowJointTrajectory):
        raise TypeError("trajectory must be RegisteredShadowJointTrajectory.")
    if not isinstance(simulation_receipt, ShadowSimulationReceipt):
        raise TypeError("simulation_receipt must be ShadowSimulationReceipt.")
    scene_hash = _required(current_scene_hash, name="current_scene_hash")
    now = _finite(evaluated_at, name="evaluated_at")
    max_age = _finite(maximum_snapshot_age_sec, name="maximum_snapshot_age_sec")
    max_arm = _finite(maximum_arm_start_drift_rad, name="maximum_arm_start_drift_rad")
    max_gripper = _finite(
        maximum_gripper_start_drift, name="maximum_gripper_start_drift"
    )
    if min(max_age, max_arm, max_gripper) < 0.0:
        raise ValueError("age and drift limits must be non-negative.")

    initial = trajectory.initial_snapshot
    arm_drift = max(
        float(np.max(np.abs(current_snapshot.qpos_by_side[side] - initial.qpos_by_side[side])))
        for side in SIDES
    )
    gripper_drift = max(
        abs(
            float(current_snapshot.gripper_close_by_side[side])
            - float(initial.gripper_close_by_side[side])
        )
        for side in SIDES
    )
    reasons: list[str] = []
    binding_matches = (
        simulation_receipt.trajectory_hash == trajectory.semantic_content_hash
        and simulation_receipt.initial_snapshot_hash
        == initial.semantic_content_hash
        and simulation_receipt.identity_hash
        == initial.identity.semantic_content_hash
        and simulation_receipt.scene_hash == trajectory.scene_hash
    )
    if not binding_matches:
        reasons.append(ShadowReplayAdmissionStatus.TRAJECTORY_BINDING_MISMATCH.value)
    if (
        simulation_receipt.status is not ShadowSimulationStatus.COMPLETED
        or not simulation_receipt.exact_planned_qpos_execution_confirmed
        or not simulation_receipt.model_identity_confirmed
        or simulation_receipt.executed_point_count != len(trajectory.points)
    ):
        reasons.append(ShadowReplayAdmissionStatus.SIM_EXECUTION_UNCONFIRMED.value)
    if not simulation_receipt.terminal_hold_confirmed:
        reasons.append(ShadowReplayAdmissionStatus.SIM_TERMINAL_HOLD_UNCONFIRMED.value)
    if (
        current_snapshot.identity.semantic_content_hash
        != initial.identity.semantic_content_hash
    ):
        reasons.append(ShadowReplayAdmissionStatus.IDENTITY_MISMATCH.value)
    if scene_hash != trajectory.scene_hash:
        reasons.append(ShadowReplayAdmissionStatus.SCENE_MISMATCH.value)
    if now < current_snapshot.available_at or now - current_snapshot.oldest_measurement_at > max_age:
        reasons.append(ShadowReplayAdmissionStatus.SNAPSHOT_STALE.value)
    if arm_drift > max_arm:
        reasons.append(ShadowReplayAdmissionStatus.ARM_START_STATE_DRIFT.value)
    if gripper_drift > max_gripper:
        reasons.append(ShadowReplayAdmissionStatus.GRIPPER_START_STATE_DRIFT.value)

    status = (
        ShadowReplayAdmissionStatus.ADMISSIBLE_PREFLIGHT_ONLY
        if not reasons
        else ShadowReplayAdmissionStatus(reasons[0])
    )
    return ShadowReplayAdmissionReceipt(
        status=status,
        reason_codes=tuple(reasons),
        current_snapshot_hash=current_snapshot.semantic_content_hash,
        trajectory_hash=trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation_receipt.semantic_content_hash,
        evaluated_at=now,
        maximum_arm_drift_rad=arm_drift,
        maximum_gripper_drift=gripper_drift,
    )
