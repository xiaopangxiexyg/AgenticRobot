"""Lower-owned mailbox for one exclusive real Skill motion operation.

The port contains no Gateway, robot backend, transport, RealTeleop, GUI, or
control-composition dependency. A Gateway worker may only submit immutable
target/HOLD requests. One lower control owner applies them, constructs commands
through an injected coupled target solver, and retains the sole safety,
dispatch, and HOLD paths.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.rm65.dual_arm_target_solver import (
    BimanualPoseTarget,
    BimanualPoseTargetSource,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.command_override import TeleopCommandOverride
from teleop_runtime.transforms import validate_pose_matrix

REAL_AGENT_MOTION_COMMAND_METADATA_KEY = "real_agent_motion_operation"
REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD = 0.01
REAL_AGENT_TRAJECTORY_CLOCK_MAX_ADVANCE_SEC = 0.10


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _immutable_pose(value: object, *, field_name: str) -> np.ndarray:
    valid, reason = validate_pose_matrix(value)
    if not valid:
        raise ValueError(f"{field_name} must be a rigid 4x4 pose: {reason}")
    pose = np.asarray(value, dtype=float).copy()
    pose.setflags(write=False)
    return pose


class RealAgentMotionPhase(str, Enum):
    INACTIVE = "inactive"
    PENDING_ACTIVATION = "pending_activation"
    ACTIVE = "active"
    HOLD_REQUESTED = "hold_requested"
    HOLD_APPLYING = "hold_applying"
    HELD = "held"
    FAULTED = "faulted"


@dataclass(frozen=True)
class RealAgentMotionPortConfig:
    """Exact lower execution binding; it grants no activation by itself."""

    active_sides: tuple[str, ...]
    component_id_by_side: Mapping[str, str]
    execution_frame_id: str
    execution_frame_spec_hash: str
    exclusive_command_claim: bool = False
    trajectory_clock_max_advance_sec: float = (
        REAL_AGENT_TRAJECTORY_CLOCK_MAX_ADVANCE_SEC
    )

    def __post_init__(self) -> None:
        active_sides = tuple(str(side).strip().lower() for side in self.active_sides)
        if not active_sides or len(set(active_sides)) != len(active_sides):
            raise ValueError("active_sides must contain unique values.")
        if not set(active_sides) <= {"left", "right"}:
            raise ValueError("active_sides must be a subset of left/right.")
        raw_components = dict(self.component_id_by_side)
        if set(raw_components) != set(active_sides):
            raise ValueError(
                "component_id_by_side must cover exactly the configured active_sides."
            )
        components = {
            side: _required(raw_components[side], field_name=f"component_id_by_side.{side}")
            for side in active_sides
        }
        if len(set(components.values())) != len(components):
            raise ValueError("component_id_by_side values must be unique.")
        object.__setattr__(self, "active_sides", active_sides)
        object.__setattr__(
            self,
            "component_id_by_side",
            MappingProxyType(components),
        )
        object.__setattr__(
            self,
            "execution_frame_id",
            _required(self.execution_frame_id, field_name="execution_frame_id"),
        )
        object.__setattr__(
            self,
            "execution_frame_spec_hash",
            _required(
                self.execution_frame_spec_hash,
                field_name="execution_frame_spec_hash",
            ),
        )
        object.__setattr__(
            self,
            "exclusive_command_claim",
            bool(self.exclusive_command_claim),
        )
        clock_max_advance = _finite(
            self.trajectory_clock_max_advance_sec,
            field_name="trajectory_clock_max_advance_sec",
        )
        if clock_max_advance <= 0.0:
            raise ValueError("trajectory_clock_max_advance_sec must be positive.")
        object.__setattr__(
            self,
            "trajectory_clock_max_advance_sec",
            clock_max_advance,
        )


@dataclass(frozen=True)
class RealAgentMotionOwnerIdentity:
    """Exact identities owned by one exclusive real Skill command port."""

    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    target_binding_id: str
    execution_manifest_hash: str
    backend_inventory_hash: str
    clock_domain_id: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__,
            domain="dexteleop.real_agent_motion.owner_identity.v1",
        )


@dataclass(frozen=True)
class RealAgentMotionOwnerCut:
    """Lower-owned command-override state captured at a control-loop cut."""

    identity: RealAgentMotionOwnerIdentity
    owner_token_id: str
    owner_kind: str
    owner_generation: int
    authority_epoch: int
    execution_epoch: int
    lifecycle_state: str
    execution_gate: str
    owner_state_revision: int
    captured_at: float
    attachment_evidence_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RealAgentMotionOwnerIdentity):
            raise TypeError("identity must be RealAgentMotionOwnerIdentity.")
        for field_name in (
            "owner_token_id",
            "owner_kind",
            "lifecycle_state",
            "execution_gate",
            "attachment_evidence_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "owner_generation",
            "authority_epoch",
            "execution_epoch",
            "owner_state_revision",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")
        object.__setattr__(
            self,
            "captured_at",
            _finite(self.captured_at, field_name="captured_at"),
        )
        expected_gate = {
            "running": "ready",
            "hold": "hold",
            "fault": "fault",
        }.get(self.lifecycle_state)
        if expected_gate != self.execution_gate:
            raise ValueError("owner lifecycle_state/execution_gate mismatch.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "identity_hash": self.identity.semantic_content_hash,
                **{
                    name: value
                    for name, value in self.__dict__.items()
                    if name != "identity"
                },
            },
            domain="dexteleop.real_agent_motion.owner_cut.v1",
        )

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return (
            self.attachment_evidence_ref,
            "real-agent-motion-owner-cut:" + self.semantic_content_hash,
        )


@dataclass(frozen=True)
class RealAgentMotionActivation:
    """One operator-granted absolute tool-TCP target in the execution frame."""

    operation_id: str
    request_id: str
    activation_grant_id: str
    activation_grant_hash: str
    component_id: str
    side: str
    primitive_id: str
    primitive_hash: str
    target_pose_world: np.ndarray
    execution_frame_id: str
    execution_frame_spec_hash: str
    measured_state_revision: int
    measured_state_hash: str
    submitted_at: float
    expires_at: float
    speed_scale: float = 1.0
    joint_trajectory: RealAgentJointTrajectory | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "operation_id",
            "request_id",
            "activation_grant_id",
            "activation_grant_hash",
            "component_id",
            "primitive_id",
            "primitive_hash",
            "execution_frame_id",
            "execution_frame_spec_hash",
            "measured_state_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        if (
            isinstance(self.measured_state_revision, bool)
            or not isinstance(self.measured_state_revision, int)
            or self.measured_state_revision < 0
        ):
            raise ValueError("measured_state_revision must be a non-negative integer.")
        submitted_at = _finite(self.submitted_at, field_name="submitted_at")
        expires_at = _finite(self.expires_at, field_name="expires_at")
        if expires_at <= submitted_at:
            raise ValueError("expires_at must be later than submitted_at.")
        object.__setattr__(self, "submitted_at", submitted_at)
        object.__setattr__(self, "expires_at", expires_at)
        speed_scale = _finite(self.speed_scale, field_name="speed_scale")
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError("speed_scale must satisfy 0 < speed_scale <= 1.")
        object.__setattr__(self, "speed_scale", speed_scale)
        if self.joint_trajectory is not None and not isinstance(
            self.joint_trajectory, RealAgentJointTrajectory
        ):
            raise TypeError("joint_trajectory must be typed or None.")
        object.__setattr__(
            self,
            "target_pose_world",
            _immutable_pose(self.target_pose_world, field_name="target_pose_world"),
        )


@dataclass(frozen=True)
class RealAgentJointTrajectory:
    """One request-bound dual-arm qpos reference generated before activation."""

    time_from_start_sec: np.ndarray
    qpos_by_side: Mapping[str, np.ndarray]
    planner_evidence_refs: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        times = np.asarray(self.time_from_start_sec, dtype=float).reshape(-1).copy()
        if (
            times.size < 2
            or not np.all(np.isfinite(times))
            or abs(float(times[0])) > 1e-12
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "time_from_start_sec must start at zero and strictly increase."
            )
        qpos = {}
        for side in ("left", "right"):
            values = np.asarray(dict(self.qpos_by_side).get(side), dtype=float).copy()
            if values.shape != (times.size, 6) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"qpos_by_side.{side} must be sample_count x 6 and finite."
                )
            values.setflags(write=False)
            qpos[side] = values
        refs = tuple(sorted({str(value).strip() for value in self.planner_evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("planner_evidence_refs must be non-empty.")
        times.setflags(write=False)
        object.__setattr__(self, "time_from_start_sec", times)
        object.__setattr__(self, "qpos_by_side", MappingProxyType(qpos))
        object.__setattr__(self, "planner_evidence_refs", refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "time_from_start_sec": times,
                    "qpos_by_side": qpos,
                    "planner_evidence_refs": refs,
                },
                domain="dexteleop.real_agent_motion.joint_trajectory.v1",
            ),
        )

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return (
            *self.planner_evidence_refs,
            "real-agent-joint-trajectory:" + self.semantic_content_hash,
        )


@dataclass(frozen=True)
class RealAgentMotionMeasuredQposCut:
    revision: int
    measured_at: float
    available_at: float
    qpos_by_side: Mapping[str, np.ndarray]
    source_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        measured_at = _finite(self.measured_at, field_name="measured_at")
        available_at = _finite(self.available_at, field_name="available_at")
        if available_at < measured_at:
            raise ValueError("available_at cannot precede measured_at.")
        qpos = {}
        for side in ("left", "right"):
            values = np.asarray(dict(self.qpos_by_side).get(side), dtype=float).reshape(-1)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"qpos_by_side.{side} must contain six finite values.")
            frozen = values.copy()
            frozen.setflags(write=False)
            qpos[side] = frozen
        refs = tuple(sorted({str(value).strip() for value in self.source_evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("source_evidence_refs must be non-empty.")
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "qpos_by_side", MappingProxyType(qpos))
        object.__setattr__(self, "source_evidence_refs", refs)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "revision": self.revision,
                "measured_at": self.measured_at,
                "available_at": self.available_at,
                "qpos_by_side": dict(self.qpos_by_side),
                "source_evidence_refs": self.source_evidence_refs,
            },
            domain="dexteleop.real_agent_motion.measured_qpos_cut.v1",
        )

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return (
            *self.source_evidence_refs,
            "real-agent-measured-qpos:" + self.semantic_content_hash,
        )


@dataclass(frozen=True)
class RealAgentMotionPortSnapshot:
    revision: int
    phase: RealAgentMotionPhase
    operation_id: str | None
    request_id: str | None
    activation_grant_id: str | None
    activated_at: float | None
    last_command_at: float | None
    dispatch_count: int
    hold_reason: str | None
    hold_requested_at: float | None
    hold_applied_at: float | None
    hold_succeeded: bool | None
    fault: str | None
    trajectory_elapsed_sec: float | None = None
    trajectory_clock_rebased_count: int = 0
    trajectory_clock_last_capped: bool = False

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "phase": self.phase.value,
            },
            domain="dexteleop.real_agent_motion.port_snapshot.v1",
        )

    @property
    def evidence_ref(self) -> str:
        return "real-agent-motion-port:" + self.semantic_content_hash


@dataclass(frozen=True)
class RealAgentMotionMeasuredPoseCut:
    """One immutable control-thread publication of measured tool-TCP poses."""

    revision: int
    measured_at: float
    available_at: float
    pose_world_by_component: Mapping[str, np.ndarray]
    source_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        measured_at = _finite(self.measured_at, field_name="measured_at")
        available_at = _finite(self.available_at, field_name="available_at")
        if available_at < measured_at:
            raise ValueError("available_at cannot precede measured_at.")
        poses = {
            _required(component_id, field_name="component_id"): _immutable_pose(
                pose,
                field_name=f"pose_world_by_component.{component_id}",
            )
            for component_id, pose in dict(self.pose_world_by_component).items()
        }
        if not poses:
            raise ValueError("pose_world_by_component must not be empty.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="source_evidence_ref")
                    for value in self.source_evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("A measured pose cut requires source evidence.")
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "pose_world_by_component", MappingProxyType(poses))
        object.__setattr__(self, "source_evidence_refs", refs)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "revision": self.revision,
                "measured_at": self.measured_at,
                "available_at": self.available_at,
                "pose_world_by_component": {
                    component_id: tuple(float(value) for value in pose.reshape(-1))
                    for component_id, pose in self.pose_world_by_component.items()
                },
                "source_evidence_refs": self.source_evidence_refs,
            },
            domain="dexteleop.real_agent_motion.measured_pose_cut.v1",
        )

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return (
            *self.source_evidence_refs,
            "real-agent-measured-pose:" + self.semantic_content_hash,
        )


class RealAgentMotionTargetBuilder(Protocol):
    """Subset of the existing coupled retargeter used by this lower port.

    The public observation/Skill contract is tool-TCP based, while the
    coupled RM65 solver consumes its configured endpoint frame.  Keeping this
    conversion on the injected lower target owner prevents the mailbox from
    duplicating tool geometry or silently treating a TCP pose as a flange pose.
    """

    def current_tool_tcp_world_targets(self) -> dict[str, np.ndarray]: ...

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side: str,
        tool_tcp_world_target: np.ndarray,
    ) -> np.ndarray: ...

    def build_command_for_target_pair(
        self,
        target: BimanualPoseTarget,
    ) -> TeleopCommand: ...

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> dict[str, np.ndarray]: ...


class RealSkillMotionControlOwner(Protocol):
    """Minimal lower-owner surface consumed by the exclusive Skill mailbox."""

    def skill_motion_owner_state(self) -> tuple[str, str]: ...

    def hold_skill_motion(self, reason: str) -> bool: ...


class _LegacyRealTeleopOwnerAdapter:
    """Compatibility adapter kept out of the independent Skill runtime path."""

    def __init__(self, session: object) -> None:
        self._session = session

    def skill_motion_owner_state(self) -> tuple[str, str]:
        raw_state = getattr(
            getattr(self._session, "state_machine", None),
            "state",
            None,
        )
        state = str(getattr(raw_state, "value", raw_state) or "").strip().lower()
        if state in {"fault", "estop"}:
            return "fault", "fault"
        if state in {"armed", "running"}:
            return "running", "ready"
        return "hold", "hold"

    def hold_skill_motion(self, reason: str) -> bool:
        hold_hardware = getattr(self._session, "hold_hardware", None)
        if not callable(hold_hardware):
            raise AttributeError("legacy lower owner does not expose hold_hardware")
        return bool(
            hold_hardware(
                reason,
                require_home_recalibration=False,
            )
        )


class RealAgentMotionPort:
    """Thread-safe request mailbox with control-thread-only effects."""

    def __init__(
        self,
        *,
        config: RealAgentMotionPortConfig,
        target_builder: RealAgentMotionTargetBuilder,
        owner_identity: RealAgentMotionOwnerIdentity | None = None,
    ) -> None:
        self.config = config
        self._target_builder = target_builder
        if not callable(
            getattr(
                target_builder,
                "solver_world_target_from_tool_tcp_world_target",
                None,
            )
        ):
            raise TypeError(
                "target_builder must expose the exact tool-TCP-to-solver-world "
                "target conversion."
            )
        if owner_identity is not None and not isinstance(
            owner_identity,
            RealAgentMotionOwnerIdentity,
        ):
            raise TypeError("owner_identity must be typed or None.")
        self._owner_identity = owner_identity
        self._lock = threading.Lock()
        self._revision = 0
        self._phase = RealAgentMotionPhase.INACTIVE
        self._pending: RealAgentMotionActivation | None = None
        self._active: RealAgentMotionActivation | None = None
        self._last_operation: RealAgentMotionActivation | None = None
        self._domain_hold_id: str | None = None
        self._latest_measured_pose: RealAgentMotionMeasuredPoseCut | None = None
        self._latest_measured_qpos: RealAgentMotionMeasuredQposCut | None = None
        self._activated_at: float | None = None
        self._last_command_at: float | None = None
        self._dispatch_count = 0
        self._hold_reason: str | None = None
        self._hold_requested_at: float | None = None
        self._hold_applied_at: float | None = None
        self._hold_succeeded: bool | None = None
        self._fault: str | None = None
        # Preplanned paths use a committed logical clock. A slow observation,
        # expired reference, or other non-dispatch cycle must not advance the
        # path by wall time and then release a future target in one frame.
        self._trajectory_elapsed_sec = 0.0
        self._trajectory_clock_wall_time: float | None = None
        self._trajectory_pending_clock: tuple[str, float, float, bool] | None = None
        self._trajectory_clock_rebased_count = 0
        self._trajectory_clock_last_capped = False
        self._owner_attachment_evidence_ref: str | None = None
        self._owner_token_id: str | None = None
        self._owner_state_revision = 0
        self._owner_execution_epoch = 1
        self._latest_owner_cut: RealAgentMotionOwnerCut | None = None

    def confirm_exclusive_attachment(self) -> str:
        """Confirm that this port is the Session's sole command override."""

        if not self.config.exclusive_command_claim:
            raise RuntimeError("exclusive attachment requires exclusive_command_claim.")
        if self._owner_identity is None:
            raise RuntimeError("exclusive attachment requires owner_identity.")
        with self._lock:
            if self._owner_attachment_evidence_ref is not None:
                return self._owner_attachment_evidence_ref
            binding_hash = canonical_content_hash(
                {
                    "owner_identity_hash": self._owner_identity.semantic_content_hash,
                    "active_sides": self.config.active_sides,
                    "component_id_by_side": dict(self.config.component_id_by_side),
                    "execution_frame_id": self.config.execution_frame_id,
                    "execution_frame_spec_hash": (
                        self.config.execution_frame_spec_hash
                    ),
                    "exclusive_command_claim": True,
                },
                domain="dexteleop.real_agent_motion.exclusive_attachment.v1",
            )
            self._owner_token_id = "real-agent-motion-owner:" + binding_hash
            self._owner_attachment_evidence_ref = (
                "real-agent-motion-exclusive-attachment:" + binding_hash
            )
            self._advance_revision()
            return self._owner_attachment_evidence_ref

    def publish_control_owner_state(
        self,
        *,
        owner: RealSkillMotionControlOwner,
        captured_at: float,
    ) -> RealAgentMotionOwnerCut:
        """Publish the actual exclusive lower-owner state for this loop cut."""

        timestamp = _finite(captured_at, field_name="captured_at")
        lifecycle_state, execution_gate = owner.skill_motion_owner_state()
        lifecycle_state = _required(
            lifecycle_state,
            field_name="owner.lifecycle_state",
        )
        execution_gate = _required(
            execution_gate,
            field_name="owner.execution_gate",
        )
        if (lifecycle_state, execution_gate) not in {
            ("running", "ready"),
            ("hold", "hold"),
            ("fault", "fault"),
        }:
            raise ValueError("lower owner lifecycle_state/execution_gate mismatch.")
        with self._lock:
            if (
                self._owner_identity is None
                or self._owner_token_id is None
                or self._owner_attachment_evidence_ref is None
            ):
                raise RuntimeError("real Agent owner is not exclusively attached.")
            if (
                self._phase == RealAgentMotionPhase.FAULTED
                or lifecycle_state == "fault"
            ):
                lifecycle_state, execution_gate = "fault", "fault"
            elif (
                lifecycle_state == "running"
                and execution_gate == "ready"
                and self._phase
                not in {
                    RealAgentMotionPhase.HOLD_REQUESTED,
                    RealAgentMotionPhase.HOLD_APPLYING,
                    RealAgentMotionPhase.HELD,
                }
            ):
                lifecycle_state, execution_gate = "running", "ready"
            else:
                lifecycle_state, execution_gate = "hold", "hold"
            prior = self._latest_owner_cut
            if prior is not None and timestamp < prior.captured_at:
                raise ValueError("owner captured_at timestamp regressed.")
            # This call is one actual owner-loop cut. Publish a fresh revision
            # even when lifecycle labels are unchanged; otherwise a stable
            # RUNNING state eventually becomes stale to the observer.
            self._owner_state_revision += 1
            cut = RealAgentMotionOwnerCut(
                identity=self._owner_identity,
                owner_token_id=self._owner_token_id,
                owner_kind="exclusive_real_skill_motion_port",
                owner_generation=1,
                authority_epoch=1,
                execution_epoch=self._owner_execution_epoch,
                lifecycle_state=lifecycle_state,
                execution_gate=execution_gate,
                owner_state_revision=self._owner_state_revision,
                captured_at=timestamp,
                attachment_evidence_ref=self._owner_attachment_evidence_ref,
            )
            self._latest_owner_cut = cut
            self._advance_revision()
            return cut

    def publish_owner_state(
        self,
        *,
        session: object,
        captured_at: float,
    ) -> RealAgentMotionOwnerCut:
        """Compatibility wrapper for the existing RealTeleop attachment."""

        return self.publish_control_owner_state(
            owner=_LegacyRealTeleopOwnerAdapter(session),
            captured_at=captured_at,
        )

    def latest_owner_state(self) -> RealAgentMotionOwnerCut | None:
        with self._lock:
            return self._latest_owner_cut

    def _advance_revision(self) -> None:
        self._revision += 1

    def _operation(self) -> RealAgentMotionActivation | None:
        if self._active is not None:
            return self._active
        if self._pending is not None:
            return self._pending
        return self._last_operation

    def _snapshot_locked(self) -> RealAgentMotionPortSnapshot:
        operation = self._operation()
        return RealAgentMotionPortSnapshot(
            revision=self._revision,
            phase=self._phase,
            operation_id=None if operation is None else operation.operation_id,
            request_id=None if operation is None else operation.request_id,
            activation_grant_id=(
                None if operation is None else operation.activation_grant_id
            ),
            activated_at=self._activated_at,
            last_command_at=self._last_command_at,
            dispatch_count=self._dispatch_count,
            hold_reason=self._hold_reason,
            hold_requested_at=self._hold_requested_at,
            hold_applied_at=self._hold_applied_at,
            hold_succeeded=self._hold_succeeded,
            fault=self._fault,
            trajectory_elapsed_sec=(
                self._trajectory_elapsed_sec
                if self._active is not None
                and self._active.joint_trajectory is not None
                else None
            ),
            trajectory_clock_rebased_count=self._trajectory_clock_rebased_count,
            trajectory_clock_last_capped=self._trajectory_clock_last_capped,
        )

    def snapshot(self) -> RealAgentMotionPortSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def is_bound_to_target_builder(self, target_builder: object) -> bool:
        """Return whether this port was prepared for the exact lower solver."""

        return self._target_builder is target_builder

    def submit_activation(
        self,
        activation: RealAgentMotionActivation,
    ) -> RealAgentMotionPortSnapshot:
        if not isinstance(activation, RealAgentMotionActivation):
            raise TypeError("activation must be RealAgentMotionActivation.")
        expected_component = self.config.component_id_by_side.get(activation.side)
        if expected_component != activation.component_id:
            raise ValueError("activation component/side does not match the lower binding.")
        if activation.execution_frame_id != self.config.execution_frame_id:
            raise ValueError("activation execution_frame_id does not match the lower binding.")
        if (
            activation.execution_frame_spec_hash
            != self.config.execution_frame_spec_hash
        ):
            raise ValueError(
                "activation execution_frame_spec_hash does not match the lower binding."
            )
        if activation.joint_trajectory is not None:
            inactive_side = "right" if activation.side == "left" else "left"
            inactive_qpos = activation.joint_trajectory.qpos_by_side[inactive_side]
            if not np.allclose(
                inactive_qpos,
                np.broadcast_to(inactive_qpos[0], inactive_qpos.shape),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "preplanned trajectory cannot move the unauthorized inactive arm."
                )
        with self._lock:
            if self._phase not in {
                RealAgentMotionPhase.INACTIVE,
                RealAgentMotionPhase.HELD,
            }:
                raise RuntimeError(
                    f"real Agent motion port is busy in phase {self._phase.value}."
                )
            self._pending = activation
            self._active = None
            self._last_operation = activation
            self._domain_hold_id = None
            self._phase = RealAgentMotionPhase.PENDING_ACTIVATION
            self._activated_at = None
            self._last_command_at = None
            self._dispatch_count = 0
            self._hold_reason = None
            self._hold_requested_at = None
            self._hold_applied_at = None
            self._hold_succeeded = None
            self._fault = None
            self._trajectory_elapsed_sec = 0.0
            self._trajectory_clock_wall_time = None
            self._trajectory_pending_clock = None
            self._trajectory_clock_rebased_count = 0
            self._trajectory_clock_last_capped = False
            self._advance_revision()
            return self._snapshot_locked()

    def request_hold(
        self,
        *,
        operation_id: str,
        reason: str,
        requested_at: float,
    ) -> RealAgentMotionPortSnapshot:
        normalized_operation_id = _required(operation_id, field_name="operation_id")
        normalized_reason = _required(reason, field_name="reason")
        normalized_requested_at = _finite(requested_at, field_name="requested_at")
        with self._lock:
            operation = self._operation()
            if operation is None or operation.operation_id != normalized_operation_id:
                raise RuntimeError("hold request does not match the current operation.")
            if self._phase not in {
                RealAgentMotionPhase.PENDING_ACTIVATION,
                RealAgentMotionPhase.ACTIVE,
                RealAgentMotionPhase.FAULTED,
            }:
                raise RuntimeError(
                    f"cannot request HOLD in phase {self._phase.value}."
                )
            self._phase = RealAgentMotionPhase.HOLD_REQUESTED
            self._hold_reason = normalized_reason
            self._hold_requested_at = normalized_requested_at
            self._advance_revision()
            return self._snapshot_locked()

    def request_domain_hold(
        self,
        *,
        reason: str,
        requested_at: float,
    ) -> RealAgentMotionPortSnapshot:
        """Queue an owner HOLD even when no Agent operation is active."""

        normalized_reason = _required(reason, field_name="reason")
        normalized_requested_at = _finite(requested_at, field_name="requested_at")
        with self._lock:
            if (
                self._phase == RealAgentMotionPhase.HELD
                and self._hold_succeeded is True
            ):
                return self._snapshot_locked()
            if self._phase not in {
                RealAgentMotionPhase.INACTIVE,
                RealAgentMotionPhase.FAULTED,
            }:
                raise RuntimeError(
                    f"cannot request domain HOLD in phase {self._phase.value}."
                )
            self._pending = None
            self._active = None
            self._last_operation = None
            self._domain_hold_id = f"real-agent-domain-hold:{self._revision + 1}"
            self._phase = RealAgentMotionPhase.HOLD_REQUESTED
            self._hold_reason = normalized_reason
            self._hold_requested_at = normalized_requested_at
            self._advance_revision()
            return self._snapshot_locked()

    def publish_measured_pose(
        self,
        *,
        pose_world_by_component: Mapping[str, np.ndarray],
        measured_at: float,
        available_at: float,
        source_evidence_refs: tuple[str, ...],
    ) -> RealAgentMotionMeasuredPoseCut:
        """Publish an already-measured FK cut; this performs no backend read."""

        with self._lock:
            next_revision = (
                0
                if self._latest_measured_pose is None
                else self._latest_measured_pose.revision + 1
            )
            cut = RealAgentMotionMeasuredPoseCut(
                revision=next_revision,
                measured_at=measured_at,
                available_at=available_at,
                pose_world_by_component=pose_world_by_component,
                source_evidence_refs=source_evidence_refs,
            )
            if set(cut.pose_world_by_component) != set(
                self.config.component_id_by_side.values()
            ):
                raise ValueError(
                    "Measured pose cut must exactly cover configured Agent components."
                )
            current = self._latest_measured_pose
            if current is not None and cut.measured_at < current.measured_at:
                raise ValueError("Measured pose timestamp regressed.")
            if current is not None and cut.available_at < current.available_at:
                raise ValueError("Measured pose availability regressed.")
            self._latest_measured_pose = cut
            self._advance_revision()
            return cut

    def publish_measured_qpos_pair(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        measured_at: float,
        available_at: float,
        source_evidence_refs: tuple[str, ...],
    ) -> RealAgentMotionMeasuredPoseCut:
        """Project an already-read joint cut through the existing FK owner."""

        if set(qpos_by_side) != {"left", "right"}:
            raise ValueError("qpos_by_side must exactly contain left and right.")
        with self._lock:
            next_revision = (
                0
                if self._latest_measured_qpos is None
                else self._latest_measured_qpos.revision + 1
            )
            qpos_cut = RealAgentMotionMeasuredQposCut(
                revision=next_revision,
                measured_at=measured_at,
                available_at=available_at,
                qpos_by_side=qpos_by_side,
                source_evidence_refs=source_evidence_refs,
            )
            current = self._latest_measured_qpos
            if current is not None and qpos_cut.measured_at < current.measured_at:
                raise ValueError("Measured qpos timestamp regressed.")
            if current is not None and qpos_cut.available_at < current.available_at:
                raise ValueError("Measured qpos availability regressed.")
            self._latest_measured_qpos = qpos_cut
        targets = self._target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=np.asarray(qpos_by_side["left"], dtype=float),
            right_qpos=np.asarray(qpos_by_side["right"], dtype=float),
        )
        if set(targets) != {"left", "right"}:
            raise RuntimeError("Tool-TCP FK must return exact left/right targets.")
        return self.publish_measured_pose(
            pose_world_by_component={
                component_id: targets[side]
                for side, component_id in self.config.component_id_by_side.items()
            },
            measured_at=measured_at,
            available_at=available_at,
            source_evidence_refs=source_evidence_refs,
        )

    def latest_measured_pose(self) -> RealAgentMotionMeasuredPoseCut | None:
        with self._lock:
            return self._latest_measured_pose

    def latest_measured_qpos(self) -> RealAgentMotionMeasuredQposCut | None:
        with self._lock:
            return self._latest_measured_qpos

    def apply_control_owner_requests(
        self,
        *,
        owner: RealSkillMotionControlOwner,
        now: float,
    ) -> RealAgentMotionPortSnapshot:
        """Apply queued activation/HOLD effects from the exclusive owner thread."""

        current_time = _finite(now, field_name="now")
        hold_reason: str | None = None
        hold_operation_id: str | None = None
        with self._lock:
            if self._phase == RealAgentMotionPhase.PENDING_ACTIVATION:
                assert self._pending is not None
                if current_time > self._pending.expires_at:
                    self._fault = "activation expired before the owner became motion-ready"
                    self._phase = RealAgentMotionPhase.FAULTED
                    self._pending = None
                    self._advance_revision()
                    return self._snapshot_locked()
                if owner.skill_motion_owner_state() != ("running", "ready"):
                    return self._snapshot_locked()
                pending = self._pending
                trajectory = pending.joint_trajectory
                start_problem = None
                if trajectory is not None:
                    latest_qpos = self._latest_measured_qpos
                    if latest_qpos is None:
                        start_problem = "preplanned trajectory activation qpos unavailable"
                    elif any(
                        float(
                            np.max(
                                np.abs(
                                    latest_qpos.qpos_by_side[side]
                                    - trajectory.qpos_by_side[side][0]
                                )
                            )
                        )
                        > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
                        for side in ("left", "right")
                    ):
                        start_problem = "preplanned trajectory start changed before activation"
                if start_problem is None:
                    self._active = pending
                    self._pending = None
                    self._phase = RealAgentMotionPhase.ACTIVE
                    self._activated_at = current_time
                    if trajectory is not None:
                        self._trajectory_elapsed_sec = 0.0
                        self._trajectory_clock_wall_time = current_time
                        self._trajectory_pending_clock = None
                        self._trajectory_clock_rebased_count = 0
                        self._trajectory_clock_last_capped = False
                    self._advance_revision()
                    return self._snapshot_locked()
                hold_operation_id = pending.operation_id
                hold_reason = start_problem
                self._hold_reason = hold_reason
                self._hold_requested_at = current_time
                self._phase = RealAgentMotionPhase.HOLD_REQUESTED
                self._advance_revision()
            if self._phase == RealAgentMotionPhase.HOLD_REQUESTED:
                operation = self._operation()
                hold_operation_id = (
                    operation.operation_id
                    if operation is not None
                    else self._domain_hold_id
                )
                hold_reason = self._hold_reason
                self._phase = RealAgentMotionPhase.HOLD_APPLYING
                self._advance_revision()
            else:
                return self._snapshot_locked()

        assert hold_operation_id is not None
        assert hold_reason is not None
        owner_reason = (
            f"real Agent motion operation {hold_operation_id}: {hold_reason}"
        )
        try:
            hold_succeeded = bool(owner.hold_skill_motion(owner_reason))
            hold_error = None
        except Exception as exc:  # noqa: BLE001 - preserve explicit lower fault evidence.
            hold_succeeded = False
            hold_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._hold_applied_at = current_time
            self._hold_succeeded = hold_succeeded
            self._active = None
            self._pending = None
            if hold_succeeded:
                self._phase = RealAgentMotionPhase.HELD
                self._owner_execution_epoch += 1
            else:
                self._phase = RealAgentMotionPhase.FAULTED
                self._fault = (
                    "lower owner HOLD failed"
                    if hold_error is None
                    else f"lower owner HOLD failed: {hold_error}"
                )
            self._advance_revision()
            return self._snapshot_locked()

    def apply_owner_requests(
        self,
        *,
        session: object,
        now: float,
    ) -> RealAgentMotionPortSnapshot:
        """Compatibility wrapper for the existing RealTeleop attachment."""

        return self.apply_control_owner_requests(
            owner=_LegacyRealTeleopOwnerAdapter(session),
            now=now,
        )

    def command_for_observation(
        self,
        observation: TeleopObservation,
        now: float,
    ) -> TeleopCommandOverride | TeleopCommand | None:
        """Build the active dynamic target or evaluate one preplanned qpos path."""

        current_time = _finite(now, field_name="now")
        with self._lock:
            if self._phase in {
                RealAgentMotionPhase.INACTIVE,
                RealAgentMotionPhase.HELD,
            }:
                return (
                    TeleopCommandOverride(command=None)
                    if self.config.exclusive_command_claim
                    else None
                )
            if self._phase != RealAgentMotionPhase.ACTIVE:
                return TeleopCommandOverride(command=None)
            assert self._active is not None
            activation = self._active
        try:
            if activation.joint_trajectory is not None:
                trajectory = activation.joint_trajectory
                singularity_replay_prevalidated = any(
                    str(reference).startswith(
                        (
                            "rm65-grasp-path:",
                            "rm65-production-urdf-fcl-path:",
                            "urdf-fcl-path:",
                        )
                    )
                    for reference in trajectory.planner_evidence_refs
                )
                with self._lock:
                    activated_at = self._activated_at
                if activated_at is None:
                    raise RuntimeError("active trajectory has no activation timestamp")
                with self._lock:
                    clock_wall_time = self._trajectory_clock_wall_time
                    if clock_wall_time is None:
                        clock_wall_time = activated_at
                        self._trajectory_clock_wall_time = clock_wall_time
                    committed_elapsed = float(self._trajectory_elapsed_sec)
                raw_clock_advance = max(0.0, current_time - clock_wall_time)
                max_clock_advance = float(
                    self.config.trajectory_clock_max_advance_sec
                )
                clock_advance = min(raw_clock_advance, max_clock_advance)
                elapsed = min(
                    float(trajectory.time_from_start_sec[-1]),
                    committed_elapsed + clock_advance,
                )
                clock_capped = bool(raw_clock_advance > clock_advance + 1e-12)
                with self._lock:
                    self._trajectory_pending_clock = (
                        activation.operation_id,
                        current_time,
                        elapsed,
                        clock_capped,
                    )
                    self._trajectory_clock_last_capped = clock_capped
                next_sample_index = int(
                    np.searchsorted(
                        trajectory.time_from_start_sec,
                        elapsed,
                        side="right",
                    )
                )
                if next_sample_index <= 0:
                    sample_index = 0
                    next_sample_index = 0
                    interpolation_alpha = 0.0
                elif next_sample_index >= trajectory.time_from_start_sec.size:
                    sample_index = trajectory.time_from_start_sec.size - 1
                    next_sample_index = sample_index
                    interpolation_alpha = 0.0
                else:
                    sample_index = next_sample_index - 1
                    lower_time = float(trajectory.time_from_start_sec[sample_index])
                    upper_time = float(
                        trajectory.time_from_start_sec[next_sample_index]
                    )
                    interpolation_alpha = min(
                        1.0,
                        max(0.0, (elapsed - lower_time) / (upper_time - lower_time)),
                    )
                command = TeleopCommand(timestamp=observation.timestamp)
                operation_metadata = {
                    "operation_id": activation.operation_id,
                    "request_id": activation.request_id,
                    "component_id": activation.component_id,
                    "side": activation.side,
                    "primitive_id": activation.primitive_id,
                    "primitive_hash": activation.primitive_hash,
                    "activation_grant_id": activation.activation_grant_id,
                    "activation_grant_hash": activation.activation_grant_hash,
                    "execution_frame_id": activation.execution_frame_id,
                    "execution_frame_spec_hash": activation.execution_frame_spec_hash,
                    "measured_state_revision": activation.measured_state_revision,
                    "measured_state_hash": activation.measured_state_hash,
                    "speed_scale": activation.speed_scale,
                    "joint_trajectory_hash": trajectory.semantic_content_hash,
                    "trajectory_sample_index": sample_index,
                    "trajectory_next_sample_index": next_sample_index,
                    "trajectory_interpolation_alpha": interpolation_alpha,
                    "trajectory_sample_count": int(
                        trajectory.time_from_start_sec.size
                    ),
                    "trajectory_duration_sec": float(
                        trajectory.time_from_start_sec[-1]
                    ),
                    "trajectory_time_from_start_sec": float(
                        trajectory.time_from_start_sec[sample_index]
                    ),
                    "target_tool_tcp_pose_world": activation.target_pose_world.copy(),
                    "target_pose_semantics": "actual_tool_tcp_world",
                    "source": (
                        "replay_qpos_trajectory"
                        if singularity_replay_prevalidated
                        else "joint_vector"
                    ),
                    "singularity_replay_slowdown_preapplied": singularity_replay_prevalidated,
                    "singularity_replay_slowdown_collision_rechecked": singularity_replay_prevalidated,
                    "singularity_policy": "real_teleop_soft_traversal_v1",
                    "trajectory_elapsed_sec": float(elapsed),
                    "trajectory_clock_wall_time": current_time,
                    "trajectory_clock_advance_sec": float(clock_advance),
                    "trajectory_clock_advance_capped": clock_capped,
                }
                command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY] = dict(
                    operation_metadata
                )
                for side in ("left", "right"):
                    command.add_command(
                        EffectorCommand(
                            effector_name=f"{side}_arm",
                            side=side,
                            command_type="joint_vector",
                            value=(
                                trajectory.qpos_by_side[side][sample_index]
                                + interpolation_alpha
                                * (
                                    trajectory.qpos_by_side[side][next_sample_index]
                                    - trajectory.qpos_by_side[side][sample_index]
                                )
                            ),
                            metadata={
                                **dict(operation_metadata),
                                REAL_AGENT_MOTION_COMMAND_METADATA_KEY: dict(
                                    operation_metadata
                                )
                            },
                        )
                    )
                with self._lock:
                    if (
                        self._phase == RealAgentMotionPhase.ACTIVE
                        and self._active is not None
                        and self._active.operation_id == activation.operation_id
                    ):
                        self._last_command_at = current_time
                        self._advance_revision()
                return command
            current_tool_tcp_targets = (
                self._target_builder.current_tool_tcp_world_targets()
            )
            if set(current_tool_tcp_targets) != {"left", "right"}:
                raise RuntimeError(
                    "coupled target builder must publish current left/right tool TCP targets."
                )
            current_tool_tcp_targets = {
                side: _immutable_pose(
                    current_tool_tcp_targets[side],
                    field_name=f"current_tool_tcp_world_targets.{side}",
                )
                for side in ("left", "right")
            }
            # ``BimanualPoseTarget`` is consumed by the coupled solver and its
            # pose values are solver-world endpoint poses.  The Agent Skill and
            # measured cuts remain tool-TCP poses; convert both the inactive
            # hold anchors and the active target at this single lower boundary.
            target_pose_by_side = {
                side: _immutable_pose(
                    self._target_builder.solver_world_target_from_tool_tcp_world_target(
                        side=side,
                        tool_tcp_world_target=current_tool_tcp_targets[side],
                    ),
                    field_name=(
                        "solver_world_target_from_tool_tcp_world_target."
                        + side
                    ),
                )
                for side in ("left", "right")
            }
            activation_tool_tcp_target = _immutable_pose(
                activation.target_pose_world,
                field_name="activation.target_tool_tcp_world",
            )
            target_pose_by_side[activation.side] = _immutable_pose(
                self._target_builder.solver_world_target_from_tool_tcp_world_target(
                    side=activation.side,
                    tool_tcp_world_target=activation_tool_tcp_target,
                ),
                field_name=(
                    "solver_world_target_from_tool_tcp_world_target."
                    + activation.side
                ),
            )
            source_by_side = {"left": None, "right": None}
            source_by_side[activation.side] = BimanualPoseTargetSource(
                command_type="tcp_pose",
                value=activation_tool_tcp_target,
                frame_name=self.config.execution_frame_id,
                metadata={
                    "operation_id": activation.operation_id,
                    "request_id": activation.request_id,
                    "component_id": activation.component_id,
                    "primitive_id": activation.primitive_id,
                    "primitive_hash": activation.primitive_hash,
                    "activation_grant_id": activation.activation_grant_id,
                    "activation_grant_hash": activation.activation_grant_hash,
                    "measured_state_revision": activation.measured_state_revision,
                    "measured_state_hash": activation.measured_state_hash,
                    "controlled_frame_id": "tool_tcp",
                    "target_pose_semantics": "tool_tcp_world",
                    "solver_target_pose_semantics": "solver_world_endpoint",
                    "speed_scale": activation.speed_scale,
                },
            )
            command = self._target_builder.build_command_for_target_pair(
                BimanualPoseTarget(
                    timestamp=observation.timestamp,
                    target_pose_by_side=target_pose_by_side,
                    active_sides=(activation.side,),
                    hard_hold_inactive_sides=True,
                    source_by_side=source_by_side,
                )
            )
            operation_metadata = {
                "operation_id": activation.operation_id,
                "request_id": activation.request_id,
                "component_id": activation.component_id,
                "side": activation.side,
                "primitive_id": activation.primitive_id,
                "primitive_hash": activation.primitive_hash,
                "activation_grant_id": activation.activation_grant_id,
                "activation_grant_hash": activation.activation_grant_hash,
                "execution_frame_id": activation.execution_frame_id,
                "execution_frame_spec_hash": activation.execution_frame_spec_hash,
                "measured_state_revision": activation.measured_state_revision,
                "measured_state_hash": activation.measured_state_hash,
                "speed_scale": activation.speed_scale,
                # Observational trace input only.  This is the already-admitted
                # actual tool-TCP target in the execution world frame; lower
                # command generation and safety continue to use ``activation``
                # directly rather than reading this metadata copy.
                "target_tool_tcp_pose_world": activation.target_pose_world.copy(),
                "target_pose_semantics": "actual_tool_tcp_world",
            }
            command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY] = dict(
                operation_metadata
            )
            selected_effectors = tuple(
                effector
                for effector in command.commands.values()
                if effector.side == activation.side
            )
            if len(selected_effectors) != 1:
                raise RuntimeError(
                    "coupled target command must contain exactly one selected-side "
                    "effector."
                )
            for effector in selected_effectors:
                effector.metadata = dict(effector.metadata or {})
                effector.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY] = dict(
                    operation_metadata
                )
        except Exception as exc:
            with self._lock:
                if (
                    self._active is not None
                    and self._active.operation_id == activation.operation_id
                ):
                    self._fault = (
                        "coupled target command construction failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._phase = RealAgentMotionPhase.FAULTED
                    self._active = None
                    self._advance_revision()
            raise
        with self._lock:
            if (
                self._phase == RealAgentMotionPhase.ACTIVE
                and self._active is not None
                and self._active.operation_id == activation.operation_id
            ):
                self._last_command_at = current_time
                self._advance_revision()
        return command

    def admit_command(
        self,
        command: TeleopCommand,
        observation: TeleopObservation,
        now: float,
    ) -> bool:
        """Deny a stale Agent command if HOLD/cancellation won the race."""

        del observation, now
        operation_metadata = command.metadata.get(
            REAL_AGENT_MOTION_COMMAND_METADATA_KEY
        )
        if not isinstance(operation_metadata, Mapping):
            return True
        operation_id = str(operation_metadata.get("operation_id", ""))
        with self._lock:
            return bool(
                self._phase == RealAgentMotionPhase.ACTIVE
                and self._active is not None
                and self._active.operation_id == operation_id
            )

    def observe_session_step(
        self,
        *,
        command: TeleopCommand | None,
        command_dispatched: bool,
    ) -> RealAgentMotionPortSnapshot:
        """Record lower dispatch evidence after the Session completes a frame."""

        if command is None:
            return self.snapshot()
        operation_metadata = command.metadata.get(
            REAL_AGENT_MOTION_COMMAND_METADATA_KEY
        )
        if not isinstance(operation_metadata, Mapping):
            return self.snapshot()
        operation_id = str(operation_metadata.get("operation_id", ""))
        with self._lock:
            pending_clock = self._trajectory_pending_clock
            if (
                pending_clock is not None
                and self._active is not None
                and self._active.operation_id == operation_id
                and pending_clock[0] == operation_id
            ):
                _, pending_wall_time, pending_elapsed, pending_capped = pending_clock
                self._trajectory_clock_wall_time = float(pending_wall_time)
                if command_dispatched:
                    self._trajectory_elapsed_sec = max(
                        self._trajectory_elapsed_sec,
                        float(pending_elapsed),
                    )
                else:
                    # A skipped/expired frame is a soft HOLD for the
                    # trajectory clock: preserve the last committed sample,
                    # but rebase wall time so the next healthy frame resumes
                    # from there instead of catching up.
                    self._trajectory_clock_rebased_count += 1
                self._trajectory_clock_last_capped = bool(pending_capped)
                self._trajectory_pending_clock = None
        with self._lock:
            if (
                command_dispatched
                and self._active is not None
                and self._active.operation_id == operation_id
            ):
                self._dispatch_count += 1
                self._advance_revision()
            return self._snapshot_locked()


__all__ = [
    "REAL_AGENT_MOTION_COMMAND_METADATA_KEY",
    "REAL_AGENT_TRAJECTORY_CLOCK_MAX_ADVANCE_SEC",
    "REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD",
    "RealAgentJointTrajectory",
    "RealAgentMotionActivation",
    "RealAgentMotionMeasuredPoseCut",
    "RealAgentMotionMeasuredQposCut",
    "RealAgentMotionOwnerCut",
    "RealAgentMotionOwnerIdentity",
    "RealAgentMotionPhase",
    "RealAgentMotionPort",
    "RealAgentMotionPortConfig",
    "RealAgentMotionPortSnapshot",
    "RealAgentMotionTargetBuilder",
    "RealSkillMotionControlOwner",
]
