"""Lower-owned, default-off command port for one real Agent motion operation.

The port contains no Gateway, robot backend, or transport dependency.  A
Gateway worker may only submit immutable target/hold requests.  The existing
RealTeleop control-loop thread applies those requests, builds commands through
the existing coupled target solver, and lets :class:`TeleopSession` retain the
only filter, guard, admission, dispatch, and HOLD paths.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from robot_motion.rm65.dual_arm_target_solver import (
    BimanualPoseTarget,
    BimanualPoseTargetSource,
)
from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.session import TeleopCommandOverride
from teleop_runtime.transforms import validate_pose_matrix


_COMMAND_METADATA_KEY = "real_agent_motion_operation"


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


@dataclass(frozen=True)
class RealAgentMotionActivation:
    """One operator-granted absolute tool-TCP target in the solver world."""

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
        object.__setattr__(
            self,
            "target_pose_world",
            _immutable_pose(self.target_pose_world, field_name="target_pose_world"),
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


class RealAgentMotionTargetBuilder(Protocol):
    """Subset of the existing coupled retargeter used by this lower port."""

    def current_tool_tcp_world_targets(self) -> dict[str, np.ndarray]: ...

    def build_command_for_target_pair(
        self,
        target: BimanualPoseTarget,
    ) -> TeleopCommand: ...


class RealAgentMotionOwnerSession(Protocol):
    state_machine: object

    def hold_hardware(
        self,
        reason: str = "manual hold",
        *,
        require_home_recalibration: bool = False,
    ) -> bool: ...


class RealAgentMotionPort:
    """Thread-safe request mailbox with control-thread-only effects."""

    def __init__(
        self,
        *,
        config: RealAgentMotionPortConfig,
        target_builder: RealAgentMotionTargetBuilder,
    ) -> None:
        self.config = config
        self._target_builder = target_builder
        self._lock = threading.Lock()
        self._revision = 0
        self._phase = RealAgentMotionPhase.INACTIVE
        self._pending: RealAgentMotionActivation | None = None
        self._active: RealAgentMotionActivation | None = None
        self._last_operation: RealAgentMotionActivation | None = None
        self._activated_at: float | None = None
        self._last_command_at: float | None = None
        self._dispatch_count = 0
        self._hold_reason: str | None = None
        self._hold_requested_at: float | None = None
        self._hold_applied_at: float | None = None
        self._hold_succeeded: bool | None = None
        self._fault: str | None = None

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
        )

    def snapshot(self) -> RealAgentMotionPortSnapshot:
        with self._lock:
            return self._snapshot_locked()

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
            self._phase = RealAgentMotionPhase.PENDING_ACTIVATION
            self._activated_at = None
            self._last_command_at = None
            self._dispatch_count = 0
            self._hold_reason = None
            self._hold_requested_at = None
            self._hold_applied_at = None
            self._hold_succeeded = None
            self._fault = None
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
            }:
                raise RuntimeError(
                    f"cannot request HOLD in phase {self._phase.value}."
                )
            self._phase = RealAgentMotionPhase.HOLD_REQUESTED
            self._hold_reason = normalized_reason
            self._hold_requested_at = normalized_requested_at
            self._advance_revision()
            return self._snapshot_locked()

    def apply_owner_requests(
        self,
        *,
        session: RealAgentMotionOwnerSession,
        now: float,
    ) -> RealAgentMotionPortSnapshot:
        """Apply queued activation/HOLD effects from the control-loop thread."""

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
                state = getattr(getattr(session, "state_machine", None), "state", None)
                if state not in {TeleopRunState.ARMED, TeleopRunState.RUNNING}:
                    return self._snapshot_locked()
                self._active = self._pending
                self._pending = None
                self._phase = RealAgentMotionPhase.ACTIVE
                self._activated_at = current_time
                self._advance_revision()
                return self._snapshot_locked()
            if self._phase == RealAgentMotionPhase.HOLD_REQUESTED:
                operation = self._operation()
                assert operation is not None
                hold_operation_id = operation.operation_id
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
            hold_succeeded = bool(
                session.hold_hardware(
                    owner_reason,
                    require_home_recalibration=False,
                )
            )
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
            else:
                self._phase = RealAgentMotionPhase.FAULTED
                self._fault = (
                    "lower owner HOLD failed"
                    if hold_error is None
                    else f"lower owner HOLD failed: {hold_error}"
                )
            self._advance_revision()
            return self._snapshot_locked()

    def command_for_observation(
        self,
        observation: TeleopObservation,
        now: float,
    ) -> TeleopCommandOverride | TeleopCommand | None:
        """Build the active target through the existing coupled safe solver."""

        current_time = _finite(now, field_name="now")
        with self._lock:
            if self._phase in {
                RealAgentMotionPhase.INACTIVE,
                RealAgentMotionPhase.HELD,
            }:
                return None
            if self._phase != RealAgentMotionPhase.ACTIVE:
                return TeleopCommandOverride(command=None)
            assert self._active is not None
            activation = self._active
        try:
            current_targets = self._target_builder.current_tool_tcp_world_targets()
            if set(current_targets) != {"left", "right"}:
                raise RuntimeError(
                    "coupled target builder must publish current left/right tool TCP targets."
                )
            target_pose_by_side = {
                side: _immutable_pose(
                    current_targets[side],
                    field_name=f"current_tool_tcp_world_targets.{side}",
                )
                for side in ("left", "right")
            }
            target_pose_by_side[activation.side] = activation.target_pose_world
            source_by_side = {"left": None, "right": None}
            source_by_side[activation.side] = BimanualPoseTargetSource(
                command_type="tcp_pose",
                value=activation.target_pose_world,
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
                },
            )
            command = self._target_builder.build_command_for_target_pair(
                BimanualPoseTarget(
                    timestamp=observation.timestamp,
                    target_pose_by_side=target_pose_by_side,
                    active_sides=self.config.active_sides,
                    source_by_side=source_by_side,
                )
            )
            command.metadata[_COMMAND_METADATA_KEY] = {
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
            }
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
        operation_metadata = command.metadata.get(_COMMAND_METADATA_KEY)
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
        operation_metadata = command.metadata.get(_COMMAND_METADATA_KEY)
        if not isinstance(operation_metadata, Mapping):
            return self.snapshot()
        operation_id = str(operation_metadata.get("operation_id", ""))
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
    "RealAgentMotionActivation",
    "RealAgentMotionPhase",
    "RealAgentMotionPort",
    "RealAgentMotionPortConfig",
    "RealAgentMotionPortSnapshot",
    "RealAgentMotionTargetBuilder",
]
