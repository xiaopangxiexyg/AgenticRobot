"""Fail-closed lower adapters for registered shadow trajectory preparation.

This module may read already-started gripper backends, but it never constructs
devices, sends commands, or grants real-motion authority.  Its output is one
causally bounded measured gripper cut that can be joined with the existing
dual-arm observation cut before MuJoCo is initialized.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowReplayAdmissionReceipt,
    ShadowSimulationReceipt,
    evaluate_shadow_replay_admission,
)

from .control_runtime import (
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)
from .shadow_trajectory import capture_authoritative_shadow_state

_SIDES = ("left", "right")
_MEASURED_GRIPPER_CUT_HASH_DOMAIN = (
    "agenticrobot.shadow_execution.measured_gripper_cut.v1"
)


class MeasuredGripperBackend(Protocol):
    """Already-started lower backend with one synchronous fresh-state read."""

    def read_gripper_state_now(self) -> Mapping[str, object] | None: ...


class RegisteredTrajectoryPointDispatcher(Protocol):
    """Trusted lower exact-qpos transport; not an Agent-facing port."""

    def __call__(
        self,
        point: RegisteredJointTrajectoryPoint,
        trajectory_hash: str,
        /,
    ) -> RegisteredTrajectoryPointDispatchReceipt: ...


class TerminalToolTcpVerifier(Protocol):
    """Verify a fresh measured-qpos cut against the requested tool TCP."""

    def __call__(
        self,
        snapshot: AuthoritativeRobotStateSnapshot,
        /,
    ) -> TerminalToolTcpVerificationReceipt: ...


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _measured_scalar(payload: Mapping[str, object], *, side: str) -> float:
    for field_name in (
        "gripper_measured_scalar",
        "joint_pos",
        "gripper_state_scalar",
    ):
        raw = payload.get(field_name)
        if raw is None:
            continue
        value = _finite(raw, name=f"{side}.{field_name}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{side}.{field_name} must be in [0, 1].")
        return value
    raise ValueError(f"{side} gripper read has no measured scalar.")


@dataclass(frozen=True)
class MeasuredGripperStateCut:
    """One fresh, non-assumed dual-gripper hardware observation cut."""

    close_by_side: Mapping[str, float]
    measured_at_by_side: Mapping[str, float]
    available_at: float
    evidence_refs: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if set(self.close_by_side) != set(_SIDES):
            raise ValueError("close_by_side must contain exactly left and right.")
        if set(self.measured_at_by_side) != set(_SIDES):
            raise ValueError(
                "measured_at_by_side must contain exactly left and right."
            )
        close = MappingProxyType(
            {
                side: _finite(
                    self.close_by_side[side], name=f"close_by_side[{side!r}]"
                )
                for side in _SIDES
            }
        )
        if any(not 0.0 <= value <= 1.0 for value in close.values()):
            raise ValueError("each measured gripper close value must be in [0, 1].")
        measured = MappingProxyType(
            {
                side: _finite(
                    self.measured_at_by_side[side],
                    name=f"measured_at_by_side[{side!r}]",
                )
                for side in _SIDES
            }
        )
        available = _finite(self.available_at, name="available_at")
        if any(value > available for value in measured.values()):
            raise ValueError("available_at cannot precede a gripper measurement.")
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("measured gripper cut requires non-empty evidence refs.")
        object.__setattr__(self, "close_by_side", close)
        object.__setattr__(self, "measured_at_by_side", measured)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "close_by_side": dict(close),
                    "measured_at_by_side": dict(measured),
                    "available_at": available,
                    "evidence_refs": refs,
                },
                domain=_MEASURED_GRIPPER_CUT_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class RegisteredTrajectoryExecutionAuthority:
    """Short-lived operator authority bound to exact simulation evidence."""

    authority_id: str
    operator_confirmation_ref: str
    trajectory_hash: str
    simulation_receipt_hash: str
    run_id: str
    owner_epoch: int
    issued_at: float
    expires_at: float
    real_motion_enabled: bool = True
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "authority_id",
            "operator_confirmation_ref",
            "trajectory_hash",
            "simulation_receipt_hash",
            "run_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        issued = _finite(self.issued_at, name="issued_at")
        expires = _finite(self.expires_at, name="expires_at")
        if expires <= issued:
            raise ValueError("expires_at must be after issued_at.")
        if (
            isinstance(self.owner_epoch, bool)
            or not isinstance(self.owner_epoch, int)
            or self.owner_epoch < 1
        ):
            raise ValueError("owner_epoch must be a positive integer.")
        if self.real_motion_enabled is not True:
            raise ValueError("execution authority must explicitly enable real motion.")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "authority_id": self.authority_id,
                    "operator_confirmation_ref": self.operator_confirmation_ref,
                    "trajectory_hash": self.trajectory_hash,
                    "simulation_receipt_hash": self.simulation_receipt_hash,
                    "run_id": self.run_id,
                    "owner_epoch": self.owner_epoch,
                    "issued_at": issued,
                    "expires_at": expires,
                    "real_motion_enabled": True,
                },
                domain="agenticrobot.shadow_execution.real_authority.v1",
            ),
        )


class RegisteredTrajectoryAuthorityConsumptionStatus(str, Enum):
    CONSUMED = "consumed"
    ALREADY_CONSUMED = "already_consumed"
    AUTHORITY_MISMATCH = "authority_mismatch"


@dataclass(frozen=True)
class RegisteredTrajectoryAuthorityConsumptionReceipt:
    """Atomic lower-owner result for one exact real-motion authority."""

    status: RegisteredTrajectoryAuthorityConsumptionStatus
    authority_hash: str
    authority_id: str
    trajectory_hash: str
    simulation_receipt_hash: str
    run_id: str
    owner_epoch: int
    consumed_at: float
    evidence_refs: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.status, RegisteredTrajectoryAuthorityConsumptionStatus
        ):
            raise TypeError(
                "status must be RegisteredTrajectoryAuthorityConsumptionStatus."
            )
        normalized: dict[str, str] = {}
        for name in (
            "authority_hash",
            "authority_id",
            "trajectory_hash",
            "simulation_receipt_hash",
            "run_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            normalized[name] = value
            object.__setattr__(self, name, value)
        if (
            isinstance(self.owner_epoch, bool)
            or not isinstance(self.owner_epoch, int)
            or self.owner_epoch < 1
        ):
            raise ValueError("owner_epoch must be a positive integer.")
        consumed_at = _finite(self.consumed_at, name="consumed_at")
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("authority consumption requires non-empty evidence refs.")
        object.__setattr__(self, "consumed_at", consumed_at)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "status": self.status.value,
                    **normalized,
                    "owner_epoch": self.owner_epoch,
                    "consumed_at": consumed_at,
                    "evidence_refs": refs,
                },
                domain="agenticrobot.shadow_execution.authority_consumption.v1",
            ),
        )

    @property
    def consumed(self) -> bool:
        return self.status is RegisteredTrajectoryAuthorityConsumptionStatus.CONSUMED


class SingleUseRegisteredTrajectoryAuthorityConsumer:
    """In-process CAS owner for one exact authority and one replay run."""

    def __init__(
        self,
        *,
        authority: RegisteredTrajectoryExecutionAuthority,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(authority, RegisteredTrajectoryExecutionAuthority):
            raise TypeError("authority must be RegisteredTrajectoryExecutionAuthority.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.authority = authority
        self.clock = clock
        self._lock = threading.Lock()
        self._consumed = False

    def consume(
        self,
        authority: RegisteredTrajectoryExecutionAuthority,
    ) -> RegisteredTrajectoryAuthorityConsumptionReceipt:
        now = _finite(self.clock(), name="clock()")
        with self._lock:
            matches = bool(
                isinstance(authority, RegisteredTrajectoryExecutionAuthority)
                and authority.semantic_content_hash
                == self.authority.semantic_content_hash
            )
            if not matches:
                status = (
                    RegisteredTrajectoryAuthorityConsumptionStatus.AUTHORITY_MISMATCH
                )
            elif self._consumed:
                status = (
                    RegisteredTrajectoryAuthorityConsumptionStatus.ALREADY_CONSUMED
                )
            else:
                self._consumed = True
                status = RegisteredTrajectoryAuthorityConsumptionStatus.CONSUMED
        observed = authority if matches else self.authority
        evidence_hash = canonical_content_hash(
            {
                "authority_hash": observed.semantic_content_hash,
                "status": status.value,
                "consumed_at": now,
            },
            domain="agenticrobot.shadow_execution.authority_consumption_evidence.v1",
        )
        return RegisteredTrajectoryAuthorityConsumptionReceipt(
            status=status,
            authority_hash=observed.semantic_content_hash,
            authority_id=observed.authority_id,
            trajectory_hash=observed.trajectory_hash,
            simulation_receipt_hash=observed.simulation_receipt_hash,
            run_id=observed.run_id,
            owner_epoch=observed.owner_epoch,
            consumed_at=now,
            evidence_refs=("authority-consumption:" + evidence_hash,),
        )


@dataclass(frozen=True)
class RegisteredTrajectoryPointDispatchReceipt:
    """Lower acceptance result bound to one exact registered point."""

    point_hash: str
    succeeded: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        point_hash = str(self.point_hash).strip()
        if not point_hash:
            raise ValueError("point_hash must be non-empty.")
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be boolean.")
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if self.succeeded and (not refs or any(not value for value in refs)):
            raise ValueError("successful point dispatch requires lower evidence.")
        object.__setattr__(self, "point_hash", point_hash)
        object.__setattr__(self, "evidence_refs", refs)


class TerminalToolTcpVerificationStatus(str, Enum):
    VERIFIED = "verified"
    TARGET_MISMATCH = "target_mismatch"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"


@dataclass(frozen=True)
class TerminalToolTcpVerificationReceipt:
    """Terminal TCP error computed from one fresh real hardware-qpos cut.

    This is an endpoint measurement receipt, not another model/asset identity
    attestation.  The caller supplies the original requested tool-TCP target;
    the verifier must not substitute FK of the planned terminal qpos as the
    target because that would merely restate joint tracking.
    """

    status: TerminalToolTcpVerificationStatus
    frame_id: str
    observed_snapshot_hash: str
    position_error_m_by_side: Mapping[str, float]
    orientation_error_deg_by_side: Mapping[str, float]
    position_tolerance_m: float
    orientation_tolerance_deg: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, TerminalToolTcpVerificationStatus):
            raise TypeError("status must be TerminalToolTcpVerificationStatus.")
        frame_id = str(self.frame_id).strip()
        snapshot_hash = str(self.observed_snapshot_hash).strip()
        if not frame_id or not snapshot_hash:
            raise ValueError("frame_id and observed_snapshot_hash must be non-empty.")
        position = {
            str(side).strip(): _finite(value, name=f"position_error_m_by_side.{side}")
            for side, value in dict(self.position_error_m_by_side).items()
        }
        orientation = {
            str(side).strip(): _finite(
                value,
                name=f"orientation_error_deg_by_side.{side}",
            )
            for side, value in dict(self.orientation_error_deg_by_side).items()
        }
        if set(position) != set(orientation):
            raise ValueError(
                "terminal TCP errors must contain the same side set."
            )
        if (
            self.status
            is not TerminalToolTcpVerificationStatus.VERIFICATION_UNAVAILABLE
            and not position
        ):
            raise ValueError("available terminal TCP verification requires a side.")
        if any(not side for side in position) or any(
            value < 0.0 for value in (*position.values(), *orientation.values())
        ):
            raise ValueError("terminal TCP sides and errors must be valid.")
        position_tolerance = _finite(
            self.position_tolerance_m,
            name="position_tolerance_m",
        )
        orientation_tolerance = _finite(
            self.orientation_tolerance_deg,
            name="orientation_tolerance_deg",
        )
        if position_tolerance < 0.0 or orientation_tolerance < 0.0:
            raise ValueError("terminal TCP tolerances must be non-negative.")
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("terminal TCP verification requires measured evidence.")
        within = all(
            position[side] <= position_tolerance
            and orientation[side] <= orientation_tolerance
            for side in position
        )
        if self.status is TerminalToolTcpVerificationStatus.VERIFIED and not within:
            raise ValueError("verified terminal TCP receipt exceeds its tolerance.")
        if self.status is TerminalToolTcpVerificationStatus.TARGET_MISMATCH and within:
            raise ValueError("target-mismatch receipt must exceed a tolerance.")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "observed_snapshot_hash", snapshot_hash)
        object.__setattr__(self, "position_error_m_by_side", MappingProxyType(position))
        object.__setattr__(
            self,
            "orientation_error_deg_by_side",
            MappingProxyType(orientation),
        )
        object.__setattr__(self, "position_tolerance_m", position_tolerance)
        object.__setattr__(self, "orientation_tolerance_deg", orientation_tolerance)
        object.__setattr__(self, "evidence_refs", refs)

    @property
    def verified(self) -> bool:
        return self.status is TerminalToolTcpVerificationStatus.VERIFIED


class RegisteredTrajectoryRealReplayStatus(str, Enum):
    COMPLETED = "completed"
    EXECUTION_DISABLED = "execution_disabled"
    AUTHORITY_REJECTED = "authority_rejected"
    PREFLIGHT_REJECTED = "preflight_rejected"
    PRE_DISPATCH_STATE_REJECTED = "pre_dispatch_state_rejected"
    TIMELINE_DEADLINE_MISSED = "timeline_deadline_missed"
    DISPATCH_FAILED = "dispatch_failed"
    FINAL_TRACKING_TIMEOUT = "final_tracking_timeout"
    FINAL_TCP_TARGET_MISMATCH = "final_tcp_target_mismatch"
    FINAL_TCP_VERIFICATION_UNAVAILABLE = "final_tcp_verification_unavailable"
    GRIPPER_HOLD_DRIFT = "gripper_hold_drift"
    LOWER_PORT_ERROR = "lower_port_error"
    TERMINAL_HOLD_FAILED = "terminal_hold_failed"


@dataclass(frozen=True)
class RegisteredTrajectoryRealReplayReceipt:
    """Terminal result from trusted replay ports; never a reusable permit."""

    status: RegisteredTrajectoryRealReplayStatus
    admission: ShadowReplayAdmissionReceipt
    executed_point_count: int
    terminal_hold_confirmed: bool
    real_motion_executed: bool
    evidence_refs: tuple[str, ...]
    real_motion_may_have_executed: bool = False
    terminal_tool_tcp_verification: TerminalToolTcpVerificationReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RegisteredTrajectoryRealReplayStatus):
            raise TypeError("status must be RegisteredTrajectoryRealReplayStatus.")
        if not isinstance(self.admission, ShadowReplayAdmissionReceipt):
            raise TypeError("admission must be ShadowReplayAdmissionReceipt.")
        if (
            isinstance(self.executed_point_count, bool)
            or not isinstance(self.executed_point_count, int)
            or self.executed_point_count < 0
        ):
            raise ValueError("executed_point_count must be a non-negative integer.")
        if not isinstance(self.terminal_hold_confirmed, bool):
            raise TypeError("terminal_hold_confirmed must be boolean.")
        if not isinstance(self.real_motion_executed, bool):
            raise TypeError("real_motion_executed must be boolean.")
        if not isinstance(self.real_motion_may_have_executed, bool):
            raise TypeError("real_motion_may_have_executed must be boolean.")
        if self.terminal_tool_tcp_verification is not None and not isinstance(
            self.terminal_tool_tcp_verification,
            TerminalToolTcpVerificationReceipt,
        ):
            raise TypeError(
                "terminal_tool_tcp_verification must be a typed receipt."
            )
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("real replay receipt requires lower evidence.")
        if self.status is RegisteredTrajectoryRealReplayStatus.COMPLETED and (
            not self.admission.admissible
            or self.executed_point_count == 0
            or not self.terminal_hold_confirmed
            or not self.real_motion_executed
            or self.real_motion_may_have_executed
            or (
                self.terminal_tool_tcp_verification is not None
                and not self.terminal_tool_tcp_verification.verified
            )
        ):
            raise ValueError("completed real replay requires admission, motion, and HOLD.")
        object.__setattr__(self, "evidence_refs", refs)


def read_measured_gripper_state_cut(
    *,
    backends_by_side: Mapping[str, MeasuredGripperBackend],
    clock: Callable[[], float] = time.time,
    maximum_inter_side_skew_sec: float = 0.05,
) -> MeasuredGripperStateCut:
    """Synchronously read both grippers and reject fallback/cached assumptions."""

    backends = dict(backends_by_side)
    if set(backends) != set(_SIDES):
        raise ValueError("backends_by_side must contain exactly left and right.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    max_skew = _finite(
        maximum_inter_side_skew_sec,
        name="maximum_inter_side_skew_sec",
    )
    if max_skew < 0.0:
        raise ValueError("maximum_inter_side_skew_sec must be non-negative.")

    close: dict[str, float] = {}
    measured: dict[str, float] = {}
    evidence: list[str] = []
    for side in _SIDES:
        reader = getattr(backends[side], "read_gripper_state_now", None)
        if not callable(reader):
            raise TypeError(
                f"{side} gripper backend must expose read_gripper_state_now()."
            )
        payload = reader()
        if payload is None:
            raise RuntimeError(f"{side} gripper fresh-state read is unavailable.")
        if not isinstance(payload, Mapping):
            raise TypeError(f"{side} gripper fresh-state read must be a mapping.")
        state = dict(payload)
        source = str(state.get("gripper_state_source", "runtime_backend")).strip()
        if bool(state.get("gripper_state_assumed", False)) or source in {
            "configured_disconnected_assumption",
            "commanded_fallback",
            "cached_assumption",
        }:
            raise RuntimeError(f"{side} gripper state is assumed, not measured.")
        close[side] = _measured_scalar(state, side=side)
        if "timestamp" not in state:
            raise RuntimeError(f"{side} gripper read has no measurement timestamp.")
        measured[side] = _finite(state["timestamp"], name=f"{side}.timestamp")
        evidence_hash = canonical_content_hash(
            {
                "side": side,
                "close": close[side],
                "measured_at": measured[side],
                "state_source": source or "runtime_backend",
            },
            domain=_MEASURED_GRIPPER_CUT_HASH_DOMAIN,
        )
        evidence.append(f"measured-gripper:{side}:{evidence_hash}")

    available = _finite(clock(), name="clock()")
    if any(value > available for value in measured.values()):
        raise RuntimeError("gripper measurement timestamp is in the future.")
    if max(measured.values()) - min(measured.values()) > max_skew:
        raise RuntimeError("dual-gripper measurements exceed the allowed time skew.")
    return MeasuredGripperStateCut(
        close_by_side=close,
        measured_at_by_side=measured,
        available_at=available,
        evidence_refs=tuple(evidence),
    )


def capture_authoritative_shadow_state_from_backends(
    *,
    arm_observation_cut: RealSkillMotionObservationCut,
    gripper_backends_by_side: Mapping[str, MeasuredGripperBackend],
    identity: ShadowExecutionIdentity,
    clock: Callable[[], float] = time.time,
    maximum_inter_side_skew_sec: float = 0.05,
    maximum_arm_gripper_skew_sec: float = 0.1,
) -> AuthoritativeRobotStateSnapshot:
    """Read measured grippers and join them with an already-owned arm cut."""

    grippers = read_measured_gripper_state_cut(
        backends_by_side=gripper_backends_by_side,
        clock=clock,
        maximum_inter_side_skew_sec=maximum_inter_side_skew_sec,
    )
    cross_source_skew = _finite(
        maximum_arm_gripper_skew_sec,
        name="maximum_arm_gripper_skew_sec",
    )
    if cross_source_skew < 0.0:
        raise ValueError("maximum_arm_gripper_skew_sec must be non-negative.")
    if max(
        abs(grippers.measured_at_by_side[side] - arm_observation_cut.measured_at)
        for side in _SIDES
    ) > cross_source_skew:
        raise RuntimeError("arm and gripper measurements exceed the allowed time skew.")
    return capture_authoritative_shadow_state(
        arm_observation_cut=arm_observation_cut,
        gripper_close_by_side=grippers.close_by_side,
        gripper_measured_at_by_side=grippers.measured_at_by_side,
        gripper_available_at=grippers.available_at,
        gripper_evidence_refs=(
            *grippers.evidence_refs,
            f"measured-gripper-cut:{grippers.semantic_content_hash}",
        ),
        identity=identity,
    )


def execute_registered_shadow_trajectory_with_ports(
    *,
    trajectory: RegisteredShadowJointTrajectory,
    simulation_receipt: ShadowSimulationReceipt,
    current_scene_hash: str,
    current_scene_hash_provider: Callable[[], str] | None = None,
    current_snapshot_provider: Callable[[], AuthoritativeRobotStateSnapshot],
    point_dispatcher: RegisteredTrajectoryPointDispatcher,
    hold_controller: Callable[[str], RealSkillMotionHoldReceipt],
    terminal_tool_tcp_verifier: TerminalToolTcpVerifier | None = None,
    terminal_tool_tcp_verification_required: bool = False,
    authority: RegisteredTrajectoryExecutionAuthority | None = None,
    consume_execution_authority: (
        Callable[
            [RegisteredTrajectoryExecutionAuthority],
            RegisteredTrajectoryAuthorityConsumptionReceipt,
        ]
        | None
    ) = None,
    begin_replay_session: (
        Callable[
            [RegisteredTrajectoryAuthorityConsumptionReceipt], tuple[str, ...]
        ]
        | None
    ) = None,
    end_replay_session: Callable[[str], tuple[str, ...]] | None = None,
    execution_enabled: bool = False,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    maximum_snapshot_age_sec: float = 0.2,
    maximum_arm_start_drift_rad: float = 0.01,
    maximum_gripper_start_drift: float = 0.02,
    final_tracking_tolerance_rad: float = 0.01,
    final_tracking_timeout_sec: float = 3.0,
    final_tracking_velocity_tolerance_rad_s: float = 0.02,
    final_tracking_dwell_sec: float = 0.08,
    feedback_poll_sec: float = 0.02,
    maximum_point_lateness_sec: float = 0.05,
) -> RegisteredTrajectoryRealReplayReceipt:
    """Exercise the post-simulation real replay sequence through injected ports.

    The preflight receipt remains non-authoritative.  Actual dispatch additionally
    requires a short-lived exact authority and an injected single-use authority
    consumer owned by the lower application.  This module does not issue either.
    """

    if not isinstance(trajectory, RegisteredShadowJointTrajectory):
        raise TypeError("trajectory must be RegisteredShadowJointTrajectory.")
    if not isinstance(simulation_receipt, ShadowSimulationReceipt):
        raise TypeError("simulation_receipt must be ShadowSimulationReceipt.")
    for name, callback in (
        ("current_snapshot_provider", current_snapshot_provider),
        ("point_dispatcher", point_dispatcher),
        ("hold_controller", hold_controller),
        ("wall_clock", wall_clock),
        ("monotonic_clock", monotonic_clock),
        ("sleep", sleep),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable.")
    if terminal_tool_tcp_verifier is not None and not callable(
        terminal_tool_tcp_verifier
    ):
        raise TypeError("terminal_tool_tcp_verifier must be callable when present.")
    if not isinstance(terminal_tool_tcp_verification_required, bool):
        raise TypeError("terminal_tool_tcp_verification_required must be boolean.")
    if not isinstance(execution_enabled, bool):
        raise TypeError("execution_enabled must be boolean.")
    scene_hash = str(current_scene_hash).strip()
    if not scene_hash:
        raise ValueError("current_scene_hash must be non-empty.")
    final_tolerance = _finite(
        final_tracking_tolerance_rad,
        name="final_tracking_tolerance_rad",
    )
    final_timeout = _finite(
        final_tracking_timeout_sec,
        name="final_tracking_timeout_sec",
    )
    final_velocity_tolerance = _finite(
        final_tracking_velocity_tolerance_rad_s,
        name="final_tracking_velocity_tolerance_rad_s",
    )
    final_dwell = _finite(
        final_tracking_dwell_sec,
        name="final_tracking_dwell_sec",
    )
    poll = _finite(feedback_poll_sec, name="feedback_poll_sec")
    max_lateness = _finite(
        maximum_point_lateness_sec,
        name="maximum_point_lateness_sec",
    )
    if (
        final_tolerance <= 0.0
        or final_timeout <= 0.0
        or final_velocity_tolerance <= 0.0
        or final_dwell <= 0.0
        or poll <= 0.0
        or max_lateness < 0.0
    ):
        raise ValueError(
            "tracking tolerance, timeout, velocity, dwell, and polling must be positive; "
            "point lateness must be non-negative."
        )

    try:
        current = current_snapshot_provider()
        if not isinstance(current, AuthoritativeRobotStateSnapshot):
            raise TypeError(
                "current_snapshot_provider must return "
                "AuthoritativeRobotStateSnapshot."
            )
        # Evaluate freshness after the joined RM65 + gripper cut is complete.
        # Hardware snapshot collection may block on multiple SDK reads, so a
        # timestamp sampled before the provider can precede ``available_at``
        # and incorrectly classify a newly captured cut as stale.
        now = _finite(wall_clock(), name="wall_clock()")
    except Exception:
        # No admission receipt exists yet, but the lower motion owner must
        # still receive HOLD before the causal read failure escapes.
        try:
            hold_controller("registered shadow replay initial state read failed")
        except Exception:  # noqa: BLE001, S110 - preserve the original failure.
            pass
        raise
    try:
        admission = evaluate_shadow_replay_admission(
            current_snapshot=current,
            trajectory=trajectory,
            simulation_receipt=simulation_receipt,
            current_scene_hash=scene_hash,
            evaluated_at=now,
            maximum_snapshot_age_sec=maximum_snapshot_age_sec,
            maximum_arm_start_drift_rad=maximum_arm_start_drift_rad,
            maximum_gripper_start_drift=maximum_gripper_start_drift,
        )
    except Exception:
        try:
            hold_controller("registered shadow replay admission evaluation failed")
        except Exception:  # noqa: BLE001, S110 - preserve the original failure.
            pass
        raise
    evidence = [f"shadow-replay-admission:{admission.semantic_content_hash}"]

    session_active = False
    terminal_tcp_receipt: TerminalToolTcpVerificationReceipt | None = None

    def terminal_hold(reason: str) -> bool:
        try:
            receipt = hold_controller(reason)
            if not isinstance(receipt, RealSkillMotionHoldReceipt):
                raise TypeError(
                    "hold_controller must return RealSkillMotionHoldReceipt."
                )
        except Exception as exc:  # noqa: BLE001 - preserve a terminal receipt.
            evidence.append(
                "hold-port-error:"
                + canonical_content_hash(
                    {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "reason": reason,
                    },
                    domain="agenticrobot.shadow_execution.hold_error.v1",
                )
            )
            return False
        evidence.extend(receipt.evidence_refs)
        return receipt.succeeded

    def terminal(
        status: RegisteredTrajectoryRealReplayStatus,
        *,
        executed_count: int,
        hold_reason: str,
        motion_may_have_executed: bool | None = None,
    ) -> RegisteredTrajectoryRealReplayReceipt:
        nonlocal session_active
        if session_active:
            try:
                assert end_replay_session is not None
                end_refs = tuple(end_replay_session(hold_reason))
                if not end_refs or any(not str(value).strip() for value in end_refs):
                    raise RuntimeError("replay session end returned no evidence.")
                evidence.extend(str(value).strip() for value in end_refs)
            except Exception as exc:  # noqa: BLE001 - HOLD still takes precedence.
                evidence.append(
                    "replay-session-end-error:"
                    + canonical_content_hash(
                        {
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "reason": hold_reason,
                        },
                        domain="agenticrobot.shadow_execution.session_end_error.v1",
                    )
                )
            finally:
                session_active = False
        held = terminal_hold(hold_reason)
        final_status = (
            status
            if held or status is not RegisteredTrajectoryRealReplayStatus.COMPLETED
            else RegisteredTrajectoryRealReplayStatus.TERMINAL_HOLD_FAILED
        )
        if not held:
            evidence.append(
                "terminal-hold-unconfirmed:"
                + canonical_content_hash(
                    {
                        "primary_status": status.value,
                        "hold_reason": hold_reason,
                    },
                    domain="agenticrobot.shadow_execution.real_hold_unconfirmed.v1",
                )
            )
        return RegisteredTrajectoryRealReplayReceipt(
            status=final_status,
            admission=admission,
            executed_point_count=executed_count,
            terminal_hold_confirmed=held,
            real_motion_executed=executed_count > 0,
            evidence_refs=tuple(evidence),
            real_motion_may_have_executed=bool(motion_may_have_executed),
            terminal_tool_tcp_verification=terminal_tcp_receipt,
        )

    def lower_port_error(
        exc: Exception,
        *,
        executed_count: int,
        motion_may_have_executed: bool = False,
    ) -> RegisteredTrajectoryRealReplayReceipt:
        evidence.append(
            "lower-port-error:"
            + canonical_content_hash(
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "executed_point_count": executed_count,
                },
                domain="agenticrobot.shadow_execution.real_replay_error.v1",
            )
        )
        return terminal(
            RegisteredTrajectoryRealReplayStatus.LOWER_PORT_ERROR,
            executed_count=executed_count,
            hold_reason="registered shadow replay lower port error",
            motion_may_have_executed=(
                motion_may_have_executed or executed_count > 0
            ),
        )

    if not admission.admissible:
        return terminal(
            RegisteredTrajectoryRealReplayStatus.PREFLIGHT_REJECTED,
            executed_count=0,
            hold_reason="registered shadow replay preflight rejected",
        )
    if not execution_enabled:
        return terminal(
            RegisteredTrajectoryRealReplayStatus.EXECUTION_DISABLED,
            executed_count=0,
            hold_reason="registered shadow replay execution disabled",
        )
    if terminal_tool_tcp_verification_required and not callable(
        terminal_tool_tcp_verifier
    ):
        evidence.append(
            "pre-dispatch-state-rejected:"
            + canonical_content_hash(
                {"reason_codes": ("terminal_tool_tcp_verifier_missing",)},
                domain="agenticrobot.shadow_execution.pre_dispatch_state.v1",
            )
        )
        return terminal(
            RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED,
            executed_count=0,
            hold_reason=(
                "registered shadow replay terminal TCP verifier missing"
            ),
        )
    if not callable(begin_replay_session) or not callable(end_replay_session):
        evidence.append(
            "pre-dispatch-state-rejected:"
            + canonical_content_hash(
                {"reason_codes": ("replay_session_controller_missing",)},
                domain="agenticrobot.shadow_execution.pre_dispatch_state.v1",
            )
        )
        return terminal(
            RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED,
            executed_count=0,
            hold_reason="registered shadow replay session controller missing",
        )
    if not callable(current_scene_hash_provider):
        evidence.append(
            "pre-dispatch-state-rejected:"
            + canonical_content_hash(
                {"reason_codes": ("current_scene_hash_provider_missing",)},
                domain="agenticrobot.shadow_execution.pre_dispatch_state.v1",
            )
        )
        return terminal(
            RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED,
            executed_count=0,
            hold_reason="registered shadow replay scene provider missing",
        )
    authority_valid = (
        isinstance(authority, RegisteredTrajectoryExecutionAuthority)
        and authority.issued_at <= now < authority.expires_at
        and authority.trajectory_hash == trajectory.semantic_content_hash
        and authority.simulation_receipt_hash
        == simulation_receipt.semantic_content_hash
    )
    if not authority_valid or not callable(consume_execution_authority):
        return terminal(
            RegisteredTrajectoryRealReplayStatus.AUTHORITY_REJECTED,
            executed_count=0,
            hold_reason="registered shadow replay authority rejected",
        )
    assert authority is not None
    try:
        consumption = consume_execution_authority(authority)
    except Exception as exc:  # noqa: BLE001 - fail closed before any dispatch.
        return lower_port_error(exc, executed_count=0)
    consumption_valid = bool(
        isinstance(consumption, RegisteredTrajectoryAuthorityConsumptionReceipt)
        and consumption.consumed
        and consumption.authority_hash == authority.semantic_content_hash
        and consumption.authority_id == authority.authority_id
        and consumption.trajectory_hash == trajectory.semantic_content_hash
        and consumption.simulation_receipt_hash
        == simulation_receipt.semantic_content_hash
        and consumption.run_id == authority.run_id
        and consumption.owner_epoch == authority.owner_epoch
        and authority.issued_at <= consumption.consumed_at < authority.expires_at
    )
    if not consumption_valid:
        if isinstance(consumption, RegisteredTrajectoryAuthorityConsumptionReceipt):
            evidence.extend(consumption.evidence_refs)
        return terminal(
            RegisteredTrajectoryRealReplayStatus.AUTHORITY_REJECTED,
            executed_count=0,
            hold_reason="registered shadow replay authority consumption failed",
        )
    evidence.extend(consumption.evidence_refs)
    evidence.append(
        "authority-consumption-receipt:" + consumption.semantic_content_hash
    )
    evidence.append(authority.operator_confirmation_ref)
    try:
        session_refs = tuple(begin_replay_session(consumption))
        if not session_refs or any(not str(value).strip() for value in session_refs):
            raise RuntimeError("replay session begin returned no evidence.")
        evidence.extend(str(value).strip() for value in session_refs)
        session_active = True
    except Exception as exc:  # noqa: BLE001 - no dispatch can follow failed begin.
        return lower_port_error(exc, executed_count=0)

    executed_count = 0
    dispatch_attempted = False
    try:
        started = _finite(monotonic_clock(), name="monotonic_clock()")
        for point_index, point in enumerate(trajectory.points):
            due = started + point.time_from_start_sec
            while True:
                current_monotonic = _finite(
                    monotonic_clock(), name="monotonic_clock()"
                )
                remaining = due - current_monotonic
                if remaining <= 0.0:
                    break
                sleep(min(remaining, poll))
            lateness = current_monotonic - due
            if lateness > max_lateness:
                evidence.append(
                    "timeline-deadline-missed:"
                    + canonical_content_hash(
                        {
                            "point_index": point_index,
                            "point_hash": point.semantic_content_hash,
                            "lateness_sec": lateness,
                            "maximum_point_lateness_sec": max_lateness,
                        },
                        domain="agenticrobot.shadow_execution.timeline_deadline.v1",
                    )
                )
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.TIMELINE_DEADLINE_MISSED,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay timeline deadline missed",
                )
            pre_dispatch = current_snapshot_provider()
            if not isinstance(pre_dispatch, AuthoritativeRobotStateSnapshot):
                raise TypeError(
                    "current_snapshot_provider must return "
                    "AuthoritativeRobotStateSnapshot."
                )
            point_now = _finite(wall_clock(), name="wall_clock()")
            pre_dispatch_reasons: list[str] = []
            if (
                pre_dispatch.identity.semantic_content_hash
                != trajectory.initial_snapshot.identity.semantic_content_hash
            ):
                pre_dispatch_reasons.append("identity_changed")
            observed_scene_hash = str(current_scene_hash_provider()).strip()
            if not observed_scene_hash or observed_scene_hash != scene_hash:
                pre_dispatch_reasons.append("scene_changed")
            if (
                point_now < pre_dispatch.available_at
                or point_now - pre_dispatch.oldest_measurement_at
                > maximum_snapshot_age_sec
            ):
                pre_dispatch_reasons.append("snapshot_stale")
            expected_start = (
                trajectory.initial_snapshot.qpos_by_side
                if point_index == 0
                else trajectory.points[point_index - 1].qpos_by_side
            )
            arm_tracking_drift = max(
                float(
                    np.max(
                        np.abs(
                            pre_dispatch.qpos_by_side[side]
                            - expected_start[side]
                        )
                    )
                )
                for side in _SIDES
            )
            if arm_tracking_drift > maximum_arm_start_drift_rad:
                pre_dispatch_reasons.append("arm_tracking_drift")
            if pre_dispatch_reasons:
                evidence.append(
                    "pre-dispatch-state-rejected:"
                    + canonical_content_hash(
                        {
                            "point_index": point_index,
                            "point_hash": point.semantic_content_hash,
                            "reason_codes": tuple(pre_dispatch_reasons),
                            "snapshot_hash": pre_dispatch.semantic_content_hash,
                            "arm_tracking_drift_rad": arm_tracking_drift,
                        },
                        domain="agenticrobot.shadow_execution.pre_dispatch_state.v1",
                    )
                )
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay pre-dispatch state rejected",
                )
            pre_dispatch_gripper_drift = max(
                abs(
                    pre_dispatch.gripper_close_by_side[side]
                    - trajectory.initial_snapshot.gripper_close_by_side[side]
                )
                for side in _SIDES
            )
            if pre_dispatch_gripper_drift > maximum_gripper_start_drift:
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.GRIPPER_HOLD_DRIFT,
                    executed_count=executed_count,
                    hold_reason=(
                        "registered shadow replay pre-dispatch gripper drift"
                    ),
                )
            dispatch_attempted = True
            dispatch = point_dispatcher(point, trajectory.semantic_content_hash)
            if not isinstance(dispatch, RegisteredTrajectoryPointDispatchReceipt):
                raise TypeError(
                    "point_dispatcher must return "
                    "RegisteredTrajectoryPointDispatchReceipt."
                )
            if (
                dispatch.point_hash != point.semantic_content_hash
                or not dispatch.succeeded
            ):
                evidence.extend(dispatch.evidence_refs)
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.DISPATCH_FAILED,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay point dispatch failed",
                    # A failed aggregate dual-arm receipt cannot prove that
                    # neither lower transport accepted its side of the batch.
                    motion_may_have_executed=True,
                )
            executed_count += 1
            dispatch_attempted = False
            evidence.extend(dispatch.evidence_refs)
            observed = current_snapshot_provider()
            if not isinstance(observed, AuthoritativeRobotStateSnapshot):
                raise TypeError(
                    "current_snapshot_provider must return "
                    "AuthoritativeRobotStateSnapshot."
                )
            gripper_drift = max(
                abs(
                    observed.gripper_close_by_side[side]
                    - trajectory.initial_snapshot.gripper_close_by_side[side]
                )
                for side in _SIDES
            )
            if gripper_drift > maximum_gripper_start_drift:
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.GRIPPER_HOLD_DRIFT,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay gripper hold drift",
                )

        final_qpos = trajectory.points[-1].qpos_by_side
        previous_final = observed
        previous_final_time = _finite(
            monotonic_clock(), name="monotonic_clock()"
        )
        deadline = previous_final_time + final_timeout
        dwell_started_at: float | None = None
        while True:
            observed = current_snapshot_provider()
            if not isinstance(observed, AuthoritativeRobotStateSnapshot):
                raise TypeError(
                    "current_snapshot_provider must return "
                    "AuthoritativeRobotStateSnapshot."
                )
            final_error = max(
                float(
                    np.max(
                        np.abs(observed.qpos_by_side[side] - final_qpos[side])
                    )
                )
                for side in _SIDES
            )
            gripper_drift = max(
                abs(
                    observed.gripper_close_by_side[side]
                    - trajectory.initial_snapshot.gripper_close_by_side[side]
                )
                for side in _SIDES
            )
            current_monotonic = _finite(
                monotonic_clock(), name="monotonic_clock()"
            )
            sample_dt = current_monotonic - previous_final_time
            if sample_dt < 0.0:
                raise RuntimeError("monotonic clock moved backwards.")
            maximum_velocity = max(
                float(
                    np.max(
                        np.abs(
                            observed.qpos_by_side[side]
                            - previous_final.qpos_by_side[side]
                        )
                    )
                )
                / sample_dt
                if sample_dt > 0.0
                else (
                    0.0
                    if np.array_equal(
                        observed.qpos_by_side[side],
                        previous_final.qpos_by_side[side],
                    )
                    else math.inf
                )
                for side in _SIDES
            )
            if gripper_drift > maximum_gripper_start_drift:
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.GRIPPER_HOLD_DRIFT,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay final gripper drift",
                )
            tcp_ready = terminal_tool_tcp_verifier is None
            if terminal_tool_tcp_verifier is not None:
                terminal_tcp_receipt = terminal_tool_tcp_verifier(observed)
                if not isinstance(
                    terminal_tcp_receipt,
                    TerminalToolTcpVerificationReceipt,
                ):
                    raise TypeError(
                        "terminal_tool_tcp_verifier must return "
                        "TerminalToolTcpVerificationReceipt."
                    )
                evidence.extend(terminal_tcp_receipt.evidence_refs)
                if terminal_tcp_receipt.status is (
                    TerminalToolTcpVerificationStatus.VERIFICATION_UNAVAILABLE
                ):
                    return terminal(
                        RegisteredTrajectoryRealReplayStatus.
                        FINAL_TCP_VERIFICATION_UNAVAILABLE,
                        executed_count=executed_count,
                        hold_reason=(
                            "registered shadow replay terminal TCP "
                            "verification unavailable"
                        ),
                    )
                tcp_ready = terminal_tcp_receipt.verified
            tracking_ready = bool(
                final_error <= final_tolerance
                and maximum_velocity <= final_velocity_tolerance
                and tcp_ready
            )
            if tracking_ready:
                if dwell_started_at is None:
                    dwell_started_at = current_monotonic
            else:
                dwell_started_at = None
            if (
                dwell_started_at is not None
                and current_monotonic - dwell_started_at >= final_dwell
            ):
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.COMPLETED,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay completed",
                )
            if current_monotonic >= deadline:
                if (
                    terminal_tcp_receipt is not None
                    and terminal_tcp_receipt.status
                    is TerminalToolTcpVerificationStatus.TARGET_MISMATCH
                    and final_error <= final_tolerance
                    and maximum_velocity <= final_velocity_tolerance
                ):
                    return terminal(
                        RegisteredTrajectoryRealReplayStatus.
                        FINAL_TCP_TARGET_MISMATCH,
                        executed_count=executed_count,
                        hold_reason=(
                            "registered shadow replay terminal TCP target mismatch"
                        ),
                    )
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.FINAL_TRACKING_TIMEOUT,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay final tracking timeout",
                )
            previous_final = observed
            previous_final_time = current_monotonic
            sleep(min(poll, deadline - current_monotonic))
    except Exception as exc:  # noqa: BLE001 - any lower error must end in HOLD.
        return lower_port_error(
            exc,
            executed_count=executed_count,
            motion_may_have_executed=dispatch_attempted,
        )


__all__ = [
    "MeasuredGripperBackend",
    "MeasuredGripperStateCut",
    "RegisteredTrajectoryAuthorityConsumptionReceipt",
    "RegisteredTrajectoryAuthorityConsumptionStatus",
    "RegisteredTrajectoryExecutionAuthority",
    "RegisteredTrajectoryPointDispatchReceipt",
    "RegisteredTrajectoryPointDispatcher",
    "RegisteredTrajectoryRealReplayReceipt",
    "RegisteredTrajectoryRealReplayStatus",
    "SingleUseRegisteredTrajectoryAuthorityConsumer",
    "TerminalToolTcpVerificationReceipt",
    "TerminalToolTcpVerificationStatus",
    "TerminalToolTcpVerifier",
    "capture_authoritative_shadow_state_from_backends",
    "execute_registered_shadow_trajectory_with_ports",
    "read_measured_gripper_state_cut",
]
