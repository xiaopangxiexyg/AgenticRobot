"""Fail-closed lower adapters for registered shadow trajectory preparation.

This module may read already-started gripper backends, but it never constructs
devices, sends commands, or grants real-motion authority.  Its output is one
causally bounded measured gripper cut that can be joined with the existing
dual-arm observation cut before MuJoCo is initialized.
"""

from __future__ import annotations

import math
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
    issued_at: float
    expires_at: float
    real_motion_enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "authority_id",
            "operator_confirmation_ref",
            "trajectory_hash",
            "simulation_receipt_hash",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        issued = _finite(self.issued_at, name="issued_at")
        expires = _finite(self.expires_at, name="expires_at")
        if expires <= issued:
            raise ValueError("expires_at must be after issued_at.")
        if self.real_motion_enabled is not True:
            raise ValueError("execution authority must explicitly enable real motion.")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)


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


class RegisteredTrajectoryRealReplayStatus(str, Enum):
    COMPLETED = "completed"
    EXECUTION_DISABLED = "execution_disabled"
    AUTHORITY_REJECTED = "authority_rejected"
    PREFLIGHT_REJECTED = "preflight_rejected"
    DISPATCH_FAILED = "dispatch_failed"
    FINAL_TRACKING_TIMEOUT = "final_tracking_timeout"
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
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("real replay receipt requires lower evidence.")
        if self.status is RegisteredTrajectoryRealReplayStatus.COMPLETED and (
            not self.admission.admissible
            or self.executed_point_count == 0
            or not self.terminal_hold_confirmed
            or not self.real_motion_executed
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
    current_snapshot_provider: Callable[[], AuthoritativeRobotStateSnapshot],
    point_dispatcher: RegisteredTrajectoryPointDispatcher,
    hold_controller: Callable[[str], RealSkillMotionHoldReceipt],
    authority: RegisteredTrajectoryExecutionAuthority | None = None,
    consume_execution_authority: (
        Callable[[RegisteredTrajectoryExecutionAuthority], tuple[str, ...]] | None
    ) = None,
    execution_enabled: bool = False,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    maximum_snapshot_age_sec: float = 0.2,
    maximum_arm_start_drift_rad: float = 0.01,
    maximum_gripper_start_drift: float = 0.02,
    final_tracking_tolerance_rad: float = 0.01,
    final_tracking_timeout_sec: float = 3.0,
    feedback_poll_sec: float = 0.02,
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
    poll = _finite(feedback_poll_sec, name="feedback_poll_sec")
    if final_tolerance <= 0.0 or final_timeout <= 0.0 or poll <= 0.0:
        raise ValueError("tracking tolerance, timeout, and polling must be positive.")

    try:
        now = _finite(wall_clock(), name="wall_clock()")
        current = current_snapshot_provider()
        if not isinstance(current, AuthoritativeRobotStateSnapshot):
            raise TypeError(
                "current_snapshot_provider must return "
                "AuthoritativeRobotStateSnapshot."
            )
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
        held = terminal_hold(hold_reason)
        final_status = (
            status
            if held
            else RegisteredTrajectoryRealReplayStatus.TERMINAL_HOLD_FAILED
        )
        return RegisteredTrajectoryRealReplayReceipt(
            status=final_status,
            admission=admission,
            executed_point_count=executed_count,
            terminal_hold_confirmed=held,
            real_motion_executed=(
                executed_count > 0
                if motion_may_have_executed is None
                else motion_may_have_executed
            ),
            evidence_refs=tuple(evidence),
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
        consumed_refs = tuple(consume_execution_authority(authority))
    except Exception as exc:  # noqa: BLE001 - fail closed before any dispatch.
        return lower_port_error(exc, executed_count=0)
    if not consumed_refs or any(not str(value).strip() for value in consumed_refs):
        return terminal(
            RegisteredTrajectoryRealReplayStatus.AUTHORITY_REJECTED,
            executed_count=0,
            hold_reason="registered shadow replay authority consumption failed",
        )
    evidence.extend(str(value).strip() for value in consumed_refs)
    evidence.append(authority.operator_confirmation_ref)

    executed_count = 0
    dispatch_attempted = False
    try:
        started = _finite(monotonic_clock(), name="monotonic_clock()")
        for point in trajectory.points:
            due = started + point.time_from_start_sec
            while True:
                current_monotonic = _finite(
                    monotonic_clock(), name="monotonic_clock()"
                )
                remaining = due - current_monotonic
                if remaining <= 0.0:
                    break
                sleep(min(remaining, poll))
            pre_dispatch = current_snapshot_provider()
            if not isinstance(pre_dispatch, AuthoritativeRobotStateSnapshot):
                raise TypeError(
                    "current_snapshot_provider must return "
                    "AuthoritativeRobotStateSnapshot."
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
                    motion_may_have_executed=executed_count > 0,
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
        deadline = (
            _finite(monotonic_clock(), name="monotonic_clock()") + final_timeout
        )
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
            if gripper_drift > maximum_gripper_start_drift:
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.GRIPPER_HOLD_DRIFT,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay final gripper drift",
                )
            if final_error <= final_tolerance:
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.COMPLETED,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay completed",
                )
            current_monotonic = _finite(
                monotonic_clock(), name="monotonic_clock()"
            )
            if current_monotonic >= deadline:
                return terminal(
                    RegisteredTrajectoryRealReplayStatus.FINAL_TRACKING_TIMEOUT,
                    executed_count=executed_count,
                    hold_reason="registered shadow replay final tracking timeout",
                )
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
    "RegisteredTrajectoryExecutionAuthority",
    "RegisteredTrajectoryPointDispatchReceipt",
    "RegisteredTrajectoryPointDispatcher",
    "RegisteredTrajectoryRealReplayReceipt",
    "RegisteredTrajectoryRealReplayStatus",
    "capture_authoritative_shadow_state_from_backends",
    "execute_registered_shadow_trajectory_with_ports",
    "read_measured_gripper_state_cut",
]
