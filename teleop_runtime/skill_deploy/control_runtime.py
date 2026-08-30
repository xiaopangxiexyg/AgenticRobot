"""Single-owner control cycle for an exclusive real Skill deployment.

The runtime is deliberately independent of RealTeleop. Concrete deployment
assembly injects already-started observation, safety/admission, dispatch, and
HOLD ports from the lower real-control stack. The Agent Gateway never receives
those ports and cannot bypass this cycle.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol

import numpy as np

from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.command_override import TeleopCommandOverride

from .motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortSnapshot,
)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


class RealSkillControlLifecycle(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    HOLD = "hold"
    FAULT = "fault"
    CLOSED = "closed"


class RealSkillMotionTargetSynchronizer(Protocol):
    """Lower numerical port that aligns the coupled target solver."""

    def __call__(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
        /,
    ) -> None: ...


@dataclass(frozen=True)
class RealSkillMotionObservationCut:
    """One already-read lower observation and its exact dual-arm qpos cut."""

    observation: TeleopObservation
    qpos_by_side: Mapping[str, np.ndarray]
    measured_at: float
    available_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TeleopObservation):
            raise TypeError("observation must be TeleopObservation.")
        measured_at = _finite(self.measured_at, field_name="measured_at")
        available_at = _finite(self.available_at, field_name="available_at")
        if available_at < measured_at:
            raise ValueError("available_at cannot precede measured_at.")
        qpos = {
            str(side): np.asarray(value, dtype=float).reshape(-1).copy()
            for side, value in dict(self.qpos_by_side).items()
        }
        if set(qpos) != {"left", "right"}:
            raise ValueError("qpos_by_side must contain exactly left and right.")
        if any(
            value.shape != (6,) or not np.all(np.isfinite(value))
            for value in qpos.values()
        ):
            raise ValueError("each qpos_by_side value must contain six finite values.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("an observation cut requires lower evidence.")
        object.__setattr__(self, "qpos_by_side", MappingProxyType(qpos))
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class RealSkillMotionDispatchReceipt:
    """Lower dispatch result; only a successful receipt advances evidence."""

    succeeded: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if self.succeeded and not refs:
            raise ValueError("a successful dispatch requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class RealSkillMotionHoldReceipt:
    """Lower-owned HOLD result retained by the exclusive deployment owner."""

    succeeded: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if self.succeeded and not refs:
            raise ValueError("a successful HOLD requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class RealSkillControlStep:
    lifecycle: RealSkillControlLifecycle
    port_snapshot: RealAgentMotionPortSnapshot
    command_present: bool
    dispatch_attempted: bool
    dispatch_succeeded: bool
    dispatch_evidence_refs: tuple[str, ...]


class ExclusiveRealSkillControlRuntime:
    """One control owner for a Skill-only real deployment.

    The injected callbacks are lower application ports. They are not exposed
    through the Gateway, and this runtime never accepts a second command source.
    """

    def __init__(
        self,
        *,
        motion_port: RealAgentMotionPort,
        observation_provider: Callable[[], RealSkillMotionObservationCut],
        synchronize_target_builder: RealSkillMotionTargetSynchronizer,
        command_admission: Callable[[TeleopCommand, TeleopObservation, float], bool],
        command_dispatch: Callable[[TeleopCommand], RealSkillMotionDispatchReceipt],
        hold_controller: Callable[[str], RealSkillMotionHoldReceipt],
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(motion_port, RealAgentMotionPort):
            raise TypeError("motion_port must be RealAgentMotionPort.")
        if not motion_port.config.exclusive_command_claim:
            raise ValueError("Skill deployment requires an exclusive motion port.")
        for name, callback in (
            ("observation_provider", observation_provider),
            ("synchronize_target_builder", synchronize_target_builder),
            ("command_admission", command_admission),
            ("command_dispatch", command_dispatch),
            ("hold_controller", hold_controller),
            ("clock", clock),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable.")
        self.motion_port = motion_port
        self._observation_provider = observation_provider
        self._synchronize_target_builder = synchronize_target_builder
        self._command_admission = command_admission
        self._command_dispatch = command_dispatch
        self._hold_controller = hold_controller
        self._clock = clock
        self._lock = threading.Lock()
        # Serialize an entire lower control cut against fail-closed/HOLD.  A
        # lifecycle flag alone is insufficient: a step that has already
        # passed admission could otherwise dispatch after a concurrent HOLD.
        self._cycle_lock = threading.Lock()
        self._lifecycle = RealSkillControlLifecycle.CREATED
        self._last_hold_receipt: RealSkillMotionHoldReceipt | None = None

    @property
    def lifecycle(self) -> RealSkillControlLifecycle:
        with self._lock:
            return self._lifecycle

    @property
    def last_hold_receipt(self) -> RealSkillMotionHoldReceipt | None:
        with self._lock:
            return self._last_hold_receipt

    def start(self) -> str:
        with self._lock:
            if self._lifecycle != RealSkillControlLifecycle.CREATED:
                raise RuntimeError("exclusive Skill runtime can start only once.")
            evidence = self.motion_port.confirm_exclusive_attachment()
            self._lifecycle = RealSkillControlLifecycle.RUNNING
            return evidence

    def skill_motion_owner_state(self) -> tuple[str, str]:
        lifecycle = self.lifecycle
        if lifecycle == RealSkillControlLifecycle.RUNNING:
            return "running", "ready"
        if lifecycle == RealSkillControlLifecycle.FAULT:
            return "fault", "fault"
        return "hold", "hold"

    def hold_skill_motion(self, reason: str) -> bool:
        receipt = self._hold_controller(
            _required(reason, field_name="hold_reason")
        )
        if not isinstance(receipt, RealSkillMotionHoldReceipt):
            raise TypeError("hold_controller must return RealSkillMotionHoldReceipt.")
        with self._lock:
            self._last_hold_receipt = receipt
            self._lifecycle = (
                RealSkillControlLifecycle.HOLD
                if receipt.succeeded
                else RealSkillControlLifecycle.FAULT
            )
        return receipt.succeeded

    def _resume_confirmed_hold_for_pending_activation(self) -> bool:
        """Reopen only a successful HOLD that now has a new activation."""

        if self.motion_port.snapshot().phase != RealAgentMotionPhase.PENDING_ACTIVATION:
            return False
        with self._lock:
            if self._lifecycle != RealSkillControlLifecycle.HOLD:
                return False
            receipt = self._last_hold_receipt
            if receipt is None or not receipt.succeeded:
                self._lifecycle = RealSkillControlLifecycle.FAULT
                raise RuntimeError(
                    "exclusive Skill runtime cannot resume without a successful "
                    "lower HOLD receipt."
                )
            # The previous receipt proves only the previous terminal boundary.
            # Clear it before exposing RUNNING so shutdown or completion must
            # obtain new lower HOLD evidence for the new operation.
            self._last_hold_receipt = None
            self._lifecycle = RealSkillControlLifecycle.RUNNING
            return True

    def step(self, *, now: float | None = None) -> RealSkillControlStep:
        with self._cycle_lock:
            return self._step(now=now)

    def _step(self, *, now: float | None = None) -> RealSkillControlStep:
        current_time = _finite(
            self._clock() if now is None else now,
            field_name="now",
        )
        if self.lifecycle in {
            RealSkillControlLifecycle.CREATED,
            RealSkillControlLifecycle.CLOSED,
        }:
            raise RuntimeError("exclusive Skill runtime is not active.")
        cut = self._observation_provider()
        if not isinstance(cut, RealSkillMotionObservationCut):
            raise TypeError(
                "observation_provider must return RealSkillMotionObservationCut."
            )
        control_cut_at = max(current_time, cut.available_at)
        self._synchronize_target_builder(cut.qpos_by_side)
        self.motion_port.publish_measured_qpos_pair(
            qpos_by_side=cut.qpos_by_side,
            measured_at=cut.measured_at,
            # The source cut may already have been available before this
            # cycle.  The derived TCP cut only becomes available to this
            # control owner when the cycle consumes it.
            available_at=control_cut_at,
            source_evidence_refs=cut.evidence_refs,
        )
        self._resume_confirmed_hold_for_pending_activation()
        snapshot = self.motion_port.apply_control_owner_requests(
            owner=self,
            now=control_cut_at,
        )
        self.motion_port.publish_control_owner_state(
            owner=self,
            captured_at=control_cut_at,
        )
        if self.lifecycle != RealSkillControlLifecycle.RUNNING:
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=snapshot,
                command_present=False,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        value = self.motion_port.command_for_observation(
            cut.observation,
            current_time,
        )
        command = value.command if isinstance(value, TeleopCommandOverride) else value
        if command is None:
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=self.motion_port.snapshot(),
                command_present=False,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        if not isinstance(command, TeleopCommand):
            raise TypeError("motion port returned an unsupported command value.")
        admitted = bool(
            self.motion_port.admit_command(command, cut.observation, current_time)
            and self._command_admission(command, cut.observation, current_time)
        )
        if not admitted:
            self.motion_port.observe_session_step(
                command=command,
                command_dispatched=False,
            )
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=self.motion_port.snapshot(),
                command_present=True,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        receipt = self._command_dispatch(command)
        if not isinstance(receipt, RealSkillMotionDispatchReceipt):
            raise TypeError(
                "command_dispatch must return RealSkillMotionDispatchReceipt."
            )
        self.motion_port.observe_session_step(
            command=command,
            command_dispatched=receipt.succeeded,
        )
        if not receipt.succeeded:
            operation_id = self.motion_port.snapshot().operation_id
            if operation_id is not None:
                self.motion_port.request_hold(
                    operation_id=operation_id,
                    reason="lower_dispatch_failed",
                    requested_at=current_time,
                )
                self.motion_port.apply_control_owner_requests(
                    owner=self,
                    now=current_time,
                )
        return RealSkillControlStep(
            lifecycle=self.lifecycle,
            port_snapshot=self.motion_port.snapshot(),
            command_present=True,
            dispatch_attempted=True,
            dispatch_succeeded=receipt.succeeded,
            dispatch_evidence_refs=receipt.evidence_refs,
        )

    def fail_closed(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> RealSkillMotionHoldReceipt:
        """Reach lower-confirmed HOLD without requiring another observation."""

        with self._cycle_lock:
            return self._fail_closed(reason, now=now)

    def _fail_closed(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> RealSkillMotionHoldReceipt:

        current_time = _finite(
            self._clock() if now is None else now,
            field_name="now",
        )
        if self.lifecycle == RealSkillControlLifecycle.CLOSED:
            raise RuntimeError("closed exclusive Skill runtime cannot request HOLD.")
        snapshot = self.motion_port.snapshot()
        receipt = self.last_hold_receipt
        if (
            snapshot.phase == RealAgentMotionPhase.HELD
            and snapshot.hold_succeeded is True
            and receipt is not None
            and receipt.succeeded
        ):
            return receipt
        normalized_reason = _required(reason, field_name="hold_reason")
        if snapshot.phase not in {
            RealAgentMotionPhase.HOLD_REQUESTED,
            RealAgentMotionPhase.HOLD_APPLYING,
        }:
            if snapshot.operation_id is None:
                self.motion_port.request_domain_hold(
                    reason=normalized_reason,
                    requested_at=current_time,
                )
            else:
                self.motion_port.request_hold(
                    operation_id=snapshot.operation_id,
                    reason=normalized_reason,
                    requested_at=current_time,
                )
        snapshot = self.motion_port.apply_control_owner_requests(
            owner=self,
            now=current_time,
        )
        self.motion_port.publish_control_owner_state(
            owner=self,
            captured_at=current_time,
        )
        receipt = self.last_hold_receipt
        if (
            snapshot.phase != RealAgentMotionPhase.HELD
            or snapshot.hold_succeeded is not True
            or receipt is None
            or not receipt.succeeded
        ):
            raise RuntimeError(
                "exclusive Skill runtime failed to reach lower-confirmed HOLD."
            )
        return receipt

    def close(self, *, reason: str = "exclusive_skill_runtime_shutdown") -> None:
        if self.lifecycle == RealSkillControlLifecycle.CLOSED:
            return
        self.fail_closed(reason)
        with self._lock:
            self._lifecycle = RealSkillControlLifecycle.CLOSED


__all__ = [
    "ExclusiveRealSkillControlRuntime",
    "RealSkillControlLifecycle",
    "RealSkillControlStep",
    "RealSkillMotionDispatchReceipt",
    "RealSkillMotionHoldReceipt",
    "RealSkillMotionObservationCut",
    "RealSkillMotionTargetSynchronizer",
]
