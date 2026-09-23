"""Exclusive lower port for finite real gripper endpoint actions.

The port adapts already-constructed gripper backends.  It never creates a
device and must be owned by the same application that owns backend lifecycle.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import EffectorCommand


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


class GripperActionBackend(Protocol):
    name: str

    def send_command(self, command: EffectorCommand) -> object | None: ...

    def read_gripper_state_now(self) -> Mapping[str, object] | None: ...

    def hold_motion(self, reason: str = "manual hold") -> None: ...


@dataclass(frozen=True)
class RealGripperActionFeedback:
    component_id: str
    close_fraction: float
    is_moving: bool
    error_code: int
    measured_at: float
    available_at: float
    state_source: str
    evidence_refs: tuple[str, ...]
    measurement_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_id", _required(self.component_id, name="component_id")
        )
        fraction = _finite(self.close_fraction, name="close_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("close_fraction must lie in [0, 1].")
        object.__setattr__(self, "close_fraction", fraction)
        if not isinstance(self.is_moving, bool):
            raise TypeError("is_moving must be boolean.")
        if isinstance(self.error_code, bool) or not isinstance(self.error_code, int):
            raise TypeError("error_code must be an integer.")
        measured_at = _finite(self.measured_at, name="measured_at")
        available_at = _finite(self.available_at, name="available_at")
        if measured_at > available_at:
            raise ValueError("feedback cannot be available before measurement.")
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(
            self, "state_source", _required(self.state_source, name="state_source")
        )
        refs = tuple(
            sorted(
                {_required(item, name="evidence_ref") for item in self.evidence_refs}
            )
        )
        if not refs:
            raise ValueError("feedback requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "measurement_hash",
            canonical_content_hash(
                {
                    "component_id": self.component_id,
                    "close_fraction": self.close_fraction,
                    "is_moving": self.is_moving,
                    "error_code": self.error_code,
                    "measured_at": self.measured_at,
                    "available_at": self.available_at,
                    "state_source": self.state_source,
                    "evidence_refs": self.evidence_refs,
                },
                domain="dexteleop.real_gripper_action_feedback.v1",
            ),
        )


@dataclass(frozen=True)
class RealGripperActionDispatchReceipt:
    lower_operation_id: str
    component_id: str
    target_close_fraction: float
    dispatched_at: float
    command_receipt_ref: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("lower_operation_id", "component_id", "command_receipt_ref"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        target = _finite(self.target_close_fraction, name="target_close_fraction")
        if target not in {0.0, 1.0}:
            raise ValueError("target_close_fraction must be a fixed endpoint.")
        object.__setattr__(self, "target_close_fraction", target)
        object.__setattr__(
            self, "dispatched_at", _finite(self.dispatched_at, name="dispatched_at")
        )
        refs = tuple(
            sorted(
                {_required(item, name="evidence_ref") for item in self.evidence_refs}
            )
        )
        if not refs:
            raise ValueError("dispatch requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class RealGripperActionHoldReceipt:
    lower_operation_id: str | None
    succeeded: bool
    status: str
    completed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lower_operation_id is not None:
            object.__setattr__(
                self,
                "lower_operation_id",
                _required(self.lower_operation_id, name="lower_operation_id"),
            )
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be boolean.")
        object.__setattr__(self, "status", _required(self.status, name="status"))
        object.__setattr__(
            self, "completed_at", _finite(self.completed_at, name="completed_at")
        )
        refs = tuple(
            sorted(
                {_required(item, name="evidence_ref") for item in self.evidence_refs}
            )
        )
        if self.succeeded and not refs:
            raise ValueError("successful HOLD requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)


class DMTacGripperActionPort:
    """Serialize fixed endpoint actions through existing DMTac backends."""

    def __init__(
        self,
        *,
        port_id: str,
        clock_domain_id: str,
        backends_by_component: Mapping[str, GripperActionBackend],
        side_by_component: Mapping[str, str],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.port_id = _required(port_id, name="port_id")
        self.clock_domain_id = _required(clock_domain_id, name="clock_domain_id")
        backends = dict(backends_by_component)
        sides = dict(side_by_component)
        if not backends or set(backends) != set(sides):
            raise ValueError(
                "backend and side maps must cover one identical component set."
            )
        for component_id, backend in backends.items():
            _required(component_id, name="component_id")
            side = _required(sides[component_id], name="side")
            if side not in {"left", "right"}:
                raise ValueError("gripper side must be left or right.")
            for method in ("send_command", "read_gripper_state_now", "hold_motion"):
                if not callable(getattr(backend, method, None)):
                    raise TypeError(f"backend for {component_id!r} lacks {method}().")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.backends_by_component = MappingProxyType(backends)
        self.side_by_component = MappingProxyType(sides)
        self._clock = clock
        self._lock = threading.RLock()
        self._active_operation_id: str | None = None
        self._active_component_id: str | None = None

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.backends_by_component))

    def _time(self) -> float:
        return _finite(self._clock(), name="clock result")

    def feedback(self, *, component_id: str) -> RealGripperActionFeedback:
        component = _required(component_id, name="component_id")
        try:
            backend = self.backends_by_component[component]
        except KeyError as exc:
            raise ValueError("component_id is not bound to this port.") from exc
        payload = backend.read_gripper_state_now()
        if not isinstance(payload, Mapping):
            raise TypeError("Lower gripper owner returned no fresh feedback mapping.")
        close_raw = payload.get("joint_pos", payload.get("gripper_measured_scalar"))
        if close_raw is None:
            raise RuntimeError("Fresh gripper feedback has no measured close fraction.")
        moving = payload.get("is_moving", payload.get("gripper_is_moving"))
        if not isinstance(moving, bool):
            raise TypeError("Fresh gripper feedback has no authoritative motion state.")
        measured_at = payload.get("timestamp", payload.get("gripper_timestamp"))
        if measured_at is None:
            raise RuntimeError("Fresh gripper feedback has no measurement timestamp.")
        error_raw = payload.get("error_code", payload.get("gripper_error_code", 0))
        if error_raw is None:
            error_raw = 0
        available_at = self._time()
        evidence_hash = canonical_content_hash(
            {
                "component_id": component,
                "close_fraction": float(close_raw),
                "is_moving": moving,
                "error_code": int(error_raw),
                "measured_at": float(measured_at),
                "available_at": available_at,
            },
            domain="dexteleop.dmtac_gripper_action_feedback_evidence.v1",
        )
        return RealGripperActionFeedback(
            component_id=component,
            close_fraction=float(close_raw),
            is_moving=moving,
            error_code=int(error_raw),
            measured_at=float(measured_at),
            available_at=available_at,
            state_source="dmtac_backend_fresh_read",
            evidence_refs=("dmtac-gripper-feedback:" + evidence_hash,),
        )

    def dispatch(
        self,
        *,
        operation_id: str,
        component_id: str,
        target_close_fraction: float,
    ) -> RealGripperActionDispatchReceipt:
        operation = _required(operation_id, name="operation_id")
        component = _required(component_id, name="component_id")
        target = _finite(target_close_fraction, name="target_close_fraction")
        if target not in {0.0, 1.0}:
            raise ValueError("Only fixed open/close endpoints are supported.")
        with self._lock:
            if self._active_operation_id not in {None, operation}:
                raise RuntimeError("Lower gripper port is busy with another operation.")
            before = self.feedback(component_id=component)
            if before.error_code != 0:
                raise RuntimeError("Lower gripper reports an active hardware fault.")
            backend = self.backends_by_component[component]
            command = EffectorCommand(
                effector_name=str(backend.name),
                side=self.side_by_component[component],
                command_type="gripper_scalar",
                value=target,
                metadata={
                    "source": "agent_gateway_fixed_gripper_endpoint_v1",
                    "operation_id": operation,
                },
            )
            # Fence the component before crossing the device command boundary.
            # If dispatch raises after the backend accepted the command, the
            # lifecycle cleanup must still HOLD this exact gripper.
            self._active_operation_id = operation
            self._active_component_id = component
            sent = backend.send_command(command)
            already_at_endpoint = (
                not before.is_moving and abs(before.close_fraction - target) <= 1e-6
            )
            if sent is False and not already_at_endpoint:
                raise RuntimeError(
                    "Lower gripper backend rejected the endpoint command."
                )
            dispatched_at = self._time()
            receipt_hash = canonical_content_hash(
                {
                    "port_id": self.port_id,
                    "operation_id": operation,
                    "component_id": component,
                    "target_close_fraction": target,
                    "dispatched_at": dispatched_at,
                    "already_at_endpoint": already_at_endpoint,
                    "initial_measurement_hash": before.measurement_hash,
                },
                domain="dexteleop.dmtac_gripper_action_dispatch.v1",
            )
            command_ref = "dmtac-gripper-command:" + receipt_hash
            return RealGripperActionDispatchReceipt(
                lower_operation_id="real-gripper:" + operation,
                component_id=component,
                target_close_fraction=target,
                dispatched_at=dispatched_at,
                command_receipt_ref=command_ref,
                evidence_refs=(command_ref, *before.evidence_refs),
            )

    def hold(
        self,
        *,
        operation_id: str | None,
        reason: str,
        feedback_validator: Callable[[RealGripperActionFeedback], bool] | None = None,
    ) -> RealGripperActionHoldReceipt:
        if feedback_validator is not None and not callable(feedback_validator):
            raise TypeError("feedback_validator must be callable or None.")
        normalized_reason = _required(reason, name="reason")
        operation = (
            None
            if operation_id is None
            else _required(operation_id, name="operation_id")
        )
        with self._lock:
            component = self._active_component_id
            active = self._active_operation_id
            if operation is not None and active not in {None, operation}:
                raise ValueError(
                    "HOLD operation does not match the active gripper action."
                )
            if component is None:
                ref = "dmtac-gripper-hold:" + canonical_content_hash(
                    {
                        "port_id": self.port_id,
                        "reason": normalized_reason,
                        "active": None,
                    },
                    domain="dexteleop.dmtac_gripper_action_hold.v1",
                )
                return RealGripperActionHoldReceipt(
                    lower_operation_id=None
                    if operation is None
                    else "real-gripper:" + operation,
                    succeeded=True,
                    status="gripper_port_idle",
                    completed_at=self._time(),
                    evidence_refs=(ref,),
                )
            self.backends_by_component[component].hold_motion(normalized_reason)
            feedback = self.feedback(component_id=component)
            feedback_accepted = (
                True
                if feedback_validator is None
                else feedback_validator(feedback)
            )
            if not isinstance(feedback_accepted, bool):
                raise TypeError("feedback_validator must return a boolean.")
            succeeded = (
                not feedback.is_moving
                and feedback.error_code == 0
                and feedback_accepted
            )
            ref = "dmtac-gripper-hold:" + canonical_content_hash(
                {
                    "port_id": self.port_id,
                    "operation_id": active,
                    "component_id": component,
                    "reason": normalized_reason,
                    "succeeded": succeeded,
                    "feedback_accepted": feedback_accepted,
                    "error_code": feedback.error_code,
                    "measurement_hash": feedback.measurement_hash,
                },
                domain="dexteleop.dmtac_gripper_action_hold.v1",
            )
            if succeeded:
                self._active_operation_id = None
                self._active_component_id = None
            return RealGripperActionHoldReceipt(
                lower_operation_id=None if active is None else "real-gripper:" + active,
                succeeded=succeeded,
                status="gripper_idle_confirmed"
                if succeeded
                else "gripper_hold_unconfirmed",
                completed_at=self._time(),
                evidence_refs=(ref, *feedback.evidence_refs),
            )


__all__ = [
    "DMTacGripperActionPort",
    "GripperActionBackend",
    "RealGripperActionDispatchReceipt",
    "RealGripperActionFeedback",
    "RealGripperActionHoldReceipt",
]
