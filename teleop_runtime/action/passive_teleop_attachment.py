"""Deferred attachment port for a passive Human component source."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.diagnostics import TeleopRunState
from teleop_runtime.action.human_proposal import HumanCommandEmission
from teleop_runtime.session import SessionStepResult

from .passive_teleop_contracts import (
    PASSIVE_TELEOP_COMPONENT_BINDING_HASH_DOMAIN_V1,
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentStep,
)


class AttachablePassiveTeleopComponentSource:
    """Stable passive source port whose concrete device may attach later.

    A policy-only runtime can start without opening Human devices. The port
    owns no receiver or worker before :meth:`attach`; attachment is single-use
    and must preserve the exact reserved component source contract.
    """

    def __init__(self, spec: PassiveTeleopComponentSourceSpec) -> None:
        if not isinstance(spec, PassiveTeleopComponentSourceSpec):
            raise TypeError("spec must be a PassiveTeleopComponentSourceSpec.")
        self.spec = spec
        payload = {
            "source_contract_hash": spec.semantic_content_hash,
            "observer_binding_id": spec.observer_binding_id,
            "session_mode": "deferred-passive-component-source",
            "rig_absent": True,
            "observer_exclusive": True,
            "dispatch_forbidden_states": (
                TeleopRunState.ARMED.value,
                TeleopRunState.RUNNING.value,
            ),
            "authoritative": False,
        }
        self.binding_receipt = PassiveTeleopComponentBindingReceipt(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=PASSIVE_TELEOP_COMPONENT_BINDING_HASH_DOMAIN_V1,
            ),
        )
        self._delegate: Any | None = None
        self._started = False
        self._calibration_ready = False
        self._closed = False
        self._lock = threading.RLock()
        self._unattached_session = SimpleNamespace(
            state_machine=SimpleNamespace(state=TeleopRunState.DEBUG_ONLY),
            sources=(),
            retargeters=(),
            calibration=None,
            home_recalibration_required=True,
            motion_hold_mode="none",
        )

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._delegate is not None

    @property
    def session(self) -> Any:
        with self._lock:
            delegate = self._delegate
            return self._unattached_session if delegate is None else delegate.session

    @property
    def session_state(self) -> TeleopRunState:
        with self._lock:
            delegate = self._delegate
            return (
                TeleopRunState.DEBUG_ONLY
                if delegate is None
                else delegate.session_state
            )

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started and not self._closed

    @property
    def last_emission(self) -> HumanCommandEmission | None:
        with self._lock:
            delegate = self._delegate
            return None if delegate is None else delegate.last_emission

    @property
    def last_observed_emission_sequence(self) -> int | None:
        with self._lock:
            delegate = self._delegate
            return (
                None
                if delegate is None
                else delegate.last_observed_emission_sequence
            )

    @property
    def last_rejection_reason(self) -> str | None:
        with self._lock:
            delegate = self._delegate
            return None if delegate is None else delegate.last_rejection_reason

    def start(self) -> PassiveTeleopComponentBindingReceipt:
        with self._lock:
            if self._closed:
                raise RuntimeError("Attachable passive teleop source is closed.")
            delegate = self._delegate
            if delegate is not None and not bool(
                getattr(delegate, "started", False)
            ):
                delegate.start()
                if self._calibration_ready:
                    delegate.set_calibration_ready()
            self._started = True
            return self.binding_receipt

    def set_calibration_ready(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Attachable passive teleop source is closed.")
            self._calibration_ready = True
            delegate = self._delegate
            if delegate is not None:
                delegate.set_calibration_ready()

    def attach(self, delegate: Any) -> dict[str, object]:
        """Attach one exact passive delegate without replacing the port."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Attachable passive teleop source is closed.")
            if self._delegate is not None:
                raise RuntimeError("A passive human source is already attached.")
            delegate_spec = getattr(delegate, "spec", None)
            if not isinstance(
                delegate_spec,
                PassiveTeleopComponentSourceSpec,
            ) or delegate_spec.semantic_content_hash != (
                self.spec.semantic_content_hash
            ):
                raise ValueError(
                    "Attached passive source does not match the reserved "
                    "component source contract."
                )
            receipt = getattr(delegate, "binding_receipt", None)
            if not bool(getattr(receipt, "rig_absent", False)) or bool(
                getattr(receipt, "authoritative", True)
            ):
                raise ValueError(
                    "Attached human source must remain passive and rig-free."
                )
            try:
                if self._started:
                    delegate.start()
                    if self._calibration_ready:
                        delegate.set_calibration_ready()
            except BaseException:
                try:
                    delegate.close()
                finally:
                    raise
            self._delegate = delegate
            return {
                "status": "attached",
                "source_contract_hash": self.spec.semantic_content_hash,
                "delegate_binding_hash": getattr(
                    receipt,
                    "semantic_content_hash",
                    None,
                ),
                "command_dispatched": False,
                "authoritative": False,
            }

    def step(self, *, now: float | None = None) -> PassiveTeleopComponentStep:
        with self._lock:
            if self._closed:
                raise RuntimeError("Attachable passive teleop source is closed.")
            delegate = self._delegate
        if delegate is not None:
            return delegate.step(now=now)
        timestamp = time.monotonic() if now is None else float(now)
        return PassiveTeleopComponentStep(
            session_step=SessionStepResult(
                timestamp=timestamp,
                state=TeleopRunState.DEBUG_ONLY,
                control_hz=0.0,
            ),
            emission=None,
            emission_valid=False,
            rejection_reason="human_source_not_attached",
            command_dispatched=False,
            authoritative=False,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            delegate = self._delegate
            self._delegate = None
        if delegate is not None:
            delegate.close()


__all__ = ["AttachablePassiveTeleopComponentSource"]
