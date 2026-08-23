"""Bounded latest-value admission for authenticated controller pose telemetry."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...protocol import PROTOCOL_VERSION
from .protocol import (
    REFERENCE_SPACE_TYPES,
    ControllerPoseSample,
    parse_controller_pose_sample,
)


@dataclass
class _ClientState:
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    reference_space_generation: int
    last_sequence: int = -1
    last_received_monotonic_ns: int = 0
    latest: ControllerPoseSample | None = None


@dataclass(frozen=True)
class ControllerPoseAcceptedEvent:
    """One sample after all session, sequence, and rate admission checks."""

    event_sequence: int
    server_instance_id: str
    client_id: int
    sample: ControllerPoseSample
    receipt_clock_domain_id: str | None


@dataclass(frozen=True)
class ControllerPoseInvalidationEvent:
    """Ordered evidence that an admitted reference epoch is no longer usable."""

    event_sequence: int
    server_instance_id: str
    client_id: int
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    previous_reference_space_generation: int
    next_reference_space_generation: int | None
    last_accepted_sample_sequence: int | None
    reason: str
    received_monotonic_ns: int
    receipt_clock_domain_id: str | None


@dataclass(frozen=True)
class ControllerPoseActiveSessionCut:
    """Pose-redacted identity visible at one atomic subscription boundary."""

    client_id: int
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    reference_space_generation: int
    last_accepted_sample_sequence: int | None


@dataclass(frozen=True)
class ControllerPoseSubscription:
    """Observer token plus the first event sequence created after registration."""

    token: int
    server_instance_id: str
    next_event_sequence: int
    active_sessions: tuple[ControllerPoseActiveSessionCut, ...]
    receipt_clock_domain_id: str | None


@dataclass(frozen=True)
class ControllerPoseEventCut:
    """Pose-redacted exclusive event watermark at one runtime lock boundary."""

    server_instance_id: str
    next_event_sequence: int
    active_sessions: tuple[ControllerPoseActiveSessionCut, ...]
    receipt_clock_domain_id: str | None
    stopped: bool


@dataclass(frozen=True)
class ControllerPoseObserverCloseCut:
    """Exclusive event watermark proven to atomically remove one observer."""

    subscription_token: int
    server_instance_id: str
    next_event_sequence: int
    active_sessions: tuple[ControllerPoseActiveSessionCut, ...]
    receipt_clock_domain_id: str | None
    stopped: bool


@dataclass(frozen=True)
class _Observer:
    on_sample: Callable[[ControllerPoseAcceptedEvent], None] | None
    on_invalidation: Callable[[ControllerPoseInvalidationEvent], None] | None


class ControllerPoseTelemetryRuntime:
    def __init__(
        self,
        server: Any,
        config: dict[str, Any],
        stop_event: threading.Event,
        *,
        monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        del stop_event
        admission = config.get("admission", {})
        if not isinstance(admission, dict):
            raise TypeError("features.controller_pose.admission must be a mapping")
        self.server = server
        self.server_instance_id = str(getattr(server, "server_instance_id", "")).strip()
        if not self.server_instance_id:
            raise ValueError("controller-pose server_instance_id must be non-empty")
        self.max_rate_hz = max(1.0, float(admission.get("max_rate_hz", 40)))
        self.max_controllers = max(0, int(admission.get("max_controllers", 2)))
        self.max_observers = max(1, int(admission.get("max_observers", 8)))
        raw_receipt_clock = config.get("receipt_clock_domain_id")
        if raw_receipt_clock is None:
            self.receipt_clock_domain_id: str | None = None
        elif (
            not isinstance(raw_receipt_clock, str)
            or not raw_receipt_clock.strip()
            or len(raw_receipt_clock) > 200
        ):
            raise ValueError(
                "features.controller_pose.receipt_clock_domain_id must be a "
                "non-empty string up to 200 chars"
            )
        else:
            self.receipt_clock_domain_id = raw_receipt_clock.strip()
        if not callable(monotonic_ns_clock):
            raise TypeError("monotonic_ns_clock must be callable")
        self.monotonic_ns_clock = monotonic_ns_clock
        self.lock = threading.RLock()
        self.states: dict[int, _ClientState] = {}
        self._observers: dict[int, _Observer] = {}
        self._next_observer_token = 1
        self._next_event_sequence = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.last_rejection: str | None = None
        self.observer_error_count = 0
        self.last_observer_error: str | None = None
        self._stopped = False

    def start(self) -> None:
        with self.lock:
            if self._stopped:
                raise RuntimeError("controller-pose runtime is stopped")
            self.server.register_client_message_handler(
                "xr_session_state", self._handle_session_state
            )
            self.server.register_client_message_handler(
                "reference_space_reset", self._handle_reference_space_reset
            )
            self.server.register_client_message_handler(
                "controller_pose_sample", self._handle_pose_sample
            )
            self.server.register_client_disconnect_handler(self._handle_disconnect)
            self.server.register_feature_status_provider(
                "controller_pose", self.snapshot_status
            )
        print(
            "[quest-dev:controller_pose] authenticated read-only telemetry enabled",
            flush=True,
        )

    def stop(self) -> None:
        with self.lock:
            if self._stopped:
                return
            self._stopped = True
            callbacks = tuple(self._observers.values())
            events = tuple(
                self._invalidation_event_locked(
                    client_id=client_id,
                    state=state,
                    reason="runtime_stopped",
                    next_generation=None,
                )
                for client_id, state in sorted(self.states.items())
            )
            self.states.clear()
            self._observers.clear()
        for event in events:
            self._notify_invalidation(event, callbacks)

    def subscribe(
        self,
        *,
        on_sample: Callable[[ControllerPoseAcceptedEvent], None] | None = None,
        on_invalidation: (
            Callable[[ControllerPoseInvalidationEvent], None] | None
        ) = None,
    ) -> int:
        """Observe future events without advancing or reading the source."""

        return self.subscribe_with_cut(
            on_sample=on_sample,
            on_invalidation=on_invalidation,
        ).token

    def subscribe_with_cut(
        self,
        *,
        on_sample: Callable[[ControllerPoseAcceptedEvent], None] | None = None,
        on_invalidation: (
            Callable[[ControllerPoseInvalidationEvent], None] | None
        ) = None,
    ) -> ControllerPoseSubscription:
        """Atomically register and return the event/session boundary.

        The returned cut intentionally contains no pose.  A consumer can use
        ``next_event_sequence`` as its ingress fence and wait for a new accepted
        event, avoiding a latest-pose/read-then-reset TOCTOU.

        Callbacks run outside the runtime lock. Concurrent HTTP handlers may
        therefore deliver them out of wall-clock order; consumers must fence
        and order by ``event_sequence`` and must not let an older callback
        resurrect an invalidated capture session.
        """

        if on_sample is None and on_invalidation is None:
            raise ValueError("at least one controller-pose observer is required")
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable or None")
        if on_invalidation is not None and not callable(on_invalidation):
            raise TypeError("on_invalidation must be callable or None")
        with self.lock:
            if self._stopped:
                raise RuntimeError("controller-pose runtime is stopped")
            if len(self._observers) >= self.max_observers:
                raise RuntimeError("controller-pose observer limit reached")
            token = self._next_observer_token
            self._next_observer_token += 1
            self._observers[token] = _Observer(on_sample, on_invalidation)
            return ControllerPoseSubscription(
                token=token,
                server_instance_id=self.server_instance_id,
                next_event_sequence=self._next_event_sequence,
                active_sessions=self._active_session_cuts_locked(),
                receipt_clock_domain_id=self.receipt_clock_domain_id,
            )

    def atomic_event_cut(self) -> ControllerPoseEventCut:
        """Return an exclusive event watermark without registering an observer.

        Events allocated before this lock boundary have sequence numbers below
        ``next_event_sequence``. Their callbacks may still be in flight because
        callbacks deliberately execute outside the source lock; consumers must
        drain through the watermark before sealing evidence.
        """

        with self.lock:
            return self._event_cut_locked()

    def _event_cut_locked(self) -> ControllerPoseEventCut:
        return ControllerPoseEventCut(
            server_instance_id=self.server_instance_id,
            next_event_sequence=self._next_event_sequence,
            active_sessions=self._active_session_cuts_locked(),
            receipt_clock_domain_id=self.receipt_clock_domain_id,
            stopped=self._stopped,
        )

    def _active_session_cuts_locked(
        self,
    ) -> tuple[ControllerPoseActiveSessionCut, ...]:
        return tuple(
            ControllerPoseActiveSessionCut(
                client_id=int(client_id),
                client_instance_id=state.client_instance_id,
                xr_session_id=state.xr_session_id,
                reference_space_type=state.reference_space_type,
                reference_space_generation=state.reference_space_generation,
                last_accepted_sample_sequence=(
                    None if state.last_sequence < 0 else state.last_sequence
                ),
            )
            for client_id, state in sorted(self.states.items())
        )

    def unsubscribe(self, token: int) -> bool:
        """Prevent future event snapshots from including ``token``.

        A callback already snapshotted by an in-flight handler may still arrive;
        consumers therefore also need their own closed/event-watermark fence.
        """

        with self.lock:
            return self._observers.pop(int(token), None) is not None

    def unsubscribe_with_cut(self, token: int) -> ControllerPoseObserverCloseCut:
        """Atomically remove one observer and return its exclusive close cut.

        Events allocated before this lock boundary already captured the
        observer and have sequence numbers below the returned watermark.
        Events allocated after it cannot capture the removed observer and have
        sequence numbers at or above the watermark.  This avoids both an
        unsubscribe-to-cut event gap and a cut-to-unsubscribe callback race.
        In-flight callbacks below the cut still need to be drained by sequence.
        """

        with self.lock:
            normalized_token = int(token)
            if self._observers.pop(normalized_token, None) is None:
                raise KeyError("controller-pose observer token is not active")
            cut = self._event_cut_locked()
            return ControllerPoseObserverCloseCut(
                subscription_token=normalized_token,
                server_instance_id=cut.server_instance_id,
                next_event_sequence=cut.next_event_sequence,
                active_sessions=cut.active_sessions,
                receipt_clock_domain_id=cut.receipt_clock_domain_id,
                stopped=cut.stopped,
            )

    def _event_sequence_locked(self) -> int:
        sequence = self._next_event_sequence
        self._next_event_sequence += 1
        return sequence

    def _invalidation_event_locked(
        self,
        *,
        client_id: int,
        state: _ClientState,
        reason: str,
        next_generation: int | None,
    ) -> ControllerPoseInvalidationEvent:
        return ControllerPoseInvalidationEvent(
            event_sequence=self._event_sequence_locked(),
            server_instance_id=self.server_instance_id,
            client_id=int(client_id),
            client_instance_id=state.client_instance_id,
            xr_session_id=state.xr_session_id,
            reference_space_type=state.reference_space_type,
            previous_reference_space_generation=(state.reference_space_generation),
            next_reference_space_generation=next_generation,
            last_accepted_sample_sequence=(
                None if state.last_sequence < 0 else state.last_sequence
            ),
            reason=str(reason),
            received_monotonic_ns=int(self.monotonic_ns_clock()),
            receipt_clock_domain_id=self.receipt_clock_domain_id,
        )

    def _record_observer_error(self, exc: Exception) -> None:
        with self.lock:
            self.observer_error_count += 1
            self.last_observer_error = f"{type(exc).__name__}: {exc}"

    def _notify_sample(
        self,
        event: ControllerPoseAcceptedEvent,
        observers: tuple[_Observer, ...],
    ) -> None:
        for observer in observers:
            if observer.on_sample is None:
                continue
            try:
                observer.on_sample(event)
            except Exception as exc:  # noqa: BLE001 - isolate passive observers.
                self._record_observer_error(exc)

    def _notify_invalidation(
        self,
        event: ControllerPoseInvalidationEvent,
        observers: tuple[_Observer, ...],
    ) -> None:
        for observer in observers:
            if observer.on_invalidation is None:
                continue
            try:
                observer.on_invalidation(event)
            except Exception as exc:  # noqa: BLE001 - isolate passive observers.
                self._record_observer_error(exc)

    @staticmethod
    def _identity(payload: dict[str, Any]) -> tuple[str, str, str, int]:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("core protocol version mismatch")
        client_instance_id = payload.get("client_instance_id")
        xr_session_id = payload.get("xr_session_id")
        reference_space_type = payload.get("reference_space_type")
        generation = payload.get("reference_space_generation")
        for name, value in (
            ("client_instance_id", client_instance_id),
            ("xr_session_id", xr_session_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 160
                or not all(
                    character.isalnum() or character in "._:-" for character in value
                )
            ):
                raise ValueError(f"{name} is not a valid session identifier")
        if reference_space_type not in REFERENCE_SPACE_TYPES:
            raise ValueError("unsupported reference-space type")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError(
                "reference_space_generation must be a non-negative integer"
            )
        return client_instance_id, xr_session_id, reference_space_type, generation

    def _reject(self, exc: Exception) -> None:
        with self.lock:
            self.rejected_count += 1
            self.last_rejection = f"{type(exc).__name__}: {exc}"

    def _handle_session_state(self, client_id: int, payload: dict[str, Any]) -> None:
        invalidation: ControllerPoseInvalidationEvent | None = None
        observers: tuple[_Observer, ...] = ()
        try:
            state = payload.get("state")
            identity = self._identity(payload)
            with self.lock:
                if self._stopped:
                    raise RuntimeError("controller-pose runtime is stopped")
                if state == "started":
                    if identity[3] != 0:
                        raise ValueError(
                            "XR session must start at reference-space generation zero"
                        )
                    if client_id in self.states:
                        raise ValueError("XR session already active for client")
                    self.states[client_id] = _ClientState(*identity)
                elif state == "ended":
                    current = self.states.get(client_id)
                    current_identity = (
                        (
                            current.client_instance_id,
                            current.xr_session_id,
                            current.reference_space_type,
                            current.reference_space_generation,
                        )
                        if current is not None
                        else None
                    )
                    if identity != current_identity:
                        raise ValueError("XR session end identity mismatch")
                    assert current is not None
                    invalidation = self._invalidation_event_locked(
                        client_id=client_id,
                        state=current,
                        reason="session_ended",
                        next_generation=None,
                    )
                    self.states.pop(client_id, None)
                    observers = tuple(self._observers.values())
                else:
                    raise ValueError("unsupported XR session state")
        except Exception as exc:  # noqa: BLE001 - reject malformed telemetry only.
            self._reject(exc)
        if invalidation is not None:
            self._notify_invalidation(invalidation, observers)

    def _handle_reference_space_reset(
        self, client_id: int, payload: dict[str, Any]
    ) -> None:
        invalidation: ControllerPoseInvalidationEvent | None = None
        observers: tuple[_Observer, ...] = ()
        try:
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("core protocol version mismatch")
            previous = payload.get("previous_generation")
            new = payload.get("new_generation")
            if (
                isinstance(previous, bool)
                or not isinstance(previous, int)
                or isinstance(new, bool)
                or not isinstance(new, int)
                or new != previous + 1
            ):
                raise ValueError(
                    "reference-space generation must advance by exactly one"
                )
            with self.lock:
                if self._stopped:
                    raise RuntimeError("controller-pose runtime is stopped")
                state = self.states.get(client_id)
                if state is None:
                    raise ValueError("reference-space reset has no active XR session")
                if payload.get("client_instance_id") != state.client_instance_id:
                    raise ValueError("reference-space reset client identity mismatch")
                if payload.get("xr_session_id") != state.xr_session_id:
                    raise ValueError("reference-space reset session identity mismatch")
                if payload.get("reference_space_type") != state.reference_space_type:
                    raise ValueError("reference-space reset type mismatch")
                if previous != state.reference_space_generation:
                    raise ValueError("reference-space reset generation mismatch")
                invalidation = self._invalidation_event_locked(
                    client_id=client_id,
                    state=state,
                    reason="reference_space_reset",
                    next_generation=new,
                )
                state.reference_space_generation = new
                state.latest = None
                state.last_received_monotonic_ns = 0
                observers = tuple(self._observers.values())
        except Exception as exc:  # noqa: BLE001 - reject malformed telemetry only.
            self._reject(exc)
        if invalidation is not None:
            self._notify_invalidation(invalidation, observers)

    def _handle_pose_sample(self, client_id: int, payload: dict[str, Any]) -> None:
        accepted: ControllerPoseAcceptedEvent | None = None
        observers: tuple[_Observer, ...] = ()
        try:
            sample = parse_controller_pose_sample(
                payload,
                received_wall_time_s=payload.get("server_timestamp"),
                received_monotonic_ns=payload.get("server_monotonic_ns"),
                max_controllers=self.max_controllers,
            )
            with self.lock:
                if self._stopped:
                    raise RuntimeError("controller-pose runtime is stopped")
                state = self.states.get(client_id)
                if state is None:
                    raise ValueError("controller pose has no active XR session")
                if sample.client_instance_id != state.client_instance_id:
                    raise ValueError("controller pose client identity mismatch")
                if sample.xr_session_id != state.xr_session_id:
                    raise ValueError("controller pose session identity mismatch")
                if sample.reference_space_type != state.reference_space_type:
                    raise ValueError("controller pose reference-space type mismatch")
                if (
                    sample.reference_space_generation
                    != state.reference_space_generation
                ):
                    raise ValueError(
                        "controller pose reference-space generation mismatch"
                    )
                if sample.sample_sequence <= state.last_sequence:
                    raise ValueError("controller pose sequence must strictly increase")
                minimum_period_ns = int(1_000_000_000 / self.max_rate_hz)
                if (
                    state.last_received_monotonic_ns
                    and sample.received_monotonic_ns - state.last_received_monotonic_ns
                    < minimum_period_ns
                ):
                    raise ValueError("controller pose admission rate exceeded")
                state.last_sequence = sample.sample_sequence
                state.last_received_monotonic_ns = sample.received_monotonic_ns
                state.latest = sample
                self.accepted_count += 1
                self.last_rejection = None
                accepted = ControllerPoseAcceptedEvent(
                    event_sequence=self._event_sequence_locked(),
                    server_instance_id=self.server_instance_id,
                    client_id=int(client_id),
                    sample=sample,
                    receipt_clock_domain_id=self.receipt_clock_domain_id,
                )
                observers = tuple(self._observers.values())
        except Exception as exc:  # noqa: BLE001 - reject malformed telemetry only.
            self._reject(exc)
        if accepted is not None:
            self._notify_sample(accepted, observers)

    def _handle_disconnect(self, client_id: int) -> None:
        with self.lock:
            if self._stopped:
                return
            state = self.states.pop(client_id, None)
            if state is None:
                return
            invalidation = self._invalidation_event_locked(
                client_id=client_id,
                state=state,
                reason="client_disconnected",
                next_generation=None,
            )
            observers = tuple(self._observers.values())
        self._notify_invalidation(invalidation, observers)

    def latest_sample(self, *, client_id: int) -> ControllerPoseSample | None:
        with self.lock:
            state = self.states.get(client_id)
            return None if state is None else state.latest

    def snapshot_status(self) -> dict[str, Any]:
        with self.lock:
            clients = {
                str(client_id): {
                    "client_instance_id": state.client_instance_id,
                    "xr_session_id": state.xr_session_id,
                    "reference_space_type": state.reference_space_type,
                    "reference_space_generation": state.reference_space_generation,
                    "last_sequence": state.last_sequence,
                    "controller_count": len(state.latest.controllers)
                    if state.latest is not None
                    else 0,
                }
                for client_id, state in self.states.items()
            }
            return {
                "mode": "latest_only",
                "accepted_count": self.accepted_count,
                "rejected_count": self.rejected_count,
                "last_rejection": self.last_rejection,
                "observer_count": len(self._observers),
                "observer_error_count": self.observer_error_count,
                "last_observer_error": self.last_observer_error,
                "receipt_clock_domain_bound": (
                    self.receipt_clock_domain_id is not None
                ),
                "stopped": self._stopped,
                "clients": clients,
            }


def create_runtime(
    server: Any, config: dict[str, Any], stop_event: threading.Event
) -> ControllerPoseTelemetryRuntime:
    return ControllerPoseTelemetryRuntime(server, config, stop_event)


__all__ = [
    "ControllerPoseAcceptedEvent",
    "ControllerPoseActiveSessionCut",
    "ControllerPoseEventCut",
    "ControllerPoseInvalidationEvent",
    "ControllerPoseObserverCloseCut",
    "ControllerPoseSubscription",
    "ControllerPoseTelemetryRuntime",
    "create_runtime",
]
