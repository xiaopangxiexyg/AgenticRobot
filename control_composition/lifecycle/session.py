"""Pure lifecycle contracts for one logical composition session.

This module owns only immutable session identity, activation state, and the
fail-closed transition reducer. It does not own Station operations,
publication barriers, backend readiness, HOLD/cancellation side effects, or
proposal admission.

The v1 hash domains retain their historical ``control_station`` strings so
this ownership migration preserves recorded identities.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_SESSION_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.composition_session_spec.v1"
)
COMPOSITION_SESSION_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.composition_session_state.v1"
)
COMPOSITION_SESSION_TRANSITION_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.composition_session_transition.v1"
)
COMPOSITION_SESSION_REARM_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.composition_session_rearm_spec.v1"
)
COMPOSITION_SESSION_REARM_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.composition_session_rearm_receipt.v1"
)
COMPOSITION_SESSION_PHASES_V1 = (
    "registered",
    "active",
    "held",
    "ended",
    "failed",
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class CompositionSessionOperationSpec:
    """Immutable public contract for one single-use composition session."""

    session_key: str
    composer_source_slot_id: str
    contributor_source_slot_ids: tuple[str, ...]
    composition_spec_hash: str
    authority_gate_spec_hash: str
    execution_binding_hash: str
    capture_boundary_id: str
    pause_behavior: str = "hold_and_cancel"
    end_behavior: str = "hold_cancel_and_end"
    resume_behavior: str = "new_activation_epoch_after_typed_rearm"

    def __post_init__(self) -> None:
        for field_name in (
            "session_key",
            "composer_source_slot_id",
            "composition_spec_hash",
            "authority_gate_spec_hash",
            "execution_binding_hash",
            "capture_boundary_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        contributors = tuple(
            _required(value, field_name="contributor_source_slot_id")
            for value in self.contributor_source_slot_ids
        )
        if not contributors or len(set(contributors)) != len(contributors):
            raise ValueError(
                "contributor_source_slot_ids must be non-empty and unique."
            )
        if self.composer_source_slot_id in contributors:
            raise ValueError("Composer cannot also be a contributor source slot.")
        object.__setattr__(
            self,
            "contributor_source_slot_ids",
            tuple(sorted(contributors)),
        )
        if self.pause_behavior != "hold_and_cancel":
            raise ValueError("Formal composition v1 pause must hold_and_cancel.")
        if self.end_behavior != "hold_cancel_and_end":
            raise ValueError(
                "Formal composition v1 end must hold_cancel_and_end."
            )
        if self.resume_behavior != "new_activation_epoch_after_typed_rearm":
            raise ValueError(
                "Formal composition v1 requires a new activation epoch on resume."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "session_key": self.session_key,
                "composer_source_slot_id": self.composer_source_slot_id,
                "contributor_source_slot_ids": self.contributor_source_slot_ids,
                "composition_spec_hash": self.composition_spec_hash,
                "authority_gate_spec_hash": self.authority_gate_spec_hash,
                "execution_binding_hash": self.execution_binding_hash,
                "capture_boundary_id": self.capture_boundary_id,
                "pause_behavior": self.pause_behavior,
                "end_behavior": self.end_behavior,
                "resume_behavior": self.resume_behavior,
            },
            domain=COMPOSITION_SESSION_SPEC_HASH_DOMAIN_V1,
        )

    @property
    def operation_spec_id(self) -> str:
        return f"composition-session-spec:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionSessionState:
    operation_spec_id: str
    operation_spec_hash: str
    session_epoch: int
    phase: str
    start_intent_id: str | None
    last_source_fact_id: str | None
    reason: str
    semantic_content_hash: str


@dataclass(frozen=True)
class CompositionSessionTransition:
    transition_id: str
    operation_spec_id: str
    operation_spec_hash: str
    session_epoch: int
    transition: str
    reason: str
    source_fact_id: str
    source_fact_hash: str
    state_before_hash: str
    state_after_hash: str
    semantic_content_hash: str


@dataclass(frozen=True)
class CompositionSessionRearmSpec:
    """Immutable request to create a new activation of one logical session."""

    rearm_key: str
    logical_operation_spec_id: str
    logical_operation_spec_hash: str
    prior_session_epoch: int
    prior_state_hash: str
    execution_binding_hash: str
    readiness_observer_binding_id: str
    readiness_timeout_sec: float
    required_readiness_checks: tuple[str, ...] = (
        "activation_input_barrier_reset",
        "composer_authority_epoch_advanced",
        "composer_producer_epoch_advanced",
        "execution_cancellation_complete",
        "execution_gate_ready",
        "measured_state_fresh",
    )

    def __post_init__(self) -> None:
        for field_name in (
            "rearm_key",
            "logical_operation_spec_id",
            "logical_operation_spec_hash",
            "prior_state_hash",
            "execution_binding_hash",
            "readiness_observer_binding_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        epoch = int(self.prior_session_epoch)
        if (
            isinstance(self.prior_session_epoch, bool)
            or epoch != self.prior_session_epoch
            or epoch < 0
        ):
            raise ValueError("prior_session_epoch must be a non-negative integer.")
        object.__setattr__(self, "prior_session_epoch", epoch)
        timeout = float(self.readiness_timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("readiness_timeout_sec must be finite and positive.")
        object.__setattr__(self, "readiness_timeout_sec", timeout)
        checks = tuple(
            sorted(
                _required(value, field_name="required_readiness_check")
                for value in self.required_readiness_checks
            )
        )
        if not checks or len(set(checks)) != len(checks):
            raise ValueError(
                "required_readiness_checks must be non-empty and unique."
            )
        object.__setattr__(self, "required_readiness_checks", checks)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "rearm_key": self.rearm_key,
                "logical_operation_spec_id": self.logical_operation_spec_id,
                "logical_operation_spec_hash": self.logical_operation_spec_hash,
                "prior_session_epoch": self.prior_session_epoch,
                "prior_state_hash": self.prior_state_hash,
                "execution_binding_hash": self.execution_binding_hash,
                "readiness_observer_binding_id": (
                    self.readiness_observer_binding_id
                ),
                "readiness_timeout_sec": self.readiness_timeout_sec,
                "required_readiness_checks": self.required_readiness_checks,
            },
            domain=COMPOSITION_SESSION_REARM_SPEC_HASH_DOMAIN_V1,
        )

    @property
    def rearm_spec_id(self) -> str:
        return f"composition-session-rearm-spec:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionSessionRearmReceipt:
    """Typed evidence that a new activation is safe to publish in one backend."""

    rearm_spec_id: str
    rearm_spec_hash: str
    logical_operation_spec_id: str
    logical_operation_spec_hash: str
    prior_session_epoch: int
    next_session_epoch: int
    prior_composer_producer_epoch: int
    composer_producer_epoch: int
    prior_authority_epoch: int
    authority_epoch: int
    prior_execution_epoch: int
    execution_epoch: int
    cancellation_receipt_id: str
    measurement_id: str
    measurement_hash: str
    producer_activation_id: str
    authority_event_ids: tuple[str, ...]
    ready_gate_event_id: str
    activation_input_barrier_time: float
    completed_at: float
    clock_domain_id: str
    satisfied_readiness_checks: tuple[str, ...]
    status: str = "composition_session_rearmed"

    def __post_init__(self) -> None:
        for field_name in (
            "rearm_spec_id",
            "rearm_spec_hash",
            "logical_operation_spec_id",
            "logical_operation_spec_hash",
            "cancellation_receipt_id",
            "measurement_id",
            "measurement_hash",
            "producer_activation_id",
            "ready_gate_event_id",
            "clock_domain_id",
            "status",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        integer_fields = (
            "prior_session_epoch",
            "next_session_epoch",
            "prior_composer_producer_epoch",
            "composer_producer_epoch",
            "prior_authority_epoch",
            "authority_epoch",
            "prior_execution_epoch",
            "execution_epoch",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            normalized = int(value)
            if isinstance(value, bool) or normalized != value or normalized < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
            object.__setattr__(self, field_name, normalized)
        if self.next_session_epoch != self.prior_session_epoch + 1:
            raise ValueError("Rearm must create the next session activation epoch.")
        if (
            self.composer_producer_epoch <= self.prior_composer_producer_epoch
            or self.authority_epoch <= self.prior_authority_epoch
            or self.execution_epoch <= self.prior_execution_epoch
        ):
            raise ValueError(
                "Rearm producer, authority, and execution epochs must advance."
            )
        authority_events = tuple(
            _required(value, field_name="authority_event_id")
            for value in self.authority_event_ids
        )
        if not authority_events or len(set(authority_events)) != len(
            authority_events
        ):
            raise ValueError("authority_event_ids must be non-empty and unique.")
        object.__setattr__(self, "authority_event_ids", authority_events)
        checks = tuple(
            sorted(
                _required(value, field_name="satisfied_readiness_check")
                for value in self.satisfied_readiness_checks
            )
        )
        if not checks or len(set(checks)) != len(checks):
            raise ValueError(
                "satisfied_readiness_checks must be non-empty and unique."
            )
        object.__setattr__(self, "satisfied_readiness_checks", checks)
        for field_name in ("activation_input_barrier_time", "completed_at"):
            normalized = float(getattr(self, field_name))
            if not math.isfinite(normalized):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, normalized)
        if self.activation_input_barrier_time > self.completed_at:
            raise ValueError("Input barrier cannot begin after rearm completion.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in (
                    "rearm_spec_id",
                    "rearm_spec_hash",
                    "logical_operation_spec_id",
                    "logical_operation_spec_hash",
                    "prior_session_epoch",
                    "next_session_epoch",
                    "prior_composer_producer_epoch",
                    "composer_producer_epoch",
                    "prior_authority_epoch",
                    "authority_epoch",
                    "prior_execution_epoch",
                    "execution_epoch",
                    "cancellation_receipt_id",
                    "measurement_id",
                    "measurement_hash",
                    "producer_activation_id",
                    "authority_event_ids",
                    "ready_gate_event_id",
                    "activation_input_barrier_time",
                    "completed_at",
                    "clock_domain_id",
                    "satisfied_readiness_checks",
                    "status",
                )
            },
            domain=COMPOSITION_SESSION_REARM_RECEIPT_HASH_DOMAIN_V1,
        )

    @property
    def receipt_id(self) -> str:
        return f"composition-session-rearm-receipt:{self.semantic_content_hash}"


def validate_composition_session_rearm(
    spec: CompositionSessionRearmSpec,
    receipt: CompositionSessionRearmReceipt,
    *,
    logical_spec: CompositionSessionOperationSpec,
    held_state: CompositionSessionState,
) -> None:
    """Validate an exact held activation and its backend readiness evidence."""

    if not isinstance(spec, CompositionSessionRearmSpec):
        raise TypeError("spec must be a CompositionSessionRearmSpec.")
    if not isinstance(receipt, CompositionSessionRearmReceipt):
        raise TypeError("receipt must be a CompositionSessionRearmReceipt.")
    if not isinstance(logical_spec, CompositionSessionOperationSpec):
        raise TypeError("logical_spec must be a CompositionSessionOperationSpec.")
    if not isinstance(held_state, CompositionSessionState):
        raise TypeError("held_state must be a CompositionSessionState.")
    if held_state.phase != "held":
        raise ValueError("Composition rearm requires one held activation.")
    expected = (
        logical_spec.operation_spec_id,
        logical_spec.semantic_content_hash,
        held_state.session_epoch,
        held_state.semantic_content_hash,
        logical_spec.execution_binding_hash,
    )
    observed = (
        spec.logical_operation_spec_id,
        spec.logical_operation_spec_hash,
        spec.prior_session_epoch,
        spec.prior_state_hash,
        spec.execution_binding_hash,
    )
    if observed != expected:
        raise ValueError("Composition rearm spec does not bind the held activation.")
    receipt_binding = (
        receipt.rearm_spec_id,
        receipt.rearm_spec_hash,
        receipt.logical_operation_spec_id,
        receipt.logical_operation_spec_hash,
        receipt.prior_session_epoch,
    )
    expected_receipt_binding = (
        spec.rearm_spec_id,
        spec.semantic_content_hash,
        logical_spec.operation_spec_id,
        logical_spec.semantic_content_hash,
        held_state.session_epoch,
    )
    if receipt_binding != expected_receipt_binding:
        raise ValueError("Composition rearm receipt binding mismatch.")
    missing = set(spec.required_readiness_checks) - set(
        receipt.satisfied_readiness_checks
    )
    if missing:
        raise ValueError(
            "Composition rearm receipt misses readiness checks: "
            + ",".join(sorted(missing))
        )


def initial_composition_session_state(
    spec: CompositionSessionOperationSpec,
    *,
    session_epoch: int,
) -> CompositionSessionState:
    if not isinstance(spec, CompositionSessionOperationSpec):
        raise TypeError("spec must be a CompositionSessionOperationSpec.")
    epoch = int(session_epoch)
    if isinstance(session_epoch, bool) or epoch != session_epoch or epoch < 0:
        raise ValueError("session_epoch must be a non-negative integer.")
    payload = {
        "operation_spec_id": spec.operation_spec_id,
        "operation_spec_hash": spec.semantic_content_hash,
        "session_epoch": epoch,
        "phase": "registered",
        "start_intent_id": None,
        "last_source_fact_id": None,
        "reason": "registered",
    }
    return CompositionSessionState(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=COMPOSITION_SESSION_STATE_HASH_DOMAIN_V1,
        ),
    )


def advance_composition_session(
    state: CompositionSessionState,
    *,
    transition: str,
    reason: str,
    source_fact_id: str,
    source_fact_hash: str,
    start_intent_id: str | None = None,
) -> tuple[CompositionSessionState, CompositionSessionTransition]:
    """Apply one fail-closed activation lifecycle transition."""

    if not isinstance(state, CompositionSessionState):
        raise TypeError("state must be a CompositionSessionState.")
    event = _required(transition, field_name="transition")
    normalized_reason = _required(reason, field_name="reason")
    source_id = _required(source_fact_id, field_name="source_fact_id")
    source_hash = _required(source_fact_hash, field_name="source_fact_hash")
    allowed = {
        ("registered", "start"): "active",
        ("registered", "start_failed"): "failed",
        ("active", "hold"): "held",
        ("active", "stop"): "ended",
        ("active", "fault"): "failed",
        ("held", "supersede"): "ended",
        ("held", "stop"): "ended",
        ("held", "fault"): "failed",
    }
    try:
        next_phase = allowed[(state.phase, event)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported composition session transition {state.phase!r}->{event!r}."
        ) from exc
    normalized_start_intent = state.start_intent_id
    if event == "start":
        normalized_start_intent = _required(
            start_intent_id,
            field_name="start_intent_id",
        )
    elif start_intent_id is not None and str(start_intent_id) != str(
        state.start_intent_id
    ):
        raise ValueError("Composition session start intent is immutable.")
    next_payload = {
        "operation_spec_id": state.operation_spec_id,
        "operation_spec_hash": state.operation_spec_hash,
        "session_epoch": state.session_epoch,
        "phase": next_phase,
        "start_intent_id": normalized_start_intent,
        "last_source_fact_id": source_id,
        "reason": normalized_reason,
    }
    next_state = CompositionSessionState(
        **next_payload,
        semantic_content_hash=canonical_content_hash(
            next_payload,
            domain=COMPOSITION_SESSION_STATE_HASH_DOMAIN_V1,
        ),
    )
    transition_payload = {
        "operation_spec_id": state.operation_spec_id,
        "operation_spec_hash": state.operation_spec_hash,
        "session_epoch": state.session_epoch,
        "transition": event,
        "reason": normalized_reason,
        "source_fact_id": source_id,
        "source_fact_hash": source_hash,
        "state_before_hash": state.semantic_content_hash,
        "state_after_hash": next_state.semantic_content_hash,
    }
    digest = canonical_content_hash(
        transition_payload,
        domain=COMPOSITION_SESSION_TRANSITION_HASH_DOMAIN_V1,
    )
    return next_state, CompositionSessionTransition(
        transition_id=f"composition-session-transition:{digest}",
        semantic_content_hash=digest,
        **transition_payload,
    )


class CompositionSessionRegistry:
    """Backend-neutral owner of session identity and activation state.

    The registry has no publication gate or Station operation knowledge. A
    control-plane adapter may use its protected state only to complete an
    already-validated atomic boundary such as typed rearm.
    """

    def __init__(
        self,
        *,
        composer_source_slot_id: str,
        contributor_source_slot_ids: tuple[str, ...],
        composition_spec_hash: str,
        authority_gate_spec_hash: str,
        execution_binding_hash: str,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        self.composer_source_slot_id = _required(
            composer_source_slot_id,
            field_name="composer_source_slot_id",
        )
        contributors = tuple(
            _required(value, field_name="contributor_source_slot_id")
            for value in contributor_source_slot_ids
        )
        if not contributors or len(set(contributors)) != len(contributors):
            raise ValueError(
                "contributor_source_slot_ids must be non-empty and unique."
            )
        if self.composer_source_slot_id in contributors:
            raise ValueError("Composer cannot also be a contributor source slot.")
        self.contributor_source_slot_ids = tuple(sorted(contributors))
        self.composition_spec_hash = _required(
            composition_spec_hash,
            field_name="composition_spec_hash",
        )
        self.authority_gate_spec_hash = _required(
            authority_gate_spec_hash,
            field_name="authority_gate_spec_hash",
        )
        self.execution_binding_hash = _required(
            execution_binding_hash,
            field_name="execution_binding_hash",
        )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.clock = clock
        self.clock_domain_id = _required(
            clock_domain_id,
            field_name="clock_domain_id",
        )
        self._spec_by_id: dict[str, CompositionSessionOperationSpec] = {}
        self._state_by_spec_id: dict[str, CompositionSessionState] = {}
        self._rearm_spec_by_id: dict[str, CompositionSessionRearmSpec] = {}
        self._current_spec_id: str | None = None
        self._next_session_epoch = 0
        self._consumed_spec_ids: set[str] = set()
        self._consumed_rearm_spec_ids: set[str] = set()
        self._spec_observer: (
            Callable[[CompositionSessionOperationSpec], object] | None
        ) = None
        self._transition_observer: Callable[..., object] | None = None
        self._rearm_spec_observer: (
            Callable[[CompositionSessionRearmSpec], object] | None
        ) = None
        self._rearm_receipt_observer: (
            Callable[[CompositionSessionRearmReceipt], object] | None
        ) = None
        self._lock = threading.RLock()

    @property
    def current_state(self) -> CompositionSessionState | None:
        with self._lock:
            if self._current_spec_id is None:
                return None
            return self._state_by_spec_id[self._current_spec_id]

    def bind_session_trace_observers(
        self,
        *,
        spec_observer: Callable[[CompositionSessionOperationSpec], object],
        transition_observer: Callable[..., object],
        rearm_spec_observer: (
            Callable[[CompositionSessionRearmSpec], object] | None
        ) = None,
        rearm_receipt_observer: (
            Callable[[CompositionSessionRearmReceipt], object] | None
        ) = None,
    ) -> None:
        if not callable(spec_observer) or not callable(transition_observer):
            raise TypeError("Session trace observers must be callable.")
        if rearm_spec_observer is not None and not callable(rearm_spec_observer):
            raise TypeError("rearm_spec_observer must be callable.")
        if (
            rearm_receipt_observer is not None
            and not callable(rearm_receipt_observer)
        ):
            raise TypeError("rearm_receipt_observer must be callable.")
        with self._lock:
            if self._spec_observer is not None or self._spec_by_id:
                raise RuntimeError(
                    "Session trace observers must bind before registration."
                )
            self._spec_observer = spec_observer
            self._transition_observer = transition_observer
            self._rearm_spec_observer = rearm_spec_observer
            self._rearm_receipt_observer = rearm_receipt_observer

    def register(self, spec: CompositionSessionOperationSpec) -> str:
        if not isinstance(spec, CompositionSessionOperationSpec):
            raise TypeError("spec must be a CompositionSessionOperationSpec.")
        if spec.composer_source_slot_id != self.composer_source_slot_id:
            raise ValueError("Composition session composer slot mismatch.")
        if (
            tuple(sorted(spec.contributor_source_slot_ids))
            != self.contributor_source_slot_ids
        ):
            raise ValueError("Composition session contributor slots mismatch.")
        if (
            spec.composition_spec_hash != self.composition_spec_hash
            or spec.authority_gate_spec_hash != self.authority_gate_spec_hash
            or spec.execution_binding_hash != self.execution_binding_hash
        ):
            raise ValueError("Composition session contract binding mismatch.")
        with self._lock:
            existing = self._spec_by_id.get(spec.operation_spec_id)
            if existing is not None:
                if existing != spec:
                    raise RuntimeError(
                        "Composition session spec identity collision."
                    )
                return spec.operation_spec_id
            current = self.current_state
            if current is not None and current.phase not in {"ended", "failed"}:
                raise RuntimeError("Another composition session is not terminal.")
            state = initial_composition_session_state(
                spec,
                session_epoch=self._next_session_epoch,
            )
            self._next_session_epoch += 1
            self._spec_by_id[spec.operation_spec_id] = spec
            self._state_by_spec_id[spec.operation_spec_id] = state
            self._current_spec_id = spec.operation_spec_id
            observer = self._spec_observer
        if observer is not None:
            observer(spec)
        return spec.operation_spec_id

    def require_current_spec(
        self,
        operation_spec_id: str | None,
    ) -> CompositionSessionOperationSpec:
        spec_id = str(operation_spec_id or "").strip()
        with self._lock:
            if not spec_id or spec_id != self._current_spec_id:
                raise ValueError("Unknown composition session operation spec.")
            return self._spec_by_id[spec_id]

    def require_registered_spec(
        self,
        operation_spec_id: str | None,
    ) -> CompositionSessionOperationSpec:
        spec = self.require_current_spec(operation_spec_id)
        with self._lock:
            if spec.operation_spec_id in self._consumed_spec_ids:
                raise RuntimeError("Composition session spec is single-use.")
            state = self._state_by_spec_id[spec.operation_spec_id]
            if state.phase != "registered":
                raise RuntimeError("Composition session is not registered.")
        return spec

    def register_rearm(
        self,
        spec: CompositionSessionRearmSpec,
    ) -> str:
        if not isinstance(spec, CompositionSessionRearmSpec):
            raise TypeError("spec must be a CompositionSessionRearmSpec.")
        with self._lock:
            if self._current_spec_id is None:
                raise RuntimeError("No current composition session.")
            logical_spec = self._spec_by_id[self._current_spec_id]
            state = self._state_by_spec_id[self._current_spec_id]
            validate_binding = (
                spec.logical_operation_spec_id,
                spec.logical_operation_spec_hash,
                spec.prior_session_epoch,
                spec.prior_state_hash,
                spec.execution_binding_hash,
            )
            expected_binding = (
                logical_spec.operation_spec_id,
                logical_spec.semantic_content_hash,
                state.session_epoch,
                state.semantic_content_hash,
                logical_spec.execution_binding_hash,
            )
            if state.phase != "held" or validate_binding != expected_binding:
                raise ValueError(
                    "Composition rearm spec does not bind the current held activation."
                )
            existing = self._rearm_spec_by_id.get(spec.rearm_spec_id)
            if existing is not None:
                if existing != spec:
                    raise RuntimeError("Composition rearm spec identity collision.")
                return spec.rearm_spec_id
            self._rearm_spec_by_id[spec.rearm_spec_id] = spec
            observer = self._rearm_spec_observer
        if observer is not None:
            observer(spec)
        return spec.rearm_spec_id

    def require_rearm_spec(
        self,
        rearm_spec_id: str | None,
    ) -> CompositionSessionRearmSpec:
        normalized = str(rearm_spec_id or "").strip()
        with self._lock:
            spec = self._rearm_spec_by_id.get(normalized)
            if spec is None:
                raise ValueError("Unknown composition session rearm spec.")
            if normalized in self._consumed_rearm_spec_ids:
                raise RuntimeError("Composition session rearm spec is single-use.")
            if self._current_spec_id is None:
                raise RuntimeError("No current composition session.")
            logical_spec = self._spec_by_id[self._current_spec_id]
            state = self._state_by_spec_id[self._current_spec_id]
        validate_binding = (
            spec.logical_operation_spec_id,
            spec.logical_operation_spec_hash,
            spec.prior_session_epoch,
            spec.prior_state_hash,
            spec.execution_binding_hash,
        )
        expected_binding = (
            logical_spec.operation_spec_id,
            logical_spec.semantic_content_hash,
            state.session_epoch,
            state.semantic_content_hash,
            logical_spec.execution_binding_hash,
        )
        if state.phase != "held" or validate_binding != expected_binding:
            raise RuntimeError(
                "Composition rearm spec is stale for the current activation."
            )
        return spec

    def advance(
        self,
        *,
        transition: str,
        reason: str,
        source_fact_id: str,
        source_fact_hash: str,
        start_intent_id: str | None = None,
    ) -> CompositionSessionTransition:
        with self._lock:
            if self._current_spec_id is None:
                raise RuntimeError("No current composition session.")
            state = self._state_by_spec_id[self._current_spec_id]
            next_state, event = advance_composition_session(
                state,
                transition=transition,
                reason=reason,
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                start_intent_id=start_intent_id,
            )
            self._state_by_spec_id[self._current_spec_id] = next_state
            observer = self._transition_observer
        if observer is not None:
            observer(
                event,
                occurred_at=float(self.clock()),
                clock_domain_id=self.clock_domain_id,
            )
        return event


__all__ = [
    "COMPOSITION_SESSION_PHASES_V1",
    "COMPOSITION_SESSION_REARM_RECEIPT_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_REARM_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_STATE_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_TRANSITION_HASH_DOMAIN_V1",
    "CompositionSessionOperationSpec",
    "CompositionSessionRegistry",
    "CompositionSessionRearmReceipt",
    "CompositionSessionRearmSpec",
    "CompositionSessionState",
    "CompositionSessionTransition",
    "advance_composition_session",
    "initial_composition_session_state",
    "validate_composition_session_rearm",
]
