"""Station-owned completion port for composition-session operations.

Lower application bindings may need time to establish an activation, source
producer epochs, readiness receipts, and an execution-gate boundary.  Those
bindings expose evidence; they do not manufacture a Station result.  This
module binds that evidence to one already-admitted pending Station operation
and completes it through :class:`ControlStationRuntime`, which remains the
only owner of result admission and Station state reduction.

The port is backend-, topology-, source-, and action-representation neutral.
It owns no composition algorithm, source worker, safety decision, target,
reference, IK, dispatch, or episode state.
"""

from __future__ import annotations

import math
import threading
from typing import Protocol, runtime_checkable

from .contracts import StationOperationResult, StationOperationResultDraft
from .runtime import ControlStationRuntime
from .state_machine import (
    COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1,
    ControlStationOperationTransition,
    ControlStationTransition,
)


_COMPOSITION_SESSION_OPERATION_KINDS = frozenset(
    {
        "start_composition_session",
        "rearm_composition_session",
        "stop_composition_session",
    }
)

_SUCCESS_STATUS_BY_OPERATION_KIND = {
    "start_composition_session": COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
    "rearm_composition_session": COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1,
    "stop_composition_session": COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1,
}


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _evidence_refs(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("required_evidence_refs must be a sequence of strings.")
    try:
        refs = tuple(
            _required(value, field_name="required_evidence_ref")
            for value in values  # type: ignore[union-attr]
        )
    except TypeError as exc:
        raise TypeError(
            "required_evidence_refs must be a sequence of strings."
        ) from exc
    if not refs:
        raise ValueError("Successful session completion requires evidence.")
    return tuple(dict.fromkeys(refs))


@runtime_checkable
class CompositionSessionOperationEvidence(Protocol):
    """Minimum lower-evidence shape accepted by the Station completion port."""

    operation_kind: str

    @property
    def required_evidence_refs(self) -> tuple[str, ...]: ...

    @property
    def causal_ready_at(self) -> float: ...

    @property
    def causal_clock_domain_id(self) -> str: ...


class StationCompositionSessionCompletionRejected(RuntimeError):
    """A result was admitted but did not complete its pending operation."""

    def __init__(self, transition: ControlStationOperationTransition) -> None:
        self.transition = transition
        super().__init__(transition.status)


class StationCompositionSessionOperationPort:
    """Complete exactly one admitted composition-session operation.

    A port instance is single-use and bound to one request transition.  The
    successful callable surface is intentionally compatible with concrete
    session ``result_provider(evidence)`` callbacks.  ``complete_failure`` is
    the application-level cleanup seam when lower activation construction
    fails before successful evidence exists.
    """

    def __init__(
        self,
        *,
        runtime: ControlStationRuntime,
        request_transition: ControlStationTransition,
    ) -> None:
        if not isinstance(runtime, ControlStationRuntime):
            raise TypeError("runtime must be a ControlStationRuntime.")
        if not isinstance(request_transition, ControlStationTransition):
            raise TypeError(
                "request_transition must be a ControlStationTransition."
            )
        intent = request_transition.intent
        if (
            not request_transition.receipt.accepted
            or request_transition.receipt.status != "operation_requested"
            or intent.intent_kind not in _COMPOSITION_SESSION_OPERATION_KINDS
        ):
            raise ValueError(
                "Port requires an admitted pending composition-session operation."
            )
        if intent.operation_spec_id is None:
            raise ValueError("Composition-session operation has no spec binding.")
        if (
            intent.runtime_session_id != runtime.state.runtime_session_id
            or intent.control_domain_id != runtime.state.control_domain_id
        ):
            raise ValueError("Request transition belongs to another Station runtime.")
        self.runtime = runtime
        self.request_transition = request_transition
        self._completion_transition: ControlStationOperationTransition | None = None
        self._lock = threading.RLock()

    @property
    def completion_transition(self) -> ControlStationOperationTransition | None:
        with self._lock:
            return self._completion_transition

    @property
    def result(self) -> StationOperationResult | None:
        completion = self.completion_transition
        return None if completion is None else completion.result

    @property
    def completed(self) -> bool:
        completion = self.completion_transition
        return completion is not None and completion.accepted

    def __call__(
        self,
        evidence: CompositionSessionOperationEvidence,
    ) -> StationOperationResult:
        return self.complete_success(evidence)

    def complete_success(
        self,
        evidence: CompositionSessionOperationEvidence,
    ) -> StationOperationResult:
        """Acknowledge exact lower readiness and return the admitted result."""

        if not isinstance(evidence, CompositionSessionOperationEvidence):
            raise TypeError(
                "evidence must implement CompositionSessionOperationEvidence."
            )
        intent = self.request_transition.intent
        operation_kind = _required(
            getattr(evidence, "operation_kind", ""),
            field_name="evidence.operation_kind",
        )
        if operation_kind != intent.intent_kind:
            raise ValueError("Evidence belongs to another operation kind.")
        causal_clock_domain_id = _required(
            getattr(evidence, "causal_clock_domain_id", ""),
            field_name="evidence.causal_clock_domain_id",
        )
        if causal_clock_domain_id != self.runtime.clock_domain_id:
            raise ValueError(
                "Cross-clock completion requires an explicit clock-conversion receipt."
            )
        causal_ready_at = float(getattr(evidence, "causal_ready_at"))
        if not math.isfinite(causal_ready_at):
            raise ValueError("evidence.causal_ready_at must be finite.")
        refs = _evidence_refs(getattr(evidence, "required_evidence_refs"))
        return self._complete(
            succeeded=True,
            status=_SUCCESS_STATUS_BY_OPERATION_KIND[operation_kind],
            evidence_refs=refs,
            not_before=causal_ready_at,
        )

    def complete_failure(
        self,
        *,
        status: str,
        evidence_refs: tuple[str, ...] = (),
    ) -> StationOperationResult:
        """Resolve the pending request after lower preparation failed.

        Failure evidence is optional because an exception can occur before a
        lower receipt exists.  The source intent/spec identities are still
        carried by the Station result itself and by the added spec reference.
        """

        normalized_status = _required(status, field_name="status")
        refs = tuple(
            dict.fromkeys(
                _required(value, field_name="evidence_ref")
                for value in evidence_refs
            )
        )
        return self._complete(
            succeeded=False,
            status=normalized_status,
            evidence_refs=refs,
            not_before=None,
        )

    def _complete(
        self,
        *,
        succeeded: bool,
        status: str,
        evidence_refs: tuple[str, ...],
        not_before: float | None,
    ) -> StationOperationResult:
        intent = self.request_transition.intent
        assert intent.operation_spec_id is not None
        refs = tuple(dict.fromkeys((*evidence_refs, intent.operation_spec_id)))
        with self._lock:
            if self._completion_transition is not None:
                raise RuntimeError("Composition-session operation was already completed.")

            def finalize() -> StationOperationResultDraft:
                completed_at = float(self.runtime.clock())
                if not math.isfinite(completed_at):
                    raise ValueError("Station completion clock must be finite.")
                if not_before is not None and completed_at < not_before:
                    raise ValueError(
                        "Station completion predates lower readiness evidence."
                    )
                return StationOperationResultDraft(
                    source_intent_id=intent.intent_id,
                    operation_kind=intent.intent_kind,
                    subject_id=str(intent.subject_id),
                    succeeded=succeeded,
                    completed_at=completed_at,
                    completed_clock_domain_id=self.runtime.clock_domain_id,
                    status=status,
                    evidence_refs=refs,
                )

            completion = self.runtime.finalize_pending_operation(
                self.request_transition,
                finalize,
            )
            self._completion_transition = completion
        if not completion.accepted:
            raise StationCompositionSessionCompletionRejected(completion)
        return completion.result


__all__ = [
    "CompositionSessionOperationEvidence",
    "StationCompositionSessionCompletionRejected",
    "StationCompositionSessionOperationPort",
]
