"""Backend-neutral exclusive Human/Policy source-switch transaction.

The switch is an application lifecycle transaction, not a second authority or
dispatch path.  A concrete adapter must establish lower HOLD, select exactly
one source, create a fresh activation, and only then report ``ready``.  This
module validates those already-owned effects; it performs no device IO, model
inference, composition, safety, target resolution, dispatch, or trace
admission.  An optional evidence observer is passive and cannot change the
transaction result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze, deep_thaw


EXCLUSIVE_HITL_POLICY_TO_HUMAN = "policy_to_human"
EXCLUSIVE_HITL_HUMAN_TO_POLICY = "human_to_policy"
EXCLUSIVE_HITL_SWITCH_DIRECTIONS = (
    EXCLUSIVE_HITL_POLICY_TO_HUMAN,
    EXCLUSIVE_HITL_HUMAN_TO_POLICY,
)
EXCLUSIVE_HITL_SWITCH_STATUSES = ("ready", "already_active", "failed")
EXCLUSIVE_HITL_SWITCH_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.exclusive_hitl_switch.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _unique_evidence(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_required(value, name="evidence_ref") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("evidence_refs cannot contain duplicates.")
    return normalized


@dataclass(frozen=True)
class ExclusiveHITLSwitchEvidence:
    """Replayable result of one explicit policy/human switch transaction.

    ``home_capture`` contains the actual session-local Human/robot anchors when
    the direction is policy-to-human.  It is core replay evidence, not a debug
    attachment.  The record is observational and never authorizes motion.
    """

    switch_id: str
    trigger_intent_id: str
    trigger_intent_hash: str
    direction: str
    previous_source: str
    target_source: str
    requested_at: float
    completed_at: float
    clock_domain_id: str
    succeeded: bool
    status: str
    hold_evidence_refs: tuple[str, ...] = ()
    home_capture_id: str | None = None
    home_capture_hash: str | None = None
    home_capture: Mapping[str, object] | None = None
    source_selection_record_id: str | None = None
    source_selection_record_hash: str | None = None
    activation_id: str | None = None
    activation_hash: str | None = None
    target_emission_id: str | None = None
    target_emission_hash: str | None = None
    target_id: str | None = None
    target_hash: str | None = None
    target_admission_receipt_id: str | None = None
    target_completion_receipt_id: str | None = None
    command_id: str | None = None
    dispatch_receipt_id: str | None = None
    source_reset_performed: bool = False
    evidence_refs: tuple[str, ...] = ()
    error: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "switch_id",
            "trigger_intent_id",
            "trigger_intent_hash",
            "previous_source",
            "target_source",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.direction not in EXCLUSIVE_HITL_SWITCH_DIRECTIONS:
            raise ValueError("Unsupported exclusive HITL switch direction.")
        if self.status not in EXCLUSIVE_HITL_SWITCH_STATUSES:
            raise ValueError("Unsupported exclusive HITL switch status.")
        expected_sources = (
            ("policy", "human")
            if self.direction == EXCLUSIVE_HITL_POLICY_TO_HUMAN
            else ("human", "policy")
        )
        if self.status == "already_active":
            if (
                self.previous_source != expected_sources[1]
                or self.target_source != expected_sources[1]
            ):
                raise ValueError(
                    "An already-active switch must report the requested target "
                    "as both the previous and target source."
                )
        elif (self.previous_source, self.target_source) != expected_sources:
            raise ValueError("Switch direction and source roles disagree.")
        requested_at = _finite(self.requested_at, name="requested_at")
        completed_at = _finite(self.completed_at, name="completed_at")
        if completed_at < requested_at:
            raise ValueError("completed_at cannot predate requested_at.")
        object.__setattr__(self, "requested_at", requested_at)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "succeeded", bool(self.succeeded))
        object.__setattr__(
            self,
            "source_reset_performed",
            bool(self.source_reset_performed),
        )
        object.__setattr__(
            self,
            "hold_evidence_refs",
            _unique_evidence(tuple(self.hold_evidence_refs)),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _unique_evidence(tuple(self.evidence_refs)),
        )
        for name in (
            "home_capture_id",
            "home_capture_hash",
            "source_selection_record_id",
            "source_selection_record_hash",
            "activation_id",
            "activation_hash",
            "target_emission_id",
            "target_emission_hash",
            "target_id",
            "target_hash",
            "target_admission_receipt_id",
            "target_completion_receipt_id",
            "command_id",
            "dispatch_receipt_id",
            "error",
        ):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        home = (
            None
            if self.home_capture is None
            else deep_freeze(dict(self.home_capture))
        )
        object.__setattr__(
            self,
            "home_capture",
            home,
        )
        if (self.home_capture_id is None) != (self.home_capture_hash is None):
            raise ValueError("Home capture ID/hash must appear together.")
        if (self.source_selection_record_id is None) != (
            self.source_selection_record_hash is None
        ):
            raise ValueError("Source-selection record ID/hash must appear together.")
        if (self.activation_id is None) != (self.activation_hash is None):
            raise ValueError("Activation ID/hash must appear together.")
        if (self.target_emission_id is None) != (self.target_emission_hash is None):
            raise ValueError("Target-emission ID/hash must appear together.")
        if (self.target_id is None) != (self.target_hash is None):
            raise ValueError("Target ID/hash must appear together.")
        if self.succeeded != (self.status in {"ready", "already_active"}):
            raise ValueError("Switch success flag and terminal status disagree.")
        if self.status == "ready":
            if not self.hold_evidence_refs:
                raise ValueError("A READY switch requires lower HOLD evidence.")
            if self.source_selection_record_id is None:
                raise ValueError("A READY switch requires source-selection evidence.")
            if self.activation_id is None or self.target_emission_id is None:
                raise ValueError("A READY switch requires fresh activation/emission.")
            if any(
                value is None
                for value in (
                    self.target_id,
                    self.target_admission_receipt_id,
                    self.target_completion_receipt_id,
                    self.command_id,
                    self.dispatch_receipt_id,
                )
            ):
                raise ValueError(
                    "A READY switch requires admitted lower-target and dispatch evidence."
                )
            if self.direction == EXCLUSIVE_HITL_POLICY_TO_HUMAN:
                if self.home_capture_id is None or home is None:
                    raise ValueError("Human takeover requires Runtime Home evidence.")
            elif not self.source_reset_performed:
                raise ValueError("Policy resume requires policy-source reset evidence.")
        if self.status == "failed" and self.error is None:
            raise ValueError("A failed switch requires an error.")

        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXCLUSIVE_HITL_SWITCH_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def authoritative(self) -> bool:
        return False

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "switch_id": self.switch_id,
            "trigger_intent_id": self.trigger_intent_id,
            "trigger_intent_hash": self.trigger_intent_hash,
            "direction": self.direction,
            "previous_source": self.previous_source,
            "target_source": self.target_source,
            "requested_at": self.requested_at,
            "completed_at": self.completed_at,
            "clock_domain_id": self.clock_domain_id,
            "succeeded": self.succeeded,
            "status": self.status,
            "hold_evidence_refs": self.hold_evidence_refs,
            "home_capture_id": self.home_capture_id,
            "home_capture_hash": self.home_capture_hash,
            "home_capture": (
                None
                if self.home_capture is None
                else deep_thaw(self.home_capture)
            ),
            "source_selection_record_id": self.source_selection_record_id,
            "source_selection_record_hash": self.source_selection_record_hash,
            "activation_id": self.activation_id,
            "activation_hash": self.activation_hash,
            "target_emission_id": self.target_emission_id,
            "target_emission_hash": self.target_emission_hash,
            "source_reset_performed": self.source_reset_performed,
            "evidence_refs": self.evidence_refs,
            "error": self.error,
            "authoritative": False,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class ExclusiveHITLTransitionObservation:
    """Backend evidence observed after a HOLD-bounded source transition."""

    target_source: str
    completed_at: float
    hold_evidence_refs: tuple[str, ...]
    source_selection_record_id: str
    source_selection_record_hash: str
    activation_id: str
    activation_hash: str
    target_emission_id: str
    target_emission_hash: str
    target_id: str
    target_hash: str
    target_admission_receipt_id: str
    target_completion_receipt_id: str
    command_id: str
    dispatch_receipt_id: str
    source_reset_performed: bool
    home_capture_id: str | None = None
    home_capture_hash: str | None = None
    home_capture: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        target = _required(self.target_source, name="target_source")
        if target not in {"policy", "human"}:
            raise ValueError("target_source must be policy or human.")
        object.__setattr__(self, "target_source", target)
        object.__setattr__(
            self,
            "completed_at",
            _finite(self.completed_at, name="completed_at"),
        )
        object.__setattr__(
            self,
            "hold_evidence_refs",
            _unique_evidence(tuple(self.hold_evidence_refs)),
        )
        if not self.hold_evidence_refs:
            raise ValueError("Transition observation requires lower HOLD evidence.")
        for name in (
            "source_selection_record_id",
            "source_selection_record_hash",
            "activation_id",
            "activation_hash",
            "target_emission_id",
            "target_emission_hash",
            "target_id",
            "target_hash",
            "target_admission_receipt_id",
            "target_completion_receipt_id",
            "command_id",
            "dispatch_receipt_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_reset_performed",
            bool(self.source_reset_performed),
        )
        home_id = _optional(self.home_capture_id)
        home_hash = _optional(self.home_capture_hash)
        if (home_id is None) != (home_hash is None):
            raise ValueError("Home capture ID/hash must appear together.")
        home = (
            None
            if self.home_capture is None
            else deep_freeze(dict(self.home_capture))
        )
        if target == "human" and (home_id is None or home is None):
            raise ValueError("Human transition requires Runtime Home evidence.")
        if target == "policy" and not self.source_reset_performed:
            raise ValueError("Policy transition requires source-reset evidence.")
        object.__setattr__(self, "home_capture_id", home_id)
        object.__setattr__(self, "home_capture_hash", home_hash)
        object.__setattr__(
            self,
            "home_capture",
            home,
        )


@runtime_checkable
class ExclusiveHITLTransitionPort(Protocol):
    """Concrete application effects required by the shared transaction.

    Sim, real, and other application bindings implement this small port while
    retaining ownership of their own lower HOLD, Runtime Home capture,
    composition-source selection, activation, and source reset operations.
    """

    def transition(
        self,
        selected_source: str,
    ) -> tuple[Mapping[str, object], ExclusiveHITLTransitionObservation]: ...

    def hold_after_failure(self) -> None: ...

    def failure_hold_evidence(self) -> tuple[str, ...]: ...


def run_exclusive_hitl_switch_transaction(
    *,
    trigger_intent_id: str,
    trigger_intent_hash: str,
    direction: str,
    current_source: str,
    requested_at: float,
    clock_domain_id: str,
    transition: Callable[
        [str],
        tuple[Mapping[str, object], ExclusiveHITLTransitionObservation],
    ],
    evidence_observer: Callable[[ExclusiveHITLSwitchEvidence], None] | None,
    hold_after_failure: Callable[[], None],
    failure_hold_evidence: Callable[[], tuple[str, ...]],
    clock: Callable[[], float],
) -> tuple[dict[str, object], ExclusiveHITLSwitchEvidence]:
    """Run a backend-neutral exclusive source-switch transaction.

    The adapter callback owns concrete HOLD/Home/source-selection/activation
    operations.  This coordinator validates their typed evidence and requests
    fail-closed HOLD only for transition or evidence failure.  The optional
    observer is best-effort observability: its failure cannot request HOLD,
    revoke readiness, or otherwise become a hidden control gate.  It never
    composes, resolves, dispatches, or admits a trace fact.
    """

    if evidence_observer is not None and not callable(evidence_observer):
        raise TypeError("evidence_observer must be callable or None.")

    def observe(evidence: ExclusiveHITLSwitchEvidence) -> None:
        if evidence_observer is None:
            return
        try:
            evidence_observer(evidence)
        except Exception:
            # The observer/capture owner exposes recording health separately.
            # Observability must not become source-switch authority.
            pass

    if direction not in EXCLUSIVE_HITL_SWITCH_DIRECTIONS:
        raise ValueError("Unsupported exclusive HITL switch direction.")
    previous_source, target_source = (
        ("policy", "human")
        if direction == EXCLUSIVE_HITL_POLICY_TO_HUMAN
        else ("human", "policy")
    )
    current = _required(current_source, name="current_source")
    requested = _finite(requested_at, name="requested_at")
    intent_id = _required(trigger_intent_id, name="trigger_intent_id")
    intent_hash = _required(trigger_intent_hash, name="trigger_intent_hash")
    clock_id = _required(clock_domain_id, name="clock_domain_id")
    if current == target_source:
        observed_at = max(requested, _finite(clock(), name="completed_at"))
        evidence = ExclusiveHITLSwitchEvidence(
            switch_id=f"exclusive-hitl-switch:{intent_id}:already-active",
            trigger_intent_id=intent_id,
            trigger_intent_hash=intent_hash,
            direction=direction,
            previous_source=target_source,
            target_source=target_source,
            requested_at=requested,
            completed_at=observed_at,
            clock_domain_id=clock_id,
            succeeded=True,
            status="already_active",
        )
        observe(evidence)
        return {}, evidence
    if current != previous_source:
        raise RuntimeError(
            "Exclusive HITL switch source mismatch: expected "
            f"{previous_source}, got {current or 'unknown'}."
        )

    try:
        result, observed = transition(target_source)
        if not isinstance(observed, ExclusiveHITLTransitionObservation):
            raise TypeError(
                "transition must return ExclusiveHITLTransitionObservation."
            )
        if observed.target_source != target_source:
            raise ValueError("Transition observation reports another target source.")
        completed_at = max(requested, observed.completed_at)
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    *observed.hold_evidence_refs,
                    observed.source_selection_record_id,
                    observed.source_selection_record_hash,
                    observed.activation_id,
                    observed.activation_hash,
                    observed.target_emission_id,
                    observed.target_emission_hash,
                    observed.target_id,
                    observed.target_hash,
                    observed.target_admission_receipt_id,
                    observed.target_completion_receipt_id,
                    observed.command_id,
                    observed.dispatch_receipt_id,
                    *(
                        ()
                        if observed.home_capture_id is None
                        else (
                            observed.home_capture_id,
                            observed.home_capture_hash,
                        )
                    ),
                )
            )
        )
        evidence = ExclusiveHITLSwitchEvidence(
            switch_id=f"exclusive-hitl-switch:{intent_id}",
            trigger_intent_id=intent_id,
            trigger_intent_hash=intent_hash,
            direction=direction,
            previous_source=previous_source,
            target_source=target_source,
            requested_at=requested,
            completed_at=completed_at,
            clock_domain_id=clock_id,
            succeeded=True,
            status="ready",
            hold_evidence_refs=observed.hold_evidence_refs,
            home_capture_id=observed.home_capture_id,
            home_capture_hash=observed.home_capture_hash,
            home_capture=observed.home_capture,
            source_selection_record_id=observed.source_selection_record_id,
            source_selection_record_hash=observed.source_selection_record_hash,
            activation_id=observed.activation_id,
            activation_hash=observed.activation_hash,
            target_emission_id=observed.target_emission_id,
            target_emission_hash=observed.target_emission_hash,
            target_id=observed.target_id,
            target_hash=observed.target_hash,
            target_admission_receipt_id=observed.target_admission_receipt_id,
            target_completion_receipt_id=observed.target_completion_receipt_id,
            command_id=observed.command_id,
            dispatch_receipt_id=observed.dispatch_receipt_id,
            source_reset_performed=observed.source_reset_performed,
            evidence_refs=evidence_refs,
        )
        observe(evidence)
        return dict(result), evidence
    except BaseException as error:
        try:
            hold_after_failure()
        except BaseException:
            pass
        try:
            failed_hold_refs = failure_hold_evidence()
        except BaseException:
            # Preserve the first transition/validation failure.  HOLD evidence
            # is useful lineage, but a broken observer must not replace the
            # causal error or prevent the failed switch fact from being built.
            failed_hold_refs = ()
        completed_at = max(requested, _finite(clock(), name="failed_at"))
        failed = ExclusiveHITLSwitchEvidence(
            switch_id=f"exclusive-hitl-switch:{intent_id}:failed",
            trigger_intent_id=intent_id,
            trigger_intent_hash=intent_hash,
            direction=direction,
            previous_source=previous_source,
            target_source=target_source,
            requested_at=requested,
            completed_at=completed_at,
            clock_domain_id=clock_id,
            succeeded=False,
            status="failed",
            hold_evidence_refs=failed_hold_refs,
            error=f"{type(error).__name__}:{error}",
        )
        observe(failed)
        raise


def run_exclusive_hitl_switch_via_port(
    *,
    trigger_intent_id: str,
    trigger_intent_hash: str,
    direction: str,
    current_source: str,
    requested_at: float,
    clock_domain_id: str,
    port: ExclusiveHITLTransitionPort,
    evidence_observer: Callable[[ExclusiveHITLSwitchEvidence], None] | None,
    clock: Callable[[], float],
) -> tuple[dict[str, object], ExclusiveHITLSwitchEvidence]:
    """Run the shared transaction through one typed concrete application port."""

    if not isinstance(port, ExclusiveHITLTransitionPort):
        raise TypeError("port must implement ExclusiveHITLTransitionPort.")
    return run_exclusive_hitl_switch_transaction(
        trigger_intent_id=trigger_intent_id,
        trigger_intent_hash=trigger_intent_hash,
        direction=direction,
        current_source=current_source,
        requested_at=requested_at,
        clock_domain_id=clock_domain_id,
        transition=port.transition,
        evidence_observer=evidence_observer,
        hold_after_failure=port.hold_after_failure,
        failure_hold_evidence=port.failure_hold_evidence,
        clock=clock,
    )


__all__ = [
    "EXCLUSIVE_HITL_HUMAN_TO_POLICY",
    "EXCLUSIVE_HITL_POLICY_TO_HUMAN",
    "EXCLUSIVE_HITL_SWITCH_DIRECTIONS",
    "EXCLUSIVE_HITL_SWITCH_STATUSES",
    "ExclusiveHITLTransitionPort",
    "ExclusiveHITLTransitionObservation",
    "ExclusiveHITLSwitchEvidence",
    "run_exclusive_hitl_switch_transaction",
    "run_exclusive_hitl_switch_via_port",
]
