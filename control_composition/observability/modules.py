"""Provisional ports for the two independent difference modules.

This file intentionally freezes only the smallest observation envelope needed
to keep the modules replaceable during development.  The payload is an
explicitly detached, capability-owned mapping; metric catalogues, thresholds,
feedback mappings, and persistence projections do not belong here.

``CompositionImmediateDifferenceObserver`` reads one input cut before
composition.  ``PostCompositionExecutionDifferenceObserver`` starts from a
composition evaluation and may later be enriched by downstream receipts.  Neither
protocol owns authority, safety, IK, dispatch, or measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

if TYPE_CHECKING:
    from ..composition import CompositionInputSnapshot, CompositionEvaluation

from ..composition import CompositionInputSnapshot, CompositionEvaluation, CompositionSpec
from .difference import (
    SourceDifferenceSpec,
    action_difference_semantic_payload,
    evaluate_composition_input_differences,
    evaluate_composition_source_differences,
)


DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE = "composition_immediate_difference"
DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION = (
    "post_composition_execution_difference"
)
DIFFERENCE_OBSERVATION_STATUSES = frozenset(
    {"available", "unavailable", "invalid"}
)
DIFFERENCE_OBSERVATION_ENVELOPE_HASH_DOMAIN = (
    "dexteleop.difference_observation_envelope.provisional"
)
DIFFERENCE_RECEIPT_REFERENCE_HASH_DOMAIN = (
    "dexteleop.difference_receipt_reference.provisional"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _sequence(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


@dataclass(frozen=True)
class DifferenceReceiptReference:
    """Opaque identity for a downstream fact used by a difference observer.

    This is deliberately not a safety or authority receipt.  The observer
    records only enough identity to establish which downstream fact was used;
    the owning package retains the full receipt type and semantics.
    """

    stage: str
    receipt_id: str
    receipt_hash: str
    observed_at: float
    clock_domain_id: str
    status: str = "observed"
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "stage": _required(self.stage, name="stage"),
            "receipt_id": _required(self.receipt_id, name="receipt_id"),
            "receipt_hash": _required(self.receipt_hash, name="receipt_hash"),
            "observed_at": _finite(self.observed_at, name="observed_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "status": _required(self.status, name="status"),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=DIFFERENCE_RECEIPT_REFERENCE_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class DifferenceObservationEnvelope:
    """Minimal lineage envelope shared by both difference modules.

    ``payload`` is intentionally capability-owned and opaque to the generic
    composition package.  A consumer must opt into the corresponding
    ``capability_id``/``payload_contract_hash`` before interpreting it.
    """

    module_id: str
    capability_id: str
    status: str
    reason: str
    control_domain_id: str
    clock_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    observed_at: float
    valid_until: float | None
    input_snapshot_id: str
    input_snapshot_hash: str
    composition_evaluation_id: str | None = None
    composition_evaluation_hash: str | None = None
    payload_contract_id: str | None = None
    payload_contract_hash: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    receipt_references: tuple[DifferenceReceiptReference, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        module_id = _required(self.module_id, name="module_id")
        capability_id = _required(self.capability_id, name="capability_id")
        status = _required(self.status, name="status")
        if status not in DIFFERENCE_OBSERVATION_STATUSES:
            raise ValueError(f"Unsupported difference observation status {status!r}.")
        reason = _required(self.reason, name="reason")
        control_domain_id = _required(
            self.control_domain_id,
            name="control_domain_id",
        )
        clock_domain_id = _required(
            self.clock_domain_id,
            name="clock_domain_id",
        )
        cutoff_at = _finite(self.cutoff_at, name="cutoff_at")
        cutoff_sequence = _sequence(
            self.cutoff_admission_sequence,
            name="cutoff_admission_sequence",
        )
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        observed_at = _finite(self.observed_at, name="observed_at")
        valid_until = self.valid_until
        if valid_until is not None:
            valid_until = _finite(valid_until, name="valid_until")
        input_id = _required(self.input_snapshot_id, name="input_snapshot_id")
        input_hash = _required(
            self.input_snapshot_hash,
            name="input_snapshot_hash",
        )
        result_id = self.composition_evaluation_id
        result_hash = self.composition_evaluation_hash
        if (result_id is None) != (result_hash is None):
            raise ValueError(
                "composition_evaluation_id/hash must appear together."
            )
        if result_id is not None:
            result_id = _required(result_id, name="composition_evaluation_id")
            result_hash = _required(
                result_hash,
                name="composition_evaluation_hash",
            )
        contract_id = self.payload_contract_id
        contract_hash = self.payload_contract_hash
        if (contract_id is None) != (contract_hash is None):
            raise ValueError(
                "payload_contract_id/hash must appear together."
            )
        if contract_id is not None:
            contract_id = _required(contract_id, name="payload_contract_id")
            contract_hash = _required(
                contract_hash,
                name="payload_contract_hash",
            )
        payload = deep_freeze(dict(self.payload))
        references = tuple(self.receipt_references)
        if any(
            not isinstance(reference, DifferenceReceiptReference)
            for reference in references
        ):
            raise TypeError(
                "receipt_references must contain DifferenceReceiptReference values."
            )
        reference_hashes = tuple(
            reference.semantic_content_hash for reference in references
        )
        if len(reference_hashes) != len(set(reference_hashes)):
            raise ValueError("receipt_references must be unique.")
        semantic_payload = {
            "module_id": module_id,
            "capability_id": capability_id,
            "status": status,
            "reason": reason,
            "control_domain_id": control_domain_id,
            "clock_domain_id": clock_domain_id,
            "cutoff_at": cutoff_at,
            "cutoff_admission_sequence": cutoff_sequence,
            "target_effective_at": target,
            "observed_at": observed_at,
            "valid_until": valid_until,
            "input_snapshot_id": input_id,
            "input_snapshot_hash": input_hash,
            "composition_evaluation_id": result_id,
            "composition_evaluation_hash": result_hash,
            "payload_contract_id": contract_id,
            "payload_contract_hash": contract_hash,
            "payload": payload,
            "receipt_reference_hashes": reference_hashes,
        }
        for name, value in (
            ("module_id", module_id),
            ("capability_id", capability_id),
            ("status", status),
            ("reason", reason),
            ("control_domain_id", control_domain_id),
            ("clock_domain_id", clock_domain_id),
            ("cutoff_at", cutoff_at),
            ("cutoff_admission_sequence", cutoff_sequence),
            ("target_effective_at", target),
            ("observed_at", observed_at),
            ("valid_until", valid_until),
            ("input_snapshot_id", input_id),
            ("input_snapshot_hash", input_hash),
            ("composition_evaluation_id", result_id),
            ("composition_evaluation_hash", result_hash),
            ("payload_contract_id", contract_id),
            ("payload_contract_hash", contract_hash),
            ("payload", payload),
            ("receipt_references", references),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic_payload,
                domain=DIFFERENCE_OBSERVATION_ENVELOPE_HASH_DOMAIN,
            ),
        )

    @property
    def observation_id(self) -> str:
        return f"difference-observation:{self.semantic_content_hash}"


def validate_difference_observation_lineage(
    observation: DifferenceObservationEnvelope,
    *,
    module_id: str,
    input_snapshot: "CompositionInputSnapshot",
    composition_evaluation: "CompositionEvaluation | None" = None,
) -> None:
    """Validate the only generic lineage facts shared by both modules."""

    if not isinstance(observation, DifferenceObservationEnvelope):
        raise TypeError(
            "observation must be a DifferenceObservationEnvelope."
        )
    if observation.module_id != _required(module_id, name="module_id"):
        raise ValueError("Difference observation module mismatch.")
    if observation.control_domain_id != input_snapshot.control_domain_id:
        raise ValueError("Difference observation control domain mismatch.")
    if observation.cutoff_at != input_snapshot.cutoff_at:
        raise ValueError("Difference observation cutoff mismatch.")
    if (
        observation.cutoff_admission_sequence
        != input_snapshot.cutoff_admission_sequence
    ):
        raise ValueError("Difference observation admission sequence mismatch.")
    if observation.target_effective_at != input_snapshot.target_effective_at:
        raise ValueError("Difference observation target mismatch.")
    if (
        observation.input_snapshot_id
        != input_snapshot.composition_input_snapshot_id
        or observation.input_snapshot_hash
        != input_snapshot.semantic_content_hash
    ):
        raise ValueError("Difference observation input lineage mismatch.")
    if composition_evaluation is None:
        if (
            observation.composition_evaluation_id is not None
            or observation.composition_evaluation_hash is not None
        ):
            raise ValueError(
                "Input-only difference observation cannot carry a result lineage."
            )
        return
    if (
        observation.composition_evaluation_id is None
        or observation.composition_evaluation_hash is None
    ):
        raise ValueError(
            "Post-composition difference observation requires result lineage."
        )
    if (
        observation.composition_evaluation_id != composition_evaluation.composition_evaluation_id
        or observation.composition_evaluation_hash
        != composition_evaluation.semantic_content_hash
    ):
        raise ValueError("Difference observation result lineage mismatch.")


class CompositionImmediateDifferenceObserver(Protocol):
    """Synchronous, same-cut source-to-source observation port."""

    capability_id: str

    def observe(
        self,
        input_snapshot: "CompositionInputSnapshot",
        *,
        observed_at: float,
        clock_domain_id: str,
    ) -> DifferenceObservationEnvelope: ...


class PostCompositionExecutionDifferenceObserver(Protocol):
    """Post-mix/execution observation port; receipt ownership stays downstream."""

    capability_id: str

    def observe(
        self,
        input_snapshot: "CompositionInputSnapshot",
        composition_evaluation: "CompositionEvaluation",
        *,
        observed_at: float,
        clock_domain_id: str,
        receipt_references: tuple[DifferenceReceiptReference, ...] = (),
    ) -> DifferenceObservationEnvelope: ...


class DisabledCompositionImmediateDifferenceObserver:
    """Explicit no-op observer for tests/profiles; performs no computation."""

    capability_id = "composition-immediate-difference.disabled"

    def observe(
        self,
        input_snapshot: "CompositionInputSnapshot",
        *,
        observed_at: float,
        clock_domain_id: str,
    ) -> DifferenceObservationEnvelope:
        return _disabled_observation(
            module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
            capability_id=self.capability_id,
            input_snapshot=input_snapshot,
            observed_at=observed_at,
            clock_domain_id=clock_domain_id,
        )


class DisabledPostCompositionExecutionDifferenceObserver:
    """Explicit no-op post-mix observer for tests/profiles."""

    capability_id = "post-composition-execution-difference.disabled"

    def observe(
        self,
        input_snapshot: "CompositionInputSnapshot",
        composition_evaluation: "CompositionEvaluation",
        *,
        observed_at: float,
        clock_domain_id: str,
        receipt_references: tuple[DifferenceReceiptReference, ...] = (),
    ) -> DifferenceObservationEnvelope:
        return _disabled_observation(
            module_id=DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
            capability_id=self.capability_id,
            input_snapshot=input_snapshot,
            composition_evaluation=composition_evaluation,
            observed_at=observed_at,
            clock_domain_id=clock_domain_id,
            receipt_references=receipt_references,
        )


class SourceDifferenceImmediateObserver:
    """Provisional pose/qpos source-to-source immediate observer.

    The metric implementation is intentionally delegated to the existing
    source-difference helpers.  The returned payload is opaque to the generic
    envelope, so a later metric profile can replace it without changing the
    causal port.
    """

    def __init__(
        self,
        *,
        composition_spec: CompositionSpec,
        difference_spec: SourceDifferenceSpec,
        capability_id: str,
    ) -> None:
        if not isinstance(composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(difference_spec, SourceDifferenceSpec):
            raise TypeError("difference_spec must be a SourceDifferenceSpec.")
        if not difference_spec.compared_reference_role_pairs:
            raise ValueError(
                "Immediate source difference requires at least one role pair."
            )
        difference_spec.validate_composition_spec(composition_spec)
        self.composition_spec = composition_spec
        self.difference_spec = difference_spec
        self.capability_id = _required(capability_id, name="capability_id")

    def observe(
        self,
        input_snapshot: "CompositionInputSnapshot",
        *,
        observed_at: float,
        clock_domain_id: str,
    ) -> DifferenceObservationEnvelope:
        observed = _finite(observed_at, name="observed_at")
        clock = _required(clock_domain_id, name="clock_domain_id")
        try:
            pairwise = evaluate_composition_input_differences(
                input_snapshot,
                spec=self.composition_spec,
                difference_spec=self.difference_spec,
                evaluated_at=observed,
                clock_domain_id=clock,
            )
        except (TypeError, ValueError) as error:
            return _build_observation_envelope(
                module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
                capability_id=self.capability_id,
                status="unavailable",
                reason=f"immediate_difference:{type(error).__name__}",
                input_snapshot=input_snapshot,
                observed_at=observed,
                clock_domain_id=clock,
                payload={"error": str(error)},
                payload_contract_id=self.difference_spec.spec_id,
                payload_contract_hash=self.difference_spec.semantic_content_hash,
            )
        if not pairwise:
            return _build_observation_envelope(
                module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
                capability_id=self.capability_id,
                status="unavailable",
                reason="immediate_difference:no_common_role_pair",
                input_snapshot=input_snapshot,
                observed_at=observed,
                clock_domain_id=clock,
                payload={"pairwise_differences": ()},
                payload_contract_id=self.difference_spec.spec_id,
                payload_contract_hash=self.difference_spec.semantic_content_hash,
            )
        observed_roles = {
            role
            for pair in self.difference_spec.compared_reference_role_pairs
            for role in pair
            if role in input_snapshot.selected_input_by_role
        }
        valid_until = min(
            input_snapshot.selected_input_by_role[role].effective_until
            for role in observed_roles
        )
        return _build_observation_envelope(
            module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
            capability_id=self.capability_id,
            status="available",
            reason="source_to_source_difference",
            input_snapshot=input_snapshot,
            observed_at=observed,
            clock_domain_id=clock,
            valid_until=valid_until,
            payload={
                "difference_scope": "source_to_source",
                "pairwise_differences": tuple(
                    action_difference_semantic_payload(item)
                    for item in pairwise
                ),
            },
            payload_contract_id=self.difference_spec.spec_id,
            payload_contract_hash=self.difference_spec.semantic_content_hash,
        )


class SourceToComposedDifferenceObserver:
    """Provisional passive post-mix source-to-composed observer.

    It implements only the post-composition portion of the second module.
    Execution receipts remain optional references and are owned by downstream
    observers/backends.
    """

    def __init__(
        self,
        *,
        composition_spec: CompositionSpec,
        difference_spec: SourceDifferenceSpec,
        capability_id: str,
    ) -> None:
        if not isinstance(composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(difference_spec, SourceDifferenceSpec):
            raise TypeError("difference_spec must be a SourceDifferenceSpec.")
        if not difference_spec.output_residual_roles:
            raise ValueError(
                "Post-mix source difference requires output residual roles."
            )
        difference_spec.validate_composition_spec(composition_spec)
        self.composition_spec = composition_spec
        self.difference_spec = difference_spec
        self.capability_id = _required(capability_id, name="capability_id")

    def observe(
        self,
        input_snapshot: "CompositionInputSnapshot",
        composition_evaluation: "CompositionEvaluation",
        *,
        observed_at: float,
        clock_domain_id: str,
        receipt_references: tuple[DifferenceReceiptReference, ...] = (),
    ) -> DifferenceObservationEnvelope:
        observed = _finite(observed_at, name="observed_at")
        clock = _required(clock_domain_id, name="clock_domain_id")
        source_observation = evaluate_composition_source_differences(
            input_snapshot,
            composition_evaluation,
            spec=self.composition_spec,
            difference_spec=self.difference_spec,
            evaluated_at=observed,
            clock_domain_id=clock,
        )
        residuals = source_observation.output_residuals
        if not residuals:
            return _build_observation_envelope(
                module_id=DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
                capability_id=self.capability_id,
                status="unavailable",
                reason="post_mix_difference:no_composed_residual",
                input_snapshot=input_snapshot,
                composition_evaluation=composition_evaluation,
                observed_at=observed,
                clock_domain_id=clock,
                receipt_references=receipt_references,
                payload={"output_residuals": ()},
                payload_contract_id=self.difference_spec.spec_id,
                payload_contract_hash=self.difference_spec.semantic_content_hash,
            )
        return _build_observation_envelope(
            module_id=DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
            capability_id=self.capability_id,
            status="available",
            reason="source_to_composed_difference",
            input_snapshot=input_snapshot,
            composition_evaluation=composition_evaluation,
            observed_at=observed,
            clock_domain_id=clock,
            valid_until=source_observation.valid_until,
            receipt_references=receipt_references,
            payload={
                "difference_scope": "source_to_composed",
                "output_residuals": tuple(
                    action_difference_semantic_payload(item)
                    for item in residuals
                ),
            },
            payload_contract_id=self.difference_spec.spec_id,
            payload_contract_hash=self.difference_spec.semantic_content_hash,
        )


def _disabled_observation(
    *,
    module_id: str,
    capability_id: str,
    input_snapshot: "CompositionInputSnapshot",
    observed_at: float,
    clock_domain_id: str,
    composition_evaluation: "CompositionEvaluation | None" = None,
    receipt_references: tuple[DifferenceReceiptReference, ...] = (),
) -> DifferenceObservationEnvelope:
    return DifferenceObservationEnvelope(
        module_id=module_id,
        capability_id=capability_id,
        status="unavailable",
        reason="disabled",
        control_domain_id=input_snapshot.control_domain_id,
        clock_domain_id=clock_domain_id,
        cutoff_at=input_snapshot.cutoff_at,
        cutoff_admission_sequence=input_snapshot.cutoff_admission_sequence,
        target_effective_at=input_snapshot.target_effective_at,
        observed_at=observed_at,
        valid_until=None,
        input_snapshot_id=input_snapshot.composition_input_snapshot_id,
        input_snapshot_hash=input_snapshot.semantic_content_hash,
        composition_evaluation_id=(
            None
            if composition_evaluation is None
            else composition_evaluation.composition_evaluation_id
        ),
        composition_evaluation_hash=(
            None
            if composition_evaluation is None
            else composition_evaluation.semantic_content_hash
        ),
        receipt_references=receipt_references,
    )


def _build_observation_envelope(
    *,
    module_id: str,
    capability_id: str,
    status: str,
    reason: str,
    input_snapshot: "CompositionInputSnapshot",
    observed_at: float,
    clock_domain_id: str,
    valid_until: float | None = None,
    composition_evaluation: "CompositionEvaluation | None" = None,
    payload_contract_id: str | None = None,
    payload_contract_hash: str | None = None,
    payload: Mapping[str, Any] | None = None,
    receipt_references: tuple[DifferenceReceiptReference, ...] = (),
) -> DifferenceObservationEnvelope:
    return DifferenceObservationEnvelope(
        module_id=module_id,
        capability_id=capability_id,
        status=status,
        reason=reason,
        control_domain_id=input_snapshot.control_domain_id,
        clock_domain_id=clock_domain_id,
        cutoff_at=input_snapshot.cutoff_at,
        cutoff_admission_sequence=input_snapshot.cutoff_admission_sequence,
        target_effective_at=input_snapshot.target_effective_at,
        observed_at=observed_at,
        valid_until=valid_until,
        input_snapshot_id=input_snapshot.composition_input_snapshot_id,
        input_snapshot_hash=input_snapshot.semantic_content_hash,
        composition_evaluation_id=(
            None
            if composition_evaluation is None
            else composition_evaluation.composition_evaluation_id
        ),
        composition_evaluation_hash=(
            None
            if composition_evaluation is None
            else composition_evaluation.semantic_content_hash
        ),
        payload_contract_id=payload_contract_id,
        payload_contract_hash=payload_contract_hash,
        payload={} if payload is None else payload,
        receipt_references=receipt_references,
    )


__all__ = [
    "DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE",
    "DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION",
    "DIFFERENCE_OBSERVATION_ENVELOPE_HASH_DOMAIN",
    "DIFFERENCE_OBSERVATION_STATUSES",
    "DIFFERENCE_RECEIPT_REFERENCE_HASH_DOMAIN",
    "CompositionImmediateDifferenceObserver",
    "DifferenceObservationEnvelope",
    "DifferenceReceiptReference",
    "DisabledCompositionImmediateDifferenceObserver",
    "DisabledPostCompositionExecutionDifferenceObserver",
    "PostCompositionExecutionDifferenceObserver",
    "SourceDifferenceImmediateObserver",
    "SourceToComposedDifferenceObserver",
    "validate_difference_observation_lineage",
]
