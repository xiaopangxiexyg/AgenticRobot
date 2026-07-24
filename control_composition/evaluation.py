"""Complete immutable return record for one composition evaluation.

The record exposes every admitted fact actually used by the numerical
decision.  Persistence, visualization, and feedback adapters can consume this
one object without rereading a live producer, simulator, model, or device.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.producer_epoch import ProducerEpochSnapshot

from .composition import (
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionContextSnapshot,
    CompositionResult,
    CompositionSpec,
)
from .observability import (
    ActionDifference,
    SourceDifferenceObservation,
    SourceDifferenceSpec,
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class CompositionDiagnostic:
    """Typed non-authoritative diagnostic from an optional evaluation stage."""

    stage: str
    code: str
    message: str
    recoverable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _required(self.stage, name="stage"))
        object.__setattr__(self, "code", _required(self.code, name="code"))
        object.__setattr__(self, "message", _required(self.message, name="message"))
        object.__setattr__(self, "recoverable", bool(self.recoverable))


@dataclass(frozen=True)
class CompositionEvaluationRecord:
    """Self-contained causal record returned for one composition decision.

    ``trace_facts_emitted`` is the exact batch accepted by the injected trace
    sink for this evaluation.  Proposal admission is deliberately outside this
    record because it is a later authority-boundary operation.
    """

    composition_spec: CompositionSpec
    control_cut: ControlCut
    producer_epoch_snapshot: ProducerEpochSnapshot
    input_snapshot: CompositionInputSnapshot
    composition_result: CompositionResult
    required_context_channels: tuple[str, ...]
    context_snapshot: CompositionContextSnapshot | None
    source_difference_spec: SourceDifferenceSpec | None
    source_difference_observation: SourceDifferenceObservation | None
    diagnostics: tuple[CompositionDiagnostic, ...]
    trace_facts_emitted: tuple[ControlTraceFact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(self.control_cut, ControlCut):
            raise TypeError("control_cut must be a ControlCut.")
        if not isinstance(self.producer_epoch_snapshot, ProducerEpochSnapshot):
            raise TypeError(
                "producer_epoch_snapshot must be a ProducerEpochSnapshot."
            )
        if not isinstance(self.input_snapshot, CompositionInputSnapshot):
            raise TypeError("input_snapshot must be a CompositionInputSnapshot.")
        if not isinstance(self.composition_result, CompositionResult):
            raise TypeError("composition_result must be a CompositionResult.")
        spec_hash = self.composition_spec.semantic_content_hash
        if (
            self.input_snapshot.composition_spec_hash != spec_hash
            or self.composition_result.composition_spec_hash != spec_hash
        ):
            raise ValueError("Composition evaluation spec lineage does not match.")
        if (
            self.composition_result.composition_input_snapshot_id
            != self.input_snapshot.composition_input_snapshot_id
            or self.composition_result.composition_input_snapshot_hash
            != self.input_snapshot.semantic_content_hash
        ):
            raise ValueError("Composition evaluation snapshot lineage does not match.")
        if (
            self.control_cut.control_domain_id
            != self.input_snapshot.control_domain_id
            or self.control_cut.cutoff_at != self.input_snapshot.cutoff_at
            or self.control_cut.cutoff_admission_sequence
            != self.input_snapshot.cutoff_admission_sequence
        ):
            raise ValueError("Composition evaluation control cut does not match.")
        if (
            self.producer_epoch_snapshot.control_domain_id
            != self.control_cut.control_domain_id
            or self.producer_epoch_snapshot.cutoff_at
            != self.control_cut.cutoff_at
            or self.producer_epoch_snapshot.cutoff_admission_sequence
            != self.control_cut.cutoff_admission_sequence
        ):
            raise ValueError("Composition evaluation producer epochs do not match.")
        channels = tuple(
            _required(value, name="required_context_channel")
            for value in self.required_context_channels
        )
        if len(channels) != len(set(channels)):
            raise ValueError("required_context_channels must be unique.")
        if channels and self.context_snapshot is None:
            raise ValueError("Required context channels need a context snapshot.")
        if self.context_snapshot is not None:
            if (
                self.context_snapshot.control_domain_id
                != self.control_cut.control_domain_id
                or self.context_snapshot.cutoff_at != self.control_cut.cutoff_at
                or self.context_snapshot.cutoff_admission_sequence
                != self.control_cut.cutoff_admission_sequence
                or self.context_snapshot.target_effective_at
                != self.input_snapshot.target_effective_at
            ):
                raise ValueError("Composition context snapshot does not share the cut.")
            if tuple(self.context_snapshot.required_channels) != channels:
                raise ValueError(
                    "Composition context snapshot channels do not match the request."
                )
        if self.source_difference_spec is None:
            if self.source_difference_observation is not None:
                raise ValueError(
                    "Source difference observation requires a difference spec."
                )
        if self.source_difference_observation is not None:
            self.source_difference_spec.validate_composition_spec(
                self.composition_spec
            )
            observation = self.source_difference_observation
            if (
                observation.composition_input_snapshot_id
                != self.input_snapshot.composition_input_snapshot_id
                or observation.composition_input_snapshot_hash
                != self.input_snapshot.semantic_content_hash
                or observation.composition_result_id
                != self.composition_result.composition_result_id
                or observation.composition_result_hash
                != self.composition_result.semantic_content_hash
                or observation.source_difference_spec_hash
                != self.source_difference_spec.semantic_content_hash
            ):
                raise ValueError(
                    "Source difference observation does not share the evaluation."
                )
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, CompositionDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain CompositionDiagnostic values.")
        facts = tuple(self.trace_facts_emitted)
        if any(not isinstance(item, ControlTraceFact) for item in facts):
            raise TypeError("trace_facts_emitted must contain ControlTraceFact values.")
        object.__setattr__(self, "required_context_channels", channels)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "trace_facts_emitted", facts)

    @property
    def selected_inputs_by_role(self) -> dict[str, CompositionInputRecord]:
        """Return a detached role map for consumers that need source lineage."""

        return dict(self.input_snapshot.selected_input_by_role)

    def differences_for_role(self, role: str) -> tuple[ActionDifference, ...]:
        """Return every available source/output difference involving ``role``."""

        requested = _required(role, name="role")
        observation = self.source_difference_observation
        if observation is None:
            return ()
        return tuple(
            difference
            for difference in (
                *observation.pairwise_differences,
                *observation.output_residuals,
            )
            if requested
            in {difference.compared_role, difference.reference_role}
        )


__all__ = [
    "CompositionDiagnostic",
    "CompositionEvaluationRecord",
]
