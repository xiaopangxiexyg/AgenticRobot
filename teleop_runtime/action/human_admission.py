"""Receiver-side admission for typed Human command emissions.

This module adapts an already-produced Human emission, records its causal
observation/projection facts, and submits one proposal through injected lower
ports.  It owns no device, queue, authority, safety, dispatch, or application
takeover lifecycle.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Protocol, runtime_checkable

from robot_motion.action.proposal import ProposalAdmissionReceipt
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.observation_cut import (
    ObservationBundle,
    ObservationCutRecord,
    ObservationSampleRef,
    ReceiverObservationCutBuilder,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)

from .human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterResult,
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@runtime_checkable
class HumanProposalAdmissionCore(Protocol):
    """Minimal authoritative-core seam shared by Human source profiles."""

    def record_external_facts(self, facts: tuple[Any, ...]) -> object: ...

    def admit_proposal(
        self,
        draft: object,
        *,
        source_event_id: str,
    ) -> ProposalAdmissionReceipt: ...


@runtime_checkable
class HumanObservationCutAdmissionCore(Protocol):
    """Receiver-side cut admission seam; independent from proposal ownership."""

    coordinator: ControlAdmissionCoordinator

    def record_observation_cut(self, record: ObservationCutRecord) -> object: ...


def admit_human_observation_cut(
    *,
    core: HumanObservationCutAdmissionCore,
    builder: ReceiverObservationCutBuilder,
    emissions: tuple[HumanCommandEmission, ...],
    payload_handles_by_stream: Mapping[str, Any],
    trigger_reason: str = "human_command_emission",
    built_at: float | None = None,
    context_dependencies: tuple[str, ...] = (),
) -> ObservationBundle:
    """Admit one causal input cut shared by one or more Human proposals.

    Independent arm/hand emissions may reference subsets of the same cut. The
    union is recorded exactly once, while every later proposal retains its own
    emission identity and selection-group coverage.
    """

    if not isinstance(core, HumanObservationCutAdmissionCore):
        raise TypeError("core does not satisfy HumanObservationCutAdmissionCore.")
    if not isinstance(builder, ReceiverObservationCutBuilder):
        raise TypeError("builder must be a ReceiverObservationCutBuilder.")
    normalized_emissions = tuple(emissions)
    if not normalized_emissions:
        raise ValueError("At least one human emission is required.")
    selected: dict[str, ObservationSampleRef] = {}
    for emission in normalized_emissions:
        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("emissions must contain HumanCommandEmission values.")
        if not emission.observation_sample_refs:
            raise ValueError(
                "A receiver-admitted human cut requires source sample refs on "
                "every emission."
            )
        for stream_id, sample in emission.observation_sample_refs.items():
            existing = selected.get(stream_id)
            if existing is not None and existing != sample:
                raise ValueError(
                    f"Human emissions disagree on source sample {stream_id!r}."
                )
            selected[stream_id] = sample
    bundle = builder.capture(
        core.coordinator,
        selected_source_samples=selected,
        payload_handles_by_stream=payload_handles_by_stream,
        trigger_reason=trigger_reason,
        built_at=built_at,
        context_dependencies=context_dependencies,
    )
    core.record_observation_cut(bundle.record)
    return bundle


def admit_human_command_emission(
    *,
    core: HumanProposalAdmissionCore,
    adapter: HumanProposalAdapter,
    emission: HumanCommandEmission,
    effective_from: float,
    effective_until: float,
    target_effective_at: float,
    occurred_at: float,
    clock_domain_id: str,
    adaptation: HumanProposalAdapterResult | None = None,
    observation_cut: ObservationCutRecord | None = None,
) -> tuple[HumanProposalAdapterResult, ProposalAdmissionReceipt | None]:
    """Adapt, trace, and receiver-admit one Human emission exactly once.

    The seam owns no queue, authority lease, safety stage, or dispatch right.
    Repeated submission is rejected by the existing ProposalBuffer identity,
    so this layer does not introduce another deduplication owner.
    """

    if not isinstance(adapter, HumanProposalAdapter):
        raise TypeError("adapter must be a HumanProposalAdapter.")
    if not isinstance(emission, HumanCommandEmission):
        raise TypeError("emission must be a HumanCommandEmission.")
    if not isinstance(core, HumanProposalAdmissionCore):
        raise TypeError("core does not satisfy HumanProposalAdmissionCore.")
    observation_cut_id: str | None = None
    if observation_cut is not None:
        if not isinstance(observation_cut, ObservationCutRecord):
            raise TypeError("observation_cut must be an ObservationCutRecord.")
        config = adapter.config
        if (
            observation_cut.runtime_session_id != config.runtime_session_id
            or observation_cut.control_domain_id != config.control_domain_id
            or observation_cut.policy_node_id != config.producer_id
            or observation_cut.policy_node_epoch != config.producer_epoch
        ):
            raise ValueError(
                "Human observation cut belongs to another producer/session."
            )
        if not emission.observation_sample_refs:
            raise ValueError(
                "An emission linked to an observation cut must carry source "
                "sample refs."
            )
        for stream_id, source_sample in emission.observation_sample_refs.items():
            admitted_sample = observation_cut.selected_sample_by_stream.get(
                stream_id
            )
            if admitted_sample is None:
                raise ValueError(
                    f"Human observation cut does not contain {stream_id!r}."
                )
            if (
                admitted_sample.sample_id != source_sample.sample_id
                or admitted_sample.payload_object_id
                != source_sample.payload_object_id
            ):
                raise ValueError(
                    "Human observation cut sample identity/content does not "
                    "match the emission."
                )
        observation_cut_id = observation_cut.observation_cut_id
    effective_start = _finite(effective_from, field_name="effective_from")
    effective_end = _finite(effective_until, field_name="effective_until")
    target_time = _finite(
        target_effective_at,
        field_name="target_effective_at",
    )
    if effective_end <= effective_start:
        raise ValueError("effective_until must be later than effective_from.")
    if not effective_start <= target_time < effective_end:
        raise ValueError("target_effective_at must be inside the effective interval.")
    if adaptation is None:
        result = adapter.adapt(
            emission,
            effective_from=effective_start,
            effective_until=effective_end,
            target_effective_at=target_time,
            observation_cut_id=observation_cut_id,
        )
    else:
        if not isinstance(adaptation, HumanProposalAdapterResult):
            raise TypeError("adaptation must be a HumanProposalAdapterResult.")
        receipt = adaptation.source_adapter_receipt
        if receipt.source_emission_semantic_hash != emission.semantic_content_hash:
            raise ValueError("Adaptation belongs to another human emission.")
        draft = adaptation.proposal_draft
        if draft is not None and observation_cut_id is not None:
            if draft.observation_cut_id not in {None, observation_cut_id}:
                raise ValueError("Adaptation belongs to another observation cut.")
            audit_metadata = dict(draft.audit_metadata)
            audit_metadata["source_observation_cut_id"] = observation_cut_id
            draft = replace(
                draft,
                observation_cut_id=observation_cut_id,
                audit_metadata=audit_metadata,
            )
            result = replace(adaptation, proposal_draft=draft)
        if draft is not None and (
            float(draft.effective_from) != effective_start
            or float(draft.effective_until) != effective_end
            or float(draft.target_effective_at) != target_time
        ):
            raise ValueError("Adaptation validity interval does not match submission.")
        result = adaptation
    config = adapter.config
    facts = project_human_proposal_adapter_result(
        emission,
        result,
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        embodiment_instance_id=config.embodiment_instance_id,
        occurred_at=_finite(occurred_at, field_name="occurred_at"),
        clock_domain_id=_required(
            clock_domain_id,
            field_name="clock_domain_id",
        ),
    )
    core.record_external_facts(facts)
    receipt = None
    if (
        result.proposal_draft is not None
        and result.source_adapter_receipt.authoritative_eligible
    ):
        receipt = core.admit_proposal(
            result.proposal_draft,
            source_event_id=result.source_adapter_receipt.source_emission_id,
        )
    return result, receipt


__all__ = [
    "HumanObservationCutAdmissionCore",
    "HumanProposalAdmissionCore",
    "admit_human_command_emission",
    "admit_human_observation_cut",
]
