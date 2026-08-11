"""Composition-session specification, rearm, and transition lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class CompositionSessionValidationIndex:
    """Indexes used by the composition-session lineage rule family."""

    numerical_spec_by_hash: FactIndex
    session_spec_by_id: FactIndex
    rearm_spec_by_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
        *,
        numerical_spec_by_hash: FactIndex,
    ) -> "CompositionSessionValidationIndex":
        return cls(
            numerical_spec_by_hash=numerical_spec_by_hash,
            session_spec_by_id=context.payload_index(
                event_type="composition_session_spec",
                payload_key="operation_spec_id",
            ),
            rearm_spec_by_id=context.payload_index(
                event_type="composition_session_rearm_spec",
                payload_key="rearm_spec_id",
            ),
        )


def _validate_session_records(
    context: ControlTraceValidationContext,
    indexes: CompositionSessionValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "composition_session_spec":
            spec_hash = str(payload["composition_spec_hash"])
            if len(indexes.numerical_spec_by_hash.get(spec_hash, ())) != 1:
                issues.add(
                    "composition_session_missing_numerical_spec",
                    fact.fact_id,
                    "Composition session must resolve its exact numerical spec.",
                    spec_hash,
                )
        elif fact.event_type == "composition_session_rearm_spec":
            logical_spec_id = str(payload["logical_operation_spec_id"])
            specs = indexes.session_spec_by_id.get(logical_spec_id, ())
            if (
                len(specs) != 1
                or specs[0].payload["semantic_content_hash"]
                != payload["logical_operation_spec_hash"]
            ):
                issues.add(
                    "composition_session_rearm_missing_logical_spec",
                    fact.fact_id,
                    "Composition rearm spec must resolve its exact logical session.",
                    logical_spec_id,
                )
        elif fact.event_type == "composition_session_rearm_receipt":
            rearm_spec_id = str(payload["rearm_spec_id"])
            specs = indexes.rearm_spec_by_id.get(rearm_spec_id, ())
            if (
                len(specs) != 1
                or specs[0].payload["semantic_content_hash"]
                != payload["rearm_spec_hash"]
            ):
                issues.add(
                    "composition_session_rearm_receipt_missing_spec",
                    fact.fact_id,
                    "Composition rearm receipt must resolve its exact rearm spec.",
                    rearm_spec_id,
                )
            if (
                int(payload["next_session_epoch"])
                != int(payload["prior_session_epoch"]) + 1
                or int(payload["composer_producer_epoch"])
                <= int(payload["prior_composer_producer_epoch"])
                or int(payload["authority_epoch"])
                <= int(payload["prior_authority_epoch"])
                or int(payload["execution_epoch"])
                <= int(payload["prior_execution_epoch"])
            ):
                issues.add(
                    "composition_session_rearm_epoch_not_advanced",
                    fact.fact_id,
                    "Composition rearm must advance activation, producer, "
                    "authority, and execution epochs.",
                )
        elif fact.event_type == "composition_session_transition":
            spec_id = str(payload["operation_spec_id"])
            specs = indexes.session_spec_by_id.get(spec_id, ())
            if (
                len(specs) != 1
                or specs[0].payload["semantic_content_hash"]
                != payload["operation_spec_hash"]
            ):
                issues.add(
                    "composition_session_transition_missing_spec",
                    fact.fact_id,
                    "Composition session transition must resolve its exact spec.",
                    spec_id,
                )
            source_id = str(payload["source_fact_id"])
            source_fact = context.by_id.get(source_id)
            if (
                source_fact is None
                or source_fact.semantic_content_hash
                != payload["source_fact_hash"]
            ):
                issues.add(
                    "composition_session_transition_missing_source_fact",
                    fact.fact_id,
                    "Composition session transition must resolve its exact "
                    "source fact.",
                    source_id,
                )


def _validate_session_state_chains(
    context: ControlTraceValidationContext,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    transitions_by_spec: dict[str, list[ControlTraceFact]] = {}
    for fact in context.facts_of_type("composition_session_transition"):
        transitions_by_spec.setdefault(
            str(fact.payload["operation_spec_id"]),
            [],
        ).append(fact)

    for spec_id, transitions in transitions_by_spec.items():
        transitions_by_epoch: dict[int, list[ControlTraceFact]] = {}
        epoch_order: list[int] = []
        for transition in transitions:
            epoch = int(transition.payload["session_epoch"])
            if epoch not in transitions_by_epoch:
                epoch_order.append(epoch)
                transitions_by_epoch[epoch] = []
            transitions_by_epoch[epoch].append(transition)
        if epoch_order != sorted(epoch_order) or any(
            next_epoch != epoch + 1
            for epoch, next_epoch in zip(epoch_order, epoch_order[1:])
        ):
            issues.add(
                "composition_session_epoch_sequence_invalid",
                transitions[0].fact_id,
                "Composition activation epochs must be ordered and contiguous.",
                spec_id,
            )
        for epoch_index, epoch in enumerate(epoch_order):
            epoch_transitions = transitions_by_epoch[epoch]
            previous_after_hash: str | None = None
            terminal_seen = False
            for index, transition_fact in enumerate(epoch_transitions):
                payload = transition_fact.payload
                event = str(payload["transition"])
                if index == 0 and event not in {"start", "start_failed"}:
                    issues.add(
                        "composition_session_missing_start",
                        transition_fact.fact_id,
                        "Each composition activation must begin with start or "
                        "start_failed.",
                        spec_id,
                    )
                if terminal_seen:
                    issues.add(
                        "composition_session_transition_after_terminal",
                        transition_fact.fact_id,
                        "A composition activation cannot transition after its "
                        "terminal event.",
                        spec_id,
                    )
                if (
                    previous_after_hash is not None
                    and payload["state_before_hash"] != previous_after_hash
                ):
                    issues.add(
                        "composition_session_state_chain_mismatch",
                        transition_fact.fact_id,
                        "State hashes must form one exact chain within an "
                        "activation epoch.",
                        spec_id,
                    )
                source_fact = context.by_id.get(str(payload["source_fact_id"]))
                expected_operation = None
                if event == "start":
                    expected_operation = (
                        "start_composition_session"
                        if epoch_index == 0
                        else "rearm_composition_session"
                    )
                elif event == "supersede":
                    expected_operation = "rearm_composition_session"
                elif event == "stop":
                    expected_operation = "stop_composition_session"
                if expected_operation is not None and (
                    source_fact is None
                    or source_fact.event_type != "station_operation_transition"
                    or source_fact.payload["operation_kind"]
                    != expected_operation
                    or not bool(source_fact.payload["succeeded"])
                ):
                    issues.add(
                        "composition_session_terminal_source_invalid",
                        transition_fact.fact_id,
                        "Session activation transitions require the matching "
                        "successful Station operation.",
                        str(payload["source_fact_id"]),
                    )
                if (
                    expected_operation == "rearm_composition_session"
                    and source_fact is not None
                    and source_fact.event_type == "station_operation_transition"
                ):
                    receipt_facts = tuple(
                        context.by_id.get(str(evidence_ref))
                        for evidence_ref in source_fact.payload["evidence_refs"]
                        if (
                            context.by_id.get(str(evidence_ref)) is not None
                            and context.by_id[str(evidence_ref)].event_type
                            == "composition_session_rearm_receipt"
                        )
                    )
                    expected_prior_epoch = (
                        epoch if event == "supersede" else epoch - 1
                    )
                    if (
                        len(receipt_facts) != 1
                        or int(
                            receipt_facts[0].payload["prior_session_epoch"]
                        )
                        != expected_prior_epoch
                        or int(
                            receipt_facts[0].payload["next_session_epoch"]
                        )
                        != expected_prior_epoch + 1
                    ):
                        issues.add(
                            "composition_session_rearm_transition_missing_receipt",
                            transition_fact.fact_id,
                            "Rearm transitions must resolve one receipt for the "
                            "exact prior/next activation epochs.",
                            str(payload["source_fact_id"]),
                        )
                previous_after_hash = str(payload["state_after_hash"])
                terminal_seen = event in {
                    "start_failed",
                    "supersede",
                    "stop",
                    "fault",
                }
            if epoch_index < len(epoch_order) - 1 and (
                not epoch_transitions
                or str(epoch_transitions[-1].payload["transition"])
                != "supersede"
            ):
                issues.add(
                    "composition_session_epoch_not_superseded",
                    epoch_transitions[-1].fact_id,
                    "A later activation requires the prior held activation to "
                    "be superseded.",
                    spec_id,
                )


def validate_composition_session_lineage(
    context: ControlTraceValidationContext,
    indexes: CompositionSessionValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate composition-session records and their ordered state chains."""

    _validate_session_records(context, indexes, issues)
    _validate_session_state_chains(context, issues)


__all__ = [
    "CompositionSessionValidationIndex",
    "validate_composition_session_lineage",
]
