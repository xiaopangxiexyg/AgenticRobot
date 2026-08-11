"""Composition input, evaluation, activation, and disposition validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class CompositionEvaluationValidationIndex:
    """Indexes shared by composition evaluation and downstream lineage."""

    input_by_id: FactIndex
    context_by_id: FactIndex
    contributor_input_by_source_fact_id: FactIndex
    snapshot_by_id: FactIndex
    evaluation_by_id: FactIndex
    disposition_by_result_id: FactIndex
    activation_scope_by_id: FactIndex
    activation_binding_by_subject_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "CompositionEvaluationValidationIndex":
        index = context.payload_index
        context_by_id: dict[str, list[ControlTraceFact]] = {
            context_id: list(facts)
            for context_id, facts in index(
                event_type="policy_context",
                payload_key="context_id",
            ).items()
        }
        for context_id, facts in index(
            event_type="composition_context",
            payload_key="context_id",
        ).items():
            context_by_id.setdefault(context_id, []).extend(facts)
        input_by_id: dict[str, list[ControlTraceFact]] = {
            input_id: list(facts)
            for input_id, facts in index(
                event_type="composition_input",
                payload_key="composition_input_id",
            ).items()
        }
        contributor_by_source: dict[str, list[ControlTraceFact]] = {}
        for input_id, facts in index(
            event_type="composition_contributor_input",
            payload_key="composition_input_id",
        ).items():
            input_by_id.setdefault(input_id, []).extend(facts)
            for contributor in facts:
                for source_fact_id in contributor.payload["source_fact_ids"]:
                    contributor_by_source.setdefault(
                        str(source_fact_id),
                        [],
                    ).append(contributor)
        return cls(
            input_by_id=input_by_id,
            context_by_id=context_by_id,
            contributor_input_by_source_fact_id=contributor_by_source,
            snapshot_by_id=index(
                event_type="composition_input_snapshot",
                payload_key="composition_input_snapshot_id",
            ),
            evaluation_by_id=index(
                event_type="composition_evaluation",
                payload_key="composition_evaluation_id",
            ),
            disposition_by_result_id=index(
                event_type="composition_authority_disposition",
                payload_key="composition_evaluation_id",
            ),
            activation_scope_by_id=index(
                event_type="composition_activation_scope",
                payload_key="activation_id",
            ),
            activation_binding_by_subject_id=index(
                event_type="composition_activation_binding",
                payload_key="subject_id",
            ),
        )


def validate_composition_evaluation_lineage(
    context: ControlTraceValidationContext,
    indexes: CompositionEvaluationValidationIndex,
    *,
    composition_spec_by_hash: FactIndex,
    component_action_schema_by_hash: FactIndex,
    composition_session_spec_by_id: FactIndex,
    direct_component_activation: bool,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate composition products without owning composition execution."""

    by_id = context.by_id
    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "composition_input_admission" and bool(
            payload["accepted"]
        ):
            input_id = str(payload["composition_input_id"])
            inputs = indexes.input_by_id.get(input_id, ())
            if len(inputs) != 1 or inputs[0].payload[
                "input_semantic_content_hash"
            ] != payload["input_semantic_content_hash"]:
                issues.add(
                    "composition_input_admission_missing_input",
                    fact.fact_id,
                    "Accepted composition input admission must resolve its exact input.",
                    input_id,
                )
        elif fact.event_type in {
            "composition_input",
            "composition_contributor_input",
        }:
            input_action_schema_hash = str(
                payload["action"]["action_schema_hash"]
            )
            if component_action_schema_by_hash and len(
                component_action_schema_by_hash.get(
                    input_action_schema_hash,
                    (),
                )
            ) != 1:
                issues.add(
                    "composition_input_missing_action_schema",
                    fact.fact_id,
                    "Composition input must resolve its action schema "
                    "definition when schema facts are active.",
                    input_action_schema_hash,
            )
            for context_id in payload["context_ids"]:
                if len(indexes.context_by_id.get(str(context_id), ())) != 1:
                    issues.add(
                        "composition_input_missing_context",
                        fact.fact_id,
                        "Composition input context dependency is absent.",
                        str(context_id),
                    )
            observation_cut_id = payload["observation_cut_id"]
            if observation_cut_id is not None:
                observation = by_id.get(str(observation_cut_id))
                if observation is None or observation.event_type != "observation_cut":
                    issues.add(
                        "composition_input_missing_observation_cut",
                        fact.fact_id,
                        "Composition input must resolve its producer observation cut.",
                        str(observation_cut_id),
                    )
            if fact.event_type == "composition_contributor_input":
                source_ids = tuple(payload["source_fact_ids"])
                source_hashes = tuple(payload["source_fact_hashes"])
                if len(source_ids) != len(source_hashes):
                    issues.add(
                        "composition_contributor_lineage_length_mismatch",
                        fact.fact_id,
                        "Contributor source IDs and hashes must have equal length.",
                    )
                for source_id, source_hash in zip(source_ids, source_hashes):
                    source_fact = by_id.get(str(source_id))
                    if (
                        source_fact is None
                        or source_fact.semantic_content_hash != source_hash
                    ):
                        issues.add(
                            "composition_contributor_missing_source_fact",
                            fact.fact_id,
                            "Contributor input must resolve each exact source fact.",
                            str(source_id),
                        )
        elif fact.event_type == "composition_input_snapshot":
            spec_hash = str(payload["composition_spec_hash"])
            if len(composition_spec_by_hash.get(spec_hash, ())) != 1:
                issues.add(
                    "composition_snapshot_missing_spec",
                    fact.fact_id,
                    "Composition snapshot must resolve its exact numerical spec.",
                    spec_hash,
                )
            selected_ids = payload["selected_input_ids_by_role"]
            selected_hashes = payload["selected_input_hashes_by_role"]
            role_schema_hashes = dict(
                payload["input_action_schema_hash_by_role"]
            )
            if role_schema_hashes:
                for schema_hash in {
                    str(payload["action_schema_hash"]),
                    *(str(value) for value in role_schema_hashes.values()),
                }:
                    if len(
                        component_action_schema_by_hash.get(schema_hash, ())
                    ) != 1:
                        issues.add(
                            "composition_snapshot_missing_action_schema",
                            fact.fact_id,
                            "Role-specific snapshot must resolve every input "
                            "and output action schema definition.",
                            schema_hash,
                        )
            for role, input_id in selected_ids.items():
                inputs = indexes.input_by_id.get(str(input_id), ())
                if len(inputs) != 1 or inputs[0].payload[
                    "input_semantic_content_hash"
                ] != selected_hashes[role]:
                    issues.add(
                        "composition_snapshot_missing_input",
                        fact.fact_id,
                        "Composition snapshot must resolve each selected input and hash.",
                        str(input_id),
                    )
                elif role_schema_hashes:
                    expected_schema_hash = role_schema_hashes.get(role)
                    if expected_schema_hash is None or str(
                        inputs[0].payload["action"]["action_schema_hash"]
                    ) != str(expected_schema_hash):
                        issues.add(
                            "composition_snapshot_role_action_schema_mismatch",
                            fact.fact_id,
                            "Selected input action schema must match the schema "
                            "declared for its role.",
                            str(input_id),
                            "" if expected_schema_hash is None else str(expected_schema_hash),
                        )
        elif fact.event_type == "composition_evaluation":
            spec_hash = str(payload["composition_spec_hash"])
            if len(composition_spec_by_hash.get(spec_hash, ())) != 1:
                issues.add(
                    "composition_evaluation_missing_spec",
                    fact.fact_id,
                    "Composition result must resolve its exact numerical spec.",
                    spec_hash,
                )
            elif (
                composition_spec_by_hash[spec_hash][0].payload[
                    "input_action_schema_hash_by_role"
                ]
                and (
                    str(
                        composition_spec_by_hash[spec_hash][0].payload[
                            "action_schema_hash"
                        ]
                    )
                    != str(payload["action_schema_hash"])
                    or len(
                        component_action_schema_by_hash.get(
                            str(payload["action_schema_hash"]),
                            (),
                        )
                    )
                    != 1
                )
            ):
                issues.add(
                    "composition_evaluation_output_action_schema_mismatch",
                    fact.fact_id,
                    "Relative composition evaluation must use and resolve the "
                    "absolute output action schema from its spec.",
                    str(payload["action_schema_hash"]),
                )
            snapshot_id = str(payload["composition_input_snapshot_id"])
            snapshots = indexes.snapshot_by_id.get(snapshot_id, ())
            if len(snapshots) != 1 or snapshots[0].payload[
                "semantic_content_hash"
            ] != payload["composition_input_snapshot_hash"]:
                issues.add(
                    "composition_evaluation_missing_snapshot",
                    fact.fact_id,
                    "Composition result must resolve its exact input snapshot.",
                    snapshot_id,
                )
            if len(payload["input_ids"]) != len(payload["input_hashes"]):
                issues.add(
                    "composition_evaluation_input_lineage_length_mismatch",
                    fact.fact_id,
                    "Composition result input IDs and hashes must have equal length.",
                )
            for input_id, input_hash in zip(
                payload["input_ids"], payload["input_hashes"]
            ):
                inputs = indexes.input_by_id.get(str(input_id), ())
                if len(inputs) != 1 or inputs[0].payload[
                    "input_semantic_content_hash"
                ] != input_hash:
                    issues.add(
                        "composition_evaluation_missing_input",
                        fact.fact_id,
                        "Composition result input lineage is absent or hash-mismatched.",
                        str(input_id),
                    )
        elif fact.event_type == "composition_activation_scope":
            operation_spec_id = str(payload["operation_spec_id"])
            specs = composition_session_spec_by_id.get(operation_spec_id, ())
            if not direct_component_activation and (
                len(specs) != 1
                or specs[0].payload["semantic_content_hash"]
                != payload["operation_spec_hash"]
            ):
                issues.add(
                    "composition_activation_missing_operation_spec",
                    fact.fact_id,
                    "Composition activation must resolve its exact operation spec.",
                    operation_spec_id,
                )
        elif fact.event_type == "composition_activation_binding":
            activation_id = str(payload["activation_id"])
            scopes = indexes.activation_scope_by_id.get(activation_id, ())
            if len(scopes) != 1 or scopes[0].payload[
                "semantic_content_hash"
            ] != payload["activation_hash"]:
                issues.add(
                    "composition_activation_binding_missing_scope",
                    fact.fact_id,
                    "Activation binding must resolve its exact scope.",
                    activation_id,
                )
            subject_id = str(payload["subject_id"])
            subject_kind = str(payload["subject_kind"])
            subject_hash_field = {
                "input": "input_semantic_content_hash",
                "snapshot": "semantic_content_hash",
                "evaluation": "semantic_content_hash",
            }.get(subject_kind)
            subject_index = {
                "input": indexes.input_by_id,
                "snapshot": indexes.snapshot_by_id,
                "evaluation": indexes.evaluation_by_id,
            }.get(subject_kind)
            subjects = () if subject_index is None else subject_index.get(
                subject_id, ()
            )
            if (
                subject_hash_field is None
                or len(subjects) != 1
                or subjects[0].payload[subject_hash_field]
                != payload["subject_hash"]
            ):
                issues.add(
                    "composition_activation_binding_missing_subject",
                    fact.fact_id,
                    "Activation binding must resolve its exact typed subject.",
                    subject_id,
                )
            if subject_kind == "snapshot" and len(subjects) == 1:
                for selected_input_id in subjects[0].payload[
                    "selected_input_ids_by_role"
                ].values():
                    input_bindings = (
                        indexes.activation_binding_by_subject_id.get(
                            str(selected_input_id), ()
                        )
                    )
                    if (
                        len(input_bindings) != 1
                        or input_bindings[0].payload["activation_id"]
                        != activation_id
                        or input_bindings[0].payload["activation_hash"]
                        != payload["activation_hash"]
                    ):
                        issues.add(
                            "composition_activation_snapshot_input_mismatch",
                            fact.fact_id,
                            "Activation-bound snapshot must select only inputs from the same activation.",
                            str(selected_input_id),
                        )
            if subject_kind == "evaluation" and len(subjects) == 1:
                snapshot_id = str(
                    subjects[0].payload["composition_input_snapshot_id"]
                )
                snapshot_bindings = (
                    indexes.activation_binding_by_subject_id.get(
                        snapshot_id, ()
                    )
                )
                if (
                    len(snapshot_bindings) != 1
                    or snapshot_bindings[0].payload["activation_id"]
                    != activation_id
                    or snapshot_bindings[0].payload["activation_hash"]
                    != payload["activation_hash"]
                ):
                    issues.add(
                        "composition_activation_evaluation_snapshot_mismatch",
                        fact.fact_id,
                        "Activation-bound evaluation must resolve a snapshot "
                        "from the same activation.",
                        snapshot_id,
                    )
        elif fact.event_type == "composition_authority_disposition":
            result_id = str(payload["composition_evaluation_id"])
            results = indexes.evaluation_by_id.get(result_id, ())
            if len(results) != 1 or results[0].payload[
                "semantic_content_hash"
            ] != payload["composition_evaluation_hash"]:
                issues.add(
                    "composition_disposition_missing_result",
                    fact.fact_id,
                    "Composition authority disposition must resolve its exact result.",
                    result_id,
                )


__all__ = [
    "CompositionEvaluationValidationIndex",
    "validate_composition_evaluation_lineage",
]
