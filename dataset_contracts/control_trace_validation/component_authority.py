"""Component-action schema and conservative compound-authority lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class ComponentAuthorityValidationIndex:
    """Indexes shared by this rule family and later composition checks."""

    composition_spec_by_hash: FactIndex
    action_payload_schema_by_hash: FactIndex
    component_action_schema_by_hash: FactIndex
    component_rule_set_by_hash: FactIndex
    formal_strategy_spec_by_hash: FactIndex
    formal_component_binding_by_spec_hash: FactIndex
    compound_authority_profile_by_hash: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ComponentAuthorityValidationIndex":
        return cls(
            composition_spec_by_hash=context.payload_index(
                event_type="composition_spec",
                payload_key="semantic_content_hash",
            ),
            action_payload_schema_by_hash=context.payload_index(
                event_type="action_payload_schema",
                payload_key="semantic_content_hash",
            ),
            component_action_schema_by_hash=context.payload_index(
                event_type="component_action_schema",
                payload_key="semantic_content_hash",
            ),
            component_rule_set_by_hash=context.payload_index(
                event_type="component_composition_rule_set",
                payload_key="semantic_content_hash",
            ),
            formal_strategy_spec_by_hash=context.payload_index(
                event_type="formal_composition_strategy_spec",
                payload_key="semantic_content_hash",
            ),
            formal_component_binding_by_spec_hash=context.payload_index(
                event_type="formal_component_composition_binding",
                payload_key="composition_spec_hash",
            ),
            compound_authority_profile_by_hash=context.payload_index(
                event_type="conservative_compound_authority_profile",
                payload_key="semantic_content_hash",
            ),
        )


def validate_component_authority_lineage(
    context: ControlTraceValidationContext,
    indexes: ComponentAuthorityValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate one closed schema/binding/profile lineage family."""

    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "component_action_schema":
            payload_hash_by_component = dict(
                payload["payload_schema_hash_by_component"]
            )
            catalog_components = tuple(
                payload["component_catalog"]["components"]
            )
            catalog_component_by_id = {
                str(component["component_id"]): component
                for component in catalog_components
            }
            if set(catalog_component_by_id) != set(
                payload_hash_by_component
            ):
                issues.add(
                    "component_action_schema_catalog_mismatch",
                    fact.fact_id,
                    "Component action schema and embedded catalog must bind "
                    "the same component IDs.",
                )
            for component_id, payload_schema_hash in (
                payload_hash_by_component.items()
            ):
                schema_hash = str(payload_schema_hash)
                if (
                    len(
                        indexes.action_payload_schema_by_hash.get(
                            schema_hash,
                            (),
                        )
                    )
                    != 1
                ):
                    issues.add(
                        "component_action_schema_missing_payload_schema",
                        fact.fact_id,
                        "Component action schema must resolve each payload "
                        "schema definition.",
                        schema_hash,
                    )
                component = catalog_component_by_id.get(str(component_id))
                if (
                    component is not None
                    and schema_hash
                    not in {
                        str(value)
                        for value in component[
                            "accepted_payload_schema_hashes"
                        ]
                    }
                ):
                    issues.add(
                        "component_action_schema_payload_not_accepted",
                        fact.fact_id,
                        "Embedded component catalog does not accept the bound "
                        "payload schema.",
                        str(component_id),
                        schema_hash,
                    )
        elif (
            fact.event_type == "composition_spec"
            and payload["algebra"] == "formal_component_strategy_v2"
        ):
            role_schema_hashes = dict(
                payload["input_action_schema_hash_by_role"]
            )
            if role_schema_hashes:
                referenced_schema_hashes = {
                    str(payload["action_schema_hash"]),
                    *(
                        str(value)
                        for value in role_schema_hashes.values()
                    ),
                }
                for schema_hash in referenced_schema_hashes:
                    if (
                        len(
                            indexes.component_action_schema_by_hash.get(
                                schema_hash,
                                (),
                            )
                        )
                        != 1
                    ):
                        issues.add(
                            "formal_component_spec_missing_action_schema",
                            fact.fact_id,
                            "Role-specific formal composition must resolve "
                            "every input and output action schema definition.",
                            schema_hash,
                        )
            bindings = indexes.formal_component_binding_by_spec_hash.get(
                str(payload["semantic_content_hash"]),
                (),
            )
            if len(bindings) != 1:
                issues.add(
                    "formal_component_spec_missing_binding",
                    fact.fact_id,
                    "Formal component CompositionSpec must resolve exactly one "
                    "rule-set/strategy binding.",
                    str(payload["semantic_content_hash"]),
                )
        elif fact.event_type == "formal_component_composition_binding":
            spec_hash = str(payload["composition_spec_hash"])
            rule_set_hash = str(payload["component_rule_set_hash"])
            strategy_spec_hash = str(payload["formal_strategy_spec_hash"])
            specs = indexes.composition_spec_by_hash.get(spec_hash, ())
            if (
                len(specs) != 1
                or specs[0].payload["algebra"]
                != "formal_component_strategy_v2"
            ):
                issues.add(
                    "formal_component_binding_missing_spec",
                    fact.fact_id,
                    "Formal component binding must resolve one v2 "
                    "CompositionSpec.",
                    spec_hash,
                )
            if (
                len(indexes.component_rule_set_by_hash.get(rule_set_hash, ()))
                != 1
            ):
                issues.add(
                    "formal_component_binding_missing_rule_set",
                    fact.fact_id,
                    "Formal component binding must resolve one typed rule set.",
                    rule_set_hash,
                )
            strategy_specs = indexes.formal_strategy_spec_by_hash.get(
                strategy_spec_hash,
                (),
            )
            if (
                len(strategy_specs) != 1
                or strategy_specs[0].payload["composition_spec_hash"]
                != spec_hash
            ):
                issues.add(
                    "formal_component_binding_missing_strategy_spec",
                    fact.fact_id,
                    "Formal component binding must resolve one strategy bound "
                    "to the same CompositionSpec.",
                    strategy_spec_hash,
                )
        elif fact.event_type == "conservative_compound_authority_profile":
            spec_hash = str(payload["composition_spec_hash"])
            rule_set_hash = str(payload["component_rule_set_hash"])
            strategy_spec_hash = str(payload["formal_strategy_spec_hash"])
            if (
                len(indexes.composition_spec_by_hash.get(spec_hash, ()))
                != 1
            ):
                issues.add(
                    "compound_authority_profile_missing_composition_spec",
                    fact.fact_id,
                    "Compound authority profile must resolve one CompositionSpec.",
                    spec_hash,
                )
            if (
                len(indexes.component_rule_set_by_hash.get(rule_set_hash, ()))
                != 1
            ):
                issues.add(
                    "compound_authority_profile_missing_rule_set",
                    fact.fact_id,
                    "Compound authority profile must resolve one component rule set.",
                    rule_set_hash,
                )
            if (
                len(
                    indexes.formal_strategy_spec_by_hash.get(
                        strategy_spec_hash,
                        (),
                    )
                )
                != 1
            ):
                issues.add(
                    "compound_authority_profile_missing_strategy_spec",
                    fact.fact_id,
                    "Compound authority profile must resolve one formal strategy spec.",
                    strategy_spec_hash,
                )
            bindings = indexes.formal_component_binding_by_spec_hash.get(
                spec_hash,
                (),
            )
            if (
                len(bindings) != 1
                or bindings[0].payload["component_rule_set_hash"]
                != rule_set_hash
                or bindings[0].payload["formal_strategy_spec_hash"]
                != strategy_spec_hash
            ):
                issues.add(
                    "compound_authority_profile_binding_mismatch",
                    fact.fact_id,
                    "Compound authority profile hashes must resolve one exact "
                    "formal component binding.",
                    spec_hash,
                    rule_set_hash,
                    strategy_spec_hash,
                )
        elif fact.event_type == "compound_authority_failure":
            profile_hash = str(payload["profile_hash"])
            if (
                len(
                    indexes.compound_authority_profile_by_hash.get(
                        profile_hash,
                        (),
                    )
                )
                != 1
            ):
                issues.add(
                    "compound_authority_failure_missing_profile",
                    fact.fact_id,
                    "Compound authority failure must resolve its exact profile.",
                    profile_hash,
                )
            missing_sources = tuple(
                source_id
                for source_id in payload["source_event_ids"]
                if str(source_id) not in context.by_id
            )
            if missing_sources:
                issues.add(
                    "compound_authority_failure_missing_source",
                    fact.fact_id,
                    "Compound authority failure sources must resolve in this trace.",
                    *missing_sources,
                )


__all__ = [
    "ComponentAuthorityValidationIndex",
    "validate_component_authority_lineage",
]
