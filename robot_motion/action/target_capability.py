"""Structural target capability for component-keyed action candidates.

This module answers one narrow question: can a lower-owned target adapter
interpret this exact ``ComponentAction`` schema and active component set?
It does not run safety, IK, planning, reference generation, dispatch, or
authority fallback.  A rejected assessment is evidence for the execution
owner; it is not itself a HOLD or no-action command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .component_action import ComponentAction, ComponentActionSchema


TARGET_COMPONENT_ROUTE_HASH_DOMAIN_V1 = (
    "dexteleop.target_component_route.v1"
)
TARGET_RESOURCE_COMBINATION_HASH_DOMAIN_V1 = (
    "dexteleop.target_resource_combination.v1"
)
COMPONENT_TARGET_CAPABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.component_target_capability.v1"
)
TARGET_CAPABILITY_ASSESSMENT_HASH_DOMAIN_V1 = (
    "dexteleop.target_capability_assessment.v1"
)
ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID = (
    "dexteleop.robot_action_v1_target_adapter.v1"
)
ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH = canonical_content_hash(
    {
        "input": "ComponentAction",
        "output": "RobotAction.v1",
        "mapping": "exact_activation_schema_compatibility_view",
        "source_and_time": "supplied_by_authoritative_decision",
        "authority": "none",
        "safety": "none",
        "dispatch": "none",
    },
    domain="dexteleop.robot_action_v1_target_adapter.contract.v1",
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _unique_identifiers(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


@dataclass(frozen=True)
class TargetComponentRouteSpec:
    """Bind one component payload to one lower-owned adapter contract."""

    component_id: str
    input_payload_schema_hash: str
    handler_contract_id: str
    handler_contract_hash: str
    output_target_schema_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "input_payload_schema_hash",
                "handler_contract_id",
                "handler_contract_hash",
                "output_target_schema_hash",
            )
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_COMPONENT_ROUTE_HASH_DOMAIN_V1,
            ),
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "input_payload_schema_hash": self.input_payload_schema_hash,
            "handler_contract_id": self.handler_contract_id,
            "handler_contract_hash": self.handler_contract_hash,
            "output_target_schema_hash": self.output_target_schema_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TargetResourceCombinationSpec:
    """Declare the exact active component sets legal for one resource.

    ``participant_component_ids`` must equal every schema component claiming
    the resource.  Each entry in ``allowed_active_component_sets`` is an exact
    set, not a preference or priority rule.  Consequently a target never
    silently drops an already-selected component.
    """

    resource_id: str
    participant_component_ids: tuple[str, ...]
    allowed_active_component_sets: tuple[tuple[str, ...], ...]
    handler_contract_id: str
    handler_contract_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        resource_id = _required(self.resource_id, name="resource_id")
        participants = _unique_identifiers(
            tuple(self.participant_component_ids),
            name="participant_component_id",
        )
        participant_order = {
            component_id: index
            for index, component_id in enumerate(participants)
        }
        normalized_sets: list[tuple[str, ...]] = []
        seen: set[frozenset[str]] = set()
        for raw_values in tuple(self.allowed_active_component_sets):
            values = _unique_identifiers(
                tuple(raw_values),
                name="allowed_active_component_id",
            )
            unknown = sorted(set(values) - set(participants))
            if unknown:
                raise ValueError(
                    "Allowed resource combination contains non-participants: "
                    f"{unknown!r}."
                )
            identity = frozenset(values)
            if identity in seen:
                raise ValueError(
                    "Allowed resource component sets must be unique."
                )
            seen.add(identity)
            normalized_sets.append(
                tuple(sorted(values, key=participant_order.__getitem__))
            )
        if not normalized_sets:
            raise ValueError(
                "allowed_active_component_sets must contain at least one set."
            )
        unused = sorted(
            set(participants)
            - set().union(*(set(values) for values in normalized_sets))
        )
        if unused:
            raise ValueError(
                "Every resource participant must occur in an allowed set: "
                f"{unused!r}."
            )
        normalized_sets.sort(
            key=lambda values: (
                len(values),
                tuple(participant_order[value] for value in values),
            )
        )
        payload = {
            "resource_id": resource_id,
            "participant_component_ids": participants,
            "allowed_active_component_sets": tuple(normalized_sets),
            "handler_contract_id": _required(
                self.handler_contract_id,
                name="handler_contract_id",
            ),
            "handler_contract_hash": _required(
                self.handler_contract_hash,
                name="handler_contract_hash",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RESOURCE_COMBINATION_HASH_DOMAIN_V1,
            ),
        )

    def accepts(self, component_ids: tuple[str, ...]) -> bool:
        requested = frozenset(component_ids)
        return any(
            requested == frozenset(allowed)
            for allowed in self.allowed_active_component_sets
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "participant_component_ids": self.participant_component_ids,
            "allowed_active_component_sets": self.allowed_active_component_sets,
            "handler_contract_id": self.handler_contract_id,
            "handler_contract_hash": self.handler_contract_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ComponentTargetCapabilitySpec:
    """Exact structural acceptance contract for one action activation."""

    capability_id: str
    capability_version: str
    action_schema: ComponentActionSchema
    component_routes: tuple[TargetComponentRouteSpec, ...]
    resource_combinations: tuple[TargetResourceCombinationSpec, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        capability_id = _required(self.capability_id, name="capability_id")
        capability_version = _required(
            self.capability_version,
            name="capability_version",
        )
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        routes = tuple(self.component_routes)
        if any(not isinstance(item, TargetComponentRouteSpec) for item in routes):
            raise TypeError(
                "component_routes must contain TargetComponentRouteSpec values."
            )
        route_ids = tuple(item.component_id for item in routes)
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("Target component routes must be unique.")
        if set(route_ids) != set(self.action_schema.component_ids):
            raise ValueError(
                "Target capability must route every action-schema component "
                "exactly once."
            )
        route_by_id = {item.component_id: item for item in routes}
        ordered_routes = tuple(
            route_by_id[component_id]
            for component_id in self.action_schema.component_ids
        )
        for route in ordered_routes:
            expected_hash = self.action_schema.payload_schema_by_component[
                route.component_id
            ].semantic_content_hash
            if route.input_payload_schema_hash != expected_hash:
                raise ValueError(
                    f"Target route for {route.component_id!r} binds another "
                    "payload schema."
                )

        combinations = tuple(self.resource_combinations)
        if any(
            not isinstance(item, TargetResourceCombinationSpec)
            for item in combinations
        ):
            raise TypeError(
                "resource_combinations must contain "
                "TargetResourceCombinationSpec values."
            )
        resource_ids = tuple(item.resource_id for item in combinations)
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("Target resource combination specs must be unique.")
        catalog = self.action_schema.component_catalog
        expected_participants: dict[str, tuple[str, ...]] = {}
        for component in catalog.components:
            for resource_id in component.resource_claims:
                expected_participants.setdefault(resource_id, ())
                expected_participants[resource_id] = (
                    *expected_participants[resource_id],
                    component.component_id,
                )
        if set(resource_ids) != set(expected_participants):
            raise ValueError(
                "Target capability must declare every claimed resource exactly once."
            )
        combination_by_id = {item.resource_id: item for item in combinations}
        ordered_combinations = tuple(
            combination_by_id[resource_id]
            for resource_id in expected_participants
        )
        for combination in ordered_combinations:
            if (
                combination.participant_component_ids
                != expected_participants[combination.resource_id]
            ):
                raise ValueError(
                    f"Resource {combination.resource_id!r} participants do not "
                    "match the action schema claims."
                )

        payload = {
            "capability_id": capability_id,
            "capability_version": capability_version,
            "action_schema_hash": self.action_schema.semantic_content_hash,
            "component_route_hashes": tuple(
                item.semantic_content_hash for item in ordered_routes
            ),
            "resource_combination_hashes": tuple(
                item.semantic_content_hash for item in ordered_combinations
            ),
        }
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "capability_version", capability_version)
        object.__setattr__(self, "component_routes", ordered_routes)
        object.__setattr__(
            self,
            "resource_combinations",
            ordered_combinations,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_TARGET_CAPABILITY_HASH_DOMAIN_V1,
            ),
        )

    @property
    def action_schema_hash(self) -> str:
        return self.action_schema.semantic_content_hash

    @property
    def route_by_component(self) -> Mapping[str, TargetComponentRouteSpec]:
        return MappingProxyType(
            {item.component_id: item for item in self.component_routes}
        )

    @property
    def combination_by_resource(
        self,
    ) -> Mapping[str, TargetResourceCombinationSpec]:
        return MappingProxyType(
            {item.resource_id: item for item in self.resource_combinations}
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "action_schema_hash": self.action_schema_hash,
            "component_routes": tuple(
                item.semantic_payload() for item in self.component_routes
            ),
            "resource_combinations": tuple(
                item.semantic_payload() for item in self.resource_combinations
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TargetCapabilityAssessment:
    """Pure structural assessment; never an execution or safety receipt."""

    capability_id: str
    capability_hash: str
    candidate_action_hash: str
    candidate_action_schema_hash: str
    component_coverage: tuple[str, ...]
    active_components_by_resource: Mapping[str, tuple[str, ...]]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "capability_id",
            "capability_hash",
            "candidate_action_hash",
            "candidate_action_schema_hash",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        coverage = _unique_identifiers(
            tuple(self.component_coverage),
            name="component_coverage",
        )
        active = {
            _required(resource_id, name="resource_id"): _unique_identifiers(
                tuple(component_ids),
                name="active_resource_component_id",
            )
            for resource_id, component_ids in dict(
                self.active_components_by_resource
            ).items()
        }
        reasons = tuple(
            _required(value, name="rejection_reason")
            for value in self.rejection_reasons
        )
        if len(reasons) != len(set(reasons)):
            raise ValueError("rejection_reasons must be unique.")
        if self.accepted and reasons:
            raise ValueError("Accepted target assessment cannot have reasons.")
        if not self.accepted and not reasons:
            raise ValueError("Rejected target assessment requires reasons.")
        object.__setattr__(self, "component_coverage", coverage)
        object.__setattr__(
            self,
            "active_components_by_resource",
            MappingProxyType(active),
        )
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def assessment_id(self) -> str:
        return f"target-capability-assessment:{self.semantic_content_hash}"

    def semantic_payload(self) -> dict[str, object]:
        return {
            "assessment_id": self.assessment_id,
            "capability_id": self.capability_id,
            "capability_hash": self.capability_hash,
            "candidate_action_hash": self.candidate_action_hash,
            "candidate_action_schema_hash": (
                self.candidate_action_schema_hash
            ),
            "component_coverage": self.component_coverage,
            "active_components_by_resource": dict(
                self.active_components_by_resource
            ),
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons,
            "semantic_content_hash": self.semantic_content_hash,
        }


def assess_component_target_capability(
    capability: ComponentTargetCapabilitySpec,
    candidate: ComponentAction,
) -> TargetCapabilityAssessment:
    """Assess exact schema, structural dependencies, and resource combinations."""

    if not isinstance(capability, ComponentTargetCapabilitySpec):
        raise TypeError("capability must be a ComponentTargetCapabilitySpec.")
    if not isinstance(candidate, ComponentAction):
        raise TypeError("candidate must be a ComponentAction.")
    reasons: list[str] = []
    if candidate.action_schema.semantic_content_hash != capability.action_schema_hash:
        reasons.append("action_schema_mismatch")

    catalog = candidate.action_schema.component_catalog
    active_ids = tuple(candidate.component_coverage)
    active_set = set(active_ids)
    for component_id in active_ids:
        route = capability.route_by_component.get(component_id)
        if route is None:
            reasons.append(f"component_route_missing:{component_id}")
            continue
        actual_payload_hash = candidate.value(
            component_id
        ).payload_schema.semantic_content_hash
        if route.input_payload_schema_hash != actual_payload_hash:
            reasons.append(f"component_payload_schema_mismatch:{component_id}")
        missing = sorted(
            set(catalog.component_by_id[component_id].dependencies) - active_set
        )
        if missing:
            reasons.append(
                f"component_dependency_missing:{component_id}:{','.join(missing)}"
            )
    for atomic_group_id in catalog.atomic_group_ids:
        members = {
            item.component_id
            for item in catalog.components_for_atomic_group(atomic_group_id)
        }
        selected = members & active_set
        if selected and selected != members:
            reasons.append(f"atomic_group_incomplete:{atomic_group_id}")

    active_by_resource: dict[str, tuple[str, ...]] = {}
    for component in catalog.components:
        if component.component_id not in active_set:
            continue
        for resource_id in component.resource_claims:
            active_by_resource.setdefault(resource_id, ())
            active_by_resource[resource_id] = (
                *active_by_resource[resource_id],
                component.component_id,
            )
    for resource_id, component_ids in active_by_resource.items():
        combination = capability.combination_by_resource.get(resource_id)
        if combination is None:
            reasons.append(f"resource_combination_missing:{resource_id}")
        elif not combination.accepts(component_ids):
            reasons.append(
                "resource_combination_not_accepted:"
                f"{resource_id}:{','.join(component_ids)}"
            )

    unique_reasons = tuple(dict.fromkeys(reasons))
    payload = {
        "capability_id": capability.capability_id,
        "capability_hash": capability.semantic_content_hash,
        "candidate_action_hash": candidate.semantic_content_hash,
        "candidate_action_schema_hash": (
            candidate.action_schema.semantic_content_hash
        ),
        "component_coverage": active_ids,
        "active_components_by_resource": active_by_resource,
        "accepted": not unique_reasons,
        "rejection_reasons": unique_reasons,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=TARGET_CAPABILITY_ASSESSMENT_HASH_DOMAIN_V1,
    )
    return TargetCapabilityAssessment(
        **payload,
        semantic_content_hash=semantic_hash,
    )


def build_robot_action_v1_target_capability(
    *,
    capability_id: str,
    action_schema: ComponentActionSchema,
    handler_contract_id: str,
    handler_contract_hash: str,
    output_target_schema_hash: str,
) -> ComponentTargetCapabilitySpec:
    """Bind an existing one-component-per-resource RobotAction v1 adapter.

    The builder rejects shared resource claims because the legacy carrier has
    no generic representation for combining independently keyed pose, wrench,
    velocity, or effort channels on one resource.
    """

    if not isinstance(action_schema, ComponentActionSchema):
        raise TypeError("action_schema must be a ComponentActionSchema.")
    handler_id = _required(
        handler_contract_id,
        name="handler_contract_id",
    )
    handler_hash = _required(
        handler_contract_hash,
        name="handler_contract_hash",
    )
    output_hash = _required(
        output_target_schema_hash,
        name="output_target_schema_hash",
    )
    catalog = action_schema.component_catalog
    participants_by_resource: dict[str, tuple[str, ...]] = {}
    for component in catalog.components:
        for resource_id in component.resource_claims:
            participants_by_resource.setdefault(resource_id, ())
            participants_by_resource[resource_id] = (
                *participants_by_resource[resource_id],
                component.component_id,
            )
    shared = {
        resource_id: participants
        for resource_id, participants in participants_by_resource.items()
        if len(participants) != 1
    }
    if shared:
        raise ValueError(
            "RobotAction v1 target capability cannot bind shared resources: "
            f"{sorted(shared)!r}."
        )
    return ComponentTargetCapabilitySpec(
        capability_id=_required(capability_id, name="capability_id"),
        capability_version="v1",
        action_schema=action_schema,
        component_routes=tuple(
            TargetComponentRouteSpec(
                component_id=component_id,
                input_payload_schema_hash=action_schema.payload_schema_by_component[
                    component_id
                ].semantic_content_hash,
                handler_contract_id=handler_id,
                handler_contract_hash=handler_hash,
                output_target_schema_hash=output_hash,
            )
            for component_id in action_schema.component_ids
        ),
        resource_combinations=tuple(
            TargetResourceCombinationSpec(
                resource_id=resource_id,
                participant_component_ids=participants,
                allowed_active_component_sets=(participants,),
                handler_contract_id=handler_id,
                handler_contract_hash=handler_hash,
            )
            for resource_id, participants in participants_by_resource.items()
        ),
    )


__all__ = [
    "COMPONENT_TARGET_CAPABILITY_HASH_DOMAIN_V1",
    "TARGET_CAPABILITY_ASSESSMENT_HASH_DOMAIN_V1",
    "TARGET_COMPONENT_ROUTE_HASH_DOMAIN_V1",
    "TARGET_RESOURCE_COMBINATION_HASH_DOMAIN_V1",
    "ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH",
    "ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID",
    "ComponentTargetCapabilitySpec",
    "TargetCapabilityAssessment",
    "TargetComponentRouteSpec",
    "TargetResourceCombinationSpec",
    "assess_component_target_capability",
    "build_robot_action_v1_target_capability",
]
