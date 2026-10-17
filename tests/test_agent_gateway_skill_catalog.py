from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    AgentSkillSchemaEntry,
    CrossTargetDeploymentReviewRef,
    SkillDefinition,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    agent_skill_schema_document_hash,
    validate_skill_route_catalog_closure,
)


def _strict_schema(*, property_name: str = "value") -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {property_name: {"type": "string"}},
        "required": [property_name],
    }


def _definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id="motion.move_tcp_to_pose",
        skill_version="1",
        skill_kind="motion",
        summary="Move one registered tool TCP to an absolute pose proposal.",
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=("right_arm",),
        parameter_semantics_id="tcp_pose_target.absolute.xyz_quat_xyzw.v1",
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1",),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def _schema_entry(definition: SkillDefinition) -> AgentSkillSchemaEntry:
    parameter_schema = _strict_schema(property_name="pose_targets_by_component")
    profile = SkillParameterProfileRef(
        parameter_profile_id="arms-only-safe-v1",
        parameter_profile_hash="profile:arms-only-safe-v1",
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema,
            schema_kind="parameter",
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=_strict_schema(property_name="execution_receipt"),
        task_evaluation_schema=_strict_schema(property_name="task_outcome"),
        parameter_profiles=(profile,),
    )


def _catalogs():
    definition = _definition()
    schema_entry = _schema_entry(definition)
    return (
        definition,
        schema_entry,
        SkillDefinitionCatalog(
            catalog_id="dexteleop.skill_definitions",
            catalog_version="1",
            skill_definitions=(definition,),
        ),
        AgentSkillSchemaCatalog(
            catalog_id="dexteleop.agent_skill_schemas",
            catalog_version="1",
            schema_entries=(schema_entry,),
        ),
    )


def _route(
    definition: SkillDefinition,
    schema_entry: AgentSkillSchemaEntry,
) -> SkillExecutionRoute:
    return SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id="sim.mujoco.rm65_inspire.openpi",
        route_version="1",
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=(
            schema_entry.execution_result_schema_hash
        ),
        task_evaluation_schema_hash=(
            schema_entry.task_evaluation_schema_hash
        ),
        allowed_parameter_profiles=schema_entry.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=5_000,
        cancellation_policy="hold_on_disconnect_or_timeout",
        requires_task_evaluation=True,
    )


def _manifest(
    route: SkillExecutionRoute,
    definition_catalog: SkillDefinitionCatalog,
    schema_catalog: AgentSkillSchemaCatalog,
    *,
    binding_hash: str = "binding:cube-lift:a",
) -> SkillRouteManifest:
    binding = SkillDeploymentBindingRef(
        deployment_binding_id="deployment:cube-lift",
        deployment_binding_hash=binding_hash,
        target_binding_id="target:mujoco:rm65-inspire",
        target_kind="sim",
        execution_manifest_hash="execution:mujoco:cube-lift",
    )
    return SkillRouteManifest(
        runtime_session_id="runtime:sim:test",
        control_domain_id="domain:arms",
        station_manifest_hash="station:sim:test",
        execution_manifest_hash="execution:mujoco:cube-lift",
        skill_definition_catalog_hash=definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=schema_catalog.semantic_content_hash,
        target_binding_id="target:mujoco:rm65-inspire",
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=("idempotency_v1",),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=binding,
                evaluation_plan_hash=(
                    "evaluation-plan:cube-lift-v1"
                    if route.requires_task_evaluation
                    else None
                ),
                availability_hash="availability:ready",
            ),
        ),
        explicit_exclusions=("real_target",),
    )


def test_skill_and_route_versions_are_separate_from_namespaced_ids() -> None:
    definition = _definition()
    assert definition.skill_id == "motion.move_tcp_to_pose"
    assert definition.skill_version == "1"
    with pytest.raises(ValueError, match="namespaced"):
        replace(definition, skill_id="move_tcp_to_pose_v1")
    with pytest.raises(ValueError, match="positive integer string"):
        replace(definition, skill_version="v1")


def test_skill_intended_target_is_explicit_but_not_a_route_binding() -> None:
    definition = _definition()
    assert definition.intended_target_kind == "target_agnostic"
    assert definition.cross_target_deployment_policy == "not_applicable"
    with pytest.raises(ValueError, match="target_agnostic Skills require"):
        replace(
            definition,
            cross_target_deployment_policy="review_required",
        )
    with pytest.raises(ValueError, match="must explicitly forbid"):
        replace(
            definition,
            intended_target_kind="real",
        )


def test_cross_target_route_requires_exact_deployment_review() -> None:
    definition = replace(
        _definition(),
        intended_target_kind="real",
        cross_target_deployment_policy="review_required",
    )
    schema_entry = _schema_entry(definition)
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.skill_definitions",
        catalog_version="1",
        skill_definitions=(definition,),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas",
        catalog_version="1",
        schema_entries=(schema_entry,),
    )
    route = _route(definition, schema_entry)
    manifest = _manifest(route, definitions, schemas)
    with pytest.raises(ValueError, match="requires an immutable"):
        validate_skill_route_catalog_closure(
            definition_catalog=definitions,
            schema_catalog=schemas,
            route_manifest=manifest,
        )

    review = CrossTargetDeploymentReviewRef(
        review_id="review:real-skill-on-sim:v1",
        skill_definition_hash=definition.semantic_content_hash,
        route_hash=route.semantic_content_hash,
        intended_target_kind="real",
        deployed_target_kind="sim",
        compatibility_assessment_hash="assessment:real-to-sim:v1",
        approval_receipt_hash="approval:deployment-review:v1",
    )
    entry = manifest.skill_routes[0]
    reviewed_manifest = replace(
        manifest,
        skill_routes=(
            replace(
                entry,
                deployment_binding=replace(
                    entry.deployment_binding,
                    cross_target_deployment_review=review,
                ),
            ),
        ),
    )
    validate_skill_route_catalog_closure(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=reviewed_manifest,
    )
    assert (
        reviewed_manifest.skill_routes[0]
        .deployment_binding.to_wire()["cross_target_deployment_review"][
            "cross_target_deployment_review_ref_hash"
        ]
        == review.semantic_content_hash
    )

    wrong_route_review = replace(review, route_hash="route:wrong")
    wrong_manifest = replace(
        manifest,
        skill_routes=(
            replace(
                entry,
                deployment_binding=replace(
                    entry.deployment_binding,
                    cross_target_deployment_review=wrong_route_review,
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not bind the exact"):
        validate_skill_route_catalog_closure(
            definition_catalog=definitions,
            schema_catalog=schemas,
            route_manifest=wrong_manifest,
        )


def test_cross_target_forbidden_policy_cannot_be_overridden_by_review() -> None:
    definition = replace(
        _definition(),
        intended_target_kind="real",
        cross_target_deployment_policy="forbidden",
    )
    schema_entry = _schema_entry(definition)
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.skill_definitions",
        catalog_version="1",
        skill_definitions=(definition,),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas",
        catalog_version="1",
        schema_entries=(schema_entry,),
    )
    route = _route(definition, schema_entry)
    manifest = _manifest(route, definitions, schemas)
    with pytest.raises(ValueError, match="forbids deployment"):
        validate_skill_route_catalog_closure(
            definition_catalog=definitions,
            schema_catalog=schemas,
            route_manifest=manifest,
        )


def test_schema_entry_requires_strict_schema_and_exact_profile_mapping() -> None:
    definition = _definition()
    with pytest.raises(ValueError, match="additionalProperties=false"):
        AgentSkillSchemaEntry(
            skill_id=definition.skill_id,
            skill_version="1",
            skill_definition_hash=definition.semantic_content_hash,
            parameter_schema={"type": "object", "properties": {}},
            execution_result_schema=_strict_schema(),
            task_evaluation_schema=_strict_schema(),
            parameter_profiles=(
                SkillParameterProfileRef("safe", "profile:safe", "wrong"),
            ),
        )

    schema_entry = _schema_entry(definition)
    with pytest.raises(ValueError, match="exact parameter schema hash"):
        replace(
            schema_entry,
            parameter_profiles=(
                replace(
                    schema_entry.parameter_profiles[0],
                    parameter_schema_hash="schema:other",
                ),
            ),
        )


def test_route_identity_is_stable_across_deployment_binding_changes() -> None:
    definition, schema_entry, definitions, schemas = _catalogs()
    route = _route(definition, schema_entry)
    first = _manifest(route, definitions, schemas, binding_hash="binding:a")
    second = _manifest(route, definitions, schemas, binding_hash="binding:b")
    assert first.skill_routes[0].route.semantic_content_hash == route.semantic_content_hash
    assert (
        first.skill_routes[0].semantic_content_hash
        != second.skill_routes[0].semantic_content_hash
    )
    assert first.semantic_content_hash != second.semantic_content_hash
    assert "evaluation_plan_hash" not in route.to_wire()
    changed_plan = replace(
        first,
        skill_routes=(
            replace(
                first.skill_routes[0],
                evaluation_plan_hash="evaluation-plan:cube-lift-v2",
            ),
        ),
    )
    assert (
        changed_plan.skill_routes[0].route.semantic_content_hash
        == route.semantic_content_hash
    )
    assert (
        changed_plan.skill_routes[0].semantic_content_hash
        != first.skill_routes[0].semantic_content_hash
    )


def test_manifest_entry_evaluation_plan_presence_matches_route_semantics() -> None:
    definition, schema_entry, definitions, schemas = _catalogs()
    route = _route(definition, schema_entry)
    manifest = _manifest(route, definitions, schemas)
    with pytest.raises(ValueError, match="must bind an exact"):
        replace(manifest.skill_routes[0], evaluation_plan_hash=None)
    with pytest.raises(ValueError, match="without task evaluation"):
        replace(
            manifest.skill_routes[0],
            route=replace(route, requires_task_evaluation=False),
        )


def test_manifest_identity_is_computed_once_per_immutable_instance(monkeypatch) -> None:
    from agent_gateway.protocol import skill_catalog

    definition, schema_entry, definitions, schemas = _catalogs()
    manifest = _manifest(_route(definition, schema_entry), definitions, schemas)
    original = skill_catalog.canonical_content_hash
    calls = []

    def counted(payload, *, domain):
        if domain == skill_catalog.AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V2:
            calls.append(payload)
        return original(payload, domain=domain)

    monkeypatch.setattr(skill_catalog, "canonical_content_hash", counted)
    identity = manifest.semantic_content_hash
    for _ in range(3):
        assert manifest.semantic_content_hash == identity
        assert manifest.to_wire()["route_manifest_hash"] == identity
    assert len(calls) == 1
    assert original(
        calls[0], domain=skill_catalog.AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V2,
    ) == identity


@pytest.mark.parametrize("changed_field", [
    "runtime", "station", "definition_catalog", "schema_catalog", "io_catalog",
    "features", "exclusions", "availability", "evaluation", "binding", "route",
])
def test_replaced_manifest_never_reuses_another_instances_identity(changed_field) -> None:
    definition, schema_entry, definitions, schemas = _catalogs()
    manifest = _manifest(_route(definition, schema_entry), definitions, schemas)
    identity = manifest.semantic_content_hash
    entry = manifest.skill_routes[0]
    changes = {
        "runtime": {"runtime_session_id": "runtime:sim:other"},
        "station": {"station_manifest_hash": "station:sim:other"},
        "definition_catalog": {"skill_definition_catalog_hash": "definitions:other"},
        "schema_catalog": {"skill_schema_catalog_hash": "schemas:other"},
        "io_catalog": {"skill_io_catalog_hash": "io:other"},
        "features": {"supported_transport_features": ("idempotency_v1", "poll_v1")},
        "exclusions": {"explicit_exclusions": ("other-exclusion",)},
        "availability": {"skill_routes": (replace(entry, availability_hash="availability:other"),)},
        "evaluation": {"skill_routes": (replace(entry, evaluation_plan_hash="evaluation:other"),)},
        "binding": {"skill_routes": (replace(entry, deployment_binding=replace(
            entry.deployment_binding, deployment_binding_hash="binding:other")),)},
        "route": {"skill_routes": (replace(entry, route=replace(
            entry.route, max_operation_duration_ms=4_000)),)},
    }
    same = replace(manifest)
    other = replace(manifest, **changes[changed_field])
    assert same == manifest
    assert same.semantic_content_hash == identity
    assert other.semantic_content_hash != identity
    assert manifest.semantic_content_hash == identity


def test_manifest_identity_is_not_changed_by_exported_wire_or_input_lists() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import FrozenInstanceError

    definition, schema_entry, definitions, schemas = _catalogs()
    original = _manifest(_route(definition, schema_entry), definitions, schemas)
    entries = list(original.skill_routes)
    manifest = replace(original, skill_routes=entries)
    entries.clear()
    with ThreadPoolExecutor(max_workers=4) as workers:
        identities = tuple(workers.map(lambda _: manifest.semantic_content_hash, range(16)))
    assert len(set(identities)) == 1
    wire = manifest.to_wire()
    wire["skill_routes"][0]["availability_hash"] = "availability:mutated"
    wire["skill_routes"][0]["route"]["resource_claims"].clear()
    wire["supported_transport_features"].clear()
    assert manifest.semantic_content_hash == identities[0]
    assert manifest.to_wire() == original.to_wire()
    with pytest.raises(FrozenInstanceError):
        manifest.runtime_session_id = "runtime:mutated"


def test_manifest_closes_over_definition_schema_profile_and_binding_hashes() -> None:
    definition, schema_entry, definitions, schemas = _catalogs()
    route = _route(definition, schema_entry)
    manifest = _manifest(route, definitions, schemas)
    validate_skill_route_catalog_closure(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
    )
    assert manifest.route(
        definition.skill_id,
        definition.skill_version,
        route.route_id,
        route.route_version,
    ).route == route

    drifted = replace(
        manifest,
        skill_definition_catalog_hash="definition-catalog:stale",
    )
    with pytest.raises(ValueError, match="definition catalog hash mismatch"):
        validate_skill_route_catalog_closure(
            definition_catalog=definitions,
            schema_catalog=schemas,
            route_manifest=drifted,
        )


def test_route_has_one_effect_class_and_no_agent_visible_lower_permit() -> None:
    definition, schema_entry, _, _ = _catalogs()
    route = _route(definition, schema_entry)
    assert route.control_effect_class == "motion"
    assert not hasattr(route, "required_lower_permit_kind")
    with pytest.raises(ValueError, match="Motion Routes"):
        replace(route, may_move_robot=False)


def test_read_only_route_cannot_select_effectful_tier_or_hold_cancellation() -> None:
    definition, schema_entry, _, _ = _catalogs()
    motion_route = _route(definition, schema_entry)
    read_only = replace(
        motion_route,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        may_move_robot=False,
        cancellation_policy="cancel_only",
    )

    with pytest.raises(ValueError, match="t0_read_only and cancel_only"):
        replace(read_only, privilege_tier="t1_sim_or_no_motion")
    with pytest.raises(ValueError, match="t0_read_only and cancel_only"):
        replace(read_only, cancellation_policy="hold_on_cancel")


def test_catalog_closure_rejects_definition_route_effect_mismatch() -> None:
    definition, schema_entry, definitions, schemas = _catalogs()
    motion_route = _route(definition, schema_entry)
    read_only = replace(
        motion_route,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        may_move_robot=False,
        cancellation_policy="cancel_only",
    )
    manifest = _manifest(read_only, definitions, schemas)

    with pytest.raises(ValueError, match="motion.*control effect 'motion'"):
        validate_skill_route_catalog_closure(
            definition_catalog=definitions,
            schema_catalog=schemas,
            route_manifest=manifest,
        )


def test_frozen_lifecycle_routes_reject_ambiguous_or_wrong_effects() -> None:
    definition = replace(
        _definition(),
        skill_id="lifecycle.rearm_safety_latch",
        skill_kind="lifecycle",
        controlled_component_ids=(),
    )
    schema_entry = _schema_entry(definition)
    base_definition = _definition()
    base_schema_entry = _schema_entry(base_definition)
    route = replace(
        _route(base_definition, base_schema_entry),
        skill_id=definition.skill_id,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema_entry.parameter_profiles,
        route_id="real.rm65.lifecycle_controller",
        control_effect_class="lifecycle",
        privilege_tier="t3_lifecycle",
        supported_modes=("real_observe",),
        target_kind="real",
        may_move_robot=False,
        requires_operator_confirmation=True,
    )
    assert route.skill_id == "lifecycle.rearm_safety_latch"
    with pytest.raises(ValueError, match="frozen lifecycle semantics"):
        replace(
            route,
            supported_modes=("real_active",),
            may_open_publication=True,
            requires_deadman=True,
        )
