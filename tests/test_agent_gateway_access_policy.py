from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
)
from agent_gateway.runtime.access_policy import (
    AgentSkillAccessPolicy,
    assess_skill_request_access,
    issue_skill_execution_grant,
)


def _route(
    *,
    skill_id: str = "lifecycle.capture_runtime_home",
    tier: str = "t3_lifecycle",
    mode: str = "real_observe",
    may_move: bool = False,
    confirmation: bool = True,
    deadman: bool = False,
    effect: str = "lifecycle",
    timeout_ms: int = 2_000,
    cancellation_policy: str = "hold_on_disconnect_or_timeout",
) -> SkillExecutionRoute:
    parameter_schema_hash = f"params:{skill_id}:v1"
    return SkillExecutionRoute(
        skill_id=skill_id,
        skill_version="1",
        skill_definition_hash=f"definition:{skill_id}:v1",
        route_id="real.rm65.lifecycle_controller",
        route_version="1",
        parameter_schema_hash=parameter_schema_hash,
        execution_result_schema_hash=f"execution-result:{skill_id}:v1",
        task_evaluation_schema_hash=f"task-evaluation:{skill_id}:v1",
        allowed_parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="safe-v1",
                parameter_profile_hash=f"profile:{skill_id}:safe-v1",
                parameter_schema_hash=parameter_schema_hash,
            ),
        ),
        control_effect_class=effect,
        privilege_tier=tier,
        supported_modes=(mode,),
        target_kind="real",
        may_move_robot=may_move,
        may_open_publication=False,
        requires_operator_confirmation=confirmation,
        requires_deadman=deadman,
        max_operation_duration_ms=timeout_ms,
        cancellation_policy=cancellation_policy,
    )


def _read_only_route() -> SkillExecutionRoute:
    return _route(
        skill_id="inspection.station_state",
        tier="t0_read_only",
        confirmation=False,
        effect="read_only",
        timeout_ms=1_000,
        cancellation_policy="cancel_only",
    )


def _manifest(
    route: SkillExecutionRoute,
    *,
    mode: str = "real_observe",
) -> SkillRouteManifest:
    return SkillRouteManifest(
        runtime_session_id="runtime:test",
        control_domain_id="domain:arms",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        skill_definition_catalog_hash="definition-catalog:test",
        skill_schema_catalog_hash="schema-catalog:test",
        target_binding_id="target:rm65",
        target_kind="real",
        current_mode=mode,
        supported_transport_features=("idempotency_v1",),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:rm65:test",
                    deployment_binding_hash="binding:rm65:test",
                    target_binding_id="target:rm65",
                    target_kind="real",
                    execution_manifest_hash="execution:test",
                ),
                evaluation_plan_hash=None,
                availability_hash="availability:ready",
            ),
        ),
        explicit_exclusions=("raw_action_submission",),
    )


def _policy(*, mode: str = "real_observe") -> AgentSkillAccessPolicy:
    return AgentSkillAccessPolicy(
        access_policy_id="agent-access:test",
        allowed_client_ids=("rpent:test",),
        allowed_modes=(mode,),
        max_request_timeout_ms=1_500,
        max_grant_lifetime_sec=30.0,
    )


def _request(
    manifest: SkillRouteManifest,
    route: SkillExecutionRoute,
    *,
    grant_id: str | None = None,
    timeout_ms: int = 1_000,
) -> AgentSkillRequest:
    entry = manifest.route(
        route.skill_id,
        route.skill_version,
        route.route_id,
        route.route_version,
    )
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="rpent:test",
        request_id="request:test:1",
        idempotency_key="idempotency:test:1",
        client_sequence=1,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=4,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={},
        observation_snapshot_id="snapshot:test:4",
        requested_mode=manifest.current_mode,
        requested_timeout_ms=timeout_ms,
        skill_execution_grant_id=grant_id,
    )


def _grant(
    manifest: SkillRouteManifest,
    route: SkillExecutionRoute,
    *,
    issued_at: float = 10.0,
    expires_at: float = 20.0,
):
    request = _request(manifest, route)
    profile = route.allowed_parameter_profiles[0]
    return issue_skill_execution_grant(
        access_policy=_policy(mode=manifest.current_mode),
        manifest=manifest,
        route=route,
        request=request,
        operator_authorization_receipt_id="operator-confirmation:test",
        client_id="rpent:test",
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        max_invocations=1,
        issued_at=issued_at,
        expires_at=expires_at,
    )


def _sim_motion_surface(
    *,
    with_selection_feature: bool,
) -> tuple[SkillExecutionRoute, SkillRouteManifest]:
    route = SkillExecutionRoute(
        skill_id="motion.move_tcp_to_pose",
        skill_version="1",
        skill_definition_hash="definition:motion.move_tcp_to_pose:v1",
        route_id="sim.mujoco.right_arm_tcp_pose",
        route_version="1",
        parameter_schema_hash="params:motion.move_tcp_to_pose:v1",
        execution_result_schema_hash="execution-result:motion.move_tcp_to_pose:v1",
        task_evaluation_schema_hash="task-evaluation:motion.move_tcp_to_pose:v1",
        allowed_parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="bounded-v1",
                parameter_profile_hash="profile:motion.move_tcp_to_pose:bounded-v1",
                parameter_schema_hash="params:motion.move_tcp_to_pose:v1",
            ),
        ),
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=2_000,
        cancellation_policy="hold_on_disconnect_or_timeout",
    )
    refs = (
        (
            SkillDeploymentSafetyEnvelopeRef(
                object_id="sim.tcp.standard@1",
                schema="dexteleop.deployment_safety_envelope.v1",
                content_hash="envelope-hash:standard",
                constraint_set_hash="constraints:standard",
                allowed_route_ids=(route.route_id,),
            ),
        )
        if with_selection_feature
        else ()
    )
    manifest = SkillRouteManifest(
        runtime_session_id="runtime:sim-motion-test",
        control_domain_id="domain:right-arm",
        station_manifest_hash="station:sim-motion-test",
        execution_manifest_hash="execution:sim-motion-test",
        skill_definition_catalog_hash="definition-catalog:sim-motion-test",
        skill_schema_catalog_hash="schema-catalog:sim-motion-test",
        target_binding_id="target:mujoco",
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=(
            ("idempotency_v1", "safety_profile_selection_v1")
            if with_selection_feature
            else ("idempotency_v1",)
        ),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:mujoco:test",
                    deployment_binding_hash="binding:mujoco:test",
                    target_binding_id="target:mujoco",
                    target_kind="sim",
                    execution_manifest_hash="execution:sim-motion-test",
                    safety_envelope_refs=refs,
                ),
                evaluation_plan_hash=None,
                availability_hash="availability:ready",
            ),
        ),
        explicit_exclusions=("raw_action_submission",),
    )
    return route, manifest


def test_t0_request_is_eligible_without_operator_or_motion_authority() -> None:
    route = _read_only_route()
    manifest = _manifest(route)
    decision = assess_skill_request_access(
        policy=_policy(),
        manifest=manifest,
        request=_request(manifest, route),
        current_station_revision=4,
        now=12.0,
    )
    assert decision.accepted
    assert decision.reason_code == "accepted"
    assert decision.grant_hash is None
    assert decision.route_manifest_entry_hash is not None
    assert decision.effective_timeout_ms == 1_000


def test_t3_request_requires_an_exact_operator_derived_grant() -> None:
    route = _route()
    manifest = _manifest(route)
    policy = _policy()
    request = _request(manifest, route)
    missing = assess_skill_request_access(
        policy=policy,
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=12.0,
    )
    assert not missing.accepted
    assert missing.reason_code == "skill_execution_grant_required"

    grant = _grant(manifest, route)
    accepted = assess_skill_request_access(
        policy=policy,
        manifest=manifest,
        request=replace(request, skill_execution_grant_id=grant.grant_id),
        current_station_revision=4,
        now=12.0,
        grant=grant,
    )
    assert accepted.accepted
    assert accepted.grant_hash == grant.semantic_content_hash
    assert grant.grant_id == f"agent-skill-grant:{grant.semantic_content_hash}"
    assert grant.to_wire()["grant_hash"] == grant.semantic_content_hash


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"client_id": "rpent:other"}, "grant_client_mismatch"),
        ({"runtime_session_id": "runtime:old"}, "grant_runtime_session_mismatch"),
        ({"control_domain_id": "domain:other"}, "grant_control_domain_mismatch"),
        ({"target_binding_id": "target:other"}, "grant_target_binding_mismatch"),
        ({"route_hash": "route:other"}, "grant_route_hash_mismatch"),
        (
            {"deployment_binding_hash": "binding:other"},
            "grant_deployment_binding_mismatch",
        ),
        (
            {"parameter_profile_hash": "profile:other"},
            "grant_parameter_profile_mismatch",
        ),
    ),
)
def test_grant_scope_cannot_be_reused_across_route_or_target(
    change: dict[str, object],
    reason: str,
) -> None:
    route = _route()
    manifest = _manifest(route)
    base_grant = _grant(manifest, route)
    mismatched_grant = replace(base_grant, **change)
    request = replace(
        _request(manifest, route),
        skill_execution_grant_id=mismatched_grant.grant_id,
    )
    decision = assess_skill_request_access(
        policy=_policy(),
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=12.0,
        grant=mismatched_grant,
    )
    assert not decision.accepted
    assert decision.reason_code == reason


def test_grant_is_half_open_and_has_a_single_invocation_budget() -> None:
    route = _route()
    manifest = _manifest(route)
    grant = _grant(manifest, route)
    request = replace(
        _request(manifest, route),
        skill_execution_grant_id=grant.grant_id,
    )
    assert grant.active_at(10.0)
    assert not grant.active_at(20.0)

    expired = assess_skill_request_access(
        policy=_policy(),
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=20.0,
        grant=grant,
    )
    assert expired.reason_code == "skill_execution_grant_inactive"

    exhausted = assess_skill_request_access(
        policy=_policy(),
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=12.0,
        grant=grant,
        grant_invocations_used=1,
    )
    assert exhausted.reason_code == "skill_execution_grant_exhausted"


def test_operator_confirmed_grant_binds_exact_request_parameters() -> None:
    route = _route()
    manifest = _manifest(route)
    grant = _grant(manifest, route)
    changed = replace(
        _request(manifest, route),
        parameters={"capture_side": "left"},
        skill_execution_grant_id=grant.grant_id,
    )
    decision = assess_skill_request_access(
        policy=_policy(),
        manifest=manifest,
        request=changed,
        current_station_revision=4,
        now=12.0,
        grant=grant,
    )
    assert not decision.accepted
    assert decision.reason_code == "grant_authorized_request_mismatch"


def test_grant_issuance_enforces_policy_lifetime_and_invocation_caps() -> None:
    route = _route()
    manifest = _manifest(route)
    request = _request(manifest, route)
    profile = route.allowed_parameter_profiles[0]
    kwargs = {
        "access_policy": _policy(),
        "manifest": manifest,
        "route": route,
        "request": request,
        "operator_authorization_receipt_id": "operator-confirmation:test",
        "client_id": "rpent:test",
        "parameter_profile_id": profile.parameter_profile_id,
        "parameter_profile_hash": profile.parameter_profile_hash,
        "issued_at": 10.0,
    }
    with pytest.raises(ValueError, match="lifetime exceeds"):
        issue_skill_execution_grant(
            **kwargs,
            max_invocations=1,
            expires_at=41.0,
        )
    with pytest.raises(ValueError, match="max_invocations exceeds"):
        issue_skill_execution_grant(
            **kwargs,
            max_invocations=2,
            expires_at=20.0,
        )


@pytest.mark.parametrize(
    ("request_change", "policy_change", "reason"),
    (
        ({"expected_station_revision": 3}, {}, "stale_station_revision"),
        ({"requested_timeout_ms": 1_501}, {}, "timeout_exceeds_access_policy"),
        (
            {"requested_timeout_ms": 1_001},
            {"max_request_timeout_ms": 5_000},
            "timeout_exceeds_route",
        ),
        (
            {"skill_route_manifest_hash": "manifest:old"},
            {},
            "skill_route_manifest_hash_mismatch",
        ),
        ({"runtime_session_id": "runtime:old"}, {}, "runtime_session_mismatch"),
        ({"route_hash": "route:old"}, {}, "route_hash_mismatch"),
        (
            {"deployment_binding_hash": "binding:old"},
            {},
            "deployment_binding_hash_mismatch",
        ),
    ),
)
def test_request_revision_route_hash_session_and_timeout_fail_closed(
    request_change: dict[str, object],
    policy_change: dict[str, object],
    reason: str,
) -> None:
    route = _read_only_route()
    manifest = _manifest(route)
    request = replace(_request(manifest, route), **request_change)
    policy = replace(_policy(), **policy_change)
    decision = assess_skill_request_access(
        policy=policy,
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=12.0,
    )
    assert not decision.accepted
    assert decision.reason_code == reason


def test_policy_cannot_disable_t2_or_t3_grant_requirements() -> None:
    with pytest.raises(ValueError, match="cannot disable grants"):
        replace(_policy(), grant_required_tiers=())


def test_real_motion_grant_still_only_yields_application_access() -> None:
    route = _route(
        skill_id="motion.execute_policy_task",
        tier="t2_real_motion",
        mode="real_active",
        may_move=True,
        confirmation=False,
        effect="motion",
    )
    route = replace(
        route,
        route_id="real.rm65.openpi",
        cancellation_policy="hold_on_disconnect_or_timeout",
    )
    manifest = _manifest(route, mode="real_active")
    policy = _policy(mode="real_active")
    grant = _grant(manifest, route)
    request = replace(
        _request(manifest, route),
        skill_execution_grant_id=grant.grant_id,
    )
    decision = assess_skill_request_access(
        policy=policy,
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=12.0,
        grant=grant,
    )
    assert decision.accepted
    assert not hasattr(decision, "dispatch_permit")
    assert not hasattr(grant, "operator_control_lease")
    assert not hasattr(route, "required_lower_permit_kind")


def test_sim_motion_accepts_only_an_exact_published_safety_envelope() -> None:
    route, manifest = _sim_motion_surface(with_selection_feature=True)
    selected = replace(
        _request(manifest, route),
        safety_profile_id="safety.standard@1",
        safety_profile_hash="safety-profile-hash:standard",
        deployment_safety_envelope_id="sim.tcp.standard@1",
        deployment_safety_envelope_hash="envelope-hash:standard",
        safety_constraint_set_hash="constraints:standard",
    )
    accepted = assess_skill_request_access(
        policy=_policy(mode="sim_active"),
        manifest=manifest,
        request=selected,
        current_station_revision=4,
        now=12.0,
    )
    assert accepted.accepted

    rejected = assess_skill_request_access(
        policy=_policy(mode="sim_active"),
        manifest=manifest,
        request=replace(
            selected,
            deployment_safety_envelope_hash="envelope-hash:unpublished",
        ),
        current_station_revision=4,
        now=12.0,
    )
    assert not rejected.accepted
    assert rejected.reason_code == "deployment_safety_envelope_not_allowed"


def test_compatibility_manifest_rejects_unadvertised_safety_selection() -> None:
    route, manifest = _sim_motion_surface(with_selection_feature=False)
    selected = replace(
        _request(manifest, route),
        safety_profile_id="safety.standard@1",
        safety_profile_hash="safety-profile-hash:standard",
        deployment_safety_envelope_id="sim.tcp.standard@1",
        deployment_safety_envelope_hash="envelope-hash:standard",
        safety_constraint_set_hash="constraints:standard",
    )

    rejected = assess_skill_request_access(
        policy=_policy(mode="sim_active"),
        manifest=manifest,
        request=selected,
        current_station_revision=4,
        now=12.0,
    )
    assert not rejected.accepted
    assert rejected.reason_code == "safety_profile_selection_not_supported"
