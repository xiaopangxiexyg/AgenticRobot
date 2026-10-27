"""Candidate Artifact contracts and passive handoff; no simulator or devices."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_agent_gateway_surface_anchor import (
    _relation_parameters,
    _v2_registrations,
    anchor_case,  # noqa: F401 - Register the shared pytest fixture.
)
from test_agent_gateway_tcp_continuation_query import _result

from agent_gateway.profiles import pose_candidate_set as module
from agent_gateway.profiles.tcp_continuation_query import (
    build_tcp_continuation_query_registration,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.skills.perception.object_goal_query import RANKING
from agent_gateway.skills.perception.pose_candidate_set import (
    POSE_CANDIDATES_SCHEMA_ID,
    validate_index_parameters,
    validate_index_result,
)
from agent_gateway.skills.perception.surface_anchor_v2 import SELECTION_FIELDS
from robot_motion.spatial_goals import generate_relation_goals

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _request(registration, parameters, *, bindings=(), sequence=0):
    route, profile = registration.entry.route, registration.entry.route.allowed_parameter_profiles[0]
    return AgentSkillRequest(client_id="client:test", request_id=f"request:{sequence}",
        idempotency_key=f"key:{sequence}", client_sequence=sequence, runtime_session_id="session:test",
        skill_route_manifest_hash="manifest:test", expected_station_revision=7,
        skill_id=route.skill_id, skill_version=route.skill_version, skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id, route_version=route.route_version, route_hash=route.semantic_content_hash,
        deployment_binding_hash=registration.entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters, input_artifact_bindings=bindings, observation_snapshot_id="snapshot:test",
        requested_mode="sim_active", requested_timeout_ms=30_000)


@pytest.fixture
def candidate_case(request):
    anchor_data = request.getfixturevalue("anchor_case")
    projection, _, _, (_capture, _, relation) = _v2_registrations()
    producer = module.build_relation_candidate_registration(relation_registration=relation)
    continuation = build_tcp_continuation_query_registration(base_route=projection.skill.routes[0],
        base_binding=projection.binding, component_id="right_arm", reference_frame_id="canonical_station",
        planning_profile_hash="planning:test")
    consumer = module.build_candidate_continuation_registration(
        continuation_registration=continuation, relation_registration=producer)
    selection = dict.fromkeys(SELECTION_FIELDS)
    selection.update(mode="exact", requested_pixel_uv=[2, 1], language_prompt=None,
        min_score=.25, algorithm_id="exact_pixel.v1", input_rgb_content_hash="c" * 64)
    anchor = {**anchor_data.payload, "selection": selection, "mask_geometry": None}
    params = _relation_parameters()
    params["offsets_xyz_m"] = [[.04, -.03, .14], [.04, -.03, .20], [.06, -.03, .20]]
    source_tcp = [0., 0., 0., 0., 0., 0., 1.]
    poses = generate_relation_goals(current_tcp_pose_xyz_quat_xyzw=source_tcp,
        anchor_xyz_m=anchor["measurement"]["world_xyz_m"],
        **{key: value for key, value in params.items() if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})
    path = {"component_id": "right_arm", "reference_frame_id": "canonical_station",
        "anchor_xyz_m": anchor["measurement"]["world_xyz_m"], "source_tcp_pose_xyz_quat_xyzw": source_tcp,
        "scene_hash": "a" * 64, "source_qpos_hash": "b" * 64, "source_sim_time_sec": 0.,
        "query_status": "no_path_found", "pose_targets_by_component": {}, "candidate_count": len(poses),
        "checked_count": 1, "all_candidates_checked": False, "selected_index": None,
        "candidate_diagnostics": [{"candidate_index": 0, "reason_code": "query_budget_exhausted", "wall_sec": .1}],
        "planning_wall_sec": .1, "ranking_criteria": RANKING, "motion_executed": False,
        "execution_authorized": False, "input_source": "caller_anchor_and_frozen_measured_scene",
        "constraint_scope": "frozen_scene_full_path"}
    result = {"surface_anchor": anchor, "anchor_artifact_id": "artifact:" + "d" * 64,
        "path_query": path, "relation_constraint_scope": "endpoint_only"}
    original = _request(producer, params)
    source = SimpleNamespace(operation_id="operation:source", status="completed", execution_outcome="completed",
        runtime_session_id=original.runtime_session_id, request_hash=original.semantic_content_hash, execution_result=result)
    binding = module.publish_candidate_set(request=original, operation=source, result=result, registration=producer,
        store=anchor_data.kwargs["store"], clock=lambda: 10., clock_domain_id="clock:test")
    source.output_artifact_bindings = (binding,)
    params = {"component_id": "right_arm", "reference_frame_id": "canonical_station", "candidate_indices": [5, 2, 0],
        "continuation_delta_xyz_m": [0, 0, .03], "path_constraint": None, "time_budget_sec": 12.}
    case = SimpleNamespace(producer=producer, consumer=consumer, base_relation=relation, base_continuation=continuation,
        store=anchor_data.kwargs["store"], original=original, source=source, binding=binding,
        revision=7, clock_domain_id="clock:test", poses=[list(item.goal.pose_xyz_quat_xyzw) for item in poses])
    case.request = _request(consumer, params, sequence=1, bindings=(replace(binding, production_receipt_ref=None),))
    case.kwargs = {"request": case.request, "registration": producer, "store": case.store,
        "lookup_operation": lambda _id: (case.original, case.source),
        "current_station_revision": lambda: case.revision, "clock_domain_id": case.clock_domain_id}
    return case


def test_new_versions_preserve_old_contracts_and_expose_one_bounded_slot(candidate_case):
    case = candidate_case
    assert case.base_relation.skill.definition.skill_version == "2"
    assert case.base_continuation.skill.definition.skill_version == "1"
    assert case.producer.skill.definition.skill_version == "3"
    assert case.consumer.skill.definition.skill_version == "2"
    assert case.producer.entry.route.control_effect_class == case.consumer.entry.route.control_effect_class == "read_only"
    assert case.producer.entry.route.cancellation_policy == case.consumer.entry.route.cancellation_policy == "cancel_only"
    assert case.producer.skill.io_contract.output_artifact_slots[0].slot_id == "pose_candidates"
    assert case.consumer.skill.io_contract.input_artifact_slots[0].allowed_schema_ids == (POSE_CANDIDATES_SCHEMA_ID,)
    assert not case.consumer.skill.io_contract.output_artifact_slots
    payload, identity = module.resolve_candidate_set(**case.kwargs)
    assert identity == case.binding.artifact.artifact_id
    assert payload["candidate_poses_xyz_quat_xyzw"] == case.poses
    assert len(payload["candidate_poses_xyz_quat_xyzw"]) == 6
    # Every generated proposal is available, even when only one sample was attempted.
    assert case.source.execution_result["path_query"]["checked_count"] == 1


@pytest.mark.parametrize("indices", [[], [True], [1.0], [-1], [64], [1, 1], list(range(9)), None])
def test_index_parameters_reject_invalid_geometry_selectors(candidate_case, indices):
    params = dict(candidate_case.request.parameters)
    params["candidate_indices"] = indices
    with pytest.raises((TypeError, ValueError)):
        validate_index_parameters(params)


@pytest.mark.parametrize("mutation", ["client", "session", "manifest", "route", "version", "request_hash",
    "revision", "clock", "missing_binding", "failed", "evicted", "forged", "wrong_frame"])
def test_candidate_artifact_rejects_wrong_or_unavailable_provenance(candidate_case, mutation):
    case = candidate_case
    if mutation in {"client", "session", "manifest", "route", "version"}:
        key, value = {"client": ("client_id", "another-client"), "session": ("runtime_session_id", "another-session"),
            "manifest": ("skill_route_manifest_hash", "another-manifest"), "route": ("route_hash", "another-route"),
            "version": ("skill_version", "2")}[mutation]
        case.original = replace(case.original, **{key: value})
    elif mutation == "request_hash":
        case.source.request_hash = "e" * 64
    elif mutation == "revision":
        case.revision = 8
    elif mutation == "clock":
        case.kwargs["clock_domain_id"] = "another-clock"
    elif mutation == "missing_binding":
        case.source.output_artifact_bindings = ()
    elif mutation == "failed":
        case.source.status = "failed"
    elif mutation == "evicted":
        case.kwargs["lookup_operation"] = lambda _id: None
    elif mutation == "wrong_frame":
        case.kwargs["request"] = replace(case.request, parameters={**case.request.parameters, "reference_frame_id": "another-frame"})
    else:
        payload, _ = module.resolve_candidate_set(**case.kwargs)
        payload["candidate_poses_xyz_quat_xyzw"][0][0] += .001
        data = module._bytes(payload)
        artifact = artifact_ref_for_bytes(data, media_type="application/json", schema_id=POSE_CANDIDATES_SCHEMA_ID,
            encoding="utf-8", compression="none")
        commit_artifact_bytes(store=case.store, data=data, artifact=artifact, producer_id="forger",
            runtime_session_id="session:test", idempotency_key="forged-candidate", now=10.)
        forged = AgentArtifactBinding(slot_id="pose_candidates", ordinal=0, artifact=artifact)
        # Even a matching claimed output does not substitute for regenerated producer content.
        case.source.output_artifact_bindings = (forged,)
        case.kwargs["request"] = replace(case.request, input_artifact_bindings=(forged,))
    with pytest.raises((TypeError, ValueError)):
        module.resolve_candidate_set(**case.kwargs)


class _CallbackExecutor:
    def __init__(self, *, read, **_kwargs):
        self.read = read
        self.cancel = False

    def cancel_requested(self, _operation_id):
        return self.cancel


@pytest.mark.parametrize("mode", ["success", "cancel", "deadline", "revision", "index_out_of_range", "changed_target"])
def test_passive_adapter_resolves_exact_indices_and_preserves_rejections(candidate_case, monkeypatch, mode):
    case = candidate_case
    monkeypatch.setattr(module, "PassiveReadSkillExecutor", _CallbackExecutor)
    calls = []

    def query(parameters, *, cancel_requested):
        calls.append(deepcopy(parameters))
        if cancel_requested():
            raise RuntimeError("planning cancelled")
        result = _result(parameters)
        if mode == "revision":
            case.revision = 8
        elif mode == "changed_target":
            result["first_pose_target_xyz_quat_xyzw"][0] += .001
        return result

    executor = module.build_candidate_continuation_executor(registration=case.consumer,
        relation_registration=case.producer, query_port=SimpleNamespace(query_continuations=query), store=case.store,
        lookup_operation=case.kwargs["lookup_operation"], current_station_revision=lambda: case.revision,
        clock=lambda: 10., clock_domain_id="clock:test", max_retained_operations=4)
    executor.cancel = mode == "cancel"
    operation = SimpleNamespace(operation_id="operation:consumer", deadline_at=9. if mode == "deadline" else 30.)
    request = case.request
    if mode == "index_out_of_range":
        request = replace(request, parameters={**request.parameters, "candidate_indices": [6]})
    if mode != "success":
        with pytest.raises((RuntimeError, ValueError)):
            executor.read(request, operation)
        assert len(calls) == (0 if mode == "index_out_of_range" else 1)
        return
    answer = executor.read(request, operation)
    result = answer.execution_result
    validate_index_result(result)
    case.consumer.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
    assert calls[0]["first_pose_candidates_xyz_quat_xyzw"] == [case.poses[i] for i in (5, 2, 0)]
    assert result["selected_index"] == 1 and result["selected_source_index"] == 2
    assert result["first_pose_target_xyz_quat_xyzw"] == case.poses[2]
    assert result["motion_executed"] is False and result["execution_authorized"] is False
    assert answer.output_artifact_bindings == ()
    assert not any("hold" in ref.lower() for ref in answer.evidence_refs)
    assert json.loads(module._bytes(result))["candidate_indices"] == [5, 2, 0]
