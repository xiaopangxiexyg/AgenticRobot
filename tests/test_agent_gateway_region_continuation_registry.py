"""Offline region-version admission and result closure through real Gateway owners.

Only the depth sample and query answers are synthetic. Registration, access
policy, exact executor selection, Artifact CAS/pinning, producer operations,
passive workers and contextual result validation are real. No native session,
simulator, model client or hardware is constructed.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_agent_gateway_region_continuation_query import (
    _parameters,
    _path,
    _registration,
)
from test_agent_gateway_service import (
    _AuthoritativeStationPort,
    _ManualClock,
    _policy,
    _request,
    _snapshot,
)
from test_agent_gateway_surface_anchor import _json_bytes, anchor_case  # noqa: F401
from test_agent_gateway_tcp_continuation_query import _select, _success

from agent_gateway.profiles.region_continuation_query import (
    build_region_continuation_executor,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.protocol.skill_catalog import SkillRouteManifest
from agent_gateway.protocol.skill_registration import build_skill_registration_catalogs
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker
from agent_gateway.runtime.event_journal import BoundedAgentGatewayEventJournal
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorRegistry
from agent_gateway.runtime.protocol_runtime import AgentGatewayProtocolRuntime
from agent_gateway.runtime.service import AgentGatewayService
from agent_gateway.skills.perception.relation_continuation_query import (
    POSES_KEY,
    explicit_continuation_parameters,
)
from agent_gateway.skills.perception.surface_anchor_v2 import SELECTION_FIELDS
from dataset_contracts.perception_artifacts import SURFACE_ANCHOR_SCHEMA_ID_V2

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture
def region_case(request):
    anchor_fixture = request.getfixturevalue("anchor_case")
    version = request.param
    capture, registration = _registration(version)
    _, other = _registration("5" if version == "4" else "4")
    catalogs = build_skill_registration_catalogs((capture.skill, registration.skill))
    binding = capture.entry.deployment_binding
    manifest = SkillRouteManifest(
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        station_manifest_hash="station:test",
        execution_manifest_hash=binding.execution_manifest_hash,
        skill_definition_catalog_hash=catalogs.definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=catalogs.schema_catalog.semantic_content_hash,
        skill_io_catalog_hash=catalogs.io_catalog.semantic_content_hash,
        target_binding_id=binding.target_binding_id,
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=("idempotency_v1", "poll_status_v1"),
        skill_routes=(capture.entry, registration.entry),
        explicit_exclusions=("motion", "publication", "device_io"),
    )
    clock = _ManualClock()
    # Include the real capture contract's optional image/mask slots, even
    # though exact-pixel capture emits only the small JSON anchor here.
    artifact_limit = max(
        slot.max_artifact_bytes
        for contract in catalogs.io_catalog.io_contracts
        for slot in (*contract.input_artifact_slots, *contract.output_artifact_slots)
    )
    store = RecoverableAgentArtifactStore(
        root=request.getfixturevalue("tmp_path") / "region-service-cas",
        quota_bytes=2 * artifact_limit,
        max_artifact_bytes=artifact_limit,
    )
    capabilities = AgentGatewaySessionCapabilities(
        runtime_session_id=manifest.runtime_session_id,
        artifact_transfer_enabled=True,
        artifact_access_mechanisms=("chunked_rpc",),
        max_control_request_bytes=131072,
        max_control_response_bytes=131072,
        max_artifact_bytes=store.max_artifact_bytes,
        max_artifact_chunk_bytes=1024,
        max_artifact_refs_per_operation=8,
        max_concurrent_uploads=1,
        max_concurrent_downloads=1,
        artifact_store_quota_bytes=store.quota_bytes,
        event_payload_max_bytes=65536,
        event_retention_count=64,
    )
    runtime = AgentGatewayProtocolRuntime(
        manifest=manifest,
        io_catalog=catalogs.io_catalog,
        session_capabilities=capabilities,
        event_journal=BoundedAgentGatewayEventJournal(
            runtime_session_id=manifest.runtime_session_id,
            max_events=64,
            event_payload_max_bytes=65536,
        ),
        artifact_store=store,
        artifact_transfer_broker=AgentArtifactTransferBroker(
            store=store, session_capabilities=capabilities, max_chunk_bytes=1024
        ),
        clock=clock,
    )
    state = SimpleNamespace(
        answer_version=version, center_selected=False, calls=[], paths=[]
    )

    def capture_read(original, operation):
        selection = dict.fromkeys(SELECTION_FIELDS)
        selection.update(
            mode="exact",
            requested_pixel_uv=[2, 1],
            language_prompt=None,
            min_score=0.25,
            algorithm_id="exact_pixel.v1",
            input_rgb_content_hash="c" * 64,
        )
        anchor = {
            **deepcopy(anchor_fixture.payload),
            "source_operation_id": operation.operation_id,
            "source_request_hash": original.semantic_content_hash,
            "selection": selection,
            "mask_geometry": None,
        }
        data = _json_bytes(anchor)
        artifact = artifact_ref_for_bytes(
            data,
            media_type="application/json",
            schema_id=SURFACE_ANCHOR_SCHEMA_ID_V2,
            encoding="utf-8",
            compression="none",
        )
        commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id=capture.skill.definition.skill_id,
            runtime_session_id=manifest.runtime_session_id,
            idempotency_key=original.semantic_content_hash,
            now=clock(),
        )
        output = AgentArtifactBinding(
            slot_id="surface_anchor",
            ordinal=0,
            artifact=artifact,
            production_receipt_ref="fixture:anchor-production:"
            + original.semantic_content_hash,
        )
        return PassiveReadResult(
            anchor, ("fixture:depth-cut",), output_artifact_bindings=(output,)
        )

    capture_executor = PassiveReadSkillExecutor(
        port_id=capture.entry.route.route_id,
        executor_id="fixture:anchor-v2",
        read=capture_read,
        current_station_revision=lambda: 7,
        clock=clock,
    )

    def query(parameters, *, cancel_requested):
        assert not cancel_requested()
        state.calls.append(deepcopy(parameters))
        path = _path(
            {k: v for k, v in parameters.items() if k != "anchor_xyz_m"},
            parameters["anchor_xyz_m"],
            version=state.answer_version,
        )
        if state.center_selected:
            # Both samplers share the centre. This isolates rejection of the
            # complete proposal list, not a different selected first target.
            path["candidate_diagnostics"][0] = _success(0, cost=(0.1, 0.2, 0.3))
            _select(
                path, explicit_continuation_parameters(parameters, path[POSES_KEY]), 0
            )
        state.paths.append(deepcopy(path))
        return path

    method = (
        "query_region_continuations"
        if version == "4"
        else "query_region_continuation_extrema"
    )
    service = None

    def lookup(operation_id):
        operation = service.get_operation(operation_id)
        request = service.operation_store.request_for_operation(operation_id)
        return None if operation is None or request is None else (request, operation)

    executor = build_region_continuation_executor(
        registration=registration,
        capture_registration=capture,
        query_port=SimpleNamespace(**{method: query}),
        store=store,
        lookup_operation=lookup,
        current_station_revision=lambda: 7,
        clock=clock,
        clock_domain_id="clock:test",
        max_retained_operations=8,
    )
    registry = SkillExecutorRegistry(
        manifest=manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=manifest, entry=item.entry, execution_port=port
            )
            for item, port in ((capture, capture_executor), (registration, executor))
        ),
    )
    domain = _AuthoritativeStationPort(clock)
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(mode="sim_active", allowed_client_ids=("client:test",)),
        executor_registry=registry,
        domain_control_port=domain,
        payload_validators=AgentSkillPayloadValidatorRegistry(
            registry_id="fixture:region-version-validators",
            validators=(capture.payload_validator, registration.payload_validator),
        ),
        protocol_runtime=runtime,
        clock=clock,
        clock_domain_id="clock:test",
    )
    snapshot = replace(
        _snapshot(manifest, station_revision=7), clock_domain_id="clock:test"
    )
    assert service.publish_observation(snapshot=snapshot).accepted
    source_request = replace(
        _request(manifest, capture.entry.route, client_id="client:test"),
        expected_station_revision=7,
        parameters={
            "role": "main",
            "selection_mode": "exact",
            "target_pixel_uv": [2, 1],
            "language_prompt": None,
            "min_score": 0.25,
        },
    )
    try:
        source = service.submit_request(
            request=source_request, current_station_revision=7
        )
        assert source.accepted and source.operation is not None, source.reason_code
        assert service.launch_operation(source.operation.operation_id).accepted
        assert capture_executor.wait(source.operation.operation_id, timeout_sec=1.0)
        completed = service.get_operation(source.operation.operation_id)
        assert completed.status == "completed", completed.progress_summary
        original = replace(
            _request(
                manifest,
                registration.entry.route,
                client_id="client:test",
                sequence=2,
                key="idem:region",
                request_id="request:region",
            ),
            expected_station_revision=7,
            parameters=_parameters(),
            input_artifact_bindings=tuple(
                replace(output, production_receipt_ref=None)
                for output in completed.output_artifact_bindings
            ),
        )
        assert len(original.input_artifact_bindings) == 1
        yield SimpleNamespace(
            version=version,
            registration=registration,
            other=other,
            manifest=manifest,
            service=service,
            registry=registry,
            executor=executor,
            request=original,
            state=state,
            anchor=completed.execution_result,
            domain=domain,
        )
    finally:
        assert capture_executor.wait_all(timeout_sec=1.0)
        assert executor.wait_all(timeout_sec=1.0)
        service.close(current_station_revision=7)
        store.close()


def _run_query(case, original=None):
    original = case.request if original is None else original
    admitted = case.service.submit_request(request=original, current_station_revision=7)
    assert admitted.accepted and admitted.operation is not None, admitted.reason_code
    assert case.service.launch_operation(admitted.operation.operation_id).accepted
    assert case.executor.wait(admitted.operation.operation_id, timeout_sec=1.0)
    return case.service.get_operation(admitted.operation.operation_id)


@pytest.mark.parametrize("region_case", ["4", "5"], indirect=True)
def test_exact_selected_region_route_executes_with_real_artifact_and_service(
    region_case,
):
    case = region_case
    assert case.registry.execution_port_for_request(case.request) is case.executor
    completed = _run_query(case)
    assert completed.status == completed.execution_outcome == "completed"
    assert len(case.state.calls) == 1
    assert completed.task_outcome == "not_assessed"
    assert tuple(receipt.operation_kind for receipt in case.executor.port_receipts) == (
        "activate",
    )
    assert all(
        not receipt.proves_motion_or_hold for receipt in case.executor.port_receipts
    )
    assert completed.output_artifact_bindings == ()
    assert case.domain.calls == []
    result = completed.execution_result
    assert result["surface_anchor"] == case.anchor
    assert result["path_query"]["motion_executed"] is False
    assert result["path_query"]["execution_authorized"] is False
    case.registration.payload_validator.validate_execution_result_for_parameters(
        case.request.parameters, result
    )


@pytest.mark.parametrize("region_case", ["4", "5"], indirect=True)
@pytest.mark.parametrize(
    "field,reason",
    [
        ("skill_version", "skill_route_not_advertised"),
        ("route_version", "skill_route_not_advertised"),
        ("route_id", "skill_route_not_advertised"),
        ("skill_definition_hash", "skill_definition_hash_mismatch"),
        ("route_hash", "route_hash_mismatch"),
        ("deployment_binding_hash", "deployment_binding_hash_mismatch"),
        ("parameter_profile_id", "parameter_profile_not_allowed"),
        ("parameter_profile_hash", "parameter_profile_not_allowed"),
    ],
)
def test_mixed_v4_v5_identity_is_rejected_before_query_port(region_case, field, reason):
    case = region_case
    other_route = case.other.entry.route
    foreign = {
        "skill_version": other_route.skill_version,
        "route_version": other_route.route_version,
        "route_id": other_route.route_id,
        "skill_definition_hash": other_route.skill_definition_hash,
        "route_hash": other_route.semantic_content_hash,
        "deployment_binding_hash": case.other.entry.deployment_binding.deployment_binding_hash,
        "parameter_profile_id": other_route.allowed_parameter_profiles[
            0
        ].parameter_profile_id,
        "parameter_profile_hash": other_route.allowed_parameter_profiles[
            0
        ].parameter_profile_hash,
    }[field]
    assert foreign != getattr(case.request, field)
    invalid = replace(case.request, **{field: foreign})
    if field.startswith("parameter_profile"):
        # Profile admission belongs to access_policy, not the executor key.
        assert case.registry.execution_port_for_request(invalid) is case.executor
    else:
        with pytest.raises(KeyError, match="exact Route/deployment"):
            case.registry.execution_port_for_request(invalid)
    before = case.service.operation_store.operations()
    rejected = case.service.submit_request(request=invalid, current_station_revision=7)
    assert not rejected.accepted and rejected.reason_code == reason
    assert rejected.operation is None
    assert case.service.operation_store.operations() == before
    assert case.state.calls == [] and case.domain.calls == []
    # The identical authenticated Artifact succeeds when only the identity is
    # corrected. A missing input must not be the reason the negative passed.
    assert _run_query(case).status == "completed"
    assert len(case.state.calls) == 1


@pytest.mark.parametrize("region_case", ["4", "5"], indirect=True)
def test_registry_refuses_foreign_version_executor_binding(region_case):
    case = region_case
    foreign = SkillExecutorBinding.from_manifest_entry(
        manifest=case.manifest, entry=case.other.entry, execution_port=case.executor
    )
    capture_binding = next(
        b
        for b in case.registry.bindings
        if b.skill_id == "perception.capture_surface_anchor"
    )
    with pytest.raises(ValueError, match="advertised exact Route"):
        SkillExecutorRegistry(
            manifest=case.manifest, bindings=(capture_binding, foreign)
        )
    assert case.state.calls == []
    assert _run_query(case).status == "completed"


@pytest.mark.parametrize("region_case", ["5"], indirect=True)
def test_v5_accepts_legitimate_zero_delta_sampler_fallback(region_case):
    case = region_case
    case.state.answer_version = "4"
    original = replace(
        case.request,
        parameters={
            **case.request.parameters,
            "continuation_delta_xyz_m": [0.0, 0.0, 0.0],
        },
    )
    completed = _run_query(case, original)
    assert completed.status == "completed" and len(case.state.calls) == 1
    expected = _path(
        original.parameters, case.anchor["measurement"]["world_xyz_m"], version="5"
    )
    assert case.state.paths[0][POSES_KEY] == expected[POSES_KEY]
    case.registration.payload_validator.validate_execution_result_for_parameters(
        original.parameters, completed.execution_result
    )
    assert case.domain.calls == []


@pytest.mark.parametrize("region_case", ["5"], indirect=True)
def test_v5_rejects_complete_v4_proposals_after_port_and_contextual_validation(
    region_case,
):
    case = region_case
    case.state.center_selected = True
    positive = _run_query(case)
    assert positive.status == "completed"
    case.state.answer_version = "4"
    second = replace(
        case.request,
        request_id="request:old-sampler",
        idempotency_key="idem:old-sampler",
        client_sequence=3,
    )
    failed = _run_query(case, second)
    assert len(case.state.calls) == 2
    old_path, new_path = case.state.paths[1], case.state.paths[0]
    assert (
        old_path["first_pose_target_xyz_quat_xyzw"]
        == new_path["first_pose_target_xyz_quat_xyzw"]
    )
    assert old_path["selected_index"] == new_path["selected_index"] == 0
    assert len(old_path[POSES_KEY]) == len(new_path[POSES_KEY]) == 6
    assert old_path[POSES_KEY] != new_path[POSES_KEY]
    old_answer = {**dict(positive.execution_result), "path_query": old_path}
    case.other.payload_validator.validate_execution_result_for_parameters(
        second.parameters, old_answer
    )
    case.registration.payload_validator.validate_execution_result(old_answer)
    with pytest.raises(ValueError, match="Region proposals changed"):
        case.registration.payload_validator.validate_execution_result_for_parameters(
            second.parameters, old_answer
        )
    assert failed.status == failed.execution_outcome == "failed"
    assert failed.reason_code == "passive_read_executor_exception:ValueError"
    assert failed.progress_summary["failure_diagnostic"] == {
        "exception_type": "ValueError",
        "message": "Region proposals changed their bounds, frozen TCP or authenticated anchor geometry.",
    }
    assert failed.execution_result is None and failed.output_artifact_bindings == ()
    assert failed.task_outcome == "not_assessed"
    assert all(
        not receipt.proves_motion_or_hold for receipt in case.executor.port_receipts
    )
    assert case.domain.calls == []
