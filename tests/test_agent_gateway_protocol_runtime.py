from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactMaterializationReceipt,
    AgentArtifactRef,
)
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    agent_skill_schema_document_hash,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOCatalog,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import (
    SkillRegistrationTemplate,
    build_skill_registration_catalogs,
)
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker
from agent_gateway.runtime.event_journal import BoundedAgentGatewayEventJournal
from agent_gateway.runtime.operation_port import (
    DeterministicNoIoAgentSkillOperationPort,
)
from agent_gateway.runtime.protocol_runtime import AgentGatewayProtocolRuntime
from agent_gateway.runtime.service import NoIoAgentGatewayService
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter


def _strict_empty_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }


def _registration(
    *,
    skill_id: str = "perception.capture_depth",
) -> tuple[
    SkillRegistrationTemplate,
    SkillDefinitionCatalog,
    AgentSkillSchemaCatalog,
    SkillIOCatalog,
]:
    definition = SkillDefinition(
        skill_id=skill_id,
        skill_version="1",
        skill_kind="inspection",
        summary="Capture one finite depth observation.",
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="depth.capture.parameters.v1",
        execution_result_semantics_id="depth.capture.result.v1",
        task_outcome_semantics_id="depth.capture.task_outcome.v1",
    )
    parameter_schema = _strict_empty_schema()
    profile = SkillParameterProfileRef(
        parameter_profile_id="default-v1",
        parameter_profile_hash="profile:depth-default:v1",
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema,
            schema_kind="parameter",
        ),
    )
    schema_entry = AgentSkillSchemaEntry(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=_strict_empty_schema(),
        task_evaluation_schema=_strict_empty_schema(),
        parameter_profiles=(profile,),
    )
    slot = SkillArtifactSlotSpec(
        slot_id="depth_frame",
        direction="output",
        min_count=1,
        max_count=1,
        allowed_media_types=("application/octet-stream",),
        allowed_schema_ids=("depth.image.u16_mm.v1",),
        allowed_encodings=("identity",),
        allowed_compressions=("none",),
        max_artifact_bytes=1024,
        max_total_bytes=1024,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        output_artifact_slots=(slot,),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id="null.fixture.depth_capture",
        route_version="1",
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=(profile,),
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("dry_run",),
        target_kind="null",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1000,
        cancellation_policy="cancel_only",
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    registration = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema_entry,
        io_contract=io_contract,
        routes=(route,),
    )
    catalogs = build_skill_registration_catalogs((registration,))
    return (
        registration,
        catalogs.definition_catalog,
        catalogs.schema_catalog,
        catalogs.io_catalog,
    )


def test_registration_builder_canonicalizes_input_order() -> None:
    later = _registration(skill_id="test.skill_z")[0]
    earlier = _registration(skill_id="test.skill_a")[0]

    catalogs = build_skill_registration_catalogs((later, earlier))

    expected = ("test.skill_a", "test.skill_z")
    assert tuple(
        item.skill_id for item in catalogs.definition_catalog.skill_definitions
    ) == expected
    assert tuple(item.skill_id for item in catalogs.schema_catalog.schema_entries) == (
        expected
    )
    assert tuple(item.skill_id for item in catalogs.io_catalog.io_contracts) == expected


def test_registration_rejects_inspection_route_with_control_effects() -> None:
    registration = _registration()[0]
    route = registration.routes[0]
    effectful = replace(
        route,
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        may_move_robot=True,
        cancellation_policy="hold_on_disconnect_or_timeout",
    )

    with pytest.raises(ValueError, match="inspection.*read_only"):
        replace(registration, routes=(effectful,))


def _artifact_runtime(tmp_path: Path):
    registration, definitions, schemas, io_catalog = _registration()
    route = registration.routes[0]
    manifest = SkillRouteManifest(
        runtime_session_id="runtime:artifact-test",
        control_domain_id="domain:inspection",
        station_manifest_hash="station:artifact-test",
        execution_manifest_hash="execution:artifact-test",
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        skill_io_catalog_hash=io_catalog.semantic_content_hash,
        target_binding_id="target:null-artifact-test",
        target_kind="null",
        current_mode="dry_run",
        supported_transport_features=(),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:null-depth",
                    deployment_binding_hash="binding:null-depth",
                    target_binding_id="target:null-artifact-test",
                    target_kind="null",
                    execution_manifest_hash="execution:artifact-test",
                ),
                evaluation_plan_hash=None,
                availability_hash="availability:depth-ready",
            ),
        ),
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifact-store",
        quota_bytes=4096,
        max_artifact_bytes=1024,
        max_chunk_bytes=4,
        staging_ttl_sec=60.0,
        minimum_retention_sec=10.0,
        handoff_retention_sec=100.0,
    )
    capabilities = AgentGatewaySessionCapabilities(
        runtime_session_id=manifest.runtime_session_id,
        artifact_transfer_enabled=True,
        artifact_access_mechanisms=("chunked_rpc", "local_path"),
        max_control_request_bytes=2 * 1024 * 1024,
        max_control_response_bytes=2 * 1024 * 1024,
        max_artifact_bytes=1024,
        max_artifact_chunk_bytes=4,
        max_artifact_refs_per_operation=4,
        max_concurrent_uploads=2,
        max_concurrent_downloads=2,
        artifact_store_quota_bytes=4096,
        event_payload_max_bytes=1024,
        event_retention_count=16,
    )
    broker = AgentArtifactTransferBroker(
        store=store,
        session_capabilities=capabilities,
        max_chunk_bytes=4,
    )
    now = iter(float(item) for item in range(1, 100))
    runtime = AgentGatewayProtocolRuntime(
        manifest=manifest,
        io_catalog=io_catalog,
        session_capabilities=capabilities,
        event_journal=BoundedAgentGatewayEventJournal(
            runtime_session_id=manifest.runtime_session_id,
            max_events=16,
            event_payload_max_bytes=1024,
        ),
        artifact_store=store,
        artifact_transfer_broker=broker,
        clock=lambda: next(now),
    )
    service = NoIoAgentGatewayService(
        manifest=manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="access:artifact-test",
            allowed_client_ids=("rpent:local",),
            allowed_modes=("dry_run",),
            max_request_timeout_ms=1000,
            max_grant_lifetime_sec=30.0,
        ),
        operation_port=DeterministicNoIoAgentSkillOperationPort(),
        protocol_runtime=runtime,
    )
    adapter = BoundAgentGatewayWireAdapter(
        authenticated_client_id="rpent:local",
        definition_catalog=definitions,
        schema_catalog=schemas,
        service=service,
        publish_observation=lambda: None,
        current_station_revision=lambda: 0,
    )
    return adapter, store


def test_protocol_runtime_rejects_event_journal_payload_bound_mismatch(
    tmp_path: Path,
) -> None:
    adapter, _store = _artifact_runtime(tmp_path)
    runtime = adapter.service.protocol_runtime

    with pytest.raises(ValueError, match="Event payload bound"):
        AgentGatewayProtocolRuntime(
            manifest=runtime.manifest,
            io_catalog=runtime.io_catalog,
            session_capabilities=runtime.session_capabilities,
            event_journal=BoundedAgentGatewayEventJournal(
                runtime_session_id=runtime.runtime_session_id,
                max_events=runtime.session_capabilities.event_retention_count,
                event_payload_max_bytes=512,
            ),
            artifact_store=runtime.artifact_store,
            artifact_transfer_broker=runtime.artifact_transfer_broker,
        )


def test_registration_template_builds_three_closed_catalogs() -> None:
    registration, definitions, schemas, io_catalog = _registration()
    identity = (
        registration.definition.skill_id,
        registration.definition.skill_version,
    )
    assert definitions.definition(*identity) == registration.definition
    assert schemas.entry(*identity) == registration.schema_entry
    assert io_catalog.contract(*identity) == registration.io_contract


def test_bound_wire_adapter_round_trips_finite_artifact_and_materialization(
    tmp_path: Path,
) -> None:
    adapter, store = _artifact_runtime(tmp_path)
    data = b"depth-bytes"
    artifact = AgentArtifactRef(
        media_type="application/octet-stream",
        schema_id="depth.image.u16_mm.v1",
        encoding="identity",
        compression="none",
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    session = "runtime:artifact-test"
    common = {"client_id": "rpent:local", "runtime_session_id": session}
    try:
        assert adapter.get_gateway_session_capabilities()["artifact_transfer_enabled"]
        assert adapter.get_skill_io_catalog()["skill_io_catalog_hash"]
        begun = adapter.begin_artifact_upload(
            {
                **common,
                "idempotency_key": "upload:depth-one",
                "artifact": artifact.to_wire(),
                "mechanism": "chunked_rpc",
                "ttl_sec": 60.0,
            }
        )
        grant_id = begun["access_grant"]["grant_id"]
        for offset in range(0, len(data), 4):
            adapter.write_artifact_chunk(
                {
                    **common,
                    "grant_id": grant_id,
                    "offset": offset,
                    "data_base64": base64.b64encode(data[offset : offset + 4]).decode(
                        "ascii"
                    ),
                }
            )
        committed = adapter.commit_artifact_upload({**common, "grant_id": grant_id})
        assert committed["artifact"]["artifact_id"] == artifact.artifact_id

        download = adapter.begin_artifact_download(
            {
                **common,
                "artifact_id": artifact.artifact_id,
                "mechanism": "chunked_rpc",
                "ttl_sec": 60.0,
            }
        )
        received = b""
        for offset in range(0, len(data), 4):
            chunk = adapter.read_artifact_chunk(
                {
                    **common,
                    "grant_id": download["grant_id"],
                    "offset": offset,
                    "max_bytes": 4,
                }
            )
            received += base64.b64decode(chunk["data_base64"])
        assert received == data
        adapter.acknowledge_artifact_download(
            {
                **common,
                "grant_id": download["grant_id"],
                "transferred_bytes": len(received),
                "verified_sha256": hashlib.sha256(received).hexdigest(),
            }
        )
        materialized = AgentArtifactMaterializationReceipt(
            receipt_id="receipt:rpent-depth-one",
            artifact_id=artifact.artifact_id,
            client_id="rpent:local",
            runtime_session_id=session,
            durable_store_id="rpent:experience",
            durable_object_ref="experience-artifact:depth-one",
            materialized_at=90.0,
            verified_sha256=artifact.sha256,
        )
        assert (
            adapter.acknowledge_artifact_materialization(materialized.to_wire())
            == materialized.to_wire()
        )
        assert store.artifact(artifact.artifact_id).materialization_receipt_hash == (
            materialized.semantic_content_hash
        )
    finally:
        store.close()
