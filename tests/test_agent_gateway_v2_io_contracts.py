from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactAccessGrant,
    AgentArtifactBinding,
    AgentArtifactMaterializationReceipt,
    AgentArtifactRef,
    AgentArtifactTransferReceipt,
)
from agent_gateway.protocol.event_contracts import (
    AgentGatewayEvent,
    AgentGatewayEventBatch,
)
from agent_gateway.protocol.session_contracts import (
    AgentGatewaySessionCapabilities,
    SkillResourceClaim,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOCatalog,
    SkillIOContract,
    inline_only_skill_io_contract,
)
from agent_gateway.protocol.version import AGENT_SKILL_PROTOCOL_VERSION_V2


_SHA_A = "a" * 64


def _artifact(*, sha256: str = _SHA_A) -> AgentArtifactRef:
    return AgentArtifactRef(
        media_type="image/png",
        schema_id="depth.image.u16_mm.v1",
        encoding="png",
        compression="deflate",
        byte_length=4096,
        sha256=sha256,
    )


def _depth_slot() -> SkillArtifactSlotSpec:
    return SkillArtifactSlotSpec(
        slot_id="depth_frame",
        direction="output",
        min_count=1,
        max_count=1,
        allowed_media_types=("image/png",),
        allowed_schema_ids=("depth.image.u16_mm.v1",),
        allowed_encodings=("png",),
        allowed_compressions=("deflate",),
        max_artifact_bytes=8 * 1024 * 1024,
        max_total_bytes=8 * 1024 * 1024,
    )


def _io_contract(*, skill_id: str = "perception.capture_depth") -> SkillIOContract:
    return SkillIOContract(
        skill_id=skill_id,
        skill_version="1",
        skill_definition_hash="definition:depth",
        parameter_schema_hash="schema:parameters",
        execution_result_schema_hash="schema:result",
        task_evaluation_schema_hash="schema:evaluation",
        output_artifact_slots=(_depth_slot(),),
    )


def _capabilities() -> AgentGatewaySessionCapabilities:
    return AgentGatewaySessionCapabilities(
        runtime_session_id="runtime:test",
        artifact_transfer_enabled=True,
        artifact_access_mechanisms=("chunked_rpc", "local_path"),
        max_control_request_bytes=256 * 1024,
        max_control_response_bytes=256 * 1024,
        max_artifact_bytes=64 * 1024 * 1024,
        max_artifact_chunk_bytes=1024 * 1024,
        max_artifact_refs_per_operation=16,
        max_concurrent_uploads=2,
        max_concurrent_downloads=4,
        artifact_store_quota_bytes=1024 * 1024 * 1024,
        event_payload_max_bytes=64 * 1024,
        event_retention_count=4096,
    )


def test_artifact_ref_is_v2_content_identity_without_a_locator() -> None:
    artifact = _artifact()
    wire = artifact.to_wire()
    assert wire["protocol_version"] == AGENT_SKILL_PROTOCOL_VERSION_V2
    assert wire["artifact_id"] == f"artifact:{artifact.semantic_content_hash}"
    assert AgentArtifactRef.from_wire(wire) == artifact
    assert not {"path", "uri", "url", "locator"}.intersection(wire)

    forged = dict(wire)
    forged["byte_length"] += 1
    with pytest.raises(ValueError, match="artifact_id does not match"):
        AgentArtifactRef.from_wire(forged)


def test_location_is_short_lived_access_state_not_artifact_identity() -> None:
    artifact = _artifact()
    first = AgentArtifactAccessGrant(
        grant_id="grant:one",
        artifact=artifact,
        direction="read",
        mechanism="local_path",
        client_id="rpent:local",
        runtime_session_id="runtime:test",
        issued_at=10.0,
        expires_at=20.0,
        max_uses=1,
        locator={"absolute_path": "/managed/artifacts/a"},
    )
    second = replace(
        first,
        grant_id="grant:two",
        locator={"absolute_path": "/managed/artifacts/b"},
    )
    assert first.artifact == second.artifact
    assert first.artifact.artifact_id == second.artifact.artifact_id
    assert first.semantic_content_hash != second.semantic_content_hash
    assert first.to_wire()["locator"]["absolute_path"].startswith("/managed/")

    with pytest.raises(ValueError, match="remaining Artifact bytes"):
        replace(first, byte_offset=4090, max_bytes=10)


def test_transfer_and_materialization_receipts_have_distinct_ownership() -> None:
    artifact = _artifact()
    transfer = AgentArtifactTransferReceipt(
        receipt_id="receipt:transfer",
        grant_id="grant:one",
        artifact_id=artifact.artifact_id,
        client_id="rpent:local",
        runtime_session_id="runtime:test",
        direction="read",
        status="completed",
        transferred_bytes=artifact.byte_length,
        verified_sha256=artifact.sha256,
        completed_at=12.0,
        reason_code="transfer_verified",
    )
    materialized = AgentArtifactMaterializationReceipt(
        receipt_id="receipt:materialized",
        artifact_id=artifact.artifact_id,
        client_id="rpent:local",
        runtime_session_id="runtime:test",
        durable_store_id="rpent:experience",
        durable_object_ref="experience-artifact:123",
        materialized_at=13.0,
        verified_sha256=artifact.sha256,
    )
    assert transfer.to_wire()["transfer_receipt_hash"]
    assert materialized.to_wire()["materialization_receipt_hash"]
    with pytest.raises(ValueError, match="requires verified_sha256"):
        replace(transfer, verified_sha256=None)


def test_every_skill_has_one_explicit_v2_io_contract() -> None:
    inline = inline_only_skill_io_contract(
        skill_id="state.read_tcp_pose",
        skill_version="1",
        skill_definition_hash="definition:pose",
        parameter_schema_hash="schema:pose-parameters",
        execution_result_schema_hash="schema:pose-result",
        task_evaluation_schema_hash="schema:pose-evaluation",
    )
    depth = _io_contract()
    catalog = SkillIOCatalog(
        catalog_id="dexteleop.agent_skill_io",
        catalog_version="1",
        io_contracts=(depth, inline),
    )
    assert catalog.protocol_version == AGENT_SKILL_PROTOCOL_VERSION_V2
    assert catalog.contract("state.read_tcp_pose", "1") == inline
    assert not inline.input_artifact_slots
    assert not inline.output_artifact_slots
    assert depth.output_artifact_slots == (_depth_slot(),)

    with pytest.raises(ValueError, match="uniquely sorted"):
        replace(catalog, io_contracts=(inline, depth))


def test_artifact_slot_bounds_are_finite() -> None:
    slot = _depth_slot()
    with pytest.raises(ValueError, match="min_count cannot exceed"):
        replace(slot, min_count=2, max_count=1)
    with pytest.raises(ValueError, match="max_total_bytes"):
        replace(slot, max_total_bytes=slot.max_artifact_bytes - 1)


def test_capabilities_are_typed_and_require_a_bounded_fallback() -> None:
    capabilities = _capabilities()
    wire = capabilities.to_wire()
    assert wire["protocol_version"] == AGENT_SKILL_PROTOCOL_VERSION_V2
    assert wire["artifact_access_mechanisms"] == [
        "chunked_rpc",
        "local_path",
    ]
    with pytest.raises(ValueError, match="require chunked_rpc"):
        replace(capabilities, artifact_access_mechanisms=("local_path",))
    with pytest.raises(ValueError, match="cannot exceed artifact_store_quota"):
        replace(
            capabilities,
            max_artifact_bytes=capabilities.artifact_store_quota_bytes + 1,
        )
    with pytest.raises(ValueError, match="protocol event bound"):
        replace(capabilities, event_payload_max_bytes=65_537)
    with pytest.raises(ValueError, match="chunk_bytes"):
        replace(
            capabilities,
            max_artifact_chunk_bytes=capabilities.max_artifact_bytes + 1,
        )

    inline_only = AgentGatewaySessionCapabilities.control_plane_only(
        runtime_session_id="runtime:inline",
        max_control_request_bytes=1024,
        max_control_response_bytes=2048,
    )
    assert not inline_only.artifact_transfer_enabled
    assert inline_only.artifact_access_mechanisms == ()
    assert inline_only.max_artifact_bytes == 0
    with pytest.raises(ValueError, match="empty mechanisms and zero limits"):
        replace(inline_only, max_artifact_bytes=1)


def test_empty_artifact_has_a_zero_length_access_range() -> None:
    artifact = AgentArtifactRef(
        media_type="application/octet-stream",
        schema_id=None,
        encoding="identity",
        compression="none",
        byte_length=0,
        sha256=(
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
    )
    grant = AgentArtifactAccessGrant(
        grant_id="grant:empty",
        artifact=artifact,
        direction="read",
        mechanism="chunked_rpc",
        client_id="rpent:test",
        runtime_session_id="runtime:test",
        issued_at=1.0,
        expires_at=2.0,
        max_uses=1,
        locator={"transfer_id": "grant:empty"},
    )
    assert grant.max_bytes == 0


def test_resource_claims_are_trusted_route_data_not_agent_parameters() -> None:
    claim = SkillResourceClaim(
        resource_id="control.motion",
        access_mode="exclusive",
    )
    assert claim.to_wire()["units"] == 1
    with pytest.raises(ValueError, match="exclusive resource claims"):
        replace(claim, units=2)


def test_events_are_bounded_notifications_with_explicit_cursor_gaps() -> None:
    binding = AgentArtifactBinding(
        slot_id="depth_frame",
        ordinal=0,
        artifact=_artifact(),
        production_receipt_ref="receipt:producer",
    )
    event = AgentGatewayEvent(
        runtime_session_id="runtime:test",
        sequence=8,
        event_kind="artifact_ready",
        emitted_at=10.0,
        operation_id="operation:one",
        operation_hash="operation-hash:one",
        artifact_binding=binding,
        payload_schema_id="artifact.ready.v1",
        payload={"note": "fetch through the Artifact plane"},
    )
    assert event.to_wire()["event_id"].startswith("event:")
    assert "data_base64" not in event.to_wire()["payload"]

    batch = AgentGatewayEventBatch(
        runtime_session_id="runtime:test",
        requested_after_sequence=2,
        oldest_available_sequence=8,
        latest_sequence=8,
        gap_detected=True,
        events=(event,),
    )
    assert batch.to_wire()["gap_detected"] is True

    with pytest.raises(ValueError, match="artifact_ready requires"):
        replace(event, artifact_binding=None)
    with pytest.raises(ValueError, match="control-plane bound"):
        replace(event, payload={"oversized": "x" * 70_000})
    with pytest.raises(ValueError, match="gap_detected does not match"):
        replace(batch, gap_detected=False)
