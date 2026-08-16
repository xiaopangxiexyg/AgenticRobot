from __future__ import annotations

import hashlib

import pytest

from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactBinding,
    AgentArtifactRef,
)
from agent_gateway.protocol.session_contracts import (
    AgentGatewaySessionCapabilities,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.runtime.artifact_validation import (
    validate_skill_artifact_bindings,
)


def _artifact(*, byte_length: int = 100) -> AgentArtifactRef:
    return AgentArtifactRef(
        media_type="image/png",
        schema_id="depth.image.u16_mm.v1",
        encoding="png",
        compression="deflate",
        byte_length=byte_length,
        sha256=hashlib.sha256(b"x" * byte_length).hexdigest(),
    )


def _slot(*, direction: str) -> SkillArtifactSlotSpec:
    return SkillArtifactSlotSpec(
        slot_id="depth_frame",
        direction=direction,
        min_count=1,
        max_count=2,
        allowed_media_types=("image/png",),
        allowed_schema_ids=("depth.image.u16_mm.v1",),
        allowed_encodings=("png",),
        allowed_compressions=("deflate",),
        max_artifact_bytes=1000,
        max_total_bytes=1500,
    )


def _contract(*, direction: str) -> SkillIOContract:
    slot = _slot(direction=direction)
    return SkillIOContract(
        skill_id="perception.capture_depth",
        skill_version="1",
        skill_definition_hash="definition:depth",
        parameter_schema_hash="schema:parameters",
        execution_result_schema_hash="schema:result",
        task_evaluation_schema_hash="schema:evaluation",
        input_artifact_slots=(slot,) if direction == "input" else (),
        output_artifact_slots=(slot,) if direction == "output" else (),
    )


def _capabilities() -> AgentGatewaySessionCapabilities:
    return AgentGatewaySessionCapabilities(
        runtime_session_id="runtime:test",
        artifact_transfer_enabled=True,
        artifact_access_mechanisms=("chunked_rpc",),
        max_control_request_bytes=1000,
        max_control_response_bytes=1000,
        max_artifact_bytes=2000,
        max_artifact_chunk_bytes=100,
        max_artifact_refs_per_operation=2,
        max_concurrent_uploads=1,
        max_concurrent_downloads=1,
        artifact_store_quota_bytes=10_000,
        event_payload_max_bytes=1000,
        event_retention_count=10,
    )


def test_input_and_output_slots_share_one_generic_validator() -> None:
    input_binding = AgentArtifactBinding(
        slot_id="depth_frame",
        ordinal=0,
        artifact=_artifact(),
    )
    validate_skill_artifact_bindings(
        io_contract=_contract(direction="input"),
        direction="input",
        bindings=(input_binding,),
        session_capabilities=_capabilities(),
    )

    output_binding = AgentArtifactBinding(
        slot_id="depth_frame",
        ordinal=0,
        artifact=_artifact(),
        production_receipt_ref="receipt:producer",
    )
    validate_skill_artifact_bindings(
        io_contract=_contract(direction="output"),
        direction="output",
        bindings=(output_binding,),
        session_capabilities=_capabilities(),
    )


def test_unknown_missing_noncontiguous_and_uncommitted_output_fail_closed() -> None:
    contract = _contract(direction="output")
    with pytest.raises(ValueError, match="outside"):
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="output",
            bindings=(),
        )
    with pytest.raises(ValueError, match="unknown"):
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="output",
            bindings=(
                AgentArtifactBinding(
                    slot_id="rgb_frame",
                    ordinal=0,
                    artifact=_artifact(),
                    production_receipt_ref="receipt:producer",
                ),
            ),
        )
    with pytest.raises(ValueError, match="contiguous"):
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="output",
            bindings=(
                AgentArtifactBinding(
                    slot_id="depth_frame",
                    ordinal=1,
                    artifact=_artifact(),
                    production_receipt_ref="receipt:producer",
                ),
            ),
        )
    with pytest.raises(ValueError, match="requires a production receipt"):
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="output",
            bindings=(
                AgentArtifactBinding(
                    slot_id="depth_frame",
                    ordinal=0,
                    artifact=_artifact(),
                ),
            ),
        )


def test_slot_and_session_byte_limits_are_both_enforced() -> None:
    contract = _contract(direction="input")
    too_large = AgentArtifactBinding(
        slot_id="depth_frame",
        ordinal=0,
        artifact=_artifact(byte_length=1001),
    )
    with pytest.raises(ValueError, match="max bytes"):
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="input",
            bindings=(too_large,),
            session_capabilities=_capabilities(),
        )

    first = AgentArtifactBinding(
        slot_id="depth_frame",
        ordinal=0,
        artifact=_artifact(byte_length=800),
    )
    second = AgentArtifactBinding(
        slot_id="depth_frame",
        ordinal=1,
        artifact=_artifact(byte_length=800),
    )
    with pytest.raises(ValueError, match="total bytes"):
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="input",
            bindings=(first, second),
            session_capabilities=_capabilities(),
        )
