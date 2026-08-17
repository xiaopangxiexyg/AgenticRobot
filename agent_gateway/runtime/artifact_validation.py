"""Pure validation of Artifact bindings against one exact Skill I/O contract."""

from __future__ import annotations

from collections import defaultdict

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)


def _validate_slot_artifact(
    *,
    slot: SkillArtifactSlotSpec,
    binding: AgentArtifactBinding,
) -> None:
    artifact = binding.artifact
    if artifact.media_type not in slot.allowed_media_types:
        raise ValueError(f"Artifact media type is not allowed for slot {slot.slot_id!r}.")
    if slot.allowed_schema_ids and artifact.schema_id not in slot.allowed_schema_ids:
        raise ValueError(f"Artifact schema is not allowed for slot {slot.slot_id!r}.")
    if artifact.encoding not in slot.allowed_encodings:
        raise ValueError(f"Artifact encoding is not allowed for slot {slot.slot_id!r}.")
    if artifact.compression not in slot.allowed_compressions:
        raise ValueError(
            f"Artifact compression is not allowed for slot {slot.slot_id!r}."
        )
    if artifact.byte_length > slot.max_artifact_bytes:
        raise ValueError(f"Artifact exceeds max bytes for slot {slot.slot_id!r}.")


def validate_skill_artifact_bindings(
    *,
    io_contract: SkillIOContract,
    direction: str,
    bindings: tuple[AgentArtifactBinding, ...],
    session_capabilities: AgentGatewaySessionCapabilities | None = None,
) -> None:
    """Fail closed over slot identity, cardinality, type, bytes, and receipts."""

    if not isinstance(io_contract, SkillIOContract):
        raise TypeError("io_contract must be a SkillIOContract.")
    if direction not in {"input", "output"}:
        raise ValueError("direction must be 'input' or 'output'.")
    normalized = tuple(bindings)
    if any(not isinstance(item, AgentArtifactBinding) for item in normalized):
        raise TypeError("bindings must contain AgentArtifactBinding values.")
    keys = tuple((item.slot_id, item.ordinal) for item in normalized)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise ValueError("Artifact bindings must be uniquely sorted by slot/ordinal.")
    if session_capabilities is not None:
        if not isinstance(session_capabilities, AgentGatewaySessionCapabilities):
            raise TypeError(
                "session_capabilities must be AgentGatewaySessionCapabilities."
            )
        if len(normalized) > session_capabilities.max_artifact_refs_per_operation:
            raise ValueError("Artifact binding count exceeds session capabilities.")
        if any(
            item.artifact.byte_length > session_capabilities.max_artifact_bytes
            for item in normalized
        ):
            raise ValueError("Artifact exceeds the session capability byte limit.")

    slots = {
        slot.slot_id: slot
        for slot in (
            io_contract.input_artifact_slots
            if direction == "input"
            else io_contract.output_artifact_slots
        )
    }
    by_slot: dict[str, list[AgentArtifactBinding]] = defaultdict(list)
    for binding in normalized:
        slot = slots.get(binding.slot_id)
        if slot is None:
            raise ValueError(
                f"Artifact binding references unknown {direction} slot "
                f"{binding.slot_id!r}."
            )
        if direction == "input" and binding.production_receipt_ref is not None:
            raise ValueError("Input Artifact binding cannot carry a production receipt.")
        if direction == "output" and binding.production_receipt_ref is None:
            raise ValueError("Output Artifact binding requires a production receipt.")
        _validate_slot_artifact(slot=slot, binding=binding)
        by_slot[binding.slot_id].append(binding)

    for slot_id, slot in slots.items():
        slot_bindings = by_slot.get(slot_id, [])
        count = len(slot_bindings)
        if not slot.min_count <= count <= slot.max_count:
            raise ValueError(
                f"Artifact count for slot {slot_id!r} is outside "
                f"[{slot.min_count}, {slot.max_count}]."
            )
        ordinals = tuple(item.ordinal for item in slot_bindings)
        if ordinals != tuple(range(count)):
            raise ValueError(
                f"Artifact ordinals for slot {slot_id!r} must be contiguous from zero."
            )
        if sum(item.artifact.byte_length for item in slot_bindings) > (
            slot.max_total_bytes
        ):
            raise ValueError(f"Artifacts exceed total bytes for slot {slot_id!r}.")


__all__ = ["validate_skill_artifact_bindings"]
