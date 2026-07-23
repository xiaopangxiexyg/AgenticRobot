"""Pure dataset-fact adapters for replay contracts.

This module selects an explicit historical action stage.  It does not decode a
``RobotAction`` or own runtime scheduling/dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.replay import (
    LEGACY_REPLAY_DERIVATION_HASH_DOMAIN_V1,
    REPLAY_ACTION_STAGE_COMMANDED,
    REPLAY_ACTION_STAGE_PROPOSAL,
    REPLAY_ACTION_STAGE_SAFE,
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    LegacyReplayContractDerivationReceipt,
    ReplayActionRecord,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
)
from dataset_contracts.v22 import DEXTELEOP_DATASET_SCHEMA_V22
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v24 import DEXTELEOP_DATASET_SCHEMA_V24


_V24_ACTION_STAGE_BY_FIELD = {
    "proposal_action": REPLAY_ACTION_STAGE_PROPOSAL,
    "selected_action": REPLAY_ACTION_STAGE_SELECTED,
    "safe_action": REPLAY_ACTION_STAGE_SAFE,
    "command": REPLAY_ACTION_STAGE_COMMANDED,
    "commanded_action": REPLAY_ACTION_STAGE_COMMANDED,
}

LEGACY_REPLAY_DERIVATION_CONTRACT_ID_V1 = (
    "dexteleop.legacy_replay_source_contract_derivation.v1"
)
LEGACY_REPLAY_DERIVATION_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": LEGACY_REPLAY_DERIVATION_CONTRACT_ID_V1,
        "supported_schemas": (
            DEXTELEOP_DATASET_SCHEMA_V22,
            DEXTELEOP_DATASET_SCHEMA_V23,
        ),
        "required_pose_dimensions": (
            "action_stage",
            "action_space",
            "pose_frame_contract_hash",
            "component_schema_hash",
        ),
    },
    domain="dexteleop.legacy_replay_derivation_contract.v1",
)


def replay_action_stage_from_contract(
    dataset_metadata: Mapping[str, Any],
    *,
    action_field: str,
) -> str:
    """Resolve an action stage without guessing from arbitrary column names."""

    field = str(action_field).strip()
    if not field:
        raise ValueError("action_field must be non-empty.")
    explicit = dataset_metadata.get("action_stage")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    schema = str(dataset_metadata.get("dexteleop_dataset_schema") or "").strip()
    if schema in {DEXTELEOP_DATASET_SCHEMA_V22, DEXTELEOP_DATASET_SCHEMA_V23}:
        if field == "action":
            return REPLAY_ACTION_STAGE_SELECTED
        raise ValueError(
            "v2.2/v2.3 compatibility replay only assigns selected_pre_safety "
            "semantics to the declared root 'action' field."
        )
    if schema == DEXTELEOP_DATASET_SCHEMA_V24:
        resolved = _V24_ACTION_STAGE_BY_FIELD.get(field)
        if resolved is not None:
            return resolved
        raise ValueError(
            "v2.4 replay requires an explicit typed action field or action_stage metadata."
        )
    raise ValueError(
        "Replay action stage is unknown; bind a supported dataset schema and action field."
    )


def replay_source_contract_from_metadata(
    dataset_metadata: Mapping[str, Any],
    *,
    source_path: str | Path,
    episode_id: str,
    action_field: str,
    source_action_space: str,
    source_dataset_id: str | None = None,
) -> ReplaySourceContract:
    schema = str(dataset_metadata.get("dexteleop_dataset_schema") or "").strip()
    if not schema:
        raise ValueError("Dataset metadata must declare dexteleop_dataset_schema.")
    path = Path(source_path).expanduser().resolve()
    dataset_id = str(
        source_dataset_id
        or dataset_metadata.get("dataset_id")
        or dataset_metadata.get("repo_id")
        or path.parent.name
    ).strip()
    stage = replay_action_stage_from_contract(
        dataset_metadata,
        action_field=action_field,
    )
    contract_id = (
        f"replay-source:{dataset_id}:{str(episode_id).strip()}:"
        f"{str(action_field).strip()}:{stage}"
    )
    return ReplaySourceContract(
        contract_id=contract_id,
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema=schema,
        source_dataset_id=dataset_id,
        capture_id=(
            dataset_metadata.get("capture_id")
            or dataset_metadata.get("capture_uuid")
        ),
        episode_id=str(episode_id),
        action_stage=stage,
        action_field=str(action_field),
        source_action_space=str(source_action_space),
        source_frame_contract_hash=(
            dataset_metadata.get("pose_frame_contract_hash")
            or dataset_metadata.get("dexteleop.pose_frame_contract_hash")
            or dataset_metadata.get("source_frame_contract_hash")
        ),
        source_component_schema_hash=(
            dataset_metadata.get("component_schema_hash")
            or dataset_metadata.get("policy_io_layout_hash")
        ),
        old_safety_profile_hash=dataset_metadata.get("safety_profile_hash"),
    )


def derive_replay_source_action_contract(
    source_contract: ReplaySourceContract,
    *,
    component_ids: tuple[str, ...],
    selection_groups: tuple[str, ...],
    coupled_selection_groups: tuple[str, ...] = (),
    action_spec_hash_by_component: Mapping[str, str] | None = None,
    units_convention_id: str = "si_m_rad_xyzw_v1",
) -> tuple[ReplaySourceActionContract, LegacyReplayContractDerivationReceipt]:
    """Derive only semantics guaranteed by a versioned dataset contract.

    Runtime defaults and target-device settings are deliberately not inputs.
    Missing source frame/component evidence remains unresolved.
    """

    components = tuple(str(value) for value in component_ids)
    groups = tuple(str(value) for value in selection_groups)
    if len(components) != len(groups) or not components:
        raise ValueError("Replay source components and selection groups must align.")
    unresolved: list[str] = []
    if source_contract.source_frame_contract_hash is None:
        unresolved.append("pose_frame_contract")
    if source_contract.source_component_schema_hash is None:
        unresolved.append("component_schema")
    status = (
        REPLAY_SOURCE_INTERPRETATION_VERIFIED
        if not unresolved
        else REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED
    )
    consumed = tuple(
        key
        for key, value in (
            ("dexteleop_dataset_schema", source_contract.dataset_schema),
            ("action_stage", source_contract.action_stage),
            ("source_action_space", source_contract.source_action_space),
            ("pose_frame_contract_hash", source_contract.source_frame_contract_hash),
            ("component_schema_hash", source_contract.source_component_schema_hash),
        )
        if value is not None
    )
    receipt_payload = {
        "derivation_contract_id": LEGACY_REPLAY_DERIVATION_CONTRACT_ID_V1,
        "derivation_contract_hash": LEGACY_REPLAY_DERIVATION_CONTRACT_HASH_V1,
        "source_dataset_schema": source_contract.dataset_schema,
        "consumed_metadata_fields": consumed,
        "derived_action_stage": source_contract.action_stage,
        "derived_pose_frame_contract_hash": source_contract.source_frame_contract_hash,
        "derived_component_schema_hash": source_contract.source_component_schema_hash,
        "unresolved_dimensions": tuple(unresolved),
        "derivation_status": status,
    }
    receipt_hash = canonical_content_hash(
        receipt_payload,
        domain=LEGACY_REPLAY_DERIVATION_HASH_DOMAIN_V1,
    )
    receipt = LegacyReplayContractDerivationReceipt(
        derivation_receipt_id=f"legacy-replay-derivation:{receipt_hash}",
        **receipt_payload,
        semantic_content_hash=receipt_hash,
    )
    source = ReplaySourceActionContract(
        contract_id=f"replay-source-action:{source_contract.contract_id}",
        interpretation_status=status,
        action_stage=str(source_contract.action_stage),
        action_space=str(source_contract.source_action_space),
        component_ids=components,
        selection_group_by_component=dict(zip(components, groups, strict=True)),
        coupled_selection_groups=tuple(coupled_selection_groups),
        pose_frame_contract_hash=source_contract.source_frame_contract_hash,
        component_schema_hash=source_contract.source_component_schema_hash,
        action_spec_hash_by_component=dict(action_spec_hash_by_component or {}),
        units_convention_id=units_convention_id,
        derivation_receipt_id=receipt.derivation_receipt_id,
        derivation_receipt_hash=receipt.semantic_content_hash,
        unresolved_dimensions=tuple(unresolved),
    )
    return source, receipt


def replay_action_record_from_materialized_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source_contract: ReplaySourceContract,
    schedule: ReplayScheduleSpec,
    canonical_action_payload: Mapping[str, Any],
    canonical_action_hash: str,
    source_timestamp: float | None = None,
    source_event_id: str | None = None,
) -> ReplayActionRecord:
    """Bind one materialized row to an immutable replay source fact.

    ``canonical_action_payload`` is supplied by the action-space adapter.  The
    dataset layer deliberately does not import robot-motion action classes.
    """

    index = int(row_index)
    if index < 0:
        raise ValueError("row_index must be non-negative.")
    timestamp = source_timestamp
    if timestamp is None:
        for key in ("decision_timestamp", "timestamp", "frame_timestamp"):
            if row.get(key) is not None:
                timestamp = float(row[key])
                break
    if timestamp is None:
        timestamp = schedule.relative_time_for_index(index)
    event_id = source_event_id
    if event_id is None:
        event_id = str(
            row.get("decision_id")
            or row.get("event_id")
            or f"{source_contract.episode_id}:frame:{index}"
        )
    return ReplayActionRecord.create(
        source_contract=source_contract,
        source_event_id=event_id,
        source_action_index=index,
        source_timestamp=float(timestamp),
        source_timestamp_clock_domain_id=(
            "capture_timestamp" if any(row.get(key) is not None for key in (
                "decision_timestamp",
                "timestamp",
                "frame_timestamp",
            )) else "capture_relative"
        ),
        canonical_action_payload=canonical_action_payload,
        canonical_action_hash=canonical_action_hash,
        schedule=schedule,
    )


__all__ = [
    "LEGACY_REPLAY_DERIVATION_CONTRACT_HASH_V1",
    "LEGACY_REPLAY_DERIVATION_CONTRACT_ID_V1",
    "derive_replay_source_action_contract",
    "replay_action_record_from_materialized_row",
    "replay_action_stage_from_contract",
    "replay_source_contract_from_metadata",
]
