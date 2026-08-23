"""Read and validate exclusive HITL metadata from formal dataset rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataset_contracts.hitl import (
    EXCLUSIVE_HITL_DATASET_METADATA_KEY,
    EXCLUSIVE_HITL_SAMPLE_METADATA_KEY,
    ExclusiveHITLDatasetContract,
    ExclusiveHITLSampleLineage,
)
from .capture_modalities import dataset_is_blank


class ExclusiveHITLDatasetReadError(ValueError):
    """Raised when dataset-level and sample-level HITL lineage disagree."""


@dataclass(frozen=True)
class ExclusiveHITLDatasetValidationSummary:
    sample_count: int
    human_sample_count: int
    policy_sample_count: int
    source_transition_count: int
    switch_count: int
    position_reference_sample_count: int
    policy_identity_status: str


def read_exclusive_hitl_dataset_contract(
    dataset_metadata: Mapping[str, Any],
) -> ExclusiveHITLDatasetContract | None:
    value = dataset_metadata.get(EXCLUSIVE_HITL_DATASET_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExclusiveHITLDatasetReadError(
            f"{EXCLUSIVE_HITL_DATASET_METADATA_KEY} must be an object."
        )
    try:
        return ExclusiveHITLDatasetContract.from_payload(value)
    except (TypeError, ValueError) as exc:
        raise ExclusiveHITLDatasetReadError(
            f"invalid {EXCLUSIVE_HITL_DATASET_METADATA_KEY}: {exc}"
        ) from exc


def require_exclusive_hitl_dataset_contract(
    dataset_metadata: Mapping[str, Any],
) -> ExclusiveHITLDatasetContract:
    contract = read_exclusive_hitl_dataset_contract(dataset_metadata)
    if contract is None:
        raise ExclusiveHITLDatasetReadError(
            f"dataset is missing {EXCLUSIVE_HITL_DATASET_METADATA_KEY}."
        )
    return contract


def validate_exclusive_hitl_dataset_append(
    dataset_root: str | Path,
    requested: ExclusiveHITLDatasetContract,
) -> None:
    """Reject mixing HITL identities before any writer version appends data."""

    root = Path(dataset_root).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        if not dataset_is_blank(root):
            raise ExclusiveHITLDatasetReadError(
                "populated dataset has no meta/info.json HITL identity."
            )
        return
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExclusiveHITLDatasetReadError(
            f"cannot read dataset metadata {info_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExclusiveHITLDatasetReadError(
            f"dataset metadata must be an object: {info_path}"
        )
    existing = read_exclusive_hitl_dataset_contract(payload)
    if existing is None:
        if not dataset_is_blank(root):
            raise ExclusiveHITLDatasetReadError(
                "refusing to append HITL samples to a populated dataset without "
                f"{EXCLUSIVE_HITL_DATASET_METADATA_KEY}."
            )
        return
    if existing != requested:
        raise ExclusiveHITLDatasetReadError(
            "existing exclusive HITL dataset contract differs from the requested profile."
        )


def _sample_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("sample_metadata_json")
    if raw is None:
        raw = row.get("sample_metadata")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        raise ExclusiveHITLDatasetReadError(
            "sample_metadata_json must be a JSON object string."
        )
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExclusiveHITLDatasetReadError(
            f"sample_metadata_json is invalid JSON: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ExclusiveHITLDatasetReadError(
            "sample_metadata_json must decode to an object."
        )
    return decoded


def _record_from_payload(value: object, *, location: str) -> ExclusiveHITLSampleLineage:
    if not isinstance(value, Mapping):
        raise ExclusiveHITLDatasetReadError(f"{location} must be an object.")
    try:
        return ExclusiveHITLSampleLineage.from_payload(value)
    except (TypeError, ValueError) as exc:
        raise ExclusiveHITLDatasetReadError(f"invalid {location}: {exc}") from exc


def read_exclusive_hitl_sample(
    row: Mapping[str, Any],
) -> ExclusiveHITLSampleLineage | None:
    metadata = _sample_metadata(row)
    value = metadata.get(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY)
    if value is None:
        return None
    return _record_from_payload(
        value,
        location=EXCLUSIVE_HITL_SAMPLE_METADATA_KEY,
    )


def _validate_redundant_lineage(
    metadata: Mapping[str, Any],
    expected: ExclusiveHITLSampleLineage,
) -> bool:
    command_metadata = metadata.get("command_metadata")
    if isinstance(command_metadata, Mapping):
        try:
            command_record = ExclusiveHITLSampleLineage.from_command_metadata(
                command_metadata
            )
        except (TypeError, ValueError) as exc:
            raise ExclusiveHITLDatasetReadError(
                f"invalid command_metadata HITL lineage: {exc}"
            ) from exc
        if command_record is not None and command_record != expected:
            raise ExclusiveHITLDatasetReadError(
                "command_metadata HITL lineage disagrees with the canonical sample record."
            )

    position_reference = metadata.get("position_reference")
    if position_reference is None:
        return False
    if not isinstance(position_reference, Mapping):
        raise ExclusiveHITLDatasetReadError("position_reference must be an object.")
    frame_metadata = position_reference.get("frame_metadata")
    if not isinstance(frame_metadata, Mapping):
        raise ExclusiveHITLDatasetReadError(
            "HITL position_reference is missing frame_metadata."
        )
    reference_record = _record_from_payload(
        frame_metadata.get(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY),
        location=(
            "position_reference.frame_metadata." + EXCLUSIVE_HITL_SAMPLE_METADATA_KEY
        ),
    )
    if reference_record != expected:
        raise ExclusiveHITLDatasetReadError(
            "position-reference HITL lineage disagrees with the canonical sample record."
        )
    return True


def validate_exclusive_hitl_dataset_rows(
    dataset_metadata: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> ExclusiveHITLDatasetValidationSummary:
    """Fail closed on missing, mixed, or causally inconsistent HITL labels."""

    contract = require_exclusive_hitl_dataset_contract(dataset_metadata)
    sample_count = 0
    source_counts = {"human": 0, "policy": 0}
    source_transition_count = 0
    position_reference_sample_count = 0
    switch_ids: set[str] = set()
    previous_source: str | None = None
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExclusiveHITLDatasetReadError(f"row {row_index} must be an object.")
        record = read_exclusive_hitl_sample(row)
        if record is None:
            raise ExclusiveHITLDatasetReadError(
                f"row {row_index} is missing {EXCLUSIVE_HITL_SAMPLE_METADATA_KEY}."
            )
        if record.policy_mode != contract.policy_mode:
            raise ExclusiveHITLDatasetReadError(
                f"row {row_index} policy_mode differs from the dataset contract."
            )
        if record.rtc_no_action != contract.rtc_no_action:
            raise ExclusiveHITLDatasetReadError(
                f"row {row_index} rtc_no_action differs from the dataset contract."
            )
        if contract.policy_mode == "rtc" and record.source == "policy":
            raise ExclusiveHITLDatasetReadError(
                f"row {row_index} claims policy action under RTC no-action v1."
            )
        metadata = _sample_metadata(row)
        if _validate_redundant_lineage(metadata, record):
            position_reference_sample_count += 1
        sample_count += 1
        source_counts[record.source] += 1
        if previous_source is not None and record.source != previous_source:
            source_transition_count += 1
        previous_source = record.source
        if record.switch_id is not None:
            switch_ids.add(record.switch_id)

    return ExclusiveHITLDatasetValidationSummary(
        sample_count=sample_count,
        human_sample_count=source_counts["human"],
        policy_sample_count=source_counts["policy"],
        source_transition_count=source_transition_count,
        switch_count=len(switch_ids),
        position_reference_sample_count=position_reference_sample_count,
        policy_identity_status=contract.policy_identity_status,
    )


__all__ = [
    "ExclusiveHITLDatasetReadError",
    "ExclusiveHITLDatasetValidationSummary",
    "read_exclusive_hitl_dataset_contract",
    "read_exclusive_hitl_sample",
    "require_exclusive_hitl_dataset_contract",
    "validate_exclusive_hitl_dataset_append",
    "validate_exclusive_hitl_dataset_rows",
]
