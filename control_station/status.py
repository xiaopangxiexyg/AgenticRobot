"""Rebuildable UI status projection for the Control Station."""

from __future__ import annotations

from typing import Any

from .contracts import ControlStationRunManifest
from .state_machine import ControlStationState


CONTROL_STATION_STATUS_SCHEMA_V1 = "dexteleop.control_station.status.v1"


def build_control_station_status_payload(
    manifest: ControlStationRunManifest,
    state: ControlStationState,
    *,
    intent_submission_enabled: bool = False,
    event_fanout_enabled: bool = False,
    operation_coordinator_enabled: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe cache view, never a control or trace fact."""

    if not isinstance(manifest, ControlStationRunManifest):
        raise TypeError("manifest must be a ControlStationRunManifest.")
    if not isinstance(state, ControlStationState):
        raise TypeError("state must be a ControlStationState.")
    if manifest.semantic_content_hash != state.manifest_hash:
        raise ValueError("Manifest/state hash mismatch.")

    lease = state.operator_control_lease
    source_specs = {slot.source_slot_id: slot for slot in manifest.source_slots}
    return {
        "schema": CONTROL_STATION_STATUS_SCHEMA_V1,
        "maturity": (
            "cs1c_no_io_operation_coordination"
            if operation_coordinator_enabled
            else (
                "cs1b_typed_intent_local_gui"
                if intent_submission_enabled
                else "cs1a_read_only_local_gui"
            )
        ),
        "read_only": not intent_submission_enabled,
        "runtime_session_id": state.runtime_session_id,
        "control_domain_id": state.control_domain_id,
        "manifest": {
            "manifest_id": manifest.manifest_id,
            "manifest_version": manifest.manifest_version,
            "semantic_content_hash": manifest.semantic_content_hash,
            "explicit_exclusions": list(manifest.explicit_exclusions),
        },
        "station": {
            "lifecycle_state": state.lifecycle_state,
            "state_revision": state.state_revision,
            "last_intent_id": state.last_intent_id,
        },
        "target": {
            "binding_id": manifest.execution_binding.binding_id,
            "target_kind": manifest.execution_binding.target_kind,
            "embodiment_instance_id": (
                manifest.execution_binding.embodiment_instance_id
            ),
            "runtime_state": state.target_state,
            "execution_capability_profile_hash": (
                manifest.execution_binding.execution_capability_profile_hash
            ),
            "motion_authorized": False,
            "authorization_status": "not_available_in_cs1a",
        },
        "sources": [
            {
                "source_slot_id": slot_id,
                "source_kind": source_specs[slot_id].source_kind,
                "producer_id": source_specs[slot_id].producer_id,
                "runtime_state": runtime_state,
                "supported_uses": list(source_specs[slot_id].supported_uses),
                "required_for_run": source_specs[slot_id].required_for_run,
                "expected_emission_seam": (
                    source_specs[slot_id].expected_emission_seam
                ),
            }
            for slot_id, runtime_state in state.source_state_by_slot.items()
        ],
        "pending_operations": [
            {
                "subject_key": subject_key,
                "source_intent_id": operation.source_intent_id,
                "operation_kind": operation.operation_kind,
                "subject_id": operation.subject_id,
                "prior_runtime_state": operation.prior_runtime_state,
                "requested_runtime_state": operation.requested_runtime_state,
            }
            for subject_key, operation in state.pending_operation_by_subject.items()
        ],
        "operator_control_lease": (
            None
            if lease is None
            else {
                "lease_id": lease.lease_id,
                "client_id": lease.client_id,
                "lease_epoch": lease.lease_epoch,
                "acquired_at": lease.acquired_at,
                "expires_at": lease.expires_at,
            }
        ),
        "control_facts": {
            "dispatch_owner": "unavailable_in_cs1a",
            "authority_epoch": "unavailable_in_cs1a",
            "execution_epoch": "unavailable_in_cs1a",
            "capture_status": "unavailable_in_cs1a",
        },
        "capabilities": {
            "state_read": True,
            "intent_submission": bool(intent_submission_enabled),
            "motion": False,
            "capture": False,
            "source_adapters": "null_or_fake_only",
            **(
                {
                    "event_fanout": bool(event_fanout_enabled),
                    "operation_coordinator": bool(operation_coordinator_enabled),
                }
                if event_fanout_enabled or operation_coordinator_enabled
                else {}
            ),
        },
    }


__all__ = [
    "CONTROL_STATION_STATUS_SCHEMA_V1",
    "build_control_station_status_payload",
]
