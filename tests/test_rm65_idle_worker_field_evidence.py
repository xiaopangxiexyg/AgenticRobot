from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_contracts.completeness import trace_requirement_profile
from teleop_runtime.backends.arms.rm65_idle_worker_preflight_binding import (
    RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1,
)
from teleop_runtime.backends.arms.rm65_execution_evidence import (
    resolve_rm65_historical_execution_evidence_contract,
)


_EVIDENCE = (
    Path(__file__).parent
    / "fixtures/control_core/"
    "rm65_m5_8b1_real_a_idle_worker_drain_field_evidence_2026-07-21.json"
)
_EXPECTED_SOURCE_SHA256 = (
    "32e4850e9cd0aafdd748226929fd0079c139125ef314378869fb44b18f3d6b32"
)


def test_historical_contract_resolver_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="Unknown RM65 historical"):
        resolve_rm65_historical_execution_evidence_contract(
            "rm65_unknown_field_evidence_v1"
        )


def test_m5_8b1_real_a_field_evidence_matches_declared_historical_contract() -> None:
    record = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    payload = record["payload"]
    contract = resolve_rm65_historical_execution_evidence_contract(
        record["evidence_schema"]
    )
    profile = trace_requirement_profile(RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1)

    assert record["evidence_scope"] == "scoped_idle_worker_drain_stop_field_evidence"
    assert record["lineage_completeness"] == "runtime_diagnostic_profile_complete_v1"
    assert record["source_sha256"] == _EXPECTED_SOURCE_SHA256
    assert payload["status"] == "idle_worker_drain_stop_only_completed"
    assert payload["profile_id"] == profile.profile_id
    assert contract.manifest_payload_status == "hash_only_not_archived"
    assert payload["manifest_id"] == contract.manifest_identity.manifest_id
    assert payload["manifest_hash"] == contract.manifest_identity.manifest_hash
    assert payload["inventory_hash"] == contract.dispatch_inventory_hash
    assert payload["worker_drain_plan_hash"] == contract.worker_drain_plan_hash

    assert payload["canonical_defaults_modified"] is False
    assert payload["motion_authorized"] is False
    assert payload["lifecycle_authorized"] is False
    assert payload["binding"]["production_binding_enabled"] is True
    assert payload["binding"]["execution_epoch"] == 1
    assert payload["binding"]["motion_authorized"] is False
    assert payload["binding"]["lifecycle_authorized"] is False

    required = list(contract.required_worker_participant_ids)
    aggregate = payload["drain_aggregate"]
    assert aggregate["complete"] is True
    assert aggregate["status"] == "complete"
    assert aggregate["required_participant_ids"] == required
    assert aggregate["accepted_participant_ids"] == required
    assert aggregate["missing_participant_ids"] == []
    assert aggregate["rejected_participant_ids"] == []

    receipts = payload["drain_receipts"]
    assert [item["participant_id"] for item in receipts] == required
    assert all(item["accepted"] for item in receipts)
    assert all(item["status"] == "drained" for item in receipts)
    assert all(item["observed_execution_epoch"] == 1 for item in receipts)
    assert all(item["invalidated_execution_epoch"] == 0 for item in receipts)
    assert all(item["in_flight_old_epoch_count"] == 0 for item in receipts)
    assert all(item["no_future_old_epoch_calls"] for item in receipts)

    for side in ("left", "right"):
        item = payload["sides"][side]
        idle = item["idle_worker_before_drain"]
        assert idle["tick_sequence"] > 0
        assert idle["send_count"] == 0
        assert idle["target_active"] is False
        assert item["send_result_code"] == 0
        assert item["send_result_text"] == "hold:Arm_Slow_Stop:0"
        assert item["stationary_drift_rad"] < 0.005
        assert item["post_stop_delta_rad"] < 0.005
        assert item["slow_stop_latency_sec"] < 0.05

    stop_receipts = payload["stop_receipts"]
    assert [item["side"] for item in stop_receipts] == ["left", "right"]
    assert len({item["permit_id"] for item in stop_receipts}) == 2
    assert all(item["dispatch_accepted"] for item in stop_receipts)
    assert all(item["dispatch_status"] == "accepted" for item in stop_receipts)
    assert all(item["sdk_result_repr"] == "0" for item in stop_receipts)

    trace = payload["trace"]
    assert trace["required_event_types"] == sorted(profile.required_event_types)
    assert trace["issued_sequence"] == trace["durable_high_watermark"] == 10
    assert trace["in_memory_complete"] is True
    assert trace["durable_complete"] is True
    assert trace["capture_commit_eligible"] is False
