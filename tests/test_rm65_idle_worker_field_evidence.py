from __future__ import annotations

import json
from pathlib import Path

from dataset_contracts.completeness import trace_requirement_profile
from teleop_runtime.backends.arms.rm65_idle_worker_preflight_binding import (
    RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1,
)
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.deploy.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)


_EVIDENCE = (
    Path(__file__).parent
    / "fixtures/control_core/"
    "rm65_m5_8b1_real_a_idle_worker_drain_field_evidence_2026-07-21.json"
)
_EXPECTED_SOURCE_SHA256 = (
    "32e4850e9cd0aafdd748226929fd0079c139125ef314378869fb44b18f3d6b32"
)


def test_m5_8b1_real_a_field_evidence_matches_current_bounded_contract() -> None:
    record = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    payload = record["payload"]
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    bundle = build_rm65_execution_evidence_bundle(manifest)
    profile = trace_requirement_profile(RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1)

    assert record["evidence_scope"] == "scoped_idle_worker_drain_stop_field_evidence"
    assert record["lineage_completeness"] == "runtime_diagnostic_profile_complete_v1"
    assert record["source_sha256"] == _EXPECTED_SOURCE_SHA256
    assert payload["status"] == "idle_worker_drain_stop_only_completed"
    assert payload["profile_id"] == profile.profile_id
    assert payload["manifest_hash"] == manifest.semantic_content_hash
    assert payload["inventory_hash"] == bundle.dispatch_inventory.semantic_content_hash
    assert payload["worker_drain_plan_hash"] == bundle.worker_drain_plan.semantic_content_hash

    assert payload["canonical_defaults_modified"] is False
    assert payload["motion_authorized"] is False
    assert payload["lifecycle_authorized"] is False
    assert payload["binding"]["production_binding_enabled"] is True
    assert payload["binding"]["execution_epoch"] == 1
    assert payload["binding"]["motion_authorized"] is False
    assert payload["binding"]["lifecycle_authorized"] is False

    required = [item.participant_id for item in bundle.worker_drain_plan.participants]
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

