from __future__ import annotations

import json
from pathlib import Path

from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)


_EVIDENCE = (
    Path(__file__).parent
    / "fixtures/control_core/rm65_m5_8b0_stop_only_field_evidence_2026-07-21.json"
)


def test_m5_8b0_field_evidence_matches_current_manifest_and_bounded_claims() -> None:
    record = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    payload = record["payload"]
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)

    assert record["evidence_scope"] == "scoped_final_call_field_evidence"
    assert record["lineage_completeness"].startswith("partial_v1_")
    assert payload["status"] == "stop_only_completed"
    assert payload["manifest_hash"] == manifest.semantic_content_hash
    assert payload["inventory_hash"] == inventory.semantic_content_hash
    assert payload["binding"]["production_binding_enabled"] is True
    assert payload["binding"]["motion_authorized"] is False
    assert payload["binding"]["attempted_sides"] == ["left", "right"]
    assert payload["binding"]["receipt_count"] == 2

    receipts = payload["receipts"]
    assert [item["side"] for item in receipts] == ["left", "right"]
    assert all(item["dispatch_accepted"] for item in receipts)
    assert all(item["dispatch_status"] == "accepted" for item in receipts)
    assert all(item["sdk_result_repr"] == "0" for item in receipts)

    for side in ("left", "right"):
        item = payload["sides"][side]
        assert item["send_result_code"] == 0
        assert item["send_result_text"] == "hold:Arm_Slow_Stop:0"
        assert item["stationary_drift_rad"] < 0.005
        assert item["post_stop_delta_rad"] < 0.005
        assert item["slow_stop_latency_sec"] < 0.05
