from __future__ import annotations

import json
from pathlib import Path

from teleop_runtime.deploy.rm65_execution_evidence import (
    resolve_rm65_historical_execution_evidence_contract,
)


_EVIDENCE = (
    Path(__file__).parent
    / "fixtures/control_core/rm65_m5_8b0_stop_only_field_evidence_2026-07-21.json"
)


def test_m5_8b0_field_evidence_matches_declared_historical_contract() -> None:
    record = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    payload = record["payload"]
    contract = resolve_rm65_historical_execution_evidence_contract(
        record["evidence_schema"]
    )

    assert record["evidence_scope"] == "scoped_final_call_field_evidence"
    assert record["lineage_completeness"].startswith("partial_v1_")
    assert payload["status"] == "stop_only_completed"
    assert payload["manifest_id"] == contract.manifest_identity.manifest_id
    assert payload["manifest_hash"] == contract.manifest_identity.manifest_hash
    assert payload["binding"]["execution_manifest_hash"] == (
        contract.manifest_identity.manifest_hash
    )
    assert payload["inventory_hash"] == contract.dispatch_inventory_hash
    assert payload["binding"]["dispatch_inventory_hash"] == (
        contract.dispatch_inventory_hash
    )
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
