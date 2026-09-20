from __future__ import annotations

from pathlib import Path

from scripts.probes.probe_openpi_rtc_step_horizon import (
    audit,
    _parse_pair,
)


_ROOT = Path(__file__).resolve().parents[1]
_TRAINING = (
    _ROOT
    / "deployments/openpi/configs/training/"
    / "writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
)
_MODEL_RTC = (
    _ROOT
    / "deployments/openpi/configs/model_rtc/"
    / "writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json"
)


def test_step_horizon_probe_is_diagnostic_only_and_reports_contract_limits() -> None:
    report = audit(
        prepared_training_profile=_TRAINING,
        model_rtc_profile=_MODEL_RTC,
        artifact_json=None,
        binding_receipt_json=None,
        pairs=((8, 32), (6, 32), (4, 24)),
    )

    assert report["runtime_started"] is False
    assert report["gpu_used"] is False
    assert report["hardware_contacted"] is False
    assert report["prepared_training_profile"]["action_horizon"] == 32
    assert report["deployment_step_allowlist"] == [8, 10]
    assert report["results"][0]["status"] == "rejected"
    assert "artifact_sidecar_required_for_runtime_check" in report["results"][0]["reasons"]
    assert "effective_num_steps_not_in_current_deployment_allowlist:[8, 10]" in report["results"][1]["reasons"]
    assert "horizon_mismatch_training_profile:32" in report["results"][2]["reasons"]


def test_pair_parser_accepts_common_forms() -> None:
    assert _parse_pair("8x32") == (8, 32)
    assert _parse_pair("6:24") == (6, 24)
    assert _parse_pair("4/32") == (4, 32)
