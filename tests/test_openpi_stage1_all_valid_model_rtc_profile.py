from __future__ import annotations

import json
from pathlib import Path

from deployments.openpi.model_rtc_profile import (
    load_prepared_model_rtc_profile,
    prepared_model_rtc_profile_hash,
)
from deployments.openpi.training.prepared_skill_launch import (
    _load_profile,
    _model_rtc_profile_matches_prepared_profile,
    _prepared_profile_for_model_rtc,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_PREPARED_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/training/"
    / "writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
)
_MODEL_RTC_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/model_rtc/"
    / "writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json"
)


def test_stage1_all_view_h32_model_rtc_profile_matches_prepared_profile() -> None:
    prepared = _load_profile(_PREPARED_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    assert prepared["profile_id"] == (
        "writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_v1"
    )
    assert model_rtc["prepared_skill_profile_id"] == prepared["profile_id"]
    assert prepared["camera_selection"] == ["front", "left_wrist", "right_wrist"]
    assert prepared["camera_slot_to_dataset_key"] == {
        "base_0_rgb": "front",
        "left_wrist_0_rgb": "left_wrist",
        "right_wrist_0_rgb": "right_wrist",
    }
    assert prepared["action_horizon"] == 32
    assert prepared["action_pose_mode"] == "absolute"
    assert prepared["expected_total_raw_episode_count"] == 3587
    assert prepared["expected_total_eligible_episode_count"] == 3380

    assert model_rtc["minimum_forecast_delay_steps"] == 1
    assert model_rtc["maximum_forecast_delay_steps"] == 12
    assert model_rtc["minimum_prefix_steps"] == 0
    assert model_rtc["maximum_prefix_steps"] == 12
    assert model_rtc["maximum_prefix_steps"] <= (
        model_rtc["maximum_forecast_delay_steps"]
    ) < prepared["action_horizon"]


def test_stage1_all_view_h32_model_rtc_sampling_is_explicit_and_hashable() -> None:
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)
    sampling = model_rtc["sampling_spec"]

    assert sampling["schema"] == "dexteleop_model_continuation_training_sampling_v1"
    assert sampling["strategy"] == "hash_indexed_uniform_explicit_pairs_v1"
    assert sampling["selection_salt"] == model_rtc["profile_id"].replace(
        "_", "-"
    )
    assert sampling["delay_prefix_pairs"] == [
        [1, 1],
        [2, 1],
        [2, 2],
        [4, 2],
        [4, 4],
        [8, 4],
        [8, 8],
        [12, 8],
        [12, 12],
    ]
    assert prepared_model_rtc_profile_hash(model_rtc).startswith("sha256:")


def test_stage1_all_view_h32_model_rtc_profile_has_no_ordinary_checkpoint_fields() -> None:
    payload = json.loads(_MODEL_RTC_PROFILE.read_text(encoding="utf-8"))

    assert "checkpoint_dir" not in payload
    assert "base_checkpoint_dir" not in payload
    assert "continuation_contract_hash" not in payload
    assert payload["output_config_name"] != (
        "writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_v1-prepared"
    )
    assert payload["prepared_final_mask_root"] == (
        "masks/model-rtc-stage1-all-view-h32-v1/derived/final_keep_masks"
    )


def test_stage1_model_rtc_overlay_requires_exact_prepared_profile_identity() -> None:
    prepared = _load_profile(_PREPARED_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    assert _model_rtc_profile_matches_prepared_profile(prepared, model_rtc)

    mismatched = dict(model_rtc)
    mismatched["prepared_skill_profile_id"] = "another-prepared-profile"
    assert not _model_rtc_profile_matches_prepared_profile(prepared, mismatched)


def test_model_rtc_mask_root_isolated_from_ordinary_prepared_profile() -> None:
    prepared = _load_profile(_PREPARED_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    effective = _prepared_profile_for_model_rtc(prepared, model_rtc)
    snapshot_root = Path(prepared["organized_dataset_root"]).parent.parent
    assert effective["openpi_final_mask_root"] == str(
        snapshot_root
        / "masks/model-rtc-stage1-all-view-h32-v1/derived/final_keep_masks"
    )
    assert prepared["openpi_final_mask_root"] != effective["openpi_final_mask_root"]
    assert _prepared_profile_for_model_rtc(prepared, None) is prepared
