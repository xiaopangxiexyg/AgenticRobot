from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployments.openpi.training.prepared_skill_launch import (
    _corrected_wrist_background_overrides,
    _load_profile,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/training/"
    "writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
)


def test_stage1_all_view_h32_profile_uses_every_canonical_stage1_source() -> None:
    profile = _load_profile(_PROFILE)
    sources = profile["sources"]
    by_id = {str(item["source_dataset_id"]): item for item in sources}

    assert list(by_id) == [
        "writing4_2026_08_03_camera_only",
        "writing4_2026_08_04_camera_only",
        "writing4_2026_08_05_camera_only",
        "writing4_2026_08_06_camera_only_merged",
        "writing4_2026_08_07_camera_only_merged",
        "writing1_2026_08_10_fixed_middle_camera_only",
        "writing1_2026_08_10_multi_choice_camera_only",
        "writing1_2026_08_11_interrupt_choice_camera_only",
        "writing1_2026_08_11_multi_choice_camera_only",
        "writing4_2026_08_19_detail_camera_only",
        "writing1_2026_08_20_detail_camera_only",
        "writing4_2026_08_21_detail_camera_only",
        "writing4_2026_08_27_camera_only_merged",
    ]
    assert [item["expected_raw_episode_count"] for item in sources] == [
        207,
        355,
        261,
        294,
        331,
        349,
        266,
        201,
        251,
        159,
        400,
        457,
        56,
    ]
    assert [item["expected_raw_frame_count"] for item in sources] == [
        88732,
        142576,
        102887,
        112817,
        113517,
        45400,
        30115,
        37069,
        29620,
        32042,
        56990,
        83891,
        36636,
    ]
    assert [item["expected_eligible_episode_count"] for item in sources] == [
        186,
        344,
        256,
        290,
        327,
        345,
        265,
        102,
        250,
        149,
        393,
        437,
        36,
    ]
    assert [item["expected_complete_h32_window_count"] for item in sources] == [
        8890,
        14298,
        11160,
        13886,
        14604,
        24156,
        15453,
        9499,
        14391,
        14836,
        27455,
        41378,
        5416,
    ]
    assert [
        item["expected_openpi_quality_valid_window_count"]
        for item in sources
    ] == [
        8634,
        12779,
        9945,
        12029,
        10309,
        10163,
        1856,
        4985,
        4987,
        2676,
        5785,
        22605,
        2001,
    ]

    assert profile["expected_total_raw_episode_count"] == 3587
    assert profile["expected_total_raw_frame_count"] == 912292
    assert profile["expected_total_eligible_episode_count"] == 3380
    assert profile["expected_complete_h32_window_count"] == 215422
    assert profile["expected_openpi_quality_valid_window_count"] == 108754
    assert profile["expected_openpi_repeated_epoch_sample_count"] == 108754
    assert len(profile["annotation_selection"]) == 3473
    assert profile["source_snapshot"]["selector_report_sha256"] == (
        "322f875d52bacc3954af9605958e4308bc23effa77bf96ebf1594a35a0c255b6"
    )
    assert profile["source_snapshot"]["strict_eligible_before_h32_count"] == 3380
    assert profile["source_snapshot"]["h32_eligible_count"] == 3380
    assert profile["source_snapshot"]["raw_and_derived_overlap_allowed"] is False
    assert len({item["dataset_root"] for item in sources}) == len(sources)
    assert profile["complete_horizon_window_exclusions"] == []
    assert not any(
        token in item["dataset_root"]
        for item in sources
        for token in ("2026-08-24", "2026-08-25")
    )

    august_21 = by_id["writing4_2026_08_21_detail_camera_only"]
    assert august_21["expected_raw_episode_count"] == 457
    assert august_21["expected_eligible_episode_count"] == 437
    assert august_21["expected_task_id"] == "4_stage_writing_detail"
    assert august_21["expected_task_id_aliases"] == [
        "dmtac_rm65_bimanual_position_v1"
    ]


def test_stage1_quality_window_contract_fails_closed(tmp_path: Path) -> None:
    profile = json.loads(_PROFILE.read_text(encoding="utf-8"))
    profile["sources"][0][
        "expected_openpi_quality_valid_window_count"
    ] += 1
    mismatched = tmp_path / "mismatched-stage1-profile.json"
    mismatched.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="source counts differ"):
        _load_profile(mismatched)


def test_stage1_all_view_h32_profile_pins_training_and_mask_contracts() -> None:
    profile = _load_profile(_PROFILE)

    assert profile["target_skill_id"] == (
        "writing.bimanual_acquire_brush_and_present_for_cap_removal"
    )
    assert profile["target_progress_step_id"] == "right_acquire_brush"
    assert profile["skill_projection_catalog_id"] == "writing_stage1_v2"
    assert profile["camera_selection"] == [
        "front",
        "left_wrist",
        "right_wrist",
    ]
    assert profile["camera_slot_to_dataset_key"] == {
        "base_0_rgb": "front",
        "left_wrist_0_rgb": "left_wrist",
        "right_wrist_0_rgb": "right_wrist",
    }
    assert profile["action_horizon"] == 32
    assert profile["action_pose_mode"] == "absolute"
    assert profile["source_mixing_policy"] == "natural_concat_shuffle_v1"
    assert profile["tail_policy"] == "drop"
    assert profile["openpi_training_mode"] == "standard_non_rtc_v1"
    assert profile["openpi_base_config"] == (
        "writing-all-view-h32-absolute-standard-v1"
    )
    assert profile["openpi_state_augmentation_from_config"] == (
        profile["openpi_base_config"]
    )
    assert profile["openpi_num_train_steps"] == 100000
    assert profile["openpi_save_interval"] == 25000
    assert profile["openpi_keep_period"] == 25000
    assert profile["openpi_num_workers"] == 4
    assert Path(profile["openpi_checkpoint_base_dir"]) == Path(
        "/home/user/zc/checkpoints/openpi"
    )
    assert profile["openpi_checkout"].endswith(
        "/code-stage3/AgenticRobot/model/openpi"
    )
    assert profile["openpi_source_revision"] == (
        "940cdc87516a7cf07db7ae46147548c1cf8382ee"
    )
    assert profile["openpi_source_identity_sha256"] == (
        "c20777aa75069d08e61eee3e23451421a8629f97954e1cfd6500ddfcc08438e5"
    )

    background = profile["openpi_wrist_background_augmentation"]
    assert background["mask_source"] == "wrist_final_keep_mask_paths"
    assert background["wrist_depth_mask_paths_fallback_allowed"] is False
    assert background["strict_media_integrity"] is True
    assert _corrected_wrist_background_overrides(profile) == {
        "background_provider": "mixed",
        "mixed_original_prob": 0.1,
        "real_image_prob": 0.4,
        "mixed_black_prob": 0.0,
        "wrist_pair_mode": "shared_scene_family",
        "real_image_root": background["real_image_root"],
        "real_image_manifest": background["real_image_manifest"],
        "strict_media_integrity": True,
    }
    assert profile["right_wrist_intrinsics_overrides"] == {}
    intrinsics = profile["wrist_intrinsics_overrides"]
    assert set(intrinsics) == {
        str(item["source_dataset_id"]) for item in profile["sources"]
    }
    assert all(
        set(source_overrides) == {"left_wrist", "right_wrist"}
        for source_overrides in intrinsics.values()
    )
    assert profile["openpi_final_mask_root_layout"] == (
        "source_dataset_id_partitioned_v1"
    )
