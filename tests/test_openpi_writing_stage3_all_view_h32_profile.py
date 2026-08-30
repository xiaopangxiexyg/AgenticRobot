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
    "writing_stage3_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
)


def test_stage3_all_view_h32_profile_uses_the_canonical_snapshot() -> None:
    profile = _load_profile(_PROFILE)
    sources = profile["sources"]
    by_id = {str(item["source_dataset_id"]): item for item in sources}

    assert list(by_id) == [
        "writing4_2026_08_03_camera_only",
        "writing4_2026_08_04_camera_only",
        "writing4_2026_08_05_camera_only",
        "writing4_2026_08_06_camera_only_merged",
        "writing4_2026_08_07_camera_only_merged",
        "writing3_2026_08_25_camera_only_merged",
        "writing4_2026_08_27_camera_only_merged",
    ]
    assert [item["expected_raw_episode_count"] for item in sources] == [
        207,
        355,
        261,
        294,
        331,
        563,
        56,
    ]
    assert [item["expected_raw_frame_count"] for item in sources] == [
        88732,
        142576,
        102887,
        112817,
        113517,
        75071,
        36636,
    ]
    assert [item["expected_eligible_episode_count"] for item in sources] == [
        186,
        344,
        246,
        289,
        322,
        513,
        36,
    ]
    assert [item["expected_complete_h32_window_count"] for item in sources] == [
        12304,
        19620,
        13690,
        17360,
        14543,
        36431,
        4880,
    ]
    assert [
        item["expected_openpi_quality_valid_window_count"]
        for item in sources
    ] == [11190, 14911, 9991, 13020, 9975, 24496, 1506]

    assert profile["expected_total_raw_episode_count"] == 2067
    assert profile["expected_total_raw_frame_count"] == 672236
    assert profile["expected_total_eligible_episode_count"] == 1936
    assert profile["expected_complete_h32_window_count"] == 118828
    assert profile["expected_openpi_quality_valid_window_count"] == 85089
    assert profile["expected_openpi_repeated_epoch_sample_count"] == 85089
    assert len(profile["annotation_selection"]) == 2014
    assert profile["complete_horizon_window_exclusions"] == []
    assert profile["source_snapshot"]["selector_report_sha256"] == (
        "5532d30bc783e805f3ab5952459d08dda153b5b468acb1bab0010165483a5753"
    )
    assert profile["source_snapshot"]["selector_audit_receipt_sha256"] == (
        "f80c157e338bc3402113383d385d066b13f3516dafb80e4b867ef887f11f063d"
    )
    assert profile["source_snapshot"]["raw_and_derived_overlap_allowed"] is False
    assert len({item["dataset_root"] for item in sources}) == len(sources)
    assert by_id["writing3_2026_08_25_camera_only_merged"][
        "expected_task_id"
    ] == "3rd_stage_writing_detail"


def test_stage3_quality_window_contract_fails_closed(tmp_path: Path) -> None:
    profile = json.loads(_PROFILE.read_text(encoding="utf-8"))
    profile["sources"][0][
        "expected_openpi_quality_valid_window_count"
    ] += 1
    mismatched = tmp_path / "mismatched-stage3-profile.json"
    mismatched.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="source counts differ"):
        _load_profile(mismatched)


def test_stage3_all_view_h32_profile_pins_training_and_mask_contracts() -> None:
    profile = _load_profile(_PROFILE)

    assert profile["target_skill_id"] == "writing.write_with_brush"
    assert profile["target_progress_step_id"] == "right_write"
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
    assert profile["openpi_source_revision"] == (
        "940cdc87516a7cf07db7ae46147548c1cf8382ee"
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
    stage3_intrinsics = intrinsics[
        "writing3_2026_08_25_camera_only_merged"
    ]
    assert {
        camera: (
            value["expected_serial"],
            value["donor_source_dataset_id"],
            value["donor_camera_key"],
        )
        for camera, value in stage3_intrinsics.items()
    } == {
        "left_wrist": (
            "260522278055",
            "writing4_2026_08_06_camera_only_merged",
            "left_wrist",
        ),
        "right_wrist": (
            "261122273924",
            "writing4_2026_08_06_camera_only_merged",
            "right_wrist",
        ),
    }
    assert profile["openpi_final_mask_root_layout"] == (
        "source_dataset_id_partitioned_v1"
    )
