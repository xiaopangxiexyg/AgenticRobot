from __future__ import annotations

from pathlib import Path

from deployments.openpi.training.prepared_skill_launch import (
    _corrected_wrist_background_overrides,
    _load_profile,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/training/"
    "writing_stage4_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
)


def test_stage4_all_view_h32_profile_uses_the_canonical_snapshot() -> None:
    profile = _load_profile(_PROFILE)
    sources = profile["sources"]
    by_id = {str(item["source_dataset_id"]): item for item in sources}

    assert list(by_id) == [
        "writing4_2026_08_03_camera_only",
        "writing4_2026_08_04_camera_only",
        "writing4_2026_08_05_camera_only",
        "writing4_2026_08_06_camera_only_merged",
        "writing4_2026_08_07_camera_only_merged",
        "writing4_2026_08_27_camera_only_merged",
    ]
    assert [item["expected_raw_episode_count"] for item in sources] == [
        207,
        355,
        261,
        294,
        331,
        56,
    ]
    assert [item["expected_raw_frame_count"] for item in sources] == [
        88732,
        142576,
        102887,
        112817,
        113517,
        36636,
    ]
    assert [item["expected_eligible_episode_count"] for item in sources] == [
        186,
        344,
        244,
        289,
        317,
        26,
    ]
    assert [item["expected_complete_h32_window_count"] for item in sources] == [
        8613,
        15661,
        9412,
        13411,
        12030,
        3872,
    ]
    assert profile["expected_total_raw_episode_count"] == 1504
    assert profile["expected_total_raw_frame_count"] == 597165
    assert profile["expected_total_eligible_episode_count"] == 1406
    assert profile["expected_complete_h32_window_count"] == 62999
    assert len(profile["annotation_selection"]) == 1490
    assert profile["source_snapshot"]["selector_report_sha256"] == (
        "f237fc9ebb7a30ca8637e34d9fe5eca5cd1da86c20aed239eda7e17d013a4b17"
    )
    assert profile["source_snapshot"]["raw_and_derived_overlap_allowed"] is False
    assert len({item["dataset_root"] for item in sources}) == len(sources)
    assert all(
        "camera_semantic_prepared_views" in item["dataset_root"]
        or "/PhysicalCollections/camera_only/" in item["dataset_root"]
        for item in sources
    )
    assert by_id["writing4_2026_08_27_camera_only_merged"][
        "expected_task_id"
    ] == "4_stage_writing_detail"


def test_stage4_all_view_h32_profile_pins_training_and_mask_contracts() -> None:
    profile = _load_profile(_PROFILE)

    assert profile["target_skill_id"] == "writing.return_brush"
    assert profile["target_progress_step_id"] == "right_return_brush"
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
    assert profile["openpi_final_mask_root_layout"] == (
        "source_dataset_id_partitioned_v1"
    )

    exclusions = profile["complete_horizon_window_exclusions"]
    assert {
        (
            item["source_dataset_id"],
            item["source_episode_index"],
            item["max_skill_interval_length"],
        )
        for item in exclusions
    } == {
        ("writing4_2026_08_07_camera_only_merged", 210, 26),
        ("writing4_2026_08_27_camera_only_merged", 31, 0),
    }
