from __future__ import annotations

import json
from pathlib import Path


_REPOSITORY = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPOSITORY / "deployments/openpi/configs/training"
_PROFILE = _CONFIG_ROOT / (
    "writing_all_four_stage_prompt_family_front_right_wrist_h32_absolute_non_rtc_v1.json"
)
_EXPECTED_STAGES = ("stage1", "stage2", "stage3", "stage4")
_EXPECTED_SOURCES = (
    "fixed_middle_2026_08_03_part1",
    "fixed_middle_2026_08_03_part2",
    "fixed_middle_2026_08_04",
    "fixed_middle_2026_08_05",
    "fixed_middle_2026_08_06_part1",
    "fixed_middle_2026_08_06_part2",
    "fixed_middle_2026_08_06_part3",
    "fixed_middle_2026_08_07_physical",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_four_stage_profile_binds_requested_training_contract() -> None:
    profile = _read(_PROFILE)
    assert profile["schema"] == (
        "dexteleop.openpi_prepared_prompt_family_training_profile.v2"
    )
    assert profile["action_horizon"] == 32
    assert profile["action_pose_mode"] == "absolute"
    assert profile["camera_selection"] == ["front", "right_wrist"]
    assert profile["camera_slot_to_dataset_key"] == {
        "base_0_rgb": "front",
        "right_wrist_0_rgb": "right_wrist",
    }
    assert profile["openpi_training_mode"] == "standard_non_rtc_v1"
    assert profile["openpi_num_train_steps"] == 600000
    assert profile["openpi_num_workers"] == 4
    assert profile["openpi_checkpoint_base_dir"] == (
        "/home/user/zc/checkpoints/openpi"
    )
    assert profile["expected_union_episode_count"] == 1402
    assert profile["expected_total_window_count"] == 241436
    assert profile["expected_effective_epoch_window_count"] == 4700

    background = profile["openpi_wrist_background_augmentation"]
    assert background["schema"] == "dexteleop.corrected_wrist_background.v1"
    assert background["mask_source"] == "wrist_final_keep_mask_paths"
    assert background["wrist_depth_mask_paths_fallback_allowed"] is False
    assert background["strict_media_integrity"] is True
    assert (
        background["original_probability"]
        + background["real_image_probability"]
        + background["procedural_probability"]
        + background["black_probability"]
    ) == 1.0

    assert tuple(source["source_dataset_id"] for source in profile["sources"]) == (
        _EXPECTED_SOURCES
    )
    assert len(profile["sources"]) == len(_EXPECTED_SOURCES)

    stages = profile["prompt_stages"]
    assert tuple(stage["stage_id"] for stage in stages) == _EXPECTED_STAGES
    for stage in stages:
        assert len(stage["training_prompt_variants"]) == 12
        assert len(stage["held_out_prompt_variants"]) == 4
        assert set(stage["training_prompt_variants"]).isdisjoint(
            stage["held_out_prompt_variants"]
        )
        child = _read(_CONFIG_ROOT / stage["profile"])
        assert child["action_horizon"] == 32
        assert child["action_pose_mode"] == "absolute"
        assert child["camera_selection"] == ["front", "right_wrist"]
        assert child["openpi_training_mode"] == "standard_non_rtc_v1"
        assert child["openpi_num_train_steps"] == 600000
        assert len(child["sources"]) == len(_EXPECTED_SOURCES)
        assert tuple(source["source_dataset_id"] for source in child["sources"]) == (
            _EXPECTED_SOURCES
        )
        child_background = child["openpi_wrist_background_augmentation"]
        assert child_background["mask_source"] == "wrist_final_keep_mask_paths"
        assert child_background["wrist_depth_mask_paths_fallback_allowed"] is False
        assert child_background["strict_media_integrity"] is True

    exclusions = {
        stage["stage_id"]: _read(_CONFIG_ROOT / stage["profile"]).get(
            "complete_horizon_window_exclusions", []
        )
        for stage in stages
    }
    assert exclusions["stage1"] == []
    assert exclusions["stage3"] == []
    assert len(exclusions["stage2"]) == 1
    assert exclusions["stage2"][0]["reason"] == "NO_COMPLETE_H32_SKILL_WINDOW"
    assert exclusions["stage2"][0]["source_episode_index"] == 154
    assert len(exclusions["stage4"]) == 1
    assert exclusions["stage4"][0]["reason"] == "NO_COMPLETE_H32_SKILL_WINDOW"
    assert exclusions["stage4"][0]["source_episode_index"] == 210
