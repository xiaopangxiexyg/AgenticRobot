from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_interface.logical_task_dataset import (
    load_logical_task_dataset_manifest,
)
from deployments.openpi.training.prepared_skill_launch import (
    STANDARD_NON_RTC_TRAINING_MODE_V1,
    _load_profile,
)
from deployments.openpi.training.prepared_skill_launch import (
    main as prepared_skill_launch_main,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_MANIFEST = (
    _REPOSITORY / "dataset_interface/logical_task_manifests/"
    "writing_stage1_all_valid_through_2026_08_20_v1.json"
)
_PROFILE = (
    _REPOSITORY / "deployments/openpi/configs/training/"
    "writing_stage1_all_valid_through_2026_08_20_"
    "front_right_wrist_h32_absolute_non_rtc_v1.json"
)


def test_stage1_all_valid_profile_matches_immutable_logical_manifest() -> None:
    manifest = load_logical_task_dataset_manifest(_MANIFEST)
    profile = _load_profile(_PROFILE)
    manifest_sources = {source.source_dataset_id: source for source in manifest.sources}

    assert profile["logical_task_dataset_manifest"] == str(
        _MANIFEST.relative_to(_REPOSITORY)
    )
    assert len(manifest_sources) == 16
    assert set(manifest_sources) == {
        str(source["source_dataset_id"]) for source in profile["sources"]
    }
    expected_weights = {
        "fixed_middle": 1,
        "initial_multi_choice": 4,
        "midcourse_interrupt": 6,
    }
    for profile_source in profile["sources"]:
        source = manifest_sources[str(profile_source["source_dataset_id"])]
        assert Path(profile_source["dataset_root"]).resolve() == source.dataset_root
        assert profile_source["expected_task_id"] == source.source_task_id
        assert profile_source["scenario_class"] == source.scenario_class
        assert profile_source["task_scope"] == source.task_scope
        assert profile_source["expected_raw_episode_count"] == source.raw_episode_count
        assert profile_source["expected_raw_frame_count"] == source.raw_frame_count
        assert (
            profile_source["repeat_weight"] == expected_weights[source.scenario_class]
        )

    assert profile["expected_total_raw_episode_count"] == sum(
        source.raw_episode_count for source in manifest.sources
    )
    assert profile["expected_total_raw_frame_count"] == sum(
        source.raw_frame_count for source in manifest.sources
    )


def test_stage1_all_valid_profile_pins_requested_training_contract() -> None:
    profile = json.loads(_PROFILE.read_text(encoding="utf-8"))
    sources = profile["sources"]

    assert profile["target_progress_step_id"] == "right_acquire_brush"
    assert profile["camera_selection"] == ["front", "right_wrist"]
    assert profile["camera_slot_to_dataset_key"] == {
        "base_0_rgb": "front",
        "right_wrist_0_rgb": "right_wrist",
    }
    assert profile["action_horizon"] == 32
    assert profile["action_pose_mode"] == "absolute"
    assert profile["openpi_training_mode"] == STANDARD_NON_RTC_TRAINING_MODE_V1
    assert profile["openpi_base_config"] == (
        "writing-stage1-front-right-wrist-h32-absolute-standard-v1"
    )

    assert profile["expected_total_eligible_episode_count"] == sum(
        int(source["expected_eligible_episode_count"]) for source in sources
    )
    assert profile["expected_prepared_interval_window_count"] == sum(
        int(source["expected_prepared_interval_window_count"]) for source in sources
    )
    assert profile["expected_prepared_interval_repeated_window_count"] == sum(
        int(source["expected_prepared_interval_window_count"])
        * int(source["repeat_weight"])
        for source in sources
    )
    assert profile["expected_openpi_quality_valid_window_count"] == sum(
        int(source["expected_openpi_quality_valid_window_count"]) for source in sources
    )
    assert profile["expected_openpi_repeated_epoch_sample_count"] == sum(
        int(source["expected_openpi_quality_valid_window_count"])
        * int(source["repeat_weight"])
        for source in sources
    )

    roots = {str(source["dataset_root"]) for source in sources}
    assert all("5_stage" not in root for root in roots)
    assert not any(
        root.endswith(
            (
                "2026-08-19_4_stage_writing_detail",
                "2026-08-20_4_stage_writing_detail",
                "2026-08-21_4_stage_writing_detail",
            )
        )
        for root in roots
    )


def test_stage1_non_rtc_profile_rejects_model_rtc_overlay() -> None:
    with pytest.raises(SystemExit) as error:
        prepared_skill_launch_main(
            [
                "sample-check",
                "--profile",
                str(_PROFILE),
                "--model-rtc-profile",
                "unused-model-rtc-profile.json",
            ]
        )

    assert error.value.code == 2
