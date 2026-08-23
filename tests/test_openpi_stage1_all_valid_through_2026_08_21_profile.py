from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_interface.logical_task_dataset import load_logical_task_dataset_manifest
from deployments.openpi.training.prepared_skill_launch import _load_profile

_REPOSITORY = Path(__file__).resolve().parents[1]
_MANIFEST = (
    _REPOSITORY
    / "dataset_interface/logical_task_manifests/"
    "writing_stage1_all_valid_through_2026_08_21_v1.json"
)
_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/training/"
    "writing_stage1_all_valid_through_2026_08_21_"
    "front_right_wrist_h32_absolute_non_rtc_v1.json"
)


def test_through_2026_08_21_profile_matches_logical_source_contract() -> None:
    manifest = load_logical_task_dataset_manifest(_MANIFEST)
    profile = _load_profile(_PROFILE)
    manifest_sources = {
        source.source_dataset_id: source for source in manifest.sources
    }

    assert len(manifest_sources) == 16
    assert set(manifest_sources) == {
        str(source["source_dataset_id"]) for source in profile["sources"]
    }
    for profile_source in profile["sources"]:
        source = manifest_sources[str(profile_source["source_dataset_id"])]
        assert Path(profile_source["dataset_root"]).resolve() == source.dataset_root
        assert profile_source["expected_task_id"] == source.source_task_id
        assert tuple(profile_source.get("expected_task_id_aliases", ())) == (
            source.source_task_id_aliases
        )
        assert profile_source["expected_raw_episode_count"] == (
            source.raw_episode_count
        )
        assert profile_source["expected_raw_frame_count"] == source.raw_frame_count

    aug07 = manifest_sources["fixed_middle_2026_08_07_physical"]
    aug10 = manifest_sources["fixed_middle_2026_08_10_physical"]
    assert "PhysicalCollections" in str(aug07.dataset_root)
    assert "PhysicalCollections" in str(aug10.dataset_root)
    aug19 = manifest_sources["fixed_middle_detail_stage1_2026_08_19"]
    assert aug19.source_task_id == "4_stage_writing_detail"
    assert aug19.source_task_id_aliases == (
        "dmtac_rm65_bimanual_position_v1",
    )
    aug20 = manifest_sources["fixed_middle_detail_stage1_2026_08_20_physical"]
    assert "PhysicalCollections" in str(aug20.dataset_root)
    assert aug20.source_task_id_aliases == ()
    aug21 = manifest_sources["fixed_middle_detail_stage1_2026_08_21"]
    assert aug21.source_task_id == "4_stage_writing_detail"
    assert aug21.source_task_id_aliases == (
        "dmtac_rm65_bimanual_position_v1",
    )


def test_through_2026_08_21_profile_totals_are_derived_from_sources() -> None:
    profile = _load_profile(_PROFILE)
    sources = profile["sources"]

    assert profile["camera_selection"] == ["front", "right_wrist"]
    assert profile["camera_slot_to_dataset_key"] == {
        "base_0_rgb": "front",
        "right_wrist_0_rgb": "right_wrist",
    }
    assert profile["action_horizon"] == 32
    assert profile["action_pose_mode"] == "absolute"
    assert profile["openpi_training_mode"] == "standard_non_rtc_v1"
    assert profile["openpi_num_workers"] == 4

    assert profile["expected_total_raw_episode_count"] == sum(
        int(source["expected_raw_episode_count"]) for source in sources
    )
    assert profile["expected_total_raw_frame_count"] == sum(
        int(source["expected_raw_frame_count"]) for source in sources
    )
    assert profile["expected_total_eligible_episode_count"] == sum(
        int(source["expected_eligible_episode_count"]) for source in sources
    )
    assert profile["expected_prepared_interval_window_count"] == sum(
        int(source["expected_prepared_interval_window_count"])
        for source in sources
    )
    assert profile["expected_prepared_interval_repeated_window_count"] == sum(
        int(source["expected_prepared_interval_window_count"])
        * int(source["repeat_weight"])
        for source in sources
    )
    assert profile["expected_openpi_quality_valid_window_count"] == sum(
        int(source["expected_openpi_quality_valid_window_count"])
        for source in sources
    )
    assert profile["expected_openpi_repeated_epoch_sample_count"] == sum(
        int(source["expected_openpi_quality_valid_window_count"])
        * int(source["repeat_weight"])
        for source in sources
    )
    by_id = {str(source["source_dataset_id"]): source for source in sources}
    assert by_id["fixed_middle_detail_stage1_2026_08_19"][
        "expected_openpi_quality_valid_window_count"
    ] == 4722
    assert by_id["fixed_middle_detail_stage1_2026_08_20_physical"][
        "expected_openpi_quality_valid_window_count"
    ] == 10362
    assert by_id["fixed_middle_detail_stage1_2026_08_21"][
        "expected_openpi_quality_valid_window_count"
    ] == 36709
    assert profile["expected_openpi_quality_valid_window_count"] == 167878
    assert profile["expected_openpi_repeated_epoch_sample_count"] == 268000


def test_through_2026_08_21_profile_rejects_untyped_task_aliases(
    tmp_path: Path,
) -> None:
    payload = json.loads(_PROFILE.read_text(encoding="utf-8"))
    payload["sources"][-1]["expected_task_id_aliases"] = "legacy-task"
    invalid = tmp_path / "invalid-profile.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="expected_task_id_aliases"):
        _load_profile(invalid)
