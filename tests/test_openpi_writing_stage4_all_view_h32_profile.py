from __future__ import annotations

import json
from pathlib import Path

from deployments.openpi.training.prepared_skill_launch import (
    _corrected_wrist_background_overrides,
    _load_profile,
)
from utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline import (
    _wrist_intrinsics_overlays_for_source,
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
    intrinsics_overrides = profile["wrist_intrinsics_overrides"]
    assert set(intrinsics_overrides) == {
        str(item["source_dataset_id"]) for item in profile["sources"]
    }
    assert all(
        set(source_overrides) == {"left_wrist", "right_wrist"}
        for source_overrides in intrinsics_overrides.values()
    )
    assert {
        (
            camera_override["target_camera_key"],
            camera_override["expected_serial"],
            camera_override["donor_source_dataset_id"],
            camera_override["donor_devices_relative_path"],
            camera_override["donor_camera_key"],
        )
        for source_overrides in intrinsics_overrides.values()
        for camera_override in source_overrides.values()
    } == {
        (
            "left_wrist",
            "260522278055",
            "writing4_2026_08_06_camera_only_merged",
            (
                "meta/source_devices/2026-08-06_dmtac_rm65_bimanual_"
                "position_v2_writing_4_stage/realsense.json"
            ),
            "left_wrist",
        ),
        (
            "right_wrist",
            "261122273924",
            "writing4_2026_08_06_camera_only_merged",
            (
                "meta/source_devices/2026-08-06_dmtac_rm65_bimanual_"
                "position_v2_writing_4_stage/realsense.json"
            ),
            "right_wrist",
        ),
    }
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


def test_dual_wrist_intrinsics_overlays_bind_each_physical_manifest_by_serial(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    donor_root = tmp_path / "donor"
    target_paths = [
        target_root / "meta/source_devices/capture-a/realsense.json",
        target_root / "meta/source_devices/capture-b/realsense.json",
    ]
    target_payload = {
        "cameras": {
            "left_wrist": {
                "serial": "left-serial",
                "semantic_role": "left_wrist",
            },
            "right_wrist": {
                "serial": "right-serial",
                "semantic_role": "right_wrist",
            },
        }
    }
    for path in target_paths:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(target_payload), encoding="utf-8")
    original_target_bytes = [path.read_bytes() for path in target_paths]

    donor_path = donor_root / "meta/source_devices/calibrated/realsense.json"
    donor_path.parent.mkdir(parents=True)
    donor_path.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {
                        "serial_number": "left-serial",
                        "intrinsics": {
                            "width": 640,
                            "height": 480,
                            "fx": 395.0,
                            "fy": 394.0,
                            "ppx": 320.0,
                            "ppy": 240.0,
                            "coeffs": [0.0] * 5,
                        },
                    },
                    "right_wrist": {
                        "serial_number": "right-serial",
                        "intrinsics": {
                            "width": 640,
                            "height": 480,
                            "fx": 393.0,
                            "fy": 392.0,
                            "ppx": 321.0,
                            "ppy": 239.0,
                            "coeffs": [0.0] * 5,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    donor_source_root = tmp_path / "calibrated"
    (donor_root / "meta/build_audit.json").write_text(
        json.dumps(
            {
                "plan": {
                    "episodes": [
                        {
                            "output_episode_index": 0,
                            "source_root": str(donor_source_root),
                            "source_episode_index": 3,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (donor_root / "meta/provenance.jsonl").write_text(
        json.dumps(
            {
                "output_episode_index": 0,
                "source_dataset_root": str(donor_source_root),
                "source_episode_index": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    camera_overrides = {
        camera_key: {
            "target_camera_key": camera_key,
            "expected_serial": expected_serial,
            "donor_source_dataset_id": "donor",
            "donor_devices_relative_path": (
                "meta/source_devices/calibrated/realsense.json"
            ),
            "donor_camera_key": camera_key,
        }
        for camera_key, expected_serial in (
            ("left_wrist", "left-serial"),
            ("right_wrist", "right-serial"),
        )
    }

    episode_overlays, contract = _wrist_intrinsics_overlays_for_source(
        profile={
            "right_wrist_intrinsics_overrides": {},
            "wrist_intrinsics_overrides": {"target": camera_overrides},
        },
        profile_sources={"target": target_root, "donor": donor_root},
        source_id="target",
        dataset_root=target_root,
        overlay_base=tmp_path / "derived/calibration_overlays",
        episodes=(0, 1),
        camera_keys=("left_wrist", "right_wrist"),
        devices_by_episode={0: target_paths[0], 1: target_paths[1]},
    )

    assert episode_overlays is not None
    assert contract is not None
    assert contract["schema"] == (
        "dexteleop.serial_bound_wrist_intrinsics_overlays.v2"
    )
    assert len(contract["target_bindings"]) == 2
    assert episode_overlays[0] != episode_overlays[1]
    for episode_index, overlay_path in episode_overlays.items():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        assert overlay["cameras"]["left_wrist"]["intrinsics"]["fx"] == 395.0
        assert overlay["cameras"]["right_wrist"]["intrinsics"]["fx"] == 393.0
        assert contract["target_bindings"][episode_index]["overlay_sha256"]
    assert [path.read_bytes() for path in target_paths] == original_target_bytes
