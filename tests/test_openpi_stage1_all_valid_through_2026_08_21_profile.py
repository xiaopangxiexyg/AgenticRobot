from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_interface.logical_task_dataset import load_logical_task_dataset_manifest
from deployments.openpi.training.prepared_skill import OpenPIPreparedSkillError
from deployments.openpi.training.prepared_skill_launch import (
    _activate_openpi_checkout,
    _bind_source_partitioned_final_masks_to_dataset,
    _corrected_wrist_background_overrides,
    _load_profile,
)
from utils.dataset_postprocess.wrist_cam_adjust.depth_mask.core import (
    pack_binary_mask,
)
from utils.dataset_postprocess.wrist_cam_adjust.depth_mask.postprocess import (
    DepthMaskPostprocessConfig,
    process_mask_stream,
)
from utils.dataset_postprocess.wrist_cam_adjust.depth_mask.visualize import (
    MaskStream,
)
from utils.dataset_postprocess.wrist_cam_adjust.geometry_mask.dataset import (
    _clone_non_video_files,
    _clone_unprocessed_videos,
)
from utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline import (
    _intrinsics_overlay_for_source,
)

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
    run_root = Path(
        "/mnt/nas2/zc/stage1_openpi_runs/20260824T0709_snapshot_v1"
    )
    assert Path(profile["organized_dataset_root"]) == (
        run_root
        / "workspace/prepared/"
        "writing_stage1_all_valid_through_2026_08_21_"
        "front_right_wrist_h32_absolute_non_rtc_v1"
    )
    assert profile["openpi_final_mask_root_layout"] == (
        "source_dataset_id_partitioned_v1"
    )
    assert Path(profile["openpi_final_mask_root"]) == (
        run_root / "workspace/derived/final_keep_masks"
    )
    assert Path(profile["openpi_assets_base_dir"]) == (
        run_root / "artifacts/assets"
    )
    assert Path(profile["openpi_checkpoint_base_dir"]) == (
        run_root / "artifacts/checkpoints"
    )
    assert Path(profile["openpi_checkout"]) == (
        Path("/home/user/zc/stage1_openpi_runs/20260824T0709_snapshot_v1")
        / "code-final7/AgenticRobot/model/openpi"
    )
    assert profile["openpi_source_revision"] == (
        "97b3720515b18e94697573ba55caa6c1bc28af69"
    )
    assert profile["openpi_source_identity_sha256"] == (
        "17bb5ec5529c2cbbf7f9b0b0b61f30ea53b2d2fc671b7e8874a56491f5130442"
    )
    background = profile["openpi_wrist_background_augmentation"]
    assert background["original_probability"] == 0.1
    assert background["real_image_probability"] == 0.4
    assert background["procedural_probability"] == 0.5
    assert background["mask_source"] == "wrist_final_keep_mask_paths"
    assert background["wrist_depth_mask_paths_fallback_allowed"] is False
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
    intrinsics_overrides = profile["right_wrist_intrinsics_overrides"]
    assert set(intrinsics_overrides) == {
        "fixed_middle_2026_08_03_part1",
        "fixed_middle_2026_08_03_part2",
    }
    assert {
        (
            item["target_camera_key"],
            item["expected_serial"],
            item["donor_source_dataset_id"],
            item["donor_camera_key"],
        )
        for item in intrinsics_overrides.values()
    } == {
        (
            "right_wrist",
            "261122273924",
            "fixed_middle_2026_08_04",
            "left_wrist",
        )
    }

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


def test_through_2026_08_21_profile_rejects_background_contract_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads(_PROFILE.read_text(encoding="utf-8"))
    payload["openpi_wrist_background_augmentation"][
        "real_image_probability"
    ] = 0.5
    invalid = tmp_path / "invalid-background-profile.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corrected 10/40/50"):
        _load_profile(invalid)


def test_openpi_checkout_activation_rejects_revision_drift(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "openpi"
    (checkout / "src/openpi").mkdir(parents=True)
    identity_path = checkout / "OPENPI_SOURCE_IDENTITY.json"
    identity_path.write_text(
        json.dumps(
            {
                "schema": "dexteleop.openpi_source_identity.v1",
                "revision": "wrong-revision",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source identity differs"):
        _activate_openpi_checkout(
            {
                "openpi_checkout": str(checkout),
                "openpi_source_revision": "expected-revision",
                "openpi_source_identity_sha256": hashlib.sha256(
                    identity_path.read_bytes()
                ).hexdigest(),
            }
        )


@pytest.mark.parametrize("package_name", ("openpi", "openpi_client"))
def test_openpi_checkout_activation_rejects_foreign_loaded_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
) -> None:
    checkout = tmp_path / "openpi"
    (checkout / "src/openpi").mkdir(parents=True)
    foreign = tmp_path / f"foreign/{package_name}/__init__.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        package_name,
        SimpleNamespace(__file__=str(foreign)),
    )

    with pytest.raises(RuntimeError, match="outside the profile checkout"):
        _activate_openpi_checkout({"openpi_checkout": str(checkout)})


def test_openpi_checkout_activation_rejects_client_contract_drift(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "openpi"
    config_path = checkout / "src/openpi/training/config.py"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("", encoding="utf-8")
    client_contract = (
        checkout
        / "packages/openpi-client/src/openpi_client/history_state_contract.py"
    )
    client_contract.parent.mkdir(parents=True)
    client_contract.write_text("contract = 'actual'\n", encoding="utf-8")
    identity_path = checkout / "OPENPI_SOURCE_IDENTITY.json"
    identity_path.write_text(
        json.dumps(
            {
                "schema": "dexteleop.openpi_source_identity.v1",
                "revision": "expected-revision",
                "workspace_source_roots": [
                    "src",
                    "packages/openpi-client/src",
                ],
                "training_config_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                "openpi_client_history_state_contract_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="client contract content differs"):
        _activate_openpi_checkout(
            {
                "openpi_checkout": str(checkout),
                "openpi_source_revision": "expected-revision",
                "openpi_source_identity_sha256": hashlib.sha256(
                    identity_path.read_bytes()
                ).hexdigest(),
            }
        )


def test_source_partitioned_final_masks_bind_before_pipeline_open(
    tmp_path: Path,
) -> None:
    source_a = (tmp_path / "source-a" / "shared-name").resolve()
    source_b = (tmp_path / "source-b" / "shared-name").resolve()
    final_base = (tmp_path / "final").resolve()
    child_a = SimpleNamespace(
        dataset_root=source_a,
        final_mask_root=tmp_path / "wrong",
        explicit_mask_keys=("wrist_final_keep_mask_paths",),
        _masked_pipeline=None,
    )
    child_b = SimpleNamespace(
        dataset_root=source_b,
        final_mask_root=tmp_path / "wrong",
        explicit_mask_keys=("wrist_final_keep_mask_paths",),
        _masked_pipeline=None,
    )
    prepared = SimpleNamespace(
        dataset=SimpleNamespace(_datasets=(child_a, child_b))
    )

    result = _bind_source_partitioned_final_masks_to_dataset(
        prepared,
        {
            source_a: ("source_a", final_base / "source_a"),
            source_b: ("source_b", final_base / "source_b"),
        },
    )

    assert result is prepared
    assert child_a.final_mask_root == final_base / "source_a"
    assert child_b.final_mask_root == final_base / "source_b"


def test_source_partitioned_final_masks_reject_raw_depth_fallback(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "source").resolve()
    prepared = SimpleNamespace(
        dataset=SimpleNamespace(
            dataset_root=source,
            final_mask_root=tmp_path / "wrong",
            explicit_mask_keys=("wrist_depth_mask_paths",),
            _masked_pipeline=None,
        )
    )

    with pytest.raises(OpenPIPreparedSkillError, match="forbids"):
        _bind_source_partitioned_final_masks_to_dataset(
            prepared,
            {source: ("source", tmp_path / "final" / "source")},
        )


def test_depth_postprocess_repairs_only_complete_trailing_frames(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "raw.bitpack.bin"
    frames = [
        np.asarray([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8),
        np.asarray([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8),
        np.asarray([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.uint8),
    ]
    source_path.write_bytes(b"".join(pack_binary_mask(frame) for frame in frames))
    stream = MaskStream(
        label="right_wrist",
        stream_path=source_path,
        height=1,
        width=8,
        frame_count=2,
        packed_bytes_per_frame=1,
    )

    result = process_mask_stream(
        stream,
        tmp_path / "derived" / "right_wrist",
        config=DepthMaskPostprocessConfig(
            close_radius_px=0,
            median_ksize=0,
            fill_hole_max_area_px=0,
            temporal_gap_fill_frames=0,
        ),
    )

    assert result.frame_count == 2
    assert result.source_declared_bytes == 2
    assert result.source_actual_bytes == 3
    assert result.source_trailing_bytes == 1
    assert result.stream_path.stat().st_size == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["integrity"]["read_policy"] == (
        "exact_declared_frames_v1"
    )
    assert manifest["source"]["integrity"]["source_modified"] is False


def test_intrinsics_overlay_binds_donor_by_serial_without_source_write(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    donor_root = tmp_path / "donor"
    target_path = target_root / "meta/devices/realsense.json"
    donor_path = donor_root / "meta/devices/realsense.json"
    target_path.parent.mkdir(parents=True)
    donor_path.parent.mkdir(parents=True)
    target_path.write_text(
        json.dumps(
            {
                "cameras": {
                    "right_wrist": {
                        "serial": "serial-a",
                        "semantic_role": "right_wrist",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    donor_path.write_text(
        json.dumps(
            {
                "left_wrist": {
                    "serial_number": "serial-a",
                    "intrinsics": {
                        "width": 640,
                        "height": 480,
                        "fx": 395.0,
                        "fy": 394.0,
                        "ppx": 320.0,
                        "ppy": 240.0,
                        "coeffs": [0.0] * 5,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    original_target = target_path.read_bytes()

    overlay_path, contract = _intrinsics_overlay_for_source(
        profile={
            "right_wrist_intrinsics_overrides": {
                "target": {
                    "target_camera_key": "right_wrist",
                    "expected_serial": "serial-a",
                    "donor_source_dataset_id": "donor",
                    "donor_camera_key": "left_wrist",
                }
            }
        },
        profile_sources={"target": target_root, "donor": donor_root},
        source_id="target",
        dataset_root=target_root,
        overlay_base=tmp_path / "derived/calibration_overlays",
    )

    assert overlay_path is not None
    assert contract is not None
    assert contract["bound_serial"] == "serial-a"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay["right_wrist"]["intrinsics"]["fx"] == 395.0
    assert overlay["right_wrist"]["serial"] == "serial-a"
    assert target_path.read_bytes() == original_target


def test_geometry_copy_mode_skip_does_not_walk_or_clone_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "meta").mkdir(parents=True)
    (source / "videos/chunk-000").mkdir(parents=True)
    (source / "meta/info.json").write_text("{}", encoding="utf-8")
    (source / "videos/chunk-000/episode.mp4").write_bytes(b"video")

    _clone_non_video_files(source, target, copy_mode="skip", overwrite=False)
    _clone_unprocessed_videos(
        source,
        target,
        processed_relative_paths=set(),
        copy_mode="skip",
        overwrite=False,
    )

    assert not target.exists()
