from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from collection_runtime.progress.profiles import load_managed_task_profile
from dataset_contracts.camera_semantic_prepared_view import (
    CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH,
    CameraSemanticPreparedViewContractError,
    CameraSemanticPreparedViewFileV1,
    CameraSemanticPreparedViewManifestV1,
)
from dataset_interface.camera_semantic_prepared_view import (
    load_camera_semantic_prepared_view,
)
from dataset_interface.progress_skill_projection import (
    load_builtin_writing_five_stage_dipping_skill_projection_catalog,
    validate_progress_skill_projection,
)
from deployments.openpi.model_rtc_profile import load_prepared_model_rtc_profile
from deployments.openpi.training.prepared_skill_launch import (
    _load_profile,
    _model_rtc_profile_matches_prepared_profile,
    _skill_projection_catalog,
)
from utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline import (
    _physical_devices_by_episode,
    _wrist_intrinsics_overlays_for_source,
)


_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = (
    _ROOT
    / "deployments/openpi/configs/training/"
    "writing_dipping_all_valid_all_view_h32_absolute_non_rtc_100k_by_date_v1.json"
)
_MODEL_RTC_PROFILE = (
    _ROOT
    / "deployments/openpi/configs/model_rtc/"
    "writing_dipping_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json"
)
_CALIBRATION_DEVICES_SHA256 = (
    "aa2f2e1d78314e8c93bcff7dad92b7dbf97cfa69e170c67f6b6f5d90eca5f023"
)


def _write_minimal_camera_prepared_view(root: Path) -> Path:
    episodes_path = root / "meta/episodes.jsonl"
    episodes_path.parent.mkdir(parents=True)
    episodes_path.write_text('{"episode_index": 0}\n', encoding="utf-8")
    file_record = CameraSemanticPreparedViewFileV1(
        relative_path="meta/episodes.jsonl",
        size_bytes=episodes_path.stat().st_size,
        sha256=hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
    )
    manifest = CameraSemanticPreparedViewManifestV1(
        source_dataset_name="dipping-source",
        source_dataset_identity_hash="1" * 64,
        selected_source_inventory_hash="2" * 64,
        camera_binding_hash="3" * 64,
        camera_materialization_receipt_hash="4" * 64,
        plan_hash="5" * 64,
        episode_indices=(0,),
        files=(file_record,),
    )
    manifest_path = root / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_by_date_entry(root: Path, manifest_path: Path, **updates: object) -> None:
    payload: dict[str, object] = {
        "dataset_relative_path": root.name,
        "destination_relative_path": root.name,
        "direct_dataset_root": True,
        "prepared_view_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    payload.update(updates)
    payload["entry_semantic_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    (root / "ENTRY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_dipping_skill_projection_matches_managed_five_stage_contract() -> None:
    bundle = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_five_stage_progress_detailed"
    )
    catalog = load_builtin_writing_five_stage_dipping_skill_projection_catalog()
    assert _skill_projection_catalog(_load_profile(_PROFILE)) == catalog
    assert len(catalog.skills) == 1
    assert len(catalog.projections) == 1
    skill = catalog.skills[0]
    projection = catalog.projections[0]
    assert skill.skill_id == "writing.dip_brush_in_ink"
    assert projection.source_progress_step_id == "right_dip_ink"
    assert projection.source_stage_ids == (
        "right_dip_ink.approach",
        "right_dip_ink.dip",
        "right_dip_ink.depart",
    )
    validate_progress_skill_projection(
        progress_profile=bundle.progress_profile,
        stage_profile=bundle.stage_profile,
        skill_contract=skill,
        projection=projection,
    )


def test_dipping_training_profile_is_all_view_h32_absolute_and_strict() -> None:
    profile = _load_profile(_PROFILE)

    assert profile["expected_total_raw_episode_count"] == 773
    assert profile["expected_total_raw_frame_count"] == 81576
    assert profile["expected_total_eligible_episode_count"] == 756
    assert len(profile["annotation_selection"]) == 773
    assert sum(source["expected_eligible_episode_count"] for source in profile["sources"]) == 756
    assert profile["camera_selection"] == ["front", "left_wrist", "right_wrist"]
    assert profile["action_horizon"] == 32
    assert profile["action_pose_mode"] == "absolute"
    assert profile["openpi_training_mode"] == "standard_non_rtc_v1"
    assert profile["openpi_num_train_steps"] == 100000
    assert profile["openpi_num_workers"] == 2
    background = profile["openpi_wrist_background_augmentation"]
    assert background["mask_source"] == "wrist_final_keep_mask_paths"
    assert background["wrist_depth_mask_paths_fallback_allowed"] is False
    assert background["strict_media_integrity"] is True
    assert (
        background["original_probability"],
        background["real_image_probability"],
        background["procedural_probability"],
        background["black_probability"],
    ) == (0.1, 0.4, 0.5, 0.0)
    assert profile["right_wrist_intrinsics_overrides"] == {}
    assert set(profile["wrist_intrinsics_overrides"]) == {
        source["source_dataset_id"] for source in profile["sources"]
    }
    for camera_overrides in profile["wrist_intrinsics_overrides"].values():
        assert set(camera_overrides) == {"left_wrist", "right_wrist"}
        assert {
            override["expected_donor_devices_sha256"]
            for override in camera_overrides.values()
        } == {_CALIBRATION_DEVICES_SHA256}


def test_dipping_model_rtc_profile_is_data_bound_but_checkpoint_independent() -> None:
    prepared = _load_profile(_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    assert _model_rtc_profile_matches_prepared_profile(prepared, model_rtc)
    assert model_rtc["training_implementation"] == "mask_aware_hard_flow_prefix_v2"
    assert model_rtc["rtc_prefix_mask_conditioning"] is True
    assert model_rtc["sampling_spec"]["delay_prefix_pairs"] == [
        [1, 0],
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
    forbidden = {
        "checkpoint_dir",
        "base_checkpoint_dir",
        "checkpoint_step",
        "continuation_contract_hash",
        "prepared_final_mask_root",
        "prepared_final_mask_inventory_receipt",
        "prepared_final_mask_inventory_receipt_sha256",
    }
    assert forbidden.isdisjoint(model_rtc)


def test_camera_prepared_view_accepts_hash_bound_by_date_entry(tmp_path: Path) -> None:
    root = tmp_path / "2026-09-14__writing-detail-s3"
    manifest_path = _write_minimal_camera_prepared_view(root)
    _write_by_date_entry(root, manifest_path)

    loaded = load_camera_semantic_prepared_view(root, verify_payload_hashes=True)

    assert loaded.episode_indices == (0,)
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        CameraSemanticPreparedViewContractError,
        match="file inventory differs",
    ):
        load_camera_semantic_prepared_view(root, verify_payload_hashes=True)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"direct_dataset_root": False}, "direct dataset root"),
        ({"dataset_relative_path": "wrong-root"}, "dataset_relative_path"),
        ({"prepared_view_manifest_sha256": "0" * 64}, "different manifest"),
        ({"tactile_receipt": "not-allowed"}, "tactile-specific metadata"),
    ),
)
def test_camera_prepared_view_rejects_invalid_by_date_entry(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "2026-09-14__writing-detail-s3"
    manifest_path = _write_minimal_camera_prepared_view(root)
    _write_by_date_entry(root, manifest_path, **updates)

    with pytest.raises(CameraSemanticPreparedViewContractError, match=message):
        load_camera_semantic_prepared_view(root, verify_payload_hashes=True)


def test_camera_prepared_view_rejects_by_date_entry_hash_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "2026-09-14__writing-detail-s3"
    manifest_path = _write_minimal_camera_prepared_view(root)
    _write_by_date_entry(root, manifest_path)
    entry_path = root / "ENTRY.json"
    payload = json.loads(entry_path.read_text(encoding="utf-8"))
    payload["destination_relative_path"] = "tampered"
    entry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        CameraSemanticPreparedViewContractError,
        match="semantic hash mismatch",
    ):
        load_camera_semantic_prepared_view(root, verify_payload_hashes=True)


def _write_flattened_physical_binding(
    dataset_root: Path,
    source_root: Path,
    *,
    intermediate_source_root: Path | None = None,
    alias: str = "/realmandmtac-provenance/unit-source",
) -> None:
    meta = dataset_root / "meta"
    copied = meta / "source_devices" / source_root.name / "realsense.json"
    copied.parent.mkdir(parents=True)
    copied.write_text(
        json.dumps({"left_wrist": {}, "right_wrist": {}}),
        encoding="utf-8",
    )
    (meta / "build_audit.json").write_text(
        json.dumps(
            {
                "plan": {
                    "episodes": [
                        {
                            "output_episode_index": 0,
                            "source_root": str(source_root),
                            "source_episode_index": 7,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (meta / "provenance.jsonl").write_text(
        json.dumps(
            {
                "output_episode_index": 0,
                "source_dataset_root": alias,
                "source_dataset_root_kind": "stable_non_sensitive_alias",
                "intermediate_source_dataset_root": str(
                    intermediate_source_root or source_root
                ),
                "raw_source_dataset_root": str(dataset_root.parent / "raw-source"),
                "source_episode_index": 7,
                "provenance_flattening": (
                    "by_date_parts_consolidated_after_verified_merge"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_flattened_by_date_provenance_binds_copied_camera_intrinsics(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "canonical-by-date"
    source_root = tmp_path / "retired-part01"
    _write_flattened_physical_binding(dataset_root, source_root)

    mapping, contract = _physical_devices_by_episode(dataset_root, (0,))

    copied = (
        dataset_root
        / "meta"
        / "source_devices"
        / source_root.name
        / "realsense.json"
    )
    assert mapping == {0: copied.resolve()}
    assert contract["source_manifests"][0]["provenance_source_binding"] == {
        "mode": "reviewed_flattened_by_date_source",
        "source_dataset_root": "/realmandmtac-provenance/unit-source",
        "source_dataset_root_kind": "stable_non_sensitive_alias",
        "intermediate_source_dataset_root": str(source_root),
        "raw_source_dataset_root": str(dataset_root.parent / "raw-source"),
    }


@pytest.mark.parametrize(
    ("intermediate_name", "alias"),
    (
        ("wrong-part", "/realmandmtac-provenance/unit-source"),
        ("retired-part01", "/unexpected-provenance/unit-source"),
    ),
)
def test_flattened_by_date_provenance_rejects_unreviewed_source_binding(
    tmp_path: Path,
    intermediate_name: str,
    alias: str,
) -> None:
    dataset_root = tmp_path / "canonical-by-date"
    source_root = tmp_path / "retired-part01"
    _write_flattened_physical_binding(
        dataset_root,
        source_root,
        intermediate_source_root=tmp_path / intermediate_name,
        alias=alias,
    )

    with pytest.raises(ValueError, match="flattened physical provenance"):
        _physical_devices_by_episode(dataset_root, (0,))


def test_external_wrist_intrinsics_donor_is_serial_and_hash_bound(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "training-source"
    target_devices = target_root / "meta/devices/realsense.json"
    target_devices.parent.mkdir(parents=True)
    target_devices.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {"serial": "left-serial"},
                    "right_wrist": {"serial": "right-serial"},
                }
            }
        ),
        encoding="utf-8",
    )
    donor_root = tmp_path / "calibration-source"
    donor_devices = donor_root / "meta/devices/realsense.json"
    donor_devices.parent.mkdir(parents=True)
    donor_devices.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {
                        "serial": "left-serial",
                        "intrinsics": {"width": 640, "height": 480, "fx": 395},
                    },
                    "right_wrist": {
                        "serial": "right-serial",
                        "intrinsics": {"width": 640, "height": 480, "fx": 393},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    donor_sha256 = hashlib.sha256(donor_devices.read_bytes()).hexdigest()
    camera_overrides = {
        camera_key: {
            "target_camera_key": camera_key,
            "expected_serial": serial,
            "donor_source_dataset_id": "calibration",
            "donor_devices_relative_path": "meta/devices/realsense.json",
            "expected_donor_devices_sha256": donor_sha256,
            "donor_camera_key": camera_key,
        }
        for camera_key, serial in (
            ("left_wrist", "left-serial"),
            ("right_wrist", "right-serial"),
        )
    }
    profile = {
        "right_wrist_intrinsics_overrides": {},
        "wrist_intrinsics_donor_sources": {
            "calibration": {"dataset_root": str(donor_root)}
        },
        "wrist_intrinsics_overrides": {"target": camera_overrides},
    }

    overlays, contract = _wrist_intrinsics_overlays_for_source(
        profile=profile,
        profile_sources={"target": target_root},
        source_id="target",
        dataset_root=target_root,
        overlay_base=tmp_path / "overlays",
        episodes=(0,),
        camera_keys=("left_wrist", "right_wrist"),
        devices_by_episode=None,
    )

    assert overlays is not None
    overlay = json.loads(overlays[0].read_text(encoding="utf-8"))
    assert overlay["cameras"]["left_wrist"]["intrinsics"]["fx"] == 395
    assert overlay["cameras"]["right_wrist"]["intrinsics"]["fx"] == 393
    assert contract is not None
    assert {
        binding["expected_donor_devices_sha256"]
        for binding in contract["target_bindings"][0]["camera_bindings"]
    } == {donor_sha256}

    camera_overrides["right_wrist"]["expected_donor_devices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="donor manifest hash differs"):
        _wrist_intrinsics_overlays_for_source(
            profile=profile,
            profile_sources={"target": target_root},
            source_id="target",
            dataset_root=target_root,
            overlay_base=tmp_path / "wrong-hash-overlays",
            episodes=(0,),
            camera_keys=("left_wrist", "right_wrist"),
            devices_by_episode=None,
        )
