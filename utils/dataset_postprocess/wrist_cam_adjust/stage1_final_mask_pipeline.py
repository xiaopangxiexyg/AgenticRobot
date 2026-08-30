from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deployments.openpi.training.prepared_skill import (
    OpenPIPreparedSkillBinding,
    load_openpi_prepared_skill_binding_manifest,
)
from deployments.openpi.training.prepared_skill_launch import _load_profile

from .depth_mask.core import DEPTH_RANGE_POSTPROCESSED_MASK_NAME
from .depth_mask.postprocess import (
    DepthMaskPostprocessConfig,
    DepthMaskPostprocessResult,
    process_mask_stream,
)
from .depth_mask.visualize import (
    find_session_mask_manifests,
    load_mask_stream,
)
from .final_mask import compose_final_masks
from .geometry_mask.dataset import adjust_dataset

PIPELINE_SCHEMA_V2 = "dexteleop_prepared_wrist_final_keep_mask_pipeline_v2"
FINAL_MASK_LAYOUT_V1 = "source_dataset_id_partitioned_v1"
FINAL_MASK_NAMESPACE_MIGRATION_SCHEMA = (
    "dexteleop_final_mask_dataset_namespace_migration_v1"
)
WRIST_CAMERA_KEYS = ("left_wrist", "right_wrist")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = _read_json(path)
        if current != dict(payload):
            raise FileExistsError(
                f"receipt already exists with different content: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_dataset_name(dataset_root: Path) -> str:
    info_path = dataset_root / "meta" / "info.json"
    info = _read_json(info_path)
    raw = info.get("dataset_name")
    if not isinstance(raw, str):
        raise TypeError(f"dataset_name must be a string: {info_path}")
    name = raw
    if (
        not name
        or name != name.strip()
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).name != name
    ):
        raise ValueError(
            f"dataset_name must be one safe path component: {name!r} in {info_path}"
        )
    return name


def _directory_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"derived namespace must not be a symlink: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"derived namespace is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "episode_directory_count": sum(
            1 for item in path.glob("episode_*") if item.is_dir()
        ),
        "manifest_count": sum(1 for _ in path.glob("episode_*/*/manifest.json")),
    }


def _migration_identity(
    *,
    profile_path: Path,
    binding_hash: str,
    source_id: str,
    dataset_root: Path,
    source_receipt_path: Path,
    canonical_dataset_name: str,
) -> dict[str, Any]:
    return {
        "schema": FINAL_MASK_NAMESPACE_MIGRATION_SCHEMA,
        "profile_sha256": _sha256(profile_path),
        "binding_hash": binding_hash,
        "source_dataset_id": source_id,
        "source_dataset_root": str(dataset_root),
        "source_receipt_sha256": _sha256(source_receipt_path),
        "generation_namespace": dataset_root.name,
        "canonical_dataset_name": canonical_dataset_name,
    }


def _validate_final_mask_namespace_migration(
    *,
    receipt_path: Path,
    profile_path: Path,
    binding_hash: str,
    source_id: str,
    dataset_root: Path,
    source_receipt_path: Path,
    canonical_dataset_name: str,
    post_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    expected_identity = _migration_identity(
        profile_path=profile_path,
        binding_hash=binding_hash,
        source_id=source_id,
        dataset_root=dataset_root,
        source_receipt_path=source_receipt_path,
        canonical_dataset_name=canonical_dataset_name,
    )
    mismatched = {
        key: (receipt.get(key), expected)
        for key, expected in expected_identity.items()
        if receipt.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            "final-mask namespace migration receipt differs from the fixed "
            f"snapshot: {receipt_path} {mismatched}"
        )
    operation = receipt.get("final_keep_namespace")
    if not isinstance(operation, Mapping):
        raise TypeError(
            f"migration receipt has no final namespace operation: {receipt_path}"
        )
    legacy_path = final_root / dataset_root.name
    canonical_path = final_root / canonical_dataset_name
    if legacy_path.exists() or legacy_path.is_symlink():
        raise ValueError(
            f"legacy final-mask namespace still exists after migration: {legacy_path}"
        )
    inventory = _directory_inventory(canonical_path)
    recorded_after = operation.get("after")
    if not isinstance(recorded_after, Mapping):
        raise TypeError(f"migration receipt has no after inventory: {receipt_path}")
    inventory_mismatch = {
        key: (recorded_after.get(key), value)
        for key, value in inventory.items()
        if recorded_after.get(key) != value
    }
    if inventory_mismatch:
        raise ValueError(
            "canonical final-mask namespace differs from its migration "
            f"receipt: {receipt_path} {inventory_mismatch}"
        )
    post = receipt.get("postprocessed_depth_keep")
    if not isinstance(post, Mapping):
        raise TypeError(f"migration receipt has no postprocessed state: {receipt_path}")
    expected_post_path = post_root / dataset_root.name
    if post.get("path") != str(expected_post_path):
        raise ValueError(
            "migration receipt binds a different postprocessed namespace: "
            f"{receipt_path}"
        )
    if post.get("state") == "retained_at_generation_namespace":
        recorded_post = post.get("inventory")
        if not isinstance(recorded_post, Mapping):
            raise TypeError(
                f"migration receipt has no postprocessed inventory: {receipt_path}"
            )
        current_post = _directory_inventory(expected_post_path)
        post_mismatch = {
            key: (recorded_post.get(key), value)
            for key, value in current_post.items()
            if recorded_post.get(key) != value
        }
        if post_mismatch:
            raise ValueError(
                "postprocessed namespace differs from its migration receipt: "
                f"{receipt_path} {post_mismatch}"
            )
    elif post.get("state") == "removed_after_validation":
        if expected_post_path.exists() or expected_post_path.is_symlink():
            raise ValueError(
                "migration receipt declares removed postprocessed masks but "
                f"the namespace exists: {expected_post_path}"
            )
    else:
        raise ValueError(
            f"unsupported postprocessed migration state: {post.get('state')!r}"
        )
    return receipt


def _migrate_final_mask_namespace(
    *,
    receipt_path: Path,
    profile_path: Path,
    binding_hash: str,
    source_id: str,
    dataset_root: Path,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    canonical_dataset_name: str,
    post_root: Path,
    final_root: Path,
    postprocessed_summary: Mapping[str, Any],
    final_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset_root.name == canonical_dataset_name:
        raise ValueError("namespace migration requires two different names")
    if receipt_path.is_file():
        return _validate_final_mask_namespace_migration(
            receipt_path=receipt_path,
            profile_path=profile_path,
            binding_hash=binding_hash,
            source_id=source_id,
            dataset_root=dataset_root,
            source_receipt_path=source_receipt_path,
            canonical_dataset_name=canonical_dataset_name,
            post_root=post_root,
            final_root=final_root,
        )

    legacy_path = final_root / dataset_root.name
    canonical_path = final_root / canonical_dataset_name
    if canonical_path.exists() or canonical_path.is_symlink():
        raise FileExistsError(
            "canonical final-mask namespace exists without a matching "
            f"migration receipt: {canonical_path}"
        )
    before = _directory_inventory(legacy_path)
    expected_manifest_count = int(final_summary["episode_count"]) * int(
        final_summary["camera_count"]
    )
    if (
        before["episode_directory_count"] != int(final_summary["episode_count"])
        or before["manifest_count"] != expected_manifest_count
    ):
        raise ValueError(
            "legacy final-mask namespace inventory differs from its verified "
            f"summary: {legacy_path}"
        )
    if legacy_path.parent != canonical_path.parent:
        raise ValueError("namespace migration must stay within one partition")
    parent_device = int(legacy_path.parent.stat().st_dev)
    if parent_device != int(before["device"]):
        raise ValueError("namespace migration would cross filesystem devices")

    retention = source_receipt.get("intermediate_retention")
    if not isinstance(retention, Mapping):
        raise TypeError("source receipt has no intermediate retention contract")
    post_path = post_root / dataset_root.name
    if retention.get("postprocessed_depth_keep") == "removed_after_validation":
        if post_path.exists() or post_path.is_symlink():
            raise ValueError(
                "source receipt declares removed postprocessed masks but the "
                f"namespace exists: {post_path}"
            )
        post_state: dict[str, Any] = {
            "state": "removed_after_validation",
            "path": str(post_path),
            "validated_summary": dict(postprocessed_summary),
        }
    else:
        post_inventory = _directory_inventory(post_path)
        expected_post_manifests = int(postprocessed_summary["episode_count"]) * int(
            postprocessed_summary["camera_count"]
        )
        if (
            post_inventory["episode_directory_count"]
            != int(postprocessed_summary["episode_count"])
            or post_inventory["manifest_count"] != expected_post_manifests
        ):
            raise ValueError(
                "postprocessed namespace inventory differs from its verified "
                f"summary: {post_path}"
            )
        post_state = {
            "state": "retained_at_generation_namespace",
            "path": str(post_path),
            "inventory": post_inventory,
            "validated_summary": dict(postprocessed_summary),
        }

    moved = False
    try:
        os.replace(legacy_path, canonical_path)
        moved = True
        after = _directory_inventory(canonical_path)
        if {
            key: before[key]
            for key in (
                "device",
                "inode",
                "episode_directory_count",
                "manifest_count",
            )
        } != {
            key: after[key]
            for key in (
                "device",
                "inode",
                "episode_directory_count",
                "manifest_count",
            )
        }:
            raise RuntimeError(
                "final-mask namespace inode or inventory changed during rename"
            )
        receipt = {
            **_migration_identity(
                profile_path=profile_path,
                binding_hash=binding_hash,
                source_id=source_id,
                dataset_root=dataset_root,
                source_receipt_path=source_receipt_path,
                canonical_dataset_name=canonical_dataset_name,
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile_path": str(profile_path),
            "source_receipt_path": str(source_receipt_path),
            "final_keep_namespace": {
                "operation": "same_filesystem_atomic_directory_rename",
                "before": before,
                "after": after,
                "validated_summary": dict(final_summary),
            },
            "postprocessed_depth_keep": post_state,
            "content_policy": {
                "mask_stream_bytes_modified": False,
                "manifest_bytes_modified": False,
                "manifest_paths_remain_generation_time_provenance": True,
                "stream_resolution": "relative_to_manifest_location",
            },
            "raw_source_modified": False,
            "source_receipt_modified": False,
            "rollback": {
                "operation": "same_filesystem_atomic_directory_rename",
                "from": str(canonical_path),
                "to": str(legacy_path),
                "required_device": int(after["device"]),
                "required_inode": int(after["inode"]),
                "preconditions": (
                    "canonical path matches this receipt and legacy path is absent"
                ),
            },
        }
        _write_receipt(receipt_path, receipt)
        return receipt
    except BaseException:
        if (
            moved
            and canonical_path.is_dir()
            and not canonical_path.is_symlink()
            and not legacy_path.exists()
            and not legacy_path.is_symlink()
            and int(canonical_path.stat().st_ino) == int(before["inode"])
            and int(canonical_path.stat().st_dev) == int(before["device"])
        ):
            os.replace(canonical_path, legacy_path)
        raise


def _episode_records(dataset_root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(record["episode_index"]): record
        for record in _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    }


def _selected_by_source(
    binding: OpenPIPreparedSkillBinding,
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for episode in binding.episodes:
        grouped.setdefault(episode.source_dataset_id, []).append(
            int(episode.source_episode_index)
        )
    return {source_id: tuple(sorted(indices)) for source_id, indices in grouped.items()}


def _selected_wrist_camera_keys(profile: Mapping[str, Any]) -> tuple[str, ...]:
    raw = profile.get("camera_selection")
    if not isinstance(raw, list):
        raise TypeError("camera_selection must be a list")
    selected = tuple(str(value).strip() for value in raw)
    cameras = tuple(key for key in WRIST_CAMERA_KEYS if key in selected)
    if not cameras:
        raise ValueError("final keep-mask generation requires a wrist camera")
    unexpected = tuple(
        key for key in selected if "wrist" in key and key not in WRIST_CAMERA_KEYS
    )
    if unexpected:
        raise ValueError(f"unsupported wrist camera keys: {unexpected}")
    return cameras


def _physical_devices_by_episode(
    dataset_root: Path,
    episode_indices: Sequence[int],
) -> tuple[dict[int, Path] | None, dict[str, Any]]:
    default_path = dataset_root / "meta" / "devices" / "realsense.json"
    audit_path = dataset_root / "meta" / "build_audit.json"
    provenance_path = dataset_root / "meta" / "provenance.jsonl"
    has_physical_binding = audit_path.is_file() and provenance_path.is_file()
    if default_path.is_file() and not has_physical_binding:
        if default_path.is_symlink():
            raise FileNotFoundError(
                f"dataset devices manifest must be a regular file: {default_path}"
            )
        return None, {
            "schema": "dexteleop.camera_devices_binding.v2",
            "mode": "dataset_default",
            "manifest_path": str(default_path.resolve()),
            "manifest_sha256": _sha256(default_path),
            "source_dataset_root": str(dataset_root.resolve()),
        }
    if not has_physical_binding:
        raise FileNotFoundError(
            "dataset has neither one devices manifest nor a physical build "
            f"audit with provenance: {dataset_root}"
        )
    audit = _read_json(audit_path)
    plan = audit.get("plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("episodes"), list):
        raise TypeError(f"physical build audit has no episode plan: {audit_path}")
    plan_rows = tuple(item for item in plan["episodes"] if isinstance(item, Mapping))
    provenance_rows = tuple(_read_jsonl(provenance_path))
    if len(plan_rows) != len(provenance_rows):
        raise ValueError(
            "physical build audit and provenance row counts differ: "
            f"audit={len(plan_rows)} provenance={len(provenance_rows)}"
        )
    provenance_by_output: dict[int, Mapping[str, Any]] = {}
    for row in provenance_rows:
        output_index = int(row["output_episode_index"])
        if output_index in provenance_by_output:
            raise ValueError(
                f"physical provenance repeats output episode index {output_index}."
            )
        provenance_by_output[output_index] = row
    selected = {int(value) for value in episode_indices}
    mapping: dict[int, Path] = {}
    source_contracts: dict[str, dict[str, Any]] = {}
    dataset_resolved = dataset_root.resolve()
    for item in plan_rows:
        output_index = int(item["output_episode_index"])
        if output_index not in selected:
            continue
        source_root = Path(str(item["source_root"])).expanduser().resolve()
        devices_candidate = (
            dataset_root
            / "meta"
            / "source_devices"
            / source_root.name
            / "realsense.json"
        )
        if devices_candidate.is_symlink():
            raise FileNotFoundError(
                "physical source devices manifest must not be a symlink for "
                f"output episode {output_index}: {devices_candidate}"
            )
        devices_path = devices_candidate.resolve()
        try:
            devices_path.relative_to(dataset_resolved)
        except ValueError as exc:
            raise ValueError(
                "physical source devices copy escapes the consolidated root: "
                f"{devices_path}"
            ) from exc
        original_path = source_root / "meta" / "devices" / "realsense.json"
        provenance = provenance_by_output.get(output_index)
        if provenance is None:
            raise ValueError(
                f"physical provenance is missing output episode {output_index}."
            )
        expected_source_episode = int(item["source_episode_index"])
        if (
            Path(str(provenance.get("source_dataset_root", ""))).expanduser().resolve()
            != source_root
            or int(provenance.get("source_episode_index", -1))
            != expected_source_episode
        ):
            raise ValueError(
                "physical provenance differs from the build plan for output "
                f"episode {output_index}."
            )
        if not devices_path.is_file() or devices_path.is_symlink():
            raise FileNotFoundError(
                "physical source devices manifest is missing for output "
                f"episode {output_index}: {devices_path}"
            )
        copied_sha256 = _sha256(devices_path)
        original_present = original_path.is_file()
        if original_present and copied_sha256 != _sha256(original_path):
            raise ValueError(
                "physical source devices copy differs from the original for "
                f"output episode {output_index}."
            )
        contract_key = str(devices_path)
        contract = {
            "source_dataset_root": str(source_root),
            "copied_manifest_path": contract_key,
            "copied_manifest_sha256": copied_sha256,
            "original_manifest_path": str(original_path),
            "original_manifest_state": (
                "copy_matches_original"
                if original_present
                else "original_absent_protected_copy_bound_by_audit_and_provenance"
            ),
            "selected_episode_count": 0,
        }
        current = source_contracts.setdefault(contract_key, contract)
        if {
            key: current[key] for key in contract if key != "selected_episode_count"
        } != {
            key: contract[key] for key in contract if key != "selected_episode_count"
        }:
            raise ValueError(
                f"physical source-device contract is inconsistent for {devices_path}."
            )
        current["selected_episode_count"] = int(current["selected_episode_count"]) + 1
        mapping[output_index] = devices_path
    missing = sorted(selected - set(mapping))
    if missing:
        raise ValueError(
            "physical build audit does not bind every selected episode to "
            f"camera intrinsics: {missing[:8]}"
        )
    return mapping, {
        "schema": "dexteleop.camera_devices_binding.v2",
        "mode": "consolidated_source_devices",
        "source_dataset_root": str(dataset_resolved),
        "build_audit_path": str(audit_path.resolve()),
        "build_audit_sha256": _sha256(audit_path),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": _sha256(provenance_path),
        "selected_episode_count": len(selected),
        "source_manifests": [source_contracts[key] for key in sorted(source_contracts)],
    }


def _camera_records(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("cameras")
    return nested if isinstance(nested, Mapping) else payload


def _camera_serial(record: Mapping[str, Any]) -> str:
    return str(record.get("serial_number") or record.get("serial") or "").strip()


def _intrinsics_overlay_for_source(
    *,
    profile: Mapping[str, Any],
    profile_sources: Mapping[str, Path],
    source_id: str,
    dataset_root: Path,
    overlay_base: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    overrides = profile.get("right_wrist_intrinsics_overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("right_wrist_intrinsics_overrides must be an object")
    raw_override = overrides.get(source_id)
    if raw_override is None:
        return None, None
    if not isinstance(raw_override, Mapping):
        raise TypeError(f"intrinsics override must be an object: {source_id}")
    target_camera_key = str(raw_override.get("target_camera_key", "")).strip()
    donor_source_id = str(raw_override.get("donor_source_dataset_id", "")).strip()
    donor_camera_key = str(raw_override.get("donor_camera_key", "")).strip()
    expected_serial = str(raw_override.get("expected_serial", "")).strip()
    if (
        target_camera_key != "right_wrist"
        or donor_source_id not in profile_sources
        or not donor_camera_key
        or not expected_serial
    ):
        raise ValueError(f"invalid right-wrist intrinsics override: {source_id}")

    target_path = dataset_root / "meta" / "devices" / "realsense.json"
    donor_path = (
        profile_sources[donor_source_id] / "meta" / "devices" / "realsense.json"
    )
    target_payload = _read_json(target_path)
    donor_payload = _read_json(donor_path)
    target_record = _camera_records(target_payload).get(target_camera_key)
    donor_record = _camera_records(donor_payload).get(donor_camera_key)
    if not isinstance(target_record, Mapping) or not isinstance(
        donor_record,
        Mapping,
    ):
        raise TypeError(f"intrinsics override camera binding is absent: {source_id}")
    target_serial = _camera_serial(target_record)
    donor_serial = _camera_serial(donor_record)
    intrinsics = donor_record.get("intrinsics")
    if (
        target_serial != expected_serial
        or donor_serial != expected_serial
        or not isinstance(intrinsics, Mapping)
    ):
        raise ValueError(
            "intrinsics donor does not match the target camera serial: "
            f"source={source_id} target={target_serial!r} "
            f"donor={donor_serial!r} expected={expected_serial!r}"
        )
    overlay_path = overlay_base / source_id / "realsense.json"
    contract = {
        "schema": "dexteleop.serial_bound_intrinsics_overlay_v1",
        "source_dataset_id": source_id,
        "target_devices_path": str(target_path),
        "target_devices_sha256": _sha256(target_path),
        "target_camera_key": target_camera_key,
        "donor_source_dataset_id": donor_source_id,
        "donor_devices_path": str(donor_path),
        "donor_devices_sha256": _sha256(donor_path),
        "donor_camera_key": donor_camera_key,
        "bound_serial": expected_serial,
        "source_modified": False,
    }
    overlay_payload = {
        "schema": "dexteleop.serial_bound_intrinsics_overlay.v1",
        target_camera_key: {
            "camera_key": target_camera_key,
            "semantic_role": target_camera_key,
            "serial": expected_serial,
            "serial_number": expected_serial,
            "intrinsics": dict(intrinsics),
        },
        "calibration_provenance": contract,
    }
    _write_receipt(overlay_path, overlay_payload)
    return overlay_path, contract


def _contained_regular_file(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to its dataset root")
    root_resolved = root.resolve()
    candidate = root_resolved / relative
    if candidate.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its dataset root: {relative}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(
            f"{label} must be a regular file inside its dataset root: {resolved}"
        )
    return resolved


def _donor_devices_contract(
    dataset_root: Path,
    donor_path: Path,
) -> dict[str, Any]:
    dataset_resolved = dataset_root.resolve()
    default_path = dataset_resolved / "meta" / "devices" / "realsense.json"
    if donor_path == default_path.resolve():
        return {
            "mode": "dataset_default",
            "source_dataset_root": str(dataset_resolved),
            "manifest_path": str(donor_path),
            "manifest_sha256": _sha256(donor_path),
        }

    audit_path = dataset_resolved / "meta" / "build_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            "a protected donor devices manifest requires build_audit.json: "
            f"{donor_path}"
        )
    audit = _read_json(audit_path)
    plan = audit.get("plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("episodes"), list):
        raise TypeError(f"physical build audit has no episode plan: {audit_path}")
    output_indices = tuple(
        int(item["output_episode_index"])
        for item in plan["episodes"]
        if isinstance(item, Mapping)
    )
    _mapping, binding_contract = _physical_devices_by_episode(
        dataset_resolved,
        output_indices,
    )
    matches = [
        item
        for item in binding_contract.get("source_manifests", [])
        if item.get("copied_manifest_path") == str(donor_path)
    ]
    if len(matches) != 1:
        raise ValueError(
            "protected donor devices manifest is not uniquely bound by the "
            f"build audit and provenance: {donor_path}"
        )
    return {
        "mode": "consolidated_source_devices",
        "source_dataset_root": str(dataset_resolved),
        "build_audit_path": binding_contract["build_audit_path"],
        "build_audit_sha256": binding_contract["build_audit_sha256"],
        "provenance_path": binding_contract["provenance_path"],
        "provenance_sha256": binding_contract["provenance_sha256"],
        "source_manifest": matches[0],
    }


def _wrist_intrinsics_overlays_for_source(
    *,
    profile: Mapping[str, Any],
    profile_sources: Mapping[str, Path],
    source_id: str,
    dataset_root: Path,
    overlay_base: Path,
    episodes: Sequence[int],
    camera_keys: Sequence[str],
    devices_by_episode: Mapping[int, Path] | None,
) -> tuple[dict[int, Path] | None, dict[str, Any] | None]:
    """Create immutable, serial-bound overlays for every selected wrist.

    The v1 right-wrist override above is retained for existing profiles.  This
    multi-camera form additionally supports physical collections whose camera
    manifests are selected per episode.  It never modifies source metadata:
    each distinct target manifest receives a new overlay under the derived
    workspace, and the returned map binds every selected episode explicitly.
    """

    raw_overrides = profile.get("wrist_intrinsics_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("wrist_intrinsics_overrides must be an object")
    raw_source_overrides = raw_overrides.get(source_id)
    if raw_source_overrides is None:
        return None, None
    if not isinstance(raw_source_overrides, Mapping):
        raise TypeError(f"wrist intrinsics overrides must be an object: {source_id}")

    selected_cameras = tuple(str(value) for value in camera_keys)
    if set(raw_source_overrides) != set(selected_cameras):
        raise ValueError(
            "wrist intrinsics overrides must cover exactly the selected wrist "
            f"cameras for {source_id}: selected={selected_cameras} "
            f"configured={tuple(raw_source_overrides)}"
        )
    legacy = profile.get("right_wrist_intrinsics_overrides", {})
    if isinstance(legacy, Mapping) and source_id in legacy:
        raise ValueError(
            "a source cannot use both right_wrist_intrinsics_overrides and "
            f"wrist_intrinsics_overrides: {source_id}"
        )

    selected_episodes = tuple(int(value) for value in episodes)
    if devices_by_episode is None:
        default_devices = (
            dataset_root / "meta" / "devices" / "realsense.json"
        ).resolve()
        target_devices_by_episode = {
            episode_index: default_devices for episode_index in selected_episodes
        }
    else:
        target_devices_by_episode = {
            int(episode_index): Path(path).expanduser().resolve()
            for episode_index, path in devices_by_episode.items()
        }
        missing = sorted(set(selected_episodes) - set(target_devices_by_episode))
        extra = sorted(set(target_devices_by_episode) - set(selected_episodes))
        if missing or extra:
            raise ValueError(
                "per-episode camera devices do not exactly match selected "
                f"episodes for {source_id}: missing={missing[:8]} extra={extra[:8]}"
            )

    dataset_resolved = dataset_root.resolve()
    episodes_by_target: dict[Path, list[int]] = {}
    for episode_index in selected_episodes:
        target_path = target_devices_by_episode[episode_index]
        try:
            target_path.relative_to(dataset_resolved)
        except ValueError as exc:
            raise ValueError(
                "target camera devices manifest escapes its dataset root: "
                f"{target_path}"
            ) from exc
        if not target_path.is_file() or target_path.is_symlink():
            raise FileNotFoundError(
                f"target camera devices manifest must be a regular file: {target_path}"
            )
        episodes_by_target.setdefault(target_path, []).append(episode_index)

    donor_cache: dict[
        tuple[str, str],
        tuple[Path, Mapping[str, Any], Mapping[str, Any]],
    ] = {}
    episode_overlays: dict[int, Path] = {}
    target_contracts: list[dict[str, Any]] = []
    for target_path in sorted(episodes_by_target, key=str):
        target_payload = _read_json(target_path)
        target_records = _camera_records(target_payload)
        overlay_cameras: dict[str, dict[str, Any]] = {}
        camera_contracts: list[dict[str, Any]] = []
        for camera_key in selected_cameras:
            raw_override = raw_source_overrides[camera_key]
            if not isinstance(raw_override, Mapping):
                raise TypeError(
                    "wrist intrinsics camera override must be an object: "
                    f"{source_id}/{camera_key}"
                )
            target_camera_key = str(raw_override.get("target_camera_key", "")).strip()
            donor_source_id = str(
                raw_override.get("donor_source_dataset_id", "")
            ).strip()
            donor_relative_path = str(
                raw_override.get("donor_devices_relative_path", "")
            ).strip()
            donor_camera_key = str(raw_override.get("donor_camera_key", "")).strip()
            expected_serial = str(raw_override.get("expected_serial", "")).strip()
            if (
                target_camera_key != camera_key
                or donor_source_id not in profile_sources
                or not donor_relative_path
                or not donor_camera_key
                or not expected_serial
            ):
                raise ValueError(
                    f"invalid wrist intrinsics override: {source_id}/{camera_key}"
                )

            cache_key = (donor_source_id, donor_relative_path)
            donor_cached = donor_cache.get(cache_key)
            if donor_cached is None:
                donor_root = profile_sources[donor_source_id]
                donor_path = _contained_regular_file(
                    donor_root,
                    donor_relative_path,
                    label="donor_devices_relative_path",
                )
                donor_cached = (
                    donor_path,
                    _camera_records(_read_json(donor_path)),
                    _donor_devices_contract(donor_root, donor_path),
                )
                donor_cache[cache_key] = donor_cached
            donor_path, donor_records, donor_contract = donor_cached

            target_record = target_records.get(target_camera_key)
            donor_record = donor_records.get(donor_camera_key)
            if not isinstance(target_record, Mapping) or not isinstance(
                donor_record,
                Mapping,
            ):
                raise TypeError(
                    "intrinsics overlay camera binding is absent: "
                    f"{source_id}/{camera_key}"
                )
            target_serial = _camera_serial(target_record)
            donor_serial = _camera_serial(donor_record)
            intrinsics = donor_record.get("intrinsics")
            if (
                target_serial != expected_serial
                or donor_serial != expected_serial
                or not isinstance(intrinsics, Mapping)
            ):
                raise ValueError(
                    "intrinsics donor does not match the target camera serial: "
                    f"source={source_id} camera={camera_key} "
                    f"target={target_serial!r} donor={donor_serial!r} "
                    f"expected={expected_serial!r}"
                )

            overlay_record = dict(target_record)
            overlay_record.update(
                {
                    "camera_key": target_camera_key,
                    "semantic_role": target_camera_key,
                    "serial": expected_serial,
                    "serial_number": expected_serial,
                    "intrinsics": dict(intrinsics),
                }
            )
            overlay_cameras[target_camera_key] = overlay_record
            camera_contracts.append(
                {
                    "target_camera_key": target_camera_key,
                    "expected_serial": expected_serial,
                    "donor_source_dataset_id": donor_source_id,
                    "donor_devices_path": str(donor_path),
                    "donor_devices_sha256": _sha256(donor_path),
                    "donor_camera_key": donor_camera_key,
                    "donor_devices_contract": donor_contract,
                }
            )

        target_relative = target_path.relative_to(dataset_resolved).as_posix()
        target_key = hashlib.sha256(target_relative.encode("utf-8")).hexdigest()[:16]
        overlay_path = overlay_base / source_id / target_key / "realsense.json"
        overlay_payload = {
            "schema": "dexteleop.serial_bound_wrist_intrinsics_overlay.v2",
            "source_dataset_id": source_id,
            "target_devices_path": str(target_path),
            "target_devices_sha256": _sha256(target_path),
            "source_modified": False,
            "cameras": overlay_cameras,
            "calibration_provenance": camera_contracts,
        }
        _write_receipt(overlay_path, overlay_payload)
        bound_episodes = sorted(episodes_by_target[target_path])
        for episode_index in bound_episodes:
            episode_overlays[episode_index] = overlay_path
        target_contracts.append(
            {
                "target_devices_path": str(target_path),
                "target_devices_sha256": _sha256(target_path),
                "target_devices_relative_path": target_relative,
                "episode_indices": bound_episodes,
                "overlay_path": str(overlay_path),
                "overlay_sha256": _sha256(overlay_path),
                "camera_bindings": camera_contracts,
            }
        )

    return episode_overlays, {
        "schema": "dexteleop.serial_bound_wrist_intrinsics_overlays.v2",
        "source_dataset_id": source_id,
        "source_dataset_root": str(dataset_resolved),
        "selected_episode_count": len(selected_episodes),
        "cameras": list(selected_cameras),
        "target_bindings": target_contracts,
        "source_modified": False,
    }


def _expected_frames(
    records: Mapping[int, Mapping[str, Any]],
    episodes: Sequence[int],
    *,
    max_frames: int,
) -> dict[int, int]:
    expected: dict[int, int] = {}
    for episode_index in episodes:
        record = records.get(int(episode_index))
        if record is None:
            raise ValueError(f"selected episode is absent: {episode_index}")
        frame_count = int(record.get("frame_count", 0))
        if frame_count <= 0:
            raise ValueError(
                f"selected episode has no positive frame_count: {episode_index}"
            )
        expected[int(episode_index)] = (
            min(frame_count, int(max_frames)) if int(max_frames) > 0 else frame_count
        )
    return expected


def _verify_mask_tree(
    root: Path,
    dataset_name: str,
    expected_frames: Mapping[int, int],
    *,
    camera_keys: Sequence[str],
    stream_name: str,
    expected_mask_name: str,
) -> dict[str, Any]:
    cameras = tuple(str(value) for value in camera_keys)
    if not cameras:
        raise ValueError("mask-tree verification requires at least one camera")
    logical_frames = sum(int(value) for value in expected_frames.values())
    camera_frames = 0
    total_bytes = 0
    by_camera: dict[str, dict[str, int]] = {}
    for camera_key in cameras:
        per_camera_frames = 0
        per_camera_bytes = 0
        for episode_index, frame_count in sorted(expected_frames.items()):
            episode_dir = (
                root / dataset_name / f"episode_{episode_index:06d}" / camera_key
            )
            manifest_path = episode_dir / "manifest.json"
            manifest = _read_json(manifest_path)
            if manifest.get("mask_name") != expected_mask_name:
                raise ValueError(f"unexpected mask semantics: {manifest_path}")
            stream = manifest.get("stream")
            if not isinstance(stream, Mapping):
                raise TypeError(f"mask manifest has no stream: {manifest_path}")
            actual_count = int(stream.get("frame_count", 0))
            if actual_count != frame_count:
                raise ValueError(
                    "mask frame count differs: "
                    f"episode={episode_index} camera={camera_key} "
                    f"actual={actual_count} expected={frame_count}"
                )
            packed = int(stream.get("packed_bytes_per_frame", 0))
            if packed <= 0:
                raise ValueError(
                    f"mask manifest has invalid packed frame size: {manifest_path}"
                )
            stream_path = episode_dir / stream_name
            actual_bytes = int(stream_path.stat().st_size)
            expected_bytes = actual_count * packed
            if actual_bytes != expected_bytes:
                raise ValueError(
                    "mask stream size differs: "
                    f"episode={episode_index} camera={camera_key} "
                    f"actual={actual_bytes} expected={expected_bytes}"
                )
            per_camera_frames += actual_count
            per_camera_bytes += actual_bytes
        by_camera[camera_key] = {
            "frame_count": per_camera_frames,
            "stream_bytes": per_camera_bytes,
        }
        camera_frames += per_camera_frames
        total_bytes += per_camera_bytes
    return {
        "episode_count": len(expected_frames),
        "camera_count": len(cameras),
        "cameras": list(cameras),
        "frame_count": logical_frames,
        "camera_frame_count": camera_frames,
        "stream_bytes": total_bytes,
        "by_camera": by_camera,
    }


def _postprocess_source(
    dataset_root: Path,
    output_root: Path,
    episodes: Sequence[int],
    *,
    dataset_name: str,
    camera_keys: Sequence[str],
    jobs: int,
    max_frames: int,
) -> tuple[list[DepthMaskPostprocessResult], list[dict[str, Any]]]:
    config = DepthMaskPostprocessConfig()

    def run(
        episode_index: int,
        camera_key: str,
    ) -> tuple[int, str, DepthMaskPostprocessResult]:
        manifests = find_session_mask_manifests(
            dataset_root,
            episode=episode_index,
            camera_keys=(camera_key,),
        )
        if len(manifests) != 1:
            raise ValueError(
                f"episode {episode_index} did not resolve one {camera_key} mask"
            )
        stream = load_mask_stream(manifests[0], root_hint=dataset_root)
        output_dir = (
            output_root / dataset_name / f"episode_{episode_index:06d}" / camera_key
        )
        return (
            episode_index,
            camera_key,
            process_mask_stream(
                stream,
                output_dir,
                config=config,
                max_frames=max_frames,
                overwrite=output_dir.exists(),
            ),
        )

    processed: list[tuple[int, str, DepthMaskPostprocessResult]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {
            executor.submit(run, episode, camera_key): (episode, camera_key)
            for episode in episodes
            for camera_key in camera_keys
        }
        for future in as_completed(futures):
            processed.append(future.result())
    processed.sort(key=lambda item: (item[0], item[1]))
    results = [item[2] for item in processed]
    trailing = [
        {
            "episode_index": episode_index,
            "camera_key": camera_key,
            "manifest_path": str(result.manifest_path),
            "source_declared_bytes": result.source_declared_bytes,
            "source_actual_bytes": result.source_actual_bytes,
            "source_trailing_bytes": result.source_trailing_bytes,
        }
        for episode_index, camera_key, result in processed
        if result.source_trailing_bytes
    ]
    return results, trailing


def _source_receipt(
    *,
    profile_path: Path,
    binding: OpenPIPreparedSkillBinding,
    source_id: str,
    dataset_root: Path,
    dataset_name: str,
    episodes: Sequence[int],
    camera_keys: Sequence[str],
    expected_frames: Mapping[int, int],
    table_root: Path,
    post_root: Path,
    final_root: Path,
    postprocessed_summary: Mapping[str, Any],
    final_summary: Mapping[str, Any],
    intermediates_removed: bool,
    trailing_sources: Sequence[Mapping[str, Any]],
    calibration_contract: Mapping[str, Any] | None,
    camera_devices_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": PIPELINE_SCHEMA_V2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "binding_hash": binding.binding_hash,
        "source_dataset_id": source_id,
        "source_dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "eligible_episode_count": len(episodes),
        "eligible_frame_count": sum(expected_frames.values()),
        "episode_indices": list(episodes),
        "cameras": list(camera_keys),
        "mask_policy": (
            "raw depth keep -> default spatial/temporal postprocess -> "
            "OR table geometry keep -> final keep mask"
        ),
        "final_mask_root_layout": FINAL_MASK_LAYOUT_V1,
        "raw_source_modified": False,
        "raw_trailing_streams": [dict(item) for item in trailing_sources],
        "calibration_contract": (
            None if calibration_contract is None else dict(calibration_contract)
        ),
        "camera_devices_contract": dict(camera_devices_contract),
        "derived_roots": {
            "table_geometry_keep": str(table_root),
            "postprocessed_depth_keep": str(post_root),
            "final_keep": str(final_root),
        },
        "validated_summaries": {
            "postprocessed_depth_keep": dict(postprocessed_summary),
            "final_keep": dict(final_summary),
        },
        "intermediate_retention": {
            "table_geometry_keep": (
                "removed_after_validation" if intermediates_removed else "retained"
            ),
            "postprocessed_depth_keep": (
                "removed_after_validation" if intermediates_removed else "retained"
            ),
        },
        "rollback": {
            "policy": "remove only the three source-partitioned derived roots",
            "raw_source_restoration_required": False,
        },
    }


def _validated_removed_intermediate_summary(
    receipt: Mapping[str, Any],
    expected_frames: Mapping[int, int],
    *,
    camera_keys: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    retention = receipt.get("intermediate_retention")
    if not isinstance(retention, Mapping):
        return None
    expected_state = "removed_after_validation"
    if (
        retention.get("table_geometry_keep") != expected_state
        or retention.get("postprocessed_depth_keep") != expected_state
    ):
        return None
    summaries = receipt.get("validated_summaries")
    if not isinstance(summaries, Mapping):
        raise ValueError("removed intermediates have no validated summaries")
    postprocessed = summaries.get("postprocessed_depth_keep")
    if not isinstance(postprocessed, Mapping):
        raise ValueError("removed postprocessed masks have no validated summary")
    cameras = tuple(
        str(value)
        for value in (
            camera_keys
            if camera_keys is not None
            else receipt.get("cameras", (receipt.get("camera", "right_wrist"),))
        )
    )
    expected_summary: dict[str, Any] = {
        "episode_count": len(expected_frames),
        "frame_count": sum(expected_frames.values()),
    }
    if camera_keys is not None or receipt.get("schema") == PIPELINE_SCHEMA_V2:
        expected_summary.update(
            {
                "camera_count": len(cameras),
                "cameras": list(cameras),
                "camera_frame_count": sum(expected_frames.values()) * len(cameras),
            }
        )
    mismatched = {
        key: (postprocessed.get(key), expected_value)
        for key, expected_value in expected_summary.items()
        if postprocessed.get(key) != expected_value
    }
    if mismatched:
        raise ValueError(
            "removed postprocessed-mask summary differs from the fixed "
            f"snapshot: {mismatched}"
        )
    stream_bytes = postprocessed.get("stream_bytes")
    if not isinstance(stream_bytes, int) or stream_bytes <= 0:
        raise ValueError("removed postprocessed-mask summary has invalid bytes")
    return dict(postprocessed)


def _remove_validated_intermediate_roots(
    *,
    intermediate_derived_root: Path,
    final_derived_root: Path,
    table_root: Path,
    post_root: Path,
) -> None:
    intermediate = intermediate_derived_root.resolve()
    final = final_derived_root.resolve()
    if intermediate == final:
        raise ValueError(
            "validated intermediates can be removed only from a separately "
            "declared intermediate workspace"
        )
    for label, candidate in (
        ("table geometry", table_root),
        ("postprocessed depth", post_root),
    ):
        resolved = candidate.resolve()
        try:
            resolved.relative_to(intermediate)
        except ValueError as exc:
            raise ValueError(
                f"{label} root is outside the intermediate workspace: {resolved}"
            ) from exc
        if resolved == intermediate or not resolved.is_dir():
            raise ValueError(f"invalid {label} root for removal: {resolved}")
    shutil.rmtree(table_root.resolve())
    shutil.rmtree(post_root.resolve())


def _summary_receipt_name(
    run_sources: Sequence[str],
    all_sources: Sequence[str],
) -> str:
    if tuple(run_sources) == tuple(all_sources):
        return "summary.json"
    digest = hashlib.sha256("\n".join(run_sources).encode("utf-8")).hexdigest()[:16]
    return f"summary-{digest}.json"


def _resolve_pipeline_derived_roots(
    *,
    configured_final_root: Path,
    workspace_root: Path | None,
    intermediate_workspace_root: Path | None,
) -> tuple[Path, Path]:
    final_derived_root = (
        configured_final_root.parent
        if workspace_root is None
        else workspace_root.expanduser().resolve() / "derived"
    )
    final_base = final_derived_root / "final_keep_masks"
    if workspace_root is None and final_base != configured_final_root:
        raise ValueError("profile final-mask path is not under one derived root")
    intermediate_derived_root = (
        final_derived_root
        if intermediate_workspace_root is None
        else intermediate_workspace_root.expanduser().resolve() / "derived"
    )
    return final_derived_root, intermediate_derived_root


def run_pipeline(
    *,
    profile_path: Path,
    binding_root: Path,
    urdf_path: Path,
    workspace_root: Path | None,
    intermediate_workspace_root: Path | None = None,
    source_ids: Sequence[str],
    jobs: int,
    max_frames: int,
    verify_only: bool,
    remove_validated_intermediates: bool = False,
    migrate_dataset_name_layout: bool = False,
) -> dict[str, Any]:
    if verify_only and migrate_dataset_name_layout:
        raise ValueError("dataset-name layout migration cannot run with verify-only")
    profile = _load_profile(profile_path)
    if profile.get("openpi_final_mask_root_layout") != FINAL_MASK_LAYOUT_V1:
        raise ValueError("profile does not declare source-partitioned final masks")
    binding = load_openpi_prepared_skill_binding_manifest(binding_root)
    camera_keys = _selected_wrist_camera_keys(profile)
    selected = _selected_by_source(binding)
    profile_sources = {
        str(item["source_dataset_id"]): Path(str(item["dataset_root"])).resolve()
        for item in profile["sources"]
    }
    if set(selected) != set(profile_sources):
        raise ValueError("binding sources differ from the prepared profile")
    wrist_overrides = profile.get("wrist_intrinsics_overrides", {})
    if not isinstance(wrist_overrides, Mapping):
        raise TypeError("wrist_intrinsics_overrides must be an object")
    unknown_override_sources = sorted(set(wrist_overrides) - set(profile_sources))
    if unknown_override_sources:
        raise ValueError(
            "wrist intrinsics overrides reference unknown source IDs: "
            f"{unknown_override_sources}"
        )
    legacy_overrides = profile.get("right_wrist_intrinsics_overrides", {})
    if not isinstance(legacy_overrides, Mapping):
        raise TypeError("right_wrist_intrinsics_overrides must be an object")
    overlapping_override_sources = sorted(set(wrist_overrides) & set(legacy_overrides))
    if overlapping_override_sources:
        raise ValueError(
            "sources cannot use both legacy and multi-wrist intrinsics "
            f"overrides: {overlapping_override_sources}"
        )
    requested = set(source_ids)
    if requested:
        unknown = sorted(requested - set(profile_sources))
        if unknown:
            raise ValueError(f"unknown source IDs: {unknown}")
    run_sources = [
        source_id
        for source_id in profile_sources
        if not requested or source_id in requested
    ]

    configured_final_root = Path(str(profile["openpi_final_mask_root"])).resolve()
    derived_root, intermediate_derived_root = _resolve_pipeline_derived_roots(
        configured_final_root=configured_final_root,
        workspace_root=workspace_root,
        intermediate_workspace_root=intermediate_workspace_root,
    )
    final_base = derived_root / "final_keep_masks"
    table_base = intermediate_derived_root / "table_geometry_keep_masks"
    post_base = intermediate_derived_root / "postprocessed_depth_keep_masks"
    profile_id = str(profile["profile_id"])
    receipt_base = (
        derived_root.parent / "receipts" / "prepared_wrist_final_masks" / profile_id
    )
    overlay_base = intermediate_derived_root / "calibration_overlays"

    summaries: list[dict[str, Any]] = []
    for source_id in run_sources:
        dataset_root = profile_sources[source_id]
        dataset_name = _canonical_dataset_name(dataset_root)
        episodes = selected[source_id]
        records = _episode_records(dataset_root)
        expected = _expected_frames(records, episodes, max_frames=max_frames)
        table_root = table_base / source_id
        post_root = post_base / source_id
        final_root = final_base / source_id
        receipt_path = receipt_base / f"{source_id}.json"
        migration_receipt_path = (
            receipt_base / "layout_migrations" / f"{source_id}.json"
        )
        existing_receipt = _read_json(receipt_path) if receipt_path.is_file() else None
        if existing_receipt is not None:
            expected_receipt_identity = {
                "schema": PIPELINE_SCHEMA_V2,
                "binding_hash": binding.binding_hash,
                "source_dataset_id": source_id,
                "source_dataset_root": str(dataset_root),
                "eligible_episode_count": len(episodes),
                "eligible_frame_count": sum(expected.values()),
                "profile_sha256": _sha256(profile_path),
                "cameras": list(camera_keys),
            }
            if "dataset_name" in existing_receipt:
                expected_receipt_identity["dataset_name"] = dataset_name
            mismatched = {
                key: (existing_receipt.get(key), expected_value)
                for key, expected_value in expected_receipt_identity.items()
                if existing_receipt.get(key) != expected_value
            }
            if mismatched:
                raise ValueError(
                    "existing source receipt differs from the requested fixed "
                    f"snapshot: {receipt_path} {mismatched}"
                )

        calibration_contract = (
            None
            if existing_receipt is None
            else existing_receipt.get("calibration_contract")
        )
        camera_devices_contract = (
            None
            if existing_receipt is None
            else existing_receipt.get("camera_devices_contract")
        )
        if not verify_only and existing_receipt is None:
            legacy_override_devices, legacy_calibration_contract = (
                _intrinsics_overlay_for_source(
                    profile=profile,
                    profile_sources=profile_sources,
                    source_id=source_id,
                    dataset_root=dataset_root,
                    overlay_base=overlay_base,
                )
            )
            devices_by_episode, camera_devices_contract = _physical_devices_by_episode(
                dataset_root,
                episodes,
            )
            wrist_overlay_devices, wrist_calibration_contract = (
                _wrist_intrinsics_overlays_for_source(
                    profile=profile,
                    profile_sources=profile_sources,
                    source_id=source_id,
                    dataset_root=dataset_root,
                    overlay_base=overlay_base,
                    episodes=episodes,
                    camera_keys=camera_keys,
                    devices_by_episode=devices_by_episode,
                )
            )
            if wrist_overlay_devices is not None:
                if legacy_override_devices is not None:
                    raise ValueError(
                        "legacy and multi-wrist intrinsics overlays cannot both "
                        f"bind source {source_id}"
                    )
                effective_devices = None
                effective_devices_by_episode = wrist_overlay_devices
                calibration_contract = wrist_calibration_contract
            else:
                if (
                    legacy_override_devices is not None
                    and devices_by_episode is not None
                ):
                    raise ValueError(
                        "legacy right-wrist intrinsics overlay cannot replace a "
                        f"per-episode devices binding: {source_id}"
                    )
                effective_devices = legacy_override_devices
                effective_devices_by_episode = devices_by_episode
                calibration_contract = legacy_calibration_contract
            adjust_dataset(
                dataset_root,
                table_root,
                urdf_path=urdf_path,
                cameras=camera_keys,
                episodes=episodes,
                devices_json=effective_devices,
                devices_json_by_episode=effective_devices_by_episode,
                output_kind="keep-mask",
                table_mask_mode="sampled-raycast",
                table_raycast_stride=2,
                copy_mode="skip",
                jobs=jobs,
                overwrite=table_root.exists(),
                max_frames=max_frames,
                show_progress=True,
            )
            _post_results, trailing = _postprocess_source(
                dataset_root,
                post_root,
                episodes,
                dataset_name=dataset_name,
                camera_keys=camera_keys,
                jobs=jobs,
                max_frames=max_frames,
            )
            compose_final_masks(
                session_root=dataset_root,
                table_mask_root=table_root,
                depth_mask_root=post_root,
                output_root=final_root,
                episodes=episodes,
                cameras=camera_keys,
                max_frames=max_frames,
                jobs=jobs,
                overwrite=(final_root / dataset_name).exists(),
                dataset_namespace=dataset_name,
            )
        else:
            trailing = list((existing_receipt or {}).get("raw_trailing_streams", []))

        legacy_receipt_layout = (
            existing_receipt is not None
            and "dataset_name" not in existing_receipt
            and dataset_root.name != dataset_name
        )
        post_dataset_name = dataset_name
        final_dataset_name = dataset_name
        needs_namespace_migration = False
        if legacy_receipt_layout:
            post_dataset_name = dataset_root.name
            legacy_final_path = final_root / dataset_root.name
            canonical_final_path = final_root / dataset_name
            if migration_receipt_path.is_file():
                _validate_final_mask_namespace_migration(
                    receipt_path=migration_receipt_path,
                    profile_path=profile_path,
                    binding_hash=binding.binding_hash,
                    source_id=source_id,
                    dataset_root=dataset_root,
                    source_receipt_path=receipt_path,
                    canonical_dataset_name=dataset_name,
                    post_root=post_root,
                    final_root=final_root,
                )
            elif canonical_final_path.exists() or canonical_final_path.is_symlink():
                raise FileExistsError(
                    "canonical final-mask namespace exists without an immutable "
                    f"migration receipt: {canonical_final_path}"
                )
            elif legacy_final_path.is_dir() and not legacy_final_path.is_symlink():
                if not migrate_dataset_name_layout:
                    raise ValueError(
                        "legacy final-mask namespace requires explicit "
                        "--migrate-dataset-name-layout before verification: "
                        f"{legacy_final_path} -> {canonical_final_path}"
                    )
                final_dataset_name = dataset_root.name
                needs_namespace_migration = True
            else:
                raise FileNotFoundError(
                    "neither the receipted legacy nor canonical final-mask "
                    f"namespace exists for {source_id}"
                )

        removed_post_summary = (
            None
            if existing_receipt is None
            else _validated_removed_intermediate_summary(
                existing_receipt,
                expected,
                camera_keys=camera_keys,
            )
        )
        post_summary = (
            removed_post_summary
            if removed_post_summary is not None
            else _verify_mask_tree(
                post_root,
                post_dataset_name,
                expected,
                camera_keys=camera_keys,
                stream_name="mask_postprocessed.bitpack.bin",
                expected_mask_name=DEPTH_RANGE_POSTPROCESSED_MASK_NAME,
            )
        )
        final_summary = _verify_mask_tree(
            final_root,
            final_dataset_name,
            expected,
            camera_keys=camera_keys,
            stream_name="final_keep.bitpack.bin",
            expected_mask_name="postprocessed_depth_or_table_keep_mask",
        )
        if needs_namespace_migration:
            if existing_receipt is None:
                raise AssertionError("legacy migration requires a source receipt")
            _migrate_final_mask_namespace(
                receipt_path=migration_receipt_path,
                profile_path=profile_path,
                binding_hash=binding.binding_hash,
                source_id=source_id,
                dataset_root=dataset_root,
                source_receipt_path=receipt_path,
                source_receipt=existing_receipt,
                canonical_dataset_name=dataset_name,
                post_root=post_root,
                final_root=final_root,
                postprocessed_summary=post_summary,
                final_summary=final_summary,
            )
            final_dataset_name = dataset_name
        receipt = _source_receipt(
            profile_path=profile_path,
            binding=binding,
            source_id=source_id,
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            episodes=episodes,
            camera_keys=camera_keys,
            expected_frames=expected,
            table_root=table_root,
            post_root=post_root,
            final_root=final_root,
            postprocessed_summary=post_summary,
            final_summary=final_summary,
            intermediates_removed=remove_validated_intermediates,
            trailing_sources=trailing,
            calibration_contract=(
                calibration_contract
                if isinstance(calibration_contract, Mapping)
                else None
            ),
            camera_devices_contract=(
                camera_devices_contract
                if isinstance(camera_devices_contract, Mapping)
                else {}
            ),
        )
        if not verify_only and existing_receipt is None:
            if remove_validated_intermediates:
                _remove_validated_intermediate_roots(
                    intermediate_derived_root=intermediate_derived_root,
                    final_derived_root=derived_root,
                    table_root=table_root,
                    post_root=post_root,
                )
            _write_receipt(receipt_path, receipt)
        summaries.append(
            {
                "source_dataset_id": source_id,
                "dataset_root": str(dataset_root),
                "postprocessed": post_summary,
                "final": final_summary,
                "raw_trailing_stream_count": len(trailing),
                "receipt_path": str(receipt_path),
            }
        )

    summary = {
        "schema": PIPELINE_SCHEMA_V2,
        "profile_path": str(profile_path),
        "binding_root": str(binding_root),
        "binding_hash": binding.binding_hash,
        "workspace_derived_root": str(derived_root),
        "intermediate_workspace_derived_root": str(intermediate_derived_root),
        "source_count": len(summaries),
        "cameras": list(camera_keys),
        "eligible_episode_count": sum(
            int(item["final"]["episode_count"]) for item in summaries
        ),
        "eligible_frame_count": sum(
            int(item["final"]["frame_count"]) for item in summaries
        ),
        "eligible_camera_frame_count": sum(
            int(item["final"]["camera_frame_count"]) for item in summaries
        ),
        "sources": summaries,
    }
    if not verify_only:
        summary_name = _summary_receipt_name(
            run_sources,
            tuple(profile_sources),
        )
        _write_receipt(receipt_base / summary_name, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and verify source-partitioned final keep masks for the "
            "wrist cameras selected by a prepared OpenPI profile, without "
            "modifying source datasets."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--binding-root", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--intermediate-workspace-root", type=Path, default=None)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--remove-validated-intermediates", action="store_true")
    parser.add_argument(
        "--migrate-dataset-name-layout",
        action="store_true",
        help=(
            "Atomically rename receipted legacy final-mask directories to the "
            "canonical meta/info.json dataset_name and publish a rollback "
            "receipt; raw data and mask bytes are not modified."
        ),
    )
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.verify_only and args.migrate_dataset_name_layout:
        parser.error(
            "--migrate-dataset-name-layout cannot be combined with --verify-only"
        )
    summary = run_pipeline(
        profile_path=args.profile.expanduser().resolve(),
        binding_root=args.binding_root.expanduser().resolve(),
        urdf_path=args.urdf.expanduser().resolve(),
        workspace_root=args.workspace_root,
        intermediate_workspace_root=args.intermediate_workspace_root,
        source_ids=tuple(str(value) for value in args.source_id),
        jobs=args.jobs,
        max_frames=args.max_frames,
        verify_only=args.verify_only,
        remove_validated_intermediates=args.remove_validated_intermediates,
        migrate_dataset_name_layout=args.migrate_dataset_name_layout,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
