"""Read-only acceptance validation for native DexTeleop v2.3 datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dataset_contracts.causal_snapshot import (
    CAUSAL_SNAPSHOT_DETAIL_METADATA_KEY,
    CAUSAL_SNAPSHOT_HASH_METADATA_KEY,
    CAUSAL_SNAPSHOT_METADATA_KEY,
    DEFAULT_CAUSAL_SNAPSHOT_CONTRACT,
)
from dataset_contracts.pose_frames import (
    POSE_FRAME_CONTRACT_METADATA_KEY,
    POSE_STORAGE_FRAME_DESK,
    PoseFrameContract,
)
from dataset_contracts.v22 import policy_io_layout_from_metadata
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v23_validation import validate_v23_row_causal_snapshots


_ARTIFACT_MAP_KEYS = (
    "video_paths",
    "tactile_paths",
    "wrist_depth_mask_paths",
    "wrist_depth_raw_paths",
)


@dataclass
class V23DatasetValidationReport:
    dataset_root: str
    episode_count: int = 0
    frame_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dataset_root": self.dataset_root,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _pose7(value: Any) -> np.ndarray | None:
    try:
        pose = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return None
    norm = float(np.linalg.norm(pose[3:]))
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0.0):
        return None
    return pose


def _packed_pose(value: Any, side: str) -> np.ndarray | None:
    try:
        packed = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    start = 0 if side == "left" else 7
    return _pose7(packed[start : start + 7]) if packed.size >= start + 7 else None


def _same_pose(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return False
    return bool(
        np.allclose(left[:3], right[:3], atol=1e-6, rtol=0.0)
        and abs(float(np.dot(left[3:], right[3:]))) >= 1.0 - 1e-6
    )


def _arm_source_clocks(contract: PoseFrameContract) -> tuple[str, ...]:
    if contract.source.source_family == "vive_umi":
        return (
            "local_wall_time",
            "vive_sender_monotonic_calibrated_to_local_wall",
            "vive_receiver_wall_fallback",
        )
    return ("local_wall_time",)


def _validate_metadata_binding(
    report: V23DatasetValidationReport,
    *,
    label: str,
    metadata: Mapping[str, Any],
    contract: PoseFrameContract,
    layout_hash: str,
) -> None:
    expected = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        "pose_frame_contract_hash": contract.contract_hash,
        "policy_io_layout_hash": layout_hash,
        CAUSAL_SNAPSHOT_METADATA_KEY: DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.schema,
        CAUSAL_SNAPSHOT_HASH_METADATA_KEY: DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.contract_hash,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            report.errors.append(f"{label}:{key} mismatch")
    if metadata.get(POSE_FRAME_CONTRACT_METADATA_KEY) != contract.to_dict():
        report.errors.append(f"{label}:pose_frame_contract differs from dataset contract")
    if metadata.get(CAUSAL_SNAPSHOT_DETAIL_METADATA_KEY) != DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.to_dict():
        report.errors.append(f"{label}:causal_snapshot_contract_detail mismatch")


def _validate_action_spec_frames(
    report: V23DatasetValidationReport,
    *,
    label: str,
    metadata: Mapping[str, Any],
    contract: PoseFrameContract,
) -> None:
    specs = metadata.get("arm_action_specs_by_side")
    references = metadata.get("arm_reference_frame_by_side")
    if not isinstance(specs, Mapping):
        report.errors.append(f"{label}:missing arm_action_specs_by_side")
        specs = {}
    if not isinstance(references, Mapping):
        report.errors.append(f"{label}:missing arm_reference_frame_by_side")
        references = {}
    for side in contract.source.active_sides:
        expected = (
            contract.canonical_absolute_frame
            if contract.pose_storage_frame == POSE_STORAGE_FRAME_DESK
            else contract.source.raw_frame(side)
        )
        spec = specs.get(side)
        if not isinstance(spec, Mapping) or spec.get("reference_frame") != expected:
            report.errors.append(f"{label}:{side} arm action reference frame is not {expected!r}")
        if references.get(side) != expected:
            report.errors.append(f"{label}:{side} arm_reference_frame is not {expected!r}")


def _validate_summary_video_paths(
    report: V23DatasetValidationReport,
    *,
    label: str,
    record: Mapping[str, Any],
) -> None:
    """Reject stale reindexing audit paths without requiring optional summaries."""

    video_paths = record.get("video_paths")
    summary = record.get("summary")
    if not isinstance(video_paths, Mapping) or not isinstance(summary, Mapping):
        return
    stats = summary.get("video_frame_stats")
    if not isinstance(stats, Mapping):
        return
    for camera_key, formal_path in video_paths.items():
        camera_stats = stats.get(camera_key)
        if not isinstance(camera_stats, Mapping) or camera_stats.get("path") is None:
            continue
        if str(camera_stats.get("path")) != str(formal_path):
            report.errors.append(
                f"{label}:summary.video_frame_stats.{camera_key}.path differs "
                f"from video_paths.{camera_key}"
            )


def _validate_row(
    report: V23DatasetValidationReport,
    *,
    row: Mapping[str, Any],
    episode_index: int,
    expected_frame_index: int,
    contract: PoseFrameContract,
    action_dim: int,
    state_dim: int,
    tactile_manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    label = f"episode={episode_index},frame={expected_frame_index}"
    if row.get("episode_index") != episode_index:
        report.errors.append(f"{label}:episode_index mismatch")
    if row.get("frame_index") != expected_frame_index:
        report.errors.append(f"{label}:frame_index mismatch")
    row_expected = {
        "dexteleop.dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        "dexteleop.pose_storage_frame": contract.pose_storage_frame,
        "dexteleop.pose_source_id": contract.source_id,
        "dexteleop.pose_frame_contract_hash": contract.contract_hash,
        "dexteleop.causal_snapshot_contract": DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.schema,
        "dexteleop.causal_snapshot_contract_hash": DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.contract_hash,
    }
    for key, value in row_expected.items():
        if row.get(key) != value:
            report.errors.append(f"{label}:{key} mismatch")
    expected_dims = {
        "action": action_dim,
        "action.arm.pose": 14,
        "observation.state": state_dim,
        "observation.arm.pose": 14,
        "observation.last_action": action_dim,
    }
    for key, expected_dim in expected_dims.items():
        try:
            size = int(np.asarray(row.get(key)).reshape(-1).size)
        except Exception:
            size = -1
        if size != int(expected_dim):
            report.errors.append(f"{label}:{key} dimension {size} != {expected_dim}")
    try:
        validate_v23_row_causal_snapshots(
            row,
            active_sides=contract.source.active_sides,
            arm_source_clocks=_arm_source_clocks(contract),
        )
    except ValueError as exc:
        report.errors.append(f"{label}:causal snapshot invalid: {exc}")

    rm65_source = contract.source_id.lower().startswith("rm65_")
    for side in contract.source.active_sides:
        prefix = f"observation.arm.{side}."
        actual = _pose7(row.get(f"{prefix}actual_tcp_pose"))
        if actual is None:
            report.errors.append(f"{label}:{side} actual_tcp_pose invalid")
            continue
        copies = {
            "observation side pose": _pose7(row.get(f"observation.{side}_arm.pose")),
            "packed observation.state": _packed_pose(row.get("observation.state"), side),
            "packed observation.arm.pose": _packed_pose(row.get("observation.arm.pose"), side),
        }
        for name, pose in copies.items():
            if not _same_pose(actual, pose):
                report.errors.append(f"{label}:{side} {name} differs from actual_tcp_pose")
        action = _packed_pose(row.get("action"), side)
        for name, pose in {
            "action.arm.pose": _packed_pose(row.get("action.arm.pose"), side),
            f"action.{side}_arm.pose": _pose7(row.get(f"action.{side}_arm.pose")),
        }.items():
            if not _same_pose(action, pose):
                report.errors.append(f"{label}:{side} {name} differs from packed action")
        if rm65_source:
            try:
                qpos = np.asarray(
                    row.get(f"{prefix}actual_hardware_qpos", []), dtype=float
                ).reshape(-1)
            except (TypeError, ValueError):
                qpos = np.asarray([], dtype=float)
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                report.errors.append(f"{label}:{side} actual_hardware_qpos invalid")
            if row.get(f"{prefix}actual_tcp_pose_source") != "hardware_qpos_fk":
                report.errors.append(f"{label}:{side} actual_tcp_pose_source is not hardware_qpos_fk")
        elif str(row.get(f"{prefix}actual_tcp_pose_source") or "") in {
            "",
            "current_tcp_pose",
            "commanded_tcp_pose",
            "last_commanded_action",
        }:
            report.errors.append(f"{label}:{side} actual_tcp_pose_source is not measured")
        for field, expected in {
            "actual_state_read_success": True,
            "actual_state_valid": True,
            "actual_state_stale": False,
            "causal_measured_state_eligible": True,
        }.items():
            if row.get(f"{prefix}{field}") is not expected:
                report.errors.append(f"{label}:{side} {field} is not {expected}")
        sequence = row.get(f"{prefix}actual_state_sequence")
        if isinstance(sequence, (bool, np.bool_)):
            valid_sequence = False
        else:
            try:
                valid_sequence = int(sequence) > 0 and int(sequence) == sequence
            except (TypeError, ValueError):
                valid_sequence = False
        if not valid_sequence:
            report.errors.append(f"{label}:{side} actual_state_sequence invalid")
        try:
            actual_timestamp = float(row.get(f"{prefix}actual_state_timestamp"))
            source_timestamp = float(row.get(f"{prefix}source_timestamp"))
            t0 = float(row.get("timestamp"))
        except (TypeError, ValueError):
            actual_timestamp = source_timestamp = t0 = math.nan
        if (
            not all(math.isfinite(value) for value in (actual_timestamp, source_timestamp, t0))
            or actual_timestamp > t0 + 1e-6
            or abs(actual_timestamp - source_timestamp) > 1e-6
        ):
            report.errors.append(f"{label}:{side} actual_state_timestamp invalid")
        audit_key = f"action.reference.{side}_arm.pose_ref"
        if row.get(audit_key) is not None:
            if _pose7(row.get(audit_key)) is None:
                report.errors.append(f"{label}:{audit_key} invalid")
            if row.get(f"{audit_key}_frame") != contract.source.raw_frame(side):
                report.errors.append(f"{label}:{audit_key}_frame mismatch")

        tactile_path = row.get(f"observation.hand.{side}.tactile_path")
        tactile_index = row.get(f"observation.hand.{side}.tactile_frame_index")
        if tactile_path is not None or tactile_index is not None:
            manifest = tactile_manifests.get(str(tactile_path))
            if manifest is None:
                report.errors.append(f"{label}:{side} tactile manifest reference is unknown")
            else:
                try:
                    index = int(tactile_index)
                    count = int(manifest.get("frame_count", -1))
                except (TypeError, ValueError):
                    index, count = -1, -1
                if index < 0 or index >= count:
                    report.errors.append(f"{label}:{side} tactile frame index out of range")


def _iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("v2.3 dataset validation requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def _validate_v23_dataset_impl(
    dataset_root: str | Path,
    *,
    max_errors: int = 200,
    allowed_missing_tactile_payloads: frozenset[str] = frozenset(),
    allowed_missing_tactile_manifests: Mapping[str, int] | None = None,
) -> V23DatasetValidationReport:
    """Validate a finalized native-v2.3 dataset without modifying it."""

    root = Path(dataset_root).expanduser().resolve()
    missing_tactile_manifests = dict(allowed_missing_tactile_manifests or {})
    report = V23DatasetValidationReport(dataset_root=str(root))
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        report.errors.append("missing meta/info.json")
        return report
    if not episodes_path.is_file():
        report.errors.append("missing meta/episodes.jsonl")
        return report
    try:
        info = _read_json(info_path)
        if not isinstance(info, dict):
            raise ValueError("info.json is not an object")
        if info.get("dexteleop_dataset_schema") != DEXTELEOP_DATASET_SCHEMA_V23:
            raise ValueError("dataset is not native DexTeleop v2.3")
        raw_contract = info.get(POSE_FRAME_CONTRACT_METADATA_KEY)
        if not isinstance(raw_contract, Mapping):
            raise ValueError("missing pose_frame_contract")
        contract = PoseFrameContract.from_dict(raw_contract)
        layout = policy_io_layout_from_metadata(info, allow_legacy_v21=False)
    except Exception as exc:
        report.errors.append(f"invalid dataset metadata: {exc}")
        return report

    _validate_metadata_binding(
        report, label="info", metadata=info, contract=contract, layout_hash=layout.layout_hash
    )
    if (root / "meta" / "progress").exists():
        try:
            # Local import keeps the core v2.3 validator usable without
            # creating a reverse dependency from progress readers.
            from .progress_artifacts import load_dataset_progress_profile

            load_dataset_progress_profile(root)
        except Exception as exc:
            report.errors.append(
                f"invalid dataset progress profile binding: {exc}"
            )
    _validate_action_spec_frames(report, label="info", metadata=info, contract=contract)
    for side in contract.source.active_sides:
        expected = (
            contract.canonical_absolute_frame
            if contract.pose_storage_frame == POSE_STORAGE_FRAME_DESK
            else contract.source.raw_frame(side)
        )
        if layout.arm_reference_frame_by_side[side] != expected:
            report.errors.append(f"info:{side} policy layout reference frame mismatch")
    try:
        episodes = _read_jsonl(episodes_path)
    except Exception as exc:
        report.errors.append(f"invalid episodes.jsonl: {exc}")
        return report
    report.episode_count = len(episodes)

    seen_episode_indices: set[int] = set()
    global_indices: list[int] = []
    for record in episodes:
        if len(report.errors) >= max_errors:
            report.errors.append(f"validation stopped after {max_errors} errors")
            break
        try:
            episode_index = int(record["episode_index"])
        except (KeyError, TypeError, ValueError):
            report.errors.append("episode record has invalid episode_index")
            continue
        label = f"episode={episode_index}"
        if episode_index in seen_episode_indices:
            report.errors.append(f"{label}:duplicate episode_index")
        seen_episode_indices.add(episode_index)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            report.errors.append(f"{label}:missing metadata")
            metadata = {}
        _validate_metadata_binding(
            report,
            label=label,
            metadata=metadata,
            contract=contract,
            layout_hash=layout.layout_hash,
        )
        _validate_action_spec_frames(report, label=label, metadata=metadata, contract=contract)
        _validate_summary_video_paths(report, label=label, record=record)

        tactile_manifests: dict[str, Mapping[str, Any]] = {}
        for map_key in _ARTIFACT_MAP_KEYS:
            paths = record.get(map_key, {})
            if not isinstance(paths, Mapping):
                report.errors.append(f"{label}:{map_key} is not an object")
                continue
            for name, relative in paths.items():
                artifact = root / str(relative)
                if not artifact.is_file():
                    if (
                        map_key == "tactile_paths"
                        and str(relative) in missing_tactile_manifests
                    ):
                        tactile_manifests[str(relative)] = {
                            "frame_count": int(
                                missing_tactile_manifests[str(relative)]
                            ),
                            "streams": {},
                        }
                        continue
                    report.errors.append(f"{label}:{map_key}.{name} missing: {relative}")
                    continue
                if map_key in {"tactile_paths", "wrist_depth_mask_paths", "wrist_depth_raw_paths"}:
                    try:
                        manifest = _read_json(artifact)
                        if not isinstance(manifest, Mapping):
                            raise ValueError("manifest is not an object")
                    except Exception as exc:
                        report.errors.append(f"{label}:{relative} invalid manifest: {exc}")
                        continue
                    if map_key == "tactile_paths":
                        tactile_manifests[str(relative)] = manifest
                    streams = manifest.get("streams", {})
                    if isinstance(streams, Mapping):
                        for stream_name, stream in streams.items():
                            if isinstance(stream, Mapping) and stream.get("path"):
                                stream_relative = str(stream["path"])
                                if (
                                    not (root / stream_relative).is_file()
                                    and not (
                                        map_key == "tactile_paths"
                                        and stream_relative
                                        in allowed_missing_tactile_payloads
                                    )
                                ):
                                    report.errors.append(
                                        f"{label}:{relative} stream {stream_name!r} is missing"
                                    )
                    elif manifest.get("stream_path") and not (root / str(manifest["stream_path"])).is_file():
                        report.errors.append(f"{label}:{relative} referenced stream is missing")

        parquet_relative = record.get("parquet_path")
        if not parquet_relative:
            report.errors.append(f"{label}:missing parquet_path")
            continue
        parquet_path = root / str(parquet_relative)
        if not parquet_path.is_file():
            report.errors.append(f"{label}:missing parquet file {parquet_relative}")
            continue
        frame_count = 0
        try:
            for frame_count, row in enumerate(_iter_parquet_rows(parquet_path), 1):
                _validate_row(
                    report,
                    row=row,
                    episode_index=episode_index,
                    expected_frame_index=frame_count - 1,
                    contract=contract,
                    action_dim=layout.action_dim,
                    state_dim=layout.state_dim,
                    tactile_manifests=tactile_manifests,
                )
                value = row.get("index")
                if isinstance(value, (int, np.integer)):
                    global_indices.append(int(value))
                else:
                    report.errors.append(f"{label},frame={frame_count - 1}:invalid global index")
                if len(report.errors) >= max_errors:
                    break
        except Exception as exc:
            report.errors.append(f"{label}:cannot scan parquet: {exc}")
        report.frame_count += frame_count
        try:
            declared_frames = int(record.get("frame_count"))
        except (TypeError, ValueError):
            declared_frames = -1
        if declared_frames != frame_count:
            report.errors.append(
                f"{label}:frame_count mismatch ({declared_frames} declared, {frame_count} parquet)"
            )

    try:
        info_episode_count = int(info.get("episode_count", -1))
    except (TypeError, ValueError):
        info_episode_count = -1
    try:
        info_frame_count = int(info.get("frame_count", -1))
    except (TypeError, ValueError):
        info_frame_count = -1
    if info_episode_count != report.episode_count:
        report.errors.append("info:episode_count mismatch")
    if info_frame_count != report.frame_count:
        report.errors.append("info:frame_count mismatch")
    if global_indices and global_indices != list(range(len(global_indices))):
        report.errors.append("dataset global index is not contiguous from zero")
    return report


def validate_v23_dataset(
    dataset_root: str | Path,
    *,
    max_errors: int = 200,
) -> V23DatasetValidationReport:
    """Validate a physically self-contained native-v2.3 dataset."""

    return _validate_v23_dataset_impl(dataset_root, max_errors=max_errors)


def validate_v23_core_dataset(
    dataset_root: str | Path,
    *,
    verified_inventories: Iterable[Any] = (),
    verified_detachment_scopes: Mapping[str, str] | None = None,
    max_errors: int = 200,
) -> V23DatasetValidationReport:
    """Validate v2.3 core artifacts while honoring verified detached modules.

    Payload-only mode keeps capture-time tactile manifests hot.  Whole-prefix
    mode may omit the complete tactile tree when the caller has already
    verified the corresponding storage receipt.  The strict
    ``validate_v23_dataset`` behavior is unchanged.
    """

    from dataset_contracts.modality_artifacts import ModalityArtifactInventoryV1
    from .modality_artifacts import (
        DETACHED_VERIFIED,
        DETACHMENT_SCOPES,
        PAYLOAD_ONLY_DETACH,
        SELF_CONTAINED,
        WHOLE_PREFIX_DETACH,
        resolve_modality_availability,
        verified_detachment_scope,
    )

    root = Path(dataset_root).expanduser().resolve()
    allowed_missing: set[str] = set()
    allowed_missing_manifests: dict[str, int] = {}
    seen_modalities: set[str] = set()
    availability_warnings: list[str] = []
    scopes = {
        str(key): str(value or "").strip().lower().replace("-", "_")
        for key, value in dict(verified_detachment_scopes or {}).items()
    }
    for inventory in tuple(verified_inventories):
        if not isinstance(inventory, ModalityArtifactInventoryV1):
            report = V23DatasetValidationReport(dataset_root=str(root))
            report.errors.append("verified inventory has an unsupported type")
            return report
        if inventory.modality_key in seen_modalities:
            report = V23DatasetValidationReport(dataset_root=str(root))
            report.errors.append(
                f"duplicate verified inventory for modality {inventory.modality_key!r}"
            )
            return report
        seen_modalities.add(inventory.modality_key)
        scope = scopes.get(inventory.modality_key, PAYLOAD_ONLY_DETACH)
        if scope not in DETACHMENT_SCOPES:
            report = V23DatasetValidationReport(dataset_root=str(root))
            report.errors.append(
                f"unsupported detachment scope for {inventory.modality_key!r}: {scope!r}"
            )
            return report
        availability = resolve_modality_availability(
            root,
            inventory,
            detachment_scope=scope,
        )
        if availability.status not in {SELF_CONTAINED, DETACHED_VERIFIED}:
            report = V23DatasetValidationReport(dataset_root=str(root))
            report.errors.append(
                "modality inventory is not core-validation eligible: "
                f"{inventory.modality_key}:{availability.status}:"
                f"{availability.reason}"
            )
            return report
        if availability.status == DETACHED_VERIFIED:
            try:
                evidenced_scope = verified_detachment_scope(
                    root,
                    inventory.semantic_content_hash,
                )
            except Exception as exc:
                report = V23DatasetValidationReport(dataset_root=str(root))
                report.errors.append(
                    f"invalid modality storage evidence for "
                    f"{inventory.modality_key!r}: {exc}"
                )
                return report
            if evidenced_scope != scope:
                report = V23DatasetValidationReport(dataset_root=str(root))
                report.errors.append(
                    "detached modality lacks a matching durable retirement receipt: "
                    f"{inventory.modality_key}:{scope}:{evidenced_scope}"
                )
                return report
            allowed_missing.update(
                artifact.logical_path
                for artifact in inventory.artifacts
                if artifact.role == "payload"
            )
            if scope == WHOLE_PREFIX_DETACH:
                allowed_missing_manifests.update(
                    {
                        artifact.logical_path: int(artifact.frame_count or 0)
                        for artifact in inventory.artifacts
                        if artifact.role == "manifest"
                    }
                )
            availability_warnings.append(
                f"{inventory.modality_key} {scope} artifacts are DETACHED_VERIFIED; "
                "tactile-enabled readers require rehydrate"
            )
    unknown_scopes = sorted(set(scopes) - seen_modalities)
    if unknown_scopes:
        report = V23DatasetValidationReport(dataset_root=str(root))
        report.errors.append(
            f"detachment scopes reference unknown modalities: {unknown_scopes}"
        )
        return report
    report = _validate_v23_dataset_impl(
        root,
        max_errors=max_errors,
        allowed_missing_tactile_payloads=frozenset(allowed_missing),
        allowed_missing_tactile_manifests=allowed_missing_manifests,
    )
    report.warnings.extend(availability_warnings)
    return report


__all__ = [
    "V23DatasetValidationReport",
    "validate_v23_core_dataset",
    "validate_v23_dataset",
]
