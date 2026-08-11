"""Explicit UMI migration from same-sample to next-observation arm actions."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from collection_runtime.artifact_integrity import digest_stored_file
from collection_runtime.modality_inventory import publish_tracked_modality_inventory
from dataset_interface.modality_artifacts import PAYLOAD_HASH_PREFER_SEALED

from .tactile_recompress import _ZstdCodec, _active_collection_processes


MIGRATION_ID = "2026-07-20_umi_arm_action_next_accepted_observation_v1"
MIGRATION_SCHEMA = "dexteleop.umi_arm_action_alignment.v1"
EXPECTED_DATASET_SCHEMA = "dexteleop_dataset_v2.3"
EXPECTED_EMBODIMENT = "umi_dual_wrist"
ARM_DIM = 14
MODEL_DIM = 16
MIN_FREE_AFTER_STAGING_BYTES = 5 * 1024**3


@dataclass(frozen=True)
class UmiArmAlignmentPlan:
    dataset_root: str
    episode_count: int
    source_frame_count: int
    output_frame_count: int
    dropped_terminal_rows: int
    video_count: int
    tactile_manifest_count: int
    tactile_frames_to_drop: int
    dataset_bytes: int
    free_bytes: int
    minimum_free_after_staging_bytes: int
    mode: str = "dry-run"
    backup_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(payload)
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    dict(payload), ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_path(root: Path, value: Any, *, name: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {name}: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes dataset root: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _alignment_payload() -> dict[str, Any]:
    return {
        "schema": MIGRATION_SCHEMA,
        "migration_id": MIGRATION_ID,
        "arm_action_t": "observation.arm.pose[t+1]",
        "arm_action_source": "next_accepted_observation",
        "gripper_action_t": "angle_trigger_final_command[t]",
        "gripper_action_alignment": "same_sample_command",
        "last_action_t": "migrated_action[t-1]",
        "first_last_action_arm": "observation.arm.pose[t]",
        "first_last_action_gripper": "angle_trigger_final_command[t]",
        "terminal_policy": "drop_last_source_sample",
        "media_policy": "drop_terminal_sample_media",
    }


def _validate_info(info: Mapping[str, Any]) -> None:
    if str(info.get("dexteleop_dataset_schema")) != EXPECTED_DATASET_SCHEMA:
        raise ValueError("UMI arm alignment only accepts DexTeleop v2.3 datasets")
    if str(info.get("embodiment")) != EXPECTED_EMBODIMENT:
        raise ValueError("UMI arm alignment only accepts embodiment=umi_dual_wrist")
    if info.get("umi_arm_action_alignment"):
        raise RuntimeError("dataset already declares umi_arm_action_alignment")
    for record in info.get("postprocess_history", ()) or ():
        if isinstance(record, Mapping) and record.get("migration_id") == MIGRATION_ID:
            raise RuntimeError(f"migration {MIGRATION_ID} was already applied")


def _parquet_compression(path: Path) -> str:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    if metadata.num_row_groups <= 0 or metadata.row_group(0).num_columns <= 0:
        return "zstd"
    return str(metadata.row_group(0).column(0).compression).lower()


def _replace_column(table: Any, name: str, values: Sequence[Any]) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        return table.append_column(name, pa.array(values))
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _required_vector(rows: Sequence[Mapping[str, Any]], name: str, length: int) -> list[list[float]]:
    output: list[list[float]] = []
    for row_index, row in enumerate(rows):
        value = row.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(
                f"row {row_index} column {name!r} must have length {length}"
            )
        vector = [float(item) for item in value]
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"row {row_index} column {name!r} is non-finite")
        output.append(vector)
    return output


def _aligned_table(table: Any, *, global_index_start: int, path: Path) -> Any:
    rows = table.to_pylist()
    if len(rows) < 2:
        raise ValueError(f"{path} has fewer than two source rows")
    frame_indices = [int(row.get("frame_index", -1)) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError(f"{path} frame_index is not contiguous from zero")

    actions = _required_vector(rows, "action", MODEL_DIM)
    states = _required_vector(rows, "observation.state", MODEL_DIM)
    action_arm = _required_vector(rows, "action.arm.pose", ARM_DIM)
    state_arm = _required_vector(rows, "observation.arm.pose", ARM_DIM)
    action_left = _required_vector(rows, "action.left_arm.pose", 7)
    action_right = _required_vector(rows, "action.right_arm.pose", 7)
    state_left = _required_vector(rows, "observation.left_arm.pose", 7)
    state_right = _required_vector(rows, "observation.right_arm.pose", 7)
    if not np.array_equal(np.asarray(actions)[:, :ARM_DIM], np.asarray(states)[:, :ARM_DIM]):
        raise ValueError(f"{path} is not legacy same-sample-coupled UMI arm data")
    for action_values, state_values, name in (
        (action_arm, state_arm, "action.arm.pose"),
        (action_left, state_left, "action.left_arm.pose"),
        (action_right, state_right, "action.right_arm.pose"),
    ):
        if not np.array_equal(np.asarray(action_values), np.asarray(state_values)):
            raise ValueError(f"{path} {name} is not same-sample-coupled")

    keep = len(rows) - 1
    migrated_actions = [states[index + 1][:ARM_DIM] + actions[index][ARM_DIM:] for index in range(keep)]
    migrated_action_arm = [state_arm[index + 1] for index in range(keep)]
    migrated_left = [state_left[index + 1] for index in range(keep)]
    migrated_right = [state_right[index + 1] for index in range(keep)]
    first_last = states[0][:ARM_DIM] + actions[0][ARM_DIM:]
    migrated_last = [first_last] + [migrated_actions[index - 1] for index in range(1, keep)]
    source_indices = [frame_indices[index + 1] for index in range(keep)]
    source_timestamps = [float(rows[index + 1]["timestamp"]) for index in range(keep)]

    output = table.slice(0, keep)
    for name, values in (
        ("action", migrated_actions),
        ("action.arm.pose", migrated_action_arm),
        ("action.left_arm.pose", migrated_left),
        ("action.right_arm.pose", migrated_right),
        ("observation.last_action", migrated_last),
        ("index", list(range(global_index_start, global_index_start + keep))),
        ("action.arm.temporal_alignment", ["next_accepted_observation"] * keep),
        ("action.arm.source_frame_index", source_indices),
        ("action.arm.source_timestamp", source_timestamps),
    ):
        output = _replace_column(output, name, values)
    return output


def _ffprobe_frame_count(path: Path) -> int:
    executable = shutil.which("ffprobe")
    if not executable:
        raise FileNotFoundError("ffprobe is required for synchronized UMI video migration")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return int(result.stdout.strip())


def _trim_video(path: Path, *, source_count: int, keep_count: int) -> None:
    actual = _ffprobe_frame_count(path)
    if actual != source_count:
        raise ValueError(f"video frame count mismatch for {path}: {actual} != {source_count}")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FileNotFoundError("ffmpeg is required for synchronized UMI video migration")
    temporary = path.with_name(f".{path.name}.umi_arm_align.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-frames:v",
                str(keep_count),
                "-c:v",
                "copy",
                "-an",
                str(temporary),
            ],
            check=True,
        )
        trimmed = _ffprobe_frame_count(temporary)
        if trimmed != keep_count:
            raise ValueError(f"trimmed video frame count mismatch: {trimmed} != {keep_count}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compression_name(spec: Mapping[str, Any]) -> str:
    value = str(spec.get("compression") or "").lower().replace("-", "_")
    if value in {"", "none", "raw", "uncompressed", "raw_c_order"}:
        return "none"
    return value


def _bytes_per_tactile_frame(spec: Mapping[str, Any]) -> int:
    shape = tuple(int(value) for value in spec.get("shape", ()) or ())
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"invalid tactile shape: {shape}")
    storage_dtype = str(spec.get("storage_dtype") or spec.get("dtype") or "")
    return int(math.prod(shape) * np.dtype(storage_dtype).itemsize)


def _copy_prefix(source: Path, target: Path, byte_count: int) -> None:
    remaining = int(byte_count)
    with source.open("rb") as input_handle, target.open("wb") as output_handle:
        while remaining:
            payload = input_handle.read(min(16 * 1024**2, remaining))
            if not payload:
                raise ValueError(f"short tactile stream read from {source}")
            output_handle.write(payload)
            remaining -= len(payload)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def _trim_raw_tactile(path: Path, spec: Mapping[str, Any], *, keep_count: int) -> None:
    bytes_per_frame = _bytes_per_tactile_frame(spec)
    source_count = int(spec.get("frame_count") or 0)
    expected_bytes = source_count * bytes_per_frame
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"raw tactile size mismatch for {path}")
    temporary = path.with_name(f".{path.name}.umi_arm_align.tmp")
    temporary.unlink(missing_ok=True)
    try:
        _copy_prefix(path, temporary, keep_count * bytes_per_frame)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trim_zstd_tactile(
    path: Path,
    spec: Mapping[str, Any],
    *,
    keep_count: int,
    codec: _ZstdCodec,
) -> list[dict[str, Any]]:
    bytes_per_frame = _bytes_per_tactile_frame(spec)
    chunks = list(spec.get("chunks") or ())
    if not chunks:
        raise ValueError(f"chunked tactile stream has no chunks: {path}")
    temporary = path.with_name(f".{path.name}.umi_arm_align.tmp")
    temporary.unlink(missing_ok=True)
    updated: list[dict[str, Any]] = []
    try:
        with path.open("rb") as input_handle, temporary.open("wb") as output_handle:
            for raw_chunk in chunks:
                chunk = dict(raw_chunk)
                start = int(chunk["frame_start"])
                count = int(chunk["frame_count"])
                if start >= keep_count:
                    break
                input_handle.seek(int(chunk["offset"]))
                compressed = input_handle.read(int(chunk["compressed_bytes"]))
                retained = min(count, keep_count - start)
                if retained < count:
                    raw = codec.decompress(compressed, raw_bytes=int(chunk["raw_bytes"]))
                    expected = count * bytes_per_frame
                    if len(raw) != expected:
                        raise ValueError(f"decompressed tactile size mismatch for {path}")
                    compressed = codec.compress(raw[: retained * bytes_per_frame])
                offset = output_handle.tell()
                output_handle.write(compressed)
                updated_chunk = dict(chunk)
                updated_chunk.update(
                    {
                        "frame_start": start,
                        "frame_count": retained,
                        "offset": offset,
                        "compressed_bytes": len(compressed),
                        "raw_bytes": retained * bytes_per_frame,
                    }
                )
                updated.append(updated_chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if sum(int(chunk["frame_count"]) for chunk in updated) != keep_count:
            raise ValueError(f"trimmed tactile chunk coverage mismatch for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return updated


def _manifest_keep_count(manifest: Mapping[str, Any], *, terminal_sample_index: int) -> int:
    frames = list(manifest.get("frames") or ())
    source_count = int(manifest.get("frame_count") or 0)
    if len(frames) != source_count:
        raise ValueError("tactile manifest frame_count does not match frames")
    keep = 0
    dropping = False
    for tactile_index, frame in enumerate(frames):
        if int(frame.get("tactile_frame_index", -1)) != tactile_index:
            raise ValueError("tactile_frame_index is not a contiguous prefix")
        sample_index = int(frame.get("sample_frame_index", tactile_index))
        should_keep = sample_index < terminal_sample_index
        if should_keep and dropping:
            raise ValueError("retained tactile frames are not a prefix")
        if should_keep:
            keep += 1
        else:
            dropping = True
    return keep


def _trim_tactile_manifest(
    root: Path,
    manifest_path: Path,
    *,
    terminal_sample_index: int,
    codec: _ZstdCodec | None,
) -> int:
    manifest = _read_json(manifest_path)
    keep = _manifest_keep_count(manifest, terminal_sample_index=terminal_sample_index)
    source_count = int(manifest["frame_count"])
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping):
        raise ValueError(f"tactile manifest has no streams mapping: {manifest_path}")
    updated_streams: dict[str, Any] = {}
    for sensor_key, raw_spec in streams.items():
        spec = dict(raw_spec)
        if int(spec.get("frame_count") or 0) != source_count:
            raise ValueError(f"tactile stream count mismatch in {manifest_path}")
        path = _safe_path(root, spec.get("path"), name="tactile stream path")
        compression = _compression_name(spec)
        if keep < source_count:
            if compression == "none":
                _trim_raw_tactile(path, spec, keep_count=keep)
            elif compression == "zstd":
                if codec is None:
                    codec = _ZstdCodec(level=1, backend="auto")
                spec["chunks"] = _trim_zstd_tactile(
                    path, spec, keep_count=keep, codec=codec
                )
            else:
                raise ValueError(f"unsupported tactile compression {compression!r}")
        spec["frame_count"] = keep
        sealed = digest_stored_file(path)
        spec.update(
            {
                "stored_size_bytes": int(sealed.size_bytes),
                "stored_sha256": sealed.sha256,
                "stored_hash_profile": sealed.profile,
            }
        )
        updated_streams[str(sensor_key)] = spec
    manifest["frame_count"] = keep
    manifest["frames"] = list(manifest.get("frames") or ())[:keep]
    manifest["streams"] = updated_streams
    manifest["umi_arm_action_alignment"] = {
        "migration_id": MIGRATION_ID,
        "terminal_sample_index": int(terminal_sample_index),
        "source_frame_count": source_count,
        "retained_frame_count": keep,
    }
    _write_json(manifest_path, manifest)
    return source_count - keep


def _updated_episode_record(
    record: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    source_count: int,
) -> dict[str, Any]:
    output = deepcopy(dict(record))
    keep = source_count - 1
    start = float(rows[0]["timestamp"])
    end = float(rows[keep - 1]["timestamp"])
    output.update(
        {
            "frame_count": keep,
            "start_timestamp": start,
            "end_timestamp": end,
            "duration_sec": end - start,
        }
    )
    metadata = dict(output.get("metadata") or {})
    metadata["umi_arm_action_alignment"] = _alignment_payload()
    output["metadata"] = metadata
    summary = dict(output.get("summary") or {})
    summary.update(
        {
            "frame_count": keep,
            "source_sample_count": source_count,
            "terminal_source_sample_dropped": True,
            "arm_action_alignment": "next_accepted_observation",
        }
    )
    video_stats = dict(summary.get("video_frame_stats") or {})
    for camera, raw_stats in video_stats.items():
        stats = dict(raw_stats)
        stats.update({"expected": keep, "written": keep, "missing": 0, "dropped": 0})
        video_stats[str(camera)] = stats
    summary["video_frame_stats"] = video_stats
    output["summary"] = summary
    return output


def _updated_stats_record(
    record: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]], source_count: int
) -> dict[str, Any]:
    output = deepcopy(dict(record))
    keep = source_count - 1
    timestamps = [float(row["timestamp"]) for row in rows[:keep]]
    duration = timestamps[-1] - timestamps[0]
    average_dt = duration / (keep - 1) if keep > 1 else None
    output.update(
        {
            "frame_count": keep,
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "duration_sec": duration,
            "average_dt_sec": average_dt,
            "average_hz": None if not average_dt or average_dt <= 0 else 1.0 / average_dt,
        }
    )
    return output


def _preflight(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], UmiArmAlignmentPlan]:
    import pyarrow.parquet as pq

    info = _read_json(root / "meta" / "info.json")
    _validate_info(info)
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
    if [int(row["episode_index"]) for row in episodes] != list(range(len(episodes))):
        raise ValueError("episode indices are not contiguous from zero")
    stats_by_episode = {int(row["episode_index"]): row for row in stats}
    source_total = 0
    output_total = 0
    video_count = 0
    tactile_count = 0
    tactile_drops = 0
    global_index = 0
    for record in episodes:
        episode_index = int(record["episode_index"])
        parquet_path = _safe_path(root, record.get("parquet_path"), name="parquet_path")
        table = pq.read_table(parquet_path)
        source_count = len(table)
        if int(record.get("frame_count") or 0) != source_count:
            raise ValueError(f"episode {episode_index} metadata/parquet frame count mismatch")
        _aligned_table(table, global_index_start=global_index, path=parquet_path)
        rows = table.to_pylist()
        global_index += source_count - 1
        source_total += source_count
        output_total += source_count - 1
        if stats and episode_index not in stats_by_episode:
            raise ValueError(f"missing episodes_stats record for episode {episode_index}")
        if record.get("wrist_depth_mask_paths") or record.get("wrist_depth_raw_paths"):
            raise ValueError("UMI arm alignment does not yet support wrist-depth media")
        video_paths = dict(record.get("video_paths") or {})
        if not video_paths:
            raise ValueError(f"episode {episode_index} has no video_paths")
        for value in video_paths.values():
            video = _safe_path(root, value, name="video_path")
            if _ffprobe_frame_count(video) != source_count:
                raise ValueError(f"episode {episode_index} video frame count mismatch: {video}")
            video_count += 1
        tactile_paths = dict(record.get("tactile_paths") or {})
        tactile_keep_by_side: dict[str, tuple[str, int]] = {}
        for side, value in tactile_paths.items():
            manifest_path = _safe_path(root, value, name="tactile_path")
            manifest = _read_json(manifest_path)
            keep = _manifest_keep_count(
                manifest, terminal_sample_index=source_count - 1
            )
            tactile_drops += int(manifest.get("frame_count") or 0) - keep
            tactile_count += 1
            tactile_keep_by_side[str(side)] = (str(value), keep)
        # Check that retained rows never reference a tactile frame being removed.
        for row in rows[:-1]:
            for side in ("left", "right"):
                raw_ref = row.get(f"observation.hand.{side}.tactile_storage_ref_json")
                if raw_ref in (None, "", "null"):
                    continue
                ref = json.loads(str(raw_ref))
                frame_index = int(ref.get("frame_index", -1))
                expected = tactile_keep_by_side.get(side)
                if expected is None:
                    raise ValueError(
                        f"episode {episode_index} has a {side} tactile ref but no manifest"
                    )
                expected_path, keep_count = expected
                if str(ref.get("path") or "") != expected_path:
                    raise ValueError(
                        f"episode {episode_index} {side} tactile ref path mismatch"
                    )
                if frame_index < 0 or frame_index >= keep_count:
                    raise ValueError(
                        f"episode {episode_index} {side} tactile ref {frame_index} "
                        f"is outside retained prefix [0, {keep_count})"
                    )
    if int(info.get("episode_count") or -1) != len(episodes):
        raise ValueError("info.json episode_count is inconsistent")
    if int(info.get("frame_count") or -1) != source_total:
        raise ValueError("info.json frame_count is inconsistent")
    plan = UmiArmAlignmentPlan(
        dataset_root=str(root),
        episode_count=len(episodes),
        source_frame_count=source_total,
        output_frame_count=output_total,
        dropped_terminal_rows=len(episodes),
        video_count=video_count,
        tactile_manifest_count=tactile_count,
        tactile_frames_to_drop=tactile_drops,
        dataset_bytes=sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        free_bytes=shutil.disk_usage(root).free,
        minimum_free_after_staging_bytes=MIN_FREE_AFTER_STAGING_BYTES,
    )
    return info, episodes, stats, plan


def migrate_umi_arm_action_to_next_observation(
    dataset_root: str | Path,
    *,
    backup_root: str | Path | None = None,
    apply_umi_arm_alignment: bool = False,
) -> UmiArmAlignmentPlan:
    """Plan or explicitly apply the historical UMI temporal correction."""

    import pyarrow.parquet as pq

    root = Path(dataset_root).expanduser().resolve()
    info, episodes, stats, plan = _preflight(root)
    if not apply_umi_arm_alignment:
        return plan
    active = _active_collection_processes()
    if active:
        raise RuntimeError(
            "Refusing UMI arm alignment while formal collection is active: "
            + "; ".join(active[:3])
        )
    if plan.free_bytes - plan.dataset_bytes < plan.minimum_free_after_staging_bytes:
        raise OSError(
            "Insufficient free space for conservative copy-on-write staging: "
            f"free={plan.free_bytes}, dataset={plan.dataset_bytes}, "
            f"reserve={plan.minimum_free_after_staging_bytes}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = (
        Path(backup_root).expanduser().resolve()
        if backup_root is not None
        else root.with_name(f"{root.name}.before_umi_arm_alignment_{stamp}")
    )
    if backup.exists():
        raise FileExistsError(f"backup target already exists: {backup}")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.umi_arm_alignment_stage_", dir=root.parent))
    shutil.rmtree(stage)
    try:
        shutil.copytree(root, stage, copy_function=os.link)
        updated_episodes: list[dict[str, Any]] = []
        stats_by_episode = {int(row["episode_index"]): row for row in stats}
        updated_stats: list[dict[str, Any]] = []
        global_index = 0
        zstd_codec: _ZstdCodec | None = None
        for record in episodes:
            episode_index = int(record["episode_index"])
            relative_parquet = Path(str(record["parquet_path"]))
            source_parquet = root / relative_parquet
            stage_parquet = stage / relative_parquet
            table = pq.read_table(source_parquet)
            rows = table.to_pylist()
            source_count = len(table)
            migrated = _aligned_table(
                table, global_index_start=global_index, path=source_parquet
            )
            global_index += len(migrated)
            temporary_parquet = stage_parquet.with_name(f".{stage_parquet.name}.tmp")
            pq.write_table(
                migrated,
                temporary_parquet,
                compression=_parquet_compression(source_parquet),
            )
            os.replace(temporary_parquet, stage_parquet)

            for value in dict(record.get("video_paths") or {}).values():
                _trim_video(
                    _safe_path(stage, value, name="video_path"),
                    source_count=source_count,
                    keep_count=source_count - 1,
                )
            for value in dict(record.get("tactile_paths") or {}).values():
                manifest_path = _safe_path(stage, value, name="tactile_path")
                manifest = _read_json(manifest_path)
                if any(
                    _compression_name(dict(spec)) == "zstd"
                    for spec in dict(manifest.get("streams") or {}).values()
                ) and zstd_codec is None:
                    zstd_codec = _ZstdCodec(level=1, backend="auto")
                _trim_tactile_manifest(
                    stage,
                    manifest_path,
                    terminal_sample_index=source_count - 1,
                    codec=zstd_codec,
                )
            updated_episodes.append(
                _updated_episode_record(record, rows=rows, source_count=source_count)
            )
            if stats:
                updated_stats.append(
                    _updated_stats_record(
                        stats_by_episode[episode_index],
                        rows=rows,
                        source_count=source_count,
                    )
                )

        applied_at = datetime.now(timezone.utc).isoformat()
        history_record = {
            **_alignment_payload(),
            "applied_at_utc": applied_at,
            "source_frame_count": plan.source_frame_count,
            "output_frame_count": plan.output_frame_count,
            "dropped_terminal_rows": plan.dropped_terminal_rows,
        }
        updated_info = deepcopy(info)
        updated_info["frame_count"] = plan.output_frame_count
        updated_info["umi_arm_action_alignment"] = _alignment_payload()
        updated_info.setdefault("postprocess_history", []).append(history_record)
        _write_json(stage / "meta" / "info.json", updated_info)
        _write_jsonl(stage / "meta" / "episodes.jsonl", updated_episodes)
        if stats:
            _write_jsonl(stage / "meta" / "episodes_stats.jsonl", updated_stats)
        _write_json(stage / "meta" / "umi_arm_action_alignment.json", history_record)

        # The staged tactile graph was copy-on-write replaced above.  Preserve
        # an opted-in dataset's immutable revision chain before swapping roots.
        publish_tracked_modality_inventory(
            stage,
            payload_hash_mode=PAYLOAD_HASH_PREFER_SEALED,
        )

        # A second full preflight cannot be used because it intentionally rejects
        # already-aligned datasets. Verify the global count and all staged files first.
        staged_rows = sum(
            len(pq.read_table(stage / str(record["parquet_path"])))
            for record in updated_episodes
        )
        if staged_rows != plan.output_frame_count:
            raise ValueError("staged Parquet frame count mismatch")

        os.replace(root, backup)
        try:
            os.replace(stage, root)
        except BaseException:
            os.replace(backup, root)
            raise
        return UmiArmAlignmentPlan(
            **{
                **plan.to_dict(),
                "mode": "applied",
                "backup_root": str(backup),
            }
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "MIGRATION_ID",
    "UmiArmAlignmentPlan",
    "migrate_umi_arm_action_to_next_observation",
]
