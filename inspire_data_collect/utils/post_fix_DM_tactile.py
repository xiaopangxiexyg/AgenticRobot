"""Patch rare missing DMTAC tactile sensor frames in a local LeRobot-style dataset.

The gripper DMTAC collector normally stores two tactile sensor images per frame as:

    observation.hand.right.tactile_regions = ["L25480055", "L25480053"]
    observation.hand.right.tactile         = flatten([sensor_55, sensor_53])

For a few frames one sensor may be missing. This script fills each missing sensor
with the average of the nearest previous and next complete valid frames in the same
episode, then rewrites only the affected parquet files.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


DEFAULT_DATASET_ROOT = Path("/home/ai/zc/datasets/2026-05-12_gripper_collect_peeling_official")
TACTILE_COLUMN_TEMPLATE = "observation.hand.{side}.tactile"
REGIONS_COLUMN_TEMPLATE = "observation.hand.{side}.tactile_regions"
DICT_COLUMN_TEMPLATE = "observation.hand.{side}.tactile_dict_json"


@dataclasses.dataclass(frozen=True)
class TactileLayout:
    side: str
    sensors: tuple[str, ...]
    sensor_shape: tuple[int, int, int]

    @property
    def per_sensor_dim(self) -> int:
        return int(np.prod(self.sensor_shape))

    @property
    def full_dim(self) -> int:
        return len(self.sensors) * self.per_sensor_dim

    @property
    def tactile_column(self) -> str:
        return TACTILE_COLUMN_TEMPLATE.format(side=self.side)

    @property
    def regions_column(self) -> str:
        return REGIONS_COLUMN_TEMPLATE.format(side=self.side)

    @property
    def dict_column(self) -> str:
        return DICT_COLUMN_TEMPLATE.format(side=self.side)


@dataclasses.dataclass(frozen=True)
class FramePatch:
    episode_index: int
    parquet_path: Path
    frame_index: int
    old_regions: tuple[str, ...]
    missing_sensors: tuple[str, ...]
    previous_frame: int
    next_frame: int


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "post_fix_DM_tactile.py requires pyarrow. Run it in an environment with pyarrow installed, "
            "for example from /home/ai/zc/openpi: "
            "`UV_CACHE_DIR=/tmp/openpi-uv-cache /home/ai/.local/bin/uv run python "
            "/home/ai/zc/DexTeleop/inspire_data_collect/utils/post_fix_DM_tactile.py --apply`."
        ) from exc
    return pa, pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError(f"Expected sensor shape as H,W,C, got {value!r}.")
    return parts


def _load_layout(dataset_root: Path, side: str, sensors_arg: str | None, shape_arg: str | None) -> TactileLayout:
    sensors = tuple(sensor.strip() for sensor in sensors_arg.split(",")) if sensors_arg else ()
    sensor_shape = _parse_shape(shape_arg) if shape_arg else None

    device_path = dataset_root / "meta" / "devices" / "dmtac_gripper.json"
    if device_path.exists():
        device = json.loads(device_path.read_text())
        side_config = device.get(side) or {}
        if not sensors:
            sensors = tuple(str(sensor) for sensor in side_config.get("tactile_ids", ()))
        if sensor_shape is None and side_config.get("tactile_saved_shape_per_sensor"):
            sensor_shape = tuple(int(dim) for dim in side_config["tactile_saved_shape_per_sensor"])

    if not sensors:
        episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
        metadata = episodes[0].get("metadata", {}) if episodes else {}
        sensors = tuple(str(sensor) for sensor in metadata.get("gripper_tactile_ids", ()))
    if sensor_shape is None:
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        for feature in info.get("features", {}).values():
            if isinstance(feature, dict) and feature.get("shape") == [144, 192, 3]:
                sensor_shape = (144, 192, 3)
                break
        if sensor_shape is None:
            sensor_shape = (144, 192, 3)

    if not sensors:
        raise ValueError("Could not infer tactile sensor ids. Pass --sensors=L25480055,L25480053 explicitly.")
    if len(sensor_shape) != 3:
        raise ValueError(f"Expected 3-D tactile sensor shape, got {sensor_shape}.")
    return TactileLayout(side=side, sensors=sensors, sensor_shape=sensor_shape)


def _regions_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in list(value))


def _is_complete_frame(regions: tuple[str, ...], tactile: Any, layout: TactileLayout) -> bool:
    if regions != layout.sensors:
        return False
    arr = np.asarray(tactile, dtype=np.float32).reshape(-1)
    return arr.shape == (layout.full_dim,) and bool(np.all(np.isfinite(arr)))


def _extract_sensor_image(
    tactile: Any,
    regions: tuple[str, ...],
    sensor_id: str,
    layout: TactileLayout,
) -> np.ndarray:
    if sensor_id not in regions:
        raise KeyError(sensor_id)
    arr = np.asarray(tactile, dtype=np.float32).reshape(-1)
    expected_dim = len(regions) * layout.per_sensor_dim
    if arr.shape != (expected_dim,):
        raise ValueError(f"Expected tactile dim {expected_dim} for regions {regions}, got {arr.shape}.")
    sensor_index = regions.index(sensor_id)
    start = sensor_index * layout.per_sensor_dim
    end = start + layout.per_sensor_dim
    image = arr[start:end].reshape(layout.sensor_shape)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Non-finite tactile values for sensor {sensor_id}.")
    return image


def _find_neighbor_complete_frame(
    frame_index: int,
    *,
    direction: int,
    regions: list[tuple[str, ...]],
    tactile_values: list[Any],
    layout: TactileLayout,
) -> int | None:
    idx = frame_index + direction
    while 0 <= idx < len(regions):
        if _is_complete_frame(regions[idx], tactile_values[idx], layout):
            return idx
        idx += direction
    return None


def _tactile_dict_json(layout: TactileLayout) -> str:
    return json.dumps(
        {sensor: {"dtype": "float32", "shape": list(layout.sensor_shape)} for sensor in layout.sensors},
        sort_keys=True,
    )


def _scan_bad_rows(regions: list[tuple[str, ...]], layout: TactileLayout) -> list[int]:
    expected = set(layout.sensors)
    bad_rows = []
    for idx, row_regions in enumerate(regions):
        if row_regions == layout.sensors:
            continue
        unknown = sorted(set(row_regions) - expected)
        if unknown:
            raise ValueError(f"Unknown tactile sensor ids at row {idx}: {unknown}; expected {layout.sensors}.")
        if len(set(row_regions)) != len(row_regions):
            raise ValueError(f"Duplicate tactile sensor ids at row {idx}: {row_regions}.")
        bad_rows.append(idx)
    return bad_rows


def _patch_episode_table(
    table: Any,
    *,
    episode_index: int,
    parquet_path: Path,
    layout: TactileLayout,
) -> tuple[Any, list[FramePatch]]:
    pa, _ = _require_pyarrow()

    tactile_idx = table.schema.get_field_index(layout.tactile_column)
    regions_idx = table.schema.get_field_index(layout.regions_column)
    dict_idx = table.schema.get_field_index(layout.dict_column)
    if min(tactile_idx, regions_idx, dict_idx) < 0:
        missing = [
            column
            for column, idx in (
                (layout.tactile_column, tactile_idx),
                (layout.regions_column, regions_idx),
                (layout.dict_column, dict_idx),
            )
            if idx < 0
        ]
        raise KeyError(f"Missing required parquet columns in {parquet_path}: {missing}")

    tactile_values = table[layout.tactile_column].to_pylist()
    regions = [_regions_tuple(value) for value in table[layout.regions_column].to_pylist()]
    dict_values = table[layout.dict_column].to_pylist()
    bad_rows = _scan_bad_rows(regions, layout)
    if not bad_rows:
        return table, []

    new_tactile_values = list(tactile_values)
    new_regions = [list(region) for region in regions]
    new_dict_values = list(dict_values)
    patches: list[FramePatch] = []

    for frame_index in bad_rows:
        row_regions = regions[frame_index]
        missing_sensors = tuple(sensor for sensor in layout.sensors if sensor not in row_regions)
        previous_frame = _find_neighbor_complete_frame(
            frame_index,
            direction=-1,
            regions=regions,
            tactile_values=tactile_values,
            layout=layout,
        )
        next_frame = _find_neighbor_complete_frame(
            frame_index,
            direction=1,
            regions=regions,
            tactile_values=tactile_values,
            layout=layout,
        )
        if previous_frame is None or next_frame is None:
            raise ValueError(
                f"Cannot patch episode {episode_index} frame {frame_index}: "
                f"missing sensors {missing_sensors}, previous={previous_frame}, next={next_frame}."
            )

        sensor_images: dict[str, np.ndarray] = {}
        for sensor in layout.sensors:
            if sensor in row_regions:
                sensor_images[sensor] = _extract_sensor_image(
                    tactile_values[frame_index],
                    row_regions,
                    sensor,
                    layout,
                )
                continue

            prev_image = _extract_sensor_image(tactile_values[previous_frame], regions[previous_frame], sensor, layout)
            next_image = _extract_sensor_image(tactile_values[next_frame], regions[next_frame], sensor, layout)
            sensor_images[sensor] = ((prev_image + next_image) * 0.5).astype(np.float32)

        new_vector = np.concatenate([sensor_images[sensor].reshape(-1) for sensor in layout.sensors], axis=0)
        new_tactile_values[frame_index] = new_vector.astype(np.float32).tolist()
        new_regions[frame_index] = list(layout.sensors)
        new_dict_values[frame_index] = _tactile_dict_json(layout)
        patches.append(
            FramePatch(
                episode_index=episode_index,
                parquet_path=parquet_path,
                frame_index=frame_index,
                old_regions=row_regions,
                missing_sensors=missing_sensors,
                previous_frame=previous_frame,
                next_frame=next_frame,
            )
        )

    table = table.set_column(
        tactile_idx,
        layout.tactile_column,
        pa.array(new_tactile_values, type=table.schema.field(tactile_idx).type),
    )
    table = table.set_column(
        regions_idx,
        layout.regions_column,
        pa.array(new_regions, type=table.schema.field(regions_idx).type),
    )
    table = table.set_column(
        dict_idx,
        layout.dict_column,
        pa.array(new_dict_values, type=table.schema.field(dict_idx).type),
    )
    return table, patches


def _write_table_in_place(
    table: Any,
    parquet_path: Path,
    *,
    compression: str,
    backup: bool,
    backup_suffix: str,
    overwrite_backup: bool,
) -> None:
    _, pq = _require_pyarrow()
    if backup:
        backup_path = parquet_path.with_name(parquet_path.name + backup_suffix)
        if backup_path.exists() and not overwrite_backup:
            raise FileExistsError(
                f"Backup already exists: {backup_path}. Pass --overwrite-backup to replace it, "
                "or remove it after verifying it is safe."
            )
        shutil.copy2(parquet_path, backup_path)

    fd, tmp_name = tempfile.mkstemp(prefix=parquet_path.name + ".", suffix=".tmp", dir=str(parquet_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path, compression=compression)
        os.replace(tmp_path, parquet_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def patch_dataset(args: argparse.Namespace) -> list[FramePatch]:
    _, pq = _require_pyarrow()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    layout = _load_layout(dataset_root, args.side, args.sensors, args.sensor_shape)
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")

    print(f"Dataset: {dataset_root}")
    print(f"Side: {layout.side}")
    print(f"Expected sensors: {layout.sensors}")
    print(f"Sensor shape: {layout.sensor_shape}, per sensor dim={layout.per_sensor_dim}, full dim={layout.full_dim}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    all_patches: list[FramePatch] = []
    patched_files = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if args.episodes and episode_index not in args.episodes:
            continue
        relative_path = episode.get("parquet_path")
        if not relative_path:
            continue
        parquet_path = dataset_root / str(relative_path)

        # Read only the small regions column first. Most episodes are already clean.
        regions_table = pq.read_table(parquet_path, columns=[layout.regions_column])
        regions = [_regions_tuple(value) for value in regions_table[layout.regions_column].to_pylist()]
        bad_rows = _scan_bad_rows(regions, layout)
        if not bad_rows:
            continue

        table = pq.read_table(parquet_path)
        patched_table, patches = _patch_episode_table(
            table,
            episode_index=episode_index,
            parquet_path=parquet_path,
            layout=layout,
        )
        if not patches:
            continue
        patched_files += 1
        all_patches.extend(patches)
        for patch in patches:
            print(
                "patch "
                f"episode={patch.episode_index} frame={patch.frame_index} "
                f"old_regions={patch.old_regions} missing={patch.missing_sensors} "
                f"prev={patch.previous_frame} next={patch.next_frame}"
            )
        if args.apply:
            _write_table_in_place(
                patched_table,
                parquet_path,
                compression=args.parquet_compression,
                backup=not args.no_backup,
                backup_suffix=args.backup_suffix,
                overwrite_backup=args.overwrite_backup,
            )

    print(f"Patched frames: {len(all_patches)}")
    print(f"Affected parquet files: {patched_files}")
    if not args.apply and all_patches:
        print("Dry-run only. Re-run with --apply to rewrite parquet files.")
    return all_patches


def _parse_episode_filter(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--side", type=str, default="right")
    parser.add_argument("--sensors", type=str, default=None, help="Comma-separated sensor order, e.g. L25480055,L25480053")
    parser.add_argument("--sensor-shape", type=str, default=None, help="Per-sensor tactile image shape as H,W,C")
    parser.add_argument("--episodes", type=_parse_episode_filter, default=None, help="Optional filter, e.g. 49,59,71")
    parser.add_argument("--apply", action="store_true", help="Rewrite affected parquet files in place.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backups before rewriting.")
    parser.add_argument("--backup-suffix", type=str, default=".bak")
    parser.add_argument("--overwrite-backup", action="store_true")
    parser.add_argument("--parquet-compression", type=str, default="zstd")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patch_dataset(args)


if __name__ == "__main__":
    main()
