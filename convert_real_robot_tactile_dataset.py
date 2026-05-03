#!/usr/bin/env python3
"""Convert real-robot Inspire tactile datasets to the 351D training format.

The converter keeps the LeRobot-style directory layout, copies metadata/videos,
rewrites parquet files under data/, converts Inspire 1062D tactile vectors to
the 351D TacGlove/UV union representation, and injects a fixed task language.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


UV_CONVERSION_ROOT = Path("/home/ai/Desktop/Being-H-series/UV_conversion")
if str(UV_CONVERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(UV_CONVERSION_ROOT))

from uv_conversion.universal_UV_converter import UniTacConverter  # noqa: E402
try:
    from uv_conversion.universal_UV_converter import UNION_IDX  # noqa: E402
except Exception:  # noqa: BLE001
    UNION_IDX = None
from uv_conversion.universal_UV_converter import HandType  # noqa: E402


DEFAULT_TASK = "Pick and place the potato chips."
DEFAULT_TASK_ID = "chips"
TACTILE_COLUMNS = (
    ("observation.hand.right.tactile", HandType.RIGHT),
    ("observation.hand.left.tactile", HandType.LEFT),
)
SANITIZE_VECTOR_COLUMNS = (
    ("observation.right_arm.pose", ("observation.arm.right.current_tcp_pose", "action.right_arm.pose")),
    ("observation.left_arm.pose", ("observation.arm.left.current_tcp_pose", "action.left_arm.pose")),
    ("observation.right_hand.qpos", ("observation.hand.right.joint_pos", "action.right_hand.joint_target")),
    ("observation.left_hand.qpos", ("observation.hand.left.joint_pos", "action.left_hand.joint_target")),
    ("action.right_arm.pose", ("teleop.action_source.right_wrist_pose_base", "observation.right_arm.pose", "observation.arm.right.current_tcp_pose")),
    ("action.left_arm.pose", ("teleop.action_source.left_wrist_pose_base", "observation.left_arm.pose", "observation.arm.left.current_tcp_pose")),
    ("action.right_hand.joint_target", ("observation.right_hand.qpos", "observation.hand.right.joint_pos")),
    ("action.left_hand.joint_target", ("observation.left_hand.qpos", "observation.hand.left.joint_pos")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a RealRobot chips-style dataset from 1062D Inspire tactile to 351D tactile."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/share/being-dex/BeingH-Tac-Datasets/RealRobot/2026-04-27_chips_single"),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to '<input>_converted'.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Language annotation written to all trajectories.",
    )
    parser.add_argument(
        "--task-id",
        default=DEFAULT_TASK_ID,
        help="Task id written to meta/tasks.jsonl and meta/info.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before conversion if it already exists.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N parquet episodes. Useful for lightweight validation.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy", "symlink", "skip"),
        default="hardlink",
        help="How to copy video files. hardlink is space-efficient and keeps paths valid.",
    )
    return parser.parse_args()


def get_union_idx(converter: UniTacConverter) -> np.ndarray:
    if UNION_IDX is not None:
        union_idx = np.asarray(UNION_IDX, dtype=np.int64)
    else:
        valid = []
        for hand_type in (HandType.RIGHT, HandType.LEFT):
            uv891 = converter.glove137_to_uv891(np.ones(137, dtype=np.float32), hand_type=hand_type, mask=True)
            valid.append(np.where(np.asarray(uv891).reshape(-1) != -1)[0])
        union_idx = np.union1d(valid[0], valid[1]).astype(np.int64)
    if union_idx.shape[0] != 351:
        raise RuntimeError(f"Expected 351 union UV vertices, got {union_idx.shape[0]}")
    return union_idx


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "skip":
        return
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def copy_static_tree(src_root: Path, dst_root: Path, video_mode: str) -> None:
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        if rel.parts and rel.parts[0] == "data":
            continue
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if rel.as_posix() == "meta/stats.json":
            continue
        if rel.parts and rel.parts[0] == "videos":
            copy_file(src, dst, video_mode)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def convert_tactile_series(
    series: pd.Series,
    converter: UniTacConverter,
    union_idx: np.ndarray,
    hand_type: HandType,
) -> tuple[pd.Series, pd.Series]:
    values = series.to_numpy()
    converted: list[np.ndarray] = [np.zeros(351, dtype=np.float32) for _ in range(len(values))]
    masks: list[np.ndarray] = [np.zeros(351, dtype=bool) for _ in range(len(values))]

    batch_indices = []
    batch_raw = []
    for row_idx, item in enumerate(values):
        if item is None:
            continue
        arr = np.asarray(item, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 351:
            valid = np.isfinite(arr) & (arr != -1.0)
            converted[row_idx] = np.where(valid, np.clip(arr, 0.0, 1.0), 0.0).astype(np.float32)
            masks[row_idx] = valid.astype(bool)
        elif arr.shape[0] == 1062:
            batch_indices.append(row_idx)
            batch_raw.append(arr)
        else:
            raise ValueError(f"Expected tactile vector length 1062 or 351, got {arr.shape[0]}")

    if batch_raw:
        safe = np.nan_to_num(np.stack(batch_raw), nan=0.0, posinf=0.0, neginf=0.0)
        uv891 = converter.inspire1062_to_uv891(safe, hand_type=hand_type, correct=True, mask=True)
        tactile351_raw = np.asarray(uv891, dtype=np.float32)[..., union_idx]
        valid = np.isfinite(tactile351_raw) & (tactile351_raw != -1.0)
        tactile351 = np.where(valid, np.clip(tactile351_raw, 0.0, 1.0), 0.0).astype(np.float32)
        for out_idx, row_idx in enumerate(batch_indices):
            converted[row_idx] = tactile351[out_idx]
            masks[row_idx] = valid[out_idx].astype(bool)

    return pd.Series(converted, index=series.index), pd.Series(masks, index=series.index)


def _as_finite_vector(item, width: int | None = None, dtype: np.dtype | type | None = None) -> np.ndarray | None:
    if item is None:
        return None
    arr = np.asarray(item).reshape(-1)
    if width is not None and arr.shape[0] != width:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    if not np.isfinite(arr.astype(np.float64, copy=False)).all():
        return None
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _repair_vector_column(df: pd.DataFrame, target_col: str, fallback_cols: tuple[str, ...]) -> int:
    if target_col not in df.columns:
        return 0

    original = df[target_col].to_numpy()
    width = None
    output_dtype = np.float32
    for item in original:
        vec = _as_finite_vector(item)
        if vec is not None:
            width = int(vec.shape[0])
            output_dtype = vec.dtype
            break
    if width is None:
        for fallback_col in fallback_cols:
            if fallback_col not in df.columns:
                continue
            for item in df[fallback_col].to_numpy():
                vec = _as_finite_vector(item)
                if vec is not None:
                    width = int(vec.shape[0])
                    output_dtype = vec.dtype
                    break
            if width is not None:
                break
    if width is None:
        return 0

    fallback_values = {
        col: df[col].to_numpy()
        for col in fallback_cols
        if col in df.columns
    }

    repaired: list[object] = list(original)
    changed = 0
    previous: np.ndarray | None = None
    invalid_indices: list[int] = []

    for row_idx, item in enumerate(original):
        current = _as_finite_vector(item, width=width, dtype=output_dtype)
        if current is not None:
            previous = current
            continue

        replacement = None
        for values in fallback_values.values():
            replacement = _as_finite_vector(values[row_idx], width=width, dtype=output_dtype)
            if replacement is not None:
                break
        if replacement is None and previous is not None:
            replacement = previous

        if replacement is None:
            invalid_indices.append(row_idx)
            continue

        repaired[row_idx] = replacement.astype(output_dtype)
        previous = replacement
        changed += 1

    next_valid: np.ndarray | None = None
    for row_idx in range(len(repaired) - 1, -1, -1):
        current = _as_finite_vector(repaired[row_idx], width=width, dtype=output_dtype)
        if current is not None:
            next_valid = current
            continue
        if row_idx not in invalid_indices or next_valid is None:
            continue
        repaired[row_idx] = next_valid.astype(output_dtype)
        changed += 1

    if changed:
        df[target_col] = pd.Series(repaired, index=df.index)
    return changed


def sanitize_numeric_vectors(df: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for target_col, fallback_cols in SANITIZE_VECTOR_COLUMNS:
        count = _repair_vector_column(df, target_col, fallback_cols)
        if count:
            counts[target_col] = count
    return counts


def iter_parquets(root: Path) -> Iterable[Path]:
    return sorted((root / "data").glob("**/*.parquet"))


def _filter_jsonl_by_episode(src_path: Path, dst_path: Path, kept_episode_indices: set[int]) -> None:
    if not src_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r") as src, dst_path.open("w") as dst:
        for line in src:
            item = json.loads(line)
            if item.get("episode_index") in kept_episode_indices:
                dst.write(json.dumps(item) + "\n")


def update_metadata(
    dst_root: Path,
    task: str,
    task_id: str,
    kept_episode_indices: set[int] | None = None,
) -> None:
    tasks_path = dst_root / "meta" / "tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task, "task_id": task_id, "description": task}) + "\n")

    info_path = dst_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info["dataset_name"] = dst_root.name
        info["task_name"] = task
        info["task_id"] = task_id
        info["task_description"] = task
        info["task_index"] = 0
        info["tasks"] = [task]
        features = info.setdefault("features", {})
        for col, _ in TACTILE_COLUMNS:
            features[col] = {"shape": [351], "dtype": "float32"}
            features[f"{col}_mask"] = {"shape": [351], "dtype": "bool"}
        features["task_index"] = {"shape": [1], "dtype": "int64"}
        if kept_episode_indices is not None:
            episodes_path = dst_root / "meta" / "episodes.jsonl"
            frame_count = 0
            episode_count = 0
            if episodes_path.exists():
                with episodes_path.open("r") as f:
                    for line in f:
                        episode = json.loads(line)
                        episode_count += 1
                        frame_count += int(episode.get("length", episode.get("frame_count", 0)))
            info["episode_count"] = episode_count
            info["frame_count"] = frame_count
        info_path.write_text(json.dumps(info, indent=2) + "\n")


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    task: str,
    task_id: str,
    overwrite: bool,
    video_mode: str,
    max_episodes: int | None = None,
) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {src_root}")
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Copying static files from {src_root} to {dst_root} (video_mode={video_mode})")
    copy_static_tree(src_root, dst_root, video_mode=video_mode)

    converter = UniTacConverter(return_union=False)
    union_idx = get_union_idx(converter)

    parquet_paths = list(iter_parquets(src_root))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")
    if max_episodes is not None:
        parquet_paths = parquet_paths[:max_episodes]

    kept_episode_indices = {int(path.stem.split("_")[-1]) for path in parquet_paths}
    _filter_jsonl_by_episode(src_root / "meta" / "episodes.jsonl", dst_root / "meta" / "episodes.jsonl", kept_episode_indices)
    _filter_jsonl_by_episode(src_root / "meta" / "episodes_stats.jsonl", dst_root / "meta" / "episodes_stats.jsonl", kept_episode_indices)

    for src_parquet in tqdm(parquet_paths, desc="Converting parquet"):
        rel = src_parquet.relative_to(src_root)
        dst_parquet = dst_root / rel
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(src_parquet)
        for col, hand_type in TACTILE_COLUMNS:
            if col in df.columns:
                tactile_values, tactile_masks = convert_tactile_series(df[col], converter, union_idx, hand_type)
                df[col] = tactile_values
                df[f"{col}_mask"] = tactile_masks
        sanitize_counts = sanitize_numeric_vectors(df)
        if sanitize_counts:
            print(f"Sanitized non-finite vectors in {src_parquet.name}: {sanitize_counts}")
        df["task_index"] = np.zeros(len(df), dtype=np.int64)
        df.to_parquet(dst_parquet, index=False)

    update_metadata(dst_root, task, task_id=task_id, kept_episode_indices=kept_episode_indices)
    print(f"Converted dataset written to {dst_root}")


def main() -> None:
    args = parse_args()
    output = args.output if args.output is not None else args.input.with_name(args.input.name + "_converted")
    convert_dataset(
        src_root=args.input,
        dst_root=output,
        task=args.task,
        task_id=args.task_id,
        overwrite=args.overwrite,
        video_mode=args.video_mode,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
