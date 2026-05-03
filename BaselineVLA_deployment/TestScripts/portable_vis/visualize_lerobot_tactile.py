#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from common.video_io import PortableVideoWriter


TACTILE_DIM = 137
DEFAULT_FPS = 30.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render tactile-only video(s) from saved LeRobot episode parquet files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python visualization/visualize_lerobot_tactile.py /path/to/dataset_root\n"
            "  python visualization/visualize_lerobot_tactile.py /path/to/dataset_root --episode 12\n"
            "  python visualization/visualize_lerobot_tactile.py /path/to/dataset_root --episode 12 15 18\n"
            "  python visualization/visualize_lerobot_tactile.py /path/to/dataset_root --episode 12,15,18\n"
            "  python visualization/visualize_lerobot_tactile.py /path/to/data/episode_000012.parquet\n"
            "  python visualization/visualize_lerobot_tactile.py /path/to/dataset_root --output_dir /tmp/tactile_renders"
        ),
    )
    parser.add_argument(
        "input_path",
        type=str,
        metavar="INPUT",
        help="Input path: one episode parquet file, a dataset root, or a dataset data/ directory.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="",
        metavar="OUTPUT",
        help=(
            "Optional output video path or output directory.\n"
            "If omitted, save to a visualizations/ directory instead of next to the parquet."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=str,
        default="",
        metavar="DATASET_ROOT",
        help="Advanced override for the dataset root containing data/meta; usually not needed.",
    )
    parser.add_argument(
        "-e",
        "--episode",
        "--episode_index",
        dest="episode_indices",
        nargs="+",
        default=None,
        metavar="EPISODE",
        help=(
            "Episode index or indices to render when INPUT points to a directory.\n"
            "Supports spaces and commas, for example: --episode 12 15 18 or --episode 12,15,18"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-target",
        "--output_target",
        dest="output_target",
        type=str,
        default="",
        metavar="OUTPUT",
        help="Generic output target. Can be a video file path or an output directory.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="",
        metavar="OUTPUT_DIR",
        help="Explicit output directory. The episode filename will be created inside this directory.",
    )
    parser.add_argument(
        "--output-file",
        "--output_file",
        "--output-video",
        "--output_video",
        "--output-path",
        "--output_path",
        dest="output_file",
        type=str,
        default="",
        metavar="OUTPUT_FILE",
        help="Explicit output video file path for a single rendered episode.",
    )
    parser.add_argument(
        "--column_prefix",
        type=str,
        default="tactile",
        choices=["tactile", "tactile_pred"],
        help="Which tactile columns to visualize.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override output FPS. When <= 0, derive from the full saved episode timing if available.",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=640,
        help="Total video width in pixels. The left/right panels split this width evenly.",
    )
    parser.add_argument(
        "--panel_height",
        type=int,
        default=320,
        help="Output video height in pixels.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional maximum number of frames to render; 0 means all frames.",
    )
    parser.add_argument(
        "--frame-interval",
        "--frame_interval",
        dest="frame_interval",
        type=int,
        default=1,
        help="Render every Nth frame while keeping the output FPS unchanged, so videos play faster when N > 1.",
    )
    parser.add_argument(
        "--fill_value",
        type=float,
        default=0.0,
        help="Fill value used by the exporter for invalid tactile frames. Used to infer invalid rows.",
    )
    parser.add_argument(
        "--fill_value_eps",
        type=float,
        default=1e-6,
        help="Tolerance used when matching a saved tactile vector against --fill_value.",
    )
    parser.add_argument(
        "--disable_fill_inference",
        action="store_true",
        help="Treat every saved tactile vector as valid instead of inferring invalid rows from all-fill vectors.",
    )
    parser.add_argument(
        "--lower_percentile",
        type=float,
        default=1.0,
        help="Lower percentile used to derive per-hand visualization limits.",
    )
    parser.add_argument(
        "--upper_percentile",
        type=float,
        default=99.0,
        help="Upper percentile used to derive per-hand visualization limits.",
    )
    return parser.parse_args()


def _parse_episode_indices(raw_values: Optional[Sequence[str]]) -> List[int]:
    if not raw_values:
        return [0]
    episode_indices: List[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            stripped = token.strip()
            if not stripped:
                continue
            value = int(stripped)
            if value not in seen:
                seen.add(value)
                episode_indices.append(value)
    return episode_indices or [0]


def _resolve_episode_file(episode_path_arg: str, dataset_root_arg: str, episode_index: int) -> str:
    direct = Path(os.path.abspath(episode_path_arg))
    if direct.is_file():
        if direct.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a parquet file, got: {direct}")
        return str(direct)

    search_dirs = []
    if dataset_root_arg:
        dataset_root = Path(os.path.abspath(dataset_root_arg))
        if dataset_root.is_dir():
            search_dirs.append(dataset_root)
            data_dir = dataset_root / "data"
            if data_dir.is_dir():
                search_dirs.append(data_dir)

    if direct.is_dir():
        search_dirs.append(direct)
        data_dir = direct / "data"
        if data_dir.is_dir():
            search_dirs.append(data_dir)

    requested_name = f"episode_{int(episode_index):06d}.parquet"
    for search_dir in search_dirs:
        candidate = search_dir / requested_name
        if candidate.is_file():
            return str(candidate)

    available = []
    for search_dir in search_dirs:
        available.extend(sorted(search_dir.glob("episode_*.parquet")))
    if not available:
        raise FileNotFoundError(f"Could not resolve any episode parquet from: {episode_path_arg}")

    preview = ", ".join(path.name for path in available[:8])
    if len(available) > 8:
        preview += ", ..."
    raise FileNotFoundError(
        f"Could not resolve {requested_name} from {episode_path_arg}. Available examples: {preview}"
    )


def _guess_dataset_root(episode_path: str, dataset_root_arg: str) -> Optional[str]:
    if dataset_root_arg:
        return os.path.abspath(dataset_root_arg)
    parent = Path(os.path.abspath(episode_path)).parent
    if parent.name == "data":
        return str(parent.parent)
    return None


def _default_output_filename(episode_path: str, column_prefix: str, episode_index: int) -> str:
    suffix = "_tactile.mp4" if column_prefix == "tactile" else f"_{column_prefix}.mp4"
    path = Path(os.path.abspath(episode_path))
    if path.stem.startswith("episode_"):
        return f"{path.stem}{suffix}"
    return f"episode_{int(episode_index):06d}{suffix}"


def _resolve_output_video_path(
    output_arg: str,
    episode_path: str,
    dataset_root: Optional[str],
    episode_index: int,
    column_prefix: str,
    force_output_dir: bool = False,
    force_output_file: bool = False,
) -> str:
    default_filename = _default_output_filename(episode_path, column_prefix, episode_index)
    if not output_arg:
        if dataset_root:
            return os.path.join(dataset_root, "visualizations", default_filename)
        episode_parent = Path(os.path.abspath(episode_path)).parent
        return str(episode_parent / "visualizations" / default_filename)

    candidate = Path(os.path.abspath(os.path.expanduser(output_arg)))
    if force_output_dir:
        return str(candidate / default_filename)
    if force_output_file:
        return str(candidate)
    if (candidate.exists() and candidate.is_dir()) or not candidate.suffix:
        return str(candidate / default_filename)
    return str(candidate)


def _resolve_cli_output_arg(args: argparse.Namespace) -> tuple[str, str]:
    raw_sources = [
        ("positional OUTPUT", str(getattr(args, "output_path", "") or "").strip(), "auto"),
        ("--output", str(getattr(args, "output_target", "") or "").strip(), "auto"),
        ("--output_dir", str(getattr(args, "output_dir", "") or "").strip(), "dir"),
        ("--output_file", str(getattr(args, "output_file", "") or "").strip(), "file"),
    ]
    sources = [
        (label, os.path.abspath(os.path.expanduser(value)), mode)
        for label, value, mode in raw_sources
        if value
    ]
    if not sources:
        return "", "auto"

    unique_paths: Dict[str, List[tuple[str, str]]] = {}
    for label, path, mode in sources:
        unique_paths.setdefault(path, []).append((label, mode))
    if len(unique_paths) > 1:
        details = ", ".join(f"{label}={path}" for label, path, _ in sources)
        raise ValueError(
            "Please provide only one output target via positional OUTPUT, --output, --output_dir, or --output_file. "
            f"Received: {details}"
        )

    selected_path = next(iter(unique_paths))
    selected_sources = unique_paths[selected_path]
    explicit_modes = {mode for _, mode in selected_sources if mode in {"dir", "file"}}
    if len(explicit_modes) > 1:
        raise ValueError("Please do not use --output_dir and --output_file for the same render command.")
    selected_mode = next(iter(explicit_modes)) if explicit_modes else "auto"
    return selected_path, selected_mode


def _safe_vector(values: Any, dim: int = TACTILE_DIM) -> np.ndarray:
    vector = np.zeros((dim,), dtype=np.float32)
    array = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
    if array.size:
        vector[: min(dim, array.size)] = array[:dim]
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _frame_debug_summary(name: str, value: Any) -> str:
    if not isinstance(value, np.ndarray):
        return f"{name}: type={type(value)!r}"
    summary = (
        f"{name}: type=np.ndarray dtype={value.dtype} shape={value.shape} "
        f"contiguous={bool(value.flags.c_contiguous)}"
    )
    if value.size <= 0:
        return summary + " size=0"
    finite = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 0:
        return summary + " finite=0"
    return summary + f" finite_min={float(finite.min()):.6g} finite_max={float(finite.max()):.6g}"


def _prepare_frame_for_writer(frame: Any, width: int, height: int) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 4:
        array = array[:, :, :3]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected frame shape HxWx3, got {array.shape!r}")
    if array.shape[1] != int(width) or array.shape[0] != int(height):
        raise ValueError(
            f"Expected frame size {(int(height), int(width), 3)!r}, got {array.shape!r}"
        )
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(np.round(array), 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _compute_valid_mask(
    vectors: np.ndarray,
    fill_value: float,
    fill_value_eps: float,
    disable_fill_inference: bool,
) -> np.ndarray:
    if disable_fill_inference:
        return np.ones((vectors.shape[0],), dtype=bool)
    if vectors.size == 0:
        return np.zeros((0,), dtype=bool)
    fill_reference = np.full((vectors.shape[1],), float(fill_value), dtype=np.float32)
    close_to_fill = np.isclose(vectors, fill_reference[None, :], atol=float(fill_value_eps), rtol=0.0)
    return ~np.all(close_to_fill, axis=1)


def _load_episode_arrays(
    episode_path: str,
    column_prefix: str,
    max_frames: int,
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(episode_path)
    row_count = int(parquet_file.metadata.num_rows)
    if row_count <= 0:
        raise RuntimeError(f"Episode parquet is empty: {episode_path}")

    frame_limit = row_count if int(max_frames) <= 0 else min(row_count, int(max_frames))
    schema_names = set(parquet_file.schema_arrow.names)
    left_key = f"{column_prefix}.left_tacglove"
    right_key = f"{column_prefix}.right_tacglove"
    requested_columns = [left_key, right_key]
    optional_columns = [name for name in ("timestamp_ms", "frame_index") if name in schema_names]
    available_columns = [name for name in requested_columns if name in schema_names] + optional_columns
    if not available_columns:
        raise KeyError(
            f"None of the requested tactile columns exist in {episode_path}. Looked for {left_key} and {right_key}."
        )

    table = pq.read_table(episode_path, columns=available_columns).slice(0, frame_limit)
    actual_rows = int(table.num_rows)
    if actual_rows <= 0:
        raise RuntimeError(f"Episode parquet has no rows after applying max_frames: {episode_path}")

    payload: dict[str, Any] = {
        "row_count": actual_rows,
        "frame_indices": np.arange(actual_rows, dtype=np.int64),
        "timestamps_ms": None,
        "left": None,
        "right": None,
    }

    if "frame_index" in table.column_names:
        payload["frame_indices"] = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    if "timestamp_ms" in table.column_names:
        payload["timestamps_ms"] = np.asarray(table.column("timestamp_ms").to_pylist(), dtype=np.float64)

    for hand_side, column_name in (("left", left_key), ("right", right_key)):
        if column_name not in table.column_names:
            continue
        rows = table.column(column_name).to_pylist()
        matrix = np.stack([_safe_vector(row, TACTILE_DIM) for row in rows], axis=0)
        payload[hand_side] = matrix

    if payload["left"] is None and payload["right"] is None:
        raise KeyError(
            f"No tactile columns were available in {episode_path}. Looked for {left_key} and {right_key}."
        )
    return payload


def _load_episode_meta_fps(dataset_root: Optional[str], episode_index: int) -> Optional[float]:
    if not dataset_root:
        return None
    episodes_meta_path = os.path.join(dataset_root, "meta", "episodes.parquet")
    if not os.path.isfile(episodes_meta_path):
        return None
    parquet_file = pq.ParquetFile(episodes_meta_path)
    for batch in parquet_file.iter_batches(columns=["episode_index", "fps"], batch_size=256):
        for row in batch.to_pylist():
            if int(row.get("episode_index", -1)) == int(episode_index):
                try:
                    value = float(row.get("fps"))
                except (TypeError, ValueError):
                    return None
                if np.isfinite(value) and value > 0.0:
                    return value
                return None
    return None


def _load_full_timestamps_ms(episode_path: str) -> Optional[np.ndarray]:
    parquet_file = pq.ParquetFile(episode_path)
    if "timestamp_ms" not in set(parquet_file.schema_arrow.names):
        return None
    table = pq.read_table(episode_path, columns=["timestamp_ms"])
    timestamps_ms = np.asarray(table.column("timestamp_ms").to_pylist(), dtype=np.float64)
    if timestamps_ms.size <= 1:
        return None
    return timestamps_ms


def _fps_from_timestamps(timestamps_ms: Optional[np.ndarray]) -> Optional[float]:
    if timestamps_ms is None or timestamps_ms.size <= 1:
        return None
    duration_seconds = float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0)
    if not np.isfinite(duration_seconds) or duration_seconds <= 1e-9:
        return None
    fps = float((timestamps_ms.size - 1) / duration_seconds)
    if not np.isfinite(fps) or fps <= 1e-9:
        return None
    return fps


def _derive_output_fps(
    episode_path: str,
    preview_timestamps_ms: Optional[np.ndarray],
    dataset_root: Optional[str],
    episode_index: int,
    override_fps: float,
) -> float:
    if float(override_fps) > 0.0:
        return float(override_fps)
    full_fps = _fps_from_timestamps(_load_full_timestamps_ms(episode_path))
    if full_fps is not None:
        return float(full_fps)
    meta_fps = _load_episode_meta_fps(dataset_root, episode_index)
    if meta_fps is not None:
        return float(meta_fps)
    preview_fps = _fps_from_timestamps(preview_timestamps_ms)
    if preview_fps is not None:
        return float(preview_fps)
    return DEFAULT_FPS


def _episode_index_from_path(path: str) -> int:
    stem = Path(path).stem
    if "_" not in stem:
        return 0
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def _build_hand_payload(
    values: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    frame_indices: np.ndarray,
    frame_offset: int,
) -> Optional[Dict[str, Any]]:
    if values is None:
        return None
    is_valid = True
    if valid_mask is not None and frame_offset < valid_mask.size:
        is_valid = bool(valid_mask[frame_offset])
    source_frame_index = frame_offset
    if frame_offset < frame_indices.size:
        source_frame_index = int(frame_indices[frame_offset])
    return {
        "values": values[frame_offset],
        "valid": is_valid,
        "interpolated": False,
        "time_delta_ms": 0.0,
        "source_frame_index": source_frame_index,
    }


def _get_tactile_visualization_functions():
    from common.tactile_visualization import compute_tactile_visualization_limits, render_combined_tactile_panel

    return compute_tactile_visualization_limits, render_combined_tactile_panel


def render_episode_tactile_video(
    episode_path_arg: str,
    *,
    dataset_root: str = "",
    output_video: str = "",
    output_mode: str = "auto",
    column_prefix: str = "tactile",
    fps: float = 0.0,
    panel_width: int = 640,
    panel_height: int = 320,
    max_frames: int = 0,
    frame_interval: int = 1,
    fill_value: float = 0.0,
    fill_value_eps: float = 1e-6,
    disable_fill_inference: bool = False,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    episode_index: int = 0,
    verbose: bool = True,
) -> str:
    compute_tactile_visualization_limits, render_combined_tactile_panel = _get_tactile_visualization_functions()

    episode_path = _resolve_episode_file(episode_path_arg, dataset_root, episode_index)
    resolved_dataset_root = _guess_dataset_root(episode_path, dataset_root)
    resolved_episode_index = _episode_index_from_path(episode_path)
    resolved_output_video = _resolve_output_video_path(
        output_arg=output_video,
        episode_path=episode_path,
        dataset_root=resolved_dataset_root,
        episode_index=resolved_episode_index,
        column_prefix=column_prefix,
        force_output_dir=(str(output_mode) == "dir"),
        force_output_file=(str(output_mode) == "file"),
    )

    episode_arrays = _load_episode_arrays(
        episode_path=episode_path,
        column_prefix=column_prefix,
        max_frames=max(0, int(max_frames)),
    )
    frame_indices = episode_arrays["frame_indices"]
    preview_timestamps_ms = episode_arrays["timestamps_ms"]
    left_values = episode_arrays["left"]
    right_values = episode_arrays["right"]

    left_valid = None if left_values is None else _compute_valid_mask(
        left_values,
        fill_value=float(fill_value),
        fill_value_eps=float(fill_value_eps),
        disable_fill_inference=bool(disable_fill_inference),
    )
    right_valid = None if right_values is None else _compute_valid_mask(
        right_values,
        fill_value=float(fill_value),
        fill_value_eps=float(fill_value_eps),
        disable_fill_inference=bool(disable_fill_inference),
    )

    limits_by_hand: Dict[str, tuple[float, float]] = {}
    if left_values is not None:
        limits_by_hand["left"] = compute_tactile_visualization_limits(
            left_values,
            valid_mask=left_valid,
            lower_percentile=float(lower_percentile),
            upper_percentile=float(upper_percentile),
        )
    if right_values is not None:
        limits_by_hand["right"] = compute_tactile_visualization_limits(
            right_values,
            valid_mask=right_valid,
            lower_percentile=float(lower_percentile),
            upper_percentile=float(upper_percentile),
        )

    resolved_fps = _derive_output_fps(
        episode_path=episode_path,
        preview_timestamps_ms=preview_timestamps_ms,
        dataset_root=resolved_dataset_root,
        episode_index=resolved_episode_index,
        override_fps=float(fps),
    )
    total_frames = int(episode_arrays["row_count"])
    panel_width = max(160, int(panel_width))
    panel_height = max(160, int(panel_height))
    frame_interval = max(1, int(frame_interval))
    frame_offsets = list(range(0, total_frames, frame_interval))

    os.makedirs(os.path.dirname(os.path.abspath(resolved_output_video)) or ".", exist_ok=True)
    writer = PortableVideoWriter(resolved_output_video, float(resolved_fps), (panel_width, panel_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {resolved_output_video}")

    if verbose:
        print(f"Episode parquet: {episode_path}")
        print(f"Output video: {resolved_output_video}")
        print(f"Writer backend: {writer.backend}")
        print(f"Column prefix: {column_prefix}")
        print(f"Frames: {total_frames}")
        print(f"FPS: {resolved_fps:.3f}")
        print(f"Frame interval: {frame_interval}")
        print(f"Rendered frames: {len(frame_offsets)}")
        for hand_side in ("left", "right"):
            if hand_side not in limits_by_hand:
                print(f"{hand_side}: missing column")
                continue
            hand_valid = left_valid if hand_side == "left" else right_valid
            valid_frames = int(np.count_nonzero(hand_valid)) if hand_valid is not None else 0
            print(
                f"{hand_side}: limits=({limits_by_hand[hand_side][0]:.3f}, {limits_by_hand[hand_side][1]:.3f}), "
                f"inferred_valid_frames={valid_frames}/{total_frames}"
            )

    try:
        for frame_offset in frame_offsets:
            tactile_by_hand = {
                "left": _build_hand_payload(left_values, left_valid, frame_indices, frame_offset),
                "right": _build_hand_payload(right_values, right_valid, frame_indices, frame_offset),
            }
            frame = render_combined_tactile_panel(
                tactile_by_hand=tactile_by_hand,
                limits_by_hand=limits_by_hand,
                total_width=panel_width,
                target_height=panel_height,
            )
            try:
                safe_frame = _prepare_frame_for_writer(frame, panel_width, panel_height)
                writer.write(safe_frame)
            except Exception as exc:
                debug_lines = [
                    f"Video write failed at frame_offset={frame_offset}",
                    f"writer_backend={writer.backend}",
                    f"numpy_version={np.__version__}",
                    f"opencv_version={cv2.__version__}",
                    _frame_debug_summary("frame", np.asarray(frame) if isinstance(frame, np.ndarray) else frame),
                ]
                raise RuntimeError("\n".join(debug_lines)) from exc
    finally:
        writer.release()

    if verbose:
        print(f"Saved tactile-only video to {resolved_output_video}")
    return resolved_output_video


def main() -> int:
    args = _parse_args()
    try:
        resolved_output_arg, resolved_output_mode = _resolve_cli_output_arg(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    episode_indices = _parse_episode_indices(args.episode_indices)
    input_path = os.path.abspath(args.input_path)
    if len(episode_indices) > 1 and resolved_output_arg:
        output_candidate = Path(os.path.abspath(os.path.expanduser(resolved_output_arg)))
        output_is_dir = (
            resolved_output_mode == "dir"
            or ((output_candidate.exists() and output_candidate.is_dir()) or not output_candidate.suffix)
        )
        if not output_is_dir:
            raise SystemExit(
                "When rendering multiple episodes, the output target must be a directory. "
                "Use positional OUTPUT as a directory or pass --output_dir."
            )

    if Path(input_path).is_file():
        episode_indices = episode_indices[:1]

    for episode_index in episode_indices:
        render_episode_tactile_video(
            episode_path_arg=input_path,
            dataset_root=args.dataset_root,
            output_video=resolved_output_arg,
            output_mode=resolved_output_mode,
            column_prefix=args.column_prefix,
            fps=float(args.fps),
            panel_width=int(args.panel_width),
            panel_height=int(args.panel_height),
            max_frames=max(0, int(args.max_frames)),
            frame_interval=max(1, int(args.frame_interval)),
            fill_value=float(args.fill_value),
            fill_value_eps=float(args.fill_value_eps),
            disable_fill_inference=bool(args.disable_fill_inference),
            lower_percentile=float(args.lower_percentile),
            upper_percentile=float(args.upper_percentile),
            episode_index=int(episode_index),
            verbose=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
