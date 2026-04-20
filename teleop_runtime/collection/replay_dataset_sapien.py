from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.pipeline import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
from rm65_inspire_tac.rendering import SapienArmHandRenderer

from ..configs import ANNOTATED_DUAL_RM65_BASE_DISTANCE_M
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
    inspire_q6_to_render_qpos,
    open_hand_render_qpos,
)
from .schema_v21 import camera_video_channel


Row = Mapping[str, Any]
EpisodeRecord = Mapping[str, Any]


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a LeRobot-style teleop collection episode in SAPIEN and print "
            "left/right tactile sums from the recorded parquet rows."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Collection dataset root containing meta/episodes.jsonl and data/chunk-*/episode_*.parquet.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Episode index, episode id such as episode_000001, or a parquet path.",
    )
    parser.add_argument(
        "--arm-source",
        choices=["action", "observation"],
        default="action",
        help=(
            "Joint source for arm replay. action uses action.replay.*_arm_joint_target; "
            "observation uses observation.arm.*.current_qpos."
        ),
    )
    parser.add_argument(
        "--hand-source",
        choices=["action", "observation", "teleop", "open"],
        default="action",
        help=(
            "Joint source for hand rendering. teleop uses teleop.action_source.*_hand_q6 "
            "and applies the default collection inversion before rendering."
        ),
    )
    parser.add_argument("--speed", type=_positive_float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--max-frames",
        type=_non_negative_int,
        default=0,
        help="Stop after N frames. 0 means replay the full episode.",
    )
    parser.add_argument(
        "--print-every",
        type=_positive_int,
        default=1,
        help="Print tactile sums every N replayed frames.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Do not preserve recorded timestamps; iterate as fast as possible.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window.")
    parser.add_argument(
        "--skip-renderer",
        action="store_true",
        help="Skip SAPIEN entirely and only read parquet / print tactile sums.",
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=None,
        help="Override SAPIEN left/right RM65 base distance in meters.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["auto", "window", "off"],
        default="auto",
        help=(
            "Collected-video playback mode. auto opens OpenCV windows only when videos exist "
            "and a display is available."
        ),
    )
    parser.add_argument(
        "--video-camera-key",
        action="append",
        default=[],
        help="Only replay this camera key. Repeat for multiple keys. Default: all recorded videos.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _parse_episode_index(token: str) -> int:
    value = str(token).strip()
    if value.startswith("episode_"):
        value = value.removeprefix("episode_")
    return int(value)


def _resolve_episode_parquet(
    dataset_root: Path,
    episode_token: str,
) -> tuple[Path, EpisodeRecord | None]:
    explicit = Path(episode_token).expanduser()
    if explicit.exists():
        return explicit.resolve(), None
    if not explicit.is_absolute():
        candidate = (dataset_root / explicit).resolve()
        if candidate.exists():
            return candidate, None

    try:
        episode_index = _parse_episode_index(episode_token)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Episode '{episode_token}' is neither an existing parquet path nor an integer episode index."
        ) from exc

    episode_id = f"episode_{episode_index:06d}"
    records = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    matches = [
        record
        for record in records
        if int(record.get("episode_index", -1)) == episode_index
        or str(record.get("episode_id", "")) == episode_id
    ]
    if matches:
        record = matches[-1]
        parquet_path = record.get("parquet_path")
        if not parquet_path:
            raise FileNotFoundError(f"Episode record for {episode_id} has no parquet_path.")
        return (dataset_root / str(parquet_path)).resolve(), record

    candidates = sorted(dataset_root.glob(f"data/chunk-*/{episode_id}.parquet"))
    if candidates:
        return candidates[-1].resolve(), None
    raise FileNotFoundError(f"Could not find parquet for {episode_id} under {dataset_root}.")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Dataset replay requires pyarrow. Install pyarrow in the active conda environment."
        ) from exc

    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows


def _finite_vector(row: Row, key: str, *, length: int | None = None) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if length is not None and array.shape != (int(length),):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array.astype(float, copy=False)


def _first_vector(row: Row, keys: list[str], *, length: int) -> np.ndarray | None:
    for key in keys:
        value = _finite_vector(row, key, length=length)
        if value is not None:
            return value
    return None


def _default_arm_qpos(side: str) -> np.ndarray:
    if side == "left":
        return DEFAULT_LEFT_ARM_QPOS.astype(float, copy=True)
    if side == "right":
        return DEFAULT_RIGHT_ARM_QPOS.astype(float, copy=True)
    raise ValueError(f"Unsupported side: {side}")


def _arm_qpos(row: Row, side: str, source: str) -> np.ndarray:
    action_keys = [
        f"action.replay.{side}_arm_joint_target",
        f"observation.arm.{side}.last_target_qpos",
    ]
    observation_keys = [
        f"observation.arm.{side}.current_qpos",
        f"observation.{side}_arm.qpos",
    ]
    ordered_keys = action_keys + observation_keys if source == "action" else observation_keys + action_keys
    qpos = _first_vector(row, ordered_keys, length=6)
    if qpos is None:
        return _default_arm_qpos(side)
    return qpos


def _hand_q6(row: Row, side: str, source: str) -> np.ndarray | None:
    if source == "open":
        return None
    if source == "teleop":
        source_q6 = _finite_vector(row, f"teleop.action_source.{side}_hand_q6", length=6)
        if source_q6 is not None:
            return 1.0 - np.clip(source_q6, 0.0, 1.0)

    action_keys = [f"action.replay.{side}_hand_joint_target"]
    observation_keys = [
        f"observation.hand.{side}.joint_pos",
        f"observation.{side}_hand.qpos",
    ]
    if source == "observation":
        ordered_keys = observation_keys + action_keys
    else:
        ordered_keys = action_keys + observation_keys
    return _first_vector(row, ordered_keys, length=6)


def _hand_render_qpos(row: Row, side: str, source: str) -> np.ndarray:
    q6 = _hand_q6(row, side, source)
    if q6 is None:
        return open_hand_render_qpos(side)
    return inspire_q6_to_render_qpos(np.clip(q6, 0.0, 1.0))


def _tactile_sum(row: Row, side: str) -> float | None:
    tactile = _finite_vector(row, f"observation.hand.{side}.tactile")
    if tactile is None:
        return None
    return float(np.sum(tactile))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.3f}"


def _row_timestamp(row: Row) -> float | None:
    value = row.get("timestamp")
    if value is None:
        return None
    try:
        timestamp = float(value)
    except Exception:
        return None
    if not np.isfinite(timestamp):
        return None
    return timestamp


def _relative_time_sec(
    row: Row,
    *,
    first_timestamp: float | None,
    frame_offset: int,
    fallback_hz: float,
) -> float:
    timestamp = _row_timestamp(row)
    if timestamp is not None and first_timestamp is not None:
        return max(0.0, timestamp - first_timestamp)
    return float(frame_offset) / float(fallback_hz)


def _nested_get(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number


def _base_distance_from_record(record: EpisodeRecord | None) -> float:
    if record is None:
        return ANNOTATED_DUAL_RM65_BASE_DISTANCE_M
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return ANNOTATED_DUAL_RM65_BASE_DISTANCE_M
    candidates = [
        metadata.get("left_right_base_distance_m"),
        metadata.get("base_distance_m"),
        _nested_get(metadata, ("calibration_snapshot", "metadata", "base_distance_m")),
        _nested_get(metadata, ("calibration_snapshot", "arms", "left_arm", "metadata", "base_distance_m")),
        _nested_get(metadata, ("calibration_snapshot", "arms", "right_arm", "metadata", "base_distance_m")),
    ]
    for candidate in candidates:
        value = _float_or_none(candidate)
        if value is not None:
            return value
    return ANNOTATED_DUAL_RM65_BASE_DISTANCE_M


def _fallback_hz(record: EpisodeRecord | None) -> float:
    if record is not None:
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            value = _float_or_none(metadata.get("hz") or metadata.get("nominal_fps"))
            if value is not None and value > 0.0:
                return value
    return 30.0


def _build_renderer(args: argparse.Namespace, *, base_distance: float) -> SapienArmHandRenderer | None:
    if args.skip_renderer:
        return None
    try:
        return SapienArmHandRenderer(
            left_urdf_path=require_asset_path("rm65_inspire_tac_left.urdf"),
            left_retargeting_joint_names=LEFT_HAND_RENDER_JOINT_NAMES,
            right_urdf_path=require_asset_path("rm65_inspire_tac_right.urdf"),
            right_retargeting_joint_names=RIGHT_HAND_RENDER_JOINT_NAMES,
            left_right_base_distance=float(base_distance),
            show_viewer=not args.headless,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to create the SAPIEN renderer. Check the render device or rerun with --skip-renderer."
        ) from exc


def _has_display() -> bool:
    if os.name != "posix":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _row_video_path_keys(row: Row) -> dict[str, str]:
    prefix = "observation.images."
    suffix = ".path"
    paths: dict[str, str] = {}
    for key, value in row.items():
        if not key.startswith(prefix) or not key.endswith(suffix) or value is None:
            continue
        camera_key = key[len(prefix) : -len(suffix)]
        if camera_key:
            paths[camera_key] = str(value)
    return paths


def _resolve_video_paths(
    dataset_root: Path,
    episode_record: EpisodeRecord | None,
    rows: list[Row],
) -> dict[str, Path]:
    video_paths: dict[str, str] = {}
    if episode_record is not None and isinstance(episode_record.get("video_paths"), Mapping):
        for camera_key, path_value in episode_record["video_paths"].items():
            if path_value is not None:
                video_paths[str(camera_key)] = str(path_value)
    for row in rows:
        for camera_key, path_value in _row_video_path_keys(row).items():
            video_paths.setdefault(camera_key, path_value)

    resolved: dict[str, Path] = {}
    for camera_key, path_text in video_paths.items():
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        resolved[camera_key] = path.resolve()
    return resolved


def _video_frame_index(row: Row, camera_key: str, fallback: int) -> int | None:
    value = row.get(f"{camera_video_channel(camera_key)}.video_frame_index")
    if value is None:
        return int(fallback)
    try:
        index = int(value)
    except Exception:
        return None
    if index < 0:
        return None
    return index


class OpenCVVideoReplay:
    def __init__(self, video_paths: Mapping[str, Path]):
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Collected-video replay requires opencv-python. Rerun with --video-mode off if needed."
            ) from exc
        self.cv2 = cv2
        self._captures: dict[str, Any] = {}
        self._last_frame_index: dict[str, int] = {}
        for camera_key, path in video_paths.items():
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                print(f"[VideoReplay] skip camera={camera_key}: failed to open {path}")
                continue
            self._captures[str(camera_key)] = capture
        if not self._captures:
            raise RuntimeError("No collected videos could be opened.")

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._captures))

    def show(self, row: Row, *, replay_index: int) -> bool:
        for camera_key, capture in self._captures.items():
            frame_index = _video_frame_index(row, camera_key, replay_index)
            if frame_index is None:
                continue
            last_index = self._last_frame_index.get(camera_key)
            if last_index is None or frame_index != last_index + 1:
                capture.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            self._last_frame_index[camera_key] = int(frame_index)
            self.cv2.imshow(f"collected:{camera_key}", frame)
        key = self.cv2.waitKey(1) & 0xFF
        return key not in {27, ord("q")}

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass


def _build_video_replay(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    episode_record: EpisodeRecord | None,
    rows: list[Row],
) -> OpenCVVideoReplay | None:
    if args.video_mode == "off":
        return None
    video_paths = _resolve_video_paths(dataset_root, episode_record, rows)
    if args.video_camera_key:
        allowed = {str(key) for key in args.video_camera_key}
        video_paths = {
            camera_key: path
            for camera_key, path in video_paths.items()
            if camera_key in allowed
        }
    existing_paths = {
        camera_key: path
        for camera_key, path in video_paths.items()
        if path.exists()
    }
    if not existing_paths:
        if video_paths:
            missing = ", ".join(f"{key}:{path}" for key, path in sorted(video_paths.items()))
            print(f"[VideoReplay] no existing video files found: {missing}")
        return None
    if args.video_mode == "auto" and not _has_display():
        print("[VideoReplay] display not detected; skip collected-video windows in auto mode.")
        return None
    return OpenCVVideoReplay(existing_paths)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    parquet_path, episode_record = _resolve_episode_parquet(dataset_root, args.episode)
    rows = _read_parquet_rows(parquet_path)
    if not rows:
        raise SystemExit(f"Episode parquet has no rows: {parquet_path}")

    max_frames = int(args.max_frames)
    rows_to_replay = rows if max_frames <= 0 else rows[:max_frames]
    first_timestamp = _row_timestamp(rows_to_replay[0])
    fallback_hz = _fallback_hz(episode_record)
    base_distance = (
        float(args.left_right_base_distance)
        if args.left_right_base_distance is not None
        else _base_distance_from_record(episode_record)
    )
    renderer = _build_renderer(args, base_distance=base_distance)
    video_replay = _build_video_replay(
        args,
        dataset_root=dataset_root,
        episode_record=episode_record,
        rows=rows,
    )

    episode_name = parquet_path.stem
    print(
        f"[ReplayDataset] dataset={dataset_root} episode={episode_name} "
        f"rows={len(rows)} replay_rows={len(rows_to_replay)} "
        f"arm_source={args.arm_source} hand_source={args.hand_source} "
        f"base_distance={base_distance:.3f} skip_renderer={bool(args.skip_renderer)} "
        f"video_cameras={list(video_replay.camera_keys) if video_replay is not None else []}"
    )

    left_accum = 0.0
    right_accum = 0.0
    left_count = 0
    right_count = 0
    try:
        replay_start_wall = time.time()
        for replay_index, row in enumerate(rows_to_replay):
            rel_t = _relative_time_sec(
                row,
                first_timestamp=first_timestamp,
                frame_offset=replay_index,
                fallback_hz=fallback_hz,
            )
            if not args.no_sleep:
                target_time = replay_start_wall + rel_t / float(args.speed)
                remaining = target_time - time.time()
                if remaining > 0.0:
                    time.sleep(remaining)

            left_arm_qpos = _arm_qpos(row, "left", args.arm_source)
            right_arm_qpos = _arm_qpos(row, "right", args.arm_source)
            left_hand_qpos = _hand_render_qpos(row, "left", args.hand_source)
            right_hand_qpos = _hand_render_qpos(row, "right", args.hand_source)
            if renderer is not None:
                renderer.render_data(
                    left_qpos_arm=left_arm_qpos,
                    left_qpos_hand=left_hand_qpos,
                    right_qpos_arm=right_arm_qpos,
                    right_qpos_hand=right_hand_qpos,
                    capture_image=False,
                )
            if video_replay is not None and not video_replay.show(row, replay_index=replay_index):
                print("[VideoReplay] stop requested by video window key.")
                break

            left_tactile = _tactile_sum(row, "left")
            right_tactile = _tactile_sum(row, "right")
            if left_tactile is not None:
                left_accum += left_tactile
                left_count += 1
            if right_tactile is not None:
                right_accum += right_tactile
                right_count += 1
            total_tactile = None
            if left_tactile is not None or right_tactile is not None:
                total_tactile = (left_tactile or 0.0) + (right_tactile or 0.0)

            if replay_index % int(args.print_every) == 0:
                frame_index = int(row.get("frame_index", replay_index) or 0)
                print(
                    f"[Tactile] replay_i={replay_index:06d} frame={frame_index:06d} "
                    f"t={rel_t:.3f}s left_sum={_format_optional_float(left_tactile)} "
                    f"right_sum={_format_optional_float(right_tactile)} "
                    f"total={_format_optional_float(total_tactile)}"
                )
    finally:
        if video_replay is not None:
            video_replay.close()

    print(
        f"[ReplayDataset] completed frames={len(rows_to_replay)} "
        f"left_tactile_accum={left_accum:.3f} frames_with_left_tactile={left_count} "
        f"right_tactile_accum={right_accum:.3f} frames_with_right_tactile={right_count} "
        f"total_tactile_accum={(left_accum + right_accum):.3f}"
    )


if __name__ == "__main__":
    main()
