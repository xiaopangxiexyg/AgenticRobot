"""Legacy camera-only hardware diagnostic, not an offline post-process module.

The camera runtime is loaded only when this CLI executes.  Importing the
offline package does not import or initialize ``teleop_runtime`` or hardware.
New offline code must not use this bridge.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from teleop_runtime.configs.camera_paths import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
from dataset_contracts.wrist_depth import (
    DepthRangeMaskConfig,
    depth_range_mask,
    mask_manifest_payload,
    pack_binary_mask,
    raw_depth_manifest_payload,
)


WRIST_CAMERA_KEYS = ("left_wrist", "right_wrist")
DEFAULT_DEPTH_STREAM_FPS = 30
DIAGNOSTIC_RUNTIME_BRIDGE = True


def _runtime_camera_api() -> tuple[Any, Any]:
    module = importlib.import_module("teleop_runtime.sources.camera")
    return module.RealSenseRGBSource, module.load_realsense_camera_config_map


@dataclass
class CameraSmokeState:
    camera_key: str
    mask_records: list[dict[str, Any]] = field(default_factory=list)
    raw_records: list[dict[str, Any]] = field(default_factory=list)
    last_depth_frame_number: int | None = None
    mask_handle: Any | None = None
    raw_handle: Any | None = None
    shape: tuple[int, int] | None = None
    captured: int = 0
    skipped_no_frame: int = 0
    skipped_no_depth: int = 0
    skipped_duplicate_depth: int = 0
    skipped_rate_limit: int = 0
    valid_fraction_sum: float = 0.0
    foreground_fraction_sum: float = 0.0
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    def close(self) -> None:
        if self.mask_handle is not None:
            self.mask_handle.close()
            self.mask_handle = None
        if self.raw_handle is not None:
            self.raw_handle.close()
            self.raw_handle = None


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Camera-only smoke test for wrist D405 depth mask/raw capture. "
            "This does not initialize arms, hands, grippers, or teleop control."
        )
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_REALSENSE_DEVICE_CONFIG_PATH,
        help=(
            "RealSense camera config. Defaults to the project D405/D435 config; "
            "only left_wrist/right_wrist are opened by this script."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/dexteleop_wrist_depth_smoke"),
        help="Output directory for masks, optional raw depth, and summary JSON.",
    )
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=15.0, help="Target save rate for mask/raw streams.")
    parser.add_argument(
        "--depth-stream-fps",
        type=int,
        default=DEFAULT_DEPTH_STREAM_FPS,
        help="RealSense depth stream fps. The saved mask/raw streams are still rate-limited by --fps.",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.05)
    parser.add_argument("--depth-max-m", type=float, default=0.5)
    parser.add_argument("--save-raw", action="store_true", help="Also save raw aligned z16 depth debug streams.")
    parser.add_argument("--camera", action="append", choices=WRIST_CAMERA_KEYS, default=None)
    parser.add_argument("--wait-timeout-ms", type=int, default=1000)
    parser.add_argument("--stale-after-sec", type=float, default=2.0)
    parser.add_argument(
        "--warmup-timeout-sec",
        type=float,
        default=2.0,
        help="Wait up to this long for each selected camera to publish an initial depth frame before timing capture.",
    )
    parser.add_argument("--no-rgb-video", action="store_true", help="Accepted for clarity; RGB video is never written.")
    args = parser.parse_args()
    if float(args.duration_sec) <= 0.0:
        parser.error("--duration-sec must be positive.")
    if float(args.fps) <= 0.0:
        parser.error("--fps must be positive.")
    if int(args.depth_stream_fps) <= 0:
        parser.error("--depth-stream-fps must be positive.")
    if float(args.warmup_timeout_sec) < 0.0:
        parser.error("--warmup-timeout-sec must be >= 0.")
    if float(args.depth_max_m) <= float(args.depth_min_m):
        parser.error("--depth-max-m must be greater than --depth-min-m.")
    return args


def _selected_configs(args: argparse.Namespace) -> dict[str, Any]:
    _source_type, load_realsense_camera_config_map = _runtime_camera_api()
    configs = load_realsense_camera_config_map(args.camera_config)
    requested = tuple(args.camera or WRIST_CAMERA_KEYS)
    selected: dict[str, Any] = {}
    missing = [key for key in requested if key not in configs]
    if missing:
        raise KeyError(f"missing wrist camera(s) in config: {missing}; available={sorted(configs)}")
    for key in requested:
        config = configs[key]
        config.enable_depth = True
        config.depth_width = int(config.width)
        config.depth_height = int(config.height)
        config.depth_fps = int(args.depth_stream_fps)
        config.align_depth_to_color = True
        config.wait_timeout_ms = int(args.wait_timeout_ms)
        config.stale_after_sec = float(args.stale_after_sec)
        selected[key] = config
    return selected


def _stream_paths(output_root: Path, camera_key: str) -> tuple[Path, Path, Path, Path]:
    camera_root = output_root / camera_key
    mask_stream = camera_root / "mask.bitpack.bin"
    mask_manifest = camera_root / "mask_manifest.json"
    raw_stream = camera_root / "raw_depth.uint16.bin"
    raw_manifest = camera_root / "raw_depth_manifest.json"
    camera_root.mkdir(parents=True, exist_ok=True)
    return mask_stream, mask_manifest, raw_stream, raw_manifest


def _prepare_output_files(output_root: Path, camera_keys: tuple[str, ...], save_raw: bool) -> None:
    (output_root / "summary.json").unlink(missing_ok=True)
    for camera_key in camera_keys:
        mask_stream, mask_manifest, raw_stream, raw_manifest = _stream_paths(output_root, camera_key)
        mask_stream.unlink(missing_ok=True)
        mask_manifest.unlink(missing_ok=True)
        raw_stream.unlink(missing_ok=True)
        raw_manifest.unlink(missing_ok=True)


def _depth_scale(metadata: Mapping[str, Any]) -> float | None:
    try:
        scale = float(metadata.get("depth_scale_m_per_unit"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return float(scale)


def _depth_frame_number(metadata: Mapping[str, Any]) -> int | None:
    try:
        return int(metadata.get("frame_number"))
    except (TypeError, ValueError):
        return None


def _record_common(
    *,
    sample_index: int,
    frame: Any,
    depth_metadata: Mapping[str, Any],
    shape: tuple[int, int],
) -> dict[str, Any]:
    return {
        "sample_frame_index": int(sample_index),
        "source_timestamp": float(frame.source_timestamp),
        "receive_timestamp": frame.receive_timestamp,
        "source_frame_index": frame.frame_index,
        "depth_camera_timestamp_sec": depth_metadata.get("camera_timestamp_sec"),
        "depth_sensor_timestamp_ms": depth_metadata.get("sensor_timestamp_ms"),
        "depth_frame_number": depth_metadata.get("frame_number"),
        "depth_scale_m_per_unit": depth_metadata.get("depth_scale_m_per_unit"),
        "shape": [int(shape[0]), int(shape[1])],
    }


def _write_frame(
    *,
    state: CameraSmokeState,
    frame: Any,
    mask_stream_path: Path,
    raw_stream_path: Path,
    config: DepthRangeMaskConfig,
    save_raw: bool,
) -> None:
    depth = np.asarray(frame.depth_image)
    if depth.ndim != 2:
        state.skipped_no_depth += 1
        return
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    depth = np.ascontiguousarray(depth)
    shape = (int(depth.shape[0]), int(depth.shape[1]))
    if state.shape is None:
        state.shape = shape
    elif state.shape != shape:
        raise ValueError(f"{state.camera_key} depth shape changed from {state.shape} to {shape}")

    metadata = dict(frame.depth_metadata)
    scale = _depth_scale(metadata)
    if scale is None:
        state.skipped_no_depth += 1
        return

    if state.mask_handle is None:
        state.mask_handle = mask_stream_path.open("wb")
    result = depth_range_mask(depth, depth_scale_m_per_unit=scale, config=config)
    state.mask_handle.write(pack_binary_mask(result.mask))
    mask_record = _record_common(
        sample_index=state.captured,
        frame=frame,
        depth_metadata=metadata,
        shape=shape,
    )
    mask_record["mask_frame_index"] = int(state.captured)
    mask_record.update(result.stats_dict())
    state.mask_records.append(mask_record)

    if save_raw:
        if state.raw_handle is None:
            state.raw_handle = raw_stream_path.open("wb")
        state.raw_handle.write(depth.tobytes(order="C"))
        raw_record = _record_common(
            sample_index=state.captured,
            frame=frame,
            depth_metadata=metadata,
            shape=shape,
        )
        raw_record["raw_frame_index"] = int(state.captured)
        state.raw_records.append(raw_record)

    state.valid_fraction_sum += float(result.valid_fraction)
    state.foreground_fraction_sum += float(result.foreground_fraction)
    state.captured += 1
    timestamp = float(frame.source_timestamp)
    if state.first_timestamp is None:
        state.first_timestamp = timestamp
    state.last_timestamp = timestamp


def _write_manifests(
    *,
    output_root: Path,
    states: Mapping[str, CameraSmokeState],
    config: DepthRangeMaskConfig,
    save_raw: bool,
) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for camera_key, state in states.items():
        state.close()
        mask_stream, mask_manifest, raw_stream, raw_manifest = _stream_paths(output_root, camera_key)
        if state.shape is None:
            manifests[camera_key] = {"error": "no depth frames captured"}
            continue
        height, width = state.shape
        mask_payload = mask_manifest_payload(
            camera_key=camera_key,
            relative_dir=str(mask_manifest.parent),
            stream_path=str(mask_stream),
            frame_count=len(state.mask_records),
            height=height,
            width=width,
            records=state.mask_records,
            config=config,
        )
        mask_manifest.write_text(json.dumps(mask_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        camera_payload: dict[str, Any] = {
            "mask_manifest": str(mask_manifest),
            "mask_stream": str(mask_stream),
        }
        if save_raw:
            raw_payload = raw_depth_manifest_payload(
                camera_key=camera_key,
                relative_dir=str(raw_manifest.parent),
                stream_path=str(raw_stream),
                frame_count=len(state.raw_records),
                height=height,
                width=width,
                records=state.raw_records,
            )
            raw_manifest.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=True), encoding="utf-8")
            camera_payload.update(
                {
                    "raw_manifest": str(raw_manifest),
                    "raw_stream": str(raw_stream),
                }
            )
        manifests[camera_key] = camera_payload
    return manifests


def _wait_for_initial_depth_frames(
    sources: Mapping[str, Any],
    timeout_sec: float,
) -> tuple[dict[str, bool], float]:
    started_at = time.time()
    ready = {camera_key: False for camera_key in sources}
    if timeout_sec <= 0.0:
        return ready, 0.0
    deadline = started_at + float(timeout_sec)
    while time.time() < deadline:
        for camera_key, source in sources.items():
            if ready[camera_key]:
                continue
            frame = source.frame_buffer().get_latest()
            ready[camera_key] = bool(frame is not None and frame.depth_image is not None)
        if all(ready.values()):
            break
        time.sleep(0.01)
    return ready, float(time.time() - started_at)


def _compact_printable_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    device_info = summary.get("device_info", {})
    compact_cameras: dict[str, Any] = {}
    for camera_key, camera_summary in summary.get("cameras", {}).items():
        info = device_info.get(camera_key, {}) if isinstance(device_info, Mapping) else {}
        compact_cameras[str(camera_key)] = {
            "captured": camera_summary.get("captured"),
            "shape": camera_summary.get("shape"),
            "approx_saved_hz": camera_summary.get("approx_saved_hz"),
            "valid_fraction_mean": camera_summary.get("valid_fraction_mean"),
            "foreground_fraction_mean": camera_summary.get("foreground_fraction_mean"),
            "skipped_no_frame": camera_summary.get("skipped_no_frame"),
            "skipped_no_depth": camera_summary.get("skipped_no_depth"),
            "skipped_duplicate_depth": camera_summary.get("skipped_duplicate_depth"),
            "device_frame_count": info.get("frame_count"),
            "device_timeout_count": info.get("timeout_count"),
            "device_restart_count": info.get("restart_count"),
            "last_error": info.get("last_error"),
            "last_runtime_error": info.get("last_runtime_error"),
        }
    return {
        "event": "finished",
        "summary_path": str(Path(str(summary.get("output_root", "."))) / "summary.json"),
        "elapsed_sec": summary.get("elapsed_sec"),
        "capture_elapsed_sec": summary.get("capture_elapsed_sec"),
        "warmup_wait_sec": summary.get("warmup_wait_sec"),
        "warmup_ready": summary.get("warmup_ready"),
        "cameras": compact_cameras,
    }


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = DepthRangeMaskConfig(
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
    )
    camera_configs = _selected_configs(args)
    _prepare_output_files(output_root, tuple(camera_configs), save_raw=bool(args.save_raw))
    RealSenseRGBSource, _config_loader = _runtime_camera_api()
    sources = {key: RealSenseRGBSource(cfg) for key, cfg in camera_configs.items()}
    states = {key: CameraSmokeState(camera_key=key) for key in sources}
    print(
        json.dumps(
            {
                "event": "starting_wrist_depth_smoke",
                "output_root": str(output_root),
                "cameras": sorted(sources),
                "duration_sec": float(args.duration_sec),
                "save_fps": float(args.fps),
                "depth_stream_fps": int(args.depth_stream_fps),
                "save_raw": bool(args.save_raw),
                "depth_range": config.to_dict(),
                "note": "arms/grippers/teleop are not initialized by this script",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    process_started_at = time.time()
    capture_started_at: float | None = None
    capture_finished_at: float | None = None
    warmup_ready: dict[str, bool] = {key: False for key in sources}
    warmup_wait_sec = 0.0
    next_save_at = process_started_at
    period_sec = 1.0 / float(args.fps)
    try:
        for source in sources.values():
            source.start()
        warmup_ready, warmup_wait_sec = _wait_for_initial_depth_frames(
            sources,
            timeout_sec=float(args.warmup_timeout_sec),
        )
        capture_started_at = time.time()
        next_save_at = capture_started_at
        deadline = capture_started_at + float(args.duration_sec)
        while time.time() < deadline:
            now = time.time()
            if now < next_save_at:
                time.sleep(min(0.01, max(0.0, next_save_at - now)))
                continue
            next_save_at += period_sec
            for camera_key, source in sources.items():
                state = states[camera_key]
                frame = source.frame_buffer().get_latest()
                if frame is None:
                    state.skipped_no_frame += 1
                    continue
                if frame.depth_image is None:
                    state.skipped_no_depth += 1
                    continue
                frame_number = _depth_frame_number(frame.depth_metadata)
                if frame_number is not None and frame_number == state.last_depth_frame_number:
                    state.skipped_duplicate_depth += 1
                    continue
                state.last_depth_frame_number = frame_number
                mask_stream, _mask_manifest, raw_stream, _raw_manifest = _stream_paths(output_root, camera_key)
                _write_frame(
                    state=state,
                    frame=frame,
                    mask_stream_path=mask_stream,
                    raw_stream_path=raw_stream,
                    config=config,
                    save_raw=bool(args.save_raw),
                )
        capture_finished_at = time.time()
    finally:
        for source in sources.values():
            try:
                source.close()
            except Exception:
                pass

    device_info = {camera_key: source.device_info() for camera_key, source in sources.items()}
    manifests = _write_manifests(
        output_root=output_root,
        states=states,
        config=config,
        save_raw=bool(args.save_raw),
    )
    summary = {
        "format": "dexteleop_wrist_depth_smoke_summary_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.time() - process_started_at),
        "capture_elapsed_sec": (
            None
            if capture_started_at is None
            else float((capture_finished_at or time.time()) - float(capture_started_at))
        ),
        "output_root": str(output_root),
        "camera_config": str(args.camera_config),
        "save_raw": bool(args.save_raw),
        "save_fps": float(args.fps),
        "depth_stream_fps": int(args.depth_stream_fps),
        "warmup_timeout_sec": float(args.warmup_timeout_sec),
        "warmup_wait_sec": float(warmup_wait_sec),
        "warmup_ready": dict(warmup_ready),
        "depth_range": config.to_dict(),
        "device_info": device_info,
        "cameras": {},
        "manifests": manifests,
    }
    for camera_key, state in states.items():
        span = (
            0.0
            if state.first_timestamp is None or state.last_timestamp is None
            else max(0.0, float(state.last_timestamp) - float(state.first_timestamp))
        )
        summary["cameras"][camera_key] = {
            **asdict(state),
            "mask_handle": None,
            "raw_handle": None,
            "shape": None if state.shape is None else list(state.shape),
            "approx_saved_hz": 0.0 if span <= 0.0 else float(max(0, state.captured - 1)) / span,
            "valid_fraction_mean": (
                0.0 if state.captured <= 0 else float(state.valid_fraction_sum) / float(state.captured)
            ),
            "foreground_fraction_mean": (
                0.0 if state.captured <= 0 else float(state.foreground_fraction_sum) / float(state.captured)
            ),
        }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    print(json.dumps(_compact_printable_summary(summary), ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
