"""Read-only status and GUI payload projections for real teleoperation.

The functions in this module accept snapshots, mappings, or already-read
runtime objects and return presentation payloads.  They must not start devices,
refresh hardware, send commands, or publish to a UI.  The legacy entrypoint
binds the one remaining formatting helper explicitly so old threshold and
monkeypatch behavior stays visible at the compatibility boundary.
"""


from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import inspect
import time

import numpy as np

from ..collection.camera_preview import CameraPreviewer
from ..collection.sample_assembler import (
    _build_pose_quality_metadata as _collection_build_pose_quality_metadata,
)
from ..real_teleop.wrist_depth_freshness import (
    WRIST_DEPTH_CAMERA_KEYS,
    WristDepthFreshnessMonitor,
)
from ..sources.camera import RealSenseRGBSource


_SUPPORT_NAMES = ("_command_backend_status_text",)


class _LegacyFunctionProxy:
    """Resolve legacy helpers dynamically so compatibility patches remain live."""

    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(f"recursive status payload support proxy: {self._name}")
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    """Bind the minimal legacy formatting support used by this module."""

    missing = [name for name in _SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "status payload support is incomplete: " + ", ".join(sorted(missing))
        )
    globals().update(
        {
            name: (
                _LegacyFunctionProxy(namespace, name)
                if inspect.isroutine(namespace[name])
                else namespace[name]
            )
            for name in _SUPPORT_NAMES
        }
    )


def _json_safe_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False).reshape(-1).round(4).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value

def _safe_float_value(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result

def _safe_int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _compact_runtime_float(value: object, *, digits: int = 3) -> str:
    numeric = _safe_float_value(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{int(digits)}f}"

def _compact_runtime_bool(value: object) -> str:
    return "1" if bool(value) else "0"

def _as_status_error_text(value: object) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"none", "false"} else ""


def _camera_status_for_source(
    camera_key: str,
    source: RealSenseRGBSource,
    *,
    now: float,
    preview_selected: bool,
    preview_active: bool,
    depth_monitor: WristDepthFreshnessMonitor | None = None,
) -> dict[str, object]:
    config = getattr(source, "config", None)
    fallback_serial = getattr(config, "serial", None)
    try:
        info = source.device_info()
    except Exception as exc:  # noqa: BLE001
        return {
            "camera_key": str(camera_key),
            "status_level": "error",
            "status_label": "状态读取失败",
            "status_message": f"{type(exc).__name__}: {exc}",
            "serial": fallback_serial,
            "preview_selected": bool(preview_selected),
            "preview_active": bool(preview_active and preview_selected),
        }
    if not isinstance(info, Mapping):
        info = {}

    frame_count = _safe_int_value(info.get("frame_count"))
    frame_count = 0 if frame_count is None else max(0, int(frame_count))
    timeout_count = _safe_int_value(info.get("timeout_count"))
    consecutive_timeout_count = _safe_int_value(info.get("consecutive_timeout_count"))
    restart_count = _safe_int_value(info.get("restart_count"))
    restart_attempt_count = _safe_int_value(info.get("restart_attempt_count"))
    restart_failure_count = _safe_int_value(info.get("restart_failure_count"))
    last_frame_age_sec = _safe_float_value(info.get("last_frame_age_sec"))
    stale_after_sec = _safe_float_value(info.get("stale_after_sec"))
    if stale_after_sec is None or stale_after_sec <= 0.0:
        stale_after_sec = 2.0
    last_error = _as_status_error_text(info.get("last_error"))
    last_runtime_error = _as_status_error_text(info.get("last_runtime_error"))
    depth_enabled = bool(info.get("depth_enabled") or getattr(config, "enable_depth", False))
    depth_required = bool(depth_enabled and str(camera_key) in WRIST_DEPTH_CAMERA_KEYS)
    depth_scale = _safe_float_value(info.get("depth_scale_m_per_unit"))
    depth_frame_present: bool | None = None
    depth_status_level = "off"
    depth_status_label = "未启用"
    depth_status_message = ""
    latest_depth_metadata: Mapping[str, object] = {}
    depth_freshness: dict[str, object] = {}
    if depth_required:
        latest_frame = None
        try:
            latest_frame = source.frame_buffer().get_latest()
        except Exception as exc:  # noqa: BLE001
            depth_status_level = "error"
            depth_status_label = "Depth 状态失败"
            depth_status_message = f"{type(exc).__name__}: {exc}"
        else:
            if latest_frame is None:
                depth_status_level = "waiting"
                depth_status_label = "等待 Wrist Depth"
                depth_status_message = "wrist depth 已启用，但还没有 color+depth 同步帧。"
            else:
                raw_depth_metadata = getattr(latest_frame, "depth_metadata", {})
                if isinstance(raw_depth_metadata, Mapping):
                    latest_depth_metadata = raw_depth_metadata
                depth_frame_present = getattr(latest_frame, "depth_image", None) is not None
                frame_depth_scale = _safe_float_value(latest_depth_metadata.get("depth_scale_m_per_unit"))
                if depth_scale is None and frame_depth_scale is not None:
                    depth_scale = frame_depth_scale
                if not depth_frame_present:
                    depth_status_level = "error"
                    depth_status_label = "Wrist Depth 失效"
                    depth_status_message = "最新 wrist 相机帧缺少 depth_image。"
                elif depth_scale is None and frame_depth_scale is None:
                    depth_status_level = "warn"
                    depth_status_label = "Depth scale 缺失"
                    depth_status_message = "depth frame 存在，但缺少有效 depth_scale_m_per_unit，mask 不能可靠生成。"
                else:
                    if depth_monitor is not None:
                        depth_freshness = depth_monitor.update(
                            str(camera_key),
                            latest_frame,
                            now=float(now),
                            depth_metadata=latest_depth_metadata,
                        )
                    freshness_level = str(depth_freshness.get("depth_freshness_status_level") or "ok")
                    freshness_label = str(depth_freshness.get("depth_freshness_status_label") or "")
                    freshness_message = str(depth_freshness.get("depth_freshness_status_message") or "")
                    if freshness_level in {"error", "warn", "waiting"}:
                        depth_status_level = freshness_level
                        depth_status_label = freshness_label or "Wrist Depth 检查中"
                        depth_status_message = freshness_message
                    else:
                        depth_status_level = "ok"
                        depth_status_label = "Wrist Depth 正常"

    status_level = "ok"
    status_label = "正常"
    status_message = ""
    if last_error:
        status_level = "error"
        status_label = "异常"
        status_message = last_error
    elif frame_count <= 0:
        status_level = "waiting"
        status_label = "等待首帧"
        status_message = last_runtime_error
    elif last_frame_age_sec is None:
        status_level = "warn"
        status_label = "缺少时间戳"
    elif last_frame_age_sec >= stale_after_sec:
        status_level = "error"
        status_label = "数据停滞"
        status_message = f"last frame age {last_frame_age_sec:.2f}s >= stale {stale_after_sec:.2f}s"
    elif last_frame_age_sec >= max(0.5, stale_after_sec * 0.5):
        status_level = "warn"
        status_label = "帧延迟"
        status_message = f"last frame age {last_frame_age_sec:.2f}s"
    elif (consecutive_timeout_count or 0) > 0 or last_runtime_error:
        status_level = "warn"
        status_label = "临时 timeout"
        status_message = last_runtime_error
    if depth_required:
        severity_rank = {"off": 0, "ok": 0, "waiting": 1, "warn": 2, "error": 3}
        if severity_rank.get(depth_status_level, 0) > severity_rank.get(status_level, 0):
            status_level = depth_status_level
            status_label = depth_status_label
            status_message = depth_status_message
        elif depth_status_message and depth_status_message not in status_message:
            status_message = (
                f"{status_message}; {depth_status_message}"
                if status_message
                else depth_status_message
            )

    metadata = info.get("metadata") if isinstance(info.get("metadata"), Mapping) else {}
    return {
        "camera_key": str(camera_key),
        "status_level": status_level,
        "status_label": status_label,
        "status_message": status_message,
        "name": info.get("name") or getattr(source, "name", str(camera_key)),
        "serial": info.get("serial") or fallback_serial,
        "serial_number": info.get("serial_number"),
        "device_name": info.get("device_name"),
        "product_line": info.get("product_line"),
        "firmware_version": info.get("firmware_version"),
        "width": info.get("width") if info.get("width") is not None else getattr(config, "width", None),
        "height": info.get("height") if info.get("height") is not None else getattr(config, "height", None),
        "fps": info.get("fps") if info.get("fps") is not None else getattr(config, "fps", None),
        "color_space": info.get("color_space") or getattr(config, "color_space", None),
        "depth_enabled": depth_enabled,
        "depth_required": depth_required,
        "depth_width": info.get("depth_width") if info.get("depth_width") is not None else getattr(config, "depth_width", None),
        "depth_height": info.get("depth_height") if info.get("depth_height") is not None else getattr(config, "depth_height", None),
        "depth_fps": info.get("depth_fps") if info.get("depth_fps") is not None else getattr(config, "depth_fps", None),
        "depth_scale_m_per_unit": depth_scale,
        "depth_frame_present": depth_frame_present,
        "depth_status_level": depth_status_level,
        "depth_status_label": depth_status_label,
        "depth_status_message": depth_status_message,
        "depth_metadata": _json_safe_value(dict(latest_depth_metadata)),
        "depth_frame_number": latest_depth_metadata.get("frame_number"),
        "depth_camera_timestamp_sec": latest_depth_metadata.get("camera_timestamp_sec"),
        "depth_sensor_timestamp_ms": latest_depth_metadata.get("sensor_timestamp_ms"),
        **depth_freshness,
        "frame_count": frame_count,
        "last_frame_age_sec": last_frame_age_sec,
        "stale_after_sec": stale_after_sec,
        "timeout_count": timeout_count,
        "consecutive_timeout_count": consecutive_timeout_count,
        "restart_count": restart_count,
        "restart_attempt_count": restart_attempt_count,
        "restart_failure_count": restart_failure_count,
        "last_error": last_error,
        "last_runtime_error": last_runtime_error,
        "last_restart_timestamp": info.get("last_restart_timestamp"),
        "preview_selected": bool(preview_selected),
        "preview_active": bool(preview_active and preview_selected),
        "metadata": _json_safe_value(metadata),
        "updated_at": float(now),
    }


def _build_camera_status(
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    *,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
    camera_previewer: CameraPreviewer | None = None,
    now: float | None = None,
    depth_monitor: WristDepthFreshnessMonitor | None = None,
) -> dict[str, object]:
    if not camera_sources:
        return {}
    current_time = time.time() if now is None else float(now)
    preview_keys = set(preview_camera_sources or {})
    preview_active = camera_previewer is not None
    return {
        str(camera_key): _camera_status_for_source(
            str(camera_key),
            source,
            now=current_time,
            preview_selected=str(camera_key) in preview_keys,
            preview_active=preview_active,
            depth_monitor=depth_monitor,
        )
        for camera_key, source in sorted(camera_sources.items(), key=lambda item: str(item[0]))
    }


def _wrist_depth_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "wrist_depth_mask", False)) or bool(
        getattr(args, "wrist_depth_raw", False)
    )


def _with_expected_wrist_depth_camera_statuses(
    camera_status: Mapping[str, object],
    *,
    args: argparse.Namespace,
    now: float,
) -> dict[str, object]:
    merged = dict(camera_status)
    if not _wrist_depth_requested(args):
        return merged
    for camera_key in WRIST_DEPTH_CAMERA_KEYS:
        if camera_key in merged:
            continue
        merged[camera_key] = {
            "camera_key": camera_key,
            "status_level": "error",
            "status_label": "Wrist 相机缺失",
            "status_message": (
                f"--wrist-depth-mask/--wrist-depth-raw requested but {camera_key} "
                "is absent from camera_sources/config."
            ),
            "name": camera_key,
            "serial": None,
            "depth_enabled": True,
            "depth_required": True,
            "depth_frame_present": False,
            "depth_status_level": "error",
            "depth_status_label": "Wrist Depth 缺失",
            "depth_status_message": f"{camera_key} 没有可用 RealSense source，无法生成 wrist depth mask。",
            "frame_count": 0,
            "last_frame_age_sec": None,
            "stale_after_sec": None,
            "timeout_count": None,
            "consecutive_timeout_count": None,
            "restart_count": None,
            "preview_selected": False,
            "preview_active": False,
            "updated_at": float(now),
        }
    return merged


def _camera_status_blocking_message(
    camera_status: Mapping[str, Mapping[str, object]] | None,
    *,
    require_ok: bool = False,
) -> str:
    if not camera_status:
        return ""
    blocking_levels = {"error", "waiting"} if not require_ok else {"error", "waiting", "warn", "off"}
    for camera_key, item in sorted(camera_status.items(), key=lambda pair: str(pair[0])):
        if not isinstance(item, Mapping):
            continue
        level = str(item.get("status_level") or "ok")
        depth_level = str(item.get("depth_status_level") or "ok")
        label = str(item.get("status_label") or level)
        message = str(item.get("status_message") or item.get("depth_status_message") or "")
        if level in blocking_levels:
            suffix = f": {message}" if message else ""
            return f"等待相机 {camera_key} 状态恢复：{label}{suffix}"
        if bool(item.get("depth_required")) and depth_level in blocking_levels:
            depth_label = str(item.get("depth_status_label") or depth_level)
            depth_message = str(item.get("depth_status_message") or "")
            suffix = f": {depth_message}" if depth_message else ""
            return f"等待相机 {camera_key} wrist depth 恢复：{depth_label}{suffix}"
    return ""

def _camera_self_check_payload(
    camera_status: Mapping[str, Mapping[str, object]],
    *,
    started_at: float,
    now: float,
    duration_sec: float,
    completed: bool,
) -> dict[str, object]:
    entries = {
        str(key): item
        for key, item in camera_status.items()
        if isinstance(item, Mapping)
    }
    total = len(entries)
    elapsed_sec = max(0.0, float(now) - float(started_at))
    remaining_sec = max(0.0, float(duration_sec) - elapsed_sec)
    failed_keys: list[str] = []
    waiting_keys: list[str] = []
    warn_keys: list[str] = []
    passed_keys: list[str] = []
    for key, item in sorted(entries.items()):
        level = str(item.get("status_level") or "ok")
        depth_level = str(item.get("depth_status_level") or "ok")
        if level == "error" or (bool(item.get("depth_required")) and depth_level == "error"):
            failed_keys.append(key)
        elif level == "waiting" or (bool(item.get("depth_required")) and depth_level == "waiting"):
            waiting_keys.append(key)
        elif level == "warn" or (bool(item.get("depth_required")) and depth_level == "warn"):
            warn_keys.append(key)
        else:
            passed_keys.append(key)

    if total <= 0:
        status = "skipped"
        status_level = "off"
        status_label = "未配置"
        message = "当前没有 RealSense camera_sources。"
    elif not completed:
        status = "running"
        status_level = "waiting"
        status_label = "自检中"
        message = f"启动相机自检 {elapsed_sec:.1f}/{float(duration_sec):.1f}s。"
    elif failed_keys or waiting_keys:
        status = "failed"
        status_level = "error"
        status_label = "自检失败"
        problem_keys = failed_keys + waiting_keys
        message = "启动相机自检失败：" + ", ".join(problem_keys)
    elif warn_keys:
        status = "warning"
        status_level = "warn"
        status_label = "自检有警告"
        message = "启动相机自检有警告：" + ", ".join(warn_keys)
    else:
        status = "passed"
        status_level = "ok"
        status_label = "自检通过"
        message = f"启动相机自检通过：{len(passed_keys)}/{total} cameras。"

    return {
        "enabled": total > 0,
        "status": status,
        "status_level": status_level,
        "status_label": status_label,
        "message": message,
        "duration_sec": float(duration_sec),
        "elapsed_sec": elapsed_sec,
        "remaining_sec": remaining_sec,
        "started_at": float(started_at),
        "completed_at": float(now) if completed else None,
        "camera_count": int(total),
        "passed_count": int(len(passed_keys)),
        "failed_keys": failed_keys,
        "waiting_keys": waiting_keys,
        "warn_keys": warn_keys,
    }

def _tactile_ready_counts(value: object) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        return 0, 0
    total = len(value)
    ready = sum(1 for item in value.values() if bool(item))
    return ready, total

def _tactile_device_count(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0

def _tactile_frame_count(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0

def _tactile_camera_status_for_side(
    side: str,
    item: Mapping[str, object],
    *,
    now: float,
) -> dict[str, object]:
    configured = bool(item.get("configured", False))
    connect_tactile = bool(item.get("connect_tactile", False))
    is_connected = bool(item.get("is_connected", False))
    telemetry_error = _as_status_error_text(item.get("telemetry_error"))
    devices = item.get("tactile_devices") or ()
    ready = item.get("tactile_ready") or {}
    frame_ids = item.get("tactile_frame_ids") or {}
    ready_count, ready_total = _tactile_ready_counts(ready)
    device_count = _tactile_device_count(devices)
    frame_count = _tactile_frame_count(frame_ids)
    expected_count = max(device_count, ready_total, frame_count)
    tactile_timestamp = _safe_float_value(item.get("tactile_timestamp"))
    tactile_age_sec = None if tactile_timestamp is None else max(0.0, float(now) - tactile_timestamp)
    tactile_read_hz = _safe_float_value(item.get("tactile_read_hz"))
    tactile_max_fps = _safe_float_value(item.get("tactile_max_fps"))
    reference_hz = tactile_read_hz if tactile_read_hz and tactile_read_hz > 0.0 else tactile_max_fps
    stale_after_sec = 2.0
    if reference_hz is not None and reference_hz > 0.0:
        stale_after_sec = max(1.0, min(3.0, 3.0 / float(reference_hz)))
    process_read = bool(item.get("tactile_process_read", False))
    worker_alive = item.get("tactile_worker_alive")
    receiver_alive = item.get("tactile_worker_receiver_alive")

    status_level = "ok"
    status_label = "正常"
    status_message = ""
    if not configured:
        status_level = "off"
        status_label = "未配置"
    elif not connect_tactile:
        status_level = "off"
        status_label = "未启用"
    elif telemetry_error:
        status_level = "error"
        status_label = "连接/读取异常"
        status_message = telemetry_error
    elif not is_connected:
        status_level = "error"
        status_label = "夹爪未连接"
    elif process_read and worker_alive is False:
        status_level = "error"
        status_label = "worker 异常"
        status_message = "tactile worker is not alive"
    elif process_read and receiver_alive is False:
        status_level = "error"
        status_label = "接收进程异常"
        status_message = "tactile worker receiver is not alive"
    elif expected_count <= 0:
        status_level = "waiting"
        status_label = "等待触觉设备"
    elif ready_total > 0 and ready_count <= 0:
        status_level = "error"
        status_label = "设备无有效数据"
    elif ready_total > 0 and ready_count < ready_total:
        status_level = "warn"
        status_label = "部分设备异常"
    elif tactile_age_sec is None:
        status_level = "waiting"
        status_label = "等待触觉帧"
    elif tactile_age_sec >= stale_after_sec * 2.0:
        status_level = "error"
        status_label = "数据停滞"
        status_message = f"last tactile frame age {tactile_age_sec:.2f}s"
    elif tactile_age_sec >= stale_after_sec:
        status_level = "warn"
        status_label = "触觉帧延迟"
        status_message = f"last tactile frame age {tactile_age_sec:.2f}s"

    return {
        "side": str(side),
        "configured": configured,
        "connect_tactile": connect_tactile,
        "is_connected": is_connected,
        "status_level": status_level,
        "status_label": status_label,
        "status_message": status_message,
        "telemetry_error": telemetry_error,
        "device_count": device_count,
        "ready_count": ready_count,
        "ready_total": ready_total,
        "frame_count": frame_count,
        "tactile_devices": _json_safe_value(devices),
        "tactile_ready": _json_safe_value(ready),
        "tactile_frame_ids": _json_safe_value(frame_ids),
        "tactile_field_by_key": _json_safe_value(item.get("tactile_field_by_key") or {}),
        "tactile_shapes": _json_safe_value(item.get("tactile_shapes") or {}),
        "tactile_timings_ms": _json_safe_value(item.get("tactile_timings_ms") or {}),
        "tactile_timestamp": tactile_timestamp,
        "tactile_age_sec": tactile_age_sec,
        "stale_after_sec": stale_after_sec,
        "tactile_version": item.get("tactile_version"),
        "tactile_read_hz": tactile_read_hz,
        "tactile_max_fps": tactile_max_fps,
        "tactile_read_elapsed_s": item.get("tactile_read_elapsed_s"),
        "tactile_gate_wait_s": item.get("tactile_gate_wait_s"),
        "tactile_read_skipped": item.get("tactile_read_skipped"),
        "tactile_process_read": process_read,
        "tactile_worker_alive": worker_alive,
        "tactile_worker_receiver_alive": receiver_alive,
        "updated_at": float(now),
    }

def _build_tactile_camera_status(
    gripper_status: Mapping[str, Mapping[str, object]] | None,
    *,
    now: float | None = None,
) -> dict[str, object]:
    if not gripper_status:
        return {}
    current_time = time.time() if now is None else float(now)
    return {
        str(side): _tactile_camera_status_for_side(str(side), item, now=current_time)
        for side, item in gripper_status.items()
        if isinstance(item, Mapping)
    }

def _debug_timing_fault_text(result) -> str:
    if result is None or not result.faults:
        return "faults=none"
    codes = [
        f"{fault.code}:{fault.channel_name}" if fault.channel_name else str(fault.code)
        for fault in result.faults
    ]
    return "faults=" + ",".join(codes)

def _pose_quality_reason_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item is not None]

def _pose_quality_level(status: object, *, available: bool, bad: bool) -> str:
    status_text = str(status or "unknown")
    if bad or status_text == "bad":
        return "error"
    if status_text == "warning":
        return "warn"
    if not available or status_text == "unknown":
        return "off"
    return "ok"


def _build_pose_quality_display_payload(active_command: object | None) -> dict[str, object]:
    quality = _collection_build_pose_quality_metadata(active_command)
    if not isinstance(quality, Mapping):
        quality = {}
    metrics = quality.get("metrics")
    metrics_dict = dict(metrics) if isinstance(metrics, Mapping) else {}
    pose_error = metrics_dict.get("ik_solve_pose_error")
    pose_error_dict = dict(pose_error) if isinstance(pose_error, Mapping) else {}
    joint_margin = metrics_dict.get("joint_limit_margin")
    joint_margin_dict = dict(joint_margin) if isinstance(joint_margin, Mapping) else {}
    available = bool(quality.get("available", False))
    status = str(quality.get("status") or "unknown")
    bad = bool(quality.get("bad", False))
    score = _float_or_none(quality.get("score"))
    return {
        "available": available,
        "status": status,
        "score": score,
        "bad": bad,
        "level": _pose_quality_level(status, available=available, bad=bad),
        "source": str(quality.get("source") or ""),
        "reasons": _pose_quality_reason_list(quality.get("reasons")),
        "bad_reasons": _pose_quality_reason_list(quality.get("bad_reasons")),
        "warning_reasons": _pose_quality_reason_list(quality.get("warning_reasons")),
        "metrics": {
            "collision_avoidance_status": metrics_dict.get("collision_avoidance_status"),
            "collision_distance_m": _float_or_none(metrics_dict.get("collision_distance_m")),
            "collision_min_distance_m": _float_or_none(metrics_dict.get("collision_min_distance_m")),
            "collision_margin_m": _float_or_none(metrics_dict.get("collision_margin_m")),
            "ik_jump_gate_status": metrics_dict.get("ik_jump_gate_status"),
            "ik_branch_candidate": metrics_dict.get("ik_branch_candidate"),
            "ik_translation_error_m": _float_or_none(pose_error_dict.get("max_translation_m")),
            "ik_rotation_error_deg": _float_or_none(pose_error_dict.get("max_rotation_deg")),
            "joint_nearest_margin_deg": _float_or_none(joint_margin_dict.get("nearest_margin_deg")),
            "joint_near_limit": bool(joint_margin_dict.get("near_limit", False)),
            "execution_step_limited": bool(metrics_dict.get("execution_step_limited", False)),
            "ik_time_budget_exceeded": bool(metrics_dict.get("ik_time_budget_exceeded", False)),
        },
    }


def _pose_quality_status_text(payload: Mapping[str, object] | None) -> str:
    if not isinstance(payload, Mapping):
        return "pose_quality=unavailable"
    status = str(payload.get("status") or "unknown")
    score = _compact_runtime_float(payload.get("score"), digits=2)
    bits = [f"pose_quality={status}", f"score={score}"]
    if not bool(payload.get("available", False)):
        bits.append("available=0")
    reasons = _pose_quality_reason_list(payload.get("reasons"))
    if reasons:
        bits.append("reason=" + ",".join(reasons[:3]))
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        collision_margin = _float_or_none(metrics.get("collision_margin_m"))
        joint_margin = _float_or_none(metrics.get("joint_nearest_margin_deg"))
        translation = _float_or_none(metrics.get("ik_translation_error_m"))
        rotation = _float_or_none(metrics.get("ik_rotation_error_deg"))
        if collision_margin is not None:
            bits.append(f"col_margin={collision_margin:.3f}m")
        if joint_margin is not None:
            bits.append(f"joint_margin={joint_margin:.1f}deg")
        if translation is not None or rotation is not None:
            bits.append(
                "pose_err="
                f"{0.0 if translation is None else translation:.3f}m/"
                f"{0.0 if rotation is None else rotation:.1f}deg"
            )
    return " ".join(bits)

def _runtime_fault_label(fault: object) -> str:
    code = str(getattr(fault, "code", "unknown"))
    channel_name = getattr(fault, "channel_name", None)
    label = f"{code}:{channel_name}" if channel_name else code
    metadata = getattr(fault, "metadata", None)
    if not isinstance(metadata, Mapping):
        return label
    details: list[str] = []
    if "translation_jump_m" in metadata:
        details.append(f"jump_m={_compact_runtime_float(metadata.get('translation_jump_m'), digits=4)}")
    if "rotation_jump_deg" in metadata:
        details.append(f"rot_deg={_compact_runtime_float(metadata.get('rotation_jump_deg'), digits=2)}")
    return f"{label}({','.join(details)})" if details else label

def _debug_timing_arm_text(
    *,
    left_hw_state: dict[str, object],
    right_hw_state: dict[str, object],
    left_command_state: dict[str, object],
    right_command_state: dict[str, object],
    target_step_scale: float,
) -> str:
    def code_text(state: dict[str, object]) -> str:
        code = state.get("hardware_result_code")
        text = state.get("hardware_result_text")
        return f"{code}:{text}" if text is not None else str(code)

    def timing_text(state: dict[str, object]) -> str:
        bits: list[str] = []
        send_timing = state.get("last_send_command_timing")
        if isinstance(send_timing, Mapping):
            bits.append(f"send={_compact_runtime_float(send_timing.get('total_ms'), digits=1)}ms")
            bits.append(f"lock={_compact_runtime_float(send_timing.get('send_lock_wait_ms'), digits=1)}ms")
        loop_timing = state.get("last_impedance_loop_timing")
        if isinstance(loop_timing, Mapping):
            bits.append(f"loop={_compact_runtime_float(loop_timing.get('total_ms'), digits=1)}ms")
            bits.append(f"lag={_compact_runtime_float(loop_timing.get('deadline_lag_ms'), digits=1)}ms")
        return ",".join(bits)

    base = (
        "arms="
        f"left_hw={code_text(left_hw_state)} "
        f"right_hw={code_text(right_hw_state)} "
        f"left_cmd={_command_backend_status_text(left_command_state)} "
        f"right_cmd={_command_backend_status_text(right_command_state)} "
        f"left_step_scale={float(left_command_state.get('effective_step_scale', 1.0)):.2f} "
        f"right_step_scale={float(right_command_state.get('effective_step_scale', 1.0)):.2f} "
        f"target_step_scale={float(target_step_scale):.2f}"
    )
    timing_bits = []
    left_timing = timing_text(left_command_state)
    if left_timing:
        timing_bits.append(f"left_timing={left_timing}")
    right_timing = timing_text(right_command_state)
    if right_timing:
        timing_bits.append(f"right_timing={right_timing}")
    return base if not timing_bits else base + " " + " ".join(timing_bits)

def _debug_timing_ik_text(active_command) -> str:
    if active_command is None:
        return "ik=none"
    metadata = getattr(active_command, "metadata", {}) or {}
    profile = metadata.get("ik_debug_profile")
    status = metadata.get("collision_avoidance_status")
    jump_gate = metadata.get("ik_jump_gate_status")
    branch = metadata.get("ik_branch_candidate")
    fallback = metadata.get("collision_avoidance_fallback_reason")
    prefix = f"ik=status={status} jump={jump_gate} branch={branch}"
    if fallback:
        prefix += f" fallback={fallback}"
    if not isinstance(profile, dict):
        return prefix
    min_phase = profile.get("min_distance_phase_ms")
    min_phase_text = ""
    if isinstance(min_phase, dict) and min_phase:
        ordered = sorted(
            ((str(name), float(value)) for name, value in min_phase.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        min_phase_text = " min_phase=" + ",".join(
            f"{name}:{value:.1f}ms" for name, value in ordered[:3]
        )
    return (
        f"{prefix} "
        f"total={float(profile.get('frame_total_ms', 0.0)):.1f}ms "
        f"solve={float(profile.get('solve_total_ms', 0.0)):.1f}ms "
        f"provider={float(profile.get('provider_total_ms', 0.0)):.1f}ms "
        f"other={float(profile.get('non_provider_ms', 0.0)):.1f}ms "
        f"fk={float(profile.get('fk_ms', 0.0)):.1f}ms "
        f"sing={float(profile.get('singularity_ms', 0.0)):.1f}ms "
        f"cand_eval={float(profile.get('candidate_eval_ms', 0.0)):.1f}ms "
        f"qp={float(profile.get('solve_qp_ms', 0.0)):.1f}ms "
        f"min_dist={float(profile.get('min_distance_total_ms', 0.0)):.1f}ms "
        f"branch={float(profile.get('branch_ms', 0.0)):.1f}ms "
        f"jump_gate={float(profile.get('jump_gate_ms', 0.0)):.1f}ms "
        f"hold_recovery={float(profile.get('hold_recovery_ms', 0.0)):.1f}ms "
        f"provider_calls={int(profile.get('provider_call_count', 0))} "
        f"candidate_evals={int(profile.get('candidate_eval_count', 0))} "
        f"solve_calls={int(profile.get('solve_calls', 0))} "
        f"min_calls={int(profile.get('min_distance_calls', 0))}"
        f"{min_phase_text}"
    )

_IK_RED_WARNING_STATUS_CODES = {"failed", "hold_last_safe"}
_IK_RED_WARNING_SINGULARITY_GATE_CODES = {"hold_entry"}
_IK_RED_WARNING_LABELS = {
    "failed": "IK failed",
    "hold_last_safe": "hold last safe",
    "hold_entry": "singularity hold entry",
}

def _float_or_none(*values: object) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None

def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

def _bool_or_false(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _mapping_or_empty(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}

def _stripped_code(value: object) -> str:
    return str(value).strip() if value is not None else ""

def _ik_warning_entries_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    source: str,
) -> list[dict[str, str]]:
    if not isinstance(metadata, Mapping):
        return []
    metadata_dict = dict(metadata)
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(code: str, field: str) -> None:
        normalized = code.strip()
        if not normalized:
            return
        key = (str(source), normalized)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "source": str(source),
                "code": normalized,
                "field": str(field),
                "label": _IK_RED_WARNING_LABELS.get(normalized, normalized),
            }
        )

    for key in ("status", "ik_status"):
        code = _stripped_code(metadata_dict.get(key))
        if code in _IK_RED_WARNING_STATUS_CODES:
            add(code, key)
    singularity_gate_status = _stripped_code(metadata_dict.get("singularity_gate_status"))
    if singularity_gate_status in _IK_RED_WARNING_SINGULARITY_GATE_CODES:
        add(singularity_gate_status, "singularity_gate_status")
    return entries

def _ik_warning_payload_from_entries(entries: Sequence[Mapping[str, str]]) -> dict[str, object]:
    compact_entries: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        compact_entry = {
            "source": str(entry.get("source") or "-"),
            "code": str(entry.get("code") or "-"),
            "field": str(entry.get("field") or "-"),
            "label": str(entry.get("label") or entry.get("code") or "-"),
        }
        key = (compact_entry["source"], compact_entry["code"], compact_entry["field"])
        if key in seen:
            continue
        seen.add(key)
        compact_entries.append(compact_entry)
    if not compact_entries:
        return {
            "active": False,
            "level": "none",
            "entries": [],
            "codes": [],
            "sources": [],
            "message": "",
        }
    codes = list(dict.fromkeys(entry["code"] for entry in compact_entries))
    sources = list(dict.fromkeys(entry["source"] for entry in compact_entries))
    message = "; ".join(
        f"{entry['source']}:{entry['label']}"
        for entry in compact_entries
    )
    return {
        "active": True,
        "level": "red",
        "entries": compact_entries,
        "codes": codes,
        "sources": sources,
        "message": message,
    }

def _build_ik_warning_payload(
    *,
    active_command,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    entries: list[dict[str, str]] = []
    if active_command is not None:
        entries.extend(
            _ik_warning_entries_from_metadata(
                getattr(active_command, "metadata", {}) or {},
                source="retargeter_command",
            )
        )
        commands = getattr(active_command, "commands", {}) or {}
        if isinstance(commands, Mapping):
            for side in ("left", "right"):
                side_command = commands.get(f"{side}_arm")
                if side_command is None:
                    continue
                entries.extend(
                    _ik_warning_entries_from_metadata(
                        getattr(side_command, "metadata", {}) or {},
                        source=f"{side}_retargeter_command",
                    )
                )
    for side, state in (
        ("left", left_command_state),
        ("right", right_command_state),
    ):
        if not isinstance(state, Mapping):
            continue
        entries.extend(
            _ik_warning_entries_from_metadata(
                _mapping_or_empty(state.get("ik_metadata")),
                source=f"{side}_backend",
            )
        )
        entries.extend(
            _ik_warning_entries_from_metadata(
                _mapping_or_empty(state.get("singularity_metadata")),
                source=f"{side}_backend",
            )
        )
    return _ik_warning_payload_from_entries(entries)

def _ik_timing_summary_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    source: str,
) -> dict[str, object] | None:
    if metadata is None:
        return None
    metadata_dict = dict(metadata)
    profile = _mapping_or_empty(metadata_dict.get("ik_debug_profile"))
    frame_total_ms = _float_or_none(
        profile.get("frame_total_ms"),
        metadata_dict.get("ik_frame_total_ms"),
        metadata_dict.get("ik_backend_solve_ms"),
    )
    solve_total_ms = _float_or_none(
        profile.get("solve_total_ms"),
        metadata_dict.get("ik_solve_total_ms"),
        metadata_dict.get("ik_backend_solve_ms"),
    )
    if frame_total_ms is None and solve_total_ms is None:
        return None
    solve_qp_ms = _float_or_none(profile.get("solve_qp_ms"), metadata_dict.get("ik_solve_qp_ms"))
    min_distance_total_ms = _float_or_none(
        profile.get("min_distance_total_ms"),
        metadata_dict.get("ik_min_distance_total_ms"),
    )
    provider_elapsed_ms = _float_or_none(
        profile.get("provider_total_ms"),
        metadata_dict.get("ik_provider_total_ms"),
        metadata_dict.get("ik_provider_elapsed_ms"),
    )
    non_provider_ms = _float_or_none(
        profile.get("non_provider_ms"),
        metadata_dict.get("ik_non_provider_ms"),
    )
    fk_ms = _float_or_none(profile.get("fk_ms"), metadata_dict.get("ik_fk_ms"))
    singularity_ms = _float_or_none(
        profile.get("singularity_ms"),
        metadata_dict.get("ik_singularity_ms"),
    )
    collision_ms = _float_or_none(profile.get("collision_ms"), metadata_dict.get("ik_collision_ms"))
    candidate_eval_ms = _float_or_none(
        profile.get("candidate_eval_ms"),
        metadata_dict.get("ik_candidate_eval_ms"),
    )
    time_budget_ms = _float_or_none(
        profile.get("time_budget_ms"),
        metadata_dict.get("ik_time_budget_ms"),
    )
    time_budget_exceeded = _bool_or_false(
        profile.get("time_budget_exceeded", metadata_dict.get("ik_time_budget_exceeded", False))
    )
    status = str(
        metadata_dict.get("collision_avoidance_status")
        or metadata_dict.get("status")
        or metadata_dict.get("ik_status")
        or "-"
    )
    return {
        "available": True,
        "source": source,
        "status": status,
        "jump_gate_status": str(metadata_dict.get("ik_jump_gate_status") or "-"),
        "branch_candidate": str(
            metadata_dict.get("ik_branch_candidate")
            or metadata_dict.get("ik_branch_selection")
            or metadata_dict.get("candidate_source")
            or "-"
        ),
        "frame_total_ms": 0.0 if frame_total_ms is None else frame_total_ms,
        "solve_total_ms": 0.0 if solve_total_ms is None else solve_total_ms,
        "solve_qp_ms": 0.0 if solve_qp_ms is None else solve_qp_ms,
        "min_distance_total_ms": 0.0 if min_distance_total_ms is None else min_distance_total_ms,
        "provider_elapsed_ms": 0.0 if provider_elapsed_ms is None else provider_elapsed_ms,
        "non_provider_ms": non_provider_ms,
        "fk_ms": fk_ms,
        "singularity_ms": singularity_ms,
        "collision_ms": collision_ms,
        "candidate_eval_ms": candidate_eval_ms,
        "provider_call_count": _int_or_zero(profile.get("provider_call_count")),
        "candidate_eval_count": _int_or_zero(profile.get("candidate_eval_count")),
        "branch_ms": _float_or_none(profile.get("branch_ms")) or 0.0,
        "jump_gate_ms": _float_or_none(profile.get("jump_gate_ms")) or 0.0,
        "hold_recovery_ms": _float_or_none(profile.get("hold_recovery_ms")) or 0.0,
        "solve_calls": _int_or_zero(profile.get("solve_calls")),
        "min_distance_calls": _int_or_zero(profile.get("min_distance_calls")),
        "time_budget_ms": time_budget_ms,
        "time_budget_exceeded": time_budget_exceeded,
    }

def _ik_timing_summaries(
    *,
    active_command,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if active_command is not None:
        summary = _ik_timing_summary_from_metadata(
            getattr(active_command, "metadata", {}) or {},
            source="retargeter_command",
        )
        if summary is not None:
            summaries.append(summary)
    for side, state in (
        ("left", left_command_state),
        ("right", right_command_state),
    ):
        if not isinstance(state, Mapping):
            continue
        summary = _ik_timing_summary_from_metadata(
            _mapping_or_empty(state.get("ik_metadata")),
            source=f"{side}_backend",
        )
        if summary is not None:
            summaries.append(summary)
    return summaries

def _build_ik_timing_payload(
    *,
    active_command,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ik_warning = _build_ik_warning_payload(
        active_command=active_command,
        left_command_state=left_command_state,
        right_command_state=right_command_state,
    )
    summaries = _ik_timing_summaries(
        active_command=active_command,
        left_command_state=left_command_state,
        right_command_state=right_command_state,
    )
    if not summaries:
        return {
            "available": False,
            "message": "暂无 IK timing 元数据；retargeter 或后端尚未提供本帧耗时。",
            "ik_warning": ik_warning,
        }
    primary = next(
        (summary for summary in summaries if summary.get("source") == "retargeter_command"),
        max(summaries, key=lambda item: float(item.get("frame_total_ms", 0.0))),
    )
    payload = dict(primary)
    payload["per_source"] = [
        {
            "source": str(summary.get("source") or "-"),
            "frame_total_ms": float(summary.get("frame_total_ms", 0.0)),
            "time_budget_exceeded": bool(summary.get("time_budget_exceeded", False)),
        }
        for summary in summaries
    ]
    payload["message"] = "per_source=" + ",".join(
        f"{item['source']}:{float(item['frame_total_ms']):.1f}ms"
        for item in payload["per_source"]
    )
    payload["ik_warning"] = ik_warning
    return payload

def _ik_timing_status_text(payload: Mapping[str, object] | None) -> str:
    if not isinstance(payload, Mapping) or not bool(payload.get("available", False)):
        return "ik_time=unavailable"
    frame_total_ms = float(payload.get("frame_total_ms", 0.0))
    budget = _float_or_none(payload.get("time_budget_ms"))
    budget_text = "budget=off" if budget is None or budget <= 0.0 else f"budget={budget:.1f}ms"
    exceeded = " exceeded" if bool(payload.get("time_budget_exceeded", False)) else ""
    return (
        f"ik_time={frame_total_ms:.1f}ms {budget_text}{exceeded} "
        f"source={payload.get('source')}"
    )

__all__ = [
    "bind_legacy_support",
    "_IK_RED_WARNING_LABELS",
    "_IK_RED_WARNING_SINGULARITY_GATE_CODES",
    "_IK_RED_WARNING_STATUS_CODES",
    "_json_safe_value",
    "_safe_float_value",
    "_safe_int_value",
    "_compact_runtime_float",
    "_compact_runtime_bool",
    "_as_status_error_text",
    "_camera_status_blocking_message",
    "_camera_self_check_payload",
    "_tactile_ready_counts",
    "_tactile_device_count",
    "_tactile_frame_count",
    "_tactile_camera_status_for_side",
    "_build_tactile_camera_status",
    "_debug_timing_fault_text",
    "_float_or_none",
    "_int_or_zero",
    "_bool_or_false",
    "_mapping_or_empty",
    "_stripped_code",
    "_ik_warning_entries_from_metadata",
    "_ik_warning_payload_from_entries",
    "_build_ik_warning_payload",
    "_ik_timing_summary_from_metadata",
    "_ik_timing_summaries",
    "_build_ik_timing_payload",
    "_ik_timing_status_text",
    "_pose_quality_reason_list",
    "_pose_quality_level",
    "_build_pose_quality_display_payload",
    "_pose_quality_status_text",
    "_runtime_fault_label",
    "_debug_timing_arm_text",
    "_debug_timing_ik_text",
]
