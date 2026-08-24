"""Shared RealTeleop/UMI browser presentation and operator-event transport.

The class in this module owns HTTP/UI resources and a rebuildable UI cache.
It does not own hardware, motion, safety, calibration, or dataset lifecycle.
"""

from __future__ import annotations

import html
import json
import math
import os
import queue
import re
import threading
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_WEB_ASSET_DIR = Path(__file__).with_name("static")


def _load_web_asset(filename: str) -> str:
    return (_WEB_ASSET_DIR / filename).read_text(encoding="utf-8")


_HTML_PAGE = _load_web_asset("calibration_web_ui.html")

_FALLBACK_SCRIPT = _load_web_asset("calibration_web_ui_fallback.js")


_PAGE_VERSION = "real_teleop_gui_progress_plugin_v1"


def _html_text(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return html.escape(str(value), quote=False)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe_value(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_value(item())
        except Exception:
            pass
    return str(value)


def _json_response_payload(value: Any) -> bytes:
    safe_value = _json_safe_value(value)
    return json.dumps(safe_value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _encode_http_event_payload(payload: Mapping[str, Any]) -> str:
    """Normalize events received from the browser before queueing them.

    Task selection sends its task metadata as sibling JSON fields.  The
    runtime event consumer expects that metadata in the event suffix, just as
    it does for episode start, so preserve the same wire format for both
    browser events.
    """
    event = str(payload.get("event", "")).strip()
    if event in {"episode_start", "task_select"}:
        details = {
            key: str(payload.get(key, "") or "").strip()
            for key in (
                "dataset_base_root",
                "task_type",
                "task_key",
                "task_english_name",
                "task_chinese_name",
                "round_trip_phase",
            )
        }
        if event in {"episode_start", "task_select"} and "plus_variant_id" in payload:
            details["plus_variant_id"] = str(payload.get("plus_variant_id", "") or "").strip()
        if event == "episode_start":
            for key in ("ignored_camera_keys", "ignored_tactile_keys"):
                value = payload.get(key, [])
                details[key] = (
                    [str(item).strip() for item in value if str(item).strip()]
                    if isinstance(value, list)
                    else []
                )
        event = event + ":" + json.dumps(
            details,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    elif event == "episode_delete":
        raw_index = str(payload.get("episode_index", "")).strip()
        if not raw_index.isdigit():
            return ""
        event = f"episode_delete:{int(raw_index)}"
    elif event == "plus_curation_save":
        raw_index = str(payload.get("episode_index", "")).strip()
        if not raw_index.isdigit():
            return ""
        details = {
            "episode_index": int(raw_index),
            "disposition": str(payload.get("disposition", "unknown") or "unknown").strip(),
            "reason_code": str(payload.get("reason_code", "unknown") or "unknown").strip(),
        }
        if "pending_token" in payload:
            details["pending_token"] = str(payload.get("pending_token", "") or "").strip()
        event = "plus_curation_save:" + json.dumps(
            details,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return event


def _choose_directory_payload(initial_dir: str = "") -> dict[str, Any]:
    initial_text = os.path.expandvars(str(initial_dir or "")).strip()
    initial_path = Path(initial_text).expanduser() if initial_text else Path.home()
    if not initial_path.exists():
        initial_path = initial_path.parent if initial_path.parent.exists() else Path.home()
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(
            parent=root,
            title="选择数据集保存根目录",
            initialdir=str(initial_path),
            mustexist=False,
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001 - GUI folder dialog is optional.
        return {
            "ok": False,
            "path": "",
            "message": f"无法打开文件夹选择器，请手动填写路径: {type(exc).__name__}: {exc}",
        }
    if not selected:
        return {"ok": False, "path": "", "message": "已取消文件夹选择。"}
    return {"ok": True, "path": str(Path(selected).expanduser()), "message": ""}


def _label_stage(value: Any) -> str:
    labels = {
        "Starting...": "启动中...",
        "RM65 真机遥操准备": "RM65 真机遥操准备",
        "Home Capture": "起点采集",
        "HomeCapture": "起点采集",
        "runtime_home": "runtime home",
        "teleop": "遥操",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def _label_phase(value: Any) -> str:
    labels = {
        "idle": "空闲",
        "ready": "准备就绪",
        "preparing": "准备倒计时",
        "capturing": "正在采集",
        "waiting": "等待样本",
        "teleop": "遥操中",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def _label_side(value: Any) -> str:
    labels = {"left": "左手", "right": "右手"}
    text = str(value or "")
    return labels.get(text, text or "-")


def _label_frame_viz(value: Any) -> str:
    labels = {"calibrated": "校准后", "raw": "原始", "both": "全部"}
    text = str(value or "")
    return labels.get(text, text or "-")


_RUNTIME_HOME_LIMIT_FAILURE_RE = re.compile(
    r"\b(?P<arm>left_arm|right_arm)\s+local\s+qpos\s+joint\s+"
    r"(?P<joint>\d+)\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))rad\s+"
    r"exceeds\s+absolute\s+limits\s+\[\s*"
    r"(?P<lower>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<upper>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]rad",
    re.IGNORECASE,
)


def _runtime_home_diagnostic_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Derive a display-only runtime-home status from the already published state."""

    hidden = {"visible": False, "level": "", "title": "", "detail": ""}
    control_hint = str(snapshot.get("control_hint") or "").strip()
    if "runtime home 标定失败" in control_hint:
        failure_line = next(
            (line.strip() for line in control_hint.splitlines() if line.strip()),
            control_hint,
        )
        limit_match = _RUNTIME_HOME_LIMIT_FAILURE_RE.search(control_hint)
        if limit_match is not None:
            groups = limit_match.groupdict()
            arm_label = "左臂" if groups["arm"] == "left_arm" else "右臂"
            return {
                "visible": True,
                "level": "error",
                "title": f"Runtime Home 标定失败：{arm_label}关节 {groups['joint']} 到达绝对限位",
                "detail": "\n".join(
                    (
                        f"当前位置：{groups['value']} rad",
                        f"允许范围：[{groups['lower']}, {groups['upper']}] rad",
                        "本次 Runtime Home 未完成，界面保持在重新标定等待状态。",
                        "请按现场安全流程检查关节姿态/限位后，点击“开始遥操”重新采集 Runtime Home。",
                        f"后端错误：{failure_line}",
                    )
                ),
            }
        return {
            "visible": True,
            "level": "error",
            "title": "Runtime Home 标定失败",
            "detail": "\n".join(
                (
                    failure_line or "后端未提供失败详情。",
                    "本次 Runtime Home 未完成，界面保持在重新标定等待状态。",
                    "请检查 VIVE tracker、机械臂状态和限位后，重新发起标定。",
                )
            ),
        }

    if (
        str(snapshot.get("capture_stage") or "") != "runtime_home"
        or str(snapshot.get("capture_phase") or "") != "capturing"
    ):
        return hidden

    min_samples = int(snapshot.get("capture_min_samples") or 0)
    left_samples = snapshot.get("capture_left_samples")
    right_samples = snapshot.get("capture_right_samples")
    samples = snapshot.get("capture_samples")
    progress_parts: list[str] = []
    if left_samples is not None:
        progress_parts.append(f"左臂 {int(left_samples or 0)}/{min_samples}")
    if right_samples is not None:
        progress_parts.append(f"右臂 {int(right_samples or 0)}/{min_samples}")
    if not progress_parts and samples is not None:
        progress_parts.append(f"{int(samples or 0)}/{min_samples}")
    validity_parts: list[str] = []
    if snapshot.get("capture_left_valid") is not None:
        validity_parts.append(
            f"左臂 {'有效' if bool(snapshot.get('capture_left_valid')) else '等待有效 tracker'}"
        )
    if snapshot.get("capture_right_valid") is not None:
        validity_parts.append(
            f"右臂 {'有效' if bool(snapshot.get('capture_right_valid')) else '等待有效 tracker'}"
        )
    if not validity_parts and snapshot.get("capture_valid") is not None:
        validity_parts.append(
            "tracker 有效" if bool(snapshot.get("capture_valid")) else "等待有效 tracker"
        )
    elapsed = snapshot.get("capture_elapsed_sec")
    timeout = snapshot.get("capture_duration_sec")
    timing = ""
    if elapsed is not None and timeout is not None:
        try:
            timing = f"采样时间：{float(elapsed):.1f}/{float(timeout):.1f}s"
        except (TypeError, ValueError):
            timing = ""
    detail_parts = [
        f"样本进度：{'；'.join(progress_parts) if progress_parts else '正在等待第一帧'}",
    ]
    if validity_parts:
        detail_parts.append(f"输入状态：{'；'.join(validity_parts)}")
    if timing:
        detail_parts.append(timing)
    if bool(snapshot.get("capture_waiting_for_samples")):
        detail_parts.append("正在等待新的有效 VIVE tracker 样本；无效帧不会计入进度。")
    else:
        detail_parts.append("请保持当前 VIVE 与机械臂 Home 姿态稳定，直到左右样本均采满。")
    return {
        "visible": True,
        "level": "capturing",
        "title": "Runtime Home 正在采样",
        "detail": "\n".join(detail_parts),
    }


def _initial_capture_phase(snapshot: dict[str, Any]) -> str:
    if bool(snapshot.get("teleop_running")):
        return "teleop"
    if snapshot.get("capture_stage"):
        return str(snapshot.get("capture_phase") or "idle")
    return "idle"


def _initial_capture_progress_and_detail(snapshot: dict[str, Any]) -> tuple[float, str]:
    phase = _initial_capture_phase(snapshot)
    capture_stage = str(snapshot.get("capture_stage") or "")
    sample_label = (
        "Home 样本"
        if capture_stage == "runtime_home"
        else "力控预热"
        if capture_stage == "force_control_warmup"
        else "标定样本"
    )
    if phase == "ready":
        return 0.0, "按“开始采集”后会先进入准备倒计时"
    if phase == "preparing":
        total = float(snapshot.get("capture_prep_total_sec") or 0.0)
        remaining = float(snapshot.get("capture_prep_remaining_sec") or 0.0)
        progress = (1.0 - remaining / total) * 100.0 if total > 0.0 else 100.0
        if capture_stage == "force_control_warmup":
            return progress, f"正在预热 RM65 力控，预计剩余 {remaining:.1f}s"
        return progress, f"倒计时 {remaining:.1f}s，当前还未写入{sample_label}"
    if phase == "capturing":
        min_samples = int(snapshot.get("capture_min_samples") or 0)
        left_samples = snapshot.get("capture_left_samples")
        right_samples = snapshot.get("capture_right_samples")
        single_samples = snapshot.get("capture_samples")
        if min_samples > 0 and (left_samples is not None or right_samples is not None):
            left = int(left_samples or 0)
            right = int(right_samples or 0)
            progress = min(left, right) / float(min_samples) * 100.0
            return progress, f"正在采集 {sample_label}，左 {left}/{min_samples}，右 {right}/{min_samples}"
        if min_samples > 0 and single_samples is not None:
            samples = int(single_samples or 0)
            progress = samples / float(min_samples) * 100.0
            return progress, f"正在采集 {sample_label}，{samples}/{min_samples}"
        duration = float(snapshot.get("capture_duration_sec") or 0.0)
        elapsed = float(snapshot.get("capture_elapsed_sec") or 0.0)
        progress = elapsed / duration * 100.0 if duration > 0.0 else 0.0
        return progress, f"正在写入样本，已采集 {elapsed:.1f} / {duration:.1f}s"
    if phase == "waiting":
        return 100.0, "等待 VIVE/Q6/相机样本进入采集窗口"
    if phase == "teleop":
        detail = "真机遥操中，Q6 夹爪控制已启用" if bool(snapshot.get("gripper_control_enabled")) else "真机遥操中"
        return 100.0, detail
    if snapshot.get("capture_stage"):
        return 0.0, "等待流程进入采集阶段"
    return 0.0, "等待后端发布实时状态"


def _initial_capture_samples(snapshot: dict[str, Any]) -> str:
    left = snapshot.get("capture_left_samples")
    right = snapshot.get("capture_right_samples")
    min_samples = int(snapshot.get("capture_min_samples") or 0)
    if left is not None or right is not None:
        return f"左 {int(left or 0)}/{min_samples}  右 {int(right or 0)}/{min_samples}"
    samples = snapshot.get("capture_samples")
    if samples is not None:
        return f"{int(samples or 0)}/{min_samples}"
    return "-"


def _initial_capture_valid(snapshot: dict[str, Any]) -> str:
    left = snapshot.get("capture_left_valid")
    right = snapshot.get("capture_right_valid")
    if left is not None or right is not None:
        return f"左 {'有效' if bool(left) else '缺失'} / 右 {'有效' if bool(right) else '缺失'}"
    valid = snapshot.get("capture_valid")
    if valid is not None:
        return "有效" if bool(valid) else "缺失"
    return "-"


def _format_initial_vector(values: Any, *, digits: int) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return "-"
    parts: list[str] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            parts.append("nan" if value is None else str(value))
        else:
            parts.append(f"{numeric:.{digits}f}")
    return ", ".join(parts)


def _initial_raw_pose_text(pose: Any) -> tuple[str, str]:
    if isinstance(pose, dict) and pose.get("enabled") is False:
        return "未启用", "当前 logical side 未启用"
    if not isinstance(pose, dict) or not pose:
        return "无数据", "未收到 channel"
    status = str(pose.get("status_label") or ("有效" if bool(pose.get("valid")) else "无效"))
    age = pose.get("age_sec")
    try:
        age_text = "-" if age is None else f"{float(age):.2f} s"
    except (TypeError, ValueError):
        age_text = str(age)
    text = "\n".join(
        (
            f"tracker: {pose.get('tracker_name') or '-'}",
            f"role:    {pose.get('tracker_role') or '-'}",
            f"seq:     {'-' if pose.get('seq') is None else pose.get('seq')}",
            f"age:     {age_text}",
            f"xyz(m):  {_format_initial_vector(pose.get('xyz_m'), digits=4)}",
            f"rpy(deg):{_format_initial_vector(pose.get('rpy_deg'), digits=1)}",
        )
    )
    return status, text


def _render_initial_html(
    snapshot: dict[str, Any],
    *,
    resizable_sidebars: bool = True,
    input_activity_indicators: bool = True,
) -> str:
    phase = _initial_capture_phase(snapshot)
    allowed_phase = phase if phase in {"idle", "ready", "preparing", "capturing", "waiting", "teleop"} else "idle"
    progress, capture_detail = _initial_capture_progress_and_detail(snapshot)
    progress = max(0.0, min(100.0, progress))
    raw_poses = snapshot.get("raw_tracker_poses") if isinstance(snapshot.get("raw_tracker_poses"), dict) else {}
    left_raw_status, left_raw_pose = _initial_raw_pose_text(raw_poses.get("left") if isinstance(raw_poses, dict) else None)
    right_raw_status, right_raw_pose = _initial_raw_pose_text(raw_poses.get("right") if isinstance(raw_poses, dict) else None)
    yaw = snapshot.get("virtual_desk_post_yaw_deg")
    yaw_text = "-" if yaw is None else f"{yaw} deg"
    runtime_home_diagnostic = snapshot.get("runtime_home_diagnostic")
    if not isinstance(runtime_home_diagnostic, dict):
        runtime_home_diagnostic = _runtime_home_diagnostic_payload(snapshot)
    diagnostic_visible = bool(runtime_home_diagnostic.get("visible"))
    diagnostic_level = str(runtime_home_diagnostic.get("level") or "")
    if diagnostic_level not in {"capturing", "error"}:
        diagnostic_level = ""
    replacements = {
        "__PAGE_TITLE__": (
            "DexTeleop UMI 数采"
            if str(snapshot.get("ui_mode")) == "umi_collection"
            else "VIVE -> RM65 标定"
        ),
        "__INITIAL_STAGE__": _label_stage(snapshot.get("stage")),
        "__INITIAL_SELECTED_SIDE__": _label_side(snapshot.get("selected_side")),
        "__INITIAL_FRAME_VIZ_MODE__": _label_frame_viz(snapshot.get("frame_viz_mode")),
        "__INITIAL_VIRTUAL_DESK_YAW__": yaw_text,
        "__INITIAL_CONTROL_HINT__": str(snapshot.get("control_hint") or "-"),
        "__INITIAL_CAPTURE_PHASE_CLASS__": allowed_phase,
        "__INITIAL_CAPTURE_LABEL__": _label_phase(phase),
        "__INITIAL_CAPTURE_DETAIL__": capture_detail,
        "__INITIAL_CAPTURE_PROGRESS__": f"{progress:.1f}",
        "__INITIAL_CAPTURE_STAGE__": _label_stage(snapshot.get("capture_stage")),
        "__INITIAL_CAPTURE_SAMPLES__": _initial_capture_samples(snapshot),
        "__INITIAL_CAPTURE_VALID__": _initial_capture_valid(snapshot),
        "__INITIAL_CAPTURE_STATUS__": "遥操中" if bool(snapshot.get("teleop_running")) else _label_phase(phase),
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_LEVEL__": diagnostic_level,
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_HIDDEN__": "" if diagnostic_visible else "hidden",
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_TITLE__": str(runtime_home_diagnostic.get("title") or ""),
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_DETAIL__": str(runtime_home_diagnostic.get("detail") or ""),
        "__INITIAL_LEFT_RAW_STATUS__": left_raw_status,
        "__INITIAL_LEFT_RAW_POSE__": left_raw_pose,
        "__INITIAL_RIGHT_RAW_STATUS__": right_raw_status,
        "__INITIAL_RIGHT_RAW_POSE__": right_raw_pose,
    }
    page = _HTML_PAGE
    raw_replacements = {
        "__RESIZABLE_SIDEBAR_MAIN_ATTRS__": (
            ' data-resizable-sidebars="true"' if resizable_sidebars else ""
        ),
        "__RESIZABLE_SIDEBAR_DIVIDER__": (
            '<div id="sidebarResizeHandle" class="sidebar-resize-handle" '
            'role="separator" aria-label="调整左右栏宽度" aria-orientation="vertical" '
            'aria-valuemin="0" aria-valuemax="100" aria-valuenow="60" tabindex="0"></div>'
            if resizable_sidebars
            else ""
        ),
        "__RESIZABLE_SIDEBAR_SCRIPT__": (
            '<script src="/sidebar-resize.js"></script>' if resizable_sidebars else ""
        ),
        "__INPUT_ACTIVITY_INDICATORS__": (
            '<div id="inputActivityStrip" class="input-activity-strip" '
            'aria-label="按键和踏板输入提示">'
            '<span class="input-activity-label">输入</span>'
            '<span class="input-activity-cell" data-input-activity="key_a">A</span>'
            '<span class="input-activity-cell" data-input-activity="key_b">B</span>'
            '<span class="input-activity-cell" data-input-activity="key_c">C</span>'
            '<span class="input-activity-cell" data-input-activity="key_g">G</span>'
            '<span class="input-activity-cell pedal" data-input-activity="pedal_1">P1</span>'
            '<span class="input-activity-cell pedal" data-input-activity="pedal_2">P2</span>'
            '<span class="input-activity-cell pedal" data-input-activity="pedal_3">P3</span>'
            '<span class="input-activity-cell pedal" data-input-activity="pedal_4">P4</span>'
            '<span class="input-activity-cell pedal" data-input-activity="pedal_5">P5</span>'
            '</div>'
            if input_activity_indicators
            else ""
        ),
        "__INPUT_ACTIVITY_SCRIPT__": (
            '<script src="/input-activity.js"></script>'
            if input_activity_indicators
            else ""
        ),
    }
    for token, value in raw_replacements.items():
        page = page.replace(token, value)
    for token, value in replacements.items():
        escaped = _html_text(value)
        page = page.replace(token, escaped)
    return page


@dataclass
class CalibrationWebUiState:
    stage: str = "Starting..."
    selected_side: str = "right"
    frame_viz_mode: str = "calibrated"
    virtual_desk_post_yaw_deg: float = 180.0
    renderer_ready: bool = False
    ui_mode: str = "web"
    host: str = ""
    port: int = 0
    load_profile: str = ""
    show_tracker_frame_viz: bool = False
    home_ready: bool = False
    teleop_running: bool = False
    raw_tracker_poses: dict[str, Any] = field(default_factory=dict)
    tracker_side_check: dict[str, Any] = field(default_factory=dict)
    debug_ik_timing: bool = False
    ik_timing: dict[str, Any] = field(default_factory=dict)
    ik_warning: dict[str, Any] = field(default_factory=dict)
    pose_quality: dict[str, Any] = field(default_factory=dict)
    debug_ik_logging: bool = False
    ik_debug_log_path: str = ""
    gripper_control_available: bool = False
    gripper_q6_available: bool = False
    gripper_control_enabled: bool = False
    gripper_status: dict[str, Any] = field(default_factory=dict)
    camera_status: dict[str, Any] = field(default_factory=dict)
    camera_self_check: dict[str, Any] = field(default_factory=dict)
    tactile_camera_status: dict[str, Any] = field(default_factory=dict)
    dataset_modality_contract: dict[str, Any] = field(default_factory=dict)
    hardware_status: dict[str, Any] = field(default_factory=dict)
    episode_recording_available: bool = False
    episode_recording: bool = False
    episode_dir: str = ""
    episode_frame_count: int | None = None
    episode_attempted_sample_count: int | None = None
    episode_rejection_counts: dict[str, Any] = field(default_factory=dict)
    episode_sample_hz: float | None = None
    episode_target_hz: float | None = None
    episode_tactile_write_hz: dict[str, Any] = field(default_factory=dict)
    episode_tactile_write_hz_mean: float | None = None
    episode_tactile_frame_counts: dict[str, Any] = field(default_factory=dict)
    episode_frequency_warning: dict[str, Any] = field(default_factory=dict)
    calibration_validity_warning: dict[str, Any] = field(default_factory=dict)
    episode_event_status: str = "unavailable"
    episode_event_message: str = ""
    episode_last_episode_index: int | None = None
    episode_last_frame_count: int | None = None
    # Rebuildable Control Station presentation only. Canonical progress facts
    # and lifecycle remain outside the browser/UI owner.
    progress_ui: dict[str, Any] = field(default_factory=dict)
    # Optional Plus episode curation is a post-finalize sidecar.  These fields
    # are inert when the feature is disabled and intentionally do not mirror
    # any raw writer/row state.
    plus_curation_enabled: bool = False
    plus_curation_task_profile_id: str = ""
    plus_curation_variants: list[dict[str, Any]] = field(default_factory=list)
    plus_curation_variant_id: str = ""
    plus_curation_disposition: str = "unknown"
    plus_curation_reason_code: str = "unknown"
    plus_curation_pending: bool = False
    plus_curation_episode_index: int | None = None
    plus_curation_pending_token: str = ""
    plus_curation_pending_dataset_root: str = ""
    plus_curation_status: str = "off"
    plus_curation_message: str = ""
    dataset_base_root: str = ""
    dataset_root: str = ""
    dataset_dir_name: str = ""
    collection_task_id: str = ""
    collection_task_key: str = ""
    collection_task_name: str = ""
    collection_task_name_zh: str = ""
    collection_task_description: str = ""
    collection_task_type: str = "general"
    collection_round_trip_phase: str = "in"
    collection_round_trip_next_phase: str = "out"
    collection_round_trip_base_name: str = ""
    collection_round_trip_base_id: str = ""
    collection_round_trip_active: bool = False
    gripper_task: str = ""
    task_registry: list[dict[str, Any]] = field(default_factory=list)
    camera_visualization_available: bool = False
    camera_visualization_active: bool = False
    tactile_visualization_available: bool = False
    tactile_visualization_active: bool = False
    tactile_solving_available: bool = False
    tactile_solving_active: bool = False
    visualization_status: str = ""
    vive_calibration_profile_path: str = ""
    vive_calibration_profile_status: str = ""
    vive_calibration_profile_ready: bool = False
    vive_calibration_profile_loaded: bool = False
    movement_multiple: float = 1.35
    movement_multiple_min: float = 0.05
    movement_multiple_max: float = 10.0
    movement_multiple_step: float = 0.1
    fix_arm_available: bool = False
    fix_arm_selected_side: str = ""
    fix_arm_fixed_side: str = ""
    fix_arm_fixed_sides: list[str] = field(default_factory=list)
    fix_arm_active: bool = False
    fix_arm_recalibrating: bool = False
    fix_arm_status: str = ""
    fix_arm_sample_count: int | None = None
    fix_arm_min_samples: int | None = None
    fix_arm_home_cache: dict[str, Any] = field(default_factory=dict)
    special_arm_control_mode: str = "manual"
    foot_pad_requested: bool = False
    foot_pad_available: bool = False
    foot_pad_count: int = 0
    foot_pad_status: str = "disabled"
    foot_pad_status_label: str = "未启用"
    foot_pad_status_message: str = ""
    foot_pad_test_active: bool = False
    foot_pad_test_completed: bool = False
    foot_pad_test_seen_count: int = 0
    foot_pad_test_last_event: str = ""
    foot_pad_test_status: str = ""
    # Rebuildable presentation pulse only. It is neither an operator intent
    # nor a control receipt and is never persisted into the dataset.
    input_activity: dict[str, Any] = field(default_factory=dict)
    init_arm_available: bool = False
    init_arm_selected_side: str = ""
    init_arm_active_side: str = ""
    init_arm_active: bool = False
    init_arm_recalibrating: bool = False
    init_arm_status: str = ""
    init_arm_sample_count: int | None = None
    init_arm_min_samples: int | None = None
    init_arm_profile_path: str = ""
    init_arm_arrival_status: str = "inactive"
    init_arm_arrival_label: str = ""
    init_arm_arrival_reason: str = ""
    init_arm_target_position_error_m: float | None = None
    init_arm_target_rotation_error_deg: float | None = None
    init_arm_stability_dwell_sec: float | None = None
    init_arm_qpos_active: bool = False
    init_arm_qpos_phase: str = "inactive"
    init_arm_qpos_step_index: int | None = None
    init_arm_qpos_step_count: int | None = None
    init_arm_qpos_max_error_deg: float | None = None
    init_arm_qpos_move_available: bool = False
    startup_qpos_maintenance_enabled: bool = False
    init_arm_sequence_active: bool = False
    init_arm_sequence_side: str = ""
    init_arm_sequence_phase: str = "idle"
    init_arm_sequence_status: str = ""
    init_arm_qpos_speed_scale_by_side: dict[str, float] = field(
        default_factory=lambda: {"left": 5.0, "right": 5.0}
    )
    init_arm_qpos_speed_scale_min: float = 0.1
    init_arm_qpos_speed_scale_max: float = 10.0
    init_arm_qpos_speed_scale_default: float = 5.0
    fix_hold_active: bool = False
    two_pedal_alert: dict[str, Any] = field(default_factory=dict)
    pose_snapshot_available: bool = False
    pose_snapshot_dir: str = ""
    pose_snapshot_count: int = 0
    pose_snapshot_last_path: str = ""
    pose_snapshot_status: str = ""
    active_controls: list[str] = field(default_factory=lambda: ["q"])
    control_hint: str = ""
    capture_stage: str = ""
    capture_phase: str = "idle"
    capturing: bool = False
    capture_channel: str = ""
    capture_samples: int | None = None
    capture_left_samples: int | None = None
    capture_right_samples: int | None = None
    capture_min_samples: int | None = None
    capture_valid: bool | None = None
    capture_left_valid: bool | None = None
    capture_right_valid: bool | None = None
    capture_elapsed_sec: float | None = None
    capture_duration_sec: float | None = None
    capture_prep_remaining_sec: float | None = None
    capture_prep_total_sec: float | None = None
    capture_waiting_for_samples: bool = False
    shutdown: bool = False
    shutdown_message: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)


class CalibrationWebUi:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        resizable_sidebars: bool = True,
        input_activity_indicators: bool = True,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._open_browser = bool(open_browser)
        self._resizable_sidebars = bool(resizable_sidebars)
        self._input_activity_indicators = bool(input_activity_indicators)
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._events: queue.Queue[str] = queue.Queue()
        self._state_lock = threading.Lock()
        self._state = CalibrationWebUiState()

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}/"
        address = self._server.server_address
        host = address[0]
        port = address[1]
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return

        ui = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/state.json"):
                    payload = _json_response_payload(ui.snapshot())
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path.startswith("/choose-directory"):
                    parsed = urlparse(self.path)
                    params = parse_qs(parsed.query)
                    initial = params.get("initial", [""])[0]
                    payload = _json_response_payload(_choose_directory_payload(initial))
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path.startswith("/calibration-web-ui-fallback.js"):
                    payload = _FALLBACK_SCRIPT.encode("utf-8")
                    self._send(200, "application/javascript; charset=utf-8", payload)
                    return
                if self.path.startswith("/real-teleop-progress.js"):
                    payload = (
                        Path(__file__)
                        .with_name("real_teleop_progress.js")
                        .read_bytes()
                    )
                    self._send(200, "application/javascript; charset=utf-8", payload)
                    return
                if self.path.startswith("/operator-module-layout.js"):
                    payload = (
                        Path(__file__)
                        .with_name("operator_module_layout.js")
                        .read_bytes()
                    )
                    self._send(200, "application/javascript; charset=utf-8", payload)
                    return
                if self.path.startswith("/sidebar-resize.js"):
                    payload = Path(__file__).with_name("sidebar_resize.js").read_bytes()
                    self._send(200, "application/javascript; charset=utf-8", payload)
                    return
                if self.path.startswith("/input-activity.js"):
                    payload = Path(__file__).with_name("input_activity.js").read_bytes()
                    self._send(200, "application/javascript; charset=utf-8", payload)
                    return
                if self.path.startswith("/debug.json"):
                    snapshot = ui.snapshot()
                    payload = _json_response_payload(
                        {
                            "page_version": _PAGE_VERSION,
                            "html_contains_legacy_poll": "__calibrationWebUiLegacyPollActive" in _HTML_PAGE,
                            "html_contains_modern_poll": "__calibrationWebUiModernScriptLoaded" in _HTML_PAGE,
                            "html_contains_external_fallback": "calibration-web-ui-fallback.js" in _HTML_PAGE,
                            "fallback_script_path": "/calibration-web-ui-fallback.js",
                            "operator_module_script_path": "/operator-module-layout.js",
                            "resizable_sidebars": ui._resizable_sidebars,
                            "sidebar_resize_script_path": (
                                "/sidebar-resize.js" if ui._resizable_sidebars else None
                            ),
                            "input_activity_indicators": ui._input_activity_indicators,
                            "input_activity_script_path": (
                                "/input-activity.js"
                                if ui._input_activity_indicators
                                else None
                            ),
                            "progress_script_path": "/real-teleop-progress.js",
                            "json_sanitized": True,
                            "snapshot": snapshot,
                        }
                    )
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path in {"/", "/index.html"}:
                    payload = _render_initial_html(
                        ui.snapshot(),
                        resizable_sidebars=ui._resizable_sidebars,
                        input_activity_indicators=ui._input_activity_indicators,
                    ).encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", payload)
                    return
                self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/event":
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    event = _encode_http_event_payload(payload)
                except Exception:
                    self._send(400, "text/plain; charset=utf-8", b"invalid json")
                    return
                if not event:
                    self._send(400, "text/plain; charset=utf-8", b"missing event")
                    return
                if not ui.push_event(event):
                    self._send(409, "text/plain; charset=utf-8", b"inactive event")
                    return
                self._send(204, "text/plain; charset=utf-8", b"")

            def log_message(self, fmt: str, *args: Any) -> None:
                del fmt, args

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        if self._open_browser:
            try:
                webbrowser.open(self.url, new=1, autoraise=True)
            except Exception:
                pass

    def close(self) -> None:
        if self._server is None:
            return
        self.mark_shutdown("服务已退出，这个网页已经失效。")
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._server_thread = None

    def push_event(self, event: str) -> bool:
        event_name = str(event)
        event_control = event_name.split(":", 1)[0]
        if ":" in event_name and event_control not in {
            "movement_multiple_set",
            "gripper_force_set",
            "gripper_force_persist",
            "init_arm_qpos_speed_set",
            "episode_start",
            "episode_delete",
            "plus_curation_save",
            "progress_advance",
            "progress_profile_select",
            "progress_entry_step_select",
            "progress_arm_control_mode_select",
            "task_select",
            "fix_arm_select",
            "init_arm_select",
        }:
            return False
        with self._state_lock:
            active_controls = set(self._state.active_controls)
            accepted = event_name in active_controls or event_control in active_controls
            if accepted and event_control == "progress_advance":
                _, separator, target = event_name.partition(":")
                configured_targets = {
                    str(item.get("progress_step_id", "")).strip()
                    for item in self._state.progress_ui.get("next_steps", ())
                    if isinstance(item, Mapping)
                    and str(item.get("progress_step_id", "")).strip()
                }
                normalized_target = target.strip()
                if separator:
                    accepted = bool(
                        normalized_target
                        and normalized_target in configured_targets
                    )
                else:
                    accepted = len(configured_targets) == 1
            if accepted and event_control == "progress_profile_select":
                _, separator, profile_id = event_name.partition(":")
                configured_profile_ids = {
                    str(item.get("profile_id", "")).strip()
                    for item in self._state.progress_ui.get(
                        "available_profiles",
                        (),
                    )
                    if isinstance(item, Mapping)
                    and str(item.get("profile_id", "")).strip()
                }
                accepted = bool(
                    separator
                    and profile_id.strip()
                    and profile_id.strip() in configured_profile_ids
                    and not bool(
                        self._state.progress_ui.get(
                            "profile_selection_locked",
                            False,
                        )
                    )
                )
            if accepted and event_control == "progress_entry_step_select":
                _, separator, progress_step_id = event_name.partition(":")
                configured_entry_step_ids = {
                    str(item.get("progress_step_id", "")).strip()
                    for item in self._state.progress_ui.get("entry_steps", ())
                    if isinstance(item, Mapping)
                    and str(item.get("progress_step_id", "")).strip()
                }
                accepted = bool(
                    separator
                    and progress_step_id.strip() in configured_entry_step_ids
                    and bool(
                        self._state.progress_ui.get(
                            "entry_step_selection_enabled",
                            False,
                        )
                    )
                )
            if accepted and event_control == "progress_arm_control_mode_select":
                _, separator, mode = event_name.partition(":")
                configured_modes = {
                    str(item).strip()
                    for item in self._state.progress_ui.get(
                        "arm_control_modes",
                        (),
                    )
                    if str(item).strip()
                }
                accepted = bool(
                    separator
                    and mode.strip() in configured_modes
                    and bool(
                        self._state.progress_ui.get(
                            "arm_control_mode_selection_enabled",
                            False,
                        )
                    )
                )
            if accepted and event_control in {
                "episode_start",
                "episode_delete",
                "device_retry",
                "task_select",
            }:
                # Dataset rebind, Episode mutation, and device rebuild share
                # one pending application-mutation slot.  The next runtime
                # state publication re-enables whichever controls are legal.
                self._state.active_controls = [
                    control
                    for control in self._state.active_controls
                    if str(control)
                    not in {
                        "episode_start",
                        "episode_delete",
                        "device_retry",
                        "task_select",
                    }
                ]
            if accepted and event_control in {
                "progress_advance",
                "progress_failure",
                "progress_undo",
            }:
                # One canonical progress state may accept at most one pending
                # operator mutation. The reducer's next rebuildable state
                # publication re-enables whichever controls remain legal.
                self._state.active_controls = [
                    control
                    for control in self._state.active_controls
                    if str(control)
                    not in {
                        "progress_advance",
                        "progress_failure",
                        "progress_undo",
                    }
                ]
            if accepted and event_control == "progress_profile_select":
                self._state.active_controls = [
                    control
                    for control in self._state.active_controls
                    if str(control) != "progress_profile_select"
                ]
            if accepted and event_control == "progress_entry_step_select":
                self._state.active_controls = [
                    control
                    for control in self._state.active_controls
                    if str(control) != "progress_entry_step_select"
                ]
            if accepted and event_control == "episode_start":
                self._state.episode_event_status = "preparing"
                self._state.episode_event_message = "开始数采请求已接收，正在等待运行时处理。"
            if accepted and event_control == "episode_delete":
                self._state.episode_event_status = "delete_queued"
                self._state.episode_event_message = "删除请求已接收，正在等待数据集维护流程执行。"
            if accepted and event_control == "plus_curation_save":
                self._state.plus_curation_status = "saving"
                self._state.plus_curation_message = "Plus 整理标签已提交，正在写入旁路 sidecar。"
            if (
                event_name in {
                    "c",
                    "fix_arm_toggle",
                    "g",
                    "init_arm_toggle",
                    "init_arm_move_to_qpos",
                    "init_arm_move_both_to_qpos",
                }
                or event_control in {"fix_arm_select", "init_arm_select", "episode_delete"}
            ):
                status = "accepted" if accepted else "rejected"
                controls = ",".join(sorted(str(item) for item in active_controls))
                self._state.messages.append(
                    {
                        "tag": "WebGUIEvent",
                        "message": (
                            f"{status}: {event_name}; active_controls=[{controls}]"
                        ),
                    }
                )
                if len(self._state.messages) > 48:
                    self._state.messages = self._state.messages[-48:]
        if not accepted:
            return False
        self._events.put(event_name)
        return True

    def consume_events(self, *, max_events: int = 64) -> list[str]:
        events: list[str] = []
        for _ in range(max(1, int(max_events))):
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events
        return events

    def publish(self, **updates: Any) -> None:
        with self._state_lock:
            for key, value in updates.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

    def publish_input_activity(
        self,
        *,
        source: str,
        event: str,
        pedal_count: int | None = None,
    ) -> None:
        """Publish one non-authoritative input edge for the GUI pulse strip."""

        if not self._input_activity_indicators:
            return
        event_name = str(event)
        pedal_index_by_event = {
            "fix_arm_left": 1,
            "fix_arm_right": 2,
            "init_arm_left": 3,
            "init_arm_right": 4,
            "progress_undo": 5,
        }
        with self._state_lock:
            previous = self._state.input_activity
            sequence = int(previous.get("sequence", 0) or 0) + 1
            entry = {
                "sequence": sequence,
                "source": str(source),
                "event": event_name,
                "pedal_index": pedal_index_by_event.get(event_name),
                "pedal_count": None if pedal_count is None else int(pedal_count),
            }
            recent = [
                dict(item)
                for item in previous.get("events", ())
                if isinstance(item, Mapping)
            ]
            recent.append(entry)
            self._state.input_activity = {
                "sequence": sequence,
                # A small bounded presentation ring prevents two pedal edges
                # inside one 500 ms state refresh from collapsing into only
                # the latest flash.
                "events": recent[-16:],
            }

    def append_message(self, tag: str, message: str) -> None:
        with self._state_lock:
            self._state.messages.append({"tag": str(tag), "message": str(message)})
            if len(self._state.messages) > 48:
                self._state.messages = self._state.messages[-48:]

    def mark_shutdown(self, message: str = "") -> None:
        with self._state_lock:
            self._state.shutdown = True
            self._state.shutdown_message = str(message)
            self._state.active_controls = []

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            snapshot = {
                "stage": self._state.stage,
                "selected_side": self._state.selected_side,
                "frame_viz_mode": self._state.frame_viz_mode,
                "virtual_desk_post_yaw_deg": self._state.virtual_desk_post_yaw_deg,
                "renderer_ready": self._state.renderer_ready,
                "ui_mode": self._state.ui_mode,
                "host": self._state.host,
                "port": self._state.port,
                "load_profile": self._state.load_profile,
                "show_tracker_frame_viz": self._state.show_tracker_frame_viz,
                "home_ready": self._state.home_ready,
                "teleop_running": self._state.teleop_running,
                "raw_tracker_poses": dict(self._state.raw_tracker_poses),
                "tracker_side_check": dict(self._state.tracker_side_check),
                "debug_ik_timing": self._state.debug_ik_timing,
                "ik_timing": dict(self._state.ik_timing),
                "ik_warning": dict(self._state.ik_warning),
                "pose_quality": dict(self._state.pose_quality),
                "debug_ik_logging": self._state.debug_ik_logging,
                "ik_debug_log_path": self._state.ik_debug_log_path,
                "gripper_control_available": self._state.gripper_control_available,
                "gripper_q6_available": self._state.gripper_q6_available,
                "gripper_control_enabled": self._state.gripper_control_enabled,
                "gripper_status": dict(self._state.gripper_status),
                "camera_status": dict(self._state.camera_status),
                "camera_self_check": dict(self._state.camera_self_check),
                "tactile_camera_status": dict(self._state.tactile_camera_status),
                "dataset_modality_contract": dict(self._state.dataset_modality_contract),
                "hardware_status": dict(self._state.hardware_status),
                "episode_recording_available": self._state.episode_recording_available,
                "episode_recording": self._state.episode_recording,
                "episode_dir": self._state.episode_dir,
                "episode_frame_count": self._state.episode_frame_count,
                "episode_attempted_sample_count": self._state.episode_attempted_sample_count,
                "episode_rejection_counts": dict(self._state.episode_rejection_counts),
                "episode_sample_hz": self._state.episode_sample_hz,
                "episode_target_hz": self._state.episode_target_hz,
                "episode_tactile_write_hz": dict(self._state.episode_tactile_write_hz),
                "episode_tactile_write_hz_mean": self._state.episode_tactile_write_hz_mean,
                "episode_tactile_frame_counts": dict(self._state.episode_tactile_frame_counts),
                "episode_frequency_warning": dict(self._state.episode_frequency_warning),
                "calibration_validity_warning": dict(
                    self._state.calibration_validity_warning
                ),
                "episode_event_status": self._state.episode_event_status,
                "episode_event_message": self._state.episode_event_message,
                "episode_last_episode_index": self._state.episode_last_episode_index,
                "episode_last_frame_count": self._state.episode_last_frame_count,
                "progress_ui": dict(self._state.progress_ui),
                "plus_curation_enabled": self._state.plus_curation_enabled,
                "plus_curation_task_profile_id": self._state.plus_curation_task_profile_id,
                "plus_curation_variants": [
                    dict(item) for item in self._state.plus_curation_variants
                ],
                "plus_curation_variant_id": self._state.plus_curation_variant_id,
                "plus_curation_disposition": self._state.plus_curation_disposition,
                "plus_curation_reason_code": self._state.plus_curation_reason_code,
                "plus_curation_pending": self._state.plus_curation_pending,
                "plus_curation_episode_index": self._state.plus_curation_episode_index,
                "plus_curation_pending_token": self._state.plus_curation_pending_token,
                "plus_curation_pending_dataset_root": self._state.plus_curation_pending_dataset_root,
                "plus_curation_status": self._state.plus_curation_status,
                "plus_curation_message": self._state.plus_curation_message,
                "dataset_base_root": self._state.dataset_base_root,
                "dataset_root": self._state.dataset_root,
                "dataset_dir_name": self._state.dataset_dir_name,
                "collection_task_id": self._state.collection_task_id,
                "collection_task_key": self._state.collection_task_key,
                "collection_task_name": self._state.collection_task_name,
                "collection_task_name_zh": self._state.collection_task_name_zh,
                "collection_task_description": self._state.collection_task_description,
                "collection_task_type": self._state.collection_task_type,
                "collection_round_trip_phase": self._state.collection_round_trip_phase,
                "collection_round_trip_next_phase": self._state.collection_round_trip_next_phase,
                "collection_round_trip_base_name": self._state.collection_round_trip_base_name,
                "collection_round_trip_base_id": self._state.collection_round_trip_base_id,
                "collection_round_trip_active": self._state.collection_round_trip_active,
                "gripper_task": self._state.gripper_task,
                "task_registry": [dict(item) for item in self._state.task_registry],
                "camera_visualization_available": self._state.camera_visualization_available,
                "camera_visualization_active": self._state.camera_visualization_active,
                "tactile_visualization_available": self._state.tactile_visualization_available,
                "tactile_visualization_active": self._state.tactile_visualization_active,
                "tactile_solving_available": self._state.tactile_solving_available,
                "tactile_solving_active": self._state.tactile_solving_active,
                "visualization_status": self._state.visualization_status,
                "vive_calibration_profile_path": self._state.vive_calibration_profile_path,
                "vive_calibration_profile_status": self._state.vive_calibration_profile_status,
                "vive_calibration_profile_ready": self._state.vive_calibration_profile_ready,
                "vive_calibration_profile_loaded": self._state.vive_calibration_profile_loaded,
                "movement_multiple": self._state.movement_multiple,
                "movement_multiple_min": self._state.movement_multiple_min,
                "movement_multiple_max": self._state.movement_multiple_max,
                "movement_multiple_step": self._state.movement_multiple_step,
                "fix_arm_available": self._state.fix_arm_available,
                "fix_arm_selected_side": self._state.fix_arm_selected_side,
                "fix_arm_fixed_side": self._state.fix_arm_fixed_side,
                "fix_arm_fixed_sides": list(self._state.fix_arm_fixed_sides),
                "fix_arm_active": self._state.fix_arm_active,
                "fix_arm_recalibrating": self._state.fix_arm_recalibrating,
                "fix_arm_status": self._state.fix_arm_status,
                "fix_arm_sample_count": self._state.fix_arm_sample_count,
                "fix_arm_min_samples": self._state.fix_arm_min_samples,
                "fix_arm_home_cache": dict(self._state.fix_arm_home_cache),
                "special_arm_control_mode": self._state.special_arm_control_mode,
                "foot_pad_requested": self._state.foot_pad_requested,
                "foot_pad_available": self._state.foot_pad_available,
                "foot_pad_count": self._state.foot_pad_count,
                "foot_pad_status": self._state.foot_pad_status,
                "foot_pad_status_label": self._state.foot_pad_status_label,
                "foot_pad_status_message": self._state.foot_pad_status_message,
                "foot_pad_test_active": self._state.foot_pad_test_active,
                "foot_pad_test_completed": self._state.foot_pad_test_completed,
                "foot_pad_test_seen_count": self._state.foot_pad_test_seen_count,
                "foot_pad_test_last_event": self._state.foot_pad_test_last_event,
                "foot_pad_test_status": self._state.foot_pad_test_status,
                "input_activity": dict(self._state.input_activity),
                "init_arm_available": self._state.init_arm_available,
                "init_arm_selected_side": self._state.init_arm_selected_side,
                "init_arm_active_side": self._state.init_arm_active_side,
                "init_arm_active": self._state.init_arm_active,
                "init_arm_recalibrating": self._state.init_arm_recalibrating,
                "init_arm_status": self._state.init_arm_status,
                "init_arm_sample_count": self._state.init_arm_sample_count,
                "init_arm_min_samples": self._state.init_arm_min_samples,
                "init_arm_profile_path": self._state.init_arm_profile_path,
                "init_arm_arrival_status": self._state.init_arm_arrival_status,
                "init_arm_arrival_label": self._state.init_arm_arrival_label,
                "init_arm_arrival_reason": self._state.init_arm_arrival_reason,
                "init_arm_target_position_error_m": self._state.init_arm_target_position_error_m,
                "init_arm_target_rotation_error_deg": self._state.init_arm_target_rotation_error_deg,
                "init_arm_stability_dwell_sec": self._state.init_arm_stability_dwell_sec,
                "init_arm_qpos_active": self._state.init_arm_qpos_active,
                "init_arm_qpos_phase": self._state.init_arm_qpos_phase,
                "init_arm_qpos_step_index": self._state.init_arm_qpos_step_index,
                "init_arm_qpos_step_count": self._state.init_arm_qpos_step_count,
                "init_arm_qpos_max_error_deg": self._state.init_arm_qpos_max_error_deg,
                "init_arm_qpos_move_available": self._state.init_arm_qpos_move_available,
                "startup_qpos_maintenance_enabled": self._state.startup_qpos_maintenance_enabled,
                "init_arm_sequence_active": self._state.init_arm_sequence_active,
                "init_arm_sequence_side": self._state.init_arm_sequence_side,
                "init_arm_sequence_phase": self._state.init_arm_sequence_phase,
                "init_arm_sequence_status": self._state.init_arm_sequence_status,
                "fix_hold_active": self._state.fix_hold_active,
                "two_pedal_alert": dict(self._state.two_pedal_alert),
                "pose_snapshot_available": self._state.pose_snapshot_available,
                "pose_snapshot_dir": self._state.pose_snapshot_dir,
                "pose_snapshot_count": self._state.pose_snapshot_count,
                "pose_snapshot_last_path": self._state.pose_snapshot_last_path,
                "pose_snapshot_status": self._state.pose_snapshot_status,
                "active_controls": list(self._state.active_controls),
                "control_hint": self._state.control_hint,
                "capture_stage": self._state.capture_stage,
                "capture_phase": self._state.capture_phase,
                "capturing": self._state.capturing,
                "capture_channel": self._state.capture_channel,
                "capture_samples": self._state.capture_samples,
                "capture_left_samples": self._state.capture_left_samples,
                "capture_right_samples": self._state.capture_right_samples,
                "capture_min_samples": self._state.capture_min_samples,
                "capture_valid": self._state.capture_valid,
                "capture_left_valid": self._state.capture_left_valid,
                "capture_right_valid": self._state.capture_right_valid,
                "capture_elapsed_sec": self._state.capture_elapsed_sec,
                "capture_duration_sec": self._state.capture_duration_sec,
                "capture_prep_remaining_sec": self._state.capture_prep_remaining_sec,
                "capture_prep_total_sec": self._state.capture_prep_total_sec,
                "capture_waiting_for_samples": self._state.capture_waiting_for_samples,
                "shutdown": self._state.shutdown,
                "shutdown_message": self._state.shutdown_message,
                "messages": list(self._state.messages),
            }
        snapshot["runtime_home_diagnostic"] = _runtime_home_diagnostic_payload(snapshot)
        return snapshot


class WebUiKeyInput:
    def __init__(self, ui: CalibrationWebUi) -> None:
        self.ui = ui
        self.enabled = True
        self.should_quit = False
        self._events: list[str] = []

    def start(self) -> None:
        self.ui.start()

    def close(self) -> None:
        return

    def poll(self) -> None:
        events = self.ui.consume_events()
        if not events:
            return
        self._events.extend(events)
        if "q" in events:
            self.should_quit = True

    def consume_events(self) -> list[str]:
        events = list(self._events)
        self._events.clear()
        return events
