"""Control Station binding for the shared VIVE calibration workflow.

This module intentionally contains application lifecycle and presentation
state only.  Capture timing, fresh-sample handling, axis fitting, and tracker
offset math are provided by
``teleop_runtime.real_teleop.vive_calibration_runtime``.  No robot, simulator,
authority, safety, or dispatch object is created here.
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from project_paths import control_station_vive_coordinate_profiles_dir
from teleop_runtime.real_teleop.vive_calibration_runtime import (
    ViveCoordinateCalibrationContext,
    calibrate_vive_coordinate_profile,
)
from teleop_runtime.real_teleop.side_mapping import build_vive_tracker_configs_for_args
from teleop_runtime.sources.pose import VivePoseSource
from teleop_runtime.transforms import as_pose_matrix

from .vive_calibration_profiles import (
    VIVE_COORDINATE_PROFILE_KIND,
    coordinate_profile_view,
    list_coordinate_profiles,
    load_coordinate_profile,
    new_coordinate_calibration,
    normalize_active_sides,
    save_coordinate_profile,
)


_DEFAULT_VIVE_PORT = 9999
_DEFAULT_CLOCK_SYNC_PORT = 10001
_STAGES = {
    "virtual_desk_x": "VIVE 坐标系 X 轴",
    "virtual_desk_y": "VIVE 坐标系 Y 轴",
    "virtual_desk_z": "VIVE 坐标系 Z 轴",
    "wrist_offsets": "Wrist tracker offset",
    "completed": "标定完成",
    "view": "查看 profile",
    "error": "当前阶段失败",
    "idle": "未开始",
}
_CAPTURE_STAGE_TO_STAGE = {
    "VirtualDeskX": "virtual_desk_x",
    "VirtualDeskY": "virtual_desk_y",
    "VirtualDeskZ": "virtual_desk_z",
    "WristOffset": "wrist_offsets",
}


def _finite_pose(value: object) -> object | None:
    try:
        pose = as_pose_matrix(value)
        if pose.shape != (4, 4):
            return None
        flattened = [float(item) for row in pose.tolist() for item in row]
        if not all(math.isfinite(item) for item in flattened):
            return None
        return pose
    except (TypeError, ValueError):
        return None


def _channel_view(channel: object | None, *, now: float) -> dict[str, Any]:
    if channel is None:
        return {"present": False, "valid": False, "age_sec": None}
    pose = _finite_pose(getattr(channel, "value", None))
    metadata = getattr(channel, "metadata", {}) or {}
    timestamp = metadata.get("receiver_timestamp", getattr(channel, "timestamp", None))
    age = None if timestamp is None else max(0.0, float(now) - float(timestamp))
    return {
        "present": True,
        "valid": bool(getattr(channel, "valid", False)) and pose is not None,
        "critical": bool(getattr(channel, "critical", False)),
        "channel_name": str(getattr(channel, "channel_name", "")),
        "frame_name": str(getattr(channel, "frame_name", "")),
        "tracker_name": str(metadata.get("tracker_name") or ""),
        "tracker_role": str(metadata.get("tracker_role") or ""),
        "seq": metadata.get("seq"),
        "age_sec": None if age is None else round(age, 3),
        "xyz_m": None
        if pose is None
        else [round(float(value), 4) for value in pose[:3, 3].tolist()],
    }


@dataclass
class ViveCoordinateCalibrationConfig:
    """Settings for a VIVE-only Control Station calibration session."""

    active_sides: tuple[str, ...] = ("left", "right")
    desk_tracker: str = "auto"
    vive_host: str = "0.0.0.0"
    vive_port: int = _DEFAULT_VIVE_PORT
    vive_clock_sync_port: int | None = _DEFAULT_CLOCK_SYNC_PORT
    left_tracker_name: str = "tracker_1"
    right_tracker_name: str = "tracker_2"
    auxiliary_tracker_name: str = ""
    sweep_capture_sec: float = 5.0
    capture_prep_sec: float = 3.0
    hold_capture_sec: float = 4.0
    capture_extra_wait_sec: float = 2.0
    min_sweep_samples: int = 30
    min_hold_samples: int = 30
    min_sweep_extent_m: float = 0.15
    min_axis_variance_ratio: float = 0.5
    min_start_end_distance_m: float = 0.08
    max_axis_correction_deg: float = 20.0
    virtual_desk_post_yaw_deg: float = 180.0
    profile_dir: Path | None = None

    def __post_init__(self) -> None:
        self.active_sides = normalize_active_sides(self.active_sides)
        self.desk_tracker = str(self.desk_tracker or "auto").strip().lower()
        if self.desk_tracker not in {"auto", "left", "right", "auxiliary"}:
            raise ValueError("desk_tracker must be auto, left, right, or auxiliary.")
        if self.desk_tracker == "auxiliary" and not str(self.auxiliary_tracker_name or "").strip():
            raise ValueError("desk_tracker=auxiliary requires auxiliary_tracker_name.")
        if not 0 <= int(self.vive_port) <= 65535:
            raise ValueError("vive_port must be in [0, 65535].")
        self.vive_port = int(self.vive_port)
        if self.vive_clock_sync_port is not None:
            self.vive_clock_sync_port = int(self.vive_clock_sync_port)
            if not 0 <= self.vive_clock_sync_port <= 65535:
                raise ValueError("vive_clock_sync_port must be in [0, 65535].")
            if self.vive_clock_sync_port and self.vive_clock_sync_port == self.vive_port:
                raise ValueError("VIVE clock sync port must differ from the pose port.")
        for field_name in (
            "sweep_capture_sec",
            "capture_prep_sec",
            "hold_capture_sec",
            "capture_extra_wait_sec",
            "min_sweep_extent_m",
            "min_axis_variance_ratio",
            "min_start_end_distance_m",
            "max_axis_correction_deg",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if float(self.sweep_capture_sec) <= 0.0 or float(self.hold_capture_sec) <= 0.0:
            raise ValueError("capture windows must be positive.")
        if int(self.min_sweep_samples) < 3 or int(self.min_hold_samples) < 3:
            raise ValueError("minimum sample counts must be at least 3.")
        self.min_sweep_samples = int(self.min_sweep_samples)
        self.min_hold_samples = int(self.min_hold_samples)
        self.profile_dir = (
            control_station_vive_coordinate_profiles_dir()
            if self.profile_dir is None
            else Path(self.profile_dir)
        )


class ViveCoordinateCalibrationController:
    """Lifecycle and UI adapter around the existing calibration workflow."""

    def __init__(
        self,
        config: ViveCoordinateCalibrationConfig | None = None,
        *,
        source: VivePoseSource | object | None = None,
    ) -> None:
        self.config = config or ViveCoordinateCalibrationConfig()
        self._source = source or self._build_source()
        self._lock = threading.RLock()
        self._workflow_events: queue.Queue[str] = queue.Queue()
        self._workflow_thread: threading.Thread | None = None
        self._source_started = False
        self._started = False
        self._draft = self._new_draft()
        self._stage = "idle"
        self._phase = "idle"
        self._message = "等待开始。"
        self._error = ""
        self._current_profile_path = ""
        self._last_saved_profile_path = ""
        self._profiles_revision = 0
        self._ui_state: dict[str, Any] = {}

    @property
    def source(self) -> object:
        return self._source

    @property
    def profile_dir(self) -> Path:
        return Path(self.config.profile_dir).expanduser().resolve()

    def _build_source(self) -> VivePoseSource:
        tracker_configs = build_vive_tracker_configs_for_args(
            SimpleNamespace(
                left_tracker_name=self.config.left_tracker_name,
                right_tracker_name=self.config.right_tracker_name,
                aux_tracker_name=self.config.auxiliary_tracker_name,
                swap_operator_sides=False,
            ),
            ("left", "right"),
        )
        return VivePoseSource(
            tracker_configs=tracker_configs,
            host=self.config.vive_host,
            port=self.config.vive_port,
            clock_sync_port=self.config.vive_clock_sync_port,
        )

    def _new_draft(self):
        return new_coordinate_calibration(
            active_sides=self.config.active_sides,
            left_tracker_name=self.config.left_tracker_name,
            right_tracker_name=self.config.right_tracker_name,
            auxiliary_tracker_name=self.config.auxiliary_tracker_name,
            desk_tracker=self.config.desk_tracker,
        )

    def start(self, *, start_source: bool = True) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        if start_source:
            start = getattr(self._source, "start", None)
            if callable(start):
                start()
            self._source_started = True

    def close(self) -> None:
        self._workflow_events.put("q")
        thread = self._workflow_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._workflow_thread = None
        if self._source_started:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
            self._source_started = False
        self._started = False

    # The shared workflow consumes this intentionally tiny UI port.
    def consume_events(self, *, max_events: int = 64) -> list[str]:
        events: list[str] = []
        for _ in range(max_events):
            try:
                events.append(self._workflow_events.get_nowait())
            except queue.Empty:
                break
        return events

    def append_message(self, tag: str, message: str) -> None:
        with self._lock:
            self._message = str(message)
            self._ui_state["last_message_tag"] = str(tag)

    def publish(self, **updates: Any) -> None:
        with self._lock:
            self._ui_state.update(updates)
            capture_stage = str(updates.get("capture_stage") or "")
            if capture_stage in _CAPTURE_STAGE_TO_STAGE:
                self._stage = _CAPTURE_STAGE_TO_STAGE[capture_stage]
            if "stage" in updates and capture_stage not in _CAPTURE_STAGE_TO_STAGE:
                stage = str(updates["stage"])
                self._stage = _CAPTURE_STAGE_TO_STAGE.get(stage, self._stage)
            if "capture_phase" in updates:
                self._phase = str(updates["capture_phase"])
            if "control_hint" in updates and updates["control_hint"]:
                self._message = str(updates["control_hint"])

    def dispatch(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        event_name = str(event or "").strip()
        if not event_name:
            raise ValueError("event must be non-empty")
        values = dict(payload or {})
        if event_name == "begin_calibration":
            self._begin_workflow()
        elif event_name == "capture":
            self._workflow_events.put("n")
        elif event_name == "reset_stage":
            self._workflow_events.put("r")
        elif event_name == "set_active_sides":
            self._set_active_sides(values.get("active_sides", "both"))
        elif event_name == "set_desk_tracker":
            self._set_desk_tracker(values.get("desk_tracker", "auto"))
        elif event_name == "save_profile":
            self._save_profile(str(values.get("profile_name", "") or ""))
        elif event_name == "load_profile":
            self._load_profile(str(values.get("profile_name", "") or ""))
        elif event_name == "refresh_profiles":
            self._profiles_revision += 1
        elif event_name == "close":
            self._workflow_events.put("q")
        else:
            raise ValueError(f"Unsupported VIVE calibration GUI event: {event_name}")

    def _set_active_sides(self, value: object) -> None:
        sides = normalize_active_sides(value if isinstance(value, str) else tuple(value))
        if self._workflow_thread is not None and self._workflow_thread.is_alive():
            raise RuntimeError("Cannot change active sides during capture.")
        with self._lock:
            self.config.active_sides = sides
            self._draft = self._new_draft()
            self._stage, self._phase = "idle", "idle"
            self._message, self._error = f"已选择 {','.join(sides)} 标定。", ""

    def _set_desk_tracker(self, value: object) -> None:
        tracker = str(value or "auto").strip().lower()
        if tracker not in {"auto", "left", "right", "auxiliary"}:
            raise ValueError("desk_tracker must be auto, left, right, or auxiliary.")
        if tracker == "auxiliary" and not str(self.config.auxiliary_tracker_name or "").strip():
            raise ValueError("desk_tracker=auxiliary requires auxiliary_tracker_name.")
        if self._workflow_thread is not None and self._workflow_thread.is_alive():
            raise RuntimeError("Cannot change desk tracker during capture.")
        with self._lock:
            self.config.desk_tracker = tracker
            self._draft = self._new_draft()
            self._stage, self._phase = "idle", "idle"
            self._message, self._error = f"坐标系 sweep tracker 已设置为 {tracker}。", ""

    def _workflow_tracker_role(self) -> str:
        tracker = self.config.desk_tracker
        if tracker == "auxiliary":
            return "auxiliary"
        if tracker in {"left", "right"}:
            return f"{tracker}_wrist"
        if str(self.config.auxiliary_tracker_name or "").strip():
            return "auxiliary"
        return "right_wrist" if "right" in self.config.active_sides else "left_wrist"

    def _workflow_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            hz=50.0,
            virtual_desk_tracker_role=self._workflow_tracker_role(),
            aux_tracker_name=self.config.auxiliary_tracker_name,
            capture_prep_sec=self.config.capture_prep_sec,
            wrist_capture_prep_sec=self.config.capture_prep_sec,
            capture_extra_wait_sec=self.config.capture_extra_wait_sec,
            sweep_capture_sec=self.config.sweep_capture_sec,
            hold_capture_sec=self.config.hold_capture_sec,
            min_sweep_samples=self.config.min_sweep_samples,
            min_hold_samples=self.config.min_hold_samples,
            min_sweep_extent_m=self.config.min_sweep_extent_m,
            min_axis_variance_ratio=self.config.min_axis_variance_ratio,
            min_start_end_distance_m=self.config.min_start_end_distance_m,
            max_axis_non_orthogonality_deg=self.config.max_axis_correction_deg,
            virtual_desk_post_yaw_deg=self.config.virtual_desk_post_yaw_deg,
        )

    def _begin_workflow(self) -> None:
        if self._workflow_thread is not None and self._workflow_thread.is_alive():
            raise RuntimeError("VIVE calibration is already running.")
        with self._lock:
            self._draft = self._new_draft()
            self._stage, self._phase = "virtual_desk_x", "ready"
            self._message, self._error = "请准备并采集 VIVE 坐标系 X 轴。", ""
            self._current_profile_path = ""
            self._ui_state = {}
        while True:
            try:
                self._workflow_events.get_nowait()
            except queue.Empty:
                break
        context = ViveCoordinateCalibrationContext(
            source=self._source,
            calibration=self._draft,
            active_sides=self.config.active_sides,
        )
        args = self._workflow_args()
        self._workflow_thread = threading.Thread(
            target=self._run_workflow,
            args=(args, context),
            daemon=True,
            name="control_station_vive_calibration_workflow",
        )
        self._workflow_thread.start()

    def _run_workflow(
        self,
        args: argparse.Namespace,
        context: ViveCoordinateCalibrationContext,
    ) -> None:
        try:
            calibrate_vive_coordinate_profile(
                args,
                context,
                self,
                save_profile=None,
                save_on_complete=False,
            )
            with self._lock:
                self._stage, self._phase = "completed", "completed"
                self._message = "标定完成。请查看结果并保存为 Control Station 独立 profile。"
                self._draft.metadata["profile_ready"] = True
        except KeyboardInterrupt:
            with self._lock:
                self._stage, self._phase = "idle", "idle"
                self._message = "标定已取消。"
        except Exception as exc:  # Surface workflow failures in the GUI.
            with self._lock:
                self._stage, self._phase = "error", "error"
                self._error = f"{type(exc).__name__}: {exc}"
                self._message = self._error

    def _save_profile(self, profile_name: str) -> None:
        with self._lock:
            if self._stage != "completed":
                raise RuntimeError("请先完成坐标系和所选 wrist offset 标定。")
            path = save_coordinate_profile(
                self._draft,
                root=self.profile_dir,
                profile_name=profile_name,
            )
            self._current_profile_path = str(path)
            self._last_saved_profile_path = str(path)
            self._profiles_revision += 1
            self._message = f"已保存 Control Station 独立 profile：{path}"

    def _load_profile(self, profile_name: str) -> None:
        if not profile_name:
            raise ValueError("profile_name must be non-empty")
        calibration, path = load_coordinate_profile(profile_name, root=self.profile_dir)
        with self._lock:
            self._draft = calibration
            self.config.active_sides = normalize_active_sides(
                calibration.metadata.get("active_sides", "both")
            )
            self.config.desk_tracker = str(calibration.metadata.get("desk_tracker", "auto"))
            self._stage, self._phase = "view", "completed"
            self._current_profile_path = str(path)
            self._message = f"已加载 profile：{path.name}。当前仅查看，不会启动执行链。"
            self._error = ""

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            profile = {
                "profile_kind": VIVE_COORDINATE_PROFILE_KIND,
                "active_sides": list(self.config.active_sides),
            }
            try:
                profile = coordinate_profile_view(self._draft)
            except Exception:
                pass
            try:
                channels = self._source.get_latest_channels()
            except Exception:
                channels = {}
            try:
                source_stats = self._source.debug_stats()
            except Exception:
                source_stats = {}
            capture_samples = self._ui_state.get("capture_samples")
            if capture_samples is None:
                capture_samples = {}
            if isinstance(capture_samples, int):
                capture_samples = {str(self._ui_state.get("capture_channel", "")): capture_samples}
            return {
                **self._ui_state,
                "application": "control_station_vive_coordinate_calibration",
                "profile_kind": VIVE_COORDINATE_PROFILE_KIND,
                "lifecycle": "running" if self._started else "stopped",
                "source_started": bool(self._source_started),
                "motion_authorized": False,
                "real_arm_motion": False,
                "active_sides": list(self.config.active_sides),
                "desk_tracker": self.config.desk_tracker,
                "stage": self._stage,
                "stage_label": _STAGES.get(self._stage, self._stage),
                "phase": self._phase,
                "message": self._message,
                "error": self._error,
                "capture_samples": capture_samples,
                "capture_min_samples": self._ui_state.get("capture_min_samples"),
                "capture_started_at": None,
                "capture_deadline": None,
                "capture_elapsed_sec": self._ui_state.get("capture_elapsed_sec"),
                "capture_duration_sec": self._ui_state.get("capture_duration_sec"),
                "capture_prep_remaining_sec": self._ui_state.get("capture_prep_remaining_sec"),
                "live_channels": {
                    name: _channel_view(channels.get(name), now=now)
                    for name in (
                        "left_wrist_pose",
                        "right_wrist_pose",
                        "auxiliary_tracker_pose",
                    )
                },
                "source_stats": source_stats,
                "profile_dir": str(self.profile_dir),
                "current_profile_path": self._current_profile_path,
                "last_saved_profile_path": self._last_saved_profile_path,
                "profile": profile,
                "profiles_revision": self._profiles_revision,
                "profiles": [item.to_dict() for item in list_coordinate_profiles(self.profile_dir)],
                "no_dispatch_owner": True,
            }


__all__ = ["ViveCoordinateCalibrationConfig", "ViveCoordinateCalibrationController"]
