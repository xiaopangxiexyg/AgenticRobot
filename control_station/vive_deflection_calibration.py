"""Shared-source calibration launcher for the standalone VIVE viewer.

The launcher adapts the viewer's single ``wrist_pose`` channel to the channel
name expected by the existing coordinate-calibration workflow.  It never
starts or closes that shared source; the viewer remains its sole lifecycle
owner.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .vive_calibration import (
    ViveCoordinateCalibrationConfig,
    ViveCoordinateCalibrationController,
)
from .vive_calibration_gui import ViveCoordinateCalibrationGui


class _SingleRoleCalibrationSource:
    """Expose one viewer channel under the calibration workflow's naming."""

    def __init__(self, source: Any, *, source_role: str) -> None:
        self._source = source
        self.source_role = str(source_role)
        self.output_channel = f"{self.source_role}_pose"

    def get_latest_channels(self) -> dict[str, Any]:
        channels = self._source.get_latest_channels()
        if not isinstance(channels, Mapping):
            return {}
        channel = channels.get("wrist_pose")
        return {} if channel is None else {self.output_channel: channel}

    def debug_stats(self) -> dict[str, Any]:
        stats = getattr(self._source, "debug_stats", None)
        return dict(stats()) if callable(stats) else {}

    @property
    def source_role_by_output_channel(self) -> dict[str, str]:
        return {self.output_channel: self.source_role}

    @property
    def available_active_sides(self) -> tuple[str, ...]:
        return (self.source_role.removesuffix("_wrist"),)


class ViveDeflectionCalibrationLauncher:
    """Lazily open the existing calibration GUI over the viewer source."""

    def __init__(
        self,
        *,
        source: Any,
        source_role: str,
        tracker_name: str,
        ui_host: str = "127.0.0.1",
        ui_port: int = 0,
        profile_dir: Path | None = None,
        profile_applier: Any | None = None,
    ) -> None:
        if source_role not in {"left_wrist", "right_wrist"}:
            raise ValueError("source_role must be left_wrist or right_wrist")
        self.source = source
        self.source_role = source_role
        self.tracker_name = str(tracker_name)
        self.ui_host = str(ui_host)
        self.ui_port = int(ui_port)
        self.profile_dir = profile_dir
        self.profile_applier = profile_applier
        self._lock = threading.RLock()
        self._controller: ViveCoordinateCalibrationController | None = None
        self._gui: ViveCoordinateCalibrationGui | None = None
        self._monitor: threading.Thread | None = None
        self._last_saved_profile_path: Path | None = None
        self._last_apply_attempted_path: Path | None = None
        self._last_applied_profile_path: Path | None = None
        self._last_apply_receipt: dict[str, Any] | None = None
        self._apply_error = ""

    def _build_session(
        self,
    ) -> tuple[ViveCoordinateCalibrationController, ViveCoordinateCalibrationGui]:
        side = self.source_role.removesuffix("_wrist")
        config = ViveCoordinateCalibrationConfig(
            active_sides=(side,),
            desk_tracker=side,
            left_tracker_name=(
                self.tracker_name if side == "left" else "tracker_1"
            ),
            right_tracker_name=(
                self.tracker_name if side == "right" else "tracker_2"
            ),
            profile_dir=self.profile_dir,
        )
        source = _SingleRoleCalibrationSource(
            self.source,
            source_role=self.source_role,
        )
        controller = ViveCoordinateCalibrationController(config, source=source)
        gui = ViveCoordinateCalibrationGui(
            controller,
            host=self.ui_host,
            port=self.ui_port,
            open_browser=False,
        )
        return controller, gui

    def open(self) -> dict[str, Any]:
        with self._lock:
            controller = self._controller
            gui = self._gui
            if (
                controller is not None
                and gui is not None
                and not controller.shutdown_requested
            ):
                return self.snapshot()
            self._close_session_locked()
            controller, gui = self._build_session()
            controller.start(start_source=False)
            try:
                gui.start()
            except BaseException:
                controller.close()
                raise
            self._controller = controller
            self._gui = gui
            monitor = threading.Thread(
                target=self._monitor_shutdown,
                args=(controller, gui),
                daemon=True,
                name="vive_deflection_calibration_monitor",
            )
            self._monitor = monitor
            monitor.start()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            controller = self._controller
            gui = self._gui
            active = bool(
                controller is not None
                and gui is not None
                and not controller.shutdown_requested
            )
            return {
                "available": True,
                "active": active,
                "url": gui.url if active and gui is not None else None,
                "source_role": self.source_role,
                "shared_source": True,
                "profile_dir": (
                    str(controller.profile_dir)
                    if controller is not None
                    else None
                ),
                "last_saved_profile_path": (
                    None
                    if self._last_saved_profile_path is None
                    else str(self._last_saved_profile_path)
                ),
                "last_applied_profile_path": (
                    None
                    if self._last_applied_profile_path is None
                    else str(self._last_applied_profile_path)
                ),
                "last_apply_receipt": self._last_apply_receipt,
                "apply_error": self._apply_error,
            }

    def _monitor_shutdown(
        self,
        controller: ViveCoordinateCalibrationController,
        gui: ViveCoordinateCalibrationGui,
    ) -> None:
        while True:
            with self._lock:
                if controller is not self._controller or gui is not self._gui:
                    return
            saved_path = controller.last_saved_profile_path
            if saved_path is not None:
                self._apply_saved_profile(controller, gui, saved_path)
            if controller.shutdown_requested:
                break
            time.sleep(0.2)
        with self._lock:
            if controller is self._controller and gui is self._gui:
                self._close_session_locked()

    def _apply_saved_profile(
        self,
        controller: ViveCoordinateCalibrationController,
        gui: ViveCoordinateCalibrationGui,
        path: Path,
    ) -> None:
        resolved = Path(path).expanduser().resolve()
        with self._lock:
            if controller is not self._controller or gui is not self._gui:
                return
            self._last_saved_profile_path = resolved
            if resolved == self._last_apply_attempted_path:
                return
            self._last_apply_attempted_path = resolved
            applier = self.profile_applier
        if applier is None:
            return
        try:
            receipt = dict(applier(resolved))
        except Exception as error:
            with self._lock:
                self._apply_error = f"{type(error).__name__}:{error}"
            return
        with self._lock:
            self._last_applied_profile_path = resolved
            self._last_apply_receipt = receipt
            self._apply_error = ""

    def _close_session_locked(self) -> None:
        gui, self._gui = self._gui, None
        controller, self._controller = self._controller, None
        self._monitor = None
        if gui is not None:
            gui.close()
        if controller is not None:
            controller.close()

    def close(self) -> None:
        with self._lock:
            self._close_session_locked()


__all__ = ["ViveDeflectionCalibrationLauncher"]
