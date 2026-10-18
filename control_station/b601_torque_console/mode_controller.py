"""Pre-start GUI selection of existing controllers, with one dispatch consumer."""

from __future__ import annotations

import math
import threading
from collections import deque

from robot_motion.rebot_b601_rs.calibration import finite

from .calibration_cycle import B601AutomaticCalibrationCycleController
from .workspace_calibration import B601WorkspaceCalibrationController


class B601CalibrationModeController:
    controller_mode = "automatic_calibration"

    def __init__(self, supported: B601AutomaticCalibrationCycleController, workspace=None) -> None:
        if not isinstance(supported, B601AutomaticCalibrationCycleController):
            raise TypeError("GUI modes require the existing native calibration cycle")
        if workspace is not None and not isinstance(workspace, B601WorkspaceCalibrationController):
            raise TypeError("workspace mode requires the native J2/J3 acquisition controller")
        self._supported = supported
        self._workspace = workspace
        self._fixed = supported.calibration
        # The owner admits both choices before CAN connection. The selected
        # child still enforces its own narrower motion envelope and budget.
        self.settings, self.limits = supported.settings, supported.limits
        self.config = supported.config
        self._selected = "fixed_position"
        self._locked = self._closed = False
        self._generation = 0
        self._selected_at = None
        self._pending_delegate = None
        self._pending_generation = None
        self._mode_ready = False
        self._base_kd = None
        self._records = deque(maxlen=32)
        self._lock = threading.RLock()

    def _delegate(self):
        if self._selected == "workspace":
            return self._workspace
        return self._supported if self._selected == "supported_cycle" else self._fixed

    def _dispatch_delegate(self):
        return self._pending_delegate or self._delegate()

    def _selection(self):
        return {"selected": self._selected, "locked": self._locked or self._closed,
                "ready": self._mode_ready and not self._closed,
                "generation": self._generation, "selected_at_sec": self._selected_at,
                "available_modes": ["fixed_position", "supported_cycle"] + (["workspace"] if self._workspace is not None else []),
                "cycle_angle_deg": math.degrees(self.cycle.offset_rad),
                "maximum_cycle_angle_deg": math.degrees(self._supported.maximum_offset_rad)}

    @property
    def cycle(self):
        return self._supported.cycle

    @property
    def workspace_configuration(self):
        return None if self._workspace is None else self._workspace.workspace

    @property
    def cycle_disable_confirmation_ready(self) -> bool:
        with self._lock:
            if self._selected == "workspace":
                return self._locked and self._workspace.cycle_disable_confirmation_ready
            return (self._locked and self._selected == "supported_cycle"
                    and self._supported.cycle_disable_confirmation_ready)

    @property
    def cycle_disable_confirmation_required(self) -> bool:
        with self._lock:
            if self._selected == "workspace":
                return True
            return (self._selected == "supported_cycle"
                    and self._supported.cycle.supported_origin_disable_confirmation)

    def configure_context(self, condition, position_limits_rad) -> None:
        with self._lock:
            self._supported.configure_context(condition, position_limits_rad)
            if self._workspace is not None:
                self._workspace.configure_context(condition, position_limits_rad)
            self._base_kd = condition.driver_kd_nm_s_per_rad

    def validate_position_limits(self, position_limits_rad) -> None:
        if self._workspace is not None:
            self._workspace.validate_position_limits(position_limits_rad)

    def owner_ready(self) -> None:
        with self._lock:
            self._supported.owner_ready()
            if self._workspace is not None:
                self._workspace.owner_ready()

    def set_human_intervention(self, human: bool) -> None:
        with self._lock:
            self._supported.set_human_intervention(human)
            if self._workspace is not None:
                self._workspace.set_human_intervention(human)

    def submit(self, action: str, *, now_sec: float, **fields) -> None:
        now = finite(now_sec, "intent time")
        with self._lock:
            if action == "select_mode":
                if set(fields) != {"operation_mode"} or not isinstance(fields.get("operation_mode"), str) or fields["operation_mode"] not in {
                    "fixed_position", "supported_cycle", "workspace"
                }:
                    raise ValueError("choose fixed_position, supported_cycle or workspace")
                if fields["operation_mode"] == "workspace" and self._workspace is None:
                    raise ValueError("workspace acquisition was not configured before connection")
                if self._locked or self._closed:
                    raise ValueError("mode is locked for this batch; withdraw or stop without switching mode")
                selected = fields["operation_mode"]
                if selected != self._selected:
                    self._supported.set_background_mode_enabled(
                        selected == "supported_cycle"
                    )
                    self.settings, self.limits = (
                        self._workspace.settings if selected == "workspace" else self._supported.settings,
                        self._workspace.limits if selected == "workspace" else self._supported.limits,
                    )
                    self._selected = selected
                    self._generation += 1
                    self._selected_at = now
                    self._mode_ready = False
                    self._records.append({
                        "kind": "automatic_calibration_mode_selected",
                        "recorded_monotonic_sec": now, "operation_mode": selected,
                        "generation": self._generation,
                    })
                return
            if action == "arm" and self._closed:
                raise ValueError("owner has stopped; mode selection cannot restart it")
            if action == "arm":
                if (fields.get("operation_mode") != self._selected
                    or set(fields) - {"operation_mode", "cycle_angle_deg"}
                    or "cycle_angle_deg" in fields and self._selected != "supported_cycle"):
                    raise ValueError("displayed mode changed; refresh the GUI before starting")
                if not self._mode_ready:
                    raise ValueError("waiting for the owner's zero-assist cycle to confirm this mode")
                fields = {k: v for k, v in fields.items() if k != "operation_mode"}
            self._delegate().submit(action, now_sec=now, **fields)
            if action == "arm":
                self._locked = True

    def step(self, **values):
        with self._lock:
            if self._pending_delegate is not None:
                raise ValueError("previous six-axis dispatch must be confirmed before the next step")
            delegate = self._delegate()
            result = delegate.step(**values)
            # Selection can change between owner calls while both choices are
            # idle. Pin THIS transaction: do not send its receipt to a newly
            # selected child, even if a GUI click races the native dispatch.
            self._pending_delegate = delegate
            self._pending_generation = self._generation
            return result

    def validate_final_command(self, **values) -> None:
        with self._lock:
            self._dispatch_delegate().validate_final_command(**values)

    def active_driver_damping(self, base_kd, *, final_command_delta_nm=None):
        with self._lock:
            return self._dispatch_delegate().active_driver_damping(
                base_kd, final_command_delta_nm=final_command_delta_nm)

    def validate_dispatch_start(self, **values) -> None:
        with self._lock:
            self._dispatch_delegate().validate_dispatch_start(**values)

    def record_dispatch(self, **values) -> None:
        with self._lock:
            if self._pending_delegate is None:
                raise ValueError("mode controller has no pending dispatch")
            self._pending_delegate.record_dispatch(**values)
            if (not self._locked and self._pending_delegate is self._delegate()
                and self._pending_generation == self._generation):
                commands = values["joint_commands"]
                self._mode_ready = (
                    all(v == 0 for v in commands["overlay_nm"])
                    and all(abs(t-g) <= 1e-9 for t, g in zip(
                        commands["total_nm"], commands["gravity_nm"], strict=True))
                    and tuple(commands["driver_kd_nm_s_per_rad"]) == self._base_kd
                )
            self._pending_delegate = None
            self._pending_generation = None

    def status(self):
        with self._lock:
            return {**self._delegate().status(), "mode_selection": self._selection()}

    def result_record(self):
        with self._lock:
            return {**self._delegate().result_record(), "mode_selection": self._selection()}

    def drain_records(self):
        with self._lock:
            records = list(self._records)
            self._records.clear()
            return [*records, *self._delegate().drain_records()]

    def recording_failed(self, reason: str) -> None:
        with self._lock:
            self._delegate().recording_failed(reason)

    def owner_stopped(self, **values) -> None:
        with self._lock:
            self._closed = True
            self._delegate().owner_stopped(**values)
