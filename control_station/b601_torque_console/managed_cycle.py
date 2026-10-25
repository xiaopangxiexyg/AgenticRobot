"""One explicitly selected workspace batch through the existing intent port."""

from __future__ import annotations

from robot_motion.rebot_b601_rs.calibration import finite


class B601ManagedCycle:
    """No devices, thread, file I/O, trajectory, retry or shutdown authority.

    The hardware owner calls tick AFTER a consumed dispatch. The existing
    workspace controller owns references, budgets, return admission and final
    disable readiness. An external declaration is recorded before construction.
    """

    def __init__(self, controller, *, now_sec):
        workspace = getattr(controller, "workspace_configuration", None)
        if workspace is None or workspace.coordinated_holding is None:
            raise ValueError("managed cycle requires the coordinated workspace profile")
        self.controller = controller
        timeout = workspace.prepare_timeout_sec
        self.prepare_timeout_sec = controller.config.prepare_timeout_sec if timeout is None else timeout
        self.phase = "waiting_owner"
        self.reason = "waiting_for_active_zero_dispatch_and_supported_origin"
        self.started_at_sec = None
        self.last_tick_sec = finite(now_sec, "managed construction time")
        self.active_since_sec = None
        self.last_permit_sec = None
        self.start_count = self.heartbeat_count = 0
        controller.submit("select_mode", now_sec=self.last_tick_sec, operation_mode="workspace")

    def tick(self, *, now_sec, owner_active):
        now = finite(now_sec, "managed tick")
        if type(owner_active) is not bool or now < self.last_tick_sec:
            raise ValueError("managed cycle requires an advancing owner clock")
        if self.phase in {"origin_confirmed", "cancelled", "operator_required", "stopped"}:
            return None
        # Final readiness can be consumed by the owner in this very frame.
        # Do not leave the last management record at "running" because the
        # periodic observer has not reached its next 100 ms tick yet.
        if self.start_count and self.controller.automatic_disable_ready:
            self.phase, self.reason = "origin_confirmed", "existing_owner_disable_protocol_ready"
            return None
        if now-self.last_tick_sec < .1:
            return None
        self.last_tick_sec = now
        if not owner_active:
            return None
        if self.active_since_sec is None:
            self.active_since_sec = now
        status = self.controller.status()
        batch = status["automatic_calibration"]
        if status.get("fault") or batch.get("human_intervention") is not False:
            self.phase, self.reason = "operator_required", status.get("fault") or "human_intervention_or_unknown"
            return None
        if not self.start_count:
            selection = status["mode_selection"]
            if selection["locked"] or batch["state"] != "idle":
                self.phase, self.reason = "operator_required", "external_start_or_mode_change"
                return None
            origin = batch.get("workspace", {}).get("supported_return", {}).get("supported_origin_qpos_rad")
            if status.get("owner_ready") and selection["ready"] and origin is not None:
                self.controller.submit("arm", now_sec=now, operation_mode="workspace", return_and_disable=True)
                self.start_count = 1
                self.started_at_sec = self.last_permit_sec = now
                self.phase, self.reason = "running", "single_workspace_batch_started"
            elif now-self.active_since_sec > self.prepare_timeout_sec:
                self.phase, self.reason = "operator_required", "managed_start_preparation_timeout"
                return "protective_hold"
            return None
        if batch["active"]:
            # A soft stop's admitted return remains active. No declaration is
            # reasserted; GUI intervention/withdrawal still wins permanently.
            self.controller.submit("heartbeat", now_sec=now)
            self.last_permit_sec = now
            self.heartbeat_count += 1
        else:
            self.phase = "cancelled" if batch["state"] == "cancelled" else "stopped"
            self.reason = batch["reason"]
        return None

    def result_record(self):
        return {"enabled": True, "phase": self.phase, "reason": self.reason,
                "start_count": self.start_count, "max_cycles": 1,
                "started_at_sec": self.started_at_sec, "last_permit_sec": self.last_permit_sec,
                "heartbeat_count": self.heartbeat_count,
                "prepare_timeout_sec": self.prepare_timeout_sec,
                "declaration_source": "explicit_managed_cycle_configuration",
                "human_contact_sensor": False,
                "return_and_disable_owner": "existing_coordinated_workspace_and_hardware_owner",
                "automatic_retry": False}


def assess_managed_result(payload, report=None):
    """Assess closed receipts; exit zero alone is never acquisition success."""
    cleanup = payload.get("cleanup", {})
    batch = payload.get("automatic_calibration", {})
    returned = batch.get("workspace", {}).get("supported_return", {})
    disable_cleanup = (all(cleanup.get(key) is True for key in (
            "device_cleanup_confirmed", "motor_disable_confirmed",
            "active_report_disable_confirmed", "driver_limits_restore_confirmed"))
        and cleanup.get("transport_release", {}).get("completed") is True
        and not cleanup.get("errors") and not payload.get("cleanup_errors"))
    lifecycle = (returned.get("completed") is True
        and payload.get("cycle_disable_confirmation", {}).get("confirmed") is True and disable_cleanup)
    complete = batch.get("state") == "complete" and batch.get("outcome", {}).get("action") == "complete"
    if report is not None:
        complete = complete and report.get("acquisition_complete") is True
    execution_error = payload.get("execution_status") in {"ERROR", "FAILED"} or payload.get("status") in {"ERROR", "FAIL"}
    report_error = payload.get("automatic_calibration_report_status") in {"ERROR", "INTERRUPTED"}
    status = ("OPERATOR_REQUIRED" if not disable_cleanup else "EXECUTION_ERROR" if execution_error
              else "RETURN_INCOMPLETE" if not lifecycle
              else "REPORT_ERROR" if report_error else "COMPLETE" if complete else "PARTIAL_ACQUISITION")
    return {"status": status,
            "lifecycle_complete": lifecycle, "acquisition_complete": complete,
            "return_complete": returned.get("completed") is True,
            "disable_cleanup_confirmed": bool(disable_cleanup),
            "acquisition_reason": batch.get("reason", payload.get("stop_reason", payload.get("error"))),
            "execution_status": payload.get("execution_status"),
            "report_status": payload.get("automatic_calibration_report_status"),
            "model_readiness": None if report is None else report.get("model_readiness"),
            "gravity_or_friction_profile_updated": False,
            "command_field_mode": batch.get("workspace", {}).get("command_field_mode", "off"),
            "automatic_retry": False}
