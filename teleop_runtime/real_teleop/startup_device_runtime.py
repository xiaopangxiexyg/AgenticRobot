"""Device-start admission for the canonical real-teleop startup lifecycle.

This module owns direct one-shot ``backend.start()`` admission for resources
outside the Session Rig, including strict versus capability admission and its
progress/error receipts.  Rig-owned effectors are started by the Rig itself;
this module only supplies their bounded post-start recovery seam.  It does not
construct devices or own steady-state health.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
import time

from robot_motion.runtime.readiness import (
    format_readiness_report,
    readiness_report_blockers,
)


_DEVICE_LABELS = {
    "arm.left.qpos": "左臂状态反馈",
    "arm.right.qpos": "右臂状态反馈",
    "arm.force_guard": "双臂位控受力急停",
    "hand.left": "左手 / 左夹爪",
    "hand.right": "右手 / 右夹爪",
    "rig.session": "遥操核心 Rig",
    "human_shadow": "Human shadow 前端",
    "vive.clock_sync": "VIVE 时钟同步",
    "dataset.writer": "数采 writer 子进程",
    "gripper.input": "Q6 / 夹爪输入",
}

# A successful read-only USB preflight and the first runtime Modbus exchange
# are separate lifecycle boundaries.  The kernel/USB stack can still be
# settling after the preflight closes its probe handle, so give a DMTAC hand
# one short recovery attempt before publishing a capability Hold.  This is
# deliberately bounded and is not a general device retry loop.
DMTAC_STARTUP_RETRY_DELAY_SEC = 0.05


def startup_device_label(key: str) -> str:
    """Return the operator-facing startup label for one resource key."""

    normalized = str(key)
    return _DEVICE_LABELS.get(normalized, normalized)


def startup_device_requirement(key: str) -> str:
    """Classify the startup resource without deciding dataset admission."""

    normalized = str(key)
    if normalized.startswith("arm.") or normalized == "rig.session":
        return "motion_safety"
    if normalized.startswith("camera.") or normalized.startswith("dataset."):
        return "required_for_collection"
    return "feature_dependent"


def startup_gripper_input_errors(
    *,
    effective_source: str,
    q6_error: str,
    angle_trigger_error: str,
) -> dict[str, str]:
    """Project configured-source failures onto logical startup resources.

    A Q6 discovery failure describes the unavailable preferred source, not the
    effective gripper input after a healthy angle-trigger fallback is active.
    Keep both diagnoses when the fallback itself failed so startup admission
    and the operator still see the actual angle-trigger blocker.
    """

    source = str(effective_source or "")
    q6_detail = str(q6_error or "")
    angle_trigger_detail = str(angle_trigger_error or "")
    errors: dict[str, str] = {}
    if angle_trigger_detail:
        errors["angle_trigger"] = angle_trigger_detail
    if q6_detail and (source != "angle_trigger" or angle_trigger_detail):
        errors["gripper.input"] = q6_detail
    return errors


def ensure_session_rig_backend_binding(
    rig: Any,
    *,
    effector_name: str,
    backend: Any,
) -> bool:
    """Ensure one configured backend remains in the authoritative Session Rig.

    Capability admission may continue after a hand startup failure, but it
    must never alter the command topology.  A missing binding would turn a
    recoverable readiness fault into a later dispatch-time ``KeyError``.  The
    helper is intentionally small and side-effect limited to inserting the
    exact backend object when the Rig exposes its mutable backend map.
    """

    if backend is None:
        return False
    backends = getattr(rig, "backends", None)
    if not isinstance(backends, dict):
        raise RuntimeError(
            "Session Rig backend binding requires a mutable dict-like backend map."
        )
    name = str(effector_name)
    existing = backends.get(name)
    if existing is None:
        backends[name] = backend
        return True
    if existing is not backend:
        raise RuntimeError(
            f"Session Rig already owns a different backend for {name}"
        )
    return False


def inspect_session_rig_readiness(
    rig: Any | None,
) -> tuple[bool, dict[str, str], str]:
    """Return a bounded readiness receipt for startup admission.

    ``is_ready`` remains authoritative.  A backend-specific report is used
    when available so startup can preserve the actual serial endpoint and
    vendor failure instead of replacing it with a generic ``Rig is not
    ready`` error.  The function performs no reconnect or command dispatch.
    """

    if rig is None:
        return False, {"rig": "Session Rig is unavailable"}, "Session Rig is unavailable"

    report_fn = getattr(rig, "readiness_report", None)
    if callable(report_fn):
        try:
            report = report_fn()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not mask admission.
            detail = f"readiness report failed: {type(exc).__name__}: {exc}"
            return False, {"rig": detail}, detail
        if isinstance(report, Mapping):
            blockers = readiness_report_blockers(report)
            ready = bool(report.get("ready", False)) and not blockers
            if ready:
                return True, {}, ""
            detail = format_readiness_report(report)
            if not detail:
                detail = "rig is not ready"
            return False, blockers or {"rig": detail}, detail

    try:
        ready = bool(rig.is_ready())
    except Exception as exc:  # noqa: BLE001 - fail closed with a bounded reason.
        detail = f"readiness check failed: {type(exc).__name__}: {exc}"
        return False, {"rig": detail}, detail
    if ready:
        return True, {}, ""
    detail = "rig is not ready"
    return False, {"rig": detail}, detail


def startup_rig_effector_key(effector_name: str) -> str:
    """Map a Rig effector name to the generic startup status key."""

    normalized = str(effector_name)
    if normalized in {"left_arm", "right_arm"}:
        return f"arm.{normalized.split('_', 1)[0]}.qpos"
    if normalized in {"left_hand", "right_hand"}:
        return f"hand.{normalized.split('_', 1)[0]}"
    return f"rig.{normalized}"


class StartupDeviceRuntime:
    """Start configured backends and publish one deterministic admission receipt."""

    def __init__(
        self,
        *,
        capability_admission: bool,
        startup_progress: Any | None,
        startup_device_errors: dict[str, str],
        control_ui: Any | None,
        dmtac_gripper_backend_type: type,
        log: Callable[..., None],
    ) -> None:
        self._capability_admission = bool(capability_admission)
        self._startup_progress = startup_progress
        self._startup_device_errors = startup_device_errors
        self._control_ui = control_ui
        self._dmtac_gripper_backend_type = dmtac_gripper_backend_type
        self._log = log

    @staticmethod
    def label(key: str) -> str:
        return startup_device_label(key)

    @staticmethod
    def requirement(key: str) -> str:
        return startup_device_requirement(key)

    @staticmethod
    def _backend_transport_identity(backend: Any) -> str:
        """Return bounded endpoint details for a failed backend, if exposed."""

        config = getattr(backend, "config", None)
        if config is None:
            return ""
        values: list[str] = []
        for name, label in (
            ("gripper_port", "port"),
            ("gripper_id", "id"),
            ("gripper_baudrate", "baud"),
        ):
            value = getattr(config, name, None)
            if value in (None, ""):
                continue
            values.append(f"{label}={value}")
        return ", ".join(values)

    def _retry_dmtac_startup_once(
        self,
        key: str,
        backend: Any,
    ) -> tuple[bool, str]:
        """Perform one bounded DMTAC recovery after an unready startup.

        The DMTAC backend intentionally swallows transport errors from its
        first ``start`` so capability-mode GUI startup can remain available.
        A preflight probe may nevertheless leave a short serial settling
        window.  Recreate the exact backend object once, then require its
        normal readiness predicate again.  No arm/session state is changed and
        no command is dispatched here; a failed recovery remains a visible
        startup blocker for the explicit GUI retry path.
        """

        if not self._capability_admission or not str(key).startswith("hand."):
            return False, ""
        if not isinstance(backend, self._dmtac_gripper_backend_type):
            return False, ""
        reconnect = getattr(backend, "force_reconnect_gripper", None)
        if not callable(reconnect):
            # Keep lightweight compatibility/test backends on the historical
            # admission path; only the concrete DMTAC reconnect seam opts into
            # this automatic recovery.
            return False, ""

        delay = DMTAC_STARTUP_RETRY_DELAY_SEC
        config = getattr(backend, "config", None)
        # Reuse the already validated startup fault-check cadence when a
        # backend exposes it, but cap the automatic recovery wait so startup
        # never turns into an unbounded serial probe.
        configured_delay = getattr(config, "startup_fault_check_interval_sec", None)
        if configured_delay not in (None, ""):
            try:
                delay = min(
                    0.25,
                    max(0.0, float(configured_delay)),
                )
            except (TypeError, ValueError):
                delay = DMTAC_STARTUP_RETRY_DELAY_SEC
        if delay > 0.0:
            time.sleep(delay)

        try:
            recovered = bool(
                reconnect(reason="startup_readiness_retry")
            )
            if not recovered:
                report_fn = getattr(backend, "readiness_report", None)
                if callable(report_fn):
                    try:
                        report = report_fn()
                    except Exception as exc:  # noqa: BLE001 - preserve bounded diagnosis.
                        return False, (
                            "force reconnect returned false; readiness report failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    if isinstance(report, Mapping):
                        reason = str(report.get("reason") or "").strip()
                        if reason:
                            return False, reason
                return False, "force reconnect did not recover a ready backend"
            is_ready = getattr(backend, "is_ready", None)
            if callable(is_ready) and not bool(is_ready()):
                report_fn = getattr(backend, "readiness_report", None)
                reason = "backend remained unready after force reconnect"
                if callable(report_fn):
                    try:
                        report = report_fn()
                    except Exception as exc:  # noqa: BLE001
                        reason = (
                            f"{reason}; readiness report failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        if isinstance(report, Mapping):
                            reason = str(report.get("reason") or reason).strip()
                return False, reason
            return True, ""
        except Exception as exc:  # noqa: BLE001 - startup remains fail-closed.
            return False, f"{type(exc).__name__}: {exc}"

    def retry_rig_effector_once(
        self,
        key: str,
        backend: Any | None,
    ) -> tuple[bool, str]:
        """Recover one Rig-owned DMTAC effector after Rig startup.

        Session Rig is the sole owner of ``backend.start()`` for effectors
        registered in its command map.  The startup coordinator therefore
        cannot perform its normal pre-Rig start path for those backends.  If
        a vendor backend reports an unready state immediately after the Rig
        start (the common post-preflight serial settling race), this method
        provides the same single bounded reconnect used by the direct device
        admission path without starting the backend a second time.
        """

        if backend is None:
            return False, "backend is unavailable"
        return self._retry_dmtac_startup_once(str(key), backend)

    def start(self, key: str, backend: Any | None) -> bool:
        """Start one backend or record its capability-mode failure."""

        if backend is None:
            return False
        progress = self._startup_progress
        if progress is not None:
            progress.begin(
                key,
                label=self.label(key),
                message="正在调用设备 backend.start()",
                requirement=self.requirement(key),
            )
        try:
            backend.start()
            # ``start`` is allowed to be a capability-mode, non-throwing
            # boundary (DMTAC is one such backend), but a successful call must
            # never publish a false-ready arm/hand.  Check every backend that
            # exposes the standard readiness contract, not just DMTAC; this
            # also catches an RM65 whose startup returned without a trusted
            # state read.
            is_ready = getattr(backend, "is_ready", None)
            if callable(is_ready) and not bool(is_ready()):
                detail = ""
                readiness_report = getattr(backend, "readiness_report", None)
                if callable(readiness_report):
                    report = readiness_report()
                    if isinstance(report, Mapping):
                        detail = str(report.get("reason", "") or "").strip()
                        if not detail:
                            details = report.get("details")
                            if isinstance(details, Mapping):
                                detail = str(
                                    details.get("telemetry_error")
                                    or details.get("state_read_error")
                                    or details.get("hardware_result_text")
                                    or details.get("gripper_error_code")
                                    or ""
                                ).strip()
                if not detail and isinstance(backend, self._dmtac_gripper_backend_type):
                    try:
                        state = backend.get_state(include_tactile_arrays=False)
                    except TypeError:
                        state = backend.get_state()
                    if isinstance(state, Mapping):
                        detail = str(
                            state.get("telemetry_error")
                            or state.get("gripper_error_code")
                            or ""
                        ).strip()
                raise RuntimeError(detail or "backend is not ready")
            # A retry may call this boundary again after a previous capability
            # failure.  Once this exact backend is freshly ready, remove only
            # its stale blocker; unrelated device and Rig errors remain owned
            # by their respective lifecycle projections.
            self._startup_device_errors.pop(str(key), None)
            if progress is not None:
                progress.succeed(key)
            return True
        except Exception as exc:  # noqa: BLE001 - capability admission records failures.
            detail = f"{type(exc).__name__}: {exc}"
            identity = self._backend_transport_identity(backend)
            if identity and identity not in detail:
                detail = f"{detail} [{identity}]"
            # DMTAC's backend.start() owns its own cleanup and deliberately
            # returns in capability mode after a transport failure.  Before
            # turning that first failure into a persistent Hold, perform one
            # exact-object recovery so a transient post-preflight serial
            # settling error does not require a second GUI interaction.
            recovered, retry_detail = self._retry_dmtac_startup_once(key, backend)
            if recovered:
                self._startup_device_errors.pop(str(key), None)
                if progress is not None:
                    progress.succeed(key)
                recovery_identity = self._backend_transport_identity(backend)
                recovery_suffix = (
                    f" ({recovery_identity})" if recovery_identity else ""
                )
                self._log(
                    "Device",
                    f"{key} startup transient failure recovered by one bounded "
                    f"DMTAC reconnect{recovery_suffix}.",
                )
                if self._control_ui is not None:
                    self._control_ui.append_message(
                        "Device",
                        f"{key} 已在启动阶段自动恢复；请继续完成 runtime Home。",
                    )
                return True
            if retry_detail:
                detail = f"{detail}; startup retry: {retry_detail}"
            if progress is not None:
                progress.fail(key, detail)
            if not self._capability_admission:
                if progress is not None:
                    progress.close()
                raise
            self._startup_device_errors[str(key)] = detail
            self._log("Device", f"WARNING: {key} unavailable: {detail}")
            if self._control_ui is not None:
                self._control_ui.append_message(
                    "Device",
                    f"{key} 连接失败：{detail}",
                )
            return False


__all__ = [
    "DMTAC_STARTUP_RETRY_DELAY_SEC",
    "StartupDeviceRuntime",
    "ensure_session_rig_backend_binding",
    "inspect_session_rig_readiness",
    "startup_device_label",
    "startup_rig_effector_key",
    "startup_gripper_input_errors",
    "startup_device_requirement",
]
