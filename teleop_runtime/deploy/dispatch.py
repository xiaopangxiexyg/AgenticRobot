from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from robot_motion.action import split_effector_name
from robot_motion.runtime import CommandDispatchReceipt
from robot_motion.runtime import TeleopCommand

from ..interfaces import EffectorBackend


POSITION_REFERENCE_DISPATCH_AUDIT_POST = "post"
POSITION_REFERENCE_DISPATCH_AUDIT_FULL = "full"
POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES = (
    POSITION_REFERENCE_DISPATCH_AUDIT_POST,
    POSITION_REFERENCE_DISPATCH_AUDIT_FULL,
)


class PositionDispatchNotReady(RuntimeError):
    """A rejected readiness cut, retained for lower-owned recovery decisions."""

    def __init__(self, message: str, *, health: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.health = health


# Real-launcher boundary:
# This file should NOT move to robot_motion. robot_motion should own common
# action/deploy preparation up to TeleopCommand. teleop_runtime owns dispatching
# TeleopCommand to real RM65/Inspire/DMTAC backends. A MuJoCo deploy launcher
# should implement an equivalent dispatcher in sim-specific code.
def dispatch_teleop_command(
    command: TeleopCommand,
    *,
    arm_backends: Mapping[str, EffectorBackend],
    hand_backends: Mapping[str, EffectorBackend],
    collect_backend_evidence: bool = False,
) -> dict[str, Any]:
    """Dispatch a command and return per-effector acceptance evidence.

    Existing callers may keep ignoring the return value.  Position deploy uses
    it to verify that a collision-planned ``joint_vector`` was accepted without
    being rewritten by an individual arm backend.  Backend state/ready polling
    is opt-in so legacy dispatch timing is unchanged.
    """

    batch_started = time.perf_counter()
    effectors: dict[str, dict[str, Any]] = {}
    accepted_all = True
    for effector_name, effector_command in command.commands.items():
        side, kind = split_effector_name(effector_name)
        if kind == "arm":
            backend = arm_backends.get(side)
        else:
            backend = hand_backends.get(side)
        if backend is None:
            raise KeyError(f"Deploy backend missing for effector {effector_name!r}.")
        effector_started = time.perf_counter()
        requested_value = _copy_command_value(effector_command.value)
        send_started = time.perf_counter()
        try:
            result = backend.send_command(effector_command)
        except Exception as exc:
            send_completed = time.perf_counter()
            if collect_backend_evidence:
                state_started = time.perf_counter()
                state, state_error = _safe_backend_state(backend)
                state_completed = time.perf_counter()
                ready_started = time.perf_counter()
                backend_ready: bool | None = _safe_backend_ready(backend)
                ready_completed = time.perf_counter()
            else:
                state, state_error = {}, None
                backend_ready = None
                state_started = state_completed = send_completed
                ready_started = ready_completed = send_completed
            timing = {
                "send_command_ms": (send_completed - send_started) * 1000.0,
                "evidence_get_state_ms": (state_completed - state_started) * 1000.0,
                "evidence_is_ready_ms": (ready_completed - ready_started) * 1000.0,
                "total_ms": (ready_completed - effector_started) * 1000.0,
            }
            receipt = {
                "accepted": False,
                "backend_type": type(backend).__name__,
                "backend_ready": backend_ready,
                "command_type": str(effector_command.command_type),
                "requested_value": requested_value,
                "final_value": _copy_command_value(effector_command.value),
                "backend_final_target_qpos": _copy_command_value(
                    effector_command.metadata.get("backend_final_target_qpos")
                ),
                "rewritten": _command_value_rewritten(
                    requested_value,
                    effector_command.value,
                ),
                "operation": None,
                "status": "exception",
                "result_code": None,
                "result_text": str(exc),
                "receipt_metadata": {},
                "state_error": state_error,
                "singularity_gate_status": _singularity_gate_status(
                    state,
                    effector_command.metadata,
                ),
                "background_streaming": _compact_background_streaming_state(state),
                "dispatch_timing": timing,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            effectors[str(effector_name)] = receipt
            effector_command.metadata["deploy_backend_dispatch_receipt"] = receipt
            command.metadata["deploy_backend_dispatch_receipts"] = dict(effectors)
            raise
        send_completed = time.perf_counter()
        final_value = _copy_command_value(effector_command.value)
        accepted = _dispatch_result_accepted(result)
        accepted_all = accepted_all and accepted
        if collect_backend_evidence:
            state_started = time.perf_counter()
            state, state_error = _safe_backend_state(backend)
            state_completed = time.perf_counter()
            ready_started = time.perf_counter()
            backend_ready: bool | None = _safe_backend_ready(backend)
            ready_completed = time.perf_counter()
        else:
            state, state_error = {}, None
            backend_ready = None
            state_started = state_completed = send_completed
            ready_started = ready_completed = send_completed
        timing = {
            "send_command_ms": (send_completed - send_started) * 1000.0,
            "evidence_get_state_ms": (state_completed - state_started) * 1000.0,
            "evidence_is_ready_ms": (ready_completed - ready_started) * 1000.0,
            "total_ms": (ready_completed - effector_started) * 1000.0,
        }
        receipt = {
            "accepted": bool(accepted),
            "backend_type": type(backend).__name__,
            "backend_ready": backend_ready,
            "command_type": str(effector_command.command_type),
            "requested_value": requested_value,
            "final_value": final_value,
            "backend_final_target_qpos": _copy_command_value(
                effector_command.metadata.get("backend_final_target_qpos")
            ),
            "rewritten": _command_value_rewritten(requested_value, final_value),
            "operation": getattr(result, "operation", None),
            "status": getattr(result, "status", None),
            "result_code": getattr(result, "result_code", None),
            "result_text": getattr(result, "result_text", ""),
            "receipt_metadata": dict(getattr(result, "metadata", {}) or {}),
            "state_error": state_error,
            "singularity_gate_status": _singularity_gate_status(
                state,
                effector_command.metadata,
            ),
            "background_streaming": _compact_background_streaming_state(state),
            "dispatch_timing": timing,
        }
        effectors[str(effector_name)] = receipt
        effector_command.metadata["deploy_backend_dispatch_receipt"] = receipt
        command.metadata["deploy_backend_dispatch_receipts"] = dict(effectors)
    batch = {
        "accepted": bool(accepted_all),
        "effectors": effectors,
        "dispatch_timing": {
            "total_ms": (time.perf_counter() - batch_started) * 1000.0,
            "effectors": {
                name: dict(receipt.get("dispatch_timing") or {})
                for name, receipt in effectors.items()
            },
        },
    }
    command.metadata["deploy_backend_dispatch_receipts"] = effectors
    return batch


def inspect_arm_backends_for_position_dispatch(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    require_background_target: bool,
    required_sides: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed readiness snapshot for position arm streams."""

    inspection_started = time.perf_counter()
    expected_sides = {str(side) for side in arm_backends}
    if required_sides is not None:
        expected_sides.update(str(side) for side in required_sides)
    sides: dict[str, dict[str, Any]] = {}
    all_ready = bool(expected_sides)
    for side in sorted(expected_sides):
        side_started = time.perf_counter()
        backend = arm_backends.get(side)
        if backend is None:
            all_ready = False
            sides[side] = {
                "ready": False,
                "reasons": ["backend_missing"],
                "backend_ready": False,
                "state_error": None,
                "hardware_ready": None,
                "emergency_stopped": False,
                "send_result_code": None,
                "singularity_metadata": {},
                "background_streaming": {},
                "timing": {
                    "is_ready_ms": 0.0,
                    "get_state_ms": 0.0,
                    "total_ms": (time.perf_counter() - side_started) * 1000.0,
                },
            }
            continue
        reasons: list[str] = []
        ready_started = time.perf_counter()
        backend_ready = _safe_backend_ready(backend)
        ready_completed = time.perf_counter()
        if not backend_ready:
            reasons.append("backend_not_ready")
        state_started = time.perf_counter()
        state, state_error = _safe_backend_state(backend)
        state_completed = time.perf_counter()
        if state_error:
            reasons.append("state_error")
        if state.get("state_error"):
            reasons.append("state_error_reported")
        if state.get("hardware_ready") is False:
            reasons.append("hardware_not_ready")
        if bool(state.get("emergency_stopped", False)):
            reasons.append("emergency_stopped")
        send_result_code = state.get("send_result_code")
        if send_result_code not in (None, 0):
            reasons.append(f"send_result_code:{send_result_code}")

        background = state.get("background_streaming")
        background_state = dict(background) if isinstance(background, Mapping) else {}
        background_warnings: list[str] = []
        if background_state.get("enabled"):
            if background_state.get("fault"):
                reasons.append("background_fault")
            health_reasons = tuple(
                str(reason)
                for reason in (background_state.get("health_reasons") or ())
            )
            if "low_send_rate" in health_reasons:
                # Keep the warning in the receipt for diagnosis, but do not
                # downgrade it: a measured stream below its contract rate is
                # unsafe for position dispatch and must stop the lane.  The
                # next command must never be used to "catch up" a late slot.
                background_warnings.append("low_send_rate")
            blocking_health_reasons = health_reasons
            if background_state.get("healthy") is False:
                if blocking_health_reasons:
                    reasons.append("background_unhealthy")
                elif not health_reasons:
                    # Unknown unhealthy states stay fail-closed.
                    reasons.append("background_unhealthy_unknown")
            if background_state.get("thread_alive") is not True:
                reasons.append("background_thread_not_alive")
            if bool(background_state.get("shutdown_requested", False)):
                reasons.append("background_shutdown_requested")
            if bool(background_state.get("shutdown_complete", False)):
                reasons.append("background_shutdown_complete")
            if bool(background_state.get("join_timed_out", False)):
                reasons.append("background_join_timed_out")
            if bool(background_state.get("target_stale", False)):
                reasons.append("background_target_stale")
            if require_background_target and background_state.get("target_active") is not True:
                reasons.append("background_target_not_active")

        side_ready = not reasons
        all_ready = all_ready and side_ready
        sides[str(side)] = {
            "ready": bool(side_ready),
            "reasons": reasons,
            "backend_ready": bool(backend_ready),
            "state_error": state_error or state.get("state_error"),
            "hardware_ready": state.get("hardware_ready"),
            "emergency_stopped": bool(state.get("emergency_stopped", False)),
            "send_result_code": send_result_code,
            "singularity_metadata": dict(state.get("singularity_metadata") or {}),
            "background_streaming": _compact_background_streaming_state(state),
            "warnings": background_warnings,
            "timing": {
                "is_ready_ms": (ready_completed - ready_started) * 1000.0,
                "get_state_ms": (state_completed - state_started) * 1000.0,
                "total_ms": (state_completed - side_started) * 1000.0,
            },
        }
    return {
        "ready": bool(all_ready),
        "sides": sides,
        "require_background_target": bool(require_background_target),
        "required_sides": sorted(expected_sides),
        "timing": {
            "total_ms": (time.perf_counter() - inspection_started) * 1000.0,
            "sides": {
                side: dict(item.get("timing") or {})
                for side, item in sides.items()
            },
        },
    }


def require_arm_backends_ready_for_position_dispatch(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    require_background_target: bool,
    stage: str,
    required_sides: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    health = inspect_arm_backends_for_position_dispatch(
        arm_backends,
        require_background_target=require_background_target,
        required_sides=required_sides,
    )
    if not bool(health["ready"]):
        failures = []
        for side, item in health["sides"].items():
            reasons = item.get("reasons") or []
            if reasons:
                failure = f"{side}=" + ",".join(
                    str(reason) for reason in reasons
                )
                background = item.get("background_streaming")
                if isinstance(background, Mapping) and background.get("fault"):
                    fault = " ".join(str(background["fault"]).split())[:512]
                    failure += f" [background_fault={fault}]"
                failures.append(failure)
        if not failures:
            failures.append("no_arm_backends")
        raise PositionDispatchNotReady(
            f"position deploy arm stream is not ready at {stage}: " + "; ".join(failures),
            health=health,
        )
    return health


def dispatch_position_reference_command(
    command: TeleopCommand,
    *,
    arm_backends: Mapping[str, EffectorBackend],
    required_sides: tuple[str, ...],
    audit_mode: str = POSITION_REFERENCE_DISPATCH_AUDIT_POST,
) -> dict[str, Any]:
    """Queue one authoritative qref and return fail-closed backend evidence.

    ``post`` is the live-control default: ``send_command`` already rejects a
    closed, emergency-stopped, untrusted, or fault-latched RM65 backend, then a
    single dual-arm post-dispatch snapshot verifies that both background
    streams accepted an active target and remain healthy. ``full`` preserves
    the former three-snapshot audit (pre, per-effector, post) as an explicit
    diagnostic rollback, at the cost of extra synchronous SDK/state-lock work
    on the 60 Hz reference thread.
    """

    mode = str(audit_mode).strip().lower()
    if mode not in POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES:
        raise ValueError(
            "position reference dispatch audit_mode must be one of "
            f"{POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES}, got {audit_mode!r}."
        )

    dispatch_started = time.perf_counter()
    pre_health: dict[str, Any] | None = None
    if mode == POSITION_REFERENCE_DISPATCH_AUDIT_FULL:
        pre_health = require_arm_backends_ready_for_position_dispatch(
            arm_backends,
            require_background_target=False,
            stage="pre_arm_reference_queue",
            required_sides=required_sides,
        )
    pre_health_completed = time.perf_counter()
    dispatch_result = dispatch_teleop_command(
        command,
        arm_backends=arm_backends,
        hand_backends={},
        collect_backend_evidence=(
            mode == POSITION_REFERENCE_DISPATCH_AUDIT_FULL
        ),
    )
    backend_dispatch_completed = time.perf_counter()
    post_health = require_arm_backends_ready_for_position_dispatch(
        arm_backends,
        require_background_target=True,
        stage="post_arm_reference_queue",
        required_sides=required_sides,
    )
    post_health_completed = time.perf_counter()
    dispatch_result["arm_backend_health"] = post_health
    dispatch_result["position_dispatch_audit_mode"] = mode
    dispatch_result["position_dispatch_timing"] = {
        "audit_mode": mode,
        "pre_health_ms": (pre_health_completed - dispatch_started) * 1000.0,
        "backend_dispatch_ms": (
            backend_dispatch_completed - pre_health_completed
        )
        * 1000.0,
        "post_health_ms": (
            post_health_completed - backend_dispatch_completed
        )
        * 1000.0,
        "total_ms": (post_health_completed - dispatch_started) * 1000.0,
        "pre_health": (
            {} if pre_health is None else dict(pre_health.get("timing") or {})
        ),
        "backend_dispatch": dict(dispatch_result.get("dispatch_timing") or {}),
        "post_health": dict(post_health.get("timing") or {}),
    }
    return dispatch_result


def _copy_command_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        array = np.asarray(value)
    except Exception:
        return value
    if array.ndim > 0:
        return array.copy()
    return value


def _command_value_rewritten(requested: object, final: object) -> bool:
    try:
        requested_array = np.asarray(requested, dtype=float)
        final_array = np.asarray(final, dtype=float)
    except Exception:
        return requested != final
    if requested_array.shape != final_array.shape:
        return True
    return not bool(
        np.allclose(
            requested_array,
            final_array,
            rtol=0.0,
            atol=1e-7,
            equal_nan=True,
        )
    )


def _dispatch_result_accepted(result: object) -> bool:
    if isinstance(result, CommandDispatchReceipt):
        return bool(result.accepted)
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return result is not False


def _safe_backend_ready(backend: EffectorBackend) -> bool:
    try:
        return bool(backend.is_ready())
    except Exception:
        return False


def _safe_backend_state(
    backend: EffectorBackend,
) -> tuple[dict[str, Any], str | None]:
    try:
        return dict(backend.get_state()), None
    except Exception as exc:  # noqa: BLE001 - health checks must report, not hide.
        return {}, f"{type(exc).__name__}: {exc}"


def _singularity_gate_status(
    state: Mapping[str, Any],
    command_metadata: Mapping[str, Any],
) -> str | None:
    for raw in (
        command_metadata.get("singularity_metadata"),
        state.get("singularity_metadata"),
        state.get("ik_metadata"),
    ):
        if isinstance(raw, Mapping) and raw.get("singularity_gate_status") is not None:
            return str(raw.get("singularity_gate_status"))
    return None


def _compact_background_streaming_state(state: Mapping[str, Any]) -> dict[str, Any]:
    raw = state.get("background_streaming")
    if not isinstance(raw, Mapping):
        return {}
    keys = (
        "enabled",
        "no_catch_up_after_send",
        "healthy",
        "health_reasons",
        "thread_alive",
        "target_active",
        "target_sequence",
        "target_age_sec",
        "target_stale",
        "fault",
        "fault_count",
        "fault_timestamp",
        "fault_stop_attempted",
        "fault_stop_success",
        "fault_stop_attempts",
        "fault_stop_timestamp",
        "shutdown_requested",
        "shutdown_complete",
        "join_timed_out",
        "actual_hz",
        "last_call_ms",
        "last_deadline_lag_ms",
        "last_send_dt_sec",
        "max_inter_send_gap_sec",
        "deadline_miss_count",
        "skipped_tick_count",
        "low_rate_count",
        "target_admission_mode",
        "last_target_admission_timing",
        "inflight_target_preemption_count",
        "last_inflight_target_preemption",
    )
    return {key: raw.get(key) for key in keys if key in raw}


__all__ = [
    "POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES",
    "POSITION_REFERENCE_DISPATCH_AUDIT_FULL",
    "POSITION_REFERENCE_DISPATCH_AUDIT_POST",
    "dispatch_teleop_command",
    "dispatch_position_reference_command",
    "inspect_arm_backends_for_position_dispatch",
    "require_arm_backends_ready_for_position_dispatch",
]
