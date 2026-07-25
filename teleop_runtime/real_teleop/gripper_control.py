"""DMTAC gripper warmup probing and Q6 command dispatch."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from robot_motion.runtime import TeleopCommand, merge_teleop_commands

from ..backends.grippers import DMTacGripperBackend
from ..gui.status_payloads import (
    _json_safe_value,
    _safe_float_value,
    _safe_int_value,
)
from ..log_utils import log_event as _log
from .components import RealTeleopComponents
from .hand_runtime import (
    _gripper_backend_for_side,
    _gripper_retargeter_for_side,
    _read_gripper_backend_scalar_now,
)


def _probe_gripper_backend_motion(
    backend: object,
    *,
    targets: Sequence[float] = (0.0, 0.5, 0.0),
    tolerance: float = 0.18,
    step_timeout_s: float = 2.0,
    settle_s: float = 0.2,
    poll_s: float = 0.05,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    send = getattr(backend, "send_fixed_joint_pos", None)
    if not callable(send):
        return {"ok": False, "error": "backend_missing_send_fixed_joint_pos", "samples": []}
    samples: list[dict[str, object]] = []
    for index, raw_target in enumerate(targets):
        if cancelled is not None and cancelled():
            return {"ok": False, "error": "cancelled", "step": int(index), "samples": samples}
        target = float(np.clip(float(raw_target), 0.0, 1.0))
        try:
            sent = bool(send(target))
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": f"send_failed:{type(exc).__name__}:{exc}",
                "target": target,
                "step": int(index),
                "samples": samples,
            }
        if not sent:
            return {
                "ok": False,
                "error": "send_returned_false",
                "target": target,
                "step": int(index),
                "samples": samples,
            }
        deadline = time.time() + max(0.0, float(step_timeout_s))
        if float(settle_s) > 0.0:
            time.sleep(float(settle_s))
        last_measured: float | None = None
        while True:
            if cancelled is not None and cancelled():
                return {"ok": False, "error": "cancelled", "step": int(index), "samples": samples}
            try:
                measured = _read_gripper_backend_scalar_now(backend)
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "error": f"read_failed:{type(exc).__name__}:{exc}",
                    "target": target,
                    "step": int(index),
                    "samples": samples,
                }
            last_measured = measured
            sample = {
                "step": int(index),
                "target": target,
                "measured": measured,
                "error_abs": None if measured is None else abs(float(measured) - target),
            }
            samples.append(sample)
            if measured is not None and abs(float(measured) - target) <= float(tolerance):
                break
            if time.time() >= deadline:
                return {
                    "ok": False,
                    "error": "target_not_reached",
                    "target": target,
                    "measured": last_measured,
                    "step": int(index),
                    "tolerance": float(tolerance),
                    "samples": samples,
                }
            time.sleep(max(0.0, float(poll_s)))
    return {"ok": True, "error": None, "samples": samples}


def _run_gripper_warmup_probe_for_side(
    side: str,
    backend: object,
    *,
    targets: Sequence[float],
    tolerance: float,
    step_timeout_s: float,
    settle_s: float,
    poll_s: float,
    reconnect_on_failure: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    first = _probe_gripper_backend_motion(
        backend,
        targets=targets,
        tolerance=tolerance,
        step_timeout_s=step_timeout_s,
        settle_s=settle_s,
        poll_s=poll_s,
        cancelled=cancelled,
    )
    attempts = [dict(first)]
    if bool(first.get("ok")):
        return {"side": str(side), "ok": True, "attempts": attempts, "reconnected": False}
    reconnected = False
    reconnect_error = None
    if reconnect_on_failure:
        reconnect = getattr(backend, "force_reconnect_gripper", None)
        if callable(reconnect):
            try:
                reconnected = bool(reconnect(f"warmup_probe_failed:{first.get('error')}"))
            except Exception as exc:  # noqa: BLE001
                reconnect_error = f"{type(exc).__name__}: {exc}"
    if reconnected:
        if cancelled is not None and cancelled():
            return {
                "side": str(side),
                "ok": False,
                "attempts": attempts,
                "reconnected": reconnected,
                "reconnect_error": reconnect_error,
                "error": "cancelled",
            }
        second = _probe_gripper_backend_motion(
            backend,
            targets=targets,
            tolerance=tolerance,
            step_timeout_s=step_timeout_s,
            settle_s=settle_s,
            poll_s=poll_s,
            cancelled=cancelled,
        )
        attempts.append(dict(second))
        if bool(second.get("ok")):
            return {"side": str(side), "ok": True, "attempts": attempts, "reconnected": True}
    return {
        "side": str(side),
        "ok": False,
        "attempts": attempts,
        "reconnected": reconnected,
        "reconnect_error": reconnect_error,
    }

def _dispatch_q6_gripper_control(
    components: RealTeleopComponents,
    observation,
    *,
    now: float,
    status_out: dict[str, dict[str, object]] | None = None,
) -> TeleopCommand | None:
    if observation is None:
        if status_out is not None:
            for side in components.active_sides:
                status_out[side] = {
                    "status": "no_observation",
                    "sent": False,
                    "timestamp": float(now),
                }
        return None
    commands: list[TeleopCommand] = []

    def backend_send_fields(backend: DMTacGripperBackend) -> dict[str, object]:
        elapsed_s = _safe_float_value(getattr(backend, "_last_send_elapsed_s", None))
        lock_wait_s = _safe_float_value(getattr(backend, "_last_send_lock_wait_s", None))
        return {
            "backend_status": str(getattr(backend, "_last_send_status", "") or ""),
            "skip_reason": str(getattr(backend, "_last_send_skip_reason", "") or ""),
            "send_elapsed_s": elapsed_s,
            "send_elapsed_ms": None if elapsed_s is None else float(elapsed_s) * 1000.0,
            "send_lock_wait_s": lock_wait_s,
            "send_lock_wait_ms": None if lock_wait_s is None else float(lock_wait_s) * 1000.0,
            "send_target": _safe_float_value(getattr(backend, "_last_send_target_scalar", None)),
        }

    for side in components.active_sides:
        side_status: dict[str, object] = {
            "status": "pending",
            "sent": False,
            "timestamp": float(now),
        }

        def finish(status: str, **fields: object) -> None:
            side_status["status"] = status
            side_status.update(fields)
            if status_out is not None:
                status_out[side] = dict(side_status)

        backend = _gripper_backend_for_side(components, side)
        retargeter = _gripper_retargeter_for_side(components, side)
        if backend is None:
            finish("no_backend")
            continue
        if retargeter is None:
            finish("no_retargeter")
            continue
        fix_arm_controller = getattr(components, "fix_arm_controller", None)
        fixed_side_active = bool(
            fix_arm_controller is not None
            and fix_arm_controller.fixed
            and fix_arm_controller.fixed_side == side
        )
        if fixed_side_active:
            command = TeleopCommand(timestamp=float(now))
            command = fix_arm_controller.filter_command(
                command,
                observation,
                now,
                include_missing_fixed_effectors=True,
                allowed_effectors=(f"{side}_hand",),
            )
        else:
            try:
                command = retargeter.retarget(observation, components.session.calibration)
            except Exception as exc:  # noqa: BLE001
                finish("retarget_exception", error=f"{type(exc).__name__}: {exc}")
                _log("Gripper", f"{side} Q6 retarget failed: {exc}", now=now)
                continue
            if fix_arm_controller is not None:
                command = fix_arm_controller.filter_command(command, observation, now)
        if not command.commands:
            finish(
                "no_retarget_command",
                skipped=_json_safe_value(command.metadata.get("skipped")),
            )
            continue
        effector_command = command.commands.get(f"{side}_hand")
        if effector_command is None:
            finish(
                "missing_effector_command",
                effector_names=tuple(str(name) for name in command.commands),
            )
            continue
        metadata = effector_command.metadata or {}
        if fixed_side_active:
            finish(
                "fixed_command",
                target=_safe_float_value(metadata.get("gripper_joint_target")),
            )
        else:
            finish(
                "retargeted",
                source_dim=_safe_int_value(metadata.get("source_dim")),
                output_dim=_safe_int_value(metadata.get("output_dim")),
                source_value=_safe_float_value(metadata.get("source_value")),
                target=_safe_float_value(metadata.get("gripper_joint_target")),
            )
        try:
            sent_ok = bool(backend.send_command(effector_command))
        except Exception as exc:  # noqa: BLE001
            finish("send_exception", error=f"{type(exc).__name__}: {exc}")
            _log("Gripper", f"{side} Q6 command failed: {exc}", now=now)
            continue
        commands.append(command)
        if not sent_ok:
            fields = backend_send_fields(backend)
            backend_status = str(fields.get("backend_status") or "send_returned_false")
            finish(backend_status, **fields)
            continue
        finish("sent", sent=True, **backend_send_fields(backend))
    if not commands:
        return None
    return merge_teleop_commands(commands)

__all__ = [
    "_dispatch_q6_gripper_control",
    "_probe_gripper_backend_motion",
    "_run_gripper_warmup_probe_for_side",
]

