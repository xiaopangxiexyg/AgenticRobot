from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from robot_motion.action import split_effector_name
from robot_motion.runtime import CommandDispatchReceipt
from robot_motion.runtime import TeleopCommand

from ..interfaces import EffectorBackend


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
        requested_value = _copy_command_value(effector_command.value)
        try:
            result = backend.send_command(effector_command)
        except Exception as exc:
            if collect_backend_evidence:
                state, state_error = _safe_backend_state(backend)
                backend_ready: bool | None = _safe_backend_ready(backend)
            else:
                state, state_error = {}, None
                backend_ready = None
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
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            effectors[str(effector_name)] = receipt
            effector_command.metadata["deploy_backend_dispatch_receipt"] = receipt
            command.metadata["deploy_backend_dispatch_receipts"] = dict(effectors)
            raise
        final_value = _copy_command_value(effector_command.value)
        accepted = _dispatch_result_accepted(result)
        accepted_all = accepted_all and accepted
        if collect_backend_evidence:
            state, state_error = _safe_backend_state(backend)
            backend_ready: bool | None = _safe_backend_ready(backend)
        else:
            state, state_error = {}, None
            backend_ready = None
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
        }
        effectors[str(effector_name)] = receipt
        effector_command.metadata["deploy_backend_dispatch_receipt"] = receipt
        command.metadata["deploy_backend_dispatch_receipts"] = dict(effectors)
    batch = {
        "accepted": bool(accepted_all),
        "effectors": effectors,
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

    expected_sides = {str(side) for side in arm_backends}
    if required_sides is not None:
        expected_sides.update(str(side) for side in required_sides)
    sides: dict[str, dict[str, Any]] = {}
    all_ready = bool(expected_sides)
    for side in sorted(expected_sides):
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
                "background_streaming": {},
            }
            continue
        reasons: list[str] = []
        backend_ready = _safe_backend_ready(backend)
        if not backend_ready:
            reasons.append("backend_not_ready")
        state, state_error = _safe_backend_state(backend)
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
        if background_state.get("enabled"):
            if background_state.get("fault"):
                reasons.append("background_fault")
            if background_state.get("healthy") is False:
                reasons.append("background_unhealthy")
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
            "background_streaming": _compact_background_streaming_state(state),
        }
    return {
        "ready": bool(all_ready),
        "sides": sides,
        "require_background_target": bool(require_background_target),
        "required_sides": sorted(expected_sides),
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
                failures.append(f"{side}=" + ",".join(str(reason) for reason in reasons))
        if not failures:
            failures.append("no_arm_backends")
        raise RuntimeError(
            f"position deploy arm stream is not ready at {stage}: " + "; ".join(failures)
        )
    return health


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
        "healthy",
        "health_reasons",
        "thread_alive",
        "target_active",
        "target_sequence",
        "target_age_sec",
        "target_stale",
        "fault",
        "fault_count",
        "shutdown_requested",
        "shutdown_complete",
        "join_timed_out",
    )
    return {key: raw.get(key) for key in keys if key in raw}


__all__ = [
    "dispatch_teleop_command",
    "inspect_arm_backends_for_position_dispatch",
    "require_arm_backends_ready_for_position_dispatch",
]
