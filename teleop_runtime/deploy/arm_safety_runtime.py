"""Deployment-owned fanout for real-arm HOLD and emergency-stop requests.

Backend implementations remain the sole owners of hardware stop mechanisms,
latched state, and acknowledgements.  This module only applies deployment-mode
gates and fans one application request out to the configured arm backends.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from robot_motion.rm65 import (
    RM65_ARM_CONTROL_MODE_POSITION,
    normalize_rm65_arm_control_mode,
)

from ..interfaces import EffectorBackend
from .impedance_force_guard import bimanual_emergency_stop


def hold_deploy_arm_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> None:
    """Request all configured real arm backends to hold before reporting errors."""

    errors: list[str] = []
    for side, backend in arm_backends.items():
        try:
            backend.hold_motion(str(reason))
        except Exception as exc:  # noqa: BLE001 - attempt every active side.
            errors.append(f"{side}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("deploy arm hold failed: " + "; ".join(errors))


def request_policy_inference_soft_hold(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> dict[str, dict[str, object]]:
    """Freeze measured arm state while synchronous policy inference is pending.

    The hook is optional so position-only and non-RM65 backends retain their
    existing semantics.  An implemented backend must fail closed when it
    cannot produce a valid measured snapshot.
    """

    receipts: dict[str, dict[str, object]] = {}
    for side, backend in arm_backends.items():
        request = getattr(backend, "request_policy_inference_soft_hold", None)
        if not callable(request):
            receipts[str(side)] = {
                "active": False,
                "status": "unsupported",
                "backend_type": type(backend).__name__,
            }
            continue
        receipt = request(str(reason))
        receipts[str(side)] = dict(receipt or {})
    return receipts


def emergency_stop_deploy_arm_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> None:
    """Request a concurrent, latched emergency stop on every active arm."""

    errors = bimanual_emergency_stop(arm_backends, reason=str(reason))
    if errors:
        raise RuntimeError(
            "deploy bimanual emergency stop was not acknowledged by all active arms: "
            + "; ".join(f"{side}={detail}" for side, detail in sorted(errors.items()))
        )


def emergency_stop_position_deploy_on_keyboard_interrupt(
    args: Any,
    *,
    send_actions: bool,
    arm_backends: Mapping[str, EffectorBackend],
    emergency_stop: Callable[..., None] = emergency_stop_deploy_arm_backends,
) -> bool:
    """Request Arm_Stop for an active real-position deployment.

    Returns False for dry-run, arm-disabled, non-position, or backend-free
    processes so Ctrl+C retains its ordinary resource-cleanup behavior there.
    Any incomplete real-arm acknowledgement is intentionally propagated.
    """

    if not (
        bool(send_actions)
        and bool(getattr(args, "arms_enabled", False))
        and normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
        == RM65_ARM_CONTROL_MODE_POSITION
        and arm_backends
    ):
        return False
    emergency_stop(
        arm_backends,
        reason="keyboard_interrupt",
    )
    return True


__all__ = [
    "emergency_stop_deploy_arm_backends",
    "emergency_stop_position_deploy_on_keyboard_interrupt",
    "hold_deploy_arm_backends",
    "request_policy_inference_soft_hold",
]
