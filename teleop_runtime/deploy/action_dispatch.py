"""Deployment action splitting and discrete-effector dispatch adapters.

This module translates already-admitted deployment commands to backend calls
and records their receipts.  It does not own policy queues, position-stream
advancement, safety authority, or backend lifecycle.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from robot_motion.runtime import TeleopCommand

from ..backends.grippers import DMTacGripperBackend
from ..backends.hands import InspireTacHandBackend
from ..interfaces import EffectorBackend


def command_effector_kind_subset(
    command: TeleopCommand | None,
    *,
    kind: str,
) -> TeleopCommand | None:
    """Copy only one effector kind while preserving command metadata."""

    if command is None:
        return None
    suffix = f"_{str(kind).strip()}"
    subset = TeleopCommand(
        timestamp=command.timestamp,
        metadata=dict(command.metadata),
    )
    for effector_name, effector_command in command.commands.items():
        if str(effector_name).endswith(suffix):
            subset.add_command(effector_command.copy())
    return subset if subset.commands else None


def dispatch_position_discrete_command_checked(
    command: TeleopCommand,
    *,
    hand_backends: Mapping[str, EffectorBackend],
) -> Mapping[str, Any]:
    """Dispatch one reference-clock-aligned hand keyframe with receipts.

    Hand backends historically used both exceptions and return values to
    report I/O failures.  Position deploy needs an explicit result so the
    asynchronous executor can distinguish a safe no-op/rate skip from a real
    rejection and hold the arm stream on the latter.
    """

    receipts: dict[str, dict[str, Any]] = {}
    accepted_all = True
    skipped_any = False
    for effector_name, effector_command in command.commands.items():
        side = str(effector_command.side or "")
        backend = hand_backends.get(side)
        if backend is None:
            raise KeyError(
                f"Deploy backend missing for discrete effector {effector_name!r}."
            )
        started = time.monotonic()
        try:
            result = backend.send_command(effector_command)
            state = (
                backend.get_state(
                    include_tactile_arrays=False,
                    include_tactile_vector=False,
                )
                if isinstance(backend, DMTacGripperBackend)
                else backend.get_state()
            )
        except Exception as exc:
            receipt = {
                "accepted": False,
                "skipped": False,
                "fatal": True,
                "status": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_ms": (time.monotonic() - started) * 1000.0,
            }
            receipts[str(effector_name)] = receipt
            command.metadata["position_discrete_backend_receipts"] = dict(receipts)
            effector_command.metadata["position_discrete_backend_receipt"] = receipt
            raise RuntimeError(
                f"{effector_name} discrete dispatch failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        status = "accepted"
        accepted = result is not False
        skipped = False
        if isinstance(backend, DMTacGripperBackend):
            status = str(state.get("gripper_last_send_status") or "unknown")
            if status == "rate_limited":
                accepted = False
                skipped = True
            elif status == "intentionally_disconnected_noop":
                accepted = True
                skipped = True
            elif status == "skipped_threshold":
                # The requested value is already within the configured no-op
                # threshold; no additional physical packet is required.
                accepted = True
            elif status != "sent" and result is False:
                accepted = False
        elif isinstance(backend, InspireTacHandBackend):
            status = "sent_or_threshold_noop" if result is not False else "rejected"

        fatal = bool(not accepted and not skipped)
        receipt = {
            "accepted": bool(accepted),
            "skipped": bool(skipped),
            "fatal": bool(fatal),
            "status": status,
            "backend_type": type(backend).__name__,
            "backend_ready": bool(backend.is_ready()),
            "telemetry_error": state.get("telemetry_error"),
            "duration_ms": (time.monotonic() - started) * 1000.0,
        }
        receipts[str(effector_name)] = receipt
        effector_command.metadata["position_discrete_backend_receipt"] = receipt
        command.metadata["position_discrete_backend_receipts"] = dict(receipts)
        if fatal:
            raise RuntimeError(
                f"{effector_name} discrete dispatch rejected: "
                f"status={status} telemetry_error={state.get('telemetry_error')}"
            )
        accepted_all = accepted_all and bool(accepted)
        skipped_any = skipped_any or bool(skipped)
    return {
        "accepted": bool(accepted_all),
        "skipped": bool(skipped_any),
        "fatal": False,
        "effectors": receipts,
    }
