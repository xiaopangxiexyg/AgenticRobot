"""Deployment adapters for optional impedance backend capabilities.

This module invokes backend-owned prepare/enable operations.  The application
and resource lifecycle remain responsible for deciding when those transitions
occur; canonical force safety, control loops, authority, and dispatch remain
with their existing lower owners.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def enable_impedance_force_control_backends(
    arm_backends: Mapping[str, Any],
    *,
    ready_timeout_sec: float = 1.0,
) -> None:
    for side, backend in arm_backends.items():
        enable = getattr(backend, "enable_force_control", None)
        if not callable(enable):
            continue
        enable()
        wait_ready = getattr(backend, "wait_until_force_control_ready", None)
        if callable(wait_ready):
            wait_ready(timeout_sec=float(ready_timeout_sec))
        print(f"[Deploy] {side} impedance force control enabled.", flush=True)


def warm_start_impedance_force_control_backends(
    arm_backends: Mapping[str, Any],
) -> tuple[str, ...]:
    """Prepare impedance controllers without starting periodic output loops.

    The pose-impedance backend performs force-sensor setup and captures its
    initial safe reference in ``warm_start_force_control``.  Starting the
    40 Hz loop is deliberately a separate, explicit action: otherwise a
    deployment which is still waiting for ``g`` would already issue hold
    commands and continuously run IK.
    """

    warmed_sides: list[str] = []
    for side, backend in arm_backends.items():
        warm_start = getattr(backend, "warm_start_force_control", None)
        if not callable(warm_start):
            continue
        warm_start()
        warmed_sides.append(str(side))
    return tuple(warmed_sides)


__all__ = [
    "enable_impedance_force_control_backends",
    "warm_start_impedance_force_control_backends",
]
