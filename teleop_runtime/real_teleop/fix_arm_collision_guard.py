"""Final coupled-collision guard for commands rewritten by fix-arm.

The normal dual-arm retargeter checks its own candidate pair.  Fix-arm is a
later command filter, so it must re-check the concrete pair it will dispatch.
This module is intentionally read-only: the session owns the fail-closed hold
and FAULT transition when this guard returns a fault.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..diagnostics import FaultEvent, FaultSeverity
from ..schemas import EffectorCommand, TeleopCommand


def _qpos_from_command(command: EffectorCommand | None) -> np.ndarray | None:
    if command is None:
        return None
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    for key in ("backend_final_target_qpos", "solver_output_qpos"):
        candidate = metadata.get(key)
        if candidate is None:
            continue
        try:
            qpos = np.asarray(candidate, dtype=float).reshape(6)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(qpos)):
            return qpos.copy()
    if str(command.command_type) == "joint_vector":
        try:
            qpos = np.asarray(command.value, dtype=float).reshape(6)
        except (TypeError, ValueError):
            return None
        if np.all(np.isfinite(qpos)):
            return qpos.copy()
    return None


def fixed_arm_final_collision_fault(
    command: TeleopCommand,
    *,
    fix_arm_controller: Any,
    dual_arm_retargeter: Any,
    now: float,
) -> FaultEvent | None:
    """Return a fatal fault when a fixed final pair cannot pass collision checks."""

    if not bool(getattr(fix_arm_controller, "fixed", False)):
        return None
    fixed_side = str(getattr(fix_arm_controller, "fixed_side", "") or "")
    if fixed_side not in {"left", "right"}:
        return FaultEvent(
            code="fix_arm_final_collision_check_failed",
            message="fix-arm is active without a valid fixed side",
            severity=FaultSeverity.ERROR,
            timestamp=float(now),
            component="fix_arm_final_collision_guard",
        )

    commands = command.commands if isinstance(command.commands, Mapping) else {}
    left_qpos = _qpos_from_command(commands.get("left_arm"))
    right_qpos = _qpos_from_command(commands.get("right_arm"))
    if left_qpos is None or right_qpos is None:
        missing_sides = [
            side
            for side, qpos in (("left", left_qpos), ("right", right_qpos))
            if qpos is None
        ]
        return FaultEvent(
            code="fix_arm_final_collision_check_failed",
            message=(
                "fix-arm final coupled collision check cannot resolve concrete qpos for "
                + ", ".join(missing_sides)
            ),
            severity=FaultSeverity.ERROR,
            timestamp=float(now),
            component="fix_arm_final_collision_guard",
            metadata={
                "fixed_side": fixed_side,
                "missing_qpos_sides": missing_sides,
            },
        )

    checker = getattr(dual_arm_retargeter, "check_qpos_pair_collision", None)
    if not callable(checker):
        return FaultEvent(
            code="fix_arm_final_collision_check_failed",
            message="fix-arm final coupled collision check is unavailable",
            severity=FaultSeverity.ERROR,
            timestamp=float(now),
            component="fix_arm_final_collision_guard",
            metadata={"fixed_side": fixed_side},
        )
    try:
        check = checker(left_qpos=left_qpos, right_qpos=right_qpos)
    except Exception as exc:  # noqa: BLE001 - this is a fail-closed pre-dispatch guard.
        return FaultEvent(
            code="fix_arm_final_collision_check_failed",
            message=(
                "fix-arm final coupled collision check raised "
                f"{type(exc).__name__}: {exc}"
            ),
            severity=FaultSeverity.ERROR,
            timestamp=float(now),
            component="fix_arm_final_collision_guard",
            metadata={"fixed_side": fixed_side},
        )
    if bool(getattr(check, "safe", False)):
        return None

    min_distance_m = getattr(check, "min_distance_m", None)
    pair_count = getattr(check, "collision_pair_count", None)
    reason = str(getattr(check, "reason", "collision_check_failed") or "collision_check_failed")
    required_min_distance_m = getattr(check, "required_min_distance_m", None)
    return FaultEvent(
        code="fix_arm_final_collision_check_failed",
        message=(
            "fix-arm final coupled collision check rejected the dispatch pair: "
            f"reason={reason} distance_m={min_distance_m} pairs={pair_count}"
        ),
        severity=FaultSeverity.ERROR,
        timestamp=float(now),
        component="fix_arm_final_collision_guard",
        metadata={
            "fixed_side": fixed_side,
            "reason": reason,
            "collision_distance_m": min_distance_m,
            "collision_pair_count": pair_count,
            "collision_min_distance_m": required_min_distance_m,
            "left_qpos": left_qpos.copy(),
            "right_qpos": right_qpos.copy(),
        },
    )


__all__ = ["fixed_arm_final_collision_fault"]
