"""Arm runtime state projection and coupled-IK synchronization helpers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import numpy as np

from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
)

from ..backends.arms import RM65ArmBackend, RM65ImpedanceArmBackend
from .arm_tool_config import _force_tare_sec_arg
from .components import RealTeleopComponents
from .control_profile import arm_control_mode as _arm_control_mode_arg


def _sync_dual_arm_retargeter_from_preview(
    components: RealTeleopComponents,
    *,
    reset_tracking: bool = True,
) -> None:
    if components.dual_arm_retargeter is None:
        return
    left_qpos = components.left_preview_backend.get_state()["current_qpos"]
    right_qpos = components.right_preview_backend.get_state()["current_qpos"]
    if left_qpos is None or right_qpos is None:
        return
    try:
        components.dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=left_qpos,
            right_qpos=right_qpos,
            reset_tracking=reset_tracking,
        )
    except TypeError as exc:
        if "reset_tracking" not in str(exc):
            raise
        components.dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=left_qpos,
            right_qpos=right_qpos,
        )


def _apply_active_arm_command_to_preview_real(
    components: RealTeleopComponents,
    active_command,
) -> None:
    if active_command is None:
        return
    left_command = active_command.commands.get("left_arm")
    if left_command is not None:
        components.left_preview_backend.send_command(left_command.copy())
    right_command = active_command.commands.get("right_arm")
    if right_command is not None:
        components.right_preview_backend.send_command(right_command.copy())


def _acceleration_limited_position_execution_active(
    components: RealTeleopComponents,
) -> bool:
    active_backends = [
        _command_backend_for_side(components, side)
        for side in components.active_sides
    ]
    return bool(active_backends) and all(
        isinstance(backend, RM65ArmBackend)
        and str(getattr(backend.config, "position_command_planner", ""))
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        for backend in active_backends
    )


def _sync_dual_arm_retargeter_for_real_control(
    components: RealTeleopComponents,
    *,
    reset_tracking: bool = True,
) -> None:
    """Seed coupled IK from the state the position transport actually accepted.

    The legacy linear path reaches each outer qref within one reference period,
    so its historical preview-state synchronization is retained. The optional
    acceleration-limited path can intentionally lag qref (especially at the
    10% degraded speed); seeding IK from preview would then certify collision
    and joint steps around a configuration the real transport has not reached.
    """

    if components.dual_arm_retargeter is None:
        return
    if not _acceleration_limited_position_execution_active(components):
        _sync_dual_arm_retargeter_from_preview(
            components,
            reset_tracking=reset_tracking,
        )
        return

    qpos_by_side: dict[str, np.ndarray] = {}
    accepted_monotonic_by_side: dict[str, float] = {}
    source_by_side: dict[str, str] = {}
    for side in ("left", "right"):
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, RM65ArmBackend):
            raise RuntimeError(
                f"Acceleration-limited coupled IK requires an RM65 command backend for {side}."
            )
        # This runs twice per outer-loop frame (before and after preview
        # application). Use the real-time snapshot path: the complete
        # get_state() default rebuilds five exact percentile windows on a
        # one-second cache cadence and creates a visible periodic control
        # stall once those windows grow.
        state = _runtime_arm_backend_state(
            backend,
            include_derived_pose=False,
        )
        raw_qpos = state.get("last_accepted_qpos")
        source = "last_accepted_qpos"
        if raw_qpos is None:
            raw_qpos = state.get("current_qpos")
            source = "current_qpos_fallback"
        try:
            qpos = np.asarray(raw_qpos, dtype=float).reshape(6)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{side}_arm has no valid accepted qcmd for coupled IK synchronization."
            ) from exc
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError(
                f"{side}_arm accepted qcmd is non-finite; refusing coupled IK synchronization."
            )
        qpos_by_side[side] = qpos.copy()
        source_by_side[side] = source
        accepted_monotonic = state.get("last_accepted_timestamp_monotonic")
        if accepted_monotonic is not None:
            try:
                normalized_time = float(accepted_monotonic)
            except (TypeError, ValueError):
                normalized_time = float("nan")
            if np.isfinite(normalized_time):
                accepted_monotonic_by_side[side] = normalized_time

    components.dual_arm_retargeter.set_current_qpos_pair(
        left_qpos=qpos_by_side["left"],
        right_qpos=qpos_by_side["right"],
        reset_tracking=reset_tracking,
    )
    timestamps = tuple(accepted_monotonic_by_side.values())
    skew_sec = None if len(timestamps) < 2 else max(timestamps) - min(timestamps)
    components.arm_loop_step_dt_tracker["retargeter_execution_state_sync"] = {
        "source_by_side": source_by_side,
        "accepted_monotonic_by_side": dict(accepted_monotonic_by_side),
        "accepted_timestamp_skew_sec": skew_sec,
        "left_qpos": qpos_by_side["left"].tolist(),
        "right_qpos": qpos_by_side["right"].tolist(),
    }


def _command_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> RM65ArmBackend | RM65ImpedanceArmBackend | None:
    return components.left_command_backend if side == "left" else components.right_command_backend


def _active_command_backend(components: RealTeleopComponents) -> RM65ArmBackend | RM65ImpedanceArmBackend | None:
    for side in tuple(getattr(components, "active_sides", ("left", "right"))):
        backend = _command_backend_for_side(components, side)
        if backend is not None:
            return backend
    return None


def _runtime_arm_backend_state(
    backend: object | None,
    *,
    include_derived_pose: bool = False,
) -> dict[str, object]:
    """Read only the position-backend fields needed by the high-rate loop.

    ``RM65ArmBackend.get_state()`` keeps its complete diagnostic default for
    tools and compatibility callers.  The live 60 Hz loop must not repeatedly
    sort timing windows or run diagnostic FK merely to publish health state.
    Force/impedance backends retain their established snapshot path.
    """

    if backend is None:
        return {}
    if isinstance(backend, RM65ArmBackend):
        return backend.get_state(
            include_derived_pose=bool(include_derived_pose),
            include_timing_percentiles=False,
        )
    get_state = getattr(backend, "get_state", None)
    if not callable(get_state):
        return {}
    state = get_state()
    return dict(state) if isinstance(state, Mapping) else {}


def _active_impedance_force_backends(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    impedance_backend_type: type[object] = RM65ImpedanceArmBackend,
) -> list[tuple[str, RM65ImpedanceArmBackend]]:
    if _arm_control_mode_arg(args) != "impedance":
        return []
    backends: list[tuple[str, RM65ImpedanceArmBackend]] = []
    for side in tuple(getattr(components, "active_sides", ("left", "right"))):
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, impedance_backend_type):
            raise RuntimeError(f"{side}_arm command backend is not RM65ImpedanceArmBackend.")
        backends.append((side, backend))
    return backends


def _impedance_force_control_warmup_backends(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> list[tuple[str, RM65ImpedanceArmBackend]]:
    backends = []
    for side, backend in _active_impedance_force_backends(args, components):
        if not bool(getattr(backend.config, "connect_hardware", True)):
            continue
        if not backend.force_control_warm_started():
            backends.append((side, backend))
    return backends


def _impedance_force_control_not_ready_sides(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    impedance_backend_type: type[object] = RM65ImpedanceArmBackend,
) -> list[str]:
    not_ready: list[str] = []
    for side, backend in _active_impedance_force_backends(
        args,
        components,
        impedance_backend_type=impedance_backend_type,
    ):
        get_state = getattr(backend, "get_state", None)
        if not callable(get_state):
            # Compatibility/test doubles without a health surface retain the
            # established begin_exact_qpos_transit admission contract.
            continue
        state = get_state()
        if not bool(state.get("impedance_enabled")):
            not_ready.append(side)
            continue
        if bool(getattr(backend.config, "connect_hardware", True)) and (
            state.get("impedance_loop_state") != "alive"
            or not isinstance(state.get("last_impedance_record"), Mapping)
        ):
            not_ready.append(side)
    return not_ready


def _estimate_impedance_force_control_warmup_sec(
    args: argparse.Namespace,
    backends: Sequence[tuple[str, RM65ImpedanceArmBackend]],
) -> float:
    if not backends:
        return 0.0
    tare_sec = max(0.0, _force_tare_sec_arg(args))
    parallel = len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1
    tare_total_sec = tare_sec if parallel else tare_sec * len(backends)
    return max(0.5, tare_total_sec + 0.5)

__all__ = [
    "_acceleration_limited_position_execution_active",
    "_active_command_backend",
    "_active_impedance_force_backends",
    "_apply_active_arm_command_to_preview_real",
    "_command_backend_for_side",
    "_estimate_impedance_force_control_warmup_sec",
    "_impedance_force_control_not_ready_sides",
    "_impedance_force_control_warmup_backends",
    "_runtime_arm_backend_state",
    "_sync_dual_arm_retargeter_for_real_control",
]
