"""Hand/gripper runtime helpers shared by the real-teleop coordinator and GUI.

The module owns command projection, backend/retargeter lookup, read-only status
inputs, and explicit tactile-solving configuration.  It does not construct
devices or own the arm/control-loop lifecycle.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping

import numpy as np

from robot_motion.runtime import TeleopCommand, merge_teleop_commands

from ..backends.grippers import (
    DMTacGripperBackend,
    normalize_dmtac_tactile_fields,
)
from ..backends.hands import InspireTacHandBackend
from ..hand_mapping import parse_tactile_regions
from ..retargeters import ScalarToGripperRetargeter
from ..sources.hand import AngleTriggerSource, HandQ6Source
from .components import RealTeleopComponents
from .hand_config import (
    HAND_BACKEND_DMTAC_GRIPPER,
    dmtac_sides as _dmtac_sides,
)

def _hand_q6_source_from_components(components: RealTeleopComponents) -> HandQ6Source | None:
    for source in components.session.sources:
        if isinstance(source, HandQ6Source):
            return source
    return None


def _gripper_input_channel_name(
    components: RealTeleopComponents,
    side: str,
) -> str | None:
    """Return the input channel, or ``None`` when input is intentionally disabled."""

    source = str(getattr(components, "gripper_input_source", "q6") or "q6")
    if source == "none":
        return None
    return f"{side}_angle_trigger" if source == "angle_trigger" else f"{side}_hand_q6"


def _gripper_input_source_from_components(components: RealTeleopComponents):
    for source in components.session.sources:
        if isinstance(source, (HandQ6Source, AngleTriggerSource)):
            return source
    return None


def _dmtac_command_contract_configured(
    *,
    gripper_input_source: str,
    angle_trigger_source: object | None,
) -> bool:
    """Whether the configured input mode has a stable command contract.

    Q6 is reconnectable, so its schema exists even when no serial device was
    present during startup.  Angle-trigger uses the same registered scalar
    command vocabulary; source startup failure makes its channel unavailable
    but must not erase the writer schema either.  ``none`` deliberately has no
    operator-input command contract; the DMTAC backend can still expose
    measured state and tactile telemetry independently.
    """

    return bool(str(gripper_input_source) in {"q6", "angle_trigger"})


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    return parse_tactile_regions(value)




def _extract_hand_command_from_command(command: TeleopCommand | None) -> TeleopCommand | None:
    if command is None:
        return None
    hand_command = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
    for effector_name in ("left_hand", "right_hand"):
        effector_command = command.commands.get(effector_name)
        if effector_command is not None:
            hand_command.add_command(effector_command.copy())
    if not hand_command.commands:
        return None
    return hand_command


def _build_runtime_hand_command_for_recovery(
    components: RealTeleopComponents,
    observation,
) -> TeleopCommand | None:
    if observation is None:
        return None
    hand_commands: list[TeleopCommand] = []
    if components.left_hand_retargeter is not None:
        hand_commands.append(
            components.left_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if components.right_hand_retargeter is not None:
        hand_commands.append(
            components.right_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if not hand_commands:
        return None
    merged = merge_teleop_commands(hand_commands)
    if not merged.commands:
        return None
    return merged


def _merge_runtime_hand_command(
    active_command: TeleopCommand | None,
    hand_command: TeleopCommand | None,
) -> TeleopCommand | None:
    if hand_command is None:
        return active_command
    if active_command is None:
        return None
    return merge_teleop_commands((active_command, hand_command))


def _hand_tactile_sum(backend: InspireTacHandBackend | DMTacGripperBackend | None) -> float | None:
    if backend is None:
        return None
    if isinstance(backend, DMTacGripperBackend):
        state = backend.get_state(include_tactile_vector=False)
    else:
        state = backend.get_state()
    tactile = state.get("tactile")
    if tactile is not None:
        array = np.asarray(tactile, dtype=float).reshape(-1)
        if array.size > 0 and np.all(np.isfinite(array)):
            return float(np.sum(array))
    tactile_dict = state.get("tactile_dict")
    if isinstance(tactile_dict, dict) and tactile_dict:
        total = 0.0
        found = False
        for value in tactile_dict.values():
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size == 0 or not np.all(np.isfinite(array)):
                continue
            total += float(np.sum(array))
            found = True
        if found:
            return total
    return None


def _emit_tactile_sum_only_status(
    components: RealTeleopComponents,
    *,
    now: float,
    stream,
) -> None:
    timestamp = time.strftime("%H:%M:%S", time.localtime(now))
    left_sum = _hand_tactile_sum(components.left_hand_backend)
    right_sum = _hand_tactile_sum(components.right_hand_backend)
    left_text = "None" if left_sum is None else f"{left_sum:.4f}"
    right_text = "None" if right_sum is None else f"{right_sum:.4f}"
    stream.write(f"[{timestamp}] [TactileSum] left={left_text} right={right_text}\n")
    stream.flush()


def _hand_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> InspireTacHandBackend | DMTacGripperBackend | None:
    if side == "left":
        return components.left_hand_backend
    if side == "right":
        return components.right_hand_backend
    return None


def _gripper_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> DMTacGripperBackend | None:
    backend = _hand_backend_for_side(components, side)
    return backend if isinstance(backend, DMTacGripperBackend) else None


def _gripper_state_scalar_from_payload(payload: object) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("joint_pos", "gripper_measured_scalar", "gripper_state_scalar"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    raw_state = payload.get("raw_gripper_state")
    if isinstance(raw_state, Mapping):
        return _gripper_state_scalar_from_payload(raw_state)
    return None


def _read_gripper_backend_scalar_now(backend: object) -> float | None:
    read_now = getattr(backend, "read_gripper_state_now", None)
    if callable(read_now):
        return _gripper_state_scalar_from_payload(read_now())
    get_state = getattr(backend, "get_state", None)
    if not callable(get_state):
        return None
    try:
        payload = get_state(
            include_tactile_arrays=False,
            include_tactile_vector=False,
            copy_tactile_arrays=False,
        )
    except TypeError:
        payload = get_state()
    return _gripper_state_scalar_from_payload(payload)


def _hold_requested_pause_reason(hold_reason: object) -> str:
    detail = str(hold_reason or "").strip() or "unspecified"
    return f"hold_requested:{detail}"




def _gripper_retargeter_for_side(
    components: RealTeleopComponents,
    side: str,
) -> ScalarToGripperRetargeter | None:
    if side == "left":
        return components.left_gripper_retargeter
    if side == "right":
        return components.right_gripper_retargeter
    return None


def _has_dmtac_gripper_backend(components: RealTeleopComponents) -> bool:
    return any(_gripper_backend_for_side(components, side) is not None for side in components.active_sides)


def _has_q6_gripper_control(components: RealTeleopComponents) -> bool:
    return any(_gripper_retargeter_for_side(components, side) is not None for side in components.active_sides)


def _gripper_gui_available(components: RealTeleopComponents) -> bool:
    return (
        str(getattr(components, "hand_backend_mode", "") or "") == HAND_BACKEND_DMTAC_GRIPPER
        or _has_dmtac_gripper_backend(components)
        or _has_q6_gripper_control(components)
    )


def _tactile_visualization_available(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> bool:
    if _dmtac_sides(args, components.active_sides):
        for side in components.active_sides:
            backend = _gripper_backend_for_side(components, side)
            tactile_fields = tuple(getattr(backend.config, "tactile_fields", ()) or ()) if backend is not None else ()
            if (
                backend is not None
                and bool(getattr(backend.config, "connect_tactile", False))
                and "distributed_force" in tactile_fields
                and str(
                    getattr(backend.config, "tactile_calibration_policy", "auto")
                ) == "required"
            ):
                return True
        return False
    return any(_hand_backend_for_side(components, side) is not None for side in components.active_sides)


def _dmtac_tactile_backends(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> tuple[DMTacGripperBackend, ...]:
    if not _dmtac_sides(args, components.active_sides):
        return ()
    return tuple(
        backend
        for side in components.active_sides
        if (backend := _gripper_backend_for_side(components, side)) is not None
        and bool(getattr(backend.config, "connect_tactile", False))
    )


def _tactile_solving_available(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> bool:
    backends = _dmtac_tactile_backends(args, components)
    return bool(backends) and all(
        bool(getattr(backend.config, "tactile_process_read", False))
        and bool(tuple(getattr(backend.config, "tactile_ids", ()) or ()))
        for backend in backends
    )


def _tactile_solving_active(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> bool:
    backends = _dmtac_tactile_backends(args, components)
    return bool(backends) and all(
        "distributed_force" in tuple(getattr(backend.config, "tactile_fields", ()) or ())
        and str(getattr(backend.config, "tactile_calibration_policy", "auto")) == "required"
        for backend in backends
    )


def _enable_gui_tactile_solving(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> int:
    backends = _dmtac_tactile_backends(args, components)
    if not backends:
        raise RuntimeError("没有已配置的 DMTAC 触觉采集 backend。")
    changed = sum(bool(backend.enable_tactile_solving()) for backend in backends)
    fields: list[str] = []
    for backend in backends:
        fields.extend(tuple(getattr(backend.config, "tactile_fields", ()) or ()))
    args.dmtac_tactile_fields = normalize_dmtac_tactile_fields(
        tuple(dict.fromkeys(fields))
    )
    args.dmtac_tactile_compute_profile = "dist-only"
    args.dmtac_tactile_calibration_policy = "required"
    args.tactile_solving_active = _tactile_solving_active(args, components)
    args.tactile_visualization_available = _tactile_visualization_available(
        args, components
    )
    return int(changed)













__all__ = [
    "_build_runtime_hand_command_for_recovery",
    "_dmtac_command_contract_configured",
    "_dmtac_tactile_backends",
    "_emit_tactile_sum_only_status",
    "_enable_gui_tactile_solving",
    "_extract_hand_command_from_command",
    "_gripper_backend_for_side",
    "_gripper_gui_available",
    "_gripper_input_channel_name",
    "_gripper_input_source_from_components",
    "_gripper_retargeter_for_side",
    "_gripper_state_scalar_from_payload",
    "_hand_backend_for_side",
    "_hand_q6_source_from_components",
    "_hand_tactile_sum",
    "_has_dmtac_gripper_backend",
    "_has_q6_gripper_control",
    "_hold_requested_pause_reason",
    "_merge_runtime_hand_command",
    "_parse_hand_tactile_regions",
    "_read_gripper_backend_scalar_now",
    "_tactile_solving_active",
    "_tactile_solving_available",
    "_tactile_visualization_available",
]
