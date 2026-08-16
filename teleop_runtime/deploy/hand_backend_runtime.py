"""Deployment-owned hand backend construction and startup diagnostics.

The deployment application resolves CLI/profile inputs and supplies an
explicit backend-mode/config plan here.  This module owns concrete hand
backend instantiation plus the existing checked startup operation, but it does
not choose application startup order, dispatch actions, or close resources.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

from robot_motion.action import (
    HandActionSpec,
    hand_action_specs_metadata,
)
from robot_motion.deploy.schema import ALL_SIDES

from ..backends.grippers import (
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
)
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..hand_mapping import parse_tactile_regions
from ..interfaces import EffectorBackend
from .hand_backend_config import (
    HAND_BACKEND_CHOICES,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_INSPIRE,
)


def build_hand_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    backend_mode_by_side: Mapping[str, str],
    dmtac_configs: Mapping[str, DMTacGripperBackendConfig],
    disconnected_gripper_sides: tuple[str, ...],
    disconnected_gripper_policy: Mapping[str, Any],
) -> dict[str, EffectorBackend]:
    """Instantiate hand backends from application-resolved configuration."""

    backends: dict[str, EffectorBackend] = {
        side: DMTacGripperBackend(config)
        for side, config in dmtac_configs.items()
        if side in active_sides
        and backend_mode_by_side[side] == HAND_BACKEND_DMTAC_GRIPPER
    }
    if disconnected_gripper_sides:
        for side in disconnected_gripper_sides:
            if side not in active_sides:
                continue
            backends[side] = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side=side,
                    effector_name=f"{side}_hand",
                    connect_hardware=False,
                    connect_tactile=False,
                    send_commands=False,
                    enable_telemetry=False,
                    command_dim=int(disconnected_gripper_policy["output_dim"]),
                    max_action=float(disconnected_gripper_policy["max_action"]),
                    disconnected_state_scalar=float(
                        args.disconnected_gripper_state
                    ),
                )
            )

    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    observation_only = bool(getattr(args, "rtc_real_shadow", False))
    for side in active_sides:
        if backend_mode_by_side[side] != HAND_BACKEND_INSPIRE:
            continue
        hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
        backends[side] = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=hand_ip or "0.0.0.0",
                port=int(args.hand_modbus_port),
                side=side,
                effector_name=f"{side}_hand",
                connect_hardware=bool(hand_ip),
                observation_only=observation_only,
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tuple(tactile_regions),
            )
        )
    return backends


def disconnected_gripper_selection_label(
    sides: set[str] | tuple[str, ...],
) -> str:
    selected = tuple(side for side in ALL_SIDES if side in set(sides))
    if not selected:
        return "none"
    if selected == tuple(ALL_SIDES):
        return "both"
    return selected[0]


def hand_backend_startup_diagnostic(
    *,
    side: str,
    backend: EffectorBackend,
    disconnected_gripper_sides: set[str] | tuple[str, ...],
) -> dict[str, Any]:
    state = backend.get_state()
    selected_sides = set(disconnected_gripper_sides)
    ready = bool(backend.is_ready())
    recommended_sides = selected_sides | {side}
    return {
        "side": side,
        "ready": ready,
        "backend_type": state.get("backend_type", type(backend).__name__),
        "selected_for_intentional_disconnect": side in selected_sides,
        "configured_disconnected_gripper_sides": disconnected_gripper_selection_label(
            selected_sides
        ),
        "backend_intentionally_disconnected": bool(
            state.get("intentionally_disconnected", False)
        ),
        "connect_hardware": state.get("connect_hardware"),
        "is_connected": state.get("is_connected"),
        "gripper_is_connected": state.get("gripper_is_connected"),
        "gripper_port": state.get("gripper_port"),
        "telemetry_error": state.get("telemetry_error"),
        "intentional_disconnect_recommendation": (
            disconnected_gripper_selection_label(recommended_sides)
            if not ready
            else None
        ),
    }


def start_hand_backends_checked(
    backends: Mapping[str, EffectorBackend],
    *,
    disconnected_gripper_sides: set[str] | tuple[str, ...],
    allow_connect_failure: bool,
    debug: bool,
) -> None:
    for side, backend in backends.items():
        backend.start()
        diagnostic = hand_backend_startup_diagnostic(
            side=side,
            backend=backend,
            disconnected_gripper_sides=disconnected_gripper_sides,
        )
        ready = bool(diagnostic["ready"])
        if debug or not ready:
            print(
                "[Deploy][HandStartup] "
                + json.dumps(diagnostic, sort_keys=True, ensure_ascii=False),
                flush=True,
            )
        if ready or allow_connect_failure:
            continue

        configured_selection = diagnostic[
            "configured_disconnected_gripper_sides"
        ]
        recommendation = diagnostic["intentional_disconnect_recommendation"]
        if bool(diagnostic["selected_for_intentional_disconnect"]):
            selection_explanation = (
                f"{side} was selected for intentional disconnection, but its "
                "no-I/O backend did not become ready; this is an internal "
                "configuration error."
            )
        else:
            selection_explanation = (
                f"{side} remains required because it is not selected by "
                f"--disconnected-gripper-sides={configured_selection}."
            )
        raise RuntimeError(
            f"{side} hand backend is not ready. {selection_explanation} "
            f"backend_type={diagnostic['backend_type']!r} "
            f"port={diagnostic['gripper_port']!r} "
            f"telemetry_error={diagnostic['telemetry_error']!r}. "
            f"If {side} is intentionally offline, use "
            f"--disconnected-gripper-sides {recommendation}; otherwise restore "
            "that device or correct its configured port."
        )


def hand_action_specs_for_modes(
    backend_mode_by_side: Mapping[str, str],
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    specs = {}
    for side in active_sides:
        if backend_mode_by_side[side] == HAND_BACKEND_DMTAC_GRIPPER:
            specs[side] = HandActionSpec.dmtac_gripper(side)
        else:
            specs[side] = HandActionSpec.inspire_tac(side)
    return hand_action_specs_metadata(specs)


def filter_policy_qpos_init_hand_action_specs(
    hand_action_specs: Mapping[str, Any],
    *,
    disconnected_gripper_sides: tuple[str, ...],
) -> dict[str, Any]:
    """Exclude no-I/O grippers from physical dataset-start initialization."""

    disconnected_sides = set(disconnected_gripper_sides)
    return {
        side: spec
        for side, spec in hand_action_specs.items()
        if side not in disconnected_sides
    }


__all__ = [
    "HAND_BACKEND_CHOICES",
    "HAND_BACKEND_DMTAC_GRIPPER",
    "HAND_BACKEND_INSPIRE",
    "build_hand_backends",
    "disconnected_gripper_selection_label",
    "filter_policy_qpos_init_hand_action_specs",
    "hand_action_specs_for_modes",
    "hand_backend_startup_diagnostic",
    "start_hand_backends_checked",
]
