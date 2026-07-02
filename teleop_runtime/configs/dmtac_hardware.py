from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_PORT,
    DMTacGripperBackendConfig,
    GRIPPER_ACTION_MAX,
)


def _dmtac_root() -> Path:
    return Path(__file__).resolve().parents[2] / "hardware" / "DMTAC_gripper"


def _ensure_dmtac_path() -> None:
    text = str(_dmtac_root())
    if text not in sys.path:
        sys.path.insert(0, text)


def _usb_assignment_module():
    _ensure_dmtac_path()
    from portable_dmtac_gripper import usb_assignment

    return usb_assignment


def load_dmtac_hardware_assignment(
    config: str | Path = "latest",
    *,
    validate_current_usb: bool = True,
    allow_missing: bool = False,
    assume_yes_for_extra: bool = False,
) -> dict[str, Any]:
    usb_assignment = _usb_assignment_module()
    assignment = usb_assignment.load_assignment(config)
    if not validate_current_usb:
        return assignment
    scan = usb_assignment.scan_usb_devices(
        probe_grippers=True,
        scan_gripper_ids=False,
        scan_baudrates=(),
    )
    comparison = usb_assignment.compare_assignment_to_scan(assignment, scan)
    usb_assignment.require_compatible_or_confirm(
        comparison,
        allow_missing=allow_missing,
        assume_yes=assume_yes_for_extra,
    )
    return comparison["effective_assignment"]


def dmtac_backend_configs_from_assignment(
    assignment: Mapping[str, Any],
    *,
    active_sides: Sequence[str] = ("left", "right"),
    tactile_backend: str = "cpu",
    speed: float = 1.0,
    torque: float = 0.2,
    max_action: float = GRIPPER_ACTION_MAX,
) -> dict[str, DMTacGripperBackendConfig]:
    usb_assignment = _usb_assignment_module()
    configs: dict[str, DMTacGripperBackendConfig] = {}
    for side in active_sides:
        side_name = str(side)
        gripper = usb_assignment.gripper_for_side(assignment, side_name)
        tactile_ids = usb_assignment.tactile_ids_for_side(assignment, side_name)
        if gripper is None and not tactile_ids:
            continue
        configs[side_name] = DMTacGripperBackendConfig(
            side=side_name,
            effector_name=f"{side_name}_hand",
            gripper_port=(
                str(gripper["port"])
                if gripper is not None
                else DEFAULT_DMTAC_GRIPPER_PORT
            ),
            gripper_id=int(gripper.get("gripper_id", 1)) if gripper is not None else 1,
            gripper_baudrate=int(gripper.get("baudrate", 115200)) if gripper is not None else 115200,
            gripper_timeout=float(gripper.get("timeout", 0.03)) if gripper is not None else 0.03,
            connect_hardware=gripper is not None,
            connect_tactile=bool(tactile_ids),
            tactile_ids=tuple(tactile_ids),
            tactile_backend=str(tactile_backend),
            speed=float(speed),
            torque=float(torque),
            max_action=float(max_action),
        )
    return configs


def load_latest_dmtac_backend_configs(
    *,
    active_sides: Sequence[str] = ("left", "right"),
    validate_current_usb: bool = True,
    allow_missing: bool = False,
    assume_yes_for_extra: bool = False,
    tactile_backend: str = "cpu",
    speed: float = 1.0,
    torque: float = 0.2,
    max_action: float = GRIPPER_ACTION_MAX,
) -> dict[str, DMTacGripperBackendConfig]:
    assignment = load_dmtac_hardware_assignment(
        "latest",
        validate_current_usb=validate_current_usb,
        allow_missing=allow_missing,
        assume_yes_for_extra=assume_yes_for_extra,
    )
    return dmtac_backend_configs_from_assignment(
        assignment,
        active_sides=active_sides,
        tactile_backend=tactile_backend,
        speed=speed,
        torque=torque,
        max_action=max_action,
    )
