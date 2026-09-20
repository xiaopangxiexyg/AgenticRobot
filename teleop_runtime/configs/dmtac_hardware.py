from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_PORT,
    GRIPPER_ACTION_MAX,
    DMTacGripperBackendConfig,
)

DMTAC_ASSIGNMENT_DEFAULT = "latest"


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_dmtac_assignment_arguments(
    parser: argparse.ArgumentParser,
    *,
    assignment_includes_tactile: bool = True,
) -> None:
    """Add the shared DMTAC transport/assignment CLI contract.

    The destination names intentionally preserve the existing runtime API.
    Older, confusing option spellings remain hidden aliases so saved launch
    scripts keep working while ``--help`` presents one canonical vocabulary.
    """

    scope = "gripper/tactile" if assignment_includes_tactile else "gripper"
    group = parser.add_argument_group("DMTAC hardware assignment")
    for side in ("left", "right"):
        group.add_argument(
            f"--{side}-gripper-port",
            default="",
            help=(
                f"Manual {side} DMTAC gripper serial port override. This replaces "
                f"the assignment for that side and does not attach tactile sensors."
            ),
        )
        group.add_argument(
            f"--{side}-gripper-id",
            type=_positive_int_arg,
            default=1,
            help=f"Modbus id used with --{side}-gripper-port (default: 1).",
        )
        group.add_argument(
            f"--{side}-gripper-baudrate",
            type=_positive_int_arg,
            default=115200,
            help=f"Baudrate used with --{side}-gripper-port (default: 115200).",
        )
        group.add_argument(
            f"--{side}-gripper-timeout",
            type=_positive_float_arg,
            default=0.10,
            help=f"Serial timeout in seconds used with --{side}-gripper-port (default: 0.10).",
        )
    group.add_argument(
        "--dmtac-config",
        default=DMTAC_ASSIGNMENT_DEFAULT,
        help=f"DMTAC {scope} assignment path or 'latest' (default: latest).",
    )
    group.add_argument(
        "--dmtac-use-config",
        dest="dmtac_default_config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the shared assignment before applying per-side manual overrides.",
    )
    group.add_argument(
        "--no-dmtac-default-config",
        dest="dmtac_default_config",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--require-dmtac-config",
        dest="dmtac_config_required",
        action="store_true",
        default=False,
        help="Fail if any active DMTAC side cannot be resolved from config/overrides.",
    )
    group.add_argument(
        "--dmtac-config-required",
        dest="dmtac_config_required",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--dmtac-allow-missing-devices",
        dest="dmtac_allow_missing",
        action="store_true",
        default=False,
        help="Allow devices saved in the assignment to be absent during USB validation.",
    )
    group.add_argument(
        "--dmtac-allow-missing",
        dest="dmtac_allow_missing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--dmtac-validate-usb",
        dest="dmtac_validate_current_usb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare the saved assignment with currently reachable USB devices.",
    )
    group.add_argument(
        "--dmtac-no-validate-current-usb",
        dest="dmtac_validate_current_usb",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--dmtac-prompt-on-extra",
        dest="dmtac_assume_yes_for_extra",
        action="store_false",
        default=True,
        help="Require terminal confirmation when additional unassigned DMTAC devices are detected.",
    )
    group.add_argument(
        "--dmtac-confirm-extra",
        dest="dmtac_assume_yes_for_extra",
        action="store_false",
        help=argparse.SUPPRESS,
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


def prepare_dmtac_hardware_assignment_before_gui(
    config: str | Path = "latest",
    *,
    input_func: Callable[[str], str] = input,
    reconfigure_func: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Resolve DMTac assignment, including terminal confirmation, before GUI startup."""

    usb_assignment = _usb_assignment_module()
    while True:
        assignment = usb_assignment.load_assignment(config)
        scan = usb_assignment.scan_usb_devices(
            probe_grippers=True,
            scan_gripper_ids=False,
            scan_baudrates=(),
        )
        comparison = usb_assignment.compare_assignment_to_scan(assignment, scan)
        completed = _unique_dmtac_gripper_completion(
            usb_assignment,
            assignment=assignment,
            scan=scan,
            comparison=comparison,
        )
        if completed is not None:
            side, current, completed_assignment = completed
            port = str(current.get("preferred_port") or current.get("port") or current.get("device"))
            answer = _terminal_assignment_answer(input_func,
                "DMTac detected exactly one missing configured gripper and one remaining "
                f"reachable candidate. Infer {side}={port}. Type YES to confirm: "
            )
            if answer.strip() == "YES":
                source_path = Path(str(assignment.get("_config_path") or "")).expanduser()
                config_dir = source_path.parent if str(source_path) else None
                usb_assignment.save_assignment(
                    completed_assignment,
                    config_dir=config_dir,
                )
                completed_comparison = usb_assignment.compare_assignment_to_scan(
                    completed_assignment, scan
                )
                usb_assignment.require_compatible_or_confirm(
                    completed_comparison,
                    input_func=input_func,
                )
                return completed_comparison["effective_assignment"]
        elif not comparison.get("missing"):
            usb_assignment.require_compatible_or_confirm(
                comparison,
                input_func=input_func,
            )
            return comparison["effective_assignment"]

        text = usb_assignment.format_comparison(comparison)
        if text:
            print(text)
        answer = _terminal_assignment_answer(input_func,
            "DMTac left/right assignment cannot be uniquely completed. "
            "Type RECONFIGURE to run terminal reassignment, or ABORT: "
        )
        if answer.strip().upper() != "RECONFIGURE":
            raise RuntimeError(
                "DMTac assignment remains unresolved; GUI was not started."
            )
        if reconfigure_func is None:
            from hardware.DMTAC_gripper.portable_dmtac_gripper.cli import main

            main(
                [
                    "configure-usb",
                    "--manual-all-slots",
                    "--merge-latest",
                    "--save",
                ]
            )
        else:
            reconfigure_func()


def _terminal_assignment_answer(
    input_func: Callable[[str], str], prompt: str
) -> str:
    try:
        return str(input_func(prompt))
    except (EOFError, OSError) as exc:
        raise RuntimeError(
            "DMTac assignment requires terminal confirmation before GUI startup, "
            "but no interactive terminal input is available."
        ) from exc


def _unique_dmtac_gripper_completion(
    usb_assignment: Any,
    *,
    assignment: Mapping[str, Any],
    scan: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    missing = list(comparison.get("missing") or [])
    extra = list(comparison.get("extra") or [])
    missing_grippers = [item for item in missing if item.get("kind") == "gripper"]
    extra_grippers = [item for item in extra if item.get("kind") == "gripper"]
    if len(missing) != 1 or len(missing_grippers) != 1:
        return None
    if len(extra) != 1 or len(extra_grippers) != 1:
        return None
    current = dict(extra_grippers[0].get("current") or {})
    probe = current.get("probe") or {}
    if probe.get("reachable") is not True:
        return None
    side = str(missing_grippers[0].get("side") or "")
    if side not in {"left", "right"}:
        return None
    token = str(
        current.get("preferred_port")
        or current.get("port")
        or current.get("device")
        or ""
    )
    if not token:
        return None
    partial = usb_assignment.build_assignment_from_tokens(
        scan,
        left_gripper=token if side == "left" else None,
        right_gripper=token if side == "right" else None,
    )
    completed = usb_assignment.merge_assignment(assignment, partial)
    verified = usb_assignment.compare_assignment_to_scan(completed, scan)
    if verified.get("missing") or verified.get("extra"):
        return None
    return side, current, completed


def dmtac_backend_configs_from_assignment(
    assignment: Mapping[str, Any],
    *,
    active_sides: Sequence[str] = ("left", "right"),
    prefer_existing_by_path_port: bool = False,
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
                _assigned_gripper_runtime_port(
                    gripper,
                    prefer_existing_by_path_port=prefer_existing_by_path_port,
                )
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


def _assigned_gripper_runtime_port(
    gripper: Mapping[str, Any],
    *,
    prefer_existing_by_path_port: bool,
) -> str:
    configured_port = str(gripper["port"])
    if not prefer_existing_by_path_port:
        return configured_port

    # A saved ttyUSB number can change when only one member of a configured
    # pair is present.  Selecting an intentionally disconnected side forbids a
    # probing scan of that device, so use the persisted topology alias for the
    # still-connected side when that alias currently resolves.  Never guess a
    # different ttyUSB endpoint: if no stable alias exists, backend startup
    # remains fail-closed on the configured port.
    aliases = tuple(
        str(alias)
        for alias in gripper.get("port_aliases", ())
        if str(alias)
    )
    for alias in aliases:
        path = Path(alias)
        if (
            path.parent.name == "by-path"
            and path.parent.parent.name == "serial"
            and path.exists()
        ):
            return alias
    return configured_port


def require_dmtac_sides_resolved(
    requested_sides: Sequence[str],
    resolved_configs: Mapping[str, DMTacGripperBackendConfig],
    *,
    config: str | Path,
) -> None:
    """Fail closed when a required DMTAC side has no usable backend config."""

    missing = [str(side) for side in requested_sides if str(side) not in resolved_configs]
    if missing:
        raise RuntimeError(
            "DMTAC hardware assignment "
            f"{str(config)!r} does not resolve active DMTAC sides: {missing}. "
            "Update the assignment or provide a complete per-side manual transport override."
        )


def manual_dmtac_backend_config_for_side(
    args: argparse.Namespace,
    side: str,
    policy: Mapping[str, Any],
    *,
    base_config: DMTacGripperBackendConfig | None = None,
) -> DMTacGripperBackendConfig | None:
    """Apply an explicit port override to one side's backend config.

    When ``base_config`` comes from the saved assignment or GUI preflight,
    preserve its tactile binding and transport settings and replace only the
    serial endpoint.  Without a base config, retain the legacy gripper-only
    manual-port fallback.
    """

    side_name = str(side)
    port = str(getattr(args, f"{side_name}_gripper_port", "") or "").strip()
    if not port:
        return None
    if base_config is not None:
        base_config.gripper_port = port
        base_config.connect_hardware = True
        return base_config
    return DMTacGripperBackendConfig(
        side=side_name,
        effector_name=f"{side_name}_hand",
        gripper_port=port,
        gripper_id=int(getattr(args, f"{side_name}_gripper_id", 1) or 1),
        gripper_baudrate=int(
            getattr(args, f"{side_name}_gripper_baudrate", 115200) or 115200
        ),
        gripper_timeout=float(
            getattr(args, f"{side_name}_gripper_timeout", 0.10) or 0.10
        ),
        command_dim=int(policy["output_dim"]),
        speed=float(policy["speed"]),
        torque=float(policy["torque"]),
        max_action=float(policy["max_action"]),
        send_threshold=float(policy["send_threshold"]),
        command_hz=float(policy["command_hz"]),
        continuous_grasp=bool(policy["continuous_grasp"]),
        enable_telemetry=True,
        telemetry_hz=float(policy["telemetry_hz"]),
        connect_tactile=False,
        tactile_ids=(),
    )


def apply_dmtac_gripper_port_overrides(
    configs: Mapping[str, DMTacGripperBackendConfig],
    args: argparse.Namespace,
    *,
    active_sides: Sequence[str],
) -> dict[str, DMTacGripperBackendConfig]:
    """Apply explicit ports without replacing assignment metadata.

    A manual port override is intentionally a gripper-only fallback and does
    not attach tactile devices.  When an assignment already resolves a side,
    this override keeps its tactile ids and transport settings while changing
    only the runtime serial endpoint.
    """

    resolved = dict(configs)
    for side in active_sides:
        side_name = str(side)
        port = str(getattr(args, f"{side_name}_gripper_port", "") or "").strip()
        if not port or side_name not in resolved:
            continue
        resolved[side_name].gripper_port = port
        resolved[side_name].connect_hardware = True
    return resolved


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
