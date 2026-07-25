"""Post-parse hardware and loop configuration for real teleoperation.

This module owns configuration resolution only.  It does not construct or start
hardware.  The legacy entrypoint refreshes the support namespace before each
call so its historical monkeypatch seams remain effective.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any


_REQUIRED_SUPPORT_NAMES = (
    "DEFAULT_CAMERA_CONFIG_PATH",
    "DevicePreflightError",
    "DevicePreflightOptions",
    "_active_sides_from_args",
    "_apply_gripper_policy_to_dmtac_config",
    "_dmtac_sides",
    "_dmtac_tactile_enabled",
    "_effective_arm_control_hz",
    "_effective_outer_control_hz",
    "_log",
    "_resolve_logical_side_assignments",
    "_resolved_gripper_policies_by_side",
    "_resolved_gripper_policy",
    "arm_ips_for_sides",
    "build_rm65_arm_ip_assignment",
    "dmtac_backend_configs_from_assignment",
    "load_dmtac_hardware_assignment",
    "load_latest_q6_ports",
    "load_rm65_arm_ip_assignment",
    "manual_dmtac_backend_config_for_side",
    "prepare_devices_for_teleop",
    "require_dmtac_sides_resolved",
    "save_rm65_arm_ip_assignment_for_config",
)


def bind_legacy_support(namespace: Mapping[str, Any]) -> None:
    """Refresh dependencies owned by the compatibility entrypoint."""

    missing = [name for name in _REQUIRED_SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "real-teleop CLI runtime configuration support is incomplete: "
            + ", ".join(missing)
        )
    module_globals = globals()
    for name in _REQUIRED_SUPPORT_NAMES:
        module_globals[name] = namespace[name]


def _apply_default_q6_ports(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    args.q6_config_loaded_ports = {}
    if not _dmtac_sides(args, _active_sides_from_args(args)):
        return
    if str(getattr(args, "gripper_input_source", "q6")) != "q6":
        return
    if not bool(getattr(args, "q6_default_config", True)):
        return
    active_sides = tuple(str(side) for side in _active_sides_from_args(args))
    missing_sides = tuple(
        side
        for side in active_sides
        if side in {"left", "right"} and not str(getattr(args, f"{side}_hand_port", "") or "")
    )
    if not missing_sides:
        return
    try:
        loaded_ports = load_latest_q6_ports(
            active_sides=missing_sides,
            config=str(getattr(args, "q6_config", "latest") or "latest"),
            allow_missing=not bool(getattr(args, "q6_config_required", False)),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to load hand Q6 assignment {getattr(args, 'q6_config', 'latest')!r}: {exc}"
        if parser is not None:
            parser.error(message)
        raise RuntimeError(message) from exc
    args.q6_config_loaded_ports = dict(loaded_ports)
    for side, port in loaded_ports.items():
        if side in missing_sides and port:
            setattr(args, f"{side}_hand_port", str(port))


def _run_gui_device_preflight(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
    parser: argparse.ArgumentParser,
) -> None:
    if not bool(getattr(args, "gui_requested", False)):
        return
    if not bool(getattr(args, "device_preflight", True)):
        return
    dmtac_sides = _dmtac_sides(args, active_sides)
    require_dmtac_stack = bool(dmtac_sides)
    require_q6 = require_dmtac_stack and str(getattr(args, "gripper_input_source", "q6")) == "q6"
    if (
        require_dmtac_stack
        and bool(getattr(args, "visualize_tactile", False))
        and getattr(args, "dmtac_tactile", None) is None
    ):
        args.dmtac_tactile = True
    require_tactile = bool(getattr(args, "device_preflight_require_tactile", True)) and _dmtac_tactile_enabled(args)
    if getattr(args, "dmtac_tactile", None) is False:
        require_tactile = False

    options = DevicePreflightOptions(
        active_sides=active_sides,
        camera_config=str(getattr(args, "camera_config", DEFAULT_CAMERA_CONFIG_PATH)),
        require_camera=not bool(getattr(args, "no_camera", False)),
        require_q6=require_q6,
        require_dmtac_gripper=require_dmtac_stack,
        require_dmtac_tactile=bool(require_dmtac_stack and require_tactile),
        auto_configure=bool(getattr(args, "device_auto_configure", True)),
        force_reconfigure=bool(getattr(args, "device_preflight_force_reconfigure", False)),
        assume_yes=bool(getattr(args, "device_preflight_yes", False)),
        fix_permissions=bool(getattr(args, "device_fix_permissions", False)),
        probe_all_serial=bool(getattr(args, "device_preflight_probe_all_serial", False)),
        q6_config=str(getattr(args, "q6_config", "latest") or "latest"),
        dmtac_config=str(getattr(args, "dmtac_config", "latest") or "latest"),
        dmtac_tactile_backend=str(getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"),
    )
    try:
        result = prepare_devices_for_teleop(options)
    except DevicePreflightError as exc:
        parser.error(f"device preflight failed: {exc}")
    if not bool(getattr(result, "ok", True)):
        parser.error("device preflight permission fix failed; rerun the permission script with sudo.")
    for message in result.messages:
        for line in str(message).splitlines():
            _log("DevicePreflight", line)
    if getattr(result, "dmtac_assignment", None) is not None:
        args._dmtac_preflight_assignment = result.dmtac_assignment
    if result.q6_ports:
        args.q6_config_loaded_ports = dict(result.q6_ports)
        for side, port in result.q6_ports.items():
            if side in active_sides:
                setattr(args, f"{side}_hand_port", str(port))


def _resolve_gripper_input_source(args: argparse.Namespace, active_sides: tuple[str, ...]) -> None:
    """Prefer usable Q6 input; fall back to the calibrated angle triggers."""
    if not _dmtac_sides(args, active_sides):
        args.gripper_input_source = "none"
        return
    if bool(getattr(args, "gripper_angle_trigger", False)):
        args.gripper_input_source = "angle_trigger"
        return
    try:
        from hardware.teleop_hand.q6_assignment import scan_q6_devices

        scan = scan_q6_devices(
            probe_all_serial=True,
            show_raw_lines=False,
        )
        expected = len(active_sides)
        if len(scan.get("q6_devices") or ()) >= expected:
            args.gripper_input_source = "q6"
            return
        if len(scan.get("q6_devices") or ()):
            raise ValueError(
                f"Detected only {len(scan.get('q6_devices') or ())} Q6 device(s), expected {expected}."
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - Q6 probing is optional when triggers are available.
        _log("GripperInput", f"Q6 probe unavailable; trying angle trigger: {type(exc).__name__}: {exc}")
    args.gripper_input_source = "angle_trigger"


def _argv_has_option(raw_argv: Sequence[str], option: str) -> bool:
    return any(str(item) == option or str(item).startswith(f"{option}=") for item in raw_argv)


def _apply_real_loop_timing_defaults(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    hz = max(1e-6, _effective_outer_control_hz(args))
    if not _argv_has_option(raw_argv, "--minimum-control-hz"):
        setattr(args, "minimum_control_hz", max(1.0, min(15.0, 0.9 * hz)))
        setattr(args, "minimum_control_hz_auto", True)
    else:
        setattr(args, "minimum_control_hz_auto", False)

    if not _argv_has_option(raw_argv, "--arm-step-dt-max-scale"):
        arm_control_hz = max(1e-6, _effective_arm_control_hz(args))
        required_scale = 1.1 * arm_control_hz / hz
        setattr(
            args,
            "arm_step_dt_max_scale",
            max(float(getattr(args, "arm_step_dt_max_scale", 3.0)), required_scale),
        )
        setattr(args, "arm_step_dt_max_scale_auto", True)
    else:
        setattr(args, "arm_step_dt_max_scale_auto", False)


def _source_arm_ip_fields_for_active_sides(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, str]:
    fields: dict[str, str] = {}
    for source_name, (logical_side, _arm_ip) in _resolve_logical_side_assignments(args).items():
        if logical_side in active_sides:
            fields[str(source_name).replace("_arm_ip", "")] = str(source_name)
    return fields


def _apply_default_arm_ips(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
) -> None:
    setattr(args, "arm_ip_config_loaded_sides", [])
    if not bool(getattr(args, "arm_ip_default_config", True)):
        return
    source_fields = _source_arm_ip_fields_for_active_sides(args, active_sides)
    missing_sources = [
        source_side
        for source_side, field in source_fields.items()
        if not str(getattr(args, field, "") or "").strip()
    ]
    if not missing_sources:
        return
    try:
        assignment = load_rm65_arm_ip_assignment(str(getattr(args, "arm_ip_config", "latest") or "latest"))
    except FileNotFoundError:
        return
    loaded = arm_ips_for_sides(assignment, sides=missing_sources)
    loaded_sides: list[str] = []
    for source_side, arm_ip in loaded.items():
        field = source_fields.get(source_side)
        if field and not str(getattr(args, field, "") or "").strip():
            setattr(args, field, str(arm_ip))
            loaded_sides.append(str(source_side))
    setattr(args, "arm_ip_config_loaded_sides", loaded_sides)


def _save_explicit_arm_ips_latest(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    if not bool(getattr(args, "arm_ip_save_latest", True)):
        return
    explicit_left = _argv_has_option(raw_argv, "--left-arm-ip") and bool(str(args.left_arm_ip or "").strip())
    explicit_right = _argv_has_option(raw_argv, "--right-arm-ip") and bool(str(args.right_arm_ip or "").strip())
    if not (explicit_left or explicit_right):
        return
    try:
        base = load_rm65_arm_ip_assignment(str(getattr(args, "arm_ip_config", "latest") or "latest"))
    except FileNotFoundError:
        base = None
    assignment = build_rm65_arm_ip_assignment(
        base=base,
        left_arm_ip=str(args.left_arm_ip).strip() if str(args.left_arm_ip or "").strip() else None,
        right_arm_ip=str(args.right_arm_ip).strip() if str(args.right_arm_ip or "").strip() else None,
    )
    saved = save_rm65_arm_ip_assignment_for_config(
        assignment,
        config=str(getattr(args, "arm_ip_config", "latest") or "latest"),
    )
    setattr(args, "arm_ip_config_saved_path", str(saved))


def _validate_required_arm_ips(args: argparse.Namespace, active_sides: tuple[str, ...]) -> None:
    logical_ip_by_side: dict[str, tuple[str, str]] = {}
    inactive_config: list[str] = []
    for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
        if not arm_ip:
            continue
        if logical_side not in active_sides:
            inactive_config.append(f"{source_name}->{logical_side}")
            continue
        logical_ip_by_side[logical_side] = (source_name, arm_ip)
    missing = [side for side in active_sides if side not in logical_ip_by_side]
    if missing:
        raise ValueError(
            "Real teleop requires an RM65 IP for every active logical side; "
            f"missing={missing} active_sides={list(active_sides)} "
            f"swap_arm_sides={bool(args.swap_arm_sides)}. "
            "Check that LEFT_ARM_IP/RIGHT_ARM_IP are exported in this shell, or pass "
            "--left-arm-ip/--right-arm-ip literal IPs."
        )
    if inactive_config:
        raise ValueError(
            "Inactive logical side received an arm IP; remove it or include that side "
            f"in --active-sides. inactive={inactive_config}"
        )


def _dmtac_backend_configs(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, DMTacGripperBackendConfig]:
    cached = getattr(args, "_dmtac_backend_configs", None)
    if isinstance(cached, dict):
        return dict(cached)
    dmtac_sides = _dmtac_sides(args, active_sides)
    if not dmtac_sides:
        setattr(args, "_dmtac_backend_configs", {})
        setattr(args, "dmtac_config_loaded_sides", [])
        return {}

    policy = _resolved_gripper_policy(args)
    policies_by_side = _resolved_gripper_policies_by_side(args, active_sides)
    configs: dict[str, DMTacGripperBackendConfig] = {}
    if bool(getattr(args, "dmtac_default_config", True)):
        preflight_assignment = getattr(args, "_dmtac_preflight_assignment", None)
        try:
            assignment = (
                dict(preflight_assignment)
                if isinstance(preflight_assignment, Mapping)
                else load_dmtac_hardware_assignment(
                    str(getattr(args, "dmtac_config", "latest") or "latest"),
                    validate_current_usb=bool(getattr(args, "dmtac_validate_current_usb", True)),
                    allow_missing=bool(getattr(args, "dmtac_allow_missing", False)),
                    assume_yes_for_extra=bool(getattr(args, "dmtac_assume_yes_for_extra", True)),
                )
            )
            for side in dmtac_sides:
                side_policy = policies_by_side.get(side, policy)
                configs.update(
                    dmtac_backend_configs_from_assignment(
                        assignment,
                        active_sides=(side,),
                        tactile_backend=str(getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"),
                        speed=float(side_policy["speed"]),
                        torque=float(side_policy["torque"]),
                        max_action=float(side_policy["max_action"]),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            if bool(getattr(args, "dmtac_config_required", False)):
                raise RuntimeError(
                    f"Failed to load DMTAC hardware assignment {getattr(args, 'dmtac_config', 'latest')!r}: {exc}"
                ) from exc
            _log("DMTAC", f"WARNING: could not load DMTAC hardware assignment: {exc}")

    for side in dmtac_sides:
        manual = manual_dmtac_backend_config_for_side(
            args,
            side,
            policies_by_side.get(side, policy),
        )
        if manual is not None:
            configs[side] = manual

    resolved = {
        side: _apply_gripper_policy_to_dmtac_config(
            config,
            args=args,
            policy=policies_by_side.get(side, policy),
        )
        for side, config in configs.items()
        if side in dmtac_sides and (config.connect_hardware or config.connect_tactile)
    }
    if bool(getattr(args, "dmtac_config_required", False)):
        require_dmtac_sides_resolved(
            dmtac_sides,
            resolved,
            config=str(getattr(args, "dmtac_config", "latest") or "latest"),
        )
    setattr(args, "_dmtac_backend_configs", dict(resolved))
    setattr(args, "dmtac_config_loaded_sides", sorted(resolved))
    return dict(resolved)


def _discover_runtime_q6_ports(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> dict[str, str]:
    """Resolve newly connected Q6 devices against the saved side assignment."""

    from hardware.teleop_hand import q6_assignment

    sides = tuple(str(side) for side in active_sides)
    scan = q6_assignment.scan_q6_devices(
        probe_all_serial=True,
        show_raw_lines=False,
    )
    devices = list(scan.get("q6_devices") or ())
    if len(devices) < len(sides):
        raise RuntimeError(
            f"Q6 count is insufficient after reconnect: detected {len(devices)}, "
            f"expected {len(sides)}."
        )
    assignment = q6_assignment.load_assignment(
        str(getattr(args, "q6_config", "latest") or "latest")
    )
    comparison = q6_assignment.compare_assignment_to_scan(assignment, scan)
    missing = [
        item
        for item in comparison.get("missing") or ()
        if str(item.get("side") or "") in sides
    ]
    effective = comparison.get("effective_assignment")
    if missing or not isinstance(effective, Mapping):
        missing_sides = sorted(
            str(item.get("side") or "unknown") for item in missing
        )
        raise RuntimeError(
            "Connected Q6 devices do not match the saved left/right assignment; "
            f"missing={missing_sides or list(sides)}. Reconfigure Q6 before retrying."
        )
    ports = q6_assignment.q6_ports_for_sides(
        effective,
        active_sides=sides,
    )
    unresolved = [side for side in sides if not str(ports.get(side) or "")]
    if unresolved:
        raise RuntimeError(
            f"Saved Q6 assignment did not resolve ports for: {unresolved}."
        )
    if comparison.get("remapped"):
        q6_assignment.save_assignment(effective)
    return {side: str(ports[side]) for side in sides}

__all__ = [
    "bind_legacy_support",
    "_apply_default_q6_ports",
    "_run_gui_device_preflight",
    "_resolve_gripper_input_source",
    "_argv_has_option",
    "_apply_real_loop_timing_defaults",
    "_source_arm_ip_fields_for_active_sides",
    "_apply_default_arm_ips",
    "_save_explicit_arm_ips_latest",
    "_validate_required_arm_ips",
    "_dmtac_backend_configs",
    "_discover_runtime_q6_ports",
]

