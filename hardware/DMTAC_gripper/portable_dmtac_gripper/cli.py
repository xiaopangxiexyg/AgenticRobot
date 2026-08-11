from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import sys
import time

from portable_dm_tac import discover_devices

from .device import (
    DMTacGripperDevice,
    GripperConfig,
    TACTILE_COMPUTE_PROFILES,
    TactileConfig,
    format_snapshot_summary,
    normalize_tactile_compute_profile,
    normalize_tactile_fields,
)
from .gripper_driver import InspireGripper, list_serial_devices
from .tactile_filter import TactileDenoiseConfig
from .usb_assignment import (
    SIDES,
    TACTILE_SLOTS,
    build_gripper_auto_assignment_tokens,
    build_assignment_from_tokens,
    compare_assignment_to_scan,
    configured_gripper_sides,
    configured_tactile_sides,
    format_comparison,
    gripper_baudrate_from_probe,
    gripper_id_from_probe,
    gripper_for_side,
    load_assignment,
    merge_assignment,
    require_compatible_or_confirm,
    resolve_gripper_token,
    resolve_tactile_token,
    run_gripper_motion_test,
    save_assignment,
    scan_usb_devices,
    tactile_ids_for_side,
)


def _parse_dev_id(text: str):
    try:
        return int(text)
    except ValueError:
        return text


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _tactile_gui_artifact_layout_default() -> str:
    """Use the repository switch when available, preserving standalone use."""

    project_root = Path(__file__).resolve().parents[3]
    if (project_root / "runtime_artifact_paths.py").is_file():
        project_root_text = str(project_root)
        if project_root_text not in sys.path:
            sys.path.insert(0, project_root_text)
        from runtime_artifact_paths import configured_runtime_artifact_layout

        return configured_runtime_artifact_layout()
    return "legacy"


def _add_hidden_alias(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    dest: str,
    action: str | None = None,
    value_type=None,
    nargs=None,
) -> None:
    kwargs = {"dest": dest, "help": argparse.SUPPRESS}
    if action is not None:
        kwargs["action"] = action
    if value_type is not None:
        kwargs["type"] = value_type
    if nargs is not None:
        kwargs["nargs"] = nargs
    parser.add_argument(option, **kwargs)


def _add_common(parser: argparse.ArgumentParser) -> None:
    transport = parser.add_argument_group("gripper transport")
    transport.add_argument("--port", default="/dev/ttyUSB0", help="Gripper serial port.")
    transport.add_argument("--gripper-id", type=int, default=1, help="Gripper Modbus id.")
    _add_hidden_alias(transport, "--id", dest="gripper_id", value_type=int)
    transport.add_argument("--gripper-baudrate", dest="baudrate", type=int, default=115200)
    _add_hidden_alias(transport, "--baudrate", dest="baudrate", value_type=int)
    transport.add_argument(
        "--gripper-timeout",
        dest="timeout",
        type=float,
        default=0.10,
        help="Gripper serial timeout in seconds.",
    )
    _add_hidden_alias(transport, "--timeout", dest="timeout", value_type=float)
    transport.add_argument("--max-open-mm", dest="max_open_mm", type=float, default=70.0)
    _add_hidden_alias(transport, "--max_open_mm", dest="max_open_mm", value_type=float)
    parser.add_argument(
        "--no-gripper-reconnect",
        action="store_true",
        help="Disable one-shot gripper serial reconnect/retry after transient Modbus I/O errors.",
    )
    parser.add_argument("--gripper-reconnect-attempts", type=int, default=1)
    parser.add_argument("--gripper-reconnect-delay", type=float, default=0.05)
    parser.add_argument("--gripper-inter-command-delay", type=float, default=0.005)
    parser.add_argument(
        "--gripper-use-write-multiple-registers",
        action="store_true",
        help="Use Modbus 0x10 to write position/speed/torque together. Disabled by default for stability.",
    )
    parser.add_argument("--debug", action="store_true")
    tactile = parser.add_argument_group("DM-Tac tactile")
    tactile.add_argument("--tactile-ids", nargs="*", default=None)
    _add_hidden_alias(tactile, "--tactile_ids", dest="tactile_ids", nargs="*")
    tactile.add_argument(
        "--tactile-backend",
        dest="backend",
        default="cpu",
        choices=["cpu", "cuda", "Flux", "flux"],
    )
    _add_hidden_alias(tactile, "--backend", dest="backend")
    tactile.add_argument(
        "--tactile-mode",
        dest="mode",
        default="standard",
        choices=["standard", "high"],
    )
    _add_hidden_alias(tactile, "--mode", dest="mode")
    tactile.add_argument("--tactile-max-fps", dest="max_fps", type=int, default=120)
    _add_hidden_alias(tactile, "--max_fps", dest="max_fps", value_type=int)
    tactile.add_argument("--tactile-fields", nargs="*", default=["raw"])
    _add_hidden_alias(tactile, "--tactile_fields", dest="tactile_fields", nargs="*")
    parser.add_argument(
        "--tactile-compute-profile",
        choices=TACTILE_COMPUTE_PROFILES,
        default="dist-only",
        help=(
            "SDK compute channels to enable for tactile reads. Default fields are raw images; "
            "use --tactile-fields distributed_force for real-time force output."
        ),
    )
    parser.add_argument(
        "--tactile-sequential-read",
        action="store_true",
        help="Read multiple tactile sensors sequentially instead of parallel for A/B latency checks.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--config",
        default="",
        help=(
            "DMTac USB assignment config path, or 'latest'. Overrides selected gripper/tactile ids. "
            "By default this trusts the saved by-path devices for low-latency control."
        ),
    )
    parser.add_argument(
        "--validate-current-usb",
        action="store_true",
        help="Probe current USB devices while loading --config. Useful after rewiring, slower for motion commands.",
    )
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--tactile-slots", nargs="*", choices=TACTILE_SLOTS, default=None)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm extra USB devices detected while loading --config.",
    )
    parser.add_argument(
        "--allow-missing-config-devices",
        action="store_true",
        help="Allow running even when devices saved in --config are missing.",
    )


def _gripper_config(args) -> GripperConfig:
    return GripperConfig(
        port=args.port,
        gripper_id=args.gripper_id,
        baudrate=args.baudrate,
        timeout=args.timeout,
        max_open_mm=args.max_open_mm,
        reconnect_on_error=not bool(args.no_gripper_reconnect),
        reconnect_attempts=args.gripper_reconnect_attempts,
        reconnect_delay=args.gripper_reconnect_delay,
        inter_command_delay=args.gripper_inter_command_delay,
        use_write_multiple_registers=bool(args.gripper_use_write_multiple_registers),
        debug=args.debug,
    )


def _tactile_config(args) -> TactileConfig:
    dev_ids = []
    if args.tactile_ids:
        for item in args.tactile_ids:
            for part in item.replace(",", " ").split():
                if part.strip():
                    dev_ids.append(_parse_dev_id(part))
        if not dev_ids:
            dev_ids = None
    else:
        dev_ids = None

    return TactileConfig(
        dev_ids=dev_ids,
        backend=args.backend,
        mode=args.mode,
        max_fps=args.max_fps,
        fields=normalize_tactile_fields(args.tactile_fields),
        compute_profile=normalize_tactile_compute_profile(getattr(args, "tactile_compute_profile", "dist-only")),
        parallel_read=not bool(getattr(args, "tactile_sequential_read", False)),
    )


def _print_gripper_state(state, as_json: bool) -> None:
    if as_json:
        _print_json(state.to_dict())
        return
    print(
        f"joint_pos={state.joint_pos:.4f} opening_mm={state.opening_mm:.2f} "
        f"moving={state.is_moving} current={state.current} temp={state.temperature_c} "
        f"error=0x{state.error_code:04X} status={state.status_code} {state.status_text}"
    )


def _fast_exit(code: int = 0) -> None:
    """Bypass SDK C++ teardown paths that may abort after tactile disconnect."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _side_from_config(args, assignment, *, kind: str) -> str:
    if args.side is not None:
        return str(args.side)
    if kind == "gripper":
        sides = configured_gripper_sides(assignment)
    elif kind == "tactile":
        sides = configured_tactile_sides(assignment)
    else:
        raise ValueError(f"unknown config selection kind: {kind}")
    if len(sides) == 1:
        return sides[0]
    if not sides:
        raise RuntimeError(f"No configured DMTac {kind} devices are available.")
    raise RuntimeError(f"--config selects multiple {kind} sides {sides}; pass --side left/right.")


def _resolve_config_args(args, *, need_gripper: bool, need_tactile: bool) -> None:
    if not getattr(args, "config", ""):
        return
    assignment = load_assignment(args.config)
    if not bool(getattr(args, "validate_current_usb", False)):
        effective = assignment
    else:
        configured_baudrates = []
        for side in SIDES:
            gripper = assignment["assignments"][side]["gripper"]
            if gripper is not None:
                configured_baudrates.append(int(gripper.get("baudrate", args.baudrate)))
        scan = scan_usb_devices(
            probe_grippers=True,
            gripper_id=args.gripper_id,
            baudrate=args.baudrate,
            timeout=args.timeout,
            scan_gripper_ids=bool(getattr(args, "scan_gripper_ids", False)),
            probe_system_ttys=bool(getattr(args, "probe_system_ttys", False)),
            scan_baudrates=tuple(
                dict.fromkeys(
                    [
                        int(args.baudrate),
                        *configured_baudrates,
                    ]
                )
            ),
        )
        comparison = compare_assignment_to_scan(assignment, scan)
        require_compatible_or_confirm(
            comparison,
            allow_missing=bool(getattr(args, "allow_missing_config_devices", False)),
            assume_yes=bool(getattr(args, "yes", False)),
        )
        effective = comparison["effective_assignment"]

    if need_gripper:
        side = _side_from_config(args, effective, kind="gripper")
        gripper = gripper_for_side(effective, side)
        if gripper is None:
            raise RuntimeError(f"No gripper configured for side {side}.")
        args.side = side
        args.port = str(gripper["port"])
        args.gripper_id = int(gripper.get("gripper_id", args.gripper_id))
        args.baudrate = int(gripper.get("baudrate", args.baudrate))
        args.timeout = float(gripper.get("timeout", args.timeout))

    if need_tactile:
        side = _side_from_config(args, effective, kind="tactile")
        slots = args.tactile_slots or list(TACTILE_SLOTS)
        ids = tactile_ids_for_side(effective, side, slots=slots)
        if not ids:
            raise RuntimeError(f"No tactile sensors configured for side {side} slots={slots}.")
        args.side = side
        args.tactile_ids = list(ids)


def _command_needs_gripper(args) -> bool:
    if args.command in {
        "verify-gripper",
        "set-speed",
        "set-torque",
        "open",
        "close",
        "move",
        "stop",
    }:
        return True
    if args.command == "read":
        return not bool(getattr(args, "no_gripper", False))
    return False


def _command_needs_tactile(args) -> bool:
    if args.command in {"verify-tactile", "visualize-force", "probe-force", "reset-tactile"}:
        return True
    if args.command == "read":
        return not bool(getattr(args, "no_tactile", False))
    return False


def _is_reachable_gripper_item(item) -> bool:
    return isinstance(item.get("probe"), dict) and item.get("probe", {}).get("reachable") is True


def _is_probe_failed_gripper_item(item) -> bool:
    return isinstance(item.get("probe"), dict) and item.get("probe", {}).get("reachable") is False


def _is_noise_serial_item(item) -> bool:
    port = str(item.get("preferred_port") or item.get("port") or item.get("device") or "")
    text = " ".join(
        str(item.get(key) or "")
        for key in ("description", "hwid", "manufacturer", "product")
    ).strip().lower()
    return (
        port.startswith("/dev/ttyS")
        and not bool(item.get("probe_attempted"))
        and text in {"", "n/a n/a", "n/a"}
    )


def _is_gripper_attention_item(item) -> bool:
    if _is_reachable_gripper_item(item):
        return False
    if _is_noise_serial_item(item):
        return False
    port = str(item.get("preferred_port") or item.get("port") or item.get("device") or "")
    return bool(
        item.get("probe_attempted")
        or item.get("probable_gripper_serial")
        or port.startswith("/dev/serial/")
        or port.startswith("/dev/ttyUSB")
        or port.startswith("/dev/ttyACM")
    )


def _format_probe_error(item) -> str:
    probe = item.get("probe") or {}
    parts = []
    skipped_reason = str(item.get("probe_skipped_reason") or "").strip()
    if skipped_reason:
        parts.append(f"skipped={skipped_reason}")
    attempted_count = probe.get("attempted_setting_count") or probe.get("attempted_gripper_id_count")
    if isinstance(attempted_count, int) and attempted_count > 1:
        parts.append(f"probes_tried={attempted_count}")
    attempted_baudrates = sorted(set(int(value) for value in probe.get("attempted_baudrates") or ()))
    if len(attempted_baudrates) > 1:
        parts.append("baudrates=" + ",".join(str(value) for value in attempted_baudrates))
    suggested = probe.get("suggested_gripper_id")
    if suggested not in (None, ""):
        parts.append(f"suggested_id={suggested}")
    error = str(probe.get("error") or "").strip()
    if error:
        parts.append(f"error={error}")
    if not parts:
        return ""
    return " ".join(parts)


def _gripper_item_gripper_id(item: dict, default_id: int = 1) -> int:
    probe_id = gripper_id_from_probe(item.get("probe"))
    if probe_id is not None:
        return int(probe_id)
    return int(default_id)


def _print_scan_summary(scan: dict) -> None:
    all_grippers = list(scan.get("grippers", []))
    reachable = [item for item in all_grippers if _is_reachable_gripper_item(item)]
    attention = [item for item in all_grippers if _is_gripper_attention_item(item)]
    hidden_count = sum(1 for item in all_grippers if _is_noise_serial_item(item))

    if bool(scan.get("gripper_probe_stopped_early")):
        print(
            "Gripper probe stopped early after finding "
            f"{int(scan.get('gripper_probe_reachable_count') or 0)} reachable devices; "
            "use --max-reachable-grippers 0 for an exhaustive scan."
        )

    print("Reachable grippers:")
    if reachable:
        for item in reachable:
            probe = item.get("probe") or {}
            state = probe.get("state") or {}
            details = [f"id={_gripper_item_gripper_id(item)}"]
            if probe.get("id_autodetected") is True:
                requested = probe.get("requested_gripper_id")
                details.append(f"auto_id_from={requested}")
            detected_baudrate = probe.get("detected_baudrate")
            if detected_baudrate not in (None, "", probe.get("requested_baudrate")):
                details.append(f"baudrate={detected_baudrate}")
            attempted_count = probe.get("attempted_setting_count") or probe.get("attempted_gripper_id_count")
            if isinstance(attempted_count, int) and attempted_count > 1:
                details.append(f"probes_tried={attempted_count}")
            if state:
                details.append(f"joint={state.get('joint_pos')}")
                details.append(f"opening_mm={state.get('opening_mm')}")
            detail_text = "" if not details else " " + " ".join(details)
            print(
                f"  [{item.get('index')}] {item.get('preferred_port') or item.get('port')}"
                f"{detail_text}"
            )
    else:
        print("  none")

    print("Gripper candidates needing attention:")
    if attention:
        for item in attention:
            probe = item.get("probe") or {}
            probe_reachable = probe.get("reachable")
            status = "reachable" if probe_reachable is True else "not_probed"
            if probe_reachable is False:
                status = "probe_failed"
            details = [
                str(value).strip()
                for value in (
                    item.get("description"),
                    item.get("hwid"),
                    _format_probe_error(item).strip(),
                )
                if str(value or "").strip()
            ]
            detail_text = "" if not details else " " + " ".join(details)
            print(
                f"  [{item.get('index')}] {item.get('preferred_port') or item.get('port')} {status}"
                f"{detail_text}"
            )
    else:
        print("  none")
    if hidden_count:
        print(f"  ({hidden_count} system ttyS ports hidden; pass --probe-all-serial only if needed.)")

    print("DM-Tac tactile sensors:")
    tactile_items = list(scan.get("tactile", []))
    if tactile_items:
        for item in tactile_items:
            extra = []
            if item.get("device_index") not in (None, ""):
                extra.append(f"device_index={item.get('device_index')}")
            if item.get("serial") not in (None, "", item.get("dev_id")):
                extra.append(f"serial={item.get('serial')}")
            suffix = "" if not extra else " " + " ".join(extra)
            print(f"  [{item.get('index')}] {item.get('dev_id')}{suffix}")
    else:
        print("  none")
    if not tactile_items and scan.get("tactile_candidates"):
        print("DM-Tac tactile candidates not usable by SDK:")
        for item in scan.get("tactile_candidates", []):
            print(
                f"  {item.get('device_node')} serial={item.get('serial') or '-'} "
                f"reason={item.get('reason')} readable={item.get('readable')} "
                f"model={item.get('model') or item.get('product') or ''}"
            )


def _prompt_token(prompt: str) -> str | None:
    text = input(prompt).strip()
    if not text or text.lower() in {"none", "skip", "-"}:
        return None
    return text


def _stream_is_tty(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _prompt_gripper_assignment_for_two(default_swap: bool = False) -> bool:
    default_text = "swap" if default_swap else "keep"
    try:
        answer = input(
            "Assign the first reachable gripper listed -> left and the second -> right. "
            f"Press Enter to {default_text}, type 's' to swap: "
        ).strip().lower()
    except EOFError:
        return bool(default_swap)
    if not answer:
        return bool(default_swap)
    return answer in {"s", "swap", "y", "yes"}


def _prompt_single_gripper_side(default_side: str = "right") -> str:
    try:
        answer = input(
            f"Only one reachable gripper detected. Assign it to left or right? [{default_side}]: "
        ).strip().lower()
    except EOFError:
        return str(default_side)
    if not answer:
        answer = str(default_side)
    if answer not in SIDES:
        raise ValueError("single gripper side must be left or right.")
    return answer


def _noninteractive_gripper_assignment_error() -> RuntimeError:
    return RuntimeError(
        "Left/right gripper confirmation requires interactive stdin. "
        "Use `conda run --no-capture-output ...` for configure-usb, "
        "or pass --yes with optional --swap/--left-gripper/--right-gripper."
    )


def _reachable_gripper_items(scan: dict) -> list[dict]:
    return [
        dict(item)
        for item in scan.get("grippers") or []
        if isinstance(item.get("probe"), dict) and item.get("probe", {}).get("reachable") is True
    ]


def _gripper_item_token(item: dict) -> str:
    return str(item.get("index"))


def _gripper_item_port(item: dict) -> str:
    return str(item.get("port") or item.get("device") or item.get("preferred_port") or "")


def _gripper_item_label(item: dict) -> str:
    return f"[{item.get('index')}] {item.get('preferred_port') or item.get('port')}"


def _prompt_side_after_jog(input_func, *, prompt: str) -> str:
    while True:
        answer = input_func(prompt).strip().lower()
        if answer in {"l", "left"}:
            return "left"
        if answer in {"r", "right"}:
            return "right"
        if answer in {"s", "skip", "none", "-"}:
            return "skip"
        if answer in {"retry", "again"}:
            return "retry"
        if answer in {"q", "quit", "done", "finish"}:
            return "quit"
        print("Please type left/right/skip/retry/q.")


def _set_gripper_side_token(
    tokens: dict[str, str | None],
    *,
    side: str,
    token: str,
    input_func,
) -> bool:
    existing = tokens.get(side)
    if existing is not None and existing != token:
        answer = input_func(
            f"{side} already assigned to gripper index {existing}. Overwrite with {token}? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            return False
    for side_name in SIDES:
        if tokens.get(side_name) == token:
            tokens[side_name] = None
    tokens[side] = token
    return True


def _motion_confirmed_gripper_tokens(
    scan: dict,
    args,
    *,
    motion_func=run_gripper_motion_test,
    input_func=input,
) -> dict[str, str | None]:
    devices = _reachable_gripper_items(scan)
    if not devices:
        return {"left": None, "right": None}
    if not _stream_is_tty(sys.stdin) and input_func is input:
        raise _noninteractive_gripper_assignment_error()

    print("\nGripper motion assignment:")
    print("For each reachable gripper, press Enter to micro-jog it, then type left/right/skip.")
    print("Keep fingers and objects clear before each jog.")
    tokens: dict[str, str | None] = {"left": None, "right": None}
    for item in devices:
        token = _gripper_item_token(item)
        label = _gripper_item_label(item)
        while True:
            answer = input_func(
                f"\n{label}: press Enter to micro-jog, type left/right to assign without jog, "
                "s=skip, q=finish: "
            ).strip().lower()
            if answer in {"q", "quit", "done", "finish"}:
                return tokens
            if answer in {"s", "skip", "none", "-"}:
                break
            if answer in {"l", "left", "r", "right"}:
                side = "left" if answer in {"l", "left"} else "right"
            else:
                print(
                    f"Micro-jogging {label} "
                    f"amplitude={float(args.gripper_jog_amplitude):.3f}..."
                )
                result = motion_func(
                    _gripper_item_port(item),
                    gripper_id=_gripper_item_gripper_id(item, int(args.gripper_id)),
                    baudrate=int(args.baudrate),
                    timeout=float(args.timeout),
                    amplitude=float(args.gripper_jog_amplitude),
                    speed=float(args.gripper_jog_speed),
                    torque=float(args.gripper_jog_torque),
                )
                if bool(getattr(args, "json", False)):
                    _print_json(result)
                side = _prompt_side_after_jog(
                    input_func,
                    prompt="Which gripper moved? [left/right/skip/retry/q]: ",
                )
                if side == "retry":
                    continue
                if side == "quit":
                    return tokens
                if side == "skip":
                    break
            if _set_gripper_side_token(tokens, side=side, token=token, input_func=input_func):
                print(f"Assigned {label} -> {side}")
                break
        if tokens.get("left") is not None and tokens.get("right") is not None:
            answer = input_func("Both left/right grippers are assigned. Press Enter to finish, c to continue: ")
            if answer.strip().lower() not in {"c", "continue"}:
                break
    return tokens


def _auto_gripper_tokens(scan: dict, args) -> dict[str, str | None]:
    devices = [
        item
        for item in scan.get("grippers") or []
        if isinstance(item.get("probe"), dict) and item.get("probe", {}).get("reachable") is True
    ]
    if len(devices) > 2:
        raise RuntimeError(
            f"Expected at most 2 reachable grippers, detected {len(devices)}. "
            "Disconnect extras or pass --left-gripper/--right-gripper explicitly."
        )
    if not devices:
        return {"left": None, "right": None}
    if (
        bool(getattr(args, "gripper_motion_confirm", True))
        and not bool(getattr(args, "yes", False))
    ):
        return _motion_confirmed_gripper_tokens(scan, args)
    swap = bool(getattr(args, "swap", False))
    single_side = str(getattr(args, "single_side", "right") or "right")
    if len(devices) == 2 and not bool(getattr(args, "yes", False)):
        if not _stream_is_tty(sys.stdin):
            raise _noninteractive_gripper_assignment_error()
        swap = _prompt_gripper_assignment_for_two(default_swap=swap)
    elif len(devices) == 1 and not bool(getattr(args, "yes", False)):
        if not _stream_is_tty(sys.stdin):
            raise _noninteractive_gripper_assignment_error()
        single_side = _prompt_single_gripper_side(single_side)
    return build_gripper_auto_assignment_tokens(
        scan,
        swap=swap,
        single_side=single_side,
        require_reachable=True,
    )


def _interactive_assignment(scan: dict, *, show_summary: bool = True):
    if bool(show_summary):
        _print_scan_summary(scan)
    print("Enter scan index, port path, dev_id, or blank/none to skip.")
    return build_assignment_from_tokens(
        scan,
        left_gripper=_prompt_token("left gripper: "),
        right_gripper=_prompt_token("right gripper: "),
        left_inner=_prompt_token("left inner tactile: "),
        left_outer=_prompt_token("left outer tactile: "),
        right_inner=_prompt_token("right inner tactile: "),
        right_outer=_prompt_token("right outer tactile: "),
    )


def _apply_configure_gripper_settings(assignment: dict, args) -> dict:
    for side in ("left", "right"):
        gripper = assignment["assignments"][side]["gripper"]
        if gripper is None:
            continue
        detected_id = gripper_id_from_probe(gripper.get("probe"))
        detected_baudrate = gripper_baudrate_from_probe(gripper.get("probe"))
        gripper["gripper_id"] = int(detected_id if detected_id is not None else args.gripper_id)
        gripper["baudrate"] = int(detected_baudrate if detected_baudrate is not None else args.baudrate)
        gripper["timeout"] = float(args.timeout)
    return assignment


def _prepare_configure_usb_mode(args) -> None:
    tactile_tokens = (
        "left_inner",
        "left_outer",
        "right_inner",
        "right_outer",
    )
    if bool(args.scan_only):
        side_effects = []
        for name in (
            "save",
            "interactive",
            "assign_tactile_gui",
            "left_gripper",
            "right_gripper",
            *tactile_tokens,
        ):
            if bool(getattr(args, name, False)):
                side_effects.append(name)
        if args.test_gripper or args.visualize_tactile or str(args.merge_config or ""):
            side_effects.append("motion/visualization/merge option")
        if side_effects:
            raise ValueError(
                "--scan-only cannot be combined with assignment, motion, visualization, merge, "
                f"or save options: {sorted(set(side_effects))}"
            )
        return

    if bool(args.grippers_only):
        invalid = [name for name in tactile_tokens if getattr(args, name)]
        if args.assign_tactile_gui or args.visualize_tactile or args.interactive:
            invalid.append("tactile/manual-all-slots option")
        if invalid:
            raise ValueError(f"--grippers-only received tactile assignment options: {invalid}")
        if not str(args.merge_config or "").strip():
            args.merge_config = "latest"

    if bool(args.tactile_only):
        invalid = []
        if args.left_gripper or args.right_gripper or args.test_gripper or args.interactive:
            invalid.append("gripper/manual-all-slots option")
        if invalid:
            raise ValueError(f"--tactile-only received gripper assignment options: {invalid}")
        args.auto_grippers = False
        if not any(getattr(args, name) for name in tactile_tokens):
            args.assign_tactile_gui = True
        if not str(args.merge_config or "").strip():
            args.merge_config = "latest"


def _run_configure_usb(args) -> None:
    _prepare_configure_usb_mode(args)
    if (
        not bool(getattr(args, "json", False))
        and not bool(args.no_probe_grippers)
        and bool(args.scan_gripper_ids)
    ):
        baudrates = tuple(args.scan_baudrates or (args.baudrate,))
        if bool(args.no_scan_baudrates):
            baudrates = (int(args.baudrate),)
        print(
            "Scanning likely gripper USB ports; if --gripper-id is not correct, "
            "Modbus ids 1..247 are tried "
            f"at baudrates {','.join(str(value) for value in baudrates)}."
        )
    scan = scan_usb_devices(
        probe_grippers=not bool(args.no_probe_grippers),
        probe_all_serial=bool(args.probe_all_serial),
        probe_system_ttys=bool(args.probe_system_ttys),
        gripper_id=args.gripper_id,
        baudrate=args.baudrate,
        timeout=args.timeout,
        scan_gripper_ids=bool(args.scan_gripper_ids),
        scan_baudrates=() if bool(args.no_scan_baudrates) else tuple(args.scan_baudrates or (args.baudrate,)),
        max_reachable_grippers=int(args.max_reachable_grippers),
    )
    if args.json and args.scan_only:
        _print_json(scan)
        return
    _print_scan_summary(scan)

    for token in args.test_gripper or ():
        item = resolve_gripper_token(scan, token)
        print(f"Testing gripper {token} on {item.get('port')} with amplitude={args.motion_amplitude}")
        result = run_gripper_motion_test(
            str(item["port"]),
            gripper_id=_gripper_item_gripper_id(item, int(args.gripper_id)),
            baudrate=args.baudrate,
            timeout=args.timeout,
            amplitude=float(args.motion_amplitude),
            speed=float(args.motion_speed),
            torque=float(args.motion_torque),
        )
        _print_json(result) if args.json else print("motion test ok", result["targets"])

    visualized_tactile = bool(args.visualize_tactile)
    for token in args.visualize_tactile or ():
        item = resolve_tactile_token(scan, token)
        print(f"Visualizing tactile {token} dev_id={item.get('dev_id')}")
        from .force_visualizer import ForceVizConfig, run_live_distributed_force_xy

        run_live_distributed_force_xy(
            tactile_config=TactileConfig(
                dev_ids=[item["dev_id"]],
                backend=args.backend,
                mode=args.mode,
                max_fps=args.max_fps,
                fields=("distributed_force",),
                compute_profile=normalize_tactile_compute_profile(args.tactile_gui_compute_profile),
            ),
            timeout_ms=args.timeout_ms,
            count=args.visualize_count,
            max_hz=args.max_hz,
            window=f"DM-Tac {item.get('dev_id')}",
            viz_config=ForceVizConfig(),
            reset_tactile=bool(args.reset_tactile),
            warmup_frames=int(args.warmup_frames),
            close_devices=False,
        )

    if bool(args.assign_tactile_gui) and not bool(args.scan_only):
        from .force_visualizer import ForceVizConfig
        from .tactile_assignment_gui import (
            TactileAssignmentGuiConfig,
            run_tactile_assignment_gui,
            tactile_slot_tokens_from_assignment,
        )

        visualized_tactile = True
        result = run_tactile_assignment_gui(
            scan,
            device_tokens=tuple(args.tactile_gui_devices or ()),
            config=TactileAssignmentGuiConfig(
                backend=args.backend,
                mode=args.mode,
                max_fps=args.max_fps,
                compute_profile=normalize_tactile_compute_profile(args.tactile_gui_compute_profile),
                timeout_ms=args.timeout_ms,
                wait_for_new=bool(args.tactile_gui_wait_for_new),
                parallel_read=not bool(args.tactile_gui_sequential_read),
                max_hz=args.tactile_gui_max_hz,
                max_devices=args.max_tactile_devices,
                force_decimation=int(args.tactile_gui_decimation),
                render_mode=str(args.tactile_gui_render_mode),
                denoise_distributed_force=bool(args.tactile_gui_denoise),
                normal_grid_rows=int(args.tactile_gui_grid_rows),
                normal_grid_cols=int(args.tactile_gui_grid_cols),
                normal_grid_vector_stride=int(args.tactile_gui_vector_stride),
                normal_sign=float(args.tactile_gui_normal_sign),
                normal_pressure_deadband=float(args.tactile_gui_pressure_deadband),
                normal_pressure_ref=float(args.tactile_gui_pressure_ref),
                normal_vector_deadband=float(args.tactile_gui_vector_deadband),
                normal_vector_ref=float(args.tactile_gui_vector_ref),
                record_readings=not bool(args.tactile_gui_no_record),
                record_dir=args.tactile_gui_record_dir or None,
                artifact_layout=str(args.tactile_gui_artifact_layout),
                record_every_n=max(1, int(args.tactile_gui_record_every_n)),
                record_grid=bool(args.tactile_gui_record_grid)
                and not bool(args.tactile_gui_record_summary_only),
                window=args.tactile_gui_window,
                reset_tactile=bool(args.reset_tactile),
                warmup_frames=int(args.warmup_frames),
                close_devices=False,
                viz_config=ForceVizConfig(
                    stride=int(args.tactile_gui_stride),
                    scale=int(args.tactile_gui_scale),
                    fixed_ref=args.tactile_gui_fixed_ref,
                    mode="delta",
                    baseline_frames=max(1, int(args.tactile_gui_baseline_frames)),
                ),
            ),
        )
        if result.cancelled:
            raise RuntimeError("Tactile GUI assignment was cancelled.")
        tactile_tokens = tactile_slot_tokens_from_assignment(result.slots)
        for name, value in tactile_tokens.items():
            if value:
                setattr(args, name, value)
        print("Tactile GUI assignment:")
        _print_json(tactile_tokens)
        if result.record_dir:
            print(f"Tactile GUI debug record: {result.record_dir}")

    if args.scan_only:
        if visualized_tactile:
            _fast_exit(0)
        return

    has_gripper_assignment_args = any(
        getattr(args, name)
        for name in (
            "left_gripper",
            "right_gripper",
        )
    )
    has_tactile_assignment_args = any(
        getattr(args, name)
        for name in (
            "left_inner",
            "left_outer",
            "right_inner",
            "right_outer",
        )
    )
    if (
        bool(args.auto_grippers)
        and not bool(args.interactive)
        and not has_gripper_assignment_args
    ):
        tokens = _auto_gripper_tokens(scan, args)
        if tokens.get("left"):
            args.left_gripper = tokens["left"]
        if tokens.get("right"):
            args.right_gripper = tokens["right"]
        has_gripper_assignment_args = bool(args.left_gripper or args.right_gripper)

    has_assignment_args = bool(has_gripper_assignment_args or has_tactile_assignment_args)
    if not has_assignment_args and not bool(args.interactive):
        if bool(args.auto_grippers) and not _reachable_gripper_items(scan):
            raise RuntimeError(
                "No reachable grippers found, so nothing was assigned. "
                "Check power/wiring/permissions/baudrate. "
                "The gripper-like USB candidates above show probe errors and the number of Modbus ids tried. "
                "Use --gripper-id <N> to force a known id, or --manual-all-slots to assign manually."
            )
    if args.interactive and bool(args.assign_tactile_gui):
        print("Enter gripper scan index/port, or blank/none to skip. Tactile slots come from GUI.")
        assignment = build_assignment_from_tokens(
            scan,
            left_gripper=_prompt_token("left gripper: "),
            right_gripper=_prompt_token("right gripper: "),
            left_inner=args.left_inner,
            left_outer=args.left_outer,
            right_inner=args.right_inner,
            right_outer=args.right_outer,
        )
    elif args.interactive or not has_assignment_args:
        assignment = _interactive_assignment(scan, show_summary=False)
    else:
        assignment = build_assignment_from_tokens(
            scan,
            left_gripper=args.left_gripper,
            right_gripper=args.right_gripper,
            left_inner=args.left_inner,
            left_outer=args.left_outer,
            right_inner=args.right_inner,
            right_outer=args.right_outer,
        )
    assignment = _apply_configure_gripper_settings(assignment, args)
    merge_config = str(getattr(args, "merge_config", "") or "").strip()
    if merge_config:
        assignment = merge_assignment(load_assignment(merge_config), assignment)
    comparison = compare_assignment_to_scan(assignment, scan)
    text = format_comparison(comparison)
    if text:
        print(text)
    print("Selected DMTac assignment:")
    for side in ("left", "right"):
        side_assignment = assignment["assignments"][side]
        gripper = side_assignment.get("gripper")
        tactile = side_assignment.get("tactile") or {}
        print(
            f"  {side}: "
            f"gripper={None if gripper is None else gripper.get('port')} "
            f"inner={None if tactile.get('inner') is None else tactile.get('inner', {}).get('dev_id')} "
            f"outer={None if tactile.get('outer') is None else tactile.get('outer', {}).get('dev_id')}"
        )
    if args.save:
        path = save_assignment(assignment, config_dir=args.output_dir or None)
        print(f"saved DMTac USB assignment: {path}")
        print("updated latest config")
    else:
        print("assignment preview; pass --save to write it")
        _print_json(assignment)
    if visualized_tactile:
        _fast_exit(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Inspire gripper + DM-Tac tactile CLI")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("list-ports", help="List serial devices without sending motion commands.")
    sub.add_parser("list-tactile", help="List DM-Tac devices visible to the SDK.")

    p = sub.add_parser(
        "configure-usb",
        help="Scan and build the shared left/right gripper/tactile assignment.",
        description=(
            "Scan DMTAC hardware and build an assignment. Use exactly one mode: "
            "--scan-only, --grippers-only, --tactile-only, or the default full assignment."
        ),
    )
    configure_mode = p.add_mutually_exclusive_group()
    configure_mode.add_argument(
        "--scan-only",
        action="store_true",
        help="Only print the scan result; assignment, motion-test, GUI, and save options are rejected.",
    )
    configure_mode.add_argument(
        "--grippers-only",
        action="store_true",
        help="Assign only left/right grippers and preserve tactile slots from --merge-config (latest by default).",
    )
    configure_mode.add_argument(
        "--tactile-only",
        action="store_true",
        help="Assign only tactile slots and preserve grippers from --merge-config (latest by default).",
    )
    p.add_argument(
        "--manual-all-slots",
        dest="interactive",
        action="store_true",
        help="Prompt for both grippers and all four tactile slots using terminal tokens.",
    )
    _add_hidden_alias(p, "--interactive", dest="interactive", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--output-dir", default="")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--merge-config",
        default="",
        help="Keep unspecified gripper/tactile slots from an existing assignment path, or 'latest'.",
    )
    p.add_argument(
        "--merge-latest",
        dest="merge_config",
        action="store_const",
        const="latest",
        help="Shortcut for --merge-config latest.",
    )
    p.add_argument("--left-gripper", default=None)
    p.add_argument("--right-gripper", default=None)
    p.add_argument(
        "--no-auto-grippers",
        dest="auto_grippers",
        action="store_false",
        default=True,
        help=(
            "Do not auto-assign reachable grippers. By default, two reachable grippers "
            "map the first listed reachable gripper -> left and the second -> right after confirmation."
        ),
    )
    p.add_argument("--swap", action="store_true", help="Swap default two-gripper left/right assignment.")
    p.add_argument("--single-side", choices=SIDES, default="right")
    p.add_argument("--yes", action="store_true", help="Use default gripper assignment without prompts.")
    p.add_argument(
        "--no-gripper-motion-confirm",
        dest="gripper_motion_confirm",
        action="store_false",
        default=True,
        help="Skip micro-jog left/right confirmation and fall back to first/second reachable gripper assignment.",
    )
    p.add_argument("--left-inner", default=None)
    p.add_argument("--left-outer", default=None)
    p.add_argument("--right-inner", default=None)
    p.add_argument("--right-outer", default=None)
    p.add_argument(
        "--motion-test-gripper",
        dest="test_gripper",
        nargs="*",
        default=(),
        help="Run an explicit test motion on selected scan tokens before assignment.",
    )
    _add_hidden_alias(p, "--test-gripper", dest="test_gripper", nargs="*")
    p.add_argument("--visualize-tactile", nargs="*", default=())
    p.add_argument(
        "--assign-tactile-gui",
        action="store_true",
        help="Open force-visual assignment GUI for up to four DM-Tac tactile sensors.",
    )
    p.add_argument(
        "--tactile-gui-devices",
        nargs="*",
        default=(),
        help="Optional tactile scan indices/dev_ids to include in assignment GUI; defaults to discovered order.",
    )
    p.add_argument("--max-tactile-devices", type=int, default=4)
    p.add_argument("--tactile-gui-window", default="DM-Tac tactile assignment")
    p.add_argument("--tactile-gui-scale", type=int, default=1)
    p.add_argument("--tactile-gui-stride", type=int, default=16)
    p.add_argument("--tactile-gui-decimation", type=int, default=2)
    p.add_argument(
        "--tactile-gui-render-mode",
        choices=["normal-grid", "xy"],
        default="normal-grid",
        help="normal-grid renders a coarse force grid with XZ arrows and positive Y normal pressure; xy uses the old dense XY force view.",
    )
    p.add_argument("--tactile-gui-grid-rows", type=int, default=20)
    p.add_argument("--tactile-gui-grid-cols", type=int, default=30)
    p.add_argument(
        "--tactile-gui-vector-stride",
        type=int,
        default=2,
        help="Draw one XZ arrow every N grid cells in normal-grid mode. Use 1 for every 20x30 cell.",
    )
    p.add_argument(
        "--tactile-gui-normal-sign",
        type=float,
        default=1.0,
        help="Positive normal pressure is max(sign * Fy, 0). Use -1 if the sensor convention is inverted.",
    )
    p.add_argument(
        "--tactile-gui-pressure-deadband",
        type=float,
        default=5e-6,
        help="Display Y+ values below this as zero to avoid no-load flicker.",
    )
    p.add_argument(
        "--tactile-gui-pressure-ref",
        type=float,
        default=5e-5,
        help="Minimum display reference for Y+ color scaling; prevents auto-amplifying no-load noise.",
    )
    p.add_argument(
        "--tactile-gui-vector-deadband",
        type=float,
        default=2e-5,
        help="Display XZ vectors below this norm as zero to avoid no-load arrow flicker.",
    )
    p.add_argument(
        "--tactile-gui-vector-ref",
        type=float,
        default=5e-5,
        help="Minimum display reference for XZ vector scaling; prevents auto-amplifying no-load noise.",
    )
    p.add_argument(
        "--tactile-gui-denoise",
        action="store_true",
        help="Enable tactile force denoise/baseline smoothing in the assignment GUI. Off by default for latency diagnosis.",
    )
    p.add_argument("--tactile-gui-max-hz", type=float, default=30.0)
    p.add_argument(
        "--tactile-gui-no-record",
        action="store_true",
        help="Disable automatic tactile GUI debug recording.",
    )
    p.add_argument(
        "--tactile-gui-record-dir",
        default="",
        help=(
            "Exact directory for tactile GUI debug records. When supplied it "
            "overrides --tactile-gui-artifact-layout."
        ),
    )
    p.add_argument(
        "--tactile-gui-artifact-layout",
        choices=("legacy", "runtime-artifacts-v1"),
        default=_tactile_gui_artifact_layout_default(),
        help=(
            "Managed debug-record layout when --tactile-gui-record-dir is "
            "omitted. Standalone use defaults to legacy; inside DexTeleop, "
            "local_paths.json can coordinate the canonical cutover."
        ),
    )
    p.add_argument(
        "--tactile-gui-record-every-n",
        type=int,
        default=1,
        help="Record every N GUI frames.",
    )
    p.add_argument(
        "--tactile-gui-record-summary-only",
        action="store_true",
        help="Record timings/fids/summaries only, without the 20x30 force grids. This is the default.",
    )
    p.add_argument(
        "--tactile-gui-record-grid",
        action="store_true",
        help="Also record the 20x30 force grids. Useful for offline analysis, but slower and much larger.",
    )
    p.add_argument(
        "--tactile-gui-wait-for-new",
        action="store_true",
        help="Wait for a new tactile frame before rendering. Off by default to reduce assignment latency.",
    )
    p.add_argument(
        "--tactile-gui-sequential-read",
        action="store_true",
        help="Read tactile sensors sequentially instead of parallel. Useful for A/B latency checks.",
    )
    p.add_argument("--tactile-gui-baseline-frames", type=int, default=8)
    p.add_argument("--tactile-gui-fixed-ref", type=float, default=None)
    p.add_argument("--no-probe-grippers", action="store_true")
    p.add_argument("--probe-all-serial", action="store_true")
    p.add_argument(
        "--max-reachable-grippers",
        type=int,
        default=2,
        help=(
            "Stop expensive serial probing after this many reachable grippers "
            "are found (default: 2; use 0 to probe every candidate)."
        ),
    )
    p.add_argument(
        "--probe-system-ttys",
        action="store_true",
        help="Also probe /dev/ttyS* system serial ports. Off by default because they are usually noisy and permission denied.",
    )
    p.add_argument(
        "--no-scan-gripper-ids",
        dest="scan_gripper_ids",
        action="store_false",
        default=True,
        help=(
            "Only probe --gripper-id instead of automatically scanning Modbus ids "
            "1..247 on likely gripper ports."
        ),
    )
    p.add_argument("--gripper-id", type=int, default=1)
    _add_hidden_alias(p, "--id", dest="gripper_id", value_type=int)
    p.add_argument("--gripper-baudrate", dest="baudrate", type=int, default=115200)
    _add_hidden_alias(p, "--baudrate", dest="baudrate", value_type=int)
    p.add_argument(
        "--scan-baudrates",
        nargs="*",
        type=int,
        default=[115200, 57600, 19200],
        help=(
            "Baudrates to try while auto-probing likely gripper ports. "
            "--gripper-baudrate is tried first."
        ),
    )
    p.add_argument(
        "--no-scan-baudrates",
        action="store_true",
        help=(
            "Only probe --gripper-baudrate instead of also trying "
            "--scan-baudrates."
        ),
    )
    p.add_argument("--gripper-timeout", dest="timeout", type=float, default=0.03)
    _add_hidden_alias(p, "--timeout", dest="timeout", value_type=float)
    p.add_argument("--motion-test-amplitude", dest="motion_amplitude", type=float, default=0.3)
    _add_hidden_alias(p, "--motion-amplitude", dest="motion_amplitude", value_type=float)
    p.add_argument("--motion-test-speed", dest="motion_speed", type=float, default=0.1)
    _add_hidden_alias(p, "--motion-speed", dest="motion_speed", value_type=float)
    p.add_argument("--motion-test-torque", dest="motion_torque", type=float, default=0.1)
    _add_hidden_alias(p, "--motion-torque", dest="motion_torque", value_type=float)
    p.add_argument(
        "--gripper-jog-amplitude",
        type=float,
        default=0.3,
        help="Small normalized motion used during left/right gripper confirmation.",
    )
    p.add_argument("--gripper-jog-speed", type=float, default=0.3)
    p.add_argument("--gripper-jog-torque", type=float, default=0.3)
    p.add_argument("--tactile-backend", dest="backend", default="cpu", choices=["cpu", "cuda", "Flux", "flux"])
    _add_hidden_alias(p, "--backend", dest="backend")
    p.add_argument("--tactile-mode", dest="mode", default="standard", choices=["standard", "high"])
    _add_hidden_alias(p, "--mode", dest="mode")
    p.add_argument("--tactile-max-fps", dest="max_fps", type=int, default=120)
    _add_hidden_alias(p, "--max_fps", dest="max_fps", value_type=int)
    p.add_argument(
        "--tactile-gui-compute-profile",
        choices=TACTILE_COMPUTE_PROFILES,
        default="dist-only",
        help=(
            "SDK compute channels for GUI distributed_force. Use legacy if this SDK/firmware "
            "does not return force in dist-only mode."
        ),
    )
    p.add_argument("--tactile-timeout-ms", dest="timeout_ms", type=int, default=100)
    _add_hidden_alias(p, "--timeout_ms", dest="timeout_ms", value_type=int)
    p.add_argument("--visualize-count", type=int, default=0)
    p.add_argument("--visualize-max-hz", dest="max_hz", type=float, default=0.0)
    _add_hidden_alias(p, "--max_hz", dest="max_hz", value_type=float)
    p.add_argument("--reset-tactile", dest="reset_tactile", action="store_true")
    _add_hidden_alias(p, "--reset_tactile", dest="reset_tactile", action="store_true")
    p.add_argument("--warmup-frames", dest="warmup_frames", type=int, default=0)
    _add_hidden_alias(p, "--warmup_frames", dest="warmup_frames", value_type=int)

    p = sub.add_parser("verify-gripper", help="Read one gripper state without commanding motion.")
    _add_common(p)

    p = sub.add_parser("verify-tactile", help="Read frame ids from configured tactile sensors.")
    _add_common(p)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--tactile-timeout-ms", dest="timeout_ms", type=int, default=500)
    _add_hidden_alias(p, "--timeout_ms", dest="timeout_ms", value_type=int)

    p = sub.add_parser("read", help="Read a combined gripper/tactile snapshot stream.")
    _add_common(p)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--period-sec", dest="period_s", type=float, default=0.0)
    _add_hidden_alias(p, "--period_s", dest="period_s", value_type=float)
    p.add_argument("--tactile-timeout-ms", dest="timeout_ms", type=int, default=100)
    _add_hidden_alias(p, "--timeout_ms", dest="timeout_ms", value_type=int)
    p.add_argument("--no-gripper", dest="no_gripper", action="store_true")
    _add_hidden_alias(p, "--no_gripper", dest="no_gripper", action="store_true")
    p.add_argument("--no-tactile", dest="no_tactile", action="store_true")
    _add_hidden_alias(p, "--no_tactile", dest="no_tactile", action="store_true")

    p = sub.add_parser("visualize-force", help="Visualize live distributed-force fields.")
    _add_common(p)
    p.add_argument("--count", type=int, default=0, help="0 means run until q/ESC")
    p.add_argument("--tactile-timeout-ms", dest="timeout_ms", type=int, default=100)
    _add_hidden_alias(p, "--timeout_ms", dest="timeout_ms", value_type=int)
    p.add_argument("--max-hz", dest="max_hz", type=float, default=0.0, help="0 means no extra sleep")
    _add_hidden_alias(p, "--max_hz", dest="max_hz", value_type=float)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--arrow-scale", dest="arrow_scale", type=float, default=18.0)
    p.add_argument("--min-arrow-norm", dest="min_arrow_norm", type=float, default=0.015)
    p.add_argument("--deadband-abs", dest="deadband_abs", type=float, default=5e-5)
    p.add_argument("--temporal-alpha", dest="temporal_alpha", type=float, default=0.85)
    p.add_argument("--spatial-blur-ksize", dest="spatial_blur_ksize", type=int, default=5)
    p.add_argument("--arrow-pool", dest="arrow_pool", type=int, default=16)
    p.add_argument(
        "--viz-mode",
        dest="viz_mode",
        default="absolute",
        choices=["delta", "absolute", "frame_delta"],
        help="Display raw SDK force by default; delta modes explicitly enable preprocessing.",
    )
    p.add_argument("--baseline-frames", dest="baseline_frames", type=int, default=8)
    p.add_argument("--fixed-ref", dest="fixed_ref", type=float, default=None)
    p.add_argument("--norm-percentile", dest="norm_percentile", type=float, default=99.0)
    p.add_argument("--reset-tactile", dest="reset_tactile", action="store_true")
    p.add_argument("--warmup-frames", dest="warmup_frames", type=int, default=0)
    for legacy, dest, value_type in (
        ("--arrow_scale", "arrow_scale", float),
        ("--min_arrow_norm", "min_arrow_norm", float),
        ("--deadband_abs", "deadband_abs", float),
        ("--temporal_alpha", "temporal_alpha", float),
        ("--spatial_blur_ksize", "spatial_blur_ksize", int),
        ("--arrow_pool", "arrow_pool", int),
        ("--baseline_frames", "baseline_frames", int),
        ("--fixed_ref", "fixed_ref", float),
        ("--norm_percentile", "norm_percentile", float),
        ("--warmup_frames", "warmup_frames", int),
    ):
        _add_hidden_alias(p, legacy, dest=dest, value_type=value_type)
    _add_hidden_alias(p, "--viz_mode", dest="viz_mode")
    _add_hidden_alias(p, "--reset_tactile", dest="reset_tactile", action="store_true")
    p.add_argument("--window", default="DM-Tac distributed_force XY")

    p = sub.add_parser("probe-force", help="Print live distributed-force diagnostics.")
    _add_common(p)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--tactile-timeout-ms", dest="timeout_ms", type=int, default=100)
    _add_hidden_alias(p, "--timeout_ms", dest="timeout_ms", value_type=int)
    p.add_argument("--period-sec", dest="period_s", type=float, default=0.0)
    p.add_argument("--reset-tactile", dest="reset_tactile", action="store_true")
    p.add_argument("--warmup-frames", dest="warmup_frames", type=int, default=0)
    p.add_argument(
        "--denoise",
        dest="denoise",
        action="store_true",
        default=False,
        help="Explicitly enable distributed-force denoise/baseline processing.",
    )
    p.add_argument("--no-denoise", dest="denoise", action="store_false", help=argparse.SUPPRESS)
    p.add_argument("--denoise-mode", dest="denoise_mode", default="delta", choices=["delta", "absolute", "frame_delta"])
    p.add_argument("--baseline-frames", dest="baseline_frames", type=int, default=20)
    p.add_argument("--deadband-abs", dest="deadband_abs", type=float, default=5e-5)
    p.add_argument("--temporal-alpha", dest="temporal_alpha", type=float, default=0.85)
    p.add_argument("--spatial-blur-ksize", dest="spatial_blur_ksize", type=int, default=5)
    for legacy, dest, action, value_type in (
        ("--period_s", "period_s", None, float),
        ("--reset_tactile", "reset_tactile", "store_true", None),
        ("--warmup_frames", "warmup_frames", None, int),
        ("--no_denoise", "denoise", "store_false", None),
        ("--denoise_mode", "denoise_mode", None, None),
        ("--baseline_frames", "baseline_frames", None, int),
        ("--deadband_abs", "deadband_abs", None, float),
        ("--temporal_alpha", "temporal_alpha", None, float),
        ("--spatial_blur_ksize", "spatial_blur_ksize", None, int),
    ):
        _add_hidden_alias(p, legacy, dest=dest, action=action, value_type=value_type)

    p = sub.add_parser("reset-tactile", help="Reset tactile baselines; requires unloaded sensors.")
    _add_common(p)

    p = sub.add_parser("set-speed", help="Set the gripper speed parameter.")
    _add_common(p)
    p.add_argument("speed", type=float)

    p = sub.add_parser("set-torque", help="Set the gripper torque/force parameter.")
    _add_common(p)
    p.add_argument("torque", type=float)

    p = sub.add_parser("open", help="Command the gripper to the open endpoint.")
    _add_common(p)
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("close", help="Command the gripper to the closed endpoint.")
    _add_common(p)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--torque", type=float, default=None)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("move", help="Command one normalized gripper position.")
    _add_common(p)
    p.add_argument("position", type=float, help="Normalized close position in [0, 1].")
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--torque", type=float, default=None)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("stop", help="Send the gripper stop command.")
    _add_common(p)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "list-ports":
        _print_json(list_serial_devices())
        return

    if args.command == "list-tactile":
        _print_json(discover_devices())
        return

    if args.command == "configure-usb":
        _run_configure_usb(args)
        return

    _resolve_config_args(
        args,
        need_gripper=_command_needs_gripper(args),
        need_tactile=_command_needs_tactile(args),
    )

    if args.command == "verify-gripper":
        with InspireGripper(**_gripper_config(args).__dict__) as gripper:
            _print_gripper_state(gripper.get_state(), args.json)
        return

    if args.command == "verify-tactile":
        with DMTacGripperDevice(
            gripper=_gripper_config(args),
            tactile=_tactile_config(args),
            connect_gripper=False,
            connect_tactile=True,
        ) as dev:
            print("ready:", dev.wait_tactile_ready(timeout_s=5))
            for i in range(args.count):
                frames = dev.read_tactile(timeout_ms=args.timeout_ms)
                if args.json:
                    _print_json({k: None if v is None else v.summary() for k, v in frames.items()})
                else:
                    print(i, {k: None if v is None else v.fid for k, v in frames.items()})
        return

    if args.command == "read":
        with DMTacGripperDevice(
            gripper=_gripper_config(args),
            tactile=_tactile_config(args),
            connect_gripper=not args.no_gripper,
            connect_tactile=not args.no_tactile,
        ) as dev:
            if not args.no_tactile:
                print("tactile ready:", dev.wait_tactile_ready(timeout_s=5))
            for _ in range(args.count):
                snapshot = dev.read(
                    read_gripper=not args.no_gripper,
                    read_tactile=not args.no_tactile,
                    tactile_timeout_ms=args.timeout_ms,
                )
                if args.json:
                    _print_json(snapshot.summary())
                else:
                    print(format_snapshot_summary(snapshot))
                if args.period_s > 0:
                    time.sleep(args.period_s)
        return

    if args.command == "visualize-force":
        from .force_visualizer import ForceVizConfig, run_live_distributed_force_xy

        run_live_distributed_force_xy(
            tactile_config=_tactile_config(args),
            timeout_ms=args.timeout_ms,
            count=args.count,
            max_hz=args.max_hz,
            window=args.window,
            viz_config=ForceVizConfig(
                stride=args.stride,
                scale=args.scale,
                arrow_scale=args.arrow_scale,
                min_arrow_norm=args.min_arrow_norm,
                deadband_abs=args.deadband_abs,
                temporal_alpha=args.temporal_alpha,
                spatial_blur_ksize=args.spatial_blur_ksize,
                arrow_pool=args.arrow_pool,
                mode=args.viz_mode,
                baseline_frames=args.baseline_frames,
                fixed_ref=args.fixed_ref,
                norm_percentile=args.norm_percentile,
            ),
            reset_tactile=args.reset_tactile,
            warmup_frames=args.warmup_frames,
            close_devices=False,
        )
        _fast_exit(0)

    if args.command == "probe-force":
        from .force_visualizer import probe_distributed_force

        tactile_config = _tactile_config(args)
        tactile_config.denoise_distributed_force = bool(args.denoise)
        tactile_config.denoise_config = TactileDenoiseConfig(
            enabled=bool(args.denoise),
            mode=args.denoise_mode,
            baseline_frames=args.baseline_frames,
            deadband_abs=args.deadband_abs,
            temporal_alpha=args.temporal_alpha,
            spatial_blur_ksize=args.spatial_blur_ksize,
        )
        probe_distributed_force(
            tactile_config=tactile_config,
            timeout_ms=args.timeout_ms,
            count=args.count,
            period_s=args.period_s,
            reset_tactile=args.reset_tactile,
            warmup_frames=args.warmup_frames,
            close_devices=False,
        )
        _fast_exit(0)

    if args.command == "reset-tactile":
        dev = DMTacGripperDevice(
            tactile=_tactile_config(args),
            connect_gripper=False,
            connect_tactile=True,
        ).connect()
        print("tactile ready:", dev.wait_tactile_ready(timeout_s=10))
        print("reset tactile: keep all sensor surfaces unloaded")
        dev.reset_tactile()
        print("tactile ready after reset:", dev.wait_tactile_ready(timeout_s=10))
        _fast_exit(0)

    with DMTacGripperDevice(
        gripper=_gripper_config(args),
        tactile=_tactile_config(args),
        connect_gripper=True,
        connect_tactile=False,
    ) as dev:
        if args.command == "set-speed":
            dev.set_speed(args.speed)
            print("speed:", dev.get_speed())
        elif args.command == "set-torque":
            dev.set_torque(args.torque)
            print("torque:", dev.get_torque())
        elif args.command == "open":
            state = dev.open(wait=args.wait)
            if state is not None:
                _print_gripper_state(state, args.json)
        elif args.command == "close":
            state = dev.close_gripper(
                speed=args.speed,
                torque=args.torque,
                continuous=args.continuous,
                wait=args.wait,
            )
            if state is not None:
                _print_gripper_state(state, args.json)
        elif args.command == "move":
            state = dev.move(
                args.position,
                speed=args.speed,
                torque=args.torque,
                continuous=args.continuous,
                wait=args.wait,
            )
            if state is not None:
                _print_gripper_state(state, args.json)
        elif args.command == "stop":
            dev.stop()


if __name__ == "__main__":
    main()
