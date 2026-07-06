from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from serial.tools import list_ports

from cameras.configs import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH

from .sources.camera import load_realsense_camera_config_map


SIDES = ("left", "right")
CAMERA_KEYS = ("front", "left_wrist", "right_wrist")


class DevicePreflightError(RuntimeError):
    pass


@dataclass
class DevicePreflightOptions:
    active_sides: tuple[str, ...] = ("left", "right")
    camera_config: str | Path = DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
    require_camera: bool = True
    require_q6: bool = True
    require_dmtac_gripper: bool = True
    require_dmtac_tactile: bool = True
    auto_configure: bool = True
    assume_yes: bool = False
    fix_permissions: bool = False
    fix_serial_conflicts: bool = False
    probe_all_serial: bool = False
    q6_config: str | Path = "latest"
    dmtac_config: str | Path = "latest"
    dmtac_tactile_gui: bool = True
    dmtac_tactile_backend: str = "cpu"


@dataclass
class DevicePreflightResult:
    ok: bool
    messages: list[str] = field(default_factory=list)
    q6_ports: dict[str, str] = field(default_factory=dict)
    dmtac_assignment: dict[str, Any] | None = None
    camera_config_path: str | None = None
    permission_fixes: list[dict[str, Any]] = field(default_factory=list)
    serial_conflict_report: dict[str, Any] | None = None

    def add(self, message: str) -> None:
        self.messages.append(str(message))


def _active_sides(active_sides: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for side in active_sides:
        text = str(side).strip().lower()
        if text == "both":
            normalized.extend(["left", "right"])
            continue
        if text not in SIDES:
            raise ValueError(f"Unsupported side {side!r}; expected left/right/both.")
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized or ("left", "right"))


def _serial_device_paths() -> list[str]:
    paths: list[str] = []
    for item in list_ports.comports():
        path = str(getattr(item, "device", "") or "")
        if path and (
            path.startswith("/dev/ttyUSB")
            or path.startswith("/dev/ttyACM")
            or path.startswith("/dev/serial/")
        ):
            paths.append(path)
    return paths


def _video_device_paths() -> list[str]:
    paths = [*Path("/dev").glob("video*"), *Path("/dev").glob("media*")]
    return sorted(str(path) for path in paths)


def _usb_bus_device_paths() -> list[str]:
    return sorted(str(path) for path in Path("/dev/bus/usb").glob("*/*"))


_BRLTTY_UDEV_RULE_PATHS = (
    Path("/etc/udev/rules.d/85-brltty.rules"),
    Path("/lib/udev/rules.d/85-brltty.rules"),
    Path("/usr/lib/udev/rules.d/85-brltty.rules"),
)
_BRLTTY_UNITS = ("brltty.service", "brltty-udev.service")
_SERIAL_DRIVER_MODULES = ("ch341", "cp210x")


def _run_command(command: Sequence[str], *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
    except FileNotFoundError as exc:
        return {
            "command": " ".join(str(item) for item in command),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "missing_command": True,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(str(item) for item in command),
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timed out after {timeout}s",
            "timeout": True,
        }
    return {
        "command": " ".join(str(item) for item in command),
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout or "").strip(),
        "stderr": str(completed.stderr or "").strip(),
    }


def _systemctl_probe(unit: str, verb: str) -> dict[str, Any]:
    report = _run_command(["systemctl", verb, unit], timeout=3.0)
    report["unit"] = str(unit)
    report["verb"] = str(verb)
    return report


def _brltty_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for comm_path in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = comm_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if "brltty" not in name.lower():
            continue
        processes.append({"pid": comm_path.parent.name, "name": name})
    return processes


def detect_brltty_serial_conflict() -> dict[str, Any]:
    rule_paths = [str(path) for path in _BRLTTY_UDEV_RULE_PATHS if path.exists()]
    executable_paths = [
        str(path)
        for path in (Path("/usr/bin/brltty"), Path("/bin/brltty"), Path("/sbin/brltty"))
        if path.exists()
    ]
    processes = _brltty_processes()
    unit_reports = []
    for unit in _BRLTTY_UNITS:
        unit_reports.append(
            {
                "unit": unit,
                "active": _systemctl_probe(unit, "is-active"),
                "enabled": _systemctl_probe(unit, "is-enabled"),
            }
        )
    active_units = [
        item["unit"]
        for item in unit_reports
        if item["active"].get("returncode") == 0 and str(item["active"].get("stdout")).strip() == "active"
    ]
    enabled_units = [
        item["unit"]
        for item in unit_reports
        if item["enabled"].get("returncode") == 0
        and str(item["enabled"].get("stdout")).strip() in {"enabled", "static", "generated", "alias"}
    ]
    return {
        "kind": "brltty_usb_serial_conflict",
        "rule_paths": rule_paths,
        "executable_paths": executable_paths,
        "processes": processes,
        "units": unit_reports,
        "active_units": active_units,
        "enabled_units": enabled_units,
        "serial_ports": _serial_device_paths(),
        "risk": bool(rule_paths or executable_paths or processes or active_units or enabled_units),
    }


def _command_missing_unit(report: Mapping[str, Any]) -> bool:
    text = f"{report.get('stdout') or ''}\n{report.get('stderr') or ''}".lower()
    return "not found" in text or "not loaded" in text or "does not exist" in text


def repair_serial_conflicts(*, apply: bool = False) -> dict[str, Any]:
    report = detect_brltty_serial_conflict()
    report["actions"] = []
    report["changed"] = False
    report["ok"] = True
    if not report.get("risk"):
        return report
    if not apply:
        return report
    if getattr(os, "geteuid", lambda: 1)() != 0:
        report["ok"] = False
        report["error"] = "root_required: rerun this step with sudo."
        return report

    for unit in _BRLTTY_UNITS:
        for verb in ("stop", "disable", "mask"):
            action = _run_command(["systemctl", verb, unit], timeout=10.0)
            action["unit"] = unit
            action["verb"] = verb
            action["nonfatal_missing_unit"] = bool(
                action.get("returncode") not in (0, None)
                and _command_missing_unit(action)
            )
            report["actions"].append(action)
            if action.get("returncode") == 0:
                report["changed"] = True
            elif not action.get("nonfatal_missing_unit"):
                report["ok"] = False

    reload_action = _run_command(["udevadm", "control", "--reload-rules"], timeout=10.0)
    report["actions"].append(reload_action)
    if reload_action.get("returncode") not in (0, None):
        report["ok"] = False

    for module in _SERIAL_DRIVER_MODULES:
        action = _run_command(["modprobe", module], timeout=10.0)
        action["module"] = module
        report["actions"].append(action)
        if action.get("returncode") == 0:
            report["changed"] = True
        elif not action.get("missing_command"):
            # Some kernels only ship one of these USB serial drivers. Keep the
            # report visible, but do not fail the brltty service fix for that.
            action["nonfatal_module_load"] = True
    return report


def _serial_conflict_report_ok(report: Mapping[str, Any] | None) -> bool:
    if not isinstance(report, Mapping):
        return True
    return bool(report.get("ok", True))


def _serial_conflict_reports_need_attention(report: Mapping[str, Any] | None) -> bool:
    if not isinstance(report, Mapping):
        return False
    return bool(report.get("risk")) and not bool(report.get("changed"))


def _serial_conflict_text(report: Mapping[str, Any] | None, *, applied: bool = False) -> str:
    if not isinstance(report, Mapping) or not report.get("risk"):
        return ""
    python = sys.executable or "python"
    lines = [
        "Potential USB serial conflict detected: brltty is installed, active, or has udev rules present.",
        "This can claim CH340/CP210x USB-serial adapters before Linux creates a stable /dev/ttyUSB* gripper port.",
    ]
    rule_paths = list(report.get("rule_paths") or [])
    if rule_paths:
        lines.append("  brltty udev rules: " + ", ".join(str(path) for path in rule_paths))
    active_units = list(report.get("active_units") or [])
    enabled_units = list(report.get("enabled_units") or [])
    if active_units:
        lines.append("  active units: " + ", ".join(str(unit) for unit in active_units))
    if enabled_units:
        lines.append("  enabled units: " + ", ".join(str(unit) for unit in enabled_units))
    processes = list(report.get("processes") or [])
    if processes:
        lines.append(
            "  running processes: "
            + ", ".join(f"{item.get('name')}({item.get('pid')})" for item in processes if isinstance(item, Mapping))
        )
    if applied:
        if report.get("ok"):
            lines.append("Applied serial-conflict fix: stopped/disabled/masked known brltty units and reloaded udev rules.")
            lines.append("Replug the gripper USB/HUB, then rerun --fix-permissions so the new ttyUSB nodes get permissions.")
        else:
            lines.append("Serial-conflict fix did not fully apply.")
            if report.get("error"):
                lines.append(f"  error: {report.get('error')}")
            for action in report.get("actions") or []:
                if not isinstance(action, Mapping):
                    continue
                if action.get("returncode") in (0, None) or action.get("nonfatal_missing_unit"):
                    continue
                lines.append(
                    f"  failed: {action.get('command')} rc={action.get('returncode')} "
                    f"stderr={action.get('stderr')}"
                )
    lines.append("Standard operation when /dev/ttyUSB* is missing or lsusb -t shows Driver=usbfs:")
    lines.append("  1. If this workstation does not use a Braille display, run:")
    lines.append(f"     sudo -E {python} -m teleop_runtime.configure_devices --fix-serial-conflicts")
    lines.append("  2. Replug the gripper USB/HUB.")
    lines.append(f"  3. Run: sudo -E {python} -m teleop_runtime.configure_devices --fix-permissions")
    lines.append("If a Braille display is required, do not disable brltty; use a device-specific udev exception instead.")
    return "\n".join(lines)


def _append_serial_conflict_report(
    result: DevicePreflightResult,
    *,
    fix_serial_conflicts: bool,
) -> None:
    report = repair_serial_conflicts(apply=bool(fix_serial_conflicts))
    result.serial_conflict_report = report
    text = _serial_conflict_text(report, applied=bool(fix_serial_conflicts))
    if text:
        result.add(text)
    if not _serial_conflict_report_ok(report):
        result.ok = False


def _existing_unique_paths(paths: Sequence[str | Path]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in paths:
        path = str(item)
        if not path or path in seen or not Path(path).exists():
            continue
        seen.add(path)
        result.append(path)
    return result


def _path_access(path: str) -> dict[str, Any]:
    try:
        stat = os.stat(path)
        mode = stat.st_mode & 0o777
        return {
            "path": path,
            "exists": True,
            "readable": bool(os.access(path, os.R_OK)),
            "writable": bool(os.access(path, os.W_OK)),
            "mode": oct(mode),
            "mode_int": int(mode),
            "uid": int(stat.st_uid),
            "gid": int(stat.st_gid),
        }
    except OSError as exc:
        return {
            "path": path,
            "exists": False,
            "readable": False,
            "writable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def repair_device_permissions(
    paths: Sequence[str | Path],
    *,
    apply: bool = False,
    user: str | None = None,
) -> list[dict[str, Any]]:
    del user
    root_required = bool(apply and getattr(os, "geteuid", lambda: 1)() != 0)
    desired_mode = 0o666
    reports: list[dict[str, Any]] = []
    for path in _existing_unique_paths(paths):
        before = _path_access(path)
        target_mode = oct(desired_mode) if apply else None
        report = {
            "path": path,
            "before": before,
            "changed": False,
            "command": None,
            "error": None,
            "target_mode": target_mode,
        }
        current_user_can_rw = bool(before.get("readable") and before.get("writable"))
        current_mode = before.get("mode_int")
        needs_action = current_mode != desired_mode if apply else not current_user_can_rw
        if not needs_action:
            report["after"] = before
            reports.append(report)
            continue
        command = ["chmod", "666", path]
        report["command"] = " ".join(command)
        if apply:
            if root_required:
                report["error"] = "root_required: rerun this permission step with sudo."
            else:
                try:
                    os.chmod(path, 0o666)
                    report["changed"] = True
                except Exception as exc:  # noqa: BLE001
                    report["error"] = f"{type(exc).__name__}: {exc}"
        report["after"] = _path_access(path)
        reports.append(report)
    return reports


def _permission_report_ok(report: Mapping[str, Any]) -> bool:
    after = report.get("after")
    if not isinstance(after, Mapping):
        return False
    if report.get("target_mode"):
        return after.get("mode") == report.get("target_mode")
    return bool(after.get("readable") and after.get("writable"))


def _permission_reports_need_attention(reports: Sequence[Mapping[str, Any]]) -> bool:
    return any(not _permission_report_ok(report) for report in reports)


def _permission_error_text(reports: Sequence[Mapping[str, Any]]) -> str:
    bad = [
        report
        for report in reports
        if not _permission_report_ok(report)
    ]
    if not bad:
        return ""
    lines = ["Some device nodes are not readable/writable by the current user:"]
    for item in bad:
        after = item.get("after") or item.get("before") or {}
        lines.append(
            f"  {item.get('path')} mode={after.get('mode')} uid={after.get('uid')} gid={after.get('gid')}"
        )
        command = item.get("command")
        if command:
            lines.append(f"    chmod: sudo {command}")
        if item.get("error"):
            lines.append(f"    error: {item.get('error')}")
    python = sys.executable or "python"
    lines.append(
        "Run the permission script with sudo, for example: "
        f"sudo -E {python} -m teleop_runtime.configure_devices --fix-permissions"
    )
    lines.append("Long term: install udev rules or add the user to dialout/video, then re-login and replug USB.")
    return "\n".join(lines)


def _camera_records_from_context() -> list[dict[str, Any]]:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise DevicePreflightError("pyrealsense2 is not importable; RealSense preflight cannot run.") from exc
    records: list[dict[str, Any]] = []
    ctx = rs.context()
    for index, dev in enumerate(ctx.query_devices()):
        record: dict[str, Any] = {"index": int(index)}
        for name, option in (
            ("name", rs.camera_info.name),
            ("serial", rs.camera_info.serial_number),
            ("product_line", rs.camera_info.product_line),
            ("usb_type", rs.camera_info.usb_type_descriptor),
        ):
            try:
                if dev.supports(option):
                    record[name] = str(dev.get_info(option))
            except Exception:
                continue
        records.append(record)
    return records


def compare_realsense_config(
    camera_config: str | Path,
    discovered: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    configs = load_realsense_camera_config_map(camera_config)
    discovered_by_serial = {
        str(item.get("serial")): dict(item)
        for item in discovered
        if item.get("serial") not in (None, "")
    }
    missing = []
    matched = []
    for key, config in configs.items():
        serial = str(config.serial)
        current = discovered_by_serial.get(serial)
        if current is None:
            missing.append({"camera_key": str(key), "serial": serial})
        else:
            matched.append({"camera_key": str(key), "serial": serial, "current": current})
    extra = [
        dict(item)
        for item in discovered
        if str(item.get("serial") or "") not in {str(config.serial) for config in configs.values()}
    ]
    return {"missing": missing, "matched": matched, "extra": extra, "configured": configs}


def _prompt_token(prompt: str, valid_tokens: set[str], *, default: str | None = None) -> str:
    while True:
        suffix = "" if default is None else f" [{default}]"
        answer = input(f"{prompt}{suffix}: ").strip()
        if not answer and default is not None:
            answer = default
        if answer in valid_tokens:
            return answer
        print(f"Invalid selection {answer!r}; choose one of {sorted(valid_tokens)}.")


def configure_realsense_interactive(
    camera_config: str | Path,
    discovered: Sequence[Mapping[str, Any]],
    *,
    assume_yes: bool = False,
) -> Path:
    path = Path(camera_config)
    configs = load_realsense_camera_config_map(path)
    serials = [str(item.get("serial")) for item in discovered if item.get("serial") not in (None, "")]
    if len(serials) < len(configs):
        raise DevicePreflightError(
            f"RealSense count is insufficient: detected {len(serials)}, configured {len(configs)}."
        )
    if not assume_yes and not sys.stdin.isatty():
        raise DevicePreflightError(
            "RealSense serial assignment requires interactive stdin. Run the device script in a terminal."
        )
    print("Detected RealSense cameras:")
    for item in discovered:
        print(
            f"  [{item.get('index')}] serial={item.get('serial')} "
            f"name={item.get('name', '-')}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DevicePreflightError(f"{path} must contain an object.")
    used: set[str] = set()
    valid = {str(item.get("index")) for item in discovered}
    valid.update(serials)
    by_index = {str(item.get("index")): str(item.get("serial")) for item in discovered}
    if assume_yes:
        selected = dict(zip(configs.keys(), serials))
    else:
        selected = {}
        for key, config in configs.items():
            default = str(config.serial) if str(config.serial) in serials and str(config.serial) not in used else None
            token = _prompt_token(f"Assign camera '{key}' by index/serial", valid, default=default)
            serial = by_index.get(token, token)
            if serial in used:
                raise DevicePreflightError(f"RealSense serial {serial} was assigned more than once.")
            used.add(serial)
            selected[str(key)] = serial
    new_payload = dict(payload)
    for key, serial in selected.items():
        item = dict(new_payload.get(key) or {})
        item["serial"] = str(serial)
        new_payload[key] = item
    path.write_text(json.dumps(new_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _load_q6_module():
    from hardware.teleop_hand import q6_assignment

    return q6_assignment


def _q6_device_label(item: Mapping[str, Any]) -> str:
    return (
        f"q6_index={item.get('q6_index')} "
        f"serial_index={item.get('index')} "
        f"port={item.get('preferred_port') or item.get('port')}"
    )


def _prompt_q6_side_token(
    q6: Any,
    scan: Mapping[str, Any],
    *,
    side: str,
    used_q6_indices: set[str],
    default: str | None = None,
    input_func: Callable[[str], str] | None = None,
) -> str:
    reader = input if input_func is None else input_func
    suffix = "" if default is None else f" [{default}]"
    while True:
        token = reader(f"Assign Q6 for {side} by q6_index / serial_index / port{suffix}: ").strip()
        if not token and default is not None:
            token = str(default)
        try:
            item = q6.resolve_q6_device_token(scan, token)
        except ValueError as exc:
            print(str(exc))
            continue
        if item is None:
            print(f"{side} Q6 cannot be empty.")
            continue
        q6_index = str(item.get("q6_index"))
        if q6_index in used_q6_indices:
            print(f"q6_index {q6_index} has already been assigned; choose another Q6.")
            continue
        used_q6_indices.add(q6_index)
        return str(token)


def _configure_q6_tokens_from_preview(
    q6: Any,
    scan: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
    assume_yes: bool,
    input_func: Callable[[str], str] | None = None,
) -> dict[str, str | None]:
    devices = list(scan.get("q6_devices") or [])
    tokens: dict[str, str | None] = {"left": None, "right": None}
    sides = tuple(str(side) for side in active_sides)
    if assume_yes:
        if len(devices) < len(sides):
            raise DevicePreflightError(f"Q6 count is insufficient: detected {len(devices)}, expected {len(sides)}.")
        if len(devices) > len(sides):
            raise DevicePreflightError(
                "More Q6 devices than active sides were detected; run interactive device configuration."
            )
        for index, side in enumerate(sides):
            tokens[side] = str(index)
        return tokens

    if not sys.stdin.isatty() and input_func is None:
        raise DevicePreflightError(
            "Q6 left/right confirmation requires interactive stdin. "
            "Run `conda run --no-capture-output -n dexteleop310 python -m "
            "teleop_runtime.debug.run_vive_rm65_real_teleop ...` or run "
            "`python hardware/teleop_hand/configure_q6.py configure` first."
        )

    print("\nQ6 assignment preview:")
    print("Move one Q6 at a time, watch which q6_index changes, then assign it to left/right.")
    try:
        q6.visualize_q6_devices(
            devices,
            baudrate=int(scan.get("baudrate", getattr(q6, "DEFAULT_BAUDRATE", 115200))),
            timeout=float(scan.get("timeout", getattr(q6, "DEFAULT_TIMEOUT", 0.05))),
            duration_s=float(getattr(q6, "DEFAULT_VISUALIZE_SEC", 20.0)),
            log_hz=float(getattr(q6, "DEFAULT_PREVIEW_LOG_HZ", 5.0)),
        )
    except KeyboardInterrupt:
        print("\nQ6 preview skipped.")
    print("Detected Q6 devices for assignment:")
    for item in devices:
        print(f"  {_q6_device_label(item)}")

    used: set[str] = set()
    for offset, side in enumerate(sides):
        default = str(offset) if len(devices) == len(sides) else None
        tokens[side] = _prompt_q6_side_token(
            q6,
            scan,
            side=side,
            used_q6_indices=used,
            default=default,
            input_func=input_func,
        )
    return tokens


def prepare_q6(
    options: DevicePreflightOptions,
    result: DevicePreflightResult,
) -> None:
    q6 = _load_q6_module()
    scan = q6.scan_q6_devices(probe_all_serial=bool(options.probe_all_serial))
    expected = len(options.active_sides)
    devices = list(scan.get("q6_devices") or [])
    result.add(q6.format_scan_summary(scan))
    if len(devices) < expected:
        raise DevicePreflightError(f"Q6 count is insufficient: detected {len(devices)}, expected {expected}.")

    try:
        assignment = q6.load_assignment(options.q6_config)
        comparison = q6.compare_assignment_to_scan(assignment, scan)
        missing = [
            item for item in comparison.get("missing") or []
            if str(item.get("side")) in options.active_sides
        ]
        effective = comparison["effective_assignment"]
        ports = q6.q6_ports_for_sides(effective, active_sides=options.active_sides)
        if not missing and all(side in ports for side in options.active_sides):
            if comparison.get("remapped"):
                saved = q6.save_assignment(effective)
                result.add(f"Q6 paths changed; saved remapped latest config: {saved}")
            result.q6_ports = dict(ports)
            return
        result.add("Q6 latest assignment is missing active sides or no longer matches current USB scan.")
    except FileNotFoundError:
        result.add("No latest Q6 assignment found.")

    if not options.auto_configure:
        raise DevicePreflightError("Q6 assignment is not ready; run `python -m teleop_runtime.configure_devices --configure`.")
    tokens = _configure_q6_tokens_from_preview(
        q6,
        scan,
        active_sides=options.active_sides,
        assume_yes=bool(options.assume_yes),
    )
    assignment = q6.build_assignment_from_tokens(
        scan,
        left=tokens.get("left"),
        right=tokens.get("right"),
    )
    saved = q6.save_assignment(assignment)
    result.q6_ports = q6.q6_ports_for_sides(assignment, active_sides=options.active_sides)
    result.add(f"Saved Q6 assignment: {saved}")


def _ensure_dmtac_path() -> None:
    root = Path(__file__).resolve().parents[1] / "hardware" / "DMTAC_gripper"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_dmtac_modules():
    _ensure_dmtac_path()
    from portable_dmtac_gripper import usb_assignment
    from portable_dmtac_gripper.tactile_assignment_gui import (
        TactileAssignmentGuiConfig,
        run_tactile_assignment_gui,
        tactile_slot_tokens_from_assignment,
    )

    return usb_assignment, TactileAssignmentGuiConfig, run_tactile_assignment_gui, tactile_slot_tokens_from_assignment


def _reachable_grippers(scan: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in scan.get("grippers") or []:
        probe = item.get("probe") or {}
        if isinstance(probe, Mapping) and probe.get("reachable") is True:
            items.append(dict(item))
    return items


def _configured_dmtac_complete(
    usb_assignment: Any,
    assignment: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
    require_tactile: bool,
) -> bool:
    for side in active_sides:
        if usb_assignment.gripper_for_side(assignment, side) is None:
            return False
        if require_tactile and len(usb_assignment.tactile_ids_for_side(assignment, side)) < 2:
            return False
    return True


def _dmtac_gripper_token(item: Mapping[str, Any]) -> str:
    return str(item.get("index"))


def _dmtac_tactile_token(item: Mapping[str, Any]) -> str:
    dev_id = item.get("dev_id")
    if dev_id not in (None, ""):
        return str(dev_id)
    return str(item.get("index"))


def _dmtac_gripper_label(item: Mapping[str, Any]) -> str:
    return str(item.get("preferred_port") or item.get("port") or item.get("device") or item.get("index"))


def _dmtac_gripper_port(item: Mapping[str, Any]) -> str:
    return str(item.get("preferred_port") or item.get("port") or item.get("device") or "")


def _dmtac_gripper_id(usb_assignment: Any, item: Mapping[str, Any], default: int = 1) -> int:
    detected = usb_assignment.gripper_id_from_probe(item.get("probe"))
    return int(detected if detected is not None else default)


def _dmtac_gripper_baudrate(usb_assignment: Any, item: Mapping[str, Any], default: int = 115200) -> int:
    detected = usb_assignment.gripper_baudrate_from_probe(item.get("probe"))
    return int(detected if detected is not None else default)


def _dmtac_missing_gripper_sides(
    usb_assignment: Any,
    assignment: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
) -> list[str]:
    missing = {
        str(item.get("side"))
        for item in comparison.get("missing") or []
        if item.get("kind") == "gripper"
    }
    for side in active_sides:
        if usb_assignment.gripper_for_side(assignment, side) is None:
            missing.add(str(side))
    return [side for side in active_sides if side in missing]


def _dmtac_missing_tactile_slots(
    usb_assignment: Any,
    assignment: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
    require_tactile: bool,
) -> list[tuple[str, str]]:
    if not require_tactile:
        return []
    missing = {
        (str(item.get("side")), str(item.get("slot")))
        for item in comparison.get("missing") or []
        if item.get("kind") == "tactile"
    }
    for side in active_sides:
        tactile = (assignment.get("assignments") or {}).get(str(side), {}).get("tactile") or {}
        for slot in ("inner", "outer"):
            if tactile.get(slot) is None:
                missing.add((str(side), str(slot)))
    return [(side, slot) for side in active_sides for slot in ("inner", "outer") if (side, slot) in missing]


def _dmtac_extra_devices_by_kind(comparison: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        dict(item.get("current") or {})
        for item in comparison.get("extra") or []
        if item.get("kind") == kind
    ]


def _confirm_dmtac_inference(
    lines: Sequence[str],
    *,
    assume_yes: bool,
    input_func: Callable[[str], str] = input,
) -> None:
    if not lines:
        return
    print("DMTAC assignment can be completed from the existing config and current USB scan:")
    for line in lines:
        print(f"  {line}")
    if assume_yes:
        return
    if not sys.stdin.isatty() and input_func is input:
        raise DevicePreflightError(
            "DMTAC partial assignment inference requires interactive stdin. "
            "Run `python -m teleop_runtime.configure_devices --configure` in a terminal."
        )
    answer = input_func("Use this inferred DMTAC assignment and save it? Enter=yes, n=no: ").strip().lower()
    if answer in {"n", "no", "q", "quit"}:
        raise DevicePreflightError("User rejected inferred DMTAC assignment.")


def _infer_dmtac_assignment_from_partial_config(
    usb_assignment: Any,
    scan: Mapping[str, Any],
    assignment: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
    require_tactile: bool,
    assume_yes: bool,
    input_func: Callable[[str], str] = input,
) -> dict[str, Any] | None:
    effective = comparison["effective_assignment"]
    missing_gripper_sides = _dmtac_missing_gripper_sides(
        usb_assignment,
        effective,
        comparison,
        active_sides=active_sides,
    )
    missing_tactile_slots = _dmtac_missing_tactile_slots(
        usb_assignment,
        effective,
        comparison,
        active_sides=active_sides,
        require_tactile=require_tactile,
    )
    if not missing_gripper_sides and not missing_tactile_slots:
        return None

    extra_grippers = _dmtac_extra_devices_by_kind(comparison, "gripper")
    extra_tactile = _dmtac_extra_devices_by_kind(comparison, "tactile")
    gripper_tokens: dict[str, str | None] = {"left": None, "right": None}
    tactile_tokens: dict[str, str | None] = {}
    explanation: list[str] = []

    if missing_gripper_sides:
        if len(missing_gripper_sides) != 1 or len(extra_grippers) != 1:
            return None
        side = missing_gripper_sides[0]
        gripper_tokens[side] = _dmtac_gripper_token(extra_grippers[0])
        explanation.append(f"{side} gripper <- {_dmtac_gripper_label(extra_grippers[0])}")

    if missing_tactile_slots:
        if len(missing_tactile_slots) != 1 or len(extra_tactile) != 1:
            return None
        side, slot = missing_tactile_slots[0]
        token_name = f"{side}_{slot}"
        tactile_tokens[token_name] = _dmtac_tactile_token(extra_tactile[0])
        explanation.append(f"{side} {slot} tactile <- {extra_tactile[0].get('dev_id')}")

    _confirm_dmtac_inference(explanation, assume_yes=assume_yes, input_func=input_func)
    inferred = usb_assignment.build_assignment_from_tokens(
        scan,
        left_gripper=gripper_tokens.get("left"),
        right_gripper=gripper_tokens.get("right"),
        left_inner=tactile_tokens.get("left_inner"),
        left_outer=tactile_tokens.get("left_outer"),
        right_inner=tactile_tokens.get("right_inner"),
        right_outer=tactile_tokens.get("right_outer"),
    )
    return usb_assignment.merge_assignment(effective, inferred)


def _prompt_dmtac_side_after_jog(input_func: Callable[[str], str]) -> str:
    while True:
        answer = input_func("Which gripper moved? [left/right/skip/retry/q]: ").strip().lower()
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


def _motion_confirm_dmtac_gripper_tokens(
    usb_assignment: Any,
    scan: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
    known_tokens: Mapping[str, str | None] | None = None,
    input_func: Callable[[str], str] = input,
    motion_func: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, str | None]:
    tokens: dict[str, str | None] = {"left": None, "right": None}
    if known_tokens:
        for side, token in known_tokens.items():
            if side in tokens and token not in (None, ""):
                tokens[side] = str(token)
    missing_sides = [side for side in active_sides if tokens.get(side) is None]
    if not missing_sides:
        return tokens
    if not sys.stdin.isatty() and input_func is input:
        raise DevicePreflightError(
            "DMTAC gripper motion assignment requires interactive stdin. "
            "Run `python -m teleop_runtime.configure_devices --configure` in a terminal."
        )
    motion = motion_func or usb_assignment.run_gripper_motion_test
    used = {token for token in tokens.values() if token is not None}
    candidates = [
        dict(item)
        for item in _reachable_grippers(scan)
        if _dmtac_gripper_token(item) not in used
    ]
    if len(candidates) < len(missing_sides):
        raise DevicePreflightError(
            f"DMTAC gripper count is insufficient after existing config: "
            f"candidates {len(candidates)}, missing {len(missing_sides)}."
        )

    print("\nDMTAC gripper motion assignment:")
    print("Press Enter to micro-jog each unassigned gripper, then type which physical side moved.")
    print("Keep fingers and objects clear before each jog.")
    for item in candidates:
        token = _dmtac_gripper_token(item)
        label = _dmtac_gripper_label(item)
        while True:
            answer = input_func(
                f"\n[{token}] {label}: press Enter to micro-jog, type left/right to assign without jog, "
                "s=skip, q=finish: "
            ).strip().lower()
            if answer in {"q", "quit", "done", "finish"}:
                return tokens
            if answer in {"s", "skip", "none", "-"}:
                break
            if answer in {"l", "left", "r", "right"}:
                side = "left" if answer in {"l", "left"} else "right"
            else:
                print(f"Micro-jogging [{token}] {label}...")
                motion(
                    _dmtac_gripper_port(item),
                    gripper_id=_dmtac_gripper_id(usb_assignment, item),
                    baudrate=_dmtac_gripper_baudrate(usb_assignment, item),
                    timeout=0.03,
                    amplitude=float(getattr(usb_assignment, "DEFAULT_GRIPPER_MOTION_AMPLITUDE", 0.3)),
                    speed=float(getattr(usb_assignment, "DEFAULT_GRIPPER_TEST_SPEED", 0.1)),
                    torque=float(getattr(usb_assignment, "DEFAULT_GRIPPER_TEST_TORQUE", 0.1)),
                )
                side = _prompt_dmtac_side_after_jog(input_func)
                if side == "retry":
                    continue
                if side == "quit":
                    return tokens
                if side == "skip":
                    break
            if side not in active_sides:
                print(f"{side} is not active in this run; active sides={tuple(active_sides)}.")
                continue
            existing = tokens.get(side)
            if existing is not None and existing != token:
                overwrite = input_func(f"{side} already assigned to gripper index {existing}. Overwrite? [y/N]: ")
                if overwrite.strip().lower() not in {"y", "yes"}:
                    continue
            for side_name, side_token in list(tokens.items()):
                if side_token == token:
                    tokens[side_name] = None
            tokens[side] = token
            print(f"Assigned [{token}] {label} -> {side}")
            break
        if all(tokens.get(side) is not None for side in active_sides):
            return tokens
    return tokens


def _known_dmtac_gripper_tokens_from_comparison(
    comparison: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
) -> dict[str, str | None]:
    tokens: dict[str, str | None] = {"left": None, "right": None}
    for item in comparison.get("matched") or []:
        if item.get("kind") != "gripper":
            continue
        side = str(item.get("side"))
        if side not in active_sides:
            continue
        current = item.get("current") or {}
        token = current.get("index")
        if token not in (None, ""):
            tokens[side] = str(token)
    return tokens


def prepare_dmtac(
    options: DevicePreflightOptions,
    result: DevicePreflightResult,
) -> None:
    usb_assignment, gui_config_cls, run_tactile_gui, tactile_tokens_from_assignment = _load_dmtac_modules()
    scan = usb_assignment.scan_usb_devices(probe_grippers=True)
    expected_grippers = len(options.active_sides)
    expected_tactile = 2 * len(options.active_sides) if options.require_dmtac_tactile else 0
    reachable_grippers = _reachable_grippers(scan)
    tactile_devices = list(scan.get("tactile") or [])
    if len(reachable_grippers) < expected_grippers:
        raise DevicePreflightError(
            f"DMTAC gripper count is insufficient: reachable {len(reachable_grippers)}, expected {expected_grippers}."
        )
    if len(tactile_devices) < expected_tactile:
        raise DevicePreflightError(
            f"DMTAC tactile count is insufficient: detected {len(tactile_devices)}, expected {expected_tactile}."
        )

    try:
        assignment = usb_assignment.load_assignment(options.dmtac_config)
        comparison = usb_assignment.compare_assignment_to_scan(assignment, scan)
        text = usb_assignment.format_comparison(comparison)
        if text:
            result.add(text)
        effective = comparison["effective_assignment"]
        if not comparison.get("missing") and _configured_dmtac_complete(
            usb_assignment,
            effective,
            active_sides=options.active_sides,
            require_tactile=options.require_dmtac_tactile,
        ):
            if comparison.get("remapped"):
                saved = usb_assignment.save_assignment(effective)
                result.add(f"DMTAC paths changed; saved remapped latest config: {saved}")
            result.dmtac_assignment = effective
            return
        inferred = _infer_dmtac_assignment_from_partial_config(
            usb_assignment,
            scan,
            assignment,
            comparison,
            active_sides=options.active_sides,
            require_tactile=options.require_dmtac_tactile,
            assume_yes=bool(options.assume_yes),
        )
        if inferred is not None and _configured_dmtac_complete(
            usb_assignment,
            inferred,
            active_sides=options.active_sides,
            require_tactile=options.require_dmtac_tactile,
        ):
            saved = usb_assignment.save_assignment(inferred)
            result.dmtac_assignment = inferred
            result.add(f"DMTAC partial assignment inferred and saved: {saved}")
            return
        result.add("DMTAC latest assignment is incomplete or no longer matches current USB scan.")
    except FileNotFoundError:
        result.add("No latest DMTAC assignment found.")
        assignment = None
        comparison = None

    if not options.auto_configure:
        raise DevicePreflightError("DMTAC assignment is not ready; run `python -m teleop_runtime.configure_devices --configure`.")

    base_assignment = comparison["effective_assignment"] if isinstance(comparison, Mapping) else None
    known_gripper_tokens: dict[str, str | None] = {"left": None, "right": None}
    if isinstance(comparison, Mapping):
        known_gripper_tokens = _known_dmtac_gripper_tokens_from_comparison(
            comparison,
            active_sides=options.active_sides,
        )
    if len(options.active_sides) == 1 and len(reachable_grippers) == 1 and not any(known_gripper_tokens.values()):
        gripper_tokens = {
            "left": None,
            "right": None,
            str(options.active_sides[0]): _dmtac_gripper_token(reachable_grippers[0]),
        }
        result.add(
            "DMTAC single active side has exactly one reachable gripper; "
            f"assigned {options.active_sides[0]} -> {_dmtac_gripper_label(reachable_grippers[0])}."
        )
    elif options.assume_yes:
        gripper_tokens = usb_assignment.build_gripper_auto_assignment_tokens(
            scan,
            swap=False,
            single_side=options.active_sides[0] if len(options.active_sides) == 1 else "right",
            require_reachable=True,
        )
    else:
        gripper_tokens = _motion_confirm_dmtac_gripper_tokens(
            usb_assignment,
            scan,
            active_sides=options.active_sides,
            known_tokens=known_gripper_tokens,
        )
    missing_gripper_sides = [side for side in options.active_sides if gripper_tokens.get(side) is None]
    if missing_gripper_sides:
        raise DevicePreflightError(
            "DMTAC gripper assignment is incomplete for active side(s): "
            + ", ".join(missing_gripper_sides)
        )

    tactile_tokens: dict[str, str | None] = {}
    if options.require_dmtac_tactile:
        tactile_already_complete = bool(
            base_assignment is not None
            and all(
                len(usb_assignment.tactile_ids_for_side(base_assignment, side)) >= 2
                for side in options.active_sides
            )
        )
        if tactile_already_complete:
            result.add("DMTAC tactile assignment kept from existing config.")
        elif not options.dmtac_tactile_gui:
            for side in options.active_sides:
                for slot_index, slot in enumerate(("inner", "outer")):
                    index = (0 if side == "left" else 2) + slot_index
                    if index < len(tactile_devices):
                        tactile_tokens[f"{side}_{slot}"] = str(tactile_devices[index].get("index"))
        else:
            gui_result = run_tactile_gui(
                scan,
                config=gui_config_cls(
                    backend=str(options.dmtac_tactile_backend),
                    max_devices=max(1, expected_tactile),
                    close_devices=False,
                ),
            )
            if gui_result.cancelled:
                raise DevicePreflightError("DMTAC tactile GUI assignment was cancelled.")
            tactile_tokens = tactile_tokens_from_assignment(gui_result.slots)

    selected = usb_assignment.build_assignment_from_tokens(
        scan,
        left_gripper=gripper_tokens.get("left") if "left" in options.active_sides else None,
        right_gripper=gripper_tokens.get("right") if "right" in options.active_sides else None,
        left_inner=tactile_tokens.get("left_inner"),
        left_outer=tactile_tokens.get("left_outer"),
        right_inner=tactile_tokens.get("right_inner"),
        right_outer=tactile_tokens.get("right_outer"),
    )
    assignment = usb_assignment.merge_assignment(base_assignment, selected) if base_assignment is not None else selected
    saved = usb_assignment.save_assignment(assignment)
    result.dmtac_assignment = assignment
    result.add(f"Saved DMTAC assignment: {saved}")


def prepare_cameras(
    options: DevicePreflightOptions,
    result: DevicePreflightResult,
    *,
    discover: Callable[[], list[dict[str, Any]]] = _camera_records_from_context,
) -> None:
    discovered = discover()
    comparison = compare_realsense_config(options.camera_config, discovered)
    if not comparison["missing"]:
        result.camera_config_path = str(options.camera_config)
        result.add(f"RealSense config ok: {len(comparison['matched'])} camera(s) matched.")
        return
    if len(discovered) < len(comparison["configured"]):
        configured = ", ".join(
            f"{item['camera_key']}={item['serial']}" for item in comparison["missing"]
        )
        current = ", ".join(str(item.get("serial")) for item in discovered) or "none"
        raise DevicePreflightError(
            f"RealSense count/config mismatch. Missing configured camera(s): {configured}. Current serials: {current}."
        )
    if not options.auto_configure:
        raise DevicePreflightError("RealSense config does not match current cameras and auto-configure is disabled.")
    path = configure_realsense_interactive(
        options.camera_config,
        discovered,
        assume_yes=options.assume_yes,
    )
    result.camera_config_path = str(path)
    result.add(f"Updated RealSense camera config: {path}")


def prepare_devices_for_teleop(options: DevicePreflightOptions) -> DevicePreflightResult:
    options.active_sides = _active_sides(options.active_sides)
    result = DevicePreflightResult(ok=True)
    _append_serial_conflict_report(result, fix_serial_conflicts=bool(options.fix_serial_conflicts))
    permission_paths = [*_serial_device_paths(), *_video_device_paths(), *_usb_bus_device_paths()]
    result.permission_fixes = repair_device_permissions(permission_paths, apply=bool(options.fix_permissions))
    permission_text = _permission_error_text(result.permission_fixes)
    if permission_text:
        if bool(options.fix_permissions):
            result.ok = False
        result.add(permission_text)

    if options.require_camera:
        prepare_cameras(options, result)
    if options.require_q6:
        prepare_q6(options, result)
    if options.require_dmtac_gripper:
        prepare_dmtac(options, result)
    return result


def prepare_device_permissions(
    *,
    fix_permissions: bool = False,
    fix_serial_conflicts: bool = False,
) -> DevicePreflightResult:
    result = DevicePreflightResult(ok=True)
    _append_serial_conflict_report(result, fix_serial_conflicts=bool(fix_serial_conflicts))
    permission_paths = [*_serial_device_paths(), *_video_device_paths(), *_usb_bus_device_paths()]
    result.permission_fixes = repair_device_permissions(permission_paths, apply=bool(fix_permissions))
    permission_text = _permission_error_text(result.permission_fixes)
    if permission_text:
        if bool(fix_permissions):
            result.ok = False
        result.add(permission_text)
    else:
        result.add("Current USB serial, video/media, and USB bus device permissions are readable/writable.")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DexTeleop USB permissions and optional device assignments.")
    parser.add_argument("--active-sides", default="both", choices=("left", "right", "both"))
    parser.add_argument("--camera-config", default=str(DEFAULT_REALSENSE_DEVICE_CONFIG_PATH))
    parser.add_argument(
        "--check-devices",
        action="store_true",
        help="After the permission pass, also validate/configure Q6, DMTAC, and RealSense assignments.",
    )
    parser.add_argument(
        "--check",
        dest="check_devices",
        action="store_true",
        help="Alias for --check-devices.",
    )
    parser.add_argument(
        "--configure",
        dest="check_devices",
        action="store_true",
        help="Alias for --check-devices; assignment flows may save latest configs.",
    )
    parser.add_argument(
        "--permissions-only",
        dest="check_devices",
        action="store_false",
        help="Only handle device-node permissions. This is the default.",
    )
    parser.set_defaults(check_devices=False)
    parser.add_argument(
        "--check-only",
        dest="auto_configure",
        action="store_false",
        default=True,
        help="When --check-devices is used, validate only and do not enter assignment flows.",
    )
    parser.add_argument("--yes", action="store_true", help="Use default left/right assignment without prompts.")
    parser.add_argument(
        "--fix-permissions",
        action="store_true",
        help="chmod 666 current USB serial, video/media, and USB bus device nodes. Run with sudo.",
    )
    parser.add_argument(
        "--fix-serial-conflicts",
        action="store_true",
        help=(
            "Stop/disable/mask brltty services that can claim CH340/CP210x gripper USB serial adapters. "
            "Run with sudo, then replug the gripper USB/HUB."
        ),
    )
    parser.add_argument("--probe-all-serial", action="store_true")
    parser.add_argument("--no-camera", dest="require_camera", action="store_false", default=True)
    parser.add_argument("--no-q6", dest="require_q6", action="store_false", default=True)
    parser.add_argument("--no-dmtac-gripper", dest="require_dmtac_gripper", action="store_false", default=True)
    parser.add_argument("--no-dmtac-tactile", dest="require_dmtac_tactile", action="store_false", default=True)
    parser.add_argument("--no-dmtac-tactile-gui", dest="dmtac_tactile_gui", action="store_false", default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    check_devices = bool(args.check_devices or "--check-only" in raw_argv)
    if not check_devices:
        result = prepare_device_permissions(
            fix_permissions=bool(args.fix_permissions),
            fix_serial_conflicts=bool(args.fix_serial_conflicts),
        )
        for message in result.messages:
            print(message)
        if not bool(result.ok):
            print("[DevicePermissions] ERROR", file=sys.stderr)
            return 2
        needs_permission_attention = _permission_reports_need_attention(result.permission_fixes)
        needs_serial_attention = _serial_conflict_reports_need_attention(result.serial_conflict_report)
        if needs_permission_attention or needs_serial_attention:
            print("[DevicePermissions] WARNING")
            return 0
        print("[DevicePermissions] OK")
        return 0

    options = DevicePreflightOptions(
        active_sides=_active_sides((args.active_sides,)),
        camera_config=args.camera_config,
        require_camera=bool(args.require_camera),
        require_q6=bool(args.require_q6),
        require_dmtac_gripper=bool(args.require_dmtac_gripper),
        require_dmtac_tactile=bool(args.require_dmtac_tactile),
        auto_configure=bool(args.auto_configure),
        assume_yes=bool(args.yes),
        fix_permissions=bool(args.fix_permissions),
        fix_serial_conflicts=bool(args.fix_serial_conflicts),
        probe_all_serial=bool(args.probe_all_serial),
        dmtac_tactile_gui=bool(args.dmtac_tactile_gui),
    )
    try:
        result = prepare_devices_for_teleop(options)
    except DevicePreflightError as exc:
        print(f"[DevicePreflight] ERROR: {exc}", file=sys.stderr)
        return 2
    for message in result.messages:
        print(message)
    print("[DevicePreflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
