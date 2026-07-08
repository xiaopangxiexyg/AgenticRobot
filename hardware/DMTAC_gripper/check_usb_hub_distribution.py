#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DMTAC_TACTILE_VID_PID = {("0bda", "5846")}
DMTAC_GRIPPER_VID_PID = {("1a86", "7523")}
TACTILE_SERIAL_PREFIX = "M261"
SIDES = ("left", "right")
TACTILE_SLOTS = ("inner", "outer")


@dataclass(frozen=True)
class UsbDevice:
    sys_name: str
    bus: str
    dev: str
    speed: str
    vid: str
    pid: str
    manufacturer: str
    product: str
    serial: str
    root_hub: str
    root_controller: str
    hub_name: str
    hub_product: str
    port_chain: str
    role: str = ""
    assignment: str = ""


def _read_text(path: Path, name: str) -> str:
    try:
        return (path / name).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _sysfs_usb_root() -> Path:
    return Path("/sys/bus/usb/devices")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_assignment_path() -> Path:
    try:
        from portable_dmtac_gripper import usb_assignment

        try:
            return usb_assignment.latest_config_path()
        except FileNotFoundError:
            return usb_assignment.default_config_dir() / usb_assignment.LATEST_CONFIG_NAME
    except Exception:
        return _repo_root() / "teleop_runtime" / "configs" / "hardware" / "dmtac" / "latest.json"


def _device_path(sys_name: str) -> Path:
    return _sysfs_usb_root() / sys_name


def _port_chain(sys_name: str) -> str:
    if "-" not in sys_name:
        return ""
    return sys_name.split("-", 1)[1]


def _parent_hub_name(sys_name: str) -> str:
    if "." in sys_name:
        return sys_name.rsplit(".", 1)[0]
    if "-" in sys_name:
        return "usb" + sys_name.split("-", 1)[0]
    return sys_name


def _root_hub_name(sys_name: str) -> str:
    bus = sys_name.split("-", 1)[0] if "-" in sys_name else ""
    return f"usb{bus}" if bus else ""


def _load_assignment_labels(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments", {})
    labels: dict[str, str] = {}
    if not isinstance(assignments, dict):
        return labels
    for side in SIDES:
        side_payload = assignments.get(side, {})
        if not isinstance(side_payload, dict):
            continue
        tactile = side_payload.get("tactile", {})
        if isinstance(tactile, dict):
            for slot in TACTILE_SLOTS:
                item = tactile.get(slot)
                if isinstance(item, dict):
                    dev_id = str(item.get("dev_id") or "").strip()
                    if dev_id:
                        labels[dev_id] = f"{side}/{slot}"
        gripper = side_payload.get("gripper")
        if isinstance(gripper, dict):
            identity = gripper.get("identity", {})
            location = str(identity.get("location") or "").strip() if isinstance(identity, dict) else ""
            if location:
                labels[f"location:{location}"] = f"{side}/gripper"
    return labels


def _usb_devices(*, assignment_labels: dict[str, str]) -> list[UsbDevice]:
    devices: list[UsbDevice] = []
    root = _sysfs_usb_root()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or ":" in path.name:
            continue
        sys_name = path.name
        vid = _read_text(path, "idVendor").lower()
        pid = _read_text(path, "idProduct").lower()
        if not vid or not pid:
            continue
        root_hub = _root_hub_name(sys_name)
        hub_name = _parent_hub_name(sys_name)
        root_controller = _read_text(_device_path(root_hub), "serial") if root_hub else ""
        hub_product = _read_text(_device_path(hub_name), "product") if hub_name else ""
        serial = _read_text(path, "serial")
        role = ""
        if (vid, pid) in DMTAC_TACTILE_VID_PID or serial.startswith(TACTILE_SERIAL_PREFIX):
            role = "tactile"
        elif (vid, pid) in DMTAC_GRIPPER_VID_PID:
            role = "gripper"
        assignment = assignment_labels.get(serial, "")
        if not assignment:
            assignment = assignment_labels.get(f"location:{sys_name}", "")
        devices.append(
            UsbDevice(
                sys_name=sys_name,
                bus=_read_text(path, "busnum"),
                dev=_read_text(path, "devnum"),
                speed=_read_text(path, "speed"),
                vid=vid,
                pid=pid,
                manufacturer=_read_text(path, "manufacturer"),
                product=_read_text(path, "product"),
                serial=serial,
                root_hub=root_hub,
                root_controller=root_controller,
                hub_name=hub_name,
                hub_product=hub_product,
                port_chain=_port_chain(sys_name),
                role=role,
                assignment=assignment,
            )
        )
    return devices


def _is_video_device(device: UsbDevice) -> bool:
    text = f"{device.manufacturer} {device.product}".lower()
    return any(token in text for token in ("realsense", "n160mu2", "camera"))


def _format_speed(speed: str) -> str:
    if not speed:
        return "?"
    try:
        value = float(speed)
    except ValueError:
        return speed
    return f"{int(value)}M" if value >= 1000 or value.is_integer() else f"{value:g}M"


def _short_device(device: UsbDevice) -> str:
    label = f" {device.assignment}" if device.assignment else ""
    serial = device.serial or "-"
    return (
        f"{serial:<10} {device.sys_name:<8} bus={int(device.bus):03d} "
        f"hub={device.hub_name:<5} speed={_format_speed(device.speed):>5}{label}"
    )


def _build_report(devices: list[UsbDevice], *, include_grippers: bool) -> dict[str, Any]:
    tactile = [item for item in devices if item.role == "tactile"]
    selected = list(tactile)
    if include_grippers:
        selected.extend(item for item in devices if item.role == "gripper")
    root_groups: dict[str, list[UsbDevice]] = defaultdict(list)
    hub_groups: dict[str, list[UsbDevice]] = defaultdict(list)
    for item in tactile:
        root_groups[item.root_hub].append(item)
        hub_groups[item.hub_name].append(item)
    dmtac_root_hubs = set(root_groups)
    neighbors = [
        item
        for item in devices
        if item.root_hub in dmtac_root_hubs and item.role != "tactile" and _is_video_device(item)
    ]
    warnings: list[str] = []
    if not tactile:
        warnings.append("No DMTAC tactile sensors found.")
    if len(root_groups) == 1 and len(tactile) > 2:
        root_name = next(iter(root_groups))
        warnings.append(
            f"All {len(tactile)} tactile sensors share root {root_name}; this is one USB root bus."
        )
    for root_name, items in sorted(root_groups.items()):
        root_speed = items[0].speed if items else ""
        if items and float(root_speed or 0.0) <= 480.0 and len(items) > 2:
            warnings.append(
                f"Root {root_name} is USB2 ({_format_speed(root_speed)}) with {len(items)} tactile sensors."
            )
    for neighbor in neighbors:
        if float(neighbor.speed or 0.0) <= 480.0:
            warnings.append(
                f"USB video neighbor {neighbor.product or neighbor.serial or neighbor.sys_name} "
                f"is also on {neighbor.root_hub} at {_format_speed(neighbor.speed)}."
            )
    return {
        "tactile": [asdict(item) for item in tactile],
        "selected": [asdict(item) for item in selected],
        "root_groups": {
            key: [asdict(item) for item in value]
            for key, value in sorted(root_groups.items())
        },
        "hub_groups": {
            key: [asdict(item) for item in value]
            for key, value in sorted(hub_groups.items())
        },
        "video_neighbors": [asdict(item) for item in neighbors],
        "warnings": warnings,
    }


def _print_report(report: dict[str, Any], *, assignment_path: Path | None) -> None:
    tactile = [UsbDevice(**item) for item in report["tactile"]]
    root_groups = {
        key: [UsbDevice(**item) for item in value]
        for key, value in report["root_groups"].items()
    }
    hub_groups = {
        key: [UsbDevice(**item) for item in value]
        for key, value in report["hub_groups"].items()
    }
    neighbors = [UsbDevice(**item) for item in report["video_neighbors"]]

    print("DMTAC USB HUB distribution")
    if assignment_path is not None:
        status = "loaded" if assignment_path.exists() else "missing"
        print(f"assignment: {assignment_path} ({status})")
    print(f"tactile sensors: {len(tactile)}")
    for item in tactile:
        print("  " + _short_device(item))

    print("\nHub groups:")
    if not hub_groups:
        print("  none")
    for hub_name, items in sorted(hub_groups.items()):
        hub_product = items[0].hub_product or "unknown hub"
        root_hub = items[0].root_hub
        print(f"  {hub_name} ({hub_product}) root={root_hub}: {len(items)} tactile")
        for item in sorted(items, key=lambda dev: dev.port_chain):
            print("    " + _short_device(item))

    print("\nRoot bus groups:")
    if not root_groups:
        print("  none")
    for root_name, items in sorted(root_groups.items()):
        controller = items[0].root_controller or "unknown-controller"
        speed = _format_speed(items[0].speed)
        hubs = ",".join(sorted({item.hub_name for item in items}))
        print(
            f"  {root_name} controller={controller} speed={speed}: "
            f"{len(items)} tactile across hubs [{hubs}]"
        )

    if neighbors:
        print("\nOther USB video devices on DMTAC root bus:")
        for item in neighbors:
            print("  " + _short_device(item) + f" product={item.product or '-'}")

    print("\nVerdict:")
    warnings = list(report["warnings"])
    if warnings:
        for warning in warnings:
            print(f"  WARN {warning}")
    else:
        print("  OK tactile sensors are split across USB root buses.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show DMTAC tactile sensor distribution across USB hubs/root buses."
    )
    parser.add_argument(
        "--assignment",
        type=str,
        default="latest",
        help="DMTAC assignment JSON path, 'latest', or 'none'. Default: latest.",
    )
    parser.add_argument(
        "--include-grippers",
        action="store_true",
        help="Include DMTAC CH340 gripper serial devices in JSON selected output.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def _resolve_assignment(text: str) -> Path | None:
    value = str(text or "").strip()
    if value.lower() in {"", "none", "off", "no"}:
        return None
    if value.lower() == "latest":
        return _default_assignment_path()
    return Path(value).expanduser().resolve()


def main() -> None:
    args = parse_args()
    assignment_path = _resolve_assignment(args.assignment)
    labels = _load_assignment_labels(assignment_path)
    devices = _usb_devices(assignment_labels=labels)
    report = _build_report(devices, include_grippers=bool(args.include_grippers))
    if args.json:
        payload = {
            "assignment_path": None if assignment_path is None else str(assignment_path),
            **report,
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return
    _print_report(report, assignment_path=assignment_path)


if __name__ == "__main__":
    main()
