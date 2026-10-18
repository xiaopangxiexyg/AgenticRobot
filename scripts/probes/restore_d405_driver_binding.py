"""Explicit, identity-scoped uvcvideo binding restoration; no device reset."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

USB_ROOT = Path("/sys/bus/usb/devices")
BIND_FILE = Path("/sys/bus/usb/drivers/uvcvideo/bind")
VIDEO_ROOT = Path("/sys/class/video4linux")


def inspect_target(port: str, serial: str) -> dict:
    if not re.fullmatch(r"[1-9]\d*-[1-9]\d*(?:\.[1-9]\d*)*", port) or not re.fullmatch(r"\d{12}", serial):
        raise ValueError("Invalid USB port or USB serial")
    device = USB_ROOT / port
    if any((device / key).read_text().strip() != value
           for key, value in (("serial", serial), ("idVendor", "8086"), ("idProduct", "0b5b"))):
        raise RuntimeError("USB identity does not match the selected Intel D405")
    interfaces = {}
    controls = []
    for interface in sorted(USB_ROOT.glob(port + ":*")):
        if (interface / "bInterfaceClass").read_text().strip() != "0e":
            raise RuntimeError("Unexpected non-video interface on selected D405")
        if (interface / "bInterfaceSubClass").read_text().strip() == "01":
            controls.append(interface.name)
        driver = interface / "driver"
        bound = driver.resolve().name if driver.exists() else None
        if bound not in (None, "uvcvideo"):
            raise RuntimeError(f"Interface belongs to another driver: {interface.name}: {bound}")
        interfaces[interface.name] = bound
    if len(controls) != 1 or controls[0] != port + ":1.0":
        raise RuntimeError("Expected one D405 VideoControl interface; refusing ambiguous binding")
    videos = ["/dev/" + entry.name for entry in sorted(VIDEO_ROOT.glob("video*"))
              if (USB_ROOT / controls[0]).resolve() in entry.resolve().parents]
    bus, number = int((device / "busnum").read_text()), int((device / "devnum").read_text())
    return {"usb_port": port, "usb_serial": serial, "interfaces": interfaces,
            "control": controls[0], "video_nodes": videos,
            "usb_node": f"/dev/bus/usb/{bus:03d}/{number:03d}"}


def restore(port: str, serial: str) -> dict:
    before = inspect_target(port, serial)
    print(json.dumps({"event": "before_bind", **before}), flush=True)
    if all(driver == "uvcvideo" for driver in before["interfaces"].values()):
        return {"status": "already_bound", "before": before, "after": before}
    if before["interfaces"][before["control"]] is not None:
        raise RuntimeError("Partially bound camera; do not unbind or manipulate streaming interfaces")
    if os.geteuid() != 0:
        raise PermissionError("Driver binding requires root; use the scoped sudo -n wrapper")
    holders = subprocess.run(["/usr/bin/fuser", before["usb_node"], *before["video_nodes"]],
                             capture_output=True, text=True, timeout=3, check=False)
    if holders.returncode != 1 or holders.stdout.strip() or holders.stderr.strip():
        raise RuntimeError(f"Selected camera is occupied or ownership is unknown: {holders.stdout} {holders.stderr}")
    if inspect_target(port, serial) != before:
        raise RuntimeError("USB identity/binding changed during preflight")
    # Bind the VideoControl interface once. uvcvideo claims its associated streams.
    with BIND_FILE.open("w") as binding:
        binding.write(before["control"])
    deadline = time.monotonic() + 3
    while True:
        after = inspect_target(port, serial)
        if all(driver == "uvcvideo" for driver in after["interfaces"].values()) and after["video_nodes"]:
            return {"status": "restored", "before": before, "after": after}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Binding did not restore all interfaces/video nodes: {after}")
        time.sleep(.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usb-port", required=True)
    parser.add_argument("--usb-serial", required=True)
    parser.add_argument("--bind", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = restore(args.usb_port, args.usb_serial) if args.bind else {
            "status": "observed", **inspect_target(args.usb_port, args.usb_serial)}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "message": str(exc)}), flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
