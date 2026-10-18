"""Bounded raw transport diagnostic, not a production RGB-D source."""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import time
from pathlib import Path

FRAME_BYTES = 640 * 480 * 2
FRAME_COUNT = 30
CAPTURE_LIMIT_SEC = 12.0
FORMATS = {"depth": "Z16 ", "color": "YUYV"}


def resolve_nodes(serial: str, sysfs=Path("/sys/class/video4linux"), dev=Path("/dev")) -> dict:
    nodes = {}
    parents = set()
    for entry in sorted(sysfs.glob("video*")):
        physical = (entry / "device").resolve(strict=True)
        usb = next((p for p in (physical, *physical.parents) if (p / "idVendor").is_file()), None)
        if usb is None or not (usb / "serial").is_file() or (usb / "serial").read_text().strip() != serial:
            continue
        if (usb / "idVendor").read_text().strip() != "8086" or (usb / "idProduct").read_text().strip() != "0b5b":
            raise RuntimeError("Selected USB identity is not an Intel D405")
        parents.add(str(usb))
        index = (entry / "index").read_text().strip()
        role = {"0": "depth", "4": "color"}.get(index)
        if role:
            if role in nodes:
                raise RuntimeError(f"Ambiguous {role} endpoint")
            node = dev / entry.name
            if not node.is_char_device():
                raise RuntimeError(f"Not a video character device: {node}")
            nodes[role] = {"device": str(node), "usb_path": str(usb), "sysfs_path": str(physical), "index": index}
    if set(nodes) != set(FORMATS) or len(parents) != 1:
        raise RuntimeError("Could not resolve a unique D405 depth/color pair for the USB serial")
    return nodes


def assert_idle() -> None:
    devices = sorted(str(p) for p in Path("/dev").glob("video*"))
    if not devices:
        raise RuntimeError("No video devices")
    holders = subprocess.run(["/usr/bin/fuser", *devices], capture_output=True, text=True, timeout=3, check=False)
    if holders.returncode != 1 or holders.stdout.strip() or holders.stderr.strip():
        raise RuntimeError(f"Camera ownership not idle: {holders.stdout.strip()} {holders.stderr.strip()}")


def capture_command(device: str, role: str, output: Path) -> list[str]:
    return ["/usr/bin/v4l2-ctl", "--device", device,
            f"--set-fmt-video=width=640,height=480,pixelformat={FORMATS[role]}",
            "--set-parm=15", "--get-fmt-video", "--get-parm", "--verbose",
            "--stream-mmap=4", "--stream-poll", f"--stream-count={FRAME_COUNT}",
            f"--stream-to={output / (role + '.raw')}"]


def stream_result(role: str, output: Path, returncode: int | None) -> dict:
    raw = output / (role + ".raw")
    size = raw.stat().st_size if raw.exists() else 0
    text = (output / (role + ".log")).read_text(errors="replace")
    fmt = re.search(r"Width/Height\s*:\s*640/480", text) is not None
    pixel = f"'{FORMATS[role]}'" in text
    fps = re.search(r"Frames per second\s*:\s*15\.000", text) is not None
    complete = size // FRAME_BYTES
    passed = returncode == 0 and fmt and pixel and fps and size == FRAME_COUNT * FRAME_BYTES
    return {"returncode": returncode, "raw_bytes": size, "complete_frames": complete,
            "trailing_bytes": size % FRAME_BYTES, "format_verified": fmt and pixel and fps,
            "passed": passed, "log": role + ".log", "raw": role + ".raw"}


def stop_children(children: dict, *, errors: list[str] | None = None) -> bool:
    # Only terminate this probe's own capture processes; never target existing PIDs.
    failures = [] if errors is None else errors
    for role, process in children.items():
        try:
            if process.poll() is None:
                process.terminate()
        except OSError as error:
            failures.append(f"{role} TERM: {type(error).__name__}: {error}")
    for role, process in children.items():
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError) as error:
            if isinstance(error, OSError):
                failures.append(f"{role} TERM wait: {type(error).__name__}: {error}")
            try:
                process.kill()
            except OSError as kill_error:
                failures.append(f"{role} KILL: {type(kill_error).__name__}: {kill_error}")
    for role, process in children.items():
        try:
            if process.wait(timeout=1) is None:
                failures.append(f"{role} final wait returned no exit code")
        except (subprocess.TimeoutExpired, OSError) as error:
            failures.append(f"{role} final wait: {type(error).__name__}: {error}")
    return not failures


def capture_pair(nodes: dict, output: Path, report: dict) -> None:
    children, logs = {}, {}
    started = time.monotonic()
    report["cleanup_complete"] = False
    try:
        # Both endpoints of ONE camera must stream together. No other camera is opened.
        for role in ("color", "depth"):
            logs[role] = (output / (role + ".log")).open("xb")
            command = capture_command(nodes[role]["device"], role, output)
            children[role] = subprocess.Popen(command, stdout=logs[role], stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            report.setdefault("processes", {})[role] = {"pid": children[role].pid, "command": command,
                                                       "launched_at": time.time()}
            print(json.dumps({"event": "raw_stream_launched", "role": role,
                              **report["processes"][role]}), flush=True)
        while any(p.poll() is None for p in children.values()):
            if any(p.poll() not in (None, 0) for p in children.values()):
                raise RuntimeError("Raw stream process failed; see per-stream logs")
            if time.monotonic() - started >= CAPTURE_LIMIT_SEC:
                raise TimeoutError("Raw paired capture exceeded 12 seconds; no reopen or retry")
            time.sleep(.05)
    finally:
        report["cleanup_started"] = True
        errors = report["cleanup_errors"] = []
        confirmed = False
        try:
            confirmed = stop_children(children, errors=errors)
        finally:
            for role, log in logs.items():
                try:
                    log.close()
                except OSError as error:
                    errors.append(f"{role} log close: {type(error).__name__}: {error}")
            report["cleanup_complete"] = confirmed and not errors
        report["capture_elapsed_sec"] = time.monotonic() - started
        report["streams"] = {role: stream_result(role, output, process.returncode)
                             for role, process in children.items()}


def interrupted(signum, _frame):
    raise InterruptedError(f"Probe interrupted by signal {signum}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usb-serial", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{12}", args.usb_serial) or not args.output.is_absolute() or args.output == Path("/"):
        parser.error("Expected 12-digit USB serial and absolute new output directory")
    # mkdir refuses existing directories and dangling links without modifying them.
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = {"schema": "agenticrobot.d405_raw_v4l2_pair.v1", "status": "failed",
              "usb_serial": args.usb_serial, "started_at": time.time(), "cleanup_complete": False,
              "requested": {"width": 640, "height": 480, "fps": 15, "frames": FRAME_COUNT},
              "production_rgbd_verified": False, "cleanup_stop_signals": []}

    def on_signal(signum, frame):
        if report.get("cleanup_started", False):
            # Repeated cancellation cannot interrupt owned-child confirmation.
            report["cleanup_stop_signals"].append(signum)
            report["status"] = "failed"
            return
        interrupted(signum, frame)

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert_idle()
        report["nodes"] = resolve_nodes(args.usb_serial)
        print(json.dumps({"event": "raw_pair_start", "nodes": report["nodes"]}), flush=True)
        capture_pair(report["nodes"], args.output, report)
        if (report["cleanup_complete"] and not report["cleanup_stop_signals"]
                and all(s["passed"] for s in report["streams"].values())):
            report["status"] = "raw_pair_passed"
        else:
            raise RuntimeError("Raw paired capture did not meet format/frame-count or cleanup requirements")
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            report["finished_at"] = time.time()
            (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0 if report["status"] == "raw_pair_passed" and not report["cleanup_stop_signals"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
