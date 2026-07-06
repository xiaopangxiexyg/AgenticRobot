#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import _bootstrap  # noqa: F401


def _parse_dev_id(text: str) -> int | str:
    try:
        return int(text)
    except ValueError:
        return text


@contextlib.contextmanager
def _suppress_fd_output(enabled: bool):
    if not enabled:
        yield
        return
    stdout_fd = None
    stderr_fd = None
    devnull = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        devnull = open(os.devnull, "w")
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    except OSError:
        pass
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if stdout_fd is not None:
            try:
                os.dup2(stdout_fd, 1)
            finally:
                os.close(stdout_fd)
        if stderr_fd is not None:
            try:
                os.dup2(stderr_fd, 2)
            finally:
                os.close(stderr_fd)
        if devnull is not None:
            devnull.close()


def _video_index_from_node(node: str) -> int | None:
    name = Path(node).name
    if not name.startswith("video"):
        return None
    try:
        return int(name[5:])
    except ValueError:
        return None


def _video_occupants(node: str) -> list[str]:
    out = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir() if proc_root.exists() else []:
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for fd_path in fds:
            try:
                if os.readlink(fd_path) != node:
                    continue
                comm = ""
                try:
                    comm = (entry / "comm").read_text(errors="ignore").strip()
                except OSError:
                    pass
                label = f"pid={entry.name}"
                if comm:
                    label += f":{comm}"
                try:
                    raw_cmdline = (entry / "cmdline").read_bytes()
                    cmdline = raw_cmdline.replace(b"\0", b" ").decode(errors="ignore").strip()
                except OSError:
                    cmdline = ""
                if cmdline:
                    label += f" cmd={cmdline[:160]}"
                out.append(label)
                break
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
    return out


def _probe_one_video(node: str, *, quiet: bool) -> dict[str, Any]:
    index = _video_index_from_node(node)
    result: dict[str, Any] = {
        "device_node": node,
        "index": index,
        "exists": os.path.exists(node),
        "occupants": _video_occupants(node),
    }
    if index is None or not result["exists"]:
        return result
    try:
        import cv2  # type: ignore  # noqa: PLC0415
    except Exception as exc:
        result["cv2_error"] = repr(exc)
        return result
    for mode, target in (("index", index), ("path", node)):
        cap = None
        try:
            with _suppress_fd_output(quiet):
                cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
                opened = bool(cap.isOpened())
                ok = False
                shape = None
                if opened:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        shape = tuple(int(value) for value in frame.shape)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if opened else None
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if opened else None
                fps = cap.get(cv2.CAP_PROP_FPS) if opened else None
            result[f"cv2_{mode}"] = {
                "opened": opened,
                "read": bool(ok),
                "shape": shape,
                "width": width,
                "height": height,
                "fps": fps,
            }
        except Exception as exc:
            result[f"cv2_{mode}"] = {"error": repr(exc)}
        finally:
            if cap is not None:
                cap.release()
    return result


def _probe_video_candidates(candidates: list[dict[str, Any]], *, quiet: bool) -> list[dict[str, Any]]:
    seen = set()
    probes = []
    for item in candidates:
        node = str(item.get("device_node") or "")
        if not node or node in seen:
            continue
        seen.add(node)
        probe = dict(item)
        probe.update(_probe_one_video(node, quiet=quiet))
        probes.append(probe)
    return probes


def _device_label(dev_id: Any, serial_by_index: dict[int, str], index_by_serial: dict[str, int]) -> str:
    if isinstance(dev_id, int):
        serial = serial_by_index.get(dev_id)
        return f"{dev_id}/{serial}" if serial else str(dev_id)
    index = index_by_serial.get(str(dev_id))
    return f"{dev_id}@{index}" if index is not None else str(dev_id)


def _shape(value: Any) -> str:
    if value is None:
        return "-"
    if hasattr(value, "img"):
        value = getattr(value, "img", None)
        if value is None:
            return "-"
    return "x".join(str(item) for item in getattr(value, "shape", ())) or type(value).__name__


def _field_names(frame: Any) -> list[str]:
    if frame is None:
        return []
    fields = []
    for name in ("raw", "infer", "deformation2d", "shear", "depth", "force", "distributed_force"):
        if getattr(frame, name, None) is not None:
            fields.append(name)
    if getattr(frame, "contact_area", None) is not None:
        fields.append("contact_area")
    if getattr(frame, "events", None) is not None:
        fields.append("events")
    return fields


def _summarize_frame(dev_id: Any, frame: Any, read_ms: float) -> str:
    if frame is None:
        return f"dev={dev_id} frame=None read={read_ms:.1f}ms"
    fields = _field_names(frame)
    dist = _shape(getattr(frame, "distributed_force", None))
    raw = _shape(getattr(frame, "raw", None))
    errors = getattr(frame, "errors", {}) or {}
    timing = getattr(frame, "timings_ms", {}) or {}
    timing_text = ",".join(f"{key}:{value:.1f}" for key, value in sorted(timing.items()))
    return (
        f"dev={dev_id} status={getattr(frame, 'status', '-')}"
        f" fid={getattr(frame, 'fid', '-')}"
        f" fields={fields or '-'}"
        f" dist={dist} raw={raw}"
        f" read={read_ms:.1f}ms"
        f" sdk_ms={timing_text or '-'}"
        f" errors={list(errors) or '-'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small DM-Tac tactile sensor smoke test.")
    parser.add_argument("--dev-ids", nargs="*", help="Device serials or SDK indices. Omit to auto-discover.")
    parser.add_argument("--num-devices", type=int, default=0, help="Limit auto-discovered devices.")
    parser.add_argument("--use-serial-ids", action="store_true", help="Auto-discover serial ids instead of SDK indices.")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=300)
    parser.add_argument("--ready-timeout-s", type=float, default=5.0)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "Flux", "flux"])
    parser.add_argument("--mode", default="standard", choices=["standard", "high"])
    parser.add_argument("--max-fps", type=int, default=30)
    parser.add_argument("--fields", nargs="*", default=["raw"])
    parser.add_argument("--raw", action="store_true", help="Also request raw camera images.")
    parser.add_argument("--ignore-status", action="store_true", help="Try getters even when getDevStatus() is nonzero.")
    parser.add_argument("--skip-ready", action="store_true", help="Skip wait_until_ready and go directly to read attempts.")
    parser.add_argument("--sequential", action="store_true", help="Read devices one by one instead of in parallel.")
    parser.add_argument("--probe-video", action="store_true", help="Probe /dev/video nodes with OpenCV before SDK connect.")
    parser.add_argument("--diagnose-only", action="store_true", help="Run discovery/video probes and exit before SDK connect.")
    parser.add_argument("--show-sdk-log", action="store_true", help="Do not suppress SDK/OpenCV stdout/stderr.")
    args = parser.parse_args()

    os.environ["DMTAC_SUPPRESS_SDK_OUTPUT"] = "0" if args.show_sdk_log else "1"

    from portable_dm_tac import DMTacMultiDevice, discover_device_candidates, discover_devices
    from portable_dmtac_gripper.device import TactileConfig, tactile_device_config

    print("[dmtac] sdk_log_suppressed=", not args.show_sdk_log, sep="")
    candidates = discover_device_candidates()
    if candidates:
        print("[udev] candidates:")
        print(json.dumps(candidates, ensure_ascii=False, indent=2, default=str))
    if args.probe_video or args.diagnose_only:
        probes = _probe_video_candidates(
            [item for item in candidates if isinstance(item, dict)],
            quiet=not args.show_sdk_log,
        )
        print("[video] opencv_probe:")
        print(json.dumps(probes, ensure_ascii=False, indent=2, default=str))

    discovered = discover_devices()
    print("[sdk] discovered:")
    print(json.dumps(discovered, ensure_ascii=False, indent=2, default=str))

    records = [item for item in discovered if isinstance(item, dict)]
    if args.num_devices > 0:
        records = records[: args.num_devices]
    serial_by_index = {
        int(item["device_index"]): str(item["serial"])
        for item in records
        if item.get("device_index") is not None and item.get("serial")
    }
    index_by_serial = {serial: index for index, serial in serial_by_index.items()}

    if args.dev_ids:
        dev_ids = [_parse_dev_id(item) for item in args.dev_ids]
    elif args.use_serial_ids:
        dev_ids = [str(item["serial"]) for item in records if item.get("serial")]
    else:
        dev_ids = [int(item["device_index"]) for item in records if item.get("device_index") is not None]
    if args.num_devices > 0:
        dev_ids = dev_ids[: args.num_devices]
    if not dev_ids:
        raise SystemExit("No DM-Tac sensors discovered. Check USB/video nodes or pass --dev-ids.")
    labels = {dev_id: _device_label(dev_id, serial_by_index, index_by_serial) for dev_id in dev_ids}
    print("[dmtac] using_dev_ids:", ", ".join(labels.values()))
    if args.diagnose_only:
        return

    fields = list(args.fields)
    if args.raw and "raw" not in fields:
        fields.append("raw")
    tactile_cfg = TactileConfig(
        dev_ids=dev_ids,
        backend=args.backend,
        mode=args.mode,
        max_fps=args.max_fps,
        fields=fields,
        compute_profile="dist-only",
        denoise_distributed_force=False,
    )
    configs = [tactile_device_config(dev_id, tactile_cfg) for dev_id in dev_ids]
    reader = DMTacMultiDevice(configs)

    good_counts = {labels[dev_id]: 0 for dev_id in dev_ids}
    try:
        reader.connect()
        if args.skip_ready:
            print("[ready] skipped")
        else:
            ready = reader.wait_until_ready(timeout_s=args.ready_timeout_s)
            print("[ready]", json.dumps(ready, ensure_ascii=False, default=str))

        for idx in range(args.count):
            started = time.perf_counter()
            frames = reader.read_all(
                timeout_ms=args.timeout_ms,
                wait=False,
                require_ready=not args.ignore_status,
                parallel=not args.sequential,
            )
            read_ms = (time.perf_counter() - started) * 1000.0
            for dev_id in dev_ids:
                frame = frames.get(dev_id)
                if _field_names(frame):
                    good_counts[labels[dev_id]] += 1
                print(f"[{idx:04d}] {_summarize_frame(labels[dev_id], frame, read_ms)}")
            time.sleep(0.02)
    finally:
        reader.close()

    print("[summary]", json.dumps({"valid_frames": good_counts, "count": args.count}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
