from __future__ import annotations

import argparse
import json
import time
from typing import List

from .core import (
    DMTacMultiDevice,
    DeviceConfig,
    discover_device_ids,
    discover_devices,
    format_frame_summary,
)


def _parse_dev_id(text: str):
    try:
        return int(text)
    except ValueError:
        return text


def _build_configs(args) -> List[DeviceConfig]:
    if args.dev_ids:
        dev_ids = [_parse_dev_id(item) for item in args.dev_ids]
    else:
        dev_ids = discover_device_ids(prefer_serial=True)
        if args.num_devices:
            dev_ids = dev_ids[: args.num_devices]

    if not dev_ids:
        raise SystemExit("No device ids provided or discovered. Use --list to inspect devices.")

    return [
        DeviceConfig(
            dev_id=dev_id,
            backend=args.backend,
            mode=args.mode,
            max_fps=args.max_fps,
            show_fps=args.show_fps,
            cuda_id=args.cuda_id,
            remote_addr=args.remote_addr,
            pc_host=args.pc_host,
            pc_port=args.pc_port,
            enable_raw=not args.no_raw,
            enable_deformation=not args.no_deformation,
            enable_shear=not args.no_shear,
            enable_depth=not args.no_depth,
            enable_force=not args.no_force,
            include_infer=not args.no_infer,
            include_distributed_force=not args.no_distributed_force,
            include_contact_area=not args.no_contact_area,
            include_base_frame=not args.no_base_frame,
            include_events=not args.no_events,
        )
        for dev_id in dev_ids
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Portable DM-Tac reader")
    parser.add_argument("--list", action="store_true", help="List SDK-compliant devices and exit")
    parser.add_argument("--dev_ids", nargs="*", help="Device ids or serials. Omit to auto-discover.")
    parser.add_argument("--num_devices", type=int, default=0, help="When auto-discovering, limit to first N devices")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "Flux", "flux"])
    parser.add_argument("--mode", default="standard", choices=["standard", "high"])
    parser.add_argument("--max_fps", type=int, default=120)
    parser.add_argument("--show_fps", action="store_true")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--remote_addr", default="192.168.127.10:50051")
    parser.add_argument("--pc_host", default="0.0.0.0")
    parser.add_argument("--pc_port", type=int, default=60001)
    parser.add_argument("--count", type=int, default=10, help="Number of read cycles. 0 means duration-based/infinite.")
    parser.add_argument("--duration_s", type=float, default=0.0, help="Stop after duration. 0 disables duration stop.")
    parser.add_argument("--timeout_ms", type=int, default=500)
    parser.add_argument("--ready_timeout_s", type=float, default=10.0)
    parser.add_argument("--poll_sleep_ms", type=float, default=0.0)
    parser.add_argument("--ignore_status", action="store_true", help="Read even if getDevStatus() is nonzero")
    parser.add_argument("--jsonl", action="store_true", help="Print one JSON summary per device per read")
    parser.add_argument("--no_raw", action="store_true")
    parser.add_argument("--no_infer", action="store_true")
    parser.add_argument("--no_deformation", action="store_true")
    parser.add_argument("--no_shear", action="store_true")
    parser.add_argument("--no_depth", action="store_true")
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--no_distributed_force", action="store_true")
    parser.add_argument("--no_contact_area", action="store_true")
    parser.add_argument("--no_base_frame", action="store_true")
    parser.add_argument("--no_events", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(discover_devices(), ensure_ascii=False, indent=2))
        return

    configs = _build_configs(args)
    reader = DMTacMultiDevice(configs)

    try:
        reader.connect()
        ready = reader.wait_until_ready(timeout_s=args.ready_timeout_s)
        for dev_id, is_ready in ready.items():
            print(f"[ready] dev={dev_id} ready={is_ready}")

        start = time.time()
        read_idx = 0
        while True:
            if args.count > 0 and read_idx >= args.count:
                break
            if args.duration_s > 0 and time.time() - start >= args.duration_s:
                break

            frames = reader.read_all(
                timeout_ms=args.timeout_ms,
                require_ready=not args.ignore_status,
            )
            for dev_id, frame in frames.items():
                if args.jsonl:
                    payload = {"read_idx": read_idx, "device_id": dev_id, "frame": None if frame is None else frame.summary()}
                    print(json.dumps(payload, ensure_ascii=False, default=str))
                else:
                    print(f"[{read_idx:06d}] {format_frame_summary(frame)}")

            read_idx += 1
            if args.poll_sleep_ms > 0:
                time.sleep(args.poll_sleep_ms / 1000.0)

    finally:
        reader.close()


if __name__ == "__main__":
    main()
