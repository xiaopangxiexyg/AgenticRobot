from __future__ import annotations

import argparse
import os
import json
import sys
import time

from portable_dm_tac import discover_devices

from .device import (
    DMTacGripperDevice,
    GripperConfig,
    TactileConfig,
    format_snapshot_summary,
    normalize_tactile_fields,
)
from .gripper_driver import InspireGripper, list_serial_devices
from .tactile_filter import TactileDenoiseConfig


def _parse_dev_id(text: str):
    try:
        return int(text)
    except ValueError:
        return text


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--id", type=int, default=1, dest="gripper_id")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.03)
    parser.add_argument("--max_open_mm", type=float, default=70.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tactile_ids", nargs="*", default=None)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "Flux", "flux"])
    parser.add_argument("--mode", default="standard", choices=["standard", "high"])
    parser.add_argument("--max_fps", type=int, default=120)
    parser.add_argument("--tactile_fields", nargs="*", default=["distributed_force"])
    parser.add_argument("--json", action="store_true")


def _gripper_config(args) -> GripperConfig:
    return GripperConfig(
        port=args.port,
        gripper_id=args.gripper_id,
        baudrate=args.baudrate,
        timeout=args.timeout,
        max_open_mm=args.max_open_mm,
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Unified Inspire gripper + DM-Tac tactile CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-ports")
    sub.add_parser("list-tactile")

    p = sub.add_parser("verify-gripper")
    _add_common(p)

    p = sub.add_parser("verify-tactile")
    _add_common(p)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--timeout_ms", type=int, default=500)

    p = sub.add_parser("read")
    _add_common(p)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--period_s", type=float, default=0.0)
    p.add_argument("--timeout_ms", type=int, default=100)
    p.add_argument("--no_gripper", action="store_true")
    p.add_argument("--no_tactile", action="store_true")

    p = sub.add_parser("visualize-force")
    _add_common(p)
    p.add_argument("--count", type=int, default=0, help="0 means run until q/ESC")
    p.add_argument("--timeout_ms", type=int, default=100)
    p.add_argument("--max_hz", type=float, default=0.0, help="0 means no extra sleep")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--arrow_scale", type=float, default=18.0)
    p.add_argument("--min_arrow_norm", type=float, default=0.015)
    p.add_argument("--deadband_abs", type=float, default=5e-5)
    p.add_argument("--temporal_alpha", type=float, default=0.35)
    p.add_argument("--spatial_blur_ksize", type=int, default=5)
    p.add_argument("--arrow_pool", type=int, default=16)
    p.add_argument("--viz_mode", default="delta", choices=["delta", "absolute", "frame_delta"])
    p.add_argument("--baseline_frames", type=int, default=8)
    p.add_argument("--fixed_ref", type=float, default=None)
    p.add_argument("--norm_percentile", type=float, default=99.0)
    p.add_argument("--reset_tactile", action="store_true")
    p.add_argument("--warmup_frames", type=int, default=0)
    p.add_argument("--window", default="DM-Tac distributed_force XY")

    p = sub.add_parser("probe-force")
    _add_common(p)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--timeout_ms", type=int, default=100)
    p.add_argument("--period_s", type=float, default=0.0)
    p.add_argument("--reset_tactile", action="store_true")
    p.add_argument("--warmup_frames", type=int, default=0)
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--denoise_mode", default="delta", choices=["delta", "absolute", "frame_delta"])
    p.add_argument("--baseline_frames", type=int, default=20)
    p.add_argument("--deadband_abs", type=float, default=5e-5)
    p.add_argument("--temporal_alpha", type=float, default=0.35)
    p.add_argument("--spatial_blur_ksize", type=int, default=5)

    p = sub.add_parser("reset-tactile")
    _add_common(p)

    p = sub.add_parser("set-speed")
    _add_common(p)
    p.add_argument("speed", type=float)

    p = sub.add_parser("set-torque")
    _add_common(p)
    p.add_argument("torque", type=float)

    p = sub.add_parser("open")
    _add_common(p)
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("close")
    _add_common(p)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--torque", type=float, default=None)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("move")
    _add_common(p)
    p.add_argument("joint_pos", type=float)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--torque", type=float, default=None)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("stop")
    _add_common(p)

    args = parser.parse_args(argv)

    if args.command == "list-ports":
        _print_json(list_serial_devices())
        return

    if args.command == "list-tactile":
        _print_json(discover_devices())
        return

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
        tactile_config.denoise_distributed_force = not args.no_denoise
        tactile_config.denoise_config = TactileDenoiseConfig(
            enabled=not args.no_denoise,
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
                args.joint_pos,
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
