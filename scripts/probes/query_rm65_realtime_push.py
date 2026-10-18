"""Read existing telemetry config or passively receive it; never configure it."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path


def query_configs(arm_ips, factory):
    results = {}
    for side in ("left", "right"):
        arm = factory(arm_ips[side])
        try:
            if arm.nSocket == -1:
                raise RuntimeError(f"{side} SDK attachment failed")
            code, config = arm.Get_Realtime_Push_Config()
            if code != 0 or not isinstance(config, dict):
                raise RuntimeError(f"{side} telemetry query failed: code={code}")
            results[side] = config
        finally:
            code = arm.Arm_Socket_Close()
            if code not in (None, 0):
                raise RuntimeError(f"{side} SDK close failed: code={code}")
    return results


def listen_existing(
    arm_ips, port, *, socket_factory=socket.socket, clock=time.monotonic
):
    from hardware.rm65.utils.scripts.rm65_udp_actuator_trace import decode_udp_payload

    by_ip = {ip: side for side, ip in arm_ips.items()}
    arrivals = {side: [] for side in arm_ips}
    samples = {}
    unknown_count = 0
    with socket_factory(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        # No REUSEADDR/REUSEPORT: an occupied port stops this probe.
        receiver.bind(("0.0.0.0", port))
        started = clock()
        deadline = started + 10.0
        while True:
            remaining = deadline - clock()
            if remaining <= 0.0:
                break
            receiver.settimeout(min(0.5, remaining))
            try:
                payload, source = receiver.recvfrom(65535)
            except TimeoutError:
                continue
            side = by_ip.get(source[0])
            if side is None:
                unknown_count += 1
                continue
            arrivals[side].append(clock())
            if side not in samples:
                decoded = decode_udp_payload(payload, compact=True, retain_raw=True)
                samples[side] = {
                    "format": decoded["payload_format"],
                    "raw_json": decoded.get("payload_json"),
                }
    return {
        "status": "completed",
        "duration_sec": clock() - started,
        "packet_count_by_side": {
            side: len(values) for side, values in arrivals.items()
        },
        "arrival_hz_by_side": {
            side: (len(values) - 1) / (values[-1] - values[0])
            if len(values) > 1 and values[-1] > values[0]
            else None
            for side, values in arrivals.items()
        },
        "first_packet_by_side": samples,
        "unknown_source_packets": unknown_count,
        "scope": "arrival_only_not_authoritative_feedback_or_motion_validation",
        "sdk_connected": False,
        "controller_config_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm-ip-assignment", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--listen-port", type=int)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    ips = json.loads(args.arm_ip_assignment.read_text())["arm_ip_by_side"]
    if set(ips) != {"left", "right"} or len(set(ips.values())) != 2:
        raise ValueError("Expected distinct explicit left/right controller addresses")
    if args.listen_port is not None and not 1 <= args.listen_port <= 65535:
        raise ValueError("listen-port must be in 1..65535")
    sys.path.insert(0, str(repo))
    if args.check_only:
        print(json.dumps({"status": "passed", "devices_started": False}))
        return 0
    if args.listen_port is not None:
        print(json.dumps(listen_existing(ips, args.listen_port), ensure_ascii=False))
        return 0
    from hardware.rm65.rm_api2_adapter import RM65, Arm
    from scripts.probes.capture_rm65_grasp_snapshot import assert_arm_connections_idle

    assert_arm_connections_idle(ips)
    results = query_configs(ips, lambda ip: Arm(RM65, ip))
    print(
        json.dumps(
            {
                "status": "passed",
                "configs": results,
                "allowed_sdk_calls": ["Get_Realtime_Push_Config", "Arm_Socket_Close"],
                "controller_config_changed": False,
                "motion_commands": [],
                "cleanup_complete": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
