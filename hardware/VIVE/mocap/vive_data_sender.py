from __future__ import annotations

import argparse
import importlib.util
import json
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRACKERS = (
    "tracker_1:left_wrist",
    "tracker_2:right_wrist",
)

DEFAULT_SEND_BUFFER_BYTES = 1024 * 1024
DEFAULT_SEND_ERROR_BACKOFF_SEC = 0.25
DEFAULT_SEND_ERROR_MAX_BACKOFF_SEC = 2.0
DEFAULT_VR_EVENT_POLL_HZ = 30.0
DEFAULT_DEVICE_REFRESH_SEC = 1.0


@dataclass(frozen=True)
class TrackerSpec:
    device_name: str
    tracker_role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read up to three VIVE trackers via triad_openvr and send structured UDP JSON "
            "packets to the teleop runtime receiver."
        )
    )
    parser.add_argument("--host", type=str, default="192.168.0.2", help="UDP receiver host")
    parser.add_argument("--port", type=int, default=9999, help="UDP receiver port")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional triad_openvr config.json path for stable serial-number device names.",
    )
    parser.add_argument(
        "--tracker",
        action="append",
        default=[],
        help=(
            "Tracker mapping in the form 'device_name:tracker_role', "
            "for example 'tracker_1:left_wrist'. May be specified up to 3 times."
        ),
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Target send rate. Use <= 0 to send as fast as OpenVR polling allows.",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=1.0,
        help="Console status print rate. Use <= 0 to disable status output.",
    )
    parser.add_argument(
        "--timing-slow-ms",
        type=float,
        default=25.0,
        help=(
            "Print per-stage timing in status lines when the sender loop is slower "
            "than this threshold. Use <= 0 to always include it."
        ),
    )
    parser.add_argument(
        "--vr-event-poll-hz",
        type=float,
        default=DEFAULT_VR_EVENT_POLL_HZ,
        help=(
            "Poll OpenVR device activation/deactivation events at this rate while sending. "
            "Use <= 0 to poll every sender frame."
        ),
    )
    parser.add_argument(
        "--device-refresh-sec",
        type=float,
        default=DEFAULT_DEVICE_REFRESH_SEC,
        help=(
            "Seconds between full OpenVR connected-device rescans. This helps recover if a "
            "tracker reconnect event is missed. Use <= 0 to disable full rescans."
        ),
    )
    parser.add_argument(
        "--wait-for-trackers",
        type=float,
        default=0.0,
        help="Seconds to wait for all requested tracker device names before sending.",
    )
    parser.add_argument(
        "--require-trackers",
        action="store_true",
        help="Exit if any requested tracker device name is unavailable at startup.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print discovered VIVE/OpenVR device names and exit.",
    )
    parser.add_argument(
        "--swap-left-right",
        "--revert-left-right",
        dest="swap_left_right",
        action="store_true",
        help="Swap device names assigned to left_wrist and right_wrist.",
    )
    parser.add_argument(
        "--send-buffer-bytes",
        type=int,
        default=DEFAULT_SEND_BUFFER_BYTES,
        help="Requested UDP SO_SNDBUF size in bytes. Use <= 0 to keep the OS default.",
    )
    parser.add_argument(
        "--send-error-backoff-sec",
        type=float,
        default=DEFAULT_SEND_ERROR_BACKOFF_SEC,
        help="Initial sleep after a UDP send error before retrying.",
    )
    parser.add_argument(
        "--send-error-max-backoff-sec",
        type=float,
        default=DEFAULT_SEND_ERROR_MAX_BACKOFF_SEC,
        help="Maximum sleep after repeated UDP send errors.",
    )
    return parser.parse_args()


def parse_tracker_specs(values: list[str]) -> list[TrackerSpec]:
    raw_values = values if values else list(DEFAULT_TRACKERS)
    if len(raw_values) > 3:
        raise ValueError("At most 3 tracker mappings are supported.")

    specs: list[TrackerSpec] = []
    for item in raw_values:
        try:
            device_name, tracker_role = item.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid tracker mapping '{item}'. Expected 'device_name:tracker_role'."
            ) from exc
        device_name = device_name.strip()
        tracker_role = tracker_role.strip()
        if not device_name or not tracker_role:
            raise ValueError(
                f"Invalid tracker mapping '{item}'. Both device_name and tracker_role are required."
            )
        specs.append(TrackerSpec(device_name=device_name, tracker_role=tracker_role))
    return specs


def swap_left_right_trackers(specs: list[TrackerSpec]) -> list[TrackerSpec]:
    left_idx: int | None = None
    right_idx: int | None = None
    for idx, spec in enumerate(specs):
        if spec.tracker_role == "left_wrist":
            left_idx = idx
        elif spec.tracker_role == "right_wrist":
            right_idx = idx

    if left_idx is None or right_idx is None:
        return specs

    swapped = list(specs)
    swapped[left_idx] = TrackerSpec(
        device_name=specs[right_idx].device_name,
        tracker_role=specs[left_idx].tracker_role,
    )
    swapped[right_idx] = TrackerSpec(
        device_name=specs[left_idx].device_name,
        tracker_role=specs[right_idx].tracker_role,
    )
    return swapped


def pose_matrix_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def openvr_pose_to_matrix4x4(pose_mat: object) -> list[list[float]]:
    if pose_mat is None:
        raise ValueError("OpenVR pose is invalid")

    r00 = float(pose_mat[0][0])
    r01 = float(pose_mat[0][1])
    r02 = float(pose_mat[0][2])
    x = float(pose_mat[0][3])
    r10 = float(pose_mat[1][0])
    r11 = float(pose_mat[1][1])
    r12 = float(pose_mat[1][2])
    y = float(pose_mat[1][3])
    r20 = float(pose_mat[2][0])
    r21 = float(pose_mat[2][1])
    r22 = float(pose_mat[2][2])
    z = float(pose_mat[2][3])

    values = (r00, r01, r02, x, r10, r11, r12, y, r20, r21, r22, z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OpenVR pose contains non-finite values")

    return [
        [r00, r01, r02, x],
        [r10, r11, r12, y],
        [r20, r21, r22, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def missing_tracker_names(vive: Any, specs: list[TrackerSpec]) -> list[str]:
    return [spec.device_name for spec in specs if spec.device_name not in vive.devices]


def tracker_mapping_text(specs: list[TrackerSpec]) -> str:
    return ", ".join(f"{spec.device_name}:{spec.tracker_role}" for spec in specs)


def wait_for_requested_trackers(
    vive: Any,
    specs: list[TrackerSpec],
    timeout_sec: float,
) -> list[str]:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    next_status = 0.0
    missing = missing_tracker_names(vive, specs)

    while missing and time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_status:
            print(f"[Wait] missing trackers: {', '.join(missing)}")
            next_status = now + 1.0
        if hasattr(vive, "poll_vr_events"):
            vive.poll_vr_events()
        time.sleep(0.1)
        missing = missing_tracker_names(vive, specs)

    return missing


def update_vive_device_registry(vive: Any, *, refresh: bool = False) -> tuple[int, str | None]:
    changes = 0
    try:
        poll_vr_events = getattr(vive, "poll_vr_events", None)
        if callable(poll_vr_events):
            result = poll_vr_events()
            if isinstance(result, int):
                changes += result
        if refresh:
            refresh_discovered_devices = getattr(vive, "refresh_discovered_devices", None)
            if callable(refresh_discovered_devices):
                result = refresh_discovered_devices()
                if isinstance(result, int):
                    changes += result
    except Exception as exc:  # OpenVR event polling can fail transiently during device churn.
        return changes, f"{type(exc).__name__}: {exc}"
    return changes, None


def log_tracker_presence_changes(
    previous_missing: set[str],
    current_missing: set[str],
    *,
    available_names: list[str],
) -> None:
    recovered = sorted(previous_missing - current_missing)
    lost = sorted(current_missing - previous_missing)
    if recovered:
        print(f"[Info] tracker device recovered: {', '.join(recovered)}")
    if lost:
        print(
            f"[Warn] tracker device missing: {', '.join(lost)}; "
            f"available: {', '.join(available_names) or '<none>'}"
        )


def format_position(pose_matrix: list[list[float]]) -> str:
    return (
        f"[{pose_matrix[0][3]: .3f} "
        f"{pose_matrix[1][3]: .3f} "
        f"{pose_matrix[2][3]: .3f}]"
    )


def load_triad_openvr() -> Any:
    local_module_path = Path(__file__).resolve().parents[1] / "triad_openvr.py"
    if local_module_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "dexteleop_triad_openvr",
            local_module_path,
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                missing = exc.name or "openvr"
                raise SystemExit(
                    f"[Error] failed to import '{missing}'. Install the VIVE sender dependencies "
                    "in this Python environment, then retry."
                ) from exc
            return module

    try:
        from triad_openvr import triad_openvr
    except ModuleNotFoundError as exc:
        if exc.name != "triad_openvr":
            missing = exc.name or "openvr"
            raise SystemExit(
                f"[Error] failed to import '{missing}'. Install the VIVE sender dependencies "
                "in this Python environment, then retry."
            ) from exc
        try:
            from hardware.VIVE import triad_openvr
        except ModuleNotFoundError as fallback_exc:
            missing = fallback_exc.name or "openvr"
            raise SystemExit(
                f"[Error] failed to import '{missing}'. Install the VIVE sender dependencies "
                "in this Python environment, then retry."
            ) from fallback_exc
    return triad_openvr


def send_error_backoff_seconds(streak: int, base_sec: float, max_sec: float) -> float:
    base = max(0.0, float(base_sec))
    if base <= 0.0 or streak <= 0:
        return 0.0
    cap = max(base, float(max_sec))
    multiplier = 2 ** min(max(0, int(streak) - 1), 4)
    return min(cap, base * multiplier)


def open_udp_sender_socket(
    host: str,
    port: int,
    *,
    send_buffer_bytes: int = DEFAULT_SEND_BUFFER_BYTES,
) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if int(send_buffer_bytes) > 0:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(send_buffer_bytes))
        except OSError as exc:
            print(f"[Warn] failed to set SO_SNDBUF={int(send_buffer_bytes)}: {exc}")
    try:
        sock.connect((host, int(port)))
    except OSError:
        close_udp_sender_socket(sock)
        raise
    return sock


def close_udp_sender_socket(sock: socket.socket | None) -> None:
    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    tracker_specs = parse_tracker_specs(args.tracker)
    if args.swap_left_right:
        tracker_specs = swap_left_right_trackers(tracker_specs)

    triad_openvr_module = load_triad_openvr()
    vive = (
        triad_openvr_module.triad_openvr(args.config)
        if args.config
        else triad_openvr_module.triad_openvr()
    )
    vive.print_discovered_objects()

    if args.list_devices:
        return

    if args.wait_for_trackers > 0.0:
        missing = wait_for_requested_trackers(vive, tracker_specs, args.wait_for_trackers)
    else:
        missing = missing_tracker_names(vive, tracker_specs)

    if missing:
        message = (
            f"requested tracker device(s) not found: {', '.join(missing)}; "
            f"available: {', '.join(sorted(vive.devices)) or '<none>'}"
        )
        if args.require_trackers:
            raise SystemExit(f"[Error] {message}")
        print(f"[Warn] {message}")

    try:
        sock: socket.socket | None = open_udp_sender_socket(
            args.host,
            args.port,
            send_buffer_bytes=int(args.send_buffer_bytes),
        )
    except OSError as exc:
        print(
            f"[Warn] failed to open UDP sender socket for {args.host}:{args.port}; "
            f"will retry in the send loop: {type(exc).__name__}: {exc}"
        )
        sock = None

    seq = 0
    period_sec = 1.0 / args.hz if args.hz > 0.0 else 0.0
    print_interval_sec = 1.0 / args.print_hz if args.print_hz > 0.0 else 0.0
    next_send_time = time.perf_counter()
    next_print_time = next_send_time + print_interval_sec
    last_print_time = next_send_time
    frames_since_print = 0
    timing_slow_ms = float(args.timing_slow_ms)
    send_error_total = 0
    send_error_streak = 0
    next_send_error_log_time = 0.0
    vr_event_poll_interval_sec = (
        1.0 / float(args.vr_event_poll_hz)
        if float(args.vr_event_poll_hz) > 0.0
        else 0.0
    )
    device_refresh_sec = max(0.0, float(args.device_refresh_sec))
    next_vr_event_poll_time = next_send_time
    next_device_refresh_time = next_send_time + device_refresh_sec
    last_missing_trackers = set(missing_tracker_names(vive, tracker_specs))
    next_registry_error_log_time = 0.0

    print(
        f"[Start] sending structured VIVE data to {args.host}:{args.port} "
        f"at {'max rate' if period_sec <= 0.0 else f'{args.hz:g} Hz'}"
    )
    print(
        f"[Start] tracker mapping: {tracker_mapping_text(tracker_specs)}"
    )

    try:
        while True:
            if period_sec > 0.0:
                sleep_sec = next_send_time - time.perf_counter()
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)

            loop_start = time.perf_counter()
            sender_timestamp = time.time()
            registry_elapsed_ms = 0.0
            registry_error = None
            registry_changes = 0
            should_poll_events = (
                vr_event_poll_interval_sec <= 0.0
                or loop_start >= next_vr_event_poll_time
            )
            should_refresh_devices = (
                device_refresh_sec > 0.0 and loop_start >= next_device_refresh_time
            )
            if should_poll_events or should_refresh_devices:
                registry_start = time.perf_counter()
                registry_changes, registry_error = update_vive_device_registry(
                    vive,
                    refresh=should_refresh_devices,
                )
                registry_elapsed_ms = (time.perf_counter() - registry_start) * 1000.0
                if vr_event_poll_interval_sec > 0.0:
                    next_vr_event_poll_time = loop_start + vr_event_poll_interval_sec
                if should_refresh_devices:
                    next_device_refresh_time = loop_start + device_refresh_sec

                current_missing = set(missing_tracker_names(vive, tracker_specs))
                if current_missing != last_missing_trackers:
                    log_tracker_presence_changes(
                        last_missing_trackers,
                        current_missing,
                        available_names=sorted(vive.devices),
                    )
                    last_missing_trackers = current_missing
                if registry_changes:
                    print(
                        f"[Info] OpenVR device registry updated changes={registry_changes} "
                        f"available={', '.join(sorted(vive.devices)) or '<none>'}"
                    )
                if registry_error is not None:
                    now = time.perf_counter()
                    if now >= next_registry_error_log_time:
                        print(f"[Warn] OpenVR device registry update failed: {registry_error}")
                        next_registry_error_log_time = now + 1.0

            get_pose_start = time.perf_counter()
            try:
                pose_snapshot = vive.get_pose()
                frame_error = None
            except Exception as exc:  # triad_openvr/OpenVR can raise backend-specific errors.
                pose_snapshot = None
                frame_error = str(exc)
            get_pose_elapsed_ms = (time.perf_counter() - get_pose_start) * 1000.0

            trackers = []
            tracker_timings: list[tuple[str, str, float]] = []
            for spec in tracker_specs:
                tracker_start = time.perf_counter()
                pose_matrix = pose_matrix_identity()
                valid = False
                error_message = None

                try:
                    if frame_error is not None:
                        raise RuntimeError(frame_error)
                    device = vive.devices[spec.device_name]
                    pose_matrix = openvr_pose_to_matrix4x4(
                        device.get_pose_matrix(pose_snapshot)
                    )
                    valid = True
                except KeyError:
                    error_message = f"device '{spec.device_name}' not found"
                except Exception as exc:  # triad_openvr raises varied runtime errors.
                    error_message = str(exc)

                tracker_elapsed_ms = (time.perf_counter() - tracker_start) * 1000.0
                tracker_timings.append(
                    (spec.device_name, spec.tracker_role, tracker_elapsed_ms)
                )
                trackers.append(
                    {
                        "tracker_name": spec.device_name,
                        "tracker_role": spec.tracker_role,
                        "valid": valid,
                        "pose_matrix_4x4": pose_matrix,
                        "error": error_message,
                        "read_elapsed_ms": tracker_elapsed_ms,
                    }
                )

            packet = {
                "protocol": "teleop_runtime.vive/v1",
                "seq": seq,
                "sender_timestamp": sender_timestamp,
                "trackers": trackers,
            }
            json_start = time.perf_counter()
            payload = json.dumps(
                packet,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            json_elapsed_ms = (time.perf_counter() - json_start) * 1000.0
            send_start = time.perf_counter()
            try:
                if sock is None:
                    sock = open_udp_sender_socket(
                        args.host,
                        args.port,
                        send_buffer_bytes=int(args.send_buffer_bytes),
                    )
                sock.send(payload)
            except OSError as exc:
                send_elapsed_ms = (time.perf_counter() - send_start) * 1000.0
                send_error_total += 1
                send_error_streak += 1
                close_udp_sender_socket(sock)
                sock = None
                backoff_sec = send_error_backoff_seconds(
                    send_error_streak,
                    float(args.send_error_backoff_sec),
                    float(args.send_error_max_backoff_sec),
                )
                now = time.perf_counter()
                if now >= next_send_error_log_time:
                    valid_count = sum(1 for tracker in trackers if tracker["valid"])
                    print(
                        f"[Warn] UDP send failed count={send_error_total} "
                        f"streak={send_error_streak} elapsed={send_elapsed_ms:.3f}ms "
                        f"backoff={backoff_sec:.2f}s valid={valid_count}/{len(trackers)} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    next_send_error_log_time = now + 1.0
                if backoff_sec > 0.0:
                    time.sleep(backoff_sec)
                if period_sec > 0.0:
                    next_send_time = time.perf_counter() + period_sec
                continue
            send_elapsed_ms = (time.perf_counter() - send_start) * 1000.0
            if send_error_streak:
                print(
                    f"[Info] UDP send recovered after {send_error_streak} "
                    f"consecutive error(s)."
                )
                send_error_streak = 0

            seq += 1
            frames_since_print += 1
            loop_elapsed_ms = (time.perf_counter() - loop_start) * 1000.0

            now = time.perf_counter()
            if print_interval_sec > 0.0 and now >= next_print_time:
                elapsed = max(now - last_print_time, 1e-9)
                actual_hz = frames_since_print / elapsed
                valid_count = sum(1 for tracker in trackers if tracker["valid"])
                tracker_total_ms = sum(elapsed_ms for _, _, elapsed_ms in tracker_timings)
                tracker_max_ms = max(
                    (elapsed_ms for _, _, elapsed_ms in tracker_timings),
                    default=0.0,
                )
                timing_suffix = ""
                if timing_slow_ms <= 0.0 or loop_elapsed_ms >= timing_slow_ms:
                    timing_suffix = (
                        f" registry={registry_elapsed_ms:.2f}ms "
                        f" get_pose={get_pose_elapsed_ms:.2f}ms "
                        f"trackers_total={tracker_total_ms:.2f}ms "
                        f"trackers_max={tracker_max_ms:.2f}ms "
                        f"json={json_elapsed_ms:.2f}ms"
                    )
                if registry_error is not None:
                    timing_suffix += f" registry_error={registry_error}"
                print(
                    f"[Status] seq={seq} rate={actual_hz:.1f}Hz "
                    f"loop={loop_elapsed_ms:.2f}ms send={send_elapsed_ms:.3f}ms "
                    f"valid={valid_count}/{len(trackers)} bytes={len(payload)} "
                    f"send_errors={send_error_total}"
                    f"{timing_suffix}"
                )
                for tracker in trackers:
                    error = tracker["error"]
                    suffix = "" if error is None else f" error={error}"
                    print(
                        f"[{tracker['tracker_name']}:{tracker['tracker_role']}] "
                        f"valid={tracker['valid']} "
                        f"read={float(tracker.get('read_elapsed_ms', 0.0)):.2f}ms "
                        f"pos={format_position(tracker['pose_matrix_4x4'])}{suffix}"
                    )
                next_print_time = now + print_interval_sec
                last_print_time = now
                frames_since_print = 0

            if period_sec > 0.0:
                next_send_time += period_sec
                now = time.perf_counter()
                if next_send_time < now:
                    next_send_time = now
    except KeyboardInterrupt:
        print("\n[Exit] stopped by user")
    finally:
        close_udp_sender_socket(sock)


if __name__ == "__main__":
    main()
