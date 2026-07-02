from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass

import numpy as np
from triad_openvr import triad_openvr
from triad_openvr.triad_openvr import matrix_to_flat_list


DEFAULT_TRACKERS = (
    "tracker_1:left_wrist",
    "tracker_2:right_wrist",
)


@dataclass(frozen=True)
class TrackerSpec:
    device_name: str
    tracker_role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read VIVE trackers via triad_openvr and send structured UDP JSON "
            "packets to the teleop runtime receiver."
        )
    )
    parser.add_argument("--host", type=str, default="192.168.20.90", help="UDP receiver host")
    parser.add_argument("--port", type=int, default=9999, help="UDP receiver port")
    parser.add_argument(
        "--tracker",
        action="append",
        default=[],
        help=(
            "Tracker mapping in the form 'device_name:tracker_role', "
            "for example 'tracker_1:left_wrist'. Two wrist trackers are sufficient; "
            "an optional third 'tracker_3:auxiliary' may still be specified."
        ),
    )
    parser.add_argument("--hz", type=float, default=60.0, help="Target send rate")
    parser.add_argument("--print-hz", type=float, default=2.0, help="Console status print rate")
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


def flat_to_pose_matrix(flat_pose: list[float]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(flat_pose[:3], dtype=float)
    pose[:3, :3] = np.asarray(flat_pose[3:], dtype=float).reshape(3, 3)
    return pose


def main() -> None:
    args = parse_args()
    tracker_specs = parse_tracker_specs(args.tracker)

    vive = triad_openvr.triad_openvr()
    vive.print_discovered_objects()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    next_print_time = 0.0

    print(
        f"[Start] sending structured VIVE data to {args.host}:{args.port} "
        f"for {[spec.device_name for spec in tracker_specs]}"
    )

    try:
        while True:
            loop_start = time.time()
            pose_snapshot = None
            frame_error = None
            try:
                pose_snapshot = vive.get_pose()
            except Exception as exc:  # triad_openvr/OpenVR can raise backend-specific errors.
                frame_error = str(exc)
            trackers = []

            for spec in tracker_specs:
                tracker_start = time.perf_counter()
                pose_matrix = np.eye(4, dtype=float)
                valid = False
                error_message = None

                try:
                    if frame_error is not None:
                        raise RuntimeError(frame_error)
                    device = vive.devices[spec.device_name]
                    flat_pose = matrix_to_flat_list(
                        device.get_pose_matrix(pose_snapshot)
                    )
                    pose_matrix = flat_to_pose_matrix(flat_pose)
                    valid = bool(np.all(np.isfinite(pose_matrix)))
                except Exception as exc:  # triad_openvr raises varied runtime errors.
                    error_message = str(exc)
                tracker_elapsed_ms = (time.perf_counter() - tracker_start) * 1000.0

                trackers.append(
                    {
                        "tracker_name": spec.device_name,
                        "tracker_role": spec.tracker_role,
                        "valid": valid,
                        "pose_matrix_4x4": pose_matrix.tolist(),
                        "error": error_message,
                        "read_elapsed_ms": tracker_elapsed_ms,
                    }
                )

            packet = {
                "protocol": "teleop_runtime.vive/v1",
                "seq": seq,
                "sender_timestamp": loop_start,
                "trackers": trackers,
            }
            sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))

            if loop_start >= next_print_time:
                for tracker in trackers:
                    position = np.asarray(tracker["pose_matrix_4x4"], dtype=float)[:3, 3]
                    print(
                        f"[{tracker['tracker_name']}:{tracker['tracker_role']}] "
                        f"valid={tracker['valid']} pos={np.array2string(position, precision=3)}"
                    )
                next_print_time = loop_start + (1.0 / max(args.print_hz, 1e-6))

            seq += 1
            remaining = (1.0 / max(args.hz, 1e-6)) - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[Exit] stopped by user")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
