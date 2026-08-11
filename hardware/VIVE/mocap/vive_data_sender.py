"""Portable VIVE pose sender.

This file is intentionally self-contained so it can be copied by itself to a
Windows SteamVR host.  Its normal runtime dependency is only the ``openvr``
Python package; clock synchronization and the minimal tracker adapter needed
by the sender are implemented below with the Python standard library.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# This compatibility hook remains mutable for tests and legacy callers.  The
# production sender no longer imports ``clock_sync.py`` and therefore has no
# clock-sync helper that can be missing after a one-file migration.
_CLOCK_SYNC_IMPORT_ERROR: ImportError | None = None

CLOCK_SYNC_PROTOCOL = "teleop_runtime.vive.clock_sync/v1"
CLOCK_SYNC_PROBE = "probe"
CLOCK_SYNC_REPLY = "reply"
CLOCK_SYNC_RESULT = "result"


@dataclass(frozen=True)
class ClockSyncReply:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int


@dataclass(frozen=True)
class ClockSyncResult:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int
    t4_sender_monotonic_ns: int


@dataclass(frozen=True)
class ClockSyncMeasurement:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int
    t4_sender_monotonic_ns: int
    collector_minus_sender_offset_ns: int
    round_trip_ns: int
    uncertainty_ns: int


@dataclass(frozen=True)
class ClockSyncEstimate:
    sender_instance_id: str
    collector_minus_sender_offset_ns: int
    round_trip_ns: int
    uncertainty_ns: int
    sample_count: int
    reference_sender_monotonic_ns: int
    reference_collector_monotonic_ns: int


class ClockSyncSampleStore:
    """Retain bounded samples and select the best recent low-RTT mapping."""

    def __init__(
        self,
        *,
        max_samples: int = 64,
        selection_window: int = 8,
        max_round_trip_ns: int | None = None,
    ) -> None:
        self.max_samples = max(1, int(max_samples))
        self.selection_window = max(1, int(selection_window))
        self.max_round_trip_ns = (
            None if max_round_trip_ns is None else max(0, int(max_round_trip_ns))
        )
        self._sender_instance_id: str | None = None
        self._samples: deque[ClockSyncMeasurement] = deque(maxlen=self.max_samples)

    def begin_instance(self, sender_instance_id: str) -> None:
        if self._sender_instance_id != sender_instance_id:
            self._sender_instance_id = sender_instance_id
            self._samples.clear()

    def add(self, measurement: ClockSyncMeasurement) -> ClockSyncEstimate | None:
        self.begin_instance(measurement.sender_instance_id)
        if (
            self.max_round_trip_ns is not None
            and measurement.round_trip_ns > self.max_round_trip_ns
        ):
            return self.estimate()
        self._samples.append(measurement)
        return self.estimate()

    def estimate(self) -> ClockSyncEstimate | None:
        if not self._samples or self._sender_instance_id is None:
            return None
        recent = list(self._samples)[-self.selection_window :]
        selected = min(
            recent,
            key=lambda sample: (sample.round_trip_ns, -sample.probe_id),
        )
        offset_spread_ns = max(
            abs(
                sample.collector_minus_sender_offset_ns
                - selected.collector_minus_sender_offset_ns
            )
            for sample in recent
        )
        return ClockSyncEstimate(
            sender_instance_id=self._sender_instance_id,
            collector_minus_sender_offset_ns=(
                selected.collector_minus_sender_offset_ns
            ),
            round_trip_ns=selected.round_trip_ns,
            uncertainty_ns=selected.uncertainty_ns + offset_spread_ns,
            sample_count=len(self._samples),
            reference_sender_monotonic_ns=(
                selected.t1_sender_monotonic_ns
                + selected.t4_sender_monotonic_ns
            )
            // 2,
            reference_collector_monotonic_ns=(
                selected.t2_collector_monotonic_ns
                + selected.t3_collector_monotonic_ns
            )
            // 2,
        )


def new_sender_instance_id() -> str:
    return uuid.uuid4().hex


def _encode_clock_sync_message(message: Mapping[str, Any]) -> bytes:
    return json.dumps(message, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def make_probe(
    *,
    sender_instance_id: str,
    probe_id: int,
    t1_sender_monotonic_ns: int,
) -> bytes:
    return _encode_clock_sync_message(
        {
            "protocol": CLOCK_SYNC_PROTOCOL,
            "type": CLOCK_SYNC_PROBE,
            "sender_instance_id": str(sender_instance_id),
            "probe_id": int(probe_id),
            "t1_sender_monotonic_ns": int(t1_sender_monotonic_ns),
        }
    )


def make_result(
    *,
    sender_instance_id: str,
    probe_id: int,
    t1_sender_monotonic_ns: int,
    t2_collector_monotonic_ns: int,
    t3_collector_monotonic_ns: int,
    t4_sender_monotonic_ns: int,
) -> bytes:
    return _encode_clock_sync_message(
        {
            "protocol": CLOCK_SYNC_PROTOCOL,
            "type": CLOCK_SYNC_RESULT,
            "sender_instance_id": str(sender_instance_id),
            "probe_id": int(probe_id),
            "t1_sender_monotonic_ns": int(t1_sender_monotonic_ns),
            "t2_collector_monotonic_ns": int(t2_collector_monotonic_ns),
            "t3_collector_monotonic_ns": int(t3_collector_monotonic_ns),
            "t4_sender_monotonic_ns": int(t4_sender_monotonic_ns),
        }
    )


def _required_nonempty_string(
    message: Mapping[str, Any],
    key: str,
) -> str | None:
    value = message.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _required_nonnegative_int(
    message: Mapping[str, Any],
    key: str,
) -> int | None:
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > (2**63 - 1):
        return None
    return int(value)


def parse_reply(packet: bytes) -> ClockSyncReply | None:
    try:
        message = json.loads(packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(message, dict):
        return None
    if (
        message.get("protocol") != CLOCK_SYNC_PROTOCOL
        or message.get("type") != CLOCK_SYNC_REPLY
    ):
        return None
    sender_instance_id = _required_nonempty_string(message, "sender_instance_id")
    probe_id = _required_nonnegative_int(message, "probe_id")
    t1 = _required_nonnegative_int(message, "t1_sender_monotonic_ns")
    t2 = _required_nonnegative_int(message, "t2_collector_monotonic_ns")
    t3 = _required_nonnegative_int(message, "t3_collector_monotonic_ns")
    if (
        sender_instance_id is None
        or probe_id is None
        or t1 is None
        or t2 is None
        or t3 is None
        or t3 < t2
    ):
        return None
    return ClockSyncReply(sender_instance_id, probe_id, t1, t2, t3)


def measurement_from_result(
    result: ClockSyncResult,
) -> ClockSyncMeasurement | None:
    processing_ns = (
        result.t3_collector_monotonic_ns - result.t2_collector_monotonic_ns
    )
    sender_elapsed_ns = (
        result.t4_sender_monotonic_ns - result.t1_sender_monotonic_ns
    )
    round_trip_ns = sender_elapsed_ns - processing_ns
    if round_trip_ns < 0:
        return None
    offset_numerator = (
        result.t2_collector_monotonic_ns
        - result.t1_sender_monotonic_ns
        + result.t3_collector_monotonic_ns
        - result.t4_sender_monotonic_ns
    )
    return ClockSyncMeasurement(
        sender_instance_id=result.sender_instance_id,
        probe_id=result.probe_id,
        t1_sender_monotonic_ns=result.t1_sender_monotonic_ns,
        t2_collector_monotonic_ns=result.t2_collector_monotonic_ns,
        t3_collector_monotonic_ns=result.t3_collector_monotonic_ns,
        t4_sender_monotonic_ns=result.t4_sender_monotonic_ns,
        collector_minus_sender_offset_ns=int(round(offset_numerator / 2.0)),
        round_trip_ns=round_trip_ns,
        uncertainty_ns=(round_trip_ns + 1) // 2,
    )


# The sender remains strict by default: without an explicit CLI override it
# requires exactly two trackers, preserving the released left/right mapping.
EXPECTED_TRACKER_COUNT = 2
SINGLE_TRACKER_COMPAT_RIGHT_NAME = "<right_wrist_unavailable>"

DEFAULT_SEND_BUFFER_BYTES = 1024 * 1024
DEFAULT_SEND_ERROR_BACKOFF_SEC = 0.25
DEFAULT_SEND_ERROR_MAX_BACKOFF_SEC = 2.0
DEFAULT_SEND_HZ = 300.0
DEFAULT_LEGACY_SEND_HZ = 60.0
DEFAULT_VR_EVENT_POLL_HZ = 30.0
DEFAULT_DEVICE_REFRESH_SEC = 1.0
DEFAULT_CLOCK_SYNC_PORT = 10001
DEFAULT_CLOCK_SYNC_PROBES = 16
DEFAULT_CLOCK_SYNC_MIN_SAMPLES = 4
DEFAULT_CLOCK_SYNC_TIMEOUT_MS = 100.0
DEFAULT_CLOCK_SYNC_MAX_RTT_MS = 25.0
DEFAULT_CLOCK_SYNC_STARTUP_INTERVAL_SEC = 0.02
DEFAULT_CLOCK_SYNC_REFRESH_SEC = 5.0
DEFAULT_CLOCK_SYNC_STARTUP_TIMEOUT_SEC = 5.0
TRIAD_OPENVR_MAX_SEARCH_DEPTH = 4
TRIAD_OPENVR_SEARCH_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
    }
)
# ``vive_data_sender.py`` is the local/new entry point.  The separately named
# ``vive_data_sender_legacy.py`` entry point pins this to ``off`` instead.
DEFAULT_CLOCK_SYNC_MODE = "optional"


@dataclass(frozen=True)
class TrackerSpec:
    device_name: str
    tracker_role: str


def tracker_count_arg(value: str) -> int:
    tracker_count = int(value)
    if tracker_count < EXPECTED_TRACKER_COUNT:
        raise argparse.ArgumentTypeError(
            f"must be at least {EXPECTED_TRACKER_COUNT}"
        )
    return tracker_count


def parse_args(
    argv: list[str] | None = None,
    *,
    clock_sync_mode_default: str = DEFAULT_CLOCK_SYNC_MODE,
    send_hz_default: float = DEFAULT_SEND_HZ,
) -> argparse.Namespace:
    if clock_sync_mode_default not in {"off", "optional", "required"}:
        raise ValueError(
            "clock_sync_mode_default must be one of: off, optional, required"
        )
    parser = argparse.ArgumentParser(
        description=(
            "Read an exact number of VIVE GenericTrackers via triad_openvr, assign them "
            "by device-name order, and send structured UDP JSON packets to the teleop "
            "runtime receiver. The default remains exactly two trackers."
        )
    )
    parser.add_argument("--host", type=str, default="192.168.0.2", help="UDP receiver host")
    parser.add_argument("--port", type=int, default=9999, help="UDP receiver port")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional triad_openvr config.json path. Configured serials keep their "
            "logical names; other GenericTrackers receive automatic names."
        ),
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(send_hz_default),
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
        help="Seconds to wait for exactly --tracker-count GenericTrackers before sending.",
    )
    parser.add_argument(
        "--tracker-count",
        type=tracker_count_arg,
        default=EXPECTED_TRACKER_COUNT,
        help=(
            "Exact number of GenericTrackers to send. The strict default is 2; "
            "set this explicitly to 3 or more to enable additional trackers."
        ),
    )
    parser.add_argument(
        "--single",
        dest="allow_single_tracker",
        action="store_true",
        help=(
            "Compatibility mode: when exactly one GenericTracker is connected, "
            "send it as left_wrist and send an invalid right_wrist record. "
            "The default remains strict two-tracker behavior."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print discovered VIVE/OpenVR device names and exit.",
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
    parser.add_argument(
        "--clock-sync-mode",
        choices=("off", "optional", "required"),
        default=clock_sync_mode_default,
        help=(
            "Separate UDP clock-control plane. This new sender defaults to "
            "'optional' and continues pose streaming if the collector has "
            "no control service; 'off' preserves the legacy sender exactly; "
            "'required' exits unless startup calibration succeeds."
        ),
    )
    parser.add_argument(
        "--clock-sync-host",
        type=str,
        default="",
        help="Clock-control collector host; defaults to --host when empty.",
    )
    parser.add_argument(
        "--clock-sync-port",
        type=int,
        default=DEFAULT_CLOCK_SYNC_PORT,
        help=(
            "Separate UDP clock-control port. It never changes or receives "
            "messages on the legacy pose --port."
        ),
    )
    parser.add_argument(
        "--clock-sync-probes",
        type=int,
        default=DEFAULT_CLOCK_SYNC_PROBES,
        help="Number of fast startup NTP-style probes before periodic refresh.",
    )
    parser.add_argument(
        "--clock-sync-min-samples",
        type=int,
        default=DEFAULT_CLOCK_SYNC_MIN_SAMPLES,
        help="Accepted clock samples required for the sender to report ready.",
    )
    parser.add_argument(
        "--clock-sync-timeout-ms",
        type=float,
        default=DEFAULT_CLOCK_SYNC_TIMEOUT_MS,
        help="Per-probe control reply timeout in milliseconds.",
    )
    parser.add_argument(
        "--clock-sync-max-rtt-ms",
        type=float,
        default=DEFAULT_CLOCK_SYNC_MAX_RTT_MS,
        help="Reject a control sample when its RTT exceeds this value; <=0 disables the cap.",
    )
    parser.add_argument(
        "--clock-sync-startup-interval-sec",
        type=float,
        default=DEFAULT_CLOCK_SYNC_STARTUP_INTERVAL_SEC,
        help="Delay between startup calibration probes.",
    )
    parser.add_argument(
        "--clock-sync-refresh-sec",
        type=float,
        default=DEFAULT_CLOCK_SYNC_REFRESH_SEC,
        help="Periodic calibration refresh interval after startup probes.",
    )
    parser.add_argument(
        "--clock-sync-startup-timeout-sec",
        type=float,
        default=DEFAULT_CLOCK_SYNC_STARTUP_TIMEOUT_SEC,
        help="Maximum wait for required startup calibration.",
    )
    return parser.parse_args(argv)


def discovered_tracker_names(vive: Any) -> list[str]:
    """Return active GenericTracker names in deterministic lexical order."""

    object_names = getattr(vive, "object_names", None)
    if isinstance(object_names, dict) and "Tracker" in object_names:
        raw_names = object_names["Tracker"]
    else:
        raw_names = []
        for name, device in getattr(vive, "devices", {}).items():
            if getattr(device, "device_class", None) == "Tracker":
                raw_names.append(name)

    devices = getattr(vive, "devices", {})
    names = {str(name) for name in raw_names if str(name) in devices}
    return sorted(names, key=lambda name: (name.casefold(), name))


def tracker_role_for_index(index: int) -> str:
    """Return the stable role assigned to a lexically ordered tracker."""

    if index < 0:
        raise ValueError("tracker index must be non-negative")
    if index == 0:
        return "left_wrist"
    if index == 1:
        return "right_wrist"
    if index == 2:
        return "auxiliary"
    return f"auxiliary_{index - 1}"


def auto_tracker_specs(
    vive: Any,
    *,
    tracker_count: int = EXPECTED_TRACKER_COUNT,
    allow_single_tracker: bool = False,
) -> list[TrackerSpec]:
    """Assign exactly ``tracker_count`` trackers by lexical device name.

    The first two roles retain the historical left/right mapping. Additional
    trackers receive ``auxiliary``, ``auxiliary_2``, ... in the same stable
    order. The explicit single-tracker compatibility mode is the only
    exception: one connected device becomes ``left_wrist`` and a synthetic
    unavailable ``right_wrist`` spec is returned.
    """

    if tracker_count < EXPECTED_TRACKER_COUNT:
        raise ValueError(
            f"tracker_count must be at least {EXPECTED_TRACKER_COUNT}; "
            f"got {tracker_count}"
        )

    names = discovered_tracker_names(vive)
    if (
        allow_single_tracker
        and tracker_count == EXPECTED_TRACKER_COUNT
        and len(names) == 1
    ):
        return [
            TrackerSpec(device_name=names[0], tracker_role="left_wrist"),
            TrackerSpec(
                device_name=SINGLE_TRACKER_COMPAT_RIGHT_NAME,
                tracker_role="right_wrist",
            ),
        ]
    if len(names) != tracker_count:
        available = ", ".join(names) or "<none>"
        raise ValueError(
            f"exactly {tracker_count} GenericTrackers are required; "
            f"found {len(names)} ({available})"
        )
    return [
        TrackerSpec(
            device_name=device_name,
            tracker_role=tracker_role_for_index(index),
        )
        for index, device_name in enumerate(names)
    ]


def pose_matrix_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pose_sample_clocks() -> tuple[float, float]:
    """Return wall time plus a high-resolution monotonic sample clock.

    ``time.monotonic()`` can advance in roughly 16 ms quanta on the Windows
    VIVE host.  ``perf_counter()`` uses the high-resolution performance
    counter and is still monotonic, so its within-session deltas remain valid
    at the 300 Hz sender rate.
    """

    return time.time(), time.perf_counter()


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


def _openvr_vector3(value: object, *, label: str) -> list[float]:
    if value is None:
        raise ValueError(f"OpenVR {label} is unavailable")

    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        components: object = [
            getattr(value, "x"),
            getattr(value, "y"),
            getattr(value, "z"),
        ]
    else:
        components = getattr(value, "v", value)
    try:
        vector = [float(components[index]) for index in range(3)]  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"OpenVR {label} must contain three numeric values") from exc
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"OpenVR {label} contains non-finite values")
    return vector


def _tracked_device_pose(device: object, pose_snapshot: object) -> object:
    for attribute in ("index", "device_index", "tracked_device_index"):
        raw_index = getattr(device, attribute, None)
        if raw_index is None:
            continue
        index = raw_index() if callable(raw_index) else raw_index
        try:
            return pose_snapshot[int(index)]  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"OpenVR pose snapshot does not contain device index {index!r}"
            ) from exc
    raise ValueError("OpenVR tracked device does not expose a pose index")


def openvr_velocity_payload(device: object, pose_snapshot: object) -> dict[str, Any]:
    """Read linear/angular velocity from the same snapshot as the pose.

    Standard ``triad_openvr`` exposes ``get_velocity`` and
    ``get_angular_velocity``.  The direct ``TrackedDevicePose_t`` fallback
    keeps this compatible with small local wrappers that expose only an index.
    A missing velocity never invalidates an otherwise valid pose.
    """

    payload: dict[str, Any] = {
        "linear_velocity_m_s": None,
        "angular_velocity_rad_s": None,
        "velocity_frame_name": "vive_world",
        "velocity_source": "openvr_tracked_device_pose",
        "velocity_valid": False,
        "velocity_error": None,
    }
    try:
        get_velocity = getattr(device, "get_velocity", None)
        get_angular_velocity = getattr(device, "get_angular_velocity", None)
        tracked_pose: object | None = None
        if callable(get_velocity):
            linear_raw = get_velocity(pose_snapshot)
        else:
            tracked_pose = _tracked_device_pose(device, pose_snapshot)
            linear_raw = getattr(tracked_pose, "vVelocity", None)
        if callable(get_angular_velocity):
            angular_raw = get_angular_velocity(pose_snapshot)
        else:
            if tracked_pose is None:
                tracked_pose = _tracked_device_pose(device, pose_snapshot)
            angular_raw = getattr(tracked_pose, "vAngularVelocity", None)

        linear_velocity = _openvr_vector3(
            linear_raw,
            label="linear velocity",
        )
        angular_velocity = _openvr_vector3(
            angular_raw,
            label="angular velocity",
        )
    except Exception as exc:  # triad/openvr variants expose different ctypes wrappers.
        payload["velocity_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    payload.update(
        {
            "linear_velocity_m_s": linear_velocity,
            "angular_velocity_rad_s": angular_velocity,
            "velocity_valid": True,
        }
    )
    return payload


def missing_tracker_names(vive: Any, specs: list[TrackerSpec]) -> list[str]:
    return [spec.device_name for spec in specs if spec.device_name not in vive.devices]


def tracker_mapping_text(specs: list[TrackerSpec]) -> str:
    return ", ".join(f"{spec.device_name}:{spec.tracker_role}" for spec in specs)


def wait_for_trackers(
    vive: Any,
    timeout_sec: float,
    *,
    tracker_count: int = EXPECTED_TRACKER_COUNT,
    allow_single_tracker: bool = False,
) -> list[TrackerSpec] | None:
    if tracker_count < EXPECTED_TRACKER_COUNT:
        raise ValueError(
            f"tracker_count must be at least {EXPECTED_TRACKER_COUNT}; "
            f"got {tracker_count}"
        )
    deadline = time.monotonic() + max(0.0, timeout_sec)
    next_status = 0.0
    while time.monotonic() < deadline:
        names = discovered_tracker_names(vive)
        if len(names) > tracker_count:
            raise ValueError(
                f"more than {tracker_count} GenericTrackers are connected; "
                f"found {len(names)} ({', '.join(names)})"
            )
        single_tracker_ready = (
            allow_single_tracker
            and tracker_count == EXPECTED_TRACKER_COUNT
            and len(names) == 1
        )
        if len(names) == tracker_count or single_tracker_ready:
            return auto_tracker_specs(
                vive,
                tracker_count=tracker_count,
                allow_single_tracker=allow_single_tracker,
            )
        now = time.monotonic()
        if now >= next_status:
            expected_text = (
                f"exactly {tracker_count} GenericTrackers"
                if not allow_single_tracker or tracker_count != EXPECTED_TRACKER_COUNT
                else "exactly 2 GenericTrackers or one compatibility tracker"
            )
            print(
                f"[Wait] waiting for {expected_text}; "
                f"found {len(names)} ({', '.join(names) or '<none>'})"
            )
            next_status = now + 1.0
        if hasattr(vive, "poll_vr_events"):
            vive.poll_vr_events()
        time.sleep(0.1)
    return None


def wait_for_tracker_pair(
    vive: Any,
    timeout_sec: float,
) -> list[TrackerSpec] | None:
    """Compatibility wrapper retaining the historical strict two-tracker wait."""

    return wait_for_trackers(
        vive,
        timeout_sec,
        tracker_count=EXPECTED_TRACKER_COUNT,
    )


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


def _openvr_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class EmbeddedOpenVRTrackedDevice:
    """Small tracked-device wrapper used by the one-file sender backend."""

    def __init__(
        self,
        vr_system: Any,
        index: int,
        device_class: str,
        openvr_module: Any,
    ) -> None:
        self.vr = vr_system
        self.index = int(index)
        self.device_class = str(device_class)
        self._openvr = openvr_module

    def get_serial(self) -> str:
        return _openvr_text(
            self.vr.getStringTrackedDeviceProperty(
                self.index,
                self._openvr.Prop_SerialNumber_String,
            )
        )

    def get_model(self) -> str:
        return _openvr_text(
            self.vr.getStringTrackedDeviceProperty(
                self.index,
                self._openvr.Prop_ModelNumber_String,
            )
        )

    def get_pose_matrix(self, pose_snapshot: object | None = None) -> object:
        snapshot = pose_snapshot
        if snapshot is None:
            snapshot = _embedded_get_pose(self.vr, self._openvr)
        tracked_pose = snapshot[self.index]  # type: ignore[index]
        if not tracked_pose.bPoseIsValid:
            return None
        return tracked_pose.mDeviceToAbsoluteTracking

    def get_velocity(self, pose_snapshot: object | None = None) -> object:
        snapshot = pose_snapshot
        if snapshot is None:
            snapshot = _embedded_get_pose(self.vr, self._openvr)
        tracked_pose = snapshot[self.index]  # type: ignore[index]
        return tracked_pose.vVelocity if tracked_pose.bPoseIsValid else None

    def get_angular_velocity(
        self,
        pose_snapshot: object | None = None,
    ) -> object:
        snapshot = pose_snapshot
        if snapshot is None:
            snapshot = _embedded_get_pose(self.vr, self._openvr)
        tracked_pose = snapshot[self.index]  # type: ignore[index]
        return tracked_pose.vAngularVelocity if tracked_pose.bPoseIsValid else None


def _embedded_get_pose(vr_system: Any, openvr_module: Any) -> object:
    return vr_system.getDeviceToAbsoluteTrackingPose(
        openvr_module.TrackingUniverseStanding,
        0,
        openvr_module.k_unMaxTrackedDeviceCount,
    )


class EmbeddedTriadOpenVR:
    """Dependency-light subset of triad_openvr required by this sender.

    It intentionally implements only discovery, naming, pose/velocity reads,
    and device refresh.  Unlike the full repository driver, it does not need
    numpy or transforms3d.
    """

    DEVICE_TYPES = ("Tracking Reference", "HMD", "Controller", "Tracker")

    def __init__(self, configfile_path: str | None = None) -> None:
        self._shutdown = False
        self._config_mode = configfile_path is not None
        try:
            self._openvr = importlib.import_module("openvr")
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "[Error] the portable VIVE sender requires the 'openvr' Python "
                "package. Install it in the sender environment and retry."
            ) from exc

        try:
            self.vr = self._openvr.init(self._openvr.VRApplication_Other)
            self.vrsystem = self._openvr.VRSystem()
        except Exception as exc:
            raise SystemExit(f"[Error] failed to initialize SteamVR/OpenVR: {exc}") from exc

        self.object_names = {device_type: [] for device_type in self.DEVICE_TYPES}
        self.devices: dict[str, EmbeddedOpenVRTrackedDevice] = {}
        self.device_index_map: dict[int, str] = {}
        self.device_serial_map: dict[int, str] = {}
        self.serial_device_name_map: dict[str, str] = {}
        self.config_devices_by_serial: dict[str, dict[str, str]] = {}
        self.refresh_interval = 0.5
        self._last_refresh_time = 0.0
        self._load_config(configfile_path)
        self.refresh_discovered_devices()

    def _load_config(self, configfile_path: str | None) -> None:
        if not configfile_path:
            return
        try:
            with Path(configfile_path).open("r", encoding="utf-8") as stream:
                config = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"[Error] failed to read triad_openvr config {configfile_path}: {exc}"
            ) from exc
        if not isinstance(config, dict):
            raise SystemExit("[Error] triad_openvr config root must be a JSON object")
        devices = config.get("devices", [])
        if not isinstance(devices, list):
            raise SystemExit("[Error] triad_openvr config 'devices' must be a list")
        for raw_device in devices:
            if not isinstance(raw_device, dict):
                continue
            serial = _openvr_text(raw_device.get("serial")).strip()
            name = _openvr_text(raw_device.get("name")).strip()
            device_type = _openvr_text(raw_device.get("type")).strip()
            if not serial or not name or device_type not in self.object_names:
                print(f"[Warn] skipping invalid triad_openvr config device: {raw_device}")
                continue
            self.config_devices_by_serial[serial] = {
                "name": name,
                "type": device_type,
                "serial": serial,
            }
            self.serial_device_name_map[serial] = name

    def close(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._openvr.shutdown()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def get_pose(self) -> object:
        return _embedded_get_pose(self.vr, self._openvr)

    def poll_vr_events(self) -> int:
        changes = 0
        event = self._openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == self._openvr.VREvent_TrackedDeviceActivated:
                changes += int(self.add_tracked_device(event.trackedDeviceIndex))
            elif event.eventType == self._openvr.VREvent_TrackedDeviceDeactivated:
                changes += int(self.remove_tracked_device(event.trackedDeviceIndex))
        if changes or time.monotonic() - self._last_refresh_time >= self.refresh_interval:
            changes += self.refresh_discovered_devices()
        return changes

    def refresh_discovered_devices(self, poses: object | None = None) -> int:
        snapshot = self.get_pose() if poses is None else poses
        connected_indexes = {
            index
            for index in range(self._openvr.k_unMaxTrackedDeviceCount)
            if snapshot[index].bDeviceIsConnected  # type: ignore[index]
        }
        changes = 0
        for index in list(self.device_index_map):
            if index not in connected_indexes:
                changes += int(self.remove_tracked_device(index))
        for index in sorted(connected_indexes):
            changes += int(self.add_tracked_device(index))
        self._last_refresh_time = time.monotonic()
        return changes

    def add_tracked_device(self, tracked_device_index: int) -> bool:
        registration = self._resolve_device_registration(tracked_device_index)
        if registration is None:
            return self.remove_tracked_device(tracked_device_index)
        device_name, device_type, serial = registration
        previous_name = self.device_index_map.get(tracked_device_index)
        if previous_name == device_name and device_name in self.devices:
            self.devices[device_name].index = tracked_device_index
            self.device_serial_map[tracked_device_index] = serial
            return False
        if previous_name and previous_name != device_name:
            self.remove_tracked_device(tracked_device_index)
        previous_device = self.devices.get(device_name)
        if previous_device is not None and previous_device.index != tracked_device_index:
            self.remove_tracked_device(previous_device.index)
        if device_name not in self.object_names[device_type]:
            self.object_names[device_type].append(device_name)
        self.devices[device_name] = EmbeddedOpenVRTrackedDevice(
            self.vr,
            tracked_device_index,
            device_type,
            self._openvr,
        )
        self.device_index_map[tracked_device_index] = device_name
        self.device_serial_map[tracked_device_index] = serial
        if serial:
            self.serial_device_name_map[serial] = device_name
        if device_type == "Tracker":
            print(f"Added tracker: {device_name}\n Index: {tracked_device_index}")
        return True

    def remove_tracked_device(self, tracked_device_index: int) -> bool:
        if tracked_device_index not in self.device_index_map:
            return False
        device_name = self.device_index_map.pop(tracked_device_index)
        self.device_serial_map.pop(tracked_device_index, None)
        device = self.devices.pop(device_name, None)
        if device is not None and device_name in self.object_names[device.device_class]:
            self.object_names[device.device_class].remove(device_name)
        return True

    def _resolve_device_registration(
        self,
        tracked_device_index: int,
    ) -> tuple[str, str, str] | None:
        device_type = self._device_type_from_index(tracked_device_index)
        if device_type is None:
            return None
        serial = self._get_device_serial(tracked_device_index)
        configured = self.config_devices_by_serial.get(serial)
        if configured is not None:
            return configured["name"], configured["type"], serial
        if self._config_mode and device_type != "Tracker":
            return None
        device_name = self.serial_device_name_map.get(serial) if serial else None
        if not device_name:
            previous_name = self.device_index_map.get(tracked_device_index)
            previous_device = self.devices.get(previous_name or "")
            if previous_device is not None and previous_device.device_class == device_type:
                device_name = previous_name
        if not device_name:
            device_name = self._next_device_name(device_type)
        return device_name, device_type, serial

    def _device_type_from_index(self, tracked_device_index: int) -> str | None:
        device_class = self.vr.getTrackedDeviceClass(tracked_device_index)
        class_map = {
            self._openvr.TrackedDeviceClass_Controller: "Controller",
            self._openvr.TrackedDeviceClass_HMD: "HMD",
            self._openvr.TrackedDeviceClass_GenericTracker: "Tracker",
            self._openvr.TrackedDeviceClass_TrackingReference: "Tracking Reference",
        }
        return class_map.get(device_class)

    def _get_device_serial(self, tracked_device_index: int) -> str:
        try:
            value = self.vr.getStringTrackedDeviceProperty(
                tracked_device_index,
                self._openvr.Prop_SerialNumber_String,
            )
        except Exception:
            return ""
        return _openvr_text(value).strip()

    def _next_device_name(self, device_type: str) -> str:
        prefixes = {
            "Controller": "controller",
            "HMD": "hmd",
            "Tracker": "tracker",
            "Tracking Reference": "tracking_reference",
        }
        prefix = prefixes[device_type]
        used_names = set(self.devices)
        used_names.update(self.object_names[device_type])
        used_names.update(self.serial_device_name_map.values())
        suffix = 1
        while f"{prefix}_{suffix}" in used_names:
            suffix += 1
        return f"{prefix}_{suffix}"

    def print_discovered_objects(self) -> None:
        for device_type, names in self.object_names.items():
            plural = device_type if len(names) == 1 else f"{device_type}s"
            print(f"Found {len(names)} {plural}")
            for name in names:
                device = self.devices[name]
                try:
                    serial = device.get_serial()
                except Exception:
                    serial = "unknown-serial"
                try:
                    model = device.get_model()
                except Exception:
                    model = "unknown-model"
                print(f"  {name} ({serial}, {model})")


class EmbeddedTriadOpenVRModule:
    """Module-shaped holder matching the external triad_openvr API."""

    triad_openvr = EmbeddedTriadOpenVR
    __file__ = str(Path(__file__).resolve())


def canonical_triad_openvr_module_path() -> Path:
    """Return the preferred repository-owned OpenVR implementation path."""
    return Path(__file__).resolve().parents[1] / "triad_openvr" / "triad_openvr.py"


def _recursive_triad_openvr_candidates(
    root: Path,
    *,
    max_depth: int = TRIAD_OPENVR_MAX_SEARCH_DEPTH,
) -> list[Path]:
    """Find driver files below ``root`` without following unbounded trees."""

    if max_depth < 0 or not root.is_dir():
        return []

    matches: list[Path] = []
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda path: (path.name.casefold(), path.name),
                reverse=True,
            )
        except OSError:
            continue

        for entry in entries:
            if entry.is_file() and entry.name.casefold() == "triad_openvr.py":
                matches.append(entry.resolve())
                continue
            if depth >= max_depth or entry.is_symlink() or not entry.is_dir():
                continue
            if entry.name.casefold() in TRIAD_OPENVR_SEARCH_SKIP_DIRS:
                continue
            pending.append((entry, depth + 1))

    return sorted(
        matches,
        key=lambda path: (
            len(path.relative_to(root).parts),
            str(path).casefold(),
            str(path),
        ),
    )


def triad_openvr_module_candidates(
    sender_path: Path | None = None,
) -> list[Path]:
    """Return deterministic local candidates for the triad_openvr driver.

    Portable VIVE sender bundles are not always copied with the repository's
    original directory shape. Search relative to the sender itself so the
    driver may live beside it, in one of its subdirectories, or in a sibling
    subtree below its parent directory.
    """

    sender_file = Path(__file__ if sender_path is None else sender_path).resolve()
    sender_dir = sender_file.parent
    parent_dir = sender_dir.parent

    local_preferred = [
        sender_dir / "triad_openvr.py",
        sender_dir / "triad_openvr" / "triad_openvr.py",
    ]
    parent_preferred = [
        parent_dir / "triad_openvr" / "triad_openvr.py",
        parent_dir / "triad_openvr.py",
    ]
    ordered = [
        *local_preferred,
        *_recursive_triad_openvr_candidates(sender_dir),
        *parent_preferred,
        *_recursive_triad_openvr_candidates(parent_dir),
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in ordered:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)
    return candidates


def find_triad_openvr_module_path(sender_path: Path | None = None) -> Path:
    """Find the first usable local triad_openvr implementation."""

    candidates = triad_openvr_module_candidates(sender_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    sender_file = Path(__file__ if sender_path is None else sender_path).resolve()
    search_roots = (sender_file.parent, sender_file.parent.parent)
    raise SystemExit(
        "[Error] triad_openvr.py was not found beside the VIVE sender, in its "
        "subdirectories, or in sibling subdirectories below its parent. "
        f"Searched roots: {', '.join(str(root) for root in search_roots)}"
    )


def load_triad_openvr(module_path: Path | None = None) -> Any:
    # Load by an explicit local file path so direct Windows execution and
    # ``python -m`` never fall through to an unrelated installed package.
    if module_path is None:
        try:
            resolved_path = find_triad_openvr_module_path()
        except SystemExit:
            print(
                "[Info] no external triad_openvr.py found; using embedded "
                "dependency-light OpenVR adapter"
            )
            return EmbeddedTriadOpenVRModule
    else:
        resolved_path = Path(module_path).resolve()
    if not resolved_path.is_file():
        raise SystemExit(
            "[Error] triad_openvr implementation is missing: "
            f"{resolved_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "dexteleop_triad_openvr",
        resolved_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            "[Error] cannot load triad_openvr implementation: "
            f"{resolved_path}"
        )

    module = importlib.util.module_from_spec(spec)
    module_name = spec.name
    had_previous = module_name in sys.modules
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    print(f"[Info] loading triad_openvr from: {resolved_path}")
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if had_previous:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        if module_path is None:
            print(
                f"[Warn] external triad_openvr dependency '{exc.name}' is unavailable; "
                "using embedded dependency-light OpenVR adapter"
            )
            return EmbeddedTriadOpenVRModule
        missing = exc.name or "openvr"
        raise SystemExit(
            f"[Error] failed to import '{missing}' from {resolved_path}. "
            "Install the VIVE sender dependencies in this Python environment, then retry."
        ) from exc
    except BaseException:
        if had_previous:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        raise
    return module


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


class ClockSyncSenderClient:
    """Sender-initiated, non-blocking clock control client.

    The VIVE host initiates this separate UDP exchange so a Windows sender
    does not need an inbound firewall exception.  It never shares a socket or
    datagram with the legacy pose stream.  A failed optional exchange is
    deliberately unable to slow or stop VIVE pose transmission.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        startup_probes: int,
        min_samples: int,
        timeout_sec: float,
        max_rtt_sec: float,
        startup_interval_sec: float,
        refresh_sec: float,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.startup_probes = max(1, int(startup_probes))
        self.min_samples = max(1, int(min_samples))
        self.timeout_sec = max(0.001, float(timeout_sec))
        self.max_rtt_ns = (
            None
            if float(max_rtt_sec) <= 0.0
            else int(float(max_rtt_sec) * 1_000_000_000)
        )
        self.startup_interval_sec = max(0.0, float(startup_interval_sec))
        self.refresh_sec = max(0.05, float(refresh_sec))
        self.sender_instance_id = new_sender_instance_id()
        self._sample_store = ClockSyncSampleStore(
            max_round_trip_ns=self.max_rtt_ns,
        )
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._next_probe_id = 0
        self._sent_count = 0
        self._accepted_count = 0
        self._timeout_count = 0
        self._rejected_count = 0
        self._error_count = 0
        self._last_error: str | None = None
        self._last_measurement_ns: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((self.host, self.port))
            sock.settimeout(self.timeout_sec)
        except OSError:
            close_udp_sender_socket(sock)
            raise
        self._sock = sock
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vive_clock_sync_sender",
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        sock = self._sock
        self._sock = None
        close_udp_sender_socket(sock)
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(1.0, self.timeout_sec * 2.0))

    def wait_until_ready(self, timeout_sec: float) -> bool:
        return self._ready.wait(timeout=max(0.0, float(timeout_sec)))

    def debug_stats(self) -> dict[str, Any]:
        with self._lock:
            estimate = self._sample_store.estimate()
            return {
                "enabled": True,
                "sender_instance_id": self.sender_instance_id,
                "ready": self._ready.is_set(),
                "sent_count": self._sent_count,
                "accepted_count": self._accepted_count,
                "timeout_count": self._timeout_count,
                "rejected_count": self._rejected_count,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "last_measurement_monotonic_ns": self._last_measurement_ns,
                "estimate": None
                if estimate is None
                else {
                    "collector_minus_sender_offset_ns": (
                        estimate.collector_minus_sender_offset_ns
                    ),
                    "round_trip_ns": estimate.round_trip_ns,
                    "uncertainty_ns": estimate.uncertainty_ns,
                    "sample_count": estimate.sample_count,
                },
            }

    def _run(self) -> None:
        startup_remaining = self.startup_probes
        while not self._stop.is_set():
            self._probe_once()
            if startup_remaining > 0:
                startup_remaining -= 1
                wait_sec = self.startup_interval_sec
            elif not self._ready.is_set():
                # The collector may be launched after the sender.  Keep a
                # bounded retry cadence until the first usable mapping rather
                # than waiting a full refresh interval after the initial fast
                # probe burst has missed it.
                wait_sec = min(1.0, self.refresh_sec)
            else:
                wait_sec = self.refresh_sec
            self._stop.wait(wait_sec)

    def _probe_once(self) -> None:
        sock = self._sock
        if sock is None or self._stop.is_set():
            return
        with self._lock:
            probe_id = self._next_probe_id
            self._next_probe_id += 1
        t1 = time.perf_counter_ns()
        try:
            sock.send(
                make_probe(
                    sender_instance_id=self.sender_instance_id,
                    probe_id=probe_id,
                    t1_sender_monotonic_ns=t1,
                )
            )
            with self._lock:
                self._sent_count += 1
        except OSError as exc:
            self._record_error(exc)
            return

        try:
            packet = sock.recv(4096)
        except socket.timeout:
            with self._lock:
                self._timeout_count += 1
            return
        except OSError as exc:
            if not self._stop.is_set():
                self._record_error(exc)
            return

        t4 = time.perf_counter_ns()
        reply = parse_reply(packet)
        if (
            reply is None
            or reply.sender_instance_id != self.sender_instance_id
            or reply.probe_id != probe_id
            or reply.t1_sender_monotonic_ns != t1
        ):
            with self._lock:
                self._rejected_count += 1
            return

        result_packet = make_result(
            sender_instance_id=self.sender_instance_id,
            probe_id=probe_id,
            t1_sender_monotonic_ns=t1,
            t2_collector_monotonic_ns=reply.t2_collector_monotonic_ns,
            t3_collector_monotonic_ns=reply.t3_collector_monotonic_ns,
            t4_sender_monotonic_ns=t4,
        )
        try:
            sock.send(result_packet)
        except OSError as exc:
            self._record_error(exc)
            return

        from_result = measurement_from_result(
            ClockSyncResult(
                sender_instance_id=self.sender_instance_id,
                probe_id=probe_id,
                t1_sender_monotonic_ns=t1,
                t2_collector_monotonic_ns=reply.t2_collector_monotonic_ns,
                t3_collector_monotonic_ns=reply.t3_collector_monotonic_ns,
                t4_sender_monotonic_ns=t4,
            )
        )
        if (
            from_result is None
            or (
                self.max_rtt_ns is not None
                and from_result.round_trip_ns > self.max_rtt_ns
            )
        ):
            with self._lock:
                self._rejected_count += 1
            return
        with self._lock:
            estimate = self._sample_store.add(from_result)
            self._accepted_count += 1
            self._last_measurement_ns = time.perf_counter_ns()
            if estimate is not None and estimate.sample_count >= self.min_samples:
                self._ready.set()

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._error_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"


def build_v1_tracking_packet(
    *,
    seq: int,
    sender_timestamp: float,
    pose_sample_timestamp: float,
    pose_sample_monotonic_time: float,
    trackers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the current v1 pose data-plane payload.

    Do not add clock-control fields here. This payload continues to be sent
    only to the historical pose port. ``build_legacy_v1_tracking_packet`` is
    deliberately narrower for released receivers that require the old field
    shape.
    """

    return {
        "protocol": "teleop_runtime.vive/v1",
        "seq": int(seq),
        "sender_timestamp": float(sender_timestamp),
        "pose_sample_timestamp": float(pose_sample_timestamp),
        "pose_sample_monotonic_time": float(pose_sample_monotonic_time),
        "trackers": trackers,
    }


_LEGACY_V1_TRACKER_FIELDS = (
    "tracker_name",
    "tracker_role",
    "valid",
    "pose_matrix_4x4",
    "error",
    "read_elapsed_ms",
)


def build_legacy_v1_tracking_packet(
    *,
    seq: int,
    sender_timestamp: float,
    trackers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the release-era strict v1 JSON shape.

    This is intentionally separate from the current local v1 payload: it
    excludes pose-sample clocks and tracker velocity fields introduced after
    the stable receiver baseline. Keep both root and per-tracker key order
    deterministic so a strict legacy parser sees the same schema it did
    before the calibrated sender was added.
    """

    legacy_trackers = [
        {field_name: tracker.get(field_name) for field_name in _LEGACY_V1_TRACKER_FIELDS}
        for tracker in trackers
    ]
    return {
        "protocol": "teleop_runtime.vive/v1",
        "seq": int(seq),
        "sender_timestamp": float(sender_timestamp),
        "trackers": legacy_trackers,
    }


def start_clock_sync_client(
    args: argparse.Namespace,
) -> ClockSyncSenderClient | None:
    """Start the optional control plane without touching the v1 pose path."""

    mode = str(getattr(args, "clock_sync_mode", "off"))
    if mode == "off":
        return None
    if _CLOCK_SYNC_IMPORT_ERROR is not None:
        message = (
            "clock-sync helper is unavailable "
            f"({type(_CLOCK_SYNC_IMPORT_ERROR).__name__}: {_CLOCK_SYNC_IMPORT_ERROR})"
        )
        if mode == "required":
            raise SystemExit(f"[Error] {message}")
        print(f"[Warn] {message}; continuing legacy VIVE pose stream without sync")
        return None
    try:
        control_port = int(args.clock_sync_port)
        pose_port = int(args.port)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[Error] invalid clock sync port: {exc}") from exc
    if not 1 <= control_port <= 65535:
        raise SystemExit("[Error] --clock-sync-port must be in [1, 65535] when enabled.")
    if control_port == pose_port:
        raise SystemExit(
            "[Error] --clock-sync-port must differ from --port so control messages "
            "can never reach the legacy VIVE pose receiver."
        )
    try:
        startup_probes = int(args.clock_sync_probes)
        min_samples = int(args.clock_sync_min_samples)
        timeout_ms = float(args.clock_sync_timeout_ms)
        max_rtt_ms = float(args.clock_sync_max_rtt_ms)
        startup_interval_sec = float(args.clock_sync_startup_interval_sec)
        refresh_sec = float(args.clock_sync_refresh_sec)
        startup_timeout_sec = float(args.clock_sync_startup_timeout_sec)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[Error] invalid clock sync option: {exc}") from exc
    if startup_probes < 1 or min_samples < 1:
        raise SystemExit("[Error] --clock-sync-probes and --clock-sync-min-samples must be >= 1.")
    if not all(
        math.isfinite(value)
        for value in (
            timeout_ms,
            max_rtt_ms,
            startup_interval_sec,
            refresh_sec,
            startup_timeout_sec,
        )
    ):
        raise SystemExit("[Error] clock-sync time values must be finite.")
    if timeout_ms <= 0.0 or max_rtt_ms < 0.0:
        raise SystemExit(
            "[Error] --clock-sync-timeout-ms must be > 0 and --clock-sync-max-rtt-ms must be >= 0."
        )
    if startup_interval_sec < 0.0 or refresh_sec <= 0.0 or startup_timeout_sec < 0.0:
        raise SystemExit(
            "[Error] clock-sync intervals must satisfy startup>=0, refresh>0, timeout>=0."
        )
    host = str(getattr(args, "clock_sync_host", "") or args.host).strip()
    if not host:
        message = "clock sync host is empty"
        if mode == "required":
            raise SystemExit(f"[Error] {message}")
        print(f"[Warn] {message}; continuing legacy VIVE pose stream without sync")
        return None

    try:
        client = ClockSyncSenderClient(
            host=host,
            port=control_port,
            startup_probes=startup_probes,
            min_samples=min_samples,
            timeout_sec=timeout_ms / 1000.0,
            max_rtt_sec=max_rtt_ms / 1000.0,
            startup_interval_sec=startup_interval_sec,
            refresh_sec=refresh_sec,
        )
        client.start()
    except (OSError, ValueError) as exc:
        message = f"clock sync control start failed for {host}:{args.clock_sync_port}: {exc}"
        if mode == "required":
            raise SystemExit(f"[Error] {message}") from exc
        print(f"[Warn] {message}; continuing legacy VIVE pose stream without sync")
        return None

    print(
        f"[ClockSync] control plane enabled mode={mode} target={host}:{control_port} "
        f"instance={client.sender_instance_id}"
    )
    if mode == "required" and not client.wait_until_ready(
        startup_timeout_sec
    ):
        stats = client.debug_stats()
        client.close()
        raise SystemExit(
            "[Error] required clock sync did not reach the minimum accepted sample count "
            f"within {startup_timeout_sec:.2f}s; stats={stats}"
        )
    return client


def main(
    argv: list[str] | None = None,
    *,
    clock_sync_mode_default: str = DEFAULT_CLOCK_SYNC_MODE,
    send_hz_default: float = DEFAULT_SEND_HZ,
    force_clock_sync_mode: str | None = None,
    legacy_wire_format: bool = False,
) -> None:
    """Run the sender, optionally pinning a compatibility entry point's mode.

    ``vive_data_sender_legacy.py`` calls this with ``force_clock_sync_mode``
    set to ``off``.  Keeping the OpenVR/data-plane implementation shared
    guarantees the legacy and new entry points cannot silently drift apart.
    """

    if force_clock_sync_mode not in {None, "off", "optional", "required"}:
        raise ValueError(
            "force_clock_sync_mode must be None, off, optional, or required"
        )
    args = parse_args(
        argv,
        clock_sync_mode_default=clock_sync_mode_default,
        send_hz_default=send_hz_default,
    )
    if force_clock_sync_mode is not None:
        requested_mode = str(args.clock_sync_mode)
        args.clock_sync_mode = force_clock_sync_mode
        if requested_mode != force_clock_sync_mode:
            print(
                "[Legacy] ignoring --clock-sync-mode "
                f"{requested_mode!r}; this entry point always uses "
                f"{force_clock_sync_mode!r}."
            )
    triad_openvr_module = load_triad_openvr()
    vive = (
        triad_openvr_module.triad_openvr(args.config)
        if args.config
        else triad_openvr_module.triad_openvr()
    )
    vive.print_discovered_objects()

    if args.list_devices:
        return

    tracker_count = int(args.tracker_count)
    allow_single_tracker = bool(args.allow_single_tracker)
    try:
        if args.wait_for_trackers > 0.0:
            tracker_specs = wait_for_trackers(
                vive,
                args.wait_for_trackers,
                tracker_count=tracker_count,
                allow_single_tracker=allow_single_tracker,
            )
            if tracker_specs is None:
                raise ValueError(
                    f"did not find exactly {tracker_count} GenericTrackers "
                    f"within {args.wait_for_trackers:.1f}s"
                )
        else:
            tracker_specs = auto_tracker_specs(
                vive,
                tracker_count=tracker_count,
                allow_single_tracker=allow_single_tracker,
            )
    except ValueError as exc:
        raise SystemExit(f"[Error] {exc}") from exc

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

    try:
        clock_sync_client = start_clock_sync_client(args)
    except BaseException:
        # In required mode, calibration can reject startup after the pose
        # socket has been opened. Close it explicitly before propagating the
        # CLI error rather than waiting for process teardown.
        close_udp_sender_socket(sock)
        raise

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

                current_tracker_names = discovered_tracker_names(vive)
                if len(current_tracker_names) > tracker_count:
                    raise SystemExit(
                        f"[Error] more than {tracker_count} GenericTrackers are connected; "
                        f"found {len(current_tracker_names)} "
                        f"({', '.join(current_tracker_names)})"
                    )
                single_tracker_compat_ready = (
                    allow_single_tracker
                    and tracker_count == EXPECTED_TRACKER_COUNT
                    and len(current_tracker_names) == 1
                )
                if (
                    len(current_tracker_names) == tracker_count
                    or single_tracker_compat_ready
                ):
                    current_specs = auto_tracker_specs(
                        vive,
                        tracker_count=tracker_count,
                        allow_single_tracker=allow_single_tracker,
                    )
                    if current_specs != tracker_specs:
                        tracker_specs = current_specs
                        print(
                            "[Info] automatic tracker mapping updated: "
                            f"{tracker_mapping_text(tracker_specs)}"
                        )

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

            pose_sample_timestamp, pose_sample_monotonic_time = pose_sample_clocks()
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
                velocity_payload: dict[str, Any] = {
                    "linear_velocity_m_s": None,
                    "angular_velocity_rad_s": None,
                    "velocity_frame_name": "vive_world",
                    "velocity_source": "openvr_tracked_device_pose",
                    "velocity_valid": False,
                    "velocity_error": "pose_unavailable",
                }

                try:
                    if frame_error is not None:
                        raise RuntimeError(frame_error)
                    device = vive.devices[spec.device_name]
                    pose_matrix = openvr_pose_to_matrix4x4(
                        device.get_pose_matrix(pose_snapshot)
                    )
                    valid = True
                    velocity_payload = openvr_velocity_payload(device, pose_snapshot)
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
                        **velocity_payload,
                    }
                )

            if legacy_wire_format:
                packet = build_legacy_v1_tracking_packet(
                    seq=seq,
                    sender_timestamp=sender_timestamp,
                    trackers=trackers,
                )
            else:
                packet = build_v1_tracking_packet(
                    seq=seq,
                    sender_timestamp=sender_timestamp,
                    pose_sample_timestamp=pose_sample_timestamp,
                    pose_sample_monotonic_time=pose_sample_monotonic_time,
                    trackers=trackers,
                )
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
                velocity_valid_count = sum(
                    1 for tracker in trackers if tracker["velocity_valid"]
                )
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
                    f"valid={valid_count}/{len(trackers)} "
                    f"velocity={velocity_valid_count}/{len(trackers)} "
                    f"bytes={len(payload)} "
                    f"send_errors={send_error_total}"
                    f"{timing_suffix}"
                )
                for tracker in trackers:
                    error = tracker["error"]
                    suffix = "" if error is None else f" error={error}"
                    velocity_suffix = " velocity=unavailable"
                    if tracker["velocity_valid"]:
                        linear_speed = math.sqrt(
                            sum(
                                float(value) ** 2
                                for value in tracker["linear_velocity_m_s"]
                            )
                        )
                        angular_speed = math.sqrt(
                            sum(
                                float(value) ** 2
                                for value in tracker["angular_velocity_rad_s"]
                            )
                        )
                        velocity_suffix = (
                            f" |v|={linear_speed:.3f}m/s"
                            f" |w|={angular_speed:.3f}rad/s"
                        )
                    elif tracker.get("velocity_error"):
                        velocity_suffix += (
                            f" velocity_error={tracker['velocity_error']}"
                        )
                    print(
                        f"[{tracker['tracker_name']}:{tracker['tracker_role']}] "
                        f"valid={tracker['valid']} "
                        f"read={float(tracker.get('read_elapsed_ms', 0.0)):.2f}ms "
                        f"pos={format_position(tracker['pose_matrix_4x4'])}"
                        f"{velocity_suffix}{suffix}"
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
        if clock_sync_client is not None:
            clock_sync_client.close()
        close_udp_sender_socket(sock)


if __name__ == "__main__":
    main()
