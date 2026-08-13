"""Low-jitter, single-file VIVE pose sender.

Copy this file by itself to the SteamVR host.  Its only non-standard runtime
dependency is the ``openvr`` Python package.  It deliberately talks to OpenVR
directly: no repository driver, numpy, transforms package, or DexTeleop import
is required.

The pose data plane remains ``teleop_runtime.vive/v1`` on UDP port 9999.  The
optional clock-control plane remains separate on UDP port 10001.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


POSE_PROTOCOL = "teleop_runtime.vive/v1"
CLOCK_PROTOCOL = "teleop_runtime.vive.clock_sync/v1"
UNAVAILABLE_RIGHT_TRACKER = "<right_wrist_unavailable>"

DEFAULT_HOST = "192.168.0.2"
DEFAULT_POSE_PORT = 9999
DEFAULT_CLOCK_PORT = 10001
DEFAULT_HZ = 300.0
DEFAULT_EVENT_POLL_HZ = 30.0
DEFAULT_DEVICE_REFRESH_SEC = 1.0
DEFAULT_SPIN_THRESHOLD_MS = 0.3


def _finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_port(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not 1 <= result <= 65535:
        raise ValueError(f"{label} must be in [1, 65535]")
    return result


def _tracker_count(value: str) -> int:
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError("must be at least 2; use --single for one tracker")
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dependency-light VIVE GenericTracker sender. It uses OpenVR directly, "
            "takes one pose snapshot per frame, and emits teleop_runtime.vive/v1 UDP JSON."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Pose receiver host")
    parser.add_argument("--port", type=int, default=DEFAULT_POSE_PORT)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument(
        "--print-hz",
        type=float,
        default=1.0,
        help="Asynchronous status rate; use 0 for a completely quiet sender",
    )
    parser.add_argument(
        "--spin-threshold-ms",
        type=float,
        default=DEFAULT_SPIN_THRESHOLD_MS,
        help="Busy-wait tail used by the deadline scheduler; 0 disables it",
    )
    parser.add_argument(
        "--event-poll-hz",
        type=float,
        default=DEFAULT_EVENT_POLL_HZ,
        help="OpenVR activation/deactivation event poll rate",
    )
    parser.add_argument(
        "--device-refresh-sec",
        type=float,
        default=DEFAULT_DEVICE_REFRESH_SEC,
        help="Fallback full registry refresh period; 0 disables periodic refresh",
    )
    parser.add_argument(
        "--tracker-count",
        type=_tracker_count,
        default=2,
        help="Exact GenericTracker count; first two map left/right by stable name",
    )
    parser.add_argument(
        "--single",
        dest="allow_single_tracker",
        action="store_true",
        help="Allow one real tracker as left_wrist and emit invalid right_wrist",
    )
    parser.add_argument(
        "--wait-for-trackers",
        type=float,
        default=0.0,
        help="Wait this many seconds for the required tracker cardinality",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON device config with devices[{serial,name,type}]",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--send-buffer-bytes",
        type=int,
        default=0,
        help="Requested SO_SNDBUF; 0 keeps the OS default to avoid hidden large queues",
    )
    parser.add_argument(
        "--socket-retry-sec",
        type=float,
        default=0.25,
        help="Non-blocking cadence for reopening a failed UDP socket",
    )
    parser.add_argument(
        "--clock-sync-mode",
        choices=("off", "optional", "required"),
        default="optional",
    )
    parser.add_argument("--clock-sync-host", default="")
    parser.add_argument("--clock-sync-port", type=int, default=DEFAULT_CLOCK_PORT)
    parser.add_argument("--clock-sync-probes", type=int, default=16)
    parser.add_argument("--clock-sync-min-samples", type=int, default=4)
    parser.add_argument("--clock-sync-timeout-ms", type=float, default=100.0)
    parser.add_argument("--clock-sync-max-rtt-ms", type=float, default=25.0)
    parser.add_argument("--clock-sync-refresh-sec", type=float, default=5.0)
    parser.add_argument("--clock-sync-startup-timeout-sec", type=float, default=5.0)
    return parser.parse_args(argv)


def _json_bytes(message: Mapping[str, Any]) -> bytes:
    return json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _nonnegative_int(message: Mapping[str, Any], key: str) -> int | None:
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


class ClockSyncClient:
    """Independent best-effort clock mapping; never blocks pose sampling."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        startup_probes: int,
        min_samples: int,
        timeout_sec: float,
        max_rtt_sec: float,
        refresh_sec: float,
    ) -> None:
        self.host = host
        self.port = port
        self.startup_probes = max(1, startup_probes)
        self.min_samples = max(1, min_samples)
        self.timeout_sec = max(0.001, timeout_sec)
        self.max_rtt_ns = None if max_rtt_sec <= 0 else int(max_rtt_sec * 1e9)
        self.refresh_sec = max(0.05, refresh_sec)
        self.sender_instance_id = uuid.uuid4().hex
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._probe_id = 0
        self._accepted = 0
        self._timeouts = 0
        self._rejected = 0
        self._errors = 0
        self._best_rtt_ns: int | None = None
        self._best_offset_ns: int | None = None
        self._recent_samples: list[tuple[int, int]] = []

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((self.host, self.port))
        sock.settimeout(self.timeout_sec)
        self._socket = sock
        self._thread = threading.Thread(
            target=self._run,
            name="vive-stable-clock-sync",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.timeout_sec * 2.0))
            self._thread = None

    def wait_ready(self, timeout_sec: float) -> bool:
        return self._ready.wait(max(0.0, timeout_sec))

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "ready": self._ready.is_set(),
                "accepted": self._accepted,
                "timeouts": self._timeouts,
                "rejected": self._rejected,
                "errors": self._errors,
                "best_rtt_ms": (
                    None if self._best_rtt_ns is None else self._best_rtt_ns / 1e6
                ),
                "collector_minus_sender_ms": (
                    None if self._best_offset_ns is None else self._best_offset_ns / 1e6
                ),
            }

    def _run(self) -> None:
        startup_left = self.startup_probes
        while not self._stop.is_set():
            self._probe_once()
            if startup_left > 0:
                startup_left -= 1
                wait_sec = 0.02
            elif not self._ready.is_set():
                wait_sec = min(1.0, self.refresh_sec)
            else:
                wait_sec = self.refresh_sec
            self._stop.wait(wait_sec)

    def _probe_once(self) -> None:
        sock = self._socket
        if sock is None:
            return
        probe_id = self._probe_id
        self._probe_id += 1
        t1 = time.perf_counter_ns()
        probe = {
            "protocol": CLOCK_PROTOCOL,
            "type": "probe",
            "sender_instance_id": self.sender_instance_id,
            "probe_id": probe_id,
            "t1_sender_monotonic_ns": t1,
        }
        try:
            sock.send(_json_bytes(probe))
            packet = sock.recv(4096)
        except socket.timeout:
            with self._lock:
                self._timeouts += 1
            return
        except OSError:
            if not self._stop.is_set():
                with self._lock:
                    self._errors += 1
            return
        t4 = time.perf_counter_ns()
        try:
            reply = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            reply = None
        if not isinstance(reply, dict):
            self._reject()
            return
        t2 = _nonnegative_int(reply, "t2_collector_monotonic_ns")
        t3 = _nonnegative_int(reply, "t3_collector_monotonic_ns")
        if (
            reply.get("protocol") != CLOCK_PROTOCOL
            or reply.get("type") != "reply"
            or reply.get("sender_instance_id") != self.sender_instance_id
            or _nonnegative_int(reply, "probe_id") != probe_id
            or _nonnegative_int(reply, "t1_sender_monotonic_ns") != t1
            or t2 is None
            or t3 is None
            or t3 < t2
        ):
            self._reject()
            return
        round_trip_ns = (t4 - t1) - (t3 - t2)
        if round_trip_ns < 0 or (
            self.max_rtt_ns is not None and round_trip_ns > self.max_rtt_ns
        ):
            self._reject()
            return
        offset_ns = int(round(((t2 - t1) + (t3 - t4)) / 2.0))
        result = {
            "protocol": CLOCK_PROTOCOL,
            "type": "result",
            "sender_instance_id": self.sender_instance_id,
            "probe_id": probe_id,
            "t1_sender_monotonic_ns": t1,
            "t2_collector_monotonic_ns": t2,
            "t3_collector_monotonic_ns": t3,
            "t4_sender_monotonic_ns": t4,
        }
        try:
            sock.send(_json_bytes(result))
        except OSError:
            with self._lock:
                self._errors += 1
            return
        with self._lock:
            self._accepted += 1
            self._recent_samples.append((round_trip_ns, offset_ns))
            self._recent_samples = self._recent_samples[-8:]
            self._best_rtt_ns, self._best_offset_ns = min(self._recent_samples)
            if self._accepted >= self.min_samples:
                self._ready.set()

    def _reject(self) -> None:
        with self._lock:
            self._rejected += 1


def start_clock_sync(args: argparse.Namespace) -> ClockSyncClient | None:
    if args.clock_sync_mode == "off":
        return None
    host = str(args.clock_sync_host or args.host).strip()
    try:
        port = _positive_port(args.clock_sync_port, label="clock sync port")
        pose_port = _positive_port(args.port, label="pose port")
        if port == pose_port:
            raise ValueError("clock sync port must differ from pose port")
        if args.clock_sync_probes < 1 or args.clock_sync_min_samples < 1:
            raise ValueError("clock sync probes and minimum samples must be >= 1")
        client = ClockSyncClient(
            host=host,
            port=port,
            startup_probes=int(args.clock_sync_probes),
            min_samples=int(args.clock_sync_min_samples),
            timeout_sec=_finite_float(args.clock_sync_timeout_ms, label="clock timeout") / 1e3,
            max_rtt_sec=_finite_float(args.clock_sync_max_rtt_ms, label="clock max RTT") / 1e3,
            refresh_sec=_finite_float(args.clock_sync_refresh_sec, label="clock refresh"),
        )
        client.start()
    except (OSError, ValueError) as exc:
        if args.clock_sync_mode == "required":
            raise SystemExit(f"[Error] clock sync startup failed: {exc}") from exc
        print(f"[Warn] clock sync unavailable: {exc}; pose streaming will continue")
        return None
    if args.clock_sync_mode == "required" and not client.wait_ready(
        _finite_float(args.clock_sync_startup_timeout_sec, label="clock startup timeout")
    ):
        stats = client.stats()
        client.close()
        raise SystemExit(f"[Error] required clock sync did not become ready: {stats}")
    return client


@dataclass(frozen=True)
class DeviceRecord:
    index: int
    name: str
    serial: str
    model: str


@dataclass(frozen=True)
class TrackerSpec:
    name: str
    role: str


def _openvr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


class OpenVRTrackerRegistry:
    """Minimal GenericTracker registry with stable in-process serial names."""

    def __init__(self, config_path: str | None = None) -> None:
        try:
            import openvr  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "[Error] vive_data_sender_stable.py requires only the 'openvr' "
                "Python package in addition to the standard library."
            ) from exc
        self.openvr = openvr
        try:
            openvr.init(openvr.VRApplication_Other)
            self.system = openvr.VRSystem()
        except Exception as exc:
            raise SystemExit(f"[Error] failed to initialize SteamVR/OpenVR: {exc}") from exc
        self._closed = False
        self._configured_names = self._load_config(config_path)
        self._name_by_serial: dict[str, str] = dict(self._configured_names)
        self._records_by_name: dict[str, DeviceRecord] = {}

    @staticmethod
    def _load_config(config_path: str | None) -> dict[str, str]:
        if not config_path:
            return {}
        try:
            with Path(config_path).open("r", encoding="utf-8") as stream:
                config = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[Error] failed to load device config: {exc}") from exc
        if not isinstance(config, dict) or not isinstance(config.get("devices", []), list):
            raise SystemExit("[Error] device config must contain a devices list")
        names: dict[str, str] = {}
        for item in config.get("devices", []):
            if not isinstance(item, dict):
                continue
            serial = _openvr_text(item.get("serial")).strip()
            name = _openvr_text(item.get("name")).strip()
            device_type = _openvr_text(item.get("type", "Tracker")).strip()
            if serial and name and device_type == "Tracker":
                names[serial] = name
        if len(set(names.values())) != len(names):
            raise SystemExit("[Error] configured tracker names must be unique")
        return names

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.openvr.shutdown()
        except Exception:
            pass

    def snapshot(self) -> object:
        return self.system.getDeviceToAbsoluteTrackingPose(
            self.openvr.TrackingUniverseStanding,
            0,
            self.openvr.k_unMaxTrackedDeviceCount,
        )

    def poll_events(self) -> bool:
        """Drain events and report registry dirtiness without scanning devices."""

        dirty = False
        event = self.openvr.VREvent_t()
        while self.system.pollNextEvent(event):
            if event.eventType in {
                self.openvr.VREvent_TrackedDeviceActivated,
                self.openvr.VREvent_TrackedDeviceDeactivated,
            }:
                dirty = True
        return dirty

    def refresh(self, snapshot: object) -> int:
        """Refresh from the frame's existing snapshot; never fetch a second pose."""

        candidates: list[tuple[int, str, str]] = []
        for index in range(self.openvr.k_unMaxTrackedDeviceCount):
            try:
                tracked_pose = snapshot[index]  # type: ignore[index]
                if not tracked_pose.bDeviceIsConnected:
                    continue
                if self.system.getTrackedDeviceClass(index) != self.openvr.TrackedDeviceClass_GenericTracker:
                    continue
                serial = _openvr_text(
                    self.system.getStringTrackedDeviceProperty(
                        index,
                        self.openvr.Prop_SerialNumber_String,
                    )
                ).strip()
                model = _openvr_text(
                    self.system.getStringTrackedDeviceProperty(
                        index,
                        self.openvr.Prop_ModelNumber_String,
                    )
                ).strip()
            except Exception:
                continue
            # SteamVR trackers normally expose a serial.  Keep an index-scoped
            # fallback so multiple malformed devices never collapse into one
            # registry entry.
            candidates.append((index, serial or f"<index:{index}>", model))

        used_names = set(self._name_by_serial.values())
        next_suffix = 1
        # Preserve the established first-seen OpenVR index convention while
        # remembering the assigned name by serial for reconnects.
        for _index, serial, _model in sorted(candidates, key=lambda item: item[0]):
            if serial in self._name_by_serial:
                continue
            while f"tracker_{next_suffix}" in used_names:
                next_suffix += 1
            name = f"tracker_{next_suffix}"
            self._name_by_serial[serial] = name
            used_names.add(name)

        new_records = {
            self._name_by_serial[serial]: DeviceRecord(
                index=index,
                name=self._name_by_serial[serial],
                serial=serial,
                model=model,
            )
            for index, serial, model in candidates
        }
        old_identity = {
            name: (record.index, record.serial) for name, record in self._records_by_name.items()
        }
        new_identity = {
            name: (record.index, record.serial) for name, record in new_records.items()
        }
        changes = len(set(old_identity.items()) ^ set(new_identity.items()))
        self._records_by_name = new_records
        return changes

    @property
    def tracker_names(self) -> list[str]:
        return sorted(self._records_by_name, key=lambda name: (name.casefold(), name))

    def record(self, name: str) -> DeviceRecord:
        return self._records_by_name[name]

    def describe(self) -> list[str]:
        return [
            f"{record.name} index={record.index} serial={record.serial or '<unknown>'} "
            f"model={record.model or '<unknown>'}"
            for record in sorted(self._records_by_name.values(), key=lambda item: item.name)
        ]


def tracker_role(index: int) -> str:
    if index == 0:
        return "left_wrist"
    if index == 1:
        return "right_wrist"
    if index == 2:
        return "auxiliary"
    return f"auxiliary_{index - 1}"


def select_trackers(
    registry: OpenVRTrackerRegistry,
    *,
    tracker_count: int,
    allow_single_tracker: bool,
) -> list[TrackerSpec]:
    names = registry.tracker_names
    if allow_single_tracker and tracker_count == 2 and len(names) == 1:
        return [
            TrackerSpec(names[0], "left_wrist"),
            TrackerSpec(UNAVAILABLE_RIGHT_TRACKER, "right_wrist"),
        ]
    if len(names) != tracker_count:
        raise ValueError(
            f"exactly {tracker_count} GenericTrackers are required; "
            f"found {len(names)} ({', '.join(names) or '<none>'})"
        )
    return [TrackerSpec(name, tracker_role(index)) for index, name in enumerate(names)]


def wait_for_trackers(
    registry: OpenVRTrackerRegistry,
    *,
    timeout_sec: float,
    tracker_count: int,
    allow_single_tracker: bool,
) -> list[TrackerSpec]:
    deadline = time.perf_counter() + max(0.0, timeout_sec)
    next_status = 0.0
    while True:
        snapshot = registry.snapshot()
        registry.poll_events()
        registry.refresh(snapshot)
        try:
            return select_trackers(
                registry,
                tracker_count=tracker_count,
                allow_single_tracker=allow_single_tracker,
            )
        except ValueError:
            now = time.perf_counter()
            if now >= deadline:
                raise
            if now >= next_status:
                print(
                    f"[Wait] trackers={len(registry.tracker_names)} "
                    f"names={registry.tracker_names or ['<none>']}"
                )
                next_status = now + 1.0
            time.sleep(min(0.1, max(0.0, deadline - now)))


def _matrix4x4(tracked_pose: object) -> list[list[float]]:
    if not getattr(tracked_pose, "bPoseIsValid", False):
        raise ValueError("OpenVR pose is invalid")
    matrix = tracked_pose.mDeviceToAbsoluteTracking
    values = [[float(matrix[row][column]) for column in range(4)] for row in range(3)]
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError("OpenVR pose contains non-finite values")
    return [*values, [0.0, 0.0, 0.0, 1.0]]


def _vector3(value: object, *, label: str) -> list[float]:
    raw = getattr(value, "v", value)
    result = [float(raw[index]) for index in range(3)]  # type: ignore[index]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} contains non-finite values")
    return result


def _identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def tracker_payload(
    registry: OpenVRTrackerRegistry,
    snapshot: object,
    spec: TrackerSpec,
) -> dict[str, object]:
    started = time.perf_counter()
    pose = _identity_matrix()
    valid = False
    error: str | None = None
    linear_velocity: list[float] | None = None
    angular_velocity: list[float] | None = None
    velocity_error: str | None = "pose_unavailable"
    try:
        record = registry.record(spec.name)
        tracked_pose = snapshot[record.index]  # type: ignore[index]
        pose = _matrix4x4(tracked_pose)
        valid = True
        try:
            linear_velocity = _vector3(tracked_pose.vVelocity, label="linear velocity")
            angular_velocity = _vector3(
                tracked_pose.vAngularVelocity,
                label="angular velocity",
            )
            velocity_error = None
        except Exception as exc:
            velocity_error = f"{type(exc).__name__}: {exc}"
    except KeyError:
        error = f"device '{spec.name}' not found"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1e3
    return {
        "tracker_name": spec.name,
        "tracker_role": spec.role,
        "valid": valid,
        "pose_matrix_4x4": pose,
        "error": error,
        "read_elapsed_ms": elapsed_ms,
        "linear_velocity_m_s": linear_velocity,
        "angular_velocity_rad_s": angular_velocity,
        "velocity_frame_name": "vive_world",
        "velocity_source": "openvr_tracked_device_pose",
        "velocity_valid": velocity_error is None,
        "velocity_error": velocity_error,
    }


def invalid_tracker_payload(spec: TrackerSpec, error: str) -> dict[str, object]:
    return {
        "tracker_name": spec.name,
        "tracker_role": spec.role,
        "valid": False,
        "pose_matrix_4x4": _identity_matrix(),
        "error": error,
        "read_elapsed_ms": 0.0,
        "linear_velocity_m_s": None,
        "angular_velocity_rad_s": None,
        "velocity_frame_name": "vive_world",
        "velocity_source": "openvr_tracked_device_pose",
        "velocity_valid": False,
        "velocity_error": "pose_unavailable",
    }


def build_tracking_packet(
    *,
    seq: int,
    sender_timestamp: float,
    pose_sample_timestamp: float,
    pose_sample_monotonic_time: float,
    trackers: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "protocol": POSE_PROTOCOL,
        "seq": int(seq),
        "sender_timestamp": float(sender_timestamp),
        "pose_sample_timestamp": float(pose_sample_timestamp),
        "pose_sample_monotonic_time": float(pose_sample_monotonic_time),
        "trackers": trackers,
    }


class DeadlineScheduler:
    """Absolute-deadline scheduler that drops missed ticks instead of catching up."""

    def __init__(self, hz: float, *, spin_threshold_sec: float) -> None:
        self.period_sec = 0.0 if hz <= 0.0 else 1.0 / hz
        self.spin_threshold_sec = max(0.0, spin_threshold_sec)
        self.next_deadline = time.perf_counter()
        self.missed_total = 0

    def wait(self) -> tuple[float, int]:
        if self.period_sec <= 0.0:
            return 0.0, 0
        while True:
            remaining = self.next_deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            if remaining > self.spin_threshold_sec:
                time.sleep(max(0.0, remaining - self.spin_threshold_sec))
            else:
                # The short spin avoids Windows timer-quantum oversleep.  It is
                # bounded by --spin-threshold-ms and can be disabled.
                pass
        now = time.perf_counter()
        lateness_sec = max(0.0, now - self.next_deadline)
        missed = int(lateness_sec // self.period_sec)
        self.missed_total += missed
        self.next_deadline += (missed + 1) * self.period_sec
        return lateness_sec, missed


class PoseDatagramSender:
    """Freshness-first UDP sender with non-blocking reopen cadence."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        send_buffer_bytes: int,
        retry_sec: float,
    ) -> None:
        self.host = host
        self.port = port
        self.send_buffer_bytes = max(0, send_buffer_bytes)
        self.retry_sec = max(0.0, retry_sec)
        self.socket: socket.socket | None = None
        self.next_retry_at = 0.0
        self.sent = 0
        self.dropped = 0
        self.errors = 0
        self.recoveries = 0
        self.last_error: str | None = None

    def close(self) -> None:
        sock, self.socket = self.socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _open(self) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if self.send_buffer_bytes > 0:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_bytes)
            sock.connect((self.host, self.port))
        except OSError as exc:
            try:
                sock.close()
            except OSError:
                pass
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.errors += 1
            return False
        if self.errors:
            self.recoveries += 1
        self.socket = sock
        return True

    def send(self, payload: bytes, *, now: float) -> tuple[bool, float]:
        started = time.perf_counter()
        if self.socket is None:
            if now < self.next_retry_at or not self._open():
                self.dropped += 1
                self.next_retry_at = max(self.next_retry_at, now + self.retry_sec)
                return False, (time.perf_counter() - started) * 1e3
        try:
            assert self.socket is not None
            sent_bytes = self.socket.send(payload)
            if sent_bytes != len(payload):
                raise OSError(f"partial UDP datagram: {sent_bytes}/{len(payload)} bytes")
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.errors += 1
            self.dropped += 1
            self.close()
            self.next_retry_at = now + self.retry_sec
            return False, (time.perf_counter() - started) * 1e3
        self.sent += 1
        return True, (time.perf_counter() - started) * 1e3


class AsyncStatusPrinter:
    """Latest-only console publisher so a slow terminal cannot stall sampling."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[list[str] | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="vive-stable-status",
                daemon=True,
            )
            self._thread.start()

    def submit(self, lines: list[str]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(lines)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(lines)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while True:
            lines = self._queue.get()
            if lines is None:
                return
            for line in lines:
                print(line, flush=True)


class IntervalMetrics:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.frames = 0
        self.sent = 0
        self.missed = 0
        self.maxima: dict[str, float] = {}

    def add(self, *, sent: bool, missed: int, **timings_ms: float) -> None:
        self.frames += 1
        self.sent += int(sent)
        self.missed += missed
        for name, value in timings_ms.items():
            self.maxima[name] = max(self.maxima.get(name, 0.0), float(value))

    def take(self, now: float) -> dict[str, object]:
        elapsed = max(now - self.started_at, 1e-9)
        result: dict[str, object] = {
            "rate_hz": self.frames / elapsed,
            "send_rate_hz": self.sent / elapsed,
            "frames": self.frames,
            "missed": self.missed,
            **{f"max_{name}_ms": value for name, value in self.maxima.items()},
        }
        self.started_at = now
        self.frames = 0
        self.sent = 0
        self.missed = 0
        self.maxima = {}
        return result


def _status_lines(
    *,
    seq: int,
    metrics: dict[str, object],
    transport: PoseDatagramSender,
    tracker_specs: list[TrackerSpec],
    trackers: list[dict[str, object]],
    clock_sync: ClockSyncClient | None,
) -> list[str]:
    lines = [
        "[Status] "
        f"seq={seq} rate={float(metrics['rate_hz']):.1f}Hz "
        f"send_ok={float(metrics['send_rate_hz']):.1f}Hz "
        f"missed={int(metrics['missed'])} "
        f"late_max={float(metrics.get('max_late_ms', 0.0)):.3f}ms "
        f"loop_max={float(metrics.get('max_loop_ms', 0.0)):.3f}ms "
        f"pose_max={float(metrics.get('max_pose_ms', 0.0)):.3f}ms "
        f"registry_max={float(metrics.get('max_registry_ms', 0.0)):.3f}ms "
        f"json_max={float(metrics.get('max_json_ms', 0.0)):.3f}ms "
        f"send_max={float(metrics.get('max_send_ms', 0.0)):.3f}ms "
        f"udp_errors={transport.errors} udp_dropped={transport.dropped}"
    ]
    if transport.last_error:
        lines.append(f"[UDP] last_error={transport.last_error}")
    if clock_sync is not None:
        lines.append(f"[ClockSync] {clock_sync.stats()}")
    by_name = {str(item["tracker_name"]): item for item in trackers}
    for spec in tracker_specs:
        item = by_name.get(spec.name, {})
        pose = item.get("pose_matrix_4x4", _identity_matrix())
        position = [float(pose[index][3]) for index in range(3)]  # type: ignore[index]
        lines.append(
            f"[{spec.name}:{spec.role}] valid={bool(item.get('valid', False))} "
            f"pos_m=[{position[0]:.3f},{position[1]:.3f},{position[2]:.3f}] "
            f"error={item.get('error')}"
        )
    return lines


def _validate_runtime_args(args: argparse.Namespace) -> None:
    _positive_port(args.port, label="pose port")
    for name in (
        "hz",
        "print_hz",
        "spin_threshold_ms",
        "event_poll_hz",
        "device_refresh_sec",
        "wait_for_trackers",
        "socket_retry_sec",
    ):
        value = _finite_float(getattr(args, name), label=name)
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        _validate_runtime_args(args)
    except ValueError as exc:
        raise SystemExit(f"[Error] {exc}") from exc

    registry = OpenVRTrackerRegistry(args.config)
    transport: PoseDatagramSender | None = None
    clock_sync: ClockSyncClient | None = None
    status = AsyncStatusPrinter(enabled=float(args.print_hz) > 0.0)
    try:
        initial_snapshot = registry.snapshot()
        registry.refresh(initial_snapshot)
        print(f"[Devices] found {len(registry.tracker_names)} GenericTracker(s)")
        for line in registry.describe():
            print(f"  {line}")
        if args.list_devices:
            return

        try:
            if float(args.wait_for_trackers) > 0.0:
                tracker_specs = wait_for_trackers(
                    registry,
                    timeout_sec=float(args.wait_for_trackers),
                    tracker_count=int(args.tracker_count),
                    allow_single_tracker=bool(args.allow_single_tracker),
                )
            else:
                tracker_specs = select_trackers(
                    registry,
                    tracker_count=int(args.tracker_count),
                    allow_single_tracker=bool(args.allow_single_tracker),
                )
        except ValueError as exc:
            raise SystemExit(f"[Error] {exc}") from exc

        transport = PoseDatagramSender(
            str(args.host),
            int(args.port),
            send_buffer_bytes=int(args.send_buffer_bytes),
            retry_sec=float(args.socket_retry_sec),
        )
        clock_sync = start_clock_sync(args)
        scheduler = DeadlineScheduler(
            float(args.hz),
            spin_threshold_sec=float(args.spin_threshold_ms) / 1e3,
        )
        event_period = 0.0 if args.event_poll_hz <= 0.0 else 1.0 / args.event_poll_hz
        refresh_period = max(0.0, float(args.device_refresh_sec))
        next_event_poll = time.perf_counter()
        next_refresh = time.perf_counter() + refresh_period
        print_period = 0.0 if args.print_hz <= 0.0 else 1.0 / args.print_hz
        next_status = time.perf_counter() + print_period
        metrics = IntervalMetrics()
        seq = 0
        trackers: list[dict[str, object]] = []

        print(
            f"[Start] stable VIVE sender -> {args.host}:{args.port} "
            f"rate={'max' if args.hz <= 0 else f'{args.hz:g}Hz'} "
            f"mapping={[(spec.name, spec.role) for spec in tracker_specs]}",
            flush=True,
        )
        while True:
            lateness_sec, missed = scheduler.wait()
            loop_started = time.perf_counter()

            registry_started = time.perf_counter()
            event_dirty = False
            if event_period <= 0.0 or loop_started >= next_event_poll:
                try:
                    event_dirty = registry.poll_events()
                except Exception:
                    event_dirty = True
                next_event_poll = loop_started + event_period

            pose_started = time.perf_counter()
            try:
                snapshot = registry.snapshot()
                frame_error = None
            except Exception as exc:
                snapshot = None
                frame_error = f"{type(exc).__name__}: {exc}"
            pose_sample_timestamp = time.time()
            pose_sample_monotonic_time = time.perf_counter()
            pose_elapsed_ms = (pose_sample_monotonic_time - pose_started) * 1e3

            refresh_due = refresh_period > 0.0 and loop_started >= next_refresh
            if snapshot is not None and (event_dirty or refresh_due):
                registry.refresh(snapshot)
                if refresh_due:
                    next_refresh = loop_started + refresh_period
                names = registry.tracker_names
                if len(names) > int(args.tracker_count):
                    raise RuntimeError(
                        f"more than {args.tracker_count} GenericTrackers are connected: {names}"
                    )
                cardinality_ready = len(names) == int(args.tracker_count) or (
                    bool(args.allow_single_tracker)
                    and int(args.tracker_count) == 2
                    and len(names) == 1
                )
                if cardinality_ready:
                    tracker_specs = select_trackers(
                        registry,
                        tracker_count=int(args.tracker_count),
                        allow_single_tracker=bool(args.allow_single_tracker),
                    )
            registry_elapsed_ms = (time.perf_counter() - registry_started) * 1e3

            trackers = (
                [tracker_payload(registry, snapshot, spec) for spec in tracker_specs]
                if snapshot is not None
                else [
                    invalid_tracker_payload(spec, frame_error or "OpenVR snapshot unavailable")
                    for spec in tracker_specs
                ]
            )
            packet = build_tracking_packet(
                seq=seq,
                sender_timestamp=time.time(),
                pose_sample_timestamp=pose_sample_timestamp,
                pose_sample_monotonic_time=pose_sample_monotonic_time,
                trackers=trackers,
            )
            json_started = time.perf_counter()
            payload = _json_bytes(packet)
            json_elapsed_ms = (time.perf_counter() - json_started) * 1e3
            sent, send_elapsed_ms = transport.send(payload, now=time.perf_counter())
            seq += 1
            loop_elapsed_ms = (time.perf_counter() - loop_started) * 1e3
            metrics.add(
                sent=sent,
                missed=missed,
                late_ms=lateness_sec * 1e3,
                loop_ms=loop_elapsed_ms,
                pose_ms=pose_elapsed_ms,
                registry_ms=registry_elapsed_ms,
                json_ms=json_elapsed_ms,
                send_ms=send_elapsed_ms,
            )

            now = time.perf_counter()
            if print_period > 0.0 and now >= next_status:
                status.submit(
                    _status_lines(
                        seq=seq,
                        metrics=metrics.take(now),
                        transport=transport,
                        tracker_specs=tracker_specs,
                        trackers=trackers,
                        clock_sync=clock_sync,
                    )
                )
                next_status = now + print_period
    except KeyboardInterrupt:
        print("\n[Exit] stopped by user")
    finally:
        status.close()
        if clock_sync is not None:
            clock_sync.close()
        if transport is not None:
            transport.close()
        registry.close()


if __name__ == "__main__":
    main()
