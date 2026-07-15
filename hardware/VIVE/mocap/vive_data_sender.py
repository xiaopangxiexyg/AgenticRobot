from __future__ import annotations

import argparse
import importlib.util
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .clock_sync import (
        ClockSyncResult,
        ClockSyncSampleStore,
        make_probe,
        make_result,
        measurement_from_result,
        new_sender_instance_id,
        parse_reply,
    )


_CLOCK_SYNC_IMPORT_ERROR: ImportError | None = None
try:  # Support both ``python -m`` and direct Windows script execution.
    from .clock_sync import (
        ClockSyncResult,
        ClockSyncSampleStore,
        make_probe,
        make_result,
        measurement_from_result,
        new_sender_instance_id,
        parse_reply,
    )
except ImportError:
    try:
        from clock_sync import (  # type: ignore[no-redef]
            ClockSyncResult,
            ClockSyncSampleStore,
            make_probe,
            make_result,
            measurement_from_result,
            new_sender_instance_id,
            parse_reply,
        )
    except ImportError as exc:  # Keep legacy --clock-sync-mode off standalone.
        _CLOCK_SYNC_IMPORT_ERROR = exc


EXPECTED_TRACKER_COUNT = 2

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
# ``vive_data_sender.py`` is the local/new entry point.  The separately named
# ``vive_data_sender_legacy.py`` entry point pins this to ``off`` instead.
DEFAULT_CLOCK_SYNC_MODE = "optional"


@dataclass(frozen=True)
class TrackerSpec:
    device_name: str
    tracker_role: str


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
            "Read exactly two VIVE GenericTrackers via triad_openvr, assign them by "
            "device-name order, and send structured UDP JSON packets to the teleop runtime receiver."
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
        help="Seconds to wait for exactly two GenericTrackers before sending.",
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


def auto_tracker_specs(vive: Any) -> list[TrackerSpec]:
    """Assign the two discovered trackers to left/right by device name."""

    names = discovered_tracker_names(vive)
    if len(names) != EXPECTED_TRACKER_COUNT:
        available = ", ".join(names) or "<none>"
        raise ValueError(
            "exactly 2 GenericTrackers are required; "
            f"found {len(names)} ({available})"
        )
    return [
        TrackerSpec(device_name=names[0], tracker_role="left_wrist"),
        TrackerSpec(device_name=names[1], tracker_role="right_wrist"),
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


def wait_for_tracker_pair(
    vive: Any,
    timeout_sec: float,
) -> list[TrackerSpec] | None:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    next_status = 0.0
    while time.monotonic() < deadline:
        names = discovered_tracker_names(vive)
        if len(names) > EXPECTED_TRACKER_COUNT:
            raise ValueError(
                "more than 2 GenericTrackers are connected; "
                f"found {len(names)} ({', '.join(names)})"
            )
        if len(names) == EXPECTED_TRACKER_COUNT:
            return auto_tracker_specs(vive)
        now = time.monotonic()
        if now >= next_status:
            print(
                f"[Wait] waiting for exactly {EXPECTED_TRACKER_COUNT} GenericTrackers; "
                f"found {len(names)} ({', '.join(names) or '<none>'})"
            )
            next_status = now + 1.0
        if hasattr(vive, "poll_vr_events"):
            vive.poll_vr_events()
        time.sleep(0.1)
    return None


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


def canonical_triad_openvr_module_path() -> Path:
    """Return the one repository-owned OpenVR implementation used by the sender."""
    return Path(__file__).resolve().parents[1] / "triad_openvr" / "triad_openvr.py"


def load_triad_openvr(module_path: Path | None = None) -> Any:
    # Load by an explicit file path so direct Windows execution and ``python -m``
    # use the same implementation.  In particular, never prefer the obsolete
    # hardware/VIVE/triad_openvr.py file or an unrelated installed package.
    resolved_path = (
        canonical_triad_openvr_module_path()
        if module_path is None
        else Path(module_path).resolve()
    )
    if not resolved_path.is_file():
        raise SystemExit(
            "[Error] canonical triad_openvr implementation is missing: "
            f"{resolved_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "dexteleop_triad_openvr",
        resolved_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            "[Error] cannot load canonical triad_openvr implementation: "
            f"{resolved_path}"
        )

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "openvr"
        raise SystemExit(
            f"[Error] failed to import '{missing}' from {resolved_path}. "
            "Install the VIVE sender dependencies in this Python environment, then retry."
        ) from exc
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

    try:
        if args.wait_for_trackers > 0.0:
            tracker_specs = wait_for_tracker_pair(vive, args.wait_for_trackers)
            if tracker_specs is None:
                raise ValueError(
                    f"did not find exactly {EXPECTED_TRACKER_COUNT} GenericTrackers "
                    f"within {args.wait_for_trackers:.1f}s"
                )
        else:
            tracker_specs = auto_tracker_specs(vive)
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
                if len(current_tracker_names) > EXPECTED_TRACKER_COUNT:
                    raise SystemExit(
                        "[Error] more than 2 GenericTrackers are connected; "
                        f"found {len(current_tracker_names)} "
                        f"({', '.join(current_tracker_names)})"
                    )
                if len(current_tracker_names) == EXPECTED_TRACKER_COUNT:
                    current_specs = auto_tracker_specs(vive)
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
