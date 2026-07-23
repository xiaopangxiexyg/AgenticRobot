#!/usr/bin/env python3
"""Record RM65 UDP feedback for offline 120 Hz actuator identification.

This tool is intentionally independent from teleop_runtime.  It never sends a
motion command.  During ``capture`` it temporarily redirects each controller's
UDP realtime push to a local socket, records every raw datagram with wall and
monotonic receive timestamps, and then restores the exact original controller
configuration.

Recovery is deliberately fail-closed:

* the complete original configuration is journaled and fsync'ed before writes;
* a no-op set/get round trip must succeed for both controllers;
* every temporary configuration and restoration is verified by get/readback;
* a detached guardian restores the journal if the parent dies unexpectedly;
* a later invocation restores every unfinished journal before it may capture.

The default ``full`` format retains every raw UDP payload as base64.  The
``compact`` format keeps fixed promoted joint fields and retains raw JSON only
for the first valid schema sample or an error/unexpected datagram.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import copy
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Protocol


# Direct execution (``python hardware/.../rm65_udp_actuator_trace.py``) puts
# only this script's directory on sys.path.  Add the repository root so the
# lazily loaded ``hardware.rm65`` SDK is available without requiring callers
# to set PYTHONPATH.  This changes Python import lookup only; it performs no
# hardware access and is also valid when launched by the detached guardian.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRACE_SCHEMA = "rm65_udp_actuator_feedback_v1"
JOURNAL_SCHEMA = "rm65_udp_push_recovery_v1"
CUSTOM_CONFIG_FIELDS = (
    "joint_speed",
    "lift_state",
    "expand_state",
    "hand_state",
    "arm_current_status",
    "aloha_state",
    "plus_base",
    "plus_state",
)


class PushConfigClient(Protocol):
    def get_config(self) -> dict[str, Any]: ...

    def set_config(self, config: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ArmSpec:
    side: str
    arm_ip: str
    local_ip: str
    push_port: int


class RestoreError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def normalize_push_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete comparable subset accepted by the RM65 API."""

    custom_raw = raw.get("custom_config")
    custom = dict(custom_raw) if isinstance(custom_raw, Mapping) else {}
    missing = [
        key
        for key in ("cycle", "enable", "port", "force_coordinate", "ip")
        if key not in raw
    ]
    if missing:
        raise ValueError(f"UDP push config is incomplete; missing {missing}")
    normalized = {
        "cycle": int(raw["cycle"]),
        "enable": bool(raw["enable"]),
        "port": int(raw["port"]),
        "force_coordinate": int(raw["force_coordinate"]),
        "ip": str(_json_safe(raw["ip"])).rstrip("\x00"),
        "custom_config": {
            field: int(custom.get(field, -1)) for field in CUSTOM_CONFIG_FIELDS
        },
    }
    # The controller/API field is a count of 5 ms intervals, not a duration
    # expressed in milliseconds: cycle=1 means 5 ms / 200 Hz.
    if normalized["cycle"] <= 0:
        raise ValueError(
            f"controller returned unsafe/unrestorable cycle={normalized['cycle']}; "
            "expected a positive count of 5 ms intervals"
        )
    if not 1 <= normalized["port"] <= 65535:
        raise ValueError(f"controller returned invalid port={normalized['port']}")
    return normalized


def temporary_push_config(
    original: Mapping[str, Any],
    *,
    local_ip: str,
    push_port: int,
    cycle_ms: int,
) -> dict[str, Any]:
    if int(cycle_ms) <= 0 or int(cycle_ms) % 5 != 0:
        raise ValueError(
            f"cycle_ms={cycle_ms} is invalid; expected a positive multiple of 5 ms"
        )
    target = copy.deepcopy(normalize_push_config(original))
    original_joint_speed = int(target["custom_config"]["joint_speed"])
    if original_joint_speed not in (0, 1):
        raise ValueError(
            "controller did not expose an explicit original joint_speed setting; "
            "refusing capture because joint-speed reporting could not be restored "
            f"exactly (observed {original_joint_speed})"
        )
    target.update(
        {
            "cycle": int(cycle_ms) // 5,
            "enable": True,
            "port": int(push_port),
            "ip": str(local_ip),
        }
    )
    # Joint position/current are part of the base packet. Explicitly request
    # joint speed while preserving every unrelated custom field.
    target["custom_config"]["joint_speed"] = 1
    return normalize_push_config(target)


def push_configs_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return normalize_push_config(left) == normalize_push_config(right)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


class Api2PushConfigClient:
    """Small API2 adapter limited to get/set realtime-push configuration."""

    def __init__(self, arm_ip: str, *, port: int = 8080):
        from hardware.rm65.Robotic_Arm.rm_ctypes_wrap import rm_thread_mode_e
        from hardware.rm65.Robotic_Arm.rm_robot_interface import RoboticArm

        self.arm_ip = str(arm_ip)
        self._robot = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
        handle = self._robot.rm_create_robot_arm(self.arm_ip, int(port), level=3)
        if int(getattr(handle, "id", -1)) == -1:
            raise RuntimeError(f"failed to connect RM65 controller {self.arm_ip}")
        self._closed = False

    def get_config(self) -> dict[str, Any]:
        code, config = self._robot.rm_get_realtime_push()
        if int(code) != 0:
            raise RuntimeError(
                f"{self.arm_ip}: rm_get_realtime_push failed with code {code}"
            )
        if not isinstance(config, Mapping):
            raise RuntimeError(
                f"{self.arm_ip}: rm_get_realtime_push returned {type(config).__name__}"
            )
        return normalize_push_config(config)

    def set_config(self, config: Mapping[str, Any]) -> None:
        from hardware.rm65.Robotic_Arm.rm_ctypes_wrap import (
            rm_realtime_push_config_t,
            rm_udp_custom_config_t,
        )

        value = normalize_push_config(config)
        custom = rm_udp_custom_config_t(**value["custom_config"])
        payload = rm_realtime_push_config_t(
            cycle=value["cycle"],
            enable=value["enable"],
            port=value["port"],
            force_coordinate=value["force_coordinate"],
            ip=value["ip"],
            custom_config=custom,
        )
        code = self._robot.rm_set_realtime_push(payload)
        if int(code) != 0:
            raise RuntimeError(
                f"{self.arm_ip}: rm_set_realtime_push failed with code {code}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        code = self._robot.rm_delete_robot_arm()
        if int(code) != 0:
            raise RuntimeError(
                f"{self.arm_ip}: rm_delete_robot_arm failed with code {code}"
            )


@contextmanager
def _client_for(arm_ip: str):
    client = Api2PushConfigClient(arm_ip)
    try:
        yield client
    finally:
        client.close()


def query_config(arm_ip: str) -> dict[str, Any]:
    with _client_for(arm_ip) as client:
        return client.get_config()


def set_and_verify_config(
    arm_ip: str,
    expected: Mapping[str, Any],
    *,
    attempts: int = 3,
    retry_delay_sec: float = 0.25,
) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, int(attempts) + 1):
        try:
            with _client_for(arm_ip) as client:
                client.set_config(expected)
                observed = client.get_config()
            if not push_configs_equal(observed, expected):
                raise RuntimeError(
                    f"readback mismatch expected={normalize_push_config(expected)} "
                    f"observed={observed}"
                )
            return observed
        except Exception as exc:  # noqa: BLE001 - restoration retries are intentional.
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < int(attempts):
                time.sleep(float(retry_delay_sec))
    raise RestoreError(f"{arm_ip}: set/readback failed: {' | '.join(errors)}")


def _journal_arms(journal: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms = journal.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("recovery journal has no arms")
    out: list[dict[str, Any]] = []
    for item in arms:
        if not isinstance(item, Mapping):
            raise ValueError("invalid arm entry in recovery journal")
        out.append(dict(item))
    return out


def restore_journal(path: Path, *, actor: str) -> None:
    journal = _load_json(path)
    if journal.get("schema") != JOURNAL_SCHEMA:
        raise RestoreError(f"unsupported recovery journal schema in {path}")
    if journal.get("status") == "restored":
        return
    journal["status"] = "restoring"
    journal["restore_actor"] = str(actor)
    journal["restore_started_wall_time"] = time.time()
    _atomic_write_json(path, journal)
    failures: list[str] = []
    restored: dict[str, Any] = {}
    for item in reversed(_journal_arms(journal)):
        side = str(item["side"])
        try:
            restored[side] = set_and_verify_config(
                str(item["arm_ip"]), item["original_config"]
            )
        except Exception as exc:  # noqa: BLE001 - collect both arm failures.
            failures.append(f"{side}: {type(exc).__name__}: {exc}")
    journal = _load_json(path)
    journal["restore_finished_wall_time"] = time.time()
    journal["restored_config_by_side"] = restored
    if failures:
        journal["status"] = "restore_failed"
        journal["restore_failures"] = failures
        _atomic_write_json(path, journal)
        raise RestoreError("; ".join(failures))
    journal["status"] = "restored"
    journal["restore_failures"] = []
    _atomic_write_json(path, journal)


def recover_unfinished_journals(root: Path) -> None:
    for path in sorted(root.glob("*.recovery.json")):
        journal = _load_json(path)
        if journal.get("schema") != JOURNAL_SCHEMA:
            continue
        if journal.get("status") == "restored":
            continue
        print(f"[recovery] unfinished journal found: {path}", flush=True)
        restore_journal(path, actor="next_invocation")
        print(f"[recovery] restored and verified: {path}", flush=True)


def _find_nested_mapping(value: Any, key: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            return candidate
        for child in value.values():
            result = _find_nested_mapping(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_nested_mapping(child, key)
            if result is not None:
                return result
    return None


def decode_udp_payload(
    payload: bytes,
    *,
    compact: bool = False,
    retain_raw: bool = False,
) -> dict[str, Any]:
    """Decode feedback while allowing compact rows to omit redundant payloads."""

    result: dict[str, Any] = {
        "payload_size": len(payload),
        "payload_format": "binary",
    }
    try:
        text = payload.rstrip(b"\x00\r\n").decode("utf-8")
        decoded = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Undecodable packets are diagnostic errors, so compact mode retains
        # their bytes as well.
        result["payload_base64"] = base64.b64encode(payload).decode("ascii")
        return result
    result["payload_format"] = "json"
    if not compact or retain_raw:
        result["payload_base64"] = base64.b64encode(payload).decode("ascii")
        result["payload_json"] = _json_safe(decoded)
        if compact:
            result["schema_sample"] = True
    joint = _find_nested_mapping(decoded, "joint_status")
    if joint is None and isinstance(decoded, Mapping):
        joint = decoded
    if isinstance(joint, Mapping):
        aliases = {
            "joint_position_deg": ("joint_position", "joint", "position"),
            "joint_speed_deg_s": ("joint_speed", "speed"),
            "joint_current_ma": ("joint_current", "current"),
            "joint_temperature_c": ("joint_temperature", "temperature"),
            "joint_voltage_v": ("joint_voltage", "voltage"),
            "joint_error_code": ("joint_err_code", "err_code"),
            "joint_enabled": ("joint_en_flag", "enabled"),
        }
        for output_key, input_keys in aliases.items():
            for input_key in input_keys:
                values = joint.get(input_key)
                if isinstance(values, (list, tuple)):
                    result[output_key] = _json_safe(list(values)[:6])
                    break
    return result


class UdpTraceWriter:
    def __init__(self, path: Path, *, queue_size: int = 65536):
        self.path = path
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=int(queue_size)
        )
        self.thread = threading.Thread(target=self._run, name="rm65_udp_trace_writer")
        self.error: BaseException | None = None
        self.dropped = 0
        self.written = 0
        self._drop_lock = threading.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def enqueue(self, row: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            with self._drop_lock:
                self.dropped += 1

    def close(self) -> None:
        deadline = time.monotonic() + 2.0
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if self.error is not None or time.monotonic() >= deadline:
                    break
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            raise RuntimeError("UDP trace writer did not stop within 10 seconds")
        if self.error is not None:
            raise RuntimeError(f"UDP trace writer failed: {self.error}") from self.error

    def _run(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
                last_sync = time.monotonic()
                while True:
                    row = self.queue.get()
                    if row is None:
                        break
                    stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
                    stream.write("\n")
                    self.written += 1
                    now = time.monotonic()
                    if now - last_sync >= 1.0:
                        stream.flush()
                        os.fsync(stream.fileno())
                        last_sync = now
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as exc:  # noqa: BLE001 - propagated by close().
            self.error = exc


class UdpReceiver:
    def __init__(
        self,
        spec: ArmSpec,
        writer: UdpTraceWriter,
        *,
        trace_format: str = "full",
    ):
        self.spec = spec
        self.writer = writer
        self.trace_format = str(trace_format).strip().lower()
        if self.trace_format not in {"full", "compact"}:
            raise ValueError(f"Unsupported UDP trace format: {trace_format!r}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        self.socket.bind((spec.local_ip, int(spec.push_port)))
        self.socket.settimeout(0.2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"rm65_udp_receiver_{spec.side}"
        )
        self.sequence = 0
        self.unexpected_source_count = 0
        self.json_payload_count = 0
        self.joint_position_count = 0
        self.joint_speed_count = 0
        self.schema_sample_recorded = False
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.socket.close()
        if self.thread.is_alive():
            raise RuntimeError(f"{self.spec.side} UDP receiver did not stop")
        if self.error is not None:
            raise RuntimeError(
                f"{self.spec.side} UDP receiver failed: {self.error}"
            ) from self.error

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    payload, source = self.socket.recvfrom(65535)
                except socket.timeout:
                    continue
                receive_wall_ns = time.time_ns()
                receive_monotonic_ns = time.monotonic_ns()
                if str(source[0]) != self.spec.arm_ip:
                    self.unexpected_source_count += 1
                    unexpected_payload = decode_udp_payload(
                        payload,
                        compact=self.trace_format == "compact",
                        retain_raw=True,
                    )
                    unexpected_payload.pop("schema_sample", None)
                    self.writer.enqueue(
                        {
                            "schema": TRACE_SCHEMA,
                            "event": "unexpected_datagram",
                            "side": self.spec.side,
                            "expected_source_ip": self.spec.arm_ip,
                            "source_ip": str(source[0]),
                            "source_port": int(source[1]),
                            "receive_wall_time_ns": int(receive_wall_ns),
                            "receive_monotonic_time_ns": int(receive_monotonic_ns),
                            **unexpected_payload,
                        }
                    )
                    continue
                self.sequence += 1
                retain_schema_sample = bool(
                    self.trace_format == "compact"
                    and not self.schema_sample_recorded
                )
                decoded = decode_udp_payload(
                    payload,
                    compact=self.trace_format == "compact",
                    retain_raw=retain_schema_sample,
                )
                if decoded.get("payload_format") == "json":
                    self.json_payload_count += 1
                    if retain_schema_sample:
                        self.schema_sample_recorded = True
                if "joint_position_deg" in decoded:
                    self.joint_position_count += 1
                if "joint_speed_deg_s" in decoded:
                    self.joint_speed_count += 1
                feedback_row = {
                    "schema": TRACE_SCHEMA,
                    "event": "feedback",
                    "side": self.spec.side,
                    "sequence": int(self.sequence),
                    "receive_wall_time_ns": int(receive_wall_ns),
                    "receive_monotonic_time_ns": int(receive_monotonic_ns),
                    **decoded,
                }
                if self.trace_format == "full":
                    feedback_row.update(
                        {
                            "arm_ip": self.spec.arm_ip,
                            "source_ip": str(source[0]),
                            "source_port": int(source[1]),
                            "local_ip": self.spec.local_ip,
                            "local_port": int(self.spec.push_port),
                        }
                    )
                self.writer.enqueue(feedback_row)
        except BaseException as exc:  # noqa: BLE001 - propagated by close().
            self.error = exc


@contextmanager
def exclusive_capture_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".rm65_udp_actuator_trace.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another UDP actuator capture owns {lock_path}") from exc
        yield


def _build_specs(args: argparse.Namespace) -> list[ArmSpec]:
    return [
        ArmSpec("left", args.left_arm_ip, args.left_local_ip, args.push_port),
        ArmSpec("right", args.right_arm_ip, args.right_local_ip, args.push_port),
    ]


def _prepare_journal(
    path: Path,
    *,
    specs: Iterable[ArmSpec],
    original_by_side: Mapping[str, Mapping[str, Any]],
    target_by_side: Mapping[str, Mapping[str, Any]],
    trace_path: Path,
) -> dict[str, Any]:
    journal = {
        "schema": JOURNAL_SCHEMA,
        "status": "prepared",
        "created_wall_time": time.time(),
        "creator_pid": os.getpid(),
        "trace_path": str(trace_path.resolve()),
        "arms": [
            {
                "side": spec.side,
                "arm_ip": spec.arm_ip,
                "local_ip": spec.local_ip,
                "push_port": spec.push_port,
                "original_config": normalize_push_config(original_by_side[spec.side]),
                "target_config": normalize_push_config(target_by_side[spec.side]),
            }
            for spec in specs
        ],
    }
    _atomic_write_json(path, journal)
    return journal


def _start_guardian(journal_path: Path) -> tuple[subprocess.Popen[str], Any]:
    guardian_log = journal_path.with_suffix(".guardian.log")
    guardian_ready = journal_path.with_suffix(".guardian.ready.json")
    guardian_ready.unlink(missing_ok=True)
    log_stream = guardian_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_guardian",
            "--journal",
            str(journal_path.resolve()),
        ],
        stdin=subprocess.PIPE,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        close_fds=True,
    )
    log_stream.close()
    if process.stdin is None:
        process.terminate()
        raise RuntimeError("failed to create guardian control pipe")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if guardian_ready.exists():
            ready = _load_json(guardian_ready)
            if int(ready.get("guardian_pid", -1)) != process.pid:
                process.terminate()
                raise RuntimeError("recovery guardian readiness PID mismatch")
            return process, process.stdin
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"recovery guardian exited before readiness with {returncode}; "
                f"see {guardian_log}"
            )
        time.sleep(0.02)
    process.terminate()
    process.wait(timeout=2.0)
    raise RuntimeError(
        f"recovery guardian did not become ready; see {guardian_log}"
    )


def _notify_guardian_restored(
    guardian: subprocess.Popen[str] | None, guardian_pipe: Any | None
) -> None:
    if guardian_pipe is not None:
        guardian_pipe.write("RESTORED\n")
        guardian_pipe.flush()
        guardian_pipe.close()
    if guardian is not None:
        guardian.wait(timeout=10.0)
        if guardian.returncode != 0:
            raise RuntimeError(f"recovery guardian exited with {guardian.returncode}")


def capture(args: argparse.Namespace) -> int:
    if not args.apply:
        raise RuntimeError("capture requires explicit --apply")
    if not args.i_understand_no_motion:
        raise RuntimeError("capture requires --i-understand-no-motion")
    if args.cycle_ms != 5:
        raise RuntimeError("actuator identification requires --cycle-ms 5")
    if not 10.0 <= args.duration_sec <= 600.0:
        raise RuntimeError("--duration-sec must be within [10, 600]")

    root = args.output_root.resolve()
    with exclusive_capture_lock(root):
        recover_unfinished_journals(root)
        specs = _build_specs(args)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_path = root / f"rm65_udp_actuator_trace_{timestamp}.jsonl"
        journal_path = root / f"rm65_udp_actuator_trace_{timestamp}.recovery.json"

        writer = UdpTraceWriter(trace_path, queue_size=args.queue_size)
        original_by_side = {spec.side: query_config(spec.arm_ip) for spec in specs}
        target_by_side = {
            spec.side: temporary_push_config(
                original_by_side[spec.side],
                local_ip=spec.local_ip,
                push_port=spec.push_port,
                cycle_ms=args.cycle_ms,
            )
            for spec in specs
        }
        # Bind both destination sockets before any controller change. Failure
        # here is harmless and must prevent configuration writes.
        receivers: list[UdpReceiver] = []
        try:
            for spec in specs:
                receivers.append(
                    UdpReceiver(
                        spec,
                        writer,
                        trace_format=str(args.trace_format),
                    )
                )
            _prepare_journal(
                journal_path,
                specs=specs,
                original_by_side=original_by_side,
                target_by_side=target_by_side,
                trace_path=trace_path,
            )
        except Exception:
            for receiver in receivers:
                receiver.socket.close()
            raise
        guardian: subprocess.Popen[str] | None = None
        guardian_pipe: Any | None = None
        restored = False
        writer_started = False
        started_receivers: list[UdpReceiver] = []
        stop_event = threading.Event()

        def request_stop(_signum: int, _frame: Any) -> None:
            stop_event.set()

        old_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        for signum in old_handlers:
            signal.signal(signum, request_stop)

        try:
            guardian, guardian_pipe = _start_guardian(journal_path)

            # A no-op round trip proves that every saved field can be restored
            # before the temporary target is ever applied.
            for spec in specs:
                set_and_verify_config(spec.arm_ip, original_by_side[spec.side])

            writer.start()
            writer_started = True
            for receiver in receivers:
                receiver.start()
                started_receivers.append(receiver)

            for spec in specs:
                set_and_verify_config(spec.arm_ip, target_by_side[spec.side])
            journal = _load_json(journal_path)
            journal["status"] = "configured"
            journal["configured_wall_time"] = time.time()
            _atomic_write_json(journal_path, journal)

            writer.enqueue(
                {
                    "schema": TRACE_SCHEMA,
                    "event": "start",
                    "wall_time_ns": time.time_ns(),
                    "monotonic_time_ns": time.monotonic_ns(),
                    "cycle_ms": args.cycle_ms,
                    "duration_sec": args.duration_sec,
                    "trace_format": str(args.trace_format),
                    "arms": [
                        {
                            "side": spec.side,
                            "arm_ip": spec.arm_ip,
                            "local_ip": spec.local_ip,
                            "push_port": int(spec.push_port),
                        }
                        for spec in specs
                    ],
                    "journal_path": str(journal_path.resolve()),
                    "original_config_by_side": original_by_side,
                    "temporary_config_by_side": target_by_side,
                }
            )
            print(f"READY trace={trace_path}", flush=True)
            print(
                "READY start teleop in the other terminal; Ctrl-C here restores UDP config",
                flush=True,
            )
            deadline = time.monotonic() + float(args.duration_sec)
            while not stop_event.is_set() and time.monotonic() < deadline:
                if writer.error is not None:
                    raise RuntimeError(f"writer failed: {writer.error}")
                for receiver in receivers:
                    if receiver.error is not None:
                        raise RuntimeError(
                            f"{receiver.spec.side} receiver failed: {receiver.error}"
                        )
                time.sleep(0.05)
        finally:
            for signum, old_handler in old_handlers.items():
                signal.signal(signum, old_handler)
            close_errors: list[str] = []
            if started_receivers:
                for receiver in reversed(started_receivers):
                    try:
                        receiver.close()
                    except Exception as exc:  # noqa: BLE001
                        close_errors.append(f"receiver {receiver.spec.side}: {exc}")
            for receiver in receivers:
                if receiver not in started_receivers:
                    receiver.socket.close()
            try:
                restore_journal(journal_path, actor="capture_finally")
                restored = True
            except Exception as exc:  # noqa: BLE001
                close_errors.append(f"restore: {type(exc).__name__}: {exc}")
            if writer_started:
                writer.enqueue(
                    {
                        "schema": TRACE_SCHEMA,
                        "event": "stop",
                        "wall_time_ns": time.time_ns(),
                        "monotonic_time_ns": time.monotonic_ns(),
                        "feedback_rows_written_before_stop": writer.written,
                        "queue_drop_count": writer.dropped,
                        "controller_config_restored": restored,
                        "receiver_counters_by_side": {
                            receiver.spec.side: {
                                "feedback": receiver.sequence,
                                "unexpected_source": receiver.unexpected_source_count,
                                "json_payload": receiver.json_payload_count,
                                "joint_position": receiver.joint_position_count,
                                "joint_speed": receiver.joint_speed_count,
                                "schema_sample_recorded": (
                                    receiver.schema_sample_recorded
                                ),
                            }
                            for receiver in receivers
                        },
                        "errors": close_errors,
                    }
                )
                try:
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    close_errors.append(f"writer: {exc}")
            if restored:
                try:
                    _notify_guardian_restored(guardian, guardian_pipe)
                    guardian_pipe = None
                except Exception as exc:  # noqa: BLE001
                    close_errors.append(f"guardian: {exc}")
            elif guardian_pipe is not None:
                # EOF tells the detached guardian to perform another restore.
                guardian_pipe.close()
                guardian_pipe = None
            if close_errors:
                raise RestoreError("; ".join(close_errors))

        validation_errors: list[str] = []
        for receiver in receivers:
            if receiver.sequence == 0:
                validation_errors.append(f"{receiver.spec.side}: no UDP feedback")
            if receiver.joint_position_count == 0:
                validation_errors.append(
                    f"{receiver.spec.side}: no decoded joint-position feedback"
                )
            if receiver.joint_speed_count == 0:
                validation_errors.append(
                    f"{receiver.spec.side}: no decoded joint-speed feedback"
                )
            if receiver.unexpected_source_count:
                validation_errors.append(
                    f"{receiver.spec.side}: {receiver.unexpected_source_count} "
                    "datagrams came from an unexpected source"
                )
        if writer.dropped:
            validation_errors.append(f"writer queue dropped {writer.dropped} rows")
        if validation_errors:
            raise RuntimeError(
                "capture restored controller config but data validation failed: "
                + "; ".join(validation_errors)
            )

        print(
            f"RESTORED verified=true rows={writer.written} dropped={writer.dropped} "
            f"trace={trace_path}",
            flush=True,
        )
        return 0


def inspect(args: argparse.Namespace) -> int:
    payload = {
        "schema": JOURNAL_SCHEMA,
        "mode": "inspect_only",
        "wall_time": time.time(),
        "arms": [
            {
                "side": spec.side,
                "arm_ip": spec.arm_ip,
                "config": query_config(spec.arm_ip),
            }
            for spec in _build_specs(args)
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def guardian(args: argparse.Namespace) -> int:
    journal = _load_json(args.journal)
    if journal.get("schema") != JOURNAL_SCHEMA:
        raise RestoreError(f"unsupported recovery journal schema in {args.journal}")
    for item in _journal_arms(journal):
        normalize_push_config(item["original_config"])
    _atomic_write_json(
        args.journal.with_suffix(".guardian.ready.json"),
        {
            "schema": JOURNAL_SCHEMA,
            "guardian_pid": os.getpid(),
            "journal_path": str(args.journal.resolve()),
            "ready_wall_time": time.time(),
        },
    )
    token = sys.stdin.readline().strip()
    if token == "RESTORED":
        return 0
    # Parent died or closed the pipe without verified restoration.
    try:
        restore_journal(args.journal, actor="detached_guardian")
    except Exception as exc:  # noqa: BLE001
        print(f"GUARDIAN RESTORE FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(f"GUARDIAN RESTORED {args.journal}", flush=True)
    return 0


def restore(args: argparse.Namespace) -> int:
    restore_journal(args.journal.resolve(), actor="manual_restore")
    print(f"RESTORED verified=true journal={args.journal.resolve()}")
    return 0


def _add_arm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--left-arm-ip", default="192.168.101.21")
    parser.add_argument("--right-arm-ip", default="192.168.1.20")
    parser.add_argument("--left-local-ip", default="192.168.101.10")
    parser.add_argument("--right-local-ip", default="192.168.1.10")
    parser.add_argument("--push-port", type=int, default=8099)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Read current UDP push configs without modifying them."
    )
    _add_arm_args(inspect_parser)
    inspect_parser.set_defaults(func=inspect)

    capture_parser = subparsers.add_parser(
        "capture", help="Temporarily configure and record 5 ms UDP feedback."
    )
    _add_arm_args(capture_parser)
    capture_parser.add_argument("--cycle-ms", type=int, default=5)
    capture_parser.add_argument("--duration-sec", type=float, required=True)
    capture_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("teleop_records/actuator_feedback"),
    )
    capture_parser.add_argument("--queue-size", type=int, default=65536)
    capture_parser.add_argument(
        "--format",
        dest="trace_format",
        choices=("full", "compact"),
        default="full",
        help=(
            "full retains every raw payload; compact retains promoted joint fields and only "
            "one raw schema sample per arm plus error packets."
        ),
    )
    capture_parser.add_argument(
        "--apply",
        action="store_true",
        help="Required acknowledgement that controller UDP config is temporarily changed.",
    )
    capture_parser.add_argument(
        "--i-understand-no-motion",
        action="store_true",
        help="Required acknowledgement: this tool records state and sends no motion.",
    )
    capture_parser.set_defaults(func=capture)

    restore_parser = subparsers.add_parser(
        "restore", help="Restore and verify one recovery journal."
    )
    restore_parser.add_argument("--journal", type=Path, required=True)
    restore_parser.set_defaults(func=restore)

    guardian_parser = subparsers.add_parser("_guardian", help=argparse.SUPPRESS)
    guardian_parser.add_argument("--journal", type=Path, required=True)
    guardian_parser.set_defaults(func=guardian)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI must fail loudly.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
