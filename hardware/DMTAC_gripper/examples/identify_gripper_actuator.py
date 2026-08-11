#!/usr/bin/env python3
"""Safely identify one DMTAC/Inspire gripper actuator without robot motion.

This standalone diagnostic controls exactly one gripper serial device.  It does
not import teleop_runtime, initialize an RM65 arm, or read/control tactile
sensors.  It records the gripper's position, current, status, temperature and
command timing while separating two different hold mechanisms:

* device continuous-grasp mode (catch-mode register 5); and
* host position hold (periodically resending the same position target).

Real motion is fail-closed behind a CLI acknowledgement and one global runtime
confirmation.  Free-motion trials then auto-advance; contact trials pause once
for object placement.  Speed is fixed at normalized 1.0.  Original speed,
force and continuous-grasp registers are journaled, verified, and restored. A
detached guardian stops the gripper and restores those settings if the parent
dies. No command in this tool writes persistent/save/default parameters.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import queue
import select
import signal
import statistics
import subprocess
import sys
import termios
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DMTAC_ROOT = PROJECT_ROOT / "hardware" / "DMTAC_gripper"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DMTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(DMTAC_ROOT))

from project_paths import teleop_records_root
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    ArtifactWriteResolution,
    RuntimeArtifactLayout,
    configured_runtime_artifact_layout,
    resolve_artifact_write_path,
)


TRACE_SCHEMA = "dmtac_gripper_actuator_trace_v2"
MANIFEST_SCHEMA = "dmtac_gripper_actuator_manifest_v1"
JOURNAL_SCHEMA = "dmtac_gripper_actuator_recovery_v1"
SPEED_NORMALIZED = 1.0
MOTION_CONFIRMATION = "GRIPPER_ONLY_TEST"
RESTORE_CONFIRMATION = "RESTORE_GRIPPER_SETTINGS"
COMMAND_MODES = ("runtime-equivalent", "inline-config")
ARTIFACT_KIND = "dmtac_gripper_actuator"


def legacy_output_root() -> Path:
    """Return the compatibility-layout write root owned by teleop records."""

    return teleop_records_root() / "gripper_actuator_identification"


def archived_logs_output_root() -> Path:
    """Return the read-only location of runs archived from the former logs root."""

    return PROJECT_ROOT / "legacy" / "logs" / "gripper_actuator_identification"


def canonical_output_family_root() -> Path:
    return RuntimeArtifactLayout.configured().run_family_dir(kind=ARTIFACT_KIND)


def canonical_output_root(*, date_partition: str | None = None) -> Path:
    partition = date_partition or datetime.now().astimezone().date().isoformat()
    return RuntimeArtifactLayout.configured().run_partition_dir(
        kind=ARTIFACT_KIND,
        date_partition=partition,
    )


def resolve_output_root(
    args: argparse.Namespace,
    *,
    date_partition: str | None = None,
) -> ArtifactWriteResolution:
    """Resolve one effective root; explicit CLI paths always take precedence."""

    return resolve_artifact_write_path(
        explicit_path=getattr(args, "output_root", None),
        artifact_layout=getattr(args, "artifact_layout", None),
        legacy_path=legacy_output_root(),
        canonical_path=canonical_output_root(date_partition=date_partition),
    )


def recovery_journal_roots(*additional_roots: Path) -> tuple[Path, ...]:
    """Return all known roots that may own an unfinished settings journal."""

    roots = (
        legacy_output_root(),
        archived_logs_output_root(),
        canonical_output_family_root(),
        *additional_roots,
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _process_lock_root() -> Path:
    return (
        Path(tempfile.gettempdir())
        / f"dexteleop-{os.getuid()}"
        / "runtime_locks"
        / ARTIFACT_KIND
    )


def _run_lock_roots(selected_root: Path) -> tuple[Path, ...]:
    """Bridge old root-local locks while providing one cross-layout lock."""

    selected = selected_root.expanduser().resolve()
    roots = [_process_lock_root(), selected]
    legacy = legacy_output_root().resolve()
    if legacy == selected or legacy.exists():
        roots.append(legacy)
    return tuple(sorted(set(roots), key=str))


class GripperLike(Protocol):
    mb: Any

    def set_joint_pos(self, joint_pos: float, **kwargs: Any) -> None: ...

    def set_speed(self, speed: float) -> None: ...

    def set_torque(self, torque: float) -> None: ...

    def set_continuous_grasp(self, continuous: bool) -> None: ...

    def get_open_limits_raw(self) -> tuple[int, int]: ...

    def estop(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Endpoint:
    side: str
    port: str
    gripper_id: int
    baudrate: int
    timeout: float
    source: str


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    protocol: str
    repeat: int
    force: float
    continuous_grasp: bool
    host_position_hold: str
    target_position: float
    duration_sec: float
    target_positions: tuple[float, ...] = ()

    @property
    def motion_targets(self) -> tuple[float, ...]:
        return self.target_positions or (float(self.target_position),)


class RecoveryError(RuntimeError):
    pass


class AsyncJsonlWriter:
    def __init__(self, path: Path, *, queue_size: int = 16384):
        self.path = path
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(queue_size)
        self.thread = threading.Thread(
            target=self._run,
            name="gripper_identification_writer",
            daemon=False,
        )
        self.error: BaseException | None = None
        self.written = 0
        self.dropped = 0
        self._next_trace_sequence = 0

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def enqueue(self, row: Mapping[str, Any]) -> None:
        payload = dict(row)
        payload.setdefault("trace_sequence", self._next_trace_sequence)
        self._next_trace_sequence += 1
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
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
            raise RuntimeError("trace writer did not stop")
        if self.error is not None:
            raise RuntimeError(f"trace writer failed: {self.error}") from self.error

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
        except BaseException as exc:  # noqa: BLE001
            self.error = exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
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
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _latest_config_path() -> Path:
    return PROJECT_ROOT / "teleop_runtime" / "configs" / "hardware" / "dmtac" / "latest.json"


def _resolve_config_path(value: str) -> Path:
    if value == "latest":
        return _latest_config_path()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_endpoint(args: argparse.Namespace) -> tuple[Endpoint, dict[str, Any] | None]:
    side = str(args.side)
    if args.port:
        endpoint = Endpoint(
            side=side,
            port=str(Path(args.port).expanduser()),
            gripper_id=int(args.gripper_id),
            baudrate=int(args.baudrate),
            timeout=float(args.timeout),
            source="cli",
        )
        return endpoint, None

    config_path = _resolve_config_path(str(args.config))
    config = _load_json(config_path)
    assignments = config.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError(f"{config_path}: missing assignments")
    side_config = assignments.get(side)
    if not isinstance(side_config, Mapping):
        raise ValueError(f"{config_path}: missing side {side!r}")
    gripper = side_config.get("gripper")
    if not isinstance(gripper, Mapping) or not bool(gripper.get("enabled", False)):
        raise ValueError(f"{config_path}: {side} gripper is missing or disabled")
    endpoint = Endpoint(
        side=side,
        port=str(gripper["port"]),
        gripper_id=int(gripper.get("gripper_id", 1)),
        baudrate=int(gripper.get("baudrate", 115200)),
        timeout=float(gripper.get("timeout", args.timeout)),
        source=str(config_path),
    )
    return endpoint, config


def validate_endpoint(endpoint: Endpoint) -> None:
    if endpoint.side not in {"left", "right"}:
        raise ValueError(f"invalid side {endpoint.side!r}")
    if not str(endpoint.port).startswith("/dev/"):
        raise ValueError(f"serial port must be an absolute /dev path, got {endpoint.port!r}")
    if not 1 <= int(endpoint.gripper_id) <= 247:
        raise ValueError("gripper id must be within [1,247]")
    if int(endpoint.baudrate) not in {19200, 57600, 115200}:
        raise ValueError(f"unsupported gripper baudrate {endpoint.baudrate}")
    if not 0.005 <= float(endpoint.timeout) <= 1.0:
        raise ValueError("serial timeout must be within [0.005,1.0] seconds")


def _parse_csv_floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("numeric list cannot be empty")
    return result


def _parse_csv_bools(value: str) -> list[bool]:
    aliases = {"true": True, "1": True, "false": False, "0": False}
    result: list[bool] = []
    for item in str(value).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"invalid boolean {item!r}; use false,true")
        result.append(aliases[token])
    if not result:
        raise ValueError("continuous mode list cannot be empty")
    return result


def _parse_host_hold_modes(value: str) -> list[str]:
    allowed = {"one-shot", "resend"}
    result = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    invalid = [item for item in result if item not in allowed]
    if invalid or not result:
        raise ValueError(f"invalid host hold modes {invalid or result}; use one-shot,resend")
    return result


def build_trial_specs(args: argparse.Namespace) -> list[TrialSpec]:
    forces = _parse_csv_floats(args.force_levels)
    if any(force < 0.0 or force > 1.0 for force in forces):
        raise ValueError("all normalized force levels must be within [0,1]")
    if max(forces) > 0.5:
        if not args.allow_high_force or args.high_force_confirmation != "HIGH_FORCE":
            raise ValueError(
                "force above 0.5 requires --allow-high-force "
                "--high-force-confirmation HIGH_FORCE"
            )
    if not 0.0 <= args.target_position <= 1.0:
        raise ValueError("--target-position must be within [0,1]")
    if not 0.0 <= args.open_position < args.target_position:
        raise ValueError("--open-position must be >=0 and below --target-position")
    if not 1 <= args.repeats <= 10:
        raise ValueError("--repeats must be within [1,10]")
    if not 1.0 <= float(args.sample_hz) <= 100.0:
        raise ValueError("--sample-hz must be within [1,100]")
    if not 0.5 <= float(args.resend_hz) <= 60.0:
        raise ValueError("--resend-hz must be within [0.5,60]")
    if not 0.5 <= float(args.free_duration_sec) <= 60.0:
        raise ValueError("--free-duration-sec must be within [0.5,60]")
    if not 0.5 <= float(args.hold_duration_sec) <= 60.0:
        raise ValueError("--hold-duration-sec must be within [0.5,60]")
    command_mode = str(getattr(args, "command_mode", "runtime-equivalent"))
    if command_mode not in COMMAND_MODES:
        raise ValueError(f"--command-mode must be one of {COMMAND_MODES}")

    target_sequence_text = str(getattr(args, "target_sequence", "") or "").strip()
    if target_sequence_text:
        target_positions = tuple(_parse_csv_floats(target_sequence_text))
        if any(not 0.0 <= target <= 1.0 for target in target_positions):
            raise ValueError("--target-sequence values must be within [0,1]")
    else:
        target_positions = (float(args.target_position),)

    if args.protocol == "free":
        combinations = [
            (bool(getattr(args, "free_continuous_grasp", False)), "one-shot")
        ]
    else:
        if target_sequence_text:
            raise ValueError("--target-sequence is only supported by protocol=free")
        continuous_modes = _parse_csv_bools(args.continuous_modes)
        host_modes = _parse_host_hold_modes(args.host_hold_modes)
        combinations = [
            (continuous, host_mode)
            for continuous in continuous_modes
            for host_mode in host_modes
        ]
        if not str(args.object_label).strip() or args.object_width_mm <= 0.0:
            raise ValueError(
                "contact protocol requires --object-label and positive --object-width-mm"
            )

    duration = args.free_duration_sec if args.protocol == "free" else args.hold_duration_sec
    specs: list[TrialSpec] = []
    index = 0
    for repeat in range(1, args.repeats + 1):
        for force in sorted(forces):
            for continuous, host_mode in combinations:
                index += 1
                specs.append(
                    TrialSpec(
                        trial_id=f"trial_{index:03d}",
                        protocol=str(args.protocol),
                        repeat=repeat,
                        force=float(force),
                        continuous_grasp=bool(continuous),
                        host_position_hold=host_mode,
                        target_position=float(args.target_position),
                        duration_sec=float(duration),
                        target_positions=target_positions,
                    )
                )
    return specs


def _open_gripper(endpoint: Endpoint) -> GripperLike:
    from portable_dmtac_gripper.gripper_driver import InspireGripper

    return InspireGripper(
        port=endpoint.port,
        gripper_id=endpoint.gripper_id,
        baudrate=endpoint.baudrate,
        timeout=endpoint.timeout,
        # Keep the TIOCEXCL serial claim valid for the full run.  Automatic
        # reconnect would close/reopen the descriptor and silently lose that
        # exclusivity.  A communication fault therefore aborts the trial and
        # enters the explicit stop/restore path.
        reconnect_on_error=False,
        reconnect_attempts=0,
        reconnect_delay=0.0,
        inter_command_delay=0.005,
        use_write_multiple_registers=False,
    )


def _exclusive_serial(gripper: GripperLike) -> None:
    serial_object = getattr(gripper.mb, "ser", None)
    if serial_object is None or not hasattr(serial_object, "fileno"):
        raise RuntimeError("gripper driver did not expose an open serial descriptor")
    try:
        fcntl.ioctl(serial_object.fileno(), termios.TIOCEXCL)
    except OSError as exc:
        raise RuntimeError(f"could not claim exclusive serial access: {exc}") from exc


def find_device_users(port: str) -> list[dict[str, Any]]:
    target = os.path.realpath(port)
    users: list[dict[str, Any]] = []
    proc = Path("/proc")
    for process_dir in proc.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        matched = False
        for descriptor in descriptors:
            try:
                destination = os.path.realpath(os.readlink(descriptor))
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if destination == target:
                matched = True
                break
        if not matched:
            continue
        try:
            cmdline = (process_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError):
            cmdline = ""
        users.append({"pid": int(process_dir.name), "cmdline": cmdline})
    return users


def read_raw_settings(gripper: GripperLike) -> dict[str, int]:
    speed_force = gripper.mb.read_holding_registers(gripper.mb.REG_SPEED_SET, 2)
    continuous = gripper.mb.read_holding_registers(gripper.mb.REG_CATCH_MODE, 1)[0]
    if int(continuous) not in (0, 1):
        raise RecoveryError(f"unexpected continuous-grasp register value {continuous}")
    return {
        "speed_raw": int(speed_force[0]),
        "force_raw": int(speed_force[1]),
        "continuous_raw": int(continuous),
    }


def read_state_safely(gripper: GripperLike) -> dict[str, Any]:
    """Read state while preserving and correctly decoding endpoint sentinels.

    The opening register is transmitted as uint16, but this gripper can report
    small negative values such as -2 (0xFFFE) at the fully closed mechanical
    endpoint.  The shared driver intentionally accepts -20..1020 before
    clamping, but its current get_state path applies that range check to the
    unsigned value.  Keep this workaround local to the identification tool so
    the experiment can record both raw representations without changing the
    production driver.
    """

    registers = gripper.mb.read_holding_registers(gripper.mb.REG_OPENLEN_ACT, 5)
    opening_u16 = int(registers[0]) & 0xFFFF
    opening_signed = opening_u16 - 0x10000 if opening_u16 & 0x8000 else opening_u16
    if not -20 <= opening_signed <= 1020:
        raise RuntimeError(
            "abnormal opening register outside tolerated endpoint range: "
            f"u16={opening_u16}, signed={opening_signed}"
        )
    opening_raw = max(0, min(opening_signed, 1000))
    current_u16 = int(registers[1]) & 0xFFFF
    current_signed = current_u16 - 0x10000 if current_u16 & 0x8000 else current_u16
    status_code = int(registers[4])
    status_texts = getattr(gripper, "STATUS_TEXT", {})
    return {
        "timestamp": time.time(),
        "joint_pos": float(1.0 - opening_raw / 1000.0),
        "is_moving": status_code in (4, 5),
        "opening_raw": int(opening_raw),
        "opening_register_raw_u16": int(opening_u16),
        "opening_register_signed": int(opening_signed),
        "opening_endpoint_clamped": opening_signed != opening_raw,
        "opening_mm": float(opening_raw / 1000.0 * gripper.max_open_mm),
        "current_raw": int(current_u16),
        "current": int(current_signed),
        "temperature_c": int(registers[2]),
        "error_code": int(registers[3]),
        "status_code": status_code,
        "status_text": status_texts.get(status_code, f"未知状态({status_code})"),
        "commanded_joint_pos": getattr(gripper, "last_joint_pos_cmd", None),
        "commanded_speed": getattr(gripper, "last_speed_cmd", None),
        "commanded_torque": getattr(gripper, "last_torque_cmd", None),
    }


def restore_raw_settings(gripper: GripperLike, expected: Mapping[str, Any]) -> None:
    writes = (
        (gripper.mb.REG_SPEED_SET, int(expected["speed_raw"])),
        (gripper.mb.REG_FORCE_SET, int(expected["force_raw"])),
        (gripper.mb.REG_CATCH_MODE, int(expected["continuous_raw"])),
    )
    for register, value in writes:
        gripper.mb.write_single_register(register, value)
    observed = read_raw_settings(gripper)
    normalized_expected = {key: int(value) for key, value in expected.items()}
    if observed != normalized_expected:
        raise RecoveryError(
            f"gripper setting restore readback mismatch: expected={normalized_expected}, "
            f"observed={observed}"
        )


def _endpoint_from_journal(journal: Mapping[str, Any]) -> Endpoint:
    raw = journal.get("endpoint")
    if not isinstance(raw, Mapping):
        raise RecoveryError("recovery journal has no endpoint")
    return Endpoint(
        side=str(raw["side"]),
        port=str(raw["port"]),
        gripper_id=int(raw["gripper_id"]),
        baudrate=int(raw["baudrate"]),
        timeout=float(raw["timeout"]),
        source=str(raw.get("source", "journal")),
    )


def restore_journal(path: Path, *, actor: str, stop_first: bool = True) -> None:
    journal = _load_json(path)
    if journal.get("schema") != JOURNAL_SCHEMA:
        raise RecoveryError(f"unsupported journal schema in {path}")
    endpoint = _endpoint_from_journal(journal)
    validate_endpoint(endpoint)
    original = journal.get("original_settings_raw")
    if not isinstance(original, Mapping):
        raise RecoveryError("recovery journal has no original settings")
    journal["status"] = "restoring"
    journal["restore_actor"] = actor
    journal["restore_started_wall_time_ns"] = time.time_ns()
    _atomic_write_json(path, journal)
    gripper: GripperLike | None = None
    try:
        users = find_device_users(endpoint.port)
        if users:
            raise RecoveryError(f"serial device is already open by {users}")
        gripper = _open_gripper(endpoint)
        _exclusive_serial(gripper)
        if stop_first:
            gripper.estop()
        restore_raw_settings(gripper, original)
        observed = read_raw_settings(gripper)
    except Exception as exc:
        failed = _load_json(path)
        failed["status"] = "restore_failed"
        failed["restore_error"] = f"{type(exc).__name__}: {exc}"
        failed["restore_finished_wall_time_ns"] = time.time_ns()
        _atomic_write_json(path, failed)
        raise
    finally:
        if gripper is not None:
            gripper.close()
    restored = _load_json(path)
    restored["status"] = "restored"
    restored["restored_settings_raw"] = observed
    restored["restore_error"] = None
    restored["restore_finished_wall_time_ns"] = time.time_ns()
    _atomic_write_json(path, restored)


def _guardian_ready_path(journal_path: Path) -> Path:
    return journal_path.with_suffix(".guardian.ready.json")


def _start_guardian(journal_path: Path) -> tuple[subprocess.Popen[str], Any]:
    ready_path = _guardian_ready_path(journal_path)
    ready_path.unlink(missing_ok=True)
    log_path = journal_path.with_suffix(".guardian.log")
    log_stream = log_path.open("a", encoding="utf-8")
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
        if ready_path.exists():
            ready = _load_json(ready_path)
            if int(ready.get("guardian_pid", -1)) != process.pid:
                process.terminate()
                raise RuntimeError("guardian readiness PID mismatch")
            return process, process.stdin
        if process.poll() is not None:
            raise RuntimeError(
                f"guardian exited before readiness; see {log_path}"
            )
        time.sleep(0.02)
    process.terminate()
    process.wait(timeout=2.0)
    raise RuntimeError(f"guardian readiness timeout; see {log_path}")


def _notify_guardian_restored(process: subprocess.Popen[str], pipe: Any) -> None:
    pipe.write("RESTORED\n")
    pipe.flush()
    pipe.close()
    process.wait(timeout=10.0)
    if process.returncode != 0:
        raise RuntimeError(f"guardian exited with {process.returncode}")


@contextmanager
def _run_lock(root: Path, endpoint: Endpoint):
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(os.path.realpath(endpoint.port).encode()).hexdigest()[:16]
    path = root / f".gripper_identification_{digest}.lock"
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another identification process owns {path}") from exc
        yield


@contextmanager
def _run_locks(roots: Iterable[Path], endpoint: Endpoint):
    """Acquire layout-independent and compatibility locks in stable order."""

    with ExitStack() as stack:
        for root in sorted({path.resolve() for path in roots}, key=str):
            stack.enter_context(_run_lock(root, endpoint))
        yield


def _unfinished_journals(root: Path, endpoint: Endpoint) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.glob("**/*.recovery.json")):
        try:
            journal = _load_json(path)
            journal_endpoint = _endpoint_from_journal(journal)
        except Exception:
            continue
        if (
            journal.get("schema") == JOURNAL_SCHEMA
            and journal.get("status") != "restored"
            and os.path.realpath(journal_endpoint.port) == os.path.realpath(endpoint.port)
        ):
            paths.append(path)
    return paths


def _trace_row(event: str, **values: Any) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA,
        "event": event,
        "wall_time_ns": time.time_ns(),
        "monotonic_time_ns": time.monotonic_ns(),
        **values,
    }


def _configure_motion(
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    spec: TrialSpec,
    *,
    stage: str,
    force: float,
    continuous: bool,
) -> dict[str, int]:
    """Apply non-position settings outside the timed target-write window."""

    begin_monotonic_ns = time.monotonic_ns()
    begin_wall_time_ns = time.time_ns()
    writer.enqueue(
        _trace_row(
            "motion_config_begin",
            trial_id=spec.trial_id,
            stage=stage,
            speed=SPEED_NORMALIZED,
            force=float(force),
            continuous_grasp=bool(continuous),
            config_begin_monotonic_ns=begin_monotonic_ns,
            config_begin_wall_time_ns=begin_wall_time_ns,
        )
    )
    try:
        gripper.set_speed(SPEED_NORMALIZED)
        gripper.set_torque(float(force))
        gripper.set_continuous_grasp(bool(continuous))
        observed = read_raw_settings(gripper)
    except Exception as exc:
        writer.enqueue(
            _trace_row(
                "motion_config_error",
                trial_id=spec.trial_id,
                stage=stage,
                config_begin_monotonic_ns=begin_monotonic_ns,
                config_end_monotonic_ns=time.monotonic_ns(),
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    writer.enqueue(
        _trace_row(
            "motion_config_end",
            trial_id=spec.trial_id,
            stage=stage,
            config_begin_monotonic_ns=begin_monotonic_ns,
            config_end_monotonic_ns=time.monotonic_ns(),
            observed_settings_raw=observed,
        )
    )
    return observed


def _send_position(
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    spec: TrialSpec,
    *,
    stage: str,
    target: float,
    force: float | None,
    continuous: bool | None,
    command_mode: str = "runtime-equivalent",
    scheduled_monotonic_ns: int | None = None,
    segment_index: int | None = None,
) -> None:
    if command_mode not in COMMAND_MODES:
        raise ValueError(f"unsupported command_mode {command_mode!r}")
    before = time.monotonic_ns()
    before_wall = time.time_ns()
    writer.enqueue(
        _trace_row(
            "command_begin",
            trial_id=spec.trial_id,
            stage=stage,
            target_position=float(target),
            speed=SPEED_NORMALIZED if force is not None else None,
            force=force,
            continuous_grasp=continuous,
            command_mode=command_mode,
            segment_index=segment_index,
            scheduled_monotonic_ns=scheduled_monotonic_ns,
            command_begin_monotonic_ns=before,
            command_begin_wall_time_ns=before_wall,
        )
    )
    try:
        if command_mode == "runtime-equivalent":
            # Speed/force/continuous are pre-applied and verified by
            # _configure_motion. This timed call contains only the steady-state
            # position register transaction used by the production backend.
            gripper.set_joint_pos(
                float(target),
                speed=None,
                torque=None,
                continuous=None,
                force_send=True,
            )
        else:
            gripper.set_joint_pos(
                float(target),
                speed=SPEED_NORMALIZED if force is not None else None,
                torque=force,
                continuous=continuous,
                force_send=True,
            )
    except Exception as exc:
        writer.enqueue(
            _trace_row(
                "command_error",
                trial_id=spec.trial_id,
                stage=stage,
                target_position=float(target),
                command_mode=command_mode,
                segment_index=segment_index,
                scheduled_monotonic_ns=scheduled_monotonic_ns,
                command_begin_monotonic_ns=before,
                command_error_monotonic_ns=time.monotonic_ns(),
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    end = time.monotonic_ns()
    writer.enqueue(
        _trace_row(
            "command_end",
            trial_id=spec.trial_id,
            stage=stage,
            target_position=float(target),
            command_mode=command_mode,
            segment_index=segment_index,
            scheduled_monotonic_ns=scheduled_monotonic_ns,
            command_begin_monotonic_ns=before,
            command_end_monotonic_ns=end,
            command_end_wall_time_ns=time.time_ns(),
            call_duration_ns=end - before,
            position_write_timing_semantics=(
                "position_register_transaction_bounds"
                if command_mode == "runtime-equivalent"
                else "combined_config_and_position_call_bounds"
            ),
        )
    )


def _sample_state(
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    spec: TrialSpec,
    *,
    stage: str,
    sequence: int,
    scheduled_monotonic_ns: int | None = None,
    segment_index: int | None = None,
    max_temperature_c: float | None = None,
) -> dict[str, Any]:
    begin = time.monotonic_ns()
    begin_wall = time.time_ns()
    state = read_state_safely(gripper)
    end = time.monotonic_ns()
    end_wall = time.time_ns()
    row = _trace_row(
        "state",
        trial_id=spec.trial_id,
        stage=stage,
        sequence=sequence,
        segment_index=segment_index,
        scheduled_monotonic_ns=scheduled_monotonic_ns,
        read_begin_monotonic_ns=begin,
        read_end_monotonic_ns=end,
        read_midpoint_monotonic_ns=(begin + end) // 2,
        read_begin_wall_time_ns=begin_wall,
        read_end_wall_time_ns=end_wall,
        read_duration_ns=end - begin,
        state=state,
    )
    writer.enqueue(row)
    if int(state.get("error_code", 0)) != 0:
        raise RuntimeError(
            f"{spec.trial_id}: gripper error_code={state.get('error_code')}"
        )
    if (
        max_temperature_c is not None
        and float(state.get("temperature_c", 0.0)) > float(max_temperature_c)
    ):
        raise RuntimeError(
            f"{spec.trial_id}: temperature={state.get('temperature_c')}C exceeds "
            f"limit={float(max_temperature_c):g}C"
        )
    return row


def _observe_trial(
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    spec: TrialSpec,
    *,
    sample_hz: float,
    resend_hz: float,
    target_position: float | None = None,
    segment_index: int = 0,
    max_temperature_c: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    sample_period = 1.0 / float(sample_hz)
    resend_period = 1.0 / float(resend_hz)
    start = time.monotonic()
    deadline = start + spec.duration_sec
    next_sample = start
    next_resend = start + resend_period
    samples: list[dict[str, Any]] = []
    sequence = 0
    resend_count = 0
    target = spec.target_position if target_position is None else float(target_position)
    while time.monotonic() < deadline:
        now = time.monotonic()
        if spec.host_position_hold == "resend" and now >= next_resend:
            _send_position(
                gripper,
                writer,
                spec,
                stage="host_position_hold_resend",
                target=target,
                force=None,
                continuous=None,
                command_mode="runtime-equivalent",
                scheduled_monotonic_ns=int(round(next_resend * 1e9)),
                segment_index=segment_index,
            )
            resend_count += 1
            while next_resend <= now:
                next_resend += resend_period
        if now >= next_sample:
            sequence += 1
            samples.append(
                _sample_state(
                    gripper,
                    writer,
                    spec,
                    stage="motion_and_hold",
                    sequence=sequence,
                    scheduled_monotonic_ns=int(round(next_sample * 1e9)),
                    segment_index=segment_index,
                    max_temperature_c=max_temperature_c,
                )
            )
            while next_sample <= now:
                next_sample += sample_period
        if writer.error is not None:
            raise RuntimeError(f"trace writer failed: {writer.error}")
        sleep_until = min(
            next_sample,
            next_resend if spec.host_position_hold == "resend" else deadline,
            deadline,
        )
        time.sleep(min(0.005, max(0.0, sleep_until - time.monotonic())))
    return samples, resend_count


def summarize_trial(
    spec: TrialSpec,
    samples: Iterable[Mapping[str, Any]],
    *,
    resend_count: int,
    measured_force_n: float | None,
    measured_force_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(samples)
    states = [dict(row["state"]) for row in rows]
    timestamps = [
        int(
            row["read_midpoint_monotonic_ns"]
            if "read_midpoint_monotonic_ns" in row
            else row["monotonic_time_ns"]
        )
        for row in rows
    ]
    gaps_ms = [
        (right - left) / 1e6 for left, right in zip(timestamps[:-1], timestamps[1:])
    ]
    positions = [float(state["joint_pos"]) for state in states]
    currents = [abs(float(state["current"])) for state in states]
    temperatures = [float(state["temperature_c"]) for state in states]
    status_codes = [int(state["status_code"]) for state in states]
    half = max(0, len(positions) // 2)
    hold_positions = positions[half:]
    segment_summaries: list[dict[str, Any]] = []
    for segment_index, target in enumerate(spec.motion_targets):
        segment_rows = [
            row for row in rows if int(row.get("segment_index", 0)) == segment_index
        ]
        segment_states = [dict(row["state"]) for row in segment_rows]
        segment_positions = [float(state["joint_pos"]) for state in segment_states]
        segment_summaries.append(
            {
                "segment_index": segment_index,
                "target_position": float(target),
                "sample_count": len(segment_rows),
                "start_position": segment_positions[0] if segment_positions else None,
                "final_position": segment_positions[-1] if segment_positions else None,
                "max_abs_tracking_error": (
                    max(abs(float(target) - value) for value in segment_positions)
                    if segment_positions
                    else None
                ),
            }
        )
    return {
        "trial": asdict(spec),
        "sample_count": len(rows),
        "resend_count": int(resend_count),
        "measured_force_n": measured_force_n,
        "measured_force_evidence": (
            None if measured_force_evidence is None else dict(measured_force_evidence)
        ),
        "segments": segment_summaries,
        "start_position": positions[0] if positions else None,
        "final_position": positions[-1] if positions else None,
        "hold_position_drift": (
            hold_positions[-1] - hold_positions[0] if len(hold_positions) >= 2 else None
        ) if len(spec.motion_targets) == 1 else None,
        "max_abs_current": max(currents) if currents else None,
        "median_abs_current": statistics.median(currents) if currents else None,
        "max_temperature_c": max(temperatures) if temperatures else None,
        "force_stop_seen": 6 in status_codes,
        "status_codes_seen": sorted(set(status_codes)),
        "sample_gap_ms": {
            "median": statistics.median(gaps_ms) if gaps_ms else None,
            "max": max(gaps_ms) if gaps_ms else None,
        },
    }


def _prompt_enter(
    message: str,
    *,
    gripper: GripperLike | None = None,
    writer: AsyncJsonlWriter | None = None,
    spec: TrialSpec | None = None,
    stage: str = "operator_wait",
    max_temperature_c: float | None = None,
) -> None:
    """Wait for Enter while retaining low-rate fault/temperature monitoring."""

    if gripper is None or writer is None or spec is None:
        input(f"{message}\nPress Enter to continue (Ctrl-C to abort): ")
        return
    print(f"{message}\nPress Enter to continue (Ctrl-C to abort): ", end="", flush=True)
    next_sample = time.monotonic()
    sequence = 0
    while True:
        now = time.monotonic()
        if now >= next_sample:
            sequence += 1
            _sample_state(
                gripper,
                writer,
                spec,
                stage=stage,
                sequence=sequence,
                scheduled_monotonic_ns=int(round(next_sample * 1e9)),
                max_temperature_c=max_temperature_c,
            )
            while next_sample <= now:
                next_sample += 1.0
        readable, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            min(0.1, max(0.0, next_sample - time.monotonic())),
        )
        if readable:
            sys.stdin.readline()
            return


def _monitored_delay(
    duration_sec: float,
    *,
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    spec: TrialSpec,
    stage: str,
    max_temperature_c: float,
) -> None:
    """Delay at one sample per second without leaving the safety gate blind."""

    deadline = time.monotonic() + float(duration_sec)
    next_sample = time.monotonic()
    sequence = 0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_sample:
            sequence += 1
            _sample_state(
                gripper,
                writer,
                spec,
                stage=stage,
                sequence=sequence,
                scheduled_monotonic_ns=int(round(next_sample * 1e9)),
                max_temperature_c=max_temperature_c,
            )
            while next_sample <= now:
                next_sample += 1.0
        time.sleep(min(0.05, max(0.0, min(next_sample, deadline) - time.monotonic())))


def _wait_for_idle(
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    spec: TrialSpec,
    *,
    stage: str,
    timeout_sec: float,
    sample_hz: float,
    max_temperature_c: float | None = None,
) -> None:
    deadline = time.monotonic() + timeout_sec
    observed_motion = False
    sequence = 0
    while time.monotonic() < deadline:
        sequence += 1
        row = _sample_state(
            gripper,
            writer,
            spec,
            stage=stage,
            sequence=sequence,
            max_temperature_c=max_temperature_c,
        )
        moving = bool(row["state"].get("is_moving", False))
        observed_motion = observed_motion or moving
        if not moving and (observed_motion or sequence >= 3):
            return
        time.sleep(1.0 / sample_hz)
    raise RuntimeError(f"{spec.trial_id}: {stage} did not become idle")


def _wait_for_suite_preflight_idle(
    gripper: GripperLike,
    *,
    timeout_sec: float,
    max_temperature_c: float,
    sample_interval_sec: float = 0.05,
    consecutive_idle_samples: int = 3,
    safe_open_position: float = 0.0,
    safe_open_tolerance: float = 0.015,
    safe_open_current_max: float = 100.0,
    consecutive_safe_open_samples: int = 5,
    evidence_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read until a suite handoff is idle or provably stationary at safe-open."""

    deadline = time.monotonic() + timeout_sec
    observations: list[dict[str, Any]] = []
    idle_count = 0
    safe_open_count = 0

    def save(decision: str, accepted: bool) -> None:
        if evidence_path is None:
            return
        _atomic_write_json(
            evidence_path,
            {
                "schema": "dmtac_gripper_suite_preflight_v1",
                "accepted": accepted,
                "decision": decision,
                "criteria": {
                    "consecutive_idle_samples": consecutive_idle_samples,
                    "safe_open_position": safe_open_position,
                    "safe_open_tolerance": safe_open_tolerance,
                    "safe_open_current_max": safe_open_current_max,
                    "consecutive_safe_open_samples": consecutive_safe_open_samples,
                },
                "observations": observations,
            },
        )

    while time.monotonic() < deadline:
        state = read_state_safely(gripper)
        observations.append(state)
        if int(state.get("error_code", 0)) != 0:
            save("nonzero_error_code", False)
            raise RuntimeError(
                f"gripper error_code={state.get('error_code')} at suite preflight; "
                "inspect and clear the fault separately"
            )
        if float(state.get("temperature_c", 0.0)) > max_temperature_c:
            save("temperature_limit", False)
            raise RuntimeError("gripper temperature exceeds configured safety maximum")
        if bool(state.get("is_moving", False)):
            idle_count = 0
        else:
            idle_count += 1
            if idle_count >= consecutive_idle_samples:
                save("status_idle", True)
                return observations
        safe_open = (
            abs(float(state.get("joint_pos", 1.0)) - safe_open_position)
            <= safe_open_tolerance
            and abs(float(state.get("current", safe_open_current_max + 1.0)))
            <= safe_open_current_max
        )
        safe_open_count = safe_open_count + 1 if safe_open else 0
        if safe_open_count >= consecutive_safe_open_samples:
            save("stable_safe_open_low_current_status_ignored", True)
            return observations
        time.sleep(sample_interval_sec)
    save("timeout_not_stationary", False)
    raise RuntimeError(
        "gripper was neither status-idle nor stably safe-open with low current "
        f"during the {timeout_sec:g}s suite preflight"
    )


def _manual_force_measurement(
    args: argparse.Namespace,
    spec: TrialSpec,
    gripper: GripperLike,
    writer: AsyncJsonlWriter,
    *,
    sample_hz: float,
    max_temperature_c: float,
) -> dict[str, Any] | None:
    if args.external_force_reference != "manual-gauge":
        return None
    timeout_sec = float(args.manual_gauge_timeout_sec)
    print(
        f"{spec.trial_id}: enter external force-gauge reading in N, or blank if unavailable "
        f"(timeout {timeout_sec:g}s; state monitoring remains active): ",
        end="",
        flush=True,
    )
    deadline = time.monotonic() + timeout_sec
    next_sample = time.monotonic()
    sequence = 0
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(f"{spec.trial_id}: manual force-gauge input timed out")
        if now >= next_sample:
            sequence += 1
            _sample_state(
                gripper,
                writer,
                spec,
                stage="manual_force_wait",
                sequence=sequence,
                scheduled_monotonic_ns=int(round(next_sample * 1e9)),
                max_temperature_c=max_temperature_c,
            )
            while next_sample <= now:
                next_sample += 1.0 / min(float(sample_hz), 10.0)
        wait_sec = min(0.1, max(0.0, next_sample - time.monotonic()), deadline - time.monotonic())
        readable, _, _ = select.select([sys.stdin], [], [], max(0.0, wait_sec))
        if readable:
            value = sys.stdin.readline().strip()
            break
    captured_monotonic_ns = time.monotonic_ns()
    captured_wall_time_ns = time.time_ns()
    if not value:
        measurement = {
            "available": False,
            "force_n": None,
            "captured_monotonic_ns": captured_monotonic_ns,
            "captured_wall_time_ns": captured_wall_time_ns,
        }
        writer.enqueue(
            _trace_row(
                "manual_force_measurement",
                trial_id=spec.trial_id,
                **measurement,
            )
        )
        return measurement
    measured = float(value)
    if measured < 0.0:
        raise ValueError("measured force cannot be negative")
    measurement = {
        "available": True,
        "force_n": measured,
        "captured_monotonic_ns": captured_monotonic_ns,
        "captured_wall_time_ns": captured_wall_time_ns,
        "semantics": "operator_transcribed_static_gauge_value_not_force_timeseries",
    }
    writer.enqueue(_trace_row("manual_force_measurement", trial_id=spec.trial_id, **measurement))
    return measurement


def plan(args: argparse.Namespace) -> int:
    specs = build_trial_specs(args)
    payload = {
        "speed_fixed": SPEED_NORMALIZED,
        "protocol": args.protocol,
        "command_mode": str(getattr(args, "command_mode", "runtime-equivalent")),
        "trial_count": len(specs),
        "trials": [asdict(spec) for spec in specs],
        "interpretation": {
            "continuous_grasp": "device catch-mode register; internal behavior",
            "host_position_hold": "one position write versus periodic target resend",
            "force": "normalized setting only; not Newtons without an external calibrated gauge",
            "runtime-equivalent": (
                "speed/force/continuous are pre-applied; the timed command contains only "
                "the position-register transaction"
            ),
            "resend": (
                "forced diagnostic write; production may suppress an unchanged target via "
                "send_threshold"
            ),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def inspect(args: argparse.Namespace) -> int:
    endpoint, config = resolve_endpoint(args)
    validate_endpoint(endpoint)
    users = find_device_users(endpoint.port)
    if users:
        raise RuntimeError(f"serial device is already open by {users}")
    gripper = _open_gripper(endpoint)
    try:
        _exclusive_serial(gripper)
        state = read_state_safely(gripper)
        payload = {
            "mode": "read_only",
            "endpoint": asdict(endpoint),
            "settings_raw": read_raw_settings(gripper),
            "open_limits_raw": list(gripper.get_open_limits_raw()),
            "state": state,
            "config_created_at": config.get("created_at") if config else None,
        }
    finally:
        gripper.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _prepare_journal(
    path: Path,
    *,
    endpoint: Endpoint,
    original_settings: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    trace_path: Path,
) -> None:
    _atomic_write_json(
        path,
        {
            "schema": JOURNAL_SCHEMA,
            "status": "prepared",
            "created_wall_time_ns": time.time_ns(),
            "creator_pid": os.getpid(),
            "endpoint": asdict(endpoint),
            "original_settings_raw": dict(original_settings),
            "initial_state": dict(initial_state),
            "trace_path": str(trace_path.resolve()),
            "position_recovery_note": (
                "Position is intentionally not auto-restored after an abnormal stop; "
                "the guardian stops motion and restores configuration registers only."
            ),
        },
    )


def run(args: argparse.Namespace) -> int:
    if not args.apply or args.confirmation_token != MOTION_CONFIRMATION:
        raise RuntimeError(
            f"run requires --apply --confirmation-token {MOTION_CONFIRMATION}"
        )
    if not sys.stdin.isatty():
        raise RuntimeError("real gripper motion requires an interactive terminal")
    specs = build_trial_specs(args)
    endpoint, config = resolve_endpoint(args)
    validate_endpoint(endpoint)
    suite_confirmation = getattr(args, "_suite_operator_confirmation", None)
    if isinstance(suite_confirmation, Mapping):
        if str(suite_confirmation.get("side", "")) != endpoint.side:
            raise RuntimeError("suite confirmation side does not match endpoint")
        suite_id = str(suite_confirmation.get("suite_id", "")).strip()
        if not suite_id:
            raise RuntimeError("suite confirmation has no suite_id")
    if not 1.0 <= float(args.motion_timeout_sec) <= 30.0:
        raise ValueError("--motion-timeout-sec must be within [1,30]")
    if not 0.0 <= float(args.cooldown_sec) <= 60.0:
        raise ValueError("--cooldown-sec must be within [0,60]")
    if not 30.0 <= float(args.max_temperature_c) <= 80.0:
        raise ValueError("--max-temperature-c must be within [30,80]")
    if not 1024 <= int(args.queue_size) <= 1_000_000:
        raise ValueError("--queue-size must be within [1024,1000000]")
    if not 1.0 <= float(args.manual_gauge_timeout_sec) <= 300.0:
        raise ValueError("--manual-gauge-timeout-sec must be within [1,300]")
    output_resolution = resolve_output_root(args)
    root = output_resolution.path.resolve()
    with _run_locks(_run_lock_roots(root), endpoint):
        unfinished = [
            journal
            for journal_root in recovery_journal_roots(root)
            if journal_root.exists()
            for journal in _unfinished_journals(journal_root, endpoint)
        ]
        unfinished = sorted({path.resolve() for path in unfinished})
        if unfinished:
            raise RecoveryError(
                "unfinished recovery journal(s) exist; run the explicit restore command first: "
                + ", ".join(str(path) for path in unfinished)
            )
        users = find_device_users(endpoint.port)
        if users:
            raise RuntimeError(f"serial device is already open by {users}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root / f"{timestamp}_{endpoint.side}_{args.protocol}"
        run_dir.mkdir(parents=True, exist_ok=False)
        trace_path = run_dir / "gripper_trace.jsonl"
        metadata_path = run_dir / "metadata.json"
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        journal_path = run_dir / "gripper_settings.recovery.json"
        preflight_path = run_dir / "preflight.json"
        writer = AsyncJsonlWriter(trace_path, queue_size=args.queue_size)
        gripper = _open_gripper(endpoint)
        guardian: subprocess.Popen[str] | None = None
        guardian_pipe: Any | None = None
        restored = False
        writer_started = False
        normal_completion = False
        summaries: list[dict[str, Any]] = []
        stop_requested = threading.Event()

        def request_stop(_signum: int, _frame: Any) -> None:
            stop_requested.set()
            raise KeyboardInterrupt

        old_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            _exclusive_serial(gripper)
            if isinstance(suite_confirmation, Mapping):
                preflight_states = _wait_for_suite_preflight_idle(
                    gripper,
                    timeout_sec=min(3.0, float(args.motion_timeout_sec)),
                    max_temperature_c=float(args.max_temperature_c),
                    safe_open_position=float(args.open_position),
                    evidence_path=preflight_path,
                )
                initial_state = preflight_states[-1]
            else:
                initial_state = read_state_safely(gripper)
                preflight_states = [initial_state]
            original_settings = read_raw_settings(gripper)
            open_limits_raw = list(gripper.get_open_limits_raw())
            if not isinstance(suite_confirmation, Mapping) and bool(
                initial_state.get("is_moving", False)
            ):
                raise RuntimeError("gripper is moving at preflight; refusing to start")
            if int(initial_state.get("error_code", 0)) != 0:
                raise RuntimeError(
                    f"gripper error_code={initial_state.get('error_code')} at preflight; "
                    "inspect and clear the fault separately"
                )
            if float(initial_state.get("temperature_c", 0)) > args.max_temperature_c:
                raise RuntimeError("gripper temperature exceeds configured safety maximum")

            _prepare_journal(
                journal_path,
                endpoint=endpoint,
                original_settings=original_settings,
                initial_state=initial_state,
                trace_path=trace_path,
            )
            guardian, guardian_pipe = _start_guardian(journal_path)
            # Prove every original register can be written and read back before
            # applying any experimental setting.
            restore_raw_settings(gripper, original_settings)

            for signum in old_handlers:
                signal.signal(signum, request_stop)
            writer.start()
            writer_started = True
            metadata = {
                "schema": TRACE_SCHEMA,
                "endpoint": asdict(endpoint),
                "speed_fixed": SPEED_NORMALIZED,
                "artifact_storage": {
                    "requested_layout": getattr(args, "artifact_layout", None),
                    "selection_source": output_resolution.source,
                    "resolved_layout": output_resolution.layout_name,
                    "resolved_output_root": str(root),
                },
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                    if key not in {"func", "confirmation_token", "high_force_confirmation"}
                },
                "trials": [asdict(spec) for spec in specs],
                "original_settings_raw": original_settings,
                "initial_state": initial_state,
                "preflight_states": preflight_states,
                "open_limits_raw": open_limits_raw,
                "max_open_mm": float(gripper.max_open_mm),
                "config_created_at": config.get("created_at") if config else None,
                "force_units": (
                    "normalized command; current is device raw feedback; "
                    "Newtons require external gauge"
                ),
                "timing_semantics": {
                    "state_timestamp": "explicit_modbus_read_begin_end_midpoint",
                    "runtime_equivalent_command": (
                        "speed_force_continuous_preconfigured_then_position_only_write"
                    ),
                    "inline_config_command": "combined_config_and_position_call_bounds",
                    "host_resend": "forced_diagnostic_write_not_production_threshold_semantics",
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "pyserial": _package_version("pyserial"),
                    "git_revision": _git_revision(),
                    "script_sha256": _sha256_file(Path(__file__).resolve()),
                    "driver_sha256": _sha256_file(
                        DMTAC_ROOT / "portable_dmtac_gripper" / "gripper_driver.py"
                    ),
                    "config_sha256": (
                        _sha256_file(Path(endpoint.source))
                        if endpoint.source != "cli" and Path(endpoint.source).is_file()
                        else None
                    ),
                },
            }
            _atomic_write_json(metadata_path, metadata)
            writer.enqueue(_trace_row("run_start", metadata_path=str(metadata_path)))

            print(f"READY output={run_dir}", flush=True)
            if isinstance(suite_confirmation, Mapping):
                print(
                    f"SUITE confirmation active suite={suite_id} side={endpoint.side}; "
                    "free trials advance automatically (Ctrl-C to abort).",
                    flush=True,
                )
                writer.enqueue(
                    _trace_row(
                        "operator_global_confirmed",
                        confirmation_source="bimanual_position_suite",
                        suite_confirmation=dict(suite_confirmation),
                    )
                )
            else:
                _prompt_enter(
                    "Confirm: RM65 arm/replay/teleop are stopped; only the selected gripper may "
                    "move; hands are outside the jaws, the full free-motion path is clear, and an "
                    "emergency disconnect is reachable. This is the only global runtime "
                    "confirmation; free trials then advance automatically.",
                    gripper=gripper,
                    writer=writer,
                    spec=specs[0],
                    stage="global_operator_wait",
                    max_temperature_c=args.max_temperature_c,
                )
                writer.enqueue(
                    _trace_row(
                        "operator_global_confirmed",
                        confirmation_source="standalone_prompt",
                    )
                )

            for index, spec in enumerate(specs, start=1):
                if stop_requested.is_set():
                    raise KeyboardInterrupt
                print(
                    f"\n[{index}/{len(specs)}] {spec.trial_id}: force={spec.force:.3f}, "
                    f"continuous={spec.continuous_grasp}, host_hold={spec.host_position_hold}",
                    flush=True,
                )
                print("Preparing trial: the gripper will OPEN automatically.", flush=True)
                if args.command_mode == "runtime-equivalent":
                    _configure_motion(
                        gripper,
                        writer,
                        spec,
                        stage="prepare_open",
                        force=min(spec.force, 0.2),
                        continuous=False,
                    )
                _send_position(
                    gripper,
                    writer,
                    spec,
                    stage="prepare_open",
                    target=args.open_position,
                    force=min(spec.force, 0.2),
                    continuous=False,
                    command_mode=args.command_mode,
                )
                _wait_for_idle(
                    gripper,
                    writer,
                    spec,
                    stage="prepare_open",
                    timeout_sec=args.motion_timeout_sec,
                    sample_hz=args.sample_hz,
                    max_temperature_c=args.max_temperature_c,
                )

                if spec.protocol == "contact":
                    _prompt_enter(
                        f"Place ONLY the documented compliant test object ({args.object_label}, "
                        f"width={args.object_width_mm:g} mm) centrally between the jaws. "
                        "Do not use fingers, brittle, sharp, liquid-filled, or unrestrained "
                        "objects. "
                        "This is the only per-trial confirmation.",
                        gripper=gripper,
                        writer=writer,
                        spec=spec,
                        stage="object_ready_wait",
                        max_temperature_c=args.max_temperature_c,
                    )
                    writer.enqueue(
                        _trace_row("operator_object_ready", trial_id=spec.trial_id)
                    )
                else:
                    print(
                        "Free-motion path was confirmed globally; closing starts in 1 second "
                        "(Ctrl-C to abort).",
                        flush=True,
                    )
                    _monitored_delay(
                        1.0,
                        gripper=gripper,
                        writer=writer,
                        spec=spec,
                        stage="free_motion_countdown",
                        max_temperature_c=args.max_temperature_c,
                    )

                if args.command_mode == "runtime-equivalent":
                    _configure_motion(
                        gripper,
                        writer,
                        spec,
                        stage="trial_motion",
                        force=spec.force,
                        continuous=spec.continuous_grasp,
                    )
                writer.enqueue(_trace_row("trial_start", trial=asdict(spec)))
                samples: list[dict[str, Any]] = []
                resend_count = 0
                for segment_index, target in enumerate(spec.motion_targets):
                    writer.enqueue(
                        _trace_row(
                            "segment_start",
                            trial_id=spec.trial_id,
                            segment_index=segment_index,
                            target_position=float(target),
                            duration_sec=float(spec.duration_sec),
                        )
                    )
                    _send_position(
                        gripper,
                        writer,
                        spec,
                        stage="motion_initial",
                        target=target,
                        force=spec.force,
                        continuous=spec.continuous_grasp,
                        command_mode=args.command_mode,
                        segment_index=segment_index,
                    )
                    segment_samples, segment_resends = _observe_trial(
                        gripper,
                        writer,
                        spec,
                        sample_hz=args.sample_hz,
                        resend_hz=args.resend_hz,
                        target_position=target,
                        segment_index=segment_index,
                        max_temperature_c=args.max_temperature_c,
                    )
                    samples.extend(segment_samples)
                    resend_count += segment_resends
                    writer.enqueue(
                        _trace_row(
                            "segment_stop",
                            trial_id=spec.trial_id,
                            segment_index=segment_index,
                            target_position=float(target),
                            sample_count=len(segment_samples),
                            resend_count=segment_resends,
                        )
                    )
                force_measurement = _manual_force_measurement(
                    args,
                    spec,
                    gripper,
                    writer,
                    sample_hz=args.sample_hz,
                    max_temperature_c=args.max_temperature_c,
                )
                measured_force = (
                    None
                    if not force_measurement
                    else force_measurement.get("force_n")
                )
                summary = summarize_trial(
                    spec,
                    samples,
                    resend_count=resend_count,
                    measured_force_n=measured_force,
                    measured_force_evidence=force_measurement,
                )
                summaries.append(summary)
                writer.enqueue(_trace_row("trial_summary", summary=summary))

                print("Releasing: the gripper will OPEN now.", flush=True)
                if args.command_mode == "runtime-equivalent":
                    _configure_motion(
                        gripper,
                        writer,
                        spec,
                        stage="release_open",
                        force=min(spec.force, 0.2),
                        continuous=False,
                    )
                _send_position(
                    gripper,
                    writer,
                    spec,
                    stage="release_open",
                    target=args.open_position,
                    force=min(spec.force, 0.2),
                    continuous=False,
                    command_mode=args.command_mode,
                )
                _wait_for_idle(
                    gripper,
                    writer,
                    spec,
                    stage="release_open",
                    timeout_sec=args.motion_timeout_sec,
                    sample_hz=args.sample_hz,
                    max_temperature_c=args.max_temperature_c,
                )
                if spec.protocol == "contact":
                    print(
                        "Object released. Remove and inspect it now; the next trial will wait at "
                        "the single object-ready prompt.",
                        flush=True,
                    )
                if args.cooldown_sec > 0:
                    _monitored_delay(
                        args.cooldown_sec,
                        gripper=gripper,
                        writer=writer,
                        spec=spec,
                        stage="cooldown",
                        max_temperature_c=args.max_temperature_c,
                    )

            normal_completion = True
        except KeyboardInterrupt:
            stop_requested.set()
            raise RuntimeError("operator interrupted the run; gripper stop requested")
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            errors: list[str] = []
            if not normal_completion:
                try:
                    gripper.estop()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"stop failed: {type(exc).__name__}: {exc}")
            try:
                if journal_path.exists():
                    restore_raw_settings(gripper, _load_json(journal_path)["original_settings_raw"])
                    journal = _load_json(journal_path)
                    journal["status"] = "restored"
                    journal["restore_actor"] = "parent_finally"
                    journal["restored_settings_raw"] = read_raw_settings(gripper)
                    journal["restore_finished_wall_time_ns"] = time.time_ns()
                    _atomic_write_json(journal_path, journal)
                    restored = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"restore failed: {type(exc).__name__}: {exc}")
            try:
                gripper.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"serial close failed: {exc}")
            if writer_started:
                writer.enqueue(
                    _trace_row(
                        "run_stop",
                        normal_completion=normal_completion,
                        settings_restored=restored,
                        queue_dropped=writer.dropped,
                        errors=errors,
                    )
                )
                try:
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"writer close failed: {exc}")
                try:
                    _atomic_write_json(
                        summary_path,
                        {
                            "schema": TRACE_SCHEMA,
                            "normal_completion": normal_completion,
                            "settings_restored": restored,
                            "speed_fixed": SPEED_NORMALIZED,
                            "trial_count": len(summaries),
                            "trials": summaries,
                            "trace_rows": writer.written,
                            "trace_queue_dropped": writer.dropped,
                            "errors": list(errors),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"partial summary write failed: {exc}")
            if restored and guardian is not None and guardian_pipe is not None:
                try:
                    _notify_guardian_restored(guardian, guardian_pipe)
                    guardian_pipe = None
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"guardian close failed: {exc}")
            elif guardian_pipe is not None:
                guardian_pipe.close()
                guardian_pipe = None
            if errors:
                raise RecoveryError("; ".join(errors))

        if writer.dropped:
            raise RuntimeError(f"trace queue dropped {writer.dropped} rows")
        final_summary = {
            "schema": TRACE_SCHEMA,
            "normal_completion": normal_completion,
            "settings_restored": restored,
            "speed_fixed": SPEED_NORMALIZED,
            "trial_count": len(summaries),
            "trials": summaries,
            "trace_rows": writer.written,
            "trace_queue_dropped": writer.dropped,
        }
        _atomic_write_json(summary_path, final_summary)
        evidence_paths = [
            preflight_path,
            metadata_path,
            trace_path,
            summary_path,
            journal_path,
            _guardian_ready_path(journal_path),
            journal_path.with_suffix(".guardian.log"),
        ]
        _atomic_write_json(
            manifest_path,
            {
                "schema": MANIFEST_SCHEMA,
                "trace_schema": TRACE_SCHEMA,
                "normal_completion": normal_completion,
                "settings_restored": restored,
                "trial_count": len(summaries),
                "trace_rows": writer.written,
                "trace_queue_dropped": writer.dropped,
                "files": {
                    path.name: {
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in evidence_paths
                    if path.is_file()
                },
            },
        )
        print(
            f"COMPLETE settings_restored=true trials={len(summaries)} "
            f"manifest={manifest_path} output={run_dir}",
            flush=True,
        )
        return 0


def manual_restore(args: argparse.Namespace) -> int:
    if not args.apply or args.confirmation_token != RESTORE_CONFIRMATION:
        raise RuntimeError(
            f"restore requires --apply --confirmation-token {RESTORE_CONFIRMATION}"
        )
    restore_journal(args.journal.resolve(), actor="manual_restore", stop_first=True)
    print(f"RESTORED verified=true journal={args.journal.resolve()}")
    return 0


def guardian(args: argparse.Namespace) -> int:
    journal = _load_json(args.journal)
    if journal.get("schema") != JOURNAL_SCHEMA:
        raise RecoveryError("guardian received unsupported journal")
    _endpoint_from_journal(journal)
    if not isinstance(journal.get("original_settings_raw"), Mapping):
        raise RecoveryError("guardian journal has no original settings")
    _atomic_write_json(
        _guardian_ready_path(args.journal),
        {
            "schema": JOURNAL_SCHEMA,
            "guardian_pid": os.getpid(),
            "ready_wall_time_ns": time.time_ns(),
        },
    )
    token = sys.stdin.readline().strip()
    if token == "RESTORED":
        return 0
    try:
        restore_journal(args.journal, actor="detached_guardian", stop_first=True)
    except Exception as exc:  # noqa: BLE001
        print(f"GUARDIAN RESTORE FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(f"GUARDIAN RESTORED {args.journal}", flush=True)
    return 0


def _add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--config", default="latest")
    parser.add_argument("--port", default=None, help="Explicit serial port; overrides --config.")
    parser.add_argument("--gripper-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.05)


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", choices=("free", "contact"), default="free")
    parser.add_argument("--force-levels", default="0.10,0.20,0.35")
    parser.add_argument("--continuous-modes", default="false,true")
    parser.add_argument(
        "--free-continuous-grasp",
        action="store_true",
        help=(
            "Use device continuous-grasp mode during free-motion trials. "
            "This is separate from the contact protocol --continuous-modes matrix."
        ),
    )
    parser.add_argument("--host-hold-modes", default="one-shot,resend")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-position", type=float, default=1.0)
    parser.add_argument(
        "--target-sequence",
        default="",
        help=(
            "Free-motion target sequence, for example 0.25,0.5,0.75,1,0.75,0.5,0.25,0. "
            "Each target is observed for --free-duration-sec."
        ),
    )
    parser.add_argument("--open-position", type=float, default=0.0)
    parser.add_argument("--free-duration-sec", type=float, default=3.0)
    parser.add_argument("--hold-duration-sec", type=float, default=5.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--resend-hz", type=float, default=15.0)
    parser.add_argument(
        "--command-mode",
        choices=COMMAND_MODES,
        default="runtime-equivalent",
        help=(
            "runtime-equivalent pre-applies speed/force/continuous and times only the "
            "position write; inline-config preserves the historical combined call."
        ),
    )
    parser.add_argument("--object-label", default="")
    parser.add_argument("--object-width-mm", type=float, default=0.0)
    parser.add_argument(
        "--external-force-reference",
        choices=("none", "manual-gauge"),
        default="none",
    )
    parser.add_argument("--allow-high-force", action="store_true")
    parser.add_argument("--high-force-confirmation", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print experiment matrix; no hardware access.")
    _add_protocol_args(plan_parser)
    plan_parser.set_defaults(func=plan)

    inspect_parser = subparsers.add_parser("inspect", help="Read one gripper; no writes or motion.")
    _add_endpoint_args(inspect_parser)
    inspect_parser.set_defaults(func=inspect)

    run_parser = subparsers.add_parser("run", help="Run interactive one-gripper motion trials.")
    _add_endpoint_args(run_parser)
    _add_protocol_args(run_parser)
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Exact output root. When provided it overrides --artifact-layout; "
            "otherwise the validated local managed-layout default is used."
        ),
    )
    run_parser.add_argument(
        "--artifact-layout",
        choices=(
            LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
        default=configured_runtime_artifact_layout(),
        help=(
            "Select the managed output layout when --output-root is omitted. "
            "The repository default is legacy; local_paths.json can coordinate "
            "a project-wide canonical cutover."
        ),
    )
    run_parser.add_argument("--motion-timeout-sec", type=float, default=5.0)
    run_parser.add_argument("--cooldown-sec", type=float, default=2.0)
    run_parser.add_argument("--max-temperature-c", type=float, default=55.0)
    run_parser.add_argument("--manual-gauge-timeout-sec", type=float, default=30.0)
    run_parser.add_argument("--queue-size", type=int, default=16384)
    run_parser.add_argument("--apply", action="store_true")
    run_parser.add_argument("--confirmation-token", default="")
    run_parser.set_defaults(func=run)

    restore_parser = subparsers.add_parser(
        "restore", help="Stop one gripper and restore a recovery journal."
    )
    restore_parser.add_argument("--journal", type=Path, required=True)
    restore_parser.add_argument("--apply", action="store_true")
    restore_parser.add_argument("--confirmation-token", default="")
    restore_parser.set_defaults(func=manual_restore)

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
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
