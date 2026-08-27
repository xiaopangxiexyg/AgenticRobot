from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand
from ...dmtac_tactile_fields import (
    DMTAC_DEFAULT_TACTILE_FIELDS,
    DMTAC_DISTRIBUTED_FORCE_SHAPE,
    DMTAC_RAW_IMAGE_SHAPE,
    DMTAC_TACTILE_ARRAY_FIELDS,
    DMTAC_TACTILE_CALIBRATED_FIELDS,
    DMTAC_TACTILE_FIELD_ALIASES,
    DMTAC_TACTILE_FIELD_DEFAULT_SHAPES,
    DMTAC_TACTILE_FIELDS,
    _normalize_dmtac_tactile_fields,
    dmtac_tactile_requires_calibration,
    normalize_dmtac_tactile_fields,
)


DEFAULT_DMTAC_GRIPPER_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
DEFAULT_DMTAC_GRIPPER_SPEED = 1.0
DEFAULT_DMTAC_GRIPPER_TORQUE = 0.2
DEFAULT_DMTAC_GRIPPER_COMMAND_HZ = 20.0
DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ = 20.0
GRIPPER_ACTION_MAX = 1.0
# A state-register read takes about 10 ms on the validated 115200-baud setup.
# Keep a short write-to-read guard, but do not hide the physical response for
# multiple collection frames after every accepted command.
DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S = 0.02
DMTAC_TELEMETRY_STOP_TIMEOUT_S = 2.0
DMTAC_TACTILE_CLOSE_GRACE_S = 3.0
DMTAC_TACTILE_GLOBAL_READ_GATE = threading.Semaphore(1)
DMTAC_TACTILE_PROCESS_STOP_TIMEOUT_S = 3.0
DMTAC_TACTILE_PROCESS_QUEUE_SIZE = 2
_DMTAC_TACTILE_PROCESS_GATE: Any | None = None


def _tactile_process_gate(ctx: Any) -> Any:
    global _DMTAC_TACTILE_PROCESS_GATE
    if _DMTAC_TACTILE_PROCESS_GATE is None:
        _DMTAC_TACTILE_PROCESS_GATE = ctx.Semaphore(1)
    return _DMTAC_TACTILE_PROCESS_GATE


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _set_worker_process_name(name: str) -> None:
    if not sys.platform.startswith("linux"):
        return
    text = str(name or "dmtac_tactile")[:15]
    try:
        Path("/proc/self/comm").write_text(text)
    except OSError:
        pass


def _isolate_tactile_worker_terminal_interrupt() -> None:
    """Make the parent the sole Ctrl+C lifecycle owner for this worker."""

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        pass


def _start_parent_watchdog(stop_event: Any, parent_pid: int, *, poll_s: float = 0.5) -> None:
    if int(parent_pid) <= 0:
        return

    def _watch_parent() -> None:
        while not stop_event.wait(float(poll_s)):
            if os.getppid() != int(parent_pid) or not _process_exists(int(parent_pid)):
                stop_event.set()
                return

    thread = threading.Thread(
        target=_watch_parent,
        name="dmtac_parent_watchdog",
        daemon=True,
    )
    thread.start()


def _put_latest(queue_out: Any, payload: dict[str, Any]) -> None:
    while True:
        try:
            queue_out.put_nowait(payload)
            return
        except queue.Full:
            try:
                queue_out.get_nowait()
            except queue.Empty:
                return


def _dmtac_tactile_array_fields(fields: Any) -> tuple[str, ...]:
    return tuple(
        field
        for field in _normalize_dmtac_tactile_fields(fields)
        if field in DMTAC_TACTILE_ARRAY_FIELDS
    )


def _dmtac_tactile_output_key(dev_id: Any, field: str, fields: tuple[str, ...]) -> str:
    base = str(dev_id)
    if len(fields) <= 1:
        return base
    return f"{base}:{field}"


def _dmtac_frame_id_for_field(frame: Any, field: str) -> int | None:
    fids = getattr(frame, "fids", None)
    if isinstance(fids, dict) and field in fids:
        return fids.get(field)
    return getattr(frame, "fid", None)


def _dmtac_frame_array(frame: Any, field: str) -> np.ndarray | None:
    value = getattr(frame, field, None)
    if value is None:
        return None
    if hasattr(value, "img"):
        value = getattr(value, "img", None)
        if value is None:
            return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    if field == "distributed_force":
        if array.ndim != 3 or array.shape[2] < 2:
            return None
        return np.ascontiguousarray(array, dtype=np.float32)
    if field in {"deformation2d", "shear"}:
        if array.ndim != 3 or array.shape[2] < 2:
            return None
        return np.ascontiguousarray(array, dtype=np.float32)
    if field == "depth":
        if array.ndim < 2:
            return None
        return np.ascontiguousarray(array, dtype=np.float32)
    if field == "force":
        vector = array.reshape(-1)
        if vector.size == 0:
            return None
        return np.ascontiguousarray(vector, dtype=np.float32)
    if field in {"raw", "infer"}:
        if array.ndim < 2:
            return None
        return np.ascontiguousarray(array)
    return None


def _dmtac_tactile_shape_per_sensor(
    shapes: dict[str, tuple[int, ...]],
    fields: Any,
) -> tuple[int, ...] | None:
    if shapes:
        return tuple(next(iter(shapes.values())))
    for field in _dmtac_tactile_array_fields(fields):
        shape = DMTAC_TACTILE_FIELD_DEFAULT_SHAPES.get(field)
        if shape is not None:
            return tuple(shape)
    return None


def _frames_payload(
    frames: dict[Any, Any],
    *,
    fields: Any = DMTAC_DEFAULT_TACTILE_FIELDS,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, int | None],
    dict[str, tuple[int, ...]],
    dict[str, dict[str, float]],
    dict[str, str],
]:
    tactile_dict: dict[str, np.ndarray] = {}
    frame_ids: dict[str, int | None] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    timings_by_device: dict[str, dict[str, float]] = {}
    field_by_key: dict[str, str] = {}
    array_fields = _dmtac_tactile_array_fields(fields)
    for dev_id, frame in frames.items():
        if frame is None:
            continue
        for field in array_fields:
            array = _dmtac_frame_array(frame, field)
            if array is None:
                continue
            key = _dmtac_tactile_output_key(dev_id, field, array_fields)
            tactile_dict[key] = array.copy()
            shapes[key] = tuple(array.shape)
            frame_ids[key] = _dmtac_frame_id_for_field(frame, field)
            field_by_key[key] = field
        key = str(dev_id)
        timings = getattr(frame, "timings_ms", None)
        if isinstance(timings, dict):
            device_timings: dict[str, float] = {}
            for name, value in timings.items():
                try:
                    device_timings[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
            if device_timings:
                timings_by_device[key] = device_timings
    return tactile_dict, frame_ids, shapes, timings_by_device, field_by_key


def _dmtac_tactile_worker_main(
    config_payload: dict[str, Any],
    queue_out: Any,
    stop_event: Any,
    read_gate: Any,
) -> None:
    # The terminal foreground process group includes multiprocessing workers.
    # Ctrl+C belongs to the real-teleop parent, which owns ordered shutdown and
    # signals this worker through ``stop_event``.  Letting the child also raise
    # KeyboardInterrupt can abort the vendor SDK disconnect, race the restart
    # supervisor, and leave the parent waiting on a half-closed worker.
    _isolate_tactile_worker_terminal_interrupt()
    _set_worker_process_name(str(config_payload.get("worker_name") or "dmtac_tactile"))
    _start_parent_watchdog(
        stop_event,
        int(config_payload.get("parent_pid") or 0),
    )
    try:
        os.nice(int(config_payload.get("tactile_process_nice", 12)))
    except Exception:
        pass
    worker_cpus = tuple(config_payload.get("tactile_process_cpu_affinity") or ())
    if worker_cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(cpu) for cpu in worker_cpus})
        except Exception:
            pass
    root = Path(config_payload["dmtac_root"])
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from portable_dmtac_gripper import (
            DMTacGripperDevice,
            TactileConfig,
            TactileDenoiseConfig,
        )

        tactile = TactileConfig(
            dev_ids=list(config_payload.get("tactile_ids") or []) or None,
            backend=str(config_payload.get("tactile_backend", "cpu")),
            mode=str(config_payload.get("tactile_mode", "standard")),
            max_fps=int(config_payload.get("tactile_max_fps", 30)),
            fields=normalize_dmtac_tactile_fields(
                config_payload.get("tactile_fields", DMTAC_DEFAULT_TACTILE_FIELDS)
            ),
            compute_profile=str(config_payload.get("tactile_compute_profile", "dist-only")),
            calibration_policy=str(config_payload.get("tactile_calibration_policy", "auto")),
            calibration_dir=config_payload.get("tactile_calibration_dir"),
            parallel_read=bool(config_payload.get("tactile_parallel_read", False)),
            denoise_distributed_force=bool(config_payload.get("tactile_denoise", False)),
            denoise_config=TactileDenoiseConfig(
                enabled=bool(config_payload.get("tactile_denoise", False)),
                mode=str(config_payload.get("tactile_denoise_mode", "delta")),
                baseline_frames=int(config_payload.get("tactile_baseline_frames", 20)),
                deadband_abs=float(config_payload.get("tactile_deadband_abs", 5e-5)),
                temporal_alpha=float(config_payload.get("tactile_temporal_alpha", 0.85)),
                spatial_blur_ksize=int(config_payload.get("tactile_spatial_blur_ksize", 5)),
            ),
        )
        device = DMTacGripperDevice(
            gripper=None,
            tactile=tactile,
            connect_gripper=False,
            connect_tactile=True,
            apply_tactile_calibration_on_connect=bool(
                config_payload.get("apply_tactile_calibration_on_connect", True)
            ),
        ).connect()
        try:
            if bool(config_payload.get("tactile_reset_on_start", False)):
                device.reset_tactile()
            ready = {
                str(key): bool(value)
                for key, value in device.wait_tactile_ready(
                    timeout_s=float(config_payload.get("tactile_ready_timeout_s", 10.0))
                ).items()
            }
            _put_latest(queue_out, {"type": "ready", "ready": ready, "error": None})
            warmup_frames = int(config_payload.get("tactile_warmup_frames", 0))
            if warmup_frames > 0:
                device.warmup_tactile_filter(
                    frames=warmup_frames,
                    timeout_ms=int(config_payload.get("tactile_timeout_ms", 100)),
                )
            read_hz = float(config_payload.get("tactile_read_hz", 1.0))
            period_s = 0.0 if read_hz <= 0.0 else 1.0 / read_hz
            next_read = time.perf_counter()
            version = 0
            while not stop_event.is_set():
                if period_s > 0.0:
                    wait_s = next_read - time.perf_counter()
                    if wait_s > 0.0 and stop_event.wait(wait_s):
                        break
                scheduled_read_start = next_read
                wall_started = time.time()
                perf_started = time.perf_counter()
                gate_wait_s = None
                gate_acquired = False
                try:
                    if bool(config_payload.get("tactile_global_read_lock", True)):
                        gate_started = time.perf_counter()
                        read_gate.acquire()
                        gate_acquired = True
                        gate_wait_s = time.perf_counter() - gate_started
                    frames = device.read_tactile(
                        timeout_ms=int(config_payload.get("tactile_timeout_ms", 100)),
                        wait=bool(config_payload.get("tactile_wait_for_new", False)),
                        require_ready=bool(config_payload.get("tactile_require_ready", True)),
                    )
                finally:
                    if gate_acquired:
                        read_gate.release()
                read_elapsed_s = time.perf_counter() - perf_started
                tactile_fields = normalize_dmtac_tactile_fields(
                    config_payload.get("tactile_fields", DMTAC_DEFAULT_TACTILE_FIELDS)
                )
                tactile_dict, frame_ids, shapes, timings_ms, field_by_key = _frames_payload(
                    frames,
                    fields=tactile_fields,
                )
                if tactile_dict:
                    version += 1
                _put_latest(
                    queue_out,
                    {
                        "type": "frame",
                        "timestamp": wall_started if tactile_dict else None,
                        "tactile_dict": tactile_dict,
                        "frame_ids": frame_ids,
                        "shapes": shapes,
                        "timings_ms": timings_ms,
                        "field_by_key": field_by_key,
                        "fields": list(tactile_fields),
                        "devices": list(tactile_dict),
                        "version": version,
                        "read_elapsed_s": read_elapsed_s,
                        "gate_wait_s": gate_wait_s,
                        "error": None,
                    },
                )
                if period_s > 0.0:
                    next_read = scheduled_read_start + period_s
                    now_after_read = time.perf_counter()
                    if next_read < now_after_read - period_s:
                        next_read = now_after_read
        finally:
            device.close()
    except Exception as exc:  # noqa: BLE001
        _put_latest(
            queue_out,
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


@dataclass
class DMTacGripperBackendConfig:
    side: str
    effector_name: str | None = None
    gripper_port: str = DEFAULT_DMTAC_GRIPPER_PORT
    gripper_id: int = 1
    gripper_baudrate: int = 115200
    gripper_timeout: float = 0.10
    max_open_mm: float = 70.0
    connect_hardware: bool = True
    # Deployment-only live observation shadow.  It permits telemetry/tactile
    # reads but suppresses all startup and command-side gripper writes.
    command_dim: int = 0
    send_commands: bool = True
    send_threshold: float = 0.002
    command_hz: float = DEFAULT_DMTAC_GRIPPER_COMMAND_HZ
    max_action: float = GRIPPER_ACTION_MAX
    speed: float = DEFAULT_DMTAC_GRIPPER_SPEED
    torque: float = DEFAULT_DMTAC_GRIPPER_TORQUE
    continuous_grasp: bool = True
    clear_fault_on_start: bool = True
    open_on_start: bool = False
    reconnect_on_error: bool = True
    reconnect_cooldown_sec: float = 1.0
    enable_telemetry: bool = True
    telemetry_hz: float = DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ
    gripper_telemetry_command_grace_s: float = DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S
    connect_tactile: bool = True
    tactile_ids: tuple[str | int, ...] = ()
    tactile_backend: str = "cpu"
    tactile_mode: str = "standard"
    tactile_max_fps: int = 30
    tactile_fields: tuple[str, ...] = DMTAC_DEFAULT_TACTILE_FIELDS
    tactile_compute_profile: str = "dist-only"
    tactile_calibration_policy: str = "auto"
    tactile_calibration_dir: str | Path | None = None
    tactile_timeout_ms: int = 100
    tactile_wait_for_new: bool = False
    tactile_read_hz: float = 1.0
    tactile_process_read: bool = True
    tactile_process_nice: int = 12
    tactile_process_cpu_affinity: tuple[int, ...] = ()
    tactile_process_queue_size: int = DMTAC_TACTILE_PROCESS_QUEUE_SIZE
    tactile_worker_restart_on_error: bool = True
    tactile_worker_restart_initial_backoff_sec: float = 0.5
    tactile_worker_restart_max_backoff_sec: float = 5.0
    tactile_parallel_read: bool = False
    tactile_global_read_lock: bool = True
    tactile_require_ready: bool = True
    tactile_reset_on_start: bool | None = False
    tactile_ready_timeout_s: float = 10.0
    tactile_warmup_frames: int = 0
    tactile_denoise: bool = False
    tactile_store_flat_vector: bool = False
    tactile_denoise_mode: str = "delta"
    tactile_baseline_frames: int = 20
    tactile_deadband_abs: float = 5e-5
    tactile_temporal_alpha: float = 0.85
    tactile_spatial_blur_ksize: int = 5
    dmtac_root: str | Path | None = None
    debug: bool = False
    # Kept after the existing public fields so legacy positional construction
    # keeps its old meaning.
    observation_only: bool = False
    # Explicit deployment-only no-I/O mode.  The configured scalar is an
    # operator-provided assumption used to preserve the model's fixed DMTAC
    # state layout when that physical gripper is intentionally disconnected.
    # It is never reported as measured hardware feedback.
    disconnected_state_scalar: float | None = None

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_hand"
        if int(self.gripper_id) <= 0:
            raise ValueError("gripper_id must be positive.")
        if int(self.gripper_baudrate) <= 0:
            raise ValueError("gripper_baudrate must be positive.")
        if float(self.gripper_timeout) <= 0.0:
            raise ValueError("gripper_timeout must be positive.")
        if float(self.max_open_mm) <= 0.0:
            raise ValueError("max_open_mm must be positive.")
        if int(self.command_dim) != 0:
            raise ValueError("DM-Tac gripper command_dim is fixed to hand slot 0.")
        # Historical profiles sometimes capped this at 0.9/0.95.  That was a
        # hardware-era workaround, not a distinct action space.  Keep one
        # canonical DM-Tac command/state range for collection and deployment.
        self.max_action = float(GRIPPER_ACTION_MAX)
        if float(self.command_hz) <= 0.0:
            raise ValueError("command_hz must be positive.")
        if float(self.telemetry_hz) <= 0.0:
            raise ValueError("telemetry_hz must be positive.")
        if float(self.gripper_telemetry_command_grace_s) < 0.0:
            raise ValueError("gripper_telemetry_command_grace_s must be non-negative.")
        if float(self.reconnect_cooldown_sec) < 0.0:
            raise ValueError("reconnect_cooldown_sec must be non-negative.")
        if self.disconnected_state_scalar is not None:
            disconnected_scalar = float(self.disconnected_state_scalar)
            if (
                not np.isfinite(disconnected_scalar)
                or not 0.0 <= disconnected_scalar <= 1.0
            ):
                raise ValueError("disconnected_state_scalar must be finite and in [0, 1].")
            if self.connect_hardware or self.connect_tactile or self.send_commands:
                raise ValueError(
                    "disconnected_state_scalar requires connect_hardware=False, "
                    "connect_tactile=False, and send_commands=False."
                )
            self.disconnected_state_scalar = disconnected_scalar
            self.enable_telemetry = False
            self.reconnect_on_error = False
            self.clear_fault_on_start = False
            self.open_on_start = False
            self.tactile_reset_on_start = False
        if int(self.tactile_max_fps) <= 0:
            raise ValueError("tactile_max_fps must be positive.")
        self.tactile_fields = normalize_dmtac_tactile_fields(self.tactile_fields)
        # A solved tactile field does not imply permission to capture a new
        # no-contact base.  Resetting changes the SDK inference state and must
        # therefore be an explicit opt-in, independent of the requested fields.
        self.tactile_reset_on_start = bool(self.tactile_reset_on_start)
        profile = str(self.tactile_compute_profile).strip().lower().replace("_", "-")
        if profile not in {"dist-only", "legacy", "requested", "shear-depth"}:
            raise ValueError("tactile_compute_profile must be dist-only, legacy, requested, or shear-depth.")
        self.tactile_compute_profile = profile
        calibration_policy = str(self.tactile_calibration_policy or "auto").strip().lower()
        if calibration_policy not in {"auto", "required", "off"}:
            raise ValueError("tactile_calibration_policy must be auto, required, or off.")
        self.tactile_calibration_policy = calibration_policy
        if self.tactile_calibration_dir is not None:
            self.tactile_calibration_dir = Path(self.tactile_calibration_dir)
        if int(self.tactile_timeout_ms) <= 0:
            raise ValueError("tactile_timeout_ms must be positive.")
        if float(self.tactile_read_hz) < 0.0:
            raise ValueError("tactile_read_hz must be >= 0; use 0 to read every telemetry cycle.")
        if not 0 <= int(self.tactile_process_nice) <= 19:
            raise ValueError("tactile_process_nice must be in [0, 19].")
        if int(self.tactile_process_queue_size) <= 0:
            raise ValueError("tactile_process_queue_size must be positive.")
        if float(self.tactile_worker_restart_initial_backoff_sec) < 0.0:
            raise ValueError("tactile_worker_restart_initial_backoff_sec must be non-negative.")
        if float(self.tactile_worker_restart_max_backoff_sec) < float(
            self.tactile_worker_restart_initial_backoff_sec
        ):
            raise ValueError(
                "tactile_worker_restart_max_backoff_sec must be >= initial backoff."
            )
        cpus = tuple(int(cpu) for cpu in self.tactile_process_cpu_affinity)
        if any(cpu < 0 for cpu in cpus):
            raise ValueError("tactile_process_cpu_affinity CPU indices must be non-negative.")
        cpu_count = os.cpu_count()
        if cpu_count is not None and any(cpu >= int(cpu_count) for cpu in cpus):
            raise ValueError(
                f"tactile_process_cpu_affinity CPU indices must be < {int(cpu_count)}."
            )
        self.tactile_process_cpu_affinity = tuple(dict.fromkeys(cpus))
        if float(self.tactile_ready_timeout_s) <= 0.0:
            raise ValueError("tactile_ready_timeout_s must be positive.")
        if int(self.tactile_warmup_frames) < 0:
            raise ValueError("tactile_warmup_frames must be non-negative.")
        if int(self.tactile_baseline_frames) <= 0:
            raise ValueError("tactile_baseline_frames must be positive.")
        self.speed = float(np.clip(float(self.speed), 0.0, 1.0))
        self.torque = float(np.clip(float(self.torque), 0.0, 1.0))
        self.max_action = float(np.clip(float(self.max_action), 0.0, GRIPPER_ACTION_MAX))
        self.tactile_ids = tuple(self.tactile_ids)
        if self.dmtac_root is None:
            self.dmtac_root = Path(__file__).resolve().parents[3] / "hardware" / "DMTAC_gripper"
        else:
            self.dmtac_root = Path(self.dmtac_root)


@register_backend("dmtac_gripper")
class DMTacGripperBackend(EffectorBackend):
    """Runtime backend for an Inspire serial gripper with DM-Tac force images.

    The physical end-effector is a gripper, while the default logical effector
    name remains ``<side>_hand`` so existing collection/deploy action vectors can
    keep using their 6-wide hand slot.
    """

    def __init__(self, config: DMTacGripperBackendConfig):
        self.config = config
        self.device: Any | None = None
        self._lock = threading.Lock()
        self._tactile_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._last_command: EffectorCommand | None = None
        self._last_sent_command: np.ndarray | None = None
        self._last_sent_command_timestamp: float | None = None
        self._last_sent_command_completed_timestamp: float | None = None
        self._latest_joint_pos: np.ndarray | None = None
        self._latest_measured_joint_pos: np.ndarray | None = None
        self._latest_joint_timestamp: float | None = None
        self._latest_gripper_error_code: int | None = None
        self._latest_gripper_status_code: int | None = None
        self._latest_gripper_status_text: str | None = None
        self._latest_gripper_is_moving: bool | None = None
        self._latest_gripper_current: int | None = None
        self._latest_gripper_temperature_c: int | None = None
        self._latest_tactile: np.ndarray | None = None
        self._latest_tactile_timestamp: float | None = None
        self._latest_tactile_dict: dict[str, np.ndarray] | None = None
        self._latest_tactile_devices: list[str] = []
        self._latest_tactile_frame_ids: dict[str, int | None] = {}
        self._latest_tactile_shapes: dict[str, tuple[int, ...]] = {}
        self._latest_tactile_timings_ms: dict[str, dict[str, float]] = {}
        self._latest_tactile_field_by_key: dict[str, str] = {}
        self._latest_tactile_version: int = 0
        self._latest_telemetry_timestamp: float | None = None
        self._latest_gripper_read_elapsed_s: float | None = None
        self._latest_gripper_read_skipped = False
        self._latest_gripper_read_skip_reason: str | None = None
        self._latest_gripper_read_forced = False
        self._latest_tactile_read_elapsed_s: float | None = None
        self._latest_tactile_gate_wait_s: float | None = None
        self._latest_tactile_read_skipped = False
        self._last_tactile_read_finished_at: float | None = None
        self._next_tactile_read_at: float | None = None
        self._latest_telemetry_elapsed_s: float | None = None
        self._telemetry_error: str | None = None
        self._tactile_ready: dict[str, bool] = {}
        self._stop_event: threading.Event | None = None
        self._telemetry_thread: threading.Thread | None = None
        self._tactile_worker_process: Any | None = None
        self._tactile_worker_queue: Any | None = None
        self._tactile_worker_stop_event: Any | None = None
        self._tactile_worker_receiver_stop: threading.Event | None = None
        self._tactile_worker_receiver_thread: threading.Thread | None = None
        self._tactile_worker_restart_pending = False
        self._tactile_worker_failure_count = 0
        self._tactile_worker_restart_count = 0
        self._tactile_worker_last_restart_attempt_at: float | None = None
        self._tactile_worker_last_exit_code: int | None = None
        self._tactile_worker_observed_process: Any | None = None
        self._tactile_worker_supervisor_lock = threading.Lock()
        self._closing = False
        self._applied_speed: float | None = None
        self._applied_torque: float | None = None
        self._applied_continuous_grasp: bool | None = None
        self._last_send_attempt_timestamp: float | None = None
        self._last_send_elapsed_s: float | None = None
        self._last_send_lock_wait_s: float | None = None
        self._last_send_status: str | None = None
        self._last_send_skip_reason: str | None = None
        self._last_send_target_scalar: float | None = None
        self._last_reconnect_attempt_at: float | None = None
        self._last_reconnect_error: str | None = None
        self._reconnect_count = 0
        if self.config.disconnected_state_scalar is not None:
            self._latest_joint_pos = self._q6_from_scalar(
                float(self.config.disconnected_state_scalar),
                dim=int(self.config.command_dim),
                max_value=float(self.config.max_action),
            )
            self._latest_joint_timestamp = time.time()

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def _ensure_import_path(self) -> None:
        root = Path(self.config.dmtac_root)
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)

    def _create_device(self) -> Any:
        self._ensure_import_path()
        from portable_dmtac_gripper import (
            DMTacGripperDevice,
            GripperConfig,
            TactileConfig,
            TactileDenoiseConfig,
        )

        gripper = GripperConfig(
            port=str(self.config.gripper_port),
            gripper_id=int(self.config.gripper_id),
            baudrate=int(self.config.gripper_baudrate),
            timeout=float(self.config.gripper_timeout),
            max_open_mm=float(self.config.max_open_mm),
            reconnect_on_error=False,
            reconnect_attempts=0,
            debug=bool(self.config.debug),
        )
        tactile = TactileConfig(
            dev_ids=list(self.config.tactile_ids) or None,
            backend=str(self.config.tactile_backend),
            mode=str(self.config.tactile_mode),
            max_fps=int(self.config.tactile_max_fps),
            fields=normalize_dmtac_tactile_fields(self.config.tactile_fields),
            compute_profile=str(self.config.tactile_compute_profile),
            calibration_policy=str(self.config.tactile_calibration_policy),
            calibration_dir=self.config.tactile_calibration_dir,
            parallel_read=bool(self.config.tactile_parallel_read),
            denoise_distributed_force=bool(self.config.tactile_denoise),
            denoise_config=TactileDenoiseConfig(
                enabled=bool(self.config.tactile_denoise),
                mode=str(self.config.tactile_denoise_mode),
                baseline_frames=int(self.config.tactile_baseline_frames),
                deadband_abs=float(self.config.tactile_deadband_abs),
                temporal_alpha=float(self.config.tactile_temporal_alpha),
                spatial_blur_ksize=int(self.config.tactile_spatial_blur_ksize),
            ),
        )
        return DMTacGripperDevice(
            gripper=gripper,
            tactile=tactile,
            connect_gripper=bool(self.config.connect_hardware),
            connect_tactile=bool(
                self.config.connect_tactile
                and not self.config.tactile_process_read
            ),
            apply_tactile_calibration_on_connect=not bool(
                self.config.observation_only
            ),
        ).connect()

    def _create_gripper(self) -> Any:
        self._ensure_import_path()
        from portable_dmtac_gripper import InspireGripper

        return InspireGripper(
            port=str(self.config.gripper_port),
            gripper_id=int(self.config.gripper_id),
            baudrate=int(self.config.gripper_baudrate),
            timeout=float(self.config.gripper_timeout),
            max_open_mm=float(self.config.max_open_mm),
            reconnect_on_error=False,
            reconnect_attempts=0,
            debug=bool(self.config.debug),
        )

    def _reset_applied_motion_config_locked(self) -> None:
        self._applied_speed = None
        self._applied_torque = None
        self._applied_continuous_grasp = None

    def _invalidate_gripper_connection_state_locked(self) -> None:
        """Discard command caches that are valid only for one serial session."""

        self._reset_applied_motion_config_locked()
        self._last_sent_command = None
        self._last_sent_command_timestamp = None
        self._last_sent_command_completed_timestamp = None

    def _apply_gripper_start_config_locked(self, *, open_on_start: bool) -> None:
        if (
            self.config.observation_only
            or not self.config.connect_hardware
            or self.device is None
            or getattr(self.device, "gripper", None) is None
        ):
            return
        self._reset_applied_motion_config_locked()
        if self.config.clear_fault_on_start:
            self.device.clear_fault()
        self.device.set_speed(float(self.config.speed))
        self._applied_speed = float(self.config.speed)
        self.device.set_torque(float(self.config.torque))
        self._applied_torque = float(self.config.torque)
        if bool(open_on_start) and self.config.open_on_start:
            self.device.open(wait=False)

    def _close_gripper_locked(self) -> None:
        device = self.device
        if device is None:
            self._invalidate_gripper_connection_state_locked()
            return
        gripper = getattr(device, "gripper", None)
        if gripper is not None:
            try:
                close = getattr(gripper, "close", None)
                if callable(close):
                    close()
            finally:
                try:
                    setattr(device, "gripper", None)
                except Exception:
                    self.device = None
        self._invalidate_gripper_connection_state_locked()

    def _close_device_locked(self) -> None:
        device = self.device
        self.device = None
        self._invalidate_gripper_connection_state_locked()
        if device is not None:
            device.close()

    def _connect_gripper_locked(self) -> None:
        if self.device is None:
            self.device = self._create_device()
        elif hasattr(self.device, "gripper"):
            self._close_gripper_locked()
            setattr(self.device, "gripper", self._create_gripper())
        else:
            self._close_device_locked()
            self.device = self._create_device()
        self._apply_gripper_start_config_locked(open_on_start=False)

    def _quarantine_gripper_io_error_locked(
        self,
        prefix: str,
        exc: BaseException,
    ) -> None:
        error_text = f"{prefix}: {exc}"
        self._telemetry_error = error_text
        self._last_reconnect_error = error_text
        try:
            self._close_gripper_locked()
        except Exception as close_exc:  # noqa: BLE001 - keep control fail-closed.
            close_error = f"close_failed: {type(close_exc).__name__}: {close_exc}"
            self._telemetry_error = f"{error_text}; {close_error}"
            self._last_reconnect_error = close_error

    def _handle_gripper_io_error_locked(self, prefix: str, exc: BaseException) -> bool:
        error_text = f"{prefix}: {exc}"
        self._quarantine_gripper_io_error_locked(prefix, exc)
        if not self.config.connect_hardware or not self.config.reconnect_on_error:
            return False
        now = time.time()
        if (
            self._last_reconnect_attempt_at is not None
            and now - self._last_reconnect_attempt_at < float(self.config.reconnect_cooldown_sec)
        ):
            self._last_reconnect_error = error_text
            return False
        self._last_reconnect_attempt_at = now
        try:
            self._connect_gripper_locked()
            assert self.device is not None
            gripper_state = self.device.read_gripper_state()
        except Exception as reconnect_exc:  # noqa: BLE001
            try:
                self._close_gripper_locked()
            except Exception as close_exc:  # noqa: BLE001
                reconnect_exc = RuntimeError(
                    f"{reconnect_exc}; close_failed: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )
            self._last_reconnect_error = f"reconnect_failed: {reconnect_exc}"
            self._telemetry_error = f"{error_text}; {self._last_reconnect_error}"
            return False
        self._apply_gripper_state(gripper_state)
        self._reconnect_count += 1
        self._last_reconnect_error = None
        self._telemetry_error = None
        return True

    def _tactile_worker_config_payload(self) -> dict[str, Any]:
        return {
            "parent_pid": os.getpid(),
            "worker_name": f"dmtac_{self.config.side}_tac",
            "dmtac_root": str(self.config.dmtac_root),
            "tactile_ids": list(self.config.tactile_ids),
            "tactile_backend": str(self.config.tactile_backend),
            "tactile_mode": str(self.config.tactile_mode),
            "tactile_max_fps": int(self.config.tactile_max_fps),
            "tactile_fields": list(self.config.tactile_fields),
            "tactile_compute_profile": str(self.config.tactile_compute_profile),
            "tactile_calibration_policy": str(self.config.tactile_calibration_policy),
            "tactile_calibration_dir": (
                None
                if self.config.tactile_calibration_dir is None
                else str(self.config.tactile_calibration_dir)
            ),
            "tactile_timeout_ms": int(self.config.tactile_timeout_ms),
            "tactile_wait_for_new": bool(self.config.tactile_wait_for_new),
            "tactile_read_hz": float(self.config.tactile_read_hz),
            "tactile_process_nice": int(self.config.tactile_process_nice),
            "tactile_process_cpu_affinity": list(self.config.tactile_process_cpu_affinity),
            "tactile_process_queue_size": int(self.config.tactile_process_queue_size),
            "tactile_parallel_read": bool(self.config.tactile_parallel_read),
            "tactile_global_read_lock": bool(self.config.tactile_global_read_lock),
            "tactile_require_ready": bool(self.config.tactile_require_ready),
            "tactile_reset_on_start": bool(self.config.tactile_reset_on_start),
            "apply_tactile_calibration_on_connect": not bool(
                self.config.observation_only
            ),
            "tactile_ready_timeout_s": float(self.config.tactile_ready_timeout_s),
            "tactile_warmup_frames": int(self.config.tactile_warmup_frames),
            "tactile_denoise": bool(self.config.tactile_denoise),
            "tactile_denoise_mode": str(self.config.tactile_denoise_mode),
            "tactile_baseline_frames": int(self.config.tactile_baseline_frames),
            "tactile_deadband_abs": float(self.config.tactile_deadband_abs),
            "tactile_temporal_alpha": float(self.config.tactile_temporal_alpha),
            "tactile_spatial_blur_ksize": int(self.config.tactile_spatial_blur_ksize),
        }

    def _start_tactile_worker(self) -> None:
        if self._tactile_worker_process is not None:
            return
        ctx = mp.get_context("spawn")
        self._tactile_worker_queue = ctx.Queue(
            maxsize=max(1, int(self.config.tactile_process_queue_size))
        )
        self._tactile_worker_stop_event = ctx.Event()
        self._tactile_worker_process = ctx.Process(
            target=_dmtac_tactile_worker_main,
            args=(
                self._tactile_worker_config_payload(),
                self._tactile_worker_queue,
                self._tactile_worker_stop_event,
                _tactile_process_gate(ctx),
            ),
            name=f"{self.name}_dmtac_tactile_worker",
            daemon=True,
        )
        self._tactile_worker_process.start()
        self._tactile_worker_restart_pending = False
        self._start_tactile_worker_receiver()

    def _stop_tactile_worker_for_reconfigure(self) -> None:
        """Stop only tactile acquisition while leaving gripper control untouched."""

        stop_event = self._tactile_worker_stop_event
        receiver_stop = self._tactile_worker_receiver_stop
        receiver = self._tactile_worker_receiver_thread
        process = self._tactile_worker_process
        queue_out = self._tactile_worker_queue
        if stop_event is not None:
            stop_event.set()
        if receiver_stop is not None:
            receiver_stop.set()
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=1.0)
        if process is not None:
            process.join(timeout=DMTAC_TACTILE_PROCESS_STOP_TIMEOUT_S)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if queue_out is not None:
            try:
                queue_out.close()
                queue_out.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
        self._tactile_worker_process = None
        self._tactile_worker_queue = None
        self._tactile_worker_stop_event = None
        self._tactile_worker_receiver_stop = None
        self._tactile_worker_receiver_thread = None
        self._tactile_worker_observed_process = None
        self._tactile_worker_restart_pending = False
        with self._state_lock:
            self._latest_tactile = None
            self._latest_tactile_timestamp = None
            self._latest_tactile_dict = None
            self._latest_tactile_devices = []
            self._latest_tactile_field_by_key = {}
            self._tactile_ready = {}

    def enable_tactile_solving(self) -> bool:
        """Enable calibrated distributed-force solving with a worker-only restart.

        This operation is intentionally one-way for a runtime session. Keeping
        the solved field enabled after the first episode prevents the dataset
        field contract from changing when a preview window is merely closed.
        """

        if not self.config.connect_tactile:
            raise RuntimeError("DMTAC tactile acquisition is not configured for this side.")
        if not self.config.tactile_process_read:
            raise RuntimeError(
                "GUI tactile solving requires the isolated tactile worker process."
            )
        self._ensure_import_path()
        from portable_dmtac_gripper import (
            default_calibration_dir,
            load_tactile_calibration,
        )

        calibration_root = Path(
            self.config.tactile_calibration_dir or default_calibration_dir()
        )
        sensor_ids = tuple(self.config.tactile_ids)
        if not sensor_ids:
            raise RuntimeError(
                "Calibrated GUI solving requires explicit DMTAC sensor ids from the hardware assignment."
            )
        for sensor_id in sensor_ids:
            load_tactile_calibration(calibration_root, sensor_id, required=True)

        fields = tuple(self.config.tactile_fields)
        already_enabled = (
            "distributed_force" in fields
            and str(self.config.tactile_calibration_policy) == "required"
        )
        if already_enabled:
            return False

        old_fields = fields
        old_profile = str(self.config.tactile_compute_profile)
        old_policy = str(self.config.tactile_calibration_policy)
        old_dir = self.config.tactile_calibration_dir
        new_fields = tuple(dict.fromkeys((*fields, "distributed_force")))
        with self._tactile_worker_supervisor_lock:
            was_closing = self._closing
            self._closing = True
            try:
                self._stop_tactile_worker_for_reconfigure()
                self.config.tactile_fields = new_fields
                self.config.tactile_compute_profile = "dist-only"
                self.config.tactile_calibration_policy = "required"
                self.config.tactile_calibration_dir = calibration_root
                self._closing = was_closing
                self._start_tactile_worker()
            except Exception:
                self.config.tactile_fields = old_fields
                self.config.tactile_compute_profile = old_profile
                self.config.tactile_calibration_policy = old_policy
                self.config.tactile_calibration_dir = old_dir
                self._closing = was_closing
                if not was_closing and self._tactile_worker_process is None:
                    try:
                        self._start_tactile_worker()
                    except Exception as restore_exc:  # noqa: BLE001
                        self._telemetry_error = (
                            "tactile_solving_restore_failed:"
                            f"{type(restore_exc).__name__}:{restore_exc}"
                        )
                raise
            finally:
                self._closing = was_closing
        self._telemetry_error = None
        return True

    def _tactile_worker_receiver_active(self) -> bool:
        thread = self._tactile_worker_receiver_thread
        return bool(thread is not None and thread.is_alive())

    def _start_tactile_worker_receiver(self) -> None:
        if self._tactile_worker_receiver_active():
            return
        self._tactile_worker_receiver_stop = threading.Event()
        self._tactile_worker_receiver_thread = threading.Thread(
            target=self._tactile_worker_receiver_loop,
            name=f"{self.name}_dmtac_tactile_receiver",
            daemon=True,
        )
        self._tactile_worker_receiver_thread.start()

    def _tactile_worker_receiver_loop(self) -> None:
        stop_event = self._tactile_worker_receiver_stop
        queue_in = self._tactile_worker_queue
        if stop_event is None or queue_in is None:
            return
        while not stop_event.is_set():
            try:
                payload = queue_in.get(timeout=0.05)
            except queue.Empty:
                self._check_tactile_worker_process()
                process = self._tactile_worker_process
                if process is not None and not process.is_alive():
                    break
                continue
            except (EOFError, OSError):
                break
            if isinstance(payload, dict):
                self._apply_tactile_worker_payload(payload)
        self._check_tactile_worker_process()

    def start(self) -> None:
        if not self.config.connect_hardware and not self.config.connect_tactile:
            return
        if self.device is not None:
            return
        self._closing = False
        try:
            with self._lock:
                self.device = self._create_device()
                if not self.config.observation_only:
                    self._apply_gripper_start_config_locked(open_on_start=True)
                if (
                    self.config.connect_tactile
                    and self.config.tactile_process_read
                ):
                    self._start_tactile_worker()
                elif self.config.connect_tactile and self.device.tactile is not None:
                    if self.config.tactile_reset_on_start:
                        self.device.reset_tactile()
                    self._tactile_ready = {
                        str(key): bool(value)
                        for key, value in self.device.wait_tactile_ready(
                            timeout_s=float(self.config.tactile_ready_timeout_s)
                        ).items()
                    }
                    if int(self.config.tactile_warmup_frames) > 0:
                        self.device.warmup_tactile_filter(
                            frames=int(self.config.tactile_warmup_frames),
                            timeout_ms=int(self.config.tactile_timeout_ms),
                        )
            self._telemetry_error = None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                device = self.device
                self.device = None
            if device is not None:
                try:
                    device.close()
                except Exception as close_exc:  # noqa: BLE001
                    self._telemetry_error = (
                        f"connect_failed: {exc}; cleanup_failed: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )
                    return
            self._telemetry_error = f"connect_failed: {exc}"
            return

        if self.config.enable_telemetry:
            self._stop_event = threading.Event()
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name=f"{self.name}_dmtac_gripper_telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

    def is_ready(self) -> bool:
        if self.config.connect_hardware:
            return (
                self.device is not None
                and getattr(self.device, "gripper", None) is not None
                and self._latest_gripper_error_code in (None, 0)
            )
        if self.config.connect_tactile:
            return self.device is not None
        return True

    def send_command(self, command: EffectorCommand) -> bool:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for '{command.effector_name}'.")
        if command.command_type not in {"joint_vector", "gripper_scalar"}:
            raise ValueError(
                f"{self.name} supports command_type='joint_vector' or 'gripper_scalar', "
                f"got '{command.command_type}'."
            )
        if self.config.observation_only:
            raise RuntimeError(
                f"{self.name} is observation-only; refusing gripper command dispatch."
            )
        if self.config.disconnected_state_scalar is not None:
            target_scalar = self._extract_gripper_scalar(command.value)
            self._last_send_attempt_timestamp = time.time()
            self._last_send_elapsed_s = 0.0
            self._last_send_lock_wait_s = 0.0
            self._last_send_status = "intentionally_disconnected_noop"
            self._last_send_skip_reason = "configured_disconnected"
            self._last_send_target_scalar = float(target_scalar)
            return True
        sent = self._send_joint_pos(command.value, bypass_send_commands=False, force=False)
        if sent:
            self._last_command = command.copy()
        return sent

    def send_fixed_joint_pos(self, joint_pos: object) -> bool:
        if self.config.observation_only:
            raise RuntimeError(
                f"{self.name} is observation-only; refusing fixed gripper command dispatch."
            )
        if self.config.disconnected_state_scalar is not None:
            self._extract_gripper_scalar(joint_pos)
            self._last_send_attempt_timestamp = time.time()
            self._last_send_elapsed_s = 0.0
            self._last_send_lock_wait_s = 0.0
            self._last_send_status = "intentionally_disconnected_noop"
            self._last_send_skip_reason = "configured_disconnected"
            return True
        return self._send_joint_pos(joint_pos, bypass_send_commands=True, force=True)

    def set_torque(self, torque: float) -> float:
        """Set normalized gripper torque and apply it to an active device now.

        A normal position command can be skipped when its target has not
        changed, so applying the torque here keeps an explicit runtime update
        independent of the next gripper position change.  A disconnected or
        command-disabled backend retains the requested value for its next
        successful command.
        """

        value = float(torque)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("gripper torque must be a finite value in [0, 1].")
        if self.config.observation_only:
            return value
        with self._lock:
            previous_torque = float(self.config.torque)
            previous_applied_torque = self._applied_torque
            self.config.torque = value
            if (
                not self.config.connect_hardware
                or not self.config.send_commands
                or self.device is None
                or getattr(self.device, "gripper", None) is None
            ):
                return value
            try:
                self.device.set_torque(value)
            except Exception:
                self.config.torque = previous_torque
                self._applied_torque = previous_applied_torque
                raise
            self._applied_torque = value
        return value

    def force_reconnect_gripper(self, reason: str = "manual_reconnect") -> bool:
        with self._lock:
            previous_attempt_at = self._last_reconnect_attempt_at
            self._last_reconnect_attempt_at = None
            try:
                return self._handle_gripper_io_error_locked(
                    f"force_reconnect:{reason}",
                    RuntimeError(str(reason)),
                )
            finally:
                if self._last_reconnect_attempt_at is None:
                    self._last_reconnect_attempt_at = previous_attempt_at

    def read_gripper_state_now(self) -> dict[str, object] | None:
        if not self.config.connect_hardware:
            return None
        with self._lock:
            if self.device is None or getattr(self.device, "gripper", None) is None:
                recovered = self._handle_gripper_io_error_locked(
                    "read_now_no_connection",
                    RuntimeError("device is not connected"),
                )
                if not recovered:
                    return None
            assert self.device is not None
            try:
                gripper_state = self.device.read_gripper_state()
            except Exception as exc:  # noqa: BLE001
                recovered = self._handle_gripper_io_error_locked("read_now_failed", exc)
                if not recovered:
                    raise
                assert self.device is not None
                gripper_state = self.device.read_gripper_state()
            return self._apply_gripper_state(gripper_state)

    def _apply_gripper_state(
        self,
        gripper_state: object,
        *,
        fallback_timestamp: float | None = None,
    ) -> dict[str, object]:
        to_dict = getattr(gripper_state, "to_dict", None)
        payload = dict(to_dict()) if callable(to_dict) else {}

        def field(name: str, default: object = None) -> object:
            return payload.get(name, getattr(gripper_state, name, default))

        joint_value = float(field("joint_pos"))
        joint_pos = self._q6_from_scalar(
            joint_value,
            dim=int(self.config.command_dim),
            max_value=float(self.config.max_action),
        )
        timestamp = float(
            field(
                "timestamp",
                time.time() if fallback_timestamp is None else fallback_timestamp,
            )
        )
        error_code_raw = field("error_code")
        status_code_raw = field("status_code")
        current_raw = field("current")
        temperature_raw = field("temperature_c")
        with self._state_lock:
            self._latest_joint_pos = joint_pos
            self._latest_measured_joint_pos = joint_pos.copy()
            self._latest_joint_timestamp = timestamp
            self._latest_gripper_error_code = (
                None if error_code_raw is None else int(error_code_raw)
            )
            self._latest_gripper_status_code = (
                None if status_code_raw is None else int(status_code_raw)
            )
            status_text = field("status_text")
            self._latest_gripper_status_text = (
                None if status_text is None else str(status_text)
            )
            moving = field("is_moving")
            self._latest_gripper_is_moving = (
                None if moving is None else bool(moving)
            )
            self._latest_gripper_current = (
                None if current_raw is None else int(current_raw)
            )
            self._latest_gripper_temperature_c = (
                None if temperature_raw is None else int(temperature_raw)
            )
        if not payload:
            payload = {
                "timestamp": timestamp,
                "joint_pos": joint_value,
                "error_code": self._latest_gripper_error_code,
                "status_code": self._latest_gripper_status_code,
                "status_text": self._latest_gripper_status_text,
                "is_moving": self._latest_gripper_is_moving,
                "current": self._latest_gripper_current,
                "temperature_c": self._latest_gripper_temperature_c,
            }
        return payload

    def _extract_gripper_scalar(self, joint_pos: object) -> float:
        target = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if target.shape == (1,):
            return float(np.clip(target[0], 0.0, float(self.config.max_action)))
        if target.shape != (6,):
            raise ValueError(f"{self.name} expects scalar or q6 shape (6,), got {target.shape}.")
        return float(np.clip(target[int(self.config.command_dim)], 0.0, float(self.config.max_action)))

    @staticmethod
    def _q6_from_scalar(value: float, *, dim: int = 1, max_value: float = 1.0) -> np.ndarray:
        q6 = np.zeros(6, dtype=np.float32)
        q6[int(dim)] = float(np.clip(float(value), 0.0, min(float(max_value), 1.0)))
        return q6

    def _send_joint_pos(
        self,
        joint_pos: object,
        *,
        bypass_send_commands: bool,
        force: bool,
    ) -> bool:
        attempt_started_perf = time.perf_counter()
        attempt_started_wall = time.time()
        lock_wait_s: float | None = None
        target_scalar = self._extract_gripper_scalar(joint_pos)
        target_scalar = float(np.clip(target_scalar, 0.0, float(self.config.max_action)))
        target = self._q6_from_scalar(
            target_scalar,
            dim=int(self.config.command_dim),
            max_value=float(self.config.max_action),
        )

        def finish(status: str, *, sent: bool, skip_reason: str | None = None) -> bool:
            self._last_send_attempt_timestamp = attempt_started_wall
            self._last_send_elapsed_s = time.perf_counter() - attempt_started_perf
            self._last_send_lock_wait_s = lock_wait_s
            self._last_send_status = str(status)
            self._last_send_skip_reason = skip_reason
            self._last_send_target_scalar = float(target_scalar)
            return bool(sent)

        if not self.config.send_commands and not bool(bypass_send_commands):
            return finish("send_disabled", sent=False, skip_reason="send_disabled")
        if self._latest_gripper_error_code not in (None, 0):
            return finish(
                "hardware_fault",
                sent=False,
                skip_reason=f"error_code={int(self._latest_gripper_error_code)}",
            )
        if self._last_sent_command is not None and not bool(force):
            max_delta = float(np.max(np.abs(target - self._last_sent_command)))
            if max_delta < float(self.config.send_threshold):
                return finish("skipped_threshold", sent=False, skip_reason="threshold")
        command_time = attempt_started_wall
        if (
            self._last_sent_command_timestamp is not None
            and not bool(force)
            and command_time - float(self._last_sent_command_timestamp)
            < 1.0 / float(self.config.command_hz)
        ):
            return finish("rate_limited", sent=False, skip_reason="command_hz")
        if self.config.connect_hardware:
            lock_started_perf = time.perf_counter()
            acquired_lock = self._lock.acquire(blocking=False)
            lock_wait_s = time.perf_counter() - lock_started_perf
            if not acquired_lock:
                return finish("io_busy", sent=False, skip_reason="io_busy")
            try:
                if self.device is None or getattr(self.device, "gripper", None) is None:
                    return finish("not_connected", sent=False, skip_reason="not_connected")
                try:
                    self._move_gripper_locked(target_scalar)
                except Exception as exc:  # noqa: BLE001
                    self._quarantine_gripper_io_error_locked("send_failed", exc)
                    return finish(
                        "send_exception",
                        sent=False,
                        skip_reason=f"{type(exc).__name__}: {exc}",
                    )
            finally:
                self._lock.release()
        self._latest_joint_pos = target.copy()
        self._last_sent_command = target.copy()
        self._last_sent_command_timestamp = command_time
        self._last_sent_command_completed_timestamp = time.time()
        return finish("sent", sent=True)

    def _move_gripper_locked(self, target_scalar: float) -> None:
        if self.device is None:
            raise RuntimeError("device is not connected")
        speed, torque, continuous = self._move_parameters_for_current_config()
        self.device.move(
            float(np.clip(target_scalar, 0.0, float(self.config.max_action))),
            speed=speed,
            torque=torque,
            continuous=continuous,
            wait=False,
        )
        if speed is not None:
            self._applied_speed = float(speed)
        if torque is not None:
            self._applied_torque = float(torque)
        if continuous is not None:
            self._applied_continuous_grasp = bool(continuous)

    def _move_parameters_for_current_config(self) -> tuple[float | None, float | None, bool | None]:
        speed = float(self.config.speed)
        torque = float(self.config.torque)
        continuous = bool(self.config.continuous_grasp)
        speed_arg = (
            speed
            if self._applied_speed is None or abs(float(self._applied_speed) - speed) > 1e-6
            else None
        )
        torque_arg = (
            torque
            if self._applied_torque is None or abs(float(self._applied_torque) - torque) > 1e-6
            else None
        )
        continuous_arg = (
            continuous
            if self._applied_continuous_grasp is None
            or bool(self._applied_continuous_grasp) != continuous
            else None
        )
        return speed_arg, torque_arg, continuous_arg

    def _skip_gripper_telemetry_after_recent_command(self, now: float) -> bool:
        if self._gripper_telemetry_force_read_due(now):
            return False
        last_sent = self._last_sent_command_completed_timestamp
        if last_sent is None:
            return False
        grace_s = float(self.config.gripper_telemetry_command_grace_s)
        return bool(grace_s > 0.0 and float(now) - float(last_sent) < grace_s)

    def _gripper_telemetry_max_deferral_s(self) -> float:
        period_s = 1.0 / float(self.config.telemetry_hz)
        grace_s = float(self.config.gripper_telemetry_command_grace_s)
        # A continuously changing command stream must not starve measured
        # position feedback.  Permit one skipped telemetry period, then force
        # a serialized register read even if it lands inside the write guard.
        return max(2.0 * period_s, grace_s + period_s)

    def _gripper_telemetry_force_read_due(self, now: float) -> bool:
        measured_at = self._latest_joint_timestamp
        if measured_at is None:
            return True
        age_s = float(now) - float(measured_at)
        max_deferral_s = self._gripper_telemetry_max_deferral_s()
        return bool(age_s >= max_deferral_s or age_s < -max_deferral_s)

    def _apply_tactile_worker_payload(self, payload: dict[str, Any]) -> bool:
        payload_type = str(payload.get("type") or "")
        if payload_type == "ready":
            ready = payload.get("ready")
            if isinstance(ready, dict):
                with self._state_lock:
                    self._tactile_ready = {
                        str(key): bool(value)
                        for key, value in ready.items()
                    }
            return False
        if payload_type == "error":
            self._telemetry_error = f"tactile_worker_failed: {payload.get('error')}"
            with self._state_lock:
                self._latest_tactile = None
                self._latest_tactile_timestamp = None
                self._latest_tactile_dict = None
                self._latest_tactile_devices = []
            return False
        if payload_type != "frame":
            return False
        tactile_dict_raw = payload.get("tactile_dict")
        tactile_dict = (
            {
                str(name): np.asarray(value).copy()
                for name, value in tactile_dict_raw.items()
            }
            if isinstance(tactile_dict_raw, dict)
            else {}
        )
        with self._state_lock:
            self._latest_tactile = None
            self._latest_tactile_timestamp = (
                None
                if payload.get("timestamp") is None or not tactile_dict
                else float(payload["timestamp"])
            )
            self._latest_tactile_dict = tactile_dict if tactile_dict else None
            self._latest_tactile_devices = list(payload.get("devices") or tactile_dict)
            self._latest_tactile_frame_ids = (
                dict(payload.get("frame_ids") or {})
                if isinstance(payload.get("frame_ids"), dict)
                else {}
            )
            self._latest_tactile_shapes = (
                {
                    str(name): tuple(value)
                    for name, value in payload.get("shapes", {}).items()
                }
                if isinstance(payload.get("shapes"), dict)
                else {name: tuple(value.shape) for name, value in tactile_dict.items()}
            )
            self._latest_tactile_field_by_key = (
                {
                    str(name): str(value)
                    for name, value in payload.get("field_by_key", {}).items()
                }
                if isinstance(payload.get("field_by_key"), dict)
                else {}
            )
            self._latest_tactile_timings_ms = (
                {
                    str(name): dict(values)
                    for name, values in payload.get("timings_ms", {}).items()
                    if isinstance(values, dict)
                }
                if isinstance(payload.get("timings_ms"), dict)
                else {}
            )
            try:
                self._latest_tactile_version = int(
                    payload.get("version", self._latest_tactile_version)
                )
            except (TypeError, ValueError):
                self._latest_tactile_version += 1
            self._latest_tactile_read_elapsed_s = payload.get("read_elapsed_s")
            self._latest_tactile_gate_wait_s = payload.get("gate_wait_s")
            self._latest_tactile_read_skipped = False
        return True

    def _check_tactile_worker_process(self) -> None:
        process = self._tactile_worker_process
        if process is not None and not process.is_alive():
            exit_code = getattr(process, "exitcode", None)
            if process is not self._tactile_worker_observed_process:
                self._tactile_worker_observed_process = process
                self._tactile_worker_failure_count += 1
                self._tactile_worker_last_exit_code = exit_code
            if not self._closing:
                self._telemetry_error = f"tactile_worker_exited:{exit_code}"
                self._tactile_worker_restart_pending = bool(
                    self.config.tactile_worker_restart_on_error
                )
            with self._state_lock:
                self._latest_tactile = None
                self._latest_tactile_timestamp = None
                self._latest_tactile_dict = None
                self._latest_tactile_devices = []

    def _maybe_restart_tactile_worker(self) -> bool:
        if not self._tactile_worker_supervisor_lock.acquire(blocking=False):
            return False
        try:
            return self._maybe_restart_tactile_worker_locked()
        finally:
            self._tactile_worker_supervisor_lock.release()

    def _maybe_restart_tactile_worker_locked(self) -> bool:
        if (
            self._closing
            or not self.config.connect_tactile
            or not self.config.tactile_process_read
            or not self.config.tactile_worker_restart_on_error
        ):
            return False
        self._check_tactile_worker_process()
        process = self._tactile_worker_process
        if process is not None and process.is_alive():
            return False
        if not self._tactile_worker_restart_pending:
            return False
        exponent = max(0, min(8, self._tactile_worker_failure_count - 1))
        backoff = min(
            float(self.config.tactile_worker_restart_max_backoff_sec),
            float(self.config.tactile_worker_restart_initial_backoff_sec) * (2**exponent),
        )
        now = time.time()
        if (
            self._tactile_worker_last_restart_attempt_at is not None
            and now - self._tactile_worker_last_restart_attempt_at < backoff
        ):
            return False
        self._tactile_worker_last_restart_attempt_at = now
        receiver_stop = self._tactile_worker_receiver_stop
        if receiver_stop is not None:
            receiver_stop.set()
        receiver = self._tactile_worker_receiver_thread
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=0.5)
        if process is not None:
            process.join(timeout=0.1)
        queue_out = self._tactile_worker_queue
        if queue_out is not None:
            try:
                queue_out.close()
                queue_out.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
        self._tactile_worker_process = None
        self._tactile_worker_queue = None
        self._tactile_worker_stop_event = None
        self._tactile_worker_receiver_stop = None
        self._tactile_worker_receiver_thread = None
        try:
            self._start_tactile_worker()
        except Exception as exc:  # noqa: BLE001
            self._tactile_worker_restart_pending = True
            self._telemetry_error = f"tactile_worker_restart_failed:{type(exc).__name__}:{exc}"
            return False
        self._tactile_worker_restart_count += 1
        self._telemetry_error = None
        return True

    def _drain_tactile_worker_queue(self) -> bool:
        if self._tactile_worker_queue is None:
            return False
        updated = False
        while True:
            try:
                payload = self._tactile_worker_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(payload, dict):
                updated = self._apply_tactile_worker_payload(payload) or updated
        if not updated:
            with self._state_lock:
                self._latest_tactile_read_elapsed_s = None
                self._latest_tactile_gate_wait_s = None
                self._latest_tactile_read_skipped = True
        self._check_tactile_worker_process()
        return updated

    def _telemetry_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                self._read_telemetry_once()
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    return
                self._telemetry_error = f"telemetry_failed: {exc}"
            remaining = (1.0 / float(self.config.telemetry_hz)) - (time.time() - loop_start)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def _read_telemetry_once(self) -> None:
        self._maybe_restart_tactile_worker()
        device = self.device
        if device is None:
            if self.config.connect_hardware:
                with self._lock:
                    self._handle_gripper_io_error_locked(
                        "telemetry_no_connection",
                        RuntimeError("device is not connected"),
                    )
            elif self.config.connect_tactile:
                self._telemetry_error = "device is not connected"
            return

        gripper_state = None
        gripper_fault_unrecovered = False
        tactile_frames = {}
        read_started_at = time.time()
        telemetry_started_at = time.perf_counter()
        gripper_read_elapsed_s = None
        gripper_read_skipped = False
        gripper_read_skip_reason = None
        gripper_read_forced = False
        tactile_read_elapsed_s = None
        tactile_gate_wait_s = None
        tactile_read_attempted = False
        tactile_read_skipped = False
        tactile_worker_active = bool(
            self.config.connect_tactile and self.config.tactile_process_read
        )
        if self.config.connect_hardware and device.gripper is None:
            with self._lock:
                if self.device is device and getattr(device, "gripper", None) is None:
                    self._handle_gripper_io_error_locked(
                        "telemetry_no_gripper",
                        RuntimeError("gripper is not connected"),
                    )
            device = self.device
            if device is None or getattr(device, "gripper", None) is None:
                return
        if self.config.connect_hardware and device.gripper is not None:
            gripper_read_forced = self._gripper_telemetry_force_read_due(read_started_at)
            if self._skip_gripper_telemetry_after_recent_command(read_started_at):
                gripper_read_skipped = True
                gripper_read_skip_reason = "recent_command"
            else:
                acquired_gripper_lock = self._lock.acquire(blocking=gripper_read_forced)
                if not acquired_gripper_lock:
                    gripper_read_skipped = True
                    gripper_read_skip_reason = "io_busy"
                else:
                    try:
                        if self.device is not device:
                            return
                        gripper_read_started_at = time.perf_counter()
                        try:
                            gripper_state = device.read_gripper_state()
                        except Exception as exc:  # noqa: BLE001
                            gripper_read_skipped = True
                            gripper_read_skip_reason = "read_failed"
                            recovered = self._handle_gripper_io_error_locked(
                                "telemetry_failed",
                                exc,
                            )
                            gripper_fault_unrecovered = not recovered
                        gripper_read_elapsed_s = time.perf_counter() - gripper_read_started_at
                    finally:
                        self._lock.release()

        if tactile_worker_active and not self._tactile_worker_receiver_active():
            self._drain_tactile_worker_queue()
        elif self.config.connect_tactile and device.tactile is not None:
            tactile_read_hz = float(self.config.tactile_read_hz)
            tactile_period_s = 0.0 if tactile_read_hz <= 0.0 else 1.0 / tactile_read_hz
            now_perf = time.perf_counter()
            next_tactile_read_at = self._next_tactile_read_at
            should_read_tactile = (
                tactile_period_s <= 0.0
                or next_tactile_read_at is None
                or now_perf >= float(next_tactile_read_at)
            )
            if not should_read_tactile:
                tactile_read_skipped = True
            else:
                tactile_read_attempted = True
        if tactile_read_attempted:
            tactile_read_started_at = time.perf_counter()
            gate_acquired = False
            try:
                if bool(self.config.tactile_global_read_lock):
                    gate_wait_started_at = time.perf_counter()
                    DMTAC_TACTILE_GLOBAL_READ_GATE.acquire()
                    gate_acquired = True
                    tactile_gate_wait_s = time.perf_counter() - gate_wait_started_at
                with self._tactile_lock:
                    if self.device is not device:
                        return
                    tactile_frames = device.read_tactile(
                        timeout_ms=int(self.config.tactile_timeout_ms),
                        wait=bool(self.config.tactile_wait_for_new),
                        require_ready=bool(self.config.tactile_require_ready),
                    )
            finally:
                if gate_acquired:
                    DMTAC_TACTILE_GLOBAL_READ_GATE.release()
            self._last_tactile_read_finished_at = time.perf_counter()
            tactile_read_elapsed_s = self._last_tactile_read_finished_at - tactile_read_started_at
            if self.config.tactile_read_hz > 0.0:
                period_s = 1.0 / float(self.config.tactile_read_hz)
                next_at = tactile_read_started_at + period_s
                if next_at < self._last_tactile_read_finished_at - period_s:
                    next_at = self._last_tactile_read_finished_at
                self._next_tactile_read_at = next_at

        if gripper_state is not None:
            self._apply_gripper_state(
                gripper_state,
                fallback_timestamp=read_started_at,
            )

        if tactile_read_attempted:
            tactile_vec, tactile_dict, frame_ids, field_by_key = self._frames_to_tactile(
                tactile_frames,
                build_vector=bool(self.config.tactile_store_flat_vector),
                fields=self.config.tactile_fields,
            )
            with self._state_lock:
                self._latest_tactile = tactile_vec
                self._latest_tactile_timestamp = read_started_at if tactile_dict else None
                self._latest_tactile_dict = tactile_dict if tactile_dict else None
                self._latest_tactile_devices = list(tactile_dict)
                self._latest_tactile_frame_ids = frame_ids
                self._latest_tactile_shapes = {
                    name: tuple(value.shape)
                    for name, value in tactile_dict.items()
                }
                self._latest_tactile_field_by_key = field_by_key
                self._latest_tactile_timings_ms = self._frame_timings_ms(tactile_frames)
                if tactile_dict:
                    self._latest_tactile_version += 1
        self._latest_telemetry_timestamp = time.time()
        self._latest_gripper_read_elapsed_s = gripper_read_elapsed_s
        self._latest_gripper_read_skipped = bool(gripper_read_skipped)
        self._latest_gripper_read_skip_reason = gripper_read_skip_reason
        self._latest_gripper_read_forced = bool(gripper_read_forced)
        if not tactile_worker_active:
            with self._state_lock:
                self._latest_tactile_read_elapsed_s = tactile_read_elapsed_s
                self._latest_tactile_gate_wait_s = tactile_gate_wait_s
                self._latest_tactile_read_skipped = bool(tactile_read_skipped)
        self._latest_telemetry_elapsed_s = time.perf_counter() - telemetry_started_at
        if not gripper_fault_unrecovered and not (
            self._telemetry_error
            and str(self._telemetry_error).startswith("tactile_worker")
        ):
            self._telemetry_error = None

    @staticmethod
    def _frames_to_tactile(
        frames: dict[Any, Any],
        *,
        build_vector: bool,
        fields: Any = DMTAC_DEFAULT_TACTILE_FIELDS,
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray], dict[str, int | None], dict[str, str]]:
        tactile_dict, frame_ids, _shapes, _timings, field_by_key = _frames_payload(
            frames,
            fields=fields,
        )
        vectors: list[np.ndarray] = []
        for key in sorted(tactile_dict):
            if build_vector:
                vectors.append(np.asarray(tactile_dict[key], dtype=np.float32).reshape(-1))
        if not vectors:
            return None, tactile_dict, frame_ids, field_by_key
        return (
            np.concatenate(vectors).astype(np.float32, copy=False),
            tactile_dict,
            frame_ids,
            field_by_key,
        )

    @staticmethod
    def _frame_timings_ms(frames: dict[Any, Any]) -> dict[str, dict[str, float]]:
        timings_by_device: dict[str, dict[str, float]] = {}
        for dev_id, frame in frames.items():
            if frame is None:
                continue
            timings = getattr(frame, "timings_ms", None)
            if not isinstance(timings, dict):
                continue
            device_timings: dict[str, float] = {}
            for name, value in timings.items():
                try:
                    device_timings[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
            if device_timings:
                timings_by_device[str(dev_id)] = device_timings
        return timings_by_device

    @staticmethod
    def _tactile_dict_to_vector(tactile_dict: dict[str, np.ndarray] | None) -> np.ndarray | None:
        if not tactile_dict:
            return None
        vectors = [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in tactile_dict.values()
            if value is not None
        ]
        if not vectors:
            return None
        return np.concatenate(vectors).astype(np.float32, copy=False)

    def get_state(
        self,
        *,
        include_tactile_arrays: bool = True,
        include_tactile_vector: bool = True,
        copy_tactile_arrays: bool = True,
    ) -> dict[str, object]:
        if (
            self.config.connect_tactile
            and self.config.tactile_process_read
            and not self._tactile_worker_receiver_active()
        ):
            self._drain_tactile_worker_queue()
        gripper_state = None
        if self.device is not None and getattr(self.device, "gripper", None) is not None:
            gripper_state = getattr(self.device.gripper, "last_state", None)
        if self.config.disconnected_state_scalar is not None:
            self._latest_joint_timestamp = time.time()
        joint_pos = self._latest_joint_pos
        measured_joint_pos = self._latest_measured_joint_pos
        commanded_joint_pos = self._last_sent_command
        scalar_index = int(self.config.command_dim)
        commanded_scalar = (
            None
            if commanded_joint_pos is None
            else float(commanded_joint_pos[scalar_index])
        )
        measured_scalar = (
            None
            if measured_joint_pos is None
            else float(measured_joint_pos[scalar_index])
        )
        with self._state_lock:
            latest_tactile = self._latest_tactile
            latest_tactile_dict = self._latest_tactile_dict
            latest_tactile_timestamp = self._latest_tactile_timestamp
            latest_telemetry_timestamp = self._latest_telemetry_timestamp
            latest_tactile_devices = tuple(self._latest_tactile_devices)
            latest_tactile_frame_ids = dict(self._latest_tactile_frame_ids)
            latest_tactile_shapes = dict(self._latest_tactile_shapes)
            latest_tactile_field_by_key = dict(self._latest_tactile_field_by_key)
            latest_tactile_timings_ms = {
                name: dict(values)
                for name, values in self._latest_tactile_timings_ms.items()
            }
            latest_tactile_version = int(self._latest_tactile_version)
            latest_tactile_read_elapsed_s = self._latest_tactile_read_elapsed_s
            latest_tactile_gate_wait_s = self._latest_tactile_gate_wait_s
            latest_tactile_read_skipped = bool(self._latest_tactile_read_skipped)
            tactile_ready = dict(self._tactile_ready)
            telemetry_error = self._telemetry_error
            tactile = None
            tactile_dict = None
            if include_tactile_arrays:
                if include_tactile_vector:
                    if latest_tactile is not None:
                        tactile = latest_tactile.copy() if copy_tactile_arrays else latest_tactile
                    elif bool(self.config.tactile_store_flat_vector):
                        tactile = self._tactile_dict_to_vector(latest_tactile_dict)
                tactile_dict = (
                    None
                    if latest_tactile_dict is None
                    else (
                        {name: value.copy() for name, value in latest_tactile_dict.items()}
                        if copy_tactile_arrays
                        else dict(latest_tactile_dict)
                    )
                )
        return {
            "last_command": None if self._last_command is None else self._last_command.copy(),
            "last_sent_command": None
            if commanded_joint_pos is None
            else commanded_joint_pos.copy(),
            "last_sent_command_timestamp": self._last_sent_command_timestamp,
            "last_sent_command_completed_timestamp": (
                self._last_sent_command_completed_timestamp
            ),
            "commanded_joint_pos": None
            if commanded_joint_pos is None
            else commanded_joint_pos.copy(),
            "joint_pos": None if joint_pos is None else joint_pos.copy(),
            "measured_joint_pos": None if measured_joint_pos is None else measured_joint_pos.copy(),
            "joint_timestamp": self._latest_joint_timestamp,
            "gripper_measurement_timestamp": (
                None
                if self.config.disconnected_state_scalar is not None
                else self._latest_joint_timestamp
            ),
            "gripper_state_source": (
                "configured_disconnected_assumption"
                if self.config.disconnected_state_scalar is not None
                else "runtime_backend"
            ),
            "gripper_state_assumed": bool(
                self.config.disconnected_state_scalar is not None
            ),
            "gripper_commanded_scalar": commanded_scalar,
            "gripper_measured_scalar": measured_scalar,
            "gripper_error_code": self._latest_gripper_error_code,
            "gripper_status_code": self._latest_gripper_status_code,
            "gripper_status_text": self._latest_gripper_status_text,
            "gripper_is_moving": self._latest_gripper_is_moving,
            "gripper_current": self._latest_gripper_current,
            "gripper_temperature_c": self._latest_gripper_temperature_c,
            "gripper_state_scalar": None if joint_pos is None else float(joint_pos[scalar_index]),
            "tactile": tactile,
            "tactile_timestamp": latest_tactile_timestamp,
            "tactile_dict": tactile_dict,
            "telemetry_timestamp": latest_telemetry_timestamp,
            "tactile_regions": latest_tactile_devices,
            "tactile_devices": latest_tactile_devices,
            "tactile_frame_ids": latest_tactile_frame_ids,
            "tactile_shapes": latest_tactile_shapes,
            "tactile_field_by_key": latest_tactile_field_by_key,
            "tactile_timings_ms": latest_tactile_timings_ms,
            "tactile_version": latest_tactile_version,
            "tactile_raw_shape_per_sensor": _dmtac_tactile_shape_per_sensor(
                latest_tactile_shapes,
                self.config.tactile_fields,
            ),
            "gripper_read_elapsed_s": self._latest_gripper_read_elapsed_s,
            "gripper_read_skipped": bool(self._latest_gripper_read_skipped),
            "gripper_read_skip_reason": self._latest_gripper_read_skip_reason,
            "gripper_read_forced": bool(self._latest_gripper_read_forced),
            "gripper_telemetry_max_deferral_s": (
                self._gripper_telemetry_max_deferral_s()
            ),
            "gripper_send_attempt_timestamp": self._last_send_attempt_timestamp,
            "gripper_send_elapsed_s": self._last_send_elapsed_s,
            "gripper_send_lock_wait_s": self._last_send_lock_wait_s,
            "gripper_last_send_status": self._last_send_status,
            "gripper_last_send_skip_reason": self._last_send_skip_reason,
            "gripper_last_send_target_scalar": self._last_send_target_scalar,
            "tactile_read_elapsed_s": latest_tactile_read_elapsed_s,
            "tactile_gate_wait_s": latest_tactile_gate_wait_s,
            "telemetry_elapsed_s": self._latest_telemetry_elapsed_s,
            "tactile_wait_for_new": bool(self.config.tactile_wait_for_new),
            "tactile_read_hz": float(self.config.tactile_read_hz),
            "tactile_read_skipped": latest_tactile_read_skipped,
            "tactile_process_read": bool(self.config.tactile_process_read),
            "tactile_process_nice": int(self.config.tactile_process_nice),
            "tactile_process_cpu_affinity": list(self.config.tactile_process_cpu_affinity),
            "tactile_process_queue_size": int(self.config.tactile_process_queue_size),
            "tactile_worker_alive": bool(
                self._tactile_worker_process is not None
                and self._tactile_worker_process.is_alive()
            ),
            "tactile_worker_receiver_alive": self._tactile_worker_receiver_active(),
            "tactile_worker_restart_on_error": bool(
                self.config.tactile_worker_restart_on_error
            ),
            "tactile_worker_restart_pending": bool(self._tactile_worker_restart_pending),
            "tactile_worker_restart_count": int(self._tactile_worker_restart_count),
            "tactile_worker_failure_count": int(self._tactile_worker_failure_count),
            "tactile_worker_last_exit_code": self._tactile_worker_last_exit_code,
            "tactile_worker_last_restart_attempt_timestamp": (
                self._tactile_worker_last_restart_attempt_at
            ),
            "tactile_parallel_read": bool(self.config.tactile_parallel_read),
            "tactile_global_read_lock": bool(self.config.tactile_global_read_lock),
            "tactile_fields": tuple(self.config.tactile_fields),
            "tactile_compute_profile": str(self.config.tactile_compute_profile),
            "tactile_calibration_policy": str(self.config.tactile_calibration_policy),
            "tactile_calibration_dir": (
                None
                if self.config.tactile_calibration_dir is None
                else str(self.config.tactile_calibration_dir)
            ),
            "tactile_solving_enabled": "distributed_force" in self.config.tactile_fields,
            "tactile_max_fps": int(self.config.tactile_max_fps),
            "tactile_reset_on_start": bool(self.config.tactile_reset_on_start),
            "tactile_warmup_frames": int(self.config.tactile_warmup_frames),
            "tactile_denoise": bool(self.config.tactile_denoise),
            "tactile_store_flat_vector": bool(self.config.tactile_store_flat_vector),
            "telemetry_error": telemetry_error,
            "connect_hardware": bool(self.config.connect_hardware),
            "connect_tactile": bool(self.config.connect_tactile),
            "observation_only": bool(self.config.observation_only),
            "send_commands": bool(self.config.send_commands),
            "intentionally_disconnected": bool(
                self.config.disconnected_state_scalar is not None
            ),
            "is_connected": self.device is not None,
            "gripper_is_connected": self.device is not None
            and getattr(self.device, "gripper", None) is not None,
            "reconnect_count": int(self._reconnect_count),
            "gripper_reconnect_on_error": bool(self.config.reconnect_on_error),
            "gripper_reconnect_cooldown_sec": float(self.config.reconnect_cooldown_sec),
            "gripper_last_reconnect_attempt_timestamp": self._last_reconnect_attempt_at,
            "gripper_last_reconnect_error": self._last_reconnect_error,
            "backend_type": "dmtac_gripper",
            "gripper_port": str(self.config.gripper_port),
            "gripper_id": int(self.config.gripper_id),
            "gripper_command_dim": scalar_index,
            "gripper_speed": float(self.config.speed),
            "gripper_torque": float(self.config.torque),
            "gripper_command_hz": float(self.config.command_hz),
            "gripper_max_action": float(self.config.max_action),
            "gripper_continuous": bool(self.config.continuous_grasp),
            "tactile_ready": tactile_ready,
            "raw_gripper_state": None
            if gripper_state is None
            else gripper_state.to_dict(),
        }

    def close(self) -> None:
        self._closing = True
        deferred_interrupt: BaseException | None = None
        if self._stop_event is not None:
            self._stop_event.set()
        if self._telemetry_thread is not None:
            try:
                self._telemetry_thread.join(timeout=DMTAC_TELEMETRY_STOP_TIMEOUT_S)
            except BaseException as exc:  # noqa: BLE001 - continue bounded child cleanup.
                deferred_interrupt = exc
            if self._telemetry_thread.is_alive():
                self._telemetry_error = "telemetry_stop_timeout: closing device while telemetry thread exits"
            else:
                self._telemetry_thread = None
        if self._tactile_worker_stop_event is not None:
            self._tactile_worker_stop_event.set()
        if self._tactile_worker_receiver_stop is not None:
            self._tactile_worker_receiver_stop.set()
        receiver = self._tactile_worker_receiver_thread
        if receiver is not None:
            try:
                receiver.join(timeout=1.0)
            except BaseException as exc:  # noqa: BLE001 - finish bounded process cleanup.
                deferred_interrupt = exc
            if receiver.is_alive():
                self._telemetry_error = (
                    "tactile_receiver_stop_timeout: closing tactile worker while receiver exits"
                )
            else:
                self._tactile_worker_receiver_thread = None
        process = self._tactile_worker_process
        if process is not None:
            try:
                process.join(timeout=DMTAC_TACTILE_PROCESS_STOP_TIMEOUT_S)
            except BaseException as exc:  # noqa: BLE001 - still terminate the child below.
                if deferred_interrupt is None:
                    deferred_interrupt = exc
            try:
                process_alive = bool(process.is_alive())
            except BaseException as exc:  # noqa: BLE001
                process_alive = True
                if deferred_interrupt is None:
                    deferred_interrupt = exc
            if process_alive:
                try:
                    process.terminate()
                except BaseException as exc:  # noqa: BLE001
                    if deferred_interrupt is None:
                        deferred_interrupt = exc
                try:
                    process.join(timeout=1.0)
                except BaseException as exc:  # noqa: BLE001
                    if deferred_interrupt is None:
                        deferred_interrupt = exc
            self._tactile_worker_process = None
        worker_queue = self._tactile_worker_queue
        if worker_queue is not None:
            try:
                worker_queue.close()
                worker_queue.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
        self._tactile_worker_queue = None
        self._tactile_worker_stop_event = None
        self._tactile_worker_receiver_stop = None
        device = self.device
        if device is not None:
            acquired_tactile_lock = self._tactile_lock.acquire(timeout=DMTAC_TACTILE_CLOSE_GRACE_S)
            if not acquired_tactile_lock:
                self._telemetry_error = "close_timeout: tactile read did not finish before device close"
            try:
                with self._lock:
                    device = self.device
                    self.device = None
                    self._invalidate_gripper_connection_state_locked()
                if device is not None:
                    device.close()
            finally:
                if acquired_tactile_lock:
                    self._tactile_lock.release()
        self._stop_event = None
        if deferred_interrupt is not None:
            raise deferred_interrupt
