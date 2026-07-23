from __future__ import annotations

import contextlib
import ctypes.util
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


DeviceId = Union[int, str]


@dataclass(frozen=True)
class DataSpec:
    key: str
    name: str
    getter: str
    return_format: str
    shape_dtype: str
    meaning: str


DATA_SPECS: Tuple[DataSpec, ...] = (
    DataSpec("raw", "raw image", "getRawImg()", "(fid, DMTacImage)", "img: (480, 640), uint8", "camera raw grayscale tactile image"),
    DataSpec("infer", "infer image", "getInferImg()", "(fid, DMTacImage)", "img: (288, 384), uint8", "SDK preprocessed/inference image"),
    DataSpec("deformation2d", "2D deformation", "getDeformation2D()", "(fid, np.ndarray)", "(288, 384, 2), float32", "dense pixel displacement field (dx, dy)"),
    DataSpec("shear", "shear field", "getShear()", "(fid, np.ndarray)", "(288, 384, 2), float32", "local tangential/shear field"),
    DataSpec("depth", "depth map", "getDepth()", "(fid, np.ndarray)", "(288, 384), float32", "normal indentation/depth estimate"),
    DataSpec("force", "6D wrench", "getForce()", "(fid, np.ndarray)", "(6,), float32", "Fx, Fy, Fz, Mx, My, Mz"),
    DataSpec("distributed_force", "distributed force", "getDistributeForce()", "(fid, np.ndarray)", "(288, 384, 3), float32", "local 3D force distribution per pixel"),
    DataSpec("contact_area", "contact area", "getContactArea()", "float", "scalar float", "SDK aggregated contact area"),
    DataSpec("base_frame", "base frame", "getBaseFrame()", "DMTacImage", "depends on raw/base state", "no-contact reference frame for reset/setBaseFrame"),
    DataSpec("events", "events", "getEvents()", "(events, last_seq)", "events: list[dict], last_seq: int", "SDK info/warning/error events"),
)


_LIBUDEV_PATCHED = False
_THREAD_EXCEPTHOOK_LOCK = threading.Lock()
_THREAD_EXCEPTHOOK_INSTALLED = False
_THREAD_EXCEPTHOOK_PREVIOUS = None
_SDK_SHUTDOWN_SUPPRESS_UNTIL = 0.0
_SDK_SHUTDOWN_SUPPRESS_SECONDS = 2.0
_SDK_DISCONNECT_SETTLE_S = 0.15
_SDK_OUTPUT_SUPPRESS_ENV = "DMTAC_SUPPRESS_SDK_OUTPUT"
_SDK_PREFLIGHT_VIDEO_ENV = "DMTAC_PREFLIGHT_VIDEO_OPEN"
_SDK_RESOLVE_SERIAL_ENV = "DMTAC_RESOLVE_SERIAL_TO_INDEX"
_SDK_OUTPUT_LOCK = threading.RLock()
_SDK_IMPORT_PATH_LOCK = threading.RLock()
_SDK_NOISE_CONFIGURED = False
_SDK_DISCOVERY_CACHE: tuple[float, List[Any]] | None = None
_SDK_DISCOVERY_CACHE_TTL_S = 2.0


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _configure_sdk_noise_filters() -> None:
    global _SDK_NOISE_CONFIGURED
    if _SDK_NOISE_CONFIGURED:
        return
    logging.getLogger("dmrobotics").setLevel(logging.WARNING)
    if _env_flag(_SDK_OUTPUT_SUPPRESS_ENV, True):
        try:
            import cv2  # type: ignore  # noqa: PLC0415

            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
        except Exception:
            pass
    _SDK_NOISE_CONFIGURED = True


@contextlib.contextmanager
def _suppress_noisy_sdk_output():
    if not _env_flag(_SDK_OUTPUT_SUPPRESS_ENV, True):
        yield
        return
    with _SDK_OUTPUT_LOCK:
        stdout_fd = None
        stderr_fd = None
        devnull = None
        stdout_redirected = False
        stderr_redirected = False
        previous_logging_disable = logging.root.manager.disable
        try:
            logging.disable(logging.INFO)
            sys.stdout.flush()
            sys.stderr.flush()
            stdout_fd = os.dup(1)
            stderr_fd = os.dup(2)
            devnull = open(os.devnull, "w")
            os.dup2(devnull.fileno(), 1)
            stdout_redirected = True
            os.dup2(devnull.fileno(), 2)
            stderr_redirected = True
        except OSError:
            pass
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            logging.disable(previous_logging_disable)
            if stdout_redirected and stdout_fd is not None:
                try:
                    os.dup2(stdout_fd, 1)
                finally:
                    os.close(stdout_fd)
            elif stdout_fd is not None:
                os.close(stdout_fd)
            if stderr_redirected and stderr_fd is not None:
                try:
                    os.dup2(stderr_fd, 2)
                finally:
                    os.close(stderr_fd)
            elif stderr_fd is not None:
                os.close(stderr_fd)
            if devnull is not None:
                devnull.close()


def _suppress_dmrobotics_shutdown_thread_errors_for(seconds: float = _SDK_SHUTDOWN_SUPPRESS_SECONDS) -> None:
    global _SDK_SHUTDOWN_SUPPRESS_UNTIL
    with _THREAD_EXCEPTHOOK_LOCK:
        _SDK_SHUTDOWN_SUPPRESS_UNTIL = max(
            _SDK_SHUTDOWN_SUPPRESS_UNTIL,
            time.monotonic() + max(0.0, float(seconds)),
        )


def _traceback_mentions_dmrobotics_update_image(tb: Any) -> bool:
    while tb is not None:
        frame = tb.tb_frame
        module_name = str(frame.f_globals.get("__name__", ""))
        filename = str(frame.f_code.co_filename)
        function_name = str(frame.f_code.co_name)
        if (
            "dmrobotics.src.dmSDK" in module_name
            or "dmrobotics.src.dmSDK" in filename
        ) and "update_image" in function_name:
            return True
        tb = tb.tb_next
    return False


def _should_suppress_dmrobotics_thread_error(args: threading.ExceptHookArgs) -> bool:
    with _THREAD_EXCEPTHOOK_LOCK:
        in_shutdown_window = time.monotonic() <= _SDK_SHUTDOWN_SUPPRESS_UNTIL
    if not in_shutdown_window:
        return False
    exc_text = f"{getattr(args, 'exc_type', '')}: {getattr(args, 'exc_value', '')}"
    if "cvarrToMat" not in exc_text or "Unknown array type" not in exc_text:
        return False
    return _traceback_mentions_dmrobotics_update_image(args.exc_traceback)


def _dmrobotics_thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if _should_suppress_dmrobotics_thread_error(args):
        logging.getLogger(__name__).debug(
            "Suppressed DM-Tac SDK image-thread shutdown exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        return
    previous = _THREAD_EXCEPTHOOK_PREVIOUS or threading.__excepthook__
    previous(args)


def _install_dmrobotics_thread_excepthook() -> None:
    global _THREAD_EXCEPTHOOK_INSTALLED, _THREAD_EXCEPTHOOK_PREVIOUS
    with _THREAD_EXCEPTHOOK_LOCK:
        if _THREAD_EXCEPTHOOK_INSTALLED:
            return
        _THREAD_EXCEPTHOOK_PREVIOUS = threading.excepthook
        threading.excepthook = _dmrobotics_thread_excepthook
        _THREAD_EXCEPTHOOK_INSTALLED = True


def _is_dmrobotics_package_path(value: object) -> bool:
    """Return whether *value* exposes the SDK package as top-level modules."""

    if not isinstance(value, (str, os.PathLike)) or not str(value):
        return False
    try:
        path = Path(value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        path.name == "dmrobotics"
        and (path / "__init__.py").is_file()
        and (path / "utils.py").is_file()
    )


@contextlib.contextmanager
def _isolate_dmrobotics_import_path():
    """Contain the vendor SDK's top-level ``sys.path`` side effect.

    dmrobotics 1.2.10 inserts its package directory itself at ``sys.path[0]``.
    A later multiprocessing ``spawn`` then resolves the repository's ``utils``
    package as ``dmrobotics/utils.py``.  Keep the SDK root fallback available,
    but never leave the package directory exposed as a top-level import root.
    """

    with _SDK_IMPORT_PATH_LOCK:
        try:
            yield
        finally:
            sys.path[:] = [
                value for value in sys.path if not _is_dmrobotics_package_path(value)
            ]


def add_local_sdk_path() -> bool:
    """Fallback to the sibling dmrobotics source tree if the package is not installed."""
    package_root = Path(__file__).resolve().parents[1]
    sdk_root = package_root.parent
    text = str(sdk_root)
    if not (sdk_root / "dmrobotics").is_dir():
        return False
    if text not in sys.path:
        sys.path.insert(0, text)
    return True


def patch_libudev_for_conda(system_libudev: str = "/lib/x86_64-linux-gnu/libudev.so.1") -> None:
    """Patch ctypes.util.find_library for conda envs that resolve udev to a directory."""
    global _LIBUDEV_PATCHED
    if _LIBUDEV_PATCHED:
        return

    real_find_library = ctypes.util.find_library

    def fixed_find_library(name: str) -> Optional[str]:
        found = real_find_library(name)
        if name == "udev" and found and found.endswith("/lib/udev"):
            return system_libudev
        return found

    ctypes.util.find_library = fixed_find_library
    _LIBUDEV_PATCHED = True


def _import_dmrobotics():
    _install_dmrobotics_thread_excepthook()
    _configure_sdk_noise_filters()
    patch_libudev_for_conda()
    with _isolate_dmrobotics_import_path():
        try:
            with _suppress_noisy_sdk_output():
                import dmrobotics  # type: ignore
        except ImportError as exc:
            if not add_local_sdk_path():
                raise RuntimeError(
                    "Cannot import dmrobotics. Install dmrobotics in the active Python "
                    "environment, or keep portable_DMTAC_gripper next to a local dmrobotics directory."
                ) from exc
            try:
                with _suppress_noisy_sdk_output():
                    import dmrobotics  # type: ignore
            except ImportError as fallback_exc:
                raise RuntimeError(
                    "Cannot import dmrobotics after checking the installed package and "
                    "the sibling local dmrobotics directory."
                ) from fallback_exc
    return dmrobotics


def _mode_value(mode: str):
    dmrobotics = _import_dmrobotics()
    value = str(mode).strip().lower()
    if value == "high":
        return dmrobotics.Mode.HIGH
    if value == "standard":
        return dmrobotics.Mode.STANDARD
    raise ValueError("mode must be 'standard' or 'high'")


def discover_devices() -> List[Any]:
    """Return the SDK's detected compliant device records."""
    _configure_sdk_noise_filters()
    patch_libudev_for_conda()
    with _isolate_dmrobotics_import_path():
        try:
            with _suppress_noisy_sdk_output():
                from dmrobotics.src.CameraDeviceManager import CameraDeviceManager  # type: ignore
        except ImportError:
            _import_dmrobotics()
            with _suppress_noisy_sdk_output():
                from dmrobotics.src.CameraDeviceManager import CameraDeviceManager  # type: ignore  # noqa: F401

        with _suppress_noisy_sdk_output():
            mgr = CameraDeviceManager()
            devices = mgr.find_devices()
    if devices is None:
        return []
    if isinstance(devices, list):
        return devices
    return list(devices)


def _discover_devices_cached(max_age_s: float = _SDK_DISCOVERY_CACHE_TTL_S) -> List[Any]:
    global _SDK_DISCOVERY_CACHE
    now = time.monotonic()
    cached = _SDK_DISCOVERY_CACHE
    if cached is not None:
        cached_at, devices = cached
        if now - cached_at <= float(max_age_s):
            return list(devices)
    devices = discover_devices()
    _SDK_DISCOVERY_CACHE = (now, list(devices))
    return devices


def _resolve_device_id_for_sensor(dev_id: DeviceId) -> DeviceId:
    if not _env_flag(_SDK_RESOLVE_SERIAL_ENV, True):
        return dev_id
    if not isinstance(dev_id, str):
        return dev_id
    target = dev_id.strip()
    if not target:
        return dev_id
    for item in _discover_devices_cached():
        if not isinstance(item, Mapping):
            continue
        serial = item.get("serial")
        index = item.get("device_index")
        if str(serial or "") != target or index is None:
            continue
        try:
            return int(index)
        except (TypeError, ValueError):
            return dev_id
    return dev_id


def _video_device_occupants(node: str) -> list[str]:
    occupants: list[str] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return occupants
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return occupants
    for entry in entries:
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        matched = False
        for fd_path in fds:
            try:
                if os.readlink(fd_path) == node:
                    matched = True
                    break
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
        if not matched:
            continue
        comm = ""
        cmdline = ""
        try:
            comm = (entry / "comm").read_text(errors="ignore").strip()
        except OSError:
            pass
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
            cmdline = raw_cmdline.replace(b"\0", b" ").decode(errors="ignore").strip()
        except OSError:
            pass
        label = f"pid={entry.name}"
        if comm:
            label += f":{comm}"
        if cmdline:
            label += f" cmd={cmdline[:160]}"
        occupants.append(label)
    return occupants


def _preflight_video_device_open(dev_id: DeviceId) -> tuple[bool, str]:
    if not _env_flag(_SDK_PREFLIGHT_VIDEO_ENV, True):
        return True, "disabled"
    if not sys.platform.startswith("linux") or not isinstance(dev_id, int):
        return True, "not_applicable"
    node = f"/dev/video{dev_id}"
    if not os.path.exists(node):
        return False, f"{node} does not exist"
    try:
        import cv2  # type: ignore  # noqa: PLC0415
    except Exception as exc:
        return True, f"cv2_unavailable:{exc}"
    cap = None
    try:
        with _suppress_noisy_sdk_output():
            cap = cv2.VideoCapture(int(dev_id), cv2.CAP_V4L2)
            opened = bool(cap.isOpened())
        if not opened:
            occupants = _video_device_occupants(node)
            suffix = f"; occupants={occupants}" if occupants else ""
            return False, (
                f"OpenCV/V4L2 cannot open {node}; device may be occupied "
                f"or the SDK index may point to the wrong video node{suffix}"
            )
        return True, f"{node} opened"
    finally:
        if cap is not None:
            cap.release()


def discover_device_candidates() -> List[Dict[str, Any]]:
    """Return N160MU2 video nodes seen by udev, including unusable nodes."""
    if sys.platform.startswith("linux"):
        try:
            import pyudev  # type: ignore
        except ImportError:
            return [{"error": "pyudev is not installed"}]
        out: List[Dict[str, Any]] = []
        ctx = pyudev.Context()
        for dev in ctx.list_devices(subsystem="video4linux"):
            node = dev.device_node or ""
            if not node.startswith("/dev/video"):
                continue
            model = dev.get("ID_MODEL") or ""
            product = dev.get("ID_V4L_PRODUCT") or ""
            text = f"{model} {product}".upper()
            if "N160MU2" not in text and "GENERIC_N160MU2" not in text:
                continue
            serial = dev.get("ID_SERIAL_SHORT") or dev.get("ID_SERIAL") or ""
            exists = os.path.exists(node)
            readable = os.access(node, os.R_OK) if exists else False
            writable = os.access(node, os.W_OK) if exists else False
            reason = "ok"
            if not exists:
                reason = "missing_device_node"
            elif not readable:
                reason = "not_readable"
            out.append(
                {
                    "device_node": node,
                    "serial": serial,
                    "model": model,
                    "product": product,
                    "exists": exists,
                    "readable": readable,
                    "writable": writable,
                    "reason": reason,
                }
            )
        return out
    return []


def discover_device_ids(prefer_serial: bool = True) -> List[DeviceId]:
    """Return device ids usable in DeviceConfig.dev_id."""
    ids: List[DeviceId] = []
    for item in discover_devices():
        if isinstance(item, Mapping):
            serial = item.get("serial")
            index = item.get("device_index")
            if prefer_serial and serial:
                ids.append(str(serial))
            elif index is not None:
                ids.append(int(index))
            elif serial:
                ids.append(str(serial))
            else:
                ids.append(str(item))
        else:
            ids.append(str(item))
    return ids


@dataclass
class DeviceConfig:
    dev_id: DeviceId
    backend: str = "cpu"
    mode: str = "standard"
    max_fps: int = 120
    show_fps: bool = False
    cuda_id: int = 0
    remote_addr: str = "192.168.127.10:50051"
    pc_host: str = "0.0.0.0"
    pc_port: int = 60001
    enable_raw: bool = True
    enable_deformation: bool = True
    enable_shear: bool = True
    enable_depth: bool = True
    enable_force: bool = True
    include_infer: bool = True
    include_distributed_force: bool = True
    include_contact_area: bool = True
    include_base_frame: bool = True
    include_events: bool = True
    read_raw: Optional[bool] = None
    read_deformation: Optional[bool] = None
    read_shear: Optional[bool] = None
    read_depth: Optional[bool] = None
    read_force: Optional[bool] = None

    def to_sensor_options(self):
        dmrobotics = _import_dmrobotics()
        return dmrobotics.SensorOptions(
            dev_id=_resolve_device_id_for_sensor(self.dev_id),
            backend=self.backend,
            mode=_mode_value(self.mode),
            max_fps=self.max_fps,
            show_fps=self.show_fps,
            cuda_id=self.cuda_id,
            enable_raw=self.enable_raw,
            enable_deformation=self.enable_deformation,
            enable_shear=self.enable_shear,
            enable_depth=self.enable_depth,
            enable_force=self.enable_force,
            remote_addr=self.remote_addr,
            pc_host=self.pc_host,
            pc_port=self.pc_port,
        )

    @staticmethod
    def _read_flag(enable_flag: bool, read_flag: Optional[bool]) -> bool:
        return enable_flag if read_flag is None else bool(read_flag)

    def should_read_raw(self) -> bool:
        return self._read_flag(self.enable_raw, self.read_raw)

    def should_read_deformation(self) -> bool:
        return self._read_flag(self.enable_deformation, self.read_deformation)

    def should_read_shear(self) -> bool:
        return self._read_flag(self.enable_shear, self.read_shear)

    def should_read_depth(self) -> bool:
        return self._read_flag(self.enable_depth, self.read_depth)

    def should_read_force(self) -> bool:
        return self._read_flag(self.enable_force, self.read_force)


@dataclass
class TacFrame:
    device_id: DeviceId
    timestamp_ns: int
    status: int
    fid: Optional[int] = None
    raw: Any = None
    infer: Any = None
    deformation2d: Optional[np.ndarray] = None
    shear: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None
    distributed_force: Optional[np.ndarray] = None
    contact_area: Optional[float] = None
    base_frame: Any = None
    events: Optional[List[Dict[str, Any]]] = None
    last_event_seq: Optional[int] = None
    fids: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "device_id": self.device_id,
            "timestamp_ns": self.timestamp_ns,
            "status": self.status,
            "fid": self.fid,
            "fids": dict(self.fids),
        }
        for key in (
            "raw",
            "infer",
            "deformation2d",
            "shear",
            "depth",
            "force",
            "distributed_force",
            "base_frame",
        ):
            out[key] = describe_value(getattr(self, key))
        out["contact_area"] = self.contact_area
        out["events"] = {
            "count": len(self.events) if self.events is not None else None,
            "last_event_seq": self.last_event_seq,
        }
        if self.errors:
            out["errors"] = dict(self.errors)
        if self.timings_ms:
            out["timings_ms"] = dict(self.timings_ms)
        return out


def describe_value(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None

    if hasattr(value, "img"):
        img = getattr(value, "img", None)
        serial = getattr(value, "serial", None)
        if img is None:
            return {"type": type(value).__name__, "serial": serial, "img": None}
        arr = np.asarray(img)
        return {
            "type": type(value).__name__,
            "serial": serial,
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "min": _safe_min(arr),
            "max": _safe_max(arr),
        }

    arr = np.asarray(value)
    return {
        "type": type(value).__name__,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": _safe_min(arr),
        "max": _safe_max(arr),
    }


def _safe_min(arr: np.ndarray) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.nanmin(arr))


def _safe_max(arr: np.ndarray) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.nanmax(arr))


def _normalize_fid_data(name: str, result: Any) -> Tuple[int, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(f"{name} returned {type(result)}, expected (fid, data)")
    fid, data = result
    return int(fid), data


class DMTacDevice:
    """Single DM-Tac sensor wrapper."""

    def __init__(self, config: DeviceConfig):
        self.config = config
        self.sensor: Any = None
        self._last_fid = -1
        self._io_lock = threading.RLock()

    @property
    def dev_id(self) -> DeviceId:
        return self.config.dev_id

    def connect(self) -> "DMTacDevice":
        with self._io_lock:
            dmrobotics = _import_dmrobotics()
            sensor_dev_id = _resolve_device_id_for_sensor(self.config.dev_id)
            preflight_ok, preflight_detail = _preflight_video_device_open(sensor_dev_id)
            if not preflight_ok:
                raise RuntimeError(
                    "DMTac video preflight failed "
                    f"for dev_id={self.config.dev_id!r} resolved_dev_id={sensor_dev_id!r}: "
                    f"{preflight_detail}. Set {_SDK_PREFLIGHT_VIDEO_ENV}=0 to bypass."
                )
            with _suppress_noisy_sdk_output():
                self.sensor = dmrobotics.Sensor(self.config.to_sensor_options())
        return self

    def close(self) -> None:
        with self._io_lock:
            sensor = self.sensor
            if sensor is None:
                return
            self.sensor = None
            _suppress_dmrobotics_shutdown_thread_errors_for()
            try:
                sensor.disconnect()
            finally:
                time.sleep(_SDK_DISCONNECT_SETTLE_S)

    def __enter__(self) -> "DMTacDevice":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ensure_connected(self) -> None:
        if self.sensor is None:
            self.connect()

    def status(self) -> int:
        with self._io_lock:
            self.ensure_connected()
            with _suppress_noisy_sdk_output():
                return int(self.sensor.getDevStatus())

    def wait_until_ready(self, timeout_s: float = 10.0, poll_s: float = 0.2) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.status() == 0:
                return True
            time.sleep(poll_s)
        return self.status() == 0

    def reset(self) -> None:
        with self._io_lock:
            self.ensure_connected()
            self.sensor.reset()
            self._last_fid = -1

    def set_base_frame(self, raw: Any) -> None:
        frame = np.asarray(raw)
        if frame.shape != (480, 640) or frame.dtype != np.uint8:
            raise ValueError(
                f"DM-Tac base frame must be uint8 (480, 640), got {frame.dtype} {frame.shape}"
            )
        with self._io_lock:
            self.ensure_connected()
            from dmrobotics.src.dmSDK import DMTacImage  # type: ignore  # noqa: PLC0415

            image = DMTacImage(img=np.ascontiguousarray(frame), serial=str(self.dev_id))
            self.sensor.setBaseFrame(image)

    def get_base_frame_array(self) -> np.ndarray:
        with self._io_lock:
            self.ensure_connected()
            image = self.sensor.getBaseFrame()
            frame = np.asarray(getattr(image, "img", None))
            if frame.shape != (480, 640) or frame.dtype != np.uint8:
                raise ValueError(
                    f"SDK returned invalid base frame for {self.dev_id!r}: {frame.dtype} {frame.shape}"
                )
            return np.ascontiguousarray(frame)

    def set_max_fps(self, max_fps: int) -> Any:
        with self._io_lock:
            self.ensure_connected()
            self.config.max_fps = int(max_fps)
            return self.sensor.setMaxFPS(int(max_fps))

    def read(
        self,
        timeout_ms: int = 500,
        wait: bool = True,
        require_ready: bool = True,
        strict: bool = False,
    ) -> Optional[TacFrame]:
        """Read one frame worth of enabled data.

        Returns None if the device is not ready or no new frame arrives before
        timeout_ms. Set strict=True to raise getter errors instead of storing
        them in TacFrame.errors.
        """
        with self._io_lock:
            self.ensure_connected()
            status_start = time.perf_counter()
            with _suppress_noisy_sdk_output():
                status = int(self.sensor.getDevStatus())
            timings_ms = {"getDevStatus": (time.perf_counter() - status_start) * 1000.0}
            if require_ready and status != 0:
                return None

            if wait and status == 0:
                wait_start = time.perf_counter()
                with _suppress_noisy_sdk_output():
                    ok = self.sensor.wait_for_new(self._last_fid, timeout_ms=timeout_ms)
                timings_ms["wait_for_new"] = (time.perf_counter() - wait_start) * 1000.0
                if not ok:
                    return None

            frame = TacFrame(device_id=self.dev_id, timestamp_ns=time.time_ns(), status=status)
            frame.timings_ms.update(timings_ms)

            self._read_fid_getter(frame, "raw", "raw", self.sensor.getRawImg, self.config.should_read_raw(), strict)
            self._read_fid_getter(frame, "infer", "infer", self.sensor.getInferImg, self.config.include_infer, strict)
            self._read_fid_getter(frame, "deformation2d", "deformation2d", self.sensor.getDeformation2D, self.config.should_read_deformation(), strict)
            self._read_fid_getter(frame, "shear", "shear", self.sensor.getShear, self.config.should_read_shear(), strict)
            self._read_fid_getter(frame, "depth", "depth", self.sensor.getDepth, self.config.should_read_depth(), strict)
            self._read_fid_getter(frame, "force", "force", self.sensor.getForce, self.config.should_read_force(), strict, to_array=True)
            self._read_fid_getter(
                frame,
                "distributed_force",
                "distributed_force",
                self.sensor.getDistributeForce,
                self.config.include_distributed_force,
                strict,
                to_array=True,
            )

            if self.config.include_contact_area:
                self._read_plain(frame, "contact_area", self.sensor.getContactArea, strict, cast_float=True)
            if self.config.include_base_frame:
                self._read_plain(frame, "base_frame", self.sensor.getBaseFrame, strict)
            if self.config.include_events:
                self._read_events(frame, strict)

            frame.fid = self._choose_frame_fid(frame.fids)
            if frame.fid is not None:
                self._last_fid = frame.fid
            return frame

    def _read_fid_getter(
        self,
        frame: TacFrame,
        attr: str,
        fid_key: str,
        getter,
        enabled: bool,
        strict: bool,
        to_array: bool = False,
    ) -> None:
        if not enabled:
            return
        start = time.perf_counter()
        try:
            with _suppress_noisy_sdk_output():
                fid, data = _normalize_fid_data(fid_key, getter())
            if to_array and data is not None:
                data = np.asarray(data, dtype=np.float32)
            setattr(frame, attr, data)
            frame.fids[fid_key] = fid
        except Exception as exc:
            if strict:
                raise
            frame.errors[fid_key] = repr(exc)
        finally:
            frame.timings_ms[attr] = (time.perf_counter() - start) * 1000.0

    def _read_plain(self, frame: TacFrame, attr: str, getter, strict: bool, cast_float: bool = False) -> None:
        start = time.perf_counter()
        try:
            with _suppress_noisy_sdk_output():
                value = getter()
            if cast_float and value is not None:
                value = float(value)
            setattr(frame, attr, value)
        except Exception as exc:
            if strict:
                raise
            frame.errors[attr] = repr(exc)
        finally:
            frame.timings_ms[attr] = (time.perf_counter() - start) * 1000.0

    def _read_events(self, frame: TacFrame, strict: bool) -> None:
        start = time.perf_counter()
        try:
            with _suppress_noisy_sdk_output():
                events, last_seq = self.sensor.getEvents()
            frame.events = list(events) if events is not None else []
            frame.last_event_seq = int(last_seq)
        except Exception as exc:
            if strict:
                raise
            frame.errors["events"] = repr(exc)
        finally:
            frame.timings_ms["events"] = (time.perf_counter() - start) * 1000.0

    @staticmethod
    def _choose_frame_fid(fids: Mapping[str, int]) -> Optional[int]:
        for key in ("raw", "infer", "deformation2d", "shear", "depth", "force", "distributed_force"):
            if key in fids:
                return int(fids[key])
        return None


class DMTacMultiDevice:
    """Multi-device wrapper. Supports sequential or parallel reads."""

    def __init__(self, configs: Sequence[DeviceConfig]):
        if not configs:
            raise ValueError("configs must contain at least one DeviceConfig")
        self.devices = [DMTacDevice(config) for config in configs]
        self._executor: Optional[ThreadPoolExecutor] = None
        self._io_lock = threading.RLock()

    @classmethod
    def from_device_ids(cls, device_ids: Sequence[DeviceId], **config_kwargs: Any) -> "DMTacMultiDevice":
        return cls([DeviceConfig(dev_id=dev_id, **config_kwargs) for dev_id in device_ids])

    def connect(self) -> "DMTacMultiDevice":
        with self._io_lock:
            for device in self.devices:
                device.connect()
        return self

    def close(self) -> None:
        with self._io_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._executor = None
            for device in reversed(self.devices):
                device.close()

    def __enter__(self) -> "DMTacMultiDevice":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def wait_until_ready(self, timeout_s: float = 10.0) -> Dict[DeviceId, bool]:
        return {device.dev_id: device.wait_until_ready(timeout_s=timeout_s) for device in self.devices}

    def read_all(
        self,
        timeout_ms: int = 500,
        wait: bool = True,
        require_ready: bool = True,
        strict: bool = False,
        parallel: bool = False,
    ) -> Dict[DeviceId, Optional[TacFrame]]:
        with self._io_lock:
            if bool(parallel) and len(self.devices) > 1:
                return self._read_all_parallel(
                    timeout_ms=timeout_ms,
                    wait=wait,
                    require_ready=require_ready,
                    strict=strict,
                )
            return {
                device.dev_id: device.read(
                    timeout_ms=timeout_ms,
                    wait=wait,
                    require_ready=require_ready,
                    strict=strict,
                )
                for device in self.devices
            }

    def _read_all_parallel(
        self,
        *,
        timeout_ms: int,
        wait: bool,
        require_ready: bool,
        strict: bool,
    ) -> Dict[DeviceId, Optional[TacFrame]]:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.devices)),
                thread_name_prefix="dmtac_read",
            )
        futures = {
            self._executor.submit(
                device.read,
                timeout_ms=timeout_ms,
                wait=wait,
                require_ready=require_ready,
                strict=strict,
            ): device.dev_id
            for device in self.devices
        }
        results: Dict[DeviceId, Optional[TacFrame]] = {}
        for future in as_completed(futures):
            dev_id = futures[future]
            results[dev_id] = future.result()
        return {device.dev_id: results.get(device.dev_id) for device in self.devices}

    def reset_all(self) -> None:
        with self._io_lock:
            for device in self.devices:
                device.reset()

    def set_base_frames(self, frames: Mapping[DeviceId, Any]) -> None:
        with self._io_lock:
            for device in self.devices:
                if device.dev_id in frames:
                    device.set_base_frame(frames[device.dev_id])

    def get_base_frames(self) -> Dict[DeviceId, np.ndarray]:
        with self._io_lock:
            return {device.dev_id: device.get_base_frame_array() for device in self.devices}


def format_frame_summary(frame: Optional[TacFrame]) -> str:
    if frame is None:
        return "None"
    parts = [f"dev={frame.device_id}", f"status={frame.status}", f"fid={frame.fid}"]
    for key in ("raw", "infer", "deformation2d", "shear", "depth", "force", "distributed_force"):
        desc = describe_value(getattr(frame, key))
        if desc is None:
            continue
        if desc.get("img") is None and "shape" not in desc:
            parts.append(f"{key}=None")
            continue
        shape = desc.get("shape")
        dtype = desc.get("dtype")
        parts.append(f"{key}={shape}/{dtype}")
    if frame.contact_area is not None:
        parts.append(f"contact_area={frame.contact_area:.6g}")
    if frame.events is not None:
        parts.append(f"events={len(frame.events)}")
    if frame.errors:
        parts.append(f"errors={list(frame.errors)}")
    return " | ".join(parts)


def close_all(devices: Iterable[DMTacDevice]) -> None:
    for device in devices:
        device.close()
