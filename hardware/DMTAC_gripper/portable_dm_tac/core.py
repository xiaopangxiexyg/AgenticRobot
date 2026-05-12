from __future__ import annotations

import ctypes.util
import sys
import time
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
    patch_libudev_for_conda()
    try:
        import dmrobotics  # type: ignore
    except ImportError as exc:
        if not add_local_sdk_path():
            raise RuntimeError(
                "Cannot import dmrobotics. Install dmrobotics in the active Python "
                "environment, or keep portable_DMTAC_gripper next to a local dmrobotics directory."
            ) from exc
        try:
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
    patch_libudev_for_conda()
    try:
        from dmrobotics.src.CameraDeviceManager import CameraDeviceManager  # type: ignore
    except ImportError as exc:
        _import_dmrobotics()
        from dmrobotics.src.CameraDeviceManager import CameraDeviceManager  # type: ignore  # noqa: F401

    mgr = CameraDeviceManager()
    devices = mgr.find_devices()
    if devices is None:
        return []
    if isinstance(devices, list):
        return devices
    return list(devices)


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
            dev_id=self.dev_id,
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

    @property
    def dev_id(self) -> DeviceId:
        return self.config.dev_id

    def connect(self) -> "DMTacDevice":
        dmrobotics = _import_dmrobotics()
        self.sensor = dmrobotics.Sensor(self.config.to_sensor_options())
        return self

    def close(self) -> None:
        if self.sensor is not None:
            self.sensor.disconnect()
            time.sleep(0.05)
            self.sensor = None

    def __enter__(self) -> "DMTacDevice":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ensure_connected(self) -> None:
        if self.sensor is None:
            self.connect()

    def status(self) -> int:
        self.ensure_connected()
        return int(self.sensor.getDevStatus())

    def wait_until_ready(self, timeout_s: float = 10.0, poll_s: float = 0.02) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.status() == 0:
                return True
            time.sleep(poll_s)
        return self.status() == 0

    def reset(self) -> None:
        self.ensure_connected()
        self.sensor.reset()
        self._last_fid = -1

    def set_max_fps(self, max_fps: int) -> Any:
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
        self.ensure_connected()
        status = int(self.sensor.getDevStatus())
        if require_ready and status != 0:
            return None

        if wait and status == 0:
            if not self.sensor.wait_for_new(self._last_fid, timeout_ms=timeout_ms):
                return None

        frame = TacFrame(device_id=self.dev_id, timestamp_ns=time.time_ns(), status=status)

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
        try:
            fid, data = _normalize_fid_data(fid_key, getter())
            if to_array and data is not None:
                data = np.asarray(data, dtype=np.float32)
            setattr(frame, attr, data)
            frame.fids[fid_key] = fid
        except Exception as exc:
            if strict:
                raise
            frame.errors[fid_key] = repr(exc)

    def _read_plain(self, frame: TacFrame, attr: str, getter, strict: bool, cast_float: bool = False) -> None:
        try:
            value = getter()
            if cast_float and value is not None:
                value = float(value)
            setattr(frame, attr, value)
        except Exception as exc:
            if strict:
                raise
            frame.errors[attr] = repr(exc)

    def _read_events(self, frame: TacFrame, strict: bool) -> None:
        try:
            events, last_seq = self.sensor.getEvents()
            frame.events = list(events) if events is not None else []
            frame.last_event_seq = int(last_seq)
        except Exception as exc:
            if strict:
                raise
            frame.errors["events"] = repr(exc)

    @staticmethod
    def _choose_frame_fid(fids: Mapping[str, int]) -> Optional[int]:
        for key in ("raw", "infer", "deformation2d", "shear", "depth", "force", "distributed_force"):
            if key in fids:
                return int(fids[key])
        return None


class DMTacMultiDevice:
    """Sequential multi-device wrapper. Supports two or more devices."""

    def __init__(self, configs: Sequence[DeviceConfig]):
        if not configs:
            raise ValueError("configs must contain at least one DeviceConfig")
        self.devices = [DMTacDevice(config) for config in configs]

    @classmethod
    def from_device_ids(cls, device_ids: Sequence[DeviceId], **config_kwargs: Any) -> "DMTacMultiDevice":
        return cls([DeviceConfig(dev_id=dev_id, **config_kwargs) for dev_id in device_ids])

    def connect(self) -> "DMTacMultiDevice":
        for device in self.devices:
            device.connect()
        return self

    def close(self) -> None:
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
    ) -> Dict[DeviceId, Optional[TacFrame]]:
        return {
            device.dev_id: device.read(
                timeout_ms=timeout_ms,
                wait=wait,
                require_ready=require_ready,
                strict=strict,
            )
            for device in self.devices
        }

    def reset_all(self) -> None:
        for device in self.devices:
            device.reset()


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
