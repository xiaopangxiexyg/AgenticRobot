from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from portable_dm_tac import (
    DMTacMultiDevice,
    DeviceConfig,
    TacFrame,
    discover_device_ids,
    format_frame_summary,
)

from .gripper_driver import GripperState, InspireGripper
from .tactile_filter import DistributedForceDenoiser, TactileDenoiseConfig


DeviceId = Union[int, str]

TACTILE_FIELDS: Tuple[str, ...] = (
    "raw",
    "infer",
    "deformation2d",
    "shear",
    "depth",
    "force",
    "distributed_force",
    "contact_area",
    "base_frame",
    "events",
)

DEFAULT_TACTILE_FIELDS: Tuple[str, ...] = ("distributed_force",)


@dataclass
class GripperConfig:
    port: str = "/dev/ttyUSB0"
    gripper_id: int = 1
    baudrate: int = 115200
    timeout: float = 0.03
    max_open_mm: float = 70.0
    debug: bool = False


@dataclass
class TactileConfig:
    dev_ids: Optional[Sequence[DeviceId]] = None
    backend: str = "cpu"
    mode: str = "standard"
    max_fps: int = 120
    fields: Sequence[str] = DEFAULT_TACTILE_FIELDS
    show_fps: bool = False
    cuda_id: int = 0
    remote_addr: str = "192.168.127.10:50051"
    pc_host: str = "0.0.0.0"
    pc_port: int = 60001
    denoise_distributed_force: bool = True
    denoise_config: TactileDenoiseConfig = field(default_factory=TactileDenoiseConfig)


@dataclass
class DeviceSnapshot:
    timestamp: float
    gripper: Optional[GripperState]
    tactile: Dict[DeviceId, Optional[TacFrame]]

    def summary(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gripper": None if self.gripper is None else self.gripper.to_dict(),
            "tactile": {
                dev_id: None if frame is None else frame.summary()
                for dev_id, frame in self.tactile.items()
            },
        }


def normalize_tactile_fields(fields: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if fields is None:
        return DEFAULT_TACTILE_FIELDS

    normalized = []
    for item in fields:
        key = str(item).strip().lower()
        if key == "all":
            return TACTILE_FIELDS
        if key not in TACTILE_FIELDS:
            raise ValueError(f"Unsupported tactile field '{item}'. Valid fields: {TACTILE_FIELDS}")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized) if normalized else DEFAULT_TACTILE_FIELDS


def tactile_device_config(dev_id: DeviceId, cfg: TactileConfig) -> DeviceConfig:
    fields = set(normalize_tactile_fields(cfg.fields))
    need_distributed_force = "distributed_force" in fields
    return DeviceConfig(
        dev_id=dev_id,
        backend=cfg.backend,
        mode=cfg.mode,
        max_fps=cfg.max_fps,
        show_fps=cfg.show_fps,
        cuda_id=cfg.cuda_id,
        remote_addr=cfg.remote_addr,
        pc_host=cfg.pc_host,
        pc_port=cfg.pc_port,
        enable_raw=("raw" in fields),
        enable_deformation=("deformation2d" in fields) or need_distributed_force,
        enable_shear=("shear" in fields) or need_distributed_force,
        enable_depth=("depth" in fields) or need_distributed_force,
        enable_force=("force" in fields) or need_distributed_force,
        include_infer=("infer" in fields),
        include_distributed_force=("distributed_force" in fields),
        include_contact_area=("contact_area" in fields),
        include_base_frame=("base_frame" in fields),
        include_events=("events" in fields),
        read_raw=("raw" in fields),
        read_deformation=("deformation2d" in fields),
        read_shear=("shear" in fields),
        read_depth=("depth" in fields),
        read_force=("force" in fields),
    )


class DMTacGripperDevice:
    """Unified gripper + dual DM-Tac tactile device wrapper."""

    def __init__(
        self,
        gripper: Optional[GripperConfig] = None,
        tactile: Optional[TactileConfig] = None,
        *,
        connect_gripper: bool = True,
        connect_tactile: bool = True,
    ) -> None:
        self.gripper_config = gripper or GripperConfig()
        self.tactile_config = tactile or TactileConfig()
        self.connect_gripper = bool(connect_gripper)
        self.connect_tactile = bool(connect_tactile)
        self.gripper: Optional[InspireGripper] = None
        self.tactile: Optional[DMTacMultiDevice] = None
        self._distributed_force_denoiser = DistributedForceDenoiser(self.tactile_config.denoise_config)

    def connect(self) -> "DMTacGripperDevice":
        if self.connect_gripper and self.gripper is None:
            cfg = self.gripper_config
            self.gripper = InspireGripper(
                port=cfg.port,
                gripper_id=cfg.gripper_id,
                baudrate=cfg.baudrate,
                timeout=cfg.timeout,
                max_open_mm=cfg.max_open_mm,
                debug=cfg.debug,
            )

        if self.connect_tactile and self.tactile is None:
            dev_ids = list(self.tactile_config.dev_ids or discover_device_ids()[:2])
            if not dev_ids:
                raise RuntimeError(
                    "No DM-Tac tactile device was discovered. Check USB with `lsusb`, "
                    "or pass explicit ids with TactileConfig(dev_ids=[...])."
                )
            configs = [tactile_device_config(dev_id, self.tactile_config) for dev_id in dev_ids]
            self.tactile = DMTacMultiDevice(configs).connect()
            self.reset_tactile_filter()

        return self

    def close(self) -> None:
        if self.tactile is not None:
            self.tactile.close()
            self.tactile = None
        if self.gripper is not None:
            self.gripper.close()
            self.gripper = None

    def __enter__(self) -> "DMTacGripperDevice":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def require_gripper(self) -> InspireGripper:
        if self.gripper is None:
            raise RuntimeError("Gripper is not connected")
        return self.gripper

    def require_tactile(self) -> DMTacMultiDevice:
        if self.tactile is None:
            raise RuntimeError("Tactile sensors are not connected")
        return self.tactile

    def set_speed(self, speed: float) -> None:
        self.require_gripper().set_speed(speed)

    def get_speed(self):
        return self.require_gripper().get_speed()

    def set_torque(self, torque: float) -> None:
        self.require_gripper().set_torque(torque)

    def get_torque(self):
        return self.require_gripper().get_torque()

    def clear_fault(self) -> None:
        self.require_gripper().clear_fault()

    def stop(self) -> None:
        self.require_gripper().estop()

    def open(self, *, wait: bool = False, wait_timeout: float = 3.0) -> Optional[GripperState]:
        gripper = self.require_gripper()
        gripper.open_hand()
        return gripper.wait_until_idle(wait_timeout) if wait else None

    def close_gripper(
        self,
        *,
        speed: Optional[float] = None,
        torque: Optional[float] = None,
        continuous: bool = False,
        wait: bool = False,
        wait_timeout: float = 3.0,
    ) -> Optional[GripperState]:
        gripper = self.require_gripper()
        gripper.close_hand(speed=speed, torque=torque, continuous=continuous)
        return gripper.wait_until_idle(wait_timeout) if wait else None

    def move(
        self,
        joint_pos: float,
        *,
        speed: Optional[float] = None,
        torque: Optional[float] = None,
        continuous: Optional[bool] = None,
        wait: bool = False,
        wait_timeout: float = 3.0,
    ) -> Optional[GripperState]:
        gripper = self.require_gripper()
        gripper.set_joint_pos(joint_pos, speed=speed, torque=torque, continuous=continuous)
        return gripper.wait_until_idle(wait_timeout) if wait else None

    def read_gripper_state(self) -> GripperState:
        return self.require_gripper().get_state()

    def wait_tactile_ready(self, timeout_s: float = 10.0) -> Dict[DeviceId, bool]:
        return self.require_tactile().wait_until_ready(timeout_s=timeout_s)

    def reset_tactile(self) -> None:
        self.require_tactile().reset_all()
        self.reset_tactile_filter()

    def reset_tactile_filter(self) -> None:
        self._distributed_force_denoiser.reset()

    def warmup_tactile_filter(self, *, frames: int = 30, timeout_ms: int = 100) -> None:
        original_denoise = self.tactile_config.denoise_distributed_force
        self.tactile_config.denoise_distributed_force = False
        try:
            for _ in range(max(0, int(frames))):
                self.read_tactile(timeout_ms=timeout_ms)
        finally:
            self.tactile_config.denoise_distributed_force = original_denoise

        self.reset_tactile_filter()
        if not original_denoise:
            return

        fields = set(normalize_tactile_fields(self.tactile_config.fields))
        if "distributed_force" not in fields:
            return

        cfg = self.tactile_config.denoise_config
        if not cfg.enabled:
            return

        mode = str(cfg.mode).strip().lower()
        if mode == "delta":
            prefill_frames = max(1, int(cfg.baseline_frames))
        elif mode == "frame_delta":
            prefill_frames = 1
        else:
            prefill_frames = 0

        for _ in range(prefill_frames):
            self.read_tactile(timeout_ms=timeout_ms)

    def read_tactile(
        self,
        *,
        timeout_ms: int = 100,
        wait: bool = True,
        require_ready: bool = True,
    ) -> Dict[DeviceId, Optional[TacFrame]]:
        frames = self.require_tactile().read_all(
            timeout_ms=timeout_ms,
            wait=wait,
            require_ready=require_ready,
        )
        return self._denoise_tactile_frames(frames)

    def _denoise_tactile_frames(
        self,
        frames: Dict[DeviceId, Optional[TacFrame]],
    ) -> Dict[DeviceId, Optional[TacFrame]]:
        if not self.tactile_config.denoise_distributed_force:
            return frames
        fields = set(normalize_tactile_fields(self.tactile_config.fields))
        if "distributed_force" not in fields:
            return frames

        for dev_id, frame in frames.items():
            if frame is None or frame.distributed_force is None:
                continue
            frame.distributed_force = self._distributed_force_denoiser.transform(
                dev_id,
                frame.distributed_force,
            )
        return frames

    def read(
        self,
        *,
        read_gripper: bool = True,
        read_tactile: bool = True,
        tactile_timeout_ms: int = 100,
        tactile_wait: bool = True,
        tactile_require_ready: bool = True,
    ) -> DeviceSnapshot:
        gripper_state = self.read_gripper_state() if read_gripper and self.gripper is not None else None
        tactile_frames = (
            self.read_tactile(
                timeout_ms=tactile_timeout_ms,
                wait=tactile_wait,
                require_ready=tactile_require_ready,
            )
            if read_tactile and self.tactile is not None
            else {}
        )
        return DeviceSnapshot(
            timestamp=time.time(),
            gripper=gripper_state,
            tactile=tactile_frames,
        )


def format_snapshot_summary(snapshot: DeviceSnapshot) -> str:
    parts = [f"t={snapshot.timestamp:.6f}"]
    if snapshot.gripper is not None:
        g = snapshot.gripper
        parts.append(
            f"gripper joint={g.joint_pos:.4f} open_mm={g.opening_mm:.2f} "
            f"moving={g.is_moving} err=0x{g.error_code:04X} status={g.status_code}"
        )
    for dev_id, frame in snapshot.tactile.items():
        parts.append(f"tactile[{dev_id}] {format_frame_summary(frame)}")
    return " | ".join(parts)
