"""Validated configuration for the dual-wrist UMI collector."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pose_adapter import as_rigid_transform


DEFAULT_LEFT_WRIST_CAMERA_SERIAL = "261122273924"
DEFAULT_RIGHT_WRIST_CAMERA_SERIAL = "260522278055"
SIDES = ("left", "right")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _transform(value: Any, *, name: str) -> tuple[tuple[float, ...], ...]:
    matrix = as_rigid_transform(value, name=name)
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class UmiSideConfig:
    side: str
    tracker_name: str
    tracker_role: str
    tracker_channel: str
    camera_serial: str
    gripper_port: str
    tactile_ids: tuple[str, ...] = ()
    T_tcp_to_tracker: tuple[tuple[float, ...], ...] = field(
        default_factory=lambda: tuple(tuple(float(v) for v in row) for row in np.eye(4))
    )
    tcp_extrinsic_status: str = "identity_placeholder_unmeasured"

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        for name in ("tracker_name", "tracker_role", "tracker_channel", "camera_serial"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{self.side}.{name} must be non-empty")
        object.__setattr__(
            self,
            "T_tcp_to_tracker",
            _transform(self.T_tcp_to_tracker, name=f"{self.side}.T_tcp_to_tracker"),
        )
        object.__setattr__(self, "tactile_ids", tuple(str(v) for v in self.tactile_ids))

    @property
    def tcp_transform(self) -> np.ndarray:
        return np.asarray(self.T_tcp_to_tracker, dtype=float).copy()

    @property
    def uses_identity_tcp_placeholder(self) -> bool:
        return bool(
            np.allclose(self.tcp_transform, np.eye(4), atol=1e-9)
            and self.tcp_extrinsic_status == "identity_placeholder_unmeasured"
        )

    @classmethod
    def from_dict(cls, side: str, payload: Mapping[str, Any]) -> "UmiSideConfig":
        default_camera = (
            DEFAULT_LEFT_WRIST_CAMERA_SERIAL
            if side == "left"
            else DEFAULT_RIGHT_WRIST_CAMERA_SERIAL
        )
        default_tracker_index = 1 if side == "left" else 2
        return cls(
            side=side,
            tracker_name=str(payload.get("tracker_name", f"tracker_{default_tracker_index}")),
            tracker_role=str(payload.get("tracker_role", f"{side}_wrist")),
            tracker_channel=str(payload.get("tracker_channel", f"{side}_wrist_pose")),
            camera_serial=str(payload.get("camera_serial", default_camera)),
            gripper_port=str(payload.get("gripper_port", "")),
            tactile_ids=tuple(payload.get("tactile_ids", ())),
            T_tcp_to_tracker=payload.get("T_tcp_to_tracker", np.eye(4).tolist()),
            tcp_extrinsic_status=str(
                payload.get("tcp_extrinsic_status", "identity_placeholder_unmeasured")
            ),
        )


@dataclass(frozen=True)
class UmiTimingConfig:
    control_hz: float = 30.0
    sample_hz: float = 15.0
    vive_max_age_sec: float = 0.10
    camera_max_age_sec: float = 0.10
    angle_trigger_max_age_sec: float = 0.20
    gripper_max_age_sec: float = 0.50
    tactile_max_age_sec: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "control_hz",
            "sample_hz",
            "vive_max_age_sec",
            "camera_max_age_sec",
            "angle_trigger_max_age_sec",
            "gripper_max_age_sec",
            "tactile_max_age_sec",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"timing.{name} must be positive")
        if self.sample_hz > self.control_hz:
            raise ValueError("timing.sample_hz cannot exceed control_hz")


@dataclass(frozen=True)
class UmiCollectionConfig:
    dataset_root: Path
    calibration_result: Path
    policy_io_layout: Path
    left: UmiSideConfig
    right: UmiSideConfig
    timing: UmiTimingConfig = field(default_factory=UmiTimingConfig)
    vive_host: str = "0.0.0.0"
    vive_port: int = 9999
    vive_clock_sync_port: int = 10001
    angle_trigger_rate_hz: float = 30.0
    angle_trigger_config_root: Path | None = None
    allow_identity_tcp_placeholder: bool = False
    calibration_validity_sec: float = 3600.0
    tactile_required: bool = True
    tactile_compression: str = "none"
    send_gripper_commands: bool = True
    dmtac_assignment: str = "latest"
    foot_pad_enabled: bool = False
    gui_host: str = "127.0.0.1"
    gui_port: int = 0
    open_browser: bool = True
    task_id: str = "umi_collection"
    task_description: str = "Dual-wrist UMI collection"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser())
        object.__setattr__(
            self, "calibration_result", Path(self.calibration_result).expanduser()
        )
        object.__setattr__(self, "policy_io_layout", Path(self.policy_io_layout).expanduser())
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("left/right UMI side bindings are invalid")
        if self.left.camera_serial == self.right.camera_serial:
            raise ValueError("left and right wrist cameras must use different serials")
        if self.left.tracker_name == self.right.tracker_name:
            raise ValueError("left and right sides must use different trackers")
        if int(self.vive_port) <= 0 or int(self.vive_clock_sync_port) < 0:
            raise ValueError("invalid Vive ports")
        if int(self.vive_port) == int(self.vive_clock_sync_port):
            raise ValueError("Vive data and clock-sync ports must differ")
        if float(self.calibration_validity_sec) <= 0.0:
            raise ValueError("calibration_validity_sec must be positive")
        if self.tactile_compression not in {"none", "lz4", "zstd"}:
            raise ValueError("tactile_compression must be one of: none, lz4, zstd")
        if not self.allow_identity_tcp_placeholder and (
            self.left.uses_identity_tcp_placeholder
            or self.right.uses_identity_tcp_placeholder
        ):
            raise ValueError(
                "identity TCP extrinsics require allow_identity_tcp_placeholder=true"
            )

    def side(self, side: str) -> UmiSideConfig:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError(f"unsupported side {side!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, base_dir: Path) -> "UmiCollectionConfig":
        sides = _mapping(payload.get("sides", {}), name="sides")
        timing_payload = _mapping(payload.get("timing", {}), name="timing")

        def resolve(value: Any) -> Path:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else base_dir / path

        angle_root = payload.get("angle_trigger_config_root")
        return cls(
            dataset_root=resolve(payload.get("dataset_root", "datasets/umi")),
            calibration_result=resolve(payload["calibration_result"]),
            policy_io_layout=resolve(
                payload.get(
                    "policy_io_layout",
                    "../policy_io_layouts/dual_umi_dmtac_v23.json",
                )
            ),
            left=UmiSideConfig.from_dict("left", _mapping(sides.get("left", {}), name="sides.left")),
            right=UmiSideConfig.from_dict("right", _mapping(sides.get("right", {}), name="sides.right")),
            timing=UmiTimingConfig(**dict(timing_payload)),
            vive_host=str(payload.get("vive_host", "0.0.0.0")),
            vive_port=int(payload.get("vive_port", 9999)),
            vive_clock_sync_port=int(payload.get("vive_clock_sync_port", 10001)),
            angle_trigger_rate_hz=float(payload.get("angle_trigger_rate_hz", 30.0)),
            angle_trigger_config_root=None if angle_root is None else resolve(angle_root),
            allow_identity_tcp_placeholder=bool(
                payload.get("allow_identity_tcp_placeholder", False)
            ),
            calibration_validity_sec=float(payload.get("calibration_validity_sec", 3600.0)),
            tactile_required=bool(payload.get("tactile_required", True)),
            tactile_compression=str(payload.get("tactile_compression", "none")),
            send_gripper_commands=bool(payload.get("send_gripper_commands", True)),
            dmtac_assignment=str(payload.get("dmtac_assignment", "latest")),
            foot_pad_enabled=bool(payload.get("foot_pad_enabled", False)),
            gui_host=str(payload.get("gui_host", "127.0.0.1")),
            gui_port=int(payload.get("gui_port", 0)),
            open_browser=bool(payload.get("open_browser", True)),
            task_id=str(payload.get("task_id", "umi_collection")),
            task_description=str(payload.get("task_description", "Dual-wrist UMI collection")),
        )


def load_umi_collection_config(path: str | Path) -> UmiCollectionConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("UMI collection config must contain a JSON object")
    return UmiCollectionConfig.from_dict(payload, base_dir=config_path.parent)


__all__ = [
    "DEFAULT_LEFT_WRIST_CAMERA_SERIAL",
    "DEFAULT_RIGHT_WRIST_CAMERA_SERIAL",
    "UmiCollectionConfig",
    "UmiSideConfig",
    "UmiTimingConfig",
    "load_umi_collection_config",
]
