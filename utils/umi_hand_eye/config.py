from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .tcp_extrinsic import DEFAULT_TCP_PROFILE, DEFAULT_URDF


CONFIG_SCHEMA = "dexteleop.umi_hand_eye.config.v1"


@dataclass(frozen=True)
class UmiSideConfig:
    camera_serial: str
    tracker_name: str
    tracker_role: str
    channel_name: str
    tcp_profile: str = DEFAULT_TCP_PROFILE

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, side: str) -> "UmiSideConfig":
        values = {
            name: str(payload.get(name, "")).strip()
            for name in (
                "camera_serial",
                "tracker_name",
                "tracker_role",
                "channel_name",
                "tcp_profile",
            )
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"UMI {side} config is missing: {', '.join(missing)}")
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        return {
            "camera_serial": self.camera_serial,
            "tracker_name": self.tracker_name,
            "tracker_role": self.tracker_role,
            "channel_name": self.channel_name,
            "tcp_profile": self.tcp_profile,
        }


@dataclass(frozen=True)
class UmiHandEyeConfig:
    sides: Mapping[str, UmiSideConfig]
    validity_seconds: float = 10800.0
    vive_host: str = "0.0.0.0"
    vive_port: int = 9999
    vive_clock_sync_port: int = 10001
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 15
    target: Mapping[str, Any] = field(default_factory=dict)
    urdf_path: Path = DEFAULT_URDF
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise ValueError(f"unsupported UMI config schema {self.schema!r}")
        normalized = {str(side).strip().lower(): value for side, value in self.sides.items()}
        if set(normalized) != {"left", "right"}:
            raise ValueError("UMI config must define exactly left and right sides")
        if float(self.validity_seconds) <= 0.0:
            raise ValueError("validity_seconds must be > 0")
        resolution = tuple(int(value) for value in self.camera_resolution)
        if len(resolution) != 2 or min(resolution) <= 0:
            raise ValueError("camera_resolution must contain two positive integers")
        if int(self.camera_fps) <= 0:
            raise ValueError("camera_fps must be > 0")
        object.__setattr__(self, "sides", normalized)
        object.__setattr__(self, "camera_resolution", resolution)
        object.__setattr__(self, "target", dict(self.target))
        object.__setattr__(self, "urdf_path", Path(self.urdf_path).expanduser().resolve())
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"UMI geometry URDF does not exist: {self.urdf_path}")

    def side(self, side: str) -> UmiSideConfig:
        normalized = str(side).strip().lower()
        try:
            return self.sides[normalized]
        except KeyError as exc:
            raise ValueError(f"side must be left or right, got {side!r}") from exc

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "UmiHandEyeConfig":
        schema = str(payload.get("schema", CONFIG_SCHEMA))
        raw_sides = payload.get("sides")
        if not isinstance(raw_sides, Mapping):
            raise ValueError("UMI config requires a sides mapping")
        sides = {
            str(side): UmiSideConfig.from_dict(value, side=str(side))
            for side, value in raw_sides.items()
        }
        urdf_value = Path(str(payload.get("urdf_path", DEFAULT_URDF))).expanduser()
        if not urdf_value.is_absolute() and base_dir is not None:
            urdf_value = base_dir / urdf_value
        return cls(
            sides=sides,
            validity_seconds=float(payload.get("validity_seconds", 10800.0)),
            vive_host=str(payload.get("vive_host", "0.0.0.0")),
            vive_port=int(payload.get("vive_port", 9999)),
            vive_clock_sync_port=int(payload.get("vive_clock_sync_port", 10001)),
            camera_resolution=tuple(payload.get("camera_resolution", (640, 480))),
            camera_fps=int(payload.get("camera_fps", 15)),
            target=dict(payload.get("target") or {}),
            urdf_path=urdf_value,
            schema=schema,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "validity_seconds": float(self.validity_seconds),
            "vive_host": self.vive_host,
            "vive_port": int(self.vive_port),
            "vive_clock_sync_port": int(self.vive_clock_sync_port),
            "camera_resolution": list(self.camera_resolution),
            "camera_fps": int(self.camera_fps),
            "sides": {side: config.to_dict() for side, config in self.sides.items()},
            "target": dict(self.target),
            "urdf_path": str(self.urdf_path),
        }


def load_config(path: str | Path) -> UmiHandEyeConfig:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("UMI config root must be an object")
    return UmiHandEyeConfig.from_dict(payload, base_dir=source.parent)
