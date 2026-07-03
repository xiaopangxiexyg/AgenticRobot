from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ArmSpec:
    """Joint and controlled-frame names for one simulated arm."""

    side: str
    joint_names: tuple[str, ...]
    controlled_body: str
    effector_name: str

    def __post_init__(self) -> None:
        if len(self.joint_names) != 6:
            raise ValueError(f"{self.side} arm expects 6 joints, got {len(self.joint_names)}.")


@dataclass(frozen=True)
class CameraMount:
    """A MuJoCo render camera mounted at a URDF link frame."""

    name: str
    link_name: str
    width: int = 640
    height: int = 480
    fovy_deg: float = 60.0
    intrinsics: Mapping[str, float] | None = None
    source: str = "urdf"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if self.fovy_deg <= 0.0:
            raise ValueError("Camera fovy_deg must be positive.")


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 80
    position_tolerance_m: float = 2e-3
    orientation_tolerance_rad: float = 2e-2
    damping: float = 5e-3
    max_joint_step_rad: float = 0.04
    orientation_weight: float = 0.35


@dataclass(frozen=True)
class Real2SimConfig:
    urdf_path: Path
    mode: str = "dual"
    camera_registry_path: Path | None = None
    camera_names: tuple[str, ...] = ("all",)
    camera_width: int = 640
    camera_height: int = 480
    default_camera_fovy_deg: float = 60.0
    timestep: float = 1.0 / 30.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ik: IKConfig = field(default_factory=IKConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"dual", "right"}:
            raise ValueError("mode must be 'dual' or 'right'.")
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive.")


def default_camera_registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "teleop_runtime"
        / "configs"
        / "cameras"
        / "realsense_registry.json"
    )


def as_tuple(values: Sequence[str] | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return default
    out = tuple(str(value) for value in values)
    return out if out else default

