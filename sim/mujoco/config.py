from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cameras.configs import DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH


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
    intrinsics: Mapping[str, object] | None = None
    source: str = "urdf"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if self.fovy_deg <= 0.0:
            raise ValueError("Camera fovy_deg must be positive.")


@dataclass(frozen=True)
class GripperSpec:
    """Normalized gripper command mapping for one simulated gripper module.

    ``close=0`` means fully open and ``close=1`` means fully closed. The mapping
    is per joint so different gripper modules can share one command surface even
    when their MuJoCo joint directions or ranges differ.
    """

    name: str
    joint_names: tuple[str, ...]
    open_qpos: tuple[float, ...]
    closed_qpos: tuple[float, ...]
    effector_name: str = "gripper"
    command_name: str = "gripper_close"
    side: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("GripperSpec.name must be non-empty.")
        joint_names = tuple(str(joint) for joint in self.joint_names)
        if not joint_names:
            raise ValueError(f"{name} gripper must define at least one joint.")
        open_qpos = tuple(float(value) for value in self.open_qpos)
        closed_qpos = tuple(float(value) for value in self.closed_qpos)
        if len(open_qpos) != len(joint_names) or len(closed_qpos) != len(joint_names):
            raise ValueError(
                f"{name} gripper qpos vectors must match joint count {len(joint_names)}."
            )
        if np.allclose(np.asarray(open_qpos), np.asarray(closed_qpos)):
            raise ValueError(f"{name} gripper open_qpos and closed_qpos must differ.")
        side = None if self.side is None else str(self.side)
        if side is not None and side not in {"left", "right"}:
            raise ValueError("GripperSpec.side must be 'left', 'right', or None.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "open_qpos", open_qpos)
        object.__setattr__(self, "closed_qpos", closed_qpos)
        object.__setattr__(self, "effector_name", str(self.effector_name or name))
        object.__setattr__(self, "command_name", str(self.command_name or "gripper_close"))
        object.__setattr__(self, "side", side)

    def qpos_from_close(self, close: float) -> np.ndarray:
        close_value = float(np.clip(float(close), 0.0, 1.0))
        open_qpos = np.asarray(self.open_qpos, dtype=float)
        closed_qpos = np.asarray(self.closed_qpos, dtype=float)
        return open_qpos + close_value * (closed_qpos - open_qpos)

    def close_from_qpos(self, qpos: Sequence[float]) -> float:
        values = np.asarray(qpos, dtype=float).reshape(len(self.joint_names))
        open_qpos = np.asarray(self.open_qpos, dtype=float)
        span = np.asarray(self.closed_qpos, dtype=float) - open_qpos
        active = np.abs(span) > 1e-12
        if not np.any(active):
            return 0.0
        close_values = (values[active] - open_qpos[active]) / span[active]
        return float(np.clip(float(np.mean(close_values)), 0.0, 1.0))

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "joint_names": list(self.joint_names),
            "open_qpos": list(self.open_qpos),
            "closed_qpos": list(self.closed_qpos),
            "effector_name": self.effector_name,
            "command_name": self.command_name,
            "side": self.side,
        }


@dataclass(frozen=True)
class Real2SimConfig:
    """Legacy per-call kinematic configuration.

    New physical-scene entrypoints should load ``sim.contracts.SimulationContract``
    instead. This type remains for compatibility with older local callers.
    """

    urdf_path: Path
    mode: str = "dual"
    camera_registry_path: Path | None = None
    camera_names: tuple[str, ...] = ("all",)
    camera_width: int = 640
    camera_height: int = 480
    default_camera_fovy_deg: float = 60.0
    timestep: float = 1.0 / 30.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.mode not in {"dual", "right"}:
            raise ValueError("mode must be 'dual' or 'right'.")
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive.")


def default_camera_registry_path() -> Path:
    return DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH


def as_tuple(values: Sequence[str] | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return default
    out = tuple(str(value) for value in values)
    return out if out else default
