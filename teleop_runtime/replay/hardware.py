from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hardware.rm65.calibration_profiles import ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD
from rm65_inspire_tac.paths import require_asset_path

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..deploy.schema import ARM_JOINT_DOF, xyzquat_to_pose_matrix
from ..schemas import EffectorCommand


ALL_SIDES = ("left", "right")


def normalize_active_sides(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"both", "all"}:
            requested = ("left", "right")
        else:
            requested = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    else:
        requested = tuple(str(part).strip().lower() for part in value)
    invalid = sorted(set(requested) - set(ALL_SIDES))
    if invalid:
        raise ValueError(f"Unsupported replay active side(s): {invalid}")
    ordered = tuple(side for side in ALL_SIDES if side in set(requested))
    if not ordered:
        raise ValueError("Replay active_sides must include left, right, or both.")
    return ordered


def _qpos_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        qpos = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(qpos)):
        return None
    return qpos.astype(float, copy=True)


def _pose_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose.astype(float, copy=True)


def _set_backend_max_angvel(backend: RM65ArmBackend, max_angvel: Sequence[float]) -> None:
    values = tuple(float(value) for value in max_angvel)
    if len(values) != ARM_JOINT_DOF or any(value <= 0.0 for value in values):
        raise ValueError("replay arm max_angvel must contain 6 positive values.")
    backend.config.max_angvel = values


@dataclass
class ReplayHardwareConfig:
    active_sides: tuple[str, ...] = ("right",)
    arm_ip_by_side: dict[str, str] = field(default_factory=dict)
    tool_profile_by_side: dict[str, str] = field(default_factory=dict)
    arm_control_hz: float = 50.0
    arm_max_angvel: tuple[float, ...] = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)
    replay_arm_max_angvel: tuple[float, ...] = (3.0, 3.0, 3.0, 4.0, 4.0, 4.0)
    background_streaming_enabled: bool = True
    connect_hardware: bool = True

    def __post_init__(self) -> None:
        self.active_sides = normalize_active_sides(self.active_sides)
        self.arm_ip_by_side = {str(side): str(ip) for side, ip in self.arm_ip_by_side.items() if str(ip)}
        self.tool_profile_by_side = {
            str(side): str(path)
            for side, path in self.tool_profile_by_side.items()
            if str(path)
        }
        if float(self.arm_control_hz) <= 0.0:
            raise ValueError("arm_control_hz must be positive.")
        self.arm_max_angvel = tuple(float(value) for value in self.arm_max_angvel)
        self.replay_arm_max_angvel = tuple(float(value) for value in self.replay_arm_max_angvel)
        if len(self.arm_max_angvel) != ARM_JOINT_DOF or len(self.replay_arm_max_angvel) != ARM_JOINT_DOF:
            raise ValueError("arm max_angvel tuples must contain 6 values.")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReplayHardwareConfig":
        active = payload.get("active_sides") or payload.get("activeSides") or "right"
        nested_ips = payload.get("arm_ip_by_side") or payload.get("armIpBySide") or {}
        nested_profiles = payload.get("tool_profile_by_side") or payload.get("toolProfileBySide") or {}
        if not isinstance(nested_ips, Mapping):
            nested_ips = {}
        if not isinstance(nested_profiles, Mapping):
            nested_profiles = {}
        arm_ip_by_side = {
            "left": str(payload.get("left_arm_ip") or payload.get("leftArmIp") or nested_ips.get("left") or ""),
            "right": str(payload.get("right_arm_ip") or payload.get("rightArmIp") or nested_ips.get("right") or ""),
        }
        tool_profile_by_side = {
            "left": str(
                payload.get("left_tool_profile")
                or payload.get("leftToolProfile")
                or nested_profiles.get("left")
                or ""
            ),
            "right": str(
                payload.get("right_tool_profile")
                or payload.get("rightToolProfile")
                or nested_profiles.get("right")
                or ""
            ),
        }
        return cls(
            active_sides=normalize_active_sides(active),
            arm_ip_by_side=arm_ip_by_side,
            tool_profile_by_side=tool_profile_by_side,
            arm_control_hz=float(payload.get("arm_control_hz") or payload.get("armControlHz") or 50.0),
            arm_max_angvel=_float_tuple(
                payload.get("arm_max_angvel") or payload.get("armMaxAngvel"),
                default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
            ),
            replay_arm_max_angvel=_float_tuple(
                payload.get("replay_arm_max_angvel") or payload.get("replayArmMaxAngvel"),
                default=(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
            ),
            background_streaming_enabled=bool(
                payload.get("background_streaming_enabled")
                if "background_streaming_enabled" in payload
                else payload.get("backgroundStreamingEnabled", True)
            ),
            connect_hardware=bool(
                payload.get("connect_hardware")
                if "connect_hardware" in payload
                else payload.get("connectHardware", True)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_sides": list(self.active_sides),
            "arm_ip_by_side": dict(self.arm_ip_by_side),
            "tool_profile_by_side": dict(self.tool_profile_by_side),
            "arm_control_hz": float(self.arm_control_hz),
            "arm_max_angvel": list(self.arm_max_angvel),
            "replay_arm_max_angvel": list(self.replay_arm_max_angvel),
            "background_streaming_enabled": bool(self.background_streaming_enabled),
            "connect_hardware": bool(self.connect_hardware),
        }


def _float_tuple(value: Any, *, default: Sequence[float]) -> tuple[float, ...]:
    if value is None or value == "":
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    return tuple(float(item) for item in parts)


class ReplayHardwareRig:
    def __init__(self, config: ReplayHardwareConfig):
        self.config = config
        self.arm_backends: dict[str, RM65ArmBackend] = {}
        self._original_max_angvel_by_side: dict[str, tuple[float, ...]] = {}

    @property
    def connected(self) -> bool:
        return all(side in self.arm_backends and self.arm_backends[side].is_ready() for side in self.config.active_sides)

    def connect(self) -> None:
        arm_urdf_path = require_asset_path("rm65_arm.urdf")
        control_dt = 1.0 / float(self.config.arm_control_hz)
        for side in self.config.active_sides:
            ip = str(self.config.arm_ip_by_side.get(side) or "").strip()
            if self.config.connect_hardware and not ip:
                raise ValueError(f"{side}_arm_ip is required for hardware replay.")
            backend = RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip=ip or "0.0.0.0",
                    side=side,
                    arm_urdf_path=arm_urdf_path,
                    connect_hardware=bool(self.config.connect_hardware),
                    control_dt=control_dt,
                    max_angvel=tuple(float(value) for value in self.config.arm_max_angvel),
                    dry_run_fast=False,
                    tool_profile=self.config.tool_profile_by_side.get(side) or None,
                    background_streaming_enabled=bool(self.config.background_streaming_enabled),
                    background_streaming_hz=float(self.config.arm_control_hz),
                )
            )
            backend.start()
            self.arm_backends[side] = backend
            qpos = self.current_qpos(side)
            if qpos is not None:
                backend.set_local_qpos(qpos)

    def close(self) -> None:
        self.restore_replay_speed()
        for backend in self.arm_backends.values():
            try:
                backend.close()
            except Exception:
                pass
        self.arm_backends.clear()

    def begin_replay_speed(self) -> None:
        for side, backend in self.arm_backends.items():
            if side not in self.config.active_sides or side in self._original_max_angvel_by_side:
                continue
            self._original_max_angvel_by_side[side] = tuple(float(value) for value in backend.config.max_angvel)
            _set_backend_max_angvel(backend, self.config.replay_arm_max_angvel)

    def restore_replay_speed(self) -> None:
        for side, max_angvel in list(self._original_max_angvel_by_side.items()):
            backend = self.arm_backends.get(side)
            if backend is not None:
                _set_backend_max_angvel(backend, max_angvel)
        self._original_max_angvel_by_side.clear()

    def refresh_state(self, side: str) -> dict[str, Any]:
        backend = self.arm_backends.get(str(side))
        if backend is None:
            return {}
        try:
            return dict(backend.refresh_state_from_hardware())
        except Exception:
            return dict(backend.get_state())

    def current_qpos(self, side: str) -> np.ndarray | None:
        state = self.refresh_state(side)
        for key in ("hardware_qpos", "current_qpos", "last_target_qpos"):
            qpos = _qpos_or_none(state.get(key))
            if qpos is not None:
                return qpos
        return None

    def current_qpos_by_side(self) -> dict[str, np.ndarray | None]:
        return {side: self.current_qpos(side) for side in self.config.active_sides}

    def current_tcp_pose(self, side: str) -> np.ndarray | None:
        backend = self.arm_backends.get(str(side))
        if backend is None:
            return None
        try:
            return _pose_or_none(backend.compute_tcp_pose())
        except Exception:
            state = self.refresh_state(side)
            return _pose_or_none(state.get("current_tcp_pose"))

    def current_tcp_pose_by_side(self) -> dict[str, np.ndarray | None]:
        return {side: self.current_tcp_pose(side) for side in self.config.active_sides}

    def current_tcp_flange(self, side: str) -> np.ndarray:
        backend = self.arm_backends.get(str(side))
        if backend is None:
            return np.asarray(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD, dtype=float).reshape(6)
        getter = getattr(backend, "_active_tcp_pose_flange_m_rad", None)
        if callable(getter):
            return np.asarray(getter(), dtype=float).reshape(6)
        value = backend.config.tcp_pose_flange_m_rad
        if value is None:
            value = ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD
        return np.asarray(value, dtype=float).reshape(6)

    def current_tcp_flange_by_side(self) -> dict[str, np.ndarray]:
        return {side: self.current_tcp_flange(side) for side in self.config.active_sides}

    def send_joint_targets(self, qpos_by_side: Mapping[str, Any], *, source: str) -> None:
        for side, qpos in qpos_by_side.items():
            backend = self.arm_backends.get(str(side))
            if backend is None:
                continue
            target = np.asarray(qpos, dtype=float).reshape(ARM_JOINT_DOF)
            backend.send_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=str(side),
                    command_type="joint_vector",
                    value=target,
                    frame_name=None,
                    metadata={"source": source},
                )
            )

    def send_action_poses(self, action_by_side: Mapping[str, Any], *, source: str) -> None:
        for side, xyzquat in action_by_side.items():
            backend = self.arm_backends.get(str(side))
            if backend is None:
                continue
            pose = xyzquat_to_pose_matrix(xyzquat)
            if pose is None:
                raise RuntimeError(f"{side} action pose is invalid; command not sent.")
            backend.send_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=str(side),
                    command_type="cartesian_pose",
                    value=pose,
                    frame_name=f"{side}_arm_base",
                    metadata={"source": source, "target_pose_base": pose.copy()},
                )
            )

    def qpos_reached(self, qpos_by_side: Mapping[str, Any], *, tolerance: float) -> bool:
        for side, target in qpos_by_side.items():
            current = self.current_qpos(str(side))
            if current is None:
                return False
            target_qpos = np.asarray(target, dtype=float).reshape(ARM_JOINT_DOF)
            if float(np.max(np.abs(target_qpos - current))) > float(tolerance):
                return False
        return True
