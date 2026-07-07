from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hardware.rm65.calibration_profiles import ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD
from rm65_inspire_tac.paths import require_asset_path

from ..action.executor import RobotActionResolver
from ..action.schema import RobotAction
from ..backends.arms import (
    DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
    RM65ArmBackend,
    RM65ArmBackendConfig,
)
from ..backends.grippers import DMTacGripperBackend, DMTacGripperBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..deploy.schema import ARM_JOINT_DOF
from ..interfaces import EffectorBackend
from ..schemas import EffectorCommand, TeleopCommand


ALL_SIDES = ("left", "right")
END_EFFECTOR_BACKEND_NONE = "none"
END_EFFECTOR_BACKEND_INSPIRE_TAC = "inspire_tac"
END_EFFECTOR_BACKEND_DMTAC_GRIPPER = "dmtac_gripper"
END_EFFECTOR_BACKENDS = (
    END_EFFECTOR_BACKEND_NONE,
    END_EFFECTOR_BACKEND_INSPIRE_TAC,
    END_EFFECTOR_BACKEND_DMTAC_GRIPPER,
)
_SINGULARITY_GATE_HOLD_STATUSES = frozenset(
    {
        "hold_entry",
    }
)


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


def normalize_end_effector_backend(value: Any) -> str:
    text = str(value or END_EFFECTOR_BACKEND_NONE).strip().lower().replace("-", "_")
    aliases = {
        "": END_EFFECTOR_BACKEND_NONE,
        "off": END_EFFECTOR_BACKEND_NONE,
        "disabled": END_EFFECTOR_BACKEND_NONE,
        "hand": END_EFFECTOR_BACKEND_INSPIRE_TAC,
        "inspire": END_EFFECTOR_BACKEND_INSPIRE_TAC,
        "inspire_hand": END_EFFECTOR_BACKEND_INSPIRE_TAC,
        "inspire_tac_hand": END_EFFECTOR_BACKEND_INSPIRE_TAC,
        "dmtac": END_EFFECTOR_BACKEND_DMTAC_GRIPPER,
        "dmtac_gripper": END_EFFECTOR_BACKEND_DMTAC_GRIPPER,
        "dm_tac_gripper": END_EFFECTOR_BACKEND_DMTAC_GRIPPER,
    }
    normalized = aliases.get(text, text)
    if normalized not in END_EFFECTOR_BACKENDS:
        raise ValueError(
            f"Unsupported end-effector backend {value!r}; expected one of {END_EFFECTOR_BACKENDS}."
        )
    return normalized


@dataclass
class ReplayHardwareConfig:
    active_sides: tuple[str, ...] = ("right",)
    arm_ip_by_side: dict[str, str] = field(default_factory=dict)
    tool_profile_by_side: dict[str, str] = field(default_factory=dict)
    end_effectors_enabled: bool = False
    end_effector_backend_by_side: dict[str, str] = field(default_factory=dict)
    hand_ip_by_side: dict[str, str] = field(default_factory=dict)
    hand_port_by_side: dict[str, int] = field(default_factory=dict)
    gripper_port_by_side: dict[str, str] = field(default_factory=dict)
    gripper_id_by_side: dict[str, int] = field(default_factory=dict)
    gripper_speed: float = 1.0
    gripper_torque: float = 1.0
    gripper_max_action: float = 1.0
    arm_control_hz: float = 50.0
    arm_max_angvel: tuple[float, ...] = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)
    replay_arm_max_angvel: tuple[float, ...] = (3.0, 3.0, 3.0, 4.0, 4.0, 4.0)
    singularity_qpos_init_slowdown: float = DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN
    background_streaming_enabled: bool = True
    connect_hardware: bool = True
    rm65_ik_framework: str = "robot_motion"

    def __post_init__(self) -> None:
        self.active_sides = normalize_active_sides(self.active_sides)
        self.arm_ip_by_side = {str(side): str(ip) for side, ip in self.arm_ip_by_side.items() if str(ip)}
        self.tool_profile_by_side = {
            str(side): str(path)
            for side, path in self.tool_profile_by_side.items()
            if str(path)
        }
        self.end_effector_backend_by_side = {
            side: normalize_end_effector_backend(self.end_effector_backend_by_side.get(side))
            for side in self.active_sides
        }
        self.hand_ip_by_side = {str(side): str(ip) for side, ip in self.hand_ip_by_side.items() if str(ip)}
        self.hand_port_by_side = {
            str(side): int(port)
            for side, port in self.hand_port_by_side.items()
            if int(port) > 0
        }
        self.gripper_port_by_side = {
            str(side): str(port)
            for side, port in self.gripper_port_by_side.items()
            if str(port)
        }
        self.gripper_id_by_side = {
            str(side): int(value)
            for side, value in self.gripper_id_by_side.items()
            if int(value) > 0
        }
        self.gripper_speed = float(np.clip(float(self.gripper_speed), 0.0, 1.0))
        self.gripper_torque = float(np.clip(float(self.gripper_torque), 0.0, 1.0))
        self.gripper_max_action = float(np.clip(float(self.gripper_max_action), 0.0, 1.0))
        if float(self.arm_control_hz) <= 0.0:
            raise ValueError("arm_control_hz must be positive.")
        self.arm_max_angvel = tuple(float(value) for value in self.arm_max_angvel)
        self.replay_arm_max_angvel = tuple(float(value) for value in self.replay_arm_max_angvel)
        if len(self.arm_max_angvel) != ARM_JOINT_DOF or len(self.replay_arm_max_angvel) != ARM_JOINT_DOF:
            raise ValueError("arm max_angvel tuples must contain 6 values.")
        self.singularity_qpos_init_slowdown = float(self.singularity_qpos_init_slowdown)
        if not 0.0 < self.singularity_qpos_init_slowdown <= 1.0:
            raise ValueError("singularity_qpos_init_slowdown must be in (0, 1].")
        self.rm65_ik_framework = str(self.rm65_ik_framework or "robot_motion")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReplayHardwareConfig":
        active = payload.get("active_sides") or payload.get("activeSides") or "right"
        nested_ips = payload.get("arm_ip_by_side") or payload.get("armIpBySide") or {}
        nested_profiles = payload.get("tool_profile_by_side") or payload.get("toolProfileBySide") or {}
        if not isinstance(nested_ips, Mapping):
            nested_ips = {}
        if not isinstance(nested_profiles, Mapping):
            nested_profiles = {}
        nested_end_effector_backends = (
            payload.get("end_effector_backend_by_side")
            or payload.get("endEffectorBackendBySide")
            or {}
        )
        nested_hand_ips = payload.get("hand_ip_by_side") or payload.get("handIpBySide") or {}
        nested_hand_ports = payload.get("hand_port_by_side") or payload.get("handPortBySide") or {}
        nested_gripper_ports = payload.get("gripper_port_by_side") or payload.get("gripperPortBySide") or {}
        nested_gripper_ids = payload.get("gripper_id_by_side") or payload.get("gripperIdBySide") or {}
        if not isinstance(nested_end_effector_backends, Mapping):
            nested_end_effector_backends = {}
        if not isinstance(nested_hand_ips, Mapping):
            nested_hand_ips = {}
        if not isinstance(nested_hand_ports, Mapping):
            nested_hand_ports = {}
        if not isinstance(nested_gripper_ports, Mapping):
            nested_gripper_ports = {}
        if not isinstance(nested_gripper_ids, Mapping):
            nested_gripper_ids = {}
        default_backend = payload.get("end_effector_backend") or payload.get("endEffectorBackend") or "none"
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
        end_effector_backend_by_side = {
            "left": str(
                payload.get("left_end_effector_backend")
                or payload.get("leftEndEffectorBackend")
                or nested_end_effector_backends.get("left")
                or default_backend
            ),
            "right": str(
                payload.get("right_end_effector_backend")
                or payload.get("rightEndEffectorBackend")
                or nested_end_effector_backends.get("right")
                or default_backend
            ),
        }
        hand_ip_by_side = {
            "left": str(payload.get("left_hand_ip") or payload.get("leftHandIp") or nested_hand_ips.get("left") or ""),
            "right": str(payload.get("right_hand_ip") or payload.get("rightHandIp") or nested_hand_ips.get("right") or ""),
        }
        hand_port_by_side: dict[str, int] = {}
        for side in ALL_SIDES:
            value = (
                payload.get(f"{side}_hand_port")
                or payload.get(f"{side}HandPort")
                or nested_hand_ports.get(side)
            )
            if value not in (None, ""):
                hand_port_by_side[side] = int(value)
        gripper_port_by_side = {
            "left": str(
                payload.get("left_gripper_port")
                or payload.get("leftGripperPort")
                or nested_gripper_ports.get("left")
                or ""
            ),
            "right": str(
                payload.get("right_gripper_port")
                or payload.get("rightGripperPort")
                or nested_gripper_ports.get("right")
                or ""
            ),
        }
        gripper_id_by_side: dict[str, int] = {}
        for side in ALL_SIDES:
            value = (
                payload.get(f"{side}_gripper_id")
                or payload.get(f"{side}GripperId")
                or nested_gripper_ids.get(side)
            )
            if value not in (None, ""):
                gripper_id_by_side[side] = int(value)
        return cls(
            active_sides=normalize_active_sides(active),
            arm_ip_by_side=arm_ip_by_side,
            tool_profile_by_side=tool_profile_by_side,
            end_effectors_enabled=bool(
                payload.get("end_effectors_enabled")
                if "end_effectors_enabled" in payload
                else payload.get(
                    "endEffectorsEnabled",
                    payload.get("hands_enabled", payload.get("handsEnabled", False)),
                )
            ),
            end_effector_backend_by_side=end_effector_backend_by_side,
            hand_ip_by_side=hand_ip_by_side,
            hand_port_by_side=hand_port_by_side,
            gripper_port_by_side=gripper_port_by_side,
            gripper_id_by_side=gripper_id_by_side,
            gripper_speed=float(payload.get("gripper_speed") or payload.get("gripperSpeed") or 1.0),
            gripper_torque=float(
                payload.get("gripper_torque")
                or payload.get("gripperTorque")
                or payload.get("gripper_force")
                or payload.get("gripperForce")
                or 1.0
            ),
            gripper_max_action=float(payload.get("gripper_max_action") or payload.get("gripperMaxAction") or 1.0),
            arm_control_hz=float(payload.get("arm_control_hz") or payload.get("armControlHz") or 50.0),
            arm_max_angvel=_float_tuple(
                payload.get("arm_max_angvel") or payload.get("armMaxAngvel"),
                default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
            ),
            replay_arm_max_angvel=_float_tuple(
                payload.get("replay_arm_max_angvel") or payload.get("replayArmMaxAngvel"),
                default=(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
            ),
            singularity_qpos_init_slowdown=float(
                payload.get(
                    "singularity_qpos_init_slowdown",
                    payload.get("singularityQposInitSlowdown", DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN),
                )
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
            rm65_ik_framework=str(
                payload.get("rm65_ik_framework")
                or payload.get("rm65IkFramework")
                or "robot_motion"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_sides": list(self.active_sides),
            "arm_ip_by_side": dict(self.arm_ip_by_side),
            "tool_profile_by_side": dict(self.tool_profile_by_side),
            "end_effectors_enabled": bool(self.end_effectors_enabled),
            "end_effector_backend_by_side": dict(self.end_effector_backend_by_side),
            "hand_ip_by_side": dict(self.hand_ip_by_side),
            "hand_port_by_side": dict(self.hand_port_by_side),
            "gripper_port_by_side": dict(self.gripper_port_by_side),
            "gripper_id_by_side": dict(self.gripper_id_by_side),
            "gripper_speed": float(self.gripper_speed),
            "gripper_torque": float(self.gripper_torque),
            "gripper_max_action": float(self.gripper_max_action),
            "arm_control_hz": float(self.arm_control_hz),
            "arm_max_angvel": list(self.arm_max_angvel),
            "replay_arm_max_angvel": list(self.replay_arm_max_angvel),
            "singularity_qpos_init_slowdown": float(self.singularity_qpos_init_slowdown),
            "background_streaming_enabled": bool(self.background_streaming_enabled),
            "connect_hardware": bool(self.connect_hardware),
            "rm65_ik_framework": str(self.rm65_ik_framework),
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
        self.effector_backends: dict[str, EffectorBackend] = {}
        self._original_max_angvel_by_side: dict[str, tuple[float, ...]] = {}
        self._action_resolver = RobotActionResolver()

    @property
    def action_resolver(self) -> RobotActionResolver:
        return self._action_resolver

    @property
    def connected(self) -> bool:
        for side in self.config.active_sides:
            arm = self.arm_backends.get(side)
            if arm is None or not arm.is_ready():
                return False
            if self.config.end_effectors_enabled:
                backend = self.effector_backends.get(f"{side}_hand")
                if backend is None or not backend.is_ready():
                    return False
        return True

    def end_effector_backend_sides(self) -> tuple[str, ...]:
        return tuple(
            side
            for side in self.config.active_sides
            if f"{side}_hand" in self.effector_backends
        )

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
                    singularity_qpos_init_slowdown=float(self.config.singularity_qpos_init_slowdown),
                    dry_run_fast=False,
                    tool_profile=self.config.tool_profile_by_side.get(side) or None,
                    ik_framework=self.config.rm65_ik_framework,
                    background_streaming_enabled=bool(self.config.background_streaming_enabled),
                    background_streaming_hz=float(self.config.arm_control_hz),
                )
            )
            backend.start()
            self.arm_backends[side] = backend
            self.effector_backends[f"{side}_arm"] = backend
            qpos = self.current_qpos(side)
            if qpos is not None:
                backend.set_local_qpos(qpos)
        if self.config.end_effectors_enabled:
            for side in self.config.active_sides:
                end_effector = self._build_end_effector_backend(side)
                if end_effector is None:
                    raise ValueError(
                        f"{side} replay end-effector backend is required when "
                        "end_effectors_enabled is true."
                    )
                end_effector.start()
                self.effector_backends[f"{side}_hand"] = end_effector

    def _build_end_effector_backend(self, side: str) -> EffectorBackend | None:
        backend_name = normalize_end_effector_backend(
            self.config.end_effector_backend_by_side.get(str(side))
        )
        if backend_name == END_EFFECTOR_BACKEND_NONE:
            return None
        if backend_name == END_EFFECTOR_BACKEND_INSPIRE_TAC:
            ip = str(self.config.hand_ip_by_side.get(str(side)) or "").strip()
            if self.config.connect_hardware and not ip:
                raise ValueError(f"{side}_hand_ip is required for inspire_tac replay end-effector.")
            return InspireTacHandBackend(
                InspireTacHandBackendConfig(
                    hand_ip=ip or "0.0.0.0",
                    side=str(side),
                    port=int(self.config.hand_port_by_side.get(str(side), 6000)),
                    connect_hardware=bool(self.config.connect_hardware),
                    enable_telemetry=False,
                )
            )
        if backend_name == END_EFFECTOR_BACKEND_DMTAC_GRIPPER:
            port = str(self.config.gripper_port_by_side.get(str(side)) or "").strip()
            if self.config.connect_hardware and not port:
                raise ValueError(f"{side}_gripper_port is required for dmtac_gripper replay end-effector.")
            return DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side=str(side),
                    gripper_port=port or "/dev/null",
                    gripper_id=int(self.config.gripper_id_by_side.get(str(side), 1)),
                    connect_hardware=bool(self.config.connect_hardware),
                    connect_tactile=False,
                    enable_telemetry=False,
                    speed=float(self.config.gripper_speed),
                    torque=float(self.config.gripper_torque),
                    max_action=float(self.config.gripper_max_action),
                )
            )
        raise ValueError(f"Unsupported replay end-effector backend {backend_name!r}.")

    def close(self) -> None:
        self.restore_replay_speed()
        closed: set[int] = set()
        for backend in self.effector_backends.values():
            marker = id(backend)
            if marker in closed:
                continue
            closed.add(marker)
            try:
                backend.close()
            except Exception:
                pass
        self.arm_backends.clear()
        self.effector_backends.clear()

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

    def hold_motion(self, reason: str = "hardware replay hold") -> None:
        errors: list[str] = []
        for effector_name, backend in self.effector_backends.items():
            if effector_name not in {
                f"{side}_{kind}"
                for side in self.config.active_sides
                for kind in ("arm", "hand")
            }:
                continue
            try:
                backend.hold_motion(reason)
            except Exception as exc:  # noqa: BLE001 - stop all active arms before reporting.
                errors.append(f"{effector_name}: {exc}")
        if errors:
            raise RuntimeError("hardware replay hold failed: " + "; ".join(errors))

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
        command = TeleopCommand(metadata={"source": source})
        for side, qpos in qpos_by_side.items():
            target = np.asarray(qpos, dtype=float).reshape(ARM_JOINT_DOF)
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=str(side),
                    command_type="joint_vector",
                    value=target,
                    frame_name=None,
                    metadata={"source": source},
                )
            )
        self.send_commands(command)

    def send_commands(
        self,
        command: TeleopCommand,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> None:
        del phase_recorder
        for effector_name, effector_command in command.commands.items():
            backend = self.effector_backends.get(str(effector_name))
            if backend is None:
                raise KeyError(f"ReplayHardwareRig backend missing for effector {effector_name!r}.")
            backend.send_command(effector_command)
            _raise_if_singularity_gate_hold(str(effector_name), backend)

    def send_action_poses(self, action_by_side: Mapping[str, Any], *, source: str) -> None:
        arm_pose_by_side: dict[str, np.ndarray] = {}
        for side, xyzquat in action_by_side.items():
            side_text = str(side)
            if self.arm_backends.get(side_text) is None:
                continue
            arm_pose_by_side[side_text] = np.asarray(xyzquat, dtype=np.float32).reshape(7)
        if not arm_pose_by_side:
            return
        action = RobotAction.bimanual_ee_pose(
            left_arm=arm_pose_by_side.get("left"),
            right_arm=arm_pose_by_side.get("right"),
            timestamp=time.time(),
            source=str(source),
            metadata={"replay_source": str(source)},
        )
        command = self._action_resolver.resolve(action)
        command.metadata["source"] = str(source)
        for effector_command in command.commands.values():
            effector_command.metadata["source"] = str(source)
            effector_command.metadata["replay_source"] = str(source)
            if effector_command.command_type == "cartesian_pose":
                pose = np.asarray(effector_command.value, dtype=np.float32).reshape(4, 4)
                controlled_frame = str(
                    effector_command.metadata.get("arm_controlled_frame")
                    or effector_command.metadata.get("target_pose_controlled_frame")
                    or "tool_tcp"
                )
                effector_command.metadata.setdefault("source_target_pose_base", pose.copy())
                effector_command.metadata.setdefault("target_pose_tcp_base", pose.copy())
                effector_command.metadata.setdefault("target_pose_retarget_frame", controlled_frame)
        self.send_commands(command)

    def qpos_reached(self, qpos_by_side: Mapping[str, Any], *, tolerance: float) -> bool:
        for side, target in qpos_by_side.items():
            current = self.current_qpos(str(side))
            if current is None:
                return False
            target_qpos = np.asarray(target, dtype=float).reshape(ARM_JOINT_DOF)
            if float(np.max(np.abs(target_qpos - current))) > float(tolerance):
                return False
        return True


def _raise_if_singularity_gate_hold(effector_name: str, backend: EffectorBackend) -> None:
    get_state = getattr(backend, "get_state", None)
    if not callable(get_state):
        return
    try:
        state = get_state()
    except Exception:
        return
    if not isinstance(state, Mapping):
        return
    metadata = state.get("singularity_metadata")
    if not isinstance(metadata, Mapping):
        ik_metadata = state.get("ik_metadata")
        metadata = ik_metadata if isinstance(ik_metadata, Mapping) else {}
    status = str(metadata.get("singularity_gate_status") or "")
    if status not in _SINGULARITY_GATE_HOLD_STATUSES:
        return
    raise RuntimeError(
        f"{effector_name} replay command blocked by singularity gate: "
        f"status={status} "
        f"source={metadata.get('source')} "
        f"target={metadata.get('target_singularity_label')} "
        f"current={metadata.get('current_singularity_label')}"
    )
