"""Read-only RobStride J7 position source for B601/RM65 simulation.

The six-joint B601 owner deliberately does not register the optional gripper.
This module is a separate observation-only boundary: it registers only the
``gripper`` motor, reads ``mechPos`` through MotorBridge's correlated
parameter-reply API, and closes the controller without issuing a motor command.
It is therefore safe to use alongside the existing J1--J6 gravity/damping
owner.  It must not be used as a motion backend.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from robot_motion.rebot_b601_rs.gripper_observation import (
    B601_RS_GRIPPER_CLOCK_DOMAIN,
    B601_RS_GRIPPER_SOURCE_ID,
    B601RSGripperPositionObservation,
)

from .backend import (
    B601_RS_MECH_POS_PARAM_ID,
    B601HardwareDependencyError,
    B601HardwareError,
)

B601_RS_GRIPPER_JOINT_NAME = "gripper"
B601_RS_PARAMETER_REPLY_CORRELATION_CAPABILITY = "parameter_reply_correlation_v1"


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive(value: object, *, name: str) -> float:
    normalized = _finite(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative(value: object, *, name: str) -> float:
    normalized = _finite(value, name=name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        normalized = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 0 <= normalized <= 255:
        raise ValueError(f"{name} must be in 0..255")
    return normalized


def _parameter_id(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("parameter_id must be an integer")
    try:
        normalized = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("parameter_id must be an integer") from exc
    if not 0 <= normalized <= 0xFFFF:
        raise ValueError("parameter_id must be in 0..0xffff")
    return normalized


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class B601RSGripperReaderConfig:
    """Identity and timeout for the optional observation-only J7 reader."""

    channel: str = "can0"
    motor_id: int = 7
    feedback_id: int = 0xFD
    model: str = "rs-00"
    vendor: str = "robstride"
    parameter_id: int = B601_RS_MECH_POS_PARAM_ID
    timeout_sec: float = 0.10
    mit_kp: float = 50.0
    mit_kd: float = 4.0
    source_config: str | None = None

    def __post_init__(self) -> None:
        channel = str(self.channel).strip()
        if not channel:
            raise ValueError("channel must be non-empty")
        vendor = str(self.vendor).strip().lower()
        if vendor != "robstride":
            raise ValueError("B601 J7 reader supports only the robstride vendor")
        model = str(self.model).strip()
        if not model:
            raise ValueError("model must be non-empty")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "vendor", vendor)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "motor_id", _integer(self.motor_id, name="motor_id"))
        object.__setattr__(self, "feedback_id", _integer(self.feedback_id, name="feedback_id"))
        object.__setattr__(self, "parameter_id", _parameter_id(self.parameter_id))
        if self.motor_id <= 0:
            raise ValueError("motor_id must be in 1..255")
        if self.parameter_id != B601_RS_MECH_POS_PARAM_ID:
            raise ValueError("J7 reader is fixed to the RobStride mechPos parameter")
        object.__setattr__(self, "timeout_sec", _positive(self.timeout_sec, name="timeout_sec"))
        object.__setattr__(self, "mit_kp", _nonnegative(self.mit_kp, name="mit_kp"))
        object.__setattr__(self, "mit_kd", _nonnegative(self.mit_kd, name="mit_kd"))
        if self.source_config is not None:
            source = str(self.source_config).strip()
            object.__setattr__(self, "source_config", source or None)

    @classmethod
    def from_rs_yaml(cls, path: str | Path) -> B601RSGripperReaderConfig:
        config_path = Path(path).expanduser().resolve(strict=True)
        if not config_path.is_file():
            raise FileNotFoundError(f"B601-RS config is not a regular file: {config_path}")
        try:
            yaml = import_module("yaml")
        except ModuleNotFoundError as exc:
            raise B601HardwareDependencyError(
                "PyYAML is required to load reBotArm RS configuration"
            ) from exc
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = _mapping(payload, name="B601-RS config")
        groups = _mapping(data.get("groups"), name="groups")
        gripper_group = _mapping(groups.get("gripper"), name="groups.gripper")
        raw_names = gripper_group.get("joints")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            raise TypeError("groups.gripper.joints must be a sequence")
        names = tuple(str(value).strip() for value in raw_names)
        if names != (B601_RS_GRIPPER_JOINT_NAME,):
            raise ValueError("groups.gripper.joints must contain only gripper")
        raw_joints = data.get("joints")
        if not isinstance(raw_joints, Sequence) or isinstance(raw_joints, (str, bytes)):
            raise TypeError("joints must be a sequence")
        selected: Mapping[str, object] | None = None
        for raw_joint in raw_joints:
            joint = _mapping(raw_joint, name="joint")
            if str(joint.get("name", "")).strip() == B601_RS_GRIPPER_JOINT_NAME:
                if selected is not None:
                    raise ValueError("duplicate B601-RS gripper joint")
                selected = joint
        if selected is None:
            raise ValueError("missing B601-RS gripper joint")
        mit = _mapping(selected.get("MIT", {}), name="gripper.MIT")
        return cls(
            channel=str(data.get("channel", "can0")),
            motor_id=_integer(selected.get("motor_id"), name="gripper.motor_id"),
            feedback_id=_integer(selected.get("feedback_id"), name="gripper.feedback_id"),
            model=str(selected.get("model", "")).strip(),
            vendor=str(selected.get("vendor", "robstride")),
            mit_kp=mit.get("kp", 50.0),
            mit_kd=mit.get("kd", 4.0),
            source_config=str(config_path),
        )


class B601RSGripperReader:
    """Explicitly read-only MotorBridge J7 adapter."""

    def __init__(
        self,
        config: B601RSGripperReaderConfig,
        *,
        controller_factory=None,
        module_loader=None,
        clock=time.monotonic,
    ) -> None:
        if not isinstance(config, B601RSGripperReaderConfig):
            raise TypeError("config must be B601RSGripperReaderConfig")
        self.config = config
        self._controller_factory = controller_factory
        self._module_loader = module_loader
        self._clock = clock
        self._controller: object | None = None
        self._motor: object | None = None
        self._sequence = 0
        self._runtime_evidence: dict[str, object] | None = None

    @property
    def connected(self) -> bool:
        return self._controller is not None

    @property
    def motor_id(self) -> int:
        return self.config.motor_id

    @property
    def motorbridge_runtime_evidence(self) -> Mapping[str, object] | None:
        return None if self._runtime_evidence is None else dict(self._runtime_evidence)

    def connect(self) -> None:
        if self.connected:
            return
        controller = None
        try:
            if self._controller_factory is not None:
                controller = self._controller_factory(self.config.channel)
            else:
                module = self._module_loader() if self._module_loader else import_module("motorbridge")
                controller = module.Controller(self.config.channel)
            add_motor = controller.add_robstride_motor
            motor = add_motor(self.config.motor_id, self.config.feedback_id, self.config.model)
            if not callable(getattr(motor, "robstride_get_param_f32_host_id", None)):
                raise B601HardwareDependencyError(
                    "MotorBridge J7 reader requires robstride_get_param_f32_host_id"
                )
            self._controller = controller
            self._motor = motor
        except ModuleNotFoundError as exc:
            if controller is not None:
                close = getattr(controller, "close", None)
                if callable(close):
                    close()
            raise B601HardwareDependencyError(
                "motorbridge is required only when reading the B601 gripper"
            ) from exc
        except Exception:
            if controller is not None:
                close = getattr(controller, "close", None)
                if callable(close):
                    close()
            raise

    def read_position(self) -> B601RSGripperPositionObservation:
        if self._controller is None or self._motor is None:
            raise B601HardwareError("B601 J7 reader is not connected")
        started = _finite(self._clock(), name="clock")
        try:
            value = self._motor.robstride_get_param_f32_host_id(
                self.config.parameter_id,
                self.config.feedback_id,
                max(1, round(self.config.timeout_sec * 1000.0)),
            )
        except Exception as exc:
            raise B601HardwareError(f"B601 J7 mechPos read failed: {exc}") from exc
        received = _finite(self._clock(), name="clock")
        self._sequence += 1
        return B601RSGripperPositionObservation(
            position_rad=_finite(value, name="gripper position"),
            sequence=self._sequence,
            source_timestamp_sec=started,
            receive_timestamp_sec=received,
        )

    def close(self) -> None:
        """Close only the observation controller; never send motor I/O."""

        controller = self._controller
        self._controller = None
        self._motor = None
        if controller is not None:
            close = getattr(controller, "close", None)
            if callable(close):
                close()


__all__ = [
    "B601_RS_GRIPPER_CLOCK_DOMAIN",
    "B601_RS_GRIPPER_JOINT_NAME",
    "B601_RS_GRIPPER_SOURCE_ID",
    "B601RSGripperPositionObservation",
    "B601RSGripperReader",
    "B601RSGripperReaderConfig",
]
