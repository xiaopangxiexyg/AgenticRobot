"""Wired RobStride backend boundary for the reBotArm B601-RS.

This module is deliberately a thin adapter around the pinned
``reBotArm_control_py``/``motorbridge`` stack.  It owns neither kinematics nor
the haptic law; those remain in :mod:`robot_motion.rebot_b601_rs`.  Importing
the module is side-effect free.  A controller is opened only by an explicit
call to :meth:`B601RSBackend.connect`, and every command path is disabled
unless the caller constructed the backend with ``allow_motion=True``.

The first version intentionally uses the vendor cache as a *poll observation*.
MotorBridge 0.4.9 does not expose the source timestamp or a receive sequence
for a RobStride frame, so this adapter does not claim verified per-frame
freshness.  The distinction is part of the public state contract and must be
closed before hardware haptic acceptance.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_JOINT_MIT_FRAME,
    B601MitReference,
)

B601_RS_POSITION_FRAME = "b601_rs/joint_position_velocity_reference"
B601_RS_FEEDBACK_FRAME = "b601_rs/vendor_feedback_cache"
B601_RS_DEFAULT_CHANNEL = "can0"
B601_RS_DEFAULT_RATE_HZ = 500.0
B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC = 0.10
B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC = 0.10


class B601HardwareError(RuntimeError):
    """Base error for the explicit wired backend boundary."""


class B601HardwareDependencyError(B601HardwareError):
    """The optional vendor runtime is unavailable."""


class B601MotionAuthorizationError(B601HardwareError):
    """A command was attempted without an explicit motion authorization."""


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_float(value: object, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _six_finite(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    try:
        normalized = tuple(_finite_float(value, name=name) for value in values)
    except TypeError:
        raise ValueError(f"{name} must contain six finite values") from None
    if len(normalized) != 6:
        raise ValueError(f"{name} must contain six finite values")
    return normalized


def _six_nonnegative(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    normalized = _six_finite(values, name=name)
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"{name} must contain six non-negative values")
    return normalized


def _yaml_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        normalized = int(value, 0) if isinstance(value, str) else int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class B601JointConfig:
    """One arm joint's identity and vendor-side command defaults."""

    name: str
    motor_id: int
    feedback_id: int
    model: str
    vendor: str = "robstride"
    mit_kp: float = 0.0
    mit_kd: float = 0.0
    position_velocity_limit_rad_s: float = 0.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("joint name must not be empty")
        vendor = str(self.vendor).strip().lower()
        if vendor != "robstride":
            raise ValueError("B601-RS backend supports only the robstride vendor")
        if not str(self.model).strip():
            raise ValueError("joint model must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "vendor", vendor)
        motor_id = _yaml_int(self.motor_id, name="motor_id")
        feedback_id = _yaml_int(self.feedback_id, name="feedback_id")
        if not 1 <= motor_id <= 255:
            raise ValueError("motor_id must be in the RobStride range 1..255")
        if not 0 <= feedback_id <= 255:
            raise ValueError("feedback_id must be in the RobStride range 0..255")
        object.__setattr__(self, "motor_id", motor_id)
        object.__setattr__(
            self,
            "feedback_id",
            feedback_id,
        )
        object.__setattr__(self, "model", str(self.model).strip())
        object.__setattr__(self, "mit_kp", _finite_float(self.mit_kp, name="mit_kp"))
        object.__setattr__(self, "mit_kd", _finite_float(self.mit_kd, name="mit_kd"))
        object.__setattr__(
            self,
            "position_velocity_limit_rad_s",
            _positive_float(
                self.position_velocity_limit_rad_s,
                name="position_velocity_limit_rad_s",
            ),
        )
        if self.mit_kp < 0.0 or self.mit_kd < 0.0:
            raise ValueError("mit gains must be non-negative")


@dataclass(frozen=True)
class B601RSBackendConfig:
    """Explicit local configuration for the six-joint wired backend.

    ``position_limits_rad`` is optional because the vendor YAML does not own
    the URDF position limits.  A motion-capable caller should supply limits
    obtained from the same pinned URDF before invoking a command method.

    ``feedback_timeout_sec`` and ``watchdog_timeout_sec`` are retained as
    explicit configuration declarations for the later guarded control loop.
    This passive adapter does not infer freshness from MotorBridge's cached
    state and does not run an automatic watchdog or safe-stop thread.
    """

    channel: str = B601_RS_DEFAULT_CHANNEL
    rate_hz: float = B601_RS_DEFAULT_RATE_HZ
    joints: tuple[B601JointConfig, ...] = ()
    feedback_timeout_sec: float = B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC
    watchdog_timeout_sec: float = B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC
    allow_motion: bool = False
    position_limits_rad: tuple[tuple[float, float], ...] | None = None
    source_config: str | None = None

    def __post_init__(self) -> None:
        channel = str(self.channel).strip()
        if not channel:
            raise ValueError("channel must not be empty")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "rate_hz", _positive_float(self.rate_hz, name="rate_hz"))
        joints = tuple(self.joints)
        if any(not isinstance(joint, B601JointConfig) for joint in joints):
            raise TypeError("joints must contain B601JointConfig values")
        if tuple(joint.name for joint in joints) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError(
                "joints must contain joint1..joint6 in B601-RS arm order"
            )
        if len({joint.motor_id for joint in joints}) != 6:
            raise ValueError("B601-RS motor IDs must be unique")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(
            self,
            "feedback_timeout_sec",
            _positive_float(self.feedback_timeout_sec, name="feedback_timeout_sec"),
        )
        object.__setattr__(
            self,
            "watchdog_timeout_sec",
            _positive_float(self.watchdog_timeout_sec, name="watchdog_timeout_sec"),
        )
        if not isinstance(self.allow_motion, bool):
            raise TypeError("allow_motion must be a bool")
        if self.position_limits_rad is not None:
            try:
                limits = tuple(
                    tuple(float(value) for value in pair)
                    for pair in self.position_limits_rad
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "position_limits_rad must contain six (lower, upper) pairs"
                ) from exc
            if len(limits) != 6 or any(len(pair) != 2 for pair in limits):
                raise ValueError("position_limits_rad must contain six (lower, upper) pairs")
            for lower, upper in limits:
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError("position_limits_rad must contain finite increasing pairs")
            object.__setattr__(self, "position_limits_rad", limits)
        if self.source_config is not None:
            source = str(self.source_config).strip()
            object.__setattr__(self, "source_config", source or None)

    @classmethod
    def from_rs_yaml(
        cls,
        path: str | Path,
        *,
        allow_motion: bool = False,
        position_limits_rad: tuple[tuple[float, float], ...] | None = None,
    ) -> "B601RSBackendConfig":
        """Load the explicit RS YAML without importing or connecting hardware."""

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
        return cls.from_mapping(
            payload,
            allow_motion=allow_motion,
            position_limits_rad=position_limits_rad,
            source_config=str(config_path),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        allow_motion: bool = False,
        position_limits_rad: tuple[tuple[float, float], ...] | None = None,
        source_config: str | None = None,
    ) -> "B601RSBackendConfig":
        data = _mapping(payload, name="B601-RS config")
        groups = _mapping(data.get("groups"), name="groups")
        arm = _mapping(groups.get("arm"), name="groups.arm")
        raw_names = arm.get("joints")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            raise ValueError("groups.arm.joints must be a sequence")
        names = tuple(str(value) for value in raw_names)
        if names != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("groups.arm.joints must match joint1..joint6")
        raw_joints = data.get("joints")
        if not isinstance(raw_joints, Sequence) or isinstance(raw_joints, (str, bytes)):
            raise ValueError("joints must be a sequence")
        by_name: dict[str, Mapping[str, object]] = {}
        for raw_joint in raw_joints:
            joint = _mapping(raw_joint, name="joint")
            name = str(joint.get("name", "")).strip()
            if name in by_name:
                raise ValueError(f"duplicate B601-RS joint: {name}")
            by_name[name] = joint
        parsed: list[B601JointConfig] = []
        for name in B601_RS_ARM_JOINT_NAMES:
            joint = by_name.get(name)
            if joint is None:
                raise ValueError(f"missing B601-RS arm joint: {name}")
            mit = _mapping(joint.get("MIT", {}), name=f"{name}.MIT")
            pos_vel = _mapping(joint.get("POS_VEL", {}), name=f"{name}.POS_VEL")
            parsed.append(
                B601JointConfig(
                    name=name,
                    motor_id=_yaml_int(joint.get("motor_id"), name=f"{name}.motor_id"),
                    feedback_id=_yaml_int(
                        joint.get("feedback_id"), name=f"{name}.feedback_id"
                    ),
                    model=str(joint.get("model", "")).strip(),
                    vendor=str(joint.get("vendor", "robstride")),
                    mit_kp=_finite_float(mit.get("kp", 0.0), name=f"{name}.MIT.kp"),
                    mit_kd=_finite_float(mit.get("kd", 0.0), name=f"{name}.MIT.kd"),
                    position_velocity_limit_rad_s=_positive_float(
                        pos_vel.get("vlim"), name=f"{name}.POS_VEL.vlim"
                    ),
                )
            )
        return cls(
            channel=str(data.get("channel", B601_RS_DEFAULT_CHANNEL)),
            rate_hz=_positive_float(data.get("rate", B601_RS_DEFAULT_RATE_HZ), name="rate"),
            joints=tuple(parsed),
            allow_motion=allow_motion,
            position_limits_rad=position_limits_rad,
            source_config=source_config,
        )


@dataclass(frozen=True)
class B601JointState:
    """One poll observation from the vendor's cached ``MotorState``."""

    name: str
    motor_id: int
    feedback_id: int
    arbitration_id: int | None
    status_code: int | None
    position_rad: float | None
    velocity_reported: float | None
    torque_reported_nm: float | None
    observed_at_sec: float | None
    sample_sequence: int
    valid: bool
    stale: bool
    error: str | None = None
    frame_id: str = B601_RS_FEEDBACK_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "motor_id", _yaml_int(self.motor_id, name="motor_id"))
        object.__setattr__(self, "feedback_id", _yaml_int(self.feedback_id, name="feedback_id"))
        if self.arbitration_id is not None:
            object.__setattr__(
                self,
                "arbitration_id",
                _yaml_int(self.arbitration_id, name="arbitration_id"),
            )
        if self.status_code is not None:
            object.__setattr__(self, "status_code", int(self.status_code))
        for field_name in (
            "position_rad",
            "velocity_reported",
            "torque_reported_nm",
            "observed_at_sec",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_float(value, name=field_name))
        if isinstance(self.sample_sequence, bool) or int(self.sample_sequence) < 0:
            raise ValueError("sample_sequence must be a non-negative integer")
        object.__setattr__(self, "sample_sequence", int(self.sample_sequence))
        if not isinstance(self.valid, bool) or not isinstance(self.stale, bool):
            raise TypeError("valid and stale must be bools")
        object.__setattr__(self, "error", None if self.error is None else str(self.error))


@dataclass(frozen=True)
class B601HardwareState:
    """Complete arm snapshot plus explicit freshness semantics."""

    sequence: int
    receive_timestamp_sec: float
    joints: tuple[B601JointState, ...]
    valid: bool
    stale: bool
    freshness_verified: bool = False
    freshness_semantics: str = "poll_observation_unverified"
    velocity_semantics: str = "vendor_reported_unverified"
    fault_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or int(self.sequence) < 0:
            raise ValueError("sequence must be a non-negative integer")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(
            self,
            "receive_timestamp_sec",
            _finite_float(self.receive_timestamp_sec, name="receive_timestamp_sec"),
        )
        joints = tuple(self.joints)
        if any(not isinstance(joint, B601JointState) for joint in joints):
            raise TypeError("joints must contain B601JointState values")
        if tuple(joint.name for joint in joints) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("joints must match the B601-RS arm order")
        object.__setattr__(self, "joints", joints)
        if not isinstance(self.valid, bool) or not isinstance(self.stale, bool):
            raise TypeError("valid and stale must be bools")
        if not isinstance(self.freshness_verified, bool):
            raise TypeError("freshness_verified must be a bool")
        object.__setattr__(self, "freshness_semantics", str(self.freshness_semantics))
        object.__setattr__(self, "velocity_semantics", str(self.velocity_semantics))
        object.__setattr__(
            self,
            "fault_reason",
            None if self.fault_reason is None else str(self.fault_reason),
        )

    @property
    def positions_rad(self) -> tuple[float, ...] | None:
        if not self.valid or any(joint.position_rad is None for joint in self.joints):
            return None
        return tuple(float(joint.position_rad) for joint in self.joints)

    @property
    def velocities_reported(self) -> tuple[float, ...] | None:
        if not self.valid or any(joint.velocity_reported is None for joint in self.joints):
            return None
        return tuple(float(joint.velocity_reported) for joint in self.joints)

    @property
    def torques_reported_nm(self) -> tuple[float, ...] | None:
        if not self.valid or any(joint.torque_reported_nm is None for joint in self.joints):
            return None
        return tuple(float(joint.torque_reported_nm) for joint in self.joints)

    def joint(self, name: str) -> B601JointState:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)


@dataclass(frozen=True)
class B601CommandReceipt:
    """Per-call dispatch result; partial vendor sends are never hidden."""

    sequence: int
    mode: str
    frame_id: str
    requested_at_sec: float
    completed_at_sec: float
    requested_joint_names: tuple[str, ...]
    sent_joint_names: tuple[str, ...]
    failed_joint_names: tuple[str, ...]
    errors: tuple[str, ...]
    accepted: bool
    source_sequence: int | None = None


@dataclass(frozen=True)
class B601ModeReceipt:
    """Result of an explicit one-time mode setup or enable/disable call."""

    operation: str
    requested_joint_names: tuple[str, ...]
    completed_joint_names: tuple[str, ...]
    failed_joint_names: tuple[str, ...]
    errors: tuple[str, ...]
    completed: bool


class B601RSBackend:
    """Thin wired backend with a fail-closed, opt-in command boundary."""

    def __init__(
        self,
        config: B601RSBackendConfig,
        *,
        controller_factory: Callable[[str], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, B601RSBackendConfig):
            raise TypeError("config must be a B601RSBackendConfig")
        self.config = config
        self._controller_factory = controller_factory
        self._clock = clock
        self._controller: object | None = None
        self._motors: dict[str, object] = {}
        self._state_sequence = 0
        self._command_sequence = 0
        self._last_state: B601HardwareState | None = None
        self._last_command: B601CommandReceipt | None = None
        self._last_fault_reason: str | None = None

    @property
    def connected(self) -> bool:
        return self._controller is not None

    @property
    def registered_joint_names(self) -> tuple[str, ...]:
        return tuple(self._motors)

    @property
    def last_state(self) -> B601HardwareState | None:
        return self._last_state

    @property
    def last_command_receipt(self) -> B601CommandReceipt | None:
        return self._last_command

    @property
    def last_fault_reason(self) -> str | None:
        return self._last_fault_reason

    def connect(self) -> None:
        """Open CAN and register the six motors, without sending commands."""

        if self.connected:
            return
        factory = self._controller_factory or self._default_controller_factory
        try:
            controller = factory(self.config.channel)
            add_motor = getattr(controller, "add_robstride_motor")
            motors = {
                joint.name: add_motor(joint.motor_id, joint.feedback_id, joint.model)
                for joint in self.config.joints
            }
        except ModuleNotFoundError as exc:
            raise B601HardwareDependencyError(
                "motorbridge is required only when connecting the B601 backend"
            ) from exc
        except Exception:
            try:
                close = getattr(locals().get("controller"), "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise
        self._controller = controller
        self._motors = motors

    @staticmethod
    def _default_controller_factory(channel: str) -> object:
        try:
            module = import_module("motorbridge")
            controller_type = getattr(module, "Controller")
        except (ModuleNotFoundError, AttributeError) as exc:
            raise B601HardwareDependencyError(
                "motorbridge.Controller is unavailable"
            ) from exc
        return controller_type(channel)

    def close(self) -> None:
        """Release the vendor handle without implicit disable or command I/O."""

        controller = self._controller
        self._controller = None
        self._motors = {}
        if controller is None:
            return
        close = getattr(controller, "close", None)
        if callable(close):
            close()

    def _require_connected(self) -> object:
        if self._controller is None:
            raise B601HardwareError("B601 backend is not connected")
        return self._controller

    def _require_motion_authorized(self) -> None:
        if not self.config.allow_motion:
            raise B601MotionAuthorizationError(
                "motion command rejected: construct the backend with allow_motion=True"
            )

    def _resolve_joint_names(self, joint_names: Sequence[str] | None) -> tuple[str, ...]:
        resolved = B601_RS_ARM_JOINT_NAMES if joint_names is None else tuple(str(name) for name in joint_names)
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("joint_names must be a non-empty unique sequence")
        unknown = set(resolved).difference(B601_RS_ARM_JOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown B601-RS joint names: {sorted(unknown)}")
        if any(name not in self._motors for name in resolved):
            raise B601HardwareError("requested joint is not registered; call connect first")
        return resolved

    def _require_position_limits(self) -> tuple[tuple[float, float], ...]:
        limits = self.config.position_limits_rad
        if limits is None:
            raise B601HardwareError(
                "motion command requires position_limits_rad from the pinned URDF"
            )
        return limits

    def poll_feedback(self) -> B601HardwareState:
        """Consume queued feedback only; never issue a feedback request frame."""

        controller = self._require_connected()
        now = _finite_float(self._clock(), name="clock")
        self._state_sequence += 1
        poll_error: str | None = None
        try:
            poll = getattr(controller, "poll_feedback_once")
            poll()
        except Exception as exc:  # vendor errors become an invalid snapshot
            poll_error = f"poll_feedback_once: {exc}"

        observations: list[B601JointState] = []
        errors: list[str] = [poll_error] if poll_error else []
        for joint in self.config.joints:
            motor = self._motors[joint.name]
            try:
                raw_state = getattr(motor, "get_state")()
            except Exception as exc:
                raw_state = None
                error = f"{joint.name}.get_state: {exc}"
                errors.append(error)
                observations.append(
                    B601JointState(
                        name=joint.name,
                        motor_id=joint.motor_id,
                        feedback_id=joint.feedback_id,
                        arbitration_id=None,
                        status_code=None,
                        position_rad=None,
                        velocity_reported=None,
                        torque_reported_nm=None,
                        observed_at_sec=now,
                        sample_sequence=self._state_sequence,
                        valid=False,
                        stale=True,
                        error=error,
                    )
                )
                continue
            if raw_state is None:
                error = f"{joint.name}: feedback cache is empty"
                errors.append(error)
                observations.append(
                    B601JointState(
                        name=joint.name,
                        motor_id=joint.motor_id,
                        feedback_id=joint.feedback_id,
                        arbitration_id=None,
                        status_code=None,
                        position_rad=None,
                        velocity_reported=None,
                        torque_reported_nm=None,
                        observed_at_sec=now,
                        sample_sequence=self._state_sequence,
                        valid=False,
                        stale=True,
                        error=error,
                    )
                )
                continue
            try:
                can_id = int(getattr(raw_state, "can_id"))
                arbitration_id = int(getattr(raw_state, "arbitration_id"))
                status_code = int(getattr(raw_state, "status_code"))
                position = _finite_float(getattr(raw_state, "pos"), name=f"{joint.name}.pos")
                velocity = _finite_float(getattr(raw_state, "vel"), name=f"{joint.name}.vel")
                torque = _finite_float(getattr(raw_state, "torq"), name=f"{joint.name}.torq")
                state_error: str | None = None
                if can_id != joint.motor_id:
                    state_error = f"{joint.name}: feedback can_id={can_id} expected {joint.motor_id}"
                elif status_code != 0:
                    state_error = f"{joint.name}: vendor status_code={status_code}"
                if state_error:
                    errors.append(state_error)
                observations.append(
                    B601JointState(
                        name=joint.name,
                        motor_id=joint.motor_id,
                        feedback_id=joint.feedback_id,
                        arbitration_id=arbitration_id,
                        status_code=status_code,
                        position_rad=position,
                        velocity_reported=velocity,
                        torque_reported_nm=torque,
                        observed_at_sec=now,
                        sample_sequence=self._state_sequence,
                        valid=state_error is None and poll_error is None,
                        stale=state_error is not None or poll_error is not None,
                        error=state_error or poll_error,
                    )
                )
            except (AttributeError, TypeError, ValueError) as exc:
                error = f"{joint.name}: malformed feedback: {exc}"
                errors.append(error)
                observations.append(
                    B601JointState(
                        name=joint.name,
                        motor_id=joint.motor_id,
                        feedback_id=joint.feedback_id,
                        arbitration_id=None,
                        status_code=None,
                        position_rad=None,
                        velocity_reported=None,
                        torque_reported_nm=None,
                        observed_at_sec=now,
                        sample_sequence=self._state_sequence,
                        valid=False,
                        stale=True,
                        error=error,
                    )
                )
        valid = not errors and all(joint.valid for joint in observations)
        state = B601HardwareState(
            sequence=self._state_sequence,
            receive_timestamp_sec=now,
            joints=tuple(observations),
            valid=valid,
            stale=not valid,
            freshness_verified=False,
            freshness_semantics="poll_observation_unverified",
            velocity_semantics="vendor_reported_unverified",
            fault_reason="; ".join(errors) if errors else None,
        )
        self._last_state = state
        self._last_fault_reason = state.fault_reason
        return state

    def _receipt(
        self,
        *,
        mode: str,
        frame_id: str,
        started: float,
        requested: tuple[str, ...],
        sent: list[str],
        failed: list[str],
        errors: list[str],
        source_sequence: int | None = None,
    ) -> B601CommandReceipt:
        self._command_sequence += 1
        completed = _finite_float(self._clock(), name="clock")
        receipt = B601CommandReceipt(
            sequence=self._command_sequence,
            mode=mode,
            frame_id=frame_id,
            requested_at_sec=started,
            completed_at_sec=completed,
            requested_joint_names=requested,
            sent_joint_names=tuple(sent),
            failed_joint_names=tuple(failed),
            errors=tuple(errors),
            accepted=not failed and not errors and len(sent) == len(requested),
            source_sequence=source_sequence,
        )
        self._last_command = receipt
        if not receipt.accepted:
            self._last_fault_reason = "; ".join(receipt.errors) or "partial command dispatch"
        return receipt

    def prepare_position_velocity_mode(
        self,
        *,
        joint_names: Sequence[str] | None = None,
        timeout_ms: int = 1000,
    ) -> B601ModeReceipt:
        """Explicitly set POS_VEL once; never call this from a high-rate loop."""

        self._require_motion_authorized()
        names = self._resolve_joint_names(joint_names)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms <= 0
        ):
            raise ValueError("timeout_ms must be a positive integer")
        try:
            mode = getattr(import_module("motorbridge"), "Mode").POS_VEL
        except (ModuleNotFoundError, AttributeError) as exc:
            raise B601HardwareDependencyError("motorbridge.Mode.POS_VEL is unavailable") from exc
        completed: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for name in names:
            try:
                getattr(self._motors[name], "ensure_mode")(mode, int(timeout_ms))
                completed.append(name)
            except Exception as exc:
                failed.append(name)
                errors.append(f"{name}: {exc}")
        return B601ModeReceipt("prepare_position_velocity_mode", names, tuple(completed), tuple(failed), tuple(errors), not failed)

    def enable(self, *, joint_names: Sequence[str] | None = None) -> B601ModeReceipt:
        self._require_motion_authorized()
        names = self._resolve_joint_names(joint_names)
        return self._run_motor_operation("enable", names, "enable")

    def disable(self, *, joint_names: Sequence[str] | None = None) -> B601ModeReceipt:
        self._require_motion_authorized()
        names = self._resolve_joint_names(joint_names)
        return self._run_motor_operation("disable", names, "disable")

    def safe_stop(self, *, joint_names: Sequence[str] | None = None) -> B601ModeReceipt:
        """Request vendor disable without requiring motion-start authorization.

        This is intentionally separate from :meth:`close` and from normal
        ``disable`` admission.  A caller that was not authorized to start
        motion must still be able to issue a reviewed stop request.  Mechanical
        support is required first: disabling a loaded arm can allow it to fall,
        so this is not a substitute for a physical emergency stop.
        """

        names = self._resolve_joint_names(joint_names)
        return self._run_motor_operation("safe_stop", names, "disable")

    def _run_motor_operation(
        self,
        operation: str,
        names: tuple[str, ...],
        method_name: str,
    ) -> B601ModeReceipt:
        completed: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for name in names:
            try:
                getattr(self._motors[name], method_name)()
                completed.append(name)
            except Exception as exc:
                failed.append(name)
                errors.append(f"{name}: {exc}")
        return B601ModeReceipt(operation, names, tuple(completed), tuple(failed), tuple(errors), not failed)

    def send_position_velocity(
        self,
        qpos_rad: Sequence[float],
        *,
        velocity_limits_rad_s: Sequence[float] | None = None,
        joint_names: Sequence[str] | None = None,
        frame_id: str = B601_RS_POSITION_FRAME,
        source_sequence: int | None = None,
    ) -> B601CommandReceipt:
        self._require_motion_authorized()
        if frame_id != B601_RS_POSITION_FRAME:
            raise ValueError("position command frame_id is not the B601 position frame")
        names = self._resolve_joint_names(joint_names)
        self._require_position_limits()
        qpos = _six_finite(qpos_rad, name="qpos_rad")
        if self.config.position_limits_rad is not None:
            for index, (lower, upper) in enumerate(self.config.position_limits_rad):
                if not lower <= qpos[index] <= upper:
                    raise ValueError(f"qpos_rad[{index}] is outside the configured position limit")
        if velocity_limits_rad_s is None:
            velocity_limits = tuple(
                joint.position_velocity_limit_rad_s for joint in self.config.joints
            )
        else:
            velocity_limits = _six_nonnegative(
                velocity_limits_rad_s,
                name="velocity_limits_rad_s",
            )
        configured_velocity_limits = tuple(
            joint.position_velocity_limit_rad_s for joint in self.config.joints
        )
        if any(
            velocity_limits[index] > configured_velocity_limits[index]
            for index in range(6)
        ):
            raise ValueError("velocity_limits_rad_s exceeds the configured per-joint limit")
        started = _finite_float(self._clock(), name="clock")
        sent: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for index, name in enumerate(B601_RS_ARM_JOINT_NAMES):
            if name not in names:
                continue
            try:
                getattr(self._motors[name], "send_pos_vel")(
                    float(qpos[index]), float(velocity_limits[index])
                )
                sent.append(name)
            except Exception as exc:
                failed.append(name)
                errors.append(f"{name}: {exc}")
        return self._receipt(
            mode="position_velocity",
            frame_id=frame_id,
            started=started,
            requested=names,
            sent=sent,
            failed=failed,
            errors=errors,
            source_sequence=source_sequence,
        )

    def send_mit(
        self,
        reference: B601MitReference,
        *,
        kp: Sequence[float],
        kd: Sequence[float],
        tau_nm: Sequence[float],
        torque_limits_nm: Sequence[float],
        joint_names: Sequence[str] | None = None,
    ) -> B601CommandReceipt:
        self._require_motion_authorized()
        if not isinstance(reference, B601MitReference):
            raise TypeError("reference must be a B601MitReference")
        if reference.frame_id != B601_RS_JOINT_MIT_FRAME:
            raise ValueError("MIT reference frame does not match B601 joint convention")
        if not reference.valid:
            raise ValueError("invalid MIT reference cannot be dispatched")
        names = self._resolve_joint_names(joint_names)
        position_limits = self._require_position_limits()
        for index, (lower, upper) in enumerate(position_limits):
            if not lower <= reference.qpos_rad[index] <= upper:
                raise ValueError(
                    f"reference.qpos_rad[{index}] is outside the configured position limit"
                )
        kp_values = _six_nonnegative(kp, name="kp")
        kd_values = _six_nonnegative(kd, name="kd")
        tau_values = _six_finite(tau_nm, name="tau_nm")
        limits = _six_nonnegative(torque_limits_nm, name="torque_limits_nm")
        configured_velocity_limits = tuple(
            joint.position_velocity_limit_rad_s for joint in self.config.joints
        )
        if any(
            abs(reference.qvel_rad_s[index]) > configured_velocity_limits[index]
            for index in range(6)
        ):
            raise ValueError("reference.qvel_rad_s exceeds the configured per-joint limit")
        if any(abs(tau_values[index]) > limits[index] for index in range(6)):
            raise ValueError("tau_nm exceeds the supplied final torque limits")
        started = _finite_float(self._clock(), name="clock")
        sent: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for index, name in enumerate(B601_RS_ARM_JOINT_NAMES):
            if name not in names:
                continue
            try:
                getattr(self._motors[name], "send_mit")(
                    float(reference.qpos_rad[index]),
                    float(reference.qvel_rad_s[index]),
                    float(kp_values[index]),
                    float(kd_values[index]),
                    float(tau_values[index]),
                )
                sent.append(name)
            except Exception as exc:
                failed.append(name)
                errors.append(f"{name}: {exc}")
        return self._receipt(
            mode="mit",
            frame_id=reference.frame_id,
            started=started,
            requested=names,
            sent=sent,
            failed=failed,
            errors=errors,
            source_sequence=reference.sequence,
        )


__all__ = [
    "B601CommandReceipt",
    "B601HardwareError",
    "B601HardwareDependencyError",
    "B601HardwareState",
    "B601JointConfig",
    "B601JointState",
    "B601ModeReceipt",
    "B601MotionAuthorizationError",
    "B601RSBackend",
    "B601RSBackendConfig",
    "B601_RS_DEFAULT_CHANNEL",
    "B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC",
    "B601_RS_DEFAULT_RATE_HZ",
    "B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC",
    "B601_RS_FEEDBACK_FRAME",
    "B601_RS_POSITION_FRAME",
]
