"""Wired RobStride backend boundary for the reBotArm B601-RS.

This module is deliberately a thin adapter around the pinned
``reBotArm_control_py``/``motorbridge`` stack.  It owns neither kinematics nor
the haptic law; those remain in :mod:`robot_motion.rebot_b601_rs`.  Importing
the module is side-effect free.  A controller is opened only by an explicit
call to :meth:`B601RSBackend.connect`, and every motor-control command path is
disabled unless the caller constructed the backend with ``allow_motion=True``.
Active-report configuration has a separate, explicit authorization and never
enables a motor or sends a motion reference.

The default feedback path intentionally uses the vendor cache as a *poll
observation*.  An active caller may opt in to an explicit vendor feedback
request before polling; this is needed by the commissioning loop because a
RobStride bus does not necessarily populate the cache until a request frame is
sent.  That compatibility path also reads ``mechPos`` (0x7019) through the
vendor parameter API.  A motion-authorized
backend additionally requires MotorBridge to advertise strict RobStride
parameter-reply correlation.  This rejects failed or mismatched type-17
replies and quarantines replies that arrive without a matching pending request.
The wire protocol has no transaction token, so this capability cannot prove
freshness between otherwise identical consecutive requests by itself.
Read-only diagnostics remain available without that capability, while every
motion-start path fails closed.  The position-only fallback is explicit in
each observation; velocity/torque consumers must still require their
corresponding fields.

Newer MotorBridge runtimes may separately advertise
``state_observation_metadata_v1``.  An explicit
``require_host_freshness=True`` poll then consumes only decoded RobStride
operation-status state carrying a per-motor receive sequence and monotonic
receive age.  The legacy path remains unchanged.  Host receive freshness is
reported separately from device-source freshness because RobStride still does
not provide a device sampling timestamp through this stack.
"""

from __future__ import annotations

import math
import struct
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
B601_RS_TORQUE_LIMIT_PARAM_ID = 0x700B
B601_RS_TORQUE_LIMIT_MAX_NM = 60.0
B601_RS_TORQUE_LIMIT_VERIFY_ABS_TOL_NM = 1e-4
B601_RS_MECH_POS_PARAM_ID = 0x7019
_B601_RS_PARAMETER_REPLY_CORRELATION_CAPABILITY = (
    "parameter_reply_correlation_v1"
)
_B601_RS_STATE_OBSERVATION_CAPABILITY = "state_observation_metadata_v1"
_B601_RS_ACTIVE_REPORT_OBSERVATION_CAPABILITY = "active_report_observation_v1"
# RobStride POS_VEL parameter IDs used by the pinned reBotArm
# ``JointGroup._write_pv_params`` implementation.  Keep these explicit so a
# future vendor/runtime update cannot silently turn mode preparation into a
# different register sequence.
B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID = 0x7017
B601_RS_POS_VEL_VEL_KP_PARAM_ID = 0x701F
B601_RS_POS_VEL_VEL_KI_PARAM_ID = 0x7020
B601_RS_POS_VEL_POS_KP_PARAM_ID = 0x701E


class B601HardwareError(RuntimeError):
    """Base error for the explicit wired backend boundary."""


class B601HardwareDependencyError(B601HardwareError):
    """The optional vendor runtime is unavailable."""


class B601MotionAuthorizationError(B601HardwareError):
    """A command was attempted without an explicit motion authorization."""


class B601ActiveReportAuthorizationError(B601HardwareError):
    """Active telemetry was attempted without its separate authorization."""


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


def _float32_bits(value: float) -> int | None:
    """Return the wire-level IEEE-754 single-precision bits when representable."""

    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        return None


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


def _motorbridge_runtime_evidence(source: object) -> dict[str, object]:
    """Inspect MotorBridge ABI capabilities without trusting its version alone."""

    provider = getattr(source, "abi_capabilities", None)
    if not callable(provider):
        provider = getattr(source, "motorbridge_abi_capabilities", None)
    if not callable(provider):
        return {
            "verified": False,
            "abi_version": None,
            "robstride_features": (),
            "error": "MotorBridge ABI capability metadata is unavailable",
        }
    try:
        capabilities = provider()
    except Exception as exc:  # noqa: BLE001 - vendor metadata is an admission boundary
        return {
            "verified": False,
            "abi_version": None,
            "robstride_features": (),
            "error": f"MotorBridge ABI capability query failed: {exc}",
        }
    if not isinstance(capabilities, Mapping):
        return {
            "verified": False,
            "abi_version": None,
            "robstride_features": (),
            "error": "MotorBridge ABI capabilities must be a mapping",
        }

    abi = capabilities.get("abi")
    abi_version = abi.get("version") if isinstance(abi, Mapping) else None
    features = capabilities.get("features")
    robstride = features.get("robstride") if isinstance(features, Mapping) else None
    if isinstance(robstride, Sequence) and not isinstance(robstride, (str, bytes)):
        robstride_features = tuple(
            str(feature) for feature in robstride if isinstance(feature, str)
        )
    else:
        robstride_features = ()
    verified = (
        _B601_RS_PARAMETER_REPLY_CORRELATION_CAPABILITY in robstride_features
    )
    return {
        "verified": verified,
        "abi_version": None if abi_version is None else str(abi_version),
        "robstride_features": robstride_features,
        "error": None
        if verified
        else (
            "MotorBridge does not advertise required RobStride capability "
            f"{_B601_RS_PARAMETER_REPLY_CORRELATION_CAPABILITY!r}"
        ),
    }


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
    position_velocity_kp: float = 0.0
    position_velocity_ki: float = 0.0
    position_kp: float = 0.0
    position_ki: float = 0.0

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
        for field_name in (
            "position_velocity_kp",
            "position_velocity_ki",
            "position_kp",
            "position_ki",
        ):
            value = _finite_float(getattr(self, field_name), name=field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        if self.mit_kp < 0.0 or self.mit_kd < 0.0:
            raise ValueError("mit gains must be non-negative")


@dataclass(frozen=True)
class B601RSBackendConfig:
    """Explicit local configuration for the six-joint wired backend.

    ``position_limits_rad`` is optional because the vendor YAML does not own
    the URDF position limits.  A motion-capable caller should supply limits
    obtained from the same pinned URDF before invoking a command method.

    ``feedback_position_limits_rad`` is a separate, read-side admission
    envelope.  It may be slightly wider than the command envelope when a
    measured encoder boundary compatibility window has been independently
    documented (for example the B601 J2/J3/J4 zero/nominal-boundary
    observations).  It never relaxes an outgoing reference check.  When it is
    omitted, the command position limits are also used for feedback admission
    when available.

    ``feedback_timeout_sec`` bounds the active RobStride ``mechPos`` parameter
    read, while both timeout values remain explicit configuration declarations
    for the later guarded control loop.  This adapter does not infer freshness
    from MotorBridge's cached state and does not run an automatic watchdog or
    safe-stop thread.
    """

    channel: str = B601_RS_DEFAULT_CHANNEL
    rate_hz: float = B601_RS_DEFAULT_RATE_HZ
    joints: tuple[B601JointConfig, ...] = ()
    feedback_timeout_sec: float = B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC
    watchdog_timeout_sec: float = B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC
    allow_motion: bool = False
    allow_active_reporting: bool = False
    position_limits_rad: tuple[tuple[float, float], ...] | None = None
    source_config: str | None = None
    feedback_position_limits_rad: tuple[tuple[float, float], ...] | None = None

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
        if not isinstance(self.allow_active_reporting, bool):
            raise TypeError("allow_active_reporting must be a bool")
        for field_name in ("position_limits_rad", "feedback_position_limits_rad"):
            raw_limits = getattr(self, field_name)
            if raw_limits is None:
                continue
            try:
                limits = tuple(
                    tuple(float(value) for value in pair)
                    for pair in raw_limits
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} must contain six (lower, upper) pairs"
                ) from exc
            if len(limits) != 6 or any(len(pair) != 2 for pair in limits):
                raise ValueError(
                    f"{field_name} must contain six (lower, upper) pairs"
                )
            for lower, upper in limits:
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError(
                        f"{field_name} must contain finite increasing pairs"
                    )
            object.__setattr__(self, field_name, limits)
        if self.source_config is not None:
            source = str(self.source_config).strip()
            object.__setattr__(self, "source_config", source or None)

    @classmethod
    def from_rs_yaml(
        cls,
        path: str | Path,
        *,
        allow_motion: bool = False,
        allow_active_reporting: bool = False,
        position_limits_rad: tuple[tuple[float, float], ...] | None = None,
        feedback_position_limits_rad: tuple[tuple[float, float], ...] | None = None,
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
            allow_active_reporting=allow_active_reporting,
            position_limits_rad=position_limits_rad,
            feedback_position_limits_rad=feedback_position_limits_rad,
            source_config=str(config_path),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        allow_motion: bool = False,
        allow_active_reporting: bool = False,
        position_limits_rad: tuple[tuple[float, float], ...] | None = None,
        feedback_position_limits_rad: tuple[tuple[float, float], ...] | None = None,
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
                    position_velocity_kp=_finite_float(
                        pos_vel.get("vel_kp", 0.0),
                        name=f"{name}.POS_VEL.vel_kp",
                    ),
                    position_velocity_ki=_finite_float(
                        pos_vel.get("vel_ki", 0.0),
                        name=f"{name}.POS_VEL.vel_ki",
                    ),
                    position_kp=_finite_float(
                        pos_vel.get("pos_kp", 0.0),
                        name=f"{name}.POS_VEL.pos_kp",
                    ),
                    position_ki=_finite_float(
                        pos_vel.get("pos_ki", 0.0),
                        name=f"{name}.POS_VEL.pos_ki",
                    ),
                )
            )
        return cls(
            channel=str(data.get("channel", B601_RS_DEFAULT_CHANNEL)),
            rate_hz=_positive_float(data.get("rate", B601_RS_DEFAULT_RATE_HZ), name="rate"),
            joints=tuple(parsed),
            allow_motion=allow_motion,
            allow_active_reporting=allow_active_reporting,
            position_limits_rad=position_limits_rad,
            feedback_position_limits_rad=feedback_position_limits_rad,
            source_config=source_config,
        )


@dataclass(frozen=True)
class B601JointState:
    """One poll observation from cache or an explicit RobStride position read."""

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
    position_source: str = "vendor_motor_state_cache"
    status_sequence: int | None = None
    receive_age_sec: float | None = None
    host_receive_freshness_verified: bool = False

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
        object.__setattr__(self, "position_source", str(self.position_source))
        if self.status_sequence is not None and (
            isinstance(self.status_sequence, bool)
            or not isinstance(self.status_sequence, int)
            or self.status_sequence <= 0
        ):
            raise ValueError("status_sequence must be a positive integer")
        if self.receive_age_sec is not None:
            age = _finite_float(self.receive_age_sec, name="receive_age_sec")
            if age < 0.0:
                raise ValueError("receive_age_sec must be non-negative")
            object.__setattr__(self, "receive_age_sec", age)
        if not isinstance(self.host_receive_freshness_verified, bool):
            raise TypeError("host_receive_freshness_verified must be a bool")


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
    host_receive_freshness_verified: bool = False

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
        if not isinstance(self.host_receive_freshness_verified, bool):
            raise TypeError("host_receive_freshness_verified must be a bool")
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


@dataclass(frozen=True)
class B601DriverTorqueLimitReceipt:
    """Auditable apply/restore result for temporary RobStride torque limits."""

    operation: str
    requested_joint_names: tuple[str, ...]
    target_limits_nm: tuple[float, ...]
    prior_limits_nm: tuple[float | None, ...]
    verified_limits_nm: tuple[float | None, ...]
    applied_joint_names: tuple[str, ...]
    restored_joint_names: tuple[str, ...]
    failed_joint_names: tuple[str, ...]
    errors: tuple[str, ...]
    completed: bool
    rollback_attempted: bool = False
    rollback_completed: bool | None = None


@dataclass(frozen=True)
class B601ActiveReportReceipt:
    """Auditable result for a RobStride active-report state transition."""

    operation: str
    requested_joint_names: tuple[str, ...]
    completed_joint_names: tuple[str, ...]
    failed_joint_names: tuple[str, ...]
    errors: tuple[str, ...]
    completed: bool
    rollback_attempted: bool = False
    rollback_completed: bool | None = None


class B601RSBackend:
    """Thin wired backend with a fail-closed, opt-in command boundary."""

    def __init__(
        self,
        config: B601RSBackendConfig,
        *,
        controller_factory: Callable[[str], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, B601RSBackendConfig):
            raise TypeError("config must be a B601RSBackendConfig")
        self.config = config
        self._controller_factory = controller_factory
        self._clock = clock
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self._sleeper = sleeper
        self._controller: object | None = None
        self._motors: dict[str, object] = {}
        self._state_sequence = 0
        self._command_sequence = 0
        self._last_state: B601HardwareState | None = None
        self._last_command: B601CommandReceipt | None = None
        self._last_fault_reason: str | None = None
        self._enabled_joint_names: set[str] = set()
        self._disabled_joint_names: set[str] = set()
        self._driver_torque_limit_original_nm: dict[str, float] = {}
        self._driver_torque_limit_applied_nm: dict[str, float | None] = {}
        self._active_report_joint_names: set[str] = set()
        self._motorbridge_runtime_evidence: dict[str, object] | None = None

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

    @property
    def enabled_joint_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in B601_RS_ARM_JOINT_NAMES if name in self._enabled_joint_names
        )

    @property
    def confirmed_disabled_joint_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in B601_RS_ARM_JOINT_NAMES if name in self._disabled_joint_names
        )

    @property
    def temporary_driver_torque_limit_joint_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in B601_RS_ARM_JOINT_NAMES
            if name in self._driver_torque_limit_original_nm
        )

    @property
    def active_report_joint_names(self) -> tuple[str, ...]:
        """Joints whose active-report stream may still be enabled."""

        return tuple(
            name
            for name in B601_RS_ARM_JOINT_NAMES
            if name in self._active_report_joint_names
        )

    @property
    def motorbridge_runtime_evidence(self) -> Mapping[str, object] | None:
        """Return immutable-value evidence used by the motion admission gate."""

        if self._motorbridge_runtime_evidence is None:
            return None
        return dict(self._motorbridge_runtime_evidence)

    @property
    def parameter_reply_correlation_verified(self) -> bool:
        """Whether the connected runtime proves strict type-17 correlation."""

        return bool(
            self._motorbridge_runtime_evidence
            and self._motorbridge_runtime_evidence.get("verified") is True
        )

    @property
    def state_observation_metadata_verified(self) -> bool:
        """Whether the runtime exposes RobStride host receive metadata."""

        features = (
            self._motorbridge_runtime_evidence.get("robstride_features", ())
            if self._motorbridge_runtime_evidence
            else ()
        )
        return _B601_RS_STATE_OBSERVATION_CAPABILITY in features

    def connect(self) -> None:
        """Open CAN and register the six motors, without sending commands."""

        if self.connected:
            return
        if (
            self._enabled_joint_names
            or self._active_report_joint_names
            or self._driver_torque_limit_original_nm
        ):
            raise B601HardwareError(
                "cannot reconnect a released backend with unconfirmed device state"
            )
        try:
            if self._controller_factory is None:
                module = self._load_motorbridge_module()
                runtime_evidence = _motorbridge_runtime_evidence(module)
                if self.config.allow_motion:
                    self._require_verified_parameter_reply_correlation(
                        runtime_evidence
                    )
                controller_type = module.Controller
                controller = controller_type(self.config.channel)
            else:
                controller = self._controller_factory(self.config.channel)
                runtime_evidence = _motorbridge_runtime_evidence(controller)
                if self.config.allow_motion:
                    self._require_verified_parameter_reply_correlation(
                        runtime_evidence
                    )
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
        self._motorbridge_runtime_evidence = runtime_evidence

    @staticmethod
    def _load_motorbridge_module() -> object:
        try:
            module = import_module("motorbridge")
            controller_type = module.Controller
        except (ModuleNotFoundError, AttributeError) as exc:
            raise B601HardwareDependencyError(
                "motorbridge.Controller is unavailable"
            ) from exc
        if not callable(controller_type):
            raise B601HardwareDependencyError(
                "motorbridge.Controller is not callable"
            )
        return module

    @staticmethod
    def _require_verified_parameter_reply_correlation(
        evidence: Mapping[str, object],
    ) -> None:
        if evidence.get("verified") is True:
            return
        version = evidence.get("abi_version")
        detail = evidence.get("error") or "capability verification failed"
        raise B601HardwareDependencyError(
            "motion-capable B601 backend requires a MotorBridge runtime with "
            "strict RobStride type-17 parameter-reply correlation; "
            f"abi_version={version!r}: {detail}"
        )

    def close(self) -> None:
        """Release the vendor handle without implicit disable or command I/O."""

        if self._active_report_joint_names:
            raise B601HardwareError(
                "active reporting must be disabled before close: "
                + ", ".join(self.active_report_joint_names)
            )
        if self._driver_torque_limit_original_nm:
            raise B601HardwareError(
                "temporary driver torque limits must be restored before close"
            )

        self.release_transport()
        self._enabled_joint_names.clear()
        self._disabled_joint_names.clear()

    def release_transport(self) -> None:
        """Release host resources at terminal shutdown, without device I/O.

        Unlike normal close, this may follow failed device cleanup. Keep all
        unresolved device-state bookkeeping; releasing a handle confirms
        neither disable, active-report withdrawal nor torque-limit restoration.
        An instance with unresolved state must not be reconnected or reused.
        """

        controller = self._controller
        if controller is None:
            return
        close_bus = getattr(controller, "close_bus", None)
        if callable(close_bus) and close_bus() is False:
            raise B601HardwareError("vendor bus transport release is incomplete")
        # MotorBridge's motor handles retain native Arc references. Dropping
        # only their Python wrappers does not free those native handles.
        handle_errors = []
        for name, motor in tuple(self._motors.items()):
            close_motor = getattr(motor, "close", None)
            if not callable(close_motor):
                continue
            try:
                if close_motor() is False:
                    raise B601HardwareError("motor handle release is incomplete")
            except Exception as exc:  # noqa: BLE001 - retain remaining native handles for retry
                handle_errors.append(f"{name}: {exc}")
            else:
                self._motors.pop(name)
        if handle_errors:
            raise B601HardwareError("motor host handle release failed: " + "; ".join(handle_errors))
        close = getattr(controller, "close", None)
        if not callable(close):
            raise B601HardwareError("vendor controller has no transport release operation")
        if close() is False:
            raise B601HardwareError("vendor controller transport release is incomplete")
        # A failed close keeps the original handle for explicit retry and
        # must never be reported as a completed resource release.
        self._controller = None
        self._motors = {}
        self._motorbridge_runtime_evidence = None

    def _require_connected(self) -> object:
        if self._controller is None:
            raise B601HardwareError("B601 backend is not connected")
        return self._controller

    def _require_motion_authorized(self) -> None:
        if not self.config.allow_motion:
            raise B601MotionAuthorizationError(
                "motion command rejected: construct the backend with allow_motion=True"
            )
        if not self.parameter_reply_correlation_verified:
            raise B601HardwareDependencyError(
                "motion command rejected: strict RobStride type-17 parameter-reply "
                "correlation was not verified at connect"
            )

    def _require_active_reporting_authorized(self) -> None:
        if not self.config.allow_active_reporting:
            raise B601ActiveReportAuthorizationError(
                "active-report request rejected: construct the backend with "
                "allow_active_reporting=True"
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

    @staticmethod
    def _validate_selected_position_limits(
        values: Sequence[float],
        *,
        joint_names: tuple[str, ...],
        limits: tuple[tuple[float, float], ...],
        field_name: str,
    ) -> None:
        """Validate only the references that the command will dispatch.

        Command APIs accept a complete six-joint vector so the arm ordering
        remains stable, while ``joint_names`` selects the motors that receive
        the frame.  A non-selected joint may legitimately be outside the
        configured envelope (for example, its current hardware pose can be
        stale or intentionally parked elsewhere); it must not block a command
        that does not address that joint.
        """

        selected = set(joint_names)
        for index, (name, (lower, upper), value) in enumerate(
            zip(B601_RS_ARM_JOINT_NAMES, limits, values)
        ):
            if name not in selected:
                continue
            if not lower <= value <= upper:
                raise ValueError(
                    f"{field_name}[{index}] is outside the configured position "
                    f"limit for {name}: {value:.6f} not in [{lower:.6f}, {upper:.6f}]"
                )

    def poll_feedback(
        self,
        *,
        request_feedback: bool = False,
        require_host_freshness: bool = False,
        prefer_active_report: bool = False,
    ) -> B601HardwareState:
        """Observe six joints, optionally requesting a fresh vendor snapshot.

        The default is deliberately receive-only: it consumes the MotorBridge
        cache and never sends a CAN frame.  A motion-capable commissioning
        loop can pass ``request_feedback=True`` to issue one explicit
        ``Motor.request_feedback()`` per configured arm joint before calling
        ``Controller.poll_feedback_once()``.  For RobStride, the active path
        then reads ``mechPos`` (0x7019) through the compatibility parameter
        path.  This remains an observation request, not a motion command.

        ``require_host_freshness=True`` is a separate fail-closed path for an
        already configured operation-status stream.  It requires MotorBridge's
        receive-sequence/age capability, does not issue parameter reads, and
        rejects cached frames older than ``feedback_timeout_sec``.  It still
        cannot claim device-source timestamp freshness.

        ``prefer_active_report=True`` selects the dedicated Type-24 cache when
        the runtime advertises it. Older runtimes retain their generic cache;
        the caller's Type-24 admission gate remains mandatory. This preference
        requires host freshness and never rescues an empty/stale/broken
        advertised dedicated cache with generic replies. Ordinary host-fresh
        consumers retain the existing newest-status contract.
        """

        if not isinstance(request_feedback, bool):
            raise TypeError("request_feedback must be a bool")
        if not isinstance(require_host_freshness, bool):
            raise TypeError("require_host_freshness must be a bool")
        if not isinstance(prefer_active_report, bool):
            raise TypeError("prefer_active_report must be a bool")
        if prefer_active_report and not require_host_freshness:
            raise ValueError("prefer_active_report requires require_host_freshness")
        if request_feedback and require_host_freshness:
            raise ValueError(
                "request_feedback and require_host_freshness are separate feedback paths"
            )
        if require_host_freshness and not self.state_observation_metadata_verified:
            raise B601HardwareDependencyError(
                "host-fresh B601 feedback requires MotorBridge RobStride capability "
                f"{_B601_RS_STATE_OBSERVATION_CAPABILITY!r}"
            )

        controller = self._require_connected()
        now = _finite_float(self._clock(), name="clock")
        self._state_sequence += 1
        request_errors: list[str] = []
        if request_feedback:
            for joint in self.config.joints:
                try:
                    request = getattr(self._motors[joint.name], "request_feedback")
                    request()
                except Exception as exc:  # vendor errors become an invalid snapshot
                    request_errors.append(f"{joint.name}.request_feedback: {exc}")
        poll_error: str | None = None
        try:
            poll = getattr(controller, "poll_feedback_once")
            poll()
        except Exception as exc:  # vendor errors become an invalid snapshot
            poll_error = f"poll_feedback_once: {exc}"

        observations: list[B601JointState] = []
        errors: list[str] = [*request_errors]
        if poll_error:
            errors.append(poll_error)
        feedback_limits = (
            self.config.feedback_position_limits_rad
            if self.config.feedback_position_limits_rad is not None
            else self.config.position_limits_rad
        )
        for joint in self.config.joints:
            motor = self._motors[joint.name]
            raw_state_error: str | None = None
            status_sequence: int | None = None
            receive_age_sec: float | None = None
            host_receive_freshness_verified = False
            observed_at = now
            if require_host_freshness:
                features = (self._motorbridge_runtime_evidence or {}).get("robstride_features", ())
                observation_method = (
                    "robstride_get_active_report_observation"
                    if prefer_active_report and _B601_RS_ACTIVE_REPORT_OBSERVATION_CAPABILITY in features
                    else "robstride_get_state_observation"
                )
                try:
                    # The dedicated stream retains its original age/sequence
                    # while Type-2 command replies update the generic cache.
                    # A missing/failing advertised getter must NOT fall back.
                    raw_observation = getattr(motor, observation_method)()
                    if raw_observation is None:
                        raw_state = None
                        raw_state_error = (
                            f"{joint.name}: operation-status cache is empty"
                        )
                    else:
                        raw_state = raw_observation.state
                        raw_sequence = raw_observation.status_sequence
                        if (
                            isinstance(raw_sequence, bool)
                            or not isinstance(raw_sequence, int)
                            or raw_sequence <= 0
                        ):
                            raise ValueError(
                                "status_sequence must be a positive integer"
                            )
                        status_sequence = raw_sequence
                        raw_age_ns = raw_observation.receive_age_ns
                        if (
                            isinstance(raw_age_ns, bool)
                            or not isinstance(raw_age_ns, int)
                            or raw_age_ns < 0
                        ):
                            raise ValueError("receive_age_ns must be non-negative")
                        receive_age_sec = raw_age_ns / 1_000_000_000.0
                        observed_at = now - receive_age_sec
                        if receive_age_sec > self.config.feedback_timeout_sec:
                            raw_state_error = (
                                f"{joint.name}: operation-status receive age "
                                f"{receive_age_sec:.6f}s exceeds feedback timeout "
                                f"{self.config.feedback_timeout_sec:.6f}s"
                            )
                        else:
                            host_receive_freshness_verified = True
                except Exception as exc:  # noqa: BLE001 - vendor observation boundary
                    raw_state = None
                    raw_state_error = (
                        f"{joint.name}.{observation_method}: {exc}"
                    )
            else:
                try:
                    raw_state = motor.get_state()
                except Exception as exc:  # noqa: BLE001 - vendor observation boundary
                    raw_state = None
                    raw_state_error = f"{joint.name}.get_state: {exc}"

            # The pinned reBotArm compatibility implementation reads mechPos
            # (0x7019) explicitly. Keep this request opt-in; the host-fresh
            # operation-status path above never mixes that parameter value
            # with metadata belonging to a different frame.
            live_position: float | None = None
            position_error: str | None = None
            position_source = (
                "robstride_operation_status_cache"
                if require_host_freshness
                else "vendor_motor_state_cache"
            )
            if request_feedback and joint.vendor == "robstride":
                try:
                    timeout_ms = max(
                        1,
                        int(round(self.config.feedback_timeout_sec * 1000.0)),
                    )
                    read_position_host = getattr(
                        motor,
                        "robstride_get_param_f32_host_id",
                        None,
                    )
                    if callable(read_position_host):
                        # MotorBridge exposes a host-id-qualified ABI for
                        # RobStride parameter reads.  Prefer it when present:
                        # a shared CAN host can otherwise accept a stale or
                        # mismatched parameter response as this motor's value.
                        raw_position = read_position_host(
                            B601_RS_MECH_POS_PARAM_ID,
                            joint.feedback_id,
                            timeout_ms,
                        )
                        position_source = (
                            "robstride_mech_pos_parameter_host_id"
                        )
                    else:
                        read_position = getattr(
                            motor,
                            "robstride_get_param_f32",
                        )
                        raw_position = read_position(
                            B601_RS_MECH_POS_PARAM_ID,
                            timeout_ms,
                        )
                        position_source = "robstride_mech_pos_parameter"
                    live_position = _finite_float(
                        raw_position,
                        name=f"{joint.name}.mechPos",
                    )
                except Exception as exc:  # vendor errors become an invalid snapshot
                    position_error = (
                        f"{joint.name}.mechPos(0x{B601_RS_MECH_POS_PARAM_ID:x}, "
                        f"host_id=0x{joint.feedback_id:02x}): {exc}"
                    )

            # Keep the read-side admission envelope separate from the
            # outgoing command envelope.  A finite IEEE-754 value can still
            # be an impossible encoder position (for example a corrupted
            # parameter response); reject it here, before any consumer can
            # treat the snapshot as valid or use it for gravity/FK/dispatch.
            if (
                position_error is None
                and feedback_limits is not None
                and live_position is not None
            ):
                lower, upper = feedback_limits[
                    B601_RS_ARM_JOINT_NAMES.index(joint.name)
                ]
                if not lower <= live_position <= upper:
                    raw_bits = _float32_bits(live_position)
                    raw_bits_text = (
                        f" raw_f32_bits=0x{raw_bits:08x}"
                        if raw_bits is not None
                        else ""
                    )
                    position_error = (
                        f"{joint.name}: mechPos(0x{B601_RS_MECH_POS_PARAM_ID:04x}) "
                        f"raw={live_position!r} motor_id={joint.motor_id} "
                        f"feedback_id={joint.feedback_id} source={position_source} "
                        f"{raw_bits_text} outside feedback "
                        f"acceptance limits [{lower!r}, {upper!r}]"
                    )

            if raw_state is None and live_position is None:
                error = (
                    position_error
                    or raw_state_error
                    or f"{joint.name}: feedback cache is empty"
                )
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
                        observed_at_sec=observed_at,
                        sample_sequence=self._state_sequence,
                        valid=False,
                        stale=True,
                        error=error,
                        status_sequence=status_sequence,
                        receive_age_sec=receive_age_sec,
                        host_receive_freshness_verified=False,
                    )
                )
                continue
            position = live_position
            if raw_state is None:
                errors.extend(
                    error
                    for error in (raw_state_error, poll_error)
                    if error and error not in errors
                )
                if position_error and position_error not in errors:
                    errors.append(position_error)
                observations.append(
                    B601JointState(
                        name=joint.name,
                        motor_id=joint.motor_id,
                        feedback_id=joint.feedback_id,
                        arbitration_id=None,
                        status_code=None,
                        position_rad=position,
                        velocity_reported=None,
                        torque_reported_nm=None,
                        observed_at_sec=observed_at,
                        sample_sequence=self._state_sequence,
                        valid=(
                            not request_errors
                            and poll_error is None
                            and position_error is None
                        ),
                        stale=bool(
                            request_errors or poll_error or position_error
                        ),
                        error=position_error or poll_error or raw_state_error,
                        position_source=position_source,
                        status_sequence=status_sequence,
                        receive_age_sec=receive_age_sec,
                        host_receive_freshness_verified=False,
                    )
                )
                continue
            try:
                can_id = int(getattr(raw_state, "can_id"))
                arbitration_id = int(getattr(raw_state, "arbitration_id"))
                status_code = int(getattr(raw_state, "status_code"))
                cached_position = _finite_float(
                    getattr(raw_state, "pos"), name=f"{joint.name}.pos"
                )
                position = cached_position if live_position is None else live_position
                velocity = _finite_float(getattr(raw_state, "vel"), name=f"{joint.name}.vel")
                torque = _finite_float(getattr(raw_state, "torq"), name=f"{joint.name}.torq")
                state_error: str | None = None
                if raw_state_error:
                    state_error = raw_state_error
                elif can_id != joint.motor_id:
                    state_error = f"{joint.name}: feedback can_id={can_id} expected {joint.motor_id}"
                elif status_code != 0:
                    state_error = f"{joint.name}: vendor status_code={status_code}"
                if (
                    position_error is None
                    and feedback_limits is not None
                    and position is not None
                ):
                    lower, upper = feedback_limits[
                        B601_RS_ARM_JOINT_NAMES.index(joint.name)
                    ]
                    if not lower <= position <= upper:
                        source_label = (
                            f"mechPos(0x{B601_RS_MECH_POS_PARAM_ID:04x})"
                            if live_position is not None
                            else "vendor_state.pos"
                        )
                        raw_bits = _float32_bits(position)
                        raw_bits_text = (
                            f" raw_f32_bits=0x{raw_bits:08x}"
                            if raw_bits is not None
                            else ""
                        )
                        position_error = (
                            f"{joint.name}: {source_label} raw={position!r} "
                            f"motor_id={joint.motor_id} feedback_id={joint.feedback_id} "
                            f"{raw_bits_text} outside feedback acceptance limits "
                            f"[{lower!r}, {upper!r}]"
                        )
                if state_error:
                    errors.append(state_error)
                if position_error:
                    errors.append(position_error)
                if raw_state_error and raw_state_error != state_error:
                    errors.append(raw_state_error)
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
                        observed_at_sec=observed_at,
                        sample_sequence=self._state_sequence,
                        valid=(
                            state_error is None
                            and position_error is None
                            and poll_error is None
                            and not request_errors
                        ),
                        stale=bool(
                            state_error
                            or position_error
                            or poll_error
                            or request_errors
                        ),
                        error=state_error or position_error or poll_error,
                        position_source=position_source,
                        status_sequence=status_sequence,
                        receive_age_sec=receive_age_sec,
                        host_receive_freshness_verified=(
                            host_receive_freshness_verified
                            and state_error is None
                            and position_error is None
                            and poll_error is None
                        ),
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
                        observed_at_sec=observed_at,
                        sample_sequence=self._state_sequence,
                        valid=False,
                        stale=True,
                        error=error,
                        status_sequence=status_sequence,
                        receive_age_sec=receive_age_sec,
                        host_receive_freshness_verified=False,
                    )
                )
        valid = not errors and all(joint.valid for joint in observations)
        host_receive_freshness_verified = bool(
            require_host_freshness
            and valid
            and all(
                joint.host_receive_freshness_verified for joint in observations
            )
        )
        state = B601HardwareState(
            sequence=self._state_sequence,
            receive_timestamp_sec=now,
            joints=tuple(observations),
            valid=valid,
            stale=not valid,
            freshness_verified=False,
            freshness_semantics=(
                "host_receive_sequence_and_age_verified; "
                "device_source_timestamp_unavailable"
                if host_receive_freshness_verified
                else "poll_observation_unverified"
            ),
            velocity_semantics=(
                "robstride_operation_status_host_receive_verified"
                if host_receive_freshness_verified
                else "vendor_reported_unverified"
            ),
            fault_reason="; ".join(errors) if errors else None,
            host_receive_freshness_verified=host_receive_freshness_verified,
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
        velocity_limits_rad_s: Sequence[float] | None = None,
        timeout_ms: int = 1000,
    ) -> B601ModeReceipt:
        """Apply reBotArm POS_VEL parameters and set the mode once.

        The pinned reBotArm source writes RobStride ``limit_spd`` and the
        velocity/position-loop gains before calling ``ensure_mode``.  Keep the
        whole sequence outside the high-rate command loop, and fail closed if
        any parameter write fails so stale vendor gains are never reported as
        a completed setup.
        """

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
                raise ValueError(
                    "velocity_limits_rad_s exceeds the configured per-joint limit"
                )
        index_by_name = {
            joint.name: index for index, joint in enumerate(self.config.joints)
        }
        completed: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for name in names:
            try:
                joint = self.config.joints[index_by_name[name]]
                write_param = getattr(self._motors[name], "robstride_write_param_f32")
                # This order and the short settling delays mirror the pinned
                # reBotArm ``JointGroup._write_pv_params`` implementation.
                write_param(
                    B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID,
                    float(velocity_limits[index_by_name[name]]),
                )
                self._sleeper(0.01)
                write_param(
                    B601_RS_POS_VEL_VEL_KP_PARAM_ID,
                    float(joint.position_velocity_kp),
                )
                self._sleeper(0.01)
                write_param(
                    B601_RS_POS_VEL_VEL_KI_PARAM_ID,
                    float(joint.position_velocity_ki),
                )
                self._sleeper(0.01)
                write_param(
                    B601_RS_POS_VEL_POS_KP_PARAM_ID,
                    float(joint.position_kp),
                )
                self._sleeper(0.02)
                getattr(self._motors[name], "ensure_mode")(mode, int(timeout_ms))
                self._sleeper(0.05)
                completed.append(name)
            except Exception as exc:
                failed.append(name)
                errors.append(f"{name}: {exc}")
        # JointGroup.mode_pos_vel() waits after the per-joint transition before
        # returning control to the caller.  This is still one-time setup, never
        # part of the 125/250/500 Hz reference loop.
        self._sleeper(0.2)
        return B601ModeReceipt("prepare_position_velocity_mode", names, tuple(completed), tuple(failed), tuple(errors), not failed)

    def prepare_mit_mode(
        self,
        *,
        joint_names: Sequence[str] | None = None,
        timeout_ms: int = 1000,
    ) -> B601ModeReceipt:
        """Set RobStride MIT mode once, outside the high-rate send loop.

        This mirrors the pinned reBotArm ``JointGroup.mode_mit()`` transition:
        one ``ensure_mode(Mode.MIT)`` call per selected joint, the same
        per-joint settling delay, and the same final group delay.  It does not
        enable a motor or select operational gains; callers pass MIT gains and
        feedforward torque explicitly to :meth:`send_mit`.
        """

        self._require_motion_authorized()
        names = self._resolve_joint_names(joint_names)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms <= 0
        ):
            raise ValueError("timeout_ms must be a positive integer")
        try:
            mode = getattr(import_module("motorbridge"), "Mode").MIT
        except (ModuleNotFoundError, AttributeError) as exc:
            raise B601HardwareDependencyError(
                "motorbridge.Mode.MIT is unavailable"
            ) from exc
        completed: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for name in names:
            try:
                getattr(self._motors[name], "ensure_mode")(mode, int(timeout_ms))
                self._sleeper(0.05)
                completed.append(name)
            except Exception as exc:
                failed.append(name)
                errors.append(f"{name}: {exc}")
        self._sleeper(0.2)
        return B601ModeReceipt(
            "prepare_mit_mode",
            names,
            tuple(completed),
            tuple(failed),
            tuple(errors),
            not failed,
        )

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
        if operation == "enable":
            self._enabled_joint_names.update(completed)
            self._disabled_joint_names.difference_update(completed)
        elif operation in ("disable", "safe_stop"):
            self._enabled_joint_names.difference_update(completed)
            self._disabled_joint_names.update(completed)
        return B601ModeReceipt(operation, names, tuple(completed), tuple(failed), tuple(errors), not failed)

    def set_active_reporting(
        self,
        enabled: bool,
        *,
        joint_names: Sequence[str] | None = None,
    ) -> B601ActiveReportReceipt:
        """Toggle RobStride type-24 active reporting without motor enable.

        This is a telemetry configuration operation with an authorization gate
        separate from motion.  Enabling is transactional across the selected
        joints: after a partial failure every successfully enabled joint is
        disabled in reverse order.  Any failed cleanup remains latched so
        :meth:`close` cannot silently discard an uncertain device state.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self._require_active_reporting_authorized()
        names = self._resolve_joint_names(joint_names)
        operation = "enable_active_reporting" if enabled else "disable_active_reporting"
        completed: list[str] = []
        failed: list[str] = []
        errors: list[str] = []

        for name in names:
            try:
                self._motors[name].robstride_set_active_report(enabled)
                completed.append(name)
            except Exception as exc:  # noqa: BLE001 - vendor failure becomes receipt evidence
                failed.append(name)
                errors.append(f"{name}: {exc}")
                if enabled:
                    break

        if not enabled:
            self._active_report_joint_names.difference_update(completed)
            return B601ActiveReportReceipt(
                operation=operation,
                requested_joint_names=names,
                completed_joint_names=tuple(completed),
                failed_joint_names=tuple(failed),
                errors=tuple(errors),
                completed=not failed,
            )

        if not failed:
            self._active_report_joint_names.update(completed)
            return B601ActiveReportReceipt(
                operation=operation,
                requested_joint_names=names,
                completed_joint_names=tuple(completed),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

        # A vendor exception does not prove the type-24 request was never put
        # on the bus.  Treat the failing joint as uncertain and send the
        # disable request to every attempted joint, not only acknowledged ones.
        attempted = (*completed, failed[0])
        rollback_completed: list[str] = []
        rollback_failed: list[str] = []
        for name in reversed(attempted):
            try:
                self._motors[name].robstride_set_active_report(False)
                rollback_completed.append(name)
            except Exception as exc:  # noqa: BLE001 - vendor failure becomes receipt evidence
                rollback_failed.append(name)
                errors.append(f"rollback {name}: {exc}")
        self._active_report_joint_names.update(rollback_failed)
        failed_names = tuple(dict.fromkeys((*failed, *rollback_failed)))
        return B601ActiveReportReceipt(
            operation=operation,
            requested_joint_names=names,
            completed_joint_names=tuple(completed),
            failed_joint_names=failed_names,
            errors=tuple(errors),
            completed=False,
            rollback_attempted=True,
            rollback_completed=not rollback_failed,
        )

    @staticmethod
    def _validate_parameter_timeout_ms(timeout_ms: int) -> int:
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        return timeout_ms

    def _read_driver_torque_limit(self, name: str, *, timeout_ms: int) -> float:
        value = _finite_float(
            self._motors[name].robstride_get_param_f32(
                B601_RS_TORQUE_LIMIT_PARAM_ID,
                timeout_ms,
            ),
            name=f"{name}.driver_torque_limit_nm",
        )
        if not 0.0 <= value <= B601_RS_TORQUE_LIMIT_MAX_NM:
            raise ValueError(
                f"{name}.driver_torque_limit_nm={value:.6f} is outside "
                f"[0, {B601_RS_TORQUE_LIMIT_MAX_NM:.1f}]"
            )
        return value

    def _write_and_verify_driver_torque_limit(
        self,
        name: str,
        value_nm: float,
        *,
        timeout_ms: int,
    ) -> float:
        self._motors[name].robstride_write_param_f32(
            B601_RS_TORQUE_LIMIT_PARAM_ID,
            value_nm,
        )
        observed = self._read_driver_torque_limit(name, timeout_ms=timeout_ms)
        if not math.isclose(
            observed,
            value_nm,
            rel_tol=0.0,
            abs_tol=B601_RS_TORQUE_LIMIT_VERIFY_ABS_TOL_NM,
        ):
            raise B601HardwareError(
                f"{name}: driver torque-limit verify mismatch: "
                f"requested={value_nm:.6f} observed={observed:.6f} Nm"
            )
        return observed

    @staticmethod
    def _aligned_limit_values(
        names: Sequence[str],
        values: Mapping[str, float | None],
    ) -> tuple[float | None, ...]:
        return tuple(values.get(name) for name in names)

    def apply_temporary_driver_torque_limits(
        self,
        limits_nm: Sequence[float],
        *,
        joint_names: Sequence[str] | None = None,
        timeout_ms: int = 500,
    ) -> B601DriverTorqueLimitReceipt:
        """Apply volatile ``limit_torque`` values with readback and rollback.

        This operation is admitted only while the selected joints are locally
        known disabled.  All original values are read before the first write.
        A write/readback failure rolls every touched joint back to its original
        value; any unverified rollback remains latched for an explicit retry.
        """

        self._require_motion_authorized()
        names = self._resolve_joint_names(joint_names)
        timeout = self._validate_parameter_timeout_ms(timeout_ms)
        if self._driver_torque_limit_original_nm:
            raise B601HardwareError(
                "temporary driver torque limits are already active; restore them first"
            )
        requested = _six_nonnegative(limits_nm, name="limits_nm")
        index_by_name = {
            name: index for index, name in enumerate(B601_RS_ARM_JOINT_NAMES)
        }
        targets = {name: requested[index_by_name[name]] for name in names}
        if any(value > B601_RS_TORQUE_LIMIT_MAX_NM for value in targets.values()):
            raise ValueError(
                f"limits_nm exceeds RobStride {B601_RS_TORQUE_LIMIT_MAX_NM:.1f} Nm"
            )
        not_confirmed_disabled = tuple(
            name for name in names if name not in self._disabled_joint_names
        )
        if not_confirmed_disabled:
            raise B601HardwareError(
                "temporary driver torque limits require a successful disable/safe_stop "
                "receipt in this backend session: " + ", ".join(not_confirmed_disabled)
            )

        originals: dict[str, float] = {}
        verified: dict[str, float] = {}
        errors: list[str] = []
        failed: list[str] = []
        for name in names:
            try:
                originals[name] = self._read_driver_torque_limit(
                    name,
                    timeout_ms=timeout,
                )
            except Exception as exc:  # noqa: BLE001 - vendor failure becomes receipt evidence
                failed.append(name)
                errors.append(f"read original {name}: {exc}")
                return B601DriverTorqueLimitReceipt(
                    operation="apply_temporary_driver_torque_limits",
                    requested_joint_names=names,
                    target_limits_nm=tuple(targets[item] for item in names),
                    prior_limits_nm=self._aligned_limit_values(names, originals),
                    verified_limits_nm=self._aligned_limit_values(names, verified),
                    applied_joint_names=(),
                    restored_joint_names=(),
                    failed_joint_names=tuple(failed),
                    errors=tuple(errors),
                    completed=False,
                )

        touched: list[str] = []
        applied: list[str] = []
        for name in names:
            touched.append(name)
            try:
                verified[name] = self._write_and_verify_driver_torque_limit(
                    name,
                    targets[name],
                    timeout_ms=timeout,
                )
                applied.append(name)
            except Exception as exc:  # noqa: BLE001 - vendor failure becomes receipt evidence
                failed.append(name)
                errors.append(f"apply {name}: {exc}")
                break

        if failed:
            restored: list[str] = []
            rollback_failed: list[str] = []
            for name in reversed(touched):
                try:
                    verified[name] = self._write_and_verify_driver_torque_limit(
                        name,
                        originals[name],
                        timeout_ms=timeout,
                    )
                    restored.append(name)
                except Exception as exc:  # noqa: BLE001 - vendor failure becomes receipt evidence
                    rollback_failed.append(name)
                    errors.append(f"rollback {name}: {exc}")
            unresolved = set(touched).difference(restored)
            self._driver_torque_limit_original_nm = {
                name: originals[name] for name in touched if name in unresolved
            }
            self._driver_torque_limit_applied_nm = {
                # A failed rollback leaves the device value uncertain even if
                # the target was previously verified.  Do not misreport a
                # stale observation as the final driver state.
                name: None
                for name in touched
                if name in unresolved
            }
            failed_names = tuple(dict.fromkeys((*failed, *rollback_failed)))
            return B601DriverTorqueLimitReceipt(
                operation="apply_temporary_driver_torque_limits",
                requested_joint_names=names,
                target_limits_nm=tuple(targets[item] for item in names),
                prior_limits_nm=self._aligned_limit_values(names, originals),
                verified_limits_nm=self._aligned_limit_values(names, verified),
                applied_joint_names=tuple(applied),
                restored_joint_names=tuple(restored),
                failed_joint_names=failed_names,
                errors=tuple(errors),
                completed=False,
                rollback_attempted=True,
                rollback_completed=not rollback_failed,
            )

        self._driver_torque_limit_original_nm = dict(originals)
        self._driver_torque_limit_applied_nm = dict(targets)
        return B601DriverTorqueLimitReceipt(
            operation="apply_temporary_driver_torque_limits",
            requested_joint_names=names,
            target_limits_nm=tuple(targets[item] for item in names),
            prior_limits_nm=self._aligned_limit_values(names, originals),
            verified_limits_nm=self._aligned_limit_values(names, verified),
            applied_joint_names=tuple(applied),
            restored_joint_names=(),
            failed_joint_names=(),
            errors=(),
            completed=True,
        )

    def restore_driver_torque_limits(
        self,
        *,
        timeout_ms: int = 500,
    ) -> B601DriverTorqueLimitReceipt:
        """Restore every latched original torque limit after motor disable."""

        self._require_connected()
        timeout = self._validate_parameter_timeout_ms(timeout_ms)
        names = self.temporary_driver_torque_limit_joint_names
        if not names:
            return B601DriverTorqueLimitReceipt(
                operation="restore_driver_torque_limits",
                requested_joint_names=(),
                target_limits_nm=(),
                prior_limits_nm=(),
                verified_limits_nm=(),
                applied_joint_names=(),
                restored_joint_names=(),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )
        enabled = tuple(name for name in names if name in self._enabled_joint_names)
        if enabled:
            raise B601HardwareError(
                "driver torque limits may be restored only after disable: "
                + ", ".join(enabled)
            )

        targets = dict(self._driver_torque_limit_original_nm)
        prior = dict(self._driver_torque_limit_applied_nm)
        verified: dict[str, float] = {}
        restored: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for name in names:
            try:
                verified[name] = self._write_and_verify_driver_torque_limit(
                    name,
                    targets[name],
                    timeout_ms=timeout,
                )
                restored.append(name)
            except Exception as exc:  # noqa: BLE001 - vendor failure becomes receipt evidence
                failed.append(name)
                errors.append(f"restore {name}: {exc}")
        for name in restored:
            self._driver_torque_limit_original_nm.pop(name, None)
            self._driver_torque_limit_applied_nm.pop(name, None)
        return B601DriverTorqueLimitReceipt(
            operation="restore_driver_torque_limits",
            requested_joint_names=names,
            target_limits_nm=tuple(targets[name] for name in names),
            prior_limits_nm=self._aligned_limit_values(names, prior),
            verified_limits_nm=self._aligned_limit_values(names, verified),
            applied_joint_names=(),
            restored_joint_names=tuple(restored),
            failed_joint_names=tuple(failed),
            errors=tuple(errors),
            completed=not failed and not self._driver_torque_limit_original_nm,
        )

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
        position_limits = self._require_position_limits()
        qpos = _six_finite(qpos_rad, name="qpos_rad")
        self._validate_selected_position_limits(
            qpos,
            joint_names=names,
            limits=position_limits,
            field_name="qpos_rad",
        )
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
        self._validate_selected_position_limits(
            reference.qpos_rad,
            joint_names=names,
            limits=position_limits,
            field_name="reference.qpos_rad",
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
    "B601_RS_DEFAULT_CHANNEL",
    "B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC",
    "B601_RS_DEFAULT_RATE_HZ",
    "B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC",
    "B601_RS_FEEDBACK_FRAME",
    "B601_RS_MECH_POS_PARAM_ID",
    "B601_RS_POSITION_FRAME",
    "B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID",
    "B601_RS_POS_VEL_POS_KP_PARAM_ID",
    "B601_RS_POS_VEL_VEL_KI_PARAM_ID",
    "B601_RS_POS_VEL_VEL_KP_PARAM_ID",
    "B601_RS_TORQUE_LIMIT_MAX_NM",
    "B601_RS_TORQUE_LIMIT_PARAM_ID",
    "B601_RS_TORQUE_LIMIT_VERIFY_ABS_TOL_NM",
    "B601CommandReceipt",
    "B601DriverTorqueLimitReceipt",
    "B601HardwareDependencyError",
    "B601HardwareError",
    "B601HardwareState",
    "B601JointConfig",
    "B601JointState",
    "B601ModeReceipt",
    "B601MotionAuthorizationError",
    "B601RSBackend",
    "B601RSBackendConfig",
]
