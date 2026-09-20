"""Pure B601 shared-control policy-assist semantics.

This module owns neither CAN nor a motor.  It turns a timestamped, short-horizon
policy joint reference into a bounded assist torque for the existing B601
hardware owner.  The hardware owner remains responsible for gravity
compensation, passive damping, final total-torque conditioning, watchdogs,
protective hold and the only actuator dispatch path.

The policy path deliberately fails passive: invalid, future, stale or replayed
commands produce zero assist.  An explicit clutch also drops assist to zero
immediately.  After an override, a strictly newer policy sequence is required
before authority can recover, so releasing the clutch cannot restore an old
absolute target and pull the arm back.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .model import B601_RS_ARM_JOINT_NAMES

B601_SHARED_POLICY_FRAME = "b601_rs/joint_policy_assist"
_ZERO_SIX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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


def _six_finite(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    try:
        normalized = tuple(_finite(value, name=name) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if len(normalized) != 6:
        raise ValueError(f"{name} must contain six finite values")
    return normalized


def _six_nonnegative(
    values: Sequence[float],
    *,
    name: str,
    strictly_positive: bool = False,
) -> tuple[float, ...]:
    normalized = _six_finite(values, name=name)
    if any(
        value <= 0.0 if strictly_positive else value < 0.0
        for value in normalized
    ):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must contain six {qualifier} values")
    return normalized


def _sequence(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class B601PolicyAssistCommand:
    """One local-clock, short-horizon B601 policy reference.

    The command is an intent sample, never an actuator command.  Both
    timestamps use the owner's host-local monotonic clock.  The first shared
    control slice accepts joint references only; task-space policy output must
    be converted to a local B601 reference before publishing this contract.
    """

    sequence: int
    source_timestamp_sec: float
    receive_timestamp_sec: float
    qpos_target_rad: tuple[float, ...]
    qvel_target_rad_s: tuple[float, ...] = _ZERO_SIX
    frame_id: str = B601_SHARED_POLICY_FRAME
    valid: bool = True

    def __post_init__(self) -> None:
        sequence = _sequence(self.sequence, name="sequence")
        source = _finite(self.source_timestamp_sec, name="source_timestamp_sec")
        receive = _finite(self.receive_timestamp_sec, name="receive_timestamp_sec")
        if source < 0.0 or receive < 0.0:
            raise ValueError("policy timestamps must be non-negative")
        if receive < source:
            raise ValueError("receive_timestamp_sec must not precede source_timestamp_sec")
        frame_id = str(self.frame_id).strip()
        if frame_id != B601_SHARED_POLICY_FRAME:
            raise ValueError(
                "frame_id must match the B601 shared-policy joint convention"
            )
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "source_timestamp_sec", source)
        object.__setattr__(self, "receive_timestamp_sec", receive)
        object.__setattr__(
            self,
            "qpos_target_rad",
            _six_finite(self.qpos_target_rad, name="qpos_target_rad"),
        )
        object.__setattr__(
            self,
            "qvel_target_rad_s",
            _six_finite(self.qvel_target_rad_s, name="qvel_target_rad_s"),
        )
        object.__setattr__(self, "frame_id", frame_id)


@dataclass(frozen=True)
class B601PolicyCommandReceipt:
    accepted: bool
    reason: str
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("reason must not be empty")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "sequence", _sequence(self.sequence, name="sequence"))


@dataclass(frozen=True)
class B601PolicyCommandSample:
    command: B601PolicyAssistCommand | None
    accepted: bool
    reason: str
    age_sec: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        if self.accepted != (self.command is not None):
            raise ValueError("accepted must be true exactly when command is present")
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("reason must not be empty")
        age = self.age_sec
        if age is not None:
            age = _finite(age, name="age_sec")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "age_sec", age)


class B601PolicyCommandBuffer:
    """Thread-safe latest-value buffer; old commands are never queued."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: B601PolicyAssistCommand | None = None

    def publish(self, command: B601PolicyAssistCommand) -> B601PolicyCommandReceipt:
        if not isinstance(command, B601PolicyAssistCommand):
            raise TypeError("command must be a B601PolicyAssistCommand")
        with self._lock:
            latest = self._latest
            if latest is not None and command.sequence <= latest.sequence:
                return B601PolicyCommandReceipt(
                    accepted=False,
                    reason="non_advancing_sequence",
                    sequence=command.sequence,
                )
            if not command.valid:
                return B601PolicyCommandReceipt(
                    accepted=False,
                    reason="command_marked_invalid",
                    sequence=command.sequence,
                )
            self._latest = command
        return B601PolicyCommandReceipt(
            accepted=True,
            reason="accepted",
            sequence=command.sequence,
        )

    def sample(
        self,
        *,
        now_sec: float,
        maximum_age_sec: float,
        maximum_future_skew_sec: float,
    ) -> B601PolicyCommandSample:
        now = _finite(now_sec, name="now_sec")
        maximum_age = _positive(maximum_age_sec, name="maximum_age_sec")
        future_skew = _finite(
            maximum_future_skew_sec,
            name="maximum_future_skew_sec",
        )
        if future_skew < 0.0:
            raise ValueError("maximum_future_skew_sec must be non-negative")
        with self._lock:
            command = self._latest
        if command is None:
            return B601PolicyCommandSample(None, False, "no_policy_command", None)
        age = now - command.source_timestamp_sec
        if age < -future_skew:
            return B601PolicyCommandSample(
                None,
                False,
                "policy_command_from_future",
                age,
            )
        if age > maximum_age:
            return B601PolicyCommandSample(None, False, "policy_command_stale", age)
        return B601PolicyCommandSample(command, True, "accepted", age)

    def sample_for_config(
        self,
        *,
        now_sec: float,
        config: B601SharedControlConfig,
    ) -> B601PolicyCommandSample:
        """Sample using the exact watchdog bounds selected by the local owner."""

        if not isinstance(config, B601SharedControlConfig):
            raise TypeError("config must be a B601SharedControlConfig")
        return self.sample(
            now_sec=now_sec,
            maximum_age_sec=config.policy_stale_timeout_sec,
            maximum_future_skew_sec=config.maximum_future_skew_sec,
        )


class B601SharedControlMode(str, Enum):
    PASSIVE = "passive"
    POLICY_ASSIST = "policy_assist"
    HUMAN_OVERRIDE = "human_override"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class B601SharedControlConfig:
    """Local limits for policy assist; these are not actuator torque limits."""

    position_kp_nm_per_rad: tuple[float, ...]
    velocity_kd_nm_s_per_rad: tuple[float, ...]
    max_target_error_rad: tuple[float, ...]
    max_abs_assist_torque_nm: tuple[float, ...]
    max_assist_torque_rate_nm_s: tuple[float, ...]
    authority_rise_per_sec: float
    policy_stale_timeout_sec: float
    maximum_future_skew_sec: float = 0.01
    joint_names: tuple[str, ...] = B601_RS_ARM_JOINT_NAMES

    def __post_init__(self) -> None:
        if tuple(self.joint_names) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("joint_names must match the B601-RS J1-J6 arm order")
        object.__setattr__(
            self,
            "position_kp_nm_per_rad",
            _six_nonnegative(
                self.position_kp_nm_per_rad,
                name="position_kp_nm_per_rad",
            ),
        )
        object.__setattr__(
            self,
            "velocity_kd_nm_s_per_rad",
            _six_nonnegative(
                self.velocity_kd_nm_s_per_rad,
                name="velocity_kd_nm_s_per_rad",
            ),
        )
        object.__setattr__(
            self,
            "max_target_error_rad",
            _six_nonnegative(
                self.max_target_error_rad,
                name="max_target_error_rad",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_assist_torque_nm",
            _six_nonnegative(
                self.max_abs_assist_torque_nm,
                name="max_abs_assist_torque_nm",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_assist_torque_rate_nm_s",
            _six_nonnegative(
                self.max_assist_torque_rate_nm_s,
                name="max_assist_torque_rate_nm_s",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "authority_rise_per_sec",
            _positive(self.authority_rise_per_sec, name="authority_rise_per_sec"),
        )
        object.__setattr__(
            self,
            "policy_stale_timeout_sec",
            _positive(
                self.policy_stale_timeout_sec,
                name="policy_stale_timeout_sec",
            ),
        )
        future_skew = _finite(
            self.maximum_future_skew_sec,
            name="maximum_future_skew_sec",
        )
        if future_skew < 0.0:
            raise ValueError("maximum_future_skew_sec must be non-negative")
        object.__setattr__(self, "maximum_future_skew_sec", future_skew)
        object.__setattr__(self, "joint_names", tuple(self.joint_names))


@dataclass(frozen=True)
class B601SharedControlState:
    mode: B601SharedControlMode = B601SharedControlMode.PASSIVE
    authority: float = 0.0
    previous_assist_torque_nm: tuple[float, ...] = _ZERO_SIX
    last_policy_sequence: int | None = None
    resume_after_sequence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, B601SharedControlMode):
            raise TypeError("mode must be a B601SharedControlMode")
        authority = _finite(self.authority, name="authority")
        if not 0.0 <= authority <= 1.0:
            raise ValueError("authority must be within [0, 1]")
        object.__setattr__(self, "authority", authority)
        object.__setattr__(
            self,
            "previous_assist_torque_nm",
            _six_finite(
                self.previous_assist_torque_nm,
                name="previous_assist_torque_nm",
            ),
        )
        for name in ("last_policy_sequence", "resume_after_sequence"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sequence(value, name=name))


@dataclass(frozen=True)
class B601SharedControlOutput:
    state: B601SharedControlState
    assist_torque_nm: tuple[float, ...]
    raw_target_error_rad: tuple[float, ...]
    bounded_target_error_rad: tuple[float, ...]
    raw_assist_torque_nm: tuple[float, ...]
    reason: str
    target_error_limited: bool
    torque_limited: bool
    slew_limited: bool


def _zero_output(
    *,
    mode: B601SharedControlMode,
    reason: str,
    last_policy_sequence: int | None,
    resume_after_sequence: int | None,
) -> B601SharedControlOutput:
    state = B601SharedControlState(
        mode=mode,
        authority=0.0,
        previous_assist_torque_nm=_ZERO_SIX,
        last_policy_sequence=last_policy_sequence,
        resume_after_sequence=resume_after_sequence,
    )
    return B601SharedControlOutput(
        state=state,
        assist_torque_nm=_ZERO_SIX,
        raw_target_error_rad=_ZERO_SIX,
        bounded_target_error_rad=_ZERO_SIX,
        raw_assist_torque_nm=_ZERO_SIX,
        reason=reason,
        target_error_limited=False,
        torque_limited=False,
        slew_limited=False,
    )


def compute_b601_shared_control_step(
    config: B601SharedControlConfig,
    state: B601SharedControlState,
    *,
    command_sample: B601PolicyCommandSample,
    measured_qpos_rad: Sequence[float],
    measured_qvel_rad_s: Sequence[float],
    dt_sec: float,
    policy_clutch: bool,
    human_override_detected: bool = False,
) -> B601SharedControlOutput:
    """Compute bounded policy assist for one owner tick.

    Explicit or detected override has immediate priority and intentionally
    bypasses the assist slew limiter while dropping to zero.  This module does
    not compute gravity, passive damping, soft limits or final command torque.
    """

    if not isinstance(config, B601SharedControlConfig):
        raise TypeError("config must be a B601SharedControlConfig")
    if not isinstance(state, B601SharedControlState):
        raise TypeError("state must be a B601SharedControlState")
    if not isinstance(command_sample, B601PolicyCommandSample):
        raise TypeError("command_sample must be a B601PolicyCommandSample")
    if not isinstance(policy_clutch, bool):
        raise TypeError("policy_clutch must be a bool")
    if not isinstance(human_override_detected, bool):
        raise TypeError("human_override_detected must be a bool")
    qpos = _six_finite(measured_qpos_rad, name="measured_qpos_rad")
    qvel = _six_finite(measured_qvel_rad_s, name="measured_qvel_rad_s")
    dt = _positive(dt_sec, name="dt_sec")
    command = command_sample.command if command_sample.accepted else None
    observed_sequence = (
        command.sequence if command is not None else state.last_policy_sequence
    )

    if policy_clutch or human_override_detected:
        return _zero_output(
            mode=B601SharedControlMode.HUMAN_OVERRIDE,
            reason=("policy_clutch" if policy_clutch else "human_override_detected"),
            last_policy_sequence=observed_sequence,
            resume_after_sequence=observed_sequence,
        )
    if command is None:
        waiting_for_rebase = state.mode in {
            B601SharedControlMode.HUMAN_OVERRIDE,
            B601SharedControlMode.RECOVERY,
        } or state.resume_after_sequence is not None
        return _zero_output(
            mode=(
                B601SharedControlMode.RECOVERY
                if waiting_for_rebase
                else B601SharedControlMode.PASSIVE
            ),
            reason=("awaiting_rebased_policy" if waiting_for_rebase else command_sample.reason),
            last_policy_sequence=state.last_policy_sequence,
            resume_after_sequence=state.resume_after_sequence,
        )

    resume_after = state.resume_after_sequence
    recovering = state.mode in {
        B601SharedControlMode.HUMAN_OVERRIDE,
        B601SharedControlMode.RECOVERY,
    } or resume_after is not None
    if resume_after is not None and command.sequence <= resume_after:
        return _zero_output(
            mode=B601SharedControlMode.RECOVERY,
            reason="awaiting_rebased_policy",
            last_policy_sequence=command.sequence,
            resume_after_sequence=resume_after,
        )

    raw_error = tuple(
        target - measured
        for target, measured in zip(command.qpos_target_rad, qpos, strict=True)
    )
    bounded_error = tuple(
        max(-limit, min(limit, error))
        for error, limit in zip(
            raw_error,
            config.max_target_error_rad,
            strict=True,
        )
    )
    raw_torque = tuple(
        kp * error + kd * (target_velocity - measured_velocity)
        for kp, kd, error, target_velocity, measured_velocity in zip(
            config.position_kp_nm_per_rad,
            config.velocity_kd_nm_s_per_rad,
            bounded_error,
            command.qvel_target_rad_s,
            qvel,
            strict=True,
        )
    )
    limited_torque = tuple(
        max(-limit, min(limit, torque))
        for torque, limit in zip(
            raw_torque,
            config.max_abs_assist_torque_nm,
            strict=True,
        )
    )
    authority = min(1.0, state.authority + config.authority_rise_per_sec * dt)
    authority_torque = tuple(authority * torque for torque in limited_torque)
    max_delta = tuple(rate * dt for rate in config.max_assist_torque_rate_nm_s)
    assist = tuple(
        max(previous - delta, min(previous + delta, requested))
        for previous, delta, requested in zip(
            state.previous_assist_torque_nm,
            max_delta,
            authority_torque,
            strict=True,
        )
    )
    mode = (
        B601SharedControlMode.RECOVERY
        if recovering and authority < 1.0
        else B601SharedControlMode.POLICY_ASSIST
    )
    next_state = B601SharedControlState(
        mode=mode,
        authority=authority,
        previous_assist_torque_nm=assist,
        last_policy_sequence=command.sequence,
        resume_after_sequence=None,
    )
    return B601SharedControlOutput(
        state=next_state,
        assist_torque_nm=assist,
        raw_target_error_rad=raw_error,
        bounded_target_error_rad=bounded_error,
        raw_assist_torque_nm=raw_torque,
        reason="recovery_ramp" if mode is B601SharedControlMode.RECOVERY else "policy_assist",
        target_error_limited=raw_error != bounded_error,
        torque_limited=raw_torque != limited_torque,
        slew_limited=authority_torque != assist,
    )


__all__ = [
    "B601_SHARED_POLICY_FRAME",
    "B601PolicyAssistCommand",
    "B601PolicyCommandBuffer",
    "B601PolicyCommandReceipt",
    "B601PolicyCommandSample",
    "B601SharedControlConfig",
    "B601SharedControlMode",
    "B601SharedControlOutput",
    "B601SharedControlState",
    "compute_b601_shared_control_step",
]
