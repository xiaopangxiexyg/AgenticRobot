"""Pure single-joint policy-assist commissioning state machine.

The state machine owns no hardware and sends no motor command.  It publishes
fresh, short-lived joint intent into :class:`B601SharedControlRuntime`, which
must remain inside the existing gravity/damping hardware owner.  The reference
is captured from admitted measured state and is withdrawn after one bounded
trial; no return-to-anchor target is ever generated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .model import B601_RS_ARM_JOINT_NAMES
from .shared_control import B601PolicyAssistCommand, B601SharedControlConfig
from .shared_control_runtime import B601SharedControlRuntime

_ZERO_SIX = (0.0,) * 6
_MAX_COMMISSIONING_OFFSET_RAD = math.radians(20.0)


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


def _six(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(_finite(value, name=name) for value in values)
    if len(normalized) != 6:
        raise ValueError(f"{name} must contain six values")
    return normalized


class B601AssistCommissioningState(str, Enum):
    WAITING = "waiting"
    COUNTDOWN = "countdown"
    ACTIVE = "active"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class B601SingleJointAssistConfig:
    """Limits for one relative, non-returning joint-assist trial."""

    joint_name: str = "joint6"
    offset_rad: float = 0.10
    position_kp_nm_per_rad: float = 1.0
    velocity_kd_nm_s_per_rad: float = 0.0
    max_target_error_rad: float = 0.05
    max_abs_assist_torque_nm: float = 0.05
    max_assist_torque_rate_nm_s: float = 0.10
    authority_rise_per_sec: float = 0.5
    command_stale_timeout_sec: float = 0.20
    countdown_sec: float = 3.0
    assist_duration_sec: float = 4.0
    stationarity_limit_rad_s: float = 0.02
    maximum_future_skew_sec: float = 0.01

    def __post_init__(self) -> None:
        if self.joint_name not in B601_RS_ARM_JOINT_NAMES:
            raise ValueError("joint_name must be one of joint1..joint6")
        offset = _finite(self.offset_rad, name="offset_rad")
        if offset == 0.0 or abs(offset) > _MAX_COMMISSIONING_OFFSET_RAD:
            raise ValueError("offset_rad must be non-zero and within 20 degrees")
        object.__setattr__(self, "offset_rad", offset)
        for name in (
            "position_kp_nm_per_rad",
            "max_target_error_rad",
            "max_abs_assist_torque_nm",
            "max_assist_torque_rate_nm_s",
            "authority_rise_per_sec",
            "command_stale_timeout_sec",
            "countdown_sec",
            "assist_duration_sec",
            "stationarity_limit_rad_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        damping = _finite(
            self.velocity_kd_nm_s_per_rad,
            name="velocity_kd_nm_s_per_rad",
        )
        if damping < 0.0:
            raise ValueError("velocity_kd_nm_s_per_rad must be non-negative")
        object.__setattr__(self, "velocity_kd_nm_s_per_rad", damping)
        future_skew = _finite(
            self.maximum_future_skew_sec,
            name="maximum_future_skew_sec",
        )
        if future_skew < 0.0:
            raise ValueError("maximum_future_skew_sec must be non-negative")
        object.__setattr__(self, "maximum_future_skew_sec", future_skew)

    @property
    def joint_index(self) -> int:
        return B601_RS_ARM_JOINT_NAMES.index(self.joint_name)

    def shared_control_config(self) -> B601SharedControlConfig:
        """Build the existing bounded assist contract without a second path."""

        def target_only(value: float) -> tuple[float, ...]:
            values = [0.0] * 6
            values[self.joint_index] = value
            return tuple(values)

        # The shared-control contract requires positive safety envelopes on
        # every axis.  Non-target axes have zero gains and therefore produce
        # zero assist even though they retain the same finite envelope.
        envelope = (self.max_target_error_rad,) * 6
        torque_envelope = (self.max_abs_assist_torque_nm,) * 6
        slew_envelope = (self.max_assist_torque_rate_nm_s,) * 6
        return B601SharedControlConfig(
            position_kp_nm_per_rad=target_only(self.position_kp_nm_per_rad),
            velocity_kd_nm_s_per_rad=target_only(
                self.velocity_kd_nm_s_per_rad
            ),
            max_target_error_rad=envelope,
            max_abs_assist_torque_nm=torque_envelope,
            max_assist_torque_rate_nm_s=slew_envelope,
            authority_rise_per_sec=self.authority_rise_per_sec,
            policy_stale_timeout_sec=self.command_stale_timeout_sec,
            maximum_future_skew_sec=self.maximum_future_skew_sec,
        )


@dataclass(frozen=True)
class B601AssistCommissioningEvent:
    event: str
    state: B601AssistCommissioningState
    timestamp_sec: float
    detail: str


class B601SingleJointAssistSession:
    """Publish one bounded relative target, then permanently withdraw it."""

    def __init__(
        self,
        config: B601SingleJointAssistConfig,
        runtime: B601SharedControlRuntime,
    ) -> None:
        if runtime.config != config.shared_control_config():
            raise ValueError("runtime config does not match commissioning config")
        self.config = config
        self.runtime = runtime
        self.state = B601AssistCommissioningState.WAITING
        self.anchor_qpos_rad: tuple[float, ...] | None = None
        self.target_qpos_rad: tuple[float, ...] | None = None
        self.armed_at_sec: float | None = None
        self.active_at_sec: float | None = None
        self.completed_at_sec: float | None = None
        self._sequence = 0
        self._last_countdown_value: int | None = None
        self.events: list[B601AssistCommissioningEvent] = []

    def _event(self, event: str, timestamp_sec: float, detail: str) -> B601AssistCommissioningEvent:
        item = B601AssistCommissioningEvent(
            event=event,
            state=self.state,
            timestamp_sec=timestamp_sec,
            detail=detail,
        )
        self.events.append(item)
        return item

    def arm(
        self,
        *,
        measured_qpos_rad: Sequence[float],
        measured_qvel_rad_s: Sequence[float],
        position_limits_rad: Sequence[Sequence[float]],
        timestamp_sec: float,
    ) -> B601AssistCommissioningEvent:
        """Capture a fresh stationary anchor and start the no-assist countdown."""

        if self.state is not B601AssistCommissioningState.WAITING:
            raise ValueError("single-joint assist can be armed only once per run")
        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        qpos = _six(measured_qpos_rad, name="measured_qpos_rad")
        qvel = _six(measured_qvel_rad_s, name="measured_qvel_rad_s")
        if max(abs(value) for value in qvel) > self.config.stationarity_limit_rad_s:
            raise ValueError(
                "arm is not stationary enough to capture assist anchor: "
                f"max_abs_qvel={max(abs(value) for value in qvel):.6f} > "
                f"{self.config.stationarity_limit_rad_s:.6f} rad/s"
            )
        limits = tuple(tuple(float(value) for value in pair) for pair in position_limits_rad)
        if len(limits) != 6 or any(len(pair) != 2 for pair in limits):
            raise ValueError("position_limits_rad must contain six lower/upper pairs")
        target = list(qpos)
        target[self.config.joint_index] += self.config.offset_rad
        lower, upper = limits[self.config.joint_index]
        if not lower <= target[self.config.joint_index] <= upper:
            raise ValueError(
                f"{self.config.joint_name} assist target {target[self.config.joint_index]:.6f} "
                f"rad is outside [{lower:.6f}, {upper:.6f}]"
            )
        self.runtime.clear_policy()
        self.anchor_qpos_rad = qpos
        self.target_qpos_rad = tuple(target)
        self.armed_at_sec = timestamp
        self._last_countdown_value = math.ceil(self.config.countdown_sec)
        self.state = B601AssistCommissioningState.COUNTDOWN
        return self._event(
            "ARMED",
            timestamp,
            "fresh stationary measured pose captured; assist remains zero; "
            f"countdown={self._last_countdown_value}",
        )

    def cancel(self, *, timestamp_sec: float, detail: str) -> B601AssistCommissioningEvent:
        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        self.runtime.clear_policy()
        self.state = B601AssistCommissioningState.CANCELLED
        self.completed_at_sec = timestamp
        return self._event("CANCELLED", timestamp, detail)

    def step(
        self,
        *,
        measured_qpos_rad: Sequence[float],
        measured_qvel_rad_s: Sequence[float],
        timestamp_sec: float,
    ) -> tuple[B601AssistCommissioningEvent, ...]:
        """Advance the one-shot trial and refresh active intent at owner rate."""

        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        _six(measured_qpos_rad, name="measured_qpos_rad")
        qvel = _six(measured_qvel_rad_s, name="measured_qvel_rad_s")
        emitted: list[B601AssistCommissioningEvent] = []
        if self.state is B601AssistCommissioningState.COUNTDOWN:
            if max(abs(value) for value in qvel) > self.config.stationarity_limit_rad_s:
                self.runtime.clear_policy()
                self.state = B601AssistCommissioningState.WAITING
                self.anchor_qpos_rad = None
                self.target_qpos_rad = None
                self.armed_at_sec = None
                self._last_countdown_value = None
                emitted.append(
                    self._event(
                        "COUNTDOWN_REJECTED",
                        timestamp,
                        "motion detected before assist; settle and press A again",
                    )
                )
                return tuple(emitted)
            assert self.armed_at_sec is not None
            remaining_sec = self.config.countdown_sec - (
                timestamp - self.armed_at_sec
            )
            countdown_value = max(0, math.ceil(remaining_sec))
            if (
                countdown_value > 0
                and countdown_value != self._last_countdown_value
            ):
                self._last_countdown_value = countdown_value
                emitted.append(
                    self._event(
                        "COUNTDOWN",
                        timestamp,
                        f"assist starts in {countdown_value}",
                    )
                )
            if remaining_sec <= 0.0:
                self.state = B601AssistCommissioningState.ACTIVE
                self.active_at_sec = timestamp
                self._last_countdown_value = 0
                emitted.append(
                    self._event(
                        "ACTIVE",
                        timestamp,
                        "bounded assist started from captured measured pose",
                    )
                )
        if self.state is B601AssistCommissioningState.ACTIVE:
            assert self.active_at_sec is not None
            assert self.target_qpos_rad is not None
            if timestamp - self.active_at_sec >= self.config.assist_duration_sec:
                self.runtime.clear_policy()
                self.state = B601AssistCommissioningState.COMPLETE
                self.completed_at_sec = timestamp
                emitted.append(
                    self._event(
                        "COMPLETE",
                        timestamp,
                        "assist withdrawn; no return target; gravity and damping remain",
                    )
                )
            else:
                self._sequence += 1
                receipt = self.runtime.publish_policy(
                    B601PolicyAssistCommand(
                        sequence=self._sequence,
                        source_timestamp_sec=timestamp,
                        receive_timestamp_sec=timestamp,
                        qpos_target_rad=self.target_qpos_rad,
                        qvel_target_rad_s=_ZERO_SIX,
                    )
                )
                if not receipt.accepted:
                    raise RuntimeError(
                        f"commissioning policy command rejected: {receipt.reason}"
                    )
        return tuple(emitted)

    def audit(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "joint_name": self.config.joint_name,
            "offset_rad": self.config.offset_rad,
            "anchor_qpos_rad": self.anchor_qpos_rad,
            "target_qpos_rad": self.target_qpos_rad,
            "armed_at_sec": self.armed_at_sec,
            "active_at_sec": self.active_at_sec,
            "completed_at_sec": self.completed_at_sec,
            "events": tuple(
                {
                    "event": event.event,
                    "state": event.state.value,
                    "timestamp_sec": event.timestamp_sec,
                    "detail": event.detail,
                }
                for event in self.events
            ),
        }


__all__ = [
    "B601AssistCommissioningEvent",
    "B601AssistCommissioningState",
    "B601SingleJointAssistConfig",
    "B601SingleJointAssistSession",
]
