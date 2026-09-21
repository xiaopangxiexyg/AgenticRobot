"""Pure policy-assist commissioning state machines.

The state machines own no hardware and send no motor command.  Joint assist
publishes fresh, short-lived intent into :class:`B601SharedControlRuntime`.
Cartesian assist captures a measured TCP position and exposes one bounded,
world-aligned equilibrium target for the existing ``J.T @ wrench`` path.  Both
remain inside the existing gravity/damping owner and permanently withdraw the
reference after one trial; no return-to-anchor target is ever generated.
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
_DEFAULT_MINIMUM_RESPONSE_RAD = math.radians(0.1)
_DEFAULT_MAXIMUM_RESPONSE_RATIO = 1.5
_DEFAULT_MAXIMUM_SUPPORT_DRIFT_RAD = math.radians(0.5)
_DEFAULT_MINIMUM_ASSIST_DELIVERY_RATIO = 0.9
_CARTESIAN_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_MAX_CARTESIAN_COMMISSIONING_OFFSET_M = 0.05
_DEFAULT_CARTESIAN_MINIMUM_RESPONSE_M = 0.001
_DEFAULT_CARTESIAN_MAXIMUM_RESPONSE_RATIO = 1.5
_DEFAULT_CARTESIAN_MAXIMUM_LATERAL_DRIFT_M = 0.005


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


def _three(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(_finite(value, name=name) for value in values)
    if len(normalized) != 3:
        raise ValueError(f"{name} must contain three values")
    return normalized


def _pose_position(
    pose: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[float, ...]:
    rows = tuple(tuple(_finite(value, name=name) for value in row) for row in pose)
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"{name} must be a 4x4 transform")
    if any(abs(rows[3][index] - expected) > 1e-9 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f"{name} must have homogeneous final row")
    return rows[0][3], rows[1][3], rows[2][3]


def _pose_rotation(
    pose: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(_finite(value, name=name) for value in row) for row in pose)
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"{name} must be a 4x4 transform")
    return tuple(tuple(row[column] for column in range(3)) for row in rows[:3])


def _rotation_distance_rad(
    current: Sequence[Sequence[float]],
    anchor: Sequence[Sequence[float]],
) -> float:
    # trace(R_current * R_anchor.T) is invariant to the selected basis.
    trace = sum(
        current[row][column] * anchor[row][column]
        for row in range(3)
        for column in range(3)
    )
    cosine = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    return math.acos(cosine)


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

    @property
    def maximum_static_assist_torque_nm(self) -> float:
        """Maximum target-direction torque reachable from rest.

        The explicit torque cap alone can overstate the applied assist when
        ``Kp * max_target_error`` is smaller.  Commissioning reports and
        acceptance use this effective value so an inert parameter change is
        never mistaken for a stronger trial.
        """

        return min(
            self.max_abs_assist_torque_nm,
            self.position_kp_nm_per_rad * self.max_target_error_rad,
        )

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
class B601AssistResponseCriteria:
    """Mechanical evidence required to accept a real assist trial.

    Completing the command state machine is intentionally insufficient: a
    commissioning run must show a measurable, correctly directed response
    while the other enabled arm joints remain bounded.
    """

    minimum_response_rad: float = _DEFAULT_MINIMUM_RESPONSE_RAD
    maximum_response_ratio: float = _DEFAULT_MAXIMUM_RESPONSE_RATIO
    maximum_support_drift_rad: float = _DEFAULT_MAXIMUM_SUPPORT_DRIFT_RAD
    minimum_assist_delivery_ratio: float = _DEFAULT_MINIMUM_ASSIST_DELIVERY_RATIO

    def __post_init__(self) -> None:
        for name in (
            "minimum_response_rad",
            "maximum_response_ratio",
            "maximum_support_drift_rad",
            "minimum_assist_delivery_ratio",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        if self.minimum_response_rad > _MAX_COMMISSIONING_OFFSET_RAD:
            raise ValueError("minimum_response_rad must be within 20 degrees")
        if self.maximum_response_ratio < 1.0:
            raise ValueError("maximum_response_ratio must be >= 1")
        if self.minimum_assist_delivery_ratio > 1.0:
            raise ValueError("minimum_assist_delivery_ratio must be <= 1")


@dataclass(frozen=True)
class B601CartesianAssistConfig:
    """Limits for one world-aligned, translation-only assist trial."""

    axis: str = "x"
    offset_m: float = 0.01
    stiffness_n_per_m: float = 50.0
    damping_n_s_per_m: float = 3.0
    lateral_stiffness_n_per_m: float = 30.0
    lateral_damping_n_s_per_m: float = 2.0
    max_abs_force_n: float = 0.5
    max_abs_joint_torque_nm: float = 0.15
    max_joint_torque_rate_nm_s: float = 0.10
    authority_rise_per_sec: float = 0.5
    countdown_sec: float = 3.0
    assist_duration_sec: float = 6.0
    stationarity_limit_rad_s: float = 0.02

    def __post_init__(self) -> None:
        axis = str(self.axis).strip().lower()
        if axis not in _CARTESIAN_AXIS_INDEX:
            raise ValueError("axis must be x, y, or z")
        object.__setattr__(self, "axis", axis)
        offset = _finite(self.offset_m, name="offset_m")
        if offset == 0.0 or abs(offset) > _MAX_CARTESIAN_COMMISSIONING_OFFSET_M:
            raise ValueError("offset_m must be non-zero and within 0.05 m")
        object.__setattr__(self, "offset_m", offset)
        for name in (
            "stiffness_n_per_m",
            "max_abs_force_n",
            "max_abs_joint_torque_nm",
            "max_joint_torque_rate_nm_s",
            "authority_rise_per_sec",
            "countdown_sec",
            "assist_duration_sec",
            "stationarity_limit_rad_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        damping = _finite(self.damping_n_s_per_m, name="damping_n_s_per_m")
        if damping < 0.0:
            raise ValueError("damping_n_s_per_m must be non-negative")
        object.__setattr__(self, "damping_n_s_per_m", damping)
        for name in (
            "lateral_stiffness_n_per_m",
            "lateral_damping_n_s_per_m",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    @property
    def axis_index(self) -> int:
        return _CARTESIAN_AXIS_INDEX[self.axis]

    @property
    def direction(self) -> int:
        return 1 if self.offset_m > 0.0 else -1

    @property
    def maximum_static_force_n(self) -> float:
        return min(
            self.max_abs_force_n,
            self.stiffness_n_per_m * abs(self.offset_m),
        )


@dataclass(frozen=True)
class B601CartesianAssistResponseCriteria:
    """Measured task-space evidence required for Cartesian acceptance."""

    minimum_response_m: float = _DEFAULT_CARTESIAN_MINIMUM_RESPONSE_M
    maximum_response_ratio: float = _DEFAULT_CARTESIAN_MAXIMUM_RESPONSE_RATIO
    maximum_lateral_drift_m: float = _DEFAULT_CARTESIAN_MAXIMUM_LATERAL_DRIFT_M
    maximum_rotation_drift_rad: float = math.radians(5.0)
    minimum_assist_delivery_ratio: float = _DEFAULT_MINIMUM_ASSIST_DELIVERY_RATIO

    def __post_init__(self) -> None:
        for name in (
            "minimum_response_m",
            "maximum_response_ratio",
            "maximum_lateral_drift_m",
            "maximum_rotation_drift_rad",
            "minimum_assist_delivery_ratio",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        if self.minimum_response_m > _MAX_CARTESIAN_COMMISSIONING_OFFSET_M:
            raise ValueError("minimum_response_m must be within 0.05 m")
        if self.maximum_response_ratio < 1.0:
            raise ValueError("maximum_response_ratio must be >= 1")
        if self.minimum_assist_delivery_ratio > 1.0:
            raise ValueError("minimum_assist_delivery_ratio must be <= 1")


def evaluate_single_joint_assist_response(
    *,
    config: B601SingleJointAssistConfig,
    state: B601AssistCommissioningState,
    anchor_qpos_rad: Sequence[float] | None,
    measured_qpos_samples_rad: Sequence[Sequence[float]],
    assist_torque_samples_nm: Sequence[Sequence[float]],
    active_joint_names: Sequence[str] = B601_RS_ARM_JOINT_NAMES,
    criteria: B601AssistResponseCriteria | None = None,
) -> dict[str, object]:
    """Classify real mechanical response separately from command completion."""

    acceptance = criteria or B601AssistResponseCriteria()
    if state is not B601AssistCommissioningState.COMPLETE:
        return {
            "status": "INCOMPLETE",
            "passed": False,
            "commissioning_state": state.value,
            "sample_count": len(measured_qpos_samples_rad),
        }
    if anchor_qpos_rad is None:
        return {
            "status": "NO_ANCHOR",
            "passed": False,
            "commissioning_state": state.value,
            "sample_count": len(measured_qpos_samples_rad),
        }
    positions = tuple(
        _six(sample, name="measured_qpos_sample_rad")
        for sample in measured_qpos_samples_rad
    )
    assist_torques = tuple(
        _six(sample, name="assist_torque_sample_nm")
        for sample in assist_torque_samples_nm
    )
    if len(positions) != len(assist_torques):
        raise ValueError("position and assist-torque sample counts must match")
    if not positions:
        return {
            "status": "NO_ACTIVE_SAMPLES",
            "passed": False,
            "commissioning_state": state.value,
            "sample_count": 0,
        }

    anchor = _six(anchor_qpos_rad, name="anchor_qpos_rad")
    target_index = config.joint_index
    direction = 1 if config.offset_rad > 0.0 else -1
    tail_size = min(5, len(positions))
    settled_qpos = tuple(
        sum(sample[index] for sample in positions[-tail_size:]) / tail_size
        for index in range(6)
    )
    settled_signed_displacement = direction * (
        settled_qpos[target_index] - anchor[target_index]
    )
    peak_signed_displacement = max(
        direction * (sample[target_index] - anchor[target_index])
        for sample in positions
    )
    max_abs_target_excursion = max(
        abs(sample[target_index] - anchor[target_index]) for sample in positions
    )
    peak_signed_assist_torque = max(
        direction * sample[target_index] for sample in assist_torques
    )
    expected_peak_assist_torque = config.maximum_static_assist_torque_nm
    assist_delivery_ratio = max(0.0, peak_signed_assist_torque) / (
        expected_peak_assist_torque
    )
    support_names = tuple(
        name
        for name in active_joint_names
        if name in B601_RS_ARM_JOINT_NAMES and name != config.joint_name
    )
    support_peak_drift = {
        name: max(
            abs(
                sample[B601_RS_ARM_JOINT_NAMES.index(name)]
                - anchor[B601_RS_ARM_JOINT_NAMES.index(name)]
            )
            for sample in positions
        )
        for name in support_names
    }
    max_support_drift = max(support_peak_drift.values(), default=0.0)
    maximum_response_rad = min(
        _MAX_COMMISSIONING_OFFSET_RAD,
        abs(config.offset_rad) * acceptance.maximum_response_ratio,
    )

    if max_abs_target_excursion > _MAX_COMMISSIONING_OFFSET_RAD:
        status = "HARD_EXCURSION_LIMIT_EXCEEDED"
    elif max_abs_target_excursion > maximum_response_rad:
        status = "EXCESSIVE_RESPONSE"
    elif max_support_drift > acceptance.maximum_support_drift_rad:
        status = "SUPPORT_DRIFT_EXCEEDED"
    elif assist_delivery_ratio < acceptance.minimum_assist_delivery_ratio:
        status = "ASSIST_NOT_DELIVERED"
    elif settled_signed_displacement < -acceptance.minimum_response_rad:
        status = "WRONG_DIRECTION"
    elif settled_signed_displacement < acceptance.minimum_response_rad:
        status = "INSUFFICIENT_RESPONSE"
    else:
        status = "PASS"

    return {
        "status": status,
        "passed": status == "PASS",
        "commissioning_state": state.value,
        "sample_count": len(positions),
        "settled_sample_count": tail_size,
        "target_joint": config.joint_name,
        "direction": direction,
        "commanded_offset_rad": config.offset_rad,
        "settled_signed_displacement_rad": settled_signed_displacement,
        "peak_signed_displacement_rad": peak_signed_displacement,
        "max_abs_target_excursion_rad": max_abs_target_excursion,
        "minimum_response_rad": acceptance.minimum_response_rad,
        "maximum_response_rad": maximum_response_rad,
        "configured_max_assist_torque_nm": config.max_abs_assist_torque_nm,
        "expected_peak_static_assist_torque_nm": expected_peak_assist_torque,
        "peak_signed_assist_torque_nm": peak_signed_assist_torque,
        "assist_delivery_ratio": assist_delivery_ratio,
        "minimum_assist_delivery_ratio": (
            acceptance.minimum_assist_delivery_ratio
        ),
        "support_joint_names": support_names,
        "support_peak_drift_rad": support_peak_drift,
        "max_support_drift_rad": max_support_drift,
        "max_support_drift_limit_rad": acceptance.maximum_support_drift_rad,
    }


def evaluate_cartesian_assist_response(
    *,
    config: B601CartesianAssistConfig,
    state: B601AssistCommissioningState,
    anchor_position_m: Sequence[float] | None,
    measured_position_samples_m: Sequence[Sequence[float]],
    force_samples_n: Sequence[Sequence[float]],
    rotation_drift_samples_rad: Sequence[float],
    criteria: B601CartesianAssistResponseCriteria | None = None,
) -> dict[str, object]:
    """Classify a task-space response without inferring success from commands."""

    acceptance = criteria or B601CartesianAssistResponseCriteria()
    if state is not B601AssistCommissioningState.COMPLETE:
        return {
            "status": "INCOMPLETE",
            "passed": False,
            "commissioning_state": state.value,
            "sample_count": len(measured_position_samples_m),
        }
    if anchor_position_m is None:
        return {
            "status": "NO_ANCHOR",
            "passed": False,
            "commissioning_state": state.value,
            "sample_count": len(measured_position_samples_m),
        }
    positions = tuple(
        _three(sample, name="measured_position_sample_m")
        for sample in measured_position_samples_m
    )
    forces = tuple(_three(sample, name="force_sample_n") for sample in force_samples_n)
    rotation_drifts = tuple(
        _finite(value, name="rotation_drift_sample_rad")
        for value in rotation_drift_samples_rad
    )
    if len(positions) != len(forces) or len(positions) != len(rotation_drifts):
        raise ValueError("position, force and rotation sample counts must match")
    if not positions:
        return {
            "status": "NO_ACTIVE_SAMPLES",
            "passed": False,
            "commissioning_state": state.value,
            "sample_count": 0,
        }

    anchor = _three(anchor_position_m, name="anchor_position_m")
    axis_index = config.axis_index
    lateral_indices = tuple(index for index in range(3) if index != axis_index)
    tail_size = min(5, len(positions))
    settled = tuple(
        sum(sample[index] for sample in positions[-tail_size:]) / tail_size
        for index in range(3)
    )
    direction = config.direction
    settled_signed_displacement = direction * (
        settled[axis_index] - anchor[axis_index]
    )
    signed_displacements = tuple(
        direction * (sample[axis_index] - anchor[axis_index])
        for sample in positions
    )
    peak_signed_displacement = max(signed_displacements)
    max_abs_axis_excursion = max(abs(value) for value in signed_displacements)
    lateral_drift = tuple(
        math.sqrt(
            sum(
                (sample[index] - anchor[index]) ** 2
                for index in lateral_indices
            )
        )
        for sample in positions
    )
    max_lateral_drift = max(lateral_drift, default=0.0)
    max_rotation_drift = max(rotation_drifts, default=0.0)
    peak_signed_force = max(direction * sample[axis_index] for sample in forces)
    expected_peak_force = config.maximum_static_force_n
    assist_delivery_ratio = max(0.0, peak_signed_force) / expected_peak_force
    maximum_response_m = min(
        _MAX_CARTESIAN_COMMISSIONING_OFFSET_M,
        abs(config.offset_m) * acceptance.maximum_response_ratio,
    )

    if max_abs_axis_excursion > _MAX_CARTESIAN_COMMISSIONING_OFFSET_M:
        status = "HARD_EXCURSION_LIMIT_EXCEEDED"
    elif max_abs_axis_excursion > maximum_response_m:
        status = "EXCESSIVE_RESPONSE"
    elif max_lateral_drift > acceptance.maximum_lateral_drift_m:
        status = "LATERAL_DRIFT_EXCEEDED"
    elif max_rotation_drift > acceptance.maximum_rotation_drift_rad:
        status = "ROTATION_DRIFT_EXCEEDED"
    elif assist_delivery_ratio < acceptance.minimum_assist_delivery_ratio:
        status = "ASSIST_NOT_DELIVERED"
    elif settled_signed_displacement < -acceptance.minimum_response_m:
        status = "WRONG_DIRECTION"
    elif settled_signed_displacement < acceptance.minimum_response_m:
        status = "INSUFFICIENT_RESPONSE"
    else:
        status = "PASS"

    return {
        "status": status,
        "passed": status == "PASS",
        "commissioning_state": state.value,
        "sample_count": len(positions),
        "settled_sample_count": tail_size,
        "axis": config.axis,
        "direction": direction,
        "commanded_offset_m": config.offset_m,
        "settled_signed_displacement_m": settled_signed_displacement,
        "peak_signed_displacement_m": peak_signed_displacement,
        "max_abs_axis_excursion_m": max_abs_axis_excursion,
        "minimum_response_m": acceptance.minimum_response_m,
        "maximum_response_m": maximum_response_m,
        "max_lateral_drift_m": max_lateral_drift,
        "max_lateral_drift_limit_m": acceptance.maximum_lateral_drift_m,
        "max_rotation_drift_rad": max_rotation_drift,
        "max_rotation_drift_limit_rad": acceptance.maximum_rotation_drift_rad,
        "configured_max_force_n": config.max_abs_force_n,
        "expected_peak_static_force_n": expected_peak_force,
        "peak_signed_force_n": peak_signed_force,
        "assist_delivery_ratio": assist_delivery_ratio,
        "minimum_assist_delivery_ratio": acceptance.minimum_assist_delivery_ratio,
    }


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

    def validate_active_motion(
        self,
        measured_qpos_rad: Sequence[float],
        *,
        criteria: B601AssistResponseCriteria,
    ) -> None:
        """Fail before another assist command if observed motion is unsafe."""

        if self.state is not B601AssistCommissioningState.ACTIVE:
            return
        if self.anchor_qpos_rad is None:
            raise ValueError("active assist has no measured anchor")
        qpos = _six(measured_qpos_rad, name="measured_qpos_rad")
        direction = 1 if self.config.offset_rad > 0.0 else -1
        target_displacement = (
            qpos[self.config.joint_index]
            - self.anchor_qpos_rad[self.config.joint_index]
        )
        signed_displacement = direction * target_displacement
        maximum_response = min(
            _MAX_COMMISSIONING_OFFSET_RAD,
            abs(self.config.offset_rad) * criteria.maximum_response_ratio,
        )
        if abs(target_displacement) > maximum_response:
            raise ValueError(
                f"{self.config.joint_name} excursion "
                f"{abs(target_displacement):.6f}rad exceeds "
                f"{maximum_response:.6f}rad"
            )
        if signed_displacement < -criteria.minimum_response_rad:
            raise ValueError(
                f"{self.config.joint_name} moved in the wrong direction by "
                f"{abs(signed_displacement):.6f}rad"
            )
        support_drift = max(
            (
                abs(current - anchor)
                for index, (current, anchor) in enumerate(
                    zip(qpos, self.anchor_qpos_rad, strict=True)
                )
                if index != self.config.joint_index
            ),
            default=0.0,
        )
        if support_drift > criteria.maximum_support_drift_rad:
            raise ValueError(
                "support-joint drift "
                f"{support_drift:.6f}rad exceeds "
                f"{criteria.maximum_support_drift_rad:.6f}rad"
            )

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


class B601CartesianAssistSession:
    """Own one measured-anchor Cartesian target without owning hardware."""

    def __init__(self, config: B601CartesianAssistConfig) -> None:
        self.config = config
        self.state = B601AssistCommissioningState.WAITING
        self.anchor_position_m: tuple[float, ...] | None = None
        self.anchor_rotation: tuple[tuple[float, ...], ...] | None = None
        self.target_position_m: tuple[float, ...] | None = None
        self.armed_at_sec: float | None = None
        self.active_at_sec: float | None = None
        self.completed_at_sec: float | None = None
        self._last_countdown_value: int | None = None
        self.events: list[B601AssistCommissioningEvent] = []

    def _event(
        self,
        event: str,
        timestamp_sec: float,
        detail: str,
    ) -> B601AssistCommissioningEvent:
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
        measured_pose: Sequence[Sequence[float]],
        measured_qvel_rad_s: Sequence[float],
        timestamp_sec: float,
    ) -> B601AssistCommissioningEvent:
        if self.state is not B601AssistCommissioningState.WAITING:
            raise ValueError("Cartesian assist can be armed only once per run")
        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        qvel = _six(measured_qvel_rad_s, name="measured_qvel_rad_s")
        max_abs_qvel = max(abs(value) for value in qvel)
        if max_abs_qvel > self.config.stationarity_limit_rad_s:
            raise ValueError(
                "arm is not stationary enough to capture Cartesian anchor: "
                f"max_abs_qvel={max_abs_qvel:.6f} > "
                f"{self.config.stationarity_limit_rad_s:.6f} rad/s"
            )
        anchor = _pose_position(measured_pose, name="measured_pose")
        anchor_rotation = _pose_rotation(measured_pose, name="measured_pose")
        target = list(anchor)
        target[self.config.axis_index] += self.config.offset_m
        self.anchor_position_m = anchor
        self.anchor_rotation = anchor_rotation
        self.target_position_m = tuple(target)
        self.armed_at_sec = timestamp
        self._last_countdown_value = math.ceil(self.config.countdown_sec)
        self.state = B601AssistCommissioningState.COUNTDOWN
        return self._event(
            "ARMED",
            timestamp,
            "fresh stationary measured TCP captured; Cartesian assist remains zero; "
            f"countdown={self._last_countdown_value}",
        )

    def cancel(
        self,
        *,
        timestamp_sec: float,
        detail: str,
    ) -> B601AssistCommissioningEvent:
        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        self.state = B601AssistCommissioningState.CANCELLED
        self.completed_at_sec = timestamp
        return self._event("CANCELLED", timestamp, detail)

    def step(
        self,
        *,
        measured_qvel_rad_s: Sequence[float],
        timestamp_sec: float,
    ) -> tuple[B601AssistCommissioningEvent, ...]:
        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        qvel = _six(measured_qvel_rad_s, name="measured_qvel_rad_s")
        emitted: list[B601AssistCommissioningEvent] = []
        if self.state is B601AssistCommissioningState.COUNTDOWN:
            if max(abs(value) for value in qvel) > self.config.stationarity_limit_rad_s:
                self.state = B601AssistCommissioningState.WAITING
                self.anchor_position_m = None
                self.anchor_rotation = None
                self.target_position_m = None
                self.armed_at_sec = None
                self._last_countdown_value = None
                emitted.append(
                    self._event(
                        "COUNTDOWN_REJECTED",
                        timestamp,
                        "motion detected before Cartesian assist; settle and press A again",
                    )
                )
                return tuple(emitted)
            assert self.armed_at_sec is not None
            remaining_sec = self.config.countdown_sec - (
                timestamp - self.armed_at_sec
            )
            countdown_value = max(0, math.ceil(remaining_sec))
            if countdown_value > 0 and countdown_value != self._last_countdown_value:
                self._last_countdown_value = countdown_value
                emitted.append(
                    self._event(
                        "COUNTDOWN",
                        timestamp,
                        f"Cartesian assist starts in {countdown_value}",
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
                        "bounded Cartesian assist started from captured TCP",
                    )
                )
        if self.state is B601AssistCommissioningState.ACTIVE:
            assert self.active_at_sec is not None
            if timestamp - self.active_at_sec >= self.config.assist_duration_sec:
                self.state = B601AssistCommissioningState.COMPLETE
                self.completed_at_sec = timestamp
                emitted.append(
                    self._event(
                        "COMPLETE",
                        timestamp,
                        "Cartesian assist withdrawn; no return target; gravity and damping remain",
                    )
                )
        return tuple(emitted)

    def authority(self, *, timestamp_sec: float) -> float:
        timestamp = _finite(timestamp_sec, name="timestamp_sec")
        if self.state is not B601AssistCommissioningState.ACTIVE:
            return 0.0
        assert self.active_at_sec is not None
        return min(
            1.0,
            max(0.0, timestamp - self.active_at_sec)
            * self.config.authority_rise_per_sec,
        )

    def validate_active_position(
        self,
        measured_position_m: Sequence[float],
        *,
        criteria: B601CartesianAssistResponseCriteria,
    ) -> None:
        if self.state is not B601AssistCommissioningState.ACTIVE:
            return
        if self.anchor_position_m is None:
            raise ValueError("active Cartesian assist has no measured anchor")
        position = _three(measured_position_m, name="measured_position_m")
        delta = tuple(
            current - anchor
            for current, anchor in zip(position, self.anchor_position_m, strict=True)
        )
        signed_displacement = self.config.direction * delta[self.config.axis_index]
        maximum_response = min(
            _MAX_CARTESIAN_COMMISSIONING_OFFSET_M,
            abs(self.config.offset_m) * criteria.maximum_response_ratio,
        )
        if abs(delta[self.config.axis_index]) > maximum_response:
            raise ValueError(
                f"Cartesian {self.config.axis} excursion "
                f"{abs(delta[self.config.axis_index]):.6f}m exceeds "
                f"{maximum_response:.6f}m"
            )
        if signed_displacement < -criteria.minimum_response_m:
            raise ValueError(
                f"Cartesian {self.config.axis} moved in the wrong direction by "
                f"{abs(signed_displacement):.6f}m"
            )
        lateral_drift = math.sqrt(
            sum(
                delta[index] ** 2
                for index in range(3)
                if index != self.config.axis_index
            )
        )
        if lateral_drift > criteria.maximum_lateral_drift_m:
            raise ValueError(
                f"Cartesian lateral drift {lateral_drift:.6f}m exceeds "
                f"{criteria.maximum_lateral_drift_m:.6f}m"
            )

    def rotation_drift_rad(
        self,
        measured_pose: Sequence[Sequence[float]],
    ) -> float:
        if self.anchor_rotation is None:
            raise ValueError("Cartesian assist has no measured anchor rotation")
        return _rotation_distance_rad(
            _pose_rotation(measured_pose, name="measured_pose"),
            self.anchor_rotation,
        )

    def validate_active_pose(
        self,
        measured_pose: Sequence[Sequence[float]],
        *,
        criteria: B601CartesianAssistResponseCriteria,
    ) -> None:
        if self.state is not B601AssistCommissioningState.ACTIVE:
            return
        self.validate_active_position(
            _pose_position(measured_pose, name="measured_pose"),
            criteria=criteria,
        )
        rotation_drift = self.rotation_drift_rad(measured_pose)
        if rotation_drift > criteria.maximum_rotation_drift_rad:
            raise ValueError(
                f"Cartesian orientation drift {rotation_drift:.6f}rad exceeds "
                f"{criteria.maximum_rotation_drift_rad:.6f}rad"
            )

    def audit(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "axis": self.config.axis,
            "offset_m": self.config.offset_m,
            "anchor_position_m": self.anchor_position_m,
            "anchor_rotation": self.anchor_rotation,
            "target_position_m": self.target_position_m,
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
    "B601AssistResponseCriteria",
    "B601CartesianAssistConfig",
    "B601CartesianAssistResponseCriteria",
    "B601CartesianAssistSession",
    "B601SingleJointAssistConfig",
    "B601SingleJointAssistSession",
    "evaluate_cartesian_assist_response",
    "evaluate_single_joint_assist_response",
]
