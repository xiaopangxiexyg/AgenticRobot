"""Versioned, backend-independent RM65 position-actuation profiles.

The profile deliberately separates the qpos transport contract from MuJoCo's
physical joint servo.  A real RM65 receives qpos targets, while its internal
motor loop is not exposed by the SDK; MuJoCo therefore needs an explicit,
fitted approximation instead of teleporting ``qpos``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence


_SIDES = ("left", "right")


def _six_finite(
    value: Sequence[float], *, name: str, strictly_positive: bool
) -> tuple[float, ...]:
    array = tuple(float(item) for item in value)
    if len(array) != 6 or not all(math.isfinite(item) for item in array):
        raise ValueError(f"{name} must contain six finite values.")
    if strictly_positive and any(item <= 0.0 for item in array):
        raise ValueError(f"{name} must contain six positive values.")
    if not strictly_positive and any(item < 0.0 for item in array):
        raise ValueError(f"{name} must contain six non-negative values.")
    return array


@dataclass(frozen=True)
class PositionActuationProfile:
    """Joint position-servo approximation for one or two RM65 arms.

    ``transport_delay_sec_by_side`` is the residual delay applied before the
    MuJoCo servo target.  It must not be interpreted as the full observed
    command-to-measured-state delay: the physical position actuator contributes
    additional phase. ``observed_effective_delay_sec_by_side_joint`` records
    the corresponding real full-path measurements for audit and validation.

    The torque, damping, and armature fields remain explicitly provisional
    until a current-to-torque / payload identification experiment is available.
    """

    name: str
    joint_names_by_side: Mapping[str, Sequence[str]]
    transport_delay_sec_by_side: Mapping[str, float]
    position_kp_nm_rad: Sequence[float]
    position_kv_nm_s_rad: Sequence[float]
    force_limit_nm: Sequence[float]
    observed_effective_delay_sec_by_side_joint: Mapping[
        str, Sequence[float]
    ] = field(default_factory=dict)
    transport_delay_sec_by_side_joint: Mapping[str, Sequence[float]] = field(
        default_factory=dict
    )
    velocity_lag_threshold_rad_s_by_side_joint: Mapping[
        str, Sequence[float]
    ] = field(default_factory=dict)
    velocity_lag_tau_gain_sec_per_rad_s_by_side_joint: Mapping[
        str, Sequence[float]
    ] = field(default_factory=dict)
    velocity_lag_max_tau_sec_by_side_joint: Mapping[
        str, Sequence[float]
    ] = field(default_factory=dict)
    joint_armature_kg_m2: Sequence[float] = (0.0,) * 6
    joint_damping_nm_s_rad: Sequence[float] = (0.0,) * 6
    joint_frictionloss_nm: Sequence[float] = (0.0,) * 6
    actuator_prefix: str = "rm65_position"
    calibration_status: str = "transport_delay_fitted_servo_torque_pending"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("PositionActuationProfile.name must be non-empty.")
        joints: dict[str, tuple[str, ...]] = {}
        for side, names in self.joint_names_by_side.items():
            normalized_side = str(side).strip()
            normalized_names = tuple(str(item).strip() for item in names)
            if (
                normalized_side not in _SIDES
                or len(normalized_names) != 6
                or not all(normalized_names)
            ):
                raise ValueError(
                    "joint_names_by_side must map left/right to six "
                    "non-empty joint names."
                )
            joints[normalized_side] = normalized_names
        if not joints:
            raise ValueError("joint_names_by_side must not be empty.")
        delays: dict[str, float] = {}
        for side in joints:
            try:
                delay = float(self.transport_delay_sec_by_side[side])
            except KeyError as exc:
                raise ValueError(f"transport delay missing for {side!r}.") from exc
            if not math.isfinite(delay) or delay < 0.0:
                raise ValueError("transport delays must be finite and non-negative.")
            delays[side] = delay
        observed_delays: dict[str, tuple[float, ...]] = {}
        for side, values in self.observed_effective_delay_sec_by_side_joint.items():
            normalized_side = str(side).strip()
            if normalized_side not in joints:
                raise ValueError(
                    "observed effective delays may contain only configured sides."
                )
            observed_delays[normalized_side] = _six_finite(
                values,
                name=f"observed_effective_delay_sec_by_side_joint[{normalized_side!r}]",
                strictly_positive=False,
            )
        joint_delay_fields = (
            "transport_delay_sec_by_side_joint",
            "velocity_lag_threshold_rad_s_by_side_joint",
            "velocity_lag_tau_gain_sec_per_rad_s_by_side_joint",
            "velocity_lag_max_tau_sec_by_side_joint",
        )
        normalized_joint_delay_fields: dict[
            str, Mapping[str, tuple[float, ...]]
        ] = {}
        for field_name in joint_delay_fields:
            values_by_side = getattr(self, field_name)
            normalized: dict[str, tuple[float, ...]] = {}
            for side, values in values_by_side.items():
                normalized_side = str(side).strip()
                if normalized_side not in joints:
                    raise ValueError(
                        f"{field_name} may contain only configured sides."
                    )
                normalized[normalized_side] = _six_finite(
                    values,
                    name=f"{field_name}[{normalized_side!r}]",
                    strictly_positive=False,
                )
            normalized_joint_delay_fields[field_name] = MappingProxyType(normalized)
        dynamic_side_sets = [
            set(normalized_joint_delay_fields[field_name])
            for field_name in joint_delay_fields[1:]
        ]
        if any(dynamic_side_sets) and not all(
            side_set == set(joints) for side_set in dynamic_side_sets
        ):
            raise ValueError(
                "velocity-dependent lag threshold, tau-gain, and max-tau fields "
                "must all define every configured side."
            )
        prefix = str(self.actuator_prefix).strip()
        if not prefix:
            raise ValueError("actuator_prefix must be non-empty.")
        status = str(self.calibration_status).strip()
        if not status:
            raise ValueError("calibration_status must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "joint_names_by_side", MappingProxyType(joints))
        object.__setattr__(self, "transport_delay_sec_by_side", MappingProxyType(delays))
        object.__setattr__(
            self,
            "observed_effective_delay_sec_by_side_joint",
            MappingProxyType(observed_delays),
        )
        for field_name, normalized in normalized_joint_delay_fields.items():
            object.__setattr__(self, field_name, normalized)
        for field_name, strictly_positive in (
            ("position_kp_nm_rad", True),
            ("position_kv_nm_s_rad", True),
            ("force_limit_nm", True),
            ("joint_armature_kg_m2", False),
            ("joint_damping_nm_s_rad", False),
            ("joint_frictionloss_nm", False),
        ):
            object.__setattr__(
                self,
                field_name,
                _six_finite(
                    getattr(self, field_name),
                    name=field_name,
                    strictly_positive=strictly_positive,
                ),
            )
        object.__setattr__(self, "actuator_prefix", prefix)
        object.__setattr__(self, "calibration_status", status)

    def actuator_name(self, side: str, joint_index: int) -> str:
        return f"{self.actuator_prefix}_{side}_joint{int(joint_index) + 1}"

    def command_delay_sec(
        self,
        side: str,
        joint_index: int,
        *,
        abs_command_velocity_rad_s: float = 0.0,
    ) -> float:
        """Return the causal residual target delay for one accepted qcmd."""

        normalized_side = str(side)
        index = int(joint_index)
        if normalized_side not in self.joint_names_by_side or not 0 <= index < 6:
            raise KeyError(f"unknown arm joint: side={side!r}, index={joint_index!r}")
        velocity = float(abs_command_velocity_rad_s)
        if not math.isfinite(velocity) or velocity < 0.0:
            raise ValueError(
                "abs_command_velocity_rad_s must be finite and non-negative."
            )
        base_by_joint = self.transport_delay_sec_by_side_joint.get(normalized_side)
        base = (
            self.transport_delay_sec_by_side[normalized_side]
            if base_by_joint is None
            else base_by_joint[index]
        )
        return float(base)

    def target_filter_tau_sec(
        self,
        side: str,
        joint_index: int,
        *,
        abs_command_velocity_rad_s: float = 0.0,
    ) -> float:
        """Return the high-speed causal target-filter time constant."""

        normalized_side = str(side)
        index = int(joint_index)
        if normalized_side not in self.joint_names_by_side or not 0 <= index < 6:
            raise KeyError(f"unknown arm joint: side={side!r}, index={joint_index!r}")
        velocity = float(abs_command_velocity_rad_s)
        if not math.isfinite(velocity) or velocity < 0.0:
            raise ValueError(
                "abs_command_velocity_rad_s must be finite and non-negative."
            )
        gain_by_joint = self.velocity_lag_tau_gain_sec_per_rad_s_by_side_joint.get(
            normalized_side
        )
        if gain_by_joint is None or gain_by_joint[index] <= 0.0:
            return 0.0
        threshold = self.velocity_lag_threshold_rad_s_by_side_joint[
            normalized_side
        ][index]
        max_tau = self.velocity_lag_max_tau_sec_by_side_joint[
            normalized_side
        ][index]
        tau = gain_by_joint[index] * max(velocity - threshold, 0.0)
        return float(min(tau, max_tau))

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "joint_names_by_side": {
                side: list(names)
                for side, names in self.joint_names_by_side.items()
            },
            "transport_delay_sec_by_side": dict(self.transport_delay_sec_by_side),
            "observed_effective_delay_sec_by_side_joint": {
                side: list(values)
                for side, values in self.observed_effective_delay_sec_by_side_joint.items()
            },
            "transport_delay_sec_by_side_joint": {
                side: list(values)
                for side, values in self.transport_delay_sec_by_side_joint.items()
            },
            "velocity_lag_threshold_rad_s_by_side_joint": {
                side: list(values)
                for side, values in self.velocity_lag_threshold_rad_s_by_side_joint.items()
            },
            "velocity_lag_tau_gain_sec_per_rad_s_by_side_joint": {
                side: list(values)
                for side, values in self.velocity_lag_tau_gain_sec_per_rad_s_by_side_joint.items()
            },
            "velocity_lag_max_tau_sec_by_side_joint": {
                side: list(values)
                for side, values in self.velocity_lag_max_tau_sec_by_side_joint.items()
            },
            "position_kp_nm_rad": list(self.position_kp_nm_rad),
            "position_kv_nm_s_rad": list(self.position_kv_nm_s_rad),
            "force_limit_nm": list(self.force_limit_nm),
            "joint_armature_kg_m2": list(self.joint_armature_kg_m2),
            "joint_damping_nm_s_rad": list(self.joint_damping_nm_s_rad),
            "joint_frictionloss_nm": list(self.joint_frictionloss_nm),
            "actuator_prefix": self.actuator_prefix,
            "calibration_status": self.calibration_status,
        }


def rm65_bimanual_udp_feedback_profile_v1() -> PositionActuationProfile:
    """Frozen first dual-RM65 free-motion fit from clean UDP traces.

    The explicit residual delay and MuJoCo servo were fitted jointly so their
    combined phase matches SDK-accepted qcmd -> measured qpos. The force limits
    remain conservative initial values, not a manufacturer torque claim or a
    contact/load identification result.
    """

    return PositionActuationProfile(
        name="rm65_bimanual_udp_feedback_20260721.v1",
        joint_names_by_side={
            side: tuple(f"{side}_arm_joint{index}" for index in range(1, 7))
            for side in _SIDES
        },
        # Residual target delay; the position actuator adds the remaining phase.
        transport_delay_sec_by_side={"left": 0.018, "right": 0.017},
        position_kp_nm_rad=(6000.0, 6000.0, 7200.0, 3600.0, 2800.0, 2000.0),
        position_kv_nm_s_rad=(45.0, 47.7, 61.6, 39.6, 25.2, 37.12),
        force_limit_nm=(220.0, 220.0, 180.0, 75.0, 55.0, 35.0),
        observed_effective_delay_sec_by_side_joint={
            "left": (0.027, 0.027, 0.028, 0.033, 0.029, 0.039),
            "right": (0.028, 0.029, 0.029, 0.029, 0.029, 0.038),
        },
        # These are kept zero until rotor/transmission identification.  URDF
        # link inertias remain active, so contacts and gravity are physical.
        joint_armature_kg_m2=(0.0,) * 6,
        joint_damping_nm_s_rad=(1.0, 1.0, 0.8, 0.3, 0.2, 0.1),
        joint_frictionloss_nm=(0.0,) * 6,
        calibration_status="multi_trace_free_motion_fitted_force_contact_provisional",
    )


def rm65_bimanual_udp_feedback_profile_v2() -> PositionActuationProfile:
    """All-data v2 fit with causal velocity-dependent J5-J6 phase."""

    degrees_per_radian = 180.0 / math.pi
    velocity_threshold_rad_s = math.radians(45.0)
    return PositionActuationProfile(
        name="rm65_bimanual_udp_feedback_20260721.v2",
        joint_names_by_side={
            side: tuple(f"{side}_arm_joint{index}" for index in range(1, 7))
            for side in _SIDES
        },
        # Kept for compatibility and as the proximal-joint fallback. The
        # per-joint residual delays below are authoritative for this profile.
        transport_delay_sec_by_side={"left": 0.018, "right": 0.017},
        transport_delay_sec_by_side_joint={
            "left": (0.018, 0.018, 0.018, 0.018, 0.018, 0.0110777),
            "right": (0.017, 0.017, 0.017, 0.017, 0.017, 0.0101775),
        },
        velocity_lag_threshold_rad_s_by_side_joint={
            side: (0.0, 0.0, 0.0) + (velocity_threshold_rad_s,) * 3
            for side in _SIDES
        },
        velocity_lag_tau_gain_sec_per_rad_s_by_side_joint={
            "left": tuple(
                value * degrees_per_radian
                for value in (0.0, 0.0, 0.0, 0.0, 0.000621110, 0.000444016)
            ),
            "right": tuple(
                value * degrees_per_radian
                for value in (0.0, 0.0, 0.0, 0.0, 0.000587959, 0.000413830)
            ),
        },
        velocity_lag_max_tau_sec_by_side_joint={
            "left": (0.0, 0.0, 0.0, 0.0, 0.01863330, 0.01332048),
            "right": (0.0, 0.0, 0.0, 0.0, 0.01763877, 0.01241490),
        },
        position_kp_nm_rad=(6000.0, 6000.0, 7200.0, 3600.0, 2800.0, 2000.0),
        position_kv_nm_s_rad=(45.0, 47.7, 61.6, 39.6, 25.2, 37.12),
        force_limit_nm=(220.0, 220.0, 180.0, 75.0, 55.0, 35.0),
        observed_effective_delay_sec_by_side_joint={
            "left": (0.029, 0.030, 0.030, 0.033, 0.031, 0.0330777),
            "right": (0.028, 0.029, 0.029, 0.032, 0.030, 0.0321775),
        },
        joint_armature_kg_m2=(0.0,) * 6,
        joint_damping_nm_s_rad=(1.0, 1.0, 0.8, 0.3, 0.2, 0.1),
        joint_frictionloss_nm=(0.0,) * 6,
        calibration_status=(
            "all_valid_traces_velocity_phase_fitted_new_heldout_pending_"
            "force_contact_provisional"
        ),
    )


def rm65_bimanual_udp_feedback_profile_v3() -> PositionActuationProfile:
    """Ten-trace v3 fit selected by weighted exact MuJoCo replay.

    Re-fitting all four distal-joint velocity terms regressed the weighted
    objective.  The only supported update is therefore the left J6 base phase
    and high-speed lag; every other v2 response parameter remains frozen.
    """

    v2 = rm65_bimanual_udp_feedback_profile_v2()
    degrees_per_radian = 180.0 / math.pi
    return replace(
        v2,
        name="rm65_bimanual_udp_feedback_20260721.v3",
        transport_delay_sec_by_side_joint={
            "left": (0.018, 0.018, 0.018, 0.018, 0.018, 0.0108740),
            "right": v2.transport_delay_sec_by_side_joint["right"],
        },
        velocity_lag_tau_gain_sec_per_rad_s_by_side_joint={
            "left": tuple(
                value * degrees_per_radian
                for value in (0.0, 0.0, 0.0, 0.0, 0.000621110, 0.000387787)
            ),
            "right": v2.velocity_lag_tau_gain_sec_per_rad_s_by_side_joint["right"],
        },
        velocity_lag_max_tau_sec_by_side_joint={
            "left": (0.0, 0.0, 0.0, 0.0, 0.01863330, 0.01163361),
            "right": v2.velocity_lag_max_tau_sec_by_side_joint["right"],
        },
        observed_effective_delay_sec_by_side_joint={
            "left": (0.029, 0.030, 0.030, 0.033, 0.031, 0.0328740),
            "right": v2.observed_effective_delay_sec_by_side_joint["right"],
        },
        calibration_status=(
            "ten_valid_traces_left_j6_refitted_new_heldout_pending_"
            "force_contact_provisional"
        ),
    )


def rm65_bimanual_udp_feedback_profile() -> PositionActuationProfile:
    """Fourteen-trace provisional v4 selected by exact MuJoCo replay.

    Four later clean captures extend the six clean and four guardian-restored
    v3 fit sessions. Two new records are excluded before fitting because one
    pair has no common command/UDP window and one UDP trace has no command
    trace. J1-J4 remain frozen: their millisecond-scale delay refit was not
    stable across clean sessions. V4 updates only the J5-J6 base phase and
    high-speed lag on both sides, using a 50 degree/second hinge capped at the
    observed 75 degree/second fit range.
    """

    v3 = rm65_bimanual_udp_feedback_profile_v3()
    degrees_per_radian = 180.0 / math.pi
    velocity_threshold_rad_s = math.radians(50.0)
    return replace(
        v3,
        name="rm65_bimanual_udp_feedback_20260721.v4",
        transport_delay_sec_by_side_joint={
            "left": (
                0.018,
                0.018,
                0.018,
                0.018,
                0.01789449411045608,
                0.009527118125949966,
            ),
            "right": (
                0.017,
                0.017,
                0.017,
                0.017,
                0.01648800307798916,
                0.010008068766796585,
            ),
        },
        velocity_lag_threshold_rad_s_by_side_joint={
            side: (0.0, 0.0, 0.0) + (velocity_threshold_rad_s,) * 3
            for side in _SIDES
        },
        velocity_lag_tau_gain_sec_per_rad_s_by_side_joint={
            "left": tuple(
                value * degrees_per_radian
                for value in (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0007381414409301171,
                    0.0005112003462640944,
                )
            ),
            "right": tuple(
                value * degrees_per_radian
                for value in (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0006118011925206194,
                    0.0004656704072877366,
                )
            ),
        },
        velocity_lag_max_tau_sec_by_side_joint={
            "left": (
                0.0,
                0.0,
                0.0,
                0.0,
                0.01845353602325293,
                0.01278000865660236,
            ),
            "right": (
                0.0,
                0.0,
                0.0,
                0.0,
                0.015295029813015485,
                0.011641760182193415,
            ),
        },
        observed_effective_delay_sec_by_side_joint={
            "left": (
                0.029,
                0.030,
                0.030,
                0.033,
                0.03089449411045608,
                0.031527118125949966,
            ),
            "right": (
                0.028,
                0.029,
                0.029,
                0.032,
                0.02948800307798916,
                0.032008068766796585,
            ),
        },
        calibration_status=(
            "fourteen_valid_traces_distal_phase_refitted_new_heldout_pending_"
            "force_contact_provisional"
        ),
    )


__all__ = [
    "PositionActuationProfile",
    "rm65_bimanual_udp_feedback_profile",
    "rm65_bimanual_udp_feedback_profile_v1",
    "rm65_bimanual_udp_feedback_profile_v2",
    "rm65_bimanual_udp_feedback_profile_v3",
]
