"""Versioned DM-Tac gripper actuation parameters for MuJoCo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from sim.rm65.actuation import (
    DmtacDirectionalPositionParameters,
    DmtacPositionActuationProfile,
    dmtac_bimanual_position_profile,
)


def _finite_tuple(
    values: Sequence[float],
    *,
    name: str,
    length: int | None = None,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} values.")
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values.")
    return result


@dataclass(frozen=True)
class GripperContactTuning:
    """Optional pad-contact override reserved for later real-data fitting."""

    friction: tuple[float, float, float] | None = None
    solref: tuple[float, float] | None = None
    solimp: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.friction is not None:
            friction = _finite_tuple(self.friction, name="friction", length=3)
            if any(value < 0.0 for value in friction):
                raise ValueError("friction values must be non-negative.")
            object.__setattr__(self, "friction", friction)
        if self.solref is not None:
            object.__setattr__(
                self,
                "solref",
                _finite_tuple(self.solref, name="solref", length=2),
            )
        if self.solimp is not None:
            object.__setattr__(
                self,
                "solimp",
                _finite_tuple(self.solimp, name="solimp", length=3),
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "friction": None if self.friction is None else list(self.friction),
            "solref": None if self.solref is None else list(self.solref),
            "solimp": None if self.solimp is None else list(self.solimp),
        }


@dataclass(frozen=True)
class GripperActuationProfile:
    """Physical and transport approximation for parallel DM-Tac grippers.

    The profile keeps transport, target motion, servo, and pad-contact
    parameters separate so later hardware identification can replace each
    layer without changing the Q6/action contract.
    """

    name: str
    joint_names_by_gripper: Mapping[str, Sequence[str]]
    side_by_gripper: Mapping[str, str]
    open_qpos_by_gripper: Mapping[str, Sequence[float]]
    closed_qpos_by_gripper: Mapping[str, Sequence[float]]
    command_hz: float
    send_threshold: float
    command_to_actuator_delay_sec_by_gripper: Mapping[str, float]
    finger_target_velocity_m_s: float
    finger_target_acceleration_m_s2: float
    position_kp_n_m: float
    position_kv_n_s_m: float
    force_limit_n: float
    joint_damping_n_s_m: float
    joint_frictionloss_n: float
    speed_setting: float = 1.0
    torque_setting: float = 1.0
    continuous_grasp: bool = True
    contact_tuning: GripperContactTuning = GripperContactTuning()
    actuator_prefix: str = "dmtac_position"
    calibration_status: str = "dataset_coarse_delay_temporary_servo"
    evidence: str = ""
    position_response_profile: DmtacPositionActuationProfile | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        prefix = str(self.actuator_prefix).strip()
        status = str(self.calibration_status).strip()
        if not name or not prefix or not status:
            raise ValueError("profile name, actuator_prefix, and calibration_status are required.")
        joints: dict[str, tuple[str, ...]] = {}
        sides: dict[str, str] = {}
        opens: dict[str, tuple[float, ...]] = {}
        closes: dict[str, tuple[float, ...]] = {}
        delays: dict[str, float] = {}
        for gripper, raw_names in self.joint_names_by_gripper.items():
            gripper_name = str(gripper).strip()
            joint_names = tuple(str(item).strip() for item in raw_names)
            if not gripper_name or len(joint_names) != 2 or not all(joint_names):
                raise ValueError("each gripper must define exactly two finger joints.")
            side = str(self.side_by_gripper.get(gripper_name, "")).strip()
            if side not in {"left", "right"}:
                raise ValueError(f"{gripper_name} must map to side left or right.")
            open_qpos = _finite_tuple(
                self.open_qpos_by_gripper[gripper_name],
                name=f"{gripper_name}.open_qpos",
                length=2,
            )
            closed_qpos = _finite_tuple(
                self.closed_qpos_by_gripper[gripper_name],
                name=f"{gripper_name}.closed_qpos",
                length=2,
            )
            if all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(open_qpos, closed_qpos)):
                raise ValueError(f"{gripper_name} open and closed qpos must differ.")
            delay = float(self.command_to_actuator_delay_sec_by_gripper[gripper_name])
            if not math.isfinite(delay) or delay < 0.0:
                raise ValueError("command-to-actuator delays must be finite and non-negative.")
            joints[gripper_name] = joint_names
            sides[gripper_name] = side
            opens[gripper_name] = open_qpos
            closes[gripper_name] = closed_qpos
            delays[gripper_name] = delay
        if not joints or len(set(sides.values())) != len(sides):
            raise ValueError("profile must define at most one gripper per side.")
        for field_name in (
            "command_hz",
            "finger_target_velocity_m_s",
            "finger_target_acceleration_m_s2",
            "position_kp_n_m",
            "position_kv_n_s_m",
            "force_limit_n",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
            object.__setattr__(self, field_name, value)
        for field_name in ("send_threshold", "joint_damping_n_s_m", "joint_frictionloss_n"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, field_name, value)
        for field_name in ("speed_setting", "torque_setting"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1].")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "actuator_prefix", prefix)
        object.__setattr__(self, "calibration_status", status)
        object.__setattr__(self, "evidence", str(self.evidence).strip())
        object.__setattr__(self, "joint_names_by_gripper", MappingProxyType(joints))
        object.__setattr__(self, "side_by_gripper", MappingProxyType(sides))
        object.__setattr__(self, "open_qpos_by_gripper", MappingProxyType(opens))
        object.__setattr__(self, "closed_qpos_by_gripper", MappingProxyType(closes))
        object.__setattr__(
            self,
            "command_to_actuator_delay_sec_by_gripper",
            MappingProxyType(delays),
        )
        response = self.position_response_profile
        if response is not None:
            if not isinstance(response, DmtacPositionActuationProfile):
                raise TypeError(
                    "position_response_profile must be a "
                    "DmtacPositionActuationProfile or None."
                )
            if set(response.parameters_by_side) != set(sides.values()):
                raise ValueError(
                    "position response sides must exactly match configured gripper sides."
                )
            if not math.isclose(response.command_hz, self.command_hz, abs_tol=1e-12):
                raise ValueError(
                    "position response command_hz must match the MuJoCo gripper profile."
                )
            if not math.isclose(
                response.send_threshold,
                self.send_threshold,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "position response send_threshold must match the MuJoCo gripper profile."
                )

    def actuator_name(self, gripper_name: str, joint_index: int) -> str:
        return f"{self.actuator_prefix}_{gripper_name}_finger{int(joint_index) + 1}"

    def equality_name(self, gripper_name: str) -> str:
        return f"{self.actuator_prefix}_{gripper_name}_symmetric"

    def gripper_for_side(self, side: str) -> str:
        matches = [name for name, value in self.side_by_gripper.items() if value == str(side)]
        if len(matches) != 1:
            raise KeyError(f"no unique gripper configured for side {side!r}.")
        return matches[0]

    def close_rate_per_sec(self, gripper_name: str) -> float:
        spans = [
            abs(closed - opened)
            for opened, closed in zip(
                self.open_qpos_by_gripper[gripper_name],
                self.closed_qpos_by_gripper[gripper_name],
            )
            if abs(closed - opened) > 1e-12
        ]
        return min(self.finger_target_velocity_m_s / span for span in spans)

    def close_acceleration_per_sec2(self, gripper_name: str) -> float:
        spans = [
            abs(closed - opened)
            for opened, closed in zip(
                self.open_qpos_by_gripper[gripper_name],
                self.closed_qpos_by_gripper[gripper_name],
            )
            if abs(closed - opened) > 1e-12
        ]
        return min(self.finger_target_acceleration_m_s2 / span for span in spans)

    def normalized_position_profile(self) -> DmtacPositionActuationProfile:
        """Return the reusable target dynamics, with a legacy fallback."""

        if self.position_response_profile is not None:
            return self.position_response_profile
        parameters = {}
        for gripper_name, side in self.side_by_gripper.items():
            directional = DmtacDirectionalPositionParameters(
                delay_sec=self.command_to_actuator_delay_sec_by_gripper[gripper_name],
                max_rate_per_sec=self.close_rate_per_sec(gripper_name),
                acceleration_per_sec2=self.close_acceleration_per_sec2(gripper_name),
            )
            parameters[side] = {"close": directional, "open": directional}
        return DmtacPositionActuationProfile(
            name=f"{self.name}.legacy_symmetric_target_dynamics",
            parameters_by_side=parameters,
            command_hz=self.command_hz,
            send_threshold=self.send_threshold,
            source_artifact="inline_legacy_mujoco_gripper_profile",
            source_payload_sha256="unavailable",
            source_conditions={
                "compatibility_fallback": True,
                "free_motion_directionality": "symmetric",
            },
        )

    def qpos_from_close(self, gripper_name: str, close: float) -> tuple[float, float]:
        value = min(1.0, max(0.0, float(close)))
        return tuple(
            opened + value * (closed - opened)
            for opened, closed in zip(
                self.open_qpos_by_gripper[gripper_name],
                self.closed_qpos_by_gripper[gripper_name],
            )
        )  # type: ignore[return-value]

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "joint_names_by_gripper": {
                name: list(joints) for name, joints in self.joint_names_by_gripper.items()
            },
            "side_by_gripper": dict(self.side_by_gripper),
            "command_hz": self.command_hz,
            "send_threshold": self.send_threshold,
            "command_to_actuator_delay_sec_by_gripper": dict(
                self.command_to_actuator_delay_sec_by_gripper
            ),
            "finger_target_velocity_m_s": self.finger_target_velocity_m_s,
            "finger_target_acceleration_m_s2": self.finger_target_acceleration_m_s2,
            "position_kp_n_m": self.position_kp_n_m,
            "position_kv_n_s_m": self.position_kv_n_s_m,
            "force_limit_n": self.force_limit_n,
            "joint_damping_n_s_m": self.joint_damping_n_s_m,
            "joint_frictionloss_n": self.joint_frictionloss_n,
            "speed_setting": self.speed_setting,
            "torque_setting": self.torque_setting,
            "continuous_grasp": bool(self.continuous_grasp),
            "contact_tuning": self.contact_tuning.to_metadata(),
            "calibration_status": self.calibration_status,
            "evidence": self.evidence,
            "position_response_profile": (
                None
                if self.position_response_profile is None
                else self.position_response_profile.to_metadata()
            ),
        }


def identified_bimanual_dmtac_gripper_profile() -> GripperActuationProfile:
    """Bilateral stage profile fitted from the 2026-07-22 hardware suite."""

    names = ("left_gripper", "right_gripper")
    response = dmtac_bimanual_position_profile()
    return GripperActuationProfile(
        name="dmtac_bimanual_position_20260722.v1",
        joint_names_by_gripper={
            name: (
                f"{name}_left_finger_joint",
                f"{name}_right_finger_joint",
            )
            for name in names
        },
        side_by_gripper={"left_gripper": "left", "right_gripper": "right"},
        open_qpos_by_gripper={name: (0.0, 0.0) for name in names},
        closed_qpos_by_gripper={name: (-0.035, -0.035) for name in names},
        command_hz=response.command_hz,
        send_threshold=response.send_threshold,
        # These compatibility fields describe the physical target envelope.
        # The executor uses the side/direction-specific response profile below.
        command_to_actuator_delay_sec_by_gripper={
            name: response.parameters(side, direction="close").delay_sec
            for name, side in (("left_gripper", "left"), ("right_gripper", "right"))
        },
        finger_target_velocity_m_s=0.035 * max(
            parameters.max_rate_per_sec
            for directions in response.parameters_by_side.values()
            for parameters in directions.values()
        ),
        finger_target_acceleration_m_s2=0.035 * max(
            parameters.acceleration_per_sec2
            for directions in response.parameters_by_side.values()
            for parameters in directions.values()
        ),
        position_kp_n_m=1200.0,
        position_kv_n_s_m=8.0,
        force_limit_n=20.0,
        joint_damping_n_s_m=2.0,
        joint_frictionloss_n=0.2,
        speed_setting=1.0,
        torque_setting=0.2,
        continuous_grasp=True,
        contact_tuning=GripperContactTuning(),
        calibration_status=(
            "bilateral_directional_free_motion_identified_"
            "fast_preemption_provisional_contact_unmodeled"
        ),
        evidence=(
            "dmtac_bimanual_position_20260722_184856_698964; independent left/right "
            "and close/open fits with 44 held-out step segments. Contact parameters "
            "remain unmodeled because no external force gauge was available."
        ),
        position_response_profile=response,
    )


def temporary_bimanual_dmtac_gripper_profile() -> GripperActuationProfile:
    """Compatibility alias for the current identified bimanual V1 profile."""

    return identified_bimanual_dmtac_gripper_profile()


__all__ = [
    "GripperActuationProfile",
    "GripperContactTuning",
    "identified_bimanual_dmtac_gripper_profile",
    "temporary_bimanual_dmtac_gripper_profile",
]
