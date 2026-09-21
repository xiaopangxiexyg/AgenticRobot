"""Minimal high-rate joint-position driver for the B601 MuJoCo plant."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robot_motion.joint_limits import JointRateLimiter, JointVelocityLimitProfile
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES
from sim.contracts import TimingSpec
from sim.mujoco.scene import MujocoScene

from .scene import (
    B601_RS_COMMAND_HZ,
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_PHYSICS_HZ,
    B601_RS_SIM_ARM_KEY,
    b601_position_actuator_name,
)

B601_RS_JOINT_POSITION_FRAME = "b601_rs/joint_position"
B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S = 5.0
# The current provisional position servo reaches about 1.09x target slew in
# the large-step fixture; retain a documented 20% simulation-only transient
# margin without presenting it as a hardware velocity limit.
B601_RS_DEFAULT_SIM_MEASURED_VELOCITY_MARGIN_RATIO = 0.20


def b601_sim_measured_velocity_limit_rad_s(
    target_velocity_limit_rad_s: float,
) -> float:
    """Return the provisional plant-response envelope for a target slew.

    The position actuator can transiently move faster than the conditioned
    target while it closes the position error.  This envelope is deliberately
    separate from the command slew limit and is only a MuJoCo visualization
    contract; it is not a hardware velocity or safety limit.
    """

    if isinstance(target_velocity_limit_rad_s, bool):
        raise TypeError("target_velocity_limit_rad_s must be finite and positive")
    normalized_velocity = float(target_velocity_limit_rad_s)
    if not math.isfinite(normalized_velocity) or normalized_velocity <= 0.0:
        raise ValueError("target_velocity_limit_rad_s must be finite and positive")
    return normalized_velocity * (
        1.0 + B601_RS_DEFAULT_SIM_MEASURED_VELOCITY_MARGIN_RATIO
    )


B601_RS_DEFAULT_SIM_MAX_MEASURED_VELOCITY_RAD_S = (
    b601_sim_measured_velocity_limit_rad_s(
        B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S
    )
)


def default_b601_position_timing() -> TimingSpec:
    """Return the explicit 500 Hz command / 1 kHz physics clock contract."""

    return TimingSpec(
        command_hz=B601_RS_COMMAND_HZ,
        physics_hz=B601_RS_PHYSICS_HZ,
        parameter_source=(
            "rebot_b601_rs_visualization_only_provisional_position_servo"
        ),
    )


def _default_b601_velocity_profile(
    timing: TimingSpec,
    *,
    max_velocity_rad_s: float = B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S,
) -> JointVelocityLimitProfile:
    """Build the provisional simulation-only position target rate profile."""

    if isinstance(max_velocity_rad_s, bool):
        raise TypeError("max_velocity_rad_s must be finite and positive")
    normalized_velocity = float(max_velocity_rad_s)
    if not math.isfinite(normalized_velocity) or normalized_velocity <= 0.0:
        raise ValueError("max_velocity_rad_s must be finite and positive")
    velocity_deg_s = math.degrees(normalized_velocity)
    return JointVelocityLimitProfile(
        name="b601_rs_visualization_only_provisional",
        max_velocity_deg_s=(velocity_deg_s,) * 6,
        max_dt_sec=1.0 / float(timing.command_hz),
        joint_names=B601_RS_ARM_JOINT_NAMES,
        metadata={
            "parameter_source": (
                "explicit_simulation_only_target_velocity_limit_"
                f"{normalized_velocity:g}_rad_s"
            ),
            "hardware_safety_claim": False,
        },
    )


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _finite_timestamp(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _six_qpos(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    qpos = np.asarray(values, dtype=float)
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"{name} must contain six finite values")
    return tuple(float(value) for value in qpos)


@dataclass(frozen=True)
class B601JointPositionCommand:
    """One timestamped command-rate B601 arm target."""

    sequence: int
    source_timestamp_sec: float
    receive_timestamp_sec: float
    qpos: tuple[float, ...]
    frame_id: str = B601_RS_JOINT_POSITION_FRAME
    valid: bool = True

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        object.__setattr__(
            self,
            "sequence",
            _non_negative_integer(self.sequence, name="sequence"),
        )
        object.__setattr__(
            self,
            "source_timestamp_sec",
            _finite_timestamp(
                self.source_timestamp_sec,
                name="source_timestamp_sec",
            ),
        )
        object.__setattr__(
            self,
            "receive_timestamp_sec",
            _finite_timestamp(
                self.receive_timestamp_sec,
                name="receive_timestamp_sec",
            ),
        )
        object.__setattr__(self, "qpos", _six_qpos(self.qpos, name="qpos"))
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "valid", bool(self.valid))


@dataclass(frozen=True)
class B601PositionTick:
    """Audit record for one physical MuJoCo step.

    ``command_applied`` means admission accepted a sample. ``target_qpos``
    remains that latest requested goal for M2 compatibility, while
    ``actuator_target_qpos`` is the rate-limited value written to MuJoCo.
    ``state_valid`` retains the finite-state meaning; limit faults are exposed
    separately through ``state_within_limits`` and ``driver_faulted``.
    """

    physics_tick: int
    physics_time_sec: float
    command_due: bool
    scheduled_command_time_sec: float | None
    command_received: bool
    command_applied: bool
    accepted_sequence: int | None
    accepted_source_timestamp_sec: float | None
    accepted_receive_timestamp_sec: float | None
    accepted_frame_id: str | None
    rejection_reason: str | None
    target_qpos: tuple[float, ...]
    measured_qpos: tuple[float, ...]
    measured_qvel: tuple[float, ...]
    state_valid: bool
    actuator_target_qpos: tuple[float, ...] = ()
    actuator_target_changed: bool = False
    slew_limited: bool = False
    slew_limit_ratio: float = 0.0
    state_within_limits: bool = True
    driver_faulted: bool = False
    fault_reason: str | None = None
    plant_advanced: bool = True


class B601PositionDriver:
    """Apply joint-limit-checked q targets through six position actuators.

    The driver owns command deadlines, validation, final actuator-target rate
    limiting, a latched simulated-state guard and one ``mj_step`` per call. IK,
    trajectories, collision planning and hardware transport remain outside this
    module. The limits here are provisional visualization parameters, not a
    hardware safety envelope.
    """

    execution_mode = "rebot_b601_rs_mujoco_position_servo"
    robot_dynamics = "mujoco_visualization_only_provisional"

    def __init__(
        self,
        scene: MujocoScene,
        timing: TimingSpec | None = None,
        *,
        velocity_profile: JointVelocityLimitProfile | None = None,
        target_velocity_limit_rad_s: float | None = None,
        state_limit_tolerance_rad: float = 1e-4,
        hold_gripper_open: bool = True,
    ) -> None:
        if not isinstance(scene, MujocoScene):
            raise TypeError("scene must be a MujocoScene")
        resolved_timing = timing or default_b601_position_timing()
        if not isinstance(resolved_timing, TimingSpec):
            raise TypeError("timing must be a TimingSpec")
        if not math.isclose(
            float(scene.model.opt.timestep),
            resolved_timing.physics_timestep_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scene timestep does not match TimingSpec.physics_hz")
        try:
            arm_spec = scene.arm_specs[B601_RS_SIM_ARM_KEY]
        except KeyError as exc:
            raise KeyError("scene has no explicit B601-RS arm spec") from exc
        if tuple(arm_spec.joint_names) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("scene B601-RS arm joint order does not match the model")

        if velocity_profile is not None and target_velocity_limit_rad_s is not None:
            raise ValueError(
                "provide either velocity_profile or target_velocity_limit_rad_s, not both"
            )
        resolved_velocity_profile = velocity_profile or _default_b601_velocity_profile(
            resolved_timing,
            max_velocity_rad_s=(
                B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S
                if target_velocity_limit_rad_s is None
                else target_velocity_limit_rad_s
            ),
        )
        if not isinstance(resolved_velocity_profile, JointVelocityLimitProfile):
            raise TypeError("velocity_profile must be a JointVelocityLimitProfile")
        if resolved_velocity_profile.dof != len(B601_RS_ARM_JOINT_NAMES):
            raise ValueError("velocity_profile must contain six joint limits")
        if resolved_velocity_profile.joint_names and (
            tuple(resolved_velocity_profile.joint_names) != B601_RS_ARM_JOINT_NAMES
        ):
            raise ValueError(
                "velocity_profile joint order does not match the B601 model"
            )
        if isinstance(state_limit_tolerance_rad, bool):
            raise TypeError("state_limit_tolerance_rad must be finite and non-negative")
        tolerance = float(state_limit_tolerance_rad)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "state_limit_tolerance_rad must be finite and non-negative"
            )
        if not isinstance(hold_gripper_open, bool):
            raise TypeError("hold_gripper_open must be bool")

        self.scene = scene
        self.timing = resolved_timing
        self.velocity_profile = resolved_velocity_profile
        self.state_limit_tolerance_rad = tolerance
        self.hold_gripper_open = hold_gripper_open
        self._qpos_addresses = np.asarray(
            [scene.joint_qpos_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._qvel_addresses = np.asarray(
            [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._actuator_ids = np.asarray(
            [self._actuator_id(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        joint_ids = np.asarray(
            [scene.joint_id(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        if not np.all(scene.model.jnt_limited[joint_ids]):
            raise ValueError("all B601-RS arm joints must be position limited")
        self._lower = np.asarray(scene.model.jnt_range[joint_ids, 0], dtype=float)
        self._upper = np.asarray(scene.model.jnt_range[joint_ids, 1], dtype=float)
        self._physics_tick = 0
        self._next_command_time_sec = 0.0
        self._requested_qpos = np.zeros(6, dtype=float)
        self._actuator_target_qpos = np.zeros(6, dtype=float)
        self._rate_limiter = JointRateLimiter(self.velocity_profile)
        self._fault_reason: str | None = None
        self._accepted_sequence: int | None = None
        self._source_timestamp_sec: float | None = None
        self._receive_timestamp_sec: float | None = None
        self._frame_id: str | None = None
        self.reset(B601_RS_DEFAULT_ARM_QPOS)

    @property
    def physics_time_sec(self) -> float:
        return self._physics_tick / float(self.timing.physics_hz)

    @property
    def command_due(self) -> bool:
        return self.physics_time_sec + 1e-12 >= self._next_command_time_sec

    @property
    def execution_metadata(self) -> dict[str, object]:
        maximum_target_velocity_rad_s = math.radians(
            max(float(value) for value in self.velocity_profile.max_velocity_deg_s)
        )
        return {
            "execution_mode": self.execution_mode,
            "robot_dynamics": self.robot_dynamics,
            "command_hz": float(self.timing.command_hz),
            "physics_hz": float(self.timing.physics_hz),
            "command_hold": "zero_order_hold_between_command_ticks",
            "gravity_mps2": tuple(
                float(value) for value in self.scene.model.opt.gravity
            ),
            "gravity_compensation": "not_implemented_m2",
            "collision": "disabled_in_m2",
            "gripper": (
                "fixed_open_geometry_only"
                if self.hold_gripper_open
                else "external_opt_in_kinematic_mapping"
            ),
            "target_slew_limit": "per_joint_velocity_limit_at_command_rate",
            "target_slew_profile": {
                "name": self.velocity_profile.name,
                "max_velocity_deg_s": tuple(
                    float(value) for value in self.velocity_profile.max_velocity_deg_s
                ),
                "max_dt_sec": float(self.velocity_profile.max_dt_sec),
                "metadata": dict(self.velocity_profile.metadata),
            },
            "measured_velocity_envelope": {
                "max_velocity_rad_s": b601_sim_measured_velocity_limit_rad_s(
                    maximum_target_velocity_rad_s
                ),
                "response_margin_ratio": (
                    B601_RS_DEFAULT_SIM_MEASURED_VELOCITY_MARGIN_RATIO
                ),
                "parameter_source": (
                    "simulation_only_position_servo_response_margin"
                ),
                "hardware_safety_claim": False,
            },
            "state_limit_guard": "latched_simulated_joint_position_guard",
            "state_limit_tolerance_rad": self.state_limit_tolerance_rad,
            "command_scope": "modest_visualization_targets_only",
            "bounded_target_stage": "simulation_only_position_target_conditioning",
            "hardware_fidelity": "not_claimed",
        }

    def reset(
        self,
        initial_qpos: Sequence[float] = B601_RS_DEFAULT_ARM_QPOS,
    ) -> None:
        """Reset simulation state; this is not a command acceptance event."""

        qpos = np.asarray(_six_qpos(initial_qpos, name="initial_qpos"), dtype=float)
        if np.any(qpos < self._lower) or np.any(qpos > self._upper):
            raise ValueError("initial_qpos exceeds B601-RS arm limits")
        self.scene.mujoco.mj_resetData(self.scene.model, self.scene.data)
        self.scene.data.qpos[self._qpos_addresses] = qpos
        self.scene.data.qvel[self._qvel_addresses] = 0.0
        self._requested_qpos = qpos.copy()
        self._actuator_target_qpos = qpos.copy()
        self._rate_limiter.reset(qpos)
        self._fault_reason = None
        self._physics_tick = 0
        self._next_command_time_sec = 0.0
        self._accepted_sequence = None
        self._source_timestamp_sec = None
        self._receive_timestamp_sec = None
        self._frame_id = None
        self._write_ctrl()
        self._hold_gripper_open()
        self.scene.forward()

    def step_physics(
        self,
        command: B601JointPositionCommand | None = None,
    ) -> B601PositionTick:
        """Validate a due command and attempt one 1 kHz plant step.

        A pre-existing non-finite simulated state latches the driver fault and
        fails closed without calling MuJoCo again. Reset is then required.
        """

        due = self.command_due
        scheduled_time = self._next_command_time_sec if due else None
        applied = False
        rejection_reason = None
        actuator_target_changed = False
        slew_limited = False
        slew_limit_ratio = 0.0
        current_state_finite = self._latch_state_fault_if_needed()
        non_finite_fault = (
            not current_state_finite
            or self._fault_reason == "non_finite_simulated_state"
        )
        if command is not None:
            if not isinstance(command, B601JointPositionCommand):
                raise TypeError("command must be a B601JointPositionCommand")
            if self._fault_reason is not None:
                rejection_reason = "driver_fault_latched"
            elif not due:
                rejection_reason = "command_before_deadline"
            else:
                rejection_reason = self._rejection_reason(command)
                if rejection_reason is None:
                    self._requested_qpos = np.asarray(command.qpos, dtype=float)
                    self._accepted_sequence = int(command.sequence)
                    self._source_timestamp_sec = float(command.source_timestamp_sec)
                    self._receive_timestamp_sec = float(command.receive_timestamp_sec)
                    self._frame_id = command.frame_id
                    applied = True
        if due and self._fault_reason is None:
            previous_target = self._actuator_target_qpos.copy()
            limited_step = self._rate_limiter.step(
                target_qpos=self._requested_qpos,
                dt_sec=1.0 / float(self.timing.command_hz),
                reference_qpos=self._actuator_target_qpos,
            )
            self._actuator_target_qpos = np.clip(
                limited_step.qpos,
                self._lower,
                self._upper,
            )
            if not np.array_equal(
                self._actuator_target_qpos,
                limited_step.qpos,
            ):
                self._rate_limiter.reset(self._actuator_target_qpos)
            slew_limited = bool(limited_step.limited)
            slew_limit_ratio = float(limited_step.limit_ratio)
            actuator_target_changed = not np.array_equal(
                self._actuator_target_qpos,
                previous_target,
            )
        if due and not non_finite_fault:
            period = 1.0 / float(self.timing.command_hz)
            while self._next_command_time_sec <= self.physics_time_sec + 1e-12:
                self._next_command_time_sec += period

        tick_index = self._physics_tick
        tick_time = self.physics_time_sec
        plant_advanced = False
        if not non_finite_fault:
            self._write_ctrl()
            self._hold_gripper_open()
            self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
            self._hold_gripper_open()
            self.scene.forward()
            self._physics_tick += 1
            plant_advanced = True
            self._latch_state_fault_if_needed()
        measured_qpos = np.asarray(
            self.scene.data.qpos[self._qpos_addresses], dtype=float
        )
        measured_qvel = np.asarray(
            self.scene.data.qvel[self._qvel_addresses], dtype=float
        )
        state_finite, state_within_limits, _reason = self._state_status()
        return B601PositionTick(
            physics_tick=tick_index,
            physics_time_sec=tick_time,
            command_due=due,
            scheduled_command_time_sec=scheduled_time,
            command_received=command is not None,
            command_applied=applied,
            accepted_sequence=self._accepted_sequence,
            accepted_source_timestamp_sec=self._source_timestamp_sec,
            accepted_receive_timestamp_sec=self._receive_timestamp_sec,
            accepted_frame_id=self._frame_id,
            rejection_reason=rejection_reason,
            target_qpos=tuple(float(value) for value in self._requested_qpos),
            measured_qpos=tuple(float(value) for value in measured_qpos),
            measured_qvel=tuple(float(value) for value in measured_qvel),
            state_valid=state_finite,
            actuator_target_qpos=tuple(
                float(value) for value in self._actuator_target_qpos
            ),
            actuator_target_changed=actuator_target_changed,
            slew_limited=slew_limited,
            slew_limit_ratio=slew_limit_ratio,
            state_within_limits=state_within_limits,
            driver_faulted=self._fault_reason is not None,
            fault_reason=self._fault_reason,
            plant_advanced=plant_advanced,
        )

    def _actuator_id(self, joint_name: str) -> int:
        actuator_id = self.scene.mujoco.mj_name2id(
            self.scene.model,
            self.scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
            b601_position_actuator_name(joint_name),
        )
        if actuator_id < 0:
            raise KeyError(f"scene has no B601 position actuator for {joint_name}")
        return int(actuator_id)

    def _rejection_reason(
        self,
        command: B601JointPositionCommand,
    ) -> str | None:
        if not command.valid:
            return "invalid_sample"
        if command.frame_id != B601_RS_JOINT_POSITION_FRAME:
            return "unexpected_frame"
        if (
            self._accepted_sequence is not None
            and command.sequence <= self._accepted_sequence
        ):
            return "stale_sequence"
        if (
            self._source_timestamp_sec is not None
            and command.source_timestamp_sec < self._source_timestamp_sec
        ):
            return "stale_source_timestamp"
        if (
            self._receive_timestamp_sec is not None
            and command.receive_timestamp_sec < self._receive_timestamp_sec
        ):
            return "stale_receive_timestamp"
        qpos = np.asarray(command.qpos, dtype=float)
        if np.any(qpos < self._lower) or np.any(qpos > self._upper):
            return "joint_limit_violation"
        return None

    def _write_ctrl(self) -> None:
        self.scene.data.ctrl[self._actuator_ids] = self._actuator_target_qpos

    def _state_status(self) -> tuple[bool, bool, str | None]:
        qpos = np.asarray(self.scene.data.qpos[self._qpos_addresses], dtype=float)
        qvel = np.asarray(self.scene.data.qvel[self._qvel_addresses], dtype=float)
        finite = bool(
            self.scene.state_is_finite()
            and np.all(np.isfinite(qpos))
            and np.all(np.isfinite(qvel))
        )
        if not finite:
            return False, False, "non_finite_simulated_state"

        lower_excursion = self._lower - qpos
        upper_excursion = qpos - self._upper
        excursion = np.maximum(lower_excursion, upper_excursion)
        index = int(np.argmax(excursion))
        maximum_excursion = float(excursion[index])
        within_limits = maximum_excursion <= self.state_limit_tolerance_rad
        if within_limits:
            return True, True, None

        direction = (
            "below_lower"
            if lower_excursion[index] >= upper_excursion[index]
            else "above_upper"
        )
        reason = (
            f"joint_state_limit_violation:{B601_RS_ARM_JOINT_NAMES[index]}:"
            f"{direction}:excursion_rad={maximum_excursion:.9g}"
        )
        return True, False, reason

    def _latch_state_fault_if_needed(self) -> bool:
        finite, _within_limits, reason = self._state_status()
        if not finite:
            self._fault_reason = "non_finite_simulated_state"
        elif self._fault_reason is None and reason is not None:
            self._fault_reason = reason
        return finite

    def _hold_gripper_open(self) -> None:
        if not self.hold_gripper_open:
            return
        for name, value in B601_RS_GRIPPER_OPEN_QPOS.items():
            self.scene.data.qpos[self.scene.joint_qpos_addr(name)] = float(value)
            self.scene.data.qvel[self.scene.joint_dof_addr(name)] = 0.0


__all__ = [
    "B601_RS_DEFAULT_SIM_MAX_MEASURED_VELOCITY_RAD_S",
    "B601_RS_DEFAULT_SIM_MEASURED_VELOCITY_MARGIN_RATIO",
    "B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S",
    "B601_RS_JOINT_POSITION_FRAME",
    "B601JointPositionCommand",
    "B601PositionDriver",
    "B601PositionTick",
    "b601_sim_measured_velocity_limit_rad_s",
    "default_b601_position_timing",
]
