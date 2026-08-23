"""Minimal high-rate joint-position driver for the B601 MuJoCo plant."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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


def default_b601_position_timing() -> TimingSpec:
    """Return the explicit 500 Hz command / 1 kHz physics clock contract."""

    return TimingSpec(
        command_hz=B601_RS_COMMAND_HZ,
        physics_hz=B601_RS_PHYSICS_HZ,
        parameter_source=(
            "rebot_b601_rs_visualization_only_provisional_position_servo"
        ),
    )


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
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
    """Audit record for one physical MuJoCo step."""

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


class B601PositionDriver:
    """Apply joint-limit-checked q targets through six position actuators.

    The driver owns only command deadlines, validation, zero-order hold and one
    ``mj_step`` per call.  IK, trajectories, collision planning and hardware
    transport remain outside this module.  It does not limit target slew or
    velocity and does not guard simulated state excursions, so M2 is restricted
    to modest visualization commands.
    """

    execution_mode = "rebot_b601_rs_mujoco_position_servo"
    robot_dynamics = "mujoco_visualization_only_provisional"

    def __init__(
        self,
        scene: MujocoScene,
        timing: TimingSpec | None = None,
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
            raise ValueError(
                "scene timestep does not match TimingSpec.physics_hz"
            )
        try:
            arm_spec = scene.arm_specs[B601_RS_SIM_ARM_KEY]
        except KeyError as exc:
            raise KeyError("scene has no explicit B601-RS arm spec") from exc
        if tuple(arm_spec.joint_names) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("scene B601-RS arm joint order does not match the model")

        self.scene = scene
        self.timing = resolved_timing
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
        self._target_qpos = np.zeros(6, dtype=float)
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
            "gripper": "fixed_open_geometry_only",
            "target_slew_limit": "not_implemented_m2",
            "state_limit_guard": "not_implemented_m2",
            "command_scope": "modest_visualization_targets_only",
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
        self._target_qpos = qpos.copy()
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
        """Validate an optional due command and advance exactly one 1 kHz tick."""

        due = self.command_due
        scheduled_time = self._next_command_time_sec if due else None
        applied = False
        rejection_reason = None
        if command is not None:
            if not isinstance(command, B601JointPositionCommand):
                raise TypeError("command must be a B601JointPositionCommand")
            if not due:
                rejection_reason = "command_before_deadline"
            else:
                rejection_reason = self._rejection_reason(command)
                if rejection_reason is None:
                    self._target_qpos = np.asarray(command.qpos, dtype=float)
                    self._accepted_sequence = int(command.sequence)
                    self._source_timestamp_sec = float(
                        command.source_timestamp_sec
                    )
                    self._receive_timestamp_sec = float(
                        command.receive_timestamp_sec
                    )
                    self._frame_id = command.frame_id
                    applied = True
        if due:
            period = 1.0 / float(self.timing.command_hz)
            while self._next_command_time_sec <= self.physics_time_sec + 1e-12:
                self._next_command_time_sec += period

        tick_index = self._physics_tick
        tick_time = self.physics_time_sec
        self._write_ctrl()
        self._hold_gripper_open()
        self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
        self._hold_gripper_open()
        self.scene.forward()
        self._physics_tick += 1
        measured_qpos = np.asarray(
            self.scene.data.qpos[self._qpos_addresses], dtype=float
        )
        measured_qvel = np.asarray(
            self.scene.data.qvel[self._qvel_addresses], dtype=float
        )
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
            target_qpos=tuple(float(value) for value in self._target_qpos),
            measured_qpos=tuple(float(value) for value in measured_qpos),
            measured_qvel=tuple(float(value) for value in measured_qvel),
            state_valid=self.scene.state_is_finite(),
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
        self.scene.data.ctrl[self._actuator_ids] = self._target_qpos

    def _hold_gripper_open(self) -> None:
        for name, value in B601_RS_GRIPPER_OPEN_QPOS.items():
            self.scene.data.qpos[self.scene.joint_qpos_addr(name)] = float(value)
            self.scene.data.qvel[self.scene.joint_dof_addr(name)] = 0.0


__all__ = [
    "B601_RS_JOINT_POSITION_FRAME",
    "B601JointPositionCommand",
    "B601PositionDriver",
    "B601PositionTick",
    "default_b601_position_timing",
]
