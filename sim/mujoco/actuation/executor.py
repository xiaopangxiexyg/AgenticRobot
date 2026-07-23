"""MuJoCo adapter for the backend-independent RM65 position actuator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sim.contracts import TimingSpec
from sim.rm65.actuation import PositionActuationProfile, Rm65PositionActuator

from ..control import DirectQposCommand


@dataclass(frozen=True)
class ActuatedPositionTick:
    """One MuJoCo physics tick after the optional 120 Hz transport send."""

    physics_tick: int
    physics_time_sec: float
    command_due: bool
    scheduled_command_time_sec: float | None
    accepted_command_sequence: int
    command_applied: bool
    actuator_target_by_side: Mapping[str, np.ndarray]
    measured_qpos_by_side: Mapping[str, np.ndarray]


class MujocoActuatedPositionExecutor:
    """Bind reusable RM65 target dynamics to MuJoCo position actuators.

    The accepted-qcmd/delay/filter state machine lives in
    :class:`sim.rm65.actuation.Rm65PositionActuator`.  This adapter only maps
    target vectors to MuJoCo actuator IDs, advances ``mj_step``, and exposes
    measured qpos.
    """

    execution_mode = "actuated_position_servo"
    robot_dynamics = "mujoco_position_actuator"

    def __init__(
        self,
        scene,
        timing: TimingSpec,
        profile: PositionActuationProfile,
        *,
        active_sides: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(timing, TimingSpec):
            raise TypeError("timing must be a TimingSpec.")
        self.scene = scene
        self.timing = timing
        self.profile = profile
        model_option = getattr(getattr(scene, "model", None), "opt", None)
        if model_option is not None and not math.isclose(
            float(model_option.timestep),
            timing.physics_timestep_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scene model timestep does not match TimingSpec.physics_hz.")
        sides = tuple(
            str(side)
            for side in (
                active_sides
                if active_sides is not None
                else tuple(profile.joint_names_by_side)
            )
        )
        if not sides or len(set(sides)) != len(sides):
            raise ValueError("active_sides must be a non-empty unique sequence.")
        unknown = set(sides) - set(profile.joint_names_by_side)
        if unknown:
            raise ValueError(f"active_sides are absent from profile: {sorted(unknown)}")
        missing = set(sides) - set(scene.arm_specs)
        if missing:
            raise KeyError(f"scene is missing actuator side(s): {sorted(missing)}")
        self.active_sides = sides
        self._actuator_ids: dict[str, np.ndarray] = {}
        for side in sides:
            joints = profile.joint_names_by_side[side]
            expected = tuple(scene.arm_specs[side].joint_names)
            if tuple(joints) != expected:
                raise ValueError(
                    f"{side} profile joints {tuple(joints)} do not match "
                    f"scene arm joints {expected}."
                )
            ids: list[int] = []
            for index in range(6):
                name = profile.actuator_name(side, index)
                actuator_id = scene.mujoco.mj_name2id(
                    scene.model,
                    scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name,
                )
                if actuator_id < 0:
                    raise KeyError(
                        f"scene has no position actuator {name!r}; compile it "
                        "with the same PositionActuationProfile."
                    )
                ids.append(int(actuator_id))
            self._actuator_ids[side] = np.asarray(ids, dtype=np.int32)
        self._actuator = Rm65PositionActuator(
            profile,
            command_hz=timing.command_hz,
            physics_hz=timing.physics_hz,
            active_sides=sides,
        )
        self._held_command: DirectQposCommand | None = None

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "robot_dynamics": self.robot_dynamics,
            "command_hz": float(self.timing.command_hz),
            "physics_hz": float(self.timing.physics_hz),
            "command_hold": "zero_order_hold_between_command_ticks",
            "transport_acceptance": "immediate_on_120hz_sender_call",
            "target_dynamics_owner": "sim.rm65.actuation.Rm65PositionActuator",
            "profile": self.profile.to_metadata(),
        }

    @property
    def physics_time_sec(self) -> float:
        return self._actuator.physics_time_sec

    @property
    def command_due(self) -> bool:
        return self._actuator.command_due

    def accepted_arm_qpos(self, side: str) -> np.ndarray:
        return self._actuator.accepted_qpos(side)

    def measured_arm_qpos(self, side: str) -> np.ndarray:
        return (
            np.asarray(self.scene.get_arm_qpos(str(side)), dtype=float)
            .reshape(6)
            .copy()
        )

    def reset(self, initial_command: DirectQposCommand) -> None:
        self._validate_command(initial_command, require_newer_than=None)
        self._held_command = initial_command
        self._actuator.reset(
            command_sequence=int(initial_command.sequence),
            qpos_by_side=initial_command.qpos_by_side,
        )
        self._write_ctrl(self._actuator.actuator_target_by_side)

    def step_physics(
        self,
        command: DirectQposCommand | None = None,
    ) -> ActuatedPositionTick:
        """Optionally accept one due 120 Hz qcmd, then advance MuJoCo once."""

        if self._held_command is None:
            raise RuntimeError("reset(initial_command) is required before stepping physics.")
        due = self.command_due
        if due:
            if command is None:
                raise RuntimeError(
                    "A DirectQposCommand is required at the current 120 Hz deadline."
                )
            self._validate_command(command, require_newer_than=self._held_command)
            core_tick = self._actuator.step(
                command_sequence=int(command.sequence),
                qpos_by_side=command.qpos_by_side,
            )
            self._held_command = command
        else:
            if command is not None:
                raise RuntimeError(
                    "DirectQposCommand arrived before the next 120 Hz deadline."
                )
            core_tick = self._actuator.step()
        self._write_ctrl(core_tick.actuator_target_by_side)
        self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
        return ActuatedPositionTick(
            physics_tick=core_tick.physics_tick,
            physics_time_sec=core_tick.physics_time_sec,
            command_due=core_tick.command_due,
            scheduled_command_time_sec=core_tick.scheduled_command_time_sec,
            accepted_command_sequence=core_tick.accepted_command_sequence,
            command_applied=core_tick.command_applied,
            actuator_target_by_side={
                side: values.copy()
                for side, values in core_tick.actuator_target_by_side.items()
            },
            measured_qpos_by_side={
                side: self.measured_arm_qpos(side) for side in self.active_sides
            },
        )

    def snapshot(self) -> dict[str, object]:
        return {
            **self.execution_metadata,
            **self._actuator.snapshot(),
            "measured_qpos_by_side": {
                side: self.measured_arm_qpos(side).tolist()
                for side in self.active_sides
            },
        }

    def _validate_command(
        self,
        command: DirectQposCommand,
        *,
        require_newer_than: DirectQposCommand | None,
    ) -> None:
        if not isinstance(command, DirectQposCommand):
            raise TypeError("command must be a DirectQposCommand.")
        if set(command.qpos_by_side) != set(self.active_sides):
            raise ValueError(
                "qcmd sides must exactly match active_sides: "
                f"expected={sorted(self.active_sides)} "
                f"got={sorted(command.qpos_by_side)}"
            )
        if command.gripper_close_by_name:
            raise ValueError("RM65 arm position sender does not accept gripper commands.")
        if (
            require_newer_than is not None
            and command.sequence <= require_newer_than.sequence
        ):
            raise ValueError("qcmd sequence must increase at each 120 Hz send.")

    def _write_ctrl(self, target_by_side: Mapping[str, np.ndarray]) -> None:
        for side, actuator_ids in self._actuator_ids.items():
            self.scene.data.ctrl[actuator_ids] = target_by_side[side]


__all__ = ["ActuatedPositionTick", "MujocoActuatedPositionExecutor"]
