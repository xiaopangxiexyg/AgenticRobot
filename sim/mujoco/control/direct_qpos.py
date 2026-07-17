from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from sim.contracts import TimingSpec

from ..scene import MujocoScene


DIRECT_QPOS_DEBUG_EXECUTION_MODE = "kinematic_direct_qpos_debug"


def _finite_non_negative(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.")
    return normalized


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _qpos_mapping(
    values: Mapping[str, Sequence[float]],
    *,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must contain at least one side.")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, raw_qpos in values.items():
        side = str(raw_side).strip()
        if not side:
            raise ValueError(f"{name} contains an empty side.")
        if side in normalized:
            raise ValueError(f"{name} contains duplicate side {side!r}.")
        qpos = np.asarray(raw_qpos, dtype=float).reshape(-1)
        if qpos.size != 6 or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{name}[{side!r}] must contain six finite qpos values.")
        qpos = qpos.copy()
        qpos.setflags(write=False)
        normalized[side] = qpos
    return MappingProxyType(normalized)


def _gripper_mapping(values: Mapping[str, float] | None) -> Mapping[str, float]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("gripper_close_by_name must be a mapping.")
    normalized: dict[str, float] = {}
    for raw_name, raw_close in values.items():
        name = str(raw_name).strip()
        close = float(raw_close)
        if not name:
            raise ValueError("gripper_close_by_name contains an empty name.")
        if not math.isfinite(close) or not 0.0 <= close <= 1.0:
            raise ValueError(f"gripper close for {name!r} must be finite and in [0, 1].")
        normalized[name] = close
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class DirectQposCommand:
    """One atomic command-rate joint-position command for the MuJoCo debug bridge."""

    sequence: int
    qpos_by_side: Mapping[str, Sequence[float]]
    gripper_close_by_name: Mapping[str, float] | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must not be empty.")
        object.__setattr__(self, "sequence", _non_negative_int(self.sequence, name="sequence"))
        object.__setattr__(
            self,
            "qpos_by_side",
            _qpos_mapping(self.qpos_by_side, name="qpos_by_side"),
        )
        object.__setattr__(
            self,
            "gripper_close_by_name",
            _gripper_mapping(self.gripper_close_by_name),
        )
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class DirectQposDebugTick:
    """Audit record for one MuJoCo physics tick in direct-qpos debug mode."""

    physics_tick: int
    physics_time_sec: float
    command_due: bool
    scheduled_command_time_sec: float | None
    active_command_sequence: int
    command_applied: bool
    safety_status: str = "unavailable"
    target_modified: bool = False
    projection_alpha: float = 1.0
    minimum_distance_m: float = float("inf")
    collision_pair: tuple[str | None, str | None] = (None, None)


class DirectQposDebugController:
    """Drive a scene at physics rate from explicit 150 Hz qpos commands.

    Each ``step_physics`` calls the scene's prescribed-motion debug bridge.
    The controller never performs pose IK or qpos interpolation: callers must
    provide the canonical, fully formed high-rate command produced upstream.
    ``command_due`` uses timestamp deadlines rather than an integer clock-ratio
    assumption, so configured command rates (including 120 or 150 Hz) schedule
    correctly against 1000 Hz physics.
    """

    def __init__(
        self,
        scene: MujocoScene,
        timing: TimingSpec,
        *,
        active_sides: Sequence[str] | None = None,
        zero_controlled_qvel: bool = True,
        record_ticks: bool = True,
    ) -> None:
        if not isinstance(timing, TimingSpec):
            raise TypeError("timing must be a SimulationContract TimingSpec.")
        self.scene = scene
        self.timing = timing
        model_timestep = getattr(getattr(scene, "model", None), "opt", None)
        if model_timestep is not None:
            physics_timestep = float(model_timestep.timestep)
            expected_timestep = timing.physics_timestep_sec
            if not math.isclose(physics_timestep, expected_timestep, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "scene model timestep does not match TimingSpec.physics_hz: "
                    f"model={physics_timestep}, timing={expected_timestep}."
                )
        sides = tuple(str(side) for side in (active_sides or tuple(sorted(scene.arm_specs))))
        if not sides or len(set(sides)) != len(sides):
            raise ValueError("active_sides must be a non-empty unique side sequence.")
        missing_sides = sorted(set(sides) - set(scene.arm_specs))
        if missing_sides:
            raise KeyError(f"scene is missing active arm side(s): {missing_sides}")
        self.active_sides = sides
        self.zero_controlled_qvel = bool(zero_controlled_qvel)
        self.record_ticks = bool(record_ticks)
        self._physics_tick = 0
        self._next_command_time_sec = 0.0
        self._held_command: DirectQposCommand | None = None
        self._accepted_controlled_qpos: np.ndarray | None = None
        self._last_requested_controlled_qpos: np.ndarray | None = None
        self._last_safety_result = None
        self._last_active_collision_check = None
        self._ticks: list[DirectQposDebugTick] = []

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": DIRECT_QPOS_DEBUG_EXECUTION_MODE,
            "robot_dynamics": "not_simulated",
            "command_hz": float(self.timing.command_hz),
            "physics_hz": float(self.timing.physics_hz),
            "command_hold": "zero_order_hold_between_command_ticks",
            "zero_controlled_qvel": self.zero_controlled_qvel,
        }

    @property
    def physics_time_sec(self) -> float:
        return self._physics_tick / self.timing.physics_hz

    @property
    def ticks(self) -> tuple[DirectQposDebugTick, ...]:
        return tuple(self._ticks)

    @property
    def last_safety_result(self):
        return self._last_safety_result

    def accepted_arm_qpos(self, side: str) -> np.ndarray:
        if self._accepted_controlled_qpos is None:
            return self.scene.get_arm_qpos(side)
        return self.scene.arm_qpos_from_kinematic_vector(
            self._accepted_controlled_qpos, side
        )

    def accepted_gripper_close(self, name: str) -> float:
        if self._accepted_controlled_qpos is None:
            return float(self.scene.get_gripper_close(name))
        return float(
            self.scene.gripper_close_from_kinematic_vector(
                self._accepted_controlled_qpos, name
            )
        )

    @property
    def last_active_collision_check(self):
        return self._last_active_collision_check

    @property
    def command_due(self) -> bool:
        return self.physics_time_sec + 1e-12 >= self._next_command_time_sec

    def reset(self, initial_command: DirectQposCommand) -> None:
        self._validate_command(initial_command, require_newer_than=None)
        self._physics_tick = 0
        # ``initial_command`` is already the command at t=0.  The first call
        # to step_physics therefore holds it and asks for a new command only at
        # the following configured command-rate deadline.
        self._next_command_time_sec = 1.0 / self.timing.command_hz
        self._held_command = initial_command
        if hasattr(self.scene, "kinematic_controlled_qpos"):
            current = self.scene.kinematic_controlled_qpos()
            target = self.scene.kinematic_target_vector(
                current,
                initial_command.qpos_by_side,
                gripper_close_by_name=initial_command.gripper_close_by_name,
            )
            projector = self.scene.collision_projector()
            if projector is not None:
                check = projector.check_qpos(self.scene.data, target)
                if not check.safe:
                    raise RuntimeError(
                        "Initial direct-qpos command is collision-unsafe: "
                        f"{check.to_dict()}"
                    )
            self._accepted_controlled_qpos = target
            self._last_requested_controlled_qpos = target.copy()
        else:
            # Small protocol fakes and legacy collision-disabled callers retain
            # the original prescribed-motion surface. Real MujocoScene objects
            # always take the validated path above.
            self._accepted_controlled_qpos = None
            self._last_requested_controlled_qpos = None
        self._last_safety_result = None
        self._last_active_collision_check = None
        self._ticks.clear()

    def step_physics(
        self,
        command: DirectQposCommand | None = None,
    ) -> DirectQposDebugTick:
        """Apply one prescribed robot state and advance exactly one physics tick.

        A command is required on the first physics tick at or after every
        configured command-rate deadline. Passing one earlier is rejected so
        callers cannot silently alter the intended command clock.
        """

        due = self.command_due
        scheduled_time = self._next_command_time_sec if due else None
        if due:
            if command is None:
                raise RuntimeError(
                    "A DirectQposCommand is required at the current command deadline."
                )
            self._validate_command(command, require_newer_than=self._held_command)
            self._held_command = command
            command_applied = True
            command_period = 1.0 / self.timing.command_hz
            while self._next_command_time_sec <= self.physics_time_sec + 1e-12:
                self._next_command_time_sec += command_period
        else:
            if command is not None:
                raise RuntimeError("DirectQposCommand arrived before the next command deadline.")
            if self._held_command is None:
                raise RuntimeError("reset(initial_command) is required before stepping physics.")
            command_applied = False

        assert self._held_command is not None
        legacy_scene = self._accepted_controlled_qpos is None
        if command_applied and not legacy_scene:
            assert self._accepted_controlled_qpos is not None
            requested = self.scene.kinematic_target_vector(
                self._accepted_controlled_qpos,
                self._held_command.qpos_by_side,
                gripper_close_by_name=self._held_command.gripper_close_by_name,
            )
            target_changed = bool(
                self._last_requested_controlled_qpos is None
                or not np.array_equal(requested, self._last_requested_controlled_qpos)
            )
            self._last_requested_controlled_qpos = requested.copy()
            projector = self.scene.collision_projector()
            if projector is None:
                self._accepted_controlled_qpos = requested
                self._last_safety_result = None
            elif target_changed:
                from ..collision_safety import project_grippers_then_arms

                self._last_safety_result = project_grippers_then_arms(
                    self.scene,
                    projector,
                    self.scene.data,
                    start_qpos=self._accepted_controlled_qpos,
                    qpos_by_side=self._held_command.qpos_by_side,
                    gripper_close_by_name=self._held_command.gripper_close_by_name,
                )
                if self._last_safety_result.status == "invalid_safe_state":
                    raise RuntimeError(
                        "Direct-qpos controller lost its collision-safe state: "
                        f"{self._last_safety_result.to_dict()}"
                    )
                self._accepted_controlled_qpos = self._last_safety_result.accepted_qpos.copy()

        safety_result = self._last_safety_result
        tick = DirectQposDebugTick(
            physics_tick=self._physics_tick,
            physics_time_sec=self.physics_time_sec,
            command_due=due,
            scheduled_command_time_sec=scheduled_time,
            active_command_sequence=self._held_command.sequence,
            command_applied=command_applied,
            safety_status=("unavailable" if safety_result is None else safety_result.status),
            target_modified=(False if safety_result is None else safety_result.modified),
            projection_alpha=(1.0 if safety_result is None else safety_result.projection_alpha),
            minimum_distance_m=(
                float("inf") if safety_result is None else safety_result.check.minimum_distance_m
            ),
            collision_pair=(
                (None, None)
                if safety_result is None
                else (safety_result.check.body1_name, safety_result.check.body2_name)
            ),
        )
        if legacy_scene:
            self.scene.step_kinematic_direct_qpos_debug(
                self._held_command.qpos_by_side,
                gripper_close_by_name=self._held_command.gripper_close_by_name,
                zero_controlled_qvel=self.zero_controlled_qvel,
            )
        else:
            assert self._accepted_controlled_qpos is not None
            self.scene.write_kinematic_controlled_qpos(
                self._accepted_controlled_qpos,
                zero_controlled_qvel=self.zero_controlled_qvel,
            )
            self.scene.forward()
            self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
            self.scene.write_kinematic_controlled_qpos(
                self._accepted_controlled_qpos,
                zero_controlled_qvel=self.zero_controlled_qvel,
            )
            self.scene.forward()
            projector = self.scene.collision_projector()
            self._last_active_collision_check = (
                None
                if projector is None
                else projector.oracle.check_contacts(self.scene.data)
            )
        if self.record_ticks:
            self._ticks.append(tick)
        self._physics_tick += 1
        return tick

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
                "command qpos sides must exactly match active_sides: "
                f"expected {self.active_sides}, got {tuple(command.qpos_by_side)}."
            )
        unknown_grippers = sorted(set(command.gripper_close_by_name) - set(self.scene.gripper_specs))
        if unknown_grippers:
            raise KeyError(f"command references unknown gripper(s): {unknown_grippers}")
        if require_newer_than is not None and command.sequence <= require_newer_than.sequence:
            raise ValueError(
                f"command sequence {command.sequence} must be newer than "
                f"active sequence {require_newer_than.sequence}."
            )


__all__ = [
    "DIRECT_QPOS_DEBUG_EXECUTION_MODE",
    "DirectQposCommand",
    "DirectQposDebugController",
    "DirectQposDebugTick",
]
