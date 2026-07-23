"""Backend-independent causal RM65 qcmd-to-actuator-target execution."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .profile import PositionActuationProfile


@dataclass(frozen=True)
class Rm65ActuatorTick:
    """One actuator-clock step, independent of a physics backend."""

    physics_tick: int
    physics_time_sec: float
    command_due: bool
    scheduled_command_time_sec: float | None
    accepted_command_sequence: int
    command_applied: bool
    actuator_target_by_side: Mapping[str, np.ndarray]


class Rm65PositionActuator:
    """Causal 120 Hz qcmd transport and target dynamics for RM65 simulation.

    This core owns accepted qcmd, per-joint transport delay, raw controller
    target, and filtered actuator target.  It does not import MuJoCo, SAPIEN,
    task code, real hardware, or teleoperation runtime.  A backend adapter only
    writes :attr:`actuator_target_by_side` to its physical servo and advances
    the backend dynamics.
    """

    def __init__(
        self,
        profile: PositionActuationProfile,
        *,
        command_hz: float,
        physics_hz: float,
        active_sides: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(profile, PositionActuationProfile):
            raise TypeError("profile must be a PositionActuationProfile.")
        command_rate = float(command_hz)
        physics_rate = float(physics_hz)
        if not math.isfinite(command_rate) or command_rate <= 0.0:
            raise ValueError("command_hz must be finite and positive.")
        if not math.isfinite(physics_rate) or physics_rate <= 0.0:
            raise ValueError("physics_hz must be finite and positive.")
        sides = tuple(
            str(side)
            for side in (
                tuple(profile.joint_names_by_side)
                if active_sides is None
                else active_sides
            )
        )
        if not sides or len(set(sides)) != len(sides):
            raise ValueError("active_sides must be a non-empty unique sequence.")
        unknown = set(sides) - set(profile.joint_names_by_side)
        if unknown:
            raise ValueError(f"active_sides are absent from profile: {sorted(unknown)}")
        self.profile = profile
        self.command_hz = command_rate
        self.physics_hz = physics_rate
        self.active_sides = sides
        self._physics_tick = 0
        self._next_command_time_sec = 0.0
        self._accepted_sequence: int | None = None
        self._accepted_qcmd_by_side: dict[str, np.ndarray] = {}
        self._raw_actuator_target_by_side: dict[str, np.ndarray] = {}
        self._actuator_target_by_side: dict[str, np.ndarray] = {}
        self._actuator_target_tau_sec_by_side = {
            side: np.zeros(6, dtype=float) for side in sides
        }
        self._delayed_targets: list[
            tuple[float, int, str, int, float, float]
        ] = []

    @property
    def physics_timestep_sec(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def physics_time_sec(self) -> float:
        return self._physics_tick * self.physics_timestep_sec

    @property
    def command_due(self) -> bool:
        self._require_reset()
        return self.physics_time_sec + 1e-12 >= self._next_command_time_sec

    @property
    def actuator_target_by_side(self) -> Mapping[str, np.ndarray]:
        self._require_reset()
        return {
            side: values.copy()
            for side, values in self._actuator_target_by_side.items()
        }

    def accepted_qpos(self, side: str) -> np.ndarray:
        self._require_reset()
        try:
            return self._accepted_qcmd_by_side[str(side)].copy()
        except KeyError as exc:
            raise KeyError(f"inactive RM65 side: {side!r}") from exc

    def reset(
        self,
        *,
        command_sequence: int,
        qpos_by_side: Mapping[str, Sequence[float] | np.ndarray],
    ) -> None:
        qpos = self._validated_qpos(qpos_by_side)
        self._physics_tick = 0
        self._next_command_time_sec = 1.0 / self.command_hz
        self._accepted_sequence = int(command_sequence)
        self._delayed_targets.clear()
        for values in self._actuator_target_tau_sec_by_side.values():
            values.fill(0.0)
        self._accepted_qcmd_by_side = {
            side: values.copy() for side, values in qpos.items()
        }
        self._raw_actuator_target_by_side = {
            side: values.copy() for side, values in qpos.items()
        }
        self._actuator_target_by_side = {
            side: values.copy() for side, values in qpos.items()
        }

    def step(
        self,
        *,
        command_sequence: int | None = None,
        qpos_by_side: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    ) -> Rm65ActuatorTick:
        """Accept an optional due qcmd and advance one actuator physics tick."""

        self._require_reset()
        due = self.command_due
        scheduled = self._next_command_time_sec if due else None
        command_applied = False
        if due:
            if command_sequence is None or qpos_by_side is None:
                raise RuntimeError("A qcmd is required at the current command deadline.")
            sequence = int(command_sequence)
            assert self._accepted_sequence is not None
            if sequence <= self._accepted_sequence:
                raise ValueError("command_sequence must increase at each command send.")
            next_qcmd = self._validated_qpos(qpos_by_side)
            period = 1.0 / self.command_hz
            for side in self.active_sides:
                velocity = np.abs(
                    next_qcmd[side] - self._accepted_qcmd_by_side[side]
                ) / period
                for joint_index, qpos in enumerate(next_qcmd[side]):
                    heapq.heappush(
                        self._delayed_targets,
                        (
                            self.physics_time_sec
                            + self.profile.command_delay_sec(side, joint_index),
                            sequence,
                            side,
                            joint_index,
                            float(qpos),
                            float(velocity[joint_index]),
                        ),
                    )
            self._accepted_qcmd_by_side = next_qcmd
            self._accepted_sequence = sequence
            command_applied = True
            while self._next_command_time_sec <= self.physics_time_sec + 1e-12:
                self._next_command_time_sec += period
        elif command_sequence is not None or qpos_by_side is not None:
            raise RuntimeError("qcmd arrived before the next command deadline.")

        while (
            self._delayed_targets
            and self._delayed_targets[0][0] <= self.physics_time_sec + 1e-12
        ):
            (
                _apply_time,
                _sequence,
                side,
                joint_index,
                qpos,
                command_velocity,
            ) = heapq.heappop(self._delayed_targets)
            self._raw_actuator_target_by_side[side][joint_index] = qpos
            self._actuator_target_tau_sec_by_side[side][joint_index] = (
                self.profile.target_filter_tau_sec(
                    side,
                    joint_index,
                    abs_command_velocity_rad_s=command_velocity,
                )
            )
        self._advance_targets()
        self._physics_tick += 1
        assert self._accepted_sequence is not None
        return Rm65ActuatorTick(
            physics_tick=self._physics_tick,
            physics_time_sec=self.physics_time_sec,
            command_due=due,
            scheduled_command_time_sec=scheduled,
            accepted_command_sequence=self._accepted_sequence,
            command_applied=command_applied,
            actuator_target_by_side=self.actuator_target_by_side,
        )

    def snapshot(self) -> dict[str, object]:
        self._require_reset()
        return {
            "physics_tick": self._physics_tick,
            "physics_time_sec": self.physics_time_sec,
            "pending_delayed_targets": len(self._delayed_targets),
            "accepted_command_sequence": self._accepted_sequence,
            "accepted_qcmd_by_side": {
                side: values.tolist()
                for side, values in self._accepted_qcmd_by_side.items()
            },
            "actuator_target_by_side": {
                side: values.tolist()
                for side, values in self._actuator_target_by_side.items()
            },
            "raw_actuator_target_by_side": {
                side: values.tolist()
                for side, values in self._raw_actuator_target_by_side.items()
            },
            "actuator_target_tau_sec_by_side": {
                side: values.tolist()
                for side, values in self._actuator_target_tau_sec_by_side.items()
            },
        }

    def _validated_qpos(
        self,
        qpos_by_side: Mapping[str, Sequence[float] | np.ndarray],
    ) -> dict[str, np.ndarray]:
        if set(qpos_by_side) != set(self.active_sides):
            raise ValueError(
                "qpos sides must exactly match active_sides: "
                f"expected={sorted(self.active_sides)} got={sorted(qpos_by_side)}"
            )
        result: dict[str, np.ndarray] = {}
        for side in self.active_sides:
            values = np.asarray(qpos_by_side[side], dtype=float).reshape(-1)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{side} qpos must contain six finite values.")
            result[side] = values.copy()
        return result

    def _require_reset(self) -> None:
        if self._accepted_sequence is None:
            raise RuntimeError("reset(...) is required before actuator execution.")

    def _advance_targets(self) -> None:
        dt = self.physics_timestep_sec
        for side in self.active_sides:
            target = self._actuator_target_by_side[side]
            raw = self._raw_actuator_target_by_side[side]
            tau = self._actuator_target_tau_sec_by_side[side]
            direct = tau <= 1e-9
            target[direct] = raw[direct]
            filtered = ~direct
            if np.any(filtered):
                alpha = 1.0 - np.exp(-dt / tau[filtered])
                target[filtered] += alpha * (raw[filtered] - target[filtered])


__all__ = ["Rm65ActuatorTick", "Rm65PositionActuator"]
