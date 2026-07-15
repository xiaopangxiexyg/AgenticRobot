"""Policy-rate stepping over the existing 150 Hz / 1000 Hz direct controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ...control import DirectQposCommand, DirectQposDebugController, DirectQposDebugTick


@dataclass(frozen=True)
class PolicyAdvance:
    physics_ticks: int
    command_ticks: int
    stopped_early: bool
    final_physics_time_sec: float


class CubeLiftPolicyScheduler:
    """Use timestamp deadlines, never a rounded policy-to-physics tick ratio."""

    def __init__(self, controller: DirectQposDebugController, *, policy_hz: float) -> None:
        self.controller = controller
        self.policy_hz = float(policy_hz)
        if self.policy_hz <= 0.0:
            raise ValueError("policy_hz must be positive.")
        self._next_policy_time_sec: float | None = None

    @property
    def policy_period_sec(self) -> float:
        return 1.0 / self.policy_hz

    def reset(self, initial_command: DirectQposCommand) -> None:
        self.controller.reset(initial_command)
        self._next_policy_time_sec = None

    def start_episode_clock(self) -> None:
        self._next_policy_time_sec = self.controller.physics_time_sec + self.policy_period_sec

    def advance_physics_ticks(
        self,
        ticks: int,
        *,
        command_supplier: Callable[[], DirectQposCommand],
        after_tick: Callable[[DirectQposDebugTick], bool] | None = None,
    ) -> PolicyAdvance:
        if ticks < 0:
            raise ValueError("ticks must be non-negative.")
        physics_ticks = 0
        command_ticks = 0
        stopped_early = False
        for _ in range(ticks):
            tick = self._step_one(command_supplier)
            physics_ticks += 1
            command_ticks += int(tick.command_applied)
            if after_tick is not None and after_tick(tick):
                stopped_early = True
                break
        return PolicyAdvance(
            physics_ticks=physics_ticks,
            command_ticks=command_ticks,
            stopped_early=stopped_early,
            final_physics_time_sec=self.controller.physics_time_sec,
        )

    def advance_one_policy_step(
        self,
        *,
        command_supplier: Callable[[], DirectQposCommand],
        after_tick: Callable[[DirectQposDebugTick], bool] | None = None,
    ) -> PolicyAdvance:
        if self._next_policy_time_sec is None:
            raise RuntimeError("start_episode_clock() is required before policy stepping.")
        physics_ticks = 0
        command_ticks = 0
        stopped_early = False
        while self.controller.physics_time_sec + 1e-12 < self._next_policy_time_sec:
            tick = self._step_one(command_supplier)
            physics_ticks += 1
            command_ticks += int(tick.command_applied)
            if after_tick is not None and after_tick(tick):
                stopped_early = True
                break
        self._next_policy_time_sec += self.policy_period_sec
        return PolicyAdvance(
            physics_ticks=physics_ticks,
            command_ticks=command_ticks,
            stopped_early=stopped_early,
            final_physics_time_sec=self.controller.physics_time_sec,
        )

    def _step_one(
        self,
        command_supplier: Callable[[], DirectQposCommand],
    ) -> DirectQposDebugTick:
        command = command_supplier() if self.controller.command_due else None
        return self.controller.step_physics(command)


__all__ = ["CubeLiftPolicyScheduler", "PolicyAdvance"]
