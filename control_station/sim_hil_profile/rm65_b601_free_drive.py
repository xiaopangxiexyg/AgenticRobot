"""RM65 observation-only free-drive to the B601 MuJoCo target.

This is an application-layer bridge, not a second robot driver.  RM65 state is
read through :class:`RM65FreeDriveSource`; frame conversion and continuous-seed
IK stay in :mod:`robot_motion.rebot_b601_rs.rm65_retarget`; and MuJoCo stepping,
command timing, slew limiting, joint limits, and simulated fault latching stay
in the existing :class:`B601PositionDriver`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from control_station.composition.rebot_b601_rm65_profile import (
    B601RM65FrameMappingProfile,
    load_b601_rm65_frame_mapping_profile,
)
from control_station.sources.rm65_free_drive import (
    RM65FreeDriveSource,
    RM65FreeDriveStep,
)
from robot_motion.rebot_b601_rs import (
    B601DampedLeastSquaresIK,
    B601PinocchioModel,
    RM65ToB601RetargetConfig,
    RM65ToB601Retargeter,
    RM65ToB601RetargetResult,
)
from sim.mujoco.rebot_b601_rs import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_SIM_ARM_KEY,
    B601JointPositionCommand,
    B601PositionDriver,
    B601PositionTick,
    build_b601_mujoco_scene,
)

RM65_TO_B601_FREE_DRIVE_STATES = (
    "inactive",
    "active",
    "clutched",
    "held",
)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return normalized


def _positive_gain(value: object, *, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three positive finite values.") from exc
    if len(values) != 3 or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError(f"{name} must contain three positive finite values.")
    return values


@dataclass(frozen=True)
class RM65ToB601FreeDriveSimConfig:
    """Application parameters that are not part of the calibrated frame graph."""

    translation_gain_target_reference_xyz: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )
    rotation_gain: float = 1.0
    source_poll_hz: float = 20.0
    maximum_source_age_sec: float = 0.25
    maximum_receive_age_sec: float = 0.25
    maximum_future_skew_sec: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_gain_target_reference_xyz",
            _positive_gain(
                self.translation_gain_target_reference_xyz,
                name="translation_gain_target_reference_xyz",
            ),
        )
        object.__setattr__(
            self,
            "rotation_gain",
            _positive_finite(self.rotation_gain, name="rotation_gain"),
        )
        for name in (
            "source_poll_hz",
            "maximum_source_age_sec",
            "maximum_receive_age_sec",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name=name),
            )
        if isinstance(self.maximum_future_skew_sec, bool):
            raise TypeError("maximum_future_skew_sec must be finite and non-negative.")
        future_skew = float(self.maximum_future_skew_sec)
        if not math.isfinite(future_skew) or future_skew < 0.0:
            raise ValueError("maximum_future_skew_sec must be finite and non-negative.")
        object.__setattr__(self, "maximum_future_skew_sec", future_skew)


@dataclass(frozen=True)
class RM65ToB601FreeDriveTick:
    """One bridge tick and its lower-owner audit records."""

    state: str
    source_step: RM65FreeDriveStep | None
    retarget_result: RM65ToB601RetargetResult | None
    position_tick: B601PositionTick
    pending_command: bool
    rejection_reason: str | None = None
    source_step_elapsed_sec: float = 0.0
    retarget_elapsed_sec: float = 0.0
    position_step_elapsed_sec: float = 0.0
    step_elapsed_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.state not in RM65_TO_B601_FREE_DRIVE_STATES:
            raise ValueError(f"Unsupported RM65/B601 bridge state: {self.state!r}")
        if not isinstance(self.position_tick, B601PositionTick):
            raise TypeError("position_tick must be a B601PositionTick.")


class RM65FreeDriveAdmissionError(RuntimeError):
    """The source cannot provide a fresh anchor for activation or recentering."""


class RM65ToB601FreeDriveBridge:
    """Own only reverse-following state above the existing simulation driver."""

    def __init__(
        self,
        source: RM65FreeDriveSource,
        retargeter: RM65ToB601Retargeter,
        driver: B601PositionDriver,
        *,
        source_poll_hz: float = 20.0,
        source_poll_clock: Callable[[], float] | None = None,
    ) -> None:
        if not callable(getattr(source, "step", None)):
            raise TypeError("source must provide step().")
        if not all(
            callable(getattr(retargeter, name, None))
            for name in ("capture_anchor", "retarget", "clear_anchor")
        ):
            raise TypeError(
                "retargeter must provide capture_anchor(), retarget(), and clear_anchor()."
            )
        if not callable(getattr(driver, "step_physics", None)):
            raise TypeError("driver must provide step_physics().")
        self.source = source
        self.retargeter = retargeter
        self.driver = driver
        self.source_poll_hz = _positive_finite(source_poll_hz, name="source_poll_hz")
        self._source_period_sec = 1.0 / self.source_poll_hz
        self._state = "inactive"
        self._pending_command: B601JointPositionCommand | None = None
        self._next_source_poll_time_sec = 0.0
        self._source_poll_clock: Callable[[], float] | None = None
        self._next_source_poll_clock_sec: float | None = None
        self._last_source_step: RM65FreeDriveStep | None = None
        self._last_retarget_result: RM65ToB601RetargetResult | None = None
        self._last_rejection_reason: str | None = None
        self.set_source_poll_clock(source_poll_clock)

    def set_source_poll_clock(self, clock: Callable[[], float] | None) -> None:
        """Select wall-clock scheduling for a live source, if requested.

        Synthetic bridge tests retain the deterministic simulation-clock
        scheduler by default.  The live RM65 application supplies
        ``time.monotonic`` so a slow viewer cannot silently reduce the source
        observation cadence.
        """

        if clock is not None and not callable(clock):
            raise TypeError("source poll clock must be callable or None")
        self._source_poll_clock = clock
        self._next_source_poll_clock_sec = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def pending_command(self) -> B601JointPositionCommand | None:
        return self._pending_command

    @property
    def last_source_step(self) -> RM65FreeDriveStep | None:
        return self._last_source_step

    @property
    def last_retarget_result(self) -> RM65ToB601RetargetResult | None:
        return self._last_retarget_result

    @property
    def last_rejection_reason(self) -> str | None:
        """Return the reason that latched the current held state, if any."""

        return self._last_rejection_reason

    def _schedule_next_source_poll(self) -> None:
        if self._source_poll_clock is None:
            self._next_source_poll_time_sec = (
                self.driver.physics_time_sec + self._source_period_sec
            )
            self._next_source_poll_clock_sec = None
            return
        now = float(self._source_poll_clock())
        if not math.isfinite(now):
            raise ValueError("source poll clock returned a non-finite value")
        self._next_source_poll_clock_sec = now + self._source_period_sec

    def _source_poll_due(self) -> bool:
        if self._source_poll_clock is None:
            return self.driver.physics_time_sec + 1e-12 >= self._next_source_poll_time_sec
        if self._next_source_poll_clock_sec is None:
            return True
        now = float(self._source_poll_clock())
        if not math.isfinite(now):
            raise ValueError("source poll clock returned a non-finite value")
        return now + 1e-12 >= self._next_source_poll_clock_sec

    def _enter_held(self, reason: str) -> None:
        self._last_rejection_reason = str(reason).strip() or "bridge_rejected_target"
        self._state = "held"

    def activate(self) -> None:
        """Capture a fresh RM65/B601 anchor and enter active following."""

        if self._state not in {"inactive", "held"}:
            raise RuntimeError(f"activate is invalid in state {self._state!r}")
        self._last_rejection_reason = None
        self._clear_target()
        source_step = self.source.step()
        self._last_source_step = source_step
        if source_step.sample is None:
            self._enter_held(
                source_step.rejection_reason or "RM65 source did not provide an anchor"
            )
            raise RM65FreeDriveAdmissionError(
                source_step.rejection_reason or "RM65 source did not provide an anchor"
            )
        try:
            self.retargeter.capture_anchor(
                source_step.sample,
                target_qpos_rad=self.driver.scene.get_arm_qpos(B601_RS_SIM_ARM_KEY),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._enter_held(str(exc))
            raise RM65FreeDriveAdmissionError(str(exc)) from exc
        self._state = "active"
        self._schedule_next_source_poll()

    def engage_clutch(self) -> None:
        """Stop following and invalidate the source anchor without stepping."""

        if self._state not in {"active", "clutched"}:
            raise RuntimeError(f"engage_clutch is invalid in state {self._state!r}")
        self._clear_target()
        self._last_rejection_reason = None
        self._state = "clutched"

    def release_clutch(self) -> None:
        """Capture a new anchor at the current target and resume following."""

        if self._state != "clutched":
            raise RuntimeError(f"release_clutch is invalid in state {self._state!r}")
        self._clear_target()
        self._last_rejection_reason = None
        source_step = self.source.step()
        self._last_source_step = source_step
        if source_step.sample is None:
            self._enter_held(
                source_step.rejection_reason or "RM65 source did not provide a recenter anchor"
            )
            raise RM65FreeDriveAdmissionError(
                source_step.rejection_reason or "RM65 source did not provide a recenter anchor"
            )
        try:
            self.retargeter.capture_anchor(
                source_step.sample,
                target_qpos_rad=self.driver.scene.get_arm_qpos(B601_RS_SIM_ARM_KEY),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._enter_held(str(exc))
            raise RM65FreeDriveAdmissionError(str(exc)) from exc
        self._state = "active"
        self._schedule_next_source_poll()

    def deactivate(self) -> None:
        """Stop following and discard all source-derived targets."""

        self._clear_target()
        self._last_rejection_reason = None
        self._state = "inactive"

    def step(self) -> RM65ToB601FreeDriveTick:
        """Advance one existing B601 physics tick.

        A rejected source or IK result revokes the pending command and enters
        ``held``.  The position driver may continue its local zero-order hold
        for this physics tick, but no rejected or stale qpos is re-issued.
        """

        step_started = time.perf_counter()
        source_step: RM65FreeDriveStep | None = None
        retarget_result: RM65ToB601RetargetResult | None = None
        source_step_elapsed_sec = 0.0
        retarget_elapsed_sec = 0.0
        position_step_elapsed_sec = 0.0
        tick_rejection_reason: str | None = None
        if self._state == "active" and self._source_poll_due():
            source_started = time.perf_counter()
            source_step = self.source.step()
            source_step_elapsed_sec = max(0.0, time.perf_counter() - source_started)
            self._last_source_step = source_step
            # A slow source read must not cause a burst of catch-up hardware
            # refreshes.  The target driver continues at its own physics rate;
            # the next observation is scheduled from the current clock cut.
            self._schedule_next_source_poll()
            if source_step.sample is None:
                self._clear_target()
                tick_rejection_reason = (
                    source_step.rejection_reason or "RM65 source rejected the current sample"
                )
                self._enter_held(tick_rejection_reason)
            else:
                retarget_started = time.perf_counter()
                retarget_result = self.retargeter.retarget(
                    source_step.sample,
                    seed_qpos_rad=self.driver.scene.get_arm_qpos(B601_RS_SIM_ARM_KEY),
                )
                retarget_elapsed_sec = max(0.0, time.perf_counter() - retarget_started)
                self._last_retarget_result = retarget_result
                if retarget_result.accepted:
                    assert retarget_result.qpos_rad is not None
                    self._pending_command = B601JointPositionCommand(
                        sequence=source_step.sample.sequence,
                        source_timestamp_sec=source_step.sample.source_timestamp_sec,
                        receive_timestamp_sec=source_step.sample.receive_timestamp_sec,
                        qpos=retarget_result.qpos_rad,
                    )
                    self._last_rejection_reason = None
                else:
                    self._pending_command = None
                    self.retargeter.clear_anchor()
                    tick_rejection_reason = (
                        retarget_result.rejection_reason
                        or "retargeter rejected the current target"
                    )
                    self._enter_held(tick_rejection_reason)

        command = self._pending_command if self.driver.command_due else None
        position_started = time.perf_counter()
        position_tick = self.driver.step_physics(command)
        position_step_elapsed_sec = max(0.0, time.perf_counter() - position_started)
        if command is not None:
            if position_tick.command_applied:
                self._pending_command = None
            else:
                # A lower-owner rejection is terminal for this source target;
                # retrying the same qpos would replay a possibly stale command.
                self._pending_command = None
                self.retargeter.clear_anchor()
                tick_rejection_reason = (
                    position_tick.rejection_reason or "position driver rejected the target"
                )
                self._enter_held(tick_rejection_reason)
        if tick_rejection_reason is None and self._state == "held":
            tick_rejection_reason = self._last_rejection_reason
        return RM65ToB601FreeDriveTick(
            state=self._state,
            source_step=source_step,
            retarget_result=retarget_result,
            position_tick=position_tick,
            pending_command=self._pending_command is not None,
            rejection_reason=tick_rejection_reason,
            source_step_elapsed_sec=source_step_elapsed_sec,
            retarget_elapsed_sec=retarget_elapsed_sec,
            position_step_elapsed_sec=position_step_elapsed_sec,
            step_elapsed_sec=max(0.0, time.perf_counter() - step_started),
        )

    def _clear_target(self) -> None:
        self._pending_command = None
        self._last_retarget_result = None
        self.retargeter.clear_anchor()


def build_rm65_to_b601_free_drive_bridge(
    source_root: str | Path,
    source: RM65FreeDriveSource,
    mapping_profile: B601RM65FrameMappingProfile,
    *,
    initial_qpos: Sequence[float] = B601_RS_DEFAULT_ARM_QPOS,
    config: RM65ToB601FreeDriveSimConfig | None = None,
    source_poll_clock: Callable[[], float] | None = None,
) -> RM65ToB601FreeDriveBridge:
    """Build the reverse sim bridge from the existing model/profile contracts."""

    if not isinstance(mapping_profile, B601RM65FrameMappingProfile):
        raise TypeError("mapping_profile must be a B601RM65FrameMappingProfile.")
    if not mapping_profile.enabled:
        raise ValueError("RM65/B601 reverse following requires an enabled mapping profile.")
    resolved_config = config or RM65ToB601FreeDriveSimConfig()
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    solver = B601DampedLeastSquaresIK(model)
    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=initial_qpos,
        actuation_mode="position",
    )
    driver = B601PositionDriver(scene)
    driver.reset(initial_qpos)
    retargeter = RM65ToB601Retargeter(
        RM65ToB601RetargetConfig(
            frame_graph=mapping_profile.build_frame_graph(),
            source_reference_frame_id=mapping_profile.target_reference_frame_id,
            source_controlled_frame_id=mapping_profile.target_controlled_frame_id,
            target_reference_frame_id=mapping_profile.source_reference_frame_id,
            target_controlled_frame_id=mapping_profile.source_controlled_frame_id,
            translation_gain_target_reference_xyz=(
                resolved_config.translation_gain_target_reference_xyz
            ),
            rotation_gain=resolved_config.rotation_gain,
            maximum_source_age_sec=resolved_config.maximum_source_age_sec,
            maximum_receive_age_sec=resolved_config.maximum_receive_age_sec,
            maximum_future_skew_sec=resolved_config.maximum_future_skew_sec,
        ),
        model,
        solver,
        wall_clock=source.wall_clock,
        monotonic_clock=source.monotonic_clock,
    )
    return RM65ToB601FreeDriveBridge(
        source,
        retargeter,
        driver,
        source_poll_hz=resolved_config.source_poll_hz,
        source_poll_clock=source_poll_clock,
    )


def load_rm65_to_b601_mapping_profile(
    path: str | Path,
) -> B601RM65FrameMappingProfile:
    """Load the canonical mapping profile without introducing a reverse copy."""

    return load_b601_rm65_frame_mapping_profile(path)


__all__ = [
    "RM65_TO_B601_FREE_DRIVE_STATES",
    "RM65FreeDriveAdmissionError",
    "RM65ToB601FreeDriveBridge",
    "RM65ToB601FreeDriveSimConfig",
    "RM65ToB601FreeDriveTick",
    "build_rm65_to_b601_free_drive_bridge",
    "load_rm65_to_b601_mapping_profile",
]
