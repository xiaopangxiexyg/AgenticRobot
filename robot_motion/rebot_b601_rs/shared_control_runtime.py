"""Single-owner adapter from B601 policy intent to an MIT overlay.

The adapter is intentionally I/O-free.  A hardware or simulation owner calls
``step`` only after admitting a complete measured state, then sends the
returned reference through its existing gravity, total-torque conditioning and
dispatch path.  No policy thread can obtain a motor handle through this API.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from .joint_motion import B601WindowVelocityEstimator
from .mit import B601MitReference
from .shared_control import (
    B601PolicyAssistCommand,
    B601PolicyCommandBuffer,
    B601PolicyCommandReceipt,
    B601PolicyCommandSample,
    B601SharedControlConfig,
    B601SharedControlOutput,
    B601SharedControlState,
    _finite,
    _positive,
    _sequence,
    _six_finite,
    compute_b601_shared_control_step,
)

_ZERO_SIX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class B601SharedControlFrame:
    """One owner-tick reference plus its auditable policy-assist decision."""

    reference: B601MitReference
    decision: B601SharedControlOutput
    control_qvel_rad_s: tuple[float, ...] | None = None


class B601SharedControlRuntime:
    """Latest-command shared-control component for one authoritative owner.

    ``publish_policy`` may be called from a policy-rate producer.  ``step`` and
    state mutation belong to exactly one servo owner.  The explicit policy
    clutch is independently writable so an operator path can remove assist
    without waiting for another policy sample.
    """

    def __init__(self, config: B601SharedControlConfig) -> None:
        if not isinstance(config, B601SharedControlConfig):
            raise TypeError("config must be a B601SharedControlConfig")
        self.config = config
        self._commands = B601PolicyCommandBuffer()
        self._state = B601SharedControlState()
        self._clutch_lock = threading.Lock()
        self._policy_clutch = False
        self._velocity_estimators = (
            tuple(
                B601WindowVelocityEstimator(config.damping_velocity_window_sec)
                for _ in range(6)
            )
            if config.damping_velocity_window_sec
            else ()
        )
        self._last_velocity_feedback: tuple[int, float] | None = None
        self._control_qvel = _ZERO_SIX

    def _damping_velocity(
        self,
        sequence: int,
        timestamp: float,
        qpos: tuple[float, ...],
        raw_qvel: tuple[float, ...],
        joint_timestamps: Sequence[float] | None,
    ) -> tuple[float, ...]:
        if not self._velocity_estimators:
            return raw_qvel
        if joint_timestamps is None:
            raise ValueError(
                "shared damping requires six joint host-receive timestamps"
            )
        times = _six_finite(joint_timestamps, name="joint_observation_timestamps_sec")
        previous = self._last_velocity_feedback
        if previous is not None and (
            sequence <= previous[0] or timestamp <= previous[1]
        ):
            raise ValueError("shared damping feedback identity must advance")
        # Validate the complete cut before mutating any estimator. Equal source
        # times are reuse, not new samples; changed positions at that time are
        # inconsistent evidence. Host receive time is NOT device source time.
        for estimator, observed, q in zip(
            self._velocity_estimators, times, qpos, strict=True
        ):
            # Receive ages have nanosecond resolution. Allow only arithmetic
            # round-off at the cut, not a new millisecond clock-skew budget.
            # Freshness admission remains the original feedback owner's job;
            # do not turn the unrelated policy TTL into a second age gate.
            if observed < 0 or timestamp - observed < -1e-9:
                raise ValueError("shared damping joint timestamp negative or future")
            last = estimator.latest_observation
            if last is not None and (
                observed < last[0] or (observed == last[0] and q != last[1])
            ):
                raise ValueError(
                    "shared damping joint evidence regressed or changed at equal time"
                )
        velocity = []
        for i, (estimator, observed, q, raw) in enumerate(
            zip(self._velocity_estimators, times, qpos, raw_qvel, strict=True)
        ):
            last = estimator.latest_observation
            # A 20 ms report interval can exceed a short smoothing window.
            # The estimator then resets, but its fallback must STILL use this
            # joint's receive interval, never the unrelated snapshot interval.
            fallback = (
                (q - last[1]) / (observed - last[0])
                if last is not None
                and 0 < observed - last[0] <= self.config.policy_stale_timeout_sec
                else raw
            )
            velocity.append(
                self._control_qvel[i]
                if last is not None and observed == last[0]
                else estimator.update(
                    timestamp_sec=observed, qpos_rad=q, fallback_rad_s=fallback
                )
            )
        self._last_velocity_feedback = (sequence, timestamp)
        self._control_qvel = tuple(velocity)
        return self._control_qvel

    @property
    def state(self) -> B601SharedControlState:
        return self._state

    @property
    def policy_clutch(self) -> bool:
        with self._clutch_lock:
            return self._policy_clutch

    def set_policy_clutch(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise TypeError("active must be a bool")
        with self._clutch_lock:
            self._policy_clutch = active

    def publish_policy(
        self,
        command: B601PolicyAssistCommand,
    ) -> B601PolicyCommandReceipt:
        return self._commands.publish(command)

    def clear_policy(self) -> int | None:
        """Withdraw current policy assist while retaining replay protection."""

        return self._commands.clear()

    def sample_policy(self, *, now_sec: float) -> B601PolicyCommandSample:
        """Read canonical intent validity without advancing servo state."""
        return self._commands.sample_for_config(now_sec=now_sec, config=self.config)

    def step(
        self,
        *,
        control_sequence: int,
        control_timestamp_sec: float,
        measured_qpos_rad: Sequence[float],
        measured_qvel_rad_s: Sequence[float],
        dt_sec: float,
        human_override_detected: bool = False,
        joint_observation_timestamps_sec: Sequence[float] | None = None,
    ) -> B601SharedControlFrame:
        """Produce a current-position MIT reference with bounded assist only.

        The returned reference deliberately uses measured position, zero
        reference velocity and no fixed position spring.  The receiving owner
        must select zero software/vendor position Kp for the active shared-
        control phase, add gravity/passive damping, condition the total torque,
        and dispatch it through its existing single-owner path.
        """

        qpos = _six_finite(measured_qpos_rad, name="measured_qpos_rad")
        raw_qvel = _six_finite(measured_qvel_rad_s, name="measured_qvel_rad_s")
        control_sequence = _sequence(control_sequence, name="control_sequence")
        control_timestamp_sec = _finite(
            control_timestamp_sec, name="control_timestamp_sec"
        )
        if control_timestamp_sec < 0:
            raise ValueError("control_timestamp_sec must be non-negative")
        dt_sec = _positive(dt_sec, name="dt_sec")
        if not isinstance(human_override_detected, bool):
            raise TypeError("human_override_detected must be a bool")
        sample = self.sample_policy(now_sec=control_timestamp_sec)
        qvel = self._damping_velocity(
            control_sequence,
            control_timestamp_sec,
            qpos,
            raw_qvel,
            joint_observation_timestamps_sec,
        )
        decision = compute_b601_shared_control_step(
            self.config,
            self._state,
            command_sample=sample,
            measured_qpos_rad=qpos,
            measured_qvel_rad_s=qvel,
            dt_sec=dt_sec,
            policy_clutch=self.policy_clutch,
            human_override_detected=human_override_detected,
        )
        reference = B601MitReference(
            sequence=control_sequence,
            source_timestamp_sec=control_timestamp_sec,
            receive_timestamp_sec=control_timestamp_sec,
            qpos_rad=tuple(measured_qpos_rad),
            qvel_rad_s=_ZERO_SIX,
            joint_torque_overlay_nm=decision.assist_torque_nm,
        )
        self._state = decision.state
        return B601SharedControlFrame(reference=reference, decision=decision,
            control_qvel_rad_s=qvel if self._velocity_estimators else None)


__all__ = [
    "B601SharedControlFrame",
    "B601SharedControlRuntime",
]
