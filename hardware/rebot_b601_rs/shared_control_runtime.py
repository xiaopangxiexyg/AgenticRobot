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

from robot_motion.rebot_b601_rs import (
    B601MitReference,
    B601PolicyAssistCommand,
    B601PolicyCommandBuffer,
    B601PolicyCommandReceipt,
    B601SharedControlConfig,
    B601SharedControlOutput,
    B601SharedControlState,
    compute_b601_shared_control_step,
)

_ZERO_SIX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class B601SharedControlFrame:
    """One owner-tick reference plus its auditable policy-assist decision."""

    reference: B601MitReference
    decision: B601SharedControlOutput


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

    def step(
        self,
        *,
        control_sequence: int,
        control_timestamp_sec: float,
        measured_qpos_rad: Sequence[float],
        measured_qvel_rad_s: Sequence[float],
        dt_sec: float,
        human_override_detected: bool = False,
    ) -> B601SharedControlFrame:
        """Produce a current-position MIT reference with bounded assist only.

        The returned reference deliberately uses measured position, zero
        reference velocity and no fixed position spring.  The receiving owner
        must select zero software/vendor position Kp for the active shared-
        control phase, add gravity/passive damping, condition the total torque,
        and dispatch it through its existing single-owner path.
        """

        sample = self._commands.sample_for_config(
            now_sec=control_timestamp_sec,
            config=self.config,
        )
        decision = compute_b601_shared_control_step(
            self.config,
            self._state,
            command_sample=sample,
            measured_qpos_rad=measured_qpos_rad,
            measured_qvel_rad_s=measured_qvel_rad_s,
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
        return B601SharedControlFrame(reference=reference, decision=decision)


__all__ = [
    "B601SharedControlFrame",
    "B601SharedControlRuntime",
]
