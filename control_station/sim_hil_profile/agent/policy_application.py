"""Lifecycle owner for one policy-backed semantic Skill attempt.

The application serializes a backend-neutral policy emission source with one
single-policy MuJoCo target.  It does not define Skill semantics, termination,
evaluation, or a public Gateway Route.  It also does not compose sources or
own lower safety, IK, reference generation, dispatch, or HOLD proof.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

from robot_motion.deploy import (
    PolicyActionEmission,
    PolicyEmissionSourcePort,
    require_policy_emission_source,
)
from robot_motion.target_safety import (
    TCPPoseEnvelopeLifecycleError,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSelection,
)
from sim.mujoco.deploy import MujocoPolicySkillMotionTarget


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class MujocoPolicySkillMotionCycle:
    """One source poll, optional admission, and lower decision cut."""

    observed_at: float
    emission: PolicyActionEmission | None
    proposal_submission: Any | None
    authoritative_step: Any

    def __post_init__(self) -> None:
        observed = float(self.observed_at)
        if not math.isfinite(observed):
            raise ValueError("observed_at must be finite.")
        if self.emission is not None and not isinstance(
            self.emission,
            PolicyActionEmission,
        ):
            raise TypeError("emission must be PolicyActionEmission or None.")
        if (self.emission is None) != (self.proposal_submission is None):
            raise ValueError(
                "proposal_submission must exist exactly when an emission exists."
            )
        object.__setattr__(self, "observed_at", observed)


class MujocoPolicySkillMotionApplication:
    """Default-off single-policy application reusable across Skill attempts."""

    def __init__(
        self,
        *,
        target: MujocoPolicySkillMotionTarget,
        policy_source: PolicyEmissionSourcePort,
    ) -> None:
        if not isinstance(target, MujocoPolicySkillMotionTarget):
            raise TypeError("target must be a MujocoPolicySkillMotionTarget.")
        runtime = target.runtime
        config = getattr(runtime, "config", None)
        binding = getattr(runtime, "component_execution_binding", None)
        if (
            getattr(config, "enable_primary_policy_authority", None) is not True
            or getattr(config, "enable_composer_authority", None) is not False
            or getattr(config, "enable_replay_authority", None) is not False
            or getattr(config, "initial_control_mode", None) != "policy"
            or getattr(config, "direct_component_proposal_source", None) is not None
            or getattr(runtime, "composition_runtime_config", None) is not None
            or not callable(getattr(binding, "policy_action_ingress", None))
        ):
            raise ValueError(
                "Policy Skill application requires the exact single-policy, "
                "no-composition target."
            )
        safety_runtime = target.tcp_pose_envelope_runtime
        if safety_runtime is not None and not isinstance(
            safety_runtime,
            TCPPoseEnvelopeSafetyRuntime,
        ):
            raise TypeError("target TCP envelope runtime has an invalid type.")
        self.target = target
        self.task_config = target.task_config
        self.scene = target.scene
        self.runtime = runtime
        self.policy_source = require_policy_emission_source(policy_source)
        self.tcp_pose_envelope_runtime = safety_runtime
        self._lock = threading.RLock()
        self._started = False
        self._held = False
        self._closed = False
        self._active_operation_id: str | None = None

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def held(self) -> bool:
        with self._lock:
            return self._held

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def active_operation_id(self) -> str | None:
        with self._lock:
            return self._active_operation_id

    def activate_operation(
        self,
        *,
        operation_id: str,
        route_id: str | None = None,
        safety_selection: TCPPoseEnvelopeSelection | None = None,
    ) -> Any:
        """Reset the selected policy under HOLD, then open one fresh READY."""

        normalized_id = _required(operation_id, field_name="operation_id")
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy Skill application is closed.")
            if self._active_operation_id not in {None, normalized_id}:
                raise RuntimeError("Another policy Skill operation is active.")
            if not self._started:
                self.runtime.start()
                # The lower runtime is live from this point even if source
                # startup fails.  Publish that lifecycle fact before the next
                # fallible call so close_after_hold cannot leak its resources.
                self._started = True
                try:
                    self.policy_source.start()
                except Exception:
                    self.runtime.transition_execution_gate(
                        "hold",
                        reason="policy_skill_source_start_failed",
                        source_event_ids=(normalized_id,),
                        force_cancellation=True,
                    )
                    self._held = True
                    raise

            gate = self.runtime.capture_execution_gate_snapshot(
                now=float(self.runtime.clock())
            )
            boundary_refs: tuple[str, ...] = ()
            if str(gate.status) != "hold":
                boundary = self.runtime.transition_execution_gate(
                    "hold",
                    reason="policy_skill_activation_boundary",
                    source_event_ids=(normalized_id,),
                    force_cancellation=True,
                )
                gate = boundary.snapshot
                boundary_refs = tuple(getattr(boundary, "lower_event_ids", ()))
                if str(gate.status) != "hold" or not boundary_refs:
                    raise RuntimeError(
                        "Lower did not establish HOLD for policy activation."
                    )
            self._held = True
            self.policy_source.reset_for_activation()
            safety_activation = None
            if self.tcp_pose_envelope_runtime is not None:
                if route_id is None or safety_selection is None:
                    raise TCPPoseEnvelopeLifecycleError(
                        "Policy Skill requires an exact lower safety selection."
                    )
                receiver = self.runtime.core.capture_receiver_state()
                execution_epoch = int(receiver.current_execution_epoch)
                safety_activation = self.tcp_pose_envelope_runtime.activate(
                    selection=safety_selection,
                    operation_id=normalized_id,
                    route_id=_required(route_id, field_name="route_id"),
                    execution_epoch=execution_epoch,
                    source_event_ids=tuple(
                        dict.fromkeys(
                            (
                                normalized_id,
                                f"lower-execution-epoch:{execution_epoch}",
                                *boundary_refs,
                            )
                        )
                    ),
                    lower_state="hold",
                )
            try:
                transition = self.runtime.transition_execution_gate(
                    "ready",
                    reason="policy_skill_operation_activation",
                    source_event_ids=(
                        normalized_id,
                        *(
                            ()
                            if safety_activation is None
                            else (safety_activation.semantic_content_hash,)
                        ),
                    ),
                )
            except Exception:
                if safety_activation is not None:
                    receiver = self.runtime.core.capture_receiver_state()
                    self.tcp_pose_envelope_runtime.deactivate_after_hold(
                        operation_id=normalized_id,
                        next_execution_epoch=int(receiver.current_execution_epoch),
                        hold_evidence_ids=("policy-skill-ready-transition-failed",),
                    )
                raise
            self._active_operation_id = normalized_id
            self._held = False
            return transition

    def step(
        self,
        *,
        operation_id: str,
        now: float | None = None,
        physics_ticks_after_dispatch: int = 0,
    ) -> MujocoPolicySkillMotionCycle:
        """Poll once and advance the sole lower decision owner once."""

        normalized_id = _required(operation_id, field_name="operation_id")
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("Policy Skill application is not running.")
            if self._held:
                raise RuntimeError("Policy Skill application is held.")
            if self._active_operation_id != normalized_id:
                raise ValueError("Operation does not own the policy Skill session.")
            timestamp = float(self.runtime.clock() if now is None else now)
            if not math.isfinite(timestamp):
                raise ValueError("step.now must be finite.")
            emission = self.policy_source.step(now=timestamp)
            observed_at = max(timestamp, float(self.runtime.clock()))
            submission = (
                None
                if emission is None
                else self.runtime.submit_policy_action_emission(emission)
            )
            authoritative = self.runtime.step(
                now=observed_at,
                physics_ticks_after_dispatch=physics_ticks_after_dispatch,
            )
            gate = self.runtime.capture_execution_gate_snapshot(
                now=max(observed_at, float(self.runtime.clock()))
            )
            if str(gate.status) != "ready":
                self._held = True
            return MujocoPolicySkillMotionCycle(
                observed_at=observed_at,
                emission=emission,
                proposal_submission=submission,
                authoritative_step=authoritative,
            )

    def request_hold(
        self,
        *,
        reason: str,
        operation_id: str | None = None,
    ) -> Any | None:
        """Request lower HOLD; the returned lower transition is the evidence."""

        normalized_reason = _required(reason, field_name="reason")
        with self._lock:
            if not self._started:
                return None
            if operation_id is not None and self._active_operation_id not in {
                None,
                operation_id,
            }:
                raise ValueError("HOLD operation identity mismatch.")
            transition = self.runtime.transition_execution_gate(
                "hold",
                reason=normalized_reason,
                source_event_ids=(
                    ()
                    if operation_id is None
                    else (_required(operation_id, field_name="operation_id"),)
                ),
                force_cancellation=True,
            )
            if str(getattr(transition.snapshot, "status", "")) != "hold":
                raise RuntimeError("Lower did not establish requested HOLD.")
            self._held = True
            self.policy_source.reset_for_activation()
            if self.tcp_pose_envelope_runtime is not None:
                receiver = self.runtime.core.capture_receiver_state()
                hold_refs = tuple(getattr(transition, "lower_event_ids", ()))
                if not hold_refs:
                    raise TCPPoseEnvelopeLifecycleError(
                        "Lower HOLD did not provide envelope deactivation evidence."
                    )
                self.tcp_pose_envelope_runtime.deactivate_after_hold(
                    operation_id=operation_id,
                    next_execution_epoch=int(receiver.current_execution_epoch),
                    hold_evidence_ids=hold_refs,
                )
            return transition

    def release_operation(self, *, operation_id: str) -> None:
        normalized_id = _required(operation_id, field_name="operation_id")
        with self._lock:
            if self._active_operation_id not in {None, normalized_id}:
                raise ValueError("Operation release identity mismatch.")
            self._active_operation_id = None

    def close_after_hold(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._started and not self._held:
                raise RuntimeError(
                    "Closing a started policy Skill application requires HOLD."
                )
            try:
                self.policy_source.close()
            finally:
                if self._started:
                    self.runtime.close()
                self._closed = True
                self._active_operation_id = None


__all__ = [
    "MujocoPolicySkillMotionApplication",
    "MujocoPolicySkillMotionCycle",
]
