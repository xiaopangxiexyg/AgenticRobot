"""Experiment-only bridge from one selected HITL source to the MuJoCo target."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from robot_motion.deploy import DeploymentObservation, NormalizedAction, PolicyDeployRuntime
from robot_motion.deploy.emission_source import DeployActionPreparer

from .hitl import HITL_EVENT_ACTIVATION, HITL_EVENT_HOLD
from .stream import StateBCStreamEvent


@dataclass(frozen=True)
class StateBCMujocoActionReceipt:
    """Evidence for one selected action traversing the existing target owner."""

    source: str
    activation_id: str
    emission_id: str
    prediction: Any
    preparation: Any
    submission: Any
    dispatch: Any


@dataclass(frozen=True)
class StateBCMujocoTransitionReceipt:
    """Lower gate evidence plus the optional fenced authority re-arm event."""

    event_kind: str
    gate_transition: Any
    authority_event: Any | None = None


class _OneActionPolicy:
    def __init__(self, action: NormalizedAction) -> None:
        self._action = action

    def predict(self, _observation: Any) -> NormalizedAction:
        return self._action


class StateBCMujocoStreamBridge:
    """Sequence an experimental stream through existing queue and target owners.

    The bridge owns only upper sequencing and correlation. ``PolicyDeployRuntime``
    keeps the policy queue/emission contract, while the injected MuJoCo runtime
    remains authoritative for HOLD, safety, IK, reference generation and dispatch.
    The caller continues to own target start, reset and close.
    """

    def __init__(
        self,
        *,
        policy_runtime: PolicyDeployRuntime,
        target: Any,
        preparer: DeployActionPreparer,
        clock: Callable[[], float],
        physics_ticks_after_dispatch: int = 1,
    ) -> None:
        if not isinstance(policy_runtime, PolicyDeployRuntime):
            raise TypeError("policy_runtime must be a PolicyDeployRuntime.")
        if not isinstance(preparer, DeployActionPreparer):
            raise TypeError("preparer must be a DeployActionPreparer.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        ticks = int(physics_ticks_after_dispatch)
        if ticks < 0:
            raise ValueError("physics_ticks_after_dispatch must be non-negative.")
        runtime = getattr(target, "runtime", None)
        for name in (
            "capture_execution_gate_snapshot",
            "transition_execution_gate",
            "submit_policy_action_emission",
            "step",
        ):
            if not callable(getattr(runtime, name, None)):
                raise TypeError(f"target.runtime must expose {name}().")
        self.policy_runtime = policy_runtime
        self.target = target
        self.preparer = preparer
        self.clock = clock
        self.physics_ticks_after_dispatch = ticks
        self._hold_outcome: Any | None = None
        self._ready_activation_id: str | None = None

    @staticmethod
    def _event_id(event: StateBCStreamEvent) -> str:
        activation = event.activation_id or "unbound"
        return (
            f"state-bc-stream:{event.event_kind}:{event.step_index}:"
            f"{event.requested_source}:{activation}"
        )

    @staticmethod
    def _require_complete_hold(outcome: Any) -> None:
        receipt = getattr(outcome, "cancellation_receipt", None)
        validation = getattr(outcome, "cancellation_dispatch_validation", None)
        if (
            getattr(getattr(outcome, "snapshot", None), "status", None) != "hold"
            or getattr(outcome, "cancellation", None) is None
            or receipt is None
            or not bool(getattr(receipt, "complete", False))
            or bool(getattr(receipt, "requires_fault_escalation", False))
            or validation is None
            or not bool(getattr(validation, "accepted", False))
        ):
            raise RuntimeError("source switch requires a complete lower HOLD receipt.")

    def transition(
        self,
        event: StateBCStreamEvent,
        _observation: DeploymentObservation,
    ) -> Any:
        runtime = self.target.runtime
        event_id = self._event_id(event)
        if event.event_kind == HITL_EVENT_HOLD:
            self.policy_runtime.clear_action_queue()
            self.policy_runtime.reset_prediction_schedule()
            outcome = runtime.transition_execution_gate(
                "hold",
                reason=f"state_bc_source_switch:{event.requested_source}",
                source_event_ids=(event_id,),
                force_cancellation=True,
            )
            self._hold_outcome = outcome
            self._ready_activation_id = None
            self._require_complete_hold(outcome)
            return StateBCMujocoTransitionReceipt(
                event_kind=event.event_kind,
                gate_transition=outcome,
            )

        if event.event_kind != HITL_EVENT_ACTIVATION:
            raise ValueError(f"unsupported target transition {event.event_kind!r}.")
        if event.activation_id is None:
            raise ValueError("activation event requires activation_id.")
        post_hold = bool(event.metadata.get("post_hold_observation", False)) or (
            self._hold_outcome is not None
        )
        authority_event = None
        if post_hold:
            if self._hold_outcome is None:
                raise RuntimeError("post-HOLD activation has no lower HOLD evidence.")
            self._require_complete_hold(self._hold_outcome)
            port = getattr(runtime, "authoritative_port", None)
            if port is None:
                raise RuntimeError("MuJoCo target lacks its authoritative port.")
            receiver_before = port.capture_receiver_state()
            authority_event = port.admit_authority_event(
                event_type="mode",
                control_mode=runtime.config.initial_control_mode,
                reason=f"state_bc_source_rearm:{event.requested_source}",
            )
            if authority_event.authority_epoch <= receiver_before.current_authority_epoch:
                raise RuntimeError("source re-arm did not advance the authority epoch.")
            source_ids = tuple(
                dict.fromkeys(
                    (
                        *getattr(self._hold_outcome, "lower_event_ids", ()),
                        authority_event.event_id,
                        event_id,
                    )
                )
            )
        else:
            source_ids = (event_id,)
        outcome = runtime.transition_execution_gate(
            "ready",
            reason=f"state_bc_source_activation:{event.requested_source}",
            source_event_ids=source_ids,
        )
        if (
            getattr(getattr(outcome, "snapshot", None), "status", None) != "ready"
            or getattr(outcome, "cancellation", None) is not None
        ):
            raise RuntimeError("source activation failed to establish lower READY.")
        self._hold_outcome = None
        self._ready_activation_id = event.activation_id
        return StateBCMujocoTransitionReceipt(
            event_kind=event.event_kind,
            gate_transition=outcome,
            authority_event=authority_event,
        )

    def execute(
        self,
        action: NormalizedAction,
        event: StateBCStreamEvent,
        observation: DeploymentObservation,
    ) -> StateBCMujocoActionReceipt:
        if event.activation_id is None or event.activation_id != self._ready_activation_id:
            raise RuntimeError("action does not belong to the current READY activation.")
        gate = self.target.runtime.capture_execution_gate_snapshot(now=self.clock())
        if getattr(gate, "status", None) != "ready":
            raise RuntimeError("lower execution gate is not READY.")
        prediction = self.policy_runtime.predict_and_queue(
            _OneActionPolicy(action),
            observation,
            now=self.clock(),
            queue_policy="replace",
        )
        if not prediction.accepted or prediction.error is not None:
            raise RuntimeError("selected action was rejected by the policy queue.")
        preparation = self.policy_runtime.prepare_next_action(
            observation,
            preparer=self.preparer,
            policy_name=f"state_bc_{event.executed_source}",
        )
        emission = preparation.emission
        if emission is None or preparation.emission_error is not None:
            raise RuntimeError("selected action did not produce a policy emission.")
        submission = self.target.runtime.submit_policy_action_emission(emission)
        if not bool(getattr(submission, "accepted", False)):
            raise RuntimeError("MuJoCo target rejected the selected action emission.")
        decision_now = max(self.clock(), float(emission.effective_from))
        next_due = getattr(self.target.runtime, "next_decision_due_at", None)
        if next_due is not None:
            decision_now = max(decision_now, float(next_due))
        if decision_now >= float(emission.effective_until):
            raise RuntimeError("no authoritative decision tick fits the emission window.")
        dispatch = self.target.runtime.step(
            now=decision_now,
            physics_ticks_after_dispatch=self.physics_ticks_after_dispatch
        )
        if getattr(dispatch, "status", None) != "dispatched":
            computation = getattr(dispatch, "computation", None)
            directive = getattr(dispatch, "directive", None)
            group_directives = getattr(directive, "directive_by_selection_group", {})
            group_status = {
                str(group): getattr(value, "directive", None)
                for group, value in dict(group_directives).items()
            }
            snapshot = getattr(dispatch, "snapshot", None)
            proposal_snapshot = getattr(snapshot, "proposal_snapshot", None)
            dispositions = [
                getattr(value, "disposition", None)
                for value in getattr(proposal_snapshot, "dispositions", ())
            ]
            raise RuntimeError(
                f"MuJoCo target did not dispatch the selected action: "
                f"status={getattr(dispatch, 'status', None)!r}, "
                f"output={getattr(computation, 'output_status', None)!r}, "
                f"global={getattr(directive, 'global_status', None)!r}, "
                f"groups={group_status!r}, dispositions={dispositions!r}."
            )
        return StateBCMujocoActionReceipt(
            source=str(event.executed_source),
            activation_id=event.activation_id,
            emission_id=emission.emission_id,
            prediction=prediction,
            preparation=preparation,
            submission=submission,
            dispatch=dispatch,
        )


__all__ = [
    "StateBCMujocoActionReceipt",
    "StateBCMujocoStreamBridge",
    "StateBCMujocoTransitionReceipt",
]
