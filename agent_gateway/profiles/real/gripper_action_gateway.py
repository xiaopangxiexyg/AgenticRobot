"""Execution adapter for fixed real gripper endpoint Skills."""

from __future__ import annotations

import time
from collections.abc import Callable

from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleEvaluationCompletion,
    StationLifecycleExecutionCompletion,
    StationOperationWorkerBinding,
)
from agent_gateway.skills.motion.definition import (
    OPEN_GRIPPER_SKILL_ID_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.skill_deploy.gripper_port import (
    DMTacGripperActionPort,
    RealGripperActionDispatchReceipt,
    RealGripperActionFeedback,
)

from .gripper_action_deployment import RealGripperActionRegistration


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("station revision must be a non-negative integer.")
    return value


class RealGripperActionSkillLifecycle:
    """Run one fixed endpoint command through the exclusive lower port."""

    def __init__(
        self,
        *,
        registration: RealGripperActionRegistration,
        port: DMTacGripperActionPort,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        current_station_revision: Callable[[], int],
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(registration, RealGripperActionRegistration):
            raise TypeError("registration must be typed.")
        if not isinstance(port, DMTacGripperActionPort):
            raise TypeError("port must be a DMTacGripperActionPort.")
        if not isinstance(request, AgentSkillRequest) or not isinstance(
            operation,
            AgentSkillOperation,
        ):
            raise TypeError("request and operation must be typed.")
        if (
            not callable(current_station_revision)
            or not callable(clock)
            or not callable(sleep)
        ):
            raise TypeError("revision, clock, and sleep providers must be callable.")
        route = registration.skill.routes[0]
        binding = registration.binding
        if request.semantic_content_hash != operation.request_hash:
            raise ValueError("Gripper action request/operation hash mismatch.")
        checks = (
            (request.skill_id == route.skill_id, "Skill"),
            (request.route_hash == route.semantic_content_hash, "Route"),
            (
                request.deployment_binding_hash == binding.semantic_content_hash,
                "deployment binding",
            ),
            (
                request.parameter_profile_hash
                == registration.profile.semantic_content_hash,
                "parameter profile",
            ),
            (request.runtime_session_id == binding.runtime_session_id, "session"),
        )
        for matched, label in checks:
            if not matched:
                raise ValueError(f"Gripper action {label} mismatch.")
        registration.payload_validator.validate_parameters(request.parameters)
        component_id = request.parameters["component_id"]
        if component_id not in port.component_ids:
            raise ValueError("Requested gripper is absent from the lower port.")
        if port.clock_domain_id != registration.profile.clock_domain_id:
            raise ValueError(
                "Lower port clock domain differs from the registered profile."
            )

        self.executor_id = binding.executor_id
        self.registration = registration
        self.port = port
        self.request = request
        self.operation = operation
        self.component_id = component_id
        self.target_close_fraction = (
            0.0 if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1 else 1.0
        )
        self.current_station_revision = current_station_revision
        self.clock = clock
        self.sleep = sleep
        self._initial_feedback: RealGripperActionFeedback | None = None
        self._dispatch_receipt: RealGripperActionDispatchReceipt | None = None
        self._closed = False

    def _validate_call(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> None:
        if request.semantic_content_hash != self.request.semantic_content_hash:
            raise ValueError("Gripper action request changed.")
        if operation.operation_id != self.operation.operation_id:
            raise ValueError("Gripper action operation changed.")

    def _validate_fresh_feedback(
        self,
        feedback: RealGripperActionFeedback,
    ) -> RealGripperActionFeedback:
        now = float(self.clock())
        profile = self.registration.profile
        if feedback.available_at > now + 1e-9 or feedback.measured_at > now + 1e-9:
            raise RuntimeError("Lower gripper feedback is from the future.")
        age_ms = (now - feedback.measured_at) * 1_000.0
        if age_ms > profile.maximum_measurement_age_ms:
            raise RuntimeError("Lower gripper feedback is stale.")
        if feedback.error_code != 0:
            raise RuntimeError("Lower gripper reports an active hardware fault.")
        return feedback

    def _fresh_feedback(self) -> RealGripperActionFeedback:
        return self._validate_fresh_feedback(
            self.port.feedback(component_id=self.component_id)
        )

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleActivationEvidence:
        self._validate_call(request=request, operation=operation)
        if cancel_requested():
            hold = self.request_hold(
                operation=operation, reason="cancelled_before_dispatch"
            )
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="gripper_action_cancelled_before_dispatch",
                lower_operation_id=hold.lower_operation_id,
                evidence_refs=hold.evidence_refs,
                lower_hold_confirmed=hold.lower_hold_confirmed,
            )
        self._initial_feedback = self._fresh_feedback()
        self._dispatch_receipt = self.port.dispatch(
            operation_id=operation.operation_id,
            component_id=self.component_id,
            target_close_fraction=self.target_close_fraction,
        )
        return StationLifecycleActivationEvidence(
            succeeded=True,
            status="real_gripper_endpoint_command_dispatched",
            lower_operation_id=self._dispatch_receipt.lower_operation_id,
            evidence_refs=self._dispatch_receipt.evidence_refs,
            lower_hold_confirmed=False,
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        receipt = self.port.hold(
            operation_id=None if operation is None else operation.operation_id,
            reason=_required(reason, name="reason"),
        )
        return StationLifecycleControlEvidence(
            succeeded=receipt.succeeded,
            status=receipt.status,
            lower_operation_id=receipt.lower_operation_id,
            evidence_refs=receipt.evidence_refs,
            lower_hold_confirmed=receipt.succeeded,
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        self._validate_call(request=request, operation=operation)
        return self.request_hold(operation=operation, reason=reason)

    def _failure(
        self,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> StationLifecycleExecutionCompletion:
        hold = self.request_hold(operation=self.operation, reason=reason_code)
        refs = tuple(sorted({*evidence_refs, *hold.evidence_refs}))
        return StationLifecycleExecutionCompletion(
            succeeded=False,
            reason_code=reason_code,
            execution_receipt_refs=refs,
            terminal_hold_confirmed=hold.lower_hold_confirmed,
        )

    def run(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleExecutionCompletion:
        self._validate_call(request=request, operation=operation)
        initial = self._initial_feedback
        dispatch = self._dispatch_receipt
        if initial is None or dispatch is None:
            return self._failure(
                reason_code="gripper_action_not_dispatched",
                evidence_refs=("gripper-action:not-dispatched",),
            )
        evidence = set(dispatch.evidence_refs)
        stable = 0
        previous_close = initial.close_fraction
        profile = self.registration.profile
        is_open = request.skill_id == OPEN_GRIPPER_SKILL_ID_V1
        initial_at_target = (
            previous_close <= profile.open_endpoint_tolerance
            if is_open
            else previous_close >= 1.0 - profile.open_endpoint_tolerance
        )
        final: RealGripperActionFeedback | None = None
        while float(self.clock()) < operation.deadline_at:
            if cancel_requested():
                return self._failure(
                    reason_code="gripper_action_cancelled",
                    evidence_refs=tuple(evidence),
                )
            try:
                feedback = self._fresh_feedback()
            except RuntimeError:
                return self._failure(
                    reason_code="gripper_action_feedback_invalid",
                    evidence_refs=tuple(evidence),
                )
            evidence.update(feedback.evidence_refs)
            close = feedback.close_fraction
            effect_delta = (
                initial.close_fraction - close
                if is_open
                else close - initial.close_fraction
            )
            if initial_at_target:
                command_effect_observed = (
                    close <= profile.open_endpoint_tolerance
                    if is_open
                    else close >= 1.0 - profile.open_endpoint_tolerance
                )
            else:
                command_effect_observed = (
                    effect_delta > profile.settled_close_delta_tolerance
                )
            settled = (
                abs(close - previous_close)
                <= profile.settled_close_delta_tolerance
            )
            if (
                feedback.measured_at + 1e-9 < dispatch.dispatched_at
                or feedback.is_moving
            ):
                stable = 0
            else:
                predicate_met = (
                    close <= profile.open_endpoint_tolerance if is_open else settled
                )
                stable = (
                    stable + 1
                    if command_effect_observed and settled and predicate_met
                    else 0
                )
            previous_close = close
            final = feedback
            if stable >= profile.stable_measurement_count:
                break
            self.sleep(profile.poll_interval_ms / 1_000.0)
        else:
            return self._failure(
                reason_code="gripper_action_deadline_exceeded",
                evidence_refs=tuple(evidence),
            )
        assert final is not None
        settled = final
        post_hold_feedback: RealGripperActionFeedback | None = None
        post_hold_failure: str | None = None

        def terminal_feedback_accepted(
            feedback: RealGripperActionFeedback,
        ) -> bool:
            nonlocal post_hold_feedback, post_hold_failure
            post_hold_feedback = feedback
            try:
                self._validate_fresh_feedback(feedback)
            except RuntimeError:
                post_hold_failure = "gripper_action_post_hold_feedback_invalid"
                return False
            if feedback.is_moving:
                post_hold_failure = "gripper_action_post_hold_motion_detected"
                return False
            if feedback.measured_at + 1e-9 < max(
                dispatch.dispatched_at,
                settled.measured_at,
            ):
                post_hold_failure = "gripper_action_post_hold_measurement_regressed"
                return False
            if (
                is_open
                and feedback.close_fraction > profile.open_endpoint_tolerance
            ):
                post_hold_failure = "gripper_action_post_hold_open_endpoint_lost"
                return False
            if (
                not is_open
                and abs(feedback.close_fraction - settled.close_fraction)
                > profile.settled_close_delta_tolerance
            ):
                post_hold_failure = "gripper_action_post_hold_close_not_settled"
                return False
            return True

        hold = self.port.hold(
            operation_id=operation.operation_id,
            reason="gripper_action_terminal_idle",
            feedback_validator=terminal_feedback_accepted,
        )
        evidence.update(hold.evidence_refs)
        if not hold.succeeded:
            return StationLifecycleExecutionCompletion(
                succeeded=False,
                reason_code=(
                    post_hold_failure
                    or "gripper_action_terminal_hold_unconfirmed"
                ),
                execution_receipt_refs=tuple(sorted(evidence)),
                terminal_hold_confirmed=False,
            )
        assert post_hold_feedback is not None
        final = post_hold_feedback
        completed_at = float(self.clock())
        result = {
            "component_id": self.component_id,
            "commanded_close_fraction": self.target_close_fraction,
            "initial_close_fraction": initial.close_fraction,
            "final_close_fraction": final.close_fraction,
            "final_measured_at": final.measured_at,
            "completed_at": completed_at,
            "station_revision": _revision(self.current_station_revision()),
            "command_receipt_ref": dispatch.command_receipt_ref,
            "final_measurement_hash": final.measurement_hash,
        }
        self.registration.payload_validator.validate_execution_result(result)
        contextual = (
            self.registration.payload_validator.validate_execution_result_for_parameters
        )
        if contextual is None:
            raise RuntimeError(
                "Gripper action registration lacks contextual validation."
            )
        contextual(request.parameters, result)
        predicate = (
            "gripper_open_endpoint_reached.v1"
            if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1
            else "gripper_close_command_settled.v1"
        )
        evaluation_hash = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "predicate": predicate,
                "result": result,
            },
            domain="dexteleop.real_gripper_action_evaluation.v1",
        )
        aggregate_ref = "gripper-action-evaluation:" + evaluation_hash
        evaluation_payload = {
            "aggregate_evaluation_receipt_ref": aggregate_ref,
            "task_outcome": "succeeded",
            "predicate": predicate,
        }
        self.registration.payload_validator.validate_task_evaluation(evaluation_payload)
        return StationLifecycleExecutionCompletion(
            succeeded=True,
            reason_code="gripper_action_completed_after_idle_confirmation",
            execution_receipt_refs=tuple(sorted(evidence)),
            terminal_hold_confirmed=True,
            execution_result=result,
            evaluation=StationLifecycleEvaluationCompletion(
                evaluation_status="completed",
                task_outcome="succeeded",
                reason_code="gripper_action_predicate_satisfied",
                evaluator_receipt_refs=(aggregate_ref,),
                aggregate_evaluation_receipt_ref=aggregate_ref,
                task_evaluation=evaluation_payload,
            ),
        )

    def close(self) -> None:
        self._closed = True


def build_real_gripper_action_executor(
    *,
    registration: RealGripperActionRegistration,
    port: DMTacGripperActionPort,
    current_station_revision: Callable[[], int],
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ReusableStationAgentSkillOperationExecutor:
    """Bind one action registration to one shared exclusive lower port."""

    if not isinstance(registration, RealGripperActionRegistration):
        raise TypeError("registration must be typed.")
    if not isinstance(port, DMTacGripperActionPort):
        raise TypeError("port must be a DMTacGripperActionPort.")
    route = registration.skill.routes[0]
    binding = StationOperationWorkerBinding(
        worker_id="real-gripper-action-worker:" + route.skill_id,
        executor_id=registration.binding.executor_id,
        runtime_session_id=registration.binding.runtime_session_id,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        deployment_binding_hash=registration.binding.semantic_content_hash,
    )

    def lifecycle_factory(
        *, request: AgentSkillRequest, operation: AgentSkillOperation
    ) -> RealGripperActionSkillLifecycle:
        return RealGripperActionSkillLifecycle(
            registration=registration,
            port=port,
            request=request,
            operation=operation,
            current_station_revision=current_station_revision,
            clock=clock,
            sleep=sleep,
        )

    def domain_hold(*, operation: AgentSkillOperation | None, reason: str):
        receipt = port.hold(
            operation_id=None if operation is None else operation.operation_id,
            reason=reason,
        )
        return StationLifecycleControlEvidence(
            succeeded=receipt.succeeded,
            status=receipt.status,
            lower_operation_id=receipt.lower_operation_id,
            evidence_refs=receipt.evidence_refs,
            lower_hold_confirmed=receipt.succeeded,
        )

    return ReusableStationAgentSkillOperationExecutor(
        port_id="agent-gateway:real-gripper-action:"
        + registration.binding.runtime_session_id,
        binding=binding,
        lifecycle_factory=lifecycle_factory,
        domain_hold=domain_hold,
        current_station_revision=current_station_revision,
        clock=clock,
    )


__all__ = [
    "RealGripperActionSkillLifecycle",
    "build_real_gripper_action_executor",
]
