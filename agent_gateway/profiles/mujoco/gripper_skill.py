"""MuJoCo execution adapter for fixed right-gripper endpoint Skills."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.protocol.evaluation_contracts import (
    SkillAggregateEvaluationReceipt,
    SkillEvaluationReceipt,
)
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
from agent_gateway.skills.motion.definition import OPEN_GRIPPER_SKILL_ID_V1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.target_safety import TCPPoseEnvelopeSelection

from .gripper_deployment import MujocoGripperActionRegistration
from .motion_application import MujocoAgentSkillMotionApplication
from .motion_skill import mujoco_agent_skill_hold_evidence_from_transition


def _step_evidence_refs(step: Any) -> tuple[str, ...]:
    refs: set[str] = set()
    admission = step.proposal_admission
    for value in (
        getattr(admission, "proposal_id", None),
        getattr(admission, "semantic_content_hash", None),
    ):
        if isinstance(value, str) and value.strip():
            refs.add(value)
    authoritative = step.authoritative_step
    if authoritative is not None:
        fact_ref_by_type: dict[str, str] = {}
        untyped_fact_refs: set[str] = set()
        for fact in tuple(getattr(authoritative, "trace_facts", ())):
            value = getattr(fact, "fact_id", None)
            if isinstance(value, str) and value.strip():
                event_type = getattr(fact, "event_type", None)
                if isinstance(event_type, str) and event_type.strip():
                    fact_ref_by_type[event_type] = value
                else:
                    untyped_fact_refs.add(value)
        priority = (
            "target_execution_request",
            "target_execution_submission",
            "shared_safety_result",
            "target_execution_completion",
            "dispatch_owner",
            "dispatch_receipt",
            "component_execution_target",
            "component_execution_target_admission",
            "measurement_alignment",
        )
        ordered_fact_refs = [
            fact_ref_by_type.pop(event_type)
            for event_type in priority
            if event_type in fact_ref_by_type
        ]
        ordered_fact_refs.extend(
            fact_ref_by_type[event_type]
            for event_type in sorted(fact_ref_by_type)
        )
        ordered_fact_refs.extend(sorted(untyped_fact_refs))
        refs.update(ordered_fact_refs[:48])
        request = getattr(authoritative, "target_execution_request", None)
        request_id = getattr(request, "request_id", None)
        if isinstance(request_id, str) and request_id.strip():
            refs.add(request_id)
    return tuple(sorted(refs))


def _dispatch_receipt_ref(step: Any) -> str | None:
    authoritative = step.authoritative_step
    if authoritative is None:
        return None
    matches = tuple(
        str(fact.fact_id)
        for fact in tuple(getattr(authoritative, "trace_facts", ()))
        if getattr(fact, "event_type", None) == "dispatch_receipt"
        and isinstance(getattr(fact, "fact_id", None), str)
        and fact.fact_id.strip()
    )
    if len(matches) > 1:
        raise RuntimeError("One MuJoCo step emitted multiple dispatch receipts.")
    return matches[0] if matches else None


class MujocoGripperActionSkillLifecycle:
    """Drive one fixed endpoint through the shared authoritative sim runtime."""

    def __init__(
        self,
        *,
        application: MujocoAgentSkillMotionApplication,
        registration: MujocoGripperActionRegistration,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        current_station_revision: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(application, MujocoAgentSkillMotionApplication):
            raise TypeError("application must be a motion application.")
        if not isinstance(registration, MujocoGripperActionRegistration):
            raise TypeError("registration must be typed.")
        if not isinstance(request, AgentSkillRequest) or not isinstance(
            operation, AgentSkillOperation
        ):
            raise TypeError("request and operation must be typed.")
        if not callable(current_station_revision) or not callable(clock):
            raise TypeError("revision and clock providers must be callable.")
        route = registration.skill.routes[0]
        binding = registration.binding
        checks = (
            (request.semantic_content_hash == operation.request_hash, "request"),
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
        for matches, label in checks:
            if not matches:
                raise ValueError(f"MuJoCo gripper action {label} mismatch.")
        registration.payload_validator.validate_parameters(request.parameters)
        component_id = str(request.parameters["component_id"])
        if component_id != registration.profile.component_id:
            raise ValueError("Requested gripper is absent from this deployment.")

        self.executor_id = binding.executor_id
        self.application = application
        self.registration = registration
        self.request = request
        self.operation = operation
        self.current_station_revision = current_station_revision
        self.clock = clock
        self.component_id = component_id
        self.target_close_fraction = (
            0.0 if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1 else 1.0
        )
        self._initial: dict[str, object] | None = None
        self._action = None
        self._activation_refs: tuple[str, ...] = ()
        self._execution_refs: set[str] = set()
        self._first_step_refs: tuple[str, ...] = ()
        self._latest_step_refs: tuple[str, ...] = ()

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

    def _capture(self) -> dict[str, object]:
        value = self.application.capture_gripper_state_by_component()[
            self.component_id
        ]
        return dict(value)

    def _remember_step_evidence(self, step: Any) -> None:
        refs = _step_evidence_refs(step)
        if not self._first_step_refs:
            self._first_step_refs = refs
        self._latest_step_refs = refs

    def _execution_evidence_refs(self) -> tuple[str, ...]:
        stable = tuple(sorted(self._execution_refs))
        if len(stable) > 128:
            raise RuntimeError("Stable gripper evidence exceeds protocol bounds.")
        dynamic = tuple(
            sorted(
                {
                    *self._first_step_refs,
                    *self._latest_step_refs,
                }
                - set(stable)
            )
        )
        return tuple(sorted((*stable, *dynamic[: 128 - len(stable)])))

    def _bound_arm_hold_selection(self) -> TCPPoseEnvelopeSelection:
        refs = tuple(self.registration.binding.safety_envelope_refs)
        if len(refs) != 1:
            raise RuntimeError("Gripper action requires one arm-hold envelope.")
        ref = refs[0]
        runtime = self.application.tcp_pose_envelope_runtime
        if runtime is None:
            raise RuntimeError("Gripper action lost its arm-hold safety runtime.")
        matches = tuple(
            entry
            for entry in runtime.catalog_entries
            if entry.envelope_id == ref.object_id
            and entry.semantic_content_hash == ref.content_hash
            and self.registration.skill.routes[0].route_id
            in entry.allowed_route_ids
        )
        if len(matches) != 1:
            raise RuntimeError("Bound gripper arm-hold envelope is unavailable.")
        entry = matches[0]
        profile_hash = canonical_content_hash(
            {
                "source": "lower_route_binding_single_envelope_default",
                "deployment_binding_hash": (
                    self.registration.binding.semantic_content_hash
                ),
                "deployment_envelope_hash": entry.semantic_content_hash,
                "constraint_set_hash": entry.constraint_set_hash,
            },
            domain="dexteleop.lower_route_bound_safety_profile.v1",
        )
        return TCPPoseEnvelopeSelection(
            safety_profile_id="lower-route-bound.gripper-arm-hold@1",
            safety_profile_hash=profile_hash,
            deployment_envelope_id=entry.envelope_id,
            deployment_envelope_hash=entry.semantic_content_hash,
            constraint_set_hash=entry.constraint_set_hash,
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
                operation=operation,
                reason="gripper_action_cancelled_before_activation",
            )
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="gripper_action_cancelled_before_activation",
                lower_operation_id=hold.lower_operation_id,
                evidence_refs=hold.evidence_refs,
                lower_hold_confirmed=hold.lower_hold_confirmed,
            )
        self._initial = self._capture()
        qpos = self.application.gripper_qpos_for_close(
            self.target_close_fraction
        )
        self._action = self.application.complete_component_action(
            {self.component_id: qpos}
        )
        selection = (
            TCPPoseEnvelopeSelection(
                safety_profile_id=request.safety_profile_id,
                safety_profile_hash=request.safety_profile_hash,
                deployment_envelope_id=(
                    request.deployment_safety_envelope_id
                ),
                deployment_envelope_hash=(
                    request.deployment_safety_envelope_hash
                ),
                constraint_set_hash=request.safety_constraint_set_hash,
            )
            if request.has_safety_profile_selection
            else self._bound_arm_hold_selection()
        )
        transition = self.application.activate_operation(
            operation_id=operation.operation_id,
            route_id=self.registration.skill.routes[0].route_id,
            safety_selection=selection,
        )
        refs = tuple(getattr(transition, "lower_event_ids", ()))
        if not refs or getattr(transition.snapshot, "status", None) != "ready":
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="mujoco_gripper_lower_ready_not_confirmed",
                lower_operation_id=operation.operation_id,
                evidence_refs=refs,
                lower_hold_confirmed=False,
            )
        self._activation_refs = tuple(str(value) for value in refs)
        self._execution_refs.update(self._activation_refs)
        return StationLifecycleActivationEvidence(
            succeeded=True,
            status="mujoco_gripper_endpoint_ready",
            lower_operation_id=operation.operation_id,
            evidence_refs=self._activation_refs,
            lower_hold_confirmed=False,
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        transition = self.application.request_hold(
            reason=reason,
            operation_id=None if operation is None else operation.operation_id,
        )
        evidence = mujoco_agent_skill_hold_evidence_from_transition(
            transition,
            lower_operation_id=(
                None if operation is None else operation.operation_id
            ),
        )
        self._execution_refs.update(evidence.evidence_refs)
        return evidence

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
        hold: StationLifecycleControlEvidence | None = None,
    ) -> StationLifecycleExecutionCompletion:
        terminal_hold = hold or self.request_hold(
            operation=self.operation,
            reason=reason_code,
        )
        refs = self._execution_evidence_refs()
        if not refs:
            raise RuntimeError("Gripper completion has no lower execution evidence.")
        return StationLifecycleExecutionCompletion(
            succeeded=False,
            reason_code=reason_code,
            execution_receipt_refs=refs,
            terminal_hold_confirmed=terminal_hold.lower_hold_confirmed,
        )

    def run(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleExecutionCompletion:
        self._validate_call(request=request, operation=operation)
        if self._initial is None or self._action is None:
            return self._failure(
                reason_code="gripper_action_not_activated",
            )
        profile = self.registration.profile
        stable = 0
        previous_close = float(self._initial["close_fraction"])
        initial_at_target = (
            previous_close >= 1.0 - profile.open_endpoint_tolerance
            if request.skill_id != OPEN_GRIPPER_SKILL_ID_V1
            else previous_close <= profile.open_endpoint_tolerance
        )
        final = self._initial
        command_receipt_ref: str | None = None
        for cycle in range(profile.max_control_cycles):
            if cancel_requested():
                return self._failure(
                    reason_code="gripper_action_cancelled",
                )
            if float(self.clock()) >= operation.deadline_at:
                return self._failure(
                    reason_code="gripper_action_deadline_exceeded",
                )
            step = self.application.submit_and_step(
                operation_id=operation.operation_id,
                action=self._action,
                source_event_id=f"{operation.operation_id}:gripper:{cycle}",
                context_ids=(request.semantic_content_hash,),
                physics_ticks_after_dispatch=(
                    profile.physics_ticks_per_control_cycle
                ),
            )
            self._remember_step_evidence(step)
            if not step.proposal_admission.accepted:
                return self._failure(
                    reason_code=(
                        "gripper_action_proposal_rejected:"
                        + str(
                            step.proposal_admission.rejection_code or "unknown"
                        )
                    ),
                )
            authoritative = step.authoritative_step
            if authoritative is None:
                return self._failure(
                    reason_code="gripper_action_authoritative_step_missing",
                )
            if authoritative.status != "dispatched":
                return self._failure(
                    reason_code="gripper_action_authoritative_step_not_dispatched",
                )
            dispatch_receipt_ref = _dispatch_receipt_ref(step)
            if dispatch_receipt_ref is None:
                return self._failure(
                    reason_code="gripper_action_dispatch_receipt_missing",
                )
            if command_receipt_ref is None:
                command_receipt_ref = dispatch_receipt_ref
                self._execution_refs.add(command_receipt_ref)
            final = self._capture()
            close = float(final["close_fraction"])
            initial_close = float(self._initial["close_fraction"])
            effect_delta = (
                initial_close - close
                if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1
                else close - initial_close
            )
            if initial_at_target:
                command_effect_observed = (
                    close <= profile.open_endpoint_tolerance
                    if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1
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
            predicate_met = (
                close <= profile.open_endpoint_tolerance
                if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1
                else settled
            )
            stable = (
                stable + 1
                if command_effect_observed and settled and predicate_met
                else 0
            )
            previous_close = close
            if stable >= profile.stable_measurement_count:
                break
        else:
            return self._failure(
                reason_code="gripper_action_control_cycles_exhausted",
            )
        hold = self.request_hold(
            operation=operation,
            reason="gripper_action_terminal_idle",
        )
        if not hold.lower_hold_confirmed:
            return self._failure(
                reason_code="gripper_action_terminal_hold_unconfirmed",
                hold=hold,
            )
        pre_hold_final_close = float(final["close_fraction"])
        final = self._capture()
        final_close = float(final["close_fraction"])
        if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1:
            if final_close > profile.open_endpoint_tolerance:
                return self._failure(
                    reason_code="gripper_open_post_hold_outside_endpoint",
                    hold=hold,
                )
        elif (
            abs(final_close - pre_hold_final_close)
            > profile.settled_close_delta_tolerance
        ):
            return self._failure(
                reason_code="gripper_close_post_hold_not_settled",
                hold=hold,
            )
        if command_receipt_ref is None:
            return self._failure(
                reason_code="gripper_action_dispatch_receipt_missing",
                hold=hold,
            )
        completed_at = float(self.clock())
        final_measured_at = float(final["measured_at"])
        measurement_hash = canonical_content_hash(
            {
                "runtime_session_id": request.runtime_session_id,
                "control_domain_id": self.registration.binding.control_domain_id,
                "operation_id": operation.operation_id,
                "component_id": self.component_id,
                "close_fraction": final_close,
                "measured_at": final_measured_at,
                "state_source": final["state_source"],
            },
            domain="dexteleop.agent_state.mujoco_gripper_measurement.v1",
        )
        result = {
            "component_id": self.component_id,
            "commanded_close_fraction": self.target_close_fraction,
            "initial_close_fraction": float(self._initial["close_fraction"]),
            "final_close_fraction": final_close,
            "final_measured_at": final_measured_at,
            "completed_at": completed_at,
            "station_revision": int(self.current_station_revision()),
            "command_receipt_ref": command_receipt_ref,
            "final_measurement_hash": measurement_hash,
        }
        contextual = (
            self.registration.payload_validator.validate_execution_result_for_parameters
        )
        self.registration.payload_validator.validate_execution_result(result)
        if contextual is None:
            raise RuntimeError("Gripper registration lacks contextual validation.")
        contextual(request.parameters, result)
        predicate = (
            "gripper_open_endpoint_reached.v1"
            if request.skill_id == OPEN_GRIPPER_SKILL_ID_V1
            else "gripper_close_command_settled.v1"
        )
        plan = self.registration.evaluation_plan
        if len(plan.evaluator_specs) != 1:
            raise RuntimeError("Gripper evaluation requires one exact evaluator.")
        evaluator_identity = plan.evaluator_specs[0]
        evaluator_spec = plan.evaluator(
            evaluator_identity.evaluator_id,
            evaluator_identity.evaluator_version,
        )
        refs = self._execution_evidence_refs()
        evaluator_receipt = SkillEvaluationReceipt(
            operation_id=operation.operation_id,
            request_hash=request.semantic_content_hash,
            skill_id=request.skill_id,
            skill_version=request.skill_version,
            skill_definition_hash=request.skill_definition_hash,
            route_id=request.route_id,
            route_version=request.route_version,
            route_hash=self.registration.skill.routes[0].semantic_content_hash,
            deployment_binding_hash=(
                self.registration.binding.semantic_content_hash
            ),
            evaluation_plan_hash=plan.semantic_content_hash,
            evaluator_id=evaluator_spec.evaluator_id,
            evaluator_version=evaluator_spec.evaluator_version,
            evaluator_hash=evaluator_spec.semantic_content_hash,
            evaluator_kind=evaluator_spec.evaluator_kind,
            evidence_domain_id=evaluator_spec.evidence_domain_id,
            input_evidence_hash=measurement_hash,
            evaluated_cut_id=f"gripper-terminal:{measurement_hash}",
            evaluated_at=completed_at,
            verdict="succeeded",
            reason_codes=(predicate,),
            evidence_refs=refs,
        )
        aggregate = SkillAggregateEvaluationReceipt(
            operation_id=operation.operation_id,
            request_hash=request.semantic_content_hash,
            evaluation_plan_hash=plan.semantic_content_hash,
            aggregation_policy_id=plan.aggregation_policy_id,
            aggregation_policy_hash=plan.aggregation_policy_hash,
            consumed_evaluation_receipt_hashes=(
                evaluator_receipt.semantic_content_hash,
            ),
            ignored_evaluation_receipt_hashes=(),
            missing_evaluator_ids=(),
            verdict="succeeded",
            reason_codes=(predicate,),
            generated_at=completed_at,
        )
        evaluator_ref = (
            "agent-skill-evaluation:" + evaluator_receipt.semantic_content_hash
        )
        aggregate_ref = (
            "agent-skill-aggregate-evaluation:" + aggregate.semantic_content_hash
        )
        evaluation_payload = {
            "aggregate_evaluation_receipt_ref": aggregate_ref,
            "task_outcome": "succeeded",
            "predicate": predicate,
        }
        self.registration.payload_validator.validate_task_evaluation(
            evaluation_payload
        )
        return StationLifecycleExecutionCompletion(
            succeeded=True,
            reason_code="gripper_action_completed_after_idle_confirmation",
            execution_receipt_refs=refs,
            terminal_hold_confirmed=True,
            execution_result=result,
            evaluation=StationLifecycleEvaluationCompletion(
                evaluation_status="completed",
                task_outcome="succeeded",
                reason_code="gripper_action_predicate_satisfied",
                evaluator_receipt_refs=(evaluator_ref,),
                aggregate_evaluation_receipt_ref=aggregate_ref,
                task_evaluation=evaluation_payload,
            ),
        )

    def close(self) -> None:
        self.application.release_operation(
            operation_id=self.operation.operation_id
        )


def build_mujoco_gripper_action_executor(
    *,
    application: MujocoAgentSkillMotionApplication,
    registration: MujocoGripperActionRegistration,
    current_station_revision: Callable[[], int],
    clock: Callable[[], float] = time.monotonic,
) -> ReusableStationAgentSkillOperationExecutor:
    """Bind one endpoint Route to the shared lower sim control domain."""

    route = registration.skill.routes[0]
    binding = registration.binding

    def lifecycle_factory(*, request, operation):
        return MujocoGripperActionSkillLifecycle(
            application=application,
            registration=registration,
            request=request,
            operation=operation,
            current_station_revision=current_station_revision,
            clock=clock,
        )

    def domain_hold(*, operation, reason):
        transition = application.request_hold(
            reason=reason,
            operation_id=None if operation is None else operation.operation_id,
        )
        return mujoco_agent_skill_hold_evidence_from_transition(
            transition,
            lower_operation_id=(
                None if operation is None else operation.operation_id
            ),
        )

    return ReusableStationAgentSkillOperationExecutor(
        port_id="agent-gateway:mujoco-gripper-action:" + binding.runtime_session_id,
        binding=StationOperationWorkerBinding(
            worker_id="mujoco-gripper-action-worker:" + route.skill_id,
            executor_id=binding.executor_id,
            runtime_session_id=binding.runtime_session_id,
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            route_id=route.route_id,
            route_version=route.route_version,
            deployment_binding_hash=binding.semantic_content_hash,
        ),
        lifecycle_factory=lifecycle_factory,
        domain_hold=domain_hold,
        current_station_revision=current_station_revision,
        clock=clock,
    )


__all__ = [
    "MujocoGripperActionSkillLifecycle",
    "build_mujoco_gripper_action_executor",
]
