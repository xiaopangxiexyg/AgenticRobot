"""Per-operation lifecycle for the bounded no-model TCP-pose Skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from control_station.agent_gateway import (
    AgentSkillOperation,
    AgentSkillRequest,
    SkillAggregateEvaluationReceipt,
    SkillEvaluationPlan,
    SkillEvaluationReceipt,
    SkillExecutionRoute,
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleEvaluationCompletion,
    StationLifecycleExecutionCompletion,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentAction

from .agent_motion_application import MujocoAgentSkillMotionApplication
from .agent_motion_deployment import (
    MujocoAgentTcpPoseDeploymentBinding,
    MujocoAgentTcpPoseMotionProfile,
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _pose(value: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 7:
        raise ValueError(f"{field_name} must contain xyz+quat_xyzw.")
    pose = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in pose):
        raise ValueError(f"{field_name} must contain finite values.")
    norm = math.sqrt(sum(item * item for item in pose[3:]))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"{field_name} quaternion must be normalized.")
    return pose


def _pose_error(
    measured: tuple[float, ...],
    target: tuple[float, ...],
) -> tuple[float, float]:
    translation = math.sqrt(
        sum((first - second) ** 2 for first, second in zip(measured[:3], target[:3]))
    )
    dot = abs(sum(first * second for first, second in zip(measured[3:], target[3:])))
    rotation = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
    return translation, rotation


def mujoco_agent_skill_hold_evidence_from_transition(
    transition: Any | None,
    *,
    lower_operation_id: str | None,
) -> StationLifecycleControlEvidence:
    if transition is None:
        return StationLifecycleControlEvidence(
            succeeded=False,
            status="lower_runtime_not_started_no_hold_transition",
            lower_operation_id=lower_operation_id,
            evidence_refs=(),
            lower_hold_confirmed=False,
        )
    snapshot = getattr(transition, "snapshot", None)
    cancellation = getattr(transition, "cancellation", None)
    receipt = getattr(transition, "cancellation_receipt", None)
    complete = bool(
        getattr(snapshot, "status", None) == "hold"
        and cancellation is not None
        and receipt is not None
        and bool(getattr(receipt, "complete", False))
        and not bool(getattr(receipt, "timed_out", False))
        and not bool(getattr(receipt, "requires_fault_escalation", False))
        and not tuple(getattr(receipt, "pending_levels", ()))
        and not tuple(getattr(receipt, "rejected_levels", ()))
    )
    refs = {
        str(value)
        for value in tuple(getattr(transition, "lower_event_ids", ()))
        if str(value).strip()
    }
    for value in (
        getattr(getattr(transition, "event", None), "event_id", None),
        getattr(cancellation, "cancellation_id", None),
        getattr(snapshot, "semantic_content_hash", None),
    ):
        if isinstance(value, str) and value.strip():
            refs.add(value)
    return StationLifecycleControlEvidence(
        succeeded=complete,
        status=(
            "mujoco_agent_skill_lower_hold_confirmed"
            if complete
            else "mujoco_agent_skill_lower_hold_unconfirmed"
        ),
        lower_operation_id=lower_operation_id,
        evidence_refs=tuple(sorted(refs)),
        lower_hold_confirmed=complete,
    )


class MujocoAgentTcpPoseSkillLifecycle:
    """One operation; a reusable executor creates a fresh instance each time."""

    def __init__(
        self,
        *,
        application: MujocoAgentSkillMotionApplication,
        route: SkillExecutionRoute,
        deployment_binding: MujocoAgentTcpPoseDeploymentBinding,
        evaluation_plan: SkillEvaluationPlan,
        motion_profile: MujocoAgentTcpPoseMotionProfile,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(application, MujocoAgentSkillMotionApplication):
            raise TypeError("application must be a motion application.")
        if not isinstance(route, SkillExecutionRoute):
            raise TypeError("route must be typed.")
        if not isinstance(
            deployment_binding,
            MujocoAgentTcpPoseDeploymentBinding,
        ):
            raise TypeError("deployment_binding must be typed.")
        if not isinstance(evaluation_plan, SkillEvaluationPlan):
            raise TypeError("evaluation_plan must be typed.")
        if not isinstance(motion_profile, MujocoAgentTcpPoseMotionProfile):
            raise TypeError("motion_profile must be typed.")
        if not callable(clock) or not callable(sleep):
            raise TypeError("clock and sleep must be callable.")
        checks = (
            (request.semantic_content_hash == operation.request_hash, "request hash"),
            (request.route_hash == route.semantic_content_hash, "Route hash"),
            (
                request.deployment_binding_hash
                == deployment_binding.semantic_content_hash,
                "deployment binding",
            ),
            (
                deployment_binding.route_hash == route.semantic_content_hash,
                "deployment Route",
            ),
            (
                deployment_binding.motion_profile_hash
                == motion_profile.semantic_content_hash,
                "motion profile",
            ),
            (
                evaluation_plan.route_hash == route.semantic_content_hash,
                "evaluation Route",
            ),
            (
                evaluation_plan.deployment_binding_hash
                == deployment_binding.semantic_content_hash,
                "evaluation deployment",
            ),
        )
        for matches, label in checks:
            if not matches:
                raise ValueError(f"TCP pose lifecycle {label} mismatch.")
        if request.runtime_session_id != application.runtime.config.runtime_session_id:
            raise ValueError("TCP pose lifecycle runtime session mismatch.")
        parameters = request.parameters
        if not isinstance(parameters, Mapping):
            raise TypeError("request.parameters must be a mapping.")
        poses = parameters.get("pose_targets_by_component")
        if not isinstance(poses, Mapping) or set(poses) != {"right_arm"}:
            raise ValueError("TCP pose lifecycle requires exactly right_arm.")
        exact = {
            "reference_frame_id": motion_profile.reference_frame_id,
            "reference_frame_spec_hash": motion_profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": motion_profile.frame_graph_hash,
            "tool_config_hash": motion_profile.tool_config_hash,
            "calibration_hash": motion_profile.calibration_hash,
        }
        for field_name, expected in exact.items():
            if parameters.get(field_name) != expected:
                raise ValueError(f"TCP pose lifecycle {field_name} mismatch.")

        self.executor_id = deployment_binding.executor_id
        self.application = application
        self.route = route
        self.deployment_binding = deployment_binding
        self.evaluation_plan = evaluation_plan
        self.motion_profile = motion_profile
        self.request = request
        self.operation = operation
        self.target_pose = _pose(
            poses["right_arm"],
            field_name="pose_targets_by_component.right_arm",
        )
        self.action = ComponentAction(
            action_schema=application.runtime.action_schema,
            values_by_component={
                "right_arm": np.asarray(self.target_pose, dtype=np.float32),
            },
        )
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.RLock()
        self._lower_operation_id = (
            "mujoco-agent-tcp-goal:"
            + canonical_content_hash(
                {
                    "operation_id": operation.operation_id,
                    "request_hash": request.semantic_content_hash,
                    "target_pose": self.target_pose,
                },
                domain="dexteleop.agent_skill.mujoco_tcp_goal_id.v1",
            )
        )
        self._activated = False
        self._closed = False
        self._execution_refs: set[str] = set()

    def _validate_call(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> None:
        if request.semantic_content_hash != self.request.semantic_content_hash:
            raise ValueError("TCP pose lifecycle request changed.")
        if operation.operation_id != self.operation.operation_id:
            raise ValueError("TCP pose lifecycle operation changed.")
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("TCP pose lifecycle request/operation mismatch.")

    def _within_terminal_tolerance(self, pose: tuple[float, ...]) -> bool:
        translation, rotation = _pose_error(pose, self.target_pose)
        return bool(
            translation <= self.motion_profile.translation_tolerance_m
            and rotation <= self.motion_profile.rotation_tolerance_rad
        )

    def _capture_pose(self) -> tuple[float, ...]:
        values = self.application.capture_tcp_pose_by_component()
        if set(values) != {"right_arm"}:
            raise RuntimeError("Measured TCP pose scope is incomplete.")
        return _pose(values["right_arm"], field_name="measured.right_arm")

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleActivationEvidence:
        self._validate_call(request=request, operation=operation)
        if bool(cancel_requested()):
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="tcp_pose_activation_cancelled_before_lower_start",
                lower_operation_id=None,
                evidence_refs=(),
                lower_hold_confirmed=False,
            )
        measured = self._capture_pose()
        translation, rotation = _pose_error(measured, self.target_pose)
        if (
            translation > self.motion_profile.max_translation_delta_m
            or rotation > self.motion_profile.max_rotation_delta_rad
        ):
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="tcp_pose_target_delta_exceeds_profile",
                lower_operation_id=None,
                evidence_refs=(),
                lower_hold_confirmed=False,
            )
        transition = self.application.activate_operation(
            operation_id=operation.operation_id,
        )
        refs = tuple(getattr(transition, "lower_event_ids", ()))
        if not refs or getattr(transition.snapshot, "status", None) != "ready":
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="tcp_pose_lower_ready_not_confirmed",
                lower_operation_id=self._lower_operation_id,
                evidence_refs=refs,
                lower_hold_confirmed=False,
            )
        with self._lock:
            self._activated = True
            self._execution_refs.update(str(item) for item in refs)
        return StationLifecycleActivationEvidence(
            succeeded=True,
            status="tcp_pose_lower_ready_for_goal_submission",
            lower_operation_id=self._lower_operation_id,
            evidence_refs=tuple(sorted(self._execution_refs)),
            lower_hold_confirmed=False,
        )

    @staticmethod
    def _step_refs(step: Any) -> tuple[str, ...]:
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
            for fact in tuple(getattr(authoritative, "trace_facts", ())):
                value = getattr(fact, "fact_id", None)
                if isinstance(value, str) and value.strip():
                    refs.add(value)
            request = getattr(authoritative, "target_execution_request", None)
            request_id = getattr(request, "request_id", None)
            if isinstance(request_id, str) and request_id.strip():
                refs.add(request_id)
        return tuple(sorted(refs))

    def _hold(self, *, reason: str) -> StationLifecycleControlEvidence:
        transition = self.application.request_hold(
            reason=reason,
            operation_id=self.operation.operation_id,
        )
        evidence = mujoco_agent_skill_hold_evidence_from_transition(
            transition,
            lower_operation_id=self._lower_operation_id,
        )
        with self._lock:
            self._execution_refs.update(evidence.evidence_refs)
        return evidence

    def _failure(
        self,
        *,
        reason_code: str,
        hold: StationLifecycleControlEvidence,
    ) -> StationLifecycleExecutionCompletion:
        refs = tuple(sorted(self._execution_refs))
        if not refs:
            raise RuntimeError(
                "TCP pose completion has no lower execution evidence."
            )
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
        if not self._activated:
            raise RuntimeError("TCP pose lifecycle was not activated.")
        stable = 0
        period = 1.0 / float(self.application.runtime.config.decision_hz)
        for cycle in range(self.motion_profile.max_control_cycles):
            if bool(cancel_requested()):
                hold = self._hold(reason="tcp_pose_operation_cancelled")
                return self._failure(
                    reason_code="tcp_pose_operation_cancelled_after_hold",
                    hold=hold,
                )
            event_id = f"{operation.operation_id}:target-cycle:{cycle}"
            step = self.application.submit_and_step(
                operation_id=operation.operation_id,
                action=self.action,
                source_event_id=event_id,
                # The Agent snapshot is a Gateway application cut, not a
                # lower ObservationCutRecord. Keep it as context lineage and
                # do not mislabel it as lower policy-observation evidence.
                observation_cut_id=None,
                context_ids=(request.observation_snapshot_id,),
                physics_ticks_after_dispatch=(
                    self.motion_profile.physics_ticks_per_control_cycle
                ),
            )
            self._execution_refs.update(self._step_refs(step))
            if not step.proposal_admission.accepted:
                hold = self._hold(reason="tcp_pose_proposal_rejected")
                return self._failure(
                    reason_code=(
                        "tcp_pose_proposal_rejected:"
                        + str(step.proposal_admission.rejection_code or "unknown")
                    ),
                    hold=hold,
                )
            authoritative = step.authoritative_step
            if authoritative is None or authoritative.status != "dispatched":
                hold = self._hold(reason="tcp_pose_authoritative_step_failed")
                return self._failure(
                    reason_code="tcp_pose_authoritative_step_failed",
                    hold=hold,
                )
            measured = self._capture_pose()
            stable = stable + 1 if self._within_terminal_tolerance(measured) else 0
            if stable >= self.motion_profile.stable_measurement_count:
                hold = self._hold(reason="tcp_pose_terminal_candidate_reached")
                if not hold.lower_hold_confirmed:
                    return self._failure(
                        reason_code="tcp_pose_terminal_hold_unconfirmed",
                        hold=hold,
                    )
                final_pose = self._capture_pose()
                if not self._within_terminal_tolerance(final_pose):
                    return self._failure(
                        reason_code="tcp_pose_post_hold_pose_outside_tolerance",
                        hold=hold,
                    )
                return self._success(
                    operation=operation,
                    request=request,
                    final_pose=final_pose,
                )
            self.sleep(period)
        hold = self._hold(reason="tcp_pose_max_control_cycles_exceeded")
        return self._failure(
            reason_code="tcp_pose_max_control_cycles_exceeded",
            hold=hold,
        )

    def _success(
        self,
        *,
        operation: AgentSkillOperation,
        request: AgentSkillRequest,
        final_pose: tuple[float, ...],
    ) -> StationLifecycleExecutionCompletion:
        measured_at = float(self.clock())
        final_state_hash = canonical_content_hash(
            {
                "runtime_session_id": request.runtime_session_id,
                "control_domain_id": self.deployment_binding.control_domain_id,
                "operation_id": operation.operation_id,
                "component_id": "right_arm",
                "pose_xyz_quat_xyzw": final_pose,
                "measured_at": measured_at,
                "reference_frame_id": self.motion_profile.reference_frame_id,
                "controlled_frame_id": "tool_tcp",
            },
            domain="dexteleop.agent_skill.mujoco_tcp_terminal_state.v1",
        )
        evaluator_spec = self.evaluation_plan.evaluator(
            "motion.tcp_pose_arrival_rule",
            "1",
        )
        refs = tuple(sorted(self._execution_refs))
        evaluator_receipt = SkillEvaluationReceipt(
            operation_id=operation.operation_id,
            request_hash=request.semantic_content_hash,
            skill_id=request.skill_id,
            skill_version=request.skill_version,
            skill_definition_hash=request.skill_definition_hash,
            route_id=request.route_id,
            route_version=request.route_version,
            route_hash=self.route.semantic_content_hash,
            deployment_binding_hash=self.deployment_binding.semantic_content_hash,
            evaluation_plan_hash=self.evaluation_plan.semantic_content_hash,
            evaluator_id=evaluator_spec.evaluator_id,
            evaluator_version=evaluator_spec.evaluator_version,
            evaluator_hash=evaluator_spec.semantic_content_hash,
            evaluator_kind=evaluator_spec.evaluator_kind,
            evidence_domain_id=evaluator_spec.evidence_domain_id,
            input_evidence_hash=final_state_hash,
            evaluated_cut_id=f"tcp-pose-terminal:{final_state_hash}",
            evaluated_at=measured_at,
            verdict="succeeded",
            reason_codes=("tcp_pose_stable_after_lower_hold",),
            evidence_refs=refs,
        )
        aggregate = SkillAggregateEvaluationReceipt(
            operation_id=operation.operation_id,
            request_hash=request.semantic_content_hash,
            evaluation_plan_hash=self.evaluation_plan.semantic_content_hash,
            aggregation_policy_id=self.evaluation_plan.aggregation_policy_id,
            aggregation_policy_hash=self.evaluation_plan.aggregation_policy_hash,
            consumed_evaluation_receipt_hashes=(
                evaluator_receipt.semantic_content_hash,
            ),
            ignored_evaluation_receipt_hashes=(),
            missing_evaluator_ids=(),
            verdict="succeeded",
            reason_codes=("tcp_pose_stable_after_lower_hold",),
            generated_at=measured_at,
        )
        evaluator_ref = (
            "agent-skill-evaluation:" + evaluator_receipt.semantic_content_hash
        )
        aggregate_ref = (
            "agent-skill-aggregate-evaluation:"
            + aggregate.semantic_content_hash
        )
        result = {
            "execution_receipt_refs": list(refs),
            "final_measured_state_hash": final_state_hash,
        }
        return StationLifecycleExecutionCompletion(
            succeeded=True,
            reason_code="tcp_pose_target_reached_and_held",
            execution_receipt_refs=refs,
            terminal_hold_confirmed=True,
            execution_result=result,
            evaluation=StationLifecycleEvaluationCompletion(
                evaluation_status="completed",
                task_outcome="succeeded",
                reason_code="tcp_pose_terminal_rule_succeeded",
                evaluator_receipt_refs=(evaluator_ref,),
                aggregate_evaluation_receipt_ref=aggregate_ref,
                task_evaluation={
                    "aggregate_evaluation_receipt_ref": aggregate_ref,
                    "task_outcome": "succeeded",
                },
            ),
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        self._validate_call(request=request, operation=operation)
        return self._hold(reason=_required(reason, field_name="reason"))

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        if operation is not None and operation.operation_id != self.operation.operation_id:
            raise ValueError("TCP pose HOLD operation mismatch.")
        return self._hold(reason=_required(reason, field_name="reason"))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.application.release_operation(
                operation_id=self.operation.operation_id,
            )
            self._closed = True


__all__ = [
    "MujocoAgentTcpPoseSkillLifecycle",
    "mujoco_agent_skill_hold_evidence_from_transition",
]
