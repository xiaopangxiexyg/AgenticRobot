"""Exact one-operation CubeLift binding to the existing Sim-HIL application.

The lifecycle coordinates one already-built application graph.  It does not
add a policy loop, queue, Arbiter, safety layer, target resolver, dispatcher,
or simulator task step.  Every motion-producing cycle remains the return value
of :meth:`MujocoSimHILApplication.step`.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_gateway.protocol.contracts import (
    AgentSkillOperation,
    AgentSkillRequest,
)
from agent_gateway.protocol.evaluation_contracts import (
    SkillEvaluationPlan,
)
from agent_gateway.protocol.skill_catalog import SkillExecutionRoute
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleEvaluationCompletion,
    StationLifecycleExecutionCompletion,
)
from agent_gateway.profiles.mujoco.skills.cube_lift.sim_route import (
    CubeLiftSimDeploymentBinding,
    CubeLiftTerminalEvaluatorConfig,
    validate_cube_lift_sim_route_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .execution_evidence import (
    CubeLiftExecutionEvidenceCollector,
    CubeLiftExecutionEvidenceCut,
)
from .terminal_evaluator import (
    CubeLiftLowerHoldProof,
    CubeLiftTerminalStableHoldEvaluator,
    MujocoCubeLiftTerminalHoldSampler,
    cube_lift_lower_hold_proof_from_transition,
)

CUBE_LIFT_CANDIDATE_OBSERVATION_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_candidate_observation.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True)
class CubeLiftCandidateObservation:
    observed_at: float
    cube_center_z_m: float
    left_finger_cube_contact: bool
    right_finger_cube_contact: bool
    table_cube_contact: bool
    finite_state: bool
    candidate_satisfied: bool
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            _finite(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(
            self,
            "cube_center_z_m",
            _finite(self.cube_center_z_m, field_name="cube_center_z_m"),
        )
        for field_name in (
            "left_finger_cube_contact",
            "right_finger_cube_contact",
            "table_cube_contact",
            "finite_state",
            "candidate_satisfied",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")
        payload = self.to_wire(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=CUBE_LIFT_CANDIDATE_OBSERVATION_HASH_DOMAIN_V1,
            ),
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "observed_at": self.observed_at,
            "cube_center_z_m": self.cube_center_z_m,
            "left_finger_cube_contact": self.left_finger_cube_contact,
            "right_finger_cube_contact": self.right_finger_cube_contact,
            "table_cube_contact": self.table_cube_contact,
            "finite_state": self.finite_state,
            "candidate_satisfied": self.candidate_satisfied,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


class MujocoCubeLiftPassiveCandidateObserver:
    """Read the current scene under the existing rig lock without stepping it."""

    def __init__(
        self,
        *,
        application: Any,
        evaluator_config: CubeLiftTerminalEvaluatorConfig,
        baseline_resting_center_z_m: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        runtime = getattr(application, "runtime", None)
        rig = getattr(runtime, "rig", None)
        if runtime is None or rig is None or not callable(
            getattr(rig, "capture_scene", None)
        ):
            raise TypeError("application must expose the existing MuJoCo rig.")
        if not isinstance(evaluator_config, CubeLiftTerminalEvaluatorConfig):
            raise TypeError("evaluator_config must be typed.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.runtime_session_id = _required(
            getattr(getattr(runtime, "config", None), "runtime_session_id", None),
            field_name="runtime_session_id",
        )
        self.control_domain_id = _required(
            getattr(getattr(runtime, "config", None), "control_domain_id", None),
            field_name="control_domain_id",
        )
        self.baseline_resting_center_z_m = _finite(
            baseline_resting_center_z_m,
            field_name="baseline_resting_center_z_m",
        )
        self.application = application
        self.rig = rig
        self.config = evaluator_config
        self.clock = clock
        self._capture_lock = threading.Lock()

    def capture(self) -> CubeLiftCandidateObservation:
        with self._capture_lock:
            measured = self.rig.capture_scene(self._capture_scene)
            observed_at = _finite(self.clock(), field_name="observed_at")
        candidate = bool(
            measured["finite_state"]
            and measured["cube_center_z_m"]
            >= self.baseline_resting_center_z_m + self.config.lift_delta_m
            and (
                not self.config.success_requires_bilateral_contact
                or (
                    measured["left_finger_cube_contact"]
                    and measured["right_finger_cube_contact"]
                )
            )
            and (
                not self.config.requires_no_table_contact
                or not measured["table_cube_contact"]
            )
        )
        return CubeLiftCandidateObservation(
            observed_at=observed_at,
            candidate_satisfied=candidate,
            **measured,
        )

    def _capture_scene(self, scene: Any) -> dict[str, object]:
        object_name = self.config.object_body_name
        pose = scene.body_pose(object_name)
        center_z = float(pose[2, 3])
        if not math.isfinite(center_z):
            raise RuntimeError("Cube center became non-finite during execution.")
        touching: set[str] = set()
        for contact in scene.contacts():
            first = str(contact.body1_name)
            second = str(contact.body2_name)
            if first == object_name:
                touching.add(second)
            elif second == object_name:
                touching.add(first)
        return {
            "cube_center_z_m": center_z,
            "left_finger_cube_contact": (
                self.config.left_finger_body_name in touching
            ),
            "right_finger_cube_contact": (
                self.config.right_finger_body_name in touching
            ),
            "table_cube_contact": self.config.table_body_name in touching,
            "finite_state": bool(scene.state_is_finite()),
        }


class CubeLiftAgentSkillLifecycle:
    """Single-use lifecycle adapter around one exact policy-only application."""

    def __init__(
        self,
        *,
        application: Any,
        route: SkillExecutionRoute,
        deployment_binding: CubeLiftSimDeploymentBinding,
        evaluation_plan: SkillEvaluationPlan,
        evaluator_config: CubeLiftTerminalEvaluatorConfig,
        station_manifest: Any,
        execution_manifest: Any,
        trace_capture: Any,
        evidence_collector: CubeLiftExecutionEvidenceCollector,
        candidate_observer: Any,
        terminal_evaluator: CubeLiftTerminalStableHoldEvaluator,
        max_control_cycles: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_cube_lift_sim_route_v1(route=route)
        if not isinstance(deployment_binding, CubeLiftSimDeploymentBinding):
            raise TypeError("deployment_binding must be typed.")
        if not isinstance(evaluation_plan, SkillEvaluationPlan):
            raise TypeError("evaluation_plan must be typed.")
        if not isinstance(evaluator_config, CubeLiftTerminalEvaluatorConfig):
            raise TypeError("evaluator_config must be typed.")
        if deployment_binding.route_hash != route.semantic_content_hash:
            raise ValueError("Deployment binding does not match the CubeLift Route.")
        if evaluation_plan.route_hash != route.semantic_content_hash:
            raise ValueError("Evaluation plan does not match the CubeLift Route.")
        if (
            evaluation_plan.deployment_binding_hash
            != deployment_binding.semantic_content_hash
        ):
            raise ValueError("Evaluation plan does not match the deployment binding.")
        if (
            deployment_binding.evaluator_configuration_hash
            != evaluator_config.semantic_content_hash
        ):
            raise ValueError("Deployment binding uses another evaluator config.")
        if (
            getattr(station_manifest, "semantic_content_hash", None)
            != deployment_binding.station_manifest_hash
            or getattr(execution_manifest, "semantic_content_hash", None)
            != deployment_binding.execution_manifest_hash
        ):
            raise ValueError("Lower Station/execution manifest hashes mismatch.")
        runtime = getattr(application, "runtime", None)
        runtime_config = getattr(runtime, "config", None)
        if (
            getattr(runtime_config, "runtime_session_id", None)
            != deployment_binding.runtime_session_id
            or getattr(runtime_config, "control_domain_id", None)
            != deployment_binding.control_domain_id
        ):
            raise ValueError("Application runtime identity does not match deployment.")
        app_config = getattr(application, "config", None)
        if (
            float(getattr(app_config, "initial_human_alpha", -1.0)) != 0.0
            or getattr(app_config, "defer_human_input", None) is not True
            or getattr(app_config, "realtime_worker", None) is not False
            or getattr(app_config, "mujoco_viewer_enabled", None) is not False
            or getattr(app_config, "execution_difference_enabled", None) is not True
        ):
            raise ValueError("CubeLift Agent Skill requires the exact policy-only config.")
        trace_session = getattr(
            getattr(trace_capture, "trace_runtime", None),
            "session",
            None,
        )
        if (
            getattr(trace_session, "runtime_session_id", None)
            != deployment_binding.runtime_session_id
            or getattr(trace_session, "control_domain_id", None)
            != deployment_binding.control_domain_id
            or getattr(trace_session, "profile_id", None)
            != deployment_binding.trace_requirement_profile_id
        ):
            raise ValueError("Trace session identity/profile does not match deployment.")
        for method_name in (
            "prepare",
            "finalize",
            "complete_runtime",
            "abort",
            "close",
        ):
            if not callable(getattr(trace_capture, method_name, None)):
                raise TypeError(f"trace_capture must expose {method_name}().")
        for method_name in ("start", "step", "hold", "end_episode", "close"):
            if not callable(getattr(application, method_name, None)):
                raise TypeError(f"application must expose {method_name}().")
        if not isinstance(evidence_collector, CubeLiftExecutionEvidenceCollector):
            raise TypeError("evidence_collector must be typed.")
        for object_value, method_name, label in (
            (candidate_observer, "capture", "candidate_observer"),
            (terminal_evaluator, "evaluate", "terminal_evaluator"),
        ):
            if not callable(getattr(object_value, method_name, None)):
                raise TypeError(f"{label} must expose {method_name}().")
        cycle_limit = int(max_control_cycles)
        if (
            isinstance(max_control_cycles, bool)
            or cycle_limit <= 0
            or cycle_limit != max_control_cycles
        ):
            raise ValueError("max_control_cycles must be a positive integer.")
        if not callable(clock) or not callable(sleep):
            raise TypeError("clock and sleep must be callable.")
        baseline = self._baseline(application, evaluator_config)
        if (
            getattr(candidate_observer, "runtime_session_id", None)
            != deployment_binding.runtime_session_id
            or getattr(candidate_observer, "control_domain_id", None)
            != deployment_binding.control_domain_id
            or abs(
                float(getattr(candidate_observer, "baseline_resting_center_z_m", math.nan))
                - baseline
            )
            > 1e-12
        ):
            raise ValueError("Candidate observer does not match the exact deployment.")
        if (
            evidence_collector.runtime_session_id
            != deployment_binding.runtime_session_id
            or evidence_collector.control_domain_id
            != deployment_binding.control_domain_id
        ):
            raise ValueError("Execution evidence collector identity mismatch.")

        self.executor_id = deployment_binding.executor_id
        self.application = application
        self.route = route
        self.deployment_binding = deployment_binding
        self.evaluation_plan = evaluation_plan
        self.evaluator_config = evaluator_config
        self.station_manifest = station_manifest
        self.execution_manifest = execution_manifest
        self.trace_capture = trace_capture
        self.evidence_collector = evidence_collector
        self.candidate_observer = candidate_observer
        self.terminal_evaluator = terminal_evaluator
        self.max_control_cycles = cycle_limit
        self.baseline_resting_center_z_m = baseline
        self.clock = clock
        self.sleep = sleep
        self._state_lock = threading.RLock()
        self._hold_lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._activated_request_hash: str | None = None
        self._activated_operation_id: str | None = None
        self._lower_operation_id: str | None = None
        self._hold_proof: CubeLiftLowerHoldProof | None = None
        self._execution_cut: CubeLiftExecutionEvidenceCut | None = None
        self._trace_finalization: Any | None = None
        self._trace_prepared = False
        self._trace_finalized = False
        self._closed = False

    @staticmethod
    def _baseline(
        application: Any,
        evaluator_config: CubeLiftTerminalEvaluatorConfig,
    ) -> float:
        info = dict(getattr(application, "task_initialization_info", {}) or {})
        task = getattr(application, "task_config", None)
        if info.get("owner") != "sim.mujoco.tasks.cube_lift.CubeLiftKernel.reset":
            raise ValueError("CubeLift Agent Skill requires Kernel-owned reset evidence.")
        if info.get("seed") != 0:
            raise ValueError("CubeLift reset evidence must bind seed 0.")
        for field_name in ("scripted_pregrasp_reset", "scripted_contact_reset"):
            if info.get(field_name) is not bool(getattr(task, field_name, False)):
                raise ValueError(f"CubeLift reset evidence {field_name} mismatch.")
        baseline = _finite(
            info.get("resting_center_z_m"),
            field_name="task_initialization_info.resting_center_z_m",
        )
        if (
            getattr(task, "object_body_name", None)
            != evaluator_config.object_body_name
        ):
            raise ValueError("CubeLift task/evaluator object identity mismatch.")
        return baseline

    def _validate_call(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        expected_status: str,
    ) -> None:
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation.")
        if operation.status != expected_status:
            raise ValueError(
                f"CubeLift lifecycle requires operation status {expected_status!r}."
            )
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("CubeLift operation/request hash mismatch.")
        checks = (
            (request.runtime_session_id, self.deployment_binding.runtime_session_id),
            (request.skill_id, self.route.skill_id),
            (request.skill_version, self.route.skill_version),
            (request.skill_definition_hash, self.route.skill_definition_hash),
            (request.route_id, self.route.route_id),
            (request.route_version, self.route.route_version),
            (request.route_hash, self.route.semantic_content_hash),
            (
                request.deployment_binding_hash,
                self.deployment_binding.semantic_content_hash,
            ),
        )
        if any(actual != expected for actual, expected in checks):
            raise ValueError("CubeLift request does not bind this exact lifecycle.")

    def _activation_scope(self) -> Any:
        scope = getattr(self.application.composition, "activation_scope", None)
        if scope is None:
            raise RuntimeError("CubeLift application has no activation scope.")
        return scope

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleActivationEvidence:
        self._validate_call(
            request=request,
            operation=operation,
            expected_status="activated",
        )
        if not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable.")
        with self._state_lock:
            if self._activated_operation_id is not None:
                raise RuntimeError("CubeLift lifecycle is single-use.")
        report = self.application.start(cancel_requested=cancel_requested)
        scope = self._activation_scope()
        lower_operation_id = _required(
            getattr(scope, "activation_id", None),
            field_name="activation_scope.activation_id",
        )
        scope_hash = _required(
            getattr(scope, "semantic_content_hash", None),
            field_name="activation_scope.semantic_content_hash",
        )
        trace_status = self.trace_capture.prepare(now=float(self.application.clock()))
        episode_uuid = _required(
            trace_status.get("episode_uuid"),
            field_name="trace episode_uuid",
        )
        preflight_ref = _required(
            getattr(report, "semantic_content_hash", scope_hash),
            field_name="preflight evidence ref",
        )
        with self._state_lock:
            self._activated_request_hash = request.semantic_content_hash
            self._activated_operation_id = operation.operation_id
            self._lower_operation_id = lower_operation_id
            self._trace_prepared = True
        evidence_refs = (
            lower_operation_id,
            scope_hash,
            preflight_ref,
            f"control-trace-episode:{episode_uuid}",
            self.deployment_binding.station_manifest_hash,
            self.deployment_binding.execution_manifest_hash,
        )
        if bool(cancel_requested()):
            proof = self._ensure_hold(reason="activation_preempted")
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="cube_lift_activation_preempted_under_lower_hold",
                lower_operation_id=lower_operation_id,
                evidence_refs=tuple(
                    dict.fromkeys((*evidence_refs, *self._hold_refs(proof)))
                ),
                lower_hold_confirmed=True,
            )
        return StationLifecycleActivationEvidence(
            succeeded=True,
            status="cube_lift_application_activated",
            lower_operation_id=lower_operation_id,
            evidence_refs=evidence_refs,
        )

    def _wait_for_station_cut(
        self,
        cancel_requested: Callable[[], bool],
    ) -> bool:
        while True:
            if bool(cancel_requested()):
                return False
            due = getattr(self.application, "next_decision_at", None)
            if due is None:
                return True
            now = _finite(self.application.clock(), field_name="application.clock")
            remaining = float(due) - now
            if remaining <= 0.0:
                return True
            self.sleep(min(remaining, 0.005))

    @staticmethod
    def _hold_refs(proof: CubeLiftLowerHoldProof) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    proof.hold_receipt_ref,
                    proof.cancellation_receipt_ref,
                    proof.gate_snapshot_hash,
                    *proof.evidence_refs,
                )
            )
        )

    def _ensure_hold(
        self,
        *,
        reason: str,
        end_episode: bool = False,
    ) -> CubeLiftLowerHoldProof:
        normalized_reason = _required(reason, field_name="reason")
        with self._hold_lock:
            if self._hold_proof is not None:
                return self._hold_proof
            outcome = None
            if end_episode:
                outcome = self.application.end_episode(
                    reason=normalized_reason,
                    force_cancellation=True,
                )
            else:
                outcome = self.application.hold(
                    reason=normalized_reason,
                    force_cancellation=True,
                )
            if outcome is None:
                outcome = getattr(self.application, "_last_hold_outcome", None)
            proof = cube_lift_lower_hold_proof_from_transition(
                outcome,
                runtime_session_id=self.deployment_binding.runtime_session_id,
                control_domain_id=self.deployment_binding.control_domain_id,
                confirmed_at=float(self.clock()),
                clock=self.clock,
            )
            self._hold_proof = proof
            return proof

    def _cancelled_completion(
        self,
        *,
        reason_code: str,
        proof: CubeLiftLowerHoldProof,
    ) -> StationLifecycleExecutionCompletion:
        return StationLifecycleExecutionCompletion(
            succeeded=False,
            reason_code=reason_code,
            execution_receipt_refs=self._hold_refs(proof),
            terminal_hold_confirmed=True,
        )

    def _failed_cycle_completion(
        self,
        cycle: Any,
        *,
        reason_code: str,
    ) -> StationLifecycleExecutionCompletion:
        """Project lower refusal evidence without claiming task evaluation."""

        proof = self._ensure_hold(reason=reason_code, end_episode=True)
        execution = getattr(cycle, "execution", None)
        diagnostic = {
            "phase": "lower_execution",
            "lower_execution_status": str(getattr(execution, "status", "missing")),
            "cycle_index": str(getattr(cycle, "cycle_index", "missing")),
        }
        references = list(self._hold_refs(proof))
        for field, id_field in (
            ("publication_attempt", "publication_attempt_id"),
            ("directive", "directive_id"),
            ("cancellation", "cancellation_id"),
        ):
            value = getattr(execution, field, None)
            reference = getattr(value, id_field, None)
            if reference is not None:
                references.append(reference)
        publication = getattr(execution, "publication_attempt", None)
        rejection_reason = getattr(publication, "rejection_reason", None)
        if rejection_reason is not None:
            diagnostic["publication_rejection_reason"] = rejection_reason
        directive = getattr(execution, "directive", None)
        if directive is not None:
            diagnostic["directive_status"] = directive.global_status
        return StationLifecycleExecutionCompletion(
            succeeded=False,
            reason_code=reason_code,
            execution_receipt_refs=tuple(dict.fromkeys(references)),
            terminal_hold_confirmed=True,
            failure_diagnostic=diagnostic,
        )

    def run(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleExecutionCompletion:
        self._validate_call(
            request=request,
            operation=operation,
            expected_status="running",
        )
        with self._state_lock:
            if (
                self._activated_operation_id != operation.operation_id
                or self._activated_request_hash != request.semantic_content_hash
            ):
                raise ValueError("CubeLift run does not match the activation.")
        observed_dispatch = False
        for _ in range(self.max_control_cycles):
            if not self._wait_for_station_cut(cancel_requested):
                proof = self._ensure_hold(reason="operation_preempted")
                return self._cancelled_completion(
                    reason_code="cube_lift_execution_preempted_under_hold",
                    proof=proof,
                )
            cycle = self.application.step()
            execution_status = getattr(getattr(cycle, "execution", None), "status", None)
            if execution_status not in {"dispatched", "not_due", "execution_kept_existing"}:
                return self._failed_cycle_completion(
                    cycle,
                    reason_code="cube_lift_lower_execution_failed_under_hold",
                )
            measurement_snapshot = None
            if execution_status == "dispatched":
                observer = getattr(self.application, "execution_difference_observer", None)
                if observer is None or not callable(getattr(observer, "snapshot", None)):
                    raise RuntimeError("CubeLift execution observer is unavailable.")
                measurement_snapshot = observer.snapshot()
            self.evidence_collector.observe_cycle(
                cycle,
                measurement_snapshot=measurement_snapshot,
            )
            observed_dispatch = observed_dispatch or execution_status == "dispatched"
            if bool(cancel_requested()):
                proof = self._ensure_hold(reason="operation_preempted_after_cycle")
                return self._cancelled_completion(
                    reason_code="cube_lift_execution_preempted_under_hold",
                    proof=proof,
                )
            if execution_status != "dispatched":
                continue
            candidate = self.candidate_observer.capture()
            if not isinstance(candidate, CubeLiftCandidateObservation):
                raise TypeError("Candidate observer returned an invalid observation.")
            if candidate.candidate_satisfied:
                break

        if not observed_dispatch:
            return self._failed_cycle_completion(
                cycle,
                reason_code="cube_lift_no_dispatch_under_hold",
            )
        proof = self._ensure_hold(
            reason="candidate_or_control_cycle_limit",
            end_episode=True,
        )
        execution_cut = self.evidence_collector.seal_for_hold(proof)
        with self._state_lock:
            self._execution_cut = execution_cut
        try:
            evaluation = self.terminal_evaluator.evaluate(
                operation=operation,
                request=request,
                route=self.route,
                deployment_binding=self.deployment_binding,
                evaluation_plan=self.evaluation_plan,
                lower_hold_proof=proof,
                baseline_resting_center_z_m=self.baseline_resting_center_z_m,
                execution_evidence_chain_ref=(
                    execution_cut.execution_evidence_chain_ref
                ),
                terminal_state_hash=execution_cut.terminal_state_hash,
                execution_lineage_complete=True,
                cancel_requested=cancel_requested,
            )
        except InterruptedError:
            return self._cancelled_completion(
                reason_code="cube_lift_terminal_evaluation_preempted_under_hold",
                proof=proof,
            )

        self.trace_capture.finalize(now=float(self.application.clock()))
        with self._state_lock:
            self._trace_finalized = True
            self._trace_prepared = False
        self.application.close()
        trace_finalization = self.trace_capture.complete_runtime()
        self.trace_capture.close(now=float(self.clock()))
        with self._state_lock:
            self._trace_finalization = trace_finalization
            self._closed = True
        execution_result = execution_cut.to_execution_result(
            lower_hold_proof=proof,
            trace_finalization=trace_finalization,
        )
        evaluator_receipt_ref = (
            "skill-evaluation-receipt:"
            f"{evaluation.evaluation_receipt.semantic_content_hash}"
        )
        aggregate_receipt_ref = (
            "skill-aggregate-evaluation-receipt:"
            f"{evaluation.aggregate_receipt.semantic_content_hash}"
        )
        evaluation_completion = StationLifecycleEvaluationCompletion(
            evaluation_status="completed",
            task_outcome=evaluation.aggregate_receipt.verdict,
            reason_code="cube_lift_required_environment_evaluation_completed",
            evaluator_receipt_refs=(evaluator_receipt_ref,),
            aggregate_evaluation_receipt_ref=aggregate_receipt_ref,
            task_evaluation=evaluation.task_evaluation,
        )
        return StationLifecycleExecutionCompletion(
            succeeded=True,
            reason_code="cube_lift_authoritative_execution_completed_under_hold",
            execution_receipt_refs=tuple(
                dict.fromkeys(
                    (
                        *execution_cut.execution_receipt_refs,
                        *self._hold_refs(proof),
                        trace_finalization.trace_terminal_receipt_ref,
                        trace_finalization.trace_commit_receipt_ref,
                        f"control-trace-finalization:{trace_finalization.semantic_content_hash}",
                    )
                )
            ),
            terminal_hold_confirmed=True,
            execution_result=execution_result,
            evaluation=evaluation_completion,
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        if not isinstance(request, AgentSkillRequest) or not isinstance(
            operation,
            AgentSkillOperation,
        ):
            raise TypeError("cancel requires typed request and operation.")
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("Cancel operation/request mismatch.")
        proof = self._ensure_hold(reason=f"cancel:{_required(reason, field_name='reason')}")
        return StationLifecycleControlEvidence(
            succeeded=True,
            status="cube_lift_cancel_confirmed_under_lower_hold",
            lower_operation_id=self._lower_operation_id,
            evidence_refs=self._hold_refs(proof),
            lower_hold_confirmed=True,
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        if operation is not None and self._activated_operation_id not in {
            None,
            operation.operation_id,
        }:
            raise ValueError("HOLD operation identity mismatch.")
        proof = self._ensure_hold(reason=f"hold:{_required(reason, field_name='reason')}")
        return StationLifecycleControlEvidence(
            succeeded=True,
            status="cube_lift_lower_hold_confirmed",
            lower_operation_id=self._lower_operation_id,
            evidence_refs=self._hold_refs(proof),
            lower_hold_confirmed=True,
        )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            errors: list[BaseException] = []
            if bool(getattr(self.application, "started", False)):
                try:
                    self._ensure_hold(reason="lifecycle_close")
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
            if self._trace_prepared and not self._trace_finalized:
                try:
                    self.trace_capture.abort(now=float(self.clock()))
                    self._trace_prepared = False
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
            try:
                self.application.close()
            except BaseException as error:  # noqa: BLE001 - trace cleanup continues.
                errors.append(error)
            if self._trace_finalized and self._trace_finalization is None:
                try:
                    self._trace_finalization = self.trace_capture.complete_runtime()
                except BaseException as error:  # noqa: BLE001 - close still required.
                    errors.append(error)
            try:
                self.trace_capture.close(now=float(self.clock()))
            except BaseException as error:  # noqa: BLE001 - report all cleanup errors.
                errors.append(error)
            self._closed = True
            if errors:
                raise RuntimeError(
                    "CubeLift lifecycle close encountered errors: "
                    + "; ".join(
                        f"{type(error).__name__}:{error}" for error in errors
                    )
                ) from errors[0]


def build_mujoco_cube_lift_agent_skill_lifecycle(
    *,
    application: Any,
    route: SkillExecutionRoute,
    deployment_binding: CubeLiftSimDeploymentBinding,
    evaluation_plan: SkillEvaluationPlan,
    evaluator_config: CubeLiftTerminalEvaluatorConfig,
    station_manifest: Any,
    execution_manifest: Any,
    trace_capture: Any,
    max_control_cycles: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> CubeLiftAgentSkillLifecycle:
    """Assemble observers/evaluator around one existing application graph."""

    info = dict(getattr(application, "task_initialization_info", {}) or {})
    baseline = _finite(
        info.get("resting_center_z_m"),
        field_name="task_initialization_info.resting_center_z_m",
    )
    collector = CubeLiftExecutionEvidenceCollector(
        runtime_session_id=deployment_binding.runtime_session_id,
        control_domain_id=deployment_binding.control_domain_id,
    )
    candidate = MujocoCubeLiftPassiveCandidateObserver(
        application=application,
        evaluator_config=evaluator_config,
        baseline_resting_center_z_m=baseline,
        clock=clock,
    )
    sampler = MujocoCubeLiftTerminalHoldSampler(
        application=application,
        task_config=application.task_config,
        task_config_hash=deployment_binding.task_config_hash,
        evaluator_config=evaluator_config,
        clock=clock,
    )
    terminal = CubeLiftTerminalStableHoldEvaluator(
        config=evaluator_config,
        sampler=sampler,
        clock=clock,
    )
    return CubeLiftAgentSkillLifecycle(
        application=application,
        route=route,
        deployment_binding=deployment_binding,
        evaluation_plan=evaluation_plan,
        evaluator_config=evaluator_config,
        station_manifest=station_manifest,
        execution_manifest=execution_manifest,
        trace_capture=trace_capture,
        evidence_collector=collector,
        candidate_observer=candidate,
        terminal_evaluator=terminal,
        max_control_cycles=max_control_cycles,
        clock=clock,
        sleep=sleep,
    )


__all__ = [
    "CUBE_LIFT_CANDIDATE_OBSERVATION_HASH_DOMAIN_V1",
    "CubeLiftAgentSkillLifecycle",
    "CubeLiftCandidateObservation",
    "MujocoCubeLiftPassiveCandidateObserver",
    "build_mujoco_cube_lift_agent_skill_lifecycle",
]
