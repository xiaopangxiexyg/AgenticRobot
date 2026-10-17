"""Per-operation lifecycle for the bounded no-model TCP-pose Skill."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution as MujocoAgentTcpTargetResolution,
)
from agent_gateway.profiles.motion_primitive_resolution import (
    resolve_level_tcp_orientation_primitive_target,
    resolve_tcp_displacement_primitive_target,
)
from agent_gateway.protocol.contracts import (
    AgentSkillOperation,
    AgentSkillRequest,
)
from agent_gateway.protocol.evaluation_contracts import (
    SkillAggregateEvaluationReceipt,
    SkillEvaluationPlan,
    SkillEvaluationReceipt,
)
from agent_gateway.protocol.skill_catalog import SkillExecutionRoute
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleEvaluationCompletion,
    StationLifecycleExecutionCompletion,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentAction
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.target_safety import (
    TCPPoseEnvelopeLifecycleError,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSelection,
)
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

from .motion_application import MujocoAgentSkillMotionApplication
from .motion_deployment import (
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


def _slerp_quaternion_xyzw(
    start: tuple[float, ...],
    target: tuple[float, ...],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Interpolate the shortest normalized quaternion arc without SciPy."""

    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    blend = min(1.0, max(0.0, float(fraction)))
    if dot > 0.9995:
        result = first + blend * (second - first)
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        result = (
            math.sin((1.0 - blend) * angle) / sine * first
            + math.sin(blend * angle) / sine * second
        )
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Interpolated TCP quaternion is invalid.")
    normalized = result / norm
    return tuple(float(value) for value in normalized)


_RETRYABLE_PUBLICATION_NO_ACTION_REASONS = frozenset(
    {
        "coalesced_before_publication",
        "publication_deadline_missed:no_action",
        "target_late:no_action",
    }
)


def _authoritative_step_keeps_existing_for_retry(authoritative: Any) -> bool:
    """Recognize bounded, non-faulting no-action timing dispositions.

    The authoritative runtime may reject a late/coalesced publication while
    explicitly keeping the previously dispatched segment alive.  That is not
    proof that the newly submitted reference executed, so callers must retry
    the same reference.  Every other non-dispatch status remains fail-closed.
    """

    status = getattr(authoritative, "status", None)
    if status == "target_support_expired_before_submission_keep_existing":
        publication = getattr(authoritative, "publication_attempt", None)
        directive = getattr(authoritative, "directive", None)
        directives = getattr(directive, "directive_by_selection_group", None)
        adaptation = getattr(authoritative, "action_adaptation", None)
        return bool(
            getattr(authoritative, "decision", None) is not None
            and publication is not None
            and bool(getattr(publication, "accepted", False))
            and getattr(directive, "global_status", None) == "normal"
            and isinstance(directives, Mapping)
            and directives
            and all(
                getattr(item, "directive", None) == "publish_selected"
                for item in directives.values()
            )
            and adaptation is not None
            and bool(
                getattr(
                    getattr(adaptation, "receipt", None),
                    "accepted",
                    False,
                )
            )
            and getattr(authoritative, "target_execution_request", None) is None
            and getattr(authoritative, "target_submission_receipt", None) is None
            and getattr(authoritative, "target_dispatch_result", None) is None
            and getattr(authoritative, "target_completion_receipt", None) is None
            and getattr(authoritative, "cancellation", None) is None
            and getattr(authoritative, "safety_result", None) is None
            and getattr(authoritative, "resolution_failure", None) is None
            and getattr(authoritative, "dispatch_validation", None) is None
            and getattr(authoritative, "dispatch_receipts", None) is None
        )
    if status != "execution_kept_existing":
        return False
    if getattr(authoritative, "decision", None) is not None:
        return False
    publication = getattr(authoritative, "publication_attempt", None)
    if (
        publication is None
        or bool(getattr(publication, "accepted", True))
        or getattr(publication, "required_fallback", None) != "no_action"
        or getattr(publication, "rejection_reason", None)
        not in _RETRYABLE_PUBLICATION_NO_ACTION_REASONS
    ):
        return False
    directive = getattr(authoritative, "directive", None)
    directives = getattr(directive, "directive_by_selection_group", None)
    if (
        getattr(directive, "global_status", None) != "normal"
        or not isinstance(directives, Mapping)
        or not directives
        or any(
            getattr(item, "directive", None) != "keep_existing_until_expiry"
            for item in directives.values()
        )
    ):
        return False
    return bool(
        getattr(authoritative, "cancellation", None) is None
        and getattr(authoritative, "safety_result", None) is None
        and getattr(authoritative, "resolution_failure", None) is None
        and getattr(authoritative, "dispatch_receipts", None) is None
    )


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
        target_resolver: (
            Callable[[tuple[float, ...]], MujocoAgentTcpTargetResolution] | None
        ) = None,
        max_reference_rotation_step_rad: float | None = None,
        max_measured_translation_excursion_m: float | None = None,
        parameter_lineage: Mapping[str, str] | None = None,
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
        if target_resolver is not None and not callable(target_resolver):
            raise TypeError("target_resolver must be callable or None.")
        for field_name, value in (
            (
                "max_reference_rotation_step_rad",
                max_reference_rotation_step_rad,
            ),
            (
                "max_measured_translation_excursion_m",
                max_measured_translation_excursion_m,
            ),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{field_name} must be finite and positive.")
        lineage = (
            {
                "reference_frame_id": motion_profile.reference_frame_id,
                "reference_frame_spec_hash": (
                    motion_profile.reference_frame_spec_hash
                ),
                "controlled_frame_id": "tool_tcp",
                "frame_graph_hash": motion_profile.frame_graph_hash,
                "tool_config_hash": motion_profile.tool_config_hash,
                "calibration_hash": motion_profile.calibration_hash,
            }
            if parameter_lineage is None
            else dict(parameter_lineage)
        )
        required_lineage_fields = {
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        }
        if set(lineage) != required_lineage_fields:
            raise ValueError(
                "parameter_lineage must exactly contain the motion lineage fields."
            )
        poses = parameters.get("pose_targets_by_component")
        if target_resolver is None:
            if not isinstance(poses, Mapping) or len(poses) != 1:
                raise ValueError(
                    "TCP pose lifecycle requires exactly one arm component."
                )
            component_id = str(next(iter(poses)))
        else:
            requested_component = parameters.get("component_id")
            if requested_component is None and isinstance(poses, Mapping) and len(
                poses
            ) == 1:
                requested_component = next(iter(poses))
            component_id = _required(
                requested_component,
                field_name="component_id",
            )
        if component_id not in {"left_arm", "right_arm"}:
            raise ValueError(
                "TCP pose lifecycle component_id must be left_arm or right_arm."
            )
        if component_id not in application.runtime.action_schema.component_ids:
            raise ValueError(
                "TCP pose lifecycle component is absent from the runtime schema."
            )
        if motion_profile.component_id != component_id:
            raise ValueError(
                "TCP pose lifecycle component does not match the motion profile."
            )
        if deployment_binding.component_id != component_id:
            raise ValueError(
                "TCP pose lifecycle component does not match the deployment "
                "binding."
            )
        for field_name, expected in lineage.items():
            if parameters.get(field_name) != expected:
                raise ValueError(f"TCP pose lifecycle {field_name} mismatch.")

        self.executor_id = deployment_binding.executor_id
        self.application = application
        self.route = route
        self.deployment_binding = deployment_binding
        self.evaluation_plan = evaluation_plan
        self.motion_profile = motion_profile
        self.component_id = component_id
        self.request = request
        self.operation = operation
        self._target_resolver = target_resolver
        self._max_reference_rotation_step_rad = (
            None
            if max_reference_rotation_step_rad is None
            else float(max_reference_rotation_step_rad)
        )
        self._max_measured_translation_excursion_m = (
            None
            if max_measured_translation_excursion_m is None
            else float(max_measured_translation_excursion_m)
        )
        self.parameter_lineage = lineage
        requested_target_pose: tuple[float, ...] | None = (
            None
            if target_resolver is not None
            else _pose(
                poses[component_id],
                field_name=f"pose_targets_by_component.{component_id}",
            )
        )
        self.public_target_pose = requested_target_pose
        self.target_pose: tuple[float, ...] | None = (
            None
            if requested_target_pose is None
            else application.transform_tcp_pose_xyzquat(
                requested_target_pose,
                source_frame_id=motion_profile.reference_frame_id,
                target_frame_id=motion_profile.execution_reference_frame_id,
            )
        )
        self.action: ComponentAction | None = (
            None
            if self.target_pose is None
            else ComponentAction(
                action_schema=application.runtime.action_schema,
                values_by_component={
                    component_id: np.asarray(self.target_pose, dtype=np.float32),
                },
            )
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
                    "target_semantics": (
                        "lower_resolved_at_activation"
                        if target_resolver is not None
                        else self.target_pose
                    ),
                },
                domain="dexteleop.agent_skill.mujoco_tcp_goal_id.v1",
            )
        )
        self._activated = False
        self._closed = False
        self._execution_refs: set[str] = set()
        self._first_step_refs: tuple[str, ...] = ()
        self._latest_step_refs: tuple[str, ...] = ()
        self._activation_measured_pose: tuple[float, ...] | None = None

    def _reference_action_for_cycle(self, cycle: int) -> ComponentAction:
        if self.action is None or self.target_pose is None:
            raise RuntimeError("TCP pose lifecycle has no resolved action.")
        step = self._max_reference_rotation_step_rad
        start = self._activation_measured_pose
        if step is None or start is None:
            return self.action
        _translation, total_rotation = _pose_error(start, self.target_pose)
        sample_count = max(1, int(math.ceil(total_rotation / step)))
        fraction = min(
            1.0,
            float(cycle + 1) / float(sample_count),
        )
        reference_pose = (
            *self.target_pose[:3],
            *_slerp_quaternion_xyzw(
                start[3:],
                self.target_pose[3:],
                fraction,
            ),
        )
        return ComponentAction(
            action_schema=self.application.runtime.action_schema,
            values_by_component={
                self.component_id: np.asarray(reference_pose, dtype=np.float32),
            },
        )

    def _measured_translation_excursion_exceeded(
        self,
        measured: tuple[float, ...],
    ) -> bool:
        limit = self._max_measured_translation_excursion_m
        start = self._activation_measured_pose
        if limit is None or start is None:
            return False
        excursion = math.sqrt(
            sum((first - second) ** 2 for first, second in zip(measured[:3], start[:3]))
        )
        return bool(excursion > limit)

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
        if self.target_pose is None:
            raise RuntimeError("TCP target has not been resolved.")
        translation, rotation = _pose_error(pose, self.target_pose)
        return bool(
            translation <= self.motion_profile.translation_tolerance_m
            and rotation <= self.motion_profile.rotation_tolerance_rad
        )

    def _capture_pose(self) -> tuple[float, ...]:
        values = self.application.capture_tcp_pose_by_component()
        if self.component_id not in values:
            raise RuntimeError(
                "Measured TCP pose does not include the selected arm component."
            )
        return _pose(
            values[self.component_id],
            field_name=f"measured.{self.component_id}",
        )

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
        self._activation_measured_pose = measured
        if self._target_resolver is not None:
            resolution = self._target_resolver(measured)
            if not isinstance(resolution, MujocoAgentTcpTargetResolution):
                raise TypeError("target_resolver returned an invalid resolution.")
            self.target_pose = resolution.target_pose
            self.public_target_pose = self.application.transform_tcp_pose_xyzquat(
                self.target_pose,
                source_frame_id=self.motion_profile.execution_reference_frame_id,
                target_frame_id=self.motion_profile.reference_frame_id,
            )
            self.action = ComponentAction(
                action_schema=self.application.runtime.action_schema,
                values_by_component={
                    self.component_id: np.asarray(
                        self.target_pose,
                        dtype=np.float32,
                    ),
                },
            )
            with self._lock:
                self._execution_refs.update(resolution.evidence_refs)
        if self.target_pose is None or self.action is None:
            raise RuntimeError("TCP target resolution did not produce an action.")
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
        if getattr(self.application, "tcp_pose_envelope_runtime", None) is None:
            transition = self.application.activate_operation(
                operation_id=operation.operation_id,
            )
        else:
            safety_runtime = self.application.tcp_pose_envelope_runtime
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
                else self._route_bound_default_safety_selection(safety_runtime)
            )
            transition = self.application.activate_operation(
                operation_id=operation.operation_id,
                route_id=request.route_id,
                safety_selection=selection,
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

    def _route_bound_default_safety_selection(
        self,
        safety_runtime: TCPPoseEnvelopeSafetyRuntime,
    ) -> TCPPoseEnvelopeSelection:
        """Select the sole envelope pinned by this exact deployment binding."""

        refs = tuple(self.deployment_binding.safety_envelope_refs)
        if len(refs) != 1:
            raise TCPPoseEnvelopeLifecycleError(
                "A motion deployment without an explicit request selection must "
                "pin exactly one safety envelope."
            )
        ref = refs[0]
        matches = tuple(
            entry
            for entry in safety_runtime.catalog_entries
            if entry.semantic_content_hash == ref.content_hash
            and entry.envelope_id == ref.object_id
        )
        if len(matches) != 1:
            raise TCPPoseEnvelopeLifecycleError(
                "The deployment-bound safety envelope is absent from lower runtime."
            )
        entry = matches[0]
        if self.route.route_id not in entry.allowed_route_ids:
            raise TCPPoseEnvelopeLifecycleError(
                "The deployment-bound safety envelope rejects this Route."
            )
        profile_hash = canonical_content_hash(
            {
                "source": "lower_route_binding_single_envelope_default",
                "deployment_binding_hash": (
                    self.deployment_binding.semantic_content_hash
                ),
                "deployment_envelope_hash": entry.semantic_content_hash,
                "constraint_set_hash": entry.constraint_set_hash,
            },
            domain="dexteleop.lower_route_bound_safety_profile.v1",
        )
        return TCPPoseEnvelopeSelection(
            safety_profile_id="lower-route-bound.standard@1",
            safety_profile_hash=profile_hash,
            deployment_envelope_id=entry.envelope_id,
            deployment_envelope_hash=entry.semantic_content_hash,
            constraint_set_hash=entry.constraint_set_hash,
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
            # Full per-cycle facts remain in the append-only ControlTrace.
            # The Skill receipt carries only a bounded representative cut.
            refs.update(ordered_fact_refs[:48])
            request = getattr(authoritative, "target_execution_request", None)
            request_id = getattr(request, "request_id", None)
            if isinstance(request_id, str) and request_id.strip():
                refs.add(request_id)
        return tuple(sorted(refs))

    def _remember_step_evidence(self, step: Any) -> None:
        refs = self._step_refs(step)
        if not self._first_step_refs:
            self._first_step_refs = refs
        self._latest_step_refs = refs

    def _execution_evidence_refs(self) -> tuple[str, ...]:
        stable = tuple(sorted(self._execution_refs))
        if len(stable) > 128:
            raise RuntimeError("Stable TCP pose evidence exceeds protocol bounds.")
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
        refs = self._execution_evidence_refs()
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

        # A semantic goal may already hold at operation activation.  Confirm
        # that fact over the same bounded stability count used after dispatch,
        # then establish lower HOLD and verify the post-HOLD state.  Do not
        # inject an unnecessary command merely to manufacture a dispatch when
        # the requested physical effect is already true.
        for sample in range(self.motion_profile.stable_measurement_count):
            if bool(cancel_requested()):
                hold = self._hold(reason="tcp_pose_operation_cancelled")
                return self._failure(
                    reason_code="tcp_pose_operation_cancelled_after_hold",
                    hold=hold,
                )
            activation_pose = getattr(
                self,
                "_activation_measured_pose",
                None,
            )
            measured = (
                activation_pose
                if sample == 0 and activation_pose is not None
                else self._capture_pose()
            )
            if not self._within_terminal_tolerance(measured):
                stable = 0
                break
            stable += 1
            if sample + 1 < self.motion_profile.stable_measurement_count:
                self.sleep(period)
        if stable >= self.motion_profile.stable_measurement_count:
            hold = self._hold(reason="tcp_pose_target_already_satisfied")
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

        if self.action is None:
            raise RuntimeError("TCP pose lifecycle has no resolved action.")
        reference_cycle = 0
        for control_attempt in range(self.motion_profile.max_control_cycles):
            if bool(cancel_requested()):
                hold = self._hold(reason="tcp_pose_operation_cancelled")
                return self._failure(
                    reason_code="tcp_pose_operation_cancelled_after_hold",
                    hold=hold,
                )
            event_id = (
                f"{operation.operation_id}:target-attempt:{control_attempt}:"
                f"reference:{reference_cycle}"
            )
            step = self.application.submit_and_step(
                operation_id=operation.operation_id,
                action=self._reference_action_for_cycle(reference_cycle),
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
            self._remember_step_evidence(step)
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
            if authoritative is None:
                hold = self._hold(reason="tcp_pose_authoritative_step_failed")
                return self._failure(
                    reason_code="tcp_pose_authoritative_step_failed",
                    hold=hold,
                )
            if authoritative.status == "dispatched":
                reference_cycle += 1
            elif not _authoritative_step_keeps_existing_for_retry(authoritative):
                hold = self._hold(reason="tcp_pose_authoritative_step_failed")
                return self._failure(
                    reason_code="tcp_pose_authoritative_step_failed",
                    hold=hold,
                )
            measured = self._capture_pose()
            if self._measured_translation_excursion_exceeded(measured):
                hold = self._hold(
                    reason="tcp_pose_measured_translation_excursion_exceeded"
                )
                return self._failure(
                    reason_code=(
                        "tcp_pose_measured_translation_excursion_exceeded"
                    ),
                    hold=hold,
                )
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
        public_final_pose = self.application.transform_tcp_pose_xyzquat(
            final_pose,
            source_frame_id=self.motion_profile.execution_reference_frame_id,
            target_frame_id=self.motion_profile.reference_frame_id,
        )
        final_state_hash = canonical_content_hash(
            {
                "runtime_session_id": request.runtime_session_id,
                "control_domain_id": self.deployment_binding.control_domain_id,
                "operation_id": operation.operation_id,
                "component_id": self.component_id,
                "pose_xyz_quat_xyzw": public_final_pose,
                "measured_at": measured_at,
                "reference_frame_id": self.motion_profile.reference_frame_id,
                "controlled_frame_id": "tool_tcp",
            },
            domain="dexteleop.agent_skill.mujoco_tcp_terminal_state.v1",
        )
        if len(self.evaluation_plan.evaluator_specs) != 1:
            raise RuntimeError("TCP target evaluation requires one exact evaluator.")
        evaluator_identity = self.evaluation_plan.evaluator_specs[0]
        evaluator_spec = self.evaluation_plan.evaluator(
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


class MujocoAgentSafeTcpPoseSkillLifecycle(MujocoAgentTcpPoseSkillLifecycle):
    """Keep the normal lifecycle and receipts; replace only lower references."""

    def __init__(self, *, path_search_time_budget_sec: float = 3.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_search_time_budget_sec = path_search_time_budget_sec
        self._planner = self.application.target.planned_tcp_motion
        if self._planner is None:
            raise ValueError("Safe TCP lifecycle requires a lower joint-path planner.")
        self._path_completed = False
        self._lower_failure_code: str | None = None

    def _remember_step_evidence(self, step: Any) -> None:
        super()._remember_step_evidence(step)
        failure = getattr(step.authoritative_step, "resolution_failure", None)
        if failure is not None:
            self._lower_failure_code = failure.reason_code
            logging.getLogger(__name__).warning(
                "Safe TCP reference rejected: %s", failure
            )

    def _failure(self, *, reason_code, hold):
        if (
            reason_code == "tcp_pose_authoritative_step_failed"
            and self._lower_failure_code is not None
        ):
            reason_code = self._lower_failure_code
        return super()._failure(reason_code=reason_code, hold=hold)

    def activate(self, *, request, operation, cancel_requested):
        self._validate_call(request=request, operation=operation)
        if self.target_pose is None:
            raise ValueError("Safe TCP motion requires an absolute target.")
        measured = self._capture_pose()
        translation, rotation = _pose_error(measured, self.target_pose)
        if (
            cancel_requested()
            or translation > self.motion_profile.max_translation_delta_m
            or rotation > self.motion_profile.max_rotation_delta_rad
        ):
            # Preserve the ordinary cancellation/delta gate before planning.
            return super().activate(
                request=request, operation=operation, cancel_requested=cancel_requested
            )
        # Bind the exact admitted envelope before search, without submitting
        # a reference. Search and execution then consult the same lower owner.
        evidence = super().activate(
            request=request, operation=operation, cancel_requested=cancel_requested
        )
        if not evidence.succeeded:
            return evidence
        safety = self.application.tcp_pose_envelope_runtime
        constraint = None
        constraint_identity = None
        if safety is not None:
            initial = safety.project({"right": xyzquat_to_pose_matrix(measured)})
            basis = (initial.envelope_profile_hash, initial.anchor_hash)
            constraint_identity = canonical_content_hash(
                {"profile": basis[0], "anchor": basis[1]},
                domain="dexteleop.mujoco.planned_tcp_envelope.v1",
            )

            def constraint(model_qpos, pair):
                pose = self._planner.pose_for_configuration(model_qpos, pair)
                projection = safety.project({"right": xyzquat_to_pose_matrix(pose)})
                same_basis = (
                    projection.envelope_profile_hash,
                    projection.anchor_hash,
                ) == basis
                return JointConfigurationCheck(
                    safe=same_basis and not projection.modified_sides,
                    minimum_distance_m=None,
                    reason=(
                        None
                        if same_basis and not projection.modified_sides
                        else "tcp_envelope_blocked"
                        if same_basis
                        else "tcp_envelope_binding_changed"
                    ),
                    metadata=projection.to_payload(),
                )

        try:
            plan = self._planner.plan(
                self.target_pose,
                translation_tolerance_m=self.motion_profile.translation_tolerance_m,
                rotation_tolerance_rad=self.motion_profile.rotation_tolerance_rad,
                cancel_requested=cancel_requested,
                configuration_constraint=constraint,
                constraint_identity=constraint_identity,
                path_search_time_budget_sec=self._path_search_time_budget_sec,
            )
            if (
                len(plan.reference_poses) + self.motion_profile.stable_measurement_count
                >= self.motion_profile.max_control_cycles
            ):
                raise MujocoTcpPlanRejected("safe_tcp_path_exceeds_control_budget")
        except MujocoTcpPlanRejected as error:
            logging.getLogger(__name__).warning("Safe TCP planning rejected: %s", error)
            hold = self._hold(reason=error.reason_code)
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status=error.reason_code,
                lower_operation_id=self._lower_operation_id,
                evidence_refs=hold.evidence_refs,
                lower_hold_confirmed=hold.lower_hold_confirmed,
            )
        try:
            if safety is not None:
                for index, pose in enumerate(plan.reference_poses):
                    projection = safety.project({"right": xyzquat_to_pose_matrix(pose)})
                    if projection.modified_sides:
                        raise MujocoTcpPlanRejected(
                            "safe_tcp_path_safety_envelope_blocked",
                            {
                                "reference_index": index,
                                "projection": projection.to_payload(),
                            },
                        )
            self._planner.activate(plan)
        except MujocoTcpPlanRejected as error:
            logging.getLogger(__name__).warning(
                "Safe TCP activation rejected: %s", error
            )
            hold = self._hold(reason=error.reason_code)
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status=error.reason_code,
                lower_operation_id=self._lower_operation_id,
                evidence_refs=hold.evidence_refs,
                lower_hold_confirmed=hold.lower_hold_confirmed,
            )
        self._execution_refs.add("mujoco-safe-joint-trajectory:" + plan.trajectory_hash)
        return evidence

    def _reference_action_for_cycle(self, cycle: int) -> ComponentAction:
        pose = self._planner.reference(cycle)
        self._path_completed = self._planner.final_reference_selected
        return ComponentAction(
            action_schema=self.application.runtime.action_schema,
            values_by_component={self.component_id: np.asarray(pose, dtype=np.float32)},
        )

    def _within_terminal_tolerance(self, pose: tuple[float, ...]) -> bool:
        # Reuse the base lifecycle's stable-observation/HOLD completion when
        # the goal already held at activation. Otherwise require the entire
        # checked path, not just an intermediate pose near the goal.
        activation_pose = self._activation_measured_pose
        already_satisfied = (
            activation_pose is not None
            and super()._within_terminal_tolerance(activation_pose)
        )
        return (
            already_satisfied or self._path_completed
        ) and super()._within_terminal_tolerance(pose)


__all__ = [
    "MujocoAgentSafeTcpPoseSkillLifecycle",
    "MujocoAgentTcpPoseSkillLifecycle",
    "MujocoAgentTcpTargetResolution",
    "mujoco_agent_skill_hold_evidence_from_transition",
    "resolve_level_tcp_orientation_primitive_target",
    "resolve_tcp_displacement_primitive_target",
]
