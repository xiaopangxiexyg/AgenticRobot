"""Per-operation lifecycle for the bounded no-model TCP-pose Skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

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
from robot_motion.coordinate_frames import CoordinateFrameGraph, CoordinateFrameSpec
from robot_motion.primitives import (
    LevelTCPOrientationPrimitive,
    TCPPoseDisplacementPrimitive,
)
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)
from robot_motion.target_safety import (
    TCPPoseEnvelopeLifecycleError,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSelection,
)

from .motion_application import MujocoAgentSkillMotionApplication
from .motion_deployment import (
    MujocoAgentTcpPoseDeploymentBinding,
    MujocoAgentTcpPoseMotionProfile,
)


@dataclass(frozen=True)
class MujocoAgentTcpTargetResolution:
    """One lower-derived absolute target and its immutable semantic evidence."""

    target_pose: tuple[float, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_pose",
            _pose(self.target_pose, field_name="target_pose"),
        )
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("A derived TCP target requires semantic evidence.")
        object.__setattr__(self, "evidence_refs", refs)


def resolve_tcp_displacement_primitive_target(
    *,
    measured_pose: tuple[float, ...],
    primitive: TCPPoseDisplacementPrimitive,
    reference_frame_spec: CoordinateFrameSpec,
    frame_graph: CoordinateFrameGraph | None = None,
    measured_reference_frame_id: str | None = None,
) -> MujocoAgentTcpTargetResolution:
    """Resolve one pure primitive at the lower Skill activation boundary."""

    if not isinstance(primitive, TCPPoseDisplacementPrimitive):
        raise TypeError("primitive must be a TCP displacement primitive.")
    if not isinstance(reference_frame_spec, CoordinateFrameSpec):
        raise TypeError("reference_frame_spec must be typed.")
    measured_matrix = xyzquat_to_pose_matrix(
        measured_pose,
        name="tcp_displacement_activation_pose",
    )
    anchor, execution_from_primitive, frame_evidence = _resolve_primitive_frames(
        measured_pose=measured_matrix,
        primitive_reference_frame_spec=reference_frame_spec,
        frame_graph=frame_graph,
        measured_reference_frame_id=measured_reference_frame_id,
    )
    target = primitive.apply(
        anchor,
        reference_frame_spec=reference_frame_spec,
    )
    execution_target = execution_from_primitive @ target.target_pose
    return MujocoAgentTcpTargetResolution(
        target_pose=tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                execution_target,
                name="tcp_displacement_target_pose",
            )
        ),
        evidence_refs=(
            "tcp-displacement-primitive:" + primitive.semantic_content_hash,
            "tcp-displacement-target:" + target.semantic_content_hash,
            *frame_evidence,
        ),
    )


def resolve_level_tcp_orientation_primitive_target(
    *,
    measured_pose: tuple[float, ...],
    primitive: LevelTCPOrientationPrimitive,
    reference_frame_spec: CoordinateFrameSpec,
    frame_graph: CoordinateFrameGraph | None = None,
    measured_reference_frame_id: str | None = None,
) -> MujocoAgentTcpTargetResolution:
    """Resolve one fixed absolute horizontal heading at activation.

    The measured pose contributes translation only.  The primitive replaces
    orientation with its profile-owned absolute heading, so calls never add a
    relative yaw to the current TCP orientation.
    """

    if not isinstance(primitive, LevelTCPOrientationPrimitive):
        raise TypeError("primitive must be a fixed level TCP orientation.")
    if not isinstance(reference_frame_spec, CoordinateFrameSpec):
        raise TypeError("reference_frame_spec must be typed.")
    measured_matrix = xyzquat_to_pose_matrix(
        measured_pose,
        name="tcp_level_orientation_activation_pose",
    )
    anchor, execution_from_primitive, frame_evidence = _resolve_primitive_frames(
        measured_pose=measured_matrix,
        primitive_reference_frame_spec=reference_frame_spec,
        frame_graph=frame_graph,
        measured_reference_frame_id=measured_reference_frame_id,
    )
    target = primitive.apply(
        anchor,
        reference_frame_spec=reference_frame_spec,
    )
    execution_target = execution_from_primitive @ target.target_pose
    return MujocoAgentTcpTargetResolution(
        target_pose=tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                execution_target,
                name="tcp_level_orientation_target_pose",
            )
        ),
        evidence_refs=(
            "tcp-level-orientation-primitive:" + primitive.semantic_content_hash,
            "tcp-level-orientation-target:" + target.semantic_content_hash,
            *frame_evidence,
        ),
    )


def _resolve_primitive_frames(
    *,
    measured_pose: np.ndarray,
    primitive_reference_frame_spec: CoordinateFrameSpec,
    frame_graph: CoordinateFrameGraph | None,
    measured_reference_frame_id: str | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Move an execution-frame measurement into and back out of Skill axes.

    MuJoCo executes arm targets in ``*_arm_base`` coordinates, while semantic
    cardinal/orientation primitives are defined in the canonical station/desk
    axes.  Keeping the two frames explicit prevents a rotated arm base from
    silently changing ``forward`` or a fixed absolute heading.
    """

    primitive_frame_id = primitive_reference_frame_spec.frame_id
    execution_frame_id = (
        primitive_frame_id
        if measured_reference_frame_id is None
        else _required(
            measured_reference_frame_id,
            field_name="measured_reference_frame_id",
        )
    )
    if execution_frame_id == primitive_frame_id:
        return measured_pose, np.eye(4, dtype=np.float64), ()
    if not isinstance(frame_graph, CoordinateFrameGraph):
        raise TypeError(
            "frame_graph is required when primitive and measured frames differ."
        )
    registered_spec = frame_graph.frame_by_id.get(primitive_frame_id)
    if (
        registered_spec is None
        or registered_spec.semantic_content_hash
        != primitive_reference_frame_spec.semantic_content_hash
    ):
        raise ValueError(
            "primitive reference frame is not the exact frame-graph member."
        )
    primitive_from_execution = frame_graph.resolve(
        execution_frame_id,
        primitive_frame_id,
    )
    execution_from_primitive = frame_graph.resolve(
        primitive_frame_id,
        execution_frame_id,
    )
    return (
        primitive_from_execution.matrix_target_source @ measured_pose,
        execution_from_primitive.matrix_target_source,
        (
            "tcp-primitive-frame-resolution:"
            + primitive_from_execution.semantic_content_hash,
            "tcp-execution-frame-resolution:"
            + execution_from_primitive.semantic_content_hash,
        ),
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
        target_resolver: (
            Callable[[tuple[float, ...]], MujocoAgentTcpTargetResolution] | None
        ) = None,
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
        self.parameter_lineage = lineage
        self.target_pose: tuple[float, ...] | None = (
            None
            if target_resolver is not None
            else _pose(
                poses[component_id],
                field_name=f"pose_targets_by_component.{component_id}",
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
                "component_id": self.component_id,
                "pose_xyz_quat_xyzw": final_pose,
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


__all__ = [
    "MujocoAgentTcpPoseSkillLifecycle",
    "MujocoAgentTcpTargetResolution",
    "mujoco_agent_skill_hold_evidence_from_transition",
    "resolve_level_tcp_orientation_primitive_target",
    "resolve_tcp_displacement_primitive_target",
]
