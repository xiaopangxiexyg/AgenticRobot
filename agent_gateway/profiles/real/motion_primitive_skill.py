"""One-use real lifecycle for a reviewed TCP primitive operation."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext

from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution,
    resolve_level_tcp_orientation_primitive_target,
    resolve_tcp_displacement_primitive_target,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.protocol.skill_catalog import SkillExecutionRoute
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleExecutionCompletion,
)
from agent_gateway.skills.motion.definition import (
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
    RealAgentJointTrajectory,
    RealAgentMotionActivation,
    RealAgentMotionMeasuredQposCut,
    RealAgentMotionPhase,
    RealAgentMotionPort,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

from .grasp_approach_plan import (
    BoundedRealGraspApproachPlanStore,
    RealGraspApproachPlan,
    RealGraspApproachPlanLookupError,
    RealGraspApproachPlanProfile,
)
from .motion_primitive_profiles import (
    RealAgentPrimitiveContractBundle,
    RealAgentPrimitiveDeploymentBinding,
    RealAgentTcpCardinalStepProfile,
    RealAgentTcpFixedOrientationProfile,
    RealAgentTcpPoseTargetProfile,
)



def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


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


def _grant_hash(grant_id: str | None) -> tuple[str, str]:
    normalized = _required(grant_id, field_name="skill_execution_grant_id")
    prefix = "agent-skill-grant:"
    if not normalized.startswith(prefix) or len(normalized) == len(prefix):
        raise ValueError("Real motion requires a content-addressed Skill grant ID.")
    return normalized, normalized.removeprefix(prefix)


_LOGGER = logging.getLogger(__name__)


class RealAgentPrimitiveSkillLifecycle:
    """Resolve, execute, evaluate, and HOLD one lower real operation."""

    def __init__(
        self,
        *,
        port: RealAgentMotionPort,
        contract_bundle: RealAgentPrimitiveContractBundle,
        route: SkillExecutionRoute,
        binding: RealAgentPrimitiveDeploymentBinding,
        profile: RealAgentTcpCardinalStepProfile
        | RealAgentTcpFixedOrientationProfile
        | RealAgentTcpPoseTargetProfile
        | RealGraspApproachPlanProfile,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        grasp_trajectory_planner: Callable[..., object] | None = None,
        tcp_trajectory_planner: Callable[..., object] | None = None,
        non_motion_read_window: Callable[[], object] | None = None,
        grasp_approach_plan_store: BoundedRealGraspApproachPlanStore | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(port, RealAgentMotionPort):
            raise TypeError("port must be RealAgentMotionPort.")
        if not isinstance(contract_bundle, RealAgentPrimitiveContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not isinstance(route, SkillExecutionRoute):
            raise TypeError("route must be typed.")
        if not isinstance(binding, RealAgentPrimitiveDeploymentBinding):
            raise TypeError("binding must be typed.")
        if not isinstance(
            profile,
            (
                RealAgentTcpCardinalStepProfile,
                RealAgentTcpFixedOrientationProfile,
                RealAgentTcpPoseTargetProfile,
                RealGraspApproachPlanProfile,
            ),
        ):
            raise TypeError("profile must be a real primitive profile.")
        if not callable(clock) or not callable(sleep):
            raise TypeError("clock and sleep must be callable.")
        if route.skill_id in {
            GRASP_APPROACH_SKILL_ID_V1, SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
        } and not callable(grasp_trajectory_planner):
            raise TypeError("Planned pose Route requires its URDF/FCL planner.")
        if route.skill_id == EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1 and not isinstance(
            grasp_approach_plan_store,
            BoundedRealGraspApproachPlanStore,
        ):
            raise TypeError(
                "Plan execution Route requires its session-local plan store."
            )
        checks = (
            (request.request_id, "request_id"),
            (operation.operation_id, "operation_id"),
        )
        for value, field_name in checks:
            _required(value, field_name=field_name)
        if request.semantic_content_hash != operation.request_hash:
            raise ValueError("Real primitive request/operation hash mismatch.")
        if request.route_hash != route.semantic_content_hash:
            raise ValueError("Real primitive request Route mismatch.")
        if binding.route_hash != route.semantic_content_hash:
            raise ValueError("Real primitive deployment Route mismatch.")
        if request.deployment_binding_hash != binding.semantic_content_hash:
            raise ValueError("Real primitive request deployment mismatch.")
        if request.parameter_profile_hash != profile.semantic_content_hash:
            raise ValueError("Real primitive request profile mismatch.")
        if binding.parameter_profile_hash != profile.semantic_content_hash:
            raise ValueError("Real primitive deployment profile mismatch.")
        if request.runtime_session_id != binding.runtime_session_id:
            raise ValueError("Real primitive runtime session mismatch.")
        if profile.component_id != binding.component_id or profile.side != binding.side:
            raise ValueError("Real primitive component/side binding mismatch.")
        if isinstance(profile, RealAgentTcpPoseTargetProfile):
            targets = request.parameters.get("pose_targets_by_component")
            if not isinstance(targets, Mapping) or set(targets) != {
                profile.component_id
            }:
                raise ValueError("Real absolute-pose request component mismatch.")
        elif request.parameters.get("component_id") != profile.component_id:
            raise ValueError("Real primitive request component mismatch.")

        self.executor_id = binding.executor_id
        self.port = port
        self.contract_bundle = contract_bundle
        self.route = route
        self.binding = binding
        self.profile = profile
        self.request = request
        self.operation = operation
        self.grasp_trajectory_planner = grasp_trajectory_planner
        self.tcp_trajectory_planner = tcp_trajectory_planner
        self.non_motion_read_window = non_motion_read_window or nullcontext
        self.grasp_approach_plan_store = grasp_approach_plan_store
        self.clock = clock
        self.sleep = sleep
        self.lower_operation_id = "real-agent-motion:" + operation.operation_id
        self.target_resolution: MotionPrimitiveTargetResolution | None = None
        self.target_pose: tuple[float, ...] | None = None
        self._planned_joint_trajectory: RealAgentJointTrajectory | None = None
        self._reserved_grasp_approach_plan: RealGraspApproachPlan | None = None
        self._closed = False

    def _validate_call(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> None:
        if request.semantic_content_hash != self.request.semantic_content_hash:
            raise ValueError("Real primitive request changed.")
        if operation.operation_id != self.operation.operation_id:
            raise ValueError("Real primitive operation changed.")

    def _latest_measured(self, *, wait_for_fresh: bool = False):
        cut = self.port.latest_measured_pose()
        now = float(self.clock())
        if cut is None:
            if wait_for_fresh:
                return None
            raise RuntimeError("No lower measured TCP-pose cut is available.")
        if self.profile.component_id not in cut.pose_world_by_component:
            raise RuntimeError("Measured TCP-pose cut omits the selected component.")
        if cut.measured_at > now or cut.available_at > now:
            if wait_for_fresh:
                return None
            raise RuntimeError("Lower measured TCP-pose cut is from the future.")
        age_ms = (float(now) - cut.measured_at) * 1_000.0
        if age_ms > self.contract_bundle.convergence_profile.max_measurement_age_ms:
            if wait_for_fresh:
                return None
            raise RuntimeError("Lower measured TCP-pose cut is stale.")
        return cut

    def _measured_qpos_problem(
        self,
        *,
        cut: RealAgentMotionMeasuredQposCut | None,
        now: float,
    ) -> str | None:
        if cut is None:
            return "unavailable"
        if cut.measured_at > now or cut.available_at > now:
            return "from_future"
        age_ms = (now - cut.measured_at) * 1_000.0
        if age_ms > self.contract_bundle.convergence_profile.max_measurement_age_ms:
            return "stale"
        return None

    def _resolve_target(
        self, measured_pose: tuple[float, ...]
    ) -> tuple[
        MotionPrimitiveTargetResolution,
        str,
        str,
    ]:
        if isinstance(self.profile, RealGraspApproachPlanProfile):
            plan = self._reserved_grasp_approach_plan
            if plan is None:
                raise RuntimeError("Grasp-approach plan was not reserved.")
            resolution = MotionPrimitiveTargetResolution(
                target_pose=tuple(
                    float(value)
                    for value in pose_matrix_to_xyzquat(
                        plan.target_pose_execution,
                        name="real_agent_grasp_approach_plan_target",
                    )
                ),
                evidence_refs=plan.all_evidence_refs,
            )
            return (
                resolution,
                EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1 + "@1",
                plan.semantic_content_hash,
            )
        if isinstance(self.profile, RealAgentTcpPoseTargetProfile):
            parameters = self.request.parameters
            targets = parameters.get("pose_targets_by_component")
            assert isinstance(targets, Mapping)
            public_pose = xyzquat_to_pose_matrix(
                targets[self.profile.component_id],
                name="real_agent_absolute_tool_tcp_target",
            )
            execution_pose, frame_resolution = (
                self.contract_bundle.frame_graph.transform_pose_matrix(
                    public_pose,
                    source_reference_frame_id=self.profile.reference_frame_id,
                    target_reference_frame_id=(
                        self.profile.execution_reference_frame_id
                    ),
                )
            )
            target_hash = canonical_content_hash(
                {
                    "profile_hash": self.profile.semantic_content_hash,
                    "component_id": self.profile.component_id,
                    "public_target_pose": tuple(
                        float(value)
                        for value in pose_matrix_to_xyzquat(
                            public_pose,
                            name="real_agent_absolute_public_target",
                        )
                    ),
                    "frame_resolution_hash": frame_resolution.semantic_content_hash,
                },
                domain="dexteleop.real_agent_motion.absolute_tcp_target.v1",
            )
            resolution = MotionPrimitiveTargetResolution(
                target_pose=tuple(
                    float(value)
                    for value in pose_matrix_to_xyzquat(
                        execution_pose,
                        name="real_agent_absolute_execution_target",
                    )
                ),
                evidence_refs=(
                    "tcp-pose-target:" + target_hash,
                    "tcp-pose-target-frame-resolution:"
                    + frame_resolution.semantic_content_hash,
                ),
            )
            return resolution, self.route.skill_id + "@1", target_hash
        reference_frame = self.contract_bundle.frame_graph.frame_by_id[
            self.profile.reference_frame_id
        ]
        if isinstance(self.profile, RealAgentTcpCardinalStepProfile):
            parameters = self.request.parameters
            if not isinstance(parameters, Mapping):
                raise TypeError("Cardinal parameters must be an object.")
            primitive = self.profile.primitive(str(parameters.get("direction")))
            resolution = resolve_tcp_displacement_primitive_target(
                measured_pose=measured_pose,
                primitive=primitive,
                reference_frame_spec=reference_frame,
                frame_graph=self.contract_bundle.frame_graph,
                measured_reference_frame_id=(self.profile.execution_reference_frame_id),
            )
            primitive_id = (
                "motion.step_tcp_cardinal." + primitive.direction.value + "@1"
            )
            return resolution, primitive_id, primitive.semantic_content_hash
        resolution = resolve_level_tcp_orientation_primitive_target(
            measured_pose=measured_pose,
            primitive=self.profile.primitive,
            reference_frame_spec=reference_frame,
            frame_graph=self.contract_bundle.frame_graph,
            measured_reference_frame_id=self.profile.execution_reference_frame_id,
        )
        return (
            resolution,
            self.profile.skill_id + "@1",
            self.profile.primitive.semantic_content_hash,
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
                reason="cancelled_before_real_activation",
            )
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="real_primitive_cancelled_before_activation",
                lower_operation_id=self.lower_operation_id,
                evidence_refs=hold.evidence_refs,
                lower_hold_confirmed=hold.lower_hold_confirmed,
            )
        cut = self._latest_measured()
        now = float(self.clock())
        measured_pose = tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                cut.pose_world_by_component[self.profile.component_id],
                name="real_agent_activation_measured_pose",
            )
        )
        if self.route.skill_id == EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1:
            assert self.grasp_approach_plan_store is not None
            try:
                self._reserved_grasp_approach_plan = (
                    self.grasp_approach_plan_store.reserve(
                        plan_id=str(self.request.parameters["grasp_approach_plan_id"]),
                        plan_hash=str(
                            self.request.parameters["grasp_approach_plan_hash"]
                        ),
                        operation_id=operation.operation_id,
                        runtime_session_id=self.request.runtime_session_id,
                        profile_hash=self.profile.semantic_content_hash,
                        component_id=self.profile.component_id,
                        expected_station_revision=(
                            self.request.expected_station_revision
                        ),
                        now=now,
                    )
                )
                # The first admitted execution attempt owns the immutable plan.
                # Consume it before any later freshness/deadline checks so a
                # failed or cancelled motion attempt cannot replay the same path.
                self.grasp_approach_plan_store.consume(
                    plan_id=self._reserved_grasp_approach_plan.plan_id,
                    operation_id=operation.operation_id,
                )
            except RealGraspApproachPlanLookupError as exc:
                hold = self.request_hold(
                    operation=operation,
                    reason=exc.reason_code,
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status=exc.reason_code,
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted({*cut.evidence_refs, *hold.evidence_refs})
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
        resolution, primitive_id, primitive_hash = self._resolve_target(measured_pose)
        joint_trajectory = (
            self._reserved_grasp_approach_plan.trajectory
            if self._reserved_grasp_approach_plan is not None
            else None
        )
        planning_evidence: tuple[str, ...] = (
            self._reserved_grasp_approach_plan.all_evidence_refs
            if self._reserved_grasp_approach_plan is not None
            else ()
        )
        planned_tcp = (self.route.skill_id == TCP_POSE_TARGET_SKILL_ID_V1
                       and self.tcp_trajectory_planner is not None)
        if self.route.skill_id in {
            GRASP_APPROACH_SKILL_ID_V1, SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
        } or planned_tcp:
            qpos_cut = self.port.latest_measured_qpos()
            now = float(self.clock())
            qpos_problem = self._measured_qpos_problem(cut=qpos_cut, now=now)
            if qpos_problem is not None:
                reason = f"grasp_path_measured_qpos_{qpos_problem}"
                hold = self.request_hold(
                    operation=operation,
                    reason=reason,
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status=reason,
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *(() if qpos_cut is None else qpos_cut.evidence_refs),
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            assert qpos_cut is not None
            planner = self.tcp_trajectory_planner if planned_tcp else self.grasp_trajectory_planner
            assert planner is not None
            try:
                # No motion is active yet. Reuse the lower confirmed-HOLD
                # read window; planning never runs inside the control cycle.
                with self.non_motion_read_window() if planned_tcp else nullcontext():
                    joint_trajectory = planner(
                        qpos_by_side=qpos_cut.qpos_by_side,
                        side=self.profile.side,
                        target_tool_tcp_world=xyzquat_to_pose_matrix(
                            resolution.target_pose, name="real_agent_planned_tcp_target"),
                        speed_scale=float(self.request.parameters.get("speed_scale", 1.0)),
                        **({"time_budget_sec": operation.deadline_at - float(self.clock())}
                           if planned_tcp or self.route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1
                           else {}),
                        **({"require_grasp_approach": False}
                           if self.route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1 else {}),
                    )
            except RM65GraspTrajectoryPlanningError as exc:
                # Preserve the lower rejection geometry without changing the
                # versioned operation schema or weakening its HOLD decision.
                _LOGGER.error(
                    "Real TCP planning rejected: operation_id=%s skill_id=%s "
                    "side=%s reason_code=%s details=%r",
                    operation.operation_id,
                    self.route.skill_id,
                    self.profile.side,
                    exc.reason_code,
                    exc.details,
                )
                hold = self.request_hold(
                    operation=operation,
                    reason=exc.reason_code,
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status=exc.reason_code,
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            planning_evidence = joint_trajectory.evidence_refs
            if cancel_requested():
                hold = self.request_hold(
                    operation=operation,
                    reason="cancelled_after_grasp_planning_before_activation",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="real_primitive_cancelled_before_activation",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            latest_qpos = self.port.latest_measured_qpos()
            now = float(self.clock())
            postplan_started_at = now
            postplan_initial_qpos = latest_qpos
            refresh_poll_count = 0
            if planned_tcp or self.route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1:
                # Both TCP planners can pause the lower sampler. Window exit
                # refreshes it, but the control loop publishes to this port
                # asynchronously. Never activate from the pre-window cut.
                refresh_deadline = min(now + 0.5, operation.deadline_at)
                while (self._measured_qpos_problem(cut=latest_qpos, now=now) != "from_future"
                       and (latest_qpos is None
                       or latest_qpos.measured_at <= qpos_cut.measured_at
                       or self._measured_qpos_problem(cut=latest_qpos, now=now))):
                    if cancel_requested() or now >= refresh_deadline:
                        break
                    self.sleep(min(0.01, refresh_deadline - now))
                    refresh_poll_count += 1
                    latest_qpos = self.port.latest_measured_qpos()
                    now = float(self.clock())
                if cancel_requested():
                    hold = self.request_hold(operation=operation, reason="cancelled_after_tcp_planning")
                    return StationLifecycleActivationEvidence(
                        succeeded=False, status="real_primitive_cancelled_before_activation",
                        lower_operation_id=self.lower_operation_id,
                        evidence_refs=hold.evidence_refs,
                        lower_hold_confirmed=hold.lower_hold_confirmed)
            postplan_qpos_problem = self._measured_qpos_problem(
                cut=latest_qpos,
                now=now,
            )
            if (postplan_qpos_problem is None
                    and (planned_tcp or self.route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1)
                    and latest_qpos.measured_at <= qpos_cut.measured_at):
                # Freshness does not prove publication after the planning window.
                postplan_qpos_problem = "not_refreshed"
            postplan_diagnostic = {
                "planning_start_measured_at": qpos_cut.measured_at,
                "initial_measured_at": None if postplan_initial_qpos is None else postplan_initial_qpos.measured_at,
                "initial_age_ms": None if postplan_initial_qpos is None else
                    (postplan_started_at - postplan_initial_qpos.measured_at) * 1000,
                "final_measured_at": None if latest_qpos is None else latest_qpos.measured_at,
                "final_available_at": None if latest_qpos is None else latest_qpos.available_at,
                "final_age_ms": None if latest_qpos is None else (now - latest_qpos.measured_at) * 1000,
                "maximum_age_ms": self.contract_bundle.convergence_profile.max_measurement_age_ms,
                "checked_at": now, "wait_sec": now - postplan_started_at,
                "poll_count": refresh_poll_count, "problem": postplan_qpos_problem,
            }
            _LOGGER.log(logging.ERROR if postplan_qpos_problem else logging.INFO,
                "Real TCP postplan_qpos: operation_id=%s skill_id=%s side=%s diagnostic=%r",
                operation.operation_id, self.route.skill_id, self.profile.side, postplan_diagnostic)
            if postplan_qpos_problem is not None:
                reason = "grasp_path_postplan_measured_qpos_" + postplan_qpos_problem
                hold = self.request_hold(operation=operation, reason=reason)
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status=reason,
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *(
                                    ()
                                    if latest_qpos is None
                                    else latest_qpos.evidence_refs
                                ),
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            assert latest_qpos is not None
            if any(
                float(
                    abs(
                        latest_qpos.qpos_by_side[side]
                        - joint_trajectory.qpos_by_side[side][0]
                    ).max()
                )
                > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
                for side in ("left", "right")
            ):
                hold = self.request_hold(
                    operation=operation,
                    reason="grasp_path_start_state_drift",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="grasp_path_start_state_drift",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *latest_qpos.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            if (
                now + float(joint_trajectory.time_from_start_sec[-1])
                >= operation.deadline_at
            ):
                hold = self.request_hold(
                    operation=operation,
                    reason="grasp_path_exceeds_operation_deadline",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="grasp_path_exceeds_operation_deadline",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
        if self.route.skill_id == EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1:
            plan = self._reserved_grasp_approach_plan
            assert plan is not None
            assert joint_trajectory is not None
            qpos_cut = self.port.latest_measured_qpos()
            now = float(self.clock())
            qpos_problem = self._measured_qpos_problem(cut=qpos_cut, now=now)
            if qpos_problem is not None:
                reason = "grasp_path_execution_measured_qpos_" + qpos_problem
                hold = self.request_hold(operation=operation, reason=reason)
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status=reason,
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *(() if qpos_cut is None else qpos_cut.evidence_refs),
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            assert qpos_cut is not None
            if cancel_requested():
                hold = self.request_hold(
                    operation=operation,
                    reason="cancelled_before_grasp_plan_execution",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="real_primitive_cancelled_before_activation",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            if now >= plan.expires_at:
                hold = self.request_hold(
                    operation=operation,
                    reason="grasp_path_plan_expired",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="grasp_path_plan_expired",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            if any(
                float(
                    abs(
                        qpos_cut.qpos_by_side[side]
                        - joint_trajectory.qpos_by_side[side][0]
                    ).max()
                )
                > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
                for side in ("left", "right")
            ):
                hold = self.request_hold(
                    operation=operation,
                    reason="grasp_path_start_state_drift",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="grasp_path_start_state_drift",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            if now + float(joint_trajectory.time_from_start_sec[-1]) >= (
                operation.deadline_at
            ):
                hold = self.request_hold(
                    operation=operation,
                    reason="grasp_path_exceeds_operation_deadline",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="grasp_path_exceeds_operation_deadline",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *cut.evidence_refs,
                                *qpos_cut.evidence_refs,
                                *planning_evidence,
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
        grant_id, grant_hash = _grant_hash(request.skill_execution_grant_id)
        activation = RealAgentMotionActivation(
            operation_id=operation.operation_id,
            request_id=request.request_id,
            activation_grant_id=grant_id,
            activation_grant_hash=grant_hash,
            component_id=self.profile.component_id,
            side=self.profile.side,
            primitive_id=primitive_id,
            primitive_hash=primitive_hash,
            target_pose_world=xyzquat_to_pose_matrix(
                resolution.target_pose,
                name="real_agent_primitive_target",
            ),
            execution_frame_id=self.profile.execution_reference_frame_id,
            execution_frame_spec_hash=(
                self.profile.execution_reference_frame_spec_hash
            ),
            measured_state_revision=cut.revision,
            measured_state_hash=cut.semantic_content_hash,
            submitted_at=now,
            expires_at=operation.deadline_at,
            speed_scale=(
                float(self.request.parameters.get("speed_scale", 1.0))
                if isinstance(self.profile, RealAgentTcpPoseTargetProfile)
                else (
                    self._reserved_grasp_approach_plan.speed_scale
                    if self._reserved_grasp_approach_plan is not None
                    else 1.0
                )
            ),
            joint_trajectory=joint_trajectory,
        )
        snapshot = self.port.submit_activation(activation)
        self._planned_joint_trajectory = joint_trajectory
        self.target_resolution = resolution
        self.target_pose = resolution.target_pose
        activation_evidence = {
            *cut.evidence_refs,
            *resolution.evidence_refs,
            *planning_evidence,
        }
        latest_snapshot_ref = snapshot.evidence_ref

        def current_activation_evidence() -> set[str]:
            return {
                *activation_evidence,
                latest_snapshot_ref,
            }

        while float(self.clock()) < operation.deadline_at:
            if cancel_requested():
                hold = self.request_hold(
                    operation=operation,
                    reason="real_activation_preempted",
                )
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="real_primitive_activation_preempted",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(
                        sorted(
                            {
                                *current_activation_evidence(),
                                *hold.evidence_refs,
                            }
                        )
                    ),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            snapshot = self.port.snapshot()
            latest_snapshot_ref = snapshot.evidence_ref
            if (
                snapshot.operation_id == operation.operation_id
                and snapshot.phase == RealAgentMotionPhase.ACTIVE
            ):
                return StationLifecycleActivationEvidence(
                    succeeded=True,
                    status="real_primitive_lower_owner_activated",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(sorted(current_activation_evidence())),
                    lower_hold_confirmed=False,
                )
            if snapshot.phase in {
                RealAgentMotionPhase.FAULTED,
                RealAgentMotionPhase.HELD,
            }:
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status="real_primitive_activation_failed_closed",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(sorted(current_activation_evidence())),
                    lower_hold_confirmed=bool(
                        snapshot.phase == RealAgentMotionPhase.HELD
                        and snapshot.hold_succeeded
                    ),
                )
            self.sleep(self.contract_bundle.convergence_profile.poll_interval_sec)
        hold = self.request_hold(
            operation=operation,
            reason="real_activation_deadline_exceeded",
        )
        return StationLifecycleActivationEvidence(
            succeeded=False,
            status="real_primitive_activation_deadline_exceeded",
            lower_operation_id=self.lower_operation_id,
            evidence_refs=tuple(
                sorted(
                    {
                        *current_activation_evidence(),
                        *hold.evidence_refs,
                    }
                )
            ),
            lower_hold_confirmed=hold.lower_hold_confirmed,
        )

    def _control_evidence_from_snapshot(
        self,
        snapshot,
        *,
        status: str,
    ) -> StationLifecycleControlEvidence:
        confirmed = bool(
            snapshot.phase == RealAgentMotionPhase.HELD
            and snapshot.hold_succeeded is True
        )
        return StationLifecycleControlEvidence(
            succeeded=confirmed,
            status=status if confirmed else status + "_unconfirmed",
            lower_operation_id=self.lower_operation_id,
            evidence_refs=(snapshot.evidence_ref,),
            lower_hold_confirmed=confirmed,
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        normalized_reason = _required(reason, field_name="reason")
        if (
            operation is not None
            and operation.operation_id != self.operation.operation_id
        ):
            raise ValueError("Real primitive HOLD operation mismatch.")
        snapshot = self.port.snapshot()
        if snapshot.phase == RealAgentMotionPhase.HELD:
            return self._control_evidence_from_snapshot(
                snapshot,
                status="real_primitive_lower_hold_confirmed",
            )
        if snapshot.operation_id == self.operation.operation_id and snapshot.phase in {
            RealAgentMotionPhase.PENDING_ACTIVATION,
            RealAgentMotionPhase.ACTIVE,
            RealAgentMotionPhase.FAULTED,
        }:
            snapshot = self.port.request_hold(
                operation_id=self.operation.operation_id,
                reason=normalized_reason,
                requested_at=float(self.clock()),
            )
        elif snapshot.phase not in {
            RealAgentMotionPhase.HOLD_REQUESTED,
            RealAgentMotionPhase.HOLD_APPLYING,
        }:
            snapshot = self.port.request_domain_hold(
                reason=normalized_reason,
                requested_at=float(self.clock()),
            )
        # The operation deadline bounds motion execution, not the lower
        # control-plane proof that the requested HOLD has been applied.  Once
        # the deadline has elapsed, the control owner still needs a bounded
        # grace window to run its next cycle and produce HELD evidence.
        hold_deadline = float(self.clock()) + (
            self.contract_bundle.convergence_profile.hold_confirmation_timeout_sec
        )
        while float(self.clock()) < hold_deadline:
            snapshot = self.port.snapshot()
            if snapshot.phase == RealAgentMotionPhase.HELD:
                return self._control_evidence_from_snapshot(
                    snapshot,
                    status="real_primitive_lower_hold_confirmed",
                )
            if (
                snapshot.phase == RealAgentMotionPhase.FAULTED
                and snapshot.hold_succeeded is False
            ):
                break
            self.sleep(self.contract_bundle.convergence_profile.poll_interval_sec)
        return self._control_evidence_from_snapshot(
            self.port.snapshot(),
            status="real_primitive_lower_hold_deadline",
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

    def _failed_execution(
        self,
        *,
        reason_code: str,
        evidence_refs: set[str],
    ) -> StationLifecycleExecutionCompletion:
        hold = self.request_hold(operation=self.operation, reason=reason_code)
        return StationLifecycleExecutionCompletion(
            succeeded=False,
            reason_code=reason_code,
            execution_receipt_refs=tuple(sorted({*evidence_refs, *hold.evidence_refs})),
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
        if self.target_pose is None or self.target_resolution is None:
            return self._failed_execution(
                reason_code="real_primitive_target_not_resolved",
                evidence_refs={self.port.snapshot().evidence_ref},
            )
        base_evidence = set(self.target_resolution.evidence_refs)
        if self._planned_joint_trajectory is not None:
            base_evidence.update(self._planned_joint_trajectory.evidence_refs)
        latest_snapshot_ref: str | None = None
        latest_cut_evidence: tuple[str, ...] = ()

        def current_evidence() -> set[str]:
            return {
                *base_evidence,
                *(() if latest_snapshot_ref is None else (latest_snapshot_ref,)),
                *latest_cut_evidence,
            }

        stable_count = 0
        last_revision: int | None = None
        latest_cut = None
        while float(self.clock()) < operation.deadline_at:
            if cancel_requested():
                return self._failed_execution(
                    reason_code="real_primitive_cancelled",
                    evidence_refs=current_evidence(),
                )
            snapshot = self.port.snapshot()
            latest_snapshot_ref = snapshot.evidence_ref
            if snapshot.phase == RealAgentMotionPhase.FAULTED:
                return self._failed_execution(
                    reason_code="real_primitive_lower_fault",
                    evidence_refs=current_evidence(),
                )
            if snapshot.phase != RealAgentMotionPhase.ACTIVE:
                lower_reason = (snapshot.hold_reason or "").split(":", 1)[0]
                _LOGGER.error(
                    "Real primitive lower operation stopped: operation_id=%s phase=%s hold_reason=%s",
                    operation.operation_id, snapshot.phase.value, snapshot.hold_reason,
                )
                return self._failed_execution(
                    reason_code={
                        "lower_dispatch_failed": "real_primitive_lower_dispatch_failed",
                        "lower_backend_not_ready": "real_primitive_lower_backend_not_ready",
                        "exclusive_skill_control_cycle_failed": "real_primitive_lower_control_cycle_failed",
                    }.get(lower_reason, "real_primitive_lower_not_active"),
                    evidence_refs=current_evidence(),
                )
            cut = self._latest_measured(wait_for_fresh=True)
            if cut is None:
                # A derived TCP cut can temporarily lag while the sole lower
                # control owner finishes an expensive IK/collision cycle.  The
                # lower measurement sampler/control loop owns the 200 ms source
                # freshness fault and HOLD; this evaluator must not race it by
                # terminating a healthy operation from an older derived cut.
                self.sleep(self.contract_bundle.convergence_profile.poll_interval_sec)
                continue
            if cut.revision != last_revision:
                latest_cut = cut
                last_revision = cut.revision
                latest_cut_evidence = cut.evidence_refs
                measured_pose = tuple(
                    float(value)
                    for value in pose_matrix_to_xyzquat(
                        cut.pose_world_by_component[self.profile.component_id],
                        name="real_agent_terminal_measured_pose",
                    )
                )
                translation, rotation = _pose_error(measured_pose, self.target_pose)
                planned_trajectory_elapsed = True
                if self._planned_joint_trajectory is not None or self.route.skill_id in {
                    GRASP_APPROACH_SKILL_ID_V1,
                    SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
                    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
                }:
                    trajectory = self._planned_joint_trajectory
                    planned_trajectory_elapsed = bool(
                        trajectory is not None
                        and snapshot.trajectory_elapsed_sec is not None
                        and snapshot.trajectory_elapsed_sec
                        >= float(trajectory.time_from_start_sec[-1]) - 1e-9
                    )
                if (
                    snapshot.dispatch_count > 0
                    and planned_trajectory_elapsed
                    and translation
                    <= self.contract_bundle.convergence_profile.translation_tolerance_m
                    and rotation
                    <= self.contract_bundle.convergence_profile.rotation_tolerance_rad
                ):
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= (
                    self.contract_bundle.convergence_profile.stable_measurement_count
                ):
                    hold = self.request_hold(
                        operation=operation,
                        reason="real_primitive_terminal_target_reached",
                    )
                    evidence = current_evidence()
                    evidence.update(hold.evidence_refs)
                    if not hold.lower_hold_confirmed:
                        return StationLifecycleExecutionCompletion(
                            succeeded=False,
                            reason_code="real_primitive_terminal_hold_unconfirmed",
                            execution_receipt_refs=tuple(sorted(evidence)),
                            terminal_hold_confirmed=False,
                        )
                    assert latest_cut is not None
                    refs = tuple(sorted(evidence))
                    return StationLifecycleExecutionCompletion(
                        succeeded=True,
                        reason_code="real_primitive_completed_after_lower_hold",
                        execution_receipt_refs=refs,
                        terminal_hold_confirmed=True,
                        execution_result={
                            "execution_receipt_refs": list(refs),
                            "final_measured_state_hash": (
                                latest_cut.semantic_content_hash
                            ),
                        },
                    )
            self.sleep(self.contract_bundle.convergence_profile.poll_interval_sec)
        return self._failed_execution(
            reason_code="real_primitive_deadline_exceeded",
            evidence_refs=current_evidence(),
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["RealAgentPrimitiveSkillLifecycle"]
