"""One-use real lifecycle for a reviewed TCP primitive operation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping

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
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentMotionActivation,
    RealAgentMotionPhase,
    RealAgentMotionPort,
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
        | RealAgentTcpPoseTargetProfile,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
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
            ),
        ):
            raise TypeError("profile must be a real primitive profile.")
        if not callable(clock) or not callable(sleep):
            raise TypeError("clock and sleep must be callable.")
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
        self.clock = clock
        self.sleep = sleep
        self.lower_operation_id = "real-agent-motion:" + operation.operation_id
        self.target_resolution: MotionPrimitiveTargetResolution | None = None
        self.target_pose: tuple[float, ...] | None = None
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

    def _latest_measured(self, *, now: float):
        cut = self.port.latest_measured_pose()
        if cut is None:
            raise RuntimeError("No lower measured TCP-pose cut is available.")
        if self.profile.component_id not in cut.pose_world_by_component:
            raise RuntimeError("Measured TCP-pose cut omits the selected component.")
        age_ms = max(0.0, (float(now) - cut.measured_at) * 1_000.0)
        if age_ms > self.contract_bundle.convergence_profile.max_measurement_age_ms:
            raise RuntimeError("Lower measured TCP-pose cut is stale.")
        return cut

    def _resolve_target(self, measured_pose: tuple[float, ...]) -> tuple[
        MotionPrimitiveTargetResolution,
        str,
        str,
    ]:
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
            return resolution, self.profile.skill_id + "@1", target_hash
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
                measured_reference_frame_id=(
                    self.profile.execution_reference_frame_id
                ),
            )
            primitive_id = (
                "motion.step_tcp_cardinal."
                + primitive.direction.value
                + "@1"
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
        now = float(self.clock())
        cut = self._latest_measured(now=now)
        measured_pose = tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                cut.pose_world_by_component[self.profile.component_id],
                name="real_agent_activation_measured_pose",
            )
        )
        resolution, primitive_id, primitive_hash = self._resolve_target(measured_pose)
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
                else 1.0
            ),
        )
        snapshot = self.port.submit_activation(activation)
        self.target_resolution = resolution
        self.target_pose = resolution.target_pose
        evidence = {
            *cut.evidence_refs,
            *resolution.evidence_refs,
            snapshot.evidence_ref,
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
                    evidence_refs=tuple(sorted({*evidence, *hold.evidence_refs})),
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            snapshot = self.port.snapshot()
            evidence.add(snapshot.evidence_ref)
            if (
                snapshot.operation_id == operation.operation_id
                and snapshot.phase == RealAgentMotionPhase.ACTIVE
            ):
                return StationLifecycleActivationEvidence(
                    succeeded=True,
                    status="real_primitive_lower_owner_activated",
                    lower_operation_id=self.lower_operation_id,
                    evidence_refs=tuple(sorted(evidence)),
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
                    evidence_refs=tuple(sorted(evidence)),
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
            evidence_refs=tuple(sorted({*evidence, *hold.evidence_refs})),
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
        if operation is not None and operation.operation_id != self.operation.operation_id:
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
            execution_receipt_refs=tuple(
                sorted({*evidence_refs, *hold.evidence_refs})
            ),
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
        evidence = set(self.target_resolution.evidence_refs)
        stable_count = 0
        last_revision: int | None = None
        latest_cut = None
        while float(self.clock()) < operation.deadline_at:
            if cancel_requested():
                return self._failed_execution(
                    reason_code="real_primitive_cancelled",
                    evidence_refs=evidence,
                )
            snapshot = self.port.snapshot()
            evidence.add(snapshot.evidence_ref)
            if snapshot.phase == RealAgentMotionPhase.FAULTED:
                return self._failed_execution(
                    reason_code="real_primitive_lower_fault",
                    evidence_refs=evidence,
                )
            if snapshot.phase != RealAgentMotionPhase.ACTIVE:
                return self._failed_execution(
                    reason_code="real_primitive_lower_not_active",
                    evidence_refs=evidence,
                )
            try:
                cut = self._latest_measured(now=float(self.clock()))
            except RuntimeError:
                return self._failed_execution(
                    reason_code="real_primitive_measurement_unavailable_or_stale",
                    evidence_refs=evidence,
                )
            if cut.revision != last_revision:
                latest_cut = cut
                last_revision = cut.revision
                evidence.update(cut.evidence_refs)
                measured_pose = tuple(
                    float(value)
                    for value in pose_matrix_to_xyzquat(
                        cut.pose_world_by_component[self.profile.component_id],
                        name="real_agent_terminal_measured_pose",
                    )
                )
                translation, rotation = _pose_error(measured_pose, self.target_pose)
                if (
                    snapshot.dispatch_count > 0
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
            evidence_refs=evidence,
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["RealAgentPrimitiveSkillLifecycle"]
