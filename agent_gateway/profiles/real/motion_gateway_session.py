"""Default-off Agent Gateway session for reviewed real TCP motion Skills."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from agent_gateway.profiles.segmentation import (
    LiveSegmentationProvider,
    build_segmentation_executor,
)
from agent_gateway.profiles.back_projection import (
    LiveBackProjectionProvider,
    build_back_projection_executor,
)
from agent_gateway.profiles.placement import (
    PLACEMENT_MAX_OPERATION_DURATION_MS_V1,
    PlacementProvider,
    build_placement_executor,
)
from agent_gateway.protocol.contracts import (
    AgentObservationSnapshot,
    AgentSkillOperation,
    AgentSkillRequest,
)
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.runtime.access_policy import (
    AgentSkillAccessPolicy,
    SkillExecutionGrant,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker
from agent_gateway.runtime.event_journal import BoundedAgentGatewayEventJournal
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.hardware_view_provider import (
    HardwareResourceBinding,
    HardwareViewProvider,
)
from agent_gateway.runtime.operation_port import AgentGatewayAuthoritativePortReceipt
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.protocol_runtime import AgentGatewayProtocolRuntime
from agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from agent_gateway.runtime.service import (
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleControlEvidence,
    StationOperationWorkerBinding,
)
from agent_gateway.runtime.station_revision import AgentStationRevisionTracker
from agent_gateway.skills.gripper.definition import SET_GRIPPER_STATE_SKILL_ID_V1
from agent_gateway.skills.gripper.runtime import (
    GripperControlPort,
    build_gripper_skill_executor,
)
from agent_gateway.skills.motion.definition import (
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
)
from agent_gateway.skills.motion.grasped_object import GRASPED_OBJECT_MOVE_SKILL_ID
from agent_gateway.skills.motion.observed_point import SKILL_ID as OBSERVED_POINT_SKILL_ID
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import pose_matrix_to_xyzquat
from teleop_runtime.skill_deploy.control_runtime import (
    RealSkillMotionEmergencyStopReceipt,
)
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPort,
)

from .back_projection import build_rgbd_geometry_executors
from .grasp_approach_plan import (
    BoundedRealGraspApproachPlanStore,
    RealGraspApproachPlanProvider,
    build_grasp_visible_scene_reader,
    build_real_grasp_approach_plan_executor,
)
from .back_projection import RealSenseBackProjectionCapturePort
from .placement import RealPlacementRobotStatePort
from .grasp_fast_plan import RealFastGraspPlanProvider, build_real_fast_grasp_plan_executor
from .grasp_observation import (
    build_grasp_observation_executor,
    read_retained_grasp_observation,
)
from .gripper_skill import RealDmtacGripperControlPort, stop_arm_and_grippers
from .localization_gateway import (
    ObjectLocalizationProvider,
    build_real_object_localization_executor,
)
from .motion_primitive_profiles import (
    RealAgentPrimitiveContractBundle,
)
from .motion_primitive_skill import RealAgentPrimitiveSkillLifecycle
from .motion_observed_scene import ObservedSafeMovePlanner, SafeMotionObservationConfig
from .observed_point_motion import (
    RealObservedAnchorMotionLifecycle,
    RealObservedPointMotionLifecycle,
)
from .object_transport import build_object_transport_executor
from .observation_gateway import RealControlOwnerState
from .perception_gateway import build_real_perception_passive_executors
from .point_grasp_gateway import (
    StationBoundPointGraspProvider,
    build_station_bound_point_grasp_executor,
)
from .point_grasp_render_gateway import (
    StationBoundPointGraspRenderProvider,
    build_station_bound_point_grasp_render_executor,
)
from .realsense_perception import RealSenseAgentPerceptionProvider
from .segmentation import RealSenseSegmentationCapturePort
from .tool_entity import RealToolEntityOptions


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _configured_gripper_ports(gripper_port, gripper_ports):
    if gripper_port is not None:
        if gripper_ports is not None:
            raise ValueError("Select the v1 single port or explicit v2 ports, not both.")
        return {gripper_port.component_id: gripper_port}
    return dict(gripper_ports or {})


class RealAgentDomainEmergencyStopPort:
    """Profile binding from Gateway safety control to the lower RM65 owner."""

    no_io = False

    def __init__(
        self,
        *,
        port_id: str,
        emergency_stop: Callable[[str], RealSkillMotionEmergencyStopReceipt],
        gripper_port: GripperControlPort | None = None,
        gripper_ports: Mapping[str, RealDmtacGripperControlPort] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.port_id = _required(port_id, field_name="port_id")
        if not callable(emergency_stop) or not callable(clock):
            raise TypeError("emergency_stop and clock must be callable.")
        self._emergency_stop = emergency_stop
        self._gripper_ports = _configured_gripper_ports(gripper_port, gripper_ports)
        self._clock = clock
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._receipts: list[AgentGatewayAuthoritativePortReceipt] = []

    @property
    def receipts(self) -> tuple[AgentGatewayAuthoritativePortReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def request_emergency_stop(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        normalized_reason = _required(reason, field_name="reason")
        if not self._gripper_ports:
            lower = self._emergency_stop(normalized_reason)
            hand_hold = None
        else:
            lower, hand_hold = stop_arm_and_grippers(
                arm_stop=lambda: self._emergency_stop(normalized_reason),
                ports=self._gripper_ports,
                reason=normalized_reason,
            )
        if not isinstance(lower, RealSkillMotionEmergencyStopReceipt):
            raise TypeError(
                "lower emergency_stop must return RealSkillMotionEmergencyStopReceipt."
            )
        stopped = lower.succeeded and (hand_hold is None or hand_hold.succeeded)
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        receipt = AgentGatewayAuthoritativePortReceipt(
            port_id=self.port_id,
            operation_kind="emergency_stop",
            operation_id=None if operation is None else operation.operation_id,
            succeeded=stopped,
            status=(
                "real_rm65_domain_emergency_stop_confirmed"
                if stopped
                else "real_rm65_domain_emergency_stop_unconfirmed"
            ),
            completed_at=float(self._clock()),
            call_sequence=sequence,
            lower_operation_id=(
                None
                if operation is None
                else "real-agent-motion:" + operation.operation_id
            ),
            evidence_refs=lower.evidence_refs
            if hand_hold is None
            else (*lower.evidence_refs, *hand_hold.evidence_refs),
            proves_motion_or_hold=stopped,
        )
        with self._lock:
            self._receipts.append(receipt)
        return receipt


class RealAgentPrimitiveObservationProducer:
    """Project lower owner state and measured port cuts without device I/O."""

    def __init__(
        self,
        *,
        contract_bundle: RealAgentPrimitiveContractBundle,
        gripper_port: GripperControlPort | None = None,
        gripper_ports: Mapping[str, RealDmtacGripperControlPort] | None = None,
        port: RealAgentMotionPort,
        owner_state_provider: Callable[[], RealControlOwnerState],
        revision_tracker: AgentStationRevisionTracker | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(contract_bundle, RealAgentPrimitiveContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not isinstance(port, RealAgentMotionPort):
            raise TypeError("port must be typed.")
        if not callable(owner_state_provider) or not callable(clock):
            raise TypeError("owner_state_provider and clock must be callable.")
        self.contract_bundle = contract_bundle
        self.gripper_port = gripper_port
        self.gripper_ports = _configured_gripper_ports(gripper_port, gripper_ports)
        self.port = port
        self.owner_state_provider = owner_state_provider
        self.revision_tracker = (
            AgentStationRevisionTracker()
            if revision_tracker is None
            else revision_tracker
        )
        if not isinstance(self.revision_tracker, AgentStationRevisionTracker):
            raise TypeError("revision_tracker must be an AgentStationRevisionTracker.")
        self.clock = clock
        self._validate_port_binding()

    @property
    def current_station_revision(self) -> int:
        return self._observe_control_state()

    def _control_fingerprint(self) -> dict[str, object]:
        """Return only state that can change admission semantics.

        The lower port revision also advances for every measured/control cycle
        and for dispatch bookkeeping. Those are fresh measurement facts, not
        optimistic-concurrency changes to the station control state.
        """

        attachment = self.contract_bundle.attachment
        port_snapshot = self.port.snapshot()
        owner_cut = self.port.latest_owner_state()
        fingerprint: dict[str, object] = {
            "runtime_session_id": attachment.runtime_session_id,
            "control_domain_id": attachment.control_domain_id,
            "target_binding_id": attachment.target_binding_id,
            "execution_manifest_hash": attachment.execution_manifest_hash,
            "phase": port_snapshot.phase.value,
            "operation_id": port_snapshot.operation_id,
            "request_id": port_snapshot.request_id,
            "activation_grant_id": port_snapshot.activation_grant_id,
            "hold_reason": port_snapshot.hold_reason,
            "hold_succeeded": port_snapshot.hold_succeeded,
            "fault": port_snapshot.fault,
        }
        if owner_cut is not None:
            fingerprint.update(
                {
                    "owner_token_id": owner_cut.owner_token_id,
                    "owner_kind": owner_cut.owner_kind,
                    "owner_generation": owner_cut.owner_generation,
                    "authority_epoch": owner_cut.authority_epoch,
                    "execution_epoch": owner_cut.execution_epoch,
                    "lifecycle_state": owner_cut.lifecycle_state,
                    "execution_gate": owner_cut.execution_gate,
                }
            )
        if self.gripper_ports:
            fingerprint["gripper_command_attempt_refs"] = {
                key: port.last_command_attempt_ref
                for key, port in self.gripper_ports.items()
            }
        return fingerprint

    def _observe_control_state(self) -> int:
        return self.revision_tracker.observe(
            self._control_fingerprint()
        ).station_revision

    def _owner_state(self) -> RealControlOwnerState:
        owner = self.owner_state_provider()
        if not isinstance(owner, RealControlOwnerState):
            raise TypeError("owner_state_provider must return RealControlOwnerState.")
        attachment = self.contract_bundle.attachment
        checks = {
            "runtime_session_id": attachment.runtime_session_id,
            "control_domain_id": attachment.control_domain_id,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "target_binding_id": attachment.target_binding_id,
            "execution_manifest_hash": attachment.execution_manifest_hash,
            "backend_inventory_hash": attachment.backend_inventory_hash,
            "clock_domain_id": attachment.clock_domain_id,
        }
        for field_name, expected in checks.items():
            if getattr(owner, field_name) != expected:
                raise ValueError(f"Real motion owner {field_name} mismatch.")
        if owner.lifecycle_state != "running" or owner.execution_gate != "ready":
            snapshot = self.port.snapshot()
            confirmed_hold = (
                owner.lifecycle_state == "hold"
                and owner.execution_gate == "hold"
                and snapshot.phase == RealAgentMotionPhase.HELD
                and snapshot.hold_succeeded is True
            )
            if not confirmed_hold:
                raise RuntimeError(
                    "Real motion owner is not RUNNING/READY or a lower-confirmed HOLD: "
                    f"lifecycle={owner.lifecycle_state}, gate={owner.execution_gate}, "
                    f"phase={snapshot.phase.value}, hold_succeeded={snapshot.hold_succeeded}."
                )
        return owner

    def _validate_port_binding(self) -> None:
        attachment = self.contract_bundle.attachment
        profile = self.contract_bundle.cardinal_profile
        config = self.port.config
        attachment_checks = {
            "public reference_frame_id": profile.reference_frame_id,
            "public reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "execution reference_frame_id": profile.execution_reference_frame_id,
            "execution reference_frame_spec_hash": (
                profile.execution_reference_frame_spec_hash
            ),
        }
        attachment_values = {
            "public reference_frame_id": attachment.reference_frame_id,
            "public reference_frame_spec_hash": (attachment.reference_frame_spec_hash),
            "execution reference_frame_id": (attachment.execution_reference_frame_id),
            "execution reference_frame_spec_hash": (
                attachment.execution_reference_frame_spec_hash
            ),
        }
        for field_name, expected in attachment_checks.items():
            if attachment_values[field_name] != expected:
                raise ValueError(f"Real observation {field_name} mismatch.")
        checks = {
            "execution_frame_id": profile.execution_reference_frame_id,
            "execution_frame_spec_hash": (profile.execution_reference_frame_spec_hash),
        }
        for field_name, expected in checks.items():
            if getattr(config, field_name) != expected:
                raise ValueError(f"Real motion port {field_name} mismatch.")
        if set(config.component_id_by_side.values()) != set(attachment.component_ids):
            raise ValueError("Real motion port component binding mismatch.")
        if config.component_id_by_side.get(profile.side) != profile.component_id:
            raise ValueError("Real motion port component/side binding mismatch.")

    def _capture_snapshot_with_read_evidence(
        self,
        *,
        active_operation_id: str | None,
    ) -> tuple[AgentObservationSnapshot, str, tuple[str, ...], float]:
        now = float(self.clock())
        owner = self._owner_state()
        cut = self.port.latest_measured_pose()
        if cut is None:
            raise RuntimeError("No lower measured TCP-pose cut is available.")
        age_ms = max(0, math.ceil((now - cut.measured_at) * 1_000.0))
        maximum_age = self.contract_bundle.convergence_profile.max_measurement_age_ms
        if age_ms > maximum_age:
            raise RuntimeError("Lower measured TCP-pose cut is stale.")
        # The owner state is refreshed on every control cut while measured TCP
        # poses advance only with a new lower hardware source sequence.  A newer
        # owner cut can therefore be combined with an older, still-fresh
        # measurement.  The combined snapshot becomes available at the later
        # timestamp while source freshness remains anchored to measured_at.
        available_at = max(cut.available_at, owner.captured_at)
        if available_at > now:
            raise RuntimeError("Lower real observation inputs are not yet available.")
        attachment = self.contract_bundle.attachment
        if set(cut.pose_world_by_component) != set(attachment.component_ids):
            raise ValueError(
                "Lower measured TCP-pose cut does not match the attached components."
            )
        poses: dict[str, list[float]] = {}
        frame_resolution_hashes: dict[str, str] = {}
        for component_id, execution_pose in cut.pose_world_by_component.items():
            public_pose, resolution = (
                self.contract_bundle.frame_graph.transform_pose_matrix(
                    execution_pose,
                    source_reference_frame_id=(attachment.execution_reference_frame_id),
                    target_reference_frame_id=attachment.reference_frame_id,
                )
            )
            poses[component_id] = [
                float(value)
                for value in pose_matrix_to_xyzquat(
                    public_pose,
                    name=f"real_agent_public_observation.{component_id}",
                )
            ]
            frame_resolution_hashes[component_id] = resolution.semantic_content_hash
        station_revision = self._observe_control_state()
        snapshot_hash = canonical_content_hash(
            {
                "manifest_hash": self.contract_bundle.route_manifest.semantic_content_hash,
                "station_revision": station_revision,
                "owner_state_hash": owner.semantic_content_hash,
                "measured_cut_hash": cut.semantic_content_hash,
                "active_operation_id": active_operation_id,
                "measurement_freshness_ms": age_ms,
                "public_pose_xyz_quat_xyzw_by_component": poses,
                "frame_resolution_hash_by_component": frame_resolution_hashes,
            },
            domain="dexteleop.real_agent_motion.observation_snapshot_id.v1",
        )
        snapshot = AgentObservationSnapshot(
            snapshot_id="real-agent-motion-observation:" + snapshot_hash,
            runtime_session_id=attachment.runtime_session_id,
            skill_route_manifest_hash=(
                self.contract_bundle.route_manifest.semantic_content_hash
            ),
            station_revision=station_revision,
            captured_at=cut.measured_at,
            available_at=available_at,
            clock_domain_id=attachment.clock_domain_id,
            authority_epoch=owner.authority_epoch,
            execution_epoch=owner.execution_epoch,
            dispatch_owner_generation=owner.owner_generation,
            lifecycle_state=owner.lifecycle_state,
            execution_gate=owner.execution_gate,
            active_operation_id=active_operation_id,
            source_binding_id=attachment.source_binding_id,
            producer_epoch=attachment.producer_epoch,
            target_binding_id=attachment.target_binding_id,
            execution_manifest_hash=attachment.execution_manifest_hash,
            frame_graph_hash=attachment.frame_graph_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            measurement_summary={
                "attachment_hash": attachment.semantic_content_hash,
                "owner_state_hash": owner.semantic_content_hash,
                "measured_state_hash": cut.semantic_content_hash,
                "pose_xyz_quat_xyzw_by_component": poses,
                "reference_frame_id": attachment.reference_frame_id,
                "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
                "execution_reference_frame_id": (
                    attachment.execution_reference_frame_id
                ),
                "execution_reference_frame_spec_hash": (
                    attachment.execution_reference_frame_spec_hash
                ),
                "frame_resolution_hash_by_component": frame_resolution_hashes,
            },
            measurement_freshness_ms=age_ms,
            measurement_valid=True,
        )
        return (
            snapshot,
            cut.semantic_content_hash,
            tuple(
                sorted(
                    {
                        *owner.evidence_refs,
                        *cut.evidence_refs,
                        *(
                            "real-agent-observation-frame-resolution:" + value
                            for value in frame_resolution_hashes.values()
                        ),
                    }
                )
            ),
            now,
        )

    def capture_snapshot(
        self,
        *,
        active_operation_id: str | None,
    ) -> AgentObservationSnapshot:
        snapshot, _, _, _ = self._capture_snapshot_with_read_evidence(
            active_operation_id=active_operation_id,
        )
        return snapshot

    def read_tcp_pose(
        self,
        *,
        active_operation_id: str,
        parameters: Mapping[str, object],
    ) -> PassiveReadResult:
        return self.capture_tcp_pose(
            active_operation_id=active_operation_id, parameters=parameters,
        )

    def capture_tcp_pose(
        self,
        *,
        active_operation_id: str,
        parameters: Mapping[str, object],
    ) -> PassiveReadResult:
        """Capture the original V1 projection once for either read Route."""
        route = self.contract_bundle.route_manifest.route(
            "state.read_tcp_pose",
            "1",
        ).route
        self.contract_bundle.payload_validators.validate_parameters(
            route=route,
            payload=parameters,
        )
        snapshot, source_measurement_hash, evidence_refs, cutoff_at = (
            self._capture_snapshot_with_read_evidence(
                active_operation_id=active_operation_id,
            )
        )
        attachment = self.contract_bundle.attachment
        requested_components = tuple(sorted(parameters["component_ids"]))
        all_poses = snapshot.measurement_summary["pose_xyz_quat_xyzw_by_component"]
        poses = {
            component_id: list(all_poses[component_id])
            for component_id in requested_components
        }
        measurement_observation_hash = canonical_content_hash(
            {
                "attachment_hash": attachment.semantic_content_hash,
                "source_measurement_hash": source_measurement_hash,
                "poses": poses,
            },
            domain="dexteleop.real_agent_motion.tcp_pose_measurement_observation.v1",
        )
        observation_cut_hash = canonical_content_hash(
            {
                "snapshot_id": snapshot.snapshot_id,
                "source_measurement_hash": source_measurement_hash,
                "cutoff_at": cutoff_at,
                "active_operation_id": active_operation_id,
            },
            domain="dexteleop.real_agent_motion.tcp_pose_read_cut.v1",
        )
        result = {
            "observation_cut_id": "real-agent-motion-cut:" + observation_cut_hash,
            "observation_cut_hash": observation_cut_hash,
            "cutoff_at": cutoff_at,
            "cutoff_clock_domain_id": attachment.clock_domain_id,
            "component_ids": list(requested_components),
            "reference_frame_id": attachment.reference_frame_id,
            "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": attachment.frame_graph_hash,
            "tool_config_hash": attachment.tool_config_hash,
            "calibration_hash": attachment.calibration_hash,
            "measurement_id": "real-agent-measured-pose:" + source_measurement_hash,
            "measurement_observation_hash": measurement_observation_hash,
            "source_measurement_hash": source_measurement_hash,
            "measured_at": snapshot.captured_at,
            "available_at": snapshot.available_at,
            "clock_domain_id": attachment.clock_domain_id,
            "pose_by_component": poses,
            "pose_schema_hash_by_component": {
                component_id: attachment.pose_schema_hash
                for component_id in requested_components
            },
            "representation_by_component": {
                component_id: "cartesian_pose" for component_id in requested_components
            },
            "observation_binding_id": attachment.observation_binding_id,
            "observation_plan_hash": attachment.observation_plan_hash,
        }
        self.contract_bundle.payload_validators.validate_execution_result(
            route=route,
            payload=result,
            request_parameters=parameters,
        )
        return PassiveReadResult(
            execution_result=result,
            evidence_refs=evidence_refs,
            reason_code="passive_real_tcp_pose_read_completed",
        )

    def capture_and_publish(self, service: AgentGatewayService):
        snapshot = self.capture_snapshot(
            active_operation_id=service.operation_store.active_operation_id,
        )
        return service.publish_observation(snapshot=snapshot, now=snapshot.available_at)


def _shared_port_hold_evidence(
    *,
    port: RealAgentMotionPort,
    reason: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    timeout_sec: float,
) -> StationLifecycleControlEvidence:
    now = float(clock())
    snapshot = port.snapshot()
    lower_operation_id = (
        None
        if snapshot.operation_id is None
        else "real-agent-motion:" + snapshot.operation_id
    )
    if snapshot.phase == RealAgentMotionPhase.HELD and snapshot.hold_succeeded:
        return StationLifecycleControlEvidence(
            succeeded=True,
            status="real_agent_domain_hold_confirmed",
            lower_operation_id=lower_operation_id,
            evidence_refs=(snapshot.evidence_ref,),
            lower_hold_confirmed=True,
        )
    if snapshot.operation_id is not None and snapshot.phase in {
        RealAgentMotionPhase.PENDING_ACTIVATION,
        RealAgentMotionPhase.ACTIVE,
        RealAgentMotionPhase.FAULTED,
    }:
        port.request_hold(
            operation_id=snapshot.operation_id,
            reason=reason,
            requested_at=now,
        )
    elif snapshot.phase not in {
        RealAgentMotionPhase.HOLD_REQUESTED,
        RealAgentMotionPhase.HOLD_APPLYING,
    }:
        port.request_domain_hold(reason=reason, requested_at=now)
    deadline = now + float(timeout_sec)
    while float(clock()) < deadline:
        snapshot = port.snapshot()
        if snapshot.phase == RealAgentMotionPhase.HELD:
            confirmed = bool(snapshot.hold_succeeded)
            return StationLifecycleControlEvidence(
                succeeded=confirmed,
                status=(
                    "real_agent_domain_hold_confirmed"
                    if confirmed
                    else "real_agent_domain_hold_unconfirmed"
                ),
                lower_operation_id=lower_operation_id,
                evidence_refs=(snapshot.evidence_ref,),
                lower_hold_confirmed=confirmed,
            )
        if snapshot.phase == RealAgentMotionPhase.FAULTED and (
            snapshot.hold_succeeded is False
        ):
            break
        sleep(0.01)
    snapshot = port.snapshot()
    return StationLifecycleControlEvidence(
        succeeded=False,
        status="real_agent_domain_hold_timeout",
        lower_operation_id=lower_operation_id,
        evidence_refs=(snapshot.evidence_ref,),
        lower_hold_confirmed=False,
    )


@dataclass
class _RealAgentDomainHoldPort:
    """Retain the existing lower HOLD boundary when no motion Route is published."""

    no_io = False
    port_id: str
    domain_hold: Callable[..., StationLifecycleControlEvidence]
    clock: Callable[[], float]
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _next_sequence: int = field(default=0, init=False)

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        evidence = self.domain_hold(
            operation=operation,
            reason=_required(reason, field_name="reason"),
        )
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        return AgentGatewayAuthoritativePortReceipt(
            port_id=self.port_id,
            operation_kind="hold",
            operation_id=None if operation is None else operation.operation_id,
            succeeded=evidence.succeeded,
            status=evidence.status,
            completed_at=float(self.clock()),
            call_sequence=sequence,
            lower_operation_id=evidence.lower_operation_id,
            evidence_refs=evidence.evidence_refs,
            proves_motion_or_hold=(evidence.succeeded and evidence.lower_hold_confirmed),
        )


@dataclass
class RealAgentPrimitiveGatewaySession:
    contract_bundle: RealAgentPrimitiveContractBundle
    port: RealAgentMotionPort
    observation_producer: RealAgentPrimitiveObservationProducer
    executors_by_skill_id: Mapping[str, ReusableStationAgentSkillOperationExecutor]
    emergency_stop_port: RealAgentDomainEmergencyStopPort | None
    state_executor: PassiveReadSkillExecutor
    perception_provider: RealSenseAgentPerceptionProvider | None
    rgb_executor: PassiveReadSkillExecutor | None
    depth_executor: PassiveReadSkillExecutor | None
    back_projection_provider: LiveBackProjectionProvider | None
    back_projection_executor: PassiveReadSkillExecutor | None
    localization_provider: ObjectLocalizationProvider | None
    localization_executor: PassiveReadSkillExecutor | None
    segmentation_provider: LiveSegmentationProvider | None
    segmentation_executor: PassiveReadSkillExecutor | None
    placement_provider: PlacementProvider | None
    placement_executor: PassiveReadSkillExecutor | None
    station_point_grasp_provider: StationBoundPointGraspProvider | None
    station_point_grasp_executor: PassiveReadSkillExecutor | None
    station_point_grasp_render_provider: StationBoundPointGraspRenderProvider | None
    station_point_grasp_render_executor: PassiveReadSkillExecutor | None
    grasp_approach_plan_store: BoundedRealGraspApproachPlanStore | None
    grasp_approach_plan_provider: RealGraspApproachPlanProvider | None
    grasp_approach_plan_executor: PassiveReadSkillExecutor | None
    artifact_store: RecoverableAgentArtifactStore | None
    owns_artifact_store: bool
    hardware_view_provider: HardwareViewProvider
    service: AgentGatewayService
    grasp_observation_executor: PassiveReadSkillExecutor | None = None
    object_transport_executor: PassiveReadSkillExecutor | None = None
    spatial_goal_executor: PassiveReadSkillExecutor | None = None
    tcp_pose_comparison_executor: PassiveReadSkillExecutor | None = None
    gripper_port: RealDmtacGripperControlPort | None = None
    rgbd_geometry_executors: tuple[PassiveReadSkillExecutor, ...] = ()
    gripper_ports: Mapping[str, RealDmtacGripperControlPort] = field(default_factory=dict)
    grasp_fast_plan_executor: PassiveReadSkillExecutor | None = None
    surface_anchor_executor: PassiveReadSkillExecutor | None = None
    tool_object_relation_executor: PassiveReadSkillExecutor | None = None
    tool_hover_executor: PassiveReadSkillExecutor | None = None
    support_release_executor: PassiveReadSkillExecutor | None = None
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _close_outcome: AgentGatewayOperationOutcome | None = field(
        default=None,
        init=False,
    )
    _artifact_store_closed: bool = field(default=False, init=False)
    _grasp_approach_plan_store_closed: bool = field(default=False, init=False)
    _closed_executor_ids: set[int] = field(default_factory=set, init=False)
    _gripper_connections: dict[str, tuple[int, bool]] | None = field(default=None, init=False)
    _gripper_hold_pending: bool = field(default=False, init=False)
    _gripper_hold_retry_at: float = field(default=0.0, init=False)

    def reconcile_gripper_connections(self, stop_arm: Callable[[str], object]) -> bool:
        """Stop old control on loss; establish HOLD on a recovered serial owner."""
        current = {key: port.connection_token for key, port in self.gripper_ports.items()}
        previous, self._gripper_connections = self._gripper_connections, current
        if previous is None:
            return True
        changed = current != previous
        if not changed and not self._gripper_hold_pending:
            return True
        if not changed and time.monotonic() < self._gripper_hold_retry_at:
            return False
        operation = self.service.operation_store.active_operation()
        request = (None if operation is None else
                   self.service.operation_store.request_for_operation(operation.operation_id))
        active_control = request is not None and self.service.manifest.route(
            request.skill_id, request.skill_version, request.route_id,
            request.route_version).route.control_effect_class != "read_only"
        recovered = any(token[1] and previous.get(key) != token for key, token in current.items())
        if not active_control and not recovered and not self._gripper_hold_pending:
            return True
        # Apply arm HOLD on its control thread before cancellation waits for
        # it. Stop hands explicitly: a used motion executor may prove only arm HOLD.
        arm_error = None

        def stop_control_owner():
            nonlocal arm_error
            try:
                return stop_arm("gripper_connection_changed")
            except BaseException as error:
                arm_error = error
                raise

        try:
            _, hand_hold = stop_arm_and_grippers(
                arm_stop=stop_control_owner, ports=self.gripper_ports,
                reason="gripper_connection_changed")
            hand_stopped, hand_error = hand_hold.succeeded, None
        except Exception as error:
            if arm_error is not None:
                raise arm_error from error
            hand_stopped, hand_error = False, str(error)[:500]
        self._gripper_hold_pending = (recovered or self._gripper_hold_pending) and not hand_stopped
        self._gripper_hold_retry_at = time.monotonic() + 1.0
        outcome = (self.service.cancel_operation(
            operation_id=operation.operation_id, reason="gripper_connection_changed",
            current_station_revision=self.observation_producer.current_station_revision,
        ) if active_control else None)
        if changed or not self._gripper_hold_pending:
            logging.getLogger(__name__).warning(
                "Gripper connection changed: operation_id=%s before=%s after=%s "
                "gripper_hold=%s hand_error=%s cancellation=%s",
                None if operation is None else operation.operation_id, previous, current,
                hand_stopped, hand_error, None if outcome is None else outcome.reason_code)
        return not self._gripper_hold_pending

    def __post_init__(self) -> None:
        if (self.contract_bundle.tcp_pose_comparison_registration is not None) != (self.tcp_pose_comparison_executor is not None):
            raise ValueError("TCP comparison registration and executor must coexist.")
        if self.tcp_pose_comparison_executor is not None and not isinstance(self.tcp_pose_comparison_executor, PassiveReadSkillExecutor):
            raise TypeError("TCP comparison executor must be passive.")
        if (self.contract_bundle.spatial_goal_registration is not None) != (self.spatial_goal_executor is not None):
            raise ValueError("Spatial goal registration and executor must coexist.")
        if (self.contract_bundle.tool_object_relation_registration is not None) != (
                self.tool_object_relation_executor is not None):
            raise ValueError("Tool-object relation registration and executor must coexist.")
        if (self.contract_bundle.tool_hover_registration is not None) != (self.tool_hover_executor is not None):
            raise ValueError("Tool hover registration and executor must coexist.")
        if (self.contract_bundle.support_release_registration is not None) != (
                self.support_release_executor is not None):
            raise ValueError("Support release registration and executor must coexist.")
        advertised = (
            "own_domain_emergency_stop_v1"
            in self.contract_bundle.route_manifest.supported_transport_features
        )
        if advertised != (self.emergency_stop_port is not None):
            raise ValueError("Real emergency-stop feature and profile port must match.")
        if (self.contract_bundle.gripper_contract is not None) != (
            bool(self.gripper_ports)
        ):
            raise ValueError("Composed gripper port and contract must coexist.")
        enabled = self.contract_bundle.perception_capture is not None
        if len(self.rgbd_geometry_executors) != len(self.contract_bundle.rgbd_geometry_registrations):
            raise ValueError("RGB-D geometry registration and executors disagree.")
        if enabled != (self.perception_provider is not None):
            raise ValueError("Real perception registration and provider disagree.")
        if enabled != (self.artifact_store is not None):
            raise ValueError("Real perception requires an exact Artifact store.")
        if enabled != (
            self.rgb_executor is not None and self.depth_executor is not None
        ):
            raise ValueError("Real perception requires both passive executors.")
        back_projection_enabled = bool(
            self.contract_bundle.perception_capture is not None
            and self.contract_bundle.perception_capture.back_projection is not None
        )
        if back_projection_enabled != all(
            item is not None
            for item in (
                self.back_projection_provider,
                self.back_projection_executor,
            )
        ):
            raise ValueError(
                "Real back-projection registration and runtime resources disagree."
            )
        grasp_enabled = self.contract_bundle.localization_registration is not None
        if grasp_enabled != all(
            item is not None
            for item in (
                self.localization_provider,
                self.localization_executor,
                self.station_point_grasp_provider,
                self.station_point_grasp_executor,
                self.station_point_grasp_render_provider,
                self.station_point_grasp_render_executor,
            )
        ):
            raise ValueError(
                "Real point-grasp registration and runtime resources disagree."
            )
        segmentation_enabled = (
            self.contract_bundle.segmentation_registration is not None
        )
        if segmentation_enabled != all(
            item is not None
            for item in (self.segmentation_provider, self.segmentation_executor)
        ):
            raise ValueError(
                "Real segmentation registration and runtime resources disagree."
            )
        if (self.contract_bundle.surface_anchor_capture_registration is not None) != (
            self.surface_anchor_executor is not None
        ):
            raise ValueError("Real surface anchor registration and executor disagree.")
        placement_enabled = self.contract_bundle.placement_registration is not None
        if placement_enabled != all(
            item is not None
            for item in (self.placement_provider, self.placement_executor)
        ):
            raise ValueError(
                "Real placement registration and runtime resources disagree."
            )
        grasp_plan_enabled = (
            self.contract_bundle.grasp_approach_plan_registration is not None
        )
        fast_enabled = self.contract_bundle.grasp_fast_plan_registration is not None
        if fast_enabled != (self.grasp_fast_plan_executor is not None):
            raise ValueError("Fast grasp registration and executor must coexist.")
        if fast_enabled and not grasp_plan_enabled:
            raise ValueError("Fast grasp planning requires the existing paired plan store.")
        point_registration = self.contract_bundle.station_point_grasp_registration
        if (self.contract_bundle.object_transport_registration is not None) != (self.object_transport_executor is not None):
            raise ValueError("Object transport registration and executor disagree.")
        observation_enabled = point_registration is not None and point_registration.observation_registration is not None
        if observation_enabled != (self.grasp_observation_executor is not None):
            raise ValueError("Frozen grasp observation registration and executor disagree.")
        if grasp_plan_enabled != all(
            item is not None
            for item in (
                self.grasp_approach_plan_store,
                self.grasp_approach_plan_provider,
                self.grasp_approach_plan_executor,
            )
        ):
            raise ValueError(
                "Grasp-approach plan registration and runtime resources disagree."
            )
        if self.owns_artifact_store and self.artifact_store is None:
            raise ValueError("Owned Artifact store must be present.")
        if not isinstance(self.state_executor, PassiveReadSkillExecutor):
            raise TypeError("Real TCP-pose state executor must be passive.")
        if not isinstance(self.hardware_view_provider, HardwareViewProvider):
            raise TypeError("Real hardware view provider must be typed.")

    @property
    def passive_read_executors_by_skill_ref(
        self,
    ) -> Mapping[tuple[str, str], PassiveReadSkillExecutor]:
        """Return every read-only executor by public Skill id and version."""

        executors = {
            "perception.compute_spatial_goal": self.spatial_goal_executor,
            "perception.plan_object_move": self.object_transport_executor,
            "perception.capture_grasp_observation": self.grasp_observation_executor,
            "perception.capture_rgb": self.rgb_executor,
            "perception.capture_depth": self.depth_executor,
            "perception.back_project": self.back_projection_executor,
            "perception.localize_objects": self.localization_executor,
            "perception.segment": self.segmentation_executor,
            "perception.capture_surface_anchor": self.surface_anchor_executor,
            "perception.describe_tool_object_relations": self.tool_object_relation_executor,
            "perception.query_tool_hover_paths": self.tool_hover_executor,
            "perception.query_support_release_pose": self.support_release_executor,
            "perception.plan_placement": self.placement_executor,
            "perception.plan_point_grasp": self.station_point_grasp_executor,
            "perception.render_point_grasp_candidate": (
                self.station_point_grasp_render_executor
            ),
            "perception.plan_grasp_approach": self.grasp_approach_plan_executor,
            "perception.plan_grasp_fast": self.grasp_fast_plan_executor,
        }
        geometry = {
            (item.skill.definition.skill_id, item.skill.definition.skill_version): executor
            for item, executor in zip(self.contract_bundle.rgbd_geometry_registrations,
                                      self.rgbd_geometry_executors, strict=True)
        }
        state = {("state.read_tcp_pose", "1"): self.state_executor}
        if self.tcp_pose_comparison_executor is not None:
            state["state.read_tcp_pose", "2"] = self.tcp_pose_comparison_executor
        return {
            (entry.route.skill_id, entry.route.skill_version): executors[entry.route.skill_id]
            for entry in self.contract_bundle.route_manifest.skill_routes
            if entry.route.skill_id in executors
            and executors[entry.route.skill_id] is not None
            and (entry.route.skill_id, entry.route.skill_version) not in geometry
        } | geometry | state

    def wire_adapter(
        self,
        *,
        authenticated_client_id: str,
        grant_resolver: (
            Callable[[AgentSkillRequest], SkillExecutionGrant | None] | None
        ) = None,
        read_only_access: bool = False,
    ) -> BoundAgentGatewayWireAdapter:
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=authenticated_client_id,
            definition_catalog=self.contract_bundle.definition_catalog,
            schema_catalog=self.contract_bundle.schema_catalog,
            service=self.service,
            publish_observation=lambda: self.observation_producer.capture_and_publish(
                self.service
            ),
            current_station_revision=lambda: (
                self.observation_producer.current_station_revision
            ),
            operation_launcher=self.service.launch_operation,
            grant_resolver=grant_resolver,
            hardware_view_provider=self.hardware_view_provider,
            read_only_access=read_only_access,
        )

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
        grant_resolver: (
            Callable[[AgentSkillRequest], SkillExecutionGrant | None] | None
        ) = None,
        read_only_client_ids: Collection[str] = (),
    ) -> AgentGatewayUnixJsonServer:
        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError("Unix peer map must exactly cover allowed clients.")
        read_only_clients = frozenset(read_only_client_ids)
        if not read_only_clients <= frozenset(peer_map.client_id_by_uid.values()):
            raise ValueError("Read-only clients must exist in the Unix peer map.")
        return AgentGatewayUnixJsonServer(
            socket_path=socket_path,
            peer_client_map=peer_map,
            facade_factory=lambda client_id: self.wire_adapter(
                authenticated_client_id=client_id,
                grant_resolver=grant_resolver,
                read_only_access=client_id in read_only_clients,
            ),
            close_gateway=self.close,
            socket_mode=socket_mode,
            max_frame_bytes=max_frame_bytes,
            peer_idle_timeout_sec=peer_idle_timeout_sec,
        )

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        active = tuple(
            executor
            for executor in self.executors_by_skill_id.values()
            if executor.active_operation_id is not None
        )
        return bool(active) and all(
            executor.wait(timeout_sec=timeout_sec) for executor in active
        )

    def close(self) -> AgentGatewayOperationOutcome:
        with self._close_lock:
            if self._close_outcome is None:
                outcome = self.service.close(
                    current_station_revision=(
                        self.observation_producer.current_station_revision
                    )
                )
                if outcome.accepted:
                    self._close_outcome = outcome
            else:
                outcome = self._close_outcome
            if not outcome.accepted:
                return outcome
            passive_reads_settled = all(
                executor is None or executor.wait_all(timeout_sec=timeout_sec)
                for executor, timeout_sec in (
                    (self.state_executor, 5.0),
                    (self.tcp_pose_comparison_executor, 5.0),
                    (self.rgb_executor, 5.0),
                    (self.depth_executor, 5.0),
                    (self.back_projection_executor, 5.0),
                    (self.surface_anchor_executor,
                     5.0 if self.contract_bundle.surface_anchor_capture_registration is None
                     else 5.0 + self.contract_bundle.surface_anchor_capture_registration.entry.route.max_operation_duration_ms / 1000.0),
                    (
                        self.placement_executor,
                        PLACEMENT_MAX_OPERATION_DURATION_MS_V1 / 1_000.0 + 5.0,
                    ),
                    (self.localization_executor, 5.0),
                    (self.station_point_grasp_executor, 5.0),
                    (self.station_point_grasp_render_executor, 5.0),
                    (self.grasp_approach_plan_executor, 5.0),
                    (self.grasp_fast_plan_executor, 5.0),
                    (self.grasp_observation_executor, 5.0),
                    (self.object_transport_executor, 5.0),
                    (self.spatial_goal_executor, 5.0),
                    *((executor, 5.0 + registration.entry.route.max_operation_duration_ms / 1000.0)
                      for executor, registration in (
                          (self.tool_object_relation_executor,
                           self.contract_bundle.tool_object_relation_registration),
                          (self.tool_hover_executor, self.contract_bundle.tool_hover_registration),
                          (self.support_release_executor, self.contract_bundle.support_release_registration))
                      if registration is not None),
                    *((executor, 5.0) for executor in self.rgbd_geometry_executors),
                )
            )
            segmentation_settled = (
                self.segmentation_executor is None
                or self.segmentation_executor.wait_all(timeout_sec=(
                    5.0 + self.contract_bundle.segmentation_registration.skill.routes[0].max_operation_duration_ms / 1000.0
                ))
            )
            if not passive_reads_settled or not segmentation_settled:
                raise RuntimeError(
                    "Passive real reads did not quiesce before Gateway close."
                )
            receipt = outcome.port_receipt
            if not (
                isinstance(receipt, AgentGatewayAuthoritativePortReceipt)
                and receipt.succeeded
                and receipt.proves_motion_or_hold
            ):
                raise RuntimeError(
                    "Real Gateway close requires authoritative lower HOLD proof."
                )
            for executor in self.executors_by_skill_id.values():
                executor_id = id(executor)
                if executor_id in self._closed_executor_ids:
                    continue
                executor_receipt = executor.request_hold(
                    operation=None,
                    reason="real_agent_gateway_close_route_fence",
                )
                if not (
                    executor_receipt.succeeded
                    and executor_receipt.proves_motion_or_hold
                ):
                    raise RuntimeError(
                        "Real Gateway route executor did not confirm lower HOLD."
                    )
                executor.close_after_hold(hold_receipt=executor_receipt)
                self._closed_executor_ids.add(executor_id)
            self._close_owned_stores()
            return outcome


    def close_unused(self) -> None:
        """Release construction-only resources after permanently sealing admission.

        The application calls this only before any control-start attempt or
        transport exposure. No result from here is a lower HOLD receipt.
        """

        with self._close_lock:
            self.service.fence_if_unused()
            for executor in self.passive_read_executors_by_skill_ref.values():
                if not executor.wait_all(timeout_sec=0.0):
                    raise RuntimeError("Unused Gateway still has a passive reader.")
            for executor in self.executors_by_skill_id.values():
                executor_id = id(executor)
                if executor_id not in self._closed_executor_ids:
                    executor.close_unused()
                    self._closed_executor_ids.add(executor_id)
            self._close_owned_stores()


    def _close_owned_stores(self) -> None:
        if (
            self.owns_artifact_store
            and self.artifact_store is not None
            and not self._artifact_store_closed
        ):
            self.artifact_store.close()
            self._artifact_store_closed = True
        if (
            self.grasp_approach_plan_store is not None
            and not self._grasp_approach_plan_store_closed
        ):
            self.grasp_approach_plan_store.close()
            self._grasp_approach_plan_store_closed = True


def _real_motion_resource_health(
    *,
    port: RealAgentMotionPort,
    clock: Callable[[], float],
    maximum_owner_age_ms: int,
) -> str:
    """Project cached lower state only; never refresh or command hardware."""

    snapshot = port.snapshot()
    if snapshot.phase == RealAgentMotionPhase.FAULTED or snapshot.fault is not None:
        return "fault"
    owner = port.latest_owner_state()
    if owner is None:
        return "unavailable"
    if owner.lifecycle_state == "fault":
        return "fault"
    age_ms = (float(clock()) - owner.captured_at) * 1_000.0
    if age_ms < 0.0 or age_ms > maximum_owner_age_ms:
        return "unavailable"
    return "ready"


def _real_motion_control_claimed(port: RealAgentMotionPort) -> bool:
    """Read the lower exclusive owner cut without inferring from READY/HOLD."""

    owner = port.latest_owner_state()
    if owner is None:
        raise RuntimeError("No lower real motion owner cut is available.")
    if owner.owner_kind != "exclusive_real_skill_motion_port":
        raise RuntimeError("Lower real motion owner kind is not exclusive.")
    return True


def _real_perception_feed_health(
    provider: RealSenseAgentPerceptionProvider,
) -> str:
    """Project cached camera lifecycle state without capturing a frame."""

    infos = []
    for source in provider.source_binding.camera_sources.values():
        device_info = getattr(source, "device_info", None)
        if not callable(device_info):
            return "unavailable"
        info = device_info()
        if not isinstance(info, Mapping):
            return "unavailable"
        infos.append(info)
    if any(str(item.get("lifecycle_state", "")).lower() == "fault" for item in infos):
        return "fault"
    if all(
        item.get("running") is True
        and item.get("worker_alive") is True
        and item.get("lifecycle_state") == "running"
        for item in infos
    ):
        return "ready"
    return "unavailable"


def _real_perception_control_claimed(
    provider: RealSenseAgentPerceptionProvider,
) -> bool:
    """A running lower camera worker proves that its device is claimed."""

    running = []
    for source in provider.source_binding.camera_sources.values():
        device_info = getattr(source, "device_info", None)
        if not callable(device_info):
            raise TypeError("Real perception source has no status provider.")
        info = device_info()
        if not isinstance(info, Mapping) or not isinstance(info.get("running"), bool):
            raise TypeError("Real perception source status is unavailable.")
        running.append(info["running"])
    return any(running)


def _real_hardware_view_provider(
    *,
    contract_bundle: RealAgentPrimitiveContractBundle,
    gripper_port: GripperControlPort | None = None,
    gripper_ports: Mapping[str, RealDmtacGripperControlPort] | None = None,
    port: RealAgentMotionPort,
    perception_provider: RealSenseAgentPerceptionProvider | None,
    localization_provider: ObjectLocalizationProvider | None,
    segmentation_provider: LiveSegmentationProvider | None,
    operation_store: BoundedInMemoryAgentSkillOperationStore,
    clock: Callable[[], float],
) -> HardwareViewProvider:
    """Bind profile-owned ownership units to existing semantic Routes."""

    arm_route_hashes = {component: [] for component in contract_bundle.attachment.component_ids}
    perception_route_hashes = []
    for entry in contract_bundle.route_manifest.skill_routes:
        route = entry.route
        definition = contract_bundle.definition_catalog.definition(
            route.skill_id,
            route.skill_version,
        )
        for component, hashes in arm_route_hashes.items():
            if component in definition.controlled_component_ids:
                hashes.append(route.semantic_content_hash)
        if route.control_effect_class == "read_only" and route.skill_id.startswith(
            "perception."
        ):
            perception_route_hashes.append(route.semantic_content_hash)
    bindings = [
        HardwareResourceBinding(
            resource_id=component_id,
            resource_kind="robot_arm",
            route_hashes=tuple(hashes),
            health_provider=lambda: _real_motion_resource_health(
                port=port,
                clock=clock,
                maximum_owner_age_ms=(
                    contract_bundle.attachment.max_measurement_age_ms
                ),
            ),
            control_claimed_provider=lambda: _real_motion_control_claimed(port),
        ) for component_id, hashes in arm_route_hashes.items()
    ]
    for component, gripper_port in _configured_gripper_ports(gripper_port, gripper_ports).items():
        hand_route = contract_bundle.route_manifest.route(
            SET_GRIPPER_STATE_SKILL_ID_V1, contract_bundle.gripper_contract.deployment_binding.skill_version
        ).route
        bindings.append(
            HardwareResourceBinding(
                resource_id="manipulation." + component,
                resource_kind="gripper",
                route_hashes=(hand_route.semantic_content_hash,),
                health_provider=lambda port=gripper_port: (
                    "ready" if port.backend.is_ready() else "unavailable"
                ),
                control_claimed_provider=lambda port=gripper_port: bool(
                    port.backend.is_ready()
                ),
            )
        )
    if perception_provider is not None:
        if not perception_route_hashes:
            raise ValueError("Real perception provider has no read-only Routes.")
        bindings.append(
            HardwareResourceBinding(
                resource_id="perception.cameras",
                resource_kind="camera_bundle",
                route_hashes=tuple(perception_route_hashes),
                health_provider=lambda: _real_perception_feed_health(
                    perception_provider
                ),
                control_claimed_provider=lambda: _real_perception_control_claimed(
                    perception_provider
                ),
            )
        )
    model_provider = segmentation_provider or localization_provider
    if model_provider is not None:
        model_route_hashes = tuple(
            entry.route.semantic_content_hash
            for entry in contract_bundle.route_manifest.skill_routes
            if entry.route.skill_id
            in {
                "perception.localize_objects",
                "perception.plan_point_grasp",
                "perception.segment",
                "perception.capture_surface_anchor",
            }
        )
        if not model_route_hashes:
            raise ValueError("Real localization provider has no model Routes.")
        bindings.append(
            HardwareResourceBinding(
                resource_id="perception.object_localization_model",
                resource_kind="local_model",
                route_hashes=model_route_hashes,
                health_provider=lambda: (
                    "ready" if model_provider.started else "unavailable"
                ),
                control_claimed_provider=lambda: False,
            )
        )
    return HardwareViewProvider(
        manifest=contract_bundle.route_manifest,
        operation_store=operation_store,
        bindings=tuple(bindings),
    )


def assemble_real_agent_primitive_gateway_session(
    *,
    contract_bundle: RealAgentPrimitiveContractBundle,
    gripper_port: GripperControlPort | None = None,
    gripper_ports: Mapping[str, RealDmtacGripperControlPort] | None = None,
    port: RealAgentMotionPort,
    owner_state_provider: Callable[[], RealControlOwnerState],
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    perception_provider: RealSenseAgentPerceptionProvider | None = None,
    localization_provider: ObjectLocalizationProvider | None = None,
    segmentation_localizer: object | None = None,
    station_point_grasp_planner: object | None = None,
    station_point_grasp_reachability_evaluator: Callable[..., object] | None = None,
    grasp_trajectory_planner: Callable[..., object] | None = None,
    safe_pose_trajectory_planner: Callable[..., object] | None = None,
    held_object_observer: Callable[..., object] | None = None,
    object_transport_planner_factory: Callable[..., object] | None = None,
    captured_robot_observation_filter: Callable[..., object] | None = None,
    non_motion_read_window: Callable[[], object] | None = None,
    tool_entity_options: RealToolEntityOptions | None = None,
    artifact_store: RecoverableAgentArtifactStore | None = None,
    owns_artifact_store: bool = False,
    revision_tracker: AgentStationRevisionTracker | None = None,
    max_session_admissions_per_route: int | None = None,
    emergency_stop: (
        Callable[[str], RealSkillMotionEmergencyStopReceipt] | None
    ) = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> RealAgentPrimitiveGatewaySession:
    """Assemble motion and optional passive Routes without starting resources."""

    if not isinstance(contract_bundle, RealAgentPrimitiveContractBundle):
        raise TypeError("contract_bundle must be typed.")
    if not isinstance(port, RealAgentMotionPort):
        raise TypeError("port must be typed.")
    clients = tuple(
        sorted({_required(item, field_name="client_id") for item in allowed_client_ids})
    )
    if not clients:
        raise ValueError("allowed_client_ids must not be empty.")
    if (
        isinstance(max_retained_operations, bool)
        or not isinstance(max_retained_operations, int)
        or max_retained_operations <= 0
    ):
        raise ValueError("max_retained_operations must be positive.")
    if not isinstance(owns_artifact_store, bool):
        raise TypeError("owns_artifact_store must be boolean.")
    emergency_stop_advertised = (
        "own_domain_emergency_stop_v1"
        in contract_bundle.route_manifest.supported_transport_features
    )
    if emergency_stop_advertised != (emergency_stop is not None):
        raise ValueError("Real emergency-stop feature and lower callback must match.")
    perception_enabled = contract_bundle.perception_capture is not None
    if perception_enabled != isinstance(
        perception_provider,
        RealSenseAgentPerceptionProvider,
    ):
        raise ValueError("Real perception registration requires its exact provider.")
    if perception_enabled != isinstance(
        artifact_store,
        RecoverableAgentArtifactStore,
    ):
        raise ValueError("Real perception registration requires its Artifact store.")
    if owns_artifact_store and artifact_store is None:
        raise ValueError("Owned Artifact store must be injected.")
    transport_enabled = contract_bundle.object_transport_registration is not None
    if (transport_enabled != callable(held_object_observer)
            or transport_enabled != callable(object_transport_planner_factory)):
        raise ValueError("Object transport requires its bound observation owner and planner factory.")
    grasp_enabled = contract_bundle.localization_registration is not None
    if grasp_enabled != (localization_provider is not None):
        raise ValueError("Localization registration requires its exact provider.")
    if grasp_enabled != (station_point_grasp_planner is not None):
        raise ValueError("Point-grasp registration requires its exact planner.")
    segmentation_enabled = contract_bundle.segmentation_registration is not None
    if segmentation_enabled != (segmentation_localizer is not None):
        raise ValueError("Segmentation registration requires its exact localizer.")
    if (
        segmentation_enabled
        and localization_provider is not None
        and segmentation_localizer is not localization_provider.localizer
    ):
        raise ValueError(
            "Segmentation must reuse the lifecycle-owned localization model."
        )
    point_grasp_registration = contract_bundle.station_point_grasp_registration
    reachability_registered = bool(
        point_grasp_registration is not None
        and point_grasp_registration.profile.arm_reachability_profile is not None
    )
    if reachability_registered != callable(station_point_grasp_reachability_evaluator):
        raise ValueError(
            "Point-grasp arm reachability registration requires its exact evaluator."
        )
    grasp_approach_enabled = any(
        entry.route.skill_id == GRASP_APPROACH_SKILL_ID_V1
        for entry in contract_bundle.route_manifest.skill_routes
    )
    if grasp_approach_enabled != callable(grasp_trajectory_planner):
        raise ValueError(
            "Grasp approach Route requires exactly one URDF/FCL trajectory planner."
        )
    safe_pose_enabled = any(
        entry.route.skill_id in {SAFE_TCP_POSE_TARGET_SKILL_ID_V1, OBSERVED_POINT_SKILL_ID}
        for entry in contract_bundle.route_manifest.skill_routes
    )
    if safe_pose_enabled != callable(safe_pose_trajectory_planner):
        raise ValueError("Safe pose Route requires its independently bound trajectory planner.")
    if contract_bundle.observed_point_profile is not None and (
            not isinstance(safe_pose_trajectory_planner, ObservedSafeMovePlanner)
            or not isinstance(safe_pose_trajectory_planner.config, SafeMotionObservationConfig)):
        raise ValueError("Observed-point Route requires the existing RGB-D safe planner and policy.")
    grasp_approach_plan_enabled = (
        contract_bundle.grasp_approach_plan_registration is not None
    )
    if grasp_approach_plan_enabled and not callable(grasp_trajectory_planner):
        raise ValueError(
            "Grasp-approach planning requires the exact URDF/FCL trajectory planner."
        )
    if captured_robot_observation_filter is not None and not callable(
        captured_robot_observation_filter
    ):
        raise TypeError("captured_robot_observation_filter must be callable or None")
    if max_session_admissions_per_route is not None and (
        isinstance(max_session_admissions_per_route, bool)
        or not isinstance(max_session_admissions_per_route, int)
        or max_session_admissions_per_route <= 0
    ):
        raise ValueError("max_session_admissions_per_route must be positive or None.")
    version = None if contract_bundle.gripper_contract is None else contract_bundle.gripper_contract.deployment_binding.skill_version
    if version == "1" and gripper_port is None:
        raise ValueError("v1 requires the existing single right-gripper port.")
    if version == "2" and gripper_port is not None:
        raise ValueError("v2 requires explicit bilateral gripper_ports.")
    gripper_ports = _configured_gripper_ports(gripper_port, gripper_ports)
    if (contract_bundle.gripper_contract is not None) != bool(gripper_ports):
        raise ValueError("Gripper registration and execution port must coexist.")
    if gripper_ports:
        if any(
            not isinstance(port, RealDmtacGripperControlPort)
            for port in gripper_ports.values()
        ):
            raise TypeError(
                "Real composed gripper requires the reviewed DMTAC adapter."
            )

        def arm_held_for_gripper():
            snapshot = port.snapshot()
            measured = port.latest_measured_pose()
            now = float(clock())
            fresh = (
                measured is not None
                and measured.available_at <= now
                and (
                    0.0
                    <= now - measured.measured_at
                    <= contract_bundle.attachment.max_measurement_age_ms / 1000.0
                )
            )
            return (
                fresh
                and snapshot.phase == RealAgentMotionPhase.HELD
                and snapshot.hold_succeeded is True
                and _real_motion_resource_health(
                    port=port,
                    clock=clock,
                    maximum_owner_age_ms=contract_bundle.attachment.max_measurement_age_ms,
                )
                == "ready"
            )

        for hand in gripper_ports.values():
            hand.bind_motion_guard(arm_held_for_gripper)
    producer = RealAgentPrimitiveObservationProducer(
        **({"gripper_port": gripper_port} if version == "1" else {"gripper_ports": gripper_ports}),
        contract_bundle=contract_bundle,
        port=port,
        owner_state_provider=owner_state_provider,
        revision_tracker=revision_tracker,
        clock=clock,
    )
    if perception_provider is not None:
        assert contract_bundle.perception_capture is not None
        perception_provider.bind_profile(contract_bundle.perception_capture.profile)

    def domain_hold(*, operation, reason):
        del operation

        def arm_stop():
            return _shared_port_hold_evidence(
                port=port,
                reason=reason,
                clock=clock,
                sleep=sleep,
                timeout_sec=contract_bundle.convergence_profile.hold_confirmation_timeout_sec,
            )

        if not gripper_ports:
            return arm_stop()
        arm_hold, hand_hold = stop_arm_and_grippers(
            arm_stop=arm_stop, ports=gripper_ports, reason=reason
        )
        proved = (
            arm_hold.succeeded and arm_hold.lower_hold_confirmed and hand_hold.succeeded
        )
        return StationLifecycleControlEvidence(
            succeeded=proved,
            status="arm_gripper_hold_confirmed"
            if proved
            else "arm_gripper_hold_failed",
            lower_operation_id=arm_hold.lower_operation_id,
            evidence_refs=(*arm_hold.evidence_refs, *hand_hold.evidence_refs),
            lower_hold_confirmed=proved,
        )

    executors: dict[str, ReusableStationAgentSkillOperationExecutor] = {}
    execution_port_by_route_hash = {}
    state_entry = contract_bundle.route_manifest.route(
        "state.read_tcp_pose",
        "1",
    )
    state_executor = PassiveReadSkillExecutor(
        port_id=(
            "agent-gateway:real-passive-tcp-pose:"
            + contract_bundle.route_manifest.runtime_session_id
        ),
        executor_id=contract_bundle.tcp_pose_observe.deployment_binding.executor_id,
        read=lambda request, operation: producer.read_tcp_pose(
            active_operation_id=operation.operation_id,
            parameters=request.parameters,
        ),
        current_station_revision=lambda: producer.current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    execution_port_by_route_hash[state_entry.route.semantic_content_hash] = (
        state_executor
    )
    tcp_pose_comparison_executor = None
    if contract_bundle.tcp_pose_comparison_registration is not None:
        from agent_gateway.profiles.tcp_pose_comparison import (
            build_tcp_pose_comparison_executor,
        )

        # The same service/store is bound before any passive reader can run.
        def lookup_target(operation_id):
            operation = service.operation_store.get(operation_id)
            request = service.operation_store.request_for_operation(operation_id)
            if operation is None or request is None:
                return None
            return request, operation

        registration = contract_bundle.tcp_pose_comparison_registration
        tcp_pose_comparison_executor = build_tcp_pose_comparison_executor(
            registration=registration,
            capture_state=lambda parameters, active_operation_id: producer.capture_tcp_pose(
                parameters=parameters, active_operation_id=active_operation_id,
            ),
            lookup_target=lookup_target,
            route_manifest=contract_bundle.route_manifest,
            payload_validators=contract_bundle.payload_validators,
            clock_domain_id=contract_bundle.attachment.clock_domain_id,
            current_station_revision=lambda: producer.current_station_revision,
            clock=clock, max_retained_operations=max_retained_operations,
        )
        execution_port_by_route_hash[registration.entry.route.semantic_content_hash] = tcp_pose_comparison_executor
    spatial_goal_executor = None
    if contract_bundle.spatial_goal_registration is not None:
        from agent_gateway.profiles.spatial_goal import build_spatial_goal_executor

        registration = contract_bundle.spatial_goal_registration
        spatial_goal_executor = build_spatial_goal_executor(
            registration=registration, current_station_revision=lambda: producer.current_station_revision,
            clock=clock, max_retained_operations=max_retained_operations,
        )
        execution_port_by_route_hash[registration.entry.route.semantic_content_hash] = spatial_goal_executor
    grasp_approach_plan_store = (
        BoundedRealGraspApproachPlanStore(
            max_retained_plans=max_retained_operations,
            planning_operation_lookup=lambda identifier: service.operation_store.get(identifier),
            clock=clock,
        )
        if grasp_approach_plan_enabled
        else None
    )
    motion_entries = tuple(
        entry
        for entry in contract_bundle.route_manifest.skill_routes
        if entry.route.control_effect_class == "motion"
        and entry.route.skill_id != SET_GRIPPER_STATE_SKILL_ID_V1
    )
    for entry in motion_entries:
        route = entry.route
        # Keep the v1 worker identity stable while giving a second version its
        # own lifecycle entry. The route registry and close fence must retain
        # both executors even though they share a public Skill id.
        executor_key = (route.skill_id if route.skill_version == "1"
                        else f"{route.skill_id}@{route.skill_version}")
        binding = contract_bundle.binding_for_skill(route.skill_id, route.skill_version)
        if route.skill_id == EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1:
            registration = contract_bundle.grasp_approach_plan_registration
            if registration is None:
                raise RuntimeError(
                    "Plan execution Route exists without its paired registration."
                )
            profile = registration.profile
        elif route.skill_id == OBSERVED_POINT_SKILL_ID:
            profile = (contract_bundle.observed_anchor_profile_v2
                       if route.skill_version == "2"
                       else contract_bundle.observed_point_profile)
            if profile is None:
                raise RuntimeError("Observed-point Route has no bound profile.")
        elif route.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID:
            profile = contract_bundle.object_motion_profile
            if profile is None:
                raise RuntimeError("Object-motion Route has no bound profile.")
        elif route.skill_id in {
            TCP_POSE_TARGET_SKILL_ID_V1,
            SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
            GRASP_APPROACH_SKILL_ID_V1,
        }:
            profile = contract_bundle.pose_target_profile
            if profile is None:
                raise RuntimeError(
                    "Absolute TCP-pose Route exists without its authorized profile."
                )
        else:
            raise ValueError(f"Unsupported real motion Skill {route.skill_id!r}.")

        def lifecycle_factory(
            *,
            request,
            operation,
            _route=route,
            _binding=binding,
            _profile=profile,
        ):
            lifecycle_type = RealAgentPrimitiveSkillLifecycle
            extra = {}
            if _route.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID:
                from .object_motion import RealObjectMotionLifecycle

                if held_object_port is None:
                    raise RuntimeError("Object-motion Route has no held-object port.")
                lifecycle_type = RealObjectMotionLifecycle
                extra = {"held_object_port": held_object_port}
            if _route.skill_id == OBSERVED_POINT_SKILL_ID:
                lifecycle_type = (RealObservedAnchorMotionLifecycle
                                  if _route.skill_version == "2"
                                  else RealObservedPointMotionLifecycle)

                def lookup_observation(identifier):
                    source_request = service.operation_store.request_for_operation(identifier)
                    source_operation = service.operation_store.get(identifier)
                    return (None if source_request is None or source_operation is None
                            else (source_request, source_operation))

                extra = {"lookup_observation": lookup_observation,
                         "observation_config": safe_pose_trajectory_planner.config}
                if _route.skill_version == "2":
                    extra.update(
                        artifact_store=artifact_store,
                        capture_registration=contract_bundle.surface_anchor_capture_registration,
                        current_station_revision=lambda: producer.current_station_revision,
                    )
            return lifecycle_type(
                port=port,
                contract_bundle=contract_bundle,
                route=_route,
                binding=_binding,
                profile=_profile,
                request=request,
                operation=operation,
                grasp_trajectory_planner=(
                    safe_pose_trajectory_planner
                    if _route.skill_id in {SAFE_TCP_POSE_TARGET_SKILL_ID_V1, OBSERVED_POINT_SKILL_ID,
                                           GRASPED_OBJECT_MOVE_SKILL_ID}
                    else grasp_trajectory_planner
                    if _route.skill_id == GRASP_APPROACH_SKILL_ID_V1
                    else None
                ),
                tcp_trajectory_planner=(
                    getattr(grasp_trajectory_planner, "plan_tcp_pose", None)
                    if _route.skill_id == TCP_POSE_TARGET_SKILL_ID_V1 else None
                ),
                non_motion_read_window=non_motion_read_window,
                grasp_approach_plan_store=(
                    grasp_approach_plan_store
                    if _route.skill_id == EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1
                    else None
                ),
                clock=clock,
                sleep=sleep,
                **extra,
            )

        executor = ReusableStationAgentSkillOperationExecutor(
            port_id=(
                "agent-gateway:real-primitive:"
                + executor_key
                + ":"
                + contract_bundle.route_manifest.runtime_session_id
            ),
            binding=StationOperationWorkerBinding(
                worker_id=(
                    "real-agent-primitive-worker:"
                    + executor_key
                    + ":"
                    + contract_bundle.route_manifest.runtime_session_id
                ),
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
            current_station_revision=lambda: producer.current_station_revision,
            clock=clock,
        )
        executors[executor_key] = executor
        execution_port_by_route_hash[route.semantic_content_hash] = executor
    if contract_bundle.gripper_contract is not None:
        gripper_executor = build_gripper_skill_executor(
            contract_bundle=contract_bundle.gripper_contract,
            **({"port": gripper_port} if version == "1" else {"ports": gripper_ports}),
            current_station_revision=lambda: producer.current_station_revision,
            domain_hold=domain_hold,
            clock=clock,
            sleep=sleep,
        )
        executors[SET_GRIPPER_STATE_SKILL_ID_V1] = gripper_executor
        entry = contract_bundle.route_manifest.route(SET_GRIPPER_STATE_SKILL_ID_V1, contract_bundle.gripper_contract.deployment_binding.skill_version)
        execution_port_by_route_hash[entry.route.semantic_content_hash] = (
            gripper_executor
        )
    rgb_executor = None
    depth_executor = None
    back_projection_provider = None
    back_projection_executor = None
    back_projection_capture_port = None
    localization_executor = None
    segmentation_provider = None
    held_object_port = None
    segmentation_executor = None
    surface_anchor_executor = None
    tool_object_relation_executor = None
    tool_hover_executor = None
    support_release_executor = None
    placement_provider = None
    placement_executor = None
    station_point_grasp_provider = None
    station_point_grasp_executor = None
    station_point_grasp_render_provider = None
    station_point_grasp_render_executor = None
    grasp_approach_plan_provider = None
    grasp_approach_plan_executor = None
    grasp_fast_plan_executor = None
    grasp_observation_executor = None
    object_transport_executor = None
    rgbd_geometry_executors = ()
    if perception_provider is not None and artifact_store is not None:
        registration = contract_bundle.perception_capture
        assert registration is not None
        rgb_executor, depth_executor = build_real_perception_passive_executors(
            registration=registration,
            provider=perception_provider,
            artifact_store=artifact_store,
            runtime_session_id=contract_bundle.attachment.runtime_session_id,
            current_station_revision=lambda: producer.current_station_revision,
            max_retained_operations=max_retained_operations,
            clock=clock,
        )
        execution_port_by_route_hash.update(
            {
                registration.rgb.routes[0].semantic_content_hash: rgb_executor,
                registration.depth.routes[0].semantic_content_hash: depth_executor,
            }
        )
        back_projection_registration = registration.back_projection
        if back_projection_registration is not None:
            frame_binding = registration.frame_binding
            if frame_binding is None:
                raise ValueError("Back-projection requires a v2 frame binding.")
            back_projection_capture_port = RealSenseBackProjectionCapturePort(
                perception_provider=perception_provider,
                frame_binding=frame_binding,
            )
            back_projection_provider = LiveBackProjectionProvider(
                capture_port=back_projection_capture_port,
                registration=back_projection_registration,
            )
            back_projection_executor = build_back_projection_executor(
                registration=back_projection_registration,
                provider=back_projection_provider,
                current_station_revision=lambda: producer.current_station_revision,
                max_retained_operations=max_retained_operations,
                clock=clock,
            )
            execution_port_by_route_hash[
                back_projection_registration.skill.routes[0].semantic_content_hash
            ] = back_projection_executor
        segmentation_registration = contract_bundle.segmentation_registration
        if segmentation_registration is not None:
            assert segmentation_localizer is not None
            segmentation_provider = LiveSegmentationProvider(
                localizer=segmentation_localizer,
                capture_port=RealSenseSegmentationCapturePort(
                    perception_provider=perception_provider,
                ),
                registration=segmentation_registration,
            )
            segmentation_executor = build_segmentation_executor(
                registration=segmentation_registration,
                provider=segmentation_provider,
                artifact_store=artifact_store,
                runtime_session_id=(contract_bundle.attachment.runtime_session_id),
                current_station_revision=(lambda: producer.current_station_revision),
                max_retained_operations=max_retained_operations,
                clock=clock,
            )
            execution_port_by_route_hash[
                segmentation_registration.skill.routes[0].semantic_content_hash
            ] = segmentation_executor
        anchor_registration = contract_bundle.surface_anchor_capture_registration
        if anchor_registration is not None:
            if back_projection_provider is None or segmentation_provider is None:
                raise ValueError("Real surface anchor requires the exact depth and segmentation providers.")
            from agent_gateway.profiles.surface_anchor import build_surface_anchor_capture_executor

            surface_anchor_executor = build_surface_anchor_capture_executor(
                registration=anchor_registration,
                provider=back_projection_provider,
                segmentation_provider=segmentation_provider,
                store=artifact_store,
                current_station_revision=lambda: producer.current_station_revision,
                clock=clock,
                clock_domain_id=contract_bundle.attachment.clock_domain_id,
                max_retained_operations=max_retained_operations,
            )
            execution_port_by_route_hash[
                anchor_registration.entry.route.semantic_content_hash
            ] = surface_anchor_executor
        placement_registration = contract_bundle.placement_registration
        if placement_registration is not None:
            assert segmentation_registration is not None
            assert segmentation_executor is not None
            assert segmentation_provider is not None
            assert back_projection_registration is not None
            assert back_projection_capture_port is not None
            frame_binding = registration.frame_binding
            if frame_binding is None:
                raise ValueError("Placement requires a v2 frame binding.")
            placement_provider = PlacementProvider(
                registration=placement_registration,
                segmentation_registration=segmentation_registration,
                segmentation_executor=segmentation_executor,
                segmentation_provider=segmentation_provider,
                back_projection_registration=back_projection_registration,
                capture_port=back_projection_capture_port,
                robot_state_port=RealPlacementRobotStatePort(
                    port=port,
                    frame_graph=contract_bundle.frame_graph,
                    component_id=placement_registration.profile.component_id,
                    execution_reference_frame_id=(
                        contract_bundle.attachment.execution_reference_frame_id
                    ),
                    reference_frame_id=(
                        placement_registration.profile.reference_frame_id
                    ),
                    maximum_measurement_age_ms=(
                        contract_bundle.convergence_profile.max_measurement_age_ms
                    ),
                    current_station_revision=lambda: (
                        producer.current_station_revision
                    ),
                    clock=clock,
                ),
            )
            placement_executor = build_placement_executor(
                registration=placement_registration,
                provider=placement_provider,
                current_station_revision=lambda: producer.current_station_revision,
                max_retained_operations=max_retained_operations,
                clock=clock,
            )
            execution_port_by_route_hash[
                placement_registration.skill.routes[0].semantic_content_hash
            ] = placement_executor
        relation_registration = contract_bundle.tool_object_relation_registration
        hover_registration = contract_bundle.tool_hover_registration
        if relation_registration is not None or hover_registration is not None:
            if tool_entity_options is None:
                raise ValueError("Real tool-entity Routes require their composition options.")
            from agent_gateway.profiles.tool_entity import (
                build_tool_hover_executor,
                build_tool_object_relation_executor,
            )

            from robot_motion.tool_entity_scene import RELATION_MAXIMUM_DEPTH_M

            from .tool_entity import RealToolScenePort

            tool_scene_port = RealToolScenePort(perception_provider=perception_provider,
                tool_tcp_poses=tool_entity_options.tool_tcp_poses,
                public_from_geometry=tool_entity_options.public_from_geometry)
        if relation_registration is not None:
            if segmentation_provider is None or back_projection_registration is None:
                raise ValueError("Real tool-object relations require segmentation and aligned RGB-D.")
            tool_object_relation_executor = build_tool_object_relation_executor(
                registration=relation_registration, scene_port=tool_scene_port,
                segmentation_provider=segmentation_provider,
                tool_entity_by_side=tool_entity_options.tool_entity_by_side,
                maximum_depth_m=min(RELATION_MAXIMUM_DEPTH_M,
                                    back_projection_registration.profile.maximum_depth_m),
                current_station_revision=lambda: producer.current_station_revision,
                clock=clock, max_retained_operations=max_retained_operations)
            execution_port_by_route_hash[relation_registration.entry.route.semantic_content_hash] = (
                tool_object_relation_executor)
        if contract_bundle.object_motion_profile is not None:
            if (tool_entity_options is None or segmentation_provider is None
                    or relation_registration is None):
                raise ValueError("Real object motion needs tool-object relations, segmentation and aligned RGB-D.")
            from .held_object import RealHeldObjectObserver
            from .object_motion import RealHeldObjectPort

            def _gripper_reader(port_item):
                return port_item.observe_state

            def _gripper_command(port_item):
                return lambda: (port_item.last_command_target_close, port_item.last_command_attempt_ref)

            gripper_readers = {
                component.split("_", 1)[0]: item for component, item in gripper_ports.items()
                if component.split("_", 1)[0] in {"left", "right"}
                and callable(getattr(item, "observe_state", None))}
            object_motion_side = contract_bundle.object_motion_profile.component_id.split("_", 1)[0]
            if object_motion_side not in gripper_readers:
                raise ValueError("Real object motion needs this arm's DMTAC gripper state.")
            held_object_port = RealHeldObjectPort(
                observer=RealHeldObjectObserver(
                    scene_port=tool_scene_port, segmentation_provider=segmentation_provider,
                    tool_entity_by_side=tool_entity_options.tool_entity_by_side,
                    roles=tuple(back_projection_registration.profile.roles),
                    gripper_state_by_side={side: _gripper_reader(item) for side, item in gripper_readers.items()},
                    gripper_command_by_side={
                        side: _gripper_command(item) for side, item in gripper_readers.items()
                        if hasattr(item, "last_command_target_close")
                        and hasattr(item, "last_command_attempt_ref")},
                    maximum_depth_m=min(RELATION_MAXIMUM_DEPTH_M,
                                        back_projection_registration.profile.maximum_depth_m)),
                public_from_geometry=tool_entity_options.public_from_geometry,
                current_station_revision=lambda: producer.current_station_revision, clock=clock)
        support_release_registration = contract_bundle.support_release_registration
        if support_release_registration is not None:
            support_release = (None if tool_entity_options is None
                               else tool_entity_options.support_release)
            if held_object_port is None or support_release is None:
                raise ValueError("Real support release queries need held-object motion and their planner.")
            from .support_release_query import (
                RealSupportReleaseQuery,
                build_real_support_release_executor,
            )

            object_motion_profile = contract_bundle.object_motion_profile
            support_release_executor = build_real_support_release_executor(
                registration=support_release_registration,
                query_port=RealSupportReleaseQuery(
                    registration=support_release_registration, scene_port=tool_scene_port,
                    segmentation_provider=segmentation_provider, held_port=held_object_port,
                    planner=support_release["planner"], object_id=object_motion_profile.object_id,
                    public_from_geometry=tool_entity_options.public_from_geometry,
                    object_workspace_bounds=object_motion_profile.object_workspace_bounds,
                    arm_base_xy_m=support_release["arm_base_xy_m"],
                    maximum_depth_m=min(RELATION_MAXIMUM_DEPTH_M,
                                        back_projection_registration.profile.maximum_depth_m),
                    clock=clock),
                current_station_revision=lambda: producer.current_station_revision,
                clock=clock, max_retained_operations=max_retained_operations)
            execution_port_by_route_hash[support_release_registration.entry.route.semantic_content_hash] = (
                support_release_executor)
        if hover_registration is not None:
            if anchor_registration is None or tool_entity_options.hover_planner is None:
                raise ValueError("Real tool hover queries require surface anchor v2 and their planner.")
            if (tool_entity_options.hover_planner.max_inter_camera_skew_sec
                    != perception_provider.max_inter_camera_skew_ms / 1000.0):
                raise ValueError("Real tool hover queries must admit scenes like the safe move.")

            def lookup_tool_anchor(identifier):
                source_request = service.operation_store.request_for_operation(identifier)
                source_operation = service.operation_store.get(identifier)
                return (None if source_request is None or source_operation is None
                        else (source_request, source_operation))

            tool_hover_executor = build_tool_hover_executor(
                registration=hover_registration, capture_registration=anchor_registration,
                scene_port=tool_scene_port, planner_port=tool_entity_options.hover_planner,
                tool_entity_by_side=tool_entity_options.tool_entity_by_side, store=artifact_store,
                lookup_operation=lookup_tool_anchor,
                current_station_revision=lambda: producer.current_station_revision, clock=clock,
                clock_domain_id=contract_bundle.attachment.clock_domain_id,
                max_retained_operations=max_retained_operations)
            execution_port_by_route_hash[hover_registration.entry.route.semantic_content_hash] = (
                tool_hover_executor)

        if contract_bundle.rgbd_geometry_registrations:
            rgbd_geometry_executors = build_rgbd_geometry_executors(
                registrations=contract_bundle.rgbd_geometry_registrations,
                perception_provider=perception_provider, frame_binding=registration.frame_binding,
                artifact_store=artifact_store, runtime_session_id=contract_bundle.attachment.runtime_session_id,
                current_station_revision=lambda: producer.current_station_revision,
                max_retained_operations=max_retained_operations, clock=clock,
                source_request_for_operation=lambda identifier: service.operation_store.request_for_operation(identifier))
            execution_port_by_route_hash.update(
                (item.skill.routes[0].semantic_content_hash, executor)
                for item, executor in zip(contract_bundle.rgbd_geometry_registrations,
                                          rgbd_geometry_executors, strict=True))
        if contract_bundle.localization_registration is not None:
            assert localization_provider is not None
            assert station_point_grasp_planner is not None
            localization_registration = contract_bundle.localization_registration
            point_grasp_registration = contract_bundle.station_point_grasp_registration
            render_registration = (
                contract_bundle.station_point_grasp_render_registration
            )
            assert point_grasp_registration is not None
            assert render_registration is not None
            frame_binding = contract_bundle.perception_capture.frame_binding
            if frame_binding is None:
                raise ValueError("Station point-grasp requires a v2 frame binding.")
            localization_executor = build_real_object_localization_executor(
                registration=localization_registration,
                provider=localization_provider,
                artifact_store=artifact_store,
                runtime_session_id=contract_bundle.attachment.runtime_session_id,
                current_station_revision=lambda: producer.current_station_revision,
                max_retained_operations=max_retained_operations,
                clock=clock,
            )
            observation_reader = None
            if point_grasp_registration.observation_registration is not None:
                observation_registration = point_grasp_registration.observation_registration
                grasp_observation_executor = build_grasp_observation_executor(
                    registration=observation_registration, perception_provider=perception_provider,
                    frame_binding=frame_binding, artifact_store=artifact_store,
                    runtime_session_id=contract_bundle.attachment.runtime_session_id,
                    current_station_revision=lambda: producer.current_station_revision,
                    max_retained_operations=max_retained_operations, clock=clock,
                    non_motion_read_window=non_motion_read_window)
                execution_port_by_route_hash[observation_registration.skill.routes[0].semantic_content_hash] = grasp_observation_executor

                def observation_reader(operation_id, revision):
                    return read_retained_grasp_observation(
                        executor=grasp_observation_executor, operation_id=operation_id, artifact_store=artifact_store,
                        runtime_session_id=contract_bundle.attachment.runtime_session_id,
                        expected_station_revision=revision, frame_binding_hash=frame_binding.semantic_content_hash)

            station_point_grasp_provider = StationBoundPointGraspProvider(
                planner=station_point_grasp_planner,
                registration=point_grasp_registration,
                perception_provider=perception_provider,
                frame_binding=frame_binding,
                current_station_revision=lambda: producer.current_station_revision,
                arm_reachability_evaluator=(station_point_grasp_reachability_evaluator),
                motion_port=port,
                captured_robot_observation_filter=(
                    captured_robot_observation_filter
                ),
                non_motion_read_window=non_motion_read_window,
                observation_reader=observation_reader,
            )
            station_point_grasp_executor = build_station_bound_point_grasp_executor(
                registration=point_grasp_registration,
                provider=station_point_grasp_provider,
                artifact_store=artifact_store,
                runtime_session_id=contract_bundle.attachment.runtime_session_id,
                current_station_revision=lambda: producer.current_station_revision,
                max_retained_operations=max_retained_operations,
                clock=clock,
            )
            station_point_grasp_render_provider = StationBoundPointGraspRenderProvider(
                registration=render_registration,
                point_grasp_registration=point_grasp_registration,
                point_grasp_executor=station_point_grasp_executor,
                perception_provider=perception_provider,
                frame_binding=frame_binding,
                solid_geometry=getattr(
                    station_point_grasp_planner, "solid_geometry", None
                ),
                runtime_session_id=contract_bundle.attachment.runtime_session_id,
                current_station_revision=lambda: producer.current_station_revision,
            )
            station_point_grasp_render_executor = (
                build_station_bound_point_grasp_render_executor(
                    registration=render_registration,
                    provider=station_point_grasp_render_provider,
                    artifact_store=artifact_store,
                    runtime_session_id=contract_bundle.attachment.runtime_session_id,
                    current_station_revision=lambda: producer.current_station_revision,
                    max_retained_operations=max_retained_operations,
                    clock=clock,
                )
            )
            plan_registration = contract_bundle.grasp_approach_plan_registration
            if plan_registration is not None:
                assert grasp_approach_plan_store is not None
                assert grasp_trajectory_planner is not None
                grasp_approach_plan_provider = RealGraspApproachPlanProvider(
                    registration=plan_registration,
                    point_grasp_registration=point_grasp_registration,
                    point_grasp_executor=station_point_grasp_executor,
                    motion_port=port,
                    frame_graph=contract_bundle.frame_graph,
                    grasp_trajectory_planner=grasp_trajectory_planner,
                    visible_scene_reader=build_grasp_visible_scene_reader(
                        artifact_store=artifact_store,
                        frame_graph=contract_bundle.frame_graph,
                        execution_frame_id=plan_registration.profile.execution_reference_frame_id,
                        captured_robot_observation_filter=(
                            captured_robot_observation_filter
                        ),
                    ),
                    plan_store=grasp_approach_plan_store,
                    runtime_session_id=(contract_bundle.attachment.runtime_session_id),
                    current_station_revision=(
                        lambda: producer.current_station_revision
                    ),
                    clock=clock,
                    non_motion_read_window=non_motion_read_window,
                )
                grasp_approach_plan_executor = build_real_grasp_approach_plan_executor(
                    registration=plan_registration,
                    provider=grasp_approach_plan_provider,
                    runtime_session_id=(contract_bundle.attachment.runtime_session_id),
                    current_station_revision=(
                        lambda: producer.current_station_revision
                    ),
                    max_retained_operations=max_retained_operations,
                    clock=clock,
                )
                fast_registration = contract_bundle.grasp_fast_plan_registration
                if fast_registration is not None:
                    if grasp_observation_executor is None:
                        raise ValueError("Fast grasp planning requires the retained observation executor.")
                    fast_provider = RealFastGraspPlanProvider(
                        registration=fast_registration, point_provider=station_point_grasp_provider,
                        paired_provider=grasp_approach_plan_provider,
                        observation_executor=grasp_observation_executor)
                    grasp_fast_plan_executor = build_real_fast_grasp_plan_executor(
                        registration=fast_registration, provider=fast_provider,
                        runtime_session_id=contract_bundle.attachment.runtime_session_id,
                        current_station_revision=lambda: producer.current_station_revision,
                        max_retained_operations=max_retained_operations, clock=clock,
                        artifact_store=artifact_store)
                    execution_port_by_route_hash[
                        fast_registration.skill.routes[0].semantic_content_hash] = grasp_fast_plan_executor
            execution_port_by_route_hash.update(
                {
                    localization_registration.skill.routes[
                        0
                    ].semantic_content_hash: localization_executor,
                    point_grasp_registration.skill.routes[
                        0
                    ].semantic_content_hash: station_point_grasp_executor,
                    render_registration.skill.routes[
                        0
                    ].semantic_content_hash: station_point_grasp_render_executor,
                    **(
                        {}
                        if plan_registration is None
                        else {
                            plan_registration.skill.routes[0].semantic_content_hash: (
                                grasp_approach_plan_executor
                            )
                        }
                    ),
                }
            )
    if transport_enabled:
        from contextlib import nullcontext

        transport_registration = contract_bundle.object_transport_registration
        object_transport_executor = build_object_transport_executor(
            registration=transport_registration, observe_held_object=held_object_observer,
            planner_factory=object_transport_planner_factory, artifact_store=artifact_store,
            runtime_session_id=contract_bundle.attachment.runtime_session_id,
            current_station_revision=lambda: producer.current_station_revision,
            max_retained_operations=max_retained_operations, clock=clock,
            non_motion_read_window=non_motion_read_window or nullcontext)
        execution_port_by_route_hash[transport_registration.skill.routes[0].semantic_content_hash] = object_transport_executor
    registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=entry,
                execution_port=execution_port_by_route_hash[
                    entry.route.semantic_content_hash
                ],
            )
            for entry in contract_bundle.route_manifest.skill_routes
        ),
    )
    access_policy = AgentSkillAccessPolicy(
        access_policy_id=(
            "agent-access:real-motion-primitives:"
            + contract_bundle.route_manifest.runtime_session_id
        ),
        allowed_client_ids=clients,
        allowed_modes=("real_active",),
        max_request_timeout_ms=max(
            entry.route.max_operation_duration_ms
            for entry in contract_bundle.route_manifest.skill_routes
        ),
        max_grant_lifetime_sec=30.0,
        max_grant_invocations=1,
    )
    domain_executor = next(iter(executors.values()), None)
    if domain_executor is None:
        domain_executor = _RealAgentDomainHoldPort(
            port_id=(
                "agent-gateway:real-domain-hold:"
                + contract_bundle.route_manifest.runtime_session_id
            ),
            domain_hold=domain_hold,
            clock=clock,
        )
    emergency_stop_port = (
        None
        if emergency_stop is None
        else RealAgentDomainEmergencyStopPort(
            gripper_ports=gripper_ports,
            port_id=(
                "agent-gateway:real-domain-emergency-stop:"
                + contract_bundle.route_manifest.runtime_session_id
            ),
            emergency_stop=emergency_stop,
            clock=clock,
        )
    )
    if artifact_store is None:
        protocol_runtime = AgentGatewayProtocolRuntime.control_plane_only(
            manifest=contract_bundle.route_manifest,
            clock=clock,
        )
    else:
        capabilities = AgentGatewaySessionCapabilities(
            runtime_session_id=contract_bundle.attachment.runtime_session_id,
            artifact_transfer_enabled=True,
            artifact_access_mechanisms=("chunked_rpc", "local_path"),
            max_control_request_bytes=2 * 1024 * 1024,
            max_control_response_bytes=2 * 1024 * 1024,
            max_artifact_bytes=artifact_store.max_artifact_bytes,
            max_artifact_chunk_bytes=artifact_store.max_chunk_bytes,
            max_artifact_refs_per_operation=8,
            max_concurrent_uploads=8,
            max_concurrent_downloads=8,
            artifact_store_quota_bytes=artifact_store.quota_bytes,
            event_payload_max_bytes=65_536,
            event_retention_count=4096,
        )
        broker = AgentArtifactTransferBroker(
            store=artifact_store,
            session_capabilities=capabilities,
            max_chunk_bytes=artifact_store.max_chunk_bytes,
        )
        protocol_runtime = AgentGatewayProtocolRuntime(
            manifest=contract_bundle.route_manifest,
            io_catalog=contract_bundle.io_catalog,
            session_capabilities=capabilities,
            event_journal=BoundedAgentGatewayEventJournal(
                runtime_session_id=contract_bundle.attachment.runtime_session_id,
                max_events=capabilities.event_retention_count,
                event_payload_max_bytes=capabilities.event_payload_max_bytes,
            ),
            artifact_store=artifact_store,
            artifact_transfer_broker=broker,
            clock=clock,
        )
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=access_policy,
        executor_registry=registry,
        domain_control_port=domain_executor,
        domain_emergency_stop_port=emergency_stop_port,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=max_retained_operations,
            max_admissions_by_route_hash=(
                {}
                if max_session_admissions_per_route is None
                else {
                    entry.route.semantic_content_hash: (
                        max_session_admissions_per_route
                    )
                    for entry in contract_bundle.route_manifest.skill_routes
                }
            ),
        ),
        protocol_runtime=protocol_runtime,
        clock=clock,
        clock_domain_id=contract_bundle.attachment.clock_domain_id,
    )
    hardware_view_provider = _real_hardware_view_provider(
        gripper_ports=gripper_ports,
        contract_bundle=contract_bundle,
        port=port,
        perception_provider=perception_provider,
        localization_provider=localization_provider,
        segmentation_provider=segmentation_provider,
        operation_store=service.operation_store,
        clock=clock,
    )
    return RealAgentPrimitiveGatewaySession(
        gripper_port=gripper_port,
        gripper_ports=gripper_ports,
        contract_bundle=contract_bundle,
        port=port,
        observation_producer=producer,
        executors_by_skill_id=executors,
        emergency_stop_port=emergency_stop_port,
        state_executor=state_executor,
        perception_provider=perception_provider,
        rgb_executor=rgb_executor,
        depth_executor=depth_executor,
        back_projection_provider=back_projection_provider,
        back_projection_executor=back_projection_executor,
        localization_provider=localization_provider,
        localization_executor=localization_executor,
        segmentation_provider=segmentation_provider,
        segmentation_executor=segmentation_executor,
        surface_anchor_executor=surface_anchor_executor,
        tool_object_relation_executor=tool_object_relation_executor,
        tool_hover_executor=tool_hover_executor,
        support_release_executor=support_release_executor,
        placement_provider=placement_provider,
        placement_executor=placement_executor,
        station_point_grasp_provider=station_point_grasp_provider,
        station_point_grasp_executor=station_point_grasp_executor,
        station_point_grasp_render_provider=station_point_grasp_render_provider,
        station_point_grasp_render_executor=station_point_grasp_render_executor,
        grasp_approach_plan_store=grasp_approach_plan_store,
        grasp_approach_plan_provider=grasp_approach_plan_provider,
        grasp_approach_plan_executor=grasp_approach_plan_executor,
        grasp_fast_plan_executor=grasp_fast_plan_executor,
        artifact_store=artifact_store,
        owns_artifact_store=owns_artifact_store,
        hardware_view_provider=hardware_view_provider,
        service=service,
        grasp_observation_executor=grasp_observation_executor,
        object_transport_executor=object_transport_executor,
        spatial_goal_executor=spatial_goal_executor,
        tcp_pose_comparison_executor=tcp_pose_comparison_executor,
        rgbd_geometry_executors=rgbd_geometry_executors,
    )


__all__ = [
    "RealAgentPrimitiveGatewaySession",
    "RealAgentPrimitiveObservationProducer",
    "assemble_real_agent_primitive_gateway_session",
]
