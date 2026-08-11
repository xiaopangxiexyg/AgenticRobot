"""Low-rate TCP planning snapshots for the exact CubeLift Sim-HIL Route.

The producer reads one application-locked passive cut and publishes only a
bounded tool-TCP pose summary plus lower-owned identities.  It never starts a
source, advances physics, opens publication, authorizes motion, or substitutes
for the lower runtime's fresh pre-dispatch measurements.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from collection_runtime.planning_history import (
    PlanningHistoryRecord,
    PlanningHistorySourceBinding,
)
from control_station.agent_gateway import (
    AgentGatewayObservationOutcome,
    AgentGatewayService,
    AgentObservationSnapshot,
    AgentStationRevisionTracker,
    validate_agent_state_result_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionMeasurementObservation,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.runtime.observation_cut import (
    ObservationCutRecord,
    ObservationSampleRef,
)
from sim.mujoco.deploy.sim_hil_target import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)

from .agent_motion_deployment import MujocoAgentTcpPoseContractBundle
from .agent_skill_deployment import MujocoCubeLiftAgentSkillContractBundle
from .application import MujocoSimHILPassiveApplicationCut

MUJOCO_CUBE_LIFT_AGENT_TCP_SUMMARY_SCHEMA_V1 = (
    "dexteleop.agent_observation.mujoco_tcp_pose_summary.v1"
)
MUJOCO_CUBE_LIFT_AGENT_TCP_MEASUREMENT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_observation.mujoco_tcp_pose_measurement.v1"
)
MUJOCO_CUBE_LIFT_AGENT_SNAPSHOT_ID_HASH_DOMAIN_V1 = (
    "dexteleop.agent_observation.mujoco_cube_lift_snapshot_id.v1"
)
MUJOCO_AGENT_TCP_POSE_SAMPLE_ID_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.mujoco_tcp_pose_sample_id.v1"
)
MUJOCO_AGENT_TCP_POSE_OBSERVATION_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": "dexteleop.agent_state.mujoco_tcp_pose_observation.v1",
        "stream_id": "robot_state.right_arm.tool_tcp_pose",
        "payload": "xyz+quat_xyzw",
        "measurement_owner": "mujoco_scene_ground_truth",
    },
    domain="dexteleop.agent_state.observation_contract.v1",
)
MUJOCO_AGENT_TCP_POSE_IO_LAYOUT_HASH_V1 = canonical_content_hash(
    {
        "layout_id": "dexteleop.agent_state.tcp_pose_read.v1",
        "input_streams": ("robot_state.right_arm.tool_tcp_pose",),
        "output": "measured_tcp_pose_projection",
    },
    domain="dexteleop.agent_state.io_layout.v1",
)


@dataclass(frozen=True)
class MujocoAgentTcpPoseStateRead:
    """One snapshot and state result produced from the same passive cut."""

    snapshot: AgentObservationSnapshot
    execution_result: Mapping[str, Any]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class MujocoAgentTcpPoseHistoryCapture:
    """One lower-owned typed cut plus its portable disk-history projection."""

    snapshot: AgentObservationSnapshot
    observation_cut: ObservationCutRecord
    measurement: ExecutionMeasurementObservation
    history_record: PlanningHistoryRecord


@dataclass(frozen=True)
class _MujocoAgentObservationCapture:
    snapshot: AgentObservationSnapshot
    observation_cut: ObservationCutRecord | None
    measurement: ExecutionMeasurementObservation | None


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _optional_identifier(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required(value, field_name=field_name)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


class MujocoCubeLiftAgentObservationProducer:
    """Single-session snapshot/revision owner for one constructed graph."""

    def __init__(
        self,
        *,
        application: Any,
        contract_bundle: (
            MujocoCubeLiftAgentSkillContractBundle
            | MujocoAgentTcpPoseContractBundle
        ),
        revision_tracker: AgentStationRevisionTracker | None = None,
    ) -> None:
        if not isinstance(
            contract_bundle,
            (
                MujocoCubeLiftAgentSkillContractBundle,
                MujocoAgentTcpPoseContractBundle,
            ),
        ):
            raise TypeError("contract_bundle must be a supported typed bundle.")
        capture_cut = getattr(application, "capture_passive_application_cut", None)
        if not callable(capture_cut):
            raise TypeError(
                "application must expose capture_passive_application_cut()."
            )
        tracker = (
            AgentStationRevisionTracker()
            if revision_tracker is None
            else revision_tracker
        )
        if not isinstance(tracker, AgentStationRevisionTracker):
            raise TypeError("revision_tracker must be typed.")

        runtime = getattr(application, "runtime", None)
        runtime_config = getattr(runtime, "config", None)
        binding = contract_bundle.deployment_binding
        if (
            getattr(runtime_config, "runtime_session_id", None)
            != binding.runtime_session_id
            or getattr(runtime_config, "control_domain_id", None)
            != binding.control_domain_id
        ):
            raise ValueError("Application runtime does not match deployment binding.")
        clock = getattr(runtime, "clock", None)
        if not callable(clock):
            raise TypeError("application runtime must expose its monotonic clock.")
        clock_domain_id = _required(
            getattr(runtime_config, "clock_domain_id", None),
            field_name="runtime clock_domain_id",
        )

        observation_plan = getattr(
            runtime_config,
            "execution_observation_plan",
            None,
        )
        if (
            not bool(getattr(observation_plan, "enabled", False))
            or getattr(observation_plan, "measurement_mode", None) == "off"
            or tuple(getattr(observation_plan, "component_ids", ()))
            != ("right_arm",)
            or getattr(observation_plan, "semantic_content_hash", None)
            != binding.observation_plan_hash
        ):
            raise ValueError("Observation producer requires the exact right TCP plan.")
        observation_binding = getattr(runtime, "execution_observation_binding", None)
        observation_binding_id = _required(
            getattr(observation_binding, "binding_id", None),
            field_name="execution observation binding_id",
        )
        if getattr(observation_binding, "plan_hash", None) != (
            binding.observation_plan_hash
        ):
            raise ValueError("Execution observation binding uses another plan.")
        observation_capability = getattr(
            runtime,
            "execution_observation_capability",
            None,
        )
        representation_by_component = dict(
            getattr(observation_capability, "representation_by_component", {})
        )
        schema_hash_by_component = dict(
            getattr(observation_capability, "schema_hash_by_component", {})
        )
        pose_schema_hash = (
            _required(
                schema_hash_by_component["right_arm"],
                field_name="right_arm pose schema hash",
            )
            if representation_by_component.get("right_arm") == "cartesian_pose"
            and "right_arm" in schema_hash_by_component
            else None
        )

        target_profile = getattr(application, "target_frame_profile", None)
        frame_graph = getattr(target_profile, "frame_graph", None)
        if getattr(frame_graph, "semantic_content_hash", None) != (
            binding.frame_graph_hash
        ):
            raise ValueError("Observation producer frame graph mismatch.")
        reference_frame_by_component = dict(
            getattr(target_profile, "reference_frame_by_component", {})
        )
        if "right_arm" not in reference_frame_by_component:
            raise ValueError("Target profile lacks the right_arm reference frame.")
        reference_frame_id = _required(
            reference_frame_by_component["right_arm"],
            field_name="right_arm reference frame",
        )
        frame_spec = getattr(frame_graph, "frame_by_id", {}).get(reference_frame_id)
        reference_frame_spec_hash = _required(
            getattr(frame_spec, "semantic_content_hash", None),
            field_name="right_arm reference frame spec hash",
        )

        tool_profile = getattr(application, "tool_frame_profile", None)
        if getattr(tool_profile, "semantic_content_hash", None) != (
            binding.tool_config_hash
        ):
            raise ValueError("Observation producer tool profile mismatch.")
        tool_binding = getattr(tool_profile, "binding_by_component", {}).get(
            "right_arm"
        )
        if getattr(tool_binding, "controlled_frame_id", None) != "tool_tcp":
            raise ValueError("Observation producer requires right_arm tool_tcp.")

        if isinstance(contract_bundle, MujocoCubeLiftAgentSkillContractBundle):
            composition = getattr(application, "composition", None)
            policy_binding = getattr(
                getattr(composition, "policy_input_adapter", None),
                "binding",
                None,
            )
            source_binding_id = _required(
                getattr(policy_binding, "source_slot_id", None),
                field_name="policy source slot",
            )
            source_binding_hash = _required(
                getattr(policy_binding, "semantic_content_hash", None),
                field_name="policy source binding hash",
            )
            producer_epoch = getattr(policy_binding, "producer_epoch", None)
            producer_epoch_label = "policy producer_epoch"
        else:
            source_slots = tuple(contract_bundle.station_manifest.source_slots)
            direct = getattr(
                runtime_config,
                "direct_component_proposal_source",
                None,
            )
            if (
                len(source_slots) != 1
                or source_slots[0].source_kind != "agent_skill"
                or getattr(direct, "producer_kind", None) != "agent_skill"
                or source_slots[0].producer_id
                != getattr(direct, "producer_id", None)
            ):
                raise ValueError(
                    "Agent motion observation source does not match its Station slot."
                )
            source_binding_id = source_slots[0].source_slot_id
            source_binding_hash = source_slots[0].semantic_content_hash
            producer_epoch = getattr(direct, "producer_epoch", None)
            producer_epoch_label = "agent_skill producer_epoch"
        if (
            isinstance(producer_epoch, bool)
            or not isinstance(producer_epoch, int)
            or producer_epoch < 0
        ):
            raise ValueError(f"{producer_epoch_label} must be non-negative.")

        self.application = application
        self.contract_bundle = contract_bundle
        self.revision_tracker = tracker
        self.clock = clock
        self.clock_domain_id = clock_domain_id
        self.observation_binding_id = observation_binding_id
        self.pose_schema_hash = pose_schema_hash
        self.reference_frame_id = reference_frame_id
        self.reference_frame_spec_hash = reference_frame_spec_hash
        self.source_binding_id = source_binding_id
        self.source_binding_hash = source_binding_hash
        self.producer_epoch = producer_epoch
        history_source_binding = getattr(
            contract_bundle,
            "history_source_binding",
            None,
        )
        if history_source_binding is not None:
            if not isinstance(
                history_source_binding,
                PlanningHistorySourceBinding,
            ):
                raise TypeError("history_source_binding must be typed.")
            if (
                history_source_binding.runtime_session_id
                != binding.runtime_session_id
                or history_source_binding.clock_domain_id != clock_domain_id
                or history_source_binding.channels != ("tcp_pose",)
            ):
                raise ValueError(
                    "History source binding does not match the observation owner."
                )
        self.history_source_binding = history_source_binding
        self._lock = threading.Lock()
        self._next_snapshot_sequence = 0

    @property
    def current_station_revision(self) -> int:
        return self.revision_tracker.current_revision

    def _capture_scene(self, scene: Any) -> dict[str, Any]:
        finite_state = bool(scene.state_is_finite())
        pose_by_component: dict[str, list[float]] = {}
        if finite_state:
            pose_by_side = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                scene,
                self.application.task_config,
            )
            vector = pose_matrix_to_xyzquat(pose_by_side["right"])
            if vector is not None:
                values = [float(item) for item in vector]
                if len(values) == 7 and all(math.isfinite(item) for item in values):
                    pose_by_component["right_arm"] = values
        return {
            "finite_state": finite_state,
            "pose_xyz_quat_xyzw_by_component": pose_by_component,
        }

    def _measurement_summary(
        self,
        cut: MujocoSimHILPassiveApplicationCut,
    ) -> tuple[dict[str, Any], bool, str]:
        projection = cut.scene_projection
        raw_pose_by_component = (
            {}
            if not isinstance(projection, Mapping)
            else projection.get("pose_xyz_quat_xyzw_by_component", {})
        )
        pose_by_component: dict[str, list[float]] = {}
        if isinstance(raw_pose_by_component, Mapping) and set(
            raw_pose_by_component
        ) == {"right_arm"}:
            raw_pose = raw_pose_by_component["right_arm"]
            if isinstance(raw_pose, (tuple, list)) and len(raw_pose) == 7:
                try:
                    pose = [float(item) for item in raw_pose]
                except (TypeError, ValueError):
                    pose = []
                quaternion_norm = math.sqrt(
                    sum(item * item for item in pose[3:])
                ) if len(pose) == 7 else math.nan
                if (
                    len(pose) == 7
                    and all(math.isfinite(item) for item in pose)
                    and abs(quaternion_norm - 1.0) <= 1e-5
                ):
                    pose_by_component["right_arm"] = pose
        measurement_valid = bool(
            isinstance(projection, Mapping)
            and projection.get("finite_state") is True
            and set(pose_by_component) == {"right_arm"}
            and cut.lifecycle_state not in {"closed", "restart_required"}
        )
        payload = {
            "schema": MUJOCO_CUBE_LIFT_AGENT_TCP_SUMMARY_SCHEMA_V1,
            "status": "valid" if measurement_valid else "invalid",
            "source_kind": "mujoco_scene_ground_truth",
            "component_ids": ["right_arm"],
            "reference_frame_id_by_component": {
                "right_arm": self.reference_frame_id,
            },
            "reference_frame_spec_hash_by_component": {
                "right_arm": self.reference_frame_spec_hash,
            },
            "controlled_frame_id_by_component": {
                "right_arm": "tool_tcp",
            },
            "pose_xyz_quat_xyzw_by_component": pose_by_component,
            "observation_binding_id": self.observation_binding_id,
            "observation_plan_hash": (
                self.contract_bundle.deployment_binding.observation_plan_hash
            ),
            "measured_at": cut.captured_at,
        }
        measurement_hash = canonical_content_hash(
            payload,
            domain=MUJOCO_CUBE_LIFT_AGENT_TCP_MEASUREMENT_HASH_DOMAIN_V1,
        )
        return (
            {**payload, "measurement_hash": measurement_hash},
            measurement_valid,
            measurement_hash,
        )

    def _typed_measurement_cut(
        self,
        *,
        cut: MujocoSimHILPassiveApplicationCut,
        summary: Mapping[str, Any],
        measurement_hash: str,
        measurement_valid: bool,
        sequence: int,
    ) -> tuple[
        ObservationCutRecord | None,
        ExecutionMeasurementObservation | None,
    ]:
        control_cut = cut.control_cut
        if (
            not measurement_valid
            or control_cut is None
            or self.pose_schema_hash is None
        ):
            return None, None
        if control_cut.clock_domain_id != self.clock_domain_id:
            raise ValueError("Passive TCP pose cut uses another clock domain.")
        pose_by_component = summary.get(
            "pose_xyz_quat_xyzw_by_component"
        )
        if not isinstance(pose_by_component, Mapping) or set(
            pose_by_component
        ) != {"right_arm"}:
            raise ValueError("Passive TCP pose measurement is incomplete.")
        sample_identity_hash = canonical_content_hash(
            {
                "runtime_session_id": control_cut.runtime_session_id,
                "control_domain_id": control_cut.control_domain_id,
                "sequence": sequence,
                "measured_at": cut.captured_at,
                "available_at": control_cut.cutoff_at,
                "source_measurement_hash": measurement_hash,
            },
            domain=MUJOCO_AGENT_TCP_POSE_SAMPLE_ID_HASH_DOMAIN_V1,
        )
        measurement_id = f"mujoco-tcp-pose-measurement:{sample_identity_hash}"
        sample = ObservationSampleRef(
            stream_id="robot_state.right_arm.tool_tcp_pose",
            sample_id=measurement_id,
            sequence=sequence,
            measured_at=cut.captured_at,
            measured_clock_domain_id=self.clock_domain_id,
            available_at=control_cut.cutoff_at,
            available_clock_domain_id=self.clock_domain_id,
            payload_object_id=f"mujoco-tcp-pose:{measurement_hash}",
            valid=True,
            stale=False,
            control_admission_sequence=(
                control_cut.cutoff_admission_sequence
            ),
            content_metadata={
                "representation": "cartesian_pose",
                "controlled_frame_id": "tool_tcp",
                "reference_frame_id": self.reference_frame_id,
                "source_kind": "mujoco_scene_ground_truth",
            },
        )
        observation_cut = ObservationCutRecord.build(
            runtime_session_id=control_cut.runtime_session_id,
            control_cut=control_cut,
            policy_node_id="control-station.agent-state-passive-reader.v1",
            policy_node_epoch=0,
            sequence=sequence,
            built_at=control_cut.cutoff_at,
            built_clock_domain_id=control_cut.clock_domain_id,
            selected_sample_by_stream={sample.stream_id: sample},
            observation_contract_hash=(
                MUJOCO_AGENT_TCP_POSE_OBSERVATION_CONTRACT_HASH_V1
            ),
            policy_io_layout_hash=MUJOCO_AGENT_TCP_POSE_IO_LAYOUT_HASH_V1,
            trigger_reason="agent_gateway_passive_tcp_pose_capture",
        )
        measurement = ExecutionMeasurementObservation(
            measurement_id=measurement_id,
            source_measurement_hash=measurement_hash,
            control_domain_id=control_cut.control_domain_id,
            measured_at=cut.captured_at,
            available_at=control_cut.cutoff_at,
            clock_domain_id=control_cut.clock_domain_id,
            value_by_component={
                "right_arm": pose_by_component["right_arm"],
            },
            representation_by_component={"right_arm": "cartesian_pose"},
            schema_hash_by_component={
                "right_arm": self.pose_schema_hash,
            },
            observation_binding_id=self.observation_binding_id,
            observation_plan_hash=(
                self.contract_bundle.deployment_binding.observation_plan_hash
            ),
        )
        return observation_cut, measurement

    def _capture(
        self,
        *,
        active_operation_id: str | None,
    ) -> _MujocoAgentObservationCapture:
        active_id = _optional_identifier(
            active_operation_id,
            field_name="active_operation_id",
        )
        with self._lock:
            cut = self.application.capture_passive_application_cut(
                scene_projector=self._capture_scene,
            )
            if not isinstance(cut, MujocoSimHILPassiveApplicationCut):
                raise TypeError(
                    "application returned an invalid passive application cut."
                )
            summary, measurement_valid, measurement_hash = (
                self._measurement_summary(cut)
            )
            sequence = self._next_snapshot_sequence
            self._next_snapshot_sequence += 1
            observation_cut, measurement = self._typed_measurement_cut(
                cut=cut,
                summary=summary,
                measurement_hash=measurement_hash,
                measurement_valid=measurement_valid,
                sequence=sequence,
            )
            binding = self.contract_bundle.deployment_binding
            manifest = self.contract_bundle.route_manifest
            control_revision = self.revision_tracker.observe(
                {
                    "runtime_session_id": binding.runtime_session_id,
                    "authority_epoch": cut.authority_epoch,
                    "execution_epoch": cut.execution_epoch,
                    "dispatch_owner_generation": (
                        cut.dispatch_owner_generation
                    ),
                    "lifecycle_state": cut.lifecycle_state,
                    "execution_gate": cut.execution_gate,
                    "execution_gate_epoch": cut.execution_gate_epoch,
                    "active_operation_id": active_id,
                    "source_binding_id": self.source_binding_id,
                    "source_binding_hash": self.source_binding_hash,
                    "producer_epoch": self.producer_epoch,
                    "target_binding_id": binding.target_binding_id,
                    "execution_manifest_hash": binding.execution_manifest_hash,
                    "frame_graph_hash": binding.frame_graph_hash,
                    "tool_config_hash": binding.tool_config_hash,
                    "calibration_hash": binding.calibration_hash,
                    "fault_class": cut.fault_class,
                    "fault_rearmable": cut.fault_rearmable,
                }
            )
            available_at = _finite(
                self.clock(),
                field_name="snapshot.available_at",
            )
            if available_at < cut.captured_at:
                raise RuntimeError("Snapshot clock regressed during capture.")
            freshness_ms = math.ceil(
                (available_at - cut.captured_at) * 1000.0
            )
            snapshot_identity_hash = canonical_content_hash(
                {
                    "runtime_session_id": binding.runtime_session_id,
                    "snapshot_sequence": sequence,
                    "station_revision": control_revision.station_revision,
                    "control_fingerprint_hash": (
                        control_revision.control_fingerprint_hash
                    ),
                    "measurement_hash": measurement_hash,
                    "captured_at": cut.captured_at,
                    "available_at": available_at,
                },
                domain=MUJOCO_CUBE_LIFT_AGENT_SNAPSHOT_ID_HASH_DOMAIN_V1,
            )
            snapshot = AgentObservationSnapshot(
                snapshot_id=f"agent-observation:{snapshot_identity_hash}",
                runtime_session_id=binding.runtime_session_id,
                skill_route_manifest_hash=manifest.semantic_content_hash,
                station_revision=control_revision.station_revision,
                captured_at=cut.captured_at,
                available_at=available_at,
                clock_domain_id=self.clock_domain_id,
                authority_epoch=cut.authority_epoch,
                execution_epoch=cut.execution_epoch,
                dispatch_owner_generation=cut.dispatch_owner_generation,
                lifecycle_state=cut.lifecycle_state,
                execution_gate=cut.execution_gate,
                active_operation_id=active_id,
                source_binding_id=self.source_binding_id,
                producer_epoch=self.producer_epoch,
                target_binding_id=binding.target_binding_id,
                execution_manifest_hash=binding.execution_manifest_hash,
                frame_graph_hash=binding.frame_graph_hash,
                tool_config_hash=binding.tool_config_hash,
                calibration_hash=binding.calibration_hash,
                measurement_summary=summary,
                measurement_freshness_ms=freshness_ms,
                measurement_valid=measurement_valid,
                fault_class=cut.fault_class,
                fault_rearmable=cut.fault_rearmable,
            )
            return _MujocoAgentObservationCapture(
                snapshot=snapshot,
                observation_cut=observation_cut,
                measurement=measurement,
            )

    def capture(
        self,
        *,
        active_operation_id: str | None,
    ) -> AgentObservationSnapshot:
        return self._capture(
            active_operation_id=active_operation_id,
        ).snapshot

    def capture_tcp_pose_state(
        self,
        *,
        active_operation_id: str,
        parameters: Mapping[str, Any],
    ) -> MujocoAgentTcpPoseStateRead:
        """Project one state result from the exact same passive capture cut."""

        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
        capture = self._capture(active_operation_id=active_operation_id)
        observation_cut = capture.observation_cut
        measurement = capture.measurement
        if observation_cut is None or measurement is None:
            raise RuntimeError(
                "A typed passive TCP pose measurement cut is unavailable."
            )
        requested = parameters.get("component_ids")
        if not isinstance(requested, (list, tuple)) or tuple(requested) != (
            "right_arm",
        ):
            raise ValueError("This deployment exposes only right_arm TCP pose.")
        if parameters.get("reference_frame_id") != self.reference_frame_id:
            raise ValueError("Requested reference frame is not deployment-bound.")
        pose = measurement.value_by_component["right_arm"]
        binding = self.contract_bundle.deployment_binding
        result = {
            "observation_cut_id": observation_cut.observation_cut_id,
            "observation_cut_hash": observation_cut.semantic_content_hash,
            "cutoff_at": observation_cut.cutoff_at,
            "cutoff_clock_domain_id": (
                observation_cut.cutoff_clock_domain_id
            ),
            "component_ids": ["right_arm"],
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": binding.frame_graph_hash,
            "tool_config_hash": binding.tool_config_hash,
            "calibration_hash": binding.calibration_hash,
            "measurement_id": measurement.measurement_id,
            "measurement_observation_hash": (
                measurement.semantic_content_hash
            ),
            "source_measurement_hash": measurement.source_measurement_hash,
            "measured_at": measurement.measured_at,
            "available_at": measurement.available_at,
            "clock_domain_id": measurement.clock_domain_id,
            "pose_by_component": {"right_arm": list(pose)},
            "pose_schema_hash_by_component": {
                "right_arm": measurement.schema_hash_by_component["right_arm"],
            },
            "representation_by_component": {
                "right_arm": "cartesian_pose",
            },
            "observation_binding_id": self.observation_binding_id,
            "observation_plan_hash": binding.observation_plan_hash,
        }
        validate_agent_state_result_v1(
            schema_kind="tcp_pose",
            parameters=parameters,
            result=result,
            allowed_component_ids=("right_arm",),
        )
        return MujocoAgentTcpPoseStateRead(
            snapshot=capture.snapshot,
            execution_result=result,
            evidence_refs=(
                observation_cut.observation_cut_id,
                "execution-measurement:" + measurement.semantic_content_hash,
            ),
        )

    def capture_tcp_pose_history_sample(
        self,
        *,
        active_operation_id: str | None,
        boundary_event: str | None = None,
    ) -> MujocoAgentTcpPoseHistoryCapture | None:
        """Read one passive cut and compile it to the bound history substrate."""

        source_binding = self.history_source_binding
        if source_binding is None:
            raise RuntimeError("TCP pose history is not enabled for this contract.")
        capture = self._capture(active_operation_id=active_operation_id)
        observation_cut = capture.observation_cut
        measurement = capture.measurement
        if observation_cut is None or measurement is None:
            return None
        pose = measurement.value_by_component["right_arm"]
        record = PlanningHistoryRecord(
            record_id=(
                "planning-history-record:" + observation_cut.semantic_content_hash
            ),
            source_binding_hash=source_binding.semantic_content_hash,
            runtime_session_id=source_binding.runtime_session_id,
            channel="tcp_pose",
            measured_at=measurement.measured_at,
            available_at=measurement.available_at,
            cutoff_at=observation_cut.cutoff_at,
            payload={
                "measurement_id": measurement.measurement_id,
                "measurement_observation_hash": measurement.semantic_content_hash,
                "source_measurement_hash": measurement.source_measurement_hash,
                "measured_at": measurement.measured_at,
                "available_at": measurement.available_at,
                "clock_domain_id": measurement.clock_domain_id,
                "pose_by_component": {"right_arm": list(pose)},
                "pose_schema_hash_by_component": {
                    "right_arm": measurement.schema_hash_by_component["right_arm"]
                },
                "representation_by_component": {
                    "right_arm": "cartesian_pose"
                },
                "observation_binding_id": self.observation_binding_id,
                "observation_plan_hash": (
                    self.contract_bundle.deployment_binding.observation_plan_hash
                ),
            },
            evidence_refs=(
                observation_cut.observation_cut_id,
                "observation-cut:" + observation_cut.semantic_content_hash,
                "execution-measurement:" + measurement.semantic_content_hash,
            ),
            boundary_event=boundary_event,
        )
        return MujocoAgentTcpPoseHistoryCapture(
            snapshot=capture.snapshot,
            observation_cut=observation_cut,
            measurement=measurement,
            history_record=record,
        )

    def capture_tcp_pose_history_record(
        self,
        *,
        active_operation_id: str | None,
        boundary_event: str | None = None,
    ) -> PlanningHistoryRecord | None:
        """Recorder callback form; it performs one source-owned passive read."""

        capture = self.capture_tcp_pose_history_sample(
            active_operation_id=active_operation_id,
            boundary_event=boundary_event,
        )
        return None if capture is None else capture.history_record

    def capture_and_publish(
        self,
        service: AgentGatewayService,
    ) -> AgentGatewayObservationOutcome:
        if not isinstance(service, AgentGatewayService):
            raise TypeError("service must be an AgentGatewayService.")
        if service.manifest.semantic_content_hash != (
            self.contract_bundle.route_manifest.semantic_content_hash
        ):
            raise ValueError("Gateway service uses another Route manifest.")
        snapshot = self.capture(
            active_operation_id=service.operation_store.active_operation_id,
        )
        return service.publish_observation(
            snapshot=snapshot,
            now=snapshot.available_at,
        )


__all__ = [
    "MUJOCO_AGENT_TCP_POSE_IO_LAYOUT_HASH_V1",
    "MUJOCO_AGENT_TCP_POSE_OBSERVATION_CONTRACT_HASH_V1",
    "MUJOCO_AGENT_TCP_POSE_SAMPLE_ID_HASH_DOMAIN_V1",
    "MUJOCO_CUBE_LIFT_AGENT_SNAPSHOT_ID_HASH_DOMAIN_V1",
    "MUJOCO_CUBE_LIFT_AGENT_TCP_MEASUREMENT_HASH_DOMAIN_V1",
    "MUJOCO_CUBE_LIFT_AGENT_TCP_SUMMARY_SCHEMA_V1",
    "MujocoAgentObservationProducer",
    "MujocoAgentTcpPoseHistoryCapture",
    "MujocoAgentTcpPoseStateRead",
    "MujocoCubeLiftAgentObservationProducer",
]


MujocoAgentObservationProducer = MujocoCubeLiftAgentObservationProducer
