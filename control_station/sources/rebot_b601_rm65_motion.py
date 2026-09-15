"""Single-human B601 residual ingress for the existing RM65 MuJoCo owner.

This binding admits proposals only. The caller owns polling and clutch/recenter;
the target owns lifecycle, canonical IK/safety, cancellation and dispatch. No
shadow evaluation is promoted, and no hardware command is issued here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from control_station.composition.rebot_b601_rm65_residual import (
    B601RM65ResidualSource,
    B601RM65ResidualStep,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    ArmActionSpec,
    ComponentAction,
    ComponentActionSchema,
    ExecutionMeasurementObservation,
)
from robot_motion.action.components import ActionComponentCatalog
from robot_motion.action.proposal import ActionProposalDraft, ProposalAdmissionReceipt
from robot_motion.action.relative import apply_relative_pose_component_value
from robot_motion.pose import pose_delta_metrics
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.runtime.execution_gate import ExecutionGateSnapshot
from robot_motion.transforms import (
    identity_pose,
    parse_pose_matrix,
    validate_pose_matrix,
)
from teleop_runtime.action.adapters import TeleopCommandActionAdapterConfig
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterConfig,
)
from teleop_runtime.action.human_residual import HumanResidualPoseAnchor


class _MotionTargetConfig(Protocol):
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_id: str
    producer_epoch: int
    clock_domain_id: str
    initial_control_mode: str
    enable_replay_authority: bool
    enable_composer_authority: bool
    enable_primary_policy_authority: bool
    enable_component_execution_handoff: bool
    direct_component_proposal_source: object | None


class _MotionTargetPort(Protocol):
    def capture_execution_gate_snapshot(
        self, *, now: float | None = None
    ) -> ExecutionGateSnapshot: ...

    def admit_proposal(
        self, proposal: ActionProposalDraft, *, source_event_id: str
    ) -> ProposalAdmissionReceipt: ...


@runtime_checkable
class B601RM65MotionTarget(Protocol):
    """Existing target authority surface; no concrete backend construction."""

    config: _MotionTargetConfig
    authoritative_port: _MotionTargetPort
    catalog: ActionComponentCatalog
    action_schema: ComponentActionSchema
    composition_port: object | None
    clock: Callable[[], float]

    def transition_execution_gate(self, status: str, *, reason: str) -> object: ...


@dataclass(frozen=True)
class B601RM65MotionAnchor:
    """Measured target baseline bound to one master anchor and gate epoch."""

    master_anchor: HumanResidualPoseAnchor
    target_measurement: ExecutionMeasurementObservation
    target_action: ComponentAction
    gate_epoch: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.master_anchor, HumanResidualPoseAnchor)
            or not isinstance(self.target_measurement, ExecutionMeasurementObservation)
            or not isinstance(self.target_action, ComponentAction)
        ):
            raise TypeError("Motion anchor requires typed master and measured target.")
        if (
            isinstance(self.gate_epoch, bool)
            or not isinstance(self.gate_epoch, int)
            or self.gate_epoch < 0
        ):
            raise ValueError("Motion anchor gate epoch must be non-negative integer.")
        for component_id, value in self.target_action.values_by_component.items():
            if (
                self.target_measurement.schema_hash_by_component.get(component_id)
                != value.payload_schema.semantic_content_hash
                or self.target_measurement.representation_by_component.get(component_id)
                != "cartesian_pose"
                or value.values.tolist()
                != (
                    value.payload_schema.validate_values(
                        self.target_measurement.value_by_component[component_id]
                    ).tolist()
                )
            ):
                raise ValueError(
                    "Target baseline must exactly match the measured pose."
                )


class B601RM65MotionBinding:
    """Default-off, bounded publication to a configured single-human target."""

    def __init__(
        self,
        runtime: B601RM65MotionTarget,
        source: B601RM65ResidualSource,
        *,
        maximum_source_age_sec: float,
        maximum_measurement_age_sec: float,
        maximum_translation_m: float,
        maximum_rotation_rad: float,
        enabled: bool = False,
    ) -> None:
        if not isinstance(runtime, B601RM65MotionTarget):
            raise TypeError(
                "runtime must provide the typed B601 RM65 motion target port."
            )
        if not isinstance(source, B601RM65ResidualSource):
            raise TypeError("source must be a B601RM65ResidualSource.")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool.")
        self.runtime = runtime
        self.source = source
        self.enabled = enabled
        self.limits = tuple(
            float(value)
            for value in (
                maximum_source_age_sec,
                maximum_measurement_age_sec,
                maximum_translation_m,
                maximum_rotation_rad,
            )
        )
        if any(not math.isfinite(value) or value <= 0 for value in self.limits):
            raise ValueError("Motion age and range limits must be finite and positive.")
        config = runtime.config
        if (
            config.initial_control_mode != "human"
            or config.enable_replay_authority
            or config.enable_composer_authority
            or not config.enable_primary_policy_authority
            or config.direct_component_proposal_source is not None
            or runtime.composition_port is not None
            or not config.enable_component_execution_handoff
        ):
            raise ValueError("Motion binding requires an explicit single-human target.")
        if runtime.catalog.action_space != BIMANUAL_EE_POSE_V1:
            raise ValueError("Motion binding requires the canonical RM65 pose target.")
        if config.clock_domain_id != source.pose_source.spec.clock_domain_id:
            raise ValueError(
                "Source and target must share the configured clock domain."
            )
        self.component_id = source.residual_adapter.output_action_schema.component_ids[
            0
        ]
        if self.component_id not in runtime.action_schema.component_ids:
            raise ValueError("Residual component is absent from the target schema.")
        target = runtime.action_schema.payload_schema_by_component[self.component_id]
        residual = (
            source.residual_adapter.output_action_schema.payload_schema_by_component[
                self.component_id
            ]
        )
        if (
            target.reference_frame != residual.reference_frame
            or target.controlled_frame != residual.controlled_frame
            or target.coordinate_names != residual.coordinate_names
            or target.units != residual.units
            or target.shape != (7,)
            or residual.value_semantics != "relative_delta"
            or dict(residual.semantic_parameters)
            != {
                "translation_application": "reference_frame_additive",
                "rotation_application": "baseline_local_right_multiply",
            }
        ):
            raise ValueError("Residual and target pose/frame conventions differ.")
        self._last_source_sequence = -1
        self._last_source_measured_at = -math.inf
        self._next_proposal_sequence = 0
        self._adapter = HumanProposalAdapter(
            HumanProposalAdapterConfig(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                embodiment_instance_id=config.embodiment_instance_id,
                producer_id=config.producer_id,
                producer_epoch=config.producer_epoch,
                produced_clock_domain_id=config.clock_domain_id,
                effective_clock_domain_id=config.clock_domain_id,
                component_catalog=runtime.catalog,
                action_schema=runtime.action_schema,
                action_adapter=TeleopCommandActionAdapterConfig(
                    arm_joint_policy="source_pose_when_available",
                    source="human",
                    include_command_metadata=False,
                    arm_specs={
                        side: ArmActionSpec.tool_tcp(
                            side,
                            reference_frame=(
                                runtime.action_schema.payload_schema_by_component[
                                    f"{side}_arm"
                                ].reference_frame
                            ),
                        )
                        for side in ("left", "right")
                    },
                ),
            )
        )

    def capture_anchor(
        self,
        master_anchor: HumanResidualPoseAnchor,
        measurement: ExecutionMeasurementObservation,
    ) -> B601RM65MotionAnchor:
        """Capture fresh measured poses; never anchor to an unaccepted proposal."""
        if not self.enabled:
            raise RuntimeError("Motion publication is disabled.")
        now = float(self.runtime.clock())
        if not isinstance(master_anchor, HumanResidualPoseAnchor):
            raise TypeError("master_anchor must be a HumanResidualPoseAnchor.")
        if not isinstance(measurement, ExecutionMeasurementObservation):
            raise TypeError("measurement must be an ExecutionMeasurementObservation.")
        clock = self.runtime.config.clock_domain_id
        if (
            measurement.control_domain_id != self.runtime.config.control_domain_id
            or measurement.clock_domain_id != clock
            or master_anchor.captured_clock_domain_id != clock
            or set(master_anchor.entry_by_component) != {self.component_id}
        ):
            raise ValueError("Anchor domain/component binding differs.")
        self._age(now, measurement.available_at, self.limits[1])
        self._age(now, measurement.measured_at, self.limits[1])
        self._age(now, master_anchor.captured_at, self.limits[0])
        self._age(now, master_anchor.entries[0].source_measured_at, self.limits[0])
        values = {}
        for (
            component_id,
            payload,
        ) in self.runtime.action_schema.payload_schema_by_component.items():
            if (
                measurement.representation_by_component.get(component_id)
                != "cartesian_pose"
                or measurement.schema_hash_by_component.get(component_id)
                != payload.semantic_content_hash
                or payload.value_semantics != "absolute_target"
            ):
                raise ValueError("Target measurement pose schema differs.")
            self._pose(measurement.value_by_component[component_id])
            values[component_id] = measurement.value_by_component[component_id]
        gate = self.runtime.authoritative_port.capture_execution_gate_snapshot(now=now)
        if gate.status != "ready":
            raise RuntimeError("Target execution gate is not ready.")
        return B601RM65MotionAnchor(
            master_anchor,
            measurement,
            ComponentAction(
                action_schema=self.runtime.action_schema, values_by_component=values
            ),
            gate.gate_epoch,
        )

    def submit(
        self,
        step: B601RM65ResidualStep,
        anchor: B601RM65MotionAnchor,
        *,
        target_effective_at: float,
        effective_until: float,
    ) -> ProposalAdmissionReceipt:
        """Admit one current source proposal; receipt is not a dispatch receipt."""
        if not self.enabled:
            return ProposalAdmissionReceipt(
                accepted=False, rejection_code="motion_profile_disabled"
            )
        now = float(self.runtime.clock())
        gate = self.runtime.authoritative_port.capture_execution_gate_snapshot(now=now)
        if gate.status != "ready":
            return ProposalAdmissionReceipt(
                accepted=False, rejection_code="target_not_ready"
            )
        try:
            if not isinstance(step, B601RM65ResidualStep) or not isinstance(
                anchor, B601RM65MotionAnchor
            ):
                raise TypeError("Motion step/anchor type differs.")
            if anchor.gate_epoch != gate.gate_epoch:
                raise ValueError("Target gate changed; capture new anchors explicitly.")
            if not step.residual_available:
                raise ValueError(f"Current source rejected: {step.rejection_reason}")
            assert step.adaptation is not None and step.relative_action is not None
            assert step.projection.pose_sample is not None
            sample = step.projection.pose_sample
            receipt = step.adaptation.receipt
            source_emission = step.projection.source_step.emission
            if (
                source_emission is None
                or source_emission.semantic_content_hash
                != step.projection.source_emission_hash
                or receipt.anchor_hash != anchor.master_anchor.semantic_content_hash
                or receipt.adapter_contract_hash
                != self.source.residual_adapter.contract_hash
                or receipt.adapted_clock_domain_id
                != self.runtime.config.clock_domain_id
                or sample.measured_clock_domain_id
                != self.runtime.config.clock_domain_id
                or step.relative_action.action_schema.semantic_content_hash
                != self.source.residual_adapter.output_action_schema.semantic_content_hash
                or anchor.target_action.action_schema.semantic_content_hash
                != self.runtime.action_schema.semantic_content_hash
                or anchor.target_measurement.control_domain_id
                != self.runtime.config.control_domain_id
                or sample.source_coordinate_binding_id
                != self.source.source_coordinate_binding_id
                or sample.source_coordinate_binding_hash
                != self.source.source_coordinate_binding_hash
                or anchor.master_anchor.frame_graph_hash
                != self.source.residual_adapter.frame_graph.semantic_content_hash
            ):
                raise ValueError("Source/anchor/schema lineage differs.")
            state = step.projection.source_step.state
            if (
                state.sequence <= self._last_source_sequence
                or sample.measured_at <= self._last_source_measured_at
            ):
                raise ValueError("Source sequence/time did not advance.")
            self._age(now, sample.measured_at, self.limits[0])
            self._age(now, state.receive_timestamp_sec, self.limits[0])
            self._age(now, receipt.adapted_at, self.limits[0])
            if sample.measured_at < anchor.master_anchor.captured_at:
                raise ValueError("Source predates the activation barrier.")
            target_time, expires = float(target_effective_at), float(effective_until)
            if (
                not math.isfinite(target_time)
                or not math.isfinite(expires)
                or not now
                <= target_time
                < expires
                <= sample.measured_at + self.limits[0]
            ):
                raise ValueError("Proposal outlives its source or has impossible time.")
            delta = self._pose(
                step.relative_action.values_by_component[self.component_id].values
            )
            translation_m, rotation_deg = pose_delta_metrics(identity_pose(), delta)
            if (
                translation_m > self.limits[2]
                or math.radians(rotation_deg) > self.limits[3]
            ):
                raise ValueError(
                    "Residual exceeds the configured translation/rotation range."
                )
            command = TeleopCommand(timestamp=state.receive_timestamp_sec)
            for component_id, value in anchor.target_action.values_by_component.items():
                pose = self._pose(value.values)
                if component_id == self.component_id:
                    pose = self._pose(
                        apply_relative_pose_component_value(
                            anchor.target_action,
                            step.relative_action,
                            component_id=component_id,
                        )
                    )
                command.add_command(
                    EffectorCommand(
                        effector_name=component_id,
                        side=component_id.removesuffix("_arm"),
                        command_type="cartesian_pose",
                        value=pose,
                        frame_name=self.runtime.action_schema.payload_schema_by_component[
                            component_id
                        ].reference_frame,
                    )
                )
            emission = HumanCommandEmission.capture(
                emission_sequence=self._next_proposal_sequence,
                emitted_at=now,
                contributions=(("b601_anchor_relative_motion", command),),
                merged_command=command,
                observation_sample_refs=source_emission.observation_sample_refs,
            )
            adapted = self._adapter.adapt(
                emission,
                effective_from=now,
                effective_until=expires,
                target_effective_at=target_time,
            )
            if (
                adapted.proposal_draft is None
                or not adapted.source_adapter_receipt.authoritative_eligible
            ):
                raise ValueError(
                    f"Human proposal rejected: {adapted.source_validation.reasons}"
                )
            proposal = replace(
                adapted.proposal_draft,
                audit_metadata={
                    **adapted.proposal_draft.audit_metadata,
                    "b601_residual_receipt_hash": receipt.semantic_content_hash,
                    "master_anchor_hash": anchor.master_anchor.semantic_content_hash,
                    "target_measurement_hash": anchor.target_measurement.semantic_content_hash,
                    "b601_source_sequence": state.sequence,
                },
            )
            self._last_source_sequence, self._last_source_measured_at = (
                state.sequence,
                sample.measured_at,
            )
            self._next_proposal_sequence += 1
            admitted = self.runtime.authoritative_port.admit_proposal(
                proposal,
                source_event_id=adapted.source_adapter_receipt.source_emission_id,
            )
            if not admitted.accepted:
                raise ValueError(f"Target proposal rejected: {admitted.rejection_code}")
            return admitted
        except (KeyError, TypeError, ValueError) as error:
            # Cancellation invalidates retained proposals/segments, not just this tick.
            self.runtime.transition_execution_gate(
                "hold", reason=f"b601_motion_rejected:{error}"
            )
            return ProposalAdmissionReceipt(
                accepted=False,
                rejection_code="b601_motion_rejected",
                message=str(error),
            )

    @staticmethod
    def _age(now: float, timestamp: float, maximum: float) -> None:
        if (
            not math.isfinite(now)
            or not math.isfinite(timestamp)
            or not 0 <= now - timestamp <= maximum
        ):
            raise ValueError(
                "Source/measurement timestamp is future, stale or non-finite."
            )

    @staticmethod
    def _pose(value: object) -> Any:
        pose = parse_pose_matrix(value)
        valid, reason = validate_pose_matrix(pose, atol=1e-7)
        if not valid:
            raise ValueError(reason)
        return pose
