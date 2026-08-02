"""CS2a1 synthetic-human-only composition over the real MuJoCo control core.

The module is an application leaf: it may import both ``sim`` and
``teleop_runtime``.  Neither lower package imports ``control_station``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    ArmActionSpec,
    ControlExecutionManifest,
    ExecutionProfileExclusion,
    build_control_execution_manifest,
    split_effector_name,
)
from robot_motion.action.proposal import ProposalAdmissionReceipt
from sim.mujoco.deploy import MujocoDeployRuntime
from teleop_runtime.action.adapters import TeleopCommandActionAdapterConfig
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterConfig,
    HumanProposalAdapterResult,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)

from ..contracts import (
    ControlStationRunManifest,
    ExecutionDomainBinding,
    SourceSlotSpec,
)
from ..sources.synthetic_human import (
    SyntheticHumanQposSourceSpec,
)
from ..sources.human import HumanPoseSourceContract
from ..sources.teleop_pose import PassiveTeleopComponentSourceSpec
from ..targets.authoritative import (
    authoritative_authority_epoch_source_id,
    authoritative_execution_epoch_source_id,
    authoritative_owner_registry_id,
)


CS2A1_MUJOCO_HUMAN_MANIFEST_ID_V1 = (
    "control_station_cs2a1_synthetic_human_mujoco_v1"
)
CS2A1_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2a1_mujoco_human_binding_receipt.v1"
)
CS2A1_PRODUCER_EPOCH_POLICY_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2a1_fixed_producer_epoch.v1"
)
CS2A2A_SYNTHETIC_HUMAN_EXECUTION_MANIFEST_ID_V1 = (
    "cs2a2a_synthetic_human_sim_arm_qpos_v1"
)
CS2A2B_SYNTHETIC_HUMAN_EXECUTION_MANIFEST_ID_V1 = (
    "cs2a2b_station_synthetic_human_sim_arm_qpos_v1"
)
CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1 = (
    "control_station_cs2a2b_synthetic_human_mujoco_station_v1"
)
CS2A2B_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2a2b_mujoco_human_binding_receipt.v1"
)


def _trace_profile_id(runtime: MujocoDeployRuntime) -> str:
    mode = runtime.reference_activation_spec.activation_mode
    if mode == "immediate_target":
        return "deployable_sim_immediate_qref_v1"
    if mode == "scheduled_segment":
        return "deployable_sim_scheduled_qpos_v1"
    raise ValueError(f"Unsupported MuJoCo activation mode: {mode!r}.")


def _station_trace_profile_id(runtime: MujocoDeployRuntime) -> str:
    mode = runtime.reference_activation_spec.activation_mode
    if mode == "immediate_target":
        return "deployable_sim_immediate_qref_station_v1"
    if mode == "scheduled_segment":
        return "deployable_sim_scheduled_qpos_station_v1"
    raise ValueError(f"Unsupported MuJoCo activation mode: {mode!r}.")


def build_mujoco_human_proposal_adapter(
    runtime: MujocoDeployRuntime,
) -> HumanProposalAdapter:
    """Bind the existing human adapter to the exact MuJoCo target contract."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    return HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=runtime.config.producer_id,
            producer_epoch=runtime.config.producer_epoch,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            component_catalog=runtime.catalog,
            action_schema=runtime.action_schema,
            action_adapter=TeleopCommandActionAdapterConfig(
                arm_joint_policy="qpos",
                source="human",
                include_command_metadata=False,
            ),
        )
    )


def build_mujoco_pose_human_proposal_adapter(
    runtime: MujocoDeployRuntime,
    source_spec: HumanPoseSourceContract,
) -> HumanProposalAdapter:
    """Bind a pose-native human emission seam to an exact pose target."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(source_spec, HumanPoseSourceContract):
        raise TypeError("source_spec must satisfy HumanPoseSourceContract.")
    if runtime.catalog.action_space != BIMANUAL_EE_POSE_V1:
        raise ValueError("Pose human adapter requires a pose-native target catalog.")
    reference_frames = dict(source_spec.reference_frame_by_effector)
    expected_effectors = {
        component.effector_name for component in runtime.catalog.components
    }
    if set(reference_frames) != expected_effectors:
        raise ValueError("Pose human source must exactly cover the target components.")
    target_contract = runtime.target_runtime_action_contract
    source_frame_hash = source_spec.pose_frame_contract_hash
    target_frame_hash = target_contract.pose_frame_contract_hash
    if source_frame_hash != target_frame_hash:
        resolver = runtime.pose_resolver
        transformer = None if resolver is None else resolver.pose_transformer
        if transformer is None:
            raise ValueError(
                "Pose human frame mismatch requires the target's verified transform."
            )
        receipt = transformer.binding_receipt
        if (
            receipt.source_pose_frame_contract_hash != source_frame_hash
            or receipt.target_pose_frame_contract_hash != target_frame_hash
        ):
            raise ValueError("Pose human transform does not bind these contracts.")
    if dict(source_spec.action_spec_hash_by_effector) != dict(
        target_contract.action_spec_hash_by_component
    ):
        raise ValueError("Pose human action specs mismatch the target contract.")
    if source_spec.units_convention_id != target_contract.units_convention_id:
        raise ValueError("Pose human units mismatch the target contract.")
    arm_specs: dict[str, ArmActionSpec] = {}
    for effector_name in sorted(expected_effectors):
        try:
            side, kind = split_effector_name(effector_name)
        except ValueError as exc:
            raise ValueError(
                "Pose human adapter currently requires canonical arm effector IDs; "
                f"got {effector_name!r}."
            ) from exc
        if kind != "arm":
            raise ValueError(
                "Pose human adapter accepts arm pose components only; "
                f"got {effector_name!r}."
            )
        arm_specs[side] = ArmActionSpec.tool_tcp(
            side,
            reference_frame=reference_frames[effector_name],
        )
    return HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=runtime.config.producer_id,
            producer_epoch=runtime.config.producer_epoch,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            component_catalog=runtime.catalog,
            action_schema=runtime.action_schema,
            action_adapter=TeleopCommandActionAdapterConfig(
                arm_joint_policy="source_pose_when_available",
                arm_specs=arm_specs,
                source="human",
                include_command_metadata=False,
            ),
        )
    )


def build_mujoco_component_human_proposal_adapter(
    runtime: MujocoDeployRuntime,
    source_spec: PassiveTeleopComponentSourceSpec,
    *,
    producer_id: str | None = None,
    producer_epoch: int | None = None,
    component_id_by_effector: Mapping[str, str] | None = None,
    component_action_projector: Any | None = None,
) -> HumanProposalAdapter:
    """Bind one pose-arm + qpos-hand source to an opt-in compound target.

    ``producer_id`` is an application binding override for profiles where the
    human contributor and the lower policy producer share one MuJoCo runtime.
    It changes no authority semantics; the returned adapter still only
    prepares a human action.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(source_spec, PassiveTeleopComponentSourceSpec):
        raise TypeError("source_spec must be a PassiveTeleopComponentSourceSpec.")
    source_components = {
        component.effector_name: component
        for component in source_spec.components
    }
    target_effectors = {
        component.effector_name: component
        for component in runtime.catalog.components
    }
    if set(source_components) != set(target_effectors):
        raise ValueError(
            "Compound human source must exactly cover the target effectors."
        )
    mapping = (
        {
            effector: target_effectors[effector].component_id
            for effector in target_effectors
        }
        if component_id_by_effector is None
        else {
            str(effector).strip(): str(component_id).strip()
            for effector, component_id in dict(
                component_id_by_effector
            ).items()
        }
    )
    if (
        set(mapping) != set(source_components)
        or set(mapping.values()) != set(runtime.action_schema.component_ids)
    ):
        raise ValueError(
            "Human effector/component mapping must exactly cover source and "
            "target schemas."
        )
    projection_hash = (
        None
        if component_action_projector is None
        else str(
            getattr(
                component_action_projector,
                "semantic_content_hash",
                "",
            )
        ).strip()
    )
    if projection_hash is None:
        target_spec_hashes = dict(
            runtime.target_runtime_action_contract.action_spec_hash_by_component
        )
        if {
            mapping[effector]: item.action_spec_hash
            for effector, item in source_components.items()
        } != target_spec_hashes:
            raise ValueError(
                "Compound human action specs mismatch the target contract."
            )
        if (
            source_spec.units_convention_id
            != runtime.target_runtime_action_contract.units_convention_id
        ):
            raise ValueError("Compound human units mismatch the target contract.")
    arm_specs = source_spec.arm_action_specs_by_side
    hand_specs = source_spec.hand_action_specs_by_side
    if not arm_specs or not hand_specs:
        raise ValueError(
            "Compound human source requires arm-pose and hand-qpos components."
        )
    return HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=(
                runtime.config.producer_id
                if producer_id is None
                else str(producer_id)
            ),
            producer_epoch=(
                runtime.config.producer_epoch
                if producer_epoch is None
                else int(producer_epoch)
            ),
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            component_catalog=runtime.catalog,
            action_schema=runtime.action_schema,
            component_id_by_effector=mapping,
            component_projection_contract_hash=projection_hash,
            action_adapter=TeleopCommandActionAdapterConfig(
                arm_joint_policy="source_pose_when_available",
                arm_specs=arm_specs,
                hand_specs=hand_specs,
                source="human",
                include_command_metadata=False,
            ),
        ),
        component_action_projector=component_action_projector,
    )


def build_cs2a1_mujoco_human_manifest(
    runtime: MujocoDeployRuntime,
    adapter: HumanProposalAdapter,
    source_spec: SyntheticHumanQposSourceSpec,
) -> ControlStationRunManifest:
    """Build the exact, single-source CS2a1 application manifest."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(adapter, HumanProposalAdapter):
        raise TypeError("adapter must be a HumanProposalAdapter.")
    if not isinstance(source_spec, SyntheticHumanQposSourceSpec):
        raise TypeError("source_spec must be a SyntheticHumanQposSourceSpec.")
    producer_epoch_policy_hash = canonical_content_hash(
        {
            "producer_id": runtime.config.producer_id,
            "producer_epoch": runtime.config.producer_epoch,
            "activation": "fixed_on_mujoco_runtime_start",
        },
        domain=CS2A1_PRODUCER_EPOCH_POLICY_HASH_DOMAIN_V1,
    )
    trace_profile_id = _trace_profile_id(runtime)
    binding = ExecutionDomainBinding(
        binding_id=f"mujoco-target:{runtime.config.control_domain_id}",
        target_kind="sim",
        control_domain_id=runtime.config.control_domain_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        target_runtime_action_contract_hash=(
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        execution_capability_profile_hash=(
            runtime.execution_capability.semantic_content_hash
        ),
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        selection_group_ids=runtime.core.config.active_selection_groups,
        owner_registry_id=authoritative_owner_registry_id(runtime),
        authority_epoch_source_id=authoritative_authority_epoch_source_id(runtime),
        execution_epoch_source_id=authoritative_execution_epoch_source_id(runtime),
        trace_requirement_profile_id=trace_profile_id,
        explicit_exclusions=(
            "real_backend",
            "hardware_device_initialization",
        ),
    )
    return ControlStationRunManifest(
        manifest_id=CS2A1_MUJOCO_HUMAN_MANIFEST_ID_V1,
        manifest_version="1",
        execution_binding=binding,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=source_spec.source_id,
                source_kind="human",
                producer_id=runtime.config.producer_id,
                source_action_contract_hash=adapter.contract_hash,
                producer_epoch_policy_hash=producer_epoch_policy_hash,
                supported_uses=("authority",),
                expected_emission_seam=(
                    "teleop_runtime.action.human_proposal.HumanCommandEmission"
                ),
                required_binding_refs=(
                    source_spec.semantic_content_hash,
                    runtime.catalog.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "human_command_emission",
                    "source_adapter_receipt",
                    "source_validation",
                    "action_proposal",
                    "arbitration_decision",
                ),
                explicit_exclusions=(
                    "physical_teleoperation_device",
                    "source_owned_action_queue",
                ),
            ),
        ),
        authority_policy_hash=(
            runtime.core.config.authority_policy.semantic_content_hash
        ),
        decision_schedule_hash=(
            runtime.core.config.decision_schedule.semantic_content_hash
        ),
        trace_requirement_profile_id=trace_profile_id,
        capture_policy_id="capture_not_owned_by_cs2a1_v1",
        allowed_intent_kinds=(
            "connect_source",
            "connect_target",
            "run_preflight",
            "hold",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "browser_raw_action_submission",
            "dataset_replay",
            "formal_episode_lifecycle",
            "mixed_source_authority",
            "physical_human_device",
            "policy_model",
            "production_entrypoint",
            "real_motion",
            "station_motion_authority_lifecycle",
        ),
    )


def build_cs2a2b_mujoco_human_station_manifest(
    runtime: MujocoDeployRuntime,
    adapter: HumanProposalAdapter,
    source_spec: SyntheticHumanQposSourceSpec,
) -> ControlStationRunManifest:
    """Build the exact scoped manifest with Station authority lifecycle.

    The manifest enables typed candidate/authority operations for the
    synthetic human source.  It still excludes real motion, browser action
    payloads, policy/replay sources, and a production entrypoint.
    """

    base = build_cs2a1_mujoco_human_manifest(runtime, adapter, source_spec)
    station_trace_profile_id = _station_trace_profile_id(runtime)
    source_slot = replace(
        base.source_slots[0],
        supported_uses=("candidate", "authority"),
        trace_requirement_refs=tuple(
            sorted(
                {
                    *base.source_slots[0].trace_requirement_refs,
                    "station_run_manifest",
                    "station_intent_transition",
                    "station_operation_transition",
                    "station_publication_barrier_transition",
                }
            )
        ),
    )
    return replace(
        base,
        manifest_id=CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1,
        execution_binding=replace(
            base.execution_binding,
            trace_requirement_profile_id=station_trace_profile_id,
        ),
        source_slots=(source_slot,),
        trace_requirement_profile_id=station_trace_profile_id,
        capture_policy_id="capture_not_owned_by_cs2a2b_v1",
        allowed_intent_kinds=(
            "acquire_control_lease",
            "release_control_lease",
            "connect_source",
            "disconnect_source",
            "connect_target",
            "disconnect_target",
            "run_preflight",
            "request_candidate",
            "release_candidate",
            "request_motion_authority",
            "release_motion_authority",
            "hold",
            "stop_runtime",
        ),
        explicit_exclusions=tuple(
            value
            for value in base.explicit_exclusions
            if value != "station_motion_authority_lifecycle"
        ),
    )


def build_cs2a2a_synthetic_human_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the scheduled human-sim execution fact used by formal capture.

    This manifest describes the source that the existing MuJoCo authority core
    actually selects. It does not state that Control Station granted that
    authority and it does not authorize real motion.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if station_manifest.manifest_id != CS2A1_MUJOCO_HUMAN_MANIFEST_ID_V1:
        raise ValueError("Execution manifest requires the exact CS2a1 station manifest.")
    if runtime.reference_activation_spec.activation_mode != "scheduled_segment":
        raise ValueError("Formal CS2a2a capture requires scheduled-segment activation.")
    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="synthetic_human_sim_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "dataset_replay",
            "formal_station_authority_lifecycle",
            "hardware_device_initialization",
            "physical_human_device",
            "policy_model",
            "real_backend",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS2A2A_SYNTHETIC_HUMAN_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=("human:synthetic_fixture",),
        human_motion_proposal_status="authorized",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("test_episode_composition",),
        lifecycle_control_requests=("start", "stop"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="not_applicable",
        exclusions=exclusions,
    )


def build_cs2a2b_synthetic_human_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the scoped Station-gated synthetic-human sim execution fact.

    The Station publication barrier is application authority evidence only;
    the existing MuJoCo core remains the sole Arbiter/owner/dispatcher.  This
    manifest therefore does not imply a production Web API or real motion.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if (
        station_manifest.manifest_id
        != CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1
    ):
        raise ValueError(
            "Station execution manifest requires the exact CS2a2b run manifest."
        )
    if runtime.reference_activation_spec.activation_mode != "scheduled_segment":
        raise ValueError(
            "Formal CS2a2b capture requires scheduled-segment activation."
        )
    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="station_synthetic_human_sim_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "dataset_replay",
            "hardware_device_initialization",
            "physical_human_device",
            "policy_model",
            "production_station_entrypoint",
            "real_backend",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS2A2B_SYNTHETIC_HUMAN_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=(
            "human:synthetic_fixture_via_control_station",
        ),
        human_motion_proposal_status="authorized",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("control_station_fixture",),
        lifecycle_control_requests=("start", "stop"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="not_applicable",
        exclusions=exclusions,
    )


@dataclass(frozen=True)
class MujocoHumanOnlyBindingReceipt:
    manifest_hash: str
    source_adapter_contract_hash: str
    target_runtime_action_contract_hash: str
    execution_capability_profile_hash: str
    component_catalog_hash: str
    authority_policy_hash: str
    decision_schedule_hash: str
    owner_token_id: str
    owner_token_hash: str
    execution_epoch_snapshot_hash: str
    authoritative_data_plane_bound: bool
    station_authority_lifecycle_bound: bool
    semantic_content_hash: str


@dataclass(frozen=True)
class MujocoHumanSubmission:
    emission: HumanCommandEmission
    adapter_result: HumanProposalAdapterResult
    proposal_admission: ProposalAdmissionReceipt | None

    @property
    def accepted(self) -> bool:
        return bool(
            self.adapter_result.source_adapter_receipt.authoritative_eligible
            and self.proposal_admission is not None
            and self.proposal_admission.accepted
        )


class MujocoHumanOnlyComposition:
    """Exact CS2a1 binding; the MuJoCo runtime remains the sole control owner."""

    def __init__(
        self,
        *,
        manifest: ControlStationRunManifest,
        runtime: MujocoDeployRuntime,
        adapter: HumanProposalAdapter,
        source_spec: SyntheticHumanQposSourceSpec,
    ) -> None:
        self.manifest = manifest
        self.runtime = runtime
        self.adapter = adapter
        self.source_spec = source_spec
        self._started = False
        self.binding_receipt = self._validate_exact_binding()

    @property
    def started(self) -> bool:
        return self._started

    def _validate_exact_binding(self) -> MujocoHumanOnlyBindingReceipt:
        if not isinstance(self.manifest, ControlStationRunManifest):
            raise TypeError("manifest must be a ControlStationRunManifest.")
        if not isinstance(self.runtime, MujocoDeployRuntime):
            raise TypeError("runtime must be a MujocoDeployRuntime.")
        if not isinstance(self.adapter, HumanProposalAdapter):
            raise TypeError("adapter must be a HumanProposalAdapter.")
        if self.manifest.manifest_id == CS2A1_MUJOCO_HUMAN_MANIFEST_ID_V1:
            expected = build_cs2a1_mujoco_human_manifest(
                self.runtime,
                self.adapter,
                self.source_spec,
            )
            station_authority_lifecycle_bound = False
            binding_hash_domain = CS2A1_BINDING_RECEIPT_HASH_DOMAIN_V1
        elif (
            self.manifest.manifest_id
            == CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1
        ):
            expected = build_cs2a2b_mujoco_human_station_manifest(
                self.runtime,
                self.adapter,
                self.source_spec,
            )
            station_authority_lifecycle_bound = True
            binding_hash_domain = CS2A2B_BINDING_RECEIPT_HASH_DOMAIN_V1
        else:
            raise ValueError("Unsupported synthetic-human MuJoCo station manifest.")
        if self.manifest.semantic_content_hash != expected.semantic_content_hash:
            raise ValueError(
                "Control Station manifest does not exactly bind this source and target."
            )
        if self.runtime.config.enable_replay_authority:
            raise ValueError("CS2a1 forbids replay authority.")
        if self.runtime.config.enable_composer_authority:
            raise ValueError("CS2a1 forbids composer authority.")
        if self.runtime.core.config.initial_control_mode != "human":
            raise ValueError("CS2a1 requires an explicit human control mode.")
        if self.runtime.config.decision_hz > 60.0:
            raise ValueError("CS2a1 ActionArbiter rate cannot exceed 60 Hz.")
        adapter_config = self.adapter.config
        if (
            adapter_config.runtime_session_id != self.runtime.config.runtime_session_id
            or adapter_config.control_domain_id != self.runtime.config.control_domain_id
            or adapter_config.embodiment_instance_id
            != self.runtime.config.embodiment_instance_id
            or adapter_config.producer_id != self.runtime.config.producer_id
            or adapter_config.producer_epoch != self.runtime.config.producer_epoch
            or adapter_config.component_catalog.semantic_content_hash
            != self.runtime.catalog.semantic_content_hash
        ):
            raise ValueError("Human adapter is not bound to the exact MuJoCo runtime.")
        if self.source_spec.joint_dof_by_effector != (
            ("left_arm", 6),
            ("right_arm", 6),
        ):
            raise ValueError("CS2a1 currently supports exact coupled bimanual q6 only.")
        owner_snapshot = self.runtime.core.dispatch_owners.snapshot()
        token = owner_snapshot.current_token
        if token is None or token.owner_kind != "shared_arbitration_runtime":
            raise ValueError("MuJoCo target lacks the shared authoritative owner token.")
        execution_snapshot = self.runtime.core.execution_epochs.snapshot()
        payload = {
            "manifest_hash": self.manifest.semantic_content_hash,
            "source_adapter_contract_hash": self.adapter.contract_hash,
            "target_runtime_action_contract_hash": (
                self.runtime.target_runtime_action_contract.semantic_content_hash
            ),
            "execution_capability_profile_hash": (
                self.runtime.execution_capability.semantic_content_hash
            ),
            "component_catalog_hash": self.runtime.catalog.semantic_content_hash,
            "authority_policy_hash": (
                self.runtime.core.config.authority_policy.semantic_content_hash
            ),
            "decision_schedule_hash": (
                self.runtime.core.config.decision_schedule.semantic_content_hash
            ),
            "owner_token_id": token.token_id,
            "owner_token_hash": token.semantic_content_hash,
            "execution_epoch_snapshot_hash": execution_snapshot.semantic_content_hash,
            "authoritative_data_plane_bound": True,
            "station_authority_lifecycle_bound": station_authority_lifecycle_bound,
        }
        return MujocoHumanOnlyBindingReceipt(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=binding_hash_domain,
            ),
        )

    def start(self) -> MujocoHumanOnlyBindingReceipt:
        if self._started:
            return self.binding_receipt
        self.runtime.start()
        self._started = True
        return self.binding_receipt

    def submit_emission(
        self,
        emission: HumanCommandEmission,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
    ) -> MujocoHumanSubmission:
        if not self._started:
            raise RuntimeError("MujocoHumanOnlyComposition.start() is required.")
        result = self.adapter.adapt(
            emission,
            effective_from=effective_from,
            effective_until=effective_until,
            target_effective_at=target_effective_at,
        )
        facts = project_human_proposal_adapter_result(
            emission,
            result,
            runtime_session_id=self.runtime.config.runtime_session_id,
            control_domain_id=self.runtime.config.control_domain_id,
            embodiment_instance_id=self.runtime.config.embodiment_instance_id,
            occurred_at=float(self.runtime.clock()),
            clock_domain_id=self.runtime.config.clock_domain_id,
        )
        self.runtime.core.record_external_facts(facts)
        receipt = None
        if (
            result.proposal_draft is not None
            and result.source_adapter_receipt.authoritative_eligible
        ):
            receipt = self.runtime.core.admit_proposal(
                result.proposal_draft,
                source_event_id=result.source_adapter_receipt.source_emission_id,
            )
        return MujocoHumanSubmission(
            emission=emission,
            adapter_result=result,
            proposal_admission=receipt,
        )

    def step(self, *, now: float | None = None, physics_ticks_after_dispatch: int = 0):
        if not self._started:
            raise RuntimeError("MujocoHumanOnlyComposition.start() is required.")
        return self.runtime.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        if not self._started:
            return
        self.runtime.close()
        self._started = False


__all__ = [
    "CS2A1_MUJOCO_HUMAN_MANIFEST_ID_V1",
    "CS2A2A_SYNTHETIC_HUMAN_EXECUTION_MANIFEST_ID_V1",
    "CS2A2B_SYNTHETIC_HUMAN_EXECUTION_MANIFEST_ID_V1",
    "CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1",
    "MujocoHumanOnlyBindingReceipt",
    "MujocoHumanOnlyComposition",
    "MujocoHumanSubmission",
    "build_cs2a1_mujoco_human_manifest",
    "build_cs2a2a_synthetic_human_execution_manifest",
    "build_cs2a2b_synthetic_human_execution_manifest",
    "build_cs2a2b_mujoco_human_station_manifest",
    "build_mujoco_component_human_proposal_adapter",
    "build_mujoco_human_proposal_adapter",
    "build_mujoco_pose_human_proposal_adapter",
]
