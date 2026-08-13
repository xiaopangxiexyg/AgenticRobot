"""Default-off policy/human allocation profiles for Sim-HIL.

This application-leaf module selects concrete source roles and development
thresholds, then joins the reusable allocation/session ports.  It does not
poll devices, run a model, compose actions, implement safety/IK, or dispatch.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    AllocationTransitionContinuityPolicy,
    ComponentAllocationEvidencePolicy,
    ComponentAllocationPolicy,
)
from control_composition.composition import DerivedCompositionInputRoleBinding
from control_composition.lifecycle import CompositionSessionOperationSpec
from control_station.composition import (
    MujocoPolicyHumanAllocationBindingConfig,
    MujocoPolicyHumanAllocationBindingFactory,
    MujocoPolicyHumanAllocationProfileBundle,
    MujocoPolicyHumanAllocationProfileConfig,
    MujocoPolicyHumanAllocationSession,
    build_mujoco_component_human_proposal_adapter,
    build_mujoco_policy_human_allocation_profile,
    build_mujoco_policy_human_allocation_station_profiles,
)
from control_station.contracts import StationOperationResultDraft
from control_station.publication_barrier import StationPublicationBarrier
from control_station.sources.policy import PolicyCompositionInputAdapter
from control_station.state_machine import admit_station_operation_result
from robot_motion.action import ComponentActionSchema
from robot_motion.coordinate_frames import CoordinateFrameGraph, CoordinateFrameSpec

from .allocation_control_cycle import AllocationProposalExecutionControlCyclePort
from .profile_selection import (
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
    SimHILApplicationControlProfile,
)


SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A = (
    "policy_human_allocation_scheme_a"
)
SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B = (
    "policy_human_allocation_scheme_b"
)
SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS = frozenset(
    {
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
        SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
    }
)
_POSE_ROTATION_NORMALIZATION_RAD = math.radians(10.0)
_SCHEME_B_TARGET_HANDOFF_DURATION_SEC = 0.25
_PERSISTENT_OVERRIDE_SELECTION_LEASE_SEC = 0.5
_PROFILE_BINDING_HASH_DOMAIN = (
    "dexteleop.control_station.sim_hil_allocation_profile_binding.development"
)
_PROFILE_PROPOSAL_GATE_HASH_DOMAIN = (
    "dexteleop.control_station.sim_hil_allocation_proposal_gate.development"
)


@dataclass(frozen=True)
class SimHILPolicyHumanAllocationPreparation:
    """Resource-free contracts selected before target construction."""

    bundle: MujocoPolicyHumanAllocationProfileBundle
    station_profiles: Any


def _allocation_frame_graph(
    *,
    profile_id: str,
    action_schema: ComponentActionSchema,
    target_frame_graph: CoordinateFrameGraph,
) -> CoordinateFrameGraph:
    frames = {frame.frame_id: frame for frame in target_frame_graph.frames}
    for component_id in action_schema.component_ids:
        schema = action_schema.payload_schema_by_component[component_id]
        if schema.representation_family != "cartesian_pose":
            continue
        for frame_id in (schema.reference_frame, schema.controlled_frame):
            if frame_id and frame_id not in frames:
                frames[frame_id] = CoordinateFrameSpec(
                    frame_id=frame_id,
                    semantic_parameters={
                        "selected_by": profile_id,
                        "component_id": component_id,
                    },
                )
    return CoordinateFrameGraph(
        graph_id=f"{target_frame_graph.graph_id}.{profile_id}",
        frames=tuple(frames.values()),
        transforms=target_frame_graph.transforms,
        consistency_atol=target_frame_graph.consistency_atol,
    )


def _prepare_policy_human_allocation_profile(
    *,
    profile_id: str,
    automatic_mode_ceiling: str,
    proposal_promotable_modes: tuple[str, ...],
    target_handoff_duration_sec: float | None,
    automatic_override_enabled: bool = True,
    rebase_operation_enabled: bool = True,
    initial_override_mode: str | None = None,
    composition_required_roles: tuple[str, ...] | None = None,
    action_schema: ComponentActionSchema,
    target_frame_graph: CoordinateFrameGraph,
    target_frame_profile_hash: str,
    tool_frame_profile_hash: str,
    composer_producer_id: str,
) -> SimHILPolicyHumanAllocationPreparation:
    """Build one selected profile without constructing sources or execution."""

    if not isinstance(action_schema, ComponentActionSchema):
        raise TypeError("action_schema must be a ComponentActionSchema.")
    if not isinstance(target_frame_graph, CoordinateFrameGraph):
        raise TypeError("target_frame_graph must be a CoordinateFrameGraph.")
    pose_components = tuple(
        component_id
        for component_id in action_schema.component_ids
        if action_schema.payload_schema_by_component[
            component_id
        ].representation_family
        == "cartesian_pose"
    )
    qpos_components = tuple(
        component_id
        for component_id in action_schema.component_ids
        if action_schema.payload_schema_by_component[
            component_id
        ].representation_family
        == "joint_position"
    )
    if set((*pose_components, *qpos_components)) != set(action_schema.component_ids):
        raise ValueError(
            "Allocation profiles currently require cartesian_pose or joint_position "
            "components; custom representations need explicit metrics."
        )
    graph = _allocation_frame_graph(
        profile_id=profile_id,
        action_schema=action_schema,
        target_frame_graph=target_frame_graph,
    )
    source_binding_hash = canonical_content_hash(
        {
            "profile_id": profile_id,
            "action_schema_hash": action_schema.semantic_content_hash,
            "target_frame_profile_hash": str(target_frame_profile_hash),
            "tool_frame_profile_hash": str(tool_frame_profile_hash),
            "frame_graph_hash": graph.semantic_content_hash,
            "source_contract": "canonical_absolute_human_component_action",
        },
        domain=_PROFILE_BINDING_HASH_DOMAIN,
    )
    thresholds = ComponentAllocationEvidencePolicy(
        relative_enter_level=1.0,
        relative_exit_level=0.5,
        absolute_enter_level=3.0,
        absolute_exit_level=2.0,
    )
    policies = tuple(
        ComponentAllocationPolicy(
            component_id=component_id,
            motion_source_role="human_activity",
            residual_source_role="human_residual",
            motion_policy=thresholds,
            residual_magnitude_policy=thresholds,
            minimum_dwell_sec=0.2,
            automatic_mode_ceiling=automatic_mode_ceiling,
        )
        for component_id in action_schema.component_ids
    )
    activity_scales = {
        component_id: (
            {
                "translation_displacement_m": 0.02,
                "rotation_displacement_rad": (
                    _POSE_ROTATION_NORMALIZATION_RAD
                ),
            }
            if component_id in pose_components
            else {"coordinate_linf_displacement_native": 0.1}
        )
        for component_id in action_schema.component_ids
    }
    residual_scales = {
        component_id: (
            {
                "translation_from_anchor_m": 0.02,
                "rotation_from_anchor_rad": (
                    _POSE_ROTATION_NORMALIZATION_RAD
                ),
            }
            if component_id in pose_components
            else {"coordinate_linf_from_anchor_native": 0.1}
        )
        for component_id in action_schema.component_ids
    }
    reference_frames = {
        component_id: str(
            action_schema.payload_schema_by_component[component_id].reference_frame
        )
        for component_id in pose_components
    }
    controlled_frames = {
        component_id: str(
            action_schema.payload_schema_by_component[component_id].controlled_frame
        )
        for component_id in pose_components
    }
    if any(value in {"", "None"} for value in (*reference_frames.values(), *controlled_frames.values())):
        raise ValueError("Allocation pose components require concrete frames.")
    transition_continuity_policy = (
        None
        if target_handoff_duration_sec is None
        else AllocationTransitionContinuityPolicy(
            policy_id=f"{profile_id}.target-transition-continuity",
            component_ids=action_schema.component_ids,
            handoff_duration_sec=target_handoff_duration_sec,
        )
    )
    bundle = build_mujoco_policy_human_allocation_profile(
        absolute_action_schema=action_schema,
        frame_graph=graph,
        concrete_reference_frame_by_component=reference_frames,
        concrete_controlled_frame_by_component=controlled_frames,
        source_binding_id="sim-hil-human-canonical-absolute",
        source_binding_hash=source_binding_hash,
        composer_producer_id=composer_producer_id,
        component_policies=policies,
        window_duration_sec=0.5,
        evidence_ttl_sec=0.25,
        activity_normalization_scale_by_component=activity_scales,
        residual_normalization_scale_by_component=residual_scales,
        transition_continuity_policy=transition_continuity_policy,
        config=MujocoPolicyHumanAllocationProfileConfig(
            profile_id=profile_id,
            enabled=True,
            # The stable attachable human port reserves both human role
            # contracts without opening a device.  Scheme-A/B therefore
            # start in base_only and admit Human only at an explicit mode
            # transition.
            require_human_roles_at_start=False,
            composition_required_roles=composition_required_roles,
            automatic_override_enabled=automatic_override_enabled,
            rebase_operation_enabled=rebase_operation_enabled,
            initial_override_mode=initial_override_mode,
            persistent_override_selection_lease_sec=(
                _PERSISTENT_OVERRIDE_SELECTION_LEASE_SEC
            ),
            proposal_promotable_modes=proposal_promotable_modes,
        ),
    )
    return SimHILPolicyHumanAllocationPreparation(
        bundle=bundle,
        station_profiles=(
            build_mujoco_policy_human_allocation_station_profiles(bundle)
        ),
    )


def prepare_exclusive_truncated_hitl_allocation_profile(
    *,
    action_schema: ComponentActionSchema,
    target_frame_graph: CoordinateFrameGraph,
    target_frame_profile_hash: str,
    tool_frame_profile_hash: str,
    composer_producer_id: str,
) -> SimHILPolicyHumanAllocationPreparation:
    """Build discrete policy/human takeover on the streaming allocation path."""

    return _prepare_policy_human_allocation_profile(
        profile_id=SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
        automatic_mode_ceiling=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        proposal_promotable_modes=(
            COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        ),
        target_handoff_duration_sec=None,
        automatic_override_enabled=False,
        rebase_operation_enabled=False,
        initial_override_mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        # Policy and Human-absolute are complete alternative owners. A static
        # Policy requirement would force unused model refreshes during Human
        # takeover merely to keep an unrelated input alive.
        composition_required_roles=(),
        action_schema=action_schema,
        target_frame_graph=target_frame_graph,
        target_frame_profile_hash=target_frame_profile_hash,
        tool_frame_profile_hash=tool_frame_profile_hash,
        composer_producer_id=composer_producer_id,
    )
def prepare_policy_human_allocation_scheme_a(
    *,
    action_schema: ComponentActionSchema,
    target_frame_graph: CoordinateFrameGraph,
    target_frame_profile_hash: str,
    tool_frame_profile_hash: str,
    composer_producer_id: str,
) -> SimHILPolicyHumanAllocationPreparation:
    """Build relative-only Scheme-A contracts without runtime resources."""

    return _prepare_policy_human_allocation_profile(
        profile_id=SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
        automatic_mode_ceiling=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        proposal_promotable_modes=(
            COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        ),
        target_handoff_duration_sec=None,
        action_schema=action_schema,
        target_frame_graph=target_frame_graph,
        target_frame_profile_hash=target_frame_profile_hash,
        tool_frame_profile_hash=tool_frame_profile_hash,
        composer_producer_id=composer_producer_id,
    )


def prepare_policy_human_allocation_scheme_b(
    *,
    action_schema: ComponentActionSchema,
    target_frame_graph: CoordinateFrameGraph,
    target_frame_profile_hash: str,
    tool_frame_profile_hash: str,
    composer_producer_id: str,
) -> SimHILPolicyHumanAllocationPreparation:
    """Build opt-in three-state Scheme-B contracts with target continuity."""

    return _prepare_policy_human_allocation_profile(
        profile_id=SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
        automatic_mode_ceiling=(
            COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER
        ),
        proposal_promotable_modes=(
            COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
            COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        ),
        target_handoff_duration_sec=_SCHEME_B_TARGET_HANDOFF_DURATION_SEC,
        action_schema=action_schema,
        target_frame_graph=target_frame_graph,
        target_frame_profile_hash=target_frame_profile_hash,
        tool_frame_profile_hash=tool_frame_profile_hash,
        composer_producer_id=composer_producer_id,
    )


class _ApplicationStationResultProvider:
    """Admit exact lower activation evidence at the Station application leaf."""

    def __init__(self, runtime: Any, *, subject_id: str) -> None:
        self.runtime = runtime
        self.subject_id = str(subject_id)
        self._next_sequence = 0
        self._lock = threading.RLock()

    def __call__(self, evidence: Any):
        from control_station.composition.mujoco_policy_human_allocation_lifecycle import (
            MUJOCO_ALLOCATION_SESSION_REARM_STATUS,
            MUJOCO_ALLOCATION_SESSION_START_STATUS,
        )

        with self._lock:
            available_at = max(
                float(self.runtime.clock()),
                float(evidence.causal_ready_at),
            )
            status = (
                MUJOCO_ALLOCATION_SESSION_START_STATUS
                if evidence.operation_kind == "start_composition_session"
                else MUJOCO_ALLOCATION_SESSION_REARM_STATUS
            )
            result = admit_station_operation_result(
                StationOperationResultDraft(
                    source_intent_id=(
                        f"sim-hil-allocation:{evidence.operation_kind}:"
                        f"{evidence.activation_scope.activation_id}"
                    ),
                    operation_kind=evidence.operation_kind,
                    subject_id=self.subject_id,
                    succeeded=True,
                    completed_at=available_at,
                    completed_clock_domain_id=(
                        self.runtime.config.clock_domain_id
                    ),
                    status=status,
                    evidence_refs=evidence.required_evidence_refs,
                ),
                runtime_session_id=self.runtime.config.runtime_session_id,
                control_domain_id=self.runtime.config.control_domain_id,
                available_at=available_at,
                available_clock_domain_id=self.runtime.config.clock_domain_id,
                admission_sequence=self._next_sequence,
            )
            self._next_sequence += 1
            return result


def _build_policy_human_allocation_control_profile(
    *,
    expected_profile_id: str,
    preparation: SimHILPolicyHumanAllocationPreparation,
    runtime: Any,
    human_source: Any,
    human_source_spec: Any,
    readiness_provider: Any,
    policy_binding: Any,
    component_action_projector: Any,
    physics_ticks_after_dispatch: int,
) -> SimHILApplicationControlProfile:
    """Join already-constructed source/target resources behind one profile."""

    bundle = preparation.bundle
    profile_id = bundle.config.profile_id
    if profile_id != expected_profile_id:
        raise ValueError(
            f"{expected_profile_id} builder received {profile_id} preparation."
        )
    policy_adapter = PolicyCompositionInputAdapter(
        policy_binding,
        component_action_projector=component_action_projector,
    )
    human_adapter = build_mujoco_component_human_proposal_adapter(
        runtime,
        human_source_spec,
        producer_id="sim-hil-human-allocation-adapter",
        producer_epoch=1,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
        component_action_projector=component_action_projector,
    )
    schemas = bundle.input_action_schema_by_role
    human_bindings = (
        DerivedCompositionInputRoleBinding(
            input_role=bundle.config.absolute_overlay_role,
            producer_id="sim-hil-human-absolute",
            producer_epoch=1,
            action_schema_hash=(
                schemas[bundle.config.absolute_overlay_role].semantic_content_hash
            ),
        ),
        DerivedCompositionInputRoleBinding(
            input_role=bundle.config.residual_role,
            producer_id="sim-hil-human-residual",
            producer_epoch=1,
            action_schema_hash=(
                schemas[bundle.config.residual_role].semantic_content_hash
            ),
        ),
    )
    composer_slot = "allocation-composer"
    barrier = StationPublicationBarrier(source_slot_id=composer_slot)
    binding_factory = MujocoPolicyHumanAllocationBindingFactory(
        runtime=runtime,
        bundle=bundle,
        station_profiles=preparation.station_profiles,
        publication_barrier=barrier,
        human_source=human_source,
        policy_input_adapter=policy_adapter,
        human_adapter=human_adapter,
        human_role_bindings=human_bindings,
        config=MujocoPolicyHumanAllocationBindingConfig(
            allocation_scope_id=f"sim-hil:{profile_id}",
            allocation_producer_id="sim-hil-allocation-context",
            allocation_producer_epoch=1,
            anchor_lifecycle_producer_id="sim-hil-human-anchor-lifecycle",
            anchor_lifecycle_producer_epoch=1,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        ),
    )
    proposal_gate_hash = canonical_content_hash(
        {
            "authority_profile_hash": bundle.authority_profile.semantic_content_hash,
            "proposal_profile_id": preparation.station_profiles.proposal.profile_id,
            "promotable_modes": (
                preparation.station_profiles.proposal.promotable_allocation_modes
            ),
            "automatic_override_enabled": (
                bundle.config.automatic_override_enabled
            ),
            "rebase_operation_enabled": bundle.config.rebase_operation_enabled,
            "initial_override_mode": bundle.config.initial_override_mode,
            "persistent_override_selection_lease_sec": (
                bundle.config.persistent_override_selection_lease_sec
            ),
            "send_actions": preparation.station_profiles.proposal.send_actions,
        },
        domain=_PROFILE_PROPOSAL_GATE_HASH_DOMAIN,
    )
    operation_spec = CompositionSessionOperationSpec(
        session_key=f"sim-hil:{profile_id}",
        composer_source_slot_id=composer_slot,
        contributor_source_slot_ids=(
            policy_binding.source_slot_id,
            human_source_spec.source_id,
        ),
        composition_spec_hash=bundle.composition_spec.semantic_content_hash,
        authority_gate_spec_hash=proposal_gate_hash,
        execution_binding_hash=(
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        capture_boundary_id="sim-hil-episode-capture-boundary",
    )
    result_provider = _ApplicationStationResultProvider(
        runtime,
        subject_id=composer_slot,
    )
    session = MujocoPolicyHumanAllocationSession(
        runtime=runtime,
        bundle=bundle,
        binding_factory=binding_factory,
        operation_spec=operation_spec,
        readiness_provider=readiness_provider,
        activation_observer=lambda _scope: None,
        start_result_provider=result_provider,
        rearm_result_provider=result_provider,
    )
    return SimHILApplicationControlProfile(
        profile_id=profile_id,
        session=session,
        control_cycle_port=AllocationProposalExecutionControlCyclePort(
            port_id=f"sim-hil:{profile_id}",
            binding_factory=binding_factory,
            activation_human_emission_required=(
                bundle.config.require_human_roles_at_start
            ),
        ),
    )


def build_policy_human_allocation_scheme_a_control_profile(
    *,
    preparation: SimHILPolicyHumanAllocationPreparation,
    runtime: Any,
    human_source: Any,
    human_source_spec: Any,
    readiness_provider: Any,
    policy_binding: Any,
    component_action_projector: Any,
    physics_ticks_after_dispatch: int,
) -> SimHILApplicationControlProfile:
    """Join Scheme-A resources behind one application profile."""

    return _build_policy_human_allocation_control_profile(
        expected_profile_id=(
            SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
        ),
        preparation=preparation,
        runtime=runtime,
        human_source=human_source,
        human_source_spec=human_source_spec,
        readiness_provider=readiness_provider,
        policy_binding=policy_binding,
        component_action_projector=component_action_projector,
        physics_ticks_after_dispatch=physics_ticks_after_dispatch,
    )


def build_exclusive_truncated_hitl_allocation_control_profile(
    *,
    preparation: SimHILPolicyHumanAllocationPreparation,
    runtime: Any,
    human_source: Any,
    human_source_spec: Any,
    readiness_provider: Any,
    policy_binding: Any,
    component_action_projector: Any,
    physics_ticks_after_dispatch: int,
) -> SimHILApplicationControlProfile:
    """Join discrete takeover to the common streaming allocation cycle."""

    return _build_policy_human_allocation_control_profile(
        expected_profile_id=SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
        preparation=preparation,
        runtime=runtime,
        human_source=human_source,
        human_source_spec=human_source_spec,
        readiness_provider=readiness_provider,
        policy_binding=policy_binding,
        component_action_projector=component_action_projector,
        physics_ticks_after_dispatch=physics_ticks_after_dispatch,
    )


def build_policy_human_allocation_scheme_b_control_profile(
    *,
    preparation: SimHILPolicyHumanAllocationPreparation,
    runtime: Any,
    human_source: Any,
    human_source_spec: Any,
    readiness_provider: Any,
    policy_binding: Any,
    component_action_projector: Any,
    physics_ticks_after_dispatch: int,
) -> SimHILApplicationControlProfile:
    """Join Scheme-B resources behind one application profile."""

    return _build_policy_human_allocation_control_profile(
        expected_profile_id=(
            SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B
        ),
        preparation=preparation,
        runtime=runtime,
        human_source=human_source,
        human_source_spec=human_source_spec,
        readiness_provider=readiness_provider,
        policy_binding=policy_binding,
        component_action_projector=component_action_projector,
        physics_ticks_after_dispatch=physics_ticks_after_dispatch,
    )


__all__ = [
    "SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A",
    "SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B",
    "SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS",
    "SimHILPolicyHumanAllocationPreparation",
    "build_exclusive_truncated_hitl_allocation_control_profile",
    "build_policy_human_allocation_scheme_a_control_profile",
    "build_policy_human_allocation_scheme_b_control_profile",
    "prepare_exclusive_truncated_hitl_allocation_profile",
    "prepare_policy_human_allocation_scheme_a",
    "prepare_policy_human_allocation_scheme_b",
]
