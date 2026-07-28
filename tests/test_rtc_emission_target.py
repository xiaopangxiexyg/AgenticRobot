from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    DecisionScheduleSpec,
    ProposalBuffer,
    RobotAction,
    build_immediate_qref_activation_spec,
    build_scheduled_reference_segment_activation_spec,
    project_robot_action_v1_to_component_action,
)
from robot_motion.position_control.capability import ExecutionCapabilityProfile
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    NormalizedAction,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    REPRESENTATION_PROOF_LEGACY_V1,
    REPRESENTATION_PROOF_TYPED_CANONICAL_V1,
    TimedPolicyActionKeyframe,
    assess_policy_emission_target_compatibility,
    build_canonical_policy_action,
    build_deployment_observation,
    nan_vector,
    prepare_policy_emission_target_proposal,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from robot_motion.replay import build_target_runtime_action_contract


ACTION_SPACE = "single_arm_pose_unit_v1"
FRAME_HASH = "frame:single-arm-base-tool-tcp-unit"
CLOCK_DOMAIN = "monotonic:rtc-target-unit"


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class TypedPassThrough:
    def prepare_canonical(
        self,
        action,
        _observation,
        *,
        policy_name,
        input_action_id,
        target_effective_at,
    ):
        del policy_name, input_action_id, target_effective_at
        return action


class LegacyPolicy:
    name = "legacy-rtc-target-unit"

    def __init__(self, row: np.ndarray) -> None:
        self.row = row

    def predict(self, _observation):
        return self.row[None, :].copy()


def _observation():
    pose = np.eye(4, dtype=float)
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=7,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=10.0,
    )


def _pose_action(*, sides: tuple[str, ...] = ("right",)) -> RobotAction:
    pose_by_side = {
        side: np.asarray(
            [0.2, -0.1 if side == "left" else 0.1, 0.3, 0, 0, 0, 1],
            dtype=np.float32,
        )
        for side in sides
    }
    return RobotAction(
        action_space=ACTION_SPACE,
        timestamp=10.0,
        arm_pose_by_side=pose_by_side,
        frame_by_effector={
            f"{side}_arm": f"{side}_arm_base" for side in sides
        },
        controlled_frame_by_effector={
            f"{side}_arm": "tool_tcp" for side in sides
        },
        source="rtc-emission-target-unit",
    )


def _action_schema():
    return project_robot_action_v1_to_component_action(
        _pose_action(),
    ).action_schema


def _typed_emission(*, target_effective_at: float = 10.0):
    clock = ManualClock()
    canonical = build_canonical_policy_action(
        _pose_action(),
        source_pose_mode="absolute",
        observation_cut_id="observation:rtc-target-unit",
        source_action_hash="source-action:rtc-target-unit",
        action_codec_hash="codec:rtc-target-unit",
        frame_contract_hash=FRAME_HASH,
        layout_hash="layout:rtc-target-unit",
        normalization_hash="normalization:none-unit",
        conversion_contract_hash="conversion:identity-unit",
        source_chunk_hash="source-chunk:rtc-target-unit",
        canonical_chunk_hash="canonical-chunk:rtc-target-unit",
        source_index=0,
    )
    keyframe = TimedPolicyActionKeyframe(
        target_effective_at=target_effective_at,
        effective_until=target_effective_at + 1.0 / 30.0,
        effective_clock_domain_id=CLOCK_DOMAIN,
        action_period_sec=1.0 / 30.0,
        decoded_absolute_action=None,
        request_id="request:rtc-target-unit",
        inference_id="inference:rtc-target-unit",
        policy_output_id="output:rtc-target-unit",
        chunk_id=1,
        chunk_keyframe_index=0,
        chunk_size=1,
        observation_cut_id="observation:rtc-target-unit",
        producer_epoch=1,
        policy_node_id="policy:rtc-target-unit",
        artifact_hash="artifact:rtc-target-unit",
        produced_at=clock(),
        schedule_spec_id="schedule:rtc-target-unit",
        schedule_spec_hash="schedule-hash:rtc-target-unit",
        admitted_plan_generation=1,
        state=POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
        valid_until=target_effective_at + 1.0,
        canonical_action=canonical,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            policy_node_id="policy:rtc-target-unit",
            policy_clock_domain_id=CLOCK_DOMAIN,
            action_keyframe_period_sec=1.0 / 30.0,
        ),
        monotonic_clock=clock,
    )
    prepared = runtime.prepare_timed_action_keyframe(
        keyframe,
        _observation(),
        preparer=TypedPassThrough(),
        policy_name="typed-rtc-target-unit",
        plan_keyframe_count_before=1,
        plan_keyframe_count_after=0,
        plan_pop_count=1,
    )
    assert prepared.emission_error is None
    assert prepared.emission is not None
    return prepared.emission


def _legacy_emission():
    clock = ManualClock()
    row = nan_vector(DEPLOY_ACTION_DIM)
    row[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.2, 0.1, 0.3, 0, 0, 0, 1],
        dtype=np.float32,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            policy_node_id="policy:rtc-target-unit",
            policy_clock_domain_id=CLOCK_DOMAIN,
        ),
        monotonic_clock=clock,
    )
    prediction = runtime.predict_and_queue(
        LegacyPolicy(row),
        _observation(),
        now=clock(),
        action_keyframe_period_sec=1.0 / 30.0,
    )
    assert prediction.accepted
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=1.0,
                max_arm_rot_step_deg=180.0,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )
    prepared = runtime.prepare_next_action(
        _observation(),
        preparer=pipeline,
        policy_name="legacy-rtc-target-unit",
    )
    assert prepared.emission_error is None
    assert prepared.emission is not None
    return prepared.emission


def _catalog(*, bimanual: bool = False) -> ActionComponentCatalog:
    sides = ("left", "right") if bimanual else ("right",)
    return ActionComponentCatalog(
        catalog_id=(
            "target-bimanual-pose-unit"
            if bimanual
            else "target-right-pose-unit"
        ),
        action_space=ACTION_SPACE,
        components=tuple(
            ActionComponentSpec(
                component_id=f"{side}_arm",
                selection_group_id="arms",
                effector_name=f"{side}_arm",
                coupled=True,
            )
            for side in sides
        ),
    )


def _target_facts(
    catalog: ActionComponentCatalog,
    *,
    scheduled: bool = False,
    frame_hash: str = FRAME_HASH,
):
    capability = ExecutionCapabilityProfile(
        profile_id="rtc-target-capability-unit",
        validated_max_selected_keyframe_hz=30.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=60.0,
        configured_coupled_ik_hz=60.0,
        reference_resample_hz=60.0,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=(ACTION_SPACE,),
        validated_selection_groups=catalog.selection_group_ids,
        supports_local_group_stop=False,
        acceptance_evidence_id="rtc-target-capability-unit",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )
    lead = 1.0 / 30.0 + 1.0 / 60.0 if scheduled else 0.0
    schedule = DecisionScheduleSpec(
        spec_id="rtc-target-schedule-unit",
        decision_hz=30.0,
        target_lead_time_sec=lead,
    )
    activation = (
        build_scheduled_reference_segment_activation_spec(
            schedule,
            capability,
            minimum_activation_margin_sec=1.0 / 60.0,
            activation_margin_evidence_id="rtc-target-margin-unit",
        )
        if scheduled
        else build_immediate_qref_activation_spec(schedule, capability)
    )
    contract = build_target_runtime_action_contract(
        target_contract_id="target:rtc-emission-unit",
        control_domain_id="domain:rtc-emission-unit",
        embodiment_instance_id="embodiment:rtc-emission-unit",
        component_catalog=catalog,
        pose_frame_contract_hash=frame_hash,
        execution_capability_profile_hash=capability.semantic_content_hash,
    )
    return contract, activation


def _assess(
    emission,
    catalog,
    *,
    scheduled: bool = False,
    frame_hash: str = FRAME_HASH,
    legacy_frame_hash: str | None = None,
    legacy_frame_binding_hash: str | None = None,
):
    contract, activation = _target_facts(
        catalog,
        scheduled=scheduled,
        frame_hash=frame_hash,
    )
    return assess_policy_emission_target_compatibility(
        emission,
        target_contract=contract,
        component_catalog=catalog,
        reference_activation_spec=activation,
        target_clock_domain_id=CLOCK_DOMAIN,
        legacy_pose_frame_contract_hash=legacy_frame_hash,
        legacy_pose_frame_binding_hash=legacy_frame_binding_hash,
    )


def test_typed_and_legacy_representation_proofs_remain_distinct() -> None:
    catalog = _catalog()
    typed = _assess(_typed_emission(), catalog)
    legacy_emission = _legacy_emission()
    contract, activation = _target_facts(catalog)
    contract = replace(
        contract,
        accepted_action_spaces_by_component={
            "right_arm": (
                ACTION_SPACE,
                legacy_emission.pre_safety_robot_action.action_space,
            )
        },
    )
    legacy = assess_policy_emission_target_compatibility(
        legacy_emission,
        target_contract=contract,
        component_catalog=catalog,
        reference_activation_spec=activation,
        target_clock_domain_id=CLOCK_DOMAIN,
        legacy_pose_frame_contract_hash=FRAME_HASH,
        legacy_pose_frame_binding_hash="binding:legacy-frame-unit",
    )

    assert typed.accepted
    assert typed.representation_proof_kind == (
        REPRESENTATION_PROOF_TYPED_CANONICAL_V1
    )
    assert legacy.accepted
    assert legacy.representation_proof_kind == REPRESENTATION_PROOF_LEGACY_V1
    assert typed.representation_trace_id != legacy.representation_trace_id
    assert typed.emission_hash == _typed_emission().semantic_content_hash


def test_due_pop_typed_emission_cannot_satisfy_scheduled_target_lead() -> None:
    assessment = _assess(
        _typed_emission(target_effective_at=10.0),
        _catalog(),
        scheduled=True,
    )

    assert not assessment.accepted
    assert "target_activation_lead_insufficient" in assessment.reasons
    assert assessment.available_target_lead_sec == pytest.approx(0.0)


def test_target_frame_and_authorized_component_scope_fail_closed() -> None:
    frame_mismatch = _assess(
        _typed_emission(),
        _catalog(),
        frame_hash="frame:another-target-unit",
    )
    component_mismatch = _assess(
        _typed_emission(),
        _catalog(bimanual=True),
    )

    assert "target_pose_frame_contract_mismatch" in frame_mismatch.reasons
    assert "authorized_component_coverage_mismatch" in (
        component_mismatch.reasons
    )


def test_authorization_cannot_split_a_coupled_target_group() -> None:
    emission = _typed_emission()
    catalog = _catalog(bimanual=True)
    contract, activation = _target_facts(catalog)

    with pytest.raises(ValueError, match="split coupled group"):
        assess_policy_emission_target_compatibility(
            emission,
            target_contract=contract,
            component_catalog=catalog,
            reference_activation_spec=activation,
            target_clock_domain_id=CLOCK_DOMAIN,
            authorized_component_ids=("right_arm",),
        )


def test_legacy_frame_fact_requires_an_independent_binding_hash() -> None:
    emission = _legacy_emission()
    catalog = _catalog()
    contract, activation = _target_facts(catalog)

    with pytest.raises(ValueError, match="must be provided together"):
        assess_policy_emission_target_compatibility(
            emission,
            target_contract=contract,
            component_catalog=catalog,
            reference_activation_spec=activation,
            target_clock_domain_id=CLOCK_DOMAIN,
            legacy_pose_frame_contract_hash=FRAME_HASH,
        )


def _proposal_buffer(*, session_id: str, producer_id: str):
    clock = ManualClock()
    coordinator = ControlAdmissionCoordinator(
        "domain:rtc-emission-unit",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id=CLOCK_DOMAIN,
            runtime_session_id=session_id,
            boot_id="boot:rtc-emission-target-unit",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id=session_id,
        coordinator=coordinator,
    )
    epochs.activate(
        producer_id=producer_id,
        producer_epoch=1,
        produced_at=9.0,
        produced_clock_domain_id=CLOCK_DOMAIN,
        reason="rtc-target-unit-startup",
    )
    return ProposalBuffer(
        runtime_session_id=session_id,
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=4,
    )


def test_target_neutral_preparation_admits_once_through_real_proposal_buffer() -> None:
    emission = _typed_emission()
    catalog = _catalog()
    contract, activation = _target_facts(catalog)
    session_id = "session:rtc-emission-target-unit"
    producer_id = "producer:rtc-emission-target-unit"
    prepared = prepare_policy_emission_target_proposal(
        emission,
        target_contract=contract,
        component_catalog=catalog,
        action_schema=_action_schema(),
        reference_activation_spec=activation,
        target_clock_domain_id=CLOCK_DOMAIN,
        runtime_session_id=session_id,
        producer_id=producer_id,
        producer_epoch=1,
        sequence=0,
    )

    assert prepared.assessment.accepted
    assert prepared.source_event_id == emission.emission_id
    assert prepared.proposal_draft is not None
    draft = prepared.proposal_draft
    assert draft.control_domain_id == contract.control_domain_id
    assert draft.embodiment_instance_id == contract.embodiment_instance_id
    assert draft.authority_request == "none"
    assert draft.authority_priority == 0
    assert draft.representation_trace_id == emission.representation_trace_id
    assert draft.audit_metadata["policy_action_emission_hash"] == (
        emission.semantic_content_hash
    )
    assert draft.audit_metadata["policy_emission_target_assessment_hash"] == (
        prepared.assessment.semantic_content_hash
    )

    target = _proposal_buffer(
        session_id=session_id,
        producer_id=producer_id,
    )
    first = target.admit(draft)
    duplicate = target.admit(draft)

    assert first.accepted
    assert first.proposal is not None
    assert first.proposal.representation_trace_id == emission.representation_trace_id
    assert not duplicate.accepted
    assert duplicate.rejection_code == "duplicate"


def test_rejected_target_preparation_cannot_create_a_proposal() -> None:
    emission = _typed_emission()
    catalog = _catalog()
    contract, activation = _target_facts(catalog, scheduled=True)
    prepared = prepare_policy_emission_target_proposal(
        emission,
        target_contract=contract,
        component_catalog=catalog,
        action_schema=_action_schema(),
        reference_activation_spec=activation,
        target_clock_domain_id=CLOCK_DOMAIN,
        runtime_session_id="session:rtc-emission-target-unit",
        producer_id="producer:rtc-emission-target-unit",
        producer_epoch=1,
        sequence=0,
    )

    assert not prepared.assessment.accepted
    assert "target_activation_lead_insufficient" in prepared.assessment.reasons
    assert prepared.proposal_draft is None


def test_legacy_preparation_preserves_legacy_representation_proof() -> None:
    emission = _legacy_emission()
    catalog = _catalog()
    contract, activation = _target_facts(catalog)
    contract = replace(
        contract,
        accepted_action_spaces_by_component={
            "right_arm": (
                ACTION_SPACE,
                emission.pre_safety_robot_action.action_space,
            )
        },
    )
    prepared = prepare_policy_emission_target_proposal(
        emission,
        target_contract=contract,
        component_catalog=catalog,
        action_schema=_action_schema(),
        reference_activation_spec=activation,
        target_clock_domain_id=CLOCK_DOMAIN,
        runtime_session_id="session:rtc-emission-target-unit",
        producer_id="producer:rtc-emission-target-unit",
        producer_epoch=1,
        sequence=0,
        legacy_pose_frame_contract_hash=FRAME_HASH,
        legacy_pose_frame_binding_hash="binding:legacy-frame-unit",
    )

    assert prepared.assessment.accepted
    assert prepared.assessment.representation_proof_kind == (
        REPRESENTATION_PROOF_LEGACY_V1
    )
    assert prepared.proposal_draft is not None
    assert prepared.proposal_draft.representation_trace_id == (
        emission.representation_trace_id
    )


@pytest.mark.parametrize(
    ("producer_epoch", "sequence"),
    ((-1, 0), (1, -1), (True, 0), (1, True)),
)
def test_target_neutral_preparation_rejects_invalid_producer_identity(
    producer_epoch,
    sequence,
) -> None:
    catalog = _catalog()
    contract, activation = _target_facts(catalog)

    with pytest.raises(ValueError, match="non-negative integer"):
        prepare_policy_emission_target_proposal(
            _typed_emission(),
            target_contract=contract,
            component_catalog=catalog,
            action_schema=_action_schema(),
            reference_activation_spec=activation,
            target_clock_domain_id=CLOCK_DOMAIN,
            runtime_session_id="session:rtc-emission-target-unit",
            producer_id="producer:rtc-emission-target-unit",
            producer_epoch=producer_epoch,
            sequence=sequence,
        )
