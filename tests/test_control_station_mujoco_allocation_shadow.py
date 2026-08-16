from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from control_composition import (
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND,
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    ResidualAnchorLifecycleDraft,
    ResidualAnchorLifecycleSpec,
    build_composition_activation_scope,
)
from control_composition.allocation import (
    COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT,
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    AllocationPublicationBasis,
    AllocationPublicationRuntime,
    ComponentAllocationEvidencePolicy,
    ComponentAllocationPolicy,
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
    CompositionAllocationSpec,
    empty_composition_allocation_state,
)
from control_composition.composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    AllocatedPoseComponentCompositionRule,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionSpec,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
)
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.trajectory import (
    FrozenTrajectoryHeldInput,
    FrozenTrajectoryResidualInput,
    ResidualTemporalWarpSpec,
    TrajectoryCompositionContributorStamp,
    activate_residual_session,
    build_trajectory_corrected_preview,
    capture_frozen_trajectory_composition_projection,
    empty_trajectory_editing_state,
    offer_nominal_trajectory,
)
from control_station import (
    StationOperationResultDraft,
    StationPublicationBarrier,
    admit_station_operation_result,
)
from control_station.composition import (
    MujocoAllocationShadowController,
    MujocoAllocationShadowProfile,
    StationAllocationPublicationPort,
    StationAllocationPublicationProfile,
    TrajectoryCorrectedPreviewController,
    TrajectoryCorrectedPreviewProfile,
)
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    ComponentAction,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    build_relative_delta_action_schema,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    MujocoCompositionRuntimeConfig,
    MujocoCoupledPoseToQposResolver,
    MujocoDeployConfig,
    MujocoDeployRuntime,
    build_mujoco_pose_action_schema,
)
from sim.mujoco.ik import RM65CanonicalIK


pytestmark = [pytest.mark.unit, pytest.mark.contract]


BASE_ROLE = "base_absolute"
RELATIVE_ROLE = "operator_relative"
ABSOLUTE_ROLE = "operator_absolute"
ALLOCATION_PRODUCER = "virtual-allocation"
CONTEXT_CHANNEL = "source_allocation"


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class ShadowScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_command_count = 0

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del qpos_by_side, gripper_close_by_name, zero_controlled_qvel
        self.physics_command_count += 1


class ShadowPlanner:
    def reset(self, _qpos_by_side):
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        identity = np.eye(4, dtype=float)
        return SimpleNamespace(
            valid=True,
            hold=False,
            status="tracking",
            q_ref_by_side={
                "left": np.zeros(6, dtype=float),
                "right": np.zeros(6, dtype=float),
            },
            target_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            realized_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            modification_reasons=(),
            metadata={"planner": "allocation-shadow-never-dispatched"},
        )


def _contracts():
    output_schema = build_mujoco_pose_action_schema({})
    relative_schema = build_relative_delta_action_schema(
        output_schema,
        schema_id="mujoco-allocation-relative-pose",
    )
    thresholds = ComponentAllocationEvidencePolicy(
        relative_enter_level=1.0,
        relative_exit_level=0.5,
        absolute_enter_level=3.0,
        absolute_exit_level=2.0,
    )
    allocation_spec = CompositionAllocationSpec(
        allocation_spec_id="mujoco-allocation-shadow",
        base_absolute_role=BASE_ROLE,
        relative_overlay_role=RELATIVE_ROLE,
        absolute_overlay_role=ABSOLUTE_ROLE,
        component_policies=tuple(
            ComponentAllocationPolicy(
                component_id=component_id,
                motion_source_role="operator_activity",
                residual_source_role=RELATIVE_ROLE,
                motion_policy=thresholds,
                residual_magnitude_policy=thresholds,
            )
            for component_id in output_schema.component_ids
        ),
    )
    rules = ComponentCompositionRuleSet(
        rule_set_id="mujoco-allocation-shadow-rules",
        action_space="component-keyed",
        rules=tuple(
            AllocatedPoseComponentCompositionRule(
                component_id=component_id,
                base_absolute_role=BASE_ROLE,
                relative_overlay_role=RELATIVE_ROLE,
                absolute_overlay_role=ABSOLUTE_ROLE,
                allocation_spec_hash=allocation_spec.semantic_content_hash,
                allocation_context_channel=CONTEXT_CHANNEL,
            )
            for component_id in output_schema.component_ids
        ),
    )
    spec = CompositionSpec(
        spec_id="mujoco-allocation-shadow-composition",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=(BASE_ROLE, RELATIVE_ROLE, ABSOLUTE_ROLE),
        required_roles=(BASE_ROLE, RELATIVE_ROLE, ABSOLUTE_ROLE),
        depends_on_roles_by_role={RELATIVE_ROLE: (BASE_ROLE,)},
        component_rule_set_hash=rules.semantic_content_hash,
        action_schema_hash=output_schema.semantic_content_hash,
        input_action_schema_hash_by_role={
            BASE_ROLE: output_schema.semantic_content_hash,
            RELATIVE_ROLE: relative_schema.semantic_content_hash,
            ABSOLUTE_ROLE: output_schema.semantic_content_hash,
        },
    )
    lifecycle_spec = ResidualAnchorLifecycleSpec(
        spec_id="mujoco-allocation-shadow-anchor-lifecycle",
        source_role=RELATIVE_ROLE,
        component_ids=output_schema.component_ids,
        absolute_action_schema_hash=output_schema.semantic_content_hash,
        relative_action_schema_hash=relative_schema.semantic_content_hash,
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="mujoco-allocation-shadow-strategy",
        strategy_version="development",
        composition_spec_hash=spec.semantic_content_hash,
        required_context_channels=(CONTEXT_CHANNEL,),
        required_residual_anchor_lifecycle_spec_hash=(
            lifecycle_spec.semantic_content_hash
        ),
        implementation_contract_hashes=(rules.semantic_content_hash,),
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=spec,
        component_rule_set=rules,
        allocation_spec_by_hash={
            allocation_spec.semantic_content_hash: allocation_spec
        },
        residual_anchor_lifecycle_spec=lifecycle_spec,
    )
    plan = CompositionInformationPlan(
        plan_id="mujoco-allocation-shadow-information",
        context_requirements=(
            CompositionContextRequirement(
                channel=CONTEXT_CHANNEL,
                accepted_payload_contract_hashes=(
                    COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT,
                ),
                payload_access="inline",
                maximum_age_sec=1.0,
            ),
        ),
    )
    return output_schema, relative_schema, allocation_spec, spec, strategy, plan


def _action(schema, *, x: float) -> ComponentAction:
    return ComponentAction(
        action_schema=schema,
        values_by_component={
            component_id: (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            for component_id in schema.component_ids
        },
    )


def _barrier() -> StationPublicationBarrier:
    barrier = StationPublicationBarrier(source_slot_id="composer")
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:start-allocation-shadow",
            operation_kind="start_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=9.9,
            completed_clock_domain_id="control.monotonic",
            status="composition_session_ready",
            evidence_refs=("virtual-allocation-shadow",),
        ),
        runtime_session_id="mujoco-allocation-shadow-session",
        control_domain_id="mujoco-allocation-shadow-domain",
        available_at=9.9,
        available_clock_domain_id="control.monotonic",
        admission_sequence=0,
    )
    barrier.enable_from_composition_session_result(result)
    return barrier


def _fixture(*, admit_context=None, enable_composer_authority: bool = False):
    clock = ManualClock()
    scene = ShadowScene()
    output, relative, allocation_spec, spec, strategy, plan = _contracts()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    trace_facts = []
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="mujoco-allocation-shadow-session",
            control_domain_id="mujoco-allocation-shadow-domain",
            embodiment_instance_id="mujoco-allocation-shadow-robot",
            # The lower runtime contract requires one configured authority;
            # this shadow never submits any proposal to it.
            enable_primary_policy_authority=True,
            enable_composer_authority=enable_composer_authority,
            enable_replay_authority=False,
            clock_domain_id="control.monotonic",
            pose_frame_contract_hash="virtual-mujoco-pose-frame-contract",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(
            information_plan=plan,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(
            RM65CanonicalIK(scene, ShadowPlanner())
        ),
        composition_strategy=strategy,
        composition_input_action_schema_by_role={
            BASE_ROLE: output,
            RELATIVE_ROLE: relative,
            ABSOLUTE_ROLE: output,
        },
        trace_sink=trace_facts.append,
        clock=clock,
    )
    runtime.start()
    for producer in (
        BASE_ROLE,
        RELATIVE_ROLE,
        ABSOLUTE_ROLE,
        ALLOCATION_PRODUCER,
    ):
        runtime.activate_policy_node_producer(producer, producer_epoch=1)
    activation = build_composition_activation_scope(
        operation_spec_id="mujoco-allocation-shadow",
        operation_spec_hash="mujoco-allocation-shadow-operation-hash",
        session_epoch=1,
        activated_at=9.9,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.9,
        required_roles=(BASE_ROLE, RELATIVE_ROLE, ABSOLUTE_ROLE),
    )
    anchor_request = ResidualAnchorRequest.for_activation(
        activation,
        request_generation=0,
        source_role=RELATIVE_ROLE,
        component_ids=output.component_ids,
        source_binding_id="virtual-human-source",
        source_binding_hash="virtual-human-source-hash",
        absolute_action_schema_hash=output.semantic_content_hash,
        relative_action_schema_hash=relative.semantic_content_hash,
        requested_at=9.95,
    )
    anchor_receipt = ResidualAnchorReceipt.capture(
        anchor_request,
        source_action_id="virtual-human-anchor-action",
        source_action_hash="virtual-human-anchor-action-hash",
        source_available_at=9.97,
        captured_at=9.98,
        component_bindings=tuple(
            ResidualAnchorComponentBinding(
                component_id=component_id,
                source_sample_id=f"virtual-anchor-sample:{component_id}",
                source_sample_hash=f"virtual-anchor-sample-hash:{component_id}",
                absolute_payload_schema_hash=(
                    output.payload_schema_by_component[
                        component_id
                    ].semantic_content_hash
                ),
                anchor_id=f"virtual-anchor:{component_id}",
                anchor_hash=f"virtual-anchor-hash:{component_id}",
            )
            for component_id in output.component_ids
        ),
    )
    lifecycle_spec = strategy.residual_anchor_lifecycle_spec
    assert lifecycle_spec is not None
    for sequence, phase, receipt in (
        (0, RESIDUAL_ANCHOR_LIFECYCLE_PENDING, None),
        (1, RESIDUAL_ANCHOR_LIFECYCLE_READY, anchor_receipt),
    ):
        lifecycle_admission = runtime.submit_residual_anchor_lifecycle(
            ResidualAnchorLifecycleDraft(
                runtime_session_id=runtime.config.runtime_session_id,
                control_domain_id=runtime.config.control_domain_id,
                producer_id=ALLOCATION_PRODUCER,
                producer_epoch=1,
                sequence=sequence,
                lifecycle_scope_id="mujoco-operator-arms",
                lifecycle_spec_hash=lifecycle_spec.semantic_content_hash,
                phase=phase,
                effective_for_target_at=10.01,
                observed_at=9.95 if receipt is None else 9.98,
                observed_clock_domain_id="control.monotonic",
                anchor_request=anchor_request,
                anchor_receipt=receipt,
            ),
            spec=lifecycle_spec,
            activation_scope=activation,
        )
        assert lifecycle_admission.accepted
    actions = {
        BASE_ROLE: _action(output, x=1.0),
        RELATIVE_ROLE: _action(relative, x=0.1),
        ABSOLUTE_ROLE: _action(output, x=9.0),
    }
    for sequence, (role, action) in enumerate(actions.items()):
        receipt = runtime.submit_composition_input(
            CompositionInputDraft(
                runtime_session_id=runtime.config.runtime_session_id,
                control_domain_id=runtime.config.control_domain_id,
                producer_id=role,
                producer_epoch=1,
                sequence=0,
                input_role=role,
                action=action,
                produced_at=10.0,
                produced_clock_domain_id="control.monotonic",
                effective_from=10.01,
                effective_until=10.3,
                effective_clock_domain_id="control.monotonic",
                composition_activation_id=activation.activation_id,
                composition_activation_hash=activation.semantic_content_hash,
                semantic_binding_id_by_kind=(
                    {}
                    if role == BASE_ROLE
                    else {
                        "residual_anchor": anchor_receipt.receipt_id
                    }
                ),
                semantic_binding_hash_by_kind=(
                    {}
                    if role == BASE_ROLE
                    else {
                        "residual_anchor": (
                            anchor_receipt.semantic_content_hash
                        )
                    }
                ),
                audit_metadata={"fixture_order": sequence},
            ),
            activation_scope=activation,
        )
        assert receipt.accepted
    publisher = AllocationPublicationRuntime(
        allocation_scope_id="mujoco-operator-arms",
        allocation_spec=allocation_spec,
        activation_scope=activation,
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        control_clock_reset_epoch=0,
        context_channel=CONTEXT_CHANNEL,
        producer_id=ALLOCATION_PRODUCER,
        producer_epoch=1,
        initial_reducer_state=empty_composition_allocation_state(
            "mujoco-allocation-reducer",
            spec=allocation_spec,
        ),
        clock=clock,
    )
    barrier = _barrier()
    port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(
            profile_id="mujoco-allocation-shadow-publication",
            enabled=True,
        ),
        allocation_runtime=publisher,
        publication_barrier=barrier,
        admit_context=(
            runtime.submit_composition_context
            if admit_context is None
            else admit_context
        ),
        resolve_context=runtime.resolve_composition_context,
    )
    preview_controller = TrajectoryCorrectedPreviewController(
        profile=TrajectoryCorrectedPreviewProfile(
            profile_id="mujoco-allocation-corrected-preview",
            enabled=True,
            send_actions=False,
        ),
        clock=clock,
        clock_domain_id="control.monotonic",
    )
    controller = MujocoAllocationShadowController(
        profile=MujocoAllocationShadowProfile(
            profile_id="mujoco-allocation-shadow",
            enabled=True,
            send_actions=False,
        ),
        runtime=runtime,
        composition_spec=spec,
        allocation_spec=allocation_spec,
        activation_scope=activation,
        publication_port=port,
        corrected_preview_controller=preview_controller,
    )
    return SimpleNamespace(
        clock=clock,
        scene=scene,
        runtime=runtime,
        trace_facts=trace_facts,
        output_schema=output,
        relative_schema=relative,
        allocation_spec=allocation_spec,
        composition_spec=spec,
        activation=activation,
        anchor_request=anchor_request,
        anchor_receipt=anchor_receipt,
        actions=actions,
        publisher=publisher,
        barrier=barrier,
        port=port,
        controller=controller,
    )


def _basis(fixture, *, level: float = 1.5) -> AllocationPublicationBasis:
    cut = fixture.runtime.capture_control_cut(cutoff_at=10.0)
    activity = {}
    magnitude = {}
    for component_id in fixture.allocation_spec.component_ids:
        activity[component_id] = ComponentSourceActivityObservation(
            source_role="operator_activity",
            metric_provider_id="virtual.activity",
            metric_provider_hash="virtual-activity-hash",
            component_id=component_id,
            representation_family="cartesian_pose",
            window_started_at=9.9,
            window_ended_at=9.99,
            available_at=10.0,
            valid_until=10.3,
            clock_domain_id="control.monotonic",
            metric_values={"motion": level},
            normalization_scale_by_metric={"motion": 1.0},
            source_fact_ids=(f"virtual-motion:{component_id}",),
            source_fact_hashes=(f"virtual-motion-hash:{component_id}",),
        )
        magnitude[component_id] = ComponentResidualMagnitudeObservation(
            source_role=RELATIVE_ROLE,
            metric_provider_id="virtual.residual",
            metric_provider_hash="virtual-residual-hash",
            component_id=component_id,
            representation_family="cartesian_pose",
            sampled_at=9.99,
            available_at=10.0,
            valid_until=10.3,
            clock_domain_id="control.monotonic",
            anchor_receipt_id=fixture.anchor_receipt.receipt_id,
            anchor_receipt_hash=(
                fixture.anchor_receipt.semantic_content_hash
            ),
            anchor_request_generation=0,
            residual_action_id=f"virtual-residual:{component_id}",
            residual_action_hash=f"virtual-residual-hash:{component_id}",
            metric_values={"magnitude": level},
            normalization_scale_by_metric={"magnitude": 1.0},
        )
    return AllocationPublicationBasis(
        allocation_scope_id="mujoco-operator-arms",
        control_cut=cut,
        activation_scope=fixture.activation,
        activity_by_component=activity,
        residual_magnitude_by_component=magnitude,
        requested_effective_from=10.01,
        support_until=10.3,
    )


def test_default_off_shadow_is_lazy_and_has_no_lower_binding() -> None:
    controller = MujocoAllocationShadowController(
        profile=MujocoAllocationShadowProfile()
    )
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("disabled shadow read a supplier")

    assert controller.start() is None
    assert controller.offer_basis(forbidden).status == "disabled"
    assert controller.evaluate_at(target_effective_at=10.0).status == "disabled"
    assert controller.run_corrected_preview(forbidden).status == "disabled"
    assert calls == []
    assert controller.close()


def test_input_anchor_binding_is_a_separate_lineage_fact() -> None:
    fixture = _fixture()
    try:
        input_facts = tuple(
            fact
            for fact in fixture.trace_facts
            if fact.event_type in {
                "composition_input",
                "composition_contributor_input",
            }
        )
        binding_facts = tuple(
            fact
            for fact in fixture.trace_facts
            if fact.event_type == "composition_input_semantic_binding"
        )

        assert len(input_facts) == 3
        assert len(binding_facts) == 2
        assert all(
            "semantic_binding_id_by_kind" not in fact.payload
            and "semantic_binding_hash_by_kind" not in fact.payload
            for fact in input_facts
        )
        assert {
            fact.payload["composition_input_id"] for fact in binding_facts
        } == {
            fact.fact_id
            for fact in input_facts
            if fact.payload["input_role"] != BASE_ROLE
        }
        assert all(
            fact.payload["binding_id_by_kind"] == {
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: (
                    fixture.anchor_receipt.receipt_id
                )
            }
            for fact in binding_facts
        )
        lifecycle_facts = tuple(
            fact
            for fact in fixture.trace_facts
            if fact.event_type == "residual_anchor_lifecycle"
        )
        assert tuple(fact.payload["phase"] for fact in lifecycle_facts) == (
            RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
            RESIDUAL_ANCHOR_LIFECYCLE_READY,
        )
        assert lifecycle_facts[-1].payload["anchor_receipt"]["receipt_id"] == (
            fixture.anchor_receipt.receipt_id
        )
        assert all(
            "action" not in fact.payload for fact in lifecycle_facts
        )
    finally:
        fixture.runtime.close()


def test_pending_context_is_invisible_then_one_later_cut_selects_it() -> None:
    entered = threading.Event()
    release = threading.Event()
    fixture_ref = {}

    def delayed_admission(draft):
        entered.set()
        assert release.wait(timeout=2.0)
        return fixture_ref["fixture"].runtime.submit_composition_context(draft)

    fixture = _fixture(admit_context=delayed_admission)
    fixture_ref["fixture"] = fixture
    try:
        fixture.controller.start()
        offered = fixture.controller.offer_basis(lambda: _basis(fixture))
        assert offered.worker_notified
        assert entered.wait(timeout=2.0)

        fixture.clock.value = 10.02
        pending = fixture.controller.evaluate_at(
            target_effective_at=10.02,
            cutoff_at=10.02,
        )
        assert pending.selected_context_id is None
        assert pending.conditional_action is None
        assert pending.submission.composition_evaluation.output_status == "no_action"
        assert fixture.publisher.committed_reducer_state.revision == 0

        release.set()
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        assert fixture.port.snapshot().latest_outcome.outcome_kind == (
            ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
        )
        fixture.clock.value = 10.04
        accepted = fixture.controller.evaluate_at(
            target_effective_at=10.05,
            cutoff_at=10.04,
        )
        assert accepted.selected_mode_by_component == {
            component_id: "relative_overlay"
            for component_id in fixture.allocation_spec.component_ids
        }
        assert accepted.conditional_action is not None
        for component_id in fixture.allocation_spec.component_ids:
            assert accepted.conditional_action.value(component_id).values[0] == (
                pytest.approx(1.1)
            )
        assert accepted.submission.proposal_admission is None
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0

        events = fixture.controller.drain_publication_events()
        facts = tuple(
            project_composition_domain_event(
                fixture.runtime.core.trace_context,
                event,
            )
            for event in events
        )
        assert fixture.port.snapshot().latest_outcome.outcome_kind == (
            ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
        )
        assert any(
            fact.event_type == "allocation_publication_outcome"
            and fact.payload["outcome_kind"] == "accepted_new"
            and fact.payload["receipt_context_id"] == accepted.selected_context_id
            for fact in facts
        )
        assert any(
            fact.event_type == "composition_evaluation" for fact in fixture.trace_facts
        )
    finally:
        release.set()
        fixture.controller.close()
        fixture.runtime.close()


def _corrected_preview(fixture):
    keyframes = tuple(
        ComponentActionTrajectoryKeyframe(
            target_effective_at=target,
            effective_until=target + 0.05,
            action=_action(fixture.output_schema, x=x),
            source_index=index,
            source_fact_ids=(f"virtual-plan-step:{index}",),
            source_fact_hashes=(f"virtual-plan-step-hash:{index}",),
        )
        for index, (target, x) in enumerate(((10.05, 1.0), (10.10, 2.0)))
    )
    trajectory = ComponentActionTrajectory(
        trajectory_id="virtual-mujoco-plan",
        revision=0,
        source_kind="virtual_policy_chunk",
        source_instance_id=BASE_ROLE,
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=keyframes,
        source_fact_ids=("virtual-plan:0",),
        source_fact_hashes=("virtual-plan-hash:0",),
        source_activation_id="virtual-base-activation:1",
        source_activation_hash="virtual-base-activation-hash:1",
    )
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="mujoco-shadow-trajectory"),
        trajectory,
        offer_sequence=0,
    )
    receipt = fixture.anchor_receipt
    session_cut = fixture.runtime.capture_control_cut(cutoff_at=10.04)
    state = activate_residual_session(
        state,
        residual_session_id="virtual-mujoco-residual-session",
        activation_scope=fixture.activation,
        control_cut=session_cut,
        residual_anchor_receipt=receipt,
        first_editable_target_at=10.05,
    )
    stamp = lambda producer: TrajectoryCompositionContributorStamp(
        producer_id=producer,
        producer_epoch=1,
        sequence=0,
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
    )
    projection = capture_frozen_trajectory_composition_projection(
        fixture.runtime.policy_topology,
        state=state,
        composition_spec=fixture.composition_spec,
        nominal_role=BASE_ROLE,
        nominal_stamp=stamp(BASE_ROLE),
        residual_input=FrozenTrajectoryResidualInput(
            input_role=RELATIVE_ROLE,
            action=fixture.actions[RELATIVE_ROLE],
            contributor_stamp=stamp(RELATIVE_ROLE),
            residual_fact_id="virtual-residual:frozen",
            residual_fact_hash="virtual-residual-frozen-hash",
            residual_anchor_receipt_id=receipt.receipt_id,
            residual_anchor_receipt_hash=receipt.semantic_content_hash,
            effective_until=10.2,
            warp_spec=ResidualTemporalWarpSpec(
                spec_id="virtual-residual-ramp",
                ramp_duration_sec=0.05,
            ),
        ),
        held_inputs=(
            FrozenTrajectoryHeldInput(
                input_role=ABSOLUTE_ROLE,
                action=fixture.actions[ABSOLUTE_ROLE],
                contributor_stamp=stamp(ABSOLUTE_ROLE),
                effective_from=10.0,
                effective_until=10.2,
                source_kind="virtual_absolute_overlay",
                source_fact_ids=("virtual-absolute:0",),
                source_fact_hashes=("virtual-absolute-hash:0",),
                semantic_binding_id_by_kind={
                    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: receipt.receipt_id,
                },
                semantic_binding_hash_by_kind={
                    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: (
                        receipt.semantic_content_hash
                    ),
                },
            ),
        ),
        residual_anchor_lifecycle_scope_id="mujoco-operator-arms",
        cutoff_at=10.04,
    )
    return build_trajectory_corrected_preview(projection)


def test_corrected_preview_uses_admitted_context_and_never_dispatches() -> None:
    fixture = _fixture()
    try:
        fixture.controller.start()
        fixture.controller.offer_basis(lambda: _basis(fixture))
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        fixture.clock.value = 10.04
        cycle = fixture.controller.run_corrected_preview(
            lambda: _corrected_preview(fixture)
        )
        assert cycle.status == "projected"
        assert cycle.preview is not None
        assert tuple(point.output_status for point in cycle.preview.points) == (
            "ready",
            "ready",
        )
        assert all(
            point.context_snapshot.selected_context_by_channel[CONTEXT_CHANNEL]
            is not None
            for point in cycle.preview.points
        )
        assert cycle.preview.authoritative is False
        assert cycle.preview.promotable is False
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        fixture.controller.close()
        fixture.runtime.close()


def test_corrected_preview_applies_pending_rebase_at_future_boundary() -> None:
    fixture = _fixture()
    try:
        fixture.controller.start()
        fixture.controller.offer_basis(lambda: _basis(fixture))
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        fixture.clock.value = 10.04
        successor = ResidualAnchorRequest.for_activation(
            fixture.activation,
            request_generation=1,
            source_role=RELATIVE_ROLE,
            component_ids=fixture.output_schema.component_ids,
            source_binding_id="virtual-human-source",
            source_binding_hash="virtual-human-source-hash",
            absolute_action_schema_hash=(
                fixture.output_schema.semantic_content_hash
            ),
            relative_action_schema_hash=(
                fixture.relative_schema.semantic_content_hash
            ),
            requested_at=10.04,
            previous_request=fixture.anchor_request,
        )
        strategy = fixture.runtime.composition_strategy
        lifecycle_spec = strategy.residual_anchor_lifecycle_spec
        assert fixture.runtime.submit_residual_anchor_lifecycle(
            ResidualAnchorLifecycleDraft(
                runtime_session_id=fixture.runtime.config.runtime_session_id,
                control_domain_id=fixture.runtime.config.control_domain_id,
                producer_id=ALLOCATION_PRODUCER,
                producer_epoch=1,
                sequence=2,
                lifecycle_scope_id="mujoco-operator-arms",
                lifecycle_spec_hash=lifecycle_spec.semantic_content_hash,
                phase=RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
                effective_for_target_at=10.10,
                observed_at=10.04,
                observed_clock_domain_id="control.monotonic",
                anchor_request=successor,
            ),
            spec=lifecycle_spec,
            activation_scope=fixture.activation,
        ).accepted

        preview = _corrected_preview(fixture)
        assert tuple(point.output_status for point in preview.points) == (
            "ready",
            "unavailable",
        )
        assert preview.points[1].residual_anchor_lifecycle_snapshot.status == (
            RESIDUAL_ANCHOR_LIFECYCLE_PENDING
        )
        assert "residual_anchor_rebase_pending" in preview.points[1].reason
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        fixture.controller.close()
        fixture.runtime.close()
