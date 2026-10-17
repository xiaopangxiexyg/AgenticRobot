from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from control_composition.trajectory import (
    ResidualTemporalWarpSpec,
    TrajectoryCompositionContributorStamp,
    activate_residual_session,
    empty_trajectory_editing_state,
    offer_nominal_trajectory,
)
from control_composition.lifecycle import CompositionSessionOperationSpec
from control_station import (
    CompositionSessionController,
    StationOperationResultDraft,
    admit_station_operation_result,
)
from control_station.composition import (
    MujocoSimHILComposition,
    MujocoSimHILProfileConfig,
    MujocoSimHILReadinessEvidence,
    MujocoTrajectoryPublicationController,
    MujocoTrajectoryPublicationProfile,
    MujocoTrajectoryPublicationRequest,
    build_mujoco_sim_hil_policy_binding,
    build_mujoco_sim_hil_composition_runtime_config,
    build_mujoco_sim_hil_strategy_control_factory,
    build_scoped_mujoco_sim_hil_composition,
)
from control_station.composition.mujoco_policy_human_residual import (
    MujocoPolicyHumanResidualProfileConfig,
    build_mujoco_policy_human_residual_profile,
)
from control_station.sources.teleop_pose import (
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
)
from robot_motion.action import (
    ArmActionSpec,
    ComponentAction,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ExecutionMeasurementObservation,
    ExecutionObservationPlan,
    ExecutionReferenceFrameObservation,
    HandActionSpec,
    project_component_action_trajectory,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)
from robot_motion.deploy import pose_matrix_to_xyzquat
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoRobotActionV1ComponentIngressAdapter,
    MujocoSimHILTargetConfig,
    build_mujoco_component_execution_binding,
    build_mujoco_sim_hil_target,
    build_rm65_right_cube_lift_component_target_profile,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)


class _PassiveSource:
    def __init__(self, spec: PassiveTeleopComponentSourceSpec) -> None:
        self.spec = spec
        self.started = False
        self.binding_receipt = PassiveTeleopComponentBindingReceipt(
            source_contract_hash=spec.semantic_content_hash,
            observer_binding_id=spec.observer_binding_id,
            session_mode="debug",
            rig_absent=True,
            observer_exclusive=True,
            dispatch_forbidden_states=("armed", "running"),
            authoritative=False,
            semantic_content_hash="virtual-trajectory-residual-source-binding",
        )

    def start(self) -> None:
        self.started = True

    def step(self, *, now: float):
        raise AssertionError(
            "The trajectory publication test does not poll devices."
        )

    def close(self) -> None:
        self.started = False


class _ObservationSink:
    def __init__(self) -> None:
        self.values: list[object] = []

    def publish_execution_observation(self, value: object) -> bool:
        self.values.append(value)
        return True


def _active_session_controller(
    composition: MujocoSimHILComposition,
) -> CompositionSessionController:
    source_slot_id = "virtual-trajectory-composer"
    contributor_source_slot_ids = (
        composition.human_source_spec.source_id,
        composition.policy_input_adapter.binding.source_slot_id,
    )
    execution_binding_hash = (
        composition.runtime.component_target_capability.semantic_content_hash
    )
    controller = CompositionSessionController(
        composer_source_slot_id=source_slot_id,
        contributor_source_slot_ids=contributor_source_slot_ids,
        composition_spec_hash=(
            composition.contracts.composition_spec.semantic_content_hash
        ),
        authority_gate_spec_hash=(
            composition.contracts.authority_gate_spec.semantic_content_hash
        ),
        execution_binding_hash=execution_binding_hash,
        clock=composition.runtime.clock,
        clock_domain_id=composition.runtime.config.clock_domain_id,
    )
    spec = CompositionSessionOperationSpec(
        session_key="virtual-trajectory-session",
        composer_source_slot_id=source_slot_id,
        contributor_source_slot_ids=contributor_source_slot_ids,
        composition_spec_hash=(
            composition.contracts.composition_spec.semantic_content_hash
        ),
        authority_gate_spec_hash=(
            composition.contracts.authority_gate_spec.semantic_content_hash
        ),
        execution_binding_hash=execution_binding_hash,
        capture_boundary_id="virtual-trajectory-episode:0",
    )
    controller.register(spec)
    available_at = float(composition.runtime.clock())
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="virtual-start-trajectory-session",
            operation_kind="start_composition_session",
            subject_id=source_slot_id,
            succeeded=True,
            completed_at=available_at,
            completed_clock_domain_id=(
                composition.runtime.config.clock_domain_id
            ),
            status="composition_session_ready",
            evidence_refs=("virtual-trajectory-session-ready",),
        ),
        runtime_session_id=composition.runtime.config.runtime_session_id,
        control_domain_id=composition.runtime.config.control_domain_id,
        available_at=available_at,
        available_clock_domain_id=(
            composition.runtime.config.clock_domain_id
        ),
        admission_sequence=0,
    )
    transition = controller.complete_start_operation(
        operation_spec_id=spec.operation_spec_id,
        result=result,
        completion_accepted=True,
        source_fact_id=result.operation_result_id,
        source_fact_hash=result.semantic_content_hash,
        start_intent_id="virtual-start-trajectory-session",
    )
    assert transition is not None
    assert controller.current_state is not None
    assert controller.current_state.phase == "active"
    return controller


def _source_spec(runtime) -> PassiveTeleopComponentSourceSpec:
    hashes = dict(
        runtime.target_runtime_action_contract.action_spec_hash_by_component
    )
    return PassiveTeleopComponentSourceSpec(
        source_id="virtual-trajectory-human-residual",
        observer_binding_id="virtual-trajectory-human-residual-observer",
        source_profile_id="virtual-trajectory-human-residual-profile",
        components=(
            PassiveTeleopComponentSpec.arm_pose(
                "right_arm",
                "right_arm",
                action_spec_hash=hashes["right_arm"],
                arm_action_spec=ArmActionSpec.tool_tcp(
                    "right",
                    reference_frame="right_arm_base",
                ),
            ),
            PassiveTeleopComponentSpec.hand_qpos(
                "right_gripper",
                "right_hand",
                action_spec_hash=hashes["right_gripper"],
                hand_action_spec=HandActionSpec.dmtac_gripper("right"),
            ),
        ),
        units_convention_id=(
            runtime.target_runtime_action_contract.units_convention_id
        ),
    )


def _action(schema, *, pose, residual_x: float | None = None) -> ComponentAction:
    pose_values = pose_matrix_to_xyzquat(pose)
    if pose_values is None:
        raise AssertionError("Fixture TCP pose cannot be represented as xyz+quat.")
    values = {}
    for component_id, payload in schema.payload_schema_by_component.items():
        if payload.representation_family == "cartesian_pose":
            values[component_id] = (
                tuple(float(value) for value in pose_values)
                if residual_x is None
                else (residual_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            )
        elif payload.representation_family == "joint_position":
            values[component_id] = tuple(0.0 for _ in range(payload.shape[0]))
        else:
            raise AssertionError(payload.representation_family)
    return ComponentAction(action_schema=schema, values_by_component=values)


def _trajectory(
    schema,
    *,
    action: ComponentAction,
    created_at: float,
    first_target: float,
    second_target: float,
    activation_id: str,
    activation_hash: str,
    producer_id: str,
) -> ComponentActionTrajectory:
    keyframes = (
        ComponentActionTrajectoryKeyframe(
            target_effective_at=first_target,
            effective_until=second_target,
            action=action,
            source_index=0,
            source_fact_ids=("virtual-chunk-keyframe:0",),
            source_fact_hashes=("virtual-chunk-keyframe-hash:0",),
        ),
        ComponentActionTrajectoryKeyframe(
            target_effective_at=second_target,
            effective_until=second_target + 0.25,
            action=action,
            source_index=1,
            source_fact_ids=("virtual-chunk-keyframe:1",),
            source_fact_hashes=("virtual-chunk-keyframe-hash:1",),
        ),
    )
    return ComponentActionTrajectory(
        trajectory_id="virtual-locked-policy-chunk:0",
        revision=0,
        source_kind="virtual_policy",
        source_instance_id=producer_id,
        clock_domain_id="mujoco.sim_hil.monotonic",
        created_at=created_at,
        keyframes=keyframes,
        source_fact_ids=("virtual-policy-plan:0",),
        source_fact_hashes=("virtual-policy-plan-hash:0",),
        source_activation_id=activation_id,
        source_activation_hash=activation_hash,
        source_immutable_through=second_target,
    )


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _build_running_fixture():
    clock = ManualClock()
    observation_plan = ExecutionObservationPlan(
        plan_id="locked-chunk-right-arm-fk-pose-observation",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={
            "right_arm": "cartesian_pose",
        },
    )
    target_config = MujocoSimHILTargetConfig(
        task_config_path=Path(
            "sim/configs/tasks/"
            "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
        ),
        realtime_worker=False,
        target_admission_margin_sec=0.20,
        execution_observation_plan=observation_plan,
        exposed_pose_sides=("right",),
    )
    prepared = prepare_mujoco_sim_hil_target(target_config)
    component_profile = build_rm65_right_cube_lift_component_target_profile(
        prepared.scene,
        prepared.task_config,
    )
    schema = component_profile.action_schema
    frames = {
        frame
        for payload in schema.payload_schema_by_component.values()
        for frame in (payload.reference_frame, payload.controlled_frame)
        if frame is not None
    }
    residual_profile = build_mujoco_policy_human_residual_profile(
        absolute_action_schema=schema,
        frame_graph=CoordinateFrameGraph(
            graph_id="virtual-mujoco-trajectory-frames",
            frames=tuple(CoordinateFrameSpec(frame) for frame in sorted(frames)),
            transforms=(),
        ),
        concrete_reference_frame_by_component={
            "right_arm": schema.payload_schema_by_component[
                "right_arm"
            ].reference_frame,
        },
        concrete_controlled_frame_by_component={
            "right_arm": schema.payload_schema_by_component[
                "right_arm"
            ].controlled_frame,
        },
        source_binding_id="virtual-human-residual-source",
        source_binding_hash="virtual-human-residual-source-hash",
        composer_producer_id=target_config.composer_producer_id,
        proposal_ttl_sec=0.20,
        config=MujocoPolicyHumanResidualProfileConfig(enabled=True),
    )
    rig_holder = {}

    def support_pose():
        rig = rig_holder["rig"]
        return rig.capture_scene(
            lambda scene: {
                "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    prepared.task_config,
                )["left"]
            }
        )

    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        component_profile,
        hand_action_spec_by_component={
            "right_gripper": HandActionSpec.dmtac_gripper("right"),
        },
        support_arm_pose_provider=support_pose,
        support_arm_sides=("left",),
        support_pose_contract_id="virtual-left-support-pose",
    )
    execution_binding = build_mujoco_component_execution_binding(
        component_profile,
        materializer,
    )
    observation_sink = _ObservationSink()
    target = build_mujoco_sim_hil_target(
        config=target_config,
        contracts=residual_profile.sim_hil_contracts,
        composition_runtime_config=(
            build_mujoco_sim_hil_composition_runtime_config(
                residual_profile.sim_hil_contracts
            )
        ),
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(
                residual_profile.sim_hil_contracts
            )
        ),
        hand_specs_by_side={"right": HandActionSpec.dmtac_gripper("right")},
        prepared_target=prepared,
        component_execution_binding=execution_binding,
        execution_observation_sink=observation_sink,
        monotonic_clock=clock,
        composition_input_action_schema_by_role=(
            residual_profile.input_action_schema_by_role
        ),
    )
    rig_holder["rig"] = target.runtime.rig
    source_spec = _source_spec(target.runtime)
    source = _PassiveSource(source_spec)
    ingress_projector = MujocoRobotActionV1ComponentIngressAdapter(
        target.scene,
        component_profile,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )
    policy_binding = build_mujoco_sim_hil_policy_binding(
        target.runtime,
        policy_active_effectors=("right_arm", "right_hand"),
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
        component_projection_contract_hash=(
            ingress_projector.semantic_content_hash
        ),
    )
    composition = build_scoped_mujoco_sim_hil_composition(
        runtime=target.runtime,
        human_source=source,
        human_source_spec=source_spec,
        human_input_transform=residual_profile.input_adapter,
        readiness_provider=lambda *, now: MujocoSimHILReadinessEvidence.all_ready(),
        policy_binding=policy_binding,
        policy_component_action_projector=ingress_projector,
        contracts=residual_profile.sim_hil_contracts,
        config=MujocoSimHILProfileConfig(
            enabled=True,
            profile_id="virtual-locked-chunk-residual-sim-hil",
            human_role=residual_profile.config.residual_role,
            policy_role=residual_profile.config.baseline_role,
            physics_ticks_after_dispatch=1,
        ),
        human_component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )
    composition.start()
    return target, composition, residual_profile, schema, observation_sink, clock


def test_sim_hil_factory_rejects_invalid_clock_before_preparation(monkeypatch) -> None:
    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("invalid clock reached native target preparation")

    monkeypatch.setattr(
        "sim.mujoco.deploy.sim_hil_target.prepare_mujoco_sim_hil_target",
        unexpected_preparation,
    )
    with pytest.raises(TypeError, match="monotonic_clock must be callable"):
        build_mujoco_sim_hil_target(
            config=MujocoSimHILTargetConfig(
                task_config_path=Path(
                    "sim/configs/tasks/"
                    "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
                )
            ),
            contracts=SimpleNamespace(),
            hand_specs_by_side={},
            monotonic_clock=None,
        )


def test_publication_profile_is_default_off_and_lazy() -> None:
    controller = MujocoTrajectoryPublicationController(
        profile=MujocoTrajectoryPublicationProfile(),
    )

    def should_not_run():
        raise AssertionError("default-off publication fetched a source")

    cycle = controller.publish_exact_target(should_not_run)

    assert cycle.status == "disabled"
    assert controller.state is None


def test_publication_profile_rejects_ambiguous_shadow_or_send_modes() -> None:
    with pytest.raises(ValueError, match="shadow"):
        MujocoTrajectoryPublicationProfile(
            profile_id="invalid-enabled-shadow",
            enabled=True,
            send_actions=False,
        )
    with pytest.raises(ValueError, match="requires enabled"):
        MujocoTrajectoryPublicationProfile(
            profile_id="invalid-disabled-send",
            enabled=False,
            send_actions=True,
        )


def test_exact_trajectory_input_identity_cannot_be_silently_substituted() -> None:
    evaluation = SimpleNamespace(
        input_snapshot=SimpleNamespace(
            selected_input_by_role={
                "policy": SimpleNamespace(
                    composition_input_id="trajectory-policy-input:0",
                    semantic_content_hash="trajectory-policy-input-hash:0",
                ),
            }
        )
    )

    MujocoSimHILComposition._validate_expected_input_selection(
        evaluation,
        {
            "policy": (
                "trajectory-policy-input:0",
                "trajectory-policy-input-hash:0",
            )
        },
    )
    with pytest.raises(ValueError, match="another input"):
        MujocoSimHILComposition._validate_expected_input_selection(
            evaluation,
            {
                "policy": (
                    "competing-policy-input:0",
                    "competing-policy-input-hash:0",
                )
            },
        )


@pytest.mark.slow_sim
def test_locked_chunk_residual_enters_existing_mujoco_chain_once() -> None:
    target, composition, residual_profile, schema, observation_sink, clock = (
        _build_running_fixture()
    )
    session_controller = None
    try:
        now = float(target.runtime.clock())
        pose = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            target.scene,
            target.task_config,
        )["right"]
        nominal_action = _action(schema, pose=pose)
        relative_schema = residual_profile.relative_contracts.residual_action_schema
        residual_action = _action(
            relative_schema,
            pose=pose,
            residual_x=0.004,
        )
        anchor_step = residual_profile.input_adapter.process_absolute_action(
            nominal_action,
            source_action_id="virtual-anchor-action",
            source_action_hash=nominal_action.semantic_content_hash,
            source_available_at=now,
            measured_at=now,
            measured_clock_domain_id=target.runtime.config.clock_domain_id,
            observed_at=now,
            observed_clock_domain_id=target.runtime.config.clock_domain_id,
        )
        assert anchor_step.status == "anchor_captured_relative_action_ready"
        anchor = residual_profile.input_adapter.anchor
        assert anchor is not None
        activation_cut = target.runtime.capture_control_cut(
            cutoff_at=float(target.runtime.clock())
        )
        policy_producer = composition.policy_input_adapter.binding.producer_id
        activation = next(
            record
            for record in reversed(
                target.runtime.core.producer_epochs.records_at_or_before(
                    activation_cut
                )
            )
            if record.producer_id == policy_producer
        )
        created_at = float(target.runtime.clock())
        first_target = created_at + 0.020
        due_target = created_at + 0.050
        trajectory = _trajectory(
            schema,
            action=nominal_action,
            created_at=created_at,
            first_target=first_target,
            second_target=due_target,
            activation_id=activation.activation_id,
            activation_hash=activation.semantic_content_hash,
            producer_id=policy_producer,
        )
        target.runtime.record_external_facts(
            (
                project_component_action_trajectory(
                    trajectory,
                    runtime_session_id=(
                        target.runtime.config.runtime_session_id
                    ),
                    control_domain_id=target.runtime.config.control_domain_id,
                    embodiment_instance_id=(
                        target.runtime.config.embodiment_instance_id
                    ),
                ),
            )
        )
        cut = target.runtime.capture_control_cut(
            cutoff_at=float(target.runtime.clock())
        )
        state = offer_nominal_trajectory(
            empty_trajectory_editing_state(reducer_id="mujoco-trajectory-editor"),
            trajectory,
            offer_sequence=0,
        )
        state = activate_residual_session(
            state,
            residual_session_id="virtual-residual-session:0",
            activation_scope=composition.activation_scope,
            control_cut=cut,
            residual_anchor_receipt=anchor.receipt,
            first_editable_target_at=first_target,
        )
        session_controller = _active_session_controller(composition)
        controller = MujocoTrajectoryPublicationController(
            profile=MujocoTrajectoryPublicationProfile(
                profile_id="mujoco-locked-chunk-residual-r3.1b",
                enabled=True,
                send_actions=True,
            ),
            composition=composition,
            initial_state=state,
            composition_spec=residual_profile.sim_hil_contracts.composition_spec,
            publication_barrier=session_controller.publication_barrier,
            nominal_role=residual_profile.config.baseline_role,
            residual_role=residual_profile.config.residual_role,
        )
        produced_at = float(target.runtime.clock())
        request = MujocoTrajectoryPublicationRequest(
            target_effective_at=due_target,
            nominal_stamp=TrajectoryCompositionContributorStamp(
                producer_id=policy_producer,
                producer_epoch=composition.policy_input_adapter.binding.producer_epoch,
                sequence=0,
                produced_at=produced_at,
                produced_clock_domain_id=target.runtime.config.clock_domain_id,
                policy_output_id="virtual-policy-output:0",
            ),
            residual_stamp=TrajectoryCompositionContributorStamp(
                producer_id=composition.config.human_producer_id,
                producer_epoch=composition.config.human_producer_epoch,
                sequence=0,
                produced_at=produced_at,
                produced_clock_domain_id=target.runtime.config.clock_domain_id,
                observation_cut_id="virtual-human-cut:0",
            ),
            residual_action=residual_action,
            residual_fact_id="virtual-human-residual:0",
            residual_fact_hash=residual_action.semantic_content_hash,
            residual_anchor_receipt_id=anchor.receipt.receipt_id,
            residual_anchor_receipt_hash=(
                anchor.receipt.semantic_content_hash
            ),
            residual_effective_until=due_target + 0.25,
            residual_warp_spec=ResidualTemporalWarpSpec(
                spec_id="virtual-residual-ramp",
                ramp_duration_sec=due_target - first_target,
            ),
        )

        # Expiry is rejected before input admission: sequence zero must remain
        # usable by the valid request below in the same real MuJoCo session.
        expired_request = replace(request, residual_effective_until=due_target)
        proposal_count = len(target.runtime.core.proposals)
        trace_count = len(target.runtime.trace_facts)

        expired = controller.publish_exact_target(lambda: expired_request)

        assert expired.status == "no_action"
        assert expired.input_preparation.reason == "residual_support_expired"
        assert controller.state.semantic_content_hash == state.semantic_content_hash
        assert len(target.runtime.core.proposals) == proposal_count
        assert expired.input_preparation.nominal_input_admission is None
        assert expired.input_preparation.residual_input_admission is None
        assert len(target.runtime.trace_facts) == trace_count

        cycle = controller.publish_exact_target(lambda: request)

        assert cycle.status == "proposal_admitted"
        assert observation_sink.values == []
        assert cycle.input_preparation.residual_sample.weight == pytest.approx(1.0)
        nominal_admission = cycle.input_preparation.nominal_input_admission
        assert nominal_admission is not None
        nominal_input = nominal_admission.composition_input
        assert nominal_input is not None
        assert nominal_input.source_fact_ids == (trajectory.fact_id,)
        residual_admission = cycle.input_preparation.residual_input_admission
        assert residual_admission is not None
        residual_input = residual_admission.composition_input
        assert residual_input is not None
        assert residual_input.source_fact_ids == (
            "virtual-human-residual:0",
            anchor.receipt.receipt_id,
        )
        receipt = cycle.composition_step.proposal_admission
        assert receipt is not None and receipt.accepted and receipt.proposal is not None
        assert controller.state.last_emitted_target_at == due_target
        assert controller.state.last_emitted_action_hash == (
            receipt.proposal.action.semantic_content_hash
        )
        assert controller.state.last_emitted_action_hash != (
            nominal_action.semantic_content_hash
        )
        # Exercise the same target time and support margins independently of
        # CPU scheduling; the native physics and all lifecycle gates stay live.
        clock.value = max(clock.value, due_target + 0.005)
        execution = target.runtime.step(
            now=float(target.runtime.clock()),
            physics_ticks_after_dispatch=100,
        )
        assert execution.status == "dispatched", execution.status
        assert execution.target_dispatch_result is not None
        pose_references = [
            value
            for value in observation_sink.values
            if isinstance(value, ExecutionReferenceFrameObservation)
        ]
        pose_measurements = [
            value
            for value in observation_sink.values
            if isinstance(value, ExecutionMeasurementObservation)
        ]
        assert pose_references and pose_measurements, [
            type(value).__name__ for value in observation_sink.values
        ]
        assert pose_references[-1].representation_by_component == {
            "right_arm": "cartesian_pose"
        }
        assert pose_references[-1].target_fact_id == (
            execution.target_dispatch_result.target.target_id
        )
        assert pose_references[-1].target_fact_hash == (
            execution.target_dispatch_result.target.semantic_content_hash
        )
        assert pose_measurements[-1].representation_by_component == {
            "right_arm": "cartesian_pose"
        }
        assert {
            fact.stage for fact in pose_references[-1].parent_facts
        }.issuperset(
            {
                "mujoco.active_qref_at_measurement",
                "mujoco.fk_projection_contract",
            }
        )
        assert {
            fact.stage for fact in pose_measurements[-1].parent_facts
        }.issuperset(
            {
                "mujoco.measured_qpos",
                "mujoco.fk_projection_contract",
            }
        )
        trace_event_types = {
            fact.event_type for fact in target.runtime.trace_facts
        }
        assert {
            "component_action_trajectory",
            "composition_input_admission",
            "composition_evaluation",
            "proposal_admission",
            "action_proposal",
            "arbitration_decision",
            "target_execution_request",
            "target_execution_submission",
            "target_execution_completion",
            "component_execution_target",
            "component_execution_target_admission",
            "dispatch_receipt",
        } <= trace_event_types
        proposal_count = len(target.runtime.core.proposals)

        duplicate = controller.publish_exact_target(lambda: request)

        assert duplicate.status == "no_action"
        assert duplicate.input_preparation.preparation.status == "already_emitted"
        assert len(target.runtime.core.proposals) == proposal_count
        assert controller.state.semantic_content_hash == (
            cycle.state_after.semantic_content_hash
        )
        assert cycle.emission_receipt_id is not None
        release_at = float(target.runtime.clock())

        def lower_hold_after_barrier_close():
            assert not session_controller.publication_barrier.snapshot().enabled
            return composition.request_hold(
                reason="virtual_trajectory_session_release",
                source_event_ids=(cycle.emission_receipt_id,),
                force_cancellation=True,
            )

        held = session_controller.prepare_safety_transition(
            intent_kind="hold",
            causal_station_fact_id="virtual-trajectory-pause-intent",
            source_fact_id="virtual-trajectory-pause-fact",
            source_fact_hash="virtual-trajectory-pause-fact-hash",
            occurred_at=release_at,
            clock_domain_id=target.runtime.config.clock_domain_id,
            lower_hold_or_fault=lower_hold_after_barrier_close,
        )
        assert held is not None and held.state_after_hash
        assert session_controller.current_state is not None
        assert session_controller.current_state.phase == "held"

        def must_not_fetch_after_release():
            raise AssertionError("Disabled publication fetched another source fact.")

        with pytest.raises(RuntimeError, match="barrier is disabled"):
            controller.publish_exact_target(must_not_fetch_after_release)
        gate = target.runtime.capture_execution_gate_snapshot(
            now=float(target.runtime.clock())
        )
        assert gate.status == "hold"

        stopped_at = float(target.runtime.clock())
        stop_result = admit_station_operation_result(
            StationOperationResultDraft(
                source_intent_id="virtual-end-trajectory-session",
                operation_kind="stop_composition_session",
                subject_id=session_controller.composer_source_slot_id,
                succeeded=True,
                completed_at=stopped_at,
                completed_clock_domain_id=(
                    target.runtime.config.clock_domain_id
                ),
                status="composition_session_stopped",
                evidence_refs=("virtual-trajectory-session-held",),
            ),
            runtime_session_id=target.runtime.config.runtime_session_id,
            control_domain_id=target.runtime.config.control_domain_id,
            available_at=stopped_at,
            available_clock_domain_id=(
                target.runtime.config.clock_domain_id
            ),
            admission_sequence=1,
        )
        ended = session_controller.complete_stop_operation(
            result=stop_result,
            completion_accepted=True,
            source_fact_id=stop_result.operation_result_id,
            source_fact_hash=stop_result.semantic_content_hash,
        )
        assert ended is not None
        assert session_controller.current_state is not None
        assert session_controller.current_state.phase == "ended"

        rearmed_at = float(target.runtime.clock())
        rearm_result = admit_station_operation_result(
            StationOperationResultDraft(
                source_intent_id="virtual-rearm-next-trajectory-session",
                operation_kind="rearm_composition_session",
                subject_id=session_controller.composer_source_slot_id,
                succeeded=True,
                completed_at=rearmed_at,
                completed_clock_domain_id=(
                    target.runtime.config.clock_domain_id
                ),
                status="composition_session_rearmed",
                evidence_refs=("virtual-next-trajectory-activation-ready",),
            ),
            runtime_session_id=target.runtime.config.runtime_session_id,
            control_domain_id=target.runtime.config.control_domain_id,
            available_at=rearmed_at,
            available_clock_domain_id=(
                target.runtime.config.clock_domain_id
            ),
            admission_sequence=2,
        )
        (
            session_controller.publication_barrier
            .enable_from_composition_session_rearm_result(rearm_result)
        )

        def must_not_fetch_for_next_activation():
            raise AssertionError("A stale trajectory controller fetched a source fact.")

        with pytest.raises(RuntimeError, match="another Station.*activation"):
            controller.publish_exact_target(must_not_fetch_for_next_activation)
    finally:
        if session_controller is not None:
            session_controller.close()
        composition.close()
