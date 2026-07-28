from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_composition import GuaranteedNextComponentAction
from control_composition.activation import build_composition_activation_scope
from control_station.composition import (
    MUJOCO_SIM_HIL_HUMAN_ROLE_V1,
    MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    MujocoSimHILComposition,
    MujocoSimHILCompositionStep,
    MujocoSimHILCycleResult,
    MujocoSimHILProfileConfig,
    MujocoSimHILReadinessEvidence,
    build_default_mujoco_sim_hil_contracts,
    build_scoped_mujoco_sim_hil_composition,
)
from control_station.composition.mujoco_policy_human_residual import (
    MujocoPolicyHumanResidualProfileConfig,
    build_mujoco_policy_human_residual_profile,
)
from control_station.sources import (
    PolicyCompositionInputAdapter,
    PolicyCompositionInputBinding,
)
from control_station.sources.teleop_pose import (
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
)
from robot_motion.action import (
    AuthoritativeControlStep,
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    ExecutionMeasurementObservation,
    ExecutionObservationPlan,
    ExecutionObservationWatermark,
    HandActionSpec,
    RobotAction,
    SourceValidationResult,
    project_robot_action_v1_to_component_action,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.deploy import NormalizedAction, PolicyActionEmission
from sim.mujoco.deploy import (
    MujocoSimHILTargetConfig,
    build_mujoco_pose_action_schema,
    build_mujoco_sim_hil_target,
)
from teleop_runtime.action.human_proposal import HumanCommandEmission


def _hand_specs() -> dict[str, HandActionSpec]:
    return {side: HandActionSpec.dmtac_gripper(side) for side in ("left", "right")}


def _action(*, arm_x: float = 0.0) -> RobotAction:
    return RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            side: np.asarray(
                [arm_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for side in ("left", "right")
        },
        hand_qpos_by_side={
            side: np.zeros(6, dtype=np.float32)
            for side in ("left", "right")
        },
        active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        frame_by_effector={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        source="sim-hil-policy-fixture",
        metadata={
            "hand_action_specs_by_side": {
                side: spec.to_metadata()
                for side, spec in _hand_specs().items()
            }
        },
    )


def _emission() -> PolicyActionEmission:
    return PolicyActionEmission(
        emission_id="policy-action-emission:fixture:0",
        admission_mode="on_emission_pop_v1",
        policy_node_id="sim-policy",
        inference_id="inference:0",
        policy_output_id="policy-output:0",
        chunk_id=0,
        chunk_keyframe_index=0,
        chunk_size=1,
        decoded_absolute_action=NormalizedAction(
            vector=np.zeros(1, dtype=np.float32)
        ),
        pre_safety_robot_action=_action(),
        source_validation=SourceValidationResult(accepted=True),
        representation_receipt=None,
        pre_safety_faults=(),
        produced_at=10.0,
        produced_clock_domain_id="sim-clock",
        emitted_at=10.0,
        target_effective_at=10.0,
        effective_from=10.0,
        effective_until=10.1,
        effective_clock_domain_id="sim-clock",
        schedule_spec_id="schedule:fixture",
        schedule_spec_hash="schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=1,
        semantic_content_hash="policy-emission-hash",
    )


def _human_source_spec(runtime) -> PassiveTeleopComponentSourceSpec:
    hashes = dict(
        runtime.target_runtime_action_contract.action_spec_hash_by_component
    )
    return PassiveTeleopComponentSourceSpec(
        components=(
            *(
                PassiveTeleopComponentSpec.arm_pose(
                    f"{side}_arm",
                    f"{side}_arm",
                    action_spec_hash=hashes[f"{side}_arm"],
                    arm_action_spec=ArmActionSpec.tool_tcp(
                        side,
                        reference_frame=f"{side}_arm_base",
                    ),
                )
                for side in ("left", "right")
            ),
            *(
                PassiveTeleopComponentSpec.hand_qpos(
                    f"{side}_hand",
                    f"{side}_hand",
                    action_spec_hash=hashes[f"{side}_hand"],
                    hand_action_spec=_hand_specs()[side],
                )
                for side in ("left", "right")
            ),
        ),
        units_convention_id=(
            runtime.target_runtime_action_contract.units_convention_id
        ),
    )


def _human_emission(
    *,
    sequence: int,
    emitted_at: float,
    arm_x: float,
    hand_qpos: float,
) -> HumanCommandEmission:
    command = TeleopCommand(timestamp=emitted_at)
    for side in ("left", "right"):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = arm_x
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="cartesian_pose",
                value=pose,
                frame_name=f"{side}_arm_base",
                metadata={
                    "arm_action_spec": ArmActionSpec.tool_tcp(
                        side,
                        reference_frame=f"{side}_arm_base",
                    ).to_metadata(),
                },
            )
        )
        values = np.full(6, hand_qpos, dtype=np.float32)
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="joint_vector",
                value=values,
                metadata={
                    "hand_action_spec": _hand_specs()[side].to_metadata(),
                },
            )
        )
    return HumanCommandEmission.capture(
        emission_sequence=sequence,
        emitted_at=emitted_at,
        contributions=(("virtual-vive-q6", command),),
        merged_command=command,
    )


def test_sim_hil_profile_is_default_off() -> None:
    assert MujocoSimHILProfileConfig().enabled is False


def test_sim_hil_cycle_hash_binds_only_target_facts_carried_at_return() -> None:
    composition_step = MujocoSimHILCompositionStep(
        decision_key="cycle-hash-fixture",
        target_effective_at=10.1,
        control_cut=None,
        readiness=None,
        evaluation=None,
        disposition=None,
        proposal_submission=None,
        hold_transition=None,
    )
    dispatch = SimpleNamespace(
        target=SimpleNamespace(target_id="target-1"),
        target_admission=SimpleNamespace(receipt_id="admission-1"),
        target_activation=None,
    )
    execution = AuthoritativeControlStep(
        status="dispatched",
        command_id="command-1",
        dispatch_receipt_id="dispatch-1",
        target_execution_request=SimpleNamespace(request_id="request-1"),
        target_submission_receipt=SimpleNamespace(receipt_id="submission-1"),
        target_completion_receipt=SimpleNamespace(receipt_id="completion-1"),
        target_dispatch_result=dispatch,
    )
    pending = MujocoSimHILCycleResult(
        cycle_index=0,
        human_step=None,
        human_submission=None,
        policy_submission=None,
        composition_step=composition_step,
        execution=execution,
    )
    activated = replace(
        pending,
        execution=replace(
            execution,
            target_dispatch_result=SimpleNamespace(
                target=dispatch.target,
                target_admission=dispatch.target_admission,
                target_activation=SimpleNamespace(
                    activation_id="activation-1"
                ),
            ),
        ),
    )
    another_request = replace(
        pending,
        execution=replace(
            execution,
            target_execution_request=SimpleNamespace(
                request_id="request-2"
            ),
        ),
    )

    assert pending.semantic_content_hash != activated.semantic_content_hash
    assert pending.semantic_content_hash != another_request.semantic_content_hash


def test_sim_hil_target_separates_policy_source_and_compose_controller_rates() -> None:
    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    composer_id = "control-station-sim-hil-composer"
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id=composer_id,
        proposal_ttl_sec=0.1,
        action_schema_hash=schema.semantic_content_hash,
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
            composer_producer_id=composer_id,
        ),
        contracts=contracts,
        hand_specs_by_side=hands,
    )
    try:
        timing = target.runtime.rig.config.timing
        assert target.runtime.config.decision_hz == 60.0
        assert timing.policy_hz == 15.0
        assert timing.sample_hz == 60.0
        assert timing.reference_hz == 60.0
        assert timing.ik_hz == 60.0
        assert timing.command_hz == 120.0
        assert target.runtime.config.enable_component_execution_handoff is True
        target_port = target.runtime.target_execution_port
        assert target_port is not None
        assert target_port.activation_mode == "required"
        assert (
            target_port.component_target_capability.semantic_content_hash
            == target.runtime.component_target_capability.semantic_content_hash
        )
        assert all(
            route.output_target_schema_hash
            == target.runtime.target_runtime_action_contract.semantic_content_hash
            for route in target.runtime.component_target_capability.component_routes
        )
    finally:
        target.runtime.close()


def test_sim_hil_target_accepts_a_profile_selected_compose_rate() -> None:
    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    composer_id = "control-station-sim-hil-composer-custom-rate"
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id=composer_id,
        proposal_ttl_sec=0.1,
        action_schema_hash=schema.semantic_content_hash,
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
            composer_producer_id=composer_id,
            # The historical position profile keeps the source-compatible
            # 15 Hz minimum while allowing a 45 Hz composed target.  The
            # downstream 60/60/120 clocks remain unchanged.
            position_control_profile_name="position",
            compose_target_hz=45.0,
        ),
        contracts=contracts,
        hand_specs_by_side=hands,
    )
    try:
        timing = target.runtime.rig.config.timing
        assert target.runtime.config.decision_hz == 45.0
        assert timing.policy_hz == 15.0
        assert timing.sample_hz == 45.0
        assert timing.reference_hz == 60.0
        assert timing.ik_hz == 60.0
        assert timing.command_hz == 120.0
    finally:
        target.runtime.close()


def test_sim_hil_target_installs_profile_role_specific_input_schemas() -> None:
    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    profile = build_mujoco_policy_human_residual_profile(
        absolute_action_schema=schema,
        frame_graph=CoordinateFrameGraph(
            graph_id="mujoco-profile-role-schema-fixture",
            frames=tuple(
                CoordinateFrameSpec(frame_id)
                for frame_id in (
                    "left_arm_base",
                    "left_arm/tool_tcp",
                    "right_arm_base",
                    "right_arm/tool_tcp",
                )
            ),
            transforms=(),
        ),
        concrete_reference_frame_by_component={
            f"{side}_arm": f"{side}_arm_base"
            for side in ("left", "right")
        },
        concrete_controlled_frame_by_component={
            f"{side}_arm": f"{side}_arm/tool_tcp"
            for side in ("left", "right")
        },
        source_binding_id="virtual-human-source",
        source_binding_hash="virtual-human-source-hash",
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.1,
        config=MujocoPolicyHumanResidualProfileConfig(enabled=True),
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
        ),
        contracts=profile.sim_hil_contracts,
        hand_specs_by_side=hands,
        composition_input_action_schema_by_role=(
            profile.input_action_schema_by_role
        ),
    )
    try:
        installed = {
            role: role_schema.semantic_content_hash
            for role, role_schema in (
                target.runtime.policy_topology.input_action_schema_by_role.items()
            )
        }
        assert installed == dict(
            profile.sim_hil_contracts.composition_spec
            .input_action_schema_hash_by_role
        )
        assert installed["policy"] == schema.semantic_content_hash
        assert installed["human_residual"] != schema.semantic_content_hash
    finally:
        target.runtime.close()


def test_sim_hil_human_transform_anchors_once_then_submits_relative_input() -> None:
    class PassiveSource:
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
                semantic_content_hash="virtual-passive-source-binding",
            )

        def start(self) -> None:
            self.started = True

        def step(self, *, now: float):
            raise AssertionError("This test submits immutable emissions directly.")

        def close(self) -> None:
            self.started = False

    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    profile = build_mujoco_policy_human_residual_profile(
        absolute_action_schema=schema,
        frame_graph=CoordinateFrameGraph(
            graph_id="mujoco-profile-transform-fixture",
            frames=tuple(
                CoordinateFrameSpec(frame_id)
                for frame_id in (
                    "left_arm_base",
                    "left_arm/tool_tcp",
                    "right_arm_base",
                    "right_arm/tool_tcp",
                )
            ),
            transforms=(),
        ),
        concrete_reference_frame_by_component={
            f"{side}_arm": f"{side}_arm_base"
            for side in ("left", "right")
        },
        concrete_controlled_frame_by_component={
            f"{side}_arm": f"{side}_arm/tool_tcp"
            for side in ("left", "right")
        },
        source_binding_id="virtual-human-source",
        source_binding_hash="virtual-human-source-hash",
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.1,
        config=MujocoPolicyHumanResidualProfileConfig(enabled=True),
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
        ),
        contracts=profile.sim_hil_contracts,
        hand_specs_by_side=hands,
        composition_input_action_schema_by_role=(
            profile.input_action_schema_by_role
        ),
    )
    source_spec = _human_source_spec(target.runtime)
    source = PassiveSource(source_spec)
    composition = build_scoped_mujoco_sim_hil_composition(
        runtime=target.runtime,
        human_source=source,
        human_source_spec=source_spec,
        human_input_transform=profile.input_adapter,
        readiness_provider=lambda *, now: (
            MujocoSimHILReadinessEvidence.all_ready()
        ),
        contracts=profile.sim_hil_contracts,
        config=MujocoSimHILProfileConfig(
            enabled=True,
            profile_id=profile.config.profile_id,
            human_role=profile.config.residual_role,
            policy_role=profile.config.baseline_role,
        ),
    )
    try:
        composition.start()
        first_at = float(target.runtime.clock()) + 0.001
        first = _human_emission(
            sequence=0,
            emitted_at=first_at,
            arm_x=0.2,
            hand_qpos=0.4,
        )
        assert composition.submit_human_emission(
            first,
            observed_at=first_at,
        ) is None
        assert composition.submit_human_emission(
            first,
            observed_at=first_at,
        ) is None
        assert len(target.runtime.policy_topology.composition_inputs) == 0

        second_at = first_at + 0.01
        second = _human_emission(
            sequence=1,
            emitted_at=second_at,
            arm_x=0.3,
            hand_qpos=0.1,
        )
        submission = composition.submit_human_emission(
            second,
            observed_at=second_at,
        )
        assert submission is not None and submission.accepted
        assert len(target.runtime.policy_topology.composition_inputs) == 1
        cut = target.runtime.capture_control_cut(
            cutoff_at=second_at + 0.001
        )
        records = (
            target.runtime.policy_topology.composition_inputs
            .records_at_or_before(cut)
        )
        assert len(records) == 1
        record = records[0]
        assert record.input_role == "human_residual"
        assert (
            record.action.action_schema.semantic_content_hash
            == profile.relative_contracts.residual_action_schema
            .semantic_content_hash
        )
        for side in ("left", "right"):
            np.testing.assert_allclose(
                record.action.value(f"{side}_arm").values[:3],
                (0.1, 0.0, 0.0),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                record.action.value(f"{side}_hand").values,
                np.full_like(
                    record.action.value(f"{side}_hand").values,
                    -0.3,
                ),
                atol=1e-6,
            )
    finally:
        composition.close()


def test_sim_hil_target_binds_non_off_execution_observation_plan() -> None:
    class ObservationSink:
        def __init__(self) -> None:
            self.values: list[object] = []

        def publish_execution_observation(self, observation: object) -> bool:
            self.values.append(observation)
            return True

    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    composer_id = "control-station-sim-hil-observation-binding"
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id=composer_id,
        proposal_ttl_sec=0.1,
        action_schema_hash=schema.semantic_content_hash,
    )
    plan = ExecutionObservationPlan(
        plan_id="sim-hil-observation:required",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
    )
    sink = ObservationSink()
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
            composer_producer_id=composer_id,
            execution_observation_plan=plan,
        ),
        contracts=contracts,
        hand_specs_by_side=hands,
        execution_observation_sink=sink,
    )
    try:
        binding = target.runtime.execution_observation_binding
        capability = target.runtime.execution_observation_capability
        assert binding is not None
        assert capability is not None
        assert binding.plan_hash == plan.semantic_content_hash
        assert binding.capability_hash == capability.semantic_content_hash
        assert binding.component_ids == ("right_arm",)
        assert sink.values == []
        target.runtime.start()
        target.runtime.step()
        assert any(
            isinstance(value, ExecutionMeasurementObservation)
            for value in sink.values
        )
        assert {
            value.stream
            for value in sink.values
            if isinstance(value, ExecutionObservationWatermark)
        } == {"reference", "measurement"}
    finally:
        target.runtime.close()


def test_sim_hil_default_off_rejects_before_starting_any_lower_runtime() -> None:
    class UnstartedRuntime:
        def __init__(self):
            self.start_count = 0

        def start(self):
            self.start_count += 1

    runtime = UnstartedRuntime()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = MujocoSimHILProfileConfig()
    composition._lock = threading.RLock()
    composition._started = False

    with pytest.raises(RuntimeError, match="default-off"):
        composition.start()
    assert runtime.start_count == 0


def test_sim_hil_lower_hold_failure_is_not_silently_dropped() -> None:
    class FailingRuntime:
        @staticmethod
        def transition_execution_gate(*_args, **_kwargs):
            raise RuntimeError("lower hold failed")

    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = FailingRuntime()

    with pytest.raises(RuntimeError, match="lower hold failed"):
        composition._hold(reason="test_lower_failure")


def test_sim_hil_existing_lower_hold_does_not_repeat_cancellation() -> None:
    class HeldRuntime:
        def __init__(self):
            self.transition_count = 0

        @staticmethod
        def clock():
            return 10.0

        @staticmethod
        def capture_execution_gate_snapshot(*, now):
            assert now == 10.0
            return SimpleNamespace(status="hold")

        def transition_execution_gate(self, *_args, **_kwargs):
            self.transition_count += 1
            raise AssertionError("existing HOLD must not be relinearized")

    runtime = HeldRuntime()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = MujocoSimHILProfileConfig(enabled=True)
    composition._lock = threading.RLock()
    composition._started = True
    composition._step_by_decision_key = {}
    composition._readiness_provider = lambda *, now: (
        MujocoSimHILReadinessEvidence.all_ready()
    )

    result = composition.compose_at(
        decision_key="already-held",
        target_effective_at=10.0,
    )

    assert result.blocked_reason == "execution_gate_not_ready:hold"
    assert result.hold_transition is None
    assert runtime.transition_count == 0


def test_default_sim_hil_contract_is_all_required_and_compound() -> None:
    contracts = build_default_mujoco_sim_hil_contracts(
        _hand_specs(),
        composer_producer_id="sim-hil-composer",
        proposal_ttl_sec=0.1,
    )

    assert contracts.composition_spec.input_roles == (
        MUJOCO_SIM_HIL_HUMAN_ROLE_V1,
        MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    )
    assert contracts.composition_spec.required_roles == contracts.composition_spec.input_roles
    assert contracts.authority_profile.component_coverage == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    assert contracts.authority_profile.selection_groups == (
        "arms",
        "left_hand",
        "right_hand",
    )
    assert contracts.authority_gate_spec.missing_behavior_by_role == {
        "human": "hold",
        "policy": "hold",
    }


def test_sim_hil_contract_can_bind_the_exact_mujoco_component_schema() -> None:
    schema = build_mujoco_pose_action_schema(_hand_specs())
    contracts = build_default_mujoco_sim_hil_contracts(
        _hand_specs(),
        composer_producer_id="sim-hil-composer",
        proposal_ttl_sec=0.1,
        policy_active_effectors=("right_arm", "right_hand"),
        action_schema_hash=schema.semantic_content_hash,
    )

    assert contracts.composition_spec.action_schema_hash == (
        schema.semantic_content_hash
    )


def test_policy_emission_becomes_non_authoritative_input_with_activation_lineage() -> None:
    binding = PolicyCompositionInputBinding(
        source_slot_id="sim-policy",
        role=MUJOCO_SIM_HIL_POLICY_ROLE_V1,
        runtime_session_id="sim-hil-session",
        control_domain_id="sim-hil-domain",
        producer_id="sim-policy",
        producer_epoch=1,
        policy_node_id="sim-policy",
        policy_clock_domain_id="sim-clock",
        component_catalog_hash="catalog-hash",
        action_space=BIMANUAL_EE_POSE_V1,
        required_active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        action_schema=project_robot_action_v1_to_component_action(
            _action()
        ).action_schema,
    )
    activation = build_composition_activation_scope(
        operation_spec_id="sim-hil",
        operation_spec_hash="sim-hil-contract",
        session_epoch=1,
        activated_at=10.0,
        clock_domain_id="sim-clock",
        input_barrier_at=10.0,
        required_roles=("human", "policy"),
    )
    adapter = PolicyCompositionInputAdapter(binding)
    draft = adapter.adapt(_emission(), activation_scope=activation)

    assert draft.input_role == "policy"
    assert draft.authority_candidate is False
    assert draft.composition_activation_id == activation.activation_id
    assert draft.composition_activation_hash == activation.semantic_content_hash
    assert draft.audit_metadata["chunk_id"] == 0
    assert draft.audit_metadata["inference_id"] == "inference:0"

    with pytest.raises(ValueError, match="another composition activation"):
        adapter.adapt(
            _emission(),
            activation_scope=build_composition_activation_scope(
                operation_spec_id="sim-hil",
                operation_spec_hash="sim-hil-contract",
                session_epoch=2,
                activated_at=11.0,
                clock_domain_id="sim-clock",
                input_barrier_at=11.0,
                required_roles=("human", "policy"),
            ),
        )


def test_policy_adapter_temporal_port_requires_explicit_guaranteed_next() -> None:
    action_schema = project_robot_action_v1_to_component_action(
        _action()
    ).action_schema
    binding = PolicyCompositionInputBinding(
        source_slot_id="sim-policy",
        role="policy",
        runtime_session_id="sim-hil-session",
        control_domain_id="sim-hil-domain",
        producer_id="sim-policy",
        producer_epoch=1,
        policy_node_id="sim-policy",
        policy_clock_domain_id="sim-clock",
        component_catalog_hash="catalog-hash",
        action_space=BIMANUAL_EE_POSE_V1,
        required_active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        action_schema=action_schema,
    )
    adapter = PolicyCompositionInputAdapter(binding)
    emission = _emission()

    fallback = adapter.adapt(emission)
    np.testing.assert_allclose(
        fallback.action.value("left_arm").values[0],
        0.0,
    )

    guaranteed = GuaranteedNextComponentAction(
        guarantee_id="sync-chunk:0:next:guaranteed",
        source_fact_id="sync-chunk:0:step:1",
        source_fact_hash="sync-chunk-step-1-hash",
        action=project_robot_action_v1_to_component_action(
            _action(arm_x=0.2),
            action_schema=action_schema,
        ),
        target_effective_at=10.1,
        clock_domain_id="sim-clock",
    )
    interpolated = adapter.adapt_interpolated_at(
        emission,
        target_effective_at=10.05,
        effective_until=10.06,
        sampled_at=10.049,
        guaranteed_next_step=guaranteed,
    )
    assert interpolated.sample.derivation_kind == (
        "interpolated_guaranteed_next"
    )
    np.testing.assert_allclose(
        interpolated.draft.action.value("left_arm").values[0],
        0.1,
    )
    assert interpolated.draft.source_fact_ids == (
        interpolated.sample.sample_id,
        emission.emission_id,
        guaranteed.source_fact_id,
    )
    with pytest.raises(TypeError, match="GuaranteedNextComponentAction"):
        adapter.adapt_interpolated_at(
            emission,
            target_effective_at=10.06,
            effective_until=10.07,
            sampled_at=10.059,
            guaranteed_next_step=None,  # type: ignore[arg-type]
        )


def test_policy_binding_rejects_partial_component_action() -> None:
    binding = PolicyCompositionInputBinding(
        source_slot_id="sim-policy",
        role="policy",
        runtime_session_id="sim-hil-session",
        control_domain_id="sim-hil-domain",
        producer_id="sim-policy",
        producer_epoch=1,
        policy_node_id="sim-policy",
        policy_clock_domain_id="sim-clock",
        component_catalog_hash="catalog-hash",
        action_space=BIMANUAL_EE_POSE_V1,
        required_active_effectors=("left_arm", "right_arm", "left_hand", "right_hand"),
        action_schema=project_robot_action_v1_to_component_action(
            _action()
        ).action_schema,
    )
    partial = _emission()
    partial_action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            side: np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for side in ("left", "right")
        },
        active_effectors=("left_arm", "right_arm"),
    )
    object.__setattr__(partial, "pre_safety_robot_action", partial_action)
    with pytest.raises(ValueError, match="exactly cover"):
        binding.validate_emission(partial)
