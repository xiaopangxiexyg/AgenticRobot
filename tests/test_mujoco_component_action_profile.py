from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from control_composition.composition import (
    CompositionInputDraft,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
)
from control_station.composition import (
    build_component_keyed_mujoco_sim_hil_contracts,
    build_mujoco_sim_hil_composition_runtime_config,
    build_mujoco_sim_hil_strategy_control_factory,
)
import numpy as np
import pytest
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    ArmActionSpec,
    ComponentAction,
    HandActionSpec,
    RobotAction,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.pose_smoothing import RM65_POSITION_TARGET_POSE_SMOOTHING_V1
from sim.demonstration_editing import (
    DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1,
    DemonstrationComponentActionAdapter,
    DemonstrationComponentActionBinding,
    DemonstrationEdit,
    DemonstrationTrajectory,
    apply_demonstration_edit,
    sample_edited_trajectory,
)
from sim.mujoco.config import GripperSpec
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoRobotActionV1ComponentIngressAdapter,
    build_mujoco_component_execution_binding,
    build_mujoco_policy_skill_motion_target,
    build_mujoco_pose_component_target_profile,
    build_mujoco_pose_qpos_component_target_profile,
    build_rm65_right_cube_lift_component_target_profile,
    build_mujoco_sim_hil_target,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    MujocoSimHILTargetConfig,
    prepare_mujoco_sim_hil_target,
)
from sim.mujoco.deploy.pose_resolution import (
    MujocoCoupledPoseToQposResolver,
)
from sim.mujoco.ik import RM65CanonicalIK


def _scene():
    return SimpleNamespace(
        arm_specs={
            "left": object(),
            "right": object(),
        },
        gripper_specs={
            "left_gripper": GripperSpec(
                name="left_gripper",
                joint_names=(
                    "left_gripper_left_finger_joint",
                    "left_gripper_right_finger_joint",
                ),
                open_qpos=(0.0, 0.0),
                closed_qpos=(-0.035, -0.035),
                side="left",
            ),
            "right_gripper": GripperSpec(
                name="right_gripper",
                joint_names=(
                    "right_gripper_left_finger_joint",
                    "right_gripper_right_finger_joint",
                ),
                open_qpos=(0.0, 0.0),
                closed_qpos=(-0.035, -0.035),
                side="right",
            ),
        },
    )


def test_pose_resolver_identity_records_disabled_collision_constraints() -> None:
    ik = object.__new__(RM65CanonicalIK)
    ik.planner = SimpleNamespace(collision_pair_count=0)
    transformer = SimpleNamespace(
        binding_receipt=SimpleNamespace(semantic_content_hash="a" * 64)
    )

    resolver = MujocoCoupledPoseToQposResolver(
        ik,
        pose_transformer=transformer,
    )

    assert resolver.resolver_contract_id == (
        "mujoco-rm65-coupled-canonical-desk-pose-to-qpos-"
        "no-collision-v1"
    )


def test_sim_hil_target_config_keeps_pose_smoothing_rollback_explicit() -> None:
    assert MujocoSimHILTargetConfig(
        task_config_path=Path("task.json")
    ).target_pose_smoothing_enabled is True
    assert MujocoSimHILTargetConfig(
        task_config_path=Path("task.json"),
        target_pose_smoothing_enabled=False,
    ).target_pose_smoothing_enabled is False


def test_right_cube_lift_profile_uses_pose_and_physical_gripper_qpos() -> None:
    profile = build_rm65_right_cube_lift_component_target_profile(
        _scene(),
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )

    assert profile.action_schema.component_ids == (
        "right_arm",
        "right_gripper",
    )
    pose = profile.action_schema.payload_schema_by_component["right_arm"]
    qpos = profile.action_schema.payload_schema_by_component["right_gripper"]
    assert pose.representation_family == "cartesian_pose"
    assert pose.reference_frame == "right_arm_base"
    assert pose.controlled_frame == "tool_tcp"
    assert qpos.representation_family == "joint_position"
    assert qpos.shape == (2,)
    assert qpos.coordinate_names == (
        "right_gripper_left_finger_joint",
        "right_gripper_right_finger_joint",
    )
    assert qpos.units == ("m", "m")
    assert qpos.semantic_parameters["canonical_value"] == (
        "physical_joint_qpos"
    )
    assert profile.arm_side_by_component == {"right_arm": "right"}
    assert profile.gripper_name_by_component == {
        "right_gripper": "right_gripper"
    }


def test_generic_profile_cardinality_is_declared_by_maps_not_side_constants() -> None:
    profile = build_mujoco_pose_qpos_component_target_profile(
        _scene(),
        profile_id="fixture.pose_qpos",
        arm_side_by_component={
            "primary_tcp": "right",
            "support_tcp": "left",
        },
        gripper_name_by_component={
            "primary_tool": "right_gripper",
            "support_tool": "left_gripper",
        },
        arm_reference_frame_by_component={
            "primary_tcp": "right_arm_base",
            "support_tcp": "left_arm_base",
        },
    )

    assert profile.action_schema.component_ids == (
        "primary_tcp",
        "support_tcp",
        "primary_tool",
        "support_tool",
    )
    assert profile.action_schema.component_catalog.selection_group_ids == (
        "primary_tcp",
        "support_tcp",
        "primary_tool",
        "support_tool",
    )


def test_arms_only_profile_exposes_right_tcp_without_a_hand_component() -> None:
    profile = build_mujoco_pose_component_target_profile(
        _scene(),
        profile_id="fixture.right_arm_only_pose",
        arm_side_by_component={"right_arm": "right"},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
        selection_group_by_component={"right_arm": "arms"},
    )

    assert profile.action_schema.component_ids == ("right_arm",)
    assert profile.arm_side_by_component == {"right_arm": "right"}
    assert profile.gripper_name_by_component == {}
    assert profile.action_schema.semantic_parameters[
        "canonical_end_effector_representation"
    ] == "none"

    materializer = MujocoComponentActionRobotActionV1Materializer(
        _scene(),
        profile,
        hand_action_spec_by_component={},
        support_arm_pose_provider=lambda: {
            "left": np.asarray(
                [0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        support_arm_sides=("left",),
        support_pose_contract_id="fixture.measured-left-support.v1",
    )
    target = materializer(
        ComponentAction(
            action_schema=profile.action_schema,
            values_by_component={
                "right_arm": np.asarray(
                    [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            },
        ),
        timestamp=1.0,
        source="agent_skill",
    )
    binding = build_mujoco_component_execution_binding(profile, materializer)

    assert target.active_effectors == ("left_arm", "right_arm")
    assert target.hand_qpos_by_side == {}
    assert binding.output_effector_by_component == {"right_arm": "right_arm"}
    assert binding.support_only_output_effectors == ("left_arm",)


def test_component_keyed_sim_hil_contract_uses_schema_and_observe_only_pose_gate() -> None:
    profile = build_rm65_right_cube_lift_component_target_profile(
        _scene(),
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        profile.action_schema,
        composer_producer_id="fixture-composer",
        proposal_ttl_sec=0.2,
    )

    assert contracts.composition_spec.action_schema_hash == (
        profile.action_schema.semantic_content_hash
    )
    assert contracts.authority_profile.component_coverage == (
        "right_arm",
        "right_gripper",
    )
    pose_rule, qpos_rule = contracts.component_rule_set.rules
    assert isinstance(pose_rule, PoseComponentCompositionRule)
    assert pose_rule.max_translation_disagreement_m is None
    assert pose_rule.max_rotation_disagreement_rad is None
    assert isinstance(qpos_rule, QposComponentCompositionRule)
    assert qpos_rule.value_contract_hash == (
        profile.action_schema.payload_schema_by_component[
            "right_gripper"
        ].semantic_content_hash
    )


def test_legacy_source_ingress_expands_close_to_physical_qpos_without_clamp() -> None:
    scene = _scene()
    profile = build_rm65_right_cube_lift_component_target_profile(
        scene,
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )
    arm_spec = ArmActionSpec.tool_tcp(
        "right",
        reference_frame="right_arm_base",
    )
    hand_spec = HandActionSpec.dmtac_gripper("right")
    hand_values = np.zeros(hand_spec.slot_count, dtype=np.float32)
    hand_values[hand_spec.slot_map["gripper_close"]] = 2.0
    source = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "right": np.asarray(
                [0.4, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        hand_qpos_by_side={"right": hand_values},
        active_effectors=("right_arm", "right_hand"),
        frame_by_effector={"right_arm": "right_arm_base"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        metadata={
            "arm_action_specs_by_side": {"right": arm_spec.to_metadata()},
            "hand_action_specs_by_side": {"right": hand_spec.to_metadata()},
        },
    )

    adapted = MujocoRobotActionV1ComponentIngressAdapter(
        scene,
        profile,
    ).adapt(
        source,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )

    assert adapted.component_coverage == ("right_arm", "right_gripper")
    np.testing.assert_allclose(
        adapted.value("right_gripper").values,
        np.asarray([-0.07, -0.07], dtype=np.float32),
    )


def test_native_profile_materializes_only_at_private_coupled_execution_edge() -> None:
    scene = _scene()
    profile = build_rm65_right_cube_lift_component_target_profile(
        scene,
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )
    source = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "right": np.asarray(
                [0.4, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        hand_qpos_by_side={
            "right": np.asarray([0.25, 0, 0, 0, 0, 0], dtype=np.float32)
        },
        active_effectors=("right_arm", "right_hand"),
        frame_by_effector={"right_arm": "right_arm_base"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        metadata={
            "arm_action_specs_by_side": {
                "right": ArmActionSpec.tool_tcp(
                    "right",
                    reference_frame="right_arm_base",
                ).to_metadata()
            },
            "hand_action_specs_by_side": {
                "right": HandActionSpec.dmtac_gripper("right").to_metadata()
            },
        },
    )
    native = MujocoRobotActionV1ComponentIngressAdapter(
        scene,
        profile,
    ).adapt(
        source,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        scene,
        profile,
        hand_action_spec_by_component={
            "right_gripper": HandActionSpec.dmtac_gripper("right")
        },
        support_arm_pose_provider=lambda: {
            "left": np.asarray(
                [0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        support_arm_sides=("left",),
        support_pose_contract_id="fixture.measured-left-pose.v1",
    )
    policy_ingress = MujocoRobotActionV1ComponentIngressAdapter(
        scene,
        profile,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )
    compatibility = materializer(
        native,
        timestamp=1.0,
        source="fixture",
    )
    binding = build_mujoco_component_execution_binding(
        profile,
        materializer,
        policy_action_ingress=policy_ingress,
    )

    assert compatibility.action_space == BIMANUAL_EE_POSE_V1
    assert compatibility.active_effectors == (
        "left_arm",
        "right_arm",
        "right_hand",
    )
    assert set(compatibility.metadata) == {
        "arm_action_specs_by_side",
        "hand_action_specs_by_side",
    }
    np.testing.assert_allclose(
        compatibility.hand_qpos_by_side["right"],
        np.asarray([0.25, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    assert binding.action_schema is profile.action_schema
    assert tuple(
        item.effector_name for item in binding.execution_catalog.components
    ) == ("right_arm", "right_hand")
    assert binding.output_effector_by_component == {
        "right_arm": "right_arm",
        "right_gripper": "right_hand",
    }
    assert binding.policy_action_ingress_contract_hash == (
        policy_ingress.semantic_content_hash
    )
    projected = binding.policy_action_ingress(source)
    assert projected.component_coverage == (
        "right_arm",
        "right_gripper",
    )


def test_policy_ingress_must_exactly_invert_private_materializer_routing() -> None:
    scene = _scene()
    profile = build_rm65_right_cube_lift_component_target_profile(
        scene,
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        scene,
        profile,
        hand_action_spec_by_component={
            "right_gripper": HandActionSpec.dmtac_gripper("right")
        },
        support_arm_pose_provider=lambda: {
            "left": np.asarray(
                [0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        support_arm_sides=("left",),
        support_pose_contract_id="fixture.measured-left-pose.v1",
    )
    incomplete_ingress = MujocoRobotActionV1ComponentIngressAdapter(
        scene,
        profile,
        component_id_by_effector={"right_arm": "right_arm"},
    )

    with pytest.raises(ValueError, match="invert the materializer"):
        build_mujoco_component_execution_binding(
            profile,
            materializer,
            policy_action_ingress=incomplete_ingress,
        )


def test_policy_skill_target_rejects_unbound_policy_ingress_before_prepare() -> None:
    scene = _scene()
    profile = build_mujoco_pose_component_target_profile(
        scene,
        profile_id="fixture.right-arm-policy-target",
        arm_side_by_component={"right_arm": "right"},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        scene,
        profile,
        hand_action_spec_by_component={},
        support_arm_pose_provider=lambda: {
            "left": np.asarray(
                [0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        support_arm_sides=("left",),
        support_pose_contract_id="fixture.measured-left-pose.v1",
    )
    binding = build_mujoco_component_execution_binding(profile, materializer)

    with pytest.raises(ValueError, match="target-owned policy ingress"):
        build_mujoco_policy_skill_motion_target(
            config=MujocoSimHILTargetConfig(
                task_config_path=Path("does-not-need-to-exist.json"),
                allow_scripted_task_reset=False,
            ),
            component_execution_binding=binding,
        )


def test_native_component_profile_reaches_one_atomic_mujoco_target() -> None:
    target_config = MujocoSimHILTargetConfig(
        task_config_path=Path(
            "sim/configs/tasks/"
            "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
        ),
        realtime_worker=False,
        exposed_pose_sides=("right",),
    )
    prepared = prepare_mujoco_sim_hil_target(target_config)
    profile = build_rm65_right_cube_lift_component_target_profile(
        prepared.scene,
        prepared.task_config,
    )
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        profile.action_schema,
        composer_producer_id=target_config.composer_producer_id,
        proposal_ttl_sec=0.1,
        policy_active_components=profile.action_schema.component_ids,
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        profile,
        hand_action_spec_by_component={
            "right_gripper": HandActionSpec.dmtac_gripper("right"),
        },
        support_arm_pose_provider=lambda: {
            "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                prepared.scene,
                prepared.task_config,
            )["left"]
        },
        support_arm_sides=("left",),
        support_pose_contract_id="test.measured_left_support.v1",
    )
    target = build_mujoco_sim_hil_target(
        config=target_config,
        contracts=contracts,
        composition_runtime_config=(
            build_mujoco_sim_hil_composition_runtime_config(contracts)
        ),
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(contracts)
        ),
        hand_specs_by_side={
            "right": HandActionSpec.dmtac_gripper("right"),
        },
        prepared_target=prepared,
        component_execution_binding=build_mujoco_component_execution_binding(
            profile,
            materializer,
        ),
        composition_input_action_schema_by_role={
            "human": profile.action_schema,
            "policy": profile.action_schema,
        },
    )
    assert target.runtime.config.target_admission_margin_sec == pytest.approx(
        target_config.target_admission_margin_sec
    )
    assert target.runtime.pose_resolver.target_pose_smoother is not None
    assert (
        target.runtime.pose_resolver.target_pose_smoother.config
        == RM65_POSITION_TARGET_POSE_SMOOTHING_V1
    )
    current_pose = pose_matrix_to_xyzquat(
        capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            target.scene,
            target.task_config,
        )["right"]
    )
    assert current_pose is not None
    gripper_spec = target.scene.gripper_specs[
        target.task_config.right_gripper_name
    ]
    gripper_qpos = np.asarray(gripper_spec.open_qpos, dtype=np.float32)
    hand_payload = profile.action_schema.payload_schema_by_component[
        "right_gripper"
    ]
    demonstration = DemonstrationTrajectory(
        trajectory_id="zero-edit-native-target-fixture",
        reference_frame="right_arm_base",
        controlled_frame="tool_tcp",
        hand_component_id="right_gripper",
        hand_target_representation="physical_joint_qpos",
        hand_target_names=hand_payload.coordinate_names,
        hand_target_units=hand_payload.units,
        timestamps_sec=np.asarray([0.0, 0.1]),
        tool_pose_xyzquat_xyzw=np.stack((current_pose, current_pose), axis=0),
        hand_targets=np.stack((gripper_qpos, gripper_qpos), axis=0),
        hand_edit_start_index=0,
        hand_edit_end_index=1,
        source_kind="committed_mujoco_control_trace_safe_action",
        source_id="zero-edit-native-target-episode",
        source_content_hash="c" * 64,
    )
    edit = DemonstrationEdit(
        edit_id="zero-edit-native-target",
        producer_id="zero-edit-native-target-probe",
        sample_id="zero-edit-native-target-sample",
        translation_m=np.zeros(3),
        rotation_rotvec_rad=np.zeros(3),
        hand_final_delta=np.zeros(gripper_qpos.shape[0]),
    )
    reference = sample_edited_trajectory(
        apply_demonstration_edit(demonstration, edit),
        0.1,
    )
    adapted = DemonstrationComponentActionAdapter(
        DemonstrationComponentActionBinding(
            binding_id="test.zero-edit-to-native-mujoco-target.v1",
            action_schema=profile.action_schema,
            pose_component_id="right_arm",
            hand_component_id="right_gripper",
            demonstration_hand_target_representation=(
                "physical_joint_qpos"
            ),
        )
    ).adapt(reference)
    assert adapted.accepted
    assert adapted.action is not None
    assert adapted.receipt.edit_sample_id == "zero-edit-native-target-sample"
    action = adapted.action
    np.testing.assert_allclose(
        action.value("right_arm").values,
        current_pose,
    )
    np.testing.assert_allclose(
        action.value("right_gripper").values,
        gripper_qpos,
    )
    materializer(action, timestamp=0.0, source="test")

    target.runtime.start()
    try:
        now = float(target.runtime.clock())
        input_receipts = []
        for role in ("human", "policy"):
            producer_id = f"native-{role}"
            target.runtime.activate_policy_node_producer(producer_id)
            receipt = target.runtime.submit_composition_input(
                CompositionInputDraft(
                    runtime_session_id=target.runtime.config.runtime_session_id,
                    control_domain_id=target.runtime.config.control_domain_id,
                    producer_id=producer_id,
                    producer_epoch=1,
                    sequence=0,
                    input_role=role,
                    action=action,
                    produced_at=now,
                    produced_clock_domain_id=target.runtime.config.clock_domain_id,
                    effective_from=now,
                    effective_until=now + 1.0,
                    effective_clock_domain_id=target.runtime.config.clock_domain_id,
                    semantic_binding_id_by_kind={
                        DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1: (
                            adapted.receipt.receipt_id
                        )
                    },
                    semantic_binding_hash_by_kind={
                        DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1: (
                            adapted.receipt.semantic_content_hash
                        )
                    },
                    audit_metadata={
                        "demonstration_edit_sample_id": (
                            adapted.receipt.edit_sample_id
                        )
                    },
                )
            )
            assert receipt.accepted
            assert receipt.composition_input is not None
            input_receipts.append(receipt)
        for receipt in input_receipts:
            record = receipt.composition_input
            assert record is not None
            assert record.semantic_binding_id_by_kind == {
                DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1: (
                    adapted.receipt.receipt_id
                )
            }
            assert record.semantic_binding_hash_by_kind == {
                DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1: (
                    adapted.receipt.semantic_content_hash
                )
            }
            assert record.audit_metadata == {
                "demonstration_edit_sample_id": adapted.receipt.edit_sample_id
            }
        target_effective_at = (
            float(target.runtime.clock())
            + target.runtime.core.config.decision_schedule.target_lead_time_sec
        )
        submission = target.runtime.compose_and_submit(
            spec=contracts.composition_spec,
            target_effective_at=target_effective_at,
        )
        assert submission.proposal_admission is not None
        assert submission.proposal_admission.accepted

        step = target.runtime.step(now=float(target.runtime.clock()))

        assert step.computation is not None
        assert not step.computation.rejections, [
            (item.selection_group_id, item.reason)
            for item in step.computation.rejections
        ]
        assert step.status == "dispatched", {
            "computation": (
                None
                if step.computation is None
                else {
                    "output_status": step.computation.output_status,
                    "execution_gate_status": (
                        step.computation.execution_gate_status
                    ),
                    "selected_by_group": dict(
                        step.computation.selected_by_group
                    ),
                    "status_by_group": dict(
                        step.computation.status_by_group
                    ),
                    "rejections": [
                        (
                            item.selection_group_id,
                            item.reason,
                        )
                        for item in step.computation.rejections
                    ],
                    "decision_cutoff_at": (
                        step.computation.decision_cutoff_at
                    ),
                    "target_effective_at": (
                        step.computation.target_effective_at
                    ),
                }
            ),
        }
        assert step.target_dispatch_result is not None
        execution_target = step.target_dispatch_result.target
        assert execution_target is not None
        assert execution_target.component_dependencies == (
            "right_arm",
            "right_gripper",
        )
        assert set(
            execution_target.payload["output_values_by_effector"]
        ) == {"left_arm", "right_arm", "right_hand"}
        assert execution_target.payload[
            "authoritative_component_to_output_effector"
        ] == {
            "right_arm": "right_arm",
            "right_gripper": "right_hand",
        }
        assert execution_target.payload[
            "support_only_output_effectors"
        ] == ("left_arm",)
    finally:
        target.runtime.close()
