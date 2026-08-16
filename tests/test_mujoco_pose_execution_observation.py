from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from robot_motion.action import (
    ExecutionMeasurementObservation,
    ExecutionObservationPlan,
    ExecutionReferenceFrameObservation,
    HandActionSpec,
)
from control_station.composition.mujoco_sim_hil import (
    build_component_keyed_mujoco_sim_hil_contracts,
    build_mujoco_sim_hil_strategy_control_factory,
)
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoSimHILTargetConfig,
    build_mujoco_component_execution_binding,
    build_mujoco_sim_hil_target,
    build_rm65_right_cube_lift_component_target_profile,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)
from sim.mujoco.deploy.execution_pose_observation import (
    MUJOCO_FK_POSE_MEASUREMENT_STAGE_ID,
    MUJOCO_FK_POSE_REFERENCE_STAGE_ID,
    project_mujoco_fk_pose_measurement_observation,
    project_mujoco_fk_pose_reference_observation,
)
from sim.mujoco.deploy.pose_kinematics import (
    MujocoPoseKinematicsBinding,
    MujocoPoseObservationProjector,
)


def _projector():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <body name="endpoint">
                <joint name="arm_joint" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.01" mass="0.1"/>
                <body name="tool" pos="1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    live_data = mujoco.MjData(model)
    live_data.qpos[0] = 0.25
    mujoco.mj_forward(model, live_data)
    binding = MujocoPoseKinematicsBinding(
        component_id="arm/tool",
        qpos_source_id="arm-chain",
        joint_names=("arm_joint",),
        joint_qpos_addresses=(0,),
        reference_frame_id="robot/base",
        reference_body_name="base",
        reference_body_id=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base",
        ),
        controlled_frame_id="tool_tcp",
        endpoint_body_name="tool",
        endpoint_body_id=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tool",
        ),
        endpoint_to_controlled=np.eye(4),
        pose_schema_hash="arm-tool-pose-schema",
    )
    projector = MujocoPoseObservationProjector(
        mujoco_module=mujoco,
        model=model,
        bindings=(binding,),
        model_contract_id="fixture-one-joint-model",
        model_contract_hash="fixture-one-joint-model-hash",
    )
    return projector, live_data


def _qref_observation() -> ExecutionReferenceFrameObservation:
    return ExecutionReferenceFrameObservation(
        reference_frame_id="qref-observation:1",
        reference_stage_id="mujoco.active_qref_at_measurement",
        source_reference_id="qref-source:1",
        source_reference_hash="qref-source-hash:1",
        target_fact_id="target:1",
        target_fact_hash="target-hash:1",
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_fence_sequence=3,
        reference_revision=4,
        reference_sequence=5,
        temporal_semantics="point_sample",
        scheduled_for=10.0,
        generated_at=10.0,
        effective_from=None,
        effective_until=None,
        available_at=10.01,
        clock_domain_id="control.monotonic",
        value_by_component={"arm/tool": (math.pi / 2.0,)},
        representation_by_component={"arm/tool": "joint_position"},
        schema_hash_by_component={"arm/tool": "arm-joint-schema"},
    )


def _qpos_observation() -> ExecutionMeasurementObservation:
    return ExecutionMeasurementObservation(
        measurement_id="qpos-observation:1",
        source_measurement_hash="qpos-source-hash:1",
        control_domain_id="robot-0",
        measured_at=10.0,
        available_at=10.015,
        clock_domain_id="control.monotonic",
        value_by_component={"arm/tool": (0.0,)},
        representation_by_component={"arm/tool": "joint_position"},
        schema_hash_by_component={"arm/tool": "arm-joint-schema"},
    )


def test_private_mjdata_fk_does_not_mutate_live_simulator_state() -> None:
    projector, live_data = _projector()

    pose = projector.project(
        model_qpos=np.asarray(live_data.qpos).copy(),
        qpos_by_source={"arm-chain": (math.pi / 2.0,)},
        component_ids=("arm/tool",),
    )["arm/tool"]

    assert live_data.qpos[0] == pytest.approx(0.25)
    assert pose[:3] == pytest.approx((0.0, 1.0, 0.0), abs=1e-7)
    assert pose[5] == pytest.approx(math.sqrt(0.5), abs=1e-7)
    assert pose[6] == pytest.approx(math.sqrt(0.5), abs=1e-7)


def test_fk_pose_observations_keep_qpos_and_model_lineage() -> None:
    projector, live_data = _projector()
    qref = _qref_observation()
    qpos = _qpos_observation()
    model_qpos = np.asarray(live_data.qpos).copy()

    reference = project_mujoco_fk_pose_reference_observation(
        qref,
        projector=projector,
        model_qpos=model_qpos,
        qpos_by_source={"arm-chain": (math.pi / 2.0,)},
        component_ids=("arm/tool",),
        projected_at=10.02,
        observation_binding_id="pose-binding:1",
        observation_plan_hash="pose-plan:1",
    )
    measurement = project_mujoco_fk_pose_measurement_observation(
        qpos,
        projector=projector,
        model_qpos=model_qpos,
        qpos_by_source={"arm-chain": (0.0,)},
        component_ids=("arm/tool",),
        projected_at=10.025,
        observation_binding_id="pose-binding:1",
        observation_plan_hash="pose-plan:1",
    )

    assert reference.reference_stage_id == MUJOCO_FK_POSE_REFERENCE_STAGE_ID
    assert reference.scheduled_for == measurement.measured_at == 10.0
    assert reference.representation_by_component["arm/tool"] == "cartesian_pose"
    assert measurement.representation_by_component["arm/tool"] == "cartesian_pose"
    assert reference.schema_hash_by_component["arm/tool"] == (
        "arm-tool-pose-schema"
    )
    assert {
        fact.stage for fact in reference.parent_facts
    } == {
        "mujoco.active_qref_at_measurement",
        "mujoco.fk_projection_contract",
    }
    assert {
        fact.stage for fact in measurement.parent_facts
    } == {
        "mujoco.measured_qpos",
        "mujoco.fk_projection_contract",
    }
    assert measurement.measurement_id.startswith("mujoco-fk-pose-measurement:")
    assert MUJOCO_FK_POSE_MEASUREMENT_STAGE_ID == "mujoco.measured_qpos_fk_pose"


def test_sim_hil_target_binds_and_emits_requested_fk_pose_measurement() -> None:
    class Sink:
        def __init__(self) -> None:
            self.values: list[object] = []

        def publish_execution_observation(self, value: object) -> bool:
            self.values.append(value)
            return True

    plan = ExecutionObservationPlan(
        plan_id="sim-hil-right-arm-fk-pose-observation",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={
            "right_arm": "cartesian_pose",
        },
    )
    config = MujocoSimHILTargetConfig(
        task_config_path=(
            "sim/configs/tasks/"
            "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
        ),
        realtime_worker=False,
        composer_producer_id="fixture-sim-hil-composer",
        execution_observation_plan=plan,
        exposed_pose_sides=("right",),
    )
    prepared = prepare_mujoco_sim_hil_target(config)
    profile = build_rm65_right_cube_lift_component_target_profile(
        prepared.scene,
        prepared.task_config,
    )
    hand_spec = HandActionSpec.dmtac_gripper("right")
    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        profile,
        hand_action_spec_by_component={"right_gripper": hand_spec},
        support_arm_pose_provider=lambda: {
            "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                prepared.scene,
                prepared.task_config,
            )["left"],
        },
        support_arm_sides=("left",),
        support_pose_contract_id="fixture.measured_left_support_pose.v1",
    )
    execution_binding = build_mujoco_component_execution_binding(
        profile,
        materializer,
    )
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        profile.action_schema,
        composer_producer_id="fixture-sim-hil-composer",
        proposal_ttl_sec=0.25,
    )
    sink = Sink()
    target = build_mujoco_sim_hil_target(
        config=config,
        contracts=contracts,
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(contracts)
        ),
        hand_specs_by_side={"right": hand_spec},
        prepared_target=prepared,
        component_execution_binding=execution_binding,
        execution_observation_sink=sink,
        composition_input_action_schema_by_role={
            "human": profile.action_schema,
            "policy": profile.action_schema,
        },
    )
    try:
        capability = target.runtime.execution_observation_capability
        assert capability is not None
        assert capability.representation_by_component["right_arm"] == (
            "cartesian_pose"
        )
        assert capability.schema_hash_by_component["right_arm"] == (
            profile.action_schema.payload_schema_by_component[
                "right_arm"
            ].semantic_content_hash
        )
        target.runtime.start()
        target.runtime.step()
        measurements = [
            value
            for value in sink.values
            if isinstance(value, ExecutionMeasurementObservation)
        ]
        assert len(measurements) == 1
        measurement = measurements[0]
        assert measurement.representation_by_component["right_arm"] == (
            "cartesian_pose"
        )
        assert measurement.measured_at <= measurement.available_at
        assert {
            fact.stage for fact in measurement.parent_facts
        }.issuperset(
            {
                "measurement",
                "mujoco.measured_qpos",
                "mujoco.fk_projection_contract",
            }
        )
    finally:
        target.runtime.close()
