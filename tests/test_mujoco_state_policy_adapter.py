from __future__ import annotations

import time

import numpy as np
import pytest

from experiments.sim_learning.state_bc.contracts import StateBCArtifactMetadata
from experiments.sim_learning.state_bc.model import fit_linear_state_bc
from robot_motion.action import HandActionSpec
from robot_motion.deploy import (
    DeploymentObservation,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
)
from robot_motion.deploy.action_adapter import DeployRobotActionAdapter
from robot_motion.deploy.emission_source import DeployActionPreparer
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from sim.mujoco.deploy.state_policy_adapter import (
    STATE_BC_LAYOUT_HASH_V1,
    StatePolicyAdapter,
    StatePolicyAdapterConfig,
)


def _observation() -> DeploymentObservation:
    left_pose = np.asarray([0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right_pose = np.asarray([0.40, 0.50, 0.60, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state = np.full(38, np.nan, dtype=np.float32)
    state[6:13] = left_pose
    state[25:32] = right_pose
    state[13:19] = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32)
    state[32:38] = np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32)
    return DeploymentObservation(
        timestamp=1.5,
        step_index=3,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        state=state,
        state_valid=np.isfinite(state),
        arm_states={"left": {}, "right": {}},
        hand_states={
            "left": {"gripper_measured_scalar": 0.2},
            "right": {"gripper_measured_scalar": 0.4},
        },
        last_action=np.full(26, np.nan, dtype=np.float32),
    )


class _DummyStatePolicy:
    metadata = {"layout_hash": STATE_BC_LAYOUT_HASH_V1}
    artifact_hash = "sha256:fixture"

    def predict(self, state):
        state = np.asarray(state, dtype=np.float32)
        assert state.shape == (16,)
        action = state.copy()
        action[14] = 0.3
        action[15] = 0.7
        return action


def test_state_policy_adapter_projects_16d_model_output_to_existing_26d_action():
    adapter = StatePolicyAdapter(
        model=_DummyStatePolicy(),
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    prediction = adapter.predict(_observation())

    assert prediction.model_state.shape == (16,)
    assert prediction.model_action.shape == (16,)
    assert len(prediction.actions) == 1
    action = prediction.actions[0]
    assert action.vector.shape == (26,)
    np.testing.assert_allclose(action.vector[0:7], prediction.model_action[0:7])
    np.testing.assert_allclose(action.vector[7:14], prediction.model_action[7:14])
    assert action.vector[14] == pytest.approx(0.3)
    assert action.vector[20] == pytest.approx(0.7)
    assert action.metadata["state_bc_model_artifact_hash"] == "sha256:fixture"
    projected = adapter.project_model_action(prediction.model_action, source="fixture")
    np.testing.assert_allclose(projected.vector, action.vector)


def test_canonical_desk_mode_requires_an_explicit_pose_frame_contract():
    adapter = StatePolicyAdapter(model=_DummyStatePolicy())
    with pytest.raises(ValueError, match="pose frame contract"):
        adapter.predict(_observation())


def test_state_policy_adapter_validates_the_serialized_artifact_hash(tmp_path):
    states = np.tile(np.arange(16, dtype=np.float32), (3, 1))
    actions = states.copy()
    model = fit_linear_state_bc(
        states,
        actions,
        artifact_metadata=StateBCArtifactMetadata(
            selector_id="fixture-artifact",
            train_episode_count=2,
            test_episode_count=0,
            train_frame_count=3,
            test_frame_count=0,
        ),
    )
    artifact = model.save(tmp_path / "state_policy.npz")

    adapter = StatePolicyAdapter(
        artifact_path=artifact,
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    assert adapter.artifact_hash == model.artifact_hash


@pytest.mark.optional
def test_state_policy_output_reaches_existing_bimanual_mujoco_target():
    """Exercise existing adapter/profile/runtime seams without new control code."""

    pytest.importorskip("mujoco")

    from pathlib import Path

    from sim.mujoco.deploy import (
        MujocoComponentActionRobotActionV1Materializer,
        MujocoRobotActionV1ComponentIngressAdapter,
        MujocoSimHILTargetConfig,
        build_mujoco_component_execution_binding,
        build_mujoco_pose_qpos_component_target_profile,
        build_mujoco_policy_skill_motion_target,
        prepare_mujoco_sim_hil_target,
    )
    from sim.mujoco.deploy.deployment_observation import (
        MujocoDeploymentObservationAdapter,
        MujocoDeploymentObservationConfig,
    )

    config = MujocoSimHILTargetConfig(
        task_config_path=Path(
            "sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json"
        ),
        realtime_worker=False,
        allow_scripted_task_reset=False,
    )
    prepared = prepare_mujoco_sim_hil_target(config)
    runtime_contract = default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    profile = build_mujoco_pose_qpos_component_target_profile(
        prepared.scene,
        profile_id="state-bc.test.bimanual-pose-qpos.v1",
        arm_side_by_component={"left_arm": "left", "right_arm": "right"},
        gripper_name_by_component={
            "left_gripper": "left_gripper",
            "right_gripper": "right_gripper",
        },
        arm_reference_frame_by_component={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        profile,
        hand_action_spec_by_component={
            "left_gripper": HandActionSpec.dmtac_gripper("left"),
            "right_gripper": HandActionSpec.dmtac_gripper("right"),
        },
        support_arm_pose_provider=lambda: {},
        support_arm_sides=(),
        support_pose_contract_id="state-bc.test.no-support.v1",
    )
    ingress = MujocoRobotActionV1ComponentIngressAdapter(
        prepared.scene,
        profile,
        component_id_by_effector={
            "left_arm": "left_arm",
            "right_arm": "right_arm",
            "left_hand": "left_gripper",
            "right_hand": "right_gripper",
        },
    )
    binding = build_mujoco_component_execution_binding(
        profile,
        materializer,
        policy_action_ingress=ingress,
    )
    target = build_mujoco_policy_skill_motion_target(
        config=config,
        component_execution_binding=binding,
        prepared_target=prepared,
    )
    try:
        observation = MujocoDeploymentObservationAdapter(
            prepared.scene,
            MujocoDeploymentObservationConfig(
                active_sides=("left", "right"),
                tcp_body_by_side={
                    "left": prepared.scene.arm_specs["left"].controlled_body,
                    "right": prepared.scene.arm_specs["right"].controlled_body,
                },
                end_effector_sides=("left", "right"),
                gripper_name_by_side={
                    "left": "left_gripper",
                    "right": "right_gripper",
                },
                pose_frame_contract=runtime_contract,
            ),
            clock=time.monotonic,
        ).capture()
        adapter = StatePolicyAdapter(
            model=_DummyStatePolicy(),
            config=StatePolicyAdapterConfig(
                policy_absolute_frame="canonical_desk",
                runtime_pose_frame_contract=runtime_contract,
            ),
        )
        prediction = adapter.predict(observation)

        class _FixedPolicy:
            def predict(self, _observation):
                return prediction.actions[0]

        policy_runtime = PolicyDeployRuntime(
            PolicyDeployRuntimeConfig(
                active_sides=("left", "right"),
                hands_enabled=True,
                policy_absolute_frame="runtime_raw",
                policy_node_id=config.policy_node_id,
                policy_clock_domain_id=config.clock_domain_id,
            ),
            monotonic_clock=time.monotonic,
        )
        prediction_result = policy_runtime.predict_and_queue(
            _FixedPolicy(),
            observation,
            now=time.monotonic(),
        )
        assert prediction_result.accepted is True
        prepared_result = policy_runtime.prepare_next_action(
            observation,
            preparer=DeployActionPreparer(
                DeployRobotActionAdapter(
                    active_sides=("left", "right"),
                    arms_enabled=True,
                    hands_enabled=True,
                )
            ),
            policy_name="state_bc",
        )
        emission = prepared_result.emission
        assert emission is not None
        assert emission.pre_safety_robot_action is not None
        assert set(emission.pre_safety_robot_action.active_effectors) == {
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        }
        target.runtime.start()
        submission = target.runtime.submit_policy_action_emission(emission)
        result = target.runtime.step(physics_ticks_after_dispatch=1)
        assert submission.accepted is True
        assert result.status == "dispatched"
        assert set(target.runtime.capture_qpos_measurement().qpos_by_side) == {
            "left",
            "right",
        }
    finally:
        target.runtime.close()
