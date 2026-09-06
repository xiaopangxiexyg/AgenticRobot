from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.sim_learning.state_bc.contracts import StateBCArtifactMetadata
from experiments.sim_learning.state_bc.model import fit_linear_state_bc
from experiments.sim_learning.state_bc.mujoco_stream import (
    StateBCMujocoStreamBridge,
)
from experiments.sim_learning.state_bc.stream import (
    StateBCStreamEvent,
    StateBCStreamingRunner,
)
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


class _IncrementingClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class _FakeMujocoRuntime:
    def __init__(self) -> None:
        self.gate = "ready"
        self.transitions = []
        self.submissions = []
        self.dispatches = 0
        self.authority_epoch = 0
        self.authoritative_port = self
        self.config = SimpleNamespace(initial_control_mode="policy")

    def capture_execution_gate_snapshot(self, *, now=None):
        return SimpleNamespace(status=self.gate, captured_at=now)

    def transition_execution_gate(
        self,
        status,
        *,
        reason,
        source_event_ids=(),
        force_cancellation=False,
    ):
        self.gate = status
        self.transitions.append((status, reason, tuple(source_event_ids)))
        cancellation = SimpleNamespace(cancellation_id="cancel:fixture") if status == "hold" else None
        receipt = (
            SimpleNamespace(complete=True, requires_fault_escalation=False)
            if status == "hold"
            else None
        )
        validation = SimpleNamespace(accepted=True) if status == "hold" else None
        return SimpleNamespace(
            snapshot=SimpleNamespace(status=status),
            cancellation=cancellation,
            cancellation_receipt=receipt,
            cancellation_dispatch_validation=validation,
            lower_event_ids=(f"lower:{len(self.transitions)}",),
        )

    def capture_receiver_state(self):
        return SimpleNamespace(current_authority_epoch=self.authority_epoch)

    def admit_authority_event(self, *, event_type, control_mode, reason):
        self.authority_epoch += 1
        return SimpleNamespace(
            event_id=f"authority:{self.authority_epoch}",
            authority_epoch=self.authority_epoch,
            event_type=event_type,
            control_mode=control_mode,
            reason=reason,
        )

    def submit_policy_action_emission(self, emission):
        self.submissions.append(emission)
        return SimpleNamespace(accepted=True, source_emission=emission)

    def step(self, *, now=None, physics_ticks_after_dispatch=0):
        self.dispatches += 1
        return SimpleNamespace(
            status="dispatched",
            decided_at=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )


class _StaticObservationSource:
    def __init__(self, observation):
        self.observation = observation

    def capture(self):
        return self.observation


class _LiveObservationSource:
    def __init__(self, observation):
        self.observation = observation
        self.step_index = int(observation.step_index)

    def capture(self):
        self.step_index += 1
        return replace(
            self.observation,
            timestamp=time.monotonic(),
            step_index=self.step_index,
        )


def _fake_stream_bridge():
    clock = _IncrementingClock()
    target_runtime = _FakeMujocoRuntime()
    policy_runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            hands_enabled=True,
            policy_absolute_frame="runtime_raw",
        ),
        monotonic_clock=clock,
    )
    bridge = StateBCMujocoStreamBridge(
        policy_runtime=policy_runtime,
        target=SimpleNamespace(runtime=target_runtime),
        preparer=DeployActionPreparer(
            DeployRobotActionAdapter(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
            )
        ),
        clock=clock,
    )
    return bridge, policy_runtime, target_runtime


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


def test_stream_bridge_routes_all_selected_sources_through_one_target():
    bridge, policy_runtime, target_runtime = _fake_stream_bridge()
    observation = _observation()
    adapter = StatePolicyAdapter(
        model=_DummyStatePolicy(),
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    rtc_calls = 0

    def rtc_policy(_state, nominal_model_action, _nominal_runtime_action):
        nonlocal rtc_calls
        rtc_calls += 1
        return None if rtc_calls == 1 else nominal_model_action

    human_action = adapter.predict(observation).actions[0]
    trace = StateBCStreamingRunner(
        observation_source=_StaticObservationSource(observation),
        policy_adapter=adapter,
        rtc_policy=rtc_policy,
        transition=bridge.transition,
        execute=bridge.execute,
    ).run(
        ["policy", "human", "policy", "rtc", "rtc"],
        human_actions={1: human_action},
    )

    assert trace.source_sequence() == ("policy", "human", "policy", "rtc")
    assert [event.execution_receipt.source for event in trace.action_events] == [
        "policy",
        "human",
        "policy",
        "rtc",
    ]
    assert len(target_runtime.submissions) == 4
    assert target_runtime.dispatches == 4
    assert [status for status, _, _ in target_runtime.transitions] == [
        "ready",
        "hold",
        "ready",
        "hold",
        "ready",
        "hold",
        "ready",
    ]
    assert target_runtime.gate == "ready"
    assert target_runtime.authority_epoch == 3
    assert policy_runtime.action_queue_length() == 0


def test_stream_bridge_rearms_a_retained_hold_before_retry_activation():
    bridge, _policy_runtime, target_runtime = _fake_stream_bridge()
    observation = _observation()
    bridge.transition(
        StateBCStreamEvent(
            step_index=1,
            event_kind="hold",
            requested_source="human",
            activation_id="activation:old",
            metadata={"previous_source": "policy"},
        ),
        observation,
    )

    receipt = bridge.transition(
        StateBCStreamEvent(
            step_index=1,
            event_kind="activation",
            requested_source="human",
            activation_id="activation:retry",
            metadata={"post_hold_observation": False},
        ),
        observation,
    )

    assert receipt.authority_event is not None
    assert target_runtime.authority_epoch == 1
    assert target_runtime.gate == "ready"


@pytest.mark.optional
def test_state_policy_output_reaches_existing_bimanual_mujoco_target():
    """Route policy, human and RTC through one authoritative MuJoCo target."""

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
        policy_runtime = PolicyDeployRuntime(
            PolicyDeployRuntimeConfig(
                active_sides=("left", "right"),
                hands_enabled=True,
                policy_absolute_frame="runtime_raw",
                policy_node_id=config.policy_node_id,
                policy_clock_domain_id=config.clock_domain_id,
                action_maximum_hold_sec=0.25,
                extend_final_keyframe_to_maximum_hold=True,
            ),
            monotonic_clock=time.monotonic,
        )
        target.runtime.start()
        bridge = StateBCMujocoStreamBridge(
            policy_runtime=policy_runtime,
            target=target,
            preparer=DeployActionPreparer(
                DeployRobotActionAdapter(
                    active_sides=("left", "right"),
                    arms_enabled=True,
                    hands_enabled=True,
                )
            ),
            clock=time.monotonic,
        )
        rtc_calls = 0

        def rtc_policy(_state, nominal_model_action, _nominal_runtime_action):
            nonlocal rtc_calls
            rtc_calls += 1
            return None if rtc_calls == 1 else nominal_model_action

        human_action = adapter.predict(observation).actions[0]
        trace = StateBCStreamingRunner(
            observation_source=_LiveObservationSource(observation),
            policy_adapter=adapter,
            rtc_policy=rtc_policy,
            transition=bridge.transition,
            execute=bridge.execute,
        ).run(
            ["policy", "human", "policy", "rtc", "rtc"],
            human_actions={1: human_action},
        )
        assert trace.source_sequence() == ("policy", "human", "policy", "rtc")
        for event in trace.action_events:
            receipt = event.execution_receipt
            assert receipt is not None
            assert receipt.submission.accepted is True
            assert receipt.dispatch.status == "dispatched"
            emission = receipt.preparation.emission
            assert emission is not None
            assert emission.pre_safety_robot_action is not None
            assert set(emission.pre_safety_robot_action.active_effectors) == {
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            }
        assert set(target.runtime.capture_qpos_measurement().qpos_by_side) == {
            "left",
            "right",
        }
    finally:
        target.runtime.close()
