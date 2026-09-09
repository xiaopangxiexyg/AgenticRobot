from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.sim_learning.state_bc.evaluation import (
    STATE_BC_CONTROL_POLICY,
    STATE_BC_CONTROL_RTC,
    STATE_BC_CONTROL_TAKEOVER,
    TASK_ASSESSMENT_FAILURE,
    TASK_ASSESSMENT_SUCCESS,
    TASK_ASSESSMENT_UNKNOWN,
    StateBCEpisodeCondition,
    StateBCResetReceipt,
    StateBCTaskAssessment,
    run_matched_state_bc_evaluation,
)
from experiments.sim_learning.state_bc.mujoco_stream import StateBCMujocoStreamBridge
from experiments.sim_learning.state_bc.stream import StateBCStreamingRunner
from robot_motion.action import HandActionSpec
from robot_motion.deploy import PolicyDeployRuntime, PolicyDeployRuntimeConfig
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


class _HoldCurrentStatePolicy:
    metadata = {"layout_hash": STATE_BC_LAYOUT_HASH_V1}
    artifact_hash = "sha256:matched-evaluation-fixture"

    def predict(self, state):
        return np.asarray(state, dtype=np.float32).copy()


class _IncrementingClock:
    def __init__(self, value: float = 10.0, step: float = 0.001) -> None:
        self.value = float(value)
        self.step = float(step)

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


@dataclass
class _MujocoEpisodeContext:
    target: object
    runner: StateBCStreamingRunner
    initial_state_id: str
    initial_cube_z_m: float
    lift_delta_m: float

    def close(self) -> None:
        self.target.runtime.close()


def _build_context() -> _MujocoEpisodeContext:
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

    clock = _IncrementingClock()
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
        profile_id="state-bc.test.matched-evaluation.v1",
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
    target = build_mujoco_policy_skill_motion_target(
        config=config,
        component_execution_binding=build_mujoco_component_execution_binding(
            profile,
            materializer,
            policy_action_ingress=ingress,
        ),
        prepared_target=prepared,
        clock=clock,
    )
    observation_source = MujocoDeploymentObservationAdapter(
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
        clock=clock,
    )
    adapter = StatePolicyAdapter(
        model=_HoldCurrentStatePolicy(),
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
        monotonic_clock=clock,
    )
    initial = np.concatenate(
        (
            np.asarray(prepared.scene.data.qpos, dtype=np.float64),
            np.asarray(prepared.scene.data.qvel, dtype=np.float64),
        )
    )
    initial_state_id = f"sha256:{hashlib.sha256(initial.tobytes()).hexdigest()}"
    initial_cube_z_m = float(
        prepared.scene.body_pose(prepared.task_config.object_body_name)[2, 3]
    )
    rtc_calls = 0

    def rtc_policy(_state, nominal_model_action, _nominal_runtime_action):
        nonlocal rtc_calls
        rtc_calls += 1
        return None if rtc_calls == 1 else nominal_model_action

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
        clock=clock,
    )
    return _MujocoEpisodeContext(
        target=target,
        runner=StateBCStreamingRunner(
            observation_source=observation_source,
            policy_adapter=adapter,
            rtc_policy=rtc_policy,
            transition=bridge.transition,
            execute=bridge.execute,
        ),
        initial_state_id=initial_state_id,
        initial_cube_z_m=initial_cube_z_m,
        lift_delta_m=float(prepared.task_config.lift_delta_m),
    )


@pytest.mark.optional
def test_three_matched_conditions_reach_one_mujoco_target_and_task_predicate():
    """Exercise the panel without claiming cube-lift or intervention benefit."""

    pytest.importorskip("mujoco")
    contexts: dict[tuple[int, str], _MujocoEpisodeContext] = {}
    reference = _build_context()
    try:
        human_action = reference.runner.policy_adapter.predict(
            reference.runner.observation_source.capture()
        ).actions[0]
        conditions = (
            StateBCEpisodeCondition(
                "policy",
                STATE_BC_CONTROL_POLICY,
                ("policy",) * 5,
            ),
            StateBCEpisodeCondition(
                "takeover",
                STATE_BC_CONTROL_TAKEOVER,
                ("policy", "human", "human", "policy", "policy"),
                {1: human_action, 2: human_action},
            ),
            StateBCEpisodeCondition(
                "rtc",
                STATE_BC_CONTROL_RTC,
                ("policy", "rtc", "rtc", "policy", "policy"),
            ),
        )

        def reset(seed, condition):
            key = (seed, condition.control_mode)
            context = reference if condition.control_mode == STATE_BC_CONTROL_POLICY else _build_context()
            contexts[key] = context
            return StateBCResetReceipt(
                seed=seed,
                reset_id=f"mujoco:{seed}:{condition.control_mode}",
                initial_state_id=context.initial_state_id,
                metadata={"owner": "prepare_mujoco_sim_hil_target"},
            )

        def runner_factory(seed, condition, _reset):
            return contexts[(seed, condition.control_mode)].runner

        def assess(seed, condition, _reset, trace):
            context = contexts[(seed, condition.control_mode)]
            for event in trace.action_events:
                assert event.execution_receipt.dispatch.status == "dispatched"
            cube_z_m = float(
                context.target.scene.body_pose(
                    context.target.task_config.object_body_name
                )[2, 3]
            )
            lift_m = cube_z_m - context.initial_cube_z_m
            if not np.isfinite(lift_m):
                return StateBCTaskAssessment(
                    TASK_ASSESSMENT_UNKNOWN,
                    reason="cube height is non-finite",
                )
            return StateBCTaskAssessment(
                TASK_ASSESSMENT_SUCCESS
                if lift_m >= context.lift_delta_m
                else TASK_ASSESSMENT_FAILURE,
                metrics={"cube_lift_m": lift_m},
            )

        report = run_matched_state_bc_evaluation(
            seeds=(0,),
            conditions=conditions,
            reset=reset,
            runner_factory=runner_factory,
            assess=assess,
        )
        summary = report.summary()
        assert all(
            record.assessment.status == TASK_ASSESSMENT_FAILURE
            for record in report.records
        )
        assert summary[STATE_BC_CONTROL_POLICY]["action_events"] == 5
        assert summary[STATE_BC_CONTROL_TAKEOVER]["source_actions"]["human"] == 2
        assert summary[STATE_BC_CONTROL_RTC]["action_events"] == 4
        assert summary[STATE_BC_CONTROL_RTC]["rtc_no_action"] == 1
    finally:
        for context in {id(value): value for value in contexts.values()}.values():
            context.close()
        if not contexts:
            reference.close()
