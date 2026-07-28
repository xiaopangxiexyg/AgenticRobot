from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from control_composition.composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionSpec,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    hand_action_spec_semantic_hash,
)
from dataset_contracts.completeness import (
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    HandActionSpec,
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    project_robot_action_v1_to_component_action,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    MujocoCoupledPoseToQposResolver,
    MujocoDeployConfig,
    MujocoDeployRuntime,
)
from sim.mujoco.ik import RM65CanonicalIK


pytestmark = [pytest.mark.unit, pytest.mark.contract]


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
                "left": np.full(6, 0.02, dtype=float),
                "right": np.full(6, -0.02, dtype=float),
            },
            target_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            realized_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            modification_reasons=(),
            metadata={"planner": "formal-component-shadow-fixture"},
        )


def _strategy():
    hand_spec = HandActionSpec.inspire_tac("left")
    rule_set = ComponentCompositionRuleSet(
        rule_set_id="mujoco-shadow-human-policy-component-v1",
        action_space=BIMANUAL_EE_POSE_V1,
        rules=(
            PoseComponentCompositionRule(
                component_id="left_arm",
                contributing_roles=("human", "policy"),
                weight_by_role={"human": 0.8, "policy": 0.2},
                max_translation_disagreement_m=0.10,
                max_rotation_disagreement_rad=0.50,
            ),
            QposComponentCompositionRule(
                component_id="left_hand",
                contributing_roles=("human", "policy"),
                weight_by_role={"human": 0.5, "policy": 0.5},
                value_contract_hash=hand_action_spec_semantic_hash(hand_spec),
            ),
        ),
    )
    composition_spec = CompositionSpec(
        spec_id="mujoco-shadow-human-policy-component-v2",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=("human", "policy"),
        required_roles=("human", "policy"),
        component_rule_set_hash=rule_set.semantic_content_hash,
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="mujoco-shadow-component-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash=composition_spec.semantic_content_hash,
        implementation_contract_hashes=(
            rule_set.semantic_content_hash,
        ),
    )
    return (
        hand_spec,
        composition_spec,
        FormalComponentCompositionStrategyV2(
            strategy_spec=strategy_spec,
            composition_spec=composition_spec,
            component_rule_set=rule_set,
            qpos_value_spec_by_component={"left_hand": hand_spec},
        ),
    )


def _input(
    runtime: MujocoDeployRuntime,
    *,
    producer_id: str,
    role: str,
    arm_x: float,
    hand_value: float,
    hand_spec: HandActionSpec,
) -> CompositionInputDraft:
    return CompositionInputDraft(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        producer_id=producer_id,
        producer_epoch=runtime.config.producer_epoch,
        sequence=0,
        input_role=role,
        action=project_robot_action_v1_to_component_action(
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                arm_pose_by_side={
                    "left": np.asarray(
                        [arm_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        dtype=np.float32,
                    )
                },
                hand_qpos_by_side={
                    "left": np.full(6, hand_value, dtype=np.float32)
                },
                active_effectors=("left_arm", "left_hand"),
                frame_by_effector={"left_arm": "left_arm_base"},
                controlled_frame_by_effector={"left_arm": "tool_tcp"},
                source=f"mujoco-shadow-{role}",
                metadata={
                    "hand_action_specs_by_side": {
                        "left": hand_spec.to_metadata()
                    }
                },
            ),
            action_schema=runtime.action_schema,
        ),
        produced_at=10.0,
        produced_clock_domain_id=runtime.config.clock_domain_id,
        effective_from=10.0,
        effective_until=10.5,
        effective_clock_domain_id=runtime.config.clock_domain_id,
    )


def test_shadow_composition_rejects_components_outside_runtime_schema() -> None:
    hand_spec, composition_spec, strategy = _strategy()
    clock = ManualClock()
    scene = ShadowScene()
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
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="formal-component-shadow-session",
            control_domain_id="formal-component-shadow-domain",
            embodiment_instance_id="formal-component-shadow-robot",
            enable_composer_authority=False,
            enable_replay_authority=False,
            reference_activation_profile_id=(
                SCHEDULED_REFERENCE_SEGMENT_SIM_V1
            ),
            pose_frame_contract_hash="virtual-pose-frame-v1",
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(
            RM65CanonicalIK(scene, ShadowPlanner())
        ),
        composition_strategy=strategy,
        clock=clock,
    )
    runtime.start()
    runtime.activate_policy_node_producer("human-shadow")
    with pytest.raises(ValueError, match="outside the runtime action schema"):
        _input(
            runtime,
            producer_id="human-shadow",
            role="human",
            arm_x=0.0,
            hand_value=0.2,
            hand_spec=hand_spec,
        )
    assert scene.physics_command_count == 0
    assert len(runtime.core.proposals) == 0
    runtime.close()
