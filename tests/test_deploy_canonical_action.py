from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.action import HandActionSpec
from teleop_runtime.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    FilteredNormalizedAction,
    NormalizedAction,
)
from teleop_runtime.deploy.canonical_action import CanonicalDeployActionPipeline
from teleop_runtime.deploy.dispatch import dispatch_teleop_command
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
)
from robot_motion.rm65 import RM65_ABSOLUTE_Q_MAX_RAD, RM65_ABSOLUTE_Q_MIN_RAD


def _observation(*, active_sides: tuple[str, ...], hands_enabled: bool) -> DeploymentObservation:
    state = np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32)
    valid = np.zeros((OBSERVATION_STATE_DIM,), dtype=bool)
    for side in active_sides:
        state[STATE_ARM_POSE_SLICES[side]] = np.array(
            [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        valid[STATE_ARM_POSE_SLICES[side]] = True
    metadata = {
        "hand_action_specs_by_side": {
            side: HandActionSpec.dmtac_gripper(side).to_metadata()
            for side in active_sides
        },
        "hand_spec_required": True,
    }
    return DeploymentObservation(
        timestamp=1.0,
        step_index=0,
        active_sides=active_sides,
        arms_enabled=True,
        hands_enabled=hands_enabled,
        state=state,
        state_valid=valid,
        metadata=metadata,
    )


class _CaptureBackend:
    def __init__(self) -> None:
        self.commands = []

    def send_command(self, command) -> None:
        self.commands.append(command.copy())


class DeployCanonicalActionTest(unittest.TestCase):
    def test_safety_config_rejects_negative_limits_and_reversed_workspace(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ActionSafetyFilterConfig(active_sides=("right",), max_arm_xyz_step_m=-0.1)
        with self.assertRaisesRegex(ValueError, "workspace_min"):
            ActionSafetyFilterConfig(
                active_sides=("right",),
                workspace_min=(1.0, 0.0, 0.0),
                workspace_max=(0.0, 1.0, 1.0),
            )

    def test_joint_keyframe_is_clipped_to_absolute_limits_before_resolution(self) -> None:
        observation = _observation(active_sides=("right",), hands_enabled=False)
        pipeline = CanonicalDeployActionPipeline(
            safety_filter=ActionSafetyFilter(
                ActionSafetyFilterConfig(
                    active_sides=("right",),
                    arms_enabled=True,
                    hands_enabled=False,
                    max_arm_joint_step_rad=0.0,
                    arm_joint_lower_rad=RM65_ABSOLUTE_Q_MIN_RAD,
                    arm_joint_upper_rad=RM65_ABSOLUTE_Q_MAX_RAD,
                )
            ),
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        )
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        target = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=np.float32) + 1.0

        result = pipeline.prepare(
            NormalizedAction(vector=vector, arm_joint_targets={"right": target}),
            observation,
            policy_name="unit_qpos",
        )

        self.assertEqual(result.faults, ())
        np.testing.assert_allclose(
            result.filtered.action.arm_joint_targets["right"],
            RM65_ABSOLUTE_Q_MAX_RAD,
        )
        self.assertTrue(result.filtered.metadata["arm"]["right"]["absolute_joint_limited"])

    def test_joint_constraint_intersection_faults_and_blocks_all_commands(self) -> None:
        observation = _observation(active_sides=("right",), hands_enabled=False)
        state = observation.state.copy()
        valid = observation.state_valid.copy()
        # The reference interval [1.05, 1.25] cannot intersect hard qpos <= 1.0.
        state[STATE_ARM_QPOS_SLICES["right"]] = np.full(6, 1.15, dtype=np.float32)
        valid[STATE_ARM_QPOS_SLICES["right"]] = True
        observation = DeploymentObservation(
            **{
                **observation.__dict__,
                "state": state,
                "state_valid": valid,
            }
        )
        pipeline = CanonicalDeployActionPipeline(
            safety_filter=ActionSafetyFilter(
                ActionSafetyFilterConfig(
                    active_sides=("right",),
                    arms_enabled=True,
                    hands_enabled=False,
                    max_arm_joint_step_rad=0.1,
                    arm_joint_lower_rad=(-1.0,) * 6,
                    arm_joint_upper_rad=(1.0,) * 6,
                )
            ),
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        )
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        result = pipeline.prepare(
            NormalizedAction(vector=vector, arm_joint_targets={"right": np.zeros(6)}),
            observation,
            policy_name="unit_qpos_intersection",
        )

        self.assertIn("right_arm_qpos_constraint_infeasible", result.faults)
        self.assertIsNone(result.command)

    def test_safety_overrides_can_tighten_but_cannot_disable_or_loosen_a_limit(self) -> None:
        observation = _observation(active_sides=("right",), hands_enabled=False)
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        vector[ACTION_ARM_POSE_SLICES["right"]] = np.array(
            [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        safety = ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.1,
                max_arm_rot_step_deg=0.0,
                max_arm_joint_step_rad=0.0,
            )
        )

        tightened = safety.filter_normalized(
            NormalizedAction(
                vector=vector,
                metadata={"safety_overrides": {"max_arm_xyz_step_m": 0.05}},
            ),
            observation,
        )
        self.assertEqual(tightened.faults, [])
        self.assertAlmostEqual(
            float(tightened.action.vector[ACTION_ARM_POSE_SLICES["right"]][0]),
            0.25,
        )

        disabled = safety.filter(
            NormalizedAction(
                vector=vector,
                metadata={"safety_overrides": {"max_arm_xyz_step_m": 0.0}},
            ),
            observation,
        )
        self.assertIn("safety_override_not_tightening:max_arm_xyz_step_m", disabled.faults)
        self.assertEqual(disabled.commands, {})

        loosened = safety.filter(
            NormalizedAction(
                vector=vector,
                metadata={"safety_overrides": {"max_arm_xyz_step_m": 0.2}},
            ),
            observation,
        )
        self.assertIn("safety_override_not_tightening:max_arm_xyz_step_m", loosened.faults)
        self.assertEqual(loosened.commands, {})

    def _pipeline(self, *, active_sides: tuple[str, ...], hands_enabled: bool) -> CanonicalDeployActionPipeline:
        return CanonicalDeployActionPipeline(
            safety_filter=ActionSafetyFilter(
                ActionSafetyFilterConfig(
                    active_sides=active_sides,
                    arms_enabled=True,
                    hands_enabled=hands_enabled,
                    max_arm_xyz_step_m=0.0,
                    max_arm_rot_step_deg=0.0,
                    max_arm_joint_step_rad=0.0,
                )
            ),
            active_sides=active_sides,
            arms_enabled=True,
            hands_enabled=hands_enabled,
        )

    def test_canonical_pipeline_maps_dmtac_q6_to_active_gripper_command(self) -> None:
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        vector[ACTION_ARM_POSE_SLICES["right"]] = np.array(
            [0.22, -0.1, 0.31, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        vector[ACTION_HAND_SLICES["right"]] = np.array(
            [0.42, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        observation = _observation(active_sides=("right",), hands_enabled=True)

        result = self._pipeline(active_sides=("right",), hands_enabled=True).prepare(
            NormalizedAction(vector=vector, metadata=dict(observation.metadata)),
            observation,
            policy_name="unit_policy",
        )

        self.assertEqual(result.faults, ())
        self.assertIsNotNone(result.command)
        assert result.command is not None
        self.assertEqual(set(result.command.commands), {"right_arm", "right_hand"})
        hand_command = result.command.commands["right_hand"]
        np.testing.assert_allclose(hand_command.value, [0.42])
        self.assertEqual(hand_command.metadata["hand_action_spec"]["hand_model"], "dmtac_gripper")
        self.assertEqual(result.command.metadata["source"], "deploy_canonical_robot_action")

    def test_canonical_pipeline_blocks_partial_command_when_action_has_fault(self) -> None:
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        vector[ACTION_ARM_POSE_SLICES["right"]] = np.array(
            [0.22, -0.1, 0.31, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        vector[ACTION_HAND_SLICES["right"]] = np.array(
            [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        observation = _observation(active_sides=("right",), hands_enabled=True)

        result = self._pipeline(active_sides=("right",), hands_enabled=True).prepare(
            NormalizedAction(vector=vector, metadata=dict(observation.metadata)),
            observation,
            policy_name="unit_policy",
        )

        self.assertIn("right_hand_action_invalid", result.faults)
        self.assertIsNone(result.command)
        self.assertTrue(result.metadata["command_blocked_by_faults"])

    def test_resolve_filtered_does_not_apply_keyframe_safety_twice(self) -> None:
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        vector[ACTION_ARM_POSE_SLICES["right"]] = np.array(
            [0.23, -0.1, 0.31, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        observation = _observation(active_sides=("right",), hands_enabled=False)
        pipeline = self._pipeline(active_sides=("right",), hands_enabled=False)
        filtered = FilteredNormalizedAction(
            action=NormalizedAction(
                vector=vector,
                metadata={
                    **dict(observation.metadata),
                    "position_reference_sequence": 7,
                    "position_interpolation_alpha": 0.5,
                },
            ),
            metadata={"arm": {"right": {"xyz_rate_limited": False}}},
        )

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("15 Hz safety filter must not run on a high-rate reference")

        pipeline.safety_filter.filter_normalized = fail_if_called  # type: ignore[method-assign]
        result = pipeline.resolve_filtered(
            filtered,
            observation,
            policy_name="unit_policy",
        )

        self.assertEqual(result.faults, ())
        self.assertIsNotNone(result.command)
        assert result.command is not None
        np.testing.assert_allclose(
            result.command.commands["right_arm"].metadata["target_xyzquat"],
            vector[ACTION_ARM_POSE_SLICES["right"]],
        )
        self.assertEqual(result.robot_action.metadata["position_reference_sequence"], 7)

    def test_real_dispatcher_routes_teleop_command_by_effector_name(self) -> None:
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        vector[ACTION_ARM_POSE_SLICES["right"]] = np.array(
            [0.22, -0.1, 0.31, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        observation = _observation(active_sides=("right",), hands_enabled=False)
        result = self._pipeline(active_sides=("right",), hands_enabled=False).prepare(
            NormalizedAction(vector=vector),
            observation,
            policy_name="unit_policy",
        )
        assert result.command is not None
        right_arm = _CaptureBackend()

        dispatch_teleop_command(
            result.command,
            arm_backends={"right": right_arm},
            hand_backends={},
        )

        self.assertEqual(len(right_arm.commands), 1)
        self.assertEqual(right_arm.commands[0].effector_name, "right_arm")


if __name__ == "__main__":
    unittest.main()
