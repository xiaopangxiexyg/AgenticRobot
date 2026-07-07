from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.action import HandActionSpec
from teleop_runtime.deploy.actions import ActionSafetyFilter, ActionSafetyFilterConfig, NormalizedAction
from teleop_runtime.deploy.canonical_action import CanonicalDeployActionPipeline
from teleop_runtime.deploy.dispatch import dispatch_teleop_command
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
)


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
