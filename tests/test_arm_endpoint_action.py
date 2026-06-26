from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.arm_frames import ARM_ENDPOINT_FRAME
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.retargeters.pose_to_arm import PoseToArmRetargeter, PoseToArmRetargeterConfig
from teleop_runtime.schemas import ObservationChannel, TeleopObservation


class ArmEndpointActionTest(unittest.TestCase):
    def test_rm65_position_backend_prefers_arm_link6_for_endpoint_commands(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )

        self.assertEqual(backend._candidate_tcp_frames(ARM_ENDPOINT_FRAME)[0], "arm_link6")

    def test_pose_retargeter_outputs_endpoint_not_hand_base_pose(self) -> None:
        calibration = TeleopCalibration(
            arms={
                "left_arm": ArmCalibration(
                    name="left_arm",
                    side="left",
                    T_D_B=np.eye(4, dtype=float),
                )
            }
        )
        observation = TeleopObservation(
            channels={
                "left_wrist_pose": ObservationChannel(
                    channel_name="left_wrist_pose",
                    capability="pose",
                    side="left",
                    value=np.eye(4, dtype=float),
                    frame_name="desk",
                    timestamp=1.0,
                    source_name="test",
                )
            },
            timestamp=1.0,
        )
        retargeter = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="left",
                input_channel_name="left_wrist_pose",
                effector_name="left_arm",
            )
        )

        command = retargeter.build_command_from_target_pose_desk(
            target_pose_desk=np.eye(4, dtype=float),
            observation=observation,
            calibration=calibration,
        )

        left = command.commands["left_arm"]
        self.assertEqual(left.metadata["arm_controlled_frame"], ARM_ENDPOINT_FRAME)
        self.assertAlmostEqual(float(left.value[2, 3]), -0.009, places=6)
        self.assertAlmostEqual(float(left.metadata["target_pose_desk"][2, 3]), -0.009, places=6)
        self.assertAlmostEqual(
            float(left.metadata["target_pose_desk_retarget_frame"][2, 3]),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
