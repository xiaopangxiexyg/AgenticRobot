from __future__ import annotations

import numpy as np
import pytest

from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
from teleop_runtime.retargeters.pose_to_arm import (
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from teleop_runtime.schemas import ObservationChannel, TeleopObservation


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def test_new_vive_retarget_names_preserve_legacy_transform_behavior() -> None:
    calibration = TeleopCalibration(
        T_W_O=_translation(0.1),
        T_O_D=_translation(0.2),
        arms={
            "left_arm": ArmCalibration(
                name="left_arm",
                side="left",
                T_D_B=_translation(0.3),
            )
        },
    )
    pose_world = _translation(1.0)

    assert calibration.vive_retarget_frame.name == "vive_retarget_frame"
    np.testing.assert_allclose(
        calibration.world_to_desk_pose(), calibration.world_to_vive_retarget_pose()
    )
    np.testing.assert_allclose(
        calibration.world_to_vive_retarget(pose_world),
        calibration.world_to_desk(pose_world),
    )
    pose_vr = calibration.world_to_vive_retarget(pose_world)
    np.testing.assert_allclose(
        calibration.vive_retarget_to_base(pose_vr, "left"),
        calibration.desk_to_base(pose_vr, "left"),
    )
    np.testing.assert_allclose(
        calibration.base_to_desk(
            calibration.desk_to_base(pose_vr, "left"),
            "left",
        ),
        pose_vr,
    )
    np.testing.assert_allclose(
        calibration.arms["left_arm"].T_VR_B,
        calibration.arms["left_arm"].T_D_B,
    )


def test_calibration_serializes_new_names_and_reads_new_only_payload() -> None:
    calibration = TeleopCalibration(
        T_W_O=np.eye(4),
        T_O_D=_translation(0.2),
        arms={
            "left_arm": ArmCalibration(
                name="left_arm",
                side="left",
                T_D_B=_translation(0.3),
            )
        },
    )
    payload = calibration.to_dict()
    assert payload["vive_retarget_frame"]["name"] == "vive_retarget_frame"
    assert payload["T_O_VR"] == payload["T_O_D"]
    assert payload["arms"]["left_arm"]["T_VR_B"] == payload["arms"]["left_arm"]["T_D_B"]

    payload.pop("desk_frame")
    payload.pop("T_O_D")
    payload["arms"]["left_arm"].pop("T_D_B")
    restored = TeleopCalibration.from_dict(payload)
    np.testing.assert_allclose(restored.T_O_VR, _translation(0.2))
    np.testing.assert_allclose(restored.arms["left_arm"].T_VR_B, _translation(0.3))


def test_old_desk_payload_is_normalized_to_vive_retarget_name() -> None:
    restored = TeleopCalibration.from_dict(
        {
            "desk_frame": {
                "name": "desk",
                "axes": {"x": "forward", "y": "left", "z": "up"},
            },
            "T_W_O": np.eye(4).tolist(),
            "T_O_D": np.eye(4).tolist(),
        }
    )
    assert restored.vive_retarget_frame.name == "vive_retarget_frame"


def test_conflicting_calibration_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting calibration aliases"):
        TeleopCalibration.from_dict(
            {
                "T_O_VR": np.eye(4).tolist(),
                "T_O_D": _translation(1.0).tolist(),
            }
        )


def test_raw_runtime_home_is_invariant_with_tracker_to_tcp_rotation() -> None:
    calibration = TeleopCalibration(
        T_W_O=np.eye(4),
        T_O_D=np.eye(4),
        arms={
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=np.eye(4),
            )
        },
    )
    retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            workspace_clip=False,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
        )
    )
    human_home = np.eye(4)
    robot_home = _translation(0.45, -0.20, 0.35)
    retargeter.set_home_anchors(
        human_pose_desk=human_home,
        robot_pose_desk=robot_home,
    )

    observation = TeleopObservation(
        channels={
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=human_home,
                frame_name="desk",
                timestamp=1.0,
                source_name="runtime-home-regression",
            )
        },
        timestamp=1.0,
    )
    command = retargeter.retarget(observation, calibration)
    target = np.asarray(
        command.commands["right_arm"].metadata["target_pose_desk"],
        dtype=float,
    )

    np.testing.assert_allclose(target, robot_home, atol=1e-12)


def test_logical_calibration_role_is_independent_of_upstream_tracker_role() -> None:
    calibration = TeleopCalibration(
        T_W_O=np.eye(4),
        T_O_D=np.eye(4),
        tracker_offsets={"right_wrist": _translation(0.25)},
        arms={
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=np.eye(4),
            )
        },
    )
    retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            tracker_role="right_wrist",
            workspace_clip=False,
        )
    )
    observation = TeleopObservation(
        channels={
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=np.eye(4),
                frame_name="vive_world",
                timestamp=1.0,
                source_name="mapped-single-vive",
                metadata={"tracker_role": "left_wrist"},
            )
        },
        timestamp=1.0,
    )

    command = retargeter.retarget(observation, calibration)
    arm = command.commands["right_arm"]

    np.testing.assert_allclose(arm.value, _translation(0.25), atol=1e-12)
    assert arm.metadata["tracker_role"] == "right_wrist"
    assert arm.metadata["source_tracker_role"] == "left_wrist"


def test_fix_arm_channel_authority_reaches_cartesian_arm_command() -> None:
    calibration = TeleopCalibration(
        T_W_O=np.eye(4),
        T_O_D=np.eye(4),
        arms={
            "left_arm": ArmCalibration(
                name="left_arm",
                side="left",
                T_D_B=np.eye(4),
            )
        },
    )
    retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            workspace_clip=False,
        )
    )
    observation = TeleopObservation(
        channels={
            "left_wrist_pose": ObservationChannel(
                channel_name="left_wrist_pose",
                capability="pose6d",
                side="left",
                value=np.eye(4),
                frame_name="vive_world",
                timestamp=1.0,
                source_name="fix-arm-regression",
                metadata={
                    "fix_arm_frozen": True,
                    "fix_arm_fixed_side": "left",
                },
            )
        },
        timestamp=1.0,
    )

    arm = retargeter.retarget(observation, calibration).commands["left_arm"]

    assert arm.metadata["fix_arm_frozen"] is True
    assert arm.metadata["fix_arm_fixed_side"] == "left"
