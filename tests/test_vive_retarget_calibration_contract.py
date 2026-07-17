from __future__ import annotations

import numpy as np
import pytest

from teleop_runtime.calibration import ArmCalibration, TeleopCalibration


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
