from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from robot_motion.rm65 import (
    DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
    DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES,
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    as_rm65_qpos,
    build_rm65_dual_arm_scene_profile,
    default_tool_profile_dir,
    endpoint_to_tool_transform,
    is_position_endpoint_profile,
    list_rm65_tool_profiles,
    load_rm65_tool_profile,
    normalize_sides,
    tool_tcp_pose_flange_m_rad,
    validate_qpos_by_side,
)


def test_default_collision_scene_is_dual_arm_hand_proxy_open():
    profile = build_rm65_dual_arm_scene_profile()

    assert DEFAULT_COLLISION_SCENE == "dual_arm_hand_proxy_open"
    assert profile.name == "dual_arm_hand_proxy_open"
    assert profile.collision_pairs == DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS
    assert len(profile.collision_pairs) == 12
    assert tuple(box.geometry_name for box in profile.fixed_scene_boxes) == (
        DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES
    )


def test_tool_profiles_load_from_existing_calibration_directory():
    profile_dir = default_tool_profile_dir()
    paths = list_rm65_tool_profiles()

    assert profile_dir == Path("hardware/rm65/utils/configs/calibrations").resolve()
    assert profile_dir.exists()
    assert profile_dir / "rm65_right_DMTAC_M.json" in paths


def test_tool_profile_tcp_transform_uses_shared_calibration_json():
    profile = load_rm65_tool_profile("DMTAC_M", side="right")
    pose = tool_tcp_pose_flange_m_rad(profile)
    transform = endpoint_to_tool_transform(profile)

    np.testing.assert_allclose(pose, np.array([0.0, 0.0, 0.14, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(transform[:3, 3], np.array([0.0, 0.0, 0.14]))
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)


def test_position_endpoint_profile_is_shared_zero_tcp():
    profile = load_rm65_tool_profile("position_endpoint", side="left")

    assert is_position_endpoint_profile(profile)
    np.testing.assert_allclose(tool_tcp_pose_flange_m_rad(profile), np.zeros(6))


def test_ambiguous_tool_profile_name_requires_side():
    with pytest.raises(ValueError, match="ambiguous"):
        load_rm65_tool_profile("DMTAC_M")


def test_rm65_qpos_layout_validation():
    qpos = as_rm65_qpos([0, 1, 2, 3, 4, 5])
    qpos_by_side = validate_qpos_by_side({"right": qpos, "left": np.zeros(6)})

    assert normalize_sides("both") == ("left", "right")
    np.testing.assert_allclose(qpos_by_side["right"], np.arange(6, dtype=float))
    np.testing.assert_allclose(qpos_by_side["left"], np.zeros(6))


def test_rm65_qpos_layout_rejects_bad_shape():
    with pytest.raises(ValueError, match="6 finite"):
        as_rm65_qpos([0.0, 1.0], label="bad_qpos")


def test_default_rm65_seeds_respect_active_urdf_joint_limits():
    urdf = Path(
        "assets/rm65_inspire_gripper_v3/"
        "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
    )
    root = ElementTree.parse(urdf).getroot()

    limits_by_side: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side, qpos in (
        ("left", DEFAULT_LEFT_ARM_QPOS),
        ("right", DEFAULT_RIGHT_ARM_QPOS),
    ):
        lower: list[float] = []
        upper: list[float] = []
        for joint_index in range(1, 7):
            joint = root.find(f"./joint[@name='{side}_arm_joint{joint_index}']")
            assert joint is not None
            limit = joint.find("limit")
            assert limit is not None
            lower.append(float(limit.attrib["lower"]))
            upper.append(float(limit.attrib["upper"]))
        values = np.asarray(qpos, dtype=float)
        assert np.all(values >= np.asarray(lower))
        assert np.all(values <= np.asarray(upper))
        limits_by_side[side] = (np.asarray(lower), np.asarray(upper))

    np.testing.assert_allclose(limits_by_side["left"][0], RM65_ABSOLUTE_Q_MIN_RAD)
    np.testing.assert_allclose(limits_by_side["left"][1], RM65_ABSOLUTE_Q_MAX_RAD)
    np.testing.assert_allclose(limits_by_side["right"][0], RM65_ABSOLUTE_Q_MIN_RAD)
    np.testing.assert_allclose(limits_by_side["right"][1], RM65_ABSOLUTE_Q_MAX_RAD)
