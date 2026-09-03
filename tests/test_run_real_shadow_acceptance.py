from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_real_shadow_acceptance as acceptance
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _profile():
    return resolve_exclusive_rm65_skill_deployment_profile(
        ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=(
                REPO_ROOT / "teleop_runtime/configs/hardware/rm65_arm_ips/latest.json"
            ),
            tool_profile_path_by_side={
                "left": REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json",
                "right": REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json",
            },
            envelope_profile_path=(
                REPO_ROOT
                / "teleop_runtime/configs/skill_deploy/agent_primitive_left_relative_tight_v1.json"
            ),
        )
    )


def test_pose_cli_rejects_missing_orientation() -> None:
    args = acceptance.build_parser().parse_args(
        [
            "--arm-ip-assignment",
            "/tmp/arms.json",
            "--dmtac-assignment",
            "/tmp/dmtac.json",
            "--target-xyz-m",
            "0.1",
            "0.2",
            "0.3",
        ]
    )
    with pytest.raises(ValueError, match="orientation"):
        acceptance._pose_from_cli(args)


def test_scene_cli_builds_sorted_cubes_and_clearance() -> None:
    args = acceptance.build_parser().parse_args(
        [
            "--arm-ip-assignment",
            "/tmp/arms.json",
            "--dmtac-assignment",
            "/tmp/dmtac.json",
            "--target-from-current",
            "--obstacle-clearance-m",
            "0.03",
            "--obstacle-cube-world-m",
            "0.2",
            "0.0",
            "0.4",
            "0.1",
            "--obstacle-cube-world-m",
            "-0.2",
            "0.0",
            "0.4",
            "0.2",
        ]
    )
    scene = acceptance._scene_from_cli(args)
    assert isinstance(scene, ObstacleSceneSnapshot)
    assert [item.obstacle_id for item in scene.obstacles] == ["cube-0", "cube-1"]
    assert scene.minimum_clearance_m == pytest.approx(0.03)
    assert scene.reference_frame_id == acceptance.OBSTACLE_FRAME_ID


def test_static_identity_and_binding_use_explicit_sim_contract() -> None:
    profile = _profile()
    identity = acceptance._build_identity(profile)
    assert identity.reference_frame_id == "desk"
    assert set(identity.tool_profile_hash_by_side) == {"left", "right"}
    binding = acceptance._build_binding(
        profile, SimpleNamespace(scene_hash="scene:test")
    )
    assert binding.scene_revision == "scene:test"
    assert binding.collision_profile_name == profile.collision_scene_profile
    assert binding.gripper_side_by_name


def test_jsonable_preserves_arrays_and_enums() -> None:
    payload = acceptance._jsonable(
        {"pose": np.eye(4), "path": Path("/tmp/result.json")}
    )
    assert payload["pose"][0][0] == 1.0
    assert payload["path"] == "/tmp/result.json"
