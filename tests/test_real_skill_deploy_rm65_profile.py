from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from teleop_runtime.configs import build_annotated_dual_rm65_calibration
from teleop_runtime.skill_deploy import rm65_profile as profile_module
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _profile() -> ExclusiveRM65SkillDeploymentProfile:
    return ExclusiveRM65SkillDeploymentProfile(
        calibration_path=REPO_ROOT / "pyproject.toml",
        arm_ip_assignment_path=REPO_ROOT / "README.md",
        tool_profile_path_by_side={
            "left": (
                REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/rm65_left_inspire.json"
            ),
            "right": (
                REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/rm65_right_inspire.json"
            ),
        },
        envelope_profile_path=(
            REPO_ROOT
            / "teleop_runtime/configs/skill_deploy/"
            "agent_primitive_left_relative_tight_v1.json"
        ),
    )


def test_profile_resolves_existing_shared_inputs_without_device_construction(
    monkeypatch,
) -> None:
    calibration = build_annotated_dual_rm65_calibration()
    monkeypatch.setattr(
        profile_module.TeleopCalibration,
        "load_json",
        lambda path: calibration,
    )
    monkeypatch.setattr(
        profile_module,
        "load_assignment",
        lambda path: {
            "assignments": {
                "left": {"arm_ip": "192.0.2.10"},
                "right": {"arm_ip": "192.0.2.11"},
            }
        },
    )

    resolved = resolve_exclusive_rm65_skill_deployment_profile(_profile())

    assert resolved.calibration is calibration
    assert set(resolved.arm_ip_by_side) == {"left", "right"}
    assert set(resolved.tool_profile_by_side) == {"left", "right"}
    assert set(resolved.tool_tcp_pose_flange_m_rad_by_side) == {
        "left",
        "right",
    }
    assert all(
        value.shape == (6,)
        and np.all(np.isfinite(value))
        for value in resolved.tool_tcp_pose_flange_m_rad_by_side.values()
    )
    assert resolved.envelope_file_spec.profile_id.endswith("left-relative-tight-v1")
    assert resolved.position_profile.name == "rm65_bimanual_position_v1"
    assert len(resolved.semantic_content_hash) == 64


def test_profile_requires_absolute_files_and_both_arm_ips(monkeypatch) -> None:
    profile = _profile()
    relative = ExclusiveRM65SkillDeploymentProfile(
        calibration_path=Path("relative-calibration.json"),
        arm_ip_assignment_path=profile.arm_ip_assignment_path,
        tool_profile_path_by_side=profile.tool_profile_path_by_side,
        envelope_profile_path=profile.envelope_profile_path,
    )
    with pytest.raises(ValueError, match="calibration_path must be an absolute"):
        resolve_exclusive_rm65_skill_deployment_profile(relative)

    monkeypatch.setattr(
        profile_module.TeleopCalibration,
        "load_json",
        lambda path: build_annotated_dual_rm65_calibration(),
    )
    monkeypatch.setattr(
        profile_module,
        "load_assignment",
        lambda path: {
            "assignments": {
                "left": {"arm_ip": "192.0.2.10"},
                "right": {"arm_ip": ""},
            }
        },
    )
    with pytest.raises(ValueError, match="bind both left and right"):
        resolve_exclusive_rm65_skill_deployment_profile(profile)


def test_profile_import_has_no_realteleop_or_composition_dependency() -> None:
    script = """
import json
import sys
import teleop_runtime.skill_deploy.rm65_profile
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'control_composition'
    or name.startswith('control_composition.')
    or name == 'teleop_runtime.real_teleop'
    or name.startswith('teleop_runtime.real_teleop.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_legacy_rm65_ip_module_is_the_canonical_module_object() -> None:
    canonical = importlib.import_module(
        "teleop_runtime.hardware_config.rm65_arm_ips"
    )
    compatibility = importlib.import_module(
        "teleop_runtime.configs.rm65_arm_ips"
    )

    assert compatibility is canonical
    assert compatibility.load_assignment is canonical.load_assignment
