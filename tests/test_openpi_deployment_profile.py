from __future__ import annotations

import json
from pathlib import Path

import pytest

from teleop_runtime.deploy.openpi_application import _rm65_bimanual_dmtac_args
from teleop_runtime.deploy.openpi_deployment_profile import (
    OpenPIRealDeploymentProfileError,
    RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID,
    load_openpi_real_deployment_profile,
    load_openpi_real_deployment_profile_file,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]
REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    REPO_ROOT
    / "teleop_runtime/configs/deployments/openpi/rm65_bimanual_dmtac_v1.json"
)


def _common_expected_args() -> list[str]:
    return [
        "--active-sides",
        "both",
        "--end-effector-sides",
        "both",
        "--arm-execution-profile",
        "impedance_40hz",
        "--left-arm-ip",
        "192.168.101.21",
        "--right-arm-ip",
        "192.168.1.20",
        "--left-tool-profile",
        str(REPO_ROOT / "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"),
        "--right-tool-profile",
        str(REPO_ROOT / "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json"),
        "--enable-hands",
        "--hand-backend",
        "dmtac-gripper",
        "--dmtac-config",
        "latest",
        "--require-dmtac-config",
        "--dmtac-no-tactile",
        "--camera-mode",
        "multi",
        "--camera-config",
        str(REPO_ROOT / "teleop_runtime/configs/devices/realsense.json"),
        "--require-camera-frame",
        "--require-ready-observation",
        "--action-queue-policy",
        "replace",
        "--min-action-queue",
        "0",
        "--init-source",
        "current",
    ]


def test_registered_profile_reproduces_frozen_interactive_arguments() -> None:
    expected = _common_expected_args() + [
        "--wait-for-ready-confirm",
        "--policy-qpos-init-dataset",
        str(REPO_ROOT / "datasets/sorted_v22/bimanual-cube-new-success"),
        "--policy-qpos-init-episode",
        "0",
        "--policy-qpos-init-frame",
        "0",
        "--policy-qpos-init-source",
        "auto",
        "--policy-qpos-init-collision-check",
    ]

    assert _rm65_bimanual_dmtac_args(rtc_real_shadow=False) == expected
    assert "--send-actions" not in expected


def test_registered_profile_reproduces_frozen_rtc_shadow_arguments() -> None:
    expected = _common_expected_args() + [
        "--auto-start",
        "--no-wait-for-ready-confirm",
        "--camera-start-stagger-sec",
        "0.5",
        "--policy-qpos-init-dataset",
        str(REPO_ROOT / "datasets/sorted_v22/bimanual-cube-new-success"),
        "--policy-qpos-init-episode",
        "0",
        "--policy-qpos-init-frame",
        "0",
        "--policy-qpos-init-source",
        "auto",
        "--policy-qpos-init-collision-check",
    ]

    assert _rm65_bimanual_dmtac_args(rtc_real_shadow=True) == expected
    assert "--send-actions" not in expected
    assert "--policy-qpos-init-dataset" in expected
    assert "--policy-qpos-init-collision-check" in expected


def test_profile_identity_is_workspace_independent_and_pinned() -> None:
    profile = load_openpi_real_deployment_profile(
        RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID
    )

    assert profile.profile_id == RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID
    assert profile.source_path == PROFILE_PATH
    assert profile.semantic_content_hash == (
        "41d159c525a2d7d0eac7f100e5ff6f20d9592bc00c6463bdfb89971b87b55084"
    )
    assert profile.runtime_args(project_root=Path("/tmp/other-checkout"))[11] == (
        "/tmp/other-checkout/hardware/rm65/utils/configs/calibrations/"
        "rm65_left_DMTAC_M.json"
    )


def test_profile_rejects_unknown_flag_surface(tmp_path: Path) -> None:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["send_actions"] = True
    candidate = tmp_path / "invalid.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OpenPIRealDeploymentProfileError, match="unknown fields"):
        load_openpi_real_deployment_profile_file(candidate)


def test_profile_rejects_workspace_escape(tmp_path: Path) -> None:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    payload["paths"]["camera_config"] = "../other/config.json"
    candidate = tmp_path / "invalid-path.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        OpenPIRealDeploymentProfileError,
        match="repository-relative path",
    ):
        load_openpi_real_deployment_profile_file(candidate)
