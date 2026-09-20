from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_motion.rebot_b601_rs import B601_RS_URDF_RELATIVE_PATH
from sim.mujoco.rebot_b601_rs.haptic_demo import main as haptic_main
from sim.mujoco.rebot_b601_rs.shared_control_demo import (
    SHARED_CONTROL_FIXTURE,
    run_b601_shared_control_demo,
)

pytestmark = [pytest.mark.contract, pytest.mark.optional, pytest.mark.slow_sim]
REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"


@pytest.fixture(scope="module", autouse=True)
def _offline_dependencies() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")


def test_shared_control_fixture_runs_existing_effort_plant(
) -> None:
    code, payload = run_b601_shared_control_demo(
        source_root=REBOT_SOURCE_ROOT,
        duration_sec=1.6,
        host_hz=100.0,
    )

    assert code == 0
    assert payload["fixture"] == SHARED_CONTROL_FIXTURE
    acceptance = payload["simulation_acceptance"]
    assert acceptance["passed"] is True
    assert acceptance["hardware_claim"] is False
    assert acceptance["driver_fault_reason"] is None
    assert acceptance["required_modes_observed"] is True
    assert acceptance["final_mode"] == "passive"
    assert acceptance["maximum_override_assist_torque_nm"] == 0.0
    assert acceptance["maximum_await_rebase_assist_torque_nm"] == 0.0
    assert acceptance["minimum_joint6_assist_torque_nm"] == 0.0
    assert acceptance["joint6_assist_direction_reversed"] is False
    assert abs(acceptance["joint6_displacement_rad"]) > 1e-4
    assert acceptance["maximum_noncommanded_displacement_rad"] <= acceptance[
        "maximum_noncommanded_displacement_limit_rad"
    ]


def test_existing_haptic_entrypoint_exposes_policy_assist_without_changing_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = haptic_main(
        [
            "--rebot-source-root",
            str(REBOT_SOURCE_ROOT),
            "--fixture",
            SHARED_CONTROL_FIXTURE,
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fixture"] == SHARED_CONTROL_FIXTURE
    assert payload["simulation_acceptance"]["passed"] is True


def test_haptic_entrypoint_can_save_full_result_and_print_short_summary(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    code = haptic_main(
        [
            "--rebot-source-root",
            str(REBOT_SOURCE_ROOT),
            "--fixture",
            SHARED_CONTROL_FIXTURE,
            "--output-root",
            str(tmp_path),
        ]
    )

    assert code == 0
    terminal = capsys.readouterr().out
    assert "status=PASS exit_code=0" in terminal
    assert '"driver"' not in terminal
    result_paths = list(tmp_path.glob("b601-policy-assist-*/result.json"))
    assert len(result_paths) == 1
    payload = json.loads(result_paths[0].read_text(encoding="utf-8"))
    assert payload["fixture"] == SHARED_CONTROL_FIXTURE
    assert payload["simulation_acceptance"]["passed"] is True
