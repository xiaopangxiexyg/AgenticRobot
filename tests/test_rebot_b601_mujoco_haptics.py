from __future__ import annotations

from pathlib import Path

import pytest

from robot_motion.rebot_b601_rs import B601_RS_URDF_RELATIVE_PATH
from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.optional, pytest.mark.contract]


@pytest.fixture(scope="module", autouse=True)
def _offline_dependencies() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")


@pytest.mark.parametrize("fixture_name", ["spring", "wall"])
def test_headless_fixture_has_bounded_closed_loop_effect(fixture_name: str) -> None:
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name=fixture_name,
        duration_sec=1.5,
        viewer=False,
    )

    effect = payload["effect_acceptance"]
    simulation = payload["simulation_acceptance"]
    driver = payload["driver"]
    assert code == 0
    assert effect["passed"] is True
    assert simulation["passed"] is True
    assert simulation["run_completed"] is True
    assert simulation["command_count"] == simulation["expected_command_count"]
    assert simulation["driver_faulted"] is False
    assert simulation["fault_reason"] is None
    assert simulation["state_valid"] is True
    assert simulation["state_within_limits"] is True
    assert simulation["torque_limited_physics_ticks"] == 0
    assert simulation["overlay_limited_physics_ticks"] == 0
    assert simulation["peak_requested_overlay_torque_nm"] < 6.0
    assert driver["network_transport"] == "out_of_scope"
    assert driver["hardware_safety_claim"] is False
    assert "does not validate a motor" in payload["fidelity"]


@pytest.mark.parametrize("fixture_name", ["spring", "wall"])
def test_effect_acceptance_rejects_a_run_too_short_to_demonstrate_effect(
    fixture_name: str,
) -> None:
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name=fixture_name,
        duration_sec=0.02,
        viewer=False,
    )

    assert code == 3
    assert payload["effect_acceptance"]["passed"] is False
    assert payload["simulation_acceptance"]["run_completed"] is True
    assert payload["simulation_acceptance"]["driver_faulted"] is False


def test_fixture_demo_rejects_invalid_mode_or_duration_before_simulation() -> None:
    with pytest.raises(ValueError, match="fixture_name"):
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT,
            fixture_name="contact",
        )
    with pytest.raises(ValueError, match="duration_sec"):
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT,
            fixture_name="spring",
            duration_sec=0.0,
        )
    with pytest.raises(TypeError, match="viewer"):
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT,
            fixture_name="spring",
            viewer="false",  # type: ignore[arg-type]
        )
