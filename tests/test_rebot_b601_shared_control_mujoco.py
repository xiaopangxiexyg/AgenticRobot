from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_motion.rebot_b601_rs import B601_RS_URDF_RELATIVE_PATH
from sim.mujoco.rebot_b601_rs import haptic_demo
from sim.mujoco.rebot_b601_rs.haptic_demo import (
    CARTESIAN_ASSIST_FIXTURE,
    run_haptic_fixture_demo,
)
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


def test_cartesian_assist_fixture_moves_tcp_in_world_axis_with_bounded_drift(
) -> None:
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name=CARTESIAN_ASSIST_FIXTURE,
        duration_sec=8.0,
    )

    effect = payload["effect_acceptance"]
    simulation = payload["simulation_acceptance"]
    assert code == 0
    assert effect["passed"] is True
    assert effect["axis"] == "x"
    assert effect["assist_withdrawn"] is True
    assert effect["maximum_signed_displacement_m"] >= 0.001
    assert effect["maximum_lateral_drift_m"] <= effect[
        "maximum_lateral_drift_limit_m"
    ]
    assert effect["maximum_rotation_drift_rad"] <= effect[
        "maximum_rotation_drift_limit_rad"
    ]
    assert simulation["peak_force_norm_n"] <= effect["force_limit_n"]
    assert simulation["peak_requested_overlay_torque_nm"] <= effect[
        "joint_overlay_limit_nm"
    ]
    assert simulation["driver_faulted"] is False


def test_cartesian_assist_viewer_accepts_terminal_a_without_window_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeViewer:
        def __init__(self, _scene, **_kwargs) -> None:
            self.closed = False

        def start(self) -> None:
            return None

        def sync(self) -> bool:
            return True

        def close(self) -> None:
            self.closed = True

    class FakeTerminal:
        def __init__(self, **_kwargs) -> None:
            self.keys = iter(("a", None))
            self.closed = False

        def __call__(self, _timeout_sec: float) -> str | None:
            return next(self.keys, None)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(haptic_demo, "MujocoPassiveViewerSession", FakeViewer)
    monkeypatch.setattr(haptic_demo, "ImmediateTerminalCommands", FakeTerminal)
    monkeypatch.setattr(haptic_demo.time, "sleep", lambda _duration: None)

    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name=CARTESIAN_ASSIST_FIXTURE,
        duration_sec=0.2,
        viewer=True,
    )

    assert code == 3
    assert payload["simulation_acceptance"]["run_completed"] is True
    assert payload["effect_acceptance"]["activation_source"] == "terminal_key_a"
    assert payload["effect_acceptance"]["operator_events"][0]["event"] == "activated"


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


def test_cartesian_assist_cli_builds_explicit_adjustable_motion_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return 0, {"simulation_acceptance": {"passed": True}}

    monkeypatch.setattr(haptic_demo, "run_haptic_fixture_demo", fake_run)

    code = haptic_demo.main(
        [
            "--rebot-source-root",
            str(REBOT_SOURCE_ROOT),
            "--fixture",
            CARTESIAN_ASSIST_FIXTURE,
            "--cartesian-assist-axis",
            "z",
            "--cartesian-assist-offset-m",
            "-0.03",
            "--cartesian-assist-stiffness-n-per-m",
            "60",
            "--cartesian-assist-max-force-n",
            "1.0",
            "--cartesian-assist-max-joint-torque-nm",
            "0.3",
            "--cartesian-assist-lateral-stiffness-n-per-m",
            "120",
            "--cartesian-assist-lateral-damping-n-s-per-m",
            "4",
        ]
    )

    assert code == 0
    config = captured["cartesian_assist_config"]
    assert config.axis == "z"
    assert config.offset_m == -0.03
    assert config.stiffness_n_per_m == 60.0
    assert config.max_abs_force_n == 1.0
    assert config.max_abs_joint_torque_nm == 0.3
    assert config.lateral_stiffness_n_per_m == 120.0
    assert config.lateral_damping_n_s_per_m == 4.0


def test_haptic_cli_turns_keyboard_interrupt_into_concise_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        haptic_demo,
        "main",
        lambda _argv=None: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert haptic_demo.cli([]) == 130
    output = capsys.readouterr().out
    assert "interrupted" in output
    assert "shell remains active" in output
