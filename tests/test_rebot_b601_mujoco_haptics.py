from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import B601_RS_URDF_RELATIVE_PATH, B601PinocchioModel
from sim.mujoco.rebot_b601_rs import haptic_demo
from sim.mujoco.rebot_b601_rs.haptic_demo import POSE_FIXTURES, run_haptic_fixture_demo
from sim.mujoco.rebot_b601_rs.mit_driver import B601MujocoEffortDriver

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


@pytest.mark.parametrize("fixture_name", POSE_FIXTURES)
def test_pose_fixture_restores_bounded_axis_or_6d_disturbance(
    fixture_name: str,
) -> None:
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name=fixture_name,
        duration_sec=1.5,
        viewer=False,
    )

    effect = payload["effect_acceptance"]
    simulation = payload["simulation_acceptance"]
    fixture = payload["fixture_contract"]
    assert code == 0
    assert effect["passed"] is True
    assert effect["translation_passed"] is True
    assert effect["rotation_passed"] is True
    assert simulation["passed"] is True
    assert simulation["driver_faulted"] is False
    assert simulation["torque_limited_physics_ticks"] == 0
    assert simulation["peak_abs_qvel_rad_s"] < 2.0
    assert simulation["peak_abs_qacc_rad_s2"] < 250.0
    assert fixture["anchor_source"] == "current_fk_before_disturbance"
    assert fixture["disturbance"]["rotation_tool_local_rad"] is not None


def test_pose_6d_acceptance_reports_translation_and_rotation_separately() -> None:
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name="pose-6d",
        duration_sec=1.5,
        viewer=False,
    )

    effect = payload["effect_acceptance"]
    assert code == 0
    assert effect["translation_required"] is True
    assert effect["rotation_required"] is True
    assert effect["initial_metric"] > effect["final_metric"]
    assert effect["initial_rotation_metric"] > effect["final_rotation_metric"]


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


def test_contact_box_trace_has_native_reaction_and_one_overlay_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = []
    original_step = B601MujocoEffortDriver.step_physics

    def record_step(self, reference=None):
        if reference is not None:
            references.append(reference)
        return original_step(self, reference)

    monkeypatch.setattr(B601MujocoEffortDriver, "step_physics", record_step)
    trace = tmp_path / "contact.jsonl"
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT, fixture_name="contact-box",
        duration_sec=1.5, trace_jsonl=trace,
    )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    diagnostics = payload["contact_diagnostics"]
    simulation = payload["simulation_acceptance"]
    assert code == 0
    assert diagnostics["nonzero_wrench_samples"] > 0
    assert diagnostics["sample_count"] == len(records) == len(references) == 750
    assert diagnostics["max_power_identity_error_w"] < 1e-9
    assert diagnostics["projection_role"] == "diagnostic_only_never_dispatched"
    assert not diagnostics["force_calibrated"]
    assert not diagnostics["passivity_claim"]
    assert simulation["driver_faulted"] is False
    assert simulation["torque_limited_physics_ticks"] == 0
    assert simulation["overlay_limited_physics_ticks"] == 0
    assert simulation["peak_abs_qvel_rad_s"] < 2.0
    assert simulation["peak_abs_qacc_rad_s2"] < 250.0
    assert sum(record["force_n"][2] for record in records) < 0.0
    model = B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)
    for index, (record, reference) in enumerate(zip(records, references, strict=True)):
        assert record["sequence"] == reference.sequence == index
        assert record["physics_time_sec"] == pytest.approx(index * 0.002, abs=1e-12)
        assert record["driver_time_sec"] == reference.source_timestamp_sec
        assert record["interval_end_time_sec"] == pytest.approx(index * 0.002 + 0.001)
        assert record["command_applied"] is True
        assert record["plant_advanced"] is True
        # Contact is already in qfrc_constraint. Never dispatch its projection.
        np.testing.assert_array_equal(
            reference.joint_torque_overlay_nm, record["spring_overlay_requested_nm"]
        )
        jacobian = model.arm_jacobian(record["qpos_rad"])
        wrench = np.asarray((*record["force_n"], *record["torque_nm"]))
        np.testing.assert_allclose(
            record["projected_contact_torque_diagnostic_nm"], jacobian.T @ wrench
        )
        np.testing.assert_allclose(
            record["end_effector_pose_world"], model.forward_kinematics(record["qpos_rad"])
        )
        assert record["contact_cartesian_power_w"] == pytest.approx(
            record["contact_joint_power_w"], abs=1e-9
        )


def test_contact_box_acceptance_rejects_a_box_outside_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_spec = haptic_demo.b601_contact_scene_spec

    def move_objects_away(*, objects=()):
        return scene_spec(objects=tuple(
            replace(obj, initial_position_m=(2.0, 2.0, 2.0)) for obj in objects
        ))

    monkeypatch.setattr(haptic_demo, "b601_contact_scene_spec", move_objects_away)
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT, fixture_name="contact-box", duration_sec=0.2,
    )
    assert code == 3
    assert payload["contact_diagnostics"]["nonzero_wrench_samples"] == 0
    assert payload["effect_acceptance"]["passed"] is False
    assert payload["simulation_acceptance"]["run_completed"] is True
    assert payload["simulation_acceptance"]["driver_faulted"] is False


def test_contact_trace_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    trace = tmp_path / "existing.jsonl"
    trace.write_text("preserved evidence\n")
    with pytest.raises(FileExistsError):
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT, fixture_name="contact-box", trace_jsonl=trace,
        )
    assert trace.read_text() == "preserved evidence\n"
    with pytest.raises(ValueError, match="requires contact-box"):
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT, fixture_name="spring", trace_jsonl=trace,
        )


def test_contact_box_replays_deterministically() -> None:
    runs = [run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT, fixture_name="contact-box", duration_sec=0.2,
    ) for _ in range(2)]
    assert runs[0][0] == runs[1][0] == 0
    np.testing.assert_array_equal(
        runs[0][1]["simulation_acceptance"]["final_qpos_rad"],
        runs[1][1]["simulation_acceptance"]["final_qpos_rad"],
    )
    assert runs[0][1]["contact_diagnostics"] == runs[1][1]["contact_diagnostics"]
