from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import B601_RS_URDF_RELATIVE_PATH, B601PinocchioModel
from sim.mujoco.rebot_b601_rs import contact
from sim.mujoco.rebot_b601_rs.haptic_demo import (
    DEFAULT_HAPTIC_INITIAL_QPOS,
    POSE_FIXTURES,
    run_haptic_fixture_demo,
)
from sim.mujoco.rebot_b601_rs.haptic_demo import (
    main as haptic_main,
)
from sim.mujoco.rebot_b601_rs.mit_driver import B601MujocoEffortDriver
from sim.mujoco.rebot_b601_rs.reflection_demo import run_force_reflection_demo

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

    if fixture_name == "pose-6d":
        # Inspect the existing full rollout, not a second identical one.
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
    scene_spec = contact.b601_contact_scene_spec

    def move_objects_away(*, objects=()):
        return scene_spec(
            objects=tuple(
                replace(obj, initial_position_m=(2.0, 2.0, 2.0)) for obj in objects
            )
        )

    monkeypatch.setattr(contact, "b601_contact_scene_spec", move_objects_away)
    code, payload = run_haptic_fixture_demo(
        source_root=REBOT_SOURCE_ROOT,
        fixture_name="contact-box",
        duration_sec=0.2,
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
            source_root=REBOT_SOURCE_ROOT,
            fixture_name="contact-box",
            trace_jsonl=trace,
        )
    assert trace.read_text() == "preserved evidence\n"
    with pytest.raises(ValueError, match="requires contact-box"):
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT,
            fixture_name="spring",
            trace_jsonl=trace,
        )


def test_contact_box_replays_deterministically() -> None:
    runs = [
        run_haptic_fixture_demo(
            source_root=REBOT_SOURCE_ROOT,
            fixture_name="contact-box",
            duration_sec=0.2,
        )
        for _ in range(2)
    ]
    assert runs[0][0] == runs[1][0] == 0
    np.testing.assert_array_equal(
        runs[0][1]["simulation_acceptance"]["final_qpos_rad"],
        runs[1][1]["simulation_acceptance"]["final_qpos_rad"],
    )
    assert runs[0][1]["contact_diagnostics"] == runs[1][1]["contact_diagnostics"]


@pytest.mark.parametrize("delay", [0.0, 0.04])
def test_force_reflection_dual_plant_trace_has_one_dispatch_owner_each(
    tmp_path, monkeypatch, delay
) -> None:
    references = {}
    original_step = B601MujocoEffortDriver.step_physics

    def record_step(self, reference=None):
        if reference is not None:
            references.setdefault(id(self), []).append(reference)
        return original_step(self, reference)

    monkeypatch.setattr(B601MujocoEffortDriver, "step_physics", record_step)
    trace = tmp_path / "reflection.jsonl"
    code, payload = run_force_reflection_demo(
        source_root=REBOT_SOURCE_ROOT,
        initial_qpos=DEFAULT_HAPTIC_INITIAL_QPOS,
        trace_jsonl=trace,
        feedback_delay_sec=delay,
    )
    assert code == 0
    assert len(references) == 2
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(records) == payload["diagnostics"]["sample_count"] == 750
    assert payload["motion_return_channel"] == "absent"
    assert payload["diagnostics"]["max_transform_power_error_w"] < 1e-9
    assert payload["diagnostics"]["max_projection_power_error_w"] < 1e-9
    assert payload["diagnostics"]["rejection_counts"] == {}
    assert (
        payload["diagnostics"]["nominal_feedback_delivered_work_j"]
        <= payload["finite_horizon_nominal_feedback_work_bound_j"]
    )
    # In this fixed virtual Ry(+90) coupling, downward follower reaction maps -X.
    assert sum(r["feedback_wrench"][0] for r in records) < 0.0
    follower_refs, master_refs = references.values()
    for row, mr, fr in zip(records, master_refs, follower_refs, strict=True):
        assert row["master_command_applied"] and row["follower_command_applied"]
        np.testing.assert_array_equal(
            fr.joint_torque_overlay_nm, row["follower_overlay_requested_nm"]
        )
        np.testing.assert_allclose(
            mr.joint_torque_overlay_nm,
            np.asarray(row["master_spring_overlay_nm"])
            + row["master_feedback_overlay_nm"],
        )
        if row["feedback_source_sequence"] is not None:
            assert row["feedback_source_sequence"] <= row["sequence"]
            assert row["feedback_source_age_sec"] >= delay - 1e-12
            assert row["feedback_receive_timestamp_sec"] <= row["physics_time_sec"]
            assert row["physics_time_sec"] - row["feedback_source_timestamp_sec"] == row["feedback_source_age_sec"]
            if row["delivered_source_timestamp_sec"] is not None:
                assert row["delivered_receive_timestamp_sec"] == row["physics_time_sec"]
                assert row["delivered_source_timestamp_sec"] == row["feedback_source_timestamp_sec"]
                assert row["delivered_receive_timestamp_sec"] == row["feedback_receive_timestamp_sec"]
    for plant in payload["plants"].values():
        assert plant["advanced_ticks"] == 1500
        assert plant["fault_reason"] is None
        assert plant["torque_limited_ticks"] == plant["overlay_limited_ticks"] == 0


def test_zero_gain_reflection_changes_only_master_response() -> None:
    results = [
        run_force_reflection_demo(
            source_root=REBOT_SOURCE_ROOT,
            initial_qpos=DEFAULT_HAPTIC_INITIAL_QPOS,
            duration_sec=0.4,
            feedback_gain=gain,
        )
        for gain in (0.0, 1.0)
    ]
    assert (
        results[0][0] == 3
    )  # An absent feedback path cannot pass reflection acceptance.
    assert results[0][1]["diagnostics"]["nonzero_master_samples"] == 0
    assert results[1][0] == 0
    np.testing.assert_array_equal(
        results[0][1]["plants"]["follower"]["final_qpos_rad"],
        results[1][1]["plants"]["follower"]["final_qpos_rad"],
    )
    assert not np.allclose(
        results[0][1]["plants"]["master"]["final_qpos_rad"],
        results[1][1]["plants"]["master"]["final_qpos_rad"],
        atol=1e-5,
        rtol=0.0,
    )


def test_expired_delay_never_generates_master_feedback(tmp_path) -> None:
    trace = tmp_path / "expired.jsonl"
    code, payload = run_force_reflection_demo(
        source_root=REBOT_SOURCE_ROOT,
        initial_qpos=DEFAULT_HAPTIC_INITIAL_QPOS,
        duration_sec=0.4,
        feedback_delay_sec=0.12,
        trace_jsonl=trace,
    )
    assert code == 3
    assert payload["diagnostics"]["rejection_counts"]["expired_source"] > 0
    assert payload["diagnostics"]["nonzero_master_samples"] == 0
    for line in trace.read_text().splitlines():
        np.testing.assert_array_equal(
            json.loads(line)["master_feedback_overlay_nm"], np.zeros(6)
        )


def test_reflection_cli_uses_existing_entrypoint_and_rejects_ignored_options(capsys) -> None:
    args = ["--rebot-source-root", str(REBOT_SOURCE_ROOT), "--fixture", "force-reflection",
            "--duration-sec", "0.2", "--feedback-delay-sec", "0.01"]
    assert haptic_main(args) == 0
    assert json.loads(capsys.readouterr().out)["fixture"] == "force-reflection"
    for extra in (["--real2sim-controller"], ["--fixture-stiffness-scale", "2"]):
        with pytest.raises(SystemExit) as error:
            haptic_main(args + extra)
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        haptic_main(["--rebot-source-root", str(REBOT_SOURCE_ROOT), "--feedback-delay-sec", "0.01"])
    assert error.value.code == 2
