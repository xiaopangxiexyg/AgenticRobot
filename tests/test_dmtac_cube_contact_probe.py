"""CAD contact diagnostics must not confuse closure or ejection with a grasp."""
import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from project_paths import REPO_ROOT
from scripts.probes.diagnose_dmtac_cube_contact import (
    CAD_URDF,
    aperture_report,
    build_fixture,
    classify_trial,
    run_case,
)
from sim.mujoco.actuation.gripper_profile import (
    GripperContactTuning,
    cad_dmtac_cube_contact_gripper_profile,
)
from sim.mujoco.modeling.gripper_actuators import add_gripper_position_actuators


@pytest.mark.parametrize("options", (
    {"geom_pairs": ("ab",)},
    {"geom_pairs": (("a", "a"),)},
    {"geom_pairs": (("a", "b"), ("b", "a"))},
    {"solver_impratio": float("nan")},
    {"solver_cone": "unknown"},
    {"mimic_solref": (0., 1.)},
    {"mimic_solimp": (.99, .9, .001, .5, 2.)},
))
def test_contact_tuning_rejects_invalid_explicit_constraint_settings(options):
    with pytest.raises(ValueError):
        GripperContactTuning(**options)


@pytest.mark.parametrize("visual_only", (False, True))
def test_explicit_contact_profile_does_not_ignore_missing_or_visual_geoms(visual_only):
    profile = cad_dmtac_cube_contact_gripper_profile()
    root = ET.Element("mujoco")
    body = ET.SubElement(root, "worldbody")
    for names in profile.joint_names_by_gripper.values():
        for name in names:
            ET.SubElement(body, "joint", name=name)
    if visual_only:
        for name in {name for pair in profile.contact_tuning.geom_pairs for name in pair}:
            ET.SubElement(body, "geom", name=name, contype="0", conaffinity="0")
    with pytest.raises(ValueError if visual_only else KeyError):
        add_gripper_position_actuators(root, profile)


@pytest.mark.parametrize("side", ("left", "right"))
def test_seventy_mm_cube_aperture_uses_reviewed_sensor_boxes(side):
    result = aperture_report((REPO_ROOT / CAD_URDF).resolve(strict=True), side, .070)
    assert .072 < result["open_gap_m"] < .074
    assert .002 < result["nominal_total_margin_m"] < .004
    assert result["closed_gap_m"] < .004
    assert np.all(np.diff([s["support_gap_m"] for s in result["samples"]]) < 0)


@pytest.mark.parametrize("edge", (0., -.07, float("nan"), float("inf")))
def test_invalid_cube_edge_rejected_before_asset_loading(edge):
    with pytest.raises(ValueError, match="edge"):
        aperture_report(REPO_ROOT / "missing.urdf", "left", edge)


@pytest.mark.parametrize("defect,expected", (
    ({}, "lifted_in_contact_fixture"),
    ({"initial_overlap_m": .0006}, "invalid_initial_placement"),
    ({"maximum_penetration_m": .020}, "invalid_contact_dynamics"),
    ({"maximum_mimic_error_rad": .2}, "invalid_contact_dynamics"),
    ({"maximum_limit_excess_rad": .1}, "invalid_contact_dynamics"),
    ({"final_normal_n": 0.}, "grasp_not_retained"),
    ({"rise_m": .002}, "grasp_not_retained"),
    ({"relative_slip_m": .01}, "grasp_not_retained"),
    ({"relative_rotation_deg": 10.}, "grasp_not_retained"),
    ({"flange_tracking_error_m": .01}, "invalid_contact_dynamics"),
    ({"hand_ground_penetration_m": .01}, "invalid_contact_dynamics"),
))
def test_trial_acceptance_checks_model_validity_before_lift(defect, expected):
    metrics = {"initial_overlap_m": 0., "maximum_penetration_m": .0005,
        "maximum_mimic_error_rad": .001, "maximum_limit_excess_rad": .001,
        "rise_m": .03, "final_normal_n": 1., "relative_slip_m": .001,
        "relative_rotation_deg": .1, "flange_tracking_error_m": .0001,
        "hand_ground_penetration_m": 0.}
    assert classify_trial(**(metrics | defect)) == expected


@pytest.mark.parametrize("side", ("left", "right"))
def test_fixture_keeps_actual_hand_and_free_cube_not_an_object_weld(side):
    xml, report = build_fixture(urdf=(REPO_ROOT / CAD_URDF).resolve(strict=True),
        side=side, orientation="horizontal", edge_m=.070, mass_kg=.08)
    root = ET.fromstring(xml)
    assert root.find(".//freejoint[@name='debug_cube_freejoint']") is not None
    assert root.find(".//body[@name='debug_cube']").get("mocap") is None
    assert len(root.findall("actuator/position")) == 6
    assert len(root.findall("equality/joint")) == 5
    welds = root.findall("equality/weld")
    assert len(welds) == 1
    assert {welds[0].get("body1"), welds[0].get("body2")} == {
        "fixture_driver", side + "_arm_link6"}
    assert root.find(f".//body[@name='{side}_arm_link6']/freejoint") is not None
    assert report["arm_path_validated"] is False
    assert report["cube_contact"]["friction"] == [1., .005, .0001]
    assert report["mass_source"] == "simulation_assumption_not_measured"
    for finger in ("l", "r"):
        assert root.find(f".//geom[@name='{side}_gripper_{finger}_dm_tac_w2m_sensor_collision_0']") is not None


@pytest.mark.integration
def test_native_executor_contact_trial_persists_measurements_and_rejects_bad_dynamics(tmp_path):
    pytest.importorskip("mujoco")
    result = run_case(tmp_path / "trial", side="right", orientation="top_down")
    rows = [json.loads(line) for line in (tmp_path / "trial/trace.jsonl").read_text().splitlines()]
    assert len(rows) == 600
    assert {r["phase"] for r in rows} == {"open", "close", "lift", "hold"}
    assert rows[-1]["flange_lift_m"] == pytest.approx(.03)
    assert all(len(r["joint_qpos_rad"]) == 6 for r in rows)
    assert result["execution"]["execution_mode"] == "dmtac_fitted_position_servo_v2"
    assert result["object_weld_or_latch"] is False
    assert result["maximum_cube_hand_penetration_m"] > .002
    assert result["diagnostic_status"] == "invalid_contact_dynamics"
    assert result["trial_lift_passed"] is False
    assert json.loads((tmp_path / "trial/result.json").read_text())["trial_lift_passed"] is False


@pytest.mark.integration
@pytest.mark.parametrize("side,orientation,friction", [
    # Preserve every two-factor interaction with four fresh physical trials.
    pytest.param("left", "horizontal", 1., id="left-horizontal-1.0"),
    pytest.param("left", "top_down", 1.2, id="left-top-down-1.2"),
    pytest.param("right", "horizontal", 1.2, id="right-horizontal-1.2"),
    pytest.param("right", "top_down", 1., id="right-top-down-1.0"),
])
def test_constraint_candidate_retains_seventy_mm_cube_without_force_increase(tmp_path, side, orientation, friction):
    pytest.importorskip("mujoco")
    baseline_xml, baseline = build_fixture(urdf=(REPO_ROOT / CAD_URDF).resolve(strict=True),
        side=side, orientation=orientation, edge_m=.070, mass_kg=.08)
    result = run_case(tmp_path / "candidate", side=side, orientation=orientation,
        object_approach_offset_m=.01, contact_model="constraint_candidate", sliding_friction=friction)
    assert result["trial_lift_passed"] is True
    assert result["maximum_mimic_error_rad"] < .01
    assert result["maximum_cube_hand_penetration_m"] < .002
    assert result["maximum_hand_ground_penetration_m"] < .002
    assert result["relative_slip_m"] < .003 and result["relative_rotation_deg"] < 2.
    assert .025 <= result["cube_rise_m"] <= .035
    changed = {"name", "contact_tuning", "calibration_status", "evidence"}
    assert {k: v for k, v in result["profile"].items() if k not in changed} == {
        k: v for k, v in baseline["profile"].items() if k not in changed}
    assert result["cube_contact"] == baseline["cube_contact"]
    assert all(v == [1., .005, .0001] for v in result["actual_sensor_friction"].values())
    assert result["explicit_pair_friction"] == [[friction, friction, .005, .0001, .0001]] * 2
    assert not ET.fromstring(baseline_xml).findall("contact/pair")
    rows = [json.loads(line) for line in (tmp_path / "candidate/trace.jsonl").read_text().splitlines()]
    assert max(abs(row["actual_flange_qvel"][2]) for row in rows if row["phase"] == "lift") > .01
