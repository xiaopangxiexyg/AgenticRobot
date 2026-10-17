"""Offline envelope proof and scene-alignment regression tests."""
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from scripts.probes.compare_rm65_dmtac_sweep import (
    CircularBoxSweep, _SIGNS, conservative_sweep_hull, load_sensor_sweeps,
    compare_cad_in_same_scene, validate_mesh_boxes, validate_sweep_fk,
)


def test_sagitta_covers_continuous_arc_not_only_endpoints():
    sweep = CircularBoxSweep("test", "opening", (0., .864), _SIGNS * .001,
        np.array((0., 0., 1.)), np.array((.042, 0., 0.)))
    naive = ConvexHull(np.concatenate([sweep.corners(0.), sweep.corners(.864)]))
    midpoint = sweep.corners(.432)
    assert np.max(midpoint @ naive.equations[:, :3].T + naive.equations[:, 3]) > .002
    hull, proof = conservative_sweep_hull((sweep,), segments=1, padding_m=0.)
    assert proof["sensors"][0]["arc_chord_bound_m"] == pytest.approx(.042*(1-np.cos(.432)))
    for q in np.linspace(0., .864, 501):
        assert np.max(sweep.corners(q) @ hull.equations[:, :3].T + hull.equations[:, 3]) < 1e-12


@pytest.mark.parametrize("segments", (0, 257, True, 1.5))
def test_invalid_arc_subdivision_is_rejected(segments):
    with pytest.raises(ValueError, match="segments"):
        conservative_sweep_hull((), segments=segments)


def _cad_path():
    path = Path(__file__).resolve().parents[1] / "assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"
    if not path.is_file():
        pytest.skip("Reviewed local DMTAC CAD asset is unavailable")
    return path


@pytest.mark.parametrize("side", ("left", "right"))
def test_reviewed_cad_sweep_has_native_fk_and_mesh_coverage(side):
    pin = pytest.importorskip("pinocchio")
    pytest.importorskip("trimesh")
    path = _cad_path()
    sweeps = load_sensor_sweeps(path, side)
    hull, proof = conservative_sweep_hull(sweeps)
    native = validate_sweep_fk(path, sweeps, hull, pin)
    assert native["maximum_corner_delta_m"] < 1e-8
    assert native["maximum_hull_violation_m"] < 0
    assert proof["sensors"][0]["arc_chord_bound_m"] < .000016
    assert len(validate_mesh_boxes(path, padding_m=.0001)) == 4


@pytest.mark.parametrize("defect", ("axis", "rotation", "mimic", "limit"))
def test_changed_linkage_cannot_silently_reuse_the_arc_proof(tmp_path, defect):
    tree = ET.parse(_cad_path())
    joint = tree.getroot().find("joint[@name='left_eg2_4b2_cad_l_3_joint']")
    if defect == "axis":
        joint.find("axis").set("xyz", "0 0 -1")
    elif defect == "rotation":
        joint.find("origin").set("rpy", "0 .01 0")
    elif defect == "mimic":
        joint.find("mimic").set("multiplier", ".99")
    else:
        joint.find("limit").set("upper", ".865")
    path = tmp_path / "changed.urdf"
    tree.write(path)
    with pytest.raises(ValueError):
        load_sensor_sweeps(path, "left")


def test_comparison_detects_table_difference_and_aligns_it_without_touching_source():
    pin = pytest.importorskip("pinocchio")
    cad = _cad_path()
    proxy = cad.parents[1] / "rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    if not proxy.is_file():
        pytest.skip("Reviewed local DMTAC proxy asset is unavailable")
    model, baseline, _ = pin.buildModelsFromUrdf(str(proxy.resolve()), package_dirs=[str(proxy.resolve().parent)])
    original = proxy.read_bytes(), cad.read_bytes()
    neutral = {s: np.zeros(6) for s in ("left", "right")}
    alignment, rows = compare_cad_in_same_scene(cad, model, baseline, [("neutral", neutral)], pin)
    table = alignment["table_0"]
    assert table["center_difference_m"] == pytest.approx(.02)
    assert rows[0]["opening_samples"] == 101
    assert original == (proxy.read_bytes(), cad.read_bytes())
    baseline_table = next(g for g in baseline.geometryObjects if g.name == "table_0")
    assert baseline_table.placement.translation[2] == pytest.approx(.005)


def test_cross_hand_comparison_uses_complete_cad_and_explicit_openings():
    pin = pytest.importorskip("pinocchio")
    cad = _cad_path()
    proxy = cad.parents[1] / "rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    if not proxy.is_file():
        pytest.skip("Reviewed local DMTAC proxy asset is unavailable")
    model, baseline, _ = pin.buildModelsFromUrdf(str(proxy.resolve()), package_dirs=[str(proxy.resolve().parent)])
    neutral = {s: np.zeros(6) for s in ("left", "right")}
    for openings in ({"left": 0., "right": .864}, {"left": float("nan"), "right": 0.}, {"left": 0.}):
        if set(openings) != {"left", "right"} or not np.isfinite(list(openings.values())).all():
            with pytest.raises(ValueError, match="openings"):
                compare_cad_in_same_scene(cad, model, baseline, [("neutral", neutral)], pin,
                    hand_opening_by_side=openings)
            continue
        _, rows = compare_cad_in_same_scene(cad, model, baseline, [("neutral", neutral)], pin,
            hand_opening_by_side=openings)
        result = rows[0]["cross_hand_cad"]
        assert result["opening_rad_by_side"] == openings
        assert result["geometry_count_by_side"] == {"left": 20, "right": 20}
        assert result["checked_pair_count"] == 400
        assert result["colliding_pairs"] == []
        assert result["closest_pair"]["distance_m"] > 0
        assert result["mesh_signed_penetration_depth_certified"] is False


@pytest.mark.parametrize("left_qpos, expected_pair", (
    ((.9594413458104385, 1.1246471152253592, .10462767121244215,
      -.009241480762586511, 1.9119465425291864, 1.5279681845891242),
     ("left_realman_to_inspire_gripper_flange_0", "right_eg2_4b2_cad_base_link_0")),
    ((1.002960245225118, .3948900283231007, 1.5536680391441728,
      -.07487970551681633, .8496690002822734, 1.7574366174408937),
     ("left_eg2_4b2_cad_base_link_0", "right_gripper_l_dm_tac_w2m_sensor_0")),
))
def test_cad_reproduces_frozen_left_endpoint_collision_beyond_sensor_boxes(left_qpos, expected_pair):
    pin = pytest.importorskip("pinocchio")
    cad = _cad_path()
    proxy = cad.parents[1] / "rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    if not proxy.is_file():
        pytest.skip("Reviewed local DMTAC proxy asset is unavailable")
    model, baseline, _ = pin.buildModelsFromUrdf(str(proxy.resolve()), package_dirs=[str(proxy.resolve().parent)])
    # Exact frozen simulation endpoints; the inactive arm and fully-open state
    # remain identical to the original failed local-chain assessment.
    pair = {"left": np.asarray(left_qpos), "right": np.array((
        -1.4745937603154737, .6181083705727584, 1.6288110274553462,
        -1.6168730083949026, -1.2114854288405503, .6198711168982818))}
    _, rows = compare_cad_in_same_scene(cad, model, baseline, [("frozen_left", pair)], pin,
        hand_opening_by_side={"left": 0., "right": 0.})
    collisions = rows[0]["cross_hand_cad"]["colliding_pairs"]
    assert any(tuple(item["names"]) == expected_pair for item in collisions)
    assert any("dm_tac" not in item["names"][0] for item in collisions)
