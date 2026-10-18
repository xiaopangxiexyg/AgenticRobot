"""Continuous sensor coverage and explicit two-piece proxy scene binding."""
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from scripts.probes.build_rm65_dmtac_stepped_proxy import (
    build_candidate, clipped_vertices, stepped_boxes, lower_box_extended_upward,
)
from robot_motion.rm65.scene_profiles import build_dual_arm_scene_profile


def test_clipping_adds_intersections_not_only_original_vertices():
    points = np.array([[x, y, z] for z, width in ((-1., .2), (1., 1.))
        for x in (-width, width) for y in (-width, width)])
    cut = clipped_vertices(ConvexHull(points), 0., lower=True)
    assert cut[:, 0].max() == pytest.approx(.6)
    assert cut[:, 2].max() == pytest.approx(0.)


def test_two_boxes_cover_both_clipped_solids_and_keep_upper_width():
    points = np.array([[x, y, z] for z, width in ((-.04, .03), (.04, .08))
        for x in (-width, width) for y in (-.015, .015)])
    hull = ConvexHull(points)
    boxes, proof = stepped_boxes(hull, np.zeros(3), np.array([.17, .04, .082]))
    assert len(boxes) == 2
    assert boxes[0]['size_m'][0] == pytest.approx(.17)
    assert boxes[1]['size_m'][0] < boxes[0]['size_m'][0]
    assert max(proof['halfspace_vertex_excess_m']) < 1e-12
    assert proof['candidate_volume_m3'] < proof['original_volume_m3']
    assert boxes[0]['center_m'][2] - boxes[0]['size_m'][2] / 2 == pytest.approx(
        boxes[1]['center_m'][2] + boxes[1]['size_m'][2] / 2)


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    pytest.importorskip("pinocchio"); pytest.importorskip("trimesh")
    assets = Path(__file__).resolve().parents[1] / "assets"
    cad = assets / "realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"
    proxy = assets / "rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    if not cad.is_file() or not proxy.is_file():
        pytest.skip("Reviewed CAD/proxy assets unavailable")
    original = cad.read_bytes(), proxy.read_bytes()
    output = tmp_path_factory.mktemp("stepped") / "candidate"
    report = build_candidate(cad, proxy, output)
    assert original == (cad.read_bytes(), proxy.read_bytes())
    return cad, proxy, output, report


def test_real_cad_continuous_and_native_fk_coverage_both_hands(candidate):
    _, proxy, output, report = candidate
    source = ET.parse(proxy).getroot()
    changed = ET.parse(output / "rm65_dual_arm_hand_proxy_dmtac_stepped.urdf").getroot()
    for side in ("left", "right"):
        row = report['sides'][side]
        assert row['native_fk']['opening_samples'] == 101
        assert row['native_fk']['maximum_delta_m'] < 1e-8
        assert row['native_fk']['maximum_mesh_union_excess_m'] <= 0
        assert max(row['split']['halfspace_vertex_excess_m']) < 1e-9
        assert row['boxes'][1]['size_m'][0] < row['boxes'][0]['size_m'][0]
        a = source.find(f"link[@name='{side}_hand_proxy']/collision")
        b = changed.find(f"link[@name='{side}_hand_proxy']/collision")
        assert ET.tostring(a) == ET.tostring(b)


def test_candidate_native_scene_binds_all_31_pairs_and_baseline_is_unchanged(candidate):
    import pinocchio as pin
    _, _, _, report = candidate
    baseline = build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_strict")
    profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_strict",
        dmtac_proxy_urdf_path=Path(report['candidate_urdf']))
    assert len(baseline.collision_pairs) == 20
    assert len(profile.collision_pairs) == 31
    names = {b.geometry_name for b in profile.hand_proxy_boxes}
    assert {f"{s}_hand_proxy_{i}" for s in ("left", "right") for i in range(3)} <= names
    model, collision, _ = pin.buildModelsFromUrdf(str(profile.urdf_path),
        package_dirs=[str(p) for p in profile.package_dirs])
    geometry_names = {str(g.name) for g in collision.geometryObjects}
    assert all(a in geometry_names and b in geometry_names for a, b in profile.collision_pairs)
    assert model.nq == 12
    assert baseline.urdf_path.name == "rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    for side in ("left", "right"):
        for other in ("table_0", "camera_pole_0"):
            assert (f"{side}_hand_proxy_2", other) in profile.collision_pairs
    with pytest.raises(ValueError, match="DMTAC"):
        build_dual_arm_scene_profile("dual_arm", dmtac_proxy_urdf_path=profile.urdf_path)


def test_other_terminal_cad_parts_do_not_lose_previous_coverage(candidate):
    import pinocchio as pin
    cad, proxy, _, report = candidate
    model, _, visual = pin.buildModelsFromUrdf(str(cad.resolve()),
        package_dirs=[str(cad.resolve().parent)])
    data, vd, q = model.createData(), visual.createData(), pin.neutral(model)
    pm = pin.buildModelFromUrdf(str(proxy.resolve())); pd = pm.createData()
    pin.forwardKinematics(pm, pd, pin.neutral(pm)); pin.updateFramePlacements(pm, pd)
    for opening in np.linspace(0., .864, 11):
        for j, name in enumerate(model.names):
            if "_eg2_4b2_cad_" in str(name) and model.joints[j].nq:
                q[model.joints[j].idx_q] = opening
        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
        pin.updateGeometryPlacements(model, data, visual, vd)
        for side in ("left", "right"):
            proxy_pose = np.asarray(pd.oMf[pm.getFrameId(f"{side}_hand_proxy")].homogeneous)
            proxy_flange = np.asarray(pd.oMf[pm.getFrameId(
                "arm_link6" if side == "left" else "right_arm_link6")].homogeneous)
            cad_flange = np.asarray(data.oMf[model.getFrameId(f"{side}_arm_link6")].homogeneous)
            transform = np.linalg.inv(proxy_pose) @ proxy_flange @ np.linalg.inv(cad_flange)
            row = report['sides'][side]
            for i, item in enumerate(visual.geometryObjects):
                if not item.name.startswith(side + '_') or not any(token in item.name
                    for token in ('eg2_4b2_cad_', 'dm_tac_w2m_')):
                    continue
                pose = transform @ np.asarray(vd.oMg[i].homogeneous)
                points = np.asarray(item.geometry.vertices()) @ pose[:3, :3].T + pose[:3, 3]
                old_inside = np.max(np.abs(points - row['original_center_m'])
                    - np.asarray(row['original_size_m']) / 2, axis=1) <= 0
                if not old_inside.any():
                    continue
                excess = np.minimum.reduce([np.max(np.abs(points[old_inside] - b['center_m'])
                    - np.asarray(b['size_m']) / 2, axis=1) for b in row['boxes']])
                assert np.max(excess) < 1e-9, (side, item.name, opening)


def test_existing_output_is_never_overwritten(candidate):
    cad, proxy, output, _ = candidate
    original = (output / "geometry-proof.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_candidate(cad, proxy, output)
    assert original == (output / "geometry-proof.json").read_bytes()


@pytest.mark.parametrize("factory_name", (
    "build_dual_arm_scene_profile", "build_rm65_dual_arm_scene_profile",
))
def test_explicit_grasp_candidate_preserves_current_table_binding(candidate, factory_name):
    from robot_motion.rm65 import scene_profiles

    _, _, _, report = candidate
    factory = getattr(scene_profiles, factory_name)
    baseline = factory("dual_arm_hand_proxy_dmtac_grasp")
    profile = factory("dual_arm_hand_proxy_dmtac_grasp",
        dmtac_proxy_urdf_path=Path(report["candidate_urdf"]))
    assert baseline.urdf_path.name == "rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    assert len(baseline.collision_pairs) == 20
    assert len(profile.collision_pairs) == 31
    assert profile.fixed_scene_boxes == baseline.fixed_scene_boxes
    assert profile.fixed_scene_box_overrides == baseline.fixed_scene_box_overrides


@pytest.mark.parametrize("mutation", ("default_path", "asymmetric"))
def test_candidate_cannot_relax_default_or_matching_hand_shapes(candidate, tmp_path, monkeypatch, mutation):
    from robot_motion.rm65 import scene_profiles

    _, _, _, report = candidate
    tree = ET.parse(report["candidate_urdf"])
    if mutation == "asymmetric":
        right = tree.getroot().find("link[@name='right_hand_proxy']")
        right.remove(right.findall("collision")[-1])
        path = tmp_path / "asymmetric.urdf"
        kwargs = {"dmtac_proxy_urdf_path": path}
    else:
        path = tmp_path / "rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
        monkeypatch.setattr(scene_profiles, "require_asset_path", lambda: tmp_path)
        kwargs = {}
    tree.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="matching sequential boxes"):
        build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_strict", **kwargs)


def test_rigid_terminal_keeps_lower_footprint_and_full_height(candidate):
    _, _, output, report = candidate
    original = ET.parse(output / 'rm65_dual_arm_hand_proxy_dmtac_stepped.urdf').getroot()
    rigid = ET.parse(Path(report['rigid_collision_urdf'])).getroot()
    for side in ('left', 'right'):
        a = original.find(f"link[@name='{side}_hand_proxy']")
        b = rigid.find(f"link[@name='{side}_hand_proxy']")
        assert len(b.findall('collision')) == len(b.findall('visual')) == 2
        assert ET.tostring(a.find('collision')) == ET.tostring(b.find('collision'))
        boxes = report['sides'][side]['boxes']
        expected = lower_box_extended_upward(boxes)
        terminal = b.findall('collision')[1]
        center = np.fromstring(terminal.find('origin').get('xyz'), sep=' ')
        size = np.fromstring(terminal.find('geometry/box').get('size'), sep=' ')
        np.testing.assert_allclose(center, expected['center_m'], atol=1e-12)
        np.testing.assert_allclose(size, expected['size_m'], atol=1e-12)
        np.testing.assert_allclose(size[:2], boxes[1]['size_m'][:2], atol=1e-12)
        assert center[2] + size[2]/2 == pytest.approx(boxes[0]['center_m'][2]+boxes[0]['size_m'][2]/2)
        assert center[2] - size[2]/2 == pytest.approx(boxes[1]['center_m'][2]-boxes[1]['size_m'][2]/2)
