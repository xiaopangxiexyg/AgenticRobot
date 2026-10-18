from xml.etree import ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scripts.probes.build_rm65_self_observation_urdf import build_candidate, validate_capsules
from robot_motion.rm65.captured_robot_observations import (
    _is_robot_visual_mesh, _tool_surface_side, captured_robot_observation_mask,
    RM65CapturedRobotObservationFilter, RM65CapturedRobotObservationFilterConfig,
)


def capsule():
    return dict(name='wrist_cable', parent_link='right_arm_link6',
                start_m=[.01, .02, .03], end_m=[.01, .02, .06], radius_m=.004)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.urdf'
    path.write_text('<robot name="fixture"><link name="right_arm_link6">'
                    '<collision><geometry><box size=".1 .1 .1"/></geometry></collision>'
                    '</link><link name="left_arm_link6"/><link name="base"/>'
                    '<joint name="left" type="fixed"><parent link="base"/>'
                    '<child link="left_arm_link6"/></joint>'
                    '<joint name="right" type="revolute"><parent link="base"/>'
                    '<child link="right_arm_link6"/><axis xyz="0 0 1"/>'
                    '<limit lower="-2" upper="2" effort="1" velocity="1"/></joint></robot>')
    return path


@pytest.mark.parametrize('change', [dict(parent_link='unknown'), dict(name='../bad'),
    dict(radius_m=-.1), dict(radius_m=True), dict(start_m=[np.nan, 0, 0]),
    dict(end_m=[.01, .02, .03]), dict(start_m=[0, 0])])
def test_invalid_attachment_fails_before_output(source, tmp_path, change):
    row = capsule(); row.update(change)
    with pytest.raises(ValueError):
        build_candidate(source, [row], tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_duplicate_attachment_rejected(source):
    with pytest.raises(ValueError, match='collides'):
        validate_capsules(ET.parse(source).getroot(), [capsule(), capsule()])


def test_visual_only_capsule_preserves_source_collision_and_parent_coordinates(source, tmp_path):
    import trimesh
    original = source.read_bytes()
    out = tmp_path / 'new'
    report = build_candidate(source, [capsule()], out)
    assert source.read_bytes() == original
    before = ET.parse(source).getroot(); after = ET.parse(out / 'self_observation.urdf').getroot()
    assert ET.tostring(before.find('link')) == ET.tostring(after.find('link'))
    assert report['collision_elements_before'] == report['collision_elements_after'] == 1
    added = after.find("link[@name='right_self_observation_wrist_cable']")
    assert added.find('collision') is None
    joint = after.find("joint[@name='right_self_observation_wrist_cable_fixed']")
    assert joint.attrib['type'] == 'fixed'
    assert joint.find('parent').attrib['link'] == 'right_arm_link6'
    mesh = trimesh.load(out / added.find('visual/geometry/mesh').attrib['filename'])
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), [.01, .02, .045], atol=1e-8)
    assert mesh.bounds[1, 2] == pytest.approx(.064, abs=1e-6)
    # Existing raycast/depth owner includes these surfaces, without applying
    # the much wider tool-surface allowance around the newly added capsules.
    name = 'right_self_observation_wrist_cable_0'
    assert _is_robot_visual_mesh(name, finger_only=False)
    assert _is_robot_visual_mesh(name, finger_only=True)
    assert _tool_surface_side(name) is None
    with pytest.raises(FileExistsError):
        build_candidate(source, [capsule()], out)


def test_native_fixed_attachment_follows_parent_without_new_dof(source, tmp_path):
    pin = pytest.importorskip('pinocchio')
    out = tmp_path / 'native'
    build_candidate(source, [capsule()], out)
    model, _, visuals = pin.buildModelsFromUrdf(str(out / 'self_observation.urdf'), package_dirs=[str(out)])
    assert model.nq == pin.buildModelFromUrdf(str(source)).nq == 1
    data = model.createData()
    for angle in (0., .7, -1.2):
        pin.forwardKinematics(model, data, np.array([angle])); pin.updateFramePlacements(model, data)
        parent = data.oMf[model.getFrameId('right_arm_link6')].homogeneous
        child = data.oMf[model.getFrameId('right_self_observation_wrist_cable')].homogeneous
        np.testing.assert_allclose(parent, child)
    assert any(_is_robot_visual_mesh(g.name, finger_only=False) for g in visuals.geometryObjects)


def test_real_capsule_depth_keeps_targets_and_foreground_background_objects(source, tmp_path):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    out = tmp_path / 'depth'
    row = capsule(); row.update(start_m=[0, 0, .99], end_m=[0, 0, 1.01])
    build_candidate(source, [row], out)
    mesh = trimesh.load(out / 'right_self_observation_wrist_cable.obj')
    hits, _, _ = RayMeshIntersector(mesh).intersects_location(
        np.zeros((1, 3)), np.array([[0., 0., 1.]]), multiple_hits=False)
    assert hits[0, 2] == pytest.approx(.986, abs=1e-6)
    # Same silhouette; only the observed cable depth is explained. Target
    # labels win even if their depth overlaps the explicitly modeled cable.
    depth = np.array([[1., 1., .94, 1.06]])
    target = np.array([[False, True, False, False]])
    nominal = np.full(depth.shape, .408)
    attachment = np.full(depth.shape, hits[0, 2])
    removed = captured_robot_observation_mask(depth_m=depth, target_mask=target,
        rendered_depth_maps_m=(nominal, attachment), silhouette_radius_px=12., depth_tolerance_m=.025)
    np.testing.assert_array_equal(removed, [[True, False, False, False]])


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('location', ['wrist', 'finger'])
def test_actual_rm65_attachment_sweeps_moving_parent_but_deduplicates_wrist(tmp_path, side, location):
    pytest.importorskip('pinocchio'); pytest.importorskip('trimesh.ray.ray_pyembree')
    source = Path(__file__).resolve().parents[1] / 'assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf'
    if not source.is_file():
        pytest.skip('Reviewed CAD unavailable')
    row = capsule(); row['parent_link'] = (side + '_arm_link6' if location == 'wrist'
                                          else side + '_gripper_r_dm_tac_w2m_sensor')
    out = tmp_path / 'candidate'
    build_candidate(source, [row], out)
    filt = RM65CapturedRobotObservationFilter(urdf_path=out / 'self_observation.urdf',
        config=RM65CapturedRobotObservationFilterConfig(geometry_mode='cad_surface',
            tool_surface_bounds_m=((-0.045, .045), (-.05, .05), (-.05, .05))),
        tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left', 'right')})
    _, evidence = filt._render_depth_maps(depth_m=np.ones((4, 4)),
        intrinsics={'fx': 10., 'fy': 10., 'cx': 2., 'cy': 2.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={s: np.zeros(6) for s in ('left', 'right')})
    assert evidence['self_observation_attachment_geometry_count'] == 1
    layers = 1 if location == 'wrist' else len(filt.config.finger_sweep_q_rad)
    assert evidence['self_observation_attachment_layer_count'] == layers
    assert evidence['rendered_depth_map_count'] == 1 + layers
