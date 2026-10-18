"""Explicit capture policy boundaries; no device or motion resources."""
from dataclasses import replace
import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter as Filter,
    RM65CapturedRobotObservationFilterConfig as Config,
    captured_robot_observation_mask, validate_observation_urdf_candidate,
)
from robot_motion.rm65.scene_profiles import load_dmtac_self_observation_proxy_boxes
from scripts.probes.build_rm65_self_observation_urdf import build_candidate
from tests.test_rm65_dmtac_stepped_proxy import candidate as stepped_candidate


def test_legacy_defaults_positional_config_hash_and_extension_rejection():
    config = Config(7., .01, 4, (0., .864))
    original = dict(silhouette_radius_px=7., depth_tolerance_m=.01,
        depth_consistent_neighbor_candidates=4, finger_sweep_q_rad=(0., .864))
    assert config.semantic_content_hash == canonical_content_hash(original,
        domain="dexteleop.rm65.captured_robot_filter_config.v1")
    assert Config().depth_tolerance_m == .015
    assert Config().body_surface_tolerance_m == 0
    assert Config().tool_surface_bounds_m is None
    for fields in (dict(depth_tolerance_m=.025), dict(body_surface_tolerance_m=.05),
                   dict(tool_surface_bounds_m=((-0.01, .01),)*3)):
        with pytest.raises(ValueError):
            Config(**fields)


@pytest.mark.parametrize("payload", [[], {"unknown": 1}, {"depth_tolerance_m": .025},
    {"geometry_mode": "automatic"}])
def test_explicit_json_rejects_unknown_or_implicit_policy(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        Config.from_json_file(path)


def test_explicit_json_roundtrips_and_changes_input_binding(tmp_path):
    path = tmp_path / "model.urdf"; path.write_text('<robot name="test"/>')
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(dict(geometry_mode="cad_surface", depth_tolerance_m=.025)))
    config = Config.from_json_file(cfg_path)
    filt = Filter.for_binding(urdf_path=path, config=config)
    assert filt.config.depth_tolerance_m == .025 and filt.input_binding_hash
    assert filt._runtime is None
    calibrated = Filter.for_binding(urdf_path=path, config=config,
        tool_tcp_pose_flange_m_rad_by_side={side: np.zeros(6) for side in ('left', 'right')})
    assert calibrated.input_binding_hash != filt.input_binding_hash
    assert Filter.for_binding(urdf_path=path).input_binding_hash is None


@pytest.mark.parametrize("option", ["observation_urdf_path", "self_observation_proxy_urdf_path"])
def test_resources_cannot_implicitly_activate_extended_policy(tmp_path, option):
    path = tmp_path / "model.urdf"; path.write_text('<robot name="test"/>')
    with pytest.raises(ValueError, match="explicit extended"):
        Filter.for_binding(urdf_path=path, **{option: path})


def test_default_finite_neighbor_policy_stays_conservative():
    depth = np.zeros((7, 7)); depth[3, 3] = 1.
    rendered = np.full((7, 7), np.nan); rendered[3, 3] = 1.016; rendered[3, 5] = 1.011
    kwargs = dict(depth_m=depth, target_mask=np.zeros_like(depth, bool),
        rendered_depth_maps_m=(rendered,), silhouette_radius_px=2., depth_tolerance_m=.015,
        depth_consistent_neighbor_candidates=1)
    assert not captured_robot_observation_mask(**kwargs)[3, 3]
    assert captured_robot_observation_mask(**kwargs, extended_policy=True)[3, 3]


@pytest.mark.parametrize('mode', ['legacy', 'cad_surface'])
def test_direct_constructor_rejects_unused_self_boxes(tmp_path, mode):
    from tests.test_rm65_self_observation_primitives import _boxes
    path = tmp_path / 'physical.urdf'; path.write_text('<robot name="test"/>')
    with pytest.raises(ValueError, match='tool surface bounds'):
        Filter(urdf_path=path, config=Config(geometry_mode=mode),
            tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left', 'right')},
            self_observation_proxy_boxes=_boxes())


@pytest.mark.parametrize("mode", ["legacy", "cad_surface"])
def test_input_binding_only_enters_extended_result_evidence(tmp_path, monkeypatch, mode):
    path = tmp_path / "model.urdf"; path.write_text('<robot name="test"/>')
    filt = Filter(urdf_path=path, config=Config(geometry_mode=mode))
    monkeypatch.setattr(filt, '_render_depth_maps', lambda **kw: ((np.ones((1, 2)),), {
        'rendered_depth_map_count': 1, 'rendered_geometry_count_by_state': (1,),
        'rendered_triangle_count_by_state': (1,), 'excluded_capture_origin_camera_meshes': [],
        'tool_from_camera_by_side': {}}))
    result = filt(depth_m=np.ones((1, 2)), target_mask=np.array([[False, True]]),
        intrinsics={'fx': 1., 'fy': 1., 'cx': 0., 'cy': 0.},
        source_from_camera=np.eye(4), capture_qpos_by_side={s: np.zeros(6) for s in ('left', 'right')})
    assert ('filter_input_binding_hash' in result.evidence) == (mode != 'legacy')
    assert result.removed_mask.tolist() == [[True, False]]


@pytest.fixture
def visual_clone(tmp_path):
    source = tmp_path / "physical.urdf"
    source.write_text('<robot name="test"><link name="right_arm_link6"><collision>'
        '<geometry><box size=".1 .1 .1"/></geometry></collision></link></robot>')
    report = build_candidate(source, [dict(name='cable', parent_link='right_arm_link6',
        start_m=[0., 0., 0.], end_m=[0., 0., .05], radius_m=.004)], tmp_path / 'clone')
    return source, report['observation_urdf']


def test_visual_clone_is_separate_and_preserves_physical_binding(visual_clone):
    source, clone = visual_clone
    original = source.read_bytes()
    assert validate_observation_urdf_candidate(source, clone).is_file()
    filt = Filter.for_binding(urdf_path=source, config=Config(geometry_mode='cad_surface'),
        observation_urdf_path=clone)
    assert str(filt.physical_urdf_path) == clone
    assert source.read_bytes() == original


@pytest.mark.parametrize("mutation", ['collision', 'joint', 'duplicate', 'physical', 'missing'])
def test_visual_clone_rejects_physical_or_ambiguous_model_changes(visual_clone, mutation):
    source, clone = visual_clone
    tree = ET.parse(clone); root = tree.getroot()
    if mutation == 'collision':
        ET.SubElement(root.findall('link')[-1], 'collision')
    elif mutation == 'joint':
        root.find('joint').set('type', 'revolute')
    elif mutation == 'duplicate':
        root.append(ET.fromstring(ET.tostring(root.find('joint'))))
    elif mutation == 'physical':
        root.find('link/collision/geometry/box').set('size', '.01 .01 .01')
    else:
        root.remove(root.find('joint'))
    tree.write(clone)
    with pytest.raises(ValueError):
        validate_observation_urdf_candidate(source, clone)


def test_explicit_self_boxes_preserve_native_flange_offset(stepped_candidate):
    _, _, output, report = stepped_candidate
    boxes = load_dmtac_self_observation_proxy_boxes(report['candidate_urdf'])
    assert len(boxes) == 4
    assert {box.parent_link_name for box in boxes} == {'arm_link6', 'right_arm_link6'}
    # The source hand base is displaced 9 mm from its flange.
    root = ET.parse(report['candidate_urdf']).getroot()
    for box in boxes:
        joint = root.find(f"joint[@name='{box.side}_hand_proxy_joint']")
        if joint is None:
            joint = next(j for j in root.findall('joint')
                         if j.find('child').get('link') == box.side + '_hand_proxy')
        xyz = np.fromstring(joint.find('origin').get('xyz'), sep=' ')
        np.testing.assert_allclose(np.asarray(box.joint_origin_xyz)-xyz, [0., 0., .009], atol=1e-12)


@pytest.mark.parametrize("mutation", ['revolute_proxy', 'revolute_base', 'duplicate', 'cycle'])
def test_self_boxes_reject_nonfixed_or_ambiguous_chain(stepped_candidate, tmp_path, mutation):
    _, _, _, report = stepped_candidate
    tree = ET.parse(report['candidate_urdf']); root = tree.getroot()
    joint = next(j for j in root.findall('joint')
                 if j.find('child').get('link') == 'left_hand_proxy')
    if mutation == 'revolute_proxy':
        joint.set('type', 'revolute')
    elif mutation == 'revolute_base':
        parent = joint.find('parent').get('link')
        next(j for j in root.findall('joint') if j.find('child').get('link') == parent).set('type', 'revolute')
    elif mutation == 'duplicate':
        root.append(ET.fromstring(ET.tostring(joint)))
    else:
        joint.find('parent').set('link', 'left_hand_proxy')
    path = tmp_path / 'changed.urdf'; tree.write(path)
    with pytest.raises(ValueError, match='fixed chain|cycle'):
        load_dmtac_self_observation_proxy_boxes(path)
