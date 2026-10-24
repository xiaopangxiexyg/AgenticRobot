"""Explicit nominal alignment identity and consumer binding; no live models or hardware."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import importlib
import subprocess
import sys

import numpy as np
import pytest

from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V4 as V4,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5 as V5,
    dmtac_graspgenx_alignment_profile,
    reviewed_graspgenx_dmtac_solid_geometry,
)
from deployments.perception.sam2_graspgenx_point_grasp import Sam2GraspGenXPointGraspPlanner
from test_agent_gateway_graspgenx_point_grasp import (
    _model_spec, _Segmenter, _PlannerClient, _shared_proposal_fixture,
)

ROOT = Path(__file__).resolve().parents[1]


def _planner(tmp_path, **kwargs):
    return Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(np.ones((4, 4), bool)),
        client=_PlannerClient(np.eye(4)[None], np.array([.9])), **kwargs)


def test_new_dimensions_are_default_and_v4_identity_is_explicitly_preserved(tmp_path):
    default = _planner(tmp_path/"default")
    explicit = _planner(tmp_path/"explicit", alignment_profile_id=V5)
    legacy = _planner(tmp_path/"legacy", alignment_profile_id=V4)
    assert default.alignment_profile_id == V5
    assert default.semantic_content_hash == explicit.semantic_content_hash
    assert legacy.semantic_content_hash == "94c00c292f8127a76b6666767cc9777bea27ee366d0670b73957154e6fad3f23"
    assert default.semantic_content_hash != legacy.semantic_content_hash
    assert default.solid_geometry == explicit.solid_geometry == reviewed_graspgenx_dmtac_solid_geometry()
    assert default.solid_geometry.finger_length_m == pytest.approx(.0897)
    assert default.solid_geometry.palm_z_m == -.024
    assert legacy.solid_geometry.finger_length_m == .101
    assert legacy.solid_geometry.palm_z_m == -.029


def test_new_profile_uses_nominal_dimensions_in_inference_and_capture(tmp_path):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path, alignment_profile_id=V5)
    planner.start()
    try:
        result = planner.plan(image, depth, **options)
        assert len(planner.client.calls) == 1
        sweep = planner.client.calls[0][1]["sweep_volume_params"]
        assert sweep["extents_open"] == (.0705, .03, .0867)
        assert sweep["offset_open"] == pytest.approx((0., 0., .01935))
        assert sweep["fingertip_depth"] == pytest.approx(.0627)
        summary = planner.last_filter_summary
        assert summary["capture_palm_z_m"] == -.024
        assert summary["capture_fingertip_z_m"] == pytest.approx(.0627)
        assert summary["capture_half_pad_height_m"] == .015
        assert summary["graspgenx_alignment_profile"] == V5
        assert result.candidates
    finally:
        planner.close()


@pytest.mark.parametrize("geometry", (reviewed_graspgenx_dmtac_solid_geometry(alignment_profile_id=V4),
    replace(reviewed_graspgenx_dmtac_solid_geometry(alignment_profile_id=V5),
        tool_from_solid_xyz_m=(0., 0., 0.))))
def test_explicit_profile_rejects_mixed_dimensions_and_transforms(tmp_path, geometry):
    with pytest.raises(ValueError, match="selected reviewed|another profile"):
        _planner(tmp_path, alignment_profile_id=V5, solid_geometry=geometry)
def test_default_common_pool_shares_one_inference_across_both_components(tmp_path):
    from deployments.perception.sam2_graspgenx_point_grasp import GraspGenXCandidateSelection
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    pools = []
    def select(pool, *_):
        pools.append(pool)
        return GraspGenXCandidateSelection((0,), {})
    planner.start()
    try:
        result = planner.plan_for_components(image, depth, **options,
            candidate_selectors={"left": select, "right": select})
        assert set(result.results_by_component) == {"left", "right"}
        assert all(value.candidates for value in result.results_by_component.values())
        assert pools[0] is pools[1]
        assert len(planner.client.calls) == 1
        assert planner.client.calls[0][1]["sweep_volume_params"]["extents_open"] == (.0705, .03, .0867)
    finally:
        planner.close()
@pytest.mark.parametrize("module", ("agent_gateway.profiles.real.gateway_cli",
    "agent_gateway.profiles.real.perception_cli", "agent_gateway.profiles.mujoco.gateway_cli",
    "deployments.perception.point_grasp_image_cli"))
def test_existing_cli_uses_v5_default_and_explicit_v4_choice(module):
    parser = importlib.import_module(module).build_parser()
    option = parser._option_string_actions["--graspgenx-alignment-profile"]
    assert option.default == V5 and tuple(option.choices) == (V5, V4)
@pytest.mark.parametrize("profile_id,side,tolerance", ((V4, "left", 2e-5), (V4, "right", 1e-12), (V5, "left", 1e-12), (V5, "right", 1e-12)))
def test_profile_provenance_matches_calibration_and_actual_q0_frame_geometry(profile_id, side, tolerance):
    from utils.graspgenx_dmtac_alignment_gui import AlignmentModel, current_dmtac_state, load_tool_tcp_pose
    path = ROOT/"deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json"
    data = json.loads(path.read_text())
    source = data["provenance"]
    assert source["frozen_input_commit"] == "1013deb1546c97599caa928bb24ddb4098d5169f"
    for key in ("source_urdf", "tool_profile", "left_tool_profile"):
        assert not Path(source[key]).is_absolute()
        assert hashlib.sha256((ROOT/source[key]).read_bytes()).hexdigest() == source[key+"_sha256"]
    profile = dmtac_graspgenx_alignment_profile(profile_id)
    if profile_id == V5:
        assert profile.source_content_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        assert profile.source_content_hash is None
    tool_profile = ROOT/f"hardware/rm65/utils/configs/calibrations/rm65_{side}_DMTAC_M.json"
    state = current_dmtac_state(side, load_tool_tcp_pose(tool_profile), [0.] * 6,
        alignment_profile_id=profile_id)
    frames, _ = AlignmentModel(ROOT/source["source_urdf"], state).frames()
    actual = np.linalg.inv(np.asarray(frames["tool_tcp"])) @ np.asarray(frames["graspgenx_model"])
    np.testing.assert_allclose(actual, profile.nominal_geometry.solid_pose(np.eye(4)), rtol=0., atol=tolerance)
    assert source["physical_acceptance"] == "not_assessed"


def test_selected_profile_is_in_setuptools_package_data_without_building():
    program = r'''
import json, sys
from pathlib import Path
from setuptools import Distribution
from setuptools.command.build_py import build_py
from setuptools.config.pyprojecttoml import read_configuration
root = Path(sys.argv[1])
config = read_configuration(root / "pyproject.toml", expand=False)
patterns = config["tool"]["setuptools"]["package-data"]
command = build_py(Distribution())
command.manifest_files = {}
command.exclude_package_data = {}
command.package_data = patterns
package = "deployments.perception"
source = str(root / "deployments/perception")
record = root / "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json"
selected = record in map(Path, command.find_data_files(package, source))
command.package_data = {key:value for key,value in patterns.items() if key != package}
selected_without_explicit_pattern = record in map(Path, command.find_data_files(package, source))
print(json.dumps(dict(selected=selected, selected_without_explicit_pattern=selected_without_explicit_pattern)))
'''
    result = subprocess.run([sys.executable, "-c", program, str(ROOT)], text=True,
        capture_output=True, timeout=30, check=True)
    assert json.loads(result.stdout) == {"selected": True, "selected_without_explicit_pattern": False}


def test_mujoco_parser_keeps_optional_perception_unloaded():
    program = r'''
import importlib.abc, sys
class BlockOptionalPerception(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "deployments.perception" or fullname.startswith("deployments.perception."):
            raise AssertionError("optional perception imported: " + fullname)
        return None
sys.meta_path.insert(0, BlockOptionalPerception())
from agent_gateway.profiles.mujoco import gateway_cli
from deployments.dmtac_graspgenx_alignment_ids import DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID
parser = gateway_cli.build_parser()
assert parser.get_default("graspgenx_alignment_profile") == DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID
assert "deployments.perception" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True, timeout=30)
