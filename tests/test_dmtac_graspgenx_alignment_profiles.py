"""Explicit nominal alignment identity and consumer binding; no live models or hardware."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import importlib
import subprocess
import sys

import numpy as np
import pytest

from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID as V4,
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


def test_default_alignment_identity_is_unchanged_and_explicit_v5_is_distinct(tmp_path):
    default = _planner(tmp_path/"default")
    explicit = _planner(tmp_path/"explicit", alignment_profile_id=V4)
    new = _planner(tmp_path/"new", alignment_profile_id=V5)
    assert default.semantic_content_hash == explicit.semantic_content_hash == (
        "94c00c292f8127a76b6666767cc9777bea27ee366d0670b73957154e6fad3f23")
    assert default.solid_geometry == explicit.solid_geometry == reviewed_graspgenx_dmtac_solid_geometry()
    assert new.alignment_profile_id == V5 and new.semantic_content_hash != default.semantic_content_hash
    assert new.solid_geometry.finger_length_m == pytest.approx(.0897)
    assert new.solid_geometry.palm_z_m == -.024
    assert default.solid_geometry.finger_length_m == .101


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


@pytest.mark.parametrize("geometry", (reviewed_graspgenx_dmtac_solid_geometry(),
    replace(reviewed_graspgenx_dmtac_solid_geometry(alignment_profile_id=V5),
        tool_from_solid_xyz_m=(0., 0., 0.))))
def test_explicit_profile_rejects_mixed_dimensions_and_transforms(tmp_path, geometry):
    with pytest.raises(ValueError, match="selected reviewed|another profile"):
        _planner(tmp_path, alignment_profile_id=V5, solid_geometry=geometry)


@pytest.mark.parametrize("sides", ((), ("left",), ("left", "right"), ("right", "right")))
def test_right_profile_rejects_unbound_left_and_multiple_motion_contexts(sides):
    with pytest.raises(ValueError, match="one right-arm"):
        dmtac_graspgenx_alignment_profile(V5).require_motion_sides(sides)
    dmtac_graspgenx_alignment_profile(V4).require_motion_sides(sides)


@pytest.mark.parametrize("bindings", (None, {"left": "left"}, {"left": "left", "right": "right"}))
def test_shared_pool_rejects_unsupported_side_before_any_inference(tmp_path, monkeypatch, bindings):
    planner = _planner(tmp_path, alignment_profile_id=V5)
    monkeypatch.setattr(planner, "_prepare_pool", lambda *a, **k: pytest.fail("No inference before side binding"))
    selectors = {key: (lambda *a: None) for key in (bindings or {"right": "right"})}
    with pytest.raises(ValueError, match="arm-side|one right-arm"):
        planner.plan_for_components(None, None, pixel_uv=(0, 0), intrinsics={},
            base_from_camera=np.eye(4), max_candidates=1, candidate_selectors=selectors,
            component_sides=bindings)
    assert planner.client.calls == []


def test_explicit_right_pool_uses_one_inference_with_named_side_binding(tmp_path):
    from deployments.perception.sam2_graspgenx_point_grasp import GraspGenXCandidateSelection
    planner, image, depth, options = _shared_proposal_fixture(tmp_path, alignment_profile_id=V5)
    planner.start()
    try:
        result = planner.plan_for_components(image, depth, **options,
            candidate_selectors={"selected": lambda pool, *_: GraspGenXCandidateSelection((0,), {})},
            component_sides={"selected": "right"})
        assert set(result.results_by_component) == {"selected"}
        assert result.results_by_component["selected"].candidates
        assert len(planner.client.calls) == 1
    finally:
        planner.close()


@pytest.mark.parametrize("component", ("left_arm", "right_arm"))
def test_native_provider_checks_selected_alignment_against_execution_component(tmp_path, component):
    from agent_gateway.profiles.mujoco.grasp_fast_plan import MujocoFastGraspPlanProvider
    planner = _planner(tmp_path, alignment_profile_id=V5)
    bundle = SimpleNamespace(fast_grasp_contract=SimpleNamespace(
        model_profile_hash=planner.semantic_content_hash, execution_contract=SimpleNamespace(
            motion_profile=SimpleNamespace(component_id=component))))
    args = dict(application=None, bundle=bundle, model=planner, observation_executor=None,
        artifact_store=None, operation_lookup=None, current_station_revision=lambda: 0, clock=lambda: 0.)
    if component == "left_arm":
        with pytest.raises(ValueError, match="one right-arm"):
            MujocoFastGraspPlanProvider(**args)
    else:
        assert MujocoFastGraspPlanProvider(**args).model is planner
    assert not planner.client.started and planner.client.calls == []


@pytest.mark.parametrize("side", ("left", "right"))
def test_real_provider_checks_alignment_against_its_reachability_binding(monkeypatch, side):
    import test_agent_gateway_real_station_point_grasp as fixture
    profile = dmtac_graspgenx_alignment_profile(V5)
    monkeypatch.setattr(fixture._Planner, "require_alignment_motion_sides",
        lambda self, sides: profile.require_motion_sides(sides), raising=False)
    original = fixture.StationPointGraspArmReachabilityProfile
    monkeypatch.setattr(fixture, "StationPointGraspArmReachabilityProfile",
        lambda **kwargs: original(**{**kwargs, "arm_side": side}))
    if side == "left":
        with pytest.raises(ValueError, match="one right-arm"):
            fixture._provider_stack(arm_reachability_evaluator=lambda *_: (), backend_id="graspgenx")
    else:
        source, _, _, _, provider = fixture._provider_stack(
            arm_reachability_evaluator=lambda *_: (), backend_id="graspgenx")
        assert provider.registration.profile.arm_reachability_profile.arm_side == "right"
        assert source.read_count == 0


@pytest.mark.parametrize("module", ("agent_gateway.profiles.real.gateway_cli",
    "agent_gateway.profiles.real.perception_cli", "agent_gateway.profiles.mujoco.gateway_cli",
    "deployments.perception.point_grasp_image_cli"))
def test_existing_cli_profile_selection_keeps_v4_default(module):
    parser = importlib.import_module(module).build_parser()
    option = parser._option_string_actions["--graspgenx-alignment-profile"]
    assert option.default == V4 and tuple(option.choices) == (V4, V5)


def test_replay_rejects_new_left_or_shared_calibration_before_reading_inputs(monkeypatch):
    from scripts.probes import replay_bimanual_grasp_snapshot as replay
    monkeypatch.setattr(replay, "prepare", lambda *_: pytest.fail("No input reads before side admission"))
    for sides in (("left",), ("left", "right")):
        with pytest.raises(ValueError, match="one right-arm"):
            replay.replay(SimpleNamespace(graspgenx_alignment_profile=V5, sides=sides))


@pytest.mark.parametrize("profile_id,side,tolerance", ((V4, "left", 2e-5), (V4, "right", 1e-12), (V5, "right", 1e-12)))
def test_profile_provenance_matches_calibration_and_actual_q0_frame_geometry(profile_id, side, tolerance):
    from utils.graspgenx_dmtac_alignment_gui import AlignmentModel, current_dmtac_state, load_tool_tcp_pose
    path = ROOT/"deployments/perception/profiles/dmtac_graspgenx_alignment_v5_right_q0.json"
    data = json.loads(path.read_text())
    source = data["provenance"]
    assert source["frozen_input_commit"] == "1013deb1546c97599caa928bb24ddb4098d5169f"
    for key in ("source_urdf", "tool_profile"):
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
record = root / "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_right_q0.json"
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
