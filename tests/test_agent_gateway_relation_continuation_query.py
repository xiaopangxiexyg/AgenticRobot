"""Offline v3 geometry/provenance boundaries; no native simulator or hardware."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_agent_gateway_surface_anchor import (
    _relation_parameters,
    _v2_registrations,
    anchor_case,  # noqa: F401 - Register the shared fixture.
)
from test_agent_gateway_tcp_continuation_query import _result

from agent_gateway.profiles import relation_continuation_query as module
from agent_gateway.profiles.query_geometry import relation_candidate_poses
from agent_gateway.profiles.tcp_continuation_query import (
    build_tcp_continuation_query_registration,
)
from agent_gateway.skills.perception.relation_continuation_query import (
    POSES_KEY,
    explicit_continuation_parameters,
    validate_relation_continuation_parameters,
)
from agent_gateway.skills.perception.surface_anchor_v2 import SELECTION_FIELDS

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _parameters():
    return {**_relation_parameters(), "continuation_delta_xyz_m": [0, 0, -.06], "path_constraint": None}


def _registrations(*, selection_policy="prompt_nearest"):
    projection, _, _, (capture, _, relation) = _v2_registrations(selection_policy=selection_policy)
    continuation = build_tcp_continuation_query_registration(base_route=projection.skill.routes[0],
        base_binding=projection.binding, component_id="right_arm", reference_frame_id="canonical_station",
        planning_profile_hash="planning:test")
    return capture, relation, continuation, module.build_relation_continuation_registration(
        capture=capture, relation=relation, continuation=continuation)


def test_continuation_registration_and_executor_reject_mixed_anchor_selection_profiles():
    capture, relation, continuation, registration = _registrations()
    other_capture, other_relation, _, other_registration = _registrations(selection_policy="max_clearance_medoid")
    assert registration.skill.definition == other_registration.skill.definition
    assert registration.entry.route.semantic_content_hash != other_registration.entry.route.semantic_content_hash
    for producer, consumer in ((capture, other_relation), (other_capture, relation)):
        with pytest.raises(ValueError, match="same exact anchor result schema"):
            module.build_relation_continuation_registration(capture=producer, relation=consumer,
                continuation=continuation)
    for producer, consumer in ((capture, other_registration), (other_capture, registration)):
        with pytest.raises(ValueError, match="same exact anchor result schema"):
            module.build_relation_continuation_executor(registration=consumer,
                capture_registration=producer, query_port=None, store=None,
                lookup_operation=None, current_station_revision=lambda: 0,
                clock=lambda: 0., clock_domain_id="test", max_retained_operations=4)


def _path(parameters, anchor):
    initial = [.1, .2, .3, 0, 0, 0, 1]
    poses = relation_candidate_poses(parameters, initial=initial, anchor_xyz_m=anchor)
    return {**_result(explicit_continuation_parameters(parameters, poses)), POSES_KEY: poses}


@pytest.mark.parametrize("offsets,turns", [(1, 8), (8, 1), (2, 4)])
def test_v3_accepts_bounded_relation_product_without_caller_pose_array(offsets, turns):
    params = {**_parameters(), "offsets_xyz_m": [[i * .01, 0, .1] for i in range(offsets)],
        "tool_axis_rotation_offsets_rad": [i * .1 for i in range(turns)]}
    validate_relation_continuation_parameters(params)
    before = deepcopy(params)
    registration = _registrations()[-1]
    registration.payload_validator.validate_parameters(params)
    assert params == before


@pytest.mark.parametrize("mutation", ["product", "poses", "xyz", "tcp", "budget", "delta", "constraint"])
def test_v3_rejects_unbounded_or_caller_authored_geometry(mutation):
    params = _parameters()
    if mutation == "product":
        params["offsets_xyz_m"] *= 5  # 5 * 2, not silently truncated to 8.
    elif mutation in {"poses", "xyz", "tcp"}:
        params[{"poses": POSES_KEY, "xyz": "anchor_xyz_m", "tcp": "current_tcp_pose_xyz_quat_xyzw"}[mutation]] = []
    elif mutation == "budget":
        params["time_budget_sec"] = True
    elif mutation == "delta":
        params["continuation_delta_xyz_m"] = [0, 0, float("nan")]
    else:
        params["path_constraint"] = {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": 0}
    with pytest.raises((TypeError, ValueError)):
        _registrations()[-1].payload_validator.validate_parameters(params)


_NEUTRAL_QUERY_IMPORT_GUARD = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
# The shared canonical codec already depends transitively on NumPy to encode
# numeric types. That is not a new motion-geometry dependency. Direct NumPy
# imports in neutral contracts remain forbidden by the existing AST gate.
# Deny the implementation owners, including their geometry submodules; do not
# preload any dependency or hide an implementation import behind lazy loading.
blocked = ('robot_motion', 'scipy', 'sim', 'agent_gateway.profiles',
           'control_station', 'deployments', 'hardware', 'teleop_runtime')
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Neutral contract imported a backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
"""


def _run_neutral_query_process(code, argument):
    return subprocess.run([sys.executable, "-I", "-c", _NEUTRAL_QUERY_IMPORT_GUARD + code,
        str(Path(__file__).resolve().parents[1]), argument],
        capture_output=True, text=True, timeout=15., check=False)


def test_neutral_query_contracts_validate_without_motion_or_profile_backends():
    code = """
import json
from agent_gateway.skills.perception.object_goal_query import validate_query_parameters
from agent_gateway.skills.perception.relation_continuation_query import validate_relation_continuation_parameters
from agent_gateway.skills.perception.region_continuation_query import validate_region_continuation_parameters
from agent_gateway.skills.perception.spatial_goal import validate_spatial_goal_result
from agent_gateway.skills.perception import surface_anchor
parameters = json.loads(sys.argv[2])
validate_query_parameters(parameters['above'])
validate_relation_continuation_parameters(parameters['relation'])
for version in ('4', '5'):
    validate_region_continuation_parameters(parameters['region'], version=version)
validate_spatial_goal_result(parameters['goal'])
assert not any(any(name == root or name.startswith(root + '.') for root in blocked) for name in sys.modules)
"""
    parameters = {"above": {"component_id": "right_arm", "reference_frame_id": "canonical_station",
        "anchor_xyz_m": [0., 0., 0.], "heights_m": [.1], "tool_yaw_offsets_rad": [0.], "time_budget_sec": 1.},
        "relation": _parameters(), "region": {**_parameters(),
            "offset_bounds_xyz_m": [[0., .1], [0., 0.], [.1, .2]],
            "tool_axis_rotation_bounds_rad": [-1., 1.], "max_candidates": 8},
        "goal": {"component_id": "right_arm", "reference_frame_id": "canonical_station",
            "pose_targets_by_component": {"right_arm": [0., 0., 0., 0., 0., 0., 1.]},
            "translation_distance_m": 0., "rotation_distance_rad": 0., "axis_alignment_error_rad": None,
            "constraint_scope": "endpoint_only", "input_source": "caller_supplied_geometry",
            "path_checked": False, "motion_executed": False}}
    del parameters["region"]["offsets_xyz_m"], parameters["region"]["tool_axis_rotation_offsets_rad"]
    result = _run_neutral_query_process(code, json.dumps(parameters))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", [
    "robot_motion", "robot_motion.spatial_goals", "robot_motion.spatial_goal_regions",
    "scipy", "scipy.spatial.transform", "sim", "sim.mujoco.deploy.planned_tcp_motion",
    "agent_gateway.profiles", "agent_gateway.profiles.query_geometry", "agent_gateway.profiles.spatial_goal",
    "control_station", "deployments", "hardware", "teleop_runtime",
])
def test_neutral_query_import_guard_rejects_backend_and_geometry_counterexamples(module_name):
    # Use the same installed finder in a fresh process, so no already-imported
    # backend can bypass it and no forbidden implementation reaches execution.
    result = _run_neutral_query_process("\nimportlib.import_module(sys.argv[2])\n", module_name)
    assert result.returncode != 0
    denied_owner = "agent_gateway.profiles" if module_name.startswith("agent_gateway.profiles") else module_name.split(".")[0]
    assert f"AssertionError: Neutral contract imported a backend: {denied_owner}\n" in result.stderr, result.stderr


@pytest.mark.parametrize("change", ["offset_bool", "turn_bool", "axis_zero", "axis_overflow",
    "frame", "preserve_axis", "preserve_turn", "empty_offsets", "empty_turns"])
def test_neutral_relation_parameters_keep_type_and_geometry_choice_guards(change):
    params = _parameters()
    if change == "offset_bool":
        params["offsets_xyz_m"][0][0] = True
    elif change == "turn_bool":
        params["tool_axis_rotation_offsets_rad"][0] = False
    elif change.startswith("axis_"):
        params["axis_reference_xyz"] = [0., 0., 0.] if change == "axis_zero" else [1.7e308] * 3
    elif change == "frame":
        params["offset_frame"] = "camera"
    elif change == "preserve_axis":
        params["orientation_mode"] = "preserve"
    elif change == "preserve_turn":
        params.update(orientation_mode="preserve", tool_axis=None, axis_reference_xyz=None,
            tool_axis_rotation_offsets_rad=[.1])
    else:
        params["offsets_xyz_m" if change == "empty_offsets" else "tool_axis_rotation_offsets_rad"] = []
    with pytest.raises((TypeError, ValueError)):
        validate_relation_continuation_parameters(params)


@pytest.mark.parametrize("version", ["3", "4", "5"])
def test_profile_rejects_an_invalid_generated_quaternion_at_admission(monkeypatch, version):
    from agent_gateway.profiles import query_geometry
    from agent_gateway.profiles.region_continuation_query import build_region_continuation_registration

    capture, relation, continuation, registration = _registrations()
    params = _parameters()
    generator = "relation_candidate_poses"
    if version != "3":
        registration = build_region_continuation_registration(capture=capture, relation=relation,
            continuation=continuation, version=version)
        params = {key: value for key, value in params.items()
            if key not in {"offsets_xyz_m", "tool_axis_rotation_offsets_rad"}}
        params.update(offset_bounds_xyz_m=[[0., .1], [0., 0.], [.1, .2]],
            tool_axis_rotation_bounds_rad=[-1., 1.], max_candidates=8)
        generator = "region_candidate_poses"
    monkeypatch.setattr(query_geometry, generator, lambda *args, **kwargs: [[0.] * 7])
    with pytest.raises(ValueError):
        registration.payload_validator.validate_parameters(params)


def test_v3_keeps_legacy_routes_and_exact_anchor_input_contract():
    capture, relation, legacy, combined = _registrations()
    assert legacy.skill.definition.skill_version == "1" and relation.skill.definition.skill_version == "2"
    assert combined.skill.definition.skill_version == "3"
    route = combined.entry.route
    assert route.control_effect_class == "read_only" and route.cancellation_policy == "cancel_only"
    assert not route.may_move_robot and not route.may_open_publication
    combined.payload_validator.validate_route(route)
    slot = combined.skill.io_contract.input_artifact_slots[0]
    assert replace(slot, direction="output") == next(item for item in capture.skill.io_contract.output_artifact_slots
        if item.slot_id == "surface_anchor")
    assert combined.skill.io_contract.output_artifact_slots == ()
    assert len({registration.entry.route.route_id for registration in (relation, legacy, combined)}) == 3
    assert _registrations()[-1].entry == combined.entry


@pytest.mark.parametrize("component,frame", [("left_arm", "canonical_station"), ("right_arm", "other_frame")])
def test_v3_rejects_inconsistent_source_binding_before_advertising(component, frame):
    projection, _, _, (capture, _, relation) = _v2_registrations()
    continuation = build_tcp_continuation_query_registration(base_route=projection.skill.routes[0],
        base_binding=projection.binding, component_id=component, reference_frame_id=frame,
        planning_profile_hash="planning:test")
    with pytest.raises(ValueError, match="same component and public frame"):
        module.build_relation_continuation_registration(capture=capture, relation=relation, continuation=continuation)


@pytest.mark.parametrize("changed", ["target", "selected_candidate"])
def test_standalone_result_cannot_select_a_different_candidate(request, changed):
    case = request.getfixturevalue("anchor_case")
    _, _, _, registration = _registrations()
    selection = dict.fromkeys(SELECTION_FIELDS)
    selection.update(mode="exact", requested_pixel_uv=[2, 1], language_prompt=None, min_score=.25,
        algorithm_id="exact_pixel.v1", input_rgb_content_hash="c" * 64)
    anchor = {**case.payload, "selection": selection, "mask_geometry": None}
    path = _path(_parameters(), anchor["measurement"]["world_xyz_m"])
    result = {"surface_anchor": anchor, "anchor_artifact_id": "artifact:test", "path_query": path,
        "relation_constraint_scope": "endpoint_only"}
    registration.payload_validator.validate_execution_result(result)
    pose = path["first_pose_target_xyz_quat_xyzw"] if changed == "target" else path[POSES_KEY][path["selected_index"]]
    pose[0] += .01
    with pytest.raises(ValueError, match="selected first target"):
        registration.payload_validator.validate_execution_result(result)


@pytest.mark.parametrize("mode", ["success", "cancel", "deadline", "revision", "client", "missing_producer",
    "wrong_anchor_version", "changed_candidate", "changed_selection", "bad_count", "authority"])
@pytest.mark.parametrize("selection_policy", ["prompt_nearest", "max_clearance_medoid"])
def test_combined_executor_authenticates_anchor_and_both_stage_evidence(request, monkeypatch, mode, selection_policy):
    capture, _, _, registration = _registrations(selection_policy=selection_policy)
    case = request.getfixturevalue("anchor_case")
    selection = dict.fromkeys(SELECTION_FIELDS)
    selection.update(mode="exact", requested_pixel_uv=[2, 1], language_prompt=None, min_score=.25,
        algorithm_id="exact_pixel.v1", input_rgb_content_hash="c" * 64)
    anchor = {**case.payload, "selection": selection, "mask_geometry": None}
    binding = case.put(anchor, version="2")
    case.source.execution_result = anchor
    case.source.output_artifact_bindings = (binding,)
    case.source_request.skill_version = "2"
    case.source_request.route_hash = capture.entry.route.semantic_content_hash
    case.source_request.deployment_binding_hash = capture.entry.deployment_binding.deployment_binding_hash
    case.request.input_artifact_bindings = (binding,)
    case.request.parameters = _parameters()
    calls, revision = [], [7]

    class CallbackExecutor:
        def __init__(self, *, read, **kwargs):
            self.read = read

        def cancel_requested(self, operation_id):
            return mode == "cancel"

    def query(parameters, *, cancel_requested):
        calls.append(deepcopy(parameters))
        if cancel_requested():
            raise RuntimeError("cancelled")
        result = _path(case.request.parameters, anchor["measurement"]["world_xyz_m"])
        if mode == "revision":
            revision[0] += 1
        elif mode == "changed_candidate":
            result[POSES_KEY][0][0] += .01
        elif mode == "changed_selection":
            result["first_pose_target_xyz_quat_xyzw"][0] += .01
        elif mode == "bad_count":
            result[POSES_KEY].pop()
        elif mode == "authority":
            result["execution_authorized"] = True
        return result

    if mode == "client":
        case.request.client_id = "another-client"
    if mode == "wrong_anchor_version":
        case.request.input_artifact_bindings = (case.put(case.payload),)
    monkeypatch.setattr(module, "PassiveReadSkillExecutor", CallbackExecutor)
    executor = module.build_relation_continuation_executor(registration=registration,
        capture_registration=capture, query_port=SimpleNamespace(query_relation_continuations=query),
        store=case.kwargs["store"], lookup_operation=(lambda _: (None, None)) if mode == "missing_producer"
            else case.kwargs["lookup_operation"], current_station_revision=lambda: revision[0],
        clock=lambda: 10., clock_domain_id="clock:test", max_retained_operations=4)
    operation = SimpleNamespace(operation_id="operation:query", deadline_at=9. if mode == "deadline" else 30.)
    if mode != "success":
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            executor.read(case.request, operation)
        assert len(calls) == (0 if mode in {"client", "wrong_anchor_version", "missing_producer"} else 1)
        return
    answer = executor.read(case.request, operation)
    registration.payload_validator.validate_execution_result_for_parameters(case.request.parameters, answer.execution_result)
    assert len(calls) == 1 and calls[0]["anchor_xyz_m"] == anchor["measurement"]["world_xyz_m"]
    assert "first_pose_candidates_xyz_quat_xyzw" not in calls[0]
    assert answer.execution_result["anchor_artifact_id"] == binding.artifact.artifact_id
    assert not answer.output_artifact_bindings
    assert not any("hold" in ref.lower() for ref in answer.evidence_refs)
