"""Region v4 compatibility and provenance checks; no devices or simulator."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_agent_gateway_relation_continuation_query import _registrations
from test_agent_gateway_surface_anchor import anchor_case  # noqa: F401
from test_agent_gateway_tcp_continuation_query import _result

from agent_gateway.profiles import region_continuation_query as region
from agent_gateway.profiles import relation_continuation_query as shared
from agent_gateway.profiles.mujoco.object_goal_query import MujocoObjectGoalQuery
from agent_gateway.profiles.query_geometry import (
    region_candidate_poses,
    validate_region_continuation_path,
)
from agent_gateway.skills.perception.region_continuation_query import validate_region_continuation_parameters
from agent_gateway.skills.perception.relation_continuation_query import (
    POSES_KEY,
    explicit_continuation_parameters,
)
from agent_gateway.skills.perception.surface_anchor_v2 import SELECTION_FIELDS

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _parameters():
    return {"component_id": "right_arm", "reference_frame_id": "canonical_station",
        "offset_bounds_xyz_m": [[.02, .06], [-.05, -.01], [.14, .20]],
        "offset_frame": "reference", "orientation_mode": "align_axis", "tool_axis": "z",
        "axis_reference_xyz": [0., 0., -1.], "tool_axis_rotation_bounds_rad": [-1., 1.],
        "max_candidates": 6, "continuation_delta_xyz_m": [0., 0., -.06],
        "path_constraint": None, "time_budget_sec": 12.}


def _registration(version="4"):
    capture, relation, continuation, _ = _registrations()
    return capture, region.build_region_continuation_registration(
        capture=capture, relation=relation, continuation=continuation, version=version)


def _path(parameters, anchor, *, initial=None, version="4"):
    initial = [.1, .2, .3, 0., 0., 0., 1.] if initial is None else initial
    poses = region_candidate_poses(parameters, initial=initial, anchor_xyz_m=anchor, version=version)
    answer = _result(explicit_continuation_parameters(parameters, poses))
    answer["source_tcp_pose_xyz_quat_xyzw"] = initial
    return {**answer, POSES_KEY: poses}


def test_v4_keeps_existing_routes_and_closed_anchor_contract_without_new_skill_names():
    capture, registration = _registration()
    _, _, legacy, combined = _registrations()
    registration.payload_validator.validate_route(registration.entry.route)
    registration.payload_validator.validate_parameters(_parameters())
    assert registration.skill.definition.skill_id == combined.skill.definition.skill_id
    assert registration.skill.definition.skill_version == "4"
    assert legacy.skill.definition.skill_version == "1" and combined.skill.definition.skill_version == "3"
    assert registration.skill.io_contract.input_artifact_slots == combined.skill.io_contract.input_artifact_slots
    assert not registration.skill.io_contract.output_artifact_slots
    assert capture.skill.definition.skill_version == "2"
    assert registration.entry.route.control_effect_class == "read_only"
    assert registration.entry.route.cancellation_policy == "cancel_only"
    assert not registration.entry.route.may_move_robot
    assert _registration()[1].entry == registration.entry
    # These identities retain the V4 contract from the frozen source candidate.
    assert registration.skill.definition.semantic_content_hash == "b311f981d917aab4ba5969d116e411f909ad6f8b245eae6a064ac13aa888e6ce"
    assert registration.entry.route.semantic_content_hash == "364e787e0bc629390197a6f1064c2c7e66f7666c8a5b48cdf64f85e35a6672fc"


def test_v5_is_explicit_and_binds_its_distinct_sampler_without_a_new_skill_name():
    _, old = _registration()
    _, new = _registration("5")
    assert old.skill.definition.skill_id == new.skill.definition.skill_id
    assert new.skill.definition.skill_version == "5"
    assert new.entry.route.semantic_content_hash != old.entry.route.semantic_content_hash
    assert new.skill.schema_entry.parameter_schema_hash == old.skill.schema_entry.parameter_schema_hash
    assert new.skill.schema_entry.execution_result_schema_hash == old.skill.schema_entry.execution_result_schema_hash
    parameters = _parameters()
    anchor = [.4, -.2, .03]
    answer = _path(parameters, anchor, version="5")
    validate_region_continuation_path(parameters, answer, anchor_xyz_m=anchor, version="5")
    assert answer[POSES_KEY] != _path(parameters, anchor)[POSES_KEY]
    with pytest.raises(ValueError):
        validate_region_continuation_path(parameters, answer, anchor_xyz_m=anchor)


@pytest.mark.parametrize("version", [None, 4, 5, True, "3", "6", "latest"])
def test_region_version_rejects_unknown_or_implicit_values(version):
    with pytest.raises(ValueError, match="explicit '4' or '5'"):
        _registration(version)


def test_v5_executor_refuses_an_old_sampler_port_without_fallback():
    capture, registration = _registration("5")
    with pytest.raises(TypeError, match="must expose query_region_continuation_extrema"):
        region.build_region_continuation_executor(registration=registration,
            capture_registration=capture, query_port=SimpleNamespace(query_region_continuations=lambda: None),
            store=None, lookup_operation=lambda _: None, current_station_revision=lambda: 0,
            clock=lambda: 0., clock_domain_id="clock:test", max_retained_operations=4)


@pytest.mark.parametrize("change", ["samples", "caller_xyz", "caller_tcp", "over_budget", "bool_budget",
    "inverted", "nonfinite", "missing", "frame", "component", "query_budget", "constraint"])
@pytest.mark.parametrize("version", ["4", "5"])
def test_region_rejects_wrong_geometry_and_contract_at_the_advertised_boundary(change, version):
    params = _parameters()
    if change in {"samples", "caller_xyz", "caller_tcp"}:
        params[{"samples": "offsets_xyz_m", "caller_xyz": "anchor_xyz_m",
            "caller_tcp": "current_tcp_pose_xyz_quat_xyzw"}[change]] = []
    elif change in {"over_budget", "bool_budget"}:
        params["max_candidates"] = 9 if change == "over_budget" else True
    elif change == "inverted":
        params["offset_bounds_xyz_m"][0] = [.1, -.1]
    elif change == "nonfinite":
        params["tool_axis_rotation_bounds_rad"][1] = float("inf")
    elif change == "missing":
        del params["max_candidates"]
    elif change in {"frame", "component"}:
        params["reference_frame_id" if change == "frame" else "component_id"] = "wrong"
    elif change == "query_budget":
        params["time_budget_sec"] = 31.
    else:
        params["path_constraint"] = {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": 0}
    with pytest.raises((TypeError, ValueError)):
        _registration(version)[1].payload_validator.validate_parameters(params)


@pytest.mark.parametrize("change", ["bool_bound", "width_overflow", "rotation_width", "rotation_bool",
    "preserve_range", "preserve_turn", "axis_zero", "offset_frame"])
@pytest.mark.parametrize("version", ["4", "5"])
def test_neutral_region_ranges_and_orientation_choices_remain_closed(change, version):
    params = _parameters()
    if change == "bool_bound":
        params["offset_bounds_xyz_m"][0][0] = False
    elif change == "width_overflow":
        params["offset_bounds_xyz_m"][0] = [-1.7e308, 1.7e308]
    elif change == "rotation_width":
        params["tool_axis_rotation_bounds_rad"] = [-4., 4.]
    elif change == "rotation_bool":
        params["tool_axis_rotation_bounds_rad"][0] = True
    elif change.startswith("preserve_"):
        params.update(orientation_mode="preserve", tool_axis=None, axis_reference_xyz=None,
            tool_axis_rotation_bounds_rad=[0., .1] if change == "preserve_range" else [.1, .1])
    elif change == "axis_zero":
        params["axis_reference_xyz"] = [0., 0., 0.]
    else:
        params["offset_frame"] = "camera"
    with pytest.raises((TypeError, ValueError)):
        validate_region_continuation_parameters(params, version=version)


@pytest.mark.parametrize("version", ["4", "5"])
def test_profile_keeps_generated_geometry_overflow_rejection(version):
    params = _parameters()
    params["offset_bounds_xyz_m"] = [[1.7e308, 1.7e308]] * 3
    with pytest.raises(ValueError):
        _registration(version)[1].payload_validator.validate_parameters(params)


def test_result_validation_binds_every_generated_proposal_to_region_and_frozen_input():
    params, anchor = _parameters(), [.4, -.2, .03]
    original = _path(params, anchor)
    validate_region_continuation_path(params, original, anchor_xyz_m=anchor)
    for field in (POSES_KEY, "source_tcp_pose_xyz_quat_xyzw", "candidate_count"):
        changed = deepcopy(original)
        if field == POSES_KEY:
            changed[field][-1][0] += .001
        elif field == "candidate_count":
            changed[field] -= 1
        else:
            # Tool-relative geometry is sensitive to the original measured orientation.
            changed[field][0] += .1
            params = {**params, "offset_frame": "tool"}
            changed = _path(params, anchor)
            changed[field][3:] = [0., 0., 1., 0.]
        with pytest.raises((TypeError, ValueError)):
            validate_region_continuation_path(params, changed, anchor_xyz_m=anchor)


@pytest.mark.parametrize("mode", ["success", "cancel", "deadline", "revision", "client",
    "missing_producer", "changed_candidate", "authority"])
@pytest.mark.parametrize("version", ["4", "5"])
def test_region_executor_reuses_authenticated_anchor_revision_and_cancellation(request, monkeypatch, mode, version):
    capture, registration = _registration(version)
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
    revision, calls = [7], []

    class CallbackExecutor:
        def __init__(self, *, read, **kwargs):
            self.read = read

        def cancel_requested(self, operation_id):
            return mode == "cancel"

    def query(parameters, *, cancel_requested):
        calls.append(deepcopy(parameters))
        if cancel_requested():
            raise RuntimeError("cancelled")
        result = _path(case.request.parameters, anchor["measurement"]["world_xyz_m"], version=version)
        if mode == "revision":
            revision[0] += 1
        elif mode == "changed_candidate":
            result[POSES_KEY][-1][0] += .01
        elif mode == "authority":
            result["execution_authorized"] = True
        return result

    if mode == "client":
        case.request.client_id = "another-client"
    monkeypatch.setattr(shared, "PassiveReadSkillExecutor", CallbackExecutor)
    method = "query_region_continuations" if version == "4" else "query_region_continuation_extrema"
    executor = region.build_region_continuation_executor(registration=registration,
        capture_registration=capture, query_port=SimpleNamespace(**{method: query}),
        store=case.kwargs["store"], lookup_operation=(lambda _: (None, None)) if mode == "missing_producer"
            else case.kwargs["lookup_operation"], current_station_revision=lambda: revision[0],
        clock=lambda: 10., clock_domain_id="clock:test", max_retained_operations=4)
    operation = SimpleNamespace(operation_id="operation:region", deadline_at=9. if mode == "deadline" else 30.)
    if mode != "success":
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            executor.read(case.request, operation)
        assert len(calls) == (0 if mode in {"client", "missing_producer"} else 1)
        return
    answer = executor.read(case.request, operation)
    assert len(calls) == 1 and calls[0]["anchor_xyz_m"] == anchor["measurement"]["world_xyz_m"]
    registration.payload_validator.validate_execution_result_for_parameters(case.request.parameters, answer.execution_result)
    assert answer.reason_code == "region_continuation_query_completed"
    assert not answer.output_artifact_bindings
    assert not any("hold" in ref.lower() for ref in answer.evidence_refs)


@pytest.mark.parametrize("version", ["4", "5"])
def test_region_adapter_generates_candidates_inside_the_existing_frozen_scene_callback(version):
    params, anchor = _parameters(), [.4, .2, .03]
    initial = [.1, .2, .3, 0., 0., 0., 1.]
    captured = []
    owner = object.__new__(MujocoObjectGoalQuery)

    def query(parameters, *, cancel_requested, candidates_for):
        captured.append(parameters)
        assert not cancel_requested()
        poses = candidates_for(initial)
        result = _result(explicit_continuation_parameters(params, poses))
        return {**result, POSES_KEY: poses}

    owner._query_continuations = query
    query = owner.query_region_continuations if version == "4" else owner.query_region_continuation_extrema
    result = query({**params, "anchor_xyz_m": anchor}, cancel_requested=lambda: False)
    assert len(captured) == 1 and captured[0][POSES_KEY] == []
    assert result[POSES_KEY] == region_candidate_poses(params, initial=initial, anchor_xyz_m=anchor, version=version)
    assert result["motion_executed"] is result["execution_authorized"] is False
