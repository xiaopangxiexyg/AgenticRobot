from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.back_projection import (
    LiveAlignedDepthCapture,
    LiveBackProjectionProvider,
    build_back_projection_registration,
)
from agent_gateway.profiles.mujoco.object_goal_query import (
    build_object_goal_query_registration,
)
from agent_gateway.profiles.spatial_goal import build_spatial_goal_registration
from agent_gateway.profiles.surface_anchor import (
    _json_bytes,
    build_surface_anchor_registrations,
    resolve_surface_anchor,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.skills.perception.object_goal_query import RANKING, RELATION_SKILL_ID
from agent_gateway.skills.perception.surface_anchor import (
    CAPTURE_SKILL_ID,
    GOAL_SKILL_ID,
    QUERY_SKILL_ID,
)
from dataset_contracts.perception_artifacts import SURFACE_ANCHOR_SCHEMA_ID_V1


def _projection():
    return build_back_projection_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 4,
                "native_height": 3,
            },
        ),
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame:test",
        frame_graph_hash="graph:test",
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        target_binding_id="target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )


class Capture:
    calls = 0

    def capture_aligned_depth(self, *, role, expected_station_revision):
        self.calls += 1
        return LiveAlignedDepthCapture(
            source_id="front",
            role=role,
            station_revision=expected_station_revision,
            captured_at=10.0,
            capture_group_id="capture:test",
            depth_content_hash="a" * 64,
            intrinsics={
                "width": 4,
                "height": 3,
                "fx": 2.0,
                "fy": 2.0,
                "cx": 2.0,
                "cy": 1.0,
            },
            depth_m=np.full((3, 4), 2.0, dtype=np.float32),
            reference_from_camera=np.eye(4),
        )


def _registrations(*, relation=False):
    projection = _projection()
    query = build_object_goal_query_registration(
        base_route=projection.skill.routes[0],
        base_binding=projection.binding,
        reference_frame_id="canonical_station",
        planning_profile_hash="profile:test",
    )
    registrations = build_surface_anchor_registrations(
        back_projection=projection, object_query=query, enable_relation_queries=relation
    )
    return projection, *registrations


class _AnchorSegmenter:
    def __init__(self, mask=None):
        self.model_spec = SimpleNamespace(model_profile_id="sam2:anchor-test",
            semantic_content_hash="model:anchor-test")
        self.semantic_content_hash = "implementation:anchor-test"
        self.started = False
        self.calls = 0
        self.mask = mask
        self.on_segment = None

    def start(self):
        self.started = True

    def close(self):
        self.started = False

    def segment(self, image, *, language_prompt, pixel_uv, min_score):
        from deployments.perception.sam2_grounding_dino import GroundedObjectSegment

        assert self.started
        self.calls += 1
        if self.on_segment:
            self.on_segment()
        mask = self.mask
        if mask is None:
            mask = np.zeros((image.height, image.width), dtype=bool)
            if pixel_uv is None:
                mask[1:-1, 1:-1] = True
            else:
                u, v = pixel_uv
                mask[max(0, v - 3):v + 4, max(0, u - 3):u + 4] = True
        rows, cols = np.nonzero(mask)
        return GroundedObjectSegment(mask=mask, score=1.0, label=language_prompt,
            box_xyxy=(int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)))

    @staticmethod
    def encode_segment_artifacts(image, segment):
        from deployments.perception.sam2_grounding_dino import (
            encode_grounded_object_segment_artifacts,
        )
        return encode_grounded_object_segment_artifacts(image, segment)


def _v2_registrations(*, selection_policy="prompt_nearest"):
    from agent_gateway.profiles.segmentation import build_segmentation_registration

    projection, _, _ = _registrations()
    model = _AnchorSegmenter()
    segmentation = build_segmentation_registration(target_kind="sim", current_mode="sim_active",
        source_id_by_role={"main": "front"}, model_profile_id=model.model_spec.model_profile_id,
        model_profile_hash=model.model_spec.semantic_content_hash,
        inference_implementation_hash=model.semantic_content_hash,
        runtime_session_id="session:test", control_domain_id="domain:test",
        target_binding_id="target:test", station_manifest_hash="station:test", execution_manifest_hash="execution:test")
    query = build_object_goal_query_registration(base_route=projection.skill.routes[0],
        base_binding=projection.binding, reference_frame_id="canonical_station", planning_profile_hash="profile:test")
    return projection, segmentation, model, build_surface_anchor_registrations(
        back_projection=projection, object_query=query, version="2", segmentation=segmentation,
        enable_relation_queries=True, selection_policy=selection_policy)


def _v2_parameters(mode="mask_interior"):
    return {"role": "main", "selection_mode": mode, "target_pixel_uv": [0, 0],
            "language_prompt": None, "min_score": 0.25}


@pytest.mark.parametrize("kind", ["relation", "goal"])
def test_anchor_executor_freezes_caller_owned_producer_list(monkeypatch, kind):
    from agent_gateway.profiles import surface_anchor

    projection, _, _, (capture, _query, relation) = _v2_registrations()
    extras = [object()]
    expected = tuple(extras)
    checked, resolved = [], []

    class CallbackExecutor:
        def __init__(self, *, read, **_kwargs):
            self.read = read

    def check(_registration, _capture, additional):
        assert isinstance(additional, tuple)
        checked.append(additional)

    def resolve(**kwargs):
        resolved.append(kwargs["accepted_capture_registrations"])
        return {"measurement": {"world_xyz_m": [0., 0., 2.]},
                "source_operation_id": "operation:anchor"}, "anchor:test"

    monkeypatch.setattr(surface_anchor, "PassiveReadSkillExecutor", CallbackExecutor)
    monkeypatch.setattr(surface_anchor, "_require_anchor_producer_set", check)
    monkeypatch.setattr(surface_anchor, "resolve_surface_anchor", resolve)
    common = dict(capture_registration=capture, store=None, lookup_operation=None,
        current_station_revision=lambda: 7, clock=lambda: 10., clock_domain_id="clock:test",
        max_retained_operations=3, accepted_capture_registrations=extras)
    if kind == "relation":
        executor = surface_anchor.build_anchored_query_executor(registration=relation,
            query_port=SimpleNamespace(query_relations=lambda *_args, **_kwargs: {}), **common)
        parameters = _relation_parameters()
    else:
        spatial = build_spatial_goal_registration(base_route=projection.skill.routes[0],
            base_binding=projection.binding, component_ids=("right_arm",), reference_frame_id="canonical_station")
        registration = surface_anchor._build_anchored_goal_registration(capture=capture,
            spatial_goal=spatial, reference_frame_id="canonical_station")
        monkeypatch.setattr(surface_anchor, "calculate_spatial_goal", lambda _parameters: {})
        executor = surface_anchor.build_anchored_goal_executor(registration=registration, **common)
        parameters = _goal_parameters()
    extras[:] = [object(), object()]
    executor.read(SimpleNamespace(parameters=parameters, expected_station_revision=7),
                  SimpleNamespace(operation_id="operation:test", deadline_at=20.))
    assert checked == resolved == [expected]


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
def test_native_paired_rgbd_segment_anchor_and_v2_consumers_share_one_cut(tmp_path):
    import json

    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig, build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from sim.mujoco.deploy import MujocoSimHILTargetConfig

    root = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = _AnchorSegmenter()
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=root / "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json",
            realtime_worker=False, allow_scripted_task_reset=False, exposed_pose_sides=("right",),
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1()),
        allowed_client_ids=("test:paired-native",),
        artifact_storage=MujocoAgentArtifactStorageConfig(root=artifacts),
        segmentation_localizer=model, enable_safe_tcp_pose=True,
        enable_object_goal_queries=True, enable_surface_anchors=True, enable_surface_anchor_v2=True,
        enable_segmentation_geometry_v3=True, enable_rgbd_capture_v3=True, enable_surface_anchor_v3=True,
        enable_relation_goal_queries=True, enable_spatial_goals=True, safe_tcp_motion_profile="object_relation")
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        snapshot = session.observation_producer.capture_and_publish(session.service).snapshot
        assert snapshot is not None
        scene = session.application.scene
        before = {name: getattr(scene.data, name).copy() for name in ("qpos", "qvel", "ctrl", "eq_active")}
        before_time, before_eq = scene.data.time, scene.model.eq_data.copy()
        port = session.back_projection_provider.capture_port
        capture = port.capture_placement_rgbd
        captures = []

        def counted(**kwargs):
            captures.append(kwargs)
            return capture(**kwargs)

        port.capture_placement_rgbd = counted
        port.capture_aligned_depth = lambda **_: pytest.fail("Implicit fresh depth capture")
        sequence = 0
        executors = {(item.skill_id, item.skill_version): item.execution_port
            for item in session.service.executor_registry.bindings}

        def submit(skill, version, parameters, inputs=()):
            nonlocal sequence
            sequence += 1
            entry = session.contract_bundle.route_manifest.route(skill, version)
            route, profile = entry.route, entry.route.allowed_parameter_profiles[0]
            request = AgentSkillRequest(client_id="test:paired-native", request_id=f"request:{sequence}",
                idempotency_key=f"idempotency:{sequence}", client_sequence=sequence,
                runtime_session_id=session.contract_bundle.route_manifest.runtime_session_id,
                skill_route_manifest_hash=session.contract_bundle.route_manifest.semantic_content_hash,
                expected_station_revision=snapshot.station_revision, observation_snapshot_id=snapshot.snapshot_id,
                skill_id=skill, skill_version=version, skill_definition_hash=route.skill_definition_hash,
                route_id=route.route_id, route_version=route.route_version, route_hash=route.semantic_content_hash,
                deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
                parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
                parameters=parameters, requested_mode="sim_active", requested_timeout_ms=route.max_operation_duration_ms,
                input_artifact_bindings=inputs)
            admitted = session.service.submit_request(request=request, current_station_revision=snapshot.station_revision)
            assert admitted.accepted, admitted
            assert session.service.launch_operation(admitted.operation.operation_id).accepted
            executor = executors[skill, version]
            assert executor.wait(admitted.operation.operation_id, timeout_sec=40.)
            outcome = session.service.get_operation(admitted.operation.operation_id)
            assert outcome.execution_outcome == "completed", executor.failure_diagnostics
            return outcome

        def output(operation, source_slot, target_slot):
            matching = [item for item in operation.output_artifact_bindings if item.slot_id == source_slot]
            assert len(matching) == 1
            return replace(matching[0], slot_id=target_slot, ordinal=0, production_receipt_ref=None)

        captured = submit("perception.capture_rgb", "3", {"selected_roles": ["main"],
            "width": 640, "height": 480, "max_total_bytes": 4194304, "include_depth": True})
        paired = output(captured, "paired_observations", "paired_observation")
        metadata = json.loads(session.artifact_store.managed_read_path(paired.artifact.artifact_id).read_bytes())
        segmented = submit("perception.segment", "3", {"role": "main",
            "rgb_capture_operation_id": captured.operation_id,
            "targets": [{"target_pixel_uv": [326, 280], "language_prompt": None, "min_score": .25}]},
            (output(captured, "rgb_frames", "input_image"),))
        anchors = []
        for mode in ("exact", "mask_interior"):
            anchored = submit(CAPTURE_SKILL_ID, "3", {"role": "main", "selection_mode": mode,
                "target_pixel_uv": [326, 280] if mode == "exact" else None,
                "rgb_capture_operation_id": captured.operation_id,
                "segmentation_operation_id": segmented.operation_id if mode == "mask_interior" else None,
                "target_index": 0 if mode == "mask_interior" else None},
                (paired,) if mode == "exact" else (paired, output(segmented, "segmentation_mask", "segmentation_mask")))
            value = anchored.execution_result
            assert value["captured_at"] == metadata["captured_at"]
            assert value["measurement"]["capture_group_id"] == metadata["capture_group_id"]
            assert value["measurement"]["depth_content_hash"] == metadata["depth_content_hash"]
            assert value["selection"]["input_rgb_content_hash"] == metadata["rgb_artifact"]["sha256"]
            assert value["source_operation_id"] == anchored.operation_id
            anchors.append(anchored)
        last_anchor = anchors[-1]
        goal = submit(GOAL_SKILL_ID, "2", {**_goal_parameters(),
            "reference_frame_id": session.contract_bundle.motion_profile.reference_frame_id},
            (output(last_anchor, "surface_anchor", "surface_anchor"),))
        assert goal.execution_result["surface_anchor"] == last_anchor.execution_result
        query = submit(RELATION_SKILL_ID, "2", {**_relation_parameters(), "time_budget_sec": 1.,
            "reference_frame_id": session.contract_bundle.motion_profile.reference_frame_id},
            (output(last_anchor, "surface_anchor", "surface_anchor"),))
        assert query.execution_result["surface_anchor"] == last_anchor.execution_result
        assert query.execution_result["path_query"]["motion_executed"] is False
        assert len(captures) == model.calls == 1
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(scene.data, name), value)
        assert scene.data.time == before_time
        np.testing.assert_array_equal(scene.model.eq_data, before_eq)
        assert not session.application.target.planned_tcp_motion.active
    finally:
        outcome = session.close()
        assert outcome.accepted


def _relation_parameters():
    return {"component_id": "right_arm", "reference_frame_id": "canonical_station",
            "offsets_xyz_m": [[.04, -.03, .14]], "offset_frame": "reference",
            "orientation_mode": "align_axis", "tool_axis": "z", "axis_reference_xyz": [0, 0, -1],
            "tool_axis_rotation_offsets_rad": [0., 1.], "time_budget_sec": 5.}


@pytest.mark.parametrize("mutation", ["xyz", "tcp", "component", "frame", "budget", "count"])
def test_relation_query_registration_is_opt_in_and_validates_exact_contract(mutation):
    _, capture, legacy, relation = _registrations(relation=True)
    assert relation.skill.definition.skill_id == RELATION_SKILL_ID
    assert len(_registrations()) == 3
    assert len({item.entry.route.route_id for item in (capture, legacy, relation)}) == 3
    assert relation.entry.route.control_effect_class == legacy.entry.route.control_effect_class
    parameters = _relation_parameters()
    relation.payload_validator.validate_parameters(parameters)
    if mutation == "xyz":
        parameters["anchor_xyz_m"] = [0, 0, 0]
    elif mutation == "tcp":
        parameters["current_tcp_pose_xyz_quat_xyzw"] = [0, 0, 0, 0, 0, 0, 1]
    elif mutation == "component":
        parameters["component_id"] = "left_arm"
    elif mutation == "frame":
        parameters["reference_frame_id"] = "camera"
    elif mutation == "budget":
        parameters["time_budget_sec"] = True
    else:
        parameters["offsets_xyz_m"] *= 33
    with pytest.raises(ValueError):
        relation.payload_validator.validate_parameters(parameters)


def test_relation_query_authenticates_anchor_and_exact_selected_endpoint(anchor_case, monkeypatch):
    from agent_gateway.profiles import surface_anchor
    from robot_motion.spatial_goals import generate_relation_goals

    class CallbackExecutor:
        def __init__(self, *, read, **_kwargs):
            self.read = read

        def cancel_requested(self, _operation_id):
            return False

    monkeypatch.setattr(surface_anchor, "PassiveReadSkillExecutor", CallbackExecutor)
    _, capture, _, relation = _registrations(relation=True)
    params = _relation_parameters()
    anchor_case.request.parameters = params
    source_tcp = [0, 0, 0, 0, 0, 0, 1]
    planned = []

    def query(value, *, cancel_requested):
        if cancel_requested():
            raise RuntimeError("Relation query cancelled before planning.")
        planned.append(value)
        assert value["anchor_xyz_m"] == [0., 0., 2.]
        candidates = generate_relation_goals(current_tcp_pose_xyz_quat_xyzw=source_tcp,
            **{key: val for key, val in value.items()
               if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})
        return {"component_id": "right_arm", "reference_frame_id": "canonical_station",
            "anchor_xyz_m": value["anchor_xyz_m"], "source_tcp_pose_xyz_quat_xyzw": source_tcp,
            "query_status": "path_found", "scene_hash": "a" * 64, "source_qpos_hash": "b" * 64,
            "source_sim_time_sec": 0., "pose_targets_by_component": {"right_arm": list(candidates[0].goal.pose_xyz_quat_xyzw)},
            "candidate_count": 2, "checked_count": 1, "all_candidates_checked": False, "selected_index": 0,
            "candidate_diagnostics": [{"candidate_index": 0, "reason_code": "path_found", "wall_sec": .1,
                "ranking_cost": [0., 1., 0.], "trajectory_hash": "c" * 64,
                "pose_target_xyz_quat_xyzw": list(candidates[0].goal.pose_xyz_quat_xyzw)}],
            "planning_wall_sec": .1, "ranking_criteria": RANKING, "motion_executed": False,
            "execution_authorized": False, "input_source": "caller_anchor_and_frozen_measured_scene",
            "constraint_scope": "frozen_scene_full_path"}

    revision = [7]
    executor = surface_anchor.build_anchored_query_executor(registration=relation,
        capture_registration=capture, query_port=SimpleNamespace(query_relations=query),
        store=anchor_case.kwargs["store"], lookup_operation=anchor_case.kwargs["lookup_operation"],
        current_station_revision=lambda: revision[0], clock=lambda: 10., clock_domain_id="clock:test",
        max_retained_operations=2)
    operation = SimpleNamespace(operation_id="operation:query", deadline_at=20.)
    answer = executor.read(anchor_case.request, operation).execution_result
    relation.payload_validator.validate_execution_result_for_parameters(params, answer)
    # The diagnostic extension is optional for existing providers, not a new
    # execution or feasibility claim. Bind nonwinning proposals to their index.
    legacy_answer = deepcopy(answer)
    legacy_answer["path_query"]["candidate_diagnostics"][0].pop("pose_target_xyz_quat_xyzw")
    relation.payload_validator.validate_execution_result_for_parameters(params, legacy_answer)
    expanded = deepcopy(answer)
    path = expanded["path_query"]
    candidates = generate_relation_goals(current_tcp_pose_xyz_quat_xyzw=source_tcp,
        anchor_xyz_m=path["anchor_xyz_m"], **{key: params[key] for key in params
            if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})
    path.update(checked_count=2, all_candidates_checked=True)
    path["candidate_diagnostics"].append({"candidate_index": 1,
        "reason_code": "candidate_rank_bound_pruned", "wall_sec": 0.,
        "pose_target_xyz_quat_xyzw": list(candidates[1].goal.pose_xyz_quat_xyzw)})
    relation.payload_validator.validate_execution_result_for_parameters(params, expanded)
    path["candidate_diagnostics"][1]["pose_target_xyz_quat_xyzw"][0] += .01
    with pytest.raises(ValueError, match="Diagnostic endpoint differs"):
        relation.payload_validator.validate_execution_result_for_parameters(params, expanded)
    assert answer["relation_constraint_scope"] == "endpoint_only"
    assert answer["surface_anchor"] == anchor_case.payload
    forged = {**legacy_answer, "path_query": {**legacy_answer["path_query"],
        "pose_targets_by_component": {"right_arm": [1, 2, 3, 0, 0, 0, 1]}}}
    with pytest.raises(ValueError, match="differs from its relation"):
        relation.payload_validator.validate_execution_result_for_parameters(params, forged)
    with pytest.raises(ValueError, match="endpoint-only"):
        relation.payload_validator.validate_execution_result({**answer, "relation_constraint_scope": "full_path"})
    anchor_case.source_request.client_id = "other"
    with pytest.raises(ValueError, match="authenticated"):
        executor.read(anchor_case.request, operation)
    assert len(planned) == 1
    anchor_case.source_request.client_id = "client:test"
    executor.cancel_requested = lambda _operation_id: True
    with pytest.raises(RuntimeError, match="cancel"):
        executor.read(anchor_case.request, operation)
    assert len(planned) == 1


def _goal_registration():
    projection, _, _ = _registrations()
    spatial = build_spatial_goal_registration(
        base_route=projection.skill.routes[0],
        base_binding=projection.binding,
        component_ids=("left_arm", "right_arm"),
        reference_frame_id="canonical_station",
    )
    query = build_object_goal_query_registration(
        base_route=projection.skill.routes[0],
        base_binding=projection.binding,
        reference_frame_id="canonical_station",
        planning_profile_hash="profile:test",
    )
    _, _, goal = build_surface_anchor_registrations(
        back_projection=projection,
        object_query=query,
        spatial_goal=spatial,
    )
    return goal


def _goal_parameters():
    return {
        "component_id": "right_arm",
        "reference_frame_id": "canonical_station",
        "current_tcp_pose_xyz_quat_xyzw": [0, 0, 0, 0, 0, 0, 1],
        "offset_xyz_m": [0.04, -0.03, 0.14],
        "offset_frame": "reference",
        "orientation_mode": "align_axis",
        "tool_axis": "z",
        "axis_reference_xyz": [0, 0, -1],
    }


@pytest.mark.parametrize("component", ["left_arm", "right_arm"])
def test_anchored_relation_reuses_canonical_geometry_and_rejects_xyz_forgery(
    anchor_case, monkeypatch, component
):
    from agent_gateway.profiles import surface_anchor

    class CallbackExecutor:
        def __init__(self, *, read, **_kwargs):
            self.read = read

    monkeypatch.setattr(surface_anchor, "PassiveReadSkillExecutor", CallbackExecutor)
    goal = _goal_registration()
    params = {**_goal_parameters(), "component_id": component}
    anchor_case.request.parameters = params
    revision = [7]
    executor = surface_anchor.build_anchored_goal_executor(
        registration=goal,
        clock=lambda: 10.0,
        max_retained_operations=2,
        **{
            key: value
            for key, value in anchor_case.kwargs.items()
            if key not in {"request", "current_station_revision"}
        },
        current_station_revision=lambda: revision[0],
    )
    answer = executor.read(
        anchor_case.request, SimpleNamespace(operation_id="operation:goal")
    )
    result = answer.execution_result
    goal.payload_validator.validate_execution_result_for_parameters(params, result)
    assert result["surface_anchor"] == anchor_case.payload
    assert (
        result["anchor_artifact_id"]
        == anchor_case.request.input_artifact_bindings[0].artifact.artifact_id
    )
    proposal = result["spatial_goal"]
    np.testing.assert_allclose(
        proposal["pose_targets_by_component"][component][:3],
        [0.04, -0.03, 2.14],
        atol=1e-12,
    )
    assert proposal["axis_alignment_error_rad"] < 1e-12
    assert (
        proposal["constraint_scope"] == "endpoint_only"
        and proposal["path_checked"] is False
    )
    assert (
        proposal["motion_executed"] is False
        and proposal["input_source"] == "caller_supplied_geometry"
    )
    for invalid in (
        {**params, "anchor_xyz_m": [1, 2, 3]},
        {**params, "reference_frame_id": "camera"},
        {**params, "offset_xyz_m": [float("nan"), 0, 0]},
    ):
        with pytest.raises(ValueError):
            goal.payload_validator.validate_parameters(invalid)
    forged = {
        **result,
        "spatial_goal": {
            **proposal,
            "pose_targets_by_component": {component: [1, 2, 3, 0, 0, 0, 1]},
        },
    }
    with pytest.raises(ValueError, match="differs from its relation"):
        goal.payload_validator.validate_execution_result_for_parameters(params, forged)
    anchor_case.source_request.client_id = "other"
    with pytest.raises(ValueError, match="authenticated"):
        executor.read(
            anchor_case.request, SimpleNamespace(operation_id="operation:forged-goal")
        )
    anchor_case.source_request.client_id = "client:test"
    calculate = surface_anchor.calculate_spatial_goal

    def calculate_after_revision_change(parameters):
        result = calculate(parameters)
        revision[0] = 8
        return result

    monkeypatch.setattr(
        surface_anchor, "calculate_spatial_goal", calculate_after_revision_change
    )
    with pytest.raises(RuntimeError, match="revision changed during anchored goal"):
        executor.read(
            anchor_case.request, SimpleNamespace(operation_id="operation:revision-goal")
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"role": "main", "target_pixel_uv": None},
        {"role": "main", "target_pixel_uv": [2, 1], "pixel_region_xyxy": [0, 0, 4, 3]},
        {"role": "other", "target_pixel_uv": [2, 1]},
        {"role": "main", "target_pixel_uv": [True, 1]},
    ],
)
def test_anchor_parameters_never_accept_region_or_ambiguous_pixel(parameters):
    _, capture, _ = _registrations()
    with pytest.raises((TypeError, ValueError)):
        capture.payload_validator.validate_parameters(parameters)


def test_anchored_query_rejects_manual_xyz_and_wrong_frame():
    _, _, query = _registrations()
    value = {
        "component_id": "right_arm",
        "reference_frame_id": "canonical_station",
        "heights_m": [0.1],
        "tool_yaw_offsets_rad": [0.0],
        "time_budget_sec": 1.0,
    }
    query.payload_validator.validate_parameters(value)
    for changed in (
        {**value, "anchor_xyz_m": [0, 0, 0]},
        {**value, "reference_frame_id": "camera"},
    ):
        with pytest.raises(ValueError):
            query.payload_validator.validate_parameters(changed)


def test_projection_returns_exact_capture_without_second_read():
    projection, _, _ = _registrations()
    port = Capture()
    provider = LiveBackProjectionProvider(capture_port=port, registration=projection)
    value = {
        "role": "main",
        "target_pixel_uv": [2, 1],
        "pixel_region_xyxy": None,
        "world_z_min_m": None,
        "world_z_max_m": None,
    }
    result, capture = provider.project_parameters(value, expected_station_revision=7)
    assert port.calls == 1 and capture.captured_at == 10.0
    assert (
        result["world_xyz_m"] == [0.0, 0.0, 2.0]
        and result["coordinate_mode"] == "pixel"
    )
    assert (
        provider.project_request(
            SimpleNamespace(parameters=value, expected_station_revision=7)
        )
        == result
    )
    assert port.calls == 2


@pytest.mark.parametrize("mode", ["exact", "mask_interior", "text"])
@pytest.mark.parametrize("fault", [None, "timestamp", "revision"])
@pytest.mark.parametrize("selection_policy", ["prompt_nearest", "max_clearance_medoid"])
def test_v2_capture_uses_one_correlated_cut_and_retains_actual_selected_pixel(
        tmp_path, monkeypatch, mode, fault, selection_policy):
    from PIL import Image

    from agent_gateway.profiles import surface_anchor
    from agent_gateway.profiles.back_projection import CorrelatedPlacementCapture
    from agent_gateway.profiles.segmentation import (
        LiveRgbSegmentationCapture,
        LiveSegmentationProvider,
    )
    from dataset_contracts.perception_artifacts import SURFACE_ANCHOR_SCHEMA_ID_V2

    projection, segmentation, model, (capture_registration, _query, _relation) = _v2_registrations(
        selection_policy=selection_policy)
    class CallbackExecutor:
        def __init__(self, *, read, **_kwargs):
            self.read = read
    monkeypatch.setattr(surface_anchor, "PassiveReadSkillExecutor", CallbackExecutor)
    revision = [7]
    class Port(Capture):
        calls = 0
        direct_calls = 0
        def capture_aligned_depth(self, **_kwargs):
            self.direct_calls += 1
            raise AssertionError("v2 must not recapture depth")
        def capture_placement_rgbd(self, *, role, expected_station_revision):
            self.calls += 1
            depth = Capture().capture_aligned_depth(role=role, expected_station_revision=expected_station_revision)
            depth_m = depth.depth_m.copy()
            depth_m[1, 2] = 0.0
            depth = replace(depth, depth_m=depth_m,
                reference_from_camera=np.array([[0., -1., 0., .5], [1., 0., 0., -.5],
                    [0., 0., 1., .1], [0., 0., 0., 1.]]))
            if fault == "revision":
                revision[0] = 8
            rgb = LiveRgbSegmentationCapture(source_id="front", role=role,
                captured_at=depth.captured_at + (1 if fault == "timestamp" else 0),
                station_revision=expected_station_revision, image=Image.new("RGB", (4, 3)), rgb_content_hash="b" * 64)
            return CorrelatedPlacementCapture(rgb=rgb, depth=depth)
    port = Port()
    provider = LiveBackProjectionProvider(capture_port=port, registration=projection)
    segmentation_provider = LiveSegmentationProvider(localizer=model, capture_port=SimpleNamespace(capture_segment_rgb=lambda **_: None),
        registration=segmentation)
    model.start()
    store = RecoverableAgentArtifactStore(root=tmp_path / "v2", quota_bytes=65536, max_artifact_bytes=16384)
    params = _v2_parameters("exact" if mode == "exact" else "mask_interior")
    if mode == "text":
        params.update(target_pixel_uv=None, language_prompt="blue block")
    request = SimpleNamespace(parameters=params, expected_station_revision=7,
        runtime_session_id="session:test", semantic_content_hash="a" * 64)
    executor = surface_anchor.build_surface_anchor_capture_executor(registration=capture_registration,
        provider=provider, segmentation_provider=segmentation_provider, store=store,
        current_station_revision=lambda: revision[0], clock=lambda: 10., clock_domain_id="clock:test", max_retained_operations=2)
    if fault:
        with pytest.raises(RuntimeError, match="capture time" if fault == "timestamp" else "Station revision"):
            executor.read(request, SimpleNamespace(operation_id="operation:v2"))
        assert port.calls == 1 and port.direct_calls == 0
        return
    answer = executor.read(request, SimpleNamespace(operation_id="operation:v2"))
    value = answer.execution_result
    capture_registration.payload_validator.validate_execution_result_for_parameters(params, value)
    assert port.calls == 1 and port.direct_calls == 0
    assert model.calls == (0 if mode == "exact" else 1)
    u, v = value["target_pixel_uv"]
    assert [u, v] != [2, 1]
    assert value["measurement"]["world_xyz_m"] == pytest.approx([.5 - (v - 1), -.5 + (u - 2), 2.1])
    assert value["selection"]["requested_pixel_uv"] == params["target_pixel_uv"]
    assert value["selection"]["input_rgb_content_hash"] == "b" * 64
    outputs = {binding.slot_id: binding for binding in answer.output_artifact_bindings}
    assert outputs["surface_anchor"].artifact.schema_id == SURFACE_ANCHOR_SCHEMA_ID_V2
    assert len(answer.output_artifact_bindings) == (1 if mode == "exact" else 3)
    if mode != "exact":
        from agent_gateway.skills.perception.surface_anchor_v2 import (
            INTERIOR_SELECTION_POLICIES,
        )

        assert value["selection"]["algorithm_id"] == INTERIOR_SELECTION_POLICIES[selection_policy]
        assert value["mask_geometry"]["valid_point_count"] > 0
        assert value["selection"]["mask_media_content_hash"] == outputs["segmentation_mask"].artifact.sha256
        assert value["selection"]["overlay_media_content_hash"] == outputs["segmentation_overlay"].artifact.sha256
        other_policy = "max_clearance_medoid" if selection_policy == "prompt_nearest" else "prompt_nearest"
        other_capture = _v2_registrations(selection_policy=other_policy)[3][0]
        with pytest.raises(ValueError, match="Mask interior selection"):
            other_capture.payload_validator.validate_execution_result(value)


@pytest.mark.parametrize("bad", ["both", "neither", "text_exact", "unknown", "extra", "boolean"])
def test_v2_parameters_require_explicit_unambiguous_selection(bad):
    _, _, _, (capture, _, _) = _v2_registrations()
    value = _v2_parameters()
    if bad == "both":
        value["language_prompt"] = "block"
    elif bad == "neither":
        value["target_pixel_uv"] = None
    elif bad == "text_exact":
        value.update(selection_mode="exact", target_pixel_uv=None, language_prompt="block")
    elif bad == "unknown":
        value["selection_mode"] = "auto"
    elif bad == "extra":
        value["world_xyz_m"] = [0, 0, 0]
    else:
        value["target_pixel_uv"] = [True, 0]
    with pytest.raises((ValueError, TypeError)):
        capture.payload_validator.validate_parameters(value)


@pytest.mark.parametrize("mismatch", ["profile", "binding"])
def test_v2_capture_rejects_wrong_runtime_provider_before_reading(mismatch):
    from agent_gateway.profiles.surface_anchor import (
        build_surface_anchor_capture_executor,
    )

    projection, segmentation, _model, (capture, _, _) = _v2_registrations()
    port = Capture()
    wrong_segmentation = SimpleNamespace(registration=SimpleNamespace(
        binding=(replace(segmentation.binding, runtime_session_id="another-session")
            if mismatch == "binding" else segmentation.binding),
        profile=segmentation.profile if mismatch == "binding" else SimpleNamespace(semantic_content_hash="another-profile")))
    with pytest.raises(ValueError, match="published"):
        build_surface_anchor_capture_executor(registration=capture,
            provider=LiveBackProjectionProvider(capture_port=port, registration=projection),
            segmentation_provider=wrong_segmentation, store=None, current_station_revision=lambda: 0,
            clock=lambda: 0., clock_domain_id="test", max_retained_operations=2)
    assert port.calls == 0


@pytest.mark.parametrize("selection_policy", ["prompt_nearest", "max_clearance_medoid"])
def test_v2_mask_selection_handles_concavity_border_holes_ties_and_prompt_component(selection_policy):
    from agent_gateway.profiles.surface_anchor_selection import (
        select_mask_interior_pixel as select,
    )
    from agent_gateway.skills.perception.surface_anchor_v2 import (
        INTERIOR_SELECTION_POLICIES,
    )

    def select_mask_interior_pixel(*args, **kwargs):
        return select(*args, **kwargs,
            interior_algorithm=INTERIOR_SELECTION_POLICIES[selection_policy])

    mask = np.zeros((9, 14), dtype=bool)
    mask[:5, :5] = True
    mask[2:8, 8:14] = True
    mask[2:5, 2:5] = False
    depth = np.full(mask.shape, 1.)
    depth[1, 1] = 0
    point, distance = select_mask_interior_pixel(mask, depth,
        requested_pixel_uv=[0, 0], maximum_depth_m=10.)
    assert point == ([0, 0] if selection_policy == "prompt_nearest" else [2, 1]) and distance == 1.
    assert mask[point[1], point[0]]
    point, distance = select_mask_interior_pixel(np.ones((3, 4), bool), np.ones((3, 4)),
        requested_pixel_uv=None, maximum_depth_m=10.)
    assert point == [1, 1] and distance == 2.
    with pytest.raises(RuntimeError, match="no_valid_depth"):
        select_mask_interior_pixel(mask, np.zeros(mask.shape), requested_pixel_uv=None, maximum_depth_m=10.)
    with pytest.raises(RuntimeError, match="prompt_not_in_mask"):
        select_mask_interior_pixel(mask, depth, requested_pixel_uv=[6, 0], maximum_depth_m=10.)


def test_medoid_policy_is_click_independent_within_one_component_but_never_changes_component():
    from agent_gateway.profiles.surface_anchor_selection import (
        select_mask_interior_pixel,
    )
    from agent_gateway.skills.perception.surface_anchor_v2 import (
        MAX_CLEARANCE_MEDOID_ALGORITHM,
    )

    mask = np.zeros((12, 25), dtype=bool)
    mask[1:6, 1:12] = True
    mask[6:11, 18:23] = True
    depth = np.ones(mask.shape)
    outputs = {
        tuple(select_mask_interior_pixel(mask, depth, requested_pixel_uv=[u, v],
            maximum_depth_m=10., interior_algorithm=MAX_CLEARANCE_MEDOID_ALGORITHM)[0])
        for v, u in np.argwhere(mask[:, :12])
    }
    assert outputs == {(6, 3)}
    assert select_mask_interior_pixel(mask, depth, requested_pixel_uv=[18, 6],
        maximum_depth_m=10., interior_algorithm=MAX_CLEARANCE_MEDOID_ALGORITHM) == ([20, 8], 3.)
    depth[3, 6] = np.nan
    selected, distance = select_mask_interior_pixel(mask, depth, requested_pixel_uv=[1, 1],
        maximum_depth_m=10., interior_algorithm=MAX_CLEARANCE_MEDOID_ALGORITHM)
    assert selected == [5, 3] and distance == 3.
    assert np.isfinite(depth[selected[1], selected[0]])


def test_medoid_policy_does_not_move_on_a_single_pixel_mask_perturbation_with_same_maxima():
    from agent_gateway.profiles.surface_anchor_selection import (
        select_mask_interior_pixel,
    )
    from agent_gateway.skills.perception.surface_anchor_v2 import (
        MAX_CLEARANCE_MEDOID_ALGORITHM,
    )

    mask = np.zeros((12, 18), dtype=bool)
    mask[1:10, 1:16] = True
    changed = mask.copy()
    changed[1, 2] = False
    depth = np.ones(mask.shape)
    points = [select_mask_interior_pixel(value, depth, requested_pixel_uv=[2, 2],
        maximum_depth_m=10., interior_algorithm=MAX_CLEARANCE_MEDOID_ALGORITHM)
        for value in (mask, changed)]
    assert points == [([8, 5], 5.), ([8, 5], 5.)]


@pytest.mark.parametrize("shape", ["rectangle", "concave", "depth_holes", "diagonal"])
def test_medoid_selection_matches_independent_brute_force_l1_objective(shape):
    from scipy.ndimage import distance_transform_edt

    from agent_gateway.profiles.surface_anchor_selection import (
        select_mask_interior_pixel,
    )
    from agent_gateway.skills.perception.surface_anchor_v2 import (
        MAX_CLEARANCE_MEDOID_ALGORITHM,
    )

    mask = np.zeros((9, 15), dtype=bool)
    mask[1:8, 1:14] = True
    depth = np.ones(mask.shape)
    if shape == "concave":
        mask[4:, 6:] = False
    elif shape == "depth_holes":
        depth[4, 4:10] = [np.nan, 0, 11., np.inf, 0, 1.]
    elif shape == "diagonal":
        mask[:] = False
        np.fill_diagonal(mask, True)
    inward = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    valid = mask & np.isfinite(depth) & (depth > 0) & (depth <= 10.)
    maximum = inward[valid].max()
    candidates = np.argwhere(valid & (inward == maximum))
    # Brute force is deliberately limited to this tiny independent test oracle.
    costs = [sum(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
                 for b in candidates) for a in candidates]
    v, u = candidates[costs.index(min(costs))]
    assert select_mask_interior_pixel(mask, depth, requested_pixel_uv=None,
        maximum_depth_m=10., interior_algorithm=MAX_CLEARANCE_MEDOID_ALGORITHM) == ([int(u), int(v)], maximum)


def test_medoid_selection_handles_a_long_thin_maximum_set_without_pairwise_storage():
    from agent_gateway.profiles.surface_anchor_selection import (
        select_mask_interior_pixel,
    )
    from agent_gateway.skills.perception.surface_anchor_v2 import (
        MAX_CLEARANCE_MEDOID_ALGORITHM,
    )

    mask = np.ones((1, 5000), dtype=bool)
    assert select_mask_interior_pixel(mask, np.ones(mask.shape), requested_pixel_uv=[4999, 0],
        maximum_depth_m=10., interior_algorithm=MAX_CLEARANCE_MEDOID_ALGORITHM) == ([2499, 0], 1.)


@pytest.mark.parametrize("bad", ["unknown", None, True, []])
def test_surface_selection_policy_rejects_unknown_values_instead_of_falling_back(bad):
    from agent_gateway.profiles.surface_anchor_selection import (
        select_mask_interior_pixel,
    )

    with pytest.raises(ValueError, match="algorithm"):
        select_mask_interior_pixel(np.ones((2, 2)), np.ones((2, 2)),
            requested_pixel_uv=[0, 0], maximum_depth_m=10., interior_algorithm=bad)
    with pytest.raises(ValueError, match="policy"):
        _v2_registrations(selection_policy=bad)


def test_surface_selection_profile_changes_all_dependent_contracts_not_the_skill_names():
    _, _, _, original = _v2_registrations()
    _, _, _, changed = _v2_registrations(selection_policy="max_clearance_medoid")
    for before, after in zip(original, changed, strict=True):
        assert before.skill.definition == after.skill.definition
        assert before.skill.schema_entry.parameter_schema_hash == after.skill.schema_entry.parameter_schema_hash
        assert before.skill.schema_entry.execution_result_schema_hash != after.skill.schema_entry.execution_result_schema_hash
        assert before.entry.route.semantic_content_hash != after.entry.route.semantic_content_hash
        assert before.entry.deployment_binding.deployment_binding_hash != after.entry.deployment_binding.deployment_binding_hash

    from agent_gateway.skills.perception.surface_anchor_v2 import (
        INTERIOR_ALGORITHM,
        MAX_CLEARANCE_MEDOID_ALGORITHM,
    )

    for registration, expected in [(original[0], INTERIOR_ALGORITHM), (changed[0], MAX_CLEARANCE_MEDOID_ALGORITHM)]:
        algorithms = registration.skill.schema_entry.execution_result_schema["properties"]["selection"]["properties"]["algorithm_id"]["enum"]
        assert list(algorithms) == ["exact_pixel.v1", expected]


@pytest.mark.parametrize("kind", ["object_query", "relation_query", "spatial_goal"])
def test_anchored_consumer_assembly_rejects_mixed_capture_selection_policies(kind):
    from agent_gateway.profiles.surface_anchor import (
        _build_anchored_goal_registration,
        build_anchored_goal_executor,
        build_anchored_query_executor,
    )

    projection, _, _, (capture, query, relation) = _v2_registrations()
    other_capture = _v2_registrations(selection_policy="max_clearance_medoid")[3][0]
    registration = query if kind == "object_query" else relation
    if kind == "spatial_goal":
        spatial = build_spatial_goal_registration(base_route=projection.skill.routes[0],
            base_binding=projection.binding, component_ids=("right_arm",),
            reference_frame_id="canonical_station")
        registration = _build_anchored_goal_registration(capture=capture, spatial_goal=spatial,
            reference_frame_id="canonical_station")
    kwargs = {"registration": registration, "capture_registration": other_capture,
        "store": None, "lookup_operation": None, "current_station_revision": lambda: 0,
        "clock": lambda: 0., "clock_domain_id": "test", "max_retained_operations": 2}
    with pytest.raises(ValueError, match="same exact anchor result schema"):
        if kind == "spatial_goal":
            build_anchored_goal_executor(**kwargs)
        else:
            build_anchored_query_executor(**kwargs, query_port=None)


@pytest.fixture
def anchor_case(tmp_path):
    projection, registration, _ = _registrations()
    result, capture = LiveBackProjectionProvider(
        capture_port=Capture(), registration=projection
    ).project_parameters(
        {
            "role": "main",
            "target_pixel_uv": [2, 1],
            "pixel_region_xyxy": None,
            "world_z_min_m": None,
            "world_z_max_m": None,
        },
        expected_station_revision=7,
    )
    payload = {
        "measurement": result,
        "target_pixel_uv": [2, 1],
        "captured_at": capture.captured_at,
        "clock_domain_id": "clock:test",
        "source_operation_id": "operation:source",
        "source_request_hash": "a" * 64,
        "runtime_session_id": "session:test",
    }
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts", quota_bytes=65536, max_artifact_bytes=8192
    )

    def put(value, *, version="1"):
        from dataset_contracts.perception_artifacts import SURFACE_ANCHOR_SCHEMA_ID_V2
        data = _json_bytes(value)
        artifact = artifact_ref_for_bytes(
            data,
            media_type="application/json",
            schema_id=SURFACE_ANCHOR_SCHEMA_ID_V1 if version == "1" else SURFACE_ANCHOR_SCHEMA_ID_V2,
            encoding="utf-8",
            compression="none",
        )
        commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id="producer:test",
            runtime_session_id="session:test",
            idempotency_key=artifact.sha256,
            now=10.0,
        )
        return AgentArtifactBinding(
            slot_id="surface_anchor", ordinal=0, artifact=artifact
        )

    binding = put(payload)
    source_request = SimpleNamespace(
        skill_id=CAPTURE_SKILL_ID,
        skill_version="1",
        route_hash=registration.entry.route.semantic_content_hash,
        deployment_binding_hash=registration.entry.deployment_binding.deployment_binding_hash,
        client_id="client:test",
        runtime_session_id="session:test",
        skill_route_manifest_hash="manifest:test",
        semantic_content_hash="a" * 64,
    )
    source = SimpleNamespace(
        status="completed",
        execution_outcome="completed",
        runtime_session_id="session:test",
        request_hash="a" * 64,
        execution_result=payload,
        output_artifact_bindings=(binding,),
    )
    request = SimpleNamespace(
        input_artifact_bindings=(binding,),
        client_id="client:test",
        runtime_session_id="session:test",
        skill_route_manifest_hash="manifest:test",
        expected_station_revision=7,
    )
    kwargs = {
        "request": request,
        "store": store,
        "lookup_operation": lambda _id: (source_request, source),
        "capture_registration": registration,
        "current_station_revision": lambda: 7,
        "clock_domain_id": "clock:test",
    }
    yield SimpleNamespace(
        payload=payload,
        source=source,
        source_request=source_request,
        request=request,
        put=put,
        kwargs=kwargs,
    )
    store.close()


def test_anchor_reuse_authenticates_exact_completed_output(anchor_case):
    payload, artifact_id = resolve_surface_anchor(**anchor_case.kwargs)
    assert payload == anchor_case.payload
    assert (
        artifact_id
        == anchor_case.request.input_artifact_bindings[0].artifact.artifact_id
    )


@pytest.mark.parametrize("mutation", ["none", "producer_version", "producer_binding", "consumer_version", "producer_policy", "client",
    "clock", "forged", "evicted", "revision"])
def test_v2_anchor_resolver_preserves_version_and_producer_authentication(anchor_case, mutation):
    from agent_gateway.skills.perception.surface_anchor_v2 import SELECTION_FIELDS

    _, _, _, (capture, _, _) = _v2_registrations()
    selection = dict.fromkeys(SELECTION_FIELDS)
    selection.update(mode="exact", requested_pixel_uv=[2, 1], language_prompt=None,
        min_score=.25, algorithm_id="exact_pixel.v1", input_rgb_content_hash="c" * 64)
    payload = {**anchor_case.payload, "selection": selection, "mask_geometry": None}
    binding = anchor_case.put(payload, version="2")
    anchor_case.source.execution_result = payload
    anchor_case.source.output_artifact_bindings = (binding,)
    anchor_case.source_request.skill_version = "2"
    anchor_case.source_request.route_hash = capture.entry.route.semantic_content_hash
    anchor_case.source_request.deployment_binding_hash = capture.entry.deployment_binding.deployment_binding_hash
    anchor_case.request.input_artifact_bindings = (binding,)
    kwargs = {**anchor_case.kwargs, "capture_registration": capture}
    if mutation == "producer_version":
        anchor_case.source_request.skill_version = "1"
    elif mutation == "producer_binding":
        anchor_case.source_request.deployment_binding_hash = "another-deployment"
    elif mutation == "consumer_version":
        kwargs["capture_registration"] = _registrations()[1]
    elif mutation == "producer_policy":
        kwargs["capture_registration"] = _v2_registrations(selection_policy="max_clearance_medoid")[3][0]
    elif mutation == "client":
        anchor_case.source_request.client_id = "another-client"
    elif mutation == "clock":
        kwargs["clock_domain_id"] = "another-clock"
    elif mutation == "forged":
        changed = {**payload, "selection": {**selection, "input_rgb_content_hash": "d" * 64}}
        anchor_case.request.input_artifact_bindings = (anchor_case.put(changed, version="2"),)
    elif mutation == "evicted":
        kwargs["lookup_operation"] = lambda _id: None
    elif mutation == "revision":
        kwargs["current_station_revision"] = lambda: 8
    if mutation == "none":
        assert resolve_surface_anchor(**kwargs)[0] == payload
    else:
        with pytest.raises(ValueError):
            resolve_surface_anchor(**kwargs)


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("source_request", "client_id", "another-client"),
        ("source_request", "runtime_session_id", "another-session"),
        ("source_request", "route_hash", "another-route"),
        ("source_request", "deployment_binding_hash", "another-deployment"),
        ("source_request", "skill_id", "perception.back_project"),
        ("source_request", "skill_route_manifest_hash", "another-manifest"),
        ("source", "status", "failed"),
        ("source", "output_artifact_bindings", ()),
    ],
)
def test_anchor_reuse_rejects_untrusted_producer(anchor_case, owner, field, value):
    setattr(getattr(anchor_case, owner), field, value)
    with pytest.raises(ValueError, match="authenticated"):
        resolve_surface_anchor(**anchor_case.kwargs)


def test_cas_integrity_alone_does_not_authenticate_forged_anchor(anchor_case):
    forged = {
        **anchor_case.payload,
        "measurement": {**anchor_case.payload["measurement"], "world_xyz_m": [1, 2, 3]},
    }
    anchor_case.request.input_artifact_bindings = (anchor_case.put(forged),)
    with pytest.raises(ValueError, match="authenticated"):
        resolve_surface_anchor(**anchor_case.kwargs)


def test_evicted_producer_and_changed_revision_require_recapture(anchor_case):
    with pytest.raises(ValueError, match="unavailable; recapture"):
        resolve_surface_anchor(
            **{**anchor_case.kwargs, "lookup_operation": lambda _id: None}
        )


def test_anchored_query_rejects_revision_change_during_planning(
    anchor_case, monkeypatch
):
    from agent_gateway.profiles import surface_anchor

    class CallbackExecutor:
        def __init__(self, *, read, **_kwargs):
            self.read = read

        def cancel_requested(self, _operation_id):
            return False

    monkeypatch.setattr(surface_anchor, "PassiveReadSkillExecutor", CallbackExecutor)
    revision = [7]

    def query(_parameters, *, cancel_requested):
        assert cancel_requested() is False
        revision[0] = 8
        return {"must_not_be_published": True}

    _, capture, registration = _registrations()
    anchor_case.request.parameters = {
        "component_id": "right_arm",
        "reference_frame_id": "canonical_station",
        "heights_m": [0.125],
        "tool_yaw_offsets_rad": [0.0],
        "time_budget_sec": 1.0,
    }
    executor = surface_anchor.build_anchored_query_executor(
        registration=registration,
        capture_registration=capture,
        query_port=SimpleNamespace(query=query),
        store=anchor_case.kwargs["store"],
        lookup_operation=anchor_case.kwargs["lookup_operation"],
        current_station_revision=lambda: revision[0],
        clock=lambda: 10.0,
        clock_domain_id="clock:test",
        max_retained_operations=2,
    )
    with pytest.raises(RuntimeError, match="revision changed during anchored"):
        executor.read(
            anchor_case.request,
            SimpleNamespace(operation_id="operation:query", deadline_at=20.0),
        )
    with pytest.raises(ValueError, match="revision changed"):
        resolve_surface_anchor(
            **{**anchor_case.kwargs, "current_station_revision": lambda: 8}
        )


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
@pytest.mark.parametrize("anchor_version,selection_policy", [
    ("1", "prompt_nearest"), ("2", "prompt_nearest"), ("2", "max_clearance_medoid")])
@pytest.mark.parametrize("with_spatial_goal,with_relation_query", [(False, False), (True, False), (False, True), (True, True)])
def test_native_anchor_capture_and_query_share_no_live_motion_owner(
    tmp_path, with_spatial_goal, with_relation_query, anchor_version, selection_policy
):
    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig,
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from sim.mujoco.deploy import MujocoSimHILTargetConfig

    root = Path(__file__).resolve().parents[1]
    client_id = "test:surface-anchor"
    (tmp_path / "artifacts").mkdir()
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=root
            / "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json",
            realtime_worker=False,
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=False,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
        ),
        allowed_client_ids=(client_id,),
        artifact_storage=MujocoAgentArtifactStorageConfig(root=tmp_path / "artifacts"),
        enable_safe_tcp_pose=True,
        enable_object_goal_queries=True,
        enable_surface_anchors=True,
        enable_surface_anchor_v2=anchor_version == "2",
        surface_anchor_selection_policy=selection_policy,
        segmentation_localizer=_AnchorSegmenter() if anchor_version == "2" else None,
        enable_spatial_goals=with_spatial_goal,
        enable_relation_goal_queries=with_relation_query,
        max_retained_operations=2,
        safe_tcp_motion_profile="object_relation",
    )
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        snapshot = session.observation_producer.capture_and_publish(
            session.service
        ).snapshot
        assert snapshot is not None
        calls = []
        port = session.back_projection_provider.capture_port
        method = "capture_aligned_depth" if anchor_version == "1" else "capture_placement_rgbd"
        original = getattr(port, method)

        def counted(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        setattr(port, method, counted)
        qpos = session.application.scene.data.qpos.copy()
        qvel = session.application.scene.data.qvel.copy()
        sim_time = session.application.scene.data.time
        executors = {registration.skill.definition.skill_id: executor
                     for registration, executor in zip(session.contract_bundle.surface_anchor_registrations,
                                                       session.surface_anchor_executors, strict=True)
                     if registration.skill.definition.skill_version == anchor_version}
        if not with_relation_query:
            with pytest.raises(KeyError):
                session.contract_bundle.route_manifest.route(RELATION_SKILL_ID, "1")
        query_id = RELATION_SKILL_ID if with_relation_query else QUERY_SKILL_ID
        query_parameters = {"component_id": "right_arm",
            "reference_frame_id": session.contract_bundle.motion_profile.reference_frame_id,
            "time_budget_sec": 5.,
            **({key: value for key, value in _relation_parameters().items()
                if key not in {"component_id", "reference_frame_id", "time_budget_sec"}}
               if with_relation_query else {"heights_m": [.125], "tool_yaw_offsets_rad": [0.]})}

        def submit(skill_id, parameters, sequence, inputs=(), expected="completed"):
            entry = session.contract_bundle.route_manifest.route(skill_id, anchor_version)
            route, profile = entry.route, entry.route.allowed_parameter_profiles[0]
            request = AgentSkillRequest(
                client_id=client_id,
                request_id=f"request:{sequence}",
                idempotency_key=f"key:{sequence}",
                client_sequence=sequence,
                runtime_session_id=session.contract_bundle.route_manifest.runtime_session_id,
                skill_route_manifest_hash=session.contract_bundle.route_manifest.semantic_content_hash,
                expected_station_revision=snapshot.station_revision,
                observation_snapshot_id=snapshot.snapshot_id,
                skill_id=skill_id,
                skill_version=anchor_version,
                skill_definition_hash=route.skill_definition_hash,
                route_id=route.route_id,
                route_version=route.route_version,
                route_hash=route.semantic_content_hash,
                deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
                parameter_profile_id=profile.parameter_profile_id,
                parameter_profile_hash=profile.parameter_profile_hash,
                parameters=parameters,
                requested_mode="sim_active",
                requested_timeout_ms=route.max_operation_duration_ms,
                input_artifact_bindings=inputs,
            )
            answer = session.service.submit_request(
                request=request, current_station_revision=snapshot.station_revision
            )
            assert answer.accepted, answer
            assert session.service.launch_operation(
                answer.operation.operation_id
            ).accepted
            executor = executors[skill_id]
            assert executor.wait(answer.operation.operation_id, timeout_sec=40.0)
            completed = session.service.get_operation(answer.operation.operation_id)
            assert completed.execution_outcome == expected, executor.failure_diagnostics
            return completed

        capture_params = {"role": "main", "target_pixel_uv": [326, 280]}
        if anchor_version == "2":
            capture_params.update(selection_mode="mask_interior", language_prompt=None, min_score=.25)
        captured = submit(CAPTURE_SKILL_ID, capture_params, 1)
        queried = submit(
            query_id,
            query_parameters,
            2,
            inputs=tuple(
                replace(binding, production_receipt_ref=None)
                for binding in captured.output_artifact_bindings
                if binding.slot_id == "surface_anchor"
            ),
        )
        assert len(calls) == 1
        assert queried.execution_result["surface_anchor"] == captured.execution_result
        assert queried.execution_result["path_query"]["motion_executed"] is False
        if with_relation_query:
            assert queried.execution_result["relation_constraint_scope"] == "endpoint_only"
            assert queried.execution_result["path_query"]["candidate_count"] == 2
        np.testing.assert_array_equal(qpos, session.application.scene.data.qpos)
        assert not session.application.target.planned_tcp_motion.active
        assert captured.output_artifact_bindings[0].production_receipt_ref
        sequence = 3
        if with_spatial_goal:
            refreshed = submit(
                CAPTURE_SKILL_ID,
                capture_params,
                sequence,
            )
            params = {
                **_goal_parameters(),
                "reference_frame_id": session.contract_bundle.motion_profile.reference_frame_id,
            }
            computed = submit(
                GOAL_SKILL_ID,
                params,
                sequence + 1,
                inputs=tuple(
                    replace(binding, production_receipt_ref=None)
                    for binding in refreshed.output_artifact_bindings
                    if binding.slot_id == "surface_anchor"
                ),
            )
            assert (
                computed.execution_result["surface_anchor"]
                == refreshed.execution_result
            )
            np.testing.assert_allclose(
                computed.execution_result["spatial_goal"]["pose_targets_by_component"][
                    "right_arm"
                ][:3],
                np.array(refreshed.execution_result["measurement"]["world_xyz_m"])
                + params["offset_xyz_m"],
                atol=1e-12,
            )
            assert len(calls) == 2
            sequence += 2
        else:
            with pytest.raises(KeyError):
                session.contract_bundle.route_manifest.route(GOAL_SKILL_ID, "1")
        submit(
            CAPTURE_SKILL_ID, capture_params, sequence
        )
        assert session.service.get_operation(captured.operation_id) is None
        rejected = submit(
            query_id,
            query_parameters,
            sequence + 1,
            inputs=tuple(
                replace(binding, production_receipt_ref=None)
                for binding in captured.output_artifact_bindings
                if binding.slot_id == "surface_anchor"
            ),
            expected="failed",
        )
        assert rejected.execution_result is None
        assert "unavailable; recapture" in str(
            executors[query_id].failure_diagnostics
        )
        if with_spatial_goal:
            rejected_goal = submit(
                GOAL_SKILL_ID,
                params,
                sequence + 2,
                inputs=tuple(
                    replace(binding, production_receipt_ref=None)
                    for binding in captured.output_artifact_bindings
                    if binding.slot_id == "surface_anchor"
                ),
                expected="failed",
            )
            assert rejected_goal.execution_result is None
            assert "unavailable; recapture" in str(
                executors[GOAL_SKILL_ID].failure_diagnostics
            )
        assert len(calls) == (3 if with_spatial_goal else 2)
        np.testing.assert_array_equal(qpos, session.application.scene.data.qpos)
        np.testing.assert_array_equal(qvel, session.application.scene.data.qvel)
        assert sim_time == session.application.scene.data.time
    finally:
        session.close()
