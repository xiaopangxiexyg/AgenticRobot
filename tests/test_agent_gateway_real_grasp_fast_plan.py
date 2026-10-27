from __future__ import annotations

import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent_gateway_real_grasp_approach_plan import (
    _planning_fixture,
    _RetainedPointGraspExecutor,
    _stationary_trajectory,
)

from agent_gateway.profiles.real.grasp_approach_plan import (
    RealGraspApproachPlanLookupError,
)
from agent_gateway.profiles.real.grasp_fast_plan import (
    RealFastGraspPlanProvider,
    build_real_fast_grasp_plan_registration,
)
from agent_gateway.profiles.real.rm65_grasp_reachability import (
    RM65PointGraspReachabilityAdapter,
)
from agent_gateway.runtime.passive_read_executor import PassiveReadError
from agent_gateway.runtime.service import AgentGatewayOperationOutcome
from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)


def _fast_fixture(*, callback=None, axial_shift_m=0., generated_pool=None, replay_snapshot=None):
    paired_registration, paired, store, port, request, operation = _planning_fixture(
        planner=lambda **kwargs: _stationary_trajectory(kwargs["qpos_by_side"]),
        finger_exchange=True,
        frozen_observation=True,
        plan_store_clock=lambda: 10.0,
        planning_operation_lookup=lambda identifier: (
            operation if identifier == operation.operation_id else None
        ),
    )
    source = replace(
        paired.point_grasp_executor.retained.operation,
        operation_id="frozen-observation:test",
        execution_result={"station_revision": 7},
    )
    observation_executor = _RetainedPointGraspExecutor(
        executor_id=paired.point_grasp_registration.observation_registration.executor_id,
        retained=AgentGatewayOperationOutcome(
            accepted=True, reason_code="completed", operation=source
        ),
    )
    fast_registration = build_real_fast_grasp_plan_registration(
        paired=paired_registration,
        point_grasp_registration=paired.point_grasp_registration,
    )
    calls = []
    selected_pose = np.diag([1.0, -1.0, -1.0, 1.0])
    selected_pose[:3, 3] = [0.25, -0.1, 0.3]

    class Native:
        target_builder = port._target_builder
        path_callback = None

        def require_static_grasp_scene(self, **kwargs):
            calls.append(("static_scene", kwargs))

        def check_grasp_geometric_reach(self, **kwargs):
            return None

        def tool_tcp_poses_for_qpos_pair(self, *, qpos_by_side):
            return self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=qpos_by_side["left"], right_qpos=qpos_by_side["right"]
            )

        def __call__(self, **kwargs):
            calls.append(("full_path", kwargs))
            if self.path_callback is not None:
                return self.path_callback(**kwargs)
            return _stationary_trajectory(kwargs["qpos_by_side"])

    adapter = SimpleNamespace(
        planner=Native(),
        arm_base_origin_candidate_m=(-1.0, 0.0, 0.0),
        profile=paired.point_grasp_registration.profile.arm_reachability_profile,
        candidate_reference_frame_id=paired.registration.profile.reference_frame_id,
        frame_graph=paired.frame_graph,
        _solver_pose=lambda pose: np.asarray(pose).copy(),
    )
    adapter.approach_preference = lambda pose, target: (
        RM65PointGraspReachabilityAdapter.approach_preference(adapter, pose, target)
    )
    if generated_pool is not None:
        # Isolate the producer/consumer seam; native feasibility is covered
        # separately by recorded-scene replay, not this fake planner port.
        adapter.approach_preference = lambda *_: (0., True, False, True)

    def point_plan(private_input, **kwargs):
        calls.append(("point_plan", private_input))
        assert kwargs["annotate_result"] is False
        assert kwargs["refine_grasp_centers"] is True
        assert kwargs["refine_grasp_axial_depth"] is True
        assert (
            private_input.parameters["observation_operation_id"] == source.operation_id
        )
        assert "candidate_ordinal" not in private_input.parameters
        scene = paired.visible_scene_reader(None, None)
        selector = kwargs["selection_factory"](
            port.latest_measured_qpos(), lambda: scene
        )
        if callback is not None:
            callback(selector, kwargs)
        candidates = [SimpleNamespace(
            base_from_tcp=selected_pose, score=0.8, visible_collision_free=True)]
        if axial_shift_m:
            child_pose = selected_pose.copy()
            child_pose[:3, 3] += selected_pose[:3, 2] * axial_shift_m
            candidates.append(SimpleNamespace(base_from_tcp=child_pose, score=.8,
                visible_collision_free=True, capture_support_score=.5, axial_parent_index=0))
        if generated_pool is not None:
            candidates = generated_pool
        selected = selector(
            candidates,
            (0.25, -0.1, 0.3),
            1,
        )
        if not selected.selected_indices:
            raise PassiveReadError(
                "No full path",
                reason_code="point_grasp_no_arm_candidate",
                diagnostic={"debug_evidence": {"sha256": "e" * 64}},
            )
        return SimpleNamespace(
            rgbd_snapshot=replay_snapshot,
            multiview_snapshot=False,
            execution_result={
                "candidates": [
                    {
                        "T_reference_from_tool_tcp": np.asarray(
                            candidates[selected.selected_indices[0]].base_from_tcp).reshape(-1).tolist(),
                        "gripper_width_m": 0.06,
                        "score": 0.8,
                    }
                ]
            }
        )

    paired.grasp_trajectory_planner = adapter.planner
    provider = RealFastGraspPlanProvider(
        registration=fast_registration,
        point_provider=SimpleNamespace(
            registration=paired.point_grasp_registration,
            arm_reachability_evaluator=adapter,
            plan_request=point_plan,
        ),
        paired_provider=paired,
        observation_executor=observation_executor,
        monotonic=lambda: 0.0,
    )
    request.parameters = {
        "component_id": "right_arm",
        "observation_operation_id": source.operation_id,
        "role": "main",
        "target_pixel_uv": [10, 20],
        "view_policy": "auto",
        "speed_scale": 0.2,
        "planning_budget_sec": 20.0,
    }
    request.request_id = "first-full-path-request:test"
    operation = replace(operation, deadline_at=40.0)
    return provider, store, port, request, operation, calls, source


def test_fast_plan_uses_actual_observation_and_same_once_only_execution_store():
    provider, store, _, request, operation, calls, source = _fast_fixture()
    result = provider.plan_request(request, operation).execution_result
    assert result["planning_status"] == "path_found"
    assert result["motion_or_hold_performed"] is False
    assert [name for name, _ in calls] == ["point_plan", "static_scene", "full_path"]
    path_kwargs = calls[-1][1]
    assert path_kwargs["require_grasp_approach"] is True
    assert path_kwargs["time_budget_sec"] == 8.0
    assert result["diagnostics"]["path_query_count"] == 1
    profile = provider.paired_provider.registration.profile
    committed = replace(
        operation,
        status="completed",
        execution_outcome="completed",
        execution_result=result,
    )
    store._planning_operation_lookup = lambda identifier: (
        committed if identifier == committed.operation_id else None
    )
    plan = store.reserve(
        plan_id=result["grasp_approach_plan_id"],
        plan_hash=result["grasp_approach_plan_hash"],
        operation_id="motion:test",
        runtime_session_id=source.runtime_session_id,
        profile_hash=profile.semantic_content_hash,
        component_id="right_arm",
        expected_station_revision=7,
        now=10.0,
    )
    assert plan.source_kind == "frozen_observation_fast"
    assert plan.source_operation_id == source.operation_id
    assert plan.source_request_hash == source.request_hash
    assert plan.source_result_hash == canonical_content_hash(
        source.execution_result, domain="dexteleop.agent_gateway.passive_read_result.v1"
    )
    assert plan.profile_hash == profile.semantic_content_hash
    store.consume(plan_id=plan.plan_id, operation_id="motion:test")
    with pytest.raises(RealGraspApproachPlanLookupError):
        store.reserve(
            plan_id=plan.plan_id,
            plan_hash=plan.semantic_content_hash,
            operation_id="second-motion:test",
            runtime_session_id=source.runtime_session_id,
            profile_hash=profile.semantic_content_hash,
            component_id="right_arm",
            expected_station_revision=7,
            now=10.0,
        )


def test_real_fast_selector_prefers_deeper_support_at_equal_motion_cost():
    def check(selector, _kwargs):
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = [.25, -.1, .3]
        candidates = tuple(SimpleNamespace(base_from_tcp=pose, score=.8,
            visible_collision_free=True, capture_support_score=score) for score in (.1, 1.))
        result = selector(candidates, (.25, -.1, .3), 1)
        assert result.selected_indices == (1,)
        assert [row["index"] for row in selector.result.attempts] == [1]

    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


def test_selected_contact_geometry_is_returned_without_claiming_physical_grasp_success():
    pose = np.diag([1., -1., -1., 1.]); pose[:3, 3] = [.25, -.1, .3]
    candidate = SimpleNamespace(base_from_tcp=pose, score=.8, visible_collision_free=True,
        capture_support_score=.39, opposed_capture_support_score=.04,
        opposed_capture_support_observed_sides=2, contact_height_support_score=.99,
        surface_alignment_score=.2, surface_alignment_observed_sides=2)
    provider, _, _, request, operation, _, _ = _fast_fixture(generated_pool=[candidate])
    result = provider.plan_request(request, operation).execution_result
    evidence = result["diagnostics"]["contact_evidence"]
    assert evidence["opposed_capture_support_score"] == .04
    assert evidence["opposed_capture_support_observed_sides"] == 2
    assert evidence["contact_height_support_score"] == .99
    assert "grasp_success" not in evidence
    assert result["motion_or_hold_performed"] is False
    provider.registration.payload_validator.validate_execution_result(result)
    for invalid in (True, float("nan"), -1., 1.01):
        modified = {**result, "diagnostics": {**result["diagnostics"], "contact_evidence": {
            **evidence, "opposed_capture_support_score": invalid}}}
        with pytest.raises((TypeError, ValueError)):
            provider.registration.payload_validator.validate_execution_result(modified)


def test_real_fast_selector_skips_top_rim_even_with_full_axial_support():
    def check(selector, _kwargs):
        pose = np.diag([1.,-1.,-1.,1.]);pose[:3,3] = [.25,-.1,.3]
        candidates = tuple(SimpleNamespace(base_from_tcp=pose,score=.8,
            visible_collision_free=True,capture_support_score=1.,
            capture_height_support_score=height) for height in (.1,.9))
        result = selector(candidates,tuple(pose[:3,3]),1)
        assert result.selected_indices == (1,)
        assert [row["index"] for row in selector.result.attempts] == [1]
        assert result.diagnostics["arm_shortlist_pad_height_rejected_count"] == 1
    provider,_,_,request,operation,_,_ = _fast_fixture(callback=check)
    assert provider.plan_request(request,operation).execution_result["planning_status"] == "path_found"


def test_corrected_fast_target_matches_query_public_result_and_immutable_plan():
    provider, store, _, request, operation, calls, source = _fast_fixture(axial_shift_m=.01)
    result = provider.plan_request(request, operation).execution_result
    queries = [kwargs for name, kwargs in calls if name == "full_path"]
    query = queries[-1]
    assert len(queries) == 2
    actual = np.asarray(result["T_reference_from_tool_tcp"]).reshape(4, 4)
    assert actual[2, 3] == pytest.approx(.29)
    np.testing.assert_array_equal(actual, query["target_tool_tcp_world"])
    assert query["time_budget_sec"] == 3.
    committed = replace(operation, status="completed", execution_outcome="completed", execution_result=result)
    store._planning_operation_lookup = lambda _: committed
    plan = store.reserve(plan_id=result["grasp_approach_plan_id"],
        plan_hash=result["grasp_approach_plan_hash"], operation_id="motion:corrected",
        runtime_session_id=source.runtime_session_id,
        profile_hash=provider.paired_provider.registration.profile.semantic_content_hash,
        component_id="right_arm", expected_station_revision=7, now=10.)
    np.testing.assert_array_equal(np.asarray(plan.target_pose_reference).reshape(4, 4), actual)
    assert plan.trajectory.semantic_content_hash == result["trajectory_hash"]
    old_pose = actual.copy()
    old_pose[2, 3] += .01
    changed = replace(plan, target_pose_reference=tuple(old_pose.ravel()))
    assert changed.plan_id != plan.plan_id
    store.consume(plan_id=plan.plan_id, operation_id="motion:corrected")
    with pytest.raises(RealGraspApproachPlanLookupError):
        store.consume(plan_id=plan.plan_id, operation_id="motion:corrected-again")


def test_actual_axial_producer_flows_through_fast_provider_and_retained_plan(tmp_path):
    from test_agent_gateway_graspgenx_point_grasp import _axial_pool
    _, pool, _, _ = _axial_pool(tmp_path)
    provider, store, _, request, operation, calls, source = _fast_fixture(generated_pool=pool.candidates)
    result = provider.plan_request(request, operation).execution_result
    assert result["planning_status"] == "path_found"
    pose = tuple(result["T_reference_from_tool_tcp"])
    child, = [c for c in pool.candidates if c.base_from_tcp == pose]
    assert child.axial_parent_index is not None
    parent = pool.candidates[child.axial_parent_index]
    assert child.base_from_tcp != parent.base_from_tcp
    query = [value for name, value in calls if name == "full_path"][-1]
    np.testing.assert_array_equal(query["target_tool_tcp_world"], np.asarray(pose).reshape(4, 4))
    store._planning_operation_lookup = lambda _: replace(operation, status="completed",
        execution_outcome="completed", execution_result=result)
    retained = store.reserve(plan_id=result["grasp_approach_plan_id"],
        plan_hash=result["grasp_approach_plan_hash"], operation_id="actual-producer:test",
        runtime_session_id=source.runtime_session_id,
        profile_hash=provider.paired_provider.registration.profile.semantic_content_hash,
        component_id="right_arm", expected_station_revision=7, now=10.)
    np.testing.assert_array_equal(np.asarray(retained.target_pose_reference).reshape(-1), child.base_from_tcp)


@pytest.mark.parametrize("candidate_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("distance_m,support_weight,expected", [
    (.01, .10, 1), (.20, .10, 0), (.01, 0., 0),
])
def test_real_fast_selector_balances_motion_and_capture_support(
    candidate_order, distance_m, support_weight, expected,
):
    def check(selector, _kwargs):
        pose = np.diag([1.0, -1.0, -1.0, 1.0])
        pose[:3, 3] = [0.25, -0.1, 0.3]
        farther_pose = pose.copy()
        direction = pose[:3, 3] - selector.current_tcp[:3, 3]
        farther_pose[:3, 3] += distance_m * direction / np.linalg.norm(direction)
        selector.policy = replace(selector.policy, capture_support_weight=support_weight)
        target = (0.25, -0.1, 0.3)
        better_priority = selector.priority(pose, 0.8, target)
        farther_priority = selector.priority(farther_pose, 0.8, target)
        assert better_priority[0] == farther_priority[0] == 0
        assert farther_priority[1] - better_priority[1] == pytest.approx(distance_m / .5)
        candidates = (
            SimpleNamespace(
                base_from_tcp=pose,
                score=0.8,
                visible_collision_free=True,
                capture_support_score=0.20,
            ),
            SimpleNamespace(
                base_from_tcp=farther_pose,
                score=0.8,
                visible_collision_free=True,
                capture_support_score=1.0,
            ),
        )
        result = selector(
            tuple(candidates[index] for index in candidate_order), target, 1
        )
        expected_index = candidate_order.index(expected)
        assert result.selected_indices == (expected_index,)
        assert [row["index"] for row in selector.result.attempts] == [expected_index]

    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    result = provider.plan_request(request, operation).execution_result
    assert result["planning_status"] == "path_found"


def _count_preference_queries(selector):
    counts = {"frame": 0, "approach": 0, "reach": 0}
    for name, owner, attribute in (
        ("frame", selector.adapter, "_solver_pose"),
        ("approach", selector.adapter, "approach_preference"),
        ("reach", selector.planner, "check_grasp_geometric_reach"),
    ):
        original = getattr(owner, attribute)
        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        setattr(owner, attribute, counted)
    return counts


def _preference_pose():
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [.25, -.1, .3]
    return pose


def test_preference_reuses_geometry_but_not_frame_checks_score_or_weights():
    def check(selector, _kwargs):
        counts = _count_preference_queries(selector)
        pose, target = _preference_pose(), (.25, -.1, .3)
        first = selector.priority(pose, .8, target)
        second = selector.priority(np.asfortranarray(pose), .2, target)
        assert second[0] == first[0]
        assert second[1]-first[1] == pytest.approx(.06)
        assert counts == {"frame": 2, "approach": 1, "reach": 1}
        selector.policy = replace(selector.policy, translation_weight=2.)
        changed = selector.priority(pose, .8, target)
        selector._preference_cache.clear()
        recomputed = selector.priority(pose, .8, target)
        assert changed == recomputed and changed[1] != first[1]
        assert counts == {"frame": 4, "approach": 2, "reach": 2}
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


def test_preference_exact_keys_change_with_pose_target_and_frozen_state():
    def check(selector, _kwargs):
        counts = _count_preference_queries(selector)
        pose, target = _preference_pose(), np.array([.25, -.1, .3])
        selector.priority(pose, .8, target)
        moved = pose.copy()
        moved[0, 3] = np.nextafter(moved[0, 3], 1.)
        selector.priority(moved, .8, target)
        changed_target = target.copy()
        changed_target[1] = np.nextafter(changed_target[1], 1.)
        selector.priority(pose, .8, changed_target)
        before = selector.qpos_cut
        changed_qpos = {s: q.copy() for s, q in before.qpos_by_side.items()}
        changed_qpos["left"][0] += .0001
        try:
            selector.qpos_cut = SimpleNamespace(qpos_by_side=changed_qpos)
            selector.priority(pose, .8, target)
        finally:
            selector.qpos_cut = before
        tcp = selector.current_tcp.copy()
        try:
            selector.current_tcp[0, 3] += .0001
            selector.priority(pose, .8, target)
        finally:
            selector.current_tcp[:] = tcp
        assert counts == {"frame": 5, "approach": 5, "reach": 5}
        assert len(selector._preference_cache) == 5
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


def test_preference_cache_hit_cannot_bypass_frame_fault_cancellation_or_deadline():
    def check(selector, _kwargs):
        pose, target = _preference_pose(), (.25, -.1, .3)
        selector.priority(pose, .8, target)
        original = selector.adapter._solver_pose
        def fail(_pose):
            raise RuntimeError("frame identity changed")
        selector.adapter._solver_pose = fail
        with pytest.raises(RuntimeError, match="frame identity changed"):
            selector.priority(pose, .8, target)
        selector.adapter._solver_pose = original
        selector.cancelled = lambda: True
        with pytest.raises(RuntimeError, match="cancelled"):
            selector.priority(pose, .8, target)
        selector.cancelled = lambda: False
        clock = selector.monotonic
        selector.monotonic = lambda: selector.deadline
        with pytest.raises(RM65GraspTrajectoryPlanningError, match="budget_exhausted"):
            selector.priority(pose, .8, target)
        selector.monotonic = clock
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


def test_preference_does_not_cache_failed_queries_or_share_across_requests():
    from agent_gateway.profiles.real.grasp_first_path import RM65FirstGraspPathSelector
    def check(selector, _kwargs):
        pose, target = _preference_pose(), (.25, -.1, .3)
        original = selector.planner.check_grasp_geometric_reach
        def fail(**_kwargs):
            raise RuntimeError("native geometry failed")
        selector.planner.check_grasp_geometric_reach = fail
        with pytest.raises(RuntimeError, match="native geometry failed"):
            selector.priority(pose, .8, target)
        assert not selector._preference_cache
        selector.planner.check_grasp_geometric_reach = original
        counts = _count_preference_queries(selector)
        expected = selector.priority(pose, .8, target)
        fresh = RM65FirstGraspPathSelector(adapter=selector.adapter, qpos_cut=selector.qpos_cut,
            visible_scene=selector.visible_scene, speed_scale=selector.speed_scale,
            deadline=selector.deadline, policy=selector.policy, monotonic=selector.monotonic)
        assert not fresh._preference_cache
        assert fresh.priority(pose, .8, target) == expected
        assert counts["reach"] == 2
        assert fresh._preference_cache is not selector._preference_cache
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


def test_preference_cache_full_recomputes_without_evicting_or_rejecting(monkeypatch):
    from agent_gateway.profiles.real import grasp_first_path
    monkeypatch.setattr(grasp_first_path, "_PREFERENCE_CACHE_MAX_ENTRIES", 2)
    def check(selector, _kwargs):
        counts = _count_preference_queries(selector)
        pose, target = _preference_pose(), (.25, -.1, .3)
        for offset in (0., .001, .002, .002, 0.):
            moved = pose.copy()
            moved[0, 3] += offset
            assert selector.priority(moved, .8, target)[0] == 0
        assert len(selector._preference_cache) == 2
        assert counts == {"frame": 5, "approach": 4, "reach": 4}
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


@pytest.mark.parametrize("binding", ["profile", "config", "target_builder"])
def test_preference_rebinding_discards_prior_geometry_and_retains_current_objects(binding):
    def check(selector, _kwargs):
        counts = _count_preference_queries(selector)
        pose, target = _preference_pose(), (.25, -.1, .3)
        expected = selector.priority(pose, .8, target)
        if binding == "profile":
            selector.adapter.profile = replace(selector.adapter.profile)
            current = selector.adapter.profile
        else:
            current = SimpleNamespace(binding=binding)
            setattr(selector.planner, binding, current)
        assert selector.priority(pose, .8, target) == expected
        assert counts["reach"] == 2 and len(selector._preference_cache) == 1
        assert any(value is current for value in selector._preference_bindings)
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


@pytest.mark.parametrize("changed", ["solver_pose", "arm_origin"])
def test_preference_changed_geometry_matches_a_fresh_computation(changed):
    def check(selector, _kwargs):
        counts = _count_preference_queries(selector)
        pose, target = _preference_pose(), (.25, -.1, .3)
        selector.priority(pose, .8, target)
        if changed == "solver_pose":
            original = selector.adapter._solver_pose
            def moved(value):
                result = original(value)
                result[0, 3] += .001
                return result
            selector.adapter._solver_pose = moved
        else:
            selector.adapter.arm_base_origin_candidate_m = (-1., .1, 0.)
        actual = selector.priority(pose, .8, target)
        selector._preference_cache.clear()
        assert selector.priority(pose, .8, target) == actual
        assert counts["reach"] == 3
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"


@pytest.mark.parametrize("outside,side_admissible,upright", [
    (True, True, True), (False, False, True), (False, True, False),
])
def test_preference_cached_negative_result_still_filters_without_a_path_query(outside, side_admissible, upright):
    def check(selector, _kwargs):
        selector.adapter.approach_preference = lambda *_: (25., side_admissible, True, upright)
        selector.planner.check_grasp_geometric_reach = lambda **_: SimpleNamespace(outside=outside)
        counts = _count_preference_queries(selector)
        pose, target = _preference_pose(), (.25, -.1, .3)
        expected = selector.priority(pose, .8, target)
        assert expected[0] == 1
        assert selector.priority(pose, .8, target) == expected
        assert counts == {"frame": 2, "approach": 1, "reach": 1}
    provider, store, _, request, operation, calls, _ = _fast_fixture(callback=check)
    with pytest.raises(PassiveReadError) as captured:
        provider.plan_request(request, operation)
    assert captured.value.reason_code == "grasp_fast_no_acceptable_path"
    assert not store._plans and not [value for name, value in calls if name == "full_path"]


def test_preference_cross_stage_reuse_still_calls_the_complete_native_planner():
    observed = []
    def check(selector, _kwargs):
        counts = _count_preference_queries(selector)
        selector.priority(_preference_pose(), .8, (.25, -.1, .3))
        observed.append((selector, counts))
    provider, _, _, request, operation, calls, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"
    selector, counts = observed[0]
    assert counts["frame"] >= 3 and counts["reach"] == counts["approach"] == 1
    assert selector._preference_cache_hits >= 1
    assert len([value for name, value in calls if name == "full_path"]) == 1


def test_fast_selector_logs_bounded_exact_ranking_without_extra_queries(caplog):
    caplog.set_level(logging.INFO, logger="agent_gateway.profiles.real.grasp_first_path")
    expected_poses = []
    pose = np.eye(4)
    pose[:3, 3] = [.25, -.1, .3]
    candidates = []
    for i in range(12):
        moved = pose.copy()
        moved[:3, 3] += i * .06
        expected_poses.append(moved.reshape(-1).tolist())
        candidates.append(SimpleNamespace(base_from_tcp=moved, score=.8, model_score=.95,
            visible_collision_free=True, capture_support_score=.5,
            capture_height_support_score=.9, opposed_capture_support_score=.8,
            surface_alignment_score=.4, pose_source="model_scored"))
    provider, _, _, request, operation, calls, _ = _fast_fixture(generated_pool=candidates)
    assert provider.plan_request(request, operation).execution_result["planning_status"] == "path_found"
    assert len([value for name, value in calls if name == "full_path"]) == 1
    records = [json.loads(record.message.split(": ", 1)[1])
        for record in caplog.records if record.message.startswith("Fast-grasp search evidence: ")]
    assert len(records) == 1
    evidence = records[0]
    assert evidence["pool_count"] == evidence["eligible_count"] == 12
    assert len(evidence["candidates"]) == 8
    assert len(evidence["attempts"]) == 1
    assert evidence["selected_index"] in evidence["lowest_cost_indices"]
    for record in evidence["candidates"]:
        assert record["target_tcp_reference"] == expected_poses[record["index"]]
        assert record["ranking_score"] == .8
        assert record["model_score"] == .95
        assert record["total_cost"] == pytest.approx(
            record["motion_cost"] + sum(record["support_costs"].values()))
        assert record["support_costs"] == pytest.approx({
            "axial_insertion": 0., "pad_height": .035, "opposed_insertion": .07,
            "surface_alignment": .21})


def test_fast_selector_reuses_advisory_seed_but_requires_a_full_path():
    def check(selector, _kwargs):
        probes = []
        seed = np.zeros(6)
        def probe(**kwargs):
            probes.append(kwargs)
            return SimpleNamespace(classification='clearance_safe', seed_qpos=seed, reason_code=None)
        selector.planner.probe_grasp_endpoint_clearance = probe
        def full(**kwargs):
            np.testing.assert_array_equal(kwargs['initial_seed_qpos'], seed)
            assert kwargs['require_grasp_approach'] is True
            return _stationary_trajectory(kwargs['qpos_by_side'])
        selector.planner.path_callback = full
    provider, _, _, request, operation, calls, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request,operation).execution_result['planning_status'] == 'path_found'
    assert len([c for c in calls if c[0]=='full_path']) == 1


def test_fast_probe_unavailable_is_not_a_retryable_candidate_miss():
    def check(selector, _kwargs):
        selector.planner.probe_grasp_endpoint_clearance = lambda **_: SimpleNamespace(
            classification='check_unavailable', reason_code='endpoint_collision_check_unavailable')
    provider, _, _, request, operation, calls, _ = _fast_fixture(callback=check)
    with pytest.raises(PassiveReadError, match='endpoint_collision_check_unavailable') as rejected:
        provider.plan_request(request,operation)
    assert rejected.value.reason_code == 'endpoint_collision_check_unavailable'
    assert not any(c[0]=='full_path' for c in calls)


def test_fast_selector_logs_late_rejections_within_existing_query_bound(caplog):
    caplog.set_level(logging.INFO, logger="agent_gateway.profiles.real.grasp_first_path")
    candidates = []
    for i in range(100):
        pose = np.eye(4)
        pose[0, 3] = i * .06
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=.8,
            visible_collision_free=True, pose_source="model_scored"))

    def reject_queries(selector, _kwargs):
        def reject(**_request):
            raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")
        selector.planner.path_callback = reject

    provider, _, _, request, operation, calls, _ = _fast_fixture(
        callback=reject_queries, generated_pool=candidates)
    with pytest.raises(PassiveReadError, match="No full path"):
        provider.plan_request(request, operation)
    assert len([value for name, value in calls if name == "full_path"]) == 64
    evidence = next(json.loads(record.message.split(": ", 1)[1])
        for record in caplog.records if record.message.startswith("Fast-grasp search evidence: "))
    assert evidence["selected_index"] is None
    assert evidence["pool_count"] == 100
    assert len(evidence["attempts"]) == 64
    assert len(evidence["candidates"]) <= 72
    records = {record["index"]: record for record in evidence["candidates"]}
    assert {attempt["index"] for attempt in evidence["attempts"]} <= records.keys()
    for index, record in records.items():
        assert record["target_tcp_reference"] == candidates[index].base_from_tcp.reshape(-1).tolist()


def test_real_selector_recovers_raw_pose_after_cheap_derived_ik_rejection():
    def check(selector, _kwargs):
        pose = np.diag([1., -1., -1., 1.])
        pose[:3,3] = [.25, -.1, .3]
        direction = pose[:3,3] - selector.current_tcp[:3,3]
        direction /= np.linalg.norm(direction)
        candidates = []
        for i, distance in enumerate((0., .2, .01)):
            target = pose.copy()
            target[:3,3] += distance * direction
            candidates.append(SimpleNamespace(base_from_tcp=tuple(target.ravel()), score=.8,
                capture_support_score=.7, visible_collision_free=True,
                pose_source='dmtac_local_orientation_variant_v1' if i == 0 else 'model_scored'))
        budgets = []
        def plan(**kwargs):
            budgets.append(kwargs['time_budget_sec'])
            if len(budgets) == 1:
                raise RM65GraspTrajectoryPlanningError('ik_search_exhausted')
            return _stationary_trajectory(kwargs['qpos_by_side'])
        selector.planner.path_callback = plan
        try:
            selected = selector(tuple(candidates), tuple(pose[:3,3]), 1)
            assert selected.selected_indices == (2,)
            assert budgets == [8., 3.]
            assert selector.result.attempts[-1]['query_mode'] == 'original_model_neighbor'
        finally:
            selector.planner.path_callback = None
    provider, _, _, request, operation, _, _ = _fast_fixture(callback=check)
    assert provider.plan_request(request, operation).execution_result['planning_status'] == 'path_found'


def test_fast_plan_rejects_a_foreign_or_missing_frozen_source_before_inference():
    provider, _, _, request, operation, calls, _ = _fast_fixture()
    request.parameters["observation_operation_id"] = "missing:source"
    with pytest.raises(ValueError, match="completed observation"):
        provider.plan_request(request, operation)
    assert calls == []


def test_fast_plan_does_not_accept_inline_artifacts_instead_of_retained_source():
    provider, _, _, request, operation, calls, _ = _fast_fixture()
    request.input_artifact_bindings = (object(),)
    with pytest.raises(ValueError, match="retained observation"):
        provider.plan_request(request, operation)
    assert calls == []


def test_fast_plan_never_stores_a_station_revision_drifted_plan():
    def drift(_selector, _kwargs):
        provider.paired_provider._current_station_revision = lambda: 8

    provider, store, _, request, operation, _, _ = _fast_fixture(callback=drift)
    result = provider.plan_request(request, operation).execution_result
    assert not result["eligible_for_execution"]
    assert result["grasp_approach_plan_id"] is None
    assert result["reason_code"] == "grasp_path_station_revision_changed"
    assert not store._plans


def test_fast_failed_goal_preserves_failure_evidence_without_a_plan():
    def reject(selector, kwargs):
        assert kwargs["failure_snapshot_sink"] is sink

        def blocked(**_kwargs):
            raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")

        selector.planner.path_callback = blocked

    sink = lambda _data: {"sha256": "e" * 64}
    provider, store, _, request, operation, _, _ = _fast_fixture(callback=reject)
    with pytest.raises(PassiveReadError) as captured:
        provider.plan_request(request, operation, failure_snapshot_sink=sink)
    assert captured.value.reason_code == "grasp_fast_no_acceptable_path"
    assert captured.value.diagnostic["debug_evidence"]["sha256"] == "e" * 64
    assert not store._plans


def test_fast_success_retains_segmented_snapshot_bound_to_actual_trajectory():
    provider, _, _, request, operation, _, _ = _fast_fixture(replay_snapshot=b"prepared-segmented-snapshot")
    retained = []
    result = provider.plan_request(request, operation,
        replay_snapshot_sink=lambda *args: retained.append(args)).execution_result
    assert result["eligible_for_execution"]
    assert retained == [(b"prepared-segmented-snapshot", False, result["trajectory_hash"])]
    assert result["motion_or_hold_performed"] is False


def test_fast_replay_retention_time_is_inside_publication_deadline():
    provider, store, _, request, operation, _, _ = _fast_fixture(replay_snapshot=b"segmented")
    now = [0.]
    provider.monotonic = lambda: now[0]
    def delayed_sink(*_):
        now[0] = 121.
    with pytest.raises(RuntimeError, match="deadline"):
        provider.plan_request(request, operation, replay_snapshot_sink=delayed_sink)
    assert not store._plans


@pytest.mark.parametrize("capacity", ["available", "full", "oversized"])
def test_real_fast_executor_retention_uses_the_shared_store_without_capacity_gate(tmp_path, caplog, monkeypatch, capacity):
    from pathlib import Path

    from agent_gateway.profiles.real.grasp_fast_plan import (
        build_real_fast_grasp_plan_executor,
    )
    from agent_gateway.runtime.artifact_producer import (
        artifact_ref_for_bytes,
        commit_artifact_bytes,
    )
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
    from dataset_contracts.grasp_rgbd_snapshot import (
        GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
        decode_grasp_rgbd_snapshot,
        encode_grasp_rgbd_snapshot,
    )

    blob = encode_grasp_rgbd_snapshot(rgb=np.zeros((2,2,3),dtype=np.uint8),
        depth_m=np.ones((2,2),dtype=np.float32), target_mask=np.ones((2,2),dtype=bool),
        metadata={"schema_id":GRASP_RGBD_SNAPSHOT_SCHEMA_ID})
    provider, _, _, request, operation, _, source = _fast_fixture(replay_snapshot=blob)
    store = RecoverableAgentArtifactStore(root=tmp_path/"artifacts", quota_bytes=8192,
        max_artifact_bytes=32 if capacity == "oversized" else 8192, max_chunk_bytes=1024)
    try:
        if capacity == "full":
            filler = b"f"*8192
            ref = artifact_ref_for_bytes(filler, media_type="application/octet-stream", schema_id=None,
                encoding="bytes",compression="none")
            commit_artifact_bytes(store=store,data=filler,artifact=ref,producer_id="fixture",
                runtime_session_id=source.runtime_session_id,idempotency_key="fill",now=10.)
        executor = build_real_fast_grasp_plan_executor(registration=provider.registration, provider=provider,
            runtime_session_id=source.runtime_session_id, current_station_revision=lambda:7,
            max_retained_operations=4, clock=lambda:10., artifact_store=store)
        # Exercise the real producer callback/store; launch/cancel ownership is
        # covered separately by the passive executor and cancellation tests.
        monkeypatch.setattr(executor, "cancel_requested", lambda _: False)
        result = executor._read(request, operation).execution_result
        assert result["eligible_for_execution"]
        if capacity == "available":
            message = next(record.getMessage() for record in caplog.records if "segmented replay retained" in record.getMessage())
            assert result["trajectory_hash"] in message
            retained = Path(message.split(" path=",1)[1]).read_bytes()
            assert retained == blob
            arrays, _ = decode_grasp_rgbd_snapshot(retained)
            assert arrays["target_mask"].all()
        else:
            assert "replay retention unavailable" in caplog.text
    finally:
        store.close()


@pytest.mark.parametrize("phase", ["before_path", "after_priority", "before_publication"])
def test_fast_cancel_never_publishes_executable_plan(phase):
    cancelled = [False]

    def cancel(selector, _kwargs):
        if phase in {"before_path", "after_priority"}:
            if phase == "after_priority":
                selector.priority(_preference_pose(), .8, (.25, -.1, .3))
            cancelled[0] = True
        else:

            def plan(**kwargs):
                path = _stationary_trajectory(kwargs["qpos_by_side"])
                cancelled[0] = True
                return path

            selector.planner.path_callback = plan

    provider, store, _, request, operation, _, _ = _fast_fixture(callback=cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.plan_request(request, operation, cancelled=lambda: cancelled[0])
    assert not store._plans


def test_fast_expired_budget_does_not_start_segmentation_or_planning():
    provider, store, _, request, operation, calls, _ = _fast_fixture()
    provider.paired_provider._clock = lambda: 41.0
    result = provider.plan_request(request, operation).execution_result
    assert result["reason_code"] == "grasp_fast_search_budget_exhausted"
    assert result["eligible_for_execution"] is False
    assert calls == [] and not store._plans


@pytest.mark.parametrize("transition", ("expired", "cancelled", "both"))
def test_fast_preparation_guard_uses_original_clock_and_cancel_priority(transition):
    now, cancelled = [0.0], [False]

    def during_preparation(selector, kwargs):
        guard = kwargs["preparation_guard"]
        assert callable(guard)
        if transition in {"expired", "both"}:
            now[0] = selector.deadline
        cancelled[0] = transition in {"cancelled", "both"}
        guard()
        pytest.fail("Preparation must stop before starting the next stage")

    provider, store, port, request, operation, calls, _ = _fast_fixture(callback=during_preparation)
    provider.monotonic = lambda: now[0]
    if transition == "expired":
        with pytest.raises(PassiveReadError) as caught:
            provider.plan_request(request, operation, cancelled=lambda: cancelled[0])
        assert caught.value.reason_code == "grasp_fast_search_budget_exhausted"
    else:
        with pytest.raises(RuntimeError, match="cancelled during proposal preparation") as caught:
            provider.plan_request(request, operation, cancelled=lambda: cancelled[0])
        assert not isinstance(caught.value, PassiveReadError)
    assert not store._plans
    assert not any(name == "full_path" for name, _ in calls)
    assert port.snapshot().dispatch_count == 0


@pytest.mark.parametrize(
    "reason",
    [
        "trajectory_record_failed",
        "ik_solver_error",
        "start_state_safety_envelope_blocked",
        "path_collision_check_unavailable",
        "unexpected_native_fault",
    ],
)
def test_fast_request_wide_fault_is_not_another_goal(reason):
    def fault(selector, _kwargs):
        def fail(**_kwargs):
            raise RM65GraspTrajectoryPlanningError(
                reason, details={"planning_phase": "full_path"}
            )

        selector.planner.path_callback = fail

    provider, store, _, request, operation, calls, _ = _fast_fixture(callback=fault)
    with pytest.raises(PassiveReadError) as captured:
        provider.plan_request(request, operation)
    assert captured.value.reason_code == reason
    assert [name for name, _ in calls].count("full_path") == 1
    assert not store._plans


@pytest.mark.parametrize(
    "mismatch", ["side", "planner", "candidate_frame", "source", "registration"]
)
def test_fast_construction_rejects_mixed_source_native_and_execution_owners(mismatch):
    provider, _, _, _, _, _, _ = _fast_fixture()
    point = provider.point_provider
    adapter = point.arm_reachability_evaluator
    registration = provider.registration
    if mismatch == "side":
        adapter.profile = replace(adapter.profile, arm_side="left")
    elif mismatch == "planner":
        adapter.planner = object()
    elif mismatch == "candidate_frame":
        adapter.candidate_reference_frame_id = "foreign:frame"
    elif mismatch == "source":
        point.registration = replace(
            point.registration,
            profile=replace(
                point.registration.profile, model_profile_hash="foreign:model"
            ),
        )
    else:
        registration = replace(registration, profile_hash="f" * 64)
    with pytest.raises(ValueError, match="differs"):
        RealFastGraspPlanProvider(
            registration=registration,
            point_provider=point,
            paired_provider=provider.paired_provider,
            observation_executor=provider.observation_executor,
        )


def test_fast_pending_or_failed_producer_is_not_execution_authority():
    provider, store, _, request, operation, _, source = _fast_fixture()
    result = provider.plan_request(request, operation).execution_result
    arguments = {
        "plan_id": result["grasp_approach_plan_id"],
        "plan_hash": result["grasp_approach_plan_hash"],
        "operation_id": "motion:test",
        "runtime_session_id": source.runtime_session_id,
        "profile_hash": provider.paired_provider.registration.profile.semantic_content_hash,
        "component_id": "right_arm",
        "expected_station_revision": 7,
        "now": 10.0,
    }
    with pytest.raises(RealGraspApproachPlanLookupError, match="not_committed"):
        store.reserve(**arguments)
    failed = replace(operation, status="cancelled", execution_outcome="cancelled")
    store._planning_operation_lookup = lambda identifier: failed
    with pytest.raises(RealGraspApproachPlanLookupError):
        store.reserve(**arguments)
    assert not store._plans


@pytest.mark.parametrize("transition", ["cancel", "expire"])
def test_fast_publication_rechecks_authoritative_state_after_waiting_for_store_lock(
    transition, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    provider, store, _, request, operation, _, _ = _fast_fixture()
    current, now = [operation], [10.0]
    store._planning_operation_lookup = lambda identifier: current[0]
    store._clock = lambda: now[0]
    reached = threading.Event()
    original = store.put

    def queued(*args, **kwargs):
        reached.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "put", queued)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with store._lock:
            pending = pool.submit(provider.plan_request, request, operation)
            assert reached.wait(timeout=1.0)
            if transition == "cancel":
                current[0] = replace(
                    operation, status="cancelled", execution_outcome="cancelled"
                )
            else:
                now[0] = operation.deadline_at + 1.0
        with pytest.raises(RealGraspApproachPlanLookupError):
            pending.result(timeout=1.0)
    assert not store._plans
