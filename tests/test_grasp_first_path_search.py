from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deployments.perception.grasp_first_path import (
    FirstGraspPathPolicy,
    FirstGraspPathResult,
    GraspPathQueryRejected,
    diverse_grasp_search_order,
    find_first_grasp_path,
    grasp_capture_height_acceptable,
    grasp_capture_support_cost_components,
    grasp_search_preference,
    original_grasp_neighbors,
    refine_first_grasp_path,
    refined_grasp_search_order,
)


def test_advisory_probe_reaches_late_feasible_goal_before_expensive_misses():
    clock = [0.]
    candidates = list(range(20))
    full = []

    def probe(index, budget):
        clock[0] += budget
        return index == 15

    def plan(index, budget):
        full.append(index)
        clock[0] += 3.
        if index != 15:
            raise GraspPathQueryRejected("ik_search_exhausted")
        return "complete_path"

    result = find_first_grasp_path(candidates=candidates, ordered_indices=range(20),
        plan=plan, endpoint_probe=probe, policy=FirstGraspPathPolicy(),
        deadline=10., monotonic=lambda: clock[0])
    assert result.selected_index == 15 and full == [15]
    assert len(result.endpoint_probes) == 16 and result.elapsed_sec < 5.


def test_probe_misses_never_remove_candidates_or_create_a_second_deadline():
    clock = [0.]
    seen = []

    def probe(index, budget):
        clock[0] += budget
        return False

    def plan(index, budget):
        seen.append((index, budget))
        clock[0] += .1
        if index == 0:
            return "full_result"
        raise GraspPathQueryRejected("ik_search_exhausted")

    result = find_first_grasp_path(candidates=[0,1], ordered_indices=[0,1], plan=plan,
        endpoint_probe=probe, policy=FirstGraspPathPolicy(), deadline=1., monotonic=lambda: clock[0])
    assert result.selected_index == 0
    assert seen[0] == pytest.approx((0,.84))
    assert [r['index'] for r in result.endpoint_probes] == [0,1]


def test_probe_pass_is_bounded_and_unknown_goals_keep_normal_full_queries():
    clock = [0.]
    def probe(_index, budget):
        clock[0] += budget
        return False
    def plan(index, _budget):
        clock[0] += .01
        raise GraspPathQueryRejected("ik_search_exhausted")
    policy = FirstGraspPathPolicy(endpoint_probe_total_budget_sec=.16)
    result = find_first_grasp_path(candidates=list(range(6)), ordered_indices=range(6),
        plan=plan, endpoint_probe=probe, policy=policy, deadline=2., monotonic=lambda:clock[0])
    assert len(result.endpoint_probes) == 2
    assert {r['index'] for r in result.attempts} == set(range(6))


def test_probe_exhaustion_restores_unknown_goals_before_full_query_cap():
    clock = [0.]
    def probe(_index,budget):
        clock[0] += budget
        return False
    def plan(index,_budget):
        assert index == 0
        return 'fully_validated'
    result = find_first_grasp_path(candidates=list(range(4)),ordered_indices=range(4),
        plan=plan,endpoint_probe=probe,policy=FirstGraspPathPolicy(
            maximum_path_queries=2,endpoint_probe_total_budget_sec=.16),
        deadline=10.,monotonic=lambda:clock[0])
    assert result.selected_index == 0 and len(result.endpoint_probes) == 2


def test_zero_cost_probes_have_a_finite_count_bound():
    result = find_first_grasp_path(candidates=list(range(100)),ordered_indices=range(100),
        plan=lambda c,b:'path',endpoint_probe=lambda c,b:False,
        policy=FirstGraspPathPolicy(maximum_path_queries=3),deadline=10.,monotonic=lambda:0.)
    assert len(result.endpoint_probes)==3 and result.selected_index==0


@pytest.mark.parametrize('value', [None, 1, 'safe'])
def test_probe_result_must_not_turn_unknown_values_into_success(value):
    with pytest.raises(TypeError, match='advisory bool'):
        find_first_grasp_path(candidates=[0], ordered_indices=[0],
            plan=lambda *_: pytest.fail('No full query after malformed probe'),
            endpoint_probe=lambda *_:value, policy=FirstGraspPathPolicy(), deadline=1., monotonic=lambda:0.)


def test_probe_failure_escapes_and_late_success_is_never_published():
    def broken(*_):
        raise RuntimeError('oracle unavailable')
    with pytest.raises(RuntimeError, match='oracle unavailable'):
        find_first_grasp_path(candidates=[0], ordered_indices=[0], plan=lambda *_:object(),
            endpoint_probe=broken, policy=FirstGraspPathPolicy(), deadline=1., monotonic=lambda:0.)
    clock = [0.]
    def late(*_):
        clock[0] = 2.
        return True
    result = find_first_grasp_path(candidates=[0], ordered_indices=[0],
        plan=lambda *_:pytest.fail('Probe exhausted the shared deadline'), endpoint_probe=late,
        policy=FirstGraspPathPolicy(), deadline=1., monotonic=lambda:clock[0])
    assert result.selected_index is None and result.reason_code == 'grasp_fast_search_budget_exhausted'


def test_capture_support_is_a_soft_shared_search_cost_not_a_hard_gate():
    policy = FirstGraspPathPolicy()
    shallow = SimpleNamespace(capture_support_score=0.0)
    deep = SimpleNamespace(capture_support_score=1.0)
    assert grasp_search_preference(shallow, 0.0, policy=policy) == (0.35, 0.0)
    assert grasp_search_preference(deep, 0.05, policy=policy) < grasp_search_preference(shallow, 0.0, policy=policy)
    assert grasp_search_preference(deep, 0.4, policy=policy) > grasp_search_preference(shallow, 0.0, policy=policy)
    assert grasp_search_preference(SimpleNamespace(), 0.1, policy=policy) == (0.1, 0.1)
    assert grasp_search_preference(shallow, 0.1,
        policy=FirstGraspPathPolicy(capture_support_weight=0.0)) == (0.1, 0.1)
    assert grasp_search_preference(SimpleNamespace(capture_support_score=.5), .1,
        policy=policy) == grasp_search_preference(deep, .1, policy=policy)


@pytest.mark.parametrize("score", [True, 0., -1., 1.01, float("nan"), float("inf"), "0.5"])
def test_capture_support_saturation_requires_a_positive_bounded_score(score):
    with pytest.raises(ValueError, match="capture_support_saturation_score"):
        FirstGraspPathPolicy(capture_support_saturation_score=score)


@pytest.mark.parametrize("score", [True, 0., -1., 1.01, float("nan"), float("inf"), "0.5"])
def test_pad_height_floor_requires_a_positive_bounded_score(score):
    with pytest.raises(ValueError, match="minimum_capture_height_support_score"):
        FirstGraspPathPolicy(minimum_capture_height_support_score=score)


@pytest.mark.parametrize("score", [True, -1., 1.01, float("nan"), float("inf"), "0.5"])
def test_pad_height_observation_rejects_invalid_values(score):
    with pytest.raises(ValueError, match="capture_height_support_score"):
        grasp_capture_height_acceptable(SimpleNamespace(capture_height_support_score=score),
            policy=FirstGraspPathPolicy())


def test_pad_height_floor_is_inclusive_and_old_ports_keep_their_contract():
    policy = FirstGraspPathPolicy()
    assert grasp_capture_height_acceptable(SimpleNamespace(), policy=policy)
    assert grasp_capture_height_acceptable(SimpleNamespace(capture_height_support_score=.5), policy=policy)
    assert not grasp_capture_height_acceptable(SimpleNamespace(capture_height_support_score=.499), policy=policy)


@pytest.mark.parametrize("surface,contact,expected", ((.5, 1., True), (.25, .9, True),
    (.24, 1., False), (0., 1., False), (.5, .49, False)))
def test_local_closing_face_support_can_replace_density_bias_but_not_the_height_floor(
    surface, contact, expected,
):
    candidate = SimpleNamespace(capture_height_support_score=.1,
        surface_alignment_score=surface, contact_height_support_score=contact)
    policy = FirstGraspPathPolicy()
    assert grasp_capture_height_acceptable(candidate, policy=policy) is expected
    assert grasp_capture_support_cost_components(candidate, policy=policy)["pad_height"] == pytest.approx(
        policy.capture_height_support_weight*(1-max(.1, min(contact, 2*surface))))


def test_uncomputed_contact_height_is_unknown_not_a_legacy_height_bypass():
    candidate = SimpleNamespace(capture_height_support_score=.1, contact_height_support_score=1.)
    assert not grasp_capture_height_acceptable(candidate, policy=FirstGraspPathPolicy())


@pytest.mark.parametrize("weight,expected", ((0., True), (2.99, True), (3., False), (20., False)))
def test_observed_closing_height_cannot_be_hidden_by_dense_top(weight, expected):
    candidate = SimpleNamespace(capture_height_support_score=.731,
        surface_alignment_score=.095, contact_height_support_score=.321,
        contact_height_evidence_weight=weight)
    policy = FirstGraspPathPolicy()
    assert grasp_capture_height_acceptable(candidate, policy=policy) is expected
    if weight >= 3:
        assert grasp_capture_support_cost_components(candidate, policy=policy)["pad_height"] == pytest.approx(.35*(1-.321))


@pytest.mark.parametrize("weight", (True, -.1, float("nan"), float("inf"), "3"))
def test_contact_evidence_weight_is_validated(weight):
    from deployments.perception.grasp_first_path import grasp_contact_height_observed
    with pytest.raises(ValueError, match="evidence weight"):
        grasp_contact_height_observed(SimpleNamespace(contact_height_evidence_weight=weight),
            policy=FirstGraspPathPolicy())


@pytest.mark.parametrize("score", (True, -.1, 1.1, float("nan"), "0.5"))
def test_contact_height_rejects_invalid_observation_scores(score):
    with pytest.raises(ValueError, match="support"):
        grasp_capture_height_acceptable(SimpleNamespace(capture_height_support_score=.1,
            surface_alignment_score=.5, contact_height_support_score=score), policy=FirstGraspPathPolicy())


def _neighbor_candidates():
    poses = [np.eye(4) for _ in range(3)]
    poses[1][0, 3] = .2
    poses[2][0, 3] = .01
    return tuple(SimpleNamespace(base_from_tcp=tuple(pose.ravel()), capture_support_score=.7,
        pose_source='dmtac_local_orientation_variant_v1' if i == 0 else 'model_scored')
        for i, pose in enumerate(poses))


def test_original_neighbor_only_promotes_eligible_close_not_worse_support():
    candidates = _neighbor_candidates()
    assert original_grasp_neighbors(candidates, (0, 1, 2), policy=FirstGraspPathPolicy()) == {0: (2,)}
    assert original_grasp_neighbors(candidates, (0, 1), policy=FirstGraspPathPolicy()) == {}
    candidates[2].capture_support_score = .1
    assert original_grasp_neighbors(candidates, (0, 1, 2), policy=FirstGraspPathPolicy()) == {}


def test_original_neighbors_prefer_nearest_pose_and_use_original_order_for_ties():
    candidates = list(_neighbor_candidates())
    farther = np.eye(4)
    farther[0,3] = .04
    candidates[1].base_from_tcp = tuple(farther.ravel())
    candidates.append(SimpleNamespace(**vars(candidates[2])))
    # Index 1 appears earlier, but is farther; equal-distance indices 3 and 2
    # retain their supplied preference rather than numeric index order.
    assert original_grasp_neighbors(candidates, (0, 1, 3, 2),
        policy=FirstGraspPathPolicy()) == {0: (3, 2, 1)}


@pytest.mark.parametrize('reason', ['ik_search_exhausted', 'safe_tcp_endpoint_search_exhausted'])
def test_cheap_derived_endpoint_failure_recovers_original_before_far_family(reason):
    candidates, clock, seen = _neighbor_candidates(), [0.], []
    def query(candidate, budget):
        index = candidates.index(candidate)
        seen.append((index, budget))
        clock[0] += .1
        if index == 0:
            raise GraspPathQueryRejected(reason)
        return object()
    result = find_first_grasp_path(candidates=candidates, ordered_indices=(0, 1, 2),
        plan=query, policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: clock[0],
        original_neighbors={0: (2,)})
    assert result.selected_index == 2 and seen == [(0, 8.), (2, 3.)]
    assert result.attempts[-1]['query_mode'] == 'original_model_neighbor'


@pytest.mark.parametrize('reason,elapsed', [('target_collision', .1), ('ik_search_exhausted', 2.1)])
def test_non_endpoint_or_slow_failure_preserves_family_rotation(reason, elapsed):
    candidates, clock, seen = _neighbor_candidates(), [0.], []
    def query(candidate, budget):
        index = candidates.index(candidate)
        seen.append(index)
        if index == 0:
            clock[0] += elapsed
            raise GraspPathQueryRejected(reason)
        return object()
    result = find_first_grasp_path(candidates=candidates, ordered_indices=(0, 1, 2),
        plan=query, policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: clock[0],
        original_neighbors={0: (2,)})
    assert result.selected_index == 1 and seen == [0, 1]


def test_short_original_probe_does_not_cancel_its_later_full_budget_query():
    candidates, clock, seen = _neighbor_candidates(), [0.], []
    def query(candidate, budget):
        index = candidates.index(candidate)
        seen.append((index, budget))
        clock[0] += min(budget, .1)
        if index == 0:
            raise GraspPathQueryRejected('ik_search_exhausted')
        if index == 1:
            raise GraspPathQueryRejected('target_collision')
        if budget == 3.:
            raise GraspPathQueryRejected('path_planning_time_budget_exhausted')
        return object()
    result = find_first_grasp_path(candidates=candidates, ordered_indices=(0, 1, 2),
        plan=query, policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: clock[0],
        original_neighbors={0: (2,)})
    assert result.selected_index == 2 and seen == [(0, 8.), (2, 3.), (1, 8.), (2, 8.)]


def test_original_probe_consumes_global_query_allowance():
    from dataclasses import replace
    candidates, seen = _neighbor_candidates(), []
    def query(candidate, budget):
        seen.append(candidates.index(candidate))
        raise GraspPathQueryRejected('ik_search_exhausted')
    result = find_first_grasp_path(candidates=candidates, ordered_indices=(0, 1, 2),
        plan=query, policy=replace(FirstGraspPathPolicy(), maximum_path_queries=2),
        deadline=20., monotonic=lambda: 0., original_neighbors={0: (2,)})
    assert result.selected_index is None and seen == [0, 2]


def test_neighbor_probe_allowance_is_shared_across_all_families():
    candidates = tuple(SimpleNamespace(pose_source='derived' if i < 3 else 'model_scored') for i in range(6))
    seen = []
    def query(candidate, budget):
        seen.append((next(i for i,c in enumerate(candidates) if c is candidate), budget))
        raise GraspPathQueryRejected('ik_search_exhausted')
    result = find_first_grasp_path(candidates=candidates, ordered_indices=tuple(range(6)),
        plan=query, policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: 0.,
        original_neighbors={0: (3,), 1: (4,), 2: (5,)})
    assert sum(a.get('query_mode') == 'original_model_neighbor' for a in result.attempts) == 2
    assert seen == [(0,8.),(3,3.),(1,8.),(4,3.),(2,8.),(3,8.),(4,8.),(5,8.)]


@pytest.mark.parametrize('late', [True, False])
def test_original_probe_never_masks_global_expiry_or_planner_fault(late):
    candidates, clock = _neighbor_candidates(), [0.]
    def query(candidate, budget):
        if candidate is candidates[0]:
            raise GraspPathQueryRejected('ik_search_exhausted')
        if not late:
            raise RuntimeError('cancelled or broken planner')
        clock[0] = 20.
        return object()
    def search():
        return find_first_grasp_path(candidates=candidates, ordered_indices=(0, 1, 2),
            plan=query, policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: clock[0],
            original_neighbors={0: (2,)})
    if late:
        result = search()
        assert result.trajectory is None and result.reason_code == 'grasp_fast_search_budget_exhausted'
    else:
        with pytest.raises(RuntimeError, match='planner'):
            search()


@pytest.mark.parametrize('value', [True, -1, 3, .5])
def test_original_neighbor_query_limit_is_fixed_and_bounded(value):
    with pytest.raises(ValueError, match='maximum_original_neighbor_queries'):
        FirstGraspPathPolicy(maximum_original_neighbor_queries=value)


@pytest.mark.parametrize('neighbors', [{True: (2,)}, {0: (1, 1)}, {0: (3,)}, {1: (2,)}, {0: (0,)}])
def test_invalid_original_neighbor_map_fails_before_query(neighbors):
    with pytest.raises(ValueError, match='Original neighbors'):
        find_first_grasp_path(candidates=_neighbor_candidates(), ordered_indices=(0, 1, 2),
            plan=lambda *_: pytest.fail('Invalid neighbor map reached planning'),
            policy=FirstGraspPathPolicy(), deadline=20., original_neighbors=neighbors)


@pytest.mark.parametrize("cost", [True, -1.0, float("nan"), float("inf"), "0.1"])
def test_search_preference_rejects_invalid_motion_cost(cost):
    with pytest.raises(ValueError, match="motion cost"):
        grasp_search_preference(SimpleNamespace(), cost, policy=FirstGraspPathPolicy())


def test_first_path_stops_after_first_complete_success():
    seen = []
    path = object()

    def plan(candidate, remaining):
        seen.append((candidate, remaining))
        if candidate == "blocked":
            raise GraspPathQueryRejected("endpoint_ik_search_exhausted")
        return path

    result = find_first_grasp_path(
        candidates=("blocked", "good", "unused"),
        ordered_indices=(0, 1, 2),
        plan=plan,
        policy=FirstGraspPathPolicy(),
        deadline=20.0,
        monotonic=lambda: 0.0,
    )
    assert result.selected_index == 1 and result.trajectory is path
    assert seen == [("blocked", 8.0), ("good", 8.0)]
    assert [a["reason_code"] for a in result.attempts] == [
        "endpoint_ik_search_exhausted",
        "path_found",
    ]


def test_first_path_does_not_reset_deadline_per_candidate():
    clock = [0.0]
    budgets = []

    def plan(_candidate, remaining):
        budgets.append(remaining)
        clock[0] += remaining
        raise GraspPathQueryRejected("path_planning_time_budget_exhausted")

    result = find_first_grasp_path(
        candidates=(0, 1, 2),
        ordered_indices=(0, 1, 2),
        plan=plan,
        policy=FirstGraspPathPolicy(),
        deadline=10.0,
        monotonic=lambda: clock[0],
    )
    assert budgets == [8.0, 2.0]
    assert result.reason_code == "grasp_fast_search_budget_exhausted"
    assert result.trajectory is None and result.selected_index is None


def test_first_path_cannot_publish_a_result_after_its_deadline():
    clock = [0.0]

    def plan(*_args):
        clock[0] = 3.0
        return object()

    result = find_first_grasp_path(
        candidates=(0,),
        ordered_indices=(0,),
        plan=plan,
        policy=FirstGraspPathPolicy(),
        deadline=2.0,
        monotonic=lambda: clock[0],
    )
    assert result.trajectory is None
    assert result.reason_code == "grasp_fast_search_budget_exhausted"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("corrupt pose"),
        RuntimeError("cancelled"),
        TypeError("broken planner"),
    ],
)
def test_first_path_does_not_treat_planner_or_input_faults_as_another_goal(error):
    def plan(*_args):
        raise error

    with pytest.raises(type(error), match=str(error)):
        find_first_grasp_path(
            candidates=(0, 1),
            ordered_indices=(0, 1),
            plan=plan,
            policy=FirstGraspPathPolicy(),
            deadline=2.0,
            monotonic=lambda: 0.0,
        )


def test_first_path_bounds_queries_even_with_clock_not_advancing():
    seen = []

    def plan(candidate, _remaining):
        seen.append(candidate)
        raise GraspPathQueryRejected("motion_quality_exceeded")

    result = find_first_grasp_path(
        candidates=tuple(range(10)),
        ordered_indices=tuple(range(10)),
        plan=plan,
        policy=FirstGraspPathPolicy(maximum_path_queries=2),
        deadline=20.0,
        monotonic=lambda: 0.0,
    )
    assert seen == [0, 1]
    assert result.reason_code == "grasp_fast_no_acceptable_path"


def test_default_coverage_does_not_stop_after_sixteen_quick_endpoint_rejections():
    clock = [0.0]
    seen = []
    path = object()

    def plan(candidate, remaining):
        assert 0 < remaining <= 8.0
        seen.append(candidate)
        clock[0] += 0.04
        if candidate < 16:
            raise GraspPathQueryRejected("safe_tcp_endpoint_search_exhausted")
        return path

    result = find_first_grasp_path(
        candidates=tuple(range(70)), ordered_indices=tuple(range(70)),
        plan=plan, policy=FirstGraspPathPolicy(), deadline=20.0,
        monotonic=lambda: clock[0],
    )
    assert result.trajectory is path and result.selected_index == 16
    assert seen == list(range(17))
    assert result.elapsed_sec < 1.0


@pytest.mark.parametrize("indices", [(0, 0), (-1,), (2,), (True,)])
def test_first_path_rejects_corrupt_order(indices):
    with pytest.raises(ValueError, match="indices"):
        find_first_grasp_path(
            candidates=(0, 1),
            ordered_indices=indices,
            plan=lambda *_: object(),
            policy=FirstGraspPathPolicy(),
            deadline=20.0,
        )


def test_first_path_needs_an_actual_trajectory_not_an_empty_success():
    with pytest.raises(TypeError, match="trajectory"):
        find_first_grasp_path(
            candidates=(0,),
            ordered_indices=(0,),
            plan=lambda *_: None,
            policy=FirstGraspPathPolicy(),
            deadline=20.0,
            monotonic=lambda: 0.0,
        )


def test_search_covers_distinct_orientations_before_near_duplicate_depths():
    first = np.eye(4)
    near = first.copy()
    near[2, 3] = 0.01
    opposite_closing_axis = np.diag([-1.0, -1.0, 1.0, 1.0])
    far = first.copy()
    far[0, 3] = 0.2
    poses = (first, near, opposite_closing_axis, far)
    assert diverse_grasp_search_order(
        poses, (0, 1, 2, 3), policy=FirstGraspPathPolicy()
    ) == (0, 2, 3, 1)
    np.testing.assert_array_equal(poses[0], np.eye(4))


@pytest.mark.parametrize("value", [True, None, "20", float("nan"), -1.0])
def test_search_policy_rejects_non_numeric_and_invalid_weights(value):
    with pytest.raises(ValueError):
        FirstGraspPathPolicy(orientation_weight=value)


def test_empty_diverse_search_order_is_valid():
    assert diverse_grasp_search_order((), (), policy=FirstGraspPathPolicy()) == ()


def test_axial_child_keeps_parent_motion_priority_and_failed_child_uses_original():
    poses = [np.eye(4) for _ in range(4)]
    poses[1][0, 3] = .2
    poses[2][2, 3] = .02
    poses[3][0, 3] = .2
    poses[3][2, 3] = .02
    candidates = tuple(SimpleNamespace(base_from_tcp=pose, axial_parent_index=parent)
        for pose, parent in zip(poses, (None, None, 0, 1)))
    # Child 3 has the best new score, but cannot promote the farther parent 1.
    order = refined_grasp_search_order(candidates, (3, 0, 2, 1), policy=FirstGraspPathPolicy())
    assert order == (0, 1)
    seen = []
    def query(candidate, budget):
        seen.append((next(i for i, c in enumerate(candidates) if c is candidate), budget))
        if candidate.axial_parent_index is not None:
            raise GraspPathQueryRejected("target_collision")
        return object()
    result = find_first_grasp_path(candidates=candidates, ordered_indices=order, plan=query,
        policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: 0.)
    result = refine_first_grasp_path(result, candidates=candidates, eligible_indices=(0, 1, 2, 3),
        plan=lambda c, b, _: query(c, b), policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: 0.)
    assert result.selected_index == 0 and seen == [(0, 8.), (2, 3.)]
    # Refinement never adds primary queries for an ineligible parent.
    assert refined_grasp_search_order(candidates, (2,), policy=FirstGraspPathPolicy()) == ()


@pytest.mark.parametrize("parent", [True, -1, 2, 1])
def test_axial_lineage_cannot_reference_itself_or_a_missing_parent(parent):
    candidates = (SimpleNamespace(base_from_tcp=np.eye(4)),
        SimpleNamespace(base_from_tcp=np.eye(4), axial_parent_index=parent))
    with pytest.raises(ValueError, match="parent"):
        refined_grasp_search_order(candidates, (0, 1), policy=FirstGraspPathPolicy())


def test_axial_query_and_parent_share_one_deadline_and_query_count():
    candidates = (SimpleNamespace(axial_parent_index=1), SimpleNamespace())
    clock, seen = [0.], []
    def query(candidate, budget):
        seen.append(budget)
        clock[0] += budget
        raise GraspPathQueryRejected("path_planning_time_budget_exhausted")
    result = find_first_grasp_path(candidates=candidates, ordered_indices=(0, 1), plan=query,
        policy=FirstGraspPathPolicy(), deadline=4., monotonic=lambda: clock[0])
    assert seen == [3., 1.] and result.selected_index is None
    assert result.reason_code == "grasp_fast_search_budget_exhausted"


@pytest.mark.parametrize("remaining,queries", [(3., 1), (20., 64)])
def test_refinement_never_gambles_a_late_success_or_creates_new_query_budget(remaining, queries):
    original = object()
    result = FirstGraspPathResult(0, original, tuple({"index": 0} for _ in range(queries)), "path_found", 1.)
    candidates = (SimpleNamespace(), SimpleNamespace(axial_parent_index=0))
    returned = refine_first_grasp_path(result, candidates=candidates, eligible_indices=(0, 1),
        plan=lambda *_: pytest.fail("No remaining refinement allowance"),
        policy=FirstGraspPathPolicy(), deadline=remaining, monotonic=lambda: 0.)
    assert returned is result


def test_refinement_cannot_keep_an_executable_result_after_global_deadline():
    clock = [0.]
    original = FirstGraspPathResult(0, object(), ({"index": 0},), "path_found", 1.)
    def query(*_):
        clock[0] = 20.
        return object()
    result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),
        SimpleNamespace(axial_parent_index=0)), eligible_indices=(0, 1), plan=query,
        policy=FirstGraspPathPolicy(), deadline=20., monotonic=lambda: clock[0])
    assert result.selected_index is result.trajectory is None
    assert result.reason_code == "grasp_fast_search_budget_exhausted"


def test_refinement_passes_absolute_deadline_so_preparation_cannot_spend_publication_reserve():
    original = FirstGraspPathResult(0, object(), ({"index": 0},), "path_found", 1.)
    clock = [0.]
    def query(_candidate, budget, deadline):
        assert budget == deadline == 3.
        clock[0] += .7
        assert deadline - clock[0] == pytest.approx(2.3)
        clock[0] = deadline
        raise GraspPathQueryRejected("path_planning_time_budget_exhausted")
    result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),
        SimpleNamespace(axial_parent_index=0)), eligible_indices=(0, 1), plan=query,
        policy=FirstGraspPathPolicy(), deadline=3.6, monotonic=lambda: clock[0])
    assert result.trajectory is original.trajectory and result.selected_index == 0
    assert 3.6 - clock[0] >= .5


def _phase15_clock(*samples):
    values = iter(samples)
    return lambda: next(values, samples[-1])


@pytest.mark.parametrize("early_return", ("query_cap", "headroom", "no_child"))
@pytest.mark.parametrize("expired", (False, True))
def test_phase15_refinement_checks_expiry_on_every_early_return(early_return, expired):
    count = 64 if early_return == "query_cap" else 1
    original = FirstGraspPathResult(0, object(), tuple({"index": 0} for _ in range(count)),
        "path_found", 1.)
    deadline = 1. if early_return == "headroom" else 10.
    now = deadline if expired else 0.
    seen = []
    result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),
        SimpleNamespace(axial_parent_index=0)),
        eligible_indices=(0,) if early_return == "no_child" else (0, 1),
        plan=lambda *args: seen.append(args), policy=FirstGraspPathPolicy(),
        deadline=deadline, monotonic=lambda: now)
    assert seen == []
    if expired:
        assert result.trajectory is result.selected_index is None
        assert result.reason_code == "grasp_fast_search_budget_exhausted"
        assert result.attempts == original.attempts
    else:
        assert result is original


def test_phase15_refinement_rechecks_expiry_after_child_selection():
    original = FirstGraspPathResult(0, object(), ({"index": 0},), "path_found", 1.)
    result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),),
        eligible_indices=(0,), plan=lambda *_: pytest.fail("No eligible child"),
        policy=FirstGraspPathPolicy(), deadline=10., monotonic=_phase15_clock(0., 10.))
    assert result.trajectory is result.selected_index is None
    assert result.reason_code == "grasp_fast_search_budget_exhausted"


@pytest.mark.parametrize("entry", ("full", "probe", "refinement"))
def test_phase15_callbacks_do_not_start_when_clock_crosses_deadline(entry):
    seen = []
    if entry == "refinement":
        original = FirstGraspPathResult(0, object(), ({"index": 0},), "path_found", 1.)
        result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),
            SimpleNamespace(axial_parent_index=0)), eligible_indices=(0, 1),
            plan=lambda *args: seen.append(args) or object(), policy=FirstGraspPathPolicy(),
            deadline=10., monotonic=_phase15_clock(0., 10.))
    else:
        result = find_first_grasp_path(candidates=(0,), ordered_indices=(0,),
            plan=lambda *args: seen.append(("plan", args)) or object(),
            endpoint_probe=(lambda *args: seen.append(("probe", args)) or True)
                if entry == "probe" else None,
            policy=FirstGraspPathPolicy(), deadline=1., monotonic=_phase15_clock(0., 0., 1.))
    assert seen == []
    assert result.selected_index is result.trajectory is None
    assert result.reason_code == "grasp_fast_search_budget_exhausted"


@pytest.mark.parametrize("entry", ("full", "probe", "refinement"))
def test_phase15_callback_budget_uses_fresh_clock_without_resetting_deadline(entry):
    seen = []
    if entry == "refinement":
        original = FirstGraspPathResult(0, object(), ({"index": 0},), "path_found", 1.)
        result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),
            SimpleNamespace(axial_parent_index=0)), eligible_indices=(0, 1),
            plan=lambda candidate, budget, deadline: seen.append((budget, deadline)) or object(),
            policy=FirstGraspPathPolicy(), deadline=10., monotonic=_phase15_clock(0., 2.))
        assert result.selected_index == 1
        assert seen == [(1., 3.)]
    else:
        def probe(_candidate, budget):
            seen.append(budget)
            return True
        result = find_first_grasp_path(candidates=(0,), ordered_indices=(0,),
            plan=lambda candidate, budget: seen.append(budget) or object(),
            endpoint_probe=probe if entry == "probe" else None,
            policy=FirstGraspPathPolicy(endpoint_probe_budget_sec=.8), deadline=1.,
            monotonic=_phase15_clock(0., .1, .7))
        assert result.selected_index == 0
        assert seen == pytest.approx([.3, .3] if entry == "probe" else [.3])


def test_phase15_expired_refinement_slice_keeps_still_valid_original():
    original = FirstGraspPathResult(0, object(), ({"index": 0},), "path_found", 1.)
    seen = []
    result = refine_first_grasp_path(original, candidates=(SimpleNamespace(),
        SimpleNamespace(axial_parent_index=0)), eligible_indices=(0, 1),
        plan=lambda *args: seen.append(args) or object(), policy=FirstGraspPathPolicy(),
        deadline=10., monotonic=_phase15_clock(0., 3.1))
    assert seen == [] and result is original


@pytest.mark.parametrize("overflow", ("support", "combined"))
def test_phase15_ranking_rejects_nonfinite_aggregate_from_finite_inputs(overflow):
    candidate = SimpleNamespace(capture_support_score=0., capture_height_support_score=0.,
        opposed_capture_support_score=0., surface_alignment_score=0.)
    policy = FirstGraspPathPolicy(capture_support_weight=1e308,
        capture_height_support_weight=1e308 if overflow == "support" else 0.,
        opposed_capture_support_weight=1e308 if overflow == "support" else 0.,
        surface_alignment_weight=1e308 if overflow == "support" else 0.)
    with pytest.raises(ValueError, match="finite"):
        grasp_search_preference(candidate, 0. if overflow == "support" else 1e308, policy=policy)


def test_phase15_large_finite_ranking_cost_is_preserved():
    candidate = SimpleNamespace(capture_support_score=0.)
    policy = FirstGraspPathPolicy(capture_support_weight=1e308)
    assert grasp_search_preference(candidate, 0., policy=policy) == (1e308, 0.)
