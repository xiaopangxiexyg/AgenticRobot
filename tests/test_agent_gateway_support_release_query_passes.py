"""perception.query_support_release_pose@1: the region and orientation pass loop.

White-box unit tests of ``MujocoSupportReleaseQuery._query`` with a fake
planner and synthetic measurements (no MuJoCo stepping, no camera). They pin
what the module docstring promises: the regions are tried largest first; in
each region the held orientation is tried before the lean; the lean pass runs
only when no held-orientation proposal of that region has a path and is
skipped when it does not fit; refusals are typed and counted over regions; and
running out of time is a budget outcome. Every result must satisfy the Skill's
own result validator.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco import support_release_query as module
from agent_gateway.skills.perception import support_release_query as skill
from robot_motion.object_transport import NoReleaseCandidates
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

pytestmark = [pytest.mark.unit]

FRAME = "mujoco_sim_hil/station"
PARAMETERS = {"support_object_id": "green_box", "reference_frame_id": FRAME}
LEAN = module.RELEASE_LEAN_RAD
PER_PASS = 3


def _pose(x=0., y=0., z=0.):
    pose = np.eye(4)
    pose[:3, 3] = (x, y, z)
    return pose


class _Region:
    """A certified top rectangle as measure_support_tops returns it."""

    def __init__(self, index):
        self.index = index
        self.region_min_xy_m = (-.40 + .02 * index, -.03)
        self.region_max_xy_m = (-.34 + .02 * index, .03)
        self.top_z_m = .042
        self.plane = SimpleNamespace(top_point_count=500, maximum_plane_residual_m=.001)
        self.tilt_deg = .2
        self.coverage = SimpleNamespace(sample_count=400)
        self.observed_min_xy_m = self.region_min_xy_m
        self.observed_max_xy_m = self.region_max_xy_m

    def observation(self, **kwargs):
        return SimpleNamespace(region=self.index, **kwargs)


class _Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class _Planner:
    """Plans a pass's first proposal when its (region, lean) is in ``paths``."""

    def __init__(self, paths=(), *, reason="safe_tcp_target_blocked", clock=None, advance_s=0.):
        self.paths = set(paths)
        self.reason = reason
        self.clock = clock
        self.advance_s = advance_s
        self.calls = []

    def __call__(self, candidates, *, cancel_requested, deadline):
        del cancel_requested, deadline
        if not candidates:
            return None
        key = (candidates[0].region, candidates[0].lean_rad)
        self.calls.append(key)
        if self.clock is not None:
            self.clock.now += self.advance_s
        if key in self.paths:
            return SimpleNamespace(candidate_index=0, diagnostics=[{"reason_code": "path_found"}])
        raise MujocoTcpPlanRejected(self.reason, {"candidates": [
            {"reason_code": self.reason} for _ in candidates]})


def _run(monkeypatch, planner, *, regions=2, fits=lambda region, lean: True, clock=None,
         deadline=45.):
    tops = [_Region(index) for index in range(regions)]
    monkeypatch.setattr(module, "measure_support_tops", lambda depth, mask, **_: tops)
    monkeypatch.setattr(module, "decode_segmentation_mask_npy",
                        lambda *_, **__: np.ones((4, 4), dtype=bool))
    proposed = []

    def candidates(held, support, **kwargs):
        lean = float(kwargs.get("lean_rad", 0.))
        if "lean_rad" in kwargs:
            assert kwargs["lean_toward_xy_m"] == (0., -.2)  # the calibrated arm base
        if not fits(support.region, lean):
            raise NoReleaseCandidates("nothing fits this region in this pass")
        proposed.append((support.region, lean))
        return [SimpleNamespace(
            proposal=SimpleNamespace(object_goal=_pose(-.37, .01 * k, .06),
                                     tool_goal=_pose(-.37, .01 * k, .12)),
            yaw_offset_rad=0., lean_rad=lean, region=support.region) for k in range(PER_PASS)]

    monkeypatch.setattr(module, "plan_release_approach_candidates", candidates)
    query = object.__new__(module.MujocoSupportReleaseQuery)
    query.clock = clock or _Clock()
    query.registration = SimpleNamespace(reference_frame_id=FRAME, roles=("wrist_right",),
                                         prompt_by_support_id={"green_box": "green box"})
    query.application = SimpleNamespace(runtime=SimpleNamespace(clock=lambda: 10.))
    paired = SimpleNamespace(depth=SimpleNamespace(depth_content_hash="a" * 64), rgb=object())
    query.capture_port = SimpleNamespace(
        capture_placement_rgbd=lambda **_: paired,
        capture_aligned_depth=lambda **_: SimpleNamespace(depth_content_hash="b" * 64,
                                                          captured_at=10.))
    prepared = SimpleNamespace(mask_npy=b"", execution_result={
        "image_width": 4, "image_height": 4, "mask_pixel_count": 16,
        "mask_bounds_xyxy": [0, 0, 4, 4], "score": .9, "mask_media_content_hash": "c" * 64})
    query.segmentation_provider = SimpleNamespace(try_segment_capture=lambda **_: prepared)
    query.maximum_depth_m = 2.
    query.held_port = SimpleNamespace(capture_held_box=lambda **_: SimpleNamespace(
        reference_from_object=_pose(-.3, 0., .1), pose_evidence_ref="held-pose",
        grasp_evidence_ref="held-grasp"))
    query.arm_base_xy_m = (0., -.2)
    query._held_verified_without_observing = lambda: True
    query._require_unchanged = lambda *_: None
    query._admissible = lambda cands, held: tuple(range(len(cands)))
    query._plan = planner
    result, evidence = query._query(PARAMETERS, revision=3, cancel_requested=lambda: False,
                                    deadline=deadline)
    skill.validate_result(result, support_object_ids=("green_box",), reference_frame_id=FRAME)
    return result, evidence, proposed


def test_the_held_orientation_of_the_first_region_is_tried_first(monkeypatch):
    planner = _Planner(paths={(0, 0.)})
    result, _evidence, proposed = _run(monkeypatch, planner)
    assert result["status"] == "release_pose_found"
    assert result["release_lean_rad"] == 0.
    assert planner.calls == [(0, 0.)] and proposed == [(0, 0.)]
    assert result["selected_candidate_index"] == 0


def test_the_lean_runs_only_when_no_held_proposal_of_the_region_has_a_path(monkeypatch):
    planner = _Planner(paths={(0, LEAN)})
    result, _evidence, _proposed = _run(monkeypatch, planner)
    assert planner.calls == [(0, 0.), (0, LEAN)]
    assert result["status"] == "release_pose_found"
    assert result["release_lean_rad"] == pytest.approx(LEAN)
    # Indices run over every proposal tried so far: the held pass came first.
    assert result["selected_candidate_index"] == PER_PASS
    assert result["candidate_count"] == 2 * PER_PASS
    assert result["candidate_rejections"] == [
        {"reason_code": "safe_tcp_target_blocked", "count": PER_PASS}]


def test_the_next_region_follows_both_passes_of_the_previous_one(monkeypatch):
    planner = _Planner(paths={(1, 0.)})
    result, _evidence, _proposed = _run(monkeypatch, planner)
    assert planner.calls == [(0, 0.), (0, LEAN), (1, 0.)]
    assert result["release_lean_rad"] == 0.
    assert result["selected_candidate_index"] == 2 * PER_PASS


def test_no_path_in_any_region_is_a_typed_rejection_counted_over_regions(monkeypatch):
    planner = _Planner()
    result, _evidence, _proposed = _run(monkeypatch, planner, regions=3)
    assert planner.calls == [(region, lean) for region in range(3) for lean in (0., LEAN)]
    assert result["status"] == "no_release_pose"
    assert result["reason"] == "no_collision_free_release_path"
    assert result["candidate_count"] == 3 * 2 * PER_PASS
    assert result["candidate_rejections"] == [
        {"reason_code": "safe_tcp_target_blocked", "count": 3 * 2 * PER_PASS}]


def test_a_lean_that_does_not_fit_is_skipped(monkeypatch):
    planner = _Planner()
    result, _evidence, proposed = _run(monkeypatch, planner,
                                       fits=lambda region, lean: lean == 0.)
    assert planner.calls == [(0, 0.), (1, 0.)]
    assert proposed == [(0, 0.), (1, 0.)]
    assert result["reason"] == "no_collision_free_release_path"


def test_nothing_fitting_anywhere_is_a_footprint_rejection(monkeypatch):
    planner = _Planner()
    result, _evidence, proposed = _run(monkeypatch, planner, fits=lambda region, lean: False)
    assert planner.calls == [] and proposed == []
    assert result["reason"] == "held_footprint_does_not_fit_support"


def test_running_out_of_time_before_the_lean_is_a_budget_outcome(monkeypatch):
    clock = _Clock()
    planner = _Planner(clock=clock, advance_s=50.)
    result, _evidence, _proposed = _run(monkeypatch, planner, clock=clock, deadline=45.)
    assert planner.calls == [(0, 0.)]
    assert result["reason"] == "query_budget_exhausted"


def test_an_exhausted_path_search_is_reported_as_a_budget_outcome(monkeypatch):
    planner = _Planner(reason="safe_tcp_path_search_exhausted")
    result, _evidence, _proposed = _run(monkeypatch, planner)
    assert len(planner.calls) == 4
    assert result["reason"] == "query_budget_exhausted"
    assert result["candidate_rejections"] == [
        {"reason_code": "safe_tcp_path_search_exhausted", "count": 4 * PER_PASS}]
