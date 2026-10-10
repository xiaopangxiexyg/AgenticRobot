from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real import point_grasp_render_gateway, realsense_perception
from agent_gateway.profiles.real.perception_application import (
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.profiles.real.perception_deployment import (
    RealPerceptionCaptureProfile,
)
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.point_grasp_gateway import (
    StationBoundPointGraspProvider,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)
from agent_gateway.profiles.real.rm65_grasp_reachability import (
    RM65PointGraspReachabilityAdapter,
)
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
    StationPointGraspArmReachabilityProfile,
    build_station_bound_point_grasp_registration,
)
from agent_gateway.profiles.real.station_point_grasp_render_deployment import (
    REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1,
    build_station_bound_point_grasp_render_registration,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
)
from deployments.perception.sam2_graspnet_point_grasp import (
    _normalize_graspnet_rigid_matrix,
)
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectSegment,
    encode_grounded_object_segment_artifacts,
)
from deployments.perception.sam2_vgn_point_grasp import (
    PointGraspCandidate,
    PointGraspResult,
)
from teleop_runtime.video_frame import VideoFrame


def test_offline_reachability_never_invents_a_live_qpos_cut():
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.motion_port = None
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200)
    with pytest.raises(RuntimeError, match="capture-matched"):
        adapter._resolve_measured_qpos(measured_qpos_cut=None, measurement_reference_time=10.)
    cut = SimpleNamespace(measured_at=9.95)
    resolved, age = adapter._resolve_measured_qpos(measured_qpos_cut=cut, measurement_reference_time=10.)
    assert resolved is cut
    assert age == pytest.approx(50.)
    with pytest.raises(RuntimeError, match="camera-matched"):
        adapter._resolve_measured_qpos(measured_qpos_cut=cut, measurement_reference_time=11.)


def test_embodied_shortlist_reports_bounded_rejection_reasons() -> None:
    pose = np.eye(4)
    pose[2, 2] = -1.0
    pose[0, 0] = -1.0
    candidates = tuple(
        SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=0.9 - i * 0.1)
        for i in range(3)
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    reasons = iter(("approach_endpoint_collision_blocked", None, "approach_ik_search_exhausted"))

    def assess(**_kwargs):
        reason = next(reasons)
        return SimpleNamespace(
            feasible=reason is None,
            selected_pregrasp_distance_m=0.08 if reason is None else None,
            elapsed_ms=25.0,
            reason_code=reason,
            attempts=({"reason_code": reason},) if reason else (),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        candidates,
        (0.0, 0.0, 0.0),
        3,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert result.selected_indices == (1,)
    attempts = json.loads(result.diagnostics["arm_shortlist_attempts_json"])
    assert len(attempts) == 3
    assert attempts[0]["reason_code"] == "approach_endpoint_collision_blocked"
    assert attempts[0]["branch_attempts"] == [
        {"reason_code": "approach_endpoint_collision_blocked"}
    ]
    assert json.loads(result.diagnostics["arm_shortlist_rejections_json"]) == {
        "approach_endpoint_collision_blocked": 1,
        "approach_ik_search_exhausted": 1,
    }
    assert result.diagnostics["arm_shortlist_rescue_activated"] is False


def test_embodied_shortlist_preserves_tilted_pose_and_direction_diversity() -> None:
    candidates = []
    poses = []
    for index, tilt_deg in enumerate((-35.0, -33.0, 5.0)):
        tilt = np.deg2rad(tilt_deg)
        approach = np.array((-np.cos(tilt), 0.0, np.sin(tilt)))
        pose = np.eye(4)
        pose[:3, 0] = np.cross((0.0, 1.0, 0.0), approach)
        pose[:3, 2] = approach
        pose[0, 3] = float(index)
        poses.append(pose.copy())
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=0.9 - index * 0.1))

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value.copy()
    assessed = []

    def assess(**kwargs):
        pose = kwargs["target_tool_tcp_world"]
        index = int(pose[0, 3])
        np.testing.assert_allclose(pose, poses[index])
        assessed.append(index)
        return SimpleNamespace(
            feasible=True, selected_pregrasp_distance_m=0.06,
            elapsed_ms=1.0, reason_code=None, attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        tuple(candidates), (0.0, 0.0, 0.0), 2,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    # The 40-degree-separated approach must precede a near duplicate.
    assert assessed == [0, 2]
    assert result.selected_indices == (0, 2)
    for candidate, pose in zip(candidates, poses, strict=True):
        np.testing.assert_array_equal(candidate.base_from_tcp, pose)


@pytest.mark.parametrize("classification", ["unresolved", "not_probed"])
def test_embodied_shortlist_explores_unknown_before_known_collision(
    monkeypatch, classification
) -> None:
    from agent_gateway.profiles.real import rm65_grasp_reachability as reachability

    if classification == "not_probed":
        monkeypatch.setattr(reachability, "_ARM_SHORTLIST_PROBE_MAX_COUNT", 1)
    candidates = []
    for index in range(2):
        pose = np.diag((-1.0, 1.0, -1.0, 1.0))
        pose[0, 3] = float(index)
        candidates.append(SimpleNamespace(base_from_tcp=tuple(pose.flat), score=0.9 - index * 0.1))
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value.copy()
    assessed = []

    def probe(**kwargs):
        index = int(kwargs["target_tool_tcp_world"][0, 3])
        return SimpleNamespace(
            classification="collision_blocked" if index == 0 else classification,
            elapsed_ms=1.0, reason_code=None,
            best_minimum_distance_m=None, attempted_seed_count=1,
        )

    def assess(**kwargs):
        index = int(kwargs["target_tool_tcp_world"][0, 3])
        assessed.append(index)
        return SimpleNamespace(
            feasible=index == 1, selected_pregrasp_distance_m=0.06,
            elapsed_ms=1.0, reason_code=None if index == 1 else "endpoint_collision_blocked",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=probe, assess_local_grasp_chain=assess,
    )
    result = adapter.select_candidates(
        tuple(candidates), (0.0, 0.0, 0.0), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert assessed == [1]
    assert result.selected_indices == (1,)


def test_embodied_shortlist_uses_endpoint_probe_only_to_reorder_full_checks() -> None:
    candidates = []
    for index in range(3):
        pose = np.diag((-1.0, 1.0, -1.0, 1.0))
        pose[0, 3] = float(index)
        candidates.append(
            SimpleNamespace(
                base_from_tcp=tuple(pose.reshape(-1)),
                score=0.9 - index * 0.1,
            )
        )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    probed = []
    assessed = []
    classifications = {
        0: "collision_blocked",
        1: "clearance_blocked",
        2: "clearance_safe",
    }

    def probe(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        probed.append(index)
        return SimpleNamespace(
            classification=classifications[index],
            elapsed_ms=1.0,
            reason_code=None,
            best_minimum_distance_m=0.03 if index == 2 else 0.003,
            attempted_seed_count=2,
        )

    def assess(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        assessed.append(index)
        return SimpleNamespace(
            feasible=index == 2,
            selected_pregrasp_distance_m=0.06 if index == 2 else None,
            elapsed_ms=2.0,
            reason_code=None if index == 2 else "ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=probe,
        assess_local_grasp_chain=assess,
    )
    result = adapter.select_candidates(
        tuple(candidates),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    assert probed == [0, 1, 2]
    assert assessed == [2]
    assert result.selected_indices == (2,)
    assert result.diagnostics["arm_shortlist_endpoint_probe_count"] == 3


def test_embodied_shortlist_probes_support_clearance_family_before_model_scores() -> None:
    candidates = []
    for index in range(6):
        pose = np.diag((-1.0, 1.0, -1.0, 1.0))
        pose[0, 3] = float(index)
        candidates.append(
            SimpleNamespace(
                base_from_tcp=tuple(pose.reshape(-1)),
                score=0.9 - index * 0.1,
                pose_source=(
                    "dmtac_support_clearance_variant_v1"
                    if index == 5
                    else "model_scored"
                ),
            )
        )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(
        max_measured_qpos_age_ms=200.0,
        arm_side="right",
    )
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    probed = []
    assessed = []

    def probe(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        probed.append(index)
        return SimpleNamespace(
            classification=("clearance_safe" if index == 5 else "collision_blocked"),
            elapsed_ms=1.0,
            reason_code=None,
            best_minimum_distance_m=0.03 if index == 5 else -0.01,
            attempted_seed_count=2,
        )

    def assess(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        assessed.append(index)
        return SimpleNamespace(
            feasible=index == 5,
            selected_pregrasp_distance_m=0.06 if index == 5 else None,
            elapsed_ms=2.0,
            reason_code=None if index == 5 else "endpoint_collision_blocked",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=probe,
        assess_local_grasp_chain=assess,
    )

    result = adapter.select_candidates(
        tuple(candidates),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    assert probed[0] == 5
    assert assessed == [5]
    assert result.selected_indices == (5,)


def test_embodied_shortlist_does_not_reject_an_unresolved_endpoint_probe() -> None:
    pose = np.diag((-1.0, 1.0, -1.0, 1.0))
    candidate = SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=0.9)
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=lambda **_kwargs: SimpleNamespace(
            classification="unresolved",
            elapsed_ms=1.0,
            reason_code="path_planning_time_budget_exhausted",
            best_minimum_distance_m=None,
            attempted_seed_count=1,
        ),
        assess_local_grasp_chain=lambda **_kwargs: SimpleNamespace(
            feasible=True,
            selected_pregrasp_distance_m=0.04,
            elapsed_ms=2.0,
            reason_code=None,
            attempts=(),
        ),
    )

    result = adapter.select_candidates(
        (candidate,),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    assert result.selected_indices == (0,)


def test_embodied_shortlist_rescues_late_feasible_pose_after_extreme_scene_budget() -> None:
    """The recorded 8-second shortlist inspected only six of 66-97 poses."""

    pose = np.eye(4)
    pose[0, 0] = -1.0
    pose[2, 2] = -1.0
    candidates = tuple(
        SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=0.9 - i * 0.01)
        for i in range(10)
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    elapsed = [0.0]
    adapter.monotonic = lambda: elapsed[0]
    adapter._solver_pose = lambda value: value
    # First five costs mirror the failed field run; the sixth has only a small
    # remainder under the old total deadline. The seventh pose is a valid
    # local chain, analogous to the later successful grasp on the same qpos.
    durations = (1.009, 1.662, 2.016, 1.068, 2.009, 0.238, 0.174)
    budgets = []

    def assess(**kwargs):
        index = len(budgets)
        budgets.append(kwargs["time_budget_sec"])
        elapsed[0] += durations[index]
        feasible = index == 6
        return SimpleNamespace(
            feasible=feasible,
            selected_pregrasp_distance_m=0.06 if feasible else None,
            elapsed_ms=durations[index] * 1000.0,
            reason_code=None if feasible else "ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        candidates,
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert result.selected_indices == (6,)
    assert len(budgets) == 7
    assert all(0 < value <= 2.0 for value in budgets)
    assert result.diagnostics["arm_shortlist_rescue_activated"] is True
    assert result.diagnostics["arm_shortlist_rescue_assessment_count"] == 2
    assert result.diagnostics["arm_shortlist_budget_sec"] == 16.0


def test_embodied_shortlist_rescue_stops_at_total_deadline() -> None:
    pose = np.diag((-1.0, 1.0, -1.0, 1.0))
    candidates = tuple(
        SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=1.0 - i * 0.01)
        for i in range(20)
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    elapsed = [0.0]
    adapter.monotonic = lambda: elapsed[0]
    adapter._solver_pose = lambda value: value
    budgets = []

    def assess(**kwargs):
        budgets.append(kwargs["time_budget_sec"])
        elapsed[0] += 2.0
        return SimpleNamespace(
            feasible=False,
            selected_pregrasp_distance_m=None,
            elapsed_ms=2000.0,
            reason_code="ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        candidates,
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert result.selected_indices == ()
    assert len(budgets) == 8
    assert elapsed[0] == 16.0
    assert result.diagnostics["arm_shortlist_budget_exhausted"] is True
    assert result.diagnostics["arm_shortlist_rescue_assessment_count"] == 4


def test_embodied_shortlist_checks_nearby_pose_before_spending_first_window() -> None:
    approaches = (
        (-1.0, 0.0, 0.0),
        (-0.8660254, -0.5, 0.0),
        (-0.5, -0.8660254, 0.0),
        (-0.8660254, 0.5, 0.0),
        (-0.5, 0.8660254, 0.0),
        (-1.0, 0.0, 0.0),
    )
    candidates = []
    for index, approach in enumerate(approaches):
        pose = np.eye(4)
        pose[:3, 2] = approach
        pose[0, 3] = index
        candidates.append(
            SimpleNamespace(
                base_from_tcp=tuple(pose.reshape(-1)),
                score=0.2 if index == 5 else 0.9 - index * 0.01,
            )
        )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    elapsed = [0.0]
    adapter.monotonic = lambda: elapsed[0]
    adapter._solver_pose = lambda value: value
    assessed = []

    def assess(**kwargs):
        index = int(kwargs["target_tool_tcp_world"][0, 3])
        assessed.append(index)
        elapsed[0] += 1.0
        feasible = index == 5
        return SimpleNamespace(
            feasible=feasible,
            selected_pregrasp_distance_m=0.06 if feasible else None,
            elapsed_ms=1000.0,
            reason_code=None if feasible else "ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        tuple(candidates),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert assessed == [0, 1, 3, 2, 5]
    assert result.selected_indices == (5,)
    assert result.diagnostics["arm_shortlist_rescue_activated"] is False


class _Source:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            camera_key="front",
            width=4,
            height=3,
            depth_width=4,
            depth_height=3,
            align_depth_to_color=True,
            enable_depth=True,
            color_space="rgb",
            serial="front-serial",
        )
        self.read_count = 0
        self.frame = VideoFrame(
            image=np.full((3, 4, 3), 32, dtype=np.uint8),
            source_timestamp=9.95,
            receive_timestamp=9.951,
            available_timestamp=9.952,
            frame_index=11,
            metadata={
                "camera_key": "front",
                "serial": "front-serial",
                "source_generation": 2,
            },
            depth_image=np.full((3, 4), 500, dtype=np.uint16),
            depth_metadata={
                "aligned_to": "color",
                "depth_scale_m_per_unit": 0.001,
            },
        )

    def get_latest_frame(self):
        self.read_count += 1
        return self.frame

    def device_info(self):
        return {
            "running": True,
            "lifecycle_state": "running",
            "worker_alive": True,
            "serial_number": "front-serial",
            "generation": 2,
            "intrinsics": {
                "width": 4,
                "height": 3,
                "fx": 4.0,
                "fy": 3.0,
                "ppx": 2.0,
                "ppy": 1.5,
            },
        }

    def close(self) -> None:
        return None


class _Planner:
    backend_id = "vgn"

    def __init__(self, point_segmenter: object, revision=None, events=None) -> None:
        self.point_segmenter = point_segmenter
        self.model_spec = SimpleNamespace(
            model_profile_id="vgn:test",
            semantic_content_hash="vgn-model:test",
        )
        self.semantic_content_hash = "station-point-grasp-planner:test"
        self.revision = revision
        self.events = events
        self.starts = 0
        self.closes = 0
        self.base_from_camera = None
        self.visible_scene_factory = None

    def start(self) -> None:
        self.starts += 1
        if self.events is not None:
            self.events.append("planner.start")

    def close(self) -> None:
        self.closes += 1
        if self.events is not None:
            self.events.append("planner.close")

    def plan(
        self,
        image,
        depth_m,
        *,
        pixel_uv,
        intrinsics,
        base_from_camera,
        max_candidates,
        visible_scene_factory=None,
    ) -> PointGraspResult:
        assert image.size == (4, 3)
        assert depth_m.shape == (3, 4)
        assert pixel_uv == (1, 1)
        assert intrinsics["width"] == 4
        assert max_candidates == 2
        self.base_from_camera = np.asarray(base_from_camera).copy()
        self.visible_scene_factory = visible_scene_factory
        camera_from_tcp = np.eye(4)
        camera_from_tcp[:3, 3] = (0.0, 0.0, 0.5)
        reference_from_tcp = self.base_from_camera @ camera_from_tcp
        target_camera = np.asarray((0.0, 0.0, 0.5, 1.0))
        target_reference = self.base_from_camera @ target_camera
        if self.revision is not None:
            self.revision["value"] += 1
        return PointGraspResult(
            target_point_camera_m=tuple(target_camera[:3]),
            target_point_base_m=tuple(target_reference[:3]),
            target_mask_bbox_xyxy=(0, 0, 4, 3),
            candidates=(
                PointGraspCandidate(
                    score=0.9,
                    model_score=0.95,
                    click_distance_m=0.01,
                    gripper_width_m=0.04,
                    camera_from_tcp=tuple(camera_from_tcp.reshape(-1)),
                    base_from_tcp=tuple(reference_from_tcp.reshape(-1)),
                    visible_collision_free=True,
                ),
            ),
        )


class _Localizer:
    def __init__(self, events=None) -> None:
        self.model_spec = SimpleNamespace(
            model_profile_id="sam2:test",
            semantic_content_hash="sam2-model:test",
        )
        self.semantic_content_hash = "sam2-localizer:test"
        self.events = events

    def start(self) -> None:
        if self.events is not None:
            self.events.append("localizer.start")

    def close(self) -> None:
        if self.events is not None:
            self.events.append("localizer.close")

    def localize(self, *_args, **_kwargs):
        return ()

    def segment_point(self, image, *, pixel_uv):
        mask = np.ones((image.height, image.width), dtype=bool)
        assert mask[pixel_uv[1], pixel_uv[0]]
        return mask

    def segment(
        self, image, *, language_prompt, pixel_uv, min_score
    ) -> GroundedObjectSegment:
        mask = np.ones((image.height, image.width), dtype=bool)
        return GroundedObjectSegment(
            mask=mask,
            score=max(0.9, float(min_score)),
            label=language_prompt,
            box_xyxy=(0, 0, image.width, image.height),
        )

    encode_segment_artifacts = staticmethod(encode_grounded_object_segment_artifacts)


def _frame_binding():
    station_from_source = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return build_real_perception_station_frame_binding(
        collision_scene_profile="station-point-grasp-test",
        scene_mount_hash="station-point-grasp-mounts:test",
        scene_world_from_canonical_station=np.linalg.inv(station_from_source),
    )


def _provider_stack(
    *,
    revision=None,
    arm_reachability_evaluator=None,
    backend_id="vgn",
    captured_robot_observation_filter=None,
):
    source = _Source()
    binding = RealPerceptionSourceBinding(
        camera_sources={"front": source},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:station-point-grasp-test",
    )
    revision = {"value": 7} if revision is None else revision
    perception = RealSenseAgentPerceptionProvider(
        source_binding=binding,
        current_station_revision=lambda: revision["value"],
        clock=lambda: 10.0,
    )
    perception.bind_profile(
        RealPerceptionCaptureProfile(
            rgb_sources=(("front", "main", 4, 3),),
            depth_sources=(("front", "main", 4, 3),),
            source_binding_hash=binding.semantic_content_hash,
        )
    )
    frame_binding = _frame_binding()
    localizer = _Localizer()
    planner = _Planner(localizer, revision=revision if revision.get("mutate") else None)
    planner.backend_id = backend_id
    if backend_id == "graspgenx":
        from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry

        planner.last_target_mask = np.ones((3, 4), dtype=bool)
        planner.solid_geometry = ParallelJawSolidGeometry(
            -0.043, 0.05, 0.01, 0.0255, 0.07
        )
    reachability_profile = (
        None
        if arm_reachability_evaluator is None
        else StationPointGraspArmReachabilityProfile(
            arm_side="right",
            evaluator_profile_id="rm65.reachability:test",
            evaluator_profile_hash="rm65-reachability-profile:test",
            solver_reference_frame_id=frame_binding.source_frame_id,
            solver_reference_frame_spec_hash=frame_binding.source_frame_spec_hash,
            frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        )
    )
    registration = build_station_bound_point_grasp_registration(
        aligned_rgbd_source_catalog=tuple(
            dict(item) for item in perception.aligned_rgbd_source_catalog()
        ),
        source_binding_hash=binding.semantic_content_hash,
        frame_binding_hash=frame_binding.semantic_content_hash,
        frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        source_frame_id=frame_binding.source_frame_id,
        source_frame_spec_hash=frame_binding.source_frame_spec_hash,
        reference_frame_id=frame_binding.reference_frame_id,
        reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
        backend_id=planner.backend_id,
        model_profile_id=planner.model_spec.model_profile_id,
        model_profile_hash=planner.model_spec.semantic_content_hash,
        inference_implementation_hash=planner.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        control_domain_id="real-perception-only:test",
        target_binding_id="real-realsense-only:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        arm_reachability_profile=reachability_profile,
    )
    provider = StationBoundPointGraspProvider(
        planner=planner,
        registration=registration,
        perception_provider=perception,
        frame_binding=frame_binding,
        current_station_revision=lambda: revision["value"],
        arm_reachability_evaluator=arm_reachability_evaluator,
        captured_robot_observation_filter=captured_robot_observation_filter,
    )
    return source, revision, planner, registration, provider


def _request(*, include_arm_reachability=None):
    parameters = {
        "role": "main",
        # Wire decoding freezes JSON arrays before provider validation.
        "target_pixel_uv": (1, 1),
        "max_candidates": 2,
    }
    if include_arm_reachability is not None:
        parameters["include_arm_reachability"] = include_arm_reachability
    return SimpleNamespace(
        parameters=parameters,
        input_artifact_bindings=(),
        expected_station_revision=7,
    )


def test_station_point_grasp_plan_runs_inside_non_motion_read_window() -> None:
    _source, _revision, _planner, _registration, provider = _provider_stack()
    events: list[str] = []

    class _Window:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            events.append("exit")
            return False

    provider.non_motion_read_window = lambda: _Window()
    provider.plan_request(_request())

    assert events == ["enter", "exit"]


def test_graspgenx_capture_filter_is_wired_to_planner() -> None:
    qpos = SimpleNamespace(
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        semantic_content_hash="measured-qpos:test",
        measured_at=9.95,
        available_at=9.952,
        revision=1,
        evidence_refs=("qpos:test",),
    )
    _, _, planner, _, provider = _provider_stack(
        backend_id="graspgenx",
        captured_robot_observation_filter=lambda **_kwargs: None,
    )
    provider.motion_port = SimpleNamespace(
        latest_measured_qpos=lambda: qpos,
    )

    provider.plan_request(_request())

    assert planner.visible_scene_factory is not None

def test_station_point_grasp_route_is_live_bound_read_only_and_proposal_only() -> None:
    source, _revision, planner, registration, provider = _provider_stack()
    route = registration.skill.routes[0]
    assert route.skill_version == "2"
    assert route.supported_modes == ("real_observe",)
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert route.may_move_robot is False
    assert route.requires_operator_confirmation is False
    assert registration.skill.io_contract.input_artifact_slots == ()
    assert registration.skill.io_contract.output_artifact_slots[0].slot_id == (
        REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2
    )

    prepared = provider.plan_request(_request())

    assert source.read_count == 1
    assert planner.base_from_camera is not None
    assert prepared.execution_result["reference_frame_id"] == "canonical_station"
    assert prepared.execution_result["station_revision"] == 7
    assert prepared.execution_result["source_generation"] == 2
    assert prepared.execution_result["frame_index"] == 11
    assert prepared.execution_result["proposal_semantics"] == (
        "ranked_unprivileged_motion_proposals_only"
    )
    assert prepared.execution_result["downstream_control_effects"] == "forbidden"
    assert "selected_ordinal" not in prepared.execution_result
    registration.payload_validator.validate_parameters(_request().parameters)
    with pytest.raises(ValueError, match="parameter fields changed"):
        registration.payload_validator.validate_parameters(
            _request().parameters
            | {"T_reference_from_camera": list(np.eye(4).reshape(-1))}
        )
    registration.payload_validator.validate_execution_result(prepared.execution_result)
    registration.payload_validator.validate_execution_result_for_parameters(
        _request().parameters,
        prepared.execution_result,
    )


def test_station_point_grasp_profile_accepts_graspgenx_backend() -> None:
    _source, _revision, _planner, registration, _provider = _provider_stack()

    profile = replace(registration.profile, backend_id="graspgenx")

    assert profile.backend_id == "graspgenx"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "station_revision",
        "target_pixel_uv",
        "camera_intrinsics",
        "T_reference_from_camera",
        "depth_content_hash",
        "candidate_score",
        "snapshot_bytes",
        "snapshot_length",
        "missing_snapshot",
    ],
)
@pytest.mark.parametrize("cad_bound", [False, True])
def test_graspgenx_capture_snapshot_artifact_is_replayed_through_real_profile(
    tmp_path, monkeypatch, mutation, cad_bound
) -> None:
    from agent_gateway.profiles.real.grasp_approach_plan import (
        build_grasp_visible_scene_reader,
    )
    from agent_gateway.profiles.real.point_grasp_gateway import (
        build_station_bound_point_grasp_executor,
    )
    from agent_gateway.profiles.real.station_point_grasp_deployment import (
        REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID,
    )
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
    from dataset_contracts.grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot

    source, _, planner, registration, provider = _provider_stack(backend_id="graspgenx")
    if cad_bound:
        from deployments.perception.dmtac_grasp_overlay import (
            reviewed_4b2_cad_solid_geometry,
        )

        planner.solid_geometry = reviewed_4b2_cad_solid_geometry()
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    executor = build_station_bound_point_grasp_executor(
        registration=registration,
        provider=provider,
        artifact_store=store,
        runtime_session_id="station-point-grasp:test",
        current_station_revision=lambda: 7,
        max_retained_operations=4,
        clock=lambda: 10.0,
    )
    # Exercise the real adapter and Artifact store without opening any device.
    request = _request()
    request.semantic_content_hash = "a" * 64
    result = executor._read(request, SimpleNamespace(operation_id="snapshot:test"))
    bindings = result.output_artifact_bindings
    snapshot = next(
        x.artifact
        for x in bindings
        if x.slot_id == REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID
    )
    arrays, metadata = decode_grasp_rgbd_snapshot(
        store.managed_read_path(snapshot.artifact_id).read_bytes()
    )
    assert source.read_count == 1
    assert arrays["depth_m"].shape == (3, 4)
    assert metadata["capture"] == result.execution_result
    assert metadata["measured_qpos"] is None
    assert arrays["left_qpos"].size == arrays["right_qpos"].size == 0
    reader = build_grasp_visible_scene_reader(
        artifact_store=store,
        frame_graph=provider.frame_binding.frame_graph,
        execution_frame_id=provider.frame_binding.source_frame_id,
    )
    # Retained operations freeze JSON arrays to tuples, unlike NPZ metadata.
    retained = AgentSkillOperation(
        operation_id="snapshot:test",
        request_hash=request.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        status="completed",
        reason_code="passive_read_completed",
        station_revision_at_admission=7,
        current_station_revision=7,
        started_at=9.0,
        updated_at=10.0,
        deadline_at=20.0,
        execution_outcome="completed",
        execution_receipt_refs=("point-grasp-receipt:test",),
        execution_result=result.execution_result,
        output_artifact_bindings=bindings,
    )
    assert isinstance(metadata["capture"]["target_pixel_uv"], list)
    assert isinstance(retained.execution_result["target_pixel_uv"], tuple)
    scene = reader(retained, retained.execution_result)
    assert len(scene.points) == 12
    assert scene.geometry == planner.solid_geometry
    assert scene.capture_qpos_by_side is None
    with pytest.raises(ValueError, match="capture differ"):
        reader(
            SimpleNamespace(output_artifact_bindings=bindings),
            {**result.execution_result, "station_revision": 8},
        )
    if mutation is None:
        return
    changed = copy.deepcopy(result.execution_result)
    expected_error = "capture differ"
    if mutation == "station_revision":
        changed[mutation] += 1
    elif mutation == "target_pixel_uv":
        changed[mutation][0] += 1
    elif mutation == "camera_intrinsics":
        changed[mutation]["fx"] += 1
    elif mutation == "T_reference_from_camera":
        changed[mutation][3] += 0.001
    elif mutation == "depth_content_hash":
        changed[mutation] = "0" * 64
    elif mutation == "candidate_score":
        changed["candidates"][0]["score"] += 0.01
    elif mutation in {"snapshot_bytes", "snapshot_length"}:
        path = store.managed_read_path(snapshot.artifact_id)
        original = path.read_bytes()
        # Simulate corrupted read bytes without modifying the immutable CAS.
        corrupted = tmp_path / "corrupted_snapshot.npz"
        corrupted.write_bytes(
            original[:-1]
            if mutation == "snapshot_length"
            else original[:-1] + bytes([original[-1] ^ 1])
        )
        monkeypatch.setattr(store, "managed_read_path", lambda _artifact_id: corrupted)
        expected_error = "Artifact identity changed"
    else:
        assert mutation == "missing_snapshot"
        retained = replace(retained, output_artifact_bindings=())
        expected_error = "no unique RGB-D snapshot"
    retained = replace(retained, execution_result=changed)
    with pytest.raises(ValueError, match=expected_error):
        reader(retained, retained.execution_result)


def test_grasp_scene_reader_filters_robot_observations_from_measured_qpos_snapshot(
    tmp_path,
) -> None:
    from agent_gateway.profiles.real.grasp_approach_plan import (
        build_grasp_visible_scene_reader,
    )
    from agent_gateway.profiles.real.point_grasp_gateway import (
        build_station_bound_point_grasp_executor,
    )
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
    from robot_motion.rm65.captured_robot_observations import (
        RM65CapturedRobotObservationFilterResult,
    )

    source, _, planner, registration, provider = _provider_stack(backend_id="graspgenx")
    target_mask = np.zeros((3, 4), dtype=bool)
    target_mask[1, 1] = True
    planner.last_target_mask = target_mask
    capture_qpos = {
        "left": np.linspace(0.0, 0.5, 6),
        "right": np.linspace(0.1, 0.6, 6),
    }
    provider.motion_port = SimpleNamespace(
        latest_measured_qpos=lambda: SimpleNamespace(
            qpos_by_side=capture_qpos,
            semantic_content_hash="measured-qpos:test",
            revision=4,
            measured_at=9.95,
            available_at=9.952,
            evidence_refs=("measured-qpos-receipt:test",),
        )
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    executor = build_station_bound_point_grasp_executor(
        registration=registration,
        provider=provider,
        artifact_store=store,
        runtime_session_id="station-point-grasp:test",
        current_station_revision=lambda: 7,
        max_retained_operations=4,
        clock=lambda: 10.0,
    )
    request = _request()
    request.semantic_content_hash = "b" * 64
    result = executor._read(
        request,
        SimpleNamespace(operation_id="snapshot:with-qpos"),
    )
    retained = AgentSkillOperation(
        operation_id="snapshot:with-qpos",
        request_hash=request.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        status="completed",
        reason_code="passive_read_completed",
        station_revision_at_admission=7,
        current_station_revision=7,
        started_at=9.0,
        updated_at=10.0,
        deadline_at=20.0,
        execution_outcome="completed",
        execution_receipt_refs=("point-grasp-receipt:test",),
        execution_result=result.execution_result,
        output_artifact_bindings=result.output_artifact_bindings,
    )
    calls = []

    def filter_capture(**kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(kwargs["target_mask"], target_mask)
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                kwargs["capture_qpos_by_side"][side], capture_qpos[side]
            )
        filtered_depth = np.asarray(kwargs["depth_m"], dtype=float).copy()
        filtered_depth[0, 0] = 0.0
        removed = np.zeros(filtered_depth.shape, dtype=bool)
        removed[0, 0] = True
        return RM65CapturedRobotObservationFilterResult(
            depth_m=filtered_depth,
            removed_mask=removed,
            evidence={
                "schema": "dexteleop.rm65.captured_robot_observation_filter.v1",
                "removed_background_points": 1,
                "removed_target_points": 0,
            },
        )

    reader = build_grasp_visible_scene_reader(
        artifact_store=store,
        frame_graph=provider.frame_binding.frame_graph,
        execution_frame_id=provider.frame_binding.source_frame_id,
        captured_robot_observation_filter=filter_capture,
    )
    scene = reader(retained, retained.execution_result)

    assert source.read_count == 1
    assert len(calls) == 1
    assert len(scene.points) == 11
    assert scene.target_mask.sum() == 1
    assert scene.capture_self_observation_evidence["removed_background_points"] == 1
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            scene.capture_qpos_by_side[side], capture_qpos[side]
        )


def test_graspgenx_inference_failure_retains_original_input_without_faking_a_mask():
    from dataset_contracts.grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot

    source, _, planner, _, provider = _provider_stack(backend_id="graspgenx")

    def fail(*_args, **_kwargs):
        raise RuntimeError("no candidate found")

    planner.plan = fail
    retained = []
    with pytest.raises(RuntimeError, match="no candidate found"):
        provider.plan_request(_request(), input_snapshot_sink=retained.append)
    assert source.read_count == 1 and len(retained) == 1
    arrays, metadata = decode_grasp_rgbd_snapshot(retained[0])
    assert arrays["depth_m"].shape == (3, 4)
    assert metadata["target_mask_available"] is False
    assert metadata["scope"] == "pre_inference_input_not_a_validated_grasp_plan"


def test_missing_measured_qpos_stops_inference_but_keeps_camera_input():
    source, _, planner, _, provider = _provider_stack(backend_id="graspgenx")
    provider.motion_port = SimpleNamespace(latest_measured_qpos=lambda: None)

    def must_not_infer(*_args, **_kwargs):
        pytest.fail("missing qpos must stop this path")

    planner.plan = must_not_infer
    retained = []
    with pytest.raises(RuntimeError, match="camera-matched measured qpos"):
        provider.plan_request(_request(), input_snapshot_sink=retained.append)
    assert source.read_count == 1 and len(retained) == 1


def test_station_point_grasp_v3_reachability_is_request_gated_and_annotation_only() -> (
    None
):
    calls = []

    def evaluate(candidates):
        calls.append(candidates)
        profile = registration.profile.arm_reachability_profile
        assert profile is not None
        context = {
            "arm_side": profile.arm_side,
            "evaluator_profile_id": profile.evaluator_profile_id,
            "evaluator_profile_hash": profile.evaluator_profile_hash,
            "candidate_reference_frame_id": registration.profile.reference_frame_id,
            "solver_reference_frame_id": profile.solver_reference_frame_id,
            "solver_reference_frame_spec_hash": (
                profile.solver_reference_frame_spec_hash
            ),
            "frame_graph_hash": profile.frame_graph_hash,
            "time_budget_ms": profile.time_budget_ms,
            "max_solve_cycles": profile.max_solve_cycles,
            "max_measured_qpos_age_ms": profile.max_measured_qpos_age_ms,
            "standard_position_tolerance_m": (profile.standard_position_tolerance_m),
            "standard_orientation_tolerance_deg": (
                profile.standard_orientation_tolerance_deg
            ),
            "basic_position_tolerance_m": profile.basic_position_tolerance_m,
            "basic_orientation_tolerance_deg": (
                profile.basic_orientation_tolerance_deg
            ),
            "measured_qpos_revision": 4,
            "measured_qpos_at": 9.95,
            "measured_qpos_age_ms": 50.0,
            "evaluation_scope": ("finite_time_endpoint_ik_not_path_or_grasp_success"),
            "candidate_order_effect": "annotation_only_order_unchanged",
        }
        annotation = {
            "classification": "standard_reachable",
            "position_residual_m": 0.001,
            "orientation_residual_deg": 0.5,
            "solve_cycles": 2,
            "elapsed_ms": 4.0,
            "termination_reason": "standard_tolerance_met",
        }
        return context, tuple(annotation for _candidate in candidates)

    _source, _revision, _planner, registration, provider = _provider_stack(
        arm_reachability_evaluator=evaluate
    )
    assert registration.skill.definition.skill_version == "3"

    disabled_request = _request(include_arm_reachability=False)
    disabled = provider.plan_request(disabled_request).execution_result
    assert calls == []
    assert "arm_reachability_context" not in disabled
    assert "arm_reachability" not in disabled["candidates"][0]
    registration.payload_validator.validate_execution_result(disabled)
    registration.payload_validator.validate_execution_result_for_parameters(
        disabled_request.parameters, disabled
    )

    enabled_request = _request(include_arm_reachability=True)
    enabled = provider.plan_request(enabled_request).execution_result
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0][0],
        np.asarray(enabled["candidates"][0]["T_reference_from_tool_tcp"]).reshape(4, 4),
    )
    assert enabled["candidates"][0]["arm_reachability"]["classification"] == (
        "standard_reachable"
    )
    registration.payload_validator.validate_parameters(enabled_request.parameters)
    registration.payload_validator.validate_execution_result(enabled)
    registration.payload_validator.validate_execution_result_for_parameters(
        enabled_request.parameters, enabled
    )

    seed_exhausted = copy.deepcopy(enabled)
    seed_exhausted_annotation = seed_exhausted["candidates"][0]["arm_reachability"]
    seed_exhausted_annotation.update(
        {
            "classification": "difficult_to_reach",
            "position_residual_m": 0.06,
            "orientation_residual_deg": 25.0,
            "solve_cycles": (
                registration.profile.arm_reachability_profile.max_solve_cycles
            ),
            "termination_reason": "seed_budget_exhausted",
        }
    )
    registration.payload_validator.validate_execution_result(seed_exhausted)

    with pytest.raises(ValueError, match="does not match the request"):
        registration.payload_validator.validate_execution_result_for_parameters(
            disabled_request.parameters, enabled
        )


def test_rm65_reachability_adapter_accepts_normalized_graspnet_candidate() -> None:
    binding = _frame_binding()
    raw_pose = np.asarray(
        [
            [
                -0.17259738000158176,
                0.18359564713816373,
                0.9677307159760798,
                -0.13079618433586204,
            ],
            [
                0.04199108059770058,
                -0.9802105728615037,
                0.19345253616022093,
                -0.07363994251000638,
            ],
            [
                0.9840969607643857,
                0.07402546072335396,
                0.16147240422405731,
                0.1374787059167376,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = _normalize_graspnet_rigid_matrix(
        raw_pose,
        field_name="GraspNet camera_from_grasp",
    )

    class _MotionPort:
        @staticmethod
        def latest_measured_qpos():
            return SimpleNamespace(
                qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
                revision=4,
                measured_at=9.95,
            )

    class _ReachabilityPlanner:
        @staticmethod
        def assess_endpoint_ik(**kwargs):
            np.testing.assert_allclose(
                kwargs["target_tool_tcp_world"][:3, :3].T
                @ kwargs["target_tool_tcp_world"][:3, :3],
                np.eye(3),
                atol=1e-12,
                rtol=0.0,
            )
            return SimpleNamespace(
                to_dict=lambda: {
                    "classification": "standard_reachable",
                    "position_residual_m": 0.001,
                    "orientation_residual_deg": 0.5,
                    "solve_cycles": 1,
                    "elapsed_ms": 1.0,
                    "termination_reason": "standard_tolerance_met",
                }
            )

    profile = SimpleNamespace(
        solver_reference_frame_id=binding.source_frame_id,
        solver_reference_frame_spec_hash=binding.source_frame_spec_hash,
        frame_graph_hash=binding.frame_graph.semantic_content_hash,
        arm_side="right",
        evaluator_profile_id="rm65.reachability:test",
        evaluator_profile_hash="rm65-reachability-profile:test",
        time_budget_ms=20.0,
        max_solve_cycles=2,
        max_measured_qpos_age_ms=200.0,
        standard_position_tolerance_m=0.005,
        standard_orientation_tolerance_deg=2.0,
        basic_position_tolerance_m=0.010,
        basic_orientation_tolerance_deg=5.0,
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = profile
    adapter.planner = _ReachabilityPlanner()
    adapter.motion_port = _MotionPort()
    adapter.frame_graph = binding.frame_graph
    adapter.candidate_reference_frame_id = binding.reference_frame_id
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0

    with pytest.raises(ValueError, match="rotation must be orthonormal"):
        adapter((raw_pose,))
    context, annotations = adapter((normalized,))

    assert context["candidate_order_effect"] == "annotation_only_order_unchanged"
    assert annotations[0]["classification"] == "standard_reachable"
    adapter.clock = lambda: 10.3
    matched_cut = adapter.motion_port.latest_measured_qpos()
    capture_context, _ = adapter(
        (normalized,),
        measured_qpos_cut=matched_cut,
        measurement_reference_time=10.0,
    )
    assert capture_context["measured_qpos_age_ms"] == pytest.approx(50.0)


def test_station_point_grasp_fails_if_revision_changes_during_inference() -> None:
    revision = {"value": 7, "mutate": True}
    _source, _revision, _planner, _registration, provider = _provider_stack(
        revision=revision
    )

    with pytest.raises(RuntimeError, match="during point-grasp model inference"):
        provider.plan_request(_request())


def test_debug_target_coordinate_overlay_projects_depth_target_in_station_frame() -> (
    None
):
    image = Image.new("RGB", (200, 160), (10, 10, 10))
    rendered = point_grasp_render_gateway._render_target_coordinate_debug_overlay(
        image,
        target_point_reference_m=(0.0, 0.0, 1.0),
        reference_from_camera=np.eye(4),
        intrinsics={
            "width": 200,
            "height": 160,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 100.0,
            "cy": 80.0,
        },
    )

    assert rendered is not image
    assert image.getpixel((100, 74)) == (10, 10, 10)
    assert rendered.getpixel((100, 74)) == (0, 255, 255)
    assert rendered.getpixel((103, 80)) == (255, 64, 64)
    assert rendered.getpixel((100, 84)) == (64, 255, 64)


def test_debug_target_coordinate_profile_is_default_off_and_hash_bound() -> None:
    _source, _revision, _planner, point_grasp, _provider = _provider_stack()
    default = build_station_bound_point_grasp_render_registration(
        point_grasp_registration=point_grasp,
    )
    enabled = build_station_bound_point_grasp_render_registration(
        point_grasp_registration=point_grasp,
        debug_target_coordinate_overlay=True,
    )

    assert default.profile.debug_target_coordinate_overlay is False
    assert enabled.profile.debug_target_coordinate_overlay is True
    assert (
        default.profile.semantic_content_hash != enabled.profile.semantic_content_hash
    )


def test_point_grasp_is_default_off_and_lifecycle_order_is_reversible(
    monkeypatch,
    tmp_path,
) -> None:
    events = []
    source = _Source()
    binding = RealPerceptionSourceBinding(
        camera_sources={"front": source},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:composition-test",
    )
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id="rpent:test",
        primary_client_uid=1000,
    )
    default = assemble_camera_only_real_perception_composition(
        source_binding=binding,
        socket_path=tmp_path / "default.sock",
        artifact_root=tmp_path,
        client_bindings=client_bindings,
        runtime_session_id="real-perception:default-test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        clock=lambda: 10.0,
    )
    try:
        assert default.station_point_grasp_registration is None
        assert default.station_point_grasp_render_registration is None
        assert [
            entry.route.skill_id
            for entry in default.contract_bundle.route_manifest.skill_routes
        ] == [
            "perception.back_project",
            "perception.capture_depth",
            "perception.capture_rgb",
        ]
    finally:
        default.application.close()

    monkeypatch.setattr(
        realsense_perception,
        "start_realsense_sources",
        lambda _sources: events.append("cameras.start"),
    )
    monkeypatch.setattr(
        realsense_perception,
        "admit_realsense_sources",
        lambda _sources: events.append("cameras.admit"),
    )
    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        lambda _sources: events.append("cameras.close"),
    )
    localizer = _Localizer(events)
    planner = _Planner(localizer, events=events)
    composition = assemble_camera_only_real_perception_composition(
        source_binding=binding,
        socket_path=tmp_path / "enabled.sock",
        artifact_root=tmp_path,
        client_bindings=client_bindings,
        runtime_session_id="real-perception:enabled-test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        object_localizer=localizer,
        point_grasp_planner=planner,
        clock=lambda: 10.0,
    )

    class _Server:
        def serve_forever(self, poll_interval=0.5):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            events.append("gateway.close")
            composition.session.close()

    composition.application.server_factory = lambda: (
        events.append("socket.create") or _Server()
    )
    composition.application.start()
    assert events[:5] == [
        "cameras.start",
        "cameras.admit",
        "localizer.start",
        "planner.start",
        "socket.create",
    ]
    adapter = composition.session.wire_adapter(authenticated_client_id="rpent:test")
    observation = adapter.get_observation_snapshot()
    manifest = composition.contract_bundle.route_manifest
    entry = manifest.route("perception.plan_point_grasp", "2")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    submitted = adapter.submit_read_only_skill_request(
        AgentSkillRequest(
            client_id="rpent:test",
            request_id="request:station-point-grasp:test",
            idempotency_key="idempotency:station-point-grasp:test",
            client_sequence=1,
            runtime_session_id=manifest.runtime_session_id,
            skill_route_manifest_hash=manifest.semantic_content_hash,
            expected_station_revision=int(observation["station_revision"]),
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=(entry.deployment_binding.deployment_binding_hash),
            parameter_profile_id=profile.parameter_profile_id,
            parameter_profile_hash=profile.parameter_profile_hash,
            parameters={
                "role": "main",
                "target_pixel_uv": [1, 1],
                "max_candidates": 2,
            },
            observation_snapshot_id=str(observation["snapshot_id"]),
            requested_mode="real_observe",
            requested_timeout_ms=100_000,
        ).to_wire()
    )
    assert submitted["accepted"] is True
    operation_id = submitted["operation"]["operation_id"]
    assert composition.session.station_point_grasp_executor is not None
    assert composition.session.station_point_grasp_executor.wait(
        operation_id,
        timeout_sec=2.0,
    )
    operation = composition.session.service.operation_store.get(operation_id)
    assert operation is not None
    assert operation.status == "completed"
    assert operation.execution_result["proposal_semantics"] == (
        "ranked_unprivileged_motion_proposals_only"
    )
    assert operation.output_artifact_bindings[0].slot_id == (
        REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2
    )
    render_entry = manifest.route("perception.render_point_grasp_candidate", "1")
    render_route = render_entry.route
    assert render_route.control_effect_class == "read_only"
    assert render_route.privilege_tier == "t0_read_only"
    assert render_route.cancellation_policy == "cancel_only"
    assert render_route.supported_modes == ("real_observe",)
    assert render_route.may_move_robot is False
    assert render_route.requires_operator_confirmation is False
    assert render_route.requires_deadman is False
    render_slot = composition.station_point_grasp_render_registration
    assert render_slot is not None
    output_slot = render_slot.skill.io_contract.output_artifact_slots[0]
    assert output_slot.slot_id == REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1
    assert output_slot.allowed_schema_ids == (POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,)
    assert output_slot.allowed_media_types == (
        LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    )
    assert output_slot.allowed_encodings == (LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,)
    assert output_slot.allowed_compressions == (
        LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    )
    render_profile = render_route.allowed_parameter_profiles[0]

    def submit_render(parameters, *, sequence, suffix, station_observation=None):
        selected_observation = (
            observation if station_observation is None else station_observation
        )
        return adapter.submit_read_only_skill_request(
            AgentSkillRequest(
                client_id="rpent:test",
                request_id=f"request:station-point-grasp-render:{suffix}",
                idempotency_key=f"idempotency:station-point-grasp-render:{suffix}",
                client_sequence=sequence,
                runtime_session_id=manifest.runtime_session_id,
                skill_route_manifest_hash=manifest.semantic_content_hash,
                expected_station_revision=int(selected_observation["station_revision"]),
                skill_id=render_route.skill_id,
                skill_version=render_route.skill_version,
                skill_definition_hash=render_route.skill_definition_hash,
                route_id=render_route.route_id,
                route_version=render_route.route_version,
                route_hash=render_route.semantic_content_hash,
                deployment_binding_hash=(
                    render_entry.deployment_binding.deployment_binding_hash
                ),
                parameter_profile_id=render_profile.parameter_profile_id,
                parameter_profile_hash=render_profile.parameter_profile_hash,
                parameters=parameters,
                observation_snapshot_id=str(selected_observation["snapshot_id"]),
                requested_mode="real_observe",
                requested_timeout_ms=20_000,
            ).to_wire()
        )

    render_parameters = {
        "point_grasp_operation_id": operation_id,
        "candidate_ordinal": 0,
        "view_role": "main",
    }
    rendered = submit_render(render_parameters, sequence=2, suffix="success")
    assert rendered["accepted"] is True
    render_operation_id = rendered["operation"]["operation_id"]
    render_executor = composition.session.station_point_grasp_render_executor
    assert render_executor is not None
    assert render_executor.wait(render_operation_id, timeout_sec=2.0)
    render_operation = composition.session.service.operation_store.get(
        render_operation_id
    )
    assert render_operation is not None
    assert render_operation.status == "completed"
    result = render_operation.execution_result
    assert result is not None
    assert result["point_grasp_operation_id"] == operation_id
    assert result["point_grasp_request_hash"] == operation.request_hash
    assert (
        result["source_capture_group_id"]
        == operation.execution_result["capture_group_id"]
    )
    assert result["candidate_ordinal"] == 0
    assert result["view_role"] == "main"
    assert result["view_source_id"] == "front"
    assert result["view_station_revision"] == result["source_station_revision"]
    assert result["depth_occlusion_tested"] is False
    assert result["downstream_control_effects"] == "forbidden"
    reference_from_tcp = np.asarray(result["T_reference_from_tool_tcp"]).reshape(4, 4)
    reference_from_camera = np.asarray(result["T_reference_from_camera"]).reshape(4, 4)
    camera_from_tcp = np.asarray(result["T_camera_from_tool_tcp"]).reshape(4, 4)
    np.testing.assert_allclose(
        np.linalg.inv(reference_from_camera) @ reference_from_tcp,
        camera_from_tcp,
    )
    assert (
        result["frame_binding_hash"] == composition.frame_binding.semantic_content_hash
    )
    assert result["frame_graph_hash"] == (
        composition.frame_binding.frame_graph.semantic_content_hash
    )
    assert result["source_frame_id"] == composition.frame_binding.source_frame_id
    assert result["reference_frame_id"] == (
        composition.frame_binding.reference_frame_id
    )
    assert len(result["frames"]) == 1
    frame = result["frames"][0]
    assert frame == {
        "ordinal": 0,
        "role": "main",
        "width": 4,
        "height": 3,
        "media_type": "image/jpeg",
        "media_content_hash": render_operation.output_artifact_bindings[
            0
        ].artifact.sha256,
    }
    artifact_binding = render_operation.output_artifact_bindings[0]
    assert artifact_binding.slot_id == REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1
    assert artifact_binding.ordinal == 0
    assert artifact_binding.artifact.schema_id == POINT_GRASP_ANNOTATED_SCHEMA_ID_V1
    assert artifact_binding.artifact.media_type == "image/jpeg"
    assert artifact_binding.artifact.encoding == "jpeg.rgb8.v1"
    assert artifact_binding.artifact.compression == "none"
    assert source.read_count == 2  # proposal generation and rendered view
    inconsistent = dict(result)
    inconsistent_camera_from_tcp = list(result["T_camera_from_tool_tcp"])
    inconsistent_camera_from_tcp[3] += 0.01
    inconsistent["T_camera_from_tool_tcp"] = inconsistent_camera_from_tcp
    with pytest.raises(ValueError, match="transform chain is inconsistent"):
        render_slot.payload_validator.validate_execution_result(inconsistent)

    unknown = submit_render(
        render_parameters | {"point_grasp_operation_id": "operation:unknown"},
        sequence=3,
        suffix="unknown",
    )
    assert unknown["accepted"] is True
    unknown_id = unknown["operation"]["operation_id"]
    assert render_executor.wait(unknown_id, timeout_sec=2.0)
    unknown_operation = composition.session.service.operation_store.get(unknown_id)
    assert unknown_operation is not None
    assert unknown_operation.status == "failed"
    assert unknown_operation.reason_code == "passive_read_executor_exception:ValueError"

    missing_ordinal = submit_render(
        render_parameters | {"candidate_ordinal": 1},
        sequence=4,
        suffix="missing-ordinal",
    )
    assert missing_ordinal["accepted"] is True
    missing_ordinal_id = missing_ordinal["operation"]["operation_id"]
    assert render_executor.wait(missing_ordinal_id, timeout_sec=2.0)
    missing_ordinal_operation = composition.session.service.operation_store.get(
        missing_ordinal_id
    )
    assert missing_ordinal_operation is not None
    assert missing_ordinal_operation.status == "failed"

    injected_matrix = submit_render(
        render_parameters | {"T_reference_from_tool_tcp": list(np.eye(4).reshape(-1))},
        sequence=5,
        suffix="injected-matrix",
    )
    assert injected_matrix["accepted"] is False
    assert injected_matrix["reason_code"] == "parameter_payload_rejected"

    unavailable_role = submit_render(
        render_parameters | {"view_role": "wrist_left"},
        sequence=6,
        suffix="unavailable-role",
    )
    assert unavailable_role["accepted"] is False
    assert unavailable_role["reason_code"] == "parameter_payload_rejected"

    composition.session.observation_producer.revision_tracker.observe(
        {"forced_test_control_fingerprint": "changed"}
    )
    changed_observation = adapter.get_observation_snapshot()
    assert changed_observation["station_revision"] > observation["station_revision"]
    stale_source = submit_render(
        render_parameters,
        sequence=7,
        suffix="stale-source",
        station_observation=changed_observation,
    )
    assert stale_source["accepted"] is True
    stale_source_id = stale_source["operation"]["operation_id"]
    assert render_executor.wait(stale_source_id, timeout_sec=2.0)
    stale_source_operation = composition.session.service.operation_store.get(
        stale_source_id
    )
    assert stale_source_operation is not None
    assert stale_source_operation.status == "failed"
    assert stale_source_operation.output_artifact_bindings == ()
    # Planning and the repeated render requests above share the one application
    # camera resource transaction; Skill calls must never restart it.
    assert events.count("cameras.start") == 1
    assert events.count("cameras.close") == 0
    composition.application.close()
    assert planner.starts == 1
    assert planner.closes == 1
    assert events.count("cameras.start") == 1
    assert events.count("cameras.close") == 1
    assert events[-4:] == [
        "gateway.close",
        "planner.close",
        "localizer.close",
        "cameras.close",
    ]
