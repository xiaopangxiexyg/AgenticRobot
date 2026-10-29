"""RealMan Gateway publication of the shared tool-entity relation and hover Skills.

Composed offline with fake arm backends and an unstarted camera provider; no
device, model or planner is started. The relation read runs end to end through
the composed executor with substituted captures and segmentation.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent_gateway_real_rm65_composition import (
    REPO_ROOT,
    _Backend,
    _ForceGuard,
    _Localizer,
    _TargetBuilder,
)

from agent_gateway.profiles.real import rm65_composition as composition_module
from agent_gateway.profiles.real.gateway_cli import (
    DEFAULT_RM65_CAMERA_MOUNT_URDF,
    DEFAULT_RM65_CAMERA_REGISTRY_V4,
)
from agent_gateway.profiles.real.motion_observed_scene import (
    SafeMotionObservationConfig,
)
from agent_gateway.profiles.real.one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
)
from agent_gateway.profiles.real.rm65_composition import (
    assemble_exclusive_rm65_agent_primitive_composition,
)
from teleop_runtime.skill_deploy import rm65_profile as profile_module
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    assemble_exclusive_rm65_skill_lower,
)
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

RELATION = ("perception.describe_tool_object_relations", "1")


def _frozen(value):
    """The operation store's immutable JSON form (tuples and read-only mappings)."""
    from types import MappingProxyType

    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value
HOVER = ("perception.query_tool_hover_paths", "1")
CALIBRATIONS = REPO_ROOT / "hardware/rm65/utils/configs/calibrations"


def _dmtac_lower(monkeypatch):
    monkeypatch.setattr(profile_module, "load_assignment", lambda path: {
        "assignments": {"left": {"arm_ip": "192.0.2.10"}, "right": {"arm_ip": "192.0.2.11"}}})
    profile = resolve_exclusive_rm65_skill_deployment_profile(ExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=REPO_ROOT / "README.md",
        tool_profile_path_by_side={side: CALIBRATIONS / f"rm65_{side}_DMTAC_M.json" for side in ("left", "right")},
        envelope_profile_path=REPO_ROOT / "teleop_runtime/configs/skill_deploy/agent_primitive_right_relative_tight_v1.json"))
    guard = _ForceGuard()
    lower = assemble_exclusive_rm65_skill_lower(profile=profile, backend_factory=_Backend,
        target_builder_factory=lambda config: _TargetBuilder(), force_guard_factory=lambda backends: guard)
    return lower, guard


def _compose(lower, tmp_path, **kwargs):
    base = {"lower": lower, "socket_path": tmp_path / "tool.sock", "authorized_side": "right",
        "authorized_action": "move_tcp_to_pose_safe", "confirmation_token": REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        "client_id": "rpent:test", "client_uid": 1000, "runtime_session_id": "tool:test",
        "control_domain_id": "tool:domain", "embodiment_instance_id": "tool:arms",
        "camera_config_path": REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json",
        "camera_registry_path": DEFAULT_RM65_CAMERA_REGISTRY_V4,
        "camera_mount_urdf_path": DEFAULT_RM65_CAMERA_MOUNT_URDF,
        "artifact_root": tmp_path / "artifacts", "enable_back_projection": True}
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    return assemble_exclusive_rm65_agent_primitive_composition(**{**base, **kwargs})


@pytest.fixture
def planners(monkeypatch):
    created = []

    class Planner:
        def __call__(self, **kwargs):
            return None

        def plan_with_observed_cloud(self, cloud, **kwargs):
            return cloud

    def planner(**kwargs):
        created.append(kwargs)
        return Planner()

    monkeypatch.setattr(composition_module, "RM65GraspJointTrajectoryPlanner", planner)
    return created


def test_tool_entity_skills_are_default_off(monkeypatch, tmp_path, planners):
    lower, guard = _dmtac_lower(monkeypatch)
    composition = _compose(lower, tmp_path)
    session = composition.deployment.gateway_session
    try:
        published = {(e.route.skill_id, e.route.skill_version) for e in session.contract_bundle.route_manifest.skill_routes}
        assert RELATION not in published and HOVER not in published
        assert session.tool_object_relation_executor is None and session.tool_hover_executor is None
    finally:
        session.close()
    assert guard.starts == 0


def test_real_publishes_read_only_relation_and_hover_routes(monkeypatch, tmp_path, planners):
    from agent_gateway.profiles.real import tool_entity as real_tool_entity

    ports = []
    original_port = real_tool_entity.RealToolScenePort.__init__

    def record_port(self, **kwargs):
        ports.append(kwargs)
        original_port(self, **kwargs)

    monkeypatch.setattr(real_tool_entity.RealToolScenePort, "__init__", record_port)
    lower, guard = _dmtac_lower(monkeypatch)
    localizer = _Localizer()
    composition = _compose(lower, tmp_path, enable_surface_anchor_v2=True, object_localizer=localizer,
        safe_motion_observation_config=SafeMotionObservationConfig(),
        enable_tool_object_relations=True, enable_tool_hover_queries=True)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        manifest = bundle.route_manifest
        for key in (RELATION, HOVER):
            route = manifest.route(*key).route
            assert (route.control_effect_class, route.privilege_tier, route.cancellation_policy) == (
                "read_only", "t0_read_only", "cancel_only")
            assert route.may_move_robot is False and route.may_open_publication is False
            assert key in session.passive_read_executors_by_skill_ref
        relation = bundle.tool_object_relation_registration
        hover = bundle.tool_hover_registration
        assert relation.roles == ("main", "wrist_left", "wrist_right")
        assert hover.roles == SafeMotionObservationConfig().roles
        assert dict(hover.component_side_by_id) == {"right_arm": "right"}
        assert hover.skill.io_contract.input_artifact_slots[0].slot_id == "surface_anchor"
        assert manifest.route(*HOVER).route.max_operation_duration_ms == 40_000
        assert manifest.route(*RELATION).route.max_operation_duration_ms == 90_000
        # Hover targets go unchanged to the safe TCP Route: same public frame.
        assert hover.reference_frame_id == bundle.pose_target_profile.reference_frame_id
        assert relation.reference_frame_id == hover.reference_frame_id
        # Captures and FK are in the Station world; results in the public Station frame.
        (port,) = ports
        np.testing.assert_allclose(port["public_from_geometry"],
            np.linalg.inv(np.asarray(lower.profile.scene_world_from_canonical_station, dtype=float)), atol=1e-12)
        # Both reviewed DMTAC_M arms are bound, including the URDF sensor boxes.
        identities = bundle.tool_object_relation_registration.skill.schema_entry.parameter_profiles
        assert len(identities) == 1
        # The hover planner is a separate lazy instance; composing starts nothing.
        assert all(backend.starts == 0 for backend in lower.arm_backends.values()) and guard.starts == 0
    finally:
        session.close()


def test_real_relation_read_runs_the_shared_core_end_to_end(monkeypatch, tmp_path, planners):
    from deployments.perception.sam2_grounding_dino import (
        GroundedObjectSegment,
        encode_grounded_object_segment_artifacts,
    )

    lower, _ = _dmtac_lower(monkeypatch)
    localizer = _Localizer()
    # Request FK: right TCP 0.2 m above a 4 cm patch, pointing down; left far away.
    down = np.diag([1., -1., -1.])

    def tool_fk(*, left_qpos, right_qpos):
        poses = {side: np.eye(4) for side in ("left", "right")}
        poses["right"][:3, :3], poses["right"][:3, 3] = down, (0., 0., 0.62)
        poses["left"][:3, 3] = (5., 5., 5.)
        return poses

    monkeypatch.setattr(composition_module, "build_exclusive_rm65_skill_target_builder",
                        lambda profile: SimpleNamespace(tool_tcp_world_targets_for_qpos_pair=tool_fk))
    composition = _compose(lower, tmp_path, object_localizer=localizer, enable_tool_object_relations=True)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        profile = session.back_projection_provider.registration.profile
        roles = bundle.tool_object_relation_registration.roles
        captures = {}
        for role in roles:
            width, height = profile.dimensions_by_role[role]
            # Camera 1 m above the table looking straight down (world Z up).
            camera = np.eye(4)
            camera[:3, :3], camera[:3, 3] = np.diag([1., -1., -1.]), (0., 0., 1.)
            captures[role] = SimpleNamespace(source_id=profile.source_id_by_role[role], role=role,
                acquired_at=time.time(), station_revision=0, rgb=np.zeros((height, width, 3), dtype=np.uint8),
                rgb_content_hash="b" * 64, depth_m=np.full((height, width), 0.6, dtype=np.float32),
                depth_content_hash="d" * 64, width=width, height=height,
                intrinsics={"fx": 400., "fy": 400., "cx": width / 2, "cy": height / 2},
                source_from_camera=tuple(camera.reshape(-1)),
                capture_qpos_by_side={side: np.zeros(6) for side in ("left", "right")})
        reads = []

        def capture_aligned_rgbd(*, role, expected_station_revision, require_stationary):
            reads.append((role, expected_station_revision, require_stationary))
            return captures[role]

        monkeypatch.setattr(session.perception_provider, "capture_aligned_rgbd", capture_aligned_rgbd)

        def segment(image, *, language_prompt, pixel_uv, min_score):
            width, height = image.size
            mask = np.zeros((height, width), dtype=bool)
            if language_prompt != "blue cube":
                return None
            mask[height // 2 - 6:height // 2 + 6, width // 2 - 6:width // 2 + 6] = True
            rows, cols = np.nonzero(mask)
            return GroundedObjectSegment(mask=mask, score=0.9, label=language_prompt,
                box_xyxy=(int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)))

        monkeypatch.setattr(localizer, "segment", segment)
        monkeypatch.setattr(localizer, "encode_segment_artifacts", encode_grounded_object_segment_artifacts)
        executor = session.tool_object_relation_executor
        monkeypatch.setattr(executor, "cancel_requested", lambda operation_id: False)
        parameters = {"objects": [{"label": "cube", "language_prompt": "blue cube"},
                                  {"label": "cup", "language_prompt": "black thermos"}],
                      "roles": None, "min_score": 0.3}
        result = executor._read(SimpleNamespace(parameters=parameters, expected_station_revision=0),
                                SimpleNamespace(operation_id="tool-relations:test", deadline_at=time.time() + 60))
        payload = result.execution_result
        validator = bundle.tool_object_relation_registration.payload_validator
        validator.validate_execution_result_for_parameters(parameters, payload)
        # Later consumers re-validate the stored, frozen form.
        validator.validate_execution_result_for_parameters(_frozen(parameters), _frozen(payload))
        assert reads == [(role, 0, True) for role in roles]
        assert payload["evaluated_sides"] == ["left", "right"] and payload["unbound_sides"] == []
        assert {(item["label"], item["reason"]) for item in payload["unmeasured"]} == {("cup", "no_mask_above_threshold")}
        fused = {(item["side"], item["label"]): item["relation"] for item in payload["objects"]}
        assert fused[("left", "cube")] == "far"
        assert fused[("right", "cube")] in {"near", "far"}
        assert payload["motion_executed"] is False
        assert {view["gripper_pose_source"] for view in payload["views"]} == {"frame_capture_qpos"}
    finally:
        session.close()


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"enable_tool_hover_queries": True, "safe_motion_observation_config": SafeMotionObservationConfig()},
     "surface anchor v2"),
    ({"enable_tool_object_relations": True}, "bound segmenter"),
])
def test_real_tool_entity_prerequisites_fail_before_resources(monkeypatch, tmp_path, planners, kwargs, message):
    lower, guard = _dmtac_lower(monkeypatch)
    with pytest.raises(ValueError, match=message):
        _compose(lower, tmp_path, **kwargs)
    assert guard.starts == 0


def test_real_tool_entities_require_the_reviewed_dmtac_tool(monkeypatch, tmp_path, planners):
    from test_agent_gateway_real_rm65_composition import _lower

    lower, _ = _lower(monkeypatch)  # Inspire hands: no reviewed gripper entity.
    with pytest.raises(ValueError, match="DMTAC_M"):
        _compose(lower, tmp_path, object_localizer=_Localizer(), enable_tool_object_relations=True)
