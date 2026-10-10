from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import rm65_composition as composition_module
from agent_gateway.profiles.real.gateway_cli import (
    DEFAULT_RM65_CAMERA_MOUNT_URDF,
    DEFAULT_RM65_CAMERA_REGISTRY_V4,
)
from agent_gateway.profiles.real.grasp_approach_plan import (
    RealGraspApproachPlan,
    RealGraspApproachPlanLookupError,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1,
)
from agent_gateway.profiles.real.motion_primitive_skill import (
    RealAgentPrimitiveSkillLifecycle,
)
from agent_gateway.profiles.real.one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
)
from agent_gateway.profiles.real.realsense_perception import (
    ExclusiveRealPerceptionResources,
)
from agent_gateway.profiles.real.rm65_composition import (
    RM65_SKILL_EXECUTION_FRAME_ID_V1,
    RM65_SKILL_STATION_FRAME_ID_V1,
    assemble_exclusive_rm65_agent_primitive_composition,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from dataset_contracts.pose_frames import load_pose_frame_contract
from robot_motion.primitives import level_tcp_orientation
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.rm65 import (
    build_rm65_dual_arm_scene_profile,
    load_world_from_arm_base_transforms,
)
from sim.mujoco.deploy.target_config import DEFAULT_STATION_POSE_CONTRACT_PATH
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.skill_deploy import rm65_profile as profile_module
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentJointTrajectory,
    RealAgentMotionPhase,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspReachabilityResult,
)
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    assemble_exclusive_rm65_skill_lower,
)
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _preflight_without_af_unix_path_binding(monkeypatch) -> None:
    """The isolated worktree path itself exceeds Linux's AF_UNIX limit."""

    monkeypatch.setattr(
        composition_module,
        "validate_agent_gateway_unix_server_preflight",
        lambda *, socket_path, peer_idle_timeout_sec: (
            str(socket_path),
            float(peer_idle_timeout_sec),
        ),
    )
    monkeypatch.setattr(
        composition_module,
        "validate_agent_gateway_unix_group_parent",
        lambda *, socket_path: 1000,
    )


class _Backend(EffectorBackend):
    def __init__(self, config) -> None:
        self.config = config
        self.starts = 0

    @property
    def name(self) -> str:
        return f"{self.config.side}_arm"

    def start(self) -> None:
        self.starts += 1

    def send_command(self, command):
        raise AssertionError("default-off composition must not dispatch")


class _TargetBuilder:
    def __init__(self) -> None:
        self.pre_ik_target_safety_filter = None

    def set_current_qpos_pair(self, **kwargs) -> None:
        raise AssertionError("default-off composition must not synchronize")

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side,
        solver_world_target,
    ):
        return np.asarray(solver_world_target, dtype=float).copy()

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

    def tool_tcp_world_targets_for_qpos_pair(self, **kwargs):
        raise AssertionError("default-off composition must not read state")

    def solver_world_targets_for_qpos_pair(self, **kwargs):
        raise AssertionError("default-off composition must not read state")


class _Localizer:
    model_spec = SimpleNamespace(
        model_profile_id="test.sam2-grounding-dino",
        semantic_content_hash="localization-model:test",
    )
    semantic_content_hash = "localization-implementation:test"

    def start(self) -> None:
        raise AssertionError("default-off composition must not start models")

    def close(self) -> None:
        return None

    def localize(self, *_args, **_kwargs):
        raise AssertionError("default-off composition must not infer")

    def segment(self, *_args, **_kwargs):
        raise AssertionError("default-off composition must not infer")

    def encode_segment_artifacts(self, *_args, **_kwargs):
        raise AssertionError("default-off composition must not encode")


class _PointGraspPlanner:
    model_spec = SimpleNamespace(
        model_profile_id="test.vgn-point-grasp",
        semantic_content_hash="point-grasp-model:test",
    )
    semantic_content_hash = "point-grasp-implementation:test"
    backend_id = "vgn"

    def __init__(self, point_segmenter) -> None:
        self.point_segmenter = point_segmenter

    def start(self) -> None:
        raise AssertionError("default-off composition must not start models")

    def close(self) -> None:
        return None

    def plan(self, *_args, **_kwargs):
        raise AssertionError("default-off composition must not infer")


class _ForceGuard:
    motion_ready = False
    fault = None

    def __init__(self) -> None:
        self.starts = 0

    def start(self, *, arm_stale_watchdog=True):
        self.starts += 1
        return {"motion_ready": False}

    def close(self) -> None:
        return None


def _lower(monkeypatch, *, envelope_side: str = "right"):
    monkeypatch.setattr(
        profile_module,
        "load_assignment",
        lambda path: {
            "assignments": {
                "left": {"arm_ip": "192.0.2.10"},
                "right": {"arm_ip": "192.0.2.11"},
            }
        },
    )
    profile = resolve_exclusive_rm65_skill_deployment_profile(
        ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=REPO_ROOT / "README.md",
            tool_profile_path_by_side={
                "left": (
                    REPO_ROOT / "hardware/rm65/utils/configs/calibrations/"
                    "rm65_left_inspire.json"
                ),
                "right": (
                    REPO_ROOT / "hardware/rm65/utils/configs/calibrations/"
                    "rm65_right_inspire.json"
                ),
            },
            envelope_profile_path=(
                REPO_ROOT / "teleop_runtime/configs/skill_deploy/"
                f"agent_primitive_{envelope_side}_relative_tight_v1.json"
            ),
        )
    )
    target_builder = _TargetBuilder()
    guard = _ForceGuard()
    lower = assemble_exclusive_rm65_skill_lower(
        profile=profile,
        backend_factory=_Backend,
        target_builder_factory=lambda config: target_builder,
        force_guard_factory=lambda backends: guard,
    )
    return lower, guard


@pytest.mark.parametrize("authorized_side", ("left", "right"))
def test_composition_binds_selected_side_to_one_lower_graph_without_start(
    monkeypatch, authorized_side: str
) -> None:
    lower, guard = _lower(monkeypatch, envelope_side=authorized_side)
    socket_path = REPO_ROOT / f"temp/test-real-rm65-{authorized_side}-composition.sock"

    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=socket_path,
        authorized_side=authorized_side,
        authorized_action="forward",
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test",
        client_uid=1000,
        ssh_read_only_client_id_by_uid={
            1101: "rpent.remote.alice",
            1102: "rpent.remote.bob",
        },
        runtime_session_id="real-rm65:test",
        control_domain_id="exclusive-real-skill:test",
        embodiment_instance_id="dual-rm65:test",
    )

    assert composition.deployment.control_runtime.motion_port is (
        composition.deployment.gateway_session.port
    )
    assert composition.application.resources is lower.ports
    assert composition.application.startup_admission is lower.startup_admission
    assert composition.port_config.active_sides == (authorized_side,)
    assert composition.port_config.exclusive_command_claim is True
    assert composition.attachment.component_ids == (f"{authorized_side}_arm",)
    assert composition.authorized_side == authorized_side
    assert composition.attachment.reference_frame_id == (
        RM65_SKILL_EXECUTION_FRAME_ID_V1
    )
    assert RM65_SKILL_STATION_FRAME_ID_V1 in composition.frame_graph.frame_by_id
    assert len(composition.safety_envelope_ref.allowed_route_ids) == 12
    assert (
        "real.rm65.agent_skill_grasp_approach"
        in composition.safety_envelope_ref.allowed_route_ids
    )
    assert (
        REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1
        in composition.safety_envelope_ref.allowed_route_ids
    )
    assert (
        composition.deployment.gateway_session.contract_bundle.pose_target_profile
        is not None
    )
    assert composition.authorization.grant_issued is False
    assert composition.read_only_client_ids == (
        "rpent.remote.alice",
        "rpent.remote.bob",
    )
    assert composition.socket_mode == 0o660
    assert (
        composition.deployment.gateway_session.service.access_policy.allowed_client_ids
    ) == ("rpent.remote.alice", "rpent.remote.bob", "rpent:test")
    assert composition.application.config.control_hz == (
        lower.profile.position_profile.rates.reference_hz
    )
    assert all(backend.starts == 0 for backend in lower.arm_backends.values())
    assert guard.starts == 0
    assert not socket_path.exists()


def test_composition_matches_mujoco_canonical_station_axes_and_pose(
    monkeypatch,
) -> None:
    lower, _guard = _lower(monkeypatch)
    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=REPO_ROOT / "temp/test-real-rm65-frame-parity.sock",
        authorized_side="right",
        authorized_action="forward",
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test",
        client_uid=1000,
        runtime_session_id="real-rm65:frame-parity-test",
        control_domain_id="exclusive-real-skill:frame-parity-test",
        embodiment_instance_id="dual-rm65:frame-parity-test",
    )
    execution_from_station = composition.frame_graph.resolve(
        RM65_SKILL_STATION_FRAME_ID_V1,
        RM65_SKILL_EXECUTION_FRAME_ID_V1,
    ).matrix_target_source
    station_from_execution = composition.frame_graph.resolve(
        RM65_SKILL_EXECUTION_FRAME_ID_V1,
        RM65_SKILL_STATION_FRAME_ID_V1,
    ).matrix_target_source
    np.testing.assert_allclose(
        station_from_execution,
        np.linalg.inv(lower.profile.scene_world_from_canonical_station),
        atol=1e-12,
    )

    sim_contract = load_pose_frame_contract(DEFAULT_STATION_POSE_CONTRACT_PATH)
    scene = build_rm65_dual_arm_scene_profile(lower.profile.collision_scene_profile)
    scene_world_from_base = load_world_from_arm_base_transforms(scene)
    sim_station_from_base = (
        sim_contract.source.effective_raw_to_canonical_desk_transform("right")
    )
    sim_world_from_station = scene_world_from_base["right"] @ np.linalg.inv(
        sim_station_from_base
    )

    np.testing.assert_allclose(
        execution_from_station,
        sim_world_from_station,
        atol=1e-12,
    )
    direction_by_name = {
        "forward": ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        "backward": ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "left": ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        "right": ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        "up": ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        "down": ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)),
    }
    for direction, (canonical_delta, expected_world_delta) in direction_by_name.items():
        np.testing.assert_allclose(
            execution_from_station[:3, :3] @ canonical_delta,
            expected_world_delta,
            atol=1e-12,
            err_msg=direction,
        )
        np.testing.assert_allclose(
            sim_world_from_station[:3, :3] @ canonical_delta,
            expected_world_delta,
            atol=1e-12,
            err_msg=direction,
        )

    station_frame = composition.frame_graph.frame_by_id[RM65_SKILL_STATION_FRAME_ID_V1]
    canonical_anchor = np.eye(4, dtype=float)
    canonical_anchor[:3, 3] = (0.25, -0.10, 0.35)
    fixed_heading = (
        level_tcp_orientation("forward_left_30")
        .apply(
            canonical_anchor,
            reference_frame_spec=station_frame,
        )
        .target_pose
    )
    np.testing.assert_allclose(
        execution_from_station @ fixed_heading,
        sim_world_from_station @ fixed_heading,
        atol=1e-12,
    )


def test_composition_opt_in_binds_real_rgb_depth_and_artifact_plane_without_start(
    monkeypatch,
    tmp_path,
) -> None:
    lower, guard = _lower(monkeypatch)
    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=REPO_ROOT / "temp/test-real-rm65-perception.sock",
        authorized_side="right",
        authorized_action="forward",
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test",
        client_uid=1000,
        runtime_session_id="real-rm65:perception-test",
        control_domain_id="exclusive-real-skill:perception-test",
        embodiment_instance_id="dual-rm65:perception-test",
        camera_config_path=(
            REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
        ),
        camera_registry_path=(
            REPO_ROOT / "robot_motion/configs/rm65_camera_registry.json"
        ),
        artifact_root=tmp_path,
    )
    session = composition.deployment.gateway_session
    try:
        assert isinstance(composition.resources, ExclusiveRealPerceptionResources)
        assert composition.application.resources is composition.resources
        assert composition.perception_capture is not None
        assert composition.perception_capture.profile.rgb_roles == (
            "main",
            "wrist_left",
            "wrist_right",
        )
        assert composition.perception_capture.profile.depth_roles == ("main",)
        assert composition.perception_frame_binding is not None
        assert composition.perception_capture.frame_binding is (
            composition.perception_frame_binding
        )
        assert composition.perception_capture.rgb.routes[0].route_version == "2"
        assert composition.perception_capture.depth.routes[0].route_version == "2"
        assert composition.perception_capture.depth_binding.frame_graph_hash == (
            composition.frame_graph.semantic_content_hash
        )
        assert session.service.protocol_runtime.session_capabilities.artifact_transfer_enabled
        assert session.rgb_executor is not None
        assert session.depth_executor is not None
        assert len(session.contract_bundle.route_manifest.skill_routes) == 16
        effects = {
            entry.route.skill_id: entry.route.control_effect_class
            for entry in session.contract_bundle.route_manifest.skill_routes
        }
        assert effects["motion.move_tcp_to_pose"] == "motion"
        assert effects["perception.capture_rgb"] == "read_only"
        assert effects["perception.capture_depth"] == "read_only"
        assert effects["state.read_tcp_pose"] == "read_only"
        assert all(backend.starts == 0 for backend in lower.arm_backends.values())
        assert guard.starts == 0
        assert all(
            source.device_info()["running"] is False
            for source in composition.perception_source_binding.camera_sources.values()
        )

        registry = json.loads(
            (REPO_ROOT / "robot_motion/configs/rm65_camera_registry.json").read_text(
                encoding="utf-8"
            )
        )
        parent_from_camera = np.asarray(
            registry["cameras"]["front"]["extrinsics"]["frames"]["head_camera_frame"][
                "T_camera_to_parent"
            ],
            dtype=float,
        )
        expected_world_from_camera = (
            lower.profile.scene_world_from_arm_base_by_side["left"] @ parent_from_camera
        )
        np.testing.assert_allclose(
            composition.perception_source_binding.world_from_depth_camera_by_source[
                "front"
            ],
            expected_world_from_camera,
        )
        pose_station_camera = (
            composition.perception_frame_binding.camera_pose_reference_xyz_quat_xyzw(
                pose_matrix_to_xyzquat(expected_world_from_camera)
            )
        )
        expected_station_from_camera = (
            np.linalg.inv(lower.profile.scene_world_from_canonical_station)
            @ expected_world_from_camera
        )
        np.testing.assert_allclose(
            xyzquat_to_pose_matrix(pose_station_camera),
            expected_station_from_camera,
            atol=1e-6,
        )
    finally:
        assert session.artifact_store is not None
        session.artifact_store.close()


def test_composition_requires_complete_perception_attachment(
    monkeypatch, tmp_path
) -> None:
    lower, _guard = _lower(monkeypatch)
    with pytest.raises(ValueError, match="provided together"):
        assemble_exclusive_rm65_agent_primitive_composition(
            lower=lower,
            socket_path=REPO_ROOT / "temp/test-real-rm65-perception.sock",
            authorized_side="right",
            authorized_action="forward",
            confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
            client_id="rpent:test",
            client_uid=1000,
            runtime_session_id="real-rm65:perception-test",
            control_domain_id="exclusive-real-skill:perception-test",
            embodiment_instance_id="dual-rm65:perception-test",
            artifact_root=tmp_path,
        )


@pytest.mark.parametrize("frozen_observation", [False, True])
def test_composition_opt_in_closes_v4_three_view_grasp_and_motion_routes(
    monkeypatch,
    tmp_path,
    frozen_observation,
) -> None:
    lower, guard = _lower(monkeypatch)
    localizer = _Localizer()
    planner = _PointGraspPlanner(localizer)
    if frozen_observation:
        planner.backend_id = "graspgenx"

    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=REPO_ROOT / "temp/test-real-rm65-grasp.sock",
        authorized_side="right",
        authorized_action="move_tcp_to_pose",
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test",
        client_uid=1000,
        runtime_session_id="real-rm65:grasp-test",
        control_domain_id="exclusive-real-skill:grasp-test",
        embodiment_instance_id="dual-rm65:grasp-test",
        camera_config_path=(
            REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
        ),
        camera_registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        artifact_root=tmp_path,
        camera_mount_urdf_path=DEFAULT_RM65_CAMERA_MOUNT_URDF,
        object_localizer=localizer,
        point_grasp_planner=planner,
        debug_render_object_coordinates=True,
        enable_grasp_observation=frozen_observation,
    )
    session = composition.deployment.gateway_session
    try:
        routes = {
            entry.route.skill_id: entry.route.control_effect_class
            for entry in session.contract_bundle.route_manifest.skill_routes
        }
        assert routes["perception.segment"] == "read_only"
        assert routes["perception.plan_placement"] == "read_only"
        assert routes["perception.plan_point_grasp"] == "read_only"
        assert routes["perception.render_point_grasp_candidate"] == "read_only"
        assert routes["perception.plan_grasp_approach"] == "read_only"
        assert routes["motion.execute_grasp_approach_plan"] == "motion"
        assert routes["motion.move_tcp_to_pose"] == "motion"
        assert "own_domain_emergency_stop_v1" in (
            session.contract_bundle.route_manifest.supported_transport_features
        )
        assert session.emergency_stop_port is not None
        assert "agent_estop" not in (
            session.contract_bundle.route_manifest.explicit_exclusions
        )
        assert composition.station_point_grasp_registration is not None
        assert composition.segmentation_registration is not None
        assert session.segmentation_executor is not None
        assert session.placement_executor is not None
        assert (
            composition.station_point_grasp_registration.skill.definition.skill_version
            == ("4" if frozen_observation else "2")
        )
        assert (
            composition.station_point_grasp_registration.profile.arm_reachability_profile
            is None
        )
        assert composition.station_point_grasp_render_registration is not None
        assert tuple(
            composition.station_point_grasp_registration.profile.source_by_role
        ) == ("main", "wrist_left", "wrist_right")
        assert composition.perception_capture is not None
        assert composition.perception_capture.profile.depth_roles == ("main",)
        assert tuple(
            composition.station_point_grasp_render_registration.profile.source_by_role
        ) == ("main", "wrist_left", "wrist_right")
        assert (
            composition.station_point_grasp_render_registration.profile.debug_target_coordinate_overlay
            is True
        )
        assert session.station_point_grasp_executor is not None
        assert session.station_point_grasp_render_executor is not None
        expected_reads = {
            "state.read_tcp_pose",
            "perception.capture_rgb",
            "perception.capture_depth",
            "perception.back_project",
            "perception.localize_objects",
            "perception.segment",
            "perception.plan_placement",
            "perception.plan_point_grasp",
            "perception.render_point_grasp_candidate",
            "perception.plan_grasp_approach",
        }
        if frozen_observation:
            expected_reads.add("perception.capture_grasp_observation")
            assert routes["perception.capture_grasp_observation"] == "read_only"
            assert callable(session.station_point_grasp_provider.observation_reader)
        assert set(session.passive_read_executors_by_skill_id) == expected_reads
        assert session.owns_artifact_store is False
        assert composition.resources.artifact_store is session.artifact_store
        assert all(backend.starts == 0 for backend in lower.arm_backends.values())
        assert guard.starts == 0
    finally:
        assert session.artifact_store is not None
        session.artifact_store.close()


def test_preplanned_grasp_execution_consumes_exact_path_without_replanning(
    monkeypatch,
    tmp_path,
) -> None:
    lower, _guard = _lower(monkeypatch)
    localizer = _Localizer()
    point_grasp_planner = _PointGraspPlanner(localizer)
    now = [10.0]
    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=REPO_ROOT / "temp/test-real-rm65-grasp-plan-execution.sock",
        authorized_side="right",
        authorized_action="execute_grasp_approach_plan",
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test",
        client_uid=1000,
        runtime_session_id="real-rm65:grasp-plan-execution-test",
        control_domain_id="exclusive-real-skill:grasp-plan-execution-test",
        embodiment_instance_id="dual-rm65:grasp-plan-execution-test",
        camera_config_path=(
            REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
        ),
        camera_registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        artifact_root=tmp_path,
        camera_mount_urdf_path=DEFAULT_RM65_CAMERA_MOUNT_URDF,
        object_localizer=localizer,
        point_grasp_planner=point_grasp_planner,
        clock=lambda: now[0],
    )
    session = composition.deployment.gateway_session
    try:
        registration = session.contract_bundle.grasp_approach_plan_registration
        store = session.grasp_approach_plan_store
        assert registration is not None
        assert store is not None
        lower.target_builder.tool_tcp_world_targets_for_qpos_pair = lambda **_kwargs: {
            "left": np.eye(4, dtype=float),
            "right": np.eye(4, dtype=float),
        }
        qpos_by_side = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        session.port.publish_measured_qpos_pair(
            qpos_by_side=qpos_by_side,
            measured_at=now[0],
            available_at=now[0],
            source_evidence_refs=("measured-qpos:grasp-plan-execution-test",),
        )
        qpos_cut = session.port.latest_measured_qpos()
        assert qpos_cut is not None
        trajectory = RealAgentJointTrajectory(
            time_from_start_sec=np.array([0.0, 0.5], dtype=float),
            qpos_by_side={
                side: np.stack((qpos_by_side[side], qpos_by_side[side]))
                for side in ("left", "right")
            },
            planner_evidence_refs=("rm65-production-urdf-fcl-path:execution-test",),
        )
        target_execution = np.eye(4, dtype=float)
        target_reference, _resolution = composition.frame_graph.transform_pose_matrix(
            target_execution,
            source_reference_frame_id=registration.profile.execution_reference_frame_id,
            target_reference_frame_id=registration.profile.reference_frame_id,
        )
        plan = RealGraspApproachPlan(
            runtime_session_id=registration.binding.runtime_session_id,
            profile_hash=registration.profile.semantic_content_hash,
            planning_operation_id="grasp-plan-operation:execution-test",
            planning_request_hash="1" * 64,
            point_grasp_operation_id="point-grasp-operation:execution-test",
            point_grasp_request_hash="2" * 64,
            point_grasp_result_hash="3" * 64,
            point_grasp_candidate_hash="4" * 64,
            source_station_revision=0,
            component_id=registration.profile.component_id,
            side=registration.profile.side,
            target_pose_reference=target_reference,
            target_pose_execution=target_execution,
            speed_scale=0.2,
            measured_qpos_revision=qpos_cut.revision,
            measured_qpos_hash=qpos_cut.semantic_content_hash,
            measured_qpos_at=qpos_cut.measured_at,
            trajectory=trajectory,
            created_at=now[0],
            expires_at=now[0] + 30.0,
            evidence_refs=(
                *qpos_cut.evidence_refs,
                *trajectory.evidence_refs,
                "point-grasp-receipt:execution-test",
            ),
        )
        store.put(plan, now=now[0])

        entry = session.contract_bundle.route_manifest.route(
            "motion.execute_grasp_approach_plan",
            "1",
        )
        route = entry.route
        binding = session.contract_bundle.binding_for_skill(route.skill_id)
        request = AgentSkillRequest(
            client_id="rpent:test",
            request_id="grasp-plan-execution-request:test",
            idempotency_key="grasp-plan-execution-idempotency:test",
            client_sequence=1,
            runtime_session_id=registration.binding.runtime_session_id,
            skill_route_manifest_hash=(
                session.contract_bundle.route_manifest.semantic_content_hash
            ),
            expected_station_revision=0,
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=binding.semantic_content_hash,
            parameter_profile_id=registration.profile.parameter_profile_id,
            parameter_profile_hash=registration.profile.semantic_content_hash,
            parameters={
                "component_id": registration.profile.component_id,
                "grasp_approach_plan_id": plan.plan_id,
                "grasp_approach_plan_hash": plan.semantic_content_hash,
            },
            observation_snapshot_id="observation:grasp-plan-execution-test",
            requested_mode="real_active",
            requested_timeout_ms=5_000,
            skill_execution_grant_id="agent-skill-grant:" + "5" * 64,
        )
        operation = AgentSkillOperation(
            operation_id="grasp-plan-execution-operation:test",
            request_hash=request.semantic_content_hash,
            runtime_session_id=request.runtime_session_id,
            status="admitted",
            reason_code="admitted",
            station_revision_at_admission=0,
            current_station_revision=0,
            started_at=now[0],
            updated_at=now[0],
            deadline_at=now[0] + 5.0,
        )

        class _Owner:
            def __init__(self) -> None:
                self.state_machine = SimpleNamespace(state=TeleopRunState.ARMED)

            def hold_hardware(
                self,
                _reason,
                *,
                require_home_recalibration=False,
            ) -> bool:
                del require_home_recalibration
                self.state_machine.state = TeleopRunState.DEBUG_ONLY
                return True

        owner = _Owner()
        replanning_calls = 0

        def forbidden_replan(**_kwargs):
            nonlocal replanning_calls
            replanning_calls += 1
            raise AssertionError("The execution Skill must not invoke a planner.")

        def apply_lower_owner(duration: float) -> None:
            now[0] += duration
            session.port.apply_owner_requests(session=owner, now=now[0])

        lifecycle = RealAgentPrimitiveSkillLifecycle(
            port=session.port,
            contract_bundle=session.contract_bundle,
            route=route,
            binding=binding,
            profile=registration.profile,
            request=request,
            operation=operation,
            grasp_trajectory_planner=forbidden_replan,
            grasp_approach_plan_store=store,
            clock=lambda: now[0],
            sleep=apply_lower_owner,
        )

        activation = lifecycle.activate(
            request=request,
            operation=operation,
            cancel_requested=lambda: False,
        )

        assert activation.succeeded is True
        assert session.port.snapshot().phase == RealAgentMotionPhase.ACTIVE
        assert replanning_calls == 0
        assert lifecycle._planned_joint_trajectory is trajectory
        with pytest.raises(
            RealGraspApproachPlanLookupError,
            match="grasp_path_plan_already_consumed",
        ):
            store.reserve(
                plan_id=plan.plan_id,
                plan_hash=plan.semantic_content_hash,
                operation_id="grasp-plan-execution-operation:replay",
                runtime_session_id=registration.binding.runtime_session_id,
                profile_hash=registration.profile.semantic_content_hash,
                component_id=registration.profile.component_id,
                expected_station_revision=0,
                now=now[0],
            )
    finally:
        assert session.artifact_store is not None
        session.artifact_store.close()


def test_composition_opt_in_binds_point_grasp_v3_to_lower_ik_without_starting(
    monkeypatch,
    tmp_path,
) -> None:
    lower, guard = _lower(monkeypatch)
    localizer = _Localizer()
    planner = _PointGraspPlanner(localizer)

    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=REPO_ROOT / "temp/test-real-rm65-grasp-reachability.sock",
        authorized_side="right",
        authorized_action="move_tcp_to_pose",
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test",
        client_uid=1000,
        runtime_session_id="real-rm65:grasp-reachability-test",
        control_domain_id="exclusive-real-skill:grasp-reachability-test",
        embodiment_instance_id="dual-rm65:grasp-reachability-test",
        camera_config_path=(
            REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
        ),
        camera_registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        artifact_root=tmp_path,
        camera_mount_urdf_path=DEFAULT_RM65_CAMERA_MOUNT_URDF,
        object_localizer=localizer,
        point_grasp_planner=planner,
        enable_point_grasp_arm_reachability=True,
        clock=lambda: 10.0,
    )
    session = composition.deployment.gateway_session
    try:
        registration = composition.station_point_grasp_registration
        assert registration is not None
        assert registration.skill.definition.skill_version == "3"
        reachability = registration.profile.arm_reachability_profile
        assert reachability is not None
        assert reachability.arm_side == "right"
        assert reachability.solver_reference_frame_id == (
            RM65_SKILL_EXECUTION_FRAME_ID_V1
        )
        assert session.station_point_grasp_provider is not None
        assert callable(session.station_point_grasp_provider.arm_reachability_evaluator)
        lower.target_builder.tool_tcp_world_targets_for_qpos_pair = lambda **_kwargs: {
            "left": np.eye(4),
            "right": np.eye(4),
        }
        evaluator = session.station_point_grasp_provider.arm_reachability_evaluator
        session.port.publish_measured_qpos_pair(
            qpos_by_side={
                "left": np.zeros(6),
                "right": np.zeros(6),
            },
            measured_at=9.0,
            available_at=9.01,
            source_evidence_refs=("measured-qpos:stale-test",),
        )
        with pytest.raises(RuntimeError, match="measured qpos is stale"):
            evaluator((np.eye(4),))

        session.port.publish_measured_qpos_pair(
            qpos_by_side={
                "left": np.zeros(6),
                "right": np.zeros(6),
            },
            measured_at=9.95,
            available_at=9.96,
            source_evidence_refs=("measured-qpos:test",),
        )
        captured = []

        def assess(**kwargs):
            captured.append(kwargs)
            return RM65GraspReachabilityResult(
                classification="standard_reachable",
                position_residual_m=0.0,
                orientation_residual_deg=0.0,
                solve_cycles=1,
                elapsed_ms=1.0,
                termination_reason="standard_tolerance_met",
            )

        evaluator.planner.assess_endpoint_ik = assess
        context, annotations = evaluator((np.eye(4),))
        np.testing.assert_allclose(
            captured[0]["target_tool_tcp_world"],
            lower.profile.scene_world_from_canonical_station,
        )
        assert context["candidate_reference_frame_id"] == "canonical_station"
        assert context["measured_qpos_age_ms"] == pytest.approx(50.0)
        assert annotations[0]["classification"] == "standard_reachable"
        evaluator.clock = lambda: 10.3
        matched_cut = session.port.latest_measured_qpos()
        assert matched_cut is not None
        matched_context, _ = evaluator(
            (np.eye(4),),
            measured_qpos_cut=matched_cut,
            measurement_reference_time=10.0,
        )
        assert matched_context["measured_qpos_age_ms"] == pytest.approx(50.0)
        assert all(backend.starts == 0 for backend in lower.arm_backends.values())
        assert guard.starts == 0
    finally:
        assert session.artifact_store is not None
        session.artifact_store.close()


def test_composition_rejects_envelope_for_another_side(monkeypatch) -> None:
    lower, _guard = _lower(monkeypatch, envelope_side="left")

    with pytest.raises(ValueError, match="does not actively constrain"):
        assemble_exclusive_rm65_agent_primitive_composition(
            lower=lower,
            socket_path=REPO_ROOT / "temp/test-real-rm65-composition.sock",
            authorized_side="right",
            authorized_action="forward",
            confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
            client_id="rpent:test",
            client_uid=1000,
            runtime_session_id="real-rm65:test",
            control_domain_id="exclusive-real-skill:test",
            embodiment_instance_id="dual-rm65:test",
        )


def test_programmatic_composition_still_requires_exact_confirmation(
    monkeypatch,
) -> None:
    lower, _guard = _lower(monkeypatch)

    with pytest.raises(ValueError, match="exact real primitive confirmation"):
        assemble_exclusive_rm65_agent_primitive_composition(
            lower=lower,
            socket_path=REPO_ROOT / "temp/test-real-rm65-composition.sock",
            authorized_side="right",
            authorized_action="forward",
            confirmation_token="wrong",
            client_id="rpent:test",
            client_uid=1000,
            runtime_session_id="real-rm65:test",
            control_domain_id="exclusive-real-skill:test",
            embodiment_instance_id="dual-rm65:test",
        )


def test_composition_rejects_socket_config_before_application_start(
    monkeypatch,
) -> None:
    lower, guard = _lower(monkeypatch)

    def reject_socket(**kwargs):
        raise ValueError("Unix socket parent directory must already exist.")

    monkeypatch.setattr(
        composition_module,
        "validate_agent_gateway_unix_server_preflight",
        reject_socket,
    )

    with pytest.raises(ValueError, match="parent directory must already exist"):
        assemble_exclusive_rm65_agent_primitive_composition(
            lower=lower,
            socket_path=(REPO_ROOT / "temp/absent-real-rm65-parent/gateway.sock"),
            authorized_side="right",
            authorized_action="forward",
            confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
            client_id="rpent:test",
            client_uid=1000,
            runtime_session_id="real-rm65:test",
            control_domain_id="exclusive-real-skill:test",
            embodiment_instance_id="dual-rm65:test",
        )

    assert all(backend.starts == 0 for backend in lower.arm_backends.values())
    assert guard.starts == 0


def test_rm65_composition_import_has_no_realteleop_or_composition_dependency() -> None:
    script = """
import json
import sys
import agent_gateway.profiles.real.rm65_composition
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'control_composition'
    or name.startswith('control_composition.')
    or name == 'teleop_runtime.real_teleop'
    or name.startswith('teleop_runtime.real_teleop.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
