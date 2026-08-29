from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.real import rm65_composition as composition_module
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
from dataset_contracts.pose_frames import load_pose_frame_contract
from robot_motion.primitives import level_tcp_orientation
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.rm65 import (
    build_rm65_dual_arm_scene_profile,
    load_world_from_arm_base_transforms,
)
from sim.mujoco.deploy.target_config import DEFAULT_STATION_POSE_CONTRACT_PATH
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.skill_deploy import rm65_profile as profile_module
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
                    REPO_ROOT
                    / "hardware/rm65/utils/configs/calibrations/"
                    "rm65_left_inspire.json"
                ),
                "right": (
                    REPO_ROOT
                    / "hardware/rm65/utils/configs/calibrations/"
                    "rm65_right_inspire.json"
                ),
            },
            envelope_profile_path=(
                REPO_ROOT
                / "teleop_runtime/configs/skill_deploy/"
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


def test_composition_binds_existing_gateway_to_one_lower_graph_without_start(
    monkeypatch,
) -> None:
    lower, guard = _lower(monkeypatch)
    socket_path = REPO_ROOT / "temp/test-real-rm65-composition.sock"

    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=socket_path,
        authorized_side="right",
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
    assert composition.port_config.active_sides == ("right",)
    assert composition.port_config.exclusive_command_claim is True
    assert composition.attachment.component_ids == ("right_arm",)
    assert composition.attachment.reference_frame_id == (
        RM65_SKILL_EXECUTION_FRAME_ID_V1
    )
    assert RM65_SKILL_STATION_FRAME_ID_V1 in composition.frame_graph.frame_by_id
    assert len(composition.safety_envelope_ref.allowed_route_ids) == 10
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
    scene = build_rm65_dual_arm_scene_profile(
        lower.profile.collision_scene_profile
    )
    scene_world_from_base = load_world_from_arm_base_transforms(scene)
    sim_station_from_base = (
        sim_contract.source.effective_raw_to_canonical_desk_transform("right")
    )
    sim_world_from_station = (
        scene_world_from_base["right"] @ np.linalg.inv(sim_station_from_base)
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
    for direction, (canonical_delta, expected_world_delta) in (
        direction_by_name.items()
    ):
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

    station_frame = composition.frame_graph.frame_by_id[
        RM65_SKILL_STATION_FRAME_ID_V1
    ]
    canonical_anchor = np.eye(4, dtype=float)
    canonical_anchor[:3, 3] = (0.25, -0.10, 0.35)
    fixed_heading = level_tcp_orientation("forward_left_30").apply(
        canonical_anchor,
        reference_frame_spec=station_frame,
    ).target_pose
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
            REPO_ROOT
            / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
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
        assert composition.perception_capture.depth.routes[0].route_version == "2"
        assert composition.perception_capture.depth_binding.frame_graph_hash == (
            composition.frame_graph.semantic_content_hash
        )
        assert session.service.protocol_runtime.session_capabilities.artifact_transfer_enabled
        assert session.rgb_executor is not None
        assert session.depth_executor is not None
        assert len(session.contract_bundle.route_manifest.skill_routes) == 13
        effects = {
            entry.route.skill_id: entry.route.control_effect_class
            for entry in session.contract_bundle.route_manifest.skill_routes
        }
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
            (
                REPO_ROOT / "robot_motion/configs/rm65_camera_registry.json"
            ).read_text(encoding="utf-8")
        )
        parent_from_camera = np.asarray(
            registry["cameras"]["front"]["extrinsics"]["frames"]
            ["head_camera_frame"]["T_camera_to_parent"],
            dtype=float,
        )
        expected_world_from_camera = (
            lower.profile.scene_world_from_arm_base_by_side["left"]
            @ parent_from_camera
        )
        np.testing.assert_allclose(
            composition.perception_source_binding
            .world_from_depth_camera_by_source["front"],
            expected_world_from_camera,
        )
        pose_station_camera = (
            composition.perception_frame_binding
            .camera_pose_reference_xyz_quat_xyzw(
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


def test_composition_requires_complete_perception_attachment(monkeypatch, tmp_path) -> None:
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
            socket_path=(
                REPO_ROOT / "temp/absent-real-rm65-parent/gateway.sock"
            ),
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
