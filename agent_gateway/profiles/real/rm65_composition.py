"""Trusted composition root for one independent real RM65 Skill process.

This module binds the neutral Agent Gateway Skill contracts to an already
assembled lower RM65 owner.  Construction is default-off: it opens no device,
socket, thread, or motion authority.  RealTeleop, VIVE, collection, GUI, and
``control_composition`` are deliberately absent from this dependency graph.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentSafetyEnvelopeRef,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_group_parent,
    validate_agent_gateway_unix_server_preflight,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from teleop_runtime.skill_deploy import (
    RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
    ExclusiveRM65SkillLowerAssembly,
    RealAgentMotionPortConfig,
    build_exclusive_rm65_skill_target_builder,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
    RM65GraspReachabilityConfig,
)

from .exclusive_application import (
    ExclusiveRealSkillApplication,
    ExclusiveRealSkillApplicationConfig,
)
from .exclusive_factory import (
    ExclusiveRealAgentPrimitiveDeployment,
    assemble_exclusive_rm65_agent_primitive_deployment,
)
from .localization_deployment import (
    RealObjectLocalizationRegistration,
    build_real_object_localization_registration,
)
from .localization_gateway import ObjectLocalizationProvider
from .motion_primitive_profiles import (
    real_agent_motion_route_ids_v1,
)
from .observation_gateway import RealTcpPoseObserveAttachmentSpec
from .one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_ACTIONS,
    RealAgentPrimitiveOneShotAuthorization,
)
from .perception_deployment import (
    REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
    RealPerceptionCaptureRegistration,
    build_real_perception_capture_registration_v2,
)
from .perception_frames import (
    REAL_PERCEPTION_SOURCE_FRAME_ID_V1,
    REAL_PERCEPTION_STATION_FRAME_ID_V1,
    RealPerceptionFrameBinding,
    build_real_perception_station_frame_binding,
)
from .realsense_perception import (
    ExclusiveRealPerceptionResources,
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
    build_real_perception_source_binding,
)
from .rm65_camera_geometry import (
    RM65RgbCameraGeometry,
    build_rm65_rgb_camera_geometry,
)
from .rm65_grasp_reachability import RM65PointGraspReachabilityAdapter
from .station_point_grasp_deployment import (
    StationBoundPointGraspRegistration,
    StationPointGraspArmReachabilityProfile,
    build_station_bound_point_grasp_registration,
)
from .station_point_grasp_render_deployment import (
    StationBoundPointGraspRenderRegistration,
    build_station_bound_point_grasp_render_registration,
)

RM65_SKILL_EXECUTION_FRAME_ID_V1 = RM65_SKILL_SCENE_WORLD_FRAME_ID_V1
RM65_SKILL_STATION_FRAME_ID_V1 = "canonical_station"
RM65_SKILL_CLOCK_DOMAIN_ID_V1 = "python-time-time-wall-clock"


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if str(value) != normalized:
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return normalized


def _hash(payload: object, *, domain: str) -> str:
    return canonical_content_hash(payload, domain=domain)


def _public_tool_profiles(lower: ExclusiveRM65SkillLowerAssembly) -> dict[str, object]:
    return {
        side: {
            key: value
            for key, value in lower.profile.tool_profile_by_side[side].items()
            if not str(key).startswith("_")
        }
        for side in ("left", "right")
    }


def _route_ids() -> tuple[str, ...]:
    return real_agent_motion_route_ids_v1(include_grasp_approach=True)


@dataclass(frozen=True)
class ExclusiveRM65AgentPrimitiveComposition:
    """Inspectible result of one exact, still-default-off composition."""

    lower: ExclusiveRM65SkillLowerAssembly
    frame_graph: CoordinateFrameGraph
    attachment: RealTcpPoseObserveAttachmentSpec
    port_config: RealAgentMotionPortConfig
    safety_envelope_ref: SkillDeploymentSafetyEnvelopeRef
    deployment: ExclusiveRealAgentPrimitiveDeployment
    authorization: RealAgentPrimitiveOneShotAuthorization
    application: ExclusiveRealSkillApplication
    resources: object
    perception_source_binding: RealPerceptionSourceBinding | None
    perception_frame_binding: RealPerceptionFrameBinding | None
    perception_capture: RealPerceptionCaptureRegistration | None
    rgb_camera_geometry: RM65RgbCameraGeometry | None
    localization_registration: RealObjectLocalizationRegistration | None
    localization_provider: ObjectLocalizationProvider | None
    station_point_grasp_registration: StationBoundPointGraspRegistration | None
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    )
    point_grasp_planner: object | None
    socket_path: Path
    authorized_side: str
    authorized_action: str | None
    authorized_actions: tuple[str, ...]
    client_id: str
    client_uid: int
    read_only_client_ids: tuple[str, ...]
    socket_mode: int

    def __post_init__(self) -> None:
        if not isinstance(self.lower, ExclusiveRM65SkillLowerAssembly):
            raise TypeError("lower must be an ExclusiveRM65SkillLowerAssembly.")
        if self.deployment.control_runtime.motion_port.config is not self.port_config:
            raise ValueError("deployment and composition must share one port config.")
        if self.application.deployment is not self.deployment:
            raise ValueError("application must own the exact composed deployment.")
        if self.authorization.gateway_session is not self.deployment.gateway_session:
            raise ValueError("authorization must bind the exact Gateway session.")
        if self.application.resources is not self.resources:
            raise ValueError("application must own the exact composed resources.")
        if self.perception_capture is None:
            if self.resources is not self.lower.ports:
                raise ValueError("motion-only composition must own lower resources.")
            if (
                self.perception_source_binding is not None
                or self.perception_frame_binding is not None
            ):
                raise ValueError(
                    "motion-only composition cannot retain perception bindings."
                )
            if any(
                item is not None
                for item in (
                    self.rgb_camera_geometry,
                    self.localization_registration,
                    self.localization_provider,
                    self.station_point_grasp_registration,
                    self.station_point_grasp_render_registration,
                    self.point_grasp_planner,
                )
            ):
                raise ValueError(
                    "motion-only composition cannot retain grasp resources."
                )
        elif not isinstance(self.resources, ExclusiveRealPerceptionResources):
            raise ValueError("perception composition requires aggregate resources.")
        elif (
            self.perception_source_binding is None
            or self.perception_frame_binding is None
            or self.perception_capture.frame_binding
            is not self.perception_frame_binding
        ):
            raise ValueError("perception composition requires one exact frame binding.")
        grasp_items = (
            self.rgb_camera_geometry,
            self.localization_registration,
            self.localization_provider,
            self.station_point_grasp_registration,
            self.station_point_grasp_render_registration,
            self.point_grasp_planner,
        )
        if any(item is not None for item in grasp_items) and not all(
            item is not None for item in grasp_items
        ):
            raise ValueError("RM65 point-grasp resources must be assembled together.")
        if self.application.startup_admission is not self.lower.startup_admission:
            raise ValueError("application must use the exact lower startup gate.")


def assemble_exclusive_rm65_agent_primitive_composition(
    *,
    lower: ExclusiveRM65SkillLowerAssembly,
    socket_path: str | Path,
    authorized_side: str,
    authorized_action: str | None,
    confirmation_token: str,
    client_id: str,
    client_uid: int,
    ssh_read_only_client_id_by_uid: Mapping[int, str] | None = None,
    authorized_actions: tuple[str, ...] | None = None,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    max_retained_operations: int = 16,
    peer_idle_timeout_sec: float = 10.0,
    grant_lifetime_sec: float = 5.0,
    camera_config_path: str | Path | None = None,
    camera_registry_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    camera_mount_urdf_path: str | Path | None = None,
    object_localizer: object | None = None,
    point_grasp_planner: object | None = None,
    debug_render_object_coordinates: bool = False,
    enable_point_grasp_arm_reachability: bool = False,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ExclusiveRM65AgentPrimitiveComposition:
    """Bind one lower RM65 graph to the existing real Skill/Gateway contracts."""

    if not isinstance(lower, ExclusiveRM65SkillLowerAssembly):
        raise TypeError("lower must be an ExclusiveRM65SkillLowerAssembly.")
    if not isinstance(debug_render_object_coordinates, bool):
        raise TypeError("debug_render_object_coordinates must be bool.")
    if not isinstance(enable_point_grasp_arm_reachability, bool):
        raise TypeError("enable_point_grasp_arm_reachability must be bool.")
    side = str(authorized_side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("authorized_side must be left or right.")
    if authorized_actions is None:
        if authorized_action is None:
            raise ValueError("authorized_action or authorized_actions is required.")
        requested_actions = (authorized_action,)
    else:
        if authorized_action is not None:
            raise ValueError(
                "authorized_action and authorized_actions are mutually exclusive."
            )
        requested_actions = tuple(authorized_actions)
    normalized_actions = tuple(
        sorted({str(item).strip() for item in requested_actions})
    )
    if not normalized_actions or any(
        item not in REAL_AGENT_PRIMITIVE_ACTIONS for item in normalized_actions
    ):
        raise ValueError("authorized_actions contains an unreviewed primitive.")
    client = _required(client_id, field_name="client_id")
    runtime_id = _required(runtime_session_id, field_name="runtime_session_id")
    domain_id = _required(control_domain_id, field_name="control_domain_id")
    embodiment_id = _required(
        embodiment_instance_id,
        field_name="embodiment_instance_id",
    )
    uid = int(client_uid)
    if isinstance(client_uid, bool) or uid < 0:
        raise ValueError("client_uid must be a non-negative integer.")
    if ssh_read_only_client_id_by_uid is None:
        read_only_pairs: tuple[tuple[str, int], ...] = ()
    elif isinstance(ssh_read_only_client_id_by_uid, Mapping):
        read_only_pairs = tuple(
            (client_id, read_only_uid)
            for read_only_uid, client_id in ssh_read_only_client_id_by_uid.items()
        )
    else:
        raise TypeError("ssh_read_only_client_id_by_uid must be a mapping or None.")
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id=client,
        primary_client_uid=uid,
        ssh_read_only_clients=read_only_pairs,
    )
    if client_bindings.read_only_client_ids:
        validate_agent_gateway_unix_group_parent(socket_path=socket_path)
    endpoint_value, peer_timeout = validate_agent_gateway_unix_server_preflight(
        socket_path=socket_path,
        peer_idle_timeout_sec=peer_idle_timeout_sec,
    )
    endpoint = Path(endpoint_value)
    if not callable(clock) or not callable(sleep):
        raise TypeError("clock and sleep must be callable.")
    perception_values = (
        camera_config_path,
        camera_registry_path,
        artifact_root,
    )
    perception_enabled = all(value is not None for value in perception_values)
    if perception_enabled != any(value is not None for value in perception_values):
        raise ValueError(
            "camera_config_path, camera_registry_path, and artifact_root must be "
            "provided together."
        )
    grasp_enabled = point_grasp_planner is not None
    if (object_localizer is None) != (point_grasp_planner is None):
        raise ValueError(
            "object_localizer and point_grasp_planner must be supplied together."
        )
    if grasp_enabled and not perception_enabled:
        raise ValueError("Point-grasp requires the composed camera resources.")
    if grasp_enabled != (camera_mount_urdf_path is not None):
        raise ValueError(
            "Point-grasp three-view rendering requires one explicit camera mount URDF."
        )
    if debug_render_object_coordinates and not grasp_enabled:
        raise ValueError(
            "Debug object-coordinate rendering requires point-grasp resources."
        )
    if enable_point_grasp_arm_reachability and not grasp_enabled:
        raise ValueError("Point-grasp arm reachability requires point-grasp resources.")
    if (
        grasp_enabled
        and getattr(point_grasp_planner, "point_segmenter", None)
        is not object_localizer
    ):
        raise ValueError("Point-grasp must share the exact object-localizer instance.")

    envelope_profile = lower.target_safety.envelope_profile
    side_envelope = envelope_profile.side_config_by_side.get(side)
    if side_envelope is None or not side_envelope.enabled:
        raise ValueError(
            f"selected lower envelope does not actively constrain {side!r}."
        )

    profile = lower.profile
    if profile.station_world_frame_id != RM65_SKILL_EXECUTION_FRAME_ID_V1:
        raise ValueError(
            "lower Station world frame does not match the real Skill execution frame"
        )
    if RM65_SKILL_EXECUTION_FRAME_ID_V1 != REAL_PERCEPTION_SOURCE_FRAME_ID_V1:
        raise ValueError("RM65 execution and Real Depth source frames disagree.")
    if RM65_SKILL_STATION_FRAME_ID_V1 != REAL_PERCEPTION_STATION_FRAME_ID_V1:
        raise ValueError("RM65 and Real Depth Station frame identities disagree.")
    tool_config_hash = _hash(
        {
            "profiles": _public_tool_profiles(lower),
            "tcp_pose_flange_m_rad_by_side": {
                name: tuple(
                    float(value)
                    for value in profile.tool_tcp_pose_flange_m_rad_by_side[name]
                )
                for name in ("left", "right")
            },
        },
        domain="dexteleop.real_skill.rm65_tool_inventory.v1",
    )
    backend_inventory_hash = _hash(
        {
            "backend_kind": "dual_rm65_position",
            "arm_ip_assignment": dict(profile.arm_ip_by_side),
            "position_profile": profile.position_profile.to_metadata(),
            "tool_config_hash": tool_config_hash,
        },
        domain="dexteleop.real_skill.rm65_backend_inventory.v1",
    )
    scene_mount_hash = _hash(
        {
            name: profile.scene_world_from_arm_base_by_side[name].tolist()
            for name in ("left", "right")
        },
        domain="dexteleop.real_skill.rm65_scene_mounts.v1",
    )
    station_geometry_hash = _hash(
        {
            "source": "collision_scene_urdf_fixed_joint_chain",
            "world_frame_id": profile.station_world_frame_id,
            "collision_scene_profile": profile.collision_scene_profile,
            "scene_mount_hash": scene_mount_hash,
            "scene_world_from_canonical_station": (
                profile.scene_world_from_canonical_station.tolist()
            ),
        },
        domain="dexteleop.real_skill.rm65_station_geometry.v1",
    )
    station_frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile=profile.collision_scene_profile,
        scene_mount_hash=scene_mount_hash,
        scene_world_from_canonical_station=(profile.scene_world_from_canonical_station),
    )
    frame_graph = station_frame_binding.frame_graph
    execution_frame = frame_graph.frame_by_id[RM65_SKILL_EXECUTION_FRAME_ID_V1]
    station_frame = frame_graph.frame_by_id[RM65_SKILL_STATION_FRAME_ID_V1]
    station_manifest_hash = _hash(
        {
            "frame_graph_hash": frame_graph.semantic_content_hash,
            "scene_mount_hash": scene_mount_hash,
            "station_geometry_hash": station_geometry_hash,
            "deployment_profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.real_skill.rm65_station_manifest.v1",
    )
    execution_manifest_hash = _hash(
        {
            "lower_owner": "teleop_runtime.skill_deploy",
            "profile_hash": profile.semantic_content_hash,
            "backend_inventory_hash": backend_inventory_hash,
            "frame_graph_hash": frame_graph.semantic_content_hash,
            "target_safety_profile_hash": envelope_profile.semantic_content_hash,
            "active_side": side,
            "inactive_side_semantics": "coupled_solver_hard_hold",
            "command_owner_count": 1,
        },
        domain="dexteleop.real_skill.rm65_execution_manifest.v1",
    )
    target_binding_hash = _hash(
        {
            "embodiment_instance_id": embodiment_id,
            "profile_hash": profile.semantic_content_hash,
            "active_side": side,
        },
        domain="dexteleop.real_skill.rm65_target_binding.v1",
    )
    target_binding_id = "real-rm65-exclusive:" + target_binding_hash
    source_binding_id = "rm65-hardware-qpos:" + backend_inventory_hash
    source_binding_hash = _hash(
        {
            "source_binding_id": source_binding_id,
            "backend_inventory_hash": backend_inventory_hash,
            "read_semantics": (
                "independently_sampled_cached_authoritative_dual_hardware_qpos"
                if lower.ports.profile.background_measurement_sampling
                else "synchronous_authoritative_dual_hardware_qpos"
            ),
            "state_refresh_hz": lower.ports.profile.state_refresh_hz,
            "max_measurement_age_sec": (
                lower.ports.profile.max_measurement_age_sec
            ),
        },
        domain="dexteleop.real_skill.rm65_source_binding.v1",
    )
    observation_plan_hash = _hash(
        {
            "source_binding_hash": source_binding_hash,
            "projection": "coupled_fk_to_tool_tcp_world",
            "component_ids": (f"{side}_arm",),
            "pull_semantics": (
                "continuous_lower_sampling_plus_skill_cycle_consumption"
                if lower.ports.profile.background_measurement_sampling
                else "gateway_request_or_skill_cycle"
            ),
        },
        domain="dexteleop.real_skill.rm65_observation_plan.v1",
    )
    observation_capability_hash = _hash(
        {
            "observation_kind": "tcp_pose",
            "representation": "xyz_quat_xyzw",
            "controlled_frame": "tool_tcp",
            "reference_frame": execution_frame.frame_id,
            "bounded_components": (f"{side}_arm",),
        },
        domain="dexteleop.real_skill.rm65_observation_capability.v1",
    )
    observation_binding_id = "rm65-tool-tcp-fk:" + observation_plan_hash
    observation_binding_hash = _hash(
        {
            "observation_binding_id": observation_binding_id,
            "observation_plan_hash": observation_plan_hash,
            "observation_capability_hash": observation_capability_hash,
            "tool_config_hash": tool_config_hash,
            "station_geometry_hash": station_geometry_hash,
        },
        domain="dexteleop.real_skill.rm65_observation_binding.v1",
    )
    pose_schema_hash = _hash(
        {
            "representation": "cartesian_pose",
            "layout": "xyz_quat_xyzw",
            "quaternion_order": "xyzw",
            "reference_frame_id": execution_frame.frame_id,
            "controlled_frame_id": "tool_tcp",
        },
        domain="dexteleop.real_skill.rm65_tcp_pose_schema.v1",
    )
    maximum_age_ms = min(
        200,
        max(1, round(lower.ports.profile.max_measurement_age_sec * 1000.0)),
    )
    attachment = RealTcpPoseObserveAttachmentSpec(
        runtime_session_id=runtime_id,
        control_domain_id=domain_id,
        embodiment_instance_id=embodiment_id,
        target_binding_id=target_binding_id,
        component_ids=(f"{side}_arm",),
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        backend_inventory_hash=backend_inventory_hash,
        source_binding_id=source_binding_id,
        source_binding_hash=source_binding_hash,
        observation_binding_id=observation_binding_id,
        observation_binding_hash=observation_binding_hash,
        observation_plan_hash=observation_plan_hash,
        observation_capability_hash=observation_capability_hash,
        pose_schema_hash=pose_schema_hash,
        reference_frame_id=execution_frame.frame_id,
        reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash=tool_config_hash,
        # ``calibration_hash`` is a v2 wire compatibility field. Independent
        # Skill deployment binds it to lower-owned Station geometry; it never
        # hashes or loads a TeleopCalibration.
        calibration_hash=station_geometry_hash,
        clock_domain_id=RM65_SKILL_CLOCK_DOMAIN_ID_V1,
        producer_epoch=1,
        max_measurement_age_ms=maximum_age_ms,
    )
    port_config = RealAgentMotionPortConfig(
        active_sides=(side,),
        component_id_by_side={side: f"{side}_arm"},
        execution_frame_id=execution_frame.frame_id,
        execution_frame_spec_hash=execution_frame.semantic_content_hash,
        exclusive_command_claim=True,
    )
    constraint_set_hash = _hash(
        {
            "profile_id": envelope_profile.profile_id,
            "profile_hash": envelope_profile.semantic_content_hash,
            "active_side": side,
            "side_constraints": side_envelope.semantic_payload(),
            "application": "active_final_pre_ik",
        },
        domain="dexteleop.real_skill.rm65_safety_constraint_set.v1",
    )
    envelope_ref = SkillDeploymentSafetyEnvelopeRef(
        object_id=f"real-rm65-envelope:{envelope_profile.profile_id}:{side}",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash=envelope_profile.semantic_content_hash,
        constraint_set_hash=constraint_set_hash,
        allowed_route_ids=_route_ids(),
    )
    perception_source_binding = None
    perception_frame_binding = None
    perception_capture = None
    rgb_camera_geometry = None
    localization_registration = None
    localization_provider = None
    station_point_grasp_registration = None
    station_point_grasp_render_registration = None
    artifact_store = None
    provider_factory = None
    arm_reachability_profile = None
    reachability_config = (
        RM65GraspReachabilityConfig() if enable_point_grasp_arm_reachability else None
    )
    resources: object = lower.ports
    if perception_enabled:
        assert camera_config_path is not None
        assert camera_registry_path is not None
        assert artifact_root is not None
        root = Path(artifact_root)
        if not root.is_absolute():
            raise ValueError("artifact_root must be an absolute path.")
        perception_source_binding = build_real_perception_source_binding(
            camera_config_path=camera_config_path,
            camera_registry_path=camera_registry_path,
            scene_world_from_arm_base_by_side=(
                lower.profile.scene_world_from_arm_base_by_side
            ),
        )
        perception_frame_binding = station_frame_binding
        if grasp_enabled:
            assert camera_mount_urdf_path is not None
            rgb_camera_geometry = build_rm65_rgb_camera_geometry(
                source_binding=perception_source_binding,
                target_builder=lower.target_builder,
                camera_registry_path=camera_registry_path,
                camera_mount_urdf_path=camera_mount_urdf_path,
                scene_world_from_arm_base_by_side=(
                    lower.profile.scene_world_from_arm_base_by_side
                ),
                scene_world_from_canonical_station=(
                    lower.profile.scene_world_from_canonical_station
                ),
            )
        catalog_provider = RealSenseAgentPerceptionProvider(
            source_binding=perception_source_binding,
            current_station_revision=lambda: 0,
            rgb_camera_geometry=rgb_camera_geometry,
        )
        perception_capture = build_real_perception_capture_registration_v2(
            rgb_source_catalog=catalog_provider.rgb_source_catalog(),
            depth_source_catalog=catalog_provider.depth_source_catalog(),
            runtime_session_id=runtime_id,
            control_domain_id=domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
            source_binding_hash=perception_source_binding.semantic_content_hash,
            frame_binding=perception_frame_binding,
        )
        if grasp_enabled:
            assert object_localizer is not None
            assert point_grasp_planner is not None
            assert rgb_camera_geometry is not None
            localization_spec = getattr(object_localizer, "model_spec", None)
            localization_model_id = str(
                getattr(localization_spec, "model_profile_id", "")
            ).strip()
            localization_model_hash = str(
                getattr(localization_spec, "semantic_content_hash", "")
            ).strip()
            localizer_hash = str(
                getattr(object_localizer, "semantic_content_hash", "")
            ).strip()
            planner_spec = getattr(point_grasp_planner, "model_spec", None)
            planner_model_id = str(
                getattr(planner_spec, "model_profile_id", "")
            ).strip()
            planner_model_hash = str(
                getattr(planner_spec, "semantic_content_hash", "")
            ).strip()
            planner_hash = str(
                getattr(point_grasp_planner, "semantic_content_hash", "")
            ).strip()
            backend_id = str(getattr(point_grasp_planner, "backend_id", "")).strip()
            if not all(
                (
                    localization_model_id,
                    localization_model_hash,
                    localizer_hash,
                    planner_model_id,
                    planner_model_hash,
                    planner_hash,
                    backend_id,
                )
            ):
                raise ValueError("Point-grasp models lack verified profile identity.")
            localization_registration = build_real_object_localization_registration(
                model_profile_id=localization_model_id,
                model_profile_hash=localization_model_hash,
                inference_implementation_hash=localizer_hash,
                local_path_enabled=False,
                runtime_session_id=runtime_id,
                control_domain_id=domain_id,
                target_binding_id=target_binding_id,
                station_manifest_hash=station_manifest_hash,
                execution_manifest_hash=execution_manifest_hash,
                current_mode="real_active",
            )
            localization_provider = ObjectLocalizationProvider(
                localizer=object_localizer,
                registration=localization_registration,
                allowed_local_image_roots=(),
            )
            station_point_grasp_registration = (
                build_station_bound_point_grasp_registration(
                    aligned_rgbd_source_catalog=tuple(
                        dict(item)
                        for item in catalog_provider.aligned_rgbd_source_catalog()
                    ),
                    source_binding_hash=(
                        perception_source_binding.semantic_content_hash
                    ),
                    frame_binding_hash=perception_frame_binding.semantic_content_hash,
                    frame_graph_hash=frame_graph.semantic_content_hash,
                    source_frame_id=perception_frame_binding.source_frame_id,
                    source_frame_spec_hash=(
                        perception_frame_binding.source_frame_spec_hash
                    ),
                    reference_frame_id=perception_frame_binding.reference_frame_id,
                    reference_frame_spec_hash=(
                        perception_frame_binding.reference_frame_spec_hash
                    ),
                    backend_id=backend_id,
                    model_profile_id=planner_model_id,
                    model_profile_hash=planner_model_hash,
                    inference_implementation_hash=planner_hash,
                    runtime_session_id=runtime_id,
                    control_domain_id=domain_id,
                    target_binding_id=target_binding_id,
                    station_manifest_hash=station_manifest_hash,
                    execution_manifest_hash=execution_manifest_hash,
                    current_mode="real_active",
                    arm_reachability_profile=(
                        StationPointGraspArmReachabilityProfile(
                            arm_side=side,
                            evaluator_profile_id=(
                                "rm65.lower_endpoint_ik_reachability.v1"
                            ),
                            evaluator_profile_hash=(
                                reachability_config.semantic_content_hash
                            ),
                            solver_reference_frame_id=execution_frame.frame_id,
                            solver_reference_frame_spec_hash=(
                                execution_frame.semantic_content_hash
                            ),
                            frame_graph_hash=frame_graph.semantic_content_hash,
                            time_budget_ms=reachability_config.time_budget_ms,
                            max_solve_cycles=reachability_config.max_solve_cycles,
                            max_measured_qpos_age_ms=float(maximum_age_ms),
                            standard_position_tolerance_m=(
                                reachability_config.standard_position_tolerance_m
                            ),
                            standard_orientation_tolerance_deg=(
                                reachability_config.standard_orientation_tolerance_deg
                            ),
                            basic_position_tolerance_m=(
                                reachability_config.basic_position_tolerance_m
                            ),
                            basic_orientation_tolerance_deg=(
                                reachability_config.basic_orientation_tolerance_deg
                            ),
                        )
                        if reachability_config is not None
                        else None
                    ),
                )
            )
            arm_reachability_profile = (
                station_point_grasp_registration.profile.arm_reachability_profile
            )
            station_point_grasp_render_registration = (
                build_station_bound_point_grasp_render_registration(
                    point_grasp_registration=station_point_grasp_registration,
                    render_rgb_source_catalog=tuple(
                        dict(item)
                        for item in catalog_provider.rgb_geometry_source_catalog()
                    ),
                    camera_geometry_binding_hash=(
                        rgb_camera_geometry.semantic_content_hash
                    ),
                    debug_target_coordinate_overlay=(debug_render_object_coordinates),
                )
            )
        partition = _hash(
            {"runtime_session_id": runtime_id},
            domain="dexteleop.agent_gateway.artifact_store_partition.v1",
        )
        artifact_store = RecoverableAgentArtifactStore(
            root=root / partition,
            quota_bytes=512 * 1024 * 1024,
            max_artifact_bytes=REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
            max_chunk_bytes=1024 * 1024,
        )

        def provider_factory(port, current_station_revision):
            assert perception_source_binding is not None
            return RealSenseAgentPerceptionProvider(
                source_binding=perception_source_binding,
                current_station_revision=current_station_revision,
                max_frame_age_ms=perception_capture.profile.max_frame_age_ms,
                max_inter_camera_skew_ms=(
                    perception_capture.profile.max_inter_camera_skew_ms
                ),
                rgb_camera_geometry=rgb_camera_geometry,
                clock=clock,
            )

    def measured_qpos_observer(cut, *, derived_available_at):
        if rgb_camera_geometry is None:
            return
        rgb_camera_geometry.publish_measured_qpos(
            qpos_by_side=cut.qpos_by_side,
            measured_at=cut.measured_at,
            # FK is derived only after this control cycle has consumed and
            # synchronized the lower qpos cut.  It cannot inherit the earlier
            # source availability timestamp.
            available_at=derived_available_at,
            evidence_refs=cut.evidence_refs,
        )

    try:
        grasp_trajectory_planner = RM65GraspJointTrajectoryPlanner(
            profile=lower.profile,
            planning_target_builder=build_exclusive_rm65_skill_target_builder(
                lower.profile
            ),
            target_envelope_filter=(
                lower.target_builder.pre_ik_target_safety_filter
            ),
            reachability_config=reachability_config,
        )

        def reachability_evaluator_factory(port):
            if arm_reachability_profile is None:
                raise RuntimeError("Arm reachability profile is unavailable.")
            return RM65PointGraspReachabilityAdapter(
                profile=arm_reachability_profile,
                planner=grasp_trajectory_planner,
                motion_port=port,
                frame_graph=frame_graph,
                candidate_reference_frame_id=station_frame.frame_id,
                clock=clock,
            )

        deployment = assemble_exclusive_rm65_agent_primitive_deployment(
            target_builder=lower.target_builder,
            rm65_ports=lower.ports,
            port_config=port_config,
            observation_attachment=attachment,
            frame_graph=frame_graph,
            primitive_reference_frame_id=station_frame.frame_id,
            safety_envelope_refs=(envelope_ref,),
            allowed_client_ids=client_bindings.allowed_client_ids,
            perception_capture=perception_capture,
            localization_registration=localization_registration,
            localization_provider=localization_provider,
            station_point_grasp_registration=station_point_grasp_registration,
            station_point_grasp_render_registration=(
                station_point_grasp_render_registration
            ),
            station_point_grasp_planner=point_grasp_planner,
            station_point_grasp_reachability_evaluator_factory=(
                reachability_evaluator_factory
                if arm_reachability_profile is not None
                else None
            ),
            grasp_trajectory_planner=grasp_trajectory_planner,
            perception_provider_factory=provider_factory,
            measured_qpos_observer=(
                measured_qpos_observer if rgb_camera_geometry is not None else None
            ),
            artifact_store=artifact_store,
            # The aggregate application resource closes the shared store after
            # Gateway executors/models and cameras have settled.
            owns_artifact_store=False,
            max_retained_operations=max_retained_operations,
            clock=clock,
            sleep=sleep,
        )
    except Exception:
        if artifact_store is not None:
            artifact_store.close()
        raise
    if perception_source_binding is not None:
        additional_resources = tuple(
            resource
            for resource in (
                localization_provider,
                deployment.gateway_session.station_point_grasp_provider,
            )
            if resource is not None
        )
        resources = ExclusiveRealPerceptionResources(
            lower=lower.ports,
            camera_sources=perception_source_binding.camera_sources,
            artifact_store=artifact_store,
            additional_resources=additional_resources,
        )
    authorization = RealAgentPrimitiveOneShotAuthorization(
        gateway_session=deployment.gateway_session,
        side=side,
        authorized_actions=normalized_actions,
        client_id=client,
        confirmation_token=confirmation_token,
        grant_lifetime_sec=grant_lifetime_sec,
        clock=clock,
    )

    def server_factory():
        return deployment.gateway_session.build_unix_json_server(
            socket_path=str(endpoint),
            client_id_by_uid=client_bindings.client_id_by_uid,
            read_only_client_ids=client_bindings.read_only_client_ids,
            socket_mode=client_bindings.socket_mode,
            peer_idle_timeout_sec=peer_timeout,
            grant_resolver=authorization.grant_resolver,
        )

    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lower.startup_admission,
        server_factory=server_factory,
        config=ExclusiveRealSkillApplicationConfig(
            control_hz=profile.position_profile.rates.reference_hz,
        ),
    )
    return ExclusiveRM65AgentPrimitiveComposition(
        lower=lower,
        frame_graph=frame_graph,
        attachment=attachment,
        port_config=port_config,
        safety_envelope_ref=envelope_ref,
        deployment=deployment,
        authorization=authorization,
        application=application,
        resources=resources,
        perception_source_binding=perception_source_binding,
        perception_frame_binding=perception_frame_binding,
        perception_capture=perception_capture,
        rgb_camera_geometry=rgb_camera_geometry,
        localization_registration=localization_registration,
        localization_provider=localization_provider,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        point_grasp_planner=point_grasp_planner,
        socket_path=endpoint,
        authorized_side=side,
        authorized_action=authorization.action,
        authorized_actions=authorization.authorized_actions,
        client_id=client,
        client_uid=uid,
        read_only_client_ids=tuple(sorted(client_bindings.read_only_client_ids)),
        socket_mode=client_bindings.socket_mode,
    )


__all__ = [
    "RM65_SKILL_CLOCK_DOMAIN_ID_V1",
    "RM65_SKILL_EXECUTION_FRAME_ID_V1",
    "RM65_SKILL_STATION_FRAME_ID_V1",
    "ExclusiveRM65AgentPrimitiveComposition",
    "assemble_exclusive_rm65_agent_primitive_composition",
]
