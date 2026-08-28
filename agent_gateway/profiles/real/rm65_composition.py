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

import numpy as np

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
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from teleop_runtime.skill_deploy import (
    RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
    ExclusiveRM65SkillLowerAssembly,
    RealAgentMotionPortConfig,
)

from .exclusive_application import (
    ExclusiveRealSkillApplication,
    ExclusiveRealSkillApplicationConfig,
)
from .exclusive_factory import (
    ExclusiveRealAgentPrimitiveDeployment,
    assemble_exclusive_rm65_agent_primitive_deployment,
)
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
    build_real_perception_capture_registration,
)
from .realsense_perception import (
    ExclusiveRealPerceptionResources,
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
    build_real_perception_source_binding,
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
    return real_agent_motion_route_ids_v1()


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
    perception_capture: RealPerceptionCaptureRegistration | None
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
        elif not isinstance(self.resources, ExclusiveRealPerceptionResources):
            raise ValueError("perception composition requires aggregate resources.")
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
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ExclusiveRM65AgentPrimitiveComposition:
    """Bind one lower RM65 graph to the existing real Skill/Gateway contracts."""

    if not isinstance(lower, ExclusiveRM65SkillLowerAssembly):
        raise TypeError("lower must be an ExclusiveRM65SkillLowerAssembly.")
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
    normalized_actions = tuple(sorted({str(item).strip() for item in requested_actions}))
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
    tool_config_hash = _hash(
        {
            "profiles": _public_tool_profiles(lower),
            "tcp_pose_flange_m_rad_by_side": {
                name: tuple(
                    float(value)
                    for value in profile.tool_tcp_pose_flange_m_rad_by_side[
                        name
                    ]
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
    canonical_station_from_execution = np.linalg.inv(
        profile.scene_world_from_canonical_station
    )

    execution_frame = CoordinateFrameSpec(
        frame_id=RM65_SKILL_EXECUTION_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "owner": "rm65_collision_scene",
            "collision_scene_profile": profile.collision_scene_profile,
            "scene_mount_hash": scene_mount_hash,
        },
    )
    station_frame = CoordinateFrameSpec(
        frame_id=RM65_SKILL_STATION_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    frame_graph = CoordinateFrameGraph(
        graph_id="exclusive-rm65-skill-station-v2",
        graph_version="2",
        frames=(execution_frame, station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="rm65-solver-world-to-canonical-station-v2",
                transform_version="2",
                source_frame_id=execution_frame.frame_id,
                target_frame_id=station_frame.frame_id,
                matrix_target_source=canonical_station_from_execution,
                semantic_parameters={
                    "basis": "inverse_collision_scene_urdf_fixed_joint_chain",
                    "collision_scene_profile": profile.collision_scene_profile,
                },
            ),
        ),
    )
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
            "read_semantics": "cached_authoritative_dual_hardware_qpos",
            "state_refresh_hz": lower.ports.profile.state_refresh_hz,
        },
        domain="dexteleop.real_skill.rm65_source_binding.v1",
    )
    observation_plan_hash = _hash(
        {
            "source_binding_hash": source_binding_hash,
            "projection": "coupled_fk_to_tool_tcp_world",
            "component_ids": (f"{side}_arm",),
            "pull_semantics": "gateway_request_or_skill_cycle",
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
        max(1, int(round(lower.ports.profile.max_measurement_age_sec * 1000.0))),
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
    perception_capture = None
    artifact_store = None
    provider_factory = None
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
        catalog_provider = RealSenseAgentPerceptionProvider(
            source_binding=perception_source_binding,
            current_station_revision=lambda: 0,
        )
        perception_capture = build_real_perception_capture_registration(
            rgb_source_catalog=catalog_provider.rgb_source_catalog(),
            depth_source_catalog=catalog_provider.depth_source_catalog(),
            runtime_session_id=runtime_id,
            control_domain_id=domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
            source_binding_hash=perception_source_binding.semantic_content_hash,
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
                clock=clock,
            )

        resources = ExclusiveRealPerceptionResources(
            lower=lower.ports,
            camera_sources=perception_source_binding.camera_sources,
            artifact_store=artifact_store,
        )
    try:
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
            perception_provider_factory=provider_factory,
            artifact_store=artifact_store,
            owns_artifact_store=artifact_store is not None,
            max_retained_operations=max_retained_operations,
            clock=clock,
            sleep=sleep,
        )
    except Exception:
        if artifact_store is not None:
            artifact_store.close()
        raise
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
        perception_capture=perception_capture,
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
    "ExclusiveRM65AgentPrimitiveComposition",
    "RM65_SKILL_CLOCK_DOMAIN_ID_V1",
    "RM65_SKILL_EXECUTION_FRAME_ID_V1",
    "RM65_SKILL_STATION_FRAME_ID_V1",
    "assemble_exclusive_rm65_agent_primitive_composition",
]
