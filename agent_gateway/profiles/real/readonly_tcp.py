"""Bind a lower read-only RM65 sampler to the existing TCP observation Skill."""

from __future__ import annotations

import threading
import time

from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.skill_deploy.rm65_readonly import (
    RM65_READONLY_POLL_INTERVAL_SEC,
    RM65ReadonlyTcpSampler,
)

from .observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
    RealTcpPoseOwnerPublication,
    RealTcpPoseOwnerPublicationBuffer,
    build_real_tcp_pose_observe_contract_bundle,
)
from .perception_frames import build_real_perception_station_frame_binding


def build_readonly_rm65_tcp_contract(profile, *, runtime_session_id: str):
    """Resolve frame/tool identities without constructing any device or owner."""
    def identity(payload, name):
        return canonical_content_hash(payload, domain=f"agenticrobot.rm65.readonly_tcp.{name}.v1")

    mounts_hash = canonical_content_hash(
        {side: profile.scene_world_from_arm_base_by_side[side].tolist() for side in ("left", "right")},
        domain="dexteleop.real_skill.rm65_scene_mounts.v1",
    )
    frames = build_real_perception_station_frame_binding(
        collision_scene_profile=profile.collision_scene_profile,
        scene_mount_hash=mounts_hash,
        scene_world_from_canonical_station=profile.scene_world_from_canonical_station,
    )
    geometry_hash = canonical_content_hash({
        "source": "collision_scene_urdf_fixed_joint_chain",
        "world_frame_id": profile.station_world_frame_id,
        "collision_scene_profile": profile.collision_scene_profile,
        "scene_mount_hash": mounts_hash,
        "scene_world_from_canonical_station": profile.scene_world_from_canonical_station.tolist(),
    }, domain="dexteleop.real_skill.rm65_station_geometry.v1")
    tool_hash = canonical_content_hash({
        "profiles": {side: {k: v for k, v in profile.tool_profile_by_side[side].items()
                            if not str(k).startswith("_")} for side in ("left", "right")},
        "tcp_pose_flange_m_rad_by_side": {
            side: tuple(float(x) for x in profile.tool_tcp_pose_flange_m_rad_by_side[side])
            for side in ("left", "right")},
    }, domain="dexteleop.real_skill.rm65_tool_inventory.v1")
    inventory = identity({"kind": "raw_sdk_readonly", "ips": dict(profile.arm_ip_by_side)}, "inventory")
    source = identity({"inventory": inventory, "request": "Get_Current_Arm_State(retry=0)",
                       "timestamp": "earliest_request_start",
                       "poll_interval_sec": RM65_READONLY_POLL_INTERVAL_SEC}, "source")
    observation = identity({"source": source, "geometry": geometry_hash, "tools": tool_hash,
                            "projection": "calibrated_fk_canonical_station"}, "observation")
    manifest = identity({"command_owner_count": 0, "control_effects": "forbidden",
                         "inventory": inventory, "observation": observation}, "manifest")
    attachment = RealTcpPoseObserveAttachmentSpec(
        runtime_session_id=runtime_session_id,
        control_domain_id="rm65-readonly-observation",
        embodiment_instance_id="rm65-dual:" + inventory,
        target_binding_id="rm65-readonly:" + observation,
        component_ids=("left_arm", "right_arm"),
        station_manifest_hash=identity({"frame_graph": frames.frame_graph.semantic_content_hash,
                                       "geometry": geometry_hash}, "station"),
        execution_manifest_hash=manifest,
        backend_inventory_hash=inventory,
        source_binding_id="rm65-readonly-qpos:" + source,
        source_binding_hash=source,
        observation_binding_id="rm65-readonly-fk:" + observation,
        observation_binding_hash=observation,
        observation_plan_hash=observation,
        observation_capability_hash=identity({"components": ["left_arm", "right_arm"],
                                             "representation": "xyz_quat_xyzw"}, "capability"),
        pose_schema_hash=identity({"reference_frame": frames.reference_frame_id,
                                   "controlled_frame": "tool_tcp", "layout": "xyz_quat_xyzw"}, "pose_schema"),
        reference_frame_id=frames.reference_frame_id,
        reference_frame_spec_hash=frames.reference_frame_spec_hash,
        frame_graph_hash=frames.frame_graph.semantic_content_hash,
        tool_config_hash=tool_hash,
        calibration_hash=geometry_hash,
        clock_domain_id="rm65.readonly.monotonic",
        producer_epoch=1,
    )
    return build_real_tcp_pose_observe_contract_bundle(attachment)


class ReadonlyRM65TcpSource:
    """Project actual sampler cuts, not synthetic command-owner state.

    No device call occurs here. Epochs/generations are zero because no control
    authority or dispatch owner exists; the lifecycle describes sampling only.
    """

    def __init__(self, sampler: RM65ReadonlyTcpSampler, bundle, *, clock=time.monotonic):
        self.sampler = sampler
        self.bundle = bundle
        self.buffer = RealTcpPoseOwnerPublicationBuffer(contract_bundle=bundle, clock=clock)
        self._lock = threading.Lock()
        self._owner_hash = canonical_content_hash({
            "runtime_session_id": bundle.attachment.runtime_session_id,
            "source_binding_hash": bundle.attachment.source_binding_hash,
            "owner_kind": "readonly_sdk_sampler", "command_owner_count": 0,
        }, domain="agenticrobot.rm65.readonly_tcp.sampling_owner.v1")

    def capture_published_tcp_pose(self, *, active_operation_id, parameters):
        with self._lock:
            sample = self.sampler.latest
            previous = self.buffer.latest_publication
            if previous is None or sample.sequence > previous.station_revision:
                sample_hash = sample.semantic_content_hash
                self.buffer.publish_from_owner(RealTcpPoseOwnerPublication(
                    publication_id="rm65-readonly-sample:" + sample_hash,
                    attachment_hash=self.bundle.attachment.semantic_content_hash,
                    clock_domain_id=self.bundle.attachment.clock_domain_id,
                    station_revision=sample.sequence,
                    owner_token_id="readonly-sampling:" + self._owner_hash,
                    owner_kind="readonly_sdk_sampler",
                    source_owner_state_hash=self._owner_hash,
                    authority_epoch=0, execution_epoch=0, dispatch_owner_generation=0,
                    source_owner_state_revision=0,
                    lifecycle_state="readonly_sampling",
                    execution_gate="unavailable_no_command_owner",
                    measurement_id="rm65-readonly-measurement:" + sample_hash,
                    source_measurement_hash=sample_hash,
                    measured_at=sample.measured_at, available_at=sample.available_at,
                    pose_xyz_quat_xyzw_by_component=sample.pose_by_component,
                    evidence_refs=("rm65-readonly-sdk-fk:" + sample_hash,),
                ))
            return self.buffer.capture_published_tcp_pose(
                active_operation_id=active_operation_id, parameters=parameters
            )
