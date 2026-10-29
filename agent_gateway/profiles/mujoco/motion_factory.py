"""Exact application graph factory for the no-model Agent TCP-pose Route.

Calling this factory explicitly prepares MuJoCo resources, but it does not
start the runtime, advance physics, bind a transport or submit motion.  The
caller must provide the dedicated target config; ordinary CubeLift Sim-HIL
configuration is rejected instead of being silently repurposed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.protocol.skill_io_contracts import (
    inline_only_skill_io_catalog_for_routes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorRegistry
from agent_gateway.skills.gripper.deployment import (
    GripperSkillDeploymentBinding,
    build_gripper_skill_contract_bundle,
)
from collection_runtime.source_observation_history import (
    SourceObservationHistoryBinding,
    SQLiteSourceObservationHistoryStore,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ExecutionObservationPlan
from robot_motion.rm65 import load_rm65_tool_profile
from robot_motion.target_safety import (
    TCPPoseEnvelopeCatalogEntry,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSideConfig,
)
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoExecutionDifferenceObserver,
    MujocoSimHILTargetConfig,
    build_mujoco_agent_skill_motion_target,
    build_mujoco_component_execution_binding,
    build_mujoco_pose_component_target_profile,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)
from sim.mujoco.deploy.grasp_contact import (
    MujocoGraspContactBinding,
    bind_task_grasp_contact,
)
from sim.mujoco.deploy.target_config import DEFAULT_STATION_POSE_CONTRACT_PATH
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

from .gripper_skill import MujocoGripperControlPort
from .motion_application import MujocoAgentSkillMotionApplication
from .motion_deployment import (
    MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V2,
    MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V3,
    MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V4,
    MUJOCO_OBSERVED_POINT_ROUTE_ID,
    MUJOCO_AGENT_GRASP_APPROACH_ROUTE_ID_V1,
    MUJOCO_AGENT_GRASP_CONTACT_ROUTE_ID_V1,
    MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V1,
    MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V2,
    MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1,
    MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V2,
    MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1,
    MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1,
    MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1,
    MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1,
    MujocoAgentTcpPoseContractBundle,
    MujocoAgentTcpPoseHistoryProfile,
    build_mujoco_agent_tcp_pose_contract_bundle,
    resolve_safe_tcp_motion_limits,
)
from .motion_gateway_session import (
    MujocoAgentTcpPoseGatewaySession,
    assemble_mujoco_agent_tcp_pose_gateway_session,
)
from .motion_observed_scene import MujocoObservedSafeMotionConfig

MUJOCO_AGENT_TCP_POSE_CALIBRATION_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.mujoco_tcp_pose_scene_calibration.v1"
)
MUJOCO_AGENT_TCP_POSE_COMPONENT_PROFILE_ID_V1 = (
    "mujoco.rm65.right_agent_tcp_pose.v1"
)
MUJOCO_AGENT_TCP_POSE_OBSERVATION_PLAN_ID_V1 = (
    "control-station-agent-tcp-pose-right-arm-observation-v1"
)
MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1 = (
    "mujoco.agent_tcp_pose.safety.standard.v1"
)
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ENVELOPE_ID_V1 = (
    "mujoco.agent_tcp_pose.safety.fixed_orientation.v1"
)


@dataclass(frozen=True)
class MujocoAgentArtifactStorageConfig:
    """Explicit persistent CAS root and finite session-local limits."""

    root: Path
    quota_bytes: int = 512 * 1024 * 1024
    max_artifact_bytes: int = 16 * 1024 * 1024
    max_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("Artifact root must be pathlib.Path.")
        if not self.root.is_absolute():
            raise ValueError("Artifact root must be an explicit absolute path.")
        for field_name in (
            "quota_bytes",
            "max_artifact_bytes",
            "max_chunk_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.max_artifact_bytes > self.quota_bytes:
            raise ValueError("max_artifact_bytes cannot exceed quota_bytes.")
        if self.max_chunk_bytes > self.max_artifact_bytes:
            raise ValueError("max_chunk_bytes cannot exceed max_artifact_bytes.")

    def store_root_for(self, runtime_session_id: str) -> Path:
        if (
            not isinstance(runtime_session_id, str)
            or not runtime_session_id
            or runtime_session_id != runtime_session_id.strip()
        ):
            raise ValueError("runtime_session_id must be a non-empty trimmed string.")
        partition = canonical_content_hash(
            {"runtime_session_id": runtime_session_id},
            domain="dexteleop.agent_gateway.artifact_store_partition.v1",
        )
        return self.root / partition


@dataclass(frozen=True)
class MujocoAgentTcpPoseHistoryStorageConfig:
    """Disk catalog for exact source-bound history stores; never auto-starts.

    A Gateway launch creates a new lower runtime/source binding.  The catalog
    root is therefore partitioned by that immutable binding instead of opening
    one mutable SQLite file across unrelated launches.
    """

    root: Path
    profile: MujocoAgentTcpPoseHistoryProfile = field(
        default_factory=MujocoAgentTcpPoseHistoryProfile
    )
    filename: str = "planning_history.sqlite3"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("history root must be pathlib.Path.")
        if not self.root.is_absolute():
            raise ValueError("history root must be an explicit absolute path.")
        if not isinstance(self.profile, MujocoAgentTcpPoseHistoryProfile):
            raise TypeError("history profile must be typed.")
        if (
            not isinstance(self.filename, str)
            or not self.filename.strip()
            or Path(self.filename).name != self.filename
        ):
            raise ValueError("history filename must be a non-empty basename.")

    def store_root_for(
        self,
        binding: SourceObservationHistoryBinding,
    ) -> Path:
        if not isinstance(binding, SourceObservationHistoryBinding):
            raise TypeError("history binding must be typed.")
        return self.root / "source_bindings" / binding.semantic_content_hash


def mujoco_agent_tcp_pose_execution_observation_plan_v1(
    *,
    motion_side: str = "right",
) -> ExecutionObservationPlan:
    side = _validated_motion_side(motion_side)
    component = f"{side}_arm"
    return ExecutionObservationPlan(
        plan_id=(
            MUJOCO_AGENT_TCP_POSE_OBSERVATION_PLAN_ID_V1
            if side == "right"
            else "control-station-agent-tcp-pose-left-arm-observation-v1"
        ),
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=(component,),
        required_representation_by_component={
            component: "cartesian_pose",
        },
    )


def validate_mujoco_agent_tcp_pose_target_config(
    config: MujocoSimHILTargetConfig,
    *,
    motion_side: str = "right",
) -> None:
    """Fail before scene construction when another Sim-HIL profile is used."""

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    side = _validated_motion_side(motion_side)
    plan = config.execution_observation_plan
    expected = mujoco_agent_tcp_pose_execution_observation_plan_v1(motion_side=side)
    if config.allow_scripted_task_reset:
        raise ValueError(
            "Agent TCP pose config requires allow_scripted_task_reset=False."
        )
    if config.exposed_pose_sides != (side,):
        raise ValueError(
            f"Agent TCP pose config must expose exactly the {side} pose side."
        )
    if (
        not isinstance(plan, ExecutionObservationPlan)
        or plan.semantic_content_hash != expected.semantic_content_hash
    ):
        raise ValueError(
            f"Agent TCP pose config requires the exact {side}-arm Cartesian "
            "execution observation plan."
        )


def _validated_motion_side(value: str) -> str:
    if type(value) is not str or value not in {"left", "right"}:
        raise ValueError("motion_side must be exactly left or right.")
    return value


def _validate_left_tool_sources(config: MujocoSimHILTargetConfig) -> None:
    """Check actual side metadata before constructing the new left graph.

    Both tool sources matter: the other arm remains a measured support target.
    Existing right-only loading semantics are deliberately not changed here.
    """
    task = CubeLiftTaskConfig.load(config.task_config_path)
    for side in ("left", "right"):
        path = getattr(task, f"{side}_tool_profile_path")
        profile = load_rm65_tool_profile(path, side=side)
        if profile.get("arm") != side:
            raise ValueError(
                f"Left motion graph requires an explicit {side} tool profile: {path}."
            )


def _calibration_hash(target: Any) -> str:
    runtime = target.runtime
    target_profile = target.target_frame_profile
    return canonical_content_hash(
        {
            "calibration_kind": "mujoco_scene_ground_truth",
            "target_frame_profile_hash": (
                target_profile.semantic_content_hash
            ),
            "frame_graph_hash": (
                target_profile.frame_graph.semantic_content_hash
            ),
            "tool_frame_profile_hash": (
                target.tool_frame_profile.semantic_content_hash
            ),
            "pose_frame_contract_hash": runtime.config.pose_frame_contract_hash,
            "observation_capability_hash": (
                runtime.execution_observation_capability.semantic_content_hash
            ),
        },
        domain=MUJOCO_AGENT_TCP_POSE_CALIBRATION_HASH_DOMAIN_V1,
    )


def _build_standard_tcp_pose_envelope_runtime(
    *,
    prepared: Any,
    target_config: MujocoSimHILTargetConfig,
    enable_safe_tcp_pose: bool = False,
    enable_safe_tcp_path_constraints: bool = False,
    safe_tcp_motion_profile: str = "local_step",
    enable_observed_point_motion: bool = False,
    enable_anchor_motion_v2: bool = False,
    enable_anchor_region_motion_v3: bool = False,
    enable_anchor_hover_motion_v4: bool = False,
    enable_grasp_approach: bool = False,
    grasp_contact_binding: MujocoGraspContactBinding | None = None,
    enable_object_motion: bool = False,
    enable_object_support_motion_v2: bool = False,
    object_support_body_names: tuple[str, ...] = (),
    motion_side: str = "right",
    enable_fast_grasp: bool = False,
) -> TCPPoseEnvelopeSafetyRuntime:
    """Bind separate lower envelopes for bounded pose and fixed heading Routes.

    Absolute-heading primitives need the full SO(3) activation range while
    preserving translation exactly.  They therefore cannot weaken the 15 deg
    envelope used by the generic pose and cardinal-step Routes.
    """

    side = _validated_motion_side(motion_side)
    component = f"{side}_arm"
    if side == "left" and enable_grasp_approach:
        raise ValueError("Left motion mode does not expose grasp approach envelopes.")
    target_profile = prepared.target_frame_profile
    tool_profile = prepared.tool_frame_profile
    reference_frame = target_profile.reference_frame_by_component[component]
    tool_binding_hash = tool_profile.binding_by_component[
        component
    ].semantic_content_hash
    target_binding_hash = canonical_content_hash(
        {
            "target_frame_profile_hash": target_profile.semantic_content_hash,
            "component_id": component,
            "side": side,
            "reference_frame_id": reference_frame,
            "controlled_frame_id": "tool_tcp",
        },
        domain="dexteleop.mujoco.agent_tcp_pose_envelope_target_binding.v1",
    )
    bounds = MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1
    delta = MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1
    profile = TCPPoseEnvelopeProfile(
        profile_id=(
            "mujoco-agent-tcp-pose-standard-envelope-v1"
            if side == "right"
            else "mujoco-agent-left-tcp-pose-standard-envelope-v1"
        ),
        target_frame_binding_hash=target_binding_hash,
        tool_frame_binding_hash_by_side={side: tool_binding_hash},
        reference_frame_by_side={side: reference_frame},
        side_config_by_side={
            side: TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(bounds[0], bounds[2], bounds[4]),
                absolute_max_xyz_m=(bounds[1], bounds[3], bounds[5]),
                relative_min_xyz_m=(-delta, -delta, -delta),
                relative_max_xyz_m=(delta, delta, delta),
                relative_axes="canonical_axes",
                max_rotation_rad=(MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1),
            )
        },
    )
    constraint_set_hash = canonical_content_hash(
        {
            "tcp_pose_envelope_profile_hash": profile.semantic_content_hash,
            "legacy_activation_delta_check_retained": True,
            "canonical_keyframe_safety_retained": True,
            "coupled_ik_safety_retained": True,
        },
        domain="dexteleop.mujoco.agent_tcp_pose_constraint_set.v1",
    )
    standard_entry = TCPPoseEnvelopeCatalogEntry(
        envelope_id=MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1,
        profile=profile,
        constraint_set_hash=constraint_set_hash,
        allowed_route_ids=(
            MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1,
            *((MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1,) if enable_safe_tcp_pose and safe_tcp_motion_profile == "local_step" else ()),
            *((MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V2,) if enable_safe_tcp_path_constraints and safe_tcp_motion_profile == "local_step" else ()),
            *((MUJOCO_OBSERVED_POINT_ROUTE_ID,) if enable_observed_point_motion else ()),
            *((MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V2,)
              if enable_anchor_motion_v2 and safe_tcp_motion_profile == "local_step" else ()),
            *((MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V3,)
              if enable_anchor_region_motion_v3 and safe_tcp_motion_profile == "local_step" else ()),
            *((MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V4,)
              if enable_anchor_hover_motion_v4 and safe_tcp_motion_profile == "local_step" else ()),
        ),
    )
    entries = [standard_entry]
    if enable_safe_tcp_pose and safe_tcp_motion_profile != "local_step":
        limits = resolve_safe_tcp_motion_limits(safe_tcp_motion_profile, enabled=True)
        relation_profile = replace(profile,
            profile_id="mujoco-agent-safe-tcp-object-relation-envelope-v1",
            side_config_by_side={side: replace(profile.side_config_by_side[side],
                relative_min_xyz_m=(-limits.max_translation_delta_m,) * 3,
                relative_max_xyz_m=(limits.max_translation_delta_m,) * 3,
                max_rotation_rad=limits.max_rotation_delta_rad)})
        entries.append(TCPPoseEnvelopeCatalogEntry(
            envelope_id="mujoco.agent_safe_tcp_pose.safety.object_relation.v1",
            profile=relation_profile,
            constraint_set_hash=canonical_content_hash({
                "tcp_pose_envelope_profile_hash": relation_profile.semantic_content_hash,
                "safe_tcp_route_only": True,
                "canonical_keyframe_safety_retained": True,
                "coupled_ik_safety_retained": True,
            }, domain="dexteleop.mujoco.agent_safe_tcp_relation_constraint_set.v1"),
            allowed_route_ids=(MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1,
                *((MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V2,) if enable_safe_tcp_path_constraints else ()),
                *((MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V2,) if enable_anchor_motion_v2 else ()),
                *((MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V3,) if enable_anchor_region_motion_v3 else ()),
                *((MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V4,) if enable_anchor_hover_motion_v4 else ()))))
    if enable_grasp_approach:
        grasp_profile = replace(
            profile,
            profile_id="mujoco-agent-grasp-absolute-envelope-v1",
            side_config_by_side={
                "right": TCPPoseEnvelopeSideConfig(
                    absolute_min_xyz_m=(bounds[0], bounds[2], bounds[4]),
                    absolute_max_xyz_m=(bounds[1], bounds[3], bounds[5]),
                    max_rotation_rad=math.pi,
                )
            },
        )
        entries.append(
            TCPPoseEnvelopeCatalogEntry(
                envelope_id="mujoco.agent_grasp_approach.safety.absolute.v1",
                profile=grasp_profile,
                constraint_set_hash=canonical_content_hash(
                    {
                        "profile": grasp_profile.semantic_content_hash,
                        "contact_permission": ("ordinary_approach_noncontact;paired_execution_uses_retained_terminal_policy"
                            if enable_fast_grasp else "none_noncontact_joint_path_only"),
                        **({"retained_terminal_policy_hash": grasp_contact_binding.semantic_content_hash}
                           if enable_fast_grasp and grasp_contact_binding is not None else {}),
                    },
                    domain="mujoco.grasp_approach.absolute_envelope.v1",
                ),
                allowed_route_ids=(MUJOCO_AGENT_GRASP_APPROACH_ROUTE_ID_V1,
                    *(("sim.mujoco.motion.execute_grasp_approach_plan.v1",) if enable_fast_grasp else ())),
            )
        )
        if grasp_contact_binding is not None:
            contact_profile = replace(grasp_profile, profile_id="mujoco-agent-grasp-contact-envelope-v1")
            entries.append(TCPPoseEnvelopeCatalogEntry(
                envelope_id="mujoco.agent_grasp_contact.safety.absolute.v1",
                profile=contact_profile,
                constraint_set_hash=canonical_content_hash(
                    {"profile":contact_profile.semantic_content_hash,
                     "contact_binding":grasp_contact_binding.to_payload()},
                    domain="mujoco.grasp_contact.absolute_envelope.v1"),
                allowed_route_ids=(MUJOCO_AGENT_GRASP_CONTACT_ROUTE_ID_V1,),
            ))
            if enable_object_motion:
                object_profile = replace(grasp_profile, profile_id="mujoco-agent-object-motion-envelope-v1")
                entries.append(TCPPoseEnvelopeCatalogEntry(
                    envelope_id="mujoco.agent_object_motion.safety.absolute.v1",
                    profile=object_profile,
                    constraint_set_hash=canonical_content_hash({
                        "profile": object_profile.semantic_content_hash,
                        "target_contact_binding": grasp_contact_binding.to_payload(),
                        "payload_policy": "frozen_measured_relation_full_native_geometry_v1",
                    }, domain="mujoco.object_motion.absolute_envelope.v1"),
                    allowed_route_ids=(MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V1,),
                ))
                if enable_object_support_motion_v2:
                    entries.append(TCPPoseEnvelopeCatalogEntry(
                        envelope_id="mujoco.agent_object_support_motion.safety.absolute.v2",
                        profile=replace(grasp_profile,
                            profile_id="mujoco-agent-object-support-motion-envelope-v2"),
                        constraint_set_hash=canonical_content_hash({
                            "target_contact_binding": grasp_contact_binding.to_payload(),
                            "support_body_names": list(object_support_body_names),
                            "payload_policy": "frozen_measured_relation_complete_dynamic_support_v2",
                        }, domain="mujoco.object_support_motion.absolute_envelope.v2"),
                        allowed_route_ids=(MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V2,),
                    ))
    return TCPPoseEnvelopeSafetyRuntime(
        catalog_entries=tuple(entries),
        target_frame_binding_hash=target_binding_hash,
        tool_frame_binding_hash_by_side={side: tool_binding_hash},
        reference_frame_by_side={side: reference_frame},
        clock_domain_id=target_config.clock_domain_id,
    )


def _bind_grippers(
    bundle: MujocoAgentTcpPoseContractBundle,
    ports: Mapping[str, MujocoGripperControlPort],
    *, release_support_body_names: tuple[str, ...] = (),
) -> MujocoAgentTcpPoseContractBundle:
    hashes = {component: port.port_binding_hash for component, port in ports.items()}
    manifest = bundle.route_manifest
    version = "3" if release_support_body_names else "2"
    binding_payload = {"ports": hashes, "release_support_body_names": release_support_body_names}
    hand = build_gripper_skill_contract_bundle(GripperSkillDeploymentBinding(
        runtime_session_id=manifest.runtime_session_id, control_domain_id=manifest.control_domain_id,
        component_id="grippers", target_binding_id=manifest.target_binding_id,
        target_kind="sim", current_mode="sim_active", executor_id="mujoco-composed-grippers-v" + version,
        port_binding_hash=(canonical_content_hash(hashes, domain="mujoco-composed-grippers-v2")
            if version == "2" else canonical_content_hash(
                binding_payload, domain="mujoco-composed-grippers-v3")),
        station_manifest_hash=manifest.station_manifest_hash,
        execution_manifest_hash=manifest.execution_manifest_hash,
        skill_version=version, port_binding_hash_by_component=hashes,
        release_support_object_ids=release_support_body_names))
    definitions = replace(bundle.definition_catalog, skill_definitions=tuple(sorted(
        (*bundle.definition_catalog.skill_definitions, *hand.definition_catalog.skill_definitions),
        key=lambda d: (d.skill_id, d.skill_version))))
    schemas = replace(bundle.schema_catalog, schema_entries=tuple(sorted(
        (*bundle.schema_catalog.schema_entries, *hand.schema_catalog.schema_entries),
        key=lambda s: (s.skill_id, s.skill_version))))
    inline = inline_only_skill_io_catalog_for_routes(e.route for e in hand.route_manifest.skill_routes)
    io = replace(bundle.io_catalog, io_contracts=tuple(sorted(
        (*bundle.io_catalog.io_contracts, *inline.io_contracts), key=lambda c: (c.skill_id, c.skill_version))))
    hand_entry = hand.route_manifest.skill_routes[0]
    # Match the existing real composition: hands have a port-bound envelope,
    # not the arm's Cartesian workspace box.
    envelope = SkillDeploymentSafetyEnvelopeRef(
        object_id="mujoco-gripper:" + hand.deployment_binding.port_binding_hash,
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash=hand.deployment_binding.port_binding_hash,
        constraint_set_hash=hand.schema_catalog.schema_entries[0].parameter_schema_hash,
        allowed_route_ids=(hand_entry.route.route_id,))
    hand_entry = replace(hand_entry, deployment_binding=replace(
        hand_entry.deployment_binding, safety_envelope_refs=(envelope,)))
    entries = tuple(sorted((*manifest.skill_routes, hand_entry),
        key=lambda e: (e.route.skill_id, e.route.skill_version, e.route.route_id, e.route.route_version)))
    manifest = replace(manifest, skill_routes=entries,
        explicit_exclusions=tuple(item for item in manifest.explicit_exclusions if item != "hand_control"),
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash, skill_io_catalog_hash=io.semantic_content_hash)
    validators = AgentSkillPayloadValidatorRegistry(registry_id=bundle.payload_validators.registry_id,
        validators=tuple(b.payload_validators.validator_for(e.route)
            for b in (bundle, hand) for e in b.route_manifest.skill_routes))
    return replace(bundle, definition_catalog=definitions, schema_catalog=schemas, io_catalog=io,
        route_manifest=manifest, payload_validators=validators, gripper_contract=hand)


def build_mujoco_agent_tcp_pose_gateway_session(
    *,
    target_config: MujocoSimHILTargetConfig,
    allowed_client_ids: tuple[str, ...],
    motion_side: str = "right",
    max_retained_operations: int = 16,
    max_session_admissions_per_route: int | None = None,
    trace_sink: Callable[[Any], None] | None = None,
    history_storage: MujocoAgentTcpPoseHistoryStorageConfig | None = None,
    artifact_storage: MujocoAgentArtifactStorageConfig | None = None,
    segmentation_localizer: object | None = None,
    enable_object_localization: bool = False,
    enable_rgbd_capture_v3: bool = False,
    enable_segmentation_batch_v2: bool = False,
    enable_segmentation_geometry_v3: bool = False,
    enable_safe_tcp_pose: bool = False,
    enable_safe_tcp_path_constraints: bool = False,
    enable_tcp_continuation_queries: bool = False,
    enable_tcp_candidate_artifacts: bool = False,
    enable_relation_tcp_continuation_queries: bool = False,
    enable_region_tcp_continuation_queries: bool = False,
    region_tcp_continuation_version: str = "4",
    enable_observed_point_motion: bool = False,
    enable_anchor_motion_v2: bool = False,
    enable_anchor_region_motion_v3: bool = False,
    enable_anchor_hover_motion_v4: bool = False,
    enable_spatial_goals: bool = False,
    enable_object_goal_queries: bool = False,
    enable_surface_anchors: bool = False,
    enable_surface_anchor_v2: bool = False,
    enable_surface_anchor_v3: bool = False,
    surface_anchor_selection_policy: str = "prompt_nearest",
    enable_relation_goal_queries: bool = False,
    safe_tcp_motion_profile: str = "local_step",
    enable_tcp_pose_comparison: bool = False,
    enable_object_state: bool = False,
    enable_object_follow_observation: bool = False,
    enable_object_motion: bool = False,
    enable_object_support_motion_v2: bool = False,
    enable_named_release_v3: bool = False,
    object_support_body_names: tuple[str, ...] = (),
    safe_tcp_planning_clearance_padding_m: float = 0.002,
    enable_grippers: bool = False,
    enable_grasp_approach: bool = False,
    grasp_contact_target_body: str | None = None,
    fast_grasp_planner: Any | None = None,
    fast_grasp_approach_config: Mapping[str, float] | None = None,
    safe_motion_observation: MujocoObservedSafeMotionConfig | None = None,
    enable_tool_object_relations: bool = False,
    enable_tool_hover_queries: bool = False,
) -> MujocoAgentTcpPoseGatewaySession:
    """Build the exact default-off graph; do not start, record, or serve it."""

    if type(enable_tool_object_relations) is not bool or type(enable_tool_hover_queries) is not bool:
        raise TypeError("Tool-entity Skill switches must be bool.")
    if enable_tool_object_relations and (artifact_storage is None or segmentation_localizer is None
                                         or not enable_safe_tcp_pose):
        # The frozen multi-view RGB-D capture belongs to the planned safe TCP owner.
        raise ValueError("Tool-object relations require Artifact storage, a segmentation localizer "
                         "and the safe TCP pose Route.")
    if enable_tool_hover_queries and (not enable_surface_anchor_v2 or safe_motion_observation is None
                                      or not enable_safe_tcp_pose):
        raise ValueError("Tool hover queries require surface anchor v2 and the observed safe TCP scene.")

    if type(enable_named_release_v3) is not bool or (enable_named_release_v3 and not (
            enable_object_support_motion_v2 and enable_grippers
            and enable_object_follow_observation and enable_object_motion
            and motion_side == "right")):
        raise ValueError("Named release v3 requires the explicit right-hand object-support graph.")
    if type(enable_object_localization) is not bool:
        raise TypeError("enable_object_localization must be boolean.")
    if enable_object_localization and (
        artifact_storage is None or segmentation_localizer is None
        or not callable(getattr(segmentation_localizer, "localize", None))
    ):
        raise ValueError("Object localization requires Artifact storage and the shared segmentation localizer.")
    if enable_object_localization and enable_surface_anchor_v3 and max_retained_operations < 4:
        raise ValueError("Localized paired anchors require capture, localization, segmentation and anchor retention.")
    if not isinstance(enable_safe_tcp_path_constraints, bool) or (
            enable_safe_tcp_path_constraints and not enable_safe_tcp_pose):
        raise ValueError("Safe TCP path constraints require explicitly enabled safe TCP motion.")
    if not isinstance(enable_tcp_continuation_queries, bool) or (enable_tcp_continuation_queries
            and not enable_safe_tcp_path_constraints):
        raise ValueError("Continuation queries require explicitly enabled safe TCP path constraints.")
    if not isinstance(enable_tcp_candidate_artifacts, bool) or (enable_tcp_candidate_artifacts
            and not (enable_tcp_continuation_queries and enable_surface_anchor_v2 and enable_relation_goal_queries)):
        raise ValueError("Candidate Artifacts require continuation queries and the v2 anchor/relation family.")
    if not isinstance(enable_relation_tcp_continuation_queries, bool) or (enable_relation_tcp_continuation_queries
            and not (enable_tcp_continuation_queries and enable_surface_anchor_v2 and enable_relation_goal_queries)):
        raise ValueError("Relation continuation v3 requires continuation queries and the v2 anchor/relation family.")
    if not isinstance(enable_region_tcp_continuation_queries, bool) or (enable_region_tcp_continuation_queries
            and not enable_relation_tcp_continuation_queries):
        raise ValueError("Region continuation v4 requires the explicitly enabled relation continuation v3 family.")
    if (type(region_tcp_continuation_version) is not str or region_tcp_continuation_version not in {"4", "5"}
            or region_tcp_continuation_version != "4" and not enable_region_tcp_continuation_queries):
        raise ValueError("Region continuation version 5 requires explicitly enabled region queries; choose '4' or '5'.")
    if enable_surface_anchor_v3 and (enable_tcp_candidate_artifacts or enable_relation_tcp_continuation_queries):
        raise ValueError("Surface anchor v3 cannot be combined with candidate Artifacts or relation continuations.")

    from agent_gateway.skills.perception.surface_anchor_v2 import INTERIOR_SELECTION_POLICIES
    if not isinstance(enable_object_goal_queries, bool) or (enable_object_goal_queries and not enable_safe_tcp_pose):
        raise ValueError("Object goal queries require explicitly enabled safe TCP geometry.")

    if not isinstance(enable_surface_anchors, bool) or (enable_surface_anchors and not (artifact_storage is not None and enable_object_goal_queries)):
        raise ValueError("Surface anchors require Artifact storage and object goal queries.")

    if enable_surface_anchors and max_retained_operations < 2:
        raise ValueError("Surface anchors require retention for at least producer and consumer operations.")

    if type(enable_surface_anchor_v3) is not bool or (enable_surface_anchor_v3 and not (
            enable_surface_anchor_v2 and enable_rgbd_capture_v3 and enable_segmentation_geometry_v3)):
        raise ValueError("Surface anchor v3 requires explicit RGB-D, segmentation v3 and compatible v2 consumers.")

    if enable_surface_anchor_v3 and max_retained_operations < 3:
        raise ValueError("Surface anchor v3 requires simultaneous capture, segmentation and anchor retention.")

    if not isinstance(enable_surface_anchor_v2, bool) or (enable_surface_anchor_v2
            and not (enable_surface_anchors and segmentation_localizer is not None)):
        raise ValueError("Surface anchor v2 requires surface anchors and a segmentation model.")

    if (not isinstance(surface_anchor_selection_policy, str)
            or surface_anchor_selection_policy not in INTERIOR_SELECTION_POLICIES
            or surface_anchor_selection_policy != "prompt_nearest" and not enable_surface_anchor_v2):
        raise ValueError("A non-default surface selection policy requires explicitly enabled v2 anchors.")

    if not isinstance(enable_relation_goal_queries, bool) or (enable_relation_goal_queries and not enable_surface_anchors):
        raise ValueError("Relation goal queries require explicitly enabled surface anchors.")

    resolve_safe_tcp_motion_limits(safe_tcp_motion_profile, enabled=enable_safe_tcp_pose)

    for name, enabled, version in (
        ("enable_segmentation_batch_v2", enable_segmentation_batch_v2, "2"),
        ("enable_segmentation_geometry_v3", enable_segmentation_geometry_v3, "3"),
    ):
        if type(enabled) is not bool:
            raise TypeError(f"{name} must be boolean.")
        if enabled and (segmentation_localizer is None or artifact_storage is None):
            raise ValueError(f"Segmentation batch v{version} requires Artifact storage and a segmentation model.")
    if enable_segmentation_geometry_v3 and max_retained_operations < 2:
        raise ValueError("RGB-input geometry requires retention for at least producer and consumer operations.")
    if enable_segmentation_batch_v2 or enable_segmentation_geometry_v3:
        if not isinstance(artifact_storage, MujocoAgentArtifactStorageConfig):
            raise TypeError("artifact_storage must be typed or None.")
        for name in ("start", "close", "segment", "encode_segment_artifacts"):
            if not callable(getattr(segmentation_localizer, name, None)):
                raise TypeError(f"segmentation_localizer must expose {name}().")
        model_spec = getattr(segmentation_localizer, "model_spec", None)
        if any(not isinstance(value, str) or not value or value != value.strip() for value in (
                getattr(model_spec, "model_profile_id", None),
                getattr(model_spec, "semantic_content_hash", None),
                getattr(segmentation_localizer, "semantic_content_hash", None))):
            raise ValueError("Segmentation batch v2/v3 requires typed model identity strings.")
    if not isinstance(target_config, MujocoSimHILTargetConfig):
        raise TypeError("target_config must be MujocoSimHILTargetConfig.")
    side = _validated_motion_side(motion_side)
    if safe_motion_observation is not None:
        if not isinstance(safe_motion_observation, MujocoObservedSafeMotionConfig):
            raise TypeError("safe_motion_observation must be MujocoObservedSafeMotionConfig.")
        if not enable_safe_tcp_pose or artifact_storage is None:
            raise ValueError("Observed safe motion requires the original safe Route and RGB-D resources.")
        if target_config.realtime_worker:
            raise ValueError("Observed safe motion requires the same deterministic physics owner.")
    if type(enable_rgbd_capture_v3) is not bool or (enable_rgbd_capture_v3 and artifact_storage is None):
        raise ValueError("RGB-D capture v3 requires explicit Artifact storage.")
    if type(enable_observed_point_motion) is not bool:
        raise TypeError("enable_observed_point_motion must be boolean.")
    if enable_observed_point_motion and safe_motion_observation is None:
        raise ValueError("Observed-point motion requires explicit sensor-shaped safe motion.")
    if type(enable_anchor_motion_v2) is not bool or (enable_anchor_motion_v2
            and not (enable_observed_point_motion and enable_surface_anchor_v2)):
        raise ValueError("Anchor motion v2 requires observed safe motion and surface anchor v2.")
    if type(enable_anchor_region_motion_v3) is not bool or (enable_anchor_region_motion_v3
            and not enable_anchor_motion_v2):
        raise ValueError("Anchor region v3 requires the opt-in anchor v2 family.")
    if type(enable_anchor_hover_motion_v4) is not bool or (enable_anchor_hover_motion_v4
            and not enable_anchor_motion_v2):
        raise ValueError("Anchor hover v4 requires the opt-in anchor v2 family.")
    component = f"{side}_arm"
    inactive = "left" if side == "right" else "right"
    if not isinstance(enable_grippers, bool):
        raise TypeError("enable_grippers must be boolean.")
    if enable_grippers and target_config.realtime_worker:
        raise ValueError(
            "Composed gripper Skills require deterministic physics ownership."
        )
    if not isinstance(enable_safe_tcp_pose, bool):
        raise TypeError("enable_safe_tcp_pose must be boolean.")
    if not isinstance(enable_spatial_goals, bool):
        raise TypeError("enable_spatial_goals must be boolean.")
    if not isinstance(enable_tcp_pose_comparison, bool):
        raise TypeError("enable_tcp_pose_comparison must be boolean.")
    if type(enable_object_state) is not bool:
        raise TypeError("enable_object_state must be boolean.")
    if type(enable_object_follow_observation) is not bool:
        raise TypeError("enable_object_follow_observation must be boolean.")
    if enable_object_follow_observation and not enable_object_state:
        raise ValueError("Object follow observation requires explicit object-state feedback.")
    if type(enable_object_motion) is not bool:
        raise TypeError("enable_object_motion must be boolean.")
    if enable_object_motion and (not enable_object_follow_observation or not enable_grippers
                                 or not enable_grasp_approach or grasp_contact_target_body is None):
        raise ValueError("Object motion requires explicit follow observation, grippers, grasp approach and contact target.")
    if (type(enable_object_support_motion_v2) is not bool
            or enable_object_support_motion_v2 != bool(object_support_body_names)
            or (enable_object_support_motion_v2 and not enable_object_motion)):
        raise ValueError("Object support v2 requires original object motion and an explicit support allowlist.")
    object_support_body_names = tuple(object_support_body_names)
    if (len(set(object_support_body_names)) != len(object_support_body_names)
            or any(not isinstance(name, str) or not name or name.strip() != name
                   for name in object_support_body_names)):
        raise ValueError("Object support IDs must be unique native names.")
    if enable_object_state and grasp_contact_target_body is None:
        raise ValueError("Object-state feedback requires an explicit grasp contact target.")
    if not isinstance(enable_grasp_approach, bool):
        raise TypeError("enable_grasp_approach must be boolean.")
    if side == "left" and (
        enable_grippers
        or enable_grasp_approach
        or grasp_contact_target_body is not None
        or enable_rgbd_capture_v3
        or enable_segmentation_batch_v2
        or enable_segmentation_geometry_v3
        or enable_spatial_goals
        or enable_object_goal_queries
        or enable_surface_anchors
        or enable_surface_anchor_v2
        or enable_surface_anchor_v3
        or enable_object_localization
        or enable_safe_tcp_path_constraints
        or enable_tcp_continuation_queries
        or enable_tcp_candidate_artifacts
        or enable_relation_tcp_continuation_queries
        or enable_region_tcp_continuation_queries
        or enable_relation_goal_queries
        or history_storage is not None
        or segmentation_localizer is not None
    ):
        raise ValueError(
            "Left motion mode currently supports baseline TCP/state Routes only; "
            "gripper, grasp, spatial goal, history and segmentation combinations are not enabled."
        )
    if enable_grasp_approach and (not enable_grippers or target_config.realtime_worker):
        raise ValueError(
            "Grasp approach requires composed grippers and deterministic physics ownership."
        )
    if grasp_contact_target_body is not None and not enable_grasp_approach:
        raise ValueError("Terminal contact requires explicit grasp approach enablement.")
    if fast_grasp_planner is not None:
        if side != "right":
            raise ValueError("Fast grasp currently requires the native right arm.")
        from deployments.perception.sam2_graspgenx_point_grasp import Sam2GraspGenXPointGraspPlanner

        if not isinstance(fast_grasp_planner, Sam2GraspGenXPointGraspPlanner):
            raise TypeError("fast_grasp_planner must be the existing typed GraspGenX planner.")
        if (not enable_grasp_approach or grasp_contact_target_body is None or artifact_storage is None):
            raise ValueError("Fast grasp requires Artifact storage and the complete target-bound native approach graph.")
    elif fast_grasp_approach_config is not None:
        raise ValueError("A fast grasp configuration requires its model binding.")
    if history_storage is not None and not isinstance(
        history_storage,
        MujocoAgentTcpPoseHistoryStorageConfig,
    ):
        raise TypeError("history_storage must be typed or None.")
    if artifact_storage is not None and not isinstance(
        artifact_storage,
        MujocoAgentArtifactStorageConfig,
    ):
        raise TypeError("artifact_storage must be typed or None.")
    if segmentation_localizer is not None:
        for name in ("start", "close", "segment"):
            if not callable(getattr(segmentation_localizer, name, None)):
                raise TypeError(f"segmentation_localizer must expose {name}().")
        if artifact_storage is None:
            raise ValueError("MuJoCo segmentation requires Artifact storage.")
        model_spec = getattr(segmentation_localizer, "model_spec", None)
        if not str(getattr(model_spec, "model_profile_id", "")).strip() or not str(
            getattr(model_spec, "semantic_content_hash", "")
        ).strip():
            raise ValueError("segmentation_localizer must expose a verified model_spec.")
        if not str(
            getattr(segmentation_localizer, "semantic_content_hash", "")
        ).strip():
            raise ValueError("segmentation_localizer must expose semantic_content_hash.")
    if target_config.station_pose_contract_path is None:
        target_config = replace(
            target_config,
            station_pose_contract_path=DEFAULT_STATION_POSE_CONTRACT_PATH,
        )
    validate_mujoco_agent_tcp_pose_target_config(target_config, motion_side=side)
    if side == "left":
        _validate_left_tool_sources(target_config)
    prepared = prepare_mujoco_sim_hil_target(target_config)
    if enable_object_support_motion_v2:
        scene_objects = {item.name: item for item in prepared.task_config.base_contract().scene.objects}
        for name in object_support_body_names:
            item = scene_objects.get(name)
            if (name == grasp_contact_target_body or item is None or item.shape != "box"):
                raise ValueError("Object support v2 requires another declared free box object.")
    if fast_grasp_planner is not None:
        from deployments.perception.dmtac_grasp_overlay import (
            DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD, reviewed_graspgenx_dmtac_solid_geometry,
        )
        from robot_motion.rm65 import load_rm65_tool_profile, tool_tcp_pose_flange_m_rad

        actual_tcp = tool_tcp_pose_flange_m_rad(load_rm65_tool_profile(
            prepared.task_config.right_tool_profile_path, side="right"))
        if (any(not math.isclose(float(a), float(b), abs_tol=1e-9, rel_tol=0)
                for a, b in zip(actual_tcp, DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD))
                or fast_grasp_planner.solid_geometry != reviewed_graspgenx_dmtac_solid_geometry(
                    maximum_opening_m=fast_grasp_planner.max_gripper_width_m,
                    alignment_profile_id=fast_grasp_planner.alignment_profile_id)
                or not all(prepared.scene.has_body(name) for name in (
                    "right_gripper_l_dm_tac_w2m_sensor", "right_gripper_r_dm_tac_w2m_sensor"))):
            raise ValueError("Fast GraspGenX requires the reviewed DMTAC M tool TCP and native sensor assembly.")
    grasp_contact_binding = (
        None if grasp_contact_target_body is None else bind_task_grasp_contact(
            prepared.scene, prepared.gripper_profile, grasp_contact_target_body)
    )
    component_profile = build_mujoco_pose_component_target_profile(
        prepared.scene,
        profile_id=(
            MUJOCO_AGENT_TCP_POSE_COMPONENT_PROFILE_ID_V1
            if side == "right"
            else "mujoco.rm65.left_agent_tcp_pose.v1"
        ),
        arm_side_by_component={component: side},
        arm_reference_frame_by_component={component: f"{side}_arm_base"},
        arm_controlled_frame_by_component={component: "tool_tcp"},
        selection_group_by_component={component: "arms"},
    )
    runtime_rig_holder: dict[str, Any] = {}

    def capture_support_arm_pose() -> Mapping[str, Any]:
        rig = runtime_rig_holder.get("rig")
        if rig is None:
            raise RuntimeError(
                "MuJoCo support-pose capture requested before target binding."
            )
        return rig.capture_scene(
            lambda scene: {
                inactive: capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    prepared.task_config,
                )[inactive]
            }
        )

    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        component_profile,
        hand_action_spec_by_component={},
        support_arm_pose_provider=capture_support_arm_pose,
        support_arm_sides=(inactive,),
        support_pose_contract_id=(
            f"mujoco.measured_{inactive}_tool_tcp_support_for_agent_coupled_ik.v1"
        ),
    )
    component_execution_binding = build_mujoco_component_execution_binding(
        component_profile,
        materializer,
    )
    observation_sink = MujocoExecutionDifferenceObserver(
        selection_group_id="arms",
        component_ids=(component,),
    )
    tcp_pose_envelope_runtime = _build_standard_tcp_pose_envelope_runtime(
        prepared=prepared,
        target_config=target_config,
        enable_safe_tcp_pose=enable_safe_tcp_pose,
        enable_safe_tcp_path_constraints=enable_safe_tcp_path_constraints,
        enable_observed_point_motion=enable_observed_point_motion,
        enable_anchor_motion_v2=enable_anchor_motion_v2,
        enable_anchor_region_motion_v3=enable_anchor_region_motion_v3,
        enable_anchor_hover_motion_v4=enable_anchor_hover_motion_v4,
        safe_tcp_motion_profile=safe_tcp_motion_profile,
        enable_grasp_approach=enable_grasp_approach,
        grasp_contact_binding=grasp_contact_binding,
        enable_fast_grasp=fast_grasp_planner is not None,
        enable_object_motion=enable_object_motion,
        enable_object_support_motion_v2=enable_object_support_motion_v2,
        object_support_body_names=object_support_body_names,
        motion_side=side,
    )
    target = build_mujoco_agent_skill_motion_target(
        config=target_config,
        component_execution_binding=component_execution_binding,
        prepared_target=prepared,
        execution_observation_sink=observation_sink,
        trace_sink=trace_sink,
        tcp_pose_envelope_runtime=tcp_pose_envelope_runtime,
        enable_safe_tcp_pose=enable_safe_tcp_pose or enable_grasp_approach,
        safe_tcp_planning_clearance_padding_m=safe_tcp_planning_clearance_padding_m,
        motion_side=side,
        prune_expired_proposals=True,
    )
    runtime_rig_holder["rig"] = target.runtime.rig
    captured_at = float(target.runtime.clock())
    initial_pose = target.runtime.rig.capture_scene(
        lambda scene: capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            scene,
            prepared.task_config,
        )[side]
    )
    initialization_event_hash = canonical_content_hash(
        {
            "task_initialization_info": dict(prepared.task_initialization_info),
            f"measured_{side}_tool_tcp": initial_pose,
            "captured_at": captured_at,
            "clock_domain_id": target_config.clock_domain_id,
        },
        domain="dexteleop.mujoco.agent_tcp_pose_qpos_initialization.v1",
    )
    tcp_pose_envelope_runtime.commit_reference(
        reference_id=("mujoco-agent-qpos-init-reference:" + initialization_event_hash),
        tcp_pose_by_side={side: initial_pose},
        captured_at_by_side={side: captured_at},
        source_event_ids=(
            "mujoco-qpos-initialization-completed:" + initialization_event_hash,
        ),
        lower_state="configured",
    )
    if target.runtime.action_schema.semantic_content_hash != (
        component_profile.action_schema.semantic_content_hash
    ):
        raise RuntimeError("Agent motion target changed its component schema.")
    application = MujocoAgentSkillMotionApplication(
        target=target,
        calibration_hash=_calibration_hash(target),
        motion_side=side,
    )
    history_store = None
    artifact_store = None
    try:
        gripper_ports = {}
        if enable_grippers:
            if sorted(prepared.gripper_profile.side_by_gripper.values()) != ["left", "right"]:
                raise ValueError("Composed grippers require both configured hands.")
            gripper_ports = {side + "_gripper": MujocoGripperControlPort(
                rig=target.runtime.rig, gripper_name=name, component_id=side + "_gripper",
                clock=target.runtime.clock, arm_held=lambda: application.held,
                target_contact_binding=(
                    grasp_contact_binding
                    if enable_object_follow_observation and side == "right" else None
                ),
                release_support_body_names=(object_support_body_names
                    if enable_named_release_v3 and side == "right" else ()),
                held_transaction=(application.run_held_gripper_transition
                    if enable_named_release_v3 and side == "right" else None))
                for name, side in prepared.gripper_profile.side_by_gripper.items()}
        tool_entity_by_side = None
        if enable_tool_object_relations or enable_tool_hover_queries:
            from deployments.perception.dmtac_tool_entity import default_tool_entity_by_side
            from robot_motion.rm65.tool_profiles import load_rm65_tool_profile

            # Same reviewed entity binding as the RealMan Gateway: tool
            # calibrations and the session's station URDF (twin geometry).
            tool_profiles = {side: load_rm65_tool_profile(getattr(application.task_config, side + "_tool_profile_path"),
                                                          side=side) for side in ("left", "right")}
            tool_entity_by_side = default_tool_entity_by_side(
                {side: tuple(float(x) for x in profile["tcp"]["pose_flange_m_rad"])
                 for side, profile in tool_profiles.items()}, tool_profiles,
                urdf_path=application.runtime.rig.scene.load_result.conversion.urdf_path)
            if not tool_entity_by_side:
                raise ValueError("Tool-entity Skills need the reviewed DMTAC_M tool on at least one arm.")
        contract_bundle = build_mujoco_agent_tcp_pose_contract_bundle(
            application=application,
            enable_tool_object_relations=enable_tool_object_relations,
            enable_tool_hover_queries=enable_tool_hover_queries,
            tool_entity_by_side=tool_entity_by_side,
            tool_hover_roles=(() if safe_motion_observation is None
                              else tuple(safe_motion_observation.observation.roles)),
            history_profile=(
                None if history_storage is None else history_storage.profile
            ),
            enable_artifact_perception=artifact_storage is not None,
            enable_object_localization=enable_object_localization,
            enable_rgbd_capture_v3=enable_rgbd_capture_v3,
            enable_segmentation_batch_v2=enable_segmentation_batch_v2,
            enable_segmentation_geometry_v3=enable_segmentation_geometry_v3,
            segmentation_model_profile_id=(
                None
                if segmentation_localizer is None
                else segmentation_localizer.model_spec.model_profile_id
            ),
            segmentation_model_profile_hash=(
                None
                if segmentation_localizer is None
                else segmentation_localizer.model_spec.semantic_content_hash
            ),
            segmentation_inference_implementation_hash=(
                None
                if segmentation_localizer is None
                else segmentation_localizer.semantic_content_hash
            ),
            segmentation_inference_execution_mode=getattr(
                segmentation_localizer, "inference_execution_mode", "in_process"),
            enable_safe_tcp_pose=enable_safe_tcp_pose,
            enable_safe_tcp_path_constraints=enable_safe_tcp_path_constraints,
            enable_tcp_continuation_queries=enable_tcp_continuation_queries,
            enable_tcp_candidate_artifacts=enable_tcp_candidate_artifacts,
            enable_relation_tcp_continuation_queries=enable_relation_tcp_continuation_queries,
            enable_region_tcp_continuation_queries=enable_region_tcp_continuation_queries,
            region_tcp_continuation_version=region_tcp_continuation_version,
            enable_observed_point_motion=enable_observed_point_motion,
            enable_anchor_motion_v2=enable_anchor_motion_v2,
            enable_anchor_region_motion_v3=enable_anchor_region_motion_v3,
            enable_anchor_hover_motion_v4=enable_anchor_hover_motion_v4,
            safe_motion_observation_profile_hash=(None if safe_motion_observation is None
                else safe_motion_observation.bound_profile_hash(
                    minimum_clearance_m=application.scene.scene_spec.collision.safety_minimum_clearance_m,
                    planning_clearance_padding_m=safe_tcp_planning_clearance_padding_m)),
            enable_spatial_goals=enable_spatial_goals,
            enable_object_goal_queries=enable_object_goal_queries,
            enable_surface_anchors=enable_surface_anchors,
            enable_surface_anchor_v2=enable_surface_anchor_v2,
            enable_surface_anchor_v3=enable_surface_anchor_v3,
            surface_anchor_selection_policy=surface_anchor_selection_policy,
            enable_relation_goal_queries=enable_relation_goal_queries,
            safe_tcp_motion_profile=safe_tcp_motion_profile,
            enable_tcp_pose_comparison=enable_tcp_pose_comparison,
            enable_object_state=enable_object_state,
            enable_object_follow_observation=enable_object_follow_observation,
            enable_object_motion=enable_object_motion,
            enable_object_support_motion_v2=enable_object_support_motion_v2,
            object_support_body_names=object_support_body_names,
            enable_grasp_approach=enable_grasp_approach,
            grasp_contact_binding=grasp_contact_binding,
        )
        if enable_grippers:
            contract_bundle = _bind_grippers(contract_bundle, gripper_ports,
                release_support_body_names=(object_support_body_names
                    if enable_named_release_v3 else ()))
        if fast_grasp_planner is not None:
            from .grasp_deployment import build_mujoco_fast_grasp_contract

            config = ({"pregrasp_distance_m": .07, "near_distance_m": .025,
                       "terminal_offset_m": 0., "terminal_speed_scale": .25}
                      if fast_grasp_approach_config is None else dict(fast_grasp_approach_config))
            contract_bundle = build_mujoco_fast_grasp_contract(contract_bundle,
                model_profile_hash=fast_grasp_planner.semantic_content_hash, approach_config=config)
        if history_storage is not None:
            source_binding = contract_bundle.history_source_binding
            if source_binding is None:
                raise RuntimeError("History contract lacks its source binding.")
            history_store = SQLiteSourceObservationHistoryStore(
                root=history_storage.store_root_for(source_binding),
                binding=source_binding,
                limits=history_storage.profile.store_limits,
                filename=history_storage.filename,
            )
        if artifact_storage is not None:
            artifact_store = RecoverableAgentArtifactStore(
                root=artifact_storage.store_root_for(
                    contract_bundle.route_manifest.runtime_session_id
                ),
                quota_bytes=artifact_storage.quota_bytes,
                max_artifact_bytes=artifact_storage.max_artifact_bytes,
                max_chunk_bytes=artifact_storage.max_chunk_bytes,
            )
        return assemble_mujoco_agent_tcp_pose_gateway_session(
            application=application,
            contract_bundle=contract_bundle,
            allowed_client_ids=allowed_client_ids,
            max_retained_operations=max_retained_operations,
            max_session_admissions_per_route=(
                max_session_admissions_per_route
            ),
            source_observation_history_store=history_store,
            owns_source_observation_history_store=history_store is not None,
            artifact_store=artifact_store,
            owns_artifact_store=artifact_store is not None,
            segmentation_localizer=segmentation_localizer,
            fast_grasp_planner=fast_grasp_planner,
            gripper_ports=gripper_ports,
            safe_motion_observation=safe_motion_observation,
            tool_entity_by_side=tool_entity_by_side,
        )
    except Exception:
        if history_store is not None:
            history_store.close()
        if artifact_store is not None:
            artifact_store.close()
        application.close_after_hold()
        raise


__all__ = [
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ENVELOPE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_CALIBRATION_HASH_DOMAIN_V1",
    "MUJOCO_AGENT_TCP_POSE_COMPONENT_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_OBSERVATION_PLAN_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1",
    "MujocoAgentArtifactStorageConfig",
    "MujocoAgentTcpPoseHistoryStorageConfig",
    "build_mujoco_agent_tcp_pose_gateway_session",
    "mujoco_agent_tcp_pose_execution_observation_plan_v1",
    "validate_mujoco_agent_tcp_pose_target_config",
]
