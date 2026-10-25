"""Compose existing owners into an independent manual no-model target.

This is a trusted in-memory assembly API, not a portable Gateway importer.
The caller fences all source ingress/decisions and supplies trusted prepared
metadata plus a fresh materializer factory. Persistent provenance, transport
fencing, upper agent context and publication remain the enclosing store's job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from robot_motion.action.adaptation import RobotActionV1RepresentationAdapter
from robot_motion.action.executor import RobotActionResolver
from robot_motion.rm65.ik_checkpoint import copy_data, state_hash
from robot_motion.rm65.retargeter_checkpoint import (
    RetargeterCheckpoint,
    capture_retargeter,
)
from sim.mujoco.adapters.checkpoint import (
    ExecutionRigCheckpoint,
    ManualExecutionClock,
    capture_execution_rig,
    restore_execution_rig,
)
from sim.mujoco.adapters.checkpoint import (
    _qualify as validate_execution_boundary,
)
from sim.mujoco.ik import RM65RetargeterIK

from .agent_skill_target import (
    MujocoAgentSkillMotionTarget,
    _MujocoAgentSharedSafety,
    build_mujoco_agent_skill_motion_target,
)
from .component_action_profile import MujocoComponentActionRobotActionV1Materializer
from .pose_resolution import MujocoCoupledPoseToQposResolver
from .runtime import _MujocoTargetExecutionSubmissionPort, _RigSerializedPoseResolver
from .runtime_checkpoint import (
    MujocoRuntimeCheckpoint,
    capture_runtime,
)
from .runtime_checkpoint import (
    _qualify as validate_runtime_boundary,
)
from .target_preparation import MujocoSimHILPreparedTarget

_RESOLVER_FIELDS = {
    "ik",
    "pose_transformer",
    "planned_tcp_motion",
    "pose_target_safety_filter",
    "target_pose_smoother",
    "support_only_arm_sides",
    "active_arm_sides",
    "resolver_contract_id",
    "resolver_contract_hash",
    "_command_resolver",
    "_resolution_by_id",
    "_last_resolve_diagnostics",
    "_lock",
}


@dataclass(frozen=True)
class MujocoTargetCheckpoint:
    config: object
    motion_side: str
    rig: ExecutionRigCheckpoint
    runtime: MujocoRuntimeCheckpoint
    retargeter: RetargeterCheckpoint
    initial_qpos: dict
    binding_hash: str
    resolver_diagnostics: dict
    diagnostics_hash: str
    public_metadata_hash: str


@dataclass(frozen=True)
class _TargetRestore:
    checkpoint: MujocoTargetCheckpoint
    rig: object
    max_native_bytes: int


def _binding(runtime):
    resolver = runtime.pose_resolver
    ik = resolver.ik
    materializer = runtime.component_execution_binding.materializer
    return state_hash(
        {
            "resolver": (
                resolver.resolver_contract_id,
                resolver.resolver_contract_hash,
                resolver.support_only_arm_sides,
                resolver.active_arm_sides,
            ),
            "transforms": ik.world_from_arm_base_by_side,
            "materializer": materializer.semantic_content_hash,
            "action_schema": runtime.action_schema.semantic_content_hash,
            "target_contract": runtime.target_runtime_action_contract.semantic_content_hash,
            "safety_profile": runtime.core.shared_safety.profile.semantic_content_hash,
            "safety_task": runtime.core.shared_safety.task_config.to_metadata(),
        }
    )


def _public_metadata(owner):
    return state_hash(
        {
            "target_frame": owner.target_frame_profile.semantic_content_hash,
            "tool_frame": owner.tool_frame_profile.semantic_content_hash,
            "initialization": dict(owner.task_initialization_info),
        }
    )


def _core_bindings(runtime):
    core = runtime.core
    wrapper = core.resolver
    if (
        type(wrapper) is not _RigSerializedPoseResolver
        or vars(wrapper) != {"_rig": runtime.rig, "_resolver": runtime.pose_resolver}
        or core.dispatcher != runtime.rig.send_commands
        or core.contextual_dispatcher != runtime._dispatch_with_context
        or core.backend_hold != runtime._backend_hold
        or core.target_execution_port is not runtime.target_execution_port
    ):
        raise ValueError("Core callbacks do not belong to the target factory graph.")
    port = runtime.target_execution_port
    if type(port) is not _MujocoTargetExecutionSubmissionPort or vars(port) != {
        "_target_handler_id": f"mujoco-pose-target-handler:{runtime.config.control_domain_id}",
        "_component_target_capability": runtime.component_target_capability,
        "_activation_mode": "omitted",
        "_minimum_support_margin_sec": runtime.config.target_admission_margin_sec,
        "_clock": runtime.clock,
    }:
        raise ValueError("Target submission binding differs from the profile.")
    binding = runtime.component_execution_binding
    adapter = core.action_representation_adapter
    if type(adapter) is not RobotActionV1RepresentationAdapter or vars(adapter) != {
        "receiver_id": f"{runtime.config.owner_runtime_id}:mujoco-component-execution-adapter",
        "_component_target_capability": runtime.component_target_capability,
        "output_source": "action_arbiter",
        "adapter_contract_id": binding.handler_contract_id,
        "adapter_contract_hash": binding.handler_contract_hash,
        "materializer": binding.materializer,
        "clock": runtime.clock,
        "expected_output_effectors": (
            *binding.output_effector_by_component.values(),
            *binding.support_only_output_effectors,
        ),
        "output_action_contract_hash": runtime.target_runtime_action_contract.semantic_content_hash,
    }:
        raise ValueError(
            "Representation adapter differs from the target factory graph."
        )


def _qualify(target):
    if (
        type(target) is not MujocoAgentSkillMotionTarget
        or target.tcp_pose_envelope_runtime is not None
        or target.planned_tcp_motion is not None
    ):
        raise ValueError(
            "Target checkpoint does not own external safety or motion plans."
        )
    runtime = target.runtime
    if type(runtime.clock) is not ManualExecutionClock:
        raise ValueError("Target checkpoint requires a manual logical clock.")
    _core_bindings(runtime)
    safety = runtime.core.shared_safety
    if (
        type(safety) is not _MujocoAgentSharedSafety
        or set(vars(safety)) != {"rig", "task_config", "clock", "profile"}
        or safety.rig is not runtime.rig
        or safety.clock is not runtime.clock
        or safety.task_config is not target.task_config
    ):
        raise ValueError("Shared safety must retain the original factory bindings.")
    resolver = runtime.pose_resolver
    if (
        type(resolver) is not MujocoCoupledPoseToQposResolver
        or set(vars(resolver)) != _RESOLVER_FIELDS
        or any(
            getattr(resolver, key) is not None
            for key in (
                "pose_transformer",
                "planned_tcp_motion",
                "pose_target_safety_filter",
                "target_pose_smoother",
            )
        )
        or resolver._resolution_by_id
        or type(resolver._command_resolver) is not RobotActionResolver
        or vars(resolver._command_resolver)
    ):
        raise ValueError("Unsupported resolver graph or unfinished resolution.")
    ik = resolver.ik
    if (
        type(ik) is not RM65RetargeterIK
        or set(vars(ik))
        != {
            "scene",
            "retargeter",
            "admitted_qpos_source",
            "world_from_arm_base_by_side",
            "planner",
        }
        or ik.planner is not ik
        or ik.scene is not target.scene
        or ik.scene is not runtime.rig.scene
        or ik.admitted_qpos_source != runtime.rig.accepted_qcmd_by_side
    ):
        raise ValueError("IK adapter bindings do not belong to this target.")
    materializer = runtime.component_execution_binding.materializer
    if (
        type(materializer) is not MujocoComponentActionRobotActionV1Materializer
        or materializer.scene is not target.scene
    ):
        raise ValueError("Materializer must bind the target's own scene.")


def capture_motion_target(target, *, prepared, max_native_bytes):
    """Capture only at an externally fenced, completed operation boundary.

    The rig lock prevents physics ticks; it cannot fence upper admission or
    tool calls. The parent must hold that application-level exclusion too.
    """
    _qualify(target)
    if (
        type(prepared) is not MujocoSimHILPreparedTarget
        or prepared.scene is not target.scene
        or prepared.task_config is not target.task_config
        or _public_metadata(prepared) != _public_metadata(target)
        or prepared.config.realtime_worker
        or not prepared.config.collision_stop_enabled
    ):
        raise ValueError(
            "Checkpoint requires the original manual coupled prepared target."
        )
    runtime = target.runtime
    with runtime.rig._lock, runtime.clock._lock:
        rig = capture_execution_rig(runtime.rig, max_native_bytes=max_native_bytes)
        logical = capture_runtime(runtime)
        ik = runtime.pose_resolver.ik.retargeter
        retargeter = capture_retargeter(ik, max_native_bytes=max_native_bytes)
        diagnostics = copy_data(runtime.pose_resolver._last_resolve_diagnostics)
        result = MujocoTargetCheckpoint(
            prepared.config,
            target.motion_side,
            rig,
            logical,
            retargeter,
            {
                side: copy_data(getattr(ik.config, side + "_initial_qpos"))
                for side in ("left", "right")
            },
            _binding(runtime),
            diagnostics,
            state_hash(diagnostics),
            _public_metadata(target),
        )
        validate_runtime_boundary(runtime)
        validate_execution_boundary(runtime.rig)
        _qualify(target)
        return result


def validate_target_restore(restore, *, config, prepared, binding, clock, motion_side):
    if (
        type(restore) is not _TargetRestore
        or type(restore.checkpoint) is not MujocoTargetCheckpoint
        or type(prepared) is not MujocoSimHILPreparedTarget
        or restore.checkpoint.config != config
        or prepared.config != config
        or not config.collision_stop_enabled
        or config.realtime_worker
        or prepared.scene is not restore.rig.scene
        or clock is not restore.rig._clock
        or restore.checkpoint.motion_side != motion_side
        or getattr(getattr(binding, "materializer", None), "scene", None)
        is not prepared.scene
        or restore.checkpoint.runtime.clock_value != restore.checkpoint.rig.clock_value
    ):
        raise ValueError("Invalid independently restored target binding.")


def restore_resolver_state(runtime, checkpoint):
    if (
        _binding(runtime) != checkpoint.binding_hash
        or state_hash(checkpoint.resolver_diagnostics) != checkpoint.diagnostics_hash
    ):
        raise ValueError("Resolver/materializer contract or diagnostics differ.")
    runtime.pose_resolver._last_resolve_diagnostics = copy_data(
        checkpoint.resolver_diagnostics
    )


def restore_motion_target(
    checkpoint,
    *,
    prepared,
    rig_config,
    expected_model_sha256,
    binding_factory,
    max_native_bytes,
):
    """Re-enter the original factory with fresh bindings and restored owners.

    ``prepared`` supplies caller-trusted static metadata, never live state.
    ``binding_factory`` must construct callbacks against the supplied new scene;
    copying a source closure is not a valid cold binding. No source is closed or
    updated, and a failed candidate is closed before it can be published.
    """
    if (
        type(checkpoint) is not MujocoTargetCheckpoint
        or type(prepared) is not MujocoSimHILPreparedTarget
        or checkpoint.config != prepared.config
        or checkpoint.public_metadata_hash != _public_metadata(prepared)
        or not callable(binding_factory)
    ):
        raise ValueError(
            "Target checkpoint requires trusted matching profile metadata."
        )
    rig = restore_execution_rig(
        checkpoint.rig,
        config=rig_config,
        load_result=prepared.scene.load_result,
        expected_model_sha256=expected_model_sha256,
        max_native_bytes=max_native_bytes,
    )
    try:
        fresh = replace(prepared, scene=rig.scene)
        target = build_mujoco_agent_skill_motion_target(
            config=checkpoint.config,
            prepared_target=fresh,
            component_execution_binding=binding_factory(fresh),
            motion_side=checkpoint.motion_side,
            clock=rig._clock,
            _restore=_TargetRestore(checkpoint, rig, max_native_bytes),
        )
        _qualify(target)
        return target
    except BaseException:
        rig.close()
        raise
