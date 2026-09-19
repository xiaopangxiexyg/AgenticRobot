"""Explicit read-only CAN source and single-human RM65 simulation assembly."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from control_station.sources.rebot_b601_rm65_motion import B601RM65MotionSession


def assemble_b601_rm65_session(
    runtime,
    *,
    backend,
    model,
    mapping_profile,
    measure_target,
    maximum_source_age_sec: float = 0.15,
    maximum_measurement_age_sec: float = 0.1,
    maximum_translation_m: float | None = 0.1,
    maximum_rotation_rad: float | None = 0.3,
) -> B601RM65MotionSession:
    """Bind existing owners; do not start or close any injected resource."""
    from control_station.composition.rebot_b601_rm65_residual import (
        B601RM65ResidualSource,
    )
    from control_station.sources.rebot_b601_pose import (
        B601ReadOnlyPoseSource,
        B601ReadOnlyPoseSourceSpec,
    )
    from control_station.sources.rebot_b601_rm65_motion import (
        B601RM65MotionBinding,
        B601RM65MotionSession,
    )
    from robot_motion.action import build_relative_delta_action_schema

    schema = build_relative_delta_action_schema(
        runtime.action_schema,
        component_ids=("right_arm",),
        schema_id="b601-rm65-readonly-human-residual",
    )
    source = B601RM65ResidualSource(
        B601ReadOnlyPoseSource(
            backend,
            model,
            spec=B601ReadOnlyPoseSourceSpec(
                clock_domain_id=runtime.config.clock_domain_id,
            ),
        ),
        mapping_profile.build_residual_adapter(schema),
    )
    binding = B601RM65MotionBinding(
        runtime,
        source,
        maximum_source_age_sec=maximum_source_age_sec,
        maximum_measurement_age_sec=maximum_measurement_age_sec,
        maximum_translation_m=maximum_translation_m,
        maximum_rotation_rad=maximum_rotation_rad,
        enabled=True,
    )
    return B601RM65MotionSession(binding, measure_target=measure_target)


def build_b601_rm65_sim_target(task_config_path: Path):
    """Prepare one existing actuated rig, canonical coupled IK and pose reader.

    Task-specific scripted resets are forbidden. No policy, composition,
    gripper primitive or direct-qpos diagnostic is enabled by this factory.
    The returned runtime is not started; its caller owns start/close.
    """
    from robot_motion.action import ExecutionMeasurementObservation
    from robot_motion.position_control import POSITION_60_120_PROFILE_NAME
    from robot_motion.rm65 import (
        DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
        RM65PositionCollisionPlannerConfig,
        endpoint_to_tool_transform,
        load_rm65_tool_profile,
    )
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from sim.mujoco.deploy import (
        MujocoCoupledPoseToQposResolver,
        MujocoDeployConfig,
        MujocoDeployRuntime,
    )
    from sim.mujoco.deploy.measurement import (
        build_mujoco_qpos_measurement_from_execution_cut,
    )
    from sim.mujoco.deploy.pose_kinematics import (
        build_mujoco_pose_observation_projector,
    )
    from sim.mujoco.deploy.target_config import MujocoSimHILTargetConfig
    from sim.mujoco.deploy.target_preparation import prepare_mujoco_sim_hil_target
    from sim.mujoco.ik import RM65CanonicalIK

    prepared = prepare_mujoco_sim_hil_target(
        MujocoSimHILTargetConfig(
            task_config_path=task_config_path,
            allow_scripted_task_reset=False,
            # B601 feedback arrives at roughly 20 Hz, while the RM65 target
            # has a validated 60 Hz IK/reference clock.  The motion session
            # explicitly reuses one still-fresh source sample between
            # hardware frames; selecting the 60 Hz profile avoids adding a
            # second 15 Hz bottleneck in the simulation loop.
            position_control_profile_name=POSITION_60_120_PROFILE_NAME,
            compose_target_hz=60.0,
        )
    )
    scene = prepared.scene
    sim_collision_clearance_m = float(
        scene.scene_spec.collision.safety_minimum_clearance_m
    )
    if not math.isfinite(sim_collision_clearance_m) or sim_collision_clearance_m < 0.0:
        raise ValueError(
            "RM65 simulation collision clearance must be finite and non-negative."
        )
    tool_paths = {
        "left": prepared.task_config.left_tool_profile_path,
        "right": prepared.task_config.right_tool_profile_path,
    }
    ik = RM65CanonicalIK.from_tool_profiles(
        scene,
        tool_paths,
        planner_config=RM65PositionCollisionPlannerConfig(
            # This resolver is called by the 60 Hz composed target.  Reuse the
            # validated RM65 position envelope instead of the planner's
            # conservative 0.35--0.50 rad/s real-side defaults; otherwise a
            # 1:1 B601 residual appears artificially slow in simulation.
            reference_hz=prepared.compose_target_hz,
            max_joint_velocity_rad_s=RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
            max_joint_acceleration_rad_s2=(
                RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
            ),
            solve_time_budget_ms=(
                1000.0 / float(prepared.position_profile.rates.ik_hz)
            ),
            # Use the task's MuJoCo collision contract (currently 2 mm), not
            # the 30 mm real-side planner default.  Collision stopping stays
            # enabled; only the scene-specific clearance is aligned.
            collision_min_distance_m=sim_collision_clearance_m,
            collision_distance_epsilon_m=min(
                DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
                sim_collision_clearance_m,
            ),
            collision_constraints_enabled=prepared.config.collision_stop_enabled,
        ),
    )
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=prepared.timing,
            actuation_profile=prepared.arm_profile,
            gripper_actuation_profile=prepared.gripper_profile,
            realtime_worker=prepared.config.realtime_worker,
        ),
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            producer_id="b601-readonly-human",
            initial_control_mode="human",
            decision_hz=prepared.compose_target_hz,
            pose_frame_contract_hash=prepared.tool_frame_profile.semantic_content_hash,
            enable_component_execution_handoff=True,
            prune_expired_proposals=True,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(ik),
    )
    sources = {f"{side}_arm": side for side in ("left", "right")}
    projector = build_mujoco_pose_observation_projector(
        scene,
        action_schema=runtime.action_schema,
        qpos_source_by_component=sources,
        joint_names_by_source={
            side: scene.arm_specs[side].joint_names for side in tool_paths
        },
        reference_body_by_component={
            component: f"{side}_arm_base_link" for component, side in sources.items()
        },
        endpoint_body_by_component={
            component: scene.arm_specs[side].controlled_body
            for component, side in sources.items()
        },
        endpoint_to_controlled_by_component={
            component: endpoint_to_tool_transform(
                load_rm65_tool_profile(tool_paths[side], side=side)
            )
            for component, side in sources.items()
        },
        model_contract_id=scene.conversion.scene_profile_name,
        model_contract_hash=prepared.target_frame_profile.semantic_content_hash,
    )

    def measure_target() -> ExecutionMeasurementObservation:
        # One lower-owner lock cut: never read live MjData alongside its worker.
        cut = rig.capture_execution_state_cut()
        measurement = build_mujoco_qpos_measurement_from_execution_cut(
            cut,
            clock_domain_id=runtime.config.clock_domain_id,
        )
        values = projector.project(
            model_qpos=cut.model_qpos,
            qpos_by_source=cut.measured_qpos_by_source,
            component_ids=runtime.action_schema.component_ids,
        )
        return ExecutionMeasurementObservation(
            measurement_id=f"fk-pose:{measurement.measurement_id}",
            source_measurement_hash=measurement.semantic_content_hash,
            control_domain_id=runtime.config.control_domain_id,
            measured_at=cut.sampled_at,
            available_at=runtime.clock(),
            clock_domain_id=runtime.config.clock_domain_id,
            value_by_component=values,
            representation_by_component={
                component: "cartesian_pose" for component in values
            },
            schema_hash_by_component=projector.schema_hash_by_component,
        )

    return runtime, measure_target, prepared.compose_target_hz


def run_b601_rm65_gui(
    session: B601RM65MotionSession,
    *,
    viewer,
    rate_hz: float,
    poll_command: Callable[[float], str | None] | None = None,
    auto_activate: bool = False,
    emit=print,
    record=None,
) -> None:
    """Observe the runtime-owned rig; GUI closure withdraws target following."""
    from control_station.sim_hil_profile.rebot_b601_rm65_foreground import (
        B601RM65TerminalCommands,
        run_b601_rm65_foreground,
    )

    commands = B601RM65TerminalCommands() if poll_command is None else poll_command

    if record is not None and not callable(record):
        raise TypeError("record must be callable or None.")

    def record_with_feedback(event: dict[str, Any]) -> None:
        if record is None:
            return
        enriched = dict(event)
        if event.get("event") in {
            "proposal_admission",
            "target_step",
            "bridge_step",
        }:
            snapshot = _feedback_snapshot_details(session)
            if snapshot is not None:
                enriched["feedback_snapshot"] = snapshot
        record(enriched)

    def command(timeout: float) -> str | None:
        if not viewer.sync():
            status = viewer.status()
            if status["state"] == "faulted":
                raise RuntimeError(f"MuJoCo viewer failed: {status}")
            return "q"
        return commands(timeout)

    def advance() -> dict[str, object]:
        pose_resolver = getattr(session.binding.runtime, "pose_resolver", None)
        clear_resolve_diagnostics = getattr(
            pose_resolver, "clear_last_resolve_diagnostics", None
        )
        if callable(clear_resolve_diagnostics):
            clear_resolve_diagnostics()
        target_started = time.perf_counter()
        result = session.binding.runtime.step()
        target_completed = float(session.binding.runtime.clock())
        target_details = _target_step_details(result)
        target_step_elapsed_sec = max(
            0.0, time.perf_counter() - target_started
        )
        target_details["target_step_elapsed_sec"] = target_step_elapsed_sec
        resolve_diagnostics = getattr(
            pose_resolver, "last_resolve_diagnostics", {}
        )
        if isinstance(resolve_diagnostics, Mapping):
            resolve_diagnostics = dict(resolve_diagnostics)
        else:
            resolve_diagnostics = {}
        resolution_elapsed = _finite_float_or_none(
            resolve_diagnostics.get("resolve_total_elapsed_sec")
        )
        target_details["target_resolution"] = resolve_diagnostics
        target_details["target_resolution_invoked"] = bool(
            resolve_diagnostics.get("resolve_invoked", False)
        )
        target_details["target_resolution_elapsed_sec"] = (
            resolution_elapsed
        )
        target_details["target_ik_elapsed_sec"] = resolve_diagnostics.get(
            "ik_elapsed_sec"
        )
        target_details["target_post_resolution_elapsed_sec"] = (
            max(
                0.0,
                target_step_elapsed_sec - resolution_elapsed,
            )
            if resolution_elapsed is not None
            else None
        )
        target_details["target_dispatch_accepted"] = (
            str(getattr(result, "status", "")) == "dispatched"
        )
        completion = getattr(result, "target_completion_receipt", None)
        dispatch_at = _finite_float_or_none(
            getattr(completion, "available_at", None)
        )
        if dispatch_at is None and target_details["target_dispatch_accepted"]:
            dispatch_at = target_completed
        target_details["target_dispatch_monotonic_sec"] = dispatch_at
        target_details["source_reused"] = session.last_source_reused
        record_with_feedback(
            {"event": "target_step", **target_details}
        )
        return target_details

    run_b601_rm65_foreground(
        session,
        advance_target=advance,
        rate_hz=rate_hz,
        poll_command=command,
        auto_activate=auto_activate,
        emit=emit,
        record=record_with_feedback,
    )


_B601_RS_STATE_OBSERVATION_CAPABILITY = "state_observation_metadata_v1"
# A 20 Hz active-report stream has a nominal 50 ms period.  The receive-only
# simulation tolerates one missed report plus bounded scheduling/USB jitter,
# while ``maximum_source_age_sec`` remains the outer source TTL.  This is
# deliberately local to the simulation process; the hardware owner's YAML
# timeout remains unchanged.
_B601_RM65_SIMULATION_FEEDBACK_TIMEOUT_SEC = 0.20


def _prepare_bundled_motorbridge_runtime() -> dict[str, object]:
    """Prefer the checked-out MotorBridge binding before opening CAN.

    The AgenticRobot repository pins the MotorBridge submodule, but Python's
    import resolver otherwise silently selects an older site-package.  That
    older package can expose the legacy cache while lacking the host receive
    metadata required by :class:`B601ReadOnlyPoseSource`.  Select a complete
    checked-out binding/library pair when available; an explicit
    ``MOTORBRIDGE_LIB`` remains an operator override and is reported as such.
    This helper performs no device I/O.
    """

    loaded = sys.modules.get("motorbridge")
    if loaded is not None:
        loaded_source = getattr(loaded, "__file__", None)
        return {
            "selection": "already_imported",
            "python_source": (
                None if loaded_source is None else str(loaded_source)
            ),
            "abi_path": os.environ.get("MOTORBRIDGE_LIB"),
            "reason": "motorbridge was imported before B601 application setup",
        }

    explicit_library = os.environ.get("MOTORBRIDGE_LIB")
    if explicit_library:
        return {
            "selection": "explicit_environment",
            "python_source": None,
            "abi_path": str(Path(explicit_library).expanduser()),
            "reason": "MOTORBRIDGE_LIB is an explicit operator override",
        }

    project_root = Path(__file__).resolve().parents[2]
    binding_root = project_root / "hardware" / "MotorBridge" / "bindings" / "python" / "src"
    package_init = binding_root / "motorbridge" / "__init__.py"
    abi_path = project_root / "hardware" / "MotorBridge" / "target" / "release" / "libmotor_abi.so"
    if package_init.is_file() and abi_path.is_file():
        binding_root_text = str(binding_root)
        if binding_root_text not in sys.path:
            sys.path.insert(0, binding_root_text)
        return {
            "selection": "project_pinned_binding",
            "python_source": str(package_init),
            "abi_path": str(abi_path),
            "reason": "complete checked-out MotorBridge binding and ABI found",
        }

    return {
        "selection": "installed_environment",
        "python_source": None,
        "abi_path": None,
        "reason": "checked-out MotorBridge binding/ABI is unavailable",
        "expected_binding_root": str(binding_root),
        "expected_abi_path": str(abi_path),
    }


def _record_motorbridge_capability(
    result: dict[str, object],
    backend: object,
    *,
    emit: Callable[[str], None],
) -> None:
    """Record capability admission and keep missing metadata retryable.

    Read-only RM65 following can stay open while the operator repairs the
    MotorBridge runtime or telemetry stream.  The source still refuses every
    observation until host receive metadata is advertised; this function only
    makes that state explicit and actionable.
    """

    evidence = getattr(backend, "motorbridge_runtime_evidence", None)
    if not isinstance(evidence, Mapping):
        return
    evidence_dict = dict(evidence)
    contract = result["teleop_contract"]
    assert isinstance(contract, dict)
    contract["motorbridge_runtime_evidence"] = evidence_dict
    contract["required_feedback_capability"] = _B601_RS_STATE_OBSERVATION_CAPABILITY
    features = evidence_dict.get("robstride_features", ())
    available = isinstance(features, (tuple, list, set, frozenset)) and (
        _B601_RS_STATE_OBSERVATION_CAPABILITY in features
    )
    contract["state_observation_metadata_available"] = available
    contract["control_admission"] = (
        "eligible_pending_activation" if available else "blocked_missing_feedback_capability"
    )
    if not available:
        detail = evidence_dict.get("error") or "capability was not advertised"
        emit(
            "[B601→RM65] waiting: host feedback capability "
            f"{_B601_RS_STATE_OBSERVATION_CAPABILITY!r} is unavailable ({detail})."
        )
        emit(
            "[B601→RM65] receive-only GUI remains open; repair the pinned "
            "MotorBridge runtime/stream, then press a to retry."
        )


def _load_mapping(path: Path):
    try:
        from control_station.composition.rebot_b601_rm65_profile import (
            load_b601_rm65_frame_mapping_profile,
        )
    except ModuleNotFoundError as error:
        if error.name != "control_station.composition.rebot_b601_rm65_profile":
            raise
        raise RuntimeError(
            "This checkout lacks the canonical B601/RM65 frame-mapping profile; "
            "use an integrator-composed checkout containing the profile and motion predecessor."
        ) from error
    return load_b601_rm65_frame_mapping_profile(path)


def _positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _finite_rotation_values(
    values: object | None,
    *,
    count: int,
    name: str,
) -> tuple[float, ...] | None:
    """Normalize a CLI rotation override without weakening frame validation."""

    if values is None:
        return None
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if len(normalized) != count or any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain {count} finite values.")
    return normalized


def _rotation_matrix_from_rpy_degrees(values: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    """Build a row-major ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` matrix."""

    roll, pitch, yaw = (math.radians(value) for value in values)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rz = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))

    def multiply(left, right):
        return tuple(
            tuple(
                sum(left[row][index] * right[index][column] for index in range(3))
                for column in range(3)
            )
            for row in range(3)
        )

    matrix = multiply(multiply(rz, ry), rx)
    return tuple(
        tuple(0.0 if abs(value) < 1e-12 else float(value) for value in row)
        for row in matrix
    )


def _mapping_rotation_override(
    mapping: object,
    *,
    reference_rpy_deg: object | None,
    reference_matrix: object | None,
    controlled_rpy_deg: object | None,
    controlled_matrix: object | None,
) -> tuple[object, dict[str, object]]:
    """Apply foreground frame tuning parameters to one loaded profile."""

    reference_rpy = _finite_rotation_values(
        reference_rpy_deg,
        count=3,
        name="--mapping-reference-rotation-rpy-deg",
    )
    controlled_rpy = _finite_rotation_values(
        controlled_rpy_deg,
        count=3,
        name="--mapping-controlled-rotation-rpy-deg",
    )
    reference_values = _finite_rotation_values(
        reference_matrix,
        count=9,
        name="--mapping-reference-rotation-matrix",
    )
    controlled_values = _finite_rotation_values(
        controlled_matrix,
        count=9,
        name="--mapping-controlled-rotation-matrix",
    )
    reference_override = (
        _rotation_matrix_from_rpy_degrees(reference_rpy)
        if reference_rpy is not None
        else (
            tuple(
                tuple(reference_values[row * 3 : row * 3 + 3])
                for row in range(3)
            )
            if reference_values is not None
            else None
        )
    )
    controlled_override = (
        _rotation_matrix_from_rpy_degrees(controlled_rpy)
        if controlled_rpy is not None
        else (
            tuple(
                tuple(controlled_values[row * 3 : row * 3 + 3])
                for row in range(3)
            )
            if controlled_values is not None
            else None
        )
    )
    if reference_override is None and controlled_override is None:
        return mapping, {}
    apply_override = getattr(mapping, "with_rotation_overrides", None)
    if not callable(apply_override):
        raise TypeError(
            "mapping profile does not support foreground rotation overrides."
        )
    tuned = apply_override(
        reference_rotation_matrix=reference_override,
        controlled_rotation_matrix=controlled_override,
    )
    metadata: dict[str, object] = {}
    if reference_rpy is not None:
        metadata["reference_rotation_rpy_deg"] = list(reference_rpy)
    elif reference_values is not None:
        metadata["reference_rotation_matrix"] = list(reference_values)
    if controlled_rpy is not None:
        metadata["controlled_rotation_rpy_deg"] = list(controlled_rpy)
    elif controlled_values is not None:
        metadata["controlled_rotation_matrix"] = list(controlled_values)
    return tuned, metadata


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _mapping_contract_metadata(mapping: object, path: Path) -> dict[str, object]:
    """Expose the loaded mapping identity and gains in the run receipt."""

    metadata: dict[str, object] = {"mapping_profile": str(path)}
    profile_hash = getattr(mapping, "semantic_content_hash", None)
    if profile_hash is not None:
        metadata["mapping_profile_semantic_content_hash"] = str(profile_hash)
    gains = getattr(mapping, "translation_gain_target_reference_xyz", None)
    if gains is not None:
        try:
            normalized_gains = [float(value) for value in gains]
        except (TypeError, ValueError, OverflowError):
            normalized_gains = []
        if len(normalized_gains) == 3 and all(
            math.isfinite(value) for value in normalized_gains
        ):
            metadata["translation_gain_target_reference_xyz"] = normalized_gains
    rotation_gain = _finite_float_or_none(getattr(mapping, "rotation_gain", None))
    if rotation_gain is not None:
        metadata["rotation_gain"] = rotation_gain
    for field_name, output_name in (
        (
            "matrix_target_reference_source_reference",
            "effective_reference_rotation_matrix",
        ),
        (
            "matrix_source_controlled_target_controlled",
            "effective_controlled_rotation_matrix",
        ),
    ):
        matrix = getattr(mapping, field_name, None)
        try:
            rotation = [
                [float(matrix[row][column]) for column in range(3)]
                for row in range(3)
            ]
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            continue
        if all(math.isfinite(value) for row in rotation for value in row):
            metadata[output_name] = rotation
    return metadata


def _feedback_snapshot_details(session: B601RM65MotionSession) -> dict[str, object] | None:
    """Return host-cache timing facts without creating a second observation path."""

    binding = getattr(session, "binding", None)
    source = getattr(binding, "source", None)
    pose_source = getattr(source, "pose_source", None)
    backend = getattr(pose_source, "backend", None)
    state = getattr(backend, "last_state", None)
    if state is None:
        return None

    ages: dict[str, float] = {}
    status_sequences: dict[str, int] = {}
    try:
        joints = tuple(getattr(state, "joints", ()) or ())
    except TypeError:
        joints = ()
    for joint in joints:
        name = str(getattr(joint, "name", "unknown"))
        age = _finite_float_or_none(getattr(joint, "receive_age_sec", None))
        if age is not None:
            ages[name] = age
        sequence = getattr(joint, "status_sequence", None)
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            status_sequences[name] = sequence

    maximum_age = max(ages.values()) if ages else None
    backend_timeout = _finite_float_or_none(
        getattr(getattr(backend, "config", None), "feedback_timeout_sec", None)
    )
    source_limit = None
    try:
        source_limit = _finite_float_or_none(binding.limits[0])
    except (AttributeError, IndexError, TypeError):
        pass
    details: dict[str, object] = {
        "sequence": getattr(state, "sequence", None),
        "receive_timestamp_sec": _finite_float_or_none(
            getattr(state, "receive_timestamp_sec", None)
        ),
        "valid": bool(getattr(state, "valid", False)),
        "stale": bool(getattr(state, "stale", False)),
        "host_receive_freshness_verified": bool(
            getattr(state, "host_receive_freshness_verified", False)
        ),
        "freshness_semantics": str(
            getattr(state, "freshness_semantics", "unknown")
        ),
        "fault_reason": getattr(state, "fault_reason", None),
        "joint_receive_age_sec": ages,
        "joint_status_sequence": status_sequences,
        "max_receive_age_sec": maximum_age,
        "feedback_timeout_sec": backend_timeout,
        "source_age_limit_sec": source_limit,
    }
    if maximum_age is not None and backend_timeout is not None:
        details["freshness_margin_sec"] = backend_timeout - maximum_age
    return details


def _target_step_details(step: object) -> dict[str, object]:
    """Project bounded target-owner diagnostics into the application log.

    The target remains authoritative for IK, collision, and lifecycle policy;
    this helper only makes an existing typed result inspectable when a long
    foreground run stops changing qpos.  It deliberately copies scalar
    resolution/cancellation fields instead of serializing private planner
    state or changing the runtime decision.
    """

    details: dict[str, object] = {
        "status": str(getattr(step, "status", "unknown")),
    }
    failure = getattr(step, "resolution_failure", None)
    if failure is not None:
        diagnostics = getattr(failure, "diagnostics", {})
        details["resolution_failure"] = {
            "reason_code": str(getattr(failure, "reason_code", "")),
            "resolver_status": str(getattr(failure, "resolver_status", "")),
            "message": str(getattr(failure, "message", "")),
            "diagnostics": (
                dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
            ),
        }
    cancellation = getattr(step, "cancellation_receipt", None)
    if cancellation is not None:
        details["cancellation"] = {
            "complete": bool(getattr(cancellation, "complete", False)),
            "status": str(getattr(cancellation, "status", "")),
        }
    completion = getattr(step, "target_completion_receipt", None)
    if completion is not None:
        details["target_completion"] = {
            "status": str(getattr(completion, "status", "")),
            "outcome_code": str(getattr(completion, "outcome_code", "")),
        }
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="load contracts/models; never open CAN or GUI",
    )
    mode.add_argument(
        "--connect-read-only",
        action="store_true",
        help="receive existing telemetry; never change real motor/report state",
    )
    parser.add_argument("--confirmation")
    for name in (
        "task-config",
        "mapping-profile",
        "config",
        "rebot-source-root",
        "log",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--maximum-source-age-sec", type=_positive, default=0.15)
    parser.add_argument("--maximum-measurement-age-sec", type=_positive, default=0.1)
    parser.add_argument("--maximum-translation-m", type=_positive, default=0.1)
    parser.add_argument("--maximum-rotation-rad", type=_positive, default=0.3)
    parser.add_argument(
        "--disable-teleop-residual-envelope",
        action="store_true",
        help=(
            "read-only simulation only: disable the application residual "
            "translation/rotation gate; canonical target safety remains active"
        ),
    )
    parser.add_argument(
        "--simulation-feedback-timeout-sec",
        type=_positive,
        default=None,
        help=(
            "read-only simulation only: override this process' host-cache "
            "freshness timeout; does not change the B601 owner"
        ),
    )
    parser.add_argument(
        "--auto-activate",
        action="store_true",
        help=(
            "read-only simulation only: capture one fresh B601 pose and start "
            "RM65 following without waiting for the a key"
        ),
    )
    reference_rotation = parser.add_mutually_exclusive_group()
    reference_rotation.add_argument(
        "--mapping-reference-rotation-rpy-deg",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "foreground override for the base-frame rotation, in degrees; "
            "composition is Rz(yaw) @ Ry(pitch) @ Rx(roll)"
        ),
    )
    reference_rotation.add_argument(
        "--mapping-reference-rotation-matrix",
        nargs=9,
        type=float,
        metavar="R",
        help="foreground override for the base-frame 3x3 rotation, row-major",
    )
    controlled_rotation = parser.add_mutually_exclusive_group()
    controlled_rotation.add_argument(
        "--mapping-controlled-rotation-rpy-deg",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "foreground override for the tool-frame rotation, in degrees; "
            "composition is Rz(yaw) @ Ry(pitch) @ Rx(roll)"
        ),
    )
    controlled_rotation.add_argument(
        "--mapping-controlled-rotation-matrix",
        nargs=9,
        type=float,
        metavar="R",
        help="foreground override for the tool-frame 3x3 rotation, row-major",
    )
    args = parser.parse_args(argv)
    try:
        _finite_rotation_values(
            args.mapping_reference_rotation_rpy_deg,
            count=3,
            name="--mapping-reference-rotation-rpy-deg",
        )
        _finite_rotation_values(
            args.mapping_reference_rotation_matrix,
            count=9,
            name="--mapping-reference-rotation-matrix",
        )
        _finite_rotation_values(
            args.mapping_controlled_rotation_rpy_deg,
            count=3,
            name="--mapping-controlled-rotation-rpy-deg",
        )
        _finite_rotation_values(
            args.mapping_controlled_rotation_matrix,
            count=9,
            name="--mapping-controlled-rotation-matrix",
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        args.connect_read_only
        and args.confirmation != "B601_RM65_READ_ONLY_CAN_SIM_TARGET"
    ):
        parser.error(
            "read-only CAN connection requires --confirmation B601_RM65_READ_ONLY_CAN_SIM_TARGET"
        )
    if args.disable_teleop_residual_envelope and not args.connect_read_only:
        parser.error(
            "--disable-teleop-residual-envelope is only valid with --connect-read-only"
        )
    if args.auto_activate and not args.connect_read_only:
        parser.error("--auto-activate is only valid with --connect-read-only")
    if args.simulation_feedback_timeout_sec is not None:
        if not args.connect_read_only:
            parser.error(
                "--simulation-feedback-timeout-sec is only valid with --connect-read-only"
            )
        if args.simulation_feedback_timeout_sec > args.maximum_source_age_sec:
            parser.error(
                "--simulation-feedback-timeout-sec must be no greater than "
                "--maximum-source-age-sec"
            )
    values = list(sys.argv[1:] if argv is None else argv)
    # Reserve a new result before any device call; logs may append across runs.
    with (
        args.output.open("x", encoding="utf-8") as output,
        args.log.open("a", encoding="utf-8") as log,
    ):

        def record(event) -> None:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()

        stdout_broken = False

        def emit(message: str) -> None:
            nonlocal stdout_broken
            if not stdout_broken:
                try:
                    print(message, flush=True)
                except BrokenPipeError:
                    # A caller may intentionally pipe the concise terminal
                    # stream through ``head``/``sed``.  The detailed JSON/log
                    # evidence remains authoritative and must still be closed
                    # and written after that presentation pipe disappears.
                    stdout_broken = True
                    try:
                        null_fd = os.open(os.devnull, os.O_WRONLY)
                    except OSError:
                        pass
                    else:
                        try:
                            os.dup2(null_fd, sys.stdout.fileno())
                        except (OSError, ValueError):
                            pass
                        finally:
                            os.close(null_fd)
            record({"event": "terminal", "message": message})

        result = {
            "status": "ERROR",
            "real_motor_commands_sent": False,
            "target_cleanup_completed": False,
            "teleop_contract": {
                "input_mode": "b601_read_only_to_rm65_mujoco",
                "rl_action_delta_scaling": "not_applied",
                "canonical_target_safety": "retained",
                "application_residual_envelope": (
                    "disabled_for_read_only_simulation"
                    if args.disable_teleop_residual_envelope
                    else "bounded"
                ),
                "residual_translation_limit_m": (
                    None
                    if args.disable_teleop_residual_envelope
                    else float(args.maximum_translation_m)
                ),
                "residual_rotation_limit_rad": (
                    None
                    if args.disable_teleop_residual_envelope
                    else float(args.maximum_rotation_rad)
                ),
                "source_age_limit_sec": float(args.maximum_source_age_sec),
                "measurement_age_limit_sec": float(
                    args.maximum_measurement_age_sec
                ),
                "simulation_feedback_timeout_override_sec": (
                    None
                    if args.simulation_feedback_timeout_sec is None
                    else float(args.simulation_feedback_timeout_sec)
                ),
                "activation_mode": (
                    "auto_on_start" if args.auto_activate else "manual_key"
                ),
                "control_admission": "not_checked",
            },
        }
        status = 1
        record(
            {
                "event": "command",
                "argv": [sys.executable, "-m", "control_station", "b601-rm65", *values],
            }
        )
        try:
            if args.connect_read_only:
                result["teleop_contract"]["motorbridge_runtime_selection"] = (
                    _prepare_bundled_motorbridge_runtime()
                )
            from hardware.rebot_b601_rs import B601RSBackend, B601RSBackendConfig
            from robot_motion.rebot_b601_rs import B601PinocchioModel
            from sim.mujoco.viewer import MujocoPassiveViewerSession

            mapping = _load_mapping(args.mapping_profile)
            mapping, rotation_overrides = _mapping_rotation_override(
                mapping,
                reference_rpy_deg=args.mapping_reference_rotation_rpy_deg,
                reference_matrix=args.mapping_reference_rotation_matrix,
                controlled_rpy_deg=args.mapping_controlled_rotation_rpy_deg,
                controlled_matrix=args.mapping_controlled_rotation_matrix,
            )
            result["teleop_contract"].update(
                _mapping_contract_metadata(mapping, args.mapping_profile)
            )
            if rotation_overrides:
                result["teleop_contract"]["mapping_rotation_overrides"] = (
                    rotation_overrides
                )
            model = B601PinocchioModel.from_rebot_source_root(args.rebot_source_root)
            config = B601RSBackendConfig.from_rs_yaml(args.config)
            simulation_feedback_timeout = args.simulation_feedback_timeout_sec
            if (
                args.connect_read_only
                and simulation_feedback_timeout is None
                and isinstance(config, B601RSBackendConfig)
            ):
                # The wired owner has a 100 ms default, but the receive-only
                # simulation polls a roughly 20 Hz stream at 60 Hz.  Give the
                # host cache one bounded extra frame of tolerance while still
                # staying inside the application source-age contract.  This
                # replacement is local to the simulation process and never
                # mutates the operator's YAML or the B601 owner.
                simulation_feedback_timeout = min(
                    _B601_RM65_SIMULATION_FEEDBACK_TIMEOUT_SEC,
                    float(args.maximum_source_age_sec),
                )
            if simulation_feedback_timeout is not None:
                config = replace(
                    config,
                    feedback_timeout_sec=simulation_feedback_timeout,
                )
                result["teleop_contract"]["simulation_feedback_timeout_source"] = (
                    "simulation_cli_override"
                    if args.simulation_feedback_timeout_sec is not None
                    else "simulation_default"
                )
            result["teleop_contract"].update(
                {
                    "backend_feedback_timeout_sec": _finite_float_or_none(
                        getattr(config, "feedback_timeout_sec", None)
                    ),
                    "backend_feedback_timeout_source": (
                        "simulation_cli_override"
                        if args.simulation_feedback_timeout_sec is not None
                        else "simulation_default"
                        if simulation_feedback_timeout is not None
                        else "backend_config"
                    ),
                }
            )
            backend = B601RSBackend(config)
            with ExitStack() as resources:
                runtime, measure, rate = build_b601_rm65_sim_target(args.task_config)
                result["teleop_contract"]["target_control_hz"] = float(rate)

                def close_target() -> None:
                    runtime.close()
                    result["target_cleanup_completed"] = True

                resources.callback(close_target)
                session = assemble_b601_rm65_session(
                    runtime,
                    backend=backend,
                    model=model,
                    mapping_profile=mapping,
                    measure_target=measure,
                    maximum_source_age_sec=args.maximum_source_age_sec,
                    maximum_measurement_age_sec=args.maximum_measurement_age_sec,
                    maximum_translation_m=(
                        None
                        if args.disable_teleop_residual_envelope
                        else args.maximum_translation_m
                    ),
                    maximum_rotation_rad=(
                        None
                        if args.disable_teleop_residual_envelope
                        else args.maximum_rotation_rad
                    ),
                )
                if not args.validate_only:
                    emit(
                        "[B601→RM65] RECEIVE ONLY: master gravity/damping and active telemetry must already have an independent owner. No real motor commands."
                    )
                    resources.callback(backend.close)
                    backend.connect()
                    _record_motorbridge_capability(result, backend, emit=emit)
                    viewer = MujocoPassiveViewerSession(
                        runtime.rig.scene, sync_callback=runtime.rig.sync_viewer
                    )
                    resources.callback(viewer.close)
                    # Launch before starting the sole physics worker.
                    viewer.start()
                    runtime.start()
                    run_b601_rm65_gui(
                        session,
                        viewer=viewer,
                        rate_hz=rate,
                        auto_activate=args.auto_activate,
                        emit=emit,
                        record=record,
                    )
                result["status"] = "VALIDATED" if args.validate_only else "COMPLETED"
                status = 0
        except KeyboardInterrupt:
            result["status"] = "INTERRUPTED"
            status = 130
            emit(
                "[B601→RM65] Interrupted; inspect target cleanup in JSON. Master gravity/damping is unchanged."
            )
        except Exception as error:  # noqa: BLE001 - CLI boundary records the complete failure and returns nonzero.
            result["error"] = f"{type(error).__name__}: {error}"
            record({"event": "error", "traceback": traceback.format_exc()})
            emit(f"[B601→RM65] ERROR: {result['error']}")
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        record({"event": "result", **result, "exit_code": status})
    return status
