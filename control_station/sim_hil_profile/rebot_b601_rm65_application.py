"""Explicit read-only CAN source and single-human RM65 simulation assembly."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
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
    from robot_motion.rm65 import (
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
        )
    )
    scene = prepared.scene
    tool_paths = {
        "left": prepared.task_config.left_tool_profile_path,
        "right": prepared.task_config.right_tool_profile_path,
    }
    ik = RM65CanonicalIK.from_tool_profiles(
        scene,
        tool_paths,
        planner_config=RM65PositionCollisionPlannerConfig(
            reference_hz=prepared.position_profile.rates.ik_hz,
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
        if event.get("event") in {"proposal_admission", "target_step"}:
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

    def advance() -> None:
        result = session.binding.runtime.step()
        record_with_feedback({"event": "target_step", "status": result.status})

    run_b601_rm65_foreground(
        session,
        advance_target=advance,
        rate_hz=rate_hz,
        poll_command=command,
        emit=emit,
        record=record_with_feedback,
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
    args = parser.parse_args(argv)
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

        def emit(message: str) -> None:
            print(message, flush=True)
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
            from hardware.rebot_b601_rs import B601RSBackend, B601RSBackendConfig
            from robot_motion.rebot_b601_rs import B601PinocchioModel
            from sim.mujoco.viewer import MujocoPassiveViewerSession

            mapping = _load_mapping(args.mapping_profile)
            result["teleop_contract"].update(
                _mapping_contract_metadata(mapping, args.mapping_profile)
            )
            model = B601PinocchioModel.from_rebot_source_root(args.rebot_source_root)
            config = B601RSBackendConfig.from_rs_yaml(args.config)
            if args.simulation_feedback_timeout_sec is not None:
                config = replace(
                    config,
                    feedback_timeout_sec=args.simulation_feedback_timeout_sec,
                )
            result["teleop_contract"].update(
                {
                    "backend_feedback_timeout_sec": _finite_float_or_none(
                        getattr(config, "feedback_timeout_sec", None)
                    ),
                    "backend_feedback_timeout_source": (
                        "simulation_cli_override"
                        if args.simulation_feedback_timeout_sec is not None
                        else "backend_config"
                    ),
                }
            )
            backend = B601RSBackend(config)
            with ExitStack() as resources:
                runtime, measure, rate = build_b601_rm65_sim_target(args.task_config)

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
                    viewer = MujocoPassiveViewerSession(
                        runtime.rig.scene, sync_callback=runtime.rig.sync_viewer
                    )
                    resources.callback(viewer.close)
                    # Launch before starting the sole physics worker.
                    viewer.start()
                    runtime.start()
                    run_b601_rm65_gui(
                        session, viewer=viewer, rate_hz=rate, emit=emit, record=record
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
