"""Control Station application root for the early Sim-HIL profile.

The profile is deliberately narrow and default-off:

* the temporary MuJoCo PPO checkpoint contributes ``right_arm`` pose plus
  ``right_hand`` qpos from its task-local gripper delta;
* real VIVE and Q6 are constructed through the passive teleop builder and can
  only produce a human emission;
* the existing formal composition and one ``MujocoDeployRuntime`` own
  authority, safety, coupled IK, reference scheduling, and MuJoCo dispatch;
* no real RM65 arm backend is constructed or enabled.

This module is an application leaf.  Model inference, device lifecycle,
composition algebra, safety, IK, and execution remain in their lower-owned
packages.  Concrete imports are lazy so invoking the default-off CLI does not
open a socket, load a device, or initialize MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE_NAME,
)
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    ArtifactWriteResolution,
    RuntimeArtifactLayout,
    configured_runtime_artifact_layout,
    resolve_artifact_write_path,
)

from .sim_hil_profile.application import (
    MujocoSimHILApplication,
    _advance_control_deadline,
)
from .sim_hil_profile.config import (
    DEFAULT_CALIBRATION_PROFILE,
    DEFAULT_CHECKPOINT,
    DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ,
    DEFAULT_STATUS_HOST,
    DEFAULT_STATUS_PORT,
    DEFAULT_TASK_CONFIG,
    DEFAULT_VIVE_COORDINATE_PROFILE,
    DEFAULT_VIVE_PORT,
    MujocoSimHILApplicationConfig,
    SIM_HIL_BLEND_FAILURE_FALLBACKS,
    SIM_HIL_HUMAN_TIMEOUT_FALLBACKS,
    acceptance_profile_config as _acceptance_profile_config,
)
from .sim_hil_profile.factory import (
    _SimHILReadinessProvider,
    _bind_sim_target_session_anchors,
    build_application_graph as _build_application_graph,
)
from .sim_hil_profile.human import (
    _SimHILHumanAttachment,
    _bind_vive_coordinate_profile,
    _build_passive_single_vive_gripper_component_source,
    _build_passive_vive_q6_component_source,
    _build_single_vive_gripper_component_source_spec,
    _file_sha256,
)
from .sim_hil_profile.operator_view import (
    build_sim_hil_operator_guidance,
)
from .sim_hil_profile.policy_human_allocation import (
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
    SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS,
)
from .sim_hil_profile.profile_selection import (
    SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
    require_sim_hil_control_profile_available,
)
from .sim_hil_profile.status import (
    SIM_HIL_STATUS_SCHEMA_V1,
    _composition_step_status,
    _emission_status,
    _execution_difference_status,
    _human_channel_status,
    _human_gripper_target_status,
    _input_submission_status,
    _mix_status,
    _mujoco_gripper_execution_status,
    _runtime_contract_status,
    _target_execution_status,
    _vive_receiver_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_HIL_ACCEPTANCE_STORAGE_SCHEMA = (
    "dexteleop.control_station.sim_hil_acceptance_storage.v1"
)


def legacy_acceptance_root() -> Path:
    return REPO_ROOT / "teleop_records" / "sim_hil_acceptance"


def canonical_acceptance_root() -> Path:
    return (
        RuntimeArtifactLayout.configured().evidence_dir(kind="sim_hil")
        / "acceptance"
    )


def resolve_acceptance_root(args: argparse.Namespace) -> ArtifactWriteResolution:
    return resolve_artifact_write_path(
        explicit_path=getattr(args, "acceptance_root", None),
        artifact_layout=getattr(args, "artifact_layout", None),
        legacy_path=legacy_acceptance_root(),
        canonical_path=canonical_acceptance_root(),
    )


def _write_acceptance_storage_receipt(
    *,
    session_root: Path,
    resolution: ArtifactWriteResolution,
    requested_layout: str,
) -> Path:
    """Durably bind one ControlTrace session to its selected storage root."""

    session_root = Path(session_root).expanduser().resolve()
    receipt_path = session_root / "artifact_storage.json"
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    payload = {
        "schema": SIM_HIL_ACCEPTANCE_STORAGE_SCHEMA,
        "created_wall_time": time.time(),
        "requested_layout": str(requested_layout),
        "selection_source": resolution.source,
        "resolved_layout": resolution.layout_name,
        "resolved_acceptance_root": str(resolution.path.expanduser().resolve()),
        "trace_session_root": str(session_root),
        "evidence_semantics": "control_trace_wal_seal_commit",
    }
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt_path)
    directory_fd = os.open(session_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt_path


def _build_application(
    config: MujocoSimHILApplicationConfig,
    *,
    trace_sink: Any | None = None,
) -> MujocoSimHILApplication:
    """Compatibility wrapper around the concrete graph factory.

    Passing the facade-level source builder preserves existing test/profile
    injection without making the factory depend back on this CLI module.
    """

    return _build_application_graph(
        config,
        trace_sink=trace_sink,
        human_source_builder=(
            _build_passive_single_vive_gripper_component_source
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m control_station sim-hil",
        description=(
            "Default-off MuJoCo-only Sim-HIL using a policy source with "
            "optional passive VIVE and gripper input."
        ),
    )
    parser.add_argument("--enable", action="store_true", help="Explicitly enable source startup and MuJoCo execution.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate PPO/task/calibration bindings and simulator task "
            "initialization without starting devices, authority, or workers."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        default=DEFAULT_CALIBRATION_PROFILE,
        help="Compatibility profile supplying the stored human Home only.",
    )
    parser.add_argument(
        "--vive-coordinate-profile",
        type=Path,
        default=DEFAULT_VIVE_COORDINATE_PROFILE,
        help=(
            "Independent Control Station VIVE coordinate profile. "
            "The default 'latest' selects the newest valid record from the "
            "independent Station profile directory. Target geometry and robot "
            "Home come from the loaded MuJoCo target."
        ),
    )
    parser.add_argument(
        "--vive-calibration-gui",
        action="store_true",
        help=(
            "Compatibility flag. VIVE readiness and on-demand calibration "
            "are now always integrated into --control-gui; no separate "
            "startup calibration page is opened."
        ),
    )
    parser.add_argument(
        "--vive-calibration-ui-port",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option; integrated calibration uses "
            "the --control-gui port."
        ),
    )
    parser.add_argument(
        "--vive-calibration-open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated compatibility option; integrated calibration does "
            "not open a second browser page."
        ),
    )
    parser.add_argument("--vive-port", type=int, default=DEFAULT_VIVE_PORT)
    parser.add_argument("--vive-clock-sync-port", type=int, default=None)
    parser.add_argument(
        "--left-q6-port",
        default=None,
        help="Reserved for another topology profile; rejected by this right-only profile.",
    )
    parser.add_argument(
        "--right-q6-port",
        default=None,
        help=(
            "Optional preferred right Q6 endpoint. In the default auto mode, "
            "the latest assignment/discovery is used when omitted."
        ),
    )
    parser.add_argument(
        "--gripper-input-source",
        choices=("auto", "q6", "angle_trigger"),
        default="auto",
        help=(
            "Gripper input selection. auto prefers a reachable Q6 and falls "
            "back to the calibrated angle trigger."
        ),
    )
    parser.add_argument(
        "--q6-config",
        default="latest",
        help="Saved Q6 side assignment used before unassigned discovery.",
    )
    parser.add_argument(
        "--angle-trigger-config-root",
        type=Path,
        default=None,
        help="Optional calibrated angle-trigger configuration directory.",
    )
    parser.add_argument(
        "--angle-trigger-rate-hz",
        type=float,
        default=30.0,
        help="Angle-trigger polling rate used after explicit selection or fallback.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--control-profile",
        choices=(
            SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
            SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
            SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
        ),
        default=SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
        help=(
            "Composition control profile. The established fixed-alpha path "
            "remains the default; dynamic allocation profiles are opt-in and "
            "can start from policy baseline before Human devices are ready."
        ),
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run seconds; 0 means until interrupted.")
    parser.add_argument(
        "--position-control-profile",
        type=str,
        default=POSITION_15_60_120_PROFILE_NAME,
        help=(
            "Lower position-controller profile. The default keeps the policy "
            "source at 15 Hz, reference at 60 Hz, and commands at 120 Hz; "
            "the independent composition rate defaults to 30 Hz."
        ),
    )
    parser.add_argument(
        "--compose-hz",
        type=float,
        default=None,
        help=(
            "Final HIL compose/authority target rate (default: 30 Hz). "
            "Policy inference keeps its own cadence."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=(
            "Compatibility alias for --compose-hz and the Station loop rate; "
            "when omitted, composition uses its 30 Hz default."
        ),
    )
    parser.add_argument("--warmup-timeout", type=float, default=10.0)
    parser.add_argument(
        "--human-emission-timeout",
        type=float,
        default=5.0,
        help=(
            "Maximum Sim-HIL human emission age tolerated across a VIVE "
            "network gap (default: 5 seconds). This does not change real "
            "teleop tracker-loss HOLD thresholds."
        ),
    )
    parser.add_argument(
        "--human-timeout-fallback",
        choices=("hold", "policy"),
        default="hold",
        help=(
            "Sim-HIL fallback after the human emission window expires: "
            "hold (default) or policy. Policy fallback is latched until "
            "the operator selects a human mode again."
        ),
    )
    parser.add_argument(
        "--blend-failure-grace",
        type=float,
        default=0.0,
        help=(
            "Seconds of consecutive proportional-blend resolution failure "
            "before switching to the selected human/policy fallback. The "
            "failed target is still held immediately; hold fallback ignores "
            "this delay (default: 0)."
        ),
    )
    parser.add_argument(
        "--human-alpha",
        type=float,
        default=0.5,
        help=(
            "Fixed-alpha profile only: initial human contribution in [0,1]. "
            "Allocation profiles ignore this compatibility field and use typed "
            "allocation operations."
        ),
    )
    parser.add_argument(
        "--execution-difference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request passive right-arm MuJoCo reference/measurement "
            "difference observations (default: enabled)."
        ),
    )
    parser.add_argument(
        "--mujoco-viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Open the lower-owned passive MuJoCo viewer. Defaults to enabled "
            "and is required for --control-gui; disabled by default for "
            "non-acceptance runs."
        ),
    )
    parser.add_argument(
        "--target-admission-margin-sec",
        type=float,
        default=0.05,
        help=(
            "Compatibility field reserved for scheduled-target profiles; "
            "the current immediate-qref Sim-HIL profile does not use it "
            "(default: 0.05 seconds)."
        ),
    )
    worker = parser.add_mutually_exclusive_group()
    worker.add_argument("--realtime-worker", dest="realtime_worker", action="store_true", default=True)
    worker.add_argument("--no-realtime-worker", dest="realtime_worker", action="store_false")
    parser.add_argument("--gui", action="store_true", help="Start the read-only status page.")
    parser.add_argument(
        "--control-gui",
        "--acceptance-gui",
        dest="acceptance_gui",
        action="store_true",
        help=(
            "Enable the loopback typed-intent GUI for formal Sim-HIL use. "
            "It shows fixed-alpha modes or profile-selected allocation operations "
            "according to --control-profile; allocation profiles attach Human "
            "devices only when a Human operation first needs them. Requires "
            "--enable and writes "
            "ControlTrace. "
            "--acceptance-gui is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        default=None,
        help=(
            "Exact root for acceptance ControlTrace WAL, seal, and commit evidence. "
            "When provided it overrides --artifact-layout."
        ),
    )
    parser.add_argument(
        "--artifact-layout",
        choices=(
            LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
        default=configured_runtime_artifact_layout(),
        help=(
            "Select the managed acceptance evidence root when --acceptance-root "
            "is omitted. The repository default is legacy; local_paths.json can "
            "coordinate a project-wide canonical cutover."
        ),
    )
    parser.add_argument(
        "--acceptance-device-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a valid passive VIVE/gripper emission.",
    )
    parser.add_argument(
        "--gui-host",
        default=DEFAULT_STATUS_HOST,
        help="Local-only status GUI host (127.0.0.1 or localhost).",
    )
    parser.add_argument("--gui-port", type=int, default=DEFAULT_STATUS_PORT)
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open the local Sim-HIL webpage automatically after the server "
            "starts (default: enabled; use --no-open-browser for headless use)."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> MujocoSimHILApplicationConfig:
    compose_target_hz = (
        args.compose_hz
        if args.compose_hz is not None
        else (
            None
            if args.rate is not None
            else DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ
        )
    )
    return MujocoSimHILApplicationConfig(
        checkpoint_path=args.checkpoint,
        task_config_path=args.task_config,
        calibration_profile_path=args.calibration_profile,
        vive_coordinate_profile_path=args.vive_coordinate_profile,
        vive_port=args.vive_port,
        vive_clock_sync_port=args.vive_clock_sync_port,
        left_q6_port=args.left_q6_port,
        right_q6_port=args.right_q6_port,
        gripper_input_source=args.gripper_input_source,
        q6_config=args.q6_config,
        angle_trigger_config_root=args.angle_trigger_config_root,
        angle_trigger_rate_hz=args.angle_trigger_rate_hz,
        device=args.device,
        enabled=bool(args.enable),
        control_profile_id=args.control_profile,
        realtime_worker=bool(args.realtime_worker),
        position_control_profile_name=args.position_control_profile,
        compose_target_hz=compose_target_hz,
        control_rate_hz=args.rate,
        warmup_timeout_sec=args.warmup_timeout,
        human_emission_timeout_sec=args.human_emission_timeout,
        target_admission_margin_sec=args.target_admission_margin_sec,
        initial_human_alpha=args.human_alpha,
        blend_failure_grace_sec=args.blend_failure_grace,
        human_timeout_fallback=args.human_timeout_fallback,
        execution_difference_enabled=bool(args.execution_difference),
        mujoco_viewer_enabled=(
            bool(args.acceptance_gui)
            if args.mujoco_viewer is None
            else bool(args.mujoco_viewer)
        ),
        status_host=args.gui_host,
        status_port=args.gui_port,
    )


def _build_dry_run_composition_contracts(
    config: MujocoSimHILApplicationConfig,
    *,
    action_schema: Any,
    prepared_target: Any,
) -> tuple[Any, Any | None]:
    """Select the same resource-free composition contracts as the live graph."""

    from control_station.composition.mujoco_sim_hil import (
        build_component_keyed_mujoco_sim_hil_contracts,
    )
    from control_station.sim_hil_profile.policy_human_allocation import (
        prepare_policy_human_allocation_scheme_a,
        prepare_policy_human_allocation_scheme_b,
    )

    policy_components = action_schema.component_ids
    allocation_profile_selected = (
        config.control_profile_id in SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS
    )
    allocation_prepare = {
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A: (
            prepare_policy_human_allocation_scheme_a
        ),
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B: (
            prepare_policy_human_allocation_scheme_b
        ),
    }.get(config.control_profile_id)
    allocation_preparation = (
        allocation_prepare(
            action_schema=action_schema,
            target_frame_graph=(
                prepared_target.target_frame_profile.frame_graph
            ),
            target_frame_profile_hash=(
                prepared_target.target_frame_profile.semantic_content_hash
            ),
            tool_frame_profile_hash=(
                prepared_target.tool_frame_profile.semantic_content_hash
            ),
            composer_producer_id="control-station-sim-hil-composer",
        )
        if allocation_profile_selected
        else None
    )
    if allocation_preparation is not None:
        return allocation_preparation.bundle, allocation_preparation

    require_sim_hil_control_profile_available(config.control_profile_id)
    return (
        build_component_keyed_mujoco_sim_hil_contracts(
            action_schema,
            composer_producer_id="control-station-sim-hil-composer",
            proposal_ttl_sec=0.25,
            policy_active_components=policy_components,
            human_weight=config.initial_human_alpha,
            policy_weight=1.0 - config.initial_human_alpha,
        ),
        None,
    )


def _run_dry_run(config: MujocoSimHILApplicationConfig) -> int:
    from sim.mujoco.deploy import (
        MujocoSimHILTargetConfig,
        build_rm65_right_cube_lift_component_target_profile,
        prepare_mujoco_sim_hil_target,
    )
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_policy_from_checkpoint,
        check_task_config_compatibility,
    )
    from sim.mujoco_temp_pipeline.policy import resolve_device
    from sim.mujoco_temp_pipeline.hil_policy import (
        validate_temp_ppo_hil_profile,
    )
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from teleop_runtime.calibration import TeleopCalibration

    task = CubeLiftTaskConfig.load(config.task_config_path)
    policy, _normalizer, payload = build_policy_from_checkpoint(
        config.checkpoint_path,
        device=resolve_device(config.device),
    )
    compatibility = check_task_config_compatibility(
        payload,
        config.task_config_path,
    )
    validate_temp_ppo_hil_profile(task, policy)
    calibration = TeleopCalibration.load_json(config.calibration_profile_path)
    calibration, calibration_binding = _bind_vive_coordinate_profile(
        calibration,
        coordinate_profile_path=config.vive_coordinate_profile_path,
        home_profile_path=config.calibration_profile_path,
        required_sides=("right",),
    )
    calibration_binding["home_profile_path"] = str(config.calibration_profile_path)
    calibration_binding["home_profile_sha256"] = _file_sha256(
        config.calibration_profile_path
    )
    home_ready = bool(calibration.metadata.get("home_ready") is True)
    prepared_target = prepare_mujoco_sim_hil_target(
        MujocoSimHILTargetConfig(
            task_config_path=config.task_config_path,
            realtime_worker=config.realtime_worker,
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            position_control_profile_name=config.position_control_profile_name,
            compose_target_hz=config.compose_target_hz,
            target_admission_margin_sec=config.target_admission_margin_sec,
            exposed_pose_sides=("right",),
        )
    )
    component_profile = build_rm65_right_cube_lift_component_target_profile(
        prepared_target.scene,
        prepared_target.task_config,
    )
    action_schema = component_profile.action_schema
    policy_components = action_schema.component_ids
    contracts, allocation_preparation = (
        _build_dry_run_composition_contracts(
            config,
            action_schema=action_schema,
            prepared_target=prepared_target,
        )
    )
    contract_preflight = {
        "passed": True,
        "control_profile_id": config.control_profile_id,
        "allocation_profile_enabled": allocation_preparation is not None,
        "action_schema_hash": action_schema.semantic_content_hash,
        "composition_spec_hash": contracts.composition_spec.semantic_content_hash,
        "strategy_spec_hash": contracts.strategy_spec.semantic_content_hash,
        "authority_profile_hash": contracts.authority_profile.semantic_content_hash,
        "required_roles": list(contracts.composition_spec.required_roles),
        "component_ids": list(action_schema.component_ids),
        "policy_active_components": list(policy_components),
        "arm_representation": "cartesian_pose",
        "end_effector_representation": "physical_joint_qpos",
        "target_handoff_required": True,
        "target_binding_validation": "startup_target_construction",
        "initial_human_alpha": (
            None
            if allocation_preparation is not None
            else config.initial_human_alpha
        ),
        "initial_human_alpha_applied": allocation_preparation is None,
        "allocation_promotable_modes": (
            None
            if allocation_preparation is None
            else list(
                allocation_preparation.station_profiles.proposal
                .promotable_allocation_modes
            )
        ),
        "allocation_automatic_mode_ceiling_by_component": (
            None
            if allocation_preparation is None
            else {
                policy.component_id: policy.automatic_mode_ceiling
                for policy in (
                    allocation_preparation.bundle.allocation_spec.component_policies
                )
            }
        ),
        "allocation_transition_continuity_policy_hash": (
            None
            if allocation_preparation is None
            else (
                allocation_preparation.bundle.allocation_spec
                .transition_continuity_policy_hash
            )
        ),
        "target_admission_margin_sec": (
            config.target_admission_margin_sec
        ),
    }
    print(
        json.dumps(
            {
                "dry_run": True,
                "enabled": bool(config.enabled),
                "control_profile_id": config.control_profile_id,
                "execution_target": "mujoco_only",
                "real_arm_motion": False,
                "checkpoint": str(config.checkpoint_path),
                "task_config": str(config.task_config_path),
                "task_name": task.name,
                "position_control_profile": (
                    prepared_target.position_profile.name
                ),
                "compose_target_hz": prepared_target.compose_target_hz,
                "reference_hz": prepared_target.timing.reference_hz,
                "command_hz": prepared_target.timing.command_hz,
                "policy_dimensions": [policy.observation_dim, policy.action_dim],
                "policy_profile_validated": True,
                "compatibility": compatibility,
                "calibration_profile": str(config.calibration_profile_path),
                "vive_coordinate_profile": calibration_binding[
                    "vive_coordinate_profile_path"
                ],
                "calibration_binding": calibration_binding,
                "home_ready": home_ready,
                "contract_preflight": contract_preflight,
                "task_initialization_info": dict(
                    prepared_target.task_initialization_info
                ),
                "device_start_skipped": True,
                "authority_and_runtime_start_skipped": True,
                "mujoco_task_initialization_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if not home_ready:
        return 2
    return 0


def run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.vive_calibration_gui and not args.acceptance_gui:
        raise ValueError(
            "--vive-calibration-gui is a compatibility flag for the "
            "integrated --control-gui calibration controls."
        )
    if args.dry_run:
        return _run_dry_run(config)
    if args.acceptance_gui and not args.enable:
        raise ValueError("--control-gui requires explicit --enable.")
    if args.acceptance_gui and not config.mujoco_viewer_enabled:
        raise ValueError(
            "--control-gui requires the MuJoCo viewer for interactive "
            "teleoperation and policy visualization."
        )
    if not args.enable:
        print(
            json.dumps(
                {
                    "enabled": False,
                    "default_off": True,
                    "profile": "control_station_mujoco_sim_hil_compound_v1",
                    "message": (
                        "Pass --enable to start MuJoCo and real passive "
                        "VIVE/gripper inputs."
                    ),
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.duration < 0.0:
        raise ValueError("duration must be non-negative.")
    if args.acceptance_gui:
        return _run_acceptance_gui(args, config)
    app = _build_application(config)
    gui_server = None
    if args.gui:
        from .sim_hil_gui import SimHILStatusServer

        gui_server = SimHILStatusServer(
            app.status,
            host=config.status_host,
            port=config.status_port,
            live_status_provider=app.vive_status,
            open_browser=bool(args.open_browser),
        )
        host, port = gui_server.start()
        print(f"Sim-HIL read-only GUI: http://{host}:{port}/", flush=True)
    try:
        report = app.start()
        print(
            json.dumps(
                {
                    "started": True,
                    "preflight_passed": report.passed,
                    "readiness_hash": report.readiness.semantic_content_hash,
                    "execution_target": "mujoco_only",
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        rate = config.control_rate_hz
        if rate is None:
            # The policy source keeps its own inference cadence.  The Station
            # loop must drive the final compose/authority schedule selected by
            # the target controller profile (60 Hz for the default Sim-HIL
            # position_60_120hz profile), rather than falling back to policy_hz.
            rate = float(app.runtime.config.decision_hz)
        period = 1.0 / rate
        started_at = time.monotonic()
        next_tick = started_at
        overrun_dropped_cycles = 0
        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - started_at >= args.duration:
                break
            app.step(now=now)
            if app.viewer_exit_requested:
                break
            next_tick, skipped = _advance_control_deadline(
                previous_deadline=next_tick,
                completed_at=time.monotonic(),
                period=period,
            )
            overrun_dropped_cycles += skipped
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(min(remaining, period))
    finally:
        try:
            app.close()
        finally:
            if gui_server is not None:
                gui_server.close()
    return 0


def _build_sim_hil_vive_setup_controller(
    config: MujocoSimHILApplicationConfig,
    *,
    source_role: str,
) -> Any:
    """Start the pre-target VIVE owner used by the unified Station GUI."""

    from .vive_calibration import (
        ViveCoordinateCalibrationConfig,
        ViveCoordinateCalibrationController,
    )

    role = str(source_role).strip()
    if role not in {"left_wrist", "right_wrist"}:
        raise ValueError(
            "Sim-HIL VIVE setup source_role must be left_wrist or "
            "right_wrist."
        )
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            desk_tracker="auto",
            vive_port=config.vive_port,
            vive_clock_sync_port=config.vive_clock_sync_port,
            initial_profile=config.vive_coordinate_profile_path,
        )
    )
    if role == "left_wrist":
        # The right-only Sim-HIL compatibility profile historically selects
        # the sender's left_wrist role for logical right_wrist_pose. Preserve
        # that mapping visibly, then let the unified GUI swap it if needed.
        swap = getattr(
            controller.source,
            "swap_left_right_source_roles",
            None,
        )
        if not callable(swap):
            raise RuntimeError(
                "Sim-HIL VIVE setup source does not support the declared "
                "single-tracker role mapping."
            )
        swap()
    try:
        controller.start(start_source=True)
    except Exception:
        # The setup receiver is optional for a policy-only runtime. Keep its
        # typed stopped/error snapshot in the GUI so PPO construction remains
        # independent while Human modes stay unavailable.
        pass
    return controller


def _sim_hil_bootstrap_status(
    config: MujocoSimHILApplicationConfig,
    *,
    setup: Mapping[str, object],
    vive_status: Mapping[str, object],
) -> dict[str, Any]:
    """Project VIVE/profile setup without constructing a MuJoCo target."""

    calibration = (
        setup.get("calibration", {})
        if isinstance(setup.get("calibration", {}), Mapping)
        else {}
    )
    input_ready = bool(setup.get("input_ready", False))
    profile_ready = bool(setup.get("profile_ready", False))
    failed_checks = [
        name
        for name, ready in (
            ("vive_input_ready", input_ready),
            ("vive_coordinate_profile_ready", profile_ready),
        )
        if not ready
    ]
    receiver = (
        vive_status.get("vive_receiver", {})
        if isinstance(vive_status.get("vive_receiver", {}), Mapping)
        else {}
    )
    return {
        "schema": SIM_HIL_STATUS_SCHEMA_V1,
        "sampled_at": time.monotonic(),
        "profile_id": "control_station_mujoco_sim_hil_compound_v1",
        "control_profile_id": config.control_profile_id,
        "enabled": bool(config.enabled),
        "started": False,
        "lifecycle": str(setup.get("phase") or "vive_setup"),
        "held": False,
        "episode_ended": False,
        "startup": {
            "restart_required": False,
            "first_failure": None,
            "closed": False,
            "phase": setup.get("phase"),
            "mujoco_constructed": False,
        },
        "first_control_fault": None,
        "visualization": {
            "enabled": bool(config.mujoco_viewer_enabled),
            "state": "not_constructed",
            "running": False,
            "sync_count": 0,
            "last_error": None,
        },
        "control_schedule": {
            "next_decision_at": None,
            "overrun_dropped_cycles": 0,
            "last_overrun": None,
            "target_admission_margin_sec": (
                config.target_admission_margin_sec
            ),
            "target_lead_time_sec": None,
            "proposal_horizon_sec": None,
        },
        "readiness": {
            "ready": False,
            "required_checks": [
                "vive_input_ready",
                "vive_coordinate_profile_ready",
            ],
            "failed_checks": failed_checks,
            "detail_by_check": {
                "vive_input_ready": (
                    "live finite fresh right_wrist_pose"
                    if input_ready
                    else (
                        calibration.get("input_preflight", {})
                        if isinstance(
                            calibration.get("input_preflight", {}),
                            Mapping,
                        )
                        else {}
                    ).get("message", "VIVE input is not ready")
                ),
                "vive_coordinate_profile_ready": (
                    str(setup.get("selected_profile_path"))
                    if profile_ready
                    else "No valid saved coordinate profile is selected."
                ),
            },
        },
        "execution_gate": {
            "status": "not_constructed",
            "reason": "waiting_for_vive_setup",
        },
        "policy": {
            "started": False,
            "ready": False,
            "refresh_state": "not_constructed",
            "checkpoint": str(config.checkpoint_path),
            "last_emission": {},
        },
        "human": {
            "session_state": "VIVE_SETUP",
            "source_started": bool(
                calibration.get("source_started", False)
            ),
            "last_emission": {},
            "last_emission_age_sec": None,
            "binding": {
                "rig_absent": True,
                "authoritative": False,
            },
            "session_safety": {},
            "channels": dict(calibration.get("live_channels") or {}),
            "vive_receiver": dict(receiver),
        },
        "composition": {
            "started": False,
            "activation": {},
            "last_cycle": {},
            "mix": {},
        },
        "execution_difference": {
            "enabled": bool(config.execution_difference_enabled),
            "latest": {},
        },
        "last_error": None,
        "execution_target": "mujoco_not_constructed",
        "real_arm_motion": False,
        "single_execution_owner": "not_created",
        "calibration": {
            "vive_coordinate_profile_path": setup.get(
                "selected_profile_path"
            ),
            "mode": "station_vive_setup_before_target",
        },
        "task_initialization": {},
        "source_configuration": {
            "vive_port": int(config.vive_port),
            "vive_clock_sync_port": config.vive_clock_sync_port,
            "active_components": [],
            "vive_source_role_by_output_channel": dict(
                calibration.get("source_role_by_output_channel") or {}
            ),
            "vive_source_swap_count": calibration.get(
                "source_swap_count",
                0,
            ),
            "input_device_bindings": [
                {
                    "component_id": "right_gripper",
                    "requested_device_kind": config.gripper_input_source,
                    "device_kind": None,
                    "endpoint": None,
                    "fallback_reason": "resolved_after_vive_setup",
                }
            ],
        },
        "contracts": {},
    }


def _run_acceptance_gui(
    args: argparse.Namespace,
    config: MujocoSimHILApplicationConfig,
) -> int:
    """Run formal interactive control with one Station-owned control loop."""

    import secrets

    from .sim_hil_acceptance import (
        SimHILAcceptanceController,
        SimHILAcceptanceTraceCapture,
    )
    from .sim_hil_gui import SimHILStatusServer

    capture = None
    acceptance_config = _acceptance_profile_config(config)
    controller = None
    gui_server = None
    try:
        storage_resolution = resolve_acceptance_root(args)
        capture = SimHILAcceptanceTraceCapture(storage_resolution.path)
        _write_acceptance_storage_receipt(
            session_root=capture.session_root,
            resolution=storage_resolution,
            requested_layout=str(
                getattr(
                    args,
                    "artifact_layout",
                    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
                )
            ),
        )

        def setup_factory(source_role: str) -> Any:
            return _build_sim_hil_vive_setup_controller(
                acceptance_config,
                source_role=source_role,
            )

        setup_controller = setup_factory(
            acceptance_config.vive_source_tracker_role
        )

        def application_factory(
            coordinate_profile_path: Path,
            source_role: str,
            initial_human_alpha: float,
        ) -> MujocoSimHILApplication:
            defer_human_input = bool(
                initial_human_alpha <= 1e-12
            )
            runtime_config = replace(
                acceptance_config,
                vive_coordinate_profile_path=(
                    None
                    if defer_human_input
                    else coordinate_profile_path
                ),
                vive_source_tracker_role=source_role,
                initial_human_alpha=float(initial_human_alpha),
                defer_human_input=defer_human_input,
            )
            return _build_application(
                runtime_config,
                trace_sink=capture.admit_trace,
            )

        controller = SimHILAcceptanceController(
            None,
            trace_capture=capture,
            device_check_timeout_sec=float(
                args.acceptance_device_timeout
            ),
            application_factory=application_factory,
            vive_setup_controller=setup_controller,
            vive_setup_factory=setup_factory,
            coordinate_profile_path=(
                acceptance_config.vive_coordinate_profile_path
                or DEFAULT_VIVE_COORDINATE_PROFILE
            ),
            allocation_runtime_enabled=(
                acceptance_config.control_profile_id
                in SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS
            ),
        )

        cached_runtime_status: dict[str, Any] | None = None

        def status_provider() -> dict[str, Any]:
            nonlocal cached_runtime_status
            acceptance_status = controller.status()
            app = controller.current_application
            if app is None:
                setup = controller.setup_status()
                value = _sim_hil_bootstrap_status(
                    acceptance_config,
                    setup=setup,
                    vive_status=controller.vive_status(),
                )
            elif acceptance_status.get("active_intent_id") is not None:
                # start/rearm/Home may hold the application lock while waiting
                # on lower resources.  Serving the last immutable projection
                # keeps cancel/exit reachable instead of blocking the HTTP
                # status request behind the operation it must interrupt.
                if cached_runtime_status is None:
                    value = _sim_hil_bootstrap_status(
                        acceptance_config,
                        setup=controller.setup_status(),
                        vive_status=controller.vive_status(),
                    )
                    value["startup"]["mujoco_constructed"] = True
                else:
                    value = dict(cached_runtime_status)
            else:
                value = app.status()
                cached_runtime_status = dict(value)
            value["acceptance"] = acceptance_status
            value["operator_guidance"] = build_sim_hil_operator_guidance(
                value
            )
            return value

        gui_server = SimHILStatusServer(
            status_provider,
            host=acceptance_config.status_host,
            port=acceptance_config.status_port,
            live_status_provider=controller.vive_status,
            intent_submitter=controller.submit_json,
            control_token=secrets.token_urlsafe(32),
            open_browser=bool(args.open_browser),
        )
        host, port = gui_server.start()
        print(
            f"Sim-HIL control GUI: http://{host}:{port}/",
            flush=True,
        )
        started_at = time.monotonic()
        passive_observation_due_at = started_at
        while not controller.shutdown_requested:
            now = time.monotonic()
            if (
                args.duration > 0.0
                and now - started_at >= args.duration
            ):
                break
            controller.process_next()
            controller.enforce_capture_health()
            app = controller.current_application
            if (
                app is not None
                and app.started
                and not app._held
                and not app._episode_ended
            ):
                due_at = app._next_decision_at
                if due_at is None or now >= due_at:
                    try:
                        app.step(now=now)
                    except Exception:
                        # The application/lower gate retains the fault and
                        # HOLD evidence. Keep the GUI alive for inspection.
                        if not app._held:
                            try:
                                app.hold(
                                    reason="acceptance_control_loop_fault"
                                )
                            except Exception:
                                pass
            elif (
                app is not None
                and app.started
                and app._held
                and not app._episode_ended
                and now >= passive_observation_due_at
            ):
                # Keep the passive device observation fresh for diagnosis and
                # Home/rearm while HOLD. This never steps policy, composition,
                # the target, or MuJoCo dispatch.
                try:
                    app.poll_passive_human_input(now=now)
                except Exception:
                    pass
                passive_observation_due_at = now + 0.05
            if app is not None and app.viewer_exit_requested:
                break
            next_due = None if app is None else app._next_decision_at
            delay = 0.01
            if (
                app is not None
                and app.started
                and not app._held
                and next_due is not None
            ):
                rate = (
                    float(acceptance_config.control_rate_hz)
                    if acceptance_config.control_rate_hz is not None
                    else float(app.runtime.config.decision_hz)
                )
                period = 1.0 / rate
                delay = max(
                    0.0005,
                    min(period, float(next_due) - time.monotonic()),
                )
            time.sleep(delay)
        return 0
    finally:
        if gui_server is not None:
            gui_server.close()
        app = (
            None
            if controller is None
            else controller.current_application
        )
        if app is not None:
            try:
                app.close()
            finally:
                if controller is not None:
                    controller.close()
                elif capture is not None:
                    capture.close()
        elif controller is not None:
            controller.close()
        elif capture is not None:
            capture.close()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return int(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - CLI reports a typed failure.
        print(f"control_station sim-hil failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CALIBRATION_PROFILE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_TASK_CONFIG",
    "DEFAULT_VIVE_COORDINATE_PROFILE",
    "MujocoSimHILApplication",
    "MujocoSimHILApplicationConfig",
    "SIM_HIL_HUMAN_TIMEOUT_FALLBACKS",
    "SIM_HIL_STATUS_SCHEMA_V1",
    "build_arg_parser",
    "main",
    "run",
]
