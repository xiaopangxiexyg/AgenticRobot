"""Canonical real-teleop application lifecycle.

The debug entrypoint parses arguments, refreshes this module's explicit
compatibility support bindings, and delegates here.  Keeping the support seam
preserves historical entrypoint monkeypatches while the lifecycle body is
split away mechanically.  Device startup and control-loop ordering remain
under :func:`run_real_teleop`; ordered best-effort release is delegated to the
shutdown runtime without transferring device or control authority.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import shutil
from typing import Any

from ..operator_ui_ports import RealTeleopInputBatchPrioritizer
from . import control_loop as _control_loop
from .artifact_context import materialize_real_teleop_artifact_context
from .control_loop import (
    RealTeleopControlLoopContext,
    run_real_teleop_control_loop,
)
from .components import RealTeleopComponents
from .device_retry_runtime import build_device_retry_runtime
from .diagnostic_setup import (
    initialize_control_debug_runtime_metadata,
    setup_actuator_command_trace,
    setup_control_debug_runtime,
)
from .episode_recording_runtime import build_episode_recording_runtime
from .foot_pad_controls import (
    apply_live_foot_pad_topology,
    foot_pad_progress_topology_guidance,
    initialize_startup_foot_pad_input,
)
from .gripper_warmup_runtime import build_gripper_warmup_probe_runtime
from .position_episode_debug import build_position_episode_debug_runtime
from .runtime_extension import RealTeleopRuntimeExtension
from .position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
    install_position_admittance_active,
    install_position_admittance_shadow,
    position_admittance_active_apply_sides,
    position_admittance_active_config,
    position_admittance_shadow_config,
    position_rotational_admittance_config,
    validate_position_admittance_active_authority,
)
from .progress_contracts import ProgressApplicationPort
from .recording_admission import (
    recording_auto_stop_reason as _recording_auto_stop_reason,
    recording_warmup_status as _recording_warmup_status,
)
from .shutdown_runtime import shutdown_real_teleop_runtime
from .startup_runtime import (
    StartupCleanupRegistry as _StartupCleanupRegistry,
    activate_startup_cleanup as _activate_startup_cleanup,
    cleanup_active_startup,
    deactivate_startup_cleanup as _deactivate_startup_cleanup,
    register_component_startup_cleanup as _register_component_startup_cleanup,
)
from .startup_progress_runtime import RealTeleopStartupProgressRuntime
from .startup_device_runtime import StartupDeviceRuntime
from .startup_admission import (
    assemble_startup_components,
    configure_startup_cameras,
    initialize_startup_ui,
)
from .tactile_idle_compression import (
    RealTeleopIdleTactileCompression,
)
from .visualization_runtime import build_visualization_runtime
from .defaults import runtime_home_fresh_data_timeout_sec

_REQUIRED_SUPPORT_NAMES = (
    "ActionDebugLogRecorder",
    "AsyncDMTacDistributedForceVisualizer",
    "AsyncDebugTimingConsole",
    "AsyncTactileUVVisualizer",
    "CalibrationWebUi",
    "CameraPreviewConfig",
    "CameraPreviewer",
    "CaptureModalityContract",
    "ChangeGate",
    "DEFAULT_ACTION_DEBUG_LOG_ROOT",
    "DEFAULT_CAMERA_FRAME_MAX_AGE_SEC",
    "DEFAULT_EPISODE_START_COUNTDOWN_SEC",
    "DEFAULT_INSPIRE_HAND_JOINT_ORDER",
    "DEFAULT_LEFT_ARM_QPOS",
    "DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC",
    "DEFAULT_RIGHT_ARM_QPOS",
    "DEFAULT_RUNTIME_HOME_START_DELAY_SEC",
    "DMTacDistributedForceVisualizer",
    "DMTacGripperBackend",
    "DebugTimingLogRecorder",
    "EpisodeRecorder",
    "EpisodeRecorderConfig",
    "FootPadSpecialArmInput",
    "FootPadUnavailableError",
    "GRIPPER_ACTION_MAX",
    "GuiStartupProgress",
    "HAND_BACKEND_NONE",
    "IkDebugLogRecorder",
    "MainLoopStallWatchdog",
    "Mapping",
    "Path",
    "PeriodicRateGate",
    "PhaseTimer",
    "PositionAuditFanout",
    "PositionForceGuardMonitor",
    "PreviewRateLimiter",
    "RealDebugTimingState",
    "RealTeleopControlInput",
    "RealTeleopDatasetRecorder",
    "SPECIAL_ARM_CONTROL_MODE_FOOT_PAD",
    "Sequence",
    "SpecialArmFixHoldCoordinator",
    "TRACKER_HOLD_AUTOMATIC",
    "TRACKER_HOLD_GRACE",
    "TeleopRunState",
    "TerminalClutchInput",
    "_active_command_backend",
    "_active_sides_from_args",
    "_advance_recovery_plan",
    "_advance_startup_gui_qpos_maintenance",
    "_apply_active_arm_command_to_preview_real",
    "_arm_after_start_calibration",
    "_arm_control_mode_arg",
    "_arm_execution_profile_metadata",
    "_attach_actuator_command_trace_sink",
    "_attach_ik_shadow_trace_sink",
    "_attach_position_audit_sink",
    "_attach_singularity_transition_trace_sink",
    "_build_action_debug_metadata",
    "_build_camera_sources",
    "_build_components",
    "_build_debug_timing_log_frame",
    "_build_episode_frame",
    "_build_episode_metadata",
    "_build_ik_debug_log_frame",
    "_build_ik_timing_payload",
    "_build_ik_warning_payload",
    "_build_pose_quality_display_payload",
    "_build_renderer_debug_boxes",
    "_build_renderer_debug_frames",
    "_build_runtime_hand_command_for_recovery",
    "_build_tactile_visualizer",
    "_camera_status_blocking_message",
    "_cancel_special_arm_modes_for_hold",
    "_canfd_follow_runtime_config",
    "_channel_pose_in_desk",
    "_channels_log_payload",
    "_close_camera_sources",
    "_collection_sample_step_result",
    "_collection_step_result",
    "_command_backend_health_signature",
    "_command_backend_status_text",
    "_copy_teleop_command",
    "_dataset_recording_debug_text",
    "_debug_episode_recording_enabled",
    "_debug_timing_maybe_report",
    "_debug_timing_metadata",
    "_debug_timing_modules",
    "_disengage_position_backends_for_manual_interaction",
    "_dispatch_q6_gripper_control",
    "_effective_arm_canfd_follow",
    "_effective_arm_control_hz",
    "_effective_arm_follow_qpos_alpha",
    "_effective_dual_arm_temporary_safety",
    "_effective_outer_control_hz",
    "_effective_position_post_ik_qpos_smoothing_alpha",
    "_emergency_stop_real_arms",
    "_emit_collision_proxy_logs",
    "_emit_home_alignment_logs",
    "_emit_runtime_orientation_logs",
    "_emit_tactile_sum_only_status",
    "_emit_tracker_axis_mapping_logs",
    "_enable_gui_tactile_solving",
    "_episode_root_dir",
    "_episode_start_countdown_ready_to_record",
    "_episode_start_countdown_remaining",
    "_expand_cli_path",
    "_extract_hand_command_from_command",
    "_faults_log_payload",
    "_filter_manual_real_command",
    "_fix_arm_debug_state",
    "_fix_arm_pose_snapshot_recorder",
    "_fix_arm_status_text",
    "_fix_arm_ui_state",
    "_force_guard_recording_stop_context",
    "_force_tare_sec_arg",
    "_gripper_backend_for_side",
    "_gripper_retargeter_for_side",
    "_gripper_runtime_health_signature",
    "_gripper_runtime_status_signature",
    "_guard_impedance_force_control",
    "_hand_backend_for_side",
    "_hand_backend_mode_for_side",
    "_hand_channel_status",
    "_hand_tactile_sum",
    "_handle_ik_debug_log_toggle_request",
    "_handle_init_qpos_emergency_stop",
    "_has_dmtac_gripper_backend",
    "_hold_requested_pause_reason",
    "_ik_timing_status_text",
    "_impedance_mode_arg",
    "_init_arm_debug_state",
    "_init_arm_status_text",
    "_init_arm_ui_state",
    "_log",
    "_log_dual_arm_readiness",
    "_manual_calibration_confirmation_required",
    "_mean_pose",
    "_merge_control_debug_runtime_metadata",
    "_merge_runtime_hand_command",
    "_movement_multiple_ui_state",
    "_movement_multiple_value",
    "_new_actuator_command_trace_recorder",
    "_new_position_debug_recorder",
    "_parse_hand_tactile_regions",
    "_pose_quality_reason_list",
    "_pose_quality_status_text",
    "_position_force_guard_collision_summary",
    "_position_force_guard_config",
    "_position_ik_iteration_metadata",
    "_promote_angle_trigger_gripper_input_to_q6",
    "_qpos_summary",
    "_real_hand_render_qpos",
    "_record_position_loop_timings",
    "_release_gui_qpos_init_transit",
    "_release_runtime_state_readers",
    "_resolve_logical_side_assignments",
    "_resolve_robot_home_for_side",
    "_restore_init_arm_after_foot_pad_fallback",
    "_run_camera_startup_self_check",
    "_run_gripper_warmup_probe_for_side",
    "_run_initial_alignment_start_check",
    "_runtime_arm_backend_state",
    "_runtime_fault_label",
    "_runtime_home_calibration",
    "_runtime_home_failure_message",
    "_safe_float_value",
    "_select_preview_camera_sources",
    "_short_status_text",
    "_side_retargeter",
    "_start_camera_sources",
    "_start_camera_sources_best_effort",
    "_start_episode_recording",
    "_start_recovery_plan",
    "_state_qpos",
    "_stop_episode_recording",
    "_strict_tool_profile_arg",
    "_sync_dual_arm_retargeter_for_real_control",
    "_sync_preview_from_real_state_real",
    "_sync_real_execution_step_limits",
    "_tactile_solving_active",
    "_tactile_solving_available",
    "_tactile_visualization_available",
    "_temporary_safety_health_signature",
    "_terminal_force_collision_summary",
    "_tracker_source_side_for_logical_side",
    "_update_init_arm_arrival_from_hardware",
    "_vive_pose_source_from_components",
    "_wait_for_real_teleop_start_request",
    "advance_fix_arm_home_capture",
    "advance_init_arm_home_capture",
    "bind_capture_modality_contract",
    "build_fix_arm_pose_snapshot_payload",
    "datetime",
    "dmtac_visualization_snapshot_from_backends",
    "legacy_tcp_mode_requested",
    "merge_teleop_commands",
    "np",
    "os",
    "replace",
    "run_best_effort_cleanup",
    "sys",
    "tactile_by_side_from_backends",
    "threading",
    "time",
)


def bind_legacy_support(namespace: Mapping[str, Any]) -> None:
    """Refresh entrypoint-owned compatibility symbols before one runtime run."""

    missing = [
        name
        for name in _REQUIRED_SUPPORT_NAMES
        if name not in namespace
    ]
    if missing:
        raise RuntimeError(
            "real-teleop runtime support is incomplete: " + ", ".join(missing)
        )
    module_globals = globals()
    for name in _REQUIRED_SUPPORT_NAMES:
        module_globals[name] = namespace[name]
    _control_loop.bind_legacy_support(namespace)


def _position_admittance_runtime_event_payload(
    *,
    requested_mode: str,
    enabled: bool,
    active_sides: tuple[str, ...] | list[str],
    coupled_pre_ik_target_filter_available: bool,
    reason: str | None = None,
    snapshot: Mapping[str, object] | None = None,
    endpoint_excursion_guard: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one bounded startup receipt without changing control state."""

    payload = dict(snapshot or {})
    payload.update(
        {
            "runtime_schema": "teleop_runtime.position_admittance_startup/v1",
            "enabled": bool(enabled),
            "requested_mode": str(requested_mode),
            "active_sides": [str(side) for side in active_sides],
            "coupled_pre_ik_target_filter_available": bool(
                coupled_pre_ik_target_filter_available
            ),
        }
    )
    if reason is not None:
        payload["reason"] = str(reason)
    if endpoint_excursion_guard is not None:
        payload["endpoint_excursion_guard"] = dict(endpoint_excursion_guard)
    return payload


def _tactile_storage_capacity_summary(args: argparse.Namespace) -> str | None:
    """Describe the effective tactile reserve without owning its admission check."""

    root = str(getattr(args, "tactile_bundle_root", "") or "").strip()
    if not root:
        return None
    minimum_gib = float(
        getattr(args, "tactile_bundle_min_free_gb", 0.0) or 0.0
    )
    source = str(
        getattr(args, "tactile_bundle_min_free_source", "unknown") or "unknown"
    )
    try:
        free_gib = float(shutil.disk_usage(root).free) / float(1024**3)
    except OSError as exc:
        return (
            f"root={root} free_gib=unavailable minimum_free_gib={minimum_gib:g} "
            f"source={source} diagnostic_error={type(exc).__name__}: {exc}"
        )
    capacity = "sufficient" if free_gib >= minimum_gib else "insufficient"
    return (
        f"root={root} free_gib={free_gib:.1f} "
        f"minimum_free_gib={minimum_gib:g} source={source} capacity={capacity}"
    )


def run_real_teleop(
    args: argparse.Namespace,
    *,
    progress_application: ProgressApplicationPort | None = None,
    input_batch_prioritizer: RealTeleopInputBatchPrioritizer | None = None,
) -> None:
    runtime_extension: RealTeleopRuntimeExtension | None = getattr(
        args,
        "_real_teleop_runtime_extension",
        None,
    )
    if hasattr(args, "_real_teleop_runtime_extension"):
        delattr(args, "_real_teleop_runtime_extension")
    position_admittance_rotation_active = bool(
        getattr(args, "position_admittance_rotation_active", False)
    )
    position_admittance_combined_active = bool(
        getattr(args, "position_admittance_combined_active", False)
    )
    if position_admittance_rotation_active and not bool(
        getattr(args, "position_admittance_active", False)
    ):
        raise RuntimeError(
            "rotation admittance Active requires position admittance Active"
        )
    if position_admittance_combined_active and not (
        position_admittance_rotation_active
        and bool(getattr(args, "position_admittance_active", False))
    ):
        raise RuntimeError(
            "combined translation and rotation admittance Active requires "
            "both position and rotation admittance Active"
        )
    if bool(getattr(args, "position_admittance_shadow", False)):
        try:
            position_admittance_shadow_config(args)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "position admittance shadow preflight failed: " + str(exc)
            ) from exc
    if bool(getattr(args, "position_admittance_rotation_shadow", False)):
        if not bool(getattr(args, "position_admittance_shadow", False)):
            raise RuntimeError(
                "rotation admittance shadow requires position admittance shadow"
            )
        if bool(getattr(args, "position_admittance_active", False)):
            raise RuntimeError(
                "rotation admittance shadow cannot authorize position Active"
            )
        try:
            position_rotational_admittance_config(
                args,
                profile=str(args.position_admittance_rotation_shadow_profile),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "rotation admittance shadow preflight failed: " + str(exc)
            ) from exc
    if bool(getattr(args, "position_admittance_active", False)):
        # Defense in depth for direct Namespace callers that bypass the
        # canonical CLI resolver.  Validate the same final Active envelope
        # before UI, devices, or arms start; the canonical installer repeats
        # this check immediately before attaching the pre-IK modifier.
        try:
            maximum_tool_excursion = float(
                getattr(
                    args,
                    "position_admittance_active_max_tool_excursion_m",
                    POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M,
                )
            )
            rotation_config = None
            maximum_tool_rotation_excursion_deg = None
            if position_admittance_rotation_active:
                rotation_profile = str(
                    getattr(
                        args,
                        "position_admittance_rotation_active_profile",
                        POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
                    )
                )
                rotation_config = position_rotational_admittance_config(
                    args,
                    profile=rotation_profile,
                )
                maximum_tool_rotation_excursion_deg = float(
                    getattr(
                        args,
                        "position_admittance_active_max_tool_rotation_excursion_deg",
                        POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG,
                    )
                )
            validate_position_admittance_active_authority(
                position_admittance_active_config(args),
                active_envelope=str(
                    getattr(args, "position_admittance_active_envelope", "cautious")
                ),
                rotational_config=rotation_config,
                combined_translation_rotation_active=(
                    position_admittance_combined_active
                ),
                endpoint_policy=str(
                    getattr(
                        args,
                        "position_admittance_active_endpoint_policy",
                        POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
                    )
                ),
                apply_sides=position_admittance_active_apply_sides(
                    str(getattr(args, "position_admittance_active_side", ""))
                ),
                max_tool_excursion_m=maximum_tool_excursion,
                max_tool_rotation_excursion_deg=(
                    maximum_tool_rotation_excursion_deg
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "active position admittance preflight failed: " + str(exc)
            ) from exc
    materialize_real_teleop_artifact_context(args)
    startup_cleanup = _activate_startup_cleanup(logger=_log)
    capability_admission = (
        str(getattr(args, "hardware_admission_mode", "strict")) == "capability"
    )
    startup_ui = initialize_startup_ui(
        args,
        startup_cleanup=startup_cleanup,
        ui_type=CalibrationWebUi,
        progress_type=GuiStartupProgress,
    )
    control_ui = startup_ui.control_ui
    startup_progress = startup_ui.progress
    foot_pad_input = initialize_startup_foot_pad_input(
        args,
        foot_pad_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        input_type=FootPadSpecialArmInput,
        unavailable_error_type=FootPadUnavailableError,
        startup_cleanup=startup_cleanup,
        startup_progress=startup_progress,
        control_ui=control_ui,
        apply_live_topology=apply_live_foot_pad_topology,
        topology_guidance=foot_pad_progress_topology_guidance,
        restore_init_after_fallback=_restore_init_arm_after_foot_pad_fallback,
        log=_log,
    )
    startup_component_assembly = assemble_startup_components(
        args,
        startup_cleanup=startup_cleanup,
        startup_progress=startup_progress,
        active_sides_from_args=_active_sides_from_args,
        build_tactile_visualizer=_build_tactile_visualizer,
        build_components=_build_components,
        register_component_cleanup=_register_component_startup_cleanup,
    )
    components = startup_component_assembly.components
    if runtime_extension is not None:
        attach_components = getattr(runtime_extension, "attach_components", None)
        if not callable(attach_components):
            raise TypeError("runtime_extension lacks attach_components().")
        components = attach_components(components)
        if not isinstance(components, RealTeleopComponents):
            raise TypeError(
                "runtime_extension.attach_components() must return "
                "RealTeleopComponents."
            )
    tactile_visualizer = startup_component_assembly.tactile_visualizer
    init_controller_available = (
        getattr(components, "init_arm_controller", None) is not None
    )
    _log(
        "InitArm",
        "startup "
        f"requested={bool(getattr(args, 'init_arm_requested', False))} "
        f"enabled={bool(getattr(args, 'init_arm_available', False))} "
        f"controller_available={init_controller_available} "
        f"disabled_reason={str(getattr(args, 'init_arm_disabled_reason', '') or '<none>')} "
        f"profile={str(getattr(args, 'init_arm_profile', '') or '<none>')}",
    )
    if (
        control_ui is not None
        and bool(getattr(args, "init_arm_requested", False))
        and not init_controller_available
    ):
        control_ui.append_message(
            "InitArm",
            "已请求机械臂归位功能，但 Init controller 未构建；"
            f"reason={str(getattr(args, 'init_arm_disabled_reason', '') or 'unknown')}。",
        )
    startup_camera_configuration = configure_startup_cameras(
        args,
        startup_cleanup=startup_cleanup,
        startup_progress=startup_progress,
        control_ui=control_ui,
        capability_admission=capability_admission,
        build_sources=_build_camera_sources,
        select_preview_sources=_select_preview_camera_sources,
        strict_failure_cleanup=(
            tactile_visualizer.close
            if tactile_visualizer is not None
            else lambda: None
        ),
        log=_log,
    )
    camera_sources = startup_camera_configuration.sources
    preview_camera_sources = startup_camera_configuration.preview_sources
    setattr(args, "tactile_solving_available", _tactile_solving_available(args, components))
    setattr(args, "tactile_solving_active", _tactile_solving_active(args, components))
    setattr(args, "tactile_visualization_available", _tactile_visualization_available(args, components))
    if tactile_visualizer is None:
        tactile_visualizer = _build_tactile_visualizer(args, components.active_sides)
    elif not bool(getattr(args, "tactile_visualization_available", False)):
        tactile_visualizer.close()
        tactile_visualizer = None
    if tactile_visualizer is not None:
        tactile_vis_mode = (
            "subprocess"
            if isinstance(
                tactile_visualizer,
                (AsyncTactileUVVisualizer, AsyncDMTacDistributedForceVisualizer),
            )
            else "dmtac_force"
            if isinstance(
                tactile_visualizer,
                (DMTacDistributedForceVisualizer, AsyncDMTacDistributedForceVisualizer),
            )
            else "same_process"
        )
        tactile_vis_detail = (
            "source=dmtac_distributed_force"
            if isinstance(tactile_visualizer, DMTacDistributedForceVisualizer)
            else f"regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', 'all')))}"
        )
        _log(
            "TactileVis",
            f"enabled mode={tactile_vis_mode} active_sides={list(components.active_sides)} "
            f"{tactile_vis_detail}",
        )
    pose_snapshot_recorder = (
        None
        if bool(getattr(args, "no_debug_disk_write", False))
        else _fix_arm_pose_snapshot_recorder(
            components,
            root_dir=(
                getattr(args, "_runtime_artifact_fix_arm_snapshot_root", "")
                or None
            ),
        )
    )
    terminal_factory = getattr(
        args,
        "_real_teleop_terminal_factory",
        TerminalClutchInput,
    )
    if not callable(terminal_factory):
        raise TypeError("_real_teleop_terminal_factory must be callable.")
    clutch_input = RealTeleopControlInput(
        terminal=terminal_factory(components.keyboard_source),
        ui=control_ui,
        args=args,
        components=components,
        pose_snapshot_recorder=pose_snapshot_recorder,
        foot_pad_input=foot_pad_input,
    )
    if (progress_application is None) != (input_batch_prioritizer is None):
        raise RuntimeError(
            "RealTeleop application ports must be injected together."
        )
    if progress_application is not None and input_batch_prioritizer is not None:
        clutch_input.bind_application_ports(
            progress_application=progress_application,
            input_batch_prioritizer=input_batch_prioritizer,
        )
    startup_cleanup.add("clutch_input", clutch_input.close)
    debug_episode_recording_enabled = _debug_episode_recording_enabled(args)
    episode_recorder = EpisodeRecorder(
        root_dir=_episode_root_dir(args, components) if debug_episode_recording_enabled else None,
        config=EpisodeRecorderConfig(enabled=debug_episode_recording_enabled),
    )
    startup_cleanup.add("episode_recorder", episode_recorder.close)
    initialize_control_debug_runtime_metadata(
        args,
        merge_runtime_metadata=_merge_control_debug_runtime_metadata,
    )
    idle_tactile_compression: RealTeleopIdleTactileCompression | None = None
    # The formal writer is deliberately lazy.  Constructing/starting the
    # process worker here used to put ``initialize_legacy_writer`` (including
    # a full persisted-dataset scan) on the global startup critical path.  A
    # slow or temporarily unavailable dataset then made the camera/arm GUI
    # appear to be a startup failure even though collection had not been
    # requested.  The recorder is now created, started, and published by the
    # episode-preparation transaction below, after the operator has selected
    # the final dataset path/profile.  This also gives every retry a fresh
    # one-shot multiprocessing.Process instance.
    dataset_recorder = None
    # A previous in-process construction receipt must not be replayed as a
    # global startup error now that writer construction is deferred.
    args.dataset_startup_error = ""
    if not bool(getattr(args, "no_collection", False)):
        _log(
            "Dataset",
            "formal dataset writer deferred until collection start or an "
            "explicit dataset/profile reconfiguration",
        )

    # Profile selection is configuration-only: it does not start an Episode,
    # move an arm, or mutate the dataset binding.  Make the selector available
    # as soon as the idle recorder/UI exist instead of making it depend on a
    # later startup/Home milestone.  The payload still disables it for a
    # locked dataset or when the managed catalog has no selectable profile.
    startup_progress_runtime = RealTeleopStartupProgressRuntime(
        args=args,
        control_input=clutch_input,
        control_ui=control_ui,
        dataset_recorder=lambda: dataset_recorder,
        log=_log,
    )
    startup_progress_runtime.publish(force=True)
    if control_ui is not None:
        control_ui.publish(
            dataset_modality_contract=(
                CaptureModalityContract(
                    ignored_camera_keys=tuple(getattr(args, "ignore_camera", ()) or ()),
                    ignored_tactile_keys=tuple(getattr(args, "ignore_tactile", ()) or ()),
                ).to_dict()
                if dataset_recorder is None
                else dataset_recorder.capture_modality_contract.to_dict()
            )
        )
    def update_control_debug_runtime_metadata(**updates: object) -> dict[str, object]:
        runtime_metadata = _merge_control_debug_runtime_metadata(args, **updates)
        if dataset_recorder is not None:
            control_debug_metadata = _debug_timing_metadata(args).get("control_debug", {})
            if isinstance(control_debug_metadata, Mapping):
                dataset_recorder.collector_components.metadata["control_debug"] = dict(
                    control_debug_metadata
                )
        return runtime_metadata

    camera_previewer: CameraPreviewer | None = None
    camera_preview_active = False
    tactile_preview_rate = PreviewRateLimiter(float(getattr(args, "tactile_vis_hz", 5.0)))
    camera_preview_rate = PreviewRateLimiter(float(getattr(args, "vis_cam_hz", 10.0)))







    if startup_progress is not None:
        startup_progress.begin(
            "runtime.diagnostics",
            label="输入与诊断组件",
            message="正在准备控制输入、debug 与 actuator trace",
            requirement="runtime_support",
        )
    console_stream = sys.__stdout__
    silenced_stdout = None
    original_stdout = None
    if args.print_tactile_sum_only:
        console_stream.write(
            "[TactileSum] tactile-sum-only mode enabled; press 'p' or Enter for calibration, 'q' to quit.\n"
        )
        console_stream.flush()
        silenced_stdout = open(os.devnull, "w")
        original_stdout = sys.stdout
        sys.stdout = silenced_stdout
    position_audit_fanout: PositionAuditFanout | None = None
    position_debug_session_recorder: PositionDebugRecorder | None = None
    position_episode_debug_runtime = None
    position_debug_session_dir: Path | None = None
    position_debug_close_threads: list[threading.Thread] = []
    actuator_trace_recorder: ActuatorCommandTraceRecorder | None = None
    actuator_trace_session_dir: Path | None = None
    control_debug_setup = setup_control_debug_runtime(
        args,
        components=components,
        startup_cleanup=startup_cleanup,
        timestamp=(
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if bool(args.control_debug_config.enabled)
            else ""
        ),
        expand_cli_path=_expand_cli_path,
        new_recorder=_new_position_debug_recorder,
        audit_fanout_type=PositionAuditFanout,
        attach_position_sink=_attach_position_audit_sink,
        update_runtime_metadata=update_control_debug_runtime_metadata,
        wall_time=time.time,
        monotonic_time=time.monotonic,
        log=_log,
    )
    position_audit_fanout = control_debug_setup.audit_fanout
    position_debug_session_recorder = control_debug_setup.recorder
    position_debug_session_dir = control_debug_setup.session_dir
    actuator_trace_setup = setup_actuator_command_trace(
        args,
        components=components,
        startup_cleanup=startup_cleanup,
        timestamp=(
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if bool(getattr(args, "actuator_trace", False))
            else ""
        ),
        expand_cli_path=_expand_cli_path,
        new_recorder=_new_actuator_command_trace_recorder,
        attach_command_sink=_attach_actuator_command_trace_sink,
        attach_singularity_sink=_attach_singularity_transition_trace_sink,
        attach_ik_shadow_sink=_attach_ik_shadow_trace_sink,
        log=_log,
    )
    actuator_trace_recorder = actuator_trace_setup.recorder
    actuator_trace_session_dir = actuator_trace_setup.session_dir
    clutch_input.start()
    if startup_progress is not None:
        startup_progress.succeed(
            "runtime.diagnostics",
            message="输入与诊断组件已就绪",
        )
    startup_device_errors: dict[str, str] = {}
    position_force_guard_monitor: PositionForceGuardMonitor | None = None
    position_admittance_shadow = None
    startup_device_runtime = StartupDeviceRuntime(
        capability_admission=capability_admission,
        startup_progress=startup_progress,
        startup_device_errors=startup_device_errors,
        control_ui=control_ui,
        dmtac_gripper_backend_type=DMTacGripperBackend,
        log=_log,
    )
    startup_device_label = startup_device_runtime.label
    startup_device_requirement = startup_device_runtime.requirement
    start_device = startup_device_runtime.start

    try:
        # From this point on the complete runtime shutdown ``finally`` owns
        # every resource registered above.  Keeping the pre-start registry
        # active until this exact boundary covers Ctrl+C after terminal cbreak
        # has been enabled but before device startup enters the main try body.
        _deactivate_startup_cleanup()
        start_device("arm.left.qpos", components.left_state_backend)
        start_device("arm.right.qpos", components.right_state_backend)
        for side, backend, via_rig in (
            ("left", components.left_hand_backend, components.left_hand_control_via_rig),
            ("right", components.right_hand_backend, components.right_hand_control_via_rig),
        ):
            if backend is None:
                continue
            if not via_rig:
                start_device(f"hand.{side}", backend)
                continue
            if start_device(f"hand.{side}", backend):
                continue
            # A failed optional hand must not make BimanualRig roll back the
            # already-independent arm startup.  Remove its command producer
            # and backend only for this runtime session.
            effector_name = f"{side}_hand"
            components.session.rig.backends.pop(effector_name, None)
            failed_retargeter = (
                components.left_hand_retargeter
                if side == "left"
                else components.right_hand_retargeter
            )
            if failed_retargeter in components.session.retargeters:
                components.session.retargeters.remove(failed_retargeter)
        session_started = False
        if startup_progress is not None:
            startup_progress.begin(
                "rig.session",
                label=startup_device_label("rig.session"),
                message="正在启动 tracker、retarget 与命令 Rig",
                requirement=startup_device_requirement("rig.session"),
            )
        try:
            components.session.start()
            session_started = True
            if startup_progress is not None:
                startup_progress.succeed(
                    "rig.session",
                    message="遥操核心 Rig 已启动",
                )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if startup_progress is not None:
                startup_progress.fail("rig.session", detail)
            if not capability_admission:
                if startup_progress is not None:
                    startup_progress.close()
                raise
            startup_device_errors["rig.session"] = detail
            _log(
                "Device",
                "WARNING: core teleop rig startup failed; GUI remains available "
                f"for status/retry: {detail}",
            )
            if control_ui is not None:
                control_ui.append_message("Device", f"遥操核心连接失败：{detail}")
        position_force_guard_requested = bool(
            _arm_control_mode_arg(args) == "position"
            and getattr(args, "position_force_guard", True)
        )
        if (
            _arm_control_mode_arg(args) == "position"
            and not position_force_guard_requested
        ):
            message = (
                "WARNING: real-position RM65 force emergency-stop monitor is "
                "disabled by --no-position-force-guard"
            )
            _log("PositionForceGuard", message)
            if control_ui is not None:
                control_ui.append_message("PositionForceGuard", message)
        if session_started and position_force_guard_requested:
            if startup_progress is not None:
                startup_progress.begin(
                    "arm.force_guard",
                    label=startup_device_label("arm.force_guard"),
                    message="正在验证双臂力传感器并启动锁存急停监控",
                    requirement="motion_safety",
                )
            try:
                state_backends = {
                    "left": components.left_state_backend,
                    "right": components.right_state_backend,
                }
                readers_by_side: dict[str, Callable[[], Mapping[str, Any]]] = {}
                for side in components.active_sides:
                    backend = state_backends.get(str(side))
                    reader = getattr(backend, "read_force_data", None)
                    if not callable(reader):
                        raise RuntimeError(
                            f"{side} RM65 state backend does not expose read_force_data"
                        )
                    readers_by_side[str(side)] = reader

                def position_force_guard_stop(
                    fault: PositionForceGuardFault,
                ) -> None:
                    # A force-guard fault is always a retained safety state,
                    # including while Calibration is waiting for the operator
                    # to press ``p``.  Do not turn the asynchronous safety
                    # callback into a process-quit request: the control loop
                    # consumes the latched fault, closes any active episode,
                    # and keeps command dispatch disabled in Hold.  This also
                    # leaves the GUI/device sessions available for inspection
                    # and a fresh operator-led Home after the stop is checked.
                    teleop_active = bool(
                        getattr(clutch_input, "_teleop_started", False)
                    )
                    stop_error: Exception | None = None
                    try:
                        _emergency_stop_real_arms(
                            components,
                            reason=(
                                "position_force_guard:"
                                f"{fault.reason}:{fault.side or 'all'}"
                            ),
                            now=fault.wall_time_sec,
                            raise_on_failure=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - report after both requests.
                        stop_error = exc
                    _log(
                        "PositionForceGuard",
                        "TRIPPED; bimanual emergency stop is latched; "
                        f"{fault.message}",
                        now=fault.wall_time_sec,
                    )
                    if control_ui is not None:
                        try:
                            if teleop_active and stop_error is None:
                                message = (
                                    "位控受力急停已触发，左右臂均已锁存停机；"
                                    "当前 episode 将结束，GUI 与设备连接保持；"
                                    "请人工调整机械臂位置后点击“开始遥操”重新采集 runtime Home。"
                                )
                            elif teleop_active:
                                message = (
                                    "位控受力急停已触发，但至少一侧 Arm_Stop 未确认；"
                                    "GUI 保持显示，禁止继续运动，必须检查设备并重启进程。"
                                )
                            else:
                                message = (
                                    "位控受力急停已触发，左右臂均已锁存停机；"
                                    "当前保持不下发指令，GUI/设备连接保持，"
                                    "请检查设备后重新采集 runtime Home。"
                                )
                            control_ui.append_message(
                                "PositionForceGuard",
                                message + f"\n{fault.message}",
                            )
                        except Exception as ui_exc:  # noqa: BLE001 - UI is not stop ownership.
                            _log(
                                "PositionForceGuard",
                                "force-trip UI notification failed; preserving the backend stop result: "
                                f"{type(ui_exc).__name__}: {ui_exc}",
                                now=fault.wall_time_sec,
                            )
                    if stop_error is not None:
                        _log(
                            "PositionForceGuard",
                            "CRITICAL: force guard tripped but one or more Arm_Stop "
                            f"requests failed: {type(stop_error).__name__}: {stop_error}",
                            now=fault.wall_time_sec,
                        )
                        raise stop_error

                position_force_guard_monitor = PositionForceGuardMonitor(
                    readers_by_side=readers_by_side,
                    emergency_stop=position_force_guard_stop,
                    config=_position_force_guard_config(args),
                )
                force_guard_snapshot = position_force_guard_monitor.start(
                    arm_stale_watchdog=False
                )
            except Exception as exc:
                monitor_fault = (
                    None
                    if position_force_guard_monitor is None
                    else position_force_guard_monitor.fault
                )
                if monitor_fault is None:
                    # Configuration/setup failures happen before the monitor
                    # can invoke its trip callback, but are still fail-closed.
                    _emergency_stop_real_arms(
                        components,
                        reason="position_force_guard_setup_failed",
                    )
                    clutch_input.should_quit = True
                    if startup_progress is not None:
                        startup_progress.fail(
                            "arm.force_guard",
                            f"{type(exc).__name__}: {exc}",
                            message="位控受力急停未能安全就绪，拒绝启动遥操",
                        )
                    # A setup failure has no latched runtime fault for the
                    # control loop to consume, so retain the historical
                    # fail-closed startup refusal in this case.
                    raise
                # ``PositionForceGuardMonitor.start`` raises when its first
                # valid sample is already over the force/torque limit.  That
                # is a runtime safety trip, not a component-construction
                # failure: the monitor has already latched the fault and
                # requested Arm_Stop.  Keep the GUI/device session alive so
                # the control loop can consume the fault, close any active
                # episode, and remain in Hold without tearing down the
                # process.
                _log(
                    "PositionForceGuard",
                    "startup force trip latched; entering the control loop in "
                    "Hold instead of terminating the GUI: "
                    f"{monitor_fault.message}",
                    now=monitor_fault.wall_time_sec,
                )
                if startup_progress is not None:
                    startup_progress.fail(
                        "arm.force_guard",
                        f"{type(exc).__name__}: {exc}",
                        message=(
                            "位控受力阈值已触发；双臂保持停机，"
                            "当前保持不下发指令并等待人工检查"
                        ),
                    )
                if control_ui is not None:
                    try:
                        control_ui.append_message(
                            "PositionForceGuard",
                            "启动阶段检测到受力超过安全阈值；"
                            "双臂已急停，当前不下发任何指令；"
                            "GUI/设备连接保持，请检查设备后重新采集 runtime Home。",
                        )
                    except Exception as ui_exc:  # noqa: BLE001 - UI is observational.
                        _log(
                            "PositionForceGuard",
                            "startup force-trip UI notification failed; "
                            "preserving the backend stop result: "
                            f"{type(ui_exc).__name__}: {ui_exc}",
                            now=monitor_fault.wall_time_sec,
                        )
            else:
                readings = force_guard_snapshot.get("last_reading_by_side", {})
                reading_text = []
                if isinstance(readings, Mapping):
                    for side in components.active_sides:
                        reading = readings.get(side, {})
                        if isinstance(reading, Mapping):
                            reading_text.append(
                                f"{side}=|F|{float(reading.get('force_norm_n', 0.0)):.2f}N/"
                                f"|T|{float(reading.get('torque_norm_nm', 0.0)):.2f}Nm"
                            )
                config = position_force_guard_monitor.config
                _log(
                    "PositionForceGuard",
                    "force_limits_armed=true stale_watchdog=deferred_until_camera_ready "
                    "fail_closed=true latched=true stop_scope=bimanual "
                    f"source={config.source} poll_hz={config.poll_hz:g} "
                    f"max_force_n={config.max_force_n:g} "
                    f"max_torque_nm={config.max_torque_nm:g} "
                    f"consecutive={config.consecutive_limit_samples} "
                    f"max_read_errors={config.max_consecutive_read_errors} "
                    f"stale_sec={config.stale_timeout_sec:g} "
                    + " ".join(reading_text),
                )
                if startup_progress is not None:
                    startup_progress.succeed(
                        "arm.force_guard",
                        message=(
                            "双臂受力阈值与读取错误急停已锁存；"
                            "数据陈旧 watchdog 将在相机启动后取得新鲜样本再启用"
                        ),
                    )
        if session_started and components.human_shadow_frontend is not None:
            if startup_progress is not None:
                startup_progress.begin(
                    "human_shadow",
                    label=startup_device_label("human_shadow"),
                    message="正在启动 shadow 前端",
                )
            try:
                components.human_shadow_frontend.start()
            except Exception as exc:
                if startup_progress is not None:
                    startup_progress.fail(
                        "human_shadow",
                        f"{type(exc).__name__}: {exc}",
                    )
                    startup_progress.close()
                raise
            else:
                if startup_progress is not None:
                    startup_progress.succeed("human_shadow")
        vive_source = _vive_pose_source_from_components(components)
        if session_started and vive_source is not None and vive_source.clock_sync_enabled:
            sync_wait_sec = (
                float(args.vive_clock_sync_startup_timeout_sec)
                if bool(args.vive_clock_sync_required)
                else 0.0
            )
            if startup_progress is not None:
                startup_progress.begin(
                    "vive.clock_sync",
                    label=startup_device_label("vive.clock_sync"),
                    message=(
                        f"正在等待同步样本，最长 {sync_wait_sec:.1f}s"
                        if sync_wait_sec > 0.0
                        else "正在检查时钟同步状态"
                    ),
                    requirement="motion_timing",
                )
            sync_ready = vive_source.wait_for_clock_sync(
                sync_wait_sec,
                min_samples=int(args.vive_clock_sync_min_samples),
            )
            sync_stats = vive_source.clock_sync_debug_stats()
            _log(
                "ViveClockSync",
                "startup "
                f"ready={sync_ready} required={bool(args.vive_clock_sync_required)} "
                f"stats={sync_stats}",
            )
            if startup_progress is not None:
                if sync_ready:
                    startup_progress.succeed(
                        "vive.clock_sync",
                        message="VIVE 时钟同步已就绪",
                    )
                elif not bool(args.vive_clock_sync_required):
                    startup_progress.warn(
                        "vive.clock_sync",
                        "clock synchronization is not ready",
                        message="检查完成；当前未同步，但配置允许继续",
                    )
                else:
                    startup_progress.fail(
                        "vive.clock_sync",
                        "clock synchronization is not ready",
                        message="必需的时钟同步未就绪",
                    )
            if bool(args.vive_clock_sync_required) and not sync_ready:
                message = (
                    "Required VIVE clock synchronization did not become ready before "
                    "real teleoperation startup. Check sender --clock-sync-* settings "
                    "and the separate UDP control port."
                )
                if not capability_admission:
                    if startup_progress is not None:
                        startup_progress.close()
                    raise RuntimeError(message)
                startup_device_errors["vive.clock_sync"] = message
                if control_ui is not None:
                    control_ui.append_message("ViveClockSync", message)
        if camera_sources:
            if capability_admission:
                startup_device_errors.update(
                    {
                        f"camera.{key}": value
                        for key, value in _start_camera_sources_best_effort(
                            camera_sources,
                            startup_progress=startup_progress,
                        ).items()
                    }
                )
            else:
                _start_camera_sources(
                    camera_sources,
                    startup_progress=startup_progress,
                )
        # Dataset writer startup is intentionally not part of this global
        # startup barrier.  ``initialize_legacy_writer`` can scan a populated
        # dataset for several seconds; collection will perform the same ready
        # handshake transactionally when the operator presses start.  Any
        # failure is then isolated to the collection capability and can be
        # retried with a fresh recorder/process instance.
        tactile_capacity_summary = _tactile_storage_capacity_summary(args)
        if tactile_capacity_summary is not None:
            _log("TactileStorage", tactile_capacity_summary)
        if getattr(args, "dmtac_startup_error", ""):
            startup_device_errors["dmtac.assignment"] = str(args.dmtac_startup_error)
        if getattr(args, "angle_trigger_startup_error", ""):
            startup_device_errors["angle_trigger"] = str(args.angle_trigger_startup_error)
        if getattr(args, "q6_startup_error", ""):
            startup_device_errors["gripper.input"] = str(args.q6_startup_error)
        if getattr(args, "camera_startup_error", ""):
            startup_device_errors["camera.config"] = str(args.camera_startup_error)
        if getattr(args, "dataset_startup_error", ""):
            startup_device_errors["dataset.writer"] = str(args.dataset_startup_error)
        if bool(getattr(args, "foot_pad_requested", False)) and getattr(
            args, "foot_pad_startup_error", ""
        ):
            startup_device_errors["foot_pad"] = str(args.foot_pad_startup_error)
        if startup_progress is not None:
            startup_progress.close()

        if bool(getattr(args, "idle_tactile_compression", False)):
            tactile_bundle_min_free_gb = float(
                getattr(args, "tactile_bundle_min_free_gb", 0.0) or 0.0
            )
            idle_tactile_compression = RealTeleopIdleTactileCompression(
                enabled=True,
                tactile_bundle_root=(
                    getattr(args, "tactile_bundle_root", "") or None
                ),
                tactile_bundle_min_free_bytes=(
                    int(tactile_bundle_min_free_gb * (1024**3))
                    if tactile_bundle_min_free_gb > 0.0
                    else None
                ),
            )
            _log(
                "TactileCompression",
                "idle lossless compression enabled; admission requires Hold/pre-Home "
                "and no active/countdown/finalizing formal episode",
            )

        def tick_idle_tactile_compression(
            *,
            episode_countdown_active: bool = False,
        ) -> None:
            if idle_tactile_compression is None:
                return
            idle_tactile_compression.tick_runtime(
                dataset_recorder=dataset_recorder,
                teleop_started=bool(
                    getattr(clutch_input, "_teleop_started", False)
                ),
                hold_active=bool(clutch_input.hold_active()),
                episode_countdown_active=bool(episode_countdown_active),
                startup_and_devices_ready=True,
                log=_log,
            )

        def begin_idle_tactile_quiesce(reason: str) -> None:
            if idle_tactile_compression is not None:
                idle_tactile_compression.begin_latency_sensitive_transition(
                    reason=reason
                )

        def require_idle_tactile_quiesced(reason: str) -> None:
            if idle_tactile_compression is None:
                return
            idle_tactile_compression.require_safe_transition(
                reason=reason,
                dataset_root=(
                    None
                    if dataset_recorder is None
                    else getattr(dataset_recorder, "dataset_root", None)
                ),
                tactile_bundle_root=getattr(
                    args, "tactile_bundle_root", ""
                )
                or None,
                integrity_enabled=bool(
                    getattr(args, "tactile_artifact_integrity", False)
                ),
                log=_log,
                timeout_sec=2.0,
            )

        def current_dataset_recorder() -> object | None:
            return dataset_recorder

        def set_current_dataset_recorder(value: object | None) -> None:
            nonlocal dataset_recorder
            require_idle_tactile_quiesced("dataset_recorder_rebind")
            dataset_recorder = value

        def reconfigure_dataset_recorder_for_selection() -> None:
            """Apply an idle task/root change to the live recorder immediately."""

            nonlocal dataset_recorder
            current = dataset_recorder
            if current is None:
                return
            require_idle_tactile_quiesced("dataset_selection_reconfigure")
            try:
                current.reconfigure_for_current_args()
            except BaseException:
                # ``reconfigure_for_current_args`` may already have closed
                # the old recorder before a new candidate fails its ready
                # handshake.  Do not leave that one-shot object published for
                # a later retry.
                process_worker = getattr(current, "process_worker", None)
                writer_pool = getattr(current, "writer_pool", None)
                candidate_is_unusable = bool(
                    getattr(current, "_lifecycle_closed", False)
                    or getattr(process_worker, "_closed", False)
                    or getattr(writer_pool, "_closed", False)
                )
                if candidate_is_unusable:
                    dataset_recorder = None
                    try:
                        set_current_dataset_recorder(None)
                    except BaseException:
                        pass
                    try:
                        current.close()
                    except BaseException:
                        pass
                raise
            args.dataset_startup_error = ""
            startup_device_errors.pop("dataset.writer", None)
            publish_runtime_hardware_status()

        bind_reconfigure = getattr(
            clutch_input,
            "bind_dataset_reconfigure_callback",
            None,
        )
        if callable(bind_reconfigure):
            bind_reconfigure(reconfigure_dataset_recorder_for_selection)

        def current_foot_pad_input() -> object | None:
            return foot_pad_input

        def set_current_foot_pad_input(value: object | None) -> None:
            nonlocal foot_pad_input
            foot_pad_input = value

        def publish_retry_camera_status() -> None:
            # A manual reconnect changes live camera health, so the startup
            # self-check snapshot must be replaced as well as the camera cards.
            # Do not recursively poll operator input from inside the retry
            # request handler.
            _run_camera_startup_self_check(
                camera_sources=camera_sources,
                clutch_input=clutch_input,
                preview_camera_sources=preview_camera_sources,
                camera_previewer=camera_previewer,
                poll_input=False,
            )

        device_retry_runtime = build_device_retry_runtime(
            args=args,
            components=components,
            control_ui=control_ui,
            clutch_input=clutch_input,
            camera_sources=camera_sources,
            vive_source=vive_source,
            startup_device_errors=startup_device_errors,
            position_force_guard_requested=position_force_guard_requested,
            dataset_recorder_provider=current_dataset_recorder,
            set_dataset_recorder=set_current_dataset_recorder,
            foot_pad_input_provider=current_foot_pad_input,
            set_foot_pad_input=set_current_foot_pad_input,
            dataset_recorder_type=RealTeleopDatasetRecorder,
            dmtac_gripper_backend_type=DMTacGripperBackend,
            foot_pad_input_type=FootPadSpecialArmInput,
            foot_pad_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            has_dmtac_gripper_backend=_has_dmtac_gripper_backend,
            promote_angle_trigger_gripper_input_to_q6=(
                _promote_angle_trigger_gripper_input_to_q6
            ),
            log=_log,
            publish_camera_status=publish_retry_camera_status,
        )
        hardware_requirement = device_retry_runtime.hardware_requirement
        hardware_status_item = device_retry_runtime.hardware_status_item
        collection_startup_blockers = (
            device_retry_runtime.collection_startup_blockers
        )
        publish_runtime_hardware_status = (
            device_retry_runtime.publish_runtime_hardware_status
        )
        sync_foot_pad_runtime_status = (
            device_retry_runtime.sync_foot_pad_runtime_status
        )
        retry_failed_devices = device_retry_runtime.retry_failed_devices
        # Startup progress and the steady-state hardware list are separate UI
        # projections.  Publish once even when both arm qpos reads succeed so
        # isolated Q6/input failures are visible before the operator presses
        # the manual retry button.
        publish_runtime_hardware_status()

        def set_current_visualization_state(
            previewer: object | None,
            camera_active: bool,
            visualizer: object | None,
        ) -> None:
            nonlocal camera_previewer, camera_preview_active, tactile_visualizer
            camera_previewer = previewer
            camera_preview_active = bool(camera_active)
            tactile_visualizer = visualizer

        visualization_runtime = build_visualization_runtime(
            args=args,
            components=components,
            clutch_input=clutch_input,
            control_ui=control_ui,
            preview_camera_sources=preview_camera_sources,
            camera_preview_rate=camera_preview_rate,
            tactile_preview_rate=tactile_preview_rate,
            initial_camera_previewer=camera_previewer,
            initial_camera_preview_active=camera_preview_active,
            initial_tactile_visualizer=tactile_visualizer,
            set_outer_state=set_current_visualization_state,
            retry_failed_devices=retry_failed_devices,
            start_camera_sources=_start_camera_sources,
            camera_previewer_type=CameraPreviewer,
            camera_preview_config_type=CameraPreviewConfig,
            build_tactile_visualizer=_build_tactile_visualizer,
            enable_gui_tactile_solving=_enable_gui_tactile_solving,
            tactile_solving_active=_tactile_solving_active,
            tactile_visualization_available=_tactile_visualization_available,
            dmtac_visualizer_types=(
                DMTacDistributedForceVisualizer,
                AsyncDMTacDistributedForceVisualizer,
            ),
            dmtac_visualization_snapshot=(
                dmtac_visualization_snapshot_from_backends
            ),
            tactile_snapshot=tactile_by_side_from_backends,
            gripper_backend_for_side=_gripper_backend_for_side,
            hand_backend_for_side=_hand_backend_for_side,
            log=_log,
        )
        close_camera_preview = visualization_runtime.close_camera_preview
        open_camera_preview = visualization_runtime.open_camera_preview
        close_tactile_preview = visualization_runtime.close_tactile_preview
        open_tactile_preview = visualization_runtime.open_tactile_preview
        handle_startup_visualization_requests_base = (
            visualization_runtime.handle_startup_requests
        )

        def handle_startup_visualization_requests(now: float) -> None:
            """Service optional UI requests while runtime Home is still armed."""

            handle_startup_visualization_requests_base(now)
            startup_progress_runtime.handle_pending_selections(now=now)

        update_startup_visualizations = (
            visualization_runtime.update_startup_visualizations
        )




        if control_ui is not None:
            configured_status_keys = {
                "arm.left.qpos",
                "arm.right.qpos",
                *(f"camera.{key}" for key in camera_sources),
            }
            if components.left_hand_backend is not None:
                configured_status_keys.add("hand.left")
            if components.right_hand_backend is not None:
                configured_status_keys.add("hand.right")
            configured_status_keys.add("dataset.writer")
            if position_force_guard_requested:
                configured_status_keys.add("arm.force_guard")
            if bool(getattr(args, "foot_pad_requested", False)):
                configured_status_keys.add("foot_pad")
            configured_status_keys.update(startup_device_errors)
            control_ui.publish(
                stage=(
                    "设备连接检查完成"
                    if not startup_device_errors
                    else f"设备连接检查完成 · {len(startup_device_errors)} 项未就绪"
                ),
                control_hint=(
                    "全部配置设备已完成启动检查。"
                    if not startup_device_errors
                    else "GUI 已可用；请查看未就绪设备，可在修复后点击“重连失败设备”。"
                ),
                hardware_status={
                    key: hardware_status_item(key)
                    for key in sorted(configured_status_keys)
                },
                active_controls=clutch_input._current_active_controls(),
            )



        clutch_input.set_visualization_state(
            camera_active=False,
            tactile_active=tactile_visualizer is not None,
            status="",
        )
        clutch_input.publish_camera_status(
            camera_sources=camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            force=True,
        )
        _run_camera_startup_self_check(
            camera_sources=camera_sources,
            clutch_input=clutch_input,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
        )
        if clutch_input.should_quit:
            return
        if position_force_guard_monitor is not None:
            startup_force_fault = position_force_guard_monitor.fault
            if startup_force_fault is not None:
                # A trip may have happened while camera startup was in
                # progress.  Do not replace the latched safety transition with
                # a startup exception; the control loop will consume it before
                # any session/rig dispatch.
                _log(
                    "PositionForceGuard",
                    "stale watchdog remains disarmed after a latched force trip; "
                    "motion admission stays blocked",
                    now=startup_force_fault.wall_time_sec,
                )
            else:
                try:
                    force_guard_snapshot = (
                        position_force_guard_monitor.arm_stale_watchdog()
                    )
                except Exception as exc:
                    # A force trip can race this arm call.  If the monitor now
                    # owns a fault, retain the process and let the control loop
                    # perform the normal episode-stop/Hold transition; only a
                    # genuine stale-watchdog setup failure aborts startup.
                    raced_force_fault = position_force_guard_monitor.fault
                    if raced_force_fault is None:
                        _log(
                            "PositionForceGuard",
                            "stale watchdog failed to arm after camera startup; "
                            f"motion remains rejected: {type(exc).__name__}: {exc}",
                        )
                        if control_ui is not None:
                            control_ui.append_message(
                                "PositionForceGuard",
                                "相机启动后无法取得双侧新鲜力数据；"
                                "已拒绝归位/遥操并锁存双臂停机。"
                                f"\n{type(exc).__name__}: {exc}",
                            )
                        raise
                    _log(
                        "PositionForceGuard",
                        "force trip raced stale-watchdog arming; "
                        "motion admission remains blocked: "
                        f"{raced_force_fault.message}",
                        now=raced_force_fault.wall_time_sec,
                    )
                else:
                    readings = force_guard_snapshot.get("last_reading_by_side", {})
                    reading_text: list[str] = []
                    if isinstance(readings, Mapping):
                        for side in components.active_sides:
                            reading = readings.get(side, {})
                            if isinstance(reading, Mapping):
                                reading_text.append(
                                    f"{side}=|F|"
                                    f"{float(reading.get('force_norm_n', 0.0)):.2f}N/"
                                    f"|T|{float(reading.get('torque_norm_nm', 0.0)):.2f}Nm"
                                )
                    _log(
                        "PositionForceGuard",
                        "stale_watchdog=armed_after_fresh_post_camera_samples "
                        "motion_admission=ready "
                        + " ".join(reading_text),
                    )
                    if control_ui is not None:
                        control_ui.append_message(
                            "PositionForceGuard",
                            "相机启动后双侧新鲜力数据已确认；"
                            "位控受力与读取陈旧急停均已锁存就绪。",
                        )
        if camera_sources:
            _log(
                "Camera",
                f"configured cameras={sorted(camera_sources)} "
                f"config={getattr(args, 'camera_config', '')} "
                f"selected={sorted(preview_camera_sources)}",
            )
        if bool(getattr(args, "vis_cam", False)) and preview_camera_sources:
            try:
                _start_camera_sources(preview_camera_sources)
                camera_previewer = CameraPreviewer(CameraPreviewConfig())
                if camera_previewer.ensure_available():
                    camera_preview_active = True
                    clutch_input.set_visualization_state(
                        camera_active=True,
                        status=f"图像窗口已打开: {', '.join(sorted(preview_camera_sources))}",
                    )
                    _log("Camera", f"preview active cameras={sorted(preview_camera_sources)}")
                else:
                    reason = camera_previewer.disabled_reason or "display unavailable"
                    camera_previewer.close()
                    camera_previewer = None
                    clutch_input.set_visualization_state(camera_active=False, status=reason)
                    _log("Camera", f"preview disabled: {reason}")
            except Exception as exc:  # noqa: BLE001
                if camera_previewer is not None:
                    camera_previewer.close()
                    camera_previewer = None
                clutch_input.set_visualization_state(camera_active=False, status=f"图像窗口启动失败: {exc}")
                _log("Camera", f"preview startup failed: {type(exc).__name__}: {exc}")
        visualization_runtime.sync_state(
            camera_previewer,
            camera_preview_active,
            tactile_visualizer,
        )
        initial_qpos_sync = _sync_preview_from_real_state_real(
            components,
            required=not capability_admission,
        )
        if not all(initial_qpos_sync.values()):
            missing_sides = [side for side, synced in initial_qpos_sync.items() if not synced]
            for side in missing_sides:
                startup_device_errors.setdefault(
                    f"arm.{side}.qpos",
                    "hardware returned no fresh current_qpos during startup sync",
                )
            message = (
                "机械臂 qpos 尚未就绪；GUI 与无关功能保持可用，归位/遥操/数采需重连后再执行: "
                + ", ".join(missing_sides)
            )
            _log("ArmState", message)
            if control_ui is not None:
                control_ui.append_message("ArmState", message)
            publish_runtime_hardware_status()
        _sync_real_execution_step_limits(components)

        for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
            if not arm_ip or logical_side not in components.active_sides:
                continue
            _log("ArmMode", f"{source_name}={arm_ip} -> logical_{logical_side}_arm")
        _log(
            "InputMapping",
            "vive_tracker_sides "
            f"swap={bool(getattr(args, 'swap_operator_sides', False))} "
            f"mapping={{"
            + ", ".join(
                f"logical_{side}:{_tracker_source_side_for_logical_side(args, side)}"
                for side in components.active_sides
            )
            + "} arms_hands_serial_profiles_unchanged fixed_after_start",
        )
        if _has_dmtac_gripper_backend(components):
            policy = components.gripper_policy
            _log(
                "Gripper",
                "DMTAC backend "
                f"q6_control_sides={[side for side in components.active_sides if _gripper_retargeter_for_side(components, side) is not None]} "
                f"backend_sides={[side for side in components.active_sides if _gripper_backend_for_side(components, side) is not None]} "
                f"left_q6_port={getattr(args, 'left_hand_port', '') or None} "
                f"right_q6_port={getattr(args, 'right_hand_port', '') or None} "
                f"dmtac_config={getattr(args, 'dmtac_config', 'latest')} "
                f"loaded_sides={list(getattr(args, 'dmtac_config_loaded_sides', []) or [])} "
                f"source_dim={policy.get('source_dim')} output_dim={policy.get('output_dim')} "
                f"invert={policy.get('invert')} max_action={float(policy.get('max_action', GRIPPER_ACTION_MAX)):.3f} "
                f"speed={float(policy.get('speed', 0.2)):.3f} torque={float(policy.get('torque', 0.2)):.3f} "
                "dispatch=gui_gated",
            )
        elif components.left_hand_retargeter is not None or components.right_hand_retargeter is not None:
            _log(
                "HandTeleop",
                "enabled "
                f"left_port={getattr(args, 'left_hand_port', '') or None} "
                f"right_port={getattr(args, 'right_hand_port', '') or None} "
                f"left_hand_ip={args.left_hand_ip or None} "
                f"right_hand_ip={args.right_hand_ip or None} "
                f"joint_order={list(getattr(args, 'hand_joint_order', DEFAULT_INSPIRE_HAND_JOINT_ORDER))} "
                f"invert={bool(getattr(args, 'hand_invert', True))} "
                f"alpha={float(getattr(args, 'hand_command_alpha', 1.0)):.3f} "
                f"telemetry_hz={float(getattr(args, 'hand_telemetry_hz', 30.0)):.1f} "
                f"tactile_regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', 'all'))) or []} "
                f"dispatch=rig_parallel",
            )
        elif components.left_hand_backend is not None or components.right_hand_backend is not None:
            _log(
                "HandTeleop",
                "telemetry-only "
                f"left_hand_ip={args.left_hand_ip or None} "
                f"right_hand_ip={args.right_hand_ip or None} "
                f"telemetry_hz={float(getattr(args, 'hand_telemetry_hz', 30.0)):.1f} "
                f"tactile_regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', 'all'))) or []}",
            )
        else:
            _log("HandTeleop", "disabled; arm teleop path remains unchanged.")
        if args.teleop_hardware_ip or args.teleop_hardware_port:
            _log(
                "ReservedIO",
                "teleop-hardware endpoint is still reserved on this entrypoint and remains inactive.",
            )
        if components.loaded_initial_alignment is not None:
            _log(
                "InitialAlignment",
                f"source={args.load_initial_alignment} "
                f"left_q={_qpos_summary(components.loaded_initial_alignment.get_side('left').arm_qpos if components.loaded_initial_alignment.get_side('left') is not None else None)} "
                f"right_q={_qpos_summary(components.loaded_initial_alignment.get_side('right').arm_qpos if components.loaded_initial_alignment.get_side('right') is not None else None)}",
            )
            start_check_status = _run_initial_alignment_start_check(args, components)
            if start_check_status == "passed":
                _log(
                    "InitialAlignment",
                    "startup joint check passed; runtime home capture at 'p' will still use the live real-arm pose.",
                )
            elif start_check_status == "ignored_mismatch":
                _log(
                    "InitialAlignment",
                    "startup joint mismatch was ignored by CLI; runtime home capture at 'p' will still use the live real-arm pose.",
                )
        _emit_tracker_axis_mapping_logs(components)
        _log(
            "IK",
            f"objective_mode={components.session.calibration.metadata.get('ik_objective_mode')} "
            f"tuning={components.session.calibration.metadata.get('ik_tuning_source')}",
        )
        _log(
            "ArmMode",
            f"execution_profile={args.arm_execution_config.resolved_profile} "
            f"requested_profile={args.arm_execution_config.requested_profile} "
            f"control_mode={_arm_control_mode_arg(args)} "
            f"impedance_mode={_impedance_mode_arg(args)} "
            f"strict_tool_profile={_strict_tool_profile_arg(args)} "
            f"force_tare_sec={_force_tare_sec_arg(args):.2f} "
            f"state_read_retries={args.rm65_hardware_read_retries} "
            f"retry_delay_sec={args.rm65_hardware_read_retry_delay_sec:.3f} "
            f"io_threads={max(1, int(args.arm_io_threads))} "
            f"sample_hz={float(args.hz):.1f} "
            f"outer_control_hz={_effective_outer_control_hz(args):.1f} "
            f"arm_control_hz={_effective_arm_control_hz(args):.1f} "
            f"minimum_control_hz={args.minimum_control_hz:.1f} "
            f"arm_max_angvel_rad_s={np.round(np.asarray(args.arm_max_angvel, dtype=float), 3).tolist()} "
            f"arm_max_angvel_effective_rad_s={np.round(np.asarray(args.arm_max_angvel, dtype=float) * (0.2 if bool(getattr(args, 'strict_degree_limit', False)) else 1.0), 3).tolist()} "
            f"position_max_command_step_deg={getattr(args, 'position_max_command_step_deg', None)} "
            f"position_target_velocity_filter_tau_sec={getattr(args, 'position_target_velocity_filter_tau_sec', DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC)} "
            f"strict_degree_limit={'on' if bool(getattr(args, 'strict_degree_limit', False)) else 'off'} "
            f"actual_dt_scaling={'on' if args.arm_step_dt_scaling else 'off'} "
            f"dt_alpha={args.arm_step_dt_alpha:.2f} "
            f"dt_max_scale={args.arm_step_dt_max_scale:.2f} "
            f"dt_stall_reset_sec={args.arm_step_dt_stall_reset_sec:.3f}",
        )
        _log(
            "ControlDebug",
            f"level={args.control_debug_config.level} "
            f"root={args.control_debug_config.root} "
            f"queue_size={args.control_debug_config.queue_size} "
            f"flush_every={args.control_debug_config.flush_every} "
            "observational_only=true",
        )
        if bool(getattr(args, "no_debug_disk_write", False)):
            _log(
                "DebugDiskWrite",
                "disabled by --no-debug-disk-write; formal dataset collection is unaffected",
            )
        if components.temporary_safety_limiter is not None:
            temp_config = components.temporary_safety_limiter.config
            _log(
                "TempSafety",
                (
                    "temporary dual-arm output limiter enabled "
                    f"max_xyz_step_m={temp_config.max_xyz_step_m:.3f} "
                    f"max_rot_step_deg={temp_config.max_rot_step_deg:.1f} "
                    f"max_joint_step_deg={temp_config.max_joint_step_deg:.1f} "
                    f"command_scale={temp_config.command_scale:.2f}; "
                    "disable with --no-dual-arm-temporary-safety"
                ),
            )
        else:
            disabled_reason = components.session.calibration.metadata.get(
                "dual_arm_temporary_safety_disabled_reason"
            )
            reason_text = "" if not disabled_reason else f" reason={disabled_reason}"
            _log("TempSafety", f"temporary dual-arm output limiter disabled{reason_text}")
        if components.collision_guard_setup_error is not None:
            _log(
                "CollisionGuard",
                "disabled automatically; "
                f"{components.collision_guard_setup_error}. "
                "Falling back to two independent arm retargeters. "
                "Use --no-collision to disable this explicitly.",
            )
        if components.dual_arm_retargeter is not None:
            _log(
                "CollisionGuard",
                "enabled "
                f"scene_profile={args.collision_scene_profile} "
                f"pairs={len(components.dual_arm_retargeter.profile.collision_pairs)} "
                f"mode={args.collision_avoidance_mode}",
            )
        else:
            _log("CollisionGuard", "disabled; using two independent arm retargeters.")
        _log_dual_arm_readiness(args, components)
        _emit_collision_proxy_logs(components)
        debug_timing_modules = _debug_timing_modules(args)
        debug_timing_console_writer = (
            AsyncDebugTimingConsole(stream=sys.stdout)
            if bool(debug_timing_modules.console)
            else None
        )
        debug_timing_state = RealDebugTimingState(
            enabled=bool(debug_timing_modules.console),
            threshold_sec=float(getattr(args, "debug_timing_threshold_sec", 0.025)),
            report_interval_sec=float(getattr(args, "debug_timing_report_interval_sec", 1.0)),
            console_writer=debug_timing_console_writer,
        )
        ik_debug_profile_base_enabled = bool(debug_timing_modules.ik_profile)
        debug_ik_0827_enabled = bool(getattr(args, "debug_ik_0827", False))
        ik_debug_log_recorder = IkDebugLogRecorder(
            root_dir=(
                getattr(args, "_runtime_artifact_ik_debug_root", "") or None
            ),
        )
        if debug_ik_0827_enabled:
            if bool(getattr(args, "no_debug_disk_write", False)):
                _log(
                    "DebugIK0827",
                    "extra IK investigation log disabled by --no-debug-disk-write",
                )
            else:
                try:
                    # The canonical runtime still receives the historical
                    # control_station compatibility wrapper, whose
                    # constructor intentionally has the old public signature.
                    # Set this optional recorder-only prefix after construction
                    # so that wrapper and legacy callers remain source-
                    # compatible while the opt-in file is clearly identified.
                    ik_debug_log_recorder.filename_prefix = "ik_debug_0827"
                    ik_debug_0827_path = ik_debug_log_recorder.start(now=time.time())
                    if (
                        components.dual_arm_retargeter is not None
                        and len(components.active_sides) > 1
                    ):
                        # The existing solver profile is read-only timing
                        # instrumentation.  Enable it only for this explicit
                        # investigation run; the normal CLI default remains
                        # untouched and the GUI toggle can still restore its
                        # original setting.
                        components.dual_arm_retargeter.config.debug_profile = True
                        components.dual_arm_retargeter.ik_solver.debug_profile = True
                    ik_debug_log_recorder.write_event(
                        "config",
                        {
                            "recording_profile": "ik-0827",
                            "source": "--debug-ik-0827",
                            "auto_started": True,
                            "log_mode": (
                                "full"
                                if bool(getattr(args, "ik_debug_log_full", False))
                                else "compact"
                            ),
                            "outer_control_hz": float(_effective_outer_control_hz(args)),
                            "arm_control_hz": float(_effective_arm_control_hz(args)),
                            "ik_profile_enabled": True,
                            "formal_dataset_debug_metadata": False,
                        },
                        now=time.time(),
                    )
                    clutch_input.set_ik_debug_logging_state(
                        active=True,
                        path=str(ik_debug_0827_path),
                    )
                    _log(
                        "DebugIK0827",
                        f"extra control-chain log started path={ik_debug_0827_path}",
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostics must not block teleop.
                    if ik_debug_log_recorder.active:
                        ik_debug_log_recorder.stop(
                            reason="debug_ik_0827_start_failed",
                            now=time.time(),
                        )
                    _log(
                        "DebugIK0827",
                        f"extra log disabled: {type(exc).__name__}: {exc}",
                    )
        debug_timing_log_recorder = DebugTimingLogRecorder(
            root_dir=(
                getattr(args, "_runtime_artifact_debug_timing_root", "")
                or None
            ),
            heartbeat_sec=max(0.1, float(getattr(args, "debug_timing_report_interval_sec", 1.0))),
            sample_hz=float(getattr(args, "debug_timing_log_sample_hz", 0.0)),
        )
        action_debug_log_recorder = ActionDebugLogRecorder(
            root_dir=_expand_cli_path(getattr(args, "debug_action_log_root", DEFAULT_ACTION_DEBUG_LOG_ROOT)),
            max_queue_size=int(getattr(args, "debug_action_queue_size", 2048)),
            flush_every=int(getattr(args, "debug_action_flush_every", 1)),
            sample_hz=float(getattr(args, "debug_action_sample_hz", 0.0)),
        )
        canfd_runtime_config = _canfd_follow_runtime_config(args)
        action_debug_position_canfd_passthrough = bool(
            _arm_control_mode_arg(args) == "position"
            and canfd_runtime_config["position_background_streaming_enabled"]
        )
        action_debug_temporary_safety = _effective_dual_arm_temporary_safety(
            args,
            position_canfd_passthrough=action_debug_position_canfd_passthrough,
        )
        main_loop_watchdog = MainLoopStallWatchdog(
            enabled=bool(debug_timing_modules.watchdog),
            timeout_sec=max(5.0, float(debug_timing_state.report_interval_sec) * 5.0),
            report_interval_sec=max(5.0, float(debug_timing_state.report_interval_sec) * 5.0),
        )
        if debug_timing_modules.light_log:
            try:
                debug_timing_path = debug_timing_log_recorder.start(now=time.time())
                debug_timing_log_recorder.write_event(
                    "config",
                    {
                        "modules": list(debug_timing_modules.enabled_names()),
                        "legacy_all": bool(debug_timing_modules.legacy_all),
                        "threshold_sec": float(debug_timing_state.threshold_sec),
                        "report_interval_sec": float(debug_timing_state.report_interval_sec),
                        "sample_hz": float(
                            getattr(args, "debug_timing_log_sample_hz", 0.0)
                        ),
                        "console": bool(debug_timing_modules.console),
                        "rm65_backend_debug_timing": bool(debug_timing_modules.rm65_backend),
                        "ik_profile": bool(debug_timing_modules.ik_profile),
                        "main_loop_watchdog_enabled": bool(debug_timing_modules.watchdog),
                        "main_loop_watchdog_timeout_sec": float(main_loop_watchdog.timeout_sec),
                    },
                    now=time.time(),
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics must not block teleop.
                debug_timing_path = None
                _log(
                    "DebugTiming",
                    f"light log disabled: {type(exc).__name__}: {exc}",
                )
        else:
            debug_timing_path = None
        if bool(getattr(args, "debug_action", False)):
            try:
                action_debug_path = action_debug_log_recorder.start(now=time.time())
                action_debug_log_recorder.write_event(
                    "config",
                    {
                        "standalone": True,
                        **_arm_execution_profile_metadata(args),
                        **_position_ik_iteration_metadata(args),
                        "control_debug": args.control_debug_config.to_metadata(),
                        "dataset_metadata_also_enabled_when_recording": True,
                        "no_collection": bool(getattr(args, "no_collection", False)),
                        "no_record": bool(getattr(args, "no_record", True)),
                        "sample_hz": float(getattr(args, "debug_action_sample_hz", 0.0)),
                        "hz": float(getattr(args, "hz", 0.0) or 0.0),
                        "arm_control_hz": float(_effective_arm_control_hz(args)),
                        "safe_debug": bool(getattr(args, "safe_debug", False)),
                        "disable_hold": bool(getattr(args, "disable_hold", False)),
                        "strict_degree_limit": bool(getattr(args, "strict_degree_limit", False)),
                        "qpos_limit_scale": (
                            0.2 if bool(getattr(args, "strict_degree_limit", False)) else 1.0
                        ),
                        "arm_canfd_follow": bool(_effective_arm_canfd_follow(args)),
                        "arm_canfd_follow_arg": getattr(args, "arm_canfd_follow", None),
                        "position_post_ik_qpos_smoothing_alpha": getattr(
                            args,
                            "position_post_ik_qpos_smoothing_alpha",
                            None,
                        ),
                        "position_post_ik_qpos_smoothing_alpha_effective": float(
                            _effective_position_post_ik_qpos_smoothing_alpha(args)
                        ),
                        "arm_follow_qpos_alpha": getattr(args, "arm_follow_qpos_alpha", None),
                        "arm_follow_qpos_alpha_effective": float(
                            _effective_arm_follow_qpos_alpha(args)
                        ),
                        "arm_canfd_follow_stream_hz": getattr(
                            args,
                            "arm_canfd_follow_stream_hz",
                            None,
                        ),
                        "arm_canfd_follow_warning_ratio": getattr(
                            args,
                            "arm_canfd_follow_warning_ratio",
                            None,
                        ),
                        "position_background_streaming_enabled": bool(
                            canfd_runtime_config["position_background_streaming_enabled"]
                        ),
                        "position_background_streaming_hz": float(
                            canfd_runtime_config["position_background_streaming_hz"]
                        ),
                        "canfd_follow_warning_ratio_effective": float(
                            canfd_runtime_config["canfd_follow_warning_ratio"]
                        ),
                        "arm_canfd_trajectory_mode": getattr(args, "arm_canfd_trajectory_mode", None),
                        "canfd_trajectory_mode_effective": int(canfd_runtime_config["canfd_trajectory_mode"]),
                        "arm_canfd_radio": getattr(args, "arm_canfd_radio", None),
                        "canfd_radio_effective": int(canfd_runtime_config["canfd_radio"]),
                        "arm_disable_force_admittance": bool(
                            getattr(args, "arm_disable_force_admittance", False)
                        ),
                        "arm_max_angvel_rad_s": [
                            float(value)
                            for value in np.asarray(
                                getattr(args, "arm_max_angvel", ()),
                                dtype=float,
                            ).reshape(-1).tolist()
                        ],
                        "arm_max_angvel_effective_rad_s": [
                            float(value)
                            for value in (
                                np.asarray(
                                    getattr(args, "arm_max_angvel", ()),
                                    dtype=float,
                                ).reshape(-1)
                                * (
                                    0.2
                                    if bool(getattr(args, "strict_degree_limit", False))
                                    else 1.0
                                )
                            ).tolist()
                        ],
                        "position_max_command_step_deg": getattr(
                            args,
                            "position_max_command_step_deg",
                            None,
                        ),
                        "position_target_velocity_filter_tau_sec": float(
                            getattr(
                                args,
                                "position_target_velocity_filter_tau_sec",
                                DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
                            )
                        ),
                        "arm_step_dt_scaling": bool(
                            getattr(args, "arm_step_dt_scaling", False)
                        ),
                        "arm_step_dt_alpha": float(
                            getattr(args, "arm_step_dt_alpha", 0.0) or 0.0
                        ),
                        "arm_step_dt_max_scale": float(
                            getattr(args, "arm_step_dt_max_scale", 0.0) or 0.0
                        ),
                        "arm_step_dt_max_scale_auto": bool(
                            getattr(args, "arm_step_dt_max_scale_auto", False)
                        ),
                        "arm_step_dt_stall_reset_sec": float(
                            getattr(args, "arm_step_dt_stall_reset_sec", 0.0) or 0.0
                        ),
                        "arm_singularity_protection": bool(
                            getattr(args, "arm_singularity_protection", True)
                        ),
                        "dual_arm_temporary_safety_requested": bool(
                            action_debug_temporary_safety["requested"]
                        ),
                        "dual_arm_temporary_safety_enabled_effective": bool(
                            action_debug_temporary_safety["enabled"]
                        ),
                        "dual_arm_temporary_safety_disabled_reason": action_debug_temporary_safety[
                            "disabled_reason"
                        ],
                        "dual_arm_temp_max_xyz_step_m": float(
                            getattr(args, "dual_arm_temp_max_xyz_step_m", 0.0) or 0.0
                        ),
                        "dual_arm_temp_max_rot_step_deg": float(
                            getattr(args, "dual_arm_temp_max_rot_step_deg", 0.0) or 0.0
                        ),
                        "dual_arm_temp_max_joint_step_deg": float(
                            getattr(args, "dual_arm_temp_max_joint_step_deg", 0.0) or 0.0
                        ),
                        "dual_arm_temp_max_joint_step_effective_deg": float(
                            (getattr(args, "dual_arm_temp_max_joint_step_deg", 0.0) or 0.0)
                            * (
                                0.2
                                if bool(getattr(args, "strict_degree_limit", False))
                                else 1.0
                            )
                        ),
                        "dual_arm_temp_command_scale": float(
                            getattr(args, "dual_arm_temp_command_scale", 0.0) or 0.0
                        ),
                    },
                    now=time.time(),
                )
                _log("DebugAction", f"standalone log path={action_debug_path}")
            except Exception as exc:  # noqa: BLE001 - diagnostics must not block teleop.
                action_debug_path = None
                _log("DebugAction", f"standalone log disabled: {type(exc).__name__}: {exc}")
        else:
            action_debug_path = None
        # The initial startup projection already admitted this same dataset
        # selection.  Let the runtime dedupe an unchanged projection here;
        # forcing a second publish would unnecessarily rebuild the pre-Home
        # UI and, after a cache miss such as a dataset/profile change, repeat
        # the O(number-of-legacy-episodes) admission scan.
        startup_progress_runtime.publish()
        startup_home_captured = False
        startup_home_qpos_snapshot: dict[str, np.ndarray] = {}
        manual_startup_request = _manual_calibration_confirmation_required(args, components)
        last_startup_camera_status_at = 0.0

        def update_startup_runtime_status() -> None:
            nonlocal last_startup_camera_status_at
            sync_foot_pad_runtime_status()
            tick_idle_tactile_compression()
            now = time.time()
            if now - last_startup_camera_status_at < 0.2:
                return
            last_startup_camera_status_at = now
            clutch_input.publish_camera_status(
                camera_sources=camera_sources,
                preview_camera_sources=preview_camera_sources,
                camera_previewer=camera_previewer,
                now=now,
                force=True,
            )

        while True:
            startup_request = "home"
            if manual_startup_request:
                requested = _wait_for_real_teleop_start_request(
                    components,
                    clutch_input,
                    arm_mode="real",
                    args=args,
                    control_ui=control_ui,
                    ik_debug_log_recorder=ik_debug_log_recorder,
                    ik_debug_profile_base_enabled=ik_debug_profile_base_enabled,
                    visualization_request_handler=handle_startup_visualization_requests,
                    visualization_update_handler=update_startup_visualizations,
                    runtime_status_update_handler=update_startup_runtime_status,
                    begin_home_transition_handler=lambda: (
                        begin_idle_tactile_quiesce("startup_runtime_home")
                    ),
                    require_home_transition_ready_handler=lambda: (
                        require_idle_tactile_quiesced("startup_runtime_home")
                    ),
                )
                if requested is None:
                    return
                startup_request = requested
            core_startup_keys = {
                "rig.session",
                "vive.clock_sync",
                *(f"arm.{side}.qpos" for side in components.active_sides),
            }
            core_startup_blockers = {
                key: startup_device_errors[key]
                for key in sorted(core_startup_keys)
                if key in startup_device_errors
            }
            if core_startup_blockers:
                detail = "; ".join(
                    f"{key}: {value}" for key, value in core_startup_blockers.items()
                )
                clutch_input.publish_calibration_waiting(
                    "机械臂安全状态尚未就绪，不能执行 runtime Home；"
                    f"请先点击重连失败设备。\n{detail}"
                )
                manual_startup_request = True
                continue
            try:
                # Automatic/non-interactive startup does not pass through the
                # GUI's two-second Home delay, so retain the same fail-closed
                # barrier immediately before sampling.
                require_idle_tactile_quiesced("startup_runtime_home")
                startup_home_qpos_snapshot = _runtime_home_calibration(
                    components,
                    arm_mode="real",
                    timeout_sec=args.calibration_timeout,
                    fresh_data_timeout_sec=runtime_home_fresh_data_timeout_sec(
                        args.calibration_timeout
                    ),
                    num_samples=args.calibration_samples,
                    progress_callback=clutch_input.publish_runtime_home_progress,
                    atomic_robot_home_snapshot=bool(
                        getattr(args, "runtime_home_atomic_arm_snapshot", True)
                    ),
                    late_weighted_human_home=bool(
                        getattr(args, "runtime_home_late_weighted_vive", True)
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                message = _runtime_home_failure_message(exc)
                _log("Calibration", message)
                clutch_input.publish_calibration_waiting(
                    f"{message}\n本次开始遥操已取消并回到未遥操状态；"
                    "请检查 VIVE sender/tracker_role 后重新点击“开始遥操”。"
                )
                if not clutch_input.enabled:
                    raise
                manual_startup_request = True
                continue
            startup_home_captured = True
            break
        _release_runtime_state_readers(components)
        _log(
            "ArmMode",
            "runtime arm state source=last_commanded_action; startup hardware-state readers are now idle after calibration.",
        )
        if startup_home_captured:
            _emit_home_alignment_logs(components)
        else:
            _log("Calibration", "runtime home alignment logs skipped because Home was not captured.")
        if legacy_tcp_mode_requested(args.requested_tcp_mode):
            _log(
                "TCP",
                "legacy --tcp-mode rm_eef was requested; runtime now uses tool_tcp actions.",
            )
        _arm_after_start_calibration(
            args,
            components,
            clutch_input,
            home_captured=startup_home_captured,
            position_force_guard_monitor=position_force_guard_monitor,
            runtime_home_qpos_snapshot=(
                startup_home_qpos_snapshot
                if bool(getattr(args, "runtime_home_atomic_arm_snapshot", True))
                else None
            ),
        )
        position_admittance_active = bool(
            getattr(args, "position_admittance_active", False)
        )
        position_admittance_shadow_requested = bool(
            getattr(args, "position_admittance_shadow", False)
        )
        if position_admittance_shadow_requested or position_admittance_active:
            admittance_mode = (
                "combined_active"
                if position_admittance_combined_active
                else "rotation_active"
                if position_admittance_rotation_active
                else "active"
                if position_admittance_active
                else "rotation_shadow"
                if bool(
                    getattr(args, "position_admittance_rotation_shadow", False)
                )
                else "shadow"
            )
            admittance_error: str | None = None
            if _arm_control_mode_arg(args) != "position":
                admittance_error = "position admittance requires position arm control"
            elif position_force_guard_monitor is None:
                admittance_error = "the position force guard monitor is unavailable"
            elif str(getattr(args, "position_force_guard_source", "tool")) != "tool":
                admittance_error = (
                    "position admittance currently requires "
                    "--position-force-guard-source tool"
                )
            elif components.dual_arm_retargeter is None:
                admittance_error = "the coupled pre-IK target filter is unavailable"
            if admittance_error is None:
                def position_admittance_side_activity(side: str) -> tuple[bool, str]:
                    if clutch_input.hold_active():
                        return False, "operator_hold"
                    init_controller = components.init_arm_controller
                    if init_controller is not None and bool(init_controller.active):
                        return False, "init_arm_active"
                    fix_controller = components.fix_arm_controller
                    fixed_sides = set(
                        getattr(fix_controller, "fixed_sides", ()) or ()
                    )
                    if side in fixed_sides:
                        return False, "fixed_side"
                    if not bool(getattr(clutch_input, "_teleop_started", False)):
                        return False, "teleop_not_started"
                    return True, "running_live_side"

                try:
                    installer = (
                        install_position_admittance_active
                        if position_admittance_active
                        else install_position_admittance_shadow
                    )
                    installer_kwargs: dict[str, object] = {}
                    if position_admittance_active:
                        installer_kwargs["apply_sides"] = (
                            position_admittance_active_apply_sides(
                                str(args.position_admittance_active_side)
                            )
                        )
                        installer_kwargs["active_envelope"] = str(
                            args.position_admittance_active_envelope
                        )
                        installer_kwargs["endpoint_policy"] = str(
                            getattr(
                                args,
                                "position_admittance_active_endpoint_policy",
                                POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
                            )
                        )
                        if position_admittance_rotation_active:
                            installer_kwargs["rotational_config"] = (
                                position_rotational_admittance_config(
                                    args,
                                    profile=str(
                                        args.position_admittance_rotation_active_profile
                                    ),
                                )
                            )
                            installer_kwargs[
                                "combined_translation_rotation_active"
                            ] = position_admittance_combined_active
                    elif bool(
                        getattr(
                            args,
                            "position_admittance_rotation_shadow",
                            False,
                        )
                    ):
                        installer_kwargs["rotational_config"] = (
                            position_rotational_admittance_config(
                                args,
                                profile=str(
                                    args.position_admittance_rotation_shadow_profile
                                ),
                            )
                        )
                    admittance_config = (
                        position_admittance_active_config(args)
                        if position_admittance_active
                        else position_admittance_shadow_config(args)
                    )
                    if bool(
                        getattr(
                            args,
                            "position_admittance_rotation_shadow",
                            False,
                        )
                    ):
                        admittance_config = replace(
                            admittance_config,
                            enabled_axes=(False, False, False),
                        )
                    position_admittance_shadow = installer(
                        retargeter=components.dual_arm_retargeter,
                        monitor=position_force_guard_monitor,
                        active_sides=components.active_sides,
                        config=admittance_config,
                        side_activity=position_admittance_side_activity,
                        **installer_kwargs,
                    )
                except Exception as exc:  # noqa: BLE001 - active mode fails closed below.
                    admittance_error = f"{type(exc).__name__}: {exc}"
            if admittance_error is not None:
                position_admittance_shadow = None
                if action_debug_log_recorder.active:
                    action_debug_log_recorder.write_event(
                        "position_admittance_disabled",
                        _position_admittance_runtime_event_payload(
                            requested_mode=admittance_mode,
                            enabled=False,
                            active_sides=list(components.active_sides),
                            coupled_pre_ik_target_filter_available=(
                                components.dual_arm_retargeter is not None
                            ),
                            reason=admittance_error,
                            endpoint_excursion_guard=(
                                None
                                if components.position_admittance_endpoint_guard is None
                                else components.position_admittance_endpoint_guard.snapshot()
                            ),
                        ),
                        now=time.time(),
                    )
                if position_admittance_active:
                    raise RuntimeError(
                        "active position admittance failed to arm: "
                        + admittance_error
                    )
                _log(
                    "PositionAdmittanceShadow",
                    "disabled without changing control: " + admittance_error,
                )
                if control_ui is not None:
                    control_ui.append_message(
                        "PositionAdmittanceShadow",
                        "位控柔顺 shadow 未能启用；现有位控目标保持不变。\n"
                        + admittance_error,
                    )
            else:
                assert position_admittance_shadow is not None
                admittance_snapshot = position_admittance_shadow.snapshot()
                endpoint_excursion_guard_snapshot = (
                    None
                    if components.position_admittance_endpoint_guard is None
                    else components.position_admittance_endpoint_guard.snapshot()
                )
                _log(
                    "PositionAdmittance",
                    f"ready mode={admittance_mode} "
                    f"shadow_only={str(not position_admittance_active).lower()} "
                    f"applied_to_target={str(position_admittance_active).lower()} "
                    "wrench_owner=position_force_guard poll_hz="
                    f"{position_force_guard_monitor.config.poll_hz:g} "
                    f"config={admittance_snapshot['config']}",
                )
                if action_debug_log_recorder.active:
                    action_debug_log_recorder.write_event(
                        "position_admittance_config",
                        _position_admittance_runtime_event_payload(
                            requested_mode=admittance_mode,
                            enabled=True,
                            active_sides=list(components.active_sides),
                            coupled_pre_ik_target_filter_available=True,
                            snapshot=admittance_snapshot,
                            endpoint_excursion_guard=(
                                endpoint_excursion_guard_snapshot
                            ),
                        ),
                        now=time.time(),
                    )
        clutch_input.consume_episode_start_request()
        clutch_input.consume_episode_stop_request()

        if args.keyboard_clutch:
            _log(
                "Keys",
                "p/Enter=start calibration, t=start episode, s=stop episode, z=left clutch, x=right clutch, b=both clutch, r=release, q=quit",
            )
        if episode_recorder.config.enabled:
            auto_start_note = " auto_start=on" if args.record_auto_start else " auto_start=off"
            _log(
                "Episode",
                f"recording ready root={episode_recorder.root_dir or _episode_root_dir(args, components)} "
                f"start_key='t' stop_key='s'{auto_start_note}",
            )
        if dataset_recorder is not None:
            _log(
                "Dataset",
                f"formal collection ready root={dataset_recorder.dataset_root} "
                f"cameras={sorted(camera_sources)} write_video={bool(args.write_video and not args.no_camera)} "
                f"tactile_storage={args.tactile_storage} "
                f"tactile_compression={getattr(args, 'tactile_compression', 'none')} "
                f"worker_mode={dataset_recorder.collection_worker_mode} "
                f"start_key='t' stop_key='s'",
            )
        elif not bool(getattr(args, "no_collection", False)):
            _log(
                "Dataset",
                "formal collection configured; writer startup is deferred "
                "until the selected dataset/profile is started",
            )

        status_gate = ChangeGate()
        orientation_gate = ChangeGate()
        tactile_sum_gate = ChangeGate()
        status_heartbeat_sec = (
            float(args.status_sec) if float(args.status_sec) > 0.0 else None
        )
        if debug_timing_modules.enabled:
            _log(
                "DebugTiming",
                "enabled "
                f"modules={list(debug_timing_modules.enabled_names())} "
                f"threshold={debug_timing_state.threshold_sec:.3f}s "
                f"report_interval={debug_timing_state.report_interval_sec:.3f}s "
                f"light_log={debug_timing_path}",
            )
        if bool(debug_timing_modules.ik_profile):
            budget = getattr(args, "ik_time_budget_ms", None)
            budget_text = "one_control_period" if budget is None else f"{float(budget):.3f}ms"
            _log(
                "DebugIKTiming",
                "enabled "
                f"gui={'on' if control_ui is not None else 'off'} "
                f"print_every={int(getattr(args, 'debug_ik_timing_print_every', 0))} "
                f"time_budget={budget_text}",
            )
        step_index = 0
        episode_last_state: str | None = None

        def current_episode_last_state() -> str | None:
            return episode_last_state

        def set_current_episode_last_state(value: str | None) -> None:
            nonlocal episode_last_state
            episode_last_state = value

        def current_step_index() -> int:
            return int(step_index)

        def set_current_step_index(value: int) -> None:
            nonlocal step_index
            step_index = int(value)

        fix_hold_coordinator = SpecialArmFixHoldCoordinator()
        sample_rate_gate = PeriodicRateGate(float(args.hz))
        outer_control_hz = _effective_outer_control_hz(args)
        gripper_warmup_probe_runtime = build_gripper_warmup_probe_runtime(
            args,
            components,
            gripper_backend_for_side=_gripper_backend_for_side,
            run_probe_for_side=_run_gripper_warmup_probe_for_side,
            safe_float_value=_safe_float_value,
            log=_log,
        )
        gripper_warmup_probe_enabled = gripper_warmup_probe_runtime.enabled
        reset_gripper_warmup_probe = gripper_warmup_probe_runtime.reset
        gripper_warmup_probe_running = gripper_warmup_probe_runtime.running
        gripper_warmup_probe_failure_message = (
            gripper_warmup_probe_runtime.failure_message
        )
        start_gripper_warmup_probe = gripper_warmup_probe_runtime.start
        gripper_warmup_probe_status = gripper_warmup_probe_runtime.status








        recording_auto_stop_reason = _recording_auto_stop_reason

        def recording_warmup_status(
            now: float,
            *,
            step_result: SessionStepResult | None,
        ) -> tuple[bool, str]:
            return _recording_warmup_status(
                now,
                step_result=step_result,
                args=args,
                components=components,
                clutch_input=clutch_input,
                dataset_recorder=dataset_recorder,
                camera_sources=camera_sources,
                preview_camera_sources=preview_camera_sources,
                camera_previewer=camera_previewer,
                recording_auto_stop_reason_fn=recording_auto_stop_reason,
                gripper_warmup_probe_status=gripper_warmup_probe_status,
                camera_status_blocking_message=_camera_status_blocking_message,
            )

        position_episode_debug_runtime = build_position_episode_debug_runtime(
            args,
            components,
            position_audit_fanout=position_audit_fanout,
            position_debug_close_threads=position_debug_close_threads,
            update_runtime_metadata=update_control_debug_runtime_metadata,
            new_position_debug_recorder=_new_position_debug_recorder,
            log=_log,
        )

        episode_recording_runtime = build_episode_recording_runtime(
            args=args,
            components=components,
            clutch_input=clutch_input,
            episode_recorder=episode_recorder,
            camera_sources=camera_sources,
            control_ui=control_ui,
            startup_device_errors=startup_device_errors,
            fix_hold_coordinator=fix_hold_coordinator,
            position_episode_debug_runtime=position_episode_debug_runtime,
            update_control_debug_runtime_metadata=(
                update_control_debug_runtime_metadata
            ),
            publish_runtime_hardware_status=publish_runtime_hardware_status,
            collection_startup_blockers=collection_startup_blockers,
            recording_warmup_status=recording_warmup_status,
            reset_gripper_warmup_probe=reset_gripper_warmup_probe,
            dataset_recorder_provider=current_dataset_recorder,
            set_dataset_recorder=set_current_dataset_recorder,
            camera_previewer_provider=lambda: (
                visualization_runtime.current_state()[0]
            ),
            episode_last_state_provider=current_episode_last_state,
            set_episode_last_state=set_current_episode_last_state,
            step_index_provider=current_step_index,
            support=globals(),
        )
        recording_active = episode_recording_runtime.recording_active
        recording_paused = episode_recording_runtime.recording_paused
        recording_pause_reason = episode_recording_runtime.recording_pause_reason
        recording_frame_count = episode_recording_runtime.recording_frame_count
        recording_path = episode_recording_runtime.recording_path
        recording_frequency_status = (
            episode_recording_runtime.recording_frequency_status
        )
        publish_collection_frequency_warning = (
            episode_recording_runtime.publish_collection_frequency_warning
        )
        start_recording = episode_recording_runtime.start_recording
        schedule_recording_countdown = (
            episode_recording_runtime.schedule_recording_countdown
        )
        cancel_recording_countdown = (
            episode_recording_runtime.cancel_recording_countdown
        )
        advance_recording_countdown = (
            episode_recording_runtime.advance_recording_countdown
        )
        stop_recording = episode_recording_runtime.stop_recording
        pause_recording = episode_recording_runtime.pause_recording
        resume_recording_if_ready = (
            episode_recording_runtime.resume_recording_if_ready
        )
        sync_fix_hold = episode_recording_runtime.sync_fix_hold
        poll_dataset_finalize = episode_recording_runtime.poll_dataset_finalize










        if args.record_auto_start:
            schedule_recording_countdown(time.time(), start_reason="auto_start")

        def current_visualization_state() -> tuple[object | None, bool, object | None]:
            return camera_previewer, bool(camera_preview_active), tactile_visualizer

        if runtime_extension is not None:
            bind_extension_runtime = getattr(runtime_extension, "bind_runtime", None)
            if callable(bind_extension_runtime):
                bind_extension_runtime(
                    components=components,
                    clutch_input=clutch_input,
                    camera_sources=camera_sources,
                    dataset_recorder=current_dataset_recorder(),
                    dataset_recorder_provider=current_dataset_recorder,
                    episode_recorder=episode_recorder,
                    control_ui=control_ui,
                    recording_active=recording_active,
                    pause_recording=pause_recording,
                )
            start_extension = getattr(runtime_extension, "start", None)
            if not callable(start_extension):
                raise TypeError("runtime_extension lacks start().")
            start_extension(components=components)

        run_real_teleop_control_loop(
            RealTeleopControlLoopContext(
                action_debug_log_recorder=action_debug_log_recorder,
                actuator_trace_recorder=actuator_trace_recorder,
                advance_recording_countdown=advance_recording_countdown,
                args=args,
                camera_preview_rate=camera_preview_rate,
                camera_sources=camera_sources,
                cancel_recording_countdown=cancel_recording_countdown,
                close_camera_preview=close_camera_preview,
                close_tactile_preview=close_tactile_preview,
                clutch_input=clutch_input,
                components=components,
                console_stream=console_stream,
                control_ui=control_ui,
                debug_timing_log_recorder=debug_timing_log_recorder,
                debug_timing_modules=debug_timing_modules,
                debug_timing_state=debug_timing_state,
                episode_recorder=episode_recorder,
                fix_hold_coordinator=fix_hold_coordinator,
                gripper_warmup_probe_running=gripper_warmup_probe_running,
                ik_debug_log_recorder=ik_debug_log_recorder,
                ik_debug_profile_base_enabled=ik_debug_profile_base_enabled,
                idle_tactile_compression_tick=tick_idle_tactile_compression,
                begin_idle_tactile_quiesce=begin_idle_tactile_quiesce,
                require_idle_tactile_quiesced=require_idle_tactile_quiesced,
                main_loop_watchdog=main_loop_watchdog,
                open_camera_preview=open_camera_preview,
                open_tactile_preview=open_tactile_preview,
                orientation_gate=orientation_gate,
                outer_control_hz=outer_control_hz,
                pause_recording=pause_recording,
                poll_dataset_finalize=poll_dataset_finalize,
                pose_snapshot_recorder=pose_snapshot_recorder,
                position_audit_fanout=position_audit_fanout,
                position_force_guard_monitor=position_force_guard_monitor,
                position_admittance_shadow=position_admittance_shadow,
                preview_camera_sources=preview_camera_sources,
                publish_collection_frequency_warning=(
                    publish_collection_frequency_warning
                ),
                recording_active=recording_active,
                recording_auto_stop_reason=recording_auto_stop_reason,
                recording_frame_count=recording_frame_count,
                recording_frequency_status=recording_frequency_status,
                recording_path=recording_path,
                recording_pause_reason=recording_pause_reason,
                recording_paused=recording_paused,
                recording_warmup_status=recording_warmup_status,
                resume_recording_if_ready=resume_recording_if_ready,
                retry_failed_devices=retry_failed_devices,
                sample_rate_gate=sample_rate_gate,
                schedule_recording_countdown=schedule_recording_countdown,
                status_gate=status_gate,
                status_heartbeat_sec=status_heartbeat_sec,
                stop_recording=stop_recording,
                sync_fix_hold=sync_fix_hold,
                sync_foot_pad_runtime_status=sync_foot_pad_runtime_status,
                tactile_preview_rate=tactile_preview_rate,
                tactile_sum_gate=tactile_sum_gate,
                dataset_recorder=current_dataset_recorder,
                visualization_state=current_visualization_state,
                episode_start_countdown_deadline=(
                    episode_recording_runtime.countdown_deadline
                ),
                episode_last_state=current_episode_last_state,
                set_episode_last_state=set_current_episode_last_state,
                step_index=current_step_index,
                set_step_index=set_current_step_index,
                runtime_extension=runtime_extension,
            )
        )
    finally:
        shutdown_real_teleop_runtime(
            locals(),
            terminal_error=sys.exc_info()[1],
            support=globals(),
        )

__all__ = [
    "bind_legacy_support",
    "cleanup_active_startup",
    "run_real_teleop",
]
