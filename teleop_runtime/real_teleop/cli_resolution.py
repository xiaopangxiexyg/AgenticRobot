"""CLI parsing, cross-option resolution, and admission validation.

Argument declaration remains in :mod:`teleop_runtime.real_teleop.cli`; this
module owns the ordered post-parse resolution that was formerly embedded in
the canonical debug entrypoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import inspect
import sys

import numpy as np

from .artifact_context import configure_real_teleop_artifact_paths
from .position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    position_admittance_active_apply_sides,
    position_admittance_active_config,
    position_admittance_shadow_config,
    position_rotational_admittance_config,
    validate_position_admittance_active_authority,
)
from .progress_adapter import (
    resolve_real_teleop_default_progress_profile,
    resolve_real_teleop_progress_entry,
    resolve_real_teleop_task_stage_profile,
    validate_real_teleop_initial_progress_profile,
)


_SUPPORT_NAMES = (
    "ANNOTATED_DUAL_RM65_BASE_DISTANCE_M",
    "DEFAULT_CALIBRATION_PROFILE_DIR",
    "DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ",
    "HAND_BACKEND_INSPIRE",
    "LEGACY_POSITION_CONTROL_PROFILE_NAME",
    "POSITION_15_60_120HZ_PROFILE_NAME",
    "POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED",
    "POSITION_COMMAND_PLANNER_LINEAR",
    "POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED",
    "TOOL_TCP_FRAME",
    "_active_sides_from_args",
    "_apply_default_arm_ips",
    "_apply_default_q6_ports",
    "_apply_formal_dataset_layout",
    "_apply_real_loop_timing_defaults",
    "_argv_has_option",
    "_dmtac_sides",
    "_hand_backend_mode_for_side",
    "_init_arm_qpos_speed_scale_for_side",
    "_latest_vive_calibration_profile_path",
    "_parse_hand_tactile_regions",
    "_resolve_gripper_input_source",
    "_resolve_movement_multiple_args",
    "_resolve_workspace_override_args",
    "_resolved_gripper_policy",
    "_run_gui_device_preflight",
    "_save_explicit_arm_ips_latest",
    "_validate_required_arm_ips",
    "apply_rm65_position_profile_defaults",
    "build_real_teleop_argument_parser",
    "configured_foot_pad_count",
    "default_rm65_workspace_near_boundary_collection_pose_frame_contract",
    "normalize_dmtac_tactile_fields",
    "normalize_position_command_planner",
    "resolve_control_debug_config",
    "resolve_real_arm_execution_config",
    "resolve_special_arm_control_mode",
    "validate_human_shadow_arguments",
)


class _LegacyFunctionProxy:
    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(f"recursive CLI-resolution support proxy: {self._name}")
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    missing = [name for name in _SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "real-teleop CLI resolution support is incomplete: "
            + ", ".join(sorted(missing))
        )
    globals().update(
        {
            name: (
                _LegacyFunctionProxy(namespace, name)
                if inspect.isroutine(namespace[name])
                else namespace[name]
            )
            for name in _SUPPORT_NAMES
        }
    )


def _parse_real_teleop_args(
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:])
    args = parser.parse_args()
    if not str(getattr(args, "dataset_root", "")).strip():
        parser.error(
            "--dataset-root must not be empty; omit it to use the configured default."
        )
    if bool(getattr(args, "plus_episode_curation", False)) and not str(
        getattr(args, "plus_annotation_root", "") or ""
    ).strip():
        parser.error("--plus-annotation-root is required with --plus-episode-curation")
    try:
        configure_real_teleop_artifact_paths(
            args,
            raw_argv=raw_argv,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    raw_arg_set = set(raw_argv)
    if bool(getattr(args, "no_debug_disk_write", False)):
        args.control_debug = "off"
        args.actuator_trace = False
        args.debug_action = False
        args.debug_timing_log = False
        args.debug_timing_auto_log = False
        args.no_record = True
        args.no_episode_recording = True
        args.debug_trajectory_recording = False
    validate_human_shadow_arguments(args, parser)
    try:
        requested_arm_profile = str(getattr(args, "arm_execution_profile", "") or "")
        tentative_arm_mode = (
            "position"
            if requested_arm_profile.startswith("position")
            else "impedance"
            if requested_arm_profile == "impedance_40hz"
            else str(getattr(args, "arm_control_mode", "impedance"))
        )
        explicit_compatibility_arm_profile = (
            _argv_has_option(raw_argv, "--arm-execution-profile")
            and requested_arm_profile
            not in ("", POSITION_15_60_120HZ_PROFILE_NAME)
        )
        if explicit_compatibility_arm_profile:
            if (
                _argv_has_option(raw_argv, "--position-control-profile")
                and args.position_control_profile
                != LEGACY_POSITION_CONTROL_PROFILE_NAME
            ):
                raise ValueError(
                    "a compatibility --arm-execution-profile conflicts with the formal "
                    "--position-control-profile; use --position-control-profile legacy"
                )
            args.position_control_profile_resolved = (
                LEGACY_POSITION_CONTROL_PROFILE_NAME
            )
            args.position_control_profile_metadata = {
                "name": LEGACY_POSITION_CONTROL_PROFILE_NAME,
                "compatibility_mode": True,
                "reason": f"explicit_arm_execution_profile:{requested_arm_profile}",
            }
        else:
            apply_rm65_position_profile_defaults(
                args,
                raw_argv=raw_argv,
                mode=tentative_arm_mode,
                include_source_shaping=True,
            )
        args.arm_execution_config = resolve_real_arm_execution_config(
            args,
            raw_argv=raw_argv,
            impedance_default_outer_control_hz=DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ,
        )
        args.control_debug_config = resolve_control_debug_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    # A named profile may select a mode without requiring the legacy mode flag.
    # Fine-grained legacy fields retain their original raw values so callers can
    # still distinguish explicit overrides from mode-derived defaults.
    args.arm_control_mode = args.arm_execution_config.mode
    if bool(getattr(args, "tcp_pose_envelope_active", False)) and not str(
        getattr(args, "tcp_pose_envelope_profile", "") or ""
    ).strip():
        parser.error(
            "--tcp-pose-envelope-active requires "
            "--tcp-pose-envelope-profile."
        )
    position_admittance_shadow = bool(
        getattr(args, "position_admittance_shadow", False)
    )
    position_admittance_active = bool(
        getattr(args, "position_admittance_active", False)
    )
    position_admittance_rotation_shadow = bool(
        getattr(args, "position_admittance_rotation_shadow", False)
    )
    position_admittance_rotation_active = bool(
        getattr(args, "position_admittance_rotation_active", False)
    )
    position_admittance_combined_active = bool(
        getattr(args, "position_admittance_combined_active", False)
    )
    position_admittance_active_envelope = str(
        getattr(
            args,
            "position_admittance_active_envelope",
            POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
        )
    )
    position_admittance_active_endpoint_policy = str(
        getattr(
            args,
            "position_admittance_active_endpoint_policy",
            POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
        )
    )
    if position_admittance_shadow and position_admittance_active:
        parser.error(
            "--position-admittance-shadow and --position-admittance-active "
            "are mutually exclusive."
        )
    if (
        position_admittance_rotation_shadow
        and not position_admittance_shadow
    ):
        parser.error(
            "--position-admittance-rotation-shadow requires "
            "--position-admittance-shadow."
        )
    if (
        not position_admittance_rotation_shadow
        and _argv_has_option(
            raw_argv,
            "--position-admittance-rotation-shadow-profile",
        )
    ):
        parser.error(
            "--position-admittance-rotation-shadow-profile requires "
            "--position-admittance-rotation-shadow."
        )
    if position_admittance_rotation_active and not position_admittance_active:
        parser.error(
            "--position-admittance-rotation-active requires "
            "--position-admittance-active."
        )
    if position_admittance_combined_active and not (
        position_admittance_active and position_admittance_rotation_active
    ):
        parser.error(
            "--position-admittance-combined-active requires both "
            "--position-admittance-active and "
            "--position-admittance-rotation-active."
        )
    if (
        not position_admittance_rotation_active
        and _argv_has_option(
            raw_argv,
            "--position-admittance-rotation-active-profile",
        )
    ):
        parser.error(
            "--position-admittance-rotation-active-profile requires "
            "--position-admittance-rotation-active."
        )
    if (
        not position_admittance_rotation_active
        and _argv_has_option(
            raw_argv,
            "--position-admittance-active-max-tool-rotation-excursion-deg",
        )
    ):
        parser.error(
            "--position-admittance-active-max-tool-rotation-excursion-deg "
            "requires --position-admittance-rotation-active."
        )
    position_admittance_active_side = getattr(
        args,
        "position_admittance_active_side",
        None,
    )
    if position_admittance_active and position_admittance_active_side is None:
        parser.error(
            "--position-admittance-active requires the explicit authority "
            "selection --position-admittance-active-side {left,right,both}."
        )
    if not position_admittance_active and position_admittance_active_side is not None:
        parser.error(
            "--position-admittance-active-side requires "
            "--position-admittance-active."
        )
    if (
        not position_admittance_active
        and position_admittance_active_envelope
        != POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE
    ):
        parser.error(
            "--position-admittance-active-envelope other than cautious requires "
            "--position-admittance-active."
        )
    if (
        not position_admittance_active
        and position_admittance_active_endpoint_policy
        != POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY
    ):
        parser.error(
            "--position-admittance-active-endpoint-policy other than "
            "runtime_home_hold requires --position-admittance-active."
        )
    position_admittance_apply_sides: tuple[str, ...] = ()
    if position_admittance_active_side is not None:
        try:
            position_admittance_apply_sides = (
                position_admittance_active_apply_sides(
                    str(position_admittance_active_side)
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
        if not set(position_admittance_apply_sides).issubset(
            set(_active_sides_from_args(args))
        ):
            parser.error(
                "--position-admittance-active-side must be included in --active-sides."
            )
        # Active admittance and fix-arm are compatible.  The runtime installer
        # receives a per-side activity callback and suspends the modifier on a
        # fixed side, while the live side retains the bounded overlay.  A
        # global --no-fix-arm requirement would disable the five-pedal
        # side-switch contract before the hardware topology is even opened.
    if position_admittance_shadow or position_admittance_active:
        if args.arm_control_mode != "position":
            parser.error("Position admittance requires position arm control mode.")
        admittance_active_sides = set(_active_sides_from_args(args))
        if admittance_active_sides != {"left", "right"}:
            parser.error(
                "Position admittance requires the coupled dual-arm topology; "
                "use --active-sides both, then select the explicit overlay "
                "authority with --position-admittance-active-side."
            )
        if bool(getattr(args, "disable_dual_arm_collision_avoidance", False)):
            parser.error(
                "Position admittance requires the coupled pre-IK target filter; "
                "--no-collision/--disable-dual-arm-collision-avoidance is not supported."
            )
        if not bool(getattr(args, "position_force_guard", False)):
            parser.error(
                "Position admittance requires --position-force-guard so the "
                "20 Hz wrench reader and independent emergency stop remain authoritative."
            )
        if str(getattr(args, "position_force_guard_source", "tool")) != "tool":
            parser.error(
                "Position admittance currently requires "
                "--position-force-guard-source tool."
            )
        try:
            translation_config = (
                position_admittance_active_config(args)
                if position_admittance_active
                else position_admittance_shadow_config(args)
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        if position_admittance_active:
            try:
                rotation_config = (
                    position_rotational_admittance_config(
                        args,
                        profile=str(
                            args.position_admittance_rotation_active_profile
                        ),
                    )
                    if position_admittance_rotation_active
                    else None
                )
                validate_position_admittance_active_authority(
                    translation_config,
                    active_envelope=position_admittance_active_envelope,
                    rotational_config=rotation_config,
                    combined_translation_rotation_active=(
                        position_admittance_combined_active
                    ),
                    endpoint_policy=position_admittance_active_endpoint_policy,
                    apply_sides=position_admittance_apply_sides,
                    max_tool_excursion_m=float(
                        args.position_admittance_active_max_tool_excursion_m
                    ),
                    max_tool_rotation_excursion_deg=(
                        float(
                            args.position_admittance_active_max_tool_rotation_excursion_deg
                        )
                        if position_admittance_rotation_active
                        else None
                    ),
                )
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
    if bool(getattr(args, "actuator_trace", False)):
        if args.arm_control_mode != "position":
            parser.error("--actuator-trace requires position arm control mode.")
        if not args.arm_execution_config.position_background_streaming_enabled:
            parser.error(
                "--actuator-trace requires the position background command stream."
            )
        if bool(args.control_debug_config.enabled):
            parser.error(
                "--actuator-trace must be used with --control-debug off; summary/full retain "
                "their existing audit overhead and would invalidate the low-disturbance capture."
            )
    if (
        args.arm_control_mode == "position"
        and not args.arm_execution_config.position_background_streaming_enabled
        and not _argv_has_option(raw_argv, "--position-command-planner")
    ):
        # Explicitly disabling the follow stream is a supported compatibility
        # override.  The formal 60->120 planner cannot run without that stream,
        # so retain the legacy direct-reference planner unless the caller
        # explicitly requested another (and therefore invalid) combination.
        args.position_command_planner = POSITION_COMMAND_PLANNER_LINEAR
        if not _argv_has_option(raw_argv, "--position-max-command-step-deg"):
            args.position_max_command_step_deg = None
        metadata = getattr(args, "position_control_profile_metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("runtime_deviations", []).append(
                "follow_stream_disabled_use_linear_command_planner"
            )
    requested_collection_worker_mode = str(args.collection_worker_mode)
    args.collection_worker_mode_resolved = (
        "process"
        if requested_collection_worker_mode == "auto"
        and args.arm_control_mode == "position"
        else "thread"
        if requested_collection_worker_mode == "auto"
        else requested_collection_worker_mode
    )
    if (
        args.collection_worker_mode_resolved == "process"
        and args.arm_control_mode != "position"
    ):
        parser.error(
            "--collection-worker-mode process is currently position-only; "
            "force/impedance collection intentionally retains the established thread path."
        )
    tactile_bundle_root = str(
        getattr(args, "tactile_bundle_root", "") or ""
    ).strip()
    tactile_bundle_min_free_gb = float(
        getattr(args, "tactile_bundle_min_free_gb", 0.0) or 0.0
    )
    if not tactile_bundle_root and tactile_bundle_min_free_gb > 0.0:
        parser.error(
            "--tactile-bundle-min-free-gb requires --tactile-bundle-root."
        )
    if tactile_bundle_root:
        if bool(getattr(args, "no_collection", False)):
            parser.error(
                "--tactile-bundle-root requires formal collection; remove "
                "--no-collection."
            )
        if str(getattr(args, "tactile_storage", "external")) != "external":
            parser.error(
                "--tactile-bundle-root requires --tactile-storage external."
            )
        if _argv_has_option(raw_argv, "--no-tactile-artifact-integrity"):
            parser.error(
                "--tactile-bundle-root cannot be combined with explicit "
                "--no-tactile-artifact-integrity."
            )
        # Native separation always seals exact payload bytes.  This is a hard
        # part of the bundle contract, not an optional diagnostic flag.
        args.tactile_artifact_integrity = True
    if bool(getattr(args, "idle_tactile_compression", False)):
        if bool(getattr(args, "no_collection", False)):
            parser.error(
                "--idle-tactile-compression requires formal collection; "
                "remove --no-collection."
            )
        if str(getattr(args, "tactile_storage", "external")) != "external":
            parser.error(
                "--idle-tactile-compression requires --tactile-storage external."
            )
        if str(getattr(args, "tactile_compression", "none")) != "none":
            parser.error(
                "--idle-tactile-compression requires --tactile-compression none; "
                "inline compression and idle recompression are mutually exclusive."
            )
    if bool(getattr(args, "tactile_artifact_integrity", False)):
        if bool(getattr(args, "no_collection", False)):
            parser.error(
                "--tactile-artifact-integrity requires formal collection; "
                "remove --no-collection."
            )
        if str(getattr(args, "tactile_storage", "external")) != "external":
            parser.error(
                "--tactile-artifact-integrity requires --tactile-storage external."
            )
    if int(args.position_ik_solve_min_iterations) > int(
        args.position_ik_solve_max_iterations
    ):
        parser.error(
            "--position-ik-solve-min-iterations must be <= "
            "--position-ik-solve-max-iterations."
        )
    if float(args.position_ik_candidate_certification_reserve_ms) > float(
        args.position_ik_iteration_reserve_ms
    ):
        parser.error(
            "--position-ik-candidate-certification-reserve-ms must be <= "
            "--position-ik-iteration-reserve-ms."
        )
    if (
        bool(getattr(args, "position_ik_admitted_motion_limit", False))
        and str(args.position_ik_step_mode) != "single_step_singularity_aware"
    ):
        parser.error(
            "--position-ik-admitted-motion-limit requires "
            "--position-ik-step-mode single_step_singularity_aware."
        )
    if (
        bool(getattr(args, "position_ik_branch_continuity_anchor", False))
        and str(args.position_ik_step_mode) != "single_step_singularity_aware"
    ):
        parser.error(
            "--position-ik-branch-continuity-anchor requires "
            "--position-ik-step-mode single_step_singularity_aware."
        )
    if args.arm_control_mode != "position" and (
        args.position_ik_orientation_weight is not None
        or str(args.position_ik_step_mode) != "legacy_iterative"
        or bool(getattr(args, "position_ik_admitted_motion_limit", False))
        or bool(getattr(args, "position_ik_branch_continuity_anchor", False))
        or str(
            getattr(args, "position_joint_limit_guard_mode", "legacy")
        )
        != "legacy"
        or float(
            getattr(
                args,
                "position_joint_limit_boundary_margin_deg",
                1.5,
            )
        )
        != 1.5
        or int(args.position_ik_solve_max_iterations) != 2
        or int(args.position_ik_solve_min_iterations) != 2
        or float(args.position_ik_damping) != 0.03
        or bool(args.position_ik_adaptive_iterations)
        or float(args.position_ik_iteration_reserve_ms) != 5.0
        or float(args.position_ik_candidate_certification_reserve_ms) != 0.0
        or bool(args.position_singularity_candidate_certificate)
        or float(args.position_ik_local_step_clip_scale) != 1.0
        or float(getattr(args, "position_wrist_branch_switch_guard_deg", 12.0)) != 12.0
        or float(getattr(args, "position_wrist_q5_hysteresis_release_deg", 6.0)) != 6.0
        or str(args.position_joint_velocity_limit_profile) != "rm65"
        or float(args.position_inactive_enter_threshold_scale) != 1.0
    ):
        parser.error(
            "--position-ik-orientation-weight, --position-ik-step-mode, "
            "--position-ik-admitted-motion-limit, "
            "--position-ik-branch-continuity-anchor, "
            "--position-joint-limit-guard-mode, "
            "--position-joint-limit-boundary-margin-deg, "
            "--position-ik-solve-max-iterations, "
            "--position-ik-solve-min-iterations, --position-ik-damping, "
            "--position-ik-adaptive-iterations, --position-ik-iteration-reserve-ms, "
            "--position-ik-candidate-certification-reserve-ms, "
            "--position-singularity-candidate-certificate, "
            "--position-ik-local-step-clip-scale, "
            "--position-wrist-branch-switch-guard-deg, "
            "--position-wrist-q5-hysteresis-release-deg, and "
            "--position-joint-velocity-limit-profile, and "
            "--position-inactive-enter-threshold-scale are position-mode options and "
            "cannot alter force/impedance IK."
        )
    args.position_command_planner = normalize_position_command_planner(
        args.position_command_planner
    )
    limited_position_planners = {
        POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    }
    if (
        args.position_command_planner in limited_position_planners
        and args.arm_control_mode != "position"
    ):
        parser.error(
            "the selected acceleration/jerk-limited position-command planner "
            "requires position arm control mode."
        )
    if args.position_max_command_step_deg is not None and args.arm_control_mode != "position":
        parser.error(
            "--position-max-command-step-deg is a position-mode option and cannot alter "
            "force/impedance control."
        )
    if (
        args.position_max_command_step_deg is not None
        and args.position_command_planner not in limited_position_planners
    ):
        parser.error(
            "--position-max-command-step-deg requires an acceleration/jerk-limited "
            "position-command planner."
        )
    if (
        args.position_command_planner in limited_position_planners
        and not args.arm_execution_config.position_background_streaming_enabled
    ):
        parser.error(
            "the selected acceleration/jerk-limited planner requires the high-rate position "
            "background stream."
        )
    if (
        args.position_command_planner in limited_position_planners
        and float(args.arm_execution_config.position_background_streaming_hz)
        < float(args.arm_execution_config.position_reference_hz)
    ):
        parser.error(
            "the selected acceleration/jerk-limited planner requires the final command "
            "rate to be at least the position reference rate; otherwise safe references "
            "can be preempted before any qcmd sample is emitted."
        )
    for compatibility_warning in args.arm_execution_config.compatibility_warnings:
        print(f"WARNING: arm execution profile: {compatibility_warning}", file=sys.stderr)
    _apply_real_loop_timing_defaults(args, raw_argv=raw_argv)
    args._dataset_name_explicit = _argv_has_option(raw_argv, "--dataset-name")
    try:
        _resolve_movement_multiple_args(args, raw_arg_set=raw_arg_set)
        _resolve_workspace_override_args(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    _apply_formal_dataset_layout(args)
    progress_profile = getattr(args, "collection_progress_profile", None)
    if bool(getattr(args, "collection_progress_gui", False)) and bool(
        getattr(args, "no_collection", False)
    ):
        parser.error(
            "--collection-progress-gui requires formal collection; "
            "remove --no-collection."
        )
    if progress_profile is None and bool(
        getattr(args, "collection_progress_gui", False)
    ):
        try:
            progress_profile = resolve_real_teleop_default_progress_profile(
                dataset_root=args.dataset_root,
                requested_profile=None,
                gui_enabled=True,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        args.collection_progress_profile = progress_profile
    if progress_profile is not None:
        if bool(getattr(args, "no_collection", False)):
            parser.error(
                "--collection-progress-profile requires formal collection; "
                "remove --no-collection."
            )
        try:
            validate_real_teleop_initial_progress_profile(
                progress_profile,
                collection_task_id=str(args.collection_task_id),
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        try:
            stage_profile = resolve_real_teleop_task_stage_profile(
                progress_profile,
                getattr(args, "collection_stage_profile", ""),
            )
            resolve_real_teleop_progress_entry(
                progress_profile=progress_profile,
                stage_profile=stage_profile,
                entry_step_id=getattr(
                    args,
                    "collection_progress_entry_step",
                    "",
                ),
                entry_stage_id=getattr(
                    args,
                    "collection_progress_entry_stage",
                    "",
                ),
                entry_condition=getattr(
                    args,
                    "collection_progress_entry_condition",
                    "nominal",
                ),
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        args.collection_stage_profile = stage_profile
    else:
        advanced_entry_requested = bool(
            str(getattr(args, "collection_stage_profile", "") or "").strip()
            or str(
                getattr(args, "collection_progress_entry_step", "") or ""
            ).strip()
            or str(
                getattr(args, "collection_progress_entry_stage", "") or ""
            ).strip()
            or str(
                getattr(
                    args,
                    "collection_progress_entry_condition",
                    "nominal",
                )
            ).strip().lower()
            != "nominal"
        )
        if advanced_entry_requested:
            parser.error(
                "Stage profile and partial/recovery entry options require "
                "--collection-progress-profile."
            )
    args.pose_frame_contract_source = (
        "explicit" if args.pose_frame_contract is not None else "disabled"
    )
    if (
        args.pose_frame_contract is None
        and not bool(args.no_pose_frame_contract)
        and not bool(args.no_collection)
    ):
        if (
            _active_sides_from_args(args) == ("left", "right")
            and not np.isclose(
                float(args.left_right_base_distance),
                float(ANNOTATED_DUAL_RM65_BASE_DISTANCE_M),
                rtol=0.0,
                atol=1e-9,
            )
        ):
            parser.error(
                "The default RM65 workspace-near-boundary pose contract is calibrated for "
                f"--left-right-base-distance {ANNOTATED_DUAL_RM65_BASE_DISTANCE_M}. "
                "Provide the matching --pose-frame-contract for another station, or "
                "use --no-pose-frame-contract for the legacy v2.2/raw path."
            )
        default_contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
            _active_sides_from_args(args)
        )
        if default_contract is not None:
            args.pose_frame_contract = default_contract
            args.pose_frame_contract_source = "default_rm65_dual_arm_workspace_near_boundary_v1"
            if not str(args.policy_io_layout).strip():
                args.policy_io_layout = "auto"
    if args.pose_frame_contract is not None and not str(args.policy_io_layout).strip():
        parser.error(
            "--pose-frame-contract requires --policy-io-layout auto or an explicit layout JSON."
        )
    if args.pose_frame_contract is not None:
        if not bool(args.record_actual_wrist_camera_pose):
            parser.error(
                "--pose-frame-contract requires measured arm feedback; remove "
                "--no-record-actual-wrist-camera-pose."
            )
        contract_sides = tuple(args.pose_frame_contract.source.active_sides)
        requested_sides = _active_sides_from_args(args)
        if contract_sides != requested_sides:
            parser.error(
                "--pose-frame-contract active_sides disagree with collection --active-sides: "
                f"{contract_sides!r} != {requested_sides!r}."
            )
    args.gui_requested = bool(getattr(args, "open_browser_gui", True)) and not (
        "--no-gui" in raw_arg_set or "--headless" in raw_arg_set
    )
    if not bool(args.headless):
        args.skip_renderer = False
    if int(args.tactile_vis_size) <= 0:
        parser.error("--tactile-vis-size must be positive.")
    if bool(args.no_camera) and bool(args.vis_cam):
        parser.error("--vis_cam requires cameras; remove --no-camera.")
    if float(args.wrist_depth_max_m) <= float(args.wrist_depth_min_m):
        parser.error("--wrist-depth-max-m must be greater than --wrist-depth-min-m.")
    wrist_cam_width = getattr(args, "wrist_cam_width", None)
    wrist_cam_height = getattr(args, "wrist_cam_height", None)
    if (wrist_cam_width is None) != (wrist_cam_height is None):
        parser.error("--wrist-cam-width and --wrist-cam-height must be set together.")
    if bool(args.no_camera) and (bool(args.wrist_depth_mask) or bool(args.wrist_depth_raw)):
        parser.error("--wrist-depth-mask/--wrist-depth-raw require cameras; remove --no-camera.")
    # The saved foot-pad topology only selects an additional input mapping.
    # It must not enable/disable Fix/Init controllers or GUI capabilities.
    args.foot_pad_config_count = configured_foot_pad_count()
    args.init_arm_disabled_reason = ""
    if bool(getattr(args, "init_arm_available", False)) and not str(
        getattr(args, "init_arm_profile", "") or ""
    ).strip():
        parser.error("--init-arm-available requires --init-arm-profile.")
    for side in ("left", "right"):
        try:
            _init_arm_qpos_speed_scale_for_side(args, side)
        except ValueError as exc:
            parser.error(str(exc))
    active_sides_for_special_arm = set(_active_sides_from_args(args))
    explicit_fix_arm = bool({"--fix-arm", "--no-fix-arm"} & raw_arg_set)
    explicit_init_arm = bool(
        {
            "--init-arm-available",
            "--init-arm-availabe",
            "--no-init-arm-available",
            "--no-init-arm-availabe",
        }
        & raw_arg_set
    )
    # The default special-arm controllers are meaningful only for the dual-arm
    # topology.  Preserve the historical single-arm behavior unless the user
    # explicitly requests one of them (which is rejected below with the clear
    # active-sides error).
    if active_sides_for_special_arm != {"left", "right"}:
        if not explicit_fix_arm:
            args.fix_arm = False
        if not explicit_init_arm:
            args.init_arm_available = False
    # Capture the effective operator request after topology defaults are
    # resolved.  Otherwise the parser's dual-arm default (True) survives as a
    # stale request in single-arm mode and startup incorrectly restores an
    # Init controller that the component builder must reject.
    args.init_arm_requested = bool(getattr(args, "init_arm_available", False))
    args.special_arm_control_mode_requested = getattr(args, "special_arm_control_mode", None)
    try:
        args.special_arm_control_mode = resolve_special_arm_control_mode(
            getattr(args, "special_arm_control_mode", None),
            fix_arm_enabled=bool(getattr(args, "fix_arm", False)),
            init_arm_enabled=bool(getattr(args, "init_arm_available", False)),
            foot_pad_count=getattr(args, "foot_pad_config_count", None),
            progress_enabled=bool(
                getattr(args, "collection_progress_profile", None) is not None
                or getattr(args, "collection_progress_gui", False)
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    special_arm_modes = {
        flag
        for flag, enabled in (
            ("--fix-arm", bool(getattr(args, "fix_arm", False))),
            ("--init-arm-available", bool(getattr(args, "init_arm_available", False))),
        )
        if enabled
    }
    if special_arm_modes and set(_active_sides_from_args(args)) != {"left", "right"}:
        parser.error(
            f"{', '.join(sorted(special_arm_modes))} requires --active-sides both."
        )
    if not str(getattr(args, "load_profile", "") or "").strip():
        latest_profile = _latest_vive_calibration_profile_path()
        if latest_profile is None:
            parser.error(
                "--load-profile was not provided and no calibration profile JSON was found in "
                f"{DEFAULT_CALIBRATION_PROFILE_DIR}"
            )
        args.load_profile = str(latest_profile)
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", TOOL_TCP_FRAME))
    args.tcp_mode = TOOL_TCP_FRAME
    active_sides = _active_sides_from_args(args)
    try:
        for side in active_sides:
            _hand_backend_mode_for_side(args, side)
        if _dmtac_sides(args, active_sides):
            _resolved_gripper_policy(args)
        _resolve_gripper_input_source(args, active_sides)
        _apply_default_arm_ips(args, active_sides=active_sides)
        _validate_required_arm_ips(args, active_sides)
        _save_explicit_arm_ips_latest(args, raw_argv=raw_argv)
        _run_gui_device_preflight(args, active_sides=active_sides, parser=parser)
        _apply_default_q6_ports(args, parser=parser)
    except ValueError as exc:
        parser.error(str(exc))
    if bool(args.visualize_tactile):
        if _dmtac_sides(args, active_sides):
            if getattr(args, "dmtac_tactile", None) is False:
                parser.error(
                    "--visualize-tactile with --hand-backend dmtac-gripper requires "
                    "DMTAC tactile reads; remove --dmtac-no-tactile."
                )
            args.dmtac_tactile = True
            requested_fields = tuple(getattr(args, "dmtac_tactile_fields", None) or ())
            args.dmtac_tactile_fields = normalize_dmtac_tactile_fields(
                (*requested_fields, "distributed_force")
            )
            args.dmtac_tactile_compute_profile = "dist-only"
            if getattr(args, "dmtac_tactile_calibration_policy", None) is None:
                args.dmtac_tactile_calibration_policy = "required"
        if any(
            _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE
            for side in active_sides
        ):
            tactile_regions = _parse_hand_tactile_regions(args.hand_tactile_regions)
            if not tactile_regions:
                parser.error(
                    "--visualize-tactile requires non-empty --hand-tactile-regions, "
                    "for example 'thumb_1,index_1,palm' or 'all'."
                )
            missing_hand_ip_sides = [
                side
                for side in active_sides
                if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE
                and not str(getattr(args, f"{side}_hand_ip") or "")
            ]
            if missing_hand_ip_sides:
                parser.error(
                    "--visualize-tactile requires real InspireTac hand IPs for active sides; "
                    f"missing={missing_hand_ip_sides}."
                )
    if not 0 <= int(args.vive_clock_sync_port) <= 65535:
        parser.error("--vive-clock-sync-port must be in [0, 65535].")
    if (
        int(args.vive_clock_sync_port)
        and int(args.vive_clock_sync_port) == int(args.port)
    ):
        parser.error("--vive-clock-sync-port must differ from the VIVE --port.")
    if bool(args.vive_clock_sync_required) and int(args.vive_clock_sync_port) <= 0:
        parser.error("--vive-clock-sync-required requires --vive-clock-sync-port.")
    if int(args.vive_clock_sync_min_samples) < 1:
        parser.error("--vive-clock-sync-min-samples must be >= 1.")
    if float(args.vive_clock_sync_max_rtt_ms) < 0.0:
        parser.error("--vive-clock-sync-max-rtt-ms must be >= 0.")
    if float(args.vive_clock_sync_stale_after_sec) < 0.0:
        parser.error("--vive-clock-sync-stale-after-sec must be >= 0.")
    if float(args.vive_clock_sync_startup_timeout_sec) < 0.0:
        parser.error("--vive-clock-sync-startup-timeout-sec must be >= 0.")
    return args


def parse_real_teleop_args() -> argparse.Namespace:
    """Preserve the frozen canonical parser entrypoint."""

    return _parse_real_teleop_args(build_real_teleop_argument_parser())


def parse_real_teleop_args_with_parser(
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Resolve a Station-extended canonical parser without duplicating rules."""

    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError("parser must be an argparse.ArgumentParser.")
    return _parse_real_teleop_args(parser)


__all__ = [
    "bind_legacy_support",
    "parse_real_teleop_args",
    "parse_real_teleop_args_with_parser",
]
