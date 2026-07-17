"""VIVE -> shared RM65 position retargeting -> MuJoCo debug teleoperation.

This is deliberately a thin debug composition root.  Device-neutral IK and
qref/qcmd execution live in ``robot_motion`` and ``sim`` respectively; no real
arm or deployment frontend is imported here.  MuJoCo currently executes
prescribed qpos, so this validates retargeting/timing/collision semantics but
not a physical RM65 servo model.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    IKObjectiveWeights,
    load_ik_tuning_config,
    load_rm65_tool_profile,
    tool_tcp_pose_flange_m_rad,
)
from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.scene import MujocoScene

from ..calibration import TeleopCalibration
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
    default_vive_tracker_configs,
)
from ..configs.pose_smoothing import (
    build_bimanual_pose_smoother,
    load_default_pose_smoothing_config,
)
from ..debug.calibration_flow import average_pose
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    tracker_hand_to_rm65_tcp_rotation,
    vive_tracker_height_correction_offset,
)
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from ..safety import TeleopSafetyConfig
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.pose import DEFAULT_VIVE_CLOCK_SYNC_PORT, VivePoseSource
from ..sources.pose.vive_motion import TrackerMotionIntentConfig
from ..transforms import as_pose_matrix


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONTRACT = _REPO_ROOT / "sim/configs/rm65_bimanual_physics_v1.json"


def _positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def _unit_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug the real-aligned 15 Hz target / 60 Hz IK-qref / 120 Hz qcmd "
            "pipeline against a 1000 Hz prescribed-qpos MuJoCo scene."
        )
    )
    parser.add_argument("--contract", type=Path, default=_DEFAULT_CONTRACT)
    parser.add_argument("--load-profile", type=Path, default=None)
    parser.add_argument(
        "--left-right-base-distance",
        type=_positive_float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    )
    parser.add_argument(
        "--left-tool-profile",
        required=True,
        help="RM65 tool calibration profile/path used for left tool-TCP IK.",
    )
    parser.add_argument(
        "--right-tool-profile",
        required=True,
        help="RM65 tool calibration profile/path used for right tool-TCP IK.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--vive-clock-sync-port", type=int, default=DEFAULT_VIVE_CLOCK_SYNC_PORT
    )
    parser.add_argument("--left-tracker-name", default="tracker_1")
    parser.add_argument("--right-tracker-name", default="tracker_2")
    parser.add_argument(
        "--control-mode",
        choices=("absolute",),
        default="absolute",
        help="The active MuJoCo runtime uses absolute pose mapping.",
    )
    parser.add_argument("--translation-scale", type=_positive_float, default=1.0)
    parser.add_argument(
        "--vive-tracker-height-correction-m",
        type=float,
        default=DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m,
    )
    parser.add_argument(
        "--no-vive-tracker-height-correction",
        action="store_true",
        help="Disable the standard local tracker-height correction.",
    )
    parser.add_argument(
        "--position-inactive-enter-threshold-scale",
        type=_unit_float,
        default=0.5,
    )
    parser.add_argument(
        "--position-ik-orientation-cost",
        type=_positive_float,
        default=1.0,
        help="Dual-arm IK position cost stays 1.0; this sets orientation cost.",
    )
    parser.add_argument("--position-ik-damping", type=_positive_float, default=0.03)
    parser.add_argument("--position-ik-solve-max-iterations", type=int, default=4)
    parser.add_argument("--position-ik-solve-min-iterations", type=int, default=1)
    parser.add_argument(
        "--no-position-ik-adaptive-iterations", action="store_true"
    )
    parser.add_argument(
        "--position-ik-iteration-reserve-ms", type=float, default=5.0
    )
    parser.add_argument(
        "--position-joint-velocity-limit-profile",
        default="rm65_position_8x",
    )
    parser.add_argument(
        "--position-ik-local-step-clip-scale", type=_positive_float, default=8.0
    )
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-alpha", type=_unit_float, default=0.14
    )
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-fast-alpha",
        type=_unit_float,
        default=0.45,
    )
    parser.add_argument(
        "--position-max-command-step-deg", type=_positive_float, default=0.675
    )
    parser.add_argument("--arm-follow-qpos-alpha", type=_unit_float, default=0.9)
    parser.add_argument(
        "--position-target-velocity-filter-tau-sec", type=float, default=0.03
    )
    parser.add_argument("--calibration-samples", type=int, default=30)
    parser.add_argument("--calibration-timeout", type=_positive_float, default=20.0)
    parser.add_argument(
        "--auto-home",
        action="store_true",
        help="Capture runtime home as soon as valid steady tracker samples arrive.",
    )
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--status-hz", type=_positive_float, default=2.0)
    args = parser.parse_args(argv)
    if args.vive_tracker_height_correction_m < 0.0:
        parser.error("--vive-tracker-height-correction-m must be >= 0")
    if args.position_ik_solve_min_iterations <= 0:
        parser.error("--position-ik-solve-min-iterations must be positive")
    if args.position_ik_solve_max_iterations < args.position_ik_solve_min_iterations:
        parser.error("IK max iterations must be >= min iterations")
    if args.position_ik_iteration_reserve_ms < 0.0:
        parser.error("--position-ik-iteration-reserve-ms must be >= 0")
    if args.position_target_velocity_filter_tau_sec < 0.0:
        parser.error("--position-target-velocity-filter-tau-sec must be >= 0")
    if args.calibration_samples <= 0:
        parser.error("--calibration-samples must be positive")
    if args.max_seconds < 0.0:
        parser.error("--max-seconds must be >= 0")
    return args


def _position_timing(base: TimingSpec) -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=base.physics_hz,
        camera_hz=base.camera_hz,
        interpolation=base.interpolation,
        command_hold=base.command_hold,
        parameter_source="vive_mujoco_position_15_60_120_debug",
    )


def _load_calibration(args: argparse.Namespace) -> TeleopCalibration:
    if args.load_profile is not None:
        return TeleopCalibration.load_json(args.load_profile)
    return build_annotated_dual_rm65_calibration(
        left_right_base_distance=float(args.left_right_base_distance)
    )


def _load_tool_tcp(profile_arg: str, side: str) -> np.ndarray:
    profile = load_rm65_tool_profile(profile_arg, side=side)
    return tool_tcp_pose_flange_m_rad(profile)


def _close_scene(scene: MujocoScene | None) -> None:
    # MjModel/MjData do not require explicit disposal. Keep this seam for a
    # future scene owner that may acquire render/context resources.
    close = None if scene is None else getattr(scene, "close", None)
    if callable(close):
        close()


def _position_ik_tuning(orientation_cost: float):
    tuning = load_ik_tuning_config()
    mode = "vive_mujoco_runtime_override"
    modes = dict(tuning.objective_modes)
    modes[mode] = IKObjectiveWeights(
        single_arm_position_weight=1.0,
        single_arm_orientation_weight=float(orientation_cost),
        dual_arm_position_cost=1.0,
        dual_arm_orientation_cost=float(orientation_cost),
    )
    return replace(tuning, ik_objective_mode=mode, objective_modes=modes)


def _motion_intent_config(scale: float) -> TrackerMotionIntentConfig:
    base = TrackerMotionIntentConfig()
    return replace(
        base,
        enter_linear_speed_m_s=base.enter_linear_speed_m_s * scale,
        enter_angular_speed_rad_s=base.enter_angular_speed_rad_s * scale,
        enter_translation_span_m=base.enter_translation_span_m * scale,
        enter_rotation_span_rad=base.enter_rotation_span_rad * scale,
    )


def _pose_in_desk(channel, calibration: TeleopCalibration) -> np.ndarray:
    pose = as_pose_matrix(channel.value)
    frame_name = channel.frame_name or ""
    tracker_role = (channel.metadata or {}).get("tracker_role")
    if frame_name in {"vive_world", "world"}:
        return calibration.world_to_desk(
            pose,
            tracker_role=str(tracker_role) if tracker_role else None,
        )
    if frame_name.endswith("_operator") or frame_name == "operator":
        return calibration.operator_to_desk(pose)
    if frame_name in {calibration.desk_frame.name, "desk"}:
        return pose
    raise ValueError(f"Unsupported tracker frame {frame_name or '<none>'!r}.")


def _capture_runtime_home(
    *,
    session: TeleopSession,
    rig: MujocoPositionExecutionRig,
    dual: DualArmCollisionAwareRetargeter,
    pose_retargeters: dict[str, PoseToArmRetargeter],
    calibration: TeleopCalibration,
    samples_required: int,
    timeout_sec: float,
) -> None:
    samples: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    deadline = time.monotonic() + timeout_sec
    while any(len(values) < samples_required for values in samples.values()):
        if time.monotonic() >= deadline:
            counts = {side: len(values) for side, values in samples.items()}
            raise TimeoutError(f"Timed out waiting for VIVE runtime-home samples: {counts}")
        channels, _ = session.poll_sources(time.time())
        for side in ("left", "right"):
            channel = channels.get(f"{side}_wrist_pose")
            if channel is None or not channel.valid or len(samples[side]) >= samples_required:
                continue
            pose_desk = _pose_in_desk(channel, calibration)
            samples[side].append(
                pose_retargeters[side].prepare_input_pose_desk(pose_desk)
            )
        time.sleep(0.01)

    accepted = rig.accepted_qcmd_by_side()
    dual.set_current_qpos_pair(
        left_qpos=accepted["left"],
        right_qpos=accepted["right"],
        reset_tracking=True,
    )
    robot_home = dual.current_tool_tcp_world_targets()
    for side in ("left", "right"):
        human_home = average_pose(samples[side])
        pose_retargeters[side].set_home_anchors(
            human_pose_desk=human_home,
            robot_pose_desk=robot_home[side],
        )
        calibration.capture_home_pose(f"{side}_arm", robot_home[side])
        calibration.capture_home_qpos(f"{side}_arm", accepted[side])
    print(
        "[vive-mujoco] runtime home captured "
        f"samples={samples_required} left_xyz={robot_home['left'][:3, 3].round(4).tolist()} "
        f"right_xyz={robot_home['right'][:3, 3].round(4).tolist()}"
    )


def _build(args: argparse.Namespace):
    base_contract = SimulationContract.load(args.contract)
    timing = _position_timing(base_contract.timing)
    contract = replace(base_contract, timing=timing)
    scene = MujocoScene.from_contract(contract)
    try:
        # This is a simulation reset, not a commanded transition from the
        # URDF's all-zero qpos. Validate the final reset state directly; a
        # swept projector would otherwise stop on collisions along that
        # artificial zero-to-home path and silently start from the wrong pose.
        scene.set_qpos_by_side(
            {"left": DEFAULT_LEFT_ARM_QPOS, "right": DEFAULT_RIGHT_ARM_QPOS}
        )
        projector = scene.collision_projector()
        if projector is not None:
            check = projector.check_qpos(
                scene.data,
                scene.kinematic_controlled_qpos(),
            )
            if not check.safe:
                raise RuntimeError(
                    "Configured MuJoCo runtime-home qpos is collision-unsafe: "
                    f"{check.to_dict()}"
                )
        initial = {
            side: scene.get_arm_qpos(side).copy() for side in ("left", "right")
        }
        calibration = _load_calibration(args)
        tracker_offset = None
        if not args.no_vive_tracker_height_correction:
            tracker_offset = vive_tracker_height_correction_offset(
                float(args.vive_tracker_height_correction_m)
            )
        intent = _motion_intent_config(
            float(args.position_inactive_enter_threshold_scale)
        )
        pose_retargeters = {
            side: PoseToArmRetargeter(
                PoseToArmRetargeterConfig(
                    side=side,
                    input_channel_name=f"{side}_wrist_pose",
                    effector_name=f"{side}_arm",
                    control_mode=args.control_mode,
                    translation_scale=float(args.translation_scale),
                    input_local_translation_offset=tracker_offset,
                    input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
                    tracker_motion_intent_enabled=True,
                    tracker_motion_intent_config=intent,
                )
            )
            for side in ("left", "right")
        }
        dual = DualArmCollisionAwareRetargeter(
            DualArmCollisionAwareRetargeterConfig(
                control_mode=args.control_mode,
                translation_scale=float(args.translation_scale),
                solver_frequency=timing.ik_hz,
                damping=float(args.position_ik_damping),
                solve_max_iterations=int(args.position_ik_solve_max_iterations),
                solve_min_iterations=int(args.position_ik_solve_min_iterations),
                solve_adaptive_iterations_enabled=(
                    not args.no_position_ik_adaptive_iterations
                ),
                solve_iteration_reserve_ms=float(args.position_ik_iteration_reserve_ms),
                bimanual_result_contract_enabled=True,
                enforce_realistic_joint_step=True,
                left_initial_qpos=initial["left"],
                right_initial_qpos=initial["right"],
                left_tool_tcp_pose_flange_m_rad=_load_tool_tcp(
                    args.left_tool_profile, "left"
                ),
                right_tool_tcp_pose_flange_m_rad=_load_tool_tcp(
                    args.right_tool_profile, "right"
                ),
                ik_tuning=_position_ik_tuning(args.position_ik_orientation_cost),
                joint_velocity_limit_profile=(
                    args.position_joint_velocity_limit_profile
                ),
                target_pose_smoother=build_bimanual_pose_smoother(
                    load_default_pose_smoothing_config()
                ),
                output_qpos_smoothing_alpha=float(
                    args.position_post_ik_qpos_smoothing_alpha
                ),
                output_qpos_smoothing_adaptive_enabled=True,
                output_qpos_smoothing_fast_alpha=float(
                    args.position_post_ik_qpos_smoothing_fast_alpha
                ),
                prefer_non_wrist_singular_branch=True,
                prefer_non_elbow_singular_branch=True,
                local_step_candidate_clip_scale=float(
                    args.position_ik_local_step_clip_scale
                ),
                inactive_side_hold_execution_release_enabled=True,
            ),
            left_pose_retargeter=pose_retargeters["left"],
            right_pose_retargeter=pose_retargeters["right"],
        )
        rig = MujocoPositionExecutionRig(
            scene,
            MujocoPositionExecutionConfig(
                timing=timing,
                max_command_step_deg=float(args.position_max_command_step_deg),
                target_velocity_filter_tau_sec=float(
                    args.position_target_velocity_filter_tau_sec
                ),
                follow_qpos_alpha=float(args.arm_follow_qpos_alpha),
            ),
        )
        source = VivePoseSource(
            tracker_configs=default_vive_tracker_configs(
                left_tracker_name=args.left_tracker_name,
                right_tracker_name=args.right_tracker_name,
            ),
            host=args.host,
            port=args.port,
            clock_sync_port=args.vive_clock_sync_port,
        )
        session = TeleopSession(
            TeleopSessionConfig(
                mode="vive_mujoco_position_debug",
                tcp_mode="tool_tcp",
                record_enabled=False,
                allow_hardware_after_debug_pass=True,
                tracker_channel_names=("left_wrist_pose", "right_wrist_pose"),
            ),
            sources=[source],
            retargeters=[dual],
            rig=rig,
            calibration=calibration,
            safety_config=TeleopSafetyConfig(
                freeze_on_fault=True,
                required_channels=("left_wrist_pose", "right_wrist_pose"),
                invalid_frame_limit=3,
            ),
        )
        return session, rig, dual, pose_retargeters, timing
    except BaseException:
        _close_scene(scene)
        raise


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session = None
    scene = None
    try:
        session, rig, dual, pose_retargeters, timing = _build(args)
        scene = rig.scene
        session.start()
        print(
            "[vive-mujoco] execution=kinematic_direct_qpos_debug "
            f"rates={timing.sample_hz:g}/{timing.ik_hz:g}/"
            f"{timing.command_hz:g}/{timing.physics_hz:g}Hz"
        )
        print(
            "[vive-mujoco] MuJoCo is the simulation chain; no real-arm, force-control, "
            "dataset, or deployment process is started."
        )
        if not args.auto_home and sys.stdin.isatty():
            input(
                "Place both trackers at runtime home, keep them steady, then press Enter... "
            )
        _capture_runtime_home(
            session=session,
            rig=rig,
            dual=dual,
            pose_retargeters=pose_retargeters,
            calibration=session.calibration,
            samples_required=int(args.calibration_samples),
            timeout_sec=float(args.calibration_timeout),
        )
        session.set_calibration_ready()
        session.mark_debug_passed()
        session.arm_hardware()

        started = time.monotonic()
        next_status = started
        period = 1.0 / timing.ik_hz
        while args.max_seconds <= 0.0 or time.monotonic() - started < args.max_seconds:
            loop_started = time.monotonic()
            result = session.run_step(now=time.time())
            accepted = rig.accepted_qcmd_by_side()
            dual.set_current_qpos_pair(
                left_qpos=accepted["left"],
                right_qpos=accepted["right"],
                reset_tracking=False,
            )
            if loop_started >= next_status:
                snapshot = rig.snapshot()
                print(
                    "[vive-mujoco] "
                    f"state={result.state.value} control_hz={result.control_hz:.1f} "
                    f"dispatched={result.command_dispatched} faults={len(result.faults)} "
                    f"qref_seq={snapshot['last_reference_sequence']} "
                    f"physics_ticks={snapshot['physics_ticks']} "
                    f"deadline_misses={snapshot['deadline_misses']}"
                )
                next_status = loop_started + 1.0 / float(args.status_hz)
            remaining = period - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[vive-mujoco] stopped")
    finally:
        if session is not None:
            session.close()
        _close_scene(scene)


if __name__ == "__main__":
    main()
