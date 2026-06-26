from __future__ import annotations

import argparse
import time

import numpy as np

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..arm_frames import TOOL_TCP_FRAME
from ..configs import ANNOTATED_DUAL_RM65_BASE_DISTANCE_M
from ..episodes import compare_start_qpos_pair, episode_start_qpos_pair, require_valid_episode
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    OPEN_LEFT_HAND_RENDER_QPOS,
    OPEN_RIGHT_HAND_RENDER_QPOS,
    RIGHT_HAND_RENDER_JOINT_NAMES,
)
from ..schemas import EffectorCommand
from .log_utils import ChangeGate, log_event
from rm65_inspire_tac.ik_tuning import load_ik_tuning_config
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.rendering import SapienArmHandRenderer


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _qpos_summary(qpos: np.ndarray | list[float] | None) -> str:
    if qpos is None:
        return "None"
    return str(np.round(np.asarray(qpos, dtype=float), 3).tolist())


def _state_qpos(backend: RM65ArmBackend) -> np.ndarray:
    qpos = backend.get_state().get("current_qpos")
    if qpos is None:
        raise RuntimeError(f"{backend.name} does not have current_qpos.")
    return np.asarray(qpos, dtype=float).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded teleop episode either in SAPIEN (virtual) or on real dual RM65 arms. "
            "Real replay validates start-state joint error before the first command is sent."
        )
    )
    parser.add_argument("episode", type=str, help="Path to episode directory or manifest.json")
    parser.add_argument(
        "--arm-mode",
        choices=["virtual", "real"],
        default="virtual",
        help="Replay in SAPIEN only, or dispatch joint replay to real RM65 hardware.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP for real replay.")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP for real replay.")
    parser.add_argument("--speed", type=_positive_float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--max-frames", type=_non_negative_int, default=0, help="Stop after N frames, 0 means full episode.")
    parser.add_argument(
        "--ik-tuning-json",
        type=str,
        default="",
        help="Optional RM65 IK tuning JSON used for start-state weighted joint error checks.",
    )
    parser.add_argument(
        "--max-start-joint-error-deg",
        type=_positive_float,
        default=5.0,
        help="Abort real replay if any joint differs from the episode start by more than this threshold.",
    )
    parser.add_argument(
        "--max-start-weighted-error-deg",
        type=_positive_float,
        default=12.0,
        help="Abort real replay if the weighted 6D start-state error exceeds this threshold.",
    )
    parser.add_argument(
        "--ignore-start-mismatch",
        action="store_true",
        help="Allow real replay to continue even when the start-state error check fails.",
    )
    parser.add_argument(
        "--start-check-only",
        action="store_true",
        help="Run the start-state validation and exit before replaying frames.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window.")
    parser.add_argument(
        "--skip-renderer",
        action="store_true",
        help="Skip renderer creation. Useful for headless real replay or validation-only runs.",
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=None,
        help="Optional base distance override for SAPIEN rendering.",
    )
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=1.0,
        help="Heartbeat interval for replay status logs.",
    )
    return parser.parse_args()


def _build_renderer(args: argparse.Namespace, *, base_distance: float) -> SapienArmHandRenderer | None:
    if args.skip_renderer:
        return None
    try:
        return SapienArmHandRenderer(
            left_urdf_path=require_asset_path("rm65_inspire_tac_left.urdf"),
            left_retargeting_joint_names=LEFT_HAND_RENDER_JOINT_NAMES,
            right_urdf_path=require_asset_path("rm65_inspire_tac_right.urdf"),
            right_retargeting_joint_names=RIGHT_HAND_RENDER_JOINT_NAMES,
            left_right_base_distance=base_distance,
            show_viewer=not args.headless,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to create the SAPIEN renderer. Check the current render device or rerun with --skip-renderer."
        ) from exc


def _render_current_state(
    renderer: SapienArmHandRenderer | None,
    *,
    left_backend: RM65ArmBackend,
    right_backend: RM65ArmBackend,
) -> None:
    if renderer is None:
        return
    renderer.render_data(
        left_qpos_arm=_state_qpos(left_backend),
        left_qpos_hand=OPEN_LEFT_HAND_RENDER_QPOS,
        right_qpos_arm=_state_qpos(right_backend),
        right_qpos_hand=OPEN_RIGHT_HAND_RENDER_QPOS,
        capture_image=False,
    )


def _start_state_summary(checks: dict[str, object]) -> str:
    parts: list[str] = []
    for side in ("left", "right"):
        check = checks[side]
        parts.append(
            f"{side}: max_abs={check.max_abs_joint_error_deg:.2f}deg "
            f"weighted={check.weighted_joint_error_deg:.2f}deg "
            f"within_limits={check.within_limits}"
        )
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    report = require_valid_episode(args.episode)
    manifest = report.manifest
    frames = report.frames
    metadata = manifest.get("metadata", {})
    recorded_start_qpos = episode_start_qpos_pair(manifest, frames)

    replay_hz = float(metadata.get("hz") or report.average_hz or 30.0)
    base_distance_value = (
        metadata.get("left_right_base_distance_m", ANNOTATED_DUAL_RM65_BASE_DISTANCE_M)
        if args.left_right_base_distance is None
        else args.left_right_base_distance
    )
    if base_distance_value is None:
        base_distance_value = ANNOTATED_DUAL_RM65_BASE_DISTANCE_M
    base_distance = float(base_distance_value)
    ik_tuning = load_ik_tuning_config(args.ik_tuning_json or metadata.get("ik_tuning_source") or None)
    joint_weights = np.asarray(ik_tuning.branch_normalization.joint_distance_weights, dtype=float)
    arm_urdf_path = require_asset_path("rm65_arm.urdf")

    if args.arm_mode == "real" and (not args.left_arm_ip or not args.right_arm_ip):
        raise SystemExit("--arm-mode real requires both --left-arm-ip and --right-arm-ip.")

    left_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0" if args.arm_mode == "virtual" else args.left_arm_ip,
            side="left",
            default_tcp_frame=TOOL_TCP_FRAME,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=bool(args.arm_mode == "real"),
            control_dt=1.0 / replay_hz,
            dry_run_fast=bool(args.arm_mode == "virtual"),
            initial_qpos=tuple(recorded_start_qpos["left"].tolist()) if args.arm_mode == "virtual" else None,
            ik_tuning=ik_tuning,
        )
    )
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0" if args.arm_mode == "virtual" else args.right_arm_ip,
            side="right",
            default_tcp_frame=TOOL_TCP_FRAME,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=bool(args.arm_mode == "real"),
            control_dt=1.0 / replay_hz,
            dry_run_fast=bool(args.arm_mode == "virtual"),
            initial_qpos=tuple(recorded_start_qpos["right"].tolist()) if args.arm_mode == "virtual" else None,
            ik_tuning=ik_tuning,
        )
    )
    renderer = None

    try:
        left_backend.start()
        right_backend.start()
        renderer = _build_renderer(args, base_distance=base_distance)

        current_qpos_pair = {
            "left": _state_qpos(left_backend),
            "right": _state_qpos(right_backend),
        }
        start_checks = compare_start_qpos_pair(
            current_qpos_pair=current_qpos_pair,
            recorded_qpos_pair=recorded_start_qpos,
            joint_distance_weights=joint_weights,
            max_abs_threshold_deg=float(args.max_start_joint_error_deg),
            weighted_threshold_deg=float(args.max_start_weighted_error_deg),
        )
        log_event("Replay", f"episode={report.episode_dir}")
        log_event(
            "Replay",
            f"mode={args.arm_mode} frames={len(frames)} duration={report.duration_sec:.3f}s speed={args.speed:.2f}x "
            f"recorded_arm_mode={metadata.get('arm_mode')}",
        )
        log_event("Replay", f"start-state check: {_start_state_summary(start_checks)}")
        for side in ("left", "right"):
            check = start_checks[side]
            log_event(
                "Replay",
                f"{side} current_q={_qpos_summary(check.current_qpos)} "
                f"recorded_start_q={_qpos_summary(check.recorded_qpos)} "
                f"abs_joint_error_deg={_qpos_summary(check.abs_joint_error_deg)}",
            )

        if args.arm_mode == "real":
            start_ok = all(check.within_limits for check in start_checks.values())
            if not start_ok and not args.ignore_start_mismatch:
                raise RuntimeError(
                    "Real replay aborted because the current start-state joint error exceeds the configured threshold."
                )

        if args.start_check_only:
            return

        _render_current_state(
            renderer,
            left_backend=left_backend,
            right_backend=right_backend,
        )

        status_gate = ChangeGate()
        replay_start_wall = time.time()
        frames_played = 0
        for frame_index, frame in enumerate(frames):
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            target_time = replay_start_wall + (float(frame["relative_time_sec"]) / float(args.speed))
            remaining = target_time - time.time()
            if remaining > 0.0:
                time.sleep(remaining)

            left_qpos = np.asarray(frame["arms"]["left"]["current_qpos"], dtype=float)
            right_qpos = np.asarray(frame["arms"]["right"]["current_qpos"], dtype=float)
            left_backend.send_command(
                EffectorCommand(
                    effector_name="left_arm",
                    side="left",
                    command_type="joint_vector",
                    value=left_qpos,
                )
            )
            right_backend.send_command(
                EffectorCommand(
                    effector_name="right_arm",
                    side="right",
                    command_type="joint_vector",
                    value=right_qpos,
                )
            )
            frames_played += 1

            _render_current_state(
                renderer,
                left_backend=left_backend,
                right_backend=right_backend,
            )

            if status_gate.should_emit(
                "replay_status",
                "running",
                now=time.time(),
                heartbeat_sec=float(args.status_sec),
            ):
                log_event(
                    "Replay",
                    f"frame={frame_index} rel_t={float(frame['relative_time_sec']):.3f}s "
                    f"state={frame.get('state')} left_q={_qpos_summary(left_qpos)} right_q={_qpos_summary(right_qpos)}",
                )

        log_event(
            "Replay",
            f"completed frames={frames_played} final_left_q={_qpos_summary(_state_qpos(left_backend))} "
            f"final_right_q={_qpos_summary(_state_qpos(right_backend))}",
        )
    finally:
        left_backend.close()
        right_backend.close()


if __name__ == "__main__":
    main()
