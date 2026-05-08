from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from project_paths import initial_alignment_states_dir

from ...backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ...backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ...calibration import TeleopCalibration
from ...configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from ...hand_mapping import parse_tactile_regions
from ...hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
    inspire_q6_to_render_qpos,
    open_hand_render_qpos,
)
from ...initial_alignment import AlignmentSideState, TeleopInitialAlignment
from ...tcp import HAND_BASE_TCP_FRAME, legacy_tcp_mode_requested
from ...transforms import identity_pose
from ..log_utils import DEFAULT_LOG_HEARTBEAT_SEC, ChangeGate, log_event, log_stage
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.rendering import SapienArmHandRenderer
from rm65_inspire_tac.pipeline import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS

DEFAULT_INITIAL_ALIGNMENT_STATES_DIR = initial_alignment_states_dir()


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


class TerminalQuitInput:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False
        self._save_requested = False

    def start(self) -> None:
        if not self.enabled:
            return
        self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def close(self) -> None:
        if self.enabled and self._old_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def poll(self) -> None:
        if not self.enabled:
            return
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return
            char = sys.stdin.read(1)
            if not char:
                return
            if char == "q":
                self.should_quit = True
            elif char == "s":
                self._save_requested = True

    def consume_save_request(self) -> bool:
        requested = self._save_requested
        self._save_requested = False
        return requested


@dataclass(frozen=True)
class RM65RealDebugComponents:
    calibration: TeleopCalibration
    left_backend: RM65ArmBackend | None
    right_backend: RM65ArmBackend | None
    left_hand_backend: InspireTacHandBackend | None
    right_hand_backend: InspireTacHandBackend | None
    renderer: SapienArmHandRenderer | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RM65 + InspireTac state visualizer. "
            "This tool only reads arm state and optional hand telemetry, mirrors the current "
            "state into SAPIEN for frame and pose inspection, and never sends motion commands."
        )
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP")
    parser.add_argument("--left-hand-ip", type=str, default="", help="Optional left InspireTac IP")
    parser.add_argument("--right-hand-ip", type=str, default="", help="Optional right InspireTac IP")
    parser.add_argument(
        "--arm-mode",
        choices=["real", "virtual"],
        default="real",
        help="Read real RM65 state when IPs are provided, or render pure virtual default poses.",
    )
    parser.add_argument(
        "--swap-arm-sides",
        action="store_true",
        help=(
            "Swap logical left/right assignment for the provided arm IPs. "
            "Useful when the physical arm connected as --left-arm-ip should appear as logical right, "
            "and vice versa."
        ),
    )
    parser.add_argument(
        "--tcp-mode",
        type=str,
        default=HAND_BASE_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Fallback base distance used when no calibration profile is loaded.",
    )
    parser.add_argument(
        "--load-profile",
        type=str,
        default="",
        help="Optional calibration profile. If omitted, the annotated dual-arm preset is used.",
    )
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="State polling and render rate.")
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=DEFAULT_LOG_HEARTBEAT_SEC,
        help="Terminal status heartbeat interval in seconds.",
    )
    parser.add_argument(
        "--debug-frame-axis-length",
        type=_positive_float,
        default=0.08,
        help="Axis length for desk/base/tcp debug frames.",
    )
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help="Max force value written to every InspireTac finger on startup.",
    )
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="all",
        help=(
            "Comma-separated tactile regions to poll, or 'all'. "
            "Defaults to all; pass an empty string to disable tactile reads."
        ),
    )
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window.")
    parser.add_argument("--skip-renderer", action="store_true", help="Skip SAPIEN renderer creation.")
    parser.add_argument(
        "--save-initial-alignment",
        type=str,
        default="",
        help=(
            "Optional output path or directory for the saved initial-alignment JSON. "
            "Press 's' during the run to save the current state."
        ),
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N steps, 0 means infinite.")
    args = parser.parse_args()
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", HAND_BASE_TCP_FRAME))
    args.tcp_mode = HAND_BASE_TCP_FRAME
    return args


def _build_backend(
    side: str,
    arm_ip: str,
    tcp_mode: str,
    arm_urdf_path: str,
    *,
    connect_hardware: bool,
    initial_qpos: tuple[float, ...] | None,
) -> RM65ArmBackend:
    return RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip=arm_ip,
            side=side,
            default_tcp_frame=tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=connect_hardware,
            initial_qpos=initial_qpos,
        )
    )


def _resolve_logical_side_assignments(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    assignments: dict[str, tuple[str, str]] = {}
    if args.left_arm_ip:
        assignments["left_arm_ip"] = (
            "right" if args.swap_arm_sides else "left",
            args.left_arm_ip,
        )
    if args.right_arm_ip:
        assignments["right_arm_ip"] = (
            "left" if args.swap_arm_sides else "right",
            args.right_arm_ip,
        )
    return assignments


def build_components(args: argparse.Namespace) -> RM65RealDebugComponents:
    hand_tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    if args.load_profile:
        calibration = TeleopCalibration.load_json(args.load_profile)
    else:
        calibration = build_annotated_dual_rm65_calibration(
            left_right_base_distance=args.left_right_base_distance
        )
    calibration.metadata.update(
        {
            "rm65_real_debug": True,
            "arm_mode": args.arm_mode,
            "tcp_mode": args.tcp_mode,
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
            "left_hand_ip": args.left_hand_ip,
            "right_hand_ip": args.right_hand_ip,
            "swap_arm_sides": bool(args.swap_arm_sides),
        }
    )

    arm_urdf_path = str(require_asset_path("rm65_arm.urdf"))
    side_assignments = _resolve_logical_side_assignments(args)
    left_backend = None
    right_backend = None
    for source_name, (logical_side, arm_ip) in side_assignments.items():
        default_qpos = (
            tuple(DEFAULT_LEFT_ARM_QPOS.tolist())
            if logical_side == "left"
            else tuple(DEFAULT_RIGHT_ARM_QPOS.tolist())
        )
        backend = _build_backend(
            logical_side,
            arm_ip,
            args.tcp_mode,
            arm_urdf_path,
            connect_hardware=bool(args.arm_mode == "real" and arm_ip),
            initial_qpos=None if args.arm_mode == "real" and arm_ip else default_qpos,
        )
        if logical_side == "left":
            left_backend = backend
        elif logical_side == "right":
            right_backend = backend
    if left_backend is None:
        left_backend = _build_backend(
            "left",
            "0.0.0.0",
            args.tcp_mode,
            arm_urdf_path,
            connect_hardware=False,
            initial_qpos=tuple(DEFAULT_LEFT_ARM_QPOS.tolist()),
        )
    if right_backend is None:
        right_backend = _build_backend(
            "right",
            "0.0.0.0",
            args.tcp_mode,
            arm_urdf_path,
            connect_hardware=False,
            initial_qpos=tuple(DEFAULT_RIGHT_ARM_QPOS.tolist()),
        )

    left_hand_backend = None
    if args.left_hand_ip:
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.left_hand_ip,
                side="left",
                connect_hardware=True,
                max_fingertip_force=int(args.max_fingertip_force),
                enable_telemetry=True,
                telemetry_hz=max(float(args.hz), 10.0),
                tactile_regions=hand_tactile_regions,
            )
        )
    right_hand_backend = None
    if args.right_hand_ip:
        right_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.right_hand_ip,
                side="right",
                connect_hardware=True,
                max_fingertip_force=int(args.max_fingertip_force),
                enable_telemetry=True,
                telemetry_hz=max(float(args.hz), 10.0),
                tactile_regions=hand_tactile_regions,
            )
        )

    renderer = None
    if not args.skip_renderer:
        renderer = SapienArmHandRenderer(
            left_urdf_path=str(require_asset_path("rm65_inspire_tac_left.urdf")) if left_backend is not None else None,
            left_retargeting_joint_names=LEFT_HAND_RENDER_JOINT_NAMES if left_backend is not None else None,
            right_urdf_path=str(require_asset_path("rm65_inspire_tac_right.urdf")) if right_backend is not None else None,
            right_retargeting_joint_names=RIGHT_HAND_RENDER_JOINT_NAMES if right_backend is not None else None,
            left_right_base_distance=args.left_right_base_distance,
            show_viewer=not args.headless,
            debug_frame_axis_length=args.debug_frame_axis_length,
        )
        renderer.reserve_debug_frames(
            ("desk", "left_arm_base", "right_arm_base", "left_tcp_actual", "right_tcp_actual")
        )

    return RM65RealDebugComponents(
        calibration=calibration,
        left_backend=left_backend,
        right_backend=right_backend,
        left_hand_backend=left_hand_backend,
        right_hand_backend=right_hand_backend,
        renderer=renderer,
    )


def _state_qpos(backend: RM65ArmBackend | None) -> np.ndarray | None:
    if backend is None:
        return None
    qpos = backend.get_state()["current_qpos"]
    if qpos is None:
        return None
    return np.asarray(qpos, dtype=float).copy()


def _tcp_pose_desk(backend: RM65ArmBackend | None, calibration: TeleopCalibration, side: str, tcp_mode: str) -> np.ndarray | None:
    if backend is None:
        return None
    arm = calibration.get_arm_for_side(side)
    if arm.T_D_B is None:
        return None
    try:
        pose_base = backend.compute_tcp_pose(tcp_mode)
    except Exception:
        return None
    return arm.T_D_B @ pose_base


def _hand_state_q6(backend: InspireTacHandBackend | None) -> np.ndarray | None:
    if backend is None:
        return None
    if backend.config.connect_hardware:
        try:
            joint_pos = backend.read_joint_pos_now()
        except Exception:
            joint_pos = backend.get_state().get("joint_pos")
    else:
        joint_pos = backend.get_state().get("joint_pos")
    if joint_pos is None:
        return None
    q6 = np.asarray(joint_pos, dtype=float).reshape(-1)
    if q6.shape != (6,) or not np.all(np.isfinite(q6)):
        return None
    return np.clip(q6, 0.0, 1.0)


def _hand_render_qpos(backend: InspireTacHandBackend | None, side: str) -> np.ndarray:
    q6 = _hand_state_q6(backend)
    if q6 is None:
        return open_hand_render_qpos(side)
    return inspire_q6_to_render_qpos(q6)


def _pose_summary(pose: np.ndarray | None) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if pose is None:
        return None
    xyz = tuple(np.round(np.asarray(pose[:3, 3], dtype=float), 4).tolist())
    rpy_deg = tuple(np.round(R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True), 1).tolist())
    return xyz, rpy_deg


def _log_side_status(
    side: str,
    state: dict[str, object],
    tcp_pose_desk: np.ndarray | None,
    *,
    gate: ChangeGate,
    heartbeat_sec: float,
    now: float,
) -> None:
    qpos = state.get("current_qpos")
    result_code = state.get("hardware_result_code")
    result_text = state.get("hardware_result_text")
    qpos_summary = None
    if qpos is not None:
        qpos_summary = tuple(np.round(np.asarray(qpos, dtype=float), 3).tolist())
    tcp_summary = _pose_summary(tcp_pose_desk)
    snapshot = (
        result_code,
        result_text,
        qpos is None,
        tcp_pose_desk is None,
    )
    if gate.should_emit(f"rm65_real_{side}", snapshot, now=now, heartbeat_sec=heartbeat_sec):
        message = f"{side} result={result_code}"
        if result_text not in (None, "", str(result_code)):
            message += f" ({result_text})"
        if qpos_summary is not None:
            message += f" q={list(qpos_summary)}"
        if tcp_summary is not None:
            xyz, rpy_deg = tcp_summary
            message += f" tcp_xyz={list(xyz)} tcp_rpy_deg={list(rpy_deg)}"
        log_event("RM65Real", message, now=now)


def _log_hand_status(
    side: str,
    backend: InspireTacHandBackend | None,
    *,
    gate: ChangeGate,
    heartbeat_sec: float,
    now: float,
) -> None:
    if backend is None:
        return
    state = backend.get_state()
    joint_pos = _hand_state_q6(backend)
    telemetry_error = state.get("telemetry_error")
    joint_summary = None
    if joint_pos is not None:
        joint_summary = tuple(np.round(np.asarray(joint_pos, dtype=float), 3).tolist())
    snapshot = (
        telemetry_error,
        joint_pos is None,
        joint_summary,
    )
    if gate.should_emit(f"rm65_real_hand_{side}", snapshot, now=now, heartbeat_sec=heartbeat_sec):
        message = f"{side}_hand"
        if telemetry_error:
            message += f" telemetry_error={telemetry_error}"
        elif joint_summary is not None:
            message += f" q6={list(joint_summary)}"
        else:
            message += " waiting_for_telemetry"
        log_event("RM65Real", message, now=now)


def _build_debug_frames(components: RM65RealDebugComponents, tcp_mode: str) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {"desk": identity_pose()}
    if components.left_backend is not None:
        left_arm = components.calibration.get_arm_for_side("left")
        if left_arm.T_D_B is not None:
            frames["left_arm_base"] = left_arm.T_D_B.copy()
        left_tcp = _tcp_pose_desk(components.left_backend, components.calibration, "left", tcp_mode)
        if left_tcp is not None:
            frames["left_tcp_actual"] = left_tcp
    if components.right_backend is not None:
        right_arm = components.calibration.get_arm_for_side("right")
        if right_arm.T_D_B is not None:
            frames["right_arm_base"] = right_arm.T_D_B.copy()
        right_tcp = _tcp_pose_desk(components.right_backend, components.calibration, "right", tcp_mode)
        if right_tcp is not None:
            frames["right_tcp_actual"] = right_tcp
    return frames


def _default_alignment_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_INITIAL_ALIGNMENT_STATES_DIR / f"{timestamp}_rm65_initial_alignment.json"


def _resolve_alignment_output_path(requested: str) -> Path:
    if not requested:
        return _default_alignment_output_path()
    path = Path(requested)
    if path.suffix.lower() == ".json":
        return path
    return path / _default_alignment_output_path().name


def _build_initial_alignment(
    components: RM65RealDebugComponents,
    *,
    tcp_mode: str,
    metadata: dict[str, object],
) -> TeleopInitialAlignment:
    sides: dict[str, AlignmentSideState] = {}
    for side, arm_backend, hand_backend in (
        ("left", components.left_backend, components.left_hand_backend),
        ("right", components.right_backend, components.right_hand_backend),
    ):
        arm_qpos = _state_qpos(arm_backend)
        tcp_pose_base = None if arm_backend is None else arm_backend.compute_tcp_pose(tcp_mode)
        tcp_pose_desk = _tcp_pose_desk(arm_backend, components.calibration, side, tcp_mode)
        hand_qpos_q6 = _hand_state_q6(hand_backend)
        sides[side] = AlignmentSideState(
            side=side,
            arm_qpos=arm_qpos,
            tcp_pose_base=tcp_pose_base,
            tcp_pose_desk=tcp_pose_desk,
            hand_qpos_q6=hand_qpos_q6,
            metadata={
                "arm_connected": bool(arm_backend is not None and arm_backend.config.connect_hardware),
                "hand_connected": bool(hand_backend is not None and hand_backend.config.connect_hardware),
            },
        )
    return TeleopInitialAlignment(
        tcp_frame=tcp_mode,
        sides=sides,
        metadata=metadata,
    )


def _save_initial_alignment(
    components: RM65RealDebugComponents,
    args: argparse.Namespace,
) -> Path:
    output_path = _resolve_alignment_output_path(args.save_initial_alignment)
    alignment = _build_initial_alignment(
        components,
        tcp_mode=args.tcp_mode,
        metadata={
            "saved_at_unix": time.time(),
            "saved_at_local": datetime.now().isoformat(timespec="seconds"),
            "tool": "rm65_real_debug",
            "arm_mode": args.arm_mode,
            "tcp_mode": args.tcp_mode,
            "load_profile": args.load_profile,
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
            "left_hand_ip": args.left_hand_ip,
            "right_hand_ip": args.right_hand_ip,
            "swap_arm_sides": bool(args.swap_arm_sides),
        },
    )
    alignment.save_json(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    components = build_components(args)
    key_input = TerminalQuitInput()
    status_gate = ChangeGate()

    log_stage("STAGE", "RM65 REAL READ-ONLY CHECK")
    log_event(
        "RM65Real",
        "READ-ONLY MODE: this tool never sends Movej_CANFD, InspireTac set_joint_pos, or any dynamic motion command.",
    )
    if legacy_tcp_mode_requested(args.requested_tcp_mode):
        log_event("RM65Real", "legacy --tcp-mode rm_eef was requested; runtime now uses hand_base only.")
    log_event(
        "RM65Real",
        f"arm_mode={args.arm_mode} tcp_mode={args.tcp_mode} status_sec={args.status_sec:.1f} s=save q=quit",
    )
    log_event("RM65Real", "viewer convention: screen-left = logical left_arm(-Y), screen-right = logical right_arm(+Y)")
    if args.load_profile:
        log_event("RM65Real", f"loaded calibration profile {args.load_profile}")
    else:
        log_event("RM65Real", "using annotated dual-arm preset calibration")
    try:
        left_arm = components.calibration.get_arm_for_side("left")
        right_arm = components.calibration.get_arm_for_side("right")
        if left_arm.T_D_B is not None and right_arm.T_D_B is not None:
            log_event(
                "RM65Real",
                "visual slots: "
                f"left_arm desk_y={left_arm.T_D_B[1, 3]:+.4f}, "
                f"right_arm desk_y={right_arm.T_D_B[1, 3]:+.4f}",
            )
    except Exception:
        pass
    side_assignments = _resolve_logical_side_assignments(args)
    if args.swap_arm_sides:
        log_event("RM65Real", "logical side assignment is SWAPPED for the provided arm IPs")
    for source_name, (logical_side, arm_ip) in side_assignments.items():
        log_event(
            "RM65Real",
            f"{source_name}={arm_ip} -> logical_{logical_side}_arm",
        )
    for side, backend in (("left", components.left_backend), ("right", components.right_backend)):
        if backend is None:
            continue
        if backend.config.connect_hardware:
            continue
        log_event("RM65Real", f"{side}_arm render state uses virtual default pose")
    for side, backend in (("left", components.left_hand_backend), ("right", components.right_hand_backend)):
        if backend is not None:
            log_event("RM65Real", f"{side}_hand telemetry enabled ip={backend.config.hand_ip}")
        else:
            log_event("RM65Real", f"{side}_hand render state uses fixed open pose")
    if args.save_initial_alignment:
        log_event("RM65Real", f"alignment save target base={_resolve_alignment_output_path(args.save_initial_alignment)}")
    else:
        log_event("RM65Real", f"alignment save target default_dir={_default_alignment_output_path().parent}")

    try:
        if components.left_backend is not None:
            components.left_backend.start()
        if components.right_backend is not None:
            components.right_backend.start()
        if components.left_hand_backend is not None:
            components.left_hand_backend.start()
        if components.right_hand_backend is not None:
            components.right_hand_backend.start()
        key_input.start()

        step_index = 0
        while True:
            loop_start = time.time()
            key_input.poll()
            if key_input.should_quit:
                return

            if key_input.consume_save_request():
                output_path = _save_initial_alignment(components, args)
                log_event("RM65Real", f"saved initial alignment -> {output_path}", now=loop_start)

            if components.left_backend is not None and components.left_backend.config.connect_hardware:
                left_state = components.left_backend.refresh_state_from_hardware()
            else:
                left_state = components.left_backend.get_state() if components.left_backend is not None else {}
            if components.right_backend is not None and components.right_backend.config.connect_hardware:
                right_state = components.right_backend.refresh_state_from_hardware()
            else:
                right_state = components.right_backend.get_state() if components.right_backend is not None else {}

            left_tcp = _tcp_pose_desk(components.left_backend, components.calibration, "left", args.tcp_mode)
            right_tcp = _tcp_pose_desk(components.right_backend, components.calibration, "right", args.tcp_mode)
            if components.left_backend is not None:
                _log_side_status(
                    "left",
                    left_state,
                    left_tcp,
                    gate=status_gate,
                    heartbeat_sec=args.status_sec,
                    now=loop_start,
                )
            if components.right_backend is not None:
                _log_side_status(
                    "right",
                    right_state,
                    right_tcp,
                    gate=status_gate,
                    heartbeat_sec=args.status_sec,
                    now=loop_start,
                )
            _log_hand_status(
                "left",
                components.left_hand_backend,
                gate=status_gate,
                heartbeat_sec=args.status_sec,
                now=loop_start,
            )
            _log_hand_status(
                "right",
                components.right_hand_backend,
                gate=status_gate,
                heartbeat_sec=args.status_sec,
                now=loop_start,
            )

            if components.renderer is not None:
                components.renderer.set_debug_frames(_build_debug_frames(components, args.tcp_mode))
                components.renderer.render_data(
                    left_qpos_arm=_state_qpos(components.left_backend),
                    left_qpos_hand=_hand_render_qpos(components.left_hand_backend, "left"),
                    right_qpos_arm=_state_qpos(components.right_backend),
                    right_qpos_hand=_hand_render_qpos(components.right_hand_backend, "right"),
                    capture_image=False,
                )

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                return
            time.sleep(max(0.0, (1.0 / args.hz) - (time.time() - loop_start)))
    finally:
        key_input.close()
        if components.left_backend is not None:
            components.left_backend.close()
        if components.right_backend is not None:
            components.right_backend.close()
        if components.left_hand_backend is not None:
            components.left_hand_backend.close()
        if components.right_hand_backend is not None:
            components.right_hand_backend.close()


if __name__ == "__main__":
    main()
