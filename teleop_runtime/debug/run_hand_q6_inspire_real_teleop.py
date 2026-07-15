from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import numpy as np

from ..backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ..configs import ANNOTATED_DUAL_RM65_BASE_DISTANCE_M
from ..hand_mapping import (
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    TARGET_INSPIRE_HAND_Q6_ORDER,
    parse_tactile_regions,
)
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
    inspire_q6_to_render_qpos,
    open_hand_render_qpos,
)
from ..rigs import HandOnlyRig
from ..retargeters import JointsToDexHandRetargeter, JointsToDexHandRetargeterConfig
from ..safety import TeleopSafetyConfig
from ..schemas import TeleopCommand
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.hand import HandQ6Source
from .log_utils import ChangeGate, log_event, log_stage
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.pipeline import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
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


def _alpha_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1]")
    return value


def _parse_joint_order(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("joint order must contain exactly 6 comma-separated indices")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid joint order: {text}") from exc
    if sorted(values) != [0, 1, 2, 3, 4, 5]:
        raise argparse.ArgumentTypeError("joint order must be a permutation of 0,1,2,3,4,5")
    return values


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    return parse_tactile_regions(value)


def _q6_summary(value: object) -> str:
    if value is None:
        return "None"
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        return "None"
    return str(np.round(array, 3).tolist())


class TerminalQuitInput:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except (OSError, termios.error):
            self.enabled = False
            self._old_attrs = None

    def close(self) -> None:
        if self.enabled and self._old_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            except (OSError, termios.error):
                pass
            self._old_attrs = None

    def poll(self) -> None:
        if not self.enabled:
            return
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                log_event("Keys", "terminal input became unavailable; keyboard controls disabled for this run.")
                return
            if not ready:
                return
            try:
                char = sys.stdin.read(1)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                log_event("Keys", "terminal input became unavailable; keyboard controls disabled for this run.")
                return
            if not char:
                return
            if char == "q":
                self.should_quit = True
                return


@dataclass(frozen=True)
class HandOnlyComponents:
    session: TeleopSession
    left_hand_backend: InspireTacHandBackend | None
    right_hand_backend: InspireTacHandBackend | None
    left_hand_control_via_rig: bool
    right_hand_control_via_rig: bool
    renderer: SapienArmHandRenderer | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one or two 6D hand teleop serial devices and drive real InspireTac hands. "
            "Both RM65 arms remain fixed in the renderer and are never commanded."
        )
    )
    parser.add_argument("--left-hand-ip", type=str, default="", help="Optional left InspireTac IP")
    parser.add_argument("--right-hand-ip", type=str, default="", help="Optional right InspireTac IP")
    parser.add_argument("--left-hand-port", type=str, default="", help="Optional left hand teleop serial port")
    parser.add_argument("--right-hand-port", type=str, default="", help="Optional right hand teleop serial port")
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200, help="Hand teleop serial baudrate")
    parser.add_argument("--hand-reader-timeout", type=float, default=0.05, help="Hand teleop serial timeout in seconds")
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Re-run exoskeleton hand zero-pose calibration and update the fixed calibration file.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=str,
        default="",
        help="Optional fixed exoskeleton hand zero-pose calibration JSON path.",
    )
    parser.add_argument("--show-hand-raw-lines", action="store_true", help="Print every raw line from hand teleop serial devices")
    parser.add_argument(
        "--hand-joint-order",
        type=_parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help=(
            "Permutation applied to input q6 before mapping to InspireTac q6. "
            f"Default maps into InspireTac order {list(TARGET_INSPIRE_HAND_Q6_ORDER)}."
        ),
    )
    parser.add_argument(
        "--hand-invert",
        dest="hand_invert",
        action="store_true",
        help="Invert normalized q6 before sending to the InspireTac hand. Default is enabled.",
    )
    parser.add_argument(
        "--hand-no-invert",
        dest="hand_invert",
        action="store_false",
        help="Disable q6 inversion if the hand already moves in the correct direction.",
    )
    parser.set_defaults(hand_invert=True)
    parser.add_argument(
        "--hand-command-alpha",
        type=_alpha_float,
        default=1.0,
        help="EMA smoothing factor for hand q6 mapping.",
    )
    parser.add_argument(
        "--hand-send-threshold",
        type=float,
        default=0.002,
        help="Only send/update hand q6 when max delta exceeds this threshold.",
    )
    parser.add_argument(
        "--hand-telemetry-hz",
        type=_positive_float,
        default=30.0,
        help="Telemetry polling rate for each real InspireTac hand backend.",
    )
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="all",
        help=(
            "Comma-separated tactile regions to poll, for example "
            "'thumb_1,index_1,palm' or 'all'. Defaults to all; pass an empty string to skip tactile reads."
        ),
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
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target control/update rate.")
    parser.add_argument(
        "--minimum-control-hz",
        type=_positive_float,
        default=10.0,
        help="Emit low-rate faults when measured control rate drops below this threshold.",
    )
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=1.0,
        help="Heartbeat interval for terminal status logs.",
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Renderer-only base distance used for the fixed RM65 arm poses.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window.")
    parser.add_argument("--skip-renderer", action="store_true", help="Skip SAPIEN renderer creation.")
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N steps, 0 means infinite.")
    args = parser.parse_args()
    if not args.left_hand_port and not args.right_hand_port:
        raise SystemExit(
            "Hand-only teleop requires at least one --left-hand-port or --right-hand-port."
        )
    return args


def _hand_state_q6(backend: InspireTacHandBackend | None) -> np.ndarray | None:
    if backend is None:
        return None
    state = backend.get_state()
    joint_pos = state.get("joint_pos")
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


def _command_q6(command: TeleopCommand | None, side: str) -> np.ndarray | None:
    if command is None:
        return None
    effector_command = command.commands.get(f"{side}_hand")
    if effector_command is None:
        return None
    q6 = np.asarray(effector_command.value, dtype=float).reshape(-1)
    if q6.shape != (6,) or not np.all(np.isfinite(q6)):
        return None
    return np.clip(q6, 0.0, 1.0)


def _input_q6(channels: dict[str, object], side: str) -> np.ndarray | None:
    channel = channels.get(f"{side}_hand_q6")
    if channel is None or not getattr(channel, "valid", False):
        return None
    q6 = np.asarray(channel.value, dtype=float).reshape(-1)
    if q6.shape != (6,) or not np.all(np.isfinite(q6)):
        return None
    return np.clip(q6, 0.0, 1.0)


def _build_components(args: argparse.Namespace) -> HandOnlyComponents:
    hand_source = HandQ6Source.from_bimanual(
        left_port=args.left_hand_port or None,
        right_port=args.right_hand_port or None,
        baudrate=int(args.hand_reader_baudrate),
        timeout=float(args.hand_reader_timeout),
        show_raw_lines=bool(args.show_hand_raw_lines),
        exo_hand_calibration=bool(args.exo_hand_calibration),
        calibration_path=str(args.exo_hand_calibration_path or "") or None,
        critical=False,
        strict_expected=False,
    )

    retargeters = []
    left_hand_retargeter = None
    if args.left_hand_port:
        left_hand_retargeter = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side="left",
                input_channel_name="left_hand_q6",
                effector_name="left_hand",
                joint_order=tuple(args.hand_joint_order),
                invert=bool(args.hand_invert),
                alpha=float(args.hand_command_alpha),
            )
        )
        retargeters.append(left_hand_retargeter)
    right_hand_retargeter = None
    if args.right_hand_port:
        right_hand_retargeter = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side="right",
                input_channel_name="right_hand_q6",
                effector_name="right_hand",
                joint_order=tuple(args.hand_joint_order),
                invert=bool(args.hand_invert),
                alpha=float(args.hand_command_alpha),
            )
        )
        retargeters.append(right_hand_retargeter)

    hand_tactile_regions = _parse_hand_tactile_regions(args.hand_tactile_regions)
    left_hand_backend = None
    if args.left_hand_port or args.left_hand_ip:
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.left_hand_ip or "0.0.0.0",
                side="left",
                connect_hardware=bool(args.left_hand_ip),
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(args.left_hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=hand_tactile_regions,
            )
        )
    right_hand_backend = None
    if args.right_hand_port or args.right_hand_ip:
        right_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.right_hand_ip or "0.0.0.0",
                side="right",
                connect_hardware=bool(args.right_hand_ip),
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(args.right_hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=hand_tactile_regions,
            )
        )

    left_hand_control_via_rig = left_hand_retargeter is not None and left_hand_backend is not None
    right_hand_control_via_rig = right_hand_retargeter is not None and right_hand_backend is not None
    rig_backends: dict[str, InspireTacHandBackend] = {}
    if left_hand_control_via_rig:
        rig_backends["left_hand"] = left_hand_backend
    if right_hand_control_via_rig:
        rig_backends["right_hand"] = right_hand_backend

    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="hand_q6_inspire_real_teleop",
            record_enabled=False,
            non_freezing_fault_codes=("low_control_rate",),
        ),
        sources=[hand_source],
        retargeters=retargeters,
        rig=HandOnlyRig(
            backends=rig_backends,
            name="hand_only_real_rig",
            parallel_dispatch=len(rig_backends) > 1,
            parallel_workers=max(2, len(rig_backends)),
        ),
        safety_config=TeleopSafetyConfig(
            freeze_on_fault=True,
            required_channels=(),
            invalid_frame_limit=3,
            minimum_control_hz=float(args.minimum_control_hz),
        ),
    )

    renderer = None
    if not args.skip_renderer:
        renderer = SapienArmHandRenderer(
            left_urdf_path=str(require_asset_path("rm65_inspire_tac_left.urdf")),
            left_retargeting_joint_names=LEFT_HAND_RENDER_JOINT_NAMES,
            right_urdf_path=str(require_asset_path("rm65_inspire_tac_right.urdf")),
            right_retargeting_joint_names=RIGHT_HAND_RENDER_JOINT_NAMES,
            left_right_base_distance=args.left_right_base_distance,
            show_viewer=not args.headless,
        )

    return HandOnlyComponents(
        session=session,
        left_hand_backend=left_hand_backend,
        right_hand_backend=right_hand_backend,
        left_hand_control_via_rig=left_hand_control_via_rig,
        right_hand_control_via_rig=right_hand_control_via_rig,
        renderer=renderer,
    )


def _side_backend(components: HandOnlyComponents, side: str) -> InspireTacHandBackend | None:
    return components.left_hand_backend if side == "left" else components.right_hand_backend


def _log_side_status(
    side: str,
    *,
    input_q6: np.ndarray | None,
    command_q6: np.ndarray | None,
    backend: InspireTacHandBackend | None,
    gate: ChangeGate,
    heartbeat_sec: float,
    now: float,
) -> None:
    if backend is None:
        return
    state = backend.get_state()
    actual_q6 = _hand_state_q6(backend)
    tactile_dict = state.get("tactile_dict")
    tactile_count = 0 if not isinstance(tactile_dict, dict) else len(tactile_dict)
    snapshot = (
        _q6_summary(input_q6),
        _q6_summary(command_q6),
        _q6_summary(actual_q6),
        state.get("telemetry_error"),
        tactile_count,
        backend.config.connect_hardware,
    )
    if gate.should_emit(f"hand_only_{side}", snapshot, now=now, heartbeat_sec=heartbeat_sec):
        message = (
            f"{side} input={_q6_summary(input_q6)} "
            f"cmd={_q6_summary(command_q6)} "
            f"actual={_q6_summary(actual_q6)} "
            f"mode={'real' if backend.config.connect_hardware else 'preview'} "
            f"tactile_regions={tactile_count}"
        )
        telemetry_error = state.get("telemetry_error")
        if telemetry_error:
            message += f" telemetry_error={telemetry_error}"
        log_event("HandOnly", message, now=now)


def main() -> None:
    args = parse_args()
    components = _build_components(args)
    key_input = TerminalQuitInput()
    status_gate = ChangeGate()

    log_stage("STAGE", "HAND Q6 TO INSPIRETAC REAL TELEOP")
    log_event(
        "HandOnly",
        "This tool only controls InspireTac hands. RM65 arms stay fixed in the renderer and no arm motion command is sent.",
    )
    log_event(
        "HandOnly",
        f"left_port={args.left_hand_port or None} right_port={args.right_hand_port or None} "
        f"left_hand_ip={args.left_hand_ip or None} right_hand_ip={args.right_hand_ip or None}",
    )
    log_event(
        "HandOnly",
        f"joint_order={list(args.hand_joint_order)} invert={bool(args.hand_invert)} "
        f"alpha={float(args.hand_command_alpha):.3f} send_threshold={float(args.hand_send_threshold):.4f}",
    )
    log_event(
        "HandOnly",
        f"telemetry_hz={float(args.hand_telemetry_hz):.1f} "
        f"tactile_regions={list(_parse_hand_tactile_regions(args.hand_tactile_regions)) or []} "
        f"status_sec={float(args.status_sec):.1f} q=quit",
    )

    try:
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            components.left_hand_backend.start()
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            components.right_hand_backend.start()
        components.session.start()
        components.session.set_calibration_ready()
        components.session.mark_debug_passed()
        components.session.arm_hardware()
        key_input.start()

        step_index = 0
        while True:
            loop_start = time.time()
            key_input.poll()
            if key_input.should_quit:
                return

            step_result = components.session.run_step(now=loop_start)

            if status_gate.should_emit(
                "hand_only_summary",
                (
                    step_result.state.value,
                    tuple(
                        f"{fault.code}:{fault.channel_name}" if fault.channel_name else fault.code
                        for fault in step_result.faults
                    ),
                ),
                now=loop_start,
                heartbeat_sec=args.status_sec,
            ):
                log_event(
                    "HandOnly",
                    f"step={step_index} hz={step_result.control_hz:.1f} "
                    f"state={step_result.state.value} "
                    f"faults={[f'{fault.code}:{fault.channel_name}' if fault.channel_name else fault.code for fault in step_result.faults]}",
                    now=loop_start,
                )

            for side in ("left", "right"):
                _log_side_status(
                    side,
                    input_q6=_input_q6(step_result.channels, side),
                    command_q6=_command_q6(step_result.command, side),
                    backend=_side_backend(components, side),
                    gate=status_gate,
                    heartbeat_sec=args.status_sec,
                    now=loop_start,
                )

            if components.renderer is not None:
                components.renderer.render_data(
                    left_qpos_arm=DEFAULT_LEFT_ARM_QPOS,
                    left_qpos_hand=_hand_render_qpos(components.left_hand_backend, "left"),
                    right_qpos_arm=DEFAULT_RIGHT_ARM_QPOS,
                    right_qpos_hand=_hand_render_qpos(components.right_hand_backend, "right"),
                    capture_image=False,
                )

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                return
            remaining = (1.0 / args.hz) - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        key_input.close()
        components.session.close()
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            components.left_hand_backend.close()
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            components.right_hand_backend.close()


if __name__ == "__main__":
    main()
