from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from teleop_runtime.sources.pose.vive import VivePoseSource, ViveTrackerConfig
from utils.hand_eye.camera import RealSenseCamera, RealSenseConfig, TargetConfig, TargetDetector
from utils.hand_eye.capture import CaptureBurstConfig
from utils.hand_eye.transforms import transform_distance

from .calibration import load_result, load_samples, save_result, save_samples, solve_umi_hand_eye
from .auto_record import AutoRecordConfig, AutoRecordState
from .capture import ViveTrackerPoseReader, capture_umi_burst
from .config import load_config
from .pose_coverage import PoseCoverageConfig, evaluate_pose_coverage
from .ui import (
    compose_dashboard,
    compose_previews,
    current_window_render_size,
    open_resizable_window,
)
from .visualization import compute_live_check_pose, draw_live_check_overlay


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "rm65_dual_wrist_umi_v1.json"


@dataclass(frozen=True)
class _OutputPaths:
    samples: Path
    result: Path
    contract: Path
    latest: Path


@dataclass
class _LiveSideSession:
    side: str
    side_config: Any
    camera: RealSenseCamera
    tracker: ViveTrackerPoseReader
    detector: TargetDetector
    auto_state: AutoRecordState | None
    paths: _OutputPaths
    pose_change_translation_m: float
    pose_change_rotation_deg: float
    pose_change_require_both: bool
    samples: list[Any] = field(default_factory=list)
    capture_future: Future | None = None
    capture_automatic: bool = False
    last_preview: Any | None = None
    last_target_status: str = "target: waiting for first frame"


class _UmiTargetDetector(TargetDetector):
    def __init__(self, config: TargetConfig, *, min_aruco_board_markers: int):
        super().__init__(config)
        self._min_aruco_board_markers = max(1, int(min_aruco_board_markers))

    def detect(self, *args, draw: bool = False, **kwargs):
        # UMI renders all text in a separate high-resolution dashboard panel.
        # Keep the RGB image clean; only draw the target coordinate axes here.
        detection, preview = super().detect(*args, draw=False, **kwargs)
        if detection is not None and "ok aruco-board" in self.last_status:
            marker_count = len(detection.marker_ids)
            if marker_count < self._min_aruco_board_markers:
                self._last_status = (
                    f"target: reject aruco-board markers={marker_count}/"
                    f"{self._min_aruco_board_markers}; show more of the board"
                )
                detection = None
        if draw and detection is not None:
            intrinsics = args[1] if len(args) > 1 else kwargs["intrinsics_matrix"]
            distortion = args[2] if len(args) > 2 else kwargs["distortion_coeffs"]
            self._draw_axes(preview, intrinsics, distortion, detection)
        return detection, preview


def _output_paths(output_dir: Path, name: str, side: str, *, dual: bool) -> _OutputPaths:
    stem = f"{name}_{side}" if dual else name
    return _OutputPaths(
        samples=output_dir / f"{stem}_samples.json",
        result=output_dir / f"{stem}.json",
        contract=output_dir / f"{stem}_pose_frame_contract.json",
        latest=output_dir / f"latest_{side}.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a rigid RealSense+Vive-tracker UMI into a board-defined desk frame."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    parser.add_argument("--output-dir", default="teleop_records/umi_hand_eye")
    parser.add_argument("--name", default=None)
    parser.add_argument("--load-samples", default=None, help="Solve offline without opening devices.")
    parser.add_argument("--check-result", default=None, help="Validate expiry/device binding and exit.")
    parser.add_argument(
        "--view-result",
        action="store_true",
        help=(
            "Open a read-only live camera/tracker/desk visualization from saved results. "
            "Defaults to output-dir/latest_left.json and/or latest_right.json."
        ),
    )
    parser.add_argument(
        "--result",
        default=None,
        help="Saved result used by --view-result in a single-side mode.",
    )
    parser.add_argument("--left-result", default=None, help="Left result override for --view-result.")
    parser.add_argument("--right-result", default=None, help="Right result override for --view-result.")
    parser.add_argument("--method", default="AUTO")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum diverse samples required independently for each selected side.",
    )
    parser.add_argument("--validity-seconds", type=float, default=None)
    parser.add_argument(
        "--tcp-profile",
        default=None,
        help="Named UMI TCP profile; defaults to the selected side config (dmgripper).",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Override the URDF used to resolve endpoint-to-camera geometry.",
    )
    parser.add_argument("--capture-seconds", type=float, default=2.0)
    parser.add_argument("--min-detections", type=int, default=20)
    parser.add_argument(
        "--auto-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically record a stable burst after the tracker has stopped at a new pose "
            "and the ChArUco target has a valid pose (default: enabled)."
        ),
    )
    parser.add_argument(
        "--auto-stable-sec",
        type=float,
        default=0.5,
        help="Required continuous still time before an automatic burst.",
    )
    parser.add_argument("--auto-still-translation-m", type=float, default=0.0015)
    parser.add_argument("--auto-still-rotation-deg", type=float, default=0.25)
    parser.add_argument(
        "--auto-target-still-translation-m",
        type=float,
        default=0.003,
        help="Maximum target translation during the automatic pre-burst dwell.",
    )
    parser.add_argument(
        "--auto-target-still-rotation-deg",
        type=float,
        default=0.8,
        help="Maximum target rotation during the automatic pre-burst dwell.",
    )
    parser.add_argument(
        "--auto-new-pose-translation-m",
        type=float,
        default=0.04,
        help="Minimum translation from every recorded pose for a new sample.",
    )
    parser.add_argument(
        "--auto-new-pose-rotation-deg",
        type=float,
        default=10.0,
        help="Minimum rotation from every recorded pose for a new sample.",
    )
    parser.add_argument(
        "--pose-change-require-both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every new sample to differ from every recorded pose in both "
            "translation and rotation (default: enabled)."
        ),
    )
    parser.add_argument("--auto-cooldown-sec", type=float, default=1.0)
    parser.add_argument(
        "--coverage-translation-m",
        type=float,
        default=0.10,
        help="Required total tracker translation span before solving.",
    )
    parser.add_argument(
        "--coverage-rotation-deg",
        type=float,
        default=30.0,
        help="Required total tracker rotation span before solving.",
    )
    parser.add_argument(
        "--coverage-axis-separation-deg",
        type=float,
        default=25.0,
        help="Required angular separation between two observed rotation axes.",
    )
    parser.add_argument(
        "--coverage-axis-observation-rotation-deg",
        type=float,
        default=8.0,
        help="Minimum relative rotation used as an observation of a rotation axis.",
    )
    parser.add_argument("--max-tracker-age-sec", type=float, default=0.25)
    parser.add_argument(
        "--camera-startup-warmup-frames",
        type=int,
        default=5,
        help="Frames to drain while verifying the selected RealSense at startup.",
    )
    parser.add_argument(
        "--camera-reset-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On startup frame/USB failure, hardware-reset only the selected camera serial, "
            "wait for reconnect, and retry once (default: enabled)."
        ),
    )
    parser.add_argument("--target-type", choices=("aruco", "charuco"), default=None)
    parser.add_argument("--dictionary", default=None)
    parser.add_argument("--marker-length-m", type=float, default=None)
    parser.add_argument("--marker-id", type=int, default=None)
    parser.add_argument("--charuco-squares-x", type=int, default=None)
    parser.add_argument("--charuco-squares-y", type=int, default=None)
    parser.add_argument("--charuco-square-length-m", type=float, default=None)
    parser.add_argument("--charuco-marker-length-m", type=float, default=None)
    parser.add_argument("--min-charuco-corners", type=int, default=None)
    parser.add_argument(
        "--min-aruco-board-markers",
        type=int,
        default=4,
        help="Minimum matched board markers accepted by the ChArUco fallback.",
    )
    parser.add_argument(
        "--allow-aruco-board-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When ChArUco interpolation is weak, estimate the same fixed board frame from "
            "its ArUco markers (default: enabled in the UMI config)."
        ),
    )
    parser.add_argument(
        "--visualize-after-solve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a live solve, keep the camera and tracker open to visualize the desk "
            "frame and live tracker pose in desk coordinates (default: enabled)."
        ),
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Initial and render-limit width of the resizable preview window.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Initial and render-limit height of the resizable preview window.",
    )
    parser.add_argument("--auto-reject-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    selected_sides = ("left", "right") if args.side == "both" else (args.side,)
    args.umi_urdf_path = args.urdf or config.urdf_path
    validity_seconds = (
        config.validity_seconds if args.validity_seconds is None else float(args.validity_seconds)
    )

    if args.view_result:
        return _view_saved_results(
            args=args,
            config=config,
            selected_sides=selected_sides,
        )

    if args.check_result:
        if args.side == "both":
            raise ValueError("--check-result requires --side left or --side right")
        side_config = config.side(args.side)
        result = load_result(args.check_result)
        result.require_valid(
            camera_serial=side_config.camera_serial,
            tracker_name=side_config.tracker_name,
        )
        result.require_tcp_profile(args.tcp_profile or side_config.tcp_profile)
        print(f"valid until {result.expires_at_utc}")
        return 0

    output_dir = Path(args.output_dir)
    name = args.name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.side}_umi"

    if args.load_samples:
        if args.side == "both":
            raise ValueError("--load-samples requires --side left or --side right")
        side_config = config.side(args.side)
        paths = _output_paths(output_dir, name, args.side, dual=False)
        samples, sample_metadata = load_samples(args.load_samples)
        result = _solve(
            samples,
            args=args,
            side=args.side,
            side_config=side_config,
            validity_seconds=validity_seconds,
            metadata={"sample_file": str(args.load_samples), **sample_metadata},
        )
        _save_outputs(
            result,
            result_path=paths.result,
            contract_path=paths.contract,
            latest_result_path=paths.latest,
        )
        return 0

    paths_by_side = {
        side: _output_paths(output_dir, name, side, dual=args.side == "both")
        for side in selected_sides
    }
    return _run_live(
        args=args,
        config=config,
        selected_sides=selected_sides,
        validity_seconds=validity_seconds,
        paths_by_side=paths_by_side,
    )


def _run_live(
    *,
    args,
    config,
    selected_sides,
    validity_seconds,
    paths_by_side,
) -> int:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python with the aruco module is required") from exc

    coverage_config = _pose_coverage_config(args)
    target_values = dict(config.target)
    target_overrides = {
        "target_type": args.target_type,
        "dictionary": args.dictionary,
        "marker_length_m": args.marker_length_m,
        "marker_id": args.marker_id,
        "charuco_squares_x": args.charuco_squares_x,
        "charuco_squares_y": args.charuco_squares_y,
        "charuco_square_length_m": args.charuco_square_length_m,
        "charuco_marker_length_m": args.charuco_marker_length_m,
        "min_charuco_corners": args.min_charuco_corners,
        "allow_aruco_board_fallback": args.allow_aruco_board_fallback,
    }
    for key, value in target_overrides.items():
        if value is not None:
            target_values[key] = value
    target_config = TargetConfig(**target_values)
    burst_config = CaptureBurstConfig(
        duration_sec=float(args.capture_seconds),
        min_detections=int(args.min_detections),
    )

    tracker_configs = [
        ViveTrackerConfig(
            name=config.side(side).tracker_name,
            side=side,
            role=config.side(side).tracker_role,
            channel_name=config.side(side).channel_name,
            legacy_index=index,
        )
        for index, side in enumerate(("left", "right"))
    ]
    source = VivePoseSource(
        tracker_configs,
        host=config.vive_host,
        port=config.vive_port,
        clock_sync_port=config.vive_clock_sync_port,
        name="umi_vive_pose",
    )
    sessions: list[_LiveSideSession] = []
    for side in selected_sides:
        side_config = config.side(side)
        auto_state = (
            AutoRecordState(
                AutoRecordConfig(
                    stable_sec=float(args.auto_stable_sec),
                    still_translation_m=float(args.auto_still_translation_m),
                    still_rotation_deg=float(args.auto_still_rotation_deg),
                    target_still_translation_m=float(
                        args.auto_target_still_translation_m
                    ),
                    target_still_rotation_deg=float(
                        args.auto_target_still_rotation_deg
                    ),
                    new_pose_translation_m=float(args.auto_new_pose_translation_m),
                    new_pose_rotation_deg=float(args.auto_new_pose_rotation_deg),
                    require_both_pose_changes=bool(args.pose_change_require_both),
                    cooldown_sec=float(args.auto_cooldown_sec),
                )
            )
            if bool(args.auto_record)
            else None
        )
        sessions.append(
            _LiveSideSession(
                side=side,
                side_config=side_config,
                camera=RealSenseCamera(
                    RealSenseConfig(
                        serial=side_config.camera_serial,
                        resolution=config.camera_resolution,
                        fps=config.camera_fps,
                        color_format="yuyv",
                        enable_depth=False,
                        startup_warmup_frames=int(args.camera_startup_warmup_frames),
                        startup_wait_timeout_ms=1000,
                        startup_hardware_reset_on_failure=bool(args.camera_reset_on_failure),
                        startup_hardware_reset_retry_attempts=1,
                    )
                ),
                tracker=ViveTrackerPoseReader(
                    source=source,
                    channel_name=side_config.channel_name,
                    tracker_name=side_config.tracker_name,
                    max_age_sec=float(args.max_tracker_age_sec),
                ),
                detector=_UmiTargetDetector(
                    target_config,
                    min_aruco_board_markers=int(args.min_aruco_board_markers),
                ),
                auto_state=auto_state,
                paths=paths_by_side[side],
                pose_change_translation_m=float(args.auto_new_pose_translation_m),
                pose_change_rotation_deg=float(args.auto_new_pose_rotation_deg),
                pose_change_require_both=bool(args.pose_change_require_both),
            )
        )

    window = (
        "UMI dual hand-eye: r=both 1=left 2=right c=solve q=quit"
        if len(sessions) == 2
        else "UMI hand-eye: auto r=manual c=solve q=quit"
    )
    try:
        source.start()
        with ExitStack() as stack:
            for session in sessions:
                stack.enter_context(session.camera)
                _print_camera_profile(session.camera.settings_report, side=session.side)
                recovery_count = int(
                    session.camera.settings_report.get(
                        "startup.hardware_reset_recovery_count", 0
                    )
                )
                if recovery_count:
                    print(
                        f"[{session.side}] RealSense startup recovered after hardware reset: "
                        f"serial={session.side_config.camera_serial}, resets={recovery_count}"
                    )
                print(
                    f"Calibrating {session.side}: camera={session.side_config.camera_serial}, "
                    f"tracker={session.side_config.tracker_name}."
                )
            executor = stack.enter_context(
                ThreadPoolExecutor(
                    max_workers=len(sessions),
                    thread_name_prefix="umi-burst",
                )
            )
            _print_target_config(target_config, bool(args.auto_record))
            if len(sessions) == 2:
                print(
                    "Dual auto-record is independent: either side records immediately when "
                    "that side alone is stable, target-valid, and pose-diverse; it never "
                    "waits for the other side."
                )
            print(
                "Pose gate: every accepted pose must differ from all recorded poses by "
                f"{float(args.auto_new_pose_translation_m) * 1000.0:.0f} mm AND "
                f"{float(args.auto_new_pose_rotation_deg):.0f} deg."
            )
            print(
                "Solve coverage: "
                f"samples>={int(args.min_samples)}, translation span>="
                f"{float(args.coverage_translation_m) * 1000.0:.0f} mm, rotation span>="
                f"{float(args.coverage_rotation_deg):.0f} deg, non-parallel rotation axes."
            )
            open_resizable_window(
                cv2,
                window,
                width=int(args.window_width),
                height=int(args.window_height),
            )
            while True:
                for session in sessions:
                    _finish_record_session(session)
                previews: dict[str, Any] = {}
                status_by_side: dict[str, list[str]] = {}
                ready_by_side: dict[str, bool] = {}
                auto_triggered: list[_LiveSideSession] = []
                for session in sessions:
                    coverage = _coverage_for_session(session, coverage_config)
                    if session.capture_future is not None:
                        if session.last_preview is None:
                            raise RuntimeError(
                                f"{session.side} burst started without a preview frame"
                            )
                        preview = session.last_preview.copy()
                        detection = None
                        auto_status = "RECORDING BURST... keep this side still"
                    else:
                        frame = session.camera.read()
                        detection, preview = session.detector.detect(
                            frame.color_bgr,
                            frame.intrinsics_matrix,
                            frame.distortion_coeffs,
                            draw=True,
                        )
                        session.last_preview = preview.copy()
                        session.last_target_status = session.detector.last_status
                        auto_status = (
                            "auto: READY; press c to solve" if coverage.ready else ""
                        )
                    auto_should_record = False
                    if (
                        session.capture_future is None
                        and session.auto_state is not None
                        and not coverage.ready
                    ):
                        try:
                            tracker_sample = session.tracker.read()
                            auto_should_record, auto_status = session.auto_state.update(
                                tracker_sample.T_gripper_to_base,
                                target_visible=detection is not None,
                                target_pose=(
                                    None
                                    if detection is None
                                    else detection.T_target_to_camera
                                ),
                            )
                        except Exception as exc:
                            auto_status = f"auto: tracker unavailable ({exc})"
                    lines = [
                        f"camera: {session.side_config.camera_serial}",
                        session.last_target_status,
                        auto_status,
                        *coverage.overlay_lines(coverage_config),
                    ]
                    previews[session.side] = preview
                    status_by_side[session.side] = [line for line in lines if line]
                    ready_by_side[session.side] = coverage.ready
                    if auto_should_record:
                        auto_triggered.append(session)

                render_width, render_height = current_window_render_size(
                    cv2,
                    window,
                    fallback_width=int(args.window_width),
                    fallback_height=int(args.window_height),
                )
                visual = compose_dashboard(
                    cv2,
                    previews,
                    status_by_side,
                    ready_by_side,
                    width=render_width,
                    height=render_height,
                )
                cv2.imshow(window, visual)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    return 0
                for session in auto_triggered:
                    _start_record_session(
                        session,
                        executor=executor,
                        burst_config=burst_config,
                        automatic=True,
                    )
                if key == ord("r"):
                    for session in sessions:
                        _start_record_session(
                            session,
                            executor=executor,
                            burst_config=burst_config,
                            automatic=False,
                        )
                if key == ord("1") and len(sessions) == 2:
                    _start_record_session(
                        sessions[0],
                        executor=executor,
                        burst_config=burst_config,
                        automatic=False,
                    )
                if key == ord("2") and len(sessions) == 2:
                    _start_record_session(
                        sessions[1],
                        executor=executor,
                        burst_config=burst_config,
                        automatic=False,
                    )
                if key == ord("c"):
                    recording = [
                        session.side
                        for session in sessions
                        if session.capture_future is not None
                    ]
                    if recording:
                        print(
                            "solve blocked while burst recording is active: "
                            + ", ".join(recording)
                        )
                        continue
                    blocked = [
                        (session, _coverage_for_session(session, coverage_config))
                        for session in sessions
                        if not _coverage_for_session(session, coverage_config).ready
                    ]
                    if blocked:
                        for session, coverage in blocked:
                            print(
                                f"[{session.side}] solve blocked: {coverage.hint}; "
                                f"samples={coverage.sample_count}, "
                                f"translation_span={coverage.translation_span_m * 1000.0:.1f}mm, "
                                f"rotation_span={coverage.rotation_span_deg:.1f}deg, "
                                f"axis_separation={coverage.rotation_axis_separation_deg:.1f}deg"
                            )
                        continue
                    results = {}
                    for session in sessions:
                        result = _solve(
                            session.samples,
                            args=args,
                            side=session.side,
                            side_config=session.side_config,
                            validity_seconds=validity_seconds,
                            metadata={"samples_path": str(session.paths.samples)},
                        )
                        _save_outputs(
                            result,
                            result_path=session.paths.result,
                            contract_path=session.paths.contract,
                            latest_result_path=session.paths.latest,
                        )
                        results[session.side] = result
                    if bool(args.visualize_after_solve):
                        _run_live_check_multi(
                            cv2=cv2,
                            sessions=sessions,
                            results=results,
                            axis_length_m=float(config.target.get("axis_length_m", 0.10)),
                            window_width=int(args.window_width),
                            window_height=int(args.window_height),
                        )
                    return 0
    finally:
        source.close()
        cv2.destroyAllWindows()


def _view_saved_results(*, args, config, selected_sides: Sequence[str]) -> int:
    if args.load_samples or args.check_result:
        raise ValueError("--view-result cannot be combined with --load-samples or --check-result")
    if args.result and len(selected_sides) != 1:
        raise ValueError("--result requires --side left or --side right")
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python with the aruco module is required") from exc

    output_dir = Path(args.output_dir)
    explicit_by_side = {
        "left": args.left_result,
        "right": args.right_result,
    }
    results: dict[str, Any] = {}
    for side in selected_sides:
        result_path = Path(
            args.result
            or explicit_by_side[side]
            or output_dir / f"latest_{side}.json"
        )
        result = load_result(result_path)
        _require_result_binding(result, side=side, side_config=config.side(side))
        results[side] = result
        print(f"[{side}] loaded visualization result: {result_path}")

    target_config = TargetConfig(**dict(config.target))
    tracker_configs = [
        ViveTrackerConfig(
            name=config.side(side).tracker_name,
            side=side,
            role=config.side(side).tracker_role,
            channel_name=config.side(side).channel_name,
            legacy_index=index,
        )
        for index, side in enumerate(("left", "right"))
    ]
    source = VivePoseSource(
        tracker_configs,
        host=config.vive_host,
        port=config.vive_port,
        clock_sync_port=config.vive_clock_sync_port,
        name="umi_vive_pose_view",
    )
    sessions: list[_LiveSideSession] = []
    for side in selected_sides:
        side_config = config.side(side)
        sessions.append(
            _LiveSideSession(
                side=side,
                side_config=side_config,
                camera=RealSenseCamera(
                    RealSenseConfig(
                        serial=side_config.camera_serial,
                        resolution=config.camera_resolution,
                        fps=config.camera_fps,
                        color_format="yuyv",
                        enable_depth=False,
                        startup_warmup_frames=int(args.camera_startup_warmup_frames),
                        startup_wait_timeout_ms=1000,
                        startup_hardware_reset_on_failure=bool(args.camera_reset_on_failure),
                        startup_hardware_reset_retry_attempts=1,
                    )
                ),
                tracker=ViveTrackerPoseReader(
                    source=source,
                    channel_name=side_config.channel_name,
                    tracker_name=side_config.tracker_name,
                    max_age_sec=float(args.max_tracker_age_sec),
                ),
                detector=_UmiTargetDetector(
                    target_config,
                    min_aruco_board_markers=int(args.min_aruco_board_markers),
                ),
                auto_state=None,
                paths=_output_paths(output_dir, "view", side, dual=len(selected_sides) == 2),
                pose_change_translation_m=0.0,
                pose_change_rotation_deg=0.0,
                pose_change_require_both=True,
            )
        )

    try:
        source.start()
        with ExitStack() as stack:
            for session in sessions:
                stack.enter_context(session.camera)
                _print_camera_profile(session.camera.settings_report, side=session.side)
            _run_live_check_multi(
                cv2=cv2,
                sessions=sessions,
                results=results,
                axis_length_m=float(config.target.get("axis_length_m", 0.10)),
                window_width=int(args.window_width),
                window_height=int(args.window_height),
            )
        return 0
    finally:
        source.close()
        cv2.destroyAllWindows()


def _require_result_binding(result: Any, *, side: str, side_config: Any) -> None:
    if result.side != side:
        raise RuntimeError(f"UMI result side mismatch: result={result.side!r}, expected={side!r}")
    if result.camera_serial != side_config.camera_serial:
        raise RuntimeError(
            "UMI result camera serial does not match the configured side: "
            f"result={result.camera_serial!r}, expected={side_config.camera_serial!r}"
        )
    if result.tracker_name != side_config.tracker_name:
        raise RuntimeError(
            "UMI result tracker does not match the configured side: "
            f"result={result.tracker_name!r}, expected={side_config.tracker_name!r}"
        )


def _coverage_for_session(
    session: _LiveSideSession,
    config: PoseCoverageConfig,
):
    return evaluate_pose_coverage(
        [sample.T_tracker_to_vive_world for sample in session.samples],
        config,
    )


def _pose_coverage_config(args) -> PoseCoverageConfig:
    return PoseCoverageConfig(
        min_samples=int(args.min_samples),
        min_translation_span_m=float(args.coverage_translation_m),
        min_rotation_span_deg=float(args.coverage_rotation_deg),
        min_rotation_axis_separation_deg=float(args.coverage_axis_separation_deg),
        min_axis_observation_rotation_deg=float(
            args.coverage_axis_observation_rotation_deg
        ),
    )


def _start_record_session(
    session: _LiveSideSession,
    *,
    executor: ThreadPoolExecutor,
    burst_config: CaptureBurstConfig,
    automatic: bool,
) -> bool:
    if session.capture_future is not None:
        print(f"[{session.side}] record request ignored: burst already running")
        return False
    session.capture_automatic = bool(automatic)
    session.capture_future = executor.submit(
        capture_umi_burst,
        camera=session.camera,
        detector=session.detector,
        tracker=session.tracker,
        config=burst_config,
        sample_index=len(session.samples),
    )
    label = "auto-record" if automatic else "record"
    print(f"[{session.side}] {label} burst started asynchronously")
    return True


def _finish_record_session(session: _LiveSideSession) -> bool:
    future = session.capture_future
    if future is None or not future.done():
        return False
    automatic = bool(session.capture_automatic)
    session.capture_future = None
    session.capture_automatic = False
    label = "auto-record" if automatic else "record"
    try:
        sample, stats = future.result()
    except Exception as exc:
        print(f"[{session.side}] {label} failed: {exc}")
        if automatic and session.auto_state is not None:
            session.auto_state.mark_attempted()
        return False
    novelty = _sample_pose_novelty(session, sample.T_tracker_to_vive_world)
    if novelty is not None:
        distance_m, distance_deg = novelty
        print(
            f"[{session.side}] {label} rejected: pose is too similar to an existing "
            f"sample ({distance_m * 1000.0:.1f}mm/{distance_deg:.1f}deg); "
            f"need {'both' if session.pose_change_require_both else 'either'} "
            f"{session.pose_change_translation_m * 1000.0:.0f}mm and "
            f"{session.pose_change_rotation_deg:.0f}deg."
        )
        if automatic and session.auto_state is not None:
            session.auto_state.mark_attempted()
        return False
    session.samples.append(sample)
    save_samples(
        session.paths.samples,
        session.samples,
        metadata=_binding_metadata(session.side, session.side_config),
    )
    if session.auto_state is not None:
        session.auto_state.mark_recorded(sample.T_tracker_to_vive_world)
    print(f"[{session.side}] {label}ed sample {len(session.samples)}: {stats}")
    return True


def _sample_pose_novelty(
    session: _LiveSideSession,
    candidate_pose: Any,
) -> tuple[float, float] | None:
    if not session.samples:
        return None
    distances = [
        transform_distance(sample.T_tracker_to_vive_world, candidate_pose)
        for sample in session.samples
    ]
    translation_limit = float(session.pose_change_translation_m)
    rotation_limit = float(session.pose_change_rotation_deg)
    invalid = [
        (distance_m, distance_deg)
        for distance_m, distance_deg in distances
        if (
            distance_m < translation_limit or distance_deg < rotation_limit
            if session.pose_change_require_both
            else distance_m < translation_limit and distance_deg < rotation_limit
        )
    ]
    if not invalid:
        return None
    return min(
        invalid,
        key=lambda value: (
            float(value[0]) / translation_limit
            + float(value[1]) / rotation_limit
        ),
    )


def _print_target_config(target: TargetConfig, auto_enabled: bool) -> None:
    if target.target_type == "charuco":
        squares_x = int(target.charuco_squares_x)
        squares_y = int(target.charuco_squares_y)
        square_length_m = float(target.charuco_square_length_m)
        marker_length_m = float(target.charuco_marker_length_m)
        max_corners = max(0, (squares_x - 1) * (squares_y - 1))
        print(
            "Target: ChArUco "
            f"dictionary={target.dictionary}, squares={squares_x}x{squares_y}, "
            f"square={square_length_m * 1000.0:.1f} mm, "
            f"marker={marker_length_m * 1000.0:.1f} mm"
        )
        print(
            "Target board body: "
            f"{squares_x * square_length_m * 1000.0:.1f} x "
            f"{squares_y * square_length_m * 1000.0:.1f} mm; "
            f"inner corners={max_corners}, required={int(target.min_charuco_corners)}"
        )
        print(
            "ArUco board fallback: "
            f"{'enabled' if bool(target.allow_aruco_board_fallback) else 'disabled'}"
        )
    else:
        print(
            "Target: ArUco "
            f"dictionary={target.dictionary}, "
            f"marker={float(target.marker_length_m) * 1000.0:.1f} mm, "
            f"marker_id={target.marker_id if target.marker_id is not None else 'single detected marker'}"
        )
    if auto_enabled:
        print("Auto-record: enabled; keep the rigid pair still until a new sample is recorded.")
    else:
        print("Auto-record: disabled; press r to record a sample manually.")


def _print_camera_profile(report: dict[str, object], *, side: str | None = None) -> None:
    requested = report.get("requested", "unknown")
    actual = report.get("fallback_color_profile", report.get("color.format", "unknown"))
    fallback = report.get("fallback_used", False)
    print(
        f"{f'[{side}] ' if side else ''}RealSense profile: "
        f"requested={requested}; actual={actual}; fallback={bool(fallback)}"
    )


def _solve(samples, *, args, side, side_config, validity_seconds, metadata):
    coverage_config = _pose_coverage_config(args)
    coverage = evaluate_pose_coverage(
        [sample.T_tracker_to_vive_world for sample in samples],
        coverage_config,
    )
    if not coverage.ready:
        raise RuntimeError(
            f"{side} solve blocked by pose coverage: {coverage.hint}; "
            f"samples={coverage.sample_count}, "
            f"translation_span={coverage.translation_span_m * 1000.0:.1f}mm, "
            f"rotation_span={coverage.rotation_span_deg:.1f}deg, "
            f"axis_separation={coverage.rotation_axis_separation_deg:.1f}deg"
        )
    metadata = {
        **dict(metadata or {}),
        "pose_coverage": {
            "translation_span_m": coverage.translation_span_m,
            "rotation_span_deg": coverage.rotation_span_deg,
            "rotation_axis_separation_deg": coverage.rotation_axis_separation_deg,
            "requirements": {
                "min_samples": coverage_config.min_samples,
                "min_translation_span_m": coverage_config.min_translation_span_m,
                "min_rotation_span_deg": coverage_config.min_rotation_span_deg,
                "min_rotation_axis_separation_deg": (
                    coverage_config.min_rotation_axis_separation_deg
                ),
            },
        },
    }
    result = solve_umi_hand_eye(
        samples,
        side=side,
        camera_serial=side_config.camera_serial,
        tracker_name=side_config.tracker_name,
        tracker_role=side_config.tracker_role,
        tcp_profile=args.tcp_profile or side_config.tcp_profile,
        urdf_path=args.umi_urdf_path,
        validity_seconds=validity_seconds,
        method=args.method,
        refine=bool(args.refine),
        auto_reject_outliers=bool(args.auto_reject_outliers),
        metadata=metadata,
    )
    _apply_umi_sample_guidance(result.metrics, min_samples=int(args.min_samples))
    print(
        "solve complete: "
        f"method={result.method}, "
        f"translation_rms={result.metrics.get('target_in_base_translation_rms_m')}, "
        f"rotation_rms_deg={result.metrics.get('target_in_base_rotation_rms_deg')}, "
        f"expires={result.expires_at_utc}"
    )
    for warning in result.metrics.get("quality_warnings", []):
        print(f"QUALITY WARNING: {warning}")
    return result


def _apply_umi_sample_guidance(metrics: dict[str, Any], *, min_samples: int) -> None:
    plan = metrics.get("sample_plan")
    if isinstance(plan, dict):
        plan["recommended_sample_count"] = (
            f">={int(min_samples)} poses that pass UMI diversity gates"
        )
        plan["warnings"] = [
            str(warning)
            for warning in plan.get("warnings", [])
            if "at least 25 samples" not in str(warning)
        ]
    metrics["quality_warnings"] = [
        str(warning)
        for warning in metrics.get("quality_warnings", [])
        if not str(warning).startswith("sample_count below 25")
        and "at least 25 samples" not in str(warning)
    ]


def _save_outputs(
    result,
    *,
    result_path: Path,
    contract_path: Path,
    latest_result_path: Path,
) -> None:
    save_result(result_path, result)
    save_result(latest_result_path, result)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(result.pose_frame_contract(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved result: {result_path}")
    print(f"updated side binding: {latest_result_path}")
    print(f"saved v2.3 pose contract: {contract_path}")


def _binding_metadata(side: str, side_config) -> dict[str, str]:
    return {
        "side": side,
        "camera_serial": side_config.camera_serial,
        "tracker_name": side_config.tracker_name,
        "tracker_role": side_config.tracker_role,
        "desk_definition": "calibration_board_frame",
    }


def _run_live_check_multi(
    *,
    cv2,
    sessions: Sequence[_LiveSideSession],
    results: dict[str, Any],
    axis_length_m: float,
    window_width: int,
    window_height: int,
) -> None:
    window = "UMI calibration check: desk + trackers (q=quit)"
    print(
        "live calibration check started: desk axes and each tracker pose are shown "
        "in their side panel. Press q/Esc to finish."
    )
    open_resizable_window(
        cv2,
        window,
        width=int(window_width),
        height=int(window_height),
    )
    while True:
        previews: dict[str, Any] = {}
        for session in sessions:
            result = results[session.side]
            frame = session.camera.read()
            detection, preview = session.detector.detect(
                frame.color_bgr,
                frame.intrinsics_matrix,
                frame.distortion_coeffs,
                draw=False,
            )
            try:
                tracker_sample = session.tracker.read()
                pose = compute_live_check_pose(
                    result,
                    tracker_sample.T_gripper_to_base,
                )
            except Exception as exc:
                _draw_live_check_error(cv2, preview, f"tracker unavailable: {exc}")
                visual = preview
            else:
                board_residual = None
                if detection is not None:
                    board_residual = transform_distance(
                        detection.T_target_to_camera,
                        pose.T_desk_to_camera,
                    )
                visual = draw_live_check_overlay(
                    cv2,
                    preview,
                    intrinsics_matrix=frame.intrinsics_matrix,
                    distortion_coeffs=frame.distortion_coeffs,
                    pose=pose,
                    tracker_name=result.tracker_name,
                    axis_length_m=max(0.01, float(axis_length_m)),
                    board_residual=board_residual,
                )
            previews[session.side] = visual
        render_width, render_height = current_window_render_size(
            cv2,
            window,
            fallback_width=int(window_width),
            fallback_height=int(window_height),
        )
        cv2.imshow(
            window,
            compose_previews(
                cv2,
                previews,
                max_width=render_width,
                max_height=render_height,
            ),
        )
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
    try:
        cv2.destroyWindow(window)
    except Exception:
        pass


def _draw_live_check_error(cv2, image, message: str) -> None:
    cv2.rectangle(image, (8, 8), (image.shape[1] - 8, 52), (0, 0, 0), -1)
    cv2.putText(
        image,
        message[:120],
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 230),
        2,
        cv2.LINE_AA,
    )
