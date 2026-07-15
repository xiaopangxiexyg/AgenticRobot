from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    __package__ = "utils.head_eye"

from .calibration import (
    CalibrationSample,
    load_samples_with_metadata,
    save_result,
    save_samples,
    solve_eye_to_hand,
)
from .camera import (
    DEFAULT_CAMERA_SERIALS,
    RealSenseCamera,
    RealSenseConfig,
    TargetConfig,
    TargetDetector,
    available_video_profiles,
)
from .capture import CaptureBurstConfig, capture_burst
from .robot import RM65PoseReader
from .transforms import transform_distance


LEFT_ARM_IP = "192.168.101.21"
RIGHT_ARM_IP = "192.168.1.20"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive RM65 fixed-camera hand-eye calibration."
    )
    parser.add_argument("--list-cameras", action="store_true", help="List RealSense devices and exit.")
    parser.add_argument(
        "--show-profiles",
        action="store_true",
        help="When listing cameras, also show supported video profiles.",
    )
    parser.add_argument("--camera", "--camera-index", dest="camera_index", type=int, default=None)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--auto-select-supported-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If the requested RealSense profile cannot start, select a compatible "
            "native color/depth profile and record it in new calibration metadata."
        ),
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--arm-ip", default="", help="Override the IP selected by --arm.")
    parser.add_argument("--expected-tool", default="", help="Optional active RM65 tool-frame name check.")
    parser.add_argument("--robot-retry", type=int, default=2)
    parser.add_argument("--output-dir", default="teleop_records/handeye")
    parser.add_argument("--name", default=None, help="Output stem. Default uses current timestamp.")
    parser.add_argument("--load-samples", default=None, help="Solve from a saved samples JSON.")
    parser.add_argument("--method", default="AUTO", help="AUTO, TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS.")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--save-legacy-npy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autosave-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-running", action="store_true", help="Do not exit after a successful solve.")

    parser.add_argument("--target", choices=("aruco", "charuco"), default="aruco")
    parser.add_argument("--dictionary", default="DICT_6X6_100")
    parser.add_argument("--marker-length-m", type=float, default=0.10)
    parser.add_argument("--marker-id", type=int, default=None)
    parser.add_argument("--charuco-squares-x", type=int, default=5)
    parser.add_argument("--charuco-squares-y", type=int, default=7)
    parser.add_argument("--charuco-square-length-m", type=float, default=0.035)
    parser.add_argument("--charuco-marker-length-m", type=float, default=0.026)
    parser.add_argument("--min-charuco-corners", type=int, default=12)

    parser.add_argument("--capture-seconds", type=float, default=0.6)
    parser.add_argument("--sample-interval-sec", type=float, default=0.03)
    parser.add_argument("--min-detections-per-sample", type=int, default=3)
    parser.add_argument("--max-robot-motion-m", type=float, default=0.0015)
    parser.add_argument("--max-robot-motion-deg", type=float, default=0.25)
    parser.add_argument("--max-target-motion-m", type=float, default=0.003)
    parser.add_argument("--max-target-motion-deg", type=float, default=0.8)
    parser.add_argument("--accept-unstable", action="store_true")
    parser.add_argument(
        "--auto-record",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically record when the robot has stopped at a new pose and the target is visible.",
    )
    parser.add_argument("--auto-stable-sec", type=float, default=3.0, help="Still time required before auto-recording.")
    parser.add_argument("--auto-still-translation-m", type=float, default=0.0015, help="Robot translation threshold for still detection.")
    parser.add_argument("--auto-still-rotation-deg", type=float, default=0.25, help="Robot rotation threshold for still detection.")
    parser.add_argument("--auto-new-pose-translation-m", type=float, default=0.03, help="Minimum translation change from the last recorded pose.")
    parser.add_argument("--auto-new-pose-rotation-deg", type=float, default=8.0, help="Minimum rotation change from the last recorded pose.")
    parser.add_argument("--auto-cooldown-sec", type=float, default=1.0, help="Cooldown after an auto-record attempt.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    session_name = args.name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_rm65_handeye"
    output_dir = Path(args.output_dir)
    arm_ip = _selected_arm_ip(args)

    if args.list_cameras:
        _print_cameras(show_profiles=args.show_profiles)
        return 0

    if args.load_samples:
        samples, args._loaded_samples_metadata = load_samples_with_metadata(args.load_samples)
        return _solve_and_save(samples, args, output_dir=output_dir, session_name=session_name, arm_ip=arm_ip)

    _print_cameras(show_profiles=False)
    target_config = TargetConfig(
        target_type=args.target,
        dictionary=args.dictionary,
        marker_length_m=args.marker_length_m,
        marker_id=args.marker_id,
        charuco_squares_x=args.charuco_squares_x,
        charuco_squares_y=args.charuco_squares_y,
        charuco_square_length_m=args.charuco_square_length_m,
        charuco_marker_length_m=args.charuco_marker_length_m,
        min_charuco_corners=args.min_charuco_corners,
    )
    camera_config = RealSenseConfig(
        serial=args.camera_serial,
        camera_index=args.camera_index,
        resolution=tuple(args.resolution),
        fps=args.fps,
        enable_depth=True,
        auto_select_supported_profile=bool(args.auto_select_supported_profile),
    )
    burst_config = CaptureBurstConfig(
        duration_sec=args.capture_seconds,
        sample_interval_sec=args.sample_interval_sec,
        min_detections=args.min_detections_per_sample,
        max_robot_motion_m=args.max_robot_motion_m,
        max_robot_motion_deg=args.max_robot_motion_deg,
        max_target_motion_m=args.max_target_motion_m,
        max_target_motion_deg=args.max_target_motion_deg,
        accept_unstable=args.accept_unstable,
    )
    samples: list[CalibrationSample] = []
    raw_path = output_dir / f"{session_name}_samples.json"

    detector = TargetDetector(target_config)
    cv2 = detector._cv2
    window_name = "RM65 hand-eye calibration"
    print("\nControls: r=record stable burst, c=compute/save, d=delete last, s=save raw, q/esc=quit")
    print("Move the end-effector target through varied positions and wrist rotations.")
    if args.auto_record:
        print(
            "Auto-record: enabled; record after "
            f"{args.auto_stable_sec:.1f}s still at a new pose with target visible."
        )
    print(f"Output directory: {output_dir.resolve()}")

    with RealSenseCamera(camera_config) as camera, RM65PoseReader(
        arm_ip,
        retry=args.robot_retry,
        expected_tool=args.expected_tool,
    ) as robot:
        tool_text = f"; tool={robot.current_tool}" if robot.current_tool else ""
        print(f"Connected RM65 arm={args.arm} ip={arm_ip}{tool_text}; camera serial={camera.serial or 'auto'}")
        args._camera_settings_report = dict(camera.settings_report)
        _print_camera_settings(camera.settings_report)
        auto_state = AutoRecordState(args) if args.auto_record else None
        try:
            while True:
                frame = camera.read()
                detection, preview = detector.detect(
                    frame.color_bgr,
                    frame.intrinsics_matrix,
                    frame.distortion_coeffs,
                    draw=True,
                )
                auto_status = ""
                auto_should_record = False
                if auto_state is not None:
                    try:
                        robot_sample = robot.read()
                        auto_should_record, auto_status = auto_state.update(
                            robot_sample.T_gripper_to_base,
                            target_visible=detection is not None,
                        )
                    except Exception as exc:
                        auto_status = f"auto robot read rejected: {exc}"
                _draw_session_overlay(cv2, preview, len(samples), args.min_samples, auto_status)
                cv2.imshow(window_name, preview)
                if auto_should_record:
                    sample = _record_sample(
                        camera=camera,
                        detector=detector,
                        robot=robot,
                        burst_config=burst_config,
                        samples=samples,
                        args=args,
                        raw_path=raw_path,
                        camera_serial=camera.serial,
                        arm_ip=arm_ip,
                        reason="auto-record",
                    )
                    if sample is not None and auto_state is not None:
                        auto_state.mark_recorded(sample.T_gripper_to_base)
                    elif auto_state is not None:
                        auto_state.mark_attempted()
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    sample = _record_sample(
                        camera=camera,
                        detector=detector,
                        robot=robot,
                        burst_config=burst_config,
                        samples=samples,
                        args=args,
                        raw_path=raw_path,
                        camera_serial=camera.serial,
                        arm_ip=arm_ip,
                        reason="record",
                    )
                    if sample is not None and auto_state is not None:
                        auto_state.mark_recorded(sample.T_gripper_to_base)
                elif key == ord("d"):
                    if samples:
                        samples.pop()
                        print(f"[record] deleted last sample; remaining={len(samples)}")
                        if auto_state is not None:
                            auto_state.reset_last_record(samples)
                        if args.autosave_samples:
                            save_samples(raw_path, samples, metadata=_session_metadata(args, camera.serial, arm_ip=arm_ip))
                    else:
                        print("[record] no samples to delete")
                elif key == ord("s"):
                    save_samples(raw_path, samples, metadata=_session_metadata(args, camera.serial, arm_ip=arm_ip))
                    print(f"[save] raw samples -> {raw_path.resolve()}")
                elif key == ord("c"):
                    if len(samples) < int(args.min_samples):
                        print(f"[solve] need at least {args.min_samples} samples, have {len(samples)}")
                        continue
                    code = _solve_and_save(
                        samples,
                        args,
                        output_dir=output_dir,
                        session_name=session_name,
                        camera_serial=camera.serial,
                        raw_path=raw_path,
                        arm_ip=arm_ip,
                    )
                    if code == 0 and not args.keep_running:
                        break
        finally:
            cv2.destroyAllWindows()
    return 0


def _solve_and_save(
    samples: Sequence[CalibrationSample],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    session_name: str,
    camera_serial: str | None = None,
    raw_path: Path | None = None,
    arm_ip: str | None = None,
) -> int:
    metadata = _session_metadata(args, camera_serial, arm_ip=arm_ip)
    result = solve_eye_to_hand(samples, method=args.method, metadata=metadata)
    if raw_path is not None:
        save_samples(raw_path, samples, metadata=metadata)
    paths = save_result(
        result,
        output_dir,
        name=session_name,
        save_legacy_npy=bool(args.save_legacy_npy),
    )
    _print_result(result, paths)
    return 0


def _print_cameras(*, show_profiles: bool = False) -> None:
    try:
        devices = RealSenseCamera.list_devices()
    except Exception as exc:
        print(f"RealSense devices unavailable: {exc}")
        print(f"Configured legacy serials: {', '.join(DEFAULT_CAMERA_SERIALS)}")
        return
    if not devices:
        print("No RealSense devices detected.")
        print(f"Configured legacy serials: {', '.join(DEFAULT_CAMERA_SERIALS)}")
        return
    print("RealSense devices:")
    for device in devices:
        fw = f", fw={device.firmware_version}" if device.firmware_version else ""
        print(f"  [{device.index}] {device.name} serial={device.serial}{fw}")
        if show_profiles:
            try:
                profiles = available_video_profiles(device.serial)
            except Exception as exc:
                print(f"    profiles unavailable: {exc}")
                continue
            for profile in profiles:
                print(f"    - {profile.describe()}")


def _print_camera_settings(settings: dict[str, object]) -> None:
    if not settings:
        print("RealSense stream settings: no report")
        return
    print("RealSense stream settings:")
    for key in sorted(settings):
        print(f"  {key}: {settings[key]}")


class AutoRecordState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.reference_pose: np.ndarray | None = None
        self.stable_since = 0.0
        self.last_record_pose: np.ndarray | None = None
        self.cooldown_until = 0.0

    def update(self, robot_pose: np.ndarray, *, target_visible: bool) -> tuple[bool, str]:
        now = time.monotonic()
        if self.reference_pose is None:
            self.reference_pose = robot_pose.copy()
            self.stable_since = now
            return False, "auto: waiting for still pose"

        moved_m, moved_deg = transform_distance(self.reference_pose, robot_pose)
        if (
            moved_m > self.args.auto_still_translation_m
            or moved_deg > self.args.auto_still_rotation_deg
        ):
            self.reference_pose = robot_pose.copy()
            self.stable_since = now
            return False, f"auto: moving {moved_m * 1000.0:.1f}mm/{moved_deg:.2f}deg"

        stable_sec = max(0.0, now - self.stable_since)
        is_new, new_m, new_deg = self._is_new_pose(robot_pose)
        if now < self.cooldown_until:
            return False, "auto: cooldown"
        if not target_visible:
            return False, f"auto: still {stable_sec:.1f}s, target not visible"
        if not is_new:
            return False, (
                f"auto: still {stable_sec:.1f}s, pose already recorded "
                f"({new_m * 1000.0:.1f}mm/{new_deg:.1f}deg)"
            )
        if stable_sec < self.args.auto_stable_sec:
            return False, f"auto: still {stable_sec:.1f}/{self.args.auto_stable_sec:.1f}s"
        return True, f"auto: recording new pose ({new_m * 1000.0:.1f}mm/{new_deg:.1f}deg)"

    def mark_recorded(self, robot_pose: np.ndarray) -> None:
        now = time.monotonic()
        self.last_record_pose = robot_pose.copy()
        self.reference_pose = robot_pose.copy()
        self.stable_since = now
        self.cooldown_until = now + self.args.auto_cooldown_sec

    def mark_attempted(self) -> None:
        self.cooldown_until = time.monotonic() + self.args.auto_cooldown_sec

    def reset_last_record(self, samples: Sequence[CalibrationSample]) -> None:
        self.last_record_pose = samples[-1].T_gripper_to_base.copy() if samples else None

    def _is_new_pose(self, robot_pose: np.ndarray) -> tuple[bool, float, float]:
        if self.last_record_pose is None:
            return True, math.inf, math.inf
        distance_m, distance_deg = transform_distance(self.last_record_pose, robot_pose)
        return (
            distance_m >= self.args.auto_new_pose_translation_m
            or distance_deg >= self.args.auto_new_pose_rotation_deg
        ), distance_m, distance_deg


def _record_sample(
    *,
    camera: RealSenseCamera,
    detector: TargetDetector,
    robot: RM65PoseReader,
    burst_config: CaptureBurstConfig,
    samples: list[CalibrationSample],
    args: argparse.Namespace,
    raw_path: Path,
    camera_serial: str | None,
    arm_ip: str,
    reason: str,
) -> CalibrationSample | None:
    try:
        sample, stats = capture_burst(
            camera=camera,
            detector=detector,
            robot=robot,
            config=burst_config,
            sample_index=len(samples) + 1,
        )
    except Exception as exc:
        print(f"[{reason}] rejected: {exc}")
        return None
    samples.append(sample)
    print(
        f"[{reason}] "
        f"#{len(samples)} detections={stats['detections']}/{stats['frames']} "
        f"robot={stats['robot_motion_max_m'] * 1000.0:.2f}mm/"
        f"{stats['robot_motion_max_deg']:.2f}deg "
        f"target={stats['target_motion_max_m'] * 1000.0:.2f}mm/"
        f"{stats['target_motion_max_deg']:.2f}deg "
        f"reproj={stats['reprojection_rmse_px']:.2f}px"
    )
    if args.autosave_samples:
        save_samples(raw_path, samples, metadata=_session_metadata(args, camera_serial, arm_ip=arm_ip))
    return sample


def _print_result(result, paths: dict[str, Path]) -> None:
    metrics = result.metrics
    print("\nCalibration complete")
    print("T_camera_to_base (meters):")
    print(np.array2string(result.T_camera_to_base, precision=8, suppress_small=False))
    print(
        "Residual: "
        f"{metrics['target_in_gripper_translation_rms_m'] * 1000.0:.2f} mm RMS, "
        f"{metrics['target_in_gripper_rotation_rms_deg']:.3f} deg RMS"
    )
    print(
        "Coverage: "
        f"robot span {metrics['robot_translation_span_m']:.3f} m, "
        f"{metrics['robot_rotation_span_deg']:.1f} deg; samples={metrics['sample_count']}"
    )
    warnings = metrics.get("quality_warnings") or []
    if warnings:
        print("Quality warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Quality checks: OK")
    worst = sorted(
        metrics.get("per_sample_errors") or [],
        key=lambda item: float(item.get("target_in_gripper_translation_m", 0.0)),
        reverse=True,
    )[:3]
    if worst:
        print("Worst samples:")
        for item in worst:
            print(
                "  "
                f"#{item['sample_index']}: "
                f"{float(item['target_in_gripper_translation_m']) * 1000.0:.2f} mm, "
                f"{float(item['target_in_gripper_rotation_deg']):.3f} deg"
            )
    print("Saved:")
    for label, path in paths.items():
        print(f"  {label}: {path.resolve()}")


def _draw_session_overlay(
    cv2,
    image: np.ndarray,
    sample_count: int,
    min_samples: int,
    status: str = "",
) -> None:
    text = f"samples {sample_count}/{min_samples}   r record   c compute   d delete   q quit"
    y0 = max(54, image.shape[0] - 34)
    box_height = 48 if status else 32
    cv2.rectangle(image, (8, y0 - 24), (min(image.shape[1] - 8, 920), y0 - 24 + box_height), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (18, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            image,
            status[:96],
            (18, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (170, 230, 170),
            1,
            cv2.LINE_AA,
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.auto_stable_sec <= 0.0:
        raise ValueError("--auto-stable-sec must be positive")
    if args.auto_still_translation_m <= 0.0:
        raise ValueError("--auto-still-translation-m must be positive")
    if args.auto_still_rotation_deg <= 0.0:
        raise ValueError("--auto-still-rotation-deg must be positive")
    if args.auto_new_pose_translation_m <= 0.0:
        raise ValueError("--auto-new-pose-translation-m must be positive")
    if args.auto_new_pose_rotation_deg <= 0.0:
        raise ValueError("--auto-new-pose-rotation-deg must be positive")
    if args.auto_cooldown_sec < 0.0:
        raise ValueError("--auto-cooldown-sec must be >= 0")


def _selected_arm_ip(args: argparse.Namespace) -> str:
    if args.arm_ip:
        return str(args.arm_ip)
    return LEFT_ARM_IP if args.arm == "left" else RIGHT_ARM_IP


def _session_metadata(
    args: argparse.Namespace,
    camera_serial: str | None,
    *,
    arm_ip: str | None = None,
) -> dict[str, object]:
    loaded_samples_metadata = dict(getattr(args, "_loaded_samples_metadata", {}) or {})
    settings_report = dict(getattr(args, "_camera_settings_report", {}) or {})
    if not settings_report:
        settings_report = dict(loaded_samples_metadata.get("realsense_settings_report") or {})
    metadata: dict[str, object] = {
        "robot": "RM65",
        "arm": args.arm,
        "arm_ip": arm_ip or _selected_arm_ip(args),
        "expected_tool": args.expected_tool,
        "camera_serial": (
            camera_serial
            or args.camera_serial
            or loaded_samples_metadata.get("camera_serial")
        ),
        "camera_index": (
            args.camera_index
            if args.camera_index is not None
            else loaded_samples_metadata.get("camera_index")
        ),
        "target": args.target,
        "dictionary": args.dictionary,
        "marker_length_m": args.marker_length_m,
        "marker_id": args.marker_id,
        "charuco_squares_x": args.charuco_squares_x,
        "charuco_squares_y": args.charuco_squares_y,
        "charuco_square_length_m": args.charuco_square_length_m,
        "charuco_marker_length_m": args.charuco_marker_length_m,
        "auto_select_supported_profile": loaded_samples_metadata.get(
            "auto_select_supported_profile",
            args.auto_select_supported_profile,
        ),
        "realsense_settings_report": settings_report,
        "units": "meters/radians",
    }
    if loaded_samples_metadata:
        metadata["source_samples_metadata"] = {
            key: value
            for key, value in loaded_samples_metadata.items()
            if key != "source_samples_metadata"
        }
    return metadata


if __name__ == "__main__":
    raise SystemExit(main())
