from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from teleop_runtime.sources.pose.vive import VivePoseSource, ViveTrackerConfig
from utils.hand_eye.camera import RealSenseCamera, RealSenseConfig, TargetConfig, TargetDetector
from utils.hand_eye.capture import CaptureBurstConfig

from .calibration import load_result, load_samples, save_result, save_samples, solve_umi_hand_eye
from .capture import ViveTrackerPoseReader, capture_umi_burst
from .config import load_config


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "rm65_dual_wrist_umi_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a rigid RealSense+Vive-tracker UMI into a board-defined desk frame."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--output-dir", default="teleop_records/umi_hand_eye")
    parser.add_argument("--name", default=None)
    parser.add_argument("--load-samples", default=None, help="Solve offline without opening devices.")
    parser.add_argument("--check-result", default=None, help="Validate expiry/device binding and exit.")
    parser.add_argument("--method", default="AUTO")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--validity-seconds", type=float, default=None)
    parser.add_argument("--capture-seconds", type=float, default=2.0)
    parser.add_argument("--min-detections", type=int, default=20)
    parser.add_argument("--max-tracker-age-sec", type=float, default=0.25)
    parser.add_argument("--auto-reject-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    side_config = config.side(args.side)
    validity_seconds = (
        config.validity_seconds if args.validity_seconds is None else float(args.validity_seconds)
    )

    if args.check_result:
        result = load_result(args.check_result)
        result.require_valid(
            camera_serial=side_config.camera_serial,
            tracker_name=side_config.tracker_name,
        )
        print(f"valid until {result.expires_at_utc}")
        return 0

    output_dir = Path(args.output_dir)
    name = args.name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.side}_umi"
    samples_path = output_dir / f"{name}_samples.json"
    result_path = output_dir / f"{name}.json"
    contract_path = output_dir / f"{name}_pose_frame_contract.json"

    if args.load_samples:
        samples, sample_metadata = load_samples(args.load_samples)
        result = _solve(
            samples,
            args=args,
            side_config=side_config,
            validity_seconds=validity_seconds,
            metadata={"sample_file": str(args.load_samples), **sample_metadata},
        )
        _save_outputs(result, result_path=result_path, contract_path=contract_path)
        return 0

    return _run_live(
        args=args,
        config=config,
        side_config=side_config,
        validity_seconds=validity_seconds,
        samples_path=samples_path,
        result_path=result_path,
        contract_path=contract_path,
    )


def _run_live(
    *, args, config, side_config, validity_seconds, samples_path, result_path, contract_path
) -> int:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python with the aruco module is required") from exc

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
    tracker = ViveTrackerPoseReader(
        source=source,
        channel_name=side_config.channel_name,
        tracker_name=side_config.tracker_name,
        max_age_sec=float(args.max_tracker_age_sec),
    )
    camera = RealSenseCamera(
        RealSenseConfig(
            serial=side_config.camera_serial,
            resolution=config.camera_resolution,
            fps=config.camera_fps,
            enable_depth=False,
        )
    )
    detector = TargetDetector(TargetConfig(**dict(config.target)))
    burst_config = CaptureBurstConfig(
        duration_sec=float(args.capture_seconds),
        min_detections=int(args.min_detections),
    )
    samples = []
    window = "UMI hand-eye: r=record c=solve q=quit"
    source.start()
    try:
        with camera:
            print(
                f"Calibrating {args.side}: camera={side_config.camera_serial}, "
                f"tracker={side_config.tracker_name}. Move the rigid pair between samples."
            )
            while True:
                frame = camera.read()
                _, preview = detector.detect(
                    frame.color_bgr,
                    frame.intrinsics_matrix,
                    frame.distortion_coeffs,
                    draw=True,
                )
                cv2.putText(
                    preview,
                    f"samples={len(samples)}/{args.min_samples}",
                    (12, preview.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("r"):
                    try:
                        sample, stats = capture_umi_burst(
                            camera=camera,
                            detector=detector,
                            tracker=tracker,
                            config=burst_config,
                            sample_index=len(samples),
                        )
                    except Exception as exc:
                        print(f"record failed: {exc}")
                    else:
                        samples.append(sample)
                        save_samples(samples_path, samples, metadata=_binding_metadata(args.side, side_config))
                        print(f"recorded sample {len(samples)}: {stats}")
                if key == ord("c"):
                    if len(samples) < int(args.min_samples):
                        print(f"need {args.min_samples} samples, currently {len(samples)}")
                        continue
                    result = _solve(
                        samples,
                        args=args,
                        side_config=side_config,
                        validity_seconds=validity_seconds,
                        metadata={"samples_path": str(samples_path)},
                    )
                    _save_outputs(result, result_path=result_path, contract_path=contract_path)
                    return 0
    finally:
        source.close()
        cv2.destroyAllWindows()


def _solve(samples, *, args, side_config, validity_seconds, metadata):
    result = solve_umi_hand_eye(
        samples,
        side=args.side,
        camera_serial=side_config.camera_serial,
        tracker_name=side_config.tracker_name,
        tracker_role=side_config.tracker_role,
        validity_seconds=validity_seconds,
        method=args.method,
        refine=bool(args.refine),
        auto_reject_outliers=bool(args.auto_reject_outliers),
        metadata=metadata,
    )
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


def _save_outputs(result, *, result_path: Path, contract_path: Path) -> None:
    save_result(result_path, result)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(result.pose_frame_contract(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved result: {result_path}")
    print(f"saved v2.3 pose contract: {contract_path}")


def _binding_metadata(side: str, side_config) -> dict[str, str]:
    return {
        "side": side,
        "camera_serial": side_config.camera_serial,
        "tracker_name": side_config.tracker_name,
        "tracker_role": side_config.tracker_role,
        "desk_definition": "calibration_board_frame",
    }
