"""Run reviewed SAM2 + selected point-grasp backend on one local aligned RGB-D pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from agent_gateway.profiles.real.point_grasp_deployment import build_real_point_grasp_registration
from agent_gateway.profiles.real.point_grasp_gateway import PointGraspProvider
from agent_gateway.skills.perception.point_grasp_schemas import validate_point_grasp_parameters_v1

from .localize_image_cli import _atomic_write
from .sam2_grounding_dino import Sam2GroundingDinoLocalizer, Sam2GroundingDinoModelSpec
from .sam2_graspnet_point_grasp import (
    GraspNetPointGraspModelSpec,
    Sam2GraspNetPointGraspPlanner,
)
from .sam2_vgn_point_grasp import Sam2VgnPointGraspPlanner, VgnPointGraspModelSpec

OUTPUT_SCHEMA_V1 = "dexteleop.perception.point_grasp_image_cli.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deployments.perception.point_grasp_image_cli",
        description="Plan ranked DMTAC M tool_tcp poses from one aligned local RGB-D frame and clicked pixel.",
    )
    parser.add_argument("--sam2-asset-bundle", type=Path, required=True)
    parser.add_argument("--point-grasp-asset-bundle", type=Path, required=True)
    parser.add_argument(
        "--point-grasp-backend",
        choices=("vgn", "graspnet"),
        default="vgn",
        help="grasp generator backend; GraspNet is noncommercial research only",
    )
    parser.add_argument("--input-rgb", type=Path, required=True)
    parser.add_argument("--input-depth", type=Path, required=True, help="C-order float32 HxW NPY in meters")
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--target-u", type=int, required=True)
    parser.add_argument("--target-v", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _existing(path: Path, *, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        raise ValueError(f"Expected {'directory' if directory else 'file'}: {resolved}")
    return resolved


def _output(path: Path, *, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.parent.is_dir():
        raise ValueError(f"Output parent does not exist: {resolved.parent}")
    if resolved.exists() and not overwrite:
        raise ValueError(f"Output exists; use --overwrite: {resolved}")
    return resolved


def run(args: argparse.Namespace) -> dict:
    sam_bundle = _existing(args.sam2_asset_bundle, directory=True)
    grasp_bundle = _existing(args.point_grasp_asset_bundle, directory=True)
    rgb = _existing(args.input_rgb)
    depth = _existing(args.input_depth)
    calibration_path = _existing(args.calibration_json)
    output_json = _output(args.output_json, overwrite=args.overwrite)
    output_image = _output(args.output_image, overwrite=args.overwrite)
    if output_image.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("--output-image must end in .jpg or .jpeg")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict) or set(calibration) != {
        "camera_frame_id", "base_frame_id", "camera_intrinsics", "T_base_from_camera"
    }:
        raise ValueError("Calibration JSON fields are invalid.")
    parameters = {
        "input_source": "local_path",
        "local_rgb_path": str(rgb),
        "local_depth_path": str(depth),
        "target_pixel_uv": [args.target_u, args.target_v],
        "camera_intrinsics": calibration["camera_intrinsics"],
        "camera_frame_id": calibration["camera_frame_id"],
        "base_frame_id": calibration["base_frame_id"],
        "T_base_from_camera": calibration["T_base_from_camera"],
        "max_candidates": args.max_candidates,
    }
    validate_point_grasp_parameters_v1(parameters, local_path_enabled=True, max_candidates=5)
    sam_spec = Sam2GroundingDinoModelSpec.from_asset_bundle(sam_bundle)
    localizer = Sam2GroundingDinoLocalizer(model_spec=sam_spec, device=args.device)
    if args.point_grasp_backend == "graspnet":
        grasp_spec = GraspNetPointGraspModelSpec.from_asset_bundle(grasp_bundle)
        planner = Sam2GraspNetPointGraspPlanner(
            model_spec=grasp_spec, point_segmenter=localizer, device=args.device
        )
    else:
        grasp_spec = VgnPointGraspModelSpec.from_asset_bundle(grasp_bundle)
        planner = Sam2VgnPointGraspPlanner(
            model_spec=grasp_spec, point_segmenter=localizer, device=args.device
        )
    registration = build_real_point_grasp_registration(
        backend_id=planner.backend_id,
        model_profile_id=grasp_spec.model_profile_id,
        model_profile_hash=grasp_spec.semantic_content_hash,
        inference_implementation_hash=planner.semantic_content_hash,
        local_path_enabled=True,
        runtime_session_id="local-point-grasp-cli",
        control_domain_id="local-read-only",
        target_binding_id="local-files",
        station_manifest_hash="not-applicable",
        execution_manifest_hash="local-point-grasp-cli",
    )
    roots = tuple(sorted({rgb.parent, depth.parent}, key=str))
    provider = PointGraspProvider(planner=planner, registration=registration, allowed_local_image_roots=roots)
    localizer.start()
    try:
        provider.start()
        try:
            prepared = provider.plan_request(
                SimpleNamespace(parameters=parameters, input_artifact_bindings=()),
                artifact_store=SimpleNamespace(),
            )
        finally:
            provider.close()
    finally:
        localizer.close()
    payload = {"schema_version": OUTPUT_SCHEMA_V1, **prepared.execution_result}
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(output_image, prepared.annotated_jpeg)
    _atomic_write(output_json, encoded)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        payload = run(parser.parse_args(argv))
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"point-grasp inference failed: {exc}\n")
    print(json.dumps({"candidate_count": payload["candidate_count"], "selected_ordinal": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
