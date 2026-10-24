"""Run reviewed SAM2 + selected point-grasp backend on one local aligned RGB-D pair."""

from __future__ import annotations

from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS,
)

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from deployments.perception.dmtac_grasp_overlay import (
    DmtacGraspOverlayCandidate,
    render_dmtac_grasp_overlays,
    reviewed_graspgenx_dmtac_solid_geometry,
)

from .graspgenx_client import GraspGenXInProcessBackend
from .localize_image_cli import MAX_INPUT_PIXELS, _atomic_write, _load_image
from .sam2_graspgenx_point_grasp import (
    GraspGenXPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
)
from .sam2_graspnet_point_grasp import (
    GraspNetPointGraspModelSpec,
    Sam2GraspNetPointGraspPlanner,
)
from .sam2_grounding_dino import Sam2GroundingDinoLocalizer, Sam2GroundingDinoModelSpec
from .sam2_vgn_point_grasp import Sam2VgnPointGraspPlanner, VgnPointGraspModelSpec

OUTPUT_SCHEMA_V1 = "dexteleop.perception.point_grasp_image_cli.v1"
MAX_DEPTH_BYTES = 64 * 1024 * 1024
MAX_CALIBRATION_BYTES = 64 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deployments.perception.point_grasp_image_cli",
        description="Plan ranked DMTAC M tool_tcp poses from one aligned local RGB-D frame and clicked pixel.",
    )
    parser.add_argument("--graspgenx-alignment-profile",
        choices=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS, default=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
        help="Explicit nominal alignment; v5 is restricted to the right-arm q0 calibration.")
    parser.add_argument("--sam2-asset-bundle", type=Path, required=True)
    parser.add_argument("--point-grasp-asset-bundle", type=Path, required=True)
    parser.add_argument(
        "--point-grasp-backend",
        choices=("graspgenx", "vgn", "graspnet"),
        default="graspgenx",
        help=(
            "grasp generator backend; GraspGenX runs in-process and is the "
            "default"
        ),
    )
    parser.add_argument(
        "--graspgenx-source-root",
        type=Path,
        default=None,
        help=(
            "optional reviewed GraspGenX source checkout for the in-process "
            "backend; defaults to model/perception/graspgenx"
        ),
    )
    parser.add_argument("--input-rgb", type=Path, required=True)
    parser.add_argument(
        "--input-depth",
        type=Path,
        required=True,
        help="C-order float32 HxW NPY in meters",
    )
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


def _validated_calibration(
    payload: object,
) -> tuple[dict[str, float | int], str, str, np.ndarray]:
    if not isinstance(payload, dict) or set(payload) != {
        "camera_frame_id",
        "base_frame_id",
        "camera_intrinsics",
        "T_base_from_camera",
    }:
        raise ValueError("Calibration JSON fields are invalid.")
    camera_frame_id = payload["camera_frame_id"]
    base_frame_id = payload["base_frame_id"]
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in (camera_frame_id, base_frame_id)
    ):
        raise ValueError("Calibration frame IDs must be non-empty trimmed strings.")
    intrinsics = payload["camera_intrinsics"]
    if not isinstance(intrinsics, dict) or set(intrinsics) != {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
    }:
        raise ValueError("Calibration camera_intrinsics fields are invalid.")
    width, height = intrinsics["width"], intrinsics["height"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, height)
    ):
        raise ValueError("Calibration image dimensions must be positive integers.")
    for field in ("fx", "fy", "cx", "cy"):
        value = intrinsics[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Calibration intrinsics must be finite numeric values.")
    if float(intrinsics["fx"]) <= 0 or float(intrinsics["fy"]) <= 0:
        raise ValueError("Calibration focal lengths must be positive.")
    transform = np.asarray(payload["T_base_from_camera"], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_base_from_camera must be a finite 4x4 transform.")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError("T_base_from_camera has an invalid homogeneous row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("T_base_from_camera rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("T_base_from_camera rotation is not right-handed.")
    return intrinsics, camera_frame_id, base_frame_id, transform


def _annotated_jpeg(
    image: Image.Image,
    pixel_uv: tuple[int, int],
    planned: object,
    intrinsics: dict[str, float | int],
    solid_geometry: object | None = None,
) -> bytes:
    # JSON retains all ranked proposals; the human preview shows only ordinal 0.
    candidates = tuple(
        DmtacGraspOverlayCandidate.from_pose(
            ordinal=ordinal,
            score=candidate.score,
            gripper_width_m=candidate.gripper_width_m,
            reference_from_tcp=candidate.camera_from_tcp,
            selected=ordinal == 0,
        )
        for ordinal, candidate in enumerate(planned.candidates[:1])
    )
    output, _projections = render_dmtac_grasp_overlays(
        image,
        candidates=candidates,
        reference_from_camera=np.eye(4),
        intrinsics=intrinsics,
        target_pixel_uv=pixel_uv,
        target_bbox_xyxy=tuple(planned.target_mask_bbox_xyxy),
        solid_geometry=solid_geometry,
    )
    stream = io.BytesIO()
    output.save(stream, format="JPEG", quality=90, optimize=True)
    return stream.getvalue()


def _build_point_grasp_planner(
    *,
    args: argparse.Namespace,
    grasp_bundle: Path,
    localizer: Sam2GroundingDinoLocalizer,
) -> tuple[object, object]:
    source_root_arg = getattr(args, "graspgenx_source_root", None)
    if args.point_grasp_backend != "graspgenx" and source_root_arg is not None:
        raise ValueError(
            "--graspgenx-source-root requires --point-grasp-backend graspgenx"
        )
    if args.point_grasp_backend == "graspgenx":
        source_root = (
            None
            if source_root_arg is None
            else _existing(source_root_arg, directory=True)
        )
        grasp_spec = GraspGenXPointGraspModelSpec.from_asset_bundle(grasp_bundle)
        backend = GraspGenXInProcessBackend(
            model_spec=grasp_spec,
            source_root=source_root,
            device=args.device,
        )
        return grasp_spec, Sam2GraspGenXPointGraspPlanner(
            model_spec=grasp_spec,
            point_segmenter=localizer,
            client=backend,
            alignment_profile_id=getattr(args, "graspgenx_alignment_profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID),
        )
    if args.point_grasp_backend == "graspnet":
        grasp_spec = GraspNetPointGraspModelSpec.from_asset_bundle(grasp_bundle)
        return grasp_spec, Sam2GraspNetPointGraspPlanner(
            model_spec=grasp_spec,
            point_segmenter=localizer,
            device=args.device,
        )
    grasp_spec = VgnPointGraspModelSpec.from_asset_bundle(grasp_bundle)
    return grasp_spec, Sam2VgnPointGraspPlanner(
        model_spec=grasp_spec,
        point_segmenter=localizer,
        device=args.device,
    )


def run(args: argparse.Namespace) -> dict:
    sam_bundle = _existing(args.sam2_asset_bundle, directory=True)
    grasp_bundle = _existing(args.point_grasp_asset_bundle, directory=True)
    rgb = _existing(args.input_rgb)
    depth = _existing(args.input_depth)
    calibration_path = _existing(args.calibration_json)
    output_json = _output(args.output_json, overwrite=args.overwrite)
    output_image = _output(args.output_image, overwrite=args.overwrite)
    if len({rgb, depth, calibration_path, output_json, output_image}) != 5:
        raise ValueError("RGB, depth, calibration, and output paths must be distinct.")
    if output_image.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("--output-image must end in .jpg or .jpeg")
    calibration_size = calibration_path.stat().st_size
    if calibration_size <= 0 or calibration_size > MAX_CALIBRATION_BYTES:
        raise ValueError("Calibration JSON must be non-empty and at most 64 KiB.")
    intrinsics, camera_frame_id, base_frame_id, base_from_camera = (
        _validated_calibration(json.loads(calibration_path.read_text(encoding="utf-8")))
    )
    pixel_uv = (args.target_u, args.target_v)
    if not 1 <= args.max_candidates <= 5:
        raise ValueError("--max-candidates must be in [1, 5].")
    if not (
        0 <= pixel_uv[0] < intrinsics["width"]
        and 0 <= pixel_uv[1] < intrinsics["height"]
    ):
        raise ValueError("Target pixel is outside the calibrated image.")
    image, _ = _load_image(rgb)
    if image.size != (intrinsics["width"], intrinsics["height"]):
        raise ValueError("RGB dimensions differ from camera_intrinsics.")
    depth_size = depth.stat().st_size
    if depth_size <= 0 or depth_size > MAX_DEPTH_BYTES:
        raise ValueError("Depth NPY must be non-empty and at most 64 MiB.")
    depth_values = np.load(depth, allow_pickle=False)
    if (
        depth_values.dtype != np.float32
        or depth_values.shape != (intrinsics["height"], intrinsics["width"])
        or depth_values.size > MAX_INPUT_PIXELS
        or not depth_values.flags.c_contiguous
        or not np.all(np.isfinite(depth_values))
        or np.any(depth_values < 0)
        or np.any(depth_values > 10.0)
    ):
        raise ValueError("Depth must be finite C-order float32 HxW aligned meters.")
    sam_spec = Sam2GroundingDinoModelSpec.from_asset_bundle(sam_bundle)
    localizer = Sam2GroundingDinoLocalizer(model_spec=sam_spec, device=args.device)
    grasp_spec, planner = _build_point_grasp_planner(
        args=args,
        grasp_bundle=grasp_bundle,
        localizer=localizer,
    )
    localizer.start()
    try:
        planner.start()
        try:
            planned = planner.plan(
                image,
                depth_values,
                pixel_uv=pixel_uv,
                intrinsics=intrinsics,
                base_from_camera=base_from_camera,
                max_candidates=args.max_candidates,
            )
        finally:
            planner.close()
    finally:
        localizer.close()
    candidates = []
    for ordinal, candidate in enumerate(planned.candidates):
        camera = list(candidate.camera_from_tcp)
        base = list(candidate.base_from_tcp)
        candidates.append(
            {
                "ordinal": ordinal,
                "score": candidate.score,
                "model_score": candidate.model_score,
                "click_distance_m": candidate.click_distance_m,
                "gripper_width_m": candidate.gripper_width_m,
                "T_camera_from_tcp": camera,
                "T_base_from_tcp": base,
                "tcp_xyz_camera_m": [camera[3], camera[7], camera[11]],
                "tcp_xyz_base_m": [base[3], base[7], base[11]],
                "visible_collision_free": candidate.visible_collision_free,
            }
        )
    annotated = _annotated_jpeg(
        image,
        pixel_uv,
        planned,
        intrinsics,
        solid_geometry=getattr(planner, "solid_geometry", None),
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA_V1,
        "input_source": "local_path",
        "target_pixel_uv": list(pixel_uv),
        "target_point_camera_m": list(planned.target_point_camera_m),
        "target_point_base_m": list(planned.target_point_base_m),
        "target_mask_bbox_xyxy": list(planned.target_mask_bbox_xyxy),
        "camera_frame_id": camera_frame_id,
        "base_frame_id": base_frame_id,
        "pose_semantics": "homogeneous_row_major_parent_from_tool_tcp_m",
        "gripper_profile_id": "DMTAC_M",
        "backend_id": planner.backend_id,
        "candidate_count": len(candidates),
        "selected_ordinal": 0,
        "candidates": candidates,
        "model_profile_id": grasp_spec.model_profile_id,
        "model_profile_hash": grasp_spec.semantic_content_hash,
        "calibration_source": "explicit_local_file_not_gateway_trusted",
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(output_image, annotated)
    _atomic_write(output_json, encoded)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.graspgenx_alignment_profile != DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID:
            if args.point_grasp_backend != "graspgenx":
                parser.error("--graspgenx-alignment-profile requires the GraspGenX backend")
        payload = run(args)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"point-grasp inference failed: {exc}\n")
    print(
        json.dumps(
            {"candidate_count": payload["candidate_count"], "selected_ordinal": 0},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
