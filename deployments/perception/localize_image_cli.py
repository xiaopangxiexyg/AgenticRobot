"""Run reviewed SAM2 + Grounding DINO inference on one local JPEG/PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps

from .sam2_grounding_dino import (
    GroundedObjectDetection,
    Sam2GroundingDinoLocalizer,
    Sam2GroundingDinoModelSpec,
)

OUTPUT_SCHEMA_V1 = "dexteleop.perception.localize_image_cli.v1"
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_INPUT_PIXELS = 16_777_216


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deployments.perception.localize_image_cli",
        description=(
            "Load the reviewed local SAM2/Grounding DINO asset bundle, localize "
            "objects in one JPEG/PNG, and write bounding-box JSON plus an annotated JPEG."
        ),
    )
    parser.add_argument("--asset-bundle", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument(
        "--language-prompt",
        default=None,
        help="Optional text target such as 'apple'; omit for automatic SAM2 proposals.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--automatic-score-threshold", type=float, default=0.80)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.70)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files.",
    )
    return parser


def _existing_file(value: Path, *, field_name: str) -> Path:
    path = value.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{field_name} must be an existing regular file: {path}")
    return path


def _existing_directory(value: Path, *, field_name: str) -> Path:
    path = value.expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory: {path}")
    return path


def _output_file(value: Path, *, field_name: str, overwrite: bool) -> Path:
    path = value.expanduser().resolve(strict=False)
    if not path.parent.is_dir():
        raise ValueError(f"{field_name} parent directory does not exist: {path.parent}")
    if path.exists() and (not path.is_file() or not overwrite):
        qualifier = "use --overwrite" if path.is_file() else "path is not a file"
        raise ValueError(f"{field_name} is unavailable ({qualifier}): {path}")
    return path


def _load_image(path: Path) -> tuple[Image.Image, str]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise ValueError("input image must be non-empty and at most 16 MiB")
    media_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("input image must be JPEG or PNG")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_INPUT_PIXELS:
                raise ValueError("input image dimensions exceed the reviewed bound")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("input image is not a valid bounded JPEG/PNG") from exc
    return image, media_hash


def _render(
    image: Image.Image,
    detections: Sequence[GroundedObjectDetection],
) -> bytes:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for ordinal, detection in enumerate(detections):
        x_min, y_min, x_max, y_max = detection.box_xyxy
        if not (
            0 <= x_min < x_max <= annotated.width
            and 0 <= y_min < y_max <= annotated.height
        ):
            raise RuntimeError("localizer returned an out-of-image bounding box")
        draw.rectangle(
            (x_min, y_min, x_max - 1, y_max - 1),
            outline=(255, 32, 32),
            width=3,
        )
        label = detection.label or "object"
        text = f"{ordinal}: {label} {float(detection.score):.3f}"
        text_box = draw.textbbox((x_min, y_min), text)
        text_height = max(1, text_box[3] - text_box[1])
        text_top = max(0, y_min - text_height - 4)
        draw.rectangle(
            (x_min, text_top, min(annotated.width - 1, text_box[2] + 4), y_min),
            fill=(255, 32, 32),
        )
        draw.text((x_min + 2, text_top + 1), text, fill=(255, 255, 255))
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as stream:
        annotated.save(stream, format="JPEG", quality=90, optimize=True)
        stream.seek(0)
        return stream.read()


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _existing_directory(args.asset_bundle, field_name="--asset-bundle")
    input_path = _existing_file(args.input_image, field_name="--input-image")
    output_json = _output_file(
        args.output_json,
        field_name="--output-json",
        overwrite=args.overwrite,
    )
    output_image = _output_file(
        args.output_image,
        field_name="--output-image",
        overwrite=args.overwrite,
    )
    if output_image.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("--output-image must end in .jpg or .jpeg")
    if len({input_path, output_json, output_image}) != 3:
        raise ValueError("input and output paths must be distinct")
    prompt = args.language_prompt
    if prompt is not None and (not prompt or prompt != prompt.strip()):
        raise ValueError("--language-prompt must be a non-empty trimmed string")
    if (
        isinstance(args.max_detections, bool)
        or not 1 <= args.max_detections <= 100
    ):
        raise ValueError("--max-detections must be in [1, 100]")

    spec = Sam2GroundingDinoModelSpec.from_asset_bundle(bundle)
    localizer = Sam2GroundingDinoLocalizer(
        model_spec=spec,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        automatic_score_threshold=args.automatic_score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )
    image, input_hash = _load_image(input_path)
    localizer.start()
    try:
        detections = localizer.localize(
            image,
            language_prompt=prompt,
            max_detections=args.max_detections,
        )
    finally:
        localizer.close()

    annotated = _render(image, detections)
    payload = {
        "schema_version": OUTPUT_SCHEMA_V1,
        "model_profile_id": spec.model_profile_id,
        "model_spec_hash": spec.semantic_content_hash,
        "input_image": str(input_path),
        "input_image_sha256": input_hash,
        "mode": "automatic" if prompt is None else "text",
        "language_prompt": prompt,
        "coordinate_format": "xyxy_pixel_0based_half_open",
        "image_size": {"width": image.width, "height": image.height},
        "detections": [
            {
                "ordinal": ordinal,
                "box_xyxy": list(detection.box_xyxy),
                "score": float(detection.score),
                "label": detection.label,
            }
            for ordinal, detection in enumerate(detections)
        ],
        "annotated_image": str(output_image),
        "annotated_image_sha256": hashlib.sha256(annotated).hexdigest(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(output_image, annotated)
    try:
        _atomic_write(output_json, encoded)
    except Exception:
        if not args.overwrite:
            output_image.unlink(missing_ok=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"object localization failed: {exc}\n")
    print(
        json.dumps(
            {
                "detections": len(payload["detections"]),
                "mode": payload["mode"],
                "output_json": str(args.output_json.expanduser().resolve(strict=False)),
                "output_image": str(args.output_image.expanduser().resolve(strict=False)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
