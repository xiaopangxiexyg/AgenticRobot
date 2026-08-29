"""Lazy local SAM2 + Grounding DINO inference adapter.

The external repositories are intentionally not vendored.  One reviewed asset
manifest pins their Git revisions and the exact checkpoint byte identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset_contracts.canonical_encoding import canonical_content_hash

SAM2_GIT_URL = "https://github.com/facebookresearch/sam2.git"
GROUNDING_DINO_GIT_URL = "https://github.com/IDEA-Research/GroundingDINO.git"
MODEL_ASSET_MANIFEST_SCHEMA_V1 = "dexteleop.perception_model_assets.v1"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _lower_hex(value: object, *, field_name: str, length: int) -> str:
    text = _required(value, field_name=field_name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be {length} lowercase hexadecimal characters.")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_git_revision(package_file: object, *, package_name: str) -> str:
    path = Path(_required(str(package_file), field_name=f"{package_name}.__file__"))
    completed = subprocess.run(
        ["git", "-C", str(path.resolve().parent), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{package_name} must be installed from a Git checkout so its revision "
            "can be verified."
        )
    return _lower_hex(
        completed.stdout.strip(),
        field_name=f"{package_name} installed Git revision",
        length=40,
    )


def _asset_path(root: Path, relative: object, *, field_name: str) -> Path:
    text = _required(relative, field_name=field_name)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field_name} must stay inside the model asset bundle.")
    resolved = (root / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes the model asset bundle.") from exc
    if not resolved.is_file():
        raise ValueError(f"{field_name} is not a regular file.")
    return resolved


@dataclass(frozen=True)
class Sam2GroundingDinoModelSpec:
    """Verified local paths plus portable source/checkpoint identities."""

    model_profile_id: str
    sam2_checkpoint_path: Path
    sam2_checkpoint_sha256: str
    sam2_model_config: str
    sam2_source_revision: str
    grounding_dino_checkpoint_path: Path
    grounding_dino_checkpoint_sha256: str
    grounding_dino_model_config: str
    grounding_dino_source_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_profile_id",
            _required(self.model_profile_id, field_name="model_profile_id"),
        )
        for field_name in ("sam2_checkpoint_path", "grounding_dino_checkpoint_path"):
            path = Path(getattr(self, field_name)).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{field_name} must be an existing file.")
            object.__setattr__(self, field_name, path)
        for field_name in (
            "sam2_checkpoint_sha256",
            "grounding_dino_checkpoint_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _lower_hex(getattr(self, field_name), field_name=field_name, length=64),
            )
        for field_name in ("sam2_source_revision", "grounding_dino_source_revision"):
            object.__setattr__(
                self,
                field_name,
                _lower_hex(getattr(self, field_name), field_name=field_name, length=40),
            )
        for field_name in ("sam2_model_config", "grounding_dino_model_config"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @classmethod
    def from_asset_bundle(cls, bundle_root: str | Path) -> Sam2GroundingDinoModelSpec:
        root = Path(bundle_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Object-localization model asset root must be a directory.")
        manifest_path = root / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read object-localization asset manifest: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "model_profile_id",
            "sources",
            "checkpoints",
        }:
            raise ValueError("Object-localization asset manifest fields are invalid.")
        if payload["schema_version"] != MODEL_ASSET_MANIFEST_SCHEMA_V1:
            raise ValueError("Unsupported object-localization asset manifest schema.")
        sources = payload["sources"]
        checkpoints = payload["checkpoints"]
        if not isinstance(sources, dict) or set(sources) != {"sam2", "grounding_dino"}:
            raise ValueError("Asset manifest must pin SAM2 and Grounding DINO sources.")
        if not isinstance(checkpoints, dict) or set(checkpoints) != {
            "sam2",
            "grounding_dino",
        }:
            raise ValueError("Asset manifest must identify both checkpoints.")
        expected_urls = {
            "sam2": SAM2_GIT_URL,
            "grounding_dino": GROUNDING_DINO_GIT_URL,
        }
        revisions: dict[str, str] = {}
        resolved: dict[str, tuple[Path, str, str]] = {}
        for name in ("sam2", "grounding_dino"):
            source = sources[name]
            checkpoint = checkpoints[name]
            if not isinstance(source, dict) or set(source) != {"git_url", "revision"}:
                raise ValueError(f"Asset manifest source {name!r} is invalid.")
            if source["git_url"] != expected_urls[name]:
                raise ValueError(f"Asset manifest source URL for {name!r} is not reviewed.")
            revisions[name] = _lower_hex(
                source["revision"],
                field_name=f"sources.{name}.revision",
                length=40,
            )
            if not isinstance(checkpoint, dict) or set(checkpoint) != {
                "relative_path",
                "sha256",
                "model_config",
            }:
                raise ValueError(f"Asset manifest checkpoint {name!r} is invalid.")
            path = _asset_path(
                root,
                checkpoint["relative_path"],
                field_name=f"checkpoints.{name}.relative_path",
            )
            expected_hash = _lower_hex(
                checkpoint["sha256"],
                field_name=f"checkpoints.{name}.sha256",
                length=64,
            )
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(f"Checkpoint hash mismatch for {name!r}.")
            resolved[name] = (
                path,
                expected_hash,
                _required(
                    checkpoint["model_config"],
                    field_name=f"checkpoints.{name}.model_config",
                ),
            )
        return cls(
            model_profile_id=payload["model_profile_id"],
            sam2_checkpoint_path=resolved["sam2"][0],
            sam2_checkpoint_sha256=resolved["sam2"][1],
            sam2_model_config=resolved["sam2"][2],
            sam2_source_revision=revisions["sam2"],
            grounding_dino_checkpoint_path=resolved["grounding_dino"][0],
            grounding_dino_checkpoint_sha256=resolved["grounding_dino"][1],
            grounding_dino_model_config=resolved["grounding_dino"][2],
            grounding_dino_source_revision=revisions["grounding_dino"],
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_profile_id": self.model_profile_id,
                "sam2": {
                    "git_url": SAM2_GIT_URL,
                    "source_revision": self.sam2_source_revision,
                    "checkpoint_sha256": self.sam2_checkpoint_sha256,
                    "model_config": self.sam2_model_config,
                },
                "grounding_dino": {
                    "git_url": GROUNDING_DINO_GIT_URL,
                    "source_revision": self.grounding_dino_source_revision,
                    "checkpoint_sha256": self.grounding_dino_checkpoint_sha256,
                    "model_config": self.grounding_dino_model_config,
                },
            },
            domain="dexteleop.perception.sam2_grounding_dino_model_spec.v1",
        )


@dataclass(frozen=True)
class GroundedObjectDetection:
    box_xyxy: tuple[int, int, int, int]
    score: float
    label: str | None

    def __post_init__(self) -> None:
        if len(self.box_xyxy) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.box_xyxy
        ):
            raise ValueError("box_xyxy must contain four integers.")
        x_min, y_min, x_max, y_max = self.box_xyxy
        if x_min < 0 or y_min < 0 or x_min >= x_max or y_min >= y_max:
            raise ValueError("box_xyxy must have positive area.")
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be finite and in [0, 1].")
        if self.label is not None:
            _required(self.label, field_name="label")


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _bounded_nms(
    detections: list[GroundedObjectDetection],
    *,
    threshold: float,
    maximum: int,
) -> tuple[GroundedObjectDetection, ...]:
    selected: list[GroundedObjectDetection] = []
    for detection in sorted(detections, key=lambda item: (-item.score, item.box_xyxy)):
        if all(_box_iou(detection.box_xyxy, item.box_xyxy) <= threshold for item in selected):
            selected.append(detection)
            if len(selected) == maximum:
                break
    return tuple(selected)


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(columns):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


class Sam2GroundingDinoLocalizer:
    """Lifecycle-owned inference engine; imports external code only on start."""

    def __init__(
        self,
        *,
        model_spec: Sam2GroundingDinoModelSpec,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        automatic_score_threshold: float = 0.80,
        nms_iou_threshold: float = 0.70,
    ) -> None:
        if not isinstance(model_spec, Sam2GroundingDinoModelSpec):
            raise TypeError("model_spec must be typed.")
        self.model_spec = model_spec
        self.device = _required(device, field_name="device")
        for field_name, value in (
            ("box_threshold", box_threshold),
            ("text_threshold", text_threshold),
            ("automatic_score_threshold", automatic_score_threshold),
            ("nms_iou_threshold", nms_iou_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{field_name} must be in [0, 1].")
            setattr(self, field_name, float(value))
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._automatic_generator: Any = None
        self._predictor: Any = None
        self._grounding_model: Any = None
        self._grounding_predict: Any = None
        self._grounding_transform: Any = None

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_spec_hash": self.model_spec.semantic_content_hash,
                "device": self.device,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "automatic_score_threshold": self.automatic_score_threshold,
                "nms_iou_threshold": self.nms_iou_threshold,
            },
            domain="dexteleop.perception.sam2_grounding_dino_localizer.v1",
        )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed object localizer cannot restart.")
            if self._started:
                return
            try:
                import groundingdino
                import sam2
                from groundingdino.datasets import transforms as grounding_transforms
                from groundingdino.util.inference import load_model, predict
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM2/Grounding DINO source packages are not installed; see "
                    "docs/agent_gateway/SAM2_GROUNDING_DINO.md."
                ) from exc
            installed_revisions = {
                "sam2": _installed_git_revision(sam2.__file__, package_name="sam2"),
                "grounding_dino": _installed_git_revision(
                    groundingdino.__file__,
                    package_name="groundingdino",
                ),
            }
            expected_revisions = {
                "sam2": self.model_spec.sam2_source_revision,
                "grounding_dino": self.model_spec.grounding_dino_source_revision,
            }
            if installed_revisions != expected_revisions:
                raise RuntimeError(
                    "Installed SAM2/Grounding DINO Git revisions differ from the "
                    "reviewed asset manifest."
                )
            sam_model = build_sam2(
                self.model_spec.sam2_model_config,
                str(self.model_spec.sam2_checkpoint_path),
                device=self.device,
            )
            self._automatic_generator = SAM2AutomaticMaskGenerator(sam_model)
            self._predictor = SAM2ImagePredictor(sam_model)
            grounding_config = Path(self.model_spec.grounding_dino_model_config)
            if not grounding_config.is_file():
                grounding_config = (
                    Path(groundingdino.__file__).resolve().parent
                    / "config"
                    / grounding_config.name
                )
            if not grounding_config.is_file():
                raise RuntimeError(
                    "Grounding DINO model config is absent from the installed source package."
                )
            self._grounding_model = load_model(
                str(grounding_config),
                str(self.model_spec.grounding_dino_checkpoint_path),
                device=self.device,
            )
            self._grounding_predict = predict
            self._grounding_transform = grounding_transforms.Compose(
                [
                    grounding_transforms.RandomResize([800], max_size=1333),
                    grounding_transforms.ToTensor(),
                    grounding_transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )
            self._started = True

    def close(self) -> None:
        with self._lock:
            self._automatic_generator = None
            self._predictor = None
            self._grounding_model = None
            self._grounding_predict = None
            self._grounding_transform = None
            self._closed = True
            self._started = False

    def localize(
        self,
        image: Image.Image,
        *,
        language_prompt: str | None,
        max_detections: int,
    ) -> tuple[GroundedObjectDetection, ...]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")
        if (
            isinstance(max_detections, bool)
            or not isinstance(max_detections, int)
            or max_detections <= 0
        ):
            raise ValueError("max_detections must be positive.")
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("Object-localization model is not running.")
            if language_prompt is None:
                detections = self._automatic(array)
            else:
                detections = self._grounded(rgb, array, language_prompt)
        return _bounded_nms(
            detections,
            threshold=self.nms_iou_threshold,
            maximum=max_detections,
        )

    def segment_point(self, image: Image.Image, *, pixel_uv: tuple[int, int]) -> np.ndarray:
        """Return the highest-confidence SAM2 mask prompted by one image pixel."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")
        if (
            not isinstance(pixel_uv, tuple)
            or len(pixel_uv) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel_uv)
        ):
            raise TypeError("pixel_uv must contain two integers.")
        u, v = pixel_uv
        rgb = image.convert("RGB")
        if not (0 <= u < rgb.width and 0 <= v < rgb.height):
            raise ValueError("pixel_uv is outside the image.")
        array = np.asarray(rgb, dtype=np.uint8)
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("Object-localization model is not running.")
            self._predictor.set_image(array)
            masks, scores, _logits = self._predictor.predict(
                point_coords=np.asarray([[u, v]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32),
                multimask_output=True,
            )
        if not len(masks):
            raise RuntimeError("SAM2 did not return a point-prompted mask.")
        best = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        mask = np.asarray(masks[best], dtype=bool)
        if mask.shape != (rgb.height, rgb.width) or not bool(mask[v, u]):
            raise RuntimeError("SAM2 point mask does not contain the requested pixel.")
        return mask

    def _automatic(self, image: np.ndarray) -> list[GroundedObjectDetection]:
        result = []
        for candidate in self._automatic_generator.generate(image):
            score = float(candidate.get("predicted_iou", 0.0))
            if score < self.automatic_score_threshold:
                continue
            box = candidate.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x, y, width, height = (float(value) for value in box)
            x_min = max(0, int(math.floor(x)))
            y_min = max(0, int(math.floor(y)))
            x_max = min(image.shape[1], int(math.ceil(x + width)))
            y_max = min(image.shape[0], int(math.ceil(y + height)))
            if x_min < x_max and y_min < y_max:
                result.append(
                    GroundedObjectDetection(
                        box_xyxy=(x_min, y_min, x_max, y_max),
                        score=min(1.0, max(0.0, score)),
                        label=None,
                    )
                )
        return result

    def _grounded(
        self,
        image: Image.Image,
        array: np.ndarray,
        language_prompt: str,
    ) -> list[GroundedObjectDetection]:
        transformed, _target = self._grounding_transform(image, None)
        boxes, logits, phrases = self._grounding_predict(
            model=self._grounding_model,
            image=transformed,
            caption=language_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        width, height = image.size
        self._predictor.set_image(array)
        result = []
        for box, logit, phrase in zip(boxes, logits, phrases, strict=True):
            cx, cy, box_width, box_height = (float(value) for value in box.tolist())
            input_box = np.asarray(
                [
                    (cx - box_width / 2.0) * width,
                    (cy - box_height / 2.0) * height,
                    (cx + box_width / 2.0) * width,
                    (cy + box_height / 2.0) * height,
                ],
                dtype=np.float32,
            )
            masks, mask_scores, _logits = self._predictor.predict(
                box=input_box,
                multimask_output=True,
            )
            if len(masks):
                best = int(np.argmax(np.asarray(mask_scores)))
                refined = _mask_box(np.asarray(masks[best]))
            else:
                refined = None
            if refined is None:
                refined = (
                    max(0, int(math.floor(input_box[0]))),
                    max(0, int(math.floor(input_box[1]))),
                    min(width, int(math.ceil(input_box[2]))),
                    min(height, int(math.ceil(input_box[3]))),
                )
            if refined[0] < refined[2] and refined[1] < refined[3]:
                result.append(
                    GroundedObjectDetection(
                        box_xyxy=refined,
                        score=min(1.0, max(0.0, float(logit))),
                        label=_required(str(phrase), field_name="grounding phrase"),
                    )
                )
        return result


__all__ = [
    "GROUNDING_DINO_GIT_URL",
    "MODEL_ASSET_MANIFEST_SCHEMA_V1",
    "SAM2_GIT_URL",
    "GroundedObjectDetection",
    "Sam2GroundingDinoLocalizer",
    "Sam2GroundingDinoModelSpec",
]
