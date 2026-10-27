"""Thin, opt-in client for the existing SAM3 text/point HTTP RPC service.

The service owns its model and process. This client never imports SAM3, starts
or stops the server, or changes the local SAM2 runtime. Its profile identifies
the configured service contract, not an attestation of the remote checkpoint.
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

from dataset_contracts.canonical_encoding import canonical_content_hash

from .sam2_grounding_dino import (
    GroundedObjectDetection,
    GroundedObjectSegment,
    Sam2PointMaskUnavailable,
    encode_grounded_object_segment_artifacts,
)

DEFAULT_SAM3_ENDPOINT = "http://127.0.0.1:8114"
MAX_RPC_RESPONSE_BYTES = 32 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("SAM3 loopback service must not redirect requests")


@dataclass(frozen=True)
class Sam3ServiceModelSpec:
    endpoint: str
    model_profile_id: str = "sam3-service-text-point-v1"
    rpc_contract: str = "sam3.segment:text_or_row_col:single_best_png_mask"

    def __post_init__(self) -> None:
        url = urllib.parse.urlsplit(self.endpoint)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or (url.port is not None and not 1 <= url.port <= 65535)
        ):
            raise ValueError(
                "SAM3 endpoint must be a loopback HTTP origin without credentials"
            )
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_profile_id": self.model_profile_id,
                "endpoint": self.endpoint,
                "rpc": self.rpc_contract,
                "checkpoint_identity": "external_service_not_attested",
            },
            domain="dexteleop.perception.sam3_service_binding.v1",
        )


class Sam3ServiceLocalizer:
    """Use the same read-only localizer/point-segmenter ports as SAM2.

    The default text request returns one instance. Opt-in multi-instance
    localization uses the service's bounded candidate RPC; unprompted automatic
    proposals remain unsupported. Model/transport failures are never retried
    or substituted with another segmentation backend.
    """

    inference_execution_mode = "external_service"

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_SAM3_ENDPOINT,
        timeout_sec: float = 120.0,
        min_score: float = 0.2,
        multi_instance_text: bool = False,
    ) -> None:
        if not isinstance(multi_instance_text, bool):
            raise TypeError("multi_instance_text must be boolean")
        self.multi_instance_text = multi_instance_text
        self.model_spec = Sam3ServiceModelSpec(
            endpoint,
            model_profile_id=(
                "sam3-service-text-candidates-v1" if multi_instance_text
                else "sam3-service-text-point-v1"
            ),
            rpc_contract=(
                "sam3.segment_candidates:text:bounded_distinct_png_masks_v1"
                if multi_instance_text else "sam3.segment:text_or_row_col:single_best_png_mask"
            ),
        )
        for name, value in (("timeout_sec", timeout_sec), ("min_score", min_score)):
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"SAM3 {name} must be finite")
        if timeout_sec <= 0 or not 0 <= min_score <= 1:
            raise ValueError(
                "SAM3 timeout must be positive and min_score within [0, 1]"
            )
        self.timeout_sec, self.min_score = float(timeout_sec), float(min_score)
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )
        self._lock = threading.RLock()
        self._started = self._closed = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started and not self._closed

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_spec_hash": self.model_spec.semantic_content_hash,
                "timeout_sec": self.timeout_sec,
                "min_score": self.min_score,
                "mask_result_contract": (
                    "grounded_object_segment_candidates_with_shared_artifacts_v1"
                    if self.multi_instance_text else "grounded_object_segment_with_shared_artifacts_v1"
                ),
            },
            domain=(
                "dexteleop.perception.sam3_service_localizer.v2"
                if self.multi_instance_text else "dexteleop.perception.sam3_service_localizer.v1"
            ),
        )

    def _call(self, method: str, kwargs: dict) -> dict:
        request = urllib.request.Request(
            self.model_spec.endpoint + "/call",
            data=json.dumps({"method": method, "args": [], "kwargs": kwargs}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_sec) as response:
                raw = response.read(MAX_RPC_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"SAM3 {method} transport failed: {exc}") from exc
        if len(raw) > MAX_RPC_RESPONSE_BYTES:
            raise RuntimeError("SAM3 response exceeds its byte bound")
        try:
            envelope = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("SAM3 service returned invalid JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            raise RuntimeError(f"SAM3 {method} RPC failed: {envelope}")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("SAM3 RPC result must be an object")  # noqa: TRY004 - remote protocol fault
        return result

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed SAM3 client cannot restart")
            if self._started:
                return
            if self._call("healthz", {}).get("status") != "ok":
                raise RuntimeError("SAM3 service health check failed")
            self._started = True

    def close(self) -> None:
        # The externally managed service may have other clients. No shutdown RPC.
        with self._lock:
            self._closed, self._started = True, False

    def _segment(
        self, image: Image.Image, prompt: dict
    ) -> tuple[np.ndarray | None, float]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("SAM3 localizer is not running")
            stream = io.BytesIO()
            image.convert("RGB").save(stream, format="PNG")
            result = self._call(
                "sam3.segment",
                {
                    "image_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
                    "min_score": self.min_score,
                    **prompt,
                },
            )
        if not isinstance(result.get("found"), bool):
            raise RuntimeError("SAM3 response lacks a boolean found field")  # noqa: TRY004 - remote protocol fault
        if not result["found"]:
            score = result.get("score")
            if score is None:
                return None, 0.0
            self._validated_score(score)
            return None, float(score)
        return self._decode_mask_result(result, image)

    @staticmethod
    def _validated_score(score: object) -> float:
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise RuntimeError("SAM3 returned an invalid score")
        return float(score)

    def _decode_mask_result(
        self, result: dict, image: Image.Image
    ) -> tuple[np.ndarray, float]:
        score = result.get("score")
        if score is None:
            raise RuntimeError("SAM3 returned found=True without a score")
        score = self._validated_score(score)
        if score < self.min_score:
            raise RuntimeError(
                "SAM3 returned found=True below its requested score threshold"
            )
        if result.get("mask_shape") != [image.height, image.width]:
            raise RuntimeError("SAM3 mask shape differs from the input image")
        try:
            raw = base64.b64decode(result["mask_png_base64"], validate=True)
            with Image.open(io.BytesIO(raw)) as mask_image:
                if (
                    mask_image.format != "PNG"
                    or mask_image.size != image.size
                    or mask_image.mode != "L"
                ):
                    raise ValueError("mask must be an input-sized grayscale PNG")
                mask_values = np.asarray(mask_image).copy()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise RuntimeError(f"SAM3 returned an invalid PNG mask: {exc}") from exc
        if not np.isin(mask_values, (0, 255)).all() or not mask_values.any():
            raise RuntimeError("SAM3 returned a non-binary or empty mask")
        return mask_values > 0, score

    def localize(
        self, image: Image.Image, *, language_prompt: str | None, max_detections: int
    ) -> tuple[GroundedObjectDetection, ...]:
        if not isinstance(language_prompt, str) or not language_prompt.strip():
            raise ValueError(
                "SAM3 service requires a text prompt; automatic proposals are unsupported"
            )
        if (
            isinstance(max_detections, bool)
            or not isinstance(max_detections, int)
            or max_detections <= 0
        ):
            raise ValueError("max_detections must be positive")
        if self.multi_instance_text:
            segments = self.segment_candidates(
                image, language_prompt=language_prompt,
                max_candidates=min(max_detections, 8),
            )
            return tuple(
                GroundedObjectDetection(
                    box_xyxy=segment.box_xyxy,
                    score=segment.score,
                    label=segment.label,
                ) for segment in segments
            )
        segment = self.segment(
            image, language_prompt=language_prompt, pixel_uv=None,
            min_score=self.min_score,
        )
        if segment is None:
            return ()
        return (
            GroundedObjectDetection(
                box_xyxy=segment.box_xyxy,
                score=segment.score,
                label=segment.label,
            ),
        )

    def segment_candidates(
        self, image: Image.Image, *, language_prompt: str,
        max_candidates: int = 4,
    ) -> tuple[GroundedObjectSegment, ...]:
        """Read bounded text instances without changing the single-mask port."""
        if not self.multi_instance_text:
            raise RuntimeError("SAM3 multi-instance text mode is not enabled")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if not isinstance(language_prompt, str) or not language_prompt.strip():
            raise ValueError("language_prompt must contain text")
        if (
            isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= 8
        ):
            raise ValueError("max_candidates must be an integer between 1 and 8")
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("SAM3 localizer is not running")
            stream = io.BytesIO()
            image.convert("RGB").save(stream, format="PNG")
            response = self._call("sam3.segment_candidates", {
                "image_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
                "text_prompt": language_prompt.strip(),
                "min_score": self.min_score,
                "max_candidates": max_candidates,
            })
        candidates = response.get("candidates")
        raw_count = response.get("raw_candidate_count")
        if (
            not isinstance(candidates, list) or len(candidates) > max_candidates
            or isinstance(raw_count, bool) or not isinstance(raw_count, int)
            or raw_count < len(candidates)
        ):
            raise RuntimeError("SAM3 returned invalid candidate collection")
        segments = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise RuntimeError("SAM3 returned invalid candidate entry")
            mask, score = self._decode_mask_result(candidate, image)
            rows, cols = np.nonzero(mask)
            segments.append(GroundedObjectSegment(
                mask=mask, score=score, label=language_prompt.strip(),
                box_xyxy=(int(cols.min()), int(rows.min()),
                          int(cols.max()) + 1, int(rows.max()) + 1),
            ))
        return tuple(segments)

    def segment_point(
        self, image: Image.Image, *, pixel_uv: tuple[int, int]
    ) -> np.ndarray:
        segment = self.segment(
            image, language_prompt=None, pixel_uv=pixel_uv, min_score=0.0,
        )
        if segment is None:
            raise Sam2PointMaskUnavailable(
                "SAM3 point mask does not contain the requested pixel"
            )
        return segment.mask

    encode_segment_artifacts = staticmethod(encode_grounded_object_segment_artifacts)

    def segment(
        self, image: Image.Image, *, language_prompt: str | None,
        pixel_uv: tuple[int, int] | None, min_score: float,
    ) -> GroundedObjectSegment | None:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if (language_prompt is None) == (pixel_uv is None):
            raise ValueError("Exactly one language_prompt or pixel_uv is required")
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score)
            or not 0 <= min_score <= 1
        ):
            raise ValueError("min_score must be finite and in [0, 1]")
        label = None
        if language_prompt is not None:
            if not isinstance(language_prompt, str) or not language_prompt.strip():
                raise ValueError("language_prompt must contain text")
            label = language_prompt.strip()
            prompt = {"text_prompt": label}
        else:
            if (
                not isinstance(pixel_uv, tuple)
                or len(pixel_uv) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in pixel_uv)
            ):
                raise TypeError("pixel_uv must contain two integers")
            u, v = pixel_uv
            if not (0 <= u < image.width and 0 <= v < image.height):
                raise ValueError("pixel_uv is outside the image")
            # The RPC uses row/column; the local port uses u/v.
            prompt = {"point": [v, u]}
        mask, score = self._segment(image, prompt)
        if mask is None:
            return None
        if pixel_uv is not None and not mask[pixel_uv[1], pixel_uv[0]]:
            raise Sam2PointMaskUnavailable(
                "SAM3 point mask does not contain the requested pixel"
            )
        if score < min_score:
            return None
        rows, cols = np.nonzero(mask)
        return GroundedObjectSegment(
            mask=mask, score=score, label=label,
            box_xyxy=(int(cols.min()), int(rows.min()),
                      int(cols.max()) + 1, int(rows.max()) + 1),
        )
