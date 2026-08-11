"""Optional, negotiated image transport for the OpenPI websocket boundary.

The historical OpenPI protocol sends RGB ``uint8`` arrays through msgpack as
raw bytes.  That path remains the default and this module never rewrites it.
Optimized clients encode images only after the server advertises this exact
versioned capability in its initial metadata frame.

This module is shared by the DexTeleop client and the prepared OpenPI server
adapter so deployment and training-time wire simulation use one codec.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from io import BytesIO
import time
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash


OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1 = "dexteleop_image_wire_v1"
OPENPI_IMAGE_WIRE_CAPABILITY_SCHEMA_V1 = (
    "dexteleop.openpi_image_wire.capability.v1"
)
OPENPI_IMAGE_WIRE_ENVELOPE_SCHEMA_V1 = "dexteleop.openpi_image_wire.envelope.v1"
OPENPI_IMAGE_WIRE_CAPABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_image_wire.capability.v1"
)
OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1 = "image_wire_training_alignment"
OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1 = (
    "dexteleop.openpi_image_wire.training_alignment.v1"
)
OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2 = (
    "dexteleop.openpi_image_wire.training_alignment.v2"
)
OPENPI_IMAGE_WIRE_TRAINING_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_image_wire.training_alignment.v1"
)
OPENPI_IMAGE_WIRE_TRAINING_HASH_DOMAIN_V2 = (
    "dexteleop.openpi_image_wire.training_alignment.v2"
)
OPENPI_IMAGE_WIRE_TRAINING_SELECTION_SCHEMA_V2 = (
    "dexteleop.openpi_image_wire.training_selection.v2"
)
OPENPI_IMAGE_WIRE_TRAINING_SELECTION_HASH_DOMAIN_V2 = (
    "dexteleop.openpi_image_wire.training_selection.v2"
)

OPENPI_IMAGE_WIRE_LEGACY_RAW_V1 = "legacy_raw_rgb8_v1"
OPENPI_IMAGE_WIRE_RESIZE_RAW_V1 = "resize_raw_rgb8_v1"
OPENPI_IMAGE_WIRE_JPEG_V1 = "jpeg_rgb8_v1"
OPENPI_IMAGE_WIRE_CODECS_V1 = frozenset(
    (
        OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,
        OPENPI_IMAGE_WIRE_JPEG_V1,
    )
)

OPENPI_IMAGE_WIRE_RESIZE_METHOD_V1 = "openpi_resize_with_pad_pil_bilinear_v1"
OPENPI_IMAGE_WIRE_JPEG_ENCODER_V1 = "pillow_rgb_420_nonprogressive_v1"
OPENPI_IMAGE_WIRE_COLOR_SPACE_RGB = "rgb"
OPENPI_IMAGE_WIRE_DTYPE_UINT8 = "uint8"
_IMAGE_WIRE_MARKER = "__dexteleop_image_wire__"


class OpenPIImageWireError(ValueError):
    """Raised when an advertised image-wire contract cannot be honored."""


@dataclass(frozen=True)
class OpenPIImageWireClientConfig:
    """Client preference; legacy raw remains the default and exact rollback."""

    preferred_codec: str = OPENPI_IMAGE_WIRE_LEGACY_RAW_V1
    allow_legacy_fallback: bool = True
    image_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        codec = str(self.preferred_codec).strip()
        if codec not in OPENPI_IMAGE_WIRE_CODECS_V1 and codec != (
            OPENPI_IMAGE_WIRE_LEGACY_RAW_V1
        ):
            raise OpenPIImageWireError(
                f"Unsupported preferred image-wire codec: {codec!r}."
            )
        keys = tuple(str(value).strip() for value in self.image_keys)
        if any(not value for value in keys) or len(set(keys)) != len(keys):
            raise OpenPIImageWireError(
                "image_keys must contain unique, non-empty strings."
            )
        object.__setattr__(self, "preferred_codec", codec)
        object.__setattr__(self, "allow_legacy_fallback", bool(self.allow_legacy_fallback))
        object.__setattr__(self, "image_keys", keys)

    @property
    def enabled(self) -> bool:
        return self.preferred_codec != OPENPI_IMAGE_WIRE_LEGACY_RAW_V1

    @classmethod
    def from_value(
        cls,
        value: "OpenPIImageWireClientConfig | Mapping[str, Any] | None",
        *,
        image_keys: Sequence[str] | None = None,
    ) -> "OpenPIImageWireClientConfig":
        if value is None:
            config = cls()
        elif isinstance(value, cls):
            config = value
        elif isinstance(value, Mapping):
            allowed = {"preferred_codec", "allow_legacy_fallback", "image_keys"}
            unexpected = set(value) - allowed
            if unexpected:
                raise OpenPIImageWireError(
                    "Unexpected image-wire client fields: "
                    f"{sorted(str(key) for key in unexpected)!r}."
                )
            config = cls(
                preferred_codec=str(
                    value.get("preferred_codec", OPENPI_IMAGE_WIRE_LEGACY_RAW_V1)
                ),
                allow_legacy_fallback=bool(
                    value.get("allow_legacy_fallback", True)
                ),
                image_keys=tuple(str(key) for key in value.get("image_keys", ())),
            )
        else:
            raise TypeError(
                "image_wire must be an OpenPIImageWireClientConfig, mapping, or None."
            )

        bound_keys = () if image_keys is None else tuple(str(key) for key in image_keys)
        if bound_keys:
            if config.image_keys and config.image_keys != bound_keys:
                raise OpenPIImageWireError(
                    "image-wire image_keys disagree with the policy camera slots."
                )
            config = replace(config, image_keys=bound_keys)
        if config.enabled and not config.image_keys:
            raise OpenPIImageWireError(
                "An optimized image-wire codec requires explicit policy image_keys."
            )
        return config


@dataclass(frozen=True)
class OpenPIImageWireServerConfig:
    """Prepared-server capability profile.

    ``jpeg`` advertises both JPEG and resized-raw so a client can roll back one
    optimization level without restarting the server.  Legacy raw ndarray
    requests are always accepted by the underlying OpenPI server.
    """

    profile: str = "off"
    target_height: int = 224
    target_width: int = 224
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        profile = str(self.profile).strip().lower().replace("_", "-")
        if profile not in ("off", "resize-raw", "jpeg"):
            raise OpenPIImageWireError(
                "image-wire server profile must be off, resize-raw, or jpeg."
            )
        height = int(self.target_height)
        width = int(self.target_width)
        quality = int(self.jpeg_quality)
        if height <= 0 or width <= 0:
            raise OpenPIImageWireError(
                "image-wire target height and width must be positive."
            )
        if not 1 <= quality <= 100:
            raise OpenPIImageWireError(
                "image-wire JPEG quality must be within [1, 100]."
            )
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "target_height", height)
        object.__setattr__(self, "target_width", width)
        object.__setattr__(self, "jpeg_quality", quality)

    @property
    def enabled(self) -> bool:
        return self.profile != "off"

    @property
    def supported_codecs(self) -> tuple[str, ...]:
        if self.profile == "off":
            return ()
        if self.profile == "resize-raw":
            return (OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,)
        return (
            OPENPI_IMAGE_WIRE_JPEG_V1,
            OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,
        )


@dataclass(frozen=True)
class OpenPIImageWireEncodeReceipt:
    requested_codec: str
    effective_codec: str
    fallback_reason: str | None
    capability_hash: str | None
    image_count: int
    source_image_bytes: int
    encoded_image_bytes: int
    encode_ms: float
    packed_request_bytes: int | None = None
    response_bytes: int | None = None

    def with_exchange_sizes(
        self,
        *,
        packed_request_bytes: int,
        response_bytes: int,
    ) -> "OpenPIImageWireEncodeReceipt":
        return replace(
            self,
            packed_request_bytes=int(packed_request_bytes),
            response_bytes=int(response_bytes),
        )


@dataclass(frozen=True)
class OpenPIImageWireDecodeReceipt:
    codecs: tuple[str, ...]
    image_count: int
    encoded_image_bytes: int
    decoded_image_bytes: int


@dataclass(frozen=True)
class OpenPIImageWireTrainingTransform:
    """Pickle-safe, opt-in simulation of the deployment wire at dataset output.

    OpenPI's normal model transform still owns model preprocessing.  This
    transform runs one step earlier: after DexTeleop dataset augmentation and
    before OpenPI's model resize.  Since it emits the advertised target shape,
    the existing OpenPI resize is then an exact no-op.
    """

    server_config: OpenPIImageWireServerConfig
    image_keys: tuple[str, ...]
    application_probability: float = 1.0
    contract_schema: str = OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1
    upstream: Any = None
    _epoch: int = 0
    _selection_contract_hash: str = field(
        default="",
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.server_config, OpenPIImageWireServerConfig):
            raise TypeError("server_config must be an OpenPIImageWireServerConfig.")
        if not self.server_config.enabled:
            raise OpenPIImageWireError(
                "Training alignment requires resize-raw or jpeg, not off."
            )
        keys = tuple(str(value).strip() for value in self.image_keys)
        if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
            raise OpenPIImageWireError(
                "Training alignment requires unique, non-empty image_keys."
            )
        if self.upstream is not None and not callable(self.upstream):
            raise TypeError("upstream training transform must be callable or None.")
        raw_probability = self.application_probability
        if isinstance(raw_probability, (bool, np.bool_)) or not isinstance(
            raw_probability,
            (int, float, np.integer, np.floating),
        ):
            raise OpenPIImageWireError(
                "Training alignment application_probability must be numeric."
            )
        probability = float(raw_probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise OpenPIImageWireError(
                "Training alignment application_probability must be within [0, 1]."
            )
        schema = str(self.contract_schema).strip()
        if schema not in (
            OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1,
            OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2,
        ):
            raise OpenPIImageWireError(
                "Unsupported openpi_image_wire_training schema."
            )
        if schema == OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1 and probability != 1.0:
            raise OpenPIImageWireError(
                "Image-wire training v1 requires application_probability=1.0."
            )
        object.__setattr__(self, "image_keys", keys)
        object.__setattr__(self, "application_probability", probability)
        object.__setattr__(self, "contract_schema", schema)
        object.__setattr__(self, "_epoch", max(0, int(self._epoch)))
        object.__setattr__(
            self,
            "_selection_contract_hash",
            self.contract["contract_hash"],
        )

    @classmethod
    def from_profile_value(
        cls,
        value: Mapping[str, Any] | None,
        *,
        image_keys: Sequence[str],
        upstream: Any = None,
    ) -> "OpenPIImageWireTrainingTransform | None":
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise OpenPIImageWireError(
                "openpi_image_wire_training must be a mapping when present."
            )
        schema = str(value.get("schema", "")).strip()
        if schema == OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1:
            required = {"schema", "profile", "target_shape", "jpeg_quality"}
        elif schema == OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2:
            required = {
                "schema",
                "profile",
                "target_shape",
                "jpeg_quality",
                "application_probability",
            }
        else:
            raise OpenPIImageWireError(
                "Unsupported openpi_image_wire_training schema."
            )
        if set(value) != required:
            raise OpenPIImageWireError(
                "openpi_image_wire_training fields differ from its schema contract."
            )
        probability: float | int = 1.0
        if schema == OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2:
            probability = value["application_probability"]
        shape_value = value["target_shape"]
        if not isinstance(shape_value, Sequence) or isinstance(
            shape_value, (str, bytes)
        ):
            raise OpenPIImageWireError(
                "openpi_image_wire_training.target_shape must be a sequence."
            )
        shape = tuple(int(item) for item in shape_value)
        if len(shape) != 3 or shape[2] != 3:
            raise OpenPIImageWireError(
                "openpi_image_wire_training.target_shape must be [H, W, 3]."
            )
        return cls(
            server_config=OpenPIImageWireServerConfig(
                profile=str(value["profile"]),
                target_height=shape[0],
                target_width=shape[1],
                jpeg_quality=int(value["jpeg_quality"]),
            ),
            image_keys=tuple(str(key) for key in image_keys),
            application_probability=probability,
            contract_schema=schema,
            upstream=upstream,
        )

    @property
    def codec(self) -> str:
        return (
            OPENPI_IMAGE_WIRE_JPEG_V1
            if self.server_config.profile == "jpeg"
            else OPENPI_IMAGE_WIRE_RESIZE_RAW_V1
        )

    @property
    def contract(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.contract_schema,
            "application_probability": self.application_probability,
            "transform_order": (
                "after_dataset_augmentation_before_openpi_model_resize_v1"
            ),
            "server_capability": build_openpi_image_wire_capability(
                self.server_config,
                image_keys=self.image_keys,
            ),
        }
        hash_domain = OPENPI_IMAGE_WIRE_TRAINING_HASH_DOMAIN_V1
        if self.contract_schema == OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2:
            payload["bypass_semantics"] = "preserve_upstream_sample_v1"
            payload["selection"] = {
                "schema": OPENPI_IMAGE_WIRE_TRAINING_SELECTION_SCHEMA_V2,
                "sample_identity": (
                    "prepared_organized_episode_ordinal_and_row_index_v1"
                ),
                "epoch": "openpi_shared_worker_epoch_v1",
                "hash_domain": (
                    OPENPI_IMAGE_WIRE_TRAINING_SELECTION_HASH_DOMAIN_V2
                ),
            }
            hash_domain = OPENPI_IMAGE_WIRE_TRAINING_HASH_DOMAIN_V2
        payload["contract_hash"] = canonical_content_hash(
            payload,
            domain=hash_domain,
        )
        return payload

    def with_upstream(self, upstream: Any) -> "OpenPIImageWireTrainingTransform":
        return replace(self, upstream=upstream)

    def _apply(self, sample: Mapping[str, Any], *, apply_codec: bool) -> Any:
        value = self.upstream(sample) if self.upstream is not None else sample
        if not isinstance(value, Mapping):
            raise OpenPIImageWireError(
                "Image-wire training transform requires a mapping sample."
            )
        if not apply_codec:
            return value
        images_value = value.get("image")
        if images_value is None:
            return value
        if not isinstance(images_value, Mapping):
            raise OpenPIImageWireError(
                "Image-wire training sample image field must be a mapping."
            )
        images = dict(images_value)
        for key in self.image_keys:
            if key not in images:
                raise OpenPIImageWireError(
                    f"Image-wire training sample is missing image key {key!r}."
                )
            images[key] = simulate_openpi_image_wire_roundtrip(
                images[key],
                codec=self.codec,
                target_height=self.server_config.target_height,
                target_width=self.server_config.target_width,
                jpeg_quality=self.server_config.jpeg_quality,
            )
        output = dict(value)
        output["image"] = images
        return output

    def applies_to_identity(
        self,
        identity: Sequence[int] | np.ndarray,
        *,
        epoch: int | None = None,
    ) -> bool:
        """Return the deterministic codec branch for one prepared sample."""

        values = np.asarray(identity)
        if values.shape != (2,) or values.dtype.kind not in {"i", "u"}:
            raise OpenPIImageWireError(
                "Image-wire training identity must contain two integers."
            )
        organized_episode_ordinal = int(values[0])
        row_index = int(values[1])
        if organized_episode_ordinal < 0 or row_index < 0:
            raise OpenPIImageWireError(
                "Image-wire training identity values must be non-negative."
            )
        if self.application_probability <= 0.0:
            return False
        if self.application_probability >= 1.0:
            return True
        selected_epoch = self._epoch if epoch is None else max(0, int(epoch))
        decision_hash = canonical_content_hash(
            {
                "schema": OPENPI_IMAGE_WIRE_TRAINING_SELECTION_SCHEMA_V2,
                "contract_hash": self._selection_contract_hash,
                "organized_episode_ordinal": organized_episode_ordinal,
                "row_index": row_index,
                "epoch": selected_epoch,
            },
            domain=OPENPI_IMAGE_WIRE_TRAINING_SELECTION_HASH_DOMAIN_V2,
        )
        threshold = int(self.application_probability * (1 << 64))
        return int(decision_hash[:16], 16) < threshold

    def apply_with_identity(
        self,
        sample: Mapping[str, Any],
        *,
        identity: Sequence[int] | np.ndarray,
    ) -> Any:
        return self._apply(
            sample,
            apply_codec=self.applies_to_identity(identity),
        )

    def selection_summary(
        self,
        identities: Iterator[Sequence[int] | np.ndarray],
        *,
        epoch: int,
    ) -> dict[str, Any]:
        total = 0
        applied = 0
        for identity in identities:
            total += 1
            applied += int(self.applies_to_identity(identity, epoch=epoch))
        bypassed = total - applied
        return {
            "schema": "dexteleop.openpi_image_wire.training_selection_summary.v1",
            "epoch": max(0, int(epoch)),
            "configured_application_probability": self.application_probability,
            "sample_count": total,
            "codec_sample_count": applied,
            "bypass_sample_count": bypassed,
            "effective_application_probability": (
                None if total == 0 else applied / total
            ),
            "contract_hash": self._selection_contract_hash,
        }

    def __call__(self, sample: Mapping[str, Any]) -> Any:
        if self.application_probability != 1.0:
            raise OpenPIImageWireError(
                "Probabilistic image-wire training requires stable sample identity."
            )
        return self._apply(sample, apply_codec=True)

    def set_epoch(self, epoch: int) -> None:
        object.__setattr__(self, "_epoch", max(0, int(epoch)))
        setter = getattr(self.upstream, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))


def build_openpi_image_wire_capability(
    config: OpenPIImageWireServerConfig,
    *,
    image_keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(config, OpenPIImageWireServerConfig):
        raise TypeError("config must be an OpenPIImageWireServerConfig.")
    if not config.enabled:
        raise OpenPIImageWireError(
            "The off profile does not advertise an image-wire capability."
        )
    keys = tuple(str(value).strip() for value in image_keys)
    if not keys or any(not value for value in keys) or len(set(keys)) != len(keys):
        raise OpenPIImageWireError(
            "Server image-wire capability requires unique, non-empty image_keys."
        )
    payload: dict[str, Any] = {
        "schema": OPENPI_IMAGE_WIRE_CAPABILITY_SCHEMA_V1,
        "wire_schema": OPENPI_IMAGE_WIRE_ENVELOPE_SCHEMA_V1,
        "supported_codecs": list(config.supported_codecs),
        "legacy_raw_supported": True,
        "target_shape": [config.target_height, config.target_width, 3],
        "dtype": OPENPI_IMAGE_WIRE_DTYPE_UINT8,
        "color_space": OPENPI_IMAGE_WIRE_COLOR_SPACE_RGB,
        "resize_method": OPENPI_IMAGE_WIRE_RESIZE_METHOD_V1,
        "image_keys": list(keys),
        "jpeg": {
            "quality": config.jpeg_quality,
            "subsampling": "4:2:0",
            "encoder": OPENPI_IMAGE_WIRE_JPEG_ENCODER_V1,
        },
    }
    payload["capability_hash"] = canonical_content_hash(
        payload,
        domain=OPENPI_IMAGE_WIRE_CAPABILITY_HASH_DOMAIN_V1,
    )
    return payload


def validate_openpi_image_wire_capability(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenPIImageWireError("OpenPI image-wire capability must be a mapping.")
    required = {
        "schema",
        "wire_schema",
        "supported_codecs",
        "legacy_raw_supported",
        "target_shape",
        "dtype",
        "color_space",
        "resize_method",
        "image_keys",
        "jpeg",
        "capability_hash",
    }
    if set(value) != required:
        raise OpenPIImageWireError(
            "OpenPI image-wire capability fields differ from the v1 contract."
        )
    payload = {str(key): item for key, item in value.items()}
    if payload["schema"] != OPENPI_IMAGE_WIRE_CAPABILITY_SCHEMA_V1:
        raise OpenPIImageWireError("Unsupported OpenPI image-wire capability schema.")
    if payload["wire_schema"] != OPENPI_IMAGE_WIRE_ENVELOPE_SCHEMA_V1:
        raise OpenPIImageWireError("Unsupported OpenPI image-wire envelope schema.")
    codecs_value = payload["supported_codecs"]
    if not isinstance(codecs_value, Sequence) or isinstance(
        codecs_value, (str, bytes)
    ):
        raise OpenPIImageWireError("supported_codecs must be a sequence.")
    codecs = tuple(str(value) for value in codecs_value)
    if not codecs or len(set(codecs)) != len(codecs):
        raise OpenPIImageWireError("supported_codecs must be unique and non-empty.")
    if any(codec not in OPENPI_IMAGE_WIRE_CODECS_V1 for codec in codecs):
        raise OpenPIImageWireError("Server advertised an unsupported image-wire codec.")
    if payload["legacy_raw_supported"] is not True:
        raise OpenPIImageWireError("Image-wire v1 requires legacy raw rollback support.")
    shape_value = payload["target_shape"]
    if not isinstance(shape_value, Sequence) or isinstance(shape_value, (str, bytes)):
        raise OpenPIImageWireError("target_shape must be a sequence.")
    shape = tuple(int(value) for value in shape_value)
    if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != 3:
        raise OpenPIImageWireError("target_shape must be [positive, positive, 3].")
    if payload["dtype"] != OPENPI_IMAGE_WIRE_DTYPE_UINT8:
        raise OpenPIImageWireError("Image-wire v1 requires uint8 images.")
    if payload["color_space"] != OPENPI_IMAGE_WIRE_COLOR_SPACE_RGB:
        raise OpenPIImageWireError("Image-wire v1 requires RGB images.")
    if payload["resize_method"] != OPENPI_IMAGE_WIRE_RESIZE_METHOD_V1:
        raise OpenPIImageWireError("Unsupported image-wire resize method.")
    keys_value = payload["image_keys"]
    if not isinstance(keys_value, Sequence) or isinstance(keys_value, (str, bytes)):
        raise OpenPIImageWireError("image_keys must be a sequence.")
    keys = tuple(str(value) for value in keys_value)
    if not keys or any(not value for value in keys) or len(set(keys)) != len(keys):
        raise OpenPIImageWireError("image_keys must be unique and non-empty.")
    jpeg = payload["jpeg"]
    if not isinstance(jpeg, Mapping) or set(jpeg) != {
        "quality",
        "subsampling",
        "encoder",
    }:
        raise OpenPIImageWireError("jpeg capability parameters are invalid.")
    quality = int(jpeg["quality"])
    if not 1 <= quality <= 100:
        raise OpenPIImageWireError("jpeg quality must be within [1, 100].")
    if jpeg["subsampling"] != "4:2:0" or jpeg["encoder"] != (
        OPENPI_IMAGE_WIRE_JPEG_ENCODER_V1
    ):
        raise OpenPIImageWireError("Unsupported JPEG image-wire parameters.")
    expected_hash = canonical_content_hash(
        {key: item for key, item in payload.items() if key != "capability_hash"},
        domain=OPENPI_IMAGE_WIRE_CAPABILITY_HASH_DOMAIN_V1,
    )
    if str(payload["capability_hash"]) != expected_hash:
        raise OpenPIImageWireError("OpenPI image-wire capability hash mismatch.")
    return {
        **payload,
        "supported_codecs": list(codecs),
        "target_shape": list(shape),
        "image_keys": list(keys),
        "jpeg": {
            "quality": quality,
            "subsampling": "4:2:0",
            "encoder": OPENPI_IMAGE_WIRE_JPEG_ENCODER_V1,
        },
        "capability_hash": expected_hash,
    }


def resize_openpi_image_with_pad_v1(
    image: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Match ``openpi_client.image_tools.resize_with_pad`` for one RGB image."""

    array = _validate_rgb8_image(image)
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise OpenPIImageWireError("Resize height and width must be positive.")
    if array.shape[:2] == (height, width):
        return array
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Optimized OpenPI image wire requires Pillow."
        ) from exc
    source = Image.fromarray(array)
    current_width, current_height = source.size
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = source.resize((resized_width, resized_height), resample=Image.BILINEAR)
    output = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    output.paste(resized, (pad_width, pad_height))
    result = np.asarray(output, dtype=np.uint8)
    return np.ascontiguousarray(result)


def encode_openpi_image_wire_request(
    request: Mapping[str, Any],
    config: OpenPIImageWireClientConfig,
    server_metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], OpenPIImageWireEncodeReceipt | None]:
    """Encode one legacy or conditioned OpenPI request without mutating it."""

    if not isinstance(request, Mapping):
        raise TypeError("OpenPI image-wire request must be a mapping.")
    if not isinstance(config, OpenPIImageWireClientConfig):
        raise TypeError("config must be an OpenPIImageWireClientConfig.")
    if not config.enabled:
        return request, None

    started = time.perf_counter()
    capability_value = server_metadata.get(OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1)
    if capability_value is None:
        return _legacy_fallback_or_raise(
            request,
            config,
            reason="server_capability_absent",
            started=started,
        )
    capability = validate_openpi_image_wire_capability(capability_value)
    codec = _negotiate_codec(config, capability)
    if codec == OPENPI_IMAGE_WIRE_LEGACY_RAW_V1:
        return _legacy_fallback_or_raise(
            request,
            config,
            reason="preferred_codec_unsupported",
            started=started,
            capability_hash=str(capability["capability_hash"]),
        )
    server_keys = tuple(str(key) for key in capability["image_keys"])
    if any(key not in server_keys for key in config.image_keys):
        raise OpenPIImageWireError(
            "Client image keys are not a subset of the server capability."
        )

    root, observation = _copy_request_observation(request)
    target_shape = tuple(int(value) for value in capability["target_shape"])
    source_bytes = 0
    encoded_bytes = 0
    for key in config.image_keys:
        if key not in observation:
            raise OpenPIImageWireError(
                f"OpenPI request is missing configured image key {key!r}."
            )
        image = _validate_rgb8_image(observation[key])
        source_bytes += int(image.nbytes)
        resized = resize_openpi_image_with_pad_v1(
            image,
            target_shape[0],
            target_shape[1],
        )
        envelope = _encode_image_envelope(resized, codec, capability)
        encoded_bytes += len(envelope["data"])
        observation[key] = envelope

    return root, OpenPIImageWireEncodeReceipt(
        requested_codec=config.preferred_codec,
        effective_codec=codec,
        fallback_reason=None,
        capability_hash=str(capability["capability_hash"]),
        image_count=len(config.image_keys),
        source_image_bytes=source_bytes,
        encoded_image_bytes=encoded_bytes,
        encode_ms=(time.perf_counter() - started) * 1000.0,
    )


def decode_openpi_image_wire_request(
    request: Mapping[str, Any],
    capability_value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], OpenPIImageWireDecodeReceipt | None]:
    """Decode advertised envelopes; legacy ndarray requests pass through exactly."""

    capability = validate_openpi_image_wire_capability(capability_value)
    root, observation = _copy_request_observation(request)
    codecs: list[str] = []
    encoded_bytes = 0
    decoded_bytes = 0
    for key in tuple(str(value) for value in capability["image_keys"]):
        value = observation.get(key)
        if not _is_image_wire_envelope(value):
            continue
        image, codec, payload_size = _decode_image_envelope(value, capability)
        observation[key] = image
        codecs.append(codec)
        encoded_bytes += payload_size
        decoded_bytes += int(image.nbytes)
    if not codecs:
        return request, None
    return root, OpenPIImageWireDecodeReceipt(
        codecs=tuple(codecs),
        image_count=len(codecs),
        encoded_image_bytes=encoded_bytes,
        decoded_image_bytes=decoded_bytes,
    )


def simulate_openpi_image_wire_roundtrip(
    image: np.ndarray,
    *,
    codec: str = OPENPI_IMAGE_WIRE_JPEG_V1,
    target_height: int = 224,
    target_width: int = 224,
    jpeg_quality: int = 90,
) -> np.ndarray:
    """Apply the exact deploy codec for deterministic training alignment."""

    profile = "jpeg" if codec == OPENPI_IMAGE_WIRE_JPEG_V1 else "resize-raw"
    if codec not in OPENPI_IMAGE_WIRE_CODECS_V1:
        raise OpenPIImageWireError(f"Unsupported training wire codec: {codec!r}.")
    server_config = OpenPIImageWireServerConfig(
        profile=profile,
        target_height=target_height,
        target_width=target_width,
        jpeg_quality=jpeg_quality,
    )
    capability = build_openpi_image_wire_capability(
        server_config,
        image_keys=("image",),
    )
    client_config = OpenPIImageWireClientConfig(
        preferred_codec=codec,
        allow_legacy_fallback=False,
        image_keys=("image",),
    )
    encoded, _ = encode_openpi_image_wire_request(
        {"image": image},
        client_config,
        {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability},
    )
    decoded, _ = decode_openpi_image_wire_request(encoded, capability)
    return np.asarray(decoded["image"], dtype=np.uint8)


@contextmanager
def install_openpi_image_wire_server_adapter(
    websocket_policy_server_module: Any,
    config: OpenPIImageWireServerConfig,
) -> Iterator[None]:
    """Temporarily wrap OpenPI's server class inside one prepared process."""

    if not isinstance(config, OpenPIImageWireServerConfig):
        raise TypeError("config must be an OpenPIImageWireServerConfig.")
    if not config.enabled:
        yield
        return
    original = websocket_policy_server_module.WebsocketPolicyServer

    class ImageWireWebsocketPolicyServer(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            image_keys = _server_image_keys(self._metadata)
            capability = build_openpi_image_wire_capability(
                config,
                image_keys=image_keys,
            )
            existing = self._metadata.get(OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1)
            if existing is not None and existing != capability:
                raise OpenPIImageWireError(
                    "OpenPI server already advertises a different image-wire capability."
                )
            self._metadata[OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1] = capability
            self._dexteleop_image_wire_capability = capability
            self.last_image_wire_decode_receipt = None

        def _infer(self, request: Any) -> Any:
            decoded, receipt = decode_openpi_image_wire_request(
                request,
                self._dexteleop_image_wire_capability,
            )
            self.last_image_wire_decode_receipt = receipt
            return super()._infer(decoded)

    ImageWireWebsocketPolicyServer.__name__ = original.__name__
    ImageWireWebsocketPolicyServer.__qualname__ = original.__qualname__
    websocket_policy_server_module.WebsocketPolicyServer = (
        ImageWireWebsocketPolicyServer
    )
    try:
        yield
    finally:
        if (
            websocket_policy_server_module.WebsocketPolicyServer
            is ImageWireWebsocketPolicyServer
        ):
            websocket_policy_server_module.WebsocketPolicyServer = original


def _legacy_fallback_or_raise(
    request: Mapping[str, Any],
    config: OpenPIImageWireClientConfig,
    *,
    reason: str,
    started: float,
    capability_hash: str | None = None,
) -> tuple[Mapping[str, Any], OpenPIImageWireEncodeReceipt]:
    if not config.allow_legacy_fallback:
        raise OpenPIImageWireError(
            f"Requested image-wire codec cannot be used: {reason}."
        )
    return request, OpenPIImageWireEncodeReceipt(
        requested_codec=config.preferred_codec,
        effective_codec=OPENPI_IMAGE_WIRE_LEGACY_RAW_V1,
        fallback_reason=reason,
        capability_hash=capability_hash,
        image_count=0,
        source_image_bytes=0,
        encoded_image_bytes=0,
        encode_ms=(time.perf_counter() - started) * 1000.0,
    )


def _negotiate_codec(
    config: OpenPIImageWireClientConfig,
    capability: Mapping[str, Any],
) -> str:
    supported = set(str(value) for value in capability["supported_codecs"])
    candidates = (config.preferred_codec,)
    if config.preferred_codec == OPENPI_IMAGE_WIRE_JPEG_V1:
        candidates = (
            OPENPI_IMAGE_WIRE_JPEG_V1,
            OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,
        )
    for codec in candidates:
        if codec in supported:
            return codec
    return OPENPI_IMAGE_WIRE_LEGACY_RAW_V1


def _copy_request_observation(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = dict(request)
    if "__openpi_request__" in root:
        observation_value = root.get("observation")
        if not isinstance(observation_value, Mapping):
            raise OpenPIImageWireError(
                "Conditioned OpenPI request observation must be a mapping."
            )
        observation = dict(observation_value)
        root["observation"] = observation
        return root, observation
    return root, root


def _validate_rgb8_image(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise OpenPIImageWireError("OpenPI image must be a NumPy array.")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise OpenPIImageWireError(
            "OpenPI image-wire v1 requires RGB HWC uint8 images."
        )
    return np.ascontiguousarray(value)


def _encode_image_envelope(
    image: np.ndarray,
    codec: str,
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    if codec == OPENPI_IMAGE_WIRE_RESIZE_RAW_V1:
        data = image.tobytes(order="C")
        parameters: dict[str, Any] = {}
    elif codec == OPENPI_IMAGE_WIRE_JPEG_V1:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "JPEG OpenPI image wire requires Pillow."
            ) from exc
        quality = int(capability["jpeg"]["quality"])
        buffer = BytesIO()
        Image.fromarray(image).save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        data = buffer.getvalue()
        parameters = {
            "quality": quality,
            "subsampling": "4:2:0",
            "encoder": OPENPI_IMAGE_WIRE_JPEG_ENCODER_V1,
        }
    else:
        raise OpenPIImageWireError(f"Unsupported negotiated codec: {codec!r}.")
    return {
        _IMAGE_WIRE_MARKER: OPENPI_IMAGE_WIRE_ENVELOPE_SCHEMA_V1,
        "codec": codec,
        "capability_hash": str(capability["capability_hash"]),
        "shape": list(image.shape),
        "dtype": OPENPI_IMAGE_WIRE_DTYPE_UINT8,
        "color_space": OPENPI_IMAGE_WIRE_COLOR_SPACE_RGB,
        "resize_method": OPENPI_IMAGE_WIRE_RESIZE_METHOD_V1,
        "parameters": parameters,
        "data": data,
    }


def _is_image_wire_envelope(value: Any) -> bool:
    return isinstance(value, Mapping) and _IMAGE_WIRE_MARKER in value


def _decode_image_envelope(
    value: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> tuple[np.ndarray, str, int]:
    required = {
        _IMAGE_WIRE_MARKER,
        "codec",
        "capability_hash",
        "shape",
        "dtype",
        "color_space",
        "resize_method",
        "parameters",
        "data",
    }
    if set(value) != required:
        raise OpenPIImageWireError("Image-wire envelope fields differ from v1.")
    if value[_IMAGE_WIRE_MARKER] != OPENPI_IMAGE_WIRE_ENVELOPE_SCHEMA_V1:
        raise OpenPIImageWireError("Unsupported image-wire envelope schema.")
    if str(value["capability_hash"]) != str(capability["capability_hash"]):
        raise OpenPIImageWireError("Image-wire envelope capability hash mismatch.")
    codec = str(value["codec"])
    if codec not in tuple(str(item) for item in capability["supported_codecs"]):
        raise OpenPIImageWireError("Envelope codec is not advertised by this server.")
    shape = tuple(int(item) for item in value["shape"])
    expected_shape = tuple(int(item) for item in capability["target_shape"])
    if shape != expected_shape:
        raise OpenPIImageWireError("Image-wire envelope shape mismatch.")
    if value["dtype"] != OPENPI_IMAGE_WIRE_DTYPE_UINT8:
        raise OpenPIImageWireError("Image-wire envelope dtype mismatch.")
    if value["color_space"] != OPENPI_IMAGE_WIRE_COLOR_SPACE_RGB:
        raise OpenPIImageWireError("Image-wire envelope color-space mismatch.")
    if value["resize_method"] != OPENPI_IMAGE_WIRE_RESIZE_METHOD_V1:
        raise OpenPIImageWireError("Image-wire envelope resize-method mismatch.")
    data = value["data"]
    if not isinstance(data, bytes):
        raise OpenPIImageWireError("Image-wire envelope data must be bytes.")
    raw_size = int(np.prod(expected_shape, dtype=np.int64))
    if len(data) > max(1_000_000, raw_size * 2):
        raise OpenPIImageWireError("Image-wire envelope exceeds the v1 size bound.")
    parameters = value["parameters"]
    if not isinstance(parameters, Mapping):
        raise OpenPIImageWireError("Image-wire envelope parameters must be a mapping.")
    if codec == OPENPI_IMAGE_WIRE_RESIZE_RAW_V1:
        if parameters or len(data) != raw_size:
            raise OpenPIImageWireError("Resized-raw image-wire payload is invalid.")
        image = np.frombuffer(data, dtype=np.uint8).reshape(expected_shape).copy()
    elif codec == OPENPI_IMAGE_WIRE_JPEG_V1:
        if dict(parameters) != dict(capability["jpeg"]):
            raise OpenPIImageWireError("JPEG envelope parameters mismatch.")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "JPEG OpenPI image wire requires Pillow."
            ) from exc
        with Image.open(BytesIO(data)) as encoded:
            if encoded.format != "JPEG":
                raise OpenPIImageWireError("JPEG envelope does not contain JPEG data.")
            decoded = encoded.convert("RGB")
            if decoded.size != (expected_shape[1], expected_shape[0]):
                raise OpenPIImageWireError("Decoded JPEG dimensions mismatch.")
            image = np.asarray(decoded, dtype=np.uint8).copy()
    else:  # pragma: no cover - capability validation guards this branch.
        raise OpenPIImageWireError(f"Unsupported image-wire codec: {codec!r}.")
    _validate_rgb8_image(image)
    return image, codec, len(data)


def _server_image_keys(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    dataset_adapter = metadata.get("dataset_adapter")
    if not isinstance(dataset_adapter, Mapping):
        raise OpenPIImageWireError(
            "Image-wire server metadata lacks dataset_adapter."
        )
    values = dataset_adapter.get("camera_slots")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise OpenPIImageWireError(
            "Image-wire server metadata lacks dataset_adapter.camera_slots."
        )
    keys = tuple(str(value).strip() for value in values)
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise OpenPIImageWireError(
            "Image-wire server camera slots must be unique and non-empty."
        )
    return keys


__all__ = [
    "OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1",
    "OPENPI_IMAGE_WIRE_CAPABILITY_SCHEMA_V1",
    "OPENPI_IMAGE_WIRE_ENVELOPE_SCHEMA_V1",
    "OPENPI_IMAGE_WIRE_JPEG_V1",
    "OPENPI_IMAGE_WIRE_LEGACY_RAW_V1",
    "OPENPI_IMAGE_WIRE_RESIZE_RAW_V1",
    "OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1",
    "OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1",
    "OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2",
    "OpenPIImageWireClientConfig",
    "OpenPIImageWireDecodeReceipt",
    "OpenPIImageWireEncodeReceipt",
    "OpenPIImageWireError",
    "OpenPIImageWireServerConfig",
    "OpenPIImageWireTrainingTransform",
    "build_openpi_image_wire_capability",
    "decode_openpi_image_wire_request",
    "encode_openpi_image_wire_request",
    "install_openpi_image_wire_server_adapter",
    "resize_openpi_image_with_pad_v1",
    "simulate_openpi_image_wire_roundtrip",
    "validate_openpi_image_wire_capability",
]
