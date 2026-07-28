"""Injected model-RTC conditioning codec and exact encoding receipt.

`robot_motion` owns the immutable canonical bundle.  This module validates an
artifact-bound provider codec, projects only the selected canonical rows into
the provider's external action representation, and records what was encoded.
The provider may subsequently transform that prefix into its sampler's
internal representation.  This module owns no plan, worker, sampler, target,
safety, or dispatch lifecycle.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    normalize_model_continuation_contract,
)
from robot_motion.deploy.canonical_policy_action import CanonicalPolicyAction
from robot_motion.deploy.rtc.conditioning import (
    CanonicalRtcConditioningBundle,
)


RTC_CONDITIONING_CODEC_IDENTITY_SCHEMA_V1 = (
    "rtc_conditioning_codec_identity_v1"
)
RTC_CONDITIONING_CODEC_IDENTITY_HASH_DOMAIN_V1 = (
    "dexteleop.deployments.rtc_conditioning_codec_identity.v1"
)
RTC_ARTIFACT_ACTION_ROWS_HASH_DOMAIN_V1 = (
    "dexteleop.deployments.rtc_artifact_action_rows.v1"
)
RTC_NUMERIC_ARRAY_HASH_SCHEMA_V1 = "dexteleop_rtc_numeric_array_v1"
RTC_CONDITIONING_ENCODING_RECEIPT_SCHEMA_V1 = (
    "rtc_conditioning_encoding_receipt_v1"
)
RTC_CONDITIONING_ENCODING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.deployments.rtc_conditioning_encoding_receipt.v1"
)
RTC_ENCODED_CONDITIONING_SCHEMA_V1 = "rtc_encoded_conditioning_v1"


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return normalized


def _finite_matrix(
    values: object,
    *,
    name: str,
    columns: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape [rows, {columns}].")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    frozen = np.ascontiguousarray(array).copy()
    frozen.setflags(write=False)
    return frozen


def _rtc_array_hash(*, dtype_tag: str, shape: tuple[int, ...], data: bytes) -> str:
    """Portable hash used by both DexTeleop and external model providers."""

    shape_text = ",".join(str(int(dimension)) for dimension in shape)
    payload = (
        RTC_NUMERIC_ARRAY_HASH_SCHEMA_V1.encode("ascii")
        + b"\0"
        + dtype_tag.encode("ascii")
        + b"\0"
        + shape_text.encode("ascii")
        + b"\0"
        + data
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def rtc_float32_array_hash(values: object) -> str:
    """Hash a finite array after canonical little-endian float32 encoding."""

    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("RTC float32 hash input must contain finite values.")
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return _rtc_array_hash(
        dtype_tag="float32_le",
        shape=canonical.shape,
        data=canonical.tobytes(order="C"),
    )


def rtc_binary_mask_hash(values: object) -> str:
    """Hash a one-dimensional binary mask after canonical uint8 encoding."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("RTC binary mask hash input must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError("RTC binary mask hash input must be finite.")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("RTC binary mask hash input must contain only 0/1.")
    canonical = np.ascontiguousarray(array, dtype=np.uint8)
    return _rtc_array_hash(
        dtype_tag="uint8_binary",
        shape=canonical.shape,
        data=canonical.tobytes(order="C"),
    )


@dataclass(frozen=True)
class RtcConditioningCodecIdentity:
    """Loaded-instance identity for the canonical/external codec."""

    implementation_id: str
    implementation_parameters_hash: str
    action_codec_hash: str
    layout_hash: str
    frame_contract_hash: str
    normalization_hash: str
    artifact_action_dim: int
    external_action_dim: int
    round_trip_atol: float = 1e-6
    round_trip_rtol: float = 1e-6
    schema: str = RTC_CONDITIONING_CODEC_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != RTC_CONDITIONING_CODEC_IDENTITY_SCHEMA_V1:
            raise ValueError(
                f"Unsupported RTC conditioning codec schema {self.schema!r}."
            )
        for name in (
            "implementation_id",
            "implementation_parameters_hash",
            "action_codec_hash",
            "layout_hash",
            "frame_contract_hash",
            "normalization_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "artifact_action_dim",
            _positive_int(
                self.artifact_action_dim,
                name="artifact_action_dim",
            ),
        )
        object.__setattr__(
            self,
            "external_action_dim",
            _positive_int(
                self.external_action_dim,
                name="external_action_dim",
            ),
        )
        object.__setattr__(
            self,
            "round_trip_atol",
            _nonnegative_finite(
                self.round_trip_atol,
                name="round_trip_atol",
            ),
        )
        object.__setattr__(
            self,
            "round_trip_rtol",
            _nonnegative_finite(
                self.round_trip_rtol,
                name="round_trip_rtol",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "implementation_id": self.implementation_id,
                "implementation_parameters_hash": (
                    self.implementation_parameters_hash
                ),
                "action_codec_hash": self.action_codec_hash,
                "layout_hash": self.layout_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "normalization_hash": self.normalization_hash,
                "artifact_action_dim": self.artifact_action_dim,
                "external_action_dim": self.external_action_dim,
                "round_trip_atol": self.round_trip_atol,
                "round_trip_rtol": self.round_trip_rtol,
            },
            domain=RTC_CONDITIONING_CODEC_IDENTITY_HASH_DOMAIN_V1,
        )


@runtime_checkable
class RtcConditioningCodec(Protocol):
    """Provider-supplied canonical/artifact/external conversion."""

    @property
    def identity(self) -> RtcConditioningCodecIdentity:
        ...

    def canonical_to_artifact_actions(
        self,
        actions: Sequence[CanonicalPolicyAction],
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        ...

    def artifact_to_canonical_actions(
        self,
        artifact_actions: np.ndarray,
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> Sequence[CanonicalPolicyAction]:
        ...

    def artifact_to_external_actions(
        self,
        artifact_actions: np.ndarray,
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        ...

    def external_to_artifact_actions(
        self,
        external_actions: np.ndarray,
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        ...


def _validate_codec_binding(
    codec: RtcConditioningCodec,
    bundle: CanonicalRtcConditioningBundle,
) -> RtcConditioningCodecIdentity:
    if not isinstance(bundle, CanonicalRtcConditioningBundle):
        raise TypeError("bundle must be a CanonicalRtcConditioningBundle.")
    if not isinstance(codec, RtcConditioningCodec):
        raise TypeError("codec must implement RtcConditioningCodec.")
    identity = codec.identity
    if not isinstance(identity, RtcConditioningCodecIdentity):
        raise TypeError(
            "codec.identity must be an RtcConditioningCodecIdentity."
        )
    contract = normalize_model_continuation_contract(
        bundle.continuation_contract
    ).runtime
    expected = {
        "action_codec_hash": contract.action_codec_hash,
        "layout_hash": contract.layout_hash,
        "frame_contract_hash": contract.frame_contract_hash,
        "normalization_hash": contract.normalization_hash,
        "external_action_dim": contract.model_action_dim,
    }
    for name, expected_value in expected.items():
        if getattr(identity, name) != expected_value:
            raise ValueError(f"Conditioning codec {name} mismatch.")
    return identity


def encode_external_continuation_artifact_actions(
    codec: RtcConditioningCodec,
    artifact_actions: np.ndarray,
    *,
    bundle: CanonicalRtcConditioningBundle,
) -> np.ndarray:
    """Use the same numerical codec path for offline and online action rows."""

    identity = _validate_codec_binding(codec, bundle)
    artifact = _finite_matrix(
        artifact_actions,
        name="artifact_actions",
        columns=identity.artifact_action_dim,
    )
    external = _finite_matrix(
        codec.artifact_to_external_actions(artifact, bundle=bundle),
        name="external_actions",
        columns=identity.external_action_dim,
    )
    if external.shape[0] != artifact.shape[0]:
        raise ValueError(
            "Conditioning codec must preserve the action-row count."
        )
    return external


@dataclass(frozen=True)
class RtcConditioningEncodingReceipt:
    conditioning_binding_hash: str
    codec_identity_hash: str
    selected_generation_indices: tuple[int, ...]
    canonical_action_content_hash: str
    artifact_action_rows_hash: str
    external_prefix_hash: str
    prefix_mask_hash: str
    decoded_canonical_action_content_hash: str
    round_trip_max_abs_error: float
    schema: str = RTC_CONDITIONING_ENCODING_RECEIPT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != RTC_CONDITIONING_ENCODING_RECEIPT_SCHEMA_V1:
            raise ValueError(
                f"Unsupported conditioning encoding receipt {self.schema!r}."
            )
        for name in (
            "conditioning_binding_hash",
            "codec_identity_hash",
            "canonical_action_content_hash",
            "artifact_action_rows_hash",
            "external_prefix_hash",
            "prefix_mask_hash",
            "decoded_canonical_action_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in self.selected_generation_indices
        ):
            raise ValueError(
                "selected_generation_indices must contain exact integers."
            )
        indices = tuple(
            int(index) for index in self.selected_generation_indices
        )
        if any(index < 0 for index in indices) or any(
            right <= left for left, right in zip(indices, indices[1:])
        ):
            raise ValueError(
                "selected_generation_indices must be strictly increasing."
            )
        object.__setattr__(self, "selected_generation_indices", indices)
        object.__setattr__(
            self,
            "round_trip_max_abs_error",
            _nonnegative_finite(
                self.round_trip_max_abs_error,
                name="round_trip_max_abs_error",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "conditioning_binding_hash": self.conditioning_binding_hash,
                "codec_identity_hash": self.codec_identity_hash,
                "selected_generation_indices": (
                    self.selected_generation_indices
                ),
                "canonical_action_content_hash": (
                    self.canonical_action_content_hash
                ),
                "artifact_action_rows_hash": (
                    self.artifact_action_rows_hash
                ),
                "external_prefix_hash": self.external_prefix_hash,
                "prefix_mask_hash": self.prefix_mask_hash,
                "decoded_canonical_action_content_hash": (
                    self.decoded_canonical_action_content_hash
                ),
                "round_trip_max_abs_error": (
                    self.round_trip_max_abs_error
                ),
            },
            domain=RTC_CONDITIONING_ENCODING_RECEIPT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RtcEncodedConditioning:
    """Full-horizon external prefix plus its exact encoding receipt."""

    external_prefix: np.ndarray
    prefix_mask: np.ndarray
    receipt: RtcConditioningEncodingReceipt
    schema: str = RTC_ENCODED_CONDITIONING_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != RTC_ENCODED_CONDITIONING_SCHEMA_V1:
            raise ValueError(
                f"Unsupported encoded conditioning schema {self.schema!r}."
            )
        if not isinstance(self.receipt, RtcConditioningEncodingReceipt):
            raise TypeError(
                "receipt must be an RtcConditioningEncodingReceipt."
            )
        prefix = np.asarray(self.external_prefix, dtype=np.float32)
        mask = np.asarray(self.prefix_mask, dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[0] < 2 or prefix.shape[1] < 1:
            raise ValueError(
                "external_prefix must be [prediction_horizon, action_dim]."
            )
        if mask.shape != (prefix.shape[0],):
            raise ValueError(
                "prefix_mask must cover the complete prediction horizon."
            )
        if not np.all(np.isfinite(prefix)) or not np.all(np.isfinite(mask)):
            raise ValueError("Encoded conditioning values must be finite.")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("prefix_mask must contain only 0.0/1.0.")
        if np.any(prefix[mask == 0.0] != 0.0):
            raise ValueError(
                "Unselected external conditioning rows must be zero-filled."
            )
        prefix = np.ascontiguousarray(prefix).copy()
        mask = np.ascontiguousarray(mask).copy()
        prefix.setflags(write=False)
        mask.setflags(write=False)
        if self.receipt.external_prefix_hash != rtc_float32_array_hash(prefix):
            raise ValueError("Encoded conditioning rows/receipt hash mismatch.")
        if self.receipt.prefix_mask_hash != rtc_binary_mask_hash(mask):
            raise ValueError("Encoded conditioning mask/receipt hash mismatch.")
        object.__setattr__(self, "external_prefix", prefix)
        object.__setattr__(self, "prefix_mask", mask)


def encode_rtc_conditioning_bundle(
    codec: RtcConditioningCodec,
    bundle: CanonicalRtcConditioningBundle,
) -> RtcEncodedConditioning:
    """Encode one canonical bundle and prove bidirectional codec parity."""

    identity = _validate_codec_binding(codec, bundle)
    actions = bundle.canonical_actions
    artifact = _finite_matrix(
        codec.canonical_to_artifact_actions(actions, bundle=bundle),
        name="artifact_actions",
        columns=identity.artifact_action_dim,
    )
    if artifact.shape[0] != len(actions):
        raise ValueError(
            "Canonical-to-artifact codec must preserve selected row count."
        )
    external_selected = encode_external_continuation_artifact_actions(
        codec,
        artifact,
        bundle=bundle,
    )
    artifact_round_trip = _finite_matrix(
        codec.external_to_artifact_actions(
            external_selected,
            bundle=bundle,
        ),
        name="round_trip_artifact_actions",
        columns=identity.artifact_action_dim,
    )
    if artifact_round_trip.shape != artifact.shape:
        raise ValueError("Conditioning codec round-trip shape mismatch.")
    if artifact.size:
        absolute_error = np.abs(artifact_round_trip - artifact)
        max_abs_error = float(np.max(absolute_error))
    else:
        max_abs_error = 0.0
    if not np.allclose(
        artifact_round_trip,
        artifact,
        rtol=identity.round_trip_rtol,
        atol=identity.round_trip_atol,
    ):
        raise ValueError(
            "Conditioning codec artifact/external round-trip exceeds tolerance."
        )
    decoded = tuple(
        codec.artifact_to_canonical_actions(
            artifact_round_trip,
            bundle=bundle,
        )
    )
    if any(not isinstance(action, CanonicalPolicyAction) for action in decoded):
        raise TypeError(
            "artifact_to_canonical_actions must return canonical actions."
        )
    source_action_hashes = tuple(
        action.canonical_action_hash for action in actions
    )
    decoded_action_hashes = tuple(
        action.canonical_action_hash for action in decoded
    )
    if decoded_action_hashes != source_action_hashes:
        raise ValueError(
            "Conditioning codec canonical round-trip changed action content."
        )

    horizon = bundle.timeline.prediction_horizon_steps
    full_external_prefix = np.zeros(
        (horizon, identity.external_action_dim),
        dtype=np.float32,
    )
    for row, generation_index in enumerate(
        bundle.selected_generation_indices
    ):
        full_external_prefix[generation_index] = external_selected[row]
    prefix_mask = np.asarray(bundle.prefix_mask, dtype=np.float32)
    artifact_hash = canonical_content_hash(
        artifact,
        domain=RTC_ARTIFACT_ACTION_ROWS_HASH_DOMAIN_V1,
    )
    external_hash = rtc_float32_array_hash(full_external_prefix)
    mask_hash = rtc_binary_mask_hash(prefix_mask)
    decoded_content_hash = canonical_content_hash(
        decoded_action_hashes,
        domain="dexteleop.deploy.rtc_canonical_action_content.v1",
    )
    receipt = RtcConditioningEncodingReceipt(
        conditioning_binding_hash=bundle.conditioning_binding_hash,
        codec_identity_hash=identity.semantic_content_hash,
        selected_generation_indices=bundle.selected_generation_indices,
        canonical_action_content_hash=bundle.canonical_action_content_hash,
        artifact_action_rows_hash=artifact_hash,
        external_prefix_hash=external_hash,
        prefix_mask_hash=mask_hash,
        decoded_canonical_action_content_hash=decoded_content_hash,
        round_trip_max_abs_error=max_abs_error,
    )
    if (
        receipt.decoded_canonical_action_content_hash
        != receipt.canonical_action_content_hash
    ):
        raise ValueError(
            "Decoded canonical content receipt differs from the source bundle."
        )
    return RtcEncodedConditioning(
        external_prefix=full_external_prefix,
        prefix_mask=prefix_mask,
        receipt=receipt,
    )
