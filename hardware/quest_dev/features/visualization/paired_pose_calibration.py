"""Replayable Quest/controller-to-robot paired-pose calibration.

This R0 feature-local solver estimates the two fixed transforms in

``T_Bref_Bmoving(i) = T_Bref_Aref @ T_Aref_Amoving(i) @ T_Amoving_Bmoving``.

For a controller mounted to a robot TCP, A is the robot frame family and B is
the Quest frame family. A fit is diagnostic evidence only; it is not an
acceptance receipt and must not be installed into an execution frame graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .transforms import (
    matrix_to_column_major,
    matrix_to_quaternion_xyzw,
    rigid_inverse,
    validate_rigid_transform,
)

DATASET_SCHEMA = "quest_dev.visualization.paired_pose_dataset.v1"
FIT_SCHEMA = "quest_dev.visualization.paired_pose_fit.v1"
ALGORITHM_ID = "ax_xb_numpy_svd.v1"
FIT_ACCEPTANCE_NOTE = "fit only; a reviewed acceptance profile is required"
PAIRING_METHODS = {"operator_static_window", "synthetic_replay"}
MIN_EFFECTIVE_RELATIVE_ROTATION_DEG = 5.0
MIN_ROTATION_AXIS_SEPARATION_DEG = 15.0
MIN_ROTATION_CONSTRAINT_RATIO = 1e-2
MAX_ROTATION_NULLSPACE_RATIO = 1e-3
MIN_ROTATION_NULLSPACE_GAP = 10.0
MIN_TRANSLATION_SINGULAR_VALUE_RATIO = 2e-2

_FIT_FRAME_KEYS = {
    "first_reference",
    "first_moving",
    "second_reference",
    "second_moving",
}
_FIT_OBSERVABILITY_INTEGER_KEYS = {
    "relative_pair_count",
    "effective_relative_rotation_count",
    "translation_system_rank",
}
_FIT_OBSERVABILITY_FLOAT_KEYS = {
    "max_relative_rotation_deg",
    "max_rotation_axis_separation_deg",
    "first_reference_translation_span_m",
    "rotation_constraint_ratio",
    "rotation_nullspace_ratio",
    "rotation_nullspace_gap",
    "translation_system_min_singular_value",
    "translation_system_singular_value_ratio",
}
_FIT_KEYS = {
    "schema",
    "algorithm_id",
    "dataset_sha256",
    "frames",
    "tool_profile_hash",
    "second_reference_from_first_reference",
    "second_reference_from_first_reference_column_major",
    "first_moving_from_second_moving",
    "first_moving_from_second_moving_column_major",
    "translation_residuals_m",
    "rotation_residuals_deg",
    "translation_rms_m",
    "translation_max_m",
    "rotation_rms_deg",
    "rotation_max_deg",
    "sample_count",
    "observability",
    "accepted",
    "acceptance_note",
    "fit_semantic_sha256",
}


def _identifier(value: Any, name: str, *, max_length: int = 200) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty string up to {max_length} chars")
    if not all(character.isalnum() or character in "._:/@+-" for character in value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _sha256_reference(value: Any, name: str) -> str:
    result = _identifier(value, name, max_length=71)
    prefix = "sha256:"
    digest = result.removeprefix(prefix)
    if (
        not result.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex chars>")
    return result


def _raw_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must contain 64 lowercase SHA-256 hex chars")
    return value


def _require_exact_keys(
    value: Mapping[Any, Any],
    required: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = required.difference(value)
    unexpected = set(value).difference(allowed)
    if missing or unexpected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(str(item) for item in unexpected)}"
        )


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 matrix") from exc
    matrix = validate_rigid_transform(matrix)
    # The persisted canonical encoder intentionally collapses signed zero.
    # Normalize at the feature boundary as well so JSON-derived semantic
    # hashes remain stable across an immutable file round trip.
    matrix[matrix == 0.0] = 0.0
    matrix.flags.writeable = False
    return matrix


def controller_wire_matrix_to_transform(value: Any) -> np.ndarray:
    """Convert WebXR's flat column-major grip matrix to a validated 4x4 SE(3)."""
    try:
        wire = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "controller wire matrix must contain 16 numeric values"
        ) from exc
    if wire.shape != (16,) or not np.all(np.isfinite(wire)):
        raise ValueError("controller wire matrix must contain 16 finite values")
    return _matrix(
        wire.reshape((4, 4), order="F"),
        "controller wire matrix_reference_from_grip",
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PoseObservation:
    observation_id: str
    source_instance_id: str
    source_epoch_id: str
    sequence: int
    observed_at: float
    clock_domain_id: str
    stationary_evidence_id: str | None
    matrix_reference_from_moving: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            _finite_float(self.observed_at, "observed_at"),
        )
        object.__setattr__(
            self,
            "matrix_reference_from_moving",
            _matrix(
                self.matrix_reference_from_moving,
                "matrix_reference_from_moving",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "observation_id": self.observation_id,
            "source_instance_id": self.source_instance_id,
            "source_epoch_id": self.source_epoch_id,
            "sequence": self.sequence,
            "observed_at": self.observed_at,
            "clock_domain_id": self.clock_domain_id,
            "matrix_reference_from_moving": self.matrix_reference_from_moving.tolist(),
        }
        if self.stationary_evidence_id is not None:
            payload["stationary_evidence_id"] = self.stationary_evidence_id
        return payload


@dataclass(frozen=True)
class PairedPoseSample:
    sample_id: str
    first: PoseObservation
    second: PoseObservation

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "first_observation": self.first.as_dict(),
            "second_observation": self.second.as_dict(),
        }


@dataclass(frozen=True)
class PairedPoseDataset:
    frames: Mapping[str, str]
    pairing_method: str
    quest_session_id: str
    reference_space_type: str
    reference_space_generation: int
    controller_source_session_id: str
    robot_source_instance_id: str
    quest_reference_epoch_id: str
    robot_source_epoch_id: str
    tool_profile_hash: str
    samples: tuple[PairedPoseSample, ...]
    semantic_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", MappingProxyType(dict(self.frames)))
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "semantic_sha256", _canonical_sha256(self.as_dict()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DATASET_SCHEMA,
            "frames": dict(self.frames),
            "pairing_method": self.pairing_method,
            "source_identity": {
                "quest_session_id": self.quest_session_id,
                "reference_space_type": self.reference_space_type,
                "reference_space_generation": self.reference_space_generation,
                "controller_source_session_id": self.controller_source_session_id,
                "robot_source_instance_id": self.robot_source_instance_id,
                "quest_reference_epoch_id": self.quest_reference_epoch_id,
                "robot_source_epoch_id": self.robot_source_epoch_id,
                "tool_profile_hash": self.tool_profile_hash,
            },
            "samples": [sample.as_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class PairedPoseFit:
    dataset_sha256: str
    frames: Mapping[str, str]
    tool_profile_hash: str
    second_reference_from_first_reference: np.ndarray
    first_moving_from_second_moving: np.ndarray
    translation_residuals_m: np.ndarray
    rotation_residuals_deg: np.ndarray
    translation_rms_m: float
    translation_max_m: float
    rotation_rms_deg: float
    rotation_max_deg: float
    sample_count: int
    observability: Mapping[str, float | int]

    def __post_init__(self) -> None:
        for name in (
            "second_reference_from_first_reference",
            "first_moving_from_second_moving",
        ):
            object.__setattr__(self, name, _matrix(getattr(self, name), name))
        for name in ("translation_residuals_m", "rotation_residuals_deg"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            if values.shape != (self.sample_count,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain one finite value per sample")
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        object.__setattr__(self, "frames", MappingProxyType(dict(self.frames)))
        object.__setattr__(
            self, "observability", MappingProxyType(dict(self.observability))
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FIT_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "dataset_sha256": self.dataset_sha256,
            "frames": dict(self.frames),
            "tool_profile_hash": self.tool_profile_hash,
            "second_reference_from_first_reference": self.second_reference_from_first_reference.tolist(),
            "second_reference_from_first_reference_column_major": matrix_to_column_major(
                self.second_reference_from_first_reference
            ),
            "first_moving_from_second_moving": self.first_moving_from_second_moving.tolist(),
            "first_moving_from_second_moving_column_major": matrix_to_column_major(
                self.first_moving_from_second_moving
            ),
            "translation_residuals_m": self.translation_residuals_m.tolist(),
            "rotation_residuals_deg": self.rotation_residuals_deg.tolist(),
            "translation_rms_m": self.translation_rms_m,
            "translation_max_m": self.translation_max_m,
            "rotation_rms_deg": self.rotation_rms_deg,
            "rotation_max_deg": self.rotation_max_deg,
            "sample_count": self.sample_count,
            "observability": dict(self.observability),
            "accepted": None,
            "acceptance_note": FIT_ACCEPTANCE_NOTE,
        }
        payload["fit_semantic_sha256"] = _canonical_sha256(payload)
        return payload


def _canonical_non_negative_float(value: Any, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a canonical float")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return 0.0 if value == 0.0 else value


def _parse_fit_float_list(value: Any, name: str, *, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list with {length} values")
    values = np.asarray(
        [
            _canonical_non_negative_float(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ],
        dtype=float,
    )
    values.flags.writeable = False
    return values


def _require_fit_summary(
    supplied: float,
    expected: float,
    name: str,
) -> None:
    if supplied != expected:
        raise ValueError(f"{name} does not match the residual values")


def parse_paired_pose_fit(payload: Mapping[str, Any]) -> PairedPoseFit:
    """Strictly reconstruct one candidate-only diagnostic fit payload.

    The parser validates every redundant representation and recomputes the
    nested semantic hash. It applies the solver's numerical-observability gates
    and algorithm-implied ranges; residual quality remains unaccepted.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("paired-pose fit must be a mapping")
    _require_exact_keys(payload, _FIT_KEYS, "fit")
    if payload.get("schema") != FIT_SCHEMA:
        raise ValueError(f"schema must be {FIT_SCHEMA!r}")
    if payload.get("algorithm_id") != ALGORITHM_ID:
        raise ValueError(f"algorithm_id must be {ALGORITHM_ID!r}")
    if payload.get("accepted") is not None:
        raise ValueError("accepted must remain null for a diagnostic fit")
    if payload.get("acceptance_note") != FIT_ACCEPTANCE_NOTE:
        raise ValueError("acceptance_note does not match the diagnostic contract")

    dataset_sha256 = _raw_sha256(payload.get("dataset_sha256"), "dataset_sha256")
    fit_semantic_sha256 = _raw_sha256(
        payload.get("fit_semantic_sha256"),
        "fit_semantic_sha256",
    )
    semantic_payload = dict(payload)
    semantic_payload.pop("fit_semantic_sha256")
    if _canonical_sha256(semantic_payload) != fit_semantic_sha256:
        raise ValueError("fit_semantic_sha256 does not match the fit payload")

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, Mapping) or set(raw_frames) != _FIT_FRAME_KEYS:
        raise ValueError(f"frames must contain exactly {sorted(_FIT_FRAME_KEYS)}")
    frames = {
        key: _identifier(raw_frames.get(key), f"frames.{key}")
        for key in sorted(_FIT_FRAME_KEYS)
    }
    tool_profile_hash = _sha256_reference(
        payload.get("tool_profile_hash"),
        "tool_profile_hash",
    )

    transforms: dict[str, np.ndarray] = {}
    for matrix_name in (
        "second_reference_from_first_reference",
        "first_moving_from_second_moving",
    ):
        matrix = _matrix(payload.get(matrix_name), matrix_name)
        column_name = f"{matrix_name}_column_major"
        raw_column = payload.get(column_name)
        if not isinstance(raw_column, list) or len(raw_column) != 16:
            raise ValueError(f"{column_name} must be a list with 16 values")
        for index, value in enumerate(raw_column):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(
                    f"{column_name}[{index}] must be a finite canonical float"
                )
        if raw_column != matrix_to_column_major(matrix):
            raise ValueError(f"{column_name} does not match {matrix_name}")
        transforms[matrix_name] = matrix

    sample_count = payload.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    translation_residuals = _parse_fit_float_list(
        payload.get("translation_residuals_m"),
        "translation_residuals_m",
        length=sample_count,
    )
    rotation_residuals = _parse_fit_float_list(
        payload.get("rotation_residuals_deg"),
        "rotation_residuals_deg",
        length=sample_count,
    )
    if np.any(rotation_residuals > 180.0):
        raise ValueError("rotation_residuals_deg values cannot exceed 180")
    summaries = {
        name: _canonical_non_negative_float(payload.get(name), name)
        for name in (
            "translation_rms_m",
            "translation_max_m",
            "rotation_rms_deg",
            "rotation_max_deg",
        )
    }
    _require_fit_summary(
        summaries["translation_rms_m"],
        float(np.sqrt(np.mean(translation_residuals**2))),
        "translation_rms_m",
    )
    _require_fit_summary(
        summaries["translation_max_m"],
        float(np.max(translation_residuals)),
        "translation_max_m",
    )
    _require_fit_summary(
        summaries["rotation_rms_deg"],
        float(np.sqrt(np.mean(rotation_residuals**2))),
        "rotation_rms_deg",
    )
    _require_fit_summary(
        summaries["rotation_max_deg"],
        float(np.max(rotation_residuals)),
        "rotation_max_deg",
    )

    raw_observability = payload.get("observability")
    observability_keys = _FIT_OBSERVABILITY_INTEGER_KEYS | _FIT_OBSERVABILITY_FLOAT_KEYS
    if (
        not isinstance(raw_observability, Mapping)
        or set(raw_observability) != observability_keys
    ):
        raise ValueError(
            f"observability must contain exactly {sorted(observability_keys)}"
        )
    observability: dict[str, float | int] = {}
    for name in sorted(_FIT_OBSERVABILITY_INTEGER_KEYS):
        value = raw_observability.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"observability.{name} must be a non-negative integer")
        observability[name] = value
    for name in sorted(_FIT_OBSERVABILITY_FLOAT_KEYS):
        observability[name] = _canonical_non_negative_float(
            raw_observability.get(name),
            f"observability.{name}",
        )

    expected_relative_pair_count = sample_count * (sample_count - 1) // 2
    if observability["relative_pair_count"] != expected_relative_pair_count:
        raise ValueError(
            "observability.relative_pair_count does not match sample_count"
        )
    if not (
        2
        <= observability["effective_relative_rotation_count"]
        <= observability["relative_pair_count"]
    ):
        raise ValueError("observability effective relative rotation count is invalid")
    if observability["max_relative_rotation_deg"] < MIN_EFFECTIVE_RELATIVE_ROTATION_DEG:
        raise ValueError("observability does not satisfy the effective rotation gate")
    if observability["max_relative_rotation_deg"] > 180.0:
        raise ValueError("observability.max_relative_rotation_deg cannot exceed 180")
    if (
        observability["max_rotation_axis_separation_deg"]
        < MIN_ROTATION_AXIS_SEPARATION_DEG
    ):
        raise ValueError("observability does not satisfy the axis separation gate")
    if observability["max_rotation_axis_separation_deg"] > 90.0:
        raise ValueError(
            "observability.max_rotation_axis_separation_deg cannot exceed 90"
        )
    if observability["rotation_constraint_ratio"] < MIN_ROTATION_CONSTRAINT_RATIO:
        raise ValueError("observability does not satisfy the rotation constraint gate")
    if observability["rotation_constraint_ratio"] > 1.0:
        raise ValueError("observability.rotation_constraint_ratio cannot exceed 1")
    if observability["rotation_nullspace_ratio"] > MAX_ROTATION_NULLSPACE_RATIO:
        raise ValueError("observability does not satisfy the rotation nullspace gate")
    if observability["rotation_nullspace_gap"] < MIN_ROTATION_NULLSPACE_GAP:
        raise ValueError(
            "observability does not satisfy the rotation nullspace gap gate"
        )
    if observability["translation_system_rank"] != 3:
        raise ValueError("observability.translation_system_rank must be 3")
    if observability["translation_system_min_singular_value"] <= 0.0:
        raise ValueError(
            "observability.translation_system_min_singular_value must be positive"
        )
    if (
        observability["translation_system_singular_value_ratio"]
        < MIN_TRANSLATION_SINGULAR_VALUE_RATIO
    ):
        raise ValueError("observability does not satisfy the translation system gate")
    if observability["translation_system_singular_value_ratio"] > 1.0:
        raise ValueError(
            "observability.translation_system_singular_value_ratio cannot exceed 1"
        )

    fit = PairedPoseFit(
        dataset_sha256=dataset_sha256,
        frames=frames,
        tool_profile_hash=tool_profile_hash,
        second_reference_from_first_reference=transforms[
            "second_reference_from_first_reference"
        ],
        first_moving_from_second_moving=transforms["first_moving_from_second_moving"],
        translation_residuals_m=translation_residuals,
        rotation_residuals_deg=rotation_residuals,
        translation_rms_m=summaries["translation_rms_m"],
        translation_max_m=summaries["translation_max_m"],
        rotation_rms_deg=summaries["rotation_rms_deg"],
        rotation_max_deg=summaries["rotation_max_deg"],
        sample_count=sample_count,
        observability=observability,
    )
    if fit.as_dict() != dict(payload):
        raise ValueError("paired-pose fit payload is not canonical")
    return fit


def _parse_observation(
    value: Any, *, name: str, require_stationary: bool
) -> PoseObservation:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(
        value,
        {
            "observation_id",
            "source_instance_id",
            "source_epoch_id",
            "sequence",
            "observed_at",
            "clock_domain_id",
            "matrix_reference_from_moving",
        },
        name,
        optional={"stationary_evidence_id"},
    )
    stationary = value.get("stationary_evidence_id")
    if stationary is not None:
        stationary = _identifier(stationary, f"{name}.stationary_evidence_id")
    if require_stationary and stationary is None:
        raise ValueError(f"{name} requires stationary_evidence_id")
    return PoseObservation(
        observation_id=_identifier(
            value.get("observation_id"), f"{name}.observation_id"
        ),
        source_instance_id=_identifier(
            value.get("source_instance_id"), f"{name}.source_instance_id"
        ),
        source_epoch_id=_identifier(
            value.get("source_epoch_id"), f"{name}.source_epoch_id"
        ),
        sequence=_non_negative_int(value.get("sequence"), f"{name}.sequence"),
        observed_at=_finite_float(value.get("observed_at"), f"{name}.observed_at"),
        clock_domain_id=_identifier(
            value.get("clock_domain_id"), f"{name}.clock_domain_id"
        ),
        stationary_evidence_id=stationary,
        matrix_reference_from_moving=_matrix(
            value.get("matrix_reference_from_moving"),
            f"{name}.matrix_reference_from_moving",
        ),
    )


def parse_paired_pose_dataset(payload: Mapping[str, Any]) -> PairedPoseDataset:
    if not isinstance(payload, Mapping):
        raise TypeError("paired-pose dataset must be a mapping")
    _require_exact_keys(
        payload,
        {"schema", "frames", "pairing_method", "source_identity", "samples"},
        "dataset",
    )
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"schema must be {DATASET_SCHEMA!r}")
    raw_frames = payload.get("frames")
    expected_frame_keys = {
        "first_reference",
        "first_moving",
        "second_reference",
        "second_moving",
    }
    if not isinstance(raw_frames, Mapping) or set(raw_frames) != expected_frame_keys:
        raise ValueError(f"frames must contain exactly {sorted(expected_frame_keys)}")
    frames = {
        key: _identifier(raw_frames.get(key), f"frames.{key}")
        for key in sorted(expected_frame_keys)
    }
    pairing_method = payload.get("pairing_method")
    if pairing_method not in PAIRING_METHODS:
        raise ValueError(f"pairing_method must be one of {sorted(PAIRING_METHODS)}")
    source = payload.get("source_identity")
    if not isinstance(source, Mapping):
        raise TypeError("source_identity must be a mapping")
    _require_exact_keys(
        source,
        {
            "quest_session_id",
            "reference_space_type",
            "reference_space_generation",
            "controller_source_session_id",
            "robot_source_instance_id",
            "quest_reference_epoch_id",
            "robot_source_epoch_id",
            "tool_profile_hash",
        },
        "source_identity",
    )
    generation = _non_negative_int(
        source.get("reference_space_generation"),
        "source_identity.reference_space_generation",
    )
    quest_session_id = _identifier(
        source.get("quest_session_id"), "source_identity.quest_session_id"
    )
    controller_source_session_id = _identifier(
        source.get("controller_source_session_id"),
        "source_identity.controller_source_session_id",
    )
    robot_source_instance_id = _identifier(
        source.get("robot_source_instance_id"),
        "source_identity.robot_source_instance_id",
    )
    quest_reference_epoch_id = _identifier(
        source.get("quest_reference_epoch_id"),
        "source_identity.quest_reference_epoch_id",
    )
    robot_source_epoch_id = _identifier(
        source.get("robot_source_epoch_id"),
        "source_identity.robot_source_epoch_id",
    )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
        raise TypeError("samples must be a sequence")
    samples: list[PairedPoseSample] = []
    sample_ids: set[str] = set()
    first_observation_ids: set[str] = set()
    second_observation_ids: set[str] = set()
    first_sequences: set[int] = set()
    second_sequences: set[int] = set()
    stationary_evidence_ids: set[str] = set()
    require_stationary = pairing_method == "operator_static_window"
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, Mapping):
            raise TypeError(f"samples[{index}] must be a mapping")
        _require_exact_keys(
            raw_sample,
            {"sample_id", "first_observation", "second_observation"},
            f"samples[{index}]",
        )
        sample_id = _identifier(
            raw_sample.get("sample_id"), f"samples[{index}].sample_id"
        )
        if sample_id in sample_ids:
            raise ValueError("sample_id values must be unique")
        first = _parse_observation(
            raw_sample.get("first_observation"),
            name=f"samples[{index}].first_observation",
            require_stationary=require_stationary,
        )
        second = _parse_observation(
            raw_sample.get("second_observation"),
            name=f"samples[{index}].second_observation",
            require_stationary=require_stationary,
        )
        if first.source_instance_id != robot_source_instance_id:
            raise ValueError("first observation source identity mismatch")
        if second.source_instance_id != controller_source_session_id:
            raise ValueError("second observation source identity mismatch")
        if first.source_epoch_id != robot_source_epoch_id:
            raise ValueError("first observation epoch identity mismatch")
        if second.source_epoch_id != quest_reference_epoch_id:
            raise ValueError("second observation reference epoch mismatch")
        if first.observation_id in first_observation_ids:
            raise ValueError("first observation IDs must be unique")
        if second.observation_id in second_observation_ids:
            raise ValueError("second observation IDs must be unique")
        if first.sequence in first_sequences:
            raise ValueError("first observation sequence values must be unique")
        if second.sequence in second_sequences:
            raise ValueError("second observation sequence values must be unique")
        if require_stationary:
            if first.stationary_evidence_id != second.stationary_evidence_id:
                raise ValueError(
                    "paired observations must share one stationary_evidence_id"
                )
            if first.stationary_evidence_id in stationary_evidence_ids:
                raise ValueError("stationary_evidence_id values must be unique")
            stationary_evidence_ids.add(str(first.stationary_evidence_id))
        sample_ids.add(sample_id)
        first_observation_ids.add(first.observation_id)
        second_observation_ids.add(second.observation_id)
        first_sequences.add(first.sequence)
        second_sequences.add(second.sequence)
        samples.append(PairedPoseSample(sample_id, first, second))
    dataset = PairedPoseDataset(
        frames=frames,
        pairing_method=str(pairing_method),
        quest_session_id=quest_session_id,
        reference_space_type=_identifier(
            source.get("reference_space_type"),
            "source_identity.reference_space_type",
        ),
        reference_space_generation=generation,
        controller_source_session_id=controller_source_session_id,
        robot_source_instance_id=robot_source_instance_id,
        quest_reference_epoch_id=quest_reference_epoch_id,
        robot_source_epoch_id=robot_source_epoch_id,
        tool_profile_hash=_sha256_reference(
            source.get("tool_profile_hash"), "source_identity.tool_profile_hash"
        ),
        samples=tuple(samples),
    )
    return dataset


def _project_rotation(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value, dtype=float)
    if np.linalg.det(raw) < 0:
        raw = -raw
    u, _, vt = np.linalg.svd(raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _rotation_axis(rotation: np.ndarray) -> tuple[float, np.ndarray]:
    quaternion = np.asarray(matrix_to_quaternion_xyzw(rotation), dtype=float)
    if quaternion[3] < 0:
        quaternion = -quaternion
    angle = float(2.0 * math.acos(float(np.clip(quaternion[3], -1.0, 1.0))))
    sine = float(np.linalg.norm(quaternion[:3]))
    if sine < 1e-9:
        return angle, np.zeros(3, dtype=float)
    return angle, quaternion[:3] / sine


def _mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    rotation_sum = sum((matrix[:3, :3] for matrix in transforms), np.zeros((3, 3)))
    result = np.eye(4, dtype=float)
    result[:3, :3] = _project_rotation(rotation_sum)
    result[:3, 3] = np.mean([matrix[:3, 3] for matrix in transforms], axis=0)
    return validate_rigid_transform(result)


def _validated_dataset(dataset: PairedPoseDataset) -> PairedPoseDataset:
    if not isinstance(dataset, PairedPoseDataset):
        raise TypeError("dataset must be a PairedPoseDataset")
    reparsed = parse_paired_pose_dataset(dataset.as_dict())
    if dataset.semantic_sha256 != reparsed.semantic_sha256:
        raise ValueError("dataset semantic hash does not match its current content")
    return reparsed


def fit_paired_pose_calibration(dataset: PairedPoseDataset) -> PairedPoseFit:
    """Fit both fixed transforms without scale using a NumPy AX=XB solve."""
    dataset = _validated_dataset(dataset)
    if len(dataset.samples) < 3:
        raise ValueError(
            "at least 3 paired poses are required; 15+ diverse poses are recommended"
        )
    first = [sample.first.matrix_reference_from_moving for sample in dataset.samples]
    second = [sample.second.matrix_reference_from_moving for sample in dataset.samples]
    relative_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    excitation_axes: list[np.ndarray] = []
    excitation_angles: list[float] = []
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            first_relative = rigid_inverse(first[i]) @ first[j]
            second_relative = rigid_inverse(second[i]) @ second[j]
            relative_pairs.append((first_relative, second_relative))
            angle, axis = _rotation_axis(first_relative[:3, :3])
            if math.degrees(angle) >= MIN_EFFECTIVE_RELATIVE_ROTATION_DEG:
                excitation_angles.append(angle)
                excitation_axes.append(axis)
    if len(excitation_axes) < 2:
        raise ValueError(
            "paired poses need at least two effective relative rotations of "
            f"{MIN_EFFECTIVE_RELATIVE_ROTATION_DEG:g} degrees or more"
        )
    axis_separations_deg = [
        math.degrees(
            math.acos(
                float(
                    np.clip(
                        abs(float(np.dot(excitation_axes[i], excitation_axes[j]))),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        for i in range(len(excitation_axes))
        for j in range(i + 1, len(excitation_axes))
    ]
    maximum_axis_separation_deg = max(axis_separations_deg)
    if maximum_axis_separation_deg < MIN_ROTATION_AXIS_SEPARATION_DEG:
        raise ValueError(
            "paired poses need effective rotations about axes separated by at least "
            f"{MIN_ROTATION_AXIS_SEPARATION_DEG:g} degrees"
        )

    rotation_rows = [
        np.kron(np.eye(3), first_relative[:3, :3])
        - np.kron(second_relative[:3, :3].T, np.eye(3))
        for first_relative, second_relative in relative_pairs
    ]
    rotation_system = np.vstack(rotation_rows)
    _, rotation_singular_values, vt = np.linalg.svd(rotation_system)
    largest_rotation_sv = float(rotation_singular_values[0])
    next_rotation_sv = float(rotation_singular_values[-2])
    smallest_rotation_sv = float(rotation_singular_values[-1])
    if largest_rotation_sv <= np.finfo(float).eps:
        raise ValueError("paired-pose rotation system has no usable constraints")
    rotation_constraint_ratio = next_rotation_sv / largest_rotation_sv
    rotation_nullspace_ratio = smallest_rotation_sv / largest_rotation_sv
    rotation_nullspace_gap = next_rotation_sv / max(
        smallest_rotation_sv, np.finfo(float).eps
    )
    if (
        rotation_constraint_ratio < MIN_ROTATION_CONSTRAINT_RATIO
        or rotation_nullspace_ratio > MAX_ROTATION_NULLSPACE_RATIO
        or rotation_nullspace_gap < MIN_ROTATION_NULLSPACE_GAP
    ):
        raise ValueError("paired-pose rotation system is numerically ambiguous")
    raw_rotation = vt[-1].reshape((3, 3), order="F")
    moving_rotation = _project_rotation(raw_rotation)

    translation_rows = []
    translation_rhs = []
    for first_relative, second_relative in relative_pairs:
        translation_rows.append(first_relative[:3, :3] - np.eye(3))
        translation_rhs.append(
            moving_rotation @ second_relative[:3, 3] - first_relative[:3, 3]
        )
    translation_system = np.vstack(translation_rows)
    moving_translation, _, _, translation_singular_values = np.linalg.lstsq(
        translation_system,
        np.concatenate(translation_rhs),
        rcond=None,
    )
    largest_translation_sv = float(translation_singular_values[0])
    smallest_translation_sv = float(translation_singular_values[-1])
    translation_singular_value_ratio = smallest_translation_sv / max(
        largest_translation_sv, np.finfo(float).eps
    )
    if translation_singular_value_ratio < MIN_TRANSLATION_SINGULAR_VALUE_RATIO:
        raise ValueError(
            "paired-pose translation system is rank deficient or ill-conditioned"
        )
    first_moving_from_second_moving = np.eye(4, dtype=float)
    first_moving_from_second_moving[:3, :3] = moving_rotation
    first_moving_from_second_moving[:3, 3] = moving_translation
    first_moving_from_second_moving = validate_rigid_transform(
        first_moving_from_second_moving
    )

    reference_candidates = [
        second_pose
        @ rigid_inverse(first_moving_from_second_moving)
        @ rigid_inverse(first_pose)
        for first_pose, second_pose in zip(first, second, strict=True)
    ]
    second_reference_from_first_reference = _mean_transform(reference_candidates)

    translation_residuals: list[float] = []
    rotation_residuals_deg: list[float] = []
    for first_pose, second_pose in zip(first, second, strict=True):
        predicted = (
            second_reference_from_first_reference
            @ first_pose
            @ first_moving_from_second_moving
        )
        error = rigid_inverse(predicted) @ second_pose
        translation_residuals.append(float(np.linalg.norm(error[:3, 3])))
        rotation_residuals_deg.append(math.degrees(_rotation_angle_rad(error[:3, :3])))
    translation_values = np.asarray(translation_residuals, dtype=float)
    rotation_values = np.asarray(rotation_residuals_deg, dtype=float)
    first_positions = np.asarray([matrix[:3, 3] for matrix in first])
    position_span = max(
        float(np.linalg.norm(first_positions[i] - first_positions[j]))
        for i in range(len(first_positions))
        for j in range(i + 1, len(first_positions))
    )
    observability: dict[str, float | int] = {
        "relative_pair_count": len(relative_pairs),
        "effective_relative_rotation_count": len(excitation_angles),
        "max_relative_rotation_deg": math.degrees(max(excitation_angles)),
        "max_rotation_axis_separation_deg": maximum_axis_separation_deg,
        "first_reference_translation_span_m": position_span,
        "rotation_constraint_ratio": rotation_constraint_ratio,
        "rotation_nullspace_ratio": rotation_nullspace_ratio,
        "rotation_nullspace_gap": rotation_nullspace_gap,
        "translation_system_rank": 3,
        "translation_system_min_singular_value": smallest_translation_sv,
        "translation_system_singular_value_ratio": translation_singular_value_ratio,
    }
    return PairedPoseFit(
        dataset_sha256=dataset.semantic_sha256,
        frames=dataset.frames,
        tool_profile_hash=dataset.tool_profile_hash,
        second_reference_from_first_reference=second_reference_from_first_reference,
        first_moving_from_second_moving=first_moving_from_second_moving,
        translation_residuals_m=translation_values,
        rotation_residuals_deg=rotation_values,
        translation_rms_m=float(np.sqrt(np.mean(translation_values**2))),
        translation_max_m=float(np.max(translation_values)),
        rotation_rms_deg=float(np.sqrt(np.mean(rotation_values**2))),
        rotation_max_deg=float(np.max(rotation_values)),
        sample_count=len(dataset.samples),
        observability=observability,
    )
