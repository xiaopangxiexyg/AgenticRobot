"""Read-only equivalence checks for native-to-v2.3 dual-write shadows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .v25p_v23_adapter import V25pV23ProjectedRow


@dataclass(frozen=True)
class V25pFieldEquivalence:
    dataset_field: str
    equivalent: bool
    reason: str
    native_dtype: str | None = None
    legacy_dtype: str | None = None
    native_shape: tuple[int, ...] | None = None
    legacy_shape: tuple[int, ...] | None = None
    max_absolute_error: float | None = None


@dataclass(frozen=True)
class V25pDualWriteEquivalenceReport:
    sample_index: int
    equivalent: bool
    fields: Mapping[str, V25pFieldEquivalence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def compare_projected_v23_row(
    projected: V25pV23ProjectedRow,
    legacy_row: Mapping[str, Any],
    *,
    float_atol: float = 0.0,
    float_rtol: float = 0.0,
) -> V25pDualWriteEquivalenceReport:
    """Compare only fields selected by the explicit compatibility projection."""

    results: dict[str, V25pFieldEquivalence] = {}
    legacy_sample_index = legacy_row.get("frame_index", legacy_row.get("index"))
    try:
        normalized_legacy_sample_index = int(legacy_sample_index)
    except (TypeError, ValueError, OverflowError):
        normalized_legacy_sample_index = None
    identity_equal = (
        not isinstance(legacy_sample_index, bool)
        and normalized_legacy_sample_index == projected.sample_index
    )
    results["__sample_index__"] = V25pFieldEquivalence(
        dataset_field="__sample_index__",
        equivalent=identity_equal,
        reason="equal" if identity_equal else "sample_index_mismatch",
        native_dtype="int64",
        legacy_dtype=(
            None if legacy_sample_index is None else np.asarray(legacy_sample_index).dtype.name
        ),
        native_shape=(),
        legacy_shape=(
            None if legacy_sample_index is None else tuple(np.asarray(legacy_sample_index).shape)
        ),
        max_absolute_error=(
            None
            if legacy_sample_index is None or isinstance(legacy_sample_index, bool)
            else (
                None
                if normalized_legacy_sample_index is None
                else abs(
                    float(projected.sample_index)
                    - float(normalized_legacy_sample_index)
                )
            )
        ),
    )
    for field, native_values in projected.fields.items():
        if field not in legacy_row:
            results[field] = V25pFieldEquivalence(
                dataset_field=field,
                equivalent=False,
                reason="legacy_field_missing",
            )
            continue
        declared_dtype = np.dtype(projected.dtype_by_field[field])
        declared_shape = projected.shape_by_field[field]
        native = np.asarray(native_values, dtype=declared_dtype).reshape(declared_shape)
        legacy_raw = np.asarray(legacy_row[field])
        common = {
            "dataset_field": field,
            "native_dtype": declared_dtype.name,
            "legacy_dtype": legacy_raw.dtype.name,
            "native_shape": tuple(native.shape),
            "legacy_shape": tuple(legacy_raw.shape),
        }
        if tuple(legacy_raw.shape) != declared_shape:
            results[field] = V25pFieldEquivalence(
                equivalent=False,
                reason="shape_mismatch",
                **common,
            )
            continue
        try:
            legacy = legacy_raw.astype(declared_dtype)
        except (TypeError, ValueError):
            results[field] = V25pFieldEquivalence(
                equivalent=False,
                reason="legacy_not_representable_by_native_contract",
                **common,
            )
            continue
        if declared_dtype.kind in "iu":
            try:
                exactly_representable = bool(
                    np.array_equal(legacy_raw.astype(np.float64), legacy.astype(np.float64))
                )
            except (TypeError, ValueError):
                exactly_representable = False
            if not exactly_representable:
                results[field] = V25pFieldEquivalence(
                    equivalent=False,
                    reason="legacy_not_representable_by_native_contract",
                    **common,
                )
                continue
        if declared_dtype.kind == "b" and not np.all(
            np.isin(legacy_raw, (False, True, 0, 1))
        ):
            results[field] = V25pFieldEquivalence(
                equivalent=False,
                reason="legacy_not_representable_by_native_contract",
                **common,
            )
            continue
        if declared_dtype.kind == "f" and not np.array_equal(
            np.isfinite(native), np.isfinite(legacy)
        ):
            results[field] = V25pFieldEquivalence(
                equivalent=False,
                reason="non_finite_pattern_mismatch",
                **common,
            )
            continue
        numeric = native.dtype.kind in "biuf" and legacy.dtype.kind in "biuf"
        if numeric:
            native64 = native.astype(np.float64)
            legacy64 = legacy.astype(np.float64)
            finite_match = np.array_equal(np.isfinite(native64), np.isfinite(legacy64))
            equivalent = finite_match and bool(
                np.allclose(
                    native64,
                    legacy64,
                    atol=float(float_atol),
                    rtol=float(float_rtol),
                    equal_nan=True,
                )
            )
            finite = np.isfinite(native64) & np.isfinite(legacy64)
            max_error = (
                float(np.max(np.abs(native64[finite] - legacy64[finite])))
                if np.any(finite)
                else 0.0
            )
            reason = "equal" if equivalent else "value_mismatch"
        else:
            equivalent = bool(np.array_equal(native, legacy))
            max_error = None
            reason = "equal" if equivalent else "value_mismatch"
        results[field] = V25pFieldEquivalence(
            equivalent=equivalent,
            reason=reason,
            max_absolute_error=max_error,
            **common,
        )
    return V25pDualWriteEquivalenceReport(
        sample_index=projected.sample_index,
        equivalent=all(item.equivalent for item in results.values()),
        fields=results,
    )


__all__ = [
    "V25pDualWriteEquivalenceReport",
    "V25pFieldEquivalence",
    "compare_projected_v23_row",
]
