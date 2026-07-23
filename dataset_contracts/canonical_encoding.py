"""Deterministic primitive encoding shared by runtime and dataset contracts.

The encoder is intentionally small and closed.  Callers must provide an
explicit semantic payload instead of passing arbitrary dataclasses or object
representations.  This keeps runtime semantic hashes independent from Python
implementation details and from later persisted-record envelopes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


CANONICAL_ENCODING_SCHEMA_V1 = "dexteleop_canonical_encoding_v1"
CANONICAL_TIMESTAMP_UNIT_V1 = "seconds"


@dataclass(frozen=True)
class CanonicalEncodingSpec:
    schema: str = CANONICAL_ENCODING_SCHEMA_V1
    hash_algorithm: str = "sha256"
    timestamp_unit: str = CANONICAL_TIMESTAMP_UNIT_V1

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_ENCODING_SCHEMA_V1:
            raise ValueError(f"Unsupported canonical encoding schema: {self.schema!r}.")
        if self.hash_algorithm != "sha256":
            raise ValueError("CanonicalEncodingSpec v1 requires SHA-256.")
        if self.timestamp_unit != CANONICAL_TIMESTAMP_UNIT_V1:
            raise ValueError("CanonicalEncodingSpec v1 timestamps use seconds.")


DEFAULT_CANONICAL_ENCODING_SPEC = CanonicalEncodingSpec()


def _normalized_float_bits(value: float, *, dtype: np.dtype[Any]) -> str:
    scalar = np.asarray(value, dtype=dtype)
    numeric = float(scalar)
    if math.isinf(numeric):
        raise ValueError("Canonical encoding rejects infinite floats.")
    if math.isnan(numeric):
        scalar = np.asarray(np.nan, dtype=dtype)
    elif numeric == 0.0:
        scalar = np.asarray(0.0, dtype=dtype)
    little = scalar.astype(dtype.newbyteorder("<"), copy=False)
    return little.tobytes().hex()


def _normalize_ndarray(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.subdtype is not None:
        raise TypeError("CanonicalEncodingSpec v1 rejects object and structured arrays.")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"Unsupported ndarray dtype for canonical encoding: {array.dtype}.")

    dtype = array.dtype.newbyteorder("<") if array.dtype.byteorder != "|" else array.dtype
    normalized = np.ascontiguousarray(array.astype(dtype, copy=True))
    if normalized.dtype.kind == "f":
        if np.any(np.isinf(normalized)):
            raise ValueError("Canonical encoding rejects infinite ndarray values.")
        zero_mask = normalized == 0
        if np.any(zero_mask):
            normalized[zero_mask] = 0.0
        nan_mask = np.isnan(normalized)
        if np.any(nan_mask):
            normalized[nan_mask] = np.asarray(np.nan, dtype=normalized.dtype)

    return {
        "$type": "ndarray",
        "dtype": normalized.dtype.str,
        "shape": [int(size) for size in normalized.shape],
        "data_hex": normalized.tobytes(order="C").hex(),
    }


def canonical_data(value: Any) -> Any:
    """Project supported primitive values into a tagged canonical tree."""

    if value is None:
        return {"$type": "none"}
    if isinstance(value, Enum):
        enum_type = type(value)
        return {
            "$type": "enum",
            "enum_type": f"{enum_type.__module__}.{enum_type.__qualname__}",
            "value": canonical_data(value.value),
        }
    if isinstance(value, (bool, np.bool_)):
        return {"$type": "bool", "value": bool(value)}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"$type": "int", "value": str(value)}
    if isinstance(value, np.integer):
        dtype = value.dtype.newbyteorder("<")
        return {
            "$type": "numpy_int",
            "dtype": dtype.str,
            "value": str(int(value)),
        }
    if isinstance(value, float):
        return {"$type": "float64", "bits": _normalized_float_bits(value, dtype=np.dtype("<f8"))}
    if isinstance(value, np.floating):
        dtype = value.dtype.newbyteorder("<")
        return {
            "$type": "numpy_float",
            "dtype": dtype.str,
            "bits": _normalized_float_bits(float(value), dtype=dtype),
        }
    if isinstance(value, str):
        return {"$type": "str", "value": value}
    if isinstance(value, bytes):
        return {"$type": "bytes", "data_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return _normalize_ndarray(value)
    if isinstance(value, Mapping):
        entries: list[list[Any]] = []
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical mappings require string keys.")
        for key in sorted(value):
            entries.append([key, canonical_data(value[key])])
        return {"$type": "mapping", "entries": entries}
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [canonical_data(item) for item in value]}
    if isinstance(value, list):
        return {"$type": "list", "items": [canonical_data(item) for item in value]}
    raise TypeError(
        "CanonicalEncodingSpec v1 only accepts explicit primitive payloads; "
        f"got {type(value).__name__}."
    )


def canonical_bytes(
    payload: Any,
    *,
    domain: str,
    spec: CanonicalEncodingSpec = DEFAULT_CANONICAL_ENCODING_SPEC,
) -> bytes:
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("Canonical encoding domain must be a non-empty string.")
    envelope = {
        "schema": spec.schema,
        "timestamp_unit": spec.timestamp_unit,
        "domain": domain,
        "payload": canonical_data(payload),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_content_hash(
    payload: Any,
    *,
    domain: str,
    spec: CanonicalEncodingSpec = DEFAULT_CANONICAL_ENCODING_SPEC,
) -> str:
    return hashlib.sha256(canonical_bytes(payload, domain=domain, spec=spec)).hexdigest()


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Canonical JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _decode_canonical_data(value: Any) -> Any:
    if not isinstance(value, dict) or "$type" not in value:
        raise ValueError("Canonical value must be a tagged object.")
    kind = value.get("$type")
    if kind == "none" and set(value) == {"$type"}:
        return None
    if kind == "bool" and set(value) == {"$type", "value"}:
        if not isinstance(value["value"], bool):
            raise ValueError("Canonical bool payload is invalid.")
        return value["value"]
    if kind == "int" and set(value) == {"$type", "value"}:
        return int(value["value"])
    if kind == "float64" and set(value) == {"$type", "bits"}:
        raw = bytes.fromhex(str(value["bits"]))
        if len(raw) != 8:
            raise ValueError("Canonical float64 payload is invalid.")
        return np.frombuffer(raw, dtype=np.dtype("<f8"))[0].item()
    if kind == "numpy_int" and set(value) == {"$type", "dtype", "value"}:
        dtype = np.dtype(str(value["dtype"]))
        if dtype.kind not in {"i", "u"}:
            raise ValueError("Canonical numpy_int dtype is invalid.")
        return np.asarray(int(value["value"]), dtype=dtype)[()]
    if kind == "numpy_float" and set(value) == {"$type", "dtype", "bits"}:
        dtype = np.dtype(str(value["dtype"]))
        if dtype.kind != "f":
            raise ValueError("Canonical numpy_float dtype is invalid.")
        raw = bytes.fromhex(str(value["bits"]))
        if len(raw) != dtype.itemsize:
            raise ValueError("Canonical numpy_float payload is invalid.")
        return np.frombuffer(raw, dtype=dtype)[0]
    if kind == "str" and set(value) == {"$type", "value"}:
        if not isinstance(value["value"], str):
            raise ValueError("Canonical str payload is invalid.")
        return value["value"]
    if kind == "bytes" and set(value) == {"$type", "data_hex"}:
        return bytes.fromhex(str(value["data_hex"]))
    if kind == "ndarray" and set(value) == {"$type", "dtype", "shape", "data_hex"}:
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(item) for item in value["shape"])
        if any(item < 0 for item in shape) or dtype.hasobject:
            raise ValueError("Canonical ndarray shape/dtype is invalid.")
        raw = bytes.fromhex(str(value["data_hex"]))
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise ValueError("Canonical ndarray byte length does not match shape/dtype.")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if kind == "mapping" and set(value) == {"$type", "entries"}:
        result: dict[str, Any] = {}
        for entry in value["entries"]:
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                raise ValueError("Canonical mapping entry is invalid.")
            key = entry[0]
            if key in result:
                raise ValueError(f"Canonical mapping contains duplicate key {key!r}.")
            result[key] = _decode_canonical_data(entry[1])
        return result
    if kind in {"tuple", "list"} and set(value) == {"$type", "items"}:
        items = [_decode_canonical_data(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "enum" and set(value) == {"$type", "enum_type", "value"}:
        # Storage decoding must not import arbitrary Python enum classes.  The
        # explicit record schema consumes the semantic value only.
        return _decode_canonical_data(value["value"])
    raise ValueError(f"Unsupported or malformed canonical value kind: {kind!r}.")


def canonical_loads(
    encoded: bytes | bytearray | memoryview,
    *,
    expected_domain: str,
    spec: CanonicalEncodingSpec = DEFAULT_CANONICAL_ENCODING_SPEC,
) -> Any:
    """Decode and validate bytes produced by :func:`canonical_bytes`."""

    if not isinstance(expected_domain, str) or not expected_domain.strip():
        raise ValueError("expected_domain must be a non-empty string.")
    try:
        envelope = json.loads(
            bytes(encoded).decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid canonical JSON encoding.") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "timestamp_unit",
        "domain",
        "payload",
    }:
        raise ValueError("Canonical encoding envelope fields are invalid.")
    if envelope["schema"] != spec.schema or envelope["timestamp_unit"] != spec.timestamp_unit:
        raise ValueError("Canonical encoding schema/timestamp unit mismatch.")
    if envelope["domain"] != expected_domain:
        raise ValueError("Canonical encoding domain mismatch.")
    return _decode_canonical_data(envelope["payload"])


__all__ = [
    "CANONICAL_ENCODING_SCHEMA_V1",
    "CANONICAL_TIMESTAMP_UNIT_V1",
    "CanonicalEncodingSpec",
    "DEFAULT_CANONICAL_ENCODING_SPEC",
    "canonical_bytes",
    "canonical_content_hash",
    "canonical_data",
    "canonical_loads",
]
