"""Strict decoder for DexTeleop canonical JSON artifacts.

This file intentionally contains only storage decoding. It does not import the
DexTeleop runtime and never instantiates arbitrary Python classes named by an
artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_SCHEMA = "dexteleop_canonical_encoding_v1"
CANONICAL_TIMESTAMP_UNIT = "seconds"


class CanonicalDecodeError(ValueError):
    """Raised when a canonical artifact is malformed or has the wrong domain."""


@dataclass(frozen=True)
class CanonicalDocument:
    domain: str
    payload: Any


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalDecodeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_tagged(value: Any) -> Any:
    if not isinstance(value, dict) or "$type" not in value:
        raise CanonicalDecodeError("canonical value must be a tagged object")
    kind = value.get("$type")
    if kind == "none" and set(value) == {"$type"}:
        return None
    if kind == "bool" and set(value) == {"$type", "value"}:
        if not isinstance(value["value"], bool):
            raise CanonicalDecodeError("invalid canonical bool")
        return value["value"]
    if kind == "int" and set(value) == {"$type", "value"}:
        return int(value["value"])
    if kind == "float64" and set(value) == {"$type", "bits"}:
        raw = bytes.fromhex(str(value["bits"]))
        if len(raw) != 8:
            raise CanonicalDecodeError("invalid canonical float64")
        return np.frombuffer(raw, dtype=np.dtype("<f8"))[0].item()
    if kind == "numpy_int" and set(value) == {"$type", "dtype", "value"}:
        dtype = np.dtype(str(value["dtype"]))
        if dtype.kind not in {"i", "u"}:
            raise CanonicalDecodeError("invalid canonical numpy integer dtype")
        return np.asarray(int(value["value"]), dtype=dtype)[()]
    if kind == "numpy_float" and set(value) == {"$type", "dtype", "bits"}:
        dtype = np.dtype(str(value["dtype"]))
        raw = bytes.fromhex(str(value["bits"]))
        if dtype.kind != "f" or len(raw) != dtype.itemsize:
            raise CanonicalDecodeError("invalid canonical numpy float")
        return np.frombuffer(raw, dtype=dtype)[0]
    if kind == "str" and set(value) == {"$type", "value"}:
        if not isinstance(value["value"], str):
            raise CanonicalDecodeError("invalid canonical string")
        return value["value"]
    if kind == "bytes" and set(value) == {"$type", "data_hex"}:
        return bytes.fromhex(str(value["data_hex"]))
    if kind == "ndarray" and set(value) == {
        "$type",
        "dtype",
        "shape",
        "data_hex",
    }:
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(item) for item in value["shape"])
        if dtype.hasobject or any(size < 0 for size in shape):
            raise CanonicalDecodeError("invalid canonical ndarray dtype or shape")
        raw = bytes.fromhex(str(value["data_hex"]))
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise CanonicalDecodeError("canonical ndarray byte length mismatch")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if kind == "mapping" and set(value) == {"$type", "entries"}:
        result: dict[str, Any] = {}
        for entry in value["entries"]:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
            ):
                raise CanonicalDecodeError("invalid canonical mapping entry")
            key = entry[0]
            if key in result:
                raise CanonicalDecodeError(f"duplicate canonical mapping key {key!r}")
            result[key] = _decode_tagged(entry[1])
        return result
    if kind in {"tuple", "list"} and set(value) == {"$type", "items"}:
        items = [_decode_tagged(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "enum" and set(value) == {"$type", "enum_type", "value"}:
        return _decode_tagged(value["value"])
    raise CanonicalDecodeError(f"unsupported canonical value kind {kind!r}")


def _decode_document(encoded: bytes, *, expected_domain: str | None) -> CanonicalDocument:
    try:
        envelope = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalDecodeError("invalid canonical JSON") from exc
    required = {"schema", "timestamp_unit", "domain", "payload"}
    if not isinstance(envelope, dict) or set(envelope) != required:
        raise CanonicalDecodeError("invalid canonical envelope fields")
    if envelope["schema"] != CANONICAL_SCHEMA:
        raise CanonicalDecodeError(f"unsupported canonical schema {envelope['schema']!r}")
    if envelope["timestamp_unit"] != CANONICAL_TIMESTAMP_UNIT:
        raise CanonicalDecodeError("canonical timestamp unit must be seconds")
    domain = envelope["domain"]
    if not isinstance(domain, str) or not domain:
        raise CanonicalDecodeError("canonical domain must be a non-empty string")
    if expected_domain is not None and domain != expected_domain:
        raise CanonicalDecodeError(
            f"canonical domain mismatch: expected={expected_domain!r} actual={domain!r}"
        )
    return CanonicalDocument(domain=domain, payload=_decode_tagged(envelope["payload"]))


def load_canonical_file(
    path: str | Path,
    *,
    expected_domain: str | None = None,
) -> CanonicalDocument:
    source = Path(path)
    try:
        encoded = source.read_bytes()
    except OSError as exc:
        raise CanonicalDecodeError(f"cannot read canonical file {source}: {exc}") from exc
    return _decode_document(encoded, expected_domain=expected_domain)


def load_canonical_lines(
    path: str | Path,
    *,
    expected_domain: str | None = None,
) -> tuple[CanonicalDocument, ...]:
    source = Path(path)
    try:
        encoded = source.read_bytes()
    except OSError as exc:
        raise CanonicalDecodeError(f"cannot read canonical lines {source}: {exc}") from exc
    if not encoded or not encoded.endswith(b"\n"):
        raise CanonicalDecodeError(f"canonical line file is empty or torn: {source}")
    documents: list[CanonicalDocument] = []
    for line_number, line in enumerate(encoded.splitlines(), start=1):
        if not line:
            raise CanonicalDecodeError(f"empty canonical line at {source}:{line_number}")
        try:
            documents.append(_decode_document(line, expected_domain=expected_domain))
        except CanonicalDecodeError as exc:
            raise CanonicalDecodeError(f"{source}:{line_number}: {exc}") from exc
    return tuple(documents)


__all__ = [
    "CanonicalDecodeError",
    "CanonicalDocument",
    "load_canonical_file",
    "load_canonical_lines",
]
