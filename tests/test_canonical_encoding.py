from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
    canonical_loads,
)


CANONICAL_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "control_core" / "canonical_encoding_v1.json"
)


def test_canonical_encoding_v1_golden_hashes() -> None:
    fixture = json.loads(CANONICAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    payloads = {
        "mapping_and_float32": {
            "b": [2, 3],
            "a": np.asarray([1.0, -0.0], dtype=np.float32),
        },
        "float_edges": {
            "scalar": float("nan"),
            "array": np.asarray([np.nan, -0.0, 1.5], dtype=np.float64),
        },
        "typed_sequence": {
            "list": [1, 2],
            "tuple": (1, 2),
            "optional": None,
            "bytes": b"abc",
        },
    }
    for name, expected in fixture["cases"].items():
        assert canonical_content_hash(payloads[name], domain=expected["domain"]) == expected["sha256"]


def test_canonical_encoding_is_map_order_and_transport_stable() -> None:
    first = {"b": [2, 3], "a": np.asarray([1.0, -0.0], dtype=np.float32)}
    second = {"a": np.asarray([1.0, 0.0], dtype=">f4"), "b": [2, 3]}

    assert canonical_bytes(first, domain="fixture") == canonical_bytes(second, domain="fixture")
    assert canonical_content_hash(first, domain="fixture") == canonical_content_hash(
        second, domain="fixture"
    )


def test_canonical_encoding_freezes_float_edge_cases_and_array_contract() -> None:
    assert canonical_content_hash(-0.0, domain="float") == canonical_content_hash(
        0.0, domain="float"
    )
    assert canonical_content_hash(float("nan"), domain="float") == canonical_content_hash(
        np.float64("nan"), domain="float"
    )
    assert canonical_content_hash(
        np.asarray([np.nan, -0.0], dtype=np.float32), domain="array"
    ) == canonical_content_hash(
        np.asarray([np.nan, 0.0], dtype=">f4"), domain="array"
    )
    assert canonical_content_hash(np.zeros((1, 2), dtype=np.float32), domain="array") != (
        canonical_content_hash(np.zeros((2,), dtype=np.float32), domain="array")
    )
    assert canonical_content_hash(np.zeros((2,), dtype=np.float32), domain="array") != (
        canonical_content_hash(np.zeros((2,), dtype=np.float64), domain="array")
    )
    with pytest.raises(ValueError, match="infinite"):
        canonical_bytes(float("inf"), domain="float")
    with pytest.raises(ValueError, match="infinite"):
        canonical_bytes(np.asarray([np.inf], dtype=np.float32), domain="array")


def test_canonical_encoding_is_typed_and_domain_separated() -> None:
    assert canonical_content_hash([1, 2], domain="typed") != canonical_content_hash(
        (1, 2), domain="typed"
    )
    assert canonical_content_hash({"value": 1}, domain="one") != canonical_content_hash(
        {"value": 1}, domain="two"
    )
    with pytest.raises(TypeError, match="explicit primitive payloads"):
        canonical_bytes(_Unsupported(value=1), domain="unsupported")
    with pytest.raises(TypeError, match="string keys"):
        canonical_bytes({1: "not-canonical"}, domain="mapping")
    assert canonical_content_hash(_Mode.READY, domain="enum") != canonical_content_hash(
        "ready", domain="enum"
    )
    quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert canonical_content_hash(quaternion, domain="quaternion") != canonical_content_hash(
        -quaternion, domain="quaternion"
    )


def test_canonical_encoding_round_trip_preserves_storage_primitives() -> None:
    payload = {
        "array": np.asarray([[1.0, np.nan]], dtype=np.float32),
        "numpy_int": np.int16(7),
        "tuple": (b"abc", None),
        "list": [True, 3.5],
    }
    encoded = canonical_bytes(payload, domain="round-trip")

    decoded = canonical_loads(encoded, expected_domain="round-trip")

    assert decoded["array"].dtype == np.dtype("<f4")
    np.testing.assert_equal(decoded["array"], payload["array"])
    assert isinstance(decoded["numpy_int"], np.int16)
    assert decoded["tuple"] == (b"abc", None)
    assert decoded["list"] == [True, 3.5]
    with pytest.raises(ValueError, match="domain mismatch"):
        canonical_loads(encoded, expected_domain="other-domain")


@dataclass
class _Unsupported:
    value: int


class _Mode(Enum):
    READY = "ready"
