"""Shared primitive checks; Progress domains retain their own error types."""

from __future__ import annotations

import math


def required_text(error_type: type[ValueError], value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{name} must be a non-empty string.")
    return value.strip()


def sha256_digest(error_type: type[ValueError], value: object, *, name: str) -> str:
    result = required_text(error_type, value, name=name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise error_type(f"{name} must be a lowercase SHA-256 digest.")
    return result


def finite_number(error_type: type[ValueError], value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise error_type(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise error_type(f"{name} must be finite.")
    return result


def positive_int(error_type: type[ValueError], value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise error_type(f"{name} must be a positive integer.")
    return value


def nonnegative_int(error_type: type[ValueError], value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise error_type(f"{name} must be a non-negative integer.")
    return value


def confidence(error_type: type[ValueError], value: object) -> float:
    result = finite_number(error_type, value, name="confidence")
    if result < 0.0 or result > 1.0:
        raise error_type("confidence must lie in [0, 1].")
    return result
