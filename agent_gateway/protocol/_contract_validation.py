"""Small closed validators shared by pure Agent Skill v2 contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SLOT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def required_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int = 512,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    if len(value) > maximum_length:
        raise ValueError(
            f"{field_name} must contain at most {maximum_length} characters."
        )
    return value


def optional_text(
    value: object | None,
    *,
    field_name: str,
    maximum_length: int = 512,
) -> str | None:
    if value is None:
        return None
    return required_text(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def identifier(value: object, *, field_name: str) -> str:
    normalized = required_text(value, field_name=field_name, maximum_length=256)
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} is not a valid identifier.")
    return normalized


def slot_identifier(value: object, *, field_name: str) -> str:
    normalized = required_text(value, field_name=field_name, maximum_length=64)
    if not _SLOT_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must use lowercase snake_case and start with a letter."
        )
    return normalized


def sha256_digest(value: object, *, field_name: str) -> str:
    normalized = required_text(value, field_name=field_name, maximum_length=64)
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return normalized


def nonnegative_integer(
    value: object,
    *,
    field_name: str,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return value


def positive_integer(
    value: object,
    *,
    field_name: str,
    maximum: int | None = None,
) -> int:
    normalized = nonnegative_integer(
        value,
        field_name=field_name,
        maximum=maximum,
    )
    if normalized == 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return normalized


def finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def enum_value(
    value: object,
    *,
    field_name: str,
    allowed: Sequence[str],
) -> str:
    normalized = required_text(value, field_name=field_name)
    if normalized not in allowed:
        raise ValueError(f"Unsupported {field_name}: {normalized!r}.")
    return normalized


def unique_sorted_text(
    values: object,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list.")
    normalized = tuple(
        required_text(value, field_name=field_name) for value in values
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return tuple(sorted(normalized))


def normalized_relative_path(value: object, *, field_name: str) -> str:
    normalized = required_text(value, field_name=field_name, maximum_length=1024)
    if "\\" in normalized:
        raise ValueError(f"{field_name} must use POSIX separators.")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() in {"", "."}
        or path.as_posix() != normalized
    ):
        raise ValueError(f"{field_name} must be a normalized relative POSIX path.")
    return normalized


def freeze_json(value: Any, *, field_name: str, depth: int = 0) -> Any:
    if depth > 16:
        raise ValueError(f"{field_name} exceeds maximum JSON depth 16.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} requires string object keys.")
        return MappingProxyType(
            {
                key: freeze_json(
                    value[key],
                    field_name=field_name,
                    depth=depth + 1,
                )
                for key in sorted(value)
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            freeze_json(item, field_name=field_name, depth=depth + 1)
            for item in value
        )
    raise TypeError(f"{field_name} must contain JSON-compatible values only.")


def freeze_json_object(value: Any, *, field_name: str) -> Mapping[str, Any]:
    frozen = freeze_json(value, field_name=field_name)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{field_name} must be a JSON object.")
    return frozen


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_size(value: Any, *, field_name: str) -> int:
    try:
        return len(
            json.dumps(
                thaw_json(value),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} is not bounded canonical JSON.") from error
