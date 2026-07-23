"""Dataset-scoped capture modality requirements.

The ignore set is deliberately a dataset contract rather than an episode
attribute.  A populated dataset must never contain episodes captured with
different missing-modality rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


CAPTURE_MODALITY_CONTRACT_SCHEMA_V1 = "dexteleop.capture_modalities.v1"
CAPTURE_MODALITY_MISSING_VALUE_POLICY_V1 = {
    "camera": "frame_absent_valid_false",
    "tactile": "value_absent_valid_false",
}


def _normalized_keys(values: Iterable[Any], *, label: str) -> tuple[str, ...]:
    keys: set[str] = set()
    for value in values:
        key = str(value).strip()
        if not key:
            raise ValueError(f"{label} must not contain an empty key")
        keys.add(key)
    return tuple(sorted(keys))


@dataclass(frozen=True)
class CaptureModalityContract:
    """Immutable missing-modality policy shared by every dataset episode."""

    ignored_camera_keys: tuple[str, ...] = ()
    ignored_tactile_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ignored_camera_keys",
            _normalized_keys(self.ignored_camera_keys, label="ignored_camera_keys"),
        )
        object.__setattr__(
            self,
            "ignored_tactile_keys",
            _normalized_keys(self.ignored_tactile_keys, label="ignored_tactile_keys"),
        )

    @property
    def is_complete(self) -> bool:
        return not self.ignored_camera_keys and not self.ignored_tactile_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPTURE_MODALITY_CONTRACT_SCHEMA_V1,
            "scope": "dataset",
            "ignored_modalities": {
                "camera": list(self.ignored_camera_keys),
                "tactile": list(self.ignored_tactile_keys),
            },
            "missing_value_policy": dict(CAPTURE_MODALITY_MISSING_VALUE_POLICY_V1),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptureModalityContract":
        schema = payload.get("schema_version")
        if schema != CAPTURE_MODALITY_CONTRACT_SCHEMA_V1:
            raise ValueError(
                "unsupported capture modality contract schema_version="
                f"{schema!r}"
            )
        if payload.get("scope", "dataset") != "dataset":
            raise ValueError("capture modality contract scope must be 'dataset'")
        if payload.get("missing_value_policy") != CAPTURE_MODALITY_MISSING_VALUE_POLICY_V1:
            raise ValueError(
                "capture modality contract missing_value_policy must be "
                f"{CAPTURE_MODALITY_MISSING_VALUE_POLICY_V1!r}"
            )
        ignored = payload.get("ignored_modalities", {})
        if not isinstance(ignored, Mapping):
            raise ValueError("ignored_modalities must be an object")
        camera = ignored.get("camera", ())
        tactile = ignored.get("tactile", ())
        if not isinstance(camera, (list, tuple)):
            raise ValueError("ignored_modalities.camera must be a list")
        if not isinstance(tactile, (list, tuple)):
            raise ValueError("ignored_modalities.tactile must be a list")
        return cls(
            ignored_camera_keys=tuple(camera),
            ignored_tactile_keys=tuple(tactile),
        )


__all__ = [
    "CAPTURE_MODALITY_CONTRACT_SCHEMA_V1",
    "CAPTURE_MODALITY_MISSING_VALUE_POLICY_V1",
    "CaptureModalityContract",
]
