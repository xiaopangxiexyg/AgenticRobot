"""Typed v2.3 causal observation-cut metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


CAUSAL_SNAPSHOT_SCHEMA_V1 = "dexteleop_causal_snapshot_v1"
CAUSAL_SNAPSHOT_METADATA_KEY = "causal_snapshot_contract"
CAUSAL_SNAPSHOT_DETAIL_METADATA_KEY = "causal_snapshot_contract_detail"
CAUSAL_SNAPSHOT_HASH_METADATA_KEY = "causal_snapshot_contract_hash"

LOCAL_WALL_CLOCK = "local_wall_time"
RECEIVER_WALL_CLOCK = "receiver_wall_time"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class CausalSnapshotContract:
    schema: str = CAUSAL_SNAPSHOT_SCHEMA_V1
    cutoff_field: str = "timestamp"
    eligibility: str = "source_timestamp<=T0_and_available_timestamp<=T0"
    available_clock: str = LOCAL_WALL_CLOCK
    modality_source_clock: Mapping[str, str] = field(
        default_factory=lambda: {
            "arm": LOCAL_WALL_CLOCK,
            "hand_telemetry": LOCAL_WALL_CLOCK,
            "tactile": LOCAL_WALL_CLOCK,
            "camera": RECEIVER_WALL_CLOCK,
        }
    )
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: {
            "camera_native_sensor_timestamp_role": "audit_only",
            "arm_source_clock_exceptions": {
                "vive_umi": [
                    "vive_sender_monotonic_calibrated_to_local_wall",
                    "vive_receiver_wall_fallback",
                ]
            },
            "missing_eligible_snapshot_policy": "invalid_or_drop_row_by_modality_requirement",
        }
    )

    def __post_init__(self) -> None:
        if str(self.schema) != CAUSAL_SNAPSHOT_SCHEMA_V1:
            raise ValueError(f"Unsupported causal snapshot schema {self.schema!r}.")
        if not str(self.cutoff_field).strip():
            raise ValueError("cutoff_field must be non-empty.")
        if str(self.eligibility) != "source_timestamp<=T0_and_available_timestamp<=T0":
            raise ValueError("v2.3 causal snapshot eligibility is fixed.")
        clocks = {str(key): str(value) for key, value in self.modality_source_clock.items()}
        required = {"arm", "hand_telemetry", "tactile", "camera"}
        if set(clocks) != required or any(not clocks[key] for key in required):
            raise ValueError(
                "modality_source_clock must define arm, hand_telemetry, tactile and camera."
            )
        object.__setattr__(self, "cutoff_field", str(self.cutoff_field))
        object.__setattr__(self, "available_clock", str(self.available_clock))
        object.__setattr__(self, "modality_source_clock", _freeze(clocks))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cutoff_field": self.cutoff_field,
            "eligibility": self.eligibility,
            "available_clock": self.available_clock,
            "modality_source_clock": _thaw(self.modality_source_clock),
            "metadata": _thaw(self.metadata),
        }

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def metadata_patch(self) -> dict[str, Any]:
        return {
            CAUSAL_SNAPSHOT_METADATA_KEY: self.schema,
            CAUSAL_SNAPSHOT_DETAIL_METADATA_KEY: self.to_dict(),
            CAUSAL_SNAPSHOT_HASH_METADATA_KEY: self.contract_hash,
        }


DEFAULT_CAUSAL_SNAPSHOT_CONTRACT = CausalSnapshotContract()


__all__ = [
    "CAUSAL_SNAPSHOT_DETAIL_METADATA_KEY",
    "CAUSAL_SNAPSHOT_HASH_METADATA_KEY",
    "CAUSAL_SNAPSHOT_METADATA_KEY",
    "CAUSAL_SNAPSHOT_SCHEMA_V1",
    "CausalSnapshotContract",
    "DEFAULT_CAUSAL_SNAPSHOT_CONTRACT",
    "LOCAL_WALL_CLOCK",
    "RECEIVER_WALL_CLOCK",
]
