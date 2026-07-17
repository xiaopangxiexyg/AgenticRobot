"""Stable v2.3 model-facing action-chunk semantics.

This contract describes how an absolute-pose episode is presented to a model.
It does not choose episodes or read parquet, and it deliberately keeps padding
out of normalization statistics through an explicit mask.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .v22 import DatasetLayoutBindingError


POSE_ACTION_CHUNK_SCHEMA_V1 = "dexteleop_pose_action_chunk_v1"
POSE_ACTION_CHUNK_METADATA_KEY = "pose_action_chunk_contract"
POSE_ACTION_MODE_ABSOLUTE = "absolute"
POSE_ACTION_MODE_RELATIVE = "relative"
POSE_ACTION_MODES = (POSE_ACTION_MODE_ABSOLUTE, POSE_ACTION_MODE_RELATIVE)
POSE_CHUNK_TAIL_DROP = "drop"
POSE_CHUNK_TAIL_PAD_LAST = "pad_last"
POSE_CHUNK_TAIL_POLICIES = (POSE_CHUNK_TAIL_DROP, POSE_CHUNK_TAIL_PAD_LAST)


def _mode(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in POSE_ACTION_MODES:
        raise DatasetLayoutBindingError(
            f"{name} must be one of {POSE_ACTION_MODES}, got {value!r}."
        )
    return normalized


def _tail_policy(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in POSE_CHUNK_TAIL_POLICIES:
        raise DatasetLayoutBindingError(
            f"tail_policy must be one of {POSE_CHUNK_TAIL_POLICIES}, got {value!r}."
        )
    return normalized


@dataclass(frozen=True)
class PoseActionChunkContract:
    """Training/action-chunk view bound to a policy artifact.

    A sample has one observation at the chunk start and ``action_horizon``
    future actions.  Incomplete tails are dropped by default.  A model may
    explicitly choose ``pad_last`` to repeat the last already-converted
    action, set ``action_is_pad`` for those positions, and exclude them from
    normalization and loss.
    """

    action_horizon: int
    action_pose_mode: str = POSE_ACTION_MODE_ABSOLUTE
    last_action_pose_mode: str | None = None
    tail_policy: str = POSE_CHUNK_TAIL_DROP
    schema: str = POSE_ACTION_CHUNK_SCHEMA_V1

    def __post_init__(self) -> None:
        horizon = int(self.action_horizon)
        if horizon < 1:
            raise DatasetLayoutBindingError("action_horizon must be at least 1.")
        if str(self.schema) != POSE_ACTION_CHUNK_SCHEMA_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported pose action chunk schema {self.schema!r}."
            )
        action_mode = _mode(self.action_pose_mode, name="action_pose_mode")
        last_mode = _mode(
            self.last_action_pose_mode or action_mode,
            name="last_action_pose_mode",
        )
        object.__setattr__(self, "action_horizon", horizon)
        object.__setattr__(self, "action_pose_mode", action_mode)
        object.__setattr__(self, "last_action_pose_mode", last_mode)
        object.__setattr__(self, "tail_policy", _tail_policy(self.tail_policy))
        object.__setattr__(self, "schema", POSE_ACTION_CHUNK_SCHEMA_V1)

    @property
    def contract_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action_horizon": self.action_horizon,
            "action_pose_mode": self.action_pose_mode,
            "last_action_pose_mode": self.last_action_pose_mode,
            "absolute_pose_frame": "canonical_desk",
            "observation_state_pose_frame": "canonical_desk",
            "tail_policy": self.tail_policy,
            "padding_mask_key": "action_is_pad",
            "padding_value_policy": "repeat_last_converted_action",
            "relative_action_anchor": "chunk_start_actual_tcp",
            "relative_last_action_anchor": "current_actual_tcp",
            "normalization_excludes_padding": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseActionChunkContract":
        expected_constants = {
            "absolute_pose_frame": "canonical_desk",
            "observation_state_pose_frame": "canonical_desk",
            "padding_mask_key": "action_is_pad",
            "padding_value_policy": "repeat_last_converted_action",
            "relative_action_anchor": "chunk_start_actual_tcp",
            "relative_last_action_anchor": "current_actual_tcp",
            "normalization_excludes_padding": True,
        }
        for key, expected in expected_constants.items():
            if key in payload and payload[key] != expected:
                raise DatasetLayoutBindingError(
                    f"Pose action chunk {key} must be {expected!r}."
                )
        return cls(
            action_horizon=int(payload["action_horizon"]),
            action_pose_mode=str(payload.get("action_pose_mode", POSE_ACTION_MODE_ABSOLUTE)),
            last_action_pose_mode=(
                None
                if payload.get("last_action_pose_mode") is None
                else str(payload["last_action_pose_mode"])
            ),
            tail_policy=str(payload.get("tail_policy", POSE_CHUNK_TAIL_DROP)),
            schema=str(payload.get("schema", "")),
        )


__all__ = [
    "POSE_ACTION_CHUNK_METADATA_KEY",
    "POSE_ACTION_CHUNK_SCHEMA_V1",
    "POSE_ACTION_MODE_ABSOLUTE",
    "POSE_ACTION_MODE_RELATIVE",
    "POSE_ACTION_MODES",
    "POSE_CHUNK_TAIL_DROP",
    "POSE_CHUNK_TAIL_PAD_LAST",
    "POSE_CHUNK_TAIL_POLICIES",
    "PoseActionChunkContract",
]
