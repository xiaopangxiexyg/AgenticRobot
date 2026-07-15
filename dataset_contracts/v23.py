"""DexTeleop v2.3 compatibility contract.

v2.3 keeps the v2.2 tensor layout and stores absolute poses on disk.  The
additional contract declares the pose source, raw/desk storage frame and the
fact that relative actions are a loader/deployment representation rather than
a new parquet physical layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .pose_frames import (
    POSE_FRAME_CONTRACT_METADATA_KEY,
    POSE_STORAGE_FRAME_DESK,
    PoseFrameContract,
    PoseFrameContractError,
    normalize_pose_storage_frame,
    pose_frame_contract_from_metadata,
)
from .v22 import PolicyIOLayout, policy_io_layout_from_metadata


DEXTELEOP_DATASET_SCHEMA_V23 = "dexteleop_dataset_v2.3"
V23_POSE_ACTION_MODE_METADATA_KEY = "arm_pose_action_mode"
V23_POSE_STORAGE_FRAME_METADATA_KEY = "pose_storage_frame"
V23_SOURCE_SCHEMA_METADATA_KEY = "v23_compatibility_source_schema"
V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY = "relative_action_supported"


@dataclass(frozen=True)
class V23DatasetContract:
    """Resolved v2.3 contract, including a v2.2 compatibility view."""

    policy_io_layout: PolicyIOLayout
    pose_frames: PoseFrameContract
    source_schema: str
    compatibility_mode: bool
    dataset_schema: str = DEXTELEOP_DATASET_SCHEMA_V23

    @property
    def pose_storage_frame(self) -> str:
        return self.pose_frames.pose_storage_frame

    @property
    def source_id(self) -> str:
        return self.pose_frames.source_id

    def metadata_patch(self) -> dict[str, Any]:
        return {
            "dexteleop_dataset_schema": self.dataset_schema,
            POSE_FRAME_CONTRACT_METADATA_KEY: self.pose_frames.to_dict(),
            V23_POSE_STORAGE_FRAME_METADATA_KEY: self.pose_storage_frame,
            V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY: True,
            V23_SOURCE_SCHEMA_METADATA_KEY: self.source_schema,
            V23_POSE_ACTION_MODE_METADATA_KEY: "absolute",
            "policy_io_layout_hash": self.policy_io_layout.layout_hash,
        }


def _active_sides(metadata: Mapping[str, Any], explicit: Sequence[str] | None) -> tuple[str, ...]:
    values = explicit if explicit is not None else metadata.get("active_sides")
    if values is None:
        return ("left", "right")
    sides = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if not sides or set(sides) - {"left", "right"}:
        raise PoseFrameContractError(f"Invalid active_sides for pose contract: {values!r}.")
    return tuple(side for side in ("left", "right") if side in sides)


def resolve_v23_dataset_contract(
    metadata: Mapping[str, Any],
    *,
    active_sides: Sequence[str] | None = None,
    allow_legacy_v21: bool = True,
    require_explicit_v23_pose_contract: bool = True,
) -> V23DatasetContract:
    """Resolve v2.3 or a v2.2-and-earlier raw compatibility view.

    A v2.3 dataset must carry an explicit pose frame contract.  Older datasets
    are treated as legacy RM65 raw data and receive a marked compatibility
    contract without modifying their rows.
    """

    info = dict(metadata)
    source_schema = str(info.get("dexteleop_dataset_schema") or "legacy_unversioned")
    is_v23 = source_schema == DEXTELEOP_DATASET_SCHEMA_V23
    if is_v23 and require_explicit_v23_pose_contract:
        if not isinstance(info.get(POSE_FRAME_CONTRACT_METADATA_KEY), Mapping):
            raise PoseFrameContractError(
                f"v2.3 metadata must include {POSE_FRAME_CONTRACT_METADATA_KEY!r}."
            )
    layout = policy_io_layout_from_metadata(info, allow_legacy_v21=allow_legacy_v21)
    frames = pose_frame_contract_from_metadata(
        info,
        active_sides=_active_sides(info, active_sides),
        allow_legacy_v22=not is_v23,
    )
    declared_storage_frame = info.get(V23_POSE_STORAGE_FRAME_METADATA_KEY)
    if declared_storage_frame is not None:
        normalized_storage_frame = normalize_pose_storage_frame(str(declared_storage_frame))
        if normalized_storage_frame != frames.pose_storage_frame:
            raise PoseFrameContractError(
                "v2.3 pose_storage_frame disagrees with pose_frame_contract: "
                f"{normalized_storage_frame!r} != {frames.pose_storage_frame!r}."
            )
    if is_v23 and frames.pose_storage_frame not in {"raw_absolute", POSE_STORAGE_FRAME_DESK}:
        raise PoseFrameContractError("v2.3 pose_storage_frame is invalid.")
    return V23DatasetContract(
        policy_io_layout=layout,
        pose_frames=frames,
        source_schema=source_schema,
        compatibility_mode=not is_v23,
    )


def v22_to_v23_compatibility_metadata(
    metadata: Mapping[str, Any],
    *,
    active_sides: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return an in-memory v2.3 metadata view while preserving source metadata."""

    contract = resolve_v23_dataset_contract(
        metadata,
        active_sides=active_sides,
        require_explicit_v23_pose_contract=False,
    )
    output = dict(metadata)
    output.update(contract.metadata_patch())
    output["postprocess_history"] = [
        *(
            list(metadata.get("postprocess_history", []))
            if isinstance(metadata.get("postprocess_history"), list)
            else []
        ),
        {
            "operation": "v22_to_v23_in_memory_pose_contract",
            "source_schema": contract.source_schema,
            "target_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            "compatibility_mode": True,
            "source_id": contract.source_id,
        },
    ]
    return output


__all__ = [
    "DEXTELEOP_DATASET_SCHEMA_V23",
    "V23DatasetContract",
    "V23_POSE_ACTION_MODE_METADATA_KEY",
    "V23_POSE_STORAGE_FRAME_METADATA_KEY",
    "V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY",
    "V23_SOURCE_SCHEMA_METADATA_KEY",
    "resolve_v23_dataset_contract",
    "v22_to_v23_compatibility_metadata",
]
