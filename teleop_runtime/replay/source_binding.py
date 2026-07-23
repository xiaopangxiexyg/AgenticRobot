"""Typed source binding for the existing GUI replay cursor.

The binding is created from the exact ``ReplayTrajectoryGroup`` admitted by
``ReplayExecutionController.preflight``.  It does not read the dataset a
second time, advance a cursor, construct a runtime emission, or authorize
motion.  Local paths remain validation context and are deliberately excluded
from semantic identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.replay import (
    REPLAY_SCHEDULE_FIXED_RATE,
    REPLAY_SCHEDULE_NATIVE,
    LegacyReplayContractDerivationReceipt,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
)
from dataset_interface.replay import (
    derive_replay_source_action_contract,
    replay_source_contract_from_metadata,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1
from robot_motion.replay.action_frame import (
    CANONICAL_DESK_ACTION_FRAME,
    replay_action_pose_frame,
)
from robot_motion.replay.trajectory import (
    REPLAY_TRAJECTORY_ACTION_POSE,
    ReplayTrajectoryGroup,
    normalize_replay_trajectory_source,
)
from robot_motion.runtime.immutable import deep_freeze, deep_thaw


REPLAY_GUI_SOURCE_BINDING_CONTRACT_ID_V1 = (
    "dexteleop.replay_gui_checked_source_binding.v1"
)
REPLAY_GUI_SOURCE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.replay_gui_checked_source_binding.v1"
)
REPLAY_GUI_SOURCE_CLOCK_DOMAIN_V1 = "local_monotonic"


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _merged_dataset_metadata(group: ReplayTrajectoryGroup) -> dict[str, Any]:
    """Flatten only source-owned metadata with explicit precedence.

    Dataset-level declarations are the base, episode declarations may narrow
    them, and direct loader facts fill locator fields.  Target/runtime defaults
    are intentionally absent.
    """

    source = _mapping(group.metadata.get("source_metadata"))
    merged: dict[str, Any] = {}
    merged.update(dict(_mapping(source.get("dataset_info"))))
    merged.update(dict(_mapping(source.get("episode_metadata"))))
    for key, value in source.items():
        if key not in {"dataset_info", "episode_metadata"}:
            merged[str(key)] = value
    return merged


def _selection_contract(
    active_sides: Sequence[str],
    *,
    include_end_effectors: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    sides = tuple(dict.fromkeys(str(side).strip() for side in active_sides))
    if not sides or any(side not in {"left", "right"} for side in sides):
        raise ValueError("GUI replay source binding requires explicit left/right sides.")
    components: list[str] = [f"{side}_arm" for side in sides]
    groups: list[str] = ["arms" if len(sides) == 2 else components[0]] * len(sides)
    coupled = ("arms",) if len(sides) == 2 else ()
    if include_end_effectors:
        for side in sides:
            component = f"{side}_hand"
            components.append(component)
            groups.append(component)
    return tuple(components), tuple(groups), coupled


def _action_spec_hashes(
    metadata: Mapping[str, Any],
    components: tuple[str, ...],
) -> dict[str, str]:
    """Read only explicitly declared source action-spec identities."""

    direct = _mapping(metadata.get("action_spec_hash_by_component"))
    values = {
        str(component): str(value).strip()
        for component, value in direct.items()
        if str(component) in components and str(value).strip()
    }
    for family, suffix in (
        ("arm_action_specs_by_side", "arm"),
        ("hand_action_specs_by_side", "hand"),
    ):
        by_side = _mapping(metadata.get(family))
        for side, payload in by_side.items():
            component = f"{str(side)}_{suffix}"
            if component not in components or component in values:
                continue
            spec = _mapping(payload)
            explicit = spec.get("action_spec_hash") or spec.get("spec_hash")
            if explicit is not None and str(explicit).strip():
                values[component] = str(explicit).strip()
    return values


@dataclass(frozen=True)
class ReplayGuiSourceBinding:
    """Immutable source evidence for one checked GUI replay group."""

    source_locator_id: str
    expected_source_path: str
    source_episode_id: str
    source_episode_index: int
    group_key: str
    source_start_index: int
    source_end_index: int
    frame_count: int
    source_hz: float
    clock_domain_id: str
    active_sides: tuple[str, ...]
    include_end_effectors: bool
    trajectory_source: str
    source_contract: ReplaySourceContract
    source_action_contract: ReplaySourceActionContract
    derivation_receipt: LegacyReplayContractDerivationReceipt
    schedule: ReplayScheduleSpec
    dataset_metadata: Mapping[str, Any] = field(repr=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_locator_id",
            "source_episode_id",
            "group_key",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        path = str(Path(self.expected_source_path).expanduser().resolve())
        object.__setattr__(self, "expected_source_path", path)
        for name in (
            "source_episode_index",
            "source_start_index",
            "source_end_index",
            "frame_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.source_end_index <= self.source_start_index:
            raise ValueError("Replay source binding requires a non-empty source interval.")
        if self.frame_count != self.source_end_index - self.source_start_index:
            raise ValueError("Replay source binding frame count/interval disagree.")
        rate = float(self.source_hz)
        if rate <= 0.0:
            raise ValueError("source_hz must be positive.")
        object.__setattr__(self, "source_hz", rate)
        sides = tuple(dict.fromkeys(_required(side, name="active_side") for side in self.active_sides))
        object.__setattr__(self, "active_sides", sides)
        trajectory = normalize_replay_trajectory_source(self.trajectory_source)
        if trajectory != REPLAY_TRAJECTORY_ACTION_POSE:
            raise ValueError(
                "The existing GUI source-frame seam is action-pose only; "
                "qpos_target replay does not emit ReplaySourceFrameEmission."
            )
        object.__setattr__(self, "trajectory_source", trajectory)
        if self.source_action_contract.action_space != BIMANUAL_EE_POSE_V1:
            raise ValueError("GUI replay source binding v1 requires bimanual_ee_pose_v1.")
        if self.source_contract.semantic_content_hash == "" or (
            self.source_action_contract.derivation_receipt_hash
            != self.derivation_receipt.semantic_content_hash
        ):
            raise ValueError("Replay source binding contract lineage is incomplete.")
        if self.schedule.source_start_index != self.source_start_index:
            raise ValueError("Replay schedule/source start indices disagree.")
        frozen_metadata = deep_freeze(dict(self.dataset_metadata))
        object.__setattr__(self, "dataset_metadata", frozen_metadata)
        payload = {
            "binding_contract_id": REPLAY_GUI_SOURCE_BINDING_CONTRACT_ID_V1,
            "source_locator_id": self.source_locator_id,
            "source_episode_id": self.source_episode_id,
            "source_episode_index": self.source_episode_index,
            "group_key": self.group_key,
            "source_start_index": self.source_start_index,
            "source_end_index": self.source_end_index,
            "frame_count": self.frame_count,
            "source_hz": self.source_hz,
            "clock_domain_id": self.clock_domain_id,
            "active_sides": self.active_sides,
            "include_end_effectors": bool(self.include_end_effectors),
            "trajectory_source": self.trajectory_source,
            "source_contract_hash": self.source_contract.semantic_content_hash,
            "source_action_contract_hash": (
                self.source_action_contract.semantic_content_hash
            ),
            "derivation_receipt_hash": self.derivation_receipt.semantic_content_hash,
            "schedule_hash": self.schedule.semantic_content_hash,
            "local_path_semantics": "validation_context_not_semantic_identity",
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=REPLAY_GUI_SOURCE_BINDING_HASH_DOMAIN_V1,
            ),
        )

    def metadata_copy(self) -> dict[str, Any]:
        return dict(deep_thaw(self.dataset_metadata))

    def to_status(self) -> dict[str, Any]:
        return {
            "available": True,
            "source_locator_id": self.source_locator_id,
            "source_episode_id": self.source_episode_id,
            "source_episode_index": self.source_episode_index,
            "group_key": self.group_key,
            "source_start_index": self.source_start_index,
            "source_end_index": self.source_end_index,
            "frame_count": self.frame_count,
            "source_hz": self.source_hz,
            "clock_domain_id": self.clock_domain_id,
            "active_sides": list(self.active_sides),
            "include_end_effectors": bool(self.include_end_effectors),
            "trajectory_source": self.trajectory_source,
            "source_contract_hash": self.source_contract.semantic_content_hash,
            "source_action_contract_hash": (
                self.source_action_contract.semantic_content_hash
            ),
            "interpretation_status": (
                self.source_action_contract.interpretation_status
            ),
            "schedule_hash": self.schedule.semantic_content_hash,
            "semantic_content_hash": self.semantic_content_hash,
            "authoritative": False,
            "cursor_owned": False,
        }


def build_replay_gui_source_binding(
    group: ReplayTrajectoryGroup,
    *,
    episode_id: str,
    active_sides: Sequence[str],
    include_end_effectors: bool,
    trajectory_source: str,
    clock_domain_id: str = REPLAY_GUI_SOURCE_CLOCK_DOMAIN_V1,
) -> ReplayGuiSourceBinding:
    """Build source evidence from the already checked lower-owned group."""

    if not isinstance(group, ReplayTrajectoryGroup):
        raise TypeError("group must be a ReplayTrajectoryGroup.")
    trajectory = normalize_replay_trajectory_source(trajectory_source)
    if trajectory != REPLAY_TRAJECTORY_ACTION_POSE:
        raise ValueError(
            "The existing GUI qpos_target branch has no source-frame emission seam."
        )
    metadata = _merged_dataset_metadata(group)
    action_field = str(group.metadata.get("action_column") or "action")
    components, selection_groups, coupled_groups = _selection_contract(
        active_sides,
        include_end_effectors=bool(include_end_effectors),
    )
    source_contract = replay_source_contract_from_metadata(
        metadata,
        source_path=group.source_path,
        episode_id=episode_id,
        action_field=action_field,
        source_action_space=BIMANUAL_EE_POSE_V1,
    )
    frame_kind = replay_action_pose_frame(group)
    if frame_kind == CANONICAL_DESK_ACTION_FRAME:
        action_metadata = _mapping(group.metadata.get("action_metadata"))
        serialized = action_metadata.get("pose_frame_contract")
        if not isinstance(serialized, Mapping):
            raise ValueError("Canonical-desk replay binding lacks a pose-frame contract.")
        parsed = PoseFrameContract.from_dict(serialized)
        if parsed.contract_hash != source_contract.source_frame_contract_hash:
            raise ValueError(
                "Replay source pose-frame hash disagrees with the serialized contract."
            )
    source_action_contract, derivation = derive_replay_source_action_contract(
        source_contract,
        component_ids=components,
        selection_groups=selection_groups,
        coupled_selection_groups=coupled_groups,
        action_spec_hash_by_component=_action_spec_hashes(metadata, components),
        units_convention_id=str(
            metadata.get("units_convention_id") or "si_m_rad_xyzw_v1"
        ),
    )
    frame_timing = _mapping(group.metadata.get("frame_timing"))
    schedule_mode = (
        REPLAY_SCHEDULE_NATIVE
        if str(frame_timing.get("source_timestamp_mode")) == "dataset"
        else REPLAY_SCHEDULE_FIXED_RATE
    )
    source_locator = str(
        source_contract.capture_id
        or source_contract.source_dataset_id
    )
    schedule = ReplayScheduleSpec(
        schedule_id=(
            f"replay-gui:{source_locator}:{source_contract.episode_id}:"
            f"{group.key}:{schedule_mode}"
        ),
        schedule_mode=schedule_mode,
        source_hz=float(group.action_hz),
        target_hz=None,
        resampling_policy="none",
        source_start_index=int(group.start_index),
    )
    return ReplayGuiSourceBinding(
        source_locator_id=(
            f"{source_locator}:{source_contract.episode_id}:{action_field}"
        ),
        expected_source_path=str(group.source_path),
        source_episode_id=str(episode_id),
        source_episode_index=int(group.source_episode_index),
        group_key=str(group.key),
        source_start_index=int(group.start_index),
        source_end_index=int(group.end_index),
        frame_count=int(group.frame_count),
        source_hz=float(group.action_hz),
        clock_domain_id=str(clock_domain_id),
        active_sides=tuple(str(side) for side in active_sides),
        include_end_effectors=bool(include_end_effectors),
        trajectory_source=trajectory,
        source_contract=source_contract,
        source_action_contract=source_action_contract,
        derivation_receipt=derivation,
        schedule=schedule,
        dataset_metadata=MappingProxyType(dict(metadata)),
    )


__all__ = [
    "REPLAY_GUI_SOURCE_BINDING_CONTRACT_ID_V1",
    "REPLAY_GUI_SOURCE_CLOCK_DOMAIN_V1",
    "ReplayGuiSourceBinding",
    "build_replay_gui_source_binding",
]
