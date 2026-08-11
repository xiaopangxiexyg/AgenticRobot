"""Bind immutable Prepared Skill intervals to an unmodified OpenPI checkout.

The adapter is intentionally installed only while OpenPI constructs a dataset.
It intersects OpenPI's ordinary quality-valid h-step index with the immutable
Skill intervals, then returns a small wrapper that makes augmentation identity
global across source datasets.  Raw Parquet, video, OpenPI source, and prepared
view artifacts remain read-only.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
)
from dataset_interface.logical_dataset import LOGICAL_EPISODE_REF_SCHEMA_V1
from dataset_interface.prepared_training import (
    LoadedPreparedTrainingView,
    PreparedTrainingViewError,
    load_prepared_training_view,
)


OPENPI_PREPARED_SKILL_BINDING_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_skill_binding.v1"
)
OPENPI_PREPARED_SKILL_BINDING_MANIFEST_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_skill_binding.manifest.v1"
)
OPENPI_PREPARED_SKILL_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_skill_binding.identity.v1"
)
OPENPI_PREPARED_PROMPT_BINDING_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_prompt_binding.v1"
)
OPENPI_PREPARED_PROMPT_BINDING_MANIFEST_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_prompt_binding.manifest.v1"
)
OPENPI_PREPARED_PROMPT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_prompt_binding.identity.v1"
)
OPENPI_PREPARED_PROMPT_VIEW_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_prompt_binding.view_set.v1"
)
OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V1 = (
    "deployments.openpi.prepared_skill_interval_adapter.v1"
)
OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V2 = (
    "deployments.openpi.prepared_skill_interval_adapter.v2"
)
OPENPI_SOURCE_MIXING_NATURAL_V1 = "natural_concat_shuffle_v1"
OPENPI_PROMPT_STAGE_SAMPLING_NATURAL_V1 = "natural_prompt_windows_v1"
OPENPI_PROMPT_STAGE_SAMPLING_BALANCED_V1 = (
    "balanced_min_windows_rotate_v1"
)


class OpenPIPreparedSkillError(ValueError):
    """Raised when OpenPI cannot consume one exact prepared Skill view."""


@dataclass(frozen=True)
class OpenPIPreparedEpisodeSelection:
    source_dataset_id: str
    source_dataset_root: Path
    source_episode_index: int
    organized_episode_ordinal: int
    episode_capture_identity_hash: str
    intervals: tuple[tuple[int, int], ...]

    def admits(self, start: int, horizon: int) -> bool:
        stop = int(start) + int(horizon)
        return any(begin <= start and stop <= end for begin, end in self.intervals)


@dataclass(frozen=True)
class OpenPIPreparedSkillBinding:
    view_root: Path
    organized_dataset_root: Path
    training_view_hash: str
    skill_id: str
    skill_training_contract_hash: str
    training_domain: str
    camera_slot_to_dataset_key: tuple[tuple[str, str], ...]
    action_horizon: int
    action_pose_mode: str
    source_mixing_policy: str
    episodes: tuple[OpenPIPreparedEpisodeSelection, ...]
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_content_hash(
                payload,
                domain=OPENPI_PREPARED_SKILL_BINDING_HASH_DOMAIN_V1,
            ),
        )

    @property
    def camera_slots(self) -> tuple[str, ...]:
        return tuple(slot for slot, _key in self.camera_slot_to_dataset_key)

    @property
    def source_dataset_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_dataset_id for item in self.episodes}))

    def selection_for(
        self,
        source_dataset_root: str | Path,
        source_episode_index: int,
    ) -> OpenPIPreparedEpisodeSelection | None:
        root = Path(source_dataset_root).expanduser().resolve()
        return next(
            (
                item
                for item in self.episodes
                if item.source_dataset_root == root
                and item.source_episode_index == int(source_episode_index)
            ),
            None,
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": OPENPI_PREPARED_SKILL_BINDING_SCHEMA_V1,
            "training_view_hash": self.training_view_hash,
            "skill_id": self.skill_id,
            "skill_training_contract_hash": self.skill_training_contract_hash,
            "training_domain": self.training_domain,
            "camera_slot_to_dataset_key": [
                {"model_slot": slot, "dataset_camera_key": key}
                for slot, key in self.camera_slot_to_dataset_key
            ],
            "action_horizon": self.action_horizon,
            "action_pose_mode": self.action_pose_mode,
            "source_mixing_policy": self.source_mixing_policy,
            "episode_count": len(self.episodes),
            "window_count": sum(
                end - begin - self.action_horizon + 1
                for episode in self.episodes
                for begin, end in episode.intervals
            ),
            "episodes": [
                {
                    "source_dataset_id": episode.source_dataset_id,
                    "source_episode_index": episode.source_episode_index,
                    "organized_episode_ordinal": (
                        episode.organized_episode_ordinal
                    ),
                    "episode_capture_identity_hash": (
                        episode.episode_capture_identity_hash
                    ),
                    "intervals": [
                        {"sample_begin": begin, "sample_end_exclusive": end}
                        for begin, end in episode.intervals
                    ],
                }
                for episode in self.episodes
            ],
            "adapter_implementation_id": (
                OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V2
            ),
        }
        if include_hash:
            payload["binding_hash"] = self.binding_hash
        return payload


@dataclass(frozen=True)
class OpenPIPreparedPromptSkill:
    """One immutable Skill/prompt member of a joint training binding."""

    stage_id: str
    skill_id: str
    prompt: str
    child_binding_hash: str
    training_view_hash: str
    skill_training_contract_hash: str
    episode_count: int
    window_count: int
    expected_openpi_window_count: int

    def __post_init__(self) -> None:
        for name in (
            "stage_id",
            "skill_id",
            "prompt",
            "child_binding_hash",
            "training_view_hash",
            "skill_training_contract_hash",
        ):
            if not str(getattr(self, name)).strip():
                raise OpenPIPreparedSkillError(
                    f"Prepared prompt Skill {name} must be non-empty."
                )
        if (
            int(self.episode_count) <= 0
            or int(self.window_count) <= 0
            or int(self.expected_openpi_window_count) <= 0
        ):
            raise OpenPIPreparedSkillError(
                "Prepared prompt Skill counts must be positive."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "skill_id": self.skill_id,
            "prompt": self.prompt,
            "child_binding_hash": self.child_binding_hash,
            "training_view_hash": self.training_view_hash,
            "skill_training_contract_hash": (
                self.skill_training_contract_hash
            ),
            "episode_count": self.episode_count,
            "window_count": self.window_count,
            "expected_openpi_window_count": (
                self.expected_openpi_window_count
            ),
        }


@dataclass(frozen=True)
class OpenPIPreparedPromptInterval:
    stage_id: str
    skill_id: str
    prompt: str
    sample_begin: int
    sample_end_exclusive: int

    def admits(self, start: int, horizon: int) -> bool:
        stop = int(start) + int(horizon)
        return self.sample_begin <= int(start) and stop <= self.sample_end_exclusive

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "skill_id": self.skill_id,
            "prompt": self.prompt,
            "sample_begin": self.sample_begin,
            "sample_end_exclusive": self.sample_end_exclusive,
        }


@dataclass(frozen=True)
class OpenPIPreparedPromptEpisodeSelection:
    source_dataset_id: str
    source_dataset_root: Path
    source_episode_index: int
    organized_episode_ordinal: int
    episode_capture_identity_hash: str
    intervals: tuple[OpenPIPreparedPromptInterval, ...]

    def admits(self, start: int, horizon: int) -> bool:
        return any(interval.admits(start, horizon) for interval in self.intervals)

    def prompt_for(self, start: int, horizon: int) -> str:
        return self.interval_for(start, horizon).prompt

    def interval_for(
        self,
        start: int,
        horizon: int,
    ) -> OpenPIPreparedPromptInterval:
        matches = tuple(
            interval
            for interval in self.intervals
            if interval.admits(start, horizon)
        )
        if len(matches) != 1:
            raise OpenPIPreparedSkillError(
                "Prepared prompt window must resolve exactly one prompt: "
                f"source={self.source_dataset_id!r} "
                f"episode={self.source_episode_index} start={int(start)} "
                f"matches={len(matches)}."
            )
        return matches[0]


@dataclass(frozen=True)
class OpenPIPreparedPromptBinding:
    """Joint immutable binding whose every legal window owns one prompt."""

    organized_dataset_root: Path
    prompt_skills: tuple[OpenPIPreparedPromptSkill, ...]
    camera_slot_to_dataset_key: tuple[tuple[str, str], ...]
    action_horizon: int
    action_pose_mode: str
    source_mixing_policy: str
    prompt_stage_sampling_policy: str
    prompt_stage_sampling_seed: int
    episodes: tuple[OpenPIPreparedPromptEpisodeSelection, ...]
    training_view_hash: str = field(init=False)
    binding_hash: str = field(init=False)
    _selection_by_key: Mapping[
        tuple[Path, int], OpenPIPreparedPromptEpisodeSelection
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.prompt_skills) < 2:
            raise OpenPIPreparedSkillError(
                "Prepared prompt binding requires at least two Skills."
            )
        for attribute in ("stage_id", "skill_id", "prompt"):
            values = tuple(
                str(getattr(skill, attribute)) for skill in self.prompt_skills
            )
            if len(set(values)) != len(values):
                raise OpenPIPreparedSkillError(
                    f"Prepared prompt binding has duplicate {attribute}."
                )
        if int(self.action_horizon) <= 0:
            raise OpenPIPreparedSkillError(
                "Prepared prompt binding action_horizon must be positive."
            )
        if self.prompt_stage_sampling_policy not in {
            OPENPI_PROMPT_STAGE_SAMPLING_NATURAL_V1,
            OPENPI_PROMPT_STAGE_SAMPLING_BALANCED_V1,
        }:
            raise OpenPIPreparedSkillError(
                "Prepared prompt binding has an unsupported stage sampler."
            )
        if not self.episodes:
            raise OpenPIPreparedSkillError(
                "Prepared prompt binding requires Episode selections."
            )
        keys: dict[tuple[Path, int], OpenPIPreparedPromptEpisodeSelection] = {}
        known_prompts = {skill.prompt for skill in self.prompt_skills}
        known_pairs = {
            (skill.stage_id, skill.skill_id, skill.prompt)
            for skill in self.prompt_skills
        }
        for episode in self.episodes:
            key = (
                episode.source_dataset_root.expanduser().resolve(),
                int(episode.source_episode_index),
            )
            if key in keys:
                raise OpenPIPreparedSkillError(
                    "Prepared prompt binding contains a duplicate source Episode."
                )
            keys[key] = episode
            if not episode.intervals:
                raise OpenPIPreparedSkillError(
                    "Prepared prompt Episode requires intervals."
                )
            intervals = tuple(
                sorted(
                    episode.intervals,
                    key=lambda value: (
                        value.sample_begin,
                        value.sample_end_exclusive,
                        value.stage_id,
                    ),
                )
            )
            for interval in intervals:
                if (
                    interval.stage_id,
                    interval.skill_id,
                    interval.prompt,
                ) not in known_pairs:
                    raise OpenPIPreparedSkillError(
                        "Prepared prompt interval is absent from the Skill map."
                    )
                if interval.prompt not in known_prompts:
                    raise OpenPIPreparedSkillError(
                        "Prepared prompt interval has an unknown prompt."
                    )
                if (
                    interval.sample_begin < 0
                    or interval.sample_end_exclusive - interval.sample_begin
                    < self.action_horizon
                ):
                    raise OpenPIPreparedSkillError(
                        "Prepared prompt interval is shorter than the action horizon."
                    )
            for earlier_index, earlier in enumerate(intervals):
                for later in intervals[earlier_index + 1 :]:
                    first_legal_end = (
                        earlier.sample_end_exclusive - self.action_horizon + 1
                    )
                    second_legal_end = (
                        later.sample_end_exclusive - self.action_horizon + 1
                    )
                    if max(earlier.sample_begin, later.sample_begin) < min(
                        first_legal_end, second_legal_end
                    ):
                        raise OpenPIPreparedSkillError(
                            "Prepared prompt intervals admit the same action window."
                        )
        object.__setattr__(self, "_selection_by_key", keys)
        view_hash = canonical_content_hash(
            {
                "children": [
                    {
                        "stage_id": skill.stage_id,
                        "training_view_hash": skill.training_view_hash,
                    }
                    for skill in self.prompt_skills
                ]
            },
            domain=OPENPI_PREPARED_PROMPT_VIEW_HASH_DOMAIN_V1,
        )
        object.__setattr__(self, "training_view_hash", view_hash)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=OPENPI_PREPARED_PROMPT_BINDING_HASH_DOMAIN_V1,
            ),
        )

    @property
    def camera_slots(self) -> tuple[str, ...]:
        return tuple(slot for slot, _key in self.camera_slot_to_dataset_key)

    @property
    def source_dataset_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_dataset_id for item in self.episodes}))

    def selection_for(
        self,
        source_dataset_root: str | Path,
        source_episode_index: int,
    ) -> OpenPIPreparedPromptEpisodeSelection | None:
        key = (
            Path(source_dataset_root).expanduser().resolve(),
            int(source_episode_index),
        )
        return self._selection_by_key.get(key)

    def prompt_for(
        self,
        source_dataset_root: str | Path,
        source_episode_index: int,
        start: int,
    ) -> str:
        selection = self.selection_for(source_dataset_root, source_episode_index)
        if selection is None:
            raise OpenPIPreparedSkillError(
                "OpenPI requested a prompt outside the prepared binding."
            )
        return selection.prompt_for(start, self.action_horizon)

    def stage_id_for(
        self,
        source_dataset_root: str | Path,
        source_episode_index: int,
        start: int,
    ) -> str:
        selection = self.selection_for(source_dataset_root, source_episode_index)
        if selection is None:
            raise OpenPIPreparedSkillError(
                "OpenPI requested a stage outside the prepared binding."
            )
        return selection.interval_for(start, self.action_horizon).stage_id

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": OPENPI_PREPARED_PROMPT_BINDING_SCHEMA_V1,
            "training_view_hash": self.training_view_hash,
            "prompt_skills": [skill.to_payload() for skill in self.prompt_skills],
            "camera_slot_to_dataset_key": [
                {"model_slot": slot, "dataset_camera_key": key}
                for slot, key in self.camera_slot_to_dataset_key
            ],
            "action_horizon": self.action_horizon,
            "action_pose_mode": self.action_pose_mode,
            "source_mixing_policy": self.source_mixing_policy,
            "prompt_stage_sampling_policy": (
                self.prompt_stage_sampling_policy
            ),
            "prompt_stage_sampling_seed": self.prompt_stage_sampling_seed,
            "episode_count": len(self.episodes),
            "window_count": sum(
                interval.sample_end_exclusive
                - interval.sample_begin
                - self.action_horizon
                + 1
                for episode in self.episodes
                for interval in episode.intervals
            ),
            "episodes": [
                {
                    "source_dataset_id": episode.source_dataset_id,
                    "source_episode_index": episode.source_episode_index,
                    "organized_episode_ordinal": episode.organized_episode_ordinal,
                    "episode_capture_identity_hash": (
                        episode.episode_capture_identity_hash
                    ),
                    "intervals": [
                        interval.to_payload() for interval in episode.intervals
                    ],
                }
                for episode in self.episodes
            ],
            "adapter_implementation_id": (
                "deployments.openpi.prepared_prompt_interval_adapter.v1"
            ),
        }
        if include_hash:
            payload["binding_hash"] = self.binding_hash
        return payload


def _normalized_camera_mapping(
    values: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    items = tuple(
        (str(slot).strip(), str(key).strip())
        for slot, key in (
            values.items() if isinstance(values, Mapping) else values
        )
    )
    if (
        not items
        or any(not slot or not key for slot, key in items)
        or len({slot for slot, _key in items}) != len(items)
        or len({key for _slot, key in items}) != len(items)
    ):
        raise OpenPIPreparedSkillError(
            "camera mapping must be non-empty and one-to-one."
        )
    return items


def load_openpi_prepared_skill_binding(
    view_root: str | Path,
    *,
    skill_id: str,
    training_domain: str = "NOMINAL",
    camera_slot_to_dataset_key: Mapping[str, str]
    | Sequence[tuple[str, str]] = (("base_0_rgb", "front"),),
    action_horizon: int = 16,
    action_pose_mode: str = "absolute",
    source_mixing_policy: str = OPENPI_SOURCE_MIXING_NATURAL_V1,
    organized_dataset_root: str | Path | None = None,
    verify_source_files: bool = True,
) -> OpenPIPreparedSkillBinding:
    """Load and validate one exact OpenPI-compatible Skill selection."""

    horizon = int(action_horizon)
    if horizon <= 0:
        raise OpenPIPreparedSkillError("action_horizon must be positive.")
    pose_mode = str(action_pose_mode).strip().lower()
    if pose_mode not in {"absolute", "relative"}:
        raise OpenPIPreparedSkillError("Unsupported action_pose_mode.")
    mixing_policy = str(source_mixing_policy).strip().lower()
    if mixing_policy != OPENPI_SOURCE_MIXING_NATURAL_V1:
        raise OpenPIPreparedSkillError(
            f"Unsupported source_mixing_policy: {mixing_policy!r}."
        )
    domain = str(training_domain).strip().upper()
    if domain not in {"NOMINAL", "FAILURE"}:
        raise OpenPIPreparedSkillError("Unsupported training_domain.")
    target_skill = str(skill_id).strip()
    if not target_skill:
        raise OpenPIPreparedSkillError("skill_id must be non-empty.")
    cameras = _normalized_camera_mapping(camera_slot_to_dataset_key)
    try:
        loaded = load_prepared_training_view(
            view_root,
            organized_dataset_root=organized_dataset_root,
            verify_source_files=verify_source_files,
        )
    except PreparedTrainingViewError as exc:
        raise OpenPIPreparedSkillError(str(exc)) from exc
    if loaded.skill_intervals is None:
        raise OpenPIPreparedSkillError(
            "Prepared view has no Skill projection intervals."
        )
    capability = loaded.manifest.get("skill_projection")
    if not isinstance(capability, Mapping):
        raise OpenPIPreparedSkillError(
            "Prepared view has no bound Skill capability."
        )
    skill_payload = capability.get("skill_contract")
    if not isinstance(skill_payload, Mapping):
        raise OpenPIPreparedSkillError("Prepared Skill contract is missing.")
    if str(skill_payload.get("skill_id", "")) != target_skill:
        raise OpenPIPreparedSkillError(
            "Prepared Skill id differs from the requested Skill."
        )
    skill_hash = str(skill_payload.get("semantic_content_hash", ""))

    episode_rows = loaded.episodes.to_pylist()
    by_ordinal = {
        int(row["organized_episode_ordinal"]): row for row in episode_rows
    }
    grouped: dict[int, list[tuple[int, int]]] = {}
    for row in loaded.skill_intervals.to_pylist():
        if str(row["skill_id"]) != target_skill:
            raise OpenPIPreparedSkillError(
                "Prepared interval table contains a foreign Skill id."
            )
        if str(row["training_domain"]).upper() != domain:
            continue
        begin = int(row["sample_begin"])
        end = int(row["sample_end_exclusive"])
        if begin < 0 or end - begin < horizon:
            continue
        grouped.setdefault(int(row["organized_episode_ordinal"]), []).append(
            (begin, end)
        )
    if not grouped:
        raise OpenPIPreparedSkillError(
            "Prepared Skill has no interval long enough for the action horizon."
        )

    selections: list[OpenPIPreparedEpisodeSelection] = []
    requested_camera_keys = {key for _slot, key in cameras}
    missing_window_ordinals = sorted(set(by_ordinal) - set(grouped))
    if missing_window_ordinals:
        raise OpenPIPreparedSkillError(
            "Every selected Episode must contribute at least one complete "
            f"Skill window; missing ordinals={missing_window_ordinals[:20]}."
        )
    for ordinal in sorted(grouped):
        row = by_ordinal.get(ordinal)
        if row is None:
            raise OpenPIPreparedSkillError(
                "Skill interval references an unknown Episode ordinal."
            )
        available_cameras = {str(key) for key in row["camera_keys"]}
        if not requested_camera_keys.issubset(available_cameras):
            raise OpenPIPreparedSkillError(
                "Prepared Episode does not satisfy the requested camera "
                f"signature: ordinal={ordinal}."
            )
        source_id = str(row["source_dataset_id"])
        source_root = loaded.source_datasets.get(source_id)
        if source_root is None:
            raise OpenPIPreparedSkillError(
                "Prepared Episode references an unknown source dataset."
            )
        intervals = tuple(sorted(set(grouped[ordinal])))
        for earlier, later in zip(intervals, intervals[1:]):
            if later[0] < earlier[1]:
                raise OpenPIPreparedSkillError(
                    "Prepared Skill intervals overlap within one Episode."
                )
        selections.append(
            OpenPIPreparedEpisodeSelection(
                source_dataset_id=source_id,
                source_dataset_root=source_root.resolve(),
                source_episode_index=int(row["source_episode_index"]),
                organized_episode_ordinal=ordinal,
                episode_capture_identity_hash=str(
                    row["episode_capture_identity_hash"]
                ),
                intervals=intervals,
            )
        )
    return OpenPIPreparedSkillBinding(
        view_root=loaded.view_root,
        organized_dataset_root=loaded.organized_dataset_root,
        training_view_hash=loaded.training_view_hash,
        skill_id=target_skill,
        skill_training_contract_hash=skill_hash,
        training_domain=domain,
        camera_slot_to_dataset_key=cameras,
        action_horizon=horizon,
        action_pose_mode=pose_mode,
        source_mixing_policy=mixing_policy,
        episodes=tuple(selections),
    )


def compose_openpi_prepared_prompt_binding(
    stages: Sequence[tuple[str, str, OpenPIPreparedSkillBinding]],
    *,
    organized_dataset_root: str | Path,
    expected_openpi_window_counts: Mapping[str, int] | None = None,
    prompt_stage_sampling_policy: str = (
        OPENPI_PROMPT_STAGE_SAMPLING_NATURAL_V1
    ),
    prompt_stage_sampling_seed: int = 20260811,
) -> OpenPIPreparedPromptBinding:
    """Compose disjoint single-Skill bindings without copying source data."""

    values = tuple(stages)
    if len(values) < 2:
        raise OpenPIPreparedSkillError(
            "Prepared prompt composition requires at least two stages."
        )
    reference = values[0][2]
    for _stage_id, _prompt, binding in values[1:]:
        actual = (
            binding.camera_slot_to_dataset_key,
            binding.action_horizon,
            binding.action_pose_mode,
            binding.source_mixing_policy,
        )
        expected = (
            reference.camera_slot_to_dataset_key,
            reference.action_horizon,
            reference.action_pose_mode,
            reference.source_mixing_policy,
        )
        if actual != expected:
            raise OpenPIPreparedSkillError(
                "Prepared prompt child bindings have incompatible model inputs."
            )

    openpi_counts = {
        str(stage_id): int(
            binding.to_payload()["window_count"]
            if expected_openpi_window_counts is None
            else expected_openpi_window_counts[str(stage_id)]
        )
        for stage_id, _prompt, binding in values
    }
    if any(count <= 0 for count in openpi_counts.values()):
        raise OpenPIPreparedSkillError(
            "Prepared prompt OpenPI window counts must be positive."
        )
    prompt_skills = tuple(
        OpenPIPreparedPromptSkill(
            stage_id=str(stage_id).strip(),
            skill_id=binding.skill_id,
            prompt=str(prompt).strip(),
            child_binding_hash=binding.binding_hash,
            training_view_hash=binding.training_view_hash,
            skill_training_contract_hash=binding.skill_training_contract_hash,
            episode_count=len(binding.episodes),
            window_count=int(binding.to_payload()["window_count"]),
            expected_openpi_window_count=openpi_counts[str(stage_id)],
        )
        for stage_id, prompt, binding in values
    )
    prompt_by_binding_hash = {
        skill.child_binding_hash: skill for skill in prompt_skills
    }
    grouped: dict[
        tuple[str, Path, int],
        tuple[str, list[OpenPIPreparedPromptInterval]],
    ] = {}
    for _stage_id, _prompt, binding in values:
        prompt_skill = prompt_by_binding_hash[binding.binding_hash]
        for episode in binding.episodes:
            key = (
                episode.source_dataset_id,
                episode.source_dataset_root.expanduser().resolve(),
                episode.source_episode_index,
            )
            current = grouped.get(key)
            if current is None:
                current = (episode.episode_capture_identity_hash, [])
                grouped[key] = current
            elif current[0] != episode.episode_capture_identity_hash:
                raise OpenPIPreparedSkillError(
                    "Prepared prompt children disagree on Episode identity."
                )
            current[1].extend(
                OpenPIPreparedPromptInterval(
                    stage_id=prompt_skill.stage_id,
                    skill_id=prompt_skill.skill_id,
                    prompt=prompt_skill.prompt,
                    sample_begin=begin,
                    sample_end_exclusive=end,
                )
                for begin, end in episode.intervals
            )
    episodes = tuple(
        OpenPIPreparedPromptEpisodeSelection(
            source_dataset_id=source_id,
            source_dataset_root=source_root,
            source_episode_index=source_episode_index,
            organized_episode_ordinal=ordinal,
            episode_capture_identity_hash=identity_and_intervals[0],
            intervals=tuple(identity_and_intervals[1]),
        )
        for ordinal, (
            (source_id, source_root, source_episode_index),
            identity_and_intervals,
        ) in enumerate(
            sorted(
                grouped.items(),
                key=lambda item: (item[0][0], item[0][2]),
            )
        )
    )
    return OpenPIPreparedPromptBinding(
        organized_dataset_root=(
            Path(organized_dataset_root).expanduser().resolve()
        ),
        prompt_skills=prompt_skills,
        camera_slot_to_dataset_key=reference.camera_slot_to_dataset_key,
        action_horizon=reference.action_horizon,
        action_pose_mode=reference.action_pose_mode,
        source_mixing_policy=reference.source_mixing_policy,
        prompt_stage_sampling_policy=str(prompt_stage_sampling_policy),
        prompt_stage_sampling_seed=int(prompt_stage_sampling_seed),
        episodes=episodes,
    )


def load_openpi_prepared_skill_binding_manifest(
    binding_root: str | Path,
) -> OpenPIPreparedSkillBinding:
    """Reconstruct and hash-verify one materialized single-Skill binding."""

    root = Path(binding_root).expanduser().resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    binding_payload = payload.get("binding")
    source_records = payload.get("source_manifests")
    if not isinstance(binding_payload, Mapping) or not isinstance(
        source_records, list
    ):
        raise OpenPIPreparedSkillError(
            "Prepared Skill binding manifest is incomplete."
        )
    if binding_payload.get("schema") != OPENPI_PREPARED_SKILL_BINDING_SCHEMA_V1:
        raise OpenPIPreparedSkillError(
            "Prepared Skill binding manifest has an unsupported schema."
        )
    roots = {
        str(record.get("source_dataset_id", "")): Path(
            str(record.get("source_dataset_root", ""))
        ).expanduser().resolve()
        for record in source_records
        if isinstance(record, Mapping)
    }
    organized_root = root.parents[2]
    training_view_hash = str(binding_payload.get("training_view_hash", ""))
    binding = OpenPIPreparedSkillBinding(
        view_root=(
            organized_root
            / "derived/training_views"
            / training_view_hash
        ).resolve(),
        organized_dataset_root=organized_root,
        training_view_hash=training_view_hash,
        skill_id=str(binding_payload.get("skill_id", "")),
        skill_training_contract_hash=str(
            binding_payload.get("skill_training_contract_hash", "")
        ),
        training_domain=str(binding_payload.get("training_domain", "")),
        camera_slot_to_dataset_key=tuple(
            (
                str(item["model_slot"]),
                str(item["dataset_camera_key"]),
            )
            for item in binding_payload.get("camera_slot_to_dataset_key", ())
            if isinstance(item, Mapping)
        ),
        action_horizon=int(binding_payload.get("action_horizon", 0)),
        action_pose_mode=str(binding_payload.get("action_pose_mode", "")),
        source_mixing_policy=str(
            binding_payload.get("source_mixing_policy", "")
        ),
        episodes=tuple(
            OpenPIPreparedEpisodeSelection(
                source_dataset_id=str(item["source_dataset_id"]),
                source_dataset_root=roots[str(item["source_dataset_id"])],
                source_episode_index=int(item["source_episode_index"]),
                organized_episode_ordinal=int(
                    item["organized_episode_ordinal"]
                ),
                episode_capture_identity_hash=str(
                    item["episode_capture_identity_hash"]
                ),
                intervals=tuple(
                    (
                        int(interval["sample_begin"]),
                        int(interval["sample_end_exclusive"]),
                    )
                    for interval in item.get("intervals", ())
                    if isinstance(interval, Mapping)
                ),
            )
            for item in binding_payload.get("episodes", ())
            if isinstance(item, Mapping)
        ),
    )
    if binding.to_payload() != dict(binding_payload):
        raise OpenPIPreparedSkillError(
            "Prepared Skill binding manifest fails identity verification."
        )
    # Also require every per-source Episode inventory to remain present. The
    # source media itself stays read-only and is validated when each child
    # binding is originally prepared.
    binding_source_manifests(binding, root)
    return binding


def load_openpi_prepared_prompt_binding(
    binding_root: str | Path,
) -> OpenPIPreparedPromptBinding:
    """Load and hash-verify one materialized prompt binding manifest."""

    root = Path(binding_root).expanduser().resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    binding_payload = payload.get("binding")
    source_records = payload.get("source_manifests")
    if not isinstance(binding_payload, Mapping) or not isinstance(
        source_records, list
    ):
        raise OpenPIPreparedSkillError(
            "Prepared prompt binding manifest is incomplete."
        )
    if binding_payload.get("schema") != OPENPI_PREPARED_PROMPT_BINDING_SCHEMA_V1:
        raise OpenPIPreparedSkillError(
            "Prepared prompt binding manifest has an unsupported schema."
        )
    roots = {
        str(record.get("source_dataset_id", "")): Path(
            str(record.get("source_dataset_root", ""))
        ).expanduser().resolve()
        for record in source_records
        if isinstance(record, Mapping)
    }
    prompt_skills = tuple(
        OpenPIPreparedPromptSkill(
            stage_id=str(item["stage_id"]),
            skill_id=str(item["skill_id"]),
            prompt=str(item["prompt"]),
            child_binding_hash=str(item["child_binding_hash"]),
            training_view_hash=str(item["training_view_hash"]),
            skill_training_contract_hash=str(
                item["skill_training_contract_hash"]
            ),
            episode_count=int(item["episode_count"]),
            window_count=int(item["window_count"]),
            expected_openpi_window_count=int(
                item["expected_openpi_window_count"]
            ),
        )
        for item in binding_payload.get("prompt_skills", ())
        if isinstance(item, Mapping)
    )
    episodes = tuple(
        OpenPIPreparedPromptEpisodeSelection(
            source_dataset_id=str(item["source_dataset_id"]),
            source_dataset_root=roots[str(item["source_dataset_id"])],
            source_episode_index=int(item["source_episode_index"]),
            organized_episode_ordinal=int(item["organized_episode_ordinal"]),
            episode_capture_identity_hash=str(
                item["episode_capture_identity_hash"]
            ),
            intervals=tuple(
                OpenPIPreparedPromptInterval(
                    stage_id=str(interval["stage_id"]),
                    skill_id=str(interval["skill_id"]),
                    prompt=str(interval["prompt"]),
                    sample_begin=int(interval["sample_begin"]),
                    sample_end_exclusive=int(
                        interval["sample_end_exclusive"]
                    ),
                )
                for interval in item.get("intervals", ())
                if isinstance(interval, Mapping)
            ),
        )
        for item in binding_payload.get("episodes", ())
        if isinstance(item, Mapping)
    )
    binding = OpenPIPreparedPromptBinding(
        organized_dataset_root=root.parents[2],
        prompt_skills=prompt_skills,
        camera_slot_to_dataset_key=tuple(
            (
                str(item["model_slot"]),
                str(item["dataset_camera_key"]),
            )
            for item in binding_payload.get("camera_slot_to_dataset_key", ())
            if isinstance(item, Mapping)
        ),
        action_horizon=int(binding_payload.get("action_horizon", 0)),
        action_pose_mode=str(binding_payload.get("action_pose_mode", "")),
        source_mixing_policy=str(
            binding_payload.get("source_mixing_policy", "")
        ),
        prompt_stage_sampling_policy=str(
            binding_payload.get("prompt_stage_sampling_policy", "")
        ),
        prompt_stage_sampling_seed=int(
            binding_payload.get("prompt_stage_sampling_seed", 0)
        ),
        episodes=episodes,
    )
    if binding.to_payload() != dict(binding_payload):
        raise OpenPIPreparedSkillError(
            "Prepared prompt binding manifest fails identity verification."
        )
    return binding


def _safe_source_filename(source_id: str, ordinal: int) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in source_id
    ).strip("_")
    return f"source_{ordinal:02d}_{normalized or 'dataset'}.jsonl"


def materialize_openpi_prepared_skill_binding(
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    output_root: str | Path,
) -> Path:
    """Write immutable Episode manifests beside, never inside, source data."""

    root = Path(output_root).expanduser().resolve()
    parent = root / "derived" / "openpi_prepared_skill_bindings"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / binding.binding_hash
    if final.is_dir():
        declared = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        if declared.get("binding") != binding.to_payload():
            raise OpenPIPreparedSkillError(
                "Existing OpenPI binding directory has conflicting content."
            )
        return final
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=str(parent)))
    try:
        by_source: dict[
            str,
            list[
                OpenPIPreparedEpisodeSelection
                | OpenPIPreparedPromptEpisodeSelection
            ],
        ] = {}
        for episode in binding.episodes:
            by_source.setdefault(episode.source_dataset_id, []).append(episode)
        source_manifests: list[dict[str, object]] = []
        for ordinal, source_id in enumerate(sorted(by_source)):
            filename = _safe_source_filename(source_id, ordinal)
            path = staging / filename
            episodes = sorted(
                by_source[source_id],
                key=lambda value: value.source_episode_index,
            )
            with path.open("w", encoding="utf-8") as handle:
                for episode in episodes:
                    record = {
                        "schema": LOGICAL_EPISODE_REF_SCHEMA_V1,
                        "source_dataset_root": str(episode.source_dataset_root),
                        "source_episode_index": episode.source_episode_index,
                        "episode_identity": (
                            episode.episode_capture_identity_hash
                        ),
                        "camera_slots": list(binding.camera_slots),
                        "view_profile": (
                            "prepared_prompt"
                            if isinstance(binding, OpenPIPreparedPromptBinding)
                            else "prepared_skill"
                        ),
                        "quality_class": "strict_success_non_abnormal",
                        "prepared_training_view_hash": (
                            binding.training_view_hash
                        ),
                        (
                            "openpi_prepared_prompt_binding_hash"
                            if isinstance(binding, OpenPIPreparedPromptBinding)
                            else "openpi_prepared_skill_binding_hash"
                        ): binding.binding_hash,
                    }
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            source_manifests.append(
                {
                    "source_dataset_id": source_id,
                    "source_dataset_root": str(episodes[0].source_dataset_root),
                    "episode_manifest": filename,
                    "episode_count": len(episodes),
                }
            )
        payload = binding.to_payload()
        payload["source_manifests"] = source_manifests
        manifest_domain = (
            OPENPI_PREPARED_PROMPT_BINDING_MANIFEST_DOMAIN_V1
            if isinstance(binding, OpenPIPreparedPromptBinding)
            else OPENPI_PREPARED_SKILL_BINDING_MANIFEST_DOMAIN_V1
        )
        (staging / "manifest.cjson").write_bytes(
            canonical_bytes(
                payload,
                domain=manifest_domain,
            )
        )
        with (staging / "manifest.cjson").open("rb") as handle:
            os.fsync(handle.fileno())
        # A plain JSON copy is convenient for OpenPI-side shell audits.
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "binding": binding.to_payload(),
                    "source_manifests": source_manifests,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with (staging / "manifest.json").open("rb") as handle:
            os.fsync(handle.fileno())
        os.rename(staging, final)
        staging = final
        return final
    finally:
        if staging.exists() and staging.name.startswith(".building-"):
            shutil.rmtree(staging)


def binding_source_manifests(
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    binding_root: str | Path,
) -> tuple[tuple[str, Path, Path], ...]:
    root = Path(binding_root).expanduser().resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("binding") != binding.to_payload():
        raise OpenPIPreparedSkillError(
            "OpenPI binding manifest differs from the loaded binding."
        )
    source_records = payload.get("source_manifests")
    if not isinstance(source_records, list):
        raise OpenPIPreparedSkillError(
            "OpenPI binding has no source manifest inventory."
        )
    record_by_id = {
        str(record.get("source_dataset_id", "")): record
        for record in source_records
        if isinstance(record, Mapping)
    }
    values: list[tuple[str, Path, Path]] = []
    for source_id in binding.source_dataset_ids:
        record = record_by_id.get(source_id)
        if record is None:
            raise OpenPIPreparedSkillError(
                f"Missing Episode manifest for source {source_id}."
            )
        source_root = Path(str(record.get("source_dataset_root", ""))).resolve()
        manifest = root / str(record.get("episode_manifest", ""))
        if not manifest.is_file():
            raise OpenPIPreparedSkillError(
                f"Missing Episode manifest file for source {source_id}."
            )
        values.append((source_id, source_root, manifest))
    return tuple(values)


def _sample_locator(dataset: Any, index: int) -> tuple[Path, int, int]:
    if hasattr(dataset, "_mix_index") and hasattr(dataset, "_datasets"):
        source_position, child_index = dataset._mix_index[int(index)]
        child = dataset._datasets[int(source_position)]
        episode_position, row_index = child._index[int(child_index)]
    else:
        child = dataset
        episode_position, row_index = child._index[int(index)]
    episode = child._episodes[int(episode_position)]
    return (
        Path(child.dataset_root).expanduser().resolve(),
        int(episode.episode_index),
        int(row_index),
    )


class OpenPIPreparedSkillDataset:
    """Pickle-safe wrapper that makes sample identity source-global."""

    def __init__(
        self,
        dataset: Any,
        binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
        state_augmentation_key: str,
        post_identity_transform: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.binding = binding
        self.state_augmentation_key = str(state_augmentation_key)
        self.post_identity_transform = post_identity_transform
        self._epoch = 0
        self._raw_indices_by_stage: dict[str, tuple[int, ...]] | None = None
        self._index_map: tuple[int, ...] | None = None
        self._balanced_composition: dict[str, int] | None = None
        if (
            isinstance(binding, OpenPIPreparedPromptBinding)
            and binding.prompt_stage_sampling_policy
            == OPENPI_PROMPT_STAGE_SAMPLING_BALANCED_V1
        ):
            grouped: dict[str, list[int]] = {
                skill.stage_id: [] for skill in binding.prompt_skills
            }
            for raw_index in range(len(dataset)):
                source_root, episode_index, row_index = _sample_locator(
                    dataset,
                    raw_index,
                )
                stage_id = binding.stage_id_for(
                    source_root,
                    episode_index,
                    row_index,
                )
                grouped[stage_id].append(raw_index)
            actual_counts = {
                stage_id: len(indices) for stage_id, indices in grouped.items()
            }
            expected_counts = {
                skill.stage_id: skill.expected_openpi_window_count
                for skill in binding.prompt_skills
            }
            if actual_counts != expected_counts:
                raise OpenPIPreparedSkillError(
                    "OpenPI prompt-stage sample counts differ from the binding: "
                    f"actual={actual_counts} expected={expected_counts}."
                )
            self._raw_indices_by_stage = {
                stage_id: tuple(indices)
                for stage_id, indices in grouped.items()
            }
            self._rebuild_balanced_index(0)

    def __len__(self) -> int:
        return (
            len(self.dataset)
            if self._index_map is None
            else len(self._index_map)
        )

    def _raw_index(self, index: int) -> int:
        normalized = int(index)
        return (
            normalized
            if self._index_map is None
            else int(self._index_map[normalized])
        )

    def _rebuild_balanced_index(self, epoch: int) -> None:
        if self._raw_indices_by_stage is None:
            return
        assert isinstance(self.binding, OpenPIPreparedPromptBinding)
        target = min(len(values) for values in self._raw_indices_by_stage.values())
        selected_by_stage: list[np.ndarray] = []
        for stage_ordinal, skill in enumerate(self.binding.prompt_skills):
            values = np.asarray(
                self._raw_indices_by_stage[skill.stage_id],
                dtype=np.int64,
            )
            random = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.binding.prompt_stage_sampling_seed,
                        int(epoch),
                        stage_ordinal,
                    ]
                )
            )
            selected_by_stage.append(random.permutation(values)[:target])
        self._index_map = tuple(
            int(selected_by_stage[stage][position])
            for position in range(target)
            for stage in range(len(selected_by_stage))
        )
        composition: dict[str, int] = {}
        source_id_by_root = {
            episode.source_dataset_root: episode.source_dataset_id
            for episode in self.binding.episodes
        }
        for raw_index in self._index_map:
            source_root, _episode_index, _row_index = _sample_locator(
                self.dataset,
                raw_index,
            )
            selection_source = source_id_by_root[source_root]
            composition[selection_source] = composition.get(selection_source, 0) + 1
        self._balanced_composition = composition

    def prepared_sample_identity(self, index: int) -> tuple[int, int]:
        index = self._raw_index(index)
        source_root, episode_index, row_index = _sample_locator(
            self.dataset,
            int(index),
        )
        selected = self.binding.selection_for(source_root, episode_index)
        if selected is None or not selected.admits(
            row_index,
            self.binding.action_horizon,
        ):
            raise OpenPIPreparedSkillError(
                "OpenPI requested a sample outside the prepared Skill binding."
            )
        return int(selected.organized_episode_ordinal), int(row_index)

    def prepared_prompt_stage_indices(self) -> dict[str, int]:
        """Return one deterministic virtual sample index for every prompt stage."""

        if not isinstance(self.binding, OpenPIPreparedPromptBinding):
            return {}
        result: dict[str, int] = {}
        for virtual_index in range(len(self)):
            raw_index = self._raw_index(virtual_index)
            source_root, episode_index, row_index = _sample_locator(
                self.dataset,
                raw_index,
            )
            stage_id = self.binding.stage_id_for(
                source_root,
                episode_index,
                row_index,
            )
            result.setdefault(stage_id, virtual_index)
            if len(result) == len(self.binding.prompt_skills):
                break
        if set(result) != {skill.stage_id for skill in self.binding.prompt_skills}:
            raise OpenPIPreparedSkillError(
                "OpenPI prepared dataset does not expose every prompt stage."
            )
        return result

    def __getitem__(self, index: int) -> Any:
        index = self._raw_index(index)
        source_root, episode_index, row_index = _sample_locator(
            self.dataset,
            int(index),
        )
        selected = self.binding.selection_for(source_root, episode_index)
        if selected is None or not selected.admits(
            row_index,
            self.binding.action_horizon,
        ):
            raise OpenPIPreparedSkillError(
                "OpenPI requested a sample outside the prepared binding."
            )
        identity = (int(selected.organized_episode_ordinal), int(row_index))
        sample = self.dataset[int(index)]
        prompt = None
        if isinstance(self.binding, OpenPIPreparedPromptBinding):
            prompt = self.binding.prompt_for(
                source_root,
                episode_index,
                row_index,
            )
        if isinstance(sample, Mapping) and (
            self.state_augmentation_key in sample
            or self.post_identity_transform is not None
            or prompt is not None
        ):
            output = dict(sample)
            output[self.state_augmentation_key] = np.asarray(identity, dtype=np.int64)
            if prompt is not None:
                # The immutable prepared binding is the authoritative prompt
                # owner. Override dataset defaults such as the legacy cube
                # prompt instead of allowing a silent fallback.
                output["prompt"] = prompt
            if self.post_identity_transform is not None:
                identity_transform = getattr(
                    self.post_identity_transform,
                    "apply_with_identity",
                    None,
                )
                if callable(identity_transform):
                    return identity_transform(output, identity=identity)
                return self.post_identity_transform(output)
            return output
        return sample

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))
        self._rebuild_balanced_index(self._epoch)
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(epoch)
        transform_setter = getattr(
            self.post_identity_transform,
            "set_epoch",
            None,
        )
        if callable(transform_setter):
            transform_setter(epoch)

    def close(self) -> None:
        closer = getattr(self.dataset, "close", None)
        if callable(closer):
            closer()

    def composition(self) -> Any:
        if self._balanced_composition is not None:
            return dict(self._balanced_composition)
        method = getattr(self.dataset, "composition", None)
        return None if not callable(method) else method()

    def mixing_metadata(self) -> Any:
        method = getattr(self.dataset, "mixing_metadata", None)
        base = None if not callable(method) else method()
        if self._raw_indices_by_stage is None:
            return base
        assert isinstance(self.binding, OpenPIPreparedPromptBinding)
        return {
            "source_mixing": base,
            "prompt_stage_sampling_policy": (
                self.binding.prompt_stage_sampling_policy
            ),
            "prompt_stage_sampling_seed": (
                self.binding.prompt_stage_sampling_seed
            ),
            "openpi_quality_valid_window_counts_by_stage": {
                key: len(value)
                for key, value in self._raw_indices_by_stage.items()
            },
            "effective_window_count_per_stage": min(
                len(value) for value in self._raw_indices_by_stage.values()
            ),
            "effective_epoch_window_count": len(self),
            "selection": "deterministic_epoch_rotated_without_replacement",
        }


class OpenPINaturalMergeSampler:
    """Deterministically shuffle every naturally merged sample once."""

    def __init__(self, dataset_size: int, group_size: int, seed: int) -> None:
        self.dataset_size = int(dataset_size)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.dataset_size <= 0 or self.group_size <= 0:
            raise ValueError(
                "OpenPI natural merge requires positive dataset/group sizes."
            )

    def __len__(self) -> int:
        return self.dataset_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.dataset_size, dtype=np.int64)
        random = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch])
        )
        random.shuffle(indices)
        yield from (int(index) for index in indices)


@contextmanager
def patch_openpi_prepared_skill_dataset(
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    *,
    post_identity_transform: Callable[[Mapping[str, Any]], Any] | None = None,
    load_images: bool | None = None,
) -> Iterator[None]:
    """Temporarily install the interval intersection at OpenPI's data seam."""

    try:
        from openpi.policies import dexteleop_v23_policy as policy
        from openpi.training import data_loader
    except ImportError as exc:
        raise OpenPIPreparedSkillError(
            "The remote OpenPI checkout is required for this adapter."
        ) from exc
    dataset_type = getattr(policy, "DexTeleopV23RelativeDataset", None)
    mixed_type = getattr(policy, "DexTeleopV23MixedRelativeDataset", None)
    if dataset_type is None or not callable(
        getattr(dataset_type, "_build_index", None)
    ):
        raise OpenPIPreparedSkillError(
            "OpenPI DexTeleop v2.3 dataset has no compatible _build_index seam."
        )
    if mixed_type is None or not callable(getattr(mixed_type, "__init__", None)):
        raise OpenPIPreparedSkillError(
            "OpenPI has no compatible DexTeleop v2.3 mixed-dataset seam."
        )
    sampler_name = "_DexTeleopV22ExactMixSampler"
    original_sampler = getattr(data_loader, sampler_name, None)
    if original_sampler is None:
        raise OpenPIPreparedSkillError(
            "OpenPI has no compatible mixed-dataset sampler seam."
        )
    if binding.source_mixing_policy != OPENPI_SOURCE_MIXING_NATURAL_V1:
        raise OpenPIPreparedSkillError(
            "Prepared binding does not request the supported natural merge."
        )
    original_build_index = dataset_type._build_index
    original_mixed_init = mixed_type.__init__
    original_mixing_metadata = getattr(mixed_type, "mixing_metadata", None)
    original_create = data_loader.create_torch_dataset

    def filtered_build_index(instance: Any) -> list[tuple[int, int]]:
        if int(instance.action_horizon) != binding.action_horizon:
            raise OpenPIPreparedSkillError(
                "OpenPI action horizon differs from the prepared binding."
            )
        base_index = original_build_index(instance)
        filtered: list[tuple[int, int]] = []
        for episode_position, row_index in base_index:
            episode = instance._episodes[int(episode_position)]
            selected = binding.selection_for(
                instance.dataset_root,
                int(episode.episode_index),
            )
            if selected is not None and selected.admits(
                int(row_index), binding.action_horizon
            ):
                filtered.append((int(episode_position), int(row_index)))
        if not filtered:
            raise OpenPIPreparedSkillError(
                "One OpenPI source produced no prepared Skill windows."
            )
        return filtered

    def natural_mixed_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        original_mixed_init(instance, *args, **kwargs)
        instance._mix_index = tuple(
            (source_position, sample_position)
            for source_position, child in enumerate(instance._datasets)
            for sample_position in range(len(child))
        )
        if not instance._mix_index:
            raise OpenPIPreparedSkillError(
                "OpenPI natural merge produced no prepared Skill windows."
            )

    def natural_mixing_metadata(instance: Any) -> dict[str, Any]:
        return {
            "schema": "dexteleop_v23_natural_merge_v1",
            "source_mixing_policy": OPENPI_SOURCE_MIXING_NATURAL_V1,
            "epoch_sample_counts": instance.composition(),
            "each_quality_valid_window_once_per_epoch": True,
            "shuffle": "deterministic_individual_sample_epoch_shuffle_v1",
            "sources": [source.to_dict() for source in instance.source_configs],
        }

    def create_prepared_dataset(*args: Any, **kwargs: Any) -> Any:
        if load_images is not None:
            kwargs["load_images"] = bool(load_images)
        dataset_type._build_index = filtered_build_index
        try:
            dataset = original_create(*args, **kwargs)
        finally:
            dataset_type._build_index = original_build_index
        return OpenPIPreparedSkillDataset(
            dataset,
            binding,
            getattr(policy, "STATE_AUGMENTATION_KEY"),
            post_identity_transform=post_identity_transform,
        )

    data_loader.create_torch_dataset = create_prepared_dataset
    mixed_type.__init__ = natural_mixed_init
    if callable(original_mixing_metadata):
        mixed_type.mixing_metadata = natural_mixing_metadata
    setattr(data_loader, sampler_name, OpenPINaturalMergeSampler)
    try:
        yield
    finally:
        data_loader.create_torch_dataset = original_create
        dataset_type._build_index = original_build_index
        mixed_type.__init__ = original_mixed_init
        if callable(original_mixing_metadata):
            mixed_type.mixing_metadata = original_mixing_metadata
        setattr(data_loader, sampler_name, original_sampler)


__all__ = [
    "OPENPI_PREPARED_PROMPT_BINDING_SCHEMA_V1",
    "OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V1",
    "OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V2",
    "OPENPI_PREPARED_SKILL_BINDING_SCHEMA_V1",
    "OPENPI_SOURCE_MIXING_NATURAL_V1",
    "OpenPINaturalMergeSampler",
    "OpenPIPreparedPromptBinding",
    "OpenPIPreparedPromptEpisodeSelection",
    "OpenPIPreparedPromptInterval",
    "OpenPIPreparedPromptSkill",
    "OpenPIPreparedEpisodeSelection",
    "OpenPIPreparedSkillBinding",
    "OpenPIPreparedSkillDataset",
    "OpenPIPreparedSkillError",
    "binding_source_manifests",
    "compose_openpi_prepared_prompt_binding",
    "load_openpi_prepared_prompt_binding",
    "load_openpi_prepared_skill_binding_manifest",
    "load_openpi_prepared_skill_binding",
    "materialize_openpi_prepared_skill_binding",
    "patch_openpi_prepared_skill_dataset",
]
