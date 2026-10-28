"""Pure ordered execution-stage profiles for collection progress.

The coarse :mod:`dataset_contracts.progress_profile` graph is the VLM-visible
task decomposition.  This module defines the strictly ordered execution stages
inside each coarse progress step.  Stage definitions are available at capture
time so a recovery Episode can declare an exact stage entry, while ordinary
stage boundaries remain post-capture annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial as _partial
from typing import Any, Mapping

from . import _progress_validation as _validation
from .canonical_encoding import canonical_content_hash
from .progress_profile import TaskProgressProfileV1


PROGRESS_STAGE_DEFINITION_SCHEMA_V1 = (
    "dexteleop.progress_stage_definition.v1"
)
TASK_STAGE_PROFILE_SCHEMA_V1 = "dexteleop.task_stage_profile.v1"
PROGRESS_STAGE_ENTRY_BINDING_SCHEMA_V1 = (
    "dexteleop.progress_stage_entry_binding.v1"
)
TASK_STAGE_PROFILE_FILE_DOMAIN_V1 = (
    "dexteleop.task_stage_profile.file.v1"
)
PROGRESS_STAGE_ENTRY_BINDING_FILE_DOMAIN_V1 = (
    "dexteleop.progress_stage_entry_binding.file.v1"
)
PROGRESS_STAGE_ENTRY_CONDITIONS_V1 = frozenset(
    {"NOMINAL", "RECOVERY_FROM_FAILURE"}
)

_PROFILE_HASH_DOMAIN = "dexteleop.task_stage_profile.semantic.v1"
_ENTRY_BINDING_HASH_DOMAIN = (
    "dexteleop.progress_stage_entry_binding.semantic.v1"
)


class ProgressStageContractError(ValueError):
    """Raised when an ordered stage profile is ambiguous or incomplete."""


_required = _partial(_validation.required_text, ProgressStageContractError)
_positive_int = _partial(_validation.positive_int, ProgressStageContractError)
_sha256 = _partial(_validation.sha256_digest, ProgressStageContractError)


@dataclass(frozen=True)
class ProgressStageDefinitionV1:
    """One execution/failure semantic inside a coarse progress step."""

    progress_step_id: str
    stage_id: str
    display_name: str
    description: str
    ordinal: int
    schema: str = field(
        default=PROGRESS_STAGE_DEFINITION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "progress_step_id",
            "stage_id",
            "display_name",
            "description",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "ordinal",
            _positive_int(self.ordinal, name="ordinal"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "progress_step_id": self.progress_step_id,
            "stage_id": self.stage_id,
            "display_name": self.display_name,
            "description": self.description,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressStageDefinitionV1":
        if payload.get("schema") != PROGRESS_STAGE_DEFINITION_SCHEMA_V1:
            raise ProgressStageContractError(
                "Unsupported progress-stage definition schema."
            )
        return cls(
            progress_step_id=payload.get("progress_step_id", ""),
            stage_id=payload.get("stage_id", ""),
            display_name=payload.get("display_name", ""),
            description=payload.get("description", ""),
            ordinal=payload.get("ordinal", 0),
        )


@dataclass(frozen=True)
class TaskStageProfileV1:
    """Strict ordered stage sequences bound to one coarse task profile."""

    profile_id: str
    profile_version: str
    task_progress_profile_hash: str
    stages: tuple[ProgressStageDefinitionV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=TASK_STAGE_PROFILE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            _required(self.profile_id, name="profile_id"),
        )
        object.__setattr__(
            self,
            "profile_version",
            _required(self.profile_version, name="profile_version"),
        )
        object.__setattr__(
            self,
            "task_progress_profile_hash",
            _sha256(
                self.task_progress_profile_hash,
                name="task_progress_profile_hash",
            ),
        )
        stages = tuple(self.stages)
        if not stages or any(
            not isinstance(item, ProgressStageDefinitionV1)
            for item in stages
        ):
            raise ProgressStageContractError(
                "stages must contain progress-stage definitions."
            )
        ids = tuple(item.stage_id for item in stages)
        if len(ids) != len(set(ids)):
            raise ProgressStageContractError(
                "stage_id values must be globally unique."
            )
        by_step: dict[str, list[ProgressStageDefinitionV1]] = {}
        for stage in stages:
            by_step.setdefault(stage.progress_step_id, []).append(stage)
        normalized: list[ProgressStageDefinitionV1] = []
        for step_id, step_stages in by_step.items():
            ordered = sorted(
                step_stages,
                key=lambda item: (item.ordinal, item.stage_id),
            )
            ordinals = tuple(item.ordinal for item in ordered)
            expected = tuple(range(1, len(ordered) + 1))
            if ordinals != expected:
                raise ProgressStageContractError(
                    "Stage ordinals must be contiguous from one within "
                    f"progress step {step_id!r}; got {ordinals}."
                )
            normalized.extend(ordered)
        stages = tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item.progress_step_id,
                    item.ordinal,
                    item.stage_id,
                ),
            )
        )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_PROFILE_HASH_DOMAIN,
            ),
        )

    @property
    def stage_by_id(self) -> Mapping[str, ProgressStageDefinitionV1]:
        return {item.stage_id: item for item in self.stages}

    def stages_for_step(
        self,
        progress_step_id: str,
    ) -> tuple[ProgressStageDefinitionV1, ...]:
        step_id = _required(progress_step_id, name="progress_step_id")
        return tuple(
            item
            for item in self.stages
            if item.progress_step_id == step_id
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "task_progress_profile_hash": self.task_progress_profile_hash,
            "stages": tuple(item.to_payload() for item in self.stages),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "TaskStageProfileV1":
        if payload.get("schema") != TASK_STAGE_PROFILE_SCHEMA_V1:
            raise ProgressStageContractError(
                "Unsupported task-stage profile schema."
            )
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            task_progress_profile_hash=payload.get(
                "task_progress_profile_hash",
                "",
            ),
            stages=tuple(
                ProgressStageDefinitionV1.from_payload(item)
                for item in payload.get("stages", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressStageContractError(
                "Task-stage profile hash mismatch."
            )
        return result


@dataclass(frozen=True)
class ProgressStageEntryBindingV1:
    """Bind one capture Episode to its exact initial execution stage.

    The coarse entry step is intentionally not repeated here: it is derived
    from ``entry_stage_id`` and must equal the existing
    :class:`ProgressEpisodeBindingV1` entry step.
    """

    episode_capture_identity_hash: str
    task_stage_profile_hash: str
    entry_stage_id: str
    entry_condition: str = "NOMINAL"
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=PROGRESS_STAGE_ENTRY_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "task_stage_profile_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "entry_stage_id",
            _required(self.entry_stage_id, name="entry_stage_id"),
        )
        condition = _required(
            self.entry_condition,
            name="entry_condition",
        ).upper()
        if condition not in PROGRESS_STAGE_ENTRY_CONDITIONS_V1:
            raise ProgressStageContractError(
                f"Unsupported entry_condition {condition!r}."
            )
        object.__setattr__(self, "entry_condition", condition)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_ENTRY_BINDING_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "task_stage_profile_hash": self.task_stage_profile_hash,
            "entry_stage_id": self.entry_stage_id,
            "entry_condition": self.entry_condition,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressStageEntryBindingV1":
        if payload.get("schema") != PROGRESS_STAGE_ENTRY_BINDING_SCHEMA_V1:
            raise ProgressStageContractError(
                "Unsupported progress-stage entry binding schema."
            )
        result = cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash",
                "",
            ),
            task_stage_profile_hash=payload.get(
                "task_stage_profile_hash",
                "",
            ),
            entry_stage_id=payload.get("entry_stage_id", ""),
            entry_condition=payload.get("entry_condition", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressStageContractError(
                "Progress-stage entry binding hash mismatch."
            )
        return result


def validate_task_stage_profile(
    *,
    progress_profile: TaskProgressProfileV1,
    stage_profile: TaskStageProfileV1,
) -> None:
    """Require one non-empty ordered stage sequence for every coarse step."""

    if not isinstance(progress_profile, TaskProgressProfileV1):
        raise TypeError("progress_profile must be TaskProgressProfileV1.")
    if not isinstance(stage_profile, TaskStageProfileV1):
        raise TypeError("stage_profile must be TaskStageProfileV1.")
    if (
        stage_profile.task_progress_profile_hash
        != progress_profile.semantic_content_hash
    ):
        raise ProgressStageContractError(
            "Stage/progress profile hash mismatch."
        )
    progress_steps = set(progress_profile.step_by_id)
    stage_steps = {item.progress_step_id for item in stage_profile.stages}
    unknown = stage_steps - progress_steps
    if unknown:
        raise ProgressStageContractError(
            f"Stage profile references unknown progress steps: {sorted(unknown)}."
        )
    missing = progress_steps - stage_steps
    if missing:
        raise ProgressStageContractError(
            "Every progress step must own at least one ordered stage; "
            f"missing={sorted(missing)}."
        )


def validate_progress_stage_entry_binding(
    *,
    progress_profile: TaskProgressProfileV1,
    progress_entry_step_id: str,
    stage_profile: TaskStageProfileV1,
    stage_binding: ProgressStageEntryBindingV1,
) -> None:
    """Validate one exact fine-stage entry against the coarse entry."""

    validate_task_stage_profile(
        progress_profile=progress_profile,
        stage_profile=stage_profile,
    )
    if not isinstance(stage_binding, ProgressStageEntryBindingV1):
        raise TypeError(
            "stage_binding must be ProgressStageEntryBindingV1."
        )
    if (
        stage_binding.task_stage_profile_hash
        != stage_profile.semantic_content_hash
    ):
        raise ProgressStageContractError(
            "Stage entry binding/profile hash mismatch."
        )
    stage = stage_profile.stage_by_id.get(stage_binding.entry_stage_id)
    if stage is None:
        raise ProgressStageContractError(
            "Stage entry binding references an unknown stage."
        )
    entry_step = _required(
        progress_entry_step_id,
        name="progress_entry_step_id",
    )
    if stage.progress_step_id != entry_step:
        raise ProgressStageContractError(
            "Fine-stage entry does not belong to the coarse progress entry "
            f"step: stage={stage.progress_step_id!r} "
            f"coarse={entry_step!r}."
        )
    if (
        stage_binding.entry_condition == "NOMINAL"
        and stage.ordinal != 1
    ):
        raise ProgressStageContractError(
            "A non-initial fine-stage entry must explicitly use "
            "RECOVERY_FROM_FAILURE."
        )


__all__ = [
    "PROGRESS_STAGE_ENTRY_BINDING_FILE_DOMAIN_V1",
    "PROGRESS_STAGE_ENTRY_BINDING_SCHEMA_V1",
    "PROGRESS_STAGE_DEFINITION_SCHEMA_V1",
    "PROGRESS_STAGE_ENTRY_CONDITIONS_V1",
    "TASK_STAGE_PROFILE_FILE_DOMAIN_V1",
    "TASK_STAGE_PROFILE_SCHEMA_V1",
    "ProgressStageContractError",
    "ProgressStageDefinitionV1",
    "ProgressStageEntryBindingV1",
    "TaskStageProfileV1",
    "validate_progress_stage_entry_binding",
    "validate_task_stage_profile",
]
