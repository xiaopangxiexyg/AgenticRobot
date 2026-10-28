"""Pure task-refinement profile contracts for post-capture annotation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial as _partial
from typing import Any, Mapping, Sequence

from . import _progress_validation as _validation
from .canonical_encoding import canonical_content_hash
from .progress_profile import TaskProgressProfileV1


REFINEMENT_BOUNDARY_DEFINITION_SCHEMA_V1 = (
    "dexteleop.refinement_boundary_definition.v1"
)
TASK_REFINEMENT_PROFILE_SCHEMA_V1 = "dexteleop.task_refinement_profile.v1"
REFINEMENT_BOUNDARY_SEMANTICS_V1 = frozenset(
    {"ENTER_STAGE", "COMPLETE_STAGE"}
)

_PROFILE_HASH_DOMAIN = "dexteleop.task_refinement_profile.semantic.v1"


class ProgressRefinementContractError(ValueError):
    """Raised when a refinement profile cannot be interpreted uniquely."""


_required = _partial(_validation.required_text, ProgressRefinementContractError)


def _optional_required(value: object | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


_sha256 = _partial(_validation.sha256_digest, ProgressRefinementContractError)
_positive_int = _partial(_validation.positive_int, ProgressRefinementContractError)


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgressRefinementContractError(f"{name} must be boolean.")
    return value


def _unique_strings(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in values)
    if len(result) != len(set(result)):
        raise ProgressRefinementContractError(f"{name} entries must be unique.")
    return result


@dataclass(frozen=True)
class RefinementBoundaryDefinitionV1:
    """One profile-defined boundary inside a coarse progress step."""

    progress_step_id: str
    refinement_boundary_id: str
    ordinal: int
    semantics: str
    parent_refinement_boundary_id: str | None = None
    eligible_as_exclusion_anchor: bool = False
    certifies_valid_prefix: bool = False
    required_predecessor_boundary_ids: tuple[str, ...] = ()
    schema: str = field(
        default=REFINEMENT_BOUNDARY_DEFINITION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "progress_step_id",
            _required(self.progress_step_id, name="progress_step_id"),
        )
        object.__setattr__(
            self,
            "refinement_boundary_id",
            _required(
                self.refinement_boundary_id,
                name="refinement_boundary_id",
            ),
        )
        object.__setattr__(
            self,
            "ordinal",
            _positive_int(self.ordinal, name="ordinal"),
        )
        semantics = _required(self.semantics, name="semantics").upper()
        if semantics not in REFINEMENT_BOUNDARY_SEMANTICS_V1:
            raise ProgressRefinementContractError(
                f"Unsupported refinement-boundary semantics {semantics!r}."
            )
        object.__setattr__(self, "semantics", semantics)
        object.__setattr__(
            self,
            "parent_refinement_boundary_id",
            _optional_required(
                self.parent_refinement_boundary_id,
                name="parent_refinement_boundary_id",
            ),
        )
        predecessors = _unique_strings(
            self.required_predecessor_boundary_ids,
            name="required_predecessor_boundary_ids",
        )
        if self.refinement_boundary_id in predecessors:
            raise ProgressRefinementContractError(
                "A refinement boundary cannot require itself."
            )
        eligible = _boolean(
            self.eligible_as_exclusion_anchor,
            name="eligible_as_exclusion_anchor",
        )
        certifies = _boolean(
            self.certifies_valid_prefix,
            name="certifies_valid_prefix",
        )
        if eligible and not certifies:
            raise ProgressRefinementContractError(
                "An exclusion anchor must certify its preceding prefix."
            )
        object.__setattr__(self, "eligible_as_exclusion_anchor", eligible)
        object.__setattr__(self, "certifies_valid_prefix", certifies)
        object.__setattr__(
            self,
            "required_predecessor_boundary_ids",
            predecessors,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "progress_step_id": self.progress_step_id,
            "refinement_boundary_id": self.refinement_boundary_id,
            "ordinal": self.ordinal,
            "semantics": self.semantics,
            "parent_refinement_boundary_id": (
                self.parent_refinement_boundary_id
            ),
            "eligible_as_exclusion_anchor": (
                self.eligible_as_exclusion_anchor
            ),
            "certifies_valid_prefix": self.certifies_valid_prefix,
            "required_predecessor_boundary_ids": (
                self.required_predecessor_boundary_ids
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RefinementBoundaryDefinitionV1":
        if payload.get("schema") != REFINEMENT_BOUNDARY_DEFINITION_SCHEMA_V1:
            raise ProgressRefinementContractError(
                "Unsupported refinement-boundary definition schema."
            )
        return cls(
            progress_step_id=payload.get("progress_step_id", ""),
            refinement_boundary_id=payload.get(
                "refinement_boundary_id",
                "",
            ),
            ordinal=payload.get("ordinal", 0),
            semantics=payload.get("semantics", ""),
            parent_refinement_boundary_id=payload.get(
                "parent_refinement_boundary_id"
            ),
            eligible_as_exclusion_anchor=payload.get(
                "eligible_as_exclusion_anchor",
                False,
            ),
            certifies_valid_prefix=payload.get(
                "certifies_valid_prefix",
                False,
            ),
            required_predecessor_boundary_ids=tuple(
                payload.get("required_predecessor_boundary_ids", ())
            ),
        )


@dataclass(frozen=True)
class TaskRefinementProfileV1:
    """Reusable fine-boundary definitions bound to one coarse profile."""

    profile_id: str
    profile_version: str
    task_progress_profile_hash: str
    boundaries: tuple[RefinementBoundaryDefinitionV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=TASK_REFINEMENT_PROFILE_SCHEMA_V1, init=False)

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
        boundaries = tuple(self.boundaries)
        if not boundaries or any(
            not isinstance(item, RefinementBoundaryDefinitionV1)
            for item in boundaries
        ):
            raise ProgressRefinementContractError(
                "boundaries must contain refinement-boundary definitions."
            )
        ids = tuple(item.refinement_boundary_id for item in boundaries)
        if len(ids) != len(set(ids)):
            raise ProgressRefinementContractError(
                "refinement_boundary_id values must be globally unique."
            )
        ordinal_keys = tuple(
            (item.progress_step_id, item.ordinal) for item in boundaries
        )
        if len(ordinal_keys) != len(set(ordinal_keys)):
            raise ProgressRefinementContractError(
                "Boundary ordinals must be unique within a progress step."
            )
        by_id = {item.refinement_boundary_id: item for item in boundaries}
        normalized: list[RefinementBoundaryDefinitionV1] = []
        for item in boundaries:
            referenced_ids = tuple(
                boundary_id
                for boundary_id in (
                    (item.parent_refinement_boundary_id,)
                    + item.required_predecessor_boundary_ids
                )
                if boundary_id is not None
            )
            unknown = set(referenced_ids) - set(by_id)
            if unknown:
                raise ProgressRefinementContractError(
                    f"Unknown refinement boundaries: {sorted(unknown)}."
                )
            for referenced_id in referenced_ids:
                referenced = by_id[referenced_id]
                if referenced.progress_step_id != item.progress_step_id:
                    raise ProgressRefinementContractError(
                        "Boundary relationships cannot cross progress steps."
                    )
                if referenced.ordinal >= item.ordinal:
                    raise ProgressRefinementContractError(
                        "Parent and predecessor boundaries must have lower ordinals."
                    )
            predecessors = tuple(
                sorted(
                    item.required_predecessor_boundary_ids,
                    key=lambda boundary_id: by_id[boundary_id].ordinal,
                )
            )
            normalized.append(
                replace(
                    item,
                    required_predecessor_boundary_ids=predecessors,
                )
            )
        boundaries = tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item.progress_step_id,
                    item.ordinal,
                    item.refinement_boundary_id,
                ),
            )
        )
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_PROFILE_HASH_DOMAIN,
            ),
        )

    @property
    def boundary_by_id(
        self,
    ) -> Mapping[str, RefinementBoundaryDefinitionV1]:
        return {
            item.refinement_boundary_id: item for item in self.boundaries
        }

    def boundaries_for_step(
        self,
        progress_step_id: str,
    ) -> tuple[RefinementBoundaryDefinitionV1, ...]:
        step_id = _required(progress_step_id, name="progress_step_id")
        return tuple(
            item
            for item in self.boundaries
            if item.progress_step_id == step_id
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "task_progress_profile_hash": self.task_progress_profile_hash,
            "boundaries": tuple(item.to_payload() for item in self.boundaries),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "TaskRefinementProfileV1":
        if payload.get("schema") != TASK_REFINEMENT_PROFILE_SCHEMA_V1:
            raise ProgressRefinementContractError(
                "Unsupported task-refinement profile schema."
            )
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            task_progress_profile_hash=payload.get(
                "task_progress_profile_hash",
                "",
            ),
            boundaries=tuple(
                RefinementBoundaryDefinitionV1.from_payload(item)
                for item in payload.get("boundaries", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressRefinementContractError(
                "Task-refinement profile hash mismatch."
            )
        return result


def validate_task_refinement_profile(
    *,
    progress_profile: TaskProgressProfileV1,
    refinement_profile: TaskRefinementProfileV1,
) -> None:
    if not isinstance(progress_profile, TaskProgressProfileV1):
        raise TypeError("progress_profile must be TaskProgressProfileV1.")
    if not isinstance(refinement_profile, TaskRefinementProfileV1):
        raise TypeError("refinement_profile must be TaskRefinementProfileV1.")
    if (
        refinement_profile.task_progress_profile_hash
        != progress_profile.semantic_content_hash
    ):
        raise ProgressRefinementContractError(
            "Refinement/progress profile hash mismatch."
        )
    unknown_steps = {
        item.progress_step_id for item in refinement_profile.boundaries
    } - set(progress_profile.step_by_id)
    if unknown_steps:
        raise ProgressRefinementContractError(
            f"Refinement boundaries reference unknown steps: {sorted(unknown_steps)}."
        )


__all__ = [
    "REFINEMENT_BOUNDARY_DEFINITION_SCHEMA_V1",
    "REFINEMENT_BOUNDARY_SEMANTICS_V1",
    "TASK_REFINEMENT_PROFILE_SCHEMA_V1",
    "ProgressRefinementContractError",
    "RefinementBoundaryDefinitionV1",
    "TaskRefinementProfileV1",
    "validate_task_refinement_profile",
]
