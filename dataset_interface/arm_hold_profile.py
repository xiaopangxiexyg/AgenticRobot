"""Strict, provisional dataset profiles for arm-hold experiments.

This module only declares model-neutral source and selection semantics.  It
does not scan datasets, select exact episodes, build windows, configure a
learner, or own deployment behavior.  The deliberately small public surface
avoids promoting the first experiment's JSON sections into stable contracts.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from dataset_contracts.arm_mode import (
    ARM_HOLD_TRAINING_PROJECTION_ID_V1,
    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


ARM_HOLD_DATASET_PROFILE_SCHEMA_V1 = (
    "dexteleop.arm_hold_dataset_profile.provisional.v1"
)
ARM_HOLD_DATASET_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_dataset_profile.semantic.provisional.v1"
)
ARM_HOLD_DATASET_PROFILE_PURPOSE_V1 = "lightweight_hold_label_prediction"

_PROFILE_KEYS = frozenset(
    {
        "schema",
        "profile_id",
        "purpose",
        "workspace_key",
        "sources",
        "required_contracts",
        "episode_admission",
        "sample_view",
        "split",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "source_dataset_id",
        "relative_dataset_root",
        "expected_dataset_name",
        "expected_episode_count",
        "expected_frame_count",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "dataset_schema",
        "layout_hash",
        "arm_hold_projection_id",
        "arm_hold_projection_hash",
        "progress_profile_hash",
        "stage_profile_hash",
    }
)
_ADMISSION_KEYS = frozenset(
    {
        "sealed_only",
        "success_policy",
        "abnormal_policy",
        "review_policy",
        "annotation_selection_policy",
        "annotation_selection",
        "required_modalities",
        "required_columns",
    }
)
_SAMPLE_VIEW_KEYS = frozenset(
    {
        "camera_key",
        "state_key",
        "horizon",
        "tail_policy",
        "running_session_state",
        "invalid_label_policy",
    }
)
_SPLIT_KEYS = frozenset(
    {
        "unit",
        "strategy",
        "train_fraction",
        "validation_fraction",
        "seed_domain",
        "stratification_keys",
        "coverage_constraints",
    }
)
_COVERAGE_KEYS = frozenset(
    {
        "event_kind",
        "event_key",
        "minimum_train_events",
        "minimum_validation_events",
    }
)
_STRATIFICATION_KEYS_V1 = frozenset(
    {"source_dataset_id", "observed_pair_states", "transition_directions"}
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ARM_SIDES = frozenset({"left", "right"})
_ARM_MODES = frozenset({"teleop", "fixed_hold"})


class ArmHoldDatasetProfileError(ValueError):
    """Raised when a profile is ambiguous or violates the provisional schema."""


def _object(
    value: object,
    *,
    name: str,
    keys: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ArmHoldDatasetProfileError(f"{name} must be an object.")
    if keys is not None:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        if missing:
            raise ArmHoldDatasetProfileError(
                f"{name} is missing required fields: {missing!r}."
            )
        if unknown:
            raise ArmHoldDatasetProfileError(
                f"{name} contains unknown fields: {unknown!r}."
            )
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArmHoldDatasetProfileError(f"{name} must be a non-empty string.")
    return value.strip()


def _identifier(value: object, *, name: str) -> str:
    normalized = _string(value, name=name)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ArmHoldDatasetProfileError(
            f"{name} must contain only letters, digits, '.', '_' or '-'."
        )
    return normalized


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArmHoldDatasetProfileError(f"{name} must be a boolean.")
    return value


def _count(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArmHoldDatasetProfileError(
            f"{name} must be a non-negative integer."
        )
    if positive and value == 0:
        raise ArmHoldDatasetProfileError(f"{name} must be positive.")
    return value


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArmHoldDatasetProfileError(f"{name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ArmHoldDatasetProfileError(
            f"{name} must be finite and strictly between zero and one."
        )
    return normalized


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArmHoldDatasetProfileError(f"{name} must be an array of strings.")
    normalized = tuple(_string(item, name=f"{name} item") for item in value)
    if not normalized:
        raise ArmHoldDatasetProfileError(f"{name} must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise ArmHoldDatasetProfileError(f"{name} must not contain duplicates.")
    return normalized


def _relative_locator(value: object) -> str:
    normalized = _string(value, name="relative_dataset_root")
    if "\\" in normalized:
        raise ArmHoldDatasetProfileError(
            "relative_dataset_root must use portable POSIX separators."
        )
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArmHoldDatasetProfileError(
            "relative_dataset_root must be clean and relative without traversal."
        )
    if PurePosixPath(normalized).is_absolute():
        raise ArmHoldDatasetProfileError(
            "relative_dataset_root must not be an absolute machine path."
        )
    return PurePosixPath(normalized).as_posix()


def _coverage_key(kind: str, value: object) -> str:
    key = _string(value, name="coverage event_key")
    if kind == "label":
        parts = key.split(":")
        valid = (
            len(parts) == 2
            and parts[0] in _ARM_SIDES
            and parts[1] in _ARM_MODES
        )
    elif kind == "pair_state":
        valid = len(key) == 2 and all(bit in "01" for bit in key)
    elif kind == "transition":
        side, separator, transition = key.partition(":")
        before, arrow, after = transition.partition("->")
        valid = (
            separator == ":"
            and arrow == "->"
            and side in _ARM_SIDES
            and before in _ARM_MODES
            and after in _ARM_MODES
            and before != after
        )
    else:
        raise ArmHoldDatasetProfileError(
            "coverage event_kind must be 'label', 'pair_state', or 'transition'."
        )
    if not valid:
        raise ArmHoldDatasetProfileError(
            f"Invalid {kind} coverage event_key {key!r}."
        )
    return key


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    return value


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ArmHoldDatasetProfileError(
                f"Profile JSON contains duplicate key {key!r}."
            )
        output[key] = value
    return output


@dataclass(frozen=True)
class ArmHoldDatasetSource:
    source_dataset_id: str
    relative_dataset_root: str
    expected_dataset_name: str
    expected_episode_count: int
    expected_frame_count: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArmHoldDatasetSource":
        value = _object(payload, name="source", keys=_SOURCE_KEYS)
        return cls(
            source_dataset_id=_identifier(
                value["source_dataset_id"], name="source_dataset_id"
            ),
            relative_dataset_root=_relative_locator(value["relative_dataset_root"]),
            expected_dataset_name=_identifier(
                value["expected_dataset_name"], name="expected_dataset_name"
            ),
            expected_episode_count=_count(
                value["expected_episode_count"], name="expected_episode_count"
            ),
            expected_frame_count=_count(
                value["expected_frame_count"], name="expected_frame_count"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dataset_id": self.source_dataset_id,
            "relative_dataset_root": self.relative_dataset_root,
            "expected_dataset_name": self.expected_dataset_name,
            "expected_episode_count": self.expected_episode_count,
            "expected_frame_count": self.expected_frame_count,
        }


@dataclass(frozen=True)
class ArmHoldDatasetProfile:
    profile_id: str
    workspace_key: str
    sources: tuple[ArmHoldDatasetSource, ...]
    required_contracts: Mapping[str, Any]
    episode_admission: Mapping[str, Any]
    sample_view: Mapping[str, Any]
    split: Mapping[str, Any]
    schema: str = ARM_HOLD_DATASET_PROFILE_SCHEMA_V1
    purpose: str = ARM_HOLD_DATASET_PROFILE_PURPOSE_V1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArmHoldDatasetProfile":
        value = _object(payload, name="profile", keys=_PROFILE_KEYS)
        schema = _string(value["schema"], name="schema")
        purpose = _string(value["purpose"], name="purpose")
        if schema != ARM_HOLD_DATASET_PROFILE_SCHEMA_V1:
            raise ArmHoldDatasetProfileError(f"Unsupported profile schema {schema!r}.")
        if purpose != ARM_HOLD_DATASET_PROFILE_PURPOSE_V1:
            raise ArmHoldDatasetProfileError(f"Unsupported profile purpose {purpose!r}.")

        raw_sources = value["sources"]
        if isinstance(raw_sources, (str, bytes)) or not isinstance(
            raw_sources, Sequence
        ):
            raise ArmHoldDatasetProfileError("sources must be a non-empty array.")
        sources = tuple(ArmHoldDatasetSource.from_dict(item) for item in raw_sources)
        if not sources:
            raise ArmHoldDatasetProfileError("sources must not be empty.")
        source_ids = tuple(source.source_dataset_id for source in sources)
        locators = tuple(source.relative_dataset_root for source in sources)
        if len(source_ids) != len(set(source_ids)):
            raise ArmHoldDatasetProfileError(
                "sources must not contain duplicate source_dataset_id values."
            )
        if len(locators) != len(set(locators)):
            raise ArmHoldDatasetProfileError(
                "sources must not contain duplicate relative_dataset_root values."
            )

        contracts = _object(
            value["required_contracts"],
            name="required_contracts",
            keys=_CONTRACT_KEYS,
        )
        normalized_contracts = {
            key: _string(contracts[key], name=key) for key in sorted(_CONTRACT_KEYS)
        }
        if normalized_contracts["arm_hold_projection_id"] != (
            ARM_HOLD_TRAINING_PROJECTION_ID_V1
        ):
            raise ArmHoldDatasetProfileError(
                "The required arm-hold projection ID is unsupported."
            )
        if normalized_contracts["arm_hold_projection_hash"] != (
            DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1.semantic_content_hash
        ):
            raise ArmHoldDatasetProfileError(
                "The required arm-hold projection hash does not match the "
                "canonical training contract."
            )

        admission = _object(
            value["episode_admission"],
            name="episode_admission",
            keys=_ADMISSION_KEYS,
        )
        selection = _object(
            admission["annotation_selection"], name="annotation_selection"
        )
        normalized_selection = {
            _string(key, name="annotation selection key"): _string(
                annotation_id, name="annotation selection value"
            )
            for key, annotation_id in selection.items()
        }
        modalities = tuple(
            sorted(_strings(admission["required_modalities"], name="required_modalities"))
        )
        unsupported_modalities = sorted(set(modalities) - {"numeric", "video"})
        if unsupported_modalities:
            raise ArmHoldDatasetProfileError(
                f"Unsupported required modalities: {unsupported_modalities!r}."
            )
        normalized_admission = {
            "sealed_only": _boolean(admission["sealed_only"], name="sealed_only"),
            "success_policy": _identifier(
                admission["success_policy"], name="success_policy"
            ),
            "abnormal_policy": _identifier(
                admission["abnormal_policy"], name="abnormal_policy"
            ),
            "review_policy": _identifier(
                admission["review_policy"], name="review_policy"
            ),
            "annotation_selection_policy": _identifier(
                admission["annotation_selection_policy"],
                name="annotation_selection_policy",
            ),
            "annotation_selection": normalized_selection,
            "required_modalities": modalities,
            "required_columns": tuple(
                sorted(_strings(admission["required_columns"], name="required_columns"))
            ),
        }
        expected_admission = {
            "sealed_only": True,
            "success_policy": "terminal_success_v1",
            "abnormal_policy": "exclude_episode_if_any_v1",
            "review_policy": "exclude_episode_review_pending_v1",
            "annotation_selection_policy": "explicit_or_unique_sealed_v1",
        }
        for key, expected in expected_admission.items():
            if normalized_admission[key] != expected:
                raise ArmHoldDatasetProfileError(
                    f"Unsupported episode admission value for {key}."
                )

        sample = _object(
            value["sample_view"], name="sample_view", keys=_SAMPLE_VIEW_KEYS
        )
        normalized_sample = {
            "camera_key": _identifier(sample["camera_key"], name="camera_key"),
            "state_key": _string(sample["state_key"], name="state_key"),
            "horizon": _count(sample["horizon"], name="horizon", positive=True),
            "tail_policy": _identifier(sample["tail_policy"], name="tail_policy"),
            "running_session_state": _identifier(
                sample["running_session_state"], name="running_session_state"
            ),
            "invalid_label_policy": _identifier(
                sample["invalid_label_policy"], name="invalid_label_policy"
            ),
        }
        if normalized_sample["tail_policy"] != "drop":
            raise ArmHoldDatasetProfileError(
                "The provisional profile requires tail_policy='drop'."
            )
        if (
            normalized_sample["invalid_label_policy"]
            != "drop_window_if_any_side_invalid_v1"
        ):
            raise ArmHoldDatasetProfileError(
                "The provisional profile requires "
                "invalid_label_policy='drop_window_if_any_side_invalid_v1'."
            )

        split = _object(value["split"], name="split", keys=_SPLIT_KEYS)
        train_fraction = _fraction(split["train_fraction"], name="train_fraction")
        validation_fraction = _fraction(
            split["validation_fraction"], name="validation_fraction"
        )
        if not math.isclose(
            train_fraction + validation_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArmHoldDatasetProfileError(
                "train_fraction and validation_fraction must sum to one."
            )
        unit = _identifier(split["unit"], name="split unit")
        strategy = _identifier(split["strategy"], name="split strategy")
        if unit != "episode_capture_identity":
            raise ArmHoldDatasetProfileError("Unsupported split unit.")
        if strategy != "source_and_transition_stratified_v1":
            raise ArmHoldDatasetProfileError("Unsupported split strategy.")
        stratification_keys = tuple(
            sorted(_strings(split["stratification_keys"], name="stratification_keys"))
        )
        if set(stratification_keys) != _STRATIFICATION_KEYS_V1:
            raise ArmHoldDatasetProfileError(
                "The v1 split requires source, pair-state, and transition strata."
            )
        raw_constraints = split["coverage_constraints"]
        if isinstance(raw_constraints, (str, bytes)) or not isinstance(
            raw_constraints, Sequence
        ):
            raise ArmHoldDatasetProfileError("coverage_constraints must be an array.")
        constraints: list[dict[str, Any]] = []
        for raw_constraint in raw_constraints:
            constraint = _object(
                raw_constraint,
                name="coverage constraint",
                keys=_COVERAGE_KEYS,
            )
            kind = _identifier(constraint["event_kind"], name="event_kind")
            constraints.append(
                {
                    "event_kind": kind,
                    "event_key": _coverage_key(kind, constraint["event_key"]),
                    "minimum_train_events": _count(
                        constraint["minimum_train_events"],
                        name="minimum_train_events",
                    ),
                    "minimum_validation_events": _count(
                        constraint["minimum_validation_events"],
                        name="minimum_validation_events",
                    ),
                }
            )
        constraint_ids = [
            (item["event_kind"], item["event_key"]) for item in constraints
        ]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ArmHoldDatasetProfileError(
                "coverage_constraints contain duplicate event identities."
            )
        normalized_split = {
            "unit": unit,
            "strategy": strategy,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "seed_domain": _identifier(split["seed_domain"], name="seed_domain"),
            "stratification_keys": stratification_keys,
            "coverage_constraints": tuple(
                sorted(
                    constraints,
                    key=lambda item: (item["event_kind"], item["event_key"]),
                )
            ),
        }

        return cls(
            schema=schema,
            profile_id=_identifier(value["profile_id"], name="profile_id"),
            purpose=purpose,
            workspace_key=_identifier(value["workspace_key"], name="workspace_key"),
            sources=tuple(sorted(sources, key=lambda source: source.source_dataset_id)),
            required_contracts=_freeze_mapping(normalized_contracts),
            episode_admission=_freeze_mapping(normalized_admission),
            sample_view=_freeze_mapping(normalized_sample),
            split=_freeze_mapping(normalized_split),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "purpose": self.purpose,
            "workspace_key": self.workspace_key,
            "sources": [source.to_dict() for source in self.sources],
            "required_contracts": _plain(self.required_contracts),
            "episode_admission": _plain(self.episode_admission),
            "sample_view": _plain(self.sample_view),
            "split": _plain(self.split),
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=ARM_HOLD_DATASET_PROFILE_HASH_DOMAIN_V1,
        )


def load_arm_hold_dataset_profile(path: str | Path) -> ArmHoldDatasetProfile:
    """Load one strict JSON profile without touching its declared datasets."""

    profile_path = Path(path).expanduser()
    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
    except ArmHoldDatasetProfileError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmHoldDatasetProfileError(
            f"Cannot load arm-hold dataset profile {profile_path}: {exc}"
        ) from exc
    return ArmHoldDatasetProfile.from_dict(payload)


def resolve_arm_hold_source_roots(
    profile: ArmHoldDatasetProfile,
    workspace_root: str | Path,
) -> Mapping[str, Path]:
    """Resolve host-local locators outside the profile semantic identity.

    Symlink resolution is intentional: the supported compatibility workspace
    uses per-dataset symlinks into the canonical data workspace.
    """

    if not isinstance(profile, ArmHoldDatasetProfile):
        raise TypeError("profile must be an ArmHoldDatasetProfile.")
    root = Path(workspace_root).expanduser().resolve()
    return MappingProxyType(
        {
            source.source_dataset_id: (
                root / source.relative_dataset_root
            ).resolve()
            for source in profile.sources
        }
    )


__all__ = [
    "ARM_HOLD_DATASET_PROFILE_HASH_DOMAIN_V1",
    "ARM_HOLD_DATASET_PROFILE_PURPOSE_V1",
    "ARM_HOLD_DATASET_PROFILE_SCHEMA_V1",
    "ArmHoldDatasetProfile",
    "ArmHoldDatasetProfileError",
    "ArmHoldDatasetSource",
    "load_arm_hold_dataset_profile",
    "resolve_arm_hold_source_roots",
]
