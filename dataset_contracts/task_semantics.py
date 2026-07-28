"""Append-only long-horizon task semantics contracts.

The event log is authoritative.  Per-frame stage columns, GUI state, and
training labels are rebuildable projections and therefore do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash


LONG_HORIZON_TASK_PROFILE_SCHEMA_V1 = "dexteleop.long_horizon_task_profile.v1"
SUBTASK_ATTEMPT_EVENT_SCHEMA_V1 = "dexteleop.subtask_attempt_event.v1"
TASK_SEMANTICS_EPISODE_MANIFEST_SCHEMA_V1 = (
    "dexteleop.task_semantics_episode_manifest.v1"
)
TASK_SEMANTICS_DUPLICATE_SUCCESS_POLICY_V1 = (
    "forbid_consecutive_same_successful_subtask_v1"
)
TASK_SEMANTICS_DEFAULT_TRACK_V1 = "primary_subtask"
SUBTASK_EVENT_KINDS_V1 = frozenset({"begin", "finish"})
SUBTASK_OUTCOMES_V1 = frozenset({"success", "failure", "aborted", "unknown"})
SUBTASK_EVENT_ORIGINS_V1 = frozenset({"online_operator", "post_annotation"})

_PROFILE_HASH_DOMAIN = "dexteleop.long_horizon_task_profile.semantic.v1"
_EVENT_HASH_DOMAIN = "dexteleop.subtask_attempt_event.semantic.v1"
_EVENT_BUNDLE_HASH_DOMAIN = "dexteleop.subtask_attempt_event_bundle.semantic.v1"
_MANIFEST_HASH_DOMAIN = "dexteleop.task_semantics_episode_manifest.semantic.v1"


class TaskSemanticsContractError(ValueError):
    """Raised when a task profile or attempt event stream is inconsistent."""


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise TaskSemanticsContractError(f"{name} must be non-empty.")
    return text


def _hash(value: object, *, name: str) -> str:
    text = _required(value, name=name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise TaskSemanticsContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TaskSemanticsContractError(f"{name} must be finite.")
    return result


def _unique_strings(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in values)
    if not result:
        raise TaskSemanticsContractError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise TaskSemanticsContractError(f"{name} entries must be unique.")
    return result


@dataclass(frozen=True)
class LongHorizonSubtask:
    subtask_id: str
    label: str
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtask_id", _required(self.subtask_id, name="subtask_id"))
        object.__setattr__(self, "label", _required(self.label, name="label"))
        object.__setattr__(self, "description", str(self.description).strip())

    def to_payload(self) -> dict[str, str]:
        return {
            "subtask_id": self.subtask_id,
            "label": self.label,
            "description": self.description,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LongHorizonSubtask":
        return cls(
            subtask_id=payload.get("subtask_id", ""),
            label=payload.get("label", ""),
            description=payload.get("description", ""),
        )


@dataclass(frozen=True)
class LongHorizonTaskProfile:
    profile_id: str
    profile_version: str
    task_id: str
    subtasks: tuple[LongHorizonSubtask, ...]
    track_ids: tuple[str, ...] = (TASK_SEMANTICS_DEFAULT_TRACK_V1,)
    allowed_transitions: tuple[tuple[str, str], ...] = ()
    duplicate_success_policy: str = TASK_SEMANTICS_DUPLICATE_SUCCESS_POLICY_V1
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=LONG_HORIZON_TASK_PROFILE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in ("profile_id", "profile_version", "task_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        subtasks = tuple(self.subtasks)
        if not subtasks:
            raise TaskSemanticsContractError("subtasks must not be empty.")
        subtask_ids = tuple(subtask.subtask_id for subtask in subtasks)
        if len(set(subtask_ids)) != len(subtask_ids):
            raise TaskSemanticsContractError("subtask_id values must be unique.")
        tracks = _unique_strings(self.track_ids, name="track_ids")
        transitions_list: list[tuple[str, str]] = []
        for pair in self.allowed_transitions:
            if len(pair) != 2:
                raise TaskSemanticsContractError(
                    "Each allowed transition must contain source and target."
                )
            transitions_list.append(
                (
                    _required(pair[0], name="transition source"),
                    _required(pair[1], name="transition target"),
                )
            )
        transitions = tuple(transitions_list)
        if len(set(transitions)) != len(transitions):
            raise TaskSemanticsContractError("allowed_transitions must be unique.")
        unknown = {
            item for pair in transitions for item in pair if item not in set(subtask_ids)
        }
        if unknown:
            raise TaskSemanticsContractError(
                f"allowed_transitions reference unknown subtasks: {sorted(unknown)}."
            )
        if self.duplicate_success_policy != TASK_SEMANTICS_DUPLICATE_SUCCESS_POLICY_V1:
            raise TaskSemanticsContractError("Unsupported duplicate success policy.")
        object.__setattr__(self, "subtasks", subtasks)
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "allowed_transitions", transitions)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_PROFILE_HASH_DOMAIN
            ),
        )

    @property
    def subtask_ids(self) -> frozenset[str]:
        return frozenset(subtask.subtask_id for subtask in self.subtasks)

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "task_id": self.task_id,
            "subtasks": [subtask.to_payload() for subtask in self.subtasks],
            "track_ids": self.track_ids,
            "allowed_transitions": [list(pair) for pair in self.allowed_transitions],
            "duplicate_success_policy": self.duplicate_success_policy,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LongHorizonTaskProfile":
        if payload.get("schema") != LONG_HORIZON_TASK_PROFILE_SCHEMA_V1:
            raise TaskSemanticsContractError("Unsupported long-horizon profile schema.")
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            task_id=payload.get("task_id", ""),
            subtasks=tuple(
                LongHorizonSubtask.from_payload(item)
                for item in payload.get("subtasks", ())
            ),
            track_ids=tuple(payload.get("track_ids", ())),
            allowed_transitions=tuple(
                tuple(pair) for pair in payload.get("allowed_transitions", ())
            ),
            duplicate_success_policy=payload.get("duplicate_success_policy", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise TaskSemanticsContractError("Long-horizon profile hash mismatch.")
        return result


@dataclass(frozen=True)
class SubtaskAttemptEvent:
    event_id: str
    episode_uuid: str
    legacy_episode_index: int
    event_sequence: int
    track_id: str
    attempt_id: str
    subtask_id: str
    event_kind: str
    effective_at: float
    available_at: float
    effective_clock_domain_id: str
    available_clock_domain_id: str
    actor_id: str
    outcome: str | None = None
    frame_index: int | None = None
    origin: str = "online_operator"
    source_event_ids: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=SUBTASK_ATTEMPT_EVENT_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "episode_uuid",
            "track_id",
            "attempt_id",
            "subtask_id",
            "event_kind",
            "effective_clock_domain_id",
            "available_clock_domain_id",
            "actor_id",
            "origin",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        index = int(self.legacy_episode_index)
        sequence = int(self.event_sequence)
        if isinstance(self.legacy_episode_index, bool) or index < 0:
            raise TaskSemanticsContractError(
                "legacy_episode_index must be a non-negative integer."
            )
        if isinstance(self.event_sequence, bool) or sequence < 1:
            raise TaskSemanticsContractError("event_sequence must be a positive integer.")
        object.__setattr__(self, "legacy_episode_index", index)
        object.__setattr__(self, "event_sequence", sequence)
        if self.event_kind not in SUBTASK_EVENT_KINDS_V1:
            raise TaskSemanticsContractError(f"Unsupported event_kind {self.event_kind!r}.")
        outcome = None if self.outcome is None else _required(self.outcome, name="outcome")
        if self.event_kind == "begin" and outcome is not None:
            raise TaskSemanticsContractError("A begin event cannot declare an outcome.")
        if self.event_kind == "finish" and outcome not in SUBTASK_OUTCOMES_V1:
            raise TaskSemanticsContractError(
                "A finish event must declare a supported outcome."
            )
        object.__setattr__(self, "outcome", outcome)
        effective = _finite(self.effective_at, name="effective_at")
        available = _finite(self.available_at, name="available_at")
        if (
            self.available_clock_domain_id == self.effective_clock_domain_id
            and available < effective
        ):
            raise TaskSemanticsContractError("available_at cannot precede effective_at.")
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "available_at", available)
        if self.frame_index is not None:
            frame = int(self.frame_index)
            if isinstance(self.frame_index, bool) or frame < 0:
                raise TaskSemanticsContractError(
                    "frame_index must be non-negative when provided."
                )
            object.__setattr__(self, "frame_index", frame)
        if self.origin not in SUBTASK_EVENT_ORIGINS_V1:
            raise TaskSemanticsContractError(f"Unsupported origin {self.origin!r}.")
        sources = tuple(_required(item, name="source_event_id") for item in self.source_event_ids)
        if len(set(sources)) != len(sources):
            raise TaskSemanticsContractError("source_event_ids must be unique.")
        if self.event_id in sources:
            raise TaskSemanticsContractError("An event cannot cite itself as a source.")
        object.__setattr__(self, "source_event_ids", sources)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_EVENT_HASH_DOMAIN
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "event_id": self.event_id,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "event_sequence": self.event_sequence,
            "track_id": self.track_id,
            "attempt_id": self.attempt_id,
            "subtask_id": self.subtask_id,
            "event_kind": self.event_kind,
            "outcome": self.outcome,
            "effective_at": self.effective_at,
            "available_at": self.available_at,
            "effective_clock_domain_id": self.effective_clock_domain_id,
            "available_clock_domain_id": self.available_clock_domain_id,
            "frame_index": self.frame_index,
            "origin": self.origin,
            "actor_id": self.actor_id,
            "source_event_ids": self.source_event_ids,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SubtaskAttemptEvent":
        if payload.get("schema") != SUBTASK_ATTEMPT_EVENT_SCHEMA_V1:
            raise TaskSemanticsContractError("Unsupported subtask event schema.")
        result = cls(
            event_id=payload.get("event_id", ""),
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            event_sequence=payload.get("event_sequence", 0),
            track_id=payload.get("track_id", ""),
            attempt_id=payload.get("attempt_id", ""),
            subtask_id=payload.get("subtask_id", ""),
            event_kind=payload.get("event_kind", ""),
            outcome=payload.get("outcome"),
            effective_at=payload.get("effective_at", float("nan")),
            available_at=payload.get("available_at", float("nan")),
            effective_clock_domain_id=payload.get("effective_clock_domain_id", ""),
            available_clock_domain_id=payload.get("available_clock_domain_id", ""),
            frame_index=payload.get("frame_index"),
            origin=payload.get("origin", ""),
            actor_id=payload.get("actor_id", ""),
            source_event_ids=tuple(payload.get("source_event_ids", ())),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise TaskSemanticsContractError("Subtask attempt event hash mismatch.")
        return result


@dataclass(frozen=True)
class TaskSemanticsValidation:
    attempt_count: int
    completed_attempt_count: int
    open_attempt_ids: tuple[str, ...]


def validate_subtask_attempt_events(
    profile: LongHorizonTaskProfile,
    events: Sequence[SubtaskAttemptEvent],
    *,
    require_closed: bool,
) -> TaskSemanticsValidation:
    ordered = tuple(events)
    if not ordered:
        return TaskSemanticsValidation(0, 0, ())
    identities = {(event.episode_uuid, event.legacy_episode_index) for event in ordered}
    if len(identities) != 1:
        raise TaskSemanticsContractError("All task events must belong to one episode.")
    sequences = tuple(event.event_sequence for event in ordered)
    if sequences != tuple(range(1, len(ordered) + 1)):
        raise TaskSemanticsContractError(
            "Task event_sequence values must be contiguous and ordered from one."
        )
    if len({event.event_id for event in ordered}) != len(ordered):
        raise TaskSemanticsContractError("Task event_id values must be unique.")
    effective_clocks = {event.effective_clock_domain_id for event in ordered}
    if len(effective_clocks) != 1:
        raise TaskSemanticsContractError(
            "One task event stream must use one effective clock domain."
        )
    active: dict[str, SubtaskAttemptEvent] = {}
    previous_completed: dict[str, tuple[str, str]] = {}
    attempts: set[str] = set()
    completed = 0
    transition_allowlist = set(profile.allowed_transitions)
    timeline = tuple(sorted(ordered, key=lambda event: (event.effective_at, event.event_sequence)))
    for event in timeline:
        if event.track_id not in profile.track_ids:
            raise TaskSemanticsContractError(f"Unknown task track {event.track_id!r}.")
        if event.subtask_id not in profile.subtask_ids:
            raise TaskSemanticsContractError(f"Unknown subtask {event.subtask_id!r}.")
        if event.event_kind == "begin":
            if event.track_id in active:
                raise TaskSemanticsContractError(
                    f"Track {event.track_id!r} already has an active attempt."
                )
            if event.attempt_id in attempts:
                raise TaskSemanticsContractError(
                    f"attempt_id {event.attempt_id!r} was already used."
                )
            previous = previous_completed.get(event.track_id)
            if (
                previous is not None
                and previous[1] == "success"
                and transition_allowlist
            ):
                if (previous[0], event.subtask_id) not in transition_allowlist:
                    raise TaskSemanticsContractError(
                        f"Transition {previous[0]!r}->{event.subtask_id!r} is not allowed."
                    )
            active[event.track_id] = event
            attempts.add(event.attempt_id)
            continue
        begin = active.get(event.track_id)
        if begin is None:
            raise TaskSemanticsContractError(
                f"Finish event has no active attempt on track {event.track_id!r}."
            )
        if event.attempt_id != begin.attempt_id or event.subtask_id != begin.subtask_id:
            raise TaskSemanticsContractError(
                "Finish event must match the active attempt and subtask."
            )
        previous = previous_completed.get(event.track_id)
        if (
            event.outcome == "success"
            and previous == (event.subtask_id, "success")
        ):
            raise TaskSemanticsContractError(
                "Two consecutive successful attempts of the same subtask are forbidden."
            )
        previous_completed[event.track_id] = (event.subtask_id, str(event.outcome))
        del active[event.track_id]
        completed += 1

    open_ids = tuple(active[track].attempt_id for track in sorted(active))
    if require_closed and open_ids:
        raise TaskSemanticsContractError(
            "Sealed task semantics require every attempt to have a finish event."
        )
    return TaskSemanticsValidation(len(attempts), completed, open_ids)


@dataclass(frozen=True)
class TaskSemanticsEpisodeManifest:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    task_profile_hash: str
    event_count: int
    attempt_count: int
    event_bundle_hash: str
    sealed_at: float
    sealed_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=TASK_SEMANTICS_EPISODE_MANIFEST_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in ("episode_uuid", "sealed_clock_domain_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        index = int(self.legacy_episode_index)
        event_count = int(self.event_count)
        attempt_count = int(self.attempt_count)
        if (
            isinstance(self.legacy_episode_index, bool)
            or isinstance(self.event_count, bool)
            or isinstance(self.attempt_count, bool)
            or index < 0
            or event_count < 0
            or attempt_count < 0
        ):
            raise TaskSemanticsContractError("Manifest counts and index must be non-negative.")
        if attempt_count > event_count:
            raise TaskSemanticsContractError("attempt_count cannot exceed event_count.")
        object.__setattr__(self, "legacy_episode_index", index)
        object.__setattr__(self, "event_count", event_count)
        object.__setattr__(self, "attempt_count", attempt_count)
        for name in ("episode_identity_hash", "task_profile_hash", "event_bundle_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(self, "sealed_at", _finite(self.sealed_at, name="sealed_at"))
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_MANIFEST_HASH_DOMAIN
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        profile: LongHorizonTaskProfile,
        episode_uuid: str,
        legacy_episode_index: int,
        episode_identity_hash: str,
        events: Sequence[SubtaskAttemptEvent],
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> "TaskSemanticsEpisodeManifest":
        validation = validate_subtask_attempt_events(profile, events, require_closed=True)
        event_tuple = tuple(events)
        if event_tuple and (
            event_tuple[0].episode_uuid != episode_uuid
            or event_tuple[0].legacy_episode_index != legacy_episode_index
        ):
            raise TaskSemanticsContractError(
                "Manifest identity does not match the task event stream."
            )
        bundle_hash = canonical_content_hash(
            tuple(event.semantic_content_hash for event in event_tuple),
            domain=_EVENT_BUNDLE_HASH_DOMAIN,
        )
        return cls(
            episode_uuid=episode_uuid,
            legacy_episode_index=legacy_episode_index,
            episode_identity_hash=episode_identity_hash,
            task_profile_hash=profile.semantic_content_hash,
            event_count=len(event_tuple),
            attempt_count=validation.attempt_count,
            event_bundle_hash=bundle_hash,
            sealed_at=sealed_at,
            sealed_clock_domain_id=sealed_clock_domain_id,
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "task_profile_hash": self.task_profile_hash,
            "event_count": self.event_count,
            "attempt_count": self.attempt_count,
            "event_bundle_hash": self.event_bundle_hash,
            "sealed_at": self.sealed_at,
            "sealed_clock_domain_id": self.sealed_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TaskSemanticsEpisodeManifest":
        if payload.get("schema") != TASK_SEMANTICS_EPISODE_MANIFEST_SCHEMA_V1:
            raise TaskSemanticsContractError("Unsupported task semantics manifest schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            task_profile_hash=payload.get("task_profile_hash", ""),
            event_count=payload.get("event_count", -1),
            attempt_count=payload.get("attempt_count", -1),
            event_bundle_hash=payload.get("event_bundle_hash", ""),
            sealed_at=payload.get("sealed_at", float("nan")),
            sealed_clock_domain_id=payload.get("sealed_clock_domain_id", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise TaskSemanticsContractError("Task semantics manifest hash mismatch.")
        return result


__all__ = [
    "LONG_HORIZON_TASK_PROFILE_SCHEMA_V1",
    "SUBTASK_ATTEMPT_EVENT_SCHEMA_V1",
    "SUBTASK_EVENT_KINDS_V1",
    "SUBTASK_EVENT_ORIGINS_V1",
    "SUBTASK_OUTCOMES_V1",
    "TASK_SEMANTICS_DEFAULT_TRACK_V1",
    "TASK_SEMANTICS_DUPLICATE_SUCCESS_POLICY_V1",
    "TASK_SEMANTICS_EPISODE_MANIFEST_SCHEMA_V1",
    "LongHorizonSubtask",
    "LongHorizonTaskProfile",
    "SubtaskAttemptEvent",
    "TaskSemanticsContractError",
    "TaskSemanticsEpisodeManifest",
    "TaskSemanticsValidation",
    "validate_subtask_attempt_events",
]
