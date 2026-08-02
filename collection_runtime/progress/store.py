"""Append-only durable progress facts subordinate to an Episode owner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading

from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_facts import (
    PROGRESS_BINDING_FILE_DOMAIN_V1,
    PROGRESS_FACT_FILE_DOMAIN_V1,
    PROGRESS_MANIFEST_FILE_DOMAIN_V1,
    PROGRESS_PROFILE_FILE_DOMAIN_V1,
    EpisodeProgressTerminationV1,
    ProgressEpisodeManifestV1,
    ProgressFactEnvelopeV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
    validate_progress_episode_binding,
)
from dataset_contracts.progress_stages import (
    PROGRESS_STAGE_ENTRY_BINDING_FILE_DOMAIN_V1,
    TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_progress_stage_entry_binding,
)

from ..immutable_canonical_store import fsync_directory, publish_immutable_canonical
from .reducer import ProgressState, replay_progress_facts


class ProgressStoreError(RuntimeError):
    """Base error for progress persistence."""


class ProgressStoreIncompleteError(ProgressStoreError):
    """A torn or unparseable append log cannot be silently repaired."""


@dataclass(frozen=True)
class ProgressAppendResult:
    fact: ProgressFactEnvelopeV1
    accepted: bool
    durable: bool
    idempotent: bool


@dataclass(frozen=True)
class ProgressSealReport:
    manifest: ProgressEpisodeManifestV1
    manifest_relative_path: str
    idempotent: bool


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        count = os.write(fd, payload[offset:])
        if count <= 0:
            raise OSError("Progress fact append made no progress.")
        offset += count


class ProgressFactStore:
    """One append owner for one externally-owned Episode.

    The Episode owner supplies the directory. This prototype intentionally
    does not freeze a repository-global dataset topology.
    """

    def __init__(
        self,
        *,
        episode_dir: str | Path,
        identity: EpisodeCaptureIdentity,
        profile: TaskProgressProfileV1,
        binding: ProgressEpisodeBindingV1,
        stage_profile: TaskStageProfileV1 | None = None,
        stage_entry_binding: ProgressStageEntryBindingV1 | None = None,
        fsync_on_append: bool = True,
    ) -> None:
        if not isinstance(identity, EpisodeCaptureIdentity):
            raise TypeError("identity must be EpisodeCaptureIdentity.")
        validate_progress_episode_binding(profile=profile, binding=binding)
        if (
            binding.episode_capture_identity_hash
            != identity.semantic_content_hash
        ):
            raise ValueError("Progress binding and Episode identity differ.")
        if (stage_profile is None) != (stage_entry_binding is None):
            raise ValueError(
                "stage_profile and stage_entry_binding must be provided "
                "together."
            )
        if stage_profile is not None and stage_entry_binding is not None:
            validate_progress_stage_entry_binding(
                progress_profile=profile,
                progress_entry_step_id=binding.entry_step_id,
                stage_profile=stage_profile,
                stage_binding=stage_entry_binding,
            )
            if (
                stage_entry_binding.episode_capture_identity_hash
                != identity.semantic_content_hash
            ):
                raise ValueError(
                    "Stage entry binding and Episode identity differ."
                )
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.identity = identity
        self.profile = profile
        self.binding = binding
        self.stage_profile = stage_profile
        self.stage_entry_binding = stage_entry_binding
        if not isinstance(fsync_on_append, bool):
            raise TypeError("fsync_on_append must be boolean.")
        self.fsync_on_append = fsync_on_append
        self.fact_path = self.episode_dir / "facts.cjsonl"
        self.manifest_path = self.episode_dir / "manifest.cjson"
        self._lock = threading.RLock()
        self._fd: int | None = None
        self._closed = False
        self._sealed = False
        self._poisoned = False
        self._poison_reason: str | None = None
        self._manifest: ProgressEpisodeManifestV1 | None = None
        self._facts: list[ProgressFactEnvelopeV1] = []
        self._fact_by_id: dict[str, ProgressFactEnvelopeV1] = {}

        self.episode_dir.mkdir(parents=True, exist_ok=True)
        publish_immutable_canonical(
            self.episode_dir / "profile.cjson",
            profile.to_payload(),
            domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
        )
        publish_immutable_canonical(
            self.episode_dir / "binding.cjson",
            binding.to_payload(),
            domain=PROGRESS_BINDING_FILE_DOMAIN_V1,
        )
        if stage_profile is not None and stage_entry_binding is not None:
            publish_immutable_canonical(
                self.episode_dir / "stages.cjson",
                stage_profile.to_payload(),
                domain=TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
            )
            publish_immutable_canonical(
                self.episode_dir / "stage_entry.cjson",
                stage_entry_binding.to_payload(),
                domain=PROGRESS_STAGE_ENTRY_BINDING_FILE_DOMAIN_V1,
            )
        self._load_facts()
        self._load_manifest_if_present()
        if not self._sealed:
            fact_path_existed = self.fact_path.exists()
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            self._fd = os.open(self.fact_path, flags, 0o600)
            if not fact_path_existed:
                fsync_directory(self.fact_path.parent)

    @property
    def facts(self) -> tuple[ProgressFactEnvelopeV1, ...]:
        with self._lock:
            return tuple(self._facts)

    @property
    def next_sequence_index(self) -> int:
        with self._lock:
            return len(self._facts) + 1

    @property
    def sealed(self) -> bool:
        with self._lock:
            return self._sealed

    @property
    def manifest(self) -> ProgressEpisodeManifestV1 | None:
        with self._lock:
            return self._manifest

    @property
    def poisoned(self) -> bool:
        with self._lock:
            return self._poisoned

    @property
    def poison_reason(self) -> str | None:
        with self._lock:
            return self._poison_reason

    def fact_by_id(self, fact_id: str) -> ProgressFactEnvelopeV1 | None:
        with self._lock:
            return self._fact_by_id.get(str(fact_id))

    def replay_state(self) -> ProgressState:
        with self._lock:
            return replay_progress_facts(
                profile=self.profile,
                binding=self.binding,
                facts=tuple(self._facts),
            )

    def append_fact(self, fact: ProgressFactEnvelopeV1) -> ProgressAppendResult:
        if not isinstance(fact, ProgressFactEnvelopeV1):
            raise TypeError("fact must be ProgressFactEnvelopeV1.")
        with self._lock:
            self._require_open()
            existing = self._fact_by_id.get(fact.fact_id)
            if existing is not None:
                if existing.semantic_content_hash != fact.semantic_content_hash:
                    raise ProgressStoreError(
                        f"Progress fact ID conflict for {fact.fact_id!r}."
                    )
                return ProgressAppendResult(
                    fact=existing,
                    accepted=False,
                    durable=True,
                    idempotent=True,
                )
            if fact.progress_sequence_index != len(self._facts) + 1:
                raise ProgressStoreError(
                    "Store append sequence must be contiguous."
                )
            assert self._fd is not None
            encoded = canonical_bytes(
                fact.to_payload(),
                domain=PROGRESS_FACT_FILE_DOMAIN_V1,
            ) + b"\n"
            try:
                _write_all(self._fd, encoded)
                if self.fsync_on_append:
                    os.fsync(self._fd)
            except OSError as exc:
                self._poison(
                    "progress fact append durability is uncertain after a "
                    f"write/fsync failure: {exc}"
                )
                raise ProgressStoreError(
                    "Progress fact append durability is uncertain; close and "
                    "reopen the store before any further operation."
                ) from exc
            self._facts.append(fact)
            self._fact_by_id[fact.fact_id] = fact
            return ProgressAppendResult(
                fact=fact,
                accepted=True,
                durable=self.fsync_on_append,
                idempotent=False,
            )

    def seal(
        self,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> ProgressSealReport:
        with self._lock:
            if self._manifest is not None:
                return ProgressSealReport(
                    manifest=self._manifest,
                    manifest_relative_path=self.manifest_path.name,
                    idempotent=True,
                )
            self._require_open()
            state = self.replay_state()
            if (
                state.phase != "TERMINATED"
                or state.terminal_fact_id is None
                or not self._facts
                or not isinstance(
                    self._facts[-1].payload,
                    EpisodeProgressTerminationV1,
                )
            ):
                raise ProgressStoreError(
                    "Progress facts must end in a valid termination before seal."
                )
            if (
                not isinstance(sealed_at, (int, float))
                or isinstance(sealed_at, bool)
            ):
                raise ValueError("sealed_at must be numeric.")
            seal_value = float(sealed_at)
            if not math.isfinite(seal_value):
                raise ValueError("sealed_at must be finite.")
            if (
                not isinstance(sealed_clock_domain_id, str)
                or not sealed_clock_domain_id.strip()
            ):
                raise ValueError(
                    "sealed_clock_domain_id must be a non-empty string."
                )
            seal_clock = sealed_clock_domain_id.strip()
            if seal_clock != self.binding.bound_clock_domain_id:
                raise ValueError(
                    "Progress seal must use the bound Episode clock."
                )
            terminal = self._facts[-1]
            if seal_value < terminal.effective_at:
                raise ValueError("Progress seal cannot precede termination.")
            assert self._fd is not None
            manifest = ProgressEpisodeManifestV1(
                episode_capture_identity_hash=self.identity.semantic_content_hash,
                task_progress_profile_hash=self.profile.semantic_content_hash,
                progress_episode_binding_hash=self.binding.semantic_content_hash,
                fact_count=len(self._facts),
                fact_bundle_hash=ProgressEpisodeManifestV1.fact_bundle_hash_for(
                    tuple(self._facts)
                ),
                terminal_fact_id=state.terminal_fact_id,
                sealed_at=seal_value,
                sealed_clock_domain_id=seal_clock,
            )
            try:
                os.fsync(self._fd)
                os.close(self._fd)
                self._fd = None
                idempotent = publish_immutable_canonical(
                    self.manifest_path,
                    manifest.to_payload(),
                    domain=PROGRESS_MANIFEST_FILE_DOMAIN_V1,
                )
            except Exception as exc:
                self._poison(
                    "progress seal durability is uncertain after an fsync/"
                    f"close/publication failure: {exc}"
                )
                raise ProgressStoreError(
                    "Progress seal durability is uncertain; close and reopen "
                    "the store before any further operation."
                ) from exc
            self._manifest = manifest
            self._sealed = True
            return ProgressSealReport(
                manifest=manifest,
                manifest_relative_path=self.manifest_path.name,
                idempotent=idempotent,
            )

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._closed = True

    def __enter__(self) -> "ProgressFactStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._poisoned:
            raise ProgressStoreError(
                "Progress store is poisoned after an uncertain persistence "
                "failure; close and reopen it before continuing."
            )
        if self._closed:
            raise ProgressStoreError("Progress store is closed.")
        if self._sealed:
            raise ProgressStoreError("Progress store is sealed.")

    def _poison(self, reason: str) -> None:
        self._poisoned = True
        self._poison_reason = reason
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _load_facts(self) -> None:
        if not self.fact_path.exists():
            return
        encoded = self.fact_path.read_bytes()
        if not encoded:
            return
        if not encoded.endswith(b"\n"):
            raise ProgressStoreIncompleteError(
                "Progress fact log has a torn final record."
            )
        for line_number, line in enumerate(encoded.splitlines(), start=1):
            if not line:
                raise ProgressStoreIncompleteError(
                    f"Progress fact log contains an empty line at {line_number}."
                )
            try:
                payload = canonical_loads(
                    line,
                    expected_domain=PROGRESS_FACT_FILE_DOMAIN_V1,
                )
                fact = ProgressFactEnvelopeV1.from_payload(payload)
            except (TypeError, ValueError) as exc:
                raise ProgressStoreIncompleteError(
                    f"Invalid progress fact at line {line_number}: {exc}"
                ) from exc
            if (
                fact.episode_capture_identity_hash
                != self.identity.semantic_content_hash
            ):
                raise ProgressStoreIncompleteError(
                    "Progress fact Episode identity mismatch."
                )
            if fact.progress_sequence_index != len(self._facts) + 1:
                raise ProgressStoreIncompleteError(
                    "Progress fact sequence is not contiguous."
                )
            if fact.fact_id in self._fact_by_id:
                raise ProgressStoreIncompleteError(
                    f"Duplicate durable progress fact ID {fact.fact_id!r}."
                )
            self._facts.append(fact)
            self._fact_by_id[fact.fact_id] = fact
        try:
            self.replay_state()
        except ValueError as exc:
            raise ProgressStoreIncompleteError(
                f"Durable progress facts do not reduce: {exc}"
            ) from exc

    def _load_manifest_if_present(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            payload = canonical_loads(
                self.manifest_path.read_bytes(),
                expected_domain=PROGRESS_MANIFEST_FILE_DOMAIN_V1,
            )
            manifest = ProgressEpisodeManifestV1.from_payload(payload)
        except (TypeError, ValueError) as exc:
            raise ProgressStoreIncompleteError(
                f"Invalid progress manifest: {exc}"
            ) from exc
        state = self.replay_state()
        if (
            state.phase != "TERMINATED"
            or manifest.episode_capture_identity_hash
            != self.identity.semantic_content_hash
            or manifest.task_progress_profile_hash
            != self.profile.semantic_content_hash
            or manifest.progress_episode_binding_hash
            != self.binding.semantic_content_hash
            or manifest.fact_count != len(self._facts)
            or manifest.fact_bundle_hash
            != ProgressEpisodeManifestV1.fact_bundle_hash_for(tuple(self._facts))
            or manifest.terminal_fact_id != state.terminal_fact_id
            or manifest.sealed_clock_domain_id
            != self.binding.bound_clock_domain_id
            or not self._facts
            or manifest.sealed_at < self._facts[-1].effective_at
        ):
            raise ProgressStoreIncompleteError(
                "Progress manifest does not bind the complete reduced fact log."
            )
        self._manifest = manifest
        self._sealed = True


__all__ = [
    "ProgressAppendResult",
    "ProgressFactStore",
    "ProgressSealReport",
    "ProgressStoreError",
    "ProgressStoreIncompleteError",
]
