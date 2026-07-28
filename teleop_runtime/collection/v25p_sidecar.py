"""Subordinate persistence for the frozen v2.3-based v2.5-p compatibility branch.

The active collection recorder owns episode lifecycle.  This store accepts its
immutable identity, publishes small capability artifacts, and can never begin,
finalize, discard, or commit the legacy dataset episode itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.embodiment import (
    CaptureProjectionProfile,
    EmbodimentCaptureProfile,
    EpisodeEmbodimentBinding,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonTaskProfile,
    SubtaskAttemptEvent,
    TaskSemanticsEpisodeManifest,
    validate_subtask_attempt_events,
)
from dataset_contracts.v25p import (
    V25P_DATASET_MANIFEST_FILE_DOMAIN_V1,
    V25P_EMBODIMENT_BINDING_FILE_DOMAIN_V1,
    V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
    V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    V25P_EPISODE_RELATIVE_ROOT_V1,
    V25P_PROFILE_FILE_DOMAIN_V1,
    V25P_PROFILE_RELATIVE_ROOT_V1,
    V25P_RELATIVE_ROOT_V1,
    V25P_TASK_EVENT_FILE_DOMAIN_V1,
    V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
    V25pDatasetManifest,
    V25pEpisodeManifest,
    v25p_base_episode_manifest_hash,
)

from .immutable_canonical_store import (
    fsync_directory as _fsync_directory,
    publish_immutable_canonical as _publish_canonical,
)


@dataclass(frozen=True)
class V25pEpisodeSidecarFinalizeReport:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    embodiment_binding_hash: str
    task_semantics_manifest_hash: str | None
    episode_manifest_hash: str
    base_episode_manifest_hash: str
    v24_capture_commit_hash: str | None
    auto_closed_attempt_count: int
    episode_manifest_relative_path: str


@dataclass
class _ActiveEpisode:
    identity: EpisodeCaptureIdentity
    binding: EpisodeEmbodimentBinding | None = None
    events: tuple[SubtaskAttemptEvent, ...] = ()


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(payload)
    return tuple(rows)


class V25pEpisodeSidecarStore:
    """Legacy-name store for v2.3-bound compatibility sidecars."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        dataset_id: str,
        embodiment_profile: EmbodimentCaptureProfile,
        capture_projection_profile: CaptureProjectionProfile,
        task_profile: LongHorizonTaskProfile | None = None,
        clock=time.time,
        clock_domain_id: str = "local_wall_clock",
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.root = self.dataset_root / V25P_RELATIVE_ROOT_V1
        self.episode_root = self.dataset_root / V25P_EPISODE_RELATIVE_ROOT_V1
        self.profile_root = self.dataset_root / V25P_PROFILE_RELATIVE_ROOT_V1
        self.dataset_id = str(dataset_id).strip()
        self.embodiment_profile = embodiment_profile
        self.capture_projection_profile = capture_projection_profile
        self.task_profile = task_profile
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        if not self.dataset_id or not self.clock_domain_id:
            raise ValueError("v2.5-p dataset and clock identity must be non-empty.")
        self._active: _ActiveEpisode | None = None
        self._lock = threading.RLock()
        self.dataset_manifest = self._bind_dataset_contracts()

    def _bind_dataset_contracts(self) -> V25pDatasetManifest:
        expected = {
            "dataset_id": self.dataset_id,
            "policy_io_layout_hash": self.capture_projection_profile.policy_io_layout_hash,
            "pose_frame_contract_hash": self.capture_projection_profile.pose_frame_contract_hash,
            "embodiment_profile_hashes": (
                self.embodiment_profile.semantic_content_hash,
            ),
            "capture_projection_profile_hashes": (
                self.capture_projection_profile.semantic_content_hash,
            ),
            "long_horizon_task_profile_hashes": (
                ()
                if self.task_profile is None
                else (self.task_profile.semantic_content_hash,)
            ),
        }
        manifest_path = self.root / "dataset_manifest.cjson"
        manifest_exists = manifest_path.exists()
        if manifest_exists:
            payload = canonical_loads(
                manifest_path.read_bytes(),
                expected_domain=V25P_DATASET_MANIFEST_FILE_DOMAIN_V1,
            )
            if not isinstance(payload, Mapping):
                raise ValueError("Existing v2.5-p dataset manifest is not a mapping.")
            manifest = V25pDatasetManifest.from_payload(payload)
            for name, value in expected.items():
                if getattr(manifest, name) != value:
                    raise ValueError(
                        f"Existing v2.5-p dataset manifest differs at {name}."
                    )
        else:
            manifest = V25pDatasetManifest(
                **expected,
                created_at=float(self.clock()),
                created_clock_domain_id=self.clock_domain_id,
            )

        profile_records = (
            (
                "embodiment",
                self.embodiment_profile.semantic_content_hash,
                self.embodiment_profile.to_payload(),
            ),
            (
                "capture_projection",
                self.capture_projection_profile.semantic_content_hash,
                self.capture_projection_profile.to_payload(),
            ),
        )
        for kind, content_hash, payload in profile_records:
            _publish_canonical(
                self.profile_root / kind / f"{content_hash}.cjson",
                payload,
                domain=V25P_PROFILE_FILE_DOMAIN_V1,
            )
        if self.task_profile is not None:
            _publish_canonical(
                self.profile_root
                / "task"
                / f"{self.task_profile.semantic_content_hash}.cjson",
                self.task_profile.to_payload(),
                domain=V25P_PROFILE_FILE_DOMAIN_V1,
            )
        if not manifest_exists:
            _publish_canonical(
                manifest_path,
                manifest.to_payload(),
                domain=V25P_DATASET_MANIFEST_FILE_DOMAIN_V1,
            )
        return manifest

    def _episode_dir(self, identity: EpisodeCaptureIdentity) -> Path:
        return self.episode_root / identity.episode_uuid

    def prepare_episode(self, identity: EpisodeCaptureIdentity) -> None:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("A v2.5-p sidecar episode is already active.")
            episode_dir = self._episode_dir(identity)
            if (episode_dir / "episode_manifest.cjson").exists():
                raise RuntimeError("The v2.5-p episode is already sealed.")
            if episode_dir.exists():
                raise RuntimeError(
                    "An unsealed v2.5-p episode directory already exists; audit or "
                    "discard it explicitly before reusing the UUID."
                )
            self._active = _ActiveEpisode(identity=identity)

    def bind_prepared_episode(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        bound_at: float,
        bound_clock_domain_id: str,
        teleop_input_profile_hash: str | None = None,
    ) -> EpisodeEmbodimentBinding:
        with self._lock:
            active = self._require_active(identity)
            if active.binding is not None:
                raise RuntimeError("The v2.5-p episode is already bound.")
            binding = EpisodeEmbodimentBinding(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                episode_identity_hash=identity.semantic_content_hash,
                embodiment_profile_hash=self.embodiment_profile.semantic_content_hash,
                capture_projection_profile_hash=(
                    self.capture_projection_profile.semantic_content_hash
                ),
                teleop_input_profile_hash=teleop_input_profile_hash,
                bound_at=bound_at,
                bound_clock_domain_id=bound_clock_domain_id,
            )
            episode_dir = self._episode_dir(identity)
            _publish_canonical(
                episode_dir / "identity.cjson",
                identity.to_payload(),
                domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
            )
            _publish_canonical(
                episode_dir / "embodiment_binding.cjson",
                binding.to_payload(),
                domain=V25P_EMBODIMENT_BINDING_FILE_DOMAIN_V1,
            )
            active.binding = binding
            return binding

    def begin_subtask(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        track_id: str,
        attempt_id: str,
        subtask_id: str,
        effective_at: float,
        available_at: float,
        effective_clock_domain_id: str,
        available_clock_domain_id: str,
        actor_id: str,
        frame_index: int | None = None,
        origin: str = "online_operator",
        source_event_ids: tuple[str, ...] = (),
    ) -> SubtaskAttemptEvent:
        return self._append_task_event(
            identity,
            track_id=track_id,
            attempt_id=attempt_id,
            subtask_id=subtask_id,
            event_kind="begin",
            outcome=None,
            effective_at=effective_at,
            available_at=available_at,
            effective_clock_domain_id=effective_clock_domain_id,
            available_clock_domain_id=available_clock_domain_id,
            actor_id=actor_id,
            frame_index=frame_index,
            origin=origin,
            source_event_ids=source_event_ids,
        )

    def finish_subtask(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        track_id: str,
        attempt_id: str,
        subtask_id: str,
        outcome: str,
        effective_at: float,
        available_at: float,
        effective_clock_domain_id: str,
        available_clock_domain_id: str,
        actor_id: str,
        frame_index: int | None = None,
        origin: str = "online_operator",
        source_event_ids: tuple[str, ...] = (),
    ) -> SubtaskAttemptEvent:
        return self._append_task_event(
            identity,
            track_id=track_id,
            attempt_id=attempt_id,
            subtask_id=subtask_id,
            event_kind="finish",
            outcome=outcome,
            effective_at=effective_at,
            available_at=available_at,
            effective_clock_domain_id=effective_clock_domain_id,
            available_clock_domain_id=available_clock_domain_id,
            actor_id=actor_id,
            frame_index=frame_index,
            origin=origin,
            source_event_ids=source_event_ids,
        )

    def _append_task_event(
        self,
        identity: EpisodeCaptureIdentity,
        **values: Any,
    ) -> SubtaskAttemptEvent:
        with self._lock:
            active = self._require_bound(identity)
            if self.task_profile is None:
                raise RuntimeError("No long-horizon task profile is configured.")
            sequence = len(active.events) + 1
            event = SubtaskAttemptEvent(
                event_id=f"task-event:{identity.episode_uuid}:{sequence}",
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                event_sequence=sequence,
                **values,
            )
            candidate = active.events + (event,)
            validate_subtask_attempt_events(
                self.task_profile,
                candidate,
                require_closed=False,
            )
            event_path = (
                self._episode_dir(identity)
                / "task_events"
                / f"{sequence:08d}.cjson"
            )
            _publish_canonical(
                event_path,
                event.to_payload(),
                domain=V25P_TASK_EVENT_FILE_DOMAIN_V1,
            )
            active.events = candidate
            return event

    def finalize_after_legacy_episode(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
        v24_capture_commit_hash: str | None = None,
        task_effective_at: float | None = None,
        task_effective_clock_domain_id: str | None = None,
    ) -> V25pEpisodeSidecarFinalizeReport:
        with self._lock:
            active = self._require_bound(identity)
            auto_closed = self._auto_close_open_attempts(
                identity,
                effective_at=(
                    sealed_at if task_effective_at is None else task_effective_at
                ),
                effective_clock_domain_id=(
                    sealed_clock_domain_id
                    if task_effective_clock_domain_id is None
                    else task_effective_clock_domain_id
                ),
            )
            task_manifest = None
            if self.task_profile is not None:
                task_manifest = TaskSemanticsEpisodeManifest.create(
                    profile=self.task_profile,
                    episode_uuid=identity.episode_uuid,
                    legacy_episode_index=identity.legacy_episode_index,
                    episode_identity_hash=identity.semantic_content_hash,
                    events=active.events,
                    sealed_at=sealed_at,
                    sealed_clock_domain_id=sealed_clock_domain_id,
                )
                _publish_canonical(
                    self._episode_dir(identity) / "task_manifest.cjson",
                    task_manifest.to_payload(),
                    domain=V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
                )
            base_record = self._finalized_base_episode_record(
                identity.legacy_episode_index
            )
            base_hash = v25p_base_episode_manifest_hash(base_record)
            assert active.binding is not None
            episode_manifest = V25pEpisodeManifest(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                episode_identity_hash=identity.semantic_content_hash,
                embodiment_binding_hash=active.binding.semantic_content_hash,
                base_episode_manifest_hash=base_hash,
                task_semantics_manifest_hash=(
                    None
                    if task_manifest is None
                    else task_manifest.semantic_content_hash
                ),
                v24_capture_commit_hash=v24_capture_commit_hash,
                sealed_at=sealed_at,
                sealed_clock_domain_id=sealed_clock_domain_id,
            )
            manifest_path = self._episode_dir(identity) / "episode_manifest.cjson"
            _publish_canonical(
                manifest_path,
                episode_manifest.to_payload(),
                domain=V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1,
            )
            report = V25pEpisodeSidecarFinalizeReport(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                episode_identity_hash=identity.semantic_content_hash,
                embodiment_binding_hash=active.binding.semantic_content_hash,
                task_semantics_manifest_hash=(
                    None
                    if task_manifest is None
                    else task_manifest.semantic_content_hash
                ),
                episode_manifest_hash=episode_manifest.semantic_content_hash,
                base_episode_manifest_hash=base_hash,
                v24_capture_commit_hash=v24_capture_commit_hash,
                auto_closed_attempt_count=auto_closed,
                episode_manifest_relative_path=manifest_path.relative_to(
                    self.dataset_root
                ).as_posix(),
            )
            self._active = None
            return report

    def _auto_close_open_attempts(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        effective_at: float,
        effective_clock_domain_id: str,
    ) -> int:
        if self.task_profile is None:
            return 0
        active = self._require_bound(identity)
        open_by_track: dict[str, SubtaskAttemptEvent] = {}
        for event in sorted(
            active.events,
            key=lambda value: (value.effective_at, value.event_sequence),
        ):
            if event.event_kind == "begin":
                open_by_track[event.track_id] = event
            else:
                open_by_track.pop(event.track_id, None)
        for begin in tuple(open_by_track.values()):
            if begin.effective_clock_domain_id != effective_clock_domain_id:
                raise RuntimeError(
                    "Auto-closing an open task attempt requires an episode-end "
                    "timestamp in the task event effective clock domain."
                )
            finish_at = max(float(effective_at), float(begin.effective_at))
            self._append_task_event(
                identity,
                track_id=begin.track_id,
                attempt_id=begin.attempt_id,
                subtask_id=begin.subtask_id,
                event_kind="finish",
                outcome="unknown",
                effective_at=finish_at,
                available_at=finish_at,
                effective_clock_domain_id=effective_clock_domain_id,
                available_clock_domain_id=effective_clock_domain_id,
                actor_id="episode_owner:auto_close",
                frame_index=None,
                origin=begin.origin,
                source_event_ids=(begin.event_id,),
            )
        return len(open_by_track)

    def discard_episode(self, identity: EpisodeCaptureIdentity) -> None:
        with self._lock:
            active = self._require_active(identity)
            episode_dir = self._episode_dir(identity)
            if (episode_dir / "episode_manifest.cjson").exists():
                raise RuntimeError("A sealed v2.5-p episode cannot be discarded.")
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
                if self.episode_root.exists():
                    _fsync_directory(self.episode_root)
            if self._active is active:
                self._active = None

    def close_unsealed(self, identity: EpisodeCaptureIdentity) -> None:
        """Release in-memory state after legacy finalize failure; retain evidence."""

        with self._lock:
            self._require_active(identity)
            self._active = None

    def _require_active(self, identity: EpisodeCaptureIdentity) -> _ActiveEpisode:
        active = self._active
        if active is None or active.identity != identity:
            raise ValueError("Episode identity is not the active v2.5-p episode.")
        return active

    def _require_bound(self, identity: EpisodeCaptureIdentity) -> _ActiveEpisode:
        active = self._require_active(identity)
        if active.binding is None:
            raise RuntimeError("The active v2.5-p episode is not bound.")
        return active

    def _finalized_base_episode_record(self, episode_index: int) -> dict[str, Any]:
        matches = tuple(
            row
            for row in _read_jsonl(self.dataset_root / "meta" / "episodes.jsonl")
            if int(row.get("episode_index", -1)) == int(episode_index)
        )
        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one finalized legacy episode record for "
                f"episode_index={episode_index}; found {len(matches)}."
            )
        return dict(matches[0])


V25pV23CompatibilitySidecarStore = V25pEpisodeSidecarStore


__all__ = [
    "V25P_TASK_EVENT_FILE_DOMAIN_V1",
    "V25pEpisodeSidecarFinalizeReport",
    "V25pEpisodeSidecarStore",
    "V25pV23CompatibilitySidecarStore",
]
