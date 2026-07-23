"""Subordinate per-episode binding and seal assessment for ControlTrace."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.completeness import evaluate_control_trace_lineage
from dataset_contracts.control_trace_episode import (
    EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1,
    EPISODE_RANGE_RECORD_HASH_DOMAIN_V1,
    EpisodeTraceDependencyRef,
    EpisodeTraceRange,
    EpisodeTraceSealAssessment,
    V24EpisodeCaptureCommit,
)
from dataset_contracts.stream_catalog import (
    NativeStreamEpisodeManifest,
    NativeStreamExternalArtifactReceipt,
)

from .lifecycle import EpisodeTraceMembership, EpisodeTraceMembershipRelease
from .episode_manifest import EpisodeCaptureManifestStore, EpisodeManifestPublishReceipt
from .runtime import ControlTraceRuntime, TraceAdmissionReceipt


def _gap_ranges(values: set[int]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    ordered = sorted(values)
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous))
        start = previous = value
    result.append((start, previous))
    return tuple(result)


@dataclass(frozen=True)
class EpisodeTraceBindingReceipt:
    membership: EpisodeTraceMembership
    admission_receipt: TraceAdmissionReceipt


@dataclass(frozen=True)
class EpisodeTraceReleaseReceipt:
    trace_range: EpisodeTraceRange
    admission_receipt: TraceAdmissionReceipt | None


@dataclass(frozen=True)
class PreparedEpisodeTraceCapture:
    episode_uuid: str
    legacy_episode_index: int
    binding_event_id: str
    unbinding_event_id: str
    seal_assessment_id: str
    prepared_at: float


@dataclass(frozen=True)
class ReleasedEpisodeTraceCapture:
    preparation: PreparedEpisodeTraceCapture
    release: EpisodeTraceReleaseReceipt
    seal: EpisodeTraceSealAssessment
    seal_publish_receipt: EpisodeManifestPublishReceipt | None = None
    seal_publish_error: str = ""
    native_stream_manifest: NativeStreamEpisodeManifest | None = None
    native_stream_seal_error: str = ""
    native_stream_pending_finalize: bool = False


@dataclass(frozen=True)
class EpisodeTraceLegacyFinalizeReport:
    episode_uuid: str
    legacy_episode_index: int
    legacy_finalize_status: str
    trace_seal_status: str
    capture_commit_candidate: V24EpisodeCaptureCommit | None
    capture_commit_status: str
    capture_commit_publish_receipt: EpisodeManifestPublishReceipt | None
    completed_at: float
    message: str
    native_stream_manifest_hash: str | None = None
    native_stream_seal_error: str = ""


class NativeStreamEpisodeCapture(Protocol):
    """Minimal subordinate storage lifecycle used by the episode owner."""

    episode_uuid: str
    legacy_episode_index: int

    def seal(
        self,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest: ...

    def close_unsealed(self) -> None: ...


@runtime_checkable
class PostFinalizeNativeStreamEpisodeCapture(Protocol):
    """Opt-in external-artifact lifecycle; absent on legacy inline writers."""

    episode_uuid: str
    legacy_episode_index: int
    requires_post_finalize_seal: bool

    def freeze_ingestion(self) -> None: ...

    def seal_after_legacy_finalize(
        self,
        receipts: Iterable[NativeStreamExternalArtifactReceipt],
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest: ...

    def close_unsealed(self) -> None: ...


class PendingEpisodeTraceRelease:
    """One FIFO boundary result produced outside the control thread."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: ReleasedEpisodeTraceCapture | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def set_result(self, result: ReleasedEpisodeTraceCapture) -> None:
        with self._lock:
            if self._event.is_set():
                raise RuntimeError("Pending episode release was already completed.")
            self._result = result
            self._event.set()

    def set_error(self, error: BaseException) -> None:
        with self._lock:
            if self._event.is_set():
                raise RuntimeError("Pending episode release was already completed.")
            self._error = error
            self._event.set()

    def resolve(self, *, timeout_sec: float = 5.0) -> ReleasedEpisodeTraceCapture:
        if not self._event.wait(float(timeout_sec)):
            raise TimeoutError("Episode trace release barrier did not complete before deadline.")
        if self._error is not None:
            raise RuntimeError("Episode trace release barrier failed.") from self._error
        if self._result is None:
            raise RuntimeError("Episode trace release barrier completed without a result.")
        return self._result


class SubordinateEpisodeTraceLifecycle:
    """Maps one external owner onto trace membership without owning episodes."""

    def __init__(
        self,
        runtime: ControlTraceRuntime,
        *,
        lifecycle_owner_id: str,
        initial_membership_epoch: int = 0,
    ) -> None:
        owner = str(lifecycle_owner_id).strip()
        if not owner:
            raise ValueError("lifecycle_owner_id must be non-empty.")
        epoch = int(initial_membership_epoch)
        if epoch < 0:
            raise ValueError("initial_membership_epoch must be non-negative.")
        self.runtime = runtime
        self.lifecycle_owner_id = owner
        self._membership_epoch = epoch
        self._active_binding: EpisodeTraceBindingReceipt | None = None
        self._lock = threading.RLock()

    @property
    def active_episode_uuid(self) -> str | None:
        with self._lock:
            active = self._active_binding
            return None if active is None else active.membership.episode_uuid

    def bind_external_episode(
        self,
        *,
        episode_uuid: str,
        binding_event_id: str,
        bound_at: float,
    ) -> EpisodeTraceBindingReceipt:
        with self._lock:
            if self._active_binding is not None:
                raise RuntimeError("A ControlTrace episode is already bound.")
            self._membership_epoch += 1
            membership = EpisodeTraceMembership(
                episode_uuid=episode_uuid,
                lifecycle_owner_id=self.lifecycle_owner_id,
                binding_event_id=binding_event_id,
                membership_epoch=self._membership_epoch,
                bound_at=bound_at,
            )
            receipt = self.runtime.bind_episode(membership)
            result = EpisodeTraceBindingReceipt(
                membership=membership,
                admission_receipt=receipt,
            )
            if not receipt.accepted:
                raise RuntimeError(
                    "ControlTrace rejected episode bind: "
                    f"{receipt.rejection_code or receipt.message}"
                )
            self._active_binding = result
            return result

    def release_external_episode(
        self,
        *,
        unbinding_event_id: str,
        unbound_at: float,
        reason: str,
    ) -> EpisodeTraceReleaseReceipt:
        with self._lock:
            active = self._active_binding
            if active is None:
                raise RuntimeError("No ControlTrace episode is currently bound.")
            self._membership_epoch += 1
            release = EpisodeTraceMembershipRelease(
                episode_uuid=active.membership.episode_uuid,
                lifecycle_owner_id=self.lifecycle_owner_id,
                unbinding_event_id=unbinding_event_id,
                membership_epoch=self._membership_epoch,
                unbound_at=unbound_at,
                reason=reason,
            )
            receipt = self.runtime.clear_episode_membership(release)
            last_sequence = (
                self.runtime.snapshot().journal_issued_sequence
                if receipt is None
                else receipt.journal_issued_sequence
            )
            trace_range = EpisodeTraceRange(
                trace_session_id=self.runtime.session.trace_session_id,
                runtime_session_id=self.runtime.session.runtime_session_id,
                control_domain_id=self.runtime.session.control_domain_id,
                clock_domain_id=self.runtime.session.clock_domain_id,
                episode_uuid=active.membership.episode_uuid,
                lifecycle_owner_id=self.lifecycle_owner_id,
                binding_event_id=active.membership.binding_event_id,
                unbinding_event_id=unbinding_event_id,
                binding_membership_epoch=active.membership.membership_epoch,
                unbinding_membership_epoch=release.membership_epoch,
                first_journal_sequence=active.admission_receipt.journal_issued_sequence,
                last_journal_sequence=last_sequence,
                bound_at=active.membership.bound_at,
                unbound_at=release.unbound_at,
            )
            self._active_binding = None
            return EpisodeTraceReleaseReceipt(
                trace_range=trace_range,
                admission_receipt=receipt,
            )


class EpisodeTraceRecorderSink:
    """Optional recorder sink; it never calls the legacy writer or publishes files."""

    def __init__(
        self,
        runtime: ControlTraceRuntime,
        *,
        lifecycle_owner_id: str,
        flush_to_durable: Callable[[], object] | None = None,
        dependency_sequences: Callable[[ControlTraceRuntime], Iterable[int]] | None = None,
        uuid_factory: Callable[[], object] = uuid.uuid4,
        control_clock: Callable[[], float] = time.monotonic,
        commit_clock: Callable[[], float] = time.time,
        commit_clock_domain_id: str = "local_wall_clock",
        manifest_store: EpisodeCaptureManifestStore | None = None,
        release_scheduler: Callable[
            [Callable[[], ReleasedEpisodeTraceCapture]], PendingEpisodeTraceRelease
        ]
        | None = None,
        release_timeout_sec: float = 5.0,
        legacy_schema: str = "lerobot_v2.3_compatible",
        committed_capabilities: tuple[str, ...] = ("control_trace_core",),
        native_stream_writer_factory: Callable[
            [PreparedEpisodeTraceCapture], NativeStreamEpisodeCapture
        ]
        | None = None,
    ) -> None:
        self.runtime = runtime
        self.lifecycle = SubordinateEpisodeTraceLifecycle(
            runtime,
            lifecycle_owner_id=lifecycle_owner_id,
        )
        self.flush_to_durable = flush_to_durable
        self.dependency_sequences = dependency_sequences or (lambda _runtime: ())
        self.uuid_factory = uuid_factory
        self.control_clock = control_clock
        self.commit_clock = commit_clock
        self.commit_clock_domain_id = str(commit_clock_domain_id).strip()
        self.manifest_store = manifest_store
        self.release_scheduler = release_scheduler
        self.release_timeout_sec = float(release_timeout_sec)
        if self.release_timeout_sec <= 0.0:
            raise ValueError("release_timeout_sec must be positive.")
        self.legacy_schema = str(legacy_schema).strip()
        self.committed_capabilities = tuple(
            dict.fromkeys(str(value) for value in committed_capabilities)
        )
        self.native_stream_writer_factory = native_stream_writer_factory
        if not self.legacy_schema or not self.committed_capabilities or not self.commit_clock_domain_id:
            raise ValueError("EpisodeTraceRecorderSink commit identity is incomplete.")
        native_capability = "native_streams" in self.committed_capabilities
        if native_capability != (self.native_stream_writer_factory is not None):
            raise ValueError(
                "native_streams capability and native_stream_writer_factory "
                "must be enabled together."
            )
        self._prepared: PreparedEpisodeTraceCapture | None = None
        self._native_stream_capture: NativeStreamEpisodeCapture | None = None
        self._release_pending = False
        self._released_by_episode_index: dict[int, ReleasedEpisodeTraceCapture] = {}
        self._native_pending_finalize_by_episode_index: dict[
            int, PostFinalizeNativeStreamEpisodeCapture
        ] = {}
        self._lock = threading.RLock()

    def control_now(self) -> float:
        return float(self.control_clock())

    def commit_now(self) -> float:
        return float(self.commit_clock())

    def prepare_episode(
        self,
        *,
        legacy_episode_index: int,
        prepared_at: float,
    ) -> PreparedEpisodeTraceCapture:
        with self._lock:
            if self._prepared is not None or self.lifecycle.active_episode_uuid is not None:
                raise RuntimeError("A ControlTrace episode preparation is already active.")
            index = int(legacy_episode_index)
            if index < 0:
                raise ValueError("legacy_episode_index must be non-negative.")
            episode_uuid = str(self.uuid_factory()).strip()
            if not episode_uuid:
                raise ValueError("uuid_factory returned an empty episode UUID.")
            prepared = PreparedEpisodeTraceCapture(
                episode_uuid=episode_uuid,
                legacy_episode_index=index,
                binding_event_id=f"episode-bind:{episode_uuid}",
                unbinding_event_id=f"episode-unbind:{episode_uuid}",
                seal_assessment_id=f"episode-seal:{episode_uuid}",
                prepared_at=float(prepared_at),
            )
            native_capture = (
                None
                if self.native_stream_writer_factory is None
                else self.native_stream_writer_factory(prepared)
            )
            if native_capture is not None and (
                str(native_capture.episode_uuid) != prepared.episode_uuid
                or int(native_capture.legacy_episode_index)
                != prepared.legacy_episode_index
            ):
                native_capture.close_unsealed()
                raise ValueError(
                    "Native stream capture must use the prepared episode "
                    "UUID and legacy index."
                )
            self._prepared = prepared
            self._native_stream_capture = native_capture
            return prepared

    def cancel_prepared_episode(self, prepared: PreparedEpisodeTraceCapture) -> None:
        with self._lock:
            if self._prepared == prepared and self.lifecycle.active_episode_uuid is None:
                native_capture = self._native_stream_capture
                if native_capture is not None:
                    native_capture.close_unsealed()
                self._native_stream_capture = None
                self._prepared = None

    def native_stream_capture(
        self,
        prepared: PreparedEpisodeTraceCapture,
    ) -> NativeStreamEpisodeCapture:
        """Return the opt-in writer bound to the current prepared episode."""

        with self._lock:
            if self._prepared != prepared:
                raise ValueError("Prepared ControlTrace episode is not current.")
            capture = self._native_stream_capture
            if capture is None:
                raise RuntimeError("Native stream capture is not enabled.")
            return capture

    def bind_prepared_episode(
        self,
        prepared: PreparedEpisodeTraceCapture,
        *,
        bound_at: float,
    ) -> EpisodeTraceBindingReceipt:
        with self._lock:
            if self._prepared != prepared:
                raise ValueError("Prepared ControlTrace episode is not current.")
            receipt = self.lifecycle.bind_external_episode(
                episode_uuid=prepared.episode_uuid,
                binding_event_id=prepared.binding_event_id,
                bound_at=bound_at,
            )
            return receipt

    def _release_episode_now(
        self,
        prepared: PreparedEpisodeTraceCapture,
        *,
        unbound_at: float,
        reason: str,
    ) -> ReleasedEpisodeTraceCapture:
        with self._lock:
            if self._prepared != prepared:
                raise ValueError("Prepared ControlTrace episode is not current.")
            release = self.lifecycle.release_external_episode(
                unbinding_event_id=prepared.unbinding_event_id,
                unbound_at=unbound_at,
                reason=reason,
            )
            native_capture = self._native_stream_capture
            self._native_stream_capture = None
            self._prepared = None
            self._release_pending = False
        if self.flush_to_durable is not None:
            self.flush_to_durable()
        native_stream_manifest = None
        native_stream_seal_error = ""
        native_stream_pending_finalize = False
        if native_capture is not None:
            post_finalize_capture = (
                native_capture
                if isinstance(
                    native_capture,
                    PostFinalizeNativeStreamEpisodeCapture,
                )
                and native_capture.requires_post_finalize_seal
                else None
            )
            if post_finalize_capture is not None:
                try:
                    post_finalize_capture.freeze_ingestion()
                    native_stream_pending_finalize = True
                    with self._lock:
                        self._native_pending_finalize_by_episode_index[
                            prepared.legacy_episode_index
                        ] = post_finalize_capture
                except BaseException as exc:
                    native_stream_seal_error = f"{type(exc).__name__}: {exc}"
                    post_finalize_capture.close_unsealed()
            else:
                try:
                    native_stream_manifest = native_capture.seal(
                        sealed_at=unbound_at,
                        sealed_clock_domain_id=(
                            self.runtime.session.clock_domain_id
                        ),
                    )
                except BaseException as exc:
                    native_stream_seal_error = f"{type(exc).__name__}: {exc}"
                    native_capture.close_unsealed()
        seal = assess_episode_trace_seal(
            self.runtime,
            release.trace_range,
            assessment_id=prepared.seal_assessment_id,
            assessed_at=unbound_at,
            dependency_sequences=tuple(self.dependency_sequences(self.runtime)),
        )
        seal_publish_receipt = None
        seal_publish_error = ""
        if self.manifest_store is not None:
            try:
                seal_publish_receipt = self.manifest_store.publish_seal_assessment(seal)
            except BaseException as exc:  # legacy finalize remains independent
                seal_publish_error = f"{type(exc).__name__}: {exc}"
        released = ReleasedEpisodeTraceCapture(
            preparation=prepared,
            release=release,
            seal=seal,
            seal_publish_receipt=seal_publish_receipt,
            seal_publish_error=seal_publish_error,
            native_stream_manifest=native_stream_manifest,
            native_stream_seal_error=native_stream_seal_error,
            native_stream_pending_finalize=native_stream_pending_finalize,
        )
        with self._lock:
            self._released_by_episode_index[prepared.legacy_episode_index] = released
        return released

    def release_episode(
        self,
        prepared: PreparedEpisodeTraceCapture,
        *,
        unbound_at: float,
        reason: str,
    ) -> ReleasedEpisodeTraceCapture | PendingEpisodeTraceRelease:
        if self.release_scheduler is None:
            return self._release_episode_now(
                prepared,
                unbound_at=unbound_at,
                reason=reason,
            )
        with self._lock:
            if self._prepared != prepared:
                raise ValueError("Prepared ControlTrace episode is not current.")
            if self._release_pending:
                raise RuntimeError("A ControlTrace episode release is already pending.")
            self._release_pending = True
        return self.release_scheduler(
            lambda: self._release_episode_now(
                prepared,
                unbound_at=unbound_at,
                reason=reason,
            )
        )

    def resolve_release(
        self,
        released: ReleasedEpisodeTraceCapture | PendingEpisodeTraceRelease | None,
    ) -> ReleasedEpisodeTraceCapture | None:
        if released is None:
            return None
        if isinstance(released, PendingEpisodeTraceRelease):
            return released.resolve(timeout_sec=self.release_timeout_sec)
        if not isinstance(released, ReleasedEpisodeTraceCapture):
            raise TypeError("Unsupported episode trace release handle.")
        return released

    def legacy_finalize_completed(
        self,
        released: ReleasedEpisodeTraceCapture | PendingEpisodeTraceRelease,
        *,
        completed_at: float,
        native_stream_artifact_receipts: Iterable[
            NativeStreamExternalArtifactReceipt
        ] = (),
    ) -> EpisodeTraceLegacyFinalizeReport:
        resolved = self.resolve_release(released)
        if resolved is None:
            raise ValueError("Legacy finalize has no ControlTrace episode release.")
        released = resolved
        prepared = released.preparation
        external_receipts = tuple(native_stream_artifact_receipts)
        if external_receipts and not released.native_stream_pending_finalize:
            raise ValueError(
                "External native stream receipts were supplied without a "
                "pending post-finalize capture."
            )
        if released.native_stream_pending_finalize:
            with self._lock:
                pending_capture = (
                    self._native_pending_finalize_by_episode_index.pop(
                        prepared.legacy_episode_index,
                        None,
                    )
                )
            native_stream_manifest = None
            native_stream_seal_error = ""
            if pending_capture is None:
                native_stream_seal_error = (
                    "RuntimeError: missing post-finalize native stream capture"
                )
            else:
                try:
                    native_stream_manifest = (
                        pending_capture.seal_after_legacy_finalize(
                            external_receipts,
                            sealed_at=self.control_now(),
                            sealed_clock_domain_id=(
                                self.runtime.session.clock_domain_id
                            ),
                        )
                    )
                except BaseException as exc:
                    native_stream_seal_error = f"{type(exc).__name__}: {exc}"
                    pending_capture.close_unsealed()
            released = replace(
                released,
                native_stream_manifest=native_stream_manifest,
                native_stream_seal_error=native_stream_seal_error,
                native_stream_pending_finalize=False,
            )
            with self._lock:
                self._released_by_episode_index[
                    prepared.legacy_episode_index
                ] = released
        with self._lock:
            if self._released_by_episode_index.get(prepared.legacy_episode_index) != released:
                raise ValueError("Released ControlTrace episode is not current for this index.")
            native_required = "native_streams" in self.committed_capabilities
            native_ready = (
                not native_required
                or released.native_stream_manifest is not None
            )
            commit_candidate = (
                V24EpisodeCaptureCommit.create(
                    commit_id=f"capture-commit:{prepared.episode_uuid}",
                    legacy_episode_index=prepared.legacy_episode_index,
                    legacy_schema=self.legacy_schema,
                    seal=released.seal,
                    capabilities=self.committed_capabilities,
                    committed_at=completed_at,
                    committed_clock_domain_id=self.commit_clock_domain_id,
                    capability_artifact_hash_by_name=(
                        {}
                        if released.native_stream_manifest is None
                        else {
                            "native_streams": (
                                released.native_stream_manifest.semantic_content_hash
                            )
                        }
                    ),
                )
                if released.seal.capture_commit_eligible and native_ready
                else None
            )
            commit_publish_receipt = None
            commit_status = "not_eligible"
            message = "legacy_finalized_without_v24_capture_commit"
            if (
                released.seal.capture_commit_eligible
                and native_required
                and not native_ready
            ):
                message = (
                    "native_streams_not_sealed:"
                    + (
                        released.native_stream_seal_error
                        or "missing_native_stream_manifest"
                    )
                )
            if commit_candidate is not None:
                commit_status = "candidate_only"
                message = "capture_commit_candidate_ready"
                if self.manifest_store is not None:
                    if released.seal_publish_receipt is None:
                        commit_status = "publication_failed"
                        message = (
                            "seal_assessment_not_durable:"
                            + (released.seal_publish_error or "missing_publish_receipt")
                        )
                    else:
                        try:
                            commit_publish_receipt = self.manifest_store.publish_capture_commit(
                                commit_candidate
                            )
                            commit_status = "committed"
                            message = "capture_commit_durable"
                        except BaseException as exc:  # legacy finalize remains successful
                            commit_status = "publication_failed"
                            message = f"capture_commit_publish:{type(exc).__name__}: {exc}"
            self._released_by_episode_index.pop(prepared.legacy_episode_index, None)
        return EpisodeTraceLegacyFinalizeReport(
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            legacy_finalize_status="finalized",
            trace_seal_status=("sealed" if released.seal.trace_sealed else "unsealed"),
            capture_commit_candidate=commit_candidate,
            capture_commit_status=commit_status,
            capture_commit_publish_receipt=commit_publish_receipt,
            completed_at=float(completed_at),
            message=message,
            native_stream_manifest_hash=(
                None
                if released.native_stream_manifest is None
                else released.native_stream_manifest.semantic_content_hash
            ),
            native_stream_seal_error=released.native_stream_seal_error,
        )

    def legacy_finalize_failed(
        self,
        released: ReleasedEpisodeTraceCapture | PendingEpisodeTraceRelease | None,
        *,
        completed_at: float,
        error: str,
    ) -> EpisodeTraceLegacyFinalizeReport | None:
        released = self.resolve_release(released)
        if released is None:
            return None
        prepared = released.preparation
        with self._lock:
            pending_capture = self._native_pending_finalize_by_episode_index.pop(
                prepared.legacy_episode_index,
                None,
            )
            self._released_by_episode_index.pop(prepared.legacy_episode_index, None)
        if pending_capture is not None:
            pending_capture.close_unsealed()
        return EpisodeTraceLegacyFinalizeReport(
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            legacy_finalize_status="failed",
            trace_seal_status=("sealed" if released.seal.trace_sealed else "unsealed"),
            capture_commit_candidate=None,
            capture_commit_status="legacy_finalize_failed",
            capture_commit_publish_receipt=None,
            completed_at=float(completed_at),
            message=str(error),
            native_stream_manifest_hash=(
                None
                if released.native_stream_manifest is None
                else released.native_stream_manifest.semantic_content_hash
            ),
            native_stream_seal_error=released.native_stream_seal_error,
        )

    def legacy_episode_discarded(
        self,
        released: ReleasedEpisodeTraceCapture | PendingEpisodeTraceRelease | None,
        *,
        completed_at: float,
    ) -> EpisodeTraceLegacyFinalizeReport | None:
        released = self.resolve_release(released)
        if released is None:
            return None
        prepared = released.preparation
        with self._lock:
            pending_capture = self._native_pending_finalize_by_episode_index.pop(
                prepared.legacy_episode_index,
                None,
            )
            self._released_by_episode_index.pop(prepared.legacy_episode_index, None)
        if pending_capture is not None:
            pending_capture.close_unsealed()
        return EpisodeTraceLegacyFinalizeReport(
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            legacy_finalize_status="discarded",
            trace_seal_status=("sealed" if released.seal.trace_sealed else "unsealed"),
            capture_commit_candidate=None,
            capture_commit_status="legacy_episode_discarded",
            capture_commit_publish_receipt=None,
            completed_at=float(completed_at),
            message="legacy_episode_discarded",
            native_stream_manifest_hash=(
                None
                if released.native_stream_manifest is None
                else released.native_stream_manifest.semantic_content_hash
            ),
            native_stream_seal_error=released.native_stream_seal_error,
        )

    def defer_legacy_episode_discarded(
        self,
        released: ReleasedEpisodeTraceCapture | PendingEpisodeTraceRelease | None,
        *,
        completed_at: float,
    ) -> None:
        thread = threading.Thread(
            target=lambda: self.legacy_episode_discarded(
                released,
                completed_at=completed_at,
            ),
            name="control-trace-empty-episode-release",
            daemon=True,
        )
        thread.start()


def assess_episode_trace_seal(
    runtime: ControlTraceRuntime,
    trace_range: EpisodeTraceRange,
    *,
    assessment_id: str,
    assessed_at: float,
    dependency_sequences: Iterable[int] = (),
) -> EpisodeTraceSealAssessment:
    """Assess one immutable episode range without terminating the runtime session."""

    session = runtime.session
    if trace_range.trace_session_id != session.trace_session_id:
        raise ValueError("Episode trace range belongs to another trace session.")
    if trace_range.runtime_session_id != session.runtime_session_id:
        raise ValueError("Episode trace range belongs to another runtime session.")
    if trace_range.control_domain_id != session.control_domain_id:
        raise ValueError("Episode trace range belongs to another control domain.")

    entries = {item.journal_sequence: item for item in runtime.journal_entries()}
    records = {item.journal_sequence: item for item in runtime.records()}
    snapshot = runtime.snapshot()
    expected = set(
        range(trace_range.first_journal_sequence, trace_range.last_journal_sequence + 1)
    )
    accepted = set(snapshot.accepted_sequences)
    durable = set(snapshot.durable_sequences)
    admission_gaps = expected - accepted
    durability_gaps = (expected & accepted) - durable
    range_entries = [entries[sequence] for sequence in sorted(expected & set(entries))]

    dependency_values = tuple(sorted({int(value) for value in dependency_sequences}))
    if any(value <= 0 for value in dependency_values):
        raise ValueError("dependency_sequences must be positive.")
    if expected & set(dependency_values):
        raise ValueError("Episode dependencies must be outside the episode journal range.")
    missing_dependencies = set(dependency_values) - set(entries)
    non_durable_dependencies = (set(dependency_values) & accepted) - durable
    dependency_entries = [
        entries[sequence]
        for sequence in dependency_values
        if sequence in entries
    ]
    dependency_refs = tuple(
        EpisodeTraceDependencyRef(
            fact_id=entry.fact.fact_id,
            journal_sequence=entry.journal_sequence,
            record_integrity_hash=records[entry.journal_sequence].record_integrity_hash,
        )
        for entry in dependency_entries
    )

    membership_mismatch = any(
        entry.episode_membership is None
        or entry.episode_membership.episode_uuid != trace_range.episode_uuid
        or entry.episode_membership.lifecycle_owner_id != trace_range.lifecycle_owner_id
        for entry in range_entries
    )
    bind_matches = any(
        entry.fact.fact_id == trace_range.binding_event_id
        and entry.fact.event_type == "episode_membership_transition"
        and entry.fact.payload["transition"] == "bind"
        for entry in range_entries
    )
    unbind_matches = any(
        entry.fact.fact_id == trace_range.unbinding_event_id
        and entry.fact.event_type == "episode_membership_transition"
        and entry.fact.payload["transition"] == "unbind"
        for entry in range_entries
    )
    runtime_terminal_inside_range = any(
        entry.fact.event_type == "runtime_trace_terminal" for entry in range_entries
    )

    assessed_facts = tuple(
        entry.fact
        for entry in sorted(
            (*dependency_entries, *range_entries),
            key=lambda item: item.journal_sequence,
        )
    )
    observed = frozenset(fact.event_type for fact in assessed_facts)
    required = runtime.profile.required_event_types - {"runtime_trace_terminal"}
    missing_required = required - observed
    observed_not_applicable = observed & runtime.profile.not_applicable_event_types
    lineage_issues = evaluate_control_trace_lineage(
        assessed_facts,
        profile=runtime.profile,
    )

    reasons: set[str] = set()
    if admission_gaps:
        reasons.add("episode_admission_gap")
    if durability_gaps:
        reasons.add("episode_durability_gap")
    if missing_dependencies:
        reasons.add("dependency_missing")
    if non_durable_dependencies:
        reasons.add("dependency_not_durable")
    if membership_mismatch:
        reasons.add("episode_membership_mismatch")
    if not bind_matches:
        reasons.add("episode_bind_missing")
    if not unbind_matches:
        reasons.add("episode_unbind_missing")
    if runtime_terminal_inside_range:
        reasons.add("runtime_terminal_inside_episode")
    if missing_required:
        reasons.add("required_event_missing")
    if observed_not_applicable:
        reasons.add("not_applicable_event_observed")
    if lineage_issues:
        reasons.add("lineage_incomplete")

    range_record_payload = tuple(
        {
            "journal_sequence": sequence,
            "record_integrity_hash": records[sequence].record_integrity_hash,
        }
        for sequence in sorted(expected & set(records))
    )
    dependency_payload = tuple(item.to_payload() for item in dependency_refs)
    sealed = not reasons
    capture_eligible = bool(
        sealed
        and runtime.profile.profile_scope == "formal_capture"
        and runtime.profile.implementation_status == "runtime_integrated"
    )
    return EpisodeTraceSealAssessment(
        assessment_id=assessment_id,
        trace_range=trace_range,
        profile_id=runtime.profile.profile_id,
        profile_scope=runtime.profile.profile_scope,
        assessed_at=assessed_at,
        observed_event_types=tuple(observed),
        missing_required_event_types=tuple(missing_required),
        observed_not_applicable_event_types=tuple(observed_not_applicable),
        admission_gap_ranges=_gap_ranges(admission_gaps),
        durability_gap_ranges=_gap_ranges(durability_gaps),
        lineage_issue_codes=tuple(issue.code for issue in lineage_issues),
        range_record_hash=canonical_content_hash(
            range_record_payload,
            domain=EPISODE_RANGE_RECORD_HASH_DOMAIN_V1,
        ),
        dependency_refs=dependency_refs,
        dependency_bundle_hash=canonical_content_hash(
            dependency_payload,
            domain=EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1,
        ),
        trace_sealed=sealed,
        capture_commit_eligible=capture_eligible,
        reason_codes=tuple(reasons),
    )


__all__ = [
    "EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1",
    "EPISODE_RANGE_RECORD_HASH_DOMAIN_V1",
    "EpisodeTraceBindingReceipt",
    "EpisodeTraceLegacyFinalizeReport",
    "EpisodeTraceReleaseReceipt",
    "EpisodeTraceRecorderSink",
    "NativeStreamEpisodeCapture",
    "PendingEpisodeTraceRelease",
    "PreparedEpisodeTraceCapture",
    "ReleasedEpisodeTraceCapture",
    "SubordinateEpisodeTraceLifecycle",
    "assess_episode_trace_seal",
]
