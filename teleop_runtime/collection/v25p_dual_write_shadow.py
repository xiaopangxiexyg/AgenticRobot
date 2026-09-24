"""Subordinate native v2.5-p dual-write verification lifecycle.

The application recorder remains the sole episode owner.  This port receives
the owner's identity and an already accepted source sample plus the legacy row
planned from that same sample.  It never starts, stops, or finalizes the legacy
writer and contains no v2.4 commit behavior.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from collection_runtime.component_capture import (
    ComponentCaptureCommitReport,
    ComponentCaptureEpisodeStore,
    ComponentCaptureSealReport,
    ProducedComponentSample,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v25p import v25p_base_episode_manifest_hash
from dataset_contracts.v25p_projection import V25pCompatibilityProjectionReceipt
from dataset_interface.v25p_dual_write import (
    V25pDualWriteEquivalenceReport,
    compare_projected_v23_row,
)
from dataset_interface.v25p_projection_plan import (
    V25pV23ProjectionPlan,
    plan_v23_compatibility_projection,
    verify_complete_v23_projection_receipt,
)
from dataset_interface.v25p_v23_adapter import V25pV23ProjectedRow

from .v25p_capture_profiles import V25pCaptureProfileBundle

_LEGACY_ROW_JOB_KIND = "lerobot_v21.row"


class V25pSampleProducer(Protocol):
    def produce(self, sample: Any) -> ProducedComponentSample: ...


@dataclass(frozen=True)
class V25pShadowAppendReport:
    sample_index: int
    equivalent: bool
    mismatch_fields: tuple[str, ...]


@dataclass(frozen=True)
class V25pShadowFinalizeReport:
    seal: ComponentCaptureSealReport
    receipt: V25pCompatibilityProjectionReceipt
    commit: ComponentCaptureCommitReport
    equivalent_sample_count: int
    mismatched_sample_count: int


def legacy_row_from_planned_jobs(jobs: object) -> Mapping[str, Any]:
    """Return the one row planned by the authoritative compatibility writer."""

    matches = tuple(
        job
        for job in tuple(jobs)  # type: ignore[arg-type]
        if getattr(job, "kind", None) == _LEGACY_ROW_JOB_KIND
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one legacy row job, found {len(matches)}."
        )
    payload = getattr(matches[0], "payload", None)
    row = payload.get("row") if isinstance(payload, Mapping) else None
    if not isinstance(row, Mapping):
        raise ValueError("Legacy row job does not contain a row mapping.")
    return row


def load_finalized_legacy_episode_record(
    dataset_root: str | Path,
    episode_index: int,
) -> Mapping[str, Any]:
    """Load exactly one finalized legacy episode record by owner index."""

    path = Path(dataset_root).expanduser().resolve() / "meta" / "episodes.jsonl"
    matches: list[Mapping[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping) and int(payload.get("episode_index", -1)) == int(
                episode_index
            ):
                matches.append(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load finalized legacy episode record: {exc}") from exc
    if len(matches) != 1:
        raise ValueError(
            "Finalized legacy episode index must resolve exactly once: "
            f"episode_index={int(episode_index)} matches={len(matches)}."
        )
    return matches[0]


def _close_failed_shadow(
    shadow: V25pDualWriteShadow,
    identity: EpisodeCaptureIdentity,
    error: str,
    record_error: Callable[[str], None] | None,
) -> str:
    try:
        shadow.close_unsealed(identity)
    except BaseException as exc:  # noqa: BLE001 - Preserve the subordinate failure.
        error += f";close_unsealed:{type(exc).__name__}: {exc}"
        if record_error is not None:
            record_error(error)
    return error


def append_shadow_from_legacy_jobs(
    shadow: V25pDualWriteShadow,
    identity: EpisodeCaptureIdentity,
    *,
    source_sample: Any,
    jobs: object,
    record_error: Callable[[str], None] | None = None,
) -> str:
    """Observe planned jobs, publishing an error before potentially slow cleanup."""

    try:
        shadow.append(
            identity,
            source_sample=source_sample,
            legacy_row=legacy_row_from_planned_jobs(jobs),
        )
    except BaseException as exc:  # noqa: BLE001 - Legacy capture stays authoritative.
        error = f"append:{type(exc).__name__}: {exc}"
        if record_error is not None:
            record_error(error)
        return _close_failed_shadow(shadow, identity, error, record_error)
    return ""


def finalize_shadow_after_legacy(
    shadow: V25pDualWriteShadow,
    identity: EpisodeCaptureIdentity,
    *,
    dataset_root: str | Path,
    episode_index: int,
    error_prefix: str = "finalize",
    incomplete_message: str = "v2.5-p compatibility projection failed",
    record_finalize_elapsed: Callable[[float], None] | None = None,
    record_report: Callable[[V25pShadowFinalizeReport], None] | None = None,
    record_error: Callable[[str], None] | None = None,
) -> tuple[V25pShadowFinalizeReport | None, str | None]:
    """Seal the subordinate shadow after its owner finalized the legacy row.

    The caller retains admission and publication ownership. Report projection
    and cleanup failures separately from the already committed legacy result;
    never close a different active identity. Timing excludes failure cleanup.
    A complete receipt returns no error update, preserving prior owner errors.
    Owners can publish a returned report before its receipt is inspected, and
    operational errors before cleanup. A failed operation never clears a report
    the owner already holds; only a new return value triggers record_report.
    """

    report = None
    try:
        started_at = time.perf_counter() if record_finalize_elapsed is not None else 0.0
        try:
            candidate_report = shadow.finalize_after_legacy(
                identity,
                legacy_episode_record=load_finalized_legacy_episode_record(
                    dataset_root, episode_index
                ),
                completed_at=time.time(),
                completed_clock_domain_id="local_wall_clock",
            )
        finally:
            if record_finalize_elapsed is not None:
                record_finalize_elapsed(time.perf_counter() - started_at)
        report = candidate_report
        if record_report is not None:
            record_report(report)
        error = (
            str(report.receipt.message or incomplete_message)
            if report.receipt.status != "complete"
            else None
        )
        if error is not None and record_error is not None:
            record_error(error)
    except BaseException as exc:  # noqa: BLE001 - Legacy finalization already succeeded.
        error = f"{error_prefix}:{type(exc).__name__}: {exc}"
        if record_error is not None:
            record_error(error)
        if shadow.active_identity == identity:
            error = _close_failed_shadow(shadow, identity, error, record_error)
    return report, error


def _project_produced_sample(
    *,
    produced: ProducedComponentSample,
    bundle: V25pCaptureProfileBundle,
    plan: V25pV23ProjectionPlan,
) -> V25pV23ProjectedRow:
    fields: dict[str, tuple[bool | int | float, ...]] = {}
    dtypes: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    stream_by_id = bundle.streams.stream_by_id
    for field, stream_id in plan.stream_id_by_dataset_field.items():
        value = produced.values_by_stream.get(stream_id)
        if value is None:
            raise ValueError(
                f"Produced sample lacks projected native stream {stream_id!r}."
            )
        if not value.valid or value.payload_reference is not None:
            raise ValueError(
                f"Projected native stream {stream_id!r} must contain a valid inline value."
            )
        stream = stream_by_id[stream_id]
        fields[field] = stream.normalize_values(value.values)
        dtypes[field] = stream.dtype
        shapes[field] = stream.shape
    return V25pV23ProjectedRow(
        sample_index=produced.sample_index,
        cut_at=produced.cut_at,
        cut_clock_domain_id=produced.cut_clock_domain_id,
        fields=fields,
        dtype_by_field=dtypes,
        shape_by_field=shapes,
    )


class V25pDualWriteShadow:
    """One active, subordinate native episode and its parity evidence."""

    def __init__(
        self,
        *,
        store: ComponentCaptureEpisodeStore,
        bundle: V25pCaptureProfileBundle,
        producer: V25pSampleProducer,
    ) -> None:
        if store.embodiment_profile != bundle.embodiment:
            raise ValueError("Shadow store and capture bundle embodiment differ.")
        if store.component_stream_profile != bundle.streams:
            raise ValueError("Shadow store and capture bundle streams differ.")
        if bundle.v23_projection.semantic_content_hash not in {
            item.semantic_content_hash
            for item in store.compatibility_projection_profiles
        }:
            raise ValueError("Shadow store does not bind the bundle v2.3 projection.")
        self.store = store
        self.bundle = bundle
        self.producer = producer
        self.plan = plan_v23_compatibility_projection(
            stream_profile=bundle.streams,
            projection_profile=bundle.v23_projection,
        )
        self._identity: EpisodeCaptureIdentity | None = None
        self._reports: list[V25pDualWriteEquivalenceReport] = []

    @property
    def active_identity(self) -> EpisodeCaptureIdentity | None:
        return self._identity

    def prepare_episode(self, identity: EpisodeCaptureIdentity) -> None:
        if self._identity is not None:
            raise RuntimeError("A v2.5-p dual-write shadow episode is already active.")
        self.store.prepare_episode(identity)
        self._identity = identity
        self._reports.clear()

    def bind_prepared_episode(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        bound_at: float,
        bound_clock_domain_id: str,
    ) -> None:
        self._require_identity(identity)
        self.store.bind_prepared_episode(
            identity,
            bound_at=bound_at,
            bound_clock_domain_id=bound_clock_domain_id,
        )

    def append(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        source_sample: Any,
        legacy_row: Mapping[str, Any],
    ) -> V25pShadowAppendReport:
        self._require_identity(identity)
        produced = self.producer.produce(source_sample)
        projected = _project_produced_sample(
            produced=produced,
            bundle=self.bundle,
            plan=self.plan,
        )
        equivalence = compare_projected_v23_row(projected, legacy_row)
        self.store.append_sample(
            identity,
            sample_index=produced.sample_index,
            cut_at=produced.cut_at,
            cut_clock_domain_id=produced.cut_clock_domain_id,
            values_by_stream=produced.values_by_stream,
        )
        self._reports.append(equivalence)
        mismatches = tuple(
            field
            for field, result in equivalence.fields.items()
            if not result.equivalent
        )
        return V25pShadowAppendReport(
            sample_index=produced.sample_index,
            equivalent=equivalence.equivalent,
            mismatch_fields=mismatches,
        )

    def finalize_after_legacy(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        legacy_episode_record: Mapping[str, Any],
        completed_at: float,
        completed_clock_domain_id: str,
    ) -> V25pShadowFinalizeReport:
        self._require_identity(identity)
        seal = self.store.seal_native_episode(
            identity,
            sealed_at=completed_at,
            sealed_clock_domain_id=completed_clock_domain_id,
        )
        mismatch_reports = tuple(item for item in self._reports if not item.equivalent)
        complete = not mismatch_reports and len(self._reports) == seal.sample_count
        if complete:
            try:
                record_index = int(legacy_episode_record.get("episode_index", -1))
            except (TypeError, ValueError, OverflowError):
                record_index = -1
            complete = record_index == identity.legacy_episode_index
        common = {
            "projection_id": "v25p_native_to_v23_dual_write_shadow",
            "episode_uuid": identity.episode_uuid,
            "legacy_episode_index": identity.legacy_episode_index,
            "episode_identity_hash": identity.semantic_content_hash,
            "native_capture_manifest_hash": seal.native_capture_manifest_hash,
            "projection_profile_hash": self.bundle.v23_projection.semantic_content_hash,
            "target_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            "completed_at": completed_at,
            "completed_clock_domain_id": completed_clock_domain_id,
        }
        if complete:
            receipt = V25pCompatibilityProjectionReceipt(
                **common,
                status="complete",
                target_episode_record_hash=v25p_base_episode_manifest_hash(
                    legacy_episode_record
                ),
                target_episode_locator=(
                    "meta/episodes.jsonl#episode_index="
                    f"{identity.legacy_episode_index}"
                ),
                lossy=self.plan.lossy,
                omitted_stream_ids=self.plan.omitted_stream_ids,
            )
            try:
                verify_complete_v23_projection_receipt(
                    dataset_root=self.store.dataset_root,
                    identity=identity,
                    receipt=receipt,
                )
            except ValueError as exc:
                complete = False
                receipt = self._failed_receipt(
                    common,
                    message=f"legacy episode verification failed: {exc}",
                )
        else:
            mismatch_fields = sorted(
                {
                    field
                    for report in mismatch_reports
                    for field, result in report.fields.items()
                    if not result.equivalent
                }
            )
            receipt = self._failed_receipt(
                common,
                message=(
                    "dual-write parity failed: "
                    f"samples={len(mismatch_reports)}/{seal.sample_count} "
                    f"fields={mismatch_fields}"
                ),
            )
        commit = self.store.commit_episode(
            identity,
            committed_at=completed_at,
            committed_clock_domain_id=completed_clock_domain_id,
            compatibility_projection_receipts=(receipt,),
        )
        equivalent_count = sum(item.equivalent for item in self._reports)
        result = V25pShadowFinalizeReport(
            seal=seal,
            receipt=receipt,
            commit=commit,
            equivalent_sample_count=equivalent_count,
            mismatched_sample_count=len(self._reports) - equivalent_count,
        )
        self._identity = None
        self._reports.clear()
        return result

    def discard_episode(self, identity: EpisodeCaptureIdentity) -> None:
        self._require_identity(identity)
        self.store.discard_episode(identity)
        self._identity = None
        self._reports.clear()

    def close_unsealed(self, identity: EpisodeCaptureIdentity) -> None:
        self._require_identity(identity)
        self.store.close_unsealed(identity)
        self._identity = None
        self._reports.clear()

    @staticmethod
    def _failed_receipt(
        common: Mapping[str, Any],
        *,
        message: str,
    ) -> V25pCompatibilityProjectionReceipt:
        return V25pCompatibilityProjectionReceipt(
            **dict(common),
            status="failed",
            message=message,
        )

    def _require_identity(self, identity: EpisodeCaptureIdentity) -> None:
        if self._identity != identity:
            raise ValueError("Episode identity is not active in the v2.5-p shadow.")
__all__ = [
    "V25pDualWriteShadow",
    "V25pSampleProducer",
    "V25pShadowAppendReport",
    "V25pShadowFinalizeReport",
    "append_shadow_from_legacy_jobs",
    "finalize_shadow_after_legacy",
    "legacy_row_from_planned_jobs",
    "load_finalized_legacy_episode_record",
]
