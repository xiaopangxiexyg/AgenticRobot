"""Typed, single-owner interactive control for the Sim-HIL application.

The browser submits application intents only.  The caller that owns the
control loop must call :meth:`SimHILAcceptanceController.process_next`; HTTP
threads never start devices, change composition parameters, seal capture, or
step the simulator directly.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


SIM_HIL_ACCEPTANCE_STATUS_SCHEMA_V1 = (
    "dexteleop.control_station.sim_hil_acceptance_status.v1"
)
SIM_HIL_ACCEPTANCE_INTENT_SCHEMA_V1 = (
    "dexteleop.control_station.sim_hil_acceptance_intent.v1"
)
SIM_HIL_ACCEPTANCE_INTENT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.sim_hil_acceptance_intent_identity.v1"
)
SIM_HIL_ACCEPTANCE_STAGES_V1 = (
    "human_teleop",
    "ppo_only",
    "proportional_blend",
)
SIM_HIL_CONTROL_MODES_V1 = SIM_HIL_ACCEPTANCE_STAGES_V1
SIM_HIL_BLEND_FAILURE_FALLBACKS_V1 = (
    "hold",
    "human",
    "policy",
)
SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1 = (
    "hold",
    "policy",
)
SIM_HIL_ALLOCATION_MODES_DEVELOPMENT = (
    "automatic",
    "base_only",
    "relative_overlay",
    "absolute_overlay_owner",
)
SIM_HIL_ACCEPTANCE_TRACE_RUNTIME_CAPACITY_V1 = 524_288
STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.station_control_trace_terminal_receipt.v1"
)
STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.station_control_trace_finalization.v1"
)
SIM_HIL_ACCEPTANCE_INTENT_KINDS_V1 = (
    "start_runtime",
    "start_allocation_runtime",
    "set_control_mode",
    "set_allocation_mode",
    "request_residual_rebase_and_clear",
    "set_blend_failure_fallback",
    "set_human_timeout_fallback",
    "start_stage",
    "confirm_stage",
    "begin_vive_calibration",
    "capture_vive_calibration",
    "reset_vive_calibration_stage",
    "cancel_vive_calibration",
    "save_vive_calibration_profile",
    "cancel_operation",
    "swap_vive_sources",
    "hold",
    "rearm",
    "capture_runtime_home",
    "rebuild_runtime",
    "finalize_capture",
    "stop_runtime",
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class SimHILAcceptanceIntent:
    """One immutable operator request without raw action payloads."""

    client_id: str
    client_sequence: int
    expected_revision: int
    intent_kind: str
    stage_id: str | None = None
    control_mode: str | None = None
    allocation_mode: str | None = None
    human_alpha: float | None = None
    fallback_source: str | None = None
    blend_failure_grace_sec: float | None = None
    profile_name: str | None = None
    reason: str = "operator_request"

    def __post_init__(self) -> None:
        client_id = _required(self.client_id, name="client_id")
        client_sequence = _nonnegative_integer(
            self.client_sequence,
            name="client_sequence",
        )
        expected_revision = _nonnegative_integer(
            self.expected_revision,
            name="expected_revision",
        )
        intent_kind = _required(self.intent_kind, name="intent_kind")
        if intent_kind not in SIM_HIL_ACCEPTANCE_INTENT_KINDS_V1:
            raise ValueError(f"Unsupported acceptance intent: {intent_kind}.")
        stage_id = _optional(self.stage_id)
        control_mode = _optional(self.control_mode)
        allocation_mode = _optional(self.allocation_mode)
        if intent_kind in {"start_stage", "confirm_stage"}:
            if stage_id not in SIM_HIL_ACCEPTANCE_STAGES_V1:
                raise ValueError(
                    "start_stage/confirm_stage requires a known stage_id."
                )
        elif stage_id is not None:
            raise ValueError(
                f"{intent_kind} must not carry stage_id."
            )
        if intent_kind in {"start_runtime", "set_control_mode"}:
            if control_mode not in SIM_HIL_CONTROL_MODES_V1:
                raise ValueError(
                    "start_runtime/set_control_mode requires a known "
                    "control_mode."
                )
        elif control_mode is not None:
            raise ValueError(
                f"{intent_kind} must not carry control_mode."
            )
        if intent_kind == "set_allocation_mode":
            if allocation_mode not in SIM_HIL_ALLOCATION_MODES_DEVELOPMENT:
                raise ValueError(
                    "set_allocation_mode requires automatic, base_only, "
                    "relative_overlay, or absolute_overlay_owner."
                )
        elif allocation_mode is not None:
            raise ValueError(
                f"{intent_kind} must not carry allocation_mode."
            )
        alpha = self.human_alpha
        if alpha is not None:
            alpha = float(alpha)
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError("human_alpha must be finite and in [0, 1].")
        blend_requested = bool(
            (
                intent_kind == "start_stage"
                and stage_id == "proportional_blend"
            )
            or (
                intent_kind in {"start_runtime", "set_control_mode"}
                and control_mode == "proportional_blend"
            )
        )
        if blend_requested:
            if alpha is None or alpha <= 0.0 or alpha >= 1.0:
                raise ValueError(
                    "proportional_blend requires human_alpha strictly "
                    "between 0 and 1."
                )
        elif alpha is not None:
            raise ValueError(
                "human_alpha is only valid for proportional_blend."
            )
        fallback_source = _optional(self.fallback_source)
        if intent_kind == "set_blend_failure_fallback":
            if fallback_source not in SIM_HIL_BLEND_FAILURE_FALLBACKS_V1:
                raise ValueError(
                    "set_blend_failure_fallback requires hold, human, or "
                    "policy."
                )
        elif intent_kind == "set_human_timeout_fallback":
            if fallback_source not in SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1:
                raise ValueError(
                    "set_human_timeout_fallback requires hold or policy."
                )
        elif fallback_source is not None:
            raise ValueError(
                "fallback_source is only valid for "
                "a fallback-selection intent."
            )
        blend_failure_grace = self.blend_failure_grace_sec
        if blend_failure_grace is not None:
            blend_failure_grace = float(blend_failure_grace)
            if (
                not math.isfinite(blend_failure_grace)
                or blend_failure_grace < 0.0
            ):
                raise ValueError(
                    "blend_failure_grace_sec must be finite and non-negative."
                )
        if (
            intent_kind != "set_blend_failure_fallback"
            and blend_failure_grace is not None
        ):
            raise ValueError(
                "blend_failure_grace_sec is only valid for "
                "set_blend_failure_fallback."
            )
        profile_name = _optional(self.profile_name)
        if intent_kind == "save_vive_calibration_profile":
            pass
        elif profile_name is not None:
            raise ValueError(
                "profile_name is only valid for save_vive_calibration_profile."
            )
        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "client_sequence", client_sequence)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "intent_kind", intent_kind)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "control_mode", control_mode)
        object.__setattr__(self, "allocation_mode", allocation_mode)
        object.__setattr__(self, "human_alpha", alpha)
        object.__setattr__(self, "fallback_source", fallback_source)
        object.__setattr__(
            self,
            "blend_failure_grace_sec",
            blend_failure_grace,
        )
        object.__setattr__(self, "profile_name", profile_name)
        object.__setattr__(
            self,
            "reason",
            _required(self.reason, name="reason"),
        )

    @property
    def intent_id(self) -> str:
        payload = {
            "schema": SIM_HIL_ACCEPTANCE_INTENT_SCHEMA_V1,
            "client_id": self.client_id,
            "client_sequence": self.client_sequence,
            "expected_revision": self.expected_revision,
            "intent_kind": self.intent_kind,
            "stage_id": self.stage_id,
            "control_mode": self.control_mode,
            "human_alpha": self.human_alpha,
            "fallback_source": self.fallback_source,
            "profile_name": self.profile_name,
            "reason": self.reason,
        }
        # Keep existing fixed-alpha intent identities byte-for-byte stable.
        # The development allocation field enters the identity only when used.
        if self.allocation_mode is not None:
            payload["allocation_mode"] = self.allocation_mode
        if self.blend_failure_grace_sec is not None:
            payload["blend_failure_grace_sec"] = (
                self.blend_failure_grace_sec
            )
        return "sim-hil-acceptance-intent:" + canonical_content_hash(
            payload,
            domain=SIM_HIL_ACCEPTANCE_INTENT_HASH_DOMAIN_V1,
        )


def sim_hil_acceptance_intent_from_json(
    payload: Mapping[str, Any],
) -> SimHILAcceptanceIntent:
    """Validate an exact browser payload before constructing an intent."""

    if not isinstance(payload, Mapping):
        raise TypeError("Acceptance intent payload must be an object.")
    allowed = {
        "client_id",
        "client_sequence",
        "expected_revision",
        "intent_kind",
        "stage_id",
        "control_mode",
        "allocation_mode",
        "human_alpha",
        "fallback_source",
        "blend_failure_grace_sec",
        "profile_name",
        "reason",
    }
    required = {
        "client_id",
        "client_sequence",
        "expected_revision",
        "intent_kind",
    }
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError(f"Unknown acceptance intent fields: {unknown}.")
    if missing:
        raise ValueError(f"Missing acceptance intent fields: {missing}.")
    return SimHILAcceptanceIntent(
        client_id=payload["client_id"],
        client_sequence=payload["client_sequence"],
        expected_revision=payload["expected_revision"],
        intent_kind=payload["intent_kind"],
        stage_id=payload.get("stage_id"),
        control_mode=payload.get("control_mode"),
        allocation_mode=payload.get("allocation_mode"),
        human_alpha=payload.get("human_alpha"),
        fallback_source=payload.get("fallback_source"),
        blend_failure_grace_sec=payload.get("blend_failure_grace_sec"),
        profile_name=payload.get("profile_name"),
        reason=payload.get("reason", "operator_request"),
    )


@dataclass(frozen=True)
class SimHILAcceptanceSubmissionReceipt:
    intent_id: str
    accepted: bool
    revision: int
    rejection_code: str | None = None
    message: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "accepted": self.accepted,
            "revision": self.revision,
            "rejection_code": self.rejection_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class StationControlTraceFinalizationReceipt:
    """Durable lower-owned references after episode commit and runtime close.

    The runtime terminal must be admitted after the formal episode is unbound:
    the episode seal intentionally excludes ``runtime_trace_terminal`` from its
    range.  This receipt therefore joins, but never fabricates, the committed
    episode metadata and the later durable terminal record.
    """

    trace_session_id: str
    runtime_session_id: str
    control_domain_id: str
    episode_uuid: str
    terminal_fact_id: str
    terminal_fact_hash: str
    terminal_journal_sequence: int
    terminal_record_integrity_hash: str
    trace_seal_hash: str
    trace_commit_hash: str
    trace_commit_relative_path: str
    trace_terminal_receipt_ref: str = ""
    trace_commit_receipt_ref: str = ""
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "trace_session_id",
            "runtime_session_id",
            "control_domain_id",
            "episode_uuid",
            "terminal_fact_id",
            "terminal_fact_hash",
            "terminal_record_integrity_hash",
            "trace_seal_hash",
            "trace_commit_hash",
            "trace_commit_relative_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), name=field_name),
            )
        sequence = int(self.terminal_journal_sequence)
        if (
            isinstance(self.terminal_journal_sequence, bool)
            or sequence <= 0
            or sequence != self.terminal_journal_sequence
        ):
            raise ValueError("terminal_journal_sequence must be a positive integer.")
        object.__setattr__(self, "terminal_journal_sequence", sequence)
        terminal_payload = {
            "trace_session_id": self.trace_session_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "terminal_fact_id": self.terminal_fact_id,
            "terminal_fact_hash": self.terminal_fact_hash,
            "terminal_journal_sequence": sequence,
            "terminal_record_integrity_hash": (
                self.terminal_record_integrity_hash
            ),
        }
        terminal_ref = (
            "control-trace-terminal-durable:"
            + canonical_content_hash(
                terminal_payload,
                domain=(
                    STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1
                ),
            )
        )
        commit_ref = f"control-trace-capture-commit:{self.trace_commit_hash}"
        payload = {
            **terminal_payload,
            "episode_uuid": self.episode_uuid,
            "trace_seal_hash": self.trace_seal_hash,
            "trace_commit_hash": self.trace_commit_hash,
            "trace_commit_relative_path": self.trace_commit_relative_path,
            "trace_terminal_receipt_ref": terminal_ref,
            "trace_commit_receipt_ref": commit_ref,
        }
        object.__setattr__(self, "trace_terminal_receipt_ref", terminal_ref)
        object.__setattr__(self, "trace_commit_receipt_ref", commit_ref)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1,
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "trace_session_id": self.trace_session_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "episode_uuid": self.episode_uuid,
            "terminal_fact_id": self.terminal_fact_id,
            "terminal_fact_hash": self.terminal_fact_hash,
            "terminal_journal_sequence": self.terminal_journal_sequence,
            "terminal_record_integrity_hash": (
                self.terminal_record_integrity_hash
            ),
            "trace_seal_hash": self.trace_seal_hash,
            "trace_commit_hash": self.trace_commit_hash,
            "trace_commit_relative_path": self.trace_commit_relative_path,
            "trace_terminal_receipt_ref": self.trace_terminal_receipt_ref,
            "trace_commit_receipt_ref": self.trace_commit_receipt_ref,
            "semantic_content_hash": self.semantic_content_hash,
        }


class StationControlTraceEpisodeSessionCapture:
    """Concrete binding to one lower-owned durable ControlTrace session."""

    def __init__(
        self,
        root: str | Path,
        *,
        trace_session_id: str | None = None,
        max_spool_bytes: int = 64 << 20,
        runtime_capacity: int = (
            SIM_HIL_ACCEPTANCE_TRACE_RUNTIME_CAPACITY_V1
        ),
        runtime_session_id: str = "control-station-sim-hil-ppo-session",
        control_domain_id: str = "mujoco-sim-hil",
        trace_requirement_profile_id: str = (
            "deployable_sim_immediate_component_policy_human_composer_v1"
        ),
        clock_domain_id: str = "mujoco.sim_hil.monotonic",
        capture_purpose: str = "sim_hil_acceptance",
        legacy_episode_index: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._max_spool_bytes = int(max_spool_bytes)
        self._runtime_capacity = int(runtime_capacity)
        if self._runtime_capacity <= 0:
            raise ValueError("runtime_capacity must be positive.")
        self._runtime_session_id = _required(
            runtime_session_id,
            name="runtime_session_id",
        )
        self._control_domain_id = _required(
            control_domain_id,
            name="control_domain_id",
        )
        self._trace_requirement_profile_id = _required(
            trace_requirement_profile_id,
            name="trace_requirement_profile_id",
        )
        self._clock_domain_id = _required(
            clock_domain_id,
            name="clock_domain_id",
        )
        self._capture_purpose = _required(
            capture_purpose,
            name="capture_purpose",
        )
        episode_index = int(legacy_episode_index)
        if (
            isinstance(legacy_episode_index, bool)
            or episode_index < 0
            or episode_index != legacy_episode_index
        ):
            raise ValueError("legacy_episode_index must be a non-negative integer.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._legacy_episode_index = episode_index
        self._clock = clock
        self._lock = threading.RLock()
        self._open_session(trace_session_id=trace_session_id)

    def _open_session(self, *, trace_session_id: str | None) -> None:
        from dataset_contracts.completeness import (
            TRACE_REQUIREMENT_PROFILES_V1,
        )
        from dataset_contracts.control_trace_episode import (
            CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
        )
        from teleop_runtime.control_trace import (
            ControlTraceRuntime,
            ControlTraceSpool,
            ControlTraceSpoolSink,
            ControlTraceSpoolWorker,
            ControlTraceWriterHealthLedger,
            EpisodeCaptureManifestStore,
            EpisodeTraceRecorderSink,
            RuntimeTraceSession,
        )

        session_id = (
            str(trace_session_id).strip()
            if trace_session_id is not None
            else self._capture_purpose.replace("_", "-") + "-" + uuid.uuid4().hex
        )
        if not session_id:
            raise ValueError("trace_session_id must be non-empty.")
        try:
            profile = TRACE_REQUIREMENT_PROFILES_V1[
                self._trace_requirement_profile_id
            ]
        except KeyError as error:
            raise ValueError(
                "Unknown ControlTrace requirement profile: "
                f"{self._trace_requirement_profile_id!r}."
            ) from error
        self.trace_runtime = ControlTraceRuntime(
            RuntimeTraceSession(
                trace_session_id=session_id,
                runtime_session_id=self._runtime_session_id,
                control_domain_id=self._control_domain_id,
                profile_id=profile.profile_id,
                started_at=float(self._clock()),
                clock_domain_id=self._clock_domain_id,
            ),
            profile=profile,
            capacity=self._runtime_capacity,
        )
        self.session_root = (
            self.root
            / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
            / session_id
        )
        self.session_root.mkdir(parents=True, exist_ok=False)
        self.spool = ControlTraceSpool(
            self.session_root / "control_trace.wal",
            trace_session_id=session_id,
            max_bytes=self._max_spool_bytes,
            physical_preallocation=False,
            allow_growth=True,
            growth_bytes=self._max_spool_bytes,
        )
        self.health = ControlTraceWriterHealthLedger(
            self.session_root / "writer_health.ledger",
            trace_session_id=session_id,
        )
        self.durable_sink = ControlTraceSpoolSink(
            sink_id=f"{session_id}:spool",
            runtime=self.trace_runtime,
            spool=self.spool,
            health_ledger=self.health,
        )
        self.spool_worker = ControlTraceSpoolWorker(
            self.durable_sink,
            poll_interval_sec=0.01,
            thread_name=f"station-trace-{session_id}",
        )
        self.spool_worker.start()
        self._dependency_sequences: tuple[int, ...] = ()
        self.recorder = EpisodeTraceRecorderSink(
            self.trace_runtime,
            lifecycle_owner_id=f"{session_id}:episode-recorder",
            flush_to_durable=self.spool_worker.flush_once,
            dependency_sequences=(
                lambda _runtime: self._dependency_sequences
            ),
            manifest_store=EpisodeCaptureManifestStore(self.root),
        )
        self._prepared: Any | None = None
        self._final_report: Any | None = None
        self._runtime_finalization: (
            StationControlTraceFinalizationReceipt | None
        ) = None
        self._admission_failure: str | None = None
        self._closed = False

    def rotate_runtime_session(self, *, now: float) -> dict[str, object]:
        """Start a fresh trace identity after one whole-runtime teardown."""

        with self._lock:
            if self._prepared is not None:
                self.abort(now=float(now))
            self.spool_worker.notify()
            self.spool_worker.stop(timeout_sec=5.0)
            self.health.close()
            self.spool.close()
            self._open_session(trace_session_id=None)
            return self.status()

    def admit_trace(self, fact: Any) -> None:
        receipt = self.trace_runtime.issue(fact)
        if not receipt.accepted:
            failure = (
                "Station ControlTrace admission failed: "
                f"{receipt.rejection_code}:{receipt.message}"
            )
            with self._lock:
                self._admission_failure = failure
            raise RuntimeError(failure)
        # Persistence is deliberately subordinate to trace admission.  Keep
        # fsync work off the Station control thread and continuously drain the
        # journal so episode release/rebuild has only a small tail to flush.
        self.spool_worker.notify()

    def prepare(self, *, now: float) -> dict[str, object]:
        with self._lock:
            if self._prepared is not None:
                return self.status()
            if self._final_report is not None:
                raise RuntimeError("ControlTrace episode was already finalized.")
            prepared = self.recorder.prepare_episode(
                legacy_episode_index=self._legacy_episode_index,
                prepared_at=float(now),
            )
            # Publish the subordinate preparation before binding so every
            # failure path can find and release it.  Previously a bind error
            # happened before ``self._prepared`` was assigned, leaving the
            # recorder's private preparation active and making the next retry
            # fail with ``episode preparation is already active``.
            self._prepared = prepared
            try:
                self.recorder.bind_prepared_episode(
                    prepared,
                    bound_at=float(now),
                )
                # Runtime/schema/source activation facts are intentionally
                # published before native viewer startup is accepted.  Freeze
                # every admitted session-level fact after the atomic bind so
                # an episode can reference that immutable bootstrap lineage
                # without pretending those facts occurred inside the episode.
                self._dependency_sequences = tuple(
                    entry.journal_sequence
                    for entry in self.trace_runtime.journal_entries()
                    if entry.episode_membership is None
                )
            except BaseException as error:  # noqa: BLE001 - preserve bind cause.
                cleanup_error: BaseException | None = None
                try:
                    lifecycle = getattr(self.recorder, "lifecycle", None)
                    active_episode = getattr(
                        lifecycle,
                        "active_episode_uuid",
                        None,
                    )
                    if active_episode is not None:
                        self.recorder.release_episode(
                            prepared,
                            unbound_at=float(now),
                            reason=f"{self._capture_purpose}_bind_failed",
                        )
                    else:
                        cancel_prepared = getattr(
                            self.recorder,
                            "cancel_prepared_episode",
                            None,
                        )
                        if not callable(cancel_prepared):
                            raise RuntimeError(
                                "ControlTrace recorder cannot cancel an "
                                "unbound preparation."
                            )
                        cancel_prepared(prepared)
                    self.spool_worker.flush_once()
                except BaseException as reset_error:  # noqa: BLE001 - retain both causes.
                    cleanup_error = reset_error
                finally:
                    self._prepared = None
                    self._dependency_sequences = ()
                if cleanup_error is not None:
                    raise RuntimeError(
                        f"{type(error).__name__}:{error}; "
                        "bind_cleanup_failed="
                        f"{type(cleanup_error).__name__}:{cleanup_error}"
                    ) from error
                raise
            return self.status()

    def finalize(self, *, now: float) -> dict[str, object]:
        with self._lock:
            prepared = self._prepared
            if prepared is None:
                raise RuntimeError("ControlTrace episode is not prepared.")
            released = self.recorder.release_episode(
                prepared,
                unbound_at=float(now),
                reason=f"{self._capture_purpose}_finalize",
            )
            finalized = self.recorder.legacy_finalize_completed(
                released,
                completed_at=float(now),
            )
            self.spool_worker.flush_once()
            self._prepared = None
            self._final_report = finalized
            if finalized.capture_commit_status != "committed":
                raise RuntimeError(
                    "ControlTrace episode did not commit: "
                    f"{finalized.message}"
                )
            return self.status()

    def complete_runtime(self) -> StationControlTraceFinalizationReceipt:
        """Join one durable commit with the later durable terminal fact.

        The owning application must already be closed so its lower runtime has
        emitted exactly one ``runtime_trace_terminal`` outside the episode.
        """

        with self._lock:
            if self._runtime_finalization is not None:
                return self._runtime_finalization
            if self._prepared is not None:
                raise RuntimeError(
                    "Runtime finalization requires the episode to be unbound."
                )
            finalized = self._final_report
            if finalized is None or finalized.capture_commit_status != "committed":
                raise RuntimeError(
                    "Runtime finalization requires a durable episode commit."
                )
            commit = finalized.capture_commit_candidate
            publish_receipt = finalized.capture_commit_publish_receipt
            if commit is None or publish_receipt is None:
                raise RuntimeError("Committed trace is missing its commit receipt.")
            if (
                not bool(publish_receipt.durable)
                or publish_receipt.semantic_content_hash
                != commit.semantic_content_hash
                or publish_receipt.episode_uuid != finalized.episode_uuid
            ):
                raise RuntimeError("ControlTrace commit publication receipt mismatch.")
            flush_report = self.spool_worker.flush_once()
            if flush_report.failure_code is not None:
                raise RuntimeError(
                    "ControlTrace terminal persistence failed: "
                    f"{flush_report.failure_code}."
                )
            terminal_entries = tuple(
                entry
                for entry in self.trace_runtime.journal_entries()
                if entry.fact.event_type == "runtime_trace_terminal"
            )
            if len(terminal_entries) != 1:
                raise RuntimeError(
                    "Runtime finalization requires exactly one terminal fact."
                )
            terminal_entry = terminal_entries[0]
            if terminal_entry.episode_membership is not None:
                raise RuntimeError(
                    "Runtime terminal fact must be outside the formal episode."
                )
            snapshot = self.trace_runtime.snapshot()
            if terminal_entry.journal_sequence not in snapshot.durable_sequences:
                raise RuntimeError("Runtime terminal fact is not durable.")
            records = {
                record.journal_sequence: record
                for record in self.trace_runtime.records()
            }
            terminal_record = records.get(terminal_entry.journal_sequence)
            if terminal_record is None:
                raise RuntimeError("Runtime terminal record is missing.")
            receipt = StationControlTraceFinalizationReceipt(
                trace_session_id=self.trace_runtime.session.trace_session_id,
                runtime_session_id=self.trace_runtime.session.runtime_session_id,
                control_domain_id=self.trace_runtime.session.control_domain_id,
                episode_uuid=finalized.episode_uuid,
                terminal_fact_id=terminal_entry.fact.fact_id,
                terminal_fact_hash=terminal_entry.fact.semantic_content_hash,
                terminal_journal_sequence=terminal_entry.journal_sequence,
                terminal_record_integrity_hash=(
                    terminal_record.record_integrity_hash
                ),
                trace_seal_hash=commit.trace_seal_hash,
                trace_commit_hash=commit.semantic_content_hash,
                trace_commit_relative_path=publish_receipt.relative_path,
            )
            self._runtime_finalization = receipt
            return receipt

    def abort(self, *, now: float) -> None:
        with self._lock:
            prepared = self._prepared
            if prepared is None:
                # Recover a lower-owned preparation that was created before a
                # bind failure could publish this wrapper's receipt.  This is
                # intentionally read-only discovery; release/cancel remains
                # owned by EpisodeTraceRecorderSink.
                prepared = getattr(self.recorder, "prepared_episode", None)
            if prepared is None:
                return
            try:
                self.recorder.release_episode(
                    prepared,
                    unbound_at=float(now),
                    reason=f"{self._capture_purpose}_aborted",
                )
            except RuntimeError as error:
                # A bind can fail after ``prepare_episode`` but before the
                # lifecycle records an active membership.  Treat that state
                # as an unbound preparation and cancel it transactionally.
                if "No ControlTrace episode is currently bound" not in str(error):
                    raise
                cancel_prepared = getattr(
                    self.recorder,
                    "cancel_prepared_episode",
                    None,
                )
                if not callable(cancel_prepared):
                    raise
                cancel_prepared(prepared)
            finally:
                self.spool_worker.flush_once()
                self._prepared = None
                self._dependency_sequences = ()

    def status(self) -> dict[str, object]:
        with self._lock:
            prepared = self._prepared
            finalized = self._final_report
            writer = self.spool_worker.snapshot()
            writer_failure = (
                self._admission_failure
                or writer.fatal_error_type
                or (
                    None
                    if writer.last_report is None
                    else writer.last_report.failure_code
                )
            )
            write_offset = self.spool.write_offset
            allocated_bytes = self.spool.max_bytes
            if writer_failure is not None:
                capture_state = "faulted"
                message = (
                    f"ControlTrace capture fault: {writer_failure}"
                )
            elif (
                finalized is not None
                and finalized.capture_commit_status == "committed"
            ):
                capture_state = "committed"
                message = finalized.message
            elif prepared is not None:
                capture_state = "recording"
                message = ""
            else:
                capture_state = "ready"
                message = "" if finalized is None else finalized.message
            return {
                "enabled": True,
                "trace_session_id": (
                    self.trace_runtime.session.trace_session_id
                ),
                "runtime_session_id": (
                    self.trace_runtime.session.runtime_session_id
                ),
                "control_domain_id": (
                    self.trace_runtime.session.control_domain_id
                ),
                "trace_requirement_profile_id": (
                    self.trace_runtime.session.profile_id
                ),
                "session_root": str(self.session_root),
                "episode_uuid": (
                    prepared.episode_uuid
                    if prepared is not None
                    else None
                    if finalized is None
                    else finalized.episode_uuid
                ),
                "status": capture_state,
                "commit_status": (
                    None
                    if finalized is None
                    else finalized.capture_commit_status
                ),
                "message": message,
                "runtime_finalization": (
                    None
                    if self._runtime_finalization is None
                    else self._runtime_finalization.to_wire()
                ),
                "dependency_count": len(self._dependency_sequences),
                "writer": {
                    "thread_alive": writer.thread_alive,
                    "flush_count": writer.flush_count,
                    "fatal_error_type": writer.fatal_error_type,
                    "last_failure_code": (
                        self._admission_failure
                        or (
                            None
                            if writer.last_report is None
                            else writer.last_report.failure_code
                        )
                    ),
                    "write_offset_bytes": write_offset,
                    "allocated_bytes": allocated_bytes,
                    "runtime_entry_count": (
                        self.trace_runtime.resident_entry_count
                    ),
                    "runtime_capacity": self._runtime_capacity,
                    "issued_sequence": self.trace_runtime.issued_sequence,
                },
            }

    def close(self, *, now: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            if self._prepared is not None:
                self.abort(
                    now=time.monotonic() if now is None else float(now)
                )
            self.spool_worker.notify()
            self.spool_worker.stop(timeout_sec=5.0)
            self.health.close()
            self.spool.close()
            self._closed = True


# Backward-compatible public name for the interactive acceptance controller.
# New Agent Skill code uses the generic Station-owned name above and supplies
# the exact fresh runtime/control-domain identity from its deployment binding.
SimHILAcceptanceTraceCapture = StationControlTraceEpisodeSessionCapture


class SimHILAcceptanceController:
    """Interactive Sim-HIL state owned by the Station control thread.

    ``start_runtime`` constructs one fixed-alpha formal session.  The distinct
    ``start_allocation_runtime`` intent constructs an allocation-aware session
    without pretending that its initial state is a proportional blend.
    Afterwards
    ``set_control_mode`` may select human-only, policy-only, or proportional
    blend in any order and any number of times.  The older staged intents
    remain temporarily accepted for migration but are not used by the formal
    GUI.
    """

    def __init__(
        self,
        application: Any | None,
        *,
        trace_capture: SimHILAcceptanceTraceCapture | None = None,
        device_check_timeout_sec: float = 10.0,
        queue_capacity: int = 8,
        application_factory: Callable[[Path, str, float], Any] | None = None,
        vive_setup_controller: Any | None = None,
        vive_setup_factory: Callable[[str], Any] | None = None,
        coordinate_profile_path: str | Path | None = None,
        allocation_runtime_enabled: bool = False,
    ) -> None:
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("queue_capacity must be positive.")
        timeout = float(device_check_timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("device_check_timeout_sec must be positive.")
        if application is None and (
            application_factory is None
            or vive_setup_controller is None
            or vive_setup_factory is None
            or coordinate_profile_path is None
        ):
            raise ValueError(
                "Lazy Sim-HIL acceptance requires application_factory, "
                "vive_setup_controller, vive_setup_factory, and "
                "coordinate_profile_path."
            )
        self.application = application
        self.application_factory = application_factory
        self._vive_setup_controller = vive_setup_controller
        self._vive_setup_factory = vive_setup_factory
        requested_coordinate_profile_path = (
            None
            if coordinate_profile_path is None
            else Path(coordinate_profile_path).expanduser()
        )
        loaded_coordinate_profile_path = (
            None
            if vive_setup_controller is None
            else getattr(vive_setup_controller, "current_profile_path", None)
        )
        # Freeze a symbolic ``latest`` selection to the exact validated file
        # already loaded by the setup controller. GUI status and later runtime
        # construction must not resolve two different files.
        self._coordinate_profile_path = (
            Path(loaded_coordinate_profile_path).expanduser()
            if loaded_coordinate_profile_path is not None
            else requested_coordinate_profile_path
        )
        self._selected_vive_source_role = "left_wrist"
        self._setup_phase = (
            "runtime_constructed"
            if application is not None
            else "vive_setup"
        )
        self._cancel_requested = threading.Event()
        self.trace_capture = trace_capture
        self.device_check_timeout_sec = timeout
        self.allocation_runtime_enabled = bool(allocation_runtime_enabled)
        self.queue_capacity = capacity
        self._queue: deque[SimHILAcceptanceIntent] = deque()
        self._revision = 0
        self._stage_status_by_id = {
            stage_id: "locked"
            for stage_id in SIM_HIL_ACCEPTANCE_STAGES_V1
        }
        self._stage_status_by_id["human_teleop"] = "available"
        self._completed: list[str] = []
        self._active_stage_id: str | None = None
        self._active_control_mode: str | None = None
        self._active_human_alpha: float | None = None
        self._control_mode_change_count = 0
        self._formal_runtime_started = False
        self._active_intent_id: str | None = None
        self._baseline_cycle_by_stage: dict[str, int] = {}
        self._requested_alpha_by_stage: dict[str, float] = {}
        self._stage_evidence_by_id: dict[str, dict[str, object]] = {}
        self._last_result: dict[str, object] | None = None
        self._last_error: str | None = None
        self._last_sequence_by_client: dict[str, int] = {}
        self._shutdown_requested = False
        self._lock = threading.RLock()

    @property
    def current_application(self) -> Any | None:
        with self._lock:
            return self.application

    def _require_application(self) -> Any:
        application = self.current_application
        if application is None:
            raise RuntimeError(
                "Sim-HIL runtime has not been constructed; complete VIVE "
                "setup first."
            )
        return application

    def _setup_snapshot(self) -> dict[str, object]:
        with self._lock:
            setup = self._vive_setup_controller
            selected_profile = self._coordinate_profile_path
            setup_phase = self._setup_phase
            selected_role = self._selected_vive_source_role
        if setup is None:
            return {
                "phase": setup_phase,
                "runtime_launch_ready": self.application is not None,
                "policy_runtime_launch_ready": self.application is not None,
                "human_runtime_launch_ready": self.application is not None,
                "selected_profile_path": (
                    None
                    if selected_profile is None
                    else str(selected_profile)
                ),
                "selected_vive_source_role": selected_role,
                "calibration": None,
            }
        calibration = dict(setup.snapshot())
        saved_profile = getattr(setup, "last_saved_profile_path", None)
        if saved_profile is not None:
            selected_profile = Path(saved_profile).expanduser()
        profile_ready = False
        if selected_profile is not None:
            if str(selected_profile) == "latest":
                profile_ready = any(
                    bool(item.get("valid", False))
                    for item in calibration.get("profiles", [])
                    if isinstance(item, Mapping)
                )
            else:
                profile_ready = selected_profile.is_file()
        input_preflight = calibration.get("input_preflight", {})
        input_ready = bool(
            isinstance(input_preflight, Mapping)
            and input_preflight.get("ready", False)
        )
        workflow_active = bool(calibration.get("workflow_active", False))
        stage = str(calibration.get("stage") or "")
        unsaved_completed = stage == "completed" and saved_profile is None
        mapping = dict(
            calibration.get("source_role_by_output_channel") or {}
        )
        selected_role = str(
            mapping.get("right_wrist_pose") or selected_role
        )
        return {
            "phase": setup_phase,
            "runtime_launch_ready": bool(
                input_ready
                and profile_ready
                and not workflow_active
                and not unsaved_completed
            ),
            # Policy-only construction neither consumes this receiver nor
            # loads its coordinate profile. Human setup may remain active in
            # parallel and is checked only at a later human-mode boundary.
            "policy_runtime_launch_ready": True,
            "human_runtime_launch_ready": bool(
                input_ready
                and profile_ready
                and not workflow_active
                and not unsaved_completed
            ),
            "input_ready": input_ready,
            "profile_ready": profile_ready,
            "unsaved_calibration": unsaved_completed,
            "selected_profile_path": (
                None if selected_profile is None else str(selected_profile)
            ),
            "selected_vive_source_role": selected_role,
            "calibration": calibration,
        }

    def setup_status(self) -> dict[str, object]:
        return self._setup_snapshot()

    def vive_status(self) -> dict[str, object]:
        application = self.current_application
        with self._lock:
            setup = self._vive_setup_controller
        if application is not None and (
            setup is None
            or bool(getattr(application, "human_input_attached", True))
        ):
            return dict(application.vive_status())
        if setup is None:
            return {
                "schema": "dexteleop.control_station.sim_hil_vive_status.v1",
                "sampled_at": time.time(),
                "vive_receiver": {
                    "status": "stopped",
                    "message": "VIVE setup receiver is not active.",
                    "trackers": {},
                },
            }
        source = getattr(setup, "source", None)
        from types import SimpleNamespace

        from control_station.sim_hil_profile.status import (
            _vive_receiver_status,
        )

        return {
            "schema": "dexteleop.control_station.sim_hil_vive_status.v1",
            "sampled_at": time.time(),
            "vive_receiver": _vive_receiver_status(
                SimpleNamespace(sources=(source,))
            ),
        }

    def _cancellation_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    def _reject(
        self,
        intent: SimHILAcceptanceIntent,
        *,
        code: str,
        message: str,
    ) -> SimHILAcceptanceSubmissionReceipt:
        return SimHILAcceptanceSubmissionReceipt(
            intent_id=intent.intent_id,
            accepted=False,
            revision=self._revision,
            rejection_code=code,
            message=message,
        )

    def submit(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> SimHILAcceptanceSubmissionReceipt:
        with self._lock:
            previous = self._last_sequence_by_client.get(intent.client_id, -1)
            if intent.client_sequence <= previous:
                return self._reject(
                    intent,
                    code="non_monotonic_client_sequence",
                    message=f"last accepted sequence is {previous}",
                )
            # Cancellation and shutdown are out-of-band lifecycle controls.
            # The status request that supplied their revision may be blocked
            # behind the operation being cancelled, so requiring an exact UI
            # revision here makes the only recovery action unreachable.
            if intent.intent_kind in {"cancel_operation", "stop_runtime"}:
                if self._active_intent_id is None and not self._queue:
                    if intent.intent_kind == "stop_runtime":
                        self._queue.append(intent)
                        self._last_sequence_by_client[intent.client_id] = (
                            intent.client_sequence
                        )
                        self._revision += 1
                        return SimHILAcceptanceSubmissionReceipt(
                            intent_id=intent.intent_id,
                            accepted=True,
                            revision=self._revision,
                            message="runtime stop queued",
                        )
                    return self._reject(
                        intent,
                        code="no_operation_in_progress",
                        message="there is no active or queued operation",
                    )
                self._cancel_requested.set()
                self._queue.clear()
                if intent.intent_kind == "stop_runtime":
                    self._queue.append(intent)
                self._last_sequence_by_client[intent.client_id] = (
                    intent.client_sequence
                )
                self._revision += 1
                return SimHILAcceptanceSubmissionReceipt(
                    intent_id=intent.intent_id,
                    accepted=True,
                    revision=self._revision,
                    message=(
                        "runtime stop queued after active-operation cancellation"
                        if intent.intent_kind == "stop_runtime"
                        else "cancellation requested"
                    ),
                )
            if intent.expected_revision != self._revision:
                return self._reject(
                    intent,
                    code="stale_revision",
                    message=(
                        f"expected {intent.expected_revision}, current "
                        f"{self._revision}"
                    ),
                )
            if len(self._queue) >= self.queue_capacity:
                return self._reject(
                    intent,
                    code="queue_full",
                    message="acceptance intent queue is full",
                )
            if self._active_intent_id is not None or self._queue:
                return self._reject(
                    intent,
                    code="operation_in_progress",
                    message="wait for the current acceptance intent",
                )
            self._queue.append(intent)
            self._last_sequence_by_client[intent.client_id] = (
                intent.client_sequence
            )
            self._revision += 1
            return SimHILAcceptanceSubmissionReceipt(
                intent_id=intent.intent_id,
                accepted=True,
                revision=self._revision,
                message="queued for the Station control thread",
            )

    def submit_json(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, object]:
        intent = sim_hil_acceptance_intent_from_json(payload)
        return self.submit(intent).to_payload()

    def _next_stage(self, stage_id: str) -> str | None:
        index = SIM_HIL_ACCEPTANCE_STAGES_V1.index(stage_id)
        if index + 1 >= len(SIM_HIL_ACCEPTANCE_STAGES_V1):
            return None
        return SIM_HIL_ACCEPTANCE_STAGES_V1[index + 1]

    def _dispatch_vive_setup(
        self,
        event: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            application = self.application
            if application is not None and bool(
                getattr(application, "human_input_attached", True)
            ):
                raise RuntimeError(
                    "VIVE calibration is unavailable after runtime "
                    "human-input attachment."
                )
            setup = self._vive_setup_controller
        if setup is None:
            raise RuntimeError("VIVE setup controller is unavailable.")
        setup.dispatch(event, payload)
        if event == "save_profile":
            saved = setup.last_saved_profile_path
            if saved is None:
                raise RuntimeError(
                    "Calibration reported save success without a profile path."
                )
            with self._lock:
                self._coordinate_profile_path = Path(saved).expanduser()
        return self._setup_snapshot()

    def _construct_application_after_vive_setup(
        self,
        *,
        initial_human_alpha: float = 1.0,
        human_required: bool | None = None,
    ) -> Any:
        setup_status = self._setup_snapshot()
        requires_human = (
            float(initial_human_alpha) > 1e-12
            if human_required is None
            else bool(human_required)
        )
        if requires_human and not bool(
            setup_status.get("runtime_launch_ready", False)
        ):
            raise RuntimeError(
                "MuJoCo construction is blocked until live VIVE input and a "
                "saved coordinate profile are both ready."
            )
        with self._lock:
            setup = self._vive_setup_controller
            factory = self.application_factory
            profile_path = self._coordinate_profile_path
        if setup is None or factory is None or profile_path is None:
            raise RuntimeError("Lazy Sim-HIL construction is not configured.")
        calibration = setup_status.get("calibration", {})
        mapping = (
            {}
            if not isinstance(calibration, Mapping)
            else dict(
                calibration.get("source_role_by_output_channel") or {}
            )
        )
        source_role = str(
            mapping.get("right_wrist_pose")
            or setup_status.get("selected_vive_source_role")
            or "left_wrist"
        )
        if source_role not in {"left_wrist", "right_wrist"}:
            raise RuntimeError(
                "Selected VIVE source role for right_wrist_pose is invalid: "
                f"{source_role!r}."
            )
        saved = setup.last_saved_profile_path
        if saved is not None:
            profile_path = Path(saved).expanduser()
        if self._cancellation_requested():
            raise InterruptedError(
                "Sim-HIL startup was cancelled before MuJoCo construction."
            )
        with self._lock:
            self._setup_phase = "constructing_runtime"
            self._selected_vive_source_role = source_role
        try:
            # Keep the setup receiver alive while model/scene construction is
            # potentially slow. The new application's source is only
            # constructed here; it is not started until after the handoff.
            application = factory(
                profile_path,
                source_role,
                float(initial_human_alpha),
            )
        except BaseException:
            with self._lock:
                self._setup_phase = "vive_setup"
            raise
        if self._cancellation_requested():
            try:
                application.close()
            finally:
                with self._lock:
                    self._setup_phase = "vive_setup"
                raise InterruptedError(
                    "Sim-HIL startup was cancelled after runtime construction "
                    "and before activation."
                )
        with self._lock:
            self._setup_phase = "vive_handoff"
        if requires_human:
            try:
                setup.close()
            except BaseException:
                try:
                    application.close()
                finally:
                    with self._lock:
                        self._setup_phase = "vive_setup_cleanup_failed"
                raise
        with self._lock:
            self.application = application
            self._coordinate_profile_path = profile_path
            if requires_human:
                self._vive_setup_controller = None
                self._setup_phase = "runtime_constructed"
            else:
                self._setup_phase = "policy_runtime_human_setup_available"
        return application

    def _restore_vive_setup_after_failed_launch(self) -> None:
        with self._lock:
            factory = self._vive_setup_factory
            application = self.application
            source_role = self._selected_vive_source_role
            existing_setup = self._vive_setup_controller
        if factory is None:
            raise RuntimeError(
                "Runtime rebuild is unavailable without a VIVE setup factory."
            )
        if application is not None:
            try:
                application.close()
            except Exception as error:
                # Do not start a replacement VIVE receiver while the previous
                # runtime reports an uncertain teardown.  That would turn a
                # cleanup fault into a port/GLFW resource leak.  The explicit
                # rebuild operation can retry the idempotent close and then
                # create a fresh setup owner.
                raise RuntimeError(
                    "Cannot restore VIVE setup after Sim-HIL runtime cleanup "
                    f"failed: {type(error).__name__}:{error}"
                ) from error
        setup = (
            existing_setup
            if existing_setup is not None
            else factory(source_role)
        )
        with self._lock:
            self.application = None
            self._vive_setup_controller = setup
            self._setup_phase = "vive_setup"

    def _abort_trace_capture_for_runtime_reset(
        self,
        application: Any | None,
    ) -> None:
        """Release an uncommitted acceptance episode before rebuilding.

        A failed first control cycle may already have prepared the trace.  A
        rebuild starts a new MuJoCo episode, so retaining that preparation
        would either collide with the next ``prepare`` call or make two
        runtime instances appear to share one capture identity.
        """

        capture = self.trace_capture
        if capture is None:
            return
        abort = getattr(capture, "abort", None)
        if not callable(abort):
            raise RuntimeError(
                "Acceptance capture cannot be reset because its abort hook "
                "is unavailable."
            )
        clock = getattr(application, "clock", None)
        now = (
            float(clock())
            if callable(clock)
            else time.monotonic()
        )
        abort(now=now)

    def _rotate_trace_capture_after_runtime_reset(self) -> None:
        """Give the next runtime a disjoint ControlTrace fact namespace."""

        capture = self.trace_capture
        if capture is None:
            return
        rotate = getattr(capture, "rotate_runtime_session", None)
        if not callable(rotate):
            raise RuntimeError(
                "Acceptance capture cannot rotate after runtime rebuild."
            )
        rotate(now=time.monotonic())

    @staticmethod
    def _control_mode_alpha(
        control_mode: str,
        *,
        requested_alpha: float | None,
    ) -> float:
        mode = str(control_mode)
        if mode == "human_teleop":
            return 1.0
        if mode == "ppo_only":
            return 0.0
        if mode == "proportional_blend":
            if requested_alpha is None:
                raise ValueError(
                    "proportional_blend requires human_alpha."
                )
            return float(requested_alpha)
        raise ValueError(f"Unsupported Sim-HIL control mode {mode!r}.")

    def _select_control_mode(
        self,
        *,
        control_mode: str,
        human_alpha: float | None,
    ) -> dict[str, object]:
        if self.allocation_runtime_enabled:
            raise RuntimeError(
                "Allocation-aware Sim-HIL does not accept fixed-alpha control "
                "modes; use set_allocation_mode."
            )
        application = self._require_application()
        application_status = application.status()
        if not bool(application_status.get("started", False)):
            raise RuntimeError(
                "Start the Sim-HIL runtime before selecting a control mode."
            )
        if bool(application_status.get("episode_ended", False)):
            raise RuntimeError(
                "The Sim-HIL episode has ended; start a new session."
            )
        alpha = self._control_mode_alpha(
            control_mode,
            requested_alpha=human_alpha,
        )
        previous_alpha = self._active_human_alpha
        if previous_alpha is None:
            mix = dict(
                dict(application_status.get("composition") or {}).get("mix")
                or {}
            )
            previous_values = tuple(
                float(value)
                for value in dict(
                    mix.get("human_alpha_by_component") or {}
                ).values()
            )
            if previous_values and all(
                abs(value - previous_values[0]) <= 1e-12
                for value in previous_values[1:]
            ):
                previous_alpha = previous_values[0]
            elif mix.get("mode") == "policy_only":
                previous_alpha = 0.0
            elif mix.get("mode") == "human_only":
                previous_alpha = 1.0
        human_preparation = None
        automatically_held = False
        previous_human_active = bool(
            previous_alpha is not None and previous_alpha > 1e-12
        )
        previous_policy_active = bool(
            previous_alpha is not None
            and previous_alpha < 1.0 - 1e-12
        )
        activates_human = alpha > 1e-12 and not previous_human_active
        activates_policy = (
            alpha < 1.0 - 1e-12 and not previous_policy_active
        )
        if activates_human:
            human_preparation, automatically_held = (
                self._prepare_human_mode(
                    application,
                    recapture_runtime_home=True,
                )
            )
        elif activates_policy:
            current = application.status()
            automatically_held = not bool(current.get("held", False))
            if automatically_held:
                application.hold(reason="policy_mode_preflight")
        try:
            receipt = application.set_human_blend_alpha(alpha)
            if automatically_held:
                application.rearm(
                    cancel_requested=self._cancellation_requested,
                )
        except BaseException:
            if previous_alpha is not None:
                try:
                    application.set_human_blend_alpha(previous_alpha)
                except BaseException:
                    pass
            if automatically_held:
                try:
                    application.rearm(
                        cancel_requested=self._cancellation_requested,
                    )
                except BaseException:
                    pass
            raise
        with self._lock:
            self._active_control_mode = str(control_mode)
            self._active_human_alpha = alpha
            self._control_mode_change_count += 1
        return {
            "control_mode": str(control_mode),
            "human_alpha": alpha,
            "blend_receipt_id": getattr(
                receipt,
                "parameter_snapshot_id",
                getattr(receipt, "semantic_content_hash", None),
            ),
            "human_preparation": human_preparation,
        }

    def _prepare_human_mode(
        self,
        application: Any,
        *,
        recapture_runtime_home: bool,
    ) -> tuple[dict[str, object], bool]:
        """Make the passive human contributor ready before mode publication."""

        status = application.status()
        automatically_held = not bool(status.get("held", False))
        if not bool(getattr(application, "human_input_attached", True)):
            setup_status = self._setup_snapshot()
            if not bool(setup_status.get("runtime_launch_ready", False)):
                raise RuntimeError(
                    "Human/blend mode requires live human input and a saved "
                    "VIVE coordinate profile."
                )
            with self._lock:
                setup = self._vive_setup_controller
                profile_path = self._coordinate_profile_path
            if setup is None or profile_path is None:
                raise RuntimeError(
                    "Human input setup is unavailable for attachment."
                )
            calibration = setup_status.get("calibration", {})
            mapping = (
                {}
                if not isinstance(calibration, Mapping)
                else dict(
                    calibration.get(
                        "source_role_by_output_channel",
                        {},
                    )
                    or {}
                )
            )
            source_role = str(
                mapping.get("right_wrist_pose")
                or setup_status.get("selected_vive_source_role")
                or "left_wrist"
            )
            saved = getattr(setup, "last_saved_profile_path", None)
            if saved is not None:
                profile_path = Path(saved).expanduser()
            if automatically_held:
                application.hold(reason="human_input_attachment")
            attachment = None
            setup_closed = False
            try:
                attachment = application.prepare_human_attachment(
                    coordinate_profile_path=profile_path,
                    source_role=source_role,
                )
                setup.close()
                setup_closed = True
                activation = application.activate_human_attachment(
                    attachment
                )
            except BaseException:
                close_source = getattr(
                    getattr(attachment, "source", None),
                    "close",
                    None,
                )
                if callable(close_source):
                    try:
                        close_source()
                    except BaseException:
                        pass
                if (
                    setup_closed
                    and not bool(
                        getattr(
                            application,
                            "human_input_attached",
                            True,
                        )
                    )
                ):
                    replacement = (
                        None
                        if self._vive_setup_factory is None
                        else self._vive_setup_factory(source_role)
                    )
                    with self._lock:
                        self._vive_setup_controller = replacement
                        self._setup_phase = (
                            "policy_runtime_human_setup_available"
                            if replacement is not None
                            else "human_setup_unavailable"
                        )
                if automatically_held:
                    try:
                        application.rearm(
                            cancel_requested=self._cancellation_requested,
                        )
                    except BaseException:
                        pass
                raise
            with self._lock:
                self._vive_setup_controller = None
                self._coordinate_profile_path = profile_path
                self._selected_vive_source_role = source_role
                self._setup_phase = "runtime_constructed"
        else:
            activation = {
                "status": "already_attached",
                "command_dispatched": False,
                "authoritative": False,
            }
            if automatically_held:
                application.hold(reason="human_mode_preflight")

        try:
            passive_input = (
                application.run_human_input_acceptance_check(
                    timeout_sec=self.device_check_timeout_sec,
                )
                if self.application_factory is None
                else application.run_human_input_acceptance_check(
                    timeout_sec=self.device_check_timeout_sec,
                    cancel_requested=self._cancellation_requested,
                )
            )
            current = application.status()
            if (
                recapture_runtime_home
                or current.get("runtime_home") is None
            ):
                runtime_home = application.capture_runtime_home()
            else:
                runtime_home = dict(current["runtime_home"])
        except BaseException:
            if automatically_held:
                try:
                    application.rearm(
                        cancel_requested=self._cancellation_requested,
                    )
                except BaseException:
                    pass
            raise
        return (
            {
                "attachment": dict(activation),
                "passive_input": dict(passive_input),
                "runtime_home": dict(runtime_home),
            },
            automatically_held,
        )

    def _start_runtime(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        if self.allocation_runtime_enabled:
            raise RuntimeError(
                "The selected allocation profile requires "
                "start_allocation_runtime."
            )
        control_mode = str(intent.control_mode)
        initial_alpha = self._control_mode_alpha(
            control_mode,
            requested_alpha=intent.human_alpha,
        )
        return self._start_application_runtime(
            intent,
            initial_human_alpha=initial_alpha,
            human_required=initial_alpha > 1e-12,
            control_mode=control_mode,
        )

    def _start_allocation_runtime(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        """Start an allocation profile without publishing fixed-alpha state."""

        if not self.allocation_runtime_enabled:
            raise RuntimeError(
                "start_allocation_runtime requires an allocation-aware "
                "application profile."
            )
        return self._start_application_runtime(
            intent,
            # Allocation profiles ignore this fixed-alpha compatibility
            # field.  Zero selects deferred human construction only; the
            # allocation context remains profile-owned and starts from the
            # policy baseline while Human is absent.
            initial_human_alpha=0.0,
            human_required=False,
            control_mode=None,
        )

    def _start_application_runtime(
        self,
        intent: SimHILAcceptanceIntent,
        *,
        initial_human_alpha: float,
        human_required: bool,
        control_mode: str | None,
    ) -> dict[str, object]:
        initial_alpha = float(initial_human_alpha)
        existing = self.current_application
        if existing is not None and bool(
            getattr(existing, "started", False)
        ):
            raise RuntimeError("The Sim-HIL runtime is already started.")
        application = (
            self._require_application()
            if existing is not None
            else self._construct_application_after_vive_setup(
                initial_human_alpha=initial_alpha,
                human_required=human_required,
            )
        )
        if human_required:
            if self.application_factory is None:
                passive_input = (
                    application.run_human_input_acceptance_check(
                        timeout_sec=self.device_check_timeout_sec,
                    )
                )
            else:
                passive_input = (
                    application.run_human_input_acceptance_check(
                        timeout_sec=self.device_check_timeout_sec,
                        cancel_requested=self._cancellation_requested,
                    )
                )
            if self._cancellation_requested():
                raise InterruptedError(
                    "Sim-HIL startup was cancelled after VIVE readiness."
                )
            runtime_home = application.capture_runtime_home()
        else:
            passive_input = {
                "status": "not_required",
                "reason": "policy_only_start",
                "command_dispatched": False,
                "authoritative": False,
            }
            runtime_home = {
                "status": "not_required",
                "reason": "policy_only_start",
                "command_dispatched": False,
                "authoritative": False,
            }
        report = (
            application.start()
            if self.application_factory is None
            else application.start(
                cancel_requested=self._cancellation_requested,
            )
        )
        if self.trace_capture is not None:
            self.trace_capture.prepare(now=application.clock())
        if existing is None and control_mode is not None:
            # Production constructs the composition contract with the selected
            # alpha. Startup readiness and first activation therefore use the
            # requested mode without a redundant post-start parameter update.
            mode = {
                "control_mode": control_mode,
                "human_alpha": initial_alpha,
                "blend_receipt_id": None,
                "mode_activation": "initial_profile",
            }
            with self._lock:
                self._active_control_mode = control_mode
                self._active_human_alpha = initial_alpha
                self._control_mode_change_count += 1
        elif existing is not None and control_mode is not None:
            # Direct application injection is retained for lightweight tests
            # and compatibility callers whose contract already exists.
            mode = self._select_control_mode(
                control_mode=control_mode,
                human_alpha=intent.human_alpha,
            )
        else:
            mode = {
                "control_profile_start": "allocation",
                "human_input_required": bool(human_required),
                "human_input_deferred": not bool(human_required),
                "initial_allocation_mode": "automatic",
            }
        with self._lock:
            self._formal_runtime_started = True
        return {
            "intent_kind": intent.intent_kind,
            "status": "running",
            "passive_input": dict(passive_input),
            "runtime_home": dict(runtime_home),
            "preflight_passed": bool(report.passed),
            "mujoco_viewer_required": True,
            **mode,
        }

    def _start_stage(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        stage_id = str(intent.stage_id)
        expected_index = len(self._completed)
        expected_stage = (
            None
            if expected_index >= len(SIM_HIL_ACCEPTANCE_STAGES_V1)
            else SIM_HIL_ACCEPTANCE_STAGES_V1[expected_index]
        )
        if stage_id != expected_stage:
            raise RuntimeError(
                f"Next acceptance stage is {expected_stage}, not {stage_id}."
            )
        if self._stage_status_by_id[stage_id] not in {
            "available",
            "faulted",
        }:
            raise RuntimeError(
                f"Acceptance stage {stage_id} is not startable."
            )
        self._stage_status_by_id[stage_id] = "running"
        self._active_stage_id = stage_id
        if stage_id == "human_teleop":
            application = (
                self._require_application()
                if self.current_application is not None
                else self._construct_application_after_vive_setup()
            )
            if self.application_factory is None:
                passive_input = (
                    application.run_human_input_acceptance_check(
                        timeout_sec=self.device_check_timeout_sec,
                    )
                )
            else:
                passive_input = (
                    application.run_human_input_acceptance_check(
                        timeout_sec=self.device_check_timeout_sec,
                        cancel_requested=self._cancellation_requested,
                    )
                )
            if self._cancellation_requested():
                raise InterruptedError(
                    "Sim-HIL startup was cancelled after VIVE readiness."
                )
            runtime_home = application.capture_runtime_home()
            before = application.status()
            self._baseline_cycle_by_stage[stage_id] = int(
                (
                    before.get("composition", {})
                    .get("last_cycle", {})
                    or {}
                ).get("cycle_index", -1)
            )
            self._requested_alpha_by_stage[stage_id] = 1.0
            # Do not prepare the episode before the native MuJoCo viewer is
            # known to have started.  ``launch_passive`` is a native call and
            # a SIGSEGV cannot run Python ``finally`` blocks; preparing here
            # would leave the recorder lifecycle active in a retryable Python
            # process and the next start would fail with
            # ``episode preparation is already active``.  Runtime startup
            # facts emitted before this point remain session-level facts; the
            # acceptance episode begins only after the complete startup
            # boundary has succeeded.
            report = (
                application.start()
                if self.application_factory is None
                else application.start(
                    cancel_requested=self._cancellation_requested,
                )
            )
            receipt = application.set_human_blend_alpha(1.0)
            if self.trace_capture is not None:
                self.trace_capture.prepare(now=application.clock())
            result = {
                "passive_input": dict(passive_input),
                "runtime_home": dict(runtime_home),
                "preflight_passed": bool(report.passed),
                "mujoco_viewer_required": True,
                "blend_receipt_id": getattr(
                    receipt,
                    "parameter_snapshot_id",
                    getattr(receipt, "semantic_content_hash", None),
                ),
                "human_alpha": 1.0,
            }
        elif stage_id == "ppo_only":
            application = self._require_application()
            before = application.status()
            self._baseline_cycle_by_stage[stage_id] = int(
                (
                    before.get("composition", {})
                    .get("last_cycle", {})
                    or {}
                ).get("cycle_index", -1)
            )
            self._requested_alpha_by_stage[stage_id] = 0.0
            receipt = application.set_human_blend_alpha(0.0)
            result = {
                "blend_receipt_id": getattr(
                    receipt,
                    "parameter_snapshot_id",
                    getattr(receipt, "semantic_content_hash", None),
                ),
                "human_alpha": 0.0,
            }
        else:
            application = self._require_application()
            before = application.status()
            self._baseline_cycle_by_stage[stage_id] = int(
                (
                    before.get("composition", {})
                    .get("last_cycle", {})
                    or {}
                ).get("cycle_index", -1)
            )
            self._requested_alpha_by_stage[stage_id] = float(
                intent.human_alpha
            )
            receipt = application.set_human_blend_alpha(
                float(intent.human_alpha)
            )
            result = {
                "blend_receipt_id": getattr(
                    receipt,
                    "parameter_snapshot_id",
                    getattr(receipt, "semantic_content_hash", None),
                ),
                "human_alpha": float(intent.human_alpha),
            }
        self._stage_status_by_id[stage_id] = "awaiting_confirmation"
        self._stage_evidence_by_id[stage_id] = dict(result)
        return {
            "intent_kind": intent.intent_kind,
            "stage_id": stage_id,
            "status": "awaiting_confirmation",
            "evidence": result,
        }

    def _stage_confirmation_failures(
        self,
        stage_id: str,
    ) -> tuple[str, ...]:
        status = self._require_application().status()
        failures: list[str] = []
        if stage_id == "human_teleop":
            evidence = self._stage_evidence_by_id.get(stage_id, {})
            passive = (
                evidence.get("passive_input", {})
                if isinstance(evidence, Mapping)
                else {}
            )
            if not isinstance(passive, Mapping) or passive.get("status") != "ready":
                failures.append("passive_input_not_ready")
            visualization = status.get("visualization", {})
            if (
                not isinstance(visualization, Mapping)
                or not bool(visualization.get("enabled", False))
                or not bool(visualization.get("running", False))
            ):
                failures.append("mujoco_viewer_not_running")
        if not bool(status.get("started", False)):
            failures.append("application_not_started")
        if bool(status.get("held", False)):
            failures.append("application_held")
        gate = status.get("execution_gate", {})
        if not isinstance(gate, Mapping) or gate.get("status") != "ready":
            failures.append("execution_gate_not_ready")
        composition = status.get("composition", {})
        last_cycle = (
            {}
            if not isinstance(composition, Mapping)
            else composition.get("last_cycle", {}) or {}
        )
        cycle_index = (
            -1
            if not isinstance(last_cycle, Mapping)
            else int(last_cycle.get("cycle_index", -1))
        )
        baseline = int(self._baseline_cycle_by_stage.get(stage_id, -1))
        if cycle_index <= baseline:
            failures.append("no_post_stage_control_cycle")
        execution = (
            {}
            if not isinstance(last_cycle, Mapping)
            else last_cycle.get("execution", {}) or {}
        )
        if (
            not isinstance(execution, Mapping)
            or execution.get("status") != "dispatched"
        ):
            failures.append("latest_cycle_not_dispatched")
        mix = (
            {}
            if not isinstance(composition, Mapping)
            else composition.get("mix", {}) or {}
        )
        expected_mode = {
            "human_teleop": "human_only",
            "ppo_only": "policy_only",
            "proportional_blend": "proportional_blend",
        }[stage_id]
        if not isinstance(mix, Mapping) or mix.get("mode") != expected_mode:
            failures.append("composition_mix_mode_mismatch")
        expected_alpha = self._requested_alpha_by_stage.get(stage_id)
        alpha_by_component = (
            {}
            if not isinstance(mix, Mapping)
            else mix.get("human_alpha_by_component", {}) or {}
        )
        if expected_alpha is None or not isinstance(
            alpha_by_component,
            Mapping,
        ) or not alpha_by_component or any(
            abs(float(value) - expected_alpha) > 1e-9
            for value in alpha_by_component.values()
        ):
            failures.append("composition_alpha_mismatch")
        if stage_id == "proportional_blend":
            composition_step = (
                {}
                if not isinstance(last_cycle, Mapping)
                else last_cycle.get("composition_step", {}) or {}
            )
            if (
                not isinstance(composition_step, Mapping)
                or composition_step.get("immediate_difference") is None
            ):
                failures.append("immediate_difference_missing")
            execution_difference = status.get(
                "execution_difference",
                {},
            )
            latest_difference = (
                {}
                if not isinstance(execution_difference, Mapping)
                else execution_difference.get("latest", {}) or {}
            )
            if (
                isinstance(execution_difference, Mapping)
                and bool(execution_difference.get("enabled", False))
                and (
                    not isinstance(latest_difference, Mapping)
                    or latest_difference.get("status") != "available"
                )
            ):
                failures.append("execution_difference_missing")
        return tuple(failures)

    def _confirm_stage(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        stage_id = str(intent.stage_id)
        if stage_id != self._active_stage_id:
            raise RuntimeError(
                f"Active acceptance stage is {self._active_stage_id}."
            )
        if self._stage_status_by_id[stage_id] != "awaiting_confirmation":
            raise RuntimeError(
                f"Acceptance stage {stage_id} is not awaiting confirmation."
            )
        failures = self._stage_confirmation_failures(stage_id)
        if failures:
            raise RuntimeError(
                "Acceptance evidence is incomplete: "
                + ",".join(failures)
            )
        self._stage_status_by_id[stage_id] = "passed"
        self._completed.append(stage_id)
        self._active_stage_id = None
        next_stage = self._next_stage(stage_id)
        if next_stage is not None:
            self._stage_status_by_id[next_stage] = "available"
        return {
            "intent_kind": intent.intent_kind,
            "stage_id": stage_id,
            "status": "passed",
            "next_stage_id": next_stage,
        }

    def _execute(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        if intent.intent_kind in {
            "start_runtime",
            "start_allocation_runtime",
            "set_control_mode",
            "set_allocation_mode",
            "request_residual_rebase_and_clear",
            "start_stage",
            "confirm_stage",
            "rearm",
        } and self.trace_capture is not None:
            capture_status = self.trace_capture.status()
            if capture_status.get("status") == "faulted":
                raise RuntimeError(
                    str(
                        capture_status.get("message")
                        or "ControlTrace writer fault"
                    )
                )
        if intent.intent_kind == "begin_vive_calibration":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup(
                    "begin_calibration"
                ),
            }
        if intent.intent_kind == "capture_vive_calibration":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup("capture"),
            }
        if intent.intent_kind == "reset_vive_calibration_stage":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup("reset_stage"),
            }
        if intent.intent_kind == "cancel_vive_calibration":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup(
                    "cancel_calibration"
                ),
            }
        if intent.intent_kind == "save_vive_calibration_profile":
            return {
                "intent_kind": intent.intent_kind,
                "status": "saved",
                "setup": self._dispatch_vive_setup(
                    "save_profile",
                    {"profile_name": intent.profile_name or ""},
                ),
            }
        if intent.intent_kind == "start_runtime":
            return self._start_runtime(intent)
        if intent.intent_kind == "start_allocation_runtime":
            return self._start_allocation_runtime(intent)
        if intent.intent_kind == "set_control_mode":
            mode = self._select_control_mode(
                control_mode=str(intent.control_mode),
                human_alpha=intent.human_alpha,
            )
            return {
                "intent_kind": intent.intent_kind,
                "status": "mode_selected",
                **mode,
            }
        if intent.intent_kind in {
            "set_allocation_mode",
            "request_residual_rebase_and_clear",
        }:
            from control_station.composition.mujoco_policy_human_allocation_operations import (
                MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER,
                MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
                MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
                MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR,
                MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY,
            )

            application = self._require_application()
            requested_mode = (
                None
                if intent.intent_kind == "request_residual_rebase_and_clear"
                else str(intent.allocation_mode)
            )
            needs_human = (
                intent.intent_kind == "request_residual_rebase_and_clear"
                or requested_mode != "base_only"
            )
            human_preparation = None
            automatically_held = False
            if needs_human:
                human_preparation, automatically_held = (
                    self._prepare_human_mode(
                        application,
                        # Runtime Home belongs to the device/target binding,
                        # not to every allocation-mode click.  First attach
                        # captures it because none exists; later mode changes
                        # retain the established Home.
                        recapture_runtime_home=False,
                    )
                )
            signal_name = (
                MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR
                if intent.intent_kind == "request_residual_rebase_and_clear"
                else {
                    "automatic": MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
                    "base_only": MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
                    "relative_overlay": (
                        MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY
                    ),
                    "absolute_overlay_owner": (
                        MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER
                    ),
                }[str(intent.allocation_mode)]
            )
            try:
                receipt = application.submit_allocation_button_operation(
                    signal_name=signal_name,
                    event_id=intent.intent_id,
                    source_fact_id=intent.intent_id,
                    source_fact_hash=intent.intent_id.rsplit(":", 1)[-1],
                    occurred_at=time.monotonic(),
                )
                if automatically_held:
                    application.rearm(
                        cancel_requested=self._cancellation_requested,
                    )
            except BaseException:
                if automatically_held:
                    try:
                        application.rearm(
                            cancel_requested=self._cancellation_requested,
                        )
                    except BaseException:
                        pass
                raise
            override_receipt = getattr(receipt, "override_receipt", None)
            anchor_request = getattr(receipt, "anchor_request", None)
            return {
                "intent_kind": intent.intent_kind,
                "status": (
                    "residual_rebase_requested"
                    if anchor_request is not None
                    else "allocation_mode_selected"
                ),
                "signal_name": signal_name,
                "allocation_mode": intent.allocation_mode,
                "operation_receipt_id": getattr(receipt, "receipt_id", None),
                "operation_receipt_hash": getattr(
                    receipt,
                    "semantic_content_hash",
                    None,
                ),
                "allocation_override_receipt_id": getattr(
                    override_receipt,
                    "receipt_id",
                    None,
                ),
                "anchor_request_id": getattr(
                    anchor_request,
                    "request_id",
                    None,
                ),
                "human_preparation": human_preparation,
            }
        if intent.intent_kind == "set_blend_failure_fallback":
            application = self._require_application()
            if not bool(application.status().get("started", False)):
                raise RuntimeError(
                    "Start the Sim-HIL runtime before selecting a blend "
                    "failure fallback."
                )
            receipt = application.set_blend_failure_fallback(
                str(intent.fallback_source),
                grace_sec=intent.blend_failure_grace_sec,
            )
            return {
                "intent_kind": intent.intent_kind,
                "status": "fallback_selected",
                **dict(receipt),
            }
        if intent.intent_kind == "set_human_timeout_fallback":
            application = self._require_application()
            if not bool(application.status().get("started", False)):
                raise RuntimeError(
                    "Start the Sim-HIL runtime before selecting a human "
                    "input timeout fallback."
                )
            receipt = application.set_human_timeout_fallback(
                str(intent.fallback_source)
            )
            return {
                "intent_kind": intent.intent_kind,
                "status": "human_timeout_fallback_selected",
                **dict(receipt),
            }
        if intent.intent_kind == "start_stage":
            return self._start_stage(intent)
        if intent.intent_kind == "confirm_stage":
            return self._confirm_stage(intent)
        if intent.intent_kind == "swap_vive_sources":
            # Swapping is a pure pre-runtime source-role mapping operation.
            # It must remain available even when the currently selected
            # sender is missing/invalid; readiness is evaluated only when the
            # resulting mapping is used to start the stage.
            application = self.current_application
            human_attached = bool(
                application is not None
                and getattr(application, "started", False)
                and getattr(application, "human_input_attached", True)
            )
            if human_attached or "human_teleop" in self._completed:
                raise RuntimeError(
                    "VIVE sources may be swapped only before human input is "
                    "attached to the Sim-HIL runtime."
                )
            with self._lock:
                setup_available = self._vive_setup_controller is not None
            if setup_available:
                setup = self._dispatch_vive_setup("swap_vive_sources")
                receipt = {
                    "status": "swapped",
                    "source_role_by_output_channel": dict(
                        (
                            setup.get("calibration", {})
                            if isinstance(
                                setup.get("calibration", {}),
                                Mapping,
                            )
                            else {}
                        ).get(
                            "source_role_by_output_channel",
                            {},
                        )
                    ),
                }
            elif application is not None:
                receipt = (
                    self._require_application()
                    .swap_vive_left_right_sources()
                )
            else:
                raise RuntimeError("VIVE setup controller is unavailable.")
            self._stage_status_by_id["human_teleop"] = "available"
            self._active_stage_id = None
            self._stage_evidence_by_id.pop("human_teleop", None)
            return {
                "intent_kind": "swap_vive_sources",
                "status": "swapped",
                "receipt": receipt,
            }
        if intent.intent_kind == "hold":
            self._require_application().hold(reason=intent.reason)
            return {"intent_kind": "hold", "status": "held"}
        if intent.intent_kind == "rearm":
            report = self._require_application().rearm(
                cancel_requested=self._cancellation_requested,
            )
            return {
                "intent_kind": "rearm",
                "status": "rearmed",
                "preflight_passed": bool(report.passed),
            }
        if intent.intent_kind == "capture_runtime_home":
            receipt = self._require_application().capture_runtime_home()
            return {
                "intent_kind": intent.intent_kind,
                "status": "runtime_home_captured",
                "receipt": dict(receipt),
            }
        if intent.intent_kind == "rebuild_runtime":
            # This is a destructive whole-runtime reset, not Home capture.
            # The formal runtime may rebuild from HOLD at any time; its
            # unfinalized trace is explicitly aborted.  Preserve the older
            # staged restriction only for migration callers.
            if self._completed and not self._formal_runtime_started:
                raise RuntimeError(
                    "Runtime rebuild is only available before an acceptance "
                    "stage has been committed."
                )
            application = self.current_application
            if application is None:
                raise RuntimeError("No constructed Sim-HIL runtime to rebuild.")
            if bool(getattr(application, "started", False)) and not bool(
                getattr(application, "_held", False)
            ):
                application.hold(reason=intent.reason)
            self._abort_trace_capture_for_runtime_reset(application)
            self._restore_vive_setup_after_failed_launch()
            self._rotate_trace_capture_after_runtime_reset()
            with self._lock:
                self._stage_status_by_id["human_teleop"] = "available"
                self._active_stage_id = None
                self._stage_evidence_by_id.pop("human_teleop", None)
                self._baseline_cycle_by_stage.pop("human_teleop", None)
                self._requested_alpha_by_stage.pop("human_teleop", None)
                self._active_control_mode = None
                self._active_human_alpha = None
                self._formal_runtime_started = False
                self._last_error = None
            return {
                "intent_kind": intent.intent_kind,
                "status": "runtime_rebuilt",
                "setup": self._setup_snapshot(),
            }
        if intent.intent_kind == "finalize_capture":
            if (
                not self._formal_runtime_started
                and tuple(self._completed) != SIM_HIL_ACCEPTANCE_STAGES_V1
            ):
                raise RuntimeError(
                    "All acceptance stages must pass before capture finalize."
                )
            application = self._require_application()
            application.end_episode(
                reason="operator_sim_hil_session_complete"
            )
            if self.trace_capture is None:
                raise RuntimeError(
                    "Acceptance capture is not bound; refusing to report "
                    "a completed data-retention stage."
                )
            capture_status = self.trace_capture.finalize(
                now=application.clock()
            )
            return {
                "intent_kind": "finalize_capture",
                "status": "complete",
                "capture": capture_status,
            }
        if intent.intent_kind == "stop_runtime":
            application = self.current_application
            if application is not None and bool(
                getattr(application, "started", False)
            ):
                application.hold(reason=intent.reason)
            self._shutdown_requested = True
            return {
                "intent_kind": "stop_runtime",
                "status": "shutdown_requested",
            }
        raise RuntimeError(
            f"Unsupported acceptance intent {intent.intent_kind}."
        )

    def process_next(self) -> dict[str, object] | None:
        """Run at most one queued operation on the Station control thread."""

        with self._lock:
            if not self._queue:
                return None
            self._cancel_requested.clear()
            intent = self._queue.popleft()
            self._active_intent_id = intent.intent_id
            self._revision += 1
        try:
            result = self._execute(intent)
        except Exception as error:
            capture_reset_error: BaseException | None = None
            startup_intent = bool(
                intent.intent_kind in {
                    "start_runtime",
                    "start_allocation_runtime",
                }
                or (
                    intent.intent_kind == "start_stage"
                    and intent.stage_id == "human_teleop"
                )
            )
            if startup_intent:
                try:
                    self._abort_trace_capture_for_runtime_reset(
                        self.current_application
                    )
                except BaseException as reset_error:  # noqa: BLE001 - preserve startup cause.
                    capture_reset_error = reset_error
                application = self.current_application
                if application is not None and bool(
                    getattr(application, "started", False)
                ) and not bool(getattr(application, "_held", False)):
                    try:
                        application.hold(
                            reason="sim_hil_startup_failed"
                        )
                    except BaseException as hold_error:  # noqa: BLE001
                        if capture_reset_error is None:
                            capture_reset_error = hold_error
            should_restore_setup = (
                startup_intent
                and self.application_factory is not None
                and (
                    self.current_application is not None
                    or self._vive_setup_controller is None
                )
                and not bool(
                    getattr(self.current_application, "started", False)
                )
            )
            if should_restore_setup:
                try:
                    self._restore_vive_setup_after_failed_launch()
                    self._rotate_trace_capture_after_runtime_reset()
                except Exception as restore_error:
                    error = RuntimeError(
                        f"{type(error).__name__}:{error}; "
                        "vive_setup_restore_failed="
                        f"{type(restore_error).__name__}:{restore_error}"
                    )
            with self._lock:
                if (
                    self._active_stage_id is not None
                    and self._stage_status_by_id[
                        self._active_stage_id
                    ]
                    == "running"
                ):
                    failed_status = (
                        "restart_required"
                        if bool(
                            getattr(
                                self.application,
                                "restart_required",
                                False,
                            )
                        )
                        else "faulted"
                    )
                    self._stage_status_by_id[
                        self._active_stage_id
                    ] = failed_status
                self._last_error = f"{type(error).__name__}:{error}"
                if capture_reset_error is not None:
                    self._last_error += (
                        "; capture_reset_failed="
                        f"{type(capture_reset_error).__name__}:"
                        f"{capture_reset_error}"
                    )
                self._last_result = {
                    "intent_id": intent.intent_id,
                    "intent_kind": intent.intent_kind,
                    "status": "failed",
                    "error": self._last_error,
                }
                self._active_intent_id = None
                self._revision += 1
                return dict(self._last_result)
        with self._lock:
            self._last_error = None
            self._last_result = {
                "intent_id": intent.intent_id,
                **result,
            }
            self._active_intent_id = None
            self._revision += 1
            return dict(self._last_result)

    def status(self) -> dict[str, object]:
        setup = self._setup_snapshot()
        with self._lock:
            stage_status = MappingProxyType(
                dict(self._stage_status_by_id)
            )
            capture = (
                {"enabled": False, "status": "not_bound"}
                if self.trace_capture is None
                else self.trace_capture.status()
            )
            return {
                "schema": SIM_HIL_ACCEPTANCE_STATUS_SCHEMA_V1,
                "revision": self._revision,
                "active_stage_id": self._active_stage_id,
                "available_control_modes": (
                    []
                    if self.allocation_runtime_enabled
                    else list(SIM_HIL_CONTROL_MODES_V1)
                ),
                "available_blend_failure_fallbacks": list(
                    SIM_HIL_BLEND_FAILURE_FALLBACKS_V1
                ),
                "available_human_timeout_fallbacks": list(
                    SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1
                ),
                "active_control_mode": self._active_control_mode,
                "active_human_alpha": self._active_human_alpha,
                "control_mode_change_count": (
                    self._control_mode_change_count
                ),
                "formal_runtime_started": self._formal_runtime_started,
                "allocation_runtime_enabled": (
                    self.allocation_runtime_enabled
                ),
                "active_intent_id": self._active_intent_id,
                "pending_intent_count": len(self._queue),
                "completed_stage_ids": list(self._completed),
                "stage_status_by_id": dict(stage_status),
                "last_result": (
                    None
                    if self._last_result is None
                    else dict(self._last_result)
                ),
                "last_error": self._last_error,
                "capture": capture,
                "shutdown_requested": self._shutdown_requested,
                "operation_cancel_available": bool(
                    self._active_intent_id is not None
                ),
                "setup": setup,
            }

    def enforce_capture_health(self) -> str | None:
        """Latch formal execution in HOLD after a capture-path fault."""

        capture = self.trace_capture
        if capture is None:
            return None
        capture_status = capture.status()
        if capture_status.get("status") != "faulted":
            return None
        message = str(
            capture_status.get("message")
            or "ControlTrace writer fault"
        )
        application = self.current_application
        hold_error = None
        if (
            application is not None
            and bool(getattr(application, "started", False))
            and not bool(getattr(application, "_held", False))
            and not bool(getattr(application, "_episode_ended", False))
        ):
            try:
                application.hold(
                    reason="formal_control_trace_capture_fault"
                )
            except BaseException as error:  # noqa: BLE001 - retain both facts.
                hold_error = f"{type(error).__name__}:{error}"
        with self._lock:
            failure = (
                message
                if hold_error is None
                else f"{message}; hold_failed={hold_error}"
            )
            if self._last_error != failure:
                self._last_error = failure
                self._revision += 1
            return self._last_error

    def close(self) -> None:
        with self._lock:
            setup = self._vive_setup_controller
            application = self.application
        cleanup_errors: list[BaseException] = []
        # Close the constructed runtime first so its viewer, composition
        # worker, and passive VIVE source are drained before the setup owner is
        # touched.  The outer Station loop may already have closed it; the
        # application lifecycle is intentionally idempotent.
        if application is not None:
            close_application = getattr(application, "close", None)
            if callable(close_application):
                try:
                    close_application()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    cleanup_errors.append(error)
        if setup is not None:
            try:
                setup.close()
            except BaseException as error:  # noqa: BLE001 - cleanup continues.
                cleanup_errors.append(error)
        capture = self.trace_capture
        if capture is not None:
            application = self.current_application
            try:
                capture.close(
                    now=(
                        time.monotonic()
                        if application is None
                        else application.clock()
                    )
                )
            except BaseException as error:  # noqa: BLE001 - report after all cleanup.
                cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError(
                "Sim-HIL acceptance cleanup encountered errors: "
                + "; ".join(
                    f"{type(error).__name__}:{error}"
                    for error in cleanup_errors
                )
            ) from cleanup_errors[0]


__all__ = [
    "SIM_HIL_ALLOCATION_MODES_DEVELOPMENT",
    "SIM_HIL_ACCEPTANCE_INTENT_KINDS_V1",
    "SIM_HIL_ACCEPTANCE_INTENT_SCHEMA_V1",
    "SIM_HIL_ACCEPTANCE_STAGES_V1",
    "SIM_HIL_ACCEPTANCE_STATUS_SCHEMA_V1",
    "SIM_HIL_CONTROL_MODES_V1",
    "SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1",
    "STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1",
    "STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1",
    "SimHILAcceptanceController",
    "SimHILAcceptanceIntent",
    "SimHILAcceptanceSubmissionReceipt",
    "SimHILAcceptanceTraceCapture",
    "StationControlTraceEpisodeSessionCapture",
    "StationControlTraceFinalizationReceipt",
    "sim_hil_acceptance_intent_from_json",
]
