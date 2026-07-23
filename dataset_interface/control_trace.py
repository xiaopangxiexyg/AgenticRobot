"""Read committed v2.4 ControlTrace facts and build typed logical views."""

from __future__ import annotations

import mmap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash, canonical_loads
from dataset_contracts.control_trace import PersistedRecordEnvelope
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_EPISODE_RELATIVE_ROOT_V1,
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
    EPISODE_COMMIT_FILE_DOMAIN_V1,
    EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1,
    EPISODE_RANGE_RECORD_HASH_DOMAIN_V1,
    EPISODE_SEAL_FILE_DOMAIN_V1,
    EPISODE_TRACE_RANGE_HASH_DOMAIN_V1,
    EPISODE_TRACE_SEAL_HASH_DOMAIN_V1,
    EPISODE_TRACE_SEAL_SCHEMA_V1,
    V24_CAPTURE_COMMIT_HASH_DOMAIN_V1,
    V24_CAPTURE_COMMIT_SCHEMA_V1,
)
from dataset_contracts.control_trace_wal import scan_control_trace_wal


SHADOW_EVALUATION_VIEW_SCHEMA_V1 = "dexteleop_shadow_evaluation_view_v1"
SHADOW_EVALUATION_VIEW_HASH_DOMAIN_V1 = "dexteleop.shadow_evaluation_view.v1"
SHADOW_EVALUATION_MATERIALIZATION_SPEC_V1 = {
    "schema": SHADOW_EVALUATION_VIEW_SCHEMA_V1,
    "source_event": "arbitration_evaluation",
    "action_stage": "would_select_pre_safety",
    "authoritative": False,
    "legacy_execution_join": "evaluation_id",
}
SHADOW_EVALUATION_MATERIALIZATION_SPEC_HASH_V1 = canonical_content_hash(
    SHADOW_EVALUATION_MATERIALIZATION_SPEC_V1,
    domain="dexteleop.shadow_evaluation_materialization_spec.v1",
)

ACTUAL_DECISION_VIEW_SCHEMA_V1 = "dexteleop_actual_decision_view_v1"
ACTUAL_DECISION_VIEW_HASH_DOMAIN_V1 = "dexteleop.actual_decision_view.v1"
ACTUAL_DECISION_MATERIALIZATION_SPEC_V1 = {
    "schema": ACTUAL_DECISION_VIEW_SCHEMA_V1,
    "source_event": "arbitration_decision",
    "action_stage": "selected_pre_safety",
    "authoritative": True,
    "timing": "variable_dt",
    "resampling": "none",
}
ACTUAL_DECISION_MATERIALIZATION_SPEC_HASH_V1 = canonical_content_hash(
    ACTUAL_DECISION_MATERIALIZATION_SPEC_V1,
    domain="dexteleop.actual_decision_materialization_spec.v1",
)


class ControlTraceReadError(ValueError):
    """The capture cannot support the requested logical read."""


def _safe_path_segment(value: object, *, field_name: str) -> str:
    segment = str(value).strip()
    if (
        not segment
        or "/" in segment
        or "\\" in segment
        or segment in {".", ".."}
    ):
        raise ControlTraceReadError(f"{field_name} is not safe for dataset lookup.")
    return segment


def _read_payload(path: Path, *, domain: str) -> dict[str, Any]:
    if not path.is_file():
        raise ControlTraceReadError(f"Required ControlTrace record is missing: {path}")
    value = canonical_loads(path.read_bytes(), expected_domain=domain)
    if not isinstance(value, dict):
        raise ControlTraceReadError(f"ControlTrace record is not a mapping: {path}")
    return value


def _verify_semantic_hash(
    payload: Mapping[str, Any],
    *,
    domain: str,
    label: str,
) -> None:
    recorded = str(payload.get("semantic_content_hash", ""))
    semantic = dict(payload)
    semantic.pop("semantic_content_hash", None)
    expected = canonical_content_hash(semantic, domain=domain)
    if recorded != expected:
        raise ControlTraceReadError(f"{label} semantic hash mismatch.")


@dataclass(frozen=True)
class EpisodeControlTrace:
    dataset_root: Path
    episode_uuid: str
    trace_session_id: str
    profile_id: str
    committed: bool
    capabilities: tuple[str, ...]
    seal_payload: Mapping[str, Any]
    commit_payload: Mapping[str, Any] | None
    episode_records: tuple[PersistedRecordEnvelope, ...]
    dependency_records: tuple[PersistedRecordEnvelope, ...]
    wal_relative_path: str
    wal_tail_status: str
    capability_artifact_hash_by_name: Mapping[str, str] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ShadowEvaluationRow:
    episode_uuid: str
    trace_session_id: str
    evaluation_id: str
    evaluation_journal_sequence: int
    evaluated_at: float
    clock_domain_id: str
    arbitration_input_snapshot_id: str
    arbitration_input_hash: str
    proposal_snapshot_id: str
    output_status: str
    would_select_action: Any
    would_select_action_hash: str | None
    would_select_by_component: Mapping[str, Any]
    would_select_by_group: Mapping[str, Any]
    legacy_execution_event_ids: tuple[str, ...]
    legacy_execution_outcomes: tuple[str, ...]
    binding_assessment_ids: tuple[str, ...]
    binding_verdicts: tuple[str, ...]
    conditional_evaluation: bool
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=SHADOW_EVALUATION_VIEW_SCHEMA_V1, init=False)
    materialization_spec_hash: str = field(
        default=SHADOW_EVALUATION_MATERIALIZATION_SPEC_HASH_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=SHADOW_EVALUATION_VIEW_HASH_DOMAIN_V1),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "materialization_spec_hash": self.materialization_spec_hash,
            "episode_uuid": self.episode_uuid,
            "trace_session_id": self.trace_session_id,
            "evaluation_id": self.evaluation_id,
            "evaluation_journal_sequence": self.evaluation_journal_sequence,
            "evaluated_at": self.evaluated_at,
            "clock_domain_id": self.clock_domain_id,
            "arbitration_input_snapshot_id": self.arbitration_input_snapshot_id,
            "arbitration_input_hash": self.arbitration_input_hash,
            "proposal_snapshot_id": self.proposal_snapshot_id,
            "output_status": self.output_status,
            "would_select_action": self.would_select_action,
            "would_select_action_hash": self.would_select_action_hash,
            "would_select_by_component": self.would_select_by_component,
            "would_select_by_group": self.would_select_by_group,
            "legacy_execution_event_ids": self.legacy_execution_event_ids,
            "legacy_execution_outcomes": self.legacy_execution_outcomes,
            "binding_assessment_ids": self.binding_assessment_ids,
            "binding_verdicts": self.binding_verdicts,
            "conditional_evaluation": self.conditional_evaluation,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class ActualDecisionRow:
    episode_uuid: str
    trace_session_id: str
    decision_index: int
    decision_id: str
    decision_journal_sequence: int
    decision_cutoff_at: float
    decided_at: float
    published_at: float
    effective_interval_start: float
    effective_interval_end: float | None
    dt_sec: float | None
    selected_action: Any
    selected_action_hash: str
    selected_by_group: Mapping[str, Any]
    execution_directive_id: str
    execution_status: str
    safe_action: Any
    safe_action_hash: str | None
    safety_status: str
    command_id: str | None
    command: Any
    dispatch_receipt_ids: tuple[str, ...]
    measurement_alignment_ids: tuple[str, ...]
    source_decision_ids: tuple[str, ...]
    action_stage: str = field(default="selected_pre_safety", init=False)
    is_resampled: bool = field(default=False, init=False)
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=ACTUAL_DECISION_VIEW_SCHEMA_V1, init=False)
    materialization_spec_hash: str = field(
        default=ACTUAL_DECISION_MATERIALIZATION_SPEC_HASH_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_by_group", dict(self.selected_by_group))
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=ACTUAL_DECISION_VIEW_HASH_DOMAIN_V1),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "materialization_spec_hash": self.materialization_spec_hash,
            "episode_uuid": self.episode_uuid,
            "trace_session_id": self.trace_session_id,
            "decision_index": self.decision_index,
            "decision_id": self.decision_id,
            "decision_journal_sequence": self.decision_journal_sequence,
            "decision_cutoff_at": self.decision_cutoff_at,
            "decided_at": self.decided_at,
            "published_at": self.published_at,
            "effective_interval_start": self.effective_interval_start,
            "effective_interval_end": self.effective_interval_end,
            "dt_sec": self.dt_sec,
            "selected_action": self.selected_action,
            "selected_action_hash": self.selected_action_hash,
            "selected_by_group": self.selected_by_group,
            "execution_directive_id": self.execution_directive_id,
            "execution_status": self.execution_status,
            "safe_action": self.safe_action,
            "safe_action_hash": self.safe_action_hash,
            "safety_status": self.safety_status,
            "command_id": self.command_id,
            "command": self.command,
            "dispatch_receipt_ids": self.dispatch_receipt_ids,
            "measurement_alignment_ids": self.measurement_alignment_ids,
            "source_decision_ids": self.source_decision_ids,
            "action_stage": self.action_stage,
            "is_resampled": self.is_resampled,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


def load_episode_control_trace(
    dataset_root: str | Path,
    episode_uuid: str,
    *,
    require_commit: bool = True,
) -> EpisodeControlTrace:
    """Load one sealed episode range from the canonical session WAL."""

    root = Path(dataset_root)
    episode = _safe_path_segment(episode_uuid, field_name="episode_uuid")
    episode_dir = root / CONTROL_TRACE_EPISODE_RELATIVE_ROOT_V1 / episode
    seal = _read_payload(
        episode_dir / "seal_assessment.cjson",
        domain=EPISODE_SEAL_FILE_DOMAIN_V1,
    )
    if seal.get("schema") != EPISODE_TRACE_SEAL_SCHEMA_V1:
        raise ControlTraceReadError("Unsupported episode seal schema.")
    _verify_semantic_hash(
        seal,
        domain=EPISODE_TRACE_SEAL_HASH_DOMAIN_V1,
        label="episode seal",
    )
    if not bool(seal.get("trace_sealed")):
        raise ControlTraceReadError("Episode ControlTrace is not sealed.")
    if seal.get("reason_codes") not in ((), []):
        raise ControlTraceReadError("Sealed episode unexpectedly contains failure reasons.")

    trace_range = seal.get("trace_range")
    if not isinstance(trace_range, Mapping):
        raise ControlTraceReadError("Episode seal has no valid trace range.")
    _verify_semantic_hash(
        trace_range,
        domain=EPISODE_TRACE_RANGE_HASH_DOMAIN_V1,
        label="episode trace range",
    )
    if trace_range.get("episode_uuid") != episode:
        raise ControlTraceReadError("Episode seal path and trace range disagree.")
    trace_session_id = _safe_path_segment(
        trace_range.get("trace_session_id", ""),
        field_name="trace_session_id",
    )

    commit_path = episode_dir / "capture_commit.cjson"
    commit = (
        _read_payload(commit_path, domain=EPISODE_COMMIT_FILE_DOMAIN_V1)
        if commit_path.is_file()
        else None
    )
    capability_artifact_hash_by_name: dict[str, str] = {}
    if require_commit and commit is None:
        raise ControlTraceReadError("Episode is sealed but has no capture commit.")
    if commit is not None:
        if commit.get("schema") != V24_CAPTURE_COMMIT_SCHEMA_V1:
            raise ControlTraceReadError("Unsupported capture commit schema.")
        _verify_semantic_hash(
            commit,
            domain=V24_CAPTURE_COMMIT_HASH_DOMAIN_V1,
            label="capture commit",
        )
        if commit.get("episode_uuid") != episode:
            raise ControlTraceReadError("Capture commit and episode path disagree.")
        if commit.get("legacy_finalize_status") != "finalized":
            raise ControlTraceReadError("Capture commit does not attest legacy finalize.")
        if commit.get("trace_seal_hash") != seal.get("semantic_content_hash"):
            raise ControlTraceReadError("Capture commit does not reference this seal.")
        if commit.get("control_trace_seal_status") != "sealed":
            raise ControlTraceReadError("Capture commit does not attest a sealed trace.")
        if not bool(seal.get("capture_commit_eligible")):
            raise ControlTraceReadError("Capture commit references an ineligible seal.")
        if "control_trace_core" not in tuple(commit.get("capabilities", ())):
            raise ControlTraceReadError("Capture commit lacks control_trace_core capability.")
        raw_artifacts = commit.get("capability_artifact_hash_by_name", {})
        if not isinstance(raw_artifacts, Mapping):
            raise ControlTraceReadError(
                "Capture commit capability artifact hashes are invalid."
            )
        capability_artifact_hash_by_name = {
            str(name): str(value) for name, value in raw_artifacts.items()
        }
        unsupported_artifacts = set(capability_artifact_hash_by_name) - {
            "native_streams"
        }
        if unsupported_artifacts:
            raise ControlTraceReadError(
                "Capture commit contains unsupported capability artifacts: "
                f"{sorted(unsupported_artifacts)}."
            )
        if "native_streams" in tuple(commit.get("capabilities", ())):
            expected_native_hash = capability_artifact_hash_by_name.get(
                "native_streams"
            )
            if expected_native_hash is None:
                raise ControlTraceReadError(
                    "native_streams capability has no episode manifest hash."
                )
            from .streams import load_native_stream_episode

            try:
                native_episode = load_native_stream_episode(
                    root,
                    episode,
                    load_payloads=False,
                )
            except ValueError as exc:
                raise ControlTraceReadError(
                    f"native_streams capability is invalid: {exc}"
                ) from exc
            if (
                native_episode.manifest.semantic_content_hash
                != expected_native_hash
            ):
                raise ControlTraceReadError(
                    "native_streams capability manifest hash mismatch."
                )

    wal_path = (
        root
        / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
        / trace_session_id
        / "control_trace.wal"
    )
    if not wal_path.is_file():
        raise ControlTraceReadError(f"ControlTrace session WAL is missing: {wal_path}")
    if wal_path.stat().st_size <= 0:
        raise ControlTraceReadError("ControlTrace session WAL is empty.")
    with wal_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            scan = scan_control_trace_wal(mapped)
    envelopes = {frame.envelope.journal_sequence: frame.envelope for frame in scan.frames}

    first = int(trace_range["first_journal_sequence"])
    last = int(trace_range["last_journal_sequence"])
    expected_sequences = tuple(range(first, last + 1))
    missing = tuple(sequence for sequence in expected_sequences if sequence not in envelopes)
    if missing:
        raise ControlTraceReadError(f"Episode WAL range is incomplete: missing={missing!r}.")
    episode_records = tuple(envelopes[sequence] for sequence in expected_sequences)
    if any(record.trace_session_id != trace_session_id for record in episode_records):
        raise ControlTraceReadError("Episode WAL contains a foreign trace session.")
    lifecycle_owner_id = str(trace_range.get("lifecycle_owner_id", ""))
    binding_event_id = str(trace_range.get("binding_event_id", ""))
    if any(record.episode_uuid != episode for record in episode_records):
        raise ControlTraceReadError("Episode WAL membership does not match the seal.")
    if any(
        record.episode_membership is None
        or record.episode_membership.lifecycle_owner_id != lifecycle_owner_id
        or record.episode_membership.binding_event_id != binding_event_id
        for record in episode_records
    ):
        raise ControlTraceReadError("Episode WAL lifecycle membership does not match the seal.")

    range_record_payload = tuple(
        {
            "journal_sequence": record.journal_sequence,
            "record_integrity_hash": record.record_integrity_hash,
        }
        for record in episode_records
    )
    if canonical_content_hash(
        range_record_payload,
        domain=EPISODE_RANGE_RECORD_HASH_DOMAIN_V1,
    ) != seal.get("range_record_hash"):
        raise ControlTraceReadError("Episode WAL range hash does not match the seal.")

    dependency_refs = tuple(seal.get("dependency_refs", ()))
    dependency_records: list[PersistedRecordEnvelope] = []
    for raw_ref in dependency_refs:
        if not isinstance(raw_ref, Mapping):
            raise ControlTraceReadError("Episode dependency ref is invalid.")
        sequence = int(raw_ref["journal_sequence"])
        record = envelopes.get(sequence)
        if record is None:
            raise ControlTraceReadError(f"Episode dependency {sequence} is missing.")
        if record.trace_session_id != trace_session_id:
            raise ControlTraceReadError(f"Episode dependency {sequence} is foreign.")
        if record.fact.fact_id != raw_ref.get("fact_id"):
            raise ControlTraceReadError(f"Episode dependency {sequence} fact ID mismatch.")
        if record.record_integrity_hash != raw_ref.get("record_integrity_hash"):
            raise ControlTraceReadError(f"Episode dependency {sequence} hash mismatch.")
        dependency_records.append(record)
    dependency_payload = tuple(dict(item) for item in dependency_refs)
    if canonical_content_hash(
        dependency_payload,
        domain=EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1,
    ) != seal.get("dependency_bundle_hash"):
        raise ControlTraceReadError("Episode dependency bundle hash does not match the seal.")

    return EpisodeControlTrace(
        dataset_root=root,
        episode_uuid=episode,
        trace_session_id=trace_session_id,
        profile_id=str(seal.get("profile_id", "")),
        committed=commit is not None,
        capabilities=(() if commit is None else tuple(commit.get("capabilities", ()))),
        capability_artifact_hash_by_name=capability_artifact_hash_by_name,
        seal_payload=seal,
        commit_payload=commit,
        episode_records=episode_records,
        dependency_records=tuple(dependency_records),
        wal_relative_path=wal_path.relative_to(root).as_posix(),
        wal_tail_status=scan.tail_status,
    )


def materialize_shadow_evaluation_view(
    trace: EpisodeControlTrace,
) -> tuple[ShadowEvaluationRow, ...]:
    """Build a deterministic shadow view; never emits selected/action columns."""

    records = (*trace.dependency_records, *trace.episode_records)
    by_type: dict[str, list[PersistedRecordEnvelope]] = {}
    by_fact_id: dict[str, PersistedRecordEnvelope] = {}
    for record in records:
        by_type.setdefault(record.fact.event_type, []).append(record)
        by_fact_id[record.fact.fact_id] = record

    legacy_by_evaluation: dict[str, list[PersistedRecordEnvelope]] = {}
    for record in by_type.get("legacy_execution_observation", ()):
        evaluation_id = record.fact.payload.get("evaluation_id")
        if evaluation_id is not None:
            legacy_by_evaluation.setdefault(str(evaluation_id), []).append(record)

    proposal_by_id = {
        str(record.fact.payload["proposal_id"]): record
        for record in by_type.get("action_proposal", ())
    }
    snapshot_by_id = {
        str(record.fact.payload["proposal_snapshot_id"]): record
        for record in by_type.get("proposal_snapshot", ())
    }
    eligibility_by_id = {
        str(record.fact.payload["eligibility_id"]): record
        for record in by_type.get("candidate_binding_eligibility", ())
    }
    assessment_by_id = {
        str(record.fact.payload["assessment_id"]): record
        for record in by_type.get("replay_binding_assessment", ())
    }

    rows: list[ShadowEvaluationRow] = []
    for record in sorted(
        (
            record
            for record in trace.episode_records
            if record.fact.event_type == "arbitration_evaluation"
        ),
        key=lambda item: item.journal_sequence,
    ):
        payload = record.fact.payload
        if payload.get("non_authoritative") is not True:
            raise ControlTraceReadError("Shadow view encountered an authoritative evaluation.")
        evaluation_id = str(payload["evaluation_id"])
        legacy = sorted(
            legacy_by_evaluation.get(evaluation_id, ()),
            key=lambda item: item.journal_sequence,
        )

        assessment_ids: set[str] = set()
        verdicts: set[str] = set()
        conditional = False
        proposal_snapshot = snapshot_by_id.get(str(payload["proposal_snapshot_id"]))
        if proposal_snapshot is not None:
            snapshot_payload = proposal_snapshot.fact.payload
            proposal_ids = {
                str(item["proposal_id"])
                for item in (
                    *snapshot_payload.get("candidates", ()),
                    *snapshot_payload.get("dispositions", ()),
                )
            }
            for proposal_id in sorted(proposal_ids):
                proposal = proposal_by_id.get(proposal_id)
                if proposal is None:
                    continue
                representation_id = proposal.fact.payload.get("representation_trace_id")
                receipt = None if representation_id is None else by_fact_id.get(str(representation_id))
                if receipt is None or receipt.fact.event_type != "replay_source_adapter_receipt":
                    continue
                eligibility = eligibility_by_id.get(
                    str(receipt.fact.payload["binding_eligibility_id"])
                )
                if eligibility is None:
                    continue
                conditional = conditional or bool(
                    eligibility.fact.payload["conditional_evaluation"]
                )
                assessment_id = str(eligibility.fact.payload["assessment_id"])
                assessment_ids.add(assessment_id)
                assessment = assessment_by_id.get(assessment_id)
                if assessment is not None:
                    verdicts.add(str(assessment.fact.payload["verdict"]))

        rows.append(
            ShadowEvaluationRow(
                episode_uuid=trace.episode_uuid,
                trace_session_id=trace.trace_session_id,
                evaluation_id=evaluation_id,
                evaluation_journal_sequence=record.journal_sequence,
                evaluated_at=float(payload["evaluated_at"]),
                clock_domain_id=record.fact.clock_domain_id,
                arbitration_input_snapshot_id=str(payload["arbitration_input_snapshot_id"]),
                arbitration_input_hash=str(payload["arbitration_input_hash"]),
                proposal_snapshot_id=str(payload["proposal_snapshot_id"]),
                output_status=str(payload["output_status"]),
                would_select_action=payload["would_select_action"],
                would_select_action_hash=payload["would_select_action_hash"],
                would_select_by_component=payload["would_select_by_component"],
                would_select_by_group=payload["would_select_by_group"],
                legacy_execution_event_ids=tuple(
                    str(item.fact.payload["event_id"]) for item in legacy
                ),
                legacy_execution_outcomes=tuple(
                    str(item.fact.payload["execution_mode"]) for item in legacy
                ),
                binding_assessment_ids=tuple(sorted(assessment_ids)),
                binding_verdicts=tuple(sorted(verdicts)),
                conditional_evaluation=conditional,
            )
        )
    return tuple(rows)


def materialize_actual_decision_view(
    trace: EpisodeControlTrace,
) -> tuple[ActualDecisionRow, ...]:
    """Build variable-dt rows from authoritative decisions only."""

    records = (*trace.dependency_records, *trace.episode_records)
    by_type: dict[str, list[PersistedRecordEnvelope]] = {}
    for record in records:
        by_type.setdefault(record.fact.event_type, []).append(record)

    def unique_by(
        key: str,
        event_type: str,
        *,
        skip_none: bool = False,
    ) -> dict[str, PersistedRecordEnvelope]:
        result: dict[str, PersistedRecordEnvelope] = {}
        for record in by_type.get(event_type, ()):
            raw_value = record.fact.payload[key]
            if skip_none and raw_value is None:
                continue
            value = str(raw_value)
            if value in result:
                raise ControlTraceReadError(
                    f"Actual decision lineage has duplicate {event_type} key {value!r}."
                )
            result[value] = record
        return result

    computations = unique_by("result_id", "arbitration_computation")
    attempts = unique_by("result_id", "arbitration_publication_attempt")
    directives = unique_by("decision_id", "execution_directive", skip_none=True)
    safety_by_decision = unique_by("decision_id", "shared_safety_result")
    command_by_decision = unique_by("decision_id", "command")
    receipts_by_command: dict[str, list[PersistedRecordEnvelope]] = {}
    for record in by_type.get("dispatch_receipt", ()):
        receipts_by_command.setdefault(
            str(record.fact.payload["command_id"]), []
        ).append(record)
    measurements_by_decision: dict[str, list[PersistedRecordEnvelope]] = {}
    for record in by_type.get("measurement_alignment", ()):
        measurements_by_decision.setdefault(
            str(record.fact.payload["decision_id"]), []
        ).append(record)

    decisions = sorted(
        (
            record
            for record in trace.episode_records
            if record.fact.event_type == "arbitration_decision"
        ),
        key=lambda item: (float(item.fact.payload["published_at"]), item.journal_sequence),
    )
    staged: list[dict[str, Any]] = []
    for record in decisions:
        decision = record.fact.payload
        decision_id = str(decision["decision_id"])
        result_id = str(decision["result_id"])
        computation = computations.get(result_id)
        attempt = attempts.get(result_id)
        directive = directives.get(decision_id)
        safety = safety_by_decision.get(decision_id)
        if computation is None or attempt is None or directive is None or safety is None:
            raise ControlTraceReadError(
                f"Authoritative decision {decision_id!r} has incomplete core lineage."
            )
        if not bool(attempt.fact.payload["accepted"]):
            raise ControlTraceReadError("Authoritative decision references a rejected publication.")
        command = command_by_decision.get(decision_id)
        command_id = None if command is None else str(command.fact.payload["command_id"])
        receipt_records = sorted(
            (() if command_id is None else receipts_by_command.get(command_id, ())),
            key=lambda item: item.journal_sequence,
        )
        measurement_records = sorted(
            measurements_by_decision.get(decision_id, ()),
            key=lambda item: item.journal_sequence,
        )
        expiries = tuple(
            float(item["segment_expiry"])
            for item in directive.fact.payload["directive_by_selection_group"].values()
            if item["segment_expiry"] is not None
        )
        staged.append(
            {
                "record": record,
                "decision": decision,
                "computation": computation.fact.payload,
                "attempt": attempt.fact.payload,
                "directive": directive.fact.payload,
                "safety": safety.fact.payload,
                "command": None if command is None else command.fact.payload,
                "receipts": receipt_records,
                "measurements": measurement_records,
                "segment_expiry": min(expiries) if expiries else None,
            }
        )

    rows: list[ActualDecisionRow] = []
    for index, item in enumerate(staged):
        decision = item["decision"]
        computation = item["computation"]
        attempt = item["attempt"]
        next_item = staged[index + 1] if index + 1 < len(staged) else None
        next_start = (
            None
            if next_item is None
            else float(next_item["computation"]["target_effective_at"])
        )
        expiry = item["segment_expiry"]
        effective_end = (
            next_start
            if expiry is None
            else expiry
            if next_start is None
            else min(expiry, next_start)
        )
        next_published = (
            None
            if next_item is None
            else float(next_item["decision"]["published_at"])
        )
        command_payload = item["command"]
        rows.append(
            ActualDecisionRow(
                episode_uuid=trace.episode_uuid,
                trace_session_id=trace.trace_session_id,
                decision_index=index,
                decision_id=str(decision["decision_id"]),
                decision_journal_sequence=item["record"].journal_sequence,
                decision_cutoff_at=float(computation["decision_cutoff_at"]),
                decided_at=float(attempt["computation_completed_at"]),
                published_at=float(decision["published_at"]),
                effective_interval_start=float(computation["target_effective_at"]),
                effective_interval_end=effective_end,
                dt_sec=(
                    None
                    if next_published is None
                    else next_published - float(decision["published_at"])
                ),
                selected_action=decision["selected_action"],
                selected_action_hash=str(decision["selected_action_hash"]),
                selected_by_group=decision["selected_by_group"],
                execution_directive_id=str(item["directive"]["directive_id"]),
                execution_status=str(item["directive"]["global_status"]),
                safe_action=item["safety"]["safe_action"],
                safe_action_hash=item["safety"]["safe_action_hash"],
                safety_status=str(item["safety"]["status"]),
                command_id=(
                    None if command_payload is None else str(command_payload["command_id"])
                ),
                command=(
                    None if command_payload is None else command_payload["commands"]
                ),
                dispatch_receipt_ids=tuple(
                    str(value.fact.payload["dispatch_receipt_id"])
                    for value in item["receipts"]
                ),
                measurement_alignment_ids=tuple(
                    str(value.fact.payload["measurement_alignment_id"])
                    for value in item["measurements"]
                ),
                source_decision_ids=(str(decision["decision_id"]),),
            )
        )
    return tuple(rows)


__all__ = [
    "ACTUAL_DECISION_MATERIALIZATION_SPEC_HASH_V1",
    "ACTUAL_DECISION_VIEW_SCHEMA_V1",
    "ActualDecisionRow",
    "ControlTraceReadError",
    "EpisodeControlTrace",
    "SHADOW_EVALUATION_MATERIALIZATION_SPEC_HASH_V1",
    "SHADOW_EVALUATION_VIEW_SCHEMA_V1",
    "ShadowEvaluationRow",
    "load_episode_control_trace",
    "materialize_actual_decision_view",
    "materialize_shadow_evaluation_view",
]
