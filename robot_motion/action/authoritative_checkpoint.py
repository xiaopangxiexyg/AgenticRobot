"""Complete logical core history for an unpublished, independent backend.

The caller fences the entire decision/admission/trace operation. Store locks
alone cannot establish that boundary. Backend, resolver, safety and trace-sink
state belong to the enclosing profile, which supplies fresh bindings through
the ordinary constructor. This internal bundle is not a portable file format
or an authorization to take over a live robot.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.runtime.execution_checkpoint import (
    ExecutionAuthorityCheckpoint,
    capture_execution_authority,
    restore_execution_authority,
)
from robot_motion.runtime.execution_gate import (
    EXECUTION_GATE_EVENT_HASH_DOMAIN_V1,
    ExecutionGateEvent,
    ExecutionGateStore,
)
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze

from .admission_checkpoint import (
    SourceAdmissionCheckpoint,
    capture_source_admission,
    restore_source_admission,
)
from .arbiter import ActionArbiter
from .authoritative_trace import (
    AuthoritativeTraceContext,
    project_authority_event,
    project_candidate_authorization_admission,
    project_execution_cancellation,
    project_execution_cancellation_ack,
    project_execution_gate_event,
    project_producer_epoch_activation,
    project_proposal_admission,
)
from .authority import AUTHORITY_EVENT_HASH_DOMAIN_V1, AuthorityEvent, AuthorityStore
from .candidate_authorization import (
    CANDIDATE_AUTHORIZATION_HASH_DOMAIN_V1,
    CandidateAuthorizationAdmissionReceipt,
    CandidateAuthorizationDraft,
    CandidateAuthorizationRecord,
    CandidateAuthorizationStore,
)
from .decision_checkpoint import (
    DecisionCheckpoint,
    capture_decision_owners,
    restore_decision_owners,
)
from .persistence import ActiveExecutionSegment
from .proposal import ProposalAdmissionReceipt

_BINDINGS = {
    "config",
    "dispatcher",
    "contextual_dispatcher",
    "backend_hold",
    "backend_name",
    "resolver",
    "shared_safety",
    "action_representation_adapter",
    "target_execution_port",
    "trace_sink",
    "clock",
    "trace_context",
}
_OWNED = {
    "coordinator",
    "producer_epochs",
    "proposals",
    "_candidate_authorizations",
    "authority_store",
    "execution_gate_store",
    "scheduler",
    "arbiter",
    "arbitration_runtime",
    "dispatch_owners",
    "dispatch_owner_token",
    "execution_epochs",
    "cancellation_acknowledgements",
    "_authority_admission_lock",
    "_trace_fact_lock",
    "_facts",
    "_fact_hash_by_id",
    "_compound_authority_profile_by_producer",
}
_STATE = {
    "_active_segment_by_group",
    "_pending_segment_by_group",
    "_backend_hold_active",
    "_next_authority_sequence",
    "_next_execution_gate_sequence",
    "_previous_decision_id",
    "_previous_selection_by_group",
    "_terminal",
    "_compound_authority_profiles_recorded",
}
_STORE_LAYOUTS = (
    {
        "runtime_session_id",
        "control_domain_id",
        "coordinator",
        "policy",
        "capacity",
        "initial_mode",
        "_records",
        "_last_sequence",
        "_lock",
    },
    {
        "runtime_session_id",
        "control_domain_id",
        "coordinator",
        "capacity",
        "initial_status",
        "_records",
        "_last_sequence",
        "_lock",
    },
    {
        "runtime_session_id",
        "control_domain_id",
        "coordinator",
        "proposals",
        "capacity",
        "_records",
        "_by_proposal_id",
        "_lock",
    },
)


@dataclass(frozen=True)
class AuthoritativeControlCheckpoint:
    configuration_hash: str
    backend_name: str
    source: SourceAdmissionCheckpoint
    execution: ExecutionAuthorityCheckpoint
    decision: DecisionCheckpoint
    authority_records: tuple[AuthorityEvent, ...]
    gate_records: tuple[ExecutionGateEvent, ...]
    authorization_records: tuple[CandidateAuthorizationRecord, ...]
    store_settings: tuple
    state: dict
    facts: tuple[ControlTraceFact, ...]


def _configuration(config):
    def value(item):
        if hasattr(item, "semantic_content_hash"):
            return {
                "type": type(item).__module__ + "." + type(item).__qualname__,
                "hash": item.semantic_content_hash,
            }
        if is_dataclass(item):
            return {
                field.name: value(getattr(item, field.name)) for field in fields(item)
            }
        if isinstance(item, Mapping):
            return {key: value(part) for key, part in item.items()}
        if isinstance(item, tuple):
            return tuple(value(part) for part in item)
        return item

    return canonical_content_hash(
        value(config), domain="control-checkpoint-configuration.v1"
    )


def _counter(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid core checkpoint counter.")
    return value


def _facts(records, config, now):
    result, by_id = [], {}
    for record in records:
        if (
            type(record) is not ControlTraceFact
            or record.runtime_session_id != config.runtime_session_id
            or record.control_domain_id != config.control_domain_id
            or record.clock_domain_id != config.clock_domain.clock_domain_id
            or record.embodiment_instance_id != config.embodiment_instance_id
            or record.occurred_at > now
        ):
            raise ValueError("Trace fact scope/time differs from the core checkpoint.")
        copied = ControlTraceFact.create(
            **{
                name: getattr(record, name)
                for name in (
                    "fact_id",
                    "event_type",
                    "runtime_session_id",
                    "control_domain_id",
                    "occurred_at",
                    "clock_domain_id",
                    "payload",
                    "embodiment_instance_id",
                )
            }
        )
        if copied.semantic_content_hash != record.semantic_content_hash:
            raise ValueError("Trace content hash differs.")
        if (
            record.fact_id in by_id
            and by_id[record.fact_id] != record.semantic_content_hash
        ):
            raise ValueError("Conflicting retained trace facts.")
        by_id[record.fact_id] = record.semantic_content_hash
        result.append(copied)
    return tuple(result), by_id


def _match_history(state, config, facts_by_id, authority, gates, authorizations):
    context = AuthoritativeTraceContext(
        config.runtime_session_id,
        config.control_domain_id,
        config.embodiment_instance_id,
        config.clock_domain.clock_domain_id,
    )

    def match(expected):
        actual = facts_by_id.get(expected.fact_id)
        if (
            actual is None
            or actual.semantic_content_hash != expected.semantic_content_hash
        ):
            raise ValueError("Core owner and trace history differ.")

    for record in state.source.producer_records:
        match(project_producer_epoch_activation(context, record))
    for record in state.source.proposal_records:
        # Only the immutable proposal projection is compared; the original
        # admission event retains its actual source ID and is not replayed.
        receipt = ProposalAdmissionReceipt(accepted=True, proposal=record)
        match(
            project_proposal_admission(
                context,
                receipt,
                source_event_id="checkpoint-projection",
                occurred_at=record.available_at,
            )[1]
        )
    for record in authority:
        match(project_authority_event(context, record))
    for record in gates:
        match(project_execution_gate_event(context, record))
    for record in authorizations:
        draft = CandidateAuthorizationDraft(
            **{
                f.name: getattr(record, f.name)
                for f in fields(CandidateAuthorizationDraft)
            }
        )
        receipt = CandidateAuthorizationAdmissionReceipt(
            accepted=True,
            authorization_id=record.authorization_id,
            available_at=record.available_at,
            available_clock_domain_id=record.available_clock_domain_id,
            control_admission_sequence=record.control_admission_sequence,
            semantic_content_hash=record.semantic_content_hash,
            record=record,
        )
        match(
            project_candidate_authorization_admission(
                context, draft, receipt, occurred_at=record.available_at
            )
        )
    for record in state.execution.cancellations:
        match(project_execution_cancellation(context, record))
    for record in state.execution.acknowledgements:
        match(project_execution_cancellation_ack(context, record))


def _execution_history(facts):
    """Verify saved state against applied facts, never re-execute a decision."""
    seen, active, selection = {}, {}, {}
    previous, hold = None, False
    pending_cancel, accepted_levels = None, set()
    for fact in facts:
        p = fact.payload
        if fact.event_type == "runtime_trace_terminal":
            raise ValueError("Terminal trace history cannot resume.")
        if fact.event_type == "execution_cancellation":
            if p["effective_scope"] == "control_domain":
                active.clear()
            else:
                for group in p["selection_group_ids"]:
                    active.pop(group, None)
            pending_cancel, accepted_levels = p, set()
            hold = not p["required_ack_levels"]
        elif fact.event_type == "execution_cancellation_ack":
            if (
                pending_cancel is None
                or p["cancellation_id"] != pending_cancel["cancellation_id"]
            ):
                raise ValueError("Cancellation ACK history has no current transaction.")
            if (
                p["accepted"]
                and p["acknowledged_at"]
                <= pending_cancel["acknowledgement_deadline_at"]
            ):
                accepted_levels.add(p["acknowledgement_level"])
            else:
                accepted_levels.discard(p["acknowledgement_level"])
            hold = set(pending_cancel["required_ack_levels"]).issubset(accepted_levels)
        elif fact.event_type == "dispatch_receipt" and p["all_accepted"]:
            command = seen.get(p["command_id"])
            if command is None or command.event_type != "command":
                raise ValueError("Dispatch history lacks its preceding command.")
            decision = seen.get(command.payload["decision_id"])
            directive = seen.get(command.payload["directive_id"])
            if (
                decision is None
                or decision.event_type != "arbitration_decision"
                or directive is None
                or directive.event_type != "execution_directive"
            ):
                raise ValueError(
                    "Dispatch history lacks its preceding decision/directive."
                )
            if directive.payload["decision_id"] != decision.fact_id:
                raise ValueError("Dispatch decision/directive lineage differs.")
            previous = decision.fact_id
            selection = dict(decision.payload["selected_by_group"])
            for group, item in directive.payload[
                "directive_by_selection_group"
            ].items():
                if item["directive"] == "publish_selected":
                    active[group] = ActiveExecutionSegment(
                        group,
                        f"segment:{previous}:{group}",
                        item["selected_action_ref"],
                        item["segment_expiry"],
                        item["group_segment_generation"],
                    )
            hold = False
            pending_cancel = None
        seen[fact.fact_id] = fact
    return active, previous, selection, hold


def _records(records, kind, domain, source, *, excluded=()):
    result = []
    previous = 0
    for record in records:
        if type(record) is not kind:
            raise ValueError("Unsupported core store record.")
        sequence = _counter(record.control_admission_sequence, 1)
        if (
            sequence <= previous
            or sequence > source.committed_high_watermark
            or record.runtime_session_id != source.clock_domain.runtime_session_id
            or record.control_domain_id != source.control_domain_id
            or record.available_clock_domain_id != source.clock_domain.clock_domain_id
            or type(record.available_at) is not float
            or not math.isfinite(record.available_at)
            or record.available_at > source.captured_at
        ):
            raise ValueError("Core record admission scope/order differs.")
        payload = {
            field.name: getattr(record, field.name)
            for field in fields(record)
            if field.name not in ("semantic_content_hash", *excluded)
        }
        if (
            canonical_content_hash(payload, domain=domain)
            != record.semantic_content_hash
        ):
            raise ValueError("Core store content hash differs.")
        result.append(
            replace(
                record,
                **(
                    {"audit_metadata": deep_freeze(record.audit_metadata)}
                    if hasattr(record, "audit_metadata")
                    else {}
                ),
            )
        )
        previous = sequence
    return tuple(result)


def _validate(state, config, clock):
    if type(state) is not AuthoritativeControlCheckpoint:
        raise ValueError("Unsupported core checkpoint.")
    if state.configuration_hash != _configuration(config):
        raise ValueError("Core checkpoint configuration differs.")
    if type(state.state) is not dict or set(state.state) != _STATE:
        raise ValueError("Core state layout differs.")
    if (
        config.defer_selected_segment_activation
        or state.state["_pending_segment_by_group"]
    ):
        raise ValueError("This checkpoint profile requires immediate activation.")
    source = restore_source_admission(
        state.source,
        control_domain_id=config.control_domain_id,
        clock_domain=config.clock_domain,
        proposal_capacity=config.proposal_capacity,
        clock=clock,
    )
    execution = restore_execution_authority(
        state.execution,
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        clock_domain_id=config.clock_domain.clock_domain_id,
        owner_runtime_id=config.owner_runtime_id,
        owner_kind="shared_arbitration_runtime",
        clock=clock,
    )
    decision = restore_decision_owners(
        state.decision, spec=config.decision_schedule, clock=clock
    )
    if (
        not state.source.captured_at
        == state.execution.captured_at
        == state.decision.captured_at
    ):
        raise ValueError("Core checkpoint components belong to different cuts.")
    facts, fact_hashes = _facts(state.facts, config, state.source.captured_at)
    facts_by_id = {fact.fact_id: fact for fact in facts}
    authority = _records(
        state.authority_records,
        AuthorityEvent,
        AUTHORITY_EVENT_HASH_DOMAIN_V1,
        state.source,
        excluded=("audit_metadata",),
    )
    gates = _records(
        state.gate_records,
        ExecutionGateEvent,
        EXECUTION_GATE_EVENT_HASH_DOMAIN_V1,
        state.source,
        excluded=("audit_metadata",),
    )
    authorizations = _records(
        state.authorization_records,
        CandidateAuthorizationRecord,
        CANDIDATE_AUTHORIZATION_HASH_DOMAIN_V1,
        state.source,
        excluded=("authorization_id",),
    )
    _match_history(state, config, facts_by_id, authority, gates, authorizations)
    if type(state.store_settings) is not tuple or len(state.store_settings) != 5:
        raise ValueError("Core store settings layout differs.")
    a_capacity, a_last, g_capacity, g_last, g_initial = state.store_settings
    for capacity, last, records, epoch, producer, namespace in (
        (
            a_capacity,
            a_last,
            authority,
            "authority_epoch",
            "authority-controller",
            "authority_event",
        ),
        (
            g_capacity,
            g_last,
            gates,
            "gate_epoch",
            "execution-gate",
            "execution_gate_event",
        ),
    ):
        _counter(capacity, 1)
        _counter(last, -1)
        if len(records) > capacity or last != (records[-1].sequence if records else -1):
            raise ValueError("Core store sequence watermark differs.")
        previous = -1
        for index, record in enumerate(records, 1):
            sequence = _counter(record.sequence)
            if (
                sequence <= previous
                or type(getattr(record, epoch)) is not int
                or getattr(record, epoch) != index
            ):
                raise ValueError("Core store epoch/sequence history differs.")
            expected = RuntimeFactIdentity(
                config.runtime_session_id,
                config.control_domain_id,
                producer,
                0,
                sequence,
                namespace,
            ).fact_id
            if expected != record.event_id:
                raise ValueError("Core store record identity differs.")
            previous = sequence
    if len(authorizations) > config.candidate_authorization_capacity or len(
        {record.proposal_id for record in authorizations}
    ) != len(authorizations):
        raise ValueError("Authorization history is duplicated or over capacity.")
    if any(
        record.authorization_id
        != "candidate-authorization:" + record.semantic_content_hash
        for record in authorizations
    ):
        raise ValueError("Authorization identity differs.")
    all_records = (
        *state.source.producer_records,
        *state.source.proposal_records,
        *authority,
        *gates,
        *authorizations,
    )
    admissions = [record.control_admission_sequence for record in all_records]
    if len(admissions) != len(set(admissions)):
        raise ValueError("Core owners reuse an admission sequence.")
    own = state.state
    for name in (
        "_terminal",
        "_backend_hold_active",
        "_compound_authority_profiles_recorded",
    ):
        if type(own[name]) is not bool:
            raise ValueError("Core lifecycle flag is not boolean.")
    if own["_terminal"]:
        raise ValueError("A terminal core cannot be resumed.")
    expected_active, expected_previous, expected_selection, expected_hold = (
        _execution_history(facts)
    )
    if (
        own["_active_segment_by_group"] != expected_active
        or own["_previous_decision_id"] != expected_previous
        or own["_previous_selection_by_group"] != expected_selection
        or own["_backend_hold_active"] != expected_hold
    ):
        raise ValueError("Core state differs from latest applied execution history.")
    for name, last in (
        ("_next_authority_sequence", a_last),
        ("_next_execution_gate_sequence", g_last),
    ):
        if _counter(own[name]) != last + 1:
            raise ValueError("Core next sequence differs from its store.")
    if execution.dispatch_owners.snapshot().current_authority_epoch != (
        authority[-1].authority_epoch if authority else config.initial_authority_epoch
    ):
        raise ValueError("Dispatch and authority history epochs differ.")
    groups = set(config.active_selection_groups)
    segments = own["_active_segment_by_group"]
    if type(segments) is not dict or set(segments) - groups:
        raise ValueError("Unknown active segment groups.")
    for group, segment in segments.items():
        if (
            type(segment) is not ActiveExecutionSegment
            or segment.selection_group_id != group
            or segment.selected_action_ref not in facts_by_id
            or type(segment.expires_at) is not float
            or not math.isfinite(segment.expires_at)
        ):
            raise ValueError("Active segment history differs.")
        _counter(segment.group_segment_generation, 1)
    selection = own["_previous_selection_by_group"]
    if type(selection) is not dict or set(selection) - groups:
        raise ValueError("Previous selection groups differ.")
    for record in authorizations:
        proposal = facts_by_id.get(record.proposal_id)
        if (
            proposal is None
            or proposal.event_type != "action_proposal"
            or proposal.payload["proposal_semantic_content_hash"]
            != record.proposal_content_hash
        ):
            raise ValueError("Authorization lacks its retained proposal fact.")
    authority_store = AuthorityStore(
        runtime_session_id=config.runtime_session_id,
        coordinator=source.coordinator,
        policy=config.authority_policy,
        initial_mode=config.initial_control_mode,
    )
    gate_store = ExecutionGateStore(
        runtime_session_id=config.runtime_session_id,
        coordinator=source.coordinator,
    )
    if (authority_store.capacity, gate_store.capacity, gate_store.initial_status) != (
        a_capacity,
        g_capacity,
        g_initial,
    ):
        raise ValueError("Core store settings differ from the destination defaults.")
    candidate_store = CandidateAuthorizationStore(
        runtime_session_id=config.runtime_session_id,
        coordinator=source.coordinator,
        proposals=source.proposals,
        capacity=config.candidate_authorization_capacity,
    )
    authority_store._records, authority_store._last_sequence = list(authority), a_last
    gate_store._records, gate_store._last_sequence = list(gates), g_last
    candidate_store._records = list(authorizations)
    candidate_store._by_proposal_id = {
        record.proposal_id: record for record in authorizations
    }
    detached = dict(own)
    detached["_active_segment_by_group"] = {
        key: replace(value) for key, value in segments.items()
    }
    detached["_pending_segment_by_group"] = {}
    detached["_previous_selection_by_group"] = dict(selection)
    return (
        source,
        execution,
        decision,
        authority_store,
        gate_store,
        candidate_store,
        detached,
        facts,
        fact_hashes,
    )


def capture_authoritative_control(core) -> AuthoritativeControlCheckpoint:
    """Capture a caller-fenced complete core; native/profile state is external."""
    from .authoritative_control import (
        AuthoritativeControlConfig,
        AuthoritativeControlRuntime,
    )

    if (
        type(core) is not AuthoritativeControlRuntime
        or set(vars(core)) != _BINDINGS | _OWNED | _STATE
    ):
        raise ValueError("Unsupported authoritative core layout.")
    if (
        type(core.config) is not AuthoritativeControlConfig
        or type(core.arbiter) is not ActionArbiter
        or vars(core.arbiter)
        or core.trace_context
        != AuthoritativeTraceContext(
            core.config.runtime_session_id,
            core.config.control_domain_id,
            core.config.embodiment_instance_id,
            core.config.clock_domain.clock_domain_id,
        )
        or core._compound_authority_profile_by_producer
        != {p.producer_id: p for p in core.config.compound_authority_profiles}
    ):
        raise ValueError("Core static owner configuration differs.")
    stores = (
        core.authority_store,
        core.execution_gate_store,
        core._candidate_authorizations,
    )
    for store, kind, layout in zip(
        stores,
        (AuthorityStore, ExecutionGateStore, CandidateAuthorizationStore),
        _STORE_LAYOUTS,
    ):
        if (
            type(store) is not kind
            or set(vars(store)) != layout
            or store.coordinator is not core.coordinator
        ):
            raise ValueError("Core store owner layout or binding differs.")
    a, g, c = stores
    if (
        a.policy != core.config.authority_policy
        or a.initial_mode != core.config.initial_control_mode
        or c.proposals is not core.proposals
        or c.capacity != core.config.candidate_authorization_capacity
        or c._by_proposal_id != {r.proposal_id: r for r in c._records}
    ):
        raise ValueError("Core store configuration/index differs.")
    # No callback is called here. The enclosing owner already excludes full
    # operations, including reentrant captures from trace/dispatch callbacks.
    with (
        core._authority_admission_lock,
        core._trace_fact_lock,
        core.coordinator._lock,
        a._lock,
        g._lock,
        c._lock,
    ):
        state = AuthoritativeControlCheckpoint(
            _configuration(core.config),
            core.backend_name,
            capture_source_admission(
                core.coordinator, core.producer_epochs, core.proposals
            ),
            capture_execution_authority(
                core.dispatch_owners,
                core.execution_epochs,
                core.cancellation_acknowledgements,
            ),
            capture_decision_owners(core.scheduler, core.arbitration_runtime),
            tuple(a._records),
            tuple(g._records),
            tuple(c._records),
            (
                a.capacity,
                a._last_sequence,
                g.capacity,
                g._last_sequence,
                g.initial_status,
            ),
            {name: getattr(core, name) for name in _STATE},
            tuple(core._facts),
        )
        _source, _execution, _decision, a, g, c, own, facts, index = _validate(
            state, core.config, core.clock
        )
        if (
            index != core._fact_hash_by_id
            or core.dispatch_owner_token
            != core.dispatch_owners.snapshot().current_token
        ):
            raise ValueError("Core token or retained trace index differs.")
        return replace(
            state,
            authority_records=tuple(a._records),
            gate_records=tuple(g._records),
            authorization_records=tuple(c._records),
            state=own,
            facts=facts,
        )


def restore_control_owners(core, state):
    """Finish ordinary constructor dependency checks without creating new history."""
    if set(vars(core)) != _BINDINGS:
        raise ValueError("Core restore requires an unpublished constructor boundary.")
    if (
        type(state) is not AuthoritativeControlCheckpoint
        or state.backend_name != core.backend_name
    ):
        raise ValueError("Checkpoint backend binding differs.")
    source, execution, decision, a, g, c, own, facts, index = _validate(
        state, core.config, core.clock
    )
    core.coordinator, core.producer_epochs, core.proposals = (
        source.coordinator,
        source.producer_epochs,
        source.proposals,
    )
    core.authority_store, core.execution_gate_store, core._candidate_authorizations = (
        a,
        g,
        c,
    )
    core.scheduler, core.arbitration_runtime = decision.scheduler, decision.publication
    core.arbiter = ActionArbiter()
    core.dispatch_owners = execution.dispatch_owners
    core.dispatch_owner_token = execution.dispatch_owners.snapshot().current_token
    core.execution_epochs = execution.execution_epochs
    core.cancellation_acknowledgements = execution.cancellation_acknowledgements
    core._authority_admission_lock = threading.RLock()
    core._trace_fact_lock = threading.RLock()
    core._facts, core._fact_hash_by_id = list(facts), index
    core._compound_authority_profile_by_producer = {
        profile.producer_id: profile
        for profile in core.config.compound_authority_profiles
    }
    for name, value in own.items():
        setattr(core, name, value)
