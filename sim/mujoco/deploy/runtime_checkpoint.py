"""Runtime companion for the caller-fenced manual immediate target cut.

The parent restores native execution and pose resolution first. This component
retains the original logical core and runtime watermarks; it neither starts a
runtime nor republishes old admissions. External observers/composers require
their own contracts and are deliberately not accepted by this first profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from robot_motion.action.adaptation import (
    ActionAdaptationReceipt,
    ActionAdaptationRequest,
    ActionAdaptationResult,
    AdaptedActionCandidate,
)
from robot_motion.action.authoritative_checkpoint import (
    AuthoritativeControlCheckpoint,
    capture_authoritative_control,
)
from robot_motion.action.authoritative_trace import (
    project_component_execution_target,
    project_component_execution_target_admission,
    project_dispatch_receipts,
    project_target_execution_completion,
    project_target_execution_request,
    project_target_execution_submission,
)
from robot_motion.action.checkpoint_codec import (
    decode_checkpoint_value,
    encode_checkpoint_value,
)
from robot_motion.action.dispatch_context import (
    AuthoritativeTargetDispatchContext,
    build_authoritative_target_dispatch_context,
)
from robot_motion.action.execution_handoff import (
    ComponentExecutionTarget,
    ComponentExecutionTargetAdmissionReceipt,
    TargetExecutionCompletionReceipt,
    TargetExecutionDispatchResult,
    TargetExecutionRequest,
    TargetExecutionSubmissionReceipt,
    TargetProcessingFactReference,
)
from robot_motion.action.schema import RobotAction
from robot_motion.action.target_capability import TargetCapabilityAssessment
from robot_motion.runtime.dispatch import CommandDispatchReceipt
from sim.mujoco.adapters.checkpoint import ManualExecutionClock

from .observation import MujocoPolicyObservationAdapter
from .runtime import (
    IMMEDIATE_QREF_SIM_V1,
    MujocoDeployConfig,
    MujocoDeployRuntime,
    MujocoDirectComponentProposalSource,
    _LatestTargetExecution,
)

_RECORDS = (
    ActionAdaptationRequest,
    ActionAdaptationResult,
    ActionAdaptationReceipt,
    AdaptedActionCandidate,
    AuthoritativeTargetDispatchContext,
    ComponentExecutionTarget,
    ComponentExecutionTargetAdmissionReceipt,
    TargetExecutionRequest,
    TargetExecutionSubmissionReceipt,
    TargetExecutionCompletionReceipt,
    TargetExecutionDispatchResult,
    TargetProcessingFactReference,
    RobotAction,
    TargetCapabilityAssessment,
    CommandDispatchReceipt,
    _LatestTargetExecution,
    MujocoDeployConfig,
    MujocoDirectComponentProposalSource,
)
_STATE = {
    "_latest_target_execution_step",
    "_last_dispatch_context",
    "_projected_reference_fact_ids",
    "_reference_trace_cursor_by_stream",
    "_immediate_component_targets",
    "_immediate_component_target_admissions",
    "_next_immediate_target_admission_sequence",
    "_immediate_target_by_reference_sequence",
    "_next_direct_component_proposal_sequence",
}
_NONE = {
    "composition_runtime_config",
    "_composition_binding",
    "composition_port",
    "composition_strategy_control",
    "policy_topology",
    "reference_receiver",
    "_composition_trace_emitter",
    "_last_compound_dispatch_manifest",
    "_execution_observation_binding",
    "_execution_observation_capability",
    "_execution_observation_sink",
    "_execution_pose_observation_projector",
    "_next_policy_chunk_target_at",
    "_compound_composer_authority_profile",
}
_ZERO = {
    "_next_proposal_sequence",
    "_next_replay_proposal_sequence",
    "_next_composite_proposal_sequence",
}
_EMPTY = {
    "_projected_policy_emission_ids",
    "_composition_promotion_envelope_by_evaluation_id",
}
_BINDINGS = {
    "rig",
    "config",
    "clock",
    "pose_resolver",
    "_pose_execution_enabled",
    "_explicit_shared_safety",
    "component_execution_binding",
    "catalog",
    "action_schema",
    "reference_action_schema_hash",
    "execution_capability",
    "execution_capability_profile_hash",
    "component_target_capability",
    "target_execution_port",
    "target_runtime_action_contract",
    "reference_activation_spec",
    "core",
    "authoritative_port",
    "_chunk_execution_policy",
    "_component_dependencies_by_group",
    "_compound_execution_enabled",
    "_compound_hand_specs",
    "_composition_promotion_policy",
    "_composer_producer_epoch",
    "_execution_observation_side_to_component_id",
    "_execution_observation_representation",
    "_execution_observation_last_published_at",
    "_execution_observation_emitted_count",
    "_execution_observation_dropped_count",
    "_execution_observation_latest_sequence",
    "_execution_observation_watermark_by_stream",
    "observation_adapter",
    "_observation_adapter_by_node",
    "_started",
    "_lifecycle_entered",
    "_trace_close_pending",
    "_rig_close_pending",
    "_lifecycle_terminal",
}


def _encode(value):
    return encode_checkpoint_value(value, extra_record_types=_RECORDS)


def _decode(value):
    return decode_checkpoint_value(value, extra_record_types=_RECORDS)


def _bytes(value):
    return json.dumps(
        _encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _counter(value):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid runtime checkpoint counter.")
    return value


@dataclass(frozen=True)
class MujocoRuntimeCheckpoint:
    configuration_hash: str
    clock_value: float
    core: AuthoritativeControlCheckpoint
    state: dict
    observations: tuple
    content_hash: str


def _configuration(config):
    if (
        type(config) is not MujocoDeployConfig
        or config.reference_activation_profile_id != IMMEDIATE_QREF_SIM_V1
        or config.enable_primary_policy_authority
        or config.enable_replay_authority
        or config.enable_composer_authority
        or not config.enable_component_execution_handoff
        or config.direct_component_proposal_source is None
        or config.execution_observation_plan is not None
        or config.compound_composer_authority_profile is not None
        or config.compound_hand_action_specs_by_side
        or config.prune_expired_proposals
    ):
        raise ValueError(
            "Runtime checkpoint requires the direct manual immediate profile without observers."
        )
    return _hash(config)


def validate_runtime_checkpoint(checkpoint, *, config, clock):
    if (
        type(checkpoint) is not MujocoRuntimeCheckpoint
        or type(clock) is not ManualExecutionClock
        or checkpoint.configuration_hash != _configuration(config)
        or checkpoint.clock_value != clock()
        or checkpoint.core.source.captured_at != checkpoint.clock_value
        or checkpoint.content_hash
        != _hash(
            (
                checkpoint.configuration_hash,
                checkpoint.clock_value,
                checkpoint.core,
                checkpoint.state,
                checkpoint.observations,
            )
        )
    ):
        raise ValueError("Runtime checkpoint configuration, clock or content differs.")


def _qualify(runtime, *, constructing=False):
    if (
        type(runtime) is not MujocoDeployRuntime
        or set(vars(runtime)) != _STATE | _NONE | _ZERO | _EMPTY | _BINDINGS
        or type(runtime.clock) is not ManualExecutionClock
        or runtime.clock is not runtime.rig._clock
        or runtime.core.clock is not runtime.clock
        or runtime.core.trace_sink is not None
        or not runtime._pose_execution_enabled
        or not runtime._explicit_shared_safety
        or runtime._compound_execution_enabled
        or runtime.rig.config.realtime_worker
        or any(getattr(runtime, name) is not None for name in _NONE)
        or any(getattr(runtime, name) != 0 for name in _ZERO)
        or any(getattr(runtime, name) for name in _EMPTY)
    ):
        raise ValueError("Unsupported runtime owner, binding or field layout.")
    _configuration(runtime.config)
    if runtime._composer_producer_epoch != runtime.config.producer_epoch:
        raise ValueError("Unsupported composer epoch state.")
    expected = not constructing
    if (
        any(
            getattr(runtime, name) is not expected
            for name in (
                "_started",
                "_lifecycle_entered",
                "_trace_close_pending",
                "_rig_close_pending",
            )
        )
        or runtime._lifecycle_terminal
    ):
        raise ValueError("Runtime must be at a live completed operation boundary.")
    if (
        runtime._execution_observation_representation != "off"
        or runtime._execution_observation_last_published_at
        != {"reference": None, "measurement": None}
        or any(
            getattr(runtime, name) != {"reference": 0, "measurement": 0}
            for name in (
                "_execution_observation_emitted_count",
                "_execution_observation_dropped_count",
                "_execution_observation_latest_sequence",
            )
        )
        or runtime._execution_observation_watermark_by_stream
    ):
        raise ValueError("Execution observer state has no continuation contract.")


def _validate_state(runtime, state, observations):
    if type(state) is not dict or set(state) != _STATE:
        raise ValueError("Runtime state field layout differs.")
    facts = {fact.fact_id: fact for fact in runtime.core.trace_facts}
    context = runtime.core.trace_context

    def match(projector, item):
        expected = projector(context, item)
        actual = facts.get(expected.fact_id)
        if (
            actual is None
            or actual.semantic_content_hash != expected.semantic_content_hash
        ):
            raise ValueError("Runtime cache and core trace history differ.")

    targets = state["_immediate_component_targets"]
    admissions = state["_immediate_component_target_admissions"]
    if (
        type(targets) is not list
        or type(admissions) is not list
        or len(targets) != len(admissions)
        or _counter(state["_next_immediate_target_admission_sequence"])
        != len(admissions)
    ):
        raise ValueError("Incomplete immediate target history.")
    projected = set()
    for sequence, (target, admission) in enumerate(zip(targets, admissions)):
        if (
            type(target) is not ComponentExecutionTarget
            or type(admission) is not ComponentExecutionTargetAdmissionReceipt
            or admission.target_id != target.target_id
            or admission.target_hash != target.semantic_content_hash
            or admission.admission_sequence != sequence
        ):
            raise ValueError("Immediate target/admission links differ.")
        for projector, item in (
            (project_component_execution_target, target),
            (project_component_execution_target_admission, admission),
        ):
            match(projector, item)
            projected.add(projector(context, item).fact_id)
    for event, values, key in (
        ("component_execution_target", targets, "target_id"),
        ("component_execution_target_admission", admissions, "receipt_id"),
    ):
        if [getattr(item, key) for item in values] != [
            fact.fact_id
            for fact in runtime.core.trace_facts
            if fact.event_type == event
        ]:
            raise ValueError(
                "Immediate history is missing or reorders published records."
            )
    cursors = state["_reference_trace_cursor_by_stream"]
    expected_cursors = {
        "immediate_component_targets": len(targets),
        "immediate_component_target_admissions": len(admissions),
    }
    if (cursors != expected_cursors and not (not targets and cursors == {})) or state[
        "_projected_reference_fact_ids"
    ] != projected:
        raise ValueError("Reference trace cursor or publication watermark differs.")
    index = state["_immediate_target_by_reference_sequence"]
    if type(index) is not tuple or len(index) > 1:
        raise ValueError("Invalid immediate reference index.")
    for sequence, pair in index:
        if (
            _counter(sequence) < 1
            or sequence >= runtime.rig._next_reference_sequence
            or type(pair) is not tuple
            or len(pair) != 2
            or not pair[1].accepted
            or not any(
                _bytes(pair) == _bytes((t, a)) for t, a in zip(targets, admissions)
            )
        ):
            raise ValueError(
                "Immediate reference index is not a retained accepted target."
            )
    latest = state["_latest_target_execution_step"]
    if latest is not None:
        if type(latest) is not _LatestTargetExecution:
            raise ValueError("Unknown latest target status cache.")
        match(project_target_execution_request, latest.target_execution_request)
        for projector, item in (
            (project_target_execution_submission, latest.target_submission_receipt),
            (project_target_execution_completion, latest.target_completion_receipt),
        ):
            if item is not None:
                if (
                    item.target_execution_request_id
                    != latest.target_execution_request.request_id
                ):
                    raise ValueError("Latest target stage links differ.")
                match(projector, item)
        dispatch = latest.target_dispatch_result
        if dispatch is not None:
            if (
                type(dispatch) is not TargetExecutionDispatchResult
                or dispatch.target_activation is not None
                or _bytes(dispatch.completion_receipt)
                != _bytes(latest.target_completion_receipt)
            ):
                raise ValueError("Latest dispatch completion differs.")
            match(project_component_execution_target, dispatch.target)
            match(
                project_component_execution_target_admission, dispatch.target_admission
            )
    dispatch_context = state["_last_dispatch_context"]
    if dispatch_context is not None:
        if type(dispatch_context) is not AuthoritativeTargetDispatchContext:
            raise ValueError("Unsupported dispatch context.")
        match(
            project_target_execution_request, dispatch_context.target_execution_request
        )
        match(project_target_execution_submission, dispatch_context.submission_receipt)
    direct = runtime.config.direct_component_proposal_source
    last = runtime.core.proposals._last_sequence.get(
        (direct.producer_id, direct.producer_epoch), -1
    )
    if _counter(state["_next_direct_component_proposal_sequence"]) != last + 1:
        raise ValueError("Direct source sequence differs from admission owner.")
    if type(observations) is not tuple or not observations:
        raise ValueError("Missing policy observation owner.")
    nodes = set()
    for node, epoch, sequence in observations:
        if (
            type(node) is not str
            or not node
            or node in nodes
            or epoch != runtime.config.producer_epoch
        ):
            raise ValueError("Invalid observation owner identity.")
        _counter(sequence)
        nodes.add(node)
    if runtime.config.policy_node_id not in nodes:
        raise ValueError("Default observation alias is missing.")
    _validate_journal_watermarks(runtime, state, observations)


def _validate_journal_watermarks(runtime, state, observations):
    """Bind caches to their own last journal event, not to one common action."""
    facts = runtime.core.trace_facts
    by_id = {fact.fact_id: fact for fact in facts}
    requests = [fact for fact in facts if fact.event_type == "target_execution_request"]
    latest = state["_latest_target_execution_step"]
    latest_id = None if latest is None else latest.target_execution_request.request_id
    if latest_id != (requests[-1].fact_id if requests else None):
        raise ValueError("Latest target cache does not match the last request.")
    if latest is not None:
        for event, value in (
            ("target_execution_submission", latest.target_submission_receipt),
            ("target_execution_completion", latest.target_completion_receipt),
        ):
            recorded = [
                fact
                for fact in facts
                if fact.event_type == event
                and fact.payload["target_execution_request_id"] == latest_id
            ]
            if (None if value is None else value.receipt_id) != (
                recorded[-1].fact_id if recorded else None
            ):
                raise ValueError("Latest target cache is missing a recorded stage.")
        resolved = [
            target
            for target in state["_immediate_component_targets"]
            if target.target_execution_request_id == latest_id
        ]
        if bool(resolved) != (latest.target_dispatch_result is not None):
            raise ValueError(
                "Incomplete or unacknowledged backend dispatch has no recovery contract."
            )
        dispatch = latest.target_dispatch_result
        if dispatch is not None:
            matches = [
                fact
                for fact in facts
                if fact.event_type == "dispatch_receipt"
                and fact.payload["command_id"] == dispatch.target.payload["command_id"]
            ]
            if len(matches) != 1:
                raise ValueError("Latest target dispatch lacks an unambiguous receipt.")
            fact = matches[0]
            _, expected = project_dispatch_receipts(
                runtime.core.trace_context,
                dispatch.dispatch_receipts,
                command_id=fact.payload["command_id"],
                dispatch_event_id=fact.payload["dispatch_event_id"],
                backend_name=fact.payload["backend_name"],
                dispatched_at=fact.payload["dispatched_at"],
            )
            if fact.semantic_content_hash != expected.semantic_content_hash:
                raise ValueError("Latest dispatch receipt differs from journal.")

    dispatches = [
        fact
        for fact in facts
        if fact.event_type == "dispatch_receipt"
        and by_id[fact.payload["dispatch_event_id"]].payload["validation_accepted"]
    ]
    context = state["_last_dispatch_context"]
    if (None if context is None else context.command_id) != (
        dispatches[-1].payload["command_id"] if dispatches else None
    ):
        raise ValueError("Last dispatch context differs from the backend history.")
    if context is not None:
        event = by_id[dispatches[-1].payload["dispatch_event_id"]].payload
        command = by_id[context.command_id].payload
        directives = by_id[command["directive_id"]].payload[
            "directive_by_selection_group"
        ]
        selected = {
            group: item
            for group, item in directives.items()
            if item["directive"] == "publish_selected"
        }
        rebuilt = build_authoritative_target_dispatch_context(
            runtime_session_id=runtime.config.runtime_session_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            directive_id=command["directive_id"],
            command_id=context.command_id,
            dispatch_permit_id=event["permit_id"],
            owner_token_id=event["token_id"],
            owner_generation=event["owner_generation"],
            group_segment_generation_by_group={
                k: v["group_segment_generation"] for k, v in selected.items()
            },
            selected_action_ref_by_group={
                k: v["selected_action_ref"] for k, v in selected.items()
            },
            target_execution_request=context.target_execution_request,
            submission_receipt=context.submission_receipt,
        )
        if command["decision_id"] != context.decision_id or _bytes(context) != _bytes(
            rebuilt
        ):
            raise ValueError("Dispatch context fields differ from their causal owners.")
    expected_index = ()
    for fact in dispatches:
        sequences = {
            item["metadata"]["reference_sequence"]
            for item in fact.payload["receipts_by_effector"].values()
            if "reference_sequence" in item["metadata"]
        }
        if sequences:
            pairs = [
                (target, admission)
                for target, admission in zip(
                    state["_immediate_component_targets"],
                    state["_immediate_component_target_admissions"],
                )
                if target.payload["command_id"] == fact.payload["command_id"]
                and admission.accepted
            ]
            if len(sequences) != 1 or len(pairs) != 1:
                raise ValueError("Ambiguous admitted reference history.")
            expected_index = ((sequences.pop(), pairs[0]),)
    if _bytes(expected_index) != _bytes(
        state["_immediate_target_by_reference_sequence"]
    ):
        raise ValueError("Reference index differs from its backend receipt.")

    expected_observations = {}
    for fact in facts:
        if fact.event_type == "observation_cut":
            key = (fact.payload["policy_node_id"], fact.payload["policy_node_epoch"])
            expected_observations[key] = max(
                expected_observations.get(key, 0), fact.payload["sequence"] + 1
            )
    actual = {(node, epoch): sequence for node, epoch, sequence in observations}
    if any(
        actual.get(key) != value for key, value in expected_observations.items()
    ) or any(
        value != expected_observations.get(key, 0) for key, value in actual.items()
    ):
        raise ValueError("Observation sequence differs from its published history.")


def capture_runtime(runtime):
    """Call only after fencing the entire admission/decision/trace operation."""
    _qualify(runtime)
    state = {name: getattr(runtime, name) for name in _STATE}
    state["_immediate_target_by_reference_sequence"] = tuple(
        runtime._immediate_target_by_reference_sequence.items()
    )
    observations = []
    for node, adapter in runtime._observation_adapter_by_node.items():
        if (
            type(adapter) is not MujocoPolicyObservationAdapter
            or set(vars(adapter))
            != {"rig", "clock", "policy_node_id", "policy_node_epoch", "_sequence"}
            or adapter.rig is not runtime.rig
            or adapter.clock is not runtime.clock
            or node != adapter.policy_node_id
        ):
            raise ValueError("Unknown policy observation owner or binding.")
        observations.append((node, adapter.policy_node_epoch, adapter._sequence))
    if (
        runtime.observation_adapter
        is not runtime._observation_adapter_by_node[runtime.config.policy_node_id]
    ):
        raise ValueError("Default observation adapter alias differs.")
    state, observations = _decode(_encode((state, tuple(observations))))
    _validate_state(runtime, state, observations)
    core = capture_authoritative_control(runtime.core)
    config_hash, now = _configuration(runtime.config), runtime.clock()
    return MujocoRuntimeCheckpoint(
        config_hash,
        now,
        core,
        state,
        observations,
        _hash((config_hash, now, core, state, observations)),
    )


def restore_runtime_state(runtime, checkpoint):
    """Finish the ordinary constructor after its original core restore path."""
    validate_runtime_checkpoint(checkpoint, config=runtime.config, clock=runtime.clock)
    _qualify(runtime, constructing=True)
    state, observations = _decode(_encode((checkpoint.state, checkpoint.observations)))
    _validate_state(runtime, state, observations)
    state["_immediate_target_by_reference_sequence"] = dict(
        state["_immediate_target_by_reference_sequence"]
    )
    for name, value in state.items():
        setattr(runtime, name, value)
    adapters = {}
    for node, epoch, sequence in observations:
        adapter = MujocoPolicyObservationAdapter(
            runtime.rig,
            policy_node_id=node,
            policy_node_epoch=epoch,
            clock=runtime.clock,
        )
        adapter._sequence = sequence
        adapters[node] = adapter
    runtime._observation_adapter_by_node = adapters
    runtime.observation_adapter = adapters[runtime.config.policy_node_id]
    for name in (
        "_started",
        "_lifecycle_entered",
        "_trace_close_pending",
        "_rig_close_pending",
    ):
        setattr(runtime, name, True)
    _qualify(runtime)
