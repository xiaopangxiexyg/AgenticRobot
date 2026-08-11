"""Application-level contracts and thin profiles for the DexTeleop Control Station.

CS-1a provides a local read-only builder, CS-1b adds null-target typed intent
admission, and CS-1c adds bounded no-I/O operation coordination. Concrete
profiles such as VIVE coordinate calibration may own application lifecycle and
GUI state, but do not create a robot/sim execution owner, motion authority, or
episode lifecycle.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Keep package-root compatibility without importing unrelated runtimes.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    'CONTROL_STATION_INTENT_KINDS_V1': ('.contracts', 'CONTROL_STATION_INTENT_KINDS_V1'),
    'CONTROL_STATION_SOURCE_KINDS_V1': ('.contracts', 'CONTROL_STATION_SOURCE_KINDS_V1'),
    'CONTROL_STATION_SOURCE_USES_V1': ('.contracts', 'CONTROL_STATION_SOURCE_USES_V1'),
    'CONTROL_STATION_TARGET_KINDS_V1': ('.contracts', 'CONTROL_STATION_TARGET_KINDS_V1'),
    'ControlStationRunManifest': ('.contracts', 'ControlStationRunManifest'),
    'ExecutionDomainBinding': ('.contracts', 'ExecutionDomainBinding'),
    'OperatorControlLease': ('.contracts', 'OperatorControlLease'),
    'SourceSlotSpec': ('.contracts', 'SourceSlotSpec'),
    'StationIntent': ('.contracts', 'StationIntent'),
    'StationIntentAdmissionReceipt': ('.contracts', 'StationIntentAdmissionReceipt'),
    'StationIntentDraft': ('.contracts', 'StationIntentDraft'),
    'StationOperationResult': ('.contracts', 'StationOperationResult'),
    'StationOperationResultDraft': ('.contracts', 'StationOperationResultDraft'),
    'COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1': ('.state_machine', 'COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1'),
    'COMPOSITION_SESSION_OPERATION_READY_STATUS_V1': ('.state_machine', 'COMPOSITION_SESSION_OPERATION_READY_STATUS_V1'),
    'COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1': ('.state_machine', 'COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1'),
    'COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1': ('.state_machine', 'COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1'),
    'CONTROL_STATION_LIFECYCLE_STATES_V1': ('.state_machine', 'CONTROL_STATION_LIFECYCLE_STATES_V1'),
    'HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1': ('.state_machine', 'HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1'),
    'HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1': ('.state_machine', 'HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1'),
    'HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1': ('.state_machine', 'HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1'),
    'SOURCE_SLOT_RUNTIME_STATES_V1': ('.state_machine', 'SOURCE_SLOT_RUNTIME_STATES_V1'),
    'TARGET_RUNTIME_STATES_V1': ('.state_machine', 'TARGET_RUNTIME_STATES_V1'),
    'ControlStationState': ('.state_machine', 'ControlStationState'),
    'ControlStationOperationTransition': ('.state_machine', 'ControlStationOperationTransition'),
    'ControlStationTransition': ('.state_machine', 'ControlStationTransition'),
    'PendingStationOperation': ('.state_machine', 'PendingStationOperation'),
    'admit_station_operation_result': ('.state_machine', 'admit_station_operation_result'),
    'admit_station_intent': ('.state_machine', 'admit_station_intent'),
    'apply_station_operation_result': ('.state_machine', 'apply_station_operation_result'),
    'apply_station_intent': ('.state_machine', 'apply_station_intent'),
    'initial_control_station_state': ('.state_machine', 'initial_control_station_state'),
    'CompositionSessionController': ('.composition_session', 'CompositionSessionController'),
    'CompositionSessionOperationEvidence': ('.composition_session_operation', 'CompositionSessionOperationEvidence'),
    'StationCompositionSessionCompletionRejected': ('.composition_session_operation', 'StationCompositionSessionCompletionRejected'),
    'StationCompositionSessionOperationPort': ('.composition_session_operation', 'StationCompositionSessionOperationPort'),
    'DeterministicNoIoStationAdapter': ('.adapters', 'DeterministicNoIoStationAdapter'),
    'FakeSourceSlotAdapter': ('.adapters', 'FakeSourceSlotAdapter'),
    'NullExecutionTargetAdapter': ('.adapters', 'NullExecutionTargetAdapter'),
    'StationOperationAdapter': ('.adapters', 'StationOperationAdapter'),
    'ControlStationRuntime': ('.runtime', 'ControlStationRuntime'),
    'StationOperationExecution': ('.runtime', 'StationOperationExecution'),
    'StationPendingOperationSuperseded': ('.runtime', 'StationPendingOperationSuperseded'),
    'STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1': ('.publication_barrier', 'STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1'),
    'StationPublicationBarrier': ('.publication_barrier', 'StationPublicationBarrier'),
    'StationPublicationBarrierSnapshot': ('.publication_barrier', 'StationPublicationBarrierSnapshot'),
    'StationControlTraceBridge': ('.control_trace', 'StationControlTraceBridge'),
    'project_station_intent_transition': ('.control_trace', 'project_station_intent_transition'),
    'project_station_operation_transition': ('.control_trace', 'project_station_operation_transition'),
    'project_station_publication_barrier_transition': ('.control_trace', 'project_station_publication_barrier_transition'),
    'project_station_run_manifest': ('.control_trace', 'project_station_run_manifest'),
    'BoundedStationEventLog': ('.events', 'BoundedStationEventLog'),
    'StationUiEvent': ('.events', 'StationUiEvent'),
    'StationUiEventSnapshot': ('.events', 'StationUiEventSnapshot'),
    'CS1C_COORDINATED_OPERATION_KINDS': ('.operation_coordinator', 'CS1C_COORDINATED_OPERATION_KINDS'),
    'CoordinatedIntentSubmission': ('.operation_coordinator', 'CoordinatedIntentSubmission'),
    'NoIoStationOperationCoordinator': ('.operation_coordinator', 'NoIoStationOperationCoordinator'),
    'SideEffectStationOperationCoordinator': ('.operation_coordinator', 'SideEffectStationOperationCoordinator'),
    'StationOperationCompletion': ('.operation_coordinator', 'StationOperationCompletion'),
    'StationOperationCoordinatorUnavailable': ('.operation_coordinator', 'StationOperationCoordinatorUnavailable'),
    'StationOperationEnqueueReceipt': ('.operation_coordinator', 'StationOperationEnqueueReceipt'),
    'CONTROL_STATION_STATUS_SCHEMA_V1': ('.status', 'CONTROL_STATION_STATUS_SCHEMA_V1'),
    'build_control_station_status_payload': ('.status', 'build_control_station_status_payload'),
}

__all__ = [
    'COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1',
    'COMPOSITION_SESSION_OPERATION_READY_STATUS_V1',
    'COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1',
    'COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1',
    'CONTROL_STATION_INTENT_KINDS_V1',
    'CONTROL_STATION_LIFECYCLE_STATES_V1',
    'CONTROL_STATION_SOURCE_KINDS_V1',
    'CONTROL_STATION_SOURCE_USES_V1',
    'CONTROL_STATION_STATUS_SCHEMA_V1',
    'CONTROL_STATION_TARGET_KINDS_V1',
    'HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1',
    'HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1',
    'HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1',
    'SOURCE_SLOT_RUNTIME_STATES_V1',
    'STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1',
    'TARGET_RUNTIME_STATES_V1',
    'CS1C_COORDINATED_OPERATION_KINDS',
    'BoundedStationEventLog',
    'CoordinatedIntentSubmission',
    'ControlStationRunManifest',
    'ControlStationRuntime',
    'ControlStationOperationTransition',
    'ControlStationState',
    'ControlStationTransition',
    'CompositionSessionController',
    'CompositionSessionOperationEvidence',
    'ExecutionDomainBinding',
    'DeterministicNoIoStationAdapter',
    'FakeSourceSlotAdapter',
    'NullExecutionTargetAdapter',
    'NoIoStationOperationCoordinator',
    'SideEffectStationOperationCoordinator',
    'OperatorControlLease',
    'PendingStationOperation',
    'SourceSlotSpec',
    'StationIntent',
    'StationIntentAdmissionReceipt',
    'StationIntentDraft',
    'StationOperationCompletion',
    'StationOperationCoordinatorUnavailable',
    'StationOperationEnqueueReceipt',
    'StationOperationAdapter',
    'StationOperationExecution',
    'StationPendingOperationSuperseded',
    'StationCompositionSessionCompletionRejected',
    'StationCompositionSessionOperationPort',
    'StationOperationResult',
    'StationOperationResultDraft',
    'StationControlTraceBridge',
    'StationPublicationBarrier',
    'StationPublicationBarrierSnapshot',
    'StationUiEvent',
    'StationUiEventSnapshot',
    'admit_station_operation_result',
    'admit_station_intent',
    'apply_station_operation_result',
    'apply_station_intent',
    'build_control_station_status_payload',
    'initial_control_station_state',
    'project_station_intent_transition',
    'project_station_operation_transition',
    'project_station_publication_barrier_transition',
    'project_station_run_manifest',
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
