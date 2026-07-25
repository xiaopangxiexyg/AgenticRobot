"""Application-level contracts for the future DexTeleop Control Station.

The package is intentionally entrypoint-free.  CS-1a provides a local read-only
builder, CS-1b adds null-target typed intent admission, and CS-1c adds bounded
no-I/O operation coordination.  There is still no device construction, policy
scheduler, dispatcher, motion authorization, or episode lifecycle here.
"""

from .contracts import (
    CONTROL_STATION_INTENT_KINDS_V1,
    CONTROL_STATION_SOURCE_KINDS_V1,
    CONTROL_STATION_SOURCE_USES_V1,
    CONTROL_STATION_TARGET_KINDS_V1,
    ControlStationRunManifest,
    ExecutionDomainBinding,
    OperatorControlLease,
    SourceSlotSpec,
    StationIntent,
    StationIntentAdmissionReceipt,
    StationIntentDraft,
    StationOperationResult,
    StationOperationResultDraft,
)
from .state_machine import (
    COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1,
    CONTROL_STATION_LIFECYCLE_STATES_V1,
    HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1,
    HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1,
    HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1,
    SOURCE_SLOT_RUNTIME_STATES_V1,
    TARGET_RUNTIME_STATES_V1,
    ControlStationState,
    ControlStationOperationTransition,
    ControlStationTransition,
    PendingStationOperation,
    admit_station_operation_result,
    admit_station_intent,
    apply_station_operation_result,
    apply_station_intent,
    initial_control_station_state,
)
from .composition_session import CompositionSessionController
from .adapters import (
    DeterministicNoIoStationAdapter,
    FakeSourceSlotAdapter,
    NullExecutionTargetAdapter,
    StationOperationAdapter,
)
from .runtime import (
    ControlStationRuntime,
    StationOperationExecution,
    StationPendingOperationSuperseded,
)
from .publication_barrier import (
    STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1,
    StationPublicationBarrier,
    StationPublicationBarrierSnapshot,
)
from .control_trace import (
    StationControlTraceBridge,
    project_station_intent_transition,
    project_station_operation_transition,
    project_station_publication_barrier_transition,
    project_station_run_manifest,
)
from .events import BoundedStationEventLog, StationUiEvent, StationUiEventSnapshot
from .operation_coordinator import (
    CS1C_COORDINATED_OPERATION_KINDS,
    CoordinatedIntentSubmission,
    NoIoStationOperationCoordinator,
    SideEffectStationOperationCoordinator,
    StationOperationCompletion,
    StationOperationCoordinatorUnavailable,
    StationOperationEnqueueReceipt,
)
from .status import (
    CONTROL_STATION_STATUS_SCHEMA_V1,
    build_control_station_status_payload,
)
__all__ = [
    "COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_READY_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1",
    "CONTROL_STATION_INTENT_KINDS_V1",
    "CONTROL_STATION_LIFECYCLE_STATES_V1",
    "CONTROL_STATION_SOURCE_KINDS_V1",
    "CONTROL_STATION_SOURCE_USES_V1",
    "CONTROL_STATION_STATUS_SCHEMA_V1",
    "CONTROL_STATION_TARGET_KINDS_V1",
    "HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1",
    "HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1",
    "HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1",
    "SOURCE_SLOT_RUNTIME_STATES_V1",
    "STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_STATES_V1",
    "CS1C_COORDINATED_OPERATION_KINDS",
    "BoundedStationEventLog",
    "CoordinatedIntentSubmission",
    "ControlStationRunManifest",
    "ControlStationRuntime",
    "ControlStationOperationTransition",
    "ControlStationState",
    "ControlStationTransition",
    "CompositionSessionController",
    "ExecutionDomainBinding",
    "DeterministicNoIoStationAdapter",
    "FakeSourceSlotAdapter",
    "NullExecutionTargetAdapter",
    "NoIoStationOperationCoordinator",
    "SideEffectStationOperationCoordinator",
    "OperatorControlLease",
    "PendingStationOperation",
    "SourceSlotSpec",
    "StationIntent",
    "StationIntentAdmissionReceipt",
    "StationIntentDraft",
    "StationOperationCompletion",
    "StationOperationCoordinatorUnavailable",
    "StationOperationEnqueueReceipt",
    "StationOperationAdapter",
    "StationOperationExecution",
    "StationPendingOperationSuperseded",
    "StationOperationResult",
    "StationOperationResultDraft",
    "StationControlTraceBridge",
    "StationPublicationBarrier",
    "StationPublicationBarrierSnapshot",
    "StationUiEvent",
    "StationUiEventSnapshot",
    "admit_station_operation_result",
    "admit_station_intent",
    "apply_station_operation_result",
    "apply_station_intent",
    "build_control_station_status_payload",
    "initial_control_station_state",
    "project_station_intent_transition",
    "project_station_operation_transition",
    "project_station_publication_barrier_transition",
    "project_station_run_manifest",
]
