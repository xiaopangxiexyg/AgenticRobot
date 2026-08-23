"""Backend-, model-, and topology-neutral HITL application contracts.

Concrete sim or real applications provide lower HOLD, Home capture, source
selection, activation, and fresh-emission effects.  ControlTrace projection
remains in :mod:`control_station.control_trace`, so observation cannot become
motion authority.
"""

from .activation import (
    HITL_ACTIVATION_PRIME_HASH_DOMAIN_DEVELOPMENT,
    HITLActivationPrimeEvidence,
    build_hitl_activation_prime_evidence,
)
from .exclusive import (
    EXCLUSIVE_HITL_HUMAN_TO_POLICY,
    EXCLUSIVE_HITL_POLICY_TO_HUMAN,
    EXCLUSIVE_HITL_SWITCH_DIRECTIONS,
    EXCLUSIVE_HITL_SWITCH_STATUSES,
    ExclusiveHITLSwitchEvidence,
    ExclusiveHITLTransitionObservation,
    ExclusiveHITLTransitionPort,
    run_exclusive_hitl_switch_transaction,
    run_exclusive_hitl_switch_via_port,
)
from .intents import (
    EXCLUSIVE_HITL_INTENT_HASH_DOMAIN_V1,
    EXCLUSIVE_HITL_SOURCES,
    ExclusiveHITLSourceIntent,
)
from .nonblocking import (
    HITL_SWITCH_FAILED,
    HITL_SWITCH_HOLD_REQUESTED,
    HITL_SWITCH_IDLE,
    HITL_SWITCH_WAITING_HUMAN_HOME,
    HITL_SWITCH_WAITING_POLICY_EMISSION,
    HITL_SWITCH_WAITING_POLICY_RESET,
    HITL_SWITCH_WAITING_TARGET_DISPATCH,
    ExclusiveHITLTransitionState,
    begin_exclusive_hitl_transition,
    complete_exclusive_hitl_transition,
    fail_exclusive_hitl_transition,
    observe_exclusive_hitl_hold,
    observe_exclusive_hitl_human_home,
    observe_exclusive_hitl_policy_emission,
    observe_exclusive_hitl_policy_reset,
)

__all__ = [
    "EXCLUSIVE_HITL_HUMAN_TO_POLICY",
    "EXCLUSIVE_HITL_POLICY_TO_HUMAN",
    "EXCLUSIVE_HITL_SWITCH_DIRECTIONS",
    "EXCLUSIVE_HITL_SWITCH_STATUSES",
    "EXCLUSIVE_HITL_INTENT_HASH_DOMAIN_V1",
    "EXCLUSIVE_HITL_SOURCES",
    "HITL_ACTIVATION_PRIME_HASH_DOMAIN_DEVELOPMENT",
    "ExclusiveHITLSwitchEvidence",
    "ExclusiveHITLTransitionObservation",
    "ExclusiveHITLTransitionPort",
    "ExclusiveHITLSourceIntent",
    "ExclusiveHITLTransitionState",
    "HITL_SWITCH_FAILED",
    "HITL_SWITCH_HOLD_REQUESTED",
    "HITL_SWITCH_IDLE",
    "HITL_SWITCH_WAITING_HUMAN_HOME",
    "HITL_SWITCH_WAITING_POLICY_EMISSION",
    "HITL_SWITCH_WAITING_POLICY_RESET",
    "HITL_SWITCH_WAITING_TARGET_DISPATCH",
    "HITLActivationPrimeEvidence",
    "build_hitl_activation_prime_evidence",
    "begin_exclusive_hitl_transition",
    "complete_exclusive_hitl_transition",
    "fail_exclusive_hitl_transition",
    "observe_exclusive_hitl_hold",
    "observe_exclusive_hitl_human_home",
    "observe_exclusive_hitl_policy_emission",
    "observe_exclusive_hitl_policy_reset",
    "run_exclusive_hitl_switch_transaction",
    "run_exclusive_hitl_switch_via_port",
]
