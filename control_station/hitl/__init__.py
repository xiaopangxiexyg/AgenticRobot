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
__all__ = [
    "EXCLUSIVE_HITL_HUMAN_TO_POLICY",
    "EXCLUSIVE_HITL_POLICY_TO_HUMAN",
    "EXCLUSIVE_HITL_SWITCH_DIRECTIONS",
    "EXCLUSIVE_HITL_SWITCH_STATUSES",
    "HITL_ACTIVATION_PRIME_HASH_DOMAIN_DEVELOPMENT",
    "ExclusiveHITLSwitchEvidence",
    "ExclusiveHITLTransitionObservation",
    "ExclusiveHITLTransitionPort",
    "HITLActivationPrimeEvidence",
    "build_hitl_activation_prime_evidence",
    "run_exclusive_hitl_switch_transaction",
    "run_exclusive_hitl_switch_via_port",
]
