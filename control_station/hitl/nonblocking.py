"""Pure multi-cycle reducer for an exclusive HITL switch.

The reducer describes facts already established by the application owner.  It
does not call HOLD, Home, policy reset, composition, safety, or dispatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dataset_contracts.canonical_encoding import canonical_content_hash

from .intents import EXCLUSIVE_HITL_SOURCES, ExclusiveHITLSourceIntent


HITL_SWITCH_IDLE = "idle"
HITL_SWITCH_HOLD_REQUESTED = "hold_requested"
HITL_SWITCH_WAITING_HUMAN_HOME = "waiting_human_home"
HITL_SWITCH_WAITING_POLICY_RESET = "waiting_policy_reset"
HITL_SWITCH_WAITING_POLICY_EMISSION = "waiting_policy_emission"
HITL_SWITCH_WAITING_TARGET_DISPATCH = "waiting_target_dispatch"
HITL_SWITCH_FAILED = "failed"
HITL_SWITCH_PHASES = (
    HITL_SWITCH_IDLE,
    HITL_SWITCH_HOLD_REQUESTED,
    HITL_SWITCH_WAITING_HUMAN_HOME,
    HITL_SWITCH_WAITING_POLICY_RESET,
    HITL_SWITCH_WAITING_POLICY_EMISSION,
    HITL_SWITCH_WAITING_TARGET_DISPATCH,
    HITL_SWITCH_FAILED,
)
HITL_SWITCH_ID_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.exclusive_hitl_switch_id.v1"
)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class ExclusiveHITLTransitionState:
    active_source: str
    phase: str = HITL_SWITCH_IDLE
    generation: int = 0
    pending_intent: ExclusiveHITLSourceIntent | None = None
    switch_id: str | None = None
    barrier_at: float | None = None
    hold_evidence_refs: tuple[str, ...] = ()
    policy_reset_performed: bool = False
    target_emission_id: str | None = None
    target_emission_hash: str | None = None
    home_capture_id: str | None = None
    home_capture_hash: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.active_source not in EXCLUSIVE_HITL_SOURCES:
            raise ValueError("active_source must be 'human' or 'policy'.")
        if self.phase not in HITL_SWITCH_PHASES:
            raise ValueError("Unsupported HITL transition phase.")
        if int(self.generation) < 0 or int(self.generation) != self.generation:
            raise ValueError("generation must be a non-negative integer.")
        pending = self.pending_intent
        if self.phase == HITL_SWITCH_IDLE:
            if pending is not None or self.switch_id is not None:
                raise ValueError("Idle HITL state cannot carry a pending switch.")
        elif self.phase != HITL_SWITCH_FAILED:
            if pending is None or not str(self.switch_id or "").strip():
                raise ValueError("An active HITL transition requires intent and switch ID.")
        if pending is not None and pending.requested_source == self.active_source:
            raise ValueError("A pending transition must target the inactive source.")
        if (self.target_emission_id is None) != (
            self.target_emission_hash is None
        ):
            raise ValueError("Target emission ID/hash must appear together.")
        if (self.home_capture_id is None) != (self.home_capture_hash is None):
            raise ValueError("Home capture ID/hash must appear together.")

    @property
    def target_source(self) -> str | None:
        return None if self.pending_intent is None else self.pending_intent.requested_source


def begin_exclusive_hitl_transition(
    state: ExclusiveHITLTransitionState,
    intent: ExclusiveHITLSourceIntent,
) -> ExclusiveHITLTransitionState:
    if not isinstance(state, ExclusiveHITLTransitionState):
        raise TypeError("state must be ExclusiveHITLTransitionState.")
    if not isinstance(intent, ExclusiveHITLSourceIntent):
        raise TypeError("intent must be ExclusiveHITLSourceIntent.")
    if state.phase != HITL_SWITCH_IDLE:
        raise RuntimeError("An exclusive HITL transition is already in progress.")
    if intent.requested_source == state.active_source:
        return state
    generation = state.generation + 1
    digest = canonical_content_hash(
        {
            "generation": generation,
            "previous_source": state.active_source,
            "target_source": intent.requested_source,
            "intent_id": intent.intent_id,
            "intent_hash": intent.semantic_content_hash,
        },
        domain=HITL_SWITCH_ID_HASH_DOMAIN_V1,
    )
    return ExclusiveHITLTransitionState(
        active_source=state.active_source,
        phase=HITL_SWITCH_HOLD_REQUESTED,
        generation=generation,
        pending_intent=intent,
        switch_id=f"exclusive-hitl-switch:{digest}",
    )


def observe_exclusive_hitl_hold(
    state: ExclusiveHITLTransitionState,
    *,
    established_at: float,
    evidence_refs: tuple[str, ...],
) -> ExclusiveHITLTransitionState:
    if state.phase != HITL_SWITCH_HOLD_REQUESTED:
        raise RuntimeError("HITL HOLD evidence is not expected in this phase.")
    refs = tuple(str(value).strip() for value in evidence_refs)
    if not refs or any(not value for value in refs) or len(set(refs)) != len(refs):
        raise ValueError("HITL HOLD requires unique non-empty evidence refs.")
    target = state.target_source
    return replace(
        state,
        phase=(
            HITL_SWITCH_WAITING_HUMAN_HOME
            if target == "human"
            else HITL_SWITCH_WAITING_POLICY_RESET
        ),
        barrier_at=_finite(established_at, name="established_at"),
        hold_evidence_refs=refs,
    )


def observe_exclusive_hitl_policy_reset(
    state: ExclusiveHITLTransitionState,
    *,
    reset_at: float,
) -> ExclusiveHITLTransitionState:
    if state.phase != HITL_SWITCH_WAITING_POLICY_RESET or state.target_source != "policy":
        raise RuntimeError("Policy reset evidence is not expected in this phase.")
    reset_time = _finite(reset_at, name="reset_at")
    if state.barrier_at is not None and reset_time < state.barrier_at:
        raise ValueError("Policy reset cannot predate lower HOLD.")
    return replace(
        state,
        phase=HITL_SWITCH_WAITING_POLICY_EMISSION,
        barrier_at=reset_time,
        policy_reset_performed=True,
    )


def observe_exclusive_hitl_policy_emission(
    state: ExclusiveHITLTransitionState,
    *,
    emission_id: str,
    emission_hash: str,
    emitted_at: float,
) -> ExclusiveHITLTransitionState:
    if (
        state.phase != HITL_SWITCH_WAITING_POLICY_EMISSION
        or state.target_source != "policy"
        or not state.policy_reset_performed
    ):
        raise RuntimeError("Fresh policy emission is not expected in this phase.")
    emission_time = _finite(emitted_at, name="emitted_at")
    if state.barrier_at is None or emission_time < state.barrier_at:
        raise ValueError("Policy emission must be fresh relative to reset.")
    identifier = str(emission_id).strip()
    digest = str(emission_hash).strip()
    if not identifier or not digest:
        raise ValueError("Policy emission ID/hash must be non-empty.")
    return replace(
        state,
        phase=HITL_SWITCH_WAITING_TARGET_DISPATCH,
        target_emission_id=identifier,
        target_emission_hash=digest,
    )


def observe_exclusive_hitl_human_home(
    state: ExclusiveHITLTransitionState,
    *,
    home_capture_id: str,
    home_capture_hash: str,
    captured_at: float,
) -> ExclusiveHITLTransitionState:
    if state.phase != HITL_SWITCH_WAITING_HUMAN_HOME or state.target_source != "human":
        raise RuntimeError("Human Home evidence is not expected in this phase.")
    captured = _finite(captured_at, name="captured_at")
    if state.barrier_at is None or captured < state.barrier_at:
        raise ValueError("Human Home must be fresh relative to lower HOLD.")
    identifier = str(home_capture_id).strip()
    digest = str(home_capture_hash).strip()
    if not identifier or not digest:
        raise ValueError("Human Home ID/hash must be non-empty.")
    return replace(
        state,
        phase=HITL_SWITCH_WAITING_TARGET_DISPATCH,
        home_capture_id=identifier,
        home_capture_hash=digest,
    )


def complete_exclusive_hitl_transition(
    state: ExclusiveHITLTransitionState,
) -> ExclusiveHITLTransitionState:
    if state.phase != HITL_SWITCH_WAITING_TARGET_DISPATCH:
        raise RuntimeError("Target dispatch is not expected in this phase.")
    assert state.target_source is not None
    return ExclusiveHITLTransitionState(
        active_source=state.target_source,
        generation=state.generation,
    )


def fail_exclusive_hitl_transition(
    state: ExclusiveHITLTransitionState,
    error: BaseException | str,
) -> ExclusiveHITLTransitionState:
    if state.phase == HITL_SWITCH_IDLE:
        raise RuntimeError("There is no HITL transition to fail.")
    text = (
        str(error).strip()
        if isinstance(error, str)
        else f"{type(error).__name__}:{error}"
    )
    if not text:
        raise ValueError("error must be non-empty.")
    return replace(state, phase=HITL_SWITCH_FAILED, error=text)


__all__ = [
    "HITL_SWITCH_FAILED",
    "HITL_SWITCH_HOLD_REQUESTED",
    "HITL_SWITCH_IDLE",
    "HITL_SWITCH_PHASES",
    "HITL_SWITCH_WAITING_HUMAN_HOME",
    "HITL_SWITCH_WAITING_POLICY_EMISSION",
    "HITL_SWITCH_WAITING_POLICY_RESET",
    "HITL_SWITCH_WAITING_TARGET_DISPATCH",
    "ExclusiveHITLTransitionState",
    "begin_exclusive_hitl_transition",
    "complete_exclusive_hitl_transition",
    "fail_exclusive_hitl_transition",
    "observe_exclusive_hitl_hold",
    "observe_exclusive_hitl_human_home",
    "observe_exclusive_hitl_policy_emission",
    "observe_exclusive_hitl_policy_reset",
]
