"""Default-off Control Station port for residual rebase-and-clear intents.

The port binds an accepted operator intent to the reusable trajectory reducer.
It never polls a device, captures an anchor sample, changes authority, creates
HOLD, publishes a proposal, or sends an action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.residual_anchor import ResidualAnchorRequest
from control_composition.trajectory import (
    RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER,
    RESIDUAL_REBASE_CAPTURE_MODES_DEVELOPMENT,
    ResidualRebaseRequest,
    TrajectoryEditingState,
    request_residual_rebase,
)

from ..contracts import StationIntent, StationIntentAdmissionReceipt


STATION_RESIDUAL_REBASE_PROFILE_OFF = "off"
STATION_RESIDUAL_REBASE_COMMAND_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.residual_rebase_command.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class StationResidualRebaseProfile:
    """Application-owned, default-off binding for one active composer slot."""

    profile_id: str = STATION_RESIDUAL_REBASE_PROFILE_OFF
    composer_source_slot_id: str = "unbound-composer"
    enabled: bool = False
    send_actions: bool = False

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        composer_slot = _required(
            self.composer_source_slot_id,
            name="composer_source_slot_id",
        )
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("Residual rebase Station port cannot send actions.")
        if self.enabled and profile_id == STATION_RESIDUAL_REBASE_PROFILE_OFF:
            raise ValueError("An enabled residual rebase profile requires an id.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "composer_source_slot_id", composer_slot)


@dataclass(frozen=True)
class StationResidualRebaseCommand:
    """Already-produced application parameters; no device work occurs here."""

    anchor_request: ResidualAnchorRequest
    effective_not_before_target_at: float
    capture_deadline: float
    capture_mode: str = RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER
    trigger_source_action_id: str | None = None
    trigger_source_action_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_request, ResidualAnchorRequest):
            raise TypeError("anchor_request must be ResidualAnchorRequest.")
        target = float(self.effective_not_before_target_at)
        deadline = float(self.capture_deadline)
        if not math.isfinite(target) or not math.isfinite(deadline):
            raise ValueError("Residual rebase command times must be finite.")
        capture_mode = _required(self.capture_mode, name="capture_mode")
        if capture_mode not in RESIDUAL_REBASE_CAPTURE_MODES_DEVELOPMENT:
            raise ValueError("Unsupported residual rebase capture mode.")
        source_id = (
            None
            if self.trigger_source_action_id is None
            else _required(
                self.trigger_source_action_id,
                name="trigger_source_action_id",
            )
        )
        source_hash = (
            None
            if self.trigger_source_action_hash is None
            else _required(
                self.trigger_source_action_hash,
                name="trigger_source_action_hash",
            )
        )
        if (source_id is None) != (source_hash is None):
            raise ValueError("Trigger source action identity must be complete.")
        payload = {
            "anchor_request_id": self.anchor_request.request_id,
            "anchor_request_hash": self.anchor_request.semantic_content_hash,
            "effective_not_before_target_at": target,
            "capture_deadline": deadline,
            "capture_mode": capture_mode,
            "trigger_source_action_id": source_id,
            "trigger_source_action_hash": source_hash,
        }
        object.__setattr__(self, "effective_not_before_target_at", target)
        object.__setattr__(self, "capture_deadline", deadline)
        object.__setattr__(self, "capture_mode", capture_mode)
        object.__setattr__(self, "trigger_source_action_id", source_id)
        object.__setattr__(self, "trigger_source_action_hash", source_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=STATION_RESIDUAL_REBASE_COMMAND_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class StationResidualRebaseCycle:
    profile_id: str
    status: str
    state_before_hash: str | None = None
    state_after: TrajectoryEditingState | None = None
    rebase_request: ResidualRebaseRequest | None = None

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        status = _required(self.status, name="status")
        if status not in {"disabled", "requested"}:
            raise ValueError("Unsupported Station residual rebase cycle status.")
        if status == "disabled":
            if any(
                value is not None
                for value in (
                    self.state_before_hash,
                    self.state_after,
                    self.rebase_request,
                )
            ):
                raise ValueError("A disabled rebase cycle cannot carry state.")
        elif (
            self.state_before_hash is None
            or not isinstance(self.state_after, TrajectoryEditingState)
            or not isinstance(self.rebase_request, ResidualRebaseRequest)
        ):
            raise ValueError("A requested rebase cycle requires complete state.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)


class StationResidualRebaseController:
    """Bind one accepted Station intent to one pure reducer transition."""

    def __init__(self, *, profile: StationResidualRebaseProfile) -> None:
        if not isinstance(profile, StationResidualRebaseProfile):
            raise TypeError("profile must be StationResidualRebaseProfile.")
        self.profile = profile

    def request_rebase(
        self,
        *,
        state: TrajectoryEditingState,
        intent: StationIntent,
        intent_receipt: StationIntentAdmissionReceipt,
        command_factory: Callable[[], StationResidualRebaseCommand],
    ) -> StationResidualRebaseCycle:
        """Do not evaluate the command supplier while the profile is off."""

        if not isinstance(state, TrajectoryEditingState):
            raise TypeError("state must be TrajectoryEditingState.")
        if not isinstance(intent, StationIntent):
            raise TypeError("intent must be StationIntent.")
        if not isinstance(intent_receipt, StationIntentAdmissionReceipt):
            raise TypeError(
                "intent_receipt must be StationIntentAdmissionReceipt."
            )
        if not callable(command_factory):
            raise TypeError("command_factory must be callable.")
        if not self.profile.enabled:
            return StationResidualRebaseCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        if (
            not intent_receipt.accepted
            or intent_receipt.intent_id != intent.intent_id
            or intent_receipt.intent_hash != intent.semantic_content_hash
            or intent_receipt.runtime_session_id != intent.runtime_session_id
            or intent_receipt.control_domain_id != intent.control_domain_id
            or intent_receipt.client_id != intent.client_id
            or intent_receipt.client_sequence != intent.client_sequence
            or intent_receipt.intent_kind != intent.intent_kind
            or intent_receipt.available_at != intent.available_at
            or intent_receipt.available_clock_domain_id
            != intent.available_clock_domain_id
            or intent_receipt.admission_sequence != intent.admission_sequence
            or intent_receipt.manifest_hash != intent.manifest_hash
            or intent_receipt.status != "residual_rebase_intent_accepted"
            or intent_receipt.state_revision_before
            != intent.expected_state_revision
            or intent_receipt.state_revision_after
            != intent_receipt.state_revision_before + 1
        ):
            raise ValueError("Residual rebase requires its accepted Station intent.")
        if intent.intent_kind != "request_residual_rebase_and_clear":
            raise ValueError("Station intent is not a residual rebase request.")
        if intent.subject_id != self.profile.composer_source_slot_id:
            raise ValueError("Residual rebase intent names another composer slot.")
        active_anchor = state.active_residual_anchor_binding
        if active_anchor is None or intent.resource_binding_id != (
            active_anchor.anchor_binding_id
        ):
            raise ValueError("Residual rebase intent names another active anchor.")
        if intent.operation_spec_id != self.profile.profile_id:
            raise ValueError("Residual rebase intent names another profile.")
        command = command_factory()
        if not isinstance(command, StationResidualRebaseCommand):
            raise TypeError(
                "command_factory must return StationResidualRebaseCommand."
            )
        if intent.available_clock_domain_id != (
            command.anchor_request.clock_domain_id
        ):
            raise ValueError(
                "Residual rebase intent and anchor request require one mapped clock."
            )
        if command.anchor_request.requested_at < intent.available_at:
            raise ValueError("Anchor request predates Station intent admission.")
        next_state = request_residual_rebase(
            state,
            anchor_request=command.anchor_request,
            effective_not_before_target_at=(
                command.effective_not_before_target_at
            ),
            capture_deadline=command.capture_deadline,
            trigger_kind=intent.intent_kind,
            trigger_fact_id=intent.intent_id,
            trigger_fact_hash=intent.semantic_content_hash,
            capture_mode=command.capture_mode,
            trigger_source_action_id=command.trigger_source_action_id,
            trigger_source_action_hash=command.trigger_source_action_hash,
        )
        request = next_state.pending_residual_rebase
        if request is None:
            raise RuntimeError("Residual rebase reducer did not retain its request.")
        return StationResidualRebaseCycle(
            profile_id=self.profile.profile_id,
            status="requested",
            state_before_hash=state.semantic_content_hash,
            state_after=next_state,
            rebase_request=request,
        )


__all__ = [
    "STATION_RESIDUAL_REBASE_COMMAND_HASH_DOMAIN_DEVELOPMENT",
    "STATION_RESIDUAL_REBASE_PROFILE_OFF",
    "StationResidualRebaseCommand",
    "StationResidualRebaseController",
    "StationResidualRebaseCycle",
    "StationResidualRebaseProfile",
]
