from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


RECOVERY_IK_STATUSES = frozenset({"smoothing_required", "failed", "recovering", "hold"})


def extract_ik_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    direct = state.get("ik_metadata")
    if isinstance(direct, Mapping):
        return dict(direct)
    record = state.get("last_impedance_record")
    if isinstance(record, Mapping):
        nested = record.get("ik_metadata")
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def arm_ik_recovery_statuses(arm_states: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for side, state in arm_states.items():
        metadata = extract_ik_metadata(state)
        status = str(metadata.get("status") or metadata.get("ik_status") or "").strip()
        jump_status = str(metadata.get("ik_jump_gate_status") or "").strip()
        singularity_status = str(metadata.get("singularity_gate_status") or "").strip()
        if not status and jump_status in {"hold_entered", "holding"}:
            status = "hold"
        if not status and singularity_status == "hold_entry":
            status = "hold"
        if status:
            statuses[str(side)] = status
    return statuses


@dataclass
class IKActionChunkGate:
    """Gate deploy action chunks around IK smoothing/recovery.

    Observations and policy inference can continue while recovery is active, but
    predicted chunks are discarded. Once recovery is stable, any old queued
    chunk remains invalid until one fresh policy prediction has been produced
    from the post-recovery observation.
    """

    stable_frames_required: int = 3
    recovery_statuses: frozenset[str] = RECOVERY_IK_STATUSES
    recovery_active: bool = False
    waiting_for_fresh_policy: bool = False
    stable_frames: int = 0
    last_statuses: dict[str, str] = field(default_factory=dict)
    last_event: str = ""

    def update(self, arm_states: Mapping[str, Mapping[str, Any]]) -> str:
        statuses = arm_ik_recovery_statuses(arm_states)
        active = any(status in self.recovery_statuses for status in statuses.values())
        self.last_statuses = statuses
        self.last_event = ""

        if active:
            if not self.recovery_active:
                self.last_event = "ik_recovery_started"
            self.recovery_active = True
            self.waiting_for_fresh_policy = True
            self.stable_frames = 0
            return self.last_event

        if self.recovery_active:
            self.stable_frames += 1
            if self.stable_frames >= max(1, int(self.stable_frames_required)):
                self.recovery_active = False
                self.waiting_for_fresh_policy = True
                self.last_event = "ik_recovery_stable_waiting_fresh_policy"
            return self.last_event

        return ""

    @property
    def discards_policy_output(self) -> bool:
        return bool(self.recovery_active)

    @property
    def blocks_action_output(self) -> bool:
        return bool(self.recovery_active or self.waiting_for_fresh_policy)

    def mark_policy_output_accepted(self) -> None:
        if not self.recovery_active:
            self.waiting_for_fresh_policy = False

    def status_text(self) -> str:
        if self.recovery_active:
            return "ik_recovery=active(discard_policy_chunks)"
        if self.waiting_for_fresh_policy:
            return "ik_recovery=stable(wait_fresh_policy_chunk)"
        return ""
