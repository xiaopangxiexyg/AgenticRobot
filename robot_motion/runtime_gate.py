from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


RECOVERY_IK_STATUSES = frozenset({"smoothing_required", "failed", "recovering", "hold"})
DEPLOYMENT_DUAL_RATE_IK_SOURCE = "deployment_dual_rate_ik"
DEPLOYMENT_DUAL_RATE_HOLD_METHOD = "deployment_dual_rate_hold_last_safe"
BACKEND_SAFE_HOLD_STATUS = "backend_safe_hold"


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


def _nonblocking_backend_hold(metadata: Mapping[str, Any], status: str) -> bool:
    """Identify a dual-rate backend safe hold that must not gate policy output.

    The impedance backend deliberately returns the previous safe command when
    the full worker is busy or the bounded local step has no candidate.  The
    backend remains fail-closed in both cases; blocking the outer policy queue
    would deadlock it on the same target and prevent a later safe result from
    being applied.
    """

    if status != "hold":
        return False
    return bool(
        str(metadata.get("source") or "").strip() == DEPLOYMENT_DUAL_RATE_IK_SOURCE
        or str(metadata.get("method") or "").strip() == DEPLOYMENT_DUAL_RATE_HOLD_METHOD
        or metadata.get("hold_reason") == "full_ik_worker_in_flight"
    )


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
        # A deployment dual-rate hold is already fail-closed inside the
        # impedance backend.  It must not block the outer policy queue: doing
        # so would keep the backend on the same unsolved target forever.
        if _nonblocking_backend_hold(metadata, status):
            status = BACKEND_SAFE_HOLD_STATUS
        if status:
            statuses[str(side)] = status
    return statuses


def arm_ik_recovery_diagnostics(
    arm_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return compact, serializable diagnostics for recovery-side metadata.

    Full impedance records contain large nested timing/debug payloads.  The
    deploy log only needs the fields that identify why the gate is active and
    whether the record is advancing.
    """

    diagnostics: dict[str, dict[str, Any]] = {}
    for side, state in arm_states.items():
        metadata = extract_ik_metadata(state)
        status = str(metadata.get("status") or metadata.get("ik_status") or "").strip()
        jump_status = str(metadata.get("ik_jump_gate_status") or "").strip()
        singularity_status = str(metadata.get("singularity_gate_status") or "").strip()
        if not status and jump_status in {"hold_entered", "holding"}:
            status = "hold"
        if not status and singularity_status == "hold_entry":
            status = "hold"
        if not status:
            continue
        gate_status = status
        if _nonblocking_backend_hold(metadata, status):
            gate_status = BACKEND_SAFE_HOLD_STATUS
        diagnostics[str(side)] = {
            "status": status,
            "gate_status": gate_status,
            "source": metadata.get("source"),
            "method": metadata.get("method"),
            "result_message": metadata.get("result_message"),
            "candidate_source": metadata.get("candidate_source"),
            "last_code": metadata.get("last_code"),
            "reason": metadata.get("reason"),
            "hold_reason": metadata.get("hold_reason"),
            "ik_jump_gate_status": jump_status or None,
            "singularity_gate_status": singularity_status or None,
            "measurement_timestamp": state.get("hardware_measurement_timestamp"),
            "measurement_sequence": state.get("hardware_measurement_sequence"),
        }
    return diagnostics


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
    last_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_event: str = ""

    def update(self, arm_states: Mapping[str, Mapping[str, Any]]) -> str:
        statuses = arm_ik_recovery_statuses(arm_states)
        self.last_diagnostics = arm_ik_recovery_diagnostics(arm_states)
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
            detail_items: list[str] = []
            for side, status in sorted(self.last_statuses.items()):
                if status not in self.recovery_statuses:
                    continue
                diagnostic = self.last_diagnostics.get(side, {})
                reason = (
                    diagnostic.get("hold_reason")
                    or diagnostic.get("reason")
                    or diagnostic.get("candidate_source")
                )
                suffix = f"({reason})" if reason else ""
                detail_items.append(f"{side}:{status}{suffix}")
            details = ",".join(detail_items)
            suffix = f"[{details}]" if details else ""
            return f"ik_recovery=active{suffix}(discard_policy_chunks)"
        if self.waiting_for_fresh_policy:
            return "ik_recovery=stable(wait_fresh_policy_chunk)"
        return ""
