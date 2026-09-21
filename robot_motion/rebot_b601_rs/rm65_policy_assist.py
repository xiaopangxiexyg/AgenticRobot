"""Bridge admitted RM65 motion into bounded B601 policy-assist intent.

The existing :mod:`rm65_retarget` module remains the sole owner of coordinate
mapping and continuous-seed IK.  This bridge only translates its current-tick
result into the latest-value shared-control contract.  It owns no device,
servo loop, gravity model or actuator dispatch.

Any source, freshness, mapping, IK or publication rejection withdraws the
previous assist command immediately.  The B601 owner therefore returns to its
existing gravity/damping behavior on the next control tick instead of holding
or replaying an older policy target.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .rm65_retarget import (
    RM65FreeDriveSample,
    RM65ToB601Anchor,
    RM65ToB601Retargeter,
    RM65ToB601RetargetResult,
)
from .shared_control import B601PolicyAssistCommand, B601PolicyCommandReceipt
from .shared_control_runtime import B601SharedControlRuntime

_ZERO_SIX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RM65ToB601PolicyAssistResult:
    """Audit result for one source sample at the policy/owner boundary."""

    retarget_result: RM65ToB601RetargetResult
    policy_command: B601PolicyAssistCommand | None
    policy_receipt: B601PolicyCommandReceipt | None
    accepted: bool
    reason: str
    revoked_policy_sequence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.retarget_result, RM65ToB601RetargetResult):
            raise TypeError("retarget_result must be RM65ToB601RetargetResult")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("reason must not be empty")
        if self.accepted:
            if self.policy_command is None or self.policy_receipt is None:
                raise ValueError("accepted results require a policy command and receipt")
            if self.policy_receipt.accepted is not True:
                raise ValueError("accepted results require an accepted policy receipt")
            if self.revoked_policy_sequence is not None:
                raise ValueError("accepted results cannot revoke a policy sequence")
        elif self.policy_receipt is not None and self.policy_receipt.accepted:
            raise ValueError("rejected results cannot carry an accepted receipt")
        if self.revoked_policy_sequence is not None and (
            isinstance(self.revoked_policy_sequence, bool)
            or not isinstance(self.revoked_policy_sequence, int)
            or self.revoked_policy_sequence < 0
        ):
            raise ValueError(
                "revoked_policy_sequence must be a non-negative integer"
            )
        object.__setattr__(self, "reason", reason)


class RM65ToB601PolicyAssistBridge:
    """Reuse reverse retargeting to feed one B601 shared-control runtime."""

    def __init__(
        self,
        retargeter: RM65ToB601Retargeter,
        shared_control: B601SharedControlRuntime,
    ) -> None:
        if not isinstance(retargeter, RM65ToB601Retargeter):
            raise TypeError("retargeter must be RM65ToB601Retargeter")
        if not isinstance(shared_control, B601SharedControlRuntime):
            raise TypeError("shared_control must be B601SharedControlRuntime")
        self.retargeter = retargeter
        self.shared_control = shared_control

    def capture_anchor(
        self,
        sample: RM65FreeDriveSample,
        *,
        measured_b601_qpos_rad: Sequence[float],
    ) -> RM65ToB601Anchor:
        """Recenter on measured B601 state and leave policy assist withdrawn."""

        self.shared_control.clear_policy()
        return self.retargeter.capture_anchor(
            sample,
            target_qpos_rad=measured_b601_qpos_rad,
        )

    def set_policy_clutch(self, active: bool) -> None:
        """Apply explicit override; release still requires a fresh recenter."""

        self.shared_control.set_policy_clutch(active)
        if active:
            self.shared_control.clear_policy()
            self.retargeter.clear_anchor()

    def process_sample(
        self,
        sample: RM65FreeDriveSample,
        *,
        measured_b601_qpos_rad: Sequence[float],
    ) -> RM65ToB601PolicyAssistResult:
        """Publish only the current accepted IK result; never retain fallback."""

        result = self.retargeter.retarget(
            sample,
            seed_qpos_rad=measured_b601_qpos_rad,
        )
        if not result.accepted:
            revoked = self.shared_control.clear_policy()
            return RM65ToB601PolicyAssistResult(
                retarget_result=result,
                policy_command=None,
                policy_receipt=None,
                accepted=False,
                reason=f"retarget_rejected:{result.rejection_reason}",
                revoked_policy_sequence=revoked,
            )

        if result.qpos_rad is None:
            raise RuntimeError("accepted retarget result omitted qpos_rad")
        receive_timestamp = float(self.retargeter.monotonic_clock())
        if not math.isfinite(receive_timestamp) or receive_timestamp < 0.0:
            revoked = self.shared_control.clear_policy()
            return RM65ToB601PolicyAssistResult(
                retarget_result=result,
                policy_command=None,
                policy_receipt=None,
                accepted=False,
                reason="policy_publish_rejected:invalid_local_monotonic_timestamp",
                revoked_policy_sequence=revoked,
            )
        try:
            command = B601PolicyAssistCommand(
                sequence=sample.sequence,
                source_timestamp_sec=sample.receive_timestamp_monotonic_sec,
                receive_timestamp_sec=receive_timestamp,
                qpos_target_rad=result.qpos_rad,
                qvel_target_rad_s=_ZERO_SIX,
            )
        except (TypeError, ValueError) as exc:
            revoked = self.shared_control.clear_policy()
            return RM65ToB601PolicyAssistResult(
                retarget_result=result,
                policy_command=None,
                policy_receipt=None,
                accepted=False,
                reason=f"policy_publish_rejected:{type(exc).__name__}:{exc}",
                revoked_policy_sequence=revoked,
            )
        receipt = self.shared_control.publish_policy(command)
        if not receipt.accepted:
            revoked = self.shared_control.clear_policy()
            return RM65ToB601PolicyAssistResult(
                retarget_result=result,
                policy_command=command,
                policy_receipt=receipt,
                accepted=False,
                reason=f"policy_publish_rejected:{receipt.reason}",
                revoked_policy_sequence=revoked,
            )
        return RM65ToB601PolicyAssistResult(
            retarget_result=result,
            policy_command=command,
            policy_receipt=receipt,
            accepted=True,
            reason="accepted",
        )


__all__ = [
    "RM65ToB601PolicyAssistBridge",
    "RM65ToB601PolicyAssistResult",
]
