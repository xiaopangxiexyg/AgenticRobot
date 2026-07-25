"""Target-neutral cadence and scheduling profile for one RTC producer.

Application execution settings such as position-reference cadence, command
cadence, lookahead, and action-send authorization deliberately do not belong
to this contract.  An embedding real or sim application may compose those
settings around this producer profile without changing its identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..inference import POLICY_INFERENCE_MODES_V1

if TYPE_CHECKING:
    from ..async_producer import PolicyAsyncExecutionProfile


RTC_PRODUCER_PROFILE_HASH_DOMAIN_V1 = "dexteleop.deploy.rtc_producer_profile.v1"


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _positive_finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


@dataclass(frozen=True)
class RtcProducerProfile:
    """One model-, embodiment-, sim/real-, and target-neutral RTC schedule."""

    profile_id: str
    rtc_mode: str
    observation_hz: float
    decision_hz: float
    action_hz: float
    request_trigger_mode: str
    due_pop_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            _nonempty(self.profile_id, name="profile_id"),
        )
        mode = _nonempty(self.rtc_mode, name="rtc_mode")
        if mode not in POLICY_INFERENCE_MODES_V1:
            raise ValueError(f"Unsupported RTC mode: {mode!r}.")
        object.__setattr__(self, "rtc_mode", mode)
        for name in ("observation_hz", "decision_hz", "action_hz"):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "request_trigger_mode",
            _nonempty(self.request_trigger_mode, name="request_trigger_mode"),
        )
        object.__setattr__(
            self,
            "due_pop_mode",
            _nonempty(self.due_pop_mode, name="due_pop_mode"),
        )

    @property
    def action_period_sec(self) -> float:
        return 1.0 / self.action_hz

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "rtc_mode": self.rtc_mode,
                "observation_hz": self.observation_hz,
                "decision_hz": self.decision_hz,
                "action_hz": self.action_hz,
                "request_trigger_mode": self.request_trigger_mode,
                "due_pop_mode": self.due_pop_mode,
            },
            domain=RTC_PRODUCER_PROFILE_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_legacy_execution_profile(
        cls,
        profile: "PolicyAsyncExecutionProfile",
    ) -> "RtcProducerProfile":
        """Project the old composed profile without target/application fields."""

        # Runtime import avoids making the legacy module depend on this package
        # during its own initialization.
        from ..async_producer import PolicyAsyncExecutionProfile

        if not isinstance(profile, PolicyAsyncExecutionProfile):
            raise TypeError("profile must be a PolicyAsyncExecutionProfile.")
        return cls(
            profile_id=profile.profile_id,
            rtc_mode=profile.rtc_mode,
            observation_hz=profile.observation_hz,
            decision_hz=profile.decision_hz,
            action_hz=profile.action_hz,
            request_trigger_mode=profile.request_trigger_mode,
            due_pop_mode=profile.due_pop_mode,
        )


__all__ = [
    "RTC_PRODUCER_PROFILE_HASH_DOMAIN_V1",
    "RtcProducerProfile",
]
