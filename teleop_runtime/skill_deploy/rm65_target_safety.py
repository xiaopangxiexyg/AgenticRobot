"""Startup binding for the independent RM65 Skill target-safety owner."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.target_safety.tcp_pose_envelope import (
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
)
from teleop_runtime.target_safety.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeTargetFrameBinding,
    TCPPoseEnvelopeTargetFilter,
    install_tcp_pose_envelope_filter,
    load_tcp_pose_envelope_profile_from_target_frame_binding,
)

from .control_runtime import (
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)


class RM65SkillTargetSafetyBuilder(Protocol):
    """Shared coupled solver surface needed for startup safety binding."""

    pre_ik_target_safety_filter: object | None

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class ExclusiveRM65SkillStartupSafetyReceipt:
    """Evidence that Active target safety was bound after lower HOLD."""

    envelope_profile_hash: str
    anchor_receipt_hash: str | None
    captured_at: float
    hold_evidence_refs: tuple[str, ...]
    observation_evidence_refs: tuple[str, ...]
    semantic_content_hash: str


class ExclusiveRM65SkillTargetSafetyRuntime:
    """Own one Active pre-IK envelope for an independent Skill process."""

    def __init__(
        self,
        *,
        target_builder: RM65SkillTargetSafetyBuilder,
        target_frame_binding: TCPPoseEnvelopeTargetFrameBinding,
        envelope_profile: TCPPoseEnvelopeProfile,
        envelope_filter: TCPPoseEnvelopeTargetFilter,
        execution_epoch: int = 1,
        clock_domain_id: str = "python-time-time-wall-clock",
    ) -> None:
        if not isinstance(
            target_frame_binding,
            TCPPoseEnvelopeTargetFrameBinding,
        ):
            raise TypeError("target_frame_binding must be typed.")
        if not isinstance(envelope_profile, TCPPoseEnvelopeProfile):
            raise TypeError("envelope_profile must be typed.")
        if not isinstance(envelope_filter, TCPPoseEnvelopeTargetFilter):
            raise TypeError("envelope_filter must be typed.")
        if envelope_filter.retargeter is not target_builder:
            raise ValueError("target builder and envelope filter must be identical.")
        if envelope_filter.profile is not envelope_profile:
            raise ValueError("envelope filter must use the exact selected profile.")
        if envelope_filter.target_frame_binding is not target_frame_binding:
            raise ValueError(
                "envelope filter must use the exact selected target frame binding."
            )
        if not envelope_filter.apply_to_target:
            raise ValueError("independent real Skill envelope must be Active.")
        epoch = int(execution_epoch)
        if isinstance(execution_epoch, bool) or epoch <= 0:
            raise ValueError("execution_epoch must be a positive integer.")
        clock_id = str(clock_domain_id).strip()
        if not clock_id:
            raise ValueError("clock_domain_id must be non-empty.")
        self.target_builder = target_builder
        self.target_frame_binding = target_frame_binding
        self.envelope_profile = envelope_profile
        self.envelope_filter = envelope_filter
        self.execution_epoch = epoch
        self.clock_domain_id = clock_id
        self._startup_receipt: ExclusiveRM65SkillStartupSafetyReceipt | None = None

    @property
    def startup_receipt(self) -> ExclusiveRM65SkillStartupSafetyReceipt | None:
        return self._startup_receipt

    @classmethod
    def from_profile_path(
        cls,
        *,
        target_builder: RM65SkillTargetSafetyBuilder,
        station_world_from_arm_base_by_side: Mapping[str, np.ndarray],
        station_world_frame_id: str,
        envelope_profile_path: str | Path,
        execution_epoch: int = 1,
        clock_domain_id: str = "python-time-time-wall-clock",
    ) -> ExclusiveRM65SkillTargetSafetyRuntime:
        """Bind the final pre-IK envelope to lower-owned Station geometry."""

        target_frame_binding = TCPPoseEnvelopeTargetFrameBinding(
            reference_frame_by_side={
                side: station_world_frame_id for side in ("left", "right")
            },
            target_world_from_base_by_side=station_world_from_arm_base_by_side,
        )
        profile = load_tcp_pose_envelope_profile_from_target_frame_binding(
            envelope_profile_path,
            target_frame_binding=target_frame_binding,
            retargeter=target_builder,
            active_sides=("left", "right"),
        )
        filter_owner = install_tcp_pose_envelope_filter(
            retargeter=target_builder,
            profile=profile,
            target_frame_binding=target_frame_binding,
            execution_epoch=int(execution_epoch),
            anchor_receipt=None,
            apply_to_target=True,
        )
        return cls(
            target_builder=target_builder,
            target_frame_binding=target_frame_binding,
            envelope_profile=profile,
            envelope_filter=filter_owner,
            execution_epoch=execution_epoch,
            clock_domain_id=clock_domain_id,
        )

    def bind_startup_anchor(
        self,
        *,
        observation_cut: RealSkillMotionObservationCut,
        startup_hold_receipt: RealSkillMotionHoldReceipt,
        captured_at: float,
    ) -> ExclusiveRM65SkillStartupSafetyReceipt:
        """Bind the live pose only after an independently confirmed lower HOLD."""

        if self._startup_receipt is not None:
            raise RuntimeError("startup target safety is already bound.")
        if not isinstance(observation_cut, RealSkillMotionObservationCut):
            raise TypeError("observation_cut must be typed.")
        if not isinstance(startup_hold_receipt, RealSkillMotionHoldReceipt):
            raise TypeError("startup_hold_receipt must be typed.")
        if not startup_hold_receipt.succeeded:
            raise RuntimeError("startup target safety requires lower-confirmed HOLD.")
        timestamp = float(captured_at)
        if not math.isfinite(timestamp) or timestamp < observation_cut.available_at:
            raise ValueError(
                "captured_at must be finite and not precede observation availability."
            )
        qpos = observation_cut.qpos_by_side
        tool_targets = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
        )
        required_sides = tuple(self.envelope_profile.anchor_required_sides)
        if not set(required_sides).issubset(tool_targets):
            raise RuntimeError("target builder omitted a required TCP anchor side.")
        anchor = (
            None
            if not required_sides
            else TCPPoseEnvelopeAnchorReceipt.capture(
                anchor_id=f"exclusive_skill_startup:{self.execution_epoch}",
                profile=self.envelope_profile,
                execution_epoch=self.execution_epoch,
                clock_domain_id=self.clock_domain_id,
                tcp_pose_by_side={
                    side: tool_targets[side] for side in required_sides
                },
                captured_at_by_side={side: timestamp for side in required_sides},
            )
        )
        self.envelope_filter.rebind(
            execution_epoch=self.execution_epoch,
            anchor_receipt=anchor,
        )
        anchor_hash = None if anchor is None else anchor.semantic_content_hash
        payload = {
            "envelope_profile_hash": self.envelope_profile.semantic_content_hash,
            "anchor_receipt_hash": anchor_hash,
            "captured_at": timestamp,
            "hold_evidence_refs": list(startup_hold_receipt.evidence_refs),
            "observation_evidence_refs": list(observation_cut.evidence_refs),
        }
        receipt = ExclusiveRM65SkillStartupSafetyReceipt(
            envelope_profile_hash=self.envelope_profile.semantic_content_hash,
            anchor_receipt_hash=anchor_hash,
            captured_at=timestamp,
            hold_evidence_refs=startup_hold_receipt.evidence_refs,
            observation_evidence_refs=observation_cut.evidence_refs,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain="dexteleop.real_skill.startup_target_safety_receipt.v1",
            ),
        )
        self._startup_receipt = receipt
        return receipt


class ExclusiveRM65SkillStartupSafetyBinder:
    """Bind one ports/solver/safety graph before Gateway transport opens."""

    def __init__(
        self,
        *,
        ports: Any,
        target_safety: ExclusiveRM65SkillTargetSafetyRuntime,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # Avoid a reverse import from this target-safety module into the ports
        # owner while still enforcing one exact lower graph at construction.
        for name in ("observe", "synchronize_target_builder"):
            if not callable(getattr(ports, name, None)):
                raise TypeError(f"ports must expose {name}.")
        if getattr(ports, "target_builder", None) is not target_safety.target_builder:
            raise ValueError("ports and target safety must share one target builder.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.ports = ports
        self.target_safety = target_safety
        self._clock = clock

    def __call__(self) -> ExclusiveRM65SkillStartupSafetyReceipt:
        if self.target_safety.startup_receipt is not None:
            raise RuntimeError("startup target safety is already bound.")
        hold_receipt = getattr(self.ports, "startup_hold_receipt", None)
        if not isinstance(hold_receipt, RealSkillMotionHoldReceipt):
            raise RuntimeError("RM65 Skill startup HOLD receipt is unavailable.")
        if not hold_receipt.succeeded:
            raise RuntimeError("RM65 Skill startup HOLD was not confirmed.")
        cut = self.ports.observe()
        self.ports.synchronize_target_builder(cut.qpos_by_side)
        return self.target_safety.bind_startup_anchor(
            observation_cut=cut,
            startup_hold_receipt=hold_receipt,
            captured_at=max(float(self._clock()), cut.available_at),
        )


__all__ = [
    "ExclusiveRM65SkillStartupSafetyReceipt",
    "ExclusiveRM65SkillStartupSafetyBinder",
    "ExclusiveRM65SkillTargetSafetyRuntime",
    "RM65SkillTargetSafetyBuilder",
]
