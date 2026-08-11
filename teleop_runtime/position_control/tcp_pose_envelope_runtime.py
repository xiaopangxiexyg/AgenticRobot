"""Real-runtime adapter for the backend-neutral TCP-pose envelope.

The target filter sits on the existing coupled pre-IK target-filter seam.  It
converts the retargeter's solver-frame target to the configured ``tool_tcp``,
projects that pose once, then converts the result back to the solver frame.
It never calls IK or FK and it does not participate in 120 Hz qpos dispatch.

Anchor capture consumes an already-produced authoritative TCP observation.
The current RM65 observation contract labels that value
``hardware_qpos_fk``; this adapter does not recompute it.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeProjectionResult,
    project_tcp_pose_envelope,
)


TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.real_tcp_pose_target_frame_binding.development"
)
TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.real_tcp_pose_tool_frame_binding.development"
)
AUTHORITATIVE_TCP_POSE_SOURCE = "hardware_qpos_fk"


class TCPPoseEnvelopeRuntimeError(RuntimeError):
    """Fail-closed runtime error raised by an Active envelope filter."""


def _required_text(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite rigid 4x4 pose") from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite rigid 4x4 pose")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def _normalized_sides(sides: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_required_text(side, name="side") for side in sides)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("sides must be non-empty and unique")
    return normalized


def target_frame_binding_hash_from_calibration(
    calibration: Any,
    *,
    sides: Sequence[str],
    reference_frame_by_side: Mapping[str, str],
) -> str:
    """Hash the exact transforms used to express coupled targets in world."""

    normalized = _normalized_sides(sides)
    payload: dict[str, object] = {
        "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
        "reference_frame_by_side": {},
        "target_world_from_base_by_side": {},
    }
    references = payload["reference_frame_by_side"]
    transforms = payload["target_world_from_base_by_side"]
    assert isinstance(references, dict)
    assert isinstance(transforms, dict)
    for side in normalized:
        frame = _required_text(
            reference_frame_by_side.get(side),
            name=f"reference_frame_by_side.{side}",
        )
        try:
            arm = calibration.get_arm_for_side(side)
        except Exception as exc:  # noqa: BLE001 - convert to one binding failure.
            raise ValueError(f"calibration is missing side {side!r}") from exc
        transform = _pose(
            getattr(arm, "T_D_B", None),
            name=f"calibration.{side}.T_D_B",
        )
        references[side] = frame
        transforms[side] = transform.tolist()
    return canonical_content_hash(
        payload,
        domain=TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
    )


def tool_frame_binding_hashes_from_retargeter(
    retargeter: Any,
    *,
    sides: Sequence[str],
) -> dict[str, str]:
    """Hash the exact fixed solver→tool_tcp transforms used by the retargeter."""

    normalized = _normalized_sides(sides)
    convert = getattr(
        retargeter,
        "tool_tcp_world_target_from_solver_world_target",
        None,
    )
    if not callable(convert):
        raise TypeError("retargeter lacks solver-to-tool_tcp conversion")
    result: dict[str, str] = {}
    identity = np.eye(4, dtype=float)
    for side in normalized:
        transform = _pose(
            convert(side=side, solver_world_target=identity),
            name=f"retargeter.{side}.solver_to_tool_tcp",
        )
        result[side] = canonical_content_hash(
            {
                "side": side,
                "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
                "solver_to_tool_tcp": transform.tolist(),
            },
            domain=TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
    return result


def capture_tcp_pose_envelope_anchor(
    *,
    anchor_id: str,
    profile: TCPPoseEnvelopeProfile,
    calibration: Any,
    actual_state_by_side: Mapping[str, Mapping[str, object]],
    execution_epoch: int,
    clock_domain_id: str,
    max_pair_skew_sec: float = 0.05,
) -> TCPPoseEnvelopeAnchorReceipt:
    """Capture a world-frame anchor from existing authoritative TCP snapshots."""

    max_skew = float(max_pair_skew_sec)
    if not math.isfinite(max_skew) or max_skew < 0.0:
        raise ValueError("max_pair_skew_sec must be finite and non-negative")
    required_sides = profile.anchor_required_sides
    if not required_sides:
        raise ValueError("profile does not require an anchor")

    actual_target_hash = target_frame_binding_hash_from_calibration(
        calibration,
        sides=profile.enabled_sides,
        reference_frame_by_side=profile.reference_frame_by_side,
    )
    if actual_target_hash != profile.target_frame_binding_hash:
        raise TCPPoseEnvelopeRuntimeError("target-frame binding changed before anchor capture")

    world_pose_by_side: dict[str, np.ndarray] = {}
    captured_at_by_side: dict[str, float] = {}
    for side in required_sides:
        state = actual_state_by_side.get(side)
        if not isinstance(state, Mapping):
            raise TCPPoseEnvelopeRuntimeError(
                f"authoritative TCP state is missing for side {side!r}"
            )
        source = str(state.get("actual_tcp_pose_source") or "")
        if source != AUTHORITATIVE_TCP_POSE_SOURCE:
            raise TCPPoseEnvelopeRuntimeError(
                f"authoritative TCP source mismatch for side {side!r}: {source!r}"
            )
        pose_base = _pose(
            state.get("actual_tcp_pose"),
            name=f"actual_state_by_side.{side}.actual_tcp_pose",
        )
        try:
            captured_at = float(state.get("actual_tcp_pose_timestamp"))
        except (TypeError, ValueError) as exc:
            raise TCPPoseEnvelopeRuntimeError(
                f"authoritative TCP timestamp is missing for side {side!r}"
            ) from exc
        if not math.isfinite(captured_at):
            raise TCPPoseEnvelopeRuntimeError(
                f"authoritative TCP timestamp is invalid for side {side!r}"
            )
        arm = calibration.get_arm_for_side(side)
        target_world_from_base = _pose(
            getattr(arm, "T_D_B", None),
            name=f"calibration.{side}.T_D_B",
        )
        world_pose_by_side[side] = target_world_from_base @ pose_base
        captured_at_by_side[side] = captured_at

    receipt = TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id=anchor_id,
        profile=profile,
        execution_epoch=execution_epoch,
        clock_domain_id=clock_domain_id,
        tcp_pose_by_side=world_pose_by_side,
        captured_at_by_side=captured_at_by_side,
    )
    if receipt.pair_skew_sec > max_skew:
        raise TCPPoseEnvelopeRuntimeError(
            "authoritative TCP anchor pair skew exceeds limit: "
            f"{receipt.pair_skew_sec:.6f}s > {max_skew:.6f}s"
        )
    return receipt


@dataclass(frozen=True)
class TCPPoseEnvelopeFilterSnapshot:
    frame_count: int
    apply_to_target: bool
    last_error: str | None
    projection: dict[str, object] | None


class TCPPoseEnvelopeTargetFilter:
    """Composable pre-IK TCP envelope observer or Active target projector."""

    def __init__(
        self,
        *,
        retargeter: Any,
        profile: TCPPoseEnvelopeProfile,
        execution_epoch: int,
        anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
        previous_filter: Callable[..., Mapping[str, np.ndarray] | None] | None,
        apply_to_target: bool = False,
    ) -> None:
        if not isinstance(profile, TCPPoseEnvelopeProfile):
            raise TypeError("profile must be TCPPoseEnvelopeProfile")
        self.retargeter = retargeter
        self.profile = profile
        self.execution_epoch = int(execution_epoch)
        self.anchor_receipt = anchor_receipt
        self.previous_filter = previous_filter
        self.apply_to_target = bool(apply_to_target)
        self._tool_frame_binding_hash_by_side = (
            tool_frame_binding_hashes_from_retargeter(
                retargeter,
                sides=profile.enabled_sides,
            )
        )
        for side in profile.enabled_sides:
            if (
                self._tool_frame_binding_hash_by_side.get(side)
                != profile.tool_frame_binding_hash_by_side[side]
            ):
                raise TCPPoseEnvelopeRuntimeError(
                    f"tool-frame binding changed before filter installation for {side!r}"
                )
        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_error: str | None = None
        self._last_projection: TCPPoseEnvelopeProjectionResult | None = None

    def _tool_targets(
        self,
        targets: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        convert = getattr(
            self.retargeter,
            "tool_tcp_world_target_from_solver_world_target",
            None,
        )
        if not callable(convert):
            raise TypeError("retargeter lacks solver-to-tool_tcp conversion")
        return {
            side: _pose(
                convert(side=side, solver_world_target=targets[side]),
                name=f"tool_tcp_target.{side}",
            )
            for side in targets
        }

    def _solver_targets(
        self,
        targets: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        convert = getattr(
            self.retargeter,
            "solver_world_target_from_tool_tcp_world_target",
            None,
        )
        if not callable(convert):
            raise TypeError("retargeter lacks tool_tcp-to-solver conversion")
        return {
            side: _pose(
                convert(side=side, tool_tcp_world_target=targets[side]),
                name=f"solver_target.{side}",
            )
            for side in targets
        }

    def __call__(
        self,
        targets: Mapping[str, np.ndarray],
        observation: Any,
        calibration: Any,
    ) -> Mapping[str, np.ndarray] | None:
        prior_result = (
            None
            if self.previous_filter is None
            else self.previous_filter(targets, observation, calibration)
        )
        effective_targets = targets if prior_result is None else prior_result
        try:
            target_hash = target_frame_binding_hash_from_calibration(
                calibration,
                sides=self.profile.enabled_sides,
                reference_frame_by_side=self.profile.reference_frame_by_side,
            )
            tool_targets = self._tool_targets(effective_targets)
            projection = project_tcp_pose_envelope(
                tool_targets,
                profile=self.profile,
                target_frame_binding_hash=target_hash,
                tool_frame_binding_hash_by_side=(
                    self._tool_frame_binding_hash_by_side
                ),
                controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
                reference_frame_by_side=self.profile.reference_frame_by_side,
                execution_epoch=self.execution_epoch,
                anchor_receipt=self.anchor_receipt,
            )
            applied_targets = self._solver_targets(
                projection.effective_tcp_pose_by_side
            )
        except Exception as exc:  # noqa: BLE001 - Shadow records; Active fails closed.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._frame_count += 1
                self._last_error = error
                self._last_projection = None
            if self.apply_to_target:
                if isinstance(exc, TCPPoseEnvelopeRuntimeError):
                    raise
                raise TCPPoseEnvelopeRuntimeError(error) from exc
            return prior_result

        with self._lock:
            self._frame_count += 1
            self._last_error = None
            self._last_projection = projection
        if not self.apply_to_target:
            return prior_result
        return applied_targets

    def snapshot(self) -> TCPPoseEnvelopeFilterSnapshot:
        with self._lock:
            projection = self._last_projection
            return TCPPoseEnvelopeFilterSnapshot(
                frame_count=int(self._frame_count),
                apply_to_target=bool(self.apply_to_target),
                last_error=self._last_error,
                projection=(
                    None
                    if projection is None
                    else copy.deepcopy(projection.to_payload())
                ),
            )


def install_tcp_pose_envelope_filter(
    *,
    retargeter: Any,
    profile: TCPPoseEnvelopeProfile,
    execution_epoch: int,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
    apply_to_target: bool = False,
) -> TCPPoseEnvelopeTargetFilter:
    """Install one outermost filter without replacing an existing modifier."""

    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        execution_epoch=execution_epoch,
        anchor_receipt=anchor_receipt,
        previous_filter=getattr(retargeter, "target_override_filter", None),
        apply_to_target=apply_to_target,
    )
    retargeter.target_override_filter = observer
    return observer


__all__ = [
    "AUTHORITATIVE_TCP_POSE_SOURCE",
    "TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "TCPPoseEnvelopeFilterSnapshot",
    "TCPPoseEnvelopeRuntimeError",
    "TCPPoseEnvelopeTargetFilter",
    "capture_tcp_pose_envelope_anchor",
    "install_tcp_pose_envelope_filter",
    "target_frame_binding_hash_from_calibration",
    "tool_frame_binding_hashes_from_retargeter",
]
