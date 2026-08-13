"""Deployment adapter for the shared final tool-TCP pose envelope.

Deployment targets arrive in each arm's runtime-raw base frame.  This adapter
uses the already-bound pose-frame contract to express them in canonical desk,
projects the target pose once, and converts only modified sides back to runtime
raw before any IK.  It never calls IK or FK and is not part of the 120 Hz qpos
dispatch path.

Relative bounds bind to one fresh ``actual_tcp_pose`` observation at an
explicit deployment lifecycle cut.  The adapter consumes that pose contract;
it never accepts or derives an anchor from qpos itself.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError
from robot_motion.target_safety.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeProjectionFeedback,
    TCPPoseEnvelopeProjectionResult,
    project_tcp_pose_envelope,
    project_tcp_pose_envelope_side,
)

from .tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeFileSpec,
    TCPPoseEnvelopeRuntimeError,
)


DEPLOY_TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.deploy_tcp_pose_target_frame_binding.development"
)
DEPLOY_TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.deploy_tcp_pose_tool_frame_binding.development"
)


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise TCPPoseEnvelopeRuntimeError(
            f"{name} must be a finite rigid 4x4 pose"
        ) from exc
    if not np.all(np.isfinite(matrix)):
        raise TCPPoseEnvelopeRuntimeError(
            f"{name} must be a finite rigid 4x4 pose"
        )
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise TCPPoseEnvelopeRuntimeError(f"{name} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise TCPPoseEnvelopeRuntimeError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise TCPPoseEnvelopeRuntimeError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def _pose_frame_contract(value: object) -> PoseFrameContract:
    if isinstance(value, PoseFrameContract):
        return value
    if isinstance(value, Mapping):
        try:
            return PoseFrameContract.from_dict(value)
        except PoseFrameContractError as exc:
            raise TCPPoseEnvelopeRuntimeError(
                f"invalid deployment pose-frame contract: {exc}"
            ) from exc
    raise TCPPoseEnvelopeRuntimeError(
        "deployment TCP pose envelope requires a runtime pose-frame contract"
    )


def deploy_target_frame_binding_hash(
    pose_frame_contract: PoseFrameContract | Mapping[str, Any],
    *,
    sides: Sequence[str],
) -> str:
    contract = _pose_frame_contract(pose_frame_contract)
    normalized_sides = tuple(str(side).strip().lower() for side in sides)
    if not normalized_sides or len(set(normalized_sides)) != len(normalized_sides):
        raise TCPPoseEnvelopeRuntimeError("deployment envelope sides must be unique")
    transforms: dict[str, object] = {}
    raw_frames: dict[str, str] = {}
    for side in normalized_sides:
        try:
            transforms[side] = (
                contract.source.effective_raw_to_canonical_desk_transform(side)
            ).tolist()
            raw_frames[side] = contract.source.raw_frame(side)
        except (PoseFrameContractError, KeyError) as exc:
            raise TCPPoseEnvelopeRuntimeError(
                f"deployment pose-frame contract is missing side {side!r}"
            ) from exc
    return canonical_content_hash(
        {
            "contract_hash": contract.contract_hash,
            "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
            "canonical_reference_frame": contract.canonical_absolute_frame,
            "raw_reference_frame_by_side": raw_frames,
            "raw_to_canonical_transform_by_side": transforms,
        },
        domain=DEPLOY_TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
    )


def deploy_tool_frame_binding_hashes_from_arm_states(
    arm_states: Mapping[str, Mapping[str, Any]],
    *,
    sides: Sequence[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_side in sides:
        side = str(raw_side).strip().lower()
        state = arm_states.get(side)
        if not isinstance(state, Mapping):
            raise TCPPoseEnvelopeRuntimeError(
                f"deployment arm state is missing configured side {side!r}"
            )
        source = str(state.get("tool_tcp_source") or "").strip()
        if not source:
            raise TCPPoseEnvelopeRuntimeError(
                f"{side} deployment arm state has no tool_tcp_source"
            )
        try:
            pose_flange = np.asarray(
                state.get("tool_tcp_pose_flange_m_rad"),
                dtype=np.float64,
            ).reshape(6)
        except (TypeError, ValueError) as exc:
            raise TCPPoseEnvelopeRuntimeError(
                f"{side} deployment arm state has no finite tool_tcp binding"
            ) from exc
        if not np.all(np.isfinite(pose_flange)):
            raise TCPPoseEnvelopeRuntimeError(
                f"{side} deployment arm state has no finite tool_tcp binding"
            )
        result[side] = canonical_content_hash(
            {
                "side": side,
                "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
                "tool_tcp_source": source,
                "tool_tcp_pose_flange_m_rad": pose_flange.tolist(),
                "active_tool_name": state.get("active_tool_name"),
                "active_tool_profile_path": state.get(
                    "active_tool_profile_path"
                ),
            },
            domain=DEPLOY_TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
    return result


@dataclass(frozen=True)
class DeployTCPPoseEnvelopeSnapshot:
    bound: bool
    apply_to_target: bool
    execution_epoch: int
    anchor_hash: str | None
    frame_count: int
    last_target_role: str | None
    last_error: str | None
    projection: dict[str, object] | None


class DeployTCPPoseEnvelopeProjector:
    """Default-off deploy target projector bound to canonical desk/tool_tcp."""

    def __init__(
        self,
        *,
        file_spec: TCPPoseEnvelopeFileSpec,
        pose_frame_contract: PoseFrameContract | Mapping[str, Any],
        active_sides: Sequence[str],
        apply_to_target: bool = False,
        anchor_max_age_sec: float = 0.25,
        anchor_pair_skew_sec: float = 0.05,
    ) -> None:
        if not isinstance(file_spec, TCPPoseEnvelopeFileSpec):
            raise TypeError("file_spec must be TCPPoseEnvelopeFileSpec")
        self.file_spec = file_spec
        self.pose_frame_contract = _pose_frame_contract(pose_frame_contract)
        self.active_sides = tuple(
            str(side).strip().lower() for side in active_sides
        )
        if not self.active_sides or len(set(self.active_sides)) != len(
            self.active_sides
        ):
            raise ValueError("active_sides must be non-empty and unique")
        configured = set(file_spec.side_config_by_side)
        inactive = sorted(configured - set(self.active_sides))
        if inactive:
            raise ValueError(
                f"deployment TCP envelope configures inactive sides: {inactive}"
            )
        self.apply_to_target = bool(apply_to_target)
        self.anchor_max_age_sec = float(anchor_max_age_sec)
        self.anchor_pair_skew_sec = float(anchor_pair_skew_sec)
        for name, value in (
            ("anchor_max_age_sec", self.anchor_max_age_sec),
            ("anchor_pair_skew_sec", self.anchor_pair_skew_sec),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self._target_frame_binding_hash = deploy_target_frame_binding_hash(
            self.pose_frame_contract,
            sides=tuple(file_spec.side_config_by_side),
        )
        self._reference_frame_by_side = {
            side: self.pose_frame_contract.canonical_absolute_frame
            for side in file_spec.side_config_by_side
        }
        self._lock = threading.Lock()
        self._profile: TCPPoseEnvelopeProfile | None = None
        self._anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None = None
        self._execution_epoch = 0
        self._frame_count = 0
        self._last_target_role: str | None = None
        self._last_error: str | None = None
        self._last_projection: TCPPoseEnvelopeProjectionResult | None = None

    @property
    def bound(self) -> bool:
        with self._lock:
            return self._profile is not None

    @property
    def requires_anchor(self) -> bool:
        return any(
            config.requires_anchor
            for config in self.file_spec.side_config_by_side.values()
        )

    def bind_from_observation(
        self,
        observation: Any,
        *,
        execution_epoch: int,
        captured_at: float,
        anchor_id: str = "deploy_runtime_start_pose",
    ) -> TCPPoseEnvelopeAnchorReceipt | None:
        """Bind static tool identity and, when needed, one fresh TCP Pose."""

        epoch = int(execution_epoch)
        if isinstance(execution_epoch, bool) or epoch < 0:
            raise ValueError("execution_epoch must be a non-negative integer")
        capture_time = float(captured_at)
        if not math.isfinite(capture_time):
            raise ValueError("captured_at must be finite")
        arm_states = getattr(observation, "arm_states", None)
        if not isinstance(arm_states, Mapping):
            raise TCPPoseEnvelopeRuntimeError(
                "deployment observation has no immutable arm_states mapping"
            )
        configured_sides = tuple(self.file_spec.side_config_by_side)
        tool_hashes = deploy_tool_frame_binding_hashes_from_arm_states(
            arm_states,
            sides=configured_sides,
        )
        profile = TCPPoseEnvelopeProfile(
            profile_id=self.file_spec.profile_id,
            target_frame_binding_hash=self._target_frame_binding_hash,
            tool_frame_binding_hash_by_side=tool_hashes,
            reference_frame_by_side=self._reference_frame_by_side,
            side_config_by_side=self.file_spec.side_config_by_side,
        )
        anchor: TCPPoseEnvelopeAnchorReceipt | None = None
        if profile.anchor_required_sides:
            poses: dict[str, np.ndarray] = {}
            timestamps: dict[str, float] = {}
            for side in profile.anchor_required_sides:
                state = arm_states.get(side)
                assert isinstance(state, Mapping)
                source = str(state.get("actual_tcp_pose_source") or "").strip()
                if not source:
                    raise TCPPoseEnvelopeRuntimeError(
                        f"{side} has no measured tool_tcp Pose source for anchor capture"
                    )
                timestamp_raw = state.get("actual_tcp_pose_timestamp")
                try:
                    timestamp = float(timestamp_raw)
                except (TypeError, ValueError) as exc:
                    raise TCPPoseEnvelopeRuntimeError(
                        f"{side} has no measured tool_tcp Pose timestamp"
                    ) from exc
                age = capture_time - timestamp
                if not math.isfinite(age) or age < -1e-6 or age > self.anchor_max_age_sec:
                    raise TCPPoseEnvelopeRuntimeError(
                        f"{side} measured tool_tcp Pose is stale for anchor capture: "
                        f"age={age:.6f}s limit={self.anchor_max_age_sec:.6f}s"
                    )
                raw_pose = _pose(
                    state.get("actual_tcp_pose"),
                    name=f"observation.arm_states.{side}.actual_tcp_pose",
                )
                try:
                    poses[side] = self.pose_frame_contract.storage_pose_to_canonical_desk(
                        raw_pose,
                        side=side,
                    )
                except PoseFrameContractError as exc:
                    raise TCPPoseEnvelopeRuntimeError(
                        f"{side} anchor Pose cannot be converted to canonical desk: {exc}"
                    ) from exc
                timestamps[side] = timestamp
            if max(timestamps.values()) - min(timestamps.values()) > self.anchor_pair_skew_sec:
                raise TCPPoseEnvelopeRuntimeError(
                    "dual-arm tool_tcp anchor timestamps exceed allowed skew: "
                    f"skew={max(timestamps.values()) - min(timestamps.values()):.6f}s "
                    f"limit={self.anchor_pair_skew_sec:.6f}s"
                )
            anchor = TCPPoseEnvelopeAnchorReceipt.capture(
                anchor_id=anchor_id,
                profile=profile,
                execution_epoch=epoch,
                clock_domain_id="python_monotonic",
                tcp_pose_by_side=poses,
                captured_at_by_side=timestamps,
            )
        with self._lock:
            self._profile = profile
            self._anchor_receipt = anchor
            self._execution_epoch = epoch
            self._last_target_role = None
            self._last_error = None
            self._last_projection = None
        return anchor

    def __call__(
        self,
        targets: Mapping[str, np.ndarray],
        *,
        observation: Any,
        execution_work_item: Any | None = None,
        target_role: str = "current",
    ) -> Mapping[str, np.ndarray]:
        del execution_work_item  # lineage is owned by PositionDeployExecutor.
        with self._lock:
            profile = self._profile
            anchor = self._anchor_receipt
            epoch = int(self._execution_epoch)
        try:
            if profile is None:
                raise TCPPoseEnvelopeRuntimeError(
                    "deployment TCP pose envelope is not bound to a fresh start Pose"
                )
            arm_states = getattr(observation, "arm_states", None)
            if not isinstance(arm_states, Mapping):
                raise TCPPoseEnvelopeRuntimeError(
                    "deployment observation has no immutable arm_states mapping"
                )
            actual_tool_hashes = deploy_tool_frame_binding_hashes_from_arm_states(
                arm_states,
                sides=tuple(profile.side_config_by_side),
            )
            canonical_targets: dict[str, np.ndarray] = {}
            for side, raw_target in targets.items():
                try:
                    canonical_targets[side] = (
                        self.pose_frame_contract.storage_pose_to_canonical_desk(
                            _pose(raw_target, name=f"deploy_target.{side}"),
                            side=side,
                        )
                    )
                except PoseFrameContractError as exc:
                    raise TCPPoseEnvelopeRuntimeError(
                        f"{side} target Pose cannot be converted to canonical desk: {exc}"
                    ) from exc
            projection = project_tcp_pose_envelope(
                canonical_targets,
                profile=profile,
                target_frame_binding_hash=self._target_frame_binding_hash,
                tool_frame_binding_hash_by_side=actual_tool_hashes,
                controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
                reference_frame_by_side=self._reference_frame_by_side,
                execution_epoch=epoch,
                anchor_receipt=anchor,
            )
            applied = {
                side: np.asarray(target, dtype=float).copy()
                for side, target in targets.items()
            }
            for side in projection.modified_sides:
                try:
                    applied[side] = self.pose_frame_contract.canonical_desk_pose_to_storage(
                        projection.effective_tcp_pose_by_side[side],
                        side=side,
                    )
                except PoseFrameContractError as exc:
                    raise TCPPoseEnvelopeRuntimeError(
                        f"{side} projected Pose cannot be converted to runtime raw: {exc}"
                    ) from exc
        except Exception as exc:  # noqa: BLE001 - Shadow records; Active fails closed.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._frame_count += 1
                self._last_target_role = str(target_role)
                self._last_error = error
                self._last_projection = None
            if self.apply_to_target:
                if isinstance(exc, TCPPoseEnvelopeRuntimeError):
                    raise
                raise TCPPoseEnvelopeRuntimeError(error) from exc
            return targets

        with self._lock:
            self._frame_count += 1
            self._last_target_role = str(target_role)
            self._last_error = None
            self._last_projection = projection
        if not self.apply_to_target:
            return targets
        return applied

    def project_arm_base_tool_tcp(
        self,
        side: str,
        arm_base_tool_tcp: np.ndarray,
    ) -> np.ndarray:
        """Project one impedance worker's final composed runtime-raw TCP."""

        normalized_side = str(side).strip().lower()
        with self._lock:
            profile = self._profile
            anchor = self._anchor_receipt
            epoch = int(self._execution_epoch)
        try:
            if profile is None:
                raise TCPPoseEnvelopeRuntimeError(
                    "deployment TCP pose envelope is not bound to a fresh start Pose"
                )
            if normalized_side not in profile.side_config_by_side:
                raise TCPPoseEnvelopeRuntimeError(
                    f"side {normalized_side!r} is absent from the TCP envelope profile"
                )
            raw_base_pose = _pose(
                arm_base_tool_tcp,
                name=f"impedance_composed_target.{normalized_side}",
            )
            try:
                canonical_pose = (
                    self.pose_frame_contract.storage_pose_to_canonical_desk(
                        raw_base_pose,
                        side=normalized_side,
                    )
                )
            except PoseFrameContractError as exc:
                raise TCPPoseEnvelopeRuntimeError(
                    f"{normalized_side} impedance TCP cannot be converted to "
                    f"canonical desk: {exc}"
                ) from exc
            projection = project_tcp_pose_envelope_side(
                normalized_side,
                canonical_pose,
                profile=profile,
                target_frame_binding_hash=self._target_frame_binding_hash,
                tool_frame_binding_hash_by_side=(
                    profile.tool_frame_binding_hash_by_side
                ),
                controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
                reference_frame_by_side=self._reference_frame_by_side,
                execution_epoch=epoch,
                anchor_receipt=anchor,
            )
            effective_canonical_pose = projection.effective_tcp_pose_by_side[
                normalized_side
            ]
            try:
                effective_base_pose = (
                    self.pose_frame_contract.canonical_desk_pose_to_storage(
                        effective_canonical_pose,
                        side=normalized_side,
                    )
                )
            except PoseFrameContractError as exc:
                raise TCPPoseEnvelopeRuntimeError(
                    f"{normalized_side} projected impedance TCP cannot be converted "
                    f"to runtime raw: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001 - Shadow records; Active fails closed.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._frame_count += 1
                self._last_target_role = "final_composed_impedance"
                self._last_error = error
                self._last_projection = None
            if self.apply_to_target:
                if isinstance(exc, TCPPoseEnvelopeRuntimeError):
                    raise
                raise TCPPoseEnvelopeRuntimeError(error) from exc
            return _pose(
                arm_base_tool_tcp,
                name=f"impedance_composed_target.{normalized_side}",
            )

        with self._lock:
            self._frame_count += 1
            self._last_target_role = "final_composed_impedance"
            self._last_error = None
            self._last_projection = projection
        if not self.apply_to_target:
            return raw_base_pose
        return _pose(
            effective_base_pose,
            name=f"effective_impedance_composed_target.{normalized_side}",
        )

    def snapshot(self) -> DeployTCPPoseEnvelopeSnapshot:
        with self._lock:
            projection = self._last_projection
            return DeployTCPPoseEnvelopeSnapshot(
                bound=self._profile is not None,
                apply_to_target=bool(self.apply_to_target),
                execution_epoch=int(self._execution_epoch),
                anchor_hash=(
                    None
                    if self._anchor_receipt is None
                    else self._anchor_receipt.semantic_content_hash
                ),
                frame_count=int(self._frame_count),
                last_target_role=self._last_target_role,
                last_error=self._last_error,
                projection=(
                    None
                    if projection is None
                    else copy.deepcopy(projection.to_payload())
                ),
            )

    def feedback_snapshot(self) -> TCPPoseEnvelopeProjectionFeedback:
        """Return the latest compact receipt for passive downstream polling."""

        with self._lock:
            profile = self._profile
            anchor_hash = (
                None
                if self._anchor_receipt is None
                else self._anchor_receipt.semantic_content_hash
            )
            return TCPPoseEnvelopeProjectionFeedback.from_evaluation(
                apply_to_target=self.apply_to_target,
                bound=profile is not None,
                profile_id=self.file_spec.profile_id,
                profile_hash=(
                    None if profile is None else profile.semantic_content_hash
                ),
                execution_epoch=self._execution_epoch,
                anchor_hash=anchor_hash,
                frame_count=self._frame_count,
                target_role=self._last_target_role,
                projection=self._last_projection,
                error=self._last_error,
            )

    def command_metadata(self) -> dict[str, object]:
        """Return JSON-safe compact lineage without raw/effective pose matrices."""

        payload = self.feedback_snapshot().to_payload()
        payload["feedback_schema_version"] = payload["schema_version"]
        payload["schema_version"] = (
            "teleop_runtime.deploy_tcp_pose_envelope_command.v1"
        )
        return payload


__all__ = [
    "DEPLOY_TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "DEPLOY_TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "DeployTCPPoseEnvelopeProjector",
    "DeployTCPPoseEnvelopeSnapshot",
    "deploy_target_frame_binding_hash",
    "deploy_tool_frame_binding_hashes_from_arm_states",
]
