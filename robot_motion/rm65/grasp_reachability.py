"""Typed endpoint reachability receipts for RM65 grasp-pose probes.

This module classifies one pose target.  It deliberately owns neither a
motion path nor dispatch: a successful receipt identifies an endpoint IK
candidate only.  Path planning and path-collision receipts remain a separate
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np


class GraspReachabilityStatus(str, Enum):
    REACHABLE = "reachable"
    KINEMATICALLY_UNREACHABLE = "kinematically_unreachable"
    IK_SEARCH_EXHAUSTED = "ik_search_exhausted"
    COLLISION_BLOCKED = "collision_blocked"
    COLLISION_SEARCH_EXHAUSTED = "collision_search_exhausted"
    INVALID_TARGET = "invalid_target"
    PLANNER_ERROR = "planner_error"


class CollisionCandidateStatus(str, Enum):
    SAFE = "safe"
    BLOCKED = "blocked"
    SEARCH_EXHAUSTED = "search_exhausted"
    ERROR = "error"


@dataclass(frozen=True)
class ConservativeWorkspaceEnvelope:
    """A one-sided proof envelope; inside never proves reachability."""

    reference_frame_id: str
    reference_frame_spec_hash: str
    origin_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    maximum_radius_m: float = 1.25
    proof_label: str = "rm65_outer_radius_conservative_v1"

    def __post_init__(self) -> None:
        frame_id = str(self.reference_frame_id).strip()
        frame_hash = str(self.reference_frame_spec_hash).strip().lower()
        if not frame_id:
            raise ValueError("reference_frame_id must not be empty.")
        if not _is_sha256(frame_hash):
            raise ValueError("reference_frame_spec_hash must be a lowercase SHA-256.")
        origin = tuple(float(value) for value in self.origin_xyz_m)
        if len(origin) != 3 or not np.all(np.isfinite(origin)):
            raise ValueError("origin_xyz_m must contain 3 finite values.")
        radius = float(self.maximum_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("maximum_radius_m must be finite and positive.")
        object.__setattr__(self, "origin_xyz_m", origin)
        object.__setattr__(self, "reference_frame_id", frame_id)
        object.__setattr__(self, "reference_frame_spec_hash", frame_hash)
        object.__setattr__(self, "maximum_radius_m", radius)
        object.__setattr__(self, "proof_label", str(self.proof_label))

    def outside_reason(self, target_pose: np.ndarray) -> tuple[str, dict[str, Any]] | None:
        distance = float(
            np.linalg.norm(
                np.asarray(target_pose[:3, 3], dtype=float)
                - np.asarray(self.origin_xyz_m, dtype=float)
            )
        )
        if distance <= self.maximum_radius_m:
            return None
        return (
            "outside_conservative_outer_workspace",
            {
                "proof_label": self.proof_label,
                "origin_distance_m": distance,
                "maximum_radius_m": self.maximum_radius_m,
            },
        )


@dataclass(frozen=True)
class IKCandidate:
    candidate_index: int
    qpos_rad: tuple[float, ...] | None
    converged: bool
    position_residual_m: float | None = None
    orientation_residual_rad: float | None = None
    iterations: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        if self.qpos_rad is not None:
            qpos = tuple(float(value) for value in self.qpos_rad)
            if len(qpos) != 6 or not np.all(np.isfinite(qpos)):
                raise ValueError("qpos_rad must contain 6 finite RM65 joint values.")
            object.__setattr__(self, "qpos_rad", qpos)
        for name in ("position_residual_m", "orientation_residual_rad"):
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not np.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative.")
                object.__setattr__(self, name, numeric)
        if self.iterations is not None:
            iterations = int(self.iterations)
            if iterations < 0:
                raise ValueError("iterations must be non-negative when present.")
            object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self, *, rejection_reasons: tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "qpos_rad": None if self.qpos_rad is None else list(self.qpos_rad),
            "converged": bool(self.converged),
            "position_residual_m": self.position_residual_m,
            "orientation_residual_rad": self.orientation_residual_rad,
            "iterations": self.iterations,
            "rejection_reasons": list(rejection_reasons),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CollisionCandidateAssessment:
    status: CollisionCandidateStatus
    reason_code: str
    minimum_distance_m: float | None = None
    required_distance_m: float | None = None
    obstacle_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CollisionCandidateStatus(self.status))
        object.__setattr__(self, "reason_code", str(self.reason_code))
        for name in ("minimum_distance_m", "required_distance_m"):
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"{name} must be finite when present.")
                if name == "required_distance_m" and numeric < 0.0:
                    raise ValueError("required_distance_m must be non-negative.")
                object.__setattr__(self, name, numeric)
        object.__setattr__(self, "obstacle_ids", tuple(str(value) for value in self.obstacle_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self, candidate_index: int) -> dict[str, Any]:
        return {
            "candidate_index": int(candidate_index),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "minimum_distance_m": self.minimum_distance_m,
            "required_distance_m": self.required_distance_m,
            "obstacle_ids": list(self.obstacle_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraspReachabilityReceipt:
    status: GraspReachabilityStatus
    reason_code: str
    proof_scope: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    target_pose: tuple[tuple[float, ...], ...] | None
    target_pose_content_hash: str | None = None
    selected_candidate_index: int | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    collision_assessments: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GraspReachabilityStatus(self.status))
        object.__setattr__(self, "reason_code", str(self.reason_code))
        object.__setattr__(self, "proof_scope", str(self.proof_scope))
        object.__setattr__(self, "reference_frame_id", str(self.reference_frame_id))
        object.__setattr__(
            self, "reference_frame_spec_hash", str(self.reference_frame_spec_hash)
        )
        if self.target_pose_content_hash is not None:
            object.__setattr__(
                self, "target_pose_content_hash", str(self.target_pose_content_hash)
            )
        object.__setattr__(self, "candidates", tuple(dict(item) for item in self.candidates))
        object.__setattr__(
            self,
            "collision_assessments",
            tuple(dict(item) for item in self.collision_assessments),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "proof_scope": self.proof_scope,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "target_pose_content_hash": self.target_pose_content_hash,
            "target_pose": (
                None if self.target_pose is None else [list(row) for row in self.target_pose]
            ),
            "selected_candidate_index": self.selected_candidate_index,
            "candidates": [dict(item) for item in self.candidates],
            "collision_assessments": [dict(item) for item in self.collision_assessments],
            "metadata": dict(self.metadata),
        }


IKCandidateProvider = Callable[[int, np.ndarray], IKCandidate]
CollisionCandidateChecker = Callable[[IKCandidate, np.ndarray], CollisionCandidateAssessment]


def evaluate_grasp_pose_reachability(
    target_pose: Any,
    *,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    candidate_provider: IKCandidateProvider,
    max_candidates: int,
    maximum_position_residual_m: float,
    maximum_orientation_residual_rad: float,
    workspace_envelope: ConservativeWorkspaceEnvelope | None = None,
    collision_checker: CollisionCandidateChecker | None = None,
    target_pose_content_hash: str | None = None,
) -> GraspReachabilityReceipt:
    """Return one terminal endpoint classification for a bounded probe.

    Failure to find IK in ``max_candidates`` is search exhaustion, not a proof
    of mathematical unreachability.  ``KINEMATICALLY_UNREACHABLE`` is reserved
    for a configured one-sided conservative proof.
    """

    frame_id = str(reference_frame_id).strip()
    frame_hash = str(reference_frame_spec_hash).strip().lower()
    frame_error = None
    if not frame_id:
        frame_error = "reference_frame_id_empty"
    elif not _is_sha256(frame_hash):
        frame_error = "reference_frame_spec_hash_invalid"
    content_hash = (
        None if target_pose_content_hash is None else str(target_pose_content_hash).strip().lower()
    )
    if frame_error is None and content_hash is not None and not _is_sha256(content_hash):
        frame_error = "target_pose_content_hash_invalid"

    pose, target_error = _validated_target_pose(target_pose)
    target_error = frame_error or target_error
    if target_error is not None:
        return GraspReachabilityReceipt(
            status=GraspReachabilityStatus.INVALID_TARGET,
            reason_code=target_error,
            proof_scope="rigid_target_validation",
            reference_frame_id=frame_id,
            reference_frame_spec_hash=frame_hash,
            target_pose=None,
            target_pose_content_hash=content_hash,
        )
    assert pose is not None
    frozen_pose = tuple(tuple(float(value) for value in row) for row in pose)

    candidate_limit = int(max_candidates)
    position_limit = float(maximum_position_residual_m)
    orientation_limit = float(maximum_orientation_residual_rad)
    if candidate_limit < 1 or not _finite_nonnegative(position_limit, orientation_limit):
        return GraspReachabilityReceipt(
            status=GraspReachabilityStatus.INVALID_TARGET,
            reason_code="invalid_probe_configuration",
            proof_scope="probe_configuration_validation",
            reference_frame_id=frame_id,
            reference_frame_spec_hash=frame_hash,
            target_pose=frozen_pose,
            target_pose_content_hash=content_hash,
        )

    if workspace_envelope is not None:
        if (
            workspace_envelope.reference_frame_id != frame_id
            or workspace_envelope.reference_frame_spec_hash != frame_hash
        ):
            return GraspReachabilityReceipt(
                status=GraspReachabilityStatus.INVALID_TARGET,
                reason_code="workspace_and_target_reference_frame_mismatch",
                proof_scope="reference_frame_identity_validation",
                reference_frame_id=frame_id,
                reference_frame_spec_hash=frame_hash,
                target_pose=frozen_pose,
                target_pose_content_hash=content_hash,
                metadata={
                    "workspace_reference_frame_id": workspace_envelope.reference_frame_id,
                    "workspace_reference_frame_spec_hash": (
                        workspace_envelope.reference_frame_spec_hash
                    ),
                },
            )
        outside = workspace_envelope.outside_reason(pose)
        if outside is not None:
            reason, proof_metadata = outside
            return GraspReachabilityReceipt(
                status=GraspReachabilityStatus.KINEMATICALLY_UNREACHABLE,
                reason_code=reason,
                proof_scope="conservative_workspace_exclusion",
                reference_frame_id=frame_id,
                reference_frame_spec_hash=frame_hash,
                target_pose=frozen_pose,
                target_pose_content_hash=content_hash,
                metadata=proof_metadata,
            )

    candidate_records: list[dict[str, Any]] = []
    collision_records: list[dict[str, Any]] = []
    eligible_candidates = 0
    provider_errors = 0
    collision_counts = {status: 0 for status in CollisionCandidateStatus}

    for candidate_index in range(candidate_limit):
        try:
            candidate = candidate_provider(candidate_index, pose.copy())
            if not isinstance(candidate, IKCandidate):
                raise TypeError("candidate provider must return IKCandidate")
            if candidate.candidate_index != candidate_index:
                raise ValueError("candidate_index must match the requested multistart index")
        except Exception as exc:  # adapter errors become typed terminal evidence
            provider_errors += 1
            candidate_records.append(
                {
                    "candidate_index": candidate_index,
                    "qpos_rad": None,
                    "converged": False,
                    "position_residual_m": None,
                    "orientation_residual_rad": None,
                    "iterations": None,
                    "rejection_reasons": ["candidate_provider_error"],
                    "metadata": {"error_type": type(exc).__name__, "error": str(exc)},
                }
            )
            continue

        rejection_reasons = _candidate_rejection_reasons(
            candidate,
            maximum_position_residual_m=position_limit,
            maximum_orientation_residual_rad=orientation_limit,
        )
        candidate_records.append(candidate.to_dict(rejection_reasons=rejection_reasons))
        if rejection_reasons:
            continue
        eligible_candidates += 1

        if collision_checker is None:
            return GraspReachabilityReceipt(
                status=GraspReachabilityStatus.REACHABLE,
                reason_code="ik_candidate_within_residual_limits",
                proof_scope="endpoint_kinematic_candidate_only",
                reference_frame_id=frame_id,
                reference_frame_spec_hash=frame_hash,
                target_pose=frozen_pose,
                target_pose_content_hash=content_hash,
                selected_candidate_index=candidate.candidate_index,
                candidates=tuple(candidate_records),
                metadata={"collision_checked": False, "candidate_limit": candidate_limit},
            )

        try:
            assessment = collision_checker(candidate, pose.copy())
            if not isinstance(assessment, CollisionCandidateAssessment):
                raise TypeError("collision checker must return CollisionCandidateAssessment")
        except Exception as exc:
            assessment = CollisionCandidateAssessment(
                status=CollisionCandidateStatus.ERROR,
                reason_code="collision_checker_error",
                metadata={"error_type": type(exc).__name__, "error": str(exc)},
            )
        collision_counts[assessment.status] += 1
        collision_records.append(assessment.to_dict(candidate.candidate_index))
        if assessment.status is CollisionCandidateStatus.SAFE:
            return GraspReachabilityReceipt(
                status=GraspReachabilityStatus.REACHABLE,
                reason_code="ik_and_endpoint_collision_candidate_accepted",
                proof_scope="endpoint_ik_and_collision_candidate",
                reference_frame_id=frame_id,
                reference_frame_spec_hash=frame_hash,
                target_pose=frozen_pose,
                target_pose_content_hash=content_hash,
                selected_candidate_index=candidate.candidate_index,
                candidates=tuple(candidate_records),
                collision_assessments=tuple(collision_records),
                metadata={"collision_checked": True, "candidate_limit": candidate_limit},
            )

    common = {
        "reference_frame_id": frame_id,
        "reference_frame_spec_hash": frame_hash,
        "target_pose": frozen_pose,
        "target_pose_content_hash": content_hash,
        "candidates": tuple(candidate_records),
        "collision_assessments": tuple(collision_records),
        "metadata": {
            "candidate_limit": candidate_limit,
            "eligible_candidate_count": eligible_candidates,
            "provider_error_count": provider_errors,
            "collision_status_counts": {
                status.value: collision_counts[status] for status in CollisionCandidateStatus
            },
        },
    }
    if eligible_candidates == 0:
        if provider_errors > 0:
            return GraspReachabilityReceipt(
                status=GraspReachabilityStatus.PLANNER_ERROR,
                reason_code="candidate_search_incomplete_due_to_provider_error",
                proof_scope="incomplete_candidate_adapter_search",
                **common,
            )
        return GraspReachabilityReceipt(
            status=GraspReachabilityStatus.IK_SEARCH_EXHAUSTED,
            reason_code="bounded_ik_multistart_exhausted",
            proof_scope="bounded_multistart_search_only",
            **common,
        )

    if provider_errors > 0:
        return GraspReachabilityReceipt(
            status=GraspReachabilityStatus.PLANNER_ERROR,
            reason_code="candidate_search_incomplete_due_to_provider_error",
            proof_scope="incomplete_candidate_adapter_search",
            **common,
        )
    if collision_counts[CollisionCandidateStatus.BLOCKED] == eligible_candidates:
        return GraspReachabilityReceipt(
            status=GraspReachabilityStatus.COLLISION_BLOCKED,
            reason_code="all_ik_candidates_explicitly_collision_blocked",
            proof_scope="provided_endpoint_collision_checks_for_all_ik_candidates",
            **common,
        )
    if collision_counts[CollisionCandidateStatus.ERROR] == eligible_candidates:
        return GraspReachabilityReceipt(
            status=GraspReachabilityStatus.PLANNER_ERROR,
            reason_code="all_collision_checker_calls_failed",
            proof_scope="collision_adapter_failure",
            **common,
        )
    return GraspReachabilityReceipt(
        status=GraspReachabilityStatus.COLLISION_SEARCH_EXHAUSTED,
        reason_code="bounded_collision_candidate_search_exhausted",
        proof_scope="bounded_endpoint_collision_search_only",
        **common,
    )


def _validated_target_pose(target_pose: Any) -> tuple[np.ndarray | None, str | None]:
    try:
        pose = np.asarray(target_pose, dtype=float)
    except (TypeError, ValueError):
        return None, "target_pose_not_numeric"
    if pose.shape != (4, 4):
        return None, "target_pose_shape_must_be_4x4"
    if not np.all(np.isfinite(pose)):
        return None, "target_pose_contains_nonfinite_values"
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        return None, "target_pose_invalid_homogeneous_row"
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        return None, "target_pose_rotation_not_orthonormal"
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        return None, "target_pose_rotation_not_proper"
    return pose.copy(), None


def _finite_nonnegative(*values: float) -> bool:
    return all(np.isfinite(value) and value >= 0.0 for value in values)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _candidate_rejection_reasons(
    candidate: IKCandidate,
    *,
    maximum_position_residual_m: float,
    maximum_orientation_residual_rad: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not candidate.converged:
        reasons.append("ik_not_converged")
    if candidate.qpos_rad is None:
        reasons.append("missing_endpoint_qpos")
    if candidate.position_residual_m is None:
        reasons.append("missing_position_residual")
    elif candidate.position_residual_m > maximum_position_residual_m:
        reasons.append("position_residual_exceeded")
    if candidate.orientation_residual_rad is None:
        reasons.append("missing_orientation_residual")
    elif candidate.orientation_residual_rad > maximum_orientation_residual_rad:
        reasons.append("orientation_residual_exceeded")
    return tuple(reasons)


__all__ = [
    "CollisionCandidateAssessment",
    "CollisionCandidateChecker",
    "CollisionCandidateStatus",
    "ConservativeWorkspaceEnvelope",
    "GraspReachabilityReceipt",
    "GraspReachabilityStatus",
    "IKCandidate",
    "IKCandidateProvider",
    "evaluate_grasp_pose_reachability",
]
