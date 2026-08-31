"""Deterministic raw-Cartesian targets for continuous MuJoCo grasp viewing.

Every sampled target is emitted exactly once.  This module never rejection-
samples for reachability: endpoint IK, endpoint collision, direct-path
collision, and execution remain owned by ``execute_grasp_motion``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.rm65.grasp_reachability import GraspReachabilityStatus
from robot_motion.rm65.ik import RM65SingleArmIKConfig, build_rm65_single_arm_ik_solver
from sim.sim_real_tests.rm65.random_grasp_pose_reachability import (
    DEFAULT_SAMPLER_ID,
    compute_target_pose_content_hash,
)

from .arbitrary_grasp_pose_motion import (
    MOTION_STATUS_COMPLETED,
    MOTION_STATUS_ENDPOINT_REJECTED,
    MOTION_STATUS_PATH_COLLISION_BLOCKED,
    GraspMotionPlayback,
    GraspMotionTarget,
)

CONTINUOUS_RANDOM_REFERENCE_FRAME_ID = "rm65.selected_arm_base"
JOINT_FK_SAMPLER_ID = "pcg64_joint_offset_dmtac_fk_v1"
CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC = {
    "schema": "rm65.reference_frame.v1",
    "reference_frame_id": CONTINUOUS_RANDOM_REFERENCE_FRAME_ID,
    "origin": "selected_rm65_arm_base",
    "axes": "selected_rm65_arm_base_coordinates",
    "target_pose_semantics": "selected_arm_base_from_actual_dmtac_tool_tcp",
    "translation_unit": "m",
}
CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC_HASH = hashlib.sha256(
    json.dumps(
        CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class ContinuousRandomGraspPoseConfig:
    seed: int = 20260831
    position_min_m: tuple[float, float, float] = (-0.70, -0.80, -0.10)
    position_max_m: tuple[float, float, float] = (0.70, 0.80, 1.20)
    sampler_id: str = DEFAULT_SAMPLER_ID
    reference_frame_id: str = CONTINUOUS_RANDOM_REFERENCE_FRAME_ID
    reference_frame_spec_hash: str = CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC_HASH

    def __post_init__(self) -> None:
        minimum = tuple(float(value) for value in self.position_min_m)
        maximum = tuple(float(value) for value in self.position_max_m)
        if len(minimum) != 3 or len(maximum) != 3:
            raise ValueError("Cartesian bounds must contain three values")
        if not np.all(np.isfinite(minimum + maximum)):
            raise ValueError("Cartesian bounds must be finite")
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError("Every position minimum must be below its maximum")
        if str(self.sampler_id) != DEFAULT_SAMPLER_ID:
            raise ValueError(f"sampler_id must be {DEFAULT_SAMPLER_ID!r}")
        frame_id = str(self.reference_frame_id).strip()
        frame_hash = str(self.reference_frame_spec_hash).strip().lower()
        if not frame_id:
            raise ValueError("reference_frame_id must not be empty")
        if len(frame_hash) != 64 or any(c not in "0123456789abcdef" for c in frame_hash):
            raise ValueError("reference_frame_spec_hash must be a lowercase SHA-256")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "position_min_m", minimum)
        object.__setattr__(self, "position_max_m", maximum)
        object.__setattr__(self, "reference_frame_id", frame_id)
        object.__setattr__(self, "reference_frame_spec_hash", frame_hash)


class RawCartesianGraspTargetStream:
    """Emit a deterministic sequence of unfiltered Cartesian tool-TCP poses."""

    def __init__(
        self,
        *,
        side: str,
        config: ContinuousRandomGraspPoseConfig | None = None,
    ) -> None:
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        self.side = normalized_side
        self.config = config or ContinuousRandomGraspPoseConfig()
        self._next_sample_index = 0

    @property
    def next_sample_index(self) -> int:
        return self._next_sample_index

    def next_target(self) -> GraspMotionTarget:
        sample_index = self._next_sample_index
        self._next_sample_index += 1
        return self.target_for_sample_index(sample_index)

    def target_for_sample_index(self, sample_index: int) -> GraspMotionTarget:
        resolved_index = int(sample_index)
        if resolved_index < 0:
            raise ValueError("sample_index must be non-negative")
        rng = _rng_for_sample(
            seed=self.config.seed,
            sample_index=resolved_index,
            stream_tag=0x43415254,
        )
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = rng.uniform(
            np.asarray(self.config.position_min_m, dtype=float),
            np.asarray(self.config.position_max_m, dtype=float),
        )
        pose[:3, :3] = _sample_shoemake_rotation_matrix(rng)
        content_hash = compute_target_pose_content_hash(
            pose,
            sampler_id=self.config.sampler_id,
            seed=self.config.seed,
            sample_index=resolved_index,
            reference_frame_id=self.config.reference_frame_id,
            reference_frame_spec_hash=self.config.reference_frame_spec_hash,
        )
        return GraspMotionTarget(
            side=self.side,
            target_pose_base_from_dmtac_tcp=pose,
            source="raw_cartesian_uniform_box_shoemake_so3",
            metadata={
                "sample_index": resolved_index,
                "sample_kind": "raw_cartesian_probe",
                "seed": self.config.seed,
                "sampler_id": self.config.sampler_id,
                "reference_frame_id": self.config.reference_frame_id,
                "reference_frame_spec_hash": self.config.reference_frame_spec_hash,
                "target_pose_content_hash": content_hash,
                "rejection_sampled": False,
            },
        )


def sample_random_joint_fk_target(
    control: Any,
    *,
    side: str,
    seed: int,
    sample_index: int,
    maximum_joint_offset_deg: float = 45.0,
) -> GraspMotionTarget:
    """Draw one joint-offset target and expose its DMTAC TCP pose without retry."""

    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    resolved_index = int(sample_index)
    if resolved_index < 0:
        raise ValueError("sample_index must be non-negative")
    maximum_offset = np.deg2rad(float(maximum_joint_offset_deg))
    if not np.isfinite(maximum_offset) or maximum_offset <= 0.0:
        raise ValueError("maximum_joint_offset_deg must be finite and positive")
    solver = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(controlled_frame="tool_tcp")
    )
    current = {
        item: np.asarray(control.scene.get_arm_qpos(item), dtype=float).copy()
        for item in ("left", "right")
    }
    rng = _rng_for_sample(
        seed=int(seed),
        sample_index=resolved_index,
        stream_tag=0x4A464B31,
    )
    offset = rng.uniform(-maximum_offset, maximum_offset, size=6)
    candidate = np.clip(
        current[normalized_side] + offset,
        np.asarray(solver.q_min, dtype=float),
        np.asarray(solver.q_max, dtype=float),
    )
    qpos_pair = {item: values.copy() for item, values in current.items()}
    qpos_pair[normalized_side] = candidate.copy()
    world_from_dmtac = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=qpos_pair["left"], right_qpos=qpos_pair["right"]
    )[normalized_side]
    base_from_dmtac = (
        control.world_to_base_by_side[normalized_side] @ world_from_dmtac
    )
    content_hash = compute_target_pose_content_hash(
        base_from_dmtac,
        sampler_id=JOINT_FK_SAMPLER_ID,
        seed=int(seed),
        sample_index=resolved_index,
        reference_frame_id=CONTINUOUS_RANDOM_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC_HASH,
    )
    return GraspMotionTarget(
        side=normalized_side,
        target_pose_base_from_dmtac_tcp=base_from_dmtac,
        source="single_draw_random_joint_offset_dmtac_fk",
        known_qpos_rad=candidate,
        metadata={
            "sample_index": resolved_index,
            "sample_kind": "random_joint_fk_candidate",
            "seed": int(seed),
            "sampler_id": JOINT_FK_SAMPLER_ID,
            "reference_frame_id": CONTINUOUS_RANDOM_REFERENCE_FRAME_ID,
            "reference_frame_spec_hash": CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC_HASH,
            "target_pose_content_hash": content_hash,
            "maximum_joint_offset_deg": float(maximum_joint_offset_deg),
            "selected_max_joint_motion_deg": float(
                np.max(np.abs(np.rad2deg(candidate - current[normalized_side])))
            ),
            "rejection_sampled": False,
        },
    )


@dataclass(frozen=True)
class RandomGraspAttemptOutcome:
    sample_index: int
    motion_status: str
    reachability_status: str
    terminal_classification: str
    reason_code: str
    motion_emitted: bool
    frame_count: int
    target_pose_content_hash: str

    @property
    def proven_unreachable(self) -> bool:
        return self.terminal_classification == "proven_kinematically_unreachable"

    def to_payload(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "motion_status": self.motion_status,
            "reachability_status": self.reachability_status,
            "terminal_classification": self.terminal_classification,
            "reason_code": self.reason_code,
            "proven_unreachable": self.proven_unreachable,
            "motion_emitted": self.motion_emitted,
            "frame_count": self.frame_count,
            "target_pose_content_hash": self.target_pose_content_hash,
        }


def classify_random_grasp_attempt(
    playback: GraspMotionPlayback,
) -> RandomGraspAttemptOutcome:
    metadata = dict(playback.target.metadata or {})
    sample_index = int(metadata.get("sample_index", -1))
    content_hash = str(metadata.get("target_pose_content_hash", ""))
    reachability_status = playback.endpoint_receipt.status.value
    reason_code = str(playback.endpoint_receipt.reason_code)

    if playback.motion_status == MOTION_STATUS_ENDPOINT_REJECTED:
        endpoint_classes = {
            GraspReachabilityStatus.KINEMATICALLY_UNREACHABLE.value: (
                "proven_kinematically_unreachable"
            ),
            GraspReachabilityStatus.IK_SEARCH_EXHAUSTED.value: (
                "ik_search_exhausted_not_proven_unreachable"
            ),
            GraspReachabilityStatus.COLLISION_BLOCKED.value: (
                "endpoint_collision_blocked"
            ),
            GraspReachabilityStatus.COLLISION_SEARCH_EXHAUSTED.value: (
                "collision_search_exhausted_not_proven_unreachable"
            ),
            GraspReachabilityStatus.INVALID_TARGET.value: "invalid_target",
            GraspReachabilityStatus.PLANNER_ERROR.value: "planner_error",
        }
        terminal_classification = endpoint_classes.get(
            reachability_status, "endpoint_rejected_unknown"
        )
    elif playback.motion_status == MOTION_STATUS_PATH_COLLISION_BLOCKED:
        terminal_classification = "direct_path_collision_blocked"
        path_preflight = playback.summary.get("path_preflight", {})
        if isinstance(path_preflight, dict):
            reason_code = str(path_preflight.get("reason_code", reason_code))
    elif playback.motion_status == MOTION_STATUS_COMPLETED:
        terminal_classification = "reachable_motion_completed"
        reason_code = "motion_completed"
    else:
        terminal_classification = "execution_failed"
        reason_code = str(playback.motion_status)

    motion_emitted = bool(playback.summary.get("motion_emitted", False))
    if playback.motion_status in {
        MOTION_STATUS_ENDPOINT_REJECTED,
        MOTION_STATUS_PATH_COLLISION_BLOCKED,
    } and (motion_emitted or playback.frame_count != 0):
        raise ValueError("preflight-rejected random targets must emit zero motion")

    return RandomGraspAttemptOutcome(
        sample_index=sample_index,
        motion_status=str(playback.motion_status),
        reachability_status=reachability_status,
        terminal_classification=terminal_classification,
        reason_code=reason_code,
        motion_emitted=motion_emitted,
        frame_count=int(playback.frame_count),
        target_pose_content_hash=content_hash,
    )


def outcome_color_rgba(outcome: RandomGraspAttemptOutcome) -> np.ndarray:
    classification = outcome.terminal_classification
    if classification == "reachable_motion_completed":
        return np.array([0.10, 0.95, 0.20, 1.0], dtype=np.float32)
    if classification == "proven_kinematically_unreachable":
        return np.array([1.0, 0.08, 0.08, 1.0], dtype=np.float32)
    if classification in {"endpoint_collision_blocked", "direct_path_collision_blocked"}:
        return np.array([1.0, 0.35, 0.05, 1.0], dtype=np.float32)
    if "exhausted" in classification:
        return np.array([1.0, 0.85, 0.05, 1.0], dtype=np.float32)
    return np.array([0.75, 0.20, 1.0, 1.0], dtype=np.float32)


def _rng_for_sample(
    *, seed: int, sample_index: int, stream_tag: int
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [int(seed), int(sample_index), int(stream_tag)]
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _sample_shoemake_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = (float(value) for value in rng.random(3))
    quaternion_xyzw = np.asarray(
        (
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ),
        dtype=float,
    )
    if quaternion_xyzw[3] < 0.0:
        quaternion_xyzw *= -1.0
    return Rotation.from_quat(quaternion_xyzw).as_matrix()


__all__ = [
    "CONTINUOUS_RANDOM_REFERENCE_FRAME_ID",
    "CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC",
    "CONTINUOUS_RANDOM_REFERENCE_FRAME_SPEC_HASH",
    "JOINT_FK_SAMPLER_ID",
    "ContinuousRandomGraspPoseConfig",
    "RandomGraspAttemptOutcome",
    "RawCartesianGraspTargetStream",
    "classify_random_grasp_attempt",
    "outcome_color_rgba",
    "sample_random_joint_fk_target",
]
