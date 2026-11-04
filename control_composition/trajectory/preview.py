"""Pure whole-horizon previews of source-owned trajectory intent.

The preview in this module never evaluates composition, advances a source
cursor, closes a composition target, publishes a proposal, or claims that an
action will execute.  It exists so any UI or observer can render one exact
``ComponentActionTrajectory`` revision without depending on a simulator,
model architecture, or device backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.trajectory import (
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
)

from ._validation import _finite, _required


TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED = "producer_locked_intent"
TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE = "speculative_prediction"
TRAJECTORY_INTENT_PREVIEW_CLASSES_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED,
        TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE,
    }
)
TRAJECTORY_INTENT_PREVIEW_POINT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_intent_preview_point.development"
)
TRAJECTORY_INTENT_PREVIEW_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_intent_preview_snapshot.development"
)
TRAJECTORY_INTENT_PREVIEW_EXECUTION_EVIDENCE_STATUS = (
    "external_downstream_observation_only"
)


def _preview_class_for(
    trajectory: ComponentActionTrajectory,
    keyframe: ComponentActionTrajectoryKeyframe,
) -> str:
    boundary = trajectory.source_immutable_through
    if boundary is not None and keyframe.target_effective_at <= boundary:
        return TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED
    return TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE


@dataclass(frozen=True)
class TrajectoryIntentPreviewPoint:
    """One displayable source-intent keyframe with an explicit trust class."""

    keyframe: ComponentActionTrajectoryKeyframe
    preview_class: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.keyframe, ComponentActionTrajectoryKeyframe):
            raise TypeError("keyframe must be a ComponentActionTrajectoryKeyframe.")
        preview_class = _required(
            self.preview_class,
            name="preview_class",
        )
        if preview_class not in TRAJECTORY_INTENT_PREVIEW_CLASSES_DEVELOPMENT:
            raise ValueError("Unsupported trajectory intent preview class.")
        object.__setattr__(self, "preview_class", preview_class)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "keyframe_hash": self.keyframe.semantic_content_hash,
                    "preview_class": preview_class,
                    "execution_claimed": False,
                },
                domain=(TRAJECTORY_INTENT_PREVIEW_POINT_HASH_DOMAIN_DEVELOPMENT),
            ),
        )

    @property
    def point_id(self) -> str:
        return f"trajectory-intent-preview-point:{self.semantic_content_hash}"

    @property
    def action(self):
        return self.keyframe.action

    @property
    def target_effective_at(self) -> float:
        return self.keyframe.target_effective_at

    @property
    def effective_until(self) -> float:
        return self.keyframe.effective_until

    @property
    def source_index(self) -> int:
        return self.keyframe.source_index

    @property
    def execution_guaranteed(self) -> bool:
        return False


@dataclass(frozen=True)
class TrajectoryIntentPreviewSnapshot:
    """A coherent preview of every keyframe in one exact trajectory revision.

    ``generated_at`` is observer metadata only.  It does not alter the source
    target-time lattice.  Actual Reference/Measurement observations remain
    separate downstream facts and may be overlaid by the UI independently.
    """

    trajectory: ComponentActionTrajectory
    generated_at: float
    generated_clock_domain_id: str
    points: tuple[TrajectoryIntentPreviewPoint, ...]
    execution_evidence_status: str = TRAJECTORY_INTENT_PREVIEW_EXECUTION_EVIDENCE_STATUS
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ComponentActionTrajectory):
            raise TypeError("trajectory must be a ComponentActionTrajectory.")
        generated_at = _finite(self.generated_at, name="generated_at")
        clock = _required(
            self.generated_clock_domain_id,
            name="generated_clock_domain_id",
        )
        points = tuple(self.points)
        if len(points) != len(self.trajectory.keyframes):
            raise ValueError(
                "A whole-horizon preview must cover every trajectory keyframe."
            )
        for point, keyframe in zip(points, self.trajectory.keyframes):
            if not isinstance(point, TrajectoryIntentPreviewPoint):
                raise TypeError("points must contain TrajectoryIntentPreviewPoint.")
            if point.keyframe.semantic_content_hash != keyframe.semantic_content_hash:
                raise ValueError(
                    "Preview points must preserve trajectory keyframe order."
                )
            if point.preview_class != _preview_class_for(
                self.trajectory,
                keyframe,
            ):
                raise ValueError(
                    "Preview class disagrees with source_immutable_through."
                )
        evidence_status = _required(
            self.execution_evidence_status,
            name="execution_evidence_status",
        )
        if evidence_status != TRAJECTORY_INTENT_PREVIEW_EXECUTION_EVIDENCE_STATUS:
            raise ValueError("Source-intent preview cannot carry execution evidence.")
        payload = {
            "trajectory_id": self.trajectory.trajectory_id,
            "trajectory_revision": self.trajectory.revision,
            "trajectory_hash": self.trajectory.semantic_content_hash,
            "generated_at": generated_at,
            "generated_clock_domain_id": clock,
            "point_hashes": tuple(point.semantic_content_hash for point in points),
            "execution_evidence_status": evidence_status,
            "authoritative": False,
        }
        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "generated_clock_domain_id", clock)
        object.__setattr__(self, "points", points)
        object.__setattr__(
            self,
            "execution_evidence_status",
            evidence_status,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(TRAJECTORY_INTENT_PREVIEW_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT),
            ),
        )

    @property
    def preview_id(self) -> str:
        return f"trajectory-intent-preview:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def producer_locked_points(self) -> tuple[TrajectoryIntentPreviewPoint, ...]:
        return tuple(
            point
            for point in self.points
            if point.preview_class == TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED
        )

    @property
    def speculative_points(self) -> tuple[TrajectoryIntentPreviewPoint, ...]:
        return tuple(
            point
            for point in self.points
            if point.preview_class == TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE
        )


def build_trajectory_intent_preview(
    trajectory: ComponentActionTrajectory,
    *,
    generated_at: float,
    generated_clock_domain_id: str,
) -> TrajectoryIntentPreviewSnapshot:
    """Project one immutable trajectory without mutating any runtime owner."""

    if not isinstance(trajectory, ComponentActionTrajectory):
        raise TypeError("trajectory must be a ComponentActionTrajectory.")
    return TrajectoryIntentPreviewSnapshot(
        trajectory=trajectory,
        generated_at=generated_at,
        generated_clock_domain_id=generated_clock_domain_id,
        points=tuple(
            TrajectoryIntentPreviewPoint(
                keyframe=keyframe,
                preview_class=_preview_class_for(trajectory, keyframe),
            )
            for keyframe in trajectory.keyframes
        ),
    )


__all__ = [
    "TRAJECTORY_INTENT_PREVIEW_CLASSES_DEVELOPMENT",
    "TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED",
    "TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE",
    "TRAJECTORY_INTENT_PREVIEW_EXECUTION_EVIDENCE_STATUS",
    "TRAJECTORY_INTENT_PREVIEW_POINT_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_INTENT_PREVIEW_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT",
    "TrajectoryIntentPreviewPoint",
    "TrajectoryIntentPreviewSnapshot",
    "build_trajectory_intent_preview",
]
