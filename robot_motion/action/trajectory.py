"""Backend-neutral trajectories of canonical component actions.

The contracts in this module describe immutable source intent over an
explicit time lattice.  They do not own a producer queue, cursor, authority,
safety, target admission, reference generation, or dispatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash

from .component_action import ComponentAction


COMPONENT_ACTION_TRAJECTORY_KEYFRAME_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_action_trajectory_keyframe.development"
)
COMPONENT_ACTION_TRAJECTORY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_action_trajectory.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _lineage(
    ids: tuple[str, ...],
    hashes: tuple[str, ...],
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_ids = tuple(
        _required(value, name=f"{name}_id") for value in ids
    )
    normalized_hashes = tuple(
        _required(value, name=f"{name}_hash") for value in hashes
    )
    if not normalized_ids or len(normalized_ids) != len(normalized_hashes):
        raise ValueError(
            f"{name} ids/hashes must be non-empty and have equal length."
        )
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError(f"{name} ids must be unique.")
    return normalized_ids, normalized_hashes


@dataclass(frozen=True)
class ComponentActionTrajectoryKeyframe:
    """One canonical action on a source-owned target-time lattice."""

    target_effective_at: float
    effective_until: float
    action: ComponentAction
    source_index: int
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        until = _finite(self.effective_until, name="effective_until")
        if until <= target:
            raise ValueError("A trajectory keyframe interval must be non-empty.")
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        source_index = _nonnegative_int(
            self.source_index,
            name="source_index",
        )
        fact_ids, fact_hashes = _lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
            name="source_fact",
        )
        payload = {
            "target_effective_at": target,
            "effective_until": until,
            "action_hash": self.action.semantic_content_hash,
            "action_schema_hash": (
                self.action.action_schema.semantic_content_hash
            ),
            "component_coverage": self.action.component_coverage,
            "source_index": source_index,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
        }
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "effective_until", until)
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "source_fact_ids", fact_ids)
        object.__setattr__(self, "source_fact_hashes", fact_hashes)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPONENT_ACTION_TRAJECTORY_KEYFRAME_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def keyframe_id(self) -> str:
        return f"component-action-trajectory-keyframe:{self.semantic_content_hash}"


@dataclass(frozen=True)
class ComponentActionTrajectory:
    """One immutable, model- and backend-neutral nominal trajectory.

    ``source_instance_id`` distinguishes independently restarted source
    instances.  The optional source activation pair identifies one explicit
    start/re-arm activation.  Pass-through projection may omit that pair, but
    a residual session's nominal-base binding requires it so a late plan from
    an earlier activation cannot become the base silently.

    ``source_immutable_through`` is optional producer lineage.  It says only
    that the source will not rewrite that prefix; it is not execution,
    target-admission, or reference-activation evidence.
    """

    trajectory_id: str
    revision: int
    source_kind: str
    source_instance_id: str
    clock_domain_id: str
    created_at: float
    keyframes: tuple[ComponentActionTrajectoryKeyframe, ...]
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    source_activation_id: str | None = None
    source_activation_hash: str | None = None
    source_immutable_through: float | None = None
    coordination_group_id: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        trajectory_id = _required(self.trajectory_id, name="trajectory_id")
        revision = _nonnegative_int(self.revision, name="revision")
        source_kind = _required(self.source_kind, name="source_kind")
        source_instance_id = _required(
            self.source_instance_id,
            name="source_instance_id",
        )
        clock_domain_id = _required(
            self.clock_domain_id,
            name="clock_domain_id",
        )
        created_at = _finite(self.created_at, name="created_at")
        keyframes = tuple(self.keyframes)
        if not keyframes:
            raise ValueError("A component action trajectory must have keyframes.")
        fact_ids, fact_hashes = _lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
            name="source_fact",
        )
        activation_id = self.source_activation_id
        activation_hash = self.source_activation_hash
        if (activation_id is None) != (activation_hash is None):
            raise ValueError(
                "source_activation_id/hash must be both present or both absent."
            )
        if activation_id is not None:
            activation_id = _required(
                activation_id,
                name="source_activation_id",
            )
            activation_hash = _required(
                activation_hash,
                name="source_activation_hash",
            )
        immutable_through = self.source_immutable_through
        if immutable_through is not None:
            immutable_through = _finite(
                immutable_through,
                name="source_immutable_through",
            )
        coordination_group_id = self.coordination_group_id
        if coordination_group_id is not None:
            coordination_group_id = _required(
                coordination_group_id,
                name="coordination_group_id",
            )

        first = keyframes[0]
        schema_hash = first.action.action_schema.semantic_content_hash
        component_coverage = first.action.component_coverage
        previous_target: float | None = None
        previous_source_index: int | None = None
        for keyframe in keyframes:
            if not isinstance(keyframe, ComponentActionTrajectoryKeyframe):
                raise TypeError(
                    "keyframes must contain ComponentActionTrajectoryKeyframe."
                )
            if previous_target is not None and (
                keyframe.target_effective_at <= previous_target
            ):
                raise ValueError(
                    "Trajectory target times must be strictly increasing."
                )
            if previous_source_index is not None and (
                keyframe.source_index <= previous_source_index
            ):
                raise ValueError(
                    "Trajectory source indices must be strictly increasing."
                )
            if (
                keyframe.action.action_schema.semantic_content_hash
                != schema_hash
            ):
                raise ValueError(
                    "Trajectory keyframes must use one action schema."
                )
            if keyframe.action.component_coverage != component_coverage:
                raise ValueError(
                    "Trajectory keyframes must keep component coverage stable."
                )
            previous_target = keyframe.target_effective_at
            previous_source_index = keyframe.source_index

        payload = {
            "trajectory_id": trajectory_id,
            "revision": revision,
            "source_kind": source_kind,
            "source_instance_id": source_instance_id,
            "clock_domain_id": clock_domain_id,
            "created_at": created_at,
            "action_schema_hash": schema_hash,
            "component_coverage": component_coverage,
            "keyframe_hashes": tuple(
                keyframe.semantic_content_hash for keyframe in keyframes
            ),
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            "source_activation_id": activation_id,
            "source_activation_hash": activation_hash,
            "source_immutable_through": immutable_through,
            "coordination_group_id": coordination_group_id,
        }
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(
            self,
            "source_instance_id",
            source_instance_id,
        )
        object.__setattr__(self, "clock_domain_id", clock_domain_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "keyframes", keyframes)
        object.__setattr__(self, "source_fact_ids", fact_ids)
        object.__setattr__(self, "source_fact_hashes", fact_hashes)
        object.__setattr__(self, "source_activation_id", activation_id)
        object.__setattr__(
            self,
            "source_activation_hash",
            activation_hash,
        )
        object.__setattr__(
            self,
            "source_immutable_through",
            immutable_through,
        )
        object.__setattr__(
            self,
            "coordination_group_id",
            coordination_group_id,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_ACTION_TRAJECTORY_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def action_schema_hash(self) -> str:
        return self.keyframes[0].action.action_schema.semantic_content_hash

    @property
    def component_coverage(self) -> tuple[str, ...]:
        return self.keyframes[0].action.component_coverage

    @property
    def fact_id(self) -> str:
        """Immutable source-fact identity for this exact revision/content."""

        return f"component-action-trajectory:{self.semantic_content_hash}"

    @property
    def effective_from(self) -> float:
        return self.keyframes[0].target_effective_at

    @property
    def effective_until(self) -> float:
        return self.keyframes[-1].effective_until

    def keyframe_at(self, target_effective_at: float) -> (
        ComponentActionTrajectoryKeyframe | None
    ):
        target = _finite(
            target_effective_at,
            name="target_effective_at",
        )
        for keyframe in self.keyframes:
            if keyframe.target_effective_at == target:
                return keyframe
        return None


__all__ = [
    "COMPONENT_ACTION_TRAJECTORY_HASH_DOMAIN_DEVELOPMENT",
    "COMPONENT_ACTION_TRAJECTORY_KEYFRAME_HASH_DOMAIN_DEVELOPMENT",
    "ComponentActionTrajectory",
    "ComponentActionTrajectoryKeyframe",
]
