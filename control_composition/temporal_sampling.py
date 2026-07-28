"""Optional source-time sampling before multi-source composition.

The composition runtime itself performs same-target selection and never reads a
producer queue or action chunk.  A source owner may use this module when it can
provide a typed next step that is guaranteed to follow the current step.  If no
such guarantee exists, the sampler returns the current action unchanged.

This is source-intent interpolation, not execution reference generation,
canonical safety, IK, smoothing, authority, or dispatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction


GUARANTEED_NEXT_COMPONENT_ACTION_HASH_DOMAIN = (
    "dexteleop.guaranteed_next_component_action.development"
)
COMPONENT_ACTION_TEMPORAL_SAMPLE_HASH_DOMAIN = (
    "dexteleop.component_action_temporal_sample.development"
)
COMPONENT_ACTION_INTERPOLATION_CONTRACT_HASH_DOMAIN = (
    "dexteleop.component_action_interpolation_contract.development"
)

SOURCE_TEMPORAL_DERIVATION_KINDS = frozenset(
    {
        "current_step_fallback",
        "interpolated_guaranteed_next",
        "guaranteed_next_endpoint",
    }
)

_INTERPOLATION_ALGORITHM_BY_FAMILY = {
    "cartesian_pose": "xyz_linear+quaternion_shortest_path_slerp",
    "joint_position": "linear",
    "continuous_native": "linear",
    "discrete_mode": "step_at_next",
}


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class GuaranteedNextComponentAction:
    """A source-owner assertion that one action is the next executed step.

    Merely peeking a queue or receiving an uncommitted prediction is not enough
    to construct this value.  The source owner must guarantee that this exact
    action follows the current step without replacement.  Callers that cannot
    make that assertion must pass ``None`` to the sampler.
    """

    guarantee_id: str
    source_fact_id: str
    source_fact_hash: str
    action: ComponentAction
    target_effective_at: float
    clock_domain_id: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        guarantee_id = _required(self.guarantee_id, name="guarantee_id")
        source_fact_id = _required(self.source_fact_id, name="source_fact_id")
        source_fact_hash = _required(
            self.source_fact_hash,
            name="source_fact_hash",
        )
        clock_domain_id = _required(
            self.clock_domain_id,
            name="clock_domain_id",
        )
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        payload = {
            "guarantee_id": guarantee_id,
            "source_fact_id": source_fact_id,
            "source_fact_hash": source_fact_hash,
            "action_hash": self.action.semantic_content_hash,
            "action_schema_hash": (
                self.action.action_schema.semantic_content_hash
            ),
            "target_effective_at": target,
            "clock_domain_id": clock_domain_id,
            "guarantee_semantics": "exact_next_step_will_execute",
        }
        object.__setattr__(self, "guarantee_id", guarantee_id)
        object.__setattr__(self, "source_fact_id", source_fact_id)
        object.__setattr__(self, "source_fact_hash", source_fact_hash)
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "clock_domain_id", clock_domain_id)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=GUARANTEED_NEXT_COMPONENT_ACTION_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class ComponentActionTemporalSample:
    """One immutable source action evaluated at a requested target time."""

    action: ComponentAction
    action_hash: str
    action_schema_hash: str
    target_effective_at: float
    clock_domain_id: str
    derivation_kind: str
    interpolation_alpha: float
    current_source_fact_id: str
    current_source_fact_hash: str
    current_action_hash: str
    next_step_guarantee_id: str | None
    next_step_guarantee_hash: str | None
    next_source_fact_id: str | None
    next_source_fact_hash: str | None
    next_action_hash: str | None
    interpolation_contract_hash: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        if self.action_hash != self.action.semantic_content_hash:
            raise ValueError("Temporal sample action hash mismatch.")
        if (
            self.action_schema_hash
            != self.action.action_schema.semantic_content_hash
        ):
            raise ValueError("Temporal sample action schema hash mismatch.")
        if self.derivation_kind not in SOURCE_TEMPORAL_DERIVATION_KINDS:
            raise ValueError("Unsupported source temporal derivation kind.")
        object.__setattr__(
            self,
            "target_effective_at",
            _finite(self.target_effective_at, name="target_effective_at"),
        )
        for name in (
            "clock_domain_id",
            "current_source_fact_id",
            "current_source_fact_hash",
            "current_action_hash",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        alpha = _finite(self.interpolation_alpha, name="interpolation_alpha")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("interpolation_alpha must be in [0, 1].")
        object.__setattr__(self, "interpolation_alpha", alpha)
        next_fields = (
            self.next_step_guarantee_id,
            self.next_step_guarantee_hash,
            self.next_source_fact_id,
            self.next_source_fact_hash,
            self.next_action_hash,
        )
        has_next = all(value is not None for value in next_fields)
        if any(value is not None for value in next_fields) != has_next:
            raise ValueError("Temporal sample next-step lineage must be complete.")
        if has_next:
            for name in (
                "next_step_guarantee_id",
                "next_step_guarantee_hash",
                "next_source_fact_id",
                "next_source_fact_hash",
                "next_action_hash",
            ):
                object.__setattr__(
                    self,
                    name,
                    _required(getattr(self, name), name=name),
                )
        if self.derivation_kind == "current_step_fallback":
            if alpha != 0.0 or has_next or self.interpolation_contract_hash is not None:
                raise ValueError(
                    "Current-step fallback cannot carry interpolation lineage."
                )
            if self.action_hash != self.current_action_hash:
                raise ValueError("Current-step fallback must preserve the action.")
        else:
            if not has_next or self.interpolation_contract_hash is None:
                raise ValueError(
                    "Interpolated samples require complete next-step lineage."
                )
            object.__setattr__(
                self,
                "interpolation_contract_hash",
                _required(
                    self.interpolation_contract_hash,
                    name="interpolation_contract_hash",
                ),
            )
            if self.derivation_kind == "interpolated_guaranteed_next" and not (
                0.0 < alpha < 1.0
            ):
                raise ValueError(
                    "Interpolated guaranteed-next samples require alpha in (0, 1)."
                )
            if self.derivation_kind == "guaranteed_next_endpoint" and alpha != 1.0:
                raise ValueError(
                    "Guaranteed-next endpoint samples require alpha equal to 1."
                )

    @property
    def sample_id(self) -> str:
        return f"component-action-temporal-sample:{self.semantic_content_hash}"

    @property
    def source_fact_ids(self) -> tuple[str, ...]:
        if self.next_source_fact_id is None:
            return (self.sample_id, self.current_source_fact_id)
        return (
            self.sample_id,
            self.current_source_fact_id,
            self.next_source_fact_id,
        )

    @property
    def source_fact_hashes(self) -> tuple[str, ...]:
        if self.next_source_fact_hash is None:
            return (self.semantic_content_hash, self.current_source_fact_hash)
        return (
            self.semantic_content_hash,
            self.current_source_fact_hash,
            self.next_source_fact_hash,
        )


def _slerp_quaternion_xyzw(
    current: np.ndarray,
    next_value: np.ndarray,
    alpha: float,
) -> np.ndarray:
    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(next_value, dtype=np.float64)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    left_weight = math.sin((1.0 - alpha) * angle) / sin_angle
    right_weight = math.sin(alpha * angle) / sin_angle
    result = left_weight * left + right_weight * right
    return result / np.linalg.norm(result)


def _interpolate_component_values(
    current: ComponentAction,
    next_action: ComponentAction,
    *,
    alpha: float,
) -> tuple[ComponentAction, str]:
    algorithms: dict[str, str] = {}
    values_by_component: dict[str, np.ndarray] = {}
    for component_id in current.component_coverage:
        current_value = current.value(component_id)
        next_value = next_action.value(component_id)
        schema = current_value.payload_schema
        family = schema.representation_family
        algorithm = _INTERPOLATION_ALGORITHM_BY_FAMILY.get(family)
        if algorithm is None:
            raise ValueError(
                "No source-time interpolation algorithm is registered for "
                f"representation family {family!r}; omit guaranteed_next_step "
                "to retain current-step behavior."
            )
        left = np.asarray(current_value.values)
        right = np.asarray(next_value.values)
        if family == "cartesian_pose":
            output = np.empty(7, dtype=np.float64)
            output[:3] = (1.0 - alpha) * left[:3] + alpha * right[:3]
            output[3:7] = _slerp_quaternion_xyzw(
                left[3:7],
                right[3:7],
                alpha,
            )
        elif family == "discrete_mode":
            output = left if alpha < 1.0 else right
        else:
            output = (1.0 - alpha) * left + alpha * right
        values_by_component[component_id] = np.asarray(
            output,
            dtype=np.dtype(schema.dtype),
        )
        algorithms[component_id] = algorithm
    contract_hash = canonical_content_hash(
        {
            "action_schema_hash": current.action_schema.semantic_content_hash,
            "component_coverage": current.component_coverage,
            "algorithm_by_component": algorithms,
        },
        domain=COMPONENT_ACTION_INTERPOLATION_CONTRACT_HASH_DOMAIN,
    )
    return (
        ComponentAction(
            action_schema=current.action_schema,
            values_by_component=values_by_component,
        ),
        contract_hash,
    )


def sample_component_action_source(
    *,
    current_action: ComponentAction,
    current_target_effective_at: float,
    current_source_fact_id: str,
    current_source_fact_hash: str,
    target_effective_at: float,
    clock_domain_id: str,
    guaranteed_next_step: GuaranteedNextComponentAction | None = None,
) -> ComponentActionTemporalSample:
    """Sample one source action without guessing a future endpoint.

    ``guaranteed_next_step=None`` is the compatibility path: the current action
    is returned unchanged.  A supplied guarantee must use the same schema,
    component coverage and clock, and its target must follow the current target.
    Structural or temporal errors in a claimed guarantee are rejected rather
    than silently treated as an unguaranteed next step.
    """

    if not isinstance(current_action, ComponentAction):
        raise TypeError("current_action must be a ComponentAction.")
    current_target = _finite(
        current_target_effective_at,
        name="current_target_effective_at",
    )
    target = _finite(target_effective_at, name="target_effective_at")
    if target < current_target:
        raise ValueError("target_effective_at cannot precede the current step.")
    current_fact_id = _required(
        current_source_fact_id,
        name="current_source_fact_id",
    )
    current_fact_hash = _required(
        current_source_fact_hash,
        name="current_source_fact_hash",
    )
    clock = _required(clock_domain_id, name="clock_domain_id")

    next_guarantee_id: str | None = None
    next_guarantee_hash: str | None = None
    next_fact_id: str | None = None
    next_fact_hash: str | None = None
    next_action_hash: str | None = None
    interpolation_contract_hash: str | None = None
    alpha = 0.0
    action = current_action
    derivation = "current_step_fallback"

    if guaranteed_next_step is not None:
        if not isinstance(
            guaranteed_next_step,
            GuaranteedNextComponentAction,
        ):
            raise TypeError(
                "guaranteed_next_step must be a "
                "GuaranteedNextComponentAction or None."
            )
        next_step = guaranteed_next_step
        if next_step.clock_domain_id != clock:
            raise ValueError("Current and guaranteed-next steps use another clock.")
        if next_step.target_effective_at <= current_target:
            raise ValueError(
                "Guaranteed-next target must follow the current target."
            )
        if target > next_step.target_effective_at:
            raise ValueError(
                "Sampling after the guaranteed-next target requires advancing "
                "the current step."
            )
        if (
            next_step.action.action_schema.semantic_content_hash
            != current_action.action_schema.semantic_content_hash
        ):
            raise ValueError(
                "Current and guaranteed-next actions use another action schema."
            )
        if next_step.action.component_coverage != current_action.component_coverage:
            raise ValueError(
                "Current and guaranteed-next actions must have equal component "
                "coverage."
            )
        alpha = (target - current_target) / (
            next_step.target_effective_at - current_target
        )
        next_guarantee_id = next_step.guarantee_id
        next_guarantee_hash = next_step.semantic_content_hash
        next_fact_id = next_step.source_fact_id
        next_fact_hash = next_step.source_fact_hash
        next_action_hash = next_step.action.semantic_content_hash
        if alpha > 0.0:
            action, interpolation_contract_hash = _interpolate_component_values(
                current_action,
                next_step.action,
                alpha=alpha,
            )
            derivation = (
                "guaranteed_next_endpoint"
                if math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-12)
                else "interpolated_guaranteed_next"
            )
        else:
            next_guarantee_id = None
            next_guarantee_hash = None
            next_fact_id = None
            next_fact_hash = None
            next_action_hash = None

    payload = {
        "action_hash": action.semantic_content_hash,
        "action_schema_hash": action.action_schema.semantic_content_hash,
        "target_effective_at": target,
        "clock_domain_id": clock,
        "derivation_kind": derivation,
        "interpolation_alpha": alpha,
        "current_source_fact_id": current_fact_id,
        "current_source_fact_hash": current_fact_hash,
        "current_action_hash": current_action.semantic_content_hash,
        "next_step_guarantee_id": next_guarantee_id,
        "next_step_guarantee_hash": next_guarantee_hash,
        "next_source_fact_id": next_fact_id,
        "next_source_fact_hash": next_fact_hash,
        "next_action_hash": next_action_hash,
        "interpolation_contract_hash": interpolation_contract_hash,
    }
    digest = canonical_content_hash(
        payload,
        domain=COMPONENT_ACTION_TEMPORAL_SAMPLE_HASH_DOMAIN,
    )
    return ComponentActionTemporalSample(
        action=action,
        semantic_content_hash=digest,
        **payload,
    )


__all__ = [
    "COMPONENT_ACTION_INTERPOLATION_CONTRACT_HASH_DOMAIN",
    "COMPONENT_ACTION_TEMPORAL_SAMPLE_HASH_DOMAIN",
    "GUARANTEED_NEXT_COMPONENT_ACTION_HASH_DOMAIN",
    "SOURCE_TEMPORAL_DERIVATION_KINDS",
    "ComponentActionTemporalSample",
    "GuaranteedNextComponentAction",
    "sample_component_action_source",
]
