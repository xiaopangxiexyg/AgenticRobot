"""Pure mixed-control adapter for a robosuite-like A0 backend port."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import RobosuiteResetManifest, RobosuiteResetReceipt

ROBOSUITE_MIXED_CONTROL_ACTION_HASH_DOMAIN_A0 = (
    "dexteleop.robosuite.mixed_control_action.a0"
)
ROBOSUITE_BACKEND_STEP_HASH_DOMAIN_A0 = "dexteleop.robosuite.backend_step.a0"
ROBOSUITE_OBSERVATION_HASH_DOMAIN_A0 = "dexteleop.robosuite.observation.a0"
ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0 = "base_only"
ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0 = "relative_overlay"
ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0 = "absolute_overlay_owner"
ROBOSUITE_MIXED_CONTROL_MODES_A0 = frozenset(
    {
        ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
        ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
        ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
    }
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite_vector(
    values: tuple[float, ...] | list[float],
    *,
    name: str,
    dimension: int,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != dimension:
        raise ValueError(f"{name} must have dimension {dimension}.")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


def _observation_hash(observation: Mapping[str, Any]) -> str:
    if not isinstance(observation, Mapping):
        raise TypeError("backend observation must be a mapping.")
    return canonical_content_hash(
        dict(observation),
        domain=ROBOSUITE_OBSERVATION_HASH_DOMAIN_A0,
    )


@dataclass(frozen=True)
class MixedControlActionSelection:
    """One final-envelope selection; it never changes control authority."""

    mode: str
    base_action: tuple[float, ...]
    intervention_action: tuple[float, ...]
    preclip_action: tuple[float, ...]
    applied_action: tuple[float, ...]
    clipped_dimensions: tuple[int, ...]
    authoritative: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        mode = _required(self.mode, name="mode")
        if mode not in ROBOSUITE_MIXED_CONTROL_MODES_A0:
            raise ValueError(f"Unsupported mixed-control mode {mode!r}.")
        dimension = len(self.applied_action)
        if dimension == 0:
            raise ValueError("applied_action must not be empty.")
        base = _finite_vector(self.base_action, name="base_action", dimension=dimension)
        intervention = _finite_vector(
            self.intervention_action,
            name="intervention_action",
            dimension=dimension,
        )
        preclip = _finite_vector(
            self.preclip_action,
            name="preclip_action",
            dimension=dimension,
        )
        applied = _finite_vector(
            self.applied_action,
            name="applied_action",
            dimension=dimension,
        )
        clipped = tuple(int(index) for index in self.clipped_dimensions)
        if any(index < 0 or index >= dimension for index in clipped):
            raise ValueError("clipped_dimensions contains an invalid index.")
        if len(clipped) != len(set(clipped)):
            raise ValueError("clipped_dimensions must be unique.")
        if self.authoritative is not False:
            raise ValueError("A0 allocation selection must remain non-authoritative.")
        payload = {
            "mode": mode,
            "base_action": base,
            "intervention_action": intervention,
            "preclip_action": preclip,
            "applied_action": applied,
            "clipped_dimensions": clipped,
            "authority_semantics": "composer_stays_authoritative_soft_selection",
        }
        for name, value in payload.items():
            if name != "authority_semantics":
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_MIXED_CONTROL_ACTION_HASH_DOMAIN_A0,
            ),
        )

    @property
    def selection_id(self) -> str:
        return f"robosuite-mixed-control-action:{self.semantic_content_hash}"


@dataclass(frozen=True)
class RobosuiteBackendStep:
    """Normalized result returned by the A0 backend seam."""

    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    observation_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        observation = MappingProxyType(dict(self.observation))
        info = MappingProxyType(dict(self.info))
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite.")
        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise TypeError("terminated and truncated must be bool values.")
        observation_hash = _observation_hash(observation)
        payload = {
            "observation_hash": observation_hash,
            "reward": reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": info,
        }
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "observation_hash", observation_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_BACKEND_STEP_HASH_DOMAIN_A0,
            ),
        )


@runtime_checkable
class RobosuiteA0Backend(Protocol):
    """Minimal injected seam; no concrete robosuite package is imported."""

    @property
    def backend_id(self) -> str: ...

    def reset(self, manifest: RobosuiteResetManifest) -> Mapping[str, Any]: ...

    def step(self, action: tuple[float, ...]) -> RobosuiteBackendStep: ...


class RobosuiteMixedControlAdapter:
    """Compose an already-selected soft mode and enforce the final envelope."""

    def __init__(
        self,
        *,
        action_low: tuple[float, ...] | list[float],
        action_high: tuple[float, ...] | list[float],
    ) -> None:
        low = tuple(float(value) for value in action_low)
        high = tuple(float(value) for value in action_high)
        if not low or len(low) != len(high):
            raise ValueError("action bounds must be non-empty and have equal length.")
        if any(
            not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
            for lower, upper in zip(low, high, strict=True)
        ):
            raise ValueError("every action bound must be finite with low < high.")
        self._action_low = low
        self._action_high = high

    @property
    def action_dimension(self) -> int:
        return len(self._action_low)

    @property
    def supported_modes(self) -> frozenset[str]:
        return ROBOSUITE_MIXED_CONTROL_MODES_A0

    def compose(
        self,
        *,
        mode: str,
        base_action: tuple[float, ...] | list[float],
        intervention_action: tuple[float, ...] | list[float] | None = None,
    ) -> MixedControlActionSelection:
        normalized_mode = _required(mode, name="mode")
        if normalized_mode not in ROBOSUITE_MIXED_CONTROL_MODES_A0:
            raise ValueError(f"Unsupported mixed-control mode {normalized_mode!r}.")
        base = _finite_vector(
            base_action,
            name="base_action",
            dimension=self.action_dimension,
        )
        if intervention_action is None:
            if normalized_mode != ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0:
                raise ValueError(
                    "intervention_action is required outside base_only mode."
                )
            intervention = (0.0,) * self.action_dimension
        else:
            intervention = _finite_vector(
                intervention_action,
                name="intervention_action",
                dimension=self.action_dimension,
            )
        if normalized_mode == ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0:
            preclip = base
        elif normalized_mode == ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0:
            preclip = tuple(
                base_value + intervention_value
                for base_value, intervention_value in zip(
                    base,
                    intervention,
                    strict=True,
                )
            )
        elif normalized_mode == ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0:
            preclip = intervention
        else:  # pragma: no cover - guarded by the mode set above.
            raise AssertionError("unreachable mixed-control mode")
        applied = tuple(
            min(max(value, lower), upper)
            for value, lower, upper in zip(
                preclip,
                self._action_low,
                self._action_high,
                strict=True,
            )
        )
        clipped = tuple(
            index
            for index, (before, after) in enumerate(zip(preclip, applied, strict=True))
            if before != after
        )
        return MixedControlActionSelection(
            mode=normalized_mode,
            base_action=base,
            intervention_action=intervention,
            preclip_action=preclip,
            applied_action=applied,
            clipped_dimensions=clipped,
        )

    def reset_backend(
        self,
        backend: RobosuiteA0Backend,
        manifest: RobosuiteResetManifest,
    ) -> tuple[Mapping[str, Any], RobosuiteResetReceipt]:
        if not isinstance(manifest, RobosuiteResetManifest):
            raise TypeError("manifest must be a RobosuiteResetManifest.")
        if manifest.action_dimension != self.action_dimension:
            raise ValueError("manifest action dimension does not match the adapter.")
        observation = MappingProxyType(dict(backend.reset(manifest)))
        if tuple(observation) != manifest.observation_keys:
            raise ValueError(
                "backend reset observation keys must exactly match the manifest."
            )
        observation_hash = _observation_hash(observation)
        receipt = RobosuiteResetReceipt(
            backend_id=_required(backend.backend_id, name="backend.backend_id"),
            manifest_id=manifest.manifest_id,
            manifest_hash=manifest.semantic_content_hash,
            initial_observation_hash=observation_hash,
        )
        return observation, receipt


__all__ = [
    "ROBOSUITE_BACKEND_STEP_HASH_DOMAIN_A0",
    "ROBOSUITE_MIXED_CONTROL_ACTION_HASH_DOMAIN_A0",
    "ROBOSUITE_MIXED_CONTROL_MODES_A0",
    "ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0",
    "ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0",
    "ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0",
    "ROBOSUITE_OBSERVATION_HASH_DOMAIN_A0",
    "MixedControlActionSelection",
    "RobosuiteA0Backend",
    "RobosuiteBackendStep",
    "RobosuiteMixedControlAdapter",
]
