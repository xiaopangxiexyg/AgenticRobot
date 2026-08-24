"""Experiment-local policy, RTC, and component-mix contracts.

This module deliberately stays below deployment and dataset ownership.  It
provides a deterministic seam for comparing a nominal policy, an RTC policy
intervention, and synthetic human takeover input against the already reviewed
robosuite A1 backend.  The A0 vector adapter remains unchanged; in particular,
the gripper rule here is explicit and non-additive so that the old A0 relative
overlay is not accidentally reused as a production safety contract.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from sim.robosuite.adapter import (
    ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
)

A2_ACTION_DIMENSION = 7
A2_ARM_COMPONENT = "arm"
A2_GRIPPER_COMPONENT = "gripper"
A2_COMPONENT_IDS = (A2_ARM_COMPONENT, A2_GRIPPER_COMPONENT)
A2_ARM_INDICES = tuple(range(6))
A2_GRIPPER_INDEX = 6

A2_POLICY_SOURCE_KIND_STANDARD = "standard"
A2_POLICY_SOURCE_KIND_RTC = "rtc"
A2_POLICY_SOURCE_KINDS = frozenset(
    {A2_POLICY_SOURCE_KIND_STANDARD, A2_POLICY_SOURCE_KIND_RTC}
)
A2_POLICY_ACTION_ROLE_NOMINAL = "nominal"
A2_POLICY_ACTION_ROLE_INTERVENTION = "intervention"
A2_POLICY_ACTION_ROLES = frozenset(
    {A2_POLICY_ACTION_ROLE_NOMINAL, A2_POLICY_ACTION_ROLE_INTERVENTION}
)
A2_COMPONENT_OWNER_BASE = "base"
A2_COMPONENT_OWNER_INTERVENTION = "intervention"
A2_COMPONENT_OWNERS = frozenset(
    {A2_COMPONENT_OWNER_BASE, A2_COMPONENT_OWNER_INTERVENTION}
)
A2_MIX_MODES = frozenset(
    {
        ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
        ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
        ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
    }
)

A2_POLICY_REQUEST_HASH_DOMAIN = "dexteleop.streaming_intervention.policy_request.a2"
A2_POLICY_PROPOSAL_HASH_DOMAIN = "dexteleop.streaming_intervention.policy_proposal.a2"
A2_FRAME_HASH_DOMAIN = "dexteleop.streaming_intervention.frame.a2"
A2_COMPONENT_SELECTION_HASH_DOMAIN = (
    "dexteleop.streaming_intervention.component_selection.a2"
)
A2_EVIDENCE_HASH_DOMAIN = "dexteleop.streaming_intervention.evidence.a2"
A2_STEP_HASH_DOMAIN = "dexteleop.streaming_intervention.step.a2"
A2_RUN_HASH_DOMAIN = "dexteleop.streaming_intervention.run.a2"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _vector(
    values: Iterable[float],
    *,
    name: str,
    dimension: int | None = None,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if dimension is not None and len(normalized) != dimension:
        raise ValueError(f"{name} must have dimension {dimension}.")
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


def _components(
    values: Iterable[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique component ids.")
    unknown = sorted(set(normalized) - set(A2_COMPONENT_IDS))
    if unknown:
        raise ValueError(f"{name} contains unsupported components: {unknown}.")
    return tuple(component for component in A2_COMPONENT_IDS if component in normalized)


@dataclass(frozen=True)
class PolicyInferenceRequestA2:
    """Deterministic input passed to either a standard or RTC source."""

    episode_id: str
    frame_id: str
    step_index: int
    observation: Mapping[str, Any]
    nominal_action: tuple[float, ...] | None = None
    expected_source_revision: int | None = None
    observation_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        episode_id = _required(self.episode_id, name="episode_id")
        frame_id = _required(self.frame_id, name="frame_id")
        step_index = _nonnegative_int(self.step_index, name="step_index")
        observation = MappingProxyType(dict(self.observation))
        if not observation:
            raise ValueError("observation must not be empty.")
        nominal_action = (
            None
            if self.nominal_action is None
            else _vector(
                self.nominal_action,
                name="nominal_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        expected_revision = self.expected_source_revision
        if expected_revision is not None:
            expected_revision = _nonnegative_int(
                expected_revision,
                name="expected_source_revision",
            )
        observation_hash = canonical_content_hash(
            dict(observation),
            domain=A2_POLICY_REQUEST_HASH_DOMAIN,
        )
        payload = {
            "episode_id": episode_id,
            "frame_id": frame_id,
            "step_index": step_index,
            "observation_hash": observation_hash,
            "nominal_action": nominal_action,
            "expected_source_revision": expected_revision,
        }
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "nominal_action", nominal_action)
        object.__setattr__(self, "expected_source_revision", expected_revision)
        object.__setattr__(self, "observation_hash", observation_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_POLICY_REQUEST_HASH_DOMAIN),
        )


@dataclass(frozen=True)
class PolicyProposalA2:
    """One content-addressed policy result with source/revision lineage."""

    source_id: str
    source_kind: str
    action_role: str
    revision: int
    inference_id: str
    frame_id: str
    step_index: int
    observation_hash: str
    action: tuple[float, ...]
    component_ids: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_id = _required(self.source_id, name="source_id")
        source_kind = _required(self.source_kind, name="source_kind")
        if source_kind not in A2_POLICY_SOURCE_KINDS:
            raise ValueError(f"Unsupported policy source kind {source_kind!r}.")
        action_role = _required(self.action_role, name="action_role")
        if action_role not in A2_POLICY_ACTION_ROLES:
            raise ValueError(f"Unsupported policy action role {action_role!r}.")
        expected_role = (
            A2_POLICY_ACTION_ROLE_NOMINAL
            if source_kind == A2_POLICY_SOURCE_KIND_STANDARD
            else A2_POLICY_ACTION_ROLE_INTERVENTION
        )
        if action_role != expected_role:
            raise ValueError(
                f"{source_kind} policy sources must emit {expected_role!r} actions."
            )
        revision = _nonnegative_int(self.revision, name="revision")
        inference_id = _required(self.inference_id, name="inference_id")
        frame_id = _required(self.frame_id, name="frame_id")
        step_index = _nonnegative_int(self.step_index, name="step_index")
        observation_hash = _required(self.observation_hash, name="observation_hash")
        action = _vector(
            self.action,
            name="action",
            dimension=A2_ACTION_DIMENSION,
        )
        component_ids = _components(self.component_ids, name="component_ids")
        payload = {
            "source_id": source_id,
            "source_kind": source_kind,
            "action_role": action_role,
            "revision": revision,
            "inference_id": inference_id,
            "frame_id": frame_id,
            "step_index": step_index,
            "observation_hash": observation_hash,
            "action": action,
            "component_ids": component_ids,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_POLICY_PROPOSAL_HASH_DOMAIN),
        )


@runtime_checkable
class PolicyInferenceSourceA2(Protocol):
    """Common seam for normal policy and RTC inference implementations."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_kind(self) -> str: ...

    @property
    def action_role(self) -> str: ...

    @property
    def action_dimension(self) -> int: ...

    @property
    def component_ids(self) -> tuple[str, ...]: ...

    @property
    def revision(self) -> int: ...

    def infer(self, request: PolicyInferenceRequestA2) -> PolicyProposalA2: ...


@dataclass(frozen=True)
class SequencePolicySourceA2:
    """Deterministic provider used by offline tests and mock experiments.

    It is intentionally not a model client.  A deployment adapter can later
    implement :class:`PolicyInferenceSourceA2` without changing the runner or
    the lineage schema.
    """

    source_id: str
    source_kind: str
    actions: tuple[tuple[float, ...], ...]
    revision: int = 0
    component_ids: tuple[str, ...] | None = None
    action_role: str = field(init=False)
    action_dimension: int = field(init=False)

    def __post_init__(self) -> None:
        source_id = _required(self.source_id, name="source_id")
        source_kind = _required(self.source_kind, name="source_kind")
        if source_kind not in A2_POLICY_SOURCE_KINDS:
            raise ValueError(f"Unsupported policy source kind {source_kind!r}.")
        revision = _nonnegative_int(self.revision, name="revision")
        actions = tuple(
            _vector(action, name=f"actions[{index}]", dimension=A2_ACTION_DIMENSION)
            for index, action in enumerate(self.actions)
        )
        if not actions:
            raise ValueError("actions must contain at least one action.")
        component_ids = (
            (A2_ARM_COMPONENT,)
            if source_kind == A2_POLICY_SOURCE_KIND_RTC
            else A2_COMPONENT_IDS
        ) if self.component_ids is None else _components(
            self.component_ids,
            name="component_ids",
        )
        action_role = (
            A2_POLICY_ACTION_ROLE_NOMINAL
            if source_kind == A2_POLICY_SOURCE_KIND_STANDARD
            else A2_POLICY_ACTION_ROLE_INTERVENTION
        )
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "action_role", action_role)
        object.__setattr__(self, "action_dimension", A2_ACTION_DIMENSION)

    def infer(self, request: PolicyInferenceRequestA2) -> PolicyProposalA2:
        if not isinstance(request, PolicyInferenceRequestA2):
            raise TypeError("request must be a PolicyInferenceRequestA2.")
        if request.expected_source_revision is not None and (
            request.expected_source_revision != self.revision
        ):
            raise RuntimeError(
                "policy source revision changed before inference; refusing stale input."
            )
        if self.source_kind == A2_POLICY_SOURCE_KIND_RTC and (
            request.nominal_action is None
        ):
            raise ValueError("RTC inference requires a nominal_action reference.")
        if request.step_index >= len(self.actions):
            raise IndexError("deterministic policy source has no action for this step.")
        inference_id = (
            f"{self.source_id}:r{self.revision}:"
            f"{request.frame_id}:s{request.step_index}"
        )
        return PolicyProposalA2(
            source_id=self.source_id,
            source_kind=self.source_kind,
            action_role=self.action_role,
            revision=self.revision,
            inference_id=inference_id,
            frame_id=request.frame_id,
            step_index=request.step_index,
            observation_hash=request.observation_hash,
            action=self.actions[request.step_index],
            component_ids=self.component_ids,
        )


@dataclass(frozen=True)
class ComponentOwnershipA2:
    """Explicit per-component ownership for one mixed-control frame."""

    arm_owner: str = A2_COMPONENT_OWNER_BASE
    gripper_owner: str = A2_COMPONENT_OWNER_BASE
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        arm_owner = _required(self.arm_owner, name="arm_owner")
        gripper_owner = _required(self.gripper_owner, name="gripper_owner")
        if arm_owner not in A2_COMPONENT_OWNERS:
            raise ValueError(f"Unsupported arm owner {arm_owner!r}.")
        if gripper_owner not in A2_COMPONENT_OWNERS:
            raise ValueError(f"Unsupported gripper owner {gripper_owner!r}.")
        payload = {
            "arm_owner": arm_owner,
            "gripper_owner": gripper_owner,
            "gripper_semantics": "replace_when_intervention_owned_never_add",
        }
        object.__setattr__(self, "arm_owner", arm_owner)
        object.__setattr__(self, "gripper_owner", gripper_owner)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_COMPONENT_SELECTION_HASH_DOMAIN),
        )

    @classmethod
    def for_components(cls, component_ids: Iterable[str]) -> ComponentOwnershipA2:
        active = set(_components(component_ids, name="component_ids", allow_empty=True))
        return cls(
            arm_owner=(
                A2_COMPONENT_OWNER_INTERVENTION
                if A2_ARM_COMPONENT in active
                else A2_COMPONENT_OWNER_BASE
            ),
            gripper_owner=(
                A2_COMPONENT_OWNER_INTERVENTION
                if A2_GRIPPER_COMPONENT in active
                else A2_COMPONENT_OWNER_BASE
            ),
        )

    @property
    def active_component_ids(self) -> tuple[str, ...]:
        return tuple(
            component
            for component, owner in (
                (A2_ARM_COMPONENT, self.arm_owner),
                (A2_GRIPPER_COMPONENT, self.gripper_owner),
            )
            if owner == A2_COMPONENT_OWNER_INTERVENTION
        )


@dataclass(frozen=True)
class ComponentMixSelectionA2:
    """Component-aware final-envelope selection for the A2 experiment seam."""

    mode: str
    base_action: tuple[float, ...]
    intervention_action: tuple[float, ...]
    ownership: ComponentOwnershipA2
    preclip_action: tuple[float, ...]
    applied_action: tuple[float, ...]
    clipped_dimensions: tuple[int, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        mode = _required(self.mode, name="mode")
        if mode not in A2_MIX_MODES:
            raise ValueError(f"Unsupported A2 mix mode {mode!r}.")
        if not isinstance(self.ownership, ComponentOwnershipA2):
            raise TypeError("ownership must be a ComponentOwnershipA2.")
        base = _vector(self.base_action, name="base_action", dimension=A2_ACTION_DIMENSION)
        intervention = _vector(
            self.intervention_action,
            name="intervention_action",
            dimension=A2_ACTION_DIMENSION,
        )
        preclip = _vector(
            self.preclip_action,
            name="preclip_action",
            dimension=A2_ACTION_DIMENSION,
        )
        applied = _vector(
            self.applied_action,
            name="applied_action",
            dimension=A2_ACTION_DIMENSION,
        )
        clipped = tuple(int(index) for index in self.clipped_dimensions)
        if len(clipped) != len(set(clipped)) or any(
            index < 0 or index >= A2_ACTION_DIMENSION for index in clipped
        ):
            raise ValueError("clipped_dimensions contains an invalid index.")
        payload = {
            "mode": mode,
            "base_action": base,
            "intervention_action": intervention,
            "ownership_hash": self.ownership.semantic_content_hash,
            "preclip_action": preclip,
            "applied_action": applied,
            "clipped_dimensions": clipped,
            "gripper_semantics": "replace_when_intervention_owned_never_add",
        }
        for name, value in payload.items():
            if name != "ownership_hash" and name != "gripper_semantics":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "ownership", self.ownership)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_COMPONENT_SELECTION_HASH_DOMAIN),
        )


class RobosuiteComponentComposerA2:
    """Compose the fixed A1 6D arm + 1D gripper action layout.

    Arm residuals are additive only for ``relative_overlay``.  A gripper
    intervention always replaces the nominal gripper value; it is never added
    to it.  The final clip is part of this experiment envelope and is not a
    claim of canonical production safety.
    """

    def __init__(
        self,
        *,
        action_low: Iterable[float],
        action_high: Iterable[float],
    ) -> None:
        low = _vector(action_low, name="action_low", dimension=A2_ACTION_DIMENSION)
        high = _vector(action_high, name="action_high", dimension=A2_ACTION_DIMENSION)
        if any(lower >= upper for lower, upper in zip(low, high, strict=True)):
            raise ValueError("every action bound must have low < high.")
        self._action_low = low
        self._action_high = high

    @property
    def action_dimension(self) -> int:
        return A2_ACTION_DIMENSION

    @property
    def action_low(self) -> tuple[float, ...]:
        return self._action_low

    @property
    def action_high(self) -> tuple[float, ...]:
        return self._action_high

    def compose(
        self,
        *,
        mode: str,
        base_action: Iterable[float],
        intervention_action: Iterable[float] | None = None,
        ownership: ComponentOwnershipA2 | None = None,
    ) -> ComponentMixSelectionA2:
        mode = _required(mode, name="mode")
        if mode not in A2_MIX_MODES:
            raise ValueError(f"Unsupported A2 mix mode {mode!r}.")
        base = _vector(base_action, name="base_action", dimension=A2_ACTION_DIMENSION)
        intervention = (
            (0.0,) * A2_ACTION_DIMENSION
            if intervention_action is None
            else _vector(
                intervention_action,
                name="intervention_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        if ownership is None:
            ownership = ComponentOwnershipA2(
                arm_owner=(
                    A2_COMPONENT_OWNER_BASE
                    if mode == ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0
                    else A2_COMPONENT_OWNER_INTERVENTION
                ),
                gripper_owner=(
                    A2_COMPONENT_OWNER_BASE
                    if mode == ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0
                    else A2_COMPONENT_OWNER_INTERVENTION
                ),
            )
        if mode == ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0:
            if ownership.active_component_ids:
                raise ValueError("base_only cannot carry intervention ownership.")
            if any(value != 0.0 for value in intervention):
                raise ValueError("base_only cannot carry a nonzero intervention.")
            preclip = base
        else:
            if not ownership.active_component_ids:
                raise ValueError("overlay modes require an intervention owner.")
            preclip_values = list(base)
            if ownership.arm_owner == A2_COMPONENT_OWNER_INTERVENTION:
                if mode == ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0:
                    preclip_values[:6] = [
                        base[index] + intervention[index] for index in A2_ARM_INDICES
                    ]
                else:
                    preclip_values[:6] = [intervention[index] for index in A2_ARM_INDICES]
            if ownership.gripper_owner == A2_COMPONENT_OWNER_INTERVENTION:
                # Explicitly replace, including for relative overlays.
                preclip_values[A2_GRIPPER_INDEX] = intervention[A2_GRIPPER_INDEX]
            preclip = tuple(preclip_values)
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
        return ComponentMixSelectionA2(
            mode=mode,
            base_action=base,
            intervention_action=intervention,
            ownership=ownership,
            preclip_action=preclip,
            applied_action=applied,
            clipped_dimensions=clipped,
        )


@dataclass(frozen=True)
class StreamingInterventionControlFrameA2:
    """One mock-control input frame before policy inference and composition."""

    frame_id: str
    mode: str
    base_action: tuple[float, ...] | None = None
    human_raw_action: tuple[float, ...] | None = None
    human_mapped_action: tuple[float, ...] | None = None
    human_component_ids: tuple[str, ...] = ()
    human_event_id: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = _required(self.frame_id, name="frame_id")
        mode = _required(self.mode, name="mode")
        if mode not in A2_MIX_MODES:
            raise ValueError(f"Unsupported A2 frame mode {mode!r}.")
        base_action = (
            None
            if self.base_action is None
            else _vector(
                self.base_action,
                name="base_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        human_raw_action = (
            None
            if self.human_raw_action is None
            else _vector(
                self.human_raw_action,
                name="human_raw_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        human_mapped_action = (
            None
            if self.human_mapped_action is None
            else _vector(
                self.human_mapped_action,
                name="human_mapped_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        if (human_raw_action is None) != (human_mapped_action is None):
            raise ValueError("human_raw_action and human_mapped_action are paired.")
        human_event_id = (
            None
            if self.human_event_id is None
            else _required(self.human_event_id, name="human_event_id")
        )
        if human_mapped_action is None and human_event_id is not None:
            raise ValueError("human_event_id requires a human action.")
        human_component_ids = _components(
            self.human_component_ids,
            name="human_component_ids",
            allow_empty=True,
        )
        if human_mapped_action is not None and not human_component_ids:
            human_component_ids = A2_COMPONENT_IDS
        if human_mapped_action is None and human_component_ids:
            raise ValueError("human_component_ids require a human action.")
        payload = {
            "frame_id": frame_id,
            "mode": mode,
            "base_action": base_action,
            "human_raw_action": human_raw_action,
            "human_mapped_action": human_mapped_action,
            "human_component_ids": human_component_ids,
            "human_event_id": human_event_id,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_FRAME_HASH_DOMAIN),
        )

    @property
    def has_human_intervention(self) -> bool:
        return self.human_mapped_action is not None


@dataclass(frozen=True)
class StreamingInterventionEvidenceA2:
    """Lineage-bearing experiment evidence for one executed action.

    ``a_safe`` means the experiment-local final action envelope.  It is not a
    production safety acceptance or hardware-readiness claim.
    """

    frame_id: str
    frame_hash: str
    mode: str
    policy_proposal: PolicyProposalA2 | None
    observation_hash: str
    human_raw_action: tuple[float, ...] | None
    human_mapped_action: tuple[float, ...] | None
    human_component_ids: tuple[str, ...]
    human_event_id: str | None
    selection_hash: str
    ownership_hash: str
    active_component_ids: tuple[str, ...]
    a_policy: tuple[float, ...] | None
    a_rtc: tuple[float, ...] | None
    a_mix: tuple[float, ...]
    a_safe: tuple[float, ...]
    a_exec: tuple[float, ...]
    backend_step_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = _required(self.frame_id, name="frame_id")
        frame_hash = _required(self.frame_hash, name="frame_hash")
        mode = _required(self.mode, name="mode")
        if mode not in A2_MIX_MODES:
            raise ValueError(f"Unsupported A2 evidence mode {mode!r}.")
        policy_proposal = self.policy_proposal
        if policy_proposal is not None and not isinstance(
            policy_proposal,
            PolicyProposalA2,
        ):
            raise TypeError("policy_proposal must be a PolicyProposalA2.")
        observation_hash = _required(self.observation_hash, name="observation_hash")
        if policy_proposal is not None and (
            policy_proposal.observation_hash != observation_hash
        ):
            raise ValueError("policy proposal does not bind evidence observation.")
        human_raw_action = (
            None
            if self.human_raw_action is None
            else _vector(
                self.human_raw_action,
                name="human_raw_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        human_mapped_action = (
            None
            if self.human_mapped_action is None
            else _vector(
                self.human_mapped_action,
                name="human_mapped_action",
                dimension=A2_ACTION_DIMENSION,
            )
        )
        if (human_raw_action is None) != (human_mapped_action is None):
            raise ValueError("human actions must be paired in evidence.")
        human_component_ids = _components(
            self.human_component_ids,
            name="human_component_ids",
            allow_empty=True,
        )
        human_event_id = (
            None
            if self.human_event_id is None
            else _required(self.human_event_id, name="human_event_id")
        )
        if human_mapped_action is None and (human_component_ids or human_event_id):
            raise ValueError("human lineage requires mapped action, components, and event.")
        if human_mapped_action is not None and (
            not human_component_ids or human_event_id is None
        ):
            raise ValueError("human lineage is incomplete.")
        selection_hash = _required(self.selection_hash, name="selection_hash")
        ownership_hash = _required(self.ownership_hash, name="ownership_hash")
        active_component_ids = _components(
            self.active_component_ids,
            name="active_component_ids",
            allow_empty=True,
        )
        a_policy = (
            None
            if self.a_policy is None
            else _vector(self.a_policy, name="a_policy", dimension=A2_ACTION_DIMENSION)
        )
        a_rtc = (
            None
            if self.a_rtc is None
            else _vector(self.a_rtc, name="a_rtc", dimension=A2_ACTION_DIMENSION)
        )
        if policy_proposal is None:
            if a_policy is not None or a_rtc is not None:
                raise ValueError("policy action lineage requires a proposal.")
        elif policy_proposal.source_kind == A2_POLICY_SOURCE_KIND_STANDARD:
            if a_policy != policy_proposal.action or a_rtc is not None:
                raise ValueError("standard policy lineage does not match proposal.")
        elif a_rtc != policy_proposal.action or a_policy is not None:
            raise ValueError("RTC policy lineage does not match proposal.")
        a_mix = _vector(self.a_mix, name="a_mix", dimension=A2_ACTION_DIMENSION)
        a_safe = _vector(self.a_safe, name="a_safe", dimension=A2_ACTION_DIMENSION)
        a_exec = _vector(self.a_exec, name="a_exec", dimension=A2_ACTION_DIMENSION)
        if a_safe != a_exec:
            raise ValueError("A2 mock evidence requires a_exec to equal a_safe.")
        backend_step_hash = _required(self.backend_step_hash, name="backend_step_hash")
        payload = {
            "frame_id": frame_id,
            "frame_hash": frame_hash,
            "mode": mode,
            "policy_proposal_hash": (
                None if policy_proposal is None else policy_proposal.semantic_content_hash
            ),
            "observation_hash": observation_hash,
            "human_raw_action": human_raw_action,
            "human_mapped_action": human_mapped_action,
            "human_component_ids": human_component_ids,
            "human_event_id": human_event_id,
            "selection_hash": selection_hash,
            "ownership_hash": ownership_hash,
            "active_component_ids": active_component_ids,
            "a_policy": a_policy,
            "a_rtc": a_rtc,
            "a_mix": a_mix,
            "a_safe": a_safe,
            "a_exec": a_exec,
            "backend_step_hash": backend_step_hash,
        }
        for name, value in payload.items():
            if name != "policy_proposal_hash":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "policy_proposal", policy_proposal)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_EVIDENCE_HASH_DOMAIN),
        )

    @property
    def policy_source_id(self) -> str | None:
        return None if self.policy_proposal is None else self.policy_proposal.source_id

    @property
    def policy_source_revision(self) -> int | None:
        return None if self.policy_proposal is None else self.policy_proposal.revision

    @property
    def policy_inference_id(self) -> str | None:
        return None if self.policy_proposal is None else self.policy_proposal.inference_id


__all__ = [
    "A2_ACTION_DIMENSION",
    "A2_ARM_COMPONENT",
    "A2_COMPONENT_IDS",
    "A2_COMPONENT_OWNER_BASE",
    "A2_COMPONENT_OWNER_INTERVENTION",
    "A2_GRIPPER_COMPONENT",
    "A2_MIX_MODES",
    "A2_POLICY_ACTION_ROLE_INTERVENTION",
    "A2_POLICY_ACTION_ROLE_NOMINAL",
    "A2_POLICY_SOURCE_KIND_RTC",
    "A2_POLICY_SOURCE_KIND_STANDARD",
    "ComponentMixSelectionA2",
    "ComponentOwnershipA2",
    "PolicyInferenceRequestA2",
    "PolicyInferenceSourceA2",
    "PolicyProposalA2",
    "RobosuiteComponentComposerA2",
    "SequencePolicySourceA2",
    "StreamingInterventionControlFrameA2",
    "StreamingInterventionEvidenceA2",
]
