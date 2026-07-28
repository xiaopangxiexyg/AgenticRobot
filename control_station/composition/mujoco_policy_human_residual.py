"""Default-off MuJoCo profile: policy baseline plus human residual.

This module is a concrete Control Station binding.  The reusable composition
contracts remain source-neutral; only this profile chooses ``policy`` as the
baseline role and ``human_residual`` as the residual role.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from control_composition import (
    CompositionActivationScope,
    ResidualAnchorRequest,
)
from control_composition.composition import (
    CompositionAuthorityGateSpec,
    RelativeBaselineResidualContracts,
    build_relative_baseline_residual_contracts,
)
from robot_motion.action import (
    ComponentAction,
    ComponentActionSchema,
    ConservativeCompoundAuthorityProfile,
    build_relative_delta_action_schema,
)
from robot_motion.coordinate_frames import CoordinateFrameGraph
from teleop_runtime.action import (
    HumanComponentResidualAdapter,
    HumanComponentResidualAnchor,
    HumanComponentResidualResult,
)

from .mujoco_sim_hil import MujocoSimHILContracts


MUJOCO_POLICY_HUMAN_RESIDUAL_PROFILE_ID_DEVELOPMENT = (
    "mujoco.policy-baseline.human-residual.development"
)
MUJOCO_POLICY_HUMAN_RESIDUAL_SOURCE_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.mujoco_policy_human_residual_source_binding.development"
)
MUJOCO_POLICY_HUMAN_RESIDUAL_STEP_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.mujoco_policy_human_residual_step.development"
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


@dataclass(frozen=True)
class MujocoPolicyHumanResidualProfileConfig:
    """Concrete profile selection; disabled unless explicitly enabled."""

    enabled: bool = False
    profile_id: str = MUJOCO_POLICY_HUMAN_RESIDUAL_PROFILE_ID_DEVELOPMENT
    baseline_role: str = "policy"
    residual_role: str = "human_residual"
    require_residual_role: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool.")
        if not isinstance(self.require_residual_role, bool):
            raise TypeError("require_residual_role must be bool.")
        for name in ("profile_id", "baseline_role", "residual_role"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.baseline_role == self.residual_role:
            raise ValueError("baseline_role and residual_role must differ.")


@dataclass(frozen=True)
class MujocoPolicyHumanResidualProfileStep:
    status: str
    source_action_id: str
    source_action_hash: str
    anchor_request_id: str
    anchor_request_hash: str
    anchor_receipt_id: str | None
    anchor_receipt_hash: str | None
    relative_result: HumanComponentResidualResult | None
    occurred_at: float
    clock_domain_id: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = _required(self.status, name="status")
        if status not in {"anchor_captured", "relative_action_ready"}:
            raise ValueError("Unsupported residual profile step status.")
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "source_action_id",
                "source_action_hash",
                "anchor_request_id",
                "anchor_request_hash",
                "clock_domain_id",
            )
        }
        if (self.anchor_receipt_id is None) != (
            self.anchor_receipt_hash is None
        ):
            raise ValueError(
                "anchor_receipt_id/hash must be both present or absent."
            )
        if self.anchor_receipt_id is None:
            raise ValueError("A profile step requires anchor receipt evidence.")
        receipt_id = _required(
            self.anchor_receipt_id,
            name="anchor_receipt_id",
        )
        receipt_hash = _required(
            self.anchor_receipt_hash,
            name="anchor_receipt_hash",
        )
        if status == "anchor_captured" and self.relative_result is not None:
            raise ValueError(
                "Anchor capture must not also publish a relative action."
            )
        if status == "relative_action_ready" and not isinstance(
            self.relative_result,
            HumanComponentResidualResult,
        ):
            raise ValueError(
                "Ready residual step requires a typed relative result."
            )
        occurred_at = _finite(self.occurred_at, name="occurred_at")
        payload = {
            "status": status,
            **text,
            "anchor_receipt_id": receipt_id,
            "anchor_receipt_hash": receipt_hash,
            "relative_result_hash": (
                None
                if self.relative_result is None
                else self.relative_result.receipt.semantic_content_hash
            ),
            "occurred_at": occurred_at,
        }
        object.__setattr__(self, "status", status)
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "anchor_receipt_id", receipt_id)
        object.__setattr__(self, "anchor_receipt_hash", receipt_hash)
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    MUJOCO_POLICY_HUMAN_RESIDUAL_STEP_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def relative_action(self) -> ComponentAction | None:
        return (
            None
            if self.relative_result is None
            else self.relative_result.relative_action
        )

    @property
    def output_action(self) -> ComponentAction | None:
        """Generic Station transform-port view of this profile-local step."""

        return self.relative_action


class MujocoPolicyHumanResidualInputAdapter:
    """Profile-local activation/recenter state around the reusable adapter."""

    def __init__(
        self,
        *,
        config: MujocoPolicyHumanResidualProfileConfig,
        adapter: HumanComponentResidualAdapter,
    ) -> None:
        if not isinstance(config, MujocoPolicyHumanResidualProfileConfig):
            raise TypeError(
                "config must be MujocoPolicyHumanResidualProfileConfig."
            )
        if not isinstance(adapter, HumanComponentResidualAdapter):
            raise TypeError(
                "adapter must be HumanComponentResidualAdapter."
            )
        self.config = config
        self.adapter = adapter
        self._lock = threading.RLock()
        self._request: ResidualAnchorRequest | None = None
        self._anchor: HumanComponentResidualAnchor | None = None

    @property
    def anchor_request(self) -> ResidualAnchorRequest | None:
        with self._lock:
            return self._request

    @property
    def anchor(self) -> HumanComponentResidualAnchor | None:
        with self._lock:
            return self._anchor

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._anchor is not None

    @property
    def input_action_schema(self) -> ComponentActionSchema:
        return self.adapter.absolute_action_schema

    @property
    def output_action_schema(self) -> ComponentActionSchema:
        return self.adapter.relative_action_schema

    def activate(
        self,
        activation_scope: CompositionActivationScope,
        *,
        activated_at: float,
    ) -> ResidualAnchorRequest:
        """Implement the generic Station input-transform activation port."""

        return self.request_anchor(
            activation_scope,
            requested_at=activated_at,
        )

    def request_anchor(
        self,
        activation_scope: CompositionActivationScope,
        *,
        requested_at: float,
    ) -> ResidualAnchorRequest:
        """Start or recenter one generation; no source sample is captured here."""

        if not self.config.enabled:
            raise RuntimeError(
                "policy-baseline/human-residual profile is default-off."
            )
        with self._lock:
            previous = self._request
            generation = (
                0
                if previous is None
                else previous.request_generation + 1
            )
            request = ResidualAnchorRequest.for_activation(
                activation_scope,
                request_generation=generation,
                source_role=self.config.residual_role,
                component_ids=self.adapter.component_ids,
                source_binding_id=self.adapter.source_binding_id,
                source_binding_hash=self.adapter.source_binding_hash,
                absolute_action_schema_hash=(
                    self.adapter.absolute_action_schema.semantic_content_hash
                ),
                relative_action_schema_hash=(
                    self.adapter.relative_action_schema.semantic_content_hash
                ),
                requested_at=requested_at,
                previous_request=previous,
            )
            self._request = request
            self._anchor = None
            return request

    def process_absolute_action(
        self,
        absolute_action: ComponentAction,
        *,
        source_action_id: str,
        source_action_hash: str,
        source_available_at: float,
        measured_at: float,
        measured_clock_domain_id: str,
        observed_at: float,
        observed_clock_domain_id: str,
    ) -> MujocoPolicyHumanResidualProfileStep:
        """Capture the pending anchor or emit one later relative action."""

        with self._lock:
            if self._request is None:
                raise RuntimeError(
                    "Residual profile requires request_anchor() before input."
                )
            request = self._request
            observed = _finite(observed_at, name="observed_at")
            observed_clock = _required(
                observed_clock_domain_id,
                name="observed_clock_domain_id",
            )
            if observed_clock != request.clock_domain_id:
                raise ValueError(
                    "Residual profile observation uses another receiver clock."
                )
            if self._anchor is None:
                anchor = self.adapter.capture_anchor(
                    absolute_action,
                    request=request,
                    source_action_id=source_action_id,
                    source_action_hash=source_action_hash,
                    source_available_at=source_available_at,
                    measured_at=measured_at,
                    measured_clock_domain_id=measured_clock_domain_id,
                    captured_at=observed,
                )
                self._anchor = anchor
                return MujocoPolicyHumanResidualProfileStep(
                    status="anchor_captured",
                    source_action_id=source_action_id,
                    source_action_hash=source_action_hash,
                    anchor_request_id=request.request_id,
                    anchor_request_hash=request.semantic_content_hash,
                    anchor_receipt_id=anchor.receipt.receipt_id,
                    anchor_receipt_hash=(
                        anchor.receipt.semantic_content_hash
                    ),
                    relative_result=None,
                    occurred_at=observed,
                    clock_domain_id=observed_clock,
                )
            result = self.adapter.adapt(
                absolute_action,
                anchor=self._anchor,
                source_action_id=source_action_id,
                source_action_hash=source_action_hash,
                measured_at=measured_at,
                measured_clock_domain_id=measured_clock_domain_id,
                adapted_at=observed,
                adapted_clock_domain_id=observed_clock,
            )
            return MujocoPolicyHumanResidualProfileStep(
                status="relative_action_ready",
                source_action_id=source_action_id,
                source_action_hash=source_action_hash,
                anchor_request_id=request.request_id,
                anchor_request_hash=request.semantic_content_hash,
                anchor_receipt_id=self._anchor.receipt.receipt_id,
                anchor_receipt_hash=(
                    self._anchor.receipt.semantic_content_hash
                ),
                relative_result=result,
                occurred_at=observed,
                clock_domain_id=observed_clock,
            )

    def transform(
        self,
        action: ComponentAction,
        *,
        source_action_id: str,
        source_action_hash: str,
        source_available_at: float,
        measured_at: float,
        measured_clock_domain_id: str,
        observed_at: float,
        observed_clock_domain_id: str,
    ) -> MujocoPolicyHumanResidualProfileStep:
        """Implement the generic Station input-transform data port."""

        return self.process_absolute_action(
            action,
            source_action_id=source_action_id,
            source_action_hash=source_action_hash,
            source_available_at=source_available_at,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock_domain_id,
            observed_at=observed_at,
            observed_clock_domain_id=observed_clock_domain_id,
        )


@dataclass(frozen=True)
class MujocoPolicyHumanResidualProfileBundle:
    config: MujocoPolicyHumanResidualProfileConfig
    relative_contracts: RelativeBaselineResidualContracts
    sim_hil_contracts: MujocoSimHILContracts
    residual_adapter: HumanComponentResidualAdapter = field(
        repr=False,
        compare=False,
    )
    input_adapter: MujocoPolicyHumanResidualInputAdapter = field(
        repr=False,
        compare=False,
    )

    @property
    def input_action_schema_by_role(
        self,
    ) -> Mapping[str, ComponentActionSchema]:
        """Exact role schemas required when installing this profile runtime."""

        return MappingProxyType(
            {
                self.config.baseline_role: (
                    self.relative_contracts.output_action_schema
                ),
                self.config.residual_role: (
                    self.relative_contracts.residual_action_schema
                ),
            }
        )


def build_mujoco_policy_human_residual_source_binding_hash(
    *,
    passive_source_binding_hash: str,
    absolute_source_adapter_hash: str,
    calibration_binding: Mapping[str, Any],
    absolute_action_schema_hash: str,
) -> str:
    """Bind the source observer, canonicalizer, calibration/Home, and schema."""

    return canonical_content_hash(
        {
            "passive_source_binding_hash": _required(
                passive_source_binding_hash,
                name="passive_source_binding_hash",
            ),
            "absolute_source_adapter_hash": _required(
                absolute_source_adapter_hash,
                name="absolute_source_adapter_hash",
            ),
            "calibration_binding": dict(calibration_binding),
            "absolute_action_schema_hash": _required(
                absolute_action_schema_hash,
                name="absolute_action_schema_hash",
            ),
        },
        domain=(
            MUJOCO_POLICY_HUMAN_RESIDUAL_SOURCE_BINDING_HASH_DOMAIN_DEVELOPMENT
        ),
    )


def build_mujoco_policy_human_residual_profile(
    *,
    absolute_action_schema: ComponentActionSchema,
    frame_graph: CoordinateFrameGraph,
    concrete_reference_frame_by_component: Mapping[str, str],
    concrete_controlled_frame_by_component: Mapping[str, str],
    source_binding_id: str,
    source_binding_hash: str,
    composer_producer_id: str,
    proposal_ttl_sec: float,
    config: MujocoPolicyHumanResidualProfileConfig | None = None,
    pose_translation_gain_by_component: Mapping[str, object] | None = None,
    joint_gain_by_component: Mapping[str, object] | None = None,
) -> MujocoPolicyHumanResidualProfileBundle:
    """Build the concrete profile without enabling or starting any runtime."""

    profile = config or MujocoPolicyHumanResidualProfileConfig()
    relative_schema = build_relative_delta_action_schema(
        absolute_action_schema,
        schema_id=f"{profile.profile_id}.human-residual-action",
        schema_version="development",
        catalog_id=f"{profile.profile_id}.human-residual-components",
    )
    required_roles = (
        (profile.baseline_role, profile.residual_role)
        if profile.require_residual_role
        else (profile.baseline_role,)
    )
    relative_contracts = build_relative_baseline_residual_contracts(
        contract_id=profile.profile_id,
        output_action_schema=absolute_action_schema,
        residual_action_schema=relative_schema,
        baseline_role=profile.baseline_role,
        residual_role=profile.residual_role,
        required_roles=required_roles,
    )
    component_coverage = tuple(absolute_action_schema.component_ids)
    selection_groups = tuple(
        absolute_action_schema.component_catalog.selection_group_ids
    )
    authority_profile = ConservativeCompoundAuthorityProfile(
        profile_id=f"{profile.profile_id}.compound-authority",
        producer_id=_required(
            composer_producer_id,
            name="composer_producer_id",
        ),
        component_coverage=component_coverage,
        selection_groups=selection_groups,
        composition_spec_hash=(
            relative_contracts.composition_spec.semantic_content_hash
        ),
        component_rule_set_hash=(
            relative_contracts.component_rule_set.semantic_content_hash
        ),
        formal_strategy_spec_hash=(
            relative_contracts.strategy_spec.semantic_content_hash
        ),
    )
    missing_behavior = {
        profile.baseline_role: "hold",
        profile.residual_role: (
            "hold" if profile.require_residual_role else "no_action"
        ),
    }
    gate = CompositionAuthorityGateSpec(
        gate_spec_id=f"{profile.profile_id}.authority-gate",
        composition_spec_hash=(
            relative_contracts.composition_spec.semantic_content_hash
        ),
        composer_producer_id=authority_profile.producer_id,
        missing_behavior_by_role=missing_behavior,
        proposal_ttl_sec=_finite(
            proposal_ttl_sec,
            name="proposal_ttl_sec",
        ),
        invalid_result_behavior="hold",
        composer_error_behavior="hold",
        all_inputs_missing_behavior="hold",
        allow_parallel_source_proposals=False,
        canonical_safety_required=True,
    )
    sim_contracts = MujocoSimHILContracts(
        composition_spec=relative_contracts.composition_spec,
        component_rule_set=relative_contracts.component_rule_set,
        strategy_spec=relative_contracts.strategy_spec,
        strategy=relative_contracts.strategy,
        authority_profile=authority_profile,
        authority_gate_spec=gate,
    )
    residual_adapter = HumanComponentResidualAdapter(
        absolute_action_schema=absolute_action_schema,
        relative_action_schema=relative_schema,
        source_binding_id=source_binding_id,
        source_binding_hash=source_binding_hash,
        frame_graph=frame_graph,
        concrete_reference_frame_by_component=(
            concrete_reference_frame_by_component
        ),
        concrete_controlled_frame_by_component=(
            concrete_controlled_frame_by_component
        ),
        pose_translation_gain_by_component=(
            pose_translation_gain_by_component
        ),
        joint_gain_by_component=joint_gain_by_component,
    )
    input_adapter = MujocoPolicyHumanResidualInputAdapter(
        config=profile,
        adapter=residual_adapter,
    )
    return MujocoPolicyHumanResidualProfileBundle(
        config=profile,
        relative_contracts=relative_contracts,
        sim_hil_contracts=sim_contracts,
        residual_adapter=residual_adapter,
        input_adapter=input_adapter,
    )


__all__ = [
    "MUJOCO_POLICY_HUMAN_RESIDUAL_PROFILE_ID_DEVELOPMENT",
    "MUJOCO_POLICY_HUMAN_RESIDUAL_SOURCE_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "MUJOCO_POLICY_HUMAN_RESIDUAL_STEP_HASH_DOMAIN_DEVELOPMENT",
    "MujocoPolicyHumanResidualInputAdapter",
    "MujocoPolicyHumanResidualProfileBundle",
    "MujocoPolicyHumanResidualProfileConfig",
    "MujocoPolicyHumanResidualProfileStep",
    "build_mujoco_policy_human_residual_profile",
    "build_mujoco_policy_human_residual_source_binding_hash",
]
