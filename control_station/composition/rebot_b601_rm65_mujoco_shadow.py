"""Default-off B601 residual composition on the MuJoCo RM65 shadow owner.

The caller owns B601 polling, anchor/recenter lifecycle, MuJoCo runtime
lifecycle, and producer activation.  This controller only freezes one fresh
target-measured pose baseline, admits the current B601 residual, and asks the
existing MuJoCo composition port for a non-authoritative shadow evaluation.

A rejected current B601 residual is a cycle-local withdrawal: this controller
does not evaluate the receiver again, so an older still-retained residual
cannot become this cycle's output.  A later continuously promotable profile
must additionally publish source eligibility to invalidate retained inputs for
other consumers; this shadow binding deliberately owns no such lifecycle.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from control_composition.activation import CompositionActivationScope
from control_composition.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionSpec,
    FormalCompositionStrategy,
    RelativeBaselineResidualContracts,
)
from control_composition.runtime import CompositeProposalSubmission
from control_station.composition.execution_measured_pose_baseline import (
    ExecutionMeasuredPoseBaselineCapture,
    ExecutionMeasuredPoseBaselineController,
    ExecutionMeasuredPoseBaselineProfile,
)
from control_station.composition.rebot_b601_rm65_motion_input import (
    B601RM65MotionInputAdaptation,
    B601RM65MotionInputProfile,
    adapt_b601_rm65_residual_to_composition_input,
)
from control_station.composition.rebot_b601_rm65_residual import (
    B601RM65ResidualStep,
)
from robot_motion.action import (
    ComponentAction,
    ComponentActionSchema,
    ExecutionMeasurementObservation,
)

B601_RM65_MUJOCO_SHADOW_PROFILE_OFF = "off"


class _RuntimeConfigPort(Protocol):
    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str


class _CompositionPort(Protocol):
    input_action_schema_by_role: Mapping[str, ComponentActionSchema]

    def admit_composition_input(
        self,
        draft: CompositionInputDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ) -> CompositionInputAdmissionReceipt: ...

    def evaluate_composition_shadow(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        activation_scope: CompositionActivationScope | None = None,
    ) -> CompositeProposalSubmission: ...


@runtime_checkable
class _MujocoShadowRuntimePort(Protocol):
    """The existing target-owned ports used by this shadow binding."""

    config: _RuntimeConfigPort
    action_schema: ComponentActionSchema
    composition_port: _CompositionPort | None
    composition_strategy: FormalCompositionStrategy | None


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class B601RM65MujocoShadowProfile:
    """Application switch for the non-authoritative motion-only shadow."""

    profile_id: str = B601_RM65_MUJOCO_SHADOW_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("B601/RM65 MuJoCo shadow cannot send actions.")
        if self.enabled and profile_id == B601_RM65_MUJOCO_SHADOW_PROFILE_OFF:
            raise ValueError("An enabled B601/RM65 shadow needs a profile id.")
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class B601RM65MujocoShadowActivation:
    """One disabled, rejected, or receiver-admitted measured baseline."""

    profile_id: str
    status: str
    activation_scope: CompositionActivationScope | None = None
    baseline_capture: ExecutionMeasuredPoseBaselineCapture | None = None
    baseline_admission: CompositionInputAdmissionReceipt | None = None

    def __post_init__(self) -> None:
        _required(self.profile_id, name="profile_id")
        if self.status not in {"disabled", "baseline_rejected", "activated"}:
            raise ValueError("Unsupported B601/RM65 shadow activation status.")
        facts = (
            self.activation_scope,
            self.baseline_capture,
            self.baseline_admission,
        )
        if self.status == "disabled":
            if any(value is not None for value in facts):
                raise ValueError("A disabled shadow activation has no facts.")
            return
        if not isinstance(self.activation_scope, CompositionActivationScope):
            raise TypeError("An enabled activation requires an activation scope.")
        if not isinstance(
            self.baseline_capture,
            ExecutionMeasuredPoseBaselineCapture,
        ):
            raise TypeError("An enabled activation requires a baseline capture.")
        if not isinstance(
            self.baseline_admission,
            CompositionInputAdmissionReceipt,
        ):
            raise TypeError("An enabled activation requires an admission receipt.")
        accepted = self.baseline_admission.accepted
        if (self.status == "activated") != accepted:
            raise ValueError("Activation status and baseline admission disagree.")

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False

    @property
    def command_dispatched(self) -> bool:
        return False


@dataclass(frozen=True)
class B601RM65MujocoShadowCycle:
    """One current-source decision and optional MuJoCo shadow evaluation."""

    profile_id: str
    status: str
    activation_scope: CompositionActivationScope | None = None
    residual_adaptation: B601RM65MotionInputAdaptation | None = None
    residual_admission: CompositionInputAdmissionReceipt | None = None
    submission: CompositeProposalSubmission | None = None

    def __post_init__(self) -> None:
        _required(self.profile_id, name="profile_id")
        if self.status not in {
            "disabled",
            "source_rejected",
            "input_rejected",
            "evaluated",
        }:
            raise ValueError("Unsupported B601/RM65 MuJoCo shadow status.")
        if self.status == "disabled":
            if any(
                value is not None
                for value in (
                    self.activation_scope,
                    self.residual_adaptation,
                    self.residual_admission,
                    self.submission,
                )
            ):
                raise ValueError("A disabled shadow cycle has no facts.")
            return
        if not isinstance(self.activation_scope, CompositionActivationScope):
            raise TypeError("An enabled cycle requires an activation scope.")
        if not isinstance(
            self.residual_adaptation,
            B601RM65MotionInputAdaptation,
        ):
            raise TypeError("An enabled cycle requires a residual adaptation.")
        if self.status == "source_rejected":
            if self.residual_adaptation.admitted:
                raise ValueError("A source-rejected cycle cannot contain a draft.")
            if self.residual_admission is not None or self.submission is not None:
                raise ValueError("A source-rejected cycle cannot reach the receiver.")
            return
        if not isinstance(
            self.residual_admission,
            CompositionInputAdmissionReceipt,
        ):
            raise TypeError("A receiver cycle requires an admission receipt.")
        if self.status == "input_rejected":
            if self.residual_admission.accepted or self.submission is not None:
                raise ValueError("An input-rejected cycle cannot be evaluated.")
            return
        if not self.residual_admission.accepted:
            raise ValueError("An evaluated cycle requires an accepted input.")
        if not isinstance(self.submission, CompositeProposalSubmission):
            raise TypeError("An evaluated cycle requires a typed submission.")
        if self.submission.proposal_admission is not None:
            raise ValueError("B601/RM65 shadow cannot admit a proposal.")

    @property
    def conditional_action(self) -> ComponentAction | None:
        if self.submission is None:
            return None
        return self.submission.composition_evaluation.composed_candidate

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False

    @property
    def command_dispatched(self) -> bool:
        return False


class B601RM65MujocoShadowController:
    """Bind current B601 motion to one already-started MuJoCo shadow port."""

    def __init__(
        self,
        *,
        profile: B601RM65MujocoShadowProfile,
        runtime: _MujocoShadowRuntimePort | None = None,
        contracts: RelativeBaselineResidualContracts | None = None,
        baseline_profile: ExecutionMeasuredPoseBaselineProfile | None = None,
        residual_profile: B601RM65MotionInputProfile | None = None,
    ) -> None:
        if not isinstance(profile, B601RM65MujocoShadowProfile):
            raise TypeError("profile must be B601RM65MujocoShadowProfile.")
        if profile.enabled:
            self._validate_enabled_bindings(
                runtime=runtime,
                contracts=contracts,
                baseline_profile=baseline_profile,
                residual_profile=residual_profile,
            )
        self.profile = profile
        self.runtime = runtime
        self.contracts = contracts
        self.baseline_profile = baseline_profile
        self.residual_profile = residual_profile
        self._lock = threading.RLock()
        self._active: B601RM65MujocoShadowActivation | None = None
        self._baseline_controller: (
            ExecutionMeasuredPoseBaselineController | None
        ) = (
            None
            if baseline_profile is None
            else ExecutionMeasuredPoseBaselineController(baseline_profile)
        )
        self._next_residual_sequence = 0

    @property
    def active(self) -> B601RM65MujocoShadowActivation | None:
        with self._lock:
            return self._active

    def activate(
        self,
        activation_scope: CompositionActivationScope,
        observation: ExecutionMeasurementObservation,
        *,
        captured_at: float,
        effective_until: float,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowActivation:
        """Freeze and admit one target-owned measured pose baseline."""

        if not self.profile.enabled:
            return B601RM65MujocoShadowActivation(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        assert self.runtime is not None
        assert self.contracts is not None
        assert self.baseline_profile is not None
        self._validate_activation_scope(activation_scope)
        with self._lock:
            # Once a new activation attempt begins, never evaluate the previous
            # activation again even if target admission rejects this baseline.
            self._active = None
            controller = self._baseline_controller
            assert controller is not None
            controller.activate(activation_scope)
            capture = controller.capture(
                observation,
                anchor_generation=0,
                captured_at=captured_at,
                effective_until=effective_until,
                context_ids=context_ids,
            )
            port = self.runtime.composition_port
            assert port is not None
            admission = port.admit_composition_input(
                capture.input_draft,
                activation_scope=activation_scope,
            )
            if not isinstance(admission, CompositionInputAdmissionReceipt):
                raise TypeError("MuJoCo returned an untyped baseline receipt.")
            status = "activated" if admission.accepted else "baseline_rejected"
            result = B601RM65MujocoShadowActivation(
                profile_id=self.profile.profile_id,
                status=status,
                activation_scope=activation_scope,
                baseline_capture=capture,
                baseline_admission=admission,
            )
            if admission.accepted:
                self._validate_admitted_input(
                    admission,
                    expected_role=self.contracts.baseline_role,
                    expected_action_hash=(
                        capture.input_draft.action.semantic_content_hash
                    ),
                    activation_scope=activation_scope,
                )
                self._active = result
            return result

    def evaluate(
        self,
        residual_step: B601RM65ResidualStep | None,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowCycle:
        """Evaluate exactly the current residual, never a cached fallback."""

        if not self.profile.enabled:
            return B601RM65MujocoShadowCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        if not isinstance(residual_step, B601RM65ResidualStep):
            raise TypeError("residual_step must be B601RM65ResidualStep.")
        target = _finite(target_effective_at, name="target_effective_at")
        with self._lock:
            active = self._active
            if active is None or active.status != "activated":
                raise RuntimeError("activate() is required before shadow evaluation.")
            assert active.activation_scope is not None
            assert active.baseline_admission is not None
            assert self.runtime is not None
            assert self.contracts is not None
            assert self.residual_profile is not None
            adaptation = adapt_b601_rm65_residual_to_composition_input(
                residual_step,
                profile=self.residual_profile,
                activation_scope=active.activation_scope,
                sequence=self._next_residual_sequence,
                effective_from=effective_from,
                effective_until=effective_until,
                context_ids=context_ids,
            )
            if adaptation.input_draft is None:
                # Deliberately do not call evaluate_composition_shadow here.
                return B601RM65MujocoShadowCycle(
                    profile_id=self.profile.profile_id,
                    status="source_rejected",
                    activation_scope=active.activation_scope,
                    residual_adaptation=adaptation,
                )
            try:
                port = self.runtime.composition_port
                assert port is not None
                admission = port.admit_composition_input(
                    adaptation.input_draft,
                    activation_scope=active.activation_scope,
                )
                if not isinstance(admission, CompositionInputAdmissionReceipt):
                    raise TypeError("MuJoCo returned an untyped residual receipt.")
            except Exception:
                self._active = None
                raise
            if not admission.accepted:
                return B601RM65MujocoShadowCycle(
                    profile_id=self.profile.profile_id,
                    status="input_rejected",
                    activation_scope=active.activation_scope,
                    residual_adaptation=adaptation,
                    residual_admission=admission,
                )
            try:
                self._validate_admitted_input(
                    admission,
                    expected_role=self.contracts.residual_role,
                    expected_action_hash=(
                        adaptation.input_draft.action.semantic_content_hash
                    ),
                    activation_scope=active.activation_scope,
                )
            except Exception:
                self._active = None
                raise
            self._next_residual_sequence += 1
            try:
                submission = self.runtime.composition_port.evaluate_composition_shadow(
                    spec=self.contracts.composition_spec,
                    target_effective_at=target,
                    activation_scope=active.activation_scope,
                )
                self._validate_exact_shadow_cut(
                    submission,
                    baseline_input_id=(
                        active.baseline_admission.composition_input_id
                    ),
                    residual_input_id=admission.composition_input_id,
                    activation_scope=active.activation_scope,
                )
            except Exception:
                # A target-side evaluation or identity fault must require a
                # fresh measured baseline before another cycle is attempted.
                self._active = None
                raise
            return B601RM65MujocoShadowCycle(
                profile_id=self.profile.profile_id,
                status="evaluated",
                activation_scope=active.activation_scope,
                residual_adaptation=adaptation,
                residual_admission=admission,
                submission=submission,
            )

    def _validate_activation_scope(
        self,
        activation_scope: CompositionActivationScope,
    ) -> None:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        assert self.runtime is not None
        assert self.contracts is not None
        required = self.contracts.composition_spec.required_roles
        if activation_scope.required_roles != required:
            raise ValueError("Activation roles differ from the shadow contract.")
        if activation_scope.clock_domain_id != self.runtime.config.clock_domain_id:
            raise ValueError("Activation and MuJoCo runtime clocks differ.")

    @staticmethod
    def _validate_admitted_input(
        admission: CompositionInputAdmissionReceipt,
        *,
        expected_role: str,
        expected_action_hash: str,
        activation_scope: CompositionActivationScope,
    ) -> None:
        record = admission.composition_input
        if (
            not admission.accepted
            or record is None
            or not admission.composition_input_id
            or admission.composition_input_id != record.composition_input_id
            or record.input_role != expected_role
            or record.action.semantic_content_hash != expected_action_hash
            or record.composition_activation_id != activation_scope.activation_id
            or record.composition_activation_hash
            != activation_scope.semantic_content_hash
        ):
            raise RuntimeError("MuJoCo admitted input differs from the submitted fact.")

    def _validate_exact_shadow_cut(
        self,
        submission: CompositeProposalSubmission,
        *,
        baseline_input_id: str | None,
        residual_input_id: str | None,
        activation_scope: CompositionActivationScope,
    ) -> None:
        if not isinstance(submission, CompositeProposalSubmission):
            raise TypeError("MuJoCo returned an untyped shadow submission.")
        if submission.proposal_admission is not None:
            raise RuntimeError("MuJoCo shadow unexpectedly admitted a proposal.")
        assert self.contracts is not None
        selected = submission.input_snapshot.selected_input_by_role
        expected_ids = {
            self.contracts.baseline_role: baseline_input_id,
            self.contracts.residual_role: residual_input_id,
        }
        if set(selected) != set(expected_ids) or any(
            selected[role].composition_input_id != expected_id
            for role, expected_id in expected_ids.items()
        ):
            raise RuntimeError(
                "MuJoCo shadow did not select this activation's exact current inputs."
            )
        if any(
            record.composition_activation_id != activation_scope.activation_id
            or record.composition_activation_hash
            != activation_scope.semantic_content_hash
            for record in selected.values()
        ):
            raise RuntimeError("MuJoCo shadow selected another activation.")
        candidate = submission.composition_evaluation.composed_candidate
        if candidate is not None and (
            candidate.action_schema.semantic_content_hash
            != self.contracts.output_action_schema.semantic_content_hash
        ):
            raise RuntimeError("MuJoCo shadow returned another action schema.")

    @staticmethod
    def _validate_enabled_bindings(
        *,
        runtime: _MujocoShadowRuntimePort | None,
        contracts: RelativeBaselineResidualContracts | None,
        baseline_profile: ExecutionMeasuredPoseBaselineProfile | None,
        residual_profile: B601RM65MotionInputProfile | None,
    ) -> None:
        if not isinstance(runtime, _MujocoShadowRuntimePort):
            raise TypeError("Enabled shadow requires the MuJoCo shadow runtime ports.")
        if not isinstance(contracts, RelativeBaselineResidualContracts):
            raise TypeError("Enabled shadow requires relative composition contracts.")
        if not isinstance(
            baseline_profile,
            ExecutionMeasuredPoseBaselineProfile,
        ):
            raise TypeError("Enabled shadow requires a measured baseline profile.")
        if not isinstance(residual_profile, B601RM65MotionInputProfile):
            raise TypeError("Enabled shadow requires a B601 motion input profile.")
        if not baseline_profile.enabled or not residual_profile.enabled:
            raise ValueError("Enabled shadow requires enabled input profiles.")
        if baseline_profile.producer_id == residual_profile.producer_id:
            raise ValueError("Baseline and residual require distinct producers.")
        config = runtime.config
        for candidate in (baseline_profile, residual_profile):
            if (
                candidate.runtime_session_id != config.runtime_session_id
                or candidate.control_domain_id != config.control_domain_id
            ):
                raise ValueError("Input profile and MuJoCo runtime identity differ.")
        if (
            baseline_profile.input_role != contracts.baseline_role
            or residual_profile.input_role != contracts.residual_role
        ):
            raise ValueError("Input roles differ from relative composition contracts.")
        if (
            baseline_profile.action_schema.semantic_content_hash
            != contracts.output_action_schema.semantic_content_hash
            or residual_profile.action_schema.semantic_content_hash
            != contracts.residual_action_schema.semantic_content_hash
            or runtime.action_schema.semantic_content_hash
            != contracts.output_action_schema.semantic_content_hash
            or baseline_profile.component_ids
            != contracts.output_action_schema.component_ids
        ):
            raise ValueError("Input or target action schema differs from contracts.")
        port = runtime.composition_port
        if port is None:
            raise ValueError("Enabled shadow requires MuJoCo composition runtime.")
        expected_schemas = {
            contracts.baseline_role: contracts.output_action_schema.semantic_content_hash,
            contracts.residual_role: contracts.residual_action_schema.semantic_content_hash,
        }
        actual_schemas = {
            role: schema.semantic_content_hash
            for role, schema in port.input_action_schema_by_role.items()
        }
        if actual_schemas != expected_schemas:
            raise ValueError("MuJoCo composition input schemas differ from contracts.")
        strategy = runtime.composition_strategy
        if (
            strategy is None
            or strategy.spec.semantic_content_hash
            != contracts.strategy_spec.semantic_content_hash
        ):
            raise ValueError("MuJoCo composition strategy differs from contracts.")


__all__ = [
    "B601_RM65_MUJOCO_SHADOW_PROFILE_OFF",
    "B601RM65MujocoShadowActivation",
    "B601RM65MujocoShadowController",
    "B601RM65MujocoShadowCycle",
    "B601RM65MujocoShadowProfile",
]
