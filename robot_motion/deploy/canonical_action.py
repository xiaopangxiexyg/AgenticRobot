from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from robot_motion.action import (
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    RobotAction,
    RobotActionResolver,
    SharedCanonicalSafetyResult,
    SourceValidationResult,
    VectorActionLayout,
    apply_canonical_keyframe_safety,
)
from robot_motion.runtime import TeleopCommand

from .actions import ActionSafetyFilter, FilteredNormalizedAction, NormalizedAction
from .action_adapter import (
    DEPLOY_VECTOR_ACTION_LAYOUT,
    DeployActionRepresentationReceipt,
    DeployRobotActionAdapter,
    DeployRobotActionAdapterResult,
)
from .observation import DeploymentObservation
from .canonical_policy_action import CanonicalPolicyAction


@dataclass(frozen=True)
class CanonicalDeployActionResult:
    filtered: FilteredNormalizedAction
    robot_action: RobotAction | None = None
    command: TeleopCommand | None = None
    faults: tuple[str, ...] = ()
    source_validation: SourceValidationResult | None = None
    representation_receipt: DeployActionRepresentationReceipt | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> NormalizedAction:
        return self.filtered.action

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


class CanonicalDeployActionPipeline:
    """Prepare deploy actions through the canonical RobotAction boundary."""

    def __init__(
        self,
        *,
        safety_filter: ActionSafetyFilter,
        active_sides: Sequence[str],
        arms_enabled: bool,
        hands_enabled: bool,
        resolver: RobotActionResolver | None = None,
        layout: VectorActionLayout = DEPLOY_VECTOR_ACTION_LAYOUT,
    ):
        self.safety_filter = safety_filter
        self.active_sides = tuple(str(side) for side in active_sides)
        self.arms_enabled = bool(arms_enabled)
        self.hands_enabled = bool(hands_enabled)
        self.resolver = resolver if resolver is not None else RobotActionResolver()
        self.layout = layout
        self.action_adapter = DeployRobotActionAdapter(
            active_sides=self.active_sides,
            arms_enabled=self.arms_enabled,
            hands_enabled=self.hands_enabled,
            layout=self.layout,
        )

    def adapt_pre_safety(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> DeployRobotActionAdapterResult:
        """Expose the decoded absolute intent without applying policy safety."""

        return self.action_adapter.adapt(
            action,
            observation,
            policy_name=policy_name,
            input_stage="decoded_absolute_pre_safety",
        )

    def prepare(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> CanonicalDeployActionResult:
        filtered = self.safety_filter.filter_normalized(
            action,
            observation,
            policy_name=policy_name,
        )
        return self.resolve_filtered(
            filtered,
            observation,
            policy_name=policy_name,
        )

    def resolve_filtered(
        self,
        filtered: FilteredNormalizedAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> CanonicalDeployActionResult:
        """Resolve an action that has already passed the keyframe safety gate.

        Position deployment deliberately separates the 15 Hz keyframe safety
        pass from the high-rate reference generator.  Calling ``prepare`` for every
        interpolated reference would incorrectly re-apply the 15 Hz step gate
        three times and would turn a safety limiter into hidden temporal
        smoothing.  This method preserves the original safety result while
        still routing every high-rate reference through the canonical
        ``RobotAction``/``TeleopCommand`` boundary.
        """

        if not isinstance(filtered, FilteredNormalizedAction):
            raise TypeError("filtered must be a FilteredNormalizedAction")
        adapted = self.action_adapter.adapt(
            filtered.action,
            observation,
            policy_name=policy_name,
            input_stage="legacy_policy_safety_output",
        )
        faults = [*filtered.faults, *adapted.faults]
        command = None
        compatibility_robot_action = adapted.compatibility_robot_action
        if compatibility_robot_action is not None and not faults:
            command = self.resolver.resolve(compatibility_robot_action)
        if command is not None:
            _annotate_canonical_command(
                command,
                policy_name=policy_name,
                safety_metadata=filtered.metadata,
            )
        return CanonicalDeployActionResult(
            filtered=filtered,
            robot_action=compatibility_robot_action,
            command=command,
            faults=tuple(faults),
            source_validation=adapted.source_validation,
            representation_receipt=adapted.representation_receipt,
            metadata={
                "canonical_action_path": True,
                "command_blocked_by_faults": bool(faults),
                "vector_build": dict(adapted.vector_build_metadata),
                "safety": dict(filtered.metadata),
                "representation_trace_id": (
                    adapted.representation_receipt.representation_trace_id
                ),
            },
        )


@dataclass(frozen=True)
class CanonicalPolicyActionResult:
    """Typed canonical preparation result without a legacy vector round-trip."""

    canonical_action: CanonicalPolicyAction
    safety_result: SharedCanonicalSafetyResult
    robot_action: RobotAction | None = None
    command: TeleopCommand | None = None
    faults: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(value) for value in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


class CanonicalPolicyActionPipeline:
    """Prepare one typed policy action through shared safety and resolution.

    The caller-provided anchor provider remains the safety-anchor owner.  This
    class performs no observation reads beyond the immutable observation value
    passed by the control thread and owns no backend or dispatch authority.
    """

    def __init__(
        self,
        *,
        safety_profile: CanonicalKeyframeSafetyProfile,
        anchor_provider: Callable[
            [CanonicalPolicyAction, DeploymentObservation, float],
            CanonicalActionSafetyAnchor,
        ],
        resolver: RobotActionResolver | None = None,
    ) -> None:
        if not isinstance(safety_profile, CanonicalKeyframeSafetyProfile):
            raise TypeError(
                "safety_profile must be a CanonicalKeyframeSafetyProfile."
            )
        if not callable(anchor_provider):
            raise TypeError("anchor_provider must be callable.")
        self.safety_profile = safety_profile
        self.anchor_provider = anchor_provider
        self.resolver = resolver if resolver is not None else RobotActionResolver()

    def prepare_canonical(
        self,
        action: CanonicalPolicyAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
        input_action_id: str,
        target_effective_at: float,
    ) -> CanonicalPolicyActionResult:
        if not isinstance(action, CanonicalPolicyAction):
            raise TypeError("action must be a CanonicalPolicyAction.")
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        target_at = float(target_effective_at)
        anchor = self.anchor_provider(action, observation, target_at)
        if not isinstance(anchor, CanonicalActionSafetyAnchor):
            raise TypeError(
                "anchor_provider must return CanonicalActionSafetyAnchor."
            )
        safety = apply_canonical_keyframe_safety(
            action.action,
            str(input_action_id),
            anchor=anchor,
            profile=self.safety_profile,
            target_effective_at=target_at,
        )
        safe_action = safety.safe_action
        command = None
        faults: tuple[str, ...] = ()
        if safety.status in {"fault", "hold"} or safe_action is None:
            faults = tuple(safety.reasons) or (
                f"canonical_policy_safety_{safety.status}",
            )
        else:
            command = self.resolver.resolve(safe_action)
            _annotate_canonical_command(
                command,
                policy_name=policy_name,
                safety_metadata={
                    "status": safety.status,
                    "reasons": safety.reasons,
                    "modified_components": safety.modified_components,
                    "safety_contract_id": safety.safety_contract_id,
                    "safety_contract_hash": safety.safety_contract_hash,
                    "anchor_id": safety.anchor_id,
                    "anchor_kind": safety.anchor_kind,
                    "anchor_hash": safety.anchor_hash,
                },
            )
        return CanonicalPolicyActionResult(
            canonical_action=action,
            safety_result=safety,
            robot_action=safe_action,
            command=command,
            faults=faults,
            metadata={
                "canonical_action_path": True,
                "typed_policy_action": True,
                "command_blocked_by_faults": bool(faults),
                "canonicalization_receipt_id": (
                    action.canonicalization_receipt.receipt_id
                ),
                "safety_status": safety.status,
            },
        )


def _annotate_canonical_command(
    command: TeleopCommand,
    *,
    policy_name: str,
    safety_metadata: Mapping[str, Any],
) -> None:
    command.metadata["source"] = "deploy_canonical_robot_action"
    command.metadata["policy_name"] = str(policy_name)
    command.metadata["deploy_safety"] = dict(safety_metadata)
    for effector_command in command.commands.values():
        effector_command.metadata["deploy_source"] = "deploy_canonical_robot_action"
        effector_command.metadata["policy_name"] = str(policy_name)
        effector_command.metadata["deploy_safety"] = dict(safety_metadata)


__all__ = [
    "CanonicalDeployActionPipeline",
    "CanonicalDeployActionResult",
    "CanonicalPolicyActionPipeline",
    "CanonicalPolicyActionResult",
    "DEPLOY_VECTOR_ACTION_LAYOUT",
]
