"""Anchor-relative B601 pose input for a configured RM65 residual schema.

This application adapter consumes the already admitted, read-only B601 FK
emission and reuses :class:`HumanResidualPoseAdapter` for frame conversion,
anchor capture, and relative pose math.  It does not own clutch state, target
IK, safety limits, watchdog behavior, authority, or command dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from control_station.sources.rebot_b601_pose import (
    B601ReadOnlyPoseSource,
    B601ReadOnlyPoseStep,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentAction
from robot_motion.rebot_b601_rs import B601_RS_END_EFFECTOR_FRAME
from teleop_runtime.action.human_residual import (
    HumanPoseSample,
    HumanResidualPoseAdaptationResult,
    HumanResidualPoseAdapter,
    HumanResidualPoseAnchor,
)

B601_RM65_RESIDUAL_SOURCE_CONTRACT_ID_V1 = (
    "control_station_b601_rm65_anchor_relative_residual_v1"
)
B601_RM65_RESIDUAL_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.b601_rm65_residual_source.v1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class B601RM65PoseProjection:
    """One B601 source step projected into a frame-explicit pose sample."""

    source_step: B601ReadOnlyPoseStep
    pose_sample: HumanPoseSample | None
    source_emission_hash: str | None
    source_observation_sample_id: str | None
    rejection_reason: str | None
    command_dispatched: bool = False
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_step, B601ReadOnlyPoseStep):
            raise TypeError("source_step must be a B601ReadOnlyPoseStep.")
        admitted = self.pose_sample is not None
        if admitted == (self.rejection_reason is not None):
            raise ValueError(
                "A B601/RM65 projection must contain exactly one of a pose "
                "sample or rejection reason."
            )
        if admitted:
            if not isinstance(self.pose_sample, HumanPoseSample):
                raise TypeError("pose_sample must be HumanPoseSample.")
            _required(
                self.source_emission_hash,
                name="source_emission_hash",
            )
            _required(
                self.source_observation_sample_id,
                name="source_observation_sample_id",
            )
        else:
            if (
                self.source_emission_hash is not None
                or self.source_observation_sample_id is not None
            ):
                raise ValueError("Rejected projections cannot retain partial lineage.")
            _required(self.rejection_reason, name="rejection_reason")
        if self.command_dispatched or self.authoritative:
            raise ValueError(
                "B601/RM65 pose projections cannot dispatch or claim authority."
            )

    @property
    def admitted(self) -> bool:
        return self.pose_sample is not None


@dataclass(frozen=True)
class B601RM65ResidualStep:
    """One relative action candidate or an explicit current-tick rejection."""

    projection: B601RM65PoseProjection
    adaptation: HumanResidualPoseAdaptationResult | None
    rejection_reason: str | None
    command_dispatched: bool = False
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.projection, B601RM65PoseProjection):
            raise TypeError("projection must be B601RM65PoseProjection.")
        admitted = self.adaptation is not None
        if admitted == (self.rejection_reason is not None):
            raise ValueError(
                "A B601/RM65 residual step must contain exactly one of an "
                "adaptation or rejection reason."
            )
        if admitted:
            if not isinstance(
                self.adaptation,
                HumanResidualPoseAdaptationResult,
            ):
                raise TypeError(
                    "adaptation must be HumanResidualPoseAdaptationResult."
                )
            if self.projection.pose_sample is None:
                raise ValueError("A residual adaptation requires an admitted sample.")
            source_hash = self.adaptation.receipt.source_sample_hash_by_component.get(
                self.projection.pose_sample.component_id
            )
            if source_hash != self.projection.pose_sample.semantic_content_hash:
                raise ValueError(
                    "Residual adaptation does not bind the projected B601 sample."
                )
        else:
            _required(self.rejection_reason, name="rejection_reason")
        if self.command_dispatched or self.authoritative:
            raise ValueError(
                "B601/RM65 residual steps cannot dispatch or claim authority."
            )

    @property
    def residual_available(self) -> bool:
        return self.adaptation is not None

    @property
    def relative_action(self) -> ComponentAction | None:
        return (
            None
            if self.adaptation is None
            else self.adaptation.relative_action
        )


class B601RM65ResidualSource:
    """Project admitted B601 poses into explicit-anchor RM65 residuals."""

    def __init__(
        self,
        pose_source: B601ReadOnlyPoseSource,
        residual_adapter: HumanResidualPoseAdapter,
        *,
        component_id: str | None = None,
    ) -> None:
        if not isinstance(pose_source, B601ReadOnlyPoseSource):
            raise TypeError("pose_source must be B601ReadOnlyPoseSource.")
        if not isinstance(residual_adapter, HumanResidualPoseAdapter):
            raise TypeError("residual_adapter must be HumanResidualPoseAdapter.")
        component = _required(
            pose_source.spec.effector_name if component_id is None else component_id,
            name="component_id",
        )
        if residual_adapter.pose_components != (component,):
            raise ValueError(
                "B601/RM65 source requires an exact single-component residual "
                f"adapter for {component!r}."
            )
        self.pose_source = pose_source
        self.residual_adapter = residual_adapter
        self.component_id = component
        self.source_coordinate_binding_id = (
            f"{pose_source.spec.source_id}:fk-pose-binding:v1"
        )
        self.source_coordinate_binding_hash = canonical_content_hash(
            {
                "contract_id": B601_RM65_RESIDUAL_SOURCE_CONTRACT_ID_V1,
                "source_contract_hash": pose_source.spec.semantic_content_hash,
                "component_id": component,
                "source_reference_frame_id": pose_source.spec.reference_frame,
                "source_controlled_frame_id": B601_RS_END_EFFECTOR_FRAME,
            },
            domain=B601_RM65_RESIDUAL_SOURCE_HASH_DOMAIN_V1,
        )
        self.contract_hash = canonical_content_hash(
            {
                "contract_id": B601_RM65_RESIDUAL_SOURCE_CONTRACT_ID_V1,
                "source_coordinate_binding_hash": (
                    self.source_coordinate_binding_hash
                ),
                "residual_adapter_contract_hash": residual_adapter.contract_hash,
            },
            domain=B601_RM65_RESIDUAL_SOURCE_HASH_DOMAIN_V1,
        )

    def poll_pose(self) -> B601RM65PoseProjection:
        """Poll once; rejection never falls back to an older admitted sample."""

        source_step = self.pose_source.step()
        if source_step.emission is None:
            return self._rejected_projection(
                source_step,
                f"source_rejected:{source_step.rejection_reason}",
            )
        try:
            emission = source_step.emission
            emission_hash = emission.semantic_content_hash
            if len(emission.merged_commands) != 1:
                raise ValueError("B601 pose emission must contain one command.")
            command = emission.merged_commands[0]
            if command.effector_name != self.pose_source.spec.effector_name:
                raise ValueError("B601 pose emission effector is incompatible.")
            if command.command_type != "cartesian_pose":
                raise ValueError("B601 pose emission is not Cartesian pose.")
            if command.frame_name != self.pose_source.spec.reference_frame:
                raise ValueError("B601 pose emission reference frame is incompatible.")
            if command.metadata.get("controlled_frame") != B601_RS_END_EFFECTOR_FRAME:
                raise ValueError("B601 pose emission controlled frame is incompatible.")
            if len(emission.observation_sample_refs) != 1:
                raise ValueError(
                    "B601 pose emission must bind one observation sample."
                )
            observation = next(iter(emission.observation_sample_refs.values()))
            if not observation.valid or observation.stale:
                raise ValueError("B601 pose observation is invalid or stale.")
            if (
                observation.measured_clock_domain_id
                != self.pose_source.spec.clock_domain_id
                or observation.available_clock_domain_id
                != self.pose_source.spec.clock_domain_id
            ):
                raise ValueError("B601 pose observation clock domain is incompatible.")
            sample = HumanPoseSample(
                component_id=self.component_id,
                sample_id=(
                    f"b601-rm65-pose:{emission_hash}:"
                    f"{observation.sample_id}"
                ),
                measured_at=observation.measured_at,
                measured_clock_domain_id=observation.measured_clock_domain_id,
                source_reference_frame_id=self.pose_source.spec.reference_frame,
                source_controlled_frame_id=B601_RS_END_EFFECTOR_FRAME,
                source_coordinate_binding_id=self.source_coordinate_binding_id,
                source_coordinate_binding_hash=(
                    self.source_coordinate_binding_hash
                ),
                pose_source_reference_source_controlled=command.value,
            )
        except (TypeError, ValueError) as error:
            return self._rejected_projection(
                source_step,
                f"projection_rejected:{type(error).__name__}:{error}",
            )
        return B601RM65PoseProjection(
            source_step=source_step,
            pose_sample=sample,
            source_emission_hash=emission_hash,
            source_observation_sample_id=observation.sample_id,
            rejection_reason=None,
        )

    def capture_anchor(
        self,
        projection: B601RM65PoseProjection,
        *,
        anchor_generation: int,
        captured_at: float,
        captured_clock_domain_id: str,
    ) -> HumanResidualPoseAnchor:
        """Capture an activation or recenter anchor from one admitted sample."""

        if not isinstance(projection, B601RM65PoseProjection):
            raise TypeError("projection must be B601RM65PoseProjection.")
        if projection.pose_sample is None:
            raise ValueError(
                "Cannot capture an anchor from a rejected B601 pose projection."
            )
        return self.residual_adapter.capture_anchor(
            (projection.pose_sample,),
            anchor_generation=anchor_generation,
            captured_at=captured_at,
            captured_clock_domain_id=captured_clock_domain_id,
        )

    def adapt(
        self,
        projection: B601RM65PoseProjection,
        *,
        anchor: HumanResidualPoseAnchor,
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> B601RM65ResidualStep:
        """Adapt the current sample; any rejection withdraws this tick."""

        if not isinstance(projection, B601RM65PoseProjection):
            raise TypeError("projection must be B601RM65PoseProjection.")
        if projection.pose_sample is None:
            return B601RM65ResidualStep(
                projection=projection,
                adaptation=None,
                rejection_reason=(
                    "current_pose_unavailable:"
                    f"{projection.rejection_reason}"
                ),
            )
        try:
            adaptation = self.residual_adapter.adapt(
                (projection.pose_sample,),
                anchor=anchor,
                adapted_at=adapted_at,
                adapted_clock_domain_id=adapted_clock_domain_id,
            )
        except (TypeError, ValueError) as error:
            return B601RM65ResidualStep(
                projection=projection,
                adaptation=None,
                rejection_reason=(
                    f"adaptation_rejected:{type(error).__name__}:{error}"
                ),
            )
        return B601RM65ResidualStep(
            projection=projection,
            adaptation=adaptation,
            rejection_reason=None,
        )

    def poll_and_adapt(
        self,
        *,
        anchor: HumanResidualPoseAnchor,
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> B601RM65ResidualStep:
        """Poll and adapt exactly once without retaining a previous action."""

        return self.adapt(
            self.poll_pose(),
            anchor=anchor,
            adapted_at=adapted_at,
            adapted_clock_domain_id=adapted_clock_domain_id,
        )

    @staticmethod
    def _rejected_projection(
        source_step: B601ReadOnlyPoseStep,
        reason: str,
    ) -> B601RM65PoseProjection:
        return B601RM65PoseProjection(
            source_step=source_step,
            pose_sample=None,
            source_emission_hash=None,
            source_observation_sample_id=None,
            rejection_reason=reason,
        )


__all__ = [
    "B601_RM65_RESIDUAL_SOURCE_CONTRACT_ID_V1",
    "B601_RM65_RESIDUAL_SOURCE_HASH_DOMAIN_V1",
    "B601RM65PoseProjection",
    "B601RM65ResidualSource",
    "B601RM65ResidualStep",
]
