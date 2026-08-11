"""Default-off Station gate for atomic derived composition inputs."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from control_composition.activation import CompositionActivationScope
from control_composition.composition import (
    CompositionInputBatchAdmissionReceipt,
    CompositionInputBatchDraft,
)
from control_composition.residual_anchor import (
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
)
from control_composition.residual_anchor_lifecycle import (
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    ResidualAnchorLifecycleAdmissionReceipt,
    ResidualAnchorLifecycleDraft,
    ResidualAnchorLifecycleSpec,
)

from ..publication_barrier import StationPublicationBarrier


STATION_DERIVED_INPUT_PUBLICATION_PROFILE_OFF = "off"
_ACTIVE_COMPOSITION_BARRIER_REASONS = frozenset(
    {
        "typed_composition_session_completion",
        "typed_composition_session_rearm_completion",
    }
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class StationDerivedInputPublicationProfile:
    profile_id: str = STATION_DERIVED_INPUT_PUBLICATION_PROFILE_OFF
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
            raise ValueError("Derived input publication cannot send actions.")
        if self.enabled and profile_id == STATION_DERIVED_INPUT_PUBLICATION_PROFILE_OFF:
            raise ValueError("Enabled derived input publication needs a profile id.")
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class StationDerivedInputPublicationCycle:
    profile_id: str
    status: str
    batch: CompositionInputBatchDraft | None = None
    receipt: CompositionInputBatchAdmissionReceipt | None = None

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        status = _required(self.status, name="status")
        if status not in {"disabled", "admitted", "rejected"}:
            raise ValueError("Unsupported derived input publication status.")
        if status == "disabled":
            if self.batch is not None or self.receipt is not None:
                raise ValueError("Disabled publication cannot carry a batch.")
        else:
            if not isinstance(self.batch, CompositionInputBatchDraft) or not isinstance(
                self.receipt,
                CompositionInputBatchAdmissionReceipt,
            ):
                raise TypeError("Active publication requires typed batch and receipt.")
            if (status == "admitted") != self.receipt.accepted:
                raise ValueError("Publication status and batch receipt disagree.")
            if (
                self.receipt.batch_id != self.batch.batch_id
                or self.receipt.batch_semantic_content_hash
                != self.batch.semantic_content_hash
            ):
                raise ValueError("Publication receipt belongs to another batch.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)


class StationDerivedInputPublicationPort:
    """Linearize one prepared input batch against a composition session."""

    def __init__(
        self,
        *,
        profile: StationDerivedInputPublicationProfile,
        activation_scope: CompositionActivationScope | None = None,
        publication_barrier: StationPublicationBarrier | None = None,
        admit_batch: (
            Callable[
                [CompositionInputBatchDraft],
                CompositionInputBatchAdmissionReceipt,
            ]
            | None
        ) = None,
        runtime_session_id: str | None = None,
        control_domain_id: str | None = None,
        lifecycle_producer_id: str | None = None,
        lifecycle_producer_epoch: int | None = None,
        lifecycle_scope_id: str | None = None,
        lifecycle_spec: ResidualAnchorLifecycleSpec | None = None,
        admit_lifecycle: (
            Callable[
                [ResidualAnchorLifecycleDraft],
                ResidualAnchorLifecycleAdmissionReceipt,
            ]
            | None
        ) = None,
    ) -> None:
        if not isinstance(profile, StationDerivedInputPublicationProfile):
            raise TypeError("profile must be StationDerivedInputPublicationProfile.")
        if profile.enabled:
            if not isinstance(activation_scope, CompositionActivationScope):
                raise TypeError("Enabled publication requires activation_scope.")
            if not isinstance(publication_barrier, StationPublicationBarrier):
                raise TypeError("Enabled publication requires Station barrier.")
            if not callable(admit_batch):
                raise TypeError("Enabled publication requires batch admitter.")
            snapshot = publication_barrier.snapshot()
            if (
                not snapshot.enabled
                or snapshot.authority_operation_result_id is None
                or snapshot.transition_reason
                not in _ACTIVE_COMPOSITION_BARRIER_REASONS
            ):
                raise RuntimeError(
                    "Enabled derived input publication requires an active "
                    "composition-session barrier."
                )
            barrier_generation = snapshot.generation
            authority_result_id = snapshot.authority_operation_result_id
        else:
            barrier_generation = None
            authority_result_id = None
        lifecycle_values = (
            runtime_session_id,
            control_domain_id,
            lifecycle_producer_id,
            lifecycle_producer_epoch,
            lifecycle_scope_id,
            lifecycle_spec,
            admit_lifecycle,
        )
        lifecycle_enabled = any(value is not None for value in lifecycle_values)
        if lifecycle_enabled and any(value is None for value in lifecycle_values):
            raise ValueError(
                "Residual anchor lifecycle publication binding is incomplete."
            )
        if lifecycle_enabled:
            if not profile.enabled:
                raise ValueError(
                    "Residual anchor lifecycle publication requires an enabled "
                    "derived-input profile."
                )
            assert lifecycle_producer_epoch is not None
            if (
                isinstance(lifecycle_producer_epoch, bool)
                or not isinstance(lifecycle_producer_epoch, int)
                or lifecycle_producer_epoch < 0
            ):
                raise ValueError(
                    "lifecycle_producer_epoch must be a non-negative integer."
                )
            if not isinstance(lifecycle_spec, ResidualAnchorLifecycleSpec):
                raise TypeError("lifecycle_spec must be typed.")
            if not callable(admit_lifecycle):
                raise TypeError("admit_lifecycle must be callable.")
        self.profile = profile
        self.activation_scope = activation_scope
        self.publication_barrier = publication_barrier
        self._admit_batch = admit_batch
        self._barrier_generation = barrier_generation
        self._authority_operation_result_id = authority_result_id
        self.runtime_session_id = (
            None
            if runtime_session_id is None
            else _required(runtime_session_id, name="runtime_session_id")
        )
        self.control_domain_id = (
            None
            if control_domain_id is None
            else _required(control_domain_id, name="control_domain_id")
        )
        self.lifecycle_producer_id = (
            None
            if lifecycle_producer_id is None
            else _required(lifecycle_producer_id, name="lifecycle_producer_id")
        )
        self.lifecycle_producer_epoch = lifecycle_producer_epoch
        self.lifecycle_scope_id = (
            None
            if lifecycle_scope_id is None
            else _required(lifecycle_scope_id, name="lifecycle_scope_id")
        )
        self.lifecycle_spec = lifecycle_spec
        self._admit_lifecycle = admit_lifecycle
        self._lifecycle_sequence = 0
        self._lifecycle_lock = threading.RLock()

    @property
    def lifecycle_enabled(self) -> bool:
        return self.lifecycle_spec is not None

    def publish_anchor_lifecycle(
        self,
        request: ResidualAnchorRequest,
        *,
        phase: str,
        effective_for_target_at: float,
        observed_at: float,
        anchor_receipt: ResidualAnchorReceipt | None = None,
    ) -> ResidualAnchorLifecycleAdmissionReceipt:
        """Publish one pending/ready transition through the same Station gate."""

        if not self.lifecycle_enabled:
            raise RuntimeError(
                "This derived-input port has no anchor lifecycle binding."
            )
        if not isinstance(request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        if phase not in {
            RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
            RESIDUAL_ANCHOR_LIFECYCLE_READY,
        }:
            raise ValueError("Unsupported residual anchor lifecycle phase.")
        if (phase == RESIDUAL_ANCHOR_LIFECYCLE_READY) != (
            anchor_receipt is not None
        ):
            raise ValueError("Only ready lifecycle carries an anchor receipt.")
        assert self.activation_scope is not None
        if (
            request.composition_activation_id
            != self.activation_scope.activation_id
            or request.composition_activation_hash
            != self.activation_scope.semantic_content_hash
        ):
            raise ValueError("Anchor lifecycle belongs to another activation.")
        assert self.publication_barrier is not None
        return self.publication_barrier.execute_if_enabled(
            lambda: self._publish_anchor_lifecycle_enabled(
                request,
                phase=phase,
                effective_for_target_at=effective_for_target_at,
                observed_at=observed_at,
                anchor_receipt=anchor_receipt,
            )
        )

    def _publish_anchor_lifecycle_enabled(
        self,
        request: ResidualAnchorRequest,
        *,
        phase: str,
        effective_for_target_at: float,
        observed_at: float,
        anchor_receipt: ResidualAnchorReceipt | None,
    ) -> ResidualAnchorLifecycleAdmissionReceipt:
        assert self.publication_barrier is not None
        snapshot = self.publication_barrier.snapshot()
        if (
            snapshot.generation != self._barrier_generation
            or snapshot.authority_operation_result_id
            != self._authority_operation_result_id
        ):
            raise RuntimeError(
                "Anchor lifecycle publisher belongs to another composition session."
            )
        with self._lifecycle_lock:
            assert self.runtime_session_id is not None
            assert self.control_domain_id is not None
            assert self.lifecycle_producer_id is not None
            assert self.lifecycle_producer_epoch is not None
            assert self.lifecycle_scope_id is not None
            assert self.lifecycle_spec is not None
            assert self._admit_lifecycle is not None
            draft = ResidualAnchorLifecycleDraft(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                producer_id=self.lifecycle_producer_id,
                producer_epoch=self.lifecycle_producer_epoch,
                sequence=self._lifecycle_sequence,
                lifecycle_scope_id=self.lifecycle_scope_id,
                lifecycle_spec_hash=self.lifecycle_spec.semantic_content_hash,
                phase=phase,
                effective_for_target_at=effective_for_target_at,
                observed_at=observed_at,
                observed_clock_domain_id=self.activation_scope.clock_domain_id,
                anchor_request=request,
                anchor_receipt=anchor_receipt,
            )
            receipt = self._admit_lifecycle(draft)
            if not isinstance(receipt, ResidualAnchorLifecycleAdmissionReceipt):
                raise TypeError("Lifecycle admitter must return a typed receipt.")
            if receipt.accepted:
                self._lifecycle_sequence += 1
            return receipt

    def publish(
        self,
        batch_factory: Callable[[], CompositionInputBatchDraft],
    ) -> StationDerivedInputPublicationCycle:
        if not callable(batch_factory):
            raise TypeError("batch_factory must be callable.")
        if not self.profile.enabled:
            return StationDerivedInputPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        assert self.publication_barrier is not None
        return self.publication_barrier.execute_if_enabled(
            lambda: self._publish_enabled(batch_factory)
        )

    def _publish_enabled(
        self,
        batch_factory: Callable[[], CompositionInputBatchDraft],
    ) -> StationDerivedInputPublicationCycle:
        assert self.publication_barrier is not None
        assert self.activation_scope is not None
        assert self._admit_batch is not None
        snapshot = self.publication_barrier.snapshot()
        if (
            snapshot.generation != self._barrier_generation
            or snapshot.authority_operation_result_id
            != self._authority_operation_result_id
        ):
            raise RuntimeError(
                "Derived input publisher belongs to another composition session."
            )
        batch = batch_factory()
        if not isinstance(batch, CompositionInputBatchDraft):
            raise TypeError("batch_factory must return CompositionInputBatchDraft.")
        for item in batch.inputs:
            if (
                item.composition_activation_id
                != self.activation_scope.activation_id
                or item.composition_activation_hash
                != self.activation_scope.semantic_content_hash
            ):
                raise ValueError("Derived input batch belongs to another activation.")
        receipt = self._admit_batch(batch)
        if not isinstance(receipt, CompositionInputBatchAdmissionReceipt):
            raise TypeError("Batch admitter must return a typed receipt.")
        return StationDerivedInputPublicationCycle(
            profile_id=self.profile.profile_id,
            status="admitted" if receipt.accepted else "rejected",
            batch=batch,
            receipt=receipt,
        )


__all__ = [
    "STATION_DERIVED_INPUT_PUBLICATION_PROFILE_OFF",
    "StationDerivedInputPublicationCycle",
    "StationDerivedInputPublicationPort",
    "StationDerivedInputPublicationProfile",
]
