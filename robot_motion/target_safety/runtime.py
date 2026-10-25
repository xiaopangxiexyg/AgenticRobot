"""Lower-owned lifecycle for selectable TCP-pose safety envelopes.

The numeric projector remains in :mod:`tcp_pose_envelope`.  This module owns
only the lower lifecycle needed to use it safely across multiple Skill
attempts:

* one deployment-reviewed envelope catalog;
* one stable measured ``SafetyReferenceEpoch``;
* one operation-local envelope/profile activation bound to the volatile
  execution epoch; and
* one final pose-action projection callable for an existing pre-IK seam.

It does not own READY/HOLD, IK, dispatch, scene reset, or Agent policy.  The
lower application must call the lifecycle methods while it owns those
transitions.  In particular, selection activation must finish before READY,
and HOLD/reset must deactivate or invalidate through the same application
lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import RobotAction

from .tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeError,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeProjectionResult,
    TCPPoseSafetyReferenceReceipt,
    project_tcp_pose_envelope,
)


DEPLOYMENT_SAFETY_ENVELOPE_SCHEMA_V1 = (
    "dexteleop.deployment_safety_envelope.v1"
)
TCP_POSE_ENVELOPE_CATALOG_ENTRY_HASH_DOMAIN_V1 = (
    "dexteleop.deployment_safety_envelope.v1"
)
TCP_POSE_ENVELOPE_SELECTION_HASH_DOMAIN_V1 = (
    "dexteleop.tcp_pose_envelope_selection.v1"
)
TCP_POSE_ENVELOPE_ACTIVATION_HASH_DOMAIN_V1 = (
    "dexteleop.tcp_pose_envelope_activation.v1"
)
TCP_POSE_ENVELOPE_ACTION_FILTER_HASH_DOMAIN_V1 = (
    "dexteleop.tcp_pose_envelope_action_filter.v1"
)


class TCPPoseEnvelopeLifecycleError(RuntimeError):
    """Fail-closed lower lifecycle or selection error."""


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace")
    return value


def _nonnegative(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _events(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    result = tuple(_required(item, field_name=field_name) for item in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return result


def _text_mapping(
    values: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    result = {
        _required(str(key), field_name=f"{field_name}.key"): _required(
            value,
            field_name=f"{field_name}.{key}",
        )
        for key, value in values.items()
    }
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class TCPPoseEnvelopeCatalogEntry:
    """One exact lower deployment envelope, independent of Agent tiers."""

    envelope_id: str
    profile: TCPPoseEnvelopeProfile
    constraint_set_hash: str
    allowed_route_ids: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        envelope_id = _required(self.envelope_id, field_name="envelope_id")
        constraint_hash = _required(
            self.constraint_set_hash,
            field_name="constraint_set_hash",
        )
        if not isinstance(self.profile, TCPPoseEnvelopeProfile):
            raise TypeError("profile must be TCPPoseEnvelopeProfile")
        routes = tuple(
            sorted(
                {
                    _required(item, field_name="allowed_route_ids.item")
                    for item in self.allowed_route_ids
                }
            )
        )
        if not routes:
            raise ValueError("allowed_route_ids must not be empty")
        object.__setattr__(self, "envelope_id", envelope_id)
        object.__setattr__(self, "constraint_set_hash", constraint_hash)
        object.__setattr__(self, "allowed_route_ids", routes)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "schema": DEPLOYMENT_SAFETY_ENVELOPE_SCHEMA_V1,
                    "envelope_id": envelope_id,
                    "tcp_pose_envelope_profile_hash": (
                        self.profile.semantic_content_hash
                    ),
                    "constraint_set_hash": constraint_hash,
                    "allowed_route_ids": routes,
                },
                domain=TCP_POSE_ENVELOPE_CATALOG_ENTRY_HASH_DOMAIN_V1,
            ),
        )

    def to_ref_payload(self) -> dict[str, str]:
        return {
            "object_id": self.envelope_id,
            "schema": DEPLOYMENT_SAFETY_ENVELOPE_SCHEMA_V1,
            "content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TCPPoseEnvelopeSelection:
    """Exact trusted selection injected into a lower Skill request."""

    safety_profile_id: str
    safety_profile_hash: str
    deployment_envelope_id: str
    deployment_envelope_hash: str
    constraint_set_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            field_name: _required(
                getattr(self, field_name),
                field_name=field_name,
            )
            for field_name in (
                "safety_profile_id",
                "safety_profile_hash",
                "deployment_envelope_id",
                "deployment_envelope_hash",
                "constraint_set_hash",
            )
        }
        for field_name, value in payload.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TCP_POSE_ENVELOPE_SELECTION_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class TCPPoseEnvelopeActivationReceipt:
    """Lower proof that one selection was bound before operation READY."""

    activation_id: str
    operation_id: str
    route_id: str
    execution_epoch: int
    selection_hash: str
    deployment_envelope_hash: str
    tcp_pose_envelope_profile_hash: str
    safety_reference_hash: str | None
    operation_anchor_hash: str | None
    source_event_ids: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload: dict[str, Any] = {
            field_name: _required(
                getattr(self, field_name),
                field_name=field_name,
            )
            for field_name in (
                "activation_id",
                "operation_id",
                "route_id",
                "selection_hash",
                "deployment_envelope_hash",
                "tcp_pose_envelope_profile_hash",
            )
        }
        epoch = _nonnegative(self.execution_epoch, field_name="execution_epoch")
        for field_name in ("safety_reference_hash", "operation_anchor_hash"):
            value = getattr(self, field_name)
            payload[field_name] = (
                None
                if value is None
                else _required(value, field_name=field_name)
            )
        if (payload["safety_reference_hash"] is None) != (
            payload["operation_anchor_hash"] is None
        ):
            raise ValueError(
                "safety-reference and operation-anchor hashes must be paired"
            )
        events = _events(self.source_event_ids, field_name="source_event_ids")
        payload["execution_epoch"] = epoch
        payload["source_event_ids"] = events
        for field_name, value in payload.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TCP_POSE_ENVELOPE_ACTIVATION_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class TCPPoseEnvelopeRuntimeSnapshot:
    safety_reference_epoch: int
    safety_reference_hash: str | None
    active_operation_id: str | None
    active_execution_epoch: int | None
    active_selection_hash: str | None
    active_envelope_hash: str | None
    active_anchor_hash: str | None
    last_projection_hash: str | None


@dataclass(frozen=True)
class _ActiveEnvelope:
    operation_id: str
    execution_epoch: int
    selection: TCPPoseEnvelopeSelection
    entry: TCPPoseEnvelopeCatalogEntry
    anchor: TCPPoseEnvelopeAnchorReceipt | None
    receipt: TCPPoseEnvelopeActivationReceipt


class TCPPoseEnvelopeSafetyRuntime:
    """One lower-owned catalog/reference/activation state machine."""

    def __init__(
        self,
        *,
        catalog_entries: tuple[TCPPoseEnvelopeCatalogEntry, ...],
        target_frame_binding_hash: str,
        tool_frame_binding_hash_by_side: Mapping[str, str],
        reference_frame_by_side: Mapping[str, str],
        clock_domain_id: str,
        controlled_frame_id: str = TCP_CONTROLLED_FRAME_ID,
    ) -> None:
        entries = tuple(catalog_entries)
        if not entries or any(
            not isinstance(item, TCPPoseEnvelopeCatalogEntry)
            for item in entries
        ):
            raise TypeError("catalog_entries must contain typed entries")
        by_hash = {item.semantic_content_hash: item for item in entries}
        by_id = {item.envelope_id: item for item in entries}
        if len(by_hash) != len(entries) or len(by_id) != len(entries):
            raise ValueError("deployment envelope IDs and hashes must be unique")
        target_hash = _required(
            target_frame_binding_hash,
            field_name="target_frame_binding_hash",
        )
        tool_hashes = _text_mapping(
            tool_frame_binding_hash_by_side,
            field_name="tool_frame_binding_hash_by_side",
        )
        references = _text_mapping(
            reference_frame_by_side,
            field_name="reference_frame_by_side",
        )
        controlled = _required(
            controlled_frame_id,
            field_name="controlled_frame_id",
        )
        if controlled != TCP_CONTROLLED_FRAME_ID:
            raise ValueError("TCP envelope runtime requires tool_tcp")
        for entry in entries:
            profile = entry.profile
            if profile.target_frame_binding_hash != target_hash:
                raise ValueError("catalog profile target-frame binding mismatch")
            for side in profile.enabled_sides:
                if profile.tool_frame_binding_hash_by_side[side] != tool_hashes.get(side):
                    raise ValueError(
                        f"catalog profile tool binding mismatch for {side!r}"
                    )
                if profile.reference_frame_by_side[side] != references.get(side):
                    raise ValueError(
                        f"catalog profile reference frame mismatch for {side!r}"
                    )
        self.catalog_entries = tuple(
            sorted(entries, key=lambda item: item.envelope_id)
        )
        self._entry_by_hash = MappingProxyType(by_hash)
        self._entry_by_id = MappingProxyType(by_id)
        self.target_frame_binding_hash = target_hash
        self.tool_frame_binding_hash_by_side = tool_hashes
        self.reference_frame_by_side = references
        self.controlled_frame_id = controlled
        self.clock_domain_id = _required(
            clock_domain_id,
            field_name="clock_domain_id",
        )
        self._lock = threading.RLock()
        self._safety_reference_epoch = 0
        self._reference: TCPPoseSafetyReferenceReceipt | None = None
        self._active: _ActiveEnvelope | None = None
        self._last_projection: TCPPoseEnvelopeProjectionResult | None = None

    def default_selection(self) -> TCPPoseEnvelopeSelection:
        """Return the sole lower default for compatibility-only callers.

        A deployment with multiple envelopes has no implicit default: its
        caller must supply an exact reviewed selection.
        """

        if len(self.catalog_entries) != 1:
            raise TCPPoseEnvelopeLifecycleError(
                "multiple safety envelopes require an explicit selection"
            )
        entry = self.catalog_entries[0]
        profile_hash = canonical_content_hash(
            {
                "source": "lower_single_envelope_compatibility_default",
                "deployment_envelope_hash": entry.semantic_content_hash,
                "constraint_set_hash": entry.constraint_set_hash,
            },
            domain="dexteleop.lower_default_safety_profile.v1",
        )
        return TCPPoseEnvelopeSelection(
            safety_profile_id="lower-default.standard@1",
            safety_profile_hash=profile_hash,
            deployment_envelope_id=entry.envelope_id,
            deployment_envelope_hash=entry.semantic_content_hash,
            constraint_set_hash=entry.constraint_set_hash,
        )

    def invalidate_reference(
        self,
        *,
        reason: str,
        source_event_ids: tuple[str, ...],
        lower_state: str,
    ) -> int:
        """Invalidate on scene/reset boundary; never on ordinary HOLD."""

        _required(reason, field_name="reason")
        _events(source_event_ids, field_name="source_event_ids")
        if lower_state not in {"configured", "hold"}:
            raise TCPPoseEnvelopeLifecycleError(
                "safety reference may be invalidated only while lower motion is closed"
            )
        with self._lock:
            if self._active is not None:
                raise TCPPoseEnvelopeLifecycleError(
                    "cannot invalidate a safety reference with an active operation"
                )
            self._safety_reference_epoch += 1
            self._reference = None
            self._last_projection = None
            return self._safety_reference_epoch

    def commit_reference(
        self,
        *,
        reference_id: str,
        tcp_pose_by_side: Mapping[str, object],
        captured_at_by_side: Mapping[str, float],
        source_event_ids: tuple[str, ...],
        lower_state: str,
    ) -> TCPPoseSafetyReferenceReceipt:
        """Commit one fresh measured cut after lower physical initialization."""

        if lower_state not in {"configured", "hold"}:
            raise TCPPoseEnvelopeLifecycleError(
                "safety reference may be committed only while lower motion is closed"
            )
        with self._lock:
            if self._active is not None:
                raise TCPPoseEnvelopeLifecycleError(
                    "cannot commit a safety reference with an active operation"
                )
            if self._reference is not None:
                raise TCPPoseEnvelopeLifecycleError(
                    "invalidate the prior safety reference before recommit"
                )
            reference = TCPPoseSafetyReferenceReceipt(
                reference_id=reference_id,
                safety_reference_epoch=self._safety_reference_epoch,
                target_frame_binding_hash=self.target_frame_binding_hash,
                tool_frame_binding_hash_by_side=self.tool_frame_binding_hash_by_side,
                controlled_frame_id=self.controlled_frame_id,
                clock_domain_id=self.clock_domain_id,
                tcp_pose_by_side=tcp_pose_by_side,
                reference_frame_by_side={
                    side: self.reference_frame_by_side[side]
                    for side in tcp_pose_by_side
                },
                captured_at_by_side=captured_at_by_side,
                source_event_ids=source_event_ids,
            )
            self._reference = reference
            self._last_projection = None
            return reference

    def activate(
        self,
        *,
        selection: TCPPoseEnvelopeSelection,
        operation_id: str,
        route_id: str,
        execution_epoch: int,
        source_event_ids: tuple[str, ...],
        lower_state: str,
    ) -> TCPPoseEnvelopeActivationReceipt:
        """Resolve and bind one exact profile before the lower opens READY."""

        if not isinstance(selection, TCPPoseEnvelopeSelection):
            raise TypeError("selection must be TCPPoseEnvelopeSelection")
        if lower_state not in {"configured", "hold"}:
            raise TCPPoseEnvelopeLifecycleError(
                "envelope selection must activate before lower READY"
            )
        normalized_operation = _required(operation_id, field_name="operation_id")
        normalized_route = _required(route_id, field_name="route_id")
        epoch = _nonnegative(execution_epoch, field_name="execution_epoch")
        events = _events(source_event_ids, field_name="source_event_ids")
        with self._lock:
            if self._active is not None:
                if (
                    self._active.operation_id == normalized_operation
                    and self._active.selection == selection
                    and self._active.execution_epoch == epoch
                ):
                    return self._active.receipt
                raise TCPPoseEnvelopeLifecycleError(
                    "another operation already owns the active safety envelope"
                )
            entry = self._entry_by_hash.get(selection.deployment_envelope_hash)
            if entry is None or entry.envelope_id != selection.deployment_envelope_id:
                raise TCPPoseEnvelopeLifecycleError(
                    "selected deployment safety envelope is not in the lower catalog"
                )
            if entry.constraint_set_hash != selection.constraint_set_hash:
                raise TCPPoseEnvelopeLifecycleError(
                    "selected safety constraint-set hash does not match lower"
                )
            if normalized_route not in entry.allowed_route_ids:
                raise TCPPoseEnvelopeLifecycleError(
                    "selected safety envelope does not allow this Route"
                )
            reference = self._reference
            anchor = None
            if entry.profile.anchor_required_sides:
                if reference is None:
                    raise TCPPoseEnvelopeLifecycleError(
                        "relative safety envelope has no committed SafetyReference"
                    )
                anchor = reference.bind_profile(
                    anchor_id=(
                        f"operation-anchor:{normalized_operation}:"
                        f"reference-{reference.safety_reference_epoch}"
                    ),
                    profile=entry.profile,
                    execution_epoch=epoch,
                )
            receipt = TCPPoseEnvelopeActivationReceipt(
                activation_id=(
                    f"tcp-envelope-activation:{normalized_operation}:epoch-{epoch}"
                ),
                operation_id=normalized_operation,
                route_id=normalized_route,
                execution_epoch=epoch,
                selection_hash=selection.semantic_content_hash,
                deployment_envelope_hash=entry.semantic_content_hash,
                tcp_pose_envelope_profile_hash=entry.profile.semantic_content_hash,
                safety_reference_hash=(
                    None
                    if anchor is None
                    else reference.semantic_content_hash
                ),
                operation_anchor_hash=(
                    None if anchor is None else anchor.semantic_content_hash
                ),
                source_event_ids=events,
            )
            self._active = _ActiveEnvelope(
                operation_id=normalized_operation,
                execution_epoch=epoch,
                selection=selection,
                entry=entry,
                anchor=anchor,
                receipt=receipt,
            )
            self._last_projection = None
            return receipt

    def deactivate_after_hold(
        self,
        *,
        operation_id: str | None,
        next_execution_epoch: int,
        hold_evidence_ids: tuple[str, ...],
    ) -> None:
        """Clear only the operation binding; preserve the stable reference."""

        _nonnegative(next_execution_epoch, field_name="next_execution_epoch")
        _events(hold_evidence_ids, field_name="hold_evidence_ids")
        with self._lock:
            if self._active is None:
                return
            if operation_id is not None and (
                self._active.operation_id
                != _required(operation_id, field_name="operation_id")
            ):
                raise TCPPoseEnvelopeLifecycleError(
                    "HOLD operation does not own the active safety envelope"
                )
            self._active = None
            self._last_projection = None

    def preview_for_envelope(
        self, raw_tcp_pose_by_side: Mapping[str, object], *,
        deployment_envelope_hash: str,
    ) -> TCPPoseEnvelopeProjectionResult:
        """Query a catalog envelope without activating or changing runtime state.

        Relative terms use the same committed initialization reference as an
        operation, never a caller-made anchor. This result cannot open READY.
        """
        with self._lock:
            entry = self._entry_by_hash.get(deployment_envelope_hash)
            if entry is None:
                raise TCPPoseEnvelopeLifecycleError("preview envelope is absent from lower catalog")
            anchor = None
            if entry.profile.anchor_required_sides:
                if self._reference is None:
                    raise TCPPoseEnvelopeLifecycleError("preview envelope has no committed SafetyReference")
                anchor = self._reference.bind_profile(
                    anchor_id="read-only-envelope-preview", profile=entry.profile,
                    execution_epoch=0,
                )
            return project_tcp_pose_envelope(
                raw_tcp_pose_by_side, profile=entry.profile,
                target_frame_binding_hash=self.target_frame_binding_hash,
                tool_frame_binding_hash_by_side=self.tool_frame_binding_hash_by_side,
                controlled_frame_id=self.controlled_frame_id,
                reference_frame_by_side=self.reference_frame_by_side,
                execution_epoch=0, anchor_receipt=anchor,
            )

    def planning_projector_for_route(
        self,
        *,
        route_id: str,
    ) -> Callable[[Mapping[str, object]], TCPPoseEnvelopeProjectionResult]:
        """Freeze an unambiguous Route/reference for read-only planning queries.

        This does not activate an envelope, issue an activation receipt, or
        update projection state. The actual execution must bind its own fresh
        operation-local selection through ``activate`` as before.
        """
        route = _required(route_id, field_name="route_id")
        with self._lock:
            if self._active is not None:
                raise TCPPoseEnvelopeLifecycleError("planning preview requires closed motion")
            entries = [entry for entry in self.catalog_entries if route in entry.allowed_route_ids]
            if len(entries) != 1:
                raise TCPPoseEnvelopeLifecycleError("planning preview requires one exact Route envelope")
            entry = entries[0]
            reference = self._reference
            anchor = None
            if entry.profile.anchor_required_sides:
                if self._reference is None:
                    raise TCPPoseEnvelopeLifecycleError("planning preview has no committed SafetyReference")
                anchor = self._reference.bind_profile(anchor_id="planning-preview:"+route,
                    profile=entry.profile, execution_epoch=0)
        def project(raw_tcp_pose_by_side: Mapping[str, object]) -> TCPPoseEnvelopeProjectionResult:
            with self._lock:
                if self._active is not None:
                    raise TCPPoseEnvelopeLifecycleError("planning preview requires closed motion")
                if self._reference is not reference:
                    raise TCPPoseEnvelopeLifecycleError("planning preview reference changed")
                return project_tcp_pose_envelope(raw_tcp_pose_by_side,
                    profile=entry.profile, target_frame_binding_hash=self.target_frame_binding_hash,
                    tool_frame_binding_hash_by_side=self.tool_frame_binding_hash_by_side,
                    controlled_frame_id=self.controlled_frame_id,
                    reference_frame_by_side=self.reference_frame_by_side,
                    execution_epoch=0, anchor_receipt=anchor)
        return project


    def project(
        self,
        raw_tcp_pose_by_side: Mapping[str, object],
    ) -> TCPPoseEnvelopeProjectionResult:
        """Apply the active exact envelope; absence is a fail-closed error."""

        with self._lock:
            active = self._active
            if active is None:
                raise TCPPoseEnvelopeLifecycleError(
                    "no operation-local safety envelope is active"
                )
            result = project_tcp_pose_envelope(
                raw_tcp_pose_by_side,
                profile=active.entry.profile,
                target_frame_binding_hash=self.target_frame_binding_hash,
                tool_frame_binding_hash_by_side=self.tool_frame_binding_hash_by_side,
                controlled_frame_id=self.controlled_frame_id,
                reference_frame_by_side=self.reference_frame_by_side,
                execution_epoch=active.execution_epoch,
                anchor_receipt=active.anchor,
            )
            self._last_projection = result
            return result

    def snapshot(self) -> TCPPoseEnvelopeRuntimeSnapshot:
        with self._lock:
            active = self._active
            reference = self._reference
            return TCPPoseEnvelopeRuntimeSnapshot(
                safety_reference_epoch=self._safety_reference_epoch,
                safety_reference_hash=(
                    None if reference is None else reference.semantic_content_hash
                ),
                active_operation_id=(
                    None if active is None else active.operation_id
                ),
                active_execution_epoch=(
                    None if active is None else active.execution_epoch
                ),
                active_selection_hash=(
                    None
                    if active is None
                    else active.selection.semantic_content_hash
                ),
                active_envelope_hash=(
                    None
                    if active is None
                    else active.entry.semantic_content_hash
                ),
                active_anchor_hash=(
                    None
                    if active is None or active.anchor is None
                    else active.anchor.semantic_content_hash
                ),
                last_projection_hash=(
                    None
                    if self._last_projection is None
                    else self._last_projection.semantic_content_hash
                ),
            )


class RobotActionTCPPoseEnvelopeFilter:
    """Apply the lower runtime to an exact arm-pose action before IK."""

    def __init__(self, runtime: TCPPoseEnvelopeSafetyRuntime) -> None:
        if not isinstance(runtime, TCPPoseEnvelopeSafetyRuntime):
            raise TypeError("runtime must be TCPPoseEnvelopeSafetyRuntime")
        self.runtime = runtime
        self.filter_contract_hash = canonical_content_hash(
            {
                "catalog_entry_hashes": tuple(
                    entry.semantic_content_hash
                    for entry in runtime.catalog_entries
                ),
                "target_frame_binding_hash": runtime.target_frame_binding_hash,
                "tool_frame_binding_hash_by_side": (
                    runtime.tool_frame_binding_hash_by_side
                ),
                "reference_frame_by_side": runtime.reference_frame_by_side,
                "controlled_frame_id": runtime.controlled_frame_id,
                "stage": "final_after_pose_transform_before_ik",
            },
            domain=TCP_POSE_ENVELOPE_ACTION_FILTER_HASH_DOMAIN_V1,
        )

    def __call__(self, action: RobotAction) -> RobotAction:
        if not isinstance(action, RobotAction):
            raise TypeError("action must be RobotAction")
        if not action.arm_pose_by_side:
            raise TCPPoseEnvelopeLifecycleError(
                "TCP envelope filter requires arm pose targets"
            )
        for side in action.arm_pose_by_side:
            if action.arm_controlled_frame(side) != TCP_CONTROLLED_FRAME_ID:
                raise TCPPoseEnvelopeLifecycleError(
                    f"TCP envelope requires tool_tcp for {side!r}"
                )
            expected_reference_frame = self.runtime.reference_frame_by_side.get(side)
            if (
                expected_reference_frame is not None
                and action.arm_frame(side) != expected_reference_frame
            ):
                raise TCPPoseEnvelopeLifecycleError(
                    "TCP envelope action reference-frame mismatch for "
                    f"{side!r}: expected {expected_reference_frame!r}, "
                    f"got {action.arm_frame(side)!r}"
                )
        raw: dict[str, np.ndarray] = {}
        for side, values in action.arm_pose_by_side.items():
            vector = np.asarray(values, dtype=np.float64).reshape(-1)
            if vector.shape != (7,) or not np.all(np.isfinite(vector)):
                raise TCPPoseEnvelopeLifecycleError(
                    f"TCP envelope requires a finite xyz+quaternion for {side!r}"
                )
            quaternion_norm = float(np.linalg.norm(vector[3:]))
            if quaternion_norm <= 1e-12:
                raise TCPPoseEnvelopeLifecycleError(
                    f"TCP envelope quaternion norm is zero for {side!r}"
                )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = Rotation.from_quat(
                vector[3:] / quaternion_norm
            ).as_matrix()
            pose[:3, 3] = vector[:3]
            raw[side] = pose
        result = self.runtime.project(raw)
        effective = {
            side: np.concatenate(
                (
                    pose[:3, 3],
                    Rotation.from_matrix(
                        np.array(pose[:3, :3], dtype=float, copy=True)
                    ).as_quat(),
                )
            ).astype(np.float32)
            for side, pose in result.effective_tcp_pose_by_side.items()
        }
        return RobotAction(
            action_space=action.action_space,
            timestamp=action.timestamp,
            arm_pose_by_side=effective,
            arm_qpos_by_side=action.arm_qpos_by_side,
            hand_qpos_by_side=action.hand_qpos_by_side,
            end_effector_targets_by_side=action.end_effector_targets_by_side,
            active_effectors=action.active_effectors,
            frame_by_effector=action.frame_by_effector,
            controlled_frame_by_effector=action.controlled_frame_by_effector,
            source=action.source,
            metadata=action.metadata,
        )


__all__ = [
    "DEPLOYMENT_SAFETY_ENVELOPE_SCHEMA_V1",
    "RobotActionTCPPoseEnvelopeFilter",
    "TCPPoseEnvelopeActivationReceipt",
    "TCPPoseEnvelopeCatalogEntry",
    "TCPPoseEnvelopeLifecycleError",
    "TCPPoseEnvelopeRuntimeSnapshot",
    "TCPPoseEnvelopeSafetyRuntime",
    "TCPPoseEnvelopeSelection",
]
