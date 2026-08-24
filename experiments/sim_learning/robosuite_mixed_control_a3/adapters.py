"""Experiment-local callable policy adapters for the robosuite A3 seam.

The A2 runner already owns source/revision lineage and component composition.
This module only turns a model-shaped callable into that existing source
protocol.  It deliberately accepts a small, model-neutral output surface
(``action``, ``action_chunk``, or an object exposing ``vector``) so tests can
stand in for either a normal policy or an RTC implementation without importing
deployment, teleop, or hardware packages.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from experiments.sim_learning.streaming_intervention.a2 import (
    A2_ACTION_DIMENSION,
    A2_COMPONENT_IDS,
    A2_POLICY_ACTION_ROLE_INTERVENTION,
    A2_POLICY_ACTION_ROLE_NOMINAL,
    A2_POLICY_SOURCE_KIND_RTC,
    A2_POLICY_SOURCE_KIND_STANDARD,
    PolicyInferenceRequestA2,
    PolicyInferenceSourceA2,
    PolicyProposalA2,
)

A3_POLICY_ADAPTER_SCHEMA = "robosuite_mixed_control_policy_adapter_a3"
A3_POLICY_ADAPTER_HASH_DOMAIN = "dexteleop.robosuite_mixed_control.policy_adapter.a3"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _action_vector(value: Iterable[float], *, name: str) -> tuple[float, ...]:
    try:
        normalized = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if len(normalized) != A2_ACTION_DIMENSION:
        raise ValueError(
            f"{name} must have dimension {A2_ACTION_DIMENSION}, got {len(normalized)}."
        )
    if any(not math.isfinite(item) for item in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


def _is_scalar(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


def _extract_action(value: Any, *, name: str, depth: int = 0) -> tuple[float, ...]:
    """Extract one 7-D action from common policy/RTC result shapes.

    The adapter does not interpret normalization, pose frames, chunks, or
    discrete outputs.  It only selects the first row of an already-decoded
    ``action_chunk`` and validates the fixed A1 action dimension.  A model
    family that needs decoding belongs behind a deployment-owned adapter.
    """

    if depth > 8:
        raise TypeError(f"{name} is nested too deeply to contain one action.")
    if isinstance(value, Mapping):
        for key in ("action", "action_vector", "a_policy", "a_rtc", "action_chunk"):
            if key in value:
                return _extract_action(
                    value[key], name=f"{name}.{key}", depth=depth + 1
                )
        raise TypeError(
            f"{name} mapping must contain one of action/action_chunk/a_policy/a_rtc."
        )

    vector = getattr(value, "vector", None)
    if vector is not None:
        return _extract_action(vector, name=f"{name}.vector", depth=depth + 1)

    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes, bytearray)):
        return _extract_action(tolist(), name=f"{name}.tolist", depth=depth + 1)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError(f"{name} must not be empty.")
        if all(_is_scalar(item) for item in value):
            return _action_vector(value, name=name)
        return _extract_action(value[0], name=f"{name}[0]", depth=depth + 1)

    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a 7-D vector, action mapping, or action chunk."
        ) from exc
    return _extract_action(values, name=name, depth=depth + 1)


@runtime_checkable
class StandardPolicyCallableA3(Protocol):
    """Optional typed surface for a normal policy provider."""

    def predict(self, observation: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class RtcPolicyCallableA3(Protocol):
    """Optional typed surface for an RTC provider.

    RTC receives the complete A2 request so it can condition on the nominal
    action while preserving the same observation/revision lineage.
    """

    def infer(self, request: PolicyInferenceRequestA2) -> Any: ...


class CallablePolicySourceA3(PolicyInferenceSourceA2):
    """Adapt one callable to the A2 standard-policy or RTC source protocol.

    Standard providers expose ``predict(observation)`` (or are directly
    callable).  RTC providers expose ``infer(request)`` (or are directly
    callable).  The callable is never allowed to choose a frame, source
    revision, component ownership, or safety envelope; those remain A2 runner
    responsibilities.
    """

    def __init__(
        self,
        *,
        source_id: str,
        source_kind: str,
        provider: object,
        revision: int = 0,
        component_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        source_id = _required(source_id, name="source_id")
        source_kind = _required(source_kind, name="source_kind")
        if source_kind not in {
            A2_POLICY_SOURCE_KIND_STANDARD,
            A2_POLICY_SOURCE_KIND_RTC,
        }:
            raise ValueError(f"Unsupported policy source kind {source_kind!r}.")
        if not callable(provider) and not any(
            callable(getattr(provider, name, None)) for name in ("predict", "infer")
        ):
            raise TypeError("provider must be callable or expose predict()/infer().")
        revision = _nonnegative_int(revision, name="revision")
        if component_ids is None:
            normalized_components = (
                A2_COMPONENT_IDS
                if source_kind == A2_POLICY_SOURCE_KIND_STANDARD
                else ("arm",)
            )
        else:
            normalized_components = tuple(str(value).strip() for value in component_ids)
            if not normalized_components or any(
                not value for value in normalized_components
            ):
                raise ValueError("component_ids must contain non-empty values.")
            if len(set(normalized_components)) != len(normalized_components):
                raise ValueError("component_ids must be unique.")
            if set(normalized_components) - set(A2_COMPONENT_IDS):
                raise ValueError("component_ids contains an unsupported component.")
            normalized_components = tuple(
                component
                for component in A2_COMPONENT_IDS
                if component in normalized_components
            )
        normalized_metadata = {
            "schema": A3_POLICY_ADAPTER_SCHEMA,
            "source_id": source_id,
            "source_kind": source_kind,
            "revision": revision,
            "component_ids": normalized_components,
        }
        if metadata is not None:
            normalized_metadata["provider_metadata"] = dict(metadata)

        self._source_id = source_id
        self._source_kind = source_kind
        self._provider = provider
        self._revision = revision
        self._component_ids = normalized_components
        self._metadata = MappingProxyType(normalized_metadata)
        self._semantic_content_hash = canonical_content_hash(
            dict(self._metadata),
            domain=A3_POLICY_ADAPTER_HASH_DOMAIN,
        )

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_kind(self) -> str:
        return self._source_kind

    @property
    def action_role(self) -> str:
        return (
            A2_POLICY_ACTION_ROLE_NOMINAL
            if self._source_kind == A2_POLICY_SOURCE_KIND_STANDARD
            else A2_POLICY_ACTION_ROLE_INTERVENTION
        )

    @property
    def action_dimension(self) -> int:
        return A2_ACTION_DIMENSION

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self._component_ids

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def semantic_content_hash(self) -> str:
        return self._semantic_content_hash

    def infer(self, request: PolicyInferenceRequestA2) -> PolicyProposalA2:
        if not isinstance(request, PolicyInferenceRequestA2):
            raise TypeError("request must be a PolicyInferenceRequestA2.")
        if request.expected_source_revision is not None and (
            request.expected_source_revision != self.revision
        ):
            raise RuntimeError(
                "policy source revision changed before inference; refusing stale input."
            )
        if self.source_kind == A2_POLICY_SOURCE_KIND_STANDARD:
            raw_output = self._invoke_standard(request)
        else:
            if request.nominal_action is None:
                raise ValueError("RTC inference requires a nominal_action reference.")
            raw_output = self._invoke_rtc(request)
        action = _extract_action(raw_output, name="policy_output")
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
            action=action,
            component_ids=self.component_ids,
        )

    def reset(self) -> None:
        reset = getattr(self._provider, "reset", None)
        if callable(reset):
            reset()

    def _invoke_standard(self, request: PolicyInferenceRequestA2) -> Any:
        predict = getattr(self._provider, "predict", None)
        if callable(predict):
            return predict(request.observation)
        if callable(self._provider):
            return self._provider(request.observation)  # type: ignore[operator]
        raise TypeError("standard provider must expose predict() or be callable.")

    def _invoke_rtc(self, request: PolicyInferenceRequestA2) -> Any:
        infer = getattr(self._provider, "infer", None)
        if callable(infer):
            return infer(request)
        if callable(self._provider):
            return self._provider(request)  # type: ignore[operator]
        raise TypeError("RTC provider must expose infer() or be callable.")


def make_standard_policy_source_a3(
    *,
    source_id: str,
    provider: object,
    revision: int = 0,
    component_ids: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CallablePolicySourceA3:
    """Build a standard nominal-policy source without importing deployment."""

    return CallablePolicySourceA3(
        source_id=source_id,
        source_kind=A2_POLICY_SOURCE_KIND_STANDARD,
        provider=provider,
        revision=revision,
        component_ids=component_ids,
        metadata=metadata,
    )


def make_rtc_policy_source_a3(
    *,
    source_id: str,
    provider: object,
    revision: int = 0,
    component_ids: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CallablePolicySourceA3:
    """Build an RTC intervention source without importing deployment."""

    return CallablePolicySourceA3(
        source_id=source_id,
        source_kind=A2_POLICY_SOURCE_KIND_RTC,
        provider=provider,
        revision=revision,
        component_ids=component_ids,
        metadata=metadata,
    )


__all__ = [
    "A3_POLICY_ADAPTER_HASH_DOMAIN",
    "A3_POLICY_ADAPTER_SCHEMA",
    "CallablePolicySourceA3",
    "RtcPolicyCallableA3",
    "StandardPolicyCallableA3",
    "make_rtc_policy_source_a3",
    "make_standard_policy_source_a3",
]
