"""Reusable RTC v2 backend for existing final-chunk model interfaces.

Framework adapters only prepare model input, invoke their existing blocking
inference function and return ``FinalChunkModelOutput``.  This module owns the
repeated capability/request construction, input-hash check, output shape/type
validation, local timing and response-envelope construction.  It still owns no
worker, timed plan, safety, target or dispatch authority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy.rtc_protocol import (
    RTC_CONTINUATION_NONE,
    RTC_PREFIX_FORBIDDEN,
    RTC_PROTOCOL_V2,
    RTC_SCHEDULER_ASYNC_TIMED_PLAN,
    RtcContinuationCapability,
    RtcInferenceEnvelopeV2,
    RtcModelCapability,
    RtcModelResponseEnvelopeV2,
    RtcProtocolMode,
    RtcProtocolRequestSpec,
)


FINAL_CHUNK_BINDING_HASH_DOMAIN_V1 = "dexteleop.final_chunk_rtc_binding.v1"
FINAL_CHUNK_OUTPUT_ID_HASH_DOMAIN_V1 = "dexteleop.final_chunk_rtc_output.v1"


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0 or normalized != value:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class FinalChunkRtcBinding:
    """Exact instance-derived identity for a legacy final-chunk interface."""

    implementation_id: str
    implementation_parameters_hash: str
    provider_id: str
    artifact_hash: str
    model_action_space: str
    model_action_dim: int
    model_action_discrete_dim: int
    action_pose_mode: str
    component_scope: tuple[str, ...]
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    prediction_horizon_steps: int
    minimum_execution_horizon_steps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "implementation_id",
            "implementation_parameters_hash",
            "provider_id",
            "artifact_hash",
            "model_action_space",
            "action_pose_mode",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        scope = tuple(
            _nonempty(value, name="component_scope")
            for value in self.component_scope
        )
        if not scope or len(set(scope)) != len(scope):
            raise ValueError("component_scope must be non-empty and unique.")
        object.__setattr__(self, "component_scope", scope)
        object.__setattr__(
            self,
            "model_action_dim",
            _positive_int(self.model_action_dim, name="model_action_dim"),
        )
        object.__setattr__(
            self,
            "model_action_discrete_dim",
            _nonnegative_int(
                self.model_action_discrete_dim,
                name="model_action_discrete_dim",
            ),
        )
        horizon = _positive_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
        )
        minimum = _positive_int(
            self.minimum_execution_horizon_steps,
            name="minimum_execution_horizon_steps",
        )
        if minimum > horizon:
            raise ValueError(
                "minimum_execution_horizon_steps exceeds prediction horizon."
            )
        object.__setattr__(self, "prediction_horizon_steps", horizon)
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "implementation_id": self.implementation_id,
                "implementation_parameters_hash": (
                    self.implementation_parameters_hash
                ),
                "provider_id": self.provider_id,
                "artifact_hash": self.artifact_hash,
                "model_action_space": self.model_action_space,
                "model_action_dim": self.model_action_dim,
                "model_action_discrete_dim": self.model_action_discrete_dim,
                "action_pose_mode": self.action_pose_mode,
                "component_scope": self.component_scope,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
                "prediction_horizon_steps": self.prediction_horizon_steps,
                "minimum_execution_horizon_steps": (
                    self.minimum_execution_horizon_steps
                ),
            },
            domain=FINAL_CHUNK_BINDING_HASH_DOMAIN_V1,
        )

    def build_capability(self) -> RtcModelCapability:
        continuation = RtcContinuationCapability(
            continuation_mode=RTC_CONTINUATION_NONE,
            implementation_id=self.implementation_id,
            implementation_parameters_hash=self.implementation_parameters_hash,
            model_action_space=self.model_action_space,
            model_action_dim=self.model_action_dim,
            model_action_discrete_dim=self.model_action_discrete_dim,
            supported_action_pose_modes=(self.action_pose_mode,),
            component_scope=self.component_scope,
            action_codec_hash=self.action_codec_hash,
            frame_contract_hash=self.frame_contract_hash,
            layout_hash=self.layout_hash,
            normalization_hash=self.normalization_hash,
            minimum_prediction_horizon_steps=self.prediction_horizon_steps,
            maximum_prediction_horizon_steps=self.prediction_horizon_steps,
            minimum_execution_horizon_steps=self.minimum_execution_horizon_steps,
            maximum_execution_horizon_steps=self.prediction_horizon_steps,
            prefix_requirement=RTC_PREFIX_FORBIDDEN,
            supports_prefix_mask=False,
            supports_conditioning_receipt=False,
        )
        return RtcModelCapability(
            capability_id=(
                f"{self.implementation_id}:{self.semantic_content_hash[:16]}"
            ),
            provider_id=self.provider_id,
            artifact_hash=self.artifact_hash,
            scheduler_modes=(RTC_SCHEDULER_ASYNC_TIMED_PLAN,),
            continuations=(continuation,),
        )

    def build_request_spec(self) -> RtcProtocolRequestSpec:
        return RtcProtocolRequestSpec(
            mode=RtcProtocolMode(
                RTC_SCHEDULER_ASYNC_TIMED_PLAN,
                RTC_CONTINUATION_NONE,
            ),
            artifact_hash=self.artifact_hash,
            model_action_space=self.model_action_space,
            model_action_dim=self.model_action_dim,
            model_action_discrete_dim=self.model_action_discrete_dim,
            action_pose_mode=self.action_pose_mode,
            component_scope=self.component_scope,
            action_codec_hash=self.action_codec_hash,
            frame_contract_hash=self.frame_contract_hash,
            layout_hash=self.layout_hash,
            normalization_hash=self.normalization_hash,
            prediction_horizon_steps=self.prediction_horizon_steps,
            minimum_execution_horizon_steps=self.minimum_execution_horizon_steps,
        )


@dataclass(frozen=True)
class FinalChunkModelOutput:
    """Framework-neutral final model rows before canonical action decoding."""

    actions: np.ndarray
    discrete_actions: np.ndarray | None = None
    source_output_id: str | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if not np.issubdtype(actions.dtype, np.floating):
            raise TypeError("final chunk actions must use a floating dtype.")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or np.any(~np.isfinite(actions)):
            raise ValueError("final chunk actions must be finite rank-2 matrix.")
        actions = np.ascontiguousarray(actions).copy()
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)

        discrete = self.discrete_actions
        if discrete is not None:
            raw = np.asarray(discrete)
            if raw.dtype.kind not in {"i", "u"}:
                raise TypeError("final chunk discrete actions must use an integer dtype.")
            discrete_array = np.asarray(raw, dtype=np.int64)
            if discrete_array.ndim != 2:
                raise ValueError("final chunk discrete actions must be rank 2.")
            discrete_array = np.ascontiguousarray(discrete_array).copy()
            discrete_array.setflags(write=False)
            object.__setattr__(self, "discrete_actions", discrete_array)
        if self.source_output_id is not None:
            object.__setattr__(
                self,
                "source_output_id",
                _nonempty(self.source_output_id, name="source_output_id"),
            )


class FinalChunkRtcBackend:
    """Turn any blocking final-chunk callable into a validated RTC backend."""

    def __init__(
        self,
        *,
        binding: FinalChunkRtcBinding,
        infer_model: Callable[[Any], FinalChunkModelOutput],
        hash_model_input: Callable[[Any], str],
        request_cancel: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(binding, FinalChunkRtcBinding):
            raise TypeError("binding must be FinalChunkRtcBinding.")
        if not callable(infer_model):
            raise TypeError("infer_model must be callable.")
        if not callable(hash_model_input):
            raise TypeError("hash_model_input must be callable.")
        if request_cancel is not None and not callable(request_cancel):
            raise TypeError("request_cancel must be callable or None.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.binding = binding
        self._infer_model = infer_model
        self._hash_model_input = hash_model_input
        self._request_cancel = request_cancel
        self._clock = clock
        self._capability = binding.build_capability()
        self._request_spec = binding.build_request_spec()

    @property
    def rtc_capability(self) -> RtcModelCapability:
        return self._capability

    @property
    def request_spec(self) -> RtcProtocolRequestSpec:
        return self._request_spec

    def infer_rtc(
        self,
        envelope: RtcInferenceEnvelopeV2,
        model_input: Any,
    ) -> RtcModelResponseEnvelopeV2:
        model_input_hash = _nonempty(
            self._hash_model_input(model_input),
            name="model_input_hash",
        )
        if model_input_hash != envelope.model_input_hash:
            raise ValueError("Prepared model input hash does not match RTC envelope.")
        started = float(self._clock())
        output = self._infer_model(model_input)
        completed = float(self._clock())
        received = float(self._clock())
        if completed < started or received < completed:
            raise ValueError("Final-chunk RTC adapter clock regressed.")
        if not isinstance(output, FinalChunkModelOutput):
            raise TypeError("infer_model must return FinalChunkModelOutput.")
        expected_continuous = (
            self.binding.prediction_horizon_steps,
            self.binding.model_action_dim,
        )
        if output.actions.shape != expected_continuous:
            raise ValueError(
                f"final chunk actions must have shape {expected_continuous}, "
                f"got {output.actions.shape}."
            )
        expected_discrete = (
            self.binding.prediction_horizon_steps,
            self.binding.model_action_discrete_dim,
        )
        if self.binding.model_action_discrete_dim == 0:
            if output.discrete_actions is not None:
                raise ValueError(
                    "final chunk discrete actions were returned for a continuous-only binding."
                )
        elif output.discrete_actions is None:
            raise ValueError("final chunk is missing required discrete actions.")
        elif output.discrete_actions.shape != expected_discrete:
            raise ValueError(
                "final chunk discrete actions must have shape "
                f"{expected_discrete}, got {output.discrete_actions.shape}."
            )
        model_output_id = canonical_content_hash(
            {
                "envelope_hash": envelope.semantic_content_hash,
                "binding_hash": self.binding.semantic_content_hash,
                "actions": output.actions,
                "discrete_actions": output.discrete_actions,
                "source_output_id": output.source_output_id,
            },
            domain=FINAL_CHUNK_OUTPUT_ID_HASH_DOMAIN_V1,
        )
        return RtcModelResponseEnvelopeV2(
            envelope_hash=envelope.semantic_content_hash,
            request_id=envelope.request.request_id,
            request_hash=envelope.request.semantic_content_hash,
            model_input_hash=envelope.model_input_hash,
            capability_hash=self._capability.semantic_content_hash,
            selected_continuation_hash=envelope.selected_continuation_hash,
            protocol_version=RTC_PROTOCOL_V2,
            scheduler_mode=RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            continuation_mode=RTC_CONTINUATION_NONE,
            artifact_hash=self.binding.artifact_hash,
            action_codec_hash=self.binding.action_codec_hash,
            frame_contract_hash=self.binding.frame_contract_hash,
            layout_hash=self.binding.layout_hash,
            normalization_hash=self.binding.normalization_hash,
            model_output_id=model_output_id,
            model_space_actions=output.actions,
            model_space_discrete_actions=output.discrete_actions,
            effective_delay_steps=envelope.request.forecast_delay_steps,
            execution_horizon_steps=self.binding.prediction_horizon_steps,
            inference_started_at=started,
            inference_completed_at=completed,
            result_received_at=received,
            conditioning_receipt=None,
        )

    def request_cancel(self) -> bool:
        if self._request_cancel is None:
            return False
        accepted = self._request_cancel()
        if not isinstance(accepted, bool):
            raise TypeError("final-chunk request_cancel() must return bool.")
        return accepted


__all__ = [
    "FinalChunkModelOutput",
    "FinalChunkRtcBackend",
    "FinalChunkRtcBinding",
]
