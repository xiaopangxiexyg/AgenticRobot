"""One-step composition for sessionless raw-RTC final-chunk models.

This module removes framework boilerplate without weakening the existing
identity boundary.  A model package still owns input preparation, its blocking
model call and model-space decoding; the shared adapter derives capability and
request spec from one ``FinalChunkRtcBinding`` and never owns a worker, timed
plan, safety, target or dispatch authority.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from robot_motion.deploy.action_representation import (
    PolicyActionRepresentationContext,
)
from robot_motion.deploy.observation_view import PolicyObservationView

from .rtc import RtcModelSpaceDecoder
from .rtc_executor import RtcModelInputPreparer
from .rtc_final_chunk import (
    FinalChunkModelOutput,
    FinalChunkRtcBackend,
    FinalChunkRtcBinding,
)
from .rtc_frontend import SimpleRtcModelRuntimeAdapter


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class MappingFinalChunkOutputAdapter:
    """Extract typed final-chunk rows from a mapping response.

    Additional response fields are intentionally ignored so a server may keep
    framework timing or diagnostics beside its action payload.  Every selected
    key is exact and required; output dimensions and dtypes remain enforced by
    ``FinalChunkRtcBackend`` against the instance binding.
    """

    actions_key: str = "actions"
    discrete_actions_key: str | None = None
    source_output_id_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actions_key",
            _nonempty(self.actions_key, name="actions_key"),
        )
        for name in ("discrete_actions_key", "source_output_id_key"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name=name))
        selected = tuple(
            value
            for value in (
                self.actions_key,
                self.discrete_actions_key,
                self.source_output_id_key,
            )
            if value is not None
        )
        if len(set(selected)) != len(selected):
            raise ValueError("final-chunk response keys must be distinct.")

    def __call__(self, response: Any) -> FinalChunkModelOutput:
        if not isinstance(response, Mapping):
            raise TypeError("final-chunk model response must be a mapping.")
        if self.actions_key not in response:
            raise ValueError(
                "final-chunk model response is missing exact output key "
                f"{self.actions_key!r} for continuous actions."
            )
        discrete_actions = None
        if self.discrete_actions_key is not None:
            if self.discrete_actions_key not in response:
                raise ValueError(
                    "final-chunk model response is missing exact discrete actions "
                    f"key {self.discrete_actions_key!r}."
                )
            discrete_actions = response[self.discrete_actions_key]
        source_output_id = None
        if self.source_output_id_key is not None:
            if self.source_output_id_key not in response:
                raise ValueError(
                    "final-chunk model response is missing exact source output id "
                    f"key {self.source_output_id_key!r}."
                )
            raw_source_output_id = response[self.source_output_id_key]
            if raw_source_output_id is not None and not isinstance(
                raw_source_output_id, str
            ):
                raise TypeError("source output id must be a string or None.")
            source_output_id = raw_source_output_id
        return FinalChunkModelOutput(
            actions=response[self.actions_key],
            discrete_actions=discrete_actions,
            source_output_id=source_output_id,
        )


def create_final_chunk_rtc_runtime_adapter(
    *,
    binding: FinalChunkRtcBinding,
    prepare_input: RtcModelInputPreparer,
    call_model: Callable[[Any], Any],
    hash_model_input: Callable[[Any], str],
    decoder: RtcModelSpaceDecoder,
    adapt_output: Callable[[Any], FinalChunkModelOutput] | None = None,
    request_cancel: Callable[[], bool] | None = None,
    representation_context_builder: (
        Callable[
            [PolicyObservationView, str],
            PolicyActionRepresentationContext,
        ]
        | None
    ) = None,
    clock: Callable[[], float] = time.monotonic,
) -> SimpleRtcModelRuntimeAdapter:
    """Compose a sessionless final-chunk model into the shared RTC frontend.

    ``adapt_output`` may be omitted only when ``call_model`` already returns a
    ``FinalChunkModelOutput``.  Frameworks returning dictionaries can use
    ``MappingFinalChunkOutputAdapter``.  The request spec is always derived
    from ``binding`` through the created backend, so integrations cannot copy
    or drift a second set of artifact/frame/layout/horizon facts.
    """

    if not isinstance(binding, FinalChunkRtcBinding):
        raise TypeError("binding must be FinalChunkRtcBinding.")
    if not callable(prepare_input):
        raise TypeError("prepare_input must be callable.")
    if not callable(call_model):
        raise TypeError("call_model must be callable.")
    if not callable(hash_model_input):
        raise TypeError("hash_model_input must be callable.")
    if not callable(decoder):
        raise TypeError("decoder must be callable.")
    if adapt_output is not None and not callable(adapt_output):
        raise TypeError("adapt_output must be callable or None.")
    if request_cancel is not None and not callable(request_cancel):
        raise TypeError("request_cancel must be callable or None.")
    if representation_context_builder is not None and not callable(
        representation_context_builder
    ):
        raise TypeError("representation_context_builder must be callable or None.")

    if isinstance(adapt_output, MappingFinalChunkOutputAdapter):
        has_discrete_key = adapt_output.discrete_actions_key is not None
        expects_discrete = binding.model_action_discrete_dim > 0
        if has_discrete_key != expects_discrete:
            raise ValueError(
                "mapping discrete output key must be present exactly when the "
                "binding declares a non-zero discrete action dimension."
            )

    def infer_model(model_input: Any) -> FinalChunkModelOutput:
        raw_output = call_model(model_input)
        output = raw_output if adapt_output is None else adapt_output(raw_output)
        if not isinstance(output, FinalChunkModelOutput):
            raise TypeError(
                "call_model/adapt_output must produce FinalChunkModelOutput."
            )
        return output

    backend = FinalChunkRtcBackend(
        binding=binding,
        infer_model=infer_model,
        hash_model_input=hash_model_input,
        request_cancel=request_cancel,
        clock=clock,
    )
    return SimpleRtcModelRuntimeAdapter(
        backend=backend,
        decoder=decoder,
        request_spec=backend.request_spec,
        prepare_input=prepare_input,
        representation_context_builder=representation_context_builder,
    )


__all__ = [
    "MappingFinalChunkOutputAdapter",
    "create_final_chunk_rtc_runtime_adapter",
]
