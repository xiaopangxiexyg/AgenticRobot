"""Offline deterministic runner for the robosuite mixed-control A0 seam."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from dataset_contracts.canonical_encoding import canonical_content_hash
from sim.robosuite import (
    MixedControlActionSelection,
    RobosuiteBackendStep,
    RobosuiteEpisodeStartArtifactA1,
    RobosuiteMixedControlAdapter,
    RobosuiteResetManifest,
    RobosuiteResetReceipt,
)

STREAMING_INTERVENTION_FRAME_HASH_DOMAIN_A0 = (
    "dexteleop.streaming_intervention.frame.a0"
)
STREAMING_INTERVENTION_STEP_HASH_DOMAIN_A0 = "dexteleop.streaming_intervention.step.a0"
STREAMING_INTERVENTION_RUN_HASH_DOMAIN_A0 = "dexteleop.streaming_intervention.run.a0"
MOCK_OBSERVATION_KEYS_A0 = ("state", "step_index", "episode_seed")


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _vector(
    values: tuple[float, ...] | list[float],
    *,
    name: str,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


@dataclass(frozen=True)
class StreamingInterventionFrame:
    """One ordered input frame for the mock streaming loop."""

    frame_id: str
    mode: str
    base_action: tuple[float, ...]
    intervention_action: tuple[float, ...] | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = _required(self.frame_id, name="frame_id")
        mode = _required(self.mode, name="mode")
        base_action = _vector(self.base_action, name="base_action")
        intervention_action = (
            None
            if self.intervention_action is None
            else _vector(self.intervention_action, name="intervention_action")
        )
        payload = {
            "frame_id": frame_id,
            "mode": mode,
            "base_action": base_action,
            "intervention_action": intervention_action,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=STREAMING_INTERVENTION_FRAME_HASH_DOMAIN_A0,
            ),
        )


class DeterministicRobosuiteMockBackend:
    """Small state machine proving adapter behavior without robosuite or devices."""

    backend_id = "robosuite-a0-deterministic-mock"

    def __init__(self) -> None:
        self._manifest: RobosuiteResetManifest | None = None
        self._state: tuple[float, ...] = ()
        self._step_index = 0
        self._closed = False

    def reset(self, manifest: RobosuiteResetManifest) -> Mapping[str, object]:
        if tuple(manifest.observation_keys) != MOCK_OBSERVATION_KEYS_A0:
            raise ValueError(
                "mock backend requires the declared A0 mock observation keys."
            )
        seed_material = f"{manifest.semantic_content_hash}:{manifest.seed}".encode()
        self._state = tuple(
            (
                (
                    hashlib.sha256(
                        seed_material + index.to_bytes(8, byteorder="big")
                    ).digest()[0]
                    / 255.0
                )
                * 2.0
            )
            - 1.0
            for index in range(manifest.action_dimension)
        )
        self._manifest = manifest
        self._step_index = 0
        self._closed = False
        return self._observation()

    def step(self, action: tuple[float, ...]) -> RobosuiteBackendStep:
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("mock backend must be reset before stepping.")
        normalized = _vector(action, name="action")
        if len(normalized) != manifest.action_dimension:
            raise ValueError("mock action dimension does not match the manifest.")
        self._state = tuple(
            min(max(state + (0.1 * delta), -1.0), 1.0)
            for state, delta in zip(self._state, normalized, strict=True)
        )
        self._step_index += 1
        truncated = self._step_index >= manifest.max_episode_steps
        return RobosuiteBackendStep(
            observation=self._observation(),
            reward=-sum(abs(value) for value in self._state),
            terminated=False,
            truncated=truncated,
            info={
                "backend_kind": "deterministic_mock",
                "manifest_hash": manifest.semantic_content_hash,
            },
        )

    def close(self) -> None:
        self._manifest = None
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def _observation(self) -> Mapping[str, object]:
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("mock backend has no active manifest.")
        return MappingProxyType(
            {
                "state": self._state,
                "step_index": self._step_index,
                "episode_seed": manifest.seed,
            }
        )


@dataclass(frozen=True)
class StreamingInterventionStep:
    """Content-addressed evidence for one consumed streaming frame."""

    frame_id: str
    frame_hash: str
    selection: MixedControlActionSelection
    backend_step: RobosuiteBackendStep
    input_frame: StreamingInterventionFrame | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = _required(self.frame_id, name="frame_id")
        frame_hash = _required(self.frame_hash, name="frame_hash")
        if not isinstance(self.selection, MixedControlActionSelection):
            raise TypeError("selection must be a MixedControlActionSelection.")
        if not isinstance(self.backend_step, RobosuiteBackendStep):
            raise TypeError("backend_step must be a RobosuiteBackendStep.")
        input_frame = self.input_frame
        if input_frame is not None:
            if not isinstance(input_frame, StreamingInterventionFrame):
                raise TypeError("input_frame must be a StreamingInterventionFrame.")
            if input_frame.frame_id != frame_id or (
                input_frame.semantic_content_hash != frame_hash
            ):
                raise ValueError("input_frame does not bind frame_id and frame_hash.")
        payload = {
            "frame_id": frame_id,
            "frame_hash": frame_hash,
            "selection_hash": self.selection.semantic_content_hash,
            "backend_step_hash": self.backend_step.semantic_content_hash,
        }
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "frame_hash", frame_hash)
        object.__setattr__(self, "input_frame", input_frame)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=STREAMING_INTERVENTION_STEP_HASH_DOMAIN_A0,
            ),
        )


@dataclass(frozen=True)
class StreamingInterventionRunResult:
    """Replay identity and ordered evidence for one completed run."""

    manifest: RobosuiteResetManifest
    reset_receipt: RobosuiteResetReceipt
    steps: tuple[StreamingInterventionStep, ...]
    stop_reason: str
    reset_artifact: RobosuiteEpisodeStartArtifactA1 | None = None
    source_effective_model_xml_sha256: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RobosuiteResetManifest):
            raise TypeError("manifest must be a RobosuiteResetManifest.")
        if not isinstance(self.reset_receipt, RobosuiteResetReceipt):
            raise TypeError("reset_receipt must be a RobosuiteResetReceipt.")
        if self.reset_receipt.manifest_id != self.manifest.manifest_id or (
            self.reset_receipt.manifest_hash != self.manifest.semantic_content_hash
        ):
            raise ValueError("reset receipt does not bind the supplied manifest.")
        steps = tuple(self.steps)
        if any(not isinstance(step, StreamingInterventionStep) for step in steps):
            raise TypeError("steps must contain StreamingInterventionStep values.")
        if len({step.frame_id for step in steps}) != len(steps):
            raise ValueError("run step frame_id values must be unique.")
        stop_reason = _required(self.stop_reason, name="stop_reason")
        if stop_reason not in {
            "backend_terminated",
            "input_exhausted",
            "manifest_step_limit",
        }:
            raise ValueError(f"Unsupported stop_reason {stop_reason!r}.")
        reset_artifact = self.reset_artifact
        source_effective_xml_hash = self.source_effective_model_xml_sha256
        if reset_artifact is not None:
            if not isinstance(reset_artifact, RobosuiteEpisodeStartArtifactA1):
                raise TypeError(
                    "reset_artifact must be a RobosuiteEpisodeStartArtifactA1."
                )
            if reset_artifact.manifest_hash != self.manifest.semantic_content_hash:
                raise ValueError("reset artifact does not bind the supplied manifest.")
            source_effective_xml_hash = _required(
                source_effective_xml_hash,
                name="source_effective_model_xml_sha256",
            )
            if source_effective_xml_hash != (reset_artifact.effective_model_xml_sha256):
                raise ValueError(
                    "source effective XML hash does not bind the reset artifact."
                )
        elif source_effective_xml_hash is not None:
            raise ValueError(
                "source effective XML hash requires an episode-start artifact."
            )
        payload = {
            "manifest_hash": self.manifest.semantic_content_hash,
            "reset_receipt_hash": self.reset_receipt.semantic_content_hash,
            "step_hashes": tuple(step.semantic_content_hash for step in steps),
            "stop_reason": stop_reason,
            "reset_artifact_hash": (
                None if reset_artifact is None else reset_artifact.semantic_content_hash
            ),
            "source_effective_model_xml_sha256": source_effective_xml_hash,
        }
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "stop_reason", stop_reason)
        object.__setattr__(self, "reset_artifact", reset_artifact)
        object.__setattr__(
            self,
            "source_effective_model_xml_sha256",
            source_effective_xml_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=STREAMING_INTERVENTION_RUN_HASH_DOMAIN_A0,
            ),
        )

    @property
    def run_id(self) -> str:
        return f"streaming-intervention-run:{self.semantic_content_hash}"


class StreamingInterventionMockRunner:
    """Consume ordered frames through the adapter and deterministic mock."""

    def __init__(self, adapter: RobosuiteMixedControlAdapter) -> None:
        if not isinstance(adapter, RobosuiteMixedControlAdapter):
            raise TypeError("adapter must be a RobosuiteMixedControlAdapter.")
        self._adapter = adapter

    def run(
        self,
        *,
        manifest: RobosuiteResetManifest,
        frames: Iterable[StreamingInterventionFrame],
    ) -> StreamingInterventionRunResult:
        from .runner import StreamingInterventionRunner

        return StreamingInterventionRunner(
            adapter=self._adapter,
            backend_factory=DeterministicRobosuiteMockBackend,
        ).run(manifest=manifest, frames=frames)


__all__ = [
    "MOCK_OBSERVATION_KEYS_A0",
    "DeterministicRobosuiteMockBackend",
    "StreamingInterventionFrame",
    "StreamingInterventionMockRunner",
    "StreamingInterventionRunResult",
    "StreamingInterventionStep",
]
