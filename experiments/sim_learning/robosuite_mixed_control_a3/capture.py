"""Experiment-local observation trace and row materialization for A3.

The capture runner wraps the already reviewed A2 runner with a passive backend
observer.  It records the reset observation and every post-step observation,
then materializes one content-addressed learning row per executed frame.  The
result is intentionally an offline experiment object, not a canonical dataset
writer or a production episode owner.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from experiments.sim_learning.streaming_intervention.a2 import (
    A2_ACTION_DIMENSION,
    A2_COMPONENT_IDS,
    A2_POLICY_REQUEST_HASH_DOMAIN,
    PolicyInferenceSourceA2,
    StreamingInterventionControlFrameA2,
)
from experiments.sim_learning.streaming_intervention.a2_runner import (
    StreamingInterventionA2RunResult,
    StreamingInterventionA2Step,
    StreamingInterventionPolicyRunnerA2,
)
from sim.robosuite import (
    RobosuiteA0Backend,
    RobosuiteBackendStep,
    RobosuiteResetManifest,
)
from sim.robosuite.adapter import (
    ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
)

from .contracts import (
    A3_COLLECTION_MODE_ORDINARY_TELEOP,
    A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
    A3_COLLECTION_MODE_TAKEOVER_TELEOP,
    A3_COLLECTION_MODES,
    A3_DATASET_HASH_DOMAIN,
    A3_OBSERVATION_HASH_DOMAIN,
    A3_POLICY_VARIANT_NONE,
    A3_POLICY_VARIANT_RTC,
    A3_POLICY_VARIANT_STANDARD,
    A3_POLICY_VARIANTS,
    A3_ROW_HASH_DOMAIN,
    A3CollectionProfile,
    policy_variant,
    resolve_collection_profile,
    validate_collection_frames,
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _vector(
    values: Iterable[float] | None,
    *,
    name: str,
    optional: bool = False,
) -> tuple[float, ...] | None:
    if values is None:
        if optional:
            return None
        raise ValueError(f"{name} must be provided.")
    normalized = tuple(float(value) for value in values)
    if len(normalized) != A2_ACTION_DIMENSION:
        raise ValueError(
            f"{name} must have dimension {A2_ACTION_DIMENSION}, got {len(normalized)}."
        )
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


def _components(
    values: Iterable[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty.")
    if any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values.")
    unknown = set(normalized) - set(A2_COMPONENT_IDS)
    if unknown:
        raise ValueError(f"{name} contains unsupported components: {sorted(unknown)}.")
    return tuple(component for component in A2_COMPONENT_IDS if component in normalized)


def _snapshot_observation(
    observation: Mapping[str, Any], *, name: str
) -> Mapping[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if not observation:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(dict(observation))


@dataclass(frozen=True)
class RobosuiteMixedControlRowA3:
    """One offline learning row with both input and executed-action lineage."""

    episode_id: str
    task_id: str
    source_run_hash: str
    step_index: int
    collection_mode: str
    policy_variant: str
    frame_id: str
    frame_hash: str
    mode: str
    observation: Mapping[str, Any]
    next_observation: Mapping[str, Any]
    observation_hash: str
    next_observation_hash: str
    base_action: tuple[float, ...]
    human_raw_action: tuple[float, ...] | None
    human_mapped_action: tuple[float, ...] | None
    a_policy: tuple[float, ...] | None
    a_rtc: tuple[float, ...] | None
    a_mix: tuple[float, ...]
    a_safe: tuple[float, ...]
    a_exec: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    human_component_ids: tuple[str, ...]
    active_component_ids: tuple[str, ...]
    human_event_id: str | None
    policy_source_id: str | None
    policy_source_revision: int | None
    policy_inference_id: str | None
    selection_hash: str
    ownership_hash: str
    backend_step_hash: str
    clipped_dimensions: tuple[int, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        episode_id = _required(self.episode_id, name="episode_id")
        task_id = _required(self.task_id, name="task_id")
        source_run_hash = _required(self.source_run_hash, name="source_run_hash")
        step_index = _nonnegative_int(self.step_index, name="step_index")
        collection_mode = _required(self.collection_mode, name="collection_mode")
        if collection_mode not in A3_COLLECTION_MODES:
            raise ValueError(f"Unsupported A3 collection mode {collection_mode!r}.")
        policy_variant = _required(self.policy_variant, name="policy_variant")
        if policy_variant not in A3_POLICY_VARIANTS:
            raise ValueError(f"Unsupported A3 policy variant {policy_variant!r}.")
        frame_id = _required(self.frame_id, name="frame_id")
        frame_hash = _required(self.frame_hash, name="frame_hash")
        mode = _required(self.mode, name="mode")
        if mode not in {
            ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
            ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
            ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
        }:
            raise ValueError(f"Unsupported robosuite mix mode {mode!r}.")
        observation = _snapshot_observation(self.observation, name="observation")
        next_observation = _snapshot_observation(
            self.next_observation,
            name="next_observation",
        )
        observation_hash = _required(self.observation_hash, name="observation_hash")
        next_observation_hash = _required(
            self.next_observation_hash,
            name="next_observation_hash",
        )
        base_action = _vector(self.base_action, name="base_action")
        human_raw_action = _vector(
            self.human_raw_action,
            name="human_raw_action",
            optional=True,
        )
        human_mapped_action = _vector(
            self.human_mapped_action,
            name="human_mapped_action",
            optional=True,
        )
        if (human_raw_action is None) != (human_mapped_action is None):
            raise ValueError("human actions must be paired.")
        a_policy = _vector(self.a_policy, name="a_policy", optional=True)
        a_rtc = _vector(self.a_rtc, name="a_rtc", optional=True)
        a_mix = _vector(self.a_mix, name="a_mix")
        a_safe = _vector(self.a_safe, name="a_safe")
        a_exec = _vector(self.a_exec, name="a_exec")
        if a_safe != a_exec:
            raise ValueError("A3 experiment rows require a_exec to equal a_safe.")
        if policy_variant == A3_POLICY_VARIANT_NONE:
            if a_policy is not None or a_rtc is not None:
                raise ValueError("policy actions require a policy variant.")
        elif policy_variant == A3_POLICY_VARIANT_STANDARD:
            if a_policy is None or a_rtc is not None:
                raise ValueError("standard rows require only a_policy.")
        elif a_rtc is None or a_policy is not None:
            raise ValueError("RTC rows require only a_rtc.")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite.")
        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise TypeError("terminated and truncated must be bool values.")
        human_component_ids = _components(
            self.human_component_ids,
            name="human_component_ids",
            allow_empty=True,
        )
        active_component_ids = _components(
            self.active_component_ids,
            name="active_component_ids",
            allow_empty=True,
        )
        human_event_id = (
            None
            if self.human_event_id is None
            else _required(self.human_event_id, name="human_event_id")
        )
        if human_mapped_action is None and (
            human_component_ids or human_event_id is not None
        ):
            raise ValueError("human lineage requires actions, components, and event.")
        if human_mapped_action is not None and (
            not human_component_ids or human_event_id is None
        ):
            raise ValueError("human lineage is incomplete.")
        policy_source_id = (
            None
            if self.policy_source_id is None
            else _required(self.policy_source_id, name="policy_source_id")
        )
        policy_source_revision = self.policy_source_revision
        if policy_source_revision is not None:
            policy_source_revision = _nonnegative_int(
                policy_source_revision,
                name="policy_source_revision",
            )
        policy_inference_id = (
            None
            if self.policy_inference_id is None
            else _required(self.policy_inference_id, name="policy_inference_id")
        )
        if policy_variant == A3_POLICY_VARIANT_NONE and any(
            value is not None
            for value in (
                policy_source_id,
                policy_source_revision,
                policy_inference_id,
            )
        ):
            raise ValueError("policy lineage requires a policy variant.")
        if policy_variant != A3_POLICY_VARIANT_NONE and any(
            value is None
            for value in (policy_source_id, policy_source_revision, policy_inference_id)
        ):
            raise ValueError("policy rows require complete source lineage.")
        selection_hash = _required(self.selection_hash, name="selection_hash")
        ownership_hash = _required(self.ownership_hash, name="ownership_hash")
        backend_step_hash = _required(self.backend_step_hash, name="backend_step_hash")
        clipped_dimensions = tuple(int(index) for index in self.clipped_dimensions)
        if len(clipped_dimensions) != len(set(clipped_dimensions)) or any(
            index < 0 or index >= A2_ACTION_DIMENSION for index in clipped_dimensions
        ):
            raise ValueError("clipped_dimensions contains an invalid index.")
        payload = {
            "episode_id": episode_id,
            "task_id": task_id,
            "source_run_hash": source_run_hash,
            "step_index": step_index,
            "collection_mode": collection_mode,
            "policy_variant": policy_variant,
            "frame_id": frame_id,
            "frame_hash": frame_hash,
            "mode": mode,
            "observation": dict(observation),
            "next_observation": dict(next_observation),
            "observation_hash": observation_hash,
            "next_observation_hash": next_observation_hash,
            "base_action": base_action,
            "human_raw_action": human_raw_action,
            "human_mapped_action": human_mapped_action,
            "a_policy": a_policy,
            "a_rtc": a_rtc,
            "a_mix": a_mix,
            "a_safe": a_safe,
            "a_exec": a_exec,
            "reward": reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "human_component_ids": human_component_ids,
            "active_component_ids": active_component_ids,
            "human_event_id": human_event_id,
            "policy_source_id": policy_source_id,
            "policy_source_revision": policy_source_revision,
            "policy_inference_id": policy_inference_id,
            "selection_hash": selection_hash,
            "ownership_hash": ownership_hash,
            "backend_step_hash": backend_step_hash,
            "clipped_dimensions": clipped_dimensions,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A3_ROW_HASH_DOMAIN),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a plain mapping suitable for an offline JSON/Parquet adapter."""

        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "source_run_hash": self.source_run_hash,
            "step_index": self.step_index,
            "collection_mode": self.collection_mode,
            "policy_variant": self.policy_variant,
            "frame_id": self.frame_id,
            "frame_hash": self.frame_hash,
            "mode": self.mode,
            "observation": dict(self.observation),
            "next_observation": dict(self.next_observation),
            "observation_hash": self.observation_hash,
            "next_observation_hash": self.next_observation_hash,
            "base_action": self.base_action,
            "human_raw_action": self.human_raw_action,
            "human_mapped_action": self.human_mapped_action,
            "a_policy": self.a_policy,
            "a_rtc": self.a_rtc,
            "a_mix": self.a_mix,
            "a_safe": self.a_safe,
            "a_exec": self.a_exec,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "human_component_ids": self.human_component_ids,
            "active_component_ids": self.active_component_ids,
            "human_event_id": self.human_event_id,
            "policy_source_id": self.policy_source_id,
            "policy_source_revision": self.policy_source_revision,
            "policy_inference_id": self.policy_inference_id,
            "selection_hash": self.selection_hash,
            "ownership_hash": self.ownership_hash,
            "backend_step_hash": self.backend_step_hash,
            "clipped_dimensions": self.clipped_dimensions,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class RobosuiteMixedControlDatasetA3:
    """Content-addressed rows for one three-mode experiment episode."""

    manifest_hash: str
    task_id: str
    episode_id: str
    collection_mode: str
    policy_variant: str
    reset_observation: Mapping[str, Any]
    source_run_hash: str
    rows: tuple[RobosuiteMixedControlRowA3, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        manifest_hash = _required(self.manifest_hash, name="manifest_hash")
        task_id = _required(self.task_id, name="task_id")
        episode_id = _required(self.episode_id, name="episode_id")
        collection_mode = _required(self.collection_mode, name="collection_mode")
        if collection_mode not in A3_COLLECTION_MODES:
            raise ValueError(f"Unsupported A3 collection mode {collection_mode!r}.")
        policy_variant = _required(self.policy_variant, name="policy_variant")
        if policy_variant not in A3_POLICY_VARIANTS:
            raise ValueError(f"Unsupported A3 policy variant {policy_variant!r}.")
        reset_observation = _snapshot_observation(
            self.reset_observation,
            name="reset_observation",
        )
        source_run_hash = _required(self.source_run_hash, name="source_run_hash")
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("rows must contain at least one executed step.")
        if any(not isinstance(row, RobosuiteMixedControlRowA3) for row in rows):
            raise TypeError("rows must contain RobosuiteMixedControlRowA3 values.")
        expected_episode = rows[0].episode_id
        if expected_episode != episode_id or any(
            row.episode_id != episode_id for row in rows
        ):
            raise ValueError("rows do not bind the dataset episode.")
        if any(row.task_id != task_id for row in rows):
            raise ValueError("rows do not bind the dataset task.")
        if any(row.source_run_hash != source_run_hash for row in rows):
            raise ValueError("rows do not bind the source run.")
        if any(row.collection_mode != collection_mode for row in rows):
            raise ValueError("rows do not bind the collection mode.")
        if any(row.policy_variant != policy_variant for row in rows):
            raise ValueError("rows do not bind the policy variant.")
        if tuple(row.step_index for row in rows) != tuple(range(len(rows))):
            raise ValueError("rows must have contiguous step indexes starting at zero.")
        payload = {
            "manifest_hash": manifest_hash,
            "task_id": task_id,
            "episode_id": episode_id,
            "collection_mode": collection_mode,
            "policy_variant": policy_variant,
            "reset_observation": dict(reset_observation),
            "source_run_hash": source_run_hash,
            "row_hashes": tuple(row.semantic_content_hash for row in rows),
        }
        object.__setattr__(self, "manifest_hash", manifest_hash)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "collection_mode", collection_mode)
        object.__setattr__(self, "policy_variant", policy_variant)
        object.__setattr__(self, "reset_observation", reset_observation)
        object.__setattr__(self, "source_run_hash", source_run_hash)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A3_DATASET_HASH_DOMAIN),
        )

    @property
    def dataset_id(self) -> str:
        return f"robosuite-mixed-control-a3:{self.semantic_content_hash}"

    def to_payload(self) -> dict[str, Any]:
        """Return an offline-friendly payload without claiming persistence."""

        return {
            "manifest_hash": self.manifest_hash,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "collection_mode": self.collection_mode,
            "policy_variant": self.policy_variant,
            "reset_observation": dict(self.reset_observation),
            "source_run_hash": self.source_run_hash,
            "rows": tuple(row.to_payload() for row in self.rows),
            "semantic_content_hash": self.semantic_content_hash,
        }


class _RecordingBackend:
    """Passive backend proxy that records only reset/next observations."""

    def __init__(self, delegate: RobosuiteA0Backend) -> None:
        self._delegate = delegate
        self.observations: list[Mapping[str, Any]] = []

    @property
    def backend_id(self) -> str:
        return str(self._delegate.backend_id)

    def reset(self, manifest: RobosuiteResetManifest) -> Mapping[str, Any]:
        observation = self._delegate.reset(manifest)
        self.observations = [
            _snapshot_observation(observation, name="reset observation")
        ]
        return observation

    def step(self, action: tuple[float, ...]) -> RobosuiteBackendStep:
        backend_step = self._delegate.step(action)
        if not isinstance(backend_step, RobosuiteBackendStep):
            raise TypeError("backend.step must return a RobosuiteBackendStep.")
        self.observations.append(
            _snapshot_observation(backend_step.observation, name="step observation")
        )
        return backend_step

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _RecordingBackendFactory:
    def __init__(self, delegate_factory: Callable[[], RobosuiteA0Backend]) -> None:
        self._delegate_factory = delegate_factory
        self.recording: _RecordingBackend | None = None

    def __call__(self) -> _RecordingBackend:
        if self.recording is not None:
            raise RuntimeError("A3 capture runner expects one backend per run.")
        self.recording = _RecordingBackend(self._delegate_factory())
        return self.recording


class RobosuiteMixedControlCaptureRunnerA3:
    """Run A2 once while materializing one learning row per executed step."""

    def __init__(
        self,
        *,
        composer: Any,
        backend_factory: Callable[[], RobosuiteA0Backend],
        policy_source: PolicyInferenceSourceA2 | None = None,
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable.")
        if policy_source is not None and not isinstance(
            policy_source,
            PolicyInferenceSourceA2,
        ):
            raise TypeError("policy_source must implement PolicyInferenceSourceA2.")
        self._composer = composer
        self._backend_factory = backend_factory
        self._policy_source = policy_source

    def run(
        self,
        *,
        manifest: RobosuiteResetManifest,
        frames: Iterable[StreamingInterventionControlFrameA2],
        collection_mode: str,
    ) -> RobosuiteMixedControlDatasetA3:
        if not isinstance(manifest, RobosuiteResetManifest):
            raise TypeError("manifest must be a RobosuiteResetManifest.")
        collection_mode = _required(collection_mode, name="collection_mode")
        frame_values = tuple(frames)
        profile, frame_values = validate_collection_frames(
            collection_mode=collection_mode,
            frames=frame_values,
            policy_source=self._policy_source,
        )
        recording_factory = _RecordingBackendFactory(self._backend_factory)
        run_result = StreamingInterventionPolicyRunnerA2(
            composer=self._composer,
            backend_factory=recording_factory,
            policy_source=self._policy_source,
        ).run(manifest=manifest, frames=frame_values)
        recording = recording_factory.recording
        if recording is None:
            raise RuntimeError("A3 capture did not create a recording backend.")
        if len(recording.observations) != len(run_result.steps) + 1:
            raise RuntimeError(
                "recorded observation trace does not match executed step count."
            )
        variant = profile.policy_variant
        rows = tuple(
            self._row_from_step(
                manifest=manifest,
                run_result=run_result,
                step=step,
                step_index=index,
                observation=recording.observations[index],
                next_observation=recording.observations[index + 1],
                collection_mode=collection_mode,
                policy_variant=variant,
            )
            for index, step in enumerate(run_result.steps)
        )
        return RobosuiteMixedControlDatasetA3(
            manifest_hash=manifest.semantic_content_hash,
            task_id=manifest.task_id,
            episode_id=manifest.episode_id,
            collection_mode=collection_mode,
            policy_variant=variant,
            reset_observation=recording.observations[0],
            source_run_hash=run_result.semantic_content_hash,
            rows=rows,
        )

    @staticmethod
    def _row_from_step(
        *,
        manifest: RobosuiteResetManifest,
        run_result: StreamingInterventionA2RunResult,
        step: StreamingInterventionA2Step,
        step_index: int,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        collection_mode: str,
        policy_variant: str,
    ) -> RobosuiteMixedControlRowA3:
        expected_observation_hash = canonical_content_hash(
            dict(observation),
            domain=A2_POLICY_REQUEST_HASH_DOMAIN,
        )
        if step.evidence.observation_hash != expected_observation_hash:
            raise RuntimeError(
                "A2 evidence observation hash does not bind the recorded observation."
            )
        return RobosuiteMixedControlRowA3(
            episode_id=manifest.episode_id,
            task_id=manifest.task_id,
            source_run_hash=run_result.semantic_content_hash,
            step_index=step_index,
            collection_mode=collection_mode,
            policy_variant=policy_variant,
            frame_id=step.frame_id,
            frame_hash=step.frame_hash,
            mode=step.evidence.mode,
            observation=observation,
            next_observation=next_observation,
            observation_hash=step.evidence.observation_hash,
            next_observation_hash=step.backend_step.observation_hash,
            base_action=step.selection.base_action,
            human_raw_action=step.evidence.human_raw_action,
            human_mapped_action=step.evidence.human_mapped_action,
            a_policy=step.evidence.a_policy,
            a_rtc=step.evidence.a_rtc,
            a_mix=step.evidence.a_mix,
            a_safe=step.evidence.a_safe,
            a_exec=step.evidence.a_exec,
            reward=step.backend_step.reward,
            terminated=step.backend_step.terminated,
            truncated=step.backend_step.truncated,
            human_component_ids=step.evidence.human_component_ids,
            active_component_ids=step.evidence.active_component_ids,
            human_event_id=step.evidence.human_event_id,
            policy_source_id=step.evidence.policy_source_id,
            policy_source_revision=step.evidence.policy_source_revision,
            policy_inference_id=step.evidence.policy_inference_id,
            selection_hash=step.evidence.selection_hash,
            ownership_hash=step.evidence.ownership_hash,
            backend_step_hash=step.evidence.backend_step_hash,
            clipped_dimensions=step.selection.clipped_dimensions,
        )


__all__ = [
    "A3_COLLECTION_MODES",
    "A3_COLLECTION_MODE_ORDINARY_TELEOP",
    "A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL",
    "A3_COLLECTION_MODE_TAKEOVER_TELEOP",
    "A3_DATASET_HASH_DOMAIN",
    "A3_OBSERVATION_HASH_DOMAIN",
    "A3_POLICY_VARIANTS",
    "A3_POLICY_VARIANT_NONE",
    "A3_POLICY_VARIANT_RTC",
    "A3_POLICY_VARIANT_STANDARD",
    "A3_ROW_HASH_DOMAIN",
    "A3CollectionProfile",
    "RobosuiteMixedControlCaptureRunnerA3",
    "RobosuiteMixedControlDatasetA3",
    "RobosuiteMixedControlRowA3",
    "policy_variant",
    "resolve_collection_profile",
    "validate_collection_frames",
]
