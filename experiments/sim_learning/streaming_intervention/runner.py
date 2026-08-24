"""Injected mixed-control runner and exact episode-start prefix replay."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from sim.robosuite import (
    RobosuiteA0Backend,
    RobosuiteEpisodeStartArtifactA1,
    RobosuiteMixedControlAdapter,
    RobosuiteResetManifest,
)

from .mock_runner import (
    StreamingInterventionFrame,
    StreamingInterventionRunResult,
    StreamingInterventionStep,
)

STREAMING_INTERVENTION_PREFIX_REPLAY_HASH_DOMAIN_A1 = (
    "dexteleop.streaming_intervention.prefix_replay.a1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _close_backend(backend: object) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class PrefixReplayReceiptA1:
    """Fail-closed comparison of one prefix against its source evidence."""

    source_run_hash: str
    reset_artifact_hash: str
    prefix_length: int
    matched: bool
    matched_steps: int
    reset_observation_match: bool
    first_mismatch_index: int | None = None
    mismatch_kind: str | None = None
    expected_hash: str | None = None
    actual_hash: str | None = None
    replay_effective_xml_sha256: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_run_hash = _required(self.source_run_hash, name="source_run_hash")
        artifact_hash = _required(
            self.reset_artifact_hash,
            name="reset_artifact_hash",
        )
        if isinstance(self.prefix_length, bool) or not isinstance(
            self.prefix_length, int
        ):
            raise TypeError("prefix_length must be an integer.")
        if self.prefix_length < 0:
            raise ValueError("prefix_length must be a non-negative integer.")
        if isinstance(self.matched_steps, bool) or not isinstance(
            self.matched_steps, int
        ):
            raise TypeError("matched_steps must be an integer.")
        if not (0 <= self.matched_steps <= self.prefix_length):
            raise ValueError("matched_steps must lie within the requested prefix.")
        if not isinstance(self.matched, bool) or not isinstance(
            self.reset_observation_match, bool
        ):
            raise TypeError("matched flags must be bool values.")
        mismatch_kind = self.mismatch_kind
        if self.matched:
            if not self.reset_observation_match:
                raise ValueError(
                    "a matched receipt requires a matching reset observation."
                )
            if self.matched_steps != self.prefix_length:
                raise ValueError("a matched receipt must cover the full prefix.")
            if any(
                value is not None
                for value in (
                    self.first_mismatch_index,
                    mismatch_kind,
                    self.expected_hash,
                    self.actual_hash,
                )
            ):
                raise ValueError("a matched receipt cannot contain mismatch fields.")
        else:
            mismatch_kind = _required(mismatch_kind, name="mismatch_kind")
            if self.expected_hash is None or self.actual_hash is None:
                raise ValueError(
                    "a mismatch receipt requires expected and actual hashes."
                )
            expected_hash = _required(self.expected_hash, name="expected_hash")
            actual_hash = _required(self.actual_hash, name="actual_hash")
            if self.first_mismatch_index is not None:
                if isinstance(self.first_mismatch_index, bool) or not isinstance(
                    self.first_mismatch_index, int
                ):
                    raise TypeError("first_mismatch_index must be an integer.")
                if not 0 <= self.first_mismatch_index < self.prefix_length:
                    raise ValueError("first_mismatch_index is outside the prefix.")
        if self.matched:
            expected_hash = None
            actual_hash = None
        replay_xml_hash = (
            None
            if self.replay_effective_xml_sha256 is None
            else _required(
                self.replay_effective_xml_sha256,
                name="replay_effective_xml_sha256",
            )
        )
        payload = {
            "source_run_hash": source_run_hash,
            "reset_artifact_hash": artifact_hash,
            "prefix_length": int(self.prefix_length),
            "matched": self.matched,
            "matched_steps": int(self.matched_steps),
            "reset_observation_match": self.reset_observation_match,
            "first_mismatch_index": self.first_mismatch_index,
            "mismatch_kind": mismatch_kind,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "replay_effective_xml_sha256": replay_xml_hash,
            "replay_semantics": "episode_start_plus_full_executed_action_prefix",
        }
        object.__setattr__(self, "source_run_hash", source_run_hash)
        object.__setattr__(self, "reset_artifact_hash", artifact_hash)
        object.__setattr__(self, "mismatch_kind", mismatch_kind)
        object.__setattr__(self, "expected_hash", expected_hash)
        object.__setattr__(self, "actual_hash", actual_hash)
        object.__setattr__(
            self,
            "replay_effective_xml_sha256",
            replay_xml_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=STREAMING_INTERVENTION_PREFIX_REPLAY_HASH_DOMAIN_A1,
            ),
        )

    @property
    def receipt_id(self) -> str:
        return f"streaming-prefix-replay:{self.semantic_content_hash}"


class StreamingInterventionRunner:
    """Run three A0 modes against an injected backend and close it reliably."""

    def __init__(
        self,
        *,
        adapter: RobosuiteMixedControlAdapter,
        backend_factory: Callable[[], RobosuiteA0Backend],
    ) -> None:
        if not isinstance(adapter, RobosuiteMixedControlAdapter):
            raise TypeError("adapter must be a RobosuiteMixedControlAdapter.")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable.")
        self._adapter = adapter
        self._backend_factory = backend_factory

    def run(
        self,
        *,
        manifest: RobosuiteResetManifest,
        frames: Iterable[StreamingInterventionFrame],
    ) -> StreamingInterventionRunResult:
        backend = self._backend_factory()
        try:
            _, reset_receipt = self._adapter.reset_backend(backend, manifest)
            reset_artifact = getattr(backend, "reset_artifact", None)
            source_effective_xml_hash: str | None = None
            if reset_artifact is not None and not isinstance(
                reset_artifact, RobosuiteEpisodeStartArtifactA1
            ):
                raise TypeError(
                    "backend reset_artifact must be RobosuiteEpisodeStartArtifactA1."
                )
            if reset_artifact is not None:
                source_effective_xml_hash = _required(
                    getattr(backend, "last_effective_model_xml_sha256", None),
                    name="backend.last_effective_model_xml_sha256",
                )
                if source_effective_xml_hash != (
                    reset_artifact.effective_model_xml_sha256
                ):
                    raise RuntimeError(
                        "backend effective XML hash does not bind its reset artifact."
                    )
            steps: list[StreamingInterventionStep] = []
            stop_reason = "input_exhausted"
            seen_frame_ids: set[str] = set()
            for frame in frames:
                if not isinstance(frame, StreamingInterventionFrame):
                    raise TypeError(
                        "frames must contain StreamingInterventionFrame values."
                    )
                if frame.frame_id in seen_frame_ids:
                    raise ValueError("frame_id values must be unique within a run.")
                seen_frame_ids.add(frame.frame_id)
                selection = self._adapter.compose(
                    mode=frame.mode,
                    base_action=frame.base_action,
                    intervention_action=frame.intervention_action,
                )
                backend_step = backend.step(selection.applied_action)
                steps.append(
                    StreamingInterventionStep(
                        frame_id=frame.frame_id,
                        frame_hash=frame.semantic_content_hash,
                        selection=selection,
                        backend_step=backend_step,
                        input_frame=frame,
                    )
                )
                if backend_step.terminated:
                    stop_reason = "backend_terminated"
                    break
                if backend_step.truncated:
                    stop_reason = "manifest_step_limit"
                    break
            return StreamingInterventionRunResult(
                manifest=manifest,
                reset_receipt=reset_receipt,
                steps=tuple(steps),
                stop_reason=stop_reason,
                reset_artifact=reset_artifact,
                source_effective_model_xml_sha256=source_effective_xml_hash,
            )
        finally:
            _close_backend(backend)

    def replay_prefix(
        self,
        *,
        source_result: StreamingInterventionRunResult,
        prefix_length: int,
    ) -> PrefixReplayReceiptA1:
        if not isinstance(source_result, StreamingInterventionRunResult):
            raise TypeError("source_result must be a StreamingInterventionRunResult.")
        if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
            raise TypeError("prefix_length must be an integer.")
        if prefix_length < 0 or prefix_length > len(source_result.steps):
            raise ValueError("prefix_length is outside the recorded run.")
        artifact = source_result.reset_artifact
        if artifact is None:
            raise ValueError(
                "exact prefix replay requires a captured episode-start artifact."
            )
        source_effective_xml_hash = _required(
            source_result.source_effective_model_xml_sha256,
            name="source_result.source_effective_model_xml_sha256",
        )
        for step in source_result.steps[:prefix_length]:
            if step.input_frame is None:
                raise ValueError(
                    "exact prefix replay requires each step's original input frame."
                )
        backend = self._backend_factory()
        try:
            restore_episode_start = getattr(
                backend,
                "restore_episode_start",
                None,
            )
            if not callable(restore_episode_start):
                raise TypeError(
                    "backend must implement RobosuiteReplayableA1Backend for replay."
                )
            replay_observation = restore_episode_start(
                source_result.manifest,
                artifact,
            )
            _, replay_reset_receipt = self._adapter.bind_reset_observation(
                backend=backend,
                manifest=source_result.manifest,
                observation=replay_observation,
            )
            replay_xml_hash = _required(
                backend.last_effective_model_xml_sha256,
                name="backend.last_effective_model_xml_sha256",
            )
            reset_matches = (
                replay_reset_receipt.initial_observation_hash
                == source_result.reset_receipt.initial_observation_hash
            )
            if replay_xml_hash != source_effective_xml_hash:
                return self._mismatch_receipt(
                    source_result=source_result,
                    prefix_length=prefix_length,
                    matched_steps=0,
                    reset_observation_match=reset_matches,
                    first_mismatch_index=None,
                    mismatch_kind="effective_model_xml_sha256",
                    expected_hash=source_effective_xml_hash,
                    actual_hash=replay_xml_hash,
                    replay_xml_hash=replay_xml_hash,
                )
            if not reset_matches:
                return self._mismatch_receipt(
                    source_result=source_result,
                    prefix_length=prefix_length,
                    matched_steps=0,
                    reset_observation_match=False,
                    first_mismatch_index=None,
                    mismatch_kind="reset_observation_hash",
                    expected_hash=source_result.reset_receipt.initial_observation_hash,
                    actual_hash=replay_reset_receipt.initial_observation_hash,
                    replay_xml_hash=replay_xml_hash,
                )
            matched_steps = 0
            for index, source_step in enumerate(source_result.steps[:prefix_length]):
                frame = source_step.input_frame
                if frame is None:  # guarded above; keeps static narrowing explicit.
                    raise AssertionError("missing replay frame")
                selection = self._adapter.compose(
                    mode=frame.mode,
                    base_action=frame.base_action,
                    intervention_action=frame.intervention_action,
                )
                if (
                    selection.semantic_content_hash
                    != source_step.selection.semantic_content_hash
                ):
                    return self._mismatch_receipt(
                        source_result=source_result,
                        prefix_length=prefix_length,
                        matched_steps=matched_steps,
                        reset_observation_match=True,
                        first_mismatch_index=index,
                        mismatch_kind="selection_hash",
                        expected_hash=source_step.selection.semantic_content_hash,
                        actual_hash=selection.semantic_content_hash,
                        replay_xml_hash=replay_xml_hash,
                    )
                replay_step = backend.step(selection.applied_action)
                if (
                    replay_step.semantic_content_hash
                    != source_step.backend_step.semantic_content_hash
                ):
                    return self._mismatch_receipt(
                        source_result=source_result,
                        prefix_length=prefix_length,
                        matched_steps=matched_steps,
                        reset_observation_match=True,
                        first_mismatch_index=index,
                        mismatch_kind="backend_step_hash",
                        expected_hash=source_step.backend_step.semantic_content_hash,
                        actual_hash=replay_step.semantic_content_hash,
                        replay_xml_hash=replay_xml_hash,
                    )
                matched_steps += 1
            return PrefixReplayReceiptA1(
                source_run_hash=source_result.semantic_content_hash,
                reset_artifact_hash=artifact.semantic_content_hash,
                prefix_length=prefix_length,
                matched=True,
                matched_steps=matched_steps,
                reset_observation_match=True,
                replay_effective_xml_sha256=replay_xml_hash,
            )
        finally:
            _close_backend(backend)

    @staticmethod
    def _mismatch_receipt(
        *,
        source_result: StreamingInterventionRunResult,
        prefix_length: int,
        matched_steps: int,
        reset_observation_match: bool,
        first_mismatch_index: int | None,
        mismatch_kind: str,
        expected_hash: str,
        actual_hash: str,
        replay_xml_hash: str,
    ) -> PrefixReplayReceiptA1:
        artifact = source_result.reset_artifact
        if artifact is None:
            raise AssertionError("mismatch receipt requires a reset artifact")
        return PrefixReplayReceiptA1(
            source_run_hash=source_result.semantic_content_hash,
            reset_artifact_hash=artifact.semantic_content_hash,
            prefix_length=prefix_length,
            matched=False,
            matched_steps=matched_steps,
            reset_observation_match=reset_observation_match,
            first_mismatch_index=first_mismatch_index,
            mismatch_kind=mismatch_kind,
            expected_hash=_required(expected_hash, name="expected_hash"),
            actual_hash=_required(actual_hash, name="actual_hash"),
            replay_effective_xml_sha256=replay_xml_hash,
        )


__all__ = [
    "STREAMING_INTERVENTION_PREFIX_REPLAY_HASH_DOMAIN_A1",
    "PrefixReplayReceiptA1",
    "StreamingInterventionRunner",
]
