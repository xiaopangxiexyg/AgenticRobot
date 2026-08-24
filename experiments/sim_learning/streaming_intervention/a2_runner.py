"""Headless policy/RTC-to-composition runner for the A2 experiment seam."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from dataset_contracts.canonical_encoding import canonical_content_hash
from sim.robosuite import (
    RobosuiteA0Backend,
    RobosuiteBackendStep,
    RobosuiteEpisodeStartArtifactA1,
    RobosuiteMixedControlAdapter,
    RobosuiteResetManifest,
    RobosuiteResetReceipt,
)
from sim.robosuite.adapter import (
    ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
)

from .a2 import (
    A2_ACTION_DIMENSION,
    A2_POLICY_REQUEST_HASH_DOMAIN,
    A2_POLICY_SOURCE_KIND_RTC,
    A2_POLICY_SOURCE_KIND_STANDARD,
    ComponentMixSelectionA2,
    ComponentOwnershipA2,
    PolicyInferenceRequestA2,
    PolicyInferenceSourceA2,
    PolicyProposalA2,
    RobosuiteComponentComposerA2,
    StreamingInterventionControlFrameA2,
    StreamingInterventionEvidenceA2,
)

A2_STEP_HASH_DOMAIN = "dexteleop.streaming_intervention.step.a2"
A2_RUN_HASH_DOMAIN = "dexteleop.streaming_intervention.run.a2"


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
class StreamingInterventionA2Step:
    """One policy/human input, mix decision, and backend result."""

    frame_id: str
    frame_hash: str
    selection: ComponentMixSelectionA2
    backend_step: RobosuiteBackendStep
    evidence: StreamingInterventionEvidenceA2
    input_frame: StreamingInterventionControlFrameA2
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = _required(self.frame_id, name="frame_id")
        frame_hash = _required(self.frame_hash, name="frame_hash")
        if not isinstance(self.selection, ComponentMixSelectionA2):
            raise TypeError("selection must be a ComponentMixSelectionA2.")
        if not isinstance(self.backend_step, RobosuiteBackendStep):
            raise TypeError("backend_step must be a RobosuiteBackendStep.")
        if not isinstance(self.evidence, StreamingInterventionEvidenceA2):
            raise TypeError("evidence must be a StreamingInterventionEvidenceA2.")
        if not isinstance(
            self.input_frame,
            StreamingInterventionControlFrameA2,
        ):
            raise TypeError(
                "input_frame must be a StreamingInterventionControlFrameA2."
            )
        if self.input_frame.frame_id != frame_id or (
            self.input_frame.semantic_content_hash != frame_hash
        ):
            raise ValueError("input_frame does not bind frame_id and frame_hash.")
        if self.evidence.frame_id != frame_id or self.evidence.frame_hash != frame_hash:
            raise ValueError("evidence does not bind frame_id and frame_hash.")
        if self.evidence.selection_hash != self.selection.semantic_content_hash:
            raise ValueError("evidence does not bind the component selection.")
        if self.evidence.ownership_hash != self.selection.ownership.semantic_content_hash:
            raise ValueError("evidence does not bind component ownership.")
        if self.evidence.active_component_ids != (
            self.selection.ownership.active_component_ids
        ):
            raise ValueError("evidence does not bind active component ownership.")
        if self.evidence.a_mix != self.selection.preclip_action:
            raise ValueError("evidence a_mix does not bind the preclip action.")
        if self.evidence.a_safe != self.selection.applied_action:
            raise ValueError("evidence a_safe does not bind the applied action.")
        if self.evidence.backend_step_hash != self.backend_step.semantic_content_hash:
            raise ValueError("evidence does not bind the backend step.")
        if self.evidence.a_exec != self.selection.applied_action:
            raise ValueError("evidence a_exec does not bind the applied action.")
        payload = {
            "frame_id": frame_id,
            "frame_hash": frame_hash,
            "selection_hash": self.selection.semantic_content_hash,
            "backend_step_hash": self.backend_step.semantic_content_hash,
            "evidence_hash": self.evidence.semantic_content_hash,
        }
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "frame_hash", frame_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=A2_STEP_HASH_DOMAIN),
        )


@dataclass(frozen=True)
class StreamingInterventionA2RunResult:
    """Content-addressed ordered evidence for one A2 policy run."""

    manifest: RobosuiteResetManifest
    reset_receipt: RobosuiteResetReceipt
    steps: tuple[StreamingInterventionA2Step, ...]
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
        if any(not isinstance(step, StreamingInterventionA2Step) for step in steps):
            raise TypeError("steps must contain StreamingInterventionA2Step values.")
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
            if source_effective_xml_hash != reset_artifact.effective_model_xml_sha256:
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
            canonical_content_hash(payload, domain=A2_RUN_HASH_DOMAIN),
        )

    @property
    def run_id(self) -> str:
        return f"streaming-intervention-a2-run:{self.semantic_content_hash}"


class StreamingInterventionPolicyRunnerA2:
    """Run a deterministic policy/RTC/human stream through an injected backend.

    At most one policy source is active.  A standard source owns the nominal
    action; an RTC source owns an intervention residual.  A synthetic human
    action may replace that intervention, but cannot be active alongside RTC.
    This keeps the current experiment bounded to one intervention source while
    retaining the source/revision seam needed by later deployment adapters.
    """

    def __init__(
        self,
        *,
        composer: RobosuiteComponentComposerA2,
        backend_factory: Callable[[], RobosuiteA0Backend],
        policy_source: PolicyInferenceSourceA2 | None = None,
    ) -> None:
        if not isinstance(composer, RobosuiteComponentComposerA2):
            raise TypeError("composer must be a RobosuiteComponentComposerA2.")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable.")
        if policy_source is not None and not isinstance(
            policy_source,
            PolicyInferenceSourceA2,
        ):
            raise TypeError("policy_source must implement PolicyInferenceSourceA2.")
        if policy_source is not None and (
            policy_source.action_dimension != A2_ACTION_DIMENSION
        ):
            raise ValueError("policy source action dimension must be seven.")
        self._composer = composer
        self._backend_factory = backend_factory
        self._policy_source = policy_source

    def run(
        self,
        *,
        manifest: RobosuiteResetManifest,
        frames: Iterable[StreamingInterventionControlFrameA2],
    ) -> StreamingInterventionA2RunResult:
        if manifest.action_dimension != A2_ACTION_DIMENSION:
            raise ValueError("A2 requires a seven-dimensional robosuite manifest.")
        adapter = RobosuiteMixedControlAdapter(
            action_low=self._composer.action_low,
            action_high=self._composer.action_high,
        )
        backend = self._backend_factory()
        try:
            observation, reset_receipt = adapter.reset_backend(backend, manifest)
            reset_artifact = getattr(backend, "reset_artifact", None)
            source_effective_xml_hash: str | None = None
            if reset_artifact is not None and not isinstance(
                reset_artifact,
                RobosuiteEpisodeStartArtifactA1,
            ):
                raise TypeError(
                    "backend reset_artifact must be RobosuiteEpisodeStartArtifactA1."
                )
            if reset_artifact is not None:
                source_effective_xml_hash = _required(
                    getattr(backend, "last_effective_model_xml_sha256", None),
                    name="backend.last_effective_model_xml_sha256",
                )
                if source_effective_xml_hash != reset_artifact.effective_model_xml_sha256:
                    raise RuntimeError(
                        "backend effective XML hash does not bind its reset artifact."
                    )
            steps: list[StreamingInterventionA2Step] = []
            seen_frame_ids: set[str] = set()
            stop_reason = "input_exhausted"
            for frame in frames:
                if not isinstance(frame, StreamingInterventionControlFrameA2):
                    raise TypeError(
                        "frames must contain StreamingInterventionControlFrameA2 values."
                    )
                if frame.frame_id in seen_frame_ids:
                    raise ValueError("frame_id values must be unique within a run.")
                seen_frame_ids.add(frame.frame_id)
                proposal = self._infer_policy(
                    manifest=manifest,
                    frame=frame,
                    observation=observation,
                    step_index=len(steps),
                )
                base_action, intervention_action, ownership = self._resolve_sources(
                    frame=frame,
                    proposal=proposal,
                )
                selection = self._composer.compose(
                    mode=frame.mode,
                    base_action=base_action,
                    intervention_action=intervention_action,
                    ownership=ownership,
                )
                backend_step = backend.step(selection.applied_action)
                if not isinstance(backend_step, RobosuiteBackendStep):
                    raise TypeError("backend.step must return a RobosuiteBackendStep.")
                evidence = StreamingInterventionEvidenceA2(
                    frame_id=frame.frame_id,
                    frame_hash=frame.semantic_content_hash,
                    mode=frame.mode,
                    policy_proposal=proposal,
                    observation_hash=canonical_content_hash(
                        dict(observation),
                        domain=A2_POLICY_REQUEST_HASH_DOMAIN,
                    ),
                    human_raw_action=frame.human_raw_action,
                    human_mapped_action=frame.human_mapped_action,
                    human_component_ids=frame.human_component_ids,
                    human_event_id=frame.human_event_id,
                    selection_hash=selection.semantic_content_hash,
                    ownership_hash=selection.ownership.semantic_content_hash,
                    active_component_ids=selection.ownership.active_component_ids,
                    a_policy=(
                        None
                        if proposal is None
                        or proposal.source_kind != A2_POLICY_SOURCE_KIND_STANDARD
                        else proposal.action
                    ),
                    a_rtc=(
                        None
                        if proposal is None
                        or proposal.source_kind != A2_POLICY_SOURCE_KIND_RTC
                        else proposal.action
                    ),
                    a_mix=selection.preclip_action,
                    a_safe=selection.applied_action,
                    a_exec=selection.applied_action,
                    backend_step_hash=backend_step.semantic_content_hash,
                )
                steps.append(
                    StreamingInterventionA2Step(
                        frame_id=frame.frame_id,
                        frame_hash=frame.semantic_content_hash,
                        selection=selection,
                        backend_step=backend_step,
                        evidence=evidence,
                        input_frame=frame,
                    )
                )
                observation = MappingProxyType(dict(backend_step.observation))
                if backend_step.terminated:
                    stop_reason = "backend_terminated"
                    break
                if backend_step.truncated:
                    stop_reason = "manifest_step_limit"
                    break
            return StreamingInterventionA2RunResult(
                manifest=manifest,
                reset_receipt=reset_receipt,
                steps=tuple(steps),
                stop_reason=stop_reason,
                reset_artifact=reset_artifact,
                source_effective_model_xml_sha256=source_effective_xml_hash,
            )
        finally:
            _close_backend(backend)

    def _infer_policy(
        self,
        *,
        manifest: RobosuiteResetManifest,
        frame: StreamingInterventionControlFrameA2,
        observation: Mapping[str, object],
        step_index: int,
    ) -> PolicyProposalA2 | None:
        source = self._policy_source
        if source is None:
            return None
        request = PolicyInferenceRequestA2(
            episode_id=manifest.episode_id,
            frame_id=frame.frame_id,
            step_index=step_index,
            observation=observation,
            nominal_action=frame.base_action,
            expected_source_revision=source.revision,
        )
        proposal = source.infer(request)
        if not isinstance(proposal, PolicyProposalA2):
            raise TypeError("policy source infer() must return PolicyProposalA2.")
        if (
            proposal.source_id != source.source_id
            or proposal.source_kind != source.source_kind
            or proposal.action_role != source.action_role
            or proposal.revision != source.revision
            or proposal.component_ids != source.component_ids
        ):
            raise RuntimeError("policy proposal does not bind the installed source.")
        if (
            proposal.frame_id != frame.frame_id
            or proposal.step_index != step_index
            or proposal.observation_hash
            != canonical_content_hash(
                dict(observation),
                domain=A2_POLICY_REQUEST_HASH_DOMAIN,
            )
        ):
            raise RuntimeError("policy proposal does not bind the current inference input.")
        return proposal

    @staticmethod
    def _resolve_sources(
        *,
        frame: StreamingInterventionControlFrameA2,
        proposal: PolicyProposalA2 | None,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...] | None,
        ComponentOwnershipA2,
    ]:
        if frame.mode == ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0:
            if proposal is not None and proposal.source_kind == A2_POLICY_SOURCE_KIND_RTC:
                raise ValueError("RTC policy cannot be used in base_only mode.")
            if proposal is None:
                if frame.base_action is None:
                    if frame.human_mapped_action is None:
                        raise ValueError(
                            "a base_only frame needs base_action or human_mapped_action."
                        )
                    base_action = frame.human_mapped_action
                else:
                    base_action = frame.base_action
                    if (
                        frame.human_mapped_action is not None
                        and frame.human_mapped_action != base_action
                    ):
                        raise ValueError(
                            "human mapped action conflicts with base_action."
                        )
            else:
                if frame.base_action is not None and frame.base_action != proposal.action:
                    raise ValueError(
                        "frame base_action conflicts with standard policy output."
                    )
                if (
                    frame.human_mapped_action is not None
                    and frame.human_mapped_action != proposal.action
                ):
                    raise ValueError(
                        "human base action conflicts with standard policy output."
                    )
                base_action = proposal.action
            return (
                base_action,
                None,
                ComponentOwnershipA2(),
            )

        if proposal is None:
            if frame.base_action is None:
                raise ValueError("a frame without a policy source needs base_action.")
            base_action = frame.base_action
        elif proposal.source_kind == A2_POLICY_SOURCE_KIND_STANDARD:
            if frame.base_action is not None and frame.base_action != proposal.action:
                raise ValueError(
                    "frame base_action conflicts with standard policy output."
                )
            base_action = proposal.action
        elif proposal.source_kind == A2_POLICY_SOURCE_KIND_RTC:
            if frame.base_action is None:
                raise ValueError("RTC intervention requires frame base_action.")
            base_action = frame.base_action
            if frame.has_human_intervention:
                raise ValueError(
                    "RTC and human intervention cannot control one frame together."
                )
        else:  # pragma: no cover - PolicyProposalA2 validates this.
            raise AssertionError("unsupported policy source kind")

        if frame.has_human_intervention:
            if proposal is not None and proposal.source_kind == A2_POLICY_SOURCE_KIND_RTC:
                raise ValueError("RTC and human intervention cannot control one frame together.")
            return (
                base_action,
                frame.human_mapped_action,
                ComponentOwnershipA2.for_components(frame.human_component_ids),
            )
        if proposal is not None and proposal.source_kind == A2_POLICY_SOURCE_KIND_RTC:
            return (
                base_action,
                proposal.action,
                ComponentOwnershipA2.for_components(proposal.component_ids),
            )
        raise ValueError("overlay mode needs a human or RTC intervention source.")


__all__ = [
    "A2_RUN_HASH_DOMAIN",
    "A2_STEP_HASH_DOMAIN",
    "StreamingInterventionA2RunResult",
    "StreamingInterventionA2Step",
    "StreamingInterventionPolicyRunnerA2",
]
