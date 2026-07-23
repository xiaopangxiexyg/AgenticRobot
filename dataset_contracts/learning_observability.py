"""Pure contracts for policy-side facts that cannot be reconstructed later.

The records in this module are intentionally independent from control
authority, policy execution, storage, and learner implementations.  A policy
adapter may emit them for an inference only when it has the corresponding
evidence.  Missing evidence must remain explicit; callers must never infer
noise, recurrent state, or model-native actions from a decoded RobotAction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .canonical_encoding import canonical_content_hash


LEARNING_OBSERVABILITY_SCHEMA_V1 = "dexteleop_learning_observability_v1"

ADAPTATION_EVIDENCE_NOT_APPLICABLE = "not_applicable"
ADAPTATION_EVIDENCE_STATE_IDENTITY = "state_identity"
ADAPTATION_EVIDENCE_CHECKPOINTED = "checkpointed_state"
ADAPTATION_EVIDENCE_LEVELS_V1 = (
    ADAPTATION_EVIDENCE_NOT_APPLICABLE,
    ADAPTATION_EVIDENCE_STATE_IDENTITY,
    ADAPTATION_EVIDENCE_CHECKPOINTED,
)

GENERATIVE_EVIDENCE_NOT_APPLICABLE = "not_applicable"
GENERATIVE_EVIDENCE_RNG_RECEIPT = "rng_receipt"
GENERATIVE_EVIDENCE_INITIAL_NOISE = "initial_noise"
GENERATIVE_EVIDENCE_LEVELS_V1 = (
    GENERATIVE_EVIDENCE_NOT_APPLICABLE,
    GENERATIVE_EVIDENCE_RNG_RECEIPT,
    GENERATIVE_EVIDENCE_INITIAL_NOISE,
)

MODEL_ACTION_CHUNK_NOT_APPLICABLE = "not_applicable"
MODEL_ACTION_CHUNK_EXACT = "exact_model_chunk"
MODEL_ACTION_CHUNK_EVIDENCE_LEVELS_V1 = (
    MODEL_ACTION_CHUNK_NOT_APPLICABLE,
    MODEL_ACTION_CHUNK_EXACT,
)

ADAPTATION_STATE_KINDS_V1 = (
    "rnn_state",
    "kv_memory",
    "fast_weights",
    "test_time_adapter",
    "other_artifact_bound",
)
ADAPTATION_STATE_SCOPES_V1 = (
    "inference",
    "episode",
    "task",
    "runtime_session",
)

GENERATIVE_NOISE_ACTUAL = "actual_initial_noise"
GENERATIVE_NOISE_RNG_ONLY = "rng_receipt_only"
GENERATIVE_NOISE_EVIDENCE_KINDS_V1 = (
    GENERATIVE_NOISE_ACTUAL,
    GENERATIVE_NOISE_RNG_ONLY,
)

HISTORY_ACTION_STAGES_V1 = (
    "policy_output",
    "decoded_proposal",
    "selected",
    "safe",
    "commanded",
    "executed_feedback",
)
HISTORY_OVERRIDE_POLICIES_V1 = (
    "use_policy_counterfactual",
    "use_human_behavior",
    "use_both_with_source_token",
    "mask_action_token",
)
HISTORY_MISSING_ACTION_POLICIES_V1 = (
    "error",
    "mask_token",
    "drop_step",
)

POLICY_ACTION_CODEC_DECODE_ONLY = "decode_only"
POLICY_ACTION_CODEC_BIDIRECTIONAL = "bidirectional"
POLICY_ACTION_CODEC_DIRECTIONS_V1 = (
    POLICY_ACTION_CODEC_DECODE_ONLY,
    POLICY_ACTION_CODEC_BIDIRECTIONAL,
)

_OBSERVABILITY_SPEC_HASH_DOMAIN_V1 = "dexteleop.policy_learning_observability_spec.v1"
_ADAPTATION_STATE_HASH_DOMAIN_V1 = "dexteleop.policy_adaptation_state.v1"
_GENERATIVE_SAMPLE_HASH_DOMAIN_V1 = "dexteleop.generative_action_sample.v1"
_HISTORY_ACTION_CONTRACT_HASH_DOMAIN_V1 = "dexteleop.history_action_contract.v1"
_POLICY_ACTION_CODEC_HASH_DOMAIN_V1 = "dexteleop.policy_action_codec.v1"
_INFERENCE_OBSERVABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.policy_inference_observability.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _unique_text_tuple(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique.")
    return result


@dataclass(frozen=True)
class PolicyLearningObservabilitySpec:
    """Artifact-bound evidence requirements for one policy node.

    ``not_applicable`` is a semantic statement about the bound artifact, not a
    substitute for evidence that a required producer failed to emit.
    """

    spec_id: str
    policy_node_id: str
    policy_artifact_hash: str
    adaptation_state_evidence: str = ADAPTATION_EVIDENCE_NOT_APPLICABLE
    generative_sample_evidence: str = GENERATIVE_EVIDENCE_NOT_APPLICABLE
    model_action_chunk_evidence: str = MODEL_ACTION_CHUNK_NOT_APPLICABLE
    history_action_contract_hashes: tuple[str, ...] = ()
    policy_action_codec_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("spec_id", "policy_node_id", "policy_artifact_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.adaptation_state_evidence not in ADAPTATION_EVIDENCE_LEVELS_V1:
            raise ValueError("Unsupported adaptation_state_evidence.")
        if self.generative_sample_evidence not in GENERATIVE_EVIDENCE_LEVELS_V1:
            raise ValueError("Unsupported generative_sample_evidence.")
        if self.model_action_chunk_evidence not in MODEL_ACTION_CHUNK_EVIDENCE_LEVELS_V1:
            raise ValueError("Unsupported model_action_chunk_evidence.")
        if (
            self.generative_sample_evidence != GENERATIVE_EVIDENCE_NOT_APPLICABLE
            and self.model_action_chunk_evidence != MODEL_ACTION_CHUNK_EXACT
        ):
            raise ValueError(
                "Generative sampling evidence requires the exact generated model-action chunk."
            )
        history_hashes = _unique_text_tuple(
            tuple(self.history_action_contract_hashes),
            name="history_action_contract_hash",
        )
        codec_hash = _optional(self.policy_action_codec_hash)
        if (
            self.model_action_chunk_evidence == MODEL_ACTION_CHUNK_EXACT
            and codec_hash is None
        ):
            raise ValueError(
                "Exact model-action evidence requires policy_action_codec_hash."
            )
        object.__setattr__(self, "history_action_contract_hashes", history_hashes)
        object.__setattr__(self, "policy_action_codec_hash", codec_hash)
        payload = {
            "spec_id": self.spec_id,
            "policy_node_id": self.policy_node_id,
            "policy_artifact_hash": self.policy_artifact_hash,
            "adaptation_state_evidence": self.adaptation_state_evidence,
            "generative_sample_evidence": self.generative_sample_evidence,
            "model_action_chunk_evidence": self.model_action_chunk_evidence,
            "history_action_contract_hashes": history_hashes,
            "policy_action_codec_hash": codec_hash,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload, domain=_OBSERVABILITY_SPEC_HASH_DOMAIN_V1
            ),
        )


@dataclass(frozen=True)
class PolicyAdaptationStateRecord:
    """Stable identity/checksum for artifact-bound sequential policy state."""

    state_id: str
    policy_node_id: str
    policy_artifact_hash: str
    adaptation_contract_hash: str
    state_kind: str
    scope: str
    updated_at: float
    clock_domain_id: str
    state_content_hash: str
    parent_state_id: str | None = None
    initialized_from_id: str | None = None
    reset_reason: str | None = None
    source_observation_cut_ids: tuple[str, ...] = ()
    source_action_history_ids: tuple[str, ...] = ()
    snapshot_blob_id: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "state_id",
            "policy_node_id",
            "policy_artifact_hash",
            "adaptation_contract_hash",
            "clock_domain_id",
            "state_content_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.state_kind not in ADAPTATION_STATE_KINDS_V1:
            raise ValueError(f"Unsupported adaptation state kind: {self.state_kind!r}.")
        if self.scope not in ADAPTATION_STATE_SCOPES_V1:
            raise ValueError(f"Unsupported adaptation state scope: {self.scope!r}.")
        updated_at = _finite(self.updated_at, name="updated_at")
        parent = _optional(self.parent_state_id)
        initialized = _optional(self.initialized_from_id)
        reset_reason = _optional(self.reset_reason)
        blob_id = _optional(self.snapshot_blob_id)
        if parent == self.state_id:
            raise ValueError("An adaptation state cannot be its own parent.")
        observations = _unique_text_tuple(
            tuple(self.source_observation_cut_ids), name="source_observation_cut_id"
        )
        actions = _unique_text_tuple(
            tuple(self.source_action_history_ids), name="source_action_history_id"
        )
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "parent_state_id", parent)
        object.__setattr__(self, "initialized_from_id", initialized)
        object.__setattr__(self, "reset_reason", reset_reason)
        object.__setattr__(self, "source_observation_cut_ids", observations)
        object.__setattr__(self, "source_action_history_ids", actions)
        object.__setattr__(self, "snapshot_blob_id", blob_id)
        payload = {
            "state_id": self.state_id,
            "policy_node_id": self.policy_node_id,
            "policy_artifact_hash": self.policy_artifact_hash,
            "adaptation_contract_hash": self.adaptation_contract_hash,
            "state_kind": self.state_kind,
            "scope": self.scope,
            "updated_at": updated_at,
            "clock_domain_id": self.clock_domain_id,
            "state_content_hash": self.state_content_hash,
            "parent_state_id": parent,
            "initialized_from_id": initialized,
            "reset_reason": reset_reason,
            "source_observation_cut_ids": observations,
            "source_action_history_ids": actions,
            "snapshot_blob_id": blob_id,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=_ADAPTATION_STATE_HASH_DOMAIN_V1),
        )


@dataclass(frozen=True)
class GenerativeActionSampleRecord:
    """Evidence for the actual random input used by one generative inference."""

    sample_id: str
    policy_inference_id: str
    policy_output_id: str
    policy_node_id: str
    base_generator_artifact_hash: str
    action_head_contract_hash: str
    observation_view_content_hash: str
    sampler_kind: str
    sampler_contract_hash: str
    integration_solver: str
    integration_steps: int
    generated_model_action_chunk_id: str
    generated_model_action_chunk_hash: str
    noise_evidence_kind: str
    policy_adaptation_state_before_id: str | None = None
    guidance_contract_hash: str | None = None
    initial_noise_blob_id: str | None = None
    initial_noise_content_hash: str | None = None
    rng_state_blob_id: str | None = None
    rng_state_content_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "policy_inference_id",
            "policy_output_id",
            "policy_node_id",
            "base_generator_artifact_hash",
            "action_head_contract_hash",
            "observation_view_content_hash",
            "sampler_kind",
            "sampler_contract_hash",
            "integration_solver",
            "generated_model_action_chunk_id",
            "generated_model_action_chunk_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        steps = int(self.integration_steps)
        if steps <= 0:
            raise ValueError("integration_steps must be positive.")
        if self.noise_evidence_kind not in GENERATIVE_NOISE_EVIDENCE_KINDS_V1:
            raise ValueError("Unsupported generative noise evidence kind.")
        state_before = _optional(self.policy_adaptation_state_before_id)
        guidance_hash = _optional(self.guidance_contract_hash)
        noise_blob = _optional(self.initial_noise_blob_id)
        noise_hash = _optional(self.initial_noise_content_hash)
        rng_blob = _optional(self.rng_state_blob_id)
        rng_hash = _optional(self.rng_state_content_hash)
        if self.noise_evidence_kind == GENERATIVE_NOISE_ACTUAL:
            if noise_blob is None or noise_hash is None:
                raise ValueError(
                    "actual_initial_noise evidence requires blob ID and content hash."
                )
        elif rng_blob is None or rng_hash is None:
            raise ValueError(
                "rng_receipt_only evidence requires RNG-state blob ID and content hash."
            )
        object.__setattr__(self, "integration_steps", steps)
        object.__setattr__(self, "policy_adaptation_state_before_id", state_before)
        object.__setattr__(self, "guidance_contract_hash", guidance_hash)
        object.__setattr__(self, "initial_noise_blob_id", noise_blob)
        object.__setattr__(self, "initial_noise_content_hash", noise_hash)
        object.__setattr__(self, "rng_state_blob_id", rng_blob)
        object.__setattr__(self, "rng_state_content_hash", rng_hash)
        payload = {
            "sample_id": self.sample_id,
            "policy_inference_id": self.policy_inference_id,
            "policy_output_id": self.policy_output_id,
            "policy_node_id": self.policy_node_id,
            "base_generator_artifact_hash": self.base_generator_artifact_hash,
            "action_head_contract_hash": self.action_head_contract_hash,
            "observation_view_content_hash": self.observation_view_content_hash,
            "policy_adaptation_state_before_id": state_before,
            "sampler_kind": self.sampler_kind,
            "sampler_contract_hash": self.sampler_contract_hash,
            "integration_solver": self.integration_solver,
            "integration_steps": steps,
            "guidance_contract_hash": guidance_hash,
            "noise_evidence_kind": self.noise_evidence_kind,
            "initial_noise_blob_id": noise_blob,
            "initial_noise_content_hash": noise_hash,
            "rng_state_blob_id": rng_blob,
            "rng_state_content_hash": rng_hash,
            "generated_model_action_chunk_id": self.generated_model_action_chunk_id,
            "generated_model_action_chunk_hash": self.generated_model_action_chunk_hash,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=_GENERATIVE_SAMPLE_HASH_DOMAIN_V1),
        )


@dataclass(frozen=True)
class HistoryActionContract:
    """Artifact-specific semantics for action tokens in a history window."""

    contract_id: str
    action_stage: str
    producer_scope: str
    component_ids: tuple[str, ...]
    human_override_policy: str
    missing_action_policy: str
    temporal_alignment_contract_hash: str
    model_action_codec_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "producer_scope",
            "temporal_alignment_contract_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.action_stage not in HISTORY_ACTION_STAGES_V1:
            raise ValueError(f"Unsupported history action stage: {self.action_stage!r}.")
        if self.human_override_policy not in HISTORY_OVERRIDE_POLICIES_V1:
            raise ValueError("Unsupported history human_override_policy.")
        if self.missing_action_policy not in HISTORY_MISSING_ACTION_POLICIES_V1:
            raise ValueError("Unsupported history missing_action_policy.")
        components = _unique_text_tuple(tuple(self.component_ids), name="component_id")
        if not components:
            raise ValueError("HistoryActionContract requires at least one component.")
        codec_hash = _optional(self.model_action_codec_hash)
        if self.action_stage == "policy_output" and codec_hash is None:
            raise ValueError("policy_output history requires model_action_codec_hash.")
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(self, "model_action_codec_hash", codec_hash)
        payload = {
            "contract_id": self.contract_id,
            "action_stage": self.action_stage,
            "producer_scope": self.producer_scope,
            "component_ids": components,
            "human_override_policy": self.human_override_policy,
            "missing_action_policy": self.missing_action_policy,
            "temporal_alignment_contract_hash": self.temporal_alignment_contract_hash,
            "model_action_codec_hash": codec_hash,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload, domain=_HISTORY_ACTION_CONTRACT_HASH_DOMAIN_V1
            ),
        )


@dataclass(frozen=True)
class PolicyActionCodecContract:
    """Identity for model-chunk decode and optional correction encode semantics."""

    contract_id: str
    direction: str
    policy_io_layout_hash: str
    model_action_schema_hash: str
    canonical_action_schema_hash: str
    pose_frame_contract_hash: str
    component_schema_hash: str
    decode_spec_hash: str
    chunk_anchor_spec_hash: str
    horizon_spec_hash: str
    schedule_spec_hash: str
    normalization_spec_hash: str
    encode_spec_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "policy_io_layout_hash",
            "model_action_schema_hash",
            "canonical_action_schema_hash",
            "pose_frame_contract_hash",
            "component_schema_hash",
            "decode_spec_hash",
            "chunk_anchor_spec_hash",
            "horizon_spec_hash",
            "schedule_spec_hash",
            "normalization_spec_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.direction not in POLICY_ACTION_CODEC_DIRECTIONS_V1:
            raise ValueError(f"Unsupported codec direction: {self.direction!r}.")
        encode_hash = _optional(self.encode_spec_hash)
        if self.direction == POLICY_ACTION_CODEC_BIDIRECTIONAL and encode_hash is None:
            raise ValueError("A bidirectional codec requires encode_spec_hash.")
        if self.direction == POLICY_ACTION_CODEC_DECODE_ONLY and encode_hash is not None:
            raise ValueError("A decode-only codec cannot declare encode_spec_hash.")
        object.__setattr__(self, "encode_spec_hash", encode_hash)
        payload = {
            "contract_id": self.contract_id,
            "direction": self.direction,
            "policy_io_layout_hash": self.policy_io_layout_hash,
            "model_action_schema_hash": self.model_action_schema_hash,
            "canonical_action_schema_hash": self.canonical_action_schema_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "component_schema_hash": self.component_schema_hash,
            "decode_spec_hash": self.decode_spec_hash,
            "encode_spec_hash": encode_hash,
            "chunk_anchor_spec_hash": self.chunk_anchor_spec_hash,
            "horizon_spec_hash": self.horizon_spec_hash,
            "schedule_spec_hash": self.schedule_spec_hash,
            "normalization_spec_hash": self.normalization_spec_hash,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=_POLICY_ACTION_CODEC_HASH_DOMAIN_V1),
        )

    @property
    def supports_encode(self) -> bool:
        return self.direction == POLICY_ACTION_CODEC_BIDIRECTIONAL


@dataclass(frozen=True)
class PolicyInferenceObservabilityRecord:
    """Join record binding optional policy-private evidence to one inference."""

    evidence_id: str
    observability_spec_id: str
    observability_spec_hash: str
    policy_inference_id: str
    policy_output_id: str
    observation_cut_id: str
    observation_view_content_hash: str
    policy_node_id: str
    policy_artifact_hash: str
    inference_started_at: float
    inference_completed_at: float
    clock_domain_id: str
    adaptation_state_before_id: str | None = None
    adaptation_state_before_hash: str | None = None
    adaptation_state_after_id: str | None = None
    adaptation_state_after_hash: str | None = None
    generative_sample_id: str | None = None
    generative_sample_hash: str | None = None
    model_action_chunk_id: str | None = None
    model_action_chunk_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "observability_spec_id",
            "observability_spec_hash",
            "policy_inference_id",
            "policy_output_id",
            "observation_cut_id",
            "observation_view_content_hash",
            "policy_node_id",
            "policy_artifact_hash",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        started = _finite(self.inference_started_at, name="inference_started_at")
        completed = _finite(self.inference_completed_at, name="inference_completed_at")
        if completed < started:
            raise ValueError("inference_completed_at must not precede inference_started_at.")
        optional_names = (
            "adaptation_state_before_id",
            "adaptation_state_before_hash",
            "adaptation_state_after_id",
            "adaptation_state_after_hash",
            "generative_sample_id",
            "generative_sample_hash",
            "model_action_chunk_id",
            "model_action_chunk_hash",
        )
        optional_values = {name: _optional(getattr(self, name)) for name in optional_names}
        for left, right in (
            ("adaptation_state_before_id", "adaptation_state_before_hash"),
            ("adaptation_state_after_id", "adaptation_state_after_hash"),
            ("generative_sample_id", "generative_sample_hash"),
            ("model_action_chunk_id", "model_action_chunk_hash"),
        ):
            if (optional_values[left] is None) != (optional_values[right] is None):
                raise ValueError(f"{left} and {right} must be present together.")
        object.__setattr__(self, "inference_started_at", started)
        object.__setattr__(self, "inference_completed_at", completed)
        for name, value in optional_values.items():
            object.__setattr__(self, name, value)
        payload = {
            "evidence_id": self.evidence_id,
            "observability_spec_id": self.observability_spec_id,
            "observability_spec_hash": self.observability_spec_hash,
            "policy_inference_id": self.policy_inference_id,
            "policy_output_id": self.policy_output_id,
            "observation_cut_id": self.observation_cut_id,
            "observation_view_content_hash": self.observation_view_content_hash,
            "policy_node_id": self.policy_node_id,
            "policy_artifact_hash": self.policy_artifact_hash,
            "inference_started_at": started,
            "inference_completed_at": completed,
            "clock_domain_id": self.clock_domain_id,
            **optional_values,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload, domain=_INFERENCE_OBSERVABILITY_HASH_DOMAIN_V1
            ),
        )


def validate_policy_inference_observability(
    record: PolicyInferenceObservabilityRecord,
    *,
    spec: PolicyLearningObservabilitySpec,
    state_before: PolicyAdaptationStateRecord | None = None,
    state_after: PolicyAdaptationStateRecord | None = None,
    generative_sample: GenerativeActionSampleRecord | None = None,
) -> None:
    """Validate one complete inference-evidence join without runtime state."""

    if record.observability_spec_id != spec.spec_id:
        raise ValueError("Inference evidence references a different observability spec ID.")
    if record.observability_spec_hash != spec.semantic_content_hash:
        raise ValueError("Inference evidence observability spec hash mismatch.")
    if record.policy_node_id != spec.policy_node_id:
        raise ValueError("Inference evidence policy node mismatch.")
    if record.policy_artifact_hash != spec.policy_artifact_hash:
        raise ValueError("Inference evidence policy artifact mismatch.")

    if spec.adaptation_state_evidence == ADAPTATION_EVIDENCE_NOT_APPLICABLE:
        if any(
            value is not None
            for value in (
                record.adaptation_state_before_id,
                record.adaptation_state_after_id,
                state_before,
                state_after,
            )
        ):
            raise ValueError("Stateless/not-applicable spec cannot carry adaptation state.")
    else:
        if state_before is None or state_after is None:
            raise ValueError("Required adaptation state evidence is missing.")
        for state in (state_before, state_after):
            if state.policy_node_id != spec.policy_node_id:
                raise ValueError("Adaptation state policy node mismatch.")
            if state.policy_artifact_hash != spec.policy_artifact_hash:
                raise ValueError("Adaptation state artifact mismatch.")
        if record.adaptation_state_before_id != state_before.state_id or (
            record.adaptation_state_before_hash != state_before.semantic_content_hash
        ):
            raise ValueError("Adaptation state-before evidence mismatch.")
        if record.adaptation_state_after_id != state_after.state_id or (
            record.adaptation_state_after_hash != state_after.semantic_content_hash
        ):
            raise ValueError("Adaptation state-after evidence mismatch.")
        if state_after.parent_state_id != state_before.state_id:
            raise ValueError("Adaptation state-after must reference state-before as parent.")

    if spec.generative_sample_evidence == GENERATIVE_EVIDENCE_NOT_APPLICABLE:
        if record.generative_sample_id is not None or generative_sample is not None:
            raise ValueError("Non-generative spec cannot carry generative sample evidence.")
    else:
        if generative_sample is None:
            raise ValueError("Required generative sample evidence is missing.")
        if record.generative_sample_id != generative_sample.sample_id or (
            record.generative_sample_hash != generative_sample.semantic_content_hash
        ):
            raise ValueError("Generative sample evidence mismatch.")
        if generative_sample.policy_inference_id != record.policy_inference_id:
            raise ValueError("Generative sample inference mismatch.")
        if generative_sample.policy_output_id != record.policy_output_id:
            raise ValueError("Generative sample policy output mismatch.")
        if generative_sample.policy_node_id != spec.policy_node_id:
            raise ValueError("Generative sample policy node mismatch.")
        if generative_sample.base_generator_artifact_hash != spec.policy_artifact_hash:
            raise ValueError("Generative sample artifact mismatch.")
        if (
            spec.generative_sample_evidence == GENERATIVE_EVIDENCE_INITIAL_NOISE
            and generative_sample.noise_evidence_kind != GENERATIVE_NOISE_ACTUAL
        ):
            raise ValueError("Initial-noise evidence cannot be satisfied by RNG-only evidence.")

    if spec.model_action_chunk_evidence == MODEL_ACTION_CHUNK_EXACT:
        if record.model_action_chunk_id is None:
            raise ValueError("Required exact model-action chunk evidence is missing.")
    elif record.model_action_chunk_id is not None:
        raise ValueError("Model-action chunk is not applicable for this spec.")


__all__ = [
    "ADAPTATION_EVIDENCE_CHECKPOINTED",
    "ADAPTATION_EVIDENCE_LEVELS_V1",
    "ADAPTATION_EVIDENCE_NOT_APPLICABLE",
    "ADAPTATION_EVIDENCE_STATE_IDENTITY",
    "ADAPTATION_STATE_KINDS_V1",
    "ADAPTATION_STATE_SCOPES_V1",
    "GENERATIVE_EVIDENCE_INITIAL_NOISE",
    "GENERATIVE_EVIDENCE_LEVELS_V1",
    "GENERATIVE_EVIDENCE_NOT_APPLICABLE",
    "GENERATIVE_EVIDENCE_RNG_RECEIPT",
    "GENERATIVE_NOISE_ACTUAL",
    "GENERATIVE_NOISE_EVIDENCE_KINDS_V1",
    "GENERATIVE_NOISE_RNG_ONLY",
    "HISTORY_ACTION_STAGES_V1",
    "HISTORY_MISSING_ACTION_POLICIES_V1",
    "HISTORY_OVERRIDE_POLICIES_V1",
    "LEARNING_OBSERVABILITY_SCHEMA_V1",
    "MODEL_ACTION_CHUNK_EVIDENCE_LEVELS_V1",
    "MODEL_ACTION_CHUNK_EXACT",
    "MODEL_ACTION_CHUNK_NOT_APPLICABLE",
    "POLICY_ACTION_CODEC_BIDIRECTIONAL",
    "POLICY_ACTION_CODEC_DECODE_ONLY",
    "POLICY_ACTION_CODEC_DIRECTIONS_V1",
    "GenerativeActionSampleRecord",
    "HistoryActionContract",
    "PolicyActionCodecContract",
    "PolicyAdaptationStateRecord",
    "PolicyInferenceObservabilityRecord",
    "PolicyLearningObservabilitySpec",
    "validate_policy_inference_observability",
]
