from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.learning_observability import (
    ADAPTATION_EVIDENCE_STATE_IDENTITY,
    GENERATIVE_EVIDENCE_INITIAL_NOISE,
    GENERATIVE_EVIDENCE_RNG_RECEIPT,
    GENERATIVE_NOISE_ACTUAL,
    GENERATIVE_NOISE_RNG_ONLY,
    MODEL_ACTION_CHUNK_EXACT,
    POLICY_ACTION_CODEC_BIDIRECTIONAL,
    POLICY_ACTION_CODEC_DECODE_ONLY,
    GenerativeActionSampleRecord,
    HistoryActionContract,
    PolicyActionCodecContract,
    PolicyAdaptationStateRecord,
    PolicyInferenceObservabilityRecord,
    PolicyLearningObservabilitySpec,
    validate_policy_inference_observability,
)


def codec(*, bidirectional: bool = False) -> PolicyActionCodecContract:
    return PolicyActionCodecContract(
        contract_id="v23-relative-pose-codec",
        direction=(
            POLICY_ACTION_CODEC_BIDIRECTIONAL
            if bidirectional
            else POLICY_ACTION_CODEC_DECODE_ONLY
        ),
        policy_io_layout_hash="layout-hash",
        model_action_schema_hash="model-action-hash",
        canonical_action_schema_hash="robot-action-hash",
        pose_frame_contract_hash="desk-frame-hash",
        component_schema_hash="dual-arm-components-hash",
        decode_spec_hash="decode-v23-hash",
        encode_spec_hash="encode-v23-hash" if bidirectional else None,
        chunk_anchor_spec_hash="anchor-hash",
        horizon_spec_hash="horizon-hash",
        schedule_spec_hash="schedule-hash",
        normalization_spec_hash="normalization-hash",
    )


def state(
    state_id: str,
    *,
    parent_state_id: str | None,
    updated_at: float,
    snapshot_blob_id: str | None = None,
) -> PolicyAdaptationStateRecord:
    return PolicyAdaptationStateRecord(
        state_id=state_id,
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
        adaptation_contract_hash="fast-weight-contract-hash",
        state_kind="fast_weights",
        scope="episode",
        updated_at=updated_at,
        clock_domain_id="policy.monotonic",
        state_content_hash=f"content-{state_id}",
        parent_state_id=parent_state_id,
        initialized_from_id="initial-state" if parent_state_id is None else None,
        reset_reason="episode_start" if parent_state_id is None else None,
        source_observation_cut_ids=("cut-0",) if parent_state_id is not None else (),
        source_action_history_ids=("history-0",) if parent_state_id is not None else (),
        snapshot_blob_id=snapshot_blob_id,
    )


def sample(*, kind: str = GENERATIVE_NOISE_ACTUAL) -> GenerativeActionSampleRecord:
    return GenerativeActionSampleRecord(
        sample_id="sample-0",
        policy_inference_id="inference-0",
        policy_output_id="output-0",
        policy_node_id="policy-0",
        base_generator_artifact_hash="artifact-hash",
        action_head_contract_hash="action-head-hash",
        observation_view_content_hash="view-hash",
        policy_adaptation_state_before_id="state-before",
        sampler_kind="flow_matching",
        sampler_contract_hash="sampler-hash",
        integration_solver="euler",
        integration_steps=10,
        guidance_contract_hash="guidance-hash",
        noise_evidence_kind=kind,
        initial_noise_blob_id="blob:noise" if kind == GENERATIVE_NOISE_ACTUAL else None,
        initial_noise_content_hash="noise-hash" if kind == GENERATIVE_NOISE_ACTUAL else None,
        rng_state_blob_id="blob:rng" if kind == GENERATIVE_NOISE_RNG_ONLY else None,
        rng_state_content_hash="rng-hash" if kind == GENERATIVE_NOISE_RNG_ONLY else None,
        generated_model_action_chunk_id="model-chunk-0",
        generated_model_action_chunk_hash="model-chunk-hash",
    )


def inference_record(
    spec: PolicyLearningObservabilitySpec,
    before: PolicyAdaptationStateRecord,
    after: PolicyAdaptationStateRecord,
    generated: GenerativeActionSampleRecord,
) -> PolicyInferenceObservabilityRecord:
    return PolicyInferenceObservabilityRecord(
        evidence_id="inference-evidence-0",
        observability_spec_id=spec.spec_id,
        observability_spec_hash=spec.semantic_content_hash,
        policy_inference_id="inference-0",
        policy_output_id="output-0",
        observation_cut_id="cut-0",
        observation_view_content_hash="view-hash",
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
        inference_started_at=10.0,
        inference_completed_at=10.02,
        clock_domain_id="policy.monotonic",
        adaptation_state_before_id=before.state_id,
        adaptation_state_before_hash=before.semantic_content_hash,
        adaptation_state_after_id=after.state_id,
        adaptation_state_after_hash=after.semantic_content_hash,
        generative_sample_id=generated.sample_id,
        generative_sample_hash=generated.semantic_content_hash,
        model_action_chunk_id="model-chunk-0",
        model_action_chunk_hash="model-chunk-hash",
    )


def test_stateless_non_generative_spec_requires_no_fake_evidence() -> None:
    spec = PolicyLearningObservabilitySpec(
        spec_id="plain-policy-v1",
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
    )
    record = PolicyInferenceObservabilityRecord(
        evidence_id="evidence-0",
        observability_spec_id=spec.spec_id,
        observability_spec_hash=spec.semantic_content_hash,
        policy_inference_id="inference-0",
        policy_output_id="output-0",
        observation_cut_id="cut-0",
        observation_view_content_hash="view-hash",
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
        inference_started_at=1.0,
        inference_completed_at=1.1,
        clock_domain_id="policy.monotonic",
    )
    validate_policy_inference_observability(record, spec=spec)

    with pytest.raises(ValueError, match="Non-generative"):
        validate_policy_inference_observability(
            replace(
                record,
                generative_sample_id="sample-0",
                generative_sample_hash="sample-hash",
            ),
            spec=spec,
        )


def test_stateful_generative_inference_has_one_joinable_evidence_chain() -> None:
    codec_contract = codec()
    history = HistoryActionContract(
        contract_id="proposal-and-behavior-history-v1",
        action_stage="policy_output",
        producer_scope="policy:policy-0",
        component_ids=("left_arm", "right_arm"),
        human_override_policy="use_both_with_source_token",
        missing_action_policy="mask_token",
        temporal_alignment_contract_hash="history-alignment-hash",
        model_action_codec_hash=codec_contract.semantic_content_hash,
    )
    spec = PolicyLearningObservabilitySpec(
        spec_id="stateful-flow-policy-v1",
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
        adaptation_state_evidence=ADAPTATION_EVIDENCE_STATE_IDENTITY,
        generative_sample_evidence=GENERATIVE_EVIDENCE_INITIAL_NOISE,
        model_action_chunk_evidence=MODEL_ACTION_CHUNK_EXACT,
        history_action_contract_hashes=(history.semantic_content_hash,),
        policy_action_codec_hash=codec_contract.semantic_content_hash,
    )
    before = state("state-before", parent_state_id=None, updated_at=9.9)
    after = state("state-after", parent_state_id=before.state_id, updated_at=10.02)
    generated = sample()
    record = inference_record(spec, before, after, generated)

    validate_policy_inference_observability(
        record,
        spec=spec,
        state_before=before,
        state_after=after,
        generative_sample=generated,
    )
    assert history.semantic_content_hash in spec.history_action_contract_hashes
    assert not codec_contract.supports_encode


def test_rng_receipt_cannot_satisfy_initial_noise_requirement() -> None:
    codec_contract = codec()
    spec = PolicyLearningObservabilitySpec(
        spec_id="exact-noise-policy-v1",
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
        adaptation_state_evidence=ADAPTATION_EVIDENCE_STATE_IDENTITY,
        generative_sample_evidence=GENERATIVE_EVIDENCE_INITIAL_NOISE,
        model_action_chunk_evidence=MODEL_ACTION_CHUNK_EXACT,
        policy_action_codec_hash=codec_contract.semantic_content_hash,
    )
    before = state("state-before", parent_state_id=None, updated_at=9.9)
    after = state("state-after", parent_state_id=before.state_id, updated_at=10.02)
    generated = sample(kind=GENERATIVE_NOISE_RNG_ONLY)
    record = inference_record(spec, before, after, generated)

    with pytest.raises(ValueError, match="RNG-only"):
        validate_policy_inference_observability(
            record,
            spec=spec,
            state_before=before,
            state_after=after,
            generative_sample=generated,
        )

    weaker_spec = replace(
        spec,
        spec_id="rng-evidence-policy-v1",
        generative_sample_evidence=GENERATIVE_EVIDENCE_RNG_RECEIPT,
    )
    weaker_record = replace(
        record,
        observability_spec_id=weaker_spec.spec_id,
        observability_spec_hash=weaker_spec.semantic_content_hash,
    )
    validate_policy_inference_observability(
        weaker_record,
        spec=weaker_spec,
        state_before=before,
        state_after=after,
        generative_sample=generated,
    )


def test_generative_evidence_cannot_omit_generated_model_chunk() -> None:
    with pytest.raises(ValueError, match="exact generated model-action chunk"):
        PolicyLearningObservabilitySpec(
            spec_id="incomplete-generative-policy-v1",
            policy_node_id="policy-0",
            policy_artifact_hash="artifact-hash",
            generative_sample_evidence=GENERATIVE_EVIDENCE_RNG_RECEIPT,
        )


def test_codec_direction_and_history_semantics_are_explicit() -> None:
    decode_only = codec()
    assert not decode_only.supports_encode
    with pytest.raises(ValueError, match="requires encode_spec_hash"):
        replace(
            decode_only,
            direction=POLICY_ACTION_CODEC_BIDIRECTIONAL,
            encode_spec_hash=None,
        )

    bidirectional = codec(bidirectional=True)
    assert bidirectional.supports_encode
    proposal_history = HistoryActionContract(
        contract_id="proposal-history-v1",
        action_stage="decoded_proposal",
        producer_scope="policy:policy-0",
        component_ids=("left_arm",),
        human_override_policy="use_policy_counterfactual",
        missing_action_policy="mask_token",
        temporal_alignment_contract_hash="alignment-hash",
    )
    behavior_history = replace(
        proposal_history,
        contract_id="behavior-history-v1",
        action_stage="safe",
        producer_scope="selected-behavior",
        human_override_policy="use_human_behavior",
    )
    assert proposal_history.semantic_content_hash != behavior_history.semantic_content_hash


def test_inference_join_rejects_wrong_state_parent_or_artifact() -> None:
    codec_contract = codec()
    spec = PolicyLearningObservabilitySpec(
        spec_id="stateful-policy-v1",
        policy_node_id="policy-0",
        policy_artifact_hash="artifact-hash",
        adaptation_state_evidence=ADAPTATION_EVIDENCE_STATE_IDENTITY,
        generative_sample_evidence=GENERATIVE_EVIDENCE_INITIAL_NOISE,
        model_action_chunk_evidence=MODEL_ACTION_CHUNK_EXACT,
        policy_action_codec_hash=codec_contract.semantic_content_hash,
    )
    before = state("state-before", parent_state_id=None, updated_at=9.9)
    after = state("state-after", parent_state_id="different-state", updated_at=10.02)
    generated = sample()
    record = inference_record(spec, before, after, generated)

    with pytest.raises(ValueError, match="state-before as parent"):
        validate_policy_inference_observability(
            record,
            spec=spec,
            state_before=before,
            state_after=after,
            generative_sample=generated,
        )

    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_policy_inference_observability(
            replace(record, policy_artifact_hash="other-artifact"),
            spec=spec,
            state_before=before,
            state_after=replace(after, parent_state_id=before.state_id),
            generative_sample=generated,
        )
