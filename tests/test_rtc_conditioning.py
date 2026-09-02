from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from dataset_contracts import DEXTELEOP_DATASET_SCHEMA_V22, PolicyArtifactContract
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from deployments.openpi.client.rtc_conditioned import (
    OPENPI_RTC_REQUEST_DEADLINE_POLICY_V1,
    _require_request_deadline,
)
from deployments.openpi.training.prepared_model_rtc import (
    PreparedModelRtcCheckpointBinding,
    _checkpoint_binding,
)
from deployments.openpi.artifact_identity import openpi_rtc_artifact_hash
from robot_motion.deploy.inference import (
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyInferencePrefix,
    PolicyInferenceRequest,
)


def _request(*, valid_until: float = 10.0) -> PolicyInferenceRequest:
    return PolicyInferenceRequest(
        request_id="rtc-conditioning-request",
        inference_id="rtc-conditioning-inference",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        session_id="rtc-conditioning-session",
        control_domain_id="rtc-conditioning-control",
        producer_id="rtc-conditioning-producer",
        producer_epoch=0,
        policy_node_id="rtc-conditioning-policy",
        artifact_hash="artifact:test",
        observation_cut_id="observation:test",
        observation_timestamp=0.0,
        observation_step_index=0,
        triggered_at=0.0,
        admitted_at=0.0,
        first_target_effective_at=1.0,
        clock_domain_id="local_monotonic",
        action_period_sec=1.0,
        prediction_horizon_steps=1,
        minimum_execution_horizon_steps=1,
        source_plan_id="plan:test",
        source_plan_generation=0,
        source_plan_hash="plan-hash:test",
        source_plan_cursor_at=0.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay-spec:test",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        valid_until=valid_until,
        conditioning_prefix=PolicyInferencePrefix(
            model_space_actions=np.zeros((1, 1), dtype=np.float32),
            valid_steps=0,
            source_plan_id="plan:test",
            source_plan_generation=0,
            source_plan_hash="plan-hash:test",
        ),
    )


def test_provider_deadline_policy_rejects_before_and_after_the_boundary() -> None:
    request = _request()

    assert OPENPI_RTC_REQUEST_DEADLINE_POLICY_V1.startswith("reject_")
    assert _require_request_deadline(request, now=9.999, boundary="before") == 9.999
    with pytest.raises(ValueError, match="before"):
        _require_request_deadline(request, now=10.0, boundary="before")
    with pytest.raises(ValueError, match="after"):
        _require_request_deadline(request, now=10.0, boundary="after")
    with pytest.raises(ValueError, match="finite"):
        _require_request_deadline(request, now=float("nan"), boundary="before")


def test_checkpoint_binding_is_absolute_and_hashes_only_valid_identities(
    tmp_path: Path,
) -> None:
    binding = PreparedModelRtcCheckpointBinding(
        checkpoint_dir=str(tmp_path / "checkpoint"),
        params_path=str(tmp_path / "checkpoint" / "params"),
        checkpoint_step=42,
        policy_artifact_hash="a" * 64,
        policy_artifact_file_hash="sha256:" + "b" * 64,
        params_metadata_hash="c" * 64,
    )

    assert binding.policy_artifact_hash == "sha256:" + "a" * 64
    assert binding.params_metadata_hash == "sha256:" + "c" * 64
    assert binding.to_dict()["semantic_content_hash"] == (
        binding.semantic_content_hash
    )
    with pytest.raises(ValueError, match="absolute path"):
        PreparedModelRtcCheckpointBinding(
            checkpoint_dir="relative/checkpoint",
            params_path="relative/checkpoint/params",
            checkpoint_step=42,
            policy_artifact_hash="a" * 64,
            policy_artifact_file_hash="b" * 64,
            params_metadata_hash="c" * 64,
        )


def test_checkpoint_binding_fails_closed_before_loading_orbax_data(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="policy_artifact.json"):
        _checkpoint_binding(tmp_path, expected_artifact=object())


def test_checkpoint_binding_verifies_sidecar_and_orbax_commit_markers(
    tmp_path: Path,
) -> None:
    layout = PolicyIOLayout(
        name="rtc-checkpoint-binding-test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )
    expected = PolicyArtifactContract(
        policy_io_layout=layout,
        model_id="rtc-checkpoint-binding-base",
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V22,
    )
    checkpoint = tmp_path / "checkpoint" / "42"
    params = checkpoint / "params"
    params.mkdir(parents=True)
    for name in (
        "_CHECKPOINT_METADATA",
        "_METADATA",
        "_sharding",
        "commit_success.txt",
        "manifest.ocdbt",
    ):
        (params / name).write_bytes(name.encode("utf-8"))
    saved = replace(expected, checkpoint_step=42)
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(saved.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    binding, loaded = _checkpoint_binding(
        checkpoint,
        expected_artifact=expected,
    )

    assert loaded == saved
    assert binding.checkpoint_step == 42
    assert binding.policy_artifact_hash == (
        "sha256:" + openpi_rtc_artifact_hash(saved)
    )
    assert binding.params_path == str(params.resolve())
