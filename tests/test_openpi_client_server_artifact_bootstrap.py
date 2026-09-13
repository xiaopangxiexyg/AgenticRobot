from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from dataset_contracts import (
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    ModelContinuationTrainingContractV2,
)
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client.server_artifact_bootstrap import (
    fetch_server_policy_artifact,
)
from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
)
from teleop_runtime.deploy import openpi_application


def _model_rtc_artifact() -> PolicyArtifactContract:
    layout = PolicyIOLayout.legacy_v21_q6()
    continuation_v1 = ModelContinuationDataContract(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="fixture-flow-inpaint-v1",
        continuation_implementation_parameters_hash="sha256:" + "a" * 64,
        prediction_horizon_steps=16,
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=12,
        minimum_prefix_steps=1,
        maximum_prefix_steps=1,
        model_action_dim=layout.action_dim,
        model_action_discrete_dim=layout.action_discrete_dim,
        component_scope=("left_arm", "right_arm"),
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
        training_prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        runtime_prefix_source_roles=(RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,),
        action_codec_hash="sha256:" + "b" * 64,
        layout_hash=layout.layout_hash,
        frame_contract_hash="frame:model-rtc-server",
        normalization_hash="normalization:model-rtc-server",
        training_projection_implementation_id="fixture-teacher-v1",
        training_projection_parameters_hash="sha256:" + "c" * 64,
    )
    return PolicyArtifactContract(
        policy_io_layout=layout,
        model_id="server-selected-model",
        checkpoint_step=75000,
        camera_input=PolicyCameraInputContract(
            inputs=(
                PolicyCameraInput(
                    source_key="front",
                    model_input_key="observation.images.base_0_rgb",
                    server_camera_slot="base_0_rgb",
                ),
            ),
        ),
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
            runtime=ModelContinuationRuntimeContractV2.from_v1(continuation_v1),
            training=ModelContinuationTrainingContractV2(
                prefix_source_kinds=(
                    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
                ),
                projection_implementation_id="fixture-teacher-v1",
                projection_parameters_hash="sha256:" + "c" * 64,
            ),
        ),
    )


def test_fetch_server_policy_artifact_reads_metadata_and_closes() -> None:
    artifact = _model_rtc_artifact()
    transport = SimpleNamespace(
        metadata={"policy_artifact": artifact.to_dict()},
        close=mock.Mock(),
    )
    factory = mock.Mock(return_value=transport)

    resolved = fetch_server_policy_artifact(
        host="model-host",
        port=8078,
        api_key="secret",
        transport_factory=factory,
    )

    assert resolved == artifact
    factory.assert_called_once_with(
        host="model-host",
        port=8078,
        api_key="secret",
        retry_interval_s=2.0,
    )
    transport.close.assert_called_once_with()


def test_fetch_server_policy_artifact_rejects_missing_artifact_and_closes() -> None:
    transport = SimpleNamespace(metadata={"model": {}}, close=mock.Mock())

    with pytest.raises(ValueError, match="does not publish policy_artifact"):
        fetch_server_policy_artifact(
            host="model-host",
            port=8078,
            transport_factory=mock.Mock(return_value=transport),
        )

    transport.close.assert_called_once_with()


def test_model_rtc_client_bootstraps_server_artifact_without_stage_or_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _model_rtc_artifact()
    bootstrap = mock.Mock(return_value=artifact)
    observed: dict[str, object] = {}

    def _run_policy(args: list[str], **kwargs: object) -> int:
        artifact_index = args.index("--policy-artifact-json") + 1
        runtime_artifact_path = Path(args[artifact_index])
        assert runtime_artifact_path.is_file()
        observed["artifact_path"] = runtime_artifact_path
        observed["artifact"] = json.loads(runtime_artifact_path.read_text())
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return 19

    monkeypatch.setattr(openpi_application, "fetch_server_policy_artifact", bootstrap)
    monkeypatch.setattr("teleop_runtime.deploy.run_policy.main", _run_policy)

    result = openpi_application.main(
        [
            "--host",
            "model-host",
            "--port",
            "8078",
            "--rtc-profile",
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
            "--rtc-real-shadow",
            "--max-steps",
            "1",
        ]
    )

    assert result == 19
    bootstrap.assert_called_once_with(host="model-host", port=8078, api_key=None)
    assert observed["artifact"] == artifact.to_dict()
    assert not Path(observed["artifact_path"]).exists()
    delegated = observed["args"]
    assert isinstance(delegated, list)
    assert "--checkpoint-dir" not in delegated
    assert "--prepared-training-profile" not in delegated
    kwargs_payload = json.loads(
        delegated[delegated.index("--policy-kwargs-json") + 1]
    )
    assert kwargs_payload["policy_artifact"] == artifact.to_dict()
    assert callable(observed["kwargs"]["rtc_model_adapter_factory"])
