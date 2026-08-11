from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

import deployments.openpi.training.launch as training_launch
import deployments.openpi.server as openpi_server
from dataset_contracts import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    POSE_CHUNK_TAIL_PAD_LAST,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    ModelContinuationCheckpointReceipt,
    ModelContinuationTrainingRunManifest,
    ModelContinuationTrainingContractV2,
    ModelContinuationTrainingSourceBinding,
    PolicyArtifactContract,
    PoseActionChunkContract,
    PoseFrameContract,
    PoseSourceProfile,
    PoseTrainingDataContract,
)
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from dataset_interface import (
    TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1,
    TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1,
    ModelContinuationTrainingSamplingSpec,
)
from deployments.openpi.training import (
    OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1,
    OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1,
    OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1,
    OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1,
    OpenPIModelRtcTrainingTransform,
    verify_openpi_model_rtc_checkpoint,
    build_openpi_v22_absolute_model_rtc_artifact_template,
)
from deployments.openpi.client.rtc import openpi_rtc_artifact_hash


def _artifact() -> PolicyArtifactContract:
    layout = PolicyIOLayout(
        name="openpi_model_rtc_training_adapter_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )
    frame = PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="openpi-model-rtc-training-adapter-test",
            source_family="test",
            source_description="fixture",
            topology="dual_arm",
            raw_to_desk_transform_by_side={
                "left": np.eye(4),
                "right": np.eye(4),
            },
        )
    )
    legacy = ModelContinuationDataContract(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="fixture-flow-inpaint-v1",
        continuation_implementation_parameters_hash="sha256:fixture",
        prediction_horizon_steps=4,
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=2,
        minimum_prefix_steps=1,
        maximum_prefix_steps=2,
        model_action_dim=layout.action_dim,
        model_action_discrete_dim=0,
        component_scope=("left_arm", "right_arm"),
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
        training_prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash="sha256:fixture-codec",
        layout_hash=layout.layout_hash,
        frame_contract_hash=frame.contract_hash,
        normalization_hash="sha256:fixture-normalization",
        training_projection_implementation_id=(
            TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
        ),
        training_projection_parameters_hash=(
            TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
        ),
    )
    return PolicyArtifactContract(
        policy_io_layout=layout,
        model_id="fixture-trained-rtc",
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=4,
            action_pose_mode="absolute",
            tail_policy=POSE_CHUNK_TAIL_PAD_LAST,
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(frame.contract_hash,),
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
            runtime=ModelContinuationRuntimeContractV2.from_v1(legacy),
            training=ModelContinuationTrainingContractV2.from_v1(legacy),
        ),
    )


def _source() -> ModelContinuationTrainingSourceBinding:
    return ModelContinuationTrainingSourceBinding(
        source_id="fixture",
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        camera_keys=("front",),
        dataset_identity_hash="1" * 64,
        episode_manifest_hash="2" * 64,
        media_index_hash="3" * 64,
    )


def _transform(*, debug: bool = False) -> OpenPIModelRtcTrainingTransform:
    return OpenPIModelRtcTrainingTransform(
        artifact=_artifact(),
        sampling_spec=ModelContinuationTrainingSamplingSpec(
            delay_prefix_pairs=((1, 1), (2, 1), (2, 2)),
            selection_salt="adapter-test",
        ),
        source_binding=_source(),
        debug=debug,
    )


def _sample() -> dict[str, np.ndarray]:
    return {
        "actions": np.arange(56, dtype=np.float32).reshape(4, 14),
        "_dexteleop_v22_augmentation_key": np.asarray([7, 11]),
        "action_is_pad": np.asarray([False, False, False, False]),
    }


def test_adapter_adds_only_the_optional_loss_mask_by_default() -> None:
    transform = _transform()
    output = transform(_sample())
    assert OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1 in output
    assert OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1 not in output
    assert OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1 not in output
    assert OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1 not in output
    loss_mask = output[OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1]
    assert loss_mask.dtype == np.bool_
    assert loss_mask.shape == (4,)
    assert np.count_nonzero(loss_mask) in (2, 3)


def test_adapter_selection_is_order_independent_and_epoch_explicit() -> None:
    first = _transform()
    second = _transform()
    first.set_epoch(3)
    second.set_epoch(3)
    np.testing.assert_array_equal(
        first(_sample())[OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1],
        second(_sample())[OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1],
    )
    first.set_epoch(4)
    assert first._epoch == 4


def test_debug_keys_are_opt_in_and_consistent() -> None:
    output = _transform(debug=True)(_sample())
    prefix = output[OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1]
    loss = output[OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1]
    np.testing.assert_array_equal(
        np.asarray(prefix, dtype=bool) | loss,
        np.ones(4, dtype=bool),
    )
    assert int(output[OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1]) in (1, 2)
    assert int(output[OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1]) in (1, 2)


def test_adapter_rejects_missing_identity_and_noncontiguous_padding() -> None:
    missing = _sample()
    missing.pop("_dexteleop_v22_augmentation_key")
    with pytest.raises(ValueError, match="augmentation_key"):
        _transform()(missing)

    bad_padding = _sample()
    bad_padding["action_is_pad"] = np.asarray(
        [False, True, False, True]
    )
    with pytest.raises(ValueError, match="contiguous false-prefix"):
        _transform()(bad_padding)


def test_adapter_rejects_inference_only_artifact() -> None:
    trained = _artifact()
    continuation = trained.model_continuation
    assert isinstance(continuation, ModelContinuationDataContractV2)
    inference_only = replace(
        trained,
        model_continuation=replace(
            continuation,
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
            training=None,
        ),
    )
    with pytest.raises(ValueError, match="trained"):
        OpenPIModelRtcTrainingTransform(
            artifact=inference_only,
            sampling_spec=ModelContinuationTrainingSamplingSpec(
                delay_prefix_pairs=((1, 1),),
                selection_salt="adapter-test",
            ),
            source_binding=_source(),
        )


def test_checkpoint_evidence_binds_actual_artifact_and_template(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "120"
    checkpoint_dir.mkdir()
    template = _artifact()
    actual = replace(template, checkpoint_step=120)
    run = ModelContinuationTrainingRunManifest(
        output_model_id=str(template.model_id),
        policy_artifact_template_hash=openpi_rtc_artifact_hash(template),
        base_checkpoint_identity_hash="4" * 64,
        sampling_spec_hash="5" * 64,
        trainer_implementation_id="fixture-trainer-v1",
        trainer_parameters_hash="6" * 64,
        train_config_hash="7" * 64,
        source_bindings=(_source(),),
    )
    receipt = ModelContinuationCheckpointReceipt(
        training_run_manifest_hash=run.semantic_content_hash,
        policy_artifact_hash=openpi_rtc_artifact_hash(actual),
        model_id=str(template.model_id),
        checkpoint_step=120,
    )
    (checkpoint_dir / "policy_artifact.json").write_text(
        json.dumps(actual.to_dict()),
        encoding="utf-8",
    )
    (
        checkpoint_dir / "model_continuation_training_run.json"
    ).write_text(json.dumps(run.to_dict()), encoding="utf-8")
    (
        checkpoint_dir / "model_continuation_checkpoint_receipt.json"
    ).write_text(json.dumps(receipt.to_dict()), encoding="utf-8")

    evidence = verify_openpi_model_rtc_checkpoint(checkpoint_dir)
    assert evidence.artifact == actual
    assert evidence.training_run == run
    assert evidence.checkpoint_receipt == receipt

    tampered = replace(actual, metadata={"tampered": True})
    (checkpoint_dir / "policy_artifact.json").write_text(
        json.dumps(tampered.to_dict()),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="actual policy artifact"):
        verify_openpi_model_rtc_checkpoint(checkpoint_dir)


def test_trained_artifact_derivation_is_additive_and_v22_compatible(
    tmp_path: Path,
) -> None:
    trained = _artifact()
    continuation = trained.model_continuation
    assert isinstance(continuation, ModelContinuationDataContractV2)
    source = PolicyArtifactContract(
        policy_io_layout=trained.policy_io_layout,
        model_id="fixture-base",
        checkpoint_step=100,
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
            runtime=continuation.runtime,
        ),
    )
    source_payload = source.to_dict()
    template = build_openpi_v22_absolute_model_rtc_artifact_template(
        source,
        output_model_id="fixture-trained-output",
        source_dataset_root=tmp_path,
    )
    assert source.to_dict() == source_payload
    assert template.model_id == "fixture-trained-output"
    assert template.checkpoint_step is None
    assert template.dataset_schema == DEXTELEOP_DATASET_SCHEMA_V23
    assert template.pose_action_chunk is not None
    assert template.pose_action_chunk.action_pose_mode == "absolute"
    assert template.pose_training_data is not None
    assert template.pose_training_data.source_dataset_schema == (
        "dexteleop_dataset_v2.2"
    )
    normalized = template.model_continuation
    assert isinstance(normalized, ModelContinuationDataContractV2)
    assert (
        normalized.conditioning_origin
        == MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1
    )
    assert normalized.training is not None


def test_training_entrypoint_registers_future_dataclass_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "openpi"
    package_path = checkout / "src" / "openpi" / "__init__.py"
    package_path.parent.mkdir(parents=True)
    package_path.write_text("", encoding="utf-8")
    train_path = checkout / "scripts" / "train.py"
    train_path.parent.mkdir()
    train_path.write_text(
        """from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Args:
    value: int = 17


def main(config):
    return config
""",
        encoding="utf-8",
    )
    openpi_module = ModuleType("openpi")
    openpi_module.__file__ = str(package_path)
    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.delitem(sys.modules, "_dexteleop_openpi_train", raising=False)

    train_main, resolved_checkout = training_launch._openpi_train_main()

    loaded = sys.modules["_dexteleop_openpi_train"]
    assert resolved_checkout == checkout
    assert train_main is loaded.main
    assert loaded.Args().value == 17


def test_training_entrypoint_uses_openpi_checkout_cwd_and_restores_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "openpi"
    caller = tmp_path / "caller"
    checkout.mkdir()
    caller.mkdir()
    observed: list[tuple[Path, object]] = []
    config = object()

    def fake_train_main(received: object) -> None:
        observed.append((Path.cwd(), received))

    monkeypatch.setattr(
        training_launch,
        "_openpi_train_main",
        lambda: (fake_train_main, checkout),
    )
    monkeypatch.chdir(caller)

    training_launch._run_openpi_train(config)

    assert observed == [(checkout, config)]
    assert Path.cwd() == caller


def test_checkpoint_server_rtc_continuation_is_explicit_and_default_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    openpi_root = tmp_path / "openpi"
    (openpi_root / "scripts").mkdir(parents=True)
    (openpi_root / "configs").mkdir()
    (openpi_root / "scripts" / "serve_dexteleop_policy.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    (openpi_root / "configs" / "dexteleop_policy_registry.json").write_text(
        '{"models": {}}\n',
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint" / "10"
    (checkpoint / "params").mkdir(parents=True)
    artifact = replace(_artifact(), checkpoint_step=10)
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(artifact.to_dict()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        openpi_server.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    assert (
        openpi_server.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--openpi-root",
                str(openpi_root),
                "--config",
                "fixture-base",
                "--dry-run",
            ]
        )
        == 0
    )
    default_command = json.loads(capsys.readouterr().out)["command"]
    assert "--enable-rtc-continuation" not in default_command

    assert (
        openpi_server.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--openpi-root",
                str(openpi_root),
                "--config",
                "fixture-base",
                "--enable-rtc-continuation",
                "--dry-run",
            ]
        )
        == 0
    )
    enabled_command = json.loads(capsys.readouterr().out)["command"]
    assert enabled_command[-1] == "--enable-rtc-continuation"
