from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    ModelContinuationTrainingContractV2,
)
from dataset_contracts.model_continuation_training import (
    ModelContinuationCheckpointReceipt,
    ModelContinuationTrainingRunManifest,
    ModelContinuationTrainingSourceBinding,
)
from dataset_contracts.normalization import (
    NormalizationContract,
    normalization_stats_hash,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.v22 import PolicyIOLayout
from dataset_interface.model_continuation import (
    TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1,
    TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1,
    ModelContinuationTrainingSamplingSpec,
)
from deployments.openpi import prepared_deployment, prepared_server_runtime
from deployments.openpi import server as openpi_server
from deployments.openpi.artifact_identity import openpi_rtc_artifact_hash
from deployments.openpi.client import real as openpi_real
from deployments.openpi.client.normalization import (
    make_openpi_quantile_normalization_contract,
)
from deployments.openpi.client.observation_adapter import (
    prepared_allowed_prompts,
    prepared_prompt_for_stage,
    resolve_prepared_prompt_selection,
)
from teleop_runtime.backends.grippers import DMTacGripperBackend
from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy.observation import build_deployment_observation
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _fixture(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    openpi_root = root / "openpi"
    source_root = root / "dataset"
    (source_root / "meta").mkdir(parents=True)
    (source_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "policy_io_layout_hash": (
                    PolicyIOLayout.legacy_v21_q6().layout_hash
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode_manifest = root / "episode_manifest.jsonl"
    episode_manifest.write_text("{}\n", encoding="utf-8")
    profile = {
        "schema": "dexteleop.openpi_prepared_skill_training_profile.v1",
        "profile_id": "prepared-front-absolute-v1",
        "organized_dataset_root": str(root / "workspace"),
        "sources": [
            {
                "source_dataset_id": "source-a",
                "dataset_root": str(source_root),
                "expected_eligible_episode_count": 1,
            }
        ],
        "target_skill_id": "writing.fixture.stage1",
        "camera_slot_to_dataset_key": {"base_0_rgb": "front"},
        "action_horizon": 16,
        "action_pose_mode": "absolute",
        "source_mixing_policy": "natural_concat_shuffle_v1",
        "tail_policy": "drop",
        "expected_policy_io_layout_hash": PolicyIOLayout.legacy_v21_q6().layout_hash,
        "openpi_checkout": str(openpi_root),
        "openpi_base_config": "fixture-base",
        "openpi_state_augmentation_from_config": "fixture-augmentation",
        "openpi_config_name": "prepared-model",
        "openpi_asset_id": "prepared_model_assets",
        "openpi_num_workers": 4,
        "default_exp_name": "prepared-exp",
        "default_prompt": "fixture prepared prompt",
    }
    profile_path = root / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    checkpoint = openpi_root / "checkpoints" / "prepared-model" / "150000"
    (checkpoint / "params").mkdir(parents=True)
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="prepared-model",
        checkpoint_step=150000,
        dataset_schema="dexteleop_dataset_v2.3",
        camera_input=PolicyCameraInputContract(
            inputs=(
                PolicyCameraInput(
                    source_key="front",
                    model_input_key="observation.images.front",
                    server_camera_slot="base_0_rgb",
                ),
            )
        ),
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=16,
            action_pose_mode="absolute",
            tail_policy="drop",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=("sha256:" + "1" * 64,),
    )
    payload = artifact.to_dict()
    payload["prepared_skill_training_binding"] = {
        "skill_id": "writing.fixture.stage1",
        "action_horizon": 16,
        "action_pose_mode": "absolute",
        "camera_slot_to_dataset_key": [
            {
                "model_slot": "base_0_rgb",
                "dataset_camera_key": "front",
            }
        ],
        "binding_hash": "fixture-binding",
    }
    payload["dataset_mixing"] = {
        "sources": [
            {
                "name": "source-a",
                "dataset_root": str(source_root),
                "sample_weight": 1,
                "front_occlusion_schedule": [
                    {"ratio": 0.0, "probability": 1.0}
                ],
                "explicit_mask_keys": ["final_mask_paths"],
                "episode_manifest": str(episode_manifest),
            }
        ]
    }
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    stats_path = (
        checkpoint
        / "assets"
        / "prepared_model_assets"
        / "norm_stats.json"
    )
    stats_path.parent.mkdir(parents=True)
    stats_path.write_bytes(b'{"norm_stats":{"state":{},"actions":{}}}\n')
    return checkpoint, profile_path, openpi_root, profile


def test_state_only_artifact_has_no_external_camera_but_keeps_prepared_camera_provenance(
    tmp_path: Path,
) -> None:
    _checkpoint, _profile_path, _openpi_root, profile = _fixture(tmp_path)
    profile["openpi_observation_mode"] = "state_only_masked_dummy_v1"
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id=str(profile["openpi_config_name"]),
        checkpoint_step=60_000,
        dataset_schema="dexteleop_dataset_v2.3",
        camera_input=PolicyCameraInputContract(inputs=(), required=False),
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=16,
            action_pose_mode="absolute",
            tail_policy="drop",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=("sha256:" + "1" * 64,),
    )

    prepared_deployment._validate_artifact_contract(artifact, profile)
    bridge = prepared_deployment.OpenPIStateOnlyInputBridge.from_profile_value(
        profile["openpi_observation_mode"],
        image_keys=("base_0_rgb",),
    )
    assert bridge is not None
    prepared_deployment._validate_state_only_training_metadata(
        {
            prepared_deployment.OPENPI_STATE_ONLY_CAPABILITY_KEY_V1: (
                bridge.contract
            )
        },
        profile,
    )


def _model_rtc_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    checkpoint, profile_path, openpi_root, profile = _fixture(root)
    artifact_path = checkpoint / "policy_artifact.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    base_artifact = PolicyArtifactContract.from_dict(payload)
    stats_path = (
        checkpoint
        / "assets"
        / str(profile["openpi_asset_id"])
        / "norm_stats.json"
    )
    normalization = make_openpi_quantile_normalization_contract(
        stats_ref=(
            "checkpoint://assets/"
            f"{profile['openpi_asset_id']}/norm_stats.json"
        ),
        stats_hash=normalization_stats_hash(stats_path.read_bytes()),
    )
    parsed_normalization = NormalizationContract.from_dict(normalization)
    runtime = ModelContinuationRuntimeContractV2(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="fixture-flow-prefix-v1",
        continuation_implementation_parameters_hash="sha256:fixture-parameters",
        prediction_horizon_steps=16,
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=12,
        minimum_prefix_steps=0,
        maximum_prefix_steps=12,
        model_action_dim=base_artifact.policy_io_layout.action_dim,
        model_action_discrete_dim=(
            base_artifact.policy_io_layout.action_discrete_dim
        ),
        component_scope=("left.arm.pose", "right.arm.pose"),
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash="sha256:fixture-codec",
        layout_hash=base_artifact.layout_hash,
        frame_contract_hash=base_artifact.pose_frame_contract_hashes[0],
        normalization_hash=parsed_normalization.contract_hash,
        supports_unconditioned_bootstrap=True,
    )
    training = ModelContinuationTrainingContractV2(
        prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        projection_implementation_id=(
            TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
        ),
        projection_parameters_hash=(
            TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
        ),
    )
    model_id = "prepared-model-rtc"
    source_root = Path(str(profile["sources"][0]["dataset_root"])).resolve()
    trained = dataclasses.replace(
        base_artifact,
        model_id=model_id,
        source_datasets=(str(source_root),),
        metadata={
            **dict(base_artifact.metadata),
            "normalization_contract": normalization,
        },
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
            runtime=runtime,
            training=training,
        ),
    )
    trained_payload = trained.to_dict()
    trained = PolicyArtifactContract.from_dict(trained_payload)
    artifact_path.write_text(json.dumps(trained_payload), encoding="utf-8")

    sampling = ModelContinuationTrainingSamplingSpec(
        delay_prefix_pairs=((1, 1), (4, 2), (8, 8), (12, 12)),
        selection_salt="prepared-model-rtc-fixture-v1",
    )
    model_profile = {
        "schema": "dexteleop.openpi_prepared_model_rtc_training_profile.v1",
        "profile_id": "prepared-model-rtc-profile-v1",
        "prepared_skill_profile_id": profile["profile_id"],
        "output_config_name": model_id,
        "default_exp_name": "prepared-model-rtc-exp",
        "minimum_forecast_delay_steps": 1,
        "maximum_forecast_delay_steps": 12,
        "minimum_prefix_steps": 0,
        "maximum_prefix_steps": 12,
        "sampling_spec": sampling.to_dict(),
    }
    model_profile_path = root / "model_rtc_profile.json"
    model_profile_path.write_text(json.dumps(model_profile), encoding="utf-8")
    binding_hash = "a" * 64
    binding_root = (
        Path(str(profile["organized_dataset_root"]))
        / "derived"
        / "openpi_prepared_skill_bindings"
        / binding_hash
    )
    binding_root.mkdir(parents=True)
    training_view_hash = "b" * 64
    binding_payload = {
        "schema": "dexteleop.openpi_prepared_skill_binding.v1",
        "training_view_hash": training_view_hash,
        "skill_id": profile["target_skill_id"],
        "skill_training_contract_hash": "c" * 64,
        "training_domain": "NOMINAL",
        "camera_slot_to_dataset_key": [
            {
                "model_slot": "base_0_rgb",
                "dataset_camera_key": "front",
            }
        ],
        "action_horizon": 16,
        "action_pose_mode": "absolute",
        "source_mixing_policy": "natural_concat_shuffle_v1",
        "episode_count": 1,
        "window_count": 1,
        "episodes": [
            {
                "source_dataset_id": "source-a",
                "source_episode_index": 0,
                "organized_episode_ordinal": 0,
                "episode_capture_identity_hash": "d" * 64,
                "intervals": [
                    {"sample_begin": 0, "sample_end_exclusive": 16}
                ],
            }
        ],
        "adapter_implementation_id": (
            "deployments.openpi.prepared_skill_interval_adapter.v2"
        ),
        "binding_hash": binding_hash,
    }
    source_manifest_name = "source_00_source-a.jsonl"
    source_manifest = binding_root / source_manifest_name
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "dexteleop_logical_episode_ref_v1",
                "source_dataset_root": str(source_root),
                "source_episode_index": 0,
                "episode_identity": "d" * 64,
                "camera_slots": ["base_0_rgb"],
                "view_profile": "prepared_skill",
                "quality_class": "strict_success_non_abnormal",
                "prepared_training_view_hash": training_view_hash,
                "openpi_prepared_skill_binding_hash": binding_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = binding_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "binding": binding_payload,
                "source_manifests": [
                    {
                        "source_dataset_id": "source-a",
                        "source_dataset_root": str(source_root),
                        "episode_manifest": source_manifest_name,
                        "episode_count": 1,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source = ModelContinuationTrainingSourceBinding(
        source_id=str(model_profile["profile_id"]),
        dataset_schema="dexteleop_dataset_v2.3",
        camera_keys=("front",),
        dataset_identity_hash="sha256:" + binding_hash,
        episode_manifest_hash=(
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ),
        media_index_hash="3" * 64,
    )
    template = dataclasses.replace(trained, checkpoint_step=None)
    run = ModelContinuationTrainingRunManifest(
        output_model_id=model_id,
        policy_artifact_template_hash=openpi_rtc_artifact_hash(template),
        base_checkpoint_identity_hash="4" * 64,
        sampling_spec_hash=sampling.semantic_content_hash,
        trainer_implementation_id="fixture-trainer-v1",
        trainer_parameters_hash="5" * 64,
        train_config_hash="6" * 64,
        source_bindings=(source,),
    )
    receipt = ModelContinuationCheckpointReceipt(
        training_run_manifest_hash=run.semantic_content_hash,
        policy_artifact_hash=openpi_rtc_artifact_hash(trained),
        model_id=model_id,
        checkpoint_step=150000,
    )
    (checkpoint / "model_continuation_training_run.json").write_text(
        json.dumps(run.to_dict()),
        encoding="utf-8",
    )
    (checkpoint / "model_continuation_checkpoint_receipt.json").write_text(
        json.dumps(receipt.to_dict()),
        encoding="utf-8",
    )
    return checkpoint, profile_path, model_profile_path, openpi_root, profile


def test_prepared_deployment_artifact_binds_profile_and_stats_without_mutation(
    tmp_path: Path,
) -> None:
    checkpoint, profile_path, _openpi_root, profile = _fixture(tmp_path)
    checkpoint_artifact = checkpoint / "policy_artifact.json"
    before = checkpoint_artifact.read_bytes()

    result = prepared_deployment.prepare_checkpoint_deployment_artifact(
        checkpoint,
        profile_path,
        output_dir=tmp_path / "runtime",
    )

    assert checkpoint_artifact.read_bytes() == before
    assert result.deployment_artifact_path != checkpoint_artifact
    assert result.binding["openpi_base_config"] == profile["openpi_base_config"]
    assert result.binding["openpi_config_name"] == profile["openpi_config_name"]
    assert result.binding["openpi_asset_id"] == profile["openpi_asset_id"]
    assert result.binding["norm_stats_hash"] == normalization_stats_hash(
        result.norm_stats_path.read_bytes()
    )
    contract = NormalizationContract.from_dict(
        result.deployment_artifact.metadata["normalization_contract"]
    )
    assert contract.contract_hash
    assert contract.state.stats_hash == result.binding["norm_stats_hash"]
    assert contract.action.stats_hash == result.binding["norm_stats_hash"]

    validated = prepared_deployment.validate_prepared_deployment_artifact(
        checkpoint,
        profile_path,
        result.deployment_artifact_path,
    )
    assert validated.binding == result.binding


def test_prepared_deployment_rejects_stats_or_profile_drift(
    tmp_path: Path,
) -> None:
    checkpoint, profile_path, _openpi_root, _profile = _fixture(tmp_path)
    result = prepared_deployment.prepare_checkpoint_deployment_artifact(
        checkpoint,
        profile_path,
        output_dir=tmp_path / "runtime",
    )
    result.norm_stats_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="differs"):
        prepared_deployment.validate_prepared_deployment_artifact(
            checkpoint,
            profile_path,
            result.deployment_artifact_path,
        )


def test_prepared_model_rtc_deployment_binds_profile_receipt_and_stats(
    tmp_path: Path,
) -> None:
    checkpoint, profile_path, model_profile_path, _openpi_root, profile = (
        _model_rtc_fixture(tmp_path)
    )
    checkpoint_artifact_path = checkpoint / "policy_artifact.json"
    before = checkpoint_artifact_path.read_bytes()

    result = prepared_deployment.prepare_checkpoint_deployment_artifact(
        checkpoint,
        profile_path,
        model_rtc_profile_path=model_profile_path,
        output_dir=tmp_path / "runtime",
    )

    assert checkpoint_artifact_path.read_bytes() == before
    assert result.openpi_config_name == "prepared-model-rtc"
    assert result.binding["schema"] == (
        prepared_deployment.PREPARED_MODEL_RTC_DEPLOYMENT_BINDING_SCHEMA_V1
    )
    assert result.binding["prepared_openpi_config_name"] == (
        profile["openpi_config_name"]
    )
    assert result.binding["model_rtc_profile_id"] == (
        "prepared-model-rtc-profile-v1"
    )
    assert result.binding["training_run_manifest_hash"].startswith("sha256:")
    assert result.binding["checkpoint_receipt_hash"].startswith("sha256:")
    assert (
        result.deployment_artifact.metadata["normalization_contract"]
        == result.checkpoint_artifact.metadata["normalization_contract"]
    )

    validated = prepared_deployment.validate_prepared_deployment_artifact(
        checkpoint,
        profile_path,
        result.deployment_artifact_path,
        model_rtc_profile_path=model_profile_path,
    )
    assert validated.binding == result.binding


def test_prepared_model_rtc_deployment_rejects_profile_sampling_drift(
    tmp_path: Path,
) -> None:
    checkpoint, profile_path, model_profile_path, _openpi_root, _profile = (
        _model_rtc_fixture(tmp_path)
    )
    model_profile = json.loads(model_profile_path.read_text(encoding="utf-8"))
    model_profile["sampling_spec"]["selection_salt"] = "drift"
    model_profile_path.write_text(json.dumps(model_profile), encoding="utf-8")

    with pytest.raises(ValueError, match="sampling profile differs"):
        prepared_deployment.prepare_checkpoint_deployment_artifact(
            checkpoint,
            profile_path,
            model_rtc_profile_path=model_profile_path,
            output_dir=tmp_path / "runtime",
        )


def test_prepared_server_dry_run_uses_dynamic_runtime_and_explicit_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint, profile_path, openpi_root, profile = _fixture(tmp_path)
    (openpi_root / "scripts").mkdir()
    (openpi_root / "scripts" / "serve_dexteleop_policy.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    (openpi_root / "configs").mkdir()
    (openpi_root / "configs" / "dexteleop_policy_registry.json").write_text(
        '{"models": {}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(openpi_server.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        prepared_deployment.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    assert openpi_server.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--openpi-root",
            str(openpi_root),
            "--prepared-training-profile",
            str(profile_path),
            "--dry-run",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["command"][1:4] == [
        "-m",
        "deployments.openpi.prepared_server_runtime",
        "--checkpoint-dir",
    ]
    assert report["prepared_deployment_binding"]["openpi_asset_id"] == (
        profile["openpi_asset_id"]
    )
    registry_path = Path(
        report["command"][report["command"].index("--registry") + 1]
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["models"]["prepared-model"]
    assert entry["config"] == profile["openpi_config_name"]
    assert entry["normalization_contract_status"] == "explicit"
    assert entry["normalization_hash"]


def test_prepared_model_rtc_server_dry_run_binds_dynamic_output_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint, profile_path, model_profile_path, openpi_root, _profile = (
        _model_rtc_fixture(tmp_path)
    )
    (openpi_root / "scripts").mkdir()
    (openpi_root / "scripts" / "serve_dexteleop_policy.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    (openpi_root / "configs").mkdir()
    (openpi_root / "configs" / "dexteleop_policy_registry.json").write_text(
        '{"models": {}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(openpi_server.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        prepared_deployment.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    assert openpi_server.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--openpi-root",
            str(openpi_root),
            "--prepared-training-profile",
            str(profile_path),
            "--model-rtc-profile",
            str(model_profile_path),
            "--enable-rtc-continuation",
            "--dry-run",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_rtc_profile"] == str(model_profile_path)
    assert "--model-rtc-profile" in report["command"]
    assert report["command"][-1] == "--enable-rtc-continuation"
    registry_path = Path(
        report["command"][report["command"].index("--registry") + 1]
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["models"]["prepared-model-rtc"]
    assert entry["config"] == "prepared-model-rtc"
    assert entry["normalization_contract_status"] == "explicit"


def test_prepared_server_config_check_only_delegates_without_legacy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, profile_path, openpi_root, _profile = _fixture(tmp_path)
    (openpi_root / "scripts").mkdir()
    (openpi_root / "scripts" / "serve_dexteleop_policy.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    (openpi_root / "configs").mkdir()
    (openpi_root / "configs" / "dexteleop_policy_registry.json").write_text(
        '{"models": {}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(openpi_server.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        prepared_deployment.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(openpi_server.os, "chdir", lambda _path: None)

    def _capture_execv(executable: str, command: list[str]) -> None:
        captured["executable"] = executable
        captured["command"] = command

    monkeypatch.setattr(openpi_server.os, "execv", _capture_execv)

    assert openpi_server.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--openpi-root",
            str(openpi_root),
            "--prepared-training-profile",
            str(profile_path),
            "--prepared-config-check-only",
        ]
    ) == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[1:3] == [
        "-m",
        "deployments.openpi.prepared_server_runtime",
    ]
    assert command[-1] == "--validate-only"


def test_prepared_server_dynamic_loader_registers_dataclass_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "server_fixture.py"
    module_path.write_text(
        """from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Args:
    port: int = 8078
""",
        encoding="utf-8",
    )
    module_name = "_dexteleop_test_prepared_server"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = prepared_server_runtime._load_module(module_path, module_name)

    assert sys.modules[module_name] is module
    assert module.Args().port == 8078


def test_prepared_server_operator_stop_is_traceback_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _InterruptedServer:
        @staticmethod
        def main(_args: object) -> None:
            raise KeyboardInterrupt

    assert (
        prepared_server_runtime._serve_until_stopped(
            _InterruptedServer,
            object(),
        )
        == 130
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "OpenPI policy server stopped by operator.\n"


def test_prepared_server_dynamic_loader_restores_module_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "broken_server_fixture.py"
    module_path.write_text(
        """from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Args:
    port: int = 8078


raise RuntimeError("fixture import failed")
""",
        encoding="utf-8",
    )
    module_name = "_dexteleop_test_broken_prepared_server"
    previous = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, previous)

    with pytest.raises(RuntimeError, match="fixture import failed"):
        prepared_server_runtime._load_module(module_path, module_name)

    assert sys.modules[module_name] is previous


@dataclasses.dataclass(frozen=True)
class _Assets:
    asset_id: str
    assets_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class _Data:
    assets: _Assets
    dataset_root: str
    layout_path: str
    layout_hash: str
    camera_slots: tuple[str, ...]
    action_pose_mode: str
    last_action_pose_mode: str
    tail_policy: str
    strict_media_integrity: bool
    state_augmentation: object
    dataset_sources: tuple[object, ...]
    default_prompt: str = "fixture base prompt"


@dataclasses.dataclass(frozen=True)
class _Model:
    action_horizon: int
    image_keys: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _Config:
    name: str
    data: _Data
    model: _Model
    exp_name: str
    resume: bool
    overwrite: bool
    num_workers: int
    policy_metadata: object = None


@dataclasses.dataclass(frozen=True)
class _SourceConfig:
    name: str
    dataset_root: str
    sample_weight: int
    front_occlusion_schedule: tuple[tuple[float, float], ...]
    explicit_mask_keys: tuple[str, ...]
    front_force_black: bool
    episode_manifest: str


@dataclasses.dataclass(frozen=True)
class _StateAugmentation:
    state_input_mode: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "state_input_mode": self.state_input_mode,
            "name": self.name,
        }


def test_rebuilt_config_preserves_all_inference_relevant_prepared_fields(
    tmp_path: Path,
) -> None:
    checkpoint, profile_path, _openpi_root, profile = _fixture(tmp_path)
    resolution = prepared_deployment.prepare_checkpoint_deployment_artifact(
        checkpoint,
        profile_path,
        output_dir=tmp_path / "runtime",
    )
    trained_augmentation = _StateAugmentation(
        state_input_mode=resolution.deployment_artifact.state_input_mode,
        name="trained-augmentation",
    )
    resolution.deployment_payload["state_augmentation"] = (
        trained_augmentation.to_dict()
    )
    base = _Config(
        name="fixture-base",
        data=_Data(
            assets=_Assets("wrong-base-assets"),
            dataset_root="wrong",
            layout_path="wrong",
            layout_hash="wrong",
            camera_slots=("base_0_rgb",),
            action_pose_mode="absolute",
            last_action_pose_mode="absolute",
            tail_policy="drop",
            strict_media_integrity=False,
            state_augmentation=_StateAugmentation(
                state_input_mode=(
                    resolution.deployment_artifact.state_input_mode
                ),
                name="base-augmentation",
            ),
            dataset_sources=(),
        ),
        model=_Model(16, ("base_0_rgb",)),
        exp_name="base-exp",
        resume=True,
        overwrite=True,
        num_workers=99,
    )
    augmentation = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            state_augmentation=trained_augmentation,
        ),
    )

    class TrainingConfig:
        @staticmethod
        def get_config(name: str) -> _Config:
            return {
                "fixture-base": base,
                "fixture-augmentation": augmentation,
            }[name]

    class DatasetPolicy:
        DexTeleopV23DatasetSourceConfig = _SourceConfig

    config = prepared_server_runtime.build_prepared_openpi_config(
        resolution,
        training_config=TrainingConfig,
        dataset_policy=DatasetPolicy,
    )

    assert config.name == "prepared-model"
    assert config.data.assets.asset_id == profile["openpi_asset_id"]
    assert config.data.assets.assets_dir == str(checkpoint / "assets")
    assert config.data.layout_hash == profile["expected_policy_io_layout_hash"]
    assert config.data.camera_slots == ("base_0_rgb",)
    assert config.data.state_augmentation == trained_augmentation
    assert config.data.strict_media_integrity is True
    assert config.data.default_prompt == "fixture prepared prompt"
    assert config.data.dataset_sources[0].name == "source-a"
    assert config.data.dataset_sources[0].front_occlusion_schedule == (
        (0.0, 1.0),
    )
    assert config.resume is False
    assert config.overwrite is False
    assert config.num_workers == 4


def test_rebuilt_config_rejects_layout_metadata_drift(tmp_path: Path) -> None:
    checkpoint, profile_path, _openpi_root, _profile = _fixture(tmp_path)
    resolution = prepared_deployment.prepare_checkpoint_deployment_artifact(
        checkpoint,
        profile_path,
        output_dir=tmp_path / "runtime",
    )
    augmentation = _StateAugmentation(
        state_input_mode=resolution.deployment_artifact.state_input_mode,
        name="augmentation",
    )
    resolution.deployment_payload["state_augmentation"] = augmentation.to_dict()
    source_root = Path(
        str(resolution.profile["sources"][0]["dataset_root"])
    )
    (source_root / "meta" / "info.json").write_text(
        '{"policy_io_layout_hash":"sha256:drift"}\n',
        encoding="utf-8",
    )
    base = _Config(
        name="fixture-base",
        data=_Data(
            assets=_Assets("base-assets"),
            dataset_root="unused",
            layout_path="unused",
            layout_hash="unused",
            camera_slots=("base_0_rgb",),
            action_pose_mode="absolute",
            last_action_pose_mode="absolute",
            tail_policy="drop",
            strict_media_integrity=False,
            state_augmentation=augmentation,
            dataset_sources=(),
        ),
        model=_Model(16, ("base_0_rgb",)),
        exp_name="base",
        resume=False,
        overwrite=False,
        num_workers=0,
    )

    class TrainingConfig:
        @staticmethod
        def get_config(_name: str) -> _Config:
            return base

    class DatasetPolicy:
        DexTeleopV23DatasetSourceConfig = _SourceConfig

    with pytest.raises(ValueError, match="policy-I/O layout differs"):
        prepared_server_runtime.build_prepared_openpi_config(
            resolution,
            training_config=TrainingConfig,
            dataset_policy=DatasetPolicy,
        )


def test_rebuilt_model_rtc_config_uses_output_identity_and_bound_artifact(
    tmp_path: Path,
) -> None:
    checkpoint, profile_path, model_profile_path, _openpi_root, _profile = (
        _model_rtc_fixture(tmp_path)
    )
    resolution = prepared_deployment.prepare_checkpoint_deployment_artifact(
        checkpoint,
        profile_path,
        model_rtc_profile_path=model_profile_path,
        output_dir=tmp_path / "runtime",
    )
    augmentation = _StateAugmentation(
        state_input_mode=resolution.deployment_artifact.state_input_mode,
        name="trained-augmentation",
    )
    resolution.deployment_payload["state_augmentation"] = augmentation.to_dict()
    base = _Config(
        name="fixture-base",
        data=_Data(
            assets=_Assets("base-assets"),
            dataset_root="unused",
            layout_path="unused",
            layout_hash="unused",
            camera_slots=("base_0_rgb",),
            action_pose_mode="absolute",
            last_action_pose_mode="absolute",
            tail_policy="drop",
            strict_media_integrity=False,
            state_augmentation=augmentation,
            dataset_sources=(),
        ),
        model=_Model(16, ("base_0_rgb",)),
        exp_name="base",
        resume=False,
        overwrite=False,
        num_workers=0,
        policy_metadata={"base": "preserved"},
    )

    class TrainingConfig:
        @staticmethod
        def get_config(_name: str) -> _Config:
            return base

    class DatasetPolicy:
        DexTeleopV23DatasetSourceConfig = _SourceConfig

    config = prepared_server_runtime.build_prepared_openpi_config(
        resolution,
        training_config=TrainingConfig,
        dataset_policy=DatasetPolicy,
    )

    assert config.name == "prepared-model-rtc"
    assert config.policy_metadata["base"] == "preserved"
    bound = PolicyArtifactContract.from_dict(
        config.policy_metadata["policy_artifact"]
    )
    assert bound == resolution.deployment_artifact
    assert bound.model_continuation is not None


def test_prepared_client_contract_check_uses_the_same_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint, profile_path, _openpi_root, _profile = _fixture(tmp_path)
    monkeypatch.setattr(
        prepared_deployment.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    assert openpi_real.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--prepared-training-profile",
            str(profile_path),
            "--contract-check-only",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["model_id"] == "prepared-model"
    assert report["action_pose_mode"] == "absolute"
    assert report["server_contract"]["normalization_hash"]


def test_prepared_model_rtc_client_contract_check_verifies_both_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint, profile_path, model_profile_path, _openpi_root, _profile = (
        _model_rtc_fixture(tmp_path)
    )
    monkeypatch.setattr(
        prepared_deployment.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    assert openpi_real.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--prepared-training-profile",
            str(profile_path),
            "--model-rtc-profile",
            str(model_profile_path),
            "--rtc-profile",
            "model_rtc_flow_inpaint_shadow_15_30_60_120_v1",
            "--contract-check-only",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["model_id"] == "prepared-model-rtc"
    assert report["server_contract"]["normalization_hash"]


def test_prepared_client_explicit_position_overlay_resolves_15_60_120(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, profile_path, _openpi_root, _profile = _fixture(tmp_path)
    qpos_dataset = tmp_path / "writing-qpos"
    qpos_dataset.mkdir()
    resolved: dict[str, object] = {}

    def _validate_runtime(argv: list[str]) -> int:
        parser = argparse.ArgumentParser()
        run_policy.add_args(parser)
        args = parser.parse_args(argv)
        run_policy.validate_args(args, raw_argv=argv)
        resolved.update(
            {
                "arm_execution_profile": args.arm_execution_profile,
                "arm_control_mode": args.arm_control_mode,
                "position_control_profile": (
                    args.position_control_profile_resolved
                ),
                "action_hz": args.action_hz,
                "position_reference_hz": args.position_reference_hz,
                "arm_control_hz": args.arm_control_hz,
                "arm_canfd_follow": args.arm_canfd_follow,
                "qpos_dataset": args.policy_qpos_init_dataset,
                "qpos_episode": args.policy_qpos_init_episode,
                "qpos_frame": args.policy_qpos_init_frame,
                "send_actions": args.send_actions,
                "disconnected_gripper_sides": args.disconnected_gripper_sides,
                "disconnected_gripper_state": args.disconnected_gripper_state,
            }
        )
        return 0

    monkeypatch.setattr(run_policy, "main", _validate_runtime)

    assert openpi_real.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--prepared-training-profile",
            str(profile_path),
            "--deployment-profile",
            "rm65_bimanual_dmtac",
            "--arm-execution-profile",
            "position_15_60_120hz",
            "--arm-control-mode",
            "position",
            "--position-control-profile",
            "rm65_bimanual_position_v1",
            "--action-hz",
            "15",
            "--policy-qpos-init-dataset",
            str(qpos_dataset),
            "--policy-qpos-init-episode",
            "0",
            "--policy-qpos-init-frame",
            "0",
            "--disconnected-gripper-sides",
            "left",
            "--disconnected-gripper-state",
            "0.25",
            "--send-actions",
        ]
    ) == 0
    assert resolved == {
        "arm_execution_profile": "position_15_60_120hz",
        "arm_control_mode": "position",
        "position_control_profile": "rm65_bimanual_position_v1",
        "action_hz": 15.0,
        "position_reference_hz": 60.0,
        "arm_control_hz": 120.0,
        "arm_canfd_follow": True,
        "qpos_dataset": qpos_dataset,
        "qpos_episode": 0,
        "qpos_frame": 0,
        "send_actions": True,
        "disconnected_gripper_sides": ("left",),
        "disconnected_gripper_state": 0.25,
    }


def test_selected_disconnected_gripper_is_no_io_and_skips_physical_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(
        [
            "--no-arms",
            "--hand-backend",
            "dmtac-gripper",
            "--active-sides",
            "both",
            "--disconnected-gripper-sides",
            "left",
            "--disconnected-gripper-state",
            "0.25",
        ]
    )
    active_sides = run_policy.validate_args(args)
    stable_right_port = tmp_path / "serial" / "by-path" / "right-gripper"
    stable_right_port.parent.mkdir(parents=True)
    stable_right_port.touch()
    assignment = {
        "assignments": {
            "right": {
                "gripper": {
                    "port": "/dev/ttyUSB1",
                    "port_aliases": [
                        "/dev/ttyUSB1",
                        str(stable_right_port),
                    ],
                    "gripper_id": 1,
                    "baudrate": 115200,
                    "timeout": 0.03,
                },
                "tactile": {"inner": None, "outer": None},
            }
        }
    }
    load_calls: list[dict[str, object]] = []

    def _load_assignment(*_args: object, **kwargs: object) -> dict[str, object]:
        load_calls.append(dict(kwargs))
        return assignment

    monkeypatch.setattr(run_policy, "load_dmtac_hardware_assignment", _load_assignment)

    backends = run_policy.build_hand_backends(args, active_sides)
    specs = run_policy.hand_action_specs_for_backends(
        args,
        backends,
        active_sides,
    )
    disconnected = backends["left"]
    assert load_calls == [
        {
            "validate_current_usb": False,
            "allow_missing": False,
            "assume_yes_for_extra": True,
        }
    ]
    assert isinstance(disconnected, DMTacGripperBackend)
    assert disconnected.config.connect_hardware is False
    assert disconnected.config.connect_tactile is False
    assert disconnected.config.send_commands is False
    assert backends["right"].config.gripper_port == str(stable_right_port)
    disconnected.start()
    assert disconnected.device is None
    assert disconnected.is_ready()

    before = disconnected.get_state()
    assert before["intentionally_disconnected"] is True
    assert before["gripper_state_assumed"] is True
    assert before["gripper_state_source"] == "configured_disconnected_assumption"
    assert before["gripper_measurement_timestamp"] is None
    assert before["gripper_reconnect_on_error"] is False
    assert float(before["joint_pos"][0]) == pytest.approx(0.25)
    assert before["gripper_measured_scalar"] is None
    assert disconnected.send_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="gripper_scalar",
            value=np.asarray([0.8], dtype=np.float32),
        )
    )
    after = disconnected.get_state()
    assert after["gripper_last_send_status"] == "intentionally_disconnected_noop"
    assert float(after["joint_pos"][0]) == pytest.approx(0.25)
    assert after["last_sent_command"] is None

    discrete = TeleopCommand(timestamp=time.time())
    discrete.add_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="gripper_scalar",
            value=np.asarray([0.6], dtype=np.float32),
        )
    )
    receipt = run_policy._dispatch_position_discrete_command_checked(
        discrete,
        hand_backends={"left": disconnected},
    )
    assert receipt["accepted"] is True
    assert receipt["skipped"] is True
    assert (
        receipt["effectors"]["left_hand"]["status"]
        == "intentionally_disconnected_noop"
    )
    assert specs["left"]["hand_model"] == "dmtac_gripper"
    assert set(run_policy.policy_qpos_init_hand_action_specs(args, specs)) == {
        "right"
    }
    observation_now = time.time()
    observation = build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=False,
        hands_enabled=True,
        end_effector_sides=("left", "right"),
        step_index=0,
        arm_states={},
        hand_states={
            "left": disconnected.get_state(),
            "right": {
                "joint_pos": np.zeros(6, dtype=np.float32),
                "joint_timestamp": observation_now,
            },
        },
        timestamp=observation_now,
        hand_state_max_age_sec=0.2,
    )
    assert observation.faults == []
    assert int(np.sum(observation.state_valid)) == 12


def test_default_connected_gripper_keeps_usb_validation_and_transport_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(
        [
            "--no-arms",
            "--hand-backend",
            "dmtac-gripper",
            "--active-sides",
            "left",
        ]
    )
    active_sides = run_policy.validate_args(args)
    assignment = {
        "assignments": {
            "left": {
                "gripper": {
                    "port": "/dev/serial/by-id/left-gripper",
                    "gripper_id": 1,
                    "baudrate": 115200,
                    "timeout": 0.03,
                },
                "tactile": {"inner": None, "outer": None},
            }
        }
    }
    load_calls: list[dict[str, object]] = []

    def _load_assignment(*_args: object, **kwargs: object) -> dict[str, object]:
        load_calls.append(dict(kwargs))
        return assignment

    monkeypatch.setattr(run_policy, "load_dmtac_hardware_assignment", _load_assignment)
    backends = run_policy.build_hand_backends(args, active_sides)

    assert load_calls == [
        {
            "validate_current_usb": True,
            "allow_missing": False,
            "assume_yes_for_extra": True,
        }
    ]
    connected = backends["left"]
    assert isinstance(connected, DMTacGripperBackend)
    assert connected.config.connect_hardware is True
    assert connected.config.send_commands is True
    assert connected.config.disconnected_state_scalar is None


def test_both_disconnected_grippers_do_not_load_assignment_or_open_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(
        [
            "--no-arms",
            "--hand-backend",
            "dmtac-gripper",
            "--active-sides",
            "both",
            "--disconnected-gripper-sides",
            "both",
        ]
    )
    active_sides = run_policy.validate_args(args)

    def _unexpected_assignment_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("both-disconnected mode must not load DMTAC assignment")

    monkeypatch.setattr(
        run_policy,
        "load_dmtac_hardware_assignment",
        _unexpected_assignment_load,
    )
    backends = run_policy.build_hand_backends(args, active_sides)
    assert set(backends) == {"left", "right"}
    for backend in backends.values():
        backend.start()
        assert isinstance(backend, DMTacGripperBackend)
        assert backend.device is None
        assert backend.is_ready()
        assert backend.get_state()["intentionally_disconnected"] is True


def test_disconnected_gripper_selection_is_strictly_bounded() -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    with pytest.raises(ValueError, match="subset"):
        run_policy.validate_args(
            parser.parse_args(
                [
                    "--no-arms",
                    "--hand-backend",
                    "dmtac-gripper",
                    "--active-sides",
                    "right",
                    "--disconnected-gripper-sides",
                    "left",
                ]
            )
        )
    with pytest.raises(ValueError, match="dmtac-gripper"):
        run_policy.validate_args(
            parser.parse_args(
                [
                    "--no-arms",
                    "--hand-backend",
                    "inspire",
                    "--active-sides",
                    "left",
                    "--left-hand-ip",
                    "127.0.0.1",
                    "--disconnected-gripper-sides",
                    "left",
                ]
            )
        )
    with pytest.raises(ValueError, match="finite and in"):
        run_policy.validate_args(
            parser.parse_args(
                [
                    "--no-arms",
                    "--hand-backend",
                    "dmtac-gripper",
                    "--active-sides",
                    "left",
                    "--disconnected-gripper-sides",
                    "left",
                    "--disconnected-gripper-state",
                    "nan",
                ]
            )
        )


def test_joint_prompt_checkpoint_metadata_is_bound_exactly() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
        / "writing_multistage_prompt_0804_0806_front_h16_absolute_v1.json"
    )
    profile = prepared_deployment.load_prepared_training_profile(profile_path)
    payload = {
        "prepared_prompt_training_binding": {
            "camera_slot_to_dataset_key": [
                {
                    "model_slot": "base_0_rgb",
                    "dataset_camera_key": "front",
                }
            ],
            "action_horizon": 16,
            "action_pose_mode": "absolute",
            "episode_count": 890,
            "window_count": 300426,
            "prompt_stage_sampling_policy": "balanced_min_windows_rotate_v1",
            "prompt_stage_sampling_seed": 20260811,
            "prompt_skills": [
                {
                    "stage_id": item["stage_id"],
                    "skill_id": item["skill_id"],
                    "prompt": item["prompt"],
                    "window_count": item["expected_window_count"],
                    "expected_openpi_window_count": item[
                        "expected_openpi_window_count"
                    ],
                }
                for item in profile["prompt_stages"]
            ],
        },
        "dataset_mixing": {
            "sources": [
                {
                    "name": item["source_dataset_id"],
                    "dataset_root": item["dataset_root"],
                }
                for item in profile["sources"]
            ]
        },
    }
    prepared_deployment._validate_checkpoint_training_metadata(
        payload,
        profile,
    )

    payload["prepared_prompt_training_binding"]["prompt_skills"][1][
        "prompt"
    ] = "unbound prompt"
    with pytest.raises(ValueError, match="disagrees"):
        prepared_deployment._validate_checkpoint_training_metadata(
            payload,
            profile,
        )


def test_stage1_scenario_checkpoint_metadata_is_bound_exactly() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
        / "writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json"
    )
    profile = prepared_deployment.load_prepared_training_profile(profile_path)
    repeat_weights = [
        {
            "source_dataset_id": item["source_dataset_id"],
            "repeat_weight": item["repeat_weight"],
        }
        for item in profile["sources"]
    ]
    payload = {
        "prepared_skill_training_binding": {
            "schema": "dexteleop.openpi_prepared_skill_binding.v2",
            "camera_slot_to_dataset_key": [
                {
                    "model_slot": "base_0_rgb",
                    "dataset_camera_key": "front",
                }
            ],
            "action_horizon": 16,
            "action_pose_mode": "absolute",
            "skill_id": profile["target_skill_id"],
            "selection_policy_id": (
                prepared_deployment.STAGE1_SCENARIO_SELECTION_POLICY_V1
            ),
            "source_mixing_policy": profile["source_mixing_policy"],
            "source_repeat_weights": repeat_weights,
            "episode_count": profile["expected_total_eligible_episode_count"],
            "window_count": profile[
                "expected_prepared_interval_window_count"
            ],
            "prepared_interval_repeated_window_count": profile[
                "expected_prepared_interval_repeated_window_count"
            ],
        },
        "dataset_mixing": {
            "schema": "dexteleop_v23_bound_source_repeat_v1",
            "source_mixing_policy": profile["source_mixing_policy"],
            "source_order": [
                item["source_dataset_id"] for item in profile["sources"]
            ],
            "each_quality_valid_window_once_per_epoch": False,
            "repeat_each_quality_valid_window_by_source_weight": True,
            "source_repeat_weights": {
                item["source_dataset_id"]: item["repeat_weight"]
                for item in profile["sources"]
            },
            "sources": [
                {
                    "name": item["source_dataset_id"],
                    "dataset_root": item["dataset_root"],
                    "sample_weight": item["repeat_weight"],
                }
                for item in profile["sources"]
            ],
        },
    }
    prepared_deployment._validate_checkpoint_training_metadata(payload, profile)

    drifted_binding = json.loads(json.dumps(payload))
    drifted_binding["prepared_skill_training_binding"][
        "source_repeat_weights"
    ][-1]["repeat_weight"] = 1
    with pytest.raises(ValueError, match="prepared binding disagrees"):
        prepared_deployment._validate_checkpoint_training_metadata(
            drifted_binding,
            profile,
        )

    drifted_mixing = json.loads(json.dumps(payload))
    drifted_mixing["dataset_mixing"]["source_repeat_weights"][
        "midcourse_interrupt_2026_08_11"
    ] = 1
    with pytest.raises(ValueError, match="dataset mixing disagrees"):
        prepared_deployment._validate_checkpoint_training_metadata(
            drifted_mixing,
            profile,
        )


def test_prompt_family_checkpoint_metadata_and_deploy_allowlist_are_exact() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
        / "writing_multistage_prompt_family_0804_0806_front_h16_absolute_v2.json"
    )
    profile = prepared_deployment.load_prepared_training_profile(profile_path)
    payload = {
        "prepared_prompt_training_binding": {
            "schema": "dexteleop.openpi_prepared_prompt_binding.v2",
            "camera_slot_to_dataset_key": [
                {
                    "model_slot": "base_0_rgb",
                    "dataset_camera_key": "front",
                }
            ],
            "action_horizon": 16,
            "action_pose_mode": "absolute",
            "episode_count": 890,
            "window_count": 300426,
            "prompt_stage_sampling_policy": "balanced_min_windows_rotate_v1",
            "prompt_stage_sampling_seed": 20260811,
            "prompt_family_sampling_policy": (
                "canonical_80_train_variants_20_stratified_v1"
            ),
            "prompt_family_sampling_seed": 20260812,
            "canonical_prompt_percent": 80,
            "training_variant_prompt_percent": 20,
            "held_out_prompts_in_training": False,
            "prompt_skills": [
                {
                    "stage_id": item["stage_id"],
                    "skill_id": item["skill_id"],
                    "prompt": item["canonical_prompt"],
                    "training_prompt_variants": list(
                        item["training_prompt_variants"]
                    ),
                    "held_out_prompt_variants": list(
                        item["held_out_prompt_variants"]
                    ),
                    "window_count": item["expected_window_count"],
                    "expected_openpi_window_count": item[
                        "expected_openpi_window_count"
                    ],
                }
                for item in profile["prompt_stages"]
            ],
        },
        "dataset_mixing": {
            "sources": [
                {
                    "name": item["source_dataset_id"],
                    "dataset_root": item["dataset_root"],
                }
                for item in profile["sources"]
            ]
        },
    }
    prepared_deployment._validate_checkpoint_training_metadata(payload, profile)

    deployment_stages = [
        {
            "stage_id": item["stage_id"],
            "skill_id": item["skill_id"],
            "prompt": item["canonical_prompt"],
            "allowed_prompts": [
                item["canonical_prompt"],
                *item["training_prompt_variants"],
            ],
        }
        for item in profile["prompt_stages"]
    ]
    allowed = prepared_allowed_prompts(
        {
            "openpi_prepared_deployment_binding": {
                "prompt_stages": deployment_stages
            }
        }
    )
    assert allowed is not None
    assert len(allowed) == 52
    assert all(
        prompt not in allowed
        for item in profile["prompt_stages"]
        for prompt in item["held_out_prompt_variants"]
    )
    metadata = {
        "openpi_prepared_deployment_binding": {
            "prompt_stages": deployment_stages
        }
    }
    for item in profile["prompt_stages"]:
        assert prepared_prompt_for_stage(metadata, item["stage_id"]) == item[
            "canonical_prompt"
        ]
    explicit_variant = profile["prompt_stages"][2]["training_prompt_variants"][4]
    assert resolve_prepared_prompt_selection(
        metadata,
        prompt=explicit_variant,
        prompt_stage=None,
    ) == (explicit_variant, None)
    assert resolve_prepared_prompt_selection(
        metadata,
        prompt=None,
        prompt_stage="stage4",
    ) == (profile["prompt_stages"][3]["canonical_prompt"], "stage4")
    with pytest.raises(ValueError, match="requires --prompt-stage or --prompt"):
        resolve_prepared_prompt_selection(
            metadata,
            prompt=None,
            prompt_stage=None,
        )
    with pytest.raises(ValueError, match="not allowed"):
        resolve_prepared_prompt_selection(
            metadata,
            prompt=profile["prompt_stages"][0]["held_out_prompt_variants"][0],
            prompt_stage=None,
        )
    with pytest.raises(ValueError, match="available stages"):
        prepared_prompt_for_stage(metadata, "stage5")

    payload["prepared_prompt_training_binding"]["prompt_skills"][0][
        "held_out_prompt_variants"
    ][0] = "changed held-out prompt"
    with pytest.raises(ValueError, match="disagrees"):
        prepared_deployment._validate_checkpoint_training_metadata(
            payload,
            profile,
        )


def test_real_client_resolves_0807_prompt_stage_before_runtime_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
        / "writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json"
    )
    profile = prepared_deployment.load_prepared_training_profile(profile_path)
    prompt_stages = [
        {
            "stage_id": item["stage_id"],
            "skill_id": item["skill_id"],
            "prompt": item["canonical_prompt"],
            "allowed_prompts": [
                item["canonical_prompt"],
                *item["training_prompt_variants"],
            ],
        }
        for item in profile["prompt_stages"]
    ]
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id=str(profile["openpi_config_name"]),
        checkpoint_step=30000,
        dataset_schema="dexteleop_dataset_v2.3",
        camera_input=PolicyCameraInputContract(
            inputs=(
                PolicyCameraInput(
                    source_key="front",
                    model_input_key="observation.images.front",
                    server_camera_slot="base_0_rgb",
                ),
            )
        ),
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=16,
            action_pose_mode="absolute",
            tail_policy="drop",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=("sha256:" + "1" * 64,),
        metadata={
            "openpi_prepared_deployment_binding": {
                "prompt_stages": prompt_stages,
            }
        },
    )
    artifact_path = tmp_path / "prompt-family.policy_artifact.json"
    artifact_path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    with pytest.raises(SystemExit):
        openpi_real.main(
            [
                "--policy-artifact-json",
                str(artifact_path),
                "--contract-check-only",
            ]
        )
    assert "requires --prompt-stage or --prompt" in capsys.readouterr().err

    assert openpi_real.main(
        [
            "--policy-artifact-json",
            str(artifact_path),
            "--prompt-stage",
            "stage3",
            "--contract-check-only",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["prompt_selection"] == {
        "source": "prepared_stage",
        "stage_id": "stage3",
        "prompt": profile["prompt_stages"][2]["canonical_prompt"],
    }

    captured: dict[str, object] = {}

    def _capture_runtime(argv: list[str], **_kwargs: object) -> int:
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(run_policy, "main", _capture_runtime)
    assert openpi_real.main(
        [
            "--policy-artifact-json",
            str(artifact_path),
            "--prompt-stage",
            "stage2",
        ]
    ) == 0
    delegated = captured["argv"]
    assert isinstance(delegated, list)
    policy_kwargs_index = delegated.index("--policy-kwargs-json") + 1
    policy_kwargs = json.loads(delegated[policy_kwargs_index])
    assert policy_kwargs["prompt"] == profile["prompt_stages"][1][
        "canonical_prompt"
    ]
    assert "prompt_stage" not in policy_kwargs
