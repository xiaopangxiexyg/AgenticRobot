from __future__ import annotations

import json

import pytest

from dataset_contracts import PolicyArtifactContract
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client import preflight_policy_artifact
from deployments.openpi.client import real as openpi_real_launcher


def _v22_relative_artifact() -> PolicyArtifactContract:
    pose_frames = PoseFrameContract.legacy_v22_raw(
        active_sides=("left", "right")
    )
    return PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="v22-relative-test",
        checkpoint_step=149999,
        dataset_schema="dexteleop_dataset_v2.3",
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=50,
            action_pose_mode="relative",
            last_action_pose_mode="relative",
            tail_policy="drop",
        ),
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=("left", "right")
        ),
        pose_frame_contract_hashes=(pose_frames.contract_hash,),
    )


def test_openpi_artifact_preflight_resolves_v22_relative_identity_binding() -> None:
    report = preflight_policy_artifact(_v22_relative_artifact()).to_dict()

    assert report["ok"] is True
    assert report["action_pose_mode"] == "relative"
    assert report["pose_training_origin"] == "v22_raw_identity_compatibility"
    assert report["runtime_pose_frame_contract_source"] == "v22_identity_compatibility"
    session = report["session_contract"]
    assert session["policy_absolute_frame"] == "canonical_desk"
    assert (
        session["runtime_pose_frame_contract"]["source"]["source_id"]
        == "rm65_real_dual_arm_v1"
    )
    assert "v22_raw_identity_compatibility_policy_state_binding" in session["warnings"]


def test_openpi_real_contract_check_only_does_not_start_runtime(
    tmp_path, monkeypatch, capsys
) -> None:
    artifact_path = tmp_path / "policy_artifact.json"
    artifact_path.write_text(
        json.dumps(_v22_relative_artifact().to_dict()),
        encoding="utf-8",
    )

    def _unexpected_runtime_import(*_args, **_kwargs):
        raise AssertionError("contract-only preflight must not launch the runtime")

    monkeypatch.setattr(
        "teleop_runtime.deploy.run_policy.main",
        _unexpected_runtime_import,
    )
    result = openpi_real_launcher.main(
        [
            "--policy-artifact-json",
            str(artifact_path),
            "--contract-check-only",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_id"] == "v22-relative-test"
    assert payload["action_pose_mode"] == "relative"


def test_openpi_real_contract_check_only_requires_artifact() -> None:
    with pytest.raises(SystemExit, match="2"):
        openpi_real_launcher.main(["--contract-check-only"])
