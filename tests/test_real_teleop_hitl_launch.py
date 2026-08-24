from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

import pytest

from dataset_contracts import PolicyArtifactContract
from dataset_contracts.hitl import EXCLUSIVE_HITL_DATASET_METADATA_KEY
from dataset_contracts.policy_artifact import PolicyCameraInputContract
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client.normalization import (
    make_openpi_quantile_normalization_contract,
)
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
)
from teleop_runtime.real_teleop.hitl_launch import (
    CANONICAL_REAL_TELEOP_MODULE,
    HITLLaunchAcceptanceError,
    main,
    prepare_hitl_launch,
)


DATE_PREFIX = "2026-08-24"


def _write_policy_artifact(path: Path) -> Path:
    pose_contract = (
        default_rm65_workspace_near_boundary_collection_pose_frame_contract(
            ("left", "right")
        )
    )
    assert pose_contract is not None
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="hitl-launch-fixture",
        checkpoint_step=50_176,
        dataset_schema="dexteleop_dataset_v2.3",
        camera_input=PolicyCameraInputContract(inputs=(), required=False),
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=8,
            action_pose_mode="absolute",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(pose_contract.contract_hash,),
        metadata={
            "normalization_contract": make_openpi_quantile_normalization_contract(
                stats_ref="norm_stats.json",
                stats_hash="sha256:" + "1" * 64,
            )
        },
    )
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def _normal_args(tmp_path: Path) -> list[str]:
    artifact_path = _write_policy_artifact(tmp_path / "policy_artifact.json")
    return [
        "--gui",
        "--hitl-policy-control",
        "--hitl-policy-mode",
        "normal",
        "--hitl-policy-artifact-json",
        str(artifact_path),
        "--dataset-root",
        str(tmp_path / "datasets"),
        "--collection-task-name",
        "Write Words",
        "--collection-task-id",
        "write_words_v1",
    ]


def _rtc_args(tmp_path: Path) -> list[str]:
    return [
        "--gui",
        "--hitl-policy-control",
        "--hitl-policy-mode",
        "rtc",
        "--dataset-root",
        str(tmp_path / "datasets"),
        "--collection-task-name",
        "Write Words RTC Shadow",
        "--collection-task-id",
        "write_words_rtc_shadow_v1",
    ]


def test_normal_offline_acceptance_binds_policy_and_dataset_without_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_socket(*_args, **_kwargs):
        raise AssertionError("offline acceptance must not open a socket")

    monkeypatch.setattr(socket, "socket", _unexpected_socket)
    prepared = prepare_hitl_launch(
        _normal_args(tmp_path),
        date_prefix=DATE_PREFIX,
    )
    report = prepared.to_dict()

    assert report["ok"] is True
    assert report["policy_mode"] == "normal"
    assert report["execution_supported"] is True
    assert report["normal_policy_preflight"]["ok"] is True
    assert report["normal_policy_preflight"]["model_id"] == (
        "hitl-launch-fixture"
    )
    assert report["policy_endpoint"]["reachability"] == "not_checked"
    assert report["dataset"]["root"] == str(
        tmp_path / "datasets" / "2026-08-24_Write_Words"
    )
    assert report["dataset"]["task_id"] == "write_words_v1"
    assert report["hitl_dataset_contract"]["policy_identity_status"] == (
        "artifact_bound"
    )
    assert all(value is False for value in report["resource_effects"].values())


def test_default_cli_is_read_only_and_never_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _unexpected_exec(*_args, **_kwargs):
        raise AssertionError("default acceptance must not execute RealTeleop")

    monkeypatch.setattr("os.execv", _unexpected_exec)
    assert main(["--", *_normal_args(tmp_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["acceptance_kind"] == "read_only_offline_contract_preflight"
    assert report["execute_requested"] is False
    assert report["canonical_command"][1:3] == [
        "-m",
        CANONICAL_REAL_TELEOP_MODULE,
    ]


def test_execute_hands_exact_arguments_to_canonical_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExecHandoff(Exception):
        pass

    captured: dict[str, object] = {}
    real_args = _normal_args(tmp_path)

    def _capture_exec(path: str, argv: list[str]) -> None:
        captured["path"] = path
        captured["argv"] = list(argv)
        raise _ExecHandoff

    monkeypatch.setattr("os.execv", _capture_exec)
    with pytest.raises(_ExecHandoff):
        main(["--execute", "--", *real_args])

    assert captured == {
        "path": sys.executable,
        "argv": [
            sys.executable,
            "-m",
            CANONICAL_REAL_TELEOP_MODULE,
            *real_args,
        ],
    }


def test_rtc_acceptance_is_no_action_and_cannot_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepare_hitl_launch(
        _rtc_args(tmp_path),
        date_prefix=DATE_PREFIX,
    )
    report = prepared.to_dict()
    assert report["policy_mode"] == "rtc"
    assert report["rtc_no_action"] is True
    assert report["execution_supported"] is False
    assert report["normal_policy_preflight"] is None
    assert report["hitl_dataset_contract"]["policy_identity_status"] == (
        "rtc_no_action"
    )

    monkeypatch.setattr(
        "os.execv",
        lambda *_args, **_kwargs: pytest.fail("RTC must not execute"),
    )
    with pytest.raises(SystemExit, match="2"):
        main(["--execute", "--", *_rtc_args(tmp_path)])
    assert "RTC remains no-action" in capsys.readouterr().err


def test_acceptance_requires_explicit_hitl_profile_and_formal_collection(
    tmp_path: Path,
) -> None:
    without_hitl = _normal_args(tmp_path)
    without_hitl.remove("--hitl-policy-control")
    with pytest.raises(HITLLaunchAcceptanceError, match="hitl-policy-control"):
        prepare_hitl_launch(without_hitl, date_prefix=DATE_PREFIX)

    with pytest.raises(HITLLaunchAcceptanceError, match="formal collection"):
        prepare_hitl_launch(
            [*_normal_args(tmp_path), "--no-collection"],
            date_prefix=DATE_PREFIX,
        )


@pytest.mark.parametrize(
    "missing_option",
    [
        "--dataset-root",
        "--collection-task-name",
        "--collection-task-id",
    ],
)
def test_acceptance_requires_explicit_collection_identity(
    tmp_path: Path,
    missing_option: str,
) -> None:
    real_args = _normal_args(tmp_path)
    option_index = real_args.index(missing_option)
    del real_args[option_index : option_index + 2]

    with pytest.raises(HITLLaunchAcceptanceError, match=missing_option):
        prepare_hitl_launch(real_args, date_prefix=DATE_PREFIX)


def test_existing_dataset_task_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    real_args = _normal_args(tmp_path)
    prepared = prepare_hitl_launch(real_args, date_prefix=DATE_PREFIX)
    report = prepared.to_dict()
    dataset_root = Path(report["dataset"]["root"])
    info_path = dataset_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "dataset_name": report["dataset"]["dataset_name"],
                "task_id": "different_task_v1",
                EXCLUSIVE_HITL_DATASET_METADATA_KEY: report[
                    "hitl_dataset_contract"
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HITLLaunchAcceptanceError,
        match="task binding differs",
    ):
        prepare_hitl_launch(real_args, date_prefix=DATE_PREFIX)


def test_populated_target_without_hitl_identity_fails_closed(
    tmp_path: Path,
) -> None:
    real_args = _normal_args(tmp_path)
    dataset_root = tmp_path / "datasets" / "2026-08-24_Write_Words"
    data_path = dataset_root / "data" / "partial.jsonl"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no meta/info.json HITL identity"):
        prepare_hitl_launch(real_args, date_prefix=DATE_PREFIX)
