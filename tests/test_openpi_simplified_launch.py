from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from dataset_contracts import (
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi import server as openpi_server
from deployments.openpi.client import real as openpi_real


def _write_checkpoint(root: Path) -> tuple[Path, PolicyArtifactContract]:
    checkpoint = root / "checkpoints" / "test-policy" / "123"
    checkpoint.mkdir(parents=True)
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="test-policy",
        checkpoint_step=123,
        camera_input=PolicyCameraInputContract(
            inputs=(
                PolicyCameraInput(
                    source_key="front",
                    model_input_key="observation.images.front",
                    server_camera_slot="base_0_rgb",
                ),
            )
        ),
    )
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(artifact.to_dict()),
        encoding="utf-8",
    )
    return checkpoint, artifact


def test_client_checkpoint_dir_resolves_artifact_and_profile(tmp_path: Path) -> None:
    checkpoint, artifact = _write_checkpoint(tmp_path)
    with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
        result = openpi_real.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--deployment-profile",
                "rm65_bimanual_dmtac",
                "--max-steps",
                "1",
            ]
        )

    assert result == 0
    launched = run_main.call_args.args[0]
    artifact_index = launched.index("--policy-artifact-json")
    assert launched[artifact_index + 1] == str(checkpoint / "policy_artifact.json")
    kwargs = json.loads(launched[launched.index("--policy-kwargs-json") + 1])
    assert kwargs["policy_artifact"]["model_id"] == artifact.model_id
    assert "--arm-execution-profile" in launched
    assert launched[launched.index("--arm-execution-profile") + 1] == "impedance_40hz"
    assert "--send-actions" not in launched
    assert launched[launched.index("--max-steps") + 1] == "1"


def test_client_checkpoint_dir_rejects_missing_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "empty-checkpoint"
    checkpoint.mkdir()
    with mock.patch("teleop_runtime.deploy.run_policy.main") as run_main:
        try:
            openpi_real.main(["--checkpoint-dir", str(checkpoint)])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("missing artifact must fail")
    run_main.assert_not_called()


def test_server_dry_run_builds_registry_from_artifact(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    openpi_root = tmp_path / "openpi"
    checkpoint, artifact = _write_checkpoint(openpi_root)
    (openpi_root / "scripts").mkdir()
    (openpi_root / "scripts/serve_dexteleop_policy.py").write_text(
        "# test fixture\n",
        encoding="utf-8",
    )
    (openpi_root / "configs").mkdir()
    (openpi_root / "configs/dexteleop_policy_registry.json").write_text(
        json.dumps({"models": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(openpi_server.tempfile, "gettempdir", lambda: str(tmp_path))

    assert openpi_server.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--openpi-root",
            str(openpi_root),
            "--dry-run",
        ]
    ) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["model_id"] == artifact.model_id
    assert report["environment"]["PYTHONPATH"].split(":")[0] == str(
        Path(openpi_server.__file__).resolve().parents[2]
    )
    registry_path = Path(report["command"][report["command"].index("--registry") + 1])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["models"][artifact.model_id]
    assert entry["config"] == artifact.model_id
    assert entry["checkpoint_dir"] == str(checkpoint)
    assert entry["action_dim"] == artifact.policy_io_layout.action_dim
    assert entry["camera_slots"] == ["base_0_rgb"]
