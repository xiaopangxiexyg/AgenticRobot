"""Launch the existing OpenPI server directly from a checkpoint directory.

The OpenPI server currently selects its training config through a registry.
This wrapper derives the registry entry from the checkpoint's authoritative
policy artifact and writes only a disposable file under the system temp
directory.  It does not modify the OpenPI checkout or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from deployments.client import load_policy_artifact
from robot_motion.deploy.normalization import (
    NormalizationContract,
    NormalizationContractError,
)


def _openpi_root_from_checkpoint(checkpoint_dir: Path) -> Path | None:
    for parent in (checkpoint_dir, *checkpoint_dir.parents):
        if parent.name == "checkpoints":
            return parent.parent
    return None


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "policy"


def _pythonpath_with_project(project_root: Path, current: str | None) -> str:
    entries = [str(project_root.resolve())]
    entries.extend(
        value
        for value in str(current or "").split(os.pathsep)
        if value and value not in entries
    )
    return os.pathsep.join(entries)


def build_runtime_registry_entry(
    checkpoint_dir: Path,
    *,
    config_name: str | None = None,
    default_prompt: str | None = None,
    action_horizon: int = 50,
) -> tuple[str, dict[str, object]]:
    artifact = load_policy_artifact(checkpoint_dir / "policy_artifact.json")
    model_id = artifact.model_id
    if model_id is None:
        raise ValueError("policy artifact must declare model_id for server launch")
    pose_chunk = artifact.pose_action_chunk
    resolved_action_horizon = (
        int(action_horizon)
        if pose_chunk is None
        else int(pose_chunk.action_horizon)
    )
    camera_slots = (
        []
        if artifact.camera_input is None
        else list(artifact.camera_input.server_camera_slots)
    )
    entry: dict[str, object] = {
        "config": config_name or model_id,
        "checkpoint_dir": str(checkpoint_dir),
        "deploy_schema_version": artifact.deploy_schema_version,
        "action_key": artifact.action_key,
        "action_discrete_key": artifact.action_discrete_key,
        "action_dim": artifact.policy_io_layout.action_dim,
        "action_discrete_dim": artifact.policy_io_layout.action_discrete_dim,
        "state_dim": artifact.state_input_dim,
        "state_discrete_dim": artifact.policy_io_layout.state_discrete_dim,
        "action_horizon": resolved_action_horizon,
        "active_sides": ["left", "right"],
        "camera_slots": camera_slots,
    }
    raw_normalization = artifact.metadata.get("normalization_contract")
    if raw_normalization is not None:
        try:
            normalization = NormalizationContract.from_optional_mapping(
                raw_normalization,
                allow_legacy=True,
            )
        except NormalizationContractError as exc:
            raise ValueError(
                "policy artifact metadata.normalization_contract is invalid: " + str(exc)
            ) from exc
        entry["normalization_contract_status"] = (
            "explicit" if normalization is not None else "legacy_opaque"
        )
        if normalization is not None:
            entry["normalization_contract"] = normalization.to_dict()
            entry["normalization_hash"] = normalization.contract_hash
    else:
        # Absence is not identity: older OpenPI artifacts simply did not
        # publish the server-side stats owner/hash.  Keep that distinction in
        # the disposable registry so callers cannot mistake an omitted field
        # for a verified no-op contract.
        entry["normalization_contract_status"] = "missing"
    if default_prompt is not None:
        entry["default_prompt"] = default_prompt
    return model_id, entry


def write_runtime_registry(
    *,
    base_registry: Path,
    checkpoint_dir: Path,
    config_name: str | None = None,
    default_prompt: str | None = None,
    action_horizon: int = 50,
) -> tuple[str, Path]:
    payload = json.loads(base_registry.read_text(encoding="utf-8"))
    models = payload.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"registry 'models' must be an object: {base_registry}")
    model_id, entry = build_runtime_registry_entry(
        checkpoint_dir,
        config_name=config_name,
        default_prompt=default_prompt,
        action_horizon=action_horizon,
    )
    models[model_id] = entry
    step = checkpoint_dir.name
    target = Path(tempfile.gettempdir()) / (
        f"dexteleop_openpi_{_safe_filename(model_id)}_{_safe_filename(step)}.json"
    )
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return model_id, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch scripts/serve_dexteleop_policy.py from a checkpoint directory."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, default=None)
    parser.add_argument("--config", default=None, help="Training config override.")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the temporary registry and print the server command without starting it.",
    )
    args = parser.parse_args(argv)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    artifact_path = checkpoint_dir / "policy_artifact.json"
    if not checkpoint_dir.is_dir():
        parser.error(f"checkpoint directory does not exist: {checkpoint_dir}")
    if not artifact_path.is_file():
        parser.error(f"checkpoint has no policy_artifact.json: {checkpoint_dir}")

    inferred_root = _openpi_root_from_checkpoint(checkpoint_dir)
    openpi_root = (
        args.openpi_root.expanduser().resolve()
        if args.openpi_root is not None
        else inferred_root
    )
    if openpi_root is None:
        parser.error("cannot infer OpenPI root; pass --openpi-root")
    server_script = openpi_root / "scripts/serve_dexteleop_policy.py"
    base_registry = openpi_root / "configs/dexteleop_policy_registry.json"
    if not server_script.is_file():
        parser.error(f"OpenPI server script does not exist: {server_script}")
    if not base_registry.is_file():
        parser.error(f"OpenPI registry does not exist: {base_registry}")

    model_id, runtime_registry = write_runtime_registry(
        base_registry=base_registry,
        checkpoint_dir=checkpoint_dir,
        config_name=args.config,
        default_prompt=args.default_prompt,
        action_horizon=args.action_horizon,
    )
    command = [
        sys.executable,
        str(server_script),
        "--model",
        model_id,
        "--registry",
        str(runtime_registry),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    if args.default_prompt is not None:
        command.extend(("--default-prompt", args.default_prompt))
    if args.record:
        command.append("--record")
    project_root = Path(__file__).resolve().parents[2]
    runtime_pythonpath = _pythonpath_with_project(
        project_root,
        os.environ.get("PYTHONPATH"),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_id": model_id,
                    "command": command,
                    "environment": {"PYTHONPATH": runtime_pythonpath},
                },
                indent=2,
            )
        )
        return 0

    os.environ["PYTHONPATH"] = runtime_pythonpath
    os.chdir(openpi_root)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
