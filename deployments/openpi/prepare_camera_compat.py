"""Create a versioned legacy wrist camera compatibility artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_interface.camera_semantics import load_camera_role_binding
from deployments.policy_artifact import load_policy_artifact

from .camera_semantics import legacy_swapped_wrist_artifact


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a new policy artifact that preserves a legacy wrist-only "
            "checkpoint's physical camera inputs after runtime left/right correction."
        )
    )
    parser.add_argument("--source-artifact-json", type=Path, required=True)
    parser.add_argument("--output-artifact-json", type=Path, required=True)
    parser.add_argument(
        "--camera-role-registry",
        type=Path,
        default=(
            _PROJECT_ROOT / "robot_motion/configs/rm65_camera_registry.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source_artifact_json.expanduser().resolve()
    output = args.output_artifact_json.expanduser()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite compatibility artifact: {output}"
        )
    artifact = legacy_swapped_wrist_artifact(
        load_policy_artifact(source),
        load_camera_role_binding(args.camera_role_registry),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": "dexteleop.legacy_wrist_camera_compat_prepare.v1",
                "source_artifact": str(source),
                "output_artifact": str(output.resolve()),
                "camera_semantics": artifact.metadata["camera_semantics"],
                "camera_role_binding_hash": artifact.metadata[
                    "camera_role_binding_hash"
                ],
                "camera_input_routing_hash": artifact.metadata[
                    "camera_input_routing_hash"
                ],
                "source_artifact_unchanged": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
