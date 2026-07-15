from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deployments.client import load_policy_artifact


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _image_key_map_json(value: str) -> dict[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--image-key-map-json must be a JSON object.") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--image-key-map-json must be a JSON object.")
    normalized: dict[str, str] = {}
    for source_key, model_input_key in payload.items():
        source = str(source_key).strip()
        destination = str(model_input_key).strip()
        if not source or not destination:
            raise argparse.ArgumentTypeError(
                "--image-key-map-json keys and values must be non-empty strings."
            )
        normalized[source] = destination
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the OpenPI client through the real teleop_runtime backend.",
        add_help=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", "--expected-model-id", dest="expected_model_id", default=None)
    parser.add_argument("--checkpoint-step", "--expected-checkpoint-step", dest="expected_checkpoint_step", default=None)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--action-key", default="actions")
    parser.add_argument(
        "--policy-artifact-json",
        type=Path,
        default=None,
        help=(
            "Checkpoint dexteleop_policy_artifact_v1 sidecar. Its v2.2 model, "
            "state/action, and camera bindings are passed to both the OpenPI "
            "client and the shared deployment runtime."
        ),
    )
    parser.add_argument("--required-camera-slot", action="append", dest="required_camera_slots", default=None)
    parser.add_argument("--required-image-key", action="append", dest="required_image_keys", default=None)
    parser.add_argument(
        "--image-key-map-json",
        type=_image_key_map_json,
        default=None,
        help=(
            "Explicit DexTeleop logical-camera to OpenPI input-key map, for example "
            "'{\"left_wrist\":\"observation.images.base_0_rgb\"}'. "
            "Normally supplied by --policy-artifact-json."
        ),
    )
    parser.add_argument("--hand-layout", default="q6_slot0_gripper")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write deploy action/pose debug JSONL through teleop_runtime.deploy.",
    )
    parser.add_argument(
        "--runtime-help",
        action="store_true",
        help="Show the delegated real-runtime mode, rate, safety, and hardware options.",
    )
    args, teleop_args = parser.parse_known_args(argv)

    if args.runtime_help:
        from teleop_runtime.deploy.run_policy import main as run_real_policy

        return run_real_policy(["--help"])

    policy_artifact = (
        None
        if args.policy_artifact_json is None
        else load_policy_artifact(args.policy_artifact_json)
    )
    policy_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": int(args.port),
        "api_key": _optional_text(args.api_key),
        "prompt": _optional_text(args.prompt),
        "expected_model_id": _optional_text(args.expected_model_id),
        "expected_checkpoint_step": _optional_text(args.expected_checkpoint_step),
        "action_dim": int(args.action_dim),
        "action_key": str(args.action_key),
        "required_camera_slots": (
            None
            if args.required_camera_slots is None
            else tuple(args.required_camera_slots)
        ),
        "required_image_keys": (
            None if args.required_image_keys is None else tuple(args.required_image_keys)
        ),
        "image_key_map": args.image_key_map_json,
        "hand_layout": _optional_text(args.hand_layout),
        "policy_artifact": (
            None if policy_artifact is None else policy_artifact.to_dict()
        ),
    }
    policy_kwargs = {key: value for key, value in policy_kwargs.items() if value is not None}

    from teleop_runtime.deploy.run_policy import main as run_real_policy

    runtime_args = list(teleop_args)
    if args.policy_artifact_json is not None:
        runtime_args.extend(("--policy-artifact-json", str(args.policy_artifact_json)))
    if args.debug:
        runtime_args.append("--debug")

    return run_real_policy(
        [
            "--policy",
            "deployments.openpi.client.policy:create_policy",
            "--policy-kwargs-json",
            json.dumps(policy_kwargs, separators=(",", ":")),
            *runtime_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
