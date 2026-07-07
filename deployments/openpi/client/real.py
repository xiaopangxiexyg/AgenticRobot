from __future__ import annotations

import argparse
import json
from typing import Any


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
    parser.add_argument("--required-camera-slot", action="append", dest="required_camera_slots", default=None)
    parser.add_argument("--required-image-key", action="append", dest="required_image_keys", default=None)
    parser.add_argument("--hand-layout", default="q6_slot0_gripper")
    args, teleop_args = parser.parse_known_args(argv)

    policy_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": int(args.port),
        "api_key": args.api_key,
        "prompt": args.prompt,
        "expected_model_id": args.expected_model_id,
        "expected_checkpoint_step": args.expected_checkpoint_step,
        "action_dim": int(args.action_dim),
        "action_key": str(args.action_key),
        "required_camera_slots": tuple(args.required_camera_slots or ("base_0_rgb",)),
        "required_image_keys": tuple(args.required_image_keys or ("front",)),
        "hand_layout": args.hand_layout,
    }
    policy_kwargs = {key: value for key, value in policy_kwargs.items() if value is not None}

    from teleop_runtime.deploy.run_policy import main as run_real_policy

    return run_real_policy(
        [
            "--policy",
            "deployments.openpi.client.policy:create_policy",
            "--policy-kwargs-json",
            json.dumps(policy_kwargs, separators=(",", ":")),
            *teleop_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
