from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "none", "null"}:
        return False
    raise SystemExit(f"{name} must be a boolean value, got {value!r}.")


def _env_action_chunk_size() -> int | None:
    value = os.environ.get("ACTION_CHUNK_SIZE", "all").strip().lower()
    if value in {"", "0", "all", "full", "none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise SystemExit(f"ACTION_CHUNK_SIZE must be positive or all/0, got {value!r}.") from error
    if parsed <= 0:
        raise SystemExit(f"ACTION_CHUNK_SIZE must be positive or all/0, got {value!r}.")
    return parsed


def _checkpoint_from_env() -> str:
    for name in ("CHECKPOINT", "CKPT", "MODEL_CKPT"):
        value = os.environ.get(name)
        if value and value.strip():
            return value
    raise SystemExit("Set CHECKPOINT=/path/to/best.pt or CHECKPOINT=/path/to/checkpoints/epoch_XXXX.pt.")


def build_policy_kwargs(*, require_tactile_default: bool) -> dict[str, Any]:
    return {
        "checkpoint": _checkpoint_from_env(),
        "device": os.environ.get("DEVICE", "cuda"),
        "camera_key": os.environ.get("CAMERA_KEY", "front"),
        "tactile_side": os.environ.get("TACTILE_SIDE", "right"),
        "require_image": _env_bool("REQUIRE_IMAGE", True),
        "require_tactile": _env_bool("REQUIRE_TACTILE", require_tactile_default),
        "strict_state_dim": _env_bool("STRICT_STATE_DIM", True),
        "strict_realman_schema": _env_bool("STRICT_REALMAN_SCHEMA", True),
        "action_clip_nan_to_hold": _env_bool("ACTION_CLIP_NAN_TO_HOLD", True),
        "action_chunk_size": _env_action_chunk_size(),
        "active_sides": os.environ.get("ACTIVE_SIDES", "right"),
        "action_layout": os.environ.get("ACTION_LAYOUT", "auto"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ManipRep Realman deploy policy kwargs JSON from env vars.")
    parser.add_argument("--require-tactile-default", choices=["0", "1"], required=True)
    args = parser.parse_args()
    payload = build_policy_kwargs(require_tactile_default=args.require_tactile_default == "1")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

