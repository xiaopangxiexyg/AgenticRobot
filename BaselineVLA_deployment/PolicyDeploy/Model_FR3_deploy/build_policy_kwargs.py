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


def build_policy_kwargs(*, require_tactile_default: bool) -> dict[str, Any]:
    return {
        "checkpoint": os.environ["CHECKPOINT"],
        "device": os.environ["DEVICE"],
        "camera_key": os.environ["CAMERA_KEY"],
        "tactile_side": os.environ["TACTILE_SIDE"],
        "require_image": _env_bool("REQUIRE_IMAGE", True),
        "require_tactile": _env_bool("REQUIRE_TACTILE", require_tactile_default),
        "strict_state_dim": _env_bool("STRICT_STATE_DIM", True),
        "action_clip_nan_to_hold": _env_bool("ACTION_CLIP_NAN_TO_HOLD", True),
        "action_chunk_size": _env_action_chunk_size(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ManipRep deploy policy kwargs JSON from env vars.")
    parser.add_argument("--require-tactile-default", choices=["0", "1"], required=True)
    args = parser.parse_args()
    payload = build_policy_kwargs(require_tactile_default=args.require_tactile_default == "1")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
