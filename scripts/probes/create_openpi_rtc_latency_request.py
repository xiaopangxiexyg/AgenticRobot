#!/usr/bin/env python3
"""Create a shape-representative conditioned OpenPI RTC request.

This is a synthetic wire fixture for transport/latency checks. It is not a
substitute for a real observation when making a model-quality claim. The
default state contract matches the current Stage1 masked deployment profile:
16 base state values followed by 16 all-available values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from deployments.openpi.client.rtc_conditioned import (
    OPENPI_CONDITIONED_RTC_OPERATION_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
)
from deployments.openpi.client.transport import packb

RAW_STATE_MODE = "raw_state_v1"
MASKED_STATE_MODE = "masked_normalized_concat_v1"


def build_request(
    *,
    state_dim: int = 16,
    state_input_mode: str = MASKED_STATE_MODE,
    action_horizon: int = 32,
    action_dim: int = 16,
    seed: int = 20260918,
    prompt: str = (
        "Pick up the capped brush with the right gripper and present it to "
        "the left gripper for uncapping."
    ),
) -> dict[str, Any]:
    """Build one deterministic synthetic request with a valid state contract."""

    if state_dim <= 0 or action_horizon <= 0 or action_dim <= 0:
        raise ValueError("state_dim, action_horizon, and action_dim must be positive.")
    if state_input_mode not in {RAW_STATE_MODE, MASKED_STATE_MODE}:
        raise ValueError(
            "state_input_mode must be 'raw_state_v1' or "
            "'masked_normalized_concat_v1'."
        )

    rng = np.random.default_rng(seed)
    images = {
        f"observation.images.{name}": rng.integers(
            0, 256, (480, 640, 3), dtype=np.uint8
        )
        for name in ("front", "left_wrist", "right_wrist")
    }
    state = np.zeros((state_dim,), dtype=np.float32)
    if state_input_mode == MASKED_STATE_MODE:
        state = np.concatenate(
            (state, np.ones((state_dim,), dtype=np.float32)), axis=0
        )
    return {
        "__openpi_request__": {
            "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
            "operation": OPENPI_CONDITIONED_RTC_OPERATION_V1,
        },
        "observation": {
            **images,
            "state": state,
            "prompt": prompt,
        },
        "conditioning_actions": np.zeros(
            (action_horizon, action_dim), dtype=np.float32
        ),
        "prefix_mask": np.zeros((action_horizon,), dtype=np.bool_),
        "conditioning_mode": "unconditioned_bootstrap_v1",
        "receipt_context": {"request_id": "synthetic-latency-fixture"},
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a synthetic conditioned OpenPI RTC request fixture."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dim", type=_positive_int, default=16)
    parser.add_argument(
        "--state-input-mode",
        choices=(RAW_STATE_MODE, MASKED_STATE_MODE),
        default=MASKED_STATE_MODE,
    )
    parser.add_argument("--action-horizon", type=_positive_int, default=32)
    parser.add_argument("--action-dim", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    request = build_request(
        state_dim=args.state_dim,
        state_input_mode=args.state_input_mode,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(packb(request))
    print(f"synthetic_request={output} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
