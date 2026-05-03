from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .policy import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    REALMAN_ACTION_DIM,
    REALMAN_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
    ManipRepRealmanPolicy,
)


@dataclass
class FakeDeploymentObservation:
    active_sides: tuple[str, ...]
    arms_enabled: bool = True
    hands_enabled: bool = True
    state: np.ndarray = field(default_factory=lambda: np.zeros((REALMAN_STATE_DIM,), dtype=np.float32))
    images: dict[str, np.ndarray] = field(default_factory=dict)
    tactile_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros((REALMAN_ACTION_DIM,), dtype=np.float32))
    metadata: dict[str, Any] = field(default_factory=dict)

    def hold_action(self) -> np.ndarray:
        action = np.zeros((REALMAN_ACTION_DIM,), dtype=np.float32)
        for side in ALL_SIDES:
            action[ACTION_ARM_POSE_SLICES[side]] = self._pose_from_state(side)
            action[ACTION_HAND_SLICES[side]] = 0.0
        return action

    def _pose_from_state(self, side: str) -> np.ndarray:
        if side == "left":
            return np.array([0.30, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.30, -0.20, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _parse_active_sides(value: str) -> tuple[str, ...]:
    text = str(value).strip().lower()
    if text == "both":
        return ("left", "right")
    if text in {"left", "right"}:
        return (text,)
    raise argparse.ArgumentTypeError("active-sides must be one of: right, left, both")


def _fake_observation(*, active_sides: tuple[str, ...], camera_key: str, with_tactile: bool) -> FakeDeploymentObservation:
    obs = FakeDeploymentObservation(active_sides=active_sides)
    obs.images[camera_key] = np.zeros((720, 1280, 3), dtype=np.uint8)
    obs.metadata["task_text"] = "unknown_task"
    for side in ALL_SIDES:
        obs.state[STATE_ARM_QPOS_SLICES[side]] = 0.0
        obs.state[STATE_ARM_POSE_SLICES[side]] = obs._pose_from_state(side)
        obs.state[STATE_HAND_SLICES[side]] = 0.0
    if with_tactile:
        obs.tactile_by_side["right"] = np.zeros((1062,), dtype=np.float32)
        obs.tactile_by_side["left"] = np.zeros((1062,), dtype=np.float32)
    return obs


def _summarize_payload(payload: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    summary: dict[str, tuple[int, ...]] = {}
    for key, value in payload.items():
        if key == "metadata":
            continue
        summary[key] = tuple(np.asarray(value).shape)
    return summary


def _try_normalize(payload: dict[str, Any], observation: FakeDeploymentObservation) -> tuple[int, tuple[int, ...] | None]:
    try:
        from teleop_runtime.deploy.actions import normalize_policy_output
    except Exception:
        return 0, None
    actions = normalize_policy_output(
        payload,
        observation,
        active_sides=observation.active_sides,
        arms_enabled=observation.arms_enabled,
        hands_enabled=observation.hands_enabled,
    )
    if not actions:
        return 0, None
    return len(actions), tuple(np.asarray(actions[0].vector).shape)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline smoke-test a ManipRep Realman deployment checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument("--active-sides", type=_parse_active_sides, default=("right",))
    parser.add_argument("--action-layout", type=str, default="auto")
    parser.add_argument(
        "--action-chunk-size",
        type=str,
        default=None,
        help="all/0 keeps the full model horizon; a positive integer keeps the first N actions.",
    )
    parser.add_argument("--no-tactile", action="store_true", help="Do not add fake 1062-D tactile vectors.")
    parser.add_argument("--require-tactile", action="store_true", default=False)
    args = parser.parse_args()

    policy = ManipRepRealmanPolicy(
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        camera_key=str(args.camera_key),
        active_sides=args.active_sides,
        action_layout=str(args.action_layout),
        action_chunk_size=args.action_chunk_size,
        require_tactile=bool(args.require_tactile),
    )
    observation = _fake_observation(
        active_sides=args.active_sides,
        camera_key=str(args.camera_key),
        with_tactile=not bool(args.no_tactile),
    )
    output = policy.predict(observation)
    finite = True
    for key, value in output.items():
        if key == "metadata":
            continue
        finite = finite and bool(np.all(np.isfinite(np.asarray(value, dtype=np.float32))))
    normalized_count, normalized_shape = _try_normalize(output, observation)
    print("checkpoint", Path(args.checkpoint).expanduser().resolve())
    print("status", policy.status_text)
    print("payload_shapes", _summarize_payload(output))
    print("payload_finite", finite)
    print("normalized_count", normalized_count)
    print("normalized_first_shape", normalized_shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
