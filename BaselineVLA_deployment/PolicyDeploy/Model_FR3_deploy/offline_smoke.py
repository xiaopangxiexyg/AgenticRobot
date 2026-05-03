from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .policy import Policy


@dataclass
class FakeDeploymentObservation:
    state: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)
    tactile_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def hold_action(self) -> np.ndarray:
        return np.nan_to_num(self.state.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)


def _load_first_row(parquet_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("offline_smoke.py requires pyarrow in the active environment.") from error
    return pq.read_table(parquet_path).slice(0, 1).to_pylist()[0]


def build_observation(parquet_path: Path, *, camera_key: str) -> FakeDeploymentObservation:
    row = _load_first_row(parquet_path)
    state = np.asarray(row["observation.state"], dtype=np.float32)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    tactile = row.get("observation.hand.right.tactile")
    tactile_by_side = {}
    if tactile is not None:
        tactile_by_side["right"] = np.asarray(tactile, dtype=np.float32)
    return FakeDeploymentObservation(
        state=state,
        images={camera_key: image},
        tactile_by_side=tactile_by_side,
        metadata={"task_text": row.get("task", "unknown_task")},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline smoke-test a ManipRep FR3 deployment checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("Data/RawData/RealRobot/chips_med/data/chunk-000/episode_000000.parquet"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument(
        "--action-chunk-size",
        type=str,
        default=None,
        help="all/0 keeps the full model horizon; a positive integer keeps the first N actions.",
    )
    args = parser.parse_args()

    policy = Policy(
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        camera_key=str(args.camera_key),
        action_chunk_size=args.action_chunk_size,
    )
    observation = build_observation(Path(args.parquet), camera_key=str(args.camera_key))
    output = policy.predict(observation)
    action = np.asarray(output["action_chunk"], dtype=np.float32)
    print("checkpoint", Path(args.checkpoint).expanduser().resolve())
    print("status", policy.status_text)
    print("action_shape", tuple(action.shape))
    print("action_finite", bool(np.all(np.isfinite(action))))
    print("action_first", action[0].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
