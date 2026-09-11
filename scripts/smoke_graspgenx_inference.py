#!/usr/bin/env python3
"""Run one non-hardware DMTac-conditioned GraspGenX inference smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

SCHEMA_VERSION = "agenticrobot.graspgenx_gpu_smoke.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _bounded_int(*, minimum: int, maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"expected an integer in [{minimum}, {maximum}]"
            )
        return parsed

    return parse


def _bounded_float(*, minimum: float, maximum: float):
    def parse(value: str) -> float:
        parsed = float(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"expected a number in [{minimum}, {maximum}]"
            )
        return parsed

    return parse


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic synthetic cube through the loopback service "
            "or direct GraspGenX backend. This smoke never reads cameras or "
            "connects to robot hardware."
        ),
        epilog="This smoke never reads cameras or connects to robot hardware.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=_bounded_int(minimum=1, maximum=65_535), default=5556
    )
    parser.add_argument(
        "--timeout-ms",
        type=_bounded_int(minimum=1_000, maximum=600_000),
        default=120_000,
    )
    parser.add_argument(
        "--planner", choices=("graspmoe", "diffusion"), default="graspmoe"
    )
    parser.add_argument(
        "--mode",
        choices=("service", "in-process"),
        default="service",
        help=(
            "call the isolated loopback service by default; use in-process only "
            "from a compatible GraspGenX Python environment"
        ),
    )
    parser.add_argument(
        "--asset-bundle",
        type=Path,
        default=None,
        help="GraspGenX asset bundle required by --mode in-process",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="optional GraspGenX source checkout for --mode in-process",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "verify the in-process runtime contract (Python, Torch, CUDA, source "
            "revision, asset manifest, checkpoint layout) without loading model "
            "weights; requires --mode in-process and --asset-bundle"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-points",
        type=_bounded_int(minimum=16, maximum=1_000_000),
        default=4096,
    )
    parser.add_argument(
        "--num-grasps",
        type=_bounded_int(minimum=1, maximum=1_000),
        default=32,
    )
    parser.add_argument(
        "--top-k", type=_bounded_int(minimum=1, maximum=1_000), default=10
    )
    parser.add_argument(
        "--cube-edge-m",
        type=_bounded_float(minimum=0.005, maximum=0.065),
        default=0.05,
    )
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args(argv)


def _cube_surface_points(*, count: int, edge_m: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    half_edge = edge_m / 2.0
    points = rng.uniform(-half_edge, half_edge, size=(count, 3)).astype(np.float32)
    faces = rng.integers(0, 6, size=count)
    axes = faces // 2
    signs = np.where((faces % 2) == 0, -1.0, 1.0).astype(np.float32)
    points[np.arange(count), axes] = signs * half_edge
    return points


def run(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np

    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from deployments.perception.graspgenx_client import (
        GraspGenXClient,
        GraspGenXInProcessBackend,
    )
    from deployments.perception.graspgenx_runtime_contract import (
        verify_graspgenx_inprocess_runtime,
    )
    from deployments.perception.sam2_graspgenx_point_grasp import (
        DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        GraspGenXPointGraspModelSpec,
    )

    if args.verify_only:
        if args.mode != "in-process":
            raise ValueError("--verify-only applies only to --mode in-process")
        if args.asset_bundle is None:
            raise ValueError("--verify-only requires --asset-bundle")
        receipt = verify_graspgenx_inprocess_runtime(
            model_spec=GraspGenXPointGraspModelSpec.from_asset_bundle(
                args.asset_bundle
            ),
            source_root=args.source_root,
            device=args.device,
        )
        return receipt.to_mapping()

    points = _cube_surface_points(
        count=args.num_points,
        edge_m=args.cube_edge_m,
        seed=args.seed,
    )
    if args.mode == "service":
        backend = GraspGenXClient(
            host=args.host,
            port=args.port,
            timeout_ms=args.timeout_ms,
        )
    else:
        if args.asset_bundle is None:
            raise ValueError("--mode in-process requires --asset-bundle")
        backend = GraspGenXInProcessBackend(
            model_spec=GraspGenXPointGraspModelSpec.from_asset_bundle(
                args.asset_bundle
            ),
            source_root=args.source_root,
            device=args.device,
        )
    try:
        backend.start()
        started = time.perf_counter()
        grasps, scores = backend.infer_object(
            points,
            sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
            planner=args.planner,
            num_grasps=args.num_grasps,
            score_threshold=-1.0,
            top_k=args.top_k,
        )
        elapsed_seconds = time.perf_counter() - started
    finally:
        backend.close()

    if len(grasps) == 0:
        raise RuntimeError("GraspGenX returned no candidate for the smoke object.")
    rotations = grasps[:, :3, :3]
    orthogonality_error = max(
        float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        for rotation in rotations
    )
    determinant_error = max(
        float(abs(np.linalg.det(rotation) - 1.0)) for rotation in rotations
    )
    return {
        "schema": SCHEMA_VERSION,
        "status": "passed",
        "backend": backend.backend_id,
        "server": getattr(backend, "address", None),
        "planner": args.planner,
        "input": {
            "shape": "cube_surface",
            "points": len(points),
            "edge_m": float(args.cube_edge_m),
            "seed": int(args.seed),
            "gripper_profile": "DMTac-M-sweep-volume-v1",
        },
        "output": {
            "returned_grasps": len(grasps),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "first_translation_m": grasps[0, :3, 3].tolist(),
            "max_rotation_orthogonality_error": orthogonality_error,
            "max_rotation_determinant_error": determinant_error,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001 - stable CLI failure boundary
        print(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "status": "failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2))
    if result.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
