"""No-write raw-resolution front/state batch and video-access benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_interface.arm_hold_split_manifest import load_arm_hold_split_manifest
from dataset_interface.arm_hold_windows import (
    ArmHoldWindowInventory,
    ArmHoldWindowReader,
    build_arm_hold_window_inventory,
)
from model.arm_hold import (
    ArmHoldBaseline,
    ArmHoldBaselineConfig,
    arm_hold_loss,
    arm_hold_tensor_batch,
)


def _batch_sizes(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch-sizes must be comma-separated integers"
        ) from exc
    if not values or values[0] <= 0 or values[-1] > 256:
        raise argparse.ArgumentTypeError(
            "batch-sizes must be unique positive values no larger than 256"
        )
    return values


def _episode_indices(
    inventory: ArmHoldWindowInventory,
    *,
    minimum_count: int,
) -> tuple[int, ...]:
    grouped: dict[str, list[int]] = {}
    for index, location in enumerate(inventory.locations):
        grouped.setdefault(location.episode_capture_identity_hash, []).append(index)
    candidates = [
        tuple(indices)
        for _, indices in sorted(grouped.items())
        if len(indices) >= minimum_count
    ]
    if not candidates:
        raise RuntimeError(
            f"No train Episode has at least {minimum_count} admitted windows."
        )
    return candidates[0]


def _case_indices(
    episode_indices: tuple[int, ...],
    *,
    order: str,
    batch_size: int,
    seed: int,
) -> tuple[int, ...]:
    values = list(episode_indices)
    if order == "shuffled_within_episode":
        random.Random(seed).shuffle(values)
    elif order != "canonical_within_episode":
        raise ValueError(f"Unsupported benchmark order {order!r}.")
    return tuple(values[:batch_size])


def _gpu_memory(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_case(
    *,
    inventory: ArmHoldWindowInventory,
    indices: tuple[int, ...],
    order: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started_read = time.monotonic()
    with ArmHoldWindowReader(
        inventory,
        input_mode="front_state",
        include_history_context=False,
        max_episode_cache=1,
        max_video_cache=1,
    ) as reader:
        samples = tuple(reader[index] for index in indices)
    read_seconds = time.monotonic() - started_read

    started_collate = time.monotonic()
    host_batch = arm_hold_tensor_batch(samples)
    collate_seconds = time.monotonic() - started_collate
    assert host_batch.front_rgb is not None
    assert host_batch.observation_state is not None
    image_shape = tuple(int(value) for value in host_batch.front_rgb.shape)

    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="front_state",
            state_dim=int(host_batch.observation_state.shape[1]),
            horizon=int(host_batch.target.shape[1]),
            image_feature_dim=64,
            state_feature_dim=64,
            fusion_dim=128,
            label_context_mode="none",
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _synchronize(device)
    started_transfer = time.monotonic()
    batch = host_batch.to(device)
    _synchronize(device)
    transfer_seconds = time.monotonic() - started_transfer

    _synchronize(device)
    started_step = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss = arm_hold_loss(model.forward_batch(batch), batch)
    loss.backward()
    optimizer.step()
    _synchronize(device)
    step_seconds = time.monotonic() - started_step
    finite_gradients = all(
        parameter.grad is not None and bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in model.parameters()
    )
    return {
        "status": "ok",
        "order": order,
        "batch_size": len(indices),
        "image_batch_shape": image_shape,
        "state_dim": int(host_batch.observation_state.shape[1]),
        "horizon": int(host_batch.target.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "read_seconds": read_seconds,
        "collate_seconds": collate_seconds,
        "transfer_seconds": transfer_seconds,
        "forward_backward_step_seconds": step_seconds,
        "loss": float(loss.item()),
        "finite_gradients": finite_gradients,
        "gpu_memory": _gpu_memory(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark raw 480x640 front/state decoding and one training step "
            "without writing artifacts or starting devices."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=_batch_sizes("1,4,8,16,32,64"),
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Required acknowledgement: this probe never starts devices or motion.",
    )
    args = parser.parse_args()
    if not args.no_motion:
        parser.error("--no-motion is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    manifest = load_arm_hold_split_manifest(args.manifest)
    inventory = build_arm_hold_window_inventory(
        manifest, args.workspace_root, split="train"
    )
    episode_indices = _episode_indices(
        inventory, minimum_count=max(args.batch_sizes)
    )
    selected_location = inventory.locations[episode_indices[0]]
    cases: list[dict[str, Any]] = []
    for order in ("canonical_within_episode", "shuffled_within_episode"):
        for batch_size in args.batch_sizes:
            indices = _case_indices(
                episode_indices,
                order=order,
                batch_size=batch_size,
                seed=args.seed,
            )
            try:
                cases.append(
                    _run_case(
                        inventory=inventory,
                        indices=indices,
                        order=order,
                        seed=args.seed,
                        device=device,
                    )
                )
            except torch.cuda.OutOfMemoryError as exc:
                cases.append(
                    {
                        "status": "cuda_oom",
                        "order": order,
                        "batch_size": batch_size,
                        "error_type": type(exc).__name__,
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            finally:
                gc.collect()

    payload = {
        "probe": "arm_hold_front_state_raw_batch_benchmark_v1",
        "writes_artifacts": False,
        "augmentation": "disabled",
        "resize": None,
        "device": str(device),
        "torch_version": torch.__version__,
        "manifest_hash": manifest.semantic_content_hash,
        "profile_hash": manifest.profile.semantic_content_hash,
        "train_window_count": len(inventory.locations),
        "source_dataset_id": selected_location.source_dataset_id,
        "source_episode_index": selected_location.source_episode_index,
        "episode_capture_identity_hash": (
            selected_location.episode_capture_identity_hash
        ),
        "episode_admitted_window_count": len(episode_indices),
        "batch_sizes": list(args.batch_sizes),
        "seed": args.seed,
        "cases": cases,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
