"""No-write lightweight arm-hold sampling/loss train-validation probe."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_FRONT_HISTORY_FREE_RUNNING_BATCH_LIMIT = 64

from dataset_interface.arm_hold_split_manifest import load_arm_hold_split_manifest
from dataset_interface.arm_hold_windows import (
    ArmHoldWindowInventory,
    ArmHoldWindowReader,
    build_arm_hold_window_inventory,
)
from model.arm_hold import (
    ArmHoldBaseline,
    ArmHoldBaselineConfig,
    ArmHoldLossConfig,
    arm_hold_chunk_sequence_evaluation,
    arm_hold_dwell_bucket_indices,
    arm_hold_fixed_lead_sequence_evaluation,
    arm_hold_free_running_evaluation,
    arm_hold_offset_zero_evaluation,
    arm_hold_natural_plus_random_once_batches,
    arm_hold_natural_plus_transition_once_batches,
    arm_hold_natural_window_batches,
    arm_hold_tensor_batch,
    arm_hold_loss,
    arm_hold_window_evaluation,
)


def _assert_transition_coverage(inventory: ArmHoldWindowInventory) -> None:
    directions = {
        "left:teleop->fixed_hold",
        "left:fixed_hold->teleop",
        "right:teleop->fixed_hold",
        "right:fixed_hold->teleop",
    }
    missing = sorted(
        direction
        for direction in directions
        if inventory.covered_transition_event_counts.get(direction, 0) == 0
        or inventory.transition_window_counts.get(direction, 0) == 0
    )
    if missing:
        raise RuntimeError(
            f"{inventory.split} lacks legal H-window coverage for {missing!r}."
        )


def _evaluate(
    *,
    model: ArmHoldBaseline,
    inventory: ArmHoldWindowInventory,
    reader: ArmHoldWindowReader,
    batch_size: int,
    device: torch.device,
    loss_config: ArmHoldLossConfig,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    loss_parts: list[torch.Tensor] = []
    transition_parts: list[torch.Tensor] = []
    history_previous_parts: list[torch.Tensor] = []
    history_available_parts: list[torch.Tensor] = []
    history_dwell_parts: list[torch.Tensor] = []
    history_sequence_previous_parts: list[torch.Tensor] = []
    history_sequence_available_parts: list[torch.Tensor] = []
    history_sequence_dwell_parts: list[torch.Tensor] = []
    metadata_parts: list[dict[str, Any]] = []
    loss_sum = 0.0
    element_count = 0
    batches = arm_hold_natural_window_batches(
        inventory.locations,
        batch_size=batch_size,
        seed=0,
        epoch=0,
        shuffle=False,
    )
    model.eval()
    with torch.inference_mode():
        for indices in batches:
            batch = arm_hold_tensor_batch(tuple(reader[index] for index in indices)).to(
                device
            )
            logits = model.forward_batch(batch)
            batch_loss = arm_hold_loss(logits, batch, loss_config)
            effective_count = int((batch.valid_mask & batch.loss_mask).sum().item())
            loss_sum += float(batch_loss.item()) * effective_count
            element_count += effective_count
            logits_parts.append(logits.cpu())
            target_parts.append(batch.target.cpu())
            valid_parts.append(batch.valid_mask.cpu())
            loss_parts.append(batch.loss_mask.cpu())
            transition_parts.append(batch.transition_mask.cpu())
            if batch.history_previous_mode is not None:
                assert batch.history_available_mask is not None
                assert batch.history_dwell_steps is not None
                history_previous_parts.append(batch.history_previous_mode.cpu())
                history_available_parts.append(batch.history_available_mask.cpu())
                history_dwell_parts.append(batch.history_dwell_steps.cpu())
            if batch.history_sequence_previous_mode is not None:
                assert batch.history_sequence_available_mask is not None
                assert batch.history_sequence_dwell_steps is not None
                history_sequence_previous_parts.append(
                    batch.history_sequence_previous_mode.cpu()
                )
                history_sequence_available_parts.append(
                    batch.history_sequence_available_mask.cpu()
                )
                history_sequence_dwell_parts.append(
                    batch.history_sequence_dwell_steps.cpu()
                )
            metadata_parts.extend(dict(item) for item in batch.metadata)
    arrays = {
        "logits": torch.cat(logits_parts),
        "target": torch.cat(target_parts),
        "valid_mask": torch.cat(valid_parts),
        "loss_mask": torch.cat(loss_parts),
        "transition_mask": torch.cat(transition_parts),
        "metadata": tuple(metadata_parts),
    }
    if history_previous_parts:
        arrays.update(
            {
                "history_previous_mode": torch.cat(history_previous_parts),
                "history_available_mask": torch.cat(history_available_parts),
                "history_dwell_steps": torch.cat(history_dwell_parts),
            }
        )
    elif history_sequence_previous_parts:
        arrays.update(
            {
                "history_sequence_previous_mode": torch.cat(
                    history_sequence_previous_parts
                ),
                "history_sequence_available_mask": torch.cat(
                    history_sequence_available_parts
                ),
                "history_sequence_dwell_steps": torch.cat(
                    history_sequence_dwell_parts
                ),
            }
        )
    return (
        loss_sum / element_count,
        arm_hold_window_evaluation(
            logits=arrays["logits"],
            target=arrays["target"],
            valid_mask=arrays["valid_mask"],
            loss_mask=arrays["loss_mask"],
            transition_mask=arrays["transition_mask"],
        ),
        arrays,
    )


def _history_context_summary(arrays: dict[str, Any]) -> dict[str, Any] | None:
    available = arrays.get("history_available_mask")
    dwell = arrays.get("history_dwell_steps")
    if available is None:
        available = arrays.get("history_sequence_available_mask")
        dwell = arrays.get("history_sequence_dwell_steps")
    if available is None or dwell is None:
        return None
    buckets = arm_hold_dwell_bucket_indices(dwell, available)
    if available.ndim == 3:
        return {
            "provenance": "previous_canonical_collected_mode_v1",
            "order": "oldest_to_newest_v1",
            "history_steps": 2,
            "step": {
                step_name: {
                    "side": {
                        side: {
                            "available_count": int(
                                available[:, step_index, side_index].sum().item()
                            ),
                            "unavailable_count": int(
                                (~available[:, step_index, side_index]).sum().item()
                            ),
                            "available_rate": float(
                                available[:, step_index, side_index]
                                .to(dtype=torch.float64)
                                .mean()
                                .item()
                            ),
                            "dwell_bucket_counts": {
                                str(bucket): int(
                                    (
                                        buckets[:, step_index, side_index]
                                        == bucket
                                    ).sum().item()
                                )
                                for bucket in range(7)
                            },
                        }
                        for side_index, side in enumerate(("left", "right"))
                    }
                }
                for step_index, step_name in enumerate(("oldest", "newest"))
            },
        }
    return {
        "provenance": "previous_canonical_collected_mode_v1",
        "side": {
            side: {
                "available_count": int(available[:, side_index].sum().item()),
                "unavailable_count": int((~available[:, side_index]).sum().item()),
                "available_rate": float(
                    available[:, side_index].to(dtype=torch.float64).mean().item()
                ),
                "dwell_bucket_counts": {
                    str(bucket): int(
                        (buckets[:, side_index] == bucket).sum().item()
                    )
                    for bucket in range(7)
                },
            }
            for side_index, side in enumerate(("left", "right"))
        },
    }


def _compact_sequence_report(report: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "scope": report["scope"],
        "window_count": report["window_count"],
        "episode_count": report["episode_count"],
        "per_side": {
            side: {
                "point_count": values["point_count"],
                "unsupported_boundary_transition_count": values[
                    "unsupported_boundary_transition_count"
                ],
                "transition": {
                    direction: {
                        "support": direction_values["support"],
                        "predicted_count": direction_values["predicted_count"],
                        "matched_exact": direction_values["matched_exact"],
                        "matched_within_tolerance": direction_values[
                            "matched_within_tolerance"
                        ],
                        "recall_exact": direction_values["recall_exact"],
                        "recall_within_tolerance": direction_values[
                            "recall_within_tolerance"
                        ],
                        "mean_signed_timing_offset_steps": direction_values[
                            "signed_timing_offset_steps"
                        ]["mean"],
                    }
                    for direction, direction_values in values["transition"].items()
                },
                "stable_false_switch_rate": values[
                    "stable_false_switch_rate"
                ],
                "unmatched_predicted_transition_rate": values[
                    "unmatched_predicted_transition_rate"
                ],
                "single_frame_interior_predicted_segment_rate": values[
                    "single_frame_interior_predicted_segment_rate"
                ],
                "median_interior_predicted_dwell_steps": values[
                    "interior_predicted_dwell_steps"
                ]["median"],
            }
            for side, values in report["per_side"].items()
        },
    }
    for key in (
        "lead_offset",
        "horizon",
        "continuity_group_count",
        "transition_occurrence_basis",
        "offset_zero_transition_policy",
    ):
        if key in report:
            payload[key] = report[key]
    return payload


def _compact_free_running_report(report: dict[str, Any]) -> dict[str, Any]:
    window = report["window_evaluation"]
    compact = {
        key: report[key]
        for key in (
            "schema",
            "scope",
            "context_mode",
            "feedback_semantics",
            "bootstrap_semantics",
            "threshold",
            "window_count",
            "rollout_segment_count",
            "bootstrap_context_window_count",
            "feedback_context_window_count",
        )
    } | {
        "window_evaluation": {
            "overall": window["overall"],
            "macro_side": window["macro_side"],
            "per_side": window["per_side"],
            "exact_transition": window["exact_transition"],
            "pair_exact_match_accuracy": window["pair"][
                "exact_match_accuracy"
            ],
        },
        "sequence_evaluation": _compact_sequence_report(
            report["sequence_evaluation"]
        ),
        "error_dynamics": report["error_dynamics"],
    }
    if "dwell_dynamics" in report:
        compact["dwell_dynamics"] = report["dwell_dynamics"]
    for key in ("inference_batch_size", "inference_segment_cohort_count"):
        if key in report:
            compact[key] = report[key]
    return compact


def _comparison_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the frozen comparison evidence without emitting debug-scale detail."""

    fixed_lead = payload["fixed_lead_sequence_evaluation"]
    selected_leads = tuple(
        index for index in (0, 7, 15) if index < len(fixed_lead)
    )
    final_validation = payload["final_validation"]
    return {
        key: payload[key]
        for key in (
            "probe",
            "writes_artifacts",
            "device",
            "torch_version",
            "manifest_hash",
            "profile_hash",
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "loss_mode",
            "sampling_mode",
            "input_mode",
            "fusion_mode",
            "label_context_mode",
            "resource_limits",
            "loss_config",
            "parameter_count",
            "train_window_count",
            "train_transition_window_union_count",
            "validation_window_count",
            "validation_history_context",
            "free_running_validation",
            "initial_validation_loss",
            "epochs_receipts",
            "elapsed_seconds",
        )
    } | {
        "final_validation": {
            "overall": final_validation["overall"],
            "macro_side": final_validation["macro_side"],
            "exact_transition": final_validation["exact_transition"],
            "pair": final_validation["pair"],
        },
        "fixed_lead_sequence_evaluation": [fixed_lead[index] for index in selected_leads],
        "within_output_chunk_sequence_evaluation": payload[
            "within_output_chunk_sequence_evaluation"
        ],
    }


def _gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Emit only the frozen promotion-gate evidence for low-noise probes."""

    final_validation = payload["final_validation"]
    final_epoch = payload["epochs_receipts"][-1]
    free_running = payload["free_running_validation"]
    free_window = free_running["window_evaluation"]
    free_sequence = free_running["sequence_evaluation"]
    free_error = free_running["error_dynamics"]
    return {
        key: payload[key]
        for key in (
            "probe",
            "writes_artifacts",
            "device",
            "torch_version",
            "manifest_hash",
            "profile_hash",
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "loss_mode",
            "sampling_mode",
            "input_mode",
            "fusion_mode",
            "label_context_mode",
            "resource_limits",
            "loss_config",
            "parameter_count",
            "train_window_count",
            "validation_window_count",
            "validation_history_context",
            "initial_validation_loss",
            "elapsed_seconds",
        )
    } | {
        "training_receipt": {
            key: final_epoch[key]
            for key in (
                "epoch",
                "train_loss",
                "validation_loss",
                "train_exposure_count",
                "bonus_exposure_count",
                "bonus_transition_window_count",
            )
        },
        "final_validation": {
            "macro_side": final_validation["macro_side"],
            "pair_exact_match_accuracy": final_validation["pair"][
                "exact_match_accuracy"
            ],
        },
        "free_running_validation": {
            "schema": free_running["schema"],
            "scope": free_running["scope"],
            "context_mode": free_running["context_mode"],
            "feedback_semantics": free_running["feedback_semantics"],
            "bootstrap_semantics": free_running["bootstrap_semantics"],
            "rollout_segment_count": free_running["rollout_segment_count"],
            "inference_batch_size": free_running.get("inference_batch_size"),
            "inference_segment_cohort_count": free_running.get(
                "inference_segment_cohort_count"
            ),
            "window_evaluation": {
                "macro_side": free_window["macro_side"],
                "pair_exact_match_accuracy": free_window[
                    "pair_exact_match_accuracy"
                ],
            },
            "sequence_evaluation": {
                "per_side": {
                    side: {
                        "transition": values["transition"],
                        "stable_false_switch_rate": values[
                            "stable_false_switch_rate"
                        ],
                        "unmatched_predicted_transition_rate": values[
                            "unmatched_predicted_transition_rate"
                        ],
                        "single_frame_interior_predicted_segment_rate": values[
                            "single_frame_interior_predicted_segment_rate"
                        ],
                        "median_interior_predicted_dwell_steps": values[
                            "median_interior_predicted_dwell_steps"
                        ],
                    }
                    for side, values in free_sequence["per_side"].items()
                }
            },
            "error_dynamics": {
                side: {
                    "constant_prediction_episode_count": values[
                        "constant_prediction_episode_count"
                    ],
                    "episode_count": values["episode_count"],
                    "error_run_recovery_rate": values[
                        "error_run_recovery_rate"
                    ],
                }
                for side, values in free_error.items()
            },
        },
    }


def _apply_resource_limits(
    *,
    cpu_threads: int | None,
    process_nice_adjustment: int,
    io_nice_level: int | None,
) -> dict[str, int | str | None]:
    if process_nice_adjustment:
        os.nice(process_nice_adjustment)
    if io_nice_level is not None:
        try:
            completed = subprocess.run(
                (
                    "ionice",
                    "-c",
                    "2",
                    "-n",
                    str(io_nice_level),
                    "-p",
                    str(os.getpid()),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "--io-nice-level requires the Linux ionice command."
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"ionice failed: {detail}")
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
    return {
        "cpu_threads": (
            torch.get_num_threads() if cpu_threads is not None else None
        ),
        "interop_threads": (
            torch.get_num_interop_threads() if cpu_threads is not None else None
        ),
        "process_nice_adjustment": process_nice_adjustment,
        "effective_process_nice": os.nice(0),
        "io_priority_class": (
            "best_effort" if io_nice_level is not None else None
        ),
        "io_nice_level": io_nice_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny hold head with a declared input/window exposure mode "
            "and evaluate the natural frozen validation split without writes."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--input-mode",
        choices=("state_only", "front_state"),
        default="state_only",
        help=(
            "Lightweight learner input. The state_only default preserves the "
            "original probe behavior."
        ),
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("concat_v1", "state_history_visual_residual_v1"),
        default="concat_v1",
        help=(
            "Input fusion for the lightweight learner. The visual residual "
            "mode is front_state-only and starts with exactly zero visual "
            "effect on the state/history logits."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        choices=("unweighted", "train_pos_weight"),
        default="unweighted",
        help="BCE class-weight mode, independent of the sampling-mode choice.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=(
            "natural",
            "natural_plus_random_once",
            "natural_plus_transition_once",
        ),
        default="natural",
        help="Training exposure mode; validation always remains natural.",
    )
    parser.add_argument(
        "--exact-transition-weight",
        type=float,
        default=1.0,
        help="Multiplicative BCE weight only where transition_mask is true.",
    )
    parser.add_argument(
        "--stable-persistence-weight",
        type=float,
        default=0.0,
        help=(
            "Additive adjacent-probability penalty only across stable effective "
            "target boundaries; zero preserves the BCE baseline."
        ),
    )
    parser.add_argument(
        "--label-context-mode",
        choices=(
            "none",
            "previous_mode",
            "previous_dwell_bucket",
            "previous_mode_dwell",
            "previous_two_dwell_bucket",
            "previous_two_mode_dwell",
        ),
        default="none",
        help=(
            "Optional teacher-forcing context derived from preceding canonical "
            "collected labels; it is not deployment-resolved history."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("full", "comparison", "gate"),
        default="full",
        help=(
            "full preserves the existing report; comparison emits only the frozen "
            "aggregate, lead 0/7/15, chunk, and context evidence; gate emits only "
            "the promotion-gate metrics and run identity."
        ),
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Optional Torch CPU thread cap for low-contention probes.",
    )
    parser.add_argument(
        "--process-nice-adjustment",
        type=int,
        default=0,
        help="Optional nonnegative POSIX nice adjustment for this probe process.",
    )
    parser.add_argument(
        "--io-nice-level",
        type=int,
        choices=range(8),
        default=None,
        help="Optional Linux best-effort ionice level (0 highest, 7 lowest).",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Required acknowledgement: this probe must never start devices or motion.",
    )
    args = parser.parse_args()
    if not args.no_motion:
        parser.error("--no-motion is required")
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or args.exact_transition_weight <= 0
        or args.stable_persistence_weight < 0
        or (args.cpu_threads is not None and args.cpu_threads <= 0)
        or not 0 <= args.process_nice_adjustment <= 19
    ):
        parser.error(
            "epochs, batch-size, learning-rate, and exact-transition-weight "
            "must be positive; stable-persistence-weight must be nonnegative; "
            "cpu-threads must be positive when set; process-nice-adjustment "
            "must be in [0,19]"
        )

    resource_limits = _apply_resource_limits(
        cpu_threads=args.cpu_threads,
        process_nice_adjustment=args.process_nice_adjustment,
        io_nice_level=args.io_nice_level,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    started = time.monotonic()
    manifest = load_arm_hold_split_manifest(args.manifest)
    train_inventory = build_arm_hold_window_inventory(
        manifest, args.workspace_root, split="train"
    )
    validation_inventory = build_arm_hold_window_inventory(
        manifest, args.workspace_root, split="validation"
    )
    _assert_transition_coverage(train_inventory)
    _assert_transition_coverage(validation_inventory)
    state_dims = {item.state_dim for item in train_inventory.episodes}
    state_dims.update(item.state_dim for item in validation_inventory.episodes)
    if len(state_dims) != 1:
        raise RuntimeError(f"Expected one state_dim, got {sorted(state_dims)!r}.")
    state_dim = next(iter(state_dims))
    horizon = int(manifest.profile.sample_view["horizon"])
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode=args.input_mode,
            state_dim=state_dim,
            horizon=horizon,
            state_feature_dim=64,
            fusion_dim=128,
            label_context_mode=args.label_context_mode,
            label_context_feature_dim=8,
            fusion_mode=args.fusion_mode,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.loss_mode == "train_pos_weight":
        left_positive_weight = (
            train_inventory.target_element_counts["left:teleop"]
            / train_inventory.target_element_counts["left:fixed_hold"]
        )
        right_positive_weight = (
            train_inventory.target_element_counts["right:teleop"]
            / train_inventory.target_element_counts["right:fixed_hold"]
        )
    else:
        left_positive_weight = 1.0
        right_positive_weight = 1.0
    loss_config = ArmHoldLossConfig(
        positive_weight_left=left_positive_weight,
        positive_weight_right=right_positive_weight,
        exact_transition_weight=args.exact_transition_weight,
        stable_persistence_weight=args.stable_persistence_weight,
    )
    train_target_counts = train_inventory.target_element_counts
    train_probability = torch.tensor(
        [
            train_target_counts["left:fixed_hold"]
            / (
                train_target_counts["left:teleop"]
                + train_target_counts["left:fixed_hold"]
            ),
            train_target_counts["right:fixed_hold"]
            / (
                train_target_counts["right:teleop"]
                + train_target_counts["right:fixed_hold"]
            ),
        ],
        dtype=torch.float64,
    )
    transition_union_count = sum(
        location.contains_transition for location in train_inventory.locations
    )

    front_history_enabled = (
        args.input_mode == "front_state" and args.label_context_mode != "none"
    )
    free_running_inference_batch_size = (
        min(args.batch_size, _FRONT_HISTORY_FREE_RUNNING_BATCH_LIMIT)
        if front_history_enabled
        else args.batch_size
    )
    validation_cache_size = (
        min(
            len(validation_inventory.episodes),
            free_running_inference_batch_size,
        )
        if front_history_enabled
        else 2
    )
    with ArmHoldWindowReader(
        train_inventory,
        input_mode=args.input_mode,
        include_history_context=args.label_context_mode != "none",
        history_context_steps=(
            2 if args.label_context_mode.startswith("previous_two_") else 1
        ),
        max_episode_cache=2,
        max_video_cache=2,
    ) as train_reader, ArmHoldWindowReader(
        validation_inventory,
        input_mode=args.input_mode,
        include_history_context=args.label_context_mode != "none",
        history_context_steps=(
            2 if args.label_context_mode.startswith("previous_two_") else 1
        ),
        max_episode_cache=validation_cache_size,
        max_video_cache=validation_cache_size,
    ) as validation_reader:
        initial_validation_loss, initial_report, _ = _evaluate(
            model=model,
            inventory=validation_inventory,
            reader=validation_reader,
            batch_size=args.batch_size,
            device=device,
            loss_config=loss_config,
        )
        epoch_receipts: list[dict[str, Any]] = []
        for epoch in range(args.epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_element_count = 0
            batch_args = {
                "batch_size": args.batch_size,
                "seed": args.seed,
                "epoch": epoch,
                "shuffle": True,
            }
            if args.sampling_mode == "natural_plus_transition_once":
                batches = arm_hold_natural_plus_transition_once_batches(
                    train_inventory.locations, **batch_args
                )
            elif args.sampling_mode == "natural_plus_random_once":
                batches = arm_hold_natural_plus_random_once_batches(
                    train_inventory.locations,
                    bonus_count=transition_union_count,
                    **batch_args,
                )
            else:
                batches = arm_hold_natural_window_batches(
                    train_inventory.locations, **batch_args
                )
            for indices in batches:
                batch = arm_hold_tensor_batch(
                    tuple(train_reader[index] for index in indices)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = arm_hold_loss(
                    model.forward_batch(batch), batch, loss_config
                )
                loss.backward()
                optimizer.step()
                effective = batch.valid_mask & batch.loss_mask
                effective_count = int(effective.sum().item())
                epoch_loss_sum += float(loss.item()) * effective_count
                epoch_element_count += effective_count
            validation_loss, validation_report, validation_arrays = _evaluate(
                model=model,
                inventory=validation_inventory,
                reader=validation_reader,
                batch_size=args.batch_size,
                device=device,
                loss_config=loss_config,
            )
            exposure_multiplicity = Counter(
                index for batch in batches for index in batch
            )
            epoch_receipts.append(
                {
                    "epoch": epoch,
                    "train_exposure_count": sum(len(batch) for batch in batches),
                    "bonus_exposure_count": sum(
                        count - 1 for count in exposure_multiplicity.values()
                    ),
                    "bonus_transition_window_count": sum(
                        count - 1
                        for index, count in exposure_multiplicity.items()
                        if train_inventory.locations[index].contains_transition
                    ),
                    "train_loss": epoch_loss_sum / epoch_element_count,
                    "validation_loss": validation_loss,
                    "validation_overall": validation_report["overall"],
                    "validation_macro_side": validation_report["macro_side"],
                    "validation_exact_transition": validation_report[
                        "exact_transition"
                    ],
                    "validation_pair_exact_match_accuracy": validation_report[
                        "pair"
                    ]["exact_match_accuracy"],
                }
            )
        if args.label_context_mode == "none":
            free_running_validation = _compact_free_running_report(
                arm_hold_offset_zero_evaluation(
                    logits=validation_arrays["logits"],
                    target=validation_arrays["target"],
                    valid_mask=validation_arrays["valid_mask"],
                    loss_mask=validation_arrays["loss_mask"],
                    transition_mask=validation_arrays["transition_mask"],
                    metadata=validation_arrays["metadata"],
                    locations=validation_inventory.locations,
                    threshold=0.5,
                    transition_tolerance_steps=1,
                )
            )
        elif args.label_context_mode in (
            "previous_mode",
            "previous_dwell_bucket",
            "previous_mode_dwell",
            "previous_two_dwell_bucket",
            "previous_two_mode_dwell",
        ):
            free_running_validation = _compact_free_running_report(
                arm_hold_free_running_evaluation(
                    model=model,
                    locations=validation_inventory.locations,
                    sample_at=validation_reader.__getitem__,
                    device=device,
                    threshold=0.5,
                    transition_tolerance_steps=1,
                    inference_batch_size=free_running_inference_batch_size,
                )
            )
        else:
            raise AssertionError(
                f"Unhandled label context mode {args.label_context_mode!r}."
            )

    constant_side_logits = torch.logit(
        train_probability.clamp(1e-6, 1.0 - 1e-6)
    ).to(dtype=torch.float32)
    constant_logits = constant_side_logits.reshape(1, 1, 2).expand_as(
        validation_arrays["target"]
    )
    constant_report = arm_hold_window_evaluation(
        logits=constant_logits,
        target=validation_arrays["target"],
        valid_mask=validation_arrays["valid_mask"],
        loss_mask=validation_arrays["loss_mask"],
        transition_mask=validation_arrays["transition_mask"],
    )
    fixed_lead_sequence = [
        _compact_sequence_report(
            arm_hold_fixed_lead_sequence_evaluation(
                logits=validation_arrays["logits"],
                target=validation_arrays["target"],
                valid_mask=validation_arrays["valid_mask"],
                loss_mask=validation_arrays["loss_mask"],
                transition_mask=validation_arrays["transition_mask"],
                metadata=validation_arrays["metadata"],
                lead_offset=lead_offset,
                threshold=0.5,
                transition_tolerance_steps=1,
                debug=False,
            )
        )
        for lead_offset in range(horizon)
    ]
    chunk_sequence = _compact_sequence_report(
        arm_hold_chunk_sequence_evaluation(
            logits=validation_arrays["logits"],
            target=validation_arrays["target"],
            valid_mask=validation_arrays["valid_mask"],
            loss_mask=validation_arrays["loss_mask"],
            transition_mask=validation_arrays["transition_mask"],
            metadata=validation_arrays["metadata"],
            threshold=0.5,
            transition_tolerance_steps=1,
            debug=False,
        )
    )
    payload = {
        "probe": (
            "arm_hold_state_only_train_validation_v1"
            if args.input_mode == "state_only"
            else "arm_hold_front_state_train_validation_v1"
        ),
        "writes_artifacts": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "manifest_hash": manifest.semantic_content_hash,
        "profile_hash": manifest.profile.semantic_content_hash,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_mode": args.loss_mode,
        "sampling_mode": args.sampling_mode,
        "input_mode": args.input_mode,
        "fusion_mode": args.fusion_mode,
        "label_context_mode": args.label_context_mode,
        "resource_limits": resource_limits,
        "loss_config": {
            "positive_weight_left": loss_config.positive_weight_left,
            "positive_weight_right": loss_config.positive_weight_right,
            "exact_transition_weight": loss_config.exact_transition_weight,
            "stable_persistence_weight": loss_config.stable_persistence_weight,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_window_count": len(train_inventory.locations),
        "train_transition_window_union_count": transition_union_count,
        "validation_window_count": len(validation_inventory.locations),
        "validation_history_context": _history_context_summary(
            validation_arrays
        ),
        "free_running_validation": free_running_validation,
        "train_transition_window_counts": dict(
            train_inventory.transition_window_counts
        ),
        "validation_transition_window_counts": dict(
            validation_inventory.transition_window_counts
        ),
        "initial_validation_loss": initial_validation_loss,
        "initial_validation_overall": initial_report["overall"],
        "train_positive_probability": train_probability.tolist(),
        "train_target_element_counts": dict(train_target_counts),
        "constant_validation": {
            "overall": constant_report["overall"],
            "macro_side": constant_report["macro_side"],
            "per_side": constant_report["per_side"],
            "exact_transition": constant_report["exact_transition"],
            "pair_exact_match_accuracy": constant_report["pair"][
                "exact_match_accuracy"
            ],
        },
        "epochs_receipts": epoch_receipts,
        "final_validation": validation_report,
        "fixed_lead_sequence_evaluation": fixed_lead_sequence,
        "within_output_chunk_sequence_evaluation": chunk_sequence,
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.output_mode == "comparison":
        output = _comparison_payload(payload)
    elif args.output_mode == "gate":
        output = _gate_payload(payload)
    else:
        output = payload
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
