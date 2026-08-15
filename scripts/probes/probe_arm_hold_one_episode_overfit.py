"""No-write state-only overfit probe for one frozen arm-hold Episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_interface.arm_hold_split_manifest import load_arm_hold_split_manifest
from dataset_interface.arm_hold_windows import (
    ArmHoldWindowReader,
    build_arm_hold_window_inventory,
)
from model.arm_hold import (
    ArmHoldBaseline,
    ArmHoldBaselineConfig,
    arm_hold_constant_logits,
    arm_hold_tensor_batch,
    arm_hold_loss,
    arm_hold_window_evaluation,
)


def _selected_indices(
    reader: ArmHoldWindowReader,
    candidate_indices: Sequence[int],
    *,
    max_windows: int,
    required_side_index: int,
    required_target: int,
) -> tuple[tuple[int, ...], int]:
    required: list[int] = []
    other_transition: list[int] = []
    stable: list[int] = []
    for index in candidate_indices:
        sample = reader[index]
        required_mask = (
            sample.arm_hold_transition_mask[:, required_side_index]
            & (
                sample.arm_hold_target[:, required_side_index]
                == float(required_target)
            )
        )
        if np.any(required_mask):
            required.append(index)
        elif np.any(sample.arm_hold_transition_mask):
            other_transition.append(index)
        else:
            stable.append(index)
    selected = list(required[:max_windows])
    remaining = max_windows - len(selected)
    if remaining > 0:
        selected.extend(other_transition[:remaining])
        remaining = max_windows - len(selected)
    if remaining > 0 and stable:
        positions = np.linspace(
            0,
            len(stable) - 1,
            num=min(remaining, len(stable)),
            dtype=np.int64,
        )
        selected.extend(stable[int(position)] for position in positions)
    return tuple(sorted(set(selected))), len(required)


def _required_transition_target(value: str) -> tuple[int, int]:
    mapping = {
        "left:teleop->fixed_hold": (0, 1),
        "left:fixed_hold->teleop": (0, 0),
        "right:teleop->fixed_hold": (1, 1),
        "right:fixed_hold->teleop": (1, 0),
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(
            "required-transition must be one of " + ", ".join(sorted(mapping))
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read one train Episode from an immutable arm-hold manifest and "
            "overfit a tiny state-only head without writing artifacts."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-windows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Required acknowledgement: this probe must never start devices or motion.",
    )
    parser.add_argument(
        "--required-transition",
        default="left:fixed_hold->teleop",
        help="Episode-level transition direction required by the probe.",
    )
    args = parser.parse_args()
    if not args.no_motion:
        parser.error("--no-motion is required")
    if args.steps <= 0 or args.max_windows <= 0 or args.learning_rate <= 0:
        parser.error("steps, max-windows, and learning-rate must be positive")
    try:
        required_side_index, required_target = _required_transition_target(
            args.required_transition
        )
    except ValueError as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    manifest = load_arm_hold_split_manifest(args.manifest)
    inventory = build_arm_hold_window_inventory(
        manifest,
        args.workspace_root,
        split="train",
    )
    train_identities = set(manifest.cohort.split.train_episode_identities)
    candidates = tuple(
        record
        for record in manifest.cohort.included_episodes
        if record.episode_capture_identity_hash in train_identities
        and args.required_transition in record.transition_directions
    )
    if not candidates:
        raise RuntimeError(
            f"No train Episode covers {args.required_transition!r}."
        )
    locations_by_identity: dict[str, list[int]] = {}
    for index, location in enumerate(inventory.locations):
        locations_by_identity.setdefault(
            location.episode_capture_identity_hash, []
        ).append(index)
    record = None
    candidate_indices: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()
    required_candidate_window_count = 0
    samples = ()
    with ArmHoldWindowReader(inventory, input_mode="state_only") as reader:
        for candidate_record in candidates:
            possible_indices = tuple(
                locations_by_identity.get(
                    candidate_record.episode_capture_identity_hash, ()
                )
            )
            if not possible_indices:
                continue
            possible_selected, possible_required_count = _selected_indices(
                reader,
                possible_indices,
                max_windows=args.max_windows,
                required_side_index=required_side_index,
                required_target=required_target,
            )
            if possible_required_count == 0:
                continue
            record = candidate_record
            candidate_indices = possible_indices
            selected = possible_selected
            required_candidate_window_count = possible_required_count
            samples = tuple(reader[index] for index in selected)
            break
    if record is None:
        raise RuntimeError(
            f"None of {len(candidates)} train Episodes with Episode-level "
            f"{args.required_transition!r} has an admitted window containing it."
        )
    required_support = int(
        sum(
            np.count_nonzero(
                sample.arm_hold_transition_mask[:, required_side_index]
                & (
                    sample.arm_hold_target[:, required_side_index]
                    == float(required_target)
                )
            )
            for sample in samples
        )
    )
    if not samples or required_support == 0:
        raise RuntimeError(
            "Probe selection contains no keyframe for the required transition."
        )

    batch = arm_hold_tensor_batch(samples).to(device)
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=int(batch.observation_state.shape[1]),
            horizon=int(batch.target.shape[1]),
            state_feature_dim=64,
            fusion_dim=128,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    with torch.no_grad():
        initial_logits = model.forward_batch(batch)
        initial_loss = float(arm_hold_loss(initial_logits, batch).item())
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = arm_hold_loss(model.forward_batch(batch), batch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_logits = model.forward_batch(batch)
        final_loss = float(arm_hold_loss(final_logits, batch).item())
        constant_logits = arm_hold_constant_logits(batch)

    report = arm_hold_window_evaluation(
        logits=final_logits,
        target=batch.target,
        valid_mask=batch.valid_mask,
        loss_mask=batch.loss_mask,
        transition_mask=batch.transition_mask,
    )
    constant_report = arm_hold_window_evaluation(
        logits=constant_logits,
        target=batch.target,
        valid_mask=batch.valid_mask,
        loss_mask=batch.loss_mask,
        transition_mask=batch.transition_mask,
    )
    payload = {
        "probe": "arm_hold_one_episode_overfit_v1",
        "writes_artifacts": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "manifest_hash": manifest.semantic_content_hash,
        "profile_hash": manifest.profile.semantic_content_hash,
        "source_dataset_id": record.source_dataset_id,
        "source_episode_index": record.source_episode_index,
        "episode_capture_identity_hash": record.episode_capture_identity_hash,
        "required_transition": args.required_transition,
        "candidate_window_count": len(candidate_indices),
        "required_transition_candidate_window_count": (
            required_candidate_window_count
        ),
        "selected_window_count": len(samples),
        "selected_transition_element_count": int(batch.transition_mask.sum().item()),
        "selected_required_transition_element_count": required_support,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "model_evaluation": report,
        "constant_evaluation": constant_report,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
