"""Command-line entrypoint for the first FormalCollection state BC fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .contracts import StateBCArtifactMetadata, StateBCSelector
from .dataset import load_state_action_rows, select_formal_collection, split_rows_by_episode
from .evaluate import action_metrics
from .model import fit_linear_state_bc


def train_from_formal_collection(
    source_dirs: Sequence[str | Path],
    *,
    artifact_path: str | Path,
    selector_id: str,
    test_fraction: float = 0.2,
    seed: int = 0,
    ridge: float = 1e-4,
    min_episode_frames: int = 2,
    selector_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit and save one artifact using only observation.state and action."""

    selector = StateBCSelector(
        selector_id=str(selector_id),
        source_dirs=tuple(str(Path(value).expanduser()) for value in source_dirs),
        min_episode_frames=int(min_episode_frames),
        metadata={} if selector_metadata is None else dict(selector_metadata),
    )
    selection = select_formal_collection(selector)
    rows = load_state_action_rows(selection)
    train_rows, test_rows = split_rows_by_episode(
        rows,
        test_fraction=float(test_fraction),
        seed=int(seed),
    )
    metadata = StateBCArtifactMetadata(
        selector_id=selector.selector_id,
        train_episode_count=train_rows.episode_count,
        test_episode_count=test_rows.episode_count,
        train_frame_count=train_rows.frame_count,
        test_frame_count=test_rows.frame_count,
        metadata={
            "selector_hash": selector.semantic_content_hash,
            "selection_hash": selection.semantic_content_hash,
            "seed": int(seed),
            "test_fraction": float(test_fraction),
            "ridge": float(ridge),
            "excluded_episode_count": len(selection.excluded_episodes),
        },
    )
    model = fit_linear_state_bc(
        train_rows.states,
        train_rows.actions,
        artifact_metadata=metadata,
        ridge=float(ridge),
    )
    train_predictions = [model.predict(state) for state in train_rows.states]
    fit_metrics: dict[str, Any] = {
        "train": action_metrics(train_predictions, train_rows.actions),
    }
    if test_rows.frame_count:
        test_predictions = [model.predict(state) for state in test_rows.states]
        fit_metrics["test"] = action_metrics(test_predictions, test_rows.actions)
    # Rebuild the immutable model with the metrics embedded in its artifact.
    from dataclasses import replace

    model = replace(model, fit_metrics=fit_metrics)
    saved = model.save(artifact_path)
    return {
        "artifact_path": str(saved),
        "artifact_hash": model.artifact_hash,
        "selector": selector.to_dict(),
        "selection": selection.to_dict(),
        "fit_metrics": fit_metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", required=True, help="FormalCollection directory; repeat for one logical day split into parts.")
    parser.add_argument("--selector-id", required=True)
    parser.add_argument("--artifact", required=True, help="Output .npz artifact path.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--min-episode-frames", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_from_formal_collection(
        args.source_dir,
        artifact_path=args.artifact,
        selector_id=args.selector_id,
        test_fraction=args.test_fraction,
        seed=args.seed,
        ridge=args.ridge,
        min_episode_frames=args.min_episode_frames,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke test.
    raise SystemExit(main())


__all__ = ["main", "train_from_formal_collection"]
