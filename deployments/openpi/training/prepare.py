"""Prepare an additive trained model-RTC artifact template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deployments.policy_artifact import load_policy_artifact

from .artifact import (
    build_openpi_v22_absolute_model_rtc_artifact_template,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a trained model-RTC artifact template from a read-only "
            "inference-only OpenPI artifact."
        )
    )
    parser.add_argument("--source-artifact-json", type=Path, required=True)
    parser.add_argument("--output-artifact-json", type=Path, required=True)
    parser.add_argument("--output-model-id", required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument(
        "--active-side",
        action="append",
        choices=("left", "right"),
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_artifact_json.expanduser()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite trained artifact template: {output}"
        )
    artifact = build_openpi_v22_absolute_model_rtc_artifact_template(
        load_policy_artifact(args.source_artifact_json.expanduser()),
        output_model_id=str(args.output_model_id),
        source_dataset_root=args.source_dataset_root,
        active_sides=tuple(args.active_side or ("left", "right")),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            artifact.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": "openpi_trained_model_rtc_artifact_prepare_v1",
                "output": str(output.resolve()),
                "model_id": artifact.model_id,
                "source_artifact_unchanged": True,
                "checkpoint_step": None,
                "continuation_contract_hash": (
                    artifact.model_continuation.contract_hash
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
