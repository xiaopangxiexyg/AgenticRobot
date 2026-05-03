from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_commands(include_schema_switch: bool) -> list[list[str]]:
    commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "Training/train.py"),
            "experiment=vanilla",
            "train.run_name=mock_vanilla_13d",
            "data.raw.filters.state_dim=13",
            "data.raw.filters.action_dim=13",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "Training/train.py"),
            "experiment=gtmed",
            "train.run_name=mock_gtmed_13d",
            "data.raw.filters.state_dim=13",
            "data.raw.filters.action_dim=13",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "Training/train.py"),
            "experiment=fulltrain",
            "train.run_name=mock_fulltrain_13d",
            "data.raw.filters.state_dim=13",
            "data.raw.filters.action_dim=13",
        ],
    ]
    if include_schema_switch:
        commands.append(
            [
                sys.executable,
                str(PROJECT_ROOT / "Training/train.py"),
                "experiment=vanilla",
                "train.run_name=mock_vanilla_8d",
                "train.epochs=1",
                "data.raw.filters.state_dim=8",
                "data.raw.filters.action_dim=8",
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ManipRep mock smoke experiments.")
    parser.add_argument("--skip-schema-switch", action="store_true", help="Skip the extra 8D schema switch smoke run.")
    args = parser.parse_args()

    commands = build_commands(include_schema_switch=not args.skip_schema_switch)
    for command in commands:
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
