"""Command-line allocation of a short project-local Gateway endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_paths import repo_root

from .unix_endpoint import allocate_agent_gateway_unix_endpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway allocate-unix-endpoint",
        description=(
            "Allocate a short private AF_UNIX endpoint under DexTeleop/temp. "
            "Pass the returned path explicitly to the Gateway and its client."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "optional allocation root (default: <DexTeleop>/temp/ag); this "
            "advanced override must itself leave enough AF_UNIX path budget"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("path", "json"),
        default="path",
        help="stdout representation (default: path for shell assignment)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root() / "temp" / "ag" if args.root is None else args.root
    try:
        allocation = allocate_agent_gateway_unix_endpoint(endpoint_root=root)
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"cannot allocate Unix endpoint: {error}")

    if args.format == "json":
        print(json.dumps(allocation.as_dict(), sort_keys=True), flush=True)
    else:
        print(allocation.socket_path, flush=True)
    return 0
