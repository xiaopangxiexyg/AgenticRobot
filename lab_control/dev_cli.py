from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from . import git
from .dev_config import DevConfig, load_dev_config


def inspect_environment(
    repository: Path, *, config: DevConfig | None = None
) -> dict[str, object]:
    root = git.repository_root(repository)
    resolved_config = config or load_dev_config(root / "integration" / "dev.json")
    modules: dict[str, object] = {}
    provenance_valid = True
    for module_name in resolved_config.import_modules:
        spec = importlib.util.find_spec(module_name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        within_checkout = origin is not None and origin.is_relative_to(root)
        provenance_valid = provenance_valid and within_checkout
        modules[module_name] = {
            "origin": str(origin) if origin else None,
            "within_checkout": within_checkout,
        }
    actual_python = (sys.version_info.major, sys.version_info.minor)
    python_compatible = actual_python == resolved_config.python_major_minor
    status = (
        "READY" if python_compatible and provenance_valid else "INVALID_ENVIRONMENT"
    )
    return {
        "schema": "lab.dev.doctor.v1",
        "status": status,
        "repository": {
            "root": str(root),
            "branch": git.current_branch(root),
            "commit": git.current_commit(root),
            "tree_oid": git.tree_oid(root),
            "dirty": bool(git.status_short(root)),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "required_major_minor": (
                f"{resolved_config.python_major_minor[0]}."
                f"{resolved_config.python_major_minor[1]}"
            ),
            "compatible": python_compatible,
        },
        "imports": modules,
        "dev_config": {
            "path": str(resolved_config.path),
            "sha256": resolved_config.sha256,
        },
    }


def _build_parser(config: DevConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dev",
        description="Repository-local development entrypoint.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="check source and Python provenance")
    doctor.add_argument("--json", action="store_true")
    test = subcommands.add_parser("test", help="run a registered local test target")
    test.add_argument("target", choices=tuple(config.test_targets))
    return parser


def _print_doctor(payload: dict[str, object]) -> None:
    print(f"Status: {payload['status']}")
    repository = payload["repository"]
    python = payload["python"]
    if isinstance(repository, dict):
        print(f"Repository: {repository.get('root')}")
        print(f"Revision: {repository.get('commit')}")
        print(f"Dirty: {repository.get('dirty')}")
    if isinstance(python, dict):
        print(f"Python: {python.get('version')}")
        print(f"Required: {python.get('required_major_minor')}")
    imports = payload.get("imports")
    if isinstance(imports, dict):
        for name, raw_import in imports.items():
            if isinstance(raw_import, dict):
                print(
                    f"Import {name}: {raw_import.get('origin')} "
                    f"within_checkout={raw_import.get('within_checkout')}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    config = load_dev_config(repository / "integration" / "dev.json")
    arguments = _build_parser(config).parse_args(argv)
    if arguments.command == "doctor":
        payload = inspect_environment(repository, config=config)
        if arguments.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_doctor(payload)
        return 0 if payload["status"] == "READY" else 1
    if arguments.command == "test":
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *config.test_targets[arguments.target],
            ],
            cwd=repository,
            check=False,
        )
        return completed.returncode
    return 2
