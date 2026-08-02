#!/usr/bin/env python3
"""Inspect DMTac SDK symbols without constructing a sensor or opening devices.

This probe imports selected SDK implementation modules, then reports bounded
metadata for classes and callables.  It intentionally never instantiates a
class and never calls an SDK function.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SDK_ROOT = _PROJECT_ROOT / "assets" / "DM_Tac_SDK_Publish_1.2.10"

_DEFAULT_MODULES = (
    "dmrobotics.src.dmSDK",
    "dmrobotics.src.constructor",
    "dmrobotics.src.loader",
    "dmrobotics.src.force_loader",
    "dmrobotics.src.img2flow_impl",
    "dmrobotics.src.flow2density_impl",
    "dmrobotics.src.flow2shear_impl",
    "dmrobotics.src.flow2force_impl",
)

_RELEVANT_TOKENS = (
    "density",
    "depth",
    "dmv1",
    "flow",
    "force",
    "load",
    "process",
    "shear",
    "tracker",
)


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _is_relevant(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _RELEVANT_TOKENS)


def _class_summary(
    cls: type[Any], *, max_members: int, include_all_members: bool
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for name in sorted(dir(cls)):
        if (
            name.startswith("__")
            and name not in {"__call__", "__enter__", "__exit__"}
        ) or (not include_all_members and not _is_relevant(name)):
            continue
        try:
            value = inspect.getattr_static(cls, name)
        except Exception as exc:  # pragma: no cover - defensive SDK reflection
            members.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        item: dict[str, Any] = {
            "name": name,
            "kind": type(value).__name__,
        }
        signature = _safe_signature(value)
        if signature is not None:
            item["signature"] = signature
        members.append(item)
        if len(members) >= max_members:
            break
    return {
        "name": cls.__name__,
        "module": cls.__module__,
        "signature": _safe_signature(cls),
        "members": members,
    }


def _module_summary(
    module_name: str,
    *,
    max_symbols: int,
    max_members: int,
    include_all_members: bool,
    include_all_symbols: bool,
) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "import_error": f"{type(exc).__name__}: {exc}",
        }

    symbols: list[dict[str, Any]] = []
    for name in sorted(dir(module)):
        if name.startswith("_") or (not include_all_symbols and not _is_relevant(name)):
            continue
        try:
            value = inspect.getattr_static(module, name)
        except Exception as exc:  # pragma: no cover - defensive SDK reflection
            symbols.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if inspect.isclass(value):
            item = {
                "kind": "class",
                **_class_summary(
                    value,
                    max_members=max_members,
                    include_all_members=include_all_members,
                ),
            }
        elif callable(value):
            item = {
                "name": name,
                "kind": "callable",
                "module": getattr(value, "__module__", None),
                "signature": _safe_signature(value),
            }
        else:
            item = {
                "name": name,
                "kind": type(value).__name__,
                "module": getattr(value, "__module__", None),
            }
        symbols.append(item)
        if len(symbols) >= max_symbols:
            break
    return {
        "module": module_name,
        "file": str(getattr(module, "__file__", "")),
        "symbols": symbols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=_DEFAULT_SDK_ROOT)
    parser.add_argument("--module", action="append", dest="modules")
    parser.add_argument("--max-symbols", type=int, default=40)
    parser.add_argument("--max-members", type=int, default=40)
    parser.add_argument(
        "--all-members",
        action="store_true",
        help="Show all non-dunder members of relevant classes, not only force/flow names.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Show all public module symbols, not only force/flow names.",
    )
    args = parser.parse_args()

    sdk_root = args.sdk_root.expanduser().resolve()
    if not sdk_root.is_dir():
        raise SystemExit(f"SDK root does not exist: {sdk_root}")
    sys.path.insert(0, str(sdk_root))

    modules = tuple(args.modules or _DEFAULT_MODULES)
    result = {
        "probe_contract": {
            "instantiates_sdk_classes": False,
            "calls_sdk_functions": False,
            "opens_devices": False,
        },
        "sdk_root": str(sdk_root),
        "modules": [
            _module_summary(
                module_name,
                max_symbols=max(1, int(args.max_symbols)),
                max_members=max(1, int(args.max_members)),
                include_all_members=bool(args.all_members),
                include_all_symbols=bool(args.all_symbols),
            )
            for module_name in modules
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
