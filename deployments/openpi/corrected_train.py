"""Launch ordinary OpenPI training with serial-resolved camera semantics.

This wrapper leaves historical OpenPI config names and source datasets
unchanged. It enables the DexTeleop dataset pipeline's explicit environment
contract, audits every wrist-bearing source before training, and then invokes
the existing OpenPI training entrypoint. The module stays outside the optional
model-RTC training package so OpenPI's lightweight environment does not import
robot-control dependencies during preflight.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from dataset_interface.camera_semantics import (
    CAMERA_ROLE_REGISTRY_ENV,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CAMERA_ROLE_REGISTRY = (
    _PROJECT_ROOT / "robot_motion/configs/rm65_camera_registry.json"
)
_CORRECTED_SUFFIX = "-corrected-v1"


def _split_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    if "--" not in values:
        raise ValueError(
            "Separate wrapper options from OpenPI training arguments with --."
        )
    separator = values.index("--")
    wrapper = values[:separator]
    training = values[separator + 1 :]
    if not training:
        raise ValueError("OpenPI training arguments must follow --.")
    return wrapper, training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch an ordinary OpenPI corrected-v1 config with read-only "
            "RealSense serial camera-role resolution."
        )
    )
    parser.add_argument(
        "--camera-role-registry",
        type=Path,
        default=_DEFAULT_CAMERA_ROLE_REGISTRY,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the OpenPI config and audit dataset camera semantics only.",
    )
    parser.add_argument(
        "--sample-check",
        action="store_true",
        help=(
            "With --dry-run, decode and transform one sample per concrete "
            "dataset source without creating a trainer or loading model weights."
        ),
    )
    return parser


def _camera_slots(data_factory: Any) -> tuple[str, ...]:
    for name in (
        "camera_slots",
        "dexteleop_v22_camera_slots",
        "dexteleop_v23_camera_slots",
    ):
        values = tuple(getattr(data_factory, name, ()) or ())
        if values:
            return tuple(str(value) for value in values)
    return ()


def _dataset_roots(data_factory: Any) -> tuple[Path, ...]:
    sources = tuple(getattr(data_factory, "dataset_sources", ()) or ())
    if sources:
        roots = tuple(
            Path(str(source.dataset_root)).expanduser().resolve()
            for source in sources
        )
    else:
        roots = tuple(
            Path(str(value)).expanduser().resolve()
            for name in (
                "dataset_root",
                "dexteleop_v22_dataset_root",
                "dexteleop_v23_dataset_root",
            )
            if (value := getattr(data_factory, name, None))
        )
    return tuple(dict.fromkeys(roots))


def _uses_wrist_cameras(data_factory: Any) -> bool:
    return any(
        "left_wrist" in slot or "right_wrist" in slot
        for slot in _camera_slots(data_factory)
    )


def _camera_audits(data_factory: Any, registry: Path) -> list[dict[str, Any]]:
    if not _uses_wrist_cameras(data_factory):
        return []
    binding = load_camera_role_binding(registry)
    roots = _dataset_roots(data_factory)
    if not roots:
        raise ValueError(
            "Corrected OpenPI wrist config has no explicit DexTeleop dataset root."
        )
    audits: list[dict[str, Any]] = []
    for root in roots:
        resolver = DatasetCameraSemanticResolver(root, binding)
        audits.append(
            {
                "camera_semantics": resolver.audit_summary(),
                "frame_index_alignment": (
                    resolver.require_frame_index_alignment()
                ),
            }
        )
    return audits


def _run_openpi_train(config: Any) -> None:
    import openpi

    checkout_root = Path(openpi.__file__).resolve().parents[2]
    train_path = checkout_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location(
        "_dexteleop_openpi_corrected_train",
        train_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenPI train module: {train_path}")
    module = importlib.util.module_from_spec(spec)
    module_name = spec.name
    had_previous = module_name in sys.modules
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if had_previous:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        raise
    train_main = getattr(module, "main", None)
    if not callable(train_main):
        raise TypeError("OpenPI scripts/train.py does not expose main(config).")
    caller_cwd = Path.cwd()
    try:
        os.chdir(checkout_root)
        train_main(config)
    finally:
        os.chdir(caller_cwd)


def _sample_checks(config: Any) -> list[dict[str, Any]]:
    from openpi.training import data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
        load_images=True,
    )
    concrete = tuple(getattr(dataset, "_datasets", ()) or (dataset,))
    checks: list[dict[str, Any]] = []
    try:
        for source in concrete:
            sample = source[0]
            pipeline = getattr(source, "_masked_pipeline", None)
            images = dict(sample.get("image", {}) or {})
            row: dict[str, Any] = {
                "dataset_root": str(getattr(source, "dataset_root", "")),
                "dataset_type": type(source).__name__,
                "image_keys": sorted(str(key) for key in images),
                "image_shapes": {
                    str(key): list(value.shape) for key, value in images.items()
                },
                "image_means": {
                    str(key): float(value.mean()) for key, value in images.items()
                },
            }
            if pipeline is not None and pipeline.camera_semantics is not None:
                episode_position, _row_index = source._index[0]
                episode_index = source._episodes[episode_position].episode_index
                resolution = pipeline.camera_semantics.resolution_for_episode(
                    episode_index
                )
                row["episode_index"] = int(episode_index)
                row["camera_semantic_status"] = resolution.status
                row["canonical_to_stored_key"] = (
                    resolution.canonical_to_stored_key
                )
            checks.append(row)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return checks


def main(argv: list[str] | None = None) -> int:
    wrapper_argv, training_argv = _split_args(
        sys.argv[1:] if argv is None else argv
    )
    args = _parser().parse_args(wrapper_argv)
    if args.sample_check and not args.dry_run:
        raise ValueError("--sample-check requires --dry-run.")
    registry = args.camera_role_registry.expanduser().resolve()
    binding = load_camera_role_binding(registry)
    os.environ[CAMERA_ROLE_REGISTRY_ENV] = str(registry)

    from openpi.training import config as openpi_config

    original_argv = sys.argv
    try:
        sys.argv = ["scripts/train.py", *training_argv]
        config = openpi_config.resolve_policy_metadata(openpi_config.cli())
    finally:
        sys.argv = original_argv
    if not str(config.name).endswith(_CORRECTED_SUFFIX):
        raise ValueError(
            "Serial-resolved ordinary training requires a versioned "
            f"{_CORRECTED_SUFFIX!r} OpenPI config; historical names remain frozen."
        )
    audits = _camera_audits(config.data, registry)
    sample_checks = _sample_checks(config) if args.sample_check else []
    print(
        json.dumps(
            {
                "schema": "dexteleop.openpi.corrected_training_preflight.v1",
                "dry_run": bool(args.dry_run),
                "config": str(config.name),
                "camera_role_registry": str(registry),
                "camera_role_binding_hash": binding.contract_hash,
                "camera_semantic_audits": audits,
                "sample_checks": sample_checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    _run_openpi_train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_camera_audits",
    "_camera_slots",
    "_dataset_roots",
    "_split_args",
    "_sample_checks",
    "_uses_wrist_cameras",
    "main",
]
