"""Lazy, exact-version binding for the isolated robosuite A1 runtime."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import sysconfig
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROBOSUITE_A1_PACKAGE_VERSIONS = MappingProxyType(
    {
        "robosuite": "1.5.2",
        "mujoco": "3.9.0",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
    }
)
_PACKAGE_ROOTS = MappingProxyType(
    {
        "robosuite": ("robosuite/",),
        "mujoco": ("mujoco/",),
        "numpy": ("numpy/", "numpy.libs/"),
        "scipy": ("scipy/", "scipy.libs/"),
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_fingerprint(distribution_name: str) -> Mapping[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    files = distribution.files
    if files is None:
        raise RuntimeError(
            f"{distribution_name} does not expose an installed file manifest."
        )
    roots = _PACKAGE_ROOTS[distribution_name]
    entries: list[tuple[str, str | None, str | None, int | None]] = []
    for package_path in files:
        relative_name = str(package_path).replace("\\", "/")
        if ".." in Path(relative_name).parts or not relative_name.startswith(roots):
            continue
        record_hash = package_path.hash
        entries.append(
            (
                relative_name,
                None if record_hash is None else record_hash.mode,
                None if record_hash is None else record_hash.value,
                None if package_path.size is None else int(package_path.size),
            )
        )
    if not entries:
        raise RuntimeError(
            f"{distribution_name} has no package-owned installed file records."
        )
    wheel_text = distribution.read_text("WHEEL") or ""
    wheel_tags = tuple(
        sorted(
            line.partition(":")[2].strip()
            for line in wheel_text.splitlines()
            if line.startswith("Tag:") and line.partition(":")[2].strip()
        )
    )
    if not wheel_tags:
        raise RuntimeError(f"{distribution_name} has no installed wheel tags.")
    manifest_bytes = json.dumps(
        sorted(entries),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return MappingProxyType(
        {
            "version": distribution.version,
            "wheel_tags": wheel_tags,
            "package_record_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "package_record_count": len(entries),
        }
    )


def _verified_mujoco_native_fingerprint() -> tuple[Mapping[str, str], ...]:
    distribution = importlib.metadata.distribution("mujoco")
    files = distribution.files
    if files is None:
        raise RuntimeError("mujoco does not expose an installed file manifest.")
    candidates = []
    for package_path in files:
        relative_name = str(package_path).replace("\\", "/")
        basename = Path(relative_name).name
        if relative_name.startswith("mujoco/") and basename.startswith(
            ("libmujoco", "mujoco.dll", "_functions.")
        ):
            candidates.append((relative_name, package_path))
    engine_count = sum(
        Path(relative_name).name.startswith(("libmujoco", "mujoco.dll"))
        for relative_name, _ in candidates
    )
    binding_count = sum(
        Path(relative_name).name.startswith("_functions.")
        for relative_name, _ in candidates
    )
    if engine_count != 1 or binding_count != 1 or len(candidates) != 2:
        raise RuntimeError(
            "mujoco must expose exactly one native engine and one _functions binding."
        )
    fingerprints: list[Mapping[str, str]] = []
    for relative_name, package_path in sorted(candidates):
        record_hash = package_path.hash
        if record_hash is None or record_hash.mode != "sha256":
            raise RuntimeError(
                f"mujoco native file {relative_name} lacks a SHA-256 RECORD entry."
            )
        located_path = Path(package_path.locate())
        actual_hex = _sha256_file(located_path)
        actual_record_value = (
            base64.urlsafe_b64encode(bytes.fromhex(actual_hex))
            .rstrip(b"=")
            .decode("ascii")
        )
        if actual_record_value != record_hash.value:
            raise RuntimeError(
                f"mujoco native file {relative_name} does not match its RECORD hash."
            )
        fingerprints.append(
            MappingProxyType(
                {
                    "relative_name": relative_name,
                    "sha256": actual_hex,
                }
            )
        )
    return tuple(fingerprints)


@dataclass(frozen=True)
class RobosuiteRuntimeBindingA1:
    """Imported runtime leaves plus a primitive provenance fingerprint."""

    suite_module: Any
    load_part_controller_config: Callable[..., Mapping[str, Any]]
    refactor_composite_controller_config: Callable[..., Mapping[str, Any]]
    fingerprint: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not callable(getattr(self.suite_module, "make", None)):
            raise TypeError("suite_module must expose callable make().")
        if not callable(self.load_part_controller_config):
            raise TypeError("load_part_controller_config must be callable.")
        if not callable(self.refactor_composite_controller_config):
            raise TypeError("refactor_composite_controller_config must be callable.")
        fingerprint = MappingProxyType(dict(self.fingerprint))
        if not fingerprint:
            raise ValueError("runtime fingerprint must not be empty.")
        object.__setattr__(self, "fingerprint", fingerprint)


def load_pinned_runtime() -> RobosuiteRuntimeBindingA1:
    """Load the reviewed A1 runtime, rejecting drift before importing it.

    Keeping imports inside this function means importing :mod:`sim.robosuite`
    never initializes robosuite, MuJoCo, a renderer, or a device.
    """

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Robosuite A1 requires Python 3.10; "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )
    installed: dict[str, str] = {}
    for distribution, expected in ROBOSUITE_A1_PACKAGE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Robosuite A1 dependency {distribution}=={expected} is not installed."
            ) from exc
        if actual != expected:
            raise RuntimeError(
                f"Robosuite A1 requires {distribution}=={expected}; found {actual}."
            )
        installed[distribution] = actual

    lock_path = Path(__file__).resolve().parents[2] / "environments/robosuite/uv.lock"
    if not lock_path.is_file():
        raise RuntimeError("Robosuite A1 requires environments/robosuite/uv.lock.")
    distribution_fingerprints = {
        name: _distribution_fingerprint(name) for name in sorted(installed)
    }
    native_fingerprint = _verified_mujoco_native_fingerprint()

    numpy_module = importlib.import_module("numpy")
    mujoco_module = importlib.import_module("mujoco")
    suite_module = importlib.import_module("robosuite")
    controller_module = importlib.import_module("robosuite.controllers")
    composite_factory = importlib.import_module(
        "robosuite.controllers.composite.composite_controller_factory"
    )
    if str(getattr(suite_module, "__version__", "")) != installed["robosuite"]:
        raise RuntimeError("robosuite package metadata and runtime version disagree.")
    if str(getattr(numpy_module, "__version__", "")) != installed["numpy"]:
        raise RuntimeError("NumPy package metadata and runtime version disagree.")
    mujoco_runtime_version = str(mujoco_module.mj_versionString())
    if mujoco_runtime_version != installed["mujoco"]:
        raise RuntimeError(
            "MuJoCo package metadata and native runtime version disagree: "
            f"{installed['mujoco']} != {mujoco_runtime_version}."
        )
    fingerprint = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "libc": {
            "name": platform.libc_ver()[0],
            "version": platform.libc_ver()[1],
        },
        "uv_lock_sha256": _sha256_file(lock_path),
        "direct_distributions": distribution_fingerprints,
        "mujoco_native": native_fingerprint,
        **installed,
        "mujoco_runtime": mujoco_runtime_version,
    }
    return RobosuiteRuntimeBindingA1(
        suite_module=suite_module,
        load_part_controller_config=controller_module.load_part_controller_config,
        refactor_composite_controller_config=(
            composite_factory.refactor_composite_controller_config
        ),
        fingerprint=fingerprint,
    )


__all__ = [
    "ROBOSUITE_A1_PACKAGE_VERSIONS",
    "RobosuiteRuntimeBindingA1",
    "load_pinned_runtime",
]
