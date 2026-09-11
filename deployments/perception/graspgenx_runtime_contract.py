"""Verifiable runtime contract for in-process GraspGenX inference.

This module defines the importable, environment-level contract that a
Gateway Python process must satisfy before the in-process GraspGenX
backend may load the official model: reviewed Python version, Torch
range, CUDA availability, reviewed source checkout revision, reviewed
asset manifest identity, and checkpoint layout. Verification never
imports the GraspGenX package, loads model weights, or touches devices;
it returns a typed fail-closed receipt for diagnostics and evidence.

The receipt proves environment compatibility only. It is not a model
load, inference, camera, or real-robot acceptance, and it does not
change the default isolated loopback service mode.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID = "graspgenx-inprocess-runtime.v1"
GRASPGENX_RUNTIME_RECEIPT_SCHEMA_V1 = (
    "dexteleop.perception.graspgenx_runtime_receipt.v1"
)
GRASPGENX_REVIEWED_PYTHON_VERSIONS: tuple[tuple[int, int], ...] = ((3, 10), (3, 11))
GRASPGENX_TORCH_MIN_VERSION = (2, 1)
GRASPGENX_TORCH_MAX_VERSION_EXCLUSIVE = (2, 7)
GRASPGENX_DEVICE_PATTERN = re.compile(r"cuda(?::[0-9]+)?")
GRASPGENX_REQUIRED_SOURCE_MODULES = (
    "graspgenx",
    "graspgenx.grasp_server",
    "graspgenx.utils.checkpoint_io",
    "graspgenx.samplers",
    "graspgenx.serving.types",
)
GRASPGENX_CHECKPOINT_SUBDIRS = ("gen", "dis")


def default_graspgenx_source_root() -> Path:
    """Return the repository's pinned GraspGenX source submodule root."""

    return Path(__file__).resolve().parents[2] / "model" / "perception" / "graspgenx"


@dataclass(frozen=True)
class GraspGenXRuntimeCheck:
    """One named contract observation with its expected value."""

    name: str
    status: bool
    expected: str
    observed: str
    detail: str = ""

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": "passed" if self.status else "failed",
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GraspGenXRuntimeReceipt:
    """Typed fail-closed verification result for one candidate runtime.

    ``passed`` is true only when every recorded check passed. A failed
    receipt must close the in-process path; callers may not retry with a
    narrower check subset or reinterpret a failed check as advisory.
    """

    contract_id: str
    device: str
    source_root: str
    model_spec_hash: str
    checks: tuple[GraspGenXRuntimeCheck, ...] = field(default=())

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.status for check in self.checks)

    @property
    def failed_checks(self) -> tuple[GraspGenXRuntimeCheck, ...]:
        return tuple(check for check in self.checks if not check.status)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": GRASPGENX_RUNTIME_RECEIPT_SCHEMA_V1,
            "contract_id": self.contract_id,
            "status": "passed" if self.passed else "failed",
            "device": self.device,
            "source_root": self.source_root,
            "model_spec_hash": self.model_spec_hash,
            "checks": [check.to_mapping() for check in self.checks],
        }

    def failure_summary(self) -> str:
        return "; ".join(
            f"{check.name}: expected {check.expected}, observed {check.observed}"
            + (f" ({check.detail})" if check.detail else "")
            for check in self.failed_checks
        )


def _git_head_revision(source_root: Path) -> str:
    """Return the HEAD commit of ``source_root`` or raise OSError."""

    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _torch_version_tuple(version: str) -> tuple[int, int] | None:
    match = re.match(r"^([0-9]+)\.([0-9]+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def verify_graspgenx_inprocess_runtime(
    *,
    model_spec: Any,
    source_root: str | Path | None = None,
    device: str = "cuda",
    import_module: Callable[[str], Any] | None = None,
    git_head_revision: Callable[[Path], str] | None = None,
    version_info: tuple[int, int, int] | None = None,
) -> GraspGenXRuntimeReceipt:
    """Verify one Gateway environment against the in-process contract.

    All checks are recorded even when earlier ones fail, so one receipt
    gives complete dependency-conflict diagnostics. Torch and GraspGenX
    are never imported here; ``import_module`` exists for deterministic
    offline tests and defaults to :func:`importlib.import_module`. The
    model bundle hash identity comes from ``model_spec`` whose loader is
    the single owner of manifest and checkpoint byte verification.
    """

    normalized_device = str(device).strip()
    if not GRASPGENX_DEVICE_PATTERN.fullmatch(normalized_device):
        raise ValueError(
            "GraspGenX in-process runtime requires a CUDA device such as 'cuda'."
        )
    if not isinstance(
        getattr(model_spec, "semantic_content_hash", None), str
    ):
        raise TypeError("model_spec must expose semantic_content_hash.")
    for required_root in ("bundle_root", "checkpoint_root"):
        if not isinstance(getattr(model_spec, required_root, None), Path):
            raise TypeError(f"model_spec must expose {required_root}.")

    importer = import_module or importlib.import_module
    git_head = git_head_revision or _git_head_revision
    resolved_source = (
        default_graspgenx_source_root() if source_root is None else Path(source_root)
    )
    checks: list[GraspGenXRuntimeCheck] = []

    major, minor, patch = version_info or sys.version_info[:3]
    checks.append(
        GraspGenXRuntimeCheck(
            name="python_version",
            status=(major, minor) in GRASPGENX_REVIEWED_PYTHON_VERSIONS,
            expected=" or ".join(
                f"{item[0]}.{item[1]}"
                for item in GRASPGENX_REVIEWED_PYTHON_VERSIONS
            ),
            observed=f"{major}.{minor}.{patch}",
            detail="reviewed Gateway Python versions for the official runtime",
        )
    )

    torch_module: Any = None
    torch_error = ""
    try:
        torch_module = importer("torch")
    except Exception as exc:  # noqa: BLE001 - recorded as a failed check
        torch_error = f"{type(exc).__name__}: {exc}"
    torch_version = str(getattr(torch_module, "__version__", "") or "")
    checks.append(
        GraspGenXRuntimeCheck(
            name="torch_import",
            status=torch_module is not None and bool(torch_version),
            expected="torch importable in the Gateway Python",
            observed=torch_version or "not imported",
            detail=torch_error,
        )
    )
    parsed_version = _torch_version_tuple(torch_version) if torch_version else None
    checks.append(
        GraspGenXRuntimeCheck(
            name="torch_version",
            status=parsed_version is not None
            and GRASPGENX_TORCH_MIN_VERSION
            <= parsed_version
            < GRASPGENX_TORCH_MAX_VERSION_EXCLUSIVE,
            expected=f"torch >= {GRASPGENX_TORCH_MIN_VERSION[0]}.{GRASPGENX_TORCH_MIN_VERSION[1]},"
            f" < {GRASPGENX_TORCH_MAX_VERSION_EXCLUSIVE[0]}.{GRASPGENX_TORCH_MAX_VERSION_EXCLUSIVE[1]}",
            observed=torch_version or "unknown",
            detail="official GraspGenX checkpoint runtime constraint",
        )
    )

    cuda_available = False
    device_count = 0
    cuda_detail = "torch was not imported"
    if torch_module is not None:
        try:
            cuda_available = bool(torch_module.cuda.is_available())
            device_count = int(torch_module.cuda.device_count())
            if not cuda_available:
                cuda_detail = "torch.cuda.is_available() returned False"
        except Exception as exc:  # noqa: BLE001 - recorded as a failed check
            cuda_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        GraspGenXRuntimeCheck(
            name="cuda_available",
            status=cuda_available,
            expected="a usable CUDA runtime visible to torch",
            observed=str(cuda_available),
            detail=cuda_detail if not cuda_available else "",
        )
    )
    requested_index: int | None = None
    if normalized_device != "cuda":
        requested_index = int(normalized_device.split(":", 1)[1])
    index_ok = cuda_available and (
        device_count >= 1 if requested_index is None else requested_index < device_count
    )
    checks.append(
        GraspGenXRuntimeCheck(
            name="cuda_device_index",
            status=index_ok,
            expected=f"{normalized_device} resolves to a visible CUDA device",
            observed=f"device_count={device_count}",
            detail="" if index_ok else "requested CUDA device is not visible",
        )
    )

    source_exists = resolved_source.is_dir()
    checks.append(
        GraspGenXRuntimeCheck(
            name="source_root",
            status=source_exists,
            expected="existing GraspGenX source checkout directory",
            observed=str(resolved_source),
            detail="" if source_exists else "source checkout is missing",
        )
    )

    observed_revision = ""
    revision_detail = ""
    if source_exists:
        try:
            observed_revision = git_head(resolved_source)
        except (OSError, subprocess.CalledProcessError) as exc:
            revision_detail = f"{type(exc).__name__}: {exc}"
    expected_revision = str(getattr(model_spec, "source_revision", "") or "")
    checks.append(
        GraspGenXRuntimeCheck(
            name="source_revision",
            status=source_exists
            and bool(observed_revision)
            and observed_revision == expected_revision,
            expected=expected_revision or "model_spec.source_revision",
            observed=observed_revision or "unknown",
            detail=revision_detail
            or (
                ""
                if observed_revision == expected_revision and observed_revision
                else "source checkout is not a Git repository at the reviewed revision"
            ),
        )
    )

    package_root = resolved_source / "graspgenx"
    package_ok = (package_root / "__init__.py").is_file()
    checks.append(
        GraspGenXRuntimeCheck(
            name="source_package",
            status=package_ok,
            expected="graspgenx package layout inside source_root",
            observed=str(package_root),
            detail="" if package_ok else "graspgenx package files are missing",
        )
    )

    bundle_root = Path(model_spec.bundle_root)
    bundle_ok = bundle_root.is_dir()
    checks.append(
        GraspGenXRuntimeCheck(
            name="asset_bundle",
            status=bundle_ok,
            expected="existing reviewed GraspGenX asset bundle",
            observed=str(bundle_root),
            detail="" if bundle_ok else "asset bundle directory is missing",
        )
    )

    checkpoint_root = Path(model_spec.checkpoint_root)
    missing_layout: list[str] = []
    if checkpoint_root.is_dir():
        for subdirectory in GRASPGENX_CHECKPOINT_SUBDIRS:
            component = checkpoint_root / subdirectory
            if not component.is_dir():
                missing_layout.append(str(component))
    else:
        missing_layout.append(str(checkpoint_root))
    checks.append(
        GraspGenXRuntimeCheck(
            name="checkpoint_layout",
            status=not missing_layout,
            expected="gen/ and dis/ checkpoint directories under checkpoint_root",
            observed=str(checkpoint_root),
            detail="missing: " + ", ".join(missing_layout) if missing_layout else "",
        )
    )

    return GraspGenXRuntimeReceipt(
        contract_id=GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
        device=normalized_device,
        source_root=str(resolved_source),
        model_spec_hash=str(model_spec.semantic_content_hash),
        checks=tuple(checks),
    )


__all__ = [
    "GRASPGENX_CHECKPOINT_SUBDIRS",
    "GRASPGENX_DEVICE_PATTERN",
    "GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID",
    "GRASPGENX_REQUIRED_SOURCE_MODULES",
    "GRASPGENX_REVIEWED_PYTHON_VERSIONS",
    "GRASPGENX_RUNTIME_RECEIPT_SCHEMA_V1",
    "GRASPGENX_TORCH_MAX_VERSION_EXCLUSIVE",
    "GRASPGENX_TORCH_MIN_VERSION",
    "GraspGenXRuntimeCheck",
    "GraspGenXRuntimeReceipt",
    "default_graspgenx_source_root",
    "verify_graspgenx_inprocess_runtime",
]
