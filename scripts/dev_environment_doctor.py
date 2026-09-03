#!/usr/bin/env python3
"""Validate the shared DexTeleop Conda development environment.

This command is intentionally read-only.  It checks dependency and source
resolution without importing runtime packages or initializing any device.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import site
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = REPO_ROOT / "pyproject.toml"
ASSET_STORE_TOOL = REPO_ROOT / "scripts" / "dev_asset_store.py"
LOCK_INDEX_URL = "https://pypi.org/simple"


@dataclass(frozen=True)
class Finding:
    level: str
    key: str
    detail: str


@dataclass(frozen=True)
class ProjectEnvironment:
    name: str
    python: str
    package_roots: tuple[str, ...]
    optional_dependencies: tuple[str, ...]
    conda_packages: tuple[str, ...]
    pip_no_deps_packages: tuple[str, ...]
    vendor_sdk: str


def _section(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^\[{re.escape(name)}\]\s*$\n(.*?)(?=^\[|\Z)",
        text,
    )
    if match is None:
        raise ValueError(f"missing [{name}] section")
    return match.group(1)


def _literal_assignment(section: str, key: str):
    match = re.search(
        rf"(?ms)^{re.escape(key)}\s*=\s*(\[.*?^\]|[^\n]+)",
        section,
    )
    if match is None:
        raise ValueError(f"missing {key!r} assignment")
    try:
        return ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid literal for {key!r}") from exc


def load_project_environment(path: Path = PROJECT_FILE) -> ProjectEnvironment:
    text = path.read_text(encoding="utf-8")
    packages_section = _section(text, "tool.setuptools.packages.find")
    environment_section = _section(text, "tool.dexteleop.environment")
    optional_section = _section(text, "project.optional-dependencies")

    include_patterns = _literal_assignment(packages_section, "include")
    package_roots = tuple(pattern.removesuffix("*") for pattern in include_patterns)

    optional_dependencies: list[str] = []
    for match in re.finditer(
        r"(?ms)^[A-Za-z0-9_]+\s*=\s*(\[.*?^\])",
        optional_section,
    ):
        try:
            values = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid optional dependency list") from exc
        optional_dependencies.extend(values)

    return ProjectEnvironment(
        name=_literal_assignment(environment_section, "name"),
        python=_literal_assignment(environment_section, "python"),
        package_roots=package_roots,
        optional_dependencies=tuple(optional_dependencies),
        conda_packages=tuple(
            _literal_assignment(environment_section, "conda_packages")
        ),
        pip_no_deps_packages=tuple(
            _literal_assignment(environment_section, "pip_no_deps_packages")
        ),
        vendor_sdk=_literal_assignment(environment_section, "vendor_sdk"),
    )


def _dependency_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise ValueError(f"cannot parse dependency name from {requirement!r}")
    return match.group(1)


def _expected_exact_version(requirement: str) -> str | None:
    match = re.match(r"\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*==\s*([^\s;]+)", requirement)
    return match.group(1) if match is not None else None


def _requirement_applies(requirement: str) -> bool:
    if ";" not in requirement:
        return True
    marker = requirement.split(";", 1)[1]
    if "sys_platform == 'linux'" in marker and sys.platform != "linux":
        return False
    return not (
        "platform_machine == 'x86_64'" in marker and platform.machine() != "x86_64"
    )


def _conda_records(prefix: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for record_path in (prefix / "conda-meta").glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = record.get("name")
        version = record.get("version")
        if isinstance(name, str) and isinstance(version, str):
            records[name] = version
    return records


def _user_site_roots() -> tuple[Path, ...]:
    configured = site.getusersitepackages()
    values = (configured,) if isinstance(configured, str) else tuple(configured)
    return tuple(Path(value).expanduser().resolve() for value in values)


def _distribution_version_outside_user_site(name: str) -> str | None:
    for distribution in metadata.distributions(name=name):
        try:
            location = Path(distribution.locate_file("")).resolve()
        except OSError:
            continue
        if not any(_is_within(str(location), root) for root in _user_site_roots()):
            return distribution.version
    return None


def _module_origins(modules: Sequence[str], cwd: Path) -> dict[str, list[str]]:
    script = r"""
import importlib.util
import json
import sys

result = {}
for name in json.loads(sys.argv[1]):
    spec = importlib.util.find_spec(name)
    if spec is None:
        result[name] = []
        continue
    paths = []
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        paths.append(spec.origin)
    if spec.submodule_search_locations:
        paths.extend(str(path) for path in spec.submodule_search_locations)
    result[name] = paths
print(json.dumps(result, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(list(modules))],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "module resolution failed")
    return json.loads(completed.stdout)


def _is_within(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, object]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _development_asset_status() -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ASSET_STORE_TOOL),
            "--repo-root",
            str(REPO_ROOT),
            "status",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(f"asset-store status was not valid JSON: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004 - subprocess protocol failure
            "asset-store status was not a JSON object"
        )
    return payload


def _append_package_findings(
    findings: list[Finding], config: ProjectEnvironment
) -> None:
    requirements = [
        requirement
        for requirement in config.optional_dependencies
        if _requirement_applies(requirement)
    ]
    requirements.extend(config.pip_no_deps_packages)
    conda_records = _conda_records(Path(sys.prefix))

    missing: list[str] = []
    mismatched: list[str] = []
    for requirement in requirements:
        name = _dependency_name(requirement)
        expected = _expected_exact_version(requirement)
        installed = _distribution_version_outside_user_site(name)
        if installed is None:
            installed = conda_records.get(name)
        if installed is None:
            missing.append(name)
        elif expected is not None and installed != expected:
            mismatched.append(f"{name}={installed} (expected {expected})")

    for requirement in config.conda_packages:
        name, expected = requirement.split("=", 1)
        installed = conda_records.get(name)
        if installed is None:
            missing.append(name)
        elif installed != expected:
            mismatched.append(f"{name}={installed} (expected {expected})")

    if missing:
        findings.append(
            Finding("FAIL", "deps.missing", ", ".join(sorted(set(missing))))
        )
    else:
        findings.append(
            Finding(
                "PASS", "deps.present", "all declared dependency groups are installed"
            )
        )
    if mismatched:
        findings.append(Finding("FAIL", "deps.versions", "; ".join(mismatched)))
    else:
        findings.append(Finding("PASS", "deps.versions", "all exact pins match"))


def collect_findings(
    args: argparse.Namespace,
) -> tuple[list[Finding], dict[str, object]]:
    findings: list[Finding] = []
    try:
        config = load_project_environment()
    except (OSError, ValueError) as exc:
        return [Finding("FAIL", "project.config", str(exc))], {}

    current_root = Path.cwd().resolve()
    if current_root == REPO_ROOT:
        findings.append(Finding("PASS", "repo.cwd", str(REPO_ROOT)))
    else:
        findings.append(
            Finding(
                "FAIL",
                "repo.cwd",
                f"run from the worktree root {REPO_ROOT}; current directory is {current_root}",
            )
        )

    expected_python = tuple(int(part) for part in config.python.split("."))
    if sys.version_info[:2] == expected_python:
        findings.append(Finding("PASS", "python.version", platform.python_version()))
    else:
        findings.append(
            Finding(
                "FAIL",
                "python.version",
                f"expected {config.python}.x, got {platform.python_version()}",
            )
        )

    conda_meta = Path(sys.prefix) / "conda-meta"
    if conda_meta.is_dir():
        active_name = Path(sys.prefix).name
        level = "PASS" if active_name == config.name else "WARN"
        findings.append(
            Finding(
                level,
                "conda.environment",
                f"{active_name} at {sys.prefix}; canonical name is {config.name}",
            )
        )
    else:
        findings.append(
            Finding(
                "FAIL", "conda.environment", f"{sys.prefix} is not a Conda environment"
            )
        )

    active_user_site = [
        entry
        for entry in sys.path
        if entry and any(_is_within(entry, root) for root in _user_site_roots())
    ]
    if active_user_site:
        findings.append(
            Finding(
                "WARN",
                "python.user_site",
                "user-site paths are active but ignored by dependency checks: "
                + ", ".join(active_user_site),
            )
        )
    else:
        findings.append(
            Finding("PASS", "python.user_site", "user-site paths are disabled")
        )

    missing_roots = [
        name
        for name in config.package_roots
        if not (REPO_ROOT / name / "__init__.py").is_file()
    ]
    if missing_roots:
        findings.append(
            Finding(
                "FAIL",
                "source.configured_roots",
                "configured package roots do not exist: " + ", ".join(missing_roots),
            )
        )
    else:
        findings.append(
            Finding("PASS", "source.configured_roots", "all configured roots exist")
        )

    try:
        root_origins = _module_origins(config.package_roots, REPO_ROOT)
    except RuntimeError as exc:
        findings.append(Finding("FAIL", "source.root_resolution", str(exc)))
    else:
        bad_root_origins = {
            name: origins
            for name, origins in root_origins.items()
            if not origins or not all(_is_within(path, REPO_ROOT) for path in origins)
        }
        if bad_root_origins:
            findings.append(
                Finding(
                    "FAIL",
                    "source.root_resolution",
                    "packages not resolved from this worktree: "
                    + json.dumps(bad_root_origins, sort_keys=True),
                )
            )
        else:
            findings.append(
                Finding(
                    "PASS",
                    "source.root_resolution",
                    "all source packages resolve from the current worktree root",
                )
            )

    try:
        with tempfile.TemporaryDirectory(prefix="dexteleop-doctor-") as temp_dir:
            off_root_origins = _module_origins(config.package_roots, Path(temp_dir))
    except RuntimeError as exc:
        findings.append(Finding("FAIL", "source.off_root_resolution", str(exc)))
    else:
        resolved_off_root = {
            name: origins for name, origins in off_root_origins.items() if origins
        }
        if resolved_off_root:
            level = "FAIL" if args.reject_repo_editable else "WARN"
            findings.append(
                Finding(
                    level,
                    "source.off_root_resolution",
                    "an editable mapping is active outside the worktree root; shared-worktree "
                    "commands must run at their own root: "
                    + json.dumps(resolved_off_root, sort_keys=True),
                )
            )
        else:
            findings.append(
                Finding(
                    "PASS",
                    "source.off_root_resolution",
                    "no repository package is pinned by an editable install",
                )
            )

    if args.skip_dependency_check:
        findings.append(Finding("WARN", "deps.check", "skipped by request"))
    else:
        _append_package_findings(findings, config)

    if not args.skip_lock_check:
        try:
            lock_environment = os.environ.copy()
            # uv considers PIP_INDEX_URL when deciding whether a lock is fresh.
            # The committed lock records PyPI, so a developer-local mirror must
            # not turn this read-only health check into a false stale-lock error.
            lock_environment["UV_DEFAULT_INDEX"] = LOCK_INDEX_URL
            lock_environment["PIP_INDEX_URL"] = LOCK_INDEX_URL
            completed = subprocess.run(
                ["uv", "lock", "--check"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=lock_environment,
            )
        except FileNotFoundError:
            findings.append(Finding("FAIL", "lock.check", "uv is not on PATH"))
        else:
            if completed.returncode == 0:
                findings.append(
                    Finding("PASS", "lock.check", "uv.lock matches pyproject.toml")
                )
            else:
                detail = completed.stderr.strip() or completed.stdout.strip()
                cache_is_read_only = "Read-only file system" in detail and (
                    "/.cache/uv/" in detail or "UV_CACHE_DIR" in detail
                )
                if cache_is_read_only:
                    findings.append(
                        Finding(
                            "WARN",
                            "lock.check",
                            "uv could not use its read-only cache; lock freshness was not "
                            f"verified in this process: {detail}",
                        )
                    )
                else:
                    findings.append(Finding("FAIL", "lock.check", detail))
    else:
        findings.append(Finding("WARN", "lock.check", "skipped by request"))

    asset_status: dict[str, object] = {}
    try:
        asset_status = _development_asset_status()
    except RuntimeError as exc:
        level = "FAIL" if args.require_development_assets else "WARN"
        findings.append(Finding(level, "assets.development_store", str(exc)))
    else:
        if asset_status.get("available") is True:
            findings.append(
                Finding(
                    "PASS",
                    "assets.development_store",
                    f"required development bundles are available at {asset_status['root']} "
                    f"via {asset_status['source']}",
                )
            )
        else:
            level = "FAIL" if args.require_development_assets else "WARN"
            error = asset_status.get("error")
            missing_bundles = [
                item.get("bundle_id")
                for item in asset_status.get("bundles", [])
                if isinstance(item, dict)
                and item.get("required_by_default") is True
                and item.get("available") is not True
            ]
            detail = (
                str(error)
                if error
                else (
                    "required development bundles are unavailable: "
                    + ", ".join(str(item) for item in missing_bundles)
                )
            )
            findings.append(Finding(level, "assets.development_store", detail))

    configured_sdk = (
        args.vendor_sdk_root
        or os.environ.get("DEXTELEOP_DMTAC_SDK_ROOT")
        or str(REPO_ROOT / config.vendor_sdk)
    )
    sdk_root = Path(configured_sdk).expanduser().resolve()
    if (sdk_root / "setup.py").is_file():
        findings.append(
            Finding("PASS", "vendor_sdk", f"DM-Tac SDK found at {sdk_root}")
        )
    else:
        level = "FAIL" if args.require_vendor_sdk else "WARN"
        findings.append(
            Finding(
                level,
                "vendor_sdk",
                f"DM-Tac SDK not found at {sdk_root}; set DEXTELEOP_DMTAC_SDK_ROOT "
                "for hardware development",
            )
        )

    fingerprints = {
        path.name: _file_sha256(path)
        for path in (PROJECT_FILE, REPO_ROOT / "environment.yml", REPO_ROOT / "uv.lock")
    }
    metadata_result: dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "python": platform.python_version(),
        "prefix": sys.prefix,
        "fingerprints": fingerprints,
        "git": _git_metadata(),
        "development_assets": asset_status,
    }
    return findings, metadata_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the shared DexTeleop Conda development environment without device I/O."
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable output"
    )
    parser.add_argument(
        "--skip-lock-check",
        action="store_true",
        help="skip the read-only 'uv lock --check' command",
    )
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="skip installed dependency checks (intended for doctor unit tests only)",
    )
    parser.add_argument(
        "--reject-repo-editable",
        action="store_true",
        help="fail when repository packages resolve outside the current worktree root",
    )
    parser.add_argument(
        "--require-vendor-sdk",
        action="store_true",
        help="require the local DM-Tac SDK instead of reporting it as optional",
    )
    parser.add_argument(
        "--vendor-sdk-root",
        help="override the DM-Tac SDK checkout path",
    )
    parser.add_argument(
        "--require-development-assets",
        action="store_true",
        help="fail unless the configured development asset store has all default bundles",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    findings, metadata_result = collect_findings(args)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": not any(item.level == "FAIL" for item in findings),
                    "findings": [asdict(item) for item in findings],
                    "metadata": metadata_result,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for item in findings:
            print(f"[{item.level}] {item.key}: {item.detail}")
        counts = {
            level: sum(item.level == level for item in findings)
            for level in ("PASS", "WARN", "FAIL")
        }
        print(
            "Summary: "
            + ", ".join(
                f"{level}={counts[level]}" for level in ("PASS", "WARN", "FAIL")
            )
        )
    return 1 if any(item.level == "FAIL" for item in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
