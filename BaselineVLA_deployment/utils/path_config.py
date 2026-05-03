from __future__ import annotations

import os
import re
import shlex
import sys
import ctypes
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_FILENAME = "local_paths.env"
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOADED_RUNTIME_LIBS: set[Path] = set()


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    try:
        parts = shlex.split(value, comments=False, posix=True)
    except ValueError:
        parts = [value.strip("\"'")]
    if not parts:
        parsed = ""
    elif len(parts) == 1:
        parsed = parts[0]
    else:
        parsed = " ".join(parts)
    return os.path.expandvars(os.path.expanduser(parsed))


def load_local_env(env_path: str | Path | None = None, *, override: bool = False) -> Path:
    """Load repo-local machine paths from local_paths.env without requiring python-dotenv."""

    path = Path(env_path).expanduser() if env_path is not None else PROJECT_ROOT / LOCAL_ENV_FILENAME
    os.environ.setdefault("MANIPREP_PROJECT_ROOT", str(PROJECT_ROOT))
    if not path.exists():
        return path

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not _KEY_PATTERN.match(key):
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _parse_env_value(raw_value)
    return path


def _current_conda_prefix() -> Path | None:
    configured = os.environ.get("CONDA_PREFIX")
    if configured:
        return Path(configured).expanduser()
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "bin":
        return executable.parent.parent
    return None


def ensure_conda_runtime_libs() -> None:
    """Prefer C++ runtime libraries bundled with the active Conda environment."""

    conda_prefix = _current_conda_prefix()
    if conda_prefix is None:
        return
    for library_name in ("libstdc++.so.6", "libgcc_s.so.1"):
        library_path = conda_prefix / "lib" / library_name
        if not library_path.exists() or library_path in _LOADED_RUNTIME_LIBS:
            continue
        ctypes.CDLL(str(library_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        _LOADED_RUNTIME_LIBS.add(library_path)


def resolve_repo_path(path: str | Path, *, base: str | Path | None = None) -> Path:
    text = os.path.expandvars(str(path)).strip()
    resolved = Path(text).expanduser()
    if resolved.is_absolute():
        return resolved.resolve()
    root = Path(base).expanduser() if base is not None else PROJECT_ROOT
    return (root / resolved).resolve()


def optional_repo_path(path: Any, *, base: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text or text.lower() == "null":
        return None
    return resolve_repo_path(text, base=base)
