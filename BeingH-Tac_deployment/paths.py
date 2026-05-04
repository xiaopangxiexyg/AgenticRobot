from __future__ import annotations

import os
import sys
from pathlib import Path


DEPLOYMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOYMENT_DIR.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(os.path.expandvars(value)).expanduser().resolve()
    return default.resolve()


DEXTELEOP_ROOT = _path_from_env("DEXTELEOP_ROOT", REPO_ROOT)
BEINGH_TAC_ROOT = _path_from_env(
    "BEINGH_TAC_ROOT",
    Path("/home/ai/Desktop/Being-H-series/Being-H-Tac"),
)
UV_CONVERSION_ROOT = _path_from_env("UV_CONVERSION_ROOT", DEXTELEOP_ROOT / "UV_conversion")


def ensure_import_paths() -> None:
    for path in (DEXTELEOP_ROOT, BEINGH_TAC_ROOT, UV_CONVERSION_ROOT):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
