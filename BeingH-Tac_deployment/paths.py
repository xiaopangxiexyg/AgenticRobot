from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List


DEPLOYMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOYMENT_DIR.parent


def _path_from_env(name: str, defaults: List[Path]) -> Path:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(os.path.expandvars(value)).expanduser().resolve()
    else:
        for default in defaults:
            if default.exists():
                return default.resolve()
    return defaults[-1].resolve()


DEXTELEOP_ROOT = _path_from_env("DEXTELEOP_ROOT", [REPO_ROOT])
BEINGH_TAC_ROOT = _path_from_env(
    "BEINGH_TAC_ROOT",
    [
        Path("/home/user/cpl/Being-H-Tac-visuo-tactile"),
        Path("/home/user/cpl/Being-H-Tac"),
        Path("/home/ai/Desktop/Being-H-series/Being-H-Tac"),
    ],
)
UV_CONVERSION_ROOT = _path_from_env("UV_CONVERSION_ROOT", [DEXTELEOP_ROOT / "UV_conversion"])


def ensure_import_paths() -> None:
    for path in (DEXTELEOP_ROOT, BEINGH_TAC_ROOT, UV_CONVERSION_ROOT):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
