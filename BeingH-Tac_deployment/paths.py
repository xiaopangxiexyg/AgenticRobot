from __future__ import annotations

import sys
from pathlib import Path


DEXTELEOP_ROOT = Path("/home/ai/Desktop/DexTeleop")
BEINGH_TAC_ROOT = Path("/home/ai/Desktop/Being-H-series/Being-H-Tac")
UV_CONVERSION_ROOT = DEXTELEOP_ROOT / "UV_conversion"


def ensure_import_paths() -> None:
    for path in (DEXTELEOP_ROOT, BEINGH_TAC_ROOT, UV_CONVERSION_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

