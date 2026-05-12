from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_paths() -> None:
    """Make this source tree usable without pip installing it."""
    package_root = Path(__file__).resolve().parents[1]

    text = str(package_root)
    if text not in sys.path:
        sys.path.insert(0, text)
