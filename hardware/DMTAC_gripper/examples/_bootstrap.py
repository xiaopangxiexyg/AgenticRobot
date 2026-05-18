from __future__ import annotations

import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = EXAMPLES_DIR.parent

text = str(PACKAGE_ROOT)
if text not in sys.path:
    sys.path.insert(0, text)
