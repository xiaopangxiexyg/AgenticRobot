#!/usr/bin/env python3
"""Gripper-only debug entrypoint; intentionally bypasses tactile package imports."""

from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent / "portable_dmtac_gripper"
module_text = str(MODULE_DIR)
if module_text not in sys.path:
    sys.path.insert(0, module_text)

from gripper_debug import main


if __name__ == "__main__":
    main()
