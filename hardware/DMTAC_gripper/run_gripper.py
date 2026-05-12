#!/usr/bin/env python3
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent

text = str(THIS_DIR)
if text not in sys.path:
    sys.path.insert(0, text)

from portable_dmtac_gripper.cli import main


if __name__ == "__main__":
    main()
