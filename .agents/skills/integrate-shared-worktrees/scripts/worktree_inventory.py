"""Compatibility CLI for callers using the installed Skill's script path."""

import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPTS))
    runpy.run_path(str(SCRIPTS / "worktree_inventory.py"), run_name="__main__")
