from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DMROBOTICS_PACKAGE_DIR = REPO_ROOT / "assets" / "DM_Tac_SDK_Publish_1.2.10" / "dmrobotics"


def _run_import_check(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_writer_v21_does_not_depend_on_or_reorder_shadowed_utils() -> None:
    result = _run_import_check(
        f"""
        import sys

        sys.path.insert(0, {str(DMROBOTICS_PACKAGE_DIR)!r})
        path_before = list(sys.path)
        from teleop_runtime.collection import writer_v21

        assert sys.path == path_before
        assert "utils" not in sys.modules
        assert writer_v21.DepthRangeMaskConfig.__module__ == "dataset_contracts.wrist_depth"
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_writer_v21_leaves_already_loaded_top_level_utils_untouched() -> None:
    result = _run_import_check(
        f"""
        import sys
        import types

        poisoned = types.ModuleType("utils")
        poisoned.__file__ = {str(DMROBOTICS_PACKAGE_DIR / "utils.py")!r}
        sys.modules["utils"] = poisoned
        sys.path.insert(0, {str(DMROBOTICS_PACKAGE_DIR)!r})

        from teleop_runtime.collection import writer_v21

        assert sys.modules["utils"] is poisoned
        assert writer_v21.DepthRangeMaskConfig.__module__ == "dataset_contracts.wrist_depth"
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
