from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DMROBOTICS_PACKAGE_DIR = (
    REPO_ROOT / "assets" / "DM_Tac_SDK_Publish_1.2.10" / "dmrobotics"
)


def _run_check(code: str) -> subprocess.CompletedProcess[str]:
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


def test_dmrobotics_adapter_removes_vendor_package_import_root(tmp_path: Path) -> None:
    package = tmp_path / "dmrobotics"
    source = package / "src"
    source.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
    )
    (package / "utils.py").write_text("VALUE = 1\n")
    (source / "__init__.py").write_text("")
    (source / "CameraDeviceManager.py").write_text(
        "class CameraDeviceManager:\n"
        "    def find_devices(self):\n"
        "        return []\n"
    )

    result = _run_check(
        f"""
        import importlib.util
        import sys

        from hardware.DMTAC_gripper.portable_dm_tac import core

        sys.path.insert(0, {str(tmp_path)!r})
        path_before = list(sys.path)
        assert core._import_dmrobotics().__name__ == "dmrobotics"
        assert sys.path == path_before
        assert core.discover_devices() == []
        assert sys.path == path_before
        assert importlib.util.find_spec("utils").origin == {str(REPO_ROOT / "utils" / "__init__.py")!r}
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_real_teleop_module_bootstrap_does_not_load_shadowed_utils() -> None:
    result = _run_check(
        f"""
        import sys

        sys.path.insert(0, {str(DMROBOTICS_PACKAGE_DIR)!r})
        import teleop_runtime.debug.run_vive_rm65_real_teleop

        loaded_utils = sys.modules.get("utils")
        assert loaded_utils is None or not str(getattr(loaded_utils, "__file__", "")).startswith(
            {str(DMROBOTICS_PACKAGE_DIR)!r}
        )
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
