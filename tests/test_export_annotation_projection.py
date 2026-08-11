from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXPORTER = _REPOSITORY_ROOT / "scripts/export_annotation_projection.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_EXPORTER), *args],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_exported_annotation_projection_verifies_and_detects_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "annotation-release"
    exported = _run("--output", str(output))
    assert exported.returncode == 0, exported.stdout + exported.stderr

    receipt = json.loads(
        (output / "annotation_projection_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema"] == (
        "dexteleop.annotation-source-projection-receipt.v1"
    )
    assert receipt["source_file_count"] >= 50
    assert isinstance(receipt["source_projection_dirty"], bool)
    assert (output / "ANNOTATION_RELEASE.md").is_file()
    assert (output / "control_station/refinement_annotation/cli.py").is_file()

    verified = _run("--verify", str(output))
    assert verified.returncode == 0, verified.stdout + verified.stderr

    existing = _run("--output", str(output))
    assert existing.returncode != 0
    assert "already exists" in existing.stderr

    projected_cli = output / "control_station/refinement_annotation/cli.py"
    projected_cli.write_text(
        projected_cli.read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    drifted = _run("--verify", str(output))
    assert drifted.returncode != 0
    assert "differs from its receipt" in drifted.stderr
