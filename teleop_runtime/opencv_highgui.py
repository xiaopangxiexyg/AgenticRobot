from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCVHighGUIProbeResult:
    ok: bool
    reason: str = ""


def _compact_output(text: str, *, max_chars: int = 500) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    output = " | ".join(lines[-4:])
    if len(output) <= max_chars:
        return output
    return output[-max_chars:]


def probe_opencv_highgui(
    *,
    timeout_sec: float = 3.0,
    window_name: str = "teleop_cv2_highgui_probe",
) -> OpenCVHighGUIProbeResult:
    """Check OpenCV HighGUI in a child process.

    Some Qt/XCB failures abort the interpreter instead of raising a Python
    exception. Running the real namedWindow/imshow path in a child lets callers
    disable optional previews without taking down the collector.
    """
    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return OpenCVHighGUIProbeResult(
            ok=False,
            reason="no DISPLAY/WAYLAND_DISPLAY is set",
        )

    script = (
        "import cv2\n"
        "import numpy as np\n"
        f"window_name = {window_name!r}\n"
        "image = np.zeros((8, 8, 3), dtype=np.uint8)\n"
        "cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)\n"
        "cv2.imshow(window_name, image)\n"
        "cv2.waitKey(1)\n"
        "cv2.destroyWindow(window_name)\n"
        "cv2.waitKey(1)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        output = _compact_output((exc.stderr or "") + "\n" + (exc.stdout or ""))
        suffix = "" if not output else f": {output}"
        return OpenCVHighGUIProbeResult(
            ok=False,
            reason=f"OpenCV HighGUI probe timed out after {timeout_sec:.1f}s{suffix}",
        )
    except Exception as exc:  # noqa: BLE001
        return OpenCVHighGUIProbeResult(
            ok=False,
            reason=f"OpenCV HighGUI probe failed: {type(exc).__name__}: {exc}",
        )

    if completed.returncode != 0:
        output = _compact_output((completed.stderr or "") + "\n" + (completed.stdout or ""))
        suffix = "" if not output else f": {output}"
        return OpenCVHighGUIProbeResult(
            ok=False,
            reason=f"OpenCV HighGUI probe exited with code {completed.returncode}{suffix}",
        )
    return OpenCVHighGUIProbeResult(ok=True)
