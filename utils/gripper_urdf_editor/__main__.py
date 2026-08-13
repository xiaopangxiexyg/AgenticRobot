from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import launch
from .core import URDFDocument, URDFValidationError
from .mesh import MeshCache


def _inspect(path: Path) -> int:
    document = URDFDocument.load(path)
    mesh_cache = MeshCache(max_triangles=200)
    pairs = []
    for pair in document.detect_gripper_pairs():
        endpoint_a = document.default_endpoint(pair.joint_a)
        endpoint_b = document.default_endpoint(pair.joint_b)
        closed, opened = document.inferred_open_closed(pair, endpoint_a, endpoint_b, mesh_cache.dimensions)
        closed_gap = document.opening_gap(pair, endpoint_a, endpoint_b, closed, mesh_cache.dimensions)
        open_gap = document.opening_gap(pair, endpoint_a, endpoint_b, opened, mesh_cache.dimensions)
        pairs.append(
            {
                "label": pair.label,
                "score": pair.score,
                "reason": pair.reason,
                "endpoint_a": f"{endpoint_a.link_name}/{endpoint_a.geometry.label}",
                "endpoint_b": f"{endpoint_b.link_name}/{endpoint_b.geometry.label}",
                "closed_q_mm": closed * 1000.0,
                "open_q_mm": opened * 1000.0,
                "closed_gap_mm": None if closed_gap is None else closed_gap * 1000.0,
                "open_gap_mm": None if open_gap is None else open_gap * 1000.0,
            }
        )
    print(json.dumps({"urdf": str(document.source_path), "pairs": pairs}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline GUI editor for two-finger parallel-gripper URDF endpoints."
    )
    parser.add_argument("urdf", nargs="?", type=Path, help="URDF to open")
    parser.add_argument("--inspect", action="store_true", help="print detected grippers without opening a GUI")
    args = parser.parse_args()
    try:
        if args.inspect:
            if args.urdf is None:
                parser.error("--inspect requires a URDF path")
            return _inspect(args.urdf)
        launch(args.urdf)
        return 0
    except URDFValidationError as exc:
        parser.exit(2, f"error: {exc}\n")
    except Exception as exc:
        if exc.__class__.__name__ == "TclError":
            parser.exit(2, f"error: GUI display is unavailable: {exc}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
