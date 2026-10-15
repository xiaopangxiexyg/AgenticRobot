"""Inspect retained cable masks only; never construct devices or a planner."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from deployments.perception.cable_diagnostics import (
    cable_depth_evidence,
    cable_prompt_from_geometry,
    separate_thin_cable_mask,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Retained RGB-D NPZ, not a device")
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--mask-key", required=True)
    parser.add_argument("--rgb", type=Path, help="RGB image if snapshot contains only depth")
    parser.add_argument("--robot-distances", type=Path,
        help="Aligned attachment_distance_m and rigid_surface_distance_m; not a 2D silhouette")
    parser.add_argument("--fx", type=float, help="Use recorded fx when not present in snapshot metadata")
    parser.add_argument("--maximum-cable-diameter-m", type=float, default=.025)
    parser.add_argument("--boundary-radius-px", type=int, default=4)
    parser.add_argument("--depth-tolerance-m", type=float, default=.015)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with np.load(args.snapshot, allow_pickle=False) as data:
        depth = data["depth_m"].copy()
        rgb = data["rgb"].copy() if "rgb" in data else None
        protection_available = "target_mask" in data
        protected = data["target_mask"].copy() if protection_available else np.zeros(depth.shape, bool)
        metadata = json.loads(data["metadata"].tobytes()) if "metadata" in data else {}
    if args.rgb is not None:
        rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    if rgb is None or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("A correlated RGB image matching the recorded depth is required")
    fx = args.fx
    if fx is None:
        fx = metadata.get("capture", {}).get("camera_intrinsics", {}).get("fx")
    if fx is None:
        raise ValueError("Provide the recorded fx with --fx; it must not be guessed")
    with np.load(args.mask_npz, allow_pickle=False) as data:
        mask = data[args.mask_key].copy()
    thin = separate_thin_cable_mask(depth_m=depth, mask=mask,
        protected_mask=protected, fx=fx, maximum_diameter_m=args.maximum_cable_diameter_m)
    evidence = cable_depth_evidence(depth_m=depth, mask=mask, protected_mask=protected,
        boundary_radius_px=args.boundary_radius_px, depth_tolerance_m=args.depth_tolerance_m)
    prompt = None
    if args.robot_distances is not None:
        with np.load(args.robot_distances, allow_pickle=False) as data:
            prompt = cable_prompt_from_geometry(depth_m=depth, thin_mask=thin.thin_mask,
                protected_mask=protected, attachment_distance_m=data["attachment_distance_m"],
                rigid_surface_distance_m=data["rigid_surface_distance_m"])
    arrays = {"thin_mask": thin.thin_mask, "thick_mask": thin.thick_mask,
              **{key: getattr(evidence, key) for key in evidence.__dataclass_fields__}}
    report = {"scope": "offline_diagnostic_no_scene_mutation_or_collision_permission",
        "counts": {key: int(value.sum()) for key, value in arrays.items()},
        "opening_radius_px": thin.opening_radius_px,
        "maximum_cable_diameter_m": args.maximum_cable_diameter_m,
        "boundary_radius_px": args.boundary_radius_px, "depth_tolerance_m": args.depth_tolerance_m,
        "target_protection_available": protection_available,
        "prompt_status": ("not_requested" if args.robot_distances is None
                          else "suggested" if prompt is not None else "no_supported_thin_component"),
        "prompt": None if prompt is None else asdict(prompt),
        "background_points_deleted": 0, "target_points_deleted": 0,
        "devices_started": False}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"diagnostic_masks.npz", **arrays)
    overlay = rgb.copy()
    for selected, color in ((thin.thin_mask, (220, 60, 190)),
                            (evidence.depth_continuous_boundary, (255, 200, 0)),
                            (evidence.depth_unexplained_boundary, (255, 90, 20)),
                            (protected, (50, 150, 255))):
        overlay[selected] = (.5*overlay[selected]+.5*np.asarray(color)).astype(np.uint8)
    sheet = Image.new("RGB", (rgb.shape[1]*2, rgb.shape[0]+65), "#202020")
    sheet.paste(Image.fromarray(rgb), (0, 65)); sheet.paste(Image.fromarray(overlay), (rgb.shape[1], 65))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Recorded RGB | Offline cable diagnostic only: no points removed", fill="white")
    draw.text((8, 27), "Magenta: thin proposal; gold: depth-continuous edge; orange: unexplained depth", fill="white")
    draw.text((8, 46), "Blue: protected target; missing depth remains unknown", fill="white")
    if prompt is not None:
        for point, color, name in ((prompt.positive_uv, "white", "+"), (prompt.negative_uv, "#00ffff", "-")):
            if point is None:
                continue
            u, v = point; u += rgb.shape[1]; v += 65
            draw.ellipse((u-5, v-5, u+5, v+5), outline=color, width=2)
            draw.text((u+7, v), name, fill=color)
    sheet.save(args.output/"overlay.png")
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
