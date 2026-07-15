from __future__ import annotations

import argparse
import json
from pathlib import Path

from .background import save_background_set
from .real_image_provider import write_image_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate procedural random backgrounds for masked dataset augmentation.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "generated")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--write-provider-manifest", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = save_background_set(
        args.output_dir,
        width=args.width,
        height=args.height,
        count=args.count,
        seed=args.seed,
    )
    provider_manifest = None
    if args.write_provider_manifest:
        provider_manifest = write_image_manifest(
            args.output_dir,
            args.output_dir / "real_background_manifest.json",
            source="procedural_color_geometry_v2",
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "count": manifest["count"],
                "provider_manifest": str(args.output_dir / "real_background_manifest.json") if provider_manifest else None,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
