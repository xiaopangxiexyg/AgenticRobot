from __future__ import annotations

import argparse
import json
from pathlib import Path

from .real_image_provider import write_image_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index a local real-image background directory.")
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--source", default="local")
    parser.add_argument("--max-images", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = args.manifest
    if manifest is None:
        manifest = args.image_root / "real_background_manifest.json"
    payload = write_image_manifest(
        args.image_root,
        manifest,
        source=args.source,
        max_images=args.max_images,
    )
    print(json.dumps({"manifest": str(manifest), "count": payload["count"]}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
