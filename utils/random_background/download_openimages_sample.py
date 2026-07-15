from __future__ import annotations

import argparse
import csv
import http.client
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .real_image_provider import write_image_manifest


VALIDATION_IDS_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
IMAGE_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"


def _read_url_text(url: str, *, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "DexTeleopRandomBackground/1.0"})
    with urlopen(request, timeout=float(timeout)) as response:
        return response.read().decode("utf-8")


def _download_file(
    url: str,
    target: Path,
    *,
    timeout: float,
    retries: int,
    max_bytes: int | None = None,
) -> tuple[bool, int]:
    if target.exists() and target.stat().st_size > 0:
        return True, int(target.stat().st_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "DexTeleopRandomBackground/1.0"})
    for attempt in range(max(1, int(retries))):
        try:
            with urlopen(request, timeout=float(timeout)) as response:
                length = response.headers.get("Content-Length")
                if max_bytes is not None and length is not None and int(length) > int(max_bytes):
                    return False, 0
                data = response.read()
            if not data:
                return False, 0
            if max_bytes is not None and len(data) > int(max_bytes):
                return False, 0
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            return True, len(data)
        except (HTTPError, URLError, TimeoutError, http.client.IncompleteRead):
            if attempt + 1 >= max(1, int(retries)):
                return False, 0
            time.sleep(0.5 * float(attempt + 1))
    return False, 0


def _sample_validation_ids(*, count: int, stride: int, offset: int, timeout: float) -> list[str]:
    text = _read_url_text(VALIDATION_IDS_URL, timeout=timeout)
    rows = csv.DictReader(text.splitlines())
    ids: list[str] = []
    stride = max(1, int(stride))
    offset = max(0, int(offset))
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if (row_index - offset) % stride != 0:
            continue
        image_id = str(row.get("ImageID") or row.get("ImageId") or "").strip()
        if image_id:
            ids.append(image_id)
        if len(ids) >= int(count):
            break
    return ids


def download_openimages_validation_sample(
    output_dir: str | Path,
    *,
    count: int = 64,
    stride: int = 137,
    offset: int = 0,
    timeout: float = 30.0,
    retries: int = 3,
    max_bytes: int = 50 * 1024 * 1024,
    image_url_template: str = IMAGE_URL_TEMPLATE,
) -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    images_dir = output / "images"
    image_ids = _sample_validation_ids(count=count, stride=stride, offset=offset, timeout=timeout)
    ok: list[str] = []
    failed: list[str] = []
    skipped_size: list[str] = []
    total_bytes = sum(path.stat().st_size for path in images_dir.glob("*.jpg")) if images_dir.exists() else 0
    for image_id in image_ids:
        remaining = int(max_bytes) - int(total_bytes)
        if remaining <= 0:
            break
        target = images_dir / f"{image_id}.jpg"
        success, written_bytes = _download_file(
            str(image_url_template).format(image_id=image_id),
            target,
            timeout=timeout,
            retries=retries,
            max_bytes=remaining,
        )
        if success:
            ok.append(image_id)
            total_bytes += int(written_bytes)
        else:
            if remaining > 0:
                failed.append(image_id)
            else:
                skipped_size.append(image_id)
    manifest_path = output / "real_background_manifest.json"
    manifest = write_image_manifest(images_dir, manifest_path, source="openimages_v7_validation_sample")
    report = {
        "source": "openimages_v7_validation_sample",
        "ids_url": VALIDATION_IDS_URL,
        "image_url_template": str(image_url_template),
        "output_dir": str(output),
        "requested_count": int(count),
        "downloaded_count": len(ok),
        "failed_count": len(failed),
        "skipped_size_count": len(skipped_size),
        "max_bytes": int(max_bytes),
        "downloaded_bytes": int(total_bytes),
        "manifest": str(manifest_path),
        "manifest_count": int(manifest["count"]),
        "downloaded_ids": ok,
        "failed_ids": failed,
        "skipped_size_ids": skipped_size,
    }
    (output / "download_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a small Open Images validation sample for real background augmentation.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "real_images" / "openimages_v7_sample")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--stride", type=int, default=137)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--image-url-template",
        default=IMAGE_URL_TEMPLATE,
        help="URL template containing {image_id}.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=50 * 1024 * 1024,
        help="Hard cap for downloaded image bytes. Default: 50 MiB.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = download_openimages_validation_sample(
        args.output_dir,
        count=args.count,
        stride=args.stride,
        offset=args.offset,
        timeout=args.timeout,
        retries=args.retries,
        max_bytes=args.max_bytes,
        image_url_template=args.image_url_template,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
