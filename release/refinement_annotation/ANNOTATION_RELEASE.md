# DexTeleop refinement annotation projection

This directory is generated from the canonical DexTeleop source projection.
It preserves the original package paths and is not an independent package or
data-protocol fork. Projection id: `59bd2fa71e591b085decb4e9d41903d40526b89015dac8f2b08d9004e15d12e8`.

It contains no dataset, annotation result, media cache, robot runtime, hardware
driver, simulator, deployment code, or training code.

## Verify

From this directory, run:

```bash
python scripts/export_annotation_projection.py --verify .
python scripts/check_annotation_projection.py
```

## Install and start

The host needs Linux, `uv`, Python 3.10, `ffmpeg`, `ffprobe`, and a modern
browser. Then run:

```bash
bash scripts/install_annotation_env.sh
source .venv-annotation/bin/activate
python -m control_station refinement-annotation
```

The GUI can select or change datasets. Dataset paths and retrospective source
roots are supplied at runtime and must not be copied into this code release.
See `README_ENV.md` and `docs/real/LONG_HORIZON_STAGE_ANNOTATION.md` for details.
