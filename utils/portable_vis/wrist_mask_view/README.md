# Wrist RGB Mask Viewer

Inspect one recorded DexTeleop episode as synchronized wrist RGB and 0/1 mask
panels. The viewer opens a local browser GUI and reads files from the Python
process, so it works with server-local dataset paths.

## Quick Start

Open a dataset episode:

```bash
python utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --episode 0
```

Open from a concrete episode artifact:

```bash
python utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean/data/chunk-000/episode_000000.parquet
```

You can also pass a wrist-depth episode directory or a mask manifest:

```bash
python utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean/wrist_depth/mask/observation.images.left_wrist/chunk-000/episode_000000
```

If you are running over SSH:

```bash
python utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py --no-open
```

## What It Reads

The formal collection writer stores wrist RGB videos under:

```text
videos/observation.images.<camera_key>/chunk-XXX/episode_XXXXXX.mp4
```

When wrist depth mask saving is enabled, the corresponding 0/1 bit-packed mask
streams are recorded under:

```text
wrist_depth/mask/observation.images.<camera_key>/chunk-XXX/episode_XXXXXX/manifest.json
wrist_depth/mask/observation.images.<camera_key>/chunk-XXX/episode_XXXXXX/stream.bitpack.bin
```

For normal dataset-root input, the viewer first resolves paths through
`meta/episodes.jsonl`. If that metadata is absent but conventional files exist,
it falls back to the standard chunked layout.

## Controls

- `space`: play or pause
- `left` / `right`: previous or next frame
- `home` / `end`: first or last frame
- mask mode: switch between raw black/white 0/1 masks and an RGB overlay
- `Fit`: keep the composed frame fully visible inside the browser viewport.
  Turn it off to inspect the rendered JPEG at its actual pixel size with scrollbars.
- `JPEG`: open the current composed frame as a standalone JPEG

Render a still frame without opening the GUI:

```bash
python utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --episode 0 \
  --snapshot /tmp/wrist_rgb_mask_ep0.png \
  --frame 120
```
