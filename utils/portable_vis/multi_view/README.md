# Multi View Video Player

Inspect several videos in one synchronized OpenCV window. This is meant for
quickly checking multi-camera recordings, replay outputs, and exported clips.

## Quick Start

Open the browser GUI:

```bash
python utils/portable_vis/multi_view/multi_view_video_player.py
```

The script starts a local `localhost` web server and opens the page in your
browser. In the page you can:

- drag local video files or folders into the drop zone
- use the file or folder buttons
- paste/type a server-local file path, folder path, or glob pattern into the path box
- reorder entries, adjust per-video offsets, scrub, change speed, and switch layouts

Dragged browser files are first played directly with object URLs. If the
browser cannot decode a dragged file, the page automatically uploads that file
to a temporary directory owned by the local Python process and switches it to
OpenCV frame-by-frame playback. Paths typed into the path box are resolved by
the Python process directly and also decoded with OpenCV, which is usually best
for `.avi`, MJPG, and other robot-recording formats.

Removing a video stops its active media element, cancels any in-flight fallback
upload, releases browser object URLs, and unregisters OpenCV frame sources from
the local Python server.

If you are running over SSH or do not want to open a browser automatically:

```bash
python utils/portable_vis/multi_view/multi_view_video_player.py --no-open
```

Preload files directly into the browser GUI:

```bash
python utils/portable_vis/multi_view/multi_view_video_player.py left.mp4 right.mp4 wrist.mp4
```

Preload a directory or glob:

```bash
python utils/portable_vis/multi_view/multi_view_video_player.py /path/to/session/videos --recursive
python utils/portable_vis/multi_view/multi_view_video_player.py "/path/to/session/**/*.mp4" --recursive
```

Render a still contact sheet without opening a GUI:

```bash
python utils/portable_vis/multi_view/multi_view_video_player.py /path/to/session/videos \
  --recursive \
  --contact-sheet /tmp/session_multiview.png \
  --at 12.5
```

Export a real/sim same-time comparison video from two dataset roots:

```bash
python utils/portable_vis/multi_view/real_sim_video_compare.py \
  --real-dataset datasets/2026-07-09_vive_rm65_real_teleop_sample_ep000000 \
  --sim-dataset sim/outputs/0709_v3_dual_replay_camera_depth_mp4_fovy_no_debug_visual_dataset \
  --episode 0 \
  --cameras front,left_wrist,right_wrist \
  --output sim/outputs/0709_v3_real_sim_compare.mp4
```

The comparison exporter reads `meta/episodes.jsonl` from both roots, resolves
same-name `video_paths`, and writes one MP4 where each camera is one row and
the columns are `real` and `sim`. Frames are sampled by relative episode time;
use `--real-offset` or `--sim-offset` when one side needs a fixed time shift.
`--cell-size WIDTHxHEIGHT` controls the size of each real/sim tile.

## Useful Options

```bash
python utils/portable_vis/multi_view/multi_view_video_player.py /path/to/videos \
  --recursive \
  --max-videos 12 \
  --layout 3x4 \
  --canvas-size 1920x1080 \
  --labels left,right,wrist,front \
  --offsets 0,-0.120,0.050,0 \
  --start-paused
```

- `--max-videos N`: load more than the old 4-video limit.
- `--layout auto|ROWSxCOLS`: use automatic layout or force a grid such as `2x3`.
- `--canvas-size WIDTHxHEIGHT`: choose the rendered canvas resolution.
- `--labels`: replace file-name labels. Values may be separated by spaces or commas.
- `--offsets`: per-video sync offsets in seconds. Local sample time is
  `global_time + offset`, so negative values delay a video.
- `--sort natural|name|path|mtime|none`: control discovery order.
- `--loop`: loop playback at the end of the synchronized timeline.
- `--snapshot-dir DIR`: where `s` saves PNG snapshots.
- `--host HOST` / `--port PORT`: choose the local web server endpoint.
- `--no-open`: print the URL without opening a browser.
- `--ui opencv`: use the older OpenCV window instead of the browser GUI.
- `--no-picker`: with `--ui opencv`, fail if no inputs are provided.

## Controls

- `space`: play or pause
- `left` / `right` or `a` / `d`: seek by `--seek-step` seconds
- `,` / `.` or `b` / `n`: previous or next frame and pause
- `up` / `down` or `+` / `-`: adjust playback speed
- `0`: reset speed to `1.0x`
- `home` / `end`: jump to start or end
- `r`: restart from the beginning
- `l`: toggle loop mode
- `s`: save the current canvas as a PNG
- `h`: show or hide the help overlay
- `q` or `Esc`: quit

The OpenCV `position` trackbar can also be dragged to scrub through the
synchronized timeline when using `--ui opencv`. In the browser GUI, use the
timeline slider at the top of the page.
