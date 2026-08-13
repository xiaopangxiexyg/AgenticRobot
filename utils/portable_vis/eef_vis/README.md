# Portable Visualizations

## End-effector trajectory

Visualize recorded single-arm or bimanual end-effector trajectories in the robot
base frame:

```bash
python utils/portable_vis/visualize_eef_trajectory.py /path/to/session_or_dataset
```

By default the generated page opens in your browser automatically. To only
write the HTML:

```bash
python utils/portable_vis/visualize_eef_trajectory.py /path/to/session_or_dataset --no-open
```

You can also launch a browser picker first and choose files/directories there:

```bash
python utils/portable_vis/visualize_eef_trajectory.py
```

The picker runs on `localhost` and only uploads selected `*.parquet` and
`frames.jsonl` files back to the local Python process for parsing.

The tool accepts:

- a LeRobot-style dataset/session root containing `data/**/*.parquet`
- a single `episode_*.parquet`
- an `EpisodeRecorder` episode directory containing `frames.jsonl`
- a parent directory containing multiple `frames.jsonl` episodes

It writes a self-contained HTML file to:

```text
<input>/visualizations/eef_trajectory.html
```

Open that HTML in a browser. Drag to rotate, mouse wheel to zoom, and
Shift+drag to pan.

The default pose source is `auto`, which prefers the recorded actual wrist
camera pose:

```text
observation.wrist_camera.{left,right}.pose_base
```

Then it falls back to recorded/current TCP pose columns and finally
`observation.state` when needed.

Useful options:

```bash
python utils/portable_vis/visualize_eef_trajectory.py /path/to/dataset \
  --episode 0 3 \
  --pose-source actual \
  --orientation-mode current \
  --orientation-samples 120 \
  --max-points-per-side 20000
```

`--orientation-mode current` is the default and only shows the current-frame
TCP axes, so the view does not get cluttered by historical orientation triads.
Use `--orientation-mode history` to show sampled historical axes.

Reading parquet files requires `pyarrow`; reading `frames.jsonl` only needs
Python and NumPy.

## Pose action alignment

Visualize one dataset episode with two synchronized 3D panels:

- left panel: `action_t` vs `observation.state_t`
- right panel: `action_t` vs `observation.state_{t+1}`

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python utils/portable_vis/eef_vis/visualize_pose_action_alignment.py \
  datasets/2026-07-06_vive_rm65_real_teleop \
  --episode 10
```

By default it writes:

```text
<dataset>/visualizations/pose_action_alignment_episode_000010.html
```

Use `--no-open` when running on a headless machine, and `--output` to choose a
specific HTML path:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python utils/portable_vis/eef_vis/visualize_pose_action_alignment.py \
  datasets/2026-07-06_vive_rm65_real_teleop/data/chunk-000/episode_000010.parquet \
  --output /tmp/pose_action_alignment_ep10.html \
  --no-open
```

The page draws action as dashed/ring markers, state or next-state as solid
markers, and residual segments between them. The timeline slider is shared by
both panels so the same `action_t` can be compared against `state_t` and
`state_{t+1}` at once.
