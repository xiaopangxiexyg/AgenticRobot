# LeRobot Dataset Tools

This repository currently stores multiple local LeRobot-style dataset roots under `temp_test/`.
Each dataset root follows the same high-level layout:

```text
<dataset_root>/
  meta/
    info.json
    stats.json
    episodes.jsonl
    tasks.jsonl
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    chunk-000/
      observation.camera_0.rgb/
        episode_000000.mp4
```

Important details confirmed from the current data:

- `meta/info.json` is the source of truth for feature schema and path templates.
- `data/*.parquet` stores numeric tensors only: `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`.
- RGB observations are stored in MP4 files under `videos/`, not inside parquet.
- There are two active schema variants in this workspace:
  - 13D `observation.state` and 13D `action`
  - 8D `observation.state` and 8D `action`
- Some dataset roots are empty (`total_episodes = 0`), so code should guard against that.

## Commands

Scan every dataset root under `temp_test`:

```bash
python tools/lerobot_dataset_tool.py scan --search-root temp_test
```

Print a detailed summary for one dataset:

```bash
python tools/lerobot_dataset_tool.py summary \
  --dataset-root temp_test/fr3_o6_command_20260313_143406
```

Render a PNG overview for one episode:

```bash
python tools/lerobot_dataset_tool.py visualize \
  --dataset-root temp_test/fr3_o6_command_20260313_143406 \
  --episode-index 0
```

The visualization PNG contains:

- dataset metadata and resolved file names
- temporal norms for state and action
- a full heatmap over time for `observation.state`
- a full heatmap over time for `action`
- evenly sampled video frames from the matching MP4

If no `--output` is provided, images are written under:

```text
outputs/lerobot_vis/<dataset_name>/episode_XXXXXX.png
```

## Interactive player

The repository also includes an interactive player:

```bash
python tools/lerobot_episode_player.py \
  --dataset-root temp_test/fr3_o6_command_20260313_143406 \
  --episode-index 0
```

Layout and controls:

- left: `observation.camera_0.rgb`
- right: three columns for `action`, `state`, and `state-action`
- for 8D datasets it shows dims `6:8`
- for larger datasets it shows the last 6 dims
- `space` play/pause, `a/d` step, `z/x` rotate ccw/cw, `q` quit

For headless verification you can render a snapshot instead of opening the window:

```bash
python tools/lerobot_episode_player.py \
  --dataset-root temp_test/fr3_o6_command_20260313_143406 \
  --episode-index 0 \
  --snapshot-output outputs/player_snapshot.png
```
