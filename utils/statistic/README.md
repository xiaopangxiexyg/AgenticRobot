# Statistic Utilities

> 本页维护统计脚本的局部示例。当前数据处理、分析和验收入口统一从
> [`docs/data/README.md`](../../docs/data/README.md) 开始。

## Pose Action vs Next State

Run:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python utils/statistic/pose_action_next_state.py \
  --dataset-root datasets \
  --output-dir utils/statistic/pose_action_next_state_reports
```

The tool scans LeRobot-style DexTeleop datasets, including v2.2/v2.3 datasets
and UMI datasets converted to v2.3, and writes:

- `pose_action_next_state_summary.json`: machine-readable full summary.
- `pose_action_next_state_summary.md`: compact human-readable table.

Metrics:

- `same_state`: compares `action_t` arm pose with `observation.state_t`.
- `next_state`: compares `action_t` arm pose with `observation.state_{t+1}`.
- `last_action`: checks `observation.last_action_t` against `action_{t-1}`.

Pose vectors are `[x, y, z, qx, qy, qz, qw]`; quaternion sign is treated as equivalent.
For v2.2/v2.3 packed vectors, the first 14 values are the left and right arm
pose7 values. End-effector dimensions after those values are ignored by the
pose comparison. Explicit per-side pose columns take priority when present.
The historical 38D v2.1 state layout remains supported as a compatibility
fallback.

## Pose Action Lag Scan

Run:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python utils/statistic/pose_action_lag_scan.py \
  --dataset-root datasets \
  --max-lag 15 \
  --output-dir utils/statistic/pose_action_lag_scan_reports
```

The lag scan compares `action_t` against `observation.state_{t+k}` for
`k=0..max_lag`, then reports the lag with the smallest mean translation error.
By default it skips duplicate dataset directories using `meta/info.json.dataset_name`
as the dedupe key, so derived copies such as masked/undistorted variants are not
counted repeatedly.

For single-trajectory timing checks, scan just the latest or a specific episode:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python utils/statistic/pose_action_lag_scan.py \
  --dataset-root datasets/2026-07-09_vive_rm65_real_teleop \
  --latest-episode \
  --max-lag 15
```

In addition to mean error per lag, the report includes `best_lag_variability`,
which chooses the best lag independently for each frame. This shows whether a
nominal lag such as `n+9` is stable or whether the best alignment jumps across
different offsets.
