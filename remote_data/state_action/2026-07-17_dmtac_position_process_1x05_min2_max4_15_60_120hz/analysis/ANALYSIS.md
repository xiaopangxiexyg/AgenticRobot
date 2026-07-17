# Latest position-control dataset: action/state lag analysis

## Dataset and scope

- Remote source: `/home/user/zc/DexTeleop/2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz`
- Dataset: 2 episodes, 297 frames, nominal 15 Hz
- Local copy contains `data/` parquet files and `meta/` only.
- `videos/`, `tactile/`, and `wrist_depth/` were intentionally excluded.
- The repository statistic utilities compare the left/right arm TCP pose in
  `action_t` with the measured arm TCP pose in `observation.state_{t+x}`.
  They do not compare hand/gripper slots.

## Main result

The best fixed alignment is `action_t -> observation.state_{t+1}`. The observed
mean `t+1 - t` interval is 66.64 ms (standard deviation 5.51 ms).

| side | state_t mean error | state_t+1 mean error | reduction | state_t+2 mean error |
|---|---:|---:|---:|---:|
| left | 20.35 mm / 3.42 deg | 9.74 mm / 1.69 deg | 52.1% translation | 11.46 mm / 1.83 deg |
| right | 22.44 mm / 3.24 deg | 10.60 mm / 1.64 deg | 52.8% translation | 11.82 mm / 1.69 deg |

At `t+1`, translation-error median/p90 are 7.96/20.56 mm for the left arm and
8.70/21.77 mm for the right arm. Errors above 10 mm occur in 36.9% of left-arm
pairs and 44.4% of right-arm pairs. Rotation errors above 5 degrees occur in
2.0% and 3.1%, respectively.

The error curve has a narrow minimum across `t+1` and `t+2`, then rises quickly:

| lag | elapsed time | left mean translation | right mean translation |
|---:|---:|---:|---:|
| 0 | 0 ms | 20.35 mm | 22.44 mm |
| 1 | 66.64 ms | 9.74 mm | 10.60 mm |
| 2 | 133.27 ms | 11.46 mm | 11.82 mm |
| 3 | 199.88 ms | 23.77 mm | 25.01 mm |
| 5 | about 333 ms | 48.67 mm | 52.34 mm |
| 15 | about 1 s | 132.99 mm | 142.92 mm |

## Per-frame and per-episode behavior

Among frames with a complete 0..15 horizon, the per-frame best lag is mostly
one or two frames:

- Left: `t+1` 150/267 (56.2%), `t+2` 94/267 (35.2%), `t` 20/267 (7.5%).
- Right: `t+1` 139/267 (52.1%), `t+2` 107/267 (40.1%), `t` 10/267 (3.7%).

Both episodes independently select `t+1` as the fixed best lag:

- Episode 0 (49 frames): left/right 5.77/6.82 mm at `t+1`.
- Episode 1 (248 frames): left/right 10.52/11.33 mm at `t+1`.

Episode 1 therefore tracks less tightly than episode 0, but the temporal
alignment conclusion is consistent.

## Correlation cross-check

As a supplementary check, Pearson correlation was computed independently for
x/y/z and then averaged. Absolute TCP position correlation peaks at `t+1`:

| side | position corr at t | t+1 | t+2 | t+3 |
|---|---:|---:|---:|---:|
| left | 0.977 | 0.995 | 0.993 | 0.971 |
| right | 0.970 | 0.994 | 0.992 | 0.966 |

For frame-to-frame translation deltas, correlation peaks at `t+2`:

| side | delta corr at t | t+1 | t+2 | t+3 |
|---|---:|---:|---:|---:|
| left | 0.754 | 0.902 | 0.912 | 0.740 |
| right | 0.761 | 0.902 | 0.923 | 0.765 |

The two metrics are compatible: the measured pose is closest to the command
after one dataset frame, while the smoothed following motion continues into
the second frame. The practical response band is therefore approximately
67-133 ms, with `t+1` the correct single fixed alignment for training or
contract checks.

`observation.last_action_t` matches `action_{t-1}` exactly in translation for
all 295 comparable pairs on both sides (only negligible quaternion numerical
roundoff remains), so the stored last-action temporal contract is internally
consistent.
