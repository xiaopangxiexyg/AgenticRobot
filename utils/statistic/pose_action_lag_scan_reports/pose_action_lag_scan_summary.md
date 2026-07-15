# Pose Action State Lag Scan

- Dataset root: `/home/zc/HandMotion/Developing/DexTeleop/datasets`
- Max lag: n+15
- Dedupe enabled: True
- Scanned datasets: 8
- Scanned episodes: 480
- Scanned rows: 67651
- Skipped duplicate datasets: 4

## Overall Best Lag

| side | best lag by mean translation |
|---|---:|
| left | n+8 (10.17mm/1.62deg) |
| right | n+8 (10.37mm/1.67deg) |
| average(left,right) | n+8 (10.27mm/1.64deg) |

## Dataset Best Lag

| dataset | rows | left best | right best | avg best |
|---|---:|---:|---:|---:|
| 2026-07-03_mujoco_real2sim_replay | 686 | n+0 (0.00mm/0.00deg) | n+0 (0.00mm/0.00deg) | n+0 (0.00mm/0.00deg) |
| 2026-07-03_vive_rm65_real_teleop | 343 | n+6 (10.08mm/1.62deg) | n+6 (16.24mm/2.89deg) | n+6 (13.16mm/2.25deg) |
| 2026-07-04_vive_rm65_real_teleop | 389 | n+7 (8.33mm/2.08deg) | n+5 (8.34mm/1.64deg) | n+6 (8.55mm/1.93deg) |
| 2026-07-05_vive_rm65_real_teleop_collision | 2302 | n+8 (8.24mm/1.56deg) | n+8 (12.07mm/1.82deg) | n+8 (10.16mm/1.69deg) |
| 2026-07-05_vive_rm65_real_teleop_init_clean_stutter30_causal | 24883 | n+8 (9.55mm/1.44deg) | n+8 (9.30mm/1.36deg) | n+8 (9.42mm/1.40deg) |
| 2026-07-06_vive_rm65_real_teleop | 39048 | n+8 (10.45mm/1.70deg) | n+8 (10.55mm/1.77deg) | n+8 (10.50mm/1.73deg) |

## Mean Translation Error By Lag (mm)

| dataset | side | n+0 | n+1 | n+2 | n+3 | n+5 | n+8 | n+10 | n+12 | n+15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-03_mujoco_real2sim_replay | left | 0.0 | 2.6 | 5.2 | 7.8 | 12.9 | 20.2 | 24.8 | 29.1 | 35.0 |
| 2026-07-03_mujoco_real2sim_replay | right | 0.0 | 3.8 | 7.6 | 11.3 | 18.6 | 29.1 | 35.5 | 41.6 | 49.6 |
| 2026-07-03_vive_rm65_real_teleop | left | 21.6 | 19.0 | 16.3 | 13.8 | 10.7 | 10.8 | 14.0 | 18.1 | 24.4 |
| 2026-07-03_vive_rm65_real_teleop | right | 33.6 | 29.9 | 26.0 | 22.3 | 17.2 | 17.4 | 21.0 | 25.9 | 33.7 |
| 2026-07-04_vive_rm65_real_teleop | left | 19.0 | 16.7 | 14.5 | 12.4 | 9.2 | 8.9 | 11.5 | 15.0 | 20.9 |
| 2026-07-04_vive_rm65_real_teleop | right | 23.4 | 19.8 | 16.1 | 12.5 | 8.3 | 13.1 | 19.2 | 25.9 | 35.8 |
| 2026-07-05_vive_rm65_real_teleop_collision | left | 21.0 | 18.7 | 16.4 | 14.2 | 10.3 | 8.2 | 9.9 | 13.1 | 18.9 |
| 2026-07-05_vive_rm65_real_teleop_collision | right | 41.3 | 36.5 | 31.8 | 27.1 | 18.7 | 12.1 | 15.2 | 22.1 | 34.2 |
| 2026-07-05_vive_rm65_real_teleop_init_clean_stutter30_causal | left | 33.7 | 29.9 | 26.1 | 22.3 | 15.0 | 9.6 | 12.8 | 19.0 | 29.3 |
| 2026-07-05_vive_rm65_real_teleop_init_clean_stutter30_causal | right | 33.6 | 29.9 | 26.2 | 22.5 | 15.2 | 9.3 | 12.2 | 18.3 | 28.7 |
| 2026-07-06_vive_rm65_real_teleop | left | 35.6 | 31.6 | 27.5 | 23.5 | 15.9 | 10.4 | 14.2 | 20.8 | 31.5 |
| 2026-07-06_vive_rm65_real_teleop | right | 37.8 | 33.9 | 30.0 | 26.0 | 18.1 | 10.5 | 12.5 | 18.7 | 29.8 |

## Skipped Duplicates

| skipped dataset | dedupe key | kept dataset |
|---|---|---|
| 2026-07-03_vive_rm65_real_teleop_undistorted | 2026-07-03_vive_rm65_real_teleop | 2026-07-03_vive_rm65_real_teleop |
| 2026-07-03_vive_rm65_real_teleop_wrist_masked | 2026-07-03_vive_rm65_real_teleop | 2026-07-03_vive_rm65_real_teleop |
| 2026-07-04_vive_rm65_real_teleop_wrist_masked | 2026-07-04_vive_rm65_real_teleop | 2026-07-04_vive_rm65_real_teleop |
| 2026-07-04_vive_rm65_real_teleop_wrist_masked_arm | 2026-07-04_vive_rm65_real_teleop | 2026-07-04_vive_rm65_real_teleop |

Notes:

- `n+k` compares `action_t` with `observation.state_{t+k}` inside the same episode.
- Best lag is selected by mean translation error; rotation mean is shown for that same lag.
- Duplicate filtering uses `meta/info.json.dataset_name` as the default key.
