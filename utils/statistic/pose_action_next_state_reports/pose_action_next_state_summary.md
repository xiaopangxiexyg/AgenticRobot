# Pose Action vs Next State Summary

- Dataset root: `/home/zc/HandMotion/Developing/DexTeleop/datasets`
- Dataset count: 12
- Scanned episodes: 486
- Scanned rows: 69115

## Dataset Summary

| dataset | rows | left same mean mm/deg | left next mean mm/deg | right same mean mm/deg | right next mean mm/deg | best left | best right |
|---|---:|---:|---:|---:|---:|---|---|
| 2026-07-01_vive_rm65_real_teleop | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 2026-07-02_vive_rm65_real_teleop | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 2026-07-03_mujoco_real2sim_replay | 686 | 0.000/0.000 | 2.622/0.409 | 0.000/0.000 | 3.802/0.743 | same_state (0.000 mm) | same_state (0.000 mm) |
| 2026-07-03_vive_rm65_real_teleop | 343 | 21.648/3.495 | 18.963/3.083 | 33.565/6.101 | 29.873/5.418 | next_state (18.963 mm) | next_state (29.873 mm) |
| 2026-07-03_vive_rm65_real_teleop_undistorted (meta: 2026-07-03_vive_rm65_real_teleop) | 343 | 21.648/3.495 | 18.963/3.083 | 33.565/6.101 | 29.873/5.418 | next_state (18.963 mm) | next_state (29.873 mm) |
| 2026-07-03_vive_rm65_real_teleop_wrist_masked (meta: 2026-07-03_vive_rm65_real_teleop) | 343 | 21.648/3.495 | 18.963/3.083 | 33.565/6.101 | 29.873/5.418 | next_state (18.963 mm) | next_state (29.873 mm) |
| 2026-07-04_vive_rm65_real_teleop | 389 | 19.020/5.689 | 16.710/4.990 | 23.450/4.723 | 19.808/4.021 | next_state (16.710 mm) | next_state (19.808 mm) |
| 2026-07-04_vive_rm65_real_teleop_wrist_masked (meta: 2026-07-04_vive_rm65_real_teleop) | 389 | 19.020/5.689 | 16.710/4.990 | 23.450/4.723 | 19.808/4.021 | next_state (16.710 mm) | next_state (19.808 mm) |
| 2026-07-04_vive_rm65_real_teleop_wrist_masked_arm (meta: 2026-07-04_vive_rm65_real_teleop) | 389 | 19.020/5.689 | 16.710/4.990 | 23.450/4.723 | 19.808/4.021 | next_state (16.710 mm) | next_state (19.808 mm) |
| 2026-07-05_vive_rm65_real_teleop_collision | 2302 | 20.980/3.654 | 18.673/3.239 | 41.303/6.200 | 36.531/5.491 | next_state (18.673 mm) | next_state (36.531 mm) |
| 2026-07-05_vive_rm65_real_teleop_init_clean_stutter30_causal | 24883 | 33.652/5.529 | 29.910/4.887 | 33.591/5.009 | 29.931/4.451 | next_state (29.910 mm) | next_state (29.931 mm) |
| 2026-07-06_vive_rm65_real_teleop | 39048 | 35.626/6.590 | 31.603/5.815 | 37.796/6.519 | 33.926/5.836 | next_state (31.603 mm) | next_state (33.926 mm) |

## Last Action Check

| dataset | left mean mm/deg | right mean mm/deg |
|---|---:|---:|
| 2026-07-01_vive_rm65_real_teleop | n/a | n/a |
| 2026-07-02_vive_rm65_real_teleop | n/a | n/a |
| 2026-07-03_mujoco_real2sim_replay | 2.622/0.409 | 3.802/0.743 |
| 2026-07-03_vive_rm65_real_teleop | 0.000/0.000 | 0.000/0.000 |
| 2026-07-03_vive_rm65_real_teleop_undistorted (meta: 2026-07-03_vive_rm65_real_teleop) | 0.000/0.000 | 0.000/0.000 |
| 2026-07-03_vive_rm65_real_teleop_wrist_masked (meta: 2026-07-03_vive_rm65_real_teleop) | 0.000/0.000 | 0.000/0.000 |
| 2026-07-04_vive_rm65_real_teleop | 0.000/0.000 | 0.000/0.000 |
| 2026-07-04_vive_rm65_real_teleop_wrist_masked (meta: 2026-07-04_vive_rm65_real_teleop) | 0.000/0.000 | 0.000/0.000 |
| 2026-07-04_vive_rm65_real_teleop_wrist_masked_arm (meta: 2026-07-04_vive_rm65_real_teleop) | 0.000/0.000 | 0.000/0.000 |
| 2026-07-05_vive_rm65_real_teleop_collision | 0.000/0.000 | 0.000/0.000 |
| 2026-07-05_vive_rm65_real_teleop_init_clean_stutter30_causal | 0.000/0.000 | 0.000/0.000 |
| 2026-07-06_vive_rm65_real_teleop | 0.000/0.000 | 0.000/0.000 |

## Notes

- `same_state` compares `action_t` with `observation.state_t` arm pose.
- `next_state` compares `action_t` with `observation.state_{t+1}` arm pose within the same episode.
- `last_action` checks `observation.last_action_t` against `action_{t-1}`.
- Pose layout is `[x, y, z, qx, qy, qz, qw]`; quaternion sign is treated as equivalent.
