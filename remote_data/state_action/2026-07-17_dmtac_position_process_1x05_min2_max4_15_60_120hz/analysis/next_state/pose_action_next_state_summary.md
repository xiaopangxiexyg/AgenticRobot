# Pose Action vs Next State Summary

- Dataset root: `/home/zc/HandMotion/Developing/DexTeleop/remote_data/state_action/2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz`
- Dataset count: 1
- Scanned episodes: 2
- Scanned rows: 297

## Dataset Summary

| dataset | rows | left same mean mm/deg | left next mean mm/deg | right same mean mm/deg | right next mean mm/deg | best left | best right |
|---|---:|---:|---:|---:|---:|---|---|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | 297 | 20.349/3.416 | 9.743/1.689 | 22.442/3.236 | 10.599/1.637 | next_state (9.743 mm) | next_state (10.599 mm) |

## Last Action Check

| dataset | left mean mm/deg | right mean mm/deg |
|---|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | 0.000/0.000 | 0.000/0.000 |

## Notes

- `same_state` compares `action_t` with `observation.state_t` arm pose.
- `next_state` compares `action_t` with `observation.state_{t+1}` arm pose within the same episode.
- `last_action` checks `observation.last_action_t` against `action_{t-1}`.
- Pose layout is `[x, y, z, qx, qy, qz, qw]`; quaternion sign is treated as equivalent.
