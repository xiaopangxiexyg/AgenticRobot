# Pose Action State Lag Scan

- Dataset root: `/home/zc/HandMotion/Developing/DexTeleop/remote_data/state_action/2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz`
- Max lag: n+15
- Dedupe enabled: True
- Scanned datasets: 1
- Scanned episodes: 1
- Scanned rows: 49
- Skipped duplicate datasets: 0

## Overall Best Lag

| side | best lag by mean translation |
|---|---:|
| left | n+1 (5.77mm/0.81deg) |
| right | n+1 (6.82mm/0.83deg) |
| average(left,right) | n+1 (6.29mm/0.82deg) |

## Dataset Best Lag

| dataset | rows | left best | right best | avg best |
|---|---:|---:|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | 49 | n+1 (5.77mm/0.81deg) | n+1 (6.82mm/0.83deg) | n+1 (6.29mm/0.82deg) |

## Best Lag Variability

| dataset | side | mode lag | mode fraction | lag mean | lag std | transitions | longest run | best error mean mm | margin mean mm |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | left | n+1 | 0.647 | 1.35 | 0.48 | 14 | n+1 x 5 | 5.62 | 4.95 |
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | right | n+1 | 0.529 | 1.47 | 0.50 | 10 | n+1 x 4 | 6.00 | 4.59 |

## Mean Translation Error By Lag (mm)

| dataset | side | n+0 | n+1 | n+2 | n+3 | n+5 | n+8 | n+10 | n+12 | n+15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | left | 15.0 | 5.8 | 9.3 | 20.4 | 43.5 | 76.4 | 96.9 | 115.5 | 136.0 |
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | right | 15.6 | 6.8 | 8.2 | 18.8 | 41.2 | 73.7 | 94.2 | 112.7 | 133.0 |

Notes:

- `n+k` compares `action_t` with `observation.state_{t+k}` inside the same episode.
- Best lag is selected by mean translation error; rotation mean is shown for that same lag.
- Duplicate filtering uses `meta/info.json.dataset_name` as the default key.
