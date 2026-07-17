# Pose Action State Lag Scan

- Dataset root: `/home/zc/HandMotion/Developing/DexTeleop/remote_data/state_action/2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz`
- Max lag: n+15
- Dedupe enabled: True
- Scanned datasets: 1
- Scanned episodes: 2
- Scanned rows: 297
- Skipped duplicate datasets: 0

## Overall Best Lag

| side | best lag by mean translation |
|---|---:|
| left | n+1 (9.74mm/1.69deg) |
| right | n+1 (10.60mm/1.64deg) |
| average(left,right) | n+1 (10.17mm/1.66deg) |

## Dataset Best Lag

| dataset | rows | left best | right best | avg best |
|---|---:|---:|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | 297 | n+1 (9.74mm/1.69deg) | n+1 (10.60mm/1.64deg) | n+1 (10.17mm/1.66deg) |

## Best Lag Variability

| dataset | side | mode lag | mode fraction | lag mean | lag std | transitions | longest run | best error mean mm | margin mean mm |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | left | n+1 | 0.562 | 1.32 | 0.75 | 94 | n+1 x 13 | 7.53 | 5.00 |
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | right | n+1 | 0.521 | 1.55 | 1.30 | 85 | n+1 x 19 | 7.42 | 6.31 |

## Mean Translation Error By Lag (mm)

| dataset | side | n+0 | n+1 | n+2 | n+3 | n+5 | n+8 | n+10 | n+12 | n+15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | left | 20.3 | 9.7 | 11.5 | 23.8 | 48.7 | 80.9 | 99.3 | 115.1 | 133.0 |
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | right | 22.4 | 10.6 | 11.8 | 25.0 | 52.3 | 87.9 | 107.7 | 124.5 | 142.9 |

Notes:

- `n+k` compares `action_t` with `observation.state_{t+k}` inside the same episode.
- Best lag is selected by mean translation error; rotation mean is shown for that same lag.
- Duplicate filtering uses `meta/info.json.dataset_name` as the default key.
