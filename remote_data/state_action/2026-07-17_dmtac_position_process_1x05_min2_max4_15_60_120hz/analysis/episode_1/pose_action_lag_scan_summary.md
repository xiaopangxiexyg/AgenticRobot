# Pose Action State Lag Scan

- Dataset root: `/home/zc/HandMotion/Developing/DexTeleop/remote_data/state_action/2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz`
- Max lag: n+15
- Dedupe enabled: True
- Scanned datasets: 1
- Scanned episodes: 1
- Scanned rows: 248
- Skipped duplicate datasets: 0

## Overall Best Lag

| side | best lag by mean translation |
|---|---:|
| left | n+1 (10.52mm/1.86deg) |
| right | n+1 (11.33mm/1.79deg) |
| average(left,right) | n+1 (10.92mm/1.83deg) |

## Dataset Best Lag

| dataset | rows | left best | right best | avg best |
|---|---:|---:|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | 248 | n+1 (10.52mm/1.86deg) | n+1 (11.33mm/1.79deg) | n+1 (10.92mm/1.83deg) |

## Best Lag Variability

| dataset | side | mode lag | mode fraction | lag mean | lag std | transitions | longest run | best error mean mm | margin mean mm |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | left | n+1 | 0.549 | 1.32 | 0.78 | 80 | n+1 x 13 | 7.80 | 5.00 |
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | right | n+1 | 0.519 | 1.56 | 1.38 | 75 | n+1 x 19 | 7.63 | 6.56 |

## Mean Translation Error By Lag (mm)

| dataset | side | n+0 | n+1 | n+2 | n+3 | n+5 | n+8 | n+10 | n+12 | n+15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | left | 21.4 | 10.5 | 11.9 | 24.4 | 49.6 | 81.7 | 99.7 | 115.0 | 132.6 |
| 2026-07-17_dmtac_position_process_1x05_min2_max4_15_60_120hz | right | 23.8 | 11.3 | 12.5 | 26.2 | 54.4 | 90.4 | 110.0 | 126.3 | 144.4 |

Notes:

- `n+k` compares `action_t` with `observation.state_{t+k}` inside the same episode.
- Best lag is selected by mean translation error; rotation mean is shown for that same lag.
- Duplicate filtering uses `meta/info.json.dataset_name` as the default key.
