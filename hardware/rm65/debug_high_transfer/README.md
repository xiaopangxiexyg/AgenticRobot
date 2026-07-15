# RM65 High Transfer Debug

This directory contains hardware-facing diagnostics for RM65 transparent-transfer
experiments.

## Current Qpos CANFD Rate Probe

`hold_current_qpos_canfd_rate.py` measures how fast the SDK can submit
`Movej_CANFD` commands while repeatedly sending the arm's current startup qpos.
It is intended to test command throughput without requesting a position offset.

This script is a real hardware control tool. It requires explicit confirmation:

```bash
--execute --confirm SEND_CURRENT_QPOS_CANFD
```

## Safety Behavior

By default, each tested arm starts an independent safety monitor:

- reads current qpos at `--safety-read-hz` (default `50`)
- compares against the startup qpos
- calls `Arm_Stop()` immediately if any joint exceeds
  `--safety-max-delta-deg` (default `1.0`)

The final state is also checked with `--max-end-delta-deg` (default `0.5`).

The safety monitor can be disabled only with:

```bash
--no-safety-monitor
```

Use that only for controlled diagnostics where another stop mechanism is ready.

## Examples

Right arm, maximum submit rate:

```bash
python hardware/rm65/debug_high_transfer/hold_current_qpos_canfd_rate.py \
  --arms right \
  --right-ip "$RIGHT_ARM_IP" \
  --hz-list 0 \
  --duration-sec 5 \
  --execute \
  --confirm SEND_CURRENT_QPOS_CANFD
```

If max-rate mode prints repeated `Send would block` messages or state reads
start failing, first run a fixed-rate sweep:

```bash
python hardware/rm65/debug_high_transfer/hold_current_qpos_canfd_rate.py \
  --arms right \
  --right-ip "$RIGHT_ARM_IP" \
  --hz-list 40,80,120,160,200 \
  --duration-sec 3 \
  --execute \
  --confirm SEND_CURRENT_QPOS_CANFD
```

For max-rate mode only, `--max-rate-yield-sec 0.001` can be used to yield the
Python thread between submissions while still sending the same startup qpos.

Both arms, fixed-rate sweep plus max-rate run:

```bash
python hardware/rm65/debug_high_transfer/hold_current_qpos_canfd_rate.py \
  --arms both \
  --right-ip "$RIGHT_ARM_IP" \
  --left-ip "$LEFT_ARM_IP" \
  --hz-list 40,80,120,200,0 \
  --duration-sec 3 \
  --execute \
  --confirm SEND_CURRENT_QPOS_CANFD \
  --output-json /tmp/rm65_canfd_hold_rate.json
```

High-follow variant:

```bash
python hardware/rm65/debug_high_transfer/hold_current_qpos_canfd_rate.py \
  --arms right \
  --right-ip "$RIGHT_ARM_IP" \
  --hz-list 120,0 \
  --follow=True \
  --duration-sec 3 \
  --execute \
  --confirm SEND_CURRENT_QPOS_CANFD
```

## Key Options

- `--arms {right,left,both}`: arm selection, default `right`.
- `--hz-list`: comma-separated rates. `0` means no sleep, maximum achievable
  submit rate.
- `--duration-sec`: duration per rate.
- `--follow`: passes `follow=True` to `Movej_CANFD`. Accepts `--follow`,
  `--follow=True`, or `--follow True`.
- `--max-rate-yield-sec`: optional yield between sends only for `--hz-list 0`.
- `--trajectory-mode {0,1,2}`: passed through to the SDK adapter.
- `--radio`: SDK smoothing/radio parameter, `0..100`.
- `--continue-on-error`: keep counting SDK return-code errors instead of
  stopping on the first one.
- `--output-json`: write a machine-readable report.

## Output

For each arm and rate, the script prints:

- `send_count`
- `actual_send_hz`
- `send_call_ms mean/p95/p99/max`
- `loop_interval_ms mean/p95/p99`
- `error_count`
- final qpos drift from startup qpos

The JSON report includes the same data plus the safety monitor summary.
