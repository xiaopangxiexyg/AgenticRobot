# VIVE Clock Control Plane

This is an optional, sender-initiated UDP clock calibration channel for a
VIVE sender and a current DexTeleop collector.  It does not replace the pose
UDP stream.

## Stable receiver compatibility

The pose data plane remains exactly the existing one:

```text
sender -> receiver: UDP 9999 (or --port)
protocol: teleop_runtime.vive/v1
```

Clock messages are never sent to that port and the v1 pose JSON is not
extended.  The repository now has two sender entry points:

- `vive_data_sender.py` is the **new local default**. It starts the separate
  control client in `optional` mode automatically. The sender-side subset of
  this protocol is embedded in that file, so copying only
  `vive_data_sender.py` to the VIVE host retains clock synchronization;
  `clock_sync.py` remains the shared receiver/test protocol implementation.
- `vive_data_sender_legacy.py` is the **stable v1 entry point**. It forces
  clock control off, even if a clock-sync argument is supplied accidentally.

Both use the same v1 protocol/data port. The legacy entry point also freezes
the release-era root and tracker field shape (and its 60 Hz default), whereas
the new entry point retains the current 300 Hz pose-sample/velocity metadata.
Use `vive_data_sender_legacy.py` when the receiver is
`DexTeleop_StableCollect_0705` or any other released receiver that should
receive no additional control-plane traffic or newer JSON fields.

## Start a calibrated pair

The current local collector, real-teleop, calibration-debug, SAPIEN-debug,
and common `VivePoseSource` builder paths listen on UDP 10001 by default.
The default is non-blocking (`auto` behavior): an old sender continues to
work and channel metadata reports `clock_sync_status=unavailable`. Use
`--vive-clock-sync-required` when a recording must not begin until a mapping
is ready.

On the collector/control machine, a strict collection profile can use:

```bash
PYTHONPATH=. uv run python -m teleop_runtime.collection.run_collect \
  ... \
  --vive-clock-sync-expected-sender-host <vive-sender-ip> \
  --vive-clock-sync-required
```

The real-teleop entry point accepts the same `--vive-clock-sync-*` options.

On the VIVE sender machine, keep the existing pose destination unchanged:

```bash
python vive_data_sender.py --host <collector-ip> --port 9999
```

The new sender's default `optional` mode keeps streaming v1 poses when the
control port is blocked, unavailable, or belongs to an older receiver. Use
`--clock-sync-mode required` only when the collector listener is known to be
running. The sender rejects a control port equal to its pose `--port` to
prevent accidental traffic to a stable receiver.

For a stable receiver, select both explicit legacy ends instead:

```bash
# sender machine
python vive_data_sender_legacy.py --host <stable-collector-ip> --port 9999

# current local receiver, only when it must consume a legacy sender
... --vive-clock-sync-port 0
```

The collector must allow inbound UDP on the control port.  The sender
initiates all control traffic, avoiding an inbound firewall requirement on the
Windows VIVE machine.

Only one VIVE control listener may use a given `(bind host, control port)` at
a time. If two local sessions must coexist, assign different
`--vive-clock-sync-port` values; do not rely on UDP socket reuse to route the
control traffic correctly.

## Timing semantics

Each probe uses local `perf_counter_ns` values:

```text
sender t1 -> collector t2/t3 -> sender t4
offset(collector - sender) = ((t2 - t1) + (t3 - t4)) / 2
RTT = (t4 - t1) - (t3 - t2)
```

The collector retains low-RTT recent samples and exposes an estimated remote
capture time plus uncertainty in VIVE channel metadata.  For a synchronized
sender, that estimate is converted into the collector's wall-clock domain and
stored as `source_timestamp`; packet processing completion is stored as
`available_timestamp`.  Causal data selection is:

```text
latest sample with source_timestamp <= T0
             and available_timestamp <= T0
```

A remote pose that was captured before `T0` but arrives after it is not
eligible for that control step.  The monotonic receive fields remain in the
payload for transport-level audit; they are not silently mixed with the wall
clock used by the rest of a dataset row.  Sender restart (`seq == 0`)
invalidates the associated mapping until a new control exchange completes.
