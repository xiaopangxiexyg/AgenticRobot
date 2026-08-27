# IK jitter investigation recording (`--debug-ik-0827`)

`--debug-ik-0827` is an explicit, opt-in observation mode for the intermittent
real-teleop jitter investigation.  It is intentionally separate from the
historical GUI IK-log toggle and from the routine `--debug-timing` modules.

Example (the normal real-teleop command and its safety gates are unchanged):

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --gui \
  --debug-ik-0827
```

## What is recorded

When the flag is present, the runtime starts one bounded asynchronous JSONL
recorder before the startup/Home wait.  The output is named
`ik_debug_0827_<timestamp>.jsonl` under the existing runtime IK-diagnostics
artifact directory (normally `diagnostics/ik`; the legacy fallback is
`teleop_records/ik_debug`).  The file uses the existing
`dexteleop_ik_debug_v1` row schema and identifies the run with
`recording_profile: "ik-0827"`.

Each control-loop frame keeps the existing compact IK/backend snapshot and
adds only for this profile:

- outer-loop phase and elapsed-time information;
- end-to-end, receiver, sender, and per-channel age information;
- VIVE/Q6 channel validity, sequence/source metadata, and tracker motion data;
- cached camera freshness/status information;
- IK timing/profile data, effective target step scale, command/backend state,
  RM65 CANFD background-stream counters and actual rate, deadline lag, SDK
  call timing, trajectory policy, and faults.

The recorder is bounded and best-effort.  A slow or unavailable diagnostic
filesystem logs a warning and leaves teleoperation running; dropped rows are
reported in the terminal `stop` payload.  Use `--ik-debug-log-full` only when
the larger historical command/pose payload is needed.

## Default-behavior boundary

Without `--debug-ik-0827`, the parser default remains `False`, the recorder is
not started, and the existing GUI toggle and timing-module behavior are
unchanged.  The investigation branch does not enable routine timing logs or
change command admission, safety limits, arm dispatch, writer startup,
progress manifests, or formal dataset contents.  In particular, frames from
this standalone investigation log are not routed into formal dataset debug
metadata.

`--no-debug-disk-write` remains authoritative.  If both flags are supplied,
the investigation recorder is not started and no debug file is written.

The flag enables the existing read-only IK timing profile only for the
explicit investigation run so that solver timing fields are populated.  It
does not alter solver targets, smoothing, trajectory policy, safety gates, or
hardware commands.  This mode is diagnostic evidence only and is not a
hardware-readiness or motion-authorization check.

## Reading a run

For a jitter interval, correlate the same frame's `timestamp` and
`outer_loop_timing` with:

1. `latency.channel_age` / `receiver_age` / `sender_age` and the corresponding
   `channels` entry;
2. `ik_timing` and the command `ik_metadata` fields;
3. each arm's `backend.background_streaming` (`actual_hz`, `last_call_ms`,
   `last_deadline_lag_ms`, `deadline_miss_count`, and `fault`);
4. `effective_control_dt_sec`, `effective_step_scale`, and trajectory-policy
   fields; and
5. camera freshness/fault fields if the stall coincides with a camera event.

Keep the exact command line, selected profile, CPU load, and JSONL file
together when comparing a normal run with a jittering run.  Do not infer that
an otherwise complete log authorizes activating a robot or changing a live
control parameter.
