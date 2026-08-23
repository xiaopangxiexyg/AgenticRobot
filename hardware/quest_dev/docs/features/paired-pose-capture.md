# Quest/Robot Manual Static Paired-Pose Capture R2

## Purpose and evidence level

R2 defines a lower-owned, passive reducer that turns already-admitted Quest
controller-pose events and already-published robot TCP-pose events into one
manual static-window calibration candidate. The reducer is deliberately:

- read-only with respect to both sources;
- in-memory and bounded;
- driven by explicit manual window boundaries;
- deterministic and replayable from its evidence;
- a candidate producer only, never a calibration acceptance authority.

The public contract is centered on:

- `StaticWindowCaptureProfile`;
- `QuestControllerCaptureBinding`;
- `PairedPoseCaptureBinding`;
- `PairedPoseCaptureFinalization`;
- `ManualStaticPairedPoseCapture`;
- `StaticWindowRejectedError` and `CaptureTerminalError`;
- `parse_paired_pose_capture_evidence()`.

The reducer is constructed as
`ManualStaticPairedPoseCapture(profile=..., binding=...)`. The application feeds
immutable source events through `ingest_robot_event()` and
`ingest_quest_event()`. The reducer never calls a refreshing robot snapshot API,
Quest `latest_sample()`, a device SDK, or a motion/control interface.

## Source bindings

### Shared receiver clock

The capture binding carries one complete
`robot_motion.runtime.identity.ClockDomainDescriptor`, not only a clock-domain
string. It must have `monotonic is True`. Its full descriptor hash is part of
the outer evidence.

Robot publications must carry that exact descriptor and descriptor hash. Quest
subscription cuts, event cuts, accepted events, and invalidation events carry
only `receipt_clock_domain_id`; every one of those IDs must equal the bound
descriptor's `clock_domain_id`. An ID match alone is not clock proof: the
application that constructs the capture binding is responsible for supplying
the exact descriptor shared by the two local receivers.

Pairing compares only the exact integer receipt times:

```text
abs(robot_publication.received_monotonic_ns
    - quest_sample.received_monotonic_ns)
```

It never subtracts Quest `xr_frame_time_ms`, `received_wall_time_s`, or a robot
native measurement timestamp from the other source. Native observed times and
their native clock-domain identities are retained in the R0 dataset, but are
not cross-source pairing clocks.

### Robot identity

One capture binds one exact robot pose stream and one exact
`PoseCoordinateBinding`. At minimum the evidence preserves:

- stream, control-domain, and component IDs;
- source instance, source epoch, source-binding hash, and source kind;
- reference frame ID and frame-spec hash;
- moving frame ID and tool-profile hash;
- FK contract ID and hash;
- pose schema hash;
- execution observation binding/plan identities and hashes;
- measurement and receipt clock-domain IDs;
- the coordinate binding's semantic hash.

The reducer consumes `PoseMeasurementAcceptedEvent` and
`PoseMeasurementInvalidationEvent` from the passive robot journal. It does not
infer a binding from a raw matrix or rebuild FK. Binding drift, source-epoch
change, sequence reuse with different content, a journal gap, or an
invalidation of the selected stream is a terminal capture fault. A recoverable
same-binding invalidation still invalidates the current capture; it is not
permission to continue an existing window.

Robot journal event sequence is global across all streams. The reducer must
advance one continuous global cursor through unrelated streams before filtering
to the selected stream. A per-stream cursor could hide an intervening gap or
invalidation and is forbidden.

### Quest identity and controller selection

The Quest binding pins the exact:

- server instance and authenticated client ID;
- client instance and XR session IDs;
- reference-space type and generation;
- controller `source_session_id`;
- handedness, target-ray mode, and the bound profiles assertion.

The controller is selected by the binding, never by array index or by taking
the first controller in a sample. Its pose must be non-emulated. Missing the
selected controller rejects that raw sample; it does not authorize switching to
another controller. A reference-space reset, session end, disconnect, runtime
stop, controller identity drift, or emulated pose fails closed. A
`next_reference_space_generation` on an invalidation event requires a new
capture binding and cannot extend the current capture.

`source_session_id` and WebXR profile strings are session-local observations,
not hardware serial numbers and not identities that may be reused across a page
reload or XR session.

## Exclusive event watermarks

Both sources are normalized to an exclusive `next_event_sequence` watermark.
For the robot journal, an atomic cut's inclusive
`PoseMeasurementJournalCut.event_sequence` is converted to
`event_sequence + 1`. Quest subscription and event cuts already expose
`next_event_sequence` directly.

A window begins with:

```python
begin_window(
    window_id=...,
    start_received_monotonic_ns=...,
    robot_start_next_event_sequence=...,
    quest_start_next_event_sequence=...,
)
```

and completion fixes:

```python
complete_window(
    end_received_monotonic_ns=...,
    robot_end_next_event_sequence=...,
    quest_end_next_event_sequence=...,
)
```

Only events in the half-open sequence intervals are eligible:

```text
robot: [robot_start_next_event_sequence, robot_end_next_event_sequence)
Quest: [quest_start_next_event_sequence, quest_end_next_event_sequence)
```

Their exact receiver timestamps must also lie within the declared manual
window. A cut's robot `latest_by_stream` and a Quest runtime's `latest_sample()`
are pre-window state and must not seed a window.

Start watermarks prevent read-latest/subscribe races. End watermarks do not mean
callbacks have finished: robot events may still need to be pulled from
`events_after()`, and Quest callbacks created before `atomic_event_cut()` may
still be in flight because callbacks execute outside the source lock.
`complete_window()` therefore returns no receipt while either source has not
been drained through its exclusive end watermark. The application must
continue delivering events and complete the asynchronous drain. It must never
seal merely because `unsubscribe()` returned.

Final capture closure uses a separate, typed close fence. The Quest source must
call `unsubscribe_with_cut(token)`, which removes the observer and returns an
exclusive `ControllerPoseObserverCloseCut` under the same runtime lock. The
distinct type carries the exact subscription token pinned by the capture
binding, so a plain `atomic_event_cut()` cannot be used as final-close evidence.
Calling
`unsubscribe()` and `atomic_event_cut()` separately is forbidden for sealing:
the former order can lose events allocated between the calls, while the latter
order can admit callbacks at or beyond the proposed cut. The robot source takes
one `PoseMeasurementJournalCut`. The application then calls:

```python
evidence = capture.seal_capture(robot_cut=robot_cut, quest_cut=quest_cut)
```

The first call pins those exact cuts and returns `None` until all robot events
and already-in-flight Quest callbacks below their exclusive watermarks have
been delivered. Repeated calls must provide the same cuts. Events at or beyond
the pinned close fence cannot enter the closed capture. A stopped Quest cut,
missing or changed bound session, subscription-token drift, invalid robot
stream, binding drift, or a cut that predates completed evidence fails closed.

The Quest callback ingress is one bounded global reorder buffer keyed by the
runtime-global `event_sequence`. Accepted and invalidation callbacks share that
buffer and are drained only in contiguous sequence order before identity
filtering. A duplicate sequence with different content, a missing sequence that
cannot be filled within the configured bound, or buffer overflow is a terminal
fault. A later accepted callback can never resurrect an already invalidated or
closed capture.

## Static-window admission and representative pair

The capture profile fixes finite bounds for memory, event reordering, window
duration/sample counts, stationarity, and receiver-time pairing skew. These are
capture-admission parameters, not calibration-quality acceptance thresholds.

For every completed window, stationarity is evaluated independently on all
eligible raw poses from each source. The evidence records at least:

- admitted raw sample count per source;
- source receipt-time span per source;
- maximum pairwise translation distance per source;
- maximum pairwise SO(3) rotation angle per source.

Insufficient samples, an out-of-profile window duration or source span, excess
motion, or absence of an eligible pair raises `StaticWindowRejectedError`.
That rejects the window without turning its data into a candidate. Malformed
or reversed API boundaries are caller errors. Identity, lifecycle, clock,
ordering, gap, or evidence-integrity failures raise `CaptureTerminalError` and
terminate the capture.

One admitted manual window produces exactly one representative raw pair. It
does not average poses and does not create several calibration poses from one
stationary hold. Candidate pairs are selected deterministically by:

1. receiver-time skew;
2. robot source sequence;
3. Quest sample sequence;
4. source event sequence tie-breakers.

The selected raw observations retain their original canonical source identity
and native timestamps. The shared `stationary_evidence_id` is the semantic hash
of a strict window stationarity payload that binds the boundaries, ordered raw
event hashes, selection, metrics, profile, and capture binding. It is unique
across admitted windows.

## Outer evidence and R0 dataset projection

`seal_capture(robot_cut=..., quest_cut=...)` returns an immutable
`PairedPoseCaptureEvidence` only after all admitted windows and the final close
fence are drained and complete. `evidence.as_dict()` is parsed again with
`parse_paired_pose_capture_evidence()` before use. The exact outer evidence
binds:

- capture profile and paired-source binding;
- the complete receiver clock descriptor and hash;
- start/end time and exclusive event watermarks for every window;
- ordered raw robot and Quest events and their canonical hashes, whose complete
  half-open sequence intervals preserve cut coverage and ordering evidence;
- robot coordinate/FK/tool/observation lineage;
- full Quest server/client/XR/reference/controller identity;
- the final typed robot journal cut and atomically unsubscribed Quest event cut,
  including active-session, stopped-state, latest-pose and close-watermark
  evidence;
- stationarity metrics and deterministic representative-pair selection;
- the projected R0 paired-pose dataset and its semantic SHA-256.

The unchanged R0 schema remains
`quest_dev.visualization.paired_pose_dataset.v1` with
`pairing_method="operator_static_window"`. Robot xyz+quaternion values are
converted to a validated rigid matrix. Quest wire matrices are converted only
through `controller_wire_matrix_to_transform()`; there is no implicit transpose
or inverse.

The R0 dataset is a compatibility projection, not the complete capture proof:
it does not itself retain receiver cuts, callback ordering, full robot binding,
or full Quest host/client identity. Those facts remain authoritative in the
outer evidence bundle.

Every sealed R2 result is still a calibration candidate. The outer evidence
sets `calibration_accepted` to `null`; any later diagnostic fit continues to
report `accepted: null`. Neither value may be changed by this reducer.

## Explicit exclusions

R2 paired capture does **not**:

- implement Control Station begin/complete/cancel intents, UI state, or
  application lifecycle;
- discover, construct, start, stop, or reconnect the unique Quest runtime;
- refresh a robot backend or create a second robot reader;
- persist evidence or datasets, choose a filesystem path, or own an episode;
- run the paired-pose fit;
- review or issue calibration acceptance;
- install a transform into a frame graph or visualization profile;
- start, home, servo, replay, hold, or otherwise command a robot or device;
- grant authority, weaken safety, or dispatch policy output.

Control Station may later provide thin operator lifecycle and source bindings,
but it must consume this reducer rather than reconstruct its ordering,
stationarity, pairing, or evidence semantics.

This reducer is provisionally Quest-dev feature-local because it consumes the
Quest runtime's typed events and projects into the pre-existing R0 Quest
calibration dataset. It does not define a new canonical dataset schema or make
`hardware/` a general calibration owner; promotion to a neutral calibration
runtime remains an explicit architecture decision before non-Quest reuse.
