# Rule-based real writing collection

The default profile is rule-based writing; the opt-in `full-replay` profile
records one complete source dataset Episode without interpreting its
detail-stage annotations.

The launcher selects the dedicated trajectory-collection application and
browser UI through the existing RealTeleop entrypoint. The specialized runtime
extension is bound before device preflight, owns the exclusive rule command
override, disables human gripper input and VIVE requirements, binds configured
DMTAC backends into the Session Rig, and uses robot-state-only runtime Home.
The integrated launcher has completed supervised real-device collection; every
new deployment still requires its normal device, calibration, workspace, and
operator preflight.

The application still composes the existing authoritative robot loop, replay,
fixed trajectory, safety chain, and formal writer. It does not introduce a
second hardware loop, safety owner, or recorder. The launcher explicitly selects
`gripper_input_source=none`, so it never probes Q6 or creates an
angle-trigger fallback; configured DMTAC backends remain available for
measured state and tactile capture. The staged dedicated UI exposes only idle
source/path/speed configuration, an ordered prevalidated queue, rule
start/stop, pre-collection gripper reset, rule-step state, and device/safety
messages. The active application accepts the rule Session after robot-state
Home in `ARMED`; it does not require a prior human-teleop command dispatch.
`FROZEN`, Hold, Init, Fix-Hold, dataset reconfiguration, and the remaining
collection gates still reject the start.

## Executed plan

The fixed plan is:

1. Move both arms through the existing qpos initialization path to the first
   `right_write.pen_down` source frame. This phase is not admitted to the
   dataset. The progress profile's left-arm `fixed_hold` is deliberately not
   applied at runtime Home: both arms remain live for these certified
   initialization waypoints.
2. Replay and record `right_write.pen_down`. The right arm follows the source;
   the left arm is pinned to the source episode's first `fixed_hold` target.
3. Generate and record one selected catalog path on the right TCP. The path is
   anchored at the causally measured pen-down endpoint. The left arm keeps the
   same source `fixed_hold` target.
4. Publish the measured writing endpoint and relative-replay and record
   `right_write.pen_up` on the right arm. The left arm receives no new target
   and remains held.

Only steps 2-4 are admitted to the formal dataset. Cameras, tactile capture,
sample synchronization, and persistence remain atomic and are owned by the
existing formal writer.

The left-arm target is selected per formal Episode, not once per launcher.
For a normal Episode, the first certified initialization moves the left arm to
frame 0 of that Episode's selected `pen_down` source while it moves the right
arm to the matching start. FixArm captures the left qpos only after measured
arrival, and the left arm then remains stationary through pen-down, generated
writing, and relative pen-up. After a successful writer close, the runtime
releases only that deferred FixArm authority and activates the next validated
batch item. The next operator Start therefore performs a new bimanual
initialization to that item's own source state. No inter-Episode motion starts
automatically, and failed or aborted attempts do not advance the batch.

An optional source-pool prelude can be inserted before step 1 with
`--pre-pen-down-replay-candidate-id ID` or the dedicated UI selector. When it
is enabled, the runtime first uses the same collision-checked qpos
initialization to approach frame 0 of that candidate, replays the candidate's
complete right-arm Episode, and then runs a second collision-checked
initialization to the selected `pen_down` frame 0. The prelude and both smooth
transitions have capture disabled, so the formal output still contains only
the existing pen-down/write/pen-up stages. The left arm action and measured
qpos/TCP initialization state are pinned to the selected pen-down source hold
throughout the prelude. Enabled DMTAC channels keep using the existing causal,
step-limited runtime path.

`--pre-pen-down-replay-normal-episodes N` controls how often the configured
prelude is used. For `N > 0`, the runtime commits `N` ordinary successful
Episodes, then uses the prelude for one Episode, resets the counter, and
repeats. An Episode counts only after the formal writer has closed
successfully; Hold, abort, failed startup, and discarded attempts do not
consume the interval. `N=0` preserves the previous behavior and uses the
configured prelude before every Episode. If no prelude candidate is selected,
the interval has no motion effect. The selected interval and the current
schedule state are written into rule-based sample provenance.

The option is disabled by default and is intentionally available only with a
source-pool manifest, where `ID` has one exact dataset/Episode identity inside
the selected pool range. Legacy single-source and `full-replay` modes reject
it instead of inferring a trajectory. Every item in a configured GUI batch
stores its own optional prelude choice, so the sequence is rebuilt before each
formal Episode without changing the writer or device lifecycle.

## Complete source-Episode replay mode

Pass `--mode full-replay` to select the additional, startup-time-isolated
profile. It does not select `pen_down`, `write`, or `pen_up`, and it never
turns one annotated substage into a complete trajectory. One startup source choice
identifies exactly one dataset and one Episode index; the runtime loads all of
that Episode's action rows and real timestamps. `--speed-m-s` is rejected in
this mode so a nominal writing speed cannot silently retime the source Episode.

Its complete plan is deliberately only:

1. Move both arms and every enabled DMTAC gripper through the same certified,
   measured-feedback initialization to source frame 0. Do not record this
   transition.
2. Replay and record source frames `0..N-1` in order. Both arms remain active;
   enabled grippers use the source's resolved DMTAC targets. Each source frame
   is admitted at most once, while arm references continue through the shared
   60 Hz reference and 120 Hz command transport.
3. Finalize the output as one independent Episode, then return to the existing
   next-item-ready state. The staged dedicated UI can queue multiple complete
   source Episodes and still requires an explicit formal Episode for every
   output.

This profile uses the existing `ReplayTrajectoryGroup`,
`RuleBasedInitialize`/`RuleBasedReplay`, collision/TCP envelope, Session/Rig,
formal writer, Hold/recovery, and successful-attempt reset. It adds no replay
cursor, device loop, writer, or production entrypoint. Its dedicated progress
binding has one initial `execute_task.full_replay` stage, so the whole Episode
has one truthful label and does not apply the writing profile's left-arm Fix.

Legacy direct source selection does not need a detail annotation directory:

```bash
bash scripts/run_rule_based_writing_collection.sh \
  --mode full-replay \
  --source-dataset /absolute/path/to/source_dataset \
  --source-episode 42 \
  --dataset-root /absolute/path/to/output_root \
  --tactile-bundle-root /absolute/path/to/tactile_root \
  --initial-alignment-speed-scale 0.5 \
  --dry-run
```

With `--source-pool-manifest`, every candidate in the selected range remains
a complete-Episode choice regardless of which writing subphases made it
eligible for the pool. The current launcher selects deterministically from
startup arguments and the dedicated UI exposes the same candidates. The source
pool is only an
immutable dataset/Episode index here; its detail annotation artifacts are not
opened.

At the final initialization waypoint, the Session first applies its normal
command filters and safety guard, the Rig/backend accepts the command, and
causal arm feedback must report consecutive arrival samples. Only then does
the rule runtime ask the existing FixArm controller to capture the left-arm
qpos from that exact filtered command. The first replay/sample remains blocked
until this handoff matches the progress profile. Any filter rewrite, stale
pre-existing Fix, or failed handoff enters Hold immediately instead of waiting
for the target-arrival timeout.

The formal Episode may already be open while that measured-arrival handoff is
pending. In this bounded state it remains command-inert and sample-inert: the
initialization cursor may continue, but no replay frame or dataset sample can
advance. Adding the left-arm Fix never invents a runtime Home transaction,
because no arm is becoming live. Only a profile transition that releases a
previously fixed arm may capture that exact newly-live side at Home. If the
rule runtime terminates during this handoff, its original abort reason is
published to the operator status path before the writer reclaims the Episode.

One rule frame advances only after both authoritative boundaries agree: every
commanded effector must have an accepted Rig/backend receipt, and the formal
writer must accept that frame's sequence index. A transient DMTAC
`rate_limited`/`io_busy` result retains and retries the same frame without
sampling; a writer rejection likewise retains the frame. Both retry paths are
bounded and enter Hold on timeout. Terminal arrival and relative-phase boundary
publication additionally wait for writer acceptance, so a dropped final sample
cannot be reported as a completed phase. Live RM65 dispatch timestamps are
fresh, strictly increasing wall-clock values; the source dataset/trajectory
timestamp is retained separately as provenance and never drives the real
transport clock.

Rule replay uses the existing frozen V2 Balanced position stack: source
timestamps define the replay duration, the shared replay resampler publishes
safe references at 60 Hz, and the established acceleration-limited backend
planner independently streams accepted commands at 120 Hz. Formal samples
remain a separate 15 Hz writer concern. The rule launcher intentionally does
not select the opt-in 60 Hz Sample-Aligned QP profile: that profile
solves a horizon synchronously at each strictly newer reference and can make
the outer reference loop contend with the command worker. Coupled IK is
anchored from the accepted qcmd. The stable default retains the previously
admitted IK velocity; an operator may explicitly enable
`--position-execution-qvel-feedback` to replace it with qvel from the same
authoritative planner snapshot. In that mode a qcmd/qvel snapshot mismatch
fails closed rather than combining states from different execution ticks.

## Hand topology

This specialized launcher intentionally uses no Q6 or angle-trigger input. By
default it disables the left hand backend and constructs only the right DMTAC
gripper backend from the saved assignment. The right-side endpoint is resolved
from the current assignment/preflight result (a `/dev/ttyUSB*` number is not a
physical identity and must not be copied between runs). Device preflight
therefore requires one right-side DMTAC stack and stops serial probing after
that one reachable gripper satisfies admission.

Replay end-effectors follow the selected source hand channels whenever the
corresponding DMTAC backend is enabled. Typed partial/NaN hand slots and the
fully resolved `action.replay.<side>_hand_joint_target` fallback are resolved by
the common replay contract, then normalized to the real DMTAC topology: one
active actuator in slot 0 and five finite zero-padding slots. During the initial
transition, each enabled hand is interpolated from its causal measured state to
the first `pen_down` target with a maximum normalized step of `0.05`; the first
replay frame is not released until both arms and enabled hands have settled.
Intermediate certified arm qrefs are streamed at 60 Hz instead of stopping at
each sparse path waypoint. Source advancement freezes if measured arm state falls more
than four configured waypoints behind, and only the final waypoint requires the
strict 0.15-degree arrival/dwell check. Repeated dispatch of one qref refreshes
the transport timestamp without inventing a new sample-aligned source cut.

The selected source `write` hand trajectory is rebased continuously from the
plan's final `pen_down` target and time-normalized to the generated rule path.
The selected `pen_up` hand trajectory is similarly rebased from the final write
target. Every cross-phase boundary is continuous, every per-frame step is
checked against the same bound, and all three phase terminal frames wait for
the commanded arms and enabled hands to settle. A selected source that cannot
provide a finite compatible DMTAC trajectory fails before any device opens.

In the default topology the left hand backend is disabled and the right DMTAC
state/tactile stream is captured; the dataset contract records the left hand as
`hand_model=none` with explicit inactive padding. Pass `--with-left-gripper`
(and, only when necessary, an identity-verified `--left-gripper-port`) to
connect and capture both DMTAC stacks. Re-run the assignment/preflight after
reconnecting hardware instead of guessing a `/dev/ttyUSB*` number. This
remains without Q6, a foot pedal, or a second device loop. The runtime status
publishes the exact control topology, current phase/frame, accepted/plan sample
count, and initialization speed multiplier.

The dedicated UI contains “松开夹爪” and “闭紧夹爪”. They are available only while
the shared Session is actually
`ARMED`. A click starts a feedback-gated sequence rather than one full-range
command: every newly advanced normalized target increment is at most `0.05`
from the latest fresh measurement. If an accepted register write produces no
measured motion, the runtime re-sends only that same bounded target at a low
rate; it neither advances another increment nor resets the no-progress timeout.
The UI reports completion only after the final measured target is reached.
The sequence also retains the DMTAC backend's configured hardware speed,
torque, and 20 Hz command admission, times out fail-closed after 20 seconds,
and rejects missing, stale, or invalid feedback. Every increment goes through
the Session filter, guard, admission, and Rig dispatch seam and never records a
formal sample. In rule mode the configured DMTAC backend is registered as the
Rig's `left_hand`/`right_hand` backend even though Q6 is absent; ordinary
teleoperation retains its separate scalar-input topology. Starting formal rule
collection cancels any queued reset and locks both buttons. Rule mode rejects
Q6 or angle-trigger input when a DMTAC backend is enabled, preventing two
hand-command owners.

While a manual gripper action is pending, both gripper buttons, rule start, and
all plan-configuration controls are disabled. The runtime independently rejects
a second queued gripper intent, so HTTP/UI races cannot replace the in-flight
feedback-gated target.

## Coordinate and admission boundary

The application fails before device opening unless all of these conditions
hold:

- the source is a typed dataset with an explicit pose-frame contract;
- the source and current real deployment pose-frame contract hashes match
  exactly;
- source action and actual-TCP anchors can be converted to `canonical_desk`;
- the detail overlay exactly binds the source sealed progress Episode and its
  base annotation set;
- the ordered detail stages are successful `pen_down`, `write`, and `pen_up`;
- every selected source subphase labels the left arm `fixed_hold`;
- both arms, position mode, collision avoidance, active TCP envelopes,
  tool-frame force guarding, and formal collection are enabled;
- all Active admittance overlays are disabled.

The launcher pins
`teleop_runtime/configs/real_teleop/rule_based_writing_bimanual_workspace_v1.json`
and enables it in Active mode. The profile covers both final `tool_tcp` targets
at the authoritative pre-IK boundary and resolves each side's absolute bounds
from the same runtime calibration workspace already selected for that arm. It
does not add a force/admittance modifier or maintain a second workspace owner.

Dataset poses remain stored as `float32` xyz-quaternions. Immediately before
the coupled target owner, both replay targets and the measured inactive-side
hold pose undergo a bounded SO(3) projection with a `1e-5` maximum
orthogonality, determinant, and projection error. This only removes
representation-scale drift (normally about `1e-8`); reflections, non-finite
values, bad homogeneous rows, and materially distorted rotations still fail
closed. Projection diagnostics are attached to the rule command metadata, and
the final Active TCP envelope retains its independent `1e-9` rigid-pose check.

The fixed path is authored in glyph-local XY and rotated only within
`canonical_desk`. Its right-arm start is the causal real TCP endpoint converted
through the current runtime contract. The existing RealTeleop resolver then
performs the canonical-desk to per-arm hardware-frame conversion.

## Built-in and operator-authored trajectories

The versioned built-in catalog contains nine digits, 24 Greek letters, and all
26 uppercase English letters. English identifiers are `latin_a_v1` through
`latin_z_v1`. They use the same one-stroke, size, four-direction, speed, desk
frame, causal anchoring, left-arm hold, and dataset-capture contracts as the
existing digits and Greek letters.

An operator can append trajectories without changing Python or the production
launcher. Put one or more paths in a separate JSON file with the same catalog
schema, then pass both the catalog and selected trajectory id:

```json
{
  "schema": "agentic_robot.rule_based_fixed_trajectories.v1",
  "path_frame": "glyph_local_xy",
  "normalization": "bounding_height_is_one",
  "trajectories": [
    {
      "trajectory_id": "custom_triangle_v1",
      "family": "custom",
      "symbol": "triangle",
      "stroke_count": 1,
      "points": [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [0.0, 0.0]]
    }
  ]
}
```

```bash
bash scripts/run_rule_based_writing_collection.sh \
  ... \
  --trajectory-catalog /absolute/path/to/operator-trajectories.json \
  --trajectory custom_triangle_v1
```

The external file is an overlay, not a replacement: all built-in identifiers
remain selectable. The launcher requires an existing absolute, non-symlink
file. Before device preflight, the plan validates its schema, finite points,
positive path height, nonzero consecutive segments, one-stroke declaration,
and uniqueness against both the external file and built-ins. A custom id cannot
shadow a built-in id. The catalog path, its SHA-256, and the combined built-in
plus external hash are retained in collection provenance.

Catalogs are immutable for one launcher process. To add or revise a trajectory,
stop while motion-inert, edit the external file, inspect with `--dry-run`, and
start a new process. Never replace the file while an Episode is running. This
keeps the generated path fixed for the complete collection attempt while making
future additions data-only changes.

## Offline command inspection

### Cross-dataset source pool

The recommended mode uses a JSON manifest. Candidate paths may be absolute or
relative to the manifest directory. A candidate can expose one or more writing
subphases:

```json
{
  "schema": "dexteleop.rule_based_writing_source_pool.v1",
  "pool_id": "writing-diversity-2026-09",
  "candidates": [
    {
      "candidate_id": "day07-episode101",
      "dataset_root": "../datasets/day07",
      "episode_index": 101,
      "detail_annotation_dir": "../datasets/day07/meta/progress/detail_stage_annotations/episode-uuid/detail-set",
      "phases": ["pen_down", "write", "pen_up"],
      "tags": ["front", "digit_1"]
    }
  ]
}
```

`--source-pool-range START:END` restricts selection to a half-open slice of the
manifest candidate list; omitting it uses the full list. For each of
`pen_down`, `write`, and `pen_up`, the launcher selects an eligible
candidate deterministically from the range using the manifest semantic hash
and `--selection-seed`. The dedicated UI contains three independent
selectors for every phase-eligible candidate, all nine fixed digit paths, all
24 fixed one-stroke Greek paths, all 26 uppercase English paths, every admitted
external path, all four directions, glyph height, writing speed, and an
initialization-speed multiplier. The optional prelude selector is accompanied
by an integer field for the number of ordinary successful Episodes between
prelude runs. Startup arguments provide the
initial values and the dedicated UI may rebuild them only while the runtime is
motion-inert. The
pre-motion form also contains the runtime-safe writer controls:
the native tactile-bundle root and minimum-free-space reserve, tactile
compression, writer/process/storage queue sizing, writer thread count,
collection-frequency warning threshold, and Parquet/video enablement.
The per-camera queue admission timeout is also configurable; when a blocking
route exceeds it, the affected recorder is quarantined instead of stalling the
control path indefinitely.
Applying a dedicated-UI change rebuilds
and validates the immutable plan on a background worker and keeps start
disabled until validation succeeds. In both paths, the exact pool hash, range,
candidate IDs, datasets, episodes, detail sets, tags, and pose contracts are
recorded in every rule command and admitted formal sample.

The dedicated UI can build and prevalidate an ordered batch without opening
hardware resources. Each queue item remains one independent formal Episode and
never starts automatically.

### Optional step-gated task chain

Rule-writing mode exposes an additional lightweight task chain in the same UI:

1. release the enabled gripper and wait for measured-success feedback;
2. close it and wait for measured-success feedback;
3. capture the current right-TCP `canonical_desk` writing height;
4. randomly choose one entry from the operator-prevalidated configuration
   batch, activate it through the existing configuration validator, then run
   its existing pen-down replay, generated writing, and relative pen-up replay
   as one automatic formal collection;
5. only after the existing writer has fully committed and closed that formal
   Episode, release the gripper and wait for measured-success feedback.

The operator advances one externally visible step with the chain button or
`Alt+N`. A pending gripper action, height capture, configuration validation, or
formal collection cannot be advanced a second time. The three motion parts in
step 4 remain automatic because they are one existing rule plan; the chain
does not add a replay cursor, motion loop, writer, or safety authority.

Random selection changes only which already validated batch item is activated.
It never adds a random TCP, joint, frame, or height offset. While a chain is
active, its manual start/gripper/configuration/height controls are locked to
prevent replacement races; Stop/Hold, device recovery, runtime Home, debug,
and quit remain on their existing paths. Any missing success receipt, Hold,
abort, validation error, or writer failure stops at the current gate and never
skips forward. After the fifth measured-success receipt, the chain reports one
completed cycle; the next press starts another cycle.

From the selected AgenticRobot checkout in the complete `dexteleop310`
environment:

```bash
bash scripts/run_rule_based_writing_collection.sh \
  --source-pool-manifest /absolute/path/to/writing_source_pool.json \
  --source-pool-range 0:200 \
  --selection-seed 42 \
  --pre-pen-down-replay-candidate-id day07-episode101 \
  --pre-pen-down-replay-normal-episodes 5 \
  --dataset-root /home/user/zc/datasets/RealmanDMTAC/FormalCollection \
  --tactile-bundle-root /absolute/path/to/existing/tactile_root \
  --trajectory digit_1_v1 \
  --size-m 0.08 \
  --direction up \
  --initial-alignment-speed-scale 0.5 \
  --dry-run
```

Only selected candidates are opened and contract-validated at application
startup; all manifest candidate paths are resolved and checked for existence.
Missing phase coverage, an empty/out-of-bounds range, an invalid detail
binding, or any selected pose-frame mismatch fails before device opening.

### Legacy single source

The original three arguments remain supported and select all three subphases
from one episode:

```bash
bash scripts/run_rule_based_writing_collection.sh \
  --source-dataset /absolute/path/to/source_dataset \
  --source-episode 0 \
  --detail-annotation-dir /absolute/path/to/detail_annotation_set \
  --dataset-root /home/user/zc/datasets/RealmanDMTAC/FormalCollection \
  --tactile-bundle-root /absolute/path/to/existing/tactile_root \
  --trajectory digit_1_v1 \
  --size-m 0.08 \
  --direction up \
  --dry-run
```

`--dry-run` validates only shell-level paths and prints the resolved command;
it does not import the source episode and does not prove hardware readiness.

## Real collection handoff (blocked in this candidate)

Do not remove `--dry-run` on this reconciliation candidate. Real collection
remains blocked until the generic entrypoint/runtime and component-builder
owners integrate the staged hooks and rerun hardware commissioning. After that
owner integration, no Q6 or foot-pad process is required. The launcher explicitly selects manual
special-arm control, so an already-running foot-pad virtual keyboard is not
opened or polled by this application. The intended integrated path captures
robot-state-only runtime Home, waits for the dedicated UI readiness, and starts
exactly one formal Episode through the rule-start intent. The plan arms only
after that Episode's writer and causal dual-arm feedback are ready.

If a required backend is recovered through the established device-retry UI, the retry path
refreshes both the generic hardware projection and the rule runtime's cached
readiness diagnosis. Recovery does not authorize motion or silently perform
Home: after the device blockers disappear, explicitly click “重新安全初始化” and
wait for the shared runtime Home to complete before starting a new formal
Episode.

The plan does not output motion until both the writer and causal dual-arm
feedback are ready. The established stop/Hold action immediately blocks
further rule commands, cancels and
discards the whole current Episode (including already accepted samples), and
enters hardware Hold. The UI then exposes only the safe recovery path: wait
for Episode cleanup, run “重新安全初始化”, and start a new attempt after the
fresh runtime Home and arm-admission receipt. External/safety Hold follows the
same rule-attempt reset. Plan completion is different: it requests writer
finalization, keeps the last certified position target while the writer drains,
then releases the rule-only FixArm authority and returns to a command-inert
ready state. It does not manufacture a hardware Hold or recapture runtime Home
after a successful Episode.

If writer preparation or Episode opening fails (for example, the configured
minimum tactile free-space reserve exceeds the mounted disk's current free
space), rule motion stays command-inert while the camera preview,
safety/control loop, warning messages, and stop/Hold remain available. In the
dedicated UI, correct and revalidate the motion-inert configuration before
retrying. The reserve is never reduced automatically.

The default fixed-path speed is derived from the right TCP average over the
selected source `write` span. Use
`--speed-m-s VALUE` only when an explicit reviewed speed is required. Supported
directions are `up`, `right`, `down`, and `left`; `--size-m` is the glyph height
before that rotation. `--initial-alignment-speed-scale VALUE` supplies the
startup value for the current-to-`pen_down` transition. The collision
planner first certifies the same bounded direct path at full scale. The shared
offline waypoint implementation then applies a C2 quintic time parameterization
at the established 60 Hz position-reference rate, with zero endpoint velocity
and acceleration and continuous velocity/acceleration/jerk certificates.
The configured value in `[0.05, 1.0]` scales only the certified execution duration;
it no longer changes path geometry or creates a sparse staircase. The values
become immutable once rule motion begins. The dedicated UI may rebuild them
while an armed rule is waiting in its explicit writer-blocked state.

By default the launcher uses the right-gripper rule-writing progress and stage
profiles. With `--with-left-gripper`, it selects
`catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress` together
with
`catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed`.
Within the `right_write` task the admitted samples carry exactly these three
ordered detail stages: `right_write.pen_down` (`replay`),
`right_write.write` (`rule-base`), and `right_write.pen_up`
(`relative_replay`). Initialization is labeled but excluded from formal sample
counts. Each rule directive is admitted to the writer at most once, so the
published count cannot grow through repeated control-loop dispatch of one held target. A
normal operator or safety Hold discards the complete attempt; only the runtime's
`rule_based_complete` event can seal this detailed event window as SUCCESS.
