# RealTeleop Control Station boundary

This package owns RealTeleop presentation policy, operator-intent mapping,
view models, and application workflow coordination. It must not open or poll
devices, sample Home, run calibration, own robot safety/control, or write an
episode. Those operations remain behind typed `teleop_runtime` and
`collection_runtime` ports.

During migration, the legacy RealTeleop entrypoint and compatibility module
may import this package. Boundary tests enumerate those exceptions exactly;
new reverse imports are forbidden.

The shared browser transport and rebuildable presentation cache live in
`control_station.web.calibration_web_ui`. RealTeleop-specific control
availability and input precedence remain in this feature package; they do not
poll the shared web queue themselves.

The optional progress feature follows the same boundary:

- `progress.py` owns typed Advance/Undo/Failure intents and rebuildable control
  availability;
- `progress_workflow.py` coordinates lower-owned ports through structural
  protocols only. It does not import `collection_runtime`, `teleop_runtime`,
  robot/session implementations, or a dataset writer;
- the workflow may request Hold/Pause after a durable Failure, but it writes no
  receipt itself until RealTeleop reports that the authoritative Session Hold
  and formal recorder Pause both took effect;
- an uninterrupted ordinary operator Hold may be promoted to Failure only when
  RealTeleop supplies its exact same-Episode Hold/Pause effects. The workflow
  preserves that original pause cut, records the later classification as
  availability time, and never manufactures or replays a second Hold;
- `control_station.web.real_teleop_progress.js` is an independent, optional
  presentation plugin over the existing RealTeleop page. It renders only
  rebuildable progress state and emits typed UI events; it owns no progress
  fact, Episode lifecycle, Hold, recorder, or robot operation;
- RealTeleop constructs progress semantics only with an explicit progress
  profile, and renders the plugin only with the separate
  `--collection-progress-gui` opt-in. Omitting the GUI flag preserves the
  existing page and A/B/C semantics; omitting both flags also restores the
  no-progress data path.

The first real-robot gate remains LINEAR-only. The selector is intentionally
capable of displaying explicit next-step choices, but RealTeleop must continue
to reject rooted-tree profiles until the branch workflow has its own gate.

## Default-off exclusive HITL collection

The canonical RealTeleop entrypoint has an additive, default-off exclusive
Human/Policy profile behind `--hitl-policy-control`. It does not enable two
sources at once: every control cut selects exactly one complete source action.
The selected Policy action is identity-composed through the backend-neutral
`control_composition` contract; the existing Human candidate is preserved
directly. Both then follow the existing Session command filter, guard,
admission, Rig dispatch, and recorder path.

Normal policy inference reuses the existing OpenPI artifact/client and the
formal collector's causal arm/hand pollers and camera buffers. It performs no
extra synchronous hardware read. A policy queue gap emits explicit no-command
and never falls back to the sampled Human command. Normal mode therefore
requires formal collection, the resolved pose-frame contract, and an existing
prepared policy artifact:

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui \
  --hitl-policy-control \
  --hitl-policy-mode normal \
  --hitl-policy-artifact-json /path/to/policy_artifact.json
```

Before a deployment session, the dedicated launcher can perform the same
policy-artifact and HITL dataset-contract admission as a read-only offline
check. It requires an explicit dataset base root, task name, and task id so a
machine-local default cannot silently select the capture target:

```bash
python -m teleop_runtime.real_teleop.hitl_launch -- \
  --gui \
  --hitl-policy-control \
  --hitl-policy-mode normal \
  --hitl-policy-artifact-json /path/to/policy_artifact.json \
  --dataset-root /path/to/formal-datasets \
  --collection-task-name "Write Words" \
  --collection-task-id write_words_v1
```

The default command only reads local contracts, the artifact sidecar, and
existing dataset metadata. Its JSON report records the resolved target,
immutable HITL contract, policy preflight, and every resource class left
unopened. It does not contact the policy endpoint, open cameras or teleop
devices, construct a writer, or command a robot. After reviewing that report,
normal mode may be handed to the canonical RealTeleop entrypoint with the
separate launcher option `--execute` placed before the `--` separator:

```bash
python -m teleop_runtime.real_teleop.hitl_launch --execute -- \
  --gui \
  --hitl-policy-control \
  --hitl-policy-mode normal \
  --hitl-policy-artifact-json /path/to/policy_artifact.json \
  --dataset-root /path/to/formal-datasets \
  --collection-task-name "Write Words" \
  --collection-task-id write_words_v1
```

`--execute` is a real-process handoff, not a dry run: the canonical entrypoint
retains all existing device, safety, calibration, operator-supervision, and
motion gates. The offline report does not establish policy-service
reachability, hardware readiness, safe motion, or downstream dataset
publication readiness.

The terminal keys are additive: `u` requests Human takeover and `i` requests
Policy return. They may be changed with `--hitl-takeover-key` and
`--hitl-policy-resume-key`; established RealTeleop keys cannot be reused.

- Policy to Human pauses the open capture segment, requests the ordinary lower
  Hold, performs the existing guarded runtime Home flow, selects Human, and
  completes only after the first Human command is safely dispatched.
- Human to Policy pauses capture, establishes lower Hold, resets the policy
  source, waits for a fresh post-reset `PolicyActionEmission`, selects Policy,
  and completes only after the first Policy command is safely dispatched.
- Capture resumes through the existing consecutive-stable-frame rule, so rows
  never bridge the Hold/Home/reset transition window.
- A source error or failed transition keeps the Session held and capture
  paused; it never restores the other source implicitly.

RTC uses the same committed `PolicyActionEmission` consumer seam and bounded
application inbox, but the repository's current RTC application profiles are
still explicitly no-action. Selecting RTC Policy therefore validates the
post-reset committed emission and then fails held; it cannot be upgraded to a
real motion command by this profile. This preserves the current RTC landing
contract: producer commit is not downstream target/safety/reference/dispatch
authorization. The RTC inbox is an application binding, not a second policy
worker or a new CLI transport owner.

RTC may use the launcher for dataset-contract inspection by replacing
`--hitl-policy-mode normal` with `rtc` and omitting the artifact option. The
report remains `rtc_no_action`; `--execute` is rejected because no external
committed-emission producer or RTC Active authorization is bound at this
entrypoint.

## Attach-only Agent observation

The top-level
`agent_gateway.profiles.real.observation_gateway` module defines the default-
off real `state.read_tcp_pose` Route. The historic
`control_station.real_teleop.agent_observation_gateway` path is a compatibility
re-export only. The Route is `real_observe`, `t0_read_only` and cannot
start/connect a device, refresh hardware, dispatch a command, request owner
HOLD, rearm, or home.

The Gateway core is embodiment-neutral. It imports no RM65 or other robot
backend and contains no fixed arm/component name. A deployment supplies an
exact `embodiment_instance_id`, `target_binding_id`, component set, reference
frame, frame graph, tool configuration, calibration, observation/source
bindings, and owner epochs. The stable `state.read_tcp_pose@1` definition is
reused across embodiments, while each concrete `DeploymentBinding` remains
different and content-addressed. This exact instance binding is required for
safe interpretation of a real pose; it is not an upper-layer dependency on a
robot model.

The existing real control owner must publish only after it has already captured
both its current registry state and a measured pose cut. The reusable
`RealTcpPoseOwnerPublicationPort` owns the monotonically increasing publication
revision and validates exact runtime/domain/embodiment/target, execution
manifest, backend inventory, owner token/generation, authority/execution epoch,
clock, source/observation binding, pose schema, and component coverage before
writing `RealTcpPoseOwnerPublicationBuffer`. The Gateway cannot supply or
increment any of those owner facts. Gateway reads only copy that immutable cut.
A missing, stale, invalid, identity-mismatched, owner-changing-without-new-
generation, or revision/epoch-regressing publication fails closed. Both port
and buffer are no-I/O: neither calls a backend or changes the lower gate.
Gateway close uses a local no-I/O receipt that explicitly proves neither motion
nor HOLD; it is not a robot safety receipt.

The top-level real Gateway profile now provides the contract, owner-publication
buffer, passive executor, Gateway service, and Unix server assembly. It
deliberately does not register a CLI. Policy-deploy now exposes a separate
programmatic default-off
`legacy_worker_binding`: one `LegacyControlDomainExecutionContext` owns its
dispatch token/epochs plus lifecycle, execution gate, and owner-state revision,
while `run_policy` submits typed transitions. The no-I/O publication port can
structurally project that exact context snapshot and refuses a missing or
identity-mismatched fact; it does not import the concrete deployment runtime.
Context cancellation advances both epoch registries and sets the gate to
`transitioning` under one lock; snapshots fail closed while a direct legacy
cancellation is between those registry linearization points. A separate pure
RM65 convergence contract requires the exact three-worker drain aggregate plus
dual-backend HOLD receipt before publishing HOLD; incomplete evidence faults.
That contract consumes evidence only and performs no worker or backend I/O.
An owner snapshot is now stable for the lifetime of one `owner_state_revision`:
its `captured_at` is the revision-effective cut, so passive re-reads cannot
silently change the owner hash. The official hardware idle-worker stop-only
preflight can compile production-scoped HOLD evidence only from exact left and
right permit-accepted `Arm_Slow_Stop` receipts whose structured SDK result code
is zero and whose evidence event is the current cancellation. This can converge
that stop-only context to HOLD, but its authorization still forbids every motion
target and lifecycle action.
The same context signs content-addressed `reference_segment` execution work only
while `RUNNING/READY`. Policy-deploy carries it through startup, ordinary
policy, and lookahead submissions; the reference worker derives
`position_command` work and the selected RM65 backend worker derives
`backend_command` work. One no-I/O fake-SDK test proves the exact owner epoch and
generation reach the backend pre-SDK guard. Missing or stale work fails closed.
The policy-deploy binding is still `fake_only_unbound`, rejects hardware, and
has only a one-step dry-run plus pure in-memory publication proof. Its
policy-motion worker HOLD/drain proof and a production real launcher remain
unbound. Zero, constants, or Gateway-generated epochs must not be used to make
the attachment appear authoritative.

## Default-off real Agent primitive motion

The top-level real profile now has a separate default-off 6+3 primitive path.
`agent_gateway.profiles.real.motion_factory` preflights four exact
`t2_real_motion` Routes (six cardinal directions plus three fixed level
orientations), prepares one `RealAgentMotionPort`, assembles the Gateway
service, and only then attaches that same port to the existing RealTeleop
Session. It returns a replacement `RealTeleopComponents`; a composition root
must use that returned aggregate. The canonical GUI entrypoint exposes this
only behind `--agent-primitive-skill-only` plus an exact side, action, private
Unix socket, Active TCP envelope, position-control mode, and explicit operator
confirmation. Ordinary GUI invocations remain unchanged and do not attach or
advertise any Agent motion Route.

The existing RealTeleop control thread remains the sole effect owner. It applies
queued activation/HOLD requests, builds the selected absolute tool-TCP target
through the existing coupled retargeter, exact-holds the unauthorized arm at
the final admitted-q boundary, rechecks the resulting coupled collision state,
and sends the result through the
Session's existing command filter, pre-dispatch guard, admission, Rig dispatch,
recorder, and `hold_hardware()` paths. After the normal hardware-state reads,
the same control frame may publish those already-read joint values through the
existing FK owner as the measured TCP cut; the Gateway performs no extra
backend read.

In Skill-only mode the Agent port is the Session's sole command override. An
idle, completed, disconnected, or held Gateway therefore emits an explicit
no-command override and cannot fall back to a Vive arm command. The Vive source
may still be present for the existing GUI/runtime-Home and health lifecycle; it
does not retain competing motion authority.

The Gateway side requires an exact RUNNING/READY lower-owner cut published by
that same RealTeleop control frame, fresh
measured cuts, a route-covering lower safety-envelope reference, and a one-use
request-bound operator Grant. The local Unix adapter mints that grant atomically
only for the one side/action pre-authorized when the process started; the client
never receives reusable grant material. It evaluates convergence only from distinct
lower cuts after dispatch and declares success only after lower-confirmed HOLD.
Disconnect, timeout, cancellation, runtime exit, and normal completion converge
through the same lower HOLD seam before Session/backend teardown. The current
tests drive all nine cases with an in-memory owner/control loop and additionally
cover exclusive idle behavior, fresh owner cuts, Unix one-use grant binding,
server teardown, and the strict relative-envelope files. They establish wiring
and fail-HOLD behavior, not safe physical motion; the first real run remains an
operator-supervised 3 cm single-arm test.

The canonical shared profiles live at
`teleop_runtime/configs/skill_deploy/agent_primitive_left_relative_tight_v1.json`
and
`teleop_runtime/configs/skill_deploy/agent_primitive_right_relative_tight_v1.json`.
The files under this package's `configs/` directory are compatibility mirrors
checked for semantic equality by tests. Each profile binds the runtime
Home TCP as the anchor, allows ±10 cm in canonical desk axes, and limits total
rotation from that anchor to 45 degrees. The runtime brings both arms online so
the coupled collision and final pre-IK safety owners remain authoritative, but
the selected profile and one-use grant authorize only one arm to move.
