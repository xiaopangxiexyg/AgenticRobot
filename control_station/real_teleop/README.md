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
The current binding is still `fake_only_unbound`, rejects hardware, and has only
a one-step dry-run plus pure in-memory publication proof. RealTeleop, measured
pose production, Gateway daemon lifetime, policy-motion worker HOLD/drain proof,
and a production real launcher remain unbound. Zero, constants, or Gateway-
generated epochs must not be used to make the attachment appear authoritative.
