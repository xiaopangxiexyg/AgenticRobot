# Quest paired-pose application binding

## Scope

This application leaf connects three existing owners without moving their
responsibilities:

1. `hardware.quest_dev` owns the unique HTTP/WebSocket host and admitted
   controller-pose runtime.
2. policy deploy owns the single RM65 `ArmStateReader` and passive pose journal.
3. `ManualStaticPairedPoseCapture` remains the pure lower reducer for typed
   static-window evidence.

The binding lives in
`teleop_runtime.deploy.quest_paired_pose_capture`. It never calls an arm
backend, `ArmStateReader.snapshot()`, `get_state()`, or `refresh()`. The factory
receives only a narrow source containing the exact receiver-clock descriptor,
the same monotonic callable, explicit side-to-coordinate bindings, and passive
`atomic_cut`/`events_after` functions.

The capture binding itself never persists or fits evidence automatically. The
separate `teleop_runtime.deploy.quest_paired_pose_evidence_store` application
leaf provides explicit candidate-evidence publication/load, an offline
diagnostic-fit wrapper, and explicit candidate diagnostic-report
publication/load. Neither module accepts calibration, installs a frame
transform, publishes a policy trajectory, or renders that trajectory on Quest.
They also add no Quest CLI default and make no controller, handedness,
robot-side, frame, or static-window-profile choice.

## Clock and cursor proof

The application supplies the same callable to:

- the HTTP server stamp that becomes an accepted Quest sample's
  `received_monotonic_ns`;
- controller-runtime invalidations;
- the existing RM65 pose journal.

It also injects the full receiver descriptor's `clock_domain_id` into a copied
controller feature config. A conflicting configured ID fails before the
runtime starts. The default standalone Quest CLI keeps its existing behavior.

Robot journal cuts are inclusive, so the reducer binding/window boundary uses
`robot_cut.event_sequence + 1`. Quest subscription, event, and observer-close
cuts already contain an exclusive `next_event_sequence` and are used directly.
Both journals advance globally before the reducer filters the selected robot
side or Quest controller.

Candidate discovery and capture use two consecutive Quest subscriptions. The
discovery subscription projects globally ordered, pose-redacted identities and
then discards pose events that are no longer needed for ordering. An explicit
selection atomically closes that observer, opens a fresh capture observer, and
requires the reviewed XR epoch to remain active with non-regressing event and
sample cursors. The application also takes a fresh robot cut at selection and
rejects a drifted or terminal-invalidated selected stream. Thus operator review
time is not hidden inside either the Quest pose-history bound or the robot
journal retention window; `max_event_history` still bounds actual reordering
and the selected static-window capture.

## Programmatic lifecycle

There is deliberately no production selection default. A coordinator can
construct an explicit factory and inject it into `run_policy.main()`:

```python
from pathlib import Path
from threading import Thread

from hardware.quest_dev.features.visualization.paired_pose_capture import (
    StaticWindowCaptureProfile,
)
from teleop_runtime.deploy.quest_paired_pose_capture import (
    QuestCaptureFrameFacts,
    QuestCaptureSelectionIntent,
    QuestPairedPoseCaptureFactory,
    QuestPairedPoseCaptureLaunchConfig,
)
from teleop_runtime.deploy.run_policy import main as run_policy_main


profile = StaticWindowCaptureProfile(
    profile_id="operator-reviewed-profile-id",
    min_window_duration_ns=...,
    max_window_duration_ns=...,
    min_source_span_ns=...,
    max_pair_skew_ns=...,
    min_robot_samples=...,
    max_robot_samples=...,
    min_quest_samples=...,
    max_quest_samples=...,
    max_robot_translation_excursion_m=...,
    max_robot_rotation_excursion_deg=...,
    max_quest_translation_excursion_m=...,
    max_quest_rotation_excursion_deg=...,
    max_quest_reorder_events=...,
    max_event_history=...,
    max_completed_windows=...,
)
factory = QuestPairedPoseCaptureFactory(
    QuestPairedPoseCaptureLaunchConfig(
        config_path=Path("hardware/quest_dev/config.yaml"),
        force_http=False,
        profile=profile,
        capture_id="operator-issued-capture-id",
        startup_timeout_s=...,
        callback_drain_timeout_s=...,
        host_join_timeout_s=...,
    )
)

deploy_thread = Thread(
    target=run_policy_main,
    kwargs={
        "argv": [
            "--rm65-pose-observation",
            "--active-sides",
            "right",
            "--right-arm-ip",
            "<reviewed-rm65-ip>",
        ],
        "quest_paired_pose_capture_factory": factory,
    },
)
deploy_thread.start()

application = factory.wait_for_application(timeout_s=...)
launch = factory.launch_info
assert launch is not None
print(launch.base_url)
print(launch.pairing_code)
```

`run_policy.main()` accepts this concrete lifecycle factory, not a marker-only
callable. It verifies that the returned concrete application is the factory's
own application and is bound to the exact passive journal passed by deploy.
After injection, policy deploy is the shutdown caller: its `finally` calls the
idempotent `factory.close()` before invalidating the robot journal or closing
backends. The coordinator should keep the factory reference until the deploy
thread has exited and must not independently race that normal shutdown.

Startup and cleanup failures are explicit. A
`QuestPairedPoseCaptureStartupError` whose message says cleanup remains
unproven means the factory still owns a host or non-daemon serving thread; keep
the object while `factory.cleanup_pending` is true and retry `factory.close()`
after the underlying shutdown condition changes.
`QuestPairedPoseCaptureCleanupError` likewise never marks the factory or
application closed without a proven observer removal, dead serving thread, and
successful host close. A permanent host cleanup failure remains latched and
cannot be converted into success by a second `close()` call.

This is a real-hardware observation profile even though action sending is
forbidden by `--rm65-pose-observation`. Use the repository's normal hardware
safety and remote-wrapper requirements before running it. The examples above
leave every threshold, endpoint, side, and timeout explicit on purpose.

After the Quest page connects, candidates are pose-redacted and do not contain
controller matrices:

```python
candidates = application.candidates()
reviewed_candidate_hash = "<operator-reviewed-candidate-hash>"
candidate = next(
    value
    for value in candidates
    if value.candidate_hash == reviewed_candidate_hash
)

application.select(
    intent=QuestCaptureSelectionIntent(
        candidate_hash=candidate.candidate_hash,
        robot_side="<explicit-left-or-right>",
    ),
    frame_facts=QuestCaptureFrameFacts(
        candidate_hash=candidate.candidate_hash,
        reference_frame_id="<explicit-quest-reference-frame-id>",
        moving_frame_id="<explicit-controller-grip-frame-id>",
        xr_frame_clock_domain_id="<explicit-webxr-frame-clock-id>",
    ),
)
```

The coordinator then declares each stationary interval mechanically:

```python
application.begin_window(window_id="stationary-window-001")

# The operator keeps the rigidly mounted controller and robot TCP stationary.
# The policy loop calls application.poll() once after its existing arm snapshot.

receipt = application.complete_window()
while receipt is None:
    # Retry drains the first immutable end tuple; it never enlarges the window.
    receipt = application.complete_window()
```

Once all intended windows are accepted, sealing is a separate explicit action:

```python
evidence = application.request_seal()
```

## Control Station operator lifecycle

`control_station.quest_calibration` provides the typed, programmatic operator
boundary around this exact application.  It does not start the Quest host,
refresh the robot, write evidence, fit or accept calibration, install a frame,
or render geometry.  The application composition root must pass the already
owned `application` instance to the controller:

```python
from control_station.quest_calibration import (
    QuestCalibrationIntent,
    QuestCalibrationOperatorController,
)

controller = QuestCalibrationOperatorController(application)
status = controller.status()

selection = controller.submit(
    QuestCalibrationIntent(
        client_id="calibration-ui",
        client_sequence=1,
        expected_revision=status["revision"],
        intent_kind="select_source",
        candidate_hash="<operator-reviewed-candidate-hash>",
        robot_side="<explicit-left-or-right>",
        reference_frame_id="<explicit-quest-reference-frame-id>",
        moving_frame_id="<explicit-controller-grip-frame-id>",
        xr_frame_clock_domain_id="<explicit-webxr-frame-clock-id>",
    )
)
assert selection.accepted
```

Each later non-cancel intent must use a strictly increasing client sequence and
the latest returned/status revision.  `begin_window` carries only `window_id`;
`complete_window` is retried with a new sequence/revision while its outcome is
`window_drain_pending`; `seal` carries no pose data.  `cancel` closes the lower
application and may bypass a stale revision so cleanup is still reachable
during an in-progress action.  Status contains candidate identity hashes,
handedness, reference-space metadata, counts, lifecycle flags, and action
availability, but no matrices, pose samples, raw events, evidence dataset, or
fit.

The local operator Web coordinator binds that controller to the unique
application retained by the same factory. Start the coordinator before or
after the deploy thread; it waits through the factory's public application
port and does not create a second Quest host, robot reader, capture reducer, or
application:

```python
from control_station.quest_calibration import QuestCalibrationWebCoordinator

coordinator = QuestCalibrationWebCoordinator(
    factory,
    host="127.0.0.1",
    port=0,
    open_browser=True,
)
coordinator.start()
try:
    deploy_thread.start()
    print(coordinator.url)
    deploy_thread.join()
finally:
    coordinator.close()
```

The page polls a pose-redacted status endpoint and sends only the five typed
operator intents above. It uses same-origin local HTTP and has no pairing-code
step. Binding to a LAN address remains an explicit caller choice; loopback is
the default. `coordinator.close()` stops only its HTTP and attachment threads.
It never closes `factory` or `application`; policy deploy remains the normal
shutdown owner, while an explicit operator `cancel` intent still calls the
lower application's bounded close path through the typed controller.

This is a programmatic composition seam, not a new production CLI. Exercising
it with the observation profile still requires the repository's real-hardware
safety procedure and explicit reviewed endpoints. The Web coordinator itself
does not establish real Quest connectivity, RM65 observation readiness,
calibration quality, acceptance, frame installation, or trajectory rendering.

## One-command operator trial

The reviewed Control Station composition is available through a thin script.
Always inspect the resolved no-action contract first; this does not open a
socket or construct a device backend:

```bash
bash scripts/run_quest_calibration.sh \
  --side right \
  --arm-ip <reviewed-rm65-ip> \
  --trusted-lan \
  --dry-run
```

For the live observation trial, remove only `--dry-run`:

```bash
bash scripts/run_quest_calibration.sh \
  --side right \
  --arm-ip <reviewed-rm65-ip> \
  --trusted-lan
```

`--trusted-lan` keeps protocol authentication enabled but selects the easy
fixed six-digit code `111111`. It is only appropriate on an explicitly trusted
isolated LAN. Omit it for a new random code, or use `--pairing-code NNNNNN` for
another explicit six-digit value. The desktop operator page opens
automatically. Once the lower application is ready, the terminal prints a
large Quest block containing the base URL, six-digit code, and one-step URL.
The normal Quest procedure is:

1. Reopen the previously entered/bookmarked `https://<LAN-IP>:8444/` address.
2. Accept the local development certificate warning if the browser asks.
3. Enter `111111`, start MR/VR, and keep the rigidly mounted controller still.
4. In the desktop page, select the candidate and arm side, then start and
   complete a stationary window. Completion drains asynchronously and advances
   the next window ID automatically.
5. Return to the terminal and press `Ctrl+C` for full ordered shutdown.

The builtin `operator-trial-v1` profile requires at least one second per
window, at least five robot samples, at least ten Quest samples, and explicit
bounded stationarity excursions. It is a candidate-data admission profile, not
a calibration acceptance policy. Real calibration diversity still calls for
at least 15 poses spanning the workspace and rotations around two or more
non-parallel axes, but the launcher does not move the arm; those poses require
a separately authorized motion procedure.

The script deliberately accepts no arbitrary policy-deploy passthrough. It
fixes `--policy hold`, enables only `--rm65-pose-observation`, selects one arm,
sets end-effectors and cameras to none, and never supplies `--send-actions`.
This prevents the launcher itself from sending trajectories. It does not prove
the ordinary RM65 backend startup is zero-write: the current backend may still
perform its pre-existing tool, singularity, or worker configuration. Use the
real-hardware safety procedure before removing `--dry-run`.

Clicking `seal` constructs candidate evidence only in the running application's
memory. This launcher does not publish artifacts, run the diagnostic fit,
accept a calibration, or install a frame. Exit only after recording any status
or hash needed for the trial; durable evidence publication remains a separate
future operator action.

The equivalent Python entrypoint is:

```bash
python -m control_station.quest_calibration.launch \
  --side right \
  --arm-ip <reviewed-rm65-ip> \
  --trusted-lan
```

Persistence is another explicit action after a successful seal. Constructing
the layout, creating the application, and ordinary `close()` do not create an
evidence directory or file:

```python
from runtime_artifact_paths import RuntimeArtifactLayout
from teleop_runtime.deploy.quest_paired_pose_evidence_store import (
    fit_quest_paired_pose_evidence,
    load_quest_paired_pose_diagnostic_fit,
    load_sealed_quest_paired_pose_evidence,
    publish_quest_paired_pose_diagnostic_fit,
    publish_sealed_quest_paired_pose_evidence,
)

layout = RuntimeArtifactLayout.configured()
commit = publish_sealed_quest_paired_pose_evidence(
    layout=layout,
    evidence=evidence,
)
sealed = load_sealed_quest_paired_pose_evidence(
    layout=layout,
    evidence_semantic_content_hash=commit.evidence_semantic_content_hash,
)
diagnostic = fit_quest_paired_pose_evidence(sealed)
diagnostic_commit = publish_quest_paired_pose_diagnostic_fit(
    layout=layout,
    diagnostic_fit=diagnostic,
)
persisted_diagnostic = load_quest_paired_pose_diagnostic_fit(
    layout=layout,
    diagnostic_fit_semantic_content_hash=(
        diagnostic_commit.diagnostic_fit_semantic_content_hash
    ),
)
```

The immutable canonical files are content-addressed relative to `layout.root`;
with the default root they are:

```text
runtime_artifacts/evidence/quest_paired_pose_capture/
  <capture-evidence-semantic-content-hash>.cjson

runtime_artifacts/derived/quest_paired_pose_diagnostic_fit/
  <diagnostic-report-semantic-content-hash>/report.cjson
```

`commit.path` and `diagnostic_commit.path` are the authoritative resolved paths
when a local configuration overrides the runtime-artifact root.
`layout.root` must remain isolated to the current artifact writer/worktree; do
not point mutable artifacts at an automatically shared symlink. Any later
rebind or migration must follow `docs/setup/LOCAL_RESOURCES.md`, including
stopping the writer and inventory/copy/hash verification.

Publication strictly replays the evidence before writing and again after a
canonical readback. Identical retries are idempotent; an existing different,
truncated, non-canonical, wrong-domain, symlinked, or hash-misnamed target fails
closed and is never replaced. The commit receipt keeps three distinct
digests: the outer capture-evidence hash, the derived R0 dataset SHA-256, and
the canonical file-byte SHA-256. They have different coverage and must not be
substituted for one another.

The diagnostic wrapper strictly reparses the full evidence and calls the
existing feature-local AX=XB solver. Its report binds both the outer evidence
hash and dataset hash and remains `candidate_only=true` with
`calibration_accepted=null`; its nested fit independently remains
`accepted=null`.  Publishing that report is a second explicit action using a
separate file domain and the outer report hash as the derived artifact id.
Strict reload verifies the outer and nested hashes, redundant transform
representations, residual summaries, observability fields, and candidate-only
sentinels without rerunning the solver.  It does not create an acceptance
receipt, install a frame, or require the referenced capture file to be present;
the evidence hash is a content reference, not proof that its source artifact is
available.  Any later acceptance workflow must load and validate both.
These checks establish structural and internal numerical consistency only;
reload neither recomputes the fit nor validates the reported transform,
residuals, or observability metrics against the referenced capture.

The capture, dataset, nested fit, outer report, and canonical file-byte hashes
have different coverage and must not be substituted for one another. The outer
report and persisted report file additionally use distinct explicit semantic
and file domains; that statement does not assign a domain envelope to every
digest. Solver observability gates still apply, including the minimum of three
poses; real capture should follow the calibration document's stronger diversity
guidance.

For the capture observer, `request_seal()` uses exactly one
`unsubscribe_with_cut()` and one typed robot cut, then boundedly drains
callbacks below the close watermark. A timeout, gap, selected-source
invalidation, clock drift, or conflicting duplicate leaves evidence absent.
Ordinary `close()` only unsubscribes and tears down resources; it never turns
completed or partial windows into sealed evidence automatically. A terminal
`poll()` raises a typed terminal error. Policy deploy disables further passive
polls immediately but defers potentially blocking host teardown to its ordered
`finally`, after the current snapshot reaches the normal IK/safety gate.

## Runtime order

The concrete caller order is:

```text
resource lifecycle start
  -> RM65 pose observation assembly
  -> single ArmStateReader
  -> narrow passive journal source
  -> unique Quest host + discovery subscription
  -> explicit candidate/side/frame selection
       -> fresh capture subscription cut + fresh robot journal cut
  -> each existing ArmStateReader.snapshot()
       -> paired application poll()
  -> factory close -> paired application unsubscribe + host teardown
  -> robot journal terminal invalidation
  -> backend/resource close
```

Quest callbacks only enter a bounded staging area. They never invoke the
reducer and never advance robot state. `poll()` performs deterministic reducer
ingress on the policy thread. Capture terminal state is non-authoritative and
cannot select a policy source, alter safety, gate IK, dispatch an action, or
commit an episode.

## Evidence boundary

Focused tests use fake servers, typed events, passive journals, real reducer and
AX=XB code on synthetic poses, and temporary local artifact directories only.
They do not bind a socket, connect a Quest, query RM65 hardware, or execute
motion. Consequently they establish the in-process application plus immutable
capture/diagnostic-report seams, not physical calibration quality, reviewed
acceptance, frame-installation readiness, or future-trajectory visualization
readiness.
