# State-only BC and simulation HITL plan

## Objective and boundary

The first mainline probe uses the low-dimensional state already recorded by
`FormalCollection` rather than images or simulator-generated observations:

```text
state[16] = left TCP xyzquat[7] + right TCP xyzquat[7]
             + left measured gripper scalar[1] + right measured gripper scalar[1]
action[16] = left TCP xyzquat[7] + right TCP xyzquat[7]
              + left/right absolute gripper targets[1]
```

The recorded pose frame is `desk_absolute`, the pose representation is
`xyz_quat_xyzw`, the action mode is absolute, and the gripper values are
`normalized_0_1`. `observation.arm.qpos` is audit-only and is not a learner
input for this probe. This deliberately avoids a simulator dynamics gap for
the policy input/output experiment; it does not claim that the learned policy
is physically valid or ready for a robot.

The live integration keeps the existing `MujocoDeployRuntime` as the only
execution owner for admission, safety, IK, and dispatch. The Task adds only a
state-BC provider binding, CLI selection, and keyboard aliases for the
existing typed source intents; it does not add a second queue, authority,
simulator control loop, or hardware path.

The stream seam deliberately has two typed spaces. State-BC inference keeps the
16D model state/action contract above, while every executable policy, human, or
RTC replacement is normalized to the current 26D
`deploy_action_vector.ee_pose_q6.v1` / `bimanual_ee_pose_v1` action boundary.
`StatePolicyAdapter.project_model_action()` is the single 16D-to-26D bridge;
the streaming runner records both representations and injects the resulting
action into an execution callback. It does not own safety, IK, dispatch,
episode reset, or a second control loop.

## Dataset selector

The selector names source directories explicitly and stores every excluded
episode reason. It reads only `observation.state` and `action` from parquet;
media, tactile streams, and qpos are not opened by the first trainer.

The read-only inventory identified these candidates on
`user@192.168.70.116:/home/user/zc/datasets/RealmanDMTAC/FormalCollection`:

1. **Strict full-task day (recommended first selector):** the two
   `2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage` directories.
   They are one logical day split into two physical folders. The selector must
   include both paths and the training manifest must retain their source paths;
   a read-only identity/time non-overlap check is required before treating
   them as one day.
2. **Quality-first single-directory fallback:**
   `2026-08-20_4_stage_writing_detail`. This is easier to attribute to one
   source directory, but it is not the same task/profile as the 2026-08-07
   full-task pair and therefore must be labelled as a different selector.
3. `2026-08-21_4_stage_writing_detail` has more frames, but its task identity
   and short-episode mix make it unsuitable as the default first benchmark.
   The high-invalid-rate 2026-08-24 directory should not be the first fit.

The remote policy I/O layout hash declared by `meta/info.json` is
`sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16`.
The checked-in v2.2 runtime layout currently derives
`sha256:52e4bc42ad0a0e562c357e41da1ee2cbd827b4dc98643ffb73fe7acfeafc66ab`
from the same named layout. Both are accepted as explicit aliases, while the
simulator always binds the current checked-in layout and records the source
hash. This is a compatibility gate, not permission to accept arbitrary layout
hashes; a future layout change must add a new reviewed alias and tests.

Before a real-data fit is admitted, the selected `meta/info.json` must still
confirm the expected `pose_storage_frame=desk_absolute`,
`arm_pose_action_mode=absolute`, state/action dimensions, and the shared
policy I/O identity. Invalid episodes remain in the selection manifest rather
than being silently deleted. The source parquet remains remote and read-only.
For the bounded 2026-08-25 probe, explicit approval was used to copy only the
manifest files and integrity-valid parquet files into an ephemeral `/tmp`
staging tree. No `videos/`, `tactile/`, or wrist-depth paths were copied.

### Completed 2026-08-07 offline probe

The selected logical day was the pair of physical directories listed above.
Remote read-only manifest evidence was:

| item | result |
| --- | --- |
| physical source directories | 2 (129 + 202 episodes) |
| complete episodes | 204 (83 + 121) |
| excluded episodes | 127 (46 + 81; retained in the selection manifest) |
| complete frames | 69,440 (29,714 + 39,726) |
| valid parquet synchronized | 204 files, 366,949,962 bytes |
| per-file transfer verification | 204/204 SHA-256 matches, 0 missing |
| source layout hash | `sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16` |

The remote `info.json` declares parquet features only for low-dimensional
`action`, `observation.state`, `observation.arm.qpos` (audit-only), and
`observation.last_action`; media and tactile streams are separate paths. The
local loader opened only `observation.state` and `action`.

The first deterministic fit used an episode-level 80/20 split, seed `0`, and
ridge `1e-4`. It produced 55,343 training frames and 14,097 held-out frames.
Held-out metrics were action MAE `0.00907047`, left/right pose-position RMSE
`0.001598/0.002521 m`, pose-orientation MAE `0.007935 rad`, and left/right
gripper MAE `0.05781/0.06542`. The self-contained artifact hash was
`sha256:61ae3388e62dd8410b7e3cdbd5d2489bf3717eb85cc5e54c60ccc7a1cd1e85e0`;
the artifact and staging tree are ephemeral and are not treated as a
versioned model resource.

## Reproducible offline flow

```text
remote metadata (read-only)
        |
explicit selector + integrity-valid episodes
        |
state/action columns only, split by episode
        |
linear ridge state BC artifact (.npz + manifest/hash)
        |
MuJoCo 38D DeploymentObservation
        |  (explicit raw -> desk pose contract when needed)
16D v2.2 state observation adapter
        |
16D absolute prediction -> v2.2 layout -> existing 26D q6 bridge
        |
MujocoDeployRuntime -> existing safety / IK / dispatch owner
```

The linear ridge model is a baseline for wiring and evidence, not the final
algorithm. It gives a deterministic artifact, normalization, frame identity,
and held-out episode metrics before a larger learner is introduced.

## HITL validation contract

HITL is tested as exclusive source selection, not action blending:

- exactly one `policy`, `human`, or `rtc` source may execute an action at a
  step;
- every source switch emits `hold` followed by a fresh activation/emission;
- after the lower HOLD completes, the runner captures a new deployment
  observation; activation, policy inference, human execution, and RTC
  conditioning use that post-HOLD sample;
- standard policy inference is the nominal action source;
- RTC inference receives the current state and nominal policy action, then
  either emits one replacement action or returns `rtc_no_action`;
- `rtc_no_action`, hold, and activation events never become policy-training
  samples;
- the trace records source, activation, inference mode, and action lineage;
- `experiments/sim_learning/state_bc/stream.py` accepts a 26D human action,
  runs ordinary policy inference, and gives RTC the 16D state plus nominal
  16D/26D actions; RTC may return no action, a 26D runtime replacement, or a
  16D replacement that uses the shared adapter bridge;
- lower runtime safety/IK/dispatch remains authoritative after the adapter.

`experiments/sim_learning/state_bc/mujoco_stream.py` is the bounded concrete
bridge for this experiment. On a switch it clears only the existing
`PolicyDeployRuntime` queue/schedule, requires a complete lower cancellation
receipt, advances the existing single-producer authority epoch while fenced,
and then establishes READY. It does not create policy/human/RTC producer
authorities: the mutually selected upper source is serialized through the one
existing policy-emission target. Each executable action is admitted through
`PolicyDeployRuntime`, submitted to `MujocoPolicySkillMotionTarget`, and
advanced by one authoritative decision; `rtc_no_action` does none of these.
Target start/reset/close and every safety, IK, reference, and dispatch decision
remain lower-owned.

This is sufficient to compare ordinary teleoperation, takeover/handback, and
RTC intervention in a deterministic mock trace. A later MuJoCo target test can
feed the same normalized actions to `MujocoDeployRuntime`; it must not bypass
that runtime with a parallel simulator control loop.

## Live state-BC Sim-HIL path

The canonical interactive entrypoint is the existing Control Station command
with an explicit state-BC artifact and the exclusive source-selection profile:

```bash
python -m control_station sim-hil \
  --enable \
  --policy-backend state_bc \
  --state-bc-artifact /absolute/path/to/state_bc.npz \
  --control-profile exclusive_truncated_hitl \
  --control-gui
```

`--state-bc-artifact` is a self-contained `.npz` produced by the state-BC
learner. Startup preflight verifies its schema, hash, accepted v2.2 layout,
and `16 -> 16` model dimensions without contacting a server. At runtime the
source captures both MuJoCo arms/grippers as a 38-D deployment observation,
projects it to the 16-D desk-frame state, predicts one absolute 16-D action,
and uses `StatePolicyAdapter` to produce the existing 26-D runtime action. The
first binding contributes the right arm and right gripper components, matching
the current Sim-HIL target; `full_bimanual_execution` remains explicitly
false.

The source uses `ThreadedPolicyEmissionSource`, so model inference stays off
the Control Station cycle while queue admission and emission creation remain
on the existing runtime boundary. `H` submits the typed Human takeover,
`P` submits the typed Policy handback, and `Esc` submits the existing HOLD
intent. Shortcuts are disabled while a form control is focused and are active
only for the `exclusive_truncated_hitl` workflow. They never write actions or
authority directly.

The live smoke acceptance criterion is: a fresh artifact passes preflight; the
MuJoCo target starts with Policy selected; a continuous VIVE stream can take
over through the typed Human path; `HOLD -> fresh emission` is visible at the
switch; and `P` returns to state-BC emissions through the same target. This
verifies wiring and source lineage, not task success, policy quality, latency
guarantees, safety-envelope acceptance, or physical readiness. No real device,
policy server, or actuator is started by the automated tests.

### State-BC GUI runbook

With `--control-gui`, the existing loopback Control Station page is the
operator surface for the complete Sim-HIL run. The separate MuJoCo viewer is
the passive scene view; the browser owns only typed acceptance intents and
status presentation. Before a target is constructed, the page's **策略运行时**
card already identifies the selected `state_bc` backend, the artifact file,
the `16D state → 16D action → 26D runtime` contract, bilateral state
observation, and right-side-only execution. After construction it updates the
same card from the live policy-source status and keeps the artifact/layout
diagnostics in the read-only panel.

The intended operator sequence is:

1. Start the command above and open the printed loopback URL (or use
   `--open-browser`). Confirm the state-BC card and the separate MuJoCo viewer
   are visible before selecting a source.
2. In the **直接接管** source cards, choose **以 state-BC 启动**. The page
   keeps VIVE setup visible but does not block policy-only startup on Human
   readiness.
3. When the policy source is running, press `H` or click the Human card to
   request takeover. The page shows the HOLD/Runtime Home transition and the
   resulting Human owner; it never writes an action directly from JavaScript.
4. Press `P` or click the state-BC card to hand control back. The page reports
   the fresh post-HOLD state-BC emission and keeps the single lower execution
   owner visible. `Esc` remains the immediate typed HOLD action.
5. Use the visible HOLD/rearm controls for a controlled pause, and expand
   **结束、保存与重建** only when finalizing, resetting, or exiting the run.

The browser cannot select a different artifact at runtime: the CLI path is
the immutable source binding for this session. This prevents an operator GUI
choice from silently changing model identity or the state/action contract.

## Evidence ceiling and next gate

The current Task now proves the contracts, explicit 2026-08-07 selector,
real-data state-only fit, self-contained artifact round trip, 16D-to-26D
projection, frame-contract failure-closed behavior, and exclusive HITL
lineage. `tests/test_state_bc_stream.py` additionally proves the bounded
streaming runner's ordinary policy, human takeover/handback, RTC replacement,
RTC `no_action`, and exclusive 26D runtime action sink; the adapter test also
checks that the reusable projection helper matches ordinary prediction. The
real-artifact adapter smoke was observation-only: it projected one
recorded state through the existing bridge and produced a finite 26D action,
without stepping physics. The real-artifact HITL trace exercised policy,
human takeover/handback, RTC replacement, and RTC `no_action`; HOLD,
activation, and `no_action` remained excluded from samples.

This evidence still does **not** prove task success, causal recovery benefit,
safety-envelope admission, or physical safety. It also does not start a real
device, GUI, policy server, or motion.

The supported development asset store is now linked into the Task worktree by
the standard `dev_asset_store.py link-worktree` flow:

```text
Task/assets
  -> /home/zc/HandMotion/Developing/AgenticRobot/assets
```

The link is ignored, read-mostly, and points at the reviewed main-checkout
asset root; the main checkout itself was not changed. With that binding, a
no-device MuJoCo smoke using
`sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json` completed one
`reset(seed=0)` and one zero-action `step`: observation shape `(28,)`, task
action shape `(7,)`, reward `0.0`, and no termination. The same settled scene
was captured through `MujocoDeploymentObservationAdapter` as a finite 38D
deployment observation, then the real state-BC artifact produced a finite 16D
model view and finite 26D normalized action.

The current Cube-Lift task is nevertheless a right-arm 7D task interface,
while the state-BC artifact is a bilateral 16D-to-26D deployment policy. The
earlier reset/observation smoke therefore validated asset loading, scene
observation, and the model adapter seam only; it did not by itself route the
26D output through the bilateral pose-target runtime or claim task success.
The bounded runtime seam is now covered by the optional test below. The next
experiment gate is repeated resettable episodes with task predicates and
matched policy/human/RTC intervention schedules, without introducing a
parallel execution loop.

### Completed bilateral runtime seam smoke

The bounded optional test
`tests/test_mujoco_state_policy_adapter.py::test_state_policy_output_reaches_existing_bimanual_mujoco_target`
now exercises that next gate with the checked-in
`rm65_right_cube_lift_rl_normal_v2` scene and `realtime_worker=False`. It
captures both TCPs and grippers, converts the state-BC desk-frame output through
the reviewed RM65 runtime pose contract, and runs the sequence `policy -> human
-> policy -> rtc(no_action) -> rtc(replacement)`. Every switch obtains a
complete lower HOLD receipt, recaptures observation, advances the same lower
policy authority while fenced, and establishes READY. The four executable 26D
actions traverse the same `PolicyDeployRuntime` emission boundary and
`MujocoPolicySkillMotionTarget`; the RTC no-action produces no emission. The
target's existing component ingress, canonical safety, coupled IK, q-reference
generation, and deterministic MuJoCo ticks complete with `dispatched` status
and finite bilateral qpos measurement.

The non-realtime test profile explicitly enables the existing bounded final
keyframe window with `action_maximum_hold_sec=0.25` and
`extend_final_keyframe_to_maximum_hold=True`. The bridge does not extend an
action: after submission it requires an authoritative decision tick inside the
emission's declared interval and fails closed otherwise. A deployment profile
must choose and justify its own timing contract.

This is an execution-path contract smoke, not a task-success or policy-quality
claim. It uses a deterministic injected policy in the test, does not use the
ephemeral real-data artifact, and does not evaluate cube lifting, recovery
benefit, safety-envelope margins, operator usability, or real-time latency.
Those remain experiment-level comparisons; no device, GUI, hardware motion, or
RTC Active deployment was started.

### Matched episode evaluation organization

`experiments/sim_learning/state_bc/evaluation.py` now defines the next
experiment-only organization boundary.  A panel contains exactly three
equal-horizon conditions: policy-only, policy with human takeover/handback, and
policy with RTC intervention.  For every seed, the caller must return a reset
receipt with the same `initial_state_id` for all three conditions.  A fresh
streaming runner is then constructed after each reset, so source and bridge
state cannot leak across episodes.

The evaluator records the existing exclusive stream trace and one typed task
assessment (`success`, `failure`, or `unknown`).  `unknown` requires a reason
and makes aggregate success rate unavailable instead of being silently counted
as failure or success.  The evaluator does not define a cube-lift predicate,
reset the simulator, construct the MuJoCo target, load a model artifact, or own
any execution lifecycle; those remain injected experiment dependencies.  This
keeps task predicates and reset manifests explicit while preventing a second
control or dispatch loop.

The optional MuJoCo panel test constructs a fresh existing policy target for
each condition, hashes the initial simulator `qpos+qvel` as the matched reset
identity, and runs a five-step schedule.  All executable actions traverse the
existing emission, safety, coupled-IK, reference, and dispatch path; the first
RTC intervention is `rtc_no_action` and therefore emits nothing.  A cube-height
predicate reports all three hold-current fixture episodes as `failure`, which
is the expected negative result rather than a task-success claim.
