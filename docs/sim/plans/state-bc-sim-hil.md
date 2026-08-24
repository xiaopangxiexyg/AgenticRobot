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

The change in this Task is limited to an offline learner, a MuJoCo observation
and action adapter, and a source-lineage test seam. The existing
`MujocoDeployRuntime` remains the only execution owner for admission, safety,
IK, and dispatch. No new queue, authority, simulator entrypoint, device
worker, GUI, or hardware path is introduced.

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

The remote manifest hash is
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
than being silently deleted. The source parquet remains remote and read-only
until an explicit data-transfer approval is given.

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
- standard policy inference is the nominal action source;
- RTC inference receives the current state and nominal policy action, then
  either emits one replacement action or returns `rtc_no_action`;
- `rtc_no_action`, hold, and activation events never become policy-training
  samples;
- the trace records source, activation, inference mode, and action lineage;
- lower runtime safety/IK/dispatch remains authoritative after the adapter.

This is sufficient to compare ordinary teleoperation, takeover/handback, and
RTC intervention in a deterministic mock trace. A later MuJoCo target test can
feed the same normalized actions to `MujocoDeployRuntime`; it must not bypass
that runtime with a parallel simulator control loop.

## Evidence ceiling and next gate

The current Task proves contracts, fixture selection, a self-contained linear
artifact round trip, 16D-to-26D projection, frame-contract failure-closed
behavior, and exclusive HITL lineage. It does **not** yet prove a trained
artifact on the remote parquet, task success, causal recovery benefit, or
physical safety. It also does not start a real device, GUI, policy server, or
motion.

The next bounded gate is a user-approved, read-only synchronization of the
selected low-dimensional parquet plus metadata (no video/tactile copy),
followed by the trainer CLI and a committed artifact manifest. Only after that
offline evidence is available should the MuJoCo adapter be exercised against a
resettable task and the three collection modes be compared.
