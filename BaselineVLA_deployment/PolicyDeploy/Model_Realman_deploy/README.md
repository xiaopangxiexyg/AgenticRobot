# ManipRep Realman Deploy Policy

This directory adapts ManipRep checkpoints to `teleop_runtime.deploy.run_policy`
for Realman/RM65 deployment.

Before replay/policy execution starts, the runtime performs a startup
initialization segment. By default `INIT_TIME_S=5.0` moves smoothly to the
replay first frame, an explicitly configured dataset frame, or the current
robot state.

## Compatibility

Native Realman checkpoints must use the collection schema from `teleop_runtime`:

```text
state_dim  = 38
action_dim = 26

state  = [left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]
action = [left_pose7, right_pose7, left_hand6, right_hand6]
```

The policy fails fast when a checkpoint has `state_dim=19`, because that is the
FR3 layout and the arm action semantics are different.

## Offline Smoke

Run this before connecting hardware:

```bash
python -m PolicyDeploy.Model_Realman_deploy.offline_smoke \
  --checkpoint /abs/path/to/best.pt \
  --device cpu \
  --active-sides right \
  --require-tactile \
  --action-chunk-size 4
```

Expected output includes `payload_shapes`, `payload_finite True`, and, when
`teleop_runtime` is importable, `normalized_first_shape (26,)`.

## Right Arm Dry Run

The wrapper defaults to dry-run. It builds the policy from `CHECKPOINT`; the
path is never hard-coded.

```bash
CHECKPOINT=/abs/path/to/best.pt \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
bash PolicyDeploy/Model_Realman_deploy/run_right_policy.sh
```

Set `SEND_ACTIONS=1` only after dry-run status is stable. Set `AUTO_START=1`
only when you do not want to press `g` manually.

## Bimanual Dry Run

```bash
CHECKPOINT=/abs/path/to/best.pt \
LEFT_ARM_IP="$LEFT_ARM_IP" \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
LEFT_HAND_IP="$LEFT_HAND_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
bash PolicyDeploy/Model_Realman_deploy/run_bimanual_policy.sh
```

## Code Path and Runtime Flow

`PolicyDeploy/Model_Realman_deploy` provides Realman wrapper scripts and the
ManipRep checkpoint adapter. The runtime movement, replay, safety filtering,
and hardware sending are handled by `teleop_runtime.deploy` from
`MANIPREP_DEPLOY_PACKAGE_ROOT`.

- `policy.py` restores the ManipRep checkpoint through the FR3 base adapter,
  checks the native Realman schema, and formats model output into runtime action
  fields.
- `build_policy_kwargs.py` converts environment variables into the policy
  constructor kwargs used by the shell wrappers.
- `run_policy.sh` activates the runtime conda env, exports
  `PYTHONPATH=${MANIPREP_DEPLOY_PACKAGE_ROOT}:${MANIPREP_PROJECT_ROOT}`, and
  calls `python -m teleop_runtime.deploy.run_policy`. `DEPLOY_MODE=model`
  builds ManipRep policy kwargs; `DEPLOY_MODE=move_frame` uses the built-in
  hold policy; `DEPLOY_MODE=replay` uses runtime dataset replay.
- `run_right_policy.sh` sets `ACTIVE_SIDES=right`; `run_bimanual_policy.sh` sets
  `ACTIVE_SIDES=both`.
- In the runtime, `startup_init.py` loads the selected initial frame and
  interpolates from the current robot state to that target. Arm xyz and hand
  joints are interpolated linearly; arm quaternions use slerp.
- `replay.py` loads dataset/parquet actions for debug replay. Replay pops one
  action per control-loop tick, so the real replay frequency is `HZ`; parquet
  timestamps are not resampled automatically.
- `actions.py` normalizes policy/replay outputs to the 26-D Realman action
  vector and applies workspace/rate safety filters before hardware commands are
  sent.

For all hardware runs, first omit `--send-actions` or keep `SEND_ACTIONS=0`.
Only enable real command sending after the status line is stable and the target
frame is known to be safe.

## Trajectory Init, Replay, and Policy Start

Frame indexes are zero-based. `INIT_FRAME=120` means row 120 in the selected
episode parquet. `--debug-replay-end-index` is exclusive. If the parquet uses a
non-default action column, set `INIT_ACTION_COLUMN` for wrapper init, or
`--init-action-column` / `--debug-replay-action-column` for direct runtime
commands.

The 26-D Realman action layout is:

```text
action = [left_arm_pose7, right_arm_pose7, left_hand6, right_hand6]
```

For `ACTIVE_SIDES=right`, only the right-side arm/hand slices are required. For
`ACTIVE_SIDES=both`, both left and right slices must be finite for every enabled
actuator family. If a trajectory contains only arm commands, disable hands with
`--no-hands` for direct runtime commands or `ENABLE_HANDS=0` for the wrappers.
The single-arm examples below use the right arm. For left-arm direct runtime
replay/movement, replace `right` with `left` and use the left-side IP variables.
For model deployment, compact single-arm checkpoint layouts in this adapter are
right-arm layouts; left-only model deployment should use a native full 26-D
Realman checkpoint.

### Wrapper Modes and Direct Runtime Setup

The Realman wrappers support all three deployment flows:

- `DEPLOY_MODE=move_frame`: smoothly move to one selected dataset/parquet frame,
  then hold the observed state.
- `DEPLOY_MODE=replay`: smoothly move to the replay first frame, then replay the
  trajectory at the reference episode frequency.
- `DEPLOY_MODE=model`: smoothly move to the selected first frame, then start the
  ManipRep checkpoint.

The direct runtime commands below remain useful for debugging lower-level
`teleop_runtime.deploy` behavior without the wrapper.

Run direct runtime commands from the ManipRep repo root:

```bash
source Scripts/load_local_env.sh
maniprep_activate_conda "$CONDA_ENV"
export PYTHONPATH="${MANIPREP_DEPLOY_PACKAGE_ROOT}:${PWD}:${PYTHONPATH:-}"
```

### 1. Smoothly Move to One Frame of a Trajectory

Use the built-in `hold` policy plus `--init-source dataset`. After startup init
finishes, the hold policy keeps sending the observed current state. Press `q` to
exit, or add a conservative `--max-steps` budget.

Wrapper right-arm/right-hand:

```bash
DEPLOY_MODE=move_frame \
ACTIVE_SIDES=right \
INIT_DATASET=/home/ai/Desktop/DexTeleop/real_datasets/2026-04-30_heavy_light \
INIT_EPISODE=2 \
INIT_FRAME=120 \
INIT_TIME_S=5.0 \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
AUTO_START=1 \
SEND_ACTIONS=1 \
bash PolicyDeploy/Model_Realman_deploy/run_right_policy.sh
```

Wrapper bimanual:

```bash
DEPLOY_MODE=move_frame \
ACTIVE_SIDES=both \
INIT_DATASET=/home/ai/Desktop/DexTeleop/real_datasets/paper_fold_datasets/large_brown_paper_fold \
INIT_EPISODE=2 \
INIT_FRAME=0 \
INIT_TIME_S=5.0 \
LEFT_ARM_IP="$LEFT_ARM_IP" \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
LEFT_HAND_IP="$LEFT_HAND_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
AUTO_START=1 \
SEND_ACTIONS=1 \
bash PolicyDeploy/Model_Realman_deploy/run_bimanual_policy.sh
```

Right-arm/right-hand:

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --camera-mode none \
  --require-ready-observation \
  --init-source dataset \
  --init-dataset /abs/path/to/dataset \
  --init-episode 2 \
  --init-frame 120 \
  --init-time-s 5.0 \
  --hz 30 \
  --auto-start \
  --send-actions
```

Bimanual:

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --camera-mode none \
  --require-ready-observation \
  --init-source dataset \
  --init-dataset /abs/path/to/dataset \
  --init-episode 2 \
  --init-frame 120 \
  --init-time-s 5.0 \
  --hz 30 \
  --auto-start \
  --send-actions
```

To use one parquet directly, replace `--init-dataset /abs/path/to/dataset` and
`--init-episode 2` with:

```bash
--init-parquet /abs/path/to/episode_000002.parquet
```

### 2. Smoothly Move to the First Frame, Then Replay at Real Frequency

Use runtime debug replay. `--init-source replay` makes startup init target the
first loaded replay action. `--debug-replay-start-index 0` means the trajectory
starts from the dataset first frame. The wrapper infers the action/control
frequency from the replay dataset/parquet reference, and falls back to `13Hz`
only when no reference is available. Replay uses append queueing by default in
the wrapper/runtime, so prefetched chunks are not dropped.

Wrapper right-arm/right-hand:

```bash
DEPLOY_MODE=replay \
ACTIVE_SIDES=right \
DEBUG_REPLAY_DATASET=/home/ai/Desktop/DexTeleop/real_datasets/2026-04-30_heavy_light \
DEBUG_REPLAY_EPISODE=2 \
DEBUG_REPLAY_START_INDEX=0 \
DEBUG_REPLAY_CHUNK_SIZE=32 \
INIT_TIME_S=5.0 \
MIN_ACTION_QUEUE=8 \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
AUTO_START=1 \
SEND_ACTIONS=1 \
bash PolicyDeploy/Model_Realman_deploy/run_right_policy.sh
```

Wrapper bimanual uses the same variables with `ACTIVE_SIDES=both` plus both
left/right arm and hand IPs, then calls `run_bimanual_policy.sh`.

```bash
DEPLOY_MODE=replay \
ACTIVE_SIDES=both \
DEBUG_REPLAY_DATASET=/home/ai/Desktop/DexTeleop/real_datasets/paper_fold_datasets/large_brown_paper_fold \
DEBUG_REPLAY_EPISODE=10 \
DEBUG_REPLAY_START_INDEX=0 \
DEBUG_REPLAY_CHUNK_SIZE=32 \
INIT_TIME_S=5.0 \
MIN_ACTION_QUEUE=8 \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
LEFT_ARM_IP="$LEFT_ARM_IP" \
LEFT_HAND_IP="$LEFT_HAND_IP" \
AUTO_START=1 \
SEND_ACTIONS=1 \
bash PolicyDeploy/Model_Realman_deploy/run_bimanual_policy.sh
```

Right-arm/right-hand:

```bash
python -m teleop_runtime.deploy.run_policy \
  --debug-replay-dataset /home/ai/Desktop/DexTeleop/real_datasets/2026-04-30_heavy_light \
  --debug-replay-episode 2 \
  --debug-replay-start-index 0 \
  --debug-replay-chunk-size 32 \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --camera-mode none \
  --require-ready-observation \
  --init-source replay \
  --init-time-s 5.0 \
  --min-action-queue 8 \
  --action-queue-policy append \
  --auto-start \
  --send-actions
```

Bimanual:

```bash
python -m teleop_runtime.deploy.run_policy \
  --debug-replay-dataset /home/ai/Desktop/DexTeleop/real_datasets/paper_fold_datasets/large_brown_paper_fold \
  --debug-replay-episode 2 \
  --debug-replay-start-index 0 \
  --debug-replay-chunk-size 32 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --camera-mode none \
  --require-ready-observation \
  --init-source replay \
  --init-time-s 5.0 \
  --min-action-queue 8 \
  --action-queue-policy append \
  --auto-start \
  --send-actions
```

For direct parquet replay, replace `--debug-replay-dataset` and
`--debug-replay-episode` with:

```bash
--debug-replay-parquet /abs/path/to/episode_000002.parquet
```

By default replay holds the final action after the last frame. Add
`--debug-replay-loop` only when repeated replay is intended.

### 3. Smoothly Move to the First Frame, Then Start the Model

Use the ManipRep wrappers. Set `INIT_SOURCE=dataset`, `INIT_FRAME=0`, and pass
the same dataset/parquet used as the task start pose. After startup init
completes, `teleop_runtime.deploy.run_policy` starts querying
`ManipRepRealmanPolicy`.

Right-arm model:

```bash
CHECKPOINT=/abs/path/to/best.pt \
DEPLOY_MODE=model \
ACTIVE_SIDES=right \
INIT_SOURCE=dataset \
INIT_DATASET=/home/ai/Desktop/DexTeleop/real_datasets/2026-04-30_heavy_light \
INIT_EPISODE=2 \
INIT_FRAME=0 \
INIT_TIME_S=5.0 \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
REALSENSE_ID=213222079798 \
SEND_ACTIONS=1 \
AUTO_START=1 \
bash PolicyDeploy/Model_Realman_deploy/run_right_policy.sh
```

Bimanual model:

```bash
CHECKPOINT=/home/ai/Desktop/DexTeleop/BaselineVLA_deployment/Outputs/server_paper_fold/vanilla/realrobot_paper_fold_vtactile_vla_ac16_nohis_vis_imgaug_tacnorm/last.pt \
DEPLOY_MODE=model \
ACTIVE_SIDES=both \
INIT_SOURCE=dataset \
INIT_DATASET=/home/ai/Desktop/DexTeleop/real_datasets/paper_fold_datasets/large_brown_paper_fold \
INIT_EPISODE=2 \
INIT_FRAME=0 \
INIT_TIME_S=5.0 \
LEFT_ARM_IP="$LEFT_ARM_IP" \
RIGHT_ARM_IP="$RIGHT_ARM_IP" \
LEFT_HAND_IP="$LEFT_HAND_IP" \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
REALSENSE_ID=213222079798 \
SEND_ACTIONS=1 \
AUTO_START=1 \
bash PolicyDeploy/Model_Realman_deploy/run_bimanual_policy.sh
```

For one parquet directly:

```bash
INIT_PARQUET=/abs/path/to/episode_000002.parquet
```

with `INIT_DATASET` unset.

## Important Environment Variables

| Env var | Default | Meaning |
| --- | --- | --- |
| `DEPLOY_MODE` | `model` | `move_frame`, `replay`, or `model`. `run_right_policy.sh` and `run_bimanual_policy.sh` both use this mode switch. |
| `CHECKPOINT` | required | Checkpoint path. Aliases: `CKPT`, `MODEL_CKPT`. A directory resolves to `last.pt`. |
| `CONDA_ENV` | from `local_paths.env` | Runtime conda env. Set this in `BaselineVLA_deployment/local_paths.env`; wrapper commands do not override it. |
| `DEVICE` | `cuda` | Torch device passed to the policy. |
| `ACTIVE_SIDES` | `right` or wrapper default | `right`, `left`, or `both`; right wrapper uses `right`, bimanual wrapper uses `both`. |
| `ACTION_LAYOUT` | `auto` | Keep `auto` for native 26-D Realman checkpoints. |
| `RIGHT_ARM_IP` / `LEFT_ARM_IP` | empty | RM65 arm IPs. Required by runtime when `SEND_ACTIONS=1`. |
| `RIGHT_HAND_IP` / `LEFT_HAND_IP` | empty | InspireTac hand IPs. Needed for tactile checkpoints with `REQUIRE_TACTILE=1`. |
| `ENABLE_ARMS` | `1` | Passes `--enable-arms` or `--no-arms`. |
| `ENABLE_HANDS` | `1` | Passes `--enable-hands` or `--no-hands`. |
| `CAMERA_MODE` | `single` | `none`, `single`, or `multi`. |
| `CAMERA_KEY` | `front` | Camera key used by both policy and runtime. |
| `REALSENSE_ID` / `CAMERA_SERIAL` | empty | RealSense serial/id for `CAMERA_MODE=single`. `REALSENSE_ID`, `REALSENSE_SERIAL`, and `CAMERA_ID` are aliases for `CAMERA_SERIAL`. |
| `CAMERA_CONFIG` | repo default when present | RealSense JSON config. Used when `CAMERA_SERIAL` is empty; required for `CAMERA_MODE=multi`. |
| `ACTION_CHUNK_SIZE` | `all` | `all`/`0` uses full model horizon; positive N keeps first N actions. |
| `MIN_ACTION_QUEUE` | `8` | Query policy when queued actions are at or below this length. |
| `REQUIRE_TACTILE` | `1` | Tactile checkpoint errors if the right tactile vector is missing. |
| `REQUIRE_IMAGE` | `1` | Policy errors when the configured camera image is missing. |
| `REQUIRE_READY_OBSERVATION` | `1` | Runtime blocks policy execution on observation faults. |
| `SEND_ACTIONS` | `0` | `1` sends filtered commands to hardware. |
| `AUTO_START` | `0` | `1` starts immediately instead of waiting for `g`. |
| `INIT_TIME_S` | `5.0` | Smooth initialization duration before replay/policy execution. `0` disables. |
| `INIT_SOURCE` | `auto` | `auto`, `current`, `dataset`, `replay`, or `none`. `auto` uses replay first frame when replaying, an explicit init dataset when provided, otherwise current state. |
| `INIT_DATASET` / `INIT_PARQUET` | empty | Optional dataset root or episode parquet for deploy initialization from a recorded frame. |
| `INIT_EPISODE` / `INIT_FRAME` | `0` / `0` | Dataset episode and frame used when `INIT_SOURCE=dataset` or `INIT_SOURCE=auto` with `INIT_DATASET`/`INIT_PARQUET`. |
| `INIT_ACTION_COLUMN` | `action` | Full action column used for init/replay parquet loading. |
| `DEBUG_REPLAY_DATASET` / `DEBUG_REPLAY_PARQUET` | empty | Dataset root or direct parquet for `DEPLOY_MODE=replay`. Aliases: `REPLAY_DATASET`, `REPLAY_PARQUET`. |
| `DEBUG_REPLAY_EPISODE` / `DEBUG_REPLAY_START_INDEX` | `INIT_EPISODE` / `0` | Replay episode and first frame for `DEPLOY_MODE=replay`. |
| `DEBUG_REPLAY_CHUNK_SIZE` | `32` | Number of replay actions prefetched per policy query. |
| `APPEND_ACTION_CHUNKS` | mode default | Defaults to `1` for `DEPLOY_MODE=replay` and `0` otherwise. |

For a native Realman tactile VLA checkpoint, use `ACTION_LAYOUT=auto`; it will
resolve to the full 26-D schema. Compact right-arm checkpoints are supported
only when their layout is unambiguous, for example `ACTION_LAYOUT=right_arm_pose`
for 7-D or `ACTION_LAYOUT=right_arm_pose_hand` for 13-D.

Example dataset-frame initialization before normal policy deploy:

```bash
INIT_SOURCE=dataset \
INIT_DATASET=/abs/path/to/dataset \
INIT_EPISODE=2 \
INIT_FRAME=0 \
bash PolicyDeploy/Model_Realman_deploy/run_right_policy.sh
```
