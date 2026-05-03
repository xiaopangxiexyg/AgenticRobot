# ManipRep FR3 Deploy Policy

This directory contains the deployment adapter for ManipRep real-robot VLA checkpoints.

The adapter is loaded by the existing `FR3_deploy` policy hook. The deploy loop now runs a startup initialization segment before replay/policy execution: by default it spends `INIT_TIME_S=5.0` seconds moving smoothly to the replay first frame, an explicitly configured dataset frame, or the current robot state.

## Dry Run

Convenience wrappers are provided and default to dry-run:

```bash
CHECKPOINT=/abs/path/to/checkpoints/epoch_XXXX.pt \
MAX_FORCE=1000 \
ACTION_CHUNK_SIZE=all \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
bash PolicyDeploy/Model_FR3_deploy/run_tactile_policy.sh
```

```bash
CHECKPOINT=/abs/path/to/best.pt \
ACTION_CHUNK_SIZE=4 \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
bash PolicyDeploy/Model_FR3_deploy/run_notactile_policy.sh
```

Set `SEND_ACTIONS=1` only after dry-run is stable. Set `AUTO_START=1` only when you do not want to press `g` manually.
Set `MANIPREP_DEPLOY_PACKAGE_ROOT` in `local_paths.env` when the real-robot deploy package is outside this repository.

## Code Path and Runtime Flow

`PolicyDeploy/Model_FR3_deploy` wraps ManipRep checkpoints for the
`inspire_data_collect.deploy` runtime. The wrapper restores the model and
formats model output; startup movement, dataset replay, safety filtering, and
hardware sending are handled by the runtime under `MANIPREP_DEPLOY_PACKAGE_ROOT`.

- `policy.py` restores the ManipRep checkpoint, rebuilds the training-time
  state/image/tactile inputs, and returns `{"action_chunk": ...}`.
- `build_policy_kwargs.py` converts wrapper environment variables into policy
  constructor kwargs.
- `run_tactile_policy.sh` and `run_notactile_policy.sh` activate the conda env,
  export `PYTHONPATH=${MANIPREP_DEPLOY_PACKAGE_ROOT}:${MANIPREP_PROJECT_ROOT}`,
  build policy kwargs JSON, and call
  `python -m inspire_data_collect.deploy.run_policy`.
- `startup_init.py` loads the selected initial frame and uses smoothstep linear
  interpolation from the current hold action to that target.
- `replay.py` loads full 19-D actions from parquet, or reconstructs them from
  supported component columns. Replay pops one action per control-loop tick, so
  the replay frequency is `HZ`; parquet timestamps are not resampled.
- `actions.py` normalizes policy/replay outputs to the 19-D FR3 action vector
  and applies safety filters before hardware commands are sent. The FR3 arm is
  joint-space qpos with safe clipping and `MAX_ARM_STEP_RAD`; hand q6 commands
  are clipped to `[0, 1]` and rate-limited by `MAX_HAND_STEP`.

For all hardware runs, first omit `--send-actions` or keep `SEND_ACTIONS=0`.
Only enable real command sending after dry-run status is stable and the target
frame is known to be safe.

## Trajectory Init, Replay, and Policy Start

Frame indexes are zero-based. `INIT_FRAME=120` means row 120 in the selected
episode parquet. `--debug-replay-end-index` is exclusive. If the parquet uses a
non-default action column, set `INIT_ACTION_COLUMN` for wrapper init, or
`--init-action-column` / `--debug-replay-action-column` for direct runtime
commands.

The FR3 deployment action/state layout is:

```text
action = [arm_qpos7, right_hand6, left_hand6]
state  = [arm_qpos7, right_hand6, left_hand6]
```

`--enable-arm --arm-side right` controls the single FR3 arm. `--active-sides`
controls which InspireTac hands are active; it is `right` for the current
`chips_med` setup. FR3 dual-arm deployment is not implemented in this runtime;
`--active-sides both` means one FR3 arm plus both hands.

### Common Direct Runtime Setup

Pure frame movement and dataset replay use the runtime entry directly, because
the ManipRep wrappers always pass a model policy and require `CHECKPOINT`.

Run direct runtime commands from the ManipRep repo root:

```bash
source Scripts/load_local_env.sh
maniprep_activate_conda "${CONDA_ENV:-ManipRep_310}"
export PYTHONPATH="${MANIPREP_DEPLOY_PACKAGE_ROOT}:${PWD}:${PYTHONPATH:-}"
```

### 1. Smoothly Move to One Frame of a Trajectory

Use the built-in `hold` policy plus `--init-source dataset`. After startup init
finishes, the hold policy keeps sending the observed current state. Press `q` to
exit, or add a conservative `--max-steps` budget.

FR3 arm plus right hand:

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --arm-side right \
  --fr3-host "$FR3_HOST" \
  --camera-mode none \
  --hand-tactile-regions "" \
  --require-ready-observation \
  --init-source dataset \
  --init-dataset /abs/path/to/dataset \
  --init-episode 3 \
  --init-frame 120 \
  --init-time-s 5.0 \
  --hz 30 \
  --auto-start \
  --send-actions
```

For one parquet directly, replace `--init-dataset /abs/path/to/dataset` and
`--init-episode 3` with:

```bash
--init-parquet /abs/path/to/episode_000003.parquet
```

For both hands, change `--active-sides right` to `--active-sides both` and add
`--left-hand-ip "$LEFT_HAND_IP"`.

### 2. Smoothly Move to the First Frame, Then Replay at Real Frequency

Use runtime debug replay. `--init-source replay` makes startup init target the
first loaded replay action. Set `--hz` to the data collection frequency,
usually `30`; this is what makes replay run at the real frequency.

FR3 arm plus right hand:

```bash
python -m inspire_data_collect.deploy.run_policy \
  --debug-replay-dataset /abs/path/to/dataset \
  --debug-replay-episode 3 \
  --debug-replay-start-index 0 \
  --debug-replay-chunk-size 32 \
  --active-sides right \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --arm-side right \
  --fr3-host "$FR3_HOST" \
  --camera-mode none \
  --hand-tactile-regions "" \
  --require-ready-observation \
  --init-source replay \
  --init-time-s 5.0 \
  --hz 30 \
  --policy-hz 10 \
  --min-action-queue 8 \
  --auto-start \
  --send-actions
```

For direct parquet replay, replace `--debug-replay-dataset` and
`--debug-replay-episode` with:

```bash
--debug-replay-parquet /abs/path/to/episode_000003.parquet
```

By default replay holds the final action after the last frame. Add
`--debug-replay-loop` only when repeated replay is intended. For both hands,
change `--active-sides right` to `--active-sides both` and add
`--left-hand-ip "$LEFT_HAND_IP"`.

### 3. Smoothly Move to the First Frame, Then Start the Model

Use the ManipRep wrappers. Set `INIT_SOURCE=dataset`, `INIT_FRAME=0`, and pass
the same dataset/parquet used as the task start pose. After startup init
completes, `inspire_data_collect.deploy.run_policy` starts querying
`ManipRepFR3Policy`.

Tactile checkpoint:

```bash
CHECKPOINT=/abs/path/to/checkpoints/epoch_XXXX.pt \
CONDA_ENV=ManipRep_310 \
INIT_SOURCE=dataset \
INIT_DATASET=/abs/path/to/dataset \
INIT_EPISODE=3 \
INIT_FRAME=0 \
INIT_TIME_S=5.0 \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
SEND_ACTIONS=1 \
AUTO_START=1 \
bash PolicyDeploy/Model_FR3_deploy/run_tactile_policy.sh
```

No-tactile checkpoint:

```bash
CHECKPOINT=/abs/path/to/best.pt \
CONDA_ENV=ManipRep_310 \
INIT_SOURCE=dataset \
INIT_DATASET=/abs/path/to/dataset \
INIT_EPISODE=3 \
INIT_FRAME=0 \
INIT_TIME_S=5.0 \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
SEND_ACTIONS=1 \
AUTO_START=1 \
bash PolicyDeploy/Model_FR3_deploy/run_notactile_policy.sh
```

For one parquet directly:

```bash
INIT_PARQUET=/abs/path/to/episode_000003.parquet
```

with `INIT_DATASET` unset.

## Wrapper Interfaces

Both `run_tactile_policy.sh` and `run_notactile_policy.sh` are configured through environment variables and still forward any extra CLI args after the script to `inspire_data_collect.deploy.run_policy`.

| Env var | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT` | required | Model checkpoint path. Aliases: `CKPT`, `MODEL_CKPT`. A checkpoint directory resolves to `last.pt`. |
| `DEVICE` | `cuda` | Torch device passed to the policy. |
| `RIGHT_HAND_IP` | empty | InspireTac right-hand IP. Required for real action sending. |
| `FR3_HOST` | `192.168.20.172` | FR3 host. |
| `CAMERA_KEY` | `front` | Camera key used by the policy and deploy loop. |
| `HZ` | `30` | Deploy control loop rate. |
| `POLICY_HZ` | `5` | Max policy query rate. |
| `MIN_ACTION_QUEUE` | `8` | Query policy when queued actions are at or below this length. |
| `ACTION_CHUNK_SIZE` | `all` | `all`/`0` uses the full model action chunk; a positive integer returns only the first N actions from each inference. |
| `APPEND_ACTION_CHUNKS` | `0` | `1` appends new chunks; default replaces the pending queue. |
| `MAX_FORCE` | `1000` | InspireTac startup max fingertip force. Aliases: `INSPIRE_HAND_MAX_FORCE`, `MAX_FINGERTIP_FORCE`. |
| `MAX_HAND_STEP` | `0.03` | Per-control-tick hand rate limit. `<=0` disables. |
| `MAX_ARM_STEP_RAD` | `0.02` | Per-control-tick FR3 joint rate limit. `<=0` disables. |
| `HAND_TACTILE_REGIONS` | `all` for tactile, empty for no-tactile | Tactile regions requested from the hand backend. |
| `REQUIRE_TACTILE` | `1` for tactile, `0` for no-tactile | Whether a tactile checkpoint errors when tactile is missing. |
| `REQUIRE_IMAGE` | `1` | Whether the policy errors when the configured camera image is missing. |
| `REQUIRE_CAMERA_FRAME` | `1` | Whether the deploy loop marks missing camera frames as observation faults. |
| `REQUIRE_READY_OBSERVATION` | `1` | Whether observation faults prevent policy execution and action sending. |
| `SEND_ACTIONS` | `0` | `1` actually sends filtered actions to hardware. |
| `AUTO_START` | `0` | `1` starts immediately instead of waiting for `g`. |
| `INIT_TIME_S` | `5.0` | Smooth initialization duration before replay/policy execution. `0` disables. |
| `INIT_SOURCE` | `auto` | `auto`, `current`, `dataset`, `replay`, or `none`. `auto` uses replay first frame when replaying, an explicit init dataset when provided, otherwise current state. |
| `INIT_DATASET` / `INIT_PARQUET` | empty | Optional dataset root or episode parquet for deploy initialization from a recorded frame. |
| `INIT_EPISODE` / `INIT_FRAME` | `0` / `0` | Dataset episode and frame used when `INIT_SOURCE=dataset` or `INIT_SOURCE=auto` with `INIT_DATASET`/`INIT_PARQUET`. |
| `INIT_ACTION_COLUMN` | `action` | Full action column used for init/replay parquet loading. |

To execute the full returned model chunk before querying again, use `ACTION_CHUNK_SIZE=all MIN_ACTION_QUEUE=0`.
To deliberately use only the first few model outputs, set `ACTION_CHUNK_SIZE=N`; keep `MIN_ACTION_QUEUE` lower than `N`, or set `APPEND_ACTION_CHUNKS=1` if you want to prefetch without discarding the tail of the current queue.

Typical short-horizon run:

```bash
CHECKPOINT=/abs/path/to/checkpoints/epoch_XXXX.pt \
ACTION_CHUNK_SIZE=4 \
MAX_FORCE=800 \
MIN_ACTION_QUEUE=2 \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
bash PolicyDeploy/Model_FR3_deploy/run_tactile_policy.sh
```

Tactile checkpoint:

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy "$PWD/PolicyDeploy/Model_FR3_deploy/policy.py:Policy" \
  --policy-kwargs-json '{"checkpoint": "/abs/path/to/checkpoints/epoch_XXXX.pt", "device": "cuda", "action_chunk_size": 4}' \
  --active-sides right \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --fr3-host "$FR3_HOST" \
  --camera-mode single \
  --camera-key front \
  --inspire-hand-max-force 800 \
  --require-camera-frame \
  --require-ready-observation \
  --hz 30 \
  --policy-hz 5 \
  --min-action-queue 8
```

No-tactile checkpoint:

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy "$PWD/PolicyDeploy/Model_FR3_deploy/policy.py:Policy" \
  --policy-kwargs-json '{"checkpoint": "/abs/path/to/best.pt", "device": "cuda"}' \
  --active-sides right \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --fr3-host "$FR3_HOST" \
  --camera-mode single \
  --camera-key front \
  --hand-tactile-regions "" \
  --require-camera-frame \
  --require-ready-observation
```

Add `--send-actions` only after dry-run status is stable. Without `--auto-start`, press `g` to start execution.

To initialize from a recorded frame before normal policy deploy:

```bash
INIT_SOURCE=dataset \
INIT_DATASET=/abs/path/to/dataset \
INIT_EPISODE=3 \
INIT_FRAME=0 \
bash PolicyDeploy/Model_FR3_deploy/run_tactile_policy.sh
```

## Offline Smoke

```bash
python -m PolicyDeploy.Model_FR3_deploy.offline_smoke \
  --checkpoint Outputs/vanilla-tactile/<run_name>/checkpoints/epoch_XXXX.pt \
  --device cpu \
  --action-chunk-size 4
```

Expected output includes `action_shape (N, 19)` and `action_finite True`; `N` is the model horizon unless `--action-chunk-size` truncates it.

## Notes

- The current `chips_med` training data uses only the right hand plus FR3 arm. Use `--active-sides right` for the first deployments.
- The deploy adapter always receives right-hand raw Inspire tactile shape `(1062,)`. It reads the checkpoint data summary to decide whether the trained policy expects raw tactile `(1062,)` or union UV tactile `(351,)`, and applies the same tactile min-max normalization parameters used in training.
- Vision is the same lightweight token used in training: RGB mean plus temporal position, shape `(1, 4)`.
- `train.epochs` checkpoints without validation should be selected from periodic `checkpoints/epoch_XXXX.pt` after manual inspection.
