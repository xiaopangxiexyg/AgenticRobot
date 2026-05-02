# BeingH-Tac Deployment Clients

This folder contains client-side code for talking to the existing Being-H-Tac
ZeroMQ inference servers while using either converted parquet episodes or live
DexTeleop hardware observations.

## What The Upstream Configs Mean

The relevant upstream definitions live in:

- `/home/ai/Desktop/Being-H-series/Being-H-Tac/configs/data_config.py`
- `/home/ai/Desktop/Being-H-series/Being-H-Tac/BeingH/inference/real_robot_client.py`
- `/home/ai/Desktop/Being-H-series/Being-H-Tac/BeingH/inference/real_robot_server.py`

The real-robot configs expose compact modality keys to the model:

- `franka`: one arm plus right Inspire hand. State/action are
  `eef_position(3) + eef_rotation(3 after transform from quat) + dexhand_position(6)`.
- `realman20`: right RM arm plus right Inspire hand, same compact right-side
  mapping as `franka`.
- `realman21`: right RM arm plus both Inspire hands. Adds
  `left_dexhand_position(6)`.
- `realman22`: dual RM arms plus both Inspire hands. Right side uses
  `eef_*` and `dexhand_position`; left side uses `left_eef_*` and
  `left_dexhand_position`.

All four use one RGB `video.front` input and tactile inputs
`tactile.right_tacglove` / `tactile.left_tacglove`. Converted training data uses
351D tactile vectors. Live InspireTac hardware reads 1062D vectors, so this
client can convert 1062D to the same 351D UV-union format before sending.

Important inference hyperparameters:

- `HISTORY_SIZE`: number of tactile history frames sent to the server. Existing
  real robot clients default to `4`.
- `HISTORY_STRIDE`: spacing in raw control steps between those history frames.
  Existing clients default to `4`.
- `ACTION_CHUNK_LENGTH`: number of future actions returned by the model. It is
  read by the server from the checkpoint config, commonly `16` or `24`; the
  client does not set it, but it consumes the returned chunk.
- `inference_delay`: RTC prefix-lock length. If non-zero, the client sends the
  previous `action_unified` as `prev_chunk` so the server can reuse the delayed
  prefix.

## DexTeleop Live Communication Summary

`teleop_runtime.deploy` builds observations from RM65 arm backends, InspireTac
hand backends, and optional RealSense RGB sources:

- Arm state comes from `RM65ArmBackend`: `current_qpos` and computed
  `current_tcp_pose`.
- Hand state comes from `InspireTacHandBackend`: `joint_pos` and optional
  tactile telemetry.
- Camera frames come from `RealSenseRGBSource.get_latest_frame()`.
- Deployment state layout is 38D:
  `[left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]`.
- Deployment action layout is 26D:
  `[left_pose7, right_pose7, left_hand6, right_hand6]`.

For sending actions, `teleop_runtime.deploy.actions` normalizes policy output to
that 26D action layout, applies workspace/rate-limit safety filters, then sends
cartesian arm pose commands and hand joint-vector commands to the hardware.

## Examples

Dataset mode against the remote Realman20 server:

```bash
conda activate grasp310_new
cd /home/ai/Desktop/DexTeleop/BeingH-Tac_deployment
python run_client_realman20.py \
  --host 127.0.0.1 \
  --source dataset \
  --dataset-path /home/ai/Desktop/Being-H-series/2026-04-30_heavy_light_converted \
  --max-requests 10
```

Dataset mode with real hardware action execution:

```bash
export RIGHT_ARM_IP=192.168.1.20
export RIGHT_HAND_IP=192.168.11.210

python run_client_realman20.py \
  --host 127.0.0.1 \
  --source dataset \
  --dataset-path /home/ai/Desktop/Being-H-series/2026-04-30_heavy_light_converted \
  --max-requests 100 \
  --dataset-frame-stride 24 \
  --action-hz 12.0 \
  --send-actions
```

In this mode, observations still come from the dataset, but the returned actions
are safety-filtered against current live RM65/InspireTac state before they are
sent to hardware. By default dataset observations are read frame-by-frame
(`0, 1, 2, ...`). Use `--dataset-frame-stride 24` for a 24-step action-chunk
policy to query `0, 24, 48, ...`, which better matches replaying one full chunk
before asking the model again. Without `--dataset-loop`, the client stops at the
end of the selected parquet episode instead of wrapping to the start.

Action chunks are dispatched by a background consumer at `--action-hz`
Hz, default `12.0`, so a 24-step model chunk is executed over roughly two
seconds. When a new chunk arrives before the previous queue is empty, the
default `--action-queue-policy replace` discards the old tail and uses the
newer chunk. Use `--action-queue-policy append` only if you explicitly want
open-loop playback with growing latency. If the queue runs dry before the next
inference result arrives, the dispatcher holds the last target and prints an
underrun warning.

Live Realman20 with RealSense RGB and sending actions:

```bash
python run_client_realman20.py \
  --host 127.0.0.1 \
  --source live \
  --camera-mode single \
  --camera-serial 213222079798 \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions all \
  --auto-start \
  --send-actions
```

Live Realman22 dry-run, without sending actions:

```bash
python run_client_realman22.py \
  --host 127.0.0.1 \
  --source live \
  --camera-mode single \
  --camera-serial 213222079798 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions all \
  --auto-start
```

Add `--send-actions` only when the robot is ready and the workspace/safety
limits are set correctly.
