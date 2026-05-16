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

Image transport notes:

- Being-H-Tac inference uses the eval vision transform
  `Resize(force_image_size) -> CenterCrop(force_image_size)`.
- This client now applies the same resize/crop logic before sending images to
  the server, controlled by `--transport-image-size` and defaulting to `224`.
- Set `--transport-image-size 448` for 448px checkpoints, or
  `--transport-image-size 0` to disable client-side resizing.

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
  --send-actions \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --which-episode 5 \
  --transport-image-size 224


python run_client_realman22.py \
  --host 127.0.0.1 \
  --source dataset \
  --dataset-path /home/ai/Desktop/Being-H-series/2026-04-28_large_brown_paper_fold_conerted \
  --max-requests 100 \
  --dataset-frame-stride 24 \
  --action-hz 12.0 \
  --send-actions \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --left-arm-ip "$LEFT_ARM_IP" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --transport-image-size 224



ssh -N -L 5555:127.0.0.1:5555 root@36.103.197.253 -v

python run_client_franka.py \
  --host 127.0.0.1 \
  --source dataset \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/real_dataset/2026-04-27_chips_single_converted \
  --max-requests 100 \
  --dataset-frame-stride 24 \
  --action-hz 12.0 \
  --send-actions \
  --right-arm-ip "192.168.20.172" \
  --right-hand-ip "192.168.11.210" \
  --transport-image-size 224 \
  --startup-init-steps 48

python run_client_franka.py \
  --host 127.0.0.1 \
  --source dataset \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/real_dataset/2026-05-07_peeling_white_carrot_converted \
  --max-requests 100 \
  --dataset-frame-stride 24 \
  --action-hz 12.0 \
  --send-actions \
  --right-arm-ip "192.168.20.172" \
  --right-hand-ip "192.168.11.210" \
  --transport-image-size 224 \
  --startup-init-steps 48 \
  --fixed-hand "(0.096, 0.057, 0.022, 0.056, 0.495, 0.0)" \
  --which-episode 106


python run_client_gripper_DM.py \
  --host 127.0.0.1 \
  --source dataset \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/datasets/2026-05-12_gripper_collect_peeling_official \
  --max-requests 100 \
  --dataset-frame-stride 24 \
  --camera-mode single \
  --camera-serial 135122074814 \
  --active-sides right \
  --right-arm-ip "192.168.20.172" \
  --enable-hands \
  --action-hz 12.0 \
  --auto-start \
  --transport-image-size 224 \
  --startup-init-steps 48 \
  --send-actions \
  --which-episode 15 \
  --max-arm-joint-step-rad 0.08


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

## Startup Dataset Replay Gate

By default, every `run_client_*` waits before opening the server-client
inference loop:

- Press `r` to randomly choose one parquet episode from `--dataset-path`, replay
  all dataset actions to the robot at `--action-hz`, then exit.
- Press `i` to randomly choose one parquet episode, read its first
  `action.replay.*_arm_joint_target` / `action.replay.*_hand_joint_target`, send
  that startup target to the robot through a short joint-space interpolation,
  sleep `--startup-init-sleep-s` seconds, then continue into the requested
  `--source dataset` or `--source live` inference mode. Tune the initialization
  motion with `--startup-init-steps` and `--startup-init-hz`.

This startup gate uses the dataset action columns directly; it does not query
the Being-H-Tac server. Add `--no-startup-key-wait` for non-interactive tests.
Use `--startup-replay-seed` for repeatable random episode selection and
`--startup-replay-start-frame` if the replay should start from a frame other
than 0. Use `--which-episode k` to select the k-th sorted parquet episode
explicitly; otherwise startup replay chooses one at random. If `i` is used in
dataset mode, the subsequent server-client dataset replay is bound to the same
episode selected for initialization.

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
  --action-hz 12.0 \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions all \
  --auto-start \
  --transport-image-size 224 \
  --max-fingertip-force 600 \
  --send-actions


python run_client_realman22.py \
  --host 127.0.0.1 \
  --source live \
  --camera-mode single \
  --camera-serial 213222079798 \
  --active-sides both \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-arm-ip "$LEFT_ARM_IP" \
  --enable-hands \
  --action-hz 12.0 \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --hand-tactile-regions all \
  --auto-start \
  --transport-image-size 224 \
  --max-fingertip-force 800 \
  --send-actions \
  --which-episode 152
  


python run_client_franka.py \
  --host 127.0.0.1 \
  --source live \
  --camera-mode single \
  --camera-serial 135122074814 \
  --active-sides right \
  --right-arm-ip "192.168.20.172" \
  --enable-hands \
  --action-hz 12.0 \
  --right-hand-ip "192.168.11.210" \
  --hand-tactile-regions all \
  --auto-start \
  --transport-image-size 224 \
  --max-fingertip-force 400 \
  --send-actions \
  --which-episode 110



ssh -N -L 5555:127.0.0.1:5555 root@36.103.197.253 -v

python run_client_franka.py \
  --host 127.0.0.1 \
  --source live \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/real_dataset/2026-04-27_chips_single_converted \
  --camera-mode single \
  --camera-serial 135122074814 \
  --active-sides right \
  --right-arm-ip "192.168.20.172" \
  --enable-hands \
  --action-hz 12.0 \
  --right-hand-ip "192.168.11.210" \
  --hand-tactile-regions all \
  --auto-start \
  --transport-image-size 224 \
  --max-fingertip-force 400 \
  --send-actions \
  --which-episode 110

python run_client_franka.py \
  --host 127.0.0.1 \
  --source live \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/real_dataset/2026-05-07_peeling_white_carrot_converted \
  --camera-mode single \
  --camera-serial 135122070754 \
  --active-sides right \
  --right-arm-ip "192.168.20.172" \
  --enable-hands \
  --action-hz 12.0 \
  --right-hand-ip "192.168.11.210" \
  --hand-tactile-regions all \
  --auto-start \
  --transport-image-size 224 \
  --max-fingertip-force 400 \
  --send-actions \
  --fixed-hand "(0.096, 0.057, 0.022, 0.056, 0.495, 0.0)" \
  --which-episode 110

python run_client_franka.py \
  --host 127.0.0.1 \
  --source live \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/datasets/2026-05-08_waping_random \
  --camera-mode single \
  --camera-serial 135122074814 \
  --active-sides right \
  --right-arm-ip "192.168.20.172" \
  --enable-hands \
  --action-hz 12.0 \
  --right-hand-ip "192.168.11.210" \
  --hand-tactile-regions all \
  --auto-start \
  --transport-image-size 224 \
  --max-fingertip-force 400 \
  --send-actions \
  --which-episode 110


python run_client_gripper_DM.py \
  --host 127.0.0.1 \
  --source live \
  --dataset-path /home/user/zc/dex_inspire_teleop_new/datasets/2026-05-12_gripper_collect_peeling_official \
  --camera-mode single \
  --camera-serial 135122070754 \
  --active-sides right \
  --right-arm-ip "192.168.20.172" \
  --enable-hands \
  --action-hz 12.0 \
  --auto-start \
  --transport-image-size 224 \
  --send-actions \
  --which-episode 15


```










In live mode, RealSense preview opens in a separate window by default. Disable
it with `--no-live-camera-preview`.

~~Live Realman22 dry-run, without sending actions:~~

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
