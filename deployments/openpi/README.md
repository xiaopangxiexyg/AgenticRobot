# DexTeleop OpenPI Deployment

This directory contains the OpenPI deployment integration.

- `src/`: symlink to the OpenPI source tree. It owns model loading and serving.
- `client/`: DexTeleop-side OpenPI adapters and launchers.
- `../client/`: model-agnostic client contracts, remote-policy glue, and unified action chunks.

The supported LAN topology is:

```text
GPU / model server machine
  OpenPI server, checkpoint, websocket port

Robot / client machine
  DexTeleop real observation, cameras, robot backends, safety checks, dispatch
```

The real runtime does not import OpenPI. The OpenPI client converts model output
to the unified dense deployment action space, then `teleop_runtime` consumes it
like any other policy output.

## 1. Network Assumptions

Both machines must be on the same LAN or otherwise routable:

- server can bind and listen on `PORT`, usually `8078`
- client can reach `SERVER_IP:PORT`
- robot arm / hand / camera devices are connected to, or reachable from, the client machine
- only model inference runs on the server machine

On the server machine, find the LAN IP:

```bash
hostname -I
ip -4 addr
```

Use the address that the client can reach, for example:

```bash
export SERVER_IP=192.168.20.74
export OPENPI_PORT=8078
export OPENPI_MODEL=dual_rm65_dmgripper_latest
```

If a firewall is enabled on the server, allow the port, for example:

```bash
sudo ufw allow 8078/tcp
```

## 2. Server Machine

Start from the OpenPI tree exposed through `deployments/openpi/src`:

```bash
cd /home/zc/HandMotion/Developing/DexTeleop/deployments/openpi/src
```

Check the registered model:

```bash
sed -n '1,200p' configs/dexteleop_policy_registry.json
```

Start the server for LAN access. The important part is `--host 0.0.0.0`; this
binds the websocket server to all network interfaces rather than localhost.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 uv run scripts/serve_dexteleop_policy.py \
  --model "$OPENPI_MODEL" \
  --host 0.0.0.0 \
  --port "$OPENPI_PORT"
```

To override the registry checkpoint at launch time:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 uv run scripts/serve_dexteleop_policy.py \
  --model "$OPENPI_MODEL" \
  --checkpoint-dir /abs/path/to/checkpoint/100000 \
  --host 0.0.0.0 \
  --port "$OPENPI_PORT"
```

The server exposes both websocket inference and simple HTTP checks:

```bash
curl http://127.0.0.1:8078/healthz
curl http://127.0.0.1:8078/metadata
```

The metadata should include at least:

- `protocol_version=openpi.local_policy.websocket.v1`
- `deploy_schema_version=dexteleop_deploy_v1`
- `model_id`
- `deployment.checkpoint_dir`
- `deployment.checkpoint_step`
- `model.action_horizon`
- `local_robot.output_action_dim`
- `local_robot.active_sides`
- `dataset_adapter.camera_slots`
- `dataset_adapter.hand_layout`

## 3. Client Machine: Connectivity Checks

Start from DexTeleop:

```bash
export SERVER_IP=192.168.20.74
export OPENPI_PORT=8078
export OPENPI_MODEL=dual_rm65_dmgripper_latest
```

Check basic network reachability:

```bash
ping "$SERVER_IP"
curl "http://$SERVER_IP:$OPENPI_PORT/healthz"
curl "http://$SERVER_IP:$OPENPI_PORT/metadata"
```

Check the OpenPI client entry:

```bash
PYTHONPATH=. uv run python -m deployments.openpi.client.real --help
```

Check websocket metadata and contract validation without running robot control:

```bash
PYTHONPATH=. uv run python -c 'import os; from deployments.openpi.client.policy import create_policy; p=create_policy(host=os.environ["SERVER_IP"], port=int(os.environ["OPENPI_PORT"]), expected_model_id=os.environ["OPENPI_MODEL"]); print(p.name); print(p.metadata["validation"])'
```

This command connects to the server, reads metadata, validates the protocol and
unified action-space dimensions, then exits. It does not read cameras or move
hardware.

## 4. Client Dry-Run With Real Observation

Dry-run performs local observation collection, server inference, action
normalization, safety filtering, and queue handling. It does not send robot
commands because `--send-actions` is omitted.

For a single local RealSense config:

```bash
PYTHONPATH=. uv run python -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --model "$OPENPI_MODEL" \
  --active-sides both \
  --enable-hands \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --camera-mode single \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --require-camera-frame \
  --require-ready-observation \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --init-source none \
  --auto-start \
  --max-steps 100
```

For multi-camera deployment, change only the camera mode/config:

```bash
--camera-mode multi \
--camera-config cameras/configs/rm65/devices/realsense.json
```

The OpenPI client expects a `front` image by default. If the local camera config
uses a different key, either rename it to `front` in the camera config or use
the direct policy-spec form with an explicit `image_key_map`.

## 5. Client Real Execution

Only add `--send-actions` after metadata check and dry-run look correct. Real
arm execution requires a recorded dataset-frame qpos init by default. To
intentionally start from the current qpos without dataset init, add
`--start-init_qpos` as an explicit safety override.

Example dual-arm + DMTAC gripper deployment:

```bash
export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

PYTHONPATH=. uv run python -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --model "$OPENPI_MODEL" \
  --active-sides both \
  --enable-hands \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --camera-mode multi \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --require-camera-frame \
  --require-ready-observation \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --init-source current \
  --wait-for-ready-confirm \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/2026-07-05_vive_rm65_real_teleop \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 1 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-arm-max-angvel 0.25,0.25,0.25,0.35,0.35,0.35 \
  --arm-workspace-min -0.80 -0.80 0.00 \
  --arm-workspace-max 0.80 0.80 0.90 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --send-actions
```

To choose a different recorded start frame, change the normal DexTeleop deploy
qpos-init flags:

```bash
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-arm-max-angvel 0.25 \
  --init-source current \
  --wait-for-ready-confirm
```

Then press `g` to move to the selected qpos frame; after startup init reaches
ready, press `A` to release policy inference/action execution.

Runtime keys are the normal `teleop_runtime.deploy` keys: `g` starts/resumes,
`A` confirms ready, `h` pauses and clears the action queue, and `q` exits.
See `teleop_runtime/deploy/README.md` for the full state/keyboard reference.

For Inspire hand deployment, provide hand IPs as well:

```bash
export LEFT_HAND_IP=192.168.12.210
export RIGHT_HAND_IP=192.168.11.210

PYTHONPATH=. uv run python -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --model "$OPENPI_MODEL" \
  --active-sides both \
  --enable-hands \
  --hand-backend inspire \
  --camera-mode single \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --require-camera-frame \
  --require-ready-observation \
  --init-source current \
  --wait-for-ready-confirm \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/2026-07-05_vive_rm65_real_teleop \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 1 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-arm-max-angvel 0.25,0.25,0.25,0.35,0.35,0.35 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --inspire-tac-max-fingertip-force 800 \
  --send-actions
```

The safety-critical runtime flags are on the client side. The server only
produces action chunks.

`--policy-qpos-init-arm-max-angvel` is optional. It sets the first-frame QPOS
initialization velocity limit in rad/s; pass one value for all six joints or six
comma-separated values for per-joint limits.

Real execution start sequence for the examples above:

1. Start the command and wait until `faults=none` and every required camera has
   nonzero frames.
2. Press `g` to start dataset-frame qpos init.
3. Wait for `policy_qpos_init=released`, then for `init=done:current` and
   `ready=wait:A`.
4. Press `A` to release policy inference and action execution.

If `--auto-start` is added, step 2 happens automatically, but `A` is still
required when `--wait-for-ready-confirm` is used. For unattended dry-run only,
use `--auto-start --no-wait-for-ready-confirm`. For real execution without a
dataset-frame qpos init, add `--start-init_qpos` only after manually confirming
the current qpos is a safe policy start state.

## 6. Direct Policy-Spec Form

The OpenPI launcher is a convenience wrapper. It expands to the normal
`teleop_runtime` policy loader:

```bash
PYTHONPATH=. uv run python -m teleop_runtime.deploy \
  --policy deployments.openpi.client.policy:create_policy \
  --policy-kwargs-json '{"host":"192.168.20.74","port":8078,"expected_model_id":"dual_rm65_dmgripper_latest"}' \
  --active-sides both \
  --enable-hands \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --camera-mode single \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --require-camera-frame \
  --require-ready-observation
```

Prefer `python -m deployments.openpi.client.real` for normal use because it
keeps OpenPI-specific arguments separate from real-runtime arguments.

## 7. Troubleshooting

If `curl http://$SERVER_IP:$OPENPI_PORT/healthz` fails:

- confirm the server was started with `--host 0.0.0.0`
- confirm `SERVER_IP` is the server LAN IP, not `127.0.0.1`
- confirm firewall rules allow `OPENPI_PORT`
- confirm both machines are on the same reachable network

If `/metadata` works but the client rejects the server:

- check `expected_model_id`
- check `local_robot.output_action_dim`
- check `local_robot.active_sides`
- check `dataset_adapter.camera_slots`
- check `dataset_adapter.hand_layout`

If dry-run connects but policy inference fails:

- confirm the client has a valid local camera frame
- confirm the camera key expected by the OpenPI adapter is present, usually `front`
- confirm the model was trained for the same unified action space

If real execution refuses to start:

- `--send-actions` requires arm IPs for all active sides
- real arm execution requires `--policy-qpos-init-dataset` /
  `--policy-qpos-init-parquet`, unless explicitly overridden with `--start-init_qpos`
- `qpos init planning failed before execution` means no command was sent; choose
  a safer `--policy-qpos-init-frame`, move to a safer pre-start qpos, or after
  checking clearance try `--policy-qpos-init-collision-scene-profile dual_arm_hand_proxy_open`
- `timed out while moving to dataset frame` means qpos init began but did not
  reach `--policy-qpos-init-qpos-tolerance` before `--policy-qpos-init-timeout-s`;
  check the reported max joint error, then increase timeout or adjust tolerance
  only after confirming the arm state is safe
- Inspire hands require hand IPs when `--send-actions` and `--enable-hands` are used
- DMTAC gripper deployment needs a valid `--dmtac-config`
- keep dry-run working before enabling `--send-actions`
