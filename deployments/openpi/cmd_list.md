# OpenPI Policy → DexTeleop 部署命令

本文档汇总远端 `/home/user/zc/openpi` 当前可用于 DexTeleop 的 policy、可选
checkpoint step、OpenPI server、契约检查、无动作 dry-run 和真实 RM65 阻抗/力控
部署命令。

本文档只记录命令，没有启动训练、OpenPI server、相机、机械臂或夹爪。

## 1. 使用规则

- `POLICY_STEP` 必须对应 checkpoint 目录中实际存在的数字 step，不能任意填写。
- artifact-backed policy 使用 `deployments.openpi.server`：它根据 checkpoint 下的
  `policy_artifact.json` 生成临时 registry，不修改正式 registry。
- 下游默认只校验 OpenPI/DexTeleop 协议兼容性；`model_id` 和
  `checkpoint_step` 是可选 provenance，不作为默认部署门槛。
- server 与 DexTeleop client 可以在同一台机器上运行。跨机器时，client 的
  `SERVER_IP` 不能使用 `127.0.0.1`，应改成 server 的可路由 IP。
- 先执行 contract check 和 no-action dry-run，再考虑 `--send-actions`。
- `--deployment-profile rm65_bimanual_dmtac` 使用当前已检查的双臂 DMTAC、
  `impedance_40hz` 和 RealSense 多相机链路；它不会自动打开 `--send-actions`。

## 2. 当前 policy 清单

检查时间：2026-07-28。以下是当前远端已经有有效 checkpoint 的 policy。

| Policy/config | 数据与动作语义 | 相机 | 可选 checkpoint step | 部署方式 |
| --- | --- | --- | --- | --- |
| `0715-cube-front-ready-mix-4to1-h16-relative` | v2.3，relative，32D state / 16D action，h16，4:1 | front + left/right wrist | `30000, 60000, 90000, 120000, 150000` | artifact wrapper |
| `0715-cube-front-ready-mix-4to1-relative` | v2.3，relative，32D / 16D，h50，4:1 | front + left/right wrist | `30000, 60000, 90000, 149999` | artifact wrapper / registry |
| `0715-cube-front-ready-mix-4to1` | v2.2，absolute，32D / 16D，h50，4:1 | front + left/right wrist | `30000, 60000, 90000, 138257` | artifact wrapper |
| `0715-cube-front-ready-full-h16-absolute-v23` | v2.3，absolute，32D / 16D，h16，全视角-only | front + left/right wrist | `30000, 60000, 90000, 120000, 150000`（`-w2` 实验） | artifact wrapper |
| `0712-wrist-cube` | v2.2，legacy absolute，16D / 16D，h50 | left/right wrist | `30000, 60000, 90000, 92599` | artifact wrapper / registry |
| `0712-cube` | v2.2，legacy absolute，16D / 16D，h50 | front + left/right wrist | `11768` | artifact wrapper / registry |
| `writing-phase1-grap-pen-h16-relative` | v2.3，relative，32D / 16D，h16 | front + left/right wrist | `30000, 60000, 90000, 120000, 150000` | artifact wrapper |
| `dual_rm65_dmgripper` | 老 v1，26D action，h50 | front + left/right wrist | registry 固定 `99999` | legacy registry |

对应的推荐 checkpoint 路径如下：

```text
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/<STEP>
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-relative/0715-cube-front-ready-mix-4to1-relative/<STEP>
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1/0715-cube-front-ready-/<STEP>
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-full-h16-absolute-v23/0715-cube-front-ready-full-h16-absolute-v23-w2/<STEP>
/home/user/zc/openpi/checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/<STEP>
/home/user/zc/openpi/checkpoints/0712-cube/0712-cube/<STEP>
/home/user/zc/openpi/checkpoints/writing-phase1-grap-pen-h16-relative/writing-phase1-grap-pen-h16-relative/<STEP>
```

当前没有有效 checkpoint、因此暂不可部署的配置：

- `0715-cube-front-ready-mix-4to1-h16-absolute-v23`：当前只有实验目录，没有数字 step。
- `0715-cube-front-ready-mix-4to1-h16-absolute`：v2.2 absolute h16 配置，暂无 checkpoint。
- `0715-cube-front-ready-full-h16-relative`：全视角-only relative h16 配置，暂无 checkpoint。
- `writing-phase1-grap-pen-h16-absolute`：writing absolute h16 配置，暂无 checkpoint。
- `dual_rm65_dmgripper_v22_masked`：registry 指向的 checkpoint 当前不存在，不能部署。

## 3. 环境变量

先在 server/client 终端分别设置基础变量。若两端是同一台机器，可以使用下面默认值。

```bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
```

## 4. 选择 policy 和 checkpoint step

每次只执行下面一个 policy 选择块，然后执行第 5～8 节的通用命令。

### 4.1 cube 4:1，v2.3 relative h16

```bash
export OPENPI_CONFIG=0715-cube-front-ready-mix-4to1-h16-relative
export OPENPI_MODEL=0715-cube-front-ready-mix-4to1-h16-relative
export OPENPI_ACTION_HORIZON=16
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative
export OPENPI_EXPERIMENT=0715-cube-front-ready-mix-4to1-h16-relative
export POLICY_STEP=150000
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

可将 `POLICY_STEP` 改为 `30000`、`60000`、`90000` 或 `120000`。

### 4.2 cube 4:1，v2.3 relative h50

```bash
export OPENPI_CONFIG=0715-cube-front-ready-mix-4to1-relative
export OPENPI_MODEL=0715-cube-front-ready-mix-4to1-relative
export OPENPI_ACTION_HORIZON=50
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-relative
export OPENPI_EXPERIMENT=0715-cube-front-ready-mix-4to1-relative
export POLICY_STEP=149999
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

可将 `POLICY_STEP` 改为 `30000`、`60000` 或 `90000`。

### 4.3 cube 4:1，v2.2 absolute h50

```bash
export OPENPI_CONFIG=0715-cube-front-ready-mix-4to1
export OPENPI_MODEL=0715-cube-front-ready-mix-4to1
export OPENPI_ACTION_HORIZON=50
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1
export OPENPI_EXPERIMENT=0715-cube-front-ready-
export POLICY_STEP=138257
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

可将 `POLICY_STEP` 改为 `30000`、`60000` 或 `90000`。

### 4.4 cube 全视角-only，v2.3 absolute h16

```bash
export OPENPI_CONFIG=0715-cube-front-ready-full-h16-absolute-v23
export OPENPI_MODEL=0715-cube-front-ready-full-h16-absolute-v23
export OPENPI_ACTION_HORIZON=16
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-full-h16-absolute-v23
export OPENPI_EXPERIMENT=0715-cube-front-ready-full-h16-absolute-v23-w2
export POLICY_STEP=150000
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/sorted_v22/new_cube_front_ready
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

可将 `POLICY_STEP` 改为 `30000`、`60000`、`90000` 或 `120000`。

### 4.5 cube wrist-only，v2.2

```bash
export OPENPI_CONFIG=0712-wrist-cube
export OPENPI_MODEL=0712-wrist-cube
export OPENPI_ACTION_HORIZON=50
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/0712-wrist-cube
export OPENPI_EXPERIMENT=0712-wrist-cube-wrist-only
export POLICY_STEP=92599
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

可将 `POLICY_STEP` 改为 `30000`、`60000` 或 `90000`。

### 4.6 cube 三相机，v2.2

```bash
export OPENPI_CONFIG=0712-cube
export OPENPI_MODEL=0712-cube
export OPENPI_ACTION_HORIZON=50
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/0712-cube
export OPENPI_EXPERIMENT=0712-cube
export POLICY_STEP=11768
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

### 4.7 writing phase1，v2.3 relative h16

```bash
export OPENPI_CONFIG=writing-phase1-grap-pen-h16-relative
export OPENPI_MODEL=writing-phase1-grap-pen-h16-relative
export OPENPI_ACTION_HORIZON=16
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-phase1-grap-pen-h16-relative
export OPENPI_EXPERIMENT=writing-phase1-grap-pen-h16-relative
export POLICY_STEP=150000
export OPENPI_QPOS_DATASET=/home/user/zc/DexTeleop/datasets/writing_phase1_grap_pen
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

可将 `POLICY_STEP` 改为 `30000`、`60000`、`90000` 或 `120000`。

## 5. artifact-backed policy：启动 OpenPI server

必须先在 server 终端执行第 4 节的一个选择块。

### 5.1 只生成临时 registry，不启动 server

```bash
cd "$DEXTELEOP_ROOT"
export PYTHONPATH="$DEXTELEOP_ROOT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

/home/user/zc/openpi/.venv/bin/python -m deployments.openpi.server \
  --checkpoint-dir "$OPENPI_CKPT" \
  --openpi-root "$OPENPI_ROOT" \
  --config "$OPENPI_CONFIG" \
  --action-horizon "$OPENPI_ACTION_HORIZON" \
  --host 0.0.0.0 \
  --port "$OPENPI_PORT" \
  --dry-run
```

### 5.2 正式启动 server

```bash
cd "$DEXTELEOP_ROOT"
export PYTHONPATH="$DEXTELEOP_ROOT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

/home/user/zc/openpi/.venv/bin/python -m deployments.openpi.server \
  --checkpoint-dir "$OPENPI_CKPT" \
  --openpi-root "$OPENPI_ROOT" \
  --config "$OPENPI_CONFIG" \
  --action-horizon "$OPENPI_ACTION_HORIZON" \
  --host 0.0.0.0 \
  --port "$OPENPI_PORT"
```

另开终端检查 server：

```bash
curl --fail --silent "http://$SERVER_IP:$OPENPI_PORT/healthz"
curl --fail --silent "http://$SERVER_IP:$OPENPI_PORT/metadata" | python -m json.tool
```

metadata 至少应满足所选 artifact 的协议要求：deploy schema、action horizon、
state/action dimension、camera slots、layout 和 pose mode。server 的 `model_id` 与
`checkpoint_step` 默认允许变化；需要逐实例审计时再使用第 6 节的 strict 选项。

## 6. artifact contract check（无 server、无相机、无硬件）

在 client 终端重新执行第 4 节对应的选择块，然后运行：

```bash
cd "$DEXTELEOP_ROOT"
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --contract-check-only
```

该检查会校验 artifact、pose action mode、layout hash、camera mapping 和 state/action
protocol，不连接 websocket、相机、夹爪或机械臂。它不会把 checkpoint step 当作协议
门槛；失败时不能用
关闭校验或手动伪造 metadata 的方式绕过。

默认不会用 `model_id`/`checkpoint_step` 阻塞上游模型切换。如果需要审计某个精确
模型实例，再显式增加：

```bash
  --strict-model-identity \
  --model "$OPENPI_MODEL" \
  --checkpoint-step "$POLICY_STEP"
```

## 7. DexTeleop no-action dry-run

该命令会连接 OpenPI server 并读取实时相机/手部观测，执行推理、归一化、动作过滤和
力控部署链路，但不添加 `--send-actions`，因此不会向机械臂或夹爪下发动作。它仍可能
打开相机、DMTAC 和 RM65 backend，现场执行前必须确认硬件范围。

```bash
cd "$DEXTELEOP_ROOT"
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --deployment-profile rm65_bimanual_dmtac \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --policy-qpos-init-dataset "$OPENPI_QPOS_DATASET" \
  --policy-qpos-init-episode 0 \
  --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --max-steps 100 \
  --debug
```

若选用不同任务或 checkpoint，应先人工确认 `OPENPI_QPOS_DATASET` 的 episode/frame
是可安全到达的初始化姿态；`0/0` 只是命令示例。

## 8. DexTeleop 真实阻抗/力控部署（L5）

只有在第 6 节 contract check、server metadata 和第 7 节 no-action dry-run 均通过，
并完成现场安全确认后，才可执行下面命令。真实下发的唯一额外开关是
`--send-actions`。

```bash
cd "$DEXTELEOP_ROOT"
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --deployment-profile rm65_bimanual_dmtac \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --policy-qpos-init-dataset "$OPENPI_QPOS_DATASET" \
  --policy-qpos-init-episode 0 \
  --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --send-actions \
  --debug
```

该 profile 使用当前部署专用的 `impedance_40hz` 链路。不要改成 position 分支，也
不要额外添加 `--start-init_qpos`，除非重新完成安全审查。手动启动时仍须遵循当前
运行时的 ready/qpos-init/策略放行流程。

## 9. 老 v1 policy：`dual_rm65_dmgripper`

该 checkpoint 没有 `policy_artifact.json`，不能使用第 6 节 artifact contract check，
也不能用 `POLICY_STEP` 切换 step。当前 registry 固定使用 `99999`：

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model dual_rm65_dmgripper_latest \
  --host 0.0.0.0 \
  --port 8078
```

老 v1 client dry-run（不发送动作）：

```bash
cd /home/user/zc/DexTeleop
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python
export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --model dual_rm65_dmgripper_latest \
  --deployment-profile rm65_bimanual_dmtac \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --max-steps 100 \
  --debug
```

老 v1 真实下发只允许在完成独立的旧 registry/26D contract 检查后，给上一个命令
追加 `--send-actions`；它不属于当前 v2 artifact 兼容链路。

## 10. 暂无 checkpoint 的配置：训练命令

下面命令只用于以后生成 checkpoint，不会自动产生可部署 artifact。首次使用配置时
先计算该配置自己的 stats；h16、h50、v2.2 和 v2.3 的 stats 不要混用。

### cube 4:1，v2.3 absolute h16

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/compute_norm_stats.py \
  --config-name 0715-cube-front-ready-mix-4to1-h16-absolute-v23
uv run scripts/train.py 0715-cube-front-ready-mix-4to1-h16-absolute-v23 \
  --exp-name=0715-cube-front-ready-mix-4to1-h16-absolute-v23-rerun
```

### cube 4:1，v2.2 absolute h16

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/compute_norm_stats.py \
  --config-name 0715-cube-front-ready-mix-4to1-h16-absolute
uv run scripts/train.py 0715-cube-front-ready-mix-4to1-h16-absolute \
  --exp-name=0715-cube-front-ready-mix-4to1-h16-absolute
```

### cube 全视角-only，v2.3 relative h16

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/compute_norm_stats.py \
  --config-name 0715-cube-front-ready-full-h16-relative
uv run scripts/train.py 0715-cube-front-ready-full-h16-relative \
  --exp-name=0715-cube-front-ready-full-h16-relative
```

### writing phase1，v2.3 absolute h16

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/compute_norm_stats.py \
  --config-name writing-phase1-grap-pen-h16-absolute
uv run scripts/train.py writing-phase1-grap-pen-h16-absolute \
  --exp-name=writing-phase1-grap-pen-h16-absolute
```

### v2.2 masked baseline

当前 `dual_rm65_dmgripper_v22_masked` 的 registry checkpoint 路径无效，若重新训练：

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/compute_norm_stats.py --config-name dual_rm65_dmgripper_v22_masked
uv run scripts/train.py dual_rm65_dmgripper_v22_masked \
  --exp-name=bimanual_cube_v22_masked_150k-rerun
```

### 其它旧配置

如果确实需要旧的 front-only 配置，训练入口为：

```bash
cd /home/user/zc/openpi
uv run scripts/compute_norm_stats.py --config-name dual_rm65_dmgripper_front
uv run scripts/train.py dual_rm65_dmgripper_front \
  --exp-name=dual_rm65_dmgripper_front
```

## 11. 训练恢复规则

只有当对应 `--exp-name` 目录下已经有有效数字 checkpoint 时才加 `--resume`：

```bash
uv run scripts/train.py <config-name> \
  --exp-name=<existing-exp-name> \
  --resume
```

如果目录只有 `wandb/`、stats 或临时文件而没有数字 checkpoint，不要直接使用
`--resume`；使用新的 `--exp-name`，避免覆盖并行工作。训练 checkpoint 生成后，按
第 2 节把该 step 加入可部署清单，再执行第 4～8 节。
