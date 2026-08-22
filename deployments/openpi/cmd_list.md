# OpenPI 详细命令与 checkpoint 目录

本文档保留远端 `/home/user/zc/openpi` policy 接入
`/home/user/zc/DexTeleop` 的具体 checkpoint/profile、实验名和长命令。系统性的路线选择、
环境分工、副作用、恢复规则和部署阶梯统一以
[`COMMANDS.md`](./COMMANDS.md) 为入口；模型契约以 checkpoint 的
`policy_artifact.json` 和 [`README.md`](./README.md) 为准。

本文件中的 checkpoint 清单是带审计日期的站点快照，不是自动更新的 registry。执行前
必须重新检查远端身份、路径、实际 step、artifact/profile hash 和进程/GPU 状态。不要从
不同章节复制参数拼成未定义的 server/client 组合。

本文档中的命令仅供操作者按节执行；本次整理没有启动训练、OpenPI server、相机、
机械臂或夹爪。

## 1. 使用规则与路线

### 1.1 按 policy 类型选择路线

| 类型 | 使用章节 | 允许的入口 |
| --- | --- | --- |
| 普通 artifact-backed policy | 第 2～8 节 | 第 4 节选择变量；依次做 server dry-run、contract check、server、no-action，最后才是 L5 |
| prepared writing stage1 | 旧模型第 9 节；新 scenario 模型第 19 节 | checkpoint 与对应的 stage1 training profile 必须同时绑定 |
| prepared writing stage2 | 第 10 节 | checkpoint 与 stage2 training profile 必须同时绑定；连续任务使用 `--no-init-qpos` |
| prepared writing 当前 Stage1–4 组合 | 第 20 节 | 四个独立 checkpoint/profile；Stage1 qpos-init，Stage2–4 当前姿态接管 |
| trained model-RTC | 第 11 节 | 仅 conditioned no-action shadow；当前没有动作部署入口 |
| 老 v1 registry policy | 第 12 节 | legacy registry；不属于 v2 artifact contract |
| 原生/prepared/prompt-family 训练与恢复 | 第 13～18 节 | 只在明确要训练时使用，不是部署前置步骤；先按 `COMMANDS.md` 第 4～5 节选路线 |

不要把不同路线的 server/client 参数混合。例如 prepared policy 不能改用通用
`--config`，RTC shadow 不能复制 `--send-actions`。

### 1.2 通用规则

- `POLICY_STEP` 必须对应 checkpoint 目录中实际存在的数字 step，不能任意填写。
- artifact-backed policy 使用 `deployments.openpi.server`：它根据 checkpoint 下的
  `policy_artifact.json` 生成临时 registry，不修改正式 registry。
- 下游默认只校验 OpenPI/DexTeleop 协议兼容性；`model_id` 和
  `checkpoint_step` 是可选 provenance，不作为默认部署门槛。
- server 与 DexTeleop client 可以在同一台机器上运行。跨机器时，client 的
  `SERVER_IP` 不能使用 `127.0.0.1`，应改成 server 的可路由 IP。
- 先执行 contract check 和 no-action，再考虑 `--send-actions`。
- `--deployment-profile rm65_bimanual_dmtac` 使用当前已检查的双臂 DMTAC、
  `impedance_40hz` 和 RealSense 多相机链路；它不会自动打开 `--send-actions`。
- no-action 不等于无硬件：凡是连接真实 observation/backend 的命令，都可能打开相机、
  DMTAC 和 RM65 backend；只有明确写着 contract/config check 的命令才是不连接设备。
- `--no-init-qpos` 只跳过初始化轨迹，不会跳过 ready、碰撞、力限、观测新鲜度或
  dispatch 安全门，也不会延续前一个 client 的 action queue、model history 或 chunk。

## 2. 当前可部署 policy 与 checkpoint

基础清单检查时间：2026-08-06；writing stage2 行于 2026-08-08、新 Stage1 scenario 行于
2026-08-13 单独完成远端只读复核。
以下是已经有有效 checkpoint 的 policy。

| Policy/config | 数据与动作语义 | 相机 | 可选 checkpoint step | 部署方式 |
| --- | --- | --- | --- | --- |
| `0715-cube-front-ready-mix-4to1-h16-relative` | v2.3，relative，32D state / 16D action，h16，4:1 | front + left/right wrist | `30000, 60000, 90000, 120000, 150000` | artifact wrapper |
| `0715-cube-front-ready-mix-4to1-relative` | v2.3，relative，32D / 16D，h50，4:1 | front + left/right wrist | `30000, 60000, 90000, 149999` | artifact wrapper / registry |
| `0715-cube-front-ready-mix-4to1` | v2.2，absolute，32D / 16D，h50，4:1 | front + left/right wrist | `30000, 60000, 90000, 138257` | artifact wrapper |
| `0715-cube-front-ready-full-h16-absolute-v23` | v2.3，absolute，32D / 16D，h16，全视角-only | front + left/right wrist | `30000, 60000, 90000, 120000, 150000`（`-w2` 实验） | artifact wrapper |
| `0712-wrist-cube` | v2.2，legacy absolute，16D / 16D，h50 | left/right wrist | `30000, 60000, 90000, 92599` | artifact wrapper / registry |
| `0712-cube` | v2.2，legacy absolute，16D / 16D，h50 | front + left/right wrist | `11768` | artifact wrapper / registry |
| `writing-phase1-grap-pen-h16-relative` | v2.3，relative，32D / 16D，h16 | front + left/right wrist | `30000, 60000, 90000, 120000, 150000` | artifact wrapper |
| `writing-stage1-0804-0805-front-h16-absolute-prepared-v1` | v2.3 严格第一阶段，absolute，32D / 16D，h16，自然合并 0804/0805 | front (`base_0_rgb`) | `30000, 60000, 90000, 120000, 150000`（`-b1-nw4-meta-v2` 实验） | checkpoint-bound prepared wrapper |
| `writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-prepared-v1` | v2.3 Stage1 三类场景，absolute，32D / 16D，h16，13 来源按 1:4:6 重复 | front (`base_0_rgb`) | `30000, 60000`（`-b1-nw4-v1` 实验，2026-08-13 快照） | checkpoint-bound prepared wrapper；第 19 节 |
| `writing-stage2-0804-0805-front-h16-absolute-prepared-v1` | v2.3 严格第二阶段，absolute，32D / 16D，h16，自然合并 0804/0805 | front (`base_0_rgb`) | `30000, 60000, 90000, 120000, 150000`（`-b1-nw4-meta-v1` 实验） | checkpoint-bound prepared wrapper |
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
/home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2/<STEP>
/home/user/zc/openpi/checkpoints/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-prepared-v1/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-b1-nw4-v1/<STEP>
/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1/<STEP>
```

当前没有有效 checkpoint、因此暂不可部署的配置：

- `0715-cube-front-ready-mix-4to1-h16-absolute-v23`：当前只有实验目录，没有数字 step。
- `0715-cube-front-ready-mix-4to1-h16-absolute`：v2.2 absolute h16 配置，暂无 checkpoint。
- `0715-cube-front-ready-full-h16-relative`：全视角-only relative h16 配置，暂无 checkpoint。
- `writing-phase1-grap-pen-h16-absolute`：writing absolute h16 配置，暂无 checkpoint。
- `dual_rm65_dmgripper_v22_masked`：registry 指向的 checkpoint 当前不存在，不能部署。

## 3. 公共环境变量

先在 server/client 终端分别设置基础变量。若两端是同一台机器，可以使用下面默认值。

```bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
```

## 4. Policy 与 checkpoint step 选择

每次只执行下面一个 policy 选择块。普通 artifact policy 随后执行第 5～8 节；checkpoint-bound
prepared policy 不能使用通用 `--config` 路径，stage1 按第 9 节、stage2 按第 10 节执行。

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

### 4.8 writing stage2（0804/0805），front + absolute h16

```bash
export OPENPI_CONFIG=writing-stage2-0804-0805-front-h16-absolute-prepared-v1
export OPENPI_MODEL=writing-stage2-0804-0805-front-h16-absolute-prepared-v1
export OPENPI_ACTION_HORIZON=16
export OPENPI_CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1
export OPENPI_EXPERIMENT=writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
export POLICY_STEP=150000
export OPENPI_TRAINING_PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json
export OPENPI_CKPT="$OPENPI_CHECKPOINT_ROOT/$OPENPI_EXPERIMENT/$POLICY_STEP"
export OPENPI_ARTIFACT="$OPENPI_CKPT/policy_artifact.json"
```

远端只读复核时间为 2026-08-08：五个数字 checkpoint 均存在；最终 `150000` 目录及
`policy_artifact.json` 生成于 2026-08-07 09:21。可将 `POLICY_STEP` 改为 `30000`、
`60000`、`90000` 或 `120000`，但每个 step 都必须继续使用同一个 prepared training
profile 生成 checkpoint-bound deployment binding。

不要把本选择块的 `OPENPI_CONFIG` 传给第 5 节通用 `--config` server 命令；stage2 必须直接
使用第 10 节同时绑定 `--checkpoint-dir` 和 `--prepared-training-profile` 的命令。

## 5. artifact-backed policy：启动 OpenPI server

必须先在 server 终端执行第 4 节的一个选择块。

server/config 的无硬件 preflight 不在各运行章节重复。统一检查样例与不同 policy
路线需要替换的绑定参数见第 6 节。

### 5.1 正式启动 server

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

## 6. 统一无硬件检查样例

全篇只保留本节这一组无 server、无相机、无机器人检查样例。普通 artifact policy 在
client 终端重新执行第 4 节对应的选择块，然后运行：

```bash
cd "$DEXTELEOP_ROOT"
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --contract-check-only
```

该检查会校验 artifact、pose action mode、layout hash、camera mapping 和 state/action
protocol，不连接 websocket、相机、夹爪或机械臂。它不会把 checkpoint step 当作协议
门槛；失败时不能关闭校验或手动伪造 metadata。

其他路线复用同一个检查入口，只替换 binding 参数，不再重复整条命令：

| 路线 | 用于替换 `--policy-artifact-json ...` 的参数 |
| --- | --- |
| prepared stage1/stage2 | 对应的 `--checkpoint-dir` 与 `--prepared-training-profile` |
| model-RTC | prepared 两项，再加 `--model-rtc-profile` 和 `--rtc-profile` |

需要 server 侧 preflight 时，复用对应章节的正式 server 命令并添加一个退出型开关：
`--dry-run` 用于生成/显示临时 artifact 和 registry；prepared 路线还可改用
`--prepared-config-check-only` 在真实 OpenPI registry 中临时构造并校验派生 config。
两者都不加载模型、不监听端口，也不创建设备 backend；一次只使用一个。

默认不会用 `model_id`/`checkpoint_step` 阻塞上游模型切换。需要审计精确模型实例时，
在同一命令中增加 `--strict-model-identity`、`--model "$OPENPI_MODEL"` 和
`--checkpoint-step "$POLICY_STEP"`。

## 7. 统一 no-action 样例

全篇只保留本节这一条 no-action 样例。它会连接 OpenPI server 并读取实时相机/手部观测，
执行推理、归一化、动作过滤和力控部署链路，但不添加 `--send-actions`，因此不会向机械臂
或夹爪下发动作。它仍可能打开相机、DMTAC 和 RM65 backend，现场执行前必须确认硬件范围。

```bash
cd "$DEXTELEOP_ROOT"
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

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

其他运行路线不再重复 no-action 命令：复制该路线的正式 client 命令，删除
`--send-actions`，并添加 `--max-steps 100`。checkpoint/profile、控制分支、相机和
`--no-init-qpos` 等路线特有参数保持不变；position force guard 参数可以保留，也可在
只验证 observation/transport 时整组移除。缺少 `--send-actions` 才是“不下发动作”的
权威条件，`--max-steps 100` 只限制检查长度。

## 8. DexTeleop 真实阻抗/力控部署（L5）

只有在第 6 节 contract check、server metadata 和第 7 节 no-action dry-run 均通过，
并完成现场安全确认后，才可执行下面命令。真实下发的唯一额外开关是
`--send-actions`。

```bash
cd "$DEXTELEOP_ROOT"
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

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

该 profile 使用当前部署专用的 `impedance_40hz` 链路。默认保持 dataset-frame qpos-init；
只有连续任务明确需要从当前 qpos 接管、并重新完成当前姿态和 policy 首帧安全审查时，才使用
`--no-init-qpos` 显式覆盖。手动启动时仍须遵循当前运行时的 ready/策略放行流程。

## 9. Prepared writing stage1：checkpoint-bound 部署

该模型不能用 `--config` 手填一个同名 OpenPI config。server 与 client 都必须同时给出
checkpoint 和训练 profile；wrapper 会生成一次性 deployment artifact，并把以下内容绑定为
同一单元：checkpoint 原始 artifact、训练 profile、派生 config、asset id、front-only camera
slots、absolute/h16 action contract，以及该 checkpoint 中实际 `norm_stats.json` 的字节 hash。
任一项漂移都会在模型或硬件创建前失败。临时 artifact/registry 只写入 `/tmp`，不修改
checkpoint、OpenPI 正式 registry、数据集或任何遥操配置。

本路线仍须先完成第 6 节无硬件检查，但不在这里重复命令。将样例中的 artifact 参数替换为
本节正式命令所列的 stage1 `--checkpoint-dir` 和 `--prepared-training-profile`；server 侧
preflight 也直接复用同一组绑定。

正式运行时至少使用两个终端：

1. **终端 A（OpenPI 环境）**执行下面的完整 server 命令；它会持续占用该终端。
2. **终端 B（DexTeleop 环境）**用于 health/metadata 检查，以及之后的 client。
3. client 退出或需要重启时不必重启终端 A；更换 checkpoint/profile 时则必须先停止并
   重新启动 server，不能在同一 server 进程里混用绑定。

终端 A：启动 OpenPI server，持续提供 OpenPI h16 chunk。`--host` 的值必须是裸地址
`0.0.0.0`，不能写成 URL 或 Markdown 链接。

运行命令只包含环境激活、必要的 CUDA/JAX 环境变量和直接的 `python -m` 调用；不要再包
子 Shell、`set -euo pipefail`、`exec` 或路径变量。`Ctrl+C` 正常关闭 server 时，prepared
runtime 会完成清理并以 130 退出，不打印 Python traceback。

```bash
cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

python -m deployments.openpi.server \
  --checkpoint-dir /home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2/150000 \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 \
  --port 8078
```

终端 B 先按第 7 节规则，从下方正式 client 命令派生 100-step no-action：删除
`--send-actions`，添加 `--max-steps 100`。该检查会连接 server、front 相机和状态 backend，
但不会下发动作。

no-action 通过并退出后，仍在终端 B 执行下面的完整命令进入带 `--send-actions` 的正式
位控流程。此命令会连接真实设备并允许机器人运动，只能在完成现场安全检查、急停可用且
工作空间无人员/障碍后执行。启动后先按 `g` 执行 dataset-frame qpos-init，人工确认 ready
和实际姿态，再按 `A` 放行 policy。

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"

python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir /home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2/150000 \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage \
  --policy-qpos-init-episode 0 \
  --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-collision-check \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --send-actions \
  --debug
```

client 显式覆盖快捷 profile 中的执行分支为 `position_15_60_120hz`，最终解析为 15 Hz
policy/action、60 Hz pose-reference/双臂 IK、120 Hz CANFD qcmd，以及
`rm65_bimanual_position_v1` 正式 position control profile。60 Hz accepted-qref 默认并在
本命令中显式封印为 `comfort_75_v1`；精确回退只需将该参数改为 `hard_v1`，不会改变
120 Hz qcmd 的 hard execution envelope，也不会改变遥操默认值。DMTAC tool TCP 和 gripper
配置继续复用既有已审核参数；artifact 会把三相机配置裁剪为仅创建 `front`。qpos-init 使用
严格 cohort 中的 0804 Episode 0/frame 0。命令未启用 RTC，也没有设置
`--action-chunk-execution-steps`，因此每次完整消费 server 返回的 h16 chunk。
RealSense 启动严格复用正式真机遥操数采链路：`bgr8`、`bounded_first_frame`、按
`metadata.startup_priority` 串行启动、默认 0.5 秒错峰；首轮无首帧时冷却 1 秒后只关闭并
重开失败 pipeline 一次，不做 USB hardware reset。全部 pipeline 启动后还必须通过与数采
相同的连续 3 个新帧、帧龄不超过 0.1 秒的 startup self-check，失败会在 policy 循环前终止。
100-step 上限只保留在 no-action 检查中；正式位控命令不设置 `--max-steps`，不会把等待
`g`、qpos-init 和等待 `A` 的循环计数误当成 policy 执行寿命。chunk 边界允许短暂间断，
队列仍保持 `replace + min_queue=0` 的既有语义。正式位控显式启用独立 20 Hz RM65 tool-frame
受力监控；超力、连续读错或 stale 会锁存并发双臂 `Arm_Stop`，且必须重启并重新 qpos-init。
在真实位控进程中按 Ctrl+C 也会先明确请求双臂 `Arm_Stop`，确认回执后再完成其余资源清理。

本节正式 position client 命令及其按第 7 节派生的 no-action 版本默认启用 plan-cut splice governor 和首个 stream 的
measured-anchor acquisition，无需额外增加参数。首个 policy Pose 与初始化后的实测 Pose
不连续时，robot motion 会按每个 15 Hz source 间隔最多 5 mm、0.5° 插入有界 SE(3)
接近段；后续 source 时间间距保持不变。该首段处理与同步 chunk、RTC emission 共用同一
typed stream-start cut，且不改变碰撞检查边界。需要原子回退到启用前行为时，同时追加：

```bash
--no-position-plan-splice-governor \
  --no-position-stream-start-acquisition
```

只关闭 stream-start acquisition、保留普通 chunk splice 时，可以仅追加
`--no-position-stream-start-acquisition`。

### 9.1 连续任务从当前 qpos 接管（显式覆盖）

当上一个任务已经把机器人留在下一个任务可接受的起始姿态，且不希望重新执行
dataset-frame qpos 初始化轨迹时，可在第 9 节 client 命令末尾追加规范参数
`--no-init-qpos`。兼容拼写 `--no--init-qpos` 也会被接受，但新命令统一使用前者。

该参数在共享 runtime 参数解析完成后生效，因此会覆盖任意 deployment profile 或同一命令中
注入的 `--policy-qpos-init-dataset` / `--policy-qpos-init-parquet`；不需要删除 profile，也不会
创建 qpos-init controller。此时：

- 不加 `--auto-start` 时，`g` 只表示 start/resume，会直接进入当前姿态下的 startup/policy
  流程，不执行 dataset-frame qpos 轨迹。
- `rm65_bimanual_dmtac` 仍保留 `init-source=current` 和 ready confirm；当前姿态 startup
  检查完成后仍需按 `A`。`--no-init-qpos` 不会自动关闭这些门。
- `--auto-start`、`--no-wait-for-ready-confirm` 是独立开关，不由 `--no-init-qpos` 隐式启用。
- 碰撞检查、position force guard、observation freshness、IK/reference 和 dispatch 安全边界均不变。
- 该参数只解决“跳过任务间 qpos 初始化轨迹”；它不会保留或拼接上一个 client 的 action queue、
  model history 或未执行 chunk，也不构成跨 checkpoint/profile 的无缝轨迹 handoff。

这是 L5 当前姿态接管 override。使用前必须确认当前双臂 qpos、TCP、夹爪状态、工作空间和新
policy 首个 chunk 都处于允许范围；absolute policy 还必须确认当前状态属于其训练/部署起始分布。

### 9.2 可选断联 DMTAC 夹爪

第 9、10 节以及后续相同 prepared client 命令默认行为不变，仍连接并控制左右夹爪。如果
某个夹爪明确不在线，可在 client 命令末尾只追加以下一组参数；`left` 也可替换为 `right`
或 `both`：

```bash
--disconnected-gripper-sides left \
  --disconnected-gripper-state 0.0
```

`--disconnected-gripper-state` 是送给模型的固定假定状态，范围 `[0,1]`，`0=open`、
`1=closed`；默认值为 `0.0`。它会在 runtime 状态中标记为
`configured_disconnected_assumption`，不是硬件测量。选中侧不会打开 gripper 串口或 DM-Tac
触觉、不会启动重连、不会发送夹爪命令；策略产生的该侧夹爪动作记录为
`intentionally_disconnected_noop`，其余夹爪、双臂、相机、碰撞、force guard、ready 和
policy 流程保持不变。Stage1 dataset-frame qpos-init 会跳过断联侧的物理夹爪初始化，但仍
正常初始化双臂和在线夹爪。

不要用该选项掩盖一个本应在线但异常掉线的夹爪；这种情况仍应保持默认 fail-closed 行为并
排查硬件。假定状态必须与现场未连接夹爪的实际开合尽量一致，否则模型看到的状态与物理场景
不一致。

## 10. Prepared writing stage2：checkpoint-bound 部署

本节用于已经完整训练的旧 stage2：`writing.remove_brush_cap_and_return_cap`。远端
`dex-remote` 在 2026-08-08 的只读复核确认实验目录包含 `30000`、`60000`、`90000`、
`120000`、`150000` 五个数字 checkpoint；最终 `150000` 及其 238 KiB
`policy_artifact.json` 的时间为 2026-08-07 09:21。artifact 声明 v2.3、front-only、
absolute h16、32D state / 16D action 和 checkpoint step 150000。

该模型与 stage1 一样必须使用 checkpoint-bound prepared wrapper。server 与 client 都要同时
给出 checkpoint 和 stage2 training profile，不能手填 `--config` 绕过 prepared dataset、
normalization 和 artifact binding。

本路线仍须先完成第 6 节无硬件检查，但不在这里重复命令。将样例中的 artifact 参数替换为
本节正式命令所列的 stage2 `--checkpoint-dir` 和 `--prepared-training-profile`；server 侧
preflight 也直接复用同一组绑定。

这三项检查已于 2026-08-08 在 `dex-remote` 实际通过：prepared binding hash 为
`sha256:09952950eb62b5683074563c3de052dcda1981f8540a343d6bb460d78fb133ef`，checkpoint
内 norm-stats hash 为
`sha256:117e32c01836794c08e30a1d3f7ee78f6805c7ee6d9c3d7db395e0905a58e000`，client
contract-check 返回 `ok=true`。检查未加载模型、未监听端口，也未创建相机或机器人 backend。

stage1 server 不能与 stage2 server 在同一个 `8078` 端口并存。由现有 server owner 正常按
Ctrl+C 停止 stage1 server，确认端口和显存释放后，在终端 A 启动 stage2 server：

```bash
cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

python -m deployments.openpi.server \
  --checkpoint-dir /home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1/150000 \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 \
  --port 8078
```

终端 B 先检查 stage2 loaded-instance health 和 metadata：

```bash
curl --fail --silent http://127.0.0.1:8078/healthz
curl --fail --silent http://127.0.0.1:8078/metadata | python -m json.tool
```

然后按第 7 节规则，从下方正式 client 命令派生 100-step no-action：删除
`--send-actions`，添加 `--max-steps 100`，其余参数包括 `--no-init-qpos` 保持不变。该检查会
连接真实 front 相机、双臂状态和 DMTAC，但不会构造 qpos-init controller，也不会下发动作。

no-action、server metadata 和现场安全检查全部通过后，才可执行下面的 L5 stage2 真机命令。
它从 stage1 留下的当前 qpos 接管，不回到任意数据集帧；启动后按 `g` 直接进入
current-pose startup，完成后仍需按 `A` 放行 policy：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir /home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1/150000 \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  --no-init-qpos \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --send-actions \
  --debug
```

`--no-init-qpos` 只跳过 qpos 初始化轨迹，不保留 stage1 client 的 action queue、model history 或
未执行 chunk。stage1 client 退出和 stage2 client 启动之间也不是无间断 command handoff；position
executor 会从 stage2 首个实测 cut 重新建立权威参考。默认启用的 stream-start measured-anchor
acquisition 和 plan-cut splice governor 会对首个 absolute Pose 接近段施加既有 5 mm / 0.5° 每
source interval 上限，但这不证明当前状态属于 stage2 训练分布。操作者必须另外确认 stage1 结束
状态、stage2 首个 chunk、夹爪和笔帽状态均匹配后，才能按 `A`。

## 11. Trained model-RTC：conditioned no-action shadow

本节只用于 trained model-RTC 的 conditioned no-action shadow，不是正式动作部署命令。
当前训练进程没有完成 150000 step；可验证的最新完整 checkpoint 是 30000。因此下面的
checkpoint 只能用于实现、时序和协议检查，不能当作最终训练结果。

trained model-RTC 必须同时绑定 prepared training profile、model-RTC training profile、
checkpoint artifact、training-run manifest、checkpoint receipt、immutable prepared dataset
binding 和 continuation contract。server 还必须显式传入 `--enable-rtc-continuation`；省略该
开关时仍保持普通 OpenPI 推理语义。client 侧 RTC profile 强制 `send_actions=false`，不得加入
`--send-actions`。

该模型和第 9 节的非 RTC prepared server 都需要大显存，不能在当前单卡上并行加载。启动本节
server 前，先由现有 server 的 owner 正常停止旧进程，并确认 `8078` 未监听且 GPU 显存已经
释放；不要由另一个终端强制结束不属于自己的进程。

本路线仍须先完成第 6 节无硬件检查，但不在这里重复命令。client contract check 复用本节
正式 server 命令中的 `--checkpoint-dir`、`--prepared-training-profile` 和
`--model-rtc-profile`，并增加 `--rtc-profile model_rtc_flow_inpaint_shadow_15_30_60_120_v1`。
server 侧 preflight 还必须保留 `--enable-rtc-continuation`。

终端 A：在确认旧 server 已停止、GPU 显存足够后启动 conditioned server：

```bash
cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

python -m deployments.openpi.server \
  --checkpoint-dir /home/user/zc/openpi/checkpoints/writing-stage1-0804-0806-front-h16-absolute-prepared-model-rtc-v1/writing-stage1-0804-0806-front-h16-absolute-model-rtc-b1-nw4-v1/30000 \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0806_front_h16_absolute_v1.json \
  --model-rtc-profile /home/user/zc/DexTeleop/deployments/openpi/configs/model_rtc/writing_stage1_0804_0806_front_h16_absolute_prepared_model_rtc_v1.json \
  --openpi-root /home/user/zc/openpi \
  --enable-rtc-continuation \
  --host 0.0.0.0 \
  --port 8078
```

终端 B：先检查 loaded-instance health 和 metadata；两项都通过后才进入 observation shadow：

```bash
curl --fail --silent http://127.0.0.1:8078/healthz
curl --fail --silent http://127.0.0.1:8078/metadata | python -m json.tool
```

终端 B 的 100-step observation/transport shadow 也复用第 7 节唯一 no-action 样例，按上段
替换三项 checkpoint/profile binding，并增加 `--rtc-profile
model_rtc_flow_inpaint_shadow_15_30_60_120_v1`、`--rtc-real-shadow`、
`--rtc-debug-level summary` 和 `--debug-dir
/home/user/zc/DexTeleop/logs/openpi_model_rtc_shadow`；删除普通样例中的 qpos-init 参数。
该流程会读取真实双臂、手和 front 相机 observation，并向 server 发 conditioned inference
请求，但不创建动作下发 authority。需要逐 tick 排查时才把 `summary` 改为 `trace`。

`--rtc-real-shadow` 是进入真实 observation 边界的显式开关：它移除 qpos-init authority，
禁止 `--send-actions`，并保持 position lookahead disabled。当前没有 trained model-RTC 的动作
部署入口；不得把第 9 节的 `--send-actions` 参数复制到本节。

## 12. Legacy v1：`dual_rm65_dmgripper`

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

老 v1 仍须使用第 7 节唯一 no-action 样例：将 artifact 参数替换为
`--model dual_rm65_dmgripper_latest`。完成独立的旧 registry/26D contract 检查和现场安全
确认后，真实下发命令为：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"
export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" \
  --port "$OPENPI_PORT" \
  --model dual_rm65_dmgripper_latest \
  --deployment-profile rm65_bimanual_dmtac \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --send-actions \
  --debug
```

该命令不属于当前 v2 artifact 兼容链路。要派生 legacy no-action，只删除
`--send-actions` 并添加 `--max-steps 100`，不要把 v2 artifact 参数复制进来。

## 13. 训练参考：当前暂无 checkpoint 的配置

本节保留的是 OpenPI 历史整数据集入口，不会按 prepared Skill 区间裁剪。训练
0804/0805 writing 子任务1应使用第 14 节，不能直接复用本节的 writing 命令。

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

## 14. 训练参考：0804/0805 writing stage1 prepared 矩阵

四个 profile 使用同一套严格 cohort：602 条成功且非异常 Episode（0804 为 344，
0805 为 258）。远端审计确认全部 602 条都具备 front、left wrist 和 right wrist，
因此 front/full 的 Episode 集合一致；两日期按有效窗口自然合并，不做数据源等权
重复采样。

| 组合 | Profile | OpenPI base | 默认实验名 |
| --- | --- | --- | --- |
| front + absolute | `writing_stage1_0804_0805_front_h16_absolute_v1.json` | front task-ready state-dropout-v2 | `writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2` |
| front + relative | `writing_stage1_0804_0805_front_h16_relative_v1.json` | front task-ready state-dropout-v2 | `writing-stage1-0804-0805-front-h16-relative-b1-nw4-meta-v2` |
| full + absolute | `writing_stage1_0804_0805_full_h16_absolute_v1.json` | full corrected-v1 | `writing-stage1-0804-0805-full-h16-absolute-nw4-meta-v2` |
| full + relative | `writing_stage1_0804_0805_full_h16_relative_v1.json` | full corrected-v1 | `writing-stage1-0804-0805-full-h16-relative-nw4-meta-v2` |

四个 profile 都是 π0.5 LoRA、h16、batch size 1、4 workers、150k steps，并显式
复用 state-dropout-v2：10% whole-state dropout 和 15% 独立 arm-pose dropout。
front 使用 100% procedural 背景；full 使用 corrected-v1 的 10% original、40%
real-image、50% procedural 三视角背景增强。

先设置公共变量：

```bash
cd /home/user/zc/DexTeleop
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE_ROOT=deployments/openpi/configs/training
```

每次只选择下面一组：

```bash
# front + absolute
export PROFILE=$PROFILE_ROOT/writing_stage1_0804_0805_front_h16_absolute_v1.json
export EXP=writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2
```

```bash
# front + relative
export PROFILE=$PROFILE_ROOT/writing_stage1_0804_0805_front_h16_relative_v1.json
export EXP=writing-stage1-0804-0805-front-h16-relative-b1-nw4-meta-v2
```

```bash
# full（三视角）+ absolute
export PROFILE=$PROFILE_ROOT/writing_stage1_0804_0805_full_h16_absolute_v1.json
export EXP=writing-stage1-0804-0805-full-h16-absolute-nw4-meta-v2
```

```bash
# full（三视角）+ relative
export PROFILE=$PROFILE_ROOT/writing_stage1_0804_0805_full_h16_relative_v1.json
export EXP=writing-stage1-0804-0805-full-h16-relative-nw4-meta-v2
```

对选中的 profile 依次执行：

```bash
# 1. 只读审计 eligibility；不写 prepared view
$OPENPI_PYTHON -m deployments.openpi.training.prepared_skill_launch \
  audit --profile "$PROFILE" --no-motion

# 2. 生成/复用 immutable prepared view 和 OpenPI binding
$OPENPI_PYTHON -m deployments.openpi.training.prepared_skill_launch \
  prepare --profile "$PROFILE" --no-motion

# 3. 无图像、无模型 sample-check；prepare 已完成完整源文件校验
$OPENPI_PYTHON -m deployments.openpi.training.prepared_skill_launch \
  sample-check --profile "$PROFILE" \
  --skip-source-file-verification --no-motion

# 4. 每个 profile 首次训练前分别生成 norm-stats
# 43860 个有效窗口可被 batch size 204 整除，不遗漏尾部样本
$OPENPI_PYTHON -m deployments.openpi.training.prepared_skill_launch \
  norm-stats --profile "$PROFILE" \
  --stats-batch-size 204 --stats-num-workers 0 \
  --skip-source-file-verification --no-motion

# 5. 正式训练；重新执行完整源文件校验
CUDA_VISIBLE_DEVICES=0 $OPENPI_PYTHON \
  -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE" --exp-name "$EXP" --no-motion
```

不要并行运行四组 norm-stats 或训练。不同 profile 使用独立 OpenPI config、asset id、
workspace 和 checkpoint 目录；relative 不得复用 absolute 的 norm-stats。

## 15. 训练恢复规则

只有当对应 `--exp-name` 目录下已经有有效数字 checkpoint 时才加 `--resume`：

```bash
uv run scripts/train.py <config-name> \
  --exp-name=<existing-exp-name> \
  --resume
```

如果目录只有 `wandb/`、stats 或临时文件而没有数字 checkpoint，不要直接使用
`--resume`；使用新的 `--exp-name`，避免覆盖并行工作。训练 checkpoint 生成后，按
第 2 节把该 step 加入可部署清单，再执行第 4～8 节。

第 14 节的 prepared 子任务训练必须通过同一 launcher 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 $OPENPI_PYTHON \
  -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE" --exp-name "$EXP" --resume --no-motion
```

## 16. 0804–0806 writing 四阶段联合 prompt 训练

该方案只训练一个 π0.5 LoRA 模型。四个子任务继续使用同一个父 `task_id`，但每个
合法动作窗口由 immutable binding 覆盖为各自的英文 prompt。不要复用原四个独立
stage checkpoint，也不要复用它们的 norm stats。

当前 binding 同时保留两类计数：prepared 标注窗口为
71,818 / 53,911 / 96,649 / 78,048；经过 OpenPI 的 `RUNNING`、有限值和
pose-quality 行级过滤后，实际训练窗口为
64,999 / 51,459 / 79,078 / 8,995。训练按 stage 等权采样，每个 epoch 从每个
stage 无放回选择 8,995 个窗口，共 35,980 个；下一 epoch 用固定 seed 轮换较大
stage 的子集。600,000 steps 等价于每个 stage 约 150,000 次样本暴露。

先设置公共变量：

```bash
cd /home/user/zc/DexTeleop
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE=deployments/openpi/configs/training/writing_multistage_prompt_0804_0806_front_h16_absolute_v1.json
```

首次训练前依次执行检查和该联合配置自己的 norm stats：

```bash
# 1. 只读重审四个 stage 的 eligibility 和固定 child binding
"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  audit --profile "$PROFILE"

# 2. 生成或复用联合 immutable binding；不启动训练
"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  prepare --profile "$PROFILE"

# 3. 不加载图像、不推理、不训练；验证四条 prompt 和平衡索引
"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  sample-check --profile "$PROFILE"

# 4. 35,980 可被 batch size 20 整除，不丢弃尾部样本
CUDA_VISIBLE_DEVICES=0 "$OPENPI_PYTHON" -u \
  -m deployments.openpi.training.prepared_skill_launch \
  norm-stats --profile "$PROFILE" \
  --stats-batch-size 20 --stats-num-workers 0 \
  --skip-source-file-verification
```

首次训练使用下面的前台交互命令。它不会后台化、不会 `exec` 替换 shell，也不经过
管道；终端保持正常交互，按 `S` 保存并继续，按一次 `Ctrl+C` 保存并退出：

```bash
cd /home/user/zc/DexTeleop
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE=deployments/openpi/configs/training/writing_multistage_prompt_0804_0806_front_h16_absolute_v1.json

"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE"
```

只有默认实验目录已经存在正整数 step checkpoint 时，才使用同一 profile 恢复：

```bash
cd /home/user/zc/DexTeleop
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE=deployments/openpi/configs/training/writing_multistage_prompt_0804_0806_front_h16_absolute_v1.json

"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE" --resume
```

默认实验名是
`writing-multistage-prompt-0804-0806-front-h16-absolute-b1-nw4-v1`。不要给首次
训练添加 `--resume`；launcher 会拒绝不存在或没有正整数 step 的 checkpoint 目录，
不会从一个空目录伪恢复。

## 17. 0804–0806 writing 四阶段 prompt-family 训练（推荐 v2）

v2 保留第 16 节单 prompt profile 作为完整回退，不覆盖其 workspace、asset、实验名或
binding。每个 stage 使用 1 条简短 canonical prompt、12 条训练同义句和 4 条 held-out
同义句。每个 8,995 样本的 stage/epoch 中，canonical 固定为 7,196（80%），12 条训练
同义句合计 1,799（20%，单条 149 或 150）；held-out 从不进入训练或生产部署 allowlist。

2026-08-11 已完成真实 OpenPI `prepare` 和 `sample-check`：binding hash 为
`1dd199320fafd3f4e53202a9b227b26ee604f12aa205d089d119d09f30597ce3`，每个 epoch
共 35,980 个唯一窗口。PaliGemma 对全部 68 条文本和 16 维全 `-1/0/1` 状态的 204 个
组合，截断前最长 103 token，低于 profile 限制 160 和模型上限 200。

检查、norm stats 和首次训练使用同一个 v2 profile：

```bash
cd /home/user/zc/DexTeleop
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE=deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0806_front_h16_absolute_v2.json

"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  prepare --profile "$PROFILE"

"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  sample-check --profile "$PROFILE"

CUDA_VISIBLE_DEVICES=0 "$OPENPI_PYTHON" -u \
  -m deployments.openpi.training.prepared_skill_launch \
  norm-stats --profile "$PROFILE" \
  --stats-batch-size 20 --stats-num-workers 0 \
  --skip-source-file-verification
```

本轮未运行上面的 norm stats。它成功完成后，首次训练使用前台交互命令；按 `S` 保存并
继续，按一次 `Ctrl+C` 保存并退出：

```bash
cd /home/user/zc/DexTeleop
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE=deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0806_front_h16_absolute_v2.json

"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE"
```

默认实验名是
`writing-multistage-prompt-family-0804-0806-front-h16-absolute-b1-nw4-v2`。只有该目录
已经存在正整数 step checkpoint 时才恢复：

```bash
cd /home/user/zc/DexTeleop
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export OPENPI_PYTHON=/home/user/zc/openpi/.venv/bin/python
export PROFILE=deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0806_front_h16_absolute_v2.json

"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE" --resume
```

产生 checkpoint 后，部署继续使用第 5～10 节的 checkpoint-bound prepared 路线，并把
`--prepared-training-profile` 指向本节 v2 profile。client 的 `--prompt` 必须精确选择
artifact 中某个 stage 的 canonical 或 12 条训练同义句；16 条 held-out 只用于离线泛化
评估，生产 client 会拒绝它们。

## 18. 0804–0807 writing 四阶段 prompt-family 训练与部署

本节对应
`writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json`。它保留第 9、10
节既有 checkpoint-bound prepared server/client 结构：训练得到一个共同 checkpoint，server
只加载一次该 checkpoint；四个阶段由 client 每次启动时选择文本条件，不创建四套 server，
也不改动既有机械臂、夹爪、相机、qpos-init、碰撞和 force-guard 参数。

### 18.1 训练命令

正式训练仍使用现有前台交互 launcher；按 `S` 保存并继续，按一次 `Ctrl+C` 保存并退出：

```bash
cd /home/user/zc/DexTeleop
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
/home/user/zc/openpi/.venv/bin/python -u -m deployments.openpi.training.prepared_skill_launch train \
  --profile deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json
```

该 profile 绑定 0804–0807 的 7 个数据源、1,217 个 union Episode 和 393,540 个合法
prepared 窗口；训练每个 effective epoch 平衡抽取 39,300 个窗口。四个 stage 均使用一个
canonical prompt、12 个训练同义句和 4 个 held-out 同义句；canonical/训练同义句按
80%/20% 进入训练，held-out 不进入训练和生产 allowlist。默认训练步数为 600,000，每
30,000 step 保存一次。不要给首次训练添加 `--resume`；恢复时仍只在实验目录已有正整数
step checkpoint 后给同一命令添加 `--resume`。

### 18.2 文本阶段选择契约

prepared deployment artifact 会绑定四个 stage、canonical prompt 和训练同义句 allowlist。
client 新增可选 `--prompt-stage STAGE_ID`，从 artifact 解析 canonical prompt：

| `STAGE_ID` | Skill | canonical prompt 摘要 |
| --- | --- | --- |
| `stage1` | `writing.bimanual_acquire_brush_and_present_for_cap_removal` | 右夹爪拿起带盖毛笔并递给左夹爪 |
| `stage2` | `writing.remove_brush_cap_and_return_cap` | 左夹爪取盖归位，右夹爪保持毛笔稳定 |
| `stage3` | `writing.write_with_brush` | 右夹爪书写，左臂和左夹爪保持静止 |
| `stage4` | `writing.return_brush` | 右夹爪归还毛笔，左侧保持静止 |

`--prompt-stage` 与原有 `--prompt` 二选一。`--prompt` 仍可用于 artifact 内的 canonical 或
12 个训练同义句，因此旧文本部署接口保留；不在 artifact allowlist 内的文本以及 16 个
held-out 文本会在创建相机、夹爪和机械臂 backend 前失败。prompt-conditioned artifact 不再
允许省略两者，以免无文本请求进入模型。

### 18.3 选择 checkpoint 并做无硬件绑定检查

把 `CKPT_STEP` 改成实际存在且包含 `policy_artifact.json`、params 和本 profile norm stats 的
数字 step。这里的 `30000` 只是可选择 step 的写法，不表示该 checkpoint 已在当前机器验证：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"
export CKPT_STEP=30000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-multistage-prompt-family-0804-0807-front-h16-absolute-prepared-v2/writing-multistage-prompt-family-0804-0807-front-h16-absolute-b1-nw4-v2/$CKPT_STEP

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json \
  --prompt-stage stage1 \
  --contract-check-only
```

输出中的 `prompt_selection.source` 应为 `prepared_stage`，`stage_id` 应为 `stage1`，prompt
必须与 artifact 中 stage1 canonical prompt 完全一致。该命令不加载模型、不连接 server、
相机、夹爪或机械臂。四个 stage 可分别替换后重复检查。

### 18.4 终端 A：同一个多阶段 checkpoint server

server 命令保持既有 prepared 结构，不需要也不建议设置 `--default-prompt`；每次推理的文本由
client 请求携带：

```bash
cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export CKPT_STEP=30000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-multistage-prompt-family-0804-0807-front-h16-absolute-prepared-v2/writing-multistage-prompt-family-0804-0807-front-h16-absolute-b1-nw4-v2/$CKPT_STEP

python -m deployments.openpi.server \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 \
  --port 8078
```

server 启动后先检查 `/healthz` 和 `/metadata`。stage1–4 共用这个进程；只有更换 checkpoint、
profile 或 OpenPI config 时才停止并重启 server。

### 18.5 终端 B：stage1 从数据集初始姿态启动

下方保留第 9 节现有 position 15/60/120 Hz、qpos-init、相机、队列、force guard 和人工
`g`/`A` 放行结构，仅把 checkpoint/profile 换成多阶段版本并新增
`--prompt-stage stage1`：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export CKPT_STEP=30000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-multistage-prompt-family-0804-0807-front-h16-absolute-prepared-v2/writing-multistage-prompt-family-0804-0807-front-h16-absolute-b1-nw4-v2/$CKPT_STEP

python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json \
  --prompt-stage stage1 \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage \
  --policy-qpos-init-episode 0 \
  --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-collision-check \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --send-actions \
  --debug
```

首次真机前按第 7 节规则派生 no-action：删除 `--send-actions` 并添加
`--max-steps 100`，其余参数包括 `--prompt-stage stage1` 保持不变。

### 18.6 终端 B：stage2/3/4 从当前姿态继续

stage1 正常退出并确认当前姿态适合作为下一阶段输入后，保持终端 A 的同一个 server。终端 B
每次只运行一个 client；先设置要执行的 stage，再执行相同 client 命令：

```bash
export PROMPT_STAGE=stage2
```

`stage2` 完成并退出后分别改为 `stage3`、`stage4` 重复。正式 client 命令如下：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export CKPT_STEP=30000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-multistage-prompt-family-0804-0807-front-h16-absolute-prepared-v2/writing-multistage-prompt-family-0804-0807-front-h16-absolute-b1-nw4-v2/$CKPT_STEP
export PROMPT_STAGE=stage2

python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json \
  --prompt-stage "$PROMPT_STAGE" \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  --no-init-qpos \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --send-actions \
  --debug
```

该方式只复用同一个模型 server 和当前物理姿态，不保留上一个 client 的 action queue、模型
history 或未执行 chunk，也不会自动判断一个 stage 是否成功完成；操作者必须在每次切换前
确认动作已停止、当前双臂/夹爪状态满足下一阶段分布，然后重新完成当前姿态 startup 和人工
ready 放行。no-action 仍按第 7 节规则派生。第 9.2 节断联夹爪选项可以原样追加，不改变文本
stage 绑定。

## 19. 新 Stage1 scenario 模型（0804–0811，repeat 1:4:6）部署

本节更新 Stage1 模型本身，保留第 9 节的 checkpoint-bound prepared server/client、
position 15/60/120 Hz、dataset-frame qpos-init、碰撞检查、action queue、force guard 和人工
`g`/`A` 放行结构。第 9 节旧 `0804–0805` Stage1 命令继续保留，作为显式回退入口；不要把
旧 profile 与本节 checkpoint 交叉组合。

2026-08-13 对 `dex-remote` 的只读检查确认以下实验目录存在 `30000`、`60000`，两者均有
`policy_artifact.json` 和 params；`60000` 还确认有本 profile 的 norm stats。下面选择
`CKPT_STEP=60000`。如果需要部署 `30000`，只改该变量，server 与 client 必须同时重启并使用
同一个值。

该 checkpoint 绑定 13 个 0804–0811 数据源、2,283 个合格 Episode、222,899 个原始
prepared 窗口，以及 fixed-middle / initial-multi-choice / midcourse-interrupt 的 1:4:6
重复权重。prepared wrapper 会在加载模型或创建硬件前核对 profile schema、选择策略、各来源
路径和重复权重。训练使用的默认文本
`Pick up the capped brush and present it for uncapping.` 由受 profile hash 约束的
`default_prompt` 自动恢复，命令中不要再手写 `--default-prompt`。

### 19.1 无硬件绑定检查

此命令只读取 checkpoint/profile 并生成一次性 deployment artifact；不加载模型、不连接
server、相机、夹爪或机械臂：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export CKPT_STEP=60000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-prepared-v1/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-b1-nw4-v1/$CKPT_STEP

python -m deployments.openpi.client.real \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json \
  --contract-check-only
```

### 19.2 终端 A：新 Stage1 server

```bash
cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export CKPT_STEP=60000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-prepared-v1/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-b1-nw4-v1/$CKPT_STEP

python -m deployments.openpi.server \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 \
  --port 8078
```

更换自旧 Stage1 或其他 checkpoint 前，先由原进程 owner 用 `Ctrl+C` 正常停止旧 server，
确认 `8078` 和对应显存已经释放，再运行本命令。不要让两个 server 争用同一端口或 GPU。

### 19.3 终端 B：新 Stage1 client

首次真机前，先从下方正式命令派生 no-action：删除 `--send-actions` 并添加
`--max-steps 100`；其余参数不变。no-action 会连接 server、front 相机和状态 backend，但不
下发动作。通过后才可在完成现场安全检查、急停确认和工作区清空后运行下方正式命令；启动后
先按 `g` 执行 dataset-frame qpos-init，确认姿态和 ready，再按 `A` 放行 policy：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export CKPT_STEP=60000
export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-prepared-v1/writing-stage1-scenarios-0804-0811-front-h16-absolute-repeat146-b1-nw4-v1/$CKPT_STEP

python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir "$OPENPI_CKPT" \
  --prepared-training-profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage \
  --policy-qpos-init-episode 0 \
  --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-collision-check \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --send-actions \
  --debug
```

第 9.2 节的断联夹爪选项可以原样追加到本节 client，不改变 checkpoint/profile 或 Stage1
默认文本绑定。这里继续使用既有 0804 Episode 0/frame 0 作为物理 qpos-init 来源；更多训练数据
只扩展策略分布，不会自动改变机器人启动轨迹。

## 20. 当前四个独立 Stage 模型部署

本节对应四个分别从 π0.5 基座训练的普通 prepared 模型，不是第 18 节的单 checkpoint
prompt-family 模型，也不是第 11 节 model-RTC。四个阶段不能使用同一个路径模板；必须逐项绑定
实际训练过的 checkpoint 和对应 profile：

| Stage | 最后一次远端已验证的 checkpoint 实验 | Training profile | 默认 step | 启动姿态 |
| --- | --- | --- | --- | --- |
| 1 | `writing-stage1-0804-0805-front-h16-absolute-prepared-v1/...-b1-nw4-meta-v2` | `writing_stage1_0804_0805_front_h16_absolute_v1.json` | `150000` | 0804 Episode 0/frame 0 qpos-init |
| 2 | `writing-stage2-0804-0805-front-h16-absolute-prepared-v1/...-b1-nw4-meta-v1` | `writing_stage2_0804_0805_front_h16_absolute_v1.json` | `150000` | 当前姿态，`--no-init-qpos` |
| 3 | `writing-stage3-0804-0806-front-h16-absolute-prepared-v1/...-b1-nw4-meta-v1` | `writing_stage3_0804_0806_front_h16_absolute_v1.json` | `150000` | 当前姿态，`--no-init-qpos` |
| 4 | `writing-stage4-0804-0806-front-h16-absolute-prepared-v1/...-b1-nw4-meta-v1` | `writing_stage4_0804_0806_front_h16_absolute_v1.json` | `150000` | 当前姿态，`--no-init-qpos` |

下列命令默认 `STAGE=1`，并为每个 Stage 选择表中的默认 step；可以在粘贴命令前分别执行如
`export STAGE=3`、`export CKPT_STEP=120000` 进行选择。命令会在启动前确认数字 step、profile、
artifact、params 和对应 norm stats 均存在，因此不存在的 step 会直接退出，不会退回其他模型。

本组合明确选择普通旧 Stage1，不选择第 19 节的 scenario repeat146 模型。Stage1/2
0804–0805 及 Stage3/4 0804–0806 的 `30000/60000/90000/120000/150000` 均于此前远端
审计确认；默认统一选择 `150000`。`writing-stage1-0804-0806-.../60000` 并不是已验证
checkpoint，不能由 Stage3/4 的命名模式推导。所有命令仍在启动前 fail closed。

### 20.1 终端 A：所选 Stage server

每次切换 Stage 或 step，先正常 `Ctrl+C` 停止旧 server，确认 `8078` 和 GPU 已释放，再在
终端 A 运行：

```bash
(
set -euo pipefail

STAGE="${STAGE:-1}"

case "$STAGE" in
  1)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2
    ASSET_ID=writing_stage1_0804_0805_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  2)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage2_0804_0805_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  3)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage3_0804_0806_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage3-0804-0806-front-h16-absolute-prepared-v1/writing-stage3-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage3_0804_0806_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  4)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage4_0804_0806_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage4-0804-0806-front-h16-absolute-prepared-v1/writing-stage4-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage4_0804_0806_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  *) printf 'Unsupported STAGE: %s (expected 1, 2, 3, or 4)\n' "$STAGE" >&2; exit 2 ;;
esac
CKPT_STEP="${CKPT_STEP:-$DEFAULT_CKPT_STEP}"
case "$CKPT_STEP" in
  ''|*[!0-9]*|0) printf 'CKPT_STEP must be a positive integer: %s\n' "$CKPT_STEP" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="$CHECKPOINT_ROOT/$CKPT_STEP"

if [[ ! -f "$PROFILE" ]]; then
  printf 'Missing training profile: %s\n' "$PROFILE" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/policy_artifact.json" ]]; then
  printf 'Missing checkpoint artifact: %s\n' "$CHECKPOINT_DIR/policy_artifact.json" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR/params" ]]; then
  printf 'Missing checkpoint params: %s\n' "$CHECKPOINT_DIR/params" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" ]]; then
  printf 'Missing checkpoint norm stats: %s\n' "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" >&2
  exit 2
fi

cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

printf '[stage%s] serving checkpoint step %s on port 8078; Ctrl+C returns to the current shell\n' "$STAGE" "$CKPT_STEP"
python -u -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 \
  --port 8078
)
```

### 20.2 终端 B：contract check 与 100-step no-action

此命令先执行完全无硬件的 `--contract-check-only`，通过后再连接 server 和 observation
backend 运行 100 step，但不包含 `--send-actions`：

```bash
(
set -euo pipefail

STAGE="${STAGE:-1}"

case "$STAGE" in
  1)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2
    ASSET_ID=writing_stage1_0804_0805_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  2)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage2_0804_0805_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  3)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage3_0804_0806_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage3-0804-0806-front-h16-absolute-prepared-v1/writing-stage3-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage3_0804_0806_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  4)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage4_0804_0806_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage4-0804-0806-front-h16-absolute-prepared-v1/writing-stage4-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage4_0804_0806_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  *) printf 'Unsupported STAGE: %s (expected 1, 2, 3, or 4)\n' "$STAGE" >&2; exit 2 ;;
esac
CKPT_STEP="${CKPT_STEP:-$DEFAULT_CKPT_STEP}"
case "$CKPT_STEP" in
  ''|*[!0-9]*|0) printf 'CKPT_STEP must be a positive integer: %s\n' "$CKPT_STEP" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="$CHECKPOINT_ROOT/$CKPT_STEP"

if [[ ! -f "$PROFILE" ]]; then
  printf 'Missing training profile: %s\n' "$PROFILE" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/policy_artifact.json" ]]; then
  printf 'Missing checkpoint artifact: %s\n' "$CHECKPOINT_DIR/policy_artifact.json" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR/params" ]]; then
  printf 'Missing checkpoint params: %s\n' "$CHECKPOINT_DIR/params" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" ]]; then
  printf 'Missing checkpoint norm stats: %s\n' "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" >&2
  exit 2
fi

if [[ "$STAGE" == 1 ]]; then
  QPOS_ARGS=(
    --policy-qpos-init-dataset /home/user/zc/DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage
    --policy-qpos-init-episode 0
    --policy-qpos-init-frame 0
    --policy-qpos-init-source auto
    --policy-qpos-init-collision-check
  )
else
  QPOS_ARGS=(--no-init-qpos)
fi

DISCONNECTED_GRIPPER_SIDES="${DISCONNECTED_GRIPPER_SIDES:-}"
DISCONNECTED_GRIPPER_STATE="${DISCONNECTED_GRIPPER_STATE:-0.0}"
case "$DISCONNECTED_GRIPPER_SIDES" in
  '') GRIPPER_ARGS=() ;;
  left|right|both)
    GRIPPER_ARGS=(
      --disconnected-gripper-sides "$DISCONNECTED_GRIPPER_SIDES"
      --disconnected-gripper-state "$DISCONNECTED_GRIPPER_STATE"
    )
    ;;
  *) printf 'Unsupported DISCONNECTED_GRIPPER_SIDES: %s\n' "$DISCONNECTED_GRIPPER_SIDES" >&2; exit 2 ;;
esac

cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
DEXTELEOP_PYTHON="$(command -v python)"

printf '[stage%s] validating the bound artifact, then running 100 no-action steps for checkpoint %s\n' "$STAGE" "$CKPT_STEP"
"$DEXTELEOP_PYTHON" -u -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --contract-check-only

"$DEXTELEOP_PYTHON" -u -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  "${QPOS_ARGS[@]}" \
  "${GRIPPER_ARGS[@]}" \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --max-steps 100 \
  --debug
)
```

### 20.3 终端 B：允许动作的正式 client

只有 20.2、server metadata 和现场安全检查都通过后才运行。Stage1 按 `g` 执行数据首帧
qpos-init，确认 ready 后按 `A`；Stage2–4 按 `g` 从当前姿态建立 startup reference，确认
ready 后按 `A`。这不会保留上一阶段的 queue、history 或未执行 chunk：

```bash
(
set -euo pipefail

STAGE="${STAGE:-1}"

case "$STAGE" in
  1)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2
    ASSET_ID=writing_stage1_0804_0805_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  2)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage2_0804_0805_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  3)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage3_0804_0806_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage3-0804-0806-front-h16-absolute-prepared-v1/writing-stage3-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage3_0804_0806_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  4)
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage4_0804_0806_front_h16_absolute_v1.json
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage4-0804-0806-front-h16-absolute-prepared-v1/writing-stage4-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    ASSET_ID=writing_stage4_0804_0806_front_h16_absolute_prepared_v1
    DEFAULT_CKPT_STEP=150000
    ;;
  *) printf 'Unsupported STAGE: %s (expected 1, 2, 3, or 4)\n' "$STAGE" >&2; exit 2 ;;
esac
CKPT_STEP="${CKPT_STEP:-$DEFAULT_CKPT_STEP}"
case "$CKPT_STEP" in
  ''|*[!0-9]*|0) printf 'CKPT_STEP must be a positive integer: %s\n' "$CKPT_STEP" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="$CHECKPOINT_ROOT/$CKPT_STEP"

if [[ ! -f "$PROFILE" ]]; then
  printf 'Missing training profile: %s\n' "$PROFILE" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/policy_artifact.json" ]]; then
  printf 'Missing checkpoint artifact: %s\n' "$CHECKPOINT_DIR/policy_artifact.json" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR/params" ]]; then
  printf 'Missing checkpoint params: %s\n' "$CHECKPOINT_DIR/params" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" ]]; then
  printf 'Missing checkpoint norm stats: %s\n' "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" >&2
  exit 2
fi

if [[ "$STAGE" == 1 ]]; then
  QPOS_ARGS=(
    --policy-qpos-init-dataset /home/user/zc/DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage
    --policy-qpos-init-episode 0
    --policy-qpos-init-frame 0
    --policy-qpos-init-source auto
    --policy-qpos-init-collision-check
  )
  printf '[stage1] checkpoint %s: press g for dataset qpos init, confirm ready, then press A\n' "$CKPT_STEP"
else
  QPOS_ARGS=(--no-init-qpos)
  printf '[stage%s] checkpoint %s: press g for current-pose startup, confirm ready, then press A\n' "$STAGE" "$CKPT_STEP"
fi

DISCONNECTED_GRIPPER_SIDES="${DISCONNECTED_GRIPPER_SIDES:-}"
DISCONNECTED_GRIPPER_STATE="${DISCONNECTED_GRIPPER_STATE:-0.0}"
case "$DISCONNECTED_GRIPPER_SIDES" in
  '') GRIPPER_ARGS=() ;;
  left|right|both)
    GRIPPER_ARGS=(
      --disconnected-gripper-sides "$DISCONNECTED_GRIPPER_SIDES"
      --disconnected-gripper-state "$DISCONNECTED_GRIPPER_STATE"
    )
    ;;
  *) printf 'Unsupported DISCONNECTED_GRIPPER_SIDES: %s\n' "$DISCONNECTED_GRIPPER_SIDES" >&2; exit 2 ;;
esac

cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
DEXTELEOP_PYTHON="$(command -v python)"

"$DEXTELEOP_PYTHON" -u -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-qref-acceleration-profile comfort_75_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  "${QPOS_ARGS[@]}" \
  "${GRIPPER_ARGS[@]}" \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --send-actions \
  --debug
)
```

默认连接双夹爪。如果某侧明确不在线，在粘贴 20.2/20.3 前设置例如：

```bash
export DISCONNECTED_GRIPPER_SIDES=left
export DISCONNECTED_GRIPPER_STATE=0.0
```

允许值为 `left/right/both`；不设置或设为空即保持默认双夹爪在线。以上命令没有启用 RTC、
`async_c2`、`auto-start` 或跳过 ready confirm；其他安全和控制结构与第 9 节一致。
