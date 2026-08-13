# OpenPI 策略部署

OpenPI 使用两个进程：OpenPI 环境中的 server 加载模型，`dexteleop310` 中的 DexTeleop
client 负责 observation、contract、现场安全门和可选真实下发。通常两端以 checkpoint
目录内的 `policy_artifact.json` 为共同事实源。历史 wrist-only checkpoint 在当前物理左右
绑定下部署时，两端必须显式选择同一份、且绑定原 checkpoint artifact hash 的版本化兼容
artifact；不得用零散命令行参数手工覆盖 action pose mode、相机或 state/action layout。

训练侧另有一条独立边界：普通 wrist `-corrected-v1` config 必须通过
`python -m deployments.openpi.corrected_train --dry-run -- ...` 先审计数据来源，再去掉
wrapper 的 `--dry-run` 启动。历史无后缀 config 保持冻结；model-RTC 从 92599 续训时必须从
`0712-wrist-cube-92599-legacy-wrist-compat.policy_artifact.json` 派生 template。完整命令见
[`deployments/openpi/README.md`](../../deployments/openpi/README.md)。

## 0. 一次性设置路径

先在当前 shell 替换三条本机路径：

```bash
export DEXTELEOP_ROOT=/path/to/DexTeleop
export OPENPI_ROOT=/path/to/openpi
export CHECKPOINT_DIR=/path/to/checkpoint-step
export CHECKPOINT_LINK=/tmp/dexteleop_openpi_ckpt
```

`CHECKPOINT_DIR` 必须直接包含 `policy_artifact.json`。当前站点曾验证过的模型是 relative
`0715-cube-front-ready-mix-4to1-relative/149999` 和 absolute
`0715-cube-front-ready-mix-4to1/138257`；模型名和 step 是审计线索，不代表其他机器拥有相同
目录结构。

## 1. 选择并检查 checkpoint（R1，只写 `/tmp`）

```bash
test -f "$CHECKPOINT_DIR/policy_artifact.json" && \
  ln -sfnT "$CHECKPOINT_DIR" "$CHECKPOINT_LINK"
```

`test` 失败时不要启动 server。软链接只写 `/tmp`，方便两个终端选择同一个 checkpoint；
它不修改 checkpoint、OpenPI registry 或源码。

## 2. 启动 server（R1，不连接机器人）

在 OpenPI 环境终端执行：

```bash
cd "$DEXTELEOP_ROOT"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

"$OPENPI_ROOT/.venv/bin/python" \
  -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_LINK" \
  --port 8078
```

该进程会加载模型并占用 GPU/端口，但不初始化 RM65。若 artifact 的 model id 与 OpenPI
training config 名称不同，可额外传 `--config CONFIG_NAME`；只有确认二者确实是同一模型时
才使用该覆盖。

## 3. 启动 client

另开终端：

```bash
cd "$DEXTELEOP_ROOT"
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export CHECKPOINT_DIR="$CHECKPOINT_LINK"
export POLICY_SERVER_HOST=127.0.0.1
```

先依次运行 [真实系统入口的 contract 检查和只观测模式](./README.md#3-deploy)。contract
检查是 R0；只观测模式会连接真实相机/状态源但不发送动作，按 H1 处理。

只有两步都通过、qpos-init 数据和现场 ready gate 已人工确认后，才允许使用下面的 M 级命令：

```bash
python -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --deployment-profile rm65_bimanual_dmtac \
  --host "$POLICY_SERVER_HOST" \
  --port 8078 \
  --send-actions
```

`rm65_bimanual_dmtac` profile 解析当前站点的双 RM65、DMTAC、三路相机、tool profile、
`impedance_40hz`、dataset-frame qpos-init 和人工 ready gate，但 profile 本身不会自动增加
`--send-actions`。

真机启动后按 `g` 执行 qpos-init，确认 ready 和实际姿态无误后再按 `A` 放行策略。不要添加
`--auto-start`、`--start-init_qpos` 或绕过 ready confirmation 的参数。

## 4. 更换 checkpoint

1. 停止 client 和 server，确认真机已经退出策略控制。
2. 修改 `CHECKPOINT_DIR`，重新执行第 1 节的 `test` 和 `ln`。
3. 重新启动 server，先重新做 contract 检查，再进入只观测模式。

若 artifact 缺失、过期，或与 server metadata 不一致，必须在创建真实执行链前失败；不要通过
手工覆盖 model id、layout hash、step 或 action pose mode 绕过。
