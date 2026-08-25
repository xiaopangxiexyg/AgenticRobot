# OpenPI 现场部署顺序

本文只保留现场两终端部署的最短顺序。训练路线、恢复、prepared/model-RTC 参数和每条
命令的副作用统一见
[`deployments/openpi/COMMANDS.md`](../../deployments/openpi/COMMANDS.md)；具体
checkpoint/profile 长命令见
[`deployments/openpi/cmd_list.md`](../../deployments/openpi/cmd_list.md)。

OpenPI server 使用 OpenPI `.venv` 加载模型；`dexteleop310` 中的 client 负责
observation、artifact contract、初始化、现场安全门和可选真实下发。两端必须绑定同一个
checkpoint/artifact；prepared checkpoint 还必须绑定同一份 training profile。

## 1. 设置路径并选择绑定

```bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
export OPENPI_PYTHON="$OPENPI_ROOT/.venv/bin/python"

conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

export CHECKPOINT_DIR=/absolute/path/to/checkpoint/STEP
export POLICY_SERVER_HOST=127.0.0.1
export OPENPI_PORT=8078

test -d "$CHECKPOINT_DIR"
test -f "$CHECKPOINT_DIR/policy_artifact.json"
```

跨主机部署时，client 的 `POLICY_SERVER_HOST` 必须改为 server 可路由地址。远端路径是
站点事实，换主机后重新核对，不要把旧 shell 的目录直接复用。

按 checkpoint 类型选择且只选择一种绑定：

| 类型 | 两端共同参数 |
| --- | --- |
| 标准 artifact | `--checkpoint-dir "$CHECKPOINT_DIR"` |
| 兼容 artifact | server 用 checkpoint + `--policy-artifact-json FILE`；client 用同一 FILE |
| prepared | checkpoint + `--prepared-training-profile "$PROFILE"` |
| prepared model-RTC | prepared 参数 + 同一 `--model-rtc-profile "$RTC_PROFILE"` |

不要把 prepared 的 profile、普通 `--config` 和兼容 artifact 混用。

## 2. 先做不打开设备的检查

client 静态 contract check：

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --contract-check-only
```

prepared checkpoint 在同一命令中增加：

```bash
--prepared-training-profile "$PROFILE"
```

该命令不连接 server、相机、夹爪或机械臂。然后做 server 命令预检：

```bash
"$OPENPI_PYTHON" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --host 0.0.0.0 --port "$OPENPI_PORT" \
  --dry-run
```

prepared checkpoint 再执行一次动态 config 检查：

```bash
"$OPENPI_PYTHON" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --prepared-config-check-only
```

`--dry-run` 和 `--prepared-config-check-only` 不加载权重、不监听端口；通过不代表推理或
真实硬件已经验证。

## 3. 终端 A：启动 server

```bash
cd "$DEXTELEOP_ROOT"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

"$OPENPI_PYTHON" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --host 0.0.0.0 --port "$OPENPI_PORT"
```

prepared、兼容 artifact 或 model-RTC 按第 1 节增加对应参数。普通 server 不加
`--enable-rtc-continuation`。另开只读终端检查：

```bash
curl --fail --silent "http://127.0.0.1:$OPENPI_PORT/healthz"
curl --fail --silent "http://127.0.0.1:$OPENPI_PORT/metadata"
```

确认 schema、state/action dim、camera slots、action mode、normalization 和 prompt/RTC
capability 与 artifact/profile 一致。

## 4. 终端 B：先 no-action

下面命令刻意不含 `--send-actions`：

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --deployment-profile rm65_bimanual_dmtac \
  --host "$POLICY_SERVER_HOST" --port "$OPENPI_PORT" \
  --max-steps 100
```

prepared/profile 参数必须与 server 相同。no-action 仍可能打开 RealSense、DMTAC 和真实
observation/backend；它只保证没有 policy action dispatch，不是无硬件检查。

## 5. 真实动作只增加一个显式门

只有完整 preflight、contract、server metadata、no-action、qpos-init 目标和现场 ready
gate 都通过后，才能在已经验证的同一条 client 命令末尾增加：

```bash
--send-actions
```

完整命令必须按 policy 类型从 `cmd_list.md` 第 8、9、10、19 或 20 节选择。
`rm65_bimanual_dmtac` profile 本身不会开启动作。

Stage1 通常执行绑定的数据集 qpos-init；Stage2–4 只有在当前姿态就是已确认的连续衔接
状态时才使用 `--no-init-qpos`。该参数不会继承前一 client 的 queue/history，也不会跳过
ready、碰撞、力限或观测新鲜度门。不要加入 `--auto-start`、`--start-init_qpos` 或
`--no-wait-for-ready-confirm` 来缩短正式流程。

## 6. 更换 checkpoint

1. 停止 client 和 server，确认机器人已经退出策略控制。
2. 设置新的 `CHECKPOINT_DIR`，重新检查 artifact/profile/receipt。
3. 从第 2 节重新执行 contract 和 server preflight。
4. 启动新 server，重新检查 metadata，并重新做 no-action。

不得通过手工覆盖 model id、layout hash、step、camera routing 或 action pose mode 来绕过
artifact/profile 错误。
