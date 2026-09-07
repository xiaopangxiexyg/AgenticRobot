# OpenPI 现场部署顺序

本文只保留现场两终端部署的最短顺序。训练路线、恢复、prepared/model-RTC 参数和每条
命令的副作用统一见
[`deployments/openpi/COMMANDS.md`](../../deployments/openpi/COMMANDS.md)；具体
checkpoint/profile 长命令见
[`deployments/openpi/cmd_list.md`](../../deployments/openpi/cmd_list.md)。

OpenPI server 使用 OpenPI `.venv` 加载模型；`dexteleop310` 中的 client 负责
observation、artifact contract、初始化、现场安全门和可选真实下发。每一端必须用自己的
checkpoint/artifact 和 prepared profile 建立完整契约，但默认不要求 client 与 server 的
model id 或 checkpoint step 相同；连接时校验的是 layout、相机、动作、normalization、prompt
和 RTC capability 等实际 wire/runtime 兼容性。部署入口不提供 model/step 精确匹配开关；
model id、checkpoint step、artifact hash 和 continuation contract hash 只保留为来源与同一次
请求/响应的因果收据，不参与跨 client/server 的实例身份准入。

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

| 类型 | 每一端自己的绑定参数 |
| --- | --- |
| 标准 artifact | `--checkpoint-dir "$CHECKPOINT_DIR"` |
| 兼容 artifact | server 用 checkpoint + `--policy-artifact-json FILE`；client 用同一 FILE |
| prepared | checkpoint + `--prepared-training-profile "$PROFILE"` |
| prepared model-RTC | prepared 参数 + 同一 `--model-rtc-profile "$RTC_PROFILE"` |

不要把 prepared 的 profile、普通 `--config` 和兼容 artifact 混用。为了测试不同 step 或
兼容模型，分别设置 `SERVER_CHECKPOINT_DIR` 与 `CLIENT_CHECKPOINT_DIR`；不要靠伪造 model id、
step 或 hash 绕过不兼容错误。

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

完整命令必须按 policy 类型从 `cmd_list.md` 第 8、9、10、19、20 或 23 节选择。
`rm65_bimanual_dmtac` profile 本身不会开启动作。

Stage1 通常执行绑定的数据集 qpos-init；Stage2–4 只有在当前姿态就是已确认的连续衔接
状态时才使用 `--no-init-qpos`。该参数不会继承前一 client 的 queue/history，也不会跳过
ready、碰撞、力限或观测新鲜度门。不要加入 `--auto-start`、`--start-init_qpos` 或
`--no-wait-for-ready-confirm` 来缩短正式流程。

model-RTC Active 还会在 qpos-init 完成后保持测量 qpos，丢弃默认 1 次冷启动 completion，
用随后 3 次 completion 做推理延迟门；默认上界为 0.4 秒，且每次 warm completion 都必须
严格小于该值。在该门通过且操作者完成 ready
确认之前不会下发策略动作。计划 underrun、producer 不可用或延迟超限都会清空待执行计划并
保持最新测量 qpos；无法建立该 Hold 时升级到更强 Hold 并报错退出。`--max-steps` 验收可配合
`--min-rtc-emissions-before-exit`，避免“进程正常退出但零有效 emission”的假通过。

trained model-RTC sampler 默认保持 10 step。若 10-step warm latency 无法通过 `<0.4 s` 门，
可在同一轮 server 与 client 命令中都显式加入 `--rtc-inference-steps 8`。这不是修改 checkpoint
或训练 profile：client 会根据 loaded provider 的完整实现参数生成 deployment-only binding
receipt，并要求把步数恢复为 10 后精确复现训练契约哈希；任何其它参数漂移或两端步数不一致
都会在动作派发前失败。server 还必须用 `--rtc-runtime-openpi-root` 显式选择当前不可变部署
快照中 gitlink 固定的 `model/openpi`，而 `--openpi-root` 仍指向 profile 固定的训练 checkout；
两者不能靠修改旧训练 checkout 合并。不要通过放宽
`--rtc-active-max-inference-latency-sec` 代替该绑定。

## 6. 更换 checkpoint 或 server 模型

1. 停止 client 和 server，确认机器人已经退出策略控制。
2. 分别设置新的 server/client checkpoint，重新检查各自 artifact/profile/receipt。
3. 对 client 重新执行 contract check，对 server 重新执行 server preflight。
4. 启动新 server，重新检查 metadata，并重新做 no-action。

默认允许 server 与 client 使用不同 model/step，但二者仍必须通过实际协议、layout、相机、
normalization、prompt 和 RTC capability 检查。不得通过手工覆盖 model id、layout hash、step、
camera routing 或 action pose mode 来绕过 artifact/profile 错误。当前 corrected Stage1 v2
`25006` 与 historical Stage2 v1 `100000` 的完整远端命令见 `cmd_list.md` 第 23 节；
该路线只开放真实 observation 的
no-action shadow。真实动作还必须显式加入 `--send-actions`，完成 dataset-frame qpos-init、
ready 确认和上述 Active latency gate；profile 本身不会授予动作权限。
