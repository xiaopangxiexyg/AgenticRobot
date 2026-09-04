# OpenPI 训练与部署命令总览

本文档是 AgenticRobot 内 OpenPI 命令的规范入口，回答四个问题：应该走哪条
训练路线、应该使用哪个环境、每一步会产生什么副作用，以及部署时如何保证 server
与 client 的实际模型契约兼容。默认不要求两端 model id 或 checkpoint step 相同；精确
identity 检查是显式 opt-in。

具体 checkpoint、实验名和现场长命令保留在
[`cmd_list.md`](./cmd_list.md)；模型 I/O、相机和动作语义见
[`README.md`](./README.md)。不要从两份历史命令中各摘一半拼成新命令。

## 1. 文档分工

| 文档 | 责任 |
| --- | --- |
| 本文 | 命令分类、选择顺序、标准模板、副作用与安全边界 |
| [`cmd_list.md`](./cmd_list.md) | 已审计 checkpoint/profile 的详细命令块和历史快照 |
| [`README.md`](./README.md) | OpenPI 与 AgenticRobot 的 artifact、相机、RTC 和运行时契约 |
| [`docs/real/OPENPI.md`](../../docs/real/OPENPI.md) | 现场两终端部署的最短操作顺序 |
| [`client/README.md`](./client/README.md) | client adapter、normalization、RTC 与 observation shadow 语义 |
| [`CORRECTED_TRAINING_CONFIGS_V1.md`](./CORRECTED_TRAINING_CONFIGS_V1.md) | 历史 config 与 corrected-v1 config 的冻结身份 |

未合入当前 development 基线的 profile、checkpoint 或临时工作区不属于可执行清单。
先完成代码/profile 集成和远端只读复核，再把它加入 `cmd_list.md`。

## 2. 环境与事实源

OpenPI 和 AgenticRobot 使用两套环境，不要交叉替代：

| 职责 | 当前站点路径 | Python/环境 |
| --- | --- | --- |
| OpenPI 配置、norm stats、训练、模型 server | `/home/user/zc/openpi` | `/home/user/zc/openpi/.venv/bin/python` |
| prepared/corrected wrapper、artifact 检查、真实 client | `/home/user/zc/DexTeleop` | `dexteleop310`；训练 wrapper 仍由 OpenPI Python 执行 |
| 数据 | profile/manifest 中的显式路径 | 路径存在不等于数据已通过 profile eligibility |

截至 2026-08-22，只读核对确认远端主机为 `user@192.168.70.116`，上述三个根目录
`/home/user/zc/openpi`、`/home/user/zc/DexTeleop`、`/home/user/zc/datasets`
均存在。它们是当前站点事实，不是可移植默认值；换主机后必须重新核对身份、路径、
Git 提交、GPU、数据挂载和 checkpoint。

训练和部署的共同事实源是 checkpoint 中的 `policy_artifact.json`。prepared 或
model-RTC checkpoint 还必须绑定 training profile、norm-stats bytes、training receipt
等附加凭据。checkpoint 目录名、config 名或 step 数字本身不能替代这些契约。

推荐先在每个终端设置：

```bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
export OPENPI_PYTHON="$OPENPI_ROOT/.venv/bin/python"
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"
export PYTHONPATH="$DEXTELEOP_ROOT:$OPENPI_ROOT/src"
```

不要把 OpenPI `.venv` 当作真实硬件 client 环境。

## 3. 先按副作用选择命令

| 类别 | 典型命令 | 会做什么 | 不会证明什么 |
| --- | --- | --- | --- |
| 只读入口检查 | `--help`、profile/artifact 查看 | 解析 CLI 或读取文件 | 数据可训练、模型可加载 |
| 训练审计 | `prepared_skill_launch audit`、corrected `--dry-run` | 审计 profile、source、相机语义 | 不生成 norm stats，不训练 |
| 训练准备 | `prepare`、`sample-check` | 生成/复用 immutable view，或解码抽样 | 不加载 trainer，不产生 checkpoint |
| 统计 | `norm-stats`、`compute_norm_stats.py` | 读取数据并写 normalization assets | 不训练模型 |
| 训练 | `train`、sequence launcher | 占用 GPU，写 checkpoint、日志、W&B 状态 | 不证明 checkpoint 可真机部署 |
| server 预检 | `server --dry-run`、`--prepared-config-check-only` | 解析绑定并生成临时配置/命令 | 不加载权重，不监听端口 |
| artifact 检查 | `client.real --contract-check-only` | 校验 artifact/profile/layout | 不连接 server、相机、夹爪或机械臂 |
| server | `deployments.openpi.server` | 加载模型，占用 GPU 并监听端口 | 不连接机器人 |
| no-action client | `client.real`，不含 `--send-actions` | 可能连接 server、相机和真实 observation/backend | 只保证不下发 policy action；不是无硬件 |
| 真实执行 | client 显式含 `--send-actions` | 进入真实动作链，仍受初始化和人工 ready gate 控制 | 不因命令能启动就获得安全授权 |

这里的“训练”本身不会控制机器人；`--no-motion` 不是训练正确性的替代检查。
任何真实 client 命令都应按现场硬件流程单独审计。

## 4. 训练路线选择

### 4.1 路线矩阵

| 数据/config 类型 | 唯一推荐入口 | 适用场景 |
| --- | --- | --- |
| 历史原生 OpenPI config | `scripts/compute_norm_stats.py` + `scripts/train.py` | 精确复现已冻结的历史语义 |
| 普通 wrist `-corrected-v1` | `deployments.openpi.corrected_train` | 需要按 RealSense serial 解析物理左右的普通训练 |
| Prepared Skill profile | `deployments.openpi.training.prepared_skill_launch` | 固定 eligibility、窗口、相机、prompt、动作语义的主路径 |
| 四阶段 model-RTC 序列 | `prepared_writing_sequence` | 按已验证依赖顺序训练/恢复 Stage1–4 model-RTC |
| 四个独立 state-only 模型 | `prepared_writing_state_sequence` | 每个 Stage 独立 ordinary policy，到 profile 规定 step |
| 单个 default-off model-RTC 实验 | `training.prepare` + `training.launch` | 从既有 artifact/checkpoint 派生 conditioned 模型实验 |

不要用原生 `scripts/train.py` 绕过 prepared profile；不要用 corrected wrapper 续训无后缀
历史 config；不要把 state-only sequence 当成 model-RTC sequence。

### 4.2 原生历史 config

这条路线只用于明确复现历史配置。wrist 数据若需要 corrected 相机语义，应改走 4.3。

```bash
cd "$OPENPI_ROOT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
uv run scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME"
```

`compute_norm_stats.py` 会写该 config 的 asset 目录。h16/h32/h50、absolute/relative、
相机集合或数据 binding 不同的 config，不能共用 norm stats。

### 4.3 普通 corrected-v1

wrapper 参数和 OpenPI 训练参数必须用 `--` 分开。先做审计和 sample decode：

```bash
cd "$DEXTELEOP_ROOT"
PYTHONPATH="$DEXTELEOP_ROOT:$OPENPI_ROOT/src" \
"$OPENPI_PYTHON" -m deployments.openpi.corrected_train \
  --dry-run --sample-check -- \
  "$CONFIG_NAME" --exp-name "$EXP_NAME"
```

确认后，移除 wrapper 的 `--dry-run --sample-check` 才进入训练：

```bash
cd "$DEXTELEOP_ROOT"
PYTHONPATH="$DEXTELEOP_ROOT:$OPENPI_ROOT/src" \
"$OPENPI_PYTHON" -m deployments.openpi.corrected_train -- \
  "$CONFIG_NAME" --exp-name "$EXP_NAME"
```

`--allow-camera-prepared-view` 是显式例外，不是常规默认。它只允许非触觉相机修正 view，
不把该 view 宣称为完整 v2.3 数据集。

### 4.4 Prepared Skill 标准流水线

每次只选一个已合入的 profile，并按顺序执行：

```bash
cd "$DEXTELEOP_ROOT"
export PROFILE=deployments/openpi/configs/training/PROFILE.json
export EXP_NAME=UNIQUE_EXPERIMENT_NAME

"$OPENPI_PYTHON" -m deployments.openpi.training.prepared_skill_launch \
  audit --profile "$PROFILE"

"$OPENPI_PYTHON" -m deployments.openpi.training.prepared_skill_launch \
  prepare --profile "$PROFILE"

"$OPENPI_PYTHON" -m deployments.openpi.training.prepared_skill_launch \
  sample-check --profile "$PROFILE"

"$OPENPI_PYTHON" -m deployments.openpi.training.prepared_skill_launch \
  norm-stats --profile "$PROFILE" \
  --stats-batch-size 256 --stats-num-workers 0

CUDA_VISIBLE_DEVICES=0 "$OPENPI_PYTHON" -u \
  -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE" --exp-name "$EXP_NAME"
```

profile 若规定了专用 stats batch size、worker、实验名或 step，以 profile 对应的
`cmd_list.md` 条目为准。`--skip-source-file-verification` 只能在同一 immutable receipt
已完成完整源文件校验时用于重复的 sample/stat 步骤；正式训练应重新做完整校验。

`audit` 是只读；`prepare` 会创建或复用 prepared view/binding；`sample-check` 会读取并
可能解码样本；`norm-stats` 写 normalization assets；`train` 写 checkpoint 和训练状态。

### 4.5 四阶段序列

先只查看每个 Stage 的当前决定：

```bash
cd "$DEXTELEOP_ROOT"
"$OPENPI_PYTHON" -m deployments.openpi.training.prepared_writing_sequence --dry-run
"$OPENPI_PYTHON" -m deployments.openpi.training.prepared_writing_state_sequence --dry-run
```

确认选择的是目标路线后，只执行其中一个实际命令：

```bash
# model-RTC Stage1 -> Stage4，按已验证顺序恢复/训练
"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_writing_sequence

# 四个独立 ordinary state-only policy
"$OPENPI_PYTHON" -u -m deployments.openpi.training.prepared_writing_state_sequence
```

训练终端中，`S` 请求保存并继续；一次 `Ctrl+C` 走 OpenPI 的保存并退出流程。不要用
`kill -9` 代替正常 checkpoint 保存。

### 4.6 单个 model-RTC 实验

先从只读 base artifact 派生版本化 template，再用 launcher 解析所有契约：

```bash
cd "$DEXTELEOP_ROOT"
"$OPENPI_PYTHON" -m deployments.openpi.training.prepare \
  --source-artifact-json "$BASE_ARTIFACT" \
  --output-artifact-json "$OUTPUT_TEMPLATE" \
  --output-model-id "$OUTPUT_MODEL_ID" \
  --source-dataset-root "$DATASET_ROOT"

"$OPENPI_PYTHON" -m deployments.openpi.training.launch \
  --base-config "$BASE_CONFIG" \
  --base-checkpoint-dir "$BASE_CHECKPOINT" \
  --artifact-template-json "$OUTPUT_TEMPLATE" \
  --sampling-spec-json "$SAMPLING_SPEC" \
  --source-dataset-root "$DATASET_ROOT" \
  --camera-key right_wrist --camera-key left_wrist \
  --output-config-name "$OUTPUT_CONFIG" \
  --exp-name "$EXP_NAME" \
  --dry-run
```

只有 dry-run 输出的 base artifact hash、camera routing、sampling 和 output identity
全部正确，才去掉 `--dry-run`。`--disable-camera-semantic-resolution` 是混合历史 wrist
数据的不安全回滚项，不应进入新命令。

## 5. 新训练、恢复与覆盖

| 意图 | 参数 | 前置条件 |
| --- | --- | --- |
| 新实验 | 不加 `--resume` / `--overwrite` | 使用新的唯一 `exp-name`；目标目录不存在 |
| 继续同一实验 | `--resume` | 同一 config/profile、同一 binding/asset，且存在正整数 step checkpoint |
| 清空并重建 | `--overwrite` | 破坏性操作；必须单独核对目标目录并获得明确授权 |

目录中只有 `wandb/`、日志、stats 或临时文件不构成可恢复 checkpoint。prepared 训练必须
通过同一个 profile 和 launcher 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 "$OPENPI_PYTHON" -u \
  -m deployments.openpi.training.prepared_skill_launch \
  train --profile "$PROFILE" --exp-name "$EXP_NAME" --resume
```

原生历史训练则使用：

```bash
uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" --resume
```

`reset_moments_restore_counts` 表示参数/step 的 warm start，不等于完整 optimizer resume。
跨 config/profile、动作模式、horizon、相机集合、norm stats 或 prompt family 的恢复均应
视为新实验，不能靠改名或 `--resume` 规避。

## 6. 部署绑定选择

server 和 client 必须选择同一行，不能混搭：

| checkpoint 类型 | server 参数 | client 参数 |
| --- | --- | --- |
| 标准 artifact sidecar | `--checkpoint-dir STEP` | `--checkpoint-dir STEP` |
| 版本化兼容 artifact | `--checkpoint-dir STEP --policy-artifact-json FILE` | `--policy-artifact-json FILE` |
| Prepared Skill | `--checkpoint-dir STEP --prepared-training-profile PROFILE` | 同样的 checkpoint + profile |
| Prepared model-RTC | prepared 参数再加 `--model-rtc-profile RTC_PROFILE`；conditioned server 另加 `--enable-rtc-continuation` | 同样的三份 binding；shadow 另选 model-RTC `--rtc-profile` |
| legacy registry v1 | OpenPI 原生 registry server | 对应 legacy client；不要加入 v2 artifact 参数 |

`--config` 仅用于已经证明 checkpoint 与 training config 同一身份的显式覆盖。它不能修复
缺失 artifact、错误相机 routing 或 action mode。prepared profile 与 server 的
`--policy-artifact-json`、`--config` 互斥。

## 7. 标准部署阶梯

### 7.1 公共变量

```bash
export CHECKPOINT_DIR=/absolute/path/to/checkpoint/STEP
export POLICY_ARTIFACT="$CHECKPOINT_DIR/policy_artifact.json"
export POLICY_SERVER_HOST=127.0.0.1
export OPENPI_PORT=8078

test -d "$CHECKPOINT_DIR"
test -f "$POLICY_ARTIFACT"
```

跨主机部署时，client 的 `POLICY_SERVER_HOST` 必须是 server 可路由地址，不可使用
`127.0.0.1`。

### 7.2 无模型、无监听的 server 预检

标准 artifact：

```bash
cd "$DEXTELEOP_ROOT"
"$OPENPI_PYTHON" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --host 0.0.0.0 --port "$OPENPI_PORT" \
  --dry-run
```

prepared checkpoint 还应验证动态 config 恢复：

```bash
"$OPENPI_PYTHON" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --prepared-config-check-only
```

这两条命令不加载模型权重、不监听端口。`--dry-run` 可能写临时 registry/config，不能据此
宣称 server 推理已通过。

### 7.3 无 server、无设备的 contract check

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --contract-check-only
```

prepared checkpoint 必须附同一 profile：

```bash
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --contract-check-only
```

该检查不连接 websocket、不打开相机/夹爪、不创建机械臂 backend。检查通过只证明静态
artifact/profile/tensor/pose 契约自洽。

### 7.4 启动 server

终端 A：

```bash
cd "$DEXTELEOP_ROOT"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

"$OPENPI_PYTHON" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --host 0.0.0.0 --port "$OPENPI_PORT"
```

prepared、兼容 artifact 或 model-RTC 按第 6 节增加同一组 binding 参数。不要把
`--enable-rtc-continuation` 加到普通非 RTC server。

另一个只读终端检查：

```bash
curl --fail --silent "http://127.0.0.1:$OPENPI_PORT/healthz"
curl --fail --silent "http://127.0.0.1:$OPENPI_PORT/metadata"
```

metadata 中的 schema、state/action dim、camera slots、action mode、normalization 和
prompt/RTC capability 应与 artifact/profile 一致。

### 7.5 no-action 真实观测

终端 B；刻意不含 `--send-actions`：

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --deployment-profile rm65_bimanual_dmtac \
  --host "$POLICY_SERVER_HOST" --port "$OPENPI_PORT" \
  --max-steps 100
```

这条命令可能打开 RealSense、DMTAC 和真实 observation/backend。它只验证没有 policy
action dispatch，不是“无硬件检查”。prepared/profile 参数必须与 server 相同；RTC
real shadow 还需要对应 artifact-bound RTC 参数，详见 `cmd_list.md` 第 11 节。

### 7.6 真实动作入口

只有完成现场 preflight、contract check、server metadata、no-action、qpos-init 目标和人工
ready gate 检查后，才在同一条已验证 client 命令末尾显式增加：

```bash
--send-actions
```

完整命令按 policy 类型从 `cmd_list.md` 第 8、9、10、19 或 20 节选取。Stage1 通常从
checkpoint/profile 绑定的数据集 qpos-init 启动；连续 Stage2–4 只有在当前姿态就是已确认
衔接状态时才使用 `--no-init-qpos`。该参数只跳过初始化轨迹，不继承前一进程的 queue、
model history 或 chunk，也不跳过 ready、碰撞、力限和观测新鲜度门。

`--deployment-profile rm65_bimanual_dmtac` 永远不会自动增加 `--send-actions`。
不要同时加入 `--auto-start`、`--start-init_qpos` 或 `--no-wait-for-ready-confirm` 来缩短
正式流程。

## 8. RTC 命令边界

- raw-RTC 保持普通 OpenPI `observation -> final action chunk` wire；需要显式
  `raw_rtc_final_chunk_15_30_60_120_v1` 与 normalization/frame identity。
- trained model-RTC 需要 checkpoint receipt、prepared profile、model-RTC profile 和 server
  `--enable-rtc-continuation` 四者一致。
- 当前 model-RTC 部署说明只开放 conditioned no-action shadow；不要从普通 client 命令复制
  `--send-actions`。
- `--rtc-real-shadow` 会打开实时 observation/transport，但配置为 no-action；仍然不是无硬件。
- client 与 server 可以选择不同的完整 checkpoint/step；每端分别验证自己的 prepared 与
  model-RTC profile，连接时再验证 layout、相机、动作、normalization、prompt 和 RTC
  capability。不要把 model/step identity 当默认强绑定，也不要降低这些真实兼容性检查。

历史 RTC 详细命令和回滚路径见 `cmd_list.md` 第 11 节；当前 Stage1/Stage2 all-view h32
model-RTC 的远端命令见第 23 节与
[`client/README.md`](./client/README.md)。

## 9. 远端操作规则

训练或 server 启动前，至少重新核对：远端身份、OpenPI/DexTeleop Git 提交、submodule 或
外部 OpenPI checkout、目标 checkpoint/profile hash、数据挂载、GPU 可用性、端口和既有
进程。不要根据旧终端里的路径直接开始训练。

项目源码只在授权的本地任务 worktree 修改。远端只读查看使用 `codex_remote/` wrappers；
远端同步、训练进程、server 进程、相机或机器人需要各自独立授权和证据。文档中的命令不
授权同步、重启服务、训练或真实动作。

## 10. 故障定位顺序

1. `test -f policy_artifact.json`；确认 checkpoint 是数字 step 且收据齐全。
2. 用 `client.real --contract-check-only` 区分静态契约错误与运行时错误。
3. 用 `server --dry-run`；prepared 再用 `--prepared-config-check-only`。
4. 启动 server 后检查 health/metadata，再排查 config、norm stats、camera slots 和 prompt。
5. contract 与 server 都通过后才进入 no-action observation，区分网络、相机、DMTAC、RM65
   backend 与 policy inference。
6. no-action 通过仍不等于动作安全；正式动作必须重新走 qpos-init、ready 和现场安全门。

不要用 `--config`、`--overwrite`、`--disable-camera-semantic-resolution`、
`--no-init-qpos` 或 `--send-actions` 作为故障绕过手段。

## 11. 详细命令索引

`cmd_list.md` 当前章节用途如下：

- 第 2～4 节：checkpoint 快照与变量选择；
- 第 5～8 节：标准 artifact server、静态检查、no-action 和真实执行；
- 第 9～10 节：prepared Stage1/Stage2；
- 第 11～12 节：model-RTC shadow 与 legacy v1；
- 第 13～18 节：原生、prepared、prompt-family 和恢复训练；
- 第 19～20 节：Stage1 scenario 与四个独立 Stage checkpoint 部署。
- 第 21～23 节：h32 四阶段、历史 Stage1 与当前 Stage1/Stage2 all-view model-RTC。

先在本文确定路线和副作用，再进入对应详细章节。checkpoint 清单带有审计日期，只是
站点快照；使用前必须重新验证远端实际目录和 artifact。
