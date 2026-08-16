# OpenPI wrist-only 部署

本目录连接 OpenPI 推理服务与 DexTeleop 真机部署运行时。OpenPI 仅负责模型加载和 websocket 推理；相机、机械臂、夹爪、初始化、限幅和力控均由 DexTeleop 负责。

## 0. 固定的 OpenPI 源码边界

AgenticRobot 将 OpenPI 固定为 `model/openpi` Git submodule。首次 clone 或切换到包含
gitlink 的提交后，先初始化源码：

```bash
git submodule update --init model/openpi
git -C model/openpi rev-parse HEAD
```

完整使用 OpenPI 上游示例时，可追加 `--recursive` 初始化其 `aloha`、`libero`
submodule；当前 DexTeleop server/client 路径不依赖这两个示例仓库。OpenPI 继续使用
自己的 Python/runtime 环境，AgenticRobot 不把它作为 `model.openpi` Python 包导入。

`deployments.openpi.server` 按以下顺序选择源码 checkout：显式
`--openpi-root`、checkpoint 的 `checkpoints/` 祖先、已初始化的
`model/openpi`。因此已有外部 checkout 和 checkpoint 路径保持兼容；未能推导外部
checkout 时才使用固定 submodule。prepared training profile 仍严格绑定其中记录的
`openpi_checkout`，不会被 submodule fallback 静默替换。

当前经过闭环 server/client 握手验证的模型是：

| 项目 | 值 |
| --- | --- |
| server model id | `0712-wrist-cube` |
| checkpoint | `checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599` |
| artifact | checkpoint 下的 `policy_artifact.json` |
| 图像 | `left_wrist`、`right_wrist`（RGB） |
| state / action | 16D / 16D，连续 DMTAC gripper 标量 |
| server schema | `dexteleop_deploy_v2` |

`92599` 是当前保存的训练 checkpoint，并不表示 150k 训练已完成或已通过任务成功率评估。部署前应先完成无硬件预检和 dry-run。

远端 OpenPI 的历史配置名现已冻结，继续保留其训练时的 mask、worker 和 epoch 行为，供已有
checkpoint 解释、复现和回退。不要用同名配置续训成另一种输入语义。

修正后的配置使用一对一的 `-corrected-v1` 后缀。它们统一启用：

- `final_keep_mask` only，缺失时 fail closed；
- 10% original、40% real-image、50% procedural、0% black；
- 左右腕共享背景源/canvas，但使用独立 crop、漂移和 view seed；
- `num_workers=4` 与 `shared_worker_v1` epoch 同步；
- 在 policy metadata 中写入训练输入 contract、配置来源和 real-image manifest SHA256。

真实图片池使用远端
`assets/random_real_image/openimages_v7_validation/real_background_manifest_corrected_v1.json`。
该 manifest 使用相对 `image_root`，可随图片目录迁移；历史
`real_background_manifest.json` 保持不变。图片池或 final mask 不完整时，修正版训练会在首个
样本前停止，不会静默退化比例。

完整旧/新名称映射和回退说明见远端 OpenPI
`docs/dexteleop_corrected_training_configs.md`。本目录中早期的三个实验 patch 已被版本化配置方案
取代，不应再直接应用到包含历史 checkpoint 的 OpenPI checkout。

新的 writing front-only h16 训练默认选择 `...-task-ready-state-dropout-v2`
配置：保留 10% whole-state dropout，并增加每侧独立的 15% arm-pose
dropout；`num_workers` 固定为 4。既有 `...-task-ready-v1` 名称不变，作为
旧 checkpoint 的显式回退，禁止跨名称 resume。

## 1. 不可变契约

`policy_artifact.json` 是 checkpoint 的唯一 I/O 权威。客户端传入 `--policy-artifact-json` 后会强制校验：

- server model id 和 checkpoint step；
- v2 schema、16D state/action、layout hash；
- 左右 wrist 的逻辑相机键、OpenPI 输入键和 server slots；
- DMTAC 的连续 gripper 映射。

### Prepared training checkpoint 的部署恢复

`writing_stage1_0804_0805_*_v1.json` 这类训练 profile 会在进程内派生 OpenPI
config，不应把派生名称静态写回 OpenPI 的共享 registry。部署时由
`deployments.openpi.server --prepared-training-profile ...` 生成一次性的
checkpoint-bound artifact 和 registry，再由
`deployments.openpi.prepared_server_runtime` 在 server 进程内恢复同一 config。

绑定同时覆盖原 checkpoint artifact、训练 profile hash、base/derived config 名称、
asset id、camera slots、action horizon/mode/tail、layout hash、state-augmentation
契约和 checkpoint 内实际 norm-stats bytes。校验只在部署层生效，不改 checkpoint、
数据集、OpenPI checkout、`teleop_runtime` 配置或遥操默认值。完整远端命令见
[`cmd_list.md` 第 13 节](./cmd_list.md#13-08040805-front--absolute-prepared-checkpoint-部署绑定)。

### v2.2 raw 数据训练出的 relative 模型

从 v2.2 数据按 chunk 起始 `actual_tcp_pose` 在线转换 relative action 后训练的模型，
对外使用 v2.3 model-facing action 语义，但其 absolute state 数值分布仍属于 v2.2
双臂 raw base。此类 checkpoint 的 artifact 必须同时声明：

```json
"pose_training_data_contract": {
  "schema": "dexteleop_pose_training_data_v1",
  "origin": "v22_raw_identity_compatibility",
  "source_dataset_schema": "dexteleop_dataset_v2.2",
  "model_state_pose_frame": "canonical_desk",
  "runtime_pose_binding": "same_source_identity_raw_to_desk",
  "expected_runtime_source_id": "rm65_real_dual_arm_v1",
  "relative_action_world_transform_invariant": true
}
```

同时，`pose_action_chunk_contract.action_pose_mode` 和
`last_action_pose_mode` 必须都是 `relative`，其 pose-frame hash 必须绑定同一
`rm65_real_dual_arm_v1` identity contract。部署器会据此自动完成：

```text
runtime measured TCP/raw state
  -> identity raw-to-canonical-desk model state
  -> relative policy chunk
  -> current control-cut measured TCP anchor
  -> runtime absolute action
  -> safety / IK / backend
```

这不是 native v2.3 canonical-desk 模型。不得为它传入 RM65 workspace-near-boundary
non-identity runtime profile；错误绑定会在 backend 创建前 fail-closed。

服务、相机或机械臂启动前，先执行只读 contract 检查：

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --contract-check-only
```

该命令不连接 websocket、不打开相机或手部设备，也不创建机械臂 backend。通过时
JSON 应包含 `ok=true`、`action_pose_mode=relative`、
`pose_training_origin=v22_raw_identity_compatibility` 和
`runtime_pose_frame_contract_source=v22_identity_compatibility`。缺少 provenance、
pose hash 不匹配或误用 native-v2.3 语义都会直接失败；DexTeleop 不根据 checkpoint
路径或数据集名称猜测并补写这些字段。

不要手动将 wrist 图像伪装成 `front`，也不要用位置顺序猜测相机。默认握手只要求
上游 server 与 artifact 的协议兼容；`model_id` 和 `checkpoint_step` 不作为下游
切换模型实例的硬绑定。需要逐实例审计时，显式使用
`--strict-model-identity --model ... --checkpoint-step ...`。

### 旧 wrist-only checkpoint 的物理左右兼容

当前物理绑定是 `left_wrist=260522278055`、
`right_wrist=261122273924`。历史 wrist-only checkpoint 的 left/right model
slots 则来自旧的反向采集标签。使用以下版本化兼容 artifact，保持 checkpoint 权重和
原始数据不变：

```bash
export LEGACY_WRIST_COMPAT_ARTIFACT="$DEXTELEOP_ROOT/deployments/openpi/configs/model_rtc/0712-wrist-cube-92599-legacy-wrist-compat.policy_artifact.json"
```

server 仍从原 checkpoint 加载 params，但必须显式使用同一兼容 artifact 构造协议：

```bash
"$OPENPI_ROOT/.venv/bin/python" -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --policy-artifact-json "$LEGACY_WRIST_COMPAT_ARTIFACT" \
  --port 8078
```

client 同样传入该 artifact，而不是 checkpoint 内旧的 sidecar：

```bash
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --policy-artifact-json "$LEGACY_WRIST_COMPAT_ARTIFACT" \
  --deployment-profile rm65_bimanual_dmtac \
  --host "$SERVER_IP" --port 8078 \
  --contract-check-only
```

兼容 artifact 将 canonical runtime `right_wrist` 路由到旧模型 left slot，将
canonical runtime `left_wrist` 路由到旧模型 right slot。server 会校验它绑定的原
checkpoint artifact hash；client 会在创建相机或机器人前校验 camera-role binding
hash。训练入口默认按数据快照 serial 解析历史/新数据；显式
`--disable-camera-semantic-resolution` 仅作为回滚，不应用于混合 wrist 数据。

### RTC-4b-A OpenPI raw-RTC final-chunk 边界

这里的 raw-RTC 指 OpenPI 仍执行既有 `observation -> final action chunk` 推理，模型和 server 不接收未来动作；
RTC 只负责客户端 worker、timed plan、admission、TTL、epoch 和 emission。model-RTC（模型消费未来
prefix/mask 并返回 conditioning receipt）是独立分支，见下一节；raw-RTC本身不暗示该能力。

`deployments.openpi.client.rtc`已经可以把现有artifact约束的OpenPI final chunk接到通用RTC v2
capability/envelope/decoder seam，但当前只声明：

```text
scheduler = async_timed_plan
continuation = none
transport verification = offline fake transport + shared frontend fixture
production caller = explicit default-off raw profile only; send_actions=false
```

该适配器强制绑定完整artifact、完整server metadata、显式camera contract、exact action horizon、prepared
OpenPI input hash、layout/codec/frame/normalization hash，以及相互独立的continuous float与int64 discrete action
矩阵。absolute artifact默认继续使用既有`legacy_normalized_v1`结果；relative artifact必须声明structured
pose chunk、匹配的pose-frame hash并使用`typed_canonical_v1`，由原request observation cut的fixed measured-TCP
anchor在plan前解为canonical absolute action。artifact/server horizon不一致、relative legacy result、metadata drift、
短chunk、NaN、非法token、input hash mismatch和时钟回退都会fail closed。该context不进入OpenPI payload，模型和
server接口不变。带discrete输出的model-RTC continuation仍需独立typed prefix/mask/receipt gate。

这不是 model-RTC，也不授予相机、机械臂或生产 motion authority。现有 OpenPI launcher 仅在显式
`--rtc-profile raw_rtc_final_chunk_15_30_60_120_v1` 时创建 raw-RTC adapter；默认仍是同步路径，且首轮
profile 强制 `send_actions=false`。其作用是先用
可替换的model adapter证明artifact→payload→final output→absolute/relative canonical decode的离线完整性；换用其他本地模型或
模型服务时，通用RTC request/response/plan/emission语义保持不变，只替换具体backend与codec。鲁棒fixture见
`tests/test_openpi_rtc_final_chunk_adapter.py`。

### OpenPI trained model-RTC（default-off）

首版训练适配不改模型结构：外部DexTeleop launcher包装任意兼容OpenPI base config，在training transform中产生可选
`model_rtc.action_loss_mask`。普通OpenPI batch缺少该key时仍走原二元batch和全量mean loss；只有显式trained run才使用
prefix-conditioned suffix loss。base checkpoint只读，输出config/experiment/checkpoint目录独立，回退时直接使用原
config/checkpoint。

准备与启动顺序：

```bash
python -m deployments.openpi.training.prepare \
  --source-artifact-json deployments/openpi/configs/model_rtc/0712-wrist-cube-92599-legacy-wrist-compat.policy_artifact.json \
  --output-artifact-json deployments/openpi/configs/model_rtc/0712-wrist-cube-model-rtc-v1.template.policy_artifact.json \
  --output-model-id 0712-wrist-cube-model-rtc-v1 \
  --source-dataset-root /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success

python -m deployments.openpi.training.launch \
  --base-config 0712-wrist-cube \
  --base-checkpoint-dir /home/user/zc/openpi/checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599 \
  --artifact-template-json deployments/openpi/configs/model_rtc/0712-wrist-cube-model-rtc-v1.template.policy_artifact.json \
  --sampling-spec-json deployments/openpi/configs/model_rtc/0712-wrist-cube-model-rtc-v1.sampling.json \
  --source-dataset-root /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --camera-key right_wrist --camera-key left_wrist \
  --output-config-name 0712-wrist-cube-model-rtc-v1 \
  --exp-name EXPERIMENT --dry-run
```

这里必须从 legacy wrist compat artifact 派生 template，并保持其 artifact source-key
顺序。否则虽然历史数据可按 serial 转成规范左右，继承自 92599 的模型 input slot
语义却会同时被翻转，形成权重与输入不一致；launcher 会拒绝没有绑定 base artifact
hash 的替代路由。

### 普通 OpenPI corrected-v1 训练

常规 `scripts/train.py` 不经过上述 model-RTC transform。包含 wrist 输入的
`-corrected-v1` config 必须通过 serial-resolution wrapper 启动；wrapper 会先只读审计
每个 source 的 RealSense snapshot，再通过环境契约让每个 dataset worker 在读取视频和
mask 时按物理角色解析：

```bash
PYTHONPATH="$DEXTELEOP_ROOT:$OPENPI_ROOT/src" \
"$OPENPI_ROOT/.venv/bin/python" -m deployments.openpi.corrected_train \
  --dry-run --sample-check -- \
  0712-wrist-cube-corrected-v1 \
  --exp-name NEW_EXPERIMENT
```

`--sample-check` 还会在不创建 trainer、不加载模型权重的前提下实际 decode/transform 每个
source 的一个 sample。确认审计后同时去掉 wrapper 的 `--dry-run --sample-check` 才会训练。
wrapper 拒绝无后缀历史 config；
不得用 corrected config resume 无后缀旧 checkpoint。直接运行历史
`scripts/train.py 0712-wrist-cube` 仍是显式复现旧语义，不会自动纠正。

去掉`--dry-run`才会加载权重和训练。每个checkpoint同时写artifact、training-run manifest和checkpoint receipt；trained
deployment会在连接server/设备前验证三者。部署仅开放显式
`--rtc-profile model_rtc_flow_inpaint_shadow_15_30_60_120_v1`的`send_actions=false`conditioned shadow；target、
safety和dispatch仍由既有部署链路拥有，RTC adapter不接管。

trained checkpoint使用既有base config恢复模型；conditioned wire也必须显式启用：

```bash
python -m deployments.openpi.server \
  --checkpoint-dir /path/to/model-rtc/EXPERIMENT/STEP \
  --config BASE_CONFIG \
  --enable-rtc-continuation
```

不传`--enable-rtc-continuation`时，checkpoint server保持原有普通推理行为。该开关只开放模型侧conditioning
operation，不启动DexTeleop target、设备或动作下发。

服务端 registry 位于 `~/zc/openpi/configs/dexteleop_policy_registry.json`，必须包含：

```json
"0712-wrist-cube": {
  "config": "0712-wrist-cube",
  "checkpoint_dir": "checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599",
  "deploy_schema_version": "dexteleop_deploy_v2",
  "action_key": "actions",
  "action_discrete_key": "actions_discrete",
  "action_dim": 16,
  "action_discrete_dim": 0,
  "state_dim": 16,
  "state_discrete_dim": 0,
  "action_horizon": 50,
  "active_sides": ["left", "right"],
  "camera_slots": ["left_wrist_0_rgb", "right_wrist_0_rgb"]
}
```

## 2. 两套环境

不要混用环境：

| 机器/职责 | 根目录 | 环境 |
| --- | --- | --- |
| OpenPI server / GPU | `/home/user/zc/openpi` | `uv` / `.venv` |
| DexTeleop client / 硬件 | `/home/user/zc/DexTeleop` | Conda 环境 `${DEXTELEOP_ENV_NAME:-dexteleop310}` |

以下变量应在 **DexTeleop client 机器**上设置；若 server 与 client 是同一台机器，`SERVER_IP=127.0.0.1` 仅适用于本机检查，局域网部署应使用可路由 IP。

```bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export DEXTELEOP_PYTHON="$(command -v python)"

export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export OPENPI_MODEL=0712-wrist-cube
export OPENPI_ARTIFACT=$OPENPI_ROOT/checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599/policy_artifact.json

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

```

## 3. 启动 OpenPI server

在 GPU/server 机器执行。`PYTHONPATH` 必须指向 DexTeleop，因为 OpenPI 的训练配置会加载共享的数据集接口。

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model 0712-wrist-cube \
  --host 0.0.0.0 \
  --port 8078
```

server 应保持在该终端运行。另开一个终端确认 health 和 metadata：

```bash
curl --fail --silent http://127.0.0.1:8078/healthz
curl --fail --silent http://127.0.0.1:8078/metadata
```

metadata 至少应显示：

```text
model_id=0712-wrist-cube
deployment.checkpoint_step=92599
deploy_schema_version=dexteleop_deploy_v2
local_robot.output_action_dim=16
dataset_adapter.camera_slots=[left_wrist_0_rgb,right_wrist_0_rgb]
```

## 4. 无硬件 websocket 预检

此命令只连接 server、读取 metadata 并验证 artifact；不打开相机、不连接机械臂或夹爪、不发送动作。

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -c '
from deployments.client import load_policy_artifact
from deployments.openpi.client.policy import create_policy
import os

artifact = load_policy_artifact(os.environ["OPENPI_ARTIFACT"])
policy = create_policy(
    host=os.environ["SERVER_IP"],
    port=int(os.environ["OPENPI_PORT"]),
    policy_artifact=artifact,
)
print(policy.name)
print(policy.metadata["validation"])
'
```

任何协议字段（16D action、layout hash、camera slot、state/action key 或 schema）
不匹配都必须先修复；默认的 model id/step 变化不需要绕过校验。需要精确实例审计时
再显式启用 `--strict-model-identity`。

## 5. qpos-init 与 v2.2 数据集

`--policy-qpos-init-dataset` 用于把机械臂安全移动到数据集的指定帧 qpos，随后才允许策略启动。v2.2 数据集的 `action` 是 16D；运行时现在会根据数据集 `meta/info.json` 内的 `policy_io_layout` 显式桥接成内部 canonical 26D debug/hold action，而不是把 16D 当成旧 action 解析。

实际 qpos 目标不从模型 action 猜测，`--policy-qpos-init-source auto` 按以下优先级选择该帧有限且完整的 6D joint target：

1. `replay_qpos`
2. `observation.arm.qpos`
3. `observation.*.current_qpos`
4. `observation.*.actual_qpos`

对 `bimanual-cube-new-success` 的 episode 0/frame 0，当前会选择 `observation_arm_qpos`，左右臂各为 6D。读取/规划失败时策略保持被阻塞，不能自动退化为任意当前姿态。

连续任务明确要求保留当前姿态时，可显式传入 `--no-init-qpos`。该参数在共享 runtime
解析层覆盖 deployment profile 或命令行注入的 dataset-frame qpos-init 输入；当 profile 的
startup source 为 `current` 时，该同位姿 Cartesian 初始化会直接判定为已满足，无需再按
`g` 发送一段冗余保持流。手动部署仍需等待 observation/碰撞/受力门正常后按 `A` 确认并启动
策略。兼容拼写 `--no--init-qpos` 也可解析，但文档和新命令统一使用 `--no-init-qpos`。该覆盖
不关闭 ready confirm、碰撞、受力、observation 或 dispatch 安全门，也不提供跨任务 action
queue/model history 的连续 handoff。

可在没有硬件的情况下验证 qpos-init 数据读取：

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -c '
from teleop_runtime.deploy.policy_qpos_init import load_deploy_policy_qpos_init_plan
plan = load_deploy_policy_qpos_init_plan(
    dataset_root="/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success",
    parquet_path=None,
    episode_index=0,
    frame_index=0,
    action_column="action",
    active_sides=("left", "right"),
    qpos_init_source="auto",
)
print(plan.metadata["action_encoding"], plan.target_sources_by_side)
'
```

预期包含 `v22_layout_to_legacy_q6_bridge` 和两个 qpos source；若不是，停止并检查数据集 metadata、episode 和 layout hash。

## 6. 相机/推理 dry-run（不发送硬件动作）

以下命令会打开两个 wrist 相机、读取真实观测、请求推理、执行动作归一化和安全过滤，但**不**加 `--send-actions`，因此不会向机械臂或夹爪发送命令。先检查两路图像、server validation、动作限幅和 fault 日志。

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --active-sides both \
  --arm-execution-profile impedance_40hz \
  --enable-hands --hand-backend dmtac-gripper --dmtac-config latest \
  --camera-mode multi \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --require-camera-frame --require-ready-observation \
  --action-queue-policy replace --min-action-queue 0 \
  --init-source current \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --max-steps 100 --debug
```

`multi` 配置可以含 front 相机；artifact 会只选择 `left_wrist` 和 `right_wrist`。缺少任一 artifact-required view 会 fail-closed。

如需让 OpenPI 的每个 action chunk 只执行前 `X` 步，再继续下一个 chunk，在上述
real 命令中加入：

```bash
--action-chunk-execution-steps X --action-queue-policy replace --min-action-queue 0
```

不传 `--action-chunk-execution-steps` 时仍执行完整返回 chunk。该开关只截断 chunk
边界，不改变 pose/relative action、anchor、safety 或 position 控制语义；当前阶段也
不额外启动推理线程。

## 7. 真机阻抗/力控执行（L5）

下面命令会连接真实 RM65、使能阻抗/力控后端、执行 dataset-frame qpos 初始化，并在启动许可后向双臂和 DMTAC 夹爪发送策略动作。只应在以下条件全部满足且操作者现场确认后执行：

- 人员远离工作空间，急停可用；
- 两臂 IP、tool frame、碰撞场景和 DMTAC assignment 已确认；
- 两路 wrist 相机在 dry-run 中已验证；
- server metadata 与 artifact 已通过第 4 节预检；
- dataset frame 0 是可安全到达的起始姿态；
- 已了解 `g` 会开始 qpos-init/策略流程，`A` 只会在 ready 后放行策略。

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --active-sides both \
  --arm-execution-profile impedance_40hz \
  --enable-hands --hand-backend dmtac-gripper --dmtac-config latest \
  --camera-mode multi \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --require-camera-frame --require-ready-observation \
  --action-queue-policy replace --min-action-queue 0 \
  --init-source current --wait-for-ready-confirm \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --left-arm-ip "$LEFT_ARM_IP" --right-arm-ip "$RIGHT_ARM_IP" \
  --send-actions --debug
```

`impedance_40hz` 选择经过测试的阻抗分支：policy 15 Hz（默认时）、臂后端 40 Hz、无 CANFD follow。OpenPI 部署会额外显式启用以下双速率 IK；这是部署专用覆盖，**不改变遥操**：

| 周期 | 工作 | 预算/优先级 |
| --- | --- | --- |
| 15 Hz | 独立的完整 Pinocchio IK（每臂一个 worker） | 66.7 ms；只使用 `rm65_pinocchio`，不会并发调用 RM65 SDK IK。超时结果丢弃，不积压旧 pose。 |
| 40 Hz | 力读取、导纳更新、限幅和安全 joint 输出 | 完整 IK 运行时只复用上一条安全命令；否则最多进行 6 ms 的本地 IK。 |

因此 40 Hz 不是“重复执行完整 IK”。完整求解在运行时优先，40 Hz 力控循环仍持续运行；若 pose 落入难解空间，输出保持最后安全关节命令，而不会让陈旧求解阻塞或追赶。不要改为位置分支，也不要添加 `--start-init_qpos`，除非已明确审查其安全影响。

手动启动（不带 `--auto-start`）时，启动日志必须先显示：

```text
[Deploy] impedance prepared without periodic output; press g/H/A to enable it, or use --auto-start.
```

这一步只完成力传感器/安全参考预热，**不会**启动 40 Hz 周期输出、执行 IK 或发送保持命令。按 `g` 才开启阻抗输出并开始 qpos-init；qpos-init 和 startup-init 完成后，按 `A` 才会请求 OpenPI 策略动作。若尚未按 `g` 就看到 `IK hold_last_safe`，说明远端仍在运行修复前代码，不能继续真机流程。

## 8. 失败定位

| 现象 | 含义与处理 |
| --- | --- |
| `must have shape (26,), got (16,)` | 使用了旧 qpos-init 代码；同步 `teleop_runtime/deploy/policy_qpos_init.py` 后重新运行第 5 节只读检查。 |
| `model_id_mismatch` 或 `checkpoint_step_mismatch` | 只有显式启用 `--strict-model-identity` 时才是阻断项；否则检查实际协议字段是否匹配。 |
| `camera_slots` 或 required image key 不匹配 | 相机配置缺少 wrist logical key，或试图用 front 替代 wrist；修正配置。 |
| qpos-init 缺少 target | 指定帧没有有效 qpos audit；换经过审查的 episode/frame，或修复数据集。 |
| `DMTAC ... config is missing` | `latest` assignment、USB 映射或 side 配置不完整；先解决硬件配置，不要允许缺失后继续执行。 |
| `deployment_dual_rate_ik.deadline_miss_count` 增长（已按 `g` 后，`--debug` record） | 15 Hz 完整 IK 没能在 66.7 ms 内完成。该次结果已丢弃，40 Hz 仍保持力/导纳和最后安全 joint command。不要按 `A` 放行策略，不要增大 deadline 来掩盖问题；检查 tool/TCP、当前 qpos/反馈、力传感器偏置、Pinocchio 迭代耗时和策略 pose 是否进入难解空间。 |
| `deployment_dual_rate_hold_last_safe`（已按 `g` 后，`--debug` record） | 本地 6 ms IK 没有安全候选，或完整 worker 正在运行；这是受控保持而非新的策略动作。连续出现时不要按 `A`，先按上一行检查完整 IK deadline 和姿态可达性。 |
| `impedance IK hold_last_safe ... hard deadline`（已按 `g` 后） | 这属于旧的单线程阻抗路径，或说明部署双速率覆盖没有生效。停止真机流程，确认运行的是同步后的 `teleop_runtime/deploy/run_policy.py`，并检查 debug config 中 `deployment_dual_rate_ik_enabled=true`。 |
| server 无法起动 CUDA | 确认在 OpenPI uv 环境启动，且设置了 `JAX_PLATFORMS=cuda`。 |

## 9. 代码边界

```text
OpenPI server
  scripts/serve_dexteleop_policy.py
  configs/dexteleop_policy_registry.json

DexTeleop OpenPI client
  deployments/openpi/client/real.py       thin CLI + artifact forwarding
  deployments/openpi/client/policy.py     websocket contract + v2.2 adapter

Shared real deployment runtime
  teleop_runtime/deploy/run_policy.py
  teleop_runtime/deploy/policy_qpos_init.py
```

OpenPI 不应导入或控制真机 SDK；真机动作只能通过 `teleop_runtime` 的初始化、限幅、状态检查和 dispatch 路径执行。
