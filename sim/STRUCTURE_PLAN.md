# DexTeleop 仿真、模型训练与 Sim/Real 部署结构规划

## 1. 文档目标

本文档定义 DexTeleop 面向长期演进的项目组织规范，覆盖：

- MuJoCo 及未来其他仿真后端的职责边界；
- 高吞吐批量 RL 采样；
- 通用 RL、BC 和离线训练；
- RL policy 批量生成训练数据；
- sim/real 共用模型、动作和观测契约；
- 模型制品、数据集、运行产物和测试的组织方式；
- 现有代码向目标结构渐进迁移的方法。

本规划采用顶层目录 `model/` 表示通用学习域。其语义与原规划中的 `learning/` 完全相同：它不仅存放网络定义，还负责 RL、BC、训练、在线采样、评估和模型导出。不得把 `model/` 理解为只存放神经网络层的目录。

当前数据集执行器、仿真场景和任务实现可以继续使用临时版本。近期实现优先级仍然是 MuJoCo 批量采样效率和稳定性，目录调整不得以牺牲热路径性能或造成大规模无收益迁移为代价。

## 2. 总体设计结论

长期架构遵循以下原则：

1. `sim/` 产生仿真经验，但不拥有 PPO、SAC、BC 等训练算法。
2. `model/` 训练和评估策略，但算法层不感知 MuJoCo 内部结构。
3. `robot_motion/` 统一 sim/real 的动作语义和部署运行时，但不包含训练、仿真或真实硬件实现。
4. `PolicyArtifactContract` 统一模型输入输出，`dataset_contracts/` 统一 sim/real 的数据格式及格式绑定 I/O。
5. 文件归属应由“允许哪些模块依赖它”决定，而不是由“当前是谁在调用它”决定。

## 3. 推荐的顶层目录

```text
DexTeleop/
├── robot_motion/          # sim/real 共用运动、动作和部署运行时
├── sim/                   # 纯仿真：环境、任务、物理后端、渲染
├── model/                 # RL、BC、训练、在线采样、评估和导出
├── dataset_contracts/     # 数据格式、Policy I/O 契约及格式绑定读写
├── dataset_interface/     # 训练侧数据读取、增强和窗口采样
├── deployments/          # 模型服务、客户端和模型族集成
├── teleop_runtime/        # 真实机器人采集与部署适配
├── hardware/              # 真实设备驱动
└── tests/                 # 单元、集成、契约和性能测试
```

### 3.1 归属判断表

| 代码内容 | 应归属 |
|---|---|
| `mjModel`、`mjData`、contact、`mj_step` | `sim/mujoco/` |
| reward、termination、reset、domain randomization | `sim/tasks/` |
| PPO、SAC、BC loss、optimizer、rollout buffer | `model/` |
| policy forward、网络结构、checkpoint loader | `model/policies/` |
| 数据 schema、字段定义、版本兼容 | `dataset_contracts/` |
| canonical parquet、episode writer、manifest writer | `dataset_contracts/io/` |
| augmentation、trajectory window sampling | `dataset_interface/` |
| 坐标系、单位、TCP、动作限幅、canonical action | `robot_motion/` |
| 模型服务通信、OpenPI 等模型族客户端 | `deployments/` |
| 相机、机械臂 SDK、真实设备采集 | `teleop_runtime/` 或 `hardware/` |

## 4. `sim/` 的目标结构

```text
sim/
├── README.md
├── STRUCTURE_PLAN.md
├── core/
│   ├── protocols.py           # ScalarEnv、VectorEnv、TaskSpec 协议
│   ├── types.py               # EnvStep、BatchStep 等基础类型
│   ├── seeding.py             # 可复现随机种子派生
│   └── registry.py            # 环境和任务注册，可选
├── contracts/
│   ├── scene.py               # 场景资产、实体和相机约定
│   ├── timing.py              # physics/control/render 频率
│   ├── simulation.py          # 仿真配置与兼容信息
│   └── manifest.py            # 仿真运行 manifest
├── tasks/
│   └── cube_lift/
│       ├── config.py
│       ├── action.py
│       ├── observation.py
│       ├── reward.py
│       ├── termination.py
│       ├── reset.py
│       └── metrics.py
├── mujoco/
│   ├── modeling/              # MJCF 构建、组合和验证
│   ├── artifacts/             # MJB/模型缓存管理
│   ├── runtime/               # worker、vector executor
│   ├── adapters/
│   │   ├── robot_motion.py    # robot_motion Backend/Dispatcher
│   │   ├── gymnasium.py
│   │   └── collection.py
│   ├── tasks/
│   │   └── cube_lift/
│   │       ├── binding.py     # joint/body/sensor 索引绑定
│   │       ├── kernel.py      # MuJoCo 性能敏感热路径
│   │       └── reference_env.py
│   ├── replay/
│   ├── rendering/
│   ├── diagnostics/
│   └── benchmarks/
├── collection/
│   ├── trajectory_adapter.py
│   └── rollout_export.py
└── configs/
    ├── scenes/
    ├── tasks/
    └── runtimes/
```

### 4.1 `sim/tasks/` 与 `sim/mujoco/tasks/`

`sim/tasks/<task>/` 存放后端无关的任务语义：action 含义、observation 字段、reward、success/failure、reset 分布和评测指标。

`sim/mujoco/tasks/<task>/` 存放 MuJoCo 实现细节：`qpos/qvel` 索引，body、geom、site、sensor ID，contact 查询，`mjModel/mjData` 操作和性能 kernel。

当前只有 MuJoCo 后端时，不应为了形式上的“后端无关”立即搬迁全部任务代码。只迁移已经确认是纯任务语义的部分并保留兼容导入。尚无第二个使用方的抽象，应优先保持现有实现清晰和高效。

### 4.2 仿真环境公共接口

`sim/core/protocols.py` 应提供轻量公共协议：

- `ScalarEnv`：单环境 reference/debug 实现；
- `VectorEnv`：批量同步或异步环境；
- `TaskSpec`：动作、观测、reward、termination 语义；
- `EnvFactory`：可序列化的 worker 构建入口。

公共协议不得暴露 `mjModel` 或 `mjData`，否则 `model/` 会被迫依赖 MuJoCo。

## 5. `model/` 的目标结构

```text
model/
├── README.md
├── core/
│   ├── protocols.py           # Policy、Learner、Sampler 协议
│   ├── types.py
│   ├── device.py
│   └── seeding.py
├── data/
│   ├── policy_batch.py
│   ├── rollout_batch.py
│   ├── trajectory_batch.py
│   ├── replay_buffer.py
│   └── sources/
│       ├── online_sim.py
│       └── offline_dataset.py
├── policies/                  # 可独立推理的本项目原生模型
│   ├── common/
│   └── <policy_family>/
│       ├── config.py
│       ├── model.py
│       ├── preprocessing.py
│       ├── inference.py
│       └── checkpoint.py
├── algorithms/
│   ├── rl/
│   │   ├── ppo/
│   │   └── sac/
│   └── il/
│       └── bc/
├── sampling/
│   ├── vector_sampler.py
│   ├── rollout_collector.py
│   ├── episode_tracker.py
│   └── evaluation.py
├── model_adapters/
│   ├── policy_adapter.py
│   └── distribution_adapter.py
├── artifacts/
│   ├── training_manifest.py
│   ├── exporter.py
│   ├── checkpoint.py
│   └── validation.py
├── runners/
│   ├── train_rl.py
│   ├── train_bc.py
│   ├── evaluate.py
│   ├── collect.py
│   └── export_policy.py
└── configs/
    ├── rl/
    ├── bc/
    ├── policies/
    ├── collection/
    └── evaluation/
```

### 5.1 `model/` 的明确语义

`model/` 是完整学习域，包含在线 RL、BC、离线学习、数据混合、policy 网络、训练适配、checkpoint、评估、导出和训练任务编排。

`model/policies/` 才是专门存放可推理模型定义的子目录。它必须保持 inference-safe：不得导入 MuJoCo、真实硬件、trainer 或 dataset loader。部署代码如需本地加载原生模型，只允许依赖这个轻量子包及其明确依赖，不能通过 `model/__init__.py` 隐式加载完整训练栈。

OpenPI 等外部模型族可继续位于 `deployments/<model_family>/`，不为形式统一强行迁移。

### 5.2 算法层不得直接依赖 MuJoCo

正确关系：

```text
model/algorithms/rl/ppo
    ↓
model/core + model/data
    ↓
ScalarEnv/VectorEnv protocol
```

只有组合层可以接入具体仿真：

```text
model/runners/train_rl.py
    ↓
model/sampling
    ↓
sim 的公共环境接口
```

PPO、SAC 和 BC 实现中不得出现 `import mujoco` 或对 `sim.mujoco` 内部模块的依赖。

## 6. 训练期内存数据模型

### 6.1 `RolloutBatch`

供在线 RL 使用：

```text
observations      [T, N, ...]
actions           [T, N, ...]
rewards           [T, N]
terminated        [T, N]
truncated         [T, N]
values            [T, N]
log_probs         [T, N]
episode_ids       [T, N]
bootstrap_values  [N]
```

它服务于高吞吐采样、GAE 和 policy update。MuJoCo step 热路径不得同步写 parquet、图片、视频或 JSON。

### 6.2 `PolicyBatch` 与 `TrajectoryBatch`

供 BC、离线 RL 和监督训练使用：

```text
inputs
target_actions
attention_mask
time_mask
domain
episode_id
metadata
```

`dataset_interface/` 负责将磁盘数据转换为训练 batch。算法只接收显式 batch，不关心数据来自 MuJoCo、真人采集还是 replay。Gymnasium `info` 只用于临时诊断和可选指标，不作为稳定训练 schema。

## 7. 标准训练与数据流程

### 7.1 在线 RL

```text
sim vector environments
    → model.sampling.VectorSampler
    → RolloutBatch
    → PPO/SAC learner
    → checkpoint + training manifest
```

热路径只保留 policy inference、MuJoCo step、reward/termination 和 rollout buffer 写入。渲染、视频、数据集落盘和详细诊断必须关闭、降频或异步执行。

### 7.2 RL policy 批量采集 BC 数据

```text
RL policy / scripted expert
    → vector sim environments
    → model rollout collector
    → sim collection adapter
    → dataset_contracts.io EpisodeWriter
    → canonical dataset
```

每条仿真 episode 至少记录：

- `domain=sim`；
- policy artifact ID；
- task、scene 和 config hash；
- seed 和 domain randomization 参数；
- simulator/backend/version；
- control timestep；
- success/failure reason。

### 7.3 BC 训练

```text
dataset_interface
    → sequence/window sampling
    → PolicyBatch
    → model.algorithms.il.bc
    → checkpoint
    → artifact exporter
```

### 7.4 混合 sim/real 数据

不得只靠目录名区分来源。每个 episode 应显式保存：

```text
domain: sim | real | replay
source_policy
task_id
robot_profile
observation_profile
action_profile
```

模型训练配置负责决定 sim/real 比例、各 domain loss 权重、失败轨迹过滤、privileged 标签和相机缺失策略。

## 8. `dataset_contracts/io/` 规划

现有 `dataset_contracts/README.md` 已明确将 canonical Parquet row I/O、codec 和 manifest 纳入该包，并允许操作级延迟导入 `pyarrow`、`lz4`、`zstandard`。因此不再新建顶层 `dataset_io/`，而是在 `dataset_contracts/` 内建立职责明确的 `io/` 子包：

```text
dataset_contracts/
├── schemas/                    # 可在后续迁移中逐步形成
├── codecs/                     # 格式级编码与解码
├── policy_artifact.py
├── policy_io.py
└── io/
    ├── episode_builder.py
    ├── episode_writer.py
    ├── episode_reader.py
    ├── manifest_writer.py
    ├── image_writer.py
    ├── parquet_writer.py
    ├── validation.py
    └── transaction.py
```

职责关系：

- `dataset_contracts` 根层和 schema 模块定义“数据是什么”；
- `dataset_contracts/io` 定义“如何按该格式可靠读写”；
- `dataset_interface` 定义“如何转换成训练样本”；
- `sim` 定义“如何取得仿真原始数据”；
- `teleop_runtime` 定义“如何取得真实设备原始数据”。

允许进入 `dataset_contracts/io/` 的内容必须与 canonical dataset format 直接绑定，包括 row/chunk reader、writer、codec、manifest、完整性校验和原子提交。采集 session 编排、异步队列、背压、相机抓帧、MuJoCo 状态提取和真实硬件访问仍分别属于 `sim/`、`teleop_runtime/` 或其 collection adapter，不能因“负责写数据”而进入 `dataset_contracts/`。

后续逐步把 `sim/mujoco/dataset_v22` 中通用且格式绑定的写入能力迁移到 `dataset_contracts/io/`。MuJoCo 侧只保留仿真字段适配和采集编排。

## 9. 同一模型在 sim/real 中部署

“同一模型”不只是同一个 checkpoint，还必须保证输入字段、图像顺序和预处理、state 顺序和单位、action 维度和单位、action frame、absolute/relative 语义、TCP、control frequency、action horizon 和 normalization 完全一致。

这些内容继续由现有 `PolicyIOLayout` 和 `PolicyArtifactContract` 统一约束，不得在 `sim/` 或 `model/` 中建立第二套部署契约。

```text
SimObservationSource ─┐
                      ├→ Policy → CanonicalActionPipeline → Dispatcher
RealObservationSource ┘
```

| 环节 | sim | real |
|---|---|---|
| ObservationSource | MuJoCo state/camera | 相机和机械臂状态 |
| Policy | 完全相同 | 完全相同 |
| action preprocessing | `robot_motion` | `robot_motion` |
| Dispatcher | MuJoCo actuator | 真实机器人 backend |

仿真部署测试不得让模型直接写 `qpos`，否则会绕过真实部署使用的坐标转换、限幅、时间语义和 canonical action pipeline。

### 9.1 Privileged observation

在线 RL 可能使用物体真值位姿、contact force、无噪声速度或其他仿真内部状态。使用这些输入的 policy 默认不可部署。

```json
{
  "deployability": {
    "status": "non_deployable",
    "reasons": ["uses object_ground_truth_pose"]
  }
}
```

只有全部输入都能由真实 `ObservationSource` 提供，且通过 artifact compatibility gate 后，才允许导出部署用 `policy_artifact.json`。不能伪造真实端不存在的输入。

## 10. 模型制品规范

```text
artifacts/<model_id>/<version>/
├── checkpoint.safetensors
├── policy_artifact.json
├── training_manifest.json
├── normalization.npz
├── metrics.json
└── checksums.json
```

- `policy_artifact.json`：沿用现有权威部署输入输出契约；
- `training_manifest.json`：算法、环境、数据、代码和随机种子；
- `normalization.npz`：训练和部署共用归一化参数；
- `metrics.json`：训练与评估摘要；
- `checksums.json`：检查损坏、不完整复制和错误组合。

`training_manifest` 至少记录：

```text
algorithm
policy_family
task_id
environment_contract_hash
scene_hash
policy_io_layout_hash
dataset IDs/hashes
code revision
config hash
seed
dependency versions
parent checkpoint
```

纯仿真且不可部署的 RL checkpoint 可以只有 training manifest。通过 deployability 检查后才允许导出完整 PolicyArtifact。

## 11. 依赖方向和禁区

```text
dataset_interface ─┐
robot_motion ──────┼──→ dataset_contracts（含 io）
sim ───────────────┤
deployments ───────┘

sim ──→ robot_motion
model/policies ──→ dataset_contracts
model algorithms ──→ model/core + model/data + model/policies
model runners/sampling ──→ sim + dataset_interface
deployments ──→ robot_motion + model/policies（仅本地原生模型）
teleop_runtime ──→ robot_motion + dataset_contracts/io + deployments client
```

`model/algorithms` 只依赖 `model/core`、`model/data` 和抽象协议；`model/runners`、`model/sampling` 或 source adapter 才能依赖 `sim` 和 `dataset_interface`。

硬性禁止：

- `robot_motion` 导入 `sim`、`model`、`teleop_runtime`；
- `dataset_contracts` 导入运行时、硬件、Torch 或 MuJoCo；
- `dataset_contracts/io` 导入 MuJoCo、真实设备 SDK、采集 runtime 或训练代码；
- `sim` 包含 PPO、SAC、BC trainer；
- `model/algorithms` 导入 `mujoco` 或 `sim.mujoco`；
- `model/policies` 导入仿真、硬件、trainer 或 dataset loader；
- `deployments` 包含训练循环和训练数据增强；
- 生产代码依赖 `tests/`；
- 使用无明确边界的 `utils.py`、`common.py`、`helpers.py` 持续堆积逻辑。

应增加 AST/import boundary 测试，并在 CI 中阻止反向依赖。

## 12. 文件命名规范

- `protocols.py`：`Protocol`/ABC，不放具体运行逻辑；
- `types.py`：轻量值类型；
- `contracts.py`：稳定、版本化、可序列化结构；
- `config.py`：配置结构及校验；
- `runtime.py`：生命周期编排，不解析 CLI；
- `cli.py`：只解析参数并调用 runner；
- `adapters/<target>.py`：跨边界转换；
- `validation.py`：纯检查，尽量无副作用；
- `manifest.py`：可追溯运行元数据；
- `benchmark_*.py`：只放 benchmark 目录；
- `reference_*.py`：慢但清晰的正确性基准；
- `kernel.py`：性能敏感热路径。

避免增加职责模糊的 `utils.py`、`common.py`、`misc.py`、`base.py` 和 `manager.py`。各包的公共 `__init__.py` 必须轻量；导入 contract 或 type 时不得隐式加载 MuJoCo、Torch、相机 SDK 或完整训练栈。

## 13. 配置规范

配置按领域归属：

```text
sim/configs/scenes/
sim/configs/tasks/
sim/configs/runtimes/
model/configs/rl/
model/configs/bc/
model/configs/policies/
model/configs/collection/
deployments/configs/
```

要求：

- 使用结构化、可校验配置；
- task、scene、robot profile 使用稳定 ID；
- 关键契约包含 schema version；
- 保存解析后的完整配置和 content hash；
- 机器相关路径通过启动配置或环境变量注入；
- 不写死最终采样服务器路径；
- 本机作为长期 debug 机器，应有轻量默认配置，但不能成为生产吞吐基准的硬编码依据。

## 14. 测试组织

```text
tests/
├── unit/
│   ├── sim/
│   ├── model/
│   ├── dataset_contracts/
│   └── robot_motion/
├── integration/
│   ├── sim_model/
│   ├── sim_dataset/
│   ├── policy_deploy/
│   └── sim_real_contract/
├── contracts/
│   ├── dependency_boundaries/
│   ├── policy_artifacts/
│   └── dataset_compatibility/
└── performance/
    └── sim/
```

重点覆盖：

1. scalar/vector environment 语义一致；
2. reference/fast kernel 一致；
3. 固定 seed 可复现；
4. rollout batch 正确区分 `terminated` 与 `truncated`；
5. sim/real observation adapter 产生相同 layout；
6. checkpoint normalization 与部署一致；
7. PolicyArtifact 不兼容时 fail fast；
8. dataset writer 中断后不产生看似完整的 episode；
9. 通用算法目录不存在 MuJoCo import；
10. `robot_motion` 不存在 sim/real 反向依赖；
11. 批量吞吐和长时间稳定性不发生性能回退。

当前测试不必一次性全部移动。优先增加分类、性能基线和依赖边界检查。

## 15. 分阶段迁移计划

### 阶段 0：冻结架构边界

- 将本文档作为新增代码的归属依据；
- 增加 import boundary 测试；
- 禁止新的训练算法进入 `sim/`；
- 禁止新的通用 writer 进入 `sim/mujoco/dataset_v22`；
- 现有临时任务和场景继续工作，不做无收益重写。

### 阶段 1：建立 `dataset_contracts/io/`

- 抽取 episode builder/writer；
- 保持现有 dataset schema 不变；
- MuJoCo 与真实采集通过 adapter 使用同一 writer；
- 建立原子提交、中断恢复和完整性校验。

### 阶段 2：建立最小 `model/`

优先实现 `PolicyBatch`、`TrajectoryBatch`、offline dataset source、BC runner、checkpoint/training manifest 和 artifact exporter。BC 依赖面较小，适合先验证数据、训练、导出和部署契约的完整链路。

### 阶段 3：统一仿真批量采集

- vector sampler；
- scripted/RL policy collector；
- rollout-to-dataset adapter；
- 批量 episode metadata；
- 异步落盘和背压控制。

### 阶段 4：接入在线 RL

- `RolloutBatch`；
- GAE；
- PPO；
- evaluator；
- checkpoint resume；
- 多进程采样；
- 性能与长稳回归测试。

初期只实现一套清晰、可验证的 PPO，不同时铺开多套算法。SAC 等在环境和 batch 契约稳定后再接入。

### 阶段 5：验证 sim/real 一致性

- MuJoCo `ObservationSource`；
- MuJoCo `Dispatcher`；
- 同一 artifact 通过 `robot_motion.deploy` 在 sim 中运行；
- observation/action parity 测试；
- real shadow 模式只推理、不下发；
- 真实设备控制继续遵守独立安全确认流程。

### 阶段 6：渐进整理 `sim/`

- 分离任务语义与 MuJoCo binding；
- 出现第二个任务复用需求后，将 task-specific vector runtime 收敛为通用 executor；
- 保留兼容导入和迁移期告警；
- 删除确认无调用方的旧实现。

## 16. 新文件归属检查清单

新增文件前依次判断：

1. 包含 MuJoCo 数据结构或 API：进入 `sim/mujoco/`。
2. 描述 task reward/reset/termination：进入 `sim/tasks/`。
3. 包含 loss、optimizer、gradient、rollout/replay buffer：进入 `model/`。
4. 只描述可独立推理的 policy 网络：进入 `model/policies/`。
5. 定义稳定数据字段或 Policy I/O：进入 `dataset_contracts/`。
6. 实现 canonical 格式绑定的数据持久化：进入 `dataset_contracts/io/`；采集编排留在数据来源侧。
7. 实现训练侧采样或增强：进入 `dataset_interface/`。
8. 定义 sim/real 共用动作、坐标、时间或部署流程：进入 `robot_motion/`。
9. 涉及模型服务和传输：进入 `deployments/`。
10. 直接操作真实设备：进入 `teleop_runtime/` 或 `hardware/`。

若一个文件同时命中多个领域，应拆成契约、适配和实现，而不是将跨域依赖隐藏在一个大文件中。

## 17. 近期执行重点

1. 保持现有 MuJoCo 临时任务、数据集执行器和场景可用；
2. 持续优化批量 step、reset、IPC、模型缓存和 worker 生命周期；
3. 建立稳定的吞吐、扩展性和长时间运行基线；
4. 用 import boundary test 阻止架构继续恶化；
5. 先新增 `model/` 和 `dataset_contracts/io/` 的最小骨架，再逐个迁移调用方；
6. 不进行一次性大规模目录重排；
7. 每次迁移保持兼容入口、测试通过且性能不回退。

最终目标链路：

```text
MuJoCo 批量采样
    → model 在线 RL
    → RL policy 批量生成标准数据集
    → model BC/离线训练
    → PolicyArtifactContract 导出
    → robot_motion.deploy
    → sim 验证 / real shadow / real 部署
```

该链路中，仿真、模型训练、数据持久化、运动控制和真实设备各自只有一个明确责任域，避免 `sim/` 演变为同时包含环境、算法、数据、模型和部署的巨型目录。
