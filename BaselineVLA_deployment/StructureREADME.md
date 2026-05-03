# ManipRep 结构与开发说明

本文面向开发和维护，说明框架目录、数据接口、模型/训练边界和扩展方式。运行命令与部署说明见 [`README.md`](./README.md)。

## 1. 总体设计

ManipRep 的核心目标是把以下几层解耦：

- 原始数据格式：LeRobot metadata、DeskTaskTac parquet、HumanMix 等。
- 统一训练样本：所有数据最终收敛到 canonical sample。
- 中间表示：vector tactile、future tactile、point-cloud tactile 等。
- 模型族：MLP、VLA、Diffusion。
- 实验范式：Vanilla、GTMed、FullTrain、GenMed。
- 运行时：训练、评估、批量实验、真机部署。

主链路：

```text
Configs -> Manifest Loader -> Raw Dataset -> CanonicalSample
        -> Input Adapter -> Model Builder -> Trainer -> Checkpoint
        -> Evaluation / Deployment
```

## 2. 目录职责

```text
ManipRep/
├── Assets/                 # 稳定资产：URDF、MANO、标定、mesh
├── Configs/                # Hydra 配置
│   ├── data/               # 数据源、字段、窗口、loader、增强
│   ├── experiment/         # vanilla / gtmed / fulltrain / genmed
│   ├── model/              # minimal_mlp / basic_vla / basic_diffusion
│   └── train/              # epoch、lr、checkpoint、wandb
├── Data/                   # 数据挂载点，只存数据，不放复杂逻辑
│   ├── RawData/
│   └── MedData/
├── Datasets/               # manifest、dataset、canonical sample、adapter
├── GenMed/                 # med provider 与 observation -> med 训练支持
├── Models/                 # encoder、decoder、policy、med support
├── Training/               # 训练入口、构建逻辑、engine、runtime
├── TestScripts/            # 评估、可视化、数据检查工具
├── Scripts/                # 多步骤训练脚本、批量实验、数据处理入口
├── PolicyDeploy/           # 真机部署 adapter
├── ExperimentSuites/       # JSON 批量实验定义
├── Outputs/                # 训练输出、checkpoint、指标
└── utils/                  # 路径、运行时库、通用工具
```

边界约定：

- `Data/` 不理解数据，只作为软链接/挂载点。
- `Datasets/` 负责理解原始数据格式并产出统一样本。
- `GenMed/` 负责从样本或未来窗口中生成/提供中间表示。
- `Models/` 不直接读原始 parquet/video，只消费 adapter 整理后的 batch。
- `Training/` 不写数据格式特判，只调 registry/build/runtime。
- `Scripts/` 放多步骤入口；单次训练优先走 `Training/train.py` 或明确 wrapper。

## 3. 配置系统

Hydra 配置按层组合：

```text
Configs/
├── realrobot_vanilla_vla.yaml
├── realrobot_vanilla_tactile.yaml
├── realrobot_paper_fold_vanilla_vla.yaml
├── data/
├── model/
├── train/
└── experiment/
```

顶层 config 决定完整实验；配置组只描述一个维度：

- `data=...`：数据根目录、manifest loader、字段、窗口、增强、loader。
- `model=...`：模型族和网络宽度。
- `train=...`：训练 epoch、优化器、日志、checkpoint。
- `experiment=...`：输入 adapter 和 loss 语义。

推荐新增实验时优先新增顶层 config，少在命令行堆过多 override。单次调试可以用：

```bash
python Training/train.py --config-name=<name> --cfg job
```

检查最终展开配置。

## 4. 路径解析

路径逻辑在 [`utils/path_config.py`](./utils/path_config.py)：

- 启动时加载仓库根目录 `local_paths.env`。
- 相对路径按 repo root 解析。
- `local_root_dirs` 优先于 `root_dirs`。
- 未配置本地路径时回退到 `Data/...` 软链接。

Shell 侧入口可复用 [`Scripts/load_local_env.sh`](./Scripts/load_local_env.sh)，它会加载 `local_paths.env` 并激活 `CONDA_ENV`。

## 5. 数据抽象

核心类型在 [`Datasets/base.py`](./Datasets/base.py)：

- `DatasetManifest`：数据集静态信息，包括维度、feature schema、episode 长度、split。
- `CanonicalSample`：单条样本统一格式，包括 `observations`、`intermediates`、`targets`、`availability`、`metadata`。
- `collate_canonical_samples`：把样本合成 batch。

模型前的输入适配在 [`Datasets/input_adapters.py`](./Datasets/input_adapters.py)：

- `vanilla`：使用 observations 预测 action。
- `gtmed`：使用 observations + ground-truth med 预测 action。
- `fulltrain`：预测 action，同时训练 med 分支。
- `genmed`：只预测 med。

这层是数据格式和实验范式的边界。新增数据集时只要产出 canonical sample，训练器不需要改。

## 6. Manifest Loaders

manifest loader 注册在 [`Datasets/manifest_loaders.py`](./Datasets/manifest_loaders.py)。

常用 loader：

- `LeRobot_MetaManifest`：读取 `meta/info.json`、`episodes.jsonl`、`tasks.jsonl`，适合 chips 和 paper_fold。
- DeskTaskTac 系列 loader：读取 `meta/dataset_info.json`、`features.json`、`episodes.parquet`。
- HumanMix loader：用于跨数据源混合。

split 规则：

- 如果 metadata 中有 `splits`，优先使用 metadata。
- 如果没有 `train`，默认全部 episode 进入 train。
- `derived_val_ratio=0.0` 时不会从 train 派生真实 val，但 loader 会保留一个最小 val fallback，避免请求 val split 时没有对象。
- 训练是否使用 validation 由 `train.use_validation` 控制。

多数据集：

- `root_dirs` 可配置多个根目录。
- `datasets[]` 可给每个数据集单独配置 root、local root、filter。
- 当前真实机器人 paper_fold 使用 `discovery_glob: "*/meta/info.json"` 从一个父目录发现三个子数据集。
- 地址混合策略由 `data.raw.sampling.mix_strategy` 控制，常用 `concat` 或 `round_robin`。

## 7. 真实机器人数据

### Chips

配置：

- [`Configs/data/realrobot_chips_med.yaml`](./Configs/data/realrobot_chips_med.yaml)
- [`Configs/realrobot_vanilla_vla.yaml`](./Configs/realrobot_vanilla_vla.yaml)
- [`Configs/realrobot_vanilla_tactile.yaml`](./Configs/realrobot_vanilla_tactile.yaml)
- [`Configs/realrobot_vanilla_tactile_norm.yaml`](./Configs/realrobot_vanilla_tactile_norm.yaml)

特点：

- LeRobot-style metadata。
- 图像字段：`observation.images.front`。
- 触觉字段：`observation.hand.right.tactile`。
- 触觉可转成 union UV 维度 `351`。
- `action_chunk` 默认 `16`。

### Paper Fold

配置：

- [`Configs/data/realrobot_paper_fold.yaml`](./Configs/data/realrobot_paper_fold.yaml)
- [`Configs/realrobot_paper_fold_vanilla_vla.yaml`](./Configs/realrobot_paper_fold_vanilla_vla.yaml)
- [`Configs/realrobot_paper_fold_vanilla_tactile.yaml`](./Configs/realrobot_paper_fold_vanilla_tactile.yaml)
- [`Configs/realrobot_paper_fold_vanilla_tactile_norm.yaml`](./Configs/realrobot_paper_fold_vanilla_tactile_norm.yaml)

特点：

- 父目录下三个子数据集共同训练。
- `state_dim=38`，`action_dim=26`。
- 图像字段同样是 `observation.images.front`。
- 右手触觉字段同样是 `observation.hand.right.tactile`。
- 训练框架可复用 chips 路径，但部署动作映射需要单独确认。

## 8. 图像与触觉预处理

视频读取和轻量 vision token 在 [`Datasets/video_utils.py`](./Datasets/video_utils.py)。

当前真实机器人 VLA 使用：

```text
RGB mean per frame + temporal position -> vision_tokens, shape [T, 4]
```

图像增强在 dataset 读取视频后、生成 vision token 前执行，只作用于 train split。增强包括：

- 对称 brightness jitter。
- contrast / saturation jitter。
- 轻微 random resized crop。
- gaussian noise。
- 轻微 gaussian blur。
- random erasing，并保护中心区域。

触觉处理在 [`Datasets/realrobot_lerobot_dataset.py`](./Datasets/realrobot_lerobot_dataset.py)：

- raw tactile 可保持原始维度，也可转成 UV `351`。
- min/max normalize 的开关和统计会写入 checkpoint data summary。
- 部署端读取 checkpoint summary，保持训练/部署 normalize 一致。

## 9. Med 与 Point-Cloud

med provider 在 [`GenMed`](./GenMed)：

- future tactile vector。
- future tactile point-cloud。
- mock med。

数据端关键配置：

```text
data.raw.sequence.med.*
data.med.*
```

模型端关键配置：

```text
model.med_branch.projector.type=auto|mlp|pointnet
model.med_branch.predictor.type=auto|mlp|point_query
```

规则：

- vector med 走 MLP projector/predictor。
- point-cloud med 可以用 PointNet-style projector 和 point-query predictor。
- `GTMed` 主要使用 projector。
- `GenMed` 主要使用 predictor。
- `FullTrain` 同时使用 projector/predictor 和 action loss。

点云细节：

- 原始点云通常是 `[T, 137, C]`。
- 默认每点特征可包含 `(x, y, z, tactile_value)`。
- point count 和 per-point feature dim 必须在混合数据集中一致。

更局部的点云实现说明见 [`Models/PointCloud/README.md`](./Models/PointCloud/README.md)。

## 10. 模型结构

模型构建入口在 [`Training/build_utils.py`](./Training/build_utils.py) 和各 experiment build 文件。

模型族：

- `minimal_mlp`：低维 baseline。
- `basic_vla`：token policy，消费 state、vision token、可选 task token、可选 tactile token。
- `basic_diffusion`：同一 observation 接口上的 diffusion action policy。

共享组件：

- 状态 encoder。
- vision token encoder。
- med projector。
- med predictor。
- action decoder。
- VLA core。
- diffusion core。

设计原则：

- 模型只关心 batch 张量和 summary，不读取原始数据。
- action horizon 从 `data.raw.sequence.action_chunk` 推断。
- med shape 从 data summary / med spec 推断。
- 不同 experiment wrapper 只表达 loss 和输入差异。

## 11. 训练运行时

统一入口：

```text
Training/train.py
```

核心模块：

- [`Training/runtime.py`](./Training/runtime.py)：Hydra 入口、输出目录、checkpoint 读写。
- [`Training/engine.py`](./Training/engine.py)：epoch loop、progress、metrics。
- [`Training/registry.py`](./Training/registry.py)：experiment trainer 注册。
- [`Training/VanillaTrain`](./Training/VanillaTrain)：observation -> action。
- [`Training/GTMed`](./Training/GTMed)：observation + med -> action。
- [`Training/FullTrain`](./Training/FullTrain)：observation -> med + action。
- [`Training/GenMed`](./Training/GenMed)：observation -> med。

输出约定：

- `last.pt`：最近 checkpoint。
- `best.pt`：有 validation 时的 best checkpoint。
- `checkpoints/epoch_XXXX.pt`：周期 checkpoint。
- `epoch_metrics.jsonl`：本地逐 epoch 指标。
- wandb 默认推荐 offline。

## 12. 评估与可视化

评估入口：

- [`TestScripts/evaluate_checkpoint.py`](./TestScripts/evaluate_checkpoint.py)
- [`TestScripts/evaluate_gtmed_pipeline.py`](./TestScripts/evaluate_gtmed_pipeline.py)

数据检查：

- [`TestScripts/visualize_video_clip_loading.py`](./TestScripts/visualize_video_clip_loading.py)
- `TestScripts/tools/` 下的 LeRobot 检查工具。

批量实验：

- [`Scripts/run_experiment_suite.py`](./Scripts/run_experiment_suite.py)
- [`ExperimentSuites`](./ExperimentSuites)

## 13. 部署结构

部署 adapter 位于 [`PolicyDeploy/Model_FR3_deploy`](./PolicyDeploy/Model_FR3_deploy)。

职责边界：

- 训练 checkpoint 提供模型权重、resolved config、data summary。
- deploy policy 负责加载 checkpoint、恢复模型、构造 observation、输出 action chunk。
- 外部真机 deploy loop 负责相机、FR3、Inspire hand、动作队列和发送。

部署关键点：

- dry-run 默认不发送动作。
- `ACTION_CHUNK_SIZE` 可控制用完整 action chunk 还是前 N 步。
- tactile normalize 从 checkpoint summary 读取，不在部署脚本里手写。
- chips 和 paper_fold 可能有不同 action/state 维度，部署映射必须与数据集一致。

## 14. 扩展方式

新增数据格式：

1. 在 `Datasets/manifest_loaders.py` 中实现 manifest discovery。
2. 在 `Datasets/` 中实现 Dataset，产出 `CanonicalSample`。
3. 在 `Datasets/registry.py` 注册 `dataset_type`。
4. 新增 `Configs/data/<name>.yaml`。
5. 用 `--cfg job` 和 batch smoke 验证。

新增 med：

1. 明确 med 原始字段和内部 shape。
2. 在 dataset 中写入 `sample.intermediates`。
3. 在 `GenMed/` 中实现 provider。
4. 更新 med spec 推断。
5. 为 GTMed / FullTrain / GenMed 各跑一个小 smoke。

新增模型：

1. 在 `Models/` 中实现模块。
2. 在对应 experiment build 文件中接入 family dispatch。
3. 只依赖 data summary，不读取原始 config 字段的隐式路径。
4. 添加最小 config 到 `Configs/model/`。

新增实验范式：

1. 新增 `Configs/experiment/<name>.yaml`。
2. 新增 input adapter 或复用现有 adapter。
3. 在 `Training/registry.py` 注册 trainer。
4. 保持 dataset/model 接口不变。

## 15. 开发注意事项

- 不要在训练器里硬编码数据字段名。
- 不要在模型里读取 parquet/video。
- 不要让同一语义出现在两份配置中，例如 action horizon 只用 `data.raw.sequence.action_chunk`。
- 新增真实数据前先做 manifest discovery smoke，再做 batch smoke，最后跑训练。
- 多数据集混合前先确认 state/action/med/vision shape 一致。
- 真机部署前先做 offline smoke 和 dry-run。
