# ManipRep 使用说明

ManipRep 是一个用于多模态真实机器人数据、触觉中间表示和策略学习的训练与部署框架。

根目录只保留两份 README：

- `README.md`：使用手册。覆盖数据软链接、第三方依赖、环境配置、基础样例、训练脚本参数和部署。
- [`StructureREADME.md`](./StructureREADME.md)：开发手册。覆盖目录结构、数据管线、模型/训练实现边界和扩展方式。

子目录 README 只保留局部细节，例如 [`PolicyDeploy/Model_FR3_deploy/README.md`](./PolicyDeploy/Model_FR3_deploy/README.md)、[`GenMed/README.md`](./GenMed/README.md)、[`Training/README.md`](./Training/README.md)。

## 1. 环境加装

默认工作目录：

```bash
cd /home/zc/HandMotion/ManipRep
```

本节按“已经有一个 conda 环境，需要把 ManipRep 主训练/评估代码依赖加装进去”来写。环境名不限，历史脚本默认使用 `ManipRep_310`；如果使用别的环境，在 `local_paths.env` 或运行脚本前设置 `CONDA_ENV` 即可。

先激活目标环境：

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate ManipRep_310
```

建议 Python 使用 3.10 系列；主训练代码对 Python 版本不强绑定，但 `chumpy`、MANO 可视化和部分老工具在较新的 Python/NumPy 组合上更容易出兼容问题。已有环境里不要为了本项目盲目重装 Python，先按下面步骤加装并跑 smoke。

本指南不覆盖 [`PolicyDeploy`](./PolicyDeploy) 下的真机部署依赖，也不覆盖 `PolicyDeploy/UV_conversion` 自身的依赖安装。当前部分真实机器人 tactile preset 会通过 `Datasets/tactile_uv.py` 调用该目录做 UV 转换；如果要跑这条链路，需要额外保证那部分依赖可用。

### 1.1 PyTorch 先单独检查

不要用通用依赖文件覆盖一个已经可用的 CUDA/PyTorch 组合。先检查：

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available())"
```

如果 `torch` 不存在，再按机器 CUDA/驱动选择对应安装方式。只需要 `torch`，本仓库主链路没有直接依赖 `torchvision` 或 `torchaudio`：

```bash
# 示例：CUDA wheel，按本机 CUDA 兼容版本选择 cu121/cu118/cu128 等。
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only 调试环境。
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1.2 依赖

```bash
python -m pip install -r rerquirements.txt
```

| 包 | 用途 |
| --- | --- |
| `numpy` | 张量/数组处理 |
| `pyarrow` | LeRobot/DeskTask parquet 数据读取 |
| `pandas` | 汇总脚本和表格处理 |
| `pyyaml` | YAML 配置相关工具 |
| `hydra-core`, `omegaconf` | Hydra 配置入口 |
| `tqdm` | 训练/评估进度条 |
| `ffmpeg` | 训练数据视频帧解码 |
| `wandb` | 默认训练配置的 offline/online tracking |
| `opencv`, `matplotlib`, `pillow`, `scipy` | 评估、渲染和便携可视化工具 |


### 1.3 不放入主依赖文件的部分

- `PolicyDeploy/`：真机部署、`UV_conversion` 及其本地硬件/图形依赖不在本指南覆盖范围内。
- `ThirdParty/PointNeXt/openpoints`：默认点云模型使用纯 PyTorch 的 `Models/PointCloud/pointnext_style.py`，不要求安装 OpenPoints、`pointnet2` CUDA 扩展或 `multimethod`。
- `chumpy`：只在旧 MANO/UV 相关可视化链路中需要，且对 Python/NumPy 组合敏感。不要为了普通训练加装；确实要跑 `TestScripts/portable_vis/visualize_lerobot_mano.py` 这类脚本时，再在兼容环境里单独安装。
- `pyqt`：仓库 Python 代码不直接 import。只有需要 OpenCV GUI 窗口且当前系统缺 Qt 后端时再按机器情况安装。

脚本默认会尝试激活 `CONDA_ENV`，默认值通常是 `ManipRep_310`。如果当前 shell 已经激活了目标环境，也可以设置 `MANIPREP_AUTO_CONDA=0` 并用 `PYTHON_BIN` 指向目标 Python。

## 2. 接入本地路径

机器相关路径不要写死进配置。复制模板后按本机修改：

```bash
cp local_paths.example.env local_paths.env
```

常用字段：

| 变量 | 用途 |
| --- | --- |
| `CONDA_ENV` | 默认 conda 环境，推荐 `ManipRep_310` |
| `CONDA_SH` | conda 初始化脚本路径，非默认安装位置时填写 |
| `PYTHON_BIN` | 手动指定 Python 入口 |
| `MANIPREP_REALROBOT_CHIPS_ROOT` | chips 数据集本地根目录 |
| `MANIPREP_REALROBOT_PAPER_FOLD_ROOT` | paper fold 三子集父目录 |
| `MANIPREP_DESKTASK_RAWHAND_ROOT` | DeskTaskTac RawHand 数据根目录 |
| `MANIPREP_DESKTASK_APRILTAG_ROOT` | DeskTask AprilTag 数据根目录 |
| `MANIPREP_RAWHAND_POINT_ROOT` | RawHand point-cloud 数据根目录 |
| `MANIPREP_HUMANMIX_ROOT` | HumanMix 数据根目录 |
| `MANIPREP_DEPLOY_PACKAGE_ROOT` | 真机部署包根目录 |

也可以直接使用软链接。配置中的相对路径都按仓库根目录解析，不依赖当前 shell 的工作目录：

```bash
mkdir -p Data/RawData/RealRobot
ln -sfn /abs/path/to/2026-04-27_chips_changed Data/RawData/RealRobot/chips_med
ln -sfn /abs/path/to/paper_fold_datasets Data/RawData/RealRobot/paper_fold
```

`paper_fold` 的目录应类似：

```text
Data/RawData/RealRobot/paper_fold/
├── 2026-04-28_black_paper_fold/
├── 2026-04-28_large_brown_paper_fold/
└── 2026-04-28_small_brown_paper_fold/
```

每个子集内部需要包含 `meta/info.json`、`meta/episodes.jsonl`、`data/` 和 `videos/`。

## 3. 基础检查

先检查配置是否能展开：

```bash
python Training/train.py --config-name=realrobot_vanilla_vla --cfg job
python Training/train.py --config-name=realrobot_paper_fold_vanilla_tactile --cfg job
```

最小 mock smoke。环境检查建议先用 CPU 并关闭 wandb，避免把 CUDA 或账号问题混进依赖验证：

```bash
python Training/train.py \
  experiment=vanilla \
  data=mock_lerobot_meta \
  model=minimal_mlp \
  train=smoke \
  train.device=cpu \
  train.wandb.enabled=false \
  train.run_name=vanilla_mock_smoke
```

真实数据安全 debug 建议先压低 dataloader 压力：

```bash
python Training/train.py \
  --config-name=realrobot_vanilla_tactile \
  train.run_name=realrobot_tactile_debug \
  data.raw.loader.batch_size=1 \
  data.raw.loader.num_workers=0 \
  data.raw.sampling.max_episode_cache_size=1
```

输出默认写入：

```text
Outputs/<experiment>/<run_name>/
```

包括 resolved config、逐 epoch 指标、`last.pt`、`best.pt` 和周期 checkpoint。

## 4. 训练入口

统一 Hydra 入口：

```bash
python Training/train.py --config-name=<config_name> <override...>
```

也可以使用四类实验组合：

```bash
python Training/train.py \
  experiment=<vanilla|gtmed|fulltrain|genmed> \
  data=<data_preset> \
  model=<minimal_mlp|basic_vla|basic_diffusion> \
  train=<smoke|real_debug|train_template> \
  train.run_name=<run_name>
```

常用必要参数：

| 参数 | 说明 |
| --- | --- |
| `--config-name` | 顶层配置，真实机器人 VLA 推荐直接使用封装好的 config |
| `train.run_name` / `RUN_NAME` | 输出目录名，建议每次实验显式指定 |
| `train.device` | `cuda` 或 `cpu` |
| `data.raw.loader.batch_size` / `BATCH_SIZE` | batch size |
| `data.raw.loader.num_workers` / `NUM_WORKERS` | DataLoader worker 数 |
| `data.raw.sequence.action_chunk` | 未来 action chunk 长度 |
| `train.save_periodic_checkpoints` | 是否保存周期 checkpoint |
| `train.checkpoint_every_n_epochs` | 周期 checkpoint 间隔 |
| `train.resume_from_checkpoint` / `RESUME_FROM_CHECKPOINT` | 断点恢复，脚本支持 `auto` |
| `train.wandb.enabled` / `train.wandb.mode` | 实验追踪；默认推荐 offline |

## 5. 真机 Chips VLA 训练

不带触觉：

```bash
bash Scripts/train_realrobot_chips_vanilla_vla.sh
```

带触觉：

```bash
bash Scripts/train_realrobot_chips_vanilla_tactile_vla.sh
```

触觉 min/max normalize：

```bash
CONFIG_NAME=realrobot_vanilla_tactile_norm \
RUN_NAME=realrobot_chips_med_vtactile_vla_ac16_nohis_vis_tacnorm \
bash Scripts/train_realrobot_chips_vanilla_tactile_vla.sh
```

图像增强：

```bash
IMAGE_AUGMENT=true \
RUN_NAME=realrobot_chips_med_vtactile_vla_ac16_nohis_vis_imgaug \
bash Scripts/train_realrobot_chips_vanilla_tactile_vla.sh
```

图像增强 + 触觉 normalize：

```bash
CONFIG_NAME=realrobot_vanilla_tactile_norm \
RUN_NAME=realrobot_chips_med_vtactile_vla_ac16_nohis_vis_imgaug_tacnorm \
IMAGE_AUGMENT=true \
bash Scripts/train_realrobot_chips_vanilla_tactile_vla.sh
```

图像增强默认只作用于 train split，不包含翻转或大角度旋转。亮度增强为对称采样，`IMAGE_AUGMENT_BRIGHTNESS=0.35` 表示亮度因子从 `[0.65, 1.35]` 采样。随机遮挡默认避开画面中心 `1/2 x 1/2` 区域。

## 6. 真机 Paper Fold 三子集训练

`paper_fold` 配置会从 `Data/RawData/RealRobot/paper_fold/*/meta/info.json` 发现三个子数据集并 concat 训练，不设置测试集，默认不启用 validation 训练流程。

不带触觉：

```bash
bash Scripts/train_realrobot_paper_fold_vanilla_vla.sh
```

带触觉：

```bash
bash Scripts/train_realrobot_paper_fold_vanilla_tactile_vla.sh
```

触觉 min/max normalize：

```bash
CONFIG_NAME=realrobot_paper_fold_vanilla_tactile_norm \
RUN_NAME=realrobot_paper_fold_vtactile_vla_ac16_nohis_vis_tacnorm \
bash Scripts/train_realrobot_paper_fold_vanilla_tactile_vla.sh
```

图像增强 + 触觉 normalize：

```bash
CONFIG_NAME=realrobot_paper_fold_vanilla_tactile_norm \
RUN_NAME=realrobot_paper_fold_vtactile_vla_ac16_nohis_vis_imgaug_tacnorm \
IMAGE_AUGMENT=true \
bash Scripts/train_realrobot_paper_fold_vanilla_tactile_vla.sh
```

注意：`paper_fold` 当前是 `state_dim=38`、`action_dim=26`；`chips_med` 是另一套动作/状态维度。训练可以共用框架，部署映射不能混用。

## 7. 训练脚本环境变量

`Scripts/train_realrobot_*_vla.sh` 通用环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CONDA_ENV` | `ManipRep_310` | 激活的 conda 环境 |
| `CONFIG_NAME` | 脚本默认 config | 顶层 Hydra config |
| `RUN_NAME` | 脚本默认名称 | 输出目录名 |
| `BATCH_SIZE` | `16` | 训练 batch size |
| `NUM_WORKERS` | `4` | DataLoader worker 数 |
| `WANDB_ENABLED` | `false` | 是否启用 wandb；服务器未安装 wandb 时保持 `false` |
| `WANDB_MODE` | `offline` | wandb 模式，仅在 `WANDB_ENABLED=true` 时生效 |
| `IMAGE_AUGMENT` | `false` | 是否启用图像增强 |
| `SAVE_PERIODIC_CHECKPOINTS` | `true` | 是否保存周期 checkpoint |
| `CHECKPOINT_EVERY_N_EPOCHS` | `5` | 周期 checkpoint 间隔 |
| `RESUME_FROM_CHECKPOINT` | 空 | checkpoint 路径；设为 `auto` 自动找 `last.pt` 或最新 epoch |
| `RESUME_LOAD_OPTIMIZER` | `true` | 恢复 optimizer |
| `RESUME_STRICT` | `true` | strict load checkpoint |
| `RESUME_EPOCHS_ARE_ADDITIONAL` | `false` | 恢复后 epochs 是否视为额外轮数 |

图像增强常用变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `IMAGE_AUGMENT_BRIGHTNESS` | `0.35` | 对称亮度范围 `[1-v, 1+v]` |
| `IMAGE_AUGMENT_CONTRAST` | `0.20` | 对比度 jitter |
| `IMAGE_AUGMENT_SATURATION` | `0.15` | 饱和度 jitter |
| `IMAGE_AUGMENT_CROP_PROB` | `0.45` | 轻微 random resized crop 概率 |
| `IMAGE_AUGMENT_NOISE_PROB` | `0.20` | 高斯噪声概率 |
| `IMAGE_AUGMENT_NOISE_STD` | `0.008` | 噪声 std 上限 |
| `IMAGE_AUGMENT_BLUR_PROB` | `0.05` | 轻微模糊概率 |
| `IMAGE_AUGMENT_BLUR_SIGMA` | `0.5` | 模糊 sigma 上限 |
| `IMAGE_AUGMENT_ERASE_PROB` | `0.10` | random erasing 概率 |
| `IMAGE_AUGMENT_ERASE_CENTER_PROTECT_FRACTION` | `0.50` | 中心保护区域比例 |

## 8. Med / Point-Cloud 训练

四种实验范式：

| experiment | 含义 |
| --- | --- |
| `vanilla` | observation -> action，不使用 med |
| `gtmed` | 使用数据里的 ground-truth med 作为额外输入 |
| `fulltrain` | 同时预测 med 和 action |
| `genmed` | 只训练 observation -> med |

常用 med 覆盖：

```bash
python Training/train.py \
  experiment=genmed \
  data=desktasktac_rawhand_point_real \
  model=minimal_mlp \
  train=real_debug \
  data.raw.sequence.med.kind=point_cloud \
  model.med_branch.predictor.type=point_query \
  train.run_name=genmed_point_debug
```

关键规则：

- vector med 使用 `model.med_branch.projector.type=mlp` 和 `predictor.type=mlp`。
- point-cloud med 可用 `projector.type=pointnet`、`predictor.type=point_query`。
- action chunk 统一由 `data.raw.sequence.action_chunk` 控制，不在模型配置里维护第二份 horizon。

更详细的 med 生成说明见 [`GenMed/README.md`](./GenMed/README.md)，点云模型细节见 [`Models/PointCloud/README.md`](./Models/PointCloud/README.md)。

## 9. 评估与批量实验

评估 checkpoint：

```bash
python TestScripts/evaluate_checkpoint.py \
  --checkpoint Outputs/<experiment>/<run_name>/best.pt \
  --split val
```

同时评估 train 和 val：

```bash
python TestScripts/evaluate_checkpoint.py \
  --checkpoint Outputs/<experiment>/<run_name>/best.pt \
  --split both
```

批量实验入口：

```bash
python Scripts/run_experiment_suite.py --help
```

常用套件放在 [`ExperimentSuites`](./ExperimentSuites)。执行前建议先 dry-run 或限制 `--limit`，避免一次性提交过多训练。

## 10. 真机部署

部署入口在 [`PolicyDeploy/Model_FR3_deploy`](./PolicyDeploy/Model_FR3_deploy)。部署前先配置：

```bash
MANIPREP_DEPLOY_PACKAGE_ROOT=/abs/path/to/dex_inspire_teleop_new
RIGHT_HAND_IP=<hand-ip>
FR3_HOST=<fr3-host>
```

先做离线 smoke：

```bash
python -m PolicyDeploy.Model_FR3_deploy.offline_smoke \
  --checkpoint Outputs/vanilla-tactile/<run_name>/checkpoints/epoch_XXXX.pt \
  --device cpu \
  --action-chunk-size 4
```

真机 dry-run，默认不发送动作：

```bash
CHECKPOINT=/abs/path/to/checkpoints/epoch_XXXX.pt \
ACTION_CHUNK_SIZE=4 \
MAX_FORCE=800 \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
bash PolicyDeploy/Model_FR3_deploy/run_tactile_policy.sh
```

确认观测和动作队列稳定后再显式打开：

```bash
SEND_ACTIONS=1 \
AUTO_START=0 \
CHECKPOINT=/abs/path/to/checkpoints/epoch_XXXX.pt \
ACTION_CHUNK_SIZE=4 \
MAX_FORCE=800 \
RIGHT_HAND_IP="$RIGHT_HAND_IP" \
FR3_HOST="$FR3_HOST" \
bash PolicyDeploy/Model_FR3_deploy/run_tactile_policy.sh
```

部署重要参数：

| 变量 | 说明 |
| --- | --- |
| `CHECKPOINT` | 模型 checkpoint；目录会解析到 `last.pt` |
| `ACTION_CHUNK_SIZE` | `all` 使用完整 chunk；正整数只执行前 N 个 action |
| `MIN_ACTION_QUEUE` | 队列低于该长度时重新推理 |
| `APPEND_ACTION_CHUNKS` | 是否追加新 chunk |
| `MAX_FORCE` | InspireTac 启动最大指尖力 |
| `MAX_HAND_STEP` | 手部每 tick 限速 |
| `MAX_ARM_STEP_RAD` | FR3 关节每 tick 限速 |
| `REQUIRE_TACTILE` | 触觉缺失时是否报错 |
| `REQUIRE_IMAGE` | 图像缺失时是否报错 |
| `SEND_ACTIONS` | `1` 才真正发动作 |

部署 adapter 会读取 checkpoint 内的 data summary，自动保持训练时的 tactile min/max normalize 设置。更完整的部署参数见 [`PolicyDeploy/Model_FR3_deploy/README.md`](./PolicyDeploy/Model_FR3_deploy/README.md)。

## 11. 常见问题

- `pyarrow` 缺失：真实 parquet 数据读取会失败，安装 `pyarrow`。
- `ffmpeg` 缺失：视频 token 解码会失败，安装系统或 conda `ffmpeg`。
- `wandb` 缺失：默认训练配置可能会失败；安装 `wandb`，或设置 `train.wandb.enabled=false` / `WANDB_DISABLED=true`。
- `cv2` 缺失：评估可视化和 LeRobot 工具会失败，优先通过 conda-forge 安装 `opencv`。
- 相对路径找不到：确认在 repo 根目录运行，或配置 `local_paths.env`。
- 触觉 normalize 训练/部署不一致：不要手写部署 normalize 参数，使用对应 checkpoint；部署端从 checkpoint summary 读取训练设置。
- `paper_fold` 部署动作维度不匹配：不要复用 chips 的部署映射，需要确认 26 维 action 到真机控制的解析方式。
