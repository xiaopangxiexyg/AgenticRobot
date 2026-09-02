# DexTeleop

## 抓姿 Skill 必读：源码、权重与一键环境

`perception.plan_point_grasp` 的现行实现是同一 `dexteleop310` Python 进程内的直接调用：
MuJoCo/相机数据、SAM2、GraspNet、IK 与运动规划不通过模型 server-client 通信。不要为它新建
`graspnet` 或其他第二 Conda 环境。

### 1. 第三方源码在哪里？

所有源码都是父仓固定 commit 的 Git submodule，位于当前 AgenticRobot worktree：

| 组件 | 源码路径 |
| --- | --- |
| SAM2 | `model/perception/sam2` |
| Grounding DINO | `model/perception/GroundingDINO` |
| VGN | `model/perception/vgn` |
| GraspNet baseline | `model/perception/graspnet-baseline` |
| PointNet2 CUDA 扩展 | `model/perception/graspnet-baseline/pointnet2` |

因此绝对路径是 `${DEXTELEOP_ROOT}/model/perception/...`；例如当前 worktree 的 SAM2 是
`/home/ai/Desktop/projects/AgenticVLA/AgenticRobot/model/perception/sam2`。PointNet2 是 GraspNet
内部的 CUDA 扩展，没有独立 checkpoint。

### 2. 模型权重在哪里？

权重不进入 Git，也不会由安装脚本下载；它们属于共享运行资产根
`${DEXTELEOP_ASSET_ROOT}`（通过 `scripts/dev_asset_store.py configure` 绑定）。应存在以下文件：

| 模型 | 权重相对 `${DEXTELEOP_ASSET_ROOT}` 的路径 |
| --- | --- |
| SAM2.1 Hiera Large | `perception/sam2_grounding_dino/sam2.1_hiera_large.pt` |
| Grounding DINO Swin-T OGC | `perception/sam2_grounding_dino/groundingdino_swint_ogc.pth` |
| VGN | `perception/sam2_vgn_point_grasp/vgn_conv.pth` |
| GraspNet RealSense | `perception/graspnet-point-grasp/checkpoint-rs.tar` |

每个 bundle 还必须有同目录的 `manifest.json`，其中记录真实 SHA-256。当前 worktree 使用默认
assets 映射时，上述路径从 `assets/perception/` 开始。权重官方下载和 manifest 制备见第 2、3 节及
[`POINT_GRASP.md`](./docs/agent_gateway/POINT_GRASP.md)。

### 3. 新机器如何一键安装？

先取得包含权重的 assets，再在 AgenticRobot 根目录运行一条命令。RTX 4090 使用 `8.9`；其他 GPU
把它换成 `python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.major}.{p.minor}")'`
的输出。

```bash
# 纯仿真/感知开发，没有 DM-Tac SDK 时：
bash scripts/install_point_grasp_env.sh \
  --skip-vendor-sdk \
  --cuda-arch-list 8.9

# 真机开发且本机已有 assets/DM_Tac_SDK_Publish_1.2.10 时，去掉 --skip-vendor-sdk。
```

该 wrapper 创建 `dexteleop310`、初始化四个 submodule，并安装/编译 SAM2、Grounding DINO、VGN、
GraspNet 和 PointNet2；它不会下载模型权重。

### 4. 已有基础 DexTeleop 如何一键更新？

先更新到包含该功能的 AgenticRobot revision，然后运行：

```bash
git pull --ff-only
bash scripts/install_point_grasp_env.sh \
  --update \
  --skip-vendor-sdk \
  --cuda-arch-list 8.9
```

已有 DM-Tac SDK 的真机开发者同样去掉 `--skip-vendor-sdk`。wrapper 会自动执行
`git submodule update --init --recursive`；更新完成后可用以下只读检查确认源码 commit、CUDA 算子和
包导入：

```bash
bash scripts/install_point_grasp_models.sh \
  --backend cuda --with-graspnet --verify-only
```

## 1. 首先下载运行资产（必需）

开始安装、测试或运行前，请先下载永久有效的
[`assets.zip`](https://disk.pku.edu.cn/link/AACC5274EC91084D4FBF25786B5A9FCE36)，并解压到本机稳定目录。
解压后，将包含所需 bundle 的 `assets/` 目录配置为本 clone 的共享资产根：

```bash
python scripts/dev_asset_store.py configure --asset-root /absolute/path/to/assets
```

不要为每个 linked worktree 重复复制基础约 3 GB 资产和可选模型权重。配置一次后，`claim_worktree.py`
会自动建立兼容映射并验证必需的 RM65 bundle；详细规则见
[`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 2. 安装 SAM2 + Grounding DINO 物体定位 Skill（可选）

`perception.localize_objects` 支持两种单帧图片输入：不传 language prompt 时由 SAM2 自动提出物体
bbox；传入 `apple` 等 prompt 时由 Grounding DINO 检测文本目标，再由 SAM2 精化 bbox。第三方源码
以固定 gitlink 的 submodule 位于 `model/perception/`；父仓只记录官方 URL 和 commit，不 vendor 其
文件历史。一键脚本会初始化 submodule、安装依赖、编译 Grounding DINO CUDA 扩展并执行 GPU 算子
smoke test：

```bash
git submodule update --init --recursive -- \
  model/perception/sam2 model/perception/GroundingDINO
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/install_perception_models.sh --backend cuda
```

源码目录固定为当前 worktree 的 `model/perception/`。重复执行是幂等的；如果 submodule checkout
存在 tracked/staged 修改，脚本会停止而不会覆盖。完整参数见：

```bash
bash scripts/install_perception_models.sh --help
```

### 需要下载的两个权重

基础 `assets.zip` 当前不包含下面两个可选模型权重，需要从模型发布方下载后放进同一个共享 assets 根：

| 模型 | 必须使用的文件名 | 官方发布页 | 直接下载地址 |
|---|---|---|---|
| SAM2.1 Hiera Large | `sam2.1_hiera_large.pt` | [Meta SAM2 checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints) | [Meta 权重文件](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| Grounding DINO Swin-T OGC | `groundingdino_swint_ogc.pth` | [Grounding DINO v0.1.0-alpha release](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha) | [Grounding DINO 权重文件](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |

```bash
export DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets
export PERCEPTION_BUNDLE="${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino"
mkdir -p "${PERCEPTION_BUNDLE}"

curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o "${PERCEPTION_BUNDLE}/sam2.1_hiera_large.pt"
curl -fL \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "${PERCEPTION_BUNDLE}/groundingdino_swint_ogc.pth"

cp docs/agent_gateway/examples/sam2_grounding_dino_asset_manifest.example.json \
  "${PERCEPTION_BUNDLE}/manifest.json"
sha256sum \
  "${PERCEPTION_BUNDLE}/sam2.1_hiera_large.pt" \
  "${PERCEPTION_BUNDLE}/groundingdino_swint_ogc.pth"
```

把 `sha256sum` 输出的两个真实值分别填入 `manifest.json` 对应的 `sha256` 字段。示例 manifest 已固定
经过审查的 SAM2 和 Grounding DINO commit；权重哈希仍故意是全零占位符，不替换就会拒绝启动。

### 用一张本地图片运行真实端到端推理

下面的验证不启动相机、不连接机械臂，也不需要 Agent Gateway。输入只接受 JPEG/PNG，输出 JSON 中的
bbox 格式固定为 `xyxy_pixel_0based_half_open`，同时生成画框 JPEG：

```bash
mkdir -p /tmp/dexteleop-perception-output

# 1. SAM2 自动发现图片中的物体
python -m deployments.perception.localize_image_cli \
  --asset-bundle "${PERCEPTION_BUNDLE}" \
  --input-image /absolute/path/to/test.jpg \
  --output-json /tmp/dexteleop-perception-output/automatic.json \
  --output-image /tmp/dexteleop-perception-output/automatic.jpg

# 2. Grounding DINO + SAM2 定位文本指定的物体
python -m deployments.perception.localize_image_cli \
  --asset-bundle "${PERCEPTION_BUNDLE}" \
  --input-image /absolute/path/to/test.jpg \
  --language-prompt apple \
  --output-json /tmp/dexteleop-perception-output/apple.json \
  --output-image /tmp/dexteleop-perception-output/apple.jpg
```

输出已存在时命令默认拒绝覆盖；确认需要替换时显式加 `--overwrite`。完整 Gateway Skill 的启动、
Artifact 上传和测试用本机路径边界见
[`SAM2_GROUNDING_DINO.md`](./docs/agent_gateway/SAM2_GROUNDING_DINO.md)。

## 3. 安装 SAM2 + VGN/GraspNet 点击抓姿 Skill（可选）

`perception.plan_point_grasp` 接受对齐 RGB-D、图片点击点、固定相机内参和
`T_base_from_camera`，返回最多 5 个排序后的 DMTAC M `tool_tcp` 候选，同时给出
`T_camera_from_tcp` 与 `T_base_from_tcp`。它是纯只读 Skill，只产生建议，不执行机械臂或夹爪命令。

默认后端是 VGN。SAM2、Grounding DINO、VGN 和 GraspNet 源码都是本仓 `model/perception/` 下的
submodule；权重仍是独立的 `assets/` 运行资产：

```bash
git submodule update --init --recursive -- \
  model/perception/sam2 model/perception/GroundingDINO \
  model/perception/vgn model/perception/graspnet-baseline
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/install_point_grasp_models.sh --backend cuda
```

脚本固定 [VGN](https://github.com/ethz-asl/vgn.git) commit
`d7af0622433f52ae88ebe81533f12b46b33e951a`，并复用上一节固定的 SAM2 checkout。所需新增权重只有
`vgn_conv.pth`：从 VGN 官方
[Google Drive 模型包](https://drive.google.com/file/d/1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ/view?usp=sharing)
下载 `data.zip`，取出 `data/models/vgn_conv.pth`，放到：

```text
assets/perception/sam2_vgn_point_grasp/vgn_conv.pth
```

复制示例 manifest，并将全零 SHA-256 替换为文件的真实哈希：

```bash
export POINT_GRASP_BUNDLE="${DEXTELEOP_ASSET_ROOT}/perception/sam2_vgn_point_grasp"
mkdir -p "${POINT_GRASP_BUNDLE}"
cp /path/extracted/data/models/vgn_conv.pth "${POINT_GRASP_BUNDLE}/vgn_conv.pth"
cp docs/agent_gateway/examples/sam2_vgn_point_grasp_asset_manifest.example.json \
  "${POINT_GRASP_BUNDLE}/manifest.json"
sha256sum "${POINT_GRASP_BUNDLE}/vgn_conv.pth"
```

也可选择 [GraspNet baseline](https://github.com/graspnet/graspnet-baseline.git) 后端。项目已经取得
源码与模型资产使用/再分发许可，因此无需额外 license flag；官方 `checkpoint-rs.tar` 按普通运行
资产放入共享 `assets/perception/graspnet-point-grasp/` bundle：

```bash
bash scripts/install_point_grasp_models.sh \
  --backend cuda \
  --with-graspnet \
  --cuda-arch-list 8.9
```

官方 checkpoint 下载页、manifest 准备方式和 bundle 目录见
[`POINT_GRASP.md`](./docs/agent_gateway/POINT_GRASP.md#graspnet-后端资产)。源码只以 submodule gitlink
进入父仓；checkpoint 不进入 Git，但可以随团队统一 `assets.zip`/网盘资产发布。

用一张本地 RGB-D 做真实 GPU 端到端推理：

```bash
python -m deployments.perception.point_grasp_image_cli \
  --sam2-asset-bundle "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --point-grasp-asset-bundle "${POINT_GRASP_BUNDLE}" \
  --input-rgb /absolute/path/to/frame.png \
  --input-depth /absolute/path/to/aligned_depth_m_float32.npy \
  --calibration-json /absolute/path/to/fixed_camera_calibration.json \
  --target-u 640 --target-v 360 --max-candidates 5 \
  --output-json /tmp/point-grasp.json \
  --output-image /tmp/point-grasp.jpg
```

使用 GraspNet 时，把默认命令中的 point-grasp bundle 参数替换为：

```bash
  --point-grasp-backend graspnet \
  --point-grasp-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/graspnet-point-grasp"
```

深度必须为对齐 RGB 的 C-order `float32` H×W NPY，单位米。标定文件格式、Gateway 完整启动命令、
官方权重包提取方式、当前 simulation 使用的 DMTAC M URDF，以及单视角碰撞检查局限见
[`POINT_GRASP.md`](./docs/agent_gateway/POINT_GRASP.md)。

DexTeleop 汇集共享 robot motion、真实机械臂遥操与数采、policy deploy、replay、simulation
和 dataset contracts。运行命令按领域统一维护在 [`docs/`](./docs/README.md)，不要从历史
实验报告或日志中拼接参数。

## 从这里开始

| 目标 | 当前入口 |
|---|---|
| 多人开发、worktree 与提交门禁 | [`CONTRIBUTING.md`](./CONTRIBUTING.md) |
| 安装环境、入口检查 | [`docs/setup/`](./docs/setup/README.md) |
| 真机遥操、正式数采、replay、deploy | [`docs/real/`](./docs/real/README.md) |
| OpenPI 固定源码与 server/client 接入 | [`deployments/openpi/`](./deployments/openpi/README.md) |
| MuJoCo、VIVE sim teleop、dataset replay | [`docs/sim/`](./docs/sim/README.md) |
| 外部 Agent Skill、Gateway 与 SSH 只读接入 | [`docs/agent_gateway/`](./docs/agent_gateway/README.md) |
| 设备枚举、USB、标定 | [`docs/hardware/`](./docs/hardware/README.md) |
| 数据后处理、分析、测试、验收 | [`docs/data/`](./docs/data/README.md) |
| 单源直通、多源组合与模块所有权 | [`docs/architecture/runtime_domain_boundaries.md`](./docs/architecture/runtime_domain_boundaries.md) |
| 被替代命令与历史报告 | [`docs/legacy/`](./docs/legacy/README.md) |

环境依赖和锁文件的完整说明仍以 [`README_ENV.md`](./README_ENV.md) 为准。
所有长期命令都通过环境名激活 Conda，不依赖 `miniconda3` / `anaconda3` 的安装路径；统一
前缀和常用命令速查见 [`docs/README.md`](./docs/README.md#命令约定)。

第一次使用建议按顺序完成：安装环境 → 运行 R0 入口检查 → 阅读目标领域的简洁入口 →
做仿真或只读设备预检 → 最后才进入 H2/M 操作。R0/R1/H1/H2/M 的含义见
[`docs/README.md`](./docs/README.md#运行等级)。

## Canonical entrypoints

- GUI human teleop / formal collection：
  `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui`
- Headless/compatibility collection：`python -m teleop_runtime.collection`
- Policy deployment：`python -m teleop_runtime.deploy`
- External Agent Gateway（MuJoCo/real/SSH read-only）：[`docs/agent_gateway/`](./docs/agent_gateway/README.md)

这里只声明入口身份；可直接运行的完整参数、前置条件和风险等级见对应 `docs/` 页面。

## 执行模式

deploy、human teleop 和 replay 默认都是各自 owner 持有的**单源直通模式**；只有两个或更多
source 同时参与选择、融合或 handoff 时，应用/profile 才显式创建 `control_composition`。

| 模式 | 当前入口或装配 | 最终执行 owner | `control_composition` |
|---|---|---|---|
| Human teleop | canonical GUI real-teleop 入口 | real-teleop session、Rig 与真实 backend | 默认不加载 |
| Policy deploy | `python -m teleop_runtime.deploy` | deploy safety/executor 与真实 backend | 默认不加载 |
| Replay | 支持入口 `python -m teleop_runtime.replay` | replay executor、`ReplayHardwareRig` 与真实 backend | 默认不加载 |
| 多源控制 | 显式 Control Station composition/Sim-HIL profile | 原 target 的唯一 safety/dispatch owner | profile 显式加载 |

被动 observer、ControlTrace、difference 或 shadow 不会因为记录了多类事实就成为多源控制，也不得
取得 authority 或 dispatch token。
详细依赖与验证边界见[运行域与可拆分边界](./docs/architecture/runtime_domain_boundaries.md)。

## 目录边界

- `control_composition/`：与本体、sim、遥操设备和模型实现解耦的多源控制组合、session/handoff 与差异观测；只产生 proposal candidate 和 lineage，不拥有 authority、安全或执行。
- `robot_motion/`：共享动作语义、Rig、Safety、Planner、position control 和 runtime contracts。
- `hardware/`：厂商 SDK 与设备 backend。
- `sim/`：仿真 backend、task 和 adapter；不得依赖 `teleop_runtime` 或真实硬件。
- `collection_runtime/`：与设备、控制和 GUI 解耦的实时采集生命周期与持久化机制。
- `teleop_runtime/`：真实遥操、GUI/headless orchestration、来源绑定的采集适配、replay 和 policy deploy frontend。
- `dataset_contracts/`、`dataset_interface/`：纯数据合约与数据适配。
- `experiments/`：非正式、可复现的算法假设、baseline、训练命令与离线评估；生产包不得反向依赖。
- `model/`：本地策略权重、checkpoint、运行 sidecar，以及固定外部源码版本的
  `model/openpi` Git submodule；不是 AgenticRobot Python 源码包。
- `deployments/`：模型 client、协议和 artifact 集成。
- `control_station/`：显式多源组合和 GUI/application profile 的应用叶子；不替代各单源入口或 lower owner。
- `agent_gateway/`：上层框架中立的 Skill 协议、接入、operation runtime，以及具体 Gateway profile；不拥有底层 authority、安全、IK 或 dispatch。
- `legacy/`、`docs/legacy/`：已确认归档的代码或说明，不是生产入口。

更严格的所有权、验证和安全规则见 [`AGENTS.md`](./AGENTS.md)。

## 安全

真实机械臂、手、夹爪、servo、replay、homing 和 force-control 都可能改变设备状态。运行前先
确认设备 side/IP/串口、tool/TCP、工作空间和急停；只读检查、仿真和 dry-run 不能替代现场
放行。文档中的 H2/M 命令需要相应硬件范围的明确授权。
