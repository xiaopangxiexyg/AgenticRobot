# DexTeleop

## 1. 运行资产（基础与可选扩展）

运行资产是 clone 共享的本地状态，不提交 Git。所有下载、解压和共享根配置在本节维护；Skill
页面只说明各 bundle 的用途、manifest 和运行命令。不要为每个 linked worktree 重复复制基础约
3 GB 资产或可选模型权重。

### 基础运行资产（必需）

开始安装、测试或运行前，请先下载永久有效的
[`基础 assets.zip`](https://disk.pku.edu.cn/link/AACC5274EC91084D4FBF25786B5A9FCE36)，并按压缩包说明解压到本机稳定目录。
解压后应得到一个包含 RM65 等基础 bundle 的 `assets/` 目录。

### 感知与抓姿扩展资产（可选）

运行 `SAM2 + Grounding DINO` 物体定位 Skill 或 VGN/GraspNet 点击抓姿 Skill 时，再下载
[`感知/抓姿 assets.zip`](https://disk.pku.edu.cn/link/AA6DAAF3490F7645A18F32230527E5480B)，并按压缩包说明把可选
`perception/` bundle 放入同一个共享资产根。各 bundle 的精确文件、manifest 和上游权重来源见
[`SAM2_GROUNDING_DINO.md`](./docs/agent_gateway/SAM2_GROUNDING_DINO.md) 和
[`POINT_GRASP.md`](./docs/agent_gateway/POINT_GRASP.md)。

从仓库根目录把包含所需 bundle 的 `assets/` 目录配置为本 clone 的共享资产根，然后检查清单：

```bash
export DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets
python scripts/dev_asset_store.py configure --asset-root "$DEXTELEOP_ASSET_ROOT"
python scripts/dev_asset_store.py status
python scripts/dev_local_resources.py audit
```

常用 bundle 的相对路径如下：

| 用途 | `${DEXTELEOP_ASSET_ROOT}` 下的路径 |
|---|---|
| RM65 基础模型 | `rm65_inspire_tac/`、`rm65_inspire_gripper_v3/` |
| SAM2 + Grounding DINO | `perception/sam2_grounding_dino/` |
| 默认 VGN 点击抓姿 | `perception/sam2_vgn_point_grasp/` |
| 可选 GraspNet 点击抓姿 | `perception/graspnet-point-grasp/` |

需要重新发布经审核的完整资产包时，在包含 `assets/` 的父目录打包整个目录并记录 SHA-256；不要只
打包单个 checkpoint，也不要把 `model/perception/` 下的第三方 Git checkout 放进资产包：

```bash
cd /absolute/path/containing-assets
zip -r assets.zip assets
sha256sum assets.zip
```

配置一次后，`claim_worktree.py` 会为后续 linked worktree 建立兼容映射并验证必需的 RM65 bundle；
更完整的共享资产规则见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 2. SAM2 + Grounding DINO 物体定位 Skill（可选）

下面的验证不启动相机、不连接机械臂，也不需要 Agent Gateway。输入只接受 JPEG/PNG，输出 JSON 中的
bbox 格式固定为 `xyxy_pixel_0based_half_open`，同时生成画框 JPEG：

```bash
export PERCEPTION_BUNDLE="${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino"
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

输出已存在时命令默认拒绝覆盖；确认需要替换时显式加 `--overwrite`。可选 object-localization
Gateway 的启动与 Artifact 上传边界见
[`SAM2_GROUNDING_DINO.md`](./docs/agent_gateway/SAM2_GROUNDING_DINO.md)。

## 3. SAM2 + VGN/GraspNet 点击抓姿 Skill（可选）

用一张本地 RGB-D 做真实 GPU 端到端推理：

```bash
export POINT_GRASP_BUNDLE="${DEXTELEOP_ASSET_ROOT}/perception/sam2_vgn_point_grasp"
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

深度必须为对齐 RGB 的 C-order `float32` H×W NPY，单位米。该入口目前仅为离线/仿真能力，
不注册 real Gateway Route；标定文件格式、权重包提取方式、当前 simulation 使用的 DMTAC M
URDF，以及单视角碰撞检查局限见
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
