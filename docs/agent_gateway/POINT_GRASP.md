# SAM2 + VGN/GraspNet/GraspGenX 点击引导 DMTAC M 抓姿

离线 CLI 接受一对像素对齐的 RGB-D 单帧、点击像素、显式本地标定，返回最多
5 个已排序的 DMTAC M `tool_tcp` 抓姿。每个候选同时包含 `T_camera_from_tcp`、
`T_base_from_tcp`、建议开口宽度、模型分数、到点击区域的距离和单帧可见点云碰撞近似结果；
`selected_ordinal=0` 是当前最佳候选，`backend_id` 明确报告 `vgn`、`graspnet` 或
`graspgenx`，并输出一张标注 JPEG 文件。启用 point-grasp 时 GraspGenX 是默认后端；VGN 是
显式兼容选项，GraspNet 则仅在操作者持有相应授权并提供显式 acknowledgement 时可选。

标注 JPEG 不再只画 TCP 十字。它按 DMTAC M 的 10 mm 指厚、25.5 mm 指高、50 mm 指长和
10 mm 掌部近似包络绘制双指/掌部线框，并绘制 TCP `+X/+Y/+Z` 三轴；几何以实际 `tool_tcp`
为原点，掌/指边界位于其局部 Z 轴后方 42.999 mm。最佳候选用高亮色，其他
候选用次级颜色。投影先裁剪相机近裁剪面，再裁剪图像边界，因此部分进入画面的抓姿仍能显示，
完全位于相机后方或画面外的候选会在图例中明确标为 `behind_camera` 或 `out_of_frame`，不会静默
消失。该图仍是简单几何叠加，不做深度遮挡测试。

该入口不创建 RM65 backend、不执行 IK、轨迹规划、夹爪命令或机械臂运动。输出中的 base-frame
姿态只绑定用户提供的本地标定文件，不是 Gateway 可信 Station pose，不能绕过下游正式 motion
owner、碰撞场景和安全门禁。

## 算法与第三方源码

- [SAM2](https://github.com/facebookresearch/sam2.git) 使用正点击点分割点击所在物体/部位；
- [VGN](https://github.com/ethz-asl/vgn.git) 从点击区域的 30 cm、40³ TSDF 生成 6DoF 二指抓姿；
- 可选的 [GraspNet baseline](https://github.com/graspnet/graspnet-baseline.git) 先用 SAM2 点击 mask
  的米制三维包围盒加 12 cm margin 限定推理工作区，再生成 6DoF 二指抓姿；候选随后仍用
  SAM2 目标表面距离、点击距离和 DMTAC M 几何过滤，完整对齐深度仍用于碰撞门禁；
- 默认的 [GraspGenX](https://github.com/NVlabs/GraspGenX) 直接消费 SAM2 mask 内的米制局部点云，
  并用 DMTAC M 的 open/mid sweep-volume 条件生成跨夹爪 6DoF 候选；默认在兼容的独立 Gateway
  Python 环境内直接加载 GPU 模型，隔离的 loopback service 保留为显式兼容选项，候选仍经过
  点击归属、实际 TCP 变换和完整可见深度碰撞过滤；
第三方源码以 `model/perception/` 下的 Git submodule 进入父仓；固定 revision、
`dexteleop310` 环境安装、CUDA 架构和只读验证统一见
[`README_ENV.md`](../../README_ENV.md) 的环境与复现指南，本页不重复通用安装命令。该 Skill 只使用
VGN 的 headless 网络部分，不要求 ROS；选择 GraspNet 时仍需其 manifest 中的项目授权 acknowledgement，
旧的 `--accept-graspnet-noncommercial-license` 仅保留为兼容 no-op，新安装不需要传入。

权重不提交 Git；本地 manifest 的 SHA-256 只证明文件与该 manifest 一致，不证明 checkpoint 已经由
项目审核。使用者必须单独确认模型资产的许可与来源边界。

## 准备 VGN/GraspNet/GraspGenX 后端权重

项目级基础/感知资产包的下载、解压和 `DEXTELEOP_ASSET_ROOT` 配置统一见项目根
[`README.md`](../../README.md) 的“运行资产”章节。以下命令假定共享根已经按该章节配置；本页只
说明 VGN/GraspNet 后端各自的权重和 manifest。

该 Skill 复用下面的 SAM2 bundle：

```text
assets/perception/sam2_grounding_dino/
├── manifest.json
├── sam2.1_hiera_large.pt
└── groundingdino_swint_ogc.pth
```

SAM2 权重下载和 manifest 准备见
[`SAM2_GROUNDING_DINO.md`](./SAM2_GROUNDING_DINO.md)。点击抓姿不调用 Grounding DINO，但共享的
localizer bundle 仍按现有严格 manifest 一起验证。

本页后续命令使用的受版本控制模板可直接从文档导航访问：

- [VGN point-grasp asset manifest](./examples/sam2_vgn_point_grasp_asset_manifest.example.json)
- [GraspNet point-grasp local manifest](./examples/graspnet_point_grasp_local_manifest.example.json)
- [GraspGenX point-grasp asset manifest](./examples/graspgenx_point_grasp_asset_manifest.example.json)
- [固定相机显式标定模板](./examples/point_grasp_fixed_camera_calibration.example.json)

VGN 官方仓库 README 的模型包：

- 官方页面：[Google Drive data folder](https://drive.google.com/file/d/1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ/view?usp=sharing)
- Google Drive 文件 ID：`1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ`
- 需要从压缩包中取出的精确文件：`data/models/vgn_conv.pth`

浏览器下载最稳妥；也可安装 `gdown` 后下载。`gdown` 不是运行依赖：

```bash
python -m pip install gdown
mkdir -p /tmp/vgn-official
gdown 1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ -O /tmp/vgn-official/data.zip
unzip -l /tmp/vgn-official/data.zip | grep 'data/models/vgn_conv.pth'
unzip -j /tmp/vgn-official/data.zip 'data/models/vgn_conv.pth' \
  -d /tmp/vgn-official

export POINT_GRASP_BUNDLE="${DEXTELEOP_ASSET_ROOT}/perception/sam2_vgn_point_grasp"
mkdir -p "${POINT_GRASP_BUNDLE}"
cp /tmp/vgn-official/vgn_conv.pth "${POINT_GRASP_BUNDLE}/vgn_conv.pth"
cp docs/agent_gateway/examples/sam2_vgn_point_grasp_asset_manifest.example.json \
  "${POINT_GRASP_BUNDLE}/manifest.json"
sha256sum "${POINT_GRASP_BUNDLE}/vgn_conv.pth"
```

把 `sha256sum` 的真实值写进 `manifest.json`，不要保留全零占位值。最终目录必须是：

```text
assets/perception/sam2_vgn_point_grasp/
├── manifest.json
└── vgn_conv.pth
```

完成后按项目根 README 的“运行资产”章节检查共享根和本地资源清单；不要把
`model/perception/` Git checkout 打进资产包。

## GraspNet 后端资产

GraspNet 使用官方 RealSense checkpoint：

- 官方源码与许可证：[graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline.git)
- 官方 checkpoint 页面：[checkpoint-rs.tar on Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)
- Google Drive 文件 ID：`1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk`
- 必须保持的文件名：`checkpoint-rs.tar`

checkpoint 不进入 Git，放入项目统一 assets 根；资产包的获取和发布边界以项目根 README 的
“运行资产”章节为准：

```bash
export GRASPNET_BUNDLE="${DEXTELEOP_ASSET_ROOT}/perception/graspnet-point-grasp"
mkdir -p "${GRASPNET_BUNDLE}"

# 浏览器下载最稳妥；也可由使用者自行安装 gdown 后从官方文件 ID 下载。
python -m pip install gdown
gdown 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk \
  -O "${GRASPNET_BUNDLE}/checkpoint-rs.tar"

cp docs/agent_gateway/examples/graspnet_point_grasp_local_manifest.example.json \
  "${GRASPNET_BUNDLE}/manifest.json"
sha256sum "${GRASPNET_BUNDLE}/checkpoint-rs.tar"
```

将 `sha256sum` 的真实值写入 `manifest.json` 的 `checkpoint.sha256`；全零占位值会被拒绝。
manifest 还固定源码 commit、`realsense` 模型类型和项目授权 acknowledgement，任一字段不匹配
都会 fail closed。`gdown` 只用于下载，不是运行依赖，也不会由安装脚本安装。

## GraspGenX 更新后端

GraspGenX 源码固定在 `model/perception/graspgenx`，generator/discriminator 权重和其运行时
统一位于 `assets/perception/graspgenx-point-grasp`。默认 in-process 模式要求 Gateway 所在
Python 环境本身兼容 GraspGenX；service 模式显式使用 bundle 内独立 venv。generator 和
discriminator 必须先从已复验的本地 asset bundle 传入；安装器默认不访问 Hugging Face，
只核对固定 revision、四个已物化文件并生成真实 SHA-256 manifest，同时保持主
`dexteleop310` 不变：

```bash
bash scripts/install_graspgenx_env.sh
bash scripts/install_graspgenx_env.sh --verify-only
```

安装器只在该 bundle 的 `runtime/.venv` 中安装 GraspGenX 推理链和 service 备选，不会修改共享
`dexteleop310`。Python 包默认使用唯一的清华 PyPI 镜像，并在正式安装前对完整依赖闭包执行
`uv --dry-run`；镜像缺包或不可用时立即停止，禁止自动回退其他索引。已验证的 GPU 组合为
`torch==2.6.0`、`torchvision==0.21.0` 和 PyPI CUDA 12.4 runtime。上游完整依赖中的
PyOpenGL/pyrender 版本互斥，因此本服务不安装未使用的渲染、训练和端到端仿真依赖。只有明确
要求从官方模型仓重新拉取权重时才传 `--allow-checkpoint-download`。

service 模式的模型服务只应绑定 loopback；启动命令见 `scripts/run_graspgenx_server.sh`。服务健康
后，在 `real-perception` 或 `real-rm65` 的原启动命令中增加：

```bash
bash scripts/run_graspgenx_server.sh \
  --asset-root "${DEXTELEOP_ASSET_ROOT}" \
  --host 127.0.0.1 \
  --port 5556
```

准备好与 GraspGenX 官方依赖兼容的 Gateway Python 环境后，不需要启动上述服务；原
`real-perception` 或 `real-rm65` 命令默认选择 direct Python backend：

```text
--point-grasp-backend graspgenx
--point-grasp-asset-bundle /absolute/path/to/assets/perception/graspgenx-point-grasp
--graspgenx-source-root /absolute/path/to/model/perception/graspgenx
--point-grasp-device cuda
```

in-process 会在 Gateway 的 `start()` 阶段加载一次 generator/discriminator，按 sweep-volume
缓存 sampler，并在同一进程内串行调用模型。它不改变 SAM2 mask、DMTAC M TCP 变换、可见深度
碰撞过滤、RM65 可达性或下游运动权限。当前官方 checkpoint 要求 PyTorch <2.7；不要把它直接
安装或升级到项目的 `dexteleop310`，应先用兼容的独立 Gateway 环境完成 GPU smoke 和仿真回归。

### in-process runtime contract 与环境预检

`deployments/perception/graspgenx_runtime_contract.py` 定义了可导入、可验证的 in-process
运行时合同 `graspgenx-inprocess-runtime.v1`。任何 Gateway Python 在加载模型前必须通过该合同，
检查项为：受审 Python 版本（3.10/3.11）、`torch>=2.1,<2.7`、CUDA 可用性与设备序号、
GraspGenX source checkout 的 Git HEAD 等于 manifest 中受审 revision（默认
`model/perception/graspgenx` submodule）、受审 asset bundle 与 `gen/`、`dis/` checkpoint 布局。
`GraspGenXInProcessBackend.start()` 在导入任何上游包之前执行该验证；任一检查失败即以完整的
typed receipt 摘要 fail closed，不会退化为 service 或继续加载。receipt（schema
`dexteleop.perception.graspgenx_runtime_receipt.v1`）记录每个检查的 expected/observed/detail，
可通过 backend 的 `runtime_receipt` 属性取用。

不加载权重的环境预检命令（同样不读相机、不连机械臂）：

```bash
PYTHONPATH="${PWD}/model/perception/graspgenx:${PYTHONPATH:-}" \
  python scripts/smoke_graspgenx_inference.py \
  --mode in-process --verify-only \
  --asset-bundle "${DEXTELEOP_ASSET_ROOT}/perception/graspgenx-point-grasp"
```

退出码为 0 且输出 JSON `status=passed` 表示当前解释器满足 in-process 合同；`status=failed`
时按 `checks` 中的逐项 detail 诊断依赖冲突。该预检只证明环境兼容性，不证明模型加载、推理质量、
相机或机械臂可用；组合运行前仍须完成下文 GPU smoke 与仿真回归。

若选择 service 模式，模型加载完成后，可在第二个终端运行不读取相机、不连接机械臂的 GPU smoke。
它复用正式 `GraspGenXClient`、DMTac M sweep-volume 和协议校验，向服务发送一个确定性的 50 mm
合成方块：

```bash
"${DEXTELEOP_ASSET_ROOT}/perception/graspgenx-point-grasp/runtime/.venv/bin/python" \
  scripts/smoke_graspgenx_inference.py \
  --mode service \
  --host 127.0.0.1 \
  --port 5556
```

退出码为 0 且 JSON 中 `status` 为 `passed` 才表示 checkpoint 加载、CUDA 模型执行、
DMTac 条件推理和客户端返回校验均已完成。该 smoke 不替代真实同帧 RGB-D 抓姿质量验收。

默认的 in-process backend 不需要运行服务，可在兼容环境中直接验证：

```bash
PYTHONPATH="${PWD}/model/perception/graspgenx:${PYTHONPATH:-}" \
  "${DEXTELEOP_ASSET_ROOT}/perception/graspgenx-point-grasp/runtime/.venv/bin/python" \
  scripts/smoke_graspgenx_inference.py \
  --asset-bundle "${DEXTELEOP_ASSET_ROOT}/perception/graspgenx-point-grasp" \
  --source-root "${PWD}/model/perception/graspgenx"
```

该命令只验证 GraspGenX direct Python backend，不读取相机、不连接机械臂，也不替代 Gateway
启动和真实 RGB-D 验收。

```text
--point-grasp-backend graspgenx
--point-grasp-asset-bundle /absolute/path/to/assets/perception/graspgenx-point-grasp
--graspgenx-host 127.0.0.1
--graspgenx-port 5556
--graspgenx-timeout-ms 120000
```

离线图片 CLI 现在与 Gateway 一样默认使用 GraspGenX in-process backend；它只读取显式 RGB-D、
点击点和标定文件，不提供 service 模式，也不会启动相机或连接机械臂。CLI 必须运行在通过上述
in-process runtime contract 的兼容 Python 环境中，并可用 `--graspgenx-source-root` 指向受审源码
checkout。真实在线路径仍由同一个 Gateway provider 在一次 retained frame 读取中送入 SAM2 和
GraspGenX，不改变现有同帧深度、Station frame 或 Artifact lineage 语义。

## 本地 RGB-D 端到端推理

RGB 必须是 JPEG/PNG。深度必须是与 RGB 同宽高、C-order、`float32` 的二维 NPY，单位米，0 表示
无效深度；所有非零值必须在 `(0, 10]` 米。相机内参尺寸必须与图片一致。

复制标定示例并替换所有示例数值，尤其是固定相机实测的 `T_base_from_camera`。矩阵按 row-major
展开，语义是 `p_base = T_base_from_camera @ p_camera`；相机坐标系应为 optical frame（+x 右、
+y 下、+z 前）。示例中的单位阵仅用于展示格式，不能当作真实外参。

```bash
cp docs/agent_gateway/examples/point_grasp_fixed_camera_calibration.example.json \
  /tmp/fixed_camera_calibration.json
mkdir -p /tmp/dexteleop-point-grasp-output
export GRASPGENX_PYTHON=/absolute/path/to/graspgenx-compatible/python
export GRASPGENX_SOURCE_ROOT=/absolute/path/to/model/perception/graspgenx

PYTHONPATH="${GRASPGENX_SOURCE_ROOT}:${PYTHONPATH:-}" \
"${GRASPGENX_PYTHON}" -m deployments.perception.point_grasp_image_cli \
  --sam2-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --point-grasp-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/graspgenx-point-grasp" \
  --graspgenx-source-root "${GRASPGENX_SOURCE_ROOT}" \
  --input-rgb /absolute/path/to/frame.png \
  --input-depth /absolute/path/to/aligned_depth_m_float32.npy \
  --calibration-json /tmp/fixed_camera_calibration.json \
  --target-u 640 --target-v 360 \
  --max-candidates 5 \
  --output-json /tmp/dexteleop-point-grasp-output/result.json \
  --output-image /tmp/dexteleop-point-grasp-output/annotated.jpg
```

该命令加载真实 SAM2/GraspGenX 并使用 GPU，但不启动 RealSense、不连接机械臂。输出存在时默认拒绝
覆盖；确认需要替换时加 `--overwrite`。选择 VGN 时显式传入
`--point-grasp-backend vgn`，并将 `--point-grasp-asset-bundle` 改为 VGN bundle；VGN 不接受
`--graspgenx-source-root`。

选择 GraspNet 时只替换 backend 和 bundle 参数，其余 RGB-D、点击点与标定输入完全相同：

```bash
python -m deployments.perception.point_grasp_image_cli \
  --sam2-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --point-grasp-backend graspnet \
  --point-grasp-asset-bundle "${GRASPNET_BUNDLE}" \
  --input-rgb /absolute/path/to/frame.png \
  --input-depth /absolute/path/to/aligned_depth_m_float32.npy \
  --calibration-json /tmp/fixed_camera_calibration.json \
  --target-u 640 --target-v 360 \
  --max-candidates 5 \
  --output-json /tmp/dexteleop-point-grasp-output/graspnet-result.json \
  --output-image /tmp/dexteleop-point-grasp-output/graspnet-annotated.jpg
```

该命令加载真实 SAM2/GraspNet 并使用 GPU，同样不会启动 RealSense 或连接机械臂。

## Gateway 边界

生产 `real-perception` 默认仍不加载抓姿模型。只有同时显式启用共享 SAM2 localizer 与
`--enable-point-grasp`，并提供经过 manifest/hash 校验的 point-grasp bundle，才注册
`perception.plan_point_grasp@2`。GraspGenX 是默认 backend，并默认在通过运行时合同的 Gateway
Python 中直接加载模型；只有显式指定 `--graspgenx-mode service` 才连接预先启动的隔离 loopback service。
VGN 是显式兼容选项；GraspNet 仍要求其资产 manifest 中的项目授权 acknowledgement。启动事务先启动并 admit 相机，再启动共享
localizer，最后启动 planner；关闭顺序
相反。任一步失败都会清理已尝试资源，Skill 调用本身不会启动、重置或关闭设备/模型。

v2 请求只有三个字段：已注册的对齐 depth `role`、`target_pixel_uv` 和 `max_candidates`。provider
只调用一次 `get_latest_frame()`，因此 RGB 与 depth 来自同一个已发布 `VideoFrame`；部署侧同时绑定
source serial/generation/frame index、采集时间、内参、静态相机标定、Station revision、frame graph
及其 reference-frame identity。模型推理后以及 Artifact commit 前后都会重新核对 Station revision。
当前只有具有受信静态外参且 depth 已对齐到 color 的角色会进入 catalog；没有动态 FK 的腕部 depth
不会被调用方字符串启用。

结果保留相机坐标候选和 `canonical_station` 候选，但不含 `selected_ordinal`，并明确标记为
`ranked_unprivileged_motion_proposals_only`。它不是 motion Grant、不会自动传给
`motion.move_tcp_to_pose@1`，也不执行 IK、完整场景/机械臂/路径碰撞检查、轨迹规划或运动。若以后由
独立、获授权的 motion client 采用某一候选，仍必须重新经过其 Station/frame/tool 身份、一次性授权、
safety envelope、IK/碰撞/奇异性、dispatch、收敛和 HOLD 门禁。

启用 `perception.plan_point_grasp@2` 时，同一 camera-only profile 还会同时注册可选的
`perception.render_point_grasp_candidate@1`。Agent 不会被强制看到额外图片；需要检查某个候选时，
它只提交 `point_grasp_operation_id`、`candidate_ordinal`、`view_role`。调用方不能提交抓姿矩阵、相机
内参、外参或夹爪宽度。provider 只接受本 session 的 Station point-grasp executor 已成功完成且仍
保留的 operation，并从该不可变结果解析 ordinal；目标视角则只能选择 deployment catalog 中具有
可信外参和 color-aligned depth 的角色。未知/已淘汰 operation、该结果中不存在的 ordinal、不同
Station revision 或未注册角色都会 fail closed。

camera-only profile 每次调用重新读取指定视角的一帧 aligned RGB-D，只把一个候选投影到 RGB 上，并返回一个
`rendered_grasp_candidate` JPEG Artifact。结果中的 source operation/result/candidate hash、两次
capture group、相机/Station frame identity 和三个齐次矩阵共同闭合来源与投影视角；同时保留
`frames[0]` 的 role、尺寸、媒体类型和内容 hash，供上层按需取得 Artifact。矩阵是只读输出证据，
不是 RPent 需要解释或回传的规划输入。该渲染仍只做几何投影，`depth_occlusion_tested=false`，
不新增选择权、运动权、IK、夹爪或任何下游控制效果。

`real-rm65` 的显式 `--enable-point-grasp` 组合扩展了同一个契约，但不会伪造腕部静态外参。该组合
在启动前要求并交叉校验 camera registry、Station/双臂 base 链和一个包含三相机安装链的 URDF；
主相机使用该 URDF 中的固定链，两个腕部相机位姿只由唯一 lower control cycle 消费的新鲜双臂
measured qpos 和既有 coupled FK owner 发布。render 请求只读取已经发布的不可变 pose cut，调用方
不能注入相机位姿。默认 mount URDF 为：

```text
assets/rm65_inspire_gripper_v4/rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf
```

本流程使用 `robot_motion/configs/rm65_camera_registry_v4.json`：它登记 v4 主相机标定和上述详细
URDF placement，同时保留现场已验证的左右腕相机 serial 绑定。组合成功时 render 的可选角色严格为
`main`、`wrist_left`、`wrist_right`；腕部 pose 缺失、过期、晚于当前时刻或与图像采集时间偏差过大
都会 fail closed。启动这个组合会连接双 RM65 并建立 startup HOLD，因此不能把它当作 camera-only
验收入口；离线测试也不构成真机坐标或运动验收。

### 可选：静止多视角观测与抓姿 v4

组合真机 Gateway 在启用 GraspGenX 的前提下，可额外传入
`--enable-grasp-observation`。默认不启用，原 v2/v3 的请求和运行方式不变。
启用后新增只读 `perception.capture_grasp_observation@1`，并将当前抓姿 Route 显式注册为
`perception.plan_point_grasp@4`。流程如下：

1. 采集 Skill 接受 `selected_roles`，保留相关 RGB-D、部署内外参、采集附近的双臂 qpos
   及同一张 RGB 的预览 Artifact。缺帧视角列入 `excluded_views`；所有视角不可用时失败。
2. 从某张保留的图像选目标；`main`、`wrist_left`、`wrist_right` 均可作为主视角。
   v4 请求在原参数之外必填 `observation_operation_id` 和 `view_policy`（`single` 或 `auto`），
   只能引用本 session、同一 Station revision 下仍保留的成功采集 operation，不能注入外参。
3. `single` 只用选中视角；`auto` 先把目标表面投影到辅助视角，检查观测深度及分割对应关系，
   再融合有效的目标/环境点。遮挡、视野外、深度不匹配、跨视角超过 200 ms 等均有排除原因。
   辅助目标若与主视角确认的背景冲突，也不能把背景改标为可接触目标。
   没有关联上目标的视角，第一阶段整幅排除，不将其目标误标成背景。
4. 沿用生产深度去噪、捕获时自体过滤及 4 mm 体素降采样；目标和背景分别保留。
   点击视角先按 14 mm 邻接距离检查目标深度连通分量；仅保留包含点击深度且占至少 80%
   目标体素的主体。分离的深度岛改标为背景，原观测几何仍参与碰撞检查，不通过删点制造空隙。
   点击落在小分量或目标严重分裂时明确拒绝，不擅自换选另一个物体。
   GraspGenX **一次推理**使用融合后的目标点，候选碰撞预筛使用对应融合场景。
   路径规划读取同一份保留的融合点云，不退回只用主视角深度重新建场景。

RPent 手动入口的对应显式选项是 `--grasp-view-policy single|auto`；`legacy` 为原默认模式。
搭配 `--debug-target-image-click` 可在同一个拼图窗口选择任一可用视角的目标，坐标会映射回
该视角原图。此组合直接使用人工点击，不另外运行文本检测；非点击模式保留选定视角的
文本检测/终端 UV 选择。单击图像以外的标题或空白区域不会提交目标。
`--graspgenx-wrist-multiview` 对应 `auto`，不再要求主视角必须为 `main`。

新采集证据为 `grasp_observation.zip`；新抓姿场景为
`dexteleop.perception.grasp_multiview_snapshot.v1`，内含各视角原始 RGB-D、目标 mask、元数据和
实际使用的融合点云/标签。结果 `view_fusion` 说明每个参与关联的视角是否采用及原因。
原 `perception.render_point_grasp_candidate@1` 仍是单独的实时投影渲染；不是深度遮挡检查，
其新图像不参与本次冻结观测的推理或规划。

边界：这不是移动中的多视角建图或主动找视角。使用已有 lower 发布的 FK/qpos 历史，要求
两臂在采集附近至少有 100 ms 的连续稳定证据（关节跨度不超过 0.002 rad），并保持原有
250 ms pose age、150 ms 图像/pose 偏差校验；不把接收时间冒充精确曝光时间。
无足够静止历史或检测到运动时拒绝，不发 HOLD、STOP 或位姿命令来制造静止。
没有持续跟踪目标，也不保证观测之后场景不变：物体/机械臂状态变化后应重新采集、选点、规划。
保持已有独立规划、确认执行、碰撞、状态新鲜度和力保护；融合成功不等于可安全执行或抓取成功。
合成测试覆盖主相机不可用、腕部作为点击视角、辅助视角遮挡/视野外、运动/过期及背景标签保护。
另外已经使用只读双腕 RGB-D/qpos 和真实 SAM2/GraspGenX 权重做冻结回放：左腕可见蓝色方块，
右腕不含目标，主相机当次因设备出帧失败不可用。该记录验证了腕部独立观测和深度岛修复，
不是三相机融合增益、完整路径或实际抓取成功的验收。

#### 无运动实验与左右臂对照

只读硬件授权下不要启动会初始化 HOLD 的手动 Gateway。使用
`codex_remote/remote_rm65_grasp_snapshot.sh` 限定探针获取 RGB-D/qpos；它不发送运动、HOLD、STOP、
夹爪或相机 reset。原始数据保留相机内参、设备时间与主机接收时间、双臂关节序列及清理结果。
模型测试由 `remote_grasp_snapshot_inference.sh` 在已保存数组上运行，不再次连接设备。
采集探针复用 rule-base / real-teleop 的 `RealSenseRGBSource` 和串行启动/反序关闭流程，
按配置优先级错峰启动并持续取帧；软件重试预算由共享生命周期管理，没有另一套原生 pipeline
预热/轮询循环。保留颜色传输格式、分辨率、帧率和 RGB-D 对齐，保存图像统一转换为 RGB。
始终禁用三种 hardware-reset 配置及曝光参数写入；诊断读取 source 缓存，不追加原生 SDK 查询。
只有相机连续新鲜帧检查通过后才连接双臂，避免相机启动阻塞连带保留机械臂连接。
阶段和初始 qpos 实时写入 `capture_stages.jsonl`；帧数据与 `snapshot.json` 在设备关闭前保存，
主错误、清理错误分别保留。若原生 USB 调用卡在 Linux D 状态，外层 60 s timeout 仍不能保证
进程退出；启动超时终止本批次，不继续开相机或连接双臂，也不自动重置设备。
`cleanup_complete=false` 不能解释为设备已释放。`captured` 只证明可用视角已完成静止采集且清理
成功，缺失视角列在 `missing_roles`，只有 `all_views_captured=true` 才表示三个视角都保存了五帧，
仍不代表三视角融合、规划或真实抓取验收。RGB-D 同帧，接收/可用/设备时间分别保留，
不会用读取缓存的当前时间把旧帧伪装成新帧。

离线多视角回放可重复传入 `--aux-segmentation DIR`，使用 `infer_frozen_grasp.py` 保存的真实辅助视角
分割。目录中的 RGB 数组和关联种子必须与请求逐一匹配；缺少/错配分割不能用全图 mask 冒充成功融合。

`scripts/probes/replay_bimanual_grasp_snapshot.py` 以冻结 RGB-D、目标 mask、原始关节状态和对应
模型输出进行本地离线对照，`--sides left right` 分别建立左右臂的工具/包络、臂基座方向先验、
候选筛选、IK、局部接近与完整路径请求。相机角色不决定执行臂，也不镜像另一臂的抓姿或 IK。
固定另一臂的实际关节状态参与碰撞；一侧规划失败不阻止另一侧产生诊断。
真实点云需要 `--refine-target-depth`；保留模型输出必须与它逐点匹配，并通过 `--model-manifest`
传入实际推理生产者的 manifest。`--diagnose-endpoints` 额外保留少量终点失败细节，不改变候选选择。

每侧结果单独保存候选池、起点碰撞、筛选漏斗、耗时和路径状态。`no_admitted_candidate` 是预算内
没有方案，不是物体物理上不可抓；`path_found` 才能声明已找到本次离线路径。局部验证在同一次
候选请求内复用固定终点的 IK 搜索，不跨候选/观测/左右臂缓存；不同预抓取距离仍各自验证接近段。
此工具不注册双臂自动执行 Skill，不向设备下发任何轨迹。实时接口仍保持当前部署的单臂授权与
独立执行确认；离线双臂对照不扩大实时运动权限。

种子和已接受关节解的去重对有限六维向量使用逐元素绝对误差比较，维持原 `rtol=0` 语义、
种子的 `1e-9 rad` 与解的 `1e-5 rad` 阈值。只降低重复比较的开销，不改变种子顺序、数量、
求解方法、迭代精度、多分支保留或任何后续准入条件。

原生包络 filter 为密集离线规划提供只读投影检查：每个样本仍校验双臂位姿、工具/参考系绑定、
Home/epoch，并使用同一个绝对、相对和旋转投影内核。无需修改的样本保留原 solver target，
不为它构造冻结的 lineage 报告或内容哈希，也不计入控制 filter 的发布样本数；这不是包络结论缓存。
需要修改的样本仍经普通投影返回完整拒绝/准入诊断。控制调用、Shadow、注入或派生 filter 的
行为不变；局部链及最终密集复核的采样密度、碰撞对、间距、姿态容差和多分支搜索不变。

端点排序探针只回答是否已找到安全终点：拿到第一个通过姿态容差、包络和原生稀疏碰撞检查的
终点便结束；没有证据时仍按原种子和预算搜索，不因此淘汰候选。它不把该关节解交给执行器，
局部链与完整路径仍搜索全部正常 IK 分支。探针的 `attempted_seed_count` 记录实际搜索尝试数，
不再误记为接受分支数。

### 可选端点可达性注释（v3，或 v4 的同名选项）

未启用冻结观测时，该组合还可额外传入 `--enable-point-grasp-arm-reachability`，此时注册显式升级的
`perception.plan_point_grasp@3`；未传入时仍注册原样的 v2，Definition、Schema、Route、Profile 与
Binding hash 均不改变。v3 请求增加必填布尔量 `include_arm_reachability`：为 `false` 时不调用 IK，
结果仍不带可达性字段；为 `true` 时，从同一份新鲜双臂 measured-qpos 快照出发，经现有 FrameGraph
把 `canonical_station` 候选转换到 `dual_arm_solver_world`，再由 lower RM65 target-builder 在每个候选
1 秒 wall-clock 搜索预算且最多 120 个 solve cycle 内评估；观察到截止时间后不再启动新 cycle，但不会
抢占一个正在运行的 lower solve，其完整耗时会写入结果。结果按同一个 IK 解的成对残差分为
`standard_reachable`（3 mm、1°以内）、`basically_reachable`（5 cm、20°以内）和
`difficult_to_reach`，同时返回实际残差、cycle 数和终止原因。该字段只用于筛选提示，不改变模型分数、
候选顺序或选择权；它也不检查整条路径，不证明无碰撞、可执行或抓取成功。qpos 不存在、过期或帧身份
不一致时整个带注释请求 fail closed，而不是把未知状态误报为不可达。

双臂 observation 不再串行等待两个控制器后才检查第一臂的新鲜度，而是对两个独立 RM65 backend
并行发起硬件读取，再在同一个 causal cutoff 下选择结果。每臂 `0.2 s` freshness 上限保持不变；
失败诊断同时保留两臂 measurement age、timestamp span 和各自 SDK read-phase timing，不能靠放宽
阈值掩盖慢读取或控制器异常。

组合 profile 还发布一对职责分离的下游 Route。`perception.plan_grasp_approach@1` 只接受当前 session
仍保留的 `point_grasp_operation_id`、`candidate_ordinal` 和 `speed_scale`，从该不可变候选恢复真实
DMTAC tool-TCP 目标，再绑定一份新鲜 measured-qpos。它依次经过当前部署的 TCP envelope、endpoint
IK、生产 URDF/FCL 场景和完整关节路径规划；不会接受调用方回传的矩阵、轨迹或坐标身份。结果明确为
`path_found` 或 `rejected`，并保留候选/backend/model、Station/frame/tool/calibration、qpos 和 planner
证据。该 Skill 是只读感知/评估：不 mint motion Grant、不 dispatch，也不会因路径失败请求 HOLD；
`motion_or_hold_performed` 恒为 `false`。

只有 `path_found` 才返回最长存活 30 秒的 session-local `grasp_approach_plan_id` 和
`grasp_approach_plan_hash`。操作者或上层 Agent 检查结果后，必须另行显式调用
`motion.execute_grasp_approach_plan@1`；执行参数只能引用同一个 component 和精确 plan ID/hash，不能
改目标或轨迹。执行 Route 不重新规划，并在 PENDING-to-ACTIVE 前重新检查计划寿命、qpos 新鲜度、
起点漂移和取消状态；计划最多消费一次。执行失败才按 motion 域规则进入 HOLD。这样候选质量、完整
路径可行性和是否运动成为三个独立决策，不会把 endpoint reachability 或单帧可见碰撞标志误当作
可执行证明，也不会因为某个候选超出当前 envelope 就自动放宽安全边界。Gateway 启动时需显式授权
独立 action `execute_grasp_approach_plan`；它与兼容 action `approach_grasp_pose` 不互相授权。

原有 `perception.plan_point_grasp@1` 与上文离线 CLI 保持兼容：它允许调用方提供本地 RGB-D、内参和
标定，因此其 `base` 输出不能被称为 deployment-trusted Station pose，也不会由生产
`real-perception` profile 发布。

## DMTAC M 和坐标语义

当前 simulation 默认 URDF 由 `sim/contracts/scene/spec.py` 选定为：

```text
assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf
```

其默认夹爪模块是 `eg2_4b2_cad`，SHA-256 为
`457a2b92c81cc4f6f2f1ada63f1ed7a4a659209c4af4aba7d2dcd8403e3b0620`。真实工具配置来自
`hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json` 和
`rm65_right_DMTAC_M.json`；`tool_tcp` 相对法兰 z=0.14 m。适配器按 DMTAC M 最大开口 0.070 m
过滤候选。当前 v3/v4 4B2 proxy attachment 的内部掌/指边界标记为 z=0.097001 m，且当前模型把
attachment translation 设为零；与已确认的 0.14 m TCP profile 一起形成了适配器显式锁定的
0.042999 m 局部接近轴变换。该 marker 不是独立的法兰安装面实测值；若真实安装或工具 profile
变化，必须更新并重新核查整个受信工具绑定，不能在 RPent/Station Z 轴上追加经验补偿。VGN 的
掌部原点除了轴映射外，还必须应用上述变换才能成为实际 `tool_tcp`；这个变换
同时用于候选、碰撞包络和渲染。VGN 候选还使用 SAM2 目标点云在开口轴和指高轴的 5%/95%
分位中点做有界居中，避免把点击目标上方的掌部原点误报成 TCP。组合真机入口会核验所选臂的
工具配置严格匹配该 DMTAC_M TCP，其他工具 fail-closed，而不在 RPent 中补偿高度。

GraspNet 的 +X
接近、+Y 开口、+Z 高度轴则分别映射到 DMTAC `tool_tcp` 的 +Z 接近、+X 开口、+Y 高度轴。
GraspNet 输出第 3 列的 grasp depth 不能丢弃：实现按官方 0.060 m 指长，把 model seed 沿
`depth - 0.060` 平移到 DMTAC 掌/指边界，再继续平移到实际 `tool_tcp`；同时使用 SAM2 目标点云沿开口轴的 5%/95% 分位
中点做最多 0.020 m 的居中校正，并沿指高轴做最多 0.030 m 的同类居中校正。居中后还必须要求
目标点云的稳健中心同时落在候选开口宽度、DMTAC 指垫高度以及掌部到指尖的纵向区间内；GraspNet
没有 VGN 体素量化对应的有界修正依据，因此越界候选直接 fail-closed，不把目标外抓姿送入运动预检。
这些语义均纳入 planner semantic content hash。
GraspGenX 与 DMTAC M 已共享 +Z 接近、+X 闭合的模型基座约定；适配器使用 open
`(70,25.5,50) mm`、mid `(35,25.5,50) mm` 的内部 sweep-volume，二者中心均在掌/指边界前
25 mm，并把模型基座沿 +Z 平移 42.999 mm 得到实际 `tool_tcp`。建议开口宽度由点击 mask 点云在
候选 +X 轴的 5%/95% 范围加 6 mm 余量得到，超过 70 mm 时拒绝，不会截断成伪可抓取宽度。
可见点云碰撞门禁使用 DMTAC M 的指体/掌体近似包络；落入该实体包络的采样点不超过 400 个时
判为通过，达到 401 个时拒绝。该阈值属于部署实现语义并纳入 planner semantic content hash。

`experiments/sim_point_grasp` 的版本化 tall-box 场景合同同样强制加载上述详细 CAD URDF，
并使用经过转动关节单位换算的六关节 mimic 驱动配置
`dmtac_bimanual_position_eg2_4b2_cad_20260829.v2`；不得回退到
`rm65_dual_arm_wrist_cameras_v3_head_camera.urdf` 中的简化平行夹爪代理。
命令通过 `--arm-side left|right` 明确选择受控臂，并将另一侧保持在镜像 home；所选侧同时决定
`base_frame_id`、CAD TCP、IK 目标、抓取体、两指接触体和夹爪执行。`right` 仅作为兼容默认值，
不是 Skill 的单臂限制。
旧 CAD profile 把直线夹爪的 `1200 N/m`、`20 N` 直接解释为每个转动 mimic 关节的
`1200 N·m/rad`、`20 N·m`，会把轻物体弹飞；v2 使用接触消融后受限的转动参数
`kp=5`、`kv=0.1`、每关节 force limit `0.5`，并保留原始物体质量与摩擦。
当前 `experiments/sim_point_grasp` 的研究消融配置把规划预防性 clearance 设为 0 mm，把
禁止接触的 physics-tick 硬穿透上限设为 5 mm。碰撞检查、硬关节限位和严格抓取成功
条件仍然启用；该配置不是真机安全 profile。
为隔离此前提出的失败改进，该实验还提供两个非默认、哈希锁定的仿真变量：
`dmtac_bimanual_position_eg2_4b2_cad_20260831.v3_strong` 把每个耦合转动关节的 force limit
从 0.50 提高到 0.75 N m，并配合 30 步、每步 0.25 的慢速持续闭合；200 g tall-box 合同则把
80 g 物体的质量和三轴惯量统一乘以 2.5。两者都仅用于刚体仿真消融，不能作为真机力控参数。
实验还可从首个点击得到的 SAM2 mask 内确定性采样共 8 个分散点，各自调用同一个正式 provider，
按 4 mm/5 度去除重复抓姿后最多把 10 个候选送入 IK 和完整场景碰撞预检；该选择不查看后续接触
或抬升结果。完整的 5×20 运行矩阵和命令见
[`experiments/sim_point_grasp/README.md`](../../experiments/sim_point_grasp/README.md)。
当前近距离消融的物体中心为世界坐标 `(-0.272, 0.0) m`，比历史中心 `(-0.472, 0.0) m`
沿 +X 向固定头部相机靠近 20 cm；X/Y 默认扰动仍为 2.5/8 cm。20 次回归的点击像素
深度为 0.572--0.601 m（均值 0.585 m），20/20 都返回候选，但严格抓取成功仍为 0/20。
该实验另用分别标记物理侧别、哈希锁定的
`rm65_left_DMTAC_M_4b2_cad_tool_tcp.json` 与
`rm65_right_DMTAC_M_4b2_cad_tool_tcp.json`。两者保留标定的 0.14 m 深度，
并把 TCP +X 旋转到 CAD 的真实张合方向、补偿约 24.5 mm 的 CAD 横向中心偏置。

输出矩阵均为 4×4 row-major 齐次变换，平移单位米：

- `T_camera_from_tcp`：把 TCP 坐标转换到相机 optical frame；
- `T_base_from_tcp`：`T_base_from_camera @ T_camera_from_tcp`；
- `tcp_xyz_*_m` 是对应矩阵平移列的便捷副本。

共享投影实现位于 `deployments.perception.dmtac_grasp_overlay`。给定同一 reference frame 下的
`T_reference_from_tcp` 与 `T_reference_from_camera`，它按
`T_camera_from_tcp = inv(T_reference_from_camera) @ T_reference_from_tcp` 得到任意视角中的抓姿，
返回线段可见数、TCP 像素和 `visible`、`partially_visible`、`behind_camera`、`out_of_frame`
之一。Skill 与离线 CLI 的 annotated preview 只渲染已选中的 ordinal 0，JSON 结果仍保留全部
排序候选，避免多副线框混在同一图像中。仿真实验进一步为每个“相机 × 候选”输出独立 PNG，
三视角 montage 也只展示 ordinal 0。

共享投影器本身不依赖 MuJoCo 或 RealSense。MuJoCo 验证覆盖 `front`、`left_wrist`、
`right_wrist` 三视角；`front` 使用场景固定相机，两个腕部验证视图保留各自相机原点，
但使用明确标记为 `visualization_only_look_at_override_from_wrist_origin` 的 MuJoCo free camera
朝向方块，并从实际 GL camera 反算投影外参。这不是生产腕部标定，也没有修改 URDF 或相机
registry。生产 Skill 只广告当前部署已经具备可信 RGB 和相机 pose 的角色。camera-only profile
仍只有具备静态外参和 aligned depth 的 `main`；组合 `real-rm65` 则由同一 lower owner 提供两个腕部
的动态 FK pose，因此可额外广告 `wrist_left`/`wrist_right`。任何角色都不能仅靠调用方字符串启用。

## 抓后物体目标规划（可选，只读）

`perception.plan_object_move@1` 在显式装配后提供物体中心的 Motion Plan：
已有抓姿/到达/夹爪 Skill → 抓后位姿与夹持验证 → 手物相对变换 → 物体目标位姿 → 携物路径规划。
它不把“闭爪命令完成”解释为抓取成功，不自动试抬、开闭夹爪、执行搬运或放置。
现有真机 CLI 默认不广告此能力；缺少抓后观测提供者时不能仅加一个开关启用。

参数为 `object_id`、`side`、`reference_frame_id`、16 个 row-major 数值的
`T_reference_from_object_goal` 和 `speed_scale`。目标位姿是**物体**的目标，不是 TCP 目标。
若抓后观测给出 `T_reference_from_tool` 与 `T_reference_from_object`，则
`T_tool_from_object = inv(T_reference_from_tool) @ T_reference_from_object`，
`T_reference_from_tool_goal = T_reference_from_object_goal @ inv(T_tool_from_object)`。
物体坐标必须是稳定的实体坐标；对称性/遮挡造成的不可观测方向不能由包围盒任意决定。
请求、观测、场景、求解器必须使用同一已绑定参考系；本入口不猜测坐标系，也不自动投影或裁剪目标。

装配入口为 `build_object_transport_registration`，把注册传入
`build_real_agent_primitive_contract_bundle(object_transport_registration=...)`，再向
`assemble_real_agent_primitive_gateway_session` 传入 `held_object_observer` 和
`object_transport_planner_factory`。前者必须提供相关联的 `GraspedObjectObservation` 和
`ObstacleSceneSnapshot`；后者每次构造独立的 `RM65ObjectTransportPlanner`。
观测来自部署绑定的提供者，调用方不得在 Skill 参数中伪造附着关系、夹持证据或删除障碍。

- 抓后观测包含双臂 qpos、工具/物体位姿、完整保守的物体定向包围盒、位姿及夹持证据引用；
  同次观测必须相关联且不超过 200 ms。必须已经离开支撑面；初始离桌接触转换不属于这个自由空间搬运规划器。
- `verify_rigid_follow` 检查已有、独立授权运动前后的手物关系；静止两帧不能证明抓稳。
  位姿引用和夹持引用由可信观测提供者承担证据责任，非空字符串本身不构成物理证明。
- 只按目标物体身份移除原世界占据；保留桌面、邻物、另一臂及固定场景结构。携物几何会随工具旋转和平移。
- 复用当前稀疏 proxy 碰撞对、IK、路径搜索、C2 时间参数化及最终稠密复查，不改安全包络和 clearance。
  仅所选臂的手部 proxy 作为已验证夹持的接触伙伴豁免；这些粗 proxy **不能证明指腹/掌部接触细节**，
  不能把本规划结果当成首次抓取质量验证。未知或不完整的物体几何拒绝规划。
- 成功返回不可变 NPZ（双臂 qpos、时间、手物关系、目标与验证信息）；`execution_authorized=false`。
  它不是 `motion.execute_grasp_approach_plan` 的令牌，不可借用该 Skill 直接执行搬运。

当前验证分开计量：原生 RM65 左右臂/两种 proxy profile 的规划测试；Gateway 注册、拒绝和 Artifact
测试；MuJoCo 简化平行夹爪物理试抬/绕障搬运测试。最后一项使用仿真真值位姿、自由物体（无物体焊接约束），
不是 DMTAC 真机抓取成功证明，也不是视觉 6D 位姿模型验收。
暂未部署新的真机位姿模型。[FoundationPose](https://github.com/NVlabs/FoundationPose) 可作为后续提供者，
但需要 CAD，或先准备参考图像/物体表示；单帧分割与可见点云不能代替完整物体几何。

## 已知局限

- 单帧深度看不到遮挡面；`visible_collision_free=true` 只表示候选未与当前可见点云的夹爪近似包络
  冲突，不是完整场景、机械臂连杆或运动路径无碰撞证明；
- SAM2 点击分割可能选中整个物体，也可能选中把手等局部，结果受图像质量和点击位置影响；
- VGN 使用合成数据训练，DMTAC M 几何仅做宽度和可见包络适配，真实抓取前需要离线数据评估、
  标定校验和正式下游规划；
- GraspNet 本身不理解点击语义；SAM2 mask 同时限定其局部三维推理工作区并参与后置表面距离
  过滤。单帧遮挡、仿真到真实的域差异和后置过滤仍可能导致无候选或错误候选；
- GraspGenX 当前使用 DMTAC M 的参数化 sweep-volume，而不是上游 gripper asset 中不存在的
  DMTAC 模型；零样本跨夹爪能力必须通过本项目离线数据和 sim 评估，不能仅凭上游声明视为真机有效；
- 离线 v1 接受固定相机显式外参；生产 v2 只公开部署已绑定的静态 depth 相机，尚不支持腕部相机
  的动态 FK 外参；
- 未下载 checkpoint 时只能完成合同/单元测试，不能声称真实模型端到端验证通过。
