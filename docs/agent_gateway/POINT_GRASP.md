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
  并用 DMTAC M 的 open/mid sweep-volume 条件生成跨夹爪 6DoF 候选；默认通过隔离的 loopback
  service 加载 GPU 模型，in-process 仅供兼容的独立 Gateway Python 环境显式选择，候选仍经过
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
统一位于 `assets/perception/graspgenx-point-grasp`。默认 service 模式使用 bundle 内独立 venv；
in-process 模式要求 Gateway 所在 Python 环境本身兼容 GraspGenX，必须显式选择。安装器自动
下载官方 Hugging Face 模型仓、生成四个文件的真实 SHA-256 manifest，并保持主
`dexteleop310` 不变：

```bash
bash scripts/install_graspgenx_env.sh
bash scripts/install_graspgenx_env.sh --verify-only
```

安装器只在该 bundle 的 `runtime/.venv` 中安装 GraspGenX 推理链和 service 备选，不会修改共享
`dexteleop310`。默认由 `uv --torch-backend auto` 选择 PyTorch wheel；已明确 CUDA
版本的 GPU 主机可传入例如 `--torch-backend cu126`。上游完整依赖中的
PyOpenGL/pyrender 版本互斥，因此本服务不安装未使用的渲染、训练和端到端仿真依赖。

service 模式的模型服务只应绑定 loopback；启动命令见 `scripts/run_graspgenx_server.sh`。服务健康
后，在 `real-perception` 或 `real-rm65` 的原启动命令中增加：

```bash
bash scripts/run_graspgenx_server.sh \
  --asset-root "${DEXTELEOP_ASSET_ROOT}" \
  --host 127.0.0.1 \
  --port 5556
```

如果已准备好与 GraspGenX 官方依赖兼容的 Gateway Python 环境，可不启动上述服务，在原
`real-perception` 或 `real-rm65` 命令中选择 direct Python backend：

```text
--point-grasp-backend graspgenx
--graspgenx-mode in-process
--point-grasp-asset-bundle /absolute/path/to/assets/perception/graspgenx-point-grasp
--graspgenx-source-root /absolute/path/to/model/perception/graspgenx
--point-grasp-device cuda
```

in-process 会在 Gateway 的 `start()` 阶段加载一次 generator/discriminator，按 sweep-volume
缓存 sampler，并在同一进程内串行调用模型。它不改变 SAM2 mask、DMTAC M TCP 变换、可见深度
碰撞过滤、RM65 可达性或下游运动权限。当前官方 checkpoint 要求 PyTorch <2.7；不要把它直接
安装或升级到项目的 `dexteleop310`，应先用兼容的独立 Gateway 环境完成 GPU smoke 和仿真回归。

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

显式的 in-process backend 不需要运行服务，可在兼容环境中直接验证：

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

GraspGenX 不进入当前 in-process 离线图片 CLI；这是有意的环境隔离。真实 RGB-D 仍由同一个
Gateway provider 在一次 retained frame 读取中送入 SAM2 和 GraspGenX，不改变现有同帧深度、
Station frame 或 Artifact lineage 语义。

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

python -m deployments.perception.point_grasp_image_cli \
  --sam2-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --point-grasp-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_vgn_point_grasp" \
  --input-rgb /absolute/path/to/frame.png \
  --input-depth /absolute/path/to/aligned_depth_m_float32.npy \
  --calibration-json /tmp/fixed_camera_calibration.json \
  --target-u 640 --target-v 360 \
  --max-candidates 5 \
  --output-json /tmp/dexteleop-point-grasp-output/result.json \
  --output-image /tmp/dexteleop-point-grasp-output/annotated.jpg
```

该命令加载真实 SAM2/VGN 并使用 GPU，但不启动 RealSense、不连接机械臂。输出存在时默认拒绝覆盖；
确认需要替换时加 `--overwrite`。

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
`perception.plan_point_grasp@2`。GraspGenX 是默认 backend，并默认要求预先启动隔离 loopback
service；只有显式指定 `--graspgenx-mode in-process` 才在兼容的 Gateway Python 中直接加载模型。
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

该组合还可额外传入 `--enable-point-grasp-arm-reachability`，此时注册显式升级的
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

```text
assets/rm65_inspire_gripper_v4/rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf
```

本流程使用 `robot_motion/configs/rm65_camera_registry_v4.json`：它登记 v4 主相机标定和上述详细
URDF placement，同时保留现场已验证的左右腕相机 serial 绑定。组合成功时 render 的可选角色严格为
`main`、`wrist_left`、`wrist_right`；腕部 pose 缺失、过期、晚于当前时刻或与图像采集时间偏差过大
都会 fail closed。启动这个组合会连接双 RM65 并建立 startup HOLD，因此不能把它当作 camera-only
验收入口；离线测试也不构成真机坐标或运动验收。

双臂 observation 不再串行等待两个控制器后才检查第一臂的新鲜度，而是对两个独立 RM65 backend
并行发起硬件读取，再在同一个 causal cutoff 下选择结果。每臂 `0.2 s` freshness 上限保持不变；
失败诊断同时保留两臂 measurement age、timestamp span 和各自 SDK read-phase timing，不能靠放宽
阈值掩盖慢读取或控制器异常。

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
中点做最多 0.020 m 的居中校正，并沿指高轴做最多 0.030 m 的同类居中校正。三项语义均纳入
planner semantic content hash。
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
