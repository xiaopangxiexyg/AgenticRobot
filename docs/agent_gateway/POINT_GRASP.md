# SAM2 + VGN/GraspNet 点击引导 DMTAC M 抓姿 Skill

`perception.plan_point_grasp` 接受一对像素对齐的 RGB-D 单帧、点击像素、固定相机标定，返回最多
5 个已排序的 DMTAC M `tool_tcp` 抓姿。每个候选同时包含 `T_camera_from_tcp`、
`T_base_from_tcp`、建议开口宽度、模型分数、到点击区域的距离和单帧可见点云碰撞近似结果；
`selected_ordinal=0` 是当前最佳候选，`backend_id` 明确报告 `vgn` 或 `graspnet`，并输出一张标注
JPEG Artifact。VGN 是默认后端；GraspNet 是已取得项目使用与再分发授权的可选后端。

该 Route 固定为 `read_only` / `t0_read_only`，不创建 RM65 backend、不执行 IK、轨迹规划、夹爪命令
或机械臂运动。输出抓姿不能绕过下游正式 motion owner、碰撞场景和安全门禁。

## 算法与第三方源码

- [SAM2](https://github.com/facebookresearch/sam2.git) 使用正点击点分割点击所在物体/部位；
- [VGN](https://github.com/ethz-asl/vgn.git) 从点击区域的 30 cm、40³ TSDF 生成 6DoF 二指抓姿；
- 可选的 [GraspNet baseline](https://github.com/graspnet/graspnet-baseline.git) 先用 SAM2 点击 mask
  的米制三维包围盒加 12 cm margin 限定推理工作区，再生成 6DoF 二指抓姿；候选随后仍用
  SAM2 目标表面距离、点击距离和 DMTAC M 几何过滤，完整对齐深度仍用于碰撞门禁；
- AgenticRobot 用 Git submodule 保存第三方源码边界，不把源码文件 vendor 到父仓历史。安装脚本固定 SAM2
  `aa9b8722d0585b661ded4b3dff1bd103540554ae` 与 VGN
  `d7af0622433f52ae88ebe81533f12b46b33e951a`；选择 GraspNet 时固定 commit
  `280c215129f759ed8649cb4e89fc5dfee55f4f80`。

四个源码仓库分别由各自上游维护，父仓只跟踪官方 URL 和固定 gitlink。项目已经取得所需源码与
模型资产的使用及再分发授权；权重仍不提交 Git，而是按运行资产合同放入统一 `assets/` bundle。

一键安装源码和 Python inference 依赖：

```bash
git submodule update --init --recursive -- \
  model/perception/sam2 model/perception/GroundingDINO \
  model/perception/vgn model/perception/graspnet-baseline
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/install_point_grasp_models.sh --backend cuda
```

脚本会先安装/验证 `model/perception/` 下的 SAM2 + Grounding DINO submodule，再从同一目录
editable 安装 VGN。Skill 只使用 VGN 的 headless 网络部分，不要求 ROS。
脚本不下载任何 checkpoint。

需要同时安装可选 GraspNet 时，一条命令完成所有源码与 PointNet2 CUDA 扩展安装：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/install_point_grasp_models.sh \
  --backend cuda \
  --with-graspnet \
  --cuda-arch-list 8.9
```

`8.9` 是 RTX 4090 的 CUDA compute capability；其他 GPU 应替换成对应值。若已用本项目脚本安装过
并验证了固定版本的 SAM2/Grounding DINO，可增加 `--reuse-installed-perception` 避免重复安装。
旧的 `--accept-graspnet-noncommercial-license` 仅保留为兼容 no-op，新安装不需要传入。

## 下载默认 VGN 权重并组成共享 assets

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

export DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets
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

随后执行：

```bash
python scripts/dev_asset_store.py configure --asset-root "${DEXTELEOP_ASSET_ROOT}"
python scripts/dev_asset_store.py status
python scripts/dev_local_resources.py audit
```

回头上传统一网盘资产时，把这两个 `perception/` bundle 与基础 assets 一起打包，并同时发布包的
SHA-256；不要把 SAM2/VGN Git checkout 打进 `assets.zip`。

## GraspNet 后端资产

GraspNet 使用官方 RealSense checkpoint：

- 官方源码与许可证：[graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline.git)
- 官方 checkpoint 页面：[checkpoint-rs.tar on Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)
- Google Drive 文件 ID：`1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk`
- 必须保持的文件名：`checkpoint-rs.tar`

checkpoint 不进入 Git，但可放入项目统一 assets 根并随审核后的 `assets.zip`/团队网盘资产发布：

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

## Gateway 启动

在完整 `real-perception` 参数上同时启用共享 SAM2 localizer 和 point-grasp：

```bash
python -m agent_gateway real-perception \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --camera-config /absolute/path/to/agent_gateway_realsense.json \
  --camera-registry /absolute/path/to/rm65_camera_registry.json \
  --artifact-root /absolute/path/to/runtime-artifacts \
  --enable-object-localization \
  --object-localization-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --enable-point-grasp \
  --point-grasp-backend vgn \
  --point-grasp-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_vgn_point_grasp"
```

Gateway 使用 GraspNet 时，把最后三个 point-grasp 参数替换为：

```bash
  --enable-point-grasp \
  --point-grasp-backend graspnet \
  --point-grasp-asset-bundle "${GRASPNET_BUNDLE}"
```

生产请求通过 `rgb_image` 与 `aligned_depth_m` 两个 Artifact slot 输入。仅同机测试时可重复添加
`--allow-local-image-root /absolute/test/root`，然后参数中的 `local_rgb_path`、`local_depth_path`
可以位于不同的已允许根；该功能默认关闭，SSH read-only client 存在时禁止启用。

## DMTAC M 和坐标语义

当前 simulation 默认 URDF 由 `sim/contracts/scene/spec.py` 选定为：

```text
assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf
```

其默认夹爪模块是 `eg2_4b2_cad`，SHA-256 为
`457a2b92c81cc4f6f2f1ada63f1ed7a4a659209c4af4aba7d2dcd8403e3b0620`。真实工具配置来自
`hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json` 和
`rm65_right_DMTAC_M.json`；`tool_tcp` 相对法兰 z=0.14 m。适配器按 DMTAC M 最大开口 0.070 m
过滤候选，并把 VGN 的 +Y 开口轴映射到 DMTAC `tool_tcp` 的 +X 开口轴；GraspNet 的 +X
接近、+Y 开口、+Z 高度轴则分别映射到 DMTAC `tool_tcp` 的 +Z 接近、+X 开口、+Y 高度轴。
GraspNet 输出第 3 列的 grasp depth 不能丢弃：实现按官方 0.060 m 指长，把 model seed 沿
`depth - 0.060` 平移到 DMTAC 掌/指边界；同时使用 SAM2 目标点云沿开口轴的 5%/95% 分位
中点做最多 0.020 m 的居中校正，并沿指高轴做最多 0.030 m 的同类居中校正。三项语义均纳入
planner semantic content hash。
可见点云碰撞门禁使用 DMTAC M 的指体/掌体近似包络；落入该实体包络的采样点不超过 400 个时
判为通过，达到 401 个时拒绝。该阈值属于部署实现语义并纳入 planner semantic content hash。

`experiments/sim_point_grasp` 的版本化 tall-box 场景合同同样强制加载上述详细 CAD URDF，
并使用经过转动关节单位换算的六关节 mimic 驱动配置
`dmtac_bimanual_position_eg2_4b2_cad_20260829.v2`；不得回退到
`rm65_dual_arm_wrist_cameras_v3_head_camera.urdf` 中的简化平行夹爪代理。
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
该实验另用哈希锁定的 `rm65_right_DMTAC_M_4b2_cad_tool_tcp.json`，保留标定的 0.14 m 深度，
并把 TCP +X 旋转到 CAD 的真实张合方向、补偿约 24.5 mm 的 CAD 横向中心偏置。

输出矩阵均为 4×4 row-major 齐次变换，平移单位米：

- `T_camera_from_tcp`：把 TCP 坐标转换到相机 optical frame；
- `T_base_from_tcp`：`T_base_from_camera @ T_camera_from_tcp`；
- `tcp_xyz_*_m` 是对应矩阵平移列的便捷副本。

## 已知局限

- 单帧深度看不到遮挡面；`visible_collision_free=true` 只表示候选未与当前可见点云的夹爪近似包络
  冲突，不是完整场景、机械臂连杆或运动路径无碰撞证明；
- SAM2 点击分割可能选中整个物体，也可能选中把手等局部，结果受图像质量和点击位置影响；
- VGN 使用合成数据训练，DMTAC M 几何仅做宽度和可见包络适配，真实抓取前需要离线数据评估、
  标定校验和正式下游规划；
- GraspNet 本身不理解点击语义；SAM2 mask 同时限定其局部三维推理工作区并参与后置表面距离
  过滤。单帧遮挡、仿真到真实的域差异和后置过滤仍可能导致无候选或错误候选；
- 当前 Skill 只接受固定相机的显式外参，不支持腕部相机的动态 FK 外参；
- 未下载 checkpoint 时只能完成合同/单元测试，不能声称真实模型端到端验证通过。
