# SAM2 + Grounding DINO 定位与分割 Skill

当前在线 Gateway 默认使用 SAM3 服务，启动和能力说明见
[默认 SAM3 服务](../../agent_gateway/README.md#默认-sam3-服务文本目标与图片点选)。
本页保留显式 SAM2 后端的资产、离线诊断和兼容能力；使用时必须选择
`--object-localization-backend sam2-grounding-dino`，不再由资产目录自动选择后端。

`perception.localize_objects` 对一张 JPEG/PNG 执行只读物体定位：

- `language_prompt=null`：SAM2 automatic mask generator 产生全图物体候选，再投影为 bounding box；
- `language_prompt="apple"`：Grounding DINO 产生文本候选，SAM2 精化 mask，再返回 mask 的 bounding box；
- 结果坐标固定为 `xyxy_pixel_0based_half_open`，同时返回一张画框后的 JPEG Artifact；
- 该 Skill 不读取相机、不控制机器人，也不取得 motion/HOLD authority。

`perception.segment` 直接读取已注册的实时 RGB 相机，用同一个 SAM2 +
Grounding DINO localizer 返回一个 mask：

- 文本提示使用 Grounding DINO 定位后交给 SAM2 分割；
- 像素点提示直接使用 SAM2；
- 输出为 bool NPY mask 和 JPEG overlay 两个 Artifact；
- Route 仍是 `read_only`，不发送运动或夹爪指令。

## 外部源码

源码以 Git submodule 位于 AgenticRobot 的 `model/perception/`，不放进项目运行资产包。使用以下官方
Git 仓库：

- [SAM2](https://github.com/facebookresearch/sam2.git)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)

统一安装到 AgenticRobot 主环境 `dexteleop310`。先激活它，确保 AgenticRobot 固定的
`torch==2.10.0+cu126`、NumPy 和 Pillow 已就绪；SAM2 与 Grounding DINO 源码包再按下述方式
从 Git checkout 安装。

部署前应检出经过审查的具体 commit；运行资产的 `manifest.json` 必须记录两个完整的 40 位
commit ID。推荐直接运行仓库内的一键脚本：

```bash
git submodule update --init --recursive -- \
  model/perception/sam2 model/perception/GroundingDINO
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/install_perception_models.sh --backend cuda
```

checkout 固定位于当前 worktree 的 `model/perception/`。脚本固定并验证 SAM2
`aa9b8722d0585b661ded4b3dff1bd103540554ae` 与 Grounding DINO
`856dde20aee659246248e20734ef9ba5214f5e44`，安装 Python 包、自动检测 GPU compute capability、
构建 `_C` 扩展并实际执行一个 CUDA 算子。无 GPU 调试机可传 `--backend cpu`。脚本不会下载权重，
也不再向 Conda prefix 另行 clone 源码。

上述两个 commit 是当前经过 API 对照的推荐版本。SAM2 的 CUDA connected-components 扩展只影响部分
后处理，因此这里关闭；Grounding DINO 的 `_C` 扩展则是 CUDA 推理必需的。官方 Grounding DINO
在该 commit 仍把旧 `Tensor::type()` 传给 dispatch 宏，AgenticRobot 固定的 PyTorch 2.10 已不再接受；
`grounding_dino_torch_2_10_compat.h` 仅在编译时预包含一个类型转换 overload，不修改第三方 checkout。
本机 RTX 4090 使用 `TORCH_CUDA_ARCH_LIST=8.9`；部署到其他 GPU 时应改为对应 compute capability。
CUDA 12.1 toolkit 编译 PyTorch cu126 扩展会报告 minor-version warning，当前实机构建已通过。

不要在 manifest 中使用分支名或 `HEAD`。示例 manifest 已固定上述两个完整 revision；如果重新审查并
升级源码，必须同时更新 manifest。
运行时会从两个已安装 package 所在的 Git checkout 读取 `HEAD` 并与 manifest 比较，因此这里使用
editable checkout；缺少 `.git` provenance 的复制目录或与 manifest 不同的 revision 会拒绝启动。

## 下载并准备权重

项目级基础/感知资产包的下载、解压和 `DEXTELEOP_ASSET_ROOT` 配置统一见项目根
[`README.md`](../../README.md) 的“运行资产”章节。以下命令假定共享根已经按该章节配置；本页只
说明如何准备本 Skill 的可选权重 bundle：

```bash
mkdir -p "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino"

curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino/sam2.1_hiera_large.pt"

curl -fL \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino/groundingdino_swint_ogc.pth"

sha256sum \
  "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino/sam2.1_hiera_large.pt" \
  "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino/groundingdino_swint_ogc.pth"
```

复制
[`sam2_grounding_dino_asset_manifest.example.json`](./examples/sam2_grounding_dino_asset_manifest.example.json)
为 bundle 中的 `manifest.json`，然后用上一步的真实 SHA-256 替换权重的全零占位值。Gateway 启动时
会重新计算 checkpoint SHA-256；文件、manifest 或源码身份不一致
都会拒绝启动。

完成 bundle 后，回到项目根 README 的“运行资产”章节执行一次 `status` 和
`dev_local_resources.py audit`；本页不重复共享资产登记或资产包发布步骤。

## 使用一张本地图片验证真实模型

离线部署还依赖 Grounding DINO 配置指定的文本编码器（当前受审配置为
`bert-base-uncased`），不只是 SAM2 和检测器的两个 checkpoint。启动前必须把
`HF_HOME` / `HF_HUB_CACHE` 绑定到已准备的模型缓存；若还设置了
`HUGGINGFACE_HUB_CACHE` 或 `TRANSFORMERS_CACHE`，须指向同一个 hub，避免优先级
覆盖。这些变量应在导入 Transformers 前由会话设置，不能依赖其他终端的缓存环境。
保持 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；缺文件应在设备启动前报错，
不得通过关闭离线模式或重开相机恢复。

```bash
python scripts/smoke_sam2_grounding_dino.py \
  --asset-bundle "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --device cuda --verify-only
```

此检查不连接设备，验证受审源码、权重、离线文本配置/分词器与文本权重，并实际执行
一个小型 Grounding DINO CUDA 算子。这样可在启动相机前拦截扩展与 PyTorch 的 ABI
不兼容，而不是只看 `_C` 文件存在或 import warning。省略 `--verify-only` 可额外
加载/关闭真实模型；它仍不启动相机或机械臂。源码 revision 一致不保证编译产物兼容，
跨环境复用扩展时必须在目标 PyTorch 环境中重新验证；需要重建时使用独立源码/构建
目录，不覆盖共享已安装包。CPU 仅在操作者明确选择 `--device cpu` 时使用，不自动降级。

这是最短的权重与模型端到端验证，不启动 RealSense 或 Gateway，也不连接机械臂：

```bash
mkdir -p /tmp/dexteleop-perception-output

# SAM2 自动 bbox
python -m deployments.perception.localize_image_cli \
  --asset-bundle "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --input-image /absolute/path/to/frame.png \
  --output-json /tmp/dexteleop-perception-output/automatic.json \
  --output-image /tmp/dexteleop-perception-output/automatic.jpg

# 文本指定目标：Grounding DINO 检测，SAM2 精化
python -m deployments.perception.localize_image_cli \
  --asset-bundle "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --input-image /absolute/path/to/frame.png \
  --language-prompt apple \
  --output-json /tmp/dexteleop-perception-output/apple.json \
  --output-image /tmp/dexteleop-perception-output/apple.jpg
```

CLI 会先验证 manifest、两个权重 SHA-256 和已安装源码 commit，再加载真实模型。JSON 输出包含原图
SHA-256、模型身份、图片尺寸、模式、prompt、score 和 `xyxy_pixel_0based_half_open` bbox；画框结果
固定为 JPEG。输出存在时默认 fail closed，需要覆盖时显式传 `--overwrite`。

## 启动完整 Gateway Skill

在原有 `real-perception` 参数上增加：

```bash
python -m agent_gateway real-perception \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --camera-config /absolute/path/to/agent_gateway_realsense.json \
  --camera-registry /absolute/path/to/rm65_camera_registry.json \
  --artifact-root /absolute/path/to/runtime-artifacts \
  --enable-object-localization \
  --object-localization-backend sam2-grounding-dino \
  --object-localization-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino"
```

模型在 Unix socket 发布前加载；缺少外部源码包、权重或正确 manifest 时启动失败。默认使用 `cuda`，
调试 CPU 路径可显式传 `--object-localization-device cpu`，但不承诺实时性能。

启用 object localizer 后，真机 perception Gateway 会在同一 endpoint 注册
`perception.localize_objects` 和 `perception.segment`。RM65 主运动 Gateway 启用点抓姿
组合时也会在同一 endpoint 注册 `perception.segment`，并复用点抓姿已加载的
localizer，不再创建第二份模型。

MuJoCo 主 Gateway 通过显式的可选资产参数启用同一 Skill：

```bash
python -m agent_gateway mujoco \
  --object-localization-backend sam2-grounding-dino \
  --artifact-root /absolute/path/to/runtime-artifacts \
  --segmentation-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino" \
  --segmentation-device cuda \
  ...
```

未启用感知功能时不注册 `perception.segment`，也不加载模型；原有无模型 Gateway 的 Route
和运行方式不变。默认 SAM3 可直接用 `--enable-segmentation --artifact-root ...` 启用，
无需 SAM2 bundle；两种后端都沿用会话启动、取消与关闭的原 owner。

## 实时分割输入

`perception.segment` 要求一个相机角色、一个阈值，以及恰好一种 prompt：

```json
{
  "role": "main",
  "language_prompt": "red cube",
  "target_pixel_uv": null,
  "min_score": 0.35
}
```

```json
{
  "role": "wrist_left",
  "language_prompt": null,
  "target_pixel_uv": [320, 240],
  "min_score": 0.0
}
```

对外 parameter schema 使用简单 nullable 字段；“文本或点恰好一个”由 Gateway 的
权威 payload validator 再次检查。这避免将 provider 兼容性不稳定的 `oneOf/anyOf`
交给 Planner，但没有放宽实际执行条件。

## 输入模式

生产模式使用一个 `input_image` Artifact binding。输入 Artifact 必须是声明为
`dexteleop.perception.object_localization_input_image.v1` 的 JPEG RGB8 或 PNG RGB8；图片通过既有
`chunked_rpc` 上传，不能把持久路径伪装成 Artifact identity。

参数示例：

```json
{
  "input_source": "artifact",
  "local_image_path": null,
  "language_prompt": "apple",
  "max_detections": 20
}
```

生产 Gateway 不接受本地路径。需要对本机文件运行模型时，使用前文独立的
`deployments.perception.localize_image_cli`；以下 `local_path` 请求不是生产合同：

```json
{
  "input_source": "local_path",
  "local_image_path": "/absolute/path/to/test-images/frame.png",
  "language_prompt": null,
  "max_detections": 50
}
```

本机路径功能默认关闭；路径必须位于显式允许根目录内，且不能同时绑定输入 Artifact。它与
Artifact data plane 的 `local_path` 下载 grant 不同，不应开放给 SSH read-only 调用方或不受信任用户。
