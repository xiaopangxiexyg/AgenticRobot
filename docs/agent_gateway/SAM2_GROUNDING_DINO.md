# SAM2 + Grounding DINO 物体定位 Skill

`perception.localize_objects` 对一张 JPEG/PNG 执行只读物体定位：

- `language_prompt=null`：SAM2 automatic mask generator 产生全图物体候选，再投影为 bounding box；
- `language_prompt="apple"`：Grounding DINO 产生文本候选，SAM2 精化 mask，再返回 mask 的 bounding box；
- 结果坐标固定为 `xyxy_pixel_0based_half_open`，同时返回一张画框后的 JPEG Artifact；
- Skill 不读取相机、不控制机器人，也不取得 motion/HOLD authority。

## 外部源码

源码以 Git submodule 位于 AgenticRobot 的 `model/perception/`，不放进 `assets.zip`。使用以下官方
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

先下载 README 指定的基础 `assets.zip` 并解压。然后在同一个共享资产根增加可选 bundle：

```bash
export DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets
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

配置并检查共享资产：

```bash
python scripts/dev_asset_store.py configure --asset-root "${DEXTELEOP_ASSET_ROOT}"
python scripts/dev_asset_store.py status
python scripts/dev_local_resources.py audit
```

如果要重新上传统一资产包，应在包含完整基础 bundles 的父目录执行打包，而不是只打包两个权重：

```bash
cd /absolute/path/containing-assets
zip -r assets.zip assets
sha256sum assets.zip
```

随后将 `assets.zip` 和其 SHA-256 上传到约定网盘；网盘地址确定后再更新仓库根 README 的下载链接。

## 使用一张本地图片验证真实模型

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
  --object-localization-asset-bundle \
    "${DEXTELEOP_ASSET_ROOT}/perception/sam2_grounding_dino"
```

模型在 Unix socket 发布前加载；缺少外部源码包、权重或正确 manifest 时启动失败。默认使用 `cuda`，
调试 CPU 路径可显式传 `--object-localization-device cpu`，但不承诺实时性能。

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
