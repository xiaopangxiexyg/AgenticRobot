# 数据、分析与测试入口

详细子工具和输出说明见 [后处理与测试](./POSTPROCESS_AND_TESTING.md)；需要做交付 gate 时使用
[v2.3 验收模板](./ACCEPTANCE_V23.md)。长程任务采集后的细粒度阶段标注见
[长程任务细粒度阶段标注](../real/LONG_HORIZON_STAGE_ANNOTATION.md)。向其他人说明或交付实际
数据集时，先阅读 [v2.5-p2 数据格式](./v25p2/README.md)，再使用
[数据集内容与交付说明](./v25p2/DATASET_CONTENT_AND_HANDOFF.md)。
数据集整体迁移、备份或将旧路径替换为软链接前，必须先阅读其中的
“数据集迁移、hardlink 与软链接”章节；不同路径可能共享同一 inode，不得将路径副本
默认视为独立备份。
以下命令均不连接真实执行器。

## 1. 数据后处理

先 dry-run，再显式应用：

```bash
python -m utils.dataset_postprocess postprocess "${DATASET_ROOT:?set DATASET_ROOT}"
python -m utils.dataset_postprocess postprocess "${DATASET_ROOT:?set DATASET_ROOT}" --apply
```

对于可能同时包含 raw、zstd/lz4，或已经通过 receipt 迁出 tactile 的数据集，使用状态感知的
`tactile-maintain`，不要手工判断后分别调用压缩与移动脚本。它默认只读，并以 `hot`、
`compressed-hot`、`detached` 三种目标状态自动规划剩余步骤。完整命令、回退开关和中断恢复语义见
[Dataset Postprocess：混合触觉数据的一体化入口](../../utils/dataset_postprocess/README.md#混合触觉数据的一体化入口推荐)。

## 2. 验证 v2.3

```bash
python -m utils.dataset_postprocess validate-v23 "${DATASET_ROOT:?set DATASET_ROOT}"
```

## 3. 测试与修改检查

通用单元测试、`git diff --check` 和 `py_compile` 只在
[setup 入口](../setup/README.md)及 [`README_ENV.md`](../../README_ENV.md) 维护；数据工具的
专项测试请从 [后处理与测试](./POSTPROCESS_AND_TESTING.md) 选择最小相关集合。
